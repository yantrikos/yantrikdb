#!/usr/bin/env python3
"""Bound the AMB opportunity reachable by explicit event-time filtering."""

from __future__ import annotations

import argparse
import calendar
import json
import re
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


_MONTHS = {
    name.casefold(): number
    for number in range(1, 13)
    for name in (calendar.month_name[number], calendar.month_abbr[number])
}
_DATE_RE = re.compile(
    r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")"
    r"\s+(\d{1,2})(?:\s*[-\u2013]\s*(\d{1,2}))?"
    r"(?:,?\s+(20\d{2}))?\b",
    re.IGNORECASE,
)
_BEFORE_RE = re.compile(r"\b(?:by|before|until|through)\b", re.IGNORECASE)
_AFTER_RE = re.compile(r"\b(?:after|since|starting)\b", re.IGNORECASE)
_AMBIGUOUS_RE = re.compile(r"\bbeyond\b", re.IGNORECASE)


@dataclass(frozen=True)
class DateAnchor:
    text: str
    start: float
    end: float
    span_start: int
    span_end: int


def _epoch(year: int, month: int, day: int, *, end: bool = False) -> float:
    value = datetime(
        year,
        month,
        day,
        23 if end else 0,
        59 if end else 0,
        59 if end else 0,
        tzinfo=timezone.utc,
    )
    return value.timestamp()


def extract_date_anchors(text: str, default_year: int = 2024) -> list[DateAnchor]:
    anchors = []
    for match in _DATE_RE.finditer(text or ""):
        month_name, first_day, last_day, year = match.groups()
        month = _MONTHS[month_name.casefold()]
        parsed_year = int(year) if year else default_year
        end_day = int(last_day) if last_day else int(first_day)
        anchors.append(
            DateAnchor(
                text=match.group(0),
                start=_epoch(parsed_year, month, int(first_day)),
                end=_epoch(parsed_year, month, end_day, end=True),
                span_start=match.start(),
                span_end=match.end(),
            )
        )
    return anchors


def classify_query_time_filter(query: str) -> dict | None:
    """Classify filter reachability from query text only, never gold or context."""
    anchors = extract_date_anchors(query)
    if not anchors:
        return None
    if len(anchors) > 1 or any(anchor.start != _day_start(anchor.end) for anchor in anchors):
        return {
            "semantics": "closed_window",
            "event_after": min(anchor.start for anchor in anchors),
            "event_before": max(anchor.end for anchor in anchors),
            "dates": [asdict(anchor) for anchor in anchors],
            "precision_scope": "closed",
        }

    anchor = anchors[0]
    before_text = query[max(0, anchor.span_start - 80) : anchor.span_start]
    if _AMBIGUOUS_RE.search(before_text):
        semantics = "ambiguous_reference"
        event_after = event_before = None
        precision_scope = "ambiguous"
    elif _BEFORE_RE.search(before_text):
        semantics = "before"
        event_after, event_before = None, anchor.end
        precision_scope = "one_sided"
    elif _AFTER_RE.search(before_text):
        semantics = "after"
        event_after, event_before = anchor.start, None
        precision_scope = "one_sided"
    else:
        semantics = "exact_day"
        event_after, event_before = anchor.start, anchor.end
        precision_scope = "closed"
    return {
        "semantics": semantics,
        "event_after": event_after,
        "event_before": event_before,
        "dates": [asdict(anchor)],
        "precision_scope": precision_scope,
    }


def _day_start(timestamp: float) -> float:
    value = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return datetime(value.year, value.month, value.day, tzinfo=timezone.utc).timestamp()


def _cohort_summary(rows: list[dict], denominator: int) -> dict:
    scores = [float(row.get("score") or 0.0) for row in rows]
    headroom = sum(1.0 - score for score in scores)
    return {
        "n": len(rows),
        "mean_baseline_score": statistics.fmean(scores) if scores else 0.0,
        "imperfect_rows": sum(score < 1.0 for score in scores),
        "perfect_arm_score_headroom": headroom,
        "perfect_arm_full_benchmark_delta_ceiling": headroom / denominator,
    }


def analyze(rows: list[dict]) -> dict:
    audited = []
    for row in rows:
        classification = classify_query_time_filter(str(row.get("query") or ""))
        if classification is None:
            continue
        audited.append(
            {
                "query_id": str(row.get("query_id") or ""),
                "query": str(row.get("query") or ""),
                "category": str(
                    (row.get("meta") or {}).get("question_category") or "unknown"
                ),
                "score": float(row.get("score") or 0.0),
                **classification,
            }
        )

    unambiguous = [row for row in audited if row["precision_scope"] != "ambiguous"]
    closed = [row for row in audited if row["precision_scope"] == "closed"]
    by_category: dict[str, list[dict]] = {}
    for row in audited:
        by_category.setdefault(row["category"], []).append(row)
    by_semantics: dict[str, list[dict]] = {}
    for row in audited:
        by_semantics.setdefault(row["semantics"], []).append(row)

    return {
        "protocol": "amb-explicit-event-time-opportunity-audit-v1",
        "interpretation": {
            "cohort_selection_uses_query_text_only": True,
            "scores_used_only_for_posthoc_upper_bound": True,
            "perfect_arm_ceiling_is_not_expected_lift": True,
            "automatic_endpoint_resolution_is_out_of_scope": True,
        },
        "all_rows": len(rows),
        "any_explicit_date": _cohort_summary(audited, len(rows)),
        "unambiguous_filter": _cohort_summary(unambiguous, len(rows)),
        "closed_window_only": _cohort_summary(closed, len(rows)),
        "by_category": {
            category: _cohort_summary(category_rows, len(rows))
            for category, category_rows in sorted(by_category.items())
        },
        "by_semantics": {
            semantics: _cohort_summary(semantic_rows, len(rows))
            for semantics, semantic_rows in sorted(by_semantics.items())
        },
        "rows": audited,
    }


def _load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, list) else payload.get("results") or []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = analyze(_load_rows(args.results))
    rendered = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
