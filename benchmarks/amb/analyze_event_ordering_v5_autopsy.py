#!/usr/bin/env python3
"""Classify the frozen v5 event-ordering score deficits without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import fmean

try:
    from .paired_frozen_context_eval import _run_fingerprint
except ImportError:  # pragma: no cover - direct script execution
    from paired_frozen_context_eval import _run_fingerprint


EXPECTED_PROTOCOL = "paired-independent-mean-of-three-v1"
KNOWN_LABEL_DEFECTS = {
    "9_event_ordering_0": (
        "Gold requires the stale Bryan event after the user corrected that history."
    ),
    "18_event_ordering_0": (
        "Patrick gold uses an unstated merge/split granularity over a complete route."
    ),
    "19_event_ordering_0": (
        "Douglas gold selects a hidden later partition over earlier valid plans."
    ),
}
QUERY_ROUTE_COHORTS = {
    "benchmark_defect_quarantine": {
        "9_event_ordering_0",
        "18_event_ordering_0",
        "19_event_ordering_0",
    },
    "exact_entity_thread": {"10_event_ordering_1"},
    "entity_plus_focus": {"13_event_ordering_1"},
    "bounded_focus_phrase": {
        "1_event_ordering_0",
        "2_event_ordering_0",
        "2_event_ordering_1",
        "4_event_ordering_0",
        "4_event_ordering_1",
        "5_event_ordering_0",
        "5_event_ordering_1",
        "9_event_ordering_1",
        "11_event_ordering_0",
        "11_event_ordering_1",
        "12_event_ordering_1",
        "13_event_ordering_0",
        "14_event_ordering_0",
        "14_event_ordering_1",
        "15_event_ordering_0",
        "15_event_ordering_1",
        "16_event_ordering_0",
        "17_event_ordering_0",
        "17_event_ordering_1",
        "19_event_ordering_1",
        "20_event_ordering_0",
        "20_event_ordering_1",
    },
    "broad_compound_topic_union": {
        "1_event_ordering_1",
        "3_event_ordering_0",
        "3_event_ordering_1",
        "6_event_ordering_0",
        "6_event_ordering_1",
        "7_event_ordering_0",
        "7_event_ordering_1",
        "8_event_ordering_0",
        "8_event_ordering_1",
        "10_event_ordering_0",
        "12_event_ordering_0",
        "16_event_ordering_1",
        "18_event_ordering_1",
    },
}
_TURN_HEADER_RE = re.compile(
    r"(?m)^\[(?:[A-Z][a-z]+-\d+-\d+ \| Turn (?P<dated>\d+)"
    r"|Turn (?P<plain>\d+))\](?: \(cont\.\))?\s+(?:User|Assistant):"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index_unique(rows: list[dict], label: str) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        query_id = str(row.get("query_id") or "")
        if not query_id:
            raise ValueError(f"{label} row is missing query_id")
        if query_id in indexed:
            raise ValueError(f"duplicate {label} query_id {query_id!r}")
        indexed[query_id] = row
    return indexed


def extract_context_turns(context: str) -> list[int]:
    """Return exact BEAM turn headers from an ordinary retrieval context."""
    return sorted(
        {
            int(match.group("dated") or match.group("plain"))
            for match in _TURN_HEADER_RE.finditer(context or "")
        }
    )


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = fmean(xs)
    mean_y = fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else None


def _summarize_classes(rows: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(row["failure_class"] for row in rows).items()))


def _query_route_map() -> dict[str, str]:
    routes = {}
    for route, query_ids in QUERY_ROUTE_COHORTS.items():
        for query_id in query_ids:
            if query_id in routes:
                raise ValueError(f"duplicate query-route assignment for {query_id}")
            routes[query_id] = route
    return routes


def _validate_run(result: dict, label: str) -> None:
    config = result.get("run_config") or {}
    if result.get("run_fingerprint") != _run_fingerprint(config):
        raise ValueError(f"{label} fingerprint does not match its run config")


def analyze(
    combined: dict,
    replicate: dict,
    membership: dict,
    *,
    replicate_sha256: str,
    expected_queries: int = 40,
    label_defects: dict[str, str] | None = None,
) -> dict:
    """Join frozen scores, control contexts, and authoritative source turns."""
    _validate_run(combined, "combined result")
    _validate_run(replicate, "replicate")
    config = combined.get("run_config") or {}
    if config.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError(f"combined result protocol is not {EXPECTED_PROTOCOL!r}")
    if combined.get("label_a") != replicate.get("label_a") or combined.get(
        "label_b"
    ) != replicate.get("label_b"):
        raise ValueError("replicate arm labels do not match the combined result")

    source_matches = [
        (index, source)
        for index, source in enumerate(combined.get("source_replicates") or [])
        if source.get("sha256") == replicate_sha256
    ]
    if len(source_matches) != 1:
        raise ValueError("replicate SHA-256 is not one unique combined source")
    replicate_index, source = source_matches[0]
    if source.get("run_fingerprint") != replicate.get("run_fingerprint"):
        raise ValueError(
            "replicate fingerprint does not match its combined source entry"
        )
    if source.get("seed") != replicate.get("seed") or source.get(
        "model_seed"
    ) != replicate.get("model_seed"):
        raise ValueError("replicate seeds do not match its combined source entry")

    combined_by_id = _index_unique(combined.get("pairs") or [], "combined")
    replicate_by_id = _index_unique(replicate.get("pairs") or [], "replicate")
    membership_by_id = _index_unique(membership.get("results") or [], "membership")
    query_ids = sorted(
        query_id for query_id in combined_by_id if "_event_ordering_" in query_id
    )
    if len(query_ids) != expected_queries:
        raise ValueError(
            f"expected {expected_queries} event-ordering queries, found {len(query_ids)}"
        )
    missing_replicate = sorted(set(query_ids) - set(replicate_by_id))
    missing_membership = sorted(set(query_ids) - set(membership_by_id))
    if missing_replicate or missing_membership:
        raise ValueError(
            "event-ordering inputs are incomplete: "
            f"replicate={missing_replicate}, membership={missing_membership}"
        )

    defects = KNOWN_LABEL_DEFECTS if label_defects is None else label_defects
    unknown_defects = sorted(set(defects) - set(query_ids))
    if unknown_defects:
        raise ValueError(f"label defects reference unknown queries: {unknown_defects}")
    query_routes = _query_route_map()
    missing_routes = sorted(set(query_ids) - set(query_routes))
    if missing_routes:
        raise ValueError(
            f"event-ordering queries lack route assignments: {missing_routes}"
        )
    if expected_queries == len(query_routes):
        extra_routes = sorted(set(query_routes) - set(query_ids))
        if extra_routes:
            raise ValueError(f"query routes reference absent queries: {extra_routes}")

    rows = []
    for query_id in query_ids:
        paired = combined_by_id[query_id]
        replicate_pair = replicate_by_id[query_id]
        source_scores_a = paired.get("replicate_scores_a") or []
        source_scores_b = paired.get("replicate_scores_b") or []
        if (
            len(source_scores_a) <= replicate_index
            or len(source_scores_b) <= replicate_index
        ):
            raise ValueError(f"combined replicate scores are incomplete for {query_id}")
        if not math.isclose(
            source_scores_a[replicate_index], replicate_pair["score_a"]
        ):
            raise ValueError(f"control score mismatch for {query_id}")
        if not math.isclose(
            source_scores_b[replicate_index], replicate_pair["score_b"]
        ):
            raise ValueError(f"treatment score mismatch for {query_id}")

        result_a = replicate_pair.get("result_a") or {}
        if (result_a.get("meta") or {}).get("question_category") != "event_ordering":
            raise ValueError(f"control result category mismatch for {query_id}")
        authoritative = membership_by_id[query_id]
        source_turns = authoritative.get("source_turns") or []
        if not source_turns or any(
            isinstance(turn, bool) or not isinstance(turn, int) for turn in source_turns
        ):
            raise ValueError(f"invalid authoritative source turns for {query_id}")
        source_turns = sorted(set(source_turns))
        context_turns = extract_context_turns(result_a.get("context") or "")
        present_turns = sorted(set(source_turns) & set(context_turns))
        missing_turns = sorted(set(source_turns) - set(context_turns))
        source_recall = len(present_turns) / len(source_turns)
        topic_handles = authoritative.get("topic_handles") or {}
        oracle_top3 = topic_handles.get("3") or topic_handles.get(3) or {}
        if "turns" not in oracle_top3:
            raise ValueError(
                f"membership row lacks top-three topic turns for {query_id}"
            )
        oracle_top3_turns = sorted(
            set(source_turns) & set(oracle_top3.get("turns") or [])
        )
        score_a = float(paired["score_a"])
        score_b = float(paired["score_b"])
        delta = float(paired["delta_b_minus_a"])
        if not math.isclose(delta, score_b - score_a):
            raise ValueError(f"paired delta mismatch for {query_id}")

        if math.isclose(score_b, 1.0):
            failure_class = "no_score_deficit"
        elif query_id in defects:
            failure_class = "label_defect"
        elif missing_turns:
            failure_class = "retrieval_miss"
        else:
            failure_class = "source_complete_answer_residual"

        route = (authoritative.get("query_matched_handles") or {}).get("route")
        rows.append(
            {
                "query_id": query_id,
                "query": result_a.get("query"),
                "control_score": score_a,
                "treatment_score": score_b,
                "delta": delta,
                "score_deficit": 1.0 - score_b,
                "source_turns": source_turns,
                "control_context_source_turns": present_turns,
                "missing_source_turns": missing_turns,
                "source_turn_recall": source_recall,
                "source_complete": not missing_turns,
                "oracle_top3_topic_source_turns": oracle_top3_turns,
                "oracle_top3_topic_source_recall": len(oracle_top3_turns)
                / len(source_turns),
                "failure_class": failure_class,
                "label_defect_reason": defects.get(query_id),
                "organizer_query_route": route or "unmatched",
                "recommended_query_route": query_routes[query_id],
            }
        )

    losses = [row for row in rows if row["delta"] < 0]
    zeros = [row for row in rows if math.isclose(row["treatment_score"], 0.0)]
    deficits = [row for row in rows if row["treatment_score"] < 1.0]
    class_score_summary = {}
    for failure_class in sorted({row["failure_class"] for row in deficits}):
        class_rows = [row for row in deficits if row["failure_class"] == failure_class]
        class_score_summary[failure_class] = {
            "queries": len(class_rows),
            "mean_treatment_score": fmean(row["treatment_score"] for row in class_rows),
            "mean_score_deficit": fmean(row["score_deficit"] for row in class_rows),
        }

    route_summary = {}
    for route in QUERY_ROUTE_COHORTS:
        route_rows = [row for row in rows if row["recommended_query_route"] == route]
        if not route_rows:
            continue
        source_references = sum(len(row["source_turns"]) for row in route_rows)
        context_hits = sum(
            len(row["control_context_source_turns"]) for row in route_rows
        )
        oracle_hits = sum(
            len(row["oracle_top3_topic_source_turns"]) for row in route_rows
        )
        route_summary[route] = {
            "queries": len(route_rows),
            "query_ids": [row["query_id"] for row in route_rows],
            "source_turn_references": source_references,
            "control_context_source_turns": context_hits,
            "control_context_micro_recall": context_hits / source_references,
            "oracle_top3_topic_source_turns": oracle_hits,
            "oracle_top3_topic_micro_recall": oracle_hits / source_references,
        }

    return {
        "protocol": "event-ordering-v5-control-coverage-autopsy-v1",
        "classification_note": (
            "BEAM scores rubric-item coverage, not ordering independently; "
            "source_complete_answer_residual therefore includes selection, "
            "granularity, presentation, and reader residuals."
        ),
        "control_context_only": True,
        "queries": len(rows),
        "mean_source_turn_recall": fmean(row["source_turn_recall"] for row in rows),
        "exact_source_coverage": sum(row["source_complete"] for row in rows),
        "zero_source_coverage": sum(
            math.isclose(row["source_turn_recall"], 0.0) for row in rows
        ),
        "source_recall_treatment_score_pearson": _pearson(
            [row["source_turn_recall"] for row in rows],
            [row["treatment_score"] for row in rows],
        ),
        "query_routes": route_summary,
        "score_deficits": {
            "queries": len(deficits),
            "by_class": _summarize_classes(deficits),
            "score_summary_by_class": class_score_summary,
        },
        "negative_deltas": {
            "queries": len(losses),
            "source_incomplete": sum(not row["source_complete"] for row in losses),
            "by_class": _summarize_classes(losses),
            "query_ids": [row["query_id"] for row in losses],
        },
        "treatment_zeros": {
            "queries": len(zeros),
            "by_class": _summarize_classes(zeros),
            "query_ids": [row["query_id"] for row in zeros],
        },
        "rows": rows,
    }


def _parse_label_defect(value: str) -> tuple[str, str]:
    query_id, separator, reason = value.partition("=")
    if not separator or not query_id.strip() or not reason.strip():
        raise argparse.ArgumentTypeError("expected QUERY_ID=REASON")
    return query_id.strip(), reason.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--replicate", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int, default=40)
    parser.add_argument(
        "--label-defect",
        action="append",
        default=[],
        type=_parse_label_defect,
        metavar="QUERY_ID=REASON",
        help="Add or replace a known benchmark-label defect.",
    )
    args = parser.parse_args()

    combined = json.loads(args.combined.read_text(encoding="utf-8"))
    replicate = json.loads(args.replicate.read_text(encoding="utf-8"))
    membership = json.loads(args.membership.read_text(encoding="utf-8"))
    defects = {**KNOWN_LABEL_DEFECTS, **dict(args.label_defect)}
    report = analyze(
        combined,
        replicate,
        membership,
        replicate_sha256=_sha256(args.replicate),
        expected_queries=args.expected_queries,
        label_defects=defects,
    )
    report["combined_sha256"] = _sha256(args.combined)
    report["replicate_sha256"] = _sha256(args.replicate)
    report["membership_sha256"] = _sha256(args.membership)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
