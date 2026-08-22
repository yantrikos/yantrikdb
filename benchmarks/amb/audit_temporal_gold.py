"""Audit explicit AMB temporal gold quantities against their stated dates."""

import argparse
import calendar
import json
import re
from dataclasses import asdict, dataclass
from datetime import date
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
_QUANTITY_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s+(days?|weeks?)\b", re.I)


@dataclass(frozen=True)
class DateMention:
    month: int
    day: int
    range_end_day: int | None
    year: int
    text: str

    def endpoint(self, *, use_range_end: bool = False) -> date:
        day = self.range_end_day if use_range_end and self.range_end_day else self.day
        return date(self.year, self.month, day)


def extract_date_mentions(text: str, default_year: int = 2024) -> list[DateMention]:
    mentions = []
    for match in _DATE_RE.finditer(text or ""):
        month_name, day, range_end, year = match.groups()
        mentions.append(
            DateMention(
                month=_MONTHS[month_name.casefold()],
                day=int(day),
                range_end_day=int(range_end) if range_end else None,
                year=int(year) if year else default_year,
                text=match.group(0),
            )
        )
    return mentions


def extract_quantity(text: str) -> tuple[float, str] | None:
    match = _QUANTITY_RE.search(text or "")
    if not match:
        return None
    unit = match.group(2).casefold()
    return float(match.group(1)), "week" if unit.startswith("week") else "day"


def stated_interval_days(text: str) -> int | None:
    mentions = extract_date_mentions(text)
    if len(mentions) < 2:
        return None
    first, second = mentions[:2]
    start = first.endpoint(use_range_end=first.range_end_day is not None)
    end = second.endpoint()
    return abs((end - start).days)


def audit_row(row: dict) -> dict | None:
    gold_text = " ".join(row.get("gold_answers") or [])
    quantity = extract_quantity(gold_text)
    interval_days = stated_interval_days(gold_text)
    if quantity is None or interval_days is None:
        return None
    claimed, unit = quantity
    claimed_days = claimed * (7 if unit == "week" else 1)
    answer_quantity = extract_quantity(str(row.get("answer") or ""))
    return {
        "query_id": row.get("query_id"),
        "query": row.get("query"),
        "score": row.get("score"),
        "gold": gold_text,
        "answer": row.get("answer"),
        "date_mentions": [
            asdict(mention) for mention in extract_date_mentions(gold_text)[:2]
        ],
        "gold_claimed_quantity": claimed,
        "gold_claimed_unit": unit,
        "gold_claimed_days": claimed_days,
        "calendar_interval_days": interval_days,
        "gold_arithmetic_matches": abs(claimed_days - interval_days) < 1e-9,
        "answer_quantity": answer_quantity[0] if answer_quantity else None,
        "answer_unit": answer_quantity[1] if answer_quantity else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8-sig"))
    rows = payload.get("results") if isinstance(payload, dict) else payload
    temporal_rows = [
        row
        for row in rows
        if (row.get("meta") or {}).get("question_category")
        == "temporal_reasoning"
    ]
    audited = [audit for row in temporal_rows if (audit := audit_row(row))]
    mismatches = [row for row in audited if not row["gold_arithmetic_matches"]]
    result = {
        "temporal_rows": len(temporal_rows),
        "auditable_rows": len(audited),
        "gold_arithmetic_mismatch_count": len(mismatches),
        "gold_arithmetic_mismatch_ids": [row["query_id"] for row in mismatches],
        "results": audited,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(
        f"temporal={len(temporal_rows)} auditable={len(audited)} "
        f"gold_mismatches={len(mismatches)}"
    )
    for row in mismatches:
        print(
            f"{row['query_id']}: gold={row['gold_claimed_days']:g}d "
            f"calendar={row['calendar_interval_days']}d "
            f"answer={row['answer']!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
