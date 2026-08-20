#!/usr/bin/env python3
"""Probe deterministic two-endpoint recall for AMB temporal questions.

The probe never generates an answer. It appends a small, provenance-preserving
set of source hits for each endpoint and measures whether decisive gold values
become available without removing anything from the baseline context.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != HERE]

from memory_bench.memory.yantrikdb import YantrikDBMemoryProvider


VALUE_RE = re.compile(
    r"\$?\b\d[\d,]*(?:\.\d+)?%?"
    r"|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+(\d{1,2})(?:,\s*\d{4})?",
    re.IGNORECASE,
)


def values(text: str) -> set[str]:
    return {
        re.sub(r"\s+", " ", match.group(0).replace(",", "")).casefold()
        for match in VALUE_RE.finditer(text or "")
    }


def dates(text: str) -> list[str]:
    seen = set()
    result = []
    for month, day in DATE_RE.findall(text or ""):
        value = f"{month[:3].casefold()}-{int(day)}"
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def split_endpoints(query: str) -> tuple[str, str] | None:
    text = query.strip().rstrip("?")
    between = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+)$", text, re.I)
    if between:
        return between.group(1).strip(), between.group(2).strip()

    after = re.search(r"\bafter\s+(.+?)\s+did\s+(?:I\s+)?(.+)$", text, re.I)
    if after:
        return after.group(1).strip(), after.group(2).strip()

    when = re.search(r"\bwhen\s+(.+?)\s+and\s+when\s+(.+)$", text, re.I)
    if when:
        return when.group(1).strip(), when.group(2).strip()
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("store", type=Path)
    parser.add_argument("--endpoint-k", type=int, default=3)
    parser.add_argument("--snippets", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = json.loads(args.results.read_text(encoding="utf-8-sig"))["results"]
    rows = [
        row for row in rows
        if (row.get("meta") or {}).get("question_category") == "temporal_reasoning"
    ]
    unit_ids = {
        str((row.get("meta") or {})["conversation_id"])
        for row in rows
    }
    provider = YantrikDBMemoryProvider()
    provider.prepare(args.store, unit_ids, reset=False)
    output = []
    split_count = gained_rows = lane_pairs = 0
    before_hits = after_hits = total_values = 0
    try:
        for row in rows:
            pair = split_endpoints(row["query"])
            baseline = row.get("context") or ""
            gold = " ".join(row.get("gold_answers") or [])
            gold_values = values(gold)
            extra_hits = []
            lanes: list[list[dict]] = []
            if pair:
                split_count += 1
                user_id = str((row.get("meta") or {})["conversation_id"])
                seen = set()
                for endpoint in pair:
                    if args.snippets:
                        lane = provider._db_for(user_id).recall(
                            query=endpoint,
                            top_k=args.endpoint_k,
                            namespace=None,
                            skip_reinforce=True,
                            snippets=True,
                        )
                    else:
                        lane = provider._recall(
                            endpoint, args.endpoint_k, user_id
                        )
                    lanes.append(lane)
                    for hit in lane:
                        rid = str(hit.get("rid") or "")
                        if rid and rid not in seen:
                            seen.add(rid)
                            extra_hits.append(hit)
            extra = "\n\n".join(hit.get("text", "") for hit in extra_hits)
            combined = baseline + ("\n\n" + extra if extra else "")
            before = {value for value in gold_values if value in baseline.casefold()}
            after = {value for value in gold_values if value in combined.casefold()}
            gained = sorted(after - before)
            gold_dates = dates(gold)
            lane_dates = [
                dates("\n".join(hit.get("text", "") for hit in lane))
                for lane in lanes
            ]
            lane_pair = bool(
                len(gold_dates) >= 2
                and len(lane_dates) == 2
                and gold_dates[0] in lane_dates[0]
                and gold_dates[1] in lane_dates[1]
            )
            lane_pairs += int(lane_pair)
            gained_rows += int(bool(gained))
            before_hits += len(before)
            after_hits += len(after)
            total_values += len(gold_values)
            result = {
                "query_id": row["query_id"],
                "score": row.get("score"),
                "endpoints": list(pair) if pair else [],
                "extra_rids": [hit.get("rid") for hit in extra_hits],
                "gold_values": sorted(gold_values),
                "before_values": sorted(before),
                "after_values": sorted(after),
                "gained_values": gained,
                "gold_dates": gold_dates,
                "lane_dates": lane_dates,
                "lane_hits": [
                    [
                        {
                            "rid": hit.get("rid"),
                            "text": hit.get("text", ""),
                            "best_span": hit.get("best_span"),
                        }
                        for hit in lane
                    ]
                    for lane in lanes
                ],
                "correct_date_pair_in_lanes": lane_pair,
            }
            output.append(result)
            print(
                f"{row['query_id']:<25} score={float(row.get('score') or 0):.1f} "
                f"split={'yes' if pair else 'no ':<3} extras={len(extra_hits):<2} "
                f"values={len(before)}/{len(gold_values)}->{len(after)}/{len(gold_values)} "
                f"lanes={'pair' if lane_pair else 'miss'} gained={gained}"
            )
    finally:
        provider.cleanup()

    print(f"\nsplit {split_count}/{len(rows)}; rows gaining a value {gained_rows}")
    print(f"correct ordered endpoint-date pair in lanes {lane_pairs}/{split_count}")
    print(f"gold-value coverage {before_hits}/{total_values}->{after_hits}/{total_values}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"results": output}, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
