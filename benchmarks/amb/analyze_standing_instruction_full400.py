#!/usr/bin/env python3
"""Apply the preregistered standing-instruction full-400 promotion gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

try:
    from .paired_frozen_context_eval import _paired_bootstrap_interval, load_rows
except ImportError:  # pragma: no cover - direct script execution
    from paired_frozen_context_eval import _paired_bootstrap_interval, load_rows


INSTRUCTION_CATEGORY = "instruction_following"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summarize(pairs: list[dict], seed: int) -> dict:
    deltas = [float(pair["score_b"]) - float(pair["score_a"]) for pair in pairs]
    scores_a = [float(pair["score_a"]) for pair in pairs]
    scores_b = [float(pair["score_b"]) for pair in pairs]
    lower, upper = _paired_bootstrap_interval(deltas, seed)
    return {
        "n": len(pairs),
        "mean_a": statistics.fmean(scores_a),
        "mean_b": statistics.fmean(scores_b),
        "mean_delta_b_minus_a": statistics.fmean(deltas),
        "paired_bootstrap_95_ci": [lower, upper],
        "wins_b": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "wins_a": sum(delta < 0 for delta in deltas),
    }


def analyze(result: dict, source_rows: list[dict], seed: int) -> dict:
    category_by_id = {
        str(row["query_id"]): str(
            (row.get("meta") or {}).get("question_category") or "unknown"
        )
        for row in source_rows
    }
    pairs = result.get("pairs") or []
    if len(pairs) != 400:
        raise ValueError(f"expected 400 completed pairs, found {len(pairs)}")
    missing = [
        pair["query_id"] for pair in pairs if pair["query_id"] not in category_by_id
    ]
    if missing:
        raise ValueError(f"missing source categories for {len(missing)} query IDs")

    by_category: dict[str, list[dict]] = {}
    for pair in pairs:
        category = category_by_id[pair["query_id"]]
        by_category.setdefault(category, []).append(pair)
    category_summaries = {
        category: _summarize(category_pairs, seed)
        for category, category_pairs in sorted(by_category.items())
    }
    instruction = category_summaries[INSTRUCTION_CATEGORY]
    other_pairs = [
        pair
        for pair in pairs
        if category_by_id[pair["query_id"]] != INSTRUCTION_CATEGORY
    ]
    overall = _summarize(pairs, seed)
    other_nine = _summarize(other_pairs, seed)

    gate_1 = (
        instruction["mean_delta_b_minus_a"] >= 0.05
        and instruction["wins_b"] > instruction["wins_a"]
    )
    gate_2 = (
        overall["mean_delta_b_minus_a"] >= 0.0
        and overall["paired_bootstrap_95_ci"][0] >= -0.01
    )
    gate_3 = other_nine["mean_delta_b_minus_a"] >= -0.01
    gate_4 = all(
        summary["mean_delta_b_minus_a"] >= -0.025
        for category, summary in category_summaries.items()
        if category != INSTRUCTION_CATEGORY
    )
    gates = {
        "instruction_lift_and_wins": gate_1,
        "overall_nonnegative_ci_floor": gate_2,
        "other_nine_pooled_floor": gate_3,
        "no_other_category_below_floor": gate_4,
    }
    return {
        "protocol": "standing-instruction-full400-preregistered-analysis-v1",
        "seed": seed,
        "overall": overall,
        "instruction_following": instruction,
        "other_nine_pooled": other_nine,
        "categories": category_summaries,
        "gates": gates,
        "promotion_passed": all(gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    report = analyze(result, load_rows(args.source), args.seed)
    report["result_sha256"] = _sha256_file(args.result)
    report["source_sha256"] = _sha256_file(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
