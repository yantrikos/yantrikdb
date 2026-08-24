#!/usr/bin/env python3
"""Apply the preregistered standing-facet applicability v4 promotion gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

try:
    from .paired_frozen_context_eval import _paired_bootstrap_interval, load_rows
except ImportError:  # pragma: no cover - direct script execution
    from paired_frozen_context_eval import _paired_bootstrap_interval, load_rows


EXPECTED_PAIRS = 400
EXPECTED_ROWS_PER_CATEGORY = 40
INSTRUCTION_CATEGORY = "instruction_following"
EVENT_ORDERING_CATEGORY = "event_ordering"
SUMMARIZATION_CATEGORY = "summarization"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summarize(pairs: list[dict], bootstrap_seed: int) -> dict:
    deltas = [float(pair["score_b"]) - float(pair["score_a"]) for pair in pairs]
    scores_a = [float(pair["score_a"]) for pair in pairs]
    scores_b = [float(pair["score_b"]) for pair in pairs]
    lower, upper = _paired_bootstrap_interval(deltas, bootstrap_seed)
    return {
        "n": len(pairs),
        "mean_a": statistics.fmean(scores_a),
        "mean_b": statistics.fmean(scores_b),
        "mean_delta_b_minus_a": statistics.fmean(deltas),
        "paired_bootstrap_95_ci": [lower, upper],
        "paired_bootstrap_seed": bootstrap_seed,
        "wins_b": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "wins_a": sum(delta < 0 for delta in deltas),
    }


def analyze(result: dict, source_rows: list[dict], bootstrap_seed: int) -> dict:
    pairs = list(result.get("pairs") or [])
    if len(pairs) != EXPECTED_PAIRS:
        raise ValueError(
            f"expected {EXPECTED_PAIRS} completed pairs, found {len(pairs)}"
        )
    query_ids = [str(pair.get("query_id") or "") for pair in pairs]
    if any(not query_id for query_id in query_ids):
        raise ValueError("every pair must have a query_id")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("result contains duplicate query_ids")

    result_bootstrap_seed = result.get("bootstrap_seed")
    summary_bootstrap_seed = (result.get("summary") or {}).get("paired_bootstrap_seed")
    run_bootstrap_seed = (result.get("run_config") or {}).get("bootstrap_seed")
    if {
        result_bootstrap_seed,
        summary_bootstrap_seed,
        run_bootstrap_seed,
    } != {bootstrap_seed}:
        raise ValueError("result and analysis bootstrap seeds do not match")

    category_by_id = {
        str(row["query_id"]): str(
            (row.get("meta") or {}).get("question_category") or "unknown"
        )
        for row in source_rows
    }
    missing = [query_id for query_id in query_ids if query_id not in category_by_id]
    if missing:
        raise ValueError(f"missing source categories for {len(missing)} query IDs")

    by_category: dict[str, list[dict]] = {}
    for pair in pairs:
        category = category_by_id[str(pair["query_id"])]
        by_category.setdefault(category, []).append(pair)
    category_counts = Counter(category_by_id[str(pair["query_id"])] for pair in pairs)
    if len(category_counts) != 10 or any(
        count != EXPECTED_ROWS_PER_CATEGORY for count in category_counts.values()
    ):
        raise ValueError(f"unexpected category census: {dict(category_counts)}")

    categories = {
        category: _summarize(category_pairs, bootstrap_seed)
        for category, category_pairs in sorted(by_category.items())
    }
    instruction = categories[INSTRUCTION_CATEGORY]
    event_ordering = categories[EVENT_ORDERING_CATEGORY]
    summarization = categories[SUMMARIZATION_CATEGORY]
    other_pairs = [
        pair
        for pair in pairs
        if category_by_id[str(pair["query_id"])] != INSTRUCTION_CATEGORY
    ]
    overall = _summarize(pairs, bootstrap_seed)
    other_nine = _summarize(other_pairs, bootstrap_seed)

    gates = {
        "instruction_delta_at_least_0_05_and_wins_exceed_losses": (
            instruction["mean_delta_b_minus_a"] >= 0.05
            and instruction["wins_b"] > instruction["wins_a"]
        ),
        "overall_nonnegative_and_ci_floor_at_least_minus_0_01": (
            overall["mean_delta_b_minus_a"] >= 0.0
            and overall["paired_bootstrap_95_ci"][0] >= -0.01
        ),
        "other_nine_pooled_delta_at_least_minus_0_005": (
            other_nine["mean_delta_b_minus_a"] >= -0.005
        ),
        "summarization_delta_at_least_minus_0_01": (
            summarization["mean_delta_b_minus_a"] >= -0.01
        ),
        "no_non_instruction_category_below_minus_0_025": all(
            summary["mean_delta_b_minus_a"] >= -0.025
            for category, summary in categories.items()
            if category != INSTRUCTION_CATEGORY
        ),
        "event_ordering_delta_nonnegative": (
            event_ordering["mean_delta_b_minus_a"] >= 0.0
        ),
    }
    return {
        "protocol": "facet-applicability-v4-preregistered-analysis-v1",
        "bootstrap_seed": bootstrap_seed,
        "overall": overall,
        "instruction_following": instruction,
        "other_nine_pooled": other_nine,
        "summarization": summarization,
        "event_ordering": event_ordering,
        "categories": categories,
        "gates": gates,
        "promotion_passed": all(gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    report = analyze(result, load_rows(args.source), args.bootstrap_seed)
    report["result_sha256"] = _sha256_file(args.result)
    report["source_sha256"] = _sha256_file(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
