#!/usr/bin/env python3
"""Combine independent paired runs by per-query arithmetic means."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

try:
    from .paired_frozen_context_eval import (
        _paired_bootstrap_interval,
        _run_fingerprint,
    )
except ImportError:  # pragma: no cover - direct script execution
    from paired_frozen_context_eval import _paired_bootstrap_interval, _run_fingerprint


EXPECTED_PAIRS = 400
EXPECTED_REPLICATES = 3
COMMON_RUN_KEYS = (
    "manifest_sha256",
    "contexts_a_sha256",
    "contexts_b_sha256",
    "query_ids_sha256",
    "label_a",
    "label_b",
    "model",
    "split",
    "answer_repeats",
    "judge_repeats",
    "workers",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise ValueError(f"{message}: expected={expected!r} actual={actual!r}")


def _validate_result(
    result: dict,
    path: Path,
    expected_seed: int,
    expected_model_seed: int,
    expected_bootstrap_seed: int,
) -> tuple[dict, list[dict]]:
    config = result.get("run_config") or {}
    pairs = result.get("pairs") or []
    prefix = path.name

    _require_equal(result.get("run_fingerprint"), _run_fingerprint(config), prefix)
    _require_equal(result.get("seed"), expected_seed, f"{prefix}: output seed")
    _require_equal(config.get("seed"), expected_seed, f"{prefix}: run seed")
    _require_equal(
        result.get("model_seed"),
        expected_model_seed,
        f"{prefix}: output model seed",
    )
    _require_equal(
        config.get("model_seed"),
        expected_model_seed,
        f"{prefix}: run model seed",
    )
    _require_equal(
        result.get("bootstrap_seed"),
        expected_bootstrap_seed,
        f"{prefix}: output bootstrap seed",
    )
    _require_equal(
        config.get("bootstrap_seed"),
        expected_bootstrap_seed,
        f"{prefix}: run bootstrap seed",
    )
    _require_equal(
        (result.get("summary") or {}).get("paired_bootstrap_seed"),
        expected_bootstrap_seed,
        f"{prefix}: summary bootstrap seed",
    )
    _require_equal(result.get("label_a"), config.get("label_a"), f"{prefix}: label_a")
    _require_equal(result.get("label_b"), config.get("label_b"), f"{prefix}: label_b")
    _require_equal(result.get("model"), config.get("model"), f"{prefix}: model")
    _require_equal(result.get("answer_repeats"), 1, f"{prefix}: answer repeats")
    _require_equal(config.get("answer_repeats"), 1, f"{prefix}: run answer repeats")
    _require_equal(result.get("judge_repeats"), 1, f"{prefix}: judge repeats")
    _require_equal(config.get("judge_repeats"), 1, f"{prefix}: run judge repeats")
    _require_equal(len(pairs), EXPECTED_PAIRS, f"{prefix}: completed pairs")
    _require_equal(
        (result.get("summary") or {}).get("n"),
        EXPECTED_PAIRS,
        f"{prefix}: summary pairs",
    )

    query_ids = [str(pair.get("query_id") or "") for pair in pairs]
    if any(not query_id for query_id in query_ids):
        raise ValueError(f"{prefix}: every pair must have a query_id")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError(f"{prefix}: duplicate query_ids")
    for pair in pairs:
        score_a = float(pair["score_a"])
        score_b = float(pair["score_b"])
        delta = float(pair["delta_b_minus_a"])
        if not all(math.isfinite(value) for value in (score_a, score_b, delta)):
            raise ValueError(f"{prefix}: non-finite score for {pair['query_id']}")
        if not math.isclose(delta, score_b - score_a, abs_tol=1e-12):
            raise ValueError(f"{prefix}: inconsistent delta for {pair['query_id']}")
    return config, pairs


def combine(
    paths: list[Path],
    expected_seeds: list[int],
    expected_model_seeds: list[int],
    bootstrap_seed: int,
) -> dict:
    if len(paths) != EXPECTED_REPLICATES:
        raise ValueError(f"expected exactly {EXPECTED_REPLICATES} replicate files")
    if len(expected_seeds) != EXPECTED_REPLICATES:
        raise ValueError(f"expected exactly {EXPECTED_REPLICATES} run seeds")
    if len(expected_model_seeds) != EXPECTED_REPLICATES:
        raise ValueError(f"expected exactly {EXPECTED_REPLICATES} model seeds")
    if len(set(expected_seeds)) != EXPECTED_REPLICATES:
        raise ValueError("run seeds must be distinct")
    if len(set(expected_model_seeds)) != EXPECTED_REPLICATES:
        raise ValueError("model seeds must be distinct")

    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    validated = [
        _validate_result(result, path, seed, model_seed, bootstrap_seed)
        for result, path, seed, model_seed in zip(
            loaded, paths, expected_seeds, expected_model_seeds
        )
    ]
    configs = [item[0] for item in validated]
    replicate_pairs = [item[1] for item in validated]
    baseline = {key: configs[0].get(key) for key in COMMON_RUN_KEYS}
    for index, config in enumerate(configs[1:], 2):
        actual = {key: config.get(key) for key in COMMON_RUN_KEYS}
        _require_equal(actual, baseline, f"replicate {index}: common run config")

    query_ids = [str(pair["query_id"]) for pair in replicate_pairs[0]]
    for index, pairs in enumerate(replicate_pairs[1:], 2):
        _require_equal(
            [str(pair["query_id"]) for pair in pairs],
            query_ids,
            f"replicate {index}: ordered query IDs",
        )

    combined_pairs = []
    for pair_index, query_id in enumerate(query_ids):
        scores_a = [float(pairs[pair_index]["score_a"]) for pairs in replicate_pairs]
        scores_b = [float(pairs[pair_index]["score_b"]) for pairs in replicate_pairs]
        score_a = statistics.fmean(scores_a)
        score_b = statistics.fmean(scores_b)
        combined_pairs.append(
            {
                "query_id": query_id,
                "score_a": score_a,
                "score_b": score_b,
                "delta_b_minus_a": score_b - score_a,
                "replicate_scores_a": scores_a,
                "replicate_scores_b": scores_b,
                "replicate_deltas_b_minus_a": [
                    value_b - value_a for value_a, value_b in zip(scores_a, scores_b)
                ],
            }
        )

    deltas = [pair["delta_b_minus_a"] for pair in combined_pairs]
    scores_a = [pair["score_a"] for pair in combined_pairs]
    scores_b = [pair["score_b"] for pair in combined_pairs]
    lower, upper = _paired_bootstrap_interval(deltas, bootstrap_seed)
    source_replicates = [
        {
            "path": str(path),
            "sha256": _sha256_file(path),
            "run_fingerprint": result["run_fingerprint"],
            "seed": seed,
            "model_seed": model_seed,
        }
        for path, result, seed, model_seed in zip(
            paths, loaded, expected_seeds, expected_model_seeds
        )
    ]
    run_config = {
        **baseline,
        "protocol": "paired-independent-mean-of-three-v1",
        "replicate_count": EXPECTED_REPLICATES,
        "replicate_seeds": expected_seeds,
        "model_seeds": expected_model_seeds,
        "bootstrap_seed": bootstrap_seed,
        "source_sha256": [source["sha256"] for source in source_replicates],
        "score_aggregation": "per-query-arm-arithmetic-mean-v1",
    }
    summary = {
        "n": len(combined_pairs),
        "replicate_count": EXPECTED_REPLICATES,
        "mean_a": statistics.fmean(scores_a),
        "mean_b": statistics.fmean(scores_b),
        "mean_delta_b_minus_a": statistics.fmean(deltas),
        "paired_bootstrap_95_ci": [lower, upper],
        "paired_bootstrap_seed": bootstrap_seed,
        "wins_b": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "wins_a": sum(delta < 0 for delta in deltas),
    }
    return {
        "run_config": run_config,
        "run_fingerprint": _run_fingerprint(run_config),
        "label_a": baseline["label_a"],
        "label_b": baseline["label_b"],
        "model": baseline["model"],
        "answer_repeats_per_replicate": 1,
        "judge_repeats_per_answer": 1,
        "replicate_seeds": expected_seeds,
        "model_seeds": expected_model_seeds,
        "bootstrap_seed": bootstrap_seed,
        "source_replicates": source_replicates,
        "summary": summary,
        "pairs": combined_pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate", type=Path, action="append", required=True)
    parser.add_argument("--expected-seed", type=int, action="append", required=True)
    parser.add_argument(
        "--expected-model-seed", type=int, action="append", required=True
    )
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = combine(
        args.replicate,
        args.expected_seed,
        args.expected_model_seed,
        args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
