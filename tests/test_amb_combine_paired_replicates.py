import json

import pytest

from benchmarks.amb.combine_paired_replicates import combine
from benchmarks.amb.paired_frozen_context_eval import _run_fingerprint


SEEDS = [20260828, 20260829, 20260830]


def _result(seed, score_a, score_b, context_b="treatment-hash"):
    config = {
        "manifest_sha256": "manifest-hash",
        "contexts_a_sha256": "control-hash",
        "contexts_b_sha256": context_b,
        "query_ids_sha256": "query-hash",
        "label_a": "control",
        "label_b": "treatment",
        "model": "test-model",
        "split": "100k",
        "answer_repeats": 1,
        "judge_repeats": 1,
        "seed": seed,
        "model_seed": seed,
        "bootstrap_seed": 20260831,
        "workers": 2,
    }
    pairs = [
        {
            "query_id": f"q{index:03d}",
            "score_a": score_a,
            "score_b": score_b,
            "delta_b_minus_a": score_b - score_a,
        }
        for index in range(400)
    ]
    return {
        "run_config": config,
        "run_fingerprint": _run_fingerprint(config),
        "label_a": "control",
        "label_b": "treatment",
        "model": "test-model",
        "answer_repeats": 1,
        "judge_repeats": 1,
        "seed": seed,
        "model_seed": seed,
        "bootstrap_seed": 20260831,
        "summary": {"n": 400, "paired_bootstrap_seed": 20260831},
        "pairs": pairs,
    }


def _write_replicates(tmp_path, results):
    paths = []
    for seed, result in zip(SEEDS, results):
        path = tmp_path / f"replicate-{seed}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        paths.append(path)
    return paths


def test_combiner_uses_per_query_arm_means_and_binds_sources(tmp_path):
    paths = _write_replicates(
        tmp_path,
        [
            _result(SEEDS[0], 0.2, 0.6),
            _result(SEEDS[1], 0.4, 0.5),
            _result(SEEDS[2], 0.6, 0.4),
        ],
    )

    combined = combine(paths, SEEDS, SEEDS, bootstrap_seed=20260831)

    pair = combined["pairs"][0]
    assert pair["replicate_scores_a"] == [0.2, 0.4, 0.6]
    assert pair["replicate_scores_b"] == [0.6, 0.5, 0.4]
    assert pair["score_a"] == pytest.approx(0.4)
    assert pair["score_b"] == pytest.approx(0.5)
    assert pair["delta_b_minus_a"] == pytest.approx(0.1)
    assert combined["summary"]["n"] == 400
    assert combined["summary"]["paired_bootstrap_seed"] == 20260831
    assert len(combined["source_replicates"]) == 3
    assert all(source["sha256"] for source in combined["source_replicates"])
    assert combined["run_fingerprint"] == _run_fingerprint(combined["run_config"])


def test_combiner_rejects_common_config_drift(tmp_path):
    results = [
        _result(SEEDS[0], 0.2, 0.6),
        _result(SEEDS[1], 0.4, 0.5, context_b="changed"),
        _result(SEEDS[2], 0.6, 0.4),
    ]
    paths = _write_replicates(tmp_path, results)

    with pytest.raises(ValueError, match="common run config"):
        combine(paths, SEEDS, SEEDS, bootstrap_seed=20260831)


def test_combiner_rejects_reordered_or_incomplete_cohort(tmp_path):
    results = [
        _result(SEEDS[0], 0.2, 0.6),
        _result(SEEDS[1], 0.4, 0.5),
        _result(SEEDS[2], 0.6, 0.4),
    ]
    results[1]["pairs"].reverse()
    paths = _write_replicates(tmp_path, results)

    with pytest.raises(ValueError, match="ordered query IDs"):
        combine(paths, SEEDS, SEEDS, bootstrap_seed=20260831)


def test_combiner_rejects_seed_or_fingerprint_mismatch(tmp_path):
    results = [
        _result(SEEDS[0], 0.2, 0.6),
        _result(SEEDS[1], 0.4, 0.5),
        _result(SEEDS[2], 0.6, 0.4),
    ]
    results[2]["model_seed"] = 7
    paths = _write_replicates(tmp_path, results)

    with pytest.raises(ValueError, match="output model seed"):
        combine(paths, SEEDS, SEEDS, bootstrap_seed=20260831)


def test_combiner_rejects_replicate_bootstrap_seed_mismatch(tmp_path):
    results = [
        _result(SEEDS[0], 0.2, 0.6),
        _result(SEEDS[1], 0.4, 0.5),
        _result(SEEDS[2], 0.6, 0.4),
    ]
    results[1]["summary"]["paired_bootstrap_seed"] = 7
    paths = _write_replicates(tmp_path, results)

    with pytest.raises(ValueError, match="summary bootstrap seed"):
        combine(paths, SEEDS, SEEDS, bootstrap_seed=20260831)
