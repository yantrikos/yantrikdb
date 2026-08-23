import json

import pytest

from benchmarks.amb.paired_frozen_context_eval import (
    _answer_arm_orders,
    _answer_repeat_comparison,
    _load_resume_checkpoint,
    _median_scored_result,
    _ordered_query_ids_sha256,
    _paired_bootstrap_interval,
    _resolve_bootstrap_seed,
    _run_fingerprint,
    _sha256_file,
    validate_manifest,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_manifest_preflight_pins_payload_and_call_budget(tmp_path):
    rows_a = [
        {"query_id": "q1", "context": "alpha"},
        {"query_id": "q2", "context": "beta"},
    ]
    rows_b = [
        {"query_id": "q1", "context": "one"},
        {"query_id": "q2", "context": "two"},
    ]
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    _write_json(path_a, rows_a)
    _write_json(path_b, rows_b)
    query_ids_sha256 = _ordered_query_ids_sha256(["q1", "q2"])
    manifest = {
        "model": "test-model",
        "query_ids_encoding": "utf8-json-compact-ordered-v1",
        "synthetic_benchmark_data_only": True,
        "real_companion_memories_included": False,
        "arms": [
            {
                "file": path.name,
                "sha256": _sha256_file(path),
                "rows": 2,
                "context_tokens": sum(len(row["context"]) for row in rows),
                "query_ids_sha256": query_ids_sha256,
            }
            for path, rows in ((path_a, rows_a), (path_b, rows_b))
        ],
        "total_context_tokens": 15,
        "answer_calls": 4,
        "judge_calls": 4,
        "judge_repeats": 1,
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)

    report = validate_manifest(
        manifest_path,
        (path_a, path_b),
        "test-model",
        1,
        len,
    )
    assert report["query_ids"] == ["q1", "q2"]
    assert report["answer_calls"] == 4
    assert report["judge_calls"] == 4

    rows_b[1]["context"] = "changed"
    _write_json(path_b, rows_b)
    with pytest.raises(ValueError, match="sha256"):
        validate_manifest(
            manifest_path,
            (path_a, path_b),
            "test-model",
            1,
            len,
        )


def test_manifest_preflight_pins_repeated_answer_budget(tmp_path):
    rows = [{"query_id": "q1", "context": "alpha"}]
    paths = (tmp_path / "a.json", tmp_path / "b.json")
    for path in paths:
        _write_json(path, rows)
    query_ids_sha256 = _ordered_query_ids_sha256(["q1"])
    manifest = {
        "model": "test-model",
        "answer_repeats": 3,
        "judge_repeats": 3,
        "answer_calls": 6,
        "judge_calls": 18,
        "total_context_tokens": 10,
        "query_ids_encoding": "utf8-json-compact-ordered-v1",
        "synthetic_benchmark_data_only": True,
        "real_companion_memories_included": False,
        "arms": [
            {
                "file": path.name,
                "sha256": _sha256_file(path),
                "rows": 1,
                "context_tokens": 5,
                "query_ids_sha256": query_ids_sha256,
            }
            for path in paths
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)

    report = validate_manifest(
        manifest_path,
        paths,
        "test-model",
        3,
        len,
        3,
    )

    assert report["answer_repeats"] == 3
    assert report["answer_calls"] == 6
    assert report["judge_calls"] == 18


def test_repeated_answers_interleave_arms_and_select_median_score():
    orders = _answer_arm_orders("q1", seed=7, repeats=3)
    assert orders[1] == orders[0][::-1]
    assert orders[2] == orders[0]

    class Result:
        def __init__(self, score):
            self.score = score

    selected = _median_scored_result([Result(0.8), Result(0.2), Result(0.5)])
    assert selected.score == 0.5

    comparison = _answer_repeat_comparison(
        {
            "a": [Result(0.2), Result(0.5), Result(0.8)],
            "b": [Result(0.4), Result(0.5), Result(0.6)],
        }
    )
    assert comparison == {
        "scores_a": [0.2, 0.5, 0.8],
        "scores_b": [0.4, 0.5, 0.6],
        "mean_a": 0.5,
        "mean_b": 0.5,
        "median_a": 0.5,
        "median_b": 0.5,
        "range_a": [0.2, 0.8],
        "range_b": [0.4, 0.6],
        "deltas_b_minus_a": [0.2, 0.0, -0.20000000000000007],
        "mean_delta_b_minus_a": -1.850371707708594e-17,
        "wins_b": 1,
        "ties": 1,
        "wins_a": 1,
    }


def test_resume_checkpoint_is_bound_to_run_fingerprint(tmp_path):
    checkpoint = tmp_path / "run.partial"
    fingerprint = _run_fingerprint({"model": "a", "seed": 7, "bootstrap_seed": 11})
    _write_json(
        checkpoint,
        {
            "run_fingerprint": fingerprint,
            "pairs": [{"query_id": "q1", "score_a": 0.2, "score_b": 0.4}],
        },
    )

    completed = _load_resume_checkpoint(checkpoint, fingerprint)
    assert set(completed) == {"q1"}
    with pytest.raises(ValueError, match="does not match"):
        _load_resume_checkpoint(
            checkpoint,
            _run_fingerprint({"model": "b", "seed": 7, "bootstrap_seed": 11}),
        )


def test_bootstrap_seed_is_independent_and_resume_bound():
    deltas = [-1.0, 0.0, 0.0, 0.5, 1.0]
    interval_a = _paired_bootstrap_interval(deltas, seed=7, samples=101)
    interval_b = _paired_bootstrap_interval(deltas, seed=8, samples=101)

    assert interval_a != interval_b
    assert _run_fingerprint({"seed": 5, "bootstrap_seed": 7}) != _run_fingerprint(
        {"seed": 5, "bootstrap_seed": 8}
    )
    assert _resolve_bootstrap_seed(5, None) == 5
    assert _resolve_bootstrap_seed(5, 8) == 8


def test_resume_checkpoint_rejects_duplicate_pairs(tmp_path):
    checkpoint = tmp_path / "run.partial"
    fingerprint = _run_fingerprint({"model": "a"})
    _write_json(
        checkpoint,
        {
            "run_fingerprint": fingerprint,
            "pairs": [{"query_id": "q1"}, {"query_id": "q1"}],
        },
    )

    with pytest.raises(ValueError, match="duplicate"):
        _load_resume_checkpoint(checkpoint, fingerprint)
