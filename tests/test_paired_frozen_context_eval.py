import json

import pytest

from benchmarks.amb.paired_frozen_context_eval import (
    _load_resume_checkpoint,
    _ordered_query_ids_sha256,
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


def test_resume_checkpoint_is_bound_to_run_fingerprint(tmp_path):
    checkpoint = tmp_path / "run.partial"
    fingerprint = _run_fingerprint({"model": "a", "seed": 7})
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
            _run_fingerprint({"model": "b", "seed": 7}),
        )


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
