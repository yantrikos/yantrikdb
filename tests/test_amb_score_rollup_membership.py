import hashlib

import pytest

from benchmarks.amb.calibrate_rollup_membership import candidate_payload_sha256
from benchmarks.amb.score_rollup_membership import (
    _prompt,
    extract_json,
    normalize_scores,
    validate_cached_rows,
)


def test_extract_json_accepts_fenced_or_embedded_objects():
    assert extract_json('```json\n{"scores": []}\n```') == {"scores": []}
    assert extract_json('Result: {"scores": []} done') == {"scores": []}


def test_normalize_scores_restores_candidate_order_and_clamps_values():
    raw = {
        "scores": [
            {"id": "b", "relevance": 120, "atomicity": 80},
            {"id": "a", "relevance": 70, "atomicity": -5},
        ]
    }

    assert normalize_scores(raw, ["a", "b"]) == [
        {"id": "a", "relevance": 70.0, "atomicity": 0.0},
        {"id": "b", "relevance": 100.0, "atomicity": 80.0},
    ]


def test_normalize_scores_rejects_incomplete_model_output():
    with pytest.raises(ValueError, match="omitted or malformed"):
        normalize_scores(
            {"scores": [{"id": "a", "relevance": 70, "atomicity": 80}]},
            ["a", "b"],
        )


def test_validate_cached_rows_rejects_stale_model_or_artifact():
    row = {
        "query": "Which milestones?",
        "requested_item_count": 1,
        "candidate_items": [{"id": "a", "item": "First milestone"}],
    }
    cached = {
        "query": row["query"],
        "model": "old-model",
        "model_metadata": {"digest": "old-digest"},
        "scorer_protocol_version": 2,
        "candidate_payload_sha256": "old-payload",
        "num_ctx": 8192,
        "temperature": 0.0,
        "seed": 0,
        "think": False,
        "candidate_artifact_sha256": "old-hash",
        "prompt_sha256": "old-prompt",
        "scores": [{"id": "a", "relevance": 80, "atomicity": 80}],
    }

    with pytest.raises(ValueError, match="cached scorer metadata"):
        validate_cached_rows(
            [cached],
            {row["query"]: row},
            model="new-model",
            num_ctx=32768,
            candidate_sha256="new-hash",
            model_metadata={"digest": "new-digest"},
        )

    cached.update(
        {
            "model": "new-model",
            "model_metadata": {"digest": "new-digest"},
            "candidate_payload_sha256": candidate_payload_sha256(row),
            "num_ctx": 32768,
            "candidate_artifact_sha256": "new-hash",
            "prompt_sha256": hashlib.sha256(_prompt(row).encode()).hexdigest(),
        }
    )
    assert validate_cached_rows(
        [cached],
        {row["query"]: row},
        model="new-model",
        num_ctx=32768,
        candidate_sha256="new-hash",
        model_metadata={"digest": "new-digest"},
    ) == {row["query"]: cached}
