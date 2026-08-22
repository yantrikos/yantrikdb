import pytest

from benchmarks.amb.calibrate_rollup_membership import (
    BASELINE_CONFIG,
    RescueConfig,
    _accept_for_product_probe,
    _baseline_ids,
    _group_sort_key,
    _index_unique,
    candidate_payload_sha256,
    dialogue_group,
    rescue_selection,
    selection_metrics,
    validate_score_artifact,
)


def _example() -> dict:
    candidate_ids = ["weak", "redundant", "strong"]
    return {
        "query_id": "7_event_ordering_0",
        "group": "7",
        "target_count": 2,
        "candidate_ids": candidate_ids,
        "baseline_ids": ["weak", "redundant"],
        "candidate_text": {candidate_id: candidate_id for candidate_id in candidate_ids},
        "features": {
            "weak": {
                "query": 0.30,
                "rank": 0.20,
                "detail": 0.5,
                "centrality": 0.2,
                "anchor": 0.2,
                "judge": 0.0,
                "turn": 10,
                "session": "2024-01-01",
            },
            "redundant": {
                "query": 0.40,
                "rank": 0.30,
                "detail": 0.5,
                "centrality": 0.2,
                "anchor": 0.2,
                "judge": 0.0,
                "turn": 20,
                "session": "2024-01-01",
            },
            "strong": {
                "query": 0.90,
                "rank": 0.80,
                "detail": 0.5,
                "centrality": 0.8,
                "anchor": 0.8,
                "judge": 0.0,
                "turn": 30,
                "session": "2024-02-01",
            },
        },
        "pair_similarity": {
            (left, right): 1.0 if left == right else 0.1
            for left in candidate_ids
            for right in candidate_ids
        },
        "gold_items": [
            {"item": "first", "turn": 20},
            {"item": "second", "turn": 30},
        ],
        "gold_similarity": {
            (0, "weak"): 0.10,
            (0, "redundant"): 0.90,
            (0, "strong"): 0.20,
            (1, "weak"): 0.20,
            (1, "redundant"): 0.30,
            (1, "strong"): 0.95,
        },
    }


def test_dialogue_group_keeps_query_variants_together():
    assert dialogue_group("12_event_ordering_0") == "12"
    assert dialogue_group("12_event_ordering_1") == "12"
    assert dialogue_group("custom") == "custom"


def test_group_sort_key_supports_numeric_and_custom_groups():
    assert sorted(["custom", "10", "2"], key=_group_sort_key) == [
        "2",
        "10",
        "custom",
    ]


def test_index_unique_rejects_duplicate_or_missing_join_keys():
    with pytest.raises(ValueError, match="duplicate gold query"):
        _index_unique(
            [{"query": "same"}, {"query": "same"}], "query", "gold"
        )
    with pytest.raises(ValueError, match="missing 'query'"):
        _index_unique([{}], "query", "gold")


def test_score_artifact_rejects_stale_candidate_payload():
    row = {
        "query": "Which milestones?",
        "requested_item_count": 1,
        "candidate_items": [{"id": "a", "item": "Current text"}],
    }
    scored = {
        "query": row["query"],
        "candidate_artifact_sha256": "artifact-hash",
        "candidate_payload_sha256": candidate_payload_sha256(
            {**row, "candidate_items": [{"id": "a", "item": "Old text"}]}
        ),
        "scorer_protocol_version": 2,
        "prompt_sha256": "a" * 64,
        "model": "model",
        "model_metadata": {"digest": "digest"},
        "num_ctx": 32768,
        "temperature": 0.0,
        "seed": 0,
        "think": False,
        "scores": [{"id": "a", "relevance": 80, "atomicity": 80}],
    }

    with pytest.raises(ValueError, match="provenance does not match"):
        validate_score_artifact(
            [scored], {row["query"]: row}, "artifact-hash"
        )


def test_baseline_ids_require_exact_source_or_text_identity():
    row = {
        "candidate_items": [
            {"id": "a", "item": "first"},
            {"id": "b", "item": "second"},
        ],
        "results": [
            {"item": "ignored", "source_item_ids": ["a"]},
            {"item": "second", "source_item_ids": []},
        ],
    }
    assert _baseline_ids(row) == ["a", "b"]


def test_baseline_config_preserves_the_served_selection():
    assert rescue_selection(_example(), BASELINE_CONFIG) == ["weak", "redundant"]


def test_rescue_swaps_only_when_the_margin_is_exceeded():
    config = RescueConfig(margin=0.10, max_replacements=1)
    assert rescue_selection(_example(), config) == ["strong", "redundant"]

    conservative = RescueConfig(margin=0.70, max_replacements=1)
    assert rescue_selection(_example(), conservative) == ["weak", "redundant"]


def test_rescue_can_prefer_a_new_conversation_session():
    example = _example()
    for candidate_id in example["candidate_ids"]:
        example["features"][candidate_id]["query"] = 0.5

    selected = rescue_selection(
        example,
        RescueConfig(session_weight=0.5, margin=0.1, max_replacements=1),
    )

    assert selected == ["strong", "redundant"]


def test_selection_metrics_separate_membership_from_position():
    metrics = selection_metrics(_example(), ["redundant", "strong"], threshold=0.55)
    assert metrics["semantic_coverage"] == 0.925
    assert metrics["matched_recall"] == 1.0
    assert metrics["selection_precision"] == 1.0
    assert metrics["turn_recall"] == 1.0
    assert metrics["chronological_similarity"] == 0.925


def test_selection_metrics_match_candidates_to_gold_one_to_one():
    example = _example()
    example["gold_similarity"] = {
        (0, "weak"): 0.9,
        (0, "redundant"): 0.8,
        (0, "strong"): 0.1,
        (1, "weak"): 0.9,
        (1, "redundant"): 0.2,
        (1, "strong"): 0.1,
    }

    metrics = selection_metrics(example, ["weak", "strong"], threshold=0.55)

    assert metrics["semantic_coverage"] == 0.5
    assert metrics["matched_recall"] == 0.5
    assert metrics["selection_precision"] == 0.5


def test_threshold_recall_maximizes_qualifying_one_to_one_pairs():
    example = _example()
    example["gold_similarity"] = {
        (0, "weak"): 0.99,
        (0, "redundant"): 0.76,
        (0, "strong"): 0.1,
        (1, "weak"): 0.76,
        (1, "redundant"): 0.54,
        (1, "strong"): 0.1,
    }

    metrics = selection_metrics(example, ["weak", "redundant"], threshold=0.55)

    assert metrics["semantic_coverage"] == 0.765
    assert metrics["matched_recall"] == 1.0
    assert metrics["selection_precision"] == 1.0


def test_chronological_similarity_normalizes_nonmonotonic_gold_alignments():
    example = _example()
    example["gold_items"] = [
        {"item": "later", "turn": 30},
        {"item": "earlier", "turn": 20},
    ]
    example["gold_similarity"] = {
        (0, "weak"): 0.1,
        (0, "redundant"): 0.1,
        (0, "strong"): 0.9,
        (1, "weak"): 0.1,
        (1, "redundant"): 0.9,
        (1, "strong"): 0.1,
    }

    metrics = selection_metrics(example, ["strong", "redundant"])

    assert metrics["chronological_similarity"] == 0.9


def test_product_probe_gate_rejects_source_turn_regression():
    delta = {
        "matched_recall": 0.01,
        "turn_recall": -0.01,
        "chronological_similarity": 0.01,
    }
    intervals = {"semantic_coverage": (0.001, 0.02)}

    assert not _accept_for_product_probe(delta, intervals)
