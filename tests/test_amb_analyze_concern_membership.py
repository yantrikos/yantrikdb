import pytest

from benchmarks.amb.analyze_concern_membership import (
    analyze,
    best_bounded_cover,
    concern_turn_sets,
    requested_item_count,
)


def test_requested_item_count_reads_exact_constraint():
    assert requested_item_count("Mention ONLY and ONLY five items.") == 5
    assert requested_item_count("Give exactly 12 events") == 12
    assert requested_item_count("Give me a timeline") is None


def test_concern_turn_sets_requires_matching_provenance():
    organizer = {
        "input_sha256": "same",
        "input_items": [
            {"id": "A1", "turn": 2},
            {"id": "A2", "turn": 8},
        ],
    }
    concerns = {
        "input_sha256": "same",
        "items": [
            {"id": "c1", "text": "First", "evidence_ids": ["A2", "A1"]}
        ],
    }
    assert concern_turn_sets(concerns, organizer) == [
        {
            "kind": "concern",
            "label": "c1",
            "text": "First",
            "turns": {2, 8},
        }
    ]
    concerns["input_sha256"] = "different"
    with pytest.raises(ValueError, match="do not share"):
        concern_turn_sets(concerns, organizer)


def test_best_bounded_cover_uses_multiple_answer_sized_records():
    collections = [
        {"label": "a", "turns": {1, 99}},
        {"label": "b", "turns": {2}},
        {"label": "c", "turns": {3}},
        {"label": "d", "turns": {50}},
    ]
    assert best_bounded_cover({1, 2, 3}, collections, 1)["recall"] == 1 / 3
    result = best_bounded_cover({1, 2, 3}, collections, 3)
    assert result["recall"] == 1.0
    assert result["turns"] == [1, 2, 3]


def test_analyze_reports_requested_count_without_using_gold_for_generation():
    organizer = {
        "input_sha256": "digest",
        "input_items": [
            {"id": "A1", "turn": 1},
            {"id": "A2", "turn": 2},
        ],
    }
    concerns = {
        "input_sha256": "digest",
        "items": [
            {"id": "c1", "text": "First", "evidence_ids": ["A1"]},
            {"id": "c2", "text": "Second", "evidence_ids": ["A2"]},
        ],
    }
    report = analyze(
        [
            {
                "query_id": "9_event_ordering_0",
                "query": "Mention exactly two items",
                "source_turn_ids": [1, 2],
            }
        ],
        {"9": concerns},
        {"9": organizer},
    )
    assert report["gold_used_for_generation_or_retrieval"] is False
    assert report["requested_count"] == {
        "queries": 1,
        "mean_source_recall": 1.0,
        "exact_queries": 1,
    }
