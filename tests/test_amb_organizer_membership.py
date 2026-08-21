from benchmarks.amb.analyze_organizer_membership import (
    analyze,
    anchor_turn_sets,
    best_cover,
    organizer_turn_sets,
)


def _artifact():
    return {
        "input_items": [
            {"id": "A1", "turn": 2},
            {"id": "A2", "turn": 8},
            {"id": "A3", "turn": 14},
        ],
        "handles": [
            {
                "label": "Draft",
                "anchor_entities": ["Carla"],
                "evidence_ids": ["A1"],
            },
            {
                "label": "Webinar",
                "anchor_entities": ["carla"],
                "evidence_ids": ["A2"],
            },
            {
                "label": "Launch",
                "anchor_entities": ["Amy"],
                "evidence_ids": ["A3"],
            },
        ],
    }


def test_turn_sets_resolve_topic_and_casefolded_anchor_membership():
    topics = organizer_turn_sets(_artifact())
    anchors = anchor_turn_sets(_artifact())

    assert [topic["turns"] for topic in topics] == [{2}, {8}, {14}]
    assert len(anchors) == 1
    assert anchors[0]["label"] == "Carla"
    assert anchors[0]["turns"] == {2, 8}
    assert anchors[0]["source_handle_count"] == 2


def test_best_cover_maximizes_membership_then_minimizes_distractors():
    collections = [
        {"kind": "topic", "label": "broad", "turns": {2, 8, 99}},
        {"kind": "topic", "label": "clean", "turns": {2, 8}},
        {"kind": "topic", "label": "late", "turns": {14}},
    ]

    one = best_cover({2, 8, 14}, collections, 1)
    two = best_cover({2, 8, 14}, collections, 2)

    assert one["labels"] == ["clean"]
    assert one["recall"] == 2 / 3
    assert two["labels"] == ["clean", "late"]
    assert two["recall"] == 1.0


def test_analysis_keeps_gold_out_of_construction_and_reports_oracle_lift():
    questions = [
        {
            "query_id": "1_event_ordering_0",
            "source_turn_ids": [2, 8],
        }
    ]

    report = analyze(questions, {"1": _artifact()}, counts=(1, 2))

    assert report["gold_used_for_generation_or_retrieval"] is False
    assert report["topic_handles"]["1"]["mean_source_recall"] == 0.5
    assert report["topics_plus_virtual_anchors"]["1"]["mean_source_recall"] == 1.0
    assert report["topic_handles"]["2"]["exact_queries"] == 1
