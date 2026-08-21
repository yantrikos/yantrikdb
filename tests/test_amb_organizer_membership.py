from benchmarks.amb.analyze_organizer_membership import (
    analyze,
    anchor_turn_sets,
    best_cover,
    organizer_turn_sets,
    query_matched_cover,
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

    assert report["protocol"] == "organizer-membership-routing-audit-v2"
    assert report["gold_used_for_generation_or_retrieval"] is False
    assert report["topic_handles"]["1"]["mean_source_recall"] == 0.5
    assert report["topics_plus_virtual_anchors"]["1"]["mean_source_recall"] == 1.0
    assert report["topic_handles"]["2"]["exact_queries"] == 1


def test_query_matched_cover_uses_product_focus_then_entity_routes():
    collections = [
        {
            "kind": "topic_handle",
            "label": "City autocomplete implementation",
            "anchor_entities": ["Weather app"],
            "turns": {2, 8},
        },
        {
            "kind": "topic_handle",
            "label": "Carla editing collaboration",
            "anchor_entities": ["Carla"],
            "turns": {50, 90},
        },
    ]

    focused = query_matched_cover(
        "List the city autocomplete feature stages in order", collections
    )
    assert focused["route"] == "focus"
    assert focused["labels"] == ["City autocomplete implementation"]

    entity = query_matched_cover(
        "Walk me through my collaboration with Carla", collections
    )
    assert entity["route"] == "entity"
    assert entity["labels"] == ["Carla editing collaboration"]

    assert query_matched_cover("List my journey", collections)["route"] is None


def test_entity_first_route_does_not_let_focus_drop_same_person_handles():
    collections = [
        {
            "kind": "topic_handle",
            "label": "Douglas shared entertainment interests",
            "anchor_entities": ["Douglas"],
            "turns": {2},
        },
        {
            "kind": "topic_handle",
            "label": "Douglas television plans",
            "anchor_entities": ["Douglas"],
            "turns": {8},
        },
    ]
    query = "List my shared entertainment interests with Douglas"

    focused_first = query_matched_cover(query, collections)
    entity_first = query_matched_cover(query, collections, entity_first=True)

    assert focused_first["labels"] == ["Douglas shared entertainment interests"]
    assert entity_first["route"] == "entity"
    assert entity_first["labels"] == [
        "Douglas shared entertainment interests",
        "Douglas television plans",
    ]
