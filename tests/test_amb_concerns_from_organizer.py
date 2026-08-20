from benchmarks.amb.concerns_from_organizer import (
    _normalize_handle_items,
    merge_cross_handle_duplicates,
)


def test_normalize_rejects_evidence_outside_handle():
    items, invalid = _normalize_handle_items(
        1,
        {"label": "Writing"},
        [
            {
                "text": "A concern",
                "anchor_entities": ["Carla"],
                "evidence_ids": ["A1", "missing"],
            }
        ],
        {"A1"},
    )

    assert items[0]["evidence_ids"] == ["A1"]
    assert invalid == ["missing"]


def test_merge_collapses_overlapping_cross_handle_views():
    items = [
        {
            "text": "Carla shared a checklist.",
            "anchor_entities": ["Carla"],
            "evidence_ids": ["A1", "A2"],
            "topic_handle": "Editing",
            "topic_index": 1,
            "local_index": 1,
        },
        {
            "text": "The checklist reduced passive voice.",
            "anchor_entities": ["Carla"],
            "evidence_ids": ["A2"],
            "topic_handle": "Carla",
            "topic_index": 2,
            "local_index": 1,
        },
    ]

    merged = merge_cross_handle_duplicates(items)

    assert len(merged) == 1
    assert merged[0]["evidence_ids"] == ["A1", "A2"]
    assert merged[0]["topic_handles"] == ["Editing", "Carla"]


def test_merge_preserves_two_concerns_split_from_one_compound_record():
    items = [
        {
            "text": "The user drafted a four-part essay outline.",
            "anchor_entities": [],
            "evidence_ids": ["A1"],
            "topic_handle": "Writing",
            "topic_index": 1,
            "local_index": 1,
        },
        {
            "text": "Wendy advised emphasizing cultural roots.",
            "anchor_entities": ["Wendy"],
            "evidence_ids": ["A1"],
            "topic_handle": "Writing",
            "topic_index": 1,
            "local_index": 2,
        },
    ]

    merged = merge_cross_handle_duplicates(items)

    assert len(merged) == 2
    assert {item["text"] for item in merged} == {item["text"] for item in items}


def test_merge_preserves_semantically_distinct_cross_handle_views():
    items = [
        {
            "text": "The user drafted a four-part essay outline.",
            "anchor_entities": [],
            "evidence_ids": ["A1"],
            "topic_handle": "Writing",
            "topic_index": 1,
            "local_index": 1,
        },
        {
            "text": "Wendy advised emphasizing cultural roots.",
            "anchor_entities": ["Wendy"],
            "evidence_ids": ["A1"],
            "topic_handle": "Family",
            "topic_index": 2,
            "local_index": 1,
        },
    ]

    assert len(merge_cross_handle_duplicates(items)) == 2
