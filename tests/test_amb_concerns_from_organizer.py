from benchmarks.amb.concerns_from_organizer import (
    _artifact_atomics,
    _handle_prompt,
    _normalize_handle_items,
    _schema,
    merge_cross_handle_duplicates,
)


def test_artifact_atomics_loads_frozen_organizer_input():
    assert _artifact_atomics(
        {
            "input_items": [
                {
                    "id": "A1",
                    "turn": 8,
                    "axis": "source_turn",
                    "text": " Patrick shared advice. ",
                    "date": 123.0,
                }
            ]
        }
    ) == [
        {
            "id": "A1",
            "turn": 8,
            "axis": "source_turn",
            "text": "Patrick shared advice.",
        }
    ]


def test_thread_prompt_requires_query_free_exhaustive_chronological_grouping():
    prompt = _handle_prompt(
        {"label": "Mentorship", "summary": "Advice over time"},
        [{"id": "A1", "turn": 4, "text": "Bryan advised storytelling."}],
        8,
    )

    assert "1-8 narrow chronological concern threads" in prompt
    assert "Cover every supplied evidence ID" in prompt
    assert "No benchmark question or expected answer" in prompt
    assert '{"items":[' in prompt
    assert _schema(8)["properties"]["items"]["maxItems"] == 8


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
