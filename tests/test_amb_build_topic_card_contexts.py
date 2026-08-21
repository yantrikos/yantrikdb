from benchmarks.amb.build_topic_card_contexts import (
    build_context_rows,
    render_topic_cards,
)


def _artifact():
    return {
        "input_sha256": "abc",
        "input_items": [
            {"id": "A0001", "date": 1709251200.0, "turn": 4},
            {"id": "A0002", "date": 1709337600.0, "turn": 8},
            {"id": "A0003", "date": 1709424000.0, "turn": 12},
        ],
        "singleton_fallback_count": 1,
        "singleton_fallback_evidence_ids": ["A0002"],
        "handles": [
            {
                "label": "Navbar fixes",
                "summary": "Guarded classList access.",
                "evidence_ids": ["A0001", "A0003"],
            },
            {
                "label": "Unclustered source item A0002",
                "summary": "Added retry backoff.",
                "evidence_ids": ["A0002"],
            },
        ],
    }


def test_render_topic_cards_preserves_organizer_order_and_fallback_text():
    context = render_topic_cards(_artifact())

    assert context == (
        "## Memory 1\nTopic: Navbar fixes (recorded 2024-03-01 to 2024-03-03; "
        "turns 4-12)\nGuarded classList access.\n\n"
        "## Memory 2\nTopic: Unclustered source item A0002 (recorded 2024-03-02; "
        "turn 8)\nAdded retry backoff."
    )


def test_build_context_rows_filters_unit_and_records_provenance():
    rows = build_context_rows(
        [
            {"query_id": "3_summarization_0", "query": "Summarize it"},
            {"query_id": "18_summarization_0", "query": "Other unit"},
        ],
        _artifact(),
        "3",
        include_singletons=False,
        query_ids={"3_summarization_0"},
    )

    assert [row["query_id"] for row in rows] == ["3_summarization_0"]
    assert "Navbar fixes" in rows[0]["context"]
    assert "retry backoff" not in rows[0]["context"]
    assert rows[0]["documents"] == [
        "Topic: Navbar fixes (recorded 2024-03-01 to 2024-03-03; turns 4-12)\n"
        "Guarded classList access."
    ]
    assert rows[0]["selection"] == {
        "mode": "query_independent_topic_cards_v1",
        "handle_count": 2,
        "singleton_fallback_count": 0,
        "organizer_input_sha256": "abc",
    }
