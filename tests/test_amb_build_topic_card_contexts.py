from benchmarks.amb.build_topic_card_contexts import (
    build_context_rows,
    rank_topic_cards,
    render_topic_cards,
    topic_index_document,
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
        "selected_handle_count": 1,
        "card_limit": None,
        "topic_index_handle_count": 0,
        "topic_index_includes_spans": False,
        "ranking": [],
        "singleton_fallback_count": 0,
        "organizer_input_sha256": "abc",
    }


def test_rank_topic_cards_combines_lexical_and_recorded_date_relevance():
    artifact = {
        "input_items": [
            {"id": "A0001", "date": 1709251200.0, "turn": 1},
            {"id": "A0002", "date": 1722470400.0, "turn": 2},
            {"id": "A0003", "date": 1709251200.0, "turn": 3},
        ],
        "handles": [
            {
                "label": "Budget planning",
                "summary": "Tracked expenses and savings.",
                "evidence_ids": ["A0001"],
            },
            {
                "label": "Writing progress",
                "summary": "Improved drafting in August.",
                "evidence_ids": ["A0002"],
            },
            {
                "label": "Writing practice",
                "summary": "Revised essays in March.",
                "evidence_ids": ["A0003"],
            },
        ],
    }

    documents, trace = rank_topic_cards(
        artifact, "Summarize my writing progress in March 2024", 1
    )

    assert "Writing practice" in documents[0]
    assert trace[0]["temporal_score"] == 2.0
    assert trace[0]["lexical_score"] > 0.0


def test_render_topic_cards_deduplicates_exact_organizer_repeats():
    artifact = _artifact()
    artifact["handles"].append(dict(artifact["handles"][0]))

    context = render_topic_cards(artifact)

    assert context.count("Topic: Navbar fixes") == 1


def test_topic_index_uses_discovered_labels_and_omits_singleton_fallbacks():
    document, count = topic_index_document(_artifact())

    assert count == 1
    assert "Navbar fixes" in document
    assert "Unclustered source item" not in document

    undated, _ = topic_index_document(_artifact(), include_spans=False)
    assert "recorded" not in undated
