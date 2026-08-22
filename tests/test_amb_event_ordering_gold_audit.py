from benchmarks.amb.audit_event_ordering_gold import (
    result_rows,
    select_first_mention_match,
    select_monotonic_matches,
    source_documents,
    split_numbered_items,
)


def test_split_numbered_items_preserves_internal_commas():
    answer = (
        "You mentioned these in order: 1) Planning scope, budget, and timing, "
        "2) Shipping the first version, 3) Reviewing the measured outcome."
    )

    assert split_numbered_items(answer) == [
        "Planning scope, budget, and timing",
        "Shipping the first version",
        "Reviewing the measured outcome",
    ]


def test_source_documents_formats_product_recall_hits():
    row = {
        "hits": [
            {
                "text": "First concern",
                "metadata": {"first_mention_turn": 24},
            },
            {"text": "Undated concern", "metadata": {}},
        ]
    }

    assert source_documents(row) == [
        "[Turn 24] User: First concern",
        "Undated concern",
    ]


def test_result_rows_accepts_wrapped_and_raw_artifacts():
    rows = [{"query_id": "q1"}]

    assert result_rows(rows) is rows
    assert result_rows({"results": rows}) is rows


def test_select_first_mention_match_uses_confidence_band():
    matches = [
        {"score": 0.80, "turn": 100},
        {"score": 0.77, "turn": 20},
        {"score": 0.70, "turn": 5},
    ]

    assert select_first_mention_match(matches, 0.05)["turn"] == 20


def test_monotonic_alignment_uses_later_recurring_mention_when_needed():
    alignments = [
        {
            "matches": [
                {"score": 0.90, "turn": 10},
                {"score": 0.80, "turn": 40},
            ]
        },
        {
            "matches": [
                {"score": 0.95, "turn": 5},
                {"score": 0.85, "turn": 30},
            ]
        },
    ]

    selected = select_monotonic_matches(alignments)

    assert [match["turn"] for match in selected] == [10, 30]


def test_monotonic_alignment_reports_missing_path():
    alignments = [
        {"matches": [{"score": 0.90, "turn": 20}]},
        {"matches": [{"score": 0.95, "turn": 10}]},
    ]

    assert select_monotonic_matches(alignments) is None
