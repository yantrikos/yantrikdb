from benchmarks.amb.temporal_summary_intent import bounded_calendar_summary_intent


def test_matches_single_named_month_with_year():
    matched, trace = bounded_calendar_summary_intent(
        "Summarize the adjustments I made in March 2024."
    )

    assert matched
    assert trace["months"] == ["march"]
    assert trace["years"] == ["2024"]


def test_matches_named_month_range_without_year():
    matched, trace = bounded_calendar_summary_intent(
        "Summarize my actions between March and early May."
    )

    assert matched
    assert trace["months"] == ["march", "may"]


def test_matches_iso_calendar_summary():
    matched, trace = bounded_calendar_summary_intent(
        "Give me a summary from 2024-07-01 through 2024-09-30."
    )

    assert matched
    assert trace["iso_dates"] == ["2024-07-01", "2024-09-30"]


def test_rejects_unbounded_or_non_summary_temporal_queries():
    queries = (
        "Can you summarize how my resume developed over the past few months?",
        "Can you summarize everything we discussed in March?",
        "What did I decide between March and May 2024?",
        "Can you give me a comprehensive project summary?",
    )

    for query in queries:
        matched, _ = bounded_calendar_summary_intent(query)
        assert not matched, query
