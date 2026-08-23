from datetime import datetime, timezone

from benchmarks.amb.audit_event_time_filter_opportunity import (
    analyze,
    classify_query_time_filter,
)


def _ts(month, day, hour=0, minute=0, second=0):
    return datetime(2024, month, day, hour, minute, second, tzinfo=timezone.utc).timestamp()


def test_classifier_maps_closed_and_one_sided_query_semantics():
    exact = classify_query_time_filter("What happened on May 12?")
    assert exact["semantics"] == "exact_day"
    assert exact["event_after"] == _ts(5, 12)
    assert exact["event_before"] == _ts(5, 12, 23, 59, 59)

    bounded = classify_query_time_filter("What changed between April 6-7 and April 8?")
    assert bounded["semantics"] == "closed_window"
    assert bounded["event_after"] == _ts(4, 6)
    assert bounded["event_before"] == _ts(4, 8, 23, 59, 59)

    before = classify_query_time_filter("What had I finished by July 5?")
    assert before["semantics"] == "before"
    assert before["event_after"] is None
    assert before["event_before"] == _ts(7, 5, 23, 59, 59)

    after = classify_query_time_filter("What improved after April 5, 2024?")
    assert after["semantics"] == "after"
    assert after["event_after"] == _ts(4, 5)
    assert after["event_before"] is None


def test_classifier_excludes_ambiguous_reference_dates_from_filter_scope():
    result = classify_query_time_filter("What advice applied beyond the April 15 deadline?")

    assert result["semantics"] == "ambiguous_reference"
    assert result["event_after"] is None
    assert result["event_before"] is None


def test_analyze_separates_query_only_cohort_from_score_ceiling():
    rows = [
        {
            "query_id": "exact",
            "query": "What happened on May 12?",
            "score": 1.0,
            "meta": {"question_category": "abstention"},
        },
        {
            "query_id": "after",
            "query": "What improved after April 5?",
            "score": 0.0,
            "meta": {"question_category": "temporal_reasoning"},
        },
        {
            "query_id": "ambiguous",
            "query": "What advice applied beyond the April 15 deadline?",
            "score": 0.0,
            "meta": {"question_category": "abstention"},
        },
        {
            "query_id": "none",
            "query": "What changed?",
            "score": 0.0,
            "meta": {"question_category": "knowledge_update"},
        },
    ]

    report = analyze(rows)

    assert report["any_explicit_date"]["n"] == 3
    assert report["any_explicit_date"]["perfect_arm_full_benchmark_delta_ceiling"] == 0.5
    assert report["unambiguous_filter"]["n"] == 2
    assert report["unambiguous_filter"]["perfect_arm_full_benchmark_delta_ceiling"] == 0.25
    assert report["closed_window_only"]["n"] == 1
    assert report["interpretation"]["perfect_arm_ceiling_is_not_expected_lift"] is True
