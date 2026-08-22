import json

from benchmarks.amb.contextual_funnel_preflight import aggregate, load_event_queries


def test_load_event_queries_flattens_nested_source_turns(tmp_path):
    source = [{
        "conversation_id": 7,
        "probing_questions": {
            "event_ordering": [{
                "question": "What happened?",
                "source_chat_ids": [3, [5, 6], True],
            }]
        },
    }]
    path = tmp_path / "beam.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    assert load_event_queries(path) == [{
        "query_id": "7_event_ordering_0",
        "user_id": "7",
        "query": "What happened?",
        "source_turns": {3, 5, 6},
    }]


def test_aggregate_reports_source_gain_and_bank_size():
    summary = aggregate([
        {
            "gold_source_turn_count": 3,
            "base_available_source_turn_count": 1,
            "bank_available_source_turn_count": 2,
            "base_all_available": False,
            "bank_all_available": False,
            "candidate_bank_count": 45,
        },
        {
            "gold_source_turn_count": 2,
            "base_available_source_turn_count": 2,
            "bank_available_source_turn_count": 2,
            "base_all_available": True,
            "bank_all_available": True,
            "candidate_bank_count": 50,
        },
    ])

    assert summary["base_source_turn_recall"] == 3 / 5
    assert summary["bank_source_turn_recall"] == 4 / 5
    assert summary["source_turn_gain"] == 1
    assert summary["candidate_bank_median"] == 47.5
