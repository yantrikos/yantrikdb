from benchmarks.amb.analyze_membership_funnel import (
    candidate_provenance_turns,
    flatten_source_chat_ids,
    load_beam_event_sources,
    source_turn_metrics,
    split_answer_items,
    split_context_memories,
    stage_metrics,
)


def test_split_context_memories_preserves_all_turn_markers():
    context = """## Memory 1
[Turn 10] User: first
[Turn 11] Assistant: reply

## Memory 2
[April-09-2024 | Turn 72] User: second
Reference to Turn 99 is prose, not a source header.
```
Turn 77
```
"""

    assert split_context_memories(context) == [
        {
            "text": "[Turn 10] User: first\n[Turn 11] Assistant: reply",
            "turns": [10, 11],
        },
        {
            "text": (
                "[April-09-2024 | Turn 72] User: second\n"
                "Reference to Turn 99 is prose, not a source header.\n"
                "```\nTurn 77\n```"
            ),
            "turns": [72],
        },
    ]


def test_split_answer_items_handles_inline_numbering_and_exact_commas():
    assert split_answer_items("1. Alpha, 2. Beta, 3. Gamma", 3) == [
        "Alpha",
        "Beta",
        "Gamma",
    ]
    assert split_answer_items("Alpha, Beta, Gamma", 3) == [
        "Alpha",
        "Beta",
        "Gamma",
    ]
    assert split_answer_items("Alpha, with detail", 3) == ["Alpha, with detail"]


def test_stage_metrics_uses_distinct_threshold_matches_and_exact_turns():
    targets = [
        {"rubric": "first", "text": "gold-a", "turn": 10},
        {"rubric": "second", "text": "gold-b", "turn": 20},
    ]
    elements = [
        {"id": "a", "text": "candidate-a", "turns": [10]},
        {"id": "b", "text": "candidate-b", "turns": [99]},
    ]
    vectors = {
        "gold-a": [1.0, 0.0],
        "gold-b": [0.0, 1.0],
        "candidate-a": [0.9, 0.1],
        "candidate-b": [0.1, 0.9],
    }

    metrics = stage_metrics(targets, elements, vectors, threshold=0.8)

    assert metrics["matched_recall"] == 1.0
    assert metrics["source_turn_recall"] == 0.5
    assert [target["present"] for target in metrics["targets"]] == [True, True]
    assert [
        target["matched_element_record"]["id"]
        for target in metrics["targets"]
    ] == ["a", "b"]


def test_source_turn_metrics_counts_flattened_exact_provenance():
    assert flatten_source_chat_ids([4, [60, 62], 60]) == [4, 60, 62]
    assert source_turn_metrics(
        [4, 60, 62],
        [{"turns": [4, 5]}, {"turns": [62]}],
    ) == {
        "expected": [4, 60, 62],
        "present": [4, 62],
        "missing": [60],
        "recall": 2 / 3,
    }


def test_candidate_provenance_prefers_all_evidence_block_turns():
    item = {
        "first_mention_turn": 4,
        "evidence_ids": ["B001", "B002"],
    }

    assert candidate_provenance_turns(
        item,
        {"B001": [4, 5], "B002": [60]},
    ) == [4, 5, 60]
    assert candidate_provenance_turns(item, {}) == [4]


def test_load_beam_event_sources_extracts_generator_turn_ids(tmp_path):
    source = tmp_path / "beam.json"
    source.write_text(
        """[{"conversation_id": 7, "probing_questions": "{'event_ordering': """
        """[{'question': 'q', 'rubric': ['r'], 'source_chat_ids': [42], """
        """'conversation_references': ['Session 42: topic']}]}"}]""",
        encoding="utf-8",
    )

    assert load_beam_event_sources(source) == [
        {
            "query_id": "7_event_ordering_0",
            "query": "q",
            "rubric": ["r"],
            "source_chat_ids": [42],
            "source_turn_ids": [42],
            "conversation_references": ["Session 42: topic"],
        }
    ]


def test_analyze_rejects_source_query_id_reuse():
    from benchmarks.amb.analyze_membership_funnel import analyze

    candidate = {
        "query": "current query",
        "candidate_items": [],
        "results": [],
    }
    gold = {
        "query_id": "7_event_ordering_0",
        "query": "current query",
        "alignments": [],
    }
    synthesis = {
        "query_id": "7_event_ordering_0",
        "meta": {"rubric": []},
    }
    baseline = {"query_id": "7_event_ordering_0"}
    source = {
        "query_id": "7_event_ordering_0",
        "query": "stale query",
        "rubric": [],
        "source_turn_ids": [],
    }

    import pytest

    with pytest.raises(ValueError, match="candidate/source queries differ"):
        analyze([candidate], [gold], [synthesis], [baseline], [source], {})
