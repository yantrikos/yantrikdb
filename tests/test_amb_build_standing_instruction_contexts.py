import pytest

from benchmarks.amb.build_standing_instruction_contexts import (
    build_contexts,
    extract_standing_instructions,
)


def _conversation():
    return {
        "chat": [
            {"id": 1, "role": "user", "content": "Ordinary fact."},
            {
                "id": 2,
                "role": "user",
                "content": "Always use APA citations. ->-> 1,2",
            },
            {"id": 3, "role": "assistant", "content": "Always be concise."},
            {"id": 4, "role": "user", "content": "Always include dates."},
        ]
    }


def test_extract_standing_instructions_is_user_authoritative_and_ordered():
    assert extract_standing_instructions(_conversation()) == [
        {"turn": 2, "text": "Always use APA citations."},
        {"turn": 4, "text": "Always include dates."},
    ]


def test_build_contexts_prepends_panel_and_caps_whole_blocks():
    context = (
        "## Memory 1\nfirst raw memory with several words\n\n"
        "## Memory 2\nsecond raw memory with several words\n\n"
        "## Memory 3\nthird raw memory with several words\n\n"
    )
    row = {
        "query_id": "1_instruction_following_0",
        "context": context,
        "meta": {
            "question_category": "instruction_following",
            "conversation_id": "1",
        },
    }

    payload = build_contexts(
        [row], {"1": _conversation()}, lambda value: len(value.split())
    )
    result = payload["results"][0]

    assert result["context"].startswith(
        "## Memory 0\nStanding user instructions (authoritative user statements):"
    )
    assert "Assistant: Always be concise" not in result["context"]
    assert result["standing_instruction_audit"]["instruction_turns"] == [2, 4]
    assert result["standing_instruction_audit"]["blocks_after_cap"] < 4
    assert payload["artifact_transform"]["treatment_within_reference_budget"] is True


def test_build_contexts_rejects_units_without_standing_instructions():
    row = {
        "query_id": "1_instruction_following_0",
        "context": "## Memory 1\nraw memory",
        "meta": {
            "question_category": "instruction_following",
            "conversation_id": "1",
        },
    }
    with pytest.raises(ValueError, match="no standing instructions"):
        build_contexts(
            [row],
            {"1": {"chat": [{"id": 1, "role": "user", "content": "Fact"}]}},
            lambda value: len(value.split()),
        )
