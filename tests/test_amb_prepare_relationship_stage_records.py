import pytest

from benchmarks.amb.prepare_relationship_stage_records import (
    _normalize_host,
    _sha256_json,
    build_preflight,
    build_prompt,
    materialize_records,
    render_records,
)


def _artifact():
    rows = [
        {
            "identity": "e1",
            "turn": 10,
            "created_at": 1_700_000_000.0,
            "source_doc_id": "s1",
            "status": "kept",
            "reason": "base_top_k",
            "contextual_score": 0.8,
            "text": "[Turn 10] User: I met mentor Robert at the library.",
        },
        {
            "identity": "e2",
            "turn": 12,
            "created_at": 1_700_000_000.0,
            "source_doc_id": "s1",
            "status": "kept",
            "reason": "context_bridge",
            "parent_turn": 10,
            "contextual_score": 0.7,
            "text": "[Turn 12] User: I planned a follow-up call.",
        },
        {
            "identity": "e3",
            "turn": 30,
            "created_at": 1_710_000_000.0,
            "source_doc_id": "s2",
            "status": "kept",
            "reason": "base_top_k",
            "contextual_score": 0.9,
            "text": "[Turn 30] User: Robert reviewed my draft.",
        },
    ]
    return {
        "query": "List my academic mentorship stages in order.",
        "preflight": {
            "evidence_ledger": rows,
            "source_evidence_ceiling": {
                "available_source_turns": [10, 30],
                "missing_source_turns": [],
            },
        },
    }


@pytest.mark.parametrize("host", ["0.0.0.0:11434", "http://0.0.0.0:11434"])
def test_normalize_host_rewrites_ollama_bind_address(host):
    assert _normalize_host(host) == "http://127.0.0.1:11434"


def test_preflight_is_global_query_blind_and_preserves_all_evidence():
    preflight = build_preflight(_artifact(), "q", "model")

    assert preflight["model_calls"] == 0
    assert preflight["query_exposed_to_synthesis"] is False
    assert preflight["source_group_count"] == 2
    assert preflight["evidence_row_count"] == 3
    assert "List my academic mentorship stages" not in preflight["prompt"]
    assert preflight["prompt"].count('"source_group": "s1"') == 1
    assert "e1" in preflight["prompt"]
    assert "e2" in preflight["prompt"]
    assert "e3" in preflight["prompt"]


def test_materialize_records_restores_source_order_and_owns_provenance():
    preflight = build_preflight(_artifact(), "q", "model")
    response = {
        "records": [
            {"source_group": "s2", "item": "  Draft feedback.  "},
            {"source_group": "s1", "item": "Library meeting and follow-up."},
        ]
    }

    records = materialize_records(response, preflight["groups"])

    assert [record["source_group"] for record in records] == ["s1", "s2"]
    assert records[0]["item"] == "Library meeting and follow-up."
    assert records[0]["first_mention_turn"] == 10
    assert records[0]["evidence_ids"] == ["e1", "e2"]
    assert records[0]["evidence_turns"] == [10, 12]


@pytest.mark.parametrize(
    "response",
    [
        {"records": [{"source_group": "s1", "item": "One."}]},
        {"records": ["not-an-object"]},
        {
            "records": [
                {"source_group": "s1", "item": "One."},
                {"source_group": "s1", "item": "Duplicate."},
            ]
        },
    ],
)
def test_materialize_records_rejects_missing_or_duplicate_groups(response):
    preflight = build_preflight(_artifact(), "q", "model")

    with pytest.raises(ValueError):
        materialize_records(response, preflight["groups"])


def test_prompt_and_render_make_the_record_contract_explicit():
    preflight = build_preflight(_artifact(), "q", "model")
    prompt = build_prompt(preflight["anchor"], preflight["groups"])
    records = materialize_records(
        {
            "records": [
                {"source_group": "s1", "item": "Library meeting and follow-up."},
                {"source_group": "s2", "item": "Draft feedback."},
            ]
        },
        preflight["groups"],
    )
    context = render_records(preflight["anchor"], records)

    assert "before any future question is known" in prompt
    assert "never merge across source_group" in prompt
    assert '{"records":[{"source_group":"COPY_FROM_INPUT"' in prompt
    assert "Source conversation: s1" in context
    assert "Evidence turns: 10, 12" in context
    assert "Evidence IDs:" not in context
    assert records[0]["evidence_ids"] == ["e1", "e2"]


def test_response_hash_is_order_independent_for_replay_binding():
    left = {"records": [{"source_group": "s1", "item": "One."}]}
    right = {"records": [{"item": "One.", "source_group": "s1"}]}

    assert _sha256_json(left) == _sha256_json(right)
