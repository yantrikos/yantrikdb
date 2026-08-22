import json

from benchmarks.amb.replay_contextual_synthesis import (
    _candidate_bank,
    _compare_preflight,
    _evidence_ledger,
    _preflight_record,
    _rank_by_turn,
    _source_evidence_ceiling,
)


def _hit(rid, turn, base_rank, score):
    return {
        "rid": rid,
        "text": f"[Session-1-1 | Turn {turn}] User: evidence {rid}",
        "_retrieval_rank": base_rank,
        "_contextual_score": score,
    }


def test_rank_by_turn_keeps_best_rank_for_duplicate_turns():
    reranked = [
        _hit("a", 10, 4, 0.9),
        _hit("b", 20, 2, 0.8),
        _hit("c", 10, 1, 0.7),
    ]

    assert _rank_by_turn(reranked) == {10: 1, 20: 2}


def test_evidence_ledger_accounts_for_every_cutoff_discard():
    reranked = [
        _hit("a", 10, 4, 0.9),
        _hit("b", 20, 2, 0.8),
        _hit("c", 30, 1, 0.7),
    ]

    ledger = _evidence_ledger(reranked, top_k=2)

    assert [row["status"] for row in ledger] == ["kept", "kept", "dropped"]
    assert [row["reason"] for row in ledger] == [
        "base_top_k",
        "base_top_k",
        "outside_candidate_bank",
    ]
    assert [row["identity"].split(":", 1)[0] for row in ledger] == [
        "a",
        "b",
        "c",
    ]


def test_evidence_ledger_preserves_source_provenance():
    hit = {
        **_hit("a", 10, 4, 0.9),
        "created_at": 123.5,
        "metadata": {"doc_id": "session-a", "turn_id": 10, "chunk_idx": 2},
    }

    row = _evidence_ledger([hit], top_k=1)[0]

    assert row["created_at"] == 123.5
    assert row["source_doc_id"] == "session-a"
    assert row["source_turn_id"] == 10
    assert row["source_chunk_idx"] == 2


def test_source_evidence_ceiling_reports_available_and_missing_turns():
    ceiling = _source_evidence_ceiling(
        {10, 20, 30}, {10: 1, 20: 3, 30: 9}, {10, 20}
    )

    assert ceiling == {
        "gold_source_turn_count": 3,
        "available_source_turn_count": 2,
        "source_turn_recall": 2 / 3,
        "all_gold_source_turns_available": False,
        "available_source_turns": [10, 20],
        "missing_source_turns": [30],
        "source_turn_ranks": {"10": 1, "20": 3, "30": 9},
    }


def test_preflight_hashes_are_stable_and_cover_model_output():
    hits = [_hit("a", 10, 1, 0.9), _hit("b", 20, 2, 0.8)]
    kwargs = {
        "query": "Which two events?",
        "model_identity": {"digest": "sha256:model"},
        "embed_model": "embedder",
        "rerank_pool": 10,
        "top_k": 1,
        "user_hits": hits,
        "reranked": hits,
        "selected": hits[:1],
        "selection_reasons": {
            _identity: {"status": "kept", "reason": "base_top_k"}
            for _identity in [
                _evidence_ledger(hits, top_k=1)[0]["identity"]
            ]
        },
        "gold_turns": {10, 20},
        "entity_seed_k": 1,
        "entity_closure_slots": 0,
        "continuation_slots": 0,
    }

    first = _preflight_record(**kwargs)
    second = _preflight_record(**kwargs)
    changed = _preflight_record(**{**kwargs, "reranked": list(reversed(hits))})

    assert first["input_sha256"] == second["input_sha256"]
    assert first["rerank_sha256"] == second["rerank_sha256"]
    assert first["rerank_sha256"] != changed["rerank_sha256"]
    assert first["source_evidence_ceiling"]["available_source_turns"] == [10]


def test_compare_preflight_checks_input_rerank_and_model_digest(tmp_path):
    frozen = {
        "preflight": {
            "input_sha256": "input-a",
            "rerank_sha256": "rank-a",
            "candidate_bank_sha256": "bank-a",
            "model_identity": {"digest": "model-a"},
        }
    }
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(frozen), encoding="utf-8")

    matched = _compare_preflight(frozen["preflight"], path)
    changed = _compare_preflight(
        {
            "input_sha256": "input-a",
            "rerank_sha256": "rank-b",
            "candidate_bank_sha256": "bank-a",
            "model_identity": {"digest": "model-a"},
        },
        path,
    )

    assert matched["matched"] is True
    assert changed["matched"] is False
    assert changed["mismatches"] == ["rerank_sha256"]


def test_candidate_bank_adds_bounded_entity_closure_and_direct_continuation():
    reranked = [
        {
            **_hit("seed", 10, 1, 0.9),
            "text": "[Turn 10] User: Bryan helped with my draft.",
        },
        _hit("base", 20, 2, 0.8),
        {
            **_hit("continuation", 22, 3, 0.7),
            "text": "[Turn 22] User: Sure, here is the revised draft.",
        },
        {
            **_hit("closure", 30, 4, 0.6),
            "text": "[Turn 30] User: Bryan agreed to send a letter.",
        },
        {
            **_hit("extra", 40, 5, 0.5),
            "text": "[Turn 40] User: unrelated evidence.",
        },
    ]

    selected, reasons = _candidate_bank(
        reranked,
        top_k=2,
        entity_seed_k=1,
        entity_closure_slots=1,
        continuation_slots=1,
    )
    selected_by_rid = {hit["rid"]: reasons for hit in selected}
    reason_values = list(reasons.values())

    assert set(selected_by_rid) == {"seed", "base", "continuation", "closure"}
    assert any(row["reason"] == "entity_closure" for row in reason_values)
    assert any(row["reason"] == "direct_user_continuation" for row in reason_values)


def test_candidate_bank_bridges_named_relationship_into_anaphoric_followup():
    reranked = [
        {
            **_hit("seed", 14, 1, 0.9),
            "text": "[Turn 14] User: I met Robert for mentorship advice.",
        },
        _hit("unrelated", 40, 2, 0.8),
        {
            **_hit("relationship", 212, 3, 0.7),
            "text": (
                "[Turn 212] User: Robert and I are discussing next steps "
                "on July 20."
            ),
        },
        {
            **_hit("followup", 214, 4, 0.6),
            "text": (
                "[Turn 214] User: I want to prepare for our Zoom meeting "
                "and review progress."
            ),
        },
    ]

    selected, reasons = _candidate_bank(
        reranked,
        top_k=1,
        entity_seed_k=1,
        entity_closure_slots=1,
        continuation_slots=0,
        context_bridge_slots=1,
    )

    assert {hit["rid"] for hit in selected} == {
        "seed", "relationship", "followup"
    }
    followup_reason = next(
        value for key, value in reasons.items() if key.startswith("followup:")
    )
    assert followup_reason == {
        "status": "kept",
        "reason": "context_bridge",
        "parent_turn": 212,
    }


def test_context_bridge_ignores_generic_such_phrase():
    reranked = [
        _hit("parent", 124, 1, 0.9),
        {
            **_hit("generic", 126, 2, 0.8),
            "text": (
                "[Turn 126] User: How can I improve in such a short time?"
            ),
        },
    ]

    selected, _ = _candidate_bank(
        reranked,
        top_k=1,
        entity_seed_k=1,
        entity_closure_slots=0,
        continuation_slots=0,
        context_bridge_slots=1,
    )

    assert [hit["rid"] for hit in selected] == ["parent"]
