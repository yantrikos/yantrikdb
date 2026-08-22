from benchmarks.amb.prepare_contextual_synthesis_pair import (
    _safe_name,
    context_row,
    render_context,
    render_relationship_support_stages,
    render_relationship_thread,
    render_selected_evidence,
    select_relationship_thread,
)


def _artifact():
    return {
        "preflight": {
            "input_sha256": "input",
            "rerank_sha256": "rerank",
            "candidate_bank_sha256": "bank",
            "source_evidence_ceiling": {
                "all_gold_source_turns_available": True,
                "available_source_turns": [4, 60],
                "missing_source_turns": [],
            },
        },
        "candidate_gold_turns": [4, 60],
        "result_gold_turns": [4],
        "items": [{
            "item": "First milestone",
            "first_mention_date": "2024-03-14",
            "first_mention_turn": 4,
            "first_mention_position": 1,
            "evidence_ids": ["B005"],
            "date_source": "source_created_at",
            "date_confidence": 0.7,
        }],
    }


def test_render_context_matches_rag_memory_shape():
    context = render_context(_artifact()["items"])

    assert context.startswith("## Memory 1\n")
    assert "[2024-03-14 | Turn 4 | Mention 1] First milestone" in context
    assert "Evidence: B005 | Date source: source_created_at" in context


def test_context_row_carries_input_ceiling_audit():
    row = context_row(_artifact(), "9_event_ordering_0", "candidate-bank50")

    assert row["query_id"] == "9_event_ordering_0"
    assert row["audit"]["gold_in_input"] is True
    assert row["audit"]["candidate_bank_sha256"] == "bank"
    assert row["audit"]["result_gold_turns"] == [4]


def test_safe_name_keeps_pair_artifacts_self_describing():
    assert _safe_name("7_event_ordering_0") == "7-event-ordering-0"
    assert _safe_name("Relationship Bridge 44") == "relationship-bridge-44"


def test_selected_evidence_context_preserves_contextual_order():
    preflight = {
        "evidence_ledger": [
            {"status": "kept", "contextual_rank": 3, "text": "third"},
            {"status": "dropped", "contextual_rank": 2, "text": "drop"},
            {"status": "kept", "contextual_rank": 1, "text": "first"},
        ]
    }

    assert render_selected_evidence(preflight) == (
        "## Memory 1\nfirst\n\n## Memory 2\nthird"
    )


def _relationship_artifact():
    return {
        "query": (
            "List my academic work and mentorship in order. "
            "Mention only two items."
        ),
        "preflight": {
            "source_evidence_ceiling": {
                "available_source_turns": [10, 22],
                "missing_source_turns": [],
            },
            "evidence_ledger": [
                {
                    "identity": "robert-10",
                    "turn": 10,
                    "created_at": 100.0,
                    "source_doc_id": "session-1",
                    "contextual_score": 0.7,
                    "status": "kept",
                    "text": (
                        "[Turn 10] User: I met Robert for academic mentorship."
                    ),
                },
                {
                    "identity": "robert-20",
                    "turn": 20,
                    "created_at": 200.0,
                    "source_doc_id": "session-2",
                    "contextual_score": 0.8,
                    "status": "kept",
                    "text": "[Turn 20] User: Robert and I are reviewing my draft.",
                },
                {
                    "identity": "bridge-22",
                    "turn": 22,
                    "created_at": 200.0,
                    "source_doc_id": "session-2",
                    "contextual_score": 0.6,
                    "status": "kept",
                    "reason": "context_bridge",
                    "parent_turn": 20,
                    "text": "[Turn 22] User: Our meeting clarified next steps.",
                },
                {
                    "identity": "carla-30",
                    "turn": 30,
                    "created_at": 300.0,
                    "source_doc_id": "session-3",
                    "contextual_score": 0.9,
                    "status": "kept",
                    "text": "[Turn 30] User: I met Carla for feedback.",
                },
            ],
        },
    }


def test_relationship_thread_selects_recurring_person_and_bridge():
    thread = select_relationship_thread(_relationship_artifact())

    assert thread["anchor"] == "robert"
    assert [row["turn"] for row in thread["members"]] == [10, 20, 22]


def test_relationship_thread_renders_one_memory_per_source_conversation():
    context, audit = render_relationship_thread(_relationship_artifact())

    assert context.count("## Memory") == 2
    assert "Relationship thread: Robert" in context
    assert context.index("Turn 10") < context.index("Turn 20")
    assert "Carla" not in context
    assert audit["source_group_count"] == 2
    assert audit["selected_gold_turns"] == [10, 22]
    assert audit["missing_gold_turns"] == []


def test_relationship_thread_representatives_drop_only_same_source_distractors():
    artifact = {
        "query": (
            "List my academic work and mentorship in order. Mention only five items."
        ),
        "preflight": {
            "source_evidence_ceiling": {
                "available_source_turns": [14, 64, 124, 170, 214],
                "missing_source_turns": [],
            },
            "evidence_ledger": [
                {
                    "identity": "meet", "turn": 14, "created_at": 100.0,
                    "source_doc_id": "s0", "contextual_score": 0.55,
                    "status": "kept",
                    "text": "[Turn 14] I met my mentor Robert.",
                },
                {
                    "identity": "essay", "turn": 64, "created_at": 200.0,
                    "source_doc_id": "s1", "contextual_score": 0.55,
                    "status": "kept",
                    "text": "[Turn 64] Robert shared his academic essay.",
                },
                {
                    "identity": "feedback", "turn": 124, "created_at": 300.0,
                    "source_doc_id": "s2", "contextual_score": 0.52,
                    "status": "kept",
                    "text": (
                        "[Turn 124] I am deciding whether to prioritize and focus "
                        "on Robert's review feedback."
                    ),
                },
                {
                    "identity": "confidence", "turn": 156, "created_at": 300.0,
                    "source_doc_id": "s2", "contextual_score": 0.49,
                    "status": "kept",
                    "text": "[Turn 156] Robert's feedback improved my confidence.",
                },
                {
                    "identity": "schedule", "turn": 168, "created_at": 400.0,
                    "source_doc_id": "s3", "contextual_score": 0.59,
                    "status": "kept",
                    "text": "[Turn 168] I prioritized a schedule Robert recommended.",
                },
                {
                    "identity": "choice", "turn": 170, "created_at": 400.0,
                    "source_doc_id": "s3", "contextual_score": 0.57,
                    "status": "kept",
                    "text": (
                        "[Turn 170] I am deciding how to approach and focus on "
                        "Robert's essay advice before the conference paper."
                    ),
                },
                {
                    "identity": "grade", "turn": 212, "created_at": 500.0,
                    "source_doc_id": "s4", "contextual_score": 0.54,
                    "status": "kept",
                    "text": "[Turn 212] Robert and I discussed next steps after my grade.",
                },
                {
                    "identity": "bridge", "turn": 214, "created_at": 500.0,
                    "source_doc_id": "s4", "contextual_score": 0.53,
                    "status": "kept", "reason": "context_bridge",
                    "parent_turn": 212,
                    "text": "[Turn 214] I planned our follow-up meeting.",
                },
            ],
        },
    }

    legacy_context, legacy_audit = render_relationship_thread(artifact)
    context, audit = render_relationship_thread(
        artifact, representative_per_source=True
    )

    assert legacy_context.count("[Turn") == 8
    assert legacy_audit["selected_row_count"] == 8
    assert context.count("[Turn") == 6
    assert audit["selected_turns"] == [14, 64, 124, 170, 212, 214]
    assert audit["selected_gold_turns"] == [14, 64, 124, 170, 214]
    assert audit["missing_gold_turns"] == []
    assert audit["dropped_turns"] == [156, 168]


def test_family_support_stages_choose_one_active_event_per_conversation():
    artifact = {
        "query": "How did my family support me across our conversations?",
        "preflight": {
            "source_evidence_ceiling": {
                "available_source_turns": [12, 30],
                "missing_source_turns": [],
            },
            "evidence_ledger": [
                {
                    "identity": "intro",
                    "turn": 10,
                    "created_at": 100.0,
                    "source_doc_id": "session-1",
                    "contextual_score": 0.6,
                    "status": "kept",
                    "text": "[Turn 10] User: My mom, Wendy, is supportive.",
                },
                {
                    "identity": "specific",
                    "turn": 12,
                    "created_at": 100.0,
                    "source_doc_id": "session-1",
                    "contextual_score": 0.9,
                    "status": "kept",
                    "text": "[Turn 12] User: Wendy encouraged my application.",
                },
                {
                    "identity": "partner",
                    "turn": 30,
                    "created_at": 200.0,
                    "source_doc_id": "session-2",
                    "contextual_score": 0.8,
                    "status": "kept",
                    "text": "[Turn 30] User: I am dating Tanya; she helped rehearse.",
                },
                {
                    "identity": "incidental",
                    "turn": 32,
                    "created_at": 200.0,
                    "source_doc_id": "session-2",
                    "contextual_score": 0.99,
                    "status": "kept",
                    "text": "[Turn 32] User: Tanya booked a flight.",
                },
            ],
        },
    }

    context, audit = render_relationship_support_stages(artifact)

    assert context.count("## Memory") == 2
    assert "Turn 12" in context
    assert "Turn 30" in context
    assert "Turn 10" not in context
    assert "Turn 32" not in context
    assert audit["selected_anchors"] == ["tanya", "wendy"]
    assert audit["selected_gold_turns"] == [12, 30]
    assert audit["representative_selection_identity_unchanged"] is True
    assert (
        audit["representative_candidate_identities"]
        == audit["legacy_selected_identities"]
    )
