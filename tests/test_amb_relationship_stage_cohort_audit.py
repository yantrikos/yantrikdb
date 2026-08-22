from benchmarks.amb.relationship_stage_cohort_audit import aggregate, audit_query


def _row(query_id: str, query: str, evidence: list[dict], gold: list[int]) -> dict:
    return {
        "query_id": query_id,
        "query": query,
        "source_turns": gold,
        "candidate_bank_count": len(evidence),
        "bank_ceiling": {"available_source_turns": gold},
        "preflight": {"evidence_ledger": evidence},
    }


def _evidence(
    identity: str, turn: int, text: str, source_doc_id: str, score: float
) -> dict:
    return {
        "identity": identity,
        "turn": turn,
        "text": text,
        "source_doc_id": source_doc_id,
        "created_at": float(turn),
        "contextual_score": score,
        "status": "kept",
    }


def test_explicit_mentorship_fires_but_professional_connections_abstains():
    evidence = [
        _evidence("a", 10, "I met Robert for mentorship advice.", "s1", 0.8),
        _evidence("b", 20, "Robert and I are reviewing a draft.", "s2", 0.7),
    ]

    mentorship = audit_query(_row(
        "mentor", "List my academic work and mentorship in order.", evidence, [10]
    ))
    connections = audit_query(_row(
        "connections", "List my professional connections and preparation.",
        evidence, [10],
    ))

    assert mentorship["route"] == "relationship_thread"
    assert mentorship["selected_source_turns"] == [10]
    assert connections["route"] == "abstain"
    assert connections["source_turn_delta"] == 0


def test_family_support_route_selects_one_active_stage_per_source():
    evidence = [
        _evidence("a", 10, "My mom, Wendy, is supportive.", "s1", 0.6),
        _evidence("b", 12, "Wendy encouraged my application.", "s1", 0.9),
        _evidence("c", 20, "I am dating Tanya; she helped rehearse.", "s2", 0.8),
    ]
    result = audit_query(_row(
        "family", "How did my family support me across conversations?",
        evidence, [12, 20],
    ))

    assert result["route"] == "relationship_support_stages"
    assert result["selected_source_turns"] == [12, 20]
    assert result["source_turn_delta"] == 0


def test_representative_audit_preserves_gold_and_reduces_same_source_rows():
    evidence = [
        _evidence("setup", 10, "Robert gave academic feedback.", "s1", 0.9),
        _evidence(
            "decision",
            12,
            "I decided whether to focus on Robert's academic feedback.",
            "s1",
            0.8,
        ),
        _evidence("followup", 20, "Robert reviewed my academic draft.", "s2", 0.7),
    ]

    result = audit_query(
        _row(
            "mentor",
            "List my academic work and mentorship in order.",
            evidence,
            [12, 20],
        ),
        relationship_representatives=True,
    )

    assert result["route"] == "relationship_thread"
    assert result["selected_row_count"] == 2
    assert result["selected_source_turns"] == [12, 20]
    assert result["source_turn_delta"] == 0
    assert result["selector"]["dropped_turns"] == [10]


def test_aggregate_reports_fire_abstention_retention_and_reduction():
    rows = [
        {
            "route": "relationship_thread",
            "bank_row_count": 40,
            "selected_row_count": 5,
            "bank_available_source_turns": [1, 2],
            "selected_source_turns": [1, 2],
            "effective_available_source_turns": [1, 2],
            "source_turn_delta": 0,
        },
        {
            "route": "abstain",
            "bank_row_count": 40,
            "selected_row_count": 40,
            "bank_available_source_turns": [3],
            "selected_source_turns": [],
            "effective_available_source_turns": [3],
            "source_turn_delta": 0,
        },
    ]

    summary = aggregate(rows)

    assert summary["fire_rate"] == 0.5
    assert summary["abstention_rate"] == 0.5
    assert summary["fired_gold_retention"] == 1.0
    assert summary["fired_row_reduction"] == 0.875
    assert summary["negative_query_count"] == 0
