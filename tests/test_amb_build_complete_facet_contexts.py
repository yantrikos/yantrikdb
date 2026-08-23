from benchmarks.amb.build_complete_facet_contexts import (
    MAX_FACET_TOKENS,
    build_context_rows,
    overlay_control_contexts,
    select_facets,
    validate_full400_preflight,
)


def _facet(rid, text, turn, first):
    return {
        "rid": rid,
        "text": text,
        "turn": turn,
        "first_mention_at": first,
        "source_rids": [f"source-{rid}"],
    }


FACETS = [
    _facet("a", "Always include versions when I ask about libraries.", 2, 2.0),
    _facet("b", "Always cite sources when I ask about research papers.", 4, 4.0),
    _facet("c", "Always use Celsius when I ask about weather.", 6, 6.0),
]


def _tokens(text):
    return len(text.split())


def test_control_context_join_is_exact_and_keeps_metadata():
    rows = [
        {"query_id": "q1", "query": "first", "context": "stale"},
        {"query_id": "q2", "query": "second", "context": "stale"},
    ]
    joined = overlay_control_contexts(
        rows,
        [
            {"query_id": "q1", "context": "control one"},
            {"query_id": "q2", "context": "control two"},
        ],
    )
    assert [row["context"] for row in joined] == ["control one", "control two"]
    assert [row["query"] for row in joined] == ["first", "second"]
    assert rows[0]["context"] == "stale"


def test_control_context_join_rejects_order_drift():
    rows = [{"query_id": "q1"}, {"query_id": "q2"}]
    try:
        overlay_control_contexts(
            rows, [{"query_id": "q2"}, {"query_id": "q1"}]
        )
    except ValueError as error:
        assert "query order" in str(error)
    else:
        raise AssertionError("query order drift did not abort")


def test_every_query_gets_complete_first_mention_order():
    selected = select_facets(list(reversed(FACETS)))
    assert [row["rid"] for row in selected] == ["a", "b", "c"]


def test_additive_composition_keeps_ordinary_context_and_target():
    row = {
        "query_id": "q1",
        "query": "Which libraries are used?",
        "context": "## Memory 1\noriginal first\n\n## Memory 2\noriginal second\n",
        "meta": {
            "conversation_id": "unit",
            "instruction_being_tested": FACETS[0]["text"],
        },
    }
    treatment, audit = build_context_rows(
        [row], {"unit": {"facets": FACETS, "omitted": 0}}, _tokens
    )
    context = treatment["results"][0]["context"]
    assert context.endswith(row["context"])
    assert audit["complete_lane_rows"] == 1
    assert audit["max_facets_per_row"] == 3
    assert audit["ordinary_contexts_exact"] == 1
    assert audit["instruction_targets_retained"] == 1
    assert audit["max_facet_tokens_observed"] <= MAX_FACET_TOKENS


def test_incomplete_lane_and_failed_full400_gate_abort():
    row = {
        "query_id": "q1",
        "query": "Which libraries are used?",
        "context": "## Memory 1\noriginal\n",
        "meta": {"conversation_id": "unit"},
    }
    try:
        build_context_rows(
            [row], {"unit": {"facets": FACETS, "omitted": 1}}, _tokens
        )
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("incomplete facet inventory did not abort")
    try:
        validate_full400_preflight(
            {
                "rows": 1,
                "instruction_target_rows": 0,
                "instruction_targets_retained": 0,
                "complete_lane_rows": 1,
                "min_facets_per_row": 5,
                "max_facets_per_row": 5,
                "ordinary_contexts_exact": 1,
                "max_facet_tokens_observed": 10,
            }
        )
    except RuntimeError as error:
        assert "pre-call facet gate failed" in str(error)
    else:
        raise AssertionError("invalid full-400 audit did not abort")
