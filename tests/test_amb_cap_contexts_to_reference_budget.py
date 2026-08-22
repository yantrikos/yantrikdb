import pytest

from benchmarks.amb.cap_contexts_to_reference_budget import cap_context, transform


def test_cap_context_keeps_a_whole_ordered_prefix_within_budget():
    context = (
        "## Memory 1\nalpha\n\n"
        "## Memory 2\nbeta beta\n\n"
        "## Memory 3\ngamma gamma gamma"
    )

    capped, audit = cap_context(context, 46, len)

    assert capped == "## Memory 1\nalpha\n\n## Memory 2\nbeta beta\n\n"
    assert audit == {
        "reference_token_budget": 46,
        "treatment_tokens_before": len(context),
        "treatment_tokens_after": len(capped),
        "blocks_before": 3,
        "blocks_after": 2,
        "prefix_preserved": True,
    }


def test_transform_matches_reference_order_and_rejects_missing_queries():
    reference = [
        {"query_id": "q2", "context": "x" * 25},
        {"query_id": "q1", "context": "x" * 25},
    ]
    treatment = [
        {"query_id": "q1", "context": "## Memory 1\na\n## Memory 2\nb"},
        {"query_id": "q2", "context": "## Memory 1\nc\n## Memory 2\nd"},
    ]

    output = transform(reference, treatment, len)

    assert [row["query_id"] for row in output["results"]] == ["q2", "q1"]
    assert output["artifact_transform"]["treatment_within_reference_budget"] is True
    assert output["artifact_transform"]["external_calls"] == 0

    with pytest.raises(ValueError, match="missing query"):
        transform(reference, treatment[:1], len)
