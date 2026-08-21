import pytest

from benchmarks.amb.build_hybrid_contexts import (
    build_hybrid_rows,
    split_memory_bodies,
)


def test_split_memory_bodies_removes_numbered_headers():
    context = "## Memory 1\nraw one\n\n## Memory 2 [user]\nraw two"

    assert split_memory_bodies(context) == ["raw one", "raw two"]


def test_build_hybrid_rows_bounds_and_orders_both_lanes():
    raw = [
        {
            "query_id": "q",
            "context": "## Memory 1\nr1\n\n## Memory 2\nr2",
        }
    ]
    derived = [
        {
            "query_id": "q",
            "documents": ["d1", "d2"],
            "selection": {"selection_mode": "derived"},
        }
    ]

    rows = build_hybrid_rows(raw, derived, raw_limit=1, derived_limit=1)

    assert rows[0]["documents"] == ["d1", "r1"]
    assert rows[0]["context"] == "## Memory 1\nd1\n\n## Memory 2\nr1"
    assert rows[0]["hybrid"] == {
        "derived_documents": 1,
        "raw_documents": 1,
        "ordering": "derived_then_raw",
    }


def test_build_hybrid_rows_requires_matching_raw_context():
    with pytest.raises(ValueError, match="missing"):
        build_hybrid_rows([], [{"query_id": "q", "documents": ["d"]}], 1, 1)


def test_build_hybrid_rows_accepts_frozen_derived_context_without_documents():
    raw = [{"query_id": "q", "context": "## Memory 1\nr1"}]
    derived = [{"query_id": "q", "context": "## Memory 1\nd1"}]

    rows = build_hybrid_rows(raw, derived, raw_limit=1, derived_limit=1)

    assert rows[0]["documents"] == ["d1", "r1"]
