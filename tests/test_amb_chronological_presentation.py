from collections import Counter

import pytest

from benchmarks.amb.chronological_presentation import (
    chronological_document_key,
    chronological_hit_key,
)
from benchmarks.amb.reorder_role_aware_contexts import (
    audit_transform,
    reorder_row,
    transform,
)


def test_hit_order_uses_event_time_then_turn_then_chunk():
    hits = [
        {
            "rid": "late",
            "created_at": 20,
            "metadata": {"turn_id": 2, "chunk_idx": 0},
        },
        {
            "rid": "second-piece",
            "created_at": 10,
            "metadata": {"turn_id": 4, "chunk_idx": 2},
        },
        {
            "rid": "first-piece",
            "created_at": 10,
            "metadata": {"turn_id": 4, "chunk_idx": 1},
        },
        {"rid": "unknown", "created_at": None, "metadata": {}},
    ]

    assert [hit["rid"] for hit in sorted(hits, key=chronological_hit_key)] == [
        "first-piece",
        "second-piece",
        "late",
        "unknown",
    ]


def test_frozen_row_reorder_preserves_selection_and_evidence_multiset():
    documents = [
        "[Speaker: User | April 04, 2024 | Turn 8] late",
        "[Speaker: User | March 14, 2024 | Turn 4] second",
        "[Speaker: User | March 14, 2024 | Turn 2] first",
    ]
    results = [{"rid": value} for value in ("late", "second", "first")]
    row = {
        "query_id": "q1",
        "documents": documents,
        "selection": {"results": results},
        "context": "old",
    }

    output = reorder_row(row)

    assert output["selection"]["results"] == results
    assert [item["rid"] for item in output["selection"]["presented_results"]] == [
        "first",
        "second",
        "late",
    ]
    assert Counter(output["documents"]) == Counter(documents)
    assert output["documents"][0].endswith("first")
    assert output["selection"]["presentation_reordered"] is True
    assert output["context"].startswith("## Memory 1\n")


def test_frozen_row_reorder_rejects_unaligned_trace():
    row = {
        "query_id": "q1",
        "documents": ["[Speaker: User | March 14, 2024 | Turn 2] first"],
        "selection": {"results": []},
    }
    row["selection"]["results"] = [{"rid": "a"}, {"rid": "b"}]

    with pytest.raises(ValueError, match="1 documents but 2 selection results"):
        reorder_row(row)


def test_unknown_display_prefix_sorts_after_dated_documents():
    dated = "[Speaker: User | March 14, 2024 | Turn 2] first"
    assert chronological_document_key(dated, 1) < chronological_document_key(
        "unparseable", 0
    )


def test_artifact_audit_proves_presentation_only_change():
    documents = [
        "[Speaker: User | April 04, 2024 | Turn 8] late",
        "[Speaker: User | March 14, 2024 | Turn 2] first",
    ]
    source = {
        "results": [{
            "query_id": "q1",
            "documents": documents,
            "selection": {"results": [{"rid": "late"}, {"rid": "first"}]},
            "context": "\n\n".join(
                f"## Memory {index}\n{content}"
                for index, content in enumerate(documents, 1)
            ),
        }]
    }

    report = audit_transform(source, transform(source))

    assert report == {
        "rows": 1,
        "documents": 2,
        "reordered_rows": 1,
        "unknown_prefixes": 0,
        "selection_changed": False,
        "context_lengths_changed": False,
    }
