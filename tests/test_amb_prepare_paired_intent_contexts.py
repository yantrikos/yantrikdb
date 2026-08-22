import importlib.util
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "amb"
    / "prepare_paired_intent_contexts.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "amb_prepare_paired_intent_contexts", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _row(query_id: str, query: str, category: str, context: str) -> dict:
    return {
        "query_id": query_id,
        "query": query,
        "context": context,
        "meta": {"question_category": category},
    }


def test_locked_intent_rule_excludes_exposed_categories():
    assert _MODULE.is_independent_count_set_row(
        _row("q1", "How many books?", "knowledge_update", "A")
    )
    assert _MODULE.is_independent_count_set_row(
        _row("q2", "How much progress?", "instruction_following", "A")
    )
    assert not _MODULE.is_independent_count_set_row(
        _row("q3", "How many days?", "temporal_reasoning", "A")
    )
    assert not _MODULE.is_independent_count_set_row(
        _row("q4", "How many roles?", "multi_session_reasoning", "A")
    )
    assert not _MODULE.is_independent_count_set_row(
        _row("q5", "Which books?", "knowledge_update", "A")
    )


def test_select_common_rows_preserves_arm_a_order_without_scores_or_gold():
    rows_a = [
        _row("q2", "What two events?", "knowledge_update", "A2"),
        _row("skip", "How many days?", "temporal_reasoning", "skip"),
        _row("q1", "How many books?", "information_extraction", "A1"),
    ]
    rows_b = [
        {"query_id": "q1", "context": "B1"},
        {"query_id": "q2", "context": "B2"},
    ]

    arm_a, arm_b = _MODULE.select_common_rows(rows_a, rows_b)

    assert [row["query_id"] for row in arm_a] == ["q2", "q1"]
    assert [row["context"] for row in arm_b] == ["B2", "B1"]
    assert set(arm_a[0]) == {"query_id", "context"}
