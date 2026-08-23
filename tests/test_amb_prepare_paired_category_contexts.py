import importlib.util
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "amb"
    / "prepare_paired_category_contexts.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "amb_prepare_paired_category_contexts", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_select_common_rows_preserves_arm_a_order_and_category():
    rows_a = [
        {
            "query_id": "2_multi_session_reasoning_0",
            "context": "A2",
            "meta": {"question_category": "multi_session_reasoning"},
        },
        {
            "query_id": "1_temporal_reasoning_0",
            "context": "skip",
            "meta": {"question_category": "temporal_reasoning"},
        },
        {
            "query_id": "1_multi_session_reasoning_0",
            "context": "A1",
            "meta": {"question_category": "multi_session_reasoning"},
        },
    ]
    rows_b = [
        {"query_id": "1_multi_session_reasoning_0", "context": "B1"},
        {"query_id": "2_multi_session_reasoning_0", "context": "B2"},
    ]

    arm_a, arm_b = _MODULE.select_common_rows(
        rows_a, rows_b, "multi_session_reasoning"
    )

    assert [row["query_id"] for row in arm_a] == [
        "2_multi_session_reasoning_0",
        "1_multi_session_reasoning_0",
    ]
    assert [row["context"] for row in arm_b] == ["B2", "B1"]


def test_select_common_rows_accepts_full_cohort():
    rows_a = [
        {
            "query_id": "1_instruction_following_0",
            "context": "A1",
            "meta": {"question_category": "instruction_following"},
        },
        {
            "query_id": "1_temporal_reasoning_0",
            "context": "A2",
            "meta": {"question_category": "temporal_reasoning"},
        },
    ]
    rows_b = [
        {"query_id": "1_instruction_following_0", "context": "B1"},
        {"query_id": "1_temporal_reasoning_0", "context": "B2"},
    ]

    arm_a, arm_b = _MODULE.select_common_rows(rows_a, rows_b, "all")

    assert [row["query_id"] for row in arm_a] == [
        "1_instruction_following_0",
        "1_temporal_reasoning_0",
    ]
    assert [row["context"] for row in arm_b] == ["B1", "B2"]
