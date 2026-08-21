import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "amb"
    / "probe_summarization_context_coverage.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "amb_probe_summarization_context_coverage", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_select_rows_uses_category_and_frozen_score_only():
    payload = {
        "results": [
            {"query_id": "a", "score": 0.4, "context": "A", "meta": {"question_category": "summarization"}},
            {"query_id": "b", "score": 0.5, "context": "B", "meta": {"question_category": "summarization"}},
            {"query_id": "c", "score": 0.0, "context": "C", "meta": {"question_category": "temporal_reasoning"}},
        ]
    }

    assert [row["query_id"] for row in _MODULE.select_rows(payload, 0.4)] == ["a"]


def test_rubric_items_remove_benchmark_prefix():
    row = {"meta": {"rubric": ["LLM response should contain: a fact", "Other"]}}

    assert _MODULE.rubric_items(row) == ["a fact", "Other"]


def test_validate_verdicts_requires_verified_quotes_and_complete_indices():
    verdicts = _MODULE.validate_verdicts(
        [
            {
                "index": 1,
                "supported": True,
                "quotes": ["budget increased to $75", "after the review"],
            },
            {"index": 2, "supported": False, "quotes": []},
        ],
        2,
        "The budget increased to $75 after the review.",
    )

    assert verdicts[0]["quotes_verified"] == [True, True]
    assert verdicts[1]["quotes_verified"] == []

    with pytest.raises(ValueError, match="unverifiable"):
        _MODULE.validate_verdicts(
            [{"index": 1, "supported": True, "quotes": ["invented fact"]}],
            1,
            "real context",
        )


def test_quote_validation_ignores_typographic_punctuation_only():
    verdicts = _MODULE.validate_verdicts(
        [
            {
                "index": 1,
                "supported": True,
                "quotes": ["Tanya's advice - revise the statement"],
            }
        ],
        1,
        "Tanya’s advice — revise the statement.",
    )

    assert verdicts[0]["quotes_verified"] == [True]
