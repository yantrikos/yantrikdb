import gzip
import importlib.util
import json
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "amb"
    / "audit_summarization_lexical_funnel.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "amb_audit_summarization_lexical_funnel", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_tokens_keep_distinctive_values_and_drop_boilerplate():
    assert _MODULE.tokens("You improved accuracy to 92% with Carla") == {
        "improved",
        "accuracy",
        "92%",
        "carla",
    }


def test_analyze_row_normalizes_context_and_answer_to_source_tokens():
    row = {
        "query_id": "q",
        "score": 0.2,
        "context": "Carla improved accuracy.",
        "answer": "Carla helped.",
        "meta": {
            "rubric": [
                "LLM response should contain: Carla improved accuracy to 92%"
            ]
        },
    }

    result = _MODULE.analyze_row(row, "Carla improved accuracy to 92%.")

    item = result["items"][0]
    assert item["source_normalized_retrieval"] == 0.75
    assert item["source_normalized_answer"] == 0.25
    assert item["source_tokens_missing_from_context"] == ["92%"]


def test_load_source_documents_groups_gzip_rows(tmp_path):
    path = tmp_path / "documents.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(
            [
                {"user_id": "1", "content": "first"},
                {"user_id": "1", "content": "second"},
                {"user_id": "2", "content": "other"},
            ],
            handle,
        )

    assert _MODULE.load_source_documents(path) == {
        "1": "first\nsecond",
        "2": "other",
    }


def test_replace_contexts_changes_only_context():
    rows = [{"query_id": "q", "context": "old", "score": 0.2}]

    replaced = _MODULE.replace_contexts(
        rows, {"results": [{"query_id": "q", "context": "new"}]}
    )

    assert replaced == [{"query_id": "q", "context": "new", "score": 0.2}]
    assert rows[0]["context"] == "old"
