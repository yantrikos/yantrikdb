import gzip
import importlib.util
import json
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "amb"
    / "audit_knowledge_update_gold.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "amb_audit_knowledge_update_gold", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_value_tokens_ignores_bare_years_but_preserves_dates_and_counts():
    assert _MODULE.value_tokens("May 10, 2024 after 3 days, usually 5-7 months") == {
        "may 10",
        "3 days",
        "5-7 months",
    }


def test_parse_document_turns_handles_dated_plain_and_continuation_headers():
    content = """[March-15-2024 | Turn 4] User: first
[Turn 5] Assistant: reply
[Turn 6] (cont.) User: continued"""

    assert _MODULE.parse_document_turns(content) == [
        {"id": 4, "role": "user", "content": "first"},
        {"id": 5, "role": "assistant", "content": "reply"},
        {"id": 6, "role": "user", "content": "continued"},
    ]


def test_load_conversations_merges_flattened_gzip_turn_fragments(tmp_path):
    path = tmp_path / "documents.json.gz"
    rows = [
        {
            "user_id": "7",
            "content": "[Turn 2] User: first half",
        },
        {
            "user_id": "7",
            "content": "[Turn 2] (cont.) User: second half\n"
            "[Turn 3] Assistant: reply",
        },
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(rows, handle)

    conversations = _MODULE.load_conversations(path)

    assert conversations["7"]["chat"] == [
        {"id": 2, "role": "user", "content": "first half second half"},
        {"id": 3, "role": "assistant", "content": "reply"},
    ]
