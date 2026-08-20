import importlib.util
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "amb"
    / "build_role_scaffold_contexts.py"
)
_SPEC = importlib.util.spec_from_file_location("amb_role_scaffold", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _words(text):
    return len(text.split())


def _truncate(text, limit):
    return " ".join(text.split()[:limit])


def test_scaffold_context_selects_before_chronological_presentation():
    users = [
        "[Speaker: User | March 02, 2024 | Turn 2] second evidence",
        "[Speaker: User | March 01, 2024 | Turn 0] first evidence",
    ]
    assistants = {
        1: "[March-01-2024 | Turn 1] Assistant: first explanation",
        3: "[March-02-2024 | Turn 3] Assistant: second explanation",
    }

    documents, trace = _MODULE.build_scaffold_documents(
        users,
        assistants,
        token_budget=100,
        assistant_tokens=10,
        max_user_documents=2,
        max_assistant_scaffolds=2,
        token_counter=_words,
        truncate_assistant=_truncate,
    )

    assert "Turn 0" in documents[0]
    assert "Turn 2" in documents[1]
    assert trace["paired_assistant_scaffolds"] == 2


def test_scaffold_context_keeps_the_shared_budget_bounded():
    users = [
        "[Speaker: User | March 01, 2024 | Turn 0] one two",
        "[Speaker: User | March 02, 2024 | Turn 2] three four",
    ]
    assistants = {
        1: "[Turn 1] Assistant: alpha beta gamma delta",
        3: "[Turn 3] Assistant: epsilon zeta eta theta",
    }

    documents, trace = _MODULE.build_scaffold_documents(
        users,
        assistants,
        token_budget=22,
        assistant_tokens=3,
        max_user_documents=2,
        max_assistant_scaffolds=2,
        token_counter=_words,
        truncate_assistant=_truncate,
    )

    assert len(documents) == 1
    assert trace["context_tokens"] <= 22
    assert "Adjacent assistant scaffold" in documents[0]


def test_assistant_turn_parser_does_not_emit_split_captures():
    document = type(
        "Document",
        (),
        {
            "user_id": "1",
            "content": (
                "[March-01-2024 | Turn 0] User: hello\n"
                "[March-01-2024 | Turn 1] Assistant: response"
            ),
        },
    )()

    assistants = _MODULE._assistant_turns([document])

    assert assistants == {
        ("1", 1): "[March-01-2024 | Turn 1] Assistant: response"
    }


def test_scaffold_cap_leaves_budget_for_more_user_evidence():
    users = [
        "[Speaker: User | March 01, 2024 | Turn 0] first evidence",
        "[Speaker: User | March 02, 2024 | Turn 2] second evidence",
    ]
    assistants = {
        1: "[Turn 1] Assistant: first explanation",
        3: "[Turn 3] Assistant: second explanation",
    }

    documents, trace = _MODULE.build_scaffold_documents(
        users,
        assistants,
        token_budget=100,
        assistant_tokens=10,
        max_user_documents=2,
        max_assistant_scaffolds=1,
        token_counter=_words,
        truncate_assistant=_truncate,
    )

    assert len(documents) == 2
    assert sum("Adjacent assistant scaffold" in value for value in documents) == 1
    assert trace["assistant_scaffold_cap"] == 1
