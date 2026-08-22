import importlib.util
from collections import Counter
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "amb"
    / "reorder_speaker_first_contexts.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "amb_reorder_speaker_first_contexts", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _block(number: int, body: str) -> str:
    return f"## Memory {number}\n{body}\n\n"


def test_reorder_context_is_stable_and_preserves_exact_blocks():
    assistant = _block(1, "[Turn 1] Assistant: Suggested fact")
    unknown = _block(2, "No speaker provenance")
    user_one = _block(3, "[Turn 3] User: First fact")
    user_two = _block(4, "[Speaker: User | Turn 4] User: Second fact")
    context = assistant + unknown + user_one + user_two

    reordered, audit = _MODULE.reorder_context(context)

    assert reordered == user_one + user_two + unknown + assistant
    assert Counter(_MODULE.split_memory_blocks(reordered)) == Counter(
        _MODULE.split_memory_blocks(context)
    )
    assert len(reordered) == len(context)
    assert audit == {
        "blocks": 4,
        "user_blocks": 2,
        "unknown_blocks": 1,
        "assistant_blocks": 1,
        "presentation_reordered": True,
    }


def test_speaker_bucket_prefers_explicit_provenance():
    block = _block(
        1,
        "[said by the ASSISTANT (a suggestion)] [Turn 1] User: Quoted user text",
    )

    assert _MODULE.speaker_bucket(block) == "assistant"


def test_split_accepts_provenance_on_memory_header_line():
    context = (
        "## Memory 1 [said by the ASSISTANT]\nA\n\n"
        "## Memory 2 [said by the USER]\nU"
    )

    blocks = _MODULE.split_memory_blocks(context)

    assert len(blocks) == 2
    assert blocks[1].startswith("## Memory 2")


def test_reorder_preserves_boundaries_when_last_block_has_no_newline():
    assistant = "## Memory 1\n[Turn 1] Assistant: A\n\n"
    user = "## Memory 2\n[Turn 2] User: U"

    reordered, _audit = _MODULE.reorder_context(assistant + user)

    assert reordered == user + "\n\n" + assistant.rstrip()


def test_transform_changes_only_context_and_adds_audit_metadata():
    context = _block(1, "[Turn 1] Assistant: A") + _block(
        2, "[Turn 2] User: U"
    )
    payload = {"run": "frozen", "results": [{"query_id": "q1", "context": context}]}

    output = _MODULE.transform(payload)

    assert payload["results"][0]["context"] == context
    assert output["run"] == "frozen"
    assert output["results"][0]["query_id"] == "q1"
    assert output["results"][0]["context"].startswith("## Memory 2")
    assert output["artifact_transform"]["selection_changed"] is False
    assert output["artifact_transform"]["audit"]["reordered_rows"] == 1
