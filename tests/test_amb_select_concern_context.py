import importlib.util
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "amb"
    / "select_concern_context.py"
)
_SPEC = importlib.util.spec_from_file_location("amb_select_concern_context", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_requested_count_understands_event_ordering_wording():
    assert _MODULE.requested_count("Mention ONLY and ONLY five items.") == 5
    assert _MODULE.requested_count("Give exactly 7 changes") == 7
    assert _MODULE.requested_count("What changed?") is None


def test_normalize_selected_ids_rejects_unknowns_and_duplicates():
    raw = {"selected_ids": ["c002", "C999", "C002", "C001", "C003"]}
    assert _MODULE.normalize_selected_ids(raw, {"C001", "C002", "C003"}, 2) == [
        "C002",
        "C001",
    ]


def test_chronological_key_uses_turn_as_a_tie_breaker():
    later = {"rid": "b", "created_at": 10, "metadata": {"first_mention_turn": 4}}
    earlier = {"rid": "a", "created_at": 10, "metadata": {"first_mention_turn": 2}}
    assert sorted([later, earlier], key=_MODULE.chronological_key) == [earlier, later]
