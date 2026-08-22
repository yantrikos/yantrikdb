import importlib.util
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "amb"
    / "two_stage_handle_probe.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "amb_two_stage_handle_probe", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_index_handles_drops_unknown_children_and_empty_handles():
    artifact = {
        "input_items": [{"id": "A0001", "turn": 4, "text": "Advice"}],
        "handles": [
            {"label": "Mentor", "evidence_ids": ["A0001", "A9999"]},
            {"label": "Empty", "evidence_ids": ["A9999"]},
        ],
    }

    handles, atomics = _MODULE.index_handles(artifact)

    assert list(handles) == ["H001"]
    assert handles["H001"]["evidence_ids"] == ["A0001"]
    assert list(atomics) == ["A0001"]


def test_hydrate_handles_preserves_overlapping_group_membership():
    atomics = {"A0001": {"id": "A0001", "turn": 4, "text": "Advice"}}
    handles = {
        "H001": {"label": "Bryan", "evidence_ids": ["A0001"]},
        "H002": {"label": "Mentors", "evidence_ids": ["A0001"]},
    }

    hydrated = _MODULE.hydrate_handles(
        ["H001", "H002"], handles, atomics
    )

    assert hydrated == [
        {
            "id": "A0001",
            "turn": 4,
            "text": "Advice",
            "handle_ids": ["H001", "H002"],
            "handle_labels": ["Bryan", "Mentors"],
        }
    ]
