import importlib.util
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "amb"
    / "role_collection_probe.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "amb_role_collection_probe", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_normalize_collections_resolves_children_and_reports_membership_errors():
    leaf_handles = {
        "H001": {"evidence_ids": ["A001", "A002"]},
        "H002": {"evidence_ids": ["A002", "A003"]},
    }
    values = [
        {
            "label": "Mentors",
            "anchor_entities": ["Bryan", "Bryan"],
            "summary": "Industry mentors.",
            "member_handle_ids": ["h001", "H999"],
        },
        {
            "label": "Other mentors",
            "anchor_entities": [],
            "summary": "A duplicate assignment.",
            "member_handle_ids": ["H001", "H002"],
        },
    ]

    collections, invalid, duplicated = _MODULE.normalize_collections(
        values, leaf_handles
    )

    assert invalid == ["H999"]
    assert duplicated == ["H001"]
    assert collections[0]["anchor_entities"] == ["Bryan"]
    assert collections[1]["evidence_ids"] == ["A001", "A002", "A003"]


def test_append_unassigned_singletons_preserves_omitted_leaf():
    collections = [
        {
            "label": "Mentors",
            "member_handle_ids": ["H001"],
            "evidence_ids": ["A001"],
        }
    ]
    leaf_handles = {
        "H001": {"evidence_ids": ["A001"]},
        "H002": {
            "label": "Formatting preferences",
            "summary": "Stable output preferences.",
            "anchor_entities": [],
            "evidence_ids": ["A002"],
        },
    }

    repaired = _MODULE.append_unassigned_singletons(collections, leaf_handles)

    assert repaired == ["H002"]
    assert collections[-1] == {
        "label": "Formatting preferences",
        "anchor_entities": [],
        "summary": "Stable output preferences.",
        "member_handle_ids": ["H002"],
        "evidence_ids": ["A002"],
        "fallback_singleton": True,
    }
