import json

import pytest

from benchmarks.amb.global_organizer_probe import (
    _apply_assignments,
    _cap_recovered_handles,
    _capacity_constrained_assignments,
    _complete_singleton_handles,
    _discovered_handle_count,
    _drop_invalid_evidence_ids,
    _normalize_handles,
    _recover_truncated_handles,
    _require_top_level_key,
)


def test_recover_truncated_handles_keeps_only_complete_array_elements():
    complete = {
        "label": "Writing",
        "anchor_entities": ["Essay"],
        "summary": "The writing trajectory.",
        "evidence_ids": ["A0001", "A0002", "A0003"],
        "selection_rationale": "One coherent project.",
    }
    raw = {
        "done_reason": "length",
        "message": {
            "content": (
                '{"handles":['
                + json.dumps(complete)
                + ","
                + json.dumps({**complete, "label": "Revision"})
                + ',{"label":"truncated'
            )
        },
    }

    recovered = _recover_truncated_handles(raw)

    assert [handle["label"] for handle in recovered] == ["Writing", "Revision"]


def test_recover_truncated_handles_rejects_non_length_responses():
    raw = {
        "done_reason": "stop",
        "message": {"content": '{"handles":[{"label":"partial"}'},
    }

    assert _recover_truncated_handles(raw) == []


def test_cap_recovered_handles_maximizes_distinct_evidence_then_keeps_order():
    handles = [
        {"label": "narrow", "evidence_ids": ["A1"]},
        {"label": "broad", "evidence_ids": ["A1", "A2"]},
        {"label": "rare", "evidence_ids": ["A3"]},
        {"label": "repeat", "evidence_ids": ["A1"]},
    ]

    selected, trace = _cap_recovered_handles(
        handles, {"A1", "A2", "A3"}, limit=2
    )

    assert [handle["label"] for handle in selected] == ["broad", "rare"]
    assert trace == {
        "pre_cap_handle_count": 4,
        "post_cap_handle_count": 2,
        "covered_evidence_count": 3,
    }


def test_normalize_handles_removes_evidence_ids_from_anchor_entities():
    known = {"A0001": {}, "A0002": {}}
    handles, misplaced = _normalize_handles(
        [
            {
                "label": "Carla timeline",
                "anchor_entities": ["A0001", "Carla", "Carla", " "],
                "summary": "Editing work with Carla",
                "evidence_ids": ["A0001", "A0001", "A0002"],
                "selection_rationale": "One relationship arc",
            }
        ],
        known,
    )

    assert handles[0]["anchor_entities"] == ["Carla"]
    assert handles[0]["evidence_ids"] == ["A0001", "A0002"]
    assert misplaced == ["A0001"]


def test_require_top_level_key_rejects_nested_partial_json():
    with pytest.raises(ValueError, match="top-level 'handles'"):
        _require_top_level_key({"label": "partial handle"}, "handles")


def test_apply_assignments_is_exhaustive_only_for_valid_references():
    handles = [
        {"evidence_ids": ["A0001"]},
        {"evidence_ids": ["A0002"]},
    ]
    assigned, invalid = _apply_assignments(
        handles,
        [
            {"id": "A0003", "handle_numbers": [1, 2]},
            {"id": "A0004", "handle_numbers": [3]},
            {"id": "A9999", "handle_numbers": [1]},
        ],
        {"A0003", "A0004"},
    )

    assert assigned == {"A0003"}
    assert handles[0]["evidence_ids"] == ["A0001", "A0003"]
    assert handles[1]["evidence_ids"] == ["A0002", "A0003"]
    assert invalid == ["A0004:handles=[3]", "A9999"]


def test_apply_assignments_enforces_handle_capacity():
    handles = [{"evidence_ids": ["A0001", "A0002"]}]

    assigned, invalid = _apply_assignments(
        handles,
        [{"id": "A0003", "handle_number": 1}],
        {"A0003"},
        max_evidence_per_handle=2,
    )

    assert assigned == set()
    assert handles[0]["evidence_ids"] == ["A0001", "A0002"]
    assert invalid == ["A0003:handle=1:full"]


def test_complete_singleton_handles_preserves_residual_source_text():
    handles = [{"label": "Existing", "evidence_ids": ["A0001"]}]
    known = {
        "A0002": {"text": "Second source item."},
        "A0003": {"text": "Third source item."},
    }

    completed = _complete_singleton_handles(
        handles, known, ["A0003", "A9999", "A0002", "A0003"]
    )

    assert completed == ["A0002", "A0003"]
    assert handles[1] == {
        "label": "Unclustered source item A0002",
        "anchor_entities": [],
        "summary": "Second source item.",
        "evidence_ids": ["A0002"],
        "selection_rationale": (
            "Deterministic fallback preserving source evidence that the "
            "query-independent organizer did not cluster."
        ),
    }
    assert handles[2]["summary"] == "Third source item."


def test_drop_invalid_evidence_ids_rejects_invented_references():
    handles = [
        {"evidence_ids": ["A0001", "A9999"]},
        {"evidence_ids": ["A8888"]},
    ]

    rejected = _drop_invalid_evidence_ids(handles, {"A0001": {}})

    assert rejected == ["A8888", "A9999"]
    assert handles == [
        {"evidence_ids": ["A0001"]},
        {"evidence_ids": []},
    ]


def test_discovered_handle_count_excludes_only_declared_fallbacks():
    handles = [
        {"evidence_ids": ["A0001", "A0002"]},
        {"evidence_ids": ["A0003"]},
        {"evidence_ids": ["A0004"]},
    ]

    assert _discovered_handle_count(handles, ["A0004"]) == 2


def test_capacity_constrained_assignment_reroutes_when_best_handle_is_full():
    handles = [
        {"evidence_ids": ["existing"]},
        {"evidence_ids": []},
    ]

    assignments, similarities = _capacity_constrained_assignments(
        handles,
        ["A0001", "A0002"],
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.9, 0.1]],
        max_evidence_per_handle=2,
    )

    assert assignments == [
        {"id": "A0001", "handle_number": 1},
        {"id": "A0002", "handle_number": 2},
    ]
    assert similarities[0] == 1.0
    assert similarities[1] > 0.0
