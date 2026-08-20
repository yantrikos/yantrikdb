from benchmarks.amb.global_organizer_probe import (
    _apply_assignments,
    _capacity_constrained_assignments,
    _normalize_handles,
)


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
