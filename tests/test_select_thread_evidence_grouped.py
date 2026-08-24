"""Contract tests for grouped temporal-spread selection.

Synthetic threads and injected deterministic scorers only — no benchmark
data, no model downloads.
"""

from __future__ import annotations

import pytest

from yantrikdb.thread import (
    ThreadSelectionPolicyInfeasible,
    select_thread_evidence_grouped,
)


def make_thread(n, topic_map=None, total=None):
    items = []
    for i in range(n):
        items.append({
            "rid": f"row-{i:03d}",
            "text": f"turn {i} text",
            "created_at": 1000.0 + i,
            "source_turn": i,
            "position": i + 1,
            "entities": [],
            "routes": ["topic"],
            "phrases": [],
            "topic_rids": (topic_map or {}).get(i, []),
        })
    return {
        "items": items,
        "total": total if total is not None else n,
        "returned": n,
        "omitted": (total - n) if total is not None else 0,
    }


def scores_by_index(mapping, default=0.0):
    def scorer(focus, texts):
        return [mapping.get(i, default) for i in range(len(texts))]
    scorer.__name__ = "fixture_scorer"
    return scorer


def groups_of(result):
    return {g["group"]: g for g in result["selection"]["admitted_groups"]}


def test_endpoints_always_survive_per_group():
    # Topic A spans rows 0..9; the most relevant rows are in the middle.
    # Flat selection would keep 4,5,6; grouped must keep 0 and 9.
    thread = make_thread(10, topic_map={i: ["A"] for i in range(10)})
    result = select_thread_evidence_grouped(
        thread, "focus", budget=3,
        scorer=scores_by_index({4: 0.9, 5: 0.8, 6: 0.7}),
        min_representatives=2,
    )
    selected = result["selection"]["selected_indices"]
    assert 0 in selected and 9 in selected
    assert len(selected) == 3


def test_largest_gap_fill_not_relevance():
    # After endpoints 0 and 9, the third slot must go to the largest-gap
    # midpoint region, not to the highest-relevance row sitting next to
    # an endpoint.
    thread = make_thread(10, topic_map={i: ["A"] for i in range(10)})
    result = select_thread_evidence_grouped(
        thread, "focus", budget=3,
        scorer=scores_by_index({1: 0.99}),  # relevance bait beside endpoint
        min_representatives=2,
    )
    selected = result["selection"]["selected_indices"]
    assert selected[0] == 0 and selected[-1] == 9
    middle = selected[1]
    assert middle in (4, 5), f"bridge slot went to {middle}, not the span middle"


def test_mass_ordering_admits_strongest_groups_first():
    # Two topics; budget affords only one group's floor. B has higher mass.
    thread = make_thread(
        6, topic_map={0: ["A"], 1: ["A"], 2: ["A"], 3: ["B"], 4: ["B"], 5: ["B"]},
    )
    result = select_thread_evidence_grouped(
        thread, "focus", budget=2,
        scorer=scores_by_index({3: 0.9, 4: 0.8, 0: 0.1}),
        min_representatives=2,
    )
    admitted = groups_of(result)
    assert list(admitted) == ["B"]
    assert admitted["B"]["selected"] == [3, 5]  # endpoints of B's span


def test_multi_topic_rows_assigned_to_highest_mass_topic():
    # Row 2 names A and B; B's mass is higher, so row 2 belongs to B for
    # allocation and A keeps only rows 0,1.
    thread = make_thread(
        5, topic_map={0: ["A"], 1: ["A"], 2: ["A", "B"], 3: ["B"], 4: ["B"]},
    )
    result = select_thread_evidence_grouped(
        thread, "focus", budget=4,
        scorer=scores_by_index({3: 0.9, 4: 0.85, 2: 0.8, 0: 0.1, 1: 0.1}),
        min_representatives=2,
    )
    admitted = groups_of(result)
    assert admitted["B"]["rows"] == [2, 3, 4]
    assert admitted["A"]["rows"] == [0, 1]


def test_mass_tie_breaks_by_topic_rid_ascending():
    # Identical scores everywhere: masses tie; multi-topic row must go to
    # the lexicographically first topic.
    thread = make_thread(3, topic_map={0: ["B", "A"], 1: ["A"], 2: ["B"]})
    result = select_thread_evidence_grouped(
        thread, "focus", budget=3,
        scorer=scores_by_index({}, default=0.5),
        min_representatives=1,
    )
    admitted = groups_of(result)
    assert admitted["A"]["rows"] == [0, 1]
    assert admitted["B"]["rows"] == [2]


def test_residual_group_reported():
    thread = make_thread(4, topic_map={0: ["A"], 1: ["A"]})
    result = select_thread_evidence_grouped(
        thread, "focus", budget=4,
        scorer=scores_by_index({2: 0.9, 3: 0.8, 0: 0.5, 1: 0.4}),
        min_representatives=2,
    )
    assert result["selection"]["residual_selected_count"] == 2
    assert "residual" in groups_of(result)


def test_leftover_budget_round_robin_in_rank_order():
    # Both groups admitted at floor 2; TWO leftover slots are spent in
    # rounds over admitted groups in rank order — one to A (rank 1),
    # one to B — never both to the higher-mass group (frozen grid-v2
    # rule: egalitarian rounds, not mass-proportional).
    thread = make_thread(
        10,
        topic_map={**{i: ["A"] for i in range(6)}, **{i: ["B"] for i in range(6, 10)}},
    )
    result = select_thread_evidence_grouped(
        thread, "focus", budget=6,
        scorer=scores_by_index({0: 0.9, 1: 0.8, 2: 0.7}),  # A has the mass
        min_representatives=2,
    )
    admitted = groups_of(result)
    assert admitted["A"]["granted"] == 3
    assert admitted["B"]["granted"] == 3
    assert sum(g["granted"] for g in admitted.values()) == 6


def test_admission_stops_at_first_unaffordable_group():
    # Rank order: A (mass .9, 3 rows), B (mass .5, 3 rows), C (mass .1,
    # 1 row). Budget 5 affords A(2) + B(2) but NOT C... wait: 2+2+1=5
    # fits. Use budget 4: A(2)+B(2)=4, then C's floor (1) exceeds the
    # remaining 0 — C is excluded. Now make the middle group the
    # unaffordable one: budget 3 affords A(2) but not B(2); admission
    # STOPS at B and never considers C even though C's floor (1) fits.
    thread = make_thread(
        7,
        topic_map={
            0: ["A"], 1: ["A"], 2: ["A"],
            3: ["B"], 4: ["B"], 5: ["B"],
            6: ["C"],
        },
    )
    result = select_thread_evidence_grouped(
        thread, "focus", budget=3,
        scorer=scores_by_index({0: 0.9, 1: 0.9, 3: 0.5, 4: 0.5}),
        min_representatives=2,
    )
    admitted = groups_of(result)
    assert list(admitted) == ["A"], "admission must STOP at the first unaffordable group"
    assert admitted["A"]["granted"] == 3  # leftover round reaches only A


def test_infeasible_budget_raises():
    thread = make_thread(4, topic_map={i: ["A"] for i in range(4)})
    with pytest.raises(ThreadSelectionPolicyInfeasible, match="cannot afford"):
        select_thread_evidence_grouped(
            thread, "focus", budget=1,
            scorer=scores_by_index({}),
            min_representatives=2,
        )


def test_small_group_floor_capped_by_group_size():
    # A single-row group must be admittable at floor 2 (capped to 1 row).
    thread = make_thread(3, topic_map={0: ["A"], 1: ["B"], 2: ["B"]})
    result = select_thread_evidence_grouped(
        thread, "focus", budget=3,
        scorer=scores_by_index({0: 0.9}),
        min_representatives=2,
    )
    admitted = groups_of(result)
    assert admitted["A"]["granted"] == 1
    assert admitted["B"]["granted"] == 2


def test_chronological_presentation_and_shape():
    thread = make_thread(8, topic_map={i: ["A"] for i in range(8)}, total=20)
    result = select_thread_evidence_grouped(
        thread, "focus", budget=4,
        scorer=scores_by_index({6: 0.9}),
        min_representatives=2,
    )
    rids = [i["rid"] for i in result["items"]]
    assert rids == sorted(rids)
    assert result["total"] == 20
    assert result["returned"] == 4
    assert result["omitted"] == 16
    assert result["selection"]["strategy"] == "grouped-spread"


def test_determinism_repeat():
    thread = make_thread(
        12, topic_map={**{i: ["A"] for i in range(6)}, **{i: ["B"] for i in range(6, 12)}},
    )
    scorer = scores_by_index({2: 0.7, 7: 0.7, 9: 0.6})
    first = select_thread_evidence_grouped(
        thread, "focus", budget=6, scorer=scorer, min_representatives=2,
    )
    second = select_thread_evidence_grouped(
        thread, "focus", budget=6, scorer=scorer, min_representatives=2,
    )
    assert first == second


def test_validation_shared_with_flat():
    thread = make_thread(3, topic_map={0: ["A"]})
    with pytest.raises(ValueError, match="exceeds thread rows"):
        select_thread_evidence_grouped(
            thread, "f", budget=4, scorer=scores_by_index({}),
        )
    with pytest.raises(TypeError):
        select_thread_evidence_grouped(
            thread, "f", budget=2, scorer=scores_by_index({}),
            min_representatives=True,
        )
    thread["total"] = 1
    with pytest.raises(ValueError, match="smaller than"):
        select_thread_evidence_grouped(
            thread, "f", budget=2, scorer=scores_by_index({}),
        )

    def nan_scorer(focus, texts):
        return [float("nan")] * len(texts)

    thread["total"] = 3
    with pytest.raises(ValueError, match="row 0"):
        select_thread_evidence_grouped(
            thread, "f", budget=2, scorer=nan_scorer,
        )
