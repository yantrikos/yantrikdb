"""Contract tests for ``select_thread_evidence`` (retrieve-wide-then-compress).

Synthetic threads and injected deterministic scorers only — no benchmark
data, no model downloads, and the module under test never imports
benchmark code.
"""

from __future__ import annotations

import sys

import pytest

from yantrikdb.thread import (
    ThreadSelectionPolicyInfeasible,
    select_thread_evidence,
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


def test_selects_top_budget_and_presents_chronologically():
    # Highest scores on late rows: selection is by score, presentation
    # must return to chronological order.
    thread = make_thread(6)
    result = select_thread_evidence(
        thread, "focus", budget=3,
        scorer=scores_by_index({5: 0.9, 1: 0.8, 3: 0.7, 0: 0.1}),
    )
    assert [i["rid"] for i in result["items"]] == ["row-001", "row-003", "row-005"]
    assert result["selection"]["selected_indices"] == [1, 3, 5]


def test_completeness_semantics_preserved():
    # A thread already truncated upstream: total stays, omitted grows.
    thread = make_thread(4, total=10)
    result = select_thread_evidence(
        thread, "focus", budget=2, scorer=scores_by_index({0: 1.0, 1: 0.9}),
    )
    assert result["total"] == 10
    assert result["returned"] == 2
    assert result["omitted"] == 8


def test_tie_break_is_chronological_position():
    thread = make_thread(4)
    result = select_thread_evidence(
        thread, "focus", budget=2, scorer=scores_by_index({}, default=0.5),
    )
    assert result["selection"]["selected_indices"] == [0, 1]


def test_identity_selection_flagged():
    thread = make_thread(3)
    result = select_thread_evidence(
        thread, "focus", budget=3, scorer=scores_by_index({}),
    )
    assert result["selection"]["identity_selection"] is True
    assert result["returned"] == 3
    assert result["omitted"] == 0


def test_budget_above_rows_is_invalid():
    thread = make_thread(3)
    with pytest.raises(ValueError, match="identity selection"):
        select_thread_evidence(thread, "focus", budget=4, scorer=scores_by_index({}))


def test_budget_and_policy_validation():
    thread = make_thread(3)
    with pytest.raises(ValueError):
        select_thread_evidence(thread, "f", budget=0, scorer=scores_by_index({}))
    with pytest.raises(TypeError):
        select_thread_evidence(thread, "f", budget=True, scorer=scores_by_index({}))
    with pytest.raises(TypeError):
        select_thread_evidence(
            thread, "f", budget=2, scorer=scores_by_index({}), min_per_topic=1.0,
        )
    with pytest.raises(ValueError):
        select_thread_evidence(
            thread, "f", budget=2, scorer=scores_by_index({}), min_per_topic=0,
        )


def test_min_per_topic_reserves_top_scored_per_topic():
    # Topic A on rows 0,1; topic B on rows 2,3; global ranking would pick
    # rows 4,5 — the floor must reserve A's and B's best first.
    thread = make_thread(
        6, topic_map={0: ["A"], 1: ["A"], 2: ["B"], 3: ["B"]},
    )
    result = select_thread_evidence(
        thread, "focus", budget=3,
        scorer=scores_by_index({4: 0.9, 5: 0.8, 1: 0.5, 3: 0.4, 0: 0.1, 2: 0.1}),
        min_per_topic=1,
    )
    # Reserved: row 1 (A's best), row 3 (B's best); remainder: row 4.
    assert result["selection"]["reserved_indices"] == [1, 3]
    assert result["selection"]["selected_indices"] == [1, 3, 4]


def test_min_per_topic_multi_topic_rows_dedupe():
    # One row carries both topics: a single reservation satisfies both.
    thread = make_thread(4, topic_map={1: ["A", "B"], 2: ["A"], 3: ["B"]})
    result = select_thread_evidence(
        thread, "focus", budget=2,
        scorer=scores_by_index({1: 0.9, 0: 0.8, 2: 0.1, 3: 0.1}),
        min_per_topic=1,
    )
    assert result["selection"]["reserved_indices"] == [1]
    assert result["selection"]["selected_indices"] == [0, 1]


def test_min_per_topic_infeasible_budget():
    thread = make_thread(4, topic_map={0: ["A"], 1: ["B"], 2: ["C"]})
    with pytest.raises(ThreadSelectionPolicyInfeasible, match="exceeding budget"):
        select_thread_evidence(
            thread, "focus", budget=2, scorer=scores_by_index({}),
            min_per_topic=1,
        )


def test_min_per_topic_infeasible_topic_rows():
    thread = make_thread(3, topic_map={0: ["A"]})
    with pytest.raises(ThreadSelectionPolicyInfeasible, match="fewer than"):
        select_thread_evidence(
            thread, "focus", budget=3, scorer=scores_by_index({}),
            min_per_topic=2,
        )


def test_topicless_rows_create_no_reservation():
    thread = make_thread(4, topic_map={0: ["A"]})
    result = select_thread_evidence(
        thread, "focus", budget=2,
        scorer=scores_by_index({3: 0.9, 0: 0.5}),
        min_per_topic=1,
    )
    assert result["selection"]["reserved_indices"] == [0]
    assert result["selection"]["selected_indices"] == [0, 3]


def test_diagnostics_are_recomputable_without_text():
    thread = make_thread(4, topic_map={2: ["T"]})
    result = select_thread_evidence(
        thread, "focus", budget=2,
        scorer=scores_by_index({2: 0.9, 0: 0.4, 1: 0.3, 3: 0.1}),
    )
    rows = result["selection"]["rows"]
    assert [r["index"] for r in rows] == [0, 1, 2, 3]
    assert all({"index", "rid", "score", "topic_rids"} <= set(r) for r in rows)
    assert "text" not in rows[0]
    # Recompute the ranking from diagnostics alone.
    ranked = sorted(rows, key=lambda r: (-r["score"], r["index"]))
    assert [r["index"] for r in ranked][:2] == [2, 0]
    assert result["selection"]["selected_indices"] == [0, 2]


def test_determinism_repeat():
    thread = make_thread(8, topic_map={1: ["A"], 5: ["A"]})
    scorer = scores_by_index({1: 0.7, 5: 0.7, 2: 0.6})
    first = select_thread_evidence(
        thread, "focus", budget=3, scorer=scorer, min_per_topic=1,
    )
    second = select_thread_evidence(
        thread, "focus", budget=3, scorer=scorer, min_per_topic=1,
    )
    assert first == second


def test_scorer_validation():
    thread = make_thread(2)
    with pytest.raises(TypeError, match="scorer must be"):
        select_thread_evidence(thread, "f", budget=1, scorer=123)

    def short_scorer(focus, texts):
        return [0.5]

    with pytest.raises(ValueError, match="1 scores for 2 rows"):
        select_thread_evidence(thread, "f", budget=1, scorer=short_scorer)


def test_empty_thread_raises():
    with pytest.raises(ValueError, match="requires a thread with items"):
        select_thread_evidence({"items": []}, "f", budget=1)


def test_input_thread_not_mutated():
    thread = make_thread(4)
    before = [i["rid"] for i in thread["items"]]
    select_thread_evidence(
        thread, "focus", budget=2, scorer=scores_by_index({3: 1.0}),
    )
    assert [i["rid"] for i in thread["items"]] == before
    assert "selection" not in thread


def test_no_benchmark_imports():
    module = sys.modules["yantrikdb.thread"]
    assert not [
        name for name in sys.modules
        if name.startswith("benchmarks") and module.__name__ in name
    ]


def test_nonfinite_and_bool_scores_are_typed_errors():
    thread = make_thread(3)
    for bad in (float("nan"), float("inf"), float("-inf"), True, "0.5", None):
        def bad_scorer(focus, texts, _bad=bad):
            return [0.5, _bad, 0.1]

        with pytest.raises(ValueError, match="row 1"):
            select_thread_evidence(thread, "f", budget=1, scorer=bad_scorer)


def test_cross_encoder_output_is_validated(monkeypatch):
    # CE returning a short / non-finite vector must fail the same way as
    # a callable scorer — the validator is shared.
    import yantrikdb.thread as thread_mod

    thread = make_thread(3)
    monkeypatch.setattr(
        thread_mod, "_cross_encoder_scores", lambda focus, texts: [0.5]
    )
    with pytest.raises(ValueError, match="1 scores for 3 rows"):
        select_thread_evidence(thread, "f", budget=1, scorer="cross-encoder")
    monkeypatch.setattr(
        thread_mod,
        "_cross_encoder_scores",
        lambda focus, texts: [0.5, float("nan"), 0.1],
    )
    with pytest.raises(ValueError, match="row 1"):
        select_thread_evidence(thread, "f", budget=1, scorer="cross-encoder")


def test_duplicate_topic_ids_count_once_for_feasibility():
    # One row lists topic A twice: floor=2 must be infeasible, not
    # silently satisfied by a single unique row.
    thread = make_thread(3, topic_map={0: ["A", "A"]})
    with pytest.raises(ThreadSelectionPolicyInfeasible, match="fewer than"):
        select_thread_evidence(
            thread, "f", budget=3, scorer=scores_by_index({}),
            min_per_topic=2,
        )


def test_malformed_total_is_rejected_not_coerced():
    thread = make_thread(3)
    thread["total"] = "10"
    with pytest.raises(ValueError, match="total must be an int"):
        select_thread_evidence(thread, "f", budget=2, scorer=scores_by_index({}))
    thread["total"] = True
    with pytest.raises(ValueError, match="total must be an int"):
        select_thread_evidence(thread, "f", budget=2, scorer=scores_by_index({}))
    thread["total"] = 2  # fewer than the 3 items present
    with pytest.raises(ValueError, match="smaller than"):
        select_thread_evidence(thread, "f", budget=2, scorer=scores_by_index({}))
