"""Contract tests for ``resolve_thread_topics`` (adaptive topic budget).

The resolver is dev-calibrated (floor=12) and cap-bounded (16); these
tests pin the QUERY-ONLY knee mechanics, the oversampling/exhaustion
semantics, the validation surface, and the exact recall invocation —
using synthetic score curves only. Nothing here reads benchmark data,
and the module under test must not import any benchmark code.
"""

from __future__ import annotations

import sys

import pytest

from yantrikdb.thread import resolve_thread_topics


class FakeDB:
    """Records recall kwargs and serves a scripted organizer ranking."""

    def __init__(self, scores, *, noise_rows=0, labels=None, rids=None,
                 scripted_batches=None):
        self.calls = []
        self._scores = scores
        self._noise_rows = noise_rows
        self._labels = labels
        self._rids = rids
        self._scripted_batches = scripted_batches

    def _organizer_hits(self):
        hits = []
        for i, score in enumerate(self._scores):
            rid = self._rids[i] if self._rids else f"topic-{i:03d}"
            label = self._labels[i] if self._labels else f"label {i}"
            hits.append({
                "rid": rid,
                "score": score,
                "metadata": {
                    "organizer_kind": "query_independent_topic",
                    "organizer_label": label,
                },
            })
        return hits

    def recall(self, **kwargs):
        self.calls.append(kwargs)
        if self._scripted_batches is not None:
            return self._scripted_batches[len(self.calls) - 1]
        # Noise ranks FIRST (score 0.99 beats every organizer topic), the
        # realistic worst case: raw inference rows bury organizer topics.
        hits = [
            {
                "rid": f"noise-{i:03d}",
                "score": 0.99,
                "metadata": {"organizer_kind": "something_else"},
            }
            for i in range(self._noise_rows)
        ]
        hits.extend(self._organizer_hits())
        return hits[: kwargs["top_k"]]


def descending(n, start=0.9, step=0.01):
    return [round(start - i * step, 6) for i in range(n)]


def test_exact_recall_kwargs_are_pinned():
    db = FakeDB(descending(16))
    resolve_thread_topics(db, "the move to Portland", "ns-a")
    assert db.calls == [{
        "query": "the move to Portland",
        "top_k": 64,
        "namespace": "ns-a",
        "source": "inference",
        "include_consolidated": True,
        "skip_reinforce": True,
    }]


def test_no_benchmark_imports():
    module = sys.modules["yantrikdb.thread"]
    assert not [
        name for name in sys.modules
        if name.startswith("benchmarks") and module.__name__ in name
    ]
    source_deps = getattr(module, "__dict__", {})
    assert "benchmarks" not in repr(source_deps.keys())


def test_knee_cuts_at_largest_gap_after_floor():
    # Big gap after position 14 (1-based): scores 0..13 tight, then drop.
    scores = descending(14, start=0.9, step=0.001) + [0.5, 0.49]
    result = resolve_thread_topics(FakeDB(scores), "q")
    assert result["selected_count"] == 14
    assert result["cut_index"] == 14
    assert result["cut_score"] == scores[13]
    assert result["largest_gap"] == pytest.approx(scores[13] - scores[14])
    assert result["flat_curve"] is False
    assert result["selected_rids"] == [f"topic-{i:03d}" for i in range(14)]


def test_knee_never_cuts_before_floor():
    # The overwhelmingly largest gap is after position 2 — but the floor
    # forbids cutting there; the winning cut is the largest gap at/after 12.
    scores = [0.9, 0.89, 0.3, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23,
              0.22, 0.21, 0.15, 0.14, 0.13, 0.12]
    result = resolve_thread_topics(FakeDB(scores), "q")
    assert result["selected_count"] == 12
    assert result["largest_gap"] == pytest.approx(0.21 - 0.15)


def test_equal_largest_gaps_cut_later():
    # Identical gaps of 0.1 after positions 12 and 14 (1-based): the later
    # cut must win (coverage-conservative).
    scores = descending(12, start=0.9, step=0.001)
    scores += [scores[-1] - 0.1]            # gap 0.1 after position 12
    scores += [scores[-1] - 0.001]
    scores += [scores[-1] - 0.1]            # gap 0.1 after position 14
    scores += [scores[-1] - 0.001]
    result = resolve_thread_topics(FakeDB(scores), "q")
    assert result["selected_count"] == 14


def test_n_at_most_floor_selects_all_with_null_diagnostics():
    for n in (1, 5, 12):
        result = resolve_thread_topics(FakeDB(descending(n)), "q")
        assert result["selected_count"] == n
        assert result["cut_index"] == n
        assert result["cut_score"] is None
        assert result["largest_gap"] is None
        assert result["flat_curve"] is False


def test_flat_curve_above_floor_selects_all():
    result = resolve_thread_topics(FakeDB([0.5] * 16), "q")
    assert result["selected_count"] == 16
    assert result["flat_curve"] is True
    assert result["cut_score"] is None
    assert result["largest_gap"] is None


def test_cap_bounds_selection_and_ranked():
    db = FakeDB(descending(30))
    result = resolve_thread_topics(db, "q")
    assert len(result["ranked"]) == 16
    assert result["selected_count"] <= 16


def test_ranked_is_full_precut_audit_surface():
    scores = descending(14, start=0.9, step=0.001) + [0.5, 0.49]
    result = resolve_thread_topics(FakeDB(scores), "q")
    # The knee must be recomputable from `ranked` alone.
    assert [r["score"] for r in result["ranked"]] == scores
    assert result["selected_count"] < len(result["ranked"])
    assert {"rid", "label", "score"} <= set(result["ranked"][0])


def test_deterministic_under_score_ties():
    rids = [f"z-{i}" for i in range(8)] + [f"a-{i}" for i in range(8)]
    db = FakeDB([0.5] * 16, rids=rids)
    first = resolve_thread_topics(db, "q")
    second = resolve_thread_topics(FakeDB([0.5] * 16, rids=list(rids)), "q")
    assert first["selected_rids"] == second["selected_rids"]
    # Ties break by rid ascending.
    assert first["selected_rids"] == sorted(rids)


def test_duplicate_rids_keep_highest_ranked_occurrence():
    batches = [[
        {"rid": "dup", "score": 0.9,
         "metadata": {"organizer_kind": "query_independent_topic"}},
        {"rid": "dup", "score": 0.4,
         "metadata": {"organizer_kind": "query_independent_topic"}},
        {"rid": "other", "score": 0.3,
         "metadata": {"organizer_kind": "query_independent_topic"}},
    ]]
    result = resolve_thread_topics(FakeDB(None, scripted_batches=batches), "q")
    assert result["selected_rids"] == ["dup", "other"]
    assert result["ranked"][0]["score"] == 0.9


def test_oversampling_doubles_until_cap_or_exhaustion():
    # 100 noise rows bury the organizer topics: the first request (64)
    # yields too few organizer hits AND returned == requested, so the
    # resolver must double and re-query.
    db = FakeDB(descending(16), noise_rows=100)
    result = resolve_thread_topics(db, "q")
    assert [c["top_k"] for c in db.calls] == [64, 128]
    assert result["store_exhausted"] is False
    assert result["candidate_cap_reached"] is False
    assert result["recall_rounds"] == 2
    assert len(result["ranked"]) == 16


def test_exhaustion_below_cap_is_valid_not_an_error():
    # Only 13 organizer topics exist in the whole store. Exhaustion is
    # orthogonal to the knee: all 13 are ranked, and the uniformly
    # descending curve knees at the floor (12) as usual.
    db = FakeDB(descending(13))
    result = resolve_thread_topics(db, "q")
    assert result["store_exhausted"] is True
    assert result["candidate_cap_reached"] is False
    assert len(result["ranked"]) == 13
    assert result["selected_count"] == 12


def test_zero_organizer_hits_raises_naming_namespace():
    db = FakeDB([])
    with pytest.raises(ValueError, match="ns-empty"):
        resolve_thread_topics(db, "q", "ns-empty")


def test_budget_validation():
    db = FakeDB(descending(16))
    with pytest.raises(ValueError):
        resolve_thread_topics(db, "q", cap=17)
    with pytest.raises(ValueError):
        resolve_thread_topics(db, "q", cap=0)
    with pytest.raises(ValueError):
        resolve_thread_topics(db, "q", cap=8, floor=9)
    with pytest.raises(TypeError):
        resolve_thread_topics(db, "q", cap=True)
    with pytest.raises(TypeError):
        resolve_thread_topics(db, "q", floor=12.0)


def test_malformed_scores_are_typed_errors():
    for bad in (None, "0.5", float("nan"), float("inf"), True):
        batches = [[
            {"rid": "t-0", "score": bad,
             "metadata": {"organizer_kind": "query_independent_topic"}},
        ]]
        with pytest.raises(ValueError):
            resolve_thread_topics(FakeDB(None, scripted_batches=batches), "q")


def test_candidate_cap_hard_stop_is_flagged():
    # Every request comes back full of noise with only a few organizer
    # topics: the resolver must walk 64,128,256,512,1024 and stop there
    # with the false-negative flag set, never widening further.
    class NoisyDB(FakeDB):
        def recall(self, **kwargs):
            self.calls.append(kwargs)
            hits = self._organizer_hits()
            while len(hits) < kwargs["top_k"]:
                hits.append({
                    "rid": f"noise-{len(hits):05d}",
                    "score": 0.99,
                    "metadata": {"organizer_kind": "something_else"},
                })
            return hits[: kwargs["top_k"]]

    db = NoisyDB(descending(5))
    result = resolve_thread_topics(db, "q")
    assert [c["top_k"] for c in db.calls] == [64, 128, 256, 512, 1024]
    assert result["candidate_cap_reached"] is True
    assert result["store_exhausted"] is False
    assert result["recall_rounds"] == 5
    assert result["selected_count"] == 5


def test_candidate_cap_validation():
    db = FakeDB(descending(16))
    with pytest.raises(ValueError):
        resolve_thread_topics(db, "q", candidate_cap=10_001)
    with pytest.raises(ValueError):
        resolve_thread_topics(db, "q", cap=16, candidate_cap=15)
    with pytest.raises(TypeError):
        resolve_thread_topics(db, "q", candidate_cap=True)
    # candidate_cap below the default first request clamps the request.
    db2 = FakeDB(descending(16))
    resolve_thread_topics(db2, "q", candidate_cap=32)
    assert [c["top_k"] for c in db2.calls] == [32]


def test_floor_overridable_and_knee_respects_it():
    scores = [0.9, 0.5] + descending(6, start=0.4, step=0.001)
    result = resolve_thread_topics(FakeDB(scores), "q", cap=8, floor=1)
    assert result["selected_count"] == 1
    assert result["largest_gap"] == pytest.approx(0.4)
