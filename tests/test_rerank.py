"""Tests for the cross-encoder rerank helpers (model stubbed — the
measured quality numbers live in the node4 clone eval, not here)."""

import math

import pytest

import yantrikdb.rerank as rr
from yantrikdb import YantrikDB, recall_reranked, rerank_hits

DIM = 8


def _vec(seed: float) -> list[float]:
    raw = [math.sin(seed * 1.7 + i * 2.3) + math.cos(seed * 0.3 + i * 3.1) for i in range(DIM)]
    n = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / n for x in raw]


class StubCE:
    """Scores a passage by how early the word 'answer' appears."""

    def predict(self, pairs):
        return [1.0 if "answer" in text else 0.0 for _, text in pairs]


@pytest.fixture(autouse=True)
def stub_cross_encoder(monkeypatch):
    monkeypatch.setitem(rr._CE_CACHE, rr.DEFAULT_MODEL, StubCE())


def test_rerank_hits_reorders_and_scores():
    hits = [
        {"rid": "a", "text": "filler text"},
        {"rid": "b", "text": "the answer lives here"},
        {"rid": "c", "text": "more filler"},
    ]
    out = rerank_hits("q", hits, top_k=2)
    assert [h["rid"] for h in out] == ["b", "a"]
    assert out[0]["rerank_score"] == 1.0
    # input untouched
    assert "rerank_score" not in hits[0]


def test_rerank_hits_empty_is_empty():
    assert rerank_hits("q", []) == []


def test_recall_reranked_end_to_end():
    db = YantrikDB(db_path=":memory:", embedding_dim=DIM)
    try:
        db.record("plain record one", embedding=_vec(1.0))
        db.record("this holds the answer", embedding=_vec(30.0))
        db.record("plain record two", embedding=_vec(2.0))
        out = recall_reranked(
            db,
            "which record holds it?",
            top_k=2,
            pool_k=10,
            query_embedding=_vec(1.0),
            skip_reinforce=True,
        )
        assert out[0]["text"] == "this holds the answer"
        assert len(out) == 2
    finally:
        db.close()


def test_recall_reranked_rejects_presentation_kwargs():
    db = YantrikDB(db_path=":memory:", embedding_dim=DIM)
    try:
        for bad in ("snippets", "min_score_ratio"):
            with pytest.raises(ValueError):
                recall_reranked(
                    db, "q", query_embedding=_vec(1.0), **{bad: True}
                )
    finally:
        db.close()
