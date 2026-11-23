"""#291: a replacing consolidation is reversible from stored data, and the
additive entry point is public and safe to retry.

Everything here goes through the PUBLIC package (`yantrikdb`, not
`yantrikdb._yantrikdb_rust`), because the reachability of these functions
from the façade is part of what #291 reported missing.
"""

import importlib
import math

import pytest

import yantrikdb
from yantrikdb import YantrikDB, restore_from_consolidation, summarize_cluster
from yantrikdb.consolidate import consolidate_cluster

DIM = 8


def _vec(seed: float) -> list[float]:
    raw = [math.sin(seed * (i + 1) * 1.7) + math.cos(seed * (i + 2) * 0.3) for i in range(DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


@pytest.fixture
def db():
    engine = YantrikDB(db_path=":memory:", embedding_dim=DIM)
    yield engine
    engine.close()


def _seed(db, items):
    return [
        db.record(text, importance=imp, embedding=_vec(seed))
        for text, imp, seed in items
    ]


def _recalled(db, seed):
    return [h["rid"] for h in db.recall(query_embedding=_vec(seed), top_k=10)]


def test_both_functions_are_on_the_public_package():
    assert "summarize_cluster" in yantrikdb.__all__
    assert "restore_from_consolidation" in yantrikdb.__all__
    # `yantrikdb.consolidate` the ATTRIBUTE is the consolidate() function,
    # which shadows the submodule, so import the module by name.
    module = importlib.import_module("yantrikdb.consolidate")
    assert "summarize_cluster" in module.__all__
    assert "restore_from_consolidation" in module.__all__


def test_restore_brings_a_retired_source_back_into_recall(db):
    a, b = _seed(db, [("alpha fact about lisbon", 0.65, 1.0), ("beta fact", 0.5, 3.0)])
    out = consolidate_cluster(db, [a, b], "merged synthesis", embedding=_vec(2.0))
    summary = out["consolidated_rid"]
    assert a not in _recalled(db, 1.0), "precondition: a consolidated source is excluded"

    result = restore_from_consolidation(db, summary)

    assert {r["rid"] for r in result["restored"]} == {a, b}
    assert all(r["basis"] == "stored" for r in result["restored"])
    by_rid = {r["rid"]: r["importance"] for r in result["restored"]}
    assert by_rid[a] == pytest.approx(0.65)
    assert by_rid[b] == pytest.approx(0.5)
    assert a in _recalled(db, 1.0)
    # The summary stays beside its sources, and carries their rids.
    summary_hit = next(h for h in db.recall(query_embedding=_vec(2.0), top_k=10)
                       if h["rid"] == summary)
    assert set(summary_hit["metadata"]["consolidated_from"]) == {a, b}
    # Idempotent: nothing left to restore.
    assert restore_from_consolidation(db, summary)["restored"] == []


def test_keyed_summarize_cluster_replays(db):
    a, b = _seed(db, [("alpha", 0.6, 1.0), ("beta", 0.5, 3.0)])
    first = summarize_cluster(db, [a, b], "topic", embedding=_vec(2.0), idempotency_key="k")
    second = summarize_cluster(db, [b, a], "topic", embedding=_vec(2.0), idempotency_key="k")
    assert second["idempotent_replay"] is True
    assert second["consolidated_rid"] == first["consolidated_rid"]
    with pytest.raises(RuntimeError):
        summarize_cluster(db, [a, b], "changed text", embedding=_vec(2.0), idempotency_key="k")
