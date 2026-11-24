"""draft_memories_from_summary with a Python embedder.

The engine drafts memories through its own embedder, which cannot reach one
attached with `set_embedder(obj)`. On a store whose only embedder was a Python
one the call raised "No embedder configured"; on a store with both it used the
engine's, against the precedence every other write follows. The binding now
splits the summary, embeds each fact with the caller's embedder and records
them with the same fields the engine path writes.
"""

import os
import random

import pytest

from yantrikdb import YantrikDB

DIM = 32  # not 64/256: those dims auto-attach a bundled engine-native embedder

SUMMARY = (
    "We decided to use keyset cursors for list_records. "
    "Alice will own the database migration next sprint. "
    "The production launch slipped to March 30 because of it."
)

# What a drafted memory is, minus rid, timestamps and vector.
FIELDS = ("type", "text", "importance", "half_life", "certainty", "source", "domain",
          "namespace", "metadata", "valence", "emotional_state")


class Emb:
    def __init__(self, dim=DIM):
        self.dim = dim
        self.calls = 0

    def encode(self, text):
        self.calls += 1
        r = random.Random(hash(text) & 0xFFFFFFFF)
        v = [r.random() for _ in range(self.dim)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]


def _drafted(db, rids):
    return [{k: db.get(rid)[k] for k in FIELDS} for rid in rids]


def test_drafts_with_only_a_python_embedder(tmp_path):
    db = YantrikDB(os.path.join(str(tmp_path), "py.db"), DIM)
    try:
        emb = Emb()
        db.set_embedder(emb, allow_unverified_embedder=True)
        emb.calls = 0
        rids = db.draft_memories_from_summary(SUMMARY, namespace="session", domain="work")
        assert len(rids) >= 2
        assert emb.calls == len(rids), "one encode() per drafted fact"
        for row in _drafted(db, rids):
            assert row["type"] == "semantic"
            assert row["importance"] == 0.5
            assert row["half_life"] == 604800.0
            assert row["certainty"] == 0.7
            assert row["source"] == "session_auto_capture"
            assert row["domain"] == "work"
            assert row["namespace"] == "session"
            assert row["metadata"] == {"provisional": True, "kind": "session_auto_capture"}
        # The stored vectors are the Python embedder's: each fact is its own
        # nearest neighbour under that embedder.
        for rid in rids:
            text = db.get(rid)["text"]
            hit = db.recall(query_embedding=Emb().encode(text), top_k=1, skip_reinforce=True)
            assert hit[0]["rid"] == rid
    finally:
        db.close()


def _native_store(path):
    db = YantrikDB(path, 64)
    if not db.has_embedder():
        db.close()
        pytest.skip("this build has no bundled engine embedder")
    return db


def test_python_path_matches_the_engine_path(tmp_path):
    engine = _native_store(os.path.join(str(tmp_path), "engine.db"))
    caller = YantrikDB(os.path.join(str(tmp_path), "caller.db"), DIM)
    try:
        caller.set_embedder(Emb(), allow_unverified_embedder=True)
        a = engine.draft_memories_from_summary(SUMMARY, namespace="session", domain="work")
        b = caller.draft_memories_from_summary(SUMMARY, namespace="session", domain="work")
        assert len(a) == len(b) >= 2
        assert _drafted(engine, a) == _drafted(caller, b)
    finally:
        engine.close()
        caller.close()


def test_an_attached_python_embedder_outranks_the_engine_one(tmp_path):
    db = _native_store(os.path.join(str(tmp_path), "both.db"))
    try:
        emb = Emb(dim=64)
        db.set_embedder(emb, allow_unverified_embedder=True)
        emb.calls = 0
        rids = db.draft_memories_from_summary(SUMMARY)
        assert rids
        assert emb.calls == len(rids), "the attached embedder made the vectors"
    finally:
        db.close()
