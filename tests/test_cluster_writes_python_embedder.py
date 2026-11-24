"""summarize_cluster, record_synthesis and consolidate_cluster with a Python embedder.

Called without an explicit `embedding`, these let the engine embed the new
text, and the engine cannot reach an embedder attached with `set_embedder(obj)`.
On a store whose only embedder was a Python one they raised "No embedder
configured". The binding now embeds the text with the attached embedder first,
the precedence every other write follows.
"""

import os
import random

import pytest

import yantrikdb._yantrikdb_rust as ext
from yantrikdb import YantrikDB

DIM = 32  # not 64/256: those dims auto-attach a bundled engine-native embedder


class Emb:
    def __init__(self):
        self.seen = []

    def encode(self, text):
        self.seen.append(text)
        r = random.Random(hash(text) & 0xFFFFFFFF)
        v = [r.random() for _ in range(DIM)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]


@pytest.fixture
def store(tmp_path):
    db = YantrikDB(os.path.join(str(tmp_path), "cluster.db"), DIM)
    emb = Emb()
    db.set_embedder(emb, allow_unverified_embedder=True)
    rids = [db.record(f"alice moved the launch to march, note {i}") for i in range(3)]
    yield db, emb, rids
    db.close()


CALLS = {
    "summarize_cluster": lambda db, rids, text: ext.summarize_cluster(
        db, rids, text, idempotency_key="k1"
    ),
    "record_synthesis": lambda db, rids, text: ext.record_synthesis(
        db, rids, text, "topic", "k2"
    ),
    "consolidate_cluster": lambda db, rids, text: ext.consolidate_cluster(db, rids, text),
}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_writes_with_only_a_python_embedder(store, name):
    db, emb, rids = store
    text = f"the launch moved to march ({name})"
    out = CALLS[name](db, rids[:2], text)
    assert out, out
    assert text in emb.seen, "the attached embedder embedded the new text"
