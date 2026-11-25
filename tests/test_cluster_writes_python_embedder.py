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


# ── keyed retries with an embedder that is not bit-stable ────────────────
#
# A vector the binding computed from the text is not part of the caller's
# payload. A keyed retry is judged on the text and fields, as on the
# engine-embedder path, so an embedder that does not reproduce its output bit
# for bit (GPU kernels, threaded BLAS) does not turn an honest retry into an
# idempotency conflict. A vector the caller passes explicitly stays part of
# the payload.


class NoisyEmb(Emb):
    def __init__(self, dim=DIM):
        super().__init__()
        self.dim = dim
        self.noise = random.Random(7)

    def encode(self, text):
        self.seen.append(text)
        r = random.Random(hash(text) & 0xFFFFFFFF)
        v = [r.random() + 1e-6 * self.noise.random() for _ in range(self.dim)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]


def _store(tmp_path, dim):
    db = YantrikDB(os.path.join(str(tmp_path), f"noisy{dim}.db"), dim)
    if dim == 64 and not db.has_embedder():
        db.close()
        pytest.skip("this build has no bundled engine embedder")
    emb = NoisyEmb(dim)
    db.set_embedder(emb, allow_unverified_embedder=True)
    rids = [db.record(f"alice moved the launch to march, note {i}") for i in range(3)]
    return db, emb, rids


KEYED = {
    "record_synthesis": lambda db, rids, text: ext.record_synthesis(
        db, rids, text, "topic", "retry-key"
    ),
    "summarize_cluster": lambda db, rids, text: ext.summarize_cluster(
        db, rids, text, idempotency_key="retry-key"
    ),
}


@pytest.mark.parametrize("dim", [DIM, 64], ids=["python-only", "both-embedders"])
@pytest.mark.parametrize("name", sorted(KEYED))
def test_a_keyed_retry_with_a_noisy_embedder_returns_the_same_rid(tmp_path, name, dim):
    db, emb, rids = _store(tmp_path, dim)
    try:
        first = KEYED[name](db, rids[:2], "the launch moved to march")
        calls = len(emb.seen)
        again = KEYED[name](db, rids[:2], "the launch moved to march")
        assert again["consolidated_rid"] == first["consolidated_rid"]
        assert again.get("idempotent_replay") is True
        assert len(emb.seen) == calls, "a retry returns before calling encode()"
        with pytest.raises(RuntimeError, match="(?i)idempot"):
            KEYED[name](db, rids[:2], "the launch moved to april")
    finally:
        db.close()


# ── whose identity a caller-embedded vector carries ──────────────────────


def test_a_caller_embedded_summary_does_not_stamp_the_engine_identity(tmp_path):
    # On a 64-dim store the engine attaches its bundled embedder, and the
    # caller attaches a Python one. A summary the Python embedder made must
    # not stamp the bundled model as the store's embedder identity: reopening
    # with the Python embedder was then refused, and seal_pack declared the
    # wrong model.
    db = YantrikDB(os.path.join(str(tmp_path), "identity.db"), 64)
    try:
        if not db.has_embedder():
            pytest.skip("this build has no bundled engine embedder")
        db.set_embedder(NoisyEmb(64), allow_unverified_embedder=True)
        rids = [db.record(f"alice moved the launch to march, note {i}") for i in range(3)]
        assert db.embedder_identity() is None
        ext.summarize_cluster(db, rids[:2], "the launch moved to march", idempotency_key="id-1")
        assert db.embedder_identity() is None, db.embedder_identity()
    finally:
        db.close()
