"""Replicated rows arrive without their vector; the engine must cope.

A follower's apply_ops() inserts memories without the embedding (the op
carries only its hash), so they sit active and hot with no vector until
refresh_embeddings() fills them in. think(), archive(), hydrate() and
evict() used to fail on them with "Invalid column type Null ... embedding".
"""

import random

from yantrikdb import YantrikDB

DIM = 32


class Emb:
    def encode(self, text):
        r = random.Random(hash(text) & 0xFFFFFFFF)
        v = [r.gauss(0, 1) for _ in range(DIM)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]


def _store(path):
    db = YantrikDB(str(path), DIM)
    db.set_embedder(Emb(), allow_unverified_embedder=True)
    return db


def _follower_with_replicated_rows(tmp_path):
    src = _store(tmp_path / "leader.db")
    for i in range(40):
        src.record(f"replicated record {i} about alice and topic {i % 5}")
    ops = src.extract_ops_since(limit=5000)
    src.close()

    dst = _store(tmp_path / "follower.db")
    for i in range(20):
        dst.record(f"local record {i} about topic {i % 5}")
    applied = dst.apply_ops(ops)
    assert applied["ops_applied"] == 40
    bare = [
        r["rid"]
        for r in dst._conn.execute(
            "SELECT rid FROM memories WHERE embedding IS NULL"
        ).fetchall()
    ]
    assert len(bare) == 40, "precondition: replicated rows carry no vector"
    return dst, bare


def test_think_after_apply_ops(tmp_path):
    db, _bare = _follower_with_replicated_rows(tmp_path)
    db.think()
    db.close()


def test_archive_hydrate_and_evict_replicated_rows(tmp_path):
    db, bare = _follower_with_replicated_rows(tmp_path)
    rid = bare[0]
    assert db.archive(rid) is True
    assert db.hydrate(rid) is True
    db.evict(10)
    db.close()


def test_refresh_embeddings_still_repairs_them(tmp_path):
    # refresh_embeddings() embeds with the engine's own embedder, so both
    # stores use the bundled one here.
    src = YantrikDB.with_default(str(tmp_path / "leader.db"))
    # 12, not 10: clear of think()'s min_active_memories threshold.
    for i in range(12):
        src.record(f"replicated record {i} about alice and topic {i % 5}")
    ops = src.extract_ops_since(limit=5000)
    src.close()
    db = YantrikDB.with_default(str(tmp_path / "follower.db"))
    db.apply_ops(ops)

    def bare():
        return db._conn.execute(
            "SELECT count(*) AS n FROM memories WHERE embedding IS NULL"
        ).fetchall()[0]["n"]

    assert bare() == 12, "precondition: replicated rows carry no vector"
    db.think()
    db.refresh_embeddings(dry_run=False)
    assert bare() == 0
    db.think()
    db.close()


def test_evict_then_refresh_while_cold_then_hydrate(tmp_path):
    # The chain that used to end in a zstd panic: replicated rows are
    # evicted to the cold tier, repaired there, then hydrated.
    src = YantrikDB.with_default(str(tmp_path / "leader.db"))
    for i in range(12):
        src.record(f"replicated record {i} about alice and topic {i % 5}")
    ops = src.extract_ops_since(limit=5000)
    src.close()
    db = YantrikDB.with_default(str(tmp_path / "follower.db"))
    db.apply_ops(ops)

    db.evict(2)
    cold = [
        r["rid"]
        for r in db._conn.execute(
            "SELECT rid FROM memories WHERE storage_tier = 'cold' "
            "AND embedding IS NULL"
        ).fetchall()
    ]
    assert cold, "precondition: eviction moved rows with no vector to cold"

    report = db.refresh_embeddings(dry_run=False)
    assert report["refreshed"] >= len(cold)
    for rid in cold:
        assert db.hydrate(rid) is True
    assert db._conn.execute(
        "SELECT count(*) AS n FROM memories WHERE embedding IS NULL"
    ).fetchall()[0]["n"] == 0
    db.close()


def test_refresh_at_follower_scale(tmp_path):
    # More replicated rows than the in-memory delta tier holds (256): the
    # repair used to fail with Backpressure after committing its SQL,
    # leaving rows out of vector recall with nothing left to retry.
    src = YantrikDB.with_default(str(tmp_path / "leader.db"))
    texts = [f"replicated record {i} about topic {i % 23} and the ferry" for i in range(1000)]
    for t in texts:
        src.record(t)
    ops = src.extract_ops_since(limit=5000)
    src.close()
    db = YantrikDB.with_default(str(tmp_path / "follower.db"))
    db.apply_ops(ops)

    db.evict(len(texts) - 20)
    cold = [
        r["rid"]
        for r in db._conn.execute(
            "SELECT rid FROM memories WHERE storage_tier = 'cold' LIMIT 10"
        ).fetchall()
    ]
    assert cold, "precondition: eviction moved rows to cold"
    for rid in cold:
        assert db.hydrate(rid) is True

    report = db.refresh_embeddings(dry_run=False)
    assert report["errors"] == []
    assert db._conn.execute(
        "SELECT count(*) AS n FROM memories WHERE embedding IS NULL"
    ).fetchall()[0]["n"] == 0
    for rid in cold:
        text = db.get(rid)["text"]
        hits = db.recall(query=text, top_k=10)
        assert any(h["rid"] == rid for h in hits), rid
    db.close()
