"""Issue #88 — `evidence_total` / `evidence_tombstoned` on `get_patterns`.

Asserted through the **Python binding** against an installed wheel, because the
contract is what a caller sees. The pattern rows are inserted with plain SQLite
(the engine has no public "create this pattern" entry point, and the miner needs
a corpus to produce one); everything read back goes through `get_patterns()`.

Scope of this patch: read-time counts only. `forget` and `upsert_pattern` are
untouched, so the re-mine test below is a guard, not a change.
"""
import hashlib
import sqlite3

import pytest

import yantrikdb

PATTERN_SQL = (
    "INSERT INTO patterns (pattern_id, pattern_type, status, confidence, description, "
    "evidence_rids, entity_names, context, first_seen, last_confirmed, occurrence_count, "
    "hlc, origin_actor) "
    "VALUES (?, ?, 'active', 0.5, 'synthetic', '[]', '[]', ?, 0.0, 0.0, 1, "
    "X'0000000000000000', 'test')"
)


class ToyEmbedder:
    """Deterministic 384-dim hashing embedder: hermetic, no model download."""

    dim = 384

    def encode(self, text):
        h = hashlib.sha256((text or "").encode()).digest()
        v = [(h[i % len(h)] / 255.0) - 0.5 for i in range(self.dim)]
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "m.db"
    db = yantrikdb.YantrikDB(db_path=str(path), embedding_dim=384)
    db.set_embedder(ToyEmbedder())
    return db, path


def _csv(conn, sql, args=()):
    with conn:
        conn.execute(sql, args)


def _seed_pattern(conn, pattern_id, pattern_type, rids):
    import json

    _csv(conn, PATTERN_SQL, (pattern_id, pattern_type, json.dumps({})))
    for rid in rids:
        _csv(
            conn,
            "INSERT OR IGNORE INTO pattern_evidence (pattern_id, rid) VALUES (?, ?)",
            (pattern_id, rid),
        )


def _record(db, texts):
    out = []
    for t in texts:
        r = db.record(
            text=t, memory_type="episodic", importance=0.5, valence=0.0,
            half_life=604800.0, metadata={}, embedding=ToyEmbedder().encode(t),
            namespace="default", source="user",
        )
        out.append(r if isinstance(r, str) else r["rid"])
    return out


def _row(db, pattern_id):
    return [p for p in db.get_patterns(limit=200) if p["pattern_id"] == pattern_id][0]


def test_forgetting_one_support_keeps_the_pattern_and_counts_it(store):
    """1 of N support records forgotten: pattern active, N total, 1 tombstoned."""
    db, path = store
    rids = _record(db, ["support a", "support b", "support c"])
    conn = sqlite3.connect(str(path), timeout=10)
    _seed_pattern(conn, "p1", "temporal_cluster", rids)
    conn.close()

    before = _row(db, "p1")
    assert before["evidence_total"] == 3
    assert before["evidence_tombstoned"] == 0

    db.forget(rids[0])

    after = _row(db, "p1")
    assert after["status"] == "active", "one dead support must not expire the pattern"
    assert after["evidence_total"] == 3, "the support was recorded and stays counted"
    assert after["evidence_tombstoned"] == 1


def test_untracked_pattern_type_reports_none_not_zero(store):
    """A miner that records no evidence reports null, not 0."""
    db, path = store
    conn = sqlite3.connect(str(path), timeout=10)
    _seed_pattern(conn, "p2", "entity_bridge", [])
    conn.close()

    row = _row(db, "p2")
    assert row["evidence_total"] is None
    assert row["evidence_tombstoned"] is None


def test_control_pattern_is_unaffected_by_another_patterns_loss(store):
    """The control owns a disjoint support set and must not move."""
    db, path = store
    rids = _record(db, ["s0", "s1", "s2", "s3"])
    conn = sqlite3.connect(str(path), timeout=10)
    _seed_pattern(conn, "subject", "temporal_cluster", rids[:2])
    _seed_pattern(conn, "control", "temporal_cluster", rids[2:])
    conn.close()

    db.forget(rids[0])

    subject = _row(db, "subject")
    control = _row(db, "control")
    assert (subject["evidence_total"], subject["evidence_tombstoned"]) == (2, 1)
    assert (control["evidence_total"], control["evidence_tombstoned"]) == (2, 0)


def test_remining_after_forget_creates_no_duplicate(store):
    """A re-mine after the forget does not duplicate the row or its evidence."""
    db, path = store
    rids = _record(db, ["m0", "m1", "m2"])
    conn = sqlite3.connect(str(path), timeout=10)
    _seed_pattern(conn, "dup", "temporal_cluster", rids)
    conn.close()
    db.forget(rids[0])

    db.think(config={"run_consolidation": False, "run_conflict_scan": False,
                     "run_pattern_mining": True})

    conn = sqlite3.connect(str(path), timeout=10)
    same_id = conn.execute(
        "SELECT COUNT(*) FROM patterns WHERE pattern_id = 'dup'"
    ).fetchone()[0]
    dup_evidence = conn.execute(
        "SELECT COUNT(*) FROM (SELECT pattern_id, rid FROM pattern_evidence "
        "GROUP BY pattern_id, rid HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    conn.close()

    assert same_id == 1, "the stored pattern row must survive the re-mine unchanged"
    assert dup_evidence == 0, "no duplicate (pattern_id, rid) pair after re-mining"
    assert _row(db, "dup")["evidence_tombstoned"] == 1

def test_untracked_type_with_evidence_rows_reports_integers(store):
    """F-01: real evidence rows win over the static type list.

    An `entity_bridge` pattern that does have a row must not report None —
    that would hide a row that exists. `None` means "no evidence recorded at
    all", not "this type is on a deny-list".
    """
    db, path = store
    rids = _record(db, ["e0"])
    conn = sqlite3.connect(str(path), timeout=10)
    _seed_pattern(conn, "bridge", "entity_bridge", rids)
    conn.close()

    row = _row(db, "bridge")
    assert (row["evidence_total"], row["evidence_tombstoned"]) == (1, 0)


def test_unknown_type_without_rows_reports_zero_not_none(store):
    """A type that is not on the deny-list is assumed to record evidence, so an
    empty pattern reports 0 (looked, found none) rather than None."""
    db, path = store
    conn = sqlite3.connect(str(path), timeout=10)
    _seed_pattern(conn, "future", "some_future_type", [])
    conn.close()

    row = _row(db, "future")
    assert (row["evidence_total"], row["evidence_tombstoned"]) == (0, 0)

