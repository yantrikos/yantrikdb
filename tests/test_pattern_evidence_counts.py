"""Issue #88 — `evidence_total` / `evidence_tombstoned` on `get_patterns`.

Asserted through the **Python binding** against an installed wheel, because the
contract is what a caller sees.

Every raw SQL touch runs in a child process (`tests/store_sql.py`): the engine
links its own SQLite and a second in-process library does not serialise with
it, which is the hazard `tests/test_foreign_sqlite.py` (#225) documents. The
pattern rows are seeded that way because no engine API creates a synthetic
pattern; every *assertion* reads through `get_patterns()` unless it needs a
column the API does not expose.

Scope of this patch: read-time counts only. `forget` and `upsert_pattern` are
untouched, so the re-mine test is a guard, not a change.
"""
import hashlib

import pytest

import yantrikdb
from tests.store_sql import query_in_subprocess, raw_sql_in_subprocess

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


def _seed_pattern(path, pattern_id, pattern_type, rids):
    raw_sql_in_subprocess(path, PATTERN_SQL, (pattern_id, pattern_type, "{}"))
    for rid in rids:
        raw_sql_in_subprocess(
            path,
            "INSERT OR IGNORE INTO pattern_evidence (pattern_id, rid) VALUES (?, ?)",
            (pattern_id, rid),
        )


def _evidence_count(path, pattern_id):
    rows = query_in_subprocess(
        path,
        "SELECT COUNT(*) FROM pattern_evidence WHERE pattern_id = ?",
        (pattern_id,),
    )
    return rows[0][0]


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
    return next(
        p for p in db.get_patterns(limit=200) if p["pattern_id"] == pattern_id
    )


def test_forgetting_one_support_keeps_the_pattern_and_counts_it(store):
    """1 of N support records forgotten: pattern active, N total, 1 tombstoned."""
    db, path = store
    rids = _record(db, ["support a", "support b", "support c"])
    _seed_pattern(path, "p1", "temporal_cluster", rids)

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
    _seed_pattern(path, "p2", "entity_bridge", [])

    row = _row(db, "p2")
    assert row["evidence_total"] is None
    assert row["evidence_tombstoned"] is None


def test_untracked_type_with_evidence_rows_reports_integers(store):
    """F-01: real evidence rows win over the static type list.

    An `entity_bridge` pattern that does have a row must not report None —
    that would hide a row that exists.
    """
    db, path = store
    rids = _record(db, ["e0"])
    _seed_pattern(path, "bridge", "entity_bridge", rids)

    row = _row(db, "bridge")
    assert (row["evidence_total"], row["evidence_tombstoned"]) == (1, 0)


def test_unknown_type_without_rows_reports_zero_not_none(store):
    """A type not on the deny-list is assumed to record evidence, so an empty
    pattern reports 0 (looked, found none) rather than None."""
    db, path = store
    _seed_pattern(path, "future", "some_future_type", [])

    row = _row(db, "future")
    assert (row["evidence_total"], row["evidence_tombstoned"]) == (0, 0)


def test_control_pattern_is_unaffected_by_another_patterns_loss(store):
    """The control owns a disjoint support set and must not move."""
    db, path = store
    rids = _record(db, ["s0", "s1", "s2", "s3"])
    _seed_pattern(path, "subject", "temporal_cluster", rids[:2])
    _seed_pattern(path, "control", "temporal_cluster", rids[2:])

    db.forget(rids[0])

    subject = _row(db, "subject")
    control = _row(db, "control")
    assert (subject["evidence_total"], subject["evidence_tombstoned"]) == (2, 1)
    assert (control["evidence_total"], control["evidence_tombstoned"]) == (2, 0)


def test_remining_after_forget_creates_no_duplicate(store):
    """A re-mine after the forget does not duplicate the row or its evidence."""
    db, path = store
    rids = _record(db, ["m0", "m1", "m2"])
    _seed_pattern(path, "dup", "temporal_cluster", rids)
    db.forget(rids[0])

    db.think(config={"run_consolidation": False, "run_conflict_scan": False,
                     "run_pattern_mining": True})

    same_id = query_in_subprocess(
        path, "SELECT COUNT(*) FROM patterns WHERE pattern_id = 'dup'"
    )[0][0]
    dup_evidence = query_in_subprocess(
        path,
        "SELECT COUNT(*) FROM (SELECT pattern_id, rid FROM pattern_evidence "
        "GROUP BY pattern_id, rid HAVING COUNT(*) > 1)",
    )[0][0]

    assert same_id == 1, "the stored pattern row must survive the re-mine unchanged"
    assert dup_evidence == 0, "no duplicate (pattern_id, rid) pair after re-mining"
    assert _evidence_count(path, "dup") == 3, (
        "re-mining must not add or drop evidence rows for the stored pattern"
    )
    assert _row(db, "dup")["evidence_tombstoned"] == 1
