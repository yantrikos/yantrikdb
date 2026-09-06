"""recall_thread v2 binding tests: typed exceptions, result shapes, and
the v1/v2 divergence pin (reviewer batch 11).

Requires the built extension (maturin develop / installed wheel); skipped
otherwise, like the other binding suites.
"""

import pytest

yantrikdb = pytest.importorskip("yantrikdb")
from yantrikdb import YantrikDB  # noqa: E402


def _vec(seed, dim=8):
    raw = [(seed + i) * 0.1 for i in range(dim)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


@pytest.fixture
def db():
    return YantrikDB(":memory:", 8)


def _seed(db, text, turn, seed):
    rid = db.record(
        text=text,
        memory_type="episodic",
        importance=0.5,
        metadata={"source_turn": turn},
        embedding=_vec(seed),
    )
    db.link_memory_entity(rid, "Alpha")
    return rid


def _raw_sql_in_subprocess(path, sql, params):
    """Run one raw write against an engine store from a separate process.

    NOT an in-process ``sqlite3.connect``: the engine links its own SQLite
    and its materializer thread may be mid-write. Two SQLite libraries in
    one process both use POSIX advisory locks, which the kernel scopes per
    process, so their writers never serialise and a WAL commit from one
    can land on top of the other's (sqlite.org/howtocorrupt.html, "multiple
    copies of SQLite linked into the same application"). Measured on the
    0.18.0 and 0.19.0 wheels under CPU contention: 2-8 % of runs left the
    memory row holding an entities page (rid = the entity name, text = a
    timestamp), and the py3.10 CI job hit it twice in one day. A child
    process holds its own locks, so the kernel serialises it against the
    engine like any other client.
    """
    import json
    import subprocess
    import sys

    prog = "; ".join([
        "import json, sqlite3, sys",
        "path, sql, params = json.loads(sys.argv[1])",
        "c = sqlite3.connect(path)",
        "c.execute(sql, params)",
        "c.commit()",
        "c.close()",
    ])
    subprocess.run(
        [sys.executable, "-c", prog, json.dumps([path, sql, params])],
        check=True,
        timeout=60,
    )


class TestThreadV2Binding:
    def test_typed_classes_are_exported(self):
        import yantrikdb as y

        for name in (
            "PhraseRouteUnavailableError",
            "SourceTurnMaintenanceRequiredError",
            "InvalidThreadTopicError",
        ):
            cls = getattr(y, name)
            assert issubclass(cls, RuntimeError), name

    def test_entity_only_recall_thread_keeps_legacy_shape(self, db):
        _seed(db, "Alpha kickoff", 1, 1.0)
        out = db.recall_thread("default", ["Alpha"])
        assert set(out.keys()) == {"items", "total", "omitted"}, (
            "entity-only recall_thread keeps the exact legacy v1 dict shape"
        )
        assert "routes" not in out["items"][0]

    def test_recall_thread_v2_is_always_v2(self, db):
        _seed(db, "Alpha kickoff", 1, 1.0)
        # Entity-only call STILL takes the v2 path: richer shape.
        out = db.recall_thread_v2("default", ["Alpha"])
        assert out["returned"] == len(out["items"]) == 1
        assert out["items"][0]["routes"] == ["entity"]
        assert out["items"][0]["phrases"] == []
        assert out["items"][0]["topic_rids"] == []
        # All-empty query: valid, empty, returned=0 — not a fault.
        empty = db.recall_thread_v2("default")
        assert (empty["total"], empty["returned"], empty["omitted"]) == (0, 0, 0)

    def test_multi_route_recall_thread_returns_richer_shape(self, db):
        _seed(db, "Alpha ran the quarterly sync", 2, 2.0)
        out = db.recall_thread(
            "default", ["Alpha"], phrases=["quarterly sync"]
        )
        assert out["returned"] == 1
        item = out["items"][0]
        assert item["routes"] == ["entity", "phrase"]
        assert item["phrases"] == ["quarterly sync"]

    def test_divergence_pin_stale_marker(self, tmp_path):
        """(batch 11 test 3) On a stale-marker store: v2 raises the typed
        maintenance error even entity-only; legacy v1 still succeeds with
        decrypt-derived semantics. The two surfaces must never converge."""
        import yantrikdb as y

        path = str(tmp_path / "stale.db")
        db = YantrikDB(path, 8)
        rid = _seed(db, "Alpha event five", 5, 1.0)
        # Raw SQL rewrite from ANOTHER PROCESS (exactly the raw-write
        # staleness the marker triggers exist to catch): turn 5 -> 7.
        _raw_sql_in_subprocess(
            path,
            "UPDATE memories SET metadata = "
            "json_set(metadata, '$.source_turn', 7) WHERE rid = ?",
            [rid],
        )

        # (a) v2, entity-only: typed MaintenanceRequired.
        with pytest.raises(y.SourceTurnMaintenanceRequiredError):
            db.recall_thread_v2("default", ["Alpha"])

        # (b) legacy v1 on the SAME engine: succeeds, decrypt-derived turn.
        out = db.recall_thread("default", ["Alpha"])
        assert out["total"] == 1
        assert out["items"][0]["source_turn"] == 7

        # Maintenance heals; v2 then serves the recomputed value.
        while not db.maintain_source_turn_backfill(1000)["complete"]:
            pass
        healed = db.recall_thread_v2("default", ["Alpha"])
        assert healed["items"][0]["source_turn"] == 7

    def test_phrase_route_unavailable_is_typed_on_encrypted(self):
        import yantrikdb as y

        db = YantrikDB(":memory:", 8, encryption_key=bytes(range(32)))
        _seed(db, "Alpha secret", 1, 1.0)
        with pytest.raises(y.PhraseRouteUnavailableError):
            db.recall_thread_v2("default", ["Alpha"], phrases=["secret"])

    def test_invalid_topic_is_typed(self, db):
        import yantrikdb as y

        _seed(db, "Alpha row", 1, 1.0)
        with pytest.raises(y.InvalidThreadTopicError):
            db.recall_thread_v2("default", ["Alpha"], topic_rids=["no-such-rid"])
