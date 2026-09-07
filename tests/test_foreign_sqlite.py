"""Issue #225 — a second SQLite library with the store open in this process.

The engine bundles its own SQLite; Python's stdlib ``sqlite3`` is a second
one. Both rely on POSIX advisory locks, which the kernel scopes per
process, so the stdlib connection's unlock releases the engine's locks
and two writers interleave WAL commits: silent page aliasing (seen on
our CI 2026-09-06, reported from the field 2026-09-07). The engine cannot
make the other library respect its locks; it can notice the second
instance (Linux: the ``-shm`` file mapped twice at the same offset in
``/proc/self/maps``) and refuse to write while it is there.

These tests deliberately do the hazardous thing on a throwaway store,
with the engine idle between steps, so the corruption window never opens.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time

import pytest

import yantrikdb
from yantrikdb import YantrikDB

linux_only = pytest.mark.skipif(
    sys.platform not in ("linux", "darwin"),
    reason="the instance detector reads /proc/self/maps (Linux) or libproc regions (macOS)",
)


@pytest.fixture
def store():
    path = os.path.join(tempfile.mkdtemp(), "guard.db")
    db = YantrikDB.with_default(path)
    yield db, path
    db.close()


def test_mode_defaults_to_refuse_and_persists(store):
    db, path = store
    assert db.foreign_sqlite_mode() == "refuse"
    s = db.stats()
    assert s["foreign_sqlite_mode"] == "refuse"
    assert s["foreign_sqlite_supported"] == (sys.platform in ("linux", "darwin"))
    assert s["foreign_sqlite_active"] is False
    db.set_foreign_sqlite_mode("warn")
    db.close()
    again = YantrikDB.with_default(path)
    try:
        assert again.foreign_sqlite_mode() == "warn"
        with pytest.raises(Exception):
            again.set_foreign_sqlite_mode("enforce")
        again.set_foreign_sqlite_mode("refuse")
    finally:
        again.close()
    # The fixture closes the (already closed) handle; reopen so it can.
    store_db = YantrikDB.with_default(path)
    store_db.close()


@linux_only
def test_refuse_latches_until_the_engine_is_reopened(store):
    db, path = store
    rid = db.record("Alice Moreau works at Fennwick Labs.")
    db.think()  # let the materializer finish before the hazardous step
    assert not db.foreign_sqlite_detected()

    foreign = sqlite3.connect(path)
    try:
        # A WAL reader maps the -shm; that alone makes the process unsafe.
        foreign.execute("SELECT COUNT(*) FROM memories").fetchone()
        assert db.foreign_sqlite_detected() is True
        with pytest.raises(yantrikdb.ForeignSqliteInstance):
            db.record("this must not be committed")
        with pytest.raises(yantrikdb.ForeignSqliteInstance):
            db.correct(rid, "nor this")
        s = db.stats()
        assert s["foreign_sqlite_active"] is True
        assert s["foreign_sqlite_tainted"] is True
        assert s["foreign_sqlite_detected_since_boot"] >= 1
        assert s["foreign_sqlite_refused_since_boot"] >= 2
        # Reads keep working.
        assert db.recall(query="Alice Moreau", top_k=3, skip_reinforce=True)
    finally:
        foreign.close()

    # The foreign library's close unlinked the shm under the engine
    # (measured 2026-09-07): the store stays tainted for this instance.
    time.sleep(0.3)  # past the rescan window
    assert db.foreign_sqlite_detected() is False
    with pytest.raises(yantrikdb.ForeignSqliteInstance):
        db.record("still refused: reopen first")
    s = db.stats()
    assert s["foreign_sqlite_active"] is False and s["foreign_sqlite_tainted"] is True

    # A reopen recovers the WAL/shm cleanly and starts untainted.
    db.close()
    again = YantrikDB.with_default(path)
    try:
        assert again.stats()["foreign_sqlite_tainted"] is False
        assert again.record("Writes resume after a reopen.")
        assert again.recall(query="Alice Moreau", top_k=3, skip_reinforce=True)
    finally:
        again.close()
    store_db = YantrikDB.with_default(path)  # the fixture closes this one
    store_db.close()


@linux_only
def test_warn_counts_but_keeps_writing(store):
    db, path = store
    db.set_foreign_sqlite_mode("warn")
    db.record("seed")
    db.think()
    foreign = sqlite3.connect(path)
    try:
        foreign.execute("SELECT COUNT(*) FROM memories").fetchone()  # maps the -shm
        assert db.foreign_sqlite_detected() is True
        assert db.record("warn mode keeps writing")  # counted, not refused
        s = db.stats()
        assert s["foreign_sqlite_detected_since_boot"] >= 1
        assert s["foreign_sqlite_refused_since_boot"] == 0
    finally:
        foreign.close()


@linux_only
def test_a_separate_process_is_not_a_foreign_instance(store):
    db, path = store
    db.record("Separate processes are serialised by the kernel.")
    db.think()
    code = textwrap.dedent(
        f"""
        import sqlite3
        c = sqlite3.connect({path!r})
        print(c.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
        c.close()
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) >= 1
    assert db.foreign_sqlite_detected() is False
    assert db.record("still writing")


def test_a_commit_from_another_process_is_counted_and_checked(store):
    """The cross-process half: a commit that did not come through this
    engine (another engine process here) is noticed on the next write,
    queues an integrity check, and a clean check leaves writes alone."""
    db, path = store
    db.record("first, from this engine")
    db.record("second, still this engine")
    db.think()
    before = db.stats()
    assert before["foreign_commits_detected_since_boot"] == 0, before
    code = textwrap.dedent(
        f"""
        from yantrikdb import YantrikDB
        other = YantrikDB.with_default({path!r})
        other.record("from another process, through the engine")
        other.think()
        other.close()
        print("done")
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert db.record("third, after the other process") , "a foreign commit is not a refusal"
    s = db.stats()
    assert s["foreign_commits_detected_since_boot"] >= 1, s
    assert s["foreign_sqlite_tainted"] is False
    assert db.integrity_check() == "ok"
    s = db.stats()
    assert s["integrity_check_pending"] is False and s["last_integrity_check"] == "ok"
    assert s["integrity_checks_since_boot"] >= 1
    # The engine's own writes never count as foreign.
    n = s["foreign_commits_detected_since_boot"]
    db.record("fourth")
    db.record("fifth")
    assert db.stats()["foreign_commits_detected_since_boot"] == n
