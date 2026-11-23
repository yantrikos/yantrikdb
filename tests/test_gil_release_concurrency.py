"""Engine calls that release the GIL must not lock other threads out of the wrapper.

`recall()` and `think()` run the engine inside `py.detach` (#271, #279, #280), so
other Python threads keep running while they work. `stats()` releases the GIL the
same way, and the third test below covers it. `scan_conflicts()` does the same,
and the fourth test below covers it. The first version kept the
wrapper's `PyRef` borrow for the whole call. While a recall loop ran in one
thread, every `set_embedder()` from another thread failed with
`RuntimeError: Already borrowed` (3.5M of 3.5M calls in 5 s), and `close()`
during `think()` failed the same way. A host that shuts down while maintenance
runs would have hit this.

These tests use the call pattern a host actually has: a worker thread inside the
engine, and the main thread calling the `&mut self` methods.
"""

import os
import random
import threading
import time

import pytest

from yantrikdb import YantrikDB

DIM = 32  # not 64/256: those dims auto-attach a bundled engine-native embedder


class Emb:
    def encode(self, text):
        r = random.Random(hash(text) & 0xFFFFFFFF)
        v = [r.random() for _ in range(DIM)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]


@pytest.fixture
def db(tmp_path):
    store = YantrikDB(os.path.join(str(tmp_path), "gil.db"), DIM)
    store.set_embedder(Emb(), allow_unverified_embedder=True)
    for i in range(1500):
        store.record(f"record {i} about topic {i % 97} and thing {i % 13}")
    return store


def _borrow_errors(errors):
    return [e for e in errors if "Already borrowed" in e]


def test_set_embedder_while_recall_runs_in_another_thread(db):
    q = Emb().encode("topic 5")
    stop = threading.Event()
    recall_calls = []
    recall_errors = []

    def worker():
        while not stop.is_set():
            recall_calls.append(1)
            try:
                db.recall(query_embedding=q, top_k=10)
            except Exception as e:  # noqa: BLE001 - every failure is the finding
                recall_errors.append(repr(e))

    t = threading.Thread(target=worker)
    t.start()
    errors = []
    calls = 0
    deadline = time.time() + 2.0
    try:
        while time.time() < deadline:
            try:
                db.set_embedder(Emb(), allow_unverified_embedder=True)
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))
            calls += 1
    finally:
        stop.set()
        t.join()

    # The test only means something if both threads really ran concurrently.
    assert len(recall_calls) > 5, "the recall worker barely ran; nothing was contended"
    assert calls > 5
    assert not _borrow_errors(errors), (
        f"{len(errors)}/{calls} set_embedder() calls failed while recall() was "
        f"detached: {errors[:1]}"
    )
    # #282 contract: set_embedder() no longer holds a mutable borrow
    # across its probe, so a recall() arriving mid-call is NOT refused
    # either. Neither direction may refuse the other with a borrow error;
    # any other worker error is still a finding, and a panic always is.
    refusals = [e for e in recall_errors if "borrowed" in e]
    assert not refusals, refusals[:2]
    unexpected = [e for e in recall_errors if "Already mutably borrowed" not in e]
    assert not unexpected, unexpected[:2]
    assert not any("PanicException" in e for e in recall_errors)


def test_set_embedder_while_stats_runs_in_another_thread(db):
    stop = threading.Event()
    stats_calls = []
    stats_errors = []

    def worker():
        while not stop.is_set():
            stats_calls.append(1)
            try:
                db.stats()
            except Exception as e:  # noqa: BLE001 - every failure is the finding
                stats_errors.append(repr(e))

    t = threading.Thread(target=worker)
    t.start()
    errors = []
    calls = 0
    deadline = time.time() + 2.0
    try:
        while time.time() < deadline:
            try:
                db.set_embedder(Emb(), allow_unverified_embedder=True)
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))
            calls += 1
    finally:
        stop.set()
        t.join()

    # The test only means something if both threads really ran concurrently.
    assert len(stats_calls) > 5, "the stats worker barely ran; nothing was contended"
    assert calls > 5
    assert not _borrow_errors(errors), (
        f"{len(errors)}/{calls} set_embedder() calls failed while stats() was "
        f"detached: {errors[:1]}"
    )
    # Same #282 contract as recall(): neither direction may refuse the
    # other with a borrow error, and a panic never may happen.
    refusals = [e for e in stats_errors if "borrowed" in e]
    assert not refusals, refusals[:2]
    unexpected = [e for e in stats_errors if "Already mutably borrowed" not in e]
    assert not unexpected, unexpected[:2]
    assert not any("PanicException" in e for e in stats_errors)


def test_close_while_think_runs_in_another_thread(db):
    out = {}
    started = threading.Event()

    def worker():
        started.set()
        try:
            out["think"] = db.think()
        except Exception as e:  # noqa: BLE001
            out["think_err"] = repr(e)

    t = threading.Thread(target=worker)
    t.start()
    started.wait()
    time.sleep(0.05)
    try:
        out["close"] = db.close()
    except Exception as e:  # noqa: BLE001
        out["close_err"] = repr(e)
    t.join()

    assert "close_err" not in out, out.get("close_err")
    assert "think_err" not in out, out.get("think_err")
    # True if think() had already finished, False if it still held the engine.
    # Either is the documented close() contract; raising is not.
    assert out["close"] in (True, False)


def test_set_embedder_while_scan_conflicts_runs_in_another_thread(db):
    # Seed contradicting identity edges so each scan walks real candidate
    # groups end to end; on an edge-free store scan_conflicts() returns in
    # microseconds and nothing is contended.
    for i in range(20):
        db.relate(f"Person {i}", f"Company A{i % 50}", "works_at", 1.0)
        db.relate(f"Person {i}", f"Company B{i % 50}", "works_at", 1.0)

    stop = threading.Event()
    scan_calls = []
    scan_errors = []

    def worker():
        while not stop.is_set():
            scan_calls.append(1)
            try:
                db.scan_conflicts()
            except Exception as e:  # noqa: BLE001 - every failure is the finding
                scan_errors.append(repr(e))

    t = threading.Thread(target=worker)
    t.start()
    errors = []
    calls = 0
    deadline = time.time() + 2.0
    try:
        while time.time() < deadline:
            try:
                db.set_embedder(Emb(), allow_unverified_embedder=True)
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))
            calls += 1
    finally:
        stop.set()
        t.join()

    # The test only means something if both threads really ran concurrently.
    assert len(scan_calls) > 5, "the scan worker barely ran; nothing was contended"
    assert calls > 5
    assert not _borrow_errors(errors), (
        f"{len(errors)}/{calls} set_embedder() calls failed while "
        f"scan_conflicts() was detached: {errors[:1]}"
    )
    # Same #282 contract as recall(): neither direction may refuse the
    # other with a borrow error, and a panic never may happen.
    refusals = [e for e in scan_errors if "borrowed" in e]
    assert not refusals, refusals[:2]
    unexpected = [e for e in scan_errors if "Already mutably borrowed" not in e]
    assert not unexpected, unexpected[:2]
    assert not any("PanicException" in e for e in scan_errors)


class _DeclaringEmb:
    """An embedder that declares the fingerprint it is given (#117)."""

    def __init__(self, fingerprint, dim=64):
        self.fingerprint = fingerprint
        self.dim = dim

    def encode(self, text):
        r = random.Random(hash(text) & 0xFFFFFFFF)
        v = [r.random() for _ in range(self.dim)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]


def test_set_embedder_identity_check_under_recall_load(tmp_path):
    # The other tests pass allow_unverified_embedder=True, so the #117
    # identity check early-returns and only the probe path is exercised.
    # This arm runs the identity check for real — a store whose vectors
    # were built by the bundled engine embedder (recorded identity), and
    # an embedder declaring the matching fingerprint — under the same
    # concurrent recall load, asserting the #282 contract holds on the
    # phase that runs under a shared borrow too.
    import re

    path = os.path.join(str(tmp_path), "gil-identity.db")
    store = YantrikDB(path, 64)  # 64 dims: the bundled engine embedder
    for i in range(50):
        store.record(f"identity record {i} about topic {i % 7}")

    # Harvest the recorded digest from the refusal message itself: a
    # wrong declared fingerprint is refused and the error names the
    # recorded digest. Public API only; no second SQLite in-process.
    with pytest.raises(RuntimeError) as excinfo:
        store.set_embedder(_DeclaringEmb("sha256:wrong"))
    m = re.search(r"digest ([^,]+),", str(excinfo.value))
    assert m, str(excinfo.value)
    digest = m.group(1)

    q = _DeclaringEmb(digest).encode("topic 3")
    stop = threading.Event()
    recall_calls = []
    recall_errors = []

    def worker():
        while not stop.is_set():
            recall_calls.append(1)
            try:
                store.recall(query_embedding=q, top_k=5)
            except Exception as e:  # noqa: BLE001 - every failure is the finding
                recall_errors.append(repr(e))

    t = threading.Thread(target=worker)
    t.start()
    errors = []
    calls = 0
    deadline = time.time() + 2.0
    try:
        while time.time() < deadline:
            try:
                # No allow_unverified: the identity check runs in full.
                store.set_embedder(_DeclaringEmb(digest))
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))
            calls += 1
    finally:
        stop.set()
        t.join()
        store.close()

    assert len(recall_calls) > 5, "the recall worker barely ran; nothing was contended"
    assert calls > 5
    assert not errors, f"identity-checked set_embedder() failed under load: {errors[:1]}"
    refusals = [e for e in recall_errors if "borrowed" in e]
    assert not refusals, refusals[:2]
    assert not any("PanicException" in e for e in recall_errors)
