"""Engine calls that release the GIL must not lock other threads out of the wrapper.

`recall()` and `think()` run the engine inside `py.detach` (#271, #279, #280), so
other Python threads keep running while they work. `stats()` releases the GIL the
same way, and the third test below covers it. `scan_conflicts()` does the same,
and the fourth test below covers it. `consolidate()` does the same, and the
fifth test below covers it. The first version kept the
wrapper's `PyRef` borrow for the whole call. While a recall loop ran in one
thread, every `set_embedder()` from another thread failed with
`RuntimeError: Already borrowed` (3.5M of 3.5M calls in 5 s), and `close()`
during `think()` failed the same way. A host that shuts down while maintenance
runs would have hit this.

These tests use the call pattern a host actually has: a worker thread inside
the engine, and the main thread calling the wrapper's `&mut self` methods
(`close()` among them; `set_embedder()` was one too, before #282).
"""

import os
import random
import threading
import time

import pytest

from yantrikdb import YantrikDB
from yantrikdb.consolidate import consolidate

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
    # either. Any worker error at all is a finding: a borrow refusal
    # breaks the contract, and anything else is unexpected by definition.
    assert not recall_errors, recall_errors[:2]


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
    # Same #282 contract as recall(): any worker error is a finding.
    assert not stats_errors, stats_errors[:2]


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
    # Same #282 contract as recall(): any worker error is a finding.
    assert not scan_errors, scan_errors[:2]


def test_set_embedder_while_consolidate_runs_in_another_thread(db):
    # consolidate() is a module-level function taking the engine, not a
    # method; the borrow hazard is the same. Before the fix it held the
    # wrapper's shared borrow (and the GIL) for the whole pass, so every
    # set_embedder() from another thread failed "Already borrowed" while
    # a consolidation pass was running.
    stop = threading.Event()
    consol_calls = []
    consol_errors = []
    in_pass = [False]

    def worker():
        while not stop.is_set():
            consol_calls.append(1)
            in_pass[0] = True
            try:
                consolidate(db)
            except Exception as e:  # noqa: BLE001 - every failure is the finding
                consol_errors.append(repr(e))
            in_pass[0] = False

    t = threading.Thread(target=worker)
    t.start()
    errors = []
    calls = 0
    try:
        # Open the contention window only once the worker has provably
        # run: on a slow runner a fixed window can elapse before the
        # worker gets through even a few passes, and the count below
        # then fails with nothing actually wrong. Wait for a floor of
        # passes first, bounded so a wedged worker still fails the
        # assert instead of hanging the suite.
        warm_deadline = time.time() + 30.0
        while len(consol_calls) < 10 and time.time() < warm_deadline:
            time.sleep(0.005)
        # The warm-up proves the worker runs, not that it was inside a pass
        # while set_embedder() was called. Count the calls that overlapped
        # one: sampled per call, so a pass longer than the window still
        # counts, and a runner's speed cannot fail it.
        overlapped = 0
        deadline = time.time() + 2.0
        while time.time() < deadline:
            try:
                db.set_embedder(Emb(), allow_unverified_embedder=True)
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))
            calls += 1
            if in_pass[0]:
                overlapped += 1
    finally:
        stop.set()
        t.join()

    # The test only means something if both threads really ran concurrently.
    assert len(consol_calls) >= 10, "the consolidate worker never warmed up"
    assert overlapped >= 1, "no set_embedder() call overlapped a consolidate pass"
    assert calls > 5
    assert not _borrow_errors(errors), (
        f"{len(errors)}/{calls} set_embedder() calls failed while "
        f"consolidate() was detached: {errors[:1]}"
    )
    # Same #282 contract as recall(): neither direction may refuse the
    # other with a borrow error, and any other worker error is a finding.
    assert not consol_errors, consol_errors[:2]


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


def test_set_embedder_identity_check_with_recall_traffic(tmp_path):
    # The other tests pass allow_unverified_embedder=True, so the #117
    # identity check early-returns and only the probe path is exercised.
    # This arm runs the identity check for real — a store whose vectors
    # were built by the bundled engine embedder (recorded identity), and
    # an embedder declaring the matching fingerprint — while recall
    # traffic runs, and asserts the full check completes cleanly
    # alongside it. It does not prove phase 2 itself is contended:
    # nothing inside the check yields the GIL, and recall() takes only
    # shared borrows, which a shared borrow cannot refuse (maintainer
    # note on #295; the test is named for exactly what it proves).
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
    assert not errors, f"identity-checked set_embedder() failed during recall traffic: {errors[:1]}"
    assert not recall_errors, recall_errors[:2]


def test_set_embedder_raises_closed_when_probe_closes_the_store(tmp_path):
    # Deterministic, no threads (maintainer request on #295): the
    # embedder's encode() closes the store during the phase-1 probe.
    # The identity check passes trivially with
    # allow_unverified_embedder=True, so the phase-3 re-check is what
    # must refuse the store, raising the closed error. On main this
    # call returned None: the identity check returns Ok when the
    # engine is already gone, and the embedder landed on a closed
    # wrapper as if the call had succeeded.
    path = os.path.join(str(tmp_path), "gil-close-probe.db")
    store = YantrikDB(path, DIM)

    class ClosingEmb:
        def encode(self, text):
            store.close()
            return Emb().encode(text)

    with pytest.raises(RuntimeError, match="YantrikDB is closed"):
        store.set_embedder(ClosingEmb(), allow_unverified_embedder=True)
