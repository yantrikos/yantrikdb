"""Engine calls that release the GIL must not lock other threads out of the wrapper.

`recall()` and `think()` run the engine inside `py.detach` (#271, #279, #280), so
other Python threads keep running while they work. `stats()` releases the GIL the
same way, and the third test below covers it. `scan_conflicts()` does the same,
and the fourth test below covers it. `record()` does the same,
and the fifth test below covers it. The first version kept the
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
    # Pre-existing, unchanged by the GIL release and identical on 0.23.4:
    # set_embedder() keeps its mutable borrow while it calls the Python
    # embedder's encode(), so a recall() arriving mid-call is refused with
    # "Already mutably borrowed". That must stay an ordinary RuntimeError —
    # the first cut of this fix turned it into a Rust panic.
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
    # Same pre-existing contract as recall(): set_embedder() keeps its
    # mutable borrow while it calls the Python embedder's encode(), so a
    # stats() arriving mid-call is refused with "Already mutably borrowed".
    # That must stay an ordinary RuntimeError, never a panic.
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
    # Same pre-existing contract as recall(): set_embedder() keeps its
    # mutable borrow while it calls the Python embedder's encode(), so a
    # scan arriving mid-call is refused with "Already mutably borrowed".
    # That must stay an ordinary RuntimeError, never a panic.
    unexpected = [e for e in scan_errors if "Already mutably borrowed" not in e]
    assert not unexpected, unexpected[:2]
    assert not any("PanicException" in e for e in scan_errors)


def test_set_embedder_while_record_runs_in_another_thread(db):
    stop = threading.Event()
    record_calls = []
    record_errors = []

    def worker():
        n = 0
        while not stop.is_set():
            n += 1
            try:
                # Explicit embedding: the pure engine-write path. Retry
                # on ingest backpressure, as a real caller must — it is
                # queue flow control, not a borrow failure.
                db.record(f"record under load {n}", embedding=Emb().encode(f"load {n}"))
                record_calls.append(1)
            except Exception as e:  # noqa: BLE001 - every failure is the finding
                if "Backpressure" in type(e).__name__ or "queue full" in repr(e):
                    time.sleep(0.05)
                    continue
                record_errors.append(repr(e))

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
    assert len(record_calls) > 5, "the record worker barely ran; nothing was contended"
    assert calls > 5
    assert not _borrow_errors(errors), (
        f"{len(errors)}/{calls} set_embedder() calls failed while "
        f"record() was detached: {errors[:1]}"
    )
    # Same pre-existing contract as recall(): set_embedder() keeps its
    # mutable borrow while it calls the Python embedder's encode(), so a
    # record() arriving mid-call is refused with "Already mutably
    # borrowed". That must stay an ordinary RuntimeError, never a panic.
    unexpected = [e for e in record_errors if "Already mutably borrowed" not in e]
    assert not unexpected, unexpected[:2]
    assert not any("PanicException" in e for e in record_errors)
