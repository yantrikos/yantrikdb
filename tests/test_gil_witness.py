"""Bindings that run engine work release the GIL while they do it (#271).

The concurrency tests in `test_gil_release_concurrency.py` prove that a detached
call holds no borrow of the wrapper. They cannot prove the GIL is released: with
the GIL held, another thread never runs during the call, so it never sees a
borrow either, and those tests pass on a build with no `py.detach` at all.

This file proves the release itself. With the switch interval at 100 s the
interpreter never forces a GIL handoff during a test, so an observer thread can
run while a call is in progress only if the call gives the GIL up itself. That
half is deterministic: a call that holds the GIL is seen exactly 0 times, on any
runner. The other half is a matter of attempts: the observer wakes every half
millisecond, so a call is seen once one of its detached stretches overlaps a
wake-up. Calls that spend tens of microseconds or more detached are seen within
a few hundred attempts; a detached stretch of a microsecond or two (a call that
fails at once, inside the detach) can be missed, and such a call is not
witnessed here. Controls keep the witness honest: a no-op must be seen 0 times,
a `sleep` at least once.

The "releases" half needs a second CPU. With one usable CPU, a detached stretch
of a few microseconds ends before the scheduler gets around to the observer,
and with the switch interval at 100 s nothing forces a handoff, so short calls
are missed: under `taskset -c 8` on Linux, 7 of 53 witnesses failed. Those
tests skip there, while the held half and the controls still run. The same
misses happen when all but one CPU is busy. CI runs pytest serially on runners
with 3 or more vCPUs, so it is not affected.

Call the binding bare. Wrapping it in anything that sleeps or blocks (a retry
with backoff, a lock) releases the GIL on the wrapper's behalf and fakes a pass.
"""

import os
import random
import sys
import threading
import time

import pytest

from yantrikdb import YantrikDB
from yantrikdb.consolidate import consolidate

DIM = 32  # not 64/256: those dims auto-attach a bundled engine-native embedder


def _usable_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not on Windows or macOS
        return os.cpu_count() or 1


# For tests that need to SEE a release (the module docstring explains why).
needs_a_second_cpu = pytest.mark.skipif(
    _usable_cpus() < 2,
    reason="seeing a short GIL release needs a second CPU for the observer",
)


class Emb:
    def encode(self, text):
        r = random.Random(hash(text) & 0xFFFFFFFF)
        v = [r.random() for _ in range(DIM)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]


class BlockingEmb(Emb):
    """Blocks inside `encode()` once armed, until released.

    Not armed during `set_embedder()`'s own probe, so attaching it does not block.
    """

    def __init__(self):
        self.armed = False
        self.entered = threading.Event()
        self.release = threading.Event()

    def encode(self, text):
        if self.armed:
            self.entered.set()
            if not self.release.wait(30):
                raise TimeoutError("BlockingEmb was never released")
        return super().encode(text)


@pytest.fixture
def db(tmp_path):
    store = YantrikDB(os.path.join(str(tmp_path), "witness.db"), DIM)
    store.set_embedder(Emb(), allow_unverified_embedder=True)
    for i in range(300):
        store.record(f"record {i} about topic {i % 31} and thing {i % 7}")
    yield store
    store.close()


def _seen_during(call, attempts=2000, interval=100):
    """How many times an observer thread ran while `call()` was in progress.

    Stops at the first sighting, so a call that releases the GIL costs a few
    attempts; only one that holds it runs all of them. 0 after `attempts` calls
    means the call never gave the GIL up for long enough to be seen.

    The pause between calls is jittered: a fixed pause can fall into step with
    the observer's own sleep, so every call starts while the observer is still
    asleep and a short release is missed every time (seen on Linux runners).

    Between calls the main thread sleeps, outside the window, and that is not
    optional. CPython forces a handoff once a waiting thread has waited a full
    switch interval with no switch, so a loop of calls that each hold the GIL
    would be seen as releasing it once their total time passed `interval` (300
    `think()` calls on 0.23.4 did exactly that). The sleep hands the GIL over
    between calls and restarts that clock, so a false sighting needs ONE call
    that holds the GIL for the whole interval.
    """
    state = {"in_call": False, "seen": 0, "stop": False}
    jitter = random.Random(271)

    def observer():
        while not state["stop"]:
            if state["in_call"]:
                state["seen"] += 1
            time.sleep(0.0005)

    old = sys.getswitchinterval()
    sys.setswitchinterval(interval)
    t = threading.Thread(target=observer, daemon=True)
    try:
        t.start()
        for _ in range(attempts):
            state["in_call"] = True
            call()
            state["in_call"] = False
            if state["seen"]:
                break
            time.sleep(0.001 + 0.001 * jitter.random())
    finally:
        state["in_call"] = False
        state["stop"] = True
        t.join()
        sys.setswitchinterval(old)
    return state["seen"]


# ── the witness's own controls ───────────────────────────────────────────


def test_witness_sees_nothing_through_a_call_that_holds_the_gil():
    def busy():
        sum(range(2000))

    assert _seen_during(busy) == 0


def test_witness_sees_nothing_through_held_calls_longer_in_total_than_the_interval():
    # The flaw the between-call sleep fixes, reproduced fast: calls that each
    # hold the GIL for 10 ms, a 50 ms switch interval, 100 calls (about 1 s,
    # twenty intervals). Without the sleep CPython forces a handoff mid-loop
    # and the witness reports a release that never happened.
    def held_10ms():
        end = time.perf_counter() + 0.010
        while time.perf_counter() < end:
            pass

    assert _seen_during(held_10ms, attempts=100, interval=0.05) == 0


def test_witness_sees_a_call_that_releases_the_gil():
    assert _seen_during(lambda: time.sleep(0.002)) >= 1


def test_witness_sees_nothing_through_a_binding_that_holds_the_gil(db):
    # A binding that still runs under the GIL, so the witness is checked on the
    # real extension too, not only on pure Python. If this binding is ever
    # detached, pick another that is not.
    rid = db.record("a record to read back")
    assert _seen_during(lambda: db.get(rid)) == 0


# ── the bindings #271 has detached so far ───────────────────────────────


@needs_a_second_cpu
def test_recall_with_an_embedding_releases_the_gil(db):
    q = Emb().encode("topic 3")
    assert _seen_during(lambda: db.recall(query_embedding=q, top_k=5)) >= 1


@needs_a_second_cpu
def test_recall_with_a_python_embedder_releases_the_gil(db):
    # encode() runs with the GIL held (it is Python); the search after it must not.
    assert _seen_during(lambda: db.recall(query="topic 3", top_k=5)) >= 1


@needs_a_second_cpu
def test_recall_with_the_engine_embedder_releases_the_gil(tmp_path):
    # 64 dims: the bundled engine-native embedder is attached, no Python one.
    store = YantrikDB(os.path.join(str(tmp_path), "native.db"), 64)
    try:
        if not store.has_embedder():
            pytest.skip("this build has no bundled engine embedder")
        for i in range(50):
            store.record(f"native record {i} about topic {i % 7}")
        assert _seen_during(lambda: store.recall(query="topic 3", top_k=5)) >= 1
    finally:
        store.close()


@needs_a_second_cpu
def test_stats_releases_the_gil(db):
    assert _seen_during(lambda: db.stats()) >= 1


@needs_a_second_cpu
def test_think_releases_the_gil(db):
    assert _seen_during(lambda: db.think()) >= 1


@needs_a_second_cpu
def test_scan_conflicts_releases_the_gil(db):
    assert _seen_during(lambda: db.scan_conflicts()) >= 1


@needs_a_second_cpu
def test_consolidate_releases_the_gil(db):
    assert _seen_during(lambda: consolidate(db)) >= 1


# ── no borrow is held across a Python embedder's encode() ────────────────


def _recall_in_a_thread_blocked_in_encode(db):
    emb = BlockingEmb()
    db.set_embedder(emb, allow_unverified_embedder=True)
    emb.armed = True
    out = {}

    def worker():
        try:
            out["results"] = db.recall(query="topic 3", top_k=5)
        except Exception as e:  # noqa: BLE001 - every failure is the finding
            out["error"] = repr(e)

    t = threading.Thread(target=worker)
    t.start()
    assert emb.entered.wait(10), "recall() never reached the embedder"
    return emb, t, out


def test_set_embedder_while_recall_is_inside_encode(db):
    # Before the shared snapshot, recall() held the wrapper's shared borrow
    # while the Python embedder ran, so this set_embedder() was refused with
    # "Already borrowed" for as long as encode() took.
    emb, t, out = _recall_in_a_thread_blocked_in_encode(db)
    try:
        db.set_embedder(Emb(), allow_unverified_embedder=True)
    finally:
        emb.release.set()
        t.join(30)
    assert not t.is_alive()
    assert "error" not in out, out.get("error")
    assert out["results"]


def test_close_with_nothing_in_flight_closes(db):
    # The documented contract: True when this call performed the exclusive
    # close. It returned False almost every time, because the idle compactor
    # held a strong engine ref across its wait (#271 review).
    assert db.close() is True
    assert db.close() is False  # already closed: idempotent, no second close


def test_close_while_recall_is_inside_encode(db):
    # close() mid-call drops the wrapper's handle and reports False, because
    # the in-flight recall() still holds the engine; it finishes on that
    # reference, and the engine closes when it goes.
    emb, t, out = _recall_in_a_thread_blocked_in_encode(db)
    try:
        assert db.close() is False
    finally:
        emb.release.set()
        t.join(30)
    assert not t.is_alive()
    assert "error" not in out, out.get("error")
    assert out["results"]


def test_close_under_write_load_keeps_other_threads_running(tmp_path):
    # close() stops the worker pool (joining a materializer mid-drain) and
    # closes SQLite (a WAL checkpoint). With the GIL held through both, a
    # Python thread beside it froze for the whole close: 15-20 ms idle, up
    # to 1.8 s under write load in the review. Detached, the thread runs.
    for attempt in range(5):
        store = YantrikDB(os.path.join(str(tmp_path), f"busy{attempt}.db"), DIM)
        store.set_embedder(Emb(), allow_unverified_embedder=True)
        for i in range(800):
            store.record(f"pending write {i} about topic {i % 13} and thing {i % 7}")
        gaps = []
        stop = threading.Event()

        def ticker():
            last = time.perf_counter()
            while not stop.is_set():
                time.sleep(0.002)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        t = threading.Thread(target=ticker)
        t.start()
        time.sleep(0.05)
        gaps.clear()
        started = time.perf_counter()
        store.close()
        took = time.perf_counter() - started
        time.sleep(0.05)
        stop.set()
        t.join()
        worst = max(gaps)
        if took < 0.04:
            continue  # too quick to tell a held GIL from a released one
        assert worst < took / 2, (
            f"the ticker stalled {worst * 1000:.0f} ms during a {took * 1000:.0f} ms close()"
        )
        return
    pytest.skip("close() never took long enough to measure on this machine")
