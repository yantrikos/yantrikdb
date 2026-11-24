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


# ── a call waiting on another process's write lock ───────────────────────
#
# A detached write that waits on SQLite's lock holds the engine for as long as
# it waits (up to busy_timeout). Under the GIL it had finished before any other
# call started; now `close()` and `set_embedder_named()` can arrive mid-wait.

_HOLDER = r"""
import sqlite3, sys, time
conn = sqlite3.connect(sys.argv[1], timeout=30, isolation_level=None)
conn.execute("BEGIN IMMEDIATE")
print("locked", flush=True)
time.sleep(float(sys.argv[2]))
conn.execute("COMMIT")
"""


def _hold_write_lock(path, seconds):
    # Another process, as in production: two SQLite libraries in one process
    # can corrupt a store.
    import subprocess
    import sys

    p = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, path, str(seconds)], stdout=subprocess.PIPE, text=True
    )
    assert p.stdout.readline().strip() == "locked"
    return p


def _turn_in_a_thread(db):
    out = {}

    def worker():
        try:
            out["id"] = db.record_turn("default", "user", "said while the lock was held")
        except Exception as e:  # noqa: BLE001 - every failure is the finding
            out["error"] = repr(e)

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.3)  # record_turn() is now waiting on the lock, detached
    return t, out


def test_close_while_a_write_waits_on_the_lock_lets_the_write_land(tmp_path):
    # A pin, not a regression test: it also passes on 0.23.6, where the
    # waiting write held the GIL, so close() ran only after it, and close()
    # returned False anyway (the idle compactor held the engine). It keeps
    # the detached behaviour honest: close() mid-write reports False, and
    # the write that was waiting still lands.
    path = os.path.join(str(tmp_path), "lockwait.db")
    db = YantrikDB(path, DIM)
    db.record_turn("default", "user", "before")
    holder = _hold_write_lock(path, 1.5)
    t, out = _turn_in_a_thread(db)
    try:
        # The waiting write still holds the engine, so this close() is not the
        # exclusive one.
        assert db.close() is False
    finally:
        t.join(30)
        holder.wait(30)
    assert "error" not in out, out.get("error")
    reopened = YantrikDB(path, DIM)
    try:
        said = [turn["content"] for turn in reopened.recent_turns("default")]
        assert "said while the lock was held" in said
    finally:
        reopened.close()


def _model_is_cached(name, tag):
    # dirs::cache_dir(), as the engine resolves it.
    import sys

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", "")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    d = os.path.join(base, "yantrikdb", "models", f"{name}-{tag}")
    files = ("model.safetensors", "tokenizer.json", "config.json", "modules.json")
    return all(os.path.isfile(os.path.join(d, f)) and os.path.getsize(os.path.join(d, f)) for f in files)


@pytest.mark.skipif(
    not _model_is_cached("potion-base-8M", "v0.1.0"),
    reason="potion-base-8M is not in the local model cache; this test never downloads",
)
def test_set_embedder_named_waits_for_a_write_waiting_on_the_lock(tmp_path):
    # Before the wait, the swap found the waiting write's reference and raised
    # "requires exclusive access … drop any ConnectionProxy", which named the
    # wrong cause and failed a call that succeeds sequentially.
    path = os.path.join(str(tmp_path), "named.db")
    db = YantrikDB(path, 256)
    try:
        db.record_turn("default", "user", "before")
        holder = _hold_write_lock(path, 1.5)
        t, out = _turn_in_a_thread(db)
        try:
            db.set_embedder_named("potion-base-8M")
        finally:
            t.join(30)
            holder.wait(30)
        assert "error" not in out, out.get("error")
        said = [turn["content"] for turn in db.recent_turns("default")]
        assert "said while the lock was held" in said
        assert db.has_embedder()
    finally:
        db.close()


@pytest.mark.skipif(
    not _model_is_cached("potion-base-8M", "v0.1.0"),
    reason="potion-base-8M is not in the local model cache; this test never downloads",
)
def test_set_embedder_named_names_both_causes_when_a_proxy_holds_the_engine(tmp_path):
    # A live _conn proxy never lets go, so the wait runs out (10 s) and the
    # error must name it as well as an in-flight call.
    db = YantrikDB(os.path.join(str(tmp_path), "proxy.db"), 256)
    try:
        proxy = db._conn
        spawns = db._worker_spawns
        with pytest.raises(RuntimeError) as err:
            db.set_embedder_named("potion-base-8M")
        msg = str(err.value)
        assert "ConnectionProxy" in msg and "another thread's call" in msg, msg
        assert "  " not in msg, f"broken line continuations in the message: {msg!r}"
        # The engine never looked free, so the workers ran through the whole
        # wait: never stopped, never restarted. They used to be stopped for
        # the whole wait, and the calls it waited for could need them.
        assert db._worker_spawns - spawns == 0, db._worker_spawns - spawns
        del proxy
        assert db.record_turn("default", "user", "workers still running") > 0
    finally:
        db.close()


@pytest.mark.skipif(
    not _model_is_cached("potion-base-8M", "v0.1.0"),
    reason="potion-base-8M is not in the local model cache; this test never downloads",
)
@pytest.mark.parametrize("traffic", ["recall", "record_turn", "recall_text"])
def test_set_embedder_named_succeeds_under_steady_traffic(tmp_path, traffic):
    # Four threads keep calling the store. Each detached call holds the
    # engine while it runs, and new ones keep starting, so a swap that only
    # waited for the engine to come free could wait forever (it gave up after
    # 10 s where 0.23.6 swapped in 1-2 s). The swap gate holds new calls at
    # their snapshot while the ones in flight finish; they then run on the
    # swapped engine, and none of them fails.
    db = YantrikDB(os.path.join(str(tmp_path), f"traffic-{traffic}.db"), 256)
    try:
        if traffic != "record_turn":
            for i in range(300):
                db.record_text(f"seed {i} about topic {i % 13} and thing {i % 7}")
        q = db.embed("topic 3")
        ops = {
            "recall": lambda k: db.recall(query_embedding=q, top_k=10),
            "record_turn": lambda k: db.record_turn(f"ns{k}", "user", "hello"),
            "recall_text": lambda k: db.recall_text("topic 3", 5),
        }
        stop = threading.Event()
        errors = []
        calls = [0]

        def worker(k):
            while not stop.is_set():
                try:
                    ops[traffic](k)
                    calls[0] += 1
                except Exception as e:  # noqa: BLE001 - every failure is the finding
                    errors.append(repr(e))

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
        for t in threads:
            t.start()
        time.sleep(0.3)
        started = time.perf_counter()
        try:
            db.set_embedder_named("potion-base-8M")
        finally:
            took = time.perf_counter() - started
            # Wait for the traffic to resume rather than sample a fixed
            # window: under load four threads finished as few as 4 calls in
            # 0.2 s, so a window that short can see none.
            before = calls[0]
            resumed_by = time.perf_counter() + 5
            while calls[0] == before and time.perf_counter() < resumed_by:
                time.sleep(0.01)
            after = calls[0]
            stop.set()
            for t in threads:
                t.join(30)
        seen = f"swap {took:.2f} s, {before} calls before it, {after} after"
        assert not errors, (seen, errors[:3])
        assert took < 8, seen
        assert after > before, f"traffic never resumed after the swap: {seen}"
    finally:
        db.close()


class _YieldingEmb(Emb):
    # Gives the GIL up inside encode(), as a real model does.
    def encode(self, text):
        time.sleep(0.001)
        return super().encode(text)


def test_close_beside_recalls_with_a_python_embedder_is_not_refused(tmp_path):
    # recall() takes the wrapper's shared borrow only for its snapshot; the
    # Python embedder's encode() runs with no borrow held. close() takes the
    # mutable borrow and retries briefly instead of failing at once. On
    # 0.23.6 recall() kept its borrow through encode(), and a close() that
    # landed then was refused with "Already borrowed".
    #
    # record() still holds its borrow across encode() until #297 lands, and
    # a tight record() loop leaves no gap with the GIL free, so close()
    # beside one can still be refused; that is #297's to fix.
    for attempt in range(5):
        db = YantrikDB(os.path.join(str(tmp_path), f"yield{attempt}.db"), DIM)
        db.set_embedder(_YieldingEmb(), allow_unverified_embedder=True)
        for i in range(50):
            db.record(f"record {i} about topic {i % 7}")
        stop = threading.Event()
        errors = []

        def reader():
            while not stop.is_set():
                try:
                    db.recall(query="topic 3", top_k=5)
                except Exception as e:  # noqa: BLE001
                    if "closed" not in str(e):
                        errors.append(repr(e))
                    return

        t = threading.Thread(target=reader)
        t.start()
        time.sleep(0.05)
        try:
            db.close()
        finally:
            stop.set()
            t.join(30)
        assert not errors, errors[:2]


_needs_potion = pytest.mark.skipif(
    not _model_is_cached("potion-base-8M", "v0.1.0"),
    reason="potion-base-8M is not in the local model cache; this test never downloads",
)


class _SlowEncodeEmb(Emb):
    # Once armed, its next encode() takes `hold` seconds with the GIL given
    # up. record() runs encode() holding the wrapper's shared borrow (until
    # #297), so that record() holds the borrow the whole time. (Since #295
    # set_embedder() holds no borrow across its own probe.)
    def __init__(self, hold, dim=DIM):
        self.hold = hold
        self.dim = dim
        self.armed = False
        self.entered = threading.Event()

    def encode(self, text):
        if self.armed:
            self.armed = False
            self.entered.set()
            time.sleep(self.hold)
        r = random.Random(hash(text) & 0xFFFFFFFF)
        v = [r.random() for _ in range(self.dim)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]


@_needs_potion
def test_set_embedder_named_beside_a_held_borrow_leaves_the_workers_running(tmp_path):
    # The swap stops the workers once the engine looks free, then takes the
    # wrapper's mutable borrow. Another thread can hold the borrow then: a
    # record() inside its Python embedder's encode(), as here. The swap took
    # the borrow with `?` after
    # stopping the workers, and a refusal skipped the respawn: the store
    # was left without workers, and writes past the 256th failed with
    # Backpressure ("ingest queue full"). Now the workers are restarted, the
    # swap tries again at most every 50 ms, and it goes through once the
    # borrow is free.
    db = YantrikDB(os.path.join(str(tmp_path), "held.db"), 256)
    try:
        proxy = db._conn  # holds the engine: the swap is waiting when the borrow is taken
        spawns = db._worker_spawns
        out = {}

        def swapper():
            started = time.perf_counter()
            try:
                db.set_embedder_named("potion-base-8M")
            except Exception as e:  # noqa: BLE001 - the failure is the finding
                out["error"] = repr(e)
            out["took"] = time.perf_counter() - started

        sw = threading.Thread(target=swapper)
        sw.start()
        time.sleep(0.5)
        # Held for longer than the old borrow retry (1.3-2.4 s measured).
        emb = _SlowEncodeEmb(3.5, dim=256)
        db.set_embedder(emb, allow_unverified_embedder=True)
        emb.armed = True
        holder = threading.Thread(target=lambda: db.record("a write held across encode()"))
        holder.start()
        assert emb.entered.wait(10)
        del proxy  # the engine is free; only the held borrow stands in the way
        holder.join(30)
        sw.join(30)
        assert not sw.is_alive()
        assert "error" not in out, out.get("error")
        for i in range(700):
            db.record(f"after the swap {i}", embedding=[0.05] * 256)
        # Restarted at most once per 50 ms try, and once at the end.
        restarts = db._worker_spawns - spawns
        assert restarts <= out["took"] / 0.05 + 3, (restarts, out["took"])
    finally:
        db.close()


def test_a_refused_close_gives_up_after_about_a_second(tmp_path):
    # The mutable borrow is retried for a second, by the clock. It was 1000
    # one-millisecond sleeps, which ran 1.3-2.4 s on Windows. The refused
    # close() leaves the store as it was. The borrow is held by a record()
    # inside its Python embedder's encode().
    db = YantrikDB(os.path.join(str(tmp_path), "refused.db"), DIM)
    emb = _SlowEncodeEmb(3.0)
    db.set_embedder(emb, allow_unverified_embedder=True)
    emb.armed = True
    t = threading.Thread(target=lambda: db.record("a write held across encode()"))
    t.start()
    try:
        assert emb.entered.wait(10)
        started = time.perf_counter()
        with pytest.raises(RuntimeError, match="Already borrowed"):
            db.close()
        took = time.perf_counter() - started
    finally:
        t.join(30)
    assert 0.9 <= took < 1.25, f"the refused close() took {took:.2f} s"
    db.record("still open", embedding=[0.1] * DIM)
    assert db.close() is True


@_needs_potion
def test_calls_wait_at_the_gate_for_as_long_as_the_swap_waits(tmp_path):
    # While the swap waits, new calls wait at their snapshot until it is done.
    # They used to go ahead after 2 s of their own, each taking the
    # reference the gate is there to keep out, so a swap held up for longer
    # was starved again by the traffic it had parked. Here a proxy holds the
    # engine for 4 s; no call may finish between the gate going up and the
    # proxy going away.
    db = YantrikDB(os.path.join(str(tmp_path), "gate.db"), 256)
    try:
        for i in range(100):
            db.record(f"seed {i} about topic {i % 13}", embedding=[0.05 + 0.001 * (i % 7)] * 256)
        q = [0.05] * 256
        proxy = db._conn
        stop = threading.Event()
        done = []
        errors = []

        def worker():
            while not stop.is_set():
                try:
                    db.recall(query_embedding=q, top_k=5)
                    done.append(time.perf_counter())
                except Exception as e:  # noqa: BLE001
                    errors.append(repr(e))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        time.sleep(0.2)
        out = {}

        def swapper():
            try:
                db.set_embedder_named("potion-base-8M")
            except Exception as e:  # noqa: BLE001
                out["error"] = repr(e)

        started = time.perf_counter()
        sw = threading.Thread(target=swapper)
        sw.start()
        time.sleep(4.0)
        released = time.perf_counter()
        del proxy
        sw.join(30)
        time.sleep(0.2)
        stop.set()
        for t in threads:
            t.join(30)
        assert "error" not in out, out.get("error")
        assert not errors, errors[:3]
        # The fetch takes well under 1.5 s from the cache; from then on the
        # gate is up until the proxy goes.
        during = [d for d in done if started + 1.5 <= d < released - 0.05]
        assert not during, f"{len(during)} calls finished while the swap waited"
        assert any(d > released for d in done), "traffic resumed after the swap"
    finally:
        db.close()


@_needs_potion
def test_close_during_a_pending_swap_leaves_the_still_shared_engine_its_workers(tmp_path):
    # close() with a proxy alive keeps the engine's workers running: the
    # engine still takes writes through the proxy. With a swap pending, the
    # swap then saw the engine free (only the proxy held it), stopped the
    # workers and found the wrapper closed, leaving them stopped (#302
    # review 4).
    db = YantrikDB(os.path.join(str(tmp_path), "close-swap.db"), 256)
    proxy = db._conn
    out = {}

    def swapper():
        try:
            db.set_embedder_named("potion-base-8M")
        except Exception as e:  # noqa: BLE001
            out["error"] = repr(e)

    sw = threading.Thread(target=swapper)
    sw.start()
    try:
        time.sleep(1.0)  # the swap is waiting on the proxy
        assert db.close() is False
        sw.join(30)
        assert not sw.is_alive()
        assert "closed" in out.get("error", ""), out
        assert db._workers_running, "the still-shared engine lost its workers"
    finally:
        del proxy


@_needs_potion
def test_a_panic_inside_the_swap_leaves_the_workers_running(tmp_path):
    # The swap holds no reference to the engine while it installs the new
    # embedder (a Weak would block Arc::get_mut), and its workers are
    # stopped then. A panic there unwound with nothing to restart them
    # from, and writes past the 256th failed (#302 review 4). The restart
    # now reaches the engine through the wrapper.
    db = YantrikDB(os.path.join(str(tmp_path), "panic.db"), 256)
    try:
        db._inject_swap_panic()
        with pytest.raises(BaseException, match="injected"):
            db.set_embedder_named("potion-base-8M")
        assert db._workers_running
        for i in range(700):
            db.record(f"after the panic {i}", embedding=[0.05] * 256)
    finally:
        db.close()
