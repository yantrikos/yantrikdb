#!/usr/bin/env python3
"""How long a write that waits on SQLite's lock freezes the host (#271).

A short call is not a harmless call. When another process holds the store's
write lock, an engine write waits for it (up to busy_timeout). A binding that
waits with the GIL held stops every Python thread in the host for that whole
wait; one that releases the GIL lets them run.

This rig makes that wait on purpose. A subprocess opens the store with
Python's own sqlite3, takes the write lock (BEGIN IMMEDIATE) and holds it for
HOLD seconds. Meanwhile the main thread calls one binding, and a 10 ms ticker
thread in the same process measures its longest gap. The lock is taken in a
subprocess because two SQLite libraries in one process can corrupt a store.

Usage, against an installed wheel in a clean venv (not PYTHONPATH=src):
    python benchmarks/gil_lock_wait_rig.py <label>
Prints one line per binding: call time, and the ticker's longest gap during it.

On 0.23.6 the three writes that waited each froze the ticker for the whole
hold (about 2,050 ms), and record_text and record failed at once with
"database is locked" (#298). Since #303 those two wait as well. On this
branch every binding here waits with the GIL released (longest gap 11-16 ms)
except record(), which still holds it for the whole wait (about 2,000 ms gap)
until #297 detaches it.
"""
import os
import random
import subprocess
import sys
import tempfile
import threading
import time

from yantrikdb import YantrikDB

HOLD = 2.0
DIM = 32

HOLDER = r"""
import sqlite3, sys, time
conn = sqlite3.connect(sys.argv[1], timeout=30, isolation_level=None)
conn.execute("BEGIN IMMEDIATE")
print("locked", flush=True)
time.sleep(float(sys.argv[2]))
conn.execute("COMMIT")
conn.close()
"""


class Emb:
    def encode(self, text):
        r = random.Random(hash(text) & 0xFFFFFFFF)
        v = [r.random() for _ in range(DIM)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]


def measure(db_path, call):
    gaps = []
    stop = threading.Event()

    def ticker():
        last = time.perf_counter()
        while not stop.is_set():
            time.sleep(0.01)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, db_path, str(HOLD)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout.readline().strip() == "locked"
    t = threading.Thread(target=ticker, daemon=True)
    t.start()
    time.sleep(0.05)
    start = time.perf_counter()
    error = None
    try:
        call()
    except Exception as e:  # noqa: BLE001 - a busy_timeout expiry is a result too
        error = f"{type(e).__name__}: {e}"
    took = time.perf_counter() - start
    stop.set()
    t.join()
    holder.wait()
    return took, max(gaps), error


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "unlabelled"
    work = tempfile.mkdtemp(prefix="gil-lock-rig-")
    path = os.path.join(work, "rig.db")
    db = YantrikDB(path, DIM)
    db.set_embedder(Emb(), allow_unverified_embedder=True)
    for i in range(200):
        db.record(f"seed record {i} about topic {i % 13}")

    calls = {
        "record_turn": lambda: db.record_turn("default", "user", "hello"),
        "task_add": lambda: db.task_add("default", "a task"),
        "relate": lambda: db.relate("alice", "acme", "works_at"),
        # Both wait for the lock since #303 (before, they failed at once with
        # "database is locked": #298), about 2,000 ms here. record_text waits
        # with the GIL released; record() still holds it for the whole wait
        # (ticker gap about 2,000 ms) until #297 detaches it.
        "record_text": lambda: db.record_text("a new record while the lock is held"),
        "record": lambda: db.record("explicit embedding", embedding=Emb().encode("x")),
    }
    print(f"{label}: another process holds the write lock for {HOLD:.1f} s")
    for name, call in calls.items():
        took, gap, error = measure(path, call)
        note = f"  ({error})" if error else ""
        print(f"  {name:12s} call {took * 1000:7.0f} ms   longest ticker gap {gap * 1000:7.0f} ms{note}")
    db.close()


if __name__ == "__main__":
    main()
