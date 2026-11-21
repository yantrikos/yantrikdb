#!/usr/bin/env python3
# GIL ticker-stall rig, contributed by @jgray-gcs3 in
# https://github.com/yantrikos/yantrikdb/issues/271 (added with their permission).
#
# Measures how long engine calls freeze the host interpreter: a 10 ms ticker
# thread runs in the same process while recall(), think() and record() run on
# the main thread, and each phase reports ticks/s, the longest tick gap and
# per-call wall times. It is the before/after instrument for releasing the GIL
# around engine calls (#271, #279, #280, #281).
#
# Usage, against an installed wheel in a clean venv (not PYTHONPATH=src):
#     python benchmarks/gil_ticker_rig.py <version-label> <workdir>
# The first run builds a fresh 23,000-record store in <workdir> (a few minutes);
# later runs reuse it, so keep <workdir> (or cache it in CI) between runs.
# Results go to rig-<version-label>.json next to this script.
#
# Reference numbers from the contributor, published wheels, same machine:
#   recall  0.23.4: 1.8 ticks/s, max gap 3,609 ms -> 0.23.5: 97.9 ticks/s, 24 ms
#   think   0.23.4: 1.4 ticks/s, max gap 2,363 ms -> 0.23.5: 99.0 ticks/s, 12.5 ms
#   record  0.23.5: 22.3 ticks/s, max gap 176 ms (still holds the GIL)
"""Round 1 GIL ticker-stall rig (rebuilt from the Nov 16 2026 deep pass profile).

Shape of record: a loaded store (Round 1 used 23,443 rows), engine IN-PROCESS,
a 10 ms ticker thread in the same process measuring how long it gets frozen
while engine calls run on the main thread. Metrics per phase: ticker rate
(ticks/s), longest tick gap (worst stall), and per-call wall times.
"""
import json, os, random, statistics, sys, threading, time

from yantrikdb import YantrikDB

TARGET_ROWS = 23000
NS = "rig"

TOPICS = [
    ("gateway", "The gateway froze while a think call held the GIL for the full duration of the call."),
    ("board", "Card t_{n} moved to review after the worker posted its result and proof."),
    ("backup", "The nightly backup snapshot completed and the manifest hash matched the vault copy."),
    ("memory", "A lesson was recorded: verify at source before claiming a wiring change is live."),
    ("recall", "Recall returned stale rows because the long lived engine never saw newer commits."),
    ("deploy", "The merge landed but deployment waits on live proof at the target host."),
    ("watch", "The capacity retry cron fired and commented the miss on the watcher card."),
    ("training", "The first QLoRA proof run trains on the rented card and the adapter comes home."),
    ("studio", "The Studio bench answers at nineteen tokens per second on the Vulkan path."),
    ("release", "The maintainer merged the consolidation fix and it ships in the next release."),
]


def gen_texts(n, seed=271):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        topic, tmpl = TOPICS[i % len(TOPICS)]
        # clusters of near-duplicates (same base text, small suffix) so
        # think()/consolidation has real work, like the production store
        cluster = i // 7
        out.append((topic, tmpl.format(n=cluster) + f" Entry {cluster} variant {i % 7}, seat note {rng.randint(0, 999)} for the {topic} log."))
    return out


class Ticker:
    def __init__(self, interval=0.010):
        self.interval = interval
        self.gaps = []
        self._stop = threading.Event()
        self._thread = None
        self.t_start = None
        self.t_end = None

    def _run(self):
        last = time.perf_counter()
        self.t_start = last
        while not self._stop.is_set():
            time.sleep(self.interval)
            now = time.perf_counter()
            self.gaps.append(now - last)
            last = now
        self.t_end = last

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)
        return False

    def stats(self):
        if not self.gaps or self.t_end is None:
            return {}
        dur = self.t_end - self.t_start
        return {
            "window_s": round(dur, 2),
            "ticks": len(self.gaps),
            "ticks_per_s": round(len(self.gaps) / dur, 1),
            "max_gap_ms": round(max(self.gaps) * 1000, 1),
            "p95_gap_ms": round(statistics.quantiles(self.gaps, n=20)[18] * 1000, 1) if len(self.gaps) >= 20 else None,
            "mean_gap_ms": round(statistics.mean(self.gaps) * 1000, 1),
        }


def phase(fn, calls_desc):
    """Run fn() repeatedly under the ticker; fn returns after `calls_desc` calls."""
    with Ticker() as tk:
        walls = fn()
    return {"ticker": tk.stats(), "calls": len(walls),
            "call_wall_s": {"min": round(min(walls), 3), "median": round(statistics.median(walls), 3),
                            "max": round(max(walls), 3)} if walls else {}}


def main():
    version = sys.argv[1]
    workdir = sys.argv[2]
    os.makedirs(workdir, exist_ok=True)
    db_path = os.path.join(workdir, "rig.db")
    result = {"version": version, "target_rows": TARGET_ROWS}

    if not os.path.exists(db_path):
        db = YantrikDB.with_default(db_path)
        texts = gen_texts(TARGET_ROWS)
        t0 = time.time()
        for i, (topic, text) in enumerate(texts):
            db.record(text=text, memory_type="fact", importance=0.5,
                      namespace=NS, domain=topic, metadata={})
            if (i + 1) % 5000 == 0:
                print(f"build {i + 1}/{TARGET_ROWS} at {time.time() - t0:.0f}s", flush=True)
        result["build_s"] = round(time.time() - t0, 1)
        try:
            result["stats_after_build"] = db.stats()
        except Exception as e:
            result["stats_error"] = str(e)[:200]
        db = None
        print("build done", result["build_s"], flush=True)
    else:
        print("reusing existing store", flush=True)

    db = YantrikDB.with_default(db_path)

    # Phase 0: ticker alone (noise floor of this VM)
    with Ticker() as tk:
        time.sleep(5)
    result["baseline_ticker"] = tk.stats()

    # Phase 1: recall loop
    queries = ["GIL freeze think call", "backup manifest hash vault", "recall stale rows commits",
               "adapter trains rented card", "merge deployment live proof"]
    def recall_loop():
        walls = []
        for i in range(40):
            t0 = time.perf_counter()
            db.recall(query=queries[i % len(queries)], top_k=10, namespace=NS,
                      domain=None, memory_type=None, include_superseded=False,
                      expand_entities=True)
            walls.append(time.perf_counter() - t0)
        return walls
    result["recall"] = phase(recall_loop, 40)

    # Phase 2: think loop (full maintenance config, as the fleet calls it)
    cfg = {"run_consolidation": True, "run_conflict_scan": True,
           "run_pattern_mining": False, "run_personality": False}
    def think_loop():
        walls = []
        for _ in range(6):
            t0 = time.perf_counter()
            db.think(cfg)
            walls.append(time.perf_counter() - t0)
        return walls
    result["think"] = phase(think_loop, 6)

    # Phase 3: record loop (maintainer: still GIL-bound in 0.23.5)
    new_texts = gen_texts(120, seed=999)
    def record_loop():
        walls = []
        for topic, text in new_texts:
            t0 = time.perf_counter()
            db.record(text=text, memory_type="fact", importance=0.5,
                      namespace=NS, domain=topic, metadata={})
            walls.append(time.perf_counter() - t0)
        return walls
    result["record"] = phase(record_loop, 120)

    try:
        result["stats_final"] = db.stats()
    except Exception as e:
        result["stats_error_final"] = str(e)[:200]

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"rig-{version}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
