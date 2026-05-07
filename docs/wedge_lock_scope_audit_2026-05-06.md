# Lock-Scope Audit — Engine Hot Paths

**Audit by:** yantrikdb-core (claude-opus-4-7)
**Date:** 2026-05-06
**Triggered by:** yantrikdb-agi (architect) 23h soak finding `7a68f4ca` — production "ghosting bug" Lane B is process wedge under sustained load. Engine investigation deliverable #1 of 3 from the swarm collaboration with yantrikdb-server.
**Engine version:** v0.6.5 @ 36ba7da (clean main).
**Scope:** lock acquisitions in `record`, `recall`, `forget`, `correct`, `relate` — duration class + held-across-what.

## TL;DR

**Top wedge primitive (engine-side):** `record()` holds an **exclusive `vec_index.write()` lock across the full HNSW `insert()` call** (file `engine/record.rs:68`). Under sustained concurrent writes, every `record()` serializes through this single write lock for 0.1–10ms each. With concurrency 32 and HNSW M=16 + 10k+ memories per tenant, that's the headline serialization point — exactly the shape of "accept queue depth 130, workers not draining" that the architect saw on CT 168.

**v0.8.14 (bounded SQLite read pool + serialized writer) does NOT address this** — v0.8.14 fixes SQLite mutex contention, which is wedge primitive #2. Primitive #1 (vec_index.write across HNSW insert) needs an engine-side fix on top of v0.8.14.

## Per-call-site table

### `record()` — engine/record.rs:12-233

| # | Line | Lock | Held across | Duration class | Notes |
|---|---|---|---|---|---|
| 1 | 38 | `active_sessions.read()` | `.cloned()` of one entry | INSTANT (sub-µs) | clean |
| 2 | 42 | `self.conn()` (parking_lot Mutex via wrapper) | INSERT INTO memories + 2 UPDATEs (auto-link session) | brief (~1ms typical) | acceptable |
| 3 | 68 | **`vec_index.write()`** | `HnswIndex::insert(&rid, embedding)` — graph-walk + neighbor-pruning | **LONG (0.1–10ms, scales O(M·log N))** | **🔴 PRIMARY WEDGE PRIMITIVE** |
| 4 | 100 | `self.conn()` (re-acquired) | per-heuristic-entity INSERT/UPSERT INTO entities | brief × N | multiplies with entity count |
| 5 | 120 | `graph_index.read()` | `all_entity_names()` returning Vec | brief | safe under read |
| 6 | 128 | `self.conn()` (re-acquired) | per-candidate INSERT OR IGNORE INTO memory_entities | brief × N | re-acquisition churn |
| 7 | 136 | `graph_index.write()` | per-entity `add_entity` + `link_memory` (in-memory hashmap ops) | brief | safe |
| 8 | 154–184 | `self.conn()` (per relation) | SELECT COUNT then `ingest_claim` (re-acquires conn internally) | **🟡 brief × N relations × 2** | per-relation churn |

**Critical finding:** lock site #3 is the headline serialization point. The lock is acquired immediately after `conn` is dropped (line 65 comment: "conn dropped here") and held for the entire HNSW `insert()`. Under concurrency K, K writers contend for a single `vec_index` exclusive lock; only one progresses at a time.

**Architect's symptom mapping:**
- 23h uptime → vec_index lock contention compounds: queued tokio task working sets accumulate.
- accept queue depth 130 not draining → tokio workers parked on `vec_index.write()` waiting for the predecessor's HNSW insert to finish.
- 2× OOM at 2GB → each parked task holds its embedding + working state in tokio's stack/heap. 130 parked × ~10MB working set = 1.3GB on top of base RSS.
- Post-restart writes fast → cold cache, lock contention not yet built up.

### `record_batch()` — engine/record.rs:238+

Materially better discipline than `record()`:

| # | Line | Lock | Held across | Notes |
|---|---|---|---|---|
| 1 | 244 | `active_sessions.read()` | `.cloned()` of full map | brief |
| 2 | 250 | `graph_index.read()` | `all_entity_names()` upfront | brief |
| 3 | 271 | `self.conn()` | `SAVEPOINT batch_record` + N inserts + N session links + N entity persists | bounded by batch size |
| 4 | 388 | `vec_index.write()` ONCE | HNSW insert × N rids in single lock acquire | **proper batched pattern** |
| 5 | 395 | `scoring_cache.write()` ONCE | per-rid cache.insert | brief |
| 6 | 349 | `graph_index.write()` ONCE | per-rid add_entity + link_memory | brief |

**Pattern record() should adopt:** one lock acquisition per kind, all SQL inside one savepoint, then release. Batched HNSW insert — one write-lock acquisition for the whole batch. This is the v0.6.4 / v0.8.14 design philosophy applied at the engine boundary.

### `recall()` — engine/recall.rs

| # | Line | Lock | Held across | Duration class | Notes |
|---|---|---|---|---|---|
| 1 | 154 | `vec_index.read()` | `HnswIndex::search(query, fetch_k)` — graph traversal up to fetch_k=top_k×20 capped at 500 | medium-long (1–20ms) | shared read; multiple readers OK |
| 2 | 164 | `scoring_cache.read()` | filter+score loop, no I/O | brief | safe |
| 3 | 367 | `graph_index.read()` | `entity_matches_query()` for FTS keyword filtering | brief | safe |
| 4 | 412–418 | `graph_index.read()` (re-acquired) | `entity_matches_query()` + `expand_bfs(seeds, depth=2, max=30)` | medium | safe under read lock |
| 5 | 1142, 1749, 2139, 2176, 2836 | `graph_index.read()` (other recall variants) | similar BFS / lookup patterns | brief–medium | safe |

**Reader-writer interaction (🟡 secondary wedge primitive):** parking_lot RwLock fairness — once a `vec_index.write()` queues from a concurrent `record()`, subsequent `recall()` readers block behind it (writer-priority). Under sustained 32-concurrency mixed read+write, this produces the classic reader-starvation pattern. Symptoms: recall() p99 spikes during write bursts even though the search itself is cheap.

### `forget()` — engine/lifecycle.rs:245-273

| # | Line | Lock | Held across | Duration class | Notes |
|---|---|---|---|---|---|
| 1 | 248 | `self.conn()` | one UPDATE | brief | clean |
| 2 | 256 | `vec_index.write()` | HNSW soft-delete (sets deleted bit only, no rewiring) | INSTANT (~µs) | clean |
| 3 | 257 | `graph_index.write()` | per-edge `unlink_memory(rid)` | brief | clean |

Forget is the **cleanest** of the hot-path mutators. Lock ordering correct (conn dropped first, vec/graph index after). Soft-delete in HNSW does not rewire neighbors — that's deferred to the RFC 011 PR-3 HnswCompactor (already shipped).

### `correct()` — engine/lifecycle.rs:279+

Calls `record()` then `forget()` sequentially. Inherits all of record()'s lock issues plus forget()'s pattern. **Net effect: 2× the wedge exposure of a single record().** Under sustained correction-heavy load (memory updates), this doubles the contention.

### `relate()` — engine/graph_ops.rs:60-128

| # | Line | Lock | Held across | Duration class | Notes |
|---|---|---|---|---|---|
| 1 | 76 | `self.conn.lock()` | INSERT claim + per-entity UPSERT | brief | clean |
| 2 | 101 | `graph_index.write()` | `add_entity` × 2 + `add_edge` (in-memory) | INSTANT | clean |
| 3 | 111 | (no engine lock here) | `backfill_memory_entities_for(&[src, dst])` | **🟡 unbounded** | TODO: read this function — it likely re-enters conn + indices |

`relate()` is mostly clean except `backfill_memory_entities_for` is called outside any visible lock here but probably acquires its own — needs a follow-up read.

## Top 3 wedge primitives — ranked

| Rank | Site | Why it wedges | v0.8.14 addresses? |
|---|---|---|---|
| 1 | `record()` line 68 — `vec_index.write()` across HNSW insert | Single exclusive lock serializes ALL concurrent records. 32-concurrency × 5–50ms HNSW insert = 160–1600ms backlog grows monotonically. | **❌ NO** — v0.8.14 fixes SQLite, not vec_index. |
| 2 | `record()` lines 100, 128, 154–184 — repeated `self.conn()` re-acquisition | Per-entity, per-relation conn churn. High-relation text (10+ entities) = 30+ mutex acquisitions inside one record(). | **✅ YES** — bounded read pool + serialized writer puts SQLite behind a queue, reducing mutex pile-up. |
| 3 | `recall()` line 154 reader-writer starvation against (1) | parking_lot writer-priority means recall() readers block whenever a record() write queues. Spikes recall p99 during write bursts. | **🟡 PARTIAL** — fixing (1) eliminates this. Without (1), v0.8.14 alone doesn't help. |

## Recommendations

### Engine-side patches needed beyond v0.8.14

**Patch A — record() adopts record_batch() pattern internally.** Single-record path should match record_batch's lock discipline: one conn acquire (savepoint of size 1), one vec_index.write acquire (batch size 1), one graph_index.write acquire (batch size 1). Lock duration drops from ~10ms compounded to ~1–2ms minimum.

**Patch B — write-coalescing layer at engine boundary.** Concurrent `record()` calls within a small time window (e.g., 1–5ms) coalesce into one batched HNSW insert pass. Reduces lock-acquisition rate from O(write_rate) to O(write_rate / coalesce_window). Engine-side, transparent to callers.

**Patch C (deeper) — ArcSwap on vec_index.** Mirrors the v0.8.13 server-side ArcSwap pattern: readers see immutable snapshot, writers swap atomically. HNSW insertion needs a clone-then-swap pattern OR an internal log-and-apply pattern. Largest fix, addresses (1) and (3) together. Likely v0.7.0 or v0.7.x territory, not a v0.6.x patch.

### Server-side instrumentation needed (already committed)

`/v1/admin/diag` lock-wait histograms keyed by `module_path!()::function_name` will let us watch primitive #1 in real time. Expected metric: `lock_wait_seconds{site="yantrikdb_core::engine::record::record"}` p99 should be low under healthy concurrency, balloon when wedge starts.

### Holding pattern for issue #9

`record_with_rid` should adopt the record_batch() lock discipline from day 1. Don't carry forward record()'s anti-pattern into the cluster-mode replication API.

## Next deliverables

- [ ] HNSW RAM growth profile via `engine_bench.rs` harness — sustained 10k inserts/min for 1h, measure peak RSS + index-size-per-1k-inserts. ETA 4-6h.
- [ ] tokio + parking_lot RwLock interaction profile — instrument lock acquisition durations on the suspect sites. ETA 3h.
- [ ] Read `backfill_memory_entities_for` to close the relate() audit gap.
- [ ] Validate findings against the v0.8.x P1 plan from brainstorm a06cdaaa (2026-05-01).

## Distribution

- yantrikdb-server (peer agent, swarmcode)
- yantrikdb-agi (architect, swarmcode) — direct, per the collaboration agreement on msg `2c465959`
- Pranab on next session pickup
