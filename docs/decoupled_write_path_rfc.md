# RFC: Decoupled Write Path (engine v0.7.0)

**Status:** draft
**Author:** yantrikdb-core
**Date:** 2026-05-07
**Tracking:** engine main; coordinated with yantrikdb-server v0.8.x roadmap; closes wedge primitive #1 from `docs/wedge_lock_scope_audit_2026-05-06.md`.

## Problem

Per the empirical baseline in `docs/wedge_empirical_baseline_2026-05-06.md`, sustained concurrent writes to a YantrikDB engine produce monotonic reader-latency degradation. With 8 writers + 4 readers at dim=384/2000-warmup over 30s, read p50 grew 30ms → 317ms (10×) and read p99 peaked at 1.89s. The mechanism: `engine/record.rs:68 self.vec_index.write().insert(...)` holds an exclusive `parking_lot::RwLock` across the entire HNSW insert (0.1–10ms, scales O(M·log N) with the per-namespace memory count). Every concurrent `record()` serializes through this single lock; `recall()` readers starve under parking_lot writer-priority once a writer queues.

This is wedge primitive #1 from the lock-scope audit. Patch A (collapse 3 conn acquisitions into 1 SAVEPOINT'd block) was implemented and empirically regressed (read tput -31%, read p99 +205%) because brief conn releases were buying interleaving for parallel writers; SAVEPOINT had no amortization at batch-size-1. Audit hypothesis-by-analogy was wrong; empirical methodology saved the regression.

The permanent fix must remove `vec_index.write()` from the request path entirely.

## Design — Postgres/MSSQL-shaped engine

```
                 ┌─────────────────────────────────────────────┐
   reads ──────► │ ArcSwap cold HNSW indexes                    │
                 │ keyed by namespace                            │
                 │ + brief read-lock over per-namespace delta    │
                 └─────────────────────────────────────────────┘
                                  ▲
                                  │ atomic swap
                                  │
                 ┌─────────────────────────────────────────────┐
                 │ Global compaction scheduler                  │
                 │ memory budget, CPU budget, fairness          │
                 └─────────────────────────────────────────────┘
                                  ▲
                                  │
                 ┌─────────────────────────────────────────────┐
                 │ Shared background ingest workers             │
                 │ N = num_cpus::get() / 2                      │
                 │ drain queue, append to per-namespace delta   │
                 └─────────────────────────────────────────────┘
                                  ▲
                                  │
                 ┌─────────────────────────────────────────────┐
   writes ─────► │ Global bounded ingest queue                  │
                 │ partitioned by namespace                     │
                 │ per-partition monotonic seq numbers          │
                 └─────────────────────────────────────────────┘
                                  ▲
                                  │
                 ┌─────────────────────────────────────────────┐
                 │ Global WAL / commit log (RFC 010 reuse)      │
                 │ durability before ack                         │
                 └─────────────────────────────────────────────┘
```

**Core principle:** shared physical machinery + logical isolation through identifiers. One engine subsystem; many namespaces; per-namespace HNSW indexes only where data size demands them.

This was arrived at after three independent redteam passes (deepseek 2×, gpt-5 external via ChatGPT) and supersedes earlier per-tenant-thread proposals which over-architected for a 10k-tenant SaaS shape that doesn't match YantrikDB's actual deployment (single brain / homelab cluster / 1–20 logical isolation domains).

## Components

### 1. Global WAL adapter

Reuse RFC 010's commit log (already shipped server-side via openraft for cluster replication). Single durability layer for both replication and engine ingestion. Fsync semantics tuned for write throughput.

`record()` writes append to the WAL *before* enqueueing for indexing. On crash, recovery replays WAL entries with seq > last_indexed_seq into the ingest queue — idempotent on rid.

### 2. Global bounded ingest queue

Single MPSC-shaped queue (or sharded MPSC keyed by namespace hash for fairness). Capacity bounded by tunable `ingest_queue_max_bytes`. Push that would exceed the bound returns `Error::Backpressure(retry_after_ms)` synchronously to the caller — explicit admission control replacing the old vec_index.write() lock's accidental backpressure.

Per-partition (per-namespace) monotonic sequence numbers assigned at enqueue time. Returned to the caller as the ack value so clients can request strict read-your-writes by waiting for `visible_seq[ns] >= my_seq`.

### 3. Shared background ingest workers

`N = num_cpus::get() / 2` worker threads (configurable; clamp to [2, 16]). Drain the queue, batch-by-namespace, apply inserts to the per-namespace mutable delta. Workers use a fair scheduling policy (round-robin across partitions) so a single hot namespace can't starve others.

### 4. Per-namespace exact mutable delta

`SkipMap<rid, (embedding, seq)>` or sorted `Vec<(rid, embedding, seq)>` per namespace. Bounded size (default `delta_max = 1024`). On overflow, the namespace's delta is *sealed* (becomes read-only, queued for compaction) and a new mutable delta is allocated.

Reads do exact-distance scan over the delta (~38ms for delta=1024 × dim=384 with SIMD; less for typical deltas). This adds a constant overhead to recall but never degrades — bounded by `delta_max` regardless of write rate.

**Why exact-scan beats hot-HNSW:** redteams flagged that "hot HNSW per namespace" introduces compaction storms, recall quality issues, and per-tenant memory amplification. Exact scan over a bounded delta is simpler, has trivial RYW (delta is always visible), and the latency cost is bounded.

### 5. Cold HNSW per namespace via ArcSwap

`ArcSwap<HnswIndex>` per namespace. Fully immutable per epoch. Reads do `cold.load()` (Arc clone, no lock) and run HNSW search on the snapshot. Writers never touch this — only the compactor swaps it.

Lazy-init: a namespace's cold HNSW is created on first compaction. Idle namespaces have only a tombstone/version map, no HNSW overhead.

### 6. Global compaction scheduler

Single background thread. Wakes on:
- Sealed deltas pending merge (size threshold)
- Time threshold (e.g., every 60s for active namespaces)
- Memory pressure signal

Hard memory budget for in-flight compactions: at most `compaction_memory_budget_bytes` of clone-rebuild RAM in flight at once. If exceeded, defer compaction (writes still flow into delta; if delta also exceeds bound, backpressure kicks in upstream).

Compaction is per-namespace: clone cold → apply sealed delta entries → swap cold via ArcSwap → drop old. RFC 011 PR-3's HnswCompactor primitive (lease-based delete-queue) handles tombstone application during the rebuild.

Staggered scheduling (jitter) prevents synchronized compaction across many namespaces.

### 7. Read path

```rust
fn recall(query: &[f32], top_k: usize, namespace: &str, ...) -> Result<Vec<RecallResult>> {
    let cold = self.cold_index_for(namespace).load();      // Arc, no lock
    let cold_results = cold.search(query, top_k * 20);

    let delta = self.delta_for(namespace).read();           // brief read lock
    let delta_results = exact_scan(delta, query, top_k);

    merge_and_dedupe(cold_results, delta_results, top_k)
}
```

Optional strict RYW via `recall_with_seq(my_last_write_seq, ...)` which waits until `visible_seq[ns] >= my_last_write_seq` before scanning. Default behavior is "delta is always visible" so RYW is satisfied without explicit wait — strict mode only matters during compaction-in-progress windows where the delta has been sealed but not yet merged.

### 8. Sequence numbers + RYW contract

Per-namespace monotonic counter. `record()` returns `(rid, seq)`. Clients wanting strict RYW pass `seq` to subsequent recall. The `visible_seq[ns]` is updated by ingest workers as they apply each delta entry.

In cluster mode (RFC 010), seq is coupled to the openraft commit log index. Cross-node RYW requires waiting for the leader's visible_seq to catch up — which composes naturally with PR 6.4's record_with_rid commit-log write path.

## API surface — record() and recall() public signatures unchanged

```rust
// Existing — unchanged behavior, but internally enqueues + returns seq
pub fn record(&self, ...) -> Result<String> { ... }   // returns rid

// New — explicit seq for strict RYW
pub fn record_returning_seq(&self, ...) -> Result<(String, u64)> { ... }

// Existing — unchanged
pub fn recall(&self, query: &[f32], top_k, ...) -> Result<Vec<RecallResult>> { ... }

// New — strict RYW variant
pub fn recall_with_seq(&self, query, top_k, namespace, min_seq, ...) -> Result<Vec<RecallResult>> { ... }
```

`record()` keeps the rid-returning shape. Internal flow becomes WAL → queue → ack. The caller pays at most ~100μs for the WAL + queue push (vs the current ~10ms HNSW insert).

## Implementation phases

| # | Work | LOC est. | Days |
|---|---|---|---|
| 1 | Global WAL adapter (reuse RFC 010 commit log primitive) | 400 | 2-3 |
| 2 | Global bounded ingest queue + backpressure | 300 | 2 |
| 3 | Shared background workers (N = cores/2) + supervisor | 500 | 3 |
| 4 | Per-namespace exact mutable delta + ArcSwap cold HNSW | 600 | 4 |
| 5 | Global compaction scheduler with memory budget | 500 | 3 |
| 6 | Sequence-number RYW on read path | 200 | 2 |
| 7 | Soak + recall regression + load tests against `wedge_repro` | — | 7 |
| **Total** | | **~2500 LOC** | **~3 weeks engine + 1 week soak** |

Each phase has a feature flag so we can ship incrementally:
- Phase 1-3 lands as v0.6.6 (infrastructure, no behavioral change yet)
- Phase 4-5 lands as v0.7.0-rc1 (decoupled writes, exact delta scan, opt-in via config)
- Phase 6 lands as v0.7.0-rc2 (RYW)
- Phase 7 → v0.7.0 GA after soak gates green

## Acceptance gates

Run `wedge_repro` with the same parameters as the empirical baseline. Targets:

| Metric | Baseline (broken) | Target | Notes |
|---|---|---|---|
| Read p50 (over 30s) | 30→317ms (10× growth) | **stable, no monotonic growth** | The headline; no growth means wedge eliminated |
| Read p99 | 593ms peak 1.89s | **< 200ms** | Production-acceptable tail |
| Read throughput | 84/s → 8/s collapse | **≥ 80/s sustained** | No collapse |
| Write p99 | 76ms | **< 50ms** | Modest improvement (write path is now WAL+queue, not HNSW) |
| Write p99.9 | 449ms | **< 150ms** | Backpressure kicks in cleanly under burst |
| Write throughput | 441/s | **≥ 800/s** | ~2× target since HNSW is no longer on critical path |
| Write ack latency p50 | (effectively HNSW insert ~10ms) | **< 1ms** | WAL+queue only |
| Recall accuracy regression vs monolithic HNSW | n/a | **< 1% recall@10 delta** | Delta scan must not lose top-k |
| RSS growth over 1h sustained load | (linear, OOM at ~2GB) | **bounded by delta_max + cold_size** | No leak |

## Concurrency scaling sweep (validation)

Run wedge_repro at writers ∈ {1, 4, 8, 16, 32, 64} after each phase ships. Map the wedge knee curve.

## What this addresses from prior redteams

Both gpt-5.5 (first redteam) and deepseek (twice) and ChatGPT-5 (external) flagged or recommended the components in this design:

- **Read-your-writes:** delta is always visible (no coalesce-window invisibility). Strict RYW via seq for callers who need it.
- **Backpressure:** explicit `Error::Backpressure(retry_after)` on bounded queue overflow. No accidental admission control.
- **Reader-writer interaction:** reads use ArcSwap (lock-free) + brief delta read lock. Writers use queue (lock-free) + worker-held delta write lock. Reads and writes share no lock primitive.
- **HNSW LSM-friendliness:** compaction is per-namespace clone-rebuild, not full-graph merge. Bounded by per-namespace size, not global.
- **Multi-tenancy memory cost:** per-namespace HNSW only (not per-tenant); shared infrastructure (queue, WAL, workers, scheduler) amortizes.
- **Compaction storms:** global scheduler with hard memory budget + jittered scheduling.
- **Failure modes:** workers run under supervisor (panic recovery via thread restart); WAL is the durable source of truth, queue/delta loss is recoverable.
- **Per-tenant overhead:** none — engine is per-namespace, infrastructure is shared.

## Alternatives considered and rejected

1. **Per-tenant freeway** (one writer thread + queue + cold per tenant) — over-architected for 1–20 actual tenants; per-tenant HNSW has superlinear memory overhead; thread management at scale is its own bottleneck. ChatGPT-5 redteam corrected this.

2. **Two-tier LSM-HNSW** (hot HNSW → cold HNSW) — hot HNSW.write() every 1–5ms recreates reader starvation in slightly altered form; HNSW is not LSM-friendly; compaction storms. gpt-5.5 + deepseek redteam rejected.

3. **Patch A** (collapse conn acquisitions in record()) — empirically regressed (-31% read tput, +205% read p99). Brief releases were buying interleaving.

4. **Sharded HNSW** (split index by rid hash) — read fanout to all shards; recall quality issues; doesn't solve durability. Strict subset of this design's expressiveness.

5. **DiskANN-style on-disk index** — large architectural shift; latency profile changes; not the immediate fix this RFC needs to ship.

## Out of scope (deferred)

- Hosted-SaaS-style hard tenant isolation (quotas, billing, separate WAL segments) — only relevant if YantrikDB becomes multi-customer. Defer until that lands.
- Distributed compaction across cluster nodes — RFC 010 PR-6 (issue #9 record_with_rid) handles cluster-side replication; per-node compaction is local.
- Embedder model migration — RFC 013 territory; this RFC's `embedding_model` field on memories provides the version pin.
- NER versioning discipline — flagged in earlier audit; separate RFC.

## Cross-references

- Empirical baseline: `docs/wedge_empirical_baseline_2026-05-06.md`
- Lock-scope audit: `docs/wedge_lock_scope_audit_2026-05-06.md`
- Wedge repro harness: `crates/yantrikdb-core/examples/wedge_repro.rs`
- Failed Patch A: branch `patch-a/record-lock-discipline` (retained as failed-experiment record)
- Server-side RFC 010 (commit log) — yantrikdb-server repo
- Server-side issue #9 (`record_with_rid` for cluster replication) — yantrikdb-server PR 6.4 hard-blocks on this RFC's v0.7.0 ship; the new internal architecture supports `record_with_rid` cleanly (caller-supplied embedding bypasses the embedder; the WAL+queue path applies it)

## Status updates

This RFC is under active implementation starting 2026-05-07. Phase tracking will be added as a separate Saga epic on yantrikdb-server's task tracker once Phase 1 begins.
