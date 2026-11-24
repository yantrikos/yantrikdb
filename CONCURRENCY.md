# YantrikDB Concurrency Invariants

This document records the load-bearing concurrency invariants of the
yantrikdb engine. Several inline `// See CONCURRENCY.md` comments in
the code resolve to this file. Violating any of these silently
regresses the wedge fix or correctness; review carefully before
changing the affected code paths.

Last updated: 2026-05-08 (saga epic 5 task 15).

---

## Why this file exists

In April–May 2026 we hunted a production wedge (yantrikos/yantrikdb-agi
CT 168 forensic) where sustained writes degraded reads to multi-second
latencies and eventually starved the request queue entirely. The
forensic + lock-scope audit
([docs/wedge_lock_scope_audit_2026-05-06.md](docs/wedge_lock_scope_audit_2026-05-06.md))
identified the primary primitive: `vec_index.write()` held across an
HNSW insert in the foreground request path. The fix
([docs/decoupled_write_path_rfc.md](docs/decoupled_write_path_rfc.md))
landed in v0.6.6 and was tuned in v0.6.7.

The fix works because of *specific structural invariants*. Without
them, a future change can re-introduce the wedge silently (no compile
error, tests still pass, only shows up at scale). This document names
each invariant so it can be referenced in code comments and reviewed
when changes touch the relevant paths.

---

## Rule 1 — Lock acquisition order

Always acquire engine locks in this order to prevent deadlocks:

```
conn → hlc → scoring_cache → vec_index (delta tier) → graph_index → active_sessions
```

Cross-references: `crates/yantrikdb-core/src/engine/mod.rs` doc on
`struct YantrikDB`. Any new field added to `YantrikDB` must declare
its position in this order in its rustdoc.

---

## Rule 2 — Foreground writes must NOT touch HNSW directly

The wedge primitive (v0.6.5 and earlier) was foreground writes calling
`HnswIndex::insert` while holding a write guard on the vector index.
HNSW insert is `O(M · log N)` with the layered-graph link traversal,
which under sustained load serializes every reader on the same
primitive that writers hold.

**Invariant.** Foreground write paths
(`record`, `record_with_rid`, `record_batch`, `tombstone_with_rid`,
`forget`, `correct`, `upsert_entity_edge_with_id`,
`delete_entity_edge_with_id`) MUST only touch:

- `DeltaIndex::append` / `DeltaIndex::tombstone` (brief
  `RwLock<Vec<DeltaEntry>>` write, O(1) push, no HNSW work)
- `assign_seq` (atomic fetch_add or fetch_max)
- `bump_visible_seq` (DashMap shard read + AtomicU64::fetch_max,
  lock-free in steady state)

They MUST NOT call:

- `HnswIndex::insert` / `HnswIndex::remove` (these belong to compaction)
- `compact()` (background only)
- Any new RwLock that the compactor also acquires for non-O(1) work

If you add a new write primitive, mirror the pattern in
`record_with_rid`: SAVEPOINT-guarded SQL block + `DeltaIndex::append` +
`bump_visible_seq`.

---

## Rule 3 — Background compaction must NOT share lock primitives with the foreground

`DeltaIndex::compact` does the expensive HNSW work: clone-rebuild cold
from the prior cold + a snapshot of the published delta entries. It
holds the `delta` RwLock only briefly at the start, to snapshot those
entries, and briefly at the end, to retire them once the new cold tier
is installed. Between those two points it does NOT hold any lock that
foreground writes acquire.

**Invariant.** The compactor must:

1. Snapshot the published delta entries
   (`snapshot_published_for_compaction()`). They stay in the live delta,
   visible to every read, until the merged tier is installed, and they
   stop counting toward `delta_max` meanwhile, so writers stay admitted.
2. Clone-rebuild a new cold `HnswIndex` *off the live ArcSwap*
   (`(*self.cold.load_full()).clone()`). Readers continue against the
   prior epoch.
3. Install the new cold tier with a compare-and-swap against the tier
   it cloned, then retire exactly the merged `(rid, seq)` entries. If the
   cold tier was replaced meanwhile, abandon the merge: the entries
   never left the delta, so the next compaction merges them into the
   new tier.

**The compaction gate.** `DeltaIndex` has one lock that compaction
holds for its whole build and install. Its only other holder is
`rebuild_vec_index()`, which takes it (`pause_compaction()`) from
before its SQL read until it installs the rebuilt tier. Without it the
two overwrote each other:
- A compaction started before a rebuild installed its merge over the
  rebuilt tier.
- A compaction finishing during a rebuild retired delta entries the
  rebuild's earlier SQL read did not contain, so those writes left the
  index until the next rebuild.

Foreground writes never take the gate. While a rebuild holds it, the
delta can fill and writers see `Backpressure` until the rebuild installs.
The compare-and-swap in step 3 covers any `install_cold` caller that
does not take the gate. Lock order: the gate, then `conn`; compaction
never takes `conn`.

Adding a "scheduler with priorities" that introduces a shared lock
primitive between foreground (P1) and compaction (P3) silently breaks
this invariant — `parking_lot` has no priority inheritance, so the
named priority is meaningless if the lock can be held by either.

---

## Rule 4 — Never hold `db.conn()` across an external call

`Mutex<Connection>` is non-reentrant. Holding it across a call that
might try to acquire it again (recursively, or on the same thread via
a callback) self-deadlocks. Holding it across a call that takes
significant CPU time (anything iterating, embedding, or doing graph
work) starves every other writer.

**Invariant.** Acquire `db.conn()` only inside a tight scope:

```rust
{
    let conn = db.conn();
    // SQL work only
}  // guard drops here
// non-SQL work continues without the guard
```

If you need both a conn and another lock (`graph_index`, `vec_index`,
etc.), acquire the conn FIRST per Rule 1, do the SQL work, drop the
guard, THEN acquire the second lock.

Search for `db.conn()` in the codebase: any callsite that holds the
guard across a `for entity in ...` loop, an `embed()` call, or a
`recall()` is suspect and should be rewritten to release the guard
first.

---

## Rule 5 — `ArcSwap<HnswIndex>` is the only inter-tier lock-free contract

The cold tier's atomicity for readers depends on `arc-swap`'s
`load_full()` returning a stable `Arc<HnswIndex>` snapshot that
survives until the reader's last reference drops. Replacing the cold
tier with `RwLock<HnswIndex>` regresses the wedge fix back to the v0.6.5
pattern.

**Invariant.** Do not replace the cold-tier `ArcSwap<HnswIndex>` with
`RwLock<HnswIndex>` or `Mutex<HnswIndex>`. If you think you have a
reason, run `crates/yantrikdb-core/examples/wedge_repro.rs` first and
preserve the post-v0.6.6 numbers (read p99 < 200ms under 8 writers /
30s, read p50 stable, write tput unchanged).

---

## Rule 6 — `visible_seq` is `DashMap<String, AtomicU64>`

The Phase 6 RYW visible_seq map MUST stay lock-free for reads. The
recall path calls `visible_seq_for(ns)` on every `recall_with_seq`,
and at cluster scale that is the dominant traffic. Replacing with
`Mutex<HashMap<...>>` regresses cluster RYW to a global-mutex hot path
(yantrikdb-server message Q4, 2026-05-07).

**Invariant.** `YantrikDB::visible_seq` field type is
`dashmap::DashMap<String, std::sync::atomic::AtomicU64>`. Reads use
`get(ns).map(|e| e.load(Acquire))`. Writes use
`get(ns).fetch_max(seq, Release)` on the fast path, falling back to
DashMap `entry().or_insert_with()` only for the namespace's first
write.

The condvar wait path (`wait_for_visible_seq`) uses a sentinel
`Mutex<()>` purely as the parking-lot guard required by
`Condvar::wait_for`. Critical race-avoidance pattern: re-check the
AtomicU64 *after* acquiring the sentinel mutex but *before* waiting,
so a writer that bumped + notified between the outer check and the
lock acquisition is observed.

---

## Rule 7 — Cluster mutation primitives ratchet `vec_seq` via `assign_seq`

The `seq: Option<u64>` parameter on the four cluster mutation
primitives is what makes the cluster's RYW guarantee work. Single-node
mode passes `None` and the engine allocates a fresh seq via
`fetch_add(1) + 1`. Cluster mode passes `Some(commit_log_index)` so
the seq IS the openraft commit-log index — leader and followers thus
agree on a single global monotonic stream.

**Invariant.** When extending or adding a cluster primitive:

- Take `seq: Option<u64>` as the last parameter.
- Resolve via `let seq = self.assign_seq(seq);` which uses
  `vec_seq.fetch_max(n, Relaxed)` for the cluster path.
- Bump `visible_seq[namespace]` AFTER the delta append/tombstone, on
  every path including idempotent re-apply (snapshot-lag determinism).
- Take `namespace: &str` as a required parameter — the bump must be
  observable even when the local rid/edge is missing (followers
  applying ahead of their snapshot).

See `crates/yantrikdb-core/src/engine/record.rs::record_with_rid`,
`lifecycle.rs::tombstone_with_rid`, and the two
`graph_ops.rs::*_entity_edge_with_id` for the canonical shape.

---

## Rule 8 — `parking_lot` over `std::sync` for engine locks

`parking_lot::Mutex/RwLock` is non-poisoning. If a thread panics while
holding an engine lock, subsequent acquirers do NOT see a `PoisonError`
and do NOT themselves panic. With `std::sync`, a single panic inside
the engine cascades into every other thread panicking on `lock()`,
which cascades the whole process.

**Invariant.** All `Mutex` and `RwLock` types on `YantrikDB` and its
contained types are from `parking_lot`, not `std::sync`. The
deadlock_detection feature is enabled in `Cargo.toml` so the server
can run a periodic deadlock check.

---

## Rule 9 — One SQLite library per process: never open the store with another one while the engine is open

The engine links its own SQLite (`rusqlite` `bundled`). A second SQLite
library in the same process — Python's stdlib `sqlite3`, a system
`libsqlite3` loaded by some other extension — opening the same store
file is a corruption hazard, not a locking inconvenience. Both libraries
serialise writers with POSIX advisory locks, and the kernel scopes those
locks per *process*: library B's unlock releases library A's lock, so a
write from B and a materializer write from A interleave their WAL
commits (sqlite.org/howtocorrupt.html, "multiple copies of SQLite linked
into the same application").

Measured 2026-09-06 on the 0.18.0 and 0.19.0 wheels under CPU contention:
an in-process `sqlite3.connect(...).execute("UPDATE memories SET metadata
= ...")` while the materializer was draining left the row holding an
`entities` page (rid = the entity name, text = a timestamp) in 2-8 % of
runs. The py3.10 CI job saw it as `Invalid column type Real at index: 1,
name: text` from `recall_thread`.

Rules:

- Raw SQL against a store the engine currently has open goes through a
  **separate process** (`sqlite3` CLI, a `subprocess`), never an
  in-process second library. Tests use `_raw_sql_in_subprocess` in
  `tests/test_thread_v2.py`.
- Alternatively `close()` the engine first; sequential use of two
  libraries is fine.
- Separate processes (dashboard, census scripts, backups via `.backup`)
  are safe: the kernel serialises them.

**The engine now guards this itself (issue #225, 0.22).** On Linux every
SQLite instance maps the store's `-shm` file region by region; the same
`<store>-shm` path mapped at the same offset twice in `/proc/self/maps`
means a second library has the store open in this process. The engine
scans at open and from the writer connection's commit hook (cached 200
ms). `foreign_sqlite_mode`, durable in `meta`, defaults to `refuse` on
every install: engine writes fail with `ForeignSqliteInstance` — and the
commit hook aborts anything that slipped past the pre-check, the
materializer's commits included — from the first detection until the
engine is reopened: measured 2026-09-07, the foreign library's close
unlinks the `-shm` file under the engine (its own lock table says it was
the last user), so "resume when it closes" would be unsafe. Reads
continue. `warn`
counts (`stats().foreign_sqlite_detected_since_boot`) and keeps writing;
`off` never scans. macOS reads the same rule through libproc region
enumeration (rescanned every second instead of every 200 ms); Windows
locks are per handle and is not affected (`foreign_sqlite_supported =
false`).

The cross-process half: SQLite's `PRAGMA data_version` on the writer
connection changes only when another connection commits, and every
engine write goes through that one connection, so a change is exactly
"someone else committed" (another engine process, the `sqlite3` CLI, a
backup tool — legitimate, but the only way the store changes under the
engine). Each is counted (`stats().foreign_commits_detected_since_boot`)
and queues one `PRAGMA quick_check`, run off the writer by the
materializer or by `integrity_check()` on demand; a failed check refuses
writes too, because writing onto a corrupt file only spreads the damage.

A failed check is its own refusal, not a foreign instance (issue #247):
writes fail with `IntegrityCheckFailed`, whose message carries
the check's result, and `stats().integrity_tainted` is set while
`foreign_sqlite_tainted` stays false. It is the only refusal possible on
Windows. A bad result is re-run once on the writer connection before it
refuses anything (a discarded one counts in
`integrity_checks_unconfirmed_since_boot`), and it does not latch: the
next check that returns `ok` resumes writes in the same process. After
repairing or restoring the store from another process, call
`integrity_check()`; a repair made by another process is itself an
outside commit, so the materializer queues that check anyway: while
writes are refused (either cause) it parks instead of draining, and
watches `data_version`. It used to retry its pending ops every tick,
each one taking the write lock and being aborted at commit, which kept
other processes out with `database is locked`, the repair included.
Do not reach for `set_foreign_sqlite_mode("warn")` to get writes back:
it persists to `meta` and disarms both refusals for every future open.

---

## Rule 10 — Every write-connection transaction takes the write lock first

`busy_timeout` only helps a transaction that asks for the write lock
before anything else. A `DEFERRED` transaction that reads and then
writes has to upgrade a read lock, and SQLite never runs the busy
handler for that upgrade: `sqlite3BtreeBeginTrans` retries only while
the connection holds no transaction. If another connection holds the
write lock at that moment, the write fails at once with `database is
locked`, and that connection can be in another process. If another
writer has committed since the read, it fails with `SQLITE_BUSY_SNAPSHOT`.
That was issue #298: `record()` failed in under a millisecond while
another process held the lock.

**Invariant.** Every transaction on the write connection is `BEGIN
IMMEDIATE`. The connection's transaction behaviour is set to `Immediate`
at open, so `unchecked_transaction()` complies without extra work.
`SavepointGuard` opens `BEGIN IMMEDIATE` when it is the outermost frame.
Never write a bare `SAVEPOINT` or `BEGIN` on the write connection; use
`SavepointGuard`. A single-statement write outside a transaction is
fine, because its first step already asks for the write lock. That holds
only while no statement on the same connection is still stepping: a
live cursor keeps a read transaction open, and the write is then an
upgrade again. Finish or drop every cursor before writing on the
connection it came from.

Caveats:

- **Waiting is not a guarantee.** SQLite's busy handler polls with
  backoff and no fairness. Another process that retakes the write lock
  almost continuously can still starve an engine write until
  `busy_timeout` (5 s) runs out, and the write then fails with
  `database is locked`. The fix turns the instant failure into a bounded
  wait; it cannot make the wait succeed. A writer that keeps its
  transactions short avoids this.
- **A read-only snapshot opens `Deferred` by name.** `read_conn()` falls
  back to the write connection when `YANTRIKDB_READ_POOL=0`, and that
  connection now defaults to `IMMEDIATE`, so a read snapshot taken there
  with `unchecked_transaction()` would take the write lock. The thread
  read in `thread.rs` uses
  `Transaction::new_unchecked(.., TransactionBehavior::Deferred)` for
  this reason.
- **Any correction-epoch or similar seqlock goes odd only after
  `BEGIN IMMEDIATE` returns.** The wait can last up to `busy_timeout`.
  A recall spinning on an odd epoch gives up after about 1M spins with
  `RecallContended`.
- **The wait happens while holding the connection mutex.** It also holds
  the GIL in bindings that have not yet released it (#271). Waiting is
  the point, but anything else in the process that needs the write
  connection waits behind it, including recall's trust stamping and
  impression logging. Before #298 those writes failed at once instead.
- **An empty transaction now fires the commit hook.** `BEGIN IMMEDIATE;
  COMMIT` with no writes runs it, which a deferred read-only commit does
  not. A read of the engine found no path that commits an empty
  transaction outside a pre-checked write. Don't add one that relies on
  the hook staying quiet.
- **The Python `_conn` test proxy passes SQL through verbatim.** A test
  that writes `BEGIN` there gets exactly what it wrote.

`engine/tests/lock_wait.rs` checks the rule in four ways:
- it holds the lock from a second connection and checks that both
  routes wait for it;
- it checks that a transaction holds the lock from `BEGIN`;
- it runs writes beside a connection that commits continually;
- it checks that `correct()` waiting for the lock does not starve
  `recall()`.

It also scans the source for a bare `SAVEPOINT`, `BEGIN` or `RELEASE`
outside the guard.

---

## Cross-stack rule — engine pressure suppresses enrichment, NEVER decay

This rule lives in yantrikdb-server's tick loop (it's the consumer of
the engine's pressure signal), but the *invariant* is owned jointly
and recorded here so the engine and server stay in sync.

**The rule.** When `db.delta_len() / db.delta_max() > 0.75` (or the
operator-configured `enrichment_pause_threshold`), the server's tick
loop pauses cognitive enrichment work for that tick:

  PAUSE on engine pressure (cluster RYW lock 2026-05-08, msg 73ae78b2):
    - consolidation
    - conflict_scan
    - pattern_mining
    - personality (derive_personality_traits)
    - graph_enrichment

  KEEP RUNNING regardless of pressure:
    - decay_loop
    - snapshot scheduler
    - WAL checkpoint
    - health probes / metrics tick
    - any observability / correctness-related work

**Why decay is exempt.** Memory aging is a function of wall-clock time,
not engine load. Pausing decay creates timeline divergence between
engines that are nominally tracking the same logical clock — a memory
that "should have decayed by now" suddenly snaps back to high score
when pressure clears. That's a correctness bug, not an optimization.

Decay touches the SQL `last_access` and `consolidation_status` columns
which are also touched by enrichment, so the temptation to bundle
them is real. Resist it. They run on different cadences for different
reasons.

**Why these specific enrichment jobs pause.** They do additional work
on top of the foreground load: extra recall queries, extra entity
edge writes, extra graph index updates. Under sustained ingest
pressure (the wedge scenario), they compound the pressure they should
back off from. The "user-visible correctness beats optimization beats
enrichment" priority law from ChatGPT's 2026-05-08 review (saga epic
5) is the right framing.

**Engine surface.** `db.delta_max()`, `db.delta_len()`, `db.cold_len()`
are public getters (commit bcf1c78, 2026-05-08). `db.count_pending_ops()`
predates them. The server reads these; the engine never knows or cares
what the server does with them.

**Anti-goal.** Do NOT add a `should_run_enrichment_now()` method to
the engine. The engine publishes pressure signals; the consumer
decides how to interpret them. Coupling the decision into the engine
silently couples cognitive scheduling to engine internals — the wrong
direction.

See saga epic 4 task 16 for the cross-team coordination thread.

---

## What this document is NOT

This is the *engine-layer* invariants doc. It does NOT cover:

- Server-layer concurrency (HTTP handlers, openraft state machine,
  tick loop) — that belongs in yantrikdb-server.
- The full cognitive-layer scheduling cadence (consolidation,
  contradiction, personality) beyond the cross-stack rule above —
  those workers live in yantrikdb-server's tick loop. The
  pressure-pause behavior is documented here because it spans both
  stacks and must stay aligned; the rest is server-internal.

If you find yourself adding a "scheduler" or "priority queue" to the
engine layer, stop and re-read Rule 3. The engine has one foreground
path (P1, sync, brief delta lock) and one background path (P3,
compactor, ArcSwap atomic install). That's the priority hierarchy. Do
not generalize it.

---

## References

- [docs/decoupled_write_path_rfc.md](docs/decoupled_write_path_rfc.md) — three-pass red-team architecture.
- [docs/wedge_lock_scope_audit_2026-05-06.md](docs/wedge_lock_scope_audit_2026-05-06.md) — primitive identification.
- [docs/wedge_empirical_baseline_2026-05-06.md](docs/wedge_empirical_baseline_2026-05-06.md) — quantified wedge.
- [docs/wedge_concurrency_sweep_2026-05-07.md](docs/wedge_concurrency_sweep_2026-05-07.md) — Patch A failure record.
- [crates/yantrikdb-core/examples/wedge_repro.rs](crates/yantrikdb-core/examples/wedge_repro.rs) — repro harness; run before/after structural lock changes.
