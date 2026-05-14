//! Decoupled write path RFC, Phase 4.1 — per-DB vector index with
//! immutable cold tier + mutable delta tier.
//!
//! The classic wedge primitive (`engine/record.rs:68 vec_index.write()`
//! held across HNSW insert) is eliminated by this two-tier design:
//!
//! - **Cold** (`ArcSwap<HnswIndex>`): immutable per epoch. Reads do
//!   `cold.load()` which is an Arc clone — lock-free, no contention with
//!   writers, no parking_lot writer-priority effects.
//!
//! - **Delta** (`RwLock<Vec<DeltaEntry>>`): bounded mutable buffer of
//!   recent writes that haven't been compacted into cold yet. Reads do
//!   exact-distance scan over the delta (small N, SIMD-friendly). Writes
//!   acquire the delta write lock briefly to append. Reader latency adds
//!   `O(delta.len() * dim)` distance computations — bounded by `delta_max`.
//!
//! - **Compaction** (Phase 5): periodically clones cold + applies delta
//!   entries to build a new cold, then atomically swaps via ArcSwap. The
//!   delta is sealed during compaction and a new mutable delta is allocated
//!   so writers don't block.
//!
//! ## Lock-acquisition contract
//!
//! - Reads: `cold.load()` (no lock) + `delta.read()` (parking_lot RwLock
//!   read; only contends with the brief append lock and the
//!   sealing-for-compaction swap).
//! - Writes: `delta.write()` for one append + drop. Never touches cold.
//! - Compaction (Phase 5): clones cold off-thread, swaps cold via ArcSwap,
//!   swaps delta atomically. Brief writer lock, no impact on readers.
//!
//! ## Visibility / read-your-writes
//!
//! Writes are visible to subsequent reads as soon as the delta append
//! returns — `delta.read()` will include them. Strict RYW (caller wants
//! to wait for cold to absorb the write) is a Phase 6 concern via
//! sequence numbers; it is not the default visibility contract.
//!
//! ## Tombstone semantics
//!
//! `tombstone(rid)` marks the entry in the delta as deleted. The
//! `search()` path filters tombstoned entries out of the merged result.
//! For rids that exist only in cold, the tombstone is appended to delta
//! as a deletion marker so readers see the rid as deleted. Compaction
//! resolves these by removing the rid from cold during rebuild.

use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::RwLock;

use crate::error::{Result, YantrikDbError};
use crate::vector::hnsw::HnswIndex;

/// One entry in the mutable delta tier.
///
/// `seq` is the monotonic sequence number assigned at append time. This is
/// the basis for read-your-writes via Phase 6's `recall_with_seq`.
///
/// `tombstoned` distinguishes a live insert from a deletion marker. A delta
/// entry with `tombstoned=true` MUST suppress the rid from search results
/// even if cold has a non-tombstoned copy of the same rid.
#[derive(Clone, Debug)]
pub struct DeltaEntry {
    pub rid: String,
    pub embedding: Vec<f32>,
    pub seq: u64,
    pub tombstoned: bool,
}

/// Default soft cap on delta size. Hitting this triggers backpressure on
/// `append()`. Phase 5 compaction will keep the delta below this in steady
/// state by sealing + swapping when it crosses the cap. Tunable per
/// deployment via `DeltaIndex::with_capacity` or the `YANTRIKDB_DELTA_MAX`
/// environment variable.
///
/// **Default 256 (was 1024 in v0.6.6).** Cross-platform empirical study by
/// yantrikdb-server (2026-05-07): at `delta_max=256` vs the previous 1024,
/// write throughput rose from ~586/s to ~996/s (+70% over v0.6.6, ≈2.2× over
/// v0.6.5) while read p50 dropped from ~300ms to ~141ms — better than even
/// v0.6.5's pre-wedge baseline. The intuition is that a smaller delta keeps
/// per-search linear scans cheap and triggers compaction often enough that the
/// cold HNSW absorbs most of the working set, so most reads never touch the
/// hot path's RwLock at all. Larger caps amortize compaction overhead but pay
/// for it in steady-state read latency.
pub const DEFAULT_DELTA_MAX: usize = 256;

/// Default max dirty age before age-based compaction fires.
///
/// **The aging gap.** The size-based trigger (`delta_len >= delta_max/2`)
/// only fires when a namespace is actively writing. A namespace whose
/// delta sits at e.g. 50 dirty entries with no new writes will *never*
/// hit the size threshold, so reads against it pay the linear delta scan
/// indefinitely. ChatGPT's 2026-05-08 review (epic 5 task 13) flagged
/// aging as the single biggest concrete gap in our scheduler-by-
/// construction design.
///
/// Fix: the compactor also fires when the *oldest* entry in a non-empty
/// delta has been sitting for more than this duration. 60s is a starting
/// default — long enough that bursty workloads still rely on the size
/// trigger (avoiding compaction churn), short enough that idle namespaces
/// merge into cold within a window where read latency hasn't deteriorated
/// noticeably. Tunable per deployment via `DeltaIndex::with_capacity_and_age`
/// or the `YANTRIKDB_MAX_DIRTY_AGE_SECS` environment variable.
pub const DEFAULT_MAX_DIRTY_AGE: std::time::Duration = std::time::Duration::from_secs(60);

/// Per-DB two-tier vector index used by the decoupled write path.
///
/// Wraps an `HnswIndex` cold tier (atomically swapped via ArcSwap at
/// compaction) and a bounded mutable delta. Replaces the foreground
/// `vec_index.write()` lock pattern that produced the wedge.
pub struct DeltaIndex {
    cold: ArcSwap<HnswIndex>,
    delta: RwLock<Vec<DeltaEntry>>,
    delta_max: usize,
    dim: usize,
    /// Wall-clock instant when the delta most recently transitioned from
    /// empty to non-empty. Cleared (set to `None`) at every successful
    /// `seal_delta_for_compaction`. The compactor reads this to decide
    /// whether the age-based trigger fires.
    ///
    /// Lives on `DeltaIndex` rather than per-`DeltaEntry` so a busy
    /// workload with frequent appends doesn't pay one Instant per entry —
    /// we only care about the *oldest* unflushed entry's age.
    oldest_dirty_at: parking_lot::Mutex<Option<std::time::Instant>>,
    /// Compaction trigger threshold for `oldest_dirty_at` — once the
    /// delta's oldest entry has been sitting longer than this, the
    /// compactor fires regardless of delta size.
    max_dirty_age: std::time::Duration,
    /// **Saga task 18 Option 4 (v0.7.2).** Event-driven compactor
    /// wake. `append()` and `tombstone()` signal this condvar when the
    /// delta crosses ~80% of `delta_max`, so the compactor wakes within
    /// microseconds of pressure rather than waiting up to its 250ms
    /// poll tick. The condvar's paired sentinel mutex is `()` — no
    /// data lives behind it; it's there because parking_lot::Condvar
    /// requires a guard. The 250ms tick stays as a backstop for the
    /// age-trigger path and graceful shutdown responsiveness. Confirmed
    /// architectural pull by yantrikdb-server msg b9c98a4d 2026-05-08
    /// (90s bench showed read p99 spikes during compactor sleep
    /// windows; this closes that gap).
    compactor_wake_cv: parking_lot::Condvar,
    compactor_wake_mu: parking_lot::Mutex<()>,
}

impl DeltaIndex {
    /// Create a new empty `DeltaIndex` with the given embedding dimension.
    /// Cold starts as a fresh empty `HnswIndex(dim)`. Delta is empty.
    pub fn new(dim: usize) -> Self {
        Self::with_capacity(dim, DEFAULT_DELTA_MAX)
    }

    /// Create a new `DeltaIndex` with a custom delta capacity.
    pub fn with_capacity(dim: usize, delta_max: usize) -> Self {
        Self::with_capacity_and_age(dim, delta_max, DEFAULT_MAX_DIRTY_AGE)
    }

    /// Create a new `DeltaIndex` with a custom delta capacity AND a custom
    /// age-based compaction trigger. Used by tests that need to exercise
    /// the age trigger in seconds rather than the production-default 60s,
    /// and by the engine constructor when reading the
    /// `YANTRIKDB_MAX_DIRTY_AGE_SECS` env override.
    pub fn with_capacity_and_age(
        dim: usize,
        delta_max: usize,
        max_dirty_age: std::time::Duration,
    ) -> Self {
        Self {
            cold: ArcSwap::new(Arc::new(HnswIndex::new(dim))),
            delta: RwLock::new(Vec::with_capacity(delta_max.min(4096))),
            delta_max,
            dim,
            oldest_dirty_at: parking_lot::Mutex::new(None),
            max_dirty_age,
            compactor_wake_cv: parking_lot::Condvar::new(),
            compactor_wake_mu: parking_lot::Mutex::new(()),
        }
    }

    /// Construct a `DeltaIndex` whose cold tier is a pre-built `HnswIndex`.
    /// Used during engine open() when the index is rebuilt from the
    /// SQLite source of truth on disk.
    pub fn from_cold(cold: HnswIndex, delta_max: usize) -> Self {
        Self::from_cold_with_age(cold, delta_max, DEFAULT_MAX_DIRTY_AGE)
    }

    /// `from_cold` sibling that also takes a custom max_dirty_age — used
    /// by the engine constructor when honoring YANTRIKDB_MAX_DIRTY_AGE_SECS.
    pub fn from_cold_with_age(
        cold: HnswIndex,
        delta_max: usize,
        max_dirty_age: std::time::Duration,
    ) -> Self {
        let dim = cold.dim();
        Self {
            cold: ArcSwap::new(Arc::new(cold)),
            delta: RwLock::new(Vec::with_capacity(delta_max.min(4096))),
            delta_max,
            dim,
            oldest_dirty_at: parking_lot::Mutex::new(None),
            max_dirty_age,
            compactor_wake_cv: parking_lot::Condvar::new(),
            compactor_wake_mu: parking_lot::Mutex::new(()),
        }
    }

    /// Embedding dimension for this index.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Soft cap on delta size before backpressure fires. Exposed so
    /// callers (notably yantrikdb-server's tick loop, RFC scheduler
    /// pressure rule) can scale enrichment thresholds proportionally
    /// rather than hard-coding a count that's wrong for non-default
    /// `delta_max` deployments.
    pub fn delta_max(&self) -> usize {
        self.delta_max
    }

    /// Append an entry to the delta tier.
    ///
    /// Backpressure: if the delta is at or above `delta_max`, returns
    /// `Error::Backpressure`. The compactor (Phase 5) is responsible for
    /// draining the delta before this happens; receiving Backpressure
    /// here means the compactor is behind.
    ///
    /// Idempotent on rid+seq: if an identical (rid, seq) already exists,
    /// the second append is silently a no-op (recovery may replay the
    /// same op multiple times).
    pub fn append(&self, rid: String, embedding: Vec<f32>, seq: u64) -> Result<()> {
        if embedding.len() != self.dim {
            return Err(YantrikDbError::InvalidInput(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.dim,
                embedding.len()
            )));
        }

        let mut delta = self.delta.write();

        // Idempotent on rid+seq.
        if delta.iter().any(|e| e.rid == rid && e.seq == seq) {
            return Ok(());
        }

        if delta.len() >= self.delta_max {
            return Err(YantrikDbError::Backpressure {
                pending: delta.len() as i64,
                max: self.delta_max as i64,
                retry_after_ms: 50,
            });
        }

        let was_empty = delta.is_empty();
        delta.push(DeltaEntry {
            rid,
            embedding,
            seq,
            tombstoned: false,
        });
        let new_len = delta.len();
        drop(delta); // release write lock before signaling

        // Stamp the dirty-age clock on first non-empty append after every
        // compaction. Subsequent appends within the same dirty window
        // don't touch it — only the oldest entry's age matters for the
        // age-based trigger.
        if was_empty {
            *self.oldest_dirty_at.lock() = Some(std::time::Instant::now());
        }
        // **Saga task 18 Option 4 (v0.7.2).** Wake the compactor early
        // when delta crosses ~80% of capacity. Without this, the
        // compactor sleeps its 250ms tick and delta can saturate
        // before the next wake (at 1000 wps + delta_max=256, refill
        // takes ~256ms — almost exactly one tick, so half the time
        // the compactor wakes to a saturated delta and reads stall).
        // Cheap notify_one — no contention since the compactor is
        // the only waiter.
        if new_len >= self.delta_max * 80 / 100 {
            self.compactor_wake_cv.notify_one();
        }
        Ok(())
    }

    /// Append a tombstone for `rid` to the delta tier.
    ///
    /// If the rid exists in the delta as a non-tombstoned entry, that entry
    /// is marked tombstoned in place. Otherwise a deletion marker is
    /// appended (so readers know to skip the rid even if cold contains it).
    ///
    /// Returns `true` if a live delta entry was tombstoned, `false` if the
    /// tombstone was appended as a marker only (rid was in cold or unknown).
    /// Either way the visibility effect is the same: subsequent searches
    /// will not return `rid`.
    pub fn tombstone(&self, rid: &str, seq: u64) -> bool {
        let mut delta = self.delta.write();
        let was_empty = delta.is_empty();
        for entry in delta.iter_mut() {
            if entry.rid == rid && !entry.tombstoned {
                entry.tombstoned = true;
                entry.seq = seq;
                // In-place mutation — delta was non-empty already, so the
                // dirty-age clock is already stamped from the original
                // append. Nothing to do.
                return true;
            }
        }
        // Not in delta — append a deletion marker.
        delta.push(DeltaEntry {
            rid: rid.to_string(),
            embedding: Vec::new(), // tombstone marker; embedding never read
            seq,
            tombstoned: true,
        });
        let new_len = delta.len();
        drop(delta); // release write lock before signaling

        // Stamp the dirty-age clock if this tombstone marker is the only
        // entry in the delta (delta was empty before the push). Same rule
        // as append(): only the *oldest* dirty entry's age matters.
        if was_empty {
            *self.oldest_dirty_at.lock() = Some(std::time::Instant::now());
        }
        // Saga task 18 Option 4: tombstone-only paths also fill the
        // delta and should wake the compactor at the same threshold.
        if new_len >= self.delta_max * 80 / 100 {
            self.compactor_wake_cv.notify_one();
        }
        false
    }

    /// Search for the top-k nearest neighbors of `query`.
    ///
    /// Searches both cold and delta, merges by distance, drops tombstoned
    /// rids, and returns up to `k` (rid, distance) pairs sorted ascending.
    ///
    /// If the same rid appears in both tiers, the delta entry wins (it's
    /// strictly newer; cold gets the corresponding update at compaction).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f64)>> {
        if query.len() != self.dim {
            return Err(YantrikDbError::InvalidInput(format!(
                "query dimension mismatch: expected {}, got {}",
                self.dim,
                query.len()
            )));
        }

        // Snapshot cold (Arc clone, no lock) + brief delta read.
        let cold = self.cold.load();
        let delta = self.delta.read();

        // Per-rid winner: highest seq wins. If the winning entry is
        // tombstoned, the rid is dead. Otherwise it is the canonical live
        // entry for the rid (and shadows any cold copy).
        //
        // This handles the archive/hydrate scenario: tombstone(rid, seq=5)
        // followed by append(rid, embedding, seq=10) — the live entry at
        // seq=10 wins, the tombstone at seq=5 loses.
        let mut winner_per_rid: std::collections::HashMap<&str, &DeltaEntry> =
            std::collections::HashMap::new();
        for entry in delta.iter() {
            match winner_per_rid.get(entry.rid.as_str()) {
                Some(existing) if existing.seq >= entry.seq => {}
                _ => {
                    winner_per_rid.insert(entry.rid.as_str(), entry);
                }
            }
        }
        let mut tombstoned: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut delta_live: Vec<(&DeltaEntry, f64)> = Vec::with_capacity(winner_per_rid.len());
        for (rid, entry) in &winner_per_rid {
            if entry.tombstoned {
                tombstoned.insert(*rid);
            } else {
                let d = cosine_distance_f64(query, &entry.embedding);
                delta_live.push((*entry, d));
            }
        }

        // Search cold for up to k * 2 candidates so we have headroom for
        // tombstone filtering + delta-shadowing without losing top-k.
        let cold_fetch = k.saturating_mul(2).max(k);
        let cold_results = cold.search(query, cold_fetch)?;

        // Merge: cold + delta, drop cold rids that are in delta_live (delta
        // wins) or tombstoned in delta.
        let delta_rid_set: std::collections::HashSet<&str> =
            delta_live.iter().map(|(e, _)| e.rid.as_str()).collect();

        let mut merged: Vec<(String, f64)> =
            Vec::with_capacity(cold_results.len() + delta_live.len());
        for (rid, dist) in &cold_results {
            if tombstoned.contains(rid.as_str()) || delta_rid_set.contains(rid.as_str()) {
                continue;
            }
            merged.push((rid.clone(), *dist));
        }
        for (entry, dist) in &delta_live {
            merged.push((entry.rid.clone(), *dist));
        }

        // Sort by distance ascending, take top-k.
        merged.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        merged.truncate(k);
        Ok(merged)
    }

    /// Number of entries in the delta tier (including tombstones).
    pub fn delta_len(&self) -> usize {
        self.delta.read().len()
    }

    /// Number of entries in the cold tier.
    pub fn cold_len(&self) -> usize {
        self.cold.load().len()
    }

    /// Total entry count across both tiers (cold + delta, including
    /// tombstone markers in delta). Approximation: a tombstone in delta
    /// shadowing a live cold entry is counted twice — matches the shape
    /// HnswIndex.len() exposes. Stats callers use this for ballpark
    /// health metrics, not for exact accounting.
    pub fn len(&self) -> usize {
        self.cold_len() + self.delta_len()
    }

    /// True iff both tiers are empty.
    pub fn is_empty(&self) -> bool {
        self.cold_len() == 0 && self.delta_len() == 0
    }

    /// Snapshot the delta entries — used by the compactor (Phase 5) to
    /// build a new cold tier. Returns a clone of the current delta vector;
    /// the actual seal-and-swap is `seal_delta_for_compaction`.
    pub fn snapshot_delta(&self) -> Vec<DeltaEntry> {
        self.delta.read().clone()
    }

    /// Atomically swap the current delta for a fresh empty one.
    /// Returns the sealed delta for the compactor to merge into cold.
    /// Phase 5 wires this into the compaction scheduler.
    ///
    /// Also clears the dirty-age clock — once entries are sealed, the
    /// next append against the now-empty delta restarts the age window.
    pub fn seal_delta_for_compaction(&self) -> Vec<DeltaEntry> {
        let mut delta = self.delta.write();
        let sealed = std::mem::replace(&mut *delta, Vec::with_capacity(self.delta_max.min(4096)));
        // Reset the dirty-age clock under the same write lock so the
        // age-trigger calculation can never observe a stale stamp paired
        // with an empty delta.
        *self.oldest_dirty_at.lock() = None;
        sealed
    }

    /// Atomically install a new cold tier (post-compaction).
    /// Old readers continue against the prior Arc; new readers see the
    /// new tier on their next `cold.load()`.
    pub fn install_cold(&self, new_cold: HnswIndex) {
        self.cold.store(Arc::new(new_cold));
    }

    /// **Decoupled write path RFC, Phase 5 — compaction.**
    ///
    /// Drain the current delta into cold by clone-rebuilding the cold tier.
    /// Atomic from the readers' perspective: the ArcSwap.store() at the end
    /// is the visible epoch boundary. Old readers finish on the prior cold
    /// snapshot; new readers see the merged cold.
    ///
    /// Algorithm:
    ///   1. seal_delta_for_compaction() — atomically swap delta for an
    ///      empty new one. Concurrent writes go to the new delta and are
    ///      preserved across the compaction.
    ///   2. Clone the current cold HnswIndex.
    ///   3. For each sealed entry in seq order:
    ///        - tombstoned: HnswIndex::remove(rid) on the clone
    ///        - live with rid not in cold: HnswIndex::insert
    ///        - live with rid in cold (update): remove + re-insert
    ///   4. ArcSwap.store(Arc::new(new_cold)).
    ///
    /// Returns the number of delta entries applied.
    ///
    /// Idempotent on empty delta — returns 0 without touching cold.
    pub fn compact(&self) -> Result<usize> {
        let sealed = self.seal_delta_for_compaction();
        if sealed.is_empty() {
            return Ok(0);
        }

        // Clone cold off the live ArcSwap so readers continue against
        // the prior epoch while we build the new one.
        let mut new_cold: HnswIndex = (*self.cold.load_full()).clone();

        // Apply sealed entries by seq order. Same-rid duplicates within the
        // sealed batch resolve via "highest seq wins" — same rule as search.
        let mut by_rid: std::collections::HashMap<String, &DeltaEntry> =
            std::collections::HashMap::with_capacity(sealed.len());
        for entry in &sealed {
            match by_rid.get(&entry.rid) {
                Some(existing) if existing.seq >= entry.seq => {}
                _ => {
                    by_rid.insert(entry.rid.clone(), entry);
                }
            }
        }

        let mut applied = 0usize;
        for entry in by_rid.values() {
            if entry.tombstoned {
                new_cold.remove(&entry.rid);
            } else {
                // remove first to handle "update" semantics — if the rid
                // was already in cold, the new embedding supersedes it.
                new_cold.remove(&entry.rid);
                new_cold.insert(&entry.rid, &entry.embedding)?;
            }
            applied += 1;
        }

        self.cold.store(Arc::new(new_cold));
        Ok(applied)
    }

    /// Whether the delta should be compacted on the next compactor tick.
    ///
    /// Two triggers, ORed:
    ///
    /// 1. **Size trigger** — `delta_len() >= delta_max / 2`. The
    ///    classic Phase 5 trigger; protects read latency under bursty
    ///    write workloads.
    ///
    /// 2. **Age trigger** — `delta_len() > 0` AND the oldest entry has
    ///    been sitting longer than `max_dirty_age` (default 60s). Closes
    ///    the gap where a low-write namespace's delta sits at, say, 50
    ///    dirty entries forever and reads pay the linear scan
    ///    indefinitely. Bolt-on per epic 5 task 13 (ChatGPT review
    ///    2026-05-08) — explicitly NOT a multi-factor scoring formula.
    ///
    /// The compactor polls this each tick (every COMPACTOR_INTERVAL).
    pub fn should_compact(&self) -> bool {
        // Size trigger.
        if self.delta_len() >= self.delta_max / 2 {
            return true;
        }
        // Age trigger.
        let stamp = *self.oldest_dirty_at.lock();
        match stamp {
            Some(t) if self.delta_len() > 0 && t.elapsed() >= self.max_dirty_age => true,
            _ => false,
        }
    }

    /// **Saga task 18 Option 4 (v0.7.2).** Compactor's wait primitive.
    /// Blocks for up to `timeout` waiting for either:
    /// - An `append()`/`tombstone()` that pushed delta past 80% capacity
    ///   (event-driven wake — wakes within microseconds of pressure).
    /// - The `timeout` expiring (backstop for the age-trigger path
    ///   and for graceful shutdown responsiveness).
    ///
    /// Pre-v0.7.2 the compactor's loop used `thread::sleep(timeout)`,
    /// which at 1000 wps + delta_max=256 meant the delta saturated
    /// inside one tick window, stalling readers. This API replaces
    /// the sleep with an event-driven wait.
    ///
    /// Returns true if woken by signal, false if the timeout fired.
    /// Caller treats both as "go check should_compact again."
    pub fn wait_for_compaction_signal(&self, timeout: std::time::Duration) -> bool {
        let mut guard = self.compactor_wake_mu.lock();
        let result = self.compactor_wake_cv.wait_for(&mut guard, timeout);
        !result.timed_out()
    }
}

// ── Helpers ──

/// Cosine distance, exposed at f64 precision for merge stability with
/// HnswIndex's f64 distances. Embedding bytes assumed already normalized
/// at the caller boundary; we still recompute norms to be safe.
fn cosine_distance_f64(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 1.0;
    }
    let mut dot: f64 = 0.0;
    let mut na: f64 = 0.0;
    let mut nb: f64 = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    let na = na.sqrt();
    let nb = nb.sqrt();
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    (1.0 - (dot / (na * nb))).clamp(0.0, 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim).map(|i| (seed + i as f32) * 0.1).collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        raw.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn empty_index_search_returns_empty() {
        let idx = DeltaIndex::new(64);
        let query = vec_seed(1.0, 64);
        let r = idx.search(&query, 10).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn append_then_search_finds_in_delta() {
        let idx = DeltaIndex::new(64);
        let emb = vec_seed(1.0, 64);
        idx.append("rid_1".to_string(), emb.clone(), 1).unwrap();

        let query = vec_seed(1.0, 64);
        let r = idx.search(&query, 5).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, "rid_1");
        assert!(r[0].1 < 0.001, "exact-match distance ~0, got {}", r[0].1);
    }

    #[test]
    fn append_dimension_mismatch_rejected() {
        let idx = DeltaIndex::new(64);
        let bad = vec![0.0f32; 32];
        let err = idx
            .append("rid_x".to_string(), bad, 1)
            .expect_err("must reject");
        assert!(matches!(err, YantrikDbError::InvalidInput(_)));
    }

    #[test]
    fn delta_full_returns_backpressure() {
        let idx = DeltaIndex::with_capacity(64, 5);
        for i in 0..5 {
            idx.append(format!("rid_{i}"), vec_seed(i as f32, 64), i as u64)
                .unwrap();
        }
        let err = idx
            .append("rid_overflow".to_string(), vec_seed(99.0, 64), 999)
            .expect_err("must backpressure");
        match err {
            YantrikDbError::Backpressure { pending, max, .. } => {
                assert_eq!(pending, 5);
                assert_eq!(max, 5);
            }
            other => panic!("expected Backpressure, got {other:?}"),
        }
    }

    #[test]
    fn append_idempotent_on_same_rid_seq() {
        let idx = DeltaIndex::new(64);
        let emb = vec_seed(1.0, 64);
        idx.append("rid_1".to_string(), emb.clone(), 1).unwrap();
        idx.append("rid_1".to_string(), emb.clone(), 1).unwrap();
        assert_eq!(idx.delta_len(), 1, "second append at same seq is no-op");
    }

    #[test]
    fn tombstone_hides_rid_from_search() {
        let idx = DeltaIndex::new(64);
        idx.append("rid_keep".to_string(), vec_seed(1.0, 64), 1)
            .unwrap();
        idx.append("rid_drop".to_string(), vec_seed(2.0, 64), 2)
            .unwrap();

        let query = vec_seed(2.0, 64);
        let r_before = idx.search(&query, 5).unwrap();
        assert_eq!(r_before.len(), 2);

        idx.tombstone("rid_drop", 3);
        let r_after = idx.search(&query, 5).unwrap();
        assert_eq!(r_after.len(), 1);
        assert_eq!(r_after[0].0, "rid_keep");
    }

    #[test]
    fn tombstone_on_cold_only_rid_appends_marker() {
        let idx = DeltaIndex::new(64);
        // Pre-populate cold by directly building one.
        let mut cold = HnswIndex::new(64);
        cold.insert("rid_in_cold", &vec_seed(5.0, 64)).unwrap();
        idx.install_cold(cold);

        // Sanity: search finds it.
        let r = idx.search(&vec_seed(5.0, 64), 5).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, "rid_in_cold");

        // Tombstone — returns false (not in delta as live), but marker appended.
        let was_in_delta = idx.tombstone("rid_in_cold", 1);
        assert!(!was_in_delta);
        assert_eq!(idx.delta_len(), 1, "tombstone marker appended");

        let r2 = idx.search(&vec_seed(5.0, 64), 5).unwrap();
        assert!(r2.is_empty(), "tombstoned cold rid hidden");
    }

    #[test]
    fn cold_and_delta_merged_in_search() {
        let idx = DeltaIndex::new(64);
        let mut cold = HnswIndex::new(64);
        cold.insert("rid_cold", &vec_seed(1.0, 64)).unwrap();
        idx.install_cold(cold);

        idx.append("rid_delta".to_string(), vec_seed(2.0, 64), 1)
            .unwrap();

        let r = idx.search(&vec_seed(1.5, 64), 5).unwrap();
        let rids: Vec<&str> = r.iter().map(|(rid, _)| rid.as_str()).collect();
        assert!(rids.contains(&"rid_cold"));
        assert!(rids.contains(&"rid_delta"));
        assert_eq!(rids.len(), 2);
    }

    #[test]
    fn delta_shadows_cold_when_rid_appears_in_both() {
        // Same rid in both tiers — delta wins on visibility (it's newer).
        // The merge must not double-count.
        let idx = DeltaIndex::new(64);
        let mut cold = HnswIndex::new(64);
        cold.insert("rid_dup", &vec_seed(1.0, 64)).unwrap();
        idx.install_cold(cold);

        // Same rid with different embedding in delta (simulates an update
        // that hasn't compacted yet).
        idx.append("rid_dup".to_string(), vec_seed(5.0, 64), 1)
            .unwrap();

        let r = idx.search(&vec_seed(5.0, 64), 5).unwrap();
        assert_eq!(r.len(), 1, "no double-count");
        assert_eq!(r[0].0, "rid_dup");
        // Distance should match the DELTA embedding, not cold's.
        assert!(r[0].1 < 0.001);
    }

    #[test]
    fn seal_delta_returns_entries_and_resets() {
        let idx = DeltaIndex::new(64);
        for i in 0..5 {
            idx.append(format!("rid_{i}"), vec_seed(i as f32, 64), i as u64)
                .unwrap();
        }
        let sealed = idx.seal_delta_for_compaction();
        assert_eq!(sealed.len(), 5);
        assert_eq!(idx.delta_len(), 0, "delta reset to empty");

        // Subsequent appends go to the new delta.
        idx.append("rid_after".to_string(), vec_seed(10.0, 64), 100)
            .unwrap();
        assert_eq!(idx.delta_len(), 1);
    }

    #[test]
    fn install_cold_atomically_swaps() {
        let idx = DeltaIndex::new(64);
        assert_eq!(idx.cold_len(), 0);

        let mut new_cold = HnswIndex::new(64);
        new_cold.insert("rid_a", &vec_seed(1.0, 64)).unwrap();
        new_cold.insert("rid_b", &vec_seed(2.0, 64)).unwrap();
        idx.install_cold(new_cold);
        assert_eq!(idx.cold_len(), 2);
    }

    // ── Epic 5 task 13: age-based compaction trigger ──
    //
    // The size-based trigger fires only when delta_len >= delta_max/2.
    // Without an age trigger, a low-write namespace whose delta sits at
    // e.g. 10 dirty entries with no new writes will *never* compact, so
    // reads against it pay the linear delta scan indefinitely. These
    // tests exercise the age-trigger path that closes that gap.
    //
    // We use `with_capacity_and_age` to set `max_dirty_age` to ~50ms
    // so the test wall-clock waits stay bounded; the production default
    // is 60s.

    #[test]
    fn age_trigger_does_not_fire_on_empty_delta() {
        // Empty delta + no oldest_dirty_at stamp => should_compact == false
        // even after waiting past max_dirty_age.
        let idx = DeltaIndex::with_capacity_and_age(64, 256, Duration::from_millis(20));
        std::thread::sleep(Duration::from_millis(40));
        assert!(
            !idx.should_compact(),
            "empty delta never triggers age compaction"
        );
    }

    #[test]
    fn age_trigger_fires_after_max_dirty_age_elapses() {
        // 10 entries (well under half-cap of 128) sitting for >50ms with
        // a 20ms max_dirty_age must trigger compaction by the age path.
        let idx = DeltaIndex::with_capacity_and_age(64, 256, Duration::from_millis(20));
        for i in 0..10 {
            idx.append(format!("rid_{i}"), vec_seed(i as f32, 64), i as u64)
                .unwrap();
        }
        assert_eq!(idx.delta_len(), 10);
        assert!(
            !idx.should_compact(),
            "below half-cap and within max_dirty_age window must NOT trigger"
        );
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            idx.should_compact(),
            "10 entries sitting for >max_dirty_age must trigger age path"
        );
    }

    #[test]
    fn age_trigger_resets_on_seal() {
        // After seal_delta_for_compaction, the oldest_dirty_at clock
        // resets — a subsequent append starts a fresh age window.
        let idx = DeltaIndex::with_capacity_and_age(64, 256, Duration::from_millis(20));
        idx.append("rid_a".to_string(), vec_seed(1.0, 64), 1)
            .unwrap();
        std::thread::sleep(Duration::from_millis(30));
        assert!(idx.should_compact(), "first window: age trigger fires");

        let _ = idx.seal_delta_for_compaction();
        assert!(!idx.should_compact(), "seal cleared dirty-age clock");

        // New append after seal: fresh age window, must NOT trigger immediately.
        idx.append("rid_b".to_string(), vec_seed(2.0, 64), 2)
            .unwrap();
        assert!(!idx.should_compact(), "fresh window after seal");
        std::thread::sleep(Duration::from_millis(30));
        assert!(
            idx.should_compact(),
            "second window: age trigger fires again"
        );
    }

    #[test]
    fn age_trigger_compacts_low_write_namespace_end_to_end() {
        // The "end-to-end" test: 10 entries, sit for >max_dirty_age,
        // run compact() (simulating the compactor tick), entries land in cold.
        let idx = DeltaIndex::with_capacity_and_age(64, 256, Duration::from_millis(20));
        for i in 0..10 {
            idx.append(format!("rid_{i}"), vec_seed(i as f32, 64), i as u64)
                .unwrap();
        }
        std::thread::sleep(Duration::from_millis(40));
        assert!(idx.should_compact(), "age trigger ready");
        let n = idx.compact().unwrap();
        assert_eq!(n, 10, "all 10 entries applied to cold");
        assert_eq!(idx.delta_len(), 0, "delta drained");
        assert_eq!(idx.cold_len(), 10, "cold absorbed all entries");
        // After compact, the dirty-age clock is reset (seal cleared it).
        assert!(!idx.should_compact(), "post-compact: nothing to do");
    }

    #[test]
    fn age_trigger_tombstone_only_delta_also_fires() {
        // Edge case: delta contains only a tombstone marker (no live
        // append). The age trigger must still fire — readers care about
        // the tombstone reaching cold so the rid no longer surfaces in
        // recall. Same `delta_len > 0` rule applies.
        let idx = DeltaIndex::with_capacity_and_age(64, 256, Duration::from_millis(20));
        // tombstone() against a rid not in delta appends a marker.
        let was_live = idx.tombstone("rid_remote", 1);
        assert!(!was_live, "tombstone of unknown rid is appended as marker");
        assert_eq!(idx.delta_len(), 1);
        std::thread::sleep(Duration::from_millis(30));
        assert!(
            idx.should_compact(),
            "tombstone-only delta also fires by age"
        );
    }

    #[test]
    fn concurrent_appends_and_reads_no_corruption() {
        let idx = Arc::new(DeltaIndex::with_capacity(64, 1024));

        // Spawn 4 writer threads, each appending 50 entries with disjoint rid spaces.
        let mut writer_handles = Vec::new();
        for w in 0..4 {
            let idx_c = Arc::clone(&idx);
            writer_handles.push(thread::spawn(move || {
                for i in 0..50 {
                    let rid = format!("w{w}_rid_{i}");
                    let emb = vec_seed((w * 100 + i) as f32, 64);
                    idx_c.append(rid, emb, (w * 100 + i) as u64).unwrap();
                }
            }));
        }

        // Spawn 4 reader threads, each doing 100 searches concurrently.
        let mut reader_handles = Vec::new();
        for r in 0..4 {
            let idx_c = Arc::clone(&idx);
            reader_handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let q = vec_seed((r * 1000 + i) as f32, 64);
                    let _ = idx_c.search(&q, 10).unwrap();
                }
            }));
        }

        for h in writer_handles {
            h.join().unwrap();
        }
        for h in reader_handles {
            h.join().unwrap();
        }

        assert_eq!(idx.delta_len(), 200, "all 4 writers contributed 50 each");
    }

    #[test]
    fn search_returns_top_k_sorted_ascending() {
        let idx = DeltaIndex::new(64);
        // Insert 5 distinct entries; query against one of them.
        for i in 0..5 {
            idx.append(format!("rid_{i}"), vec_seed(i as f32, 64), i as u64)
                .unwrap();
        }
        let r = idx.search(&vec_seed(2.0, 64), 3).unwrap();
        assert_eq!(r.len(), 3);
        // Distances must be non-decreasing.
        for w in r.windows(2) {
            assert!(w[0].1 <= w[1].1, "distances must sort ascending");
        }
        // Top result should be rid_2 (exact match).
        assert_eq!(r[0].0, "rid_2");
    }

    #[test]
    fn from_cold_preserves_existing_entries() {
        let mut cold = HnswIndex::new(64);
        cold.insert("rid_a", &vec_seed(1.0, 64)).unwrap();
        cold.insert("rid_b", &vec_seed(2.0, 64)).unwrap();
        let idx = DeltaIndex::from_cold(cold, 64);
        assert_eq!(idx.cold_len(), 2);
        assert_eq!(idx.delta_len(), 0);

        let r = idx.search(&vec_seed(1.0, 64), 5).unwrap();
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn should_compact_at_half_capacity() {
        let idx = DeltaIndex::with_capacity(64, 10);
        assert!(!idx.should_compact());
        for i in 0..4 {
            idx.append(format!("rid_{i}"), vec_seed(i as f32, 64), i as u64)
                .unwrap();
        }
        assert!(!idx.should_compact(), "below half cap");
        idx.append("rid_5".to_string(), vec_seed(5.0, 64), 5)
            .unwrap();
        assert!(idx.should_compact(), "at half cap = should compact");
    }

    #[test]
    fn compact_drains_delta_into_cold() {
        let idx = DeltaIndex::new(64);
        for i in 0..10 {
            idx.append(format!("rid_{i}"), vec_seed(i as f32, 64), i as u64)
                .unwrap();
        }
        assert_eq!(idx.delta_len(), 10);
        assert_eq!(idx.cold_len(), 0);

        let n = idx.compact().unwrap();
        assert_eq!(n, 10);
        assert_eq!(idx.delta_len(), 0, "delta drained");
        assert_eq!(idx.cold_len(), 10, "cold has all 10 entries now");

        // Search still finds them (now in cold tier).
        let r = idx.search(&vec_seed(5.0, 64), 3).unwrap();
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].0, "rid_5", "exact match still found post-compaction");
    }

    #[test]
    fn compact_applies_tombstones_to_cold() {
        let idx = DeltaIndex::new(64);
        // Pre-seed cold with two rids.
        let mut cold = HnswIndex::new(64);
        cold.insert("rid_keep", &vec_seed(1.0, 64)).unwrap();
        cold.insert("rid_drop", &vec_seed(2.0, 64)).unwrap();
        idx.install_cold(cold);
        assert_eq!(idx.cold_len(), 2);

        // Tombstone rid_drop in delta.
        idx.tombstone("rid_drop", 1);
        // Compact applies the tombstone to cold.
        let n = idx.compact().unwrap();
        assert_eq!(n, 1);
        // HnswIndex.len() includes tombstoned nodes (same as pre-Phase 4),
        // but search filters them. Verify via search:
        let r = idx.search(&vec_seed(2.0, 64), 5).unwrap();
        let rids: Vec<&str> = r.iter().map(|(rid, _)| rid.as_str()).collect();
        assert!(!rids.contains(&"rid_drop"), "tombstone applied to cold");
        assert!(rids.contains(&"rid_keep"));
    }

    #[test]
    fn compact_applies_archive_then_hydrate_correctly() {
        // The scenario engine_tests::test_hydrate_memory exercises:
        //   1. record(rid)         -> append at seq=1
        //   2. archive(rid)        -> tombstone at seq=2
        //   3. hydrate(rid)        -> append at seq=3
        // Compact must produce a cold where rid_X is LIVE (highest seq wins).
        let idx = DeltaIndex::new(64);
        idx.append("rid_X".to_string(), vec_seed(5.0, 64), 1)
            .unwrap();
        idx.tombstone("rid_X", 2);
        idx.append("rid_X".to_string(), vec_seed(5.0, 64), 3)
            .unwrap();

        let n = idx.compact().unwrap();
        assert_eq!(n, 1, "highest-seq winner applied once");

        let r = idx.search(&vec_seed(5.0, 64), 5).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, "rid_X", "rid alive in cold post-compaction");
    }

    #[test]
    fn compact_idempotent_on_empty_delta() {
        let idx = DeltaIndex::new(64);
        assert_eq!(idx.compact().unwrap(), 0);
        assert_eq!(idx.compact().unwrap(), 0);
    }

    #[test]
    fn compact_preserves_in_flight_writes() {
        // Writers appending during a compaction window must not be lost.
        // Sealing swaps in a fresh delta atomically, so writes that arrive
        // after the swap go to the new delta. The compactor sees only the
        // sealed entries.
        let idx = DeltaIndex::new(64);
        for i in 0..5 {
            idx.append(format!("before_{i}"), vec_seed(i as f32, 64), i as u64)
                .unwrap();
        }
        // Seal manually + start "compaction" but before applying, append more.
        let sealed = idx.seal_delta_for_compaction();
        assert_eq!(sealed.len(), 5);
        assert_eq!(idx.delta_len(), 0, "fresh delta after seal");

        for i in 0..3 {
            idx.append(
                format!("after_{i}"),
                vec_seed((100 + i) as f32, 64),
                (100 + i) as u64,
            )
            .unwrap();
        }
        assert_eq!(
            idx.delta_len(),
            3,
            "after-seal writes accumulate in new delta"
        );

        // Note: the actual compactor (compact()) does seal+apply in one step.
        // This test exercises the seal-only primitive to prove the swap
        // does not lose subsequent writes.
    }

    #[test]
    fn compact_threshold_drives_compaction() {
        // Realistic loop: every time should_compact() returns true, run
        // compact(), and the delta gets bounded.
        let idx = DeltaIndex::with_capacity(64, 10);
        for i in 0..50 {
            // If we are above half-cap, compact first to make room.
            if idx.should_compact() {
                idx.compact().unwrap();
            }
            idx.append(format!("rid_{i}"), vec_seed(i as f32, 64), i as u64)
                .unwrap();
        }
        // After 50 inserts with periodic compaction, cold has all 50.
        idx.compact().unwrap(); // drain final delta
        assert_eq!(idx.cold_len(), 50);
        assert_eq!(idx.delta_len(), 0);
    }
}
