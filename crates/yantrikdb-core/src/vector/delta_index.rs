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
/// deployment via `DeltaIndex::with_capacity`.
pub const DEFAULT_DELTA_MAX: usize = 1024;

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
}

impl DeltaIndex {
    /// Create a new empty `DeltaIndex` with the given embedding dimension.
    /// Cold starts as a fresh empty `HnswIndex(dim)`. Delta is empty.
    pub fn new(dim: usize) -> Self {
        Self::with_capacity(dim, DEFAULT_DELTA_MAX)
    }

    /// Create a new `DeltaIndex` with a custom delta capacity.
    pub fn with_capacity(dim: usize, delta_max: usize) -> Self {
        Self {
            cold: ArcSwap::new(Arc::new(HnswIndex::new(dim))),
            delta: RwLock::new(Vec::with_capacity(delta_max.min(4096))),
            delta_max,
            dim,
        }
    }

    /// Construct a `DeltaIndex` whose cold tier is a pre-built `HnswIndex`.
    /// Used during engine open() when the index is rebuilt from the
    /// SQLite source of truth on disk.
    pub fn from_cold(cold: HnswIndex, delta_max: usize) -> Self {
        let dim = cold.dim();
        Self {
            cold: ArcSwap::new(Arc::new(cold)),
            delta: RwLock::new(Vec::with_capacity(delta_max.min(4096))),
            delta_max,
            dim,
        }
    }

    /// Embedding dimension for this index.
    pub fn dim(&self) -> usize {
        self.dim
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

        delta.push(DeltaEntry {
            rid,
            embedding,
            seq,
            tombstoned: false,
        });
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
        for entry in delta.iter_mut() {
            if entry.rid == rid && !entry.tombstoned {
                entry.tombstoned = true;
                entry.seq = seq;
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

        // Tombstone set + delta hit set: rids tombstoned in delta or
        // present in delta as a live entry should not pull from cold.
        let mut tombstoned: std::collections::HashSet<&str> =
            std::collections::HashSet::new();
        let mut delta_live: Vec<(&DeltaEntry, f64)> = Vec::with_capacity(delta.len());
        for entry in delta.iter() {
            if entry.tombstoned {
                tombstoned.insert(entry.rid.as_str());
            } else {
                let d = cosine_distance_f64(query, &entry.embedding);
                delta_live.push((entry, d));
            }
        }
        // Live entries that ALSO have a later tombstone in delta:
        // suppress them. We just iterate again and filter.
        delta_live.retain(|(e, _)| !tombstoned.contains(e.rid.as_str()));

        // Search cold for up to k * 2 candidates so we have headroom for
        // tombstone filtering + delta-shadowing without losing top-k.
        let cold_fetch = k.saturating_mul(2).max(k);
        let cold_results = cold.search(query, cold_fetch)?;

        // Merge: cold + delta, drop cold rids that are in delta_live (delta
        // wins) or tombstoned in delta.
        let delta_rid_set: std::collections::HashSet<&str> = delta_live
            .iter()
            .map(|(e, _)| e.rid.as_str())
            .collect();

        let mut merged: Vec<(String, f64)> = Vec::with_capacity(cold_results.len() + delta_live.len());
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

    /// Snapshot the delta entries — used by the compactor (Phase 5) to
    /// build a new cold tier. Returns a clone of the current delta vector;
    /// the actual seal-and-swap is `seal_delta_for_compaction`.
    pub fn snapshot_delta(&self) -> Vec<DeltaEntry> {
        self.delta.read().clone()
    }

    /// Atomically swap the current delta for a fresh empty one.
    /// Returns the sealed delta for the compactor to merge into cold.
    /// Phase 5 wires this into the compaction scheduler.
    pub fn seal_delta_for_compaction(&self) -> Vec<DeltaEntry> {
        let mut delta = self.delta.write();
        std::mem::replace(&mut *delta, Vec::with_capacity(self.delta_max.min(4096)))
    }

    /// Atomically install a new cold tier (post-compaction).
    /// Old readers continue against the prior Arc; new readers see the
    /// new tier on their next `cold.load()`.
    pub fn install_cold(&self, new_cold: HnswIndex) {
        self.cold.store(Arc::new(new_cold));
    }

    /// Whether the delta has reached its compaction threshold.
    /// Phase 5 compactor polls this.
    pub fn should_compact(&self) -> bool {
        self.delta_len() >= self.delta_max / 2
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
        idx.append("rid_keep".to_string(), vec_seed(1.0, 64), 1).unwrap();
        idx.append("rid_drop".to_string(), vec_seed(2.0, 64), 2).unwrap();

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

        idx.append("rid_delta".to_string(), vec_seed(2.0, 64), 1).unwrap();

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
        idx.append("rid_5".to_string(), vec_seed(5.0, 64), 5).unwrap();
        assert!(idx.should_compact(), "at half cap = should compact");
    }
}
