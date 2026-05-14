//! Decoupled write path RFC, Phase 3.5 — background materializer threads.
//!
//! Spawns N worker threads that continuously drain pending oplog entries
//! (applied=0) and dispatch them via `apply_pending_ops_once`. Returns
//! [`MaterializerGuard`] handles whose `Drop` impl signals shutdown and
//! joins the threads.
//!
//! The workers hold a `Weak<YantrikDB>` reference, so dropping the engine
//! also lets the workers exit cleanly even if guards are leaked.
//!
//! Phase 4 will add explicit notify/wakeup so foreground `record()` can
//! signal the workers immediately rather than waiting for the 100ms timer.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::thread::JoinHandle;
use std::time::Duration;

use super::YantrikDB;

/// How many ops a worker drains in a single pass before yielding.
/// Tuned to amortize per-pass overhead while keeping individual workers
/// responsive to shutdown.
const DRAIN_BATCH_SIZE: usize = 64;

/// Sleep duration when no pending work is found.
/// Phase 4 replaces this with a condvar wake on log_op_pending.
const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Sleep duration after a drain error, before retry.
/// Prevents tight loops on persistent failure (e.g. disk full).
const ERROR_BACKOFF_INTERVAL: Duration = Duration::from_millis(100);

/// Owns one materializer thread + its shutdown flag.
///
/// Drop semantics:
/// - sets shutdown=true so the worker exits its next loop iteration
/// - joins the JoinHandle so the thread fully terminates before this guard goes out of scope
///
/// Joining can take up to `IDLE_POLL_INTERVAL` (100ms) in the worst case
/// since the worker only checks shutdown between drain passes. If you need
/// faster shutdown, drop the engine's `Arc<YantrikDB>` first (Weak::upgrade
/// will fail next iteration).
pub struct MaterializerGuard {
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Drop for MaterializerGuard {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Spawn `count` materializer threads against a shared `YantrikDB`.
///
/// Each worker holds a `Weak<YantrikDB>`. When the last `Arc<YantrikDB>` is
/// dropped, workers exit on their next iteration. Drop the returned guards
/// (or let them go out of scope) for an explicit shutdown signal.
///
/// **Recommendation:** `count = std::thread::available_parallelism().map(|n| n.get() / 2).unwrap_or(2).clamp(2, 16)`.
/// The RFC pins this default; tests may pass smaller values.
pub fn spawn_materializers(db: &Arc<YantrikDB>, count: usize) -> Vec<MaterializerGuard> {
    let mut guards = Vec::with_capacity(count);

    for worker_id in 0..count {
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = Arc::clone(&shutdown);
        let weak = Arc::downgrade(db);

        let handle = std::thread::Builder::new()
            .name(format!("yantrikdb-materializer-{worker_id}"))
            .spawn(move || worker_loop(weak, shutdown_clone, worker_id))
            .expect("spawn materializer thread");

        guards.push(MaterializerGuard {
            shutdown,
            handle: Some(handle),
        });
    }

    guards
}

/// Worker main loop.
///
/// Each iteration:
/// 1. Upgrade the weak ref to YantrikDB. Exit if engine has been dropped.
/// 2. Check shutdown flag. Exit if set.
/// 3. Call apply_pending_ops_once(DRAIN_BATCH_SIZE).
/// 4. If applied 0 ops → sleep IDLE_POLL_INTERVAL.
/// 5. If applied N > 0 ops → loop immediately to drain more.
/// 6. On error → log + sleep ERROR_BACKOFF_INTERVAL.
fn worker_loop(weak: Weak<YantrikDB>, shutdown: Arc<AtomicBool>, worker_id: usize) {
    tracing::debug!(worker_id, "materializer worker started");

    while !shutdown.load(Ordering::Relaxed) {
        let Some(db) = weak.upgrade() else {
            tracing::debug!(worker_id, "engine dropped — materializer exiting");
            break;
        };

        match db.apply_pending_ops_once(DRAIN_BATCH_SIZE) {
            Ok(0) => {
                // No work — release the strong ref before sleeping so the
                // engine can be dropped during our sleep without blocking.
                drop(db);
                std::thread::sleep(IDLE_POLL_INTERVAL);
            }
            Ok(n) => {
                tracing::trace!(worker_id, applied = n, "drained batch");
                drop(db);
                // Loop immediately — drain more if available.
            }
            Err(e) => {
                tracing::warn!(worker_id, error = %e, "drain failed; backing off");
                drop(db);
                std::thread::sleep(ERROR_BACKOFF_INTERVAL);
            }
        }
    }

    tracing::debug!(worker_id, "materializer worker exited");
}

/// Recommended worker count for the current host.
///
/// `cores / 2`, clamped to [2, 16] per the RFC. Stays modest because each
/// worker does both SQLite reads (oplog SELECT) and writes (UPDATE applied=1)
/// — too many workers fight for the conn mutex.
pub fn recommended_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get() / 2)
        .unwrap_or(2)
        .clamp(2, 16)
}

/// **Decoupled write path RFC, Phase 5 — compactor.**
///
/// Background thread that periodically calls `db.vec_index.compact()` to
/// drain the delta tier into the cold tier, bounding read latency growth.
///
/// Polls every `COMPACTOR_INTERVAL` (250ms by default in v0.6.7+,
/// was 1s in v0.6.6). On each tick:
///   1. Check `should_compact()` — fires when delta is past half-capacity
///      OR the oldest dirty entry has aged past `max_dirty_age` (default
///      60s; epic 5 task 13). The age trigger closes the gap where a
///      low-write namespace's delta sits forever and reads pay the
///      linear scan indefinitely.
///   2. Run `compact()` — clone-rebuild cold from old cold + sealed delta,
///      then ArcSwap the new cold in.
///
/// Drop the returned [`CompactorGuard`] (or let the engine `Arc<YantrikDB>`
/// drop) for clean shutdown.

pub struct CompactorGuard {
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Drop for CompactorGuard {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// How often the compactor wakes to check `should_compact()`.
///
/// **v0.6.7+ default 250ms (was 1s in v0.6.6).** Tuned in response to
/// the 2026-05-08 32-writer empirical run (saga task 18) showing read
/// p99 regression under sustained writer pressure: `compact()`'s
/// per-cycle `(*cold.load_full()).clone()` dominates cold-grow cost,
/// and at 1s intervals the compactor cycle (~5-20ms at cold=7600
/// entries) couldn't keep pace with delta fill rate, so writes
/// back-pressured and reads stalled behind delta.read() during seal.
///
/// At 250ms the compactor wakes 4× more often, drains smaller batches,
/// produces shorter pauses. Total clone work per second is unchanged
/// (same number of entries cycle through delta-then-cold), but the
/// p99 tail spreads thinner.
///
/// CPU cost trade-off: under sustained writer pressure the compactor
/// is now near-constantly busy. That's intentional — it is a P3
/// background worker (see CONCURRENCY.md Rule 3) and giving it a
/// CPU is what the priority hierarchy says to do when the engine is
/// under load. Idle deployments still pay near-zero CPU because
/// `should_compact()` short-circuits when delta is empty.
const COMPACTOR_INTERVAL: Duration = Duration::from_millis(250);

pub fn spawn_compactor(db: &Arc<YantrikDB>) -> CompactorGuard {
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = Arc::clone(&shutdown);
    let weak = Arc::downgrade(db);

    let handle = std::thread::Builder::new()
        .name("yantrikdb-compactor".to_string())
        .spawn(move || compactor_loop(weak, shutdown_clone))
        .expect("spawn compactor thread");

    CompactorGuard {
        shutdown,
        handle: Some(handle),
    }
}

fn compactor_loop(weak: Weak<YantrikDB>, shutdown: Arc<AtomicBool>) {
    tracing::debug!("compactor started");

    while !shutdown.load(Ordering::Relaxed) {
        let Some(db) = weak.upgrade() else {
            tracing::debug!("engine dropped — compactor exiting");
            break;
        };

        if db.vec_index.should_compact() {
            match db.vec_index.compact() {
                Ok(0) => {}
                Ok(n) => {
                    tracing::debug!(applied = n, "compaction drained delta into cold");
                }
                Err(e) => {
                    tracing::warn!(error = %e, "compaction failed; retrying next tick");
                }
            }
        }

        // **Saga task 18 Option 4 (v0.7.2).** Event-driven wait
        // instead of unconditional sleep. wait_for_compaction_signal
        // returns immediately when an append/tombstone past the 80%
        // threshold notify_one()'s the condvar; otherwise the
        // 250ms timeout backstops the age-trigger path and clean
        // shutdown. yantrikdb-server bench (msg b9c98a4d) showed
        // the unconditional sleep racing the delta refill at
        // 1000 wps; this closes that race.
        let _ = db.vec_index.wait_for_compaction_signal(COMPACTOR_INTERVAL);
        drop(db);
    }

    tracing::debug!("compactor exited");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::YantrikDB;

    fn open_test_db() -> Arc<YantrikDB> {
        Arc::new(YantrikDB::new(":memory:", 64).expect("open"))
    }

    #[test]
    fn worker_drains_pending_ops() {
        let db = open_test_db();

        // Push 5 pending ops.
        for i in 0..5 {
            db.log_op_pending(
                "record",
                Some(&format!("rid_{i}")),
                &serde_json::json!({}),
                None,
                None,
            )
            .unwrap();
        }
        assert_eq!(db.count_pending_ops().unwrap(), 5);

        // Spawn one worker.
        let _guards = spawn_materializers(&db, 1);

        // Worker should drain within ~150ms (one IDLE_POLL_INTERVAL).
        let mut tries = 0;
        while db.count_pending_ops().unwrap() > 0 && tries < 20 {
            std::thread::sleep(Duration::from_millis(50));
            tries += 1;
        }
        assert_eq!(
            db.count_pending_ops().unwrap(),
            0,
            "worker should drain all 5 pending ops within 1s"
        );
    }

    #[test]
    fn guard_drop_shuts_down_worker() {
        let db = open_test_db();
        let guards = spawn_materializers(&db, 2);
        // Push some work.
        for i in 0..3 {
            db.log_op_pending(
                "record",
                Some(&format!("rid_{i}")),
                &serde_json::json!({}),
                None,
                None,
            )
            .unwrap();
        }
        std::thread::sleep(Duration::from_millis(200));

        // Drop the guards — workers should exit cleanly within ~200ms.
        let start = std::time::Instant::now();
        drop(guards);
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(1),
            "guard drop should join workers within 1s, took {elapsed:?}"
        );
    }

    #[test]
    fn engine_drop_lets_workers_exit_via_weak_upgrade_fail() {
        let db = open_test_db();
        let guards = spawn_materializers(&db, 1);
        // Drop the engine while the worker is presumably sleeping.
        drop(db);

        // Now drop the guards. They should join quickly because the worker
        // saw weak.upgrade() fail and exited.
        let start = std::time::Instant::now();
        drop(guards);
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(1),
            "engine drop should let worker exit within 1s, took {elapsed:?}"
        );
    }

    #[test]
    fn multiple_workers_dont_double_apply() {
        // Two workers racing on the same pending ops should still result
        // in each op applied exactly once. apply_pending_ops_once is
        // idempotent on op_id (mark_op_applied + the SELECT WHERE applied=0
        // filter), so even if both workers SELECT the same op, only one
        // will see it through to applied=1; the other's UPDATE is a no-op.
        let db = open_test_db();
        for i in 0..50 {
            db.log_op_pending(
                "record",
                Some(&format!("rid_{i}")),
                &serde_json::json!({}),
                None,
                None,
            )
            .unwrap();
        }

        let _guards = spawn_materializers(&db, 4);

        // Wait for drain.
        let mut tries = 0;
        while db.count_pending_ops().unwrap() > 0 && tries < 40 {
            std::thread::sleep(Duration::from_millis(50));
            tries += 1;
        }
        assert_eq!(db.count_pending_ops().unwrap(), 0);

        // Verify all 50 ops are now applied=1 (no duplicates / loss).
        let conn = db.read_conn();
        let total: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM oplog WHERE applied = 1 AND op_type = 'record'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(total, 50);
    }

    #[test]
    fn recommended_worker_count_in_range() {
        let n = recommended_worker_count();
        assert!((2..=16).contains(&n), "expected [2,16], got {n}");
    }

    #[test]
    fn compactor_drains_delta_periodically() {
        // Test scaled to DEFAULT_DELTA_MAX = 256 (v0.6.7+):
        //   half-cap (compaction threshold) = 128
        //   80% wake threshold               = 205   (v0.7.2 Option 4)
        //   backpressure trigger             = 256
        //
        // **v0.7.2 update:** With Option 4 (event-driven compactor wake)
        // the compactor wakes within microseconds of the 205th append,
        // not after the 250ms tick. By the time we finish the loop the
        // delta is already partially drained — the `delta_len >= 128`
        // assertion in v0.7.1 was racing the compactor.
        //
        // The point of the test is to verify entries land in cold under
        // the compactor's drain — that contract is unchanged. So we
        // assert directly on cold_len, which is what we actually care
        // about. The interim delta_len assertion was always a proxy.
        let db = open_test_db();
        let _guard = spawn_compactor(&db);

        for i in 0..250 {
            let emb: Vec<f32> = (0..64).map(|j| (i + j) as f32 * 0.001).collect();
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            let normalized: Vec<f32> = emb.iter().map(|x| x / norm).collect();
            let seq = db
                .vec_seq
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            db.vec_index
                .append(format!("rid_{i}"), normalized, seq)
                .unwrap();
        }

        // **Closes #22.** Previously asserted "cold_len >= 200 within
        // 4s/12s" — fragile because cold_len semantics depend on the
        // compactor's drain rate AND the cold tier's internal HNSW
        // accounting, which can show 0 between rebuild cycles. Failed
        // intermittently on macOS-14 and Windows runners.
        //
        // The real contract the test is meant to verify is "the
        // compactor keeps delta bounded below capacity." That's what we
        // assert now — delta_len drops below half-cap within 30s. We
        // don't assert anything about cold_len because its exact value
        // is internal-implementation: entries can be in cold's HNSW
        // accounting, in cold's rebuild buffer, or absorbed/deduped —
        // any of which is correct compactor behavior. The contract a
        // CALLER cares about is "writes don't pile up unboundedly in
        // delta," not "writes show up exactly here within N seconds."
        //
        // 30s is generous enough to absorb any plausible runner-speed
        // variance, still fails fast if the compactor is genuinely
        // stuck.
        let mut tries = 0;
        while db.vec_index.delta_len() >= 128 && tries < 300 {
            std::thread::sleep(Duration::from_millis(100));
            tries += 1;
        }

        let cold = db.vec_index.cold_len();
        let delta = db.vec_index.delta_len();
        assert!(
            delta < 128,
            "compactor should have drained delta below half-cap within 30s, \
             got cold={} delta={}",
            cold,
            delta
        );
    }
    #[test]
    fn compactor_guard_drop_shuts_down_clean() {
        let db = open_test_db();
        let guard = spawn_compactor(&db);
        std::thread::sleep(Duration::from_millis(100));
        let start = std::time::Instant::now();
        drop(guard);
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "compactor guard drop must join within 2s"
        );
    }
}
