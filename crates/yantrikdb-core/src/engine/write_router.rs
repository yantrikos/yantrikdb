//! Synchronized write-routing barrier for reembed (issue #41).
//!
//! ## Why this exists
//!
//! Without a real synchronization primitive, the reembed operation has
//! a window where in-flight synchronous writes can commit to the
//! about-to-be-discarded old HNSW after the rebuild snapshot was
//! taken — and never reach the new index. Brainstorm-2 gpt-5.5
//! caught this; see yantrikos/yantrikdb#41 comment chain for the bad
//! interleaving timeline.
//!
//! `WriteRouter` provides the cutover gate: writers acquire a
//! `SyncWriteGuard` for the full duration of a synchronous write
//! (memories INSERT + vec_index.append + oplog write all under the
//! guard). Reembed flips the router state to Queueing, waits for
//! `inflight == 0`, and then can safely capture `build_hwm` knowing
//! no synchronous writer can still commit to the old generation.
//!
//! ## Invariants enforced
//!
//! From brainstorm-2 §8:
//! - **Single classification** of every write into pre-barrier-sync OR
//!   post-barrier-queued — no middle state.
//! - **No old application after barrier** — once `wait_for_no_sync_writers()`
//!   returns, no `applied_generation = old_generation` writes can
//!   appear (because sync writers all left and queued writers don't
//!   apply directly).
//!
//! ## Implementation choice
//!
//! mutex/condvar + RAII guard instead of atomic counter. The atomic
//! version drafted in brainstorm-2 round 2 had too many opportunities
//! for missing-decrement bugs on error / panic paths. The RAII guard
//! is panic-safe by construction via `Drop`.

use std::sync::Arc;

use parking_lot::{Condvar, Mutex};

/// State machine of the write-routing gate.
///
/// Transitions:
/// - `Normal` → `Queueing`   (reembed cutover start)
/// - `Queueing` → `Normal`   (reembed completed or aborted)
///
/// There is no separate "Draining" state in the final design; the
/// `Queueing` state both rejects new sync writers AND waits for
/// in-flight ones to leave (callers use `wait_for_no_sync_writers`
/// explicitly). This keeps the state space minimal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterState {
    /// `try_enter_sync_writer()` returns Some(guard); writes take the
    /// synchronous path (record memories table + vec_index + oplog
    /// applied=1).
    Normal,
    /// `try_enter_sync_writer()` returns None; writes must take the
    /// queued path (log_op_pending with applied=0 +
    /// embedding_model=current_embedder_name).
    Queueing,
}

/// Shared inner state behind the router. Locked by a single mutex; the
/// state field is small + writes are rare (only on reembed cutover) so
/// the lock is not a contention point.
struct RouterInner {
    state: RouterState,
    inflight: u64,
}

/// Synchronized write-routing barrier.
///
/// One instance per engine. Construct via [`WriteRouter::new()`]
/// (defaults to `Normal` state, 0 inflight). Wrap in `Arc` if you need
/// to share across threads; the implementation is `Send + Sync` via
/// the `parking_lot` primitives.
pub struct WriteRouter {
    inner: Mutex<RouterInner>,
    /// Notified when `inflight` decrements to 0 (waking
    /// `wait_for_no_sync_writers`).
    drained: Condvar,
    /// Notified when `state` changes (currently unused externally; held
    /// in case a future async API wants to wake on state transitions
    /// without polling).
    #[allow(dead_code)]
    state_changed: Condvar,
}

/// RAII guard held by an in-flight synchronous writer. Drop decrements
/// the inflight counter and notifies the cutover-drain condvar if the
/// counter reached zero.
///
/// Panic-safe: even if the writer's record/append/log_op chain panics,
/// the guard's Drop still runs and the counter stays coherent.
pub struct SyncWriteGuard<'a> {
    router: &'a WriteRouter,
}

impl<'a> Drop for SyncWriteGuard<'a> {
    fn drop(&mut self) {
        let mut g = self.router.inner.lock();
        debug_assert!(g.inflight > 0, "SyncWriteGuard drop on zero inflight");
        g.inflight -= 1;
        if g.inflight == 0 {
            // Wake any waiter inside wait_for_no_sync_writers.
            self.router.drained.notify_all();
        }
    }
}

impl WriteRouter {
    /// Construct a new router in `Normal` state with 0 inflight writers.
    pub fn new() -> Self {
        WriteRouter {
            inner: Mutex::new(RouterInner {
                state: RouterState::Normal,
                inflight: 0,
            }),
            drained: Condvar::new(),
            state_changed: Condvar::new(),
        }
    }

    /// Current state. Mainly for tests + observability; do NOT use this
    /// for routing decisions on the write path (use
    /// `try_enter_sync_writer` which atomically checks + increments).
    pub fn state(&self) -> RouterState {
        self.inner.lock().state
    }

    /// Current in-flight synchronous-writer count. Tests +
    /// observability.
    pub fn inflight(&self) -> u64 {
        self.inner.lock().inflight
    }

    /// Try to enter the synchronous write path. Returns `Some(guard)`
    /// if the router is in `Normal` state (writer must hold the guard
    /// for the full sync-write duration). Returns `None` if the router
    /// is in `Queueing` state (writer must take the queued path
    /// instead).
    ///
    /// The state check + inflight increment happen atomically under
    /// the inner mutex, so there is no TOCTOU window between "saw
    /// Normal" and "counted as in-flight." That property is what makes
    /// the cutover safe.
    pub fn try_enter_sync_writer(&self) -> Option<SyncWriteGuard<'_>> {
        let mut g = self.inner.lock();
        if g.state == RouterState::Normal {
            g.inflight += 1;
            Some(SyncWriteGuard { router: self })
        } else {
            None
        }
    }

    /// Flip state to `Queueing`. New sync writers will see `None` from
    /// `try_enter_sync_writer` from now on. Already-acquired guards
    /// continue executing; cutover must call
    /// `wait_for_no_sync_writers` to drain them.
    pub fn switch_to_queueing(&self) {
        let mut g = self.inner.lock();
        g.state = RouterState::Queueing;
        self.state_changed.notify_all();
    }

    /// Flip state back to `Normal`. Reembed calls this at completion
    /// (or abort) to restore the synchronous write path. Inflight
    /// queued writes continue draining via the materializer; this only
    /// affects classification of NEW writes.
    pub fn switch_to_normal(&self) {
        let mut g = self.inner.lock();
        g.state = RouterState::Normal;
        self.state_changed.notify_all();
    }

    /// Block until `inflight == 0`. Reembed cutover calls this AFTER
    /// `switch_to_queueing()` to ensure all in-flight synchronous
    /// writers have left before capturing `build_hwm` and starting
    /// Encoding.
    ///
    /// Returns immediately if `inflight` is already 0.
    pub fn wait_for_no_sync_writers(&self) {
        let mut g = self.inner.lock();
        while g.inflight > 0 {
            self.drained.wait(&mut g);
        }
    }

    /// Block until `inflight == 0` OR `timeout` elapses. Returns `true`
    /// if drained, `false` if timed out. Useful for tests that don't
    /// want to hang.
    pub fn wait_for_no_sync_writers_timeout(&self, timeout: std::time::Duration) -> bool {
        use std::time::Instant;
        let deadline = Instant::now() + timeout;
        let mut g = self.inner.lock();
        while g.inflight > 0 {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let remaining = deadline - now;
            // wait_for returns a WaitTimeoutResult-like guard plus a
            // timed_out indicator; we just loop and check inflight.
            let _ = self.drained.wait_for(&mut g, remaining);
        }
        true
    }
}

impl Default for WriteRouter {
    fn default() -> Self {
        Self::new()
    }
}

// `Arc<WriteRouter>` is the expected shape for callers. WriteRouter
// itself is Send + Sync via the parking_lot primitives.
pub type SharedWriteRouter = Arc<WriteRouter>;

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn new_router_is_normal_with_zero_inflight() {
        let r = WriteRouter::new();
        assert_eq!(r.state(), RouterState::Normal);
        assert_eq!(r.inflight(), 0);
    }

    #[test]
    fn enter_sync_writer_in_normal_returns_guard_and_increments() {
        let r = WriteRouter::new();
        let g = r.try_enter_sync_writer().expect("Normal must yield guard");
        assert_eq!(r.inflight(), 1);
        drop(g);
        assert_eq!(r.inflight(), 0);
    }

    #[test]
    fn enter_sync_writer_in_queueing_returns_none() {
        let r = WriteRouter::new();
        r.switch_to_queueing();
        assert!(
            r.try_enter_sync_writer().is_none(),
            "Queueing state must reject sync writers — this is the cutover invariant"
        );
        assert_eq!(r.inflight(), 0);
    }

    #[test]
    fn inflight_counter_correct_under_multiple_concurrent_writers() {
        let r = Arc::new(WriteRouter::new());
        let mut handles = vec![];
        // Spawn 10 writers; each holds its guard for ~50ms then drops.
        for _ in 0..10 {
            let r_clone = Arc::clone(&r);
            handles.push(thread::spawn(move || {
                let g = r_clone
                    .try_enter_sync_writer()
                    .expect("Normal must yield guard");
                thread::sleep(Duration::from_millis(50));
                drop(g);
            }));
        }
        // Wait for at least one writer thread to acquire its guard.
        // Polling avoids the macos-14 aarch64 flake where 10ms fixed
        // sleep wasn't enough for scheduler warm-up (sibling fix above).
        let acquire_deadline = std::time::Instant::now() + Duration::from_millis(500);
        while r.inflight() == 0 && std::time::Instant::now() < acquire_deadline {
            thread::sleep(Duration::from_millis(1));
        }
        let observed = r.inflight();
        assert!(
            observed > 0 && observed <= 10,
            "expected 1..=10 concurrent inflight writers, got {observed}"
        );
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(r.inflight(), 0, "all writers must have released guards");
    }

    #[test]
    fn wait_for_drain_blocks_until_inflight_zero() {
        // Spawn a writer holding a guard, then call wait_for from main
        // thread. wait must return only after the guard is dropped.
        let r = Arc::new(WriteRouter::new());
        let writer_done = Arc::new(parking_lot::Mutex::new(false));

        let r_clone = Arc::clone(&r);
        let done_clone = Arc::clone(&writer_done);
        let writer = thread::spawn(move || {
            let g = r_clone
                .try_enter_sync_writer()
                .expect("Normal must yield guard");
            thread::sleep(Duration::from_millis(100));
            *done_clone.lock() = true;
            drop(g);
        });

        // Wait for the writer thread to acquire the guard. Polling
        // is used instead of a fixed sleep because the macos-14
        // aarch64 CI runner sometimes hasn't scheduled the spawned
        // thread within 20ms — failure mode locked by PR #43 CI
        // (job 76888502814, 2026-05-20). Up to 500ms is plenty for
        // any plausible scheduler stall; the test still completes
        // within tens of ms on healthy hosts.
        let acquire_deadline = std::time::Instant::now() + Duration::from_millis(500);
        while r.inflight() == 0 && std::time::Instant::now() < acquire_deadline {
            thread::sleep(Duration::from_millis(1));
        }
        // Should be inflight=1 once the writer acquired.
        assert_eq!(
            r.inflight(),
            1,
            "writer should have acquired guard within 500ms"
        );
        // wait must block until the writer drops the guard.
        r.wait_for_no_sync_writers();
        assert!(
            *writer_done.lock(),
            "wait_for_no_sync_writers returned before writer set done flag"
        );
        writer.join().unwrap();
    }

    #[test]
    fn wait_for_drain_returns_immediately_when_already_zero() {
        let r = WriteRouter::new();
        let start = std::time::Instant::now();
        r.wait_for_no_sync_writers();
        assert!(
            start.elapsed() < Duration::from_millis(50),
            "wait_for must short-circuit when inflight=0"
        );
    }

    #[test]
    fn wait_for_timeout_returns_false_when_writers_dont_drain() {
        let r = Arc::new(WriteRouter::new());
        let _g = r.try_enter_sync_writer().unwrap();
        let drained = r.wait_for_no_sync_writers_timeout(Duration::from_millis(50));
        assert!(
            !drained,
            "timeout must return false when writers never drain"
        );
    }

    #[test]
    fn cutover_sequence_serializes_writers_and_reembed() {
        // The headline correctness scenario: simulate the brainstorm-2
        // bad interleaving and confirm the router prevents it.
        //
        // Steps:
        //   1. Writer A enters Normal, acquires sync guard.
        //   2. Reembed starts: switch_to_queueing.
        //   3. Writer B tries to enter; gets None (must take queued path).
        //   4. Reembed waits for no sync writers (blocks on A's guard).
        //   5. Writer A finishes, drops guard.
        //   6. Reembed unblocks, proceeds with rebuild snapshot.
        //   7. After reembed, switch_to_normal.
        //   8. Writer C enters Normal again, gets guard.
        let r = Arc::new(WriteRouter::new());

        // Writer A: holds guard for 200ms.
        let r_a = Arc::clone(&r);
        let a_handle = thread::spawn(move || {
            let g = r_a.try_enter_sync_writer().expect("A: Normal yields guard");
            thread::sleep(Duration::from_millis(200));
            drop(g);
        });
        // Let A acquire — poll instead of fixed sleep (macos-14
        // aarch64 scheduler warm-up flake; see sibling fixes above).
        let acquire_deadline = std::time::Instant::now() + Duration::from_millis(500);
        while r.inflight() == 0 && std::time::Instant::now() < acquire_deadline {
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(r.inflight(), 1);

        // Reembed cutover: flip state.
        r.switch_to_queueing();
        assert_eq!(r.state(), RouterState::Queueing);

        // Writer B: must get None.
        assert!(
            r.try_enter_sync_writer().is_none(),
            "B: Queueing must reject sync writers"
        );

        // Reembed waits for A to drain.
        let drain_start = std::time::Instant::now();
        r.wait_for_no_sync_writers();
        let drain_dur = drain_start.elapsed();
        assert!(
            drain_dur >= Duration::from_millis(100),
            "wait_for should have blocked >=100ms waiting for A (took {drain_dur:?})"
        );
        assert_eq!(r.inflight(), 0);

        // Reembed completes; switch back to Normal.
        r.switch_to_normal();
        assert_eq!(r.state(), RouterState::Normal);

        // Writer C: Normal again, gets guard.
        let g_c = r
            .try_enter_sync_writer()
            .expect("C: post-reembed Normal must yield guard");
        assert_eq!(r.inflight(), 1);
        drop(g_c);
        assert_eq!(r.inflight(), 0);

        a_handle.join().unwrap();
    }

    #[test]
    fn guard_drop_is_panic_safe() {
        // If a sync writer panics while holding the guard, Drop must
        // still run and the counter must stay coherent. This is the
        // key reason we use RAII instead of atomic-counter +
        // explicit-decrement.
        let r = Arc::new(WriteRouter::new());

        let r_clone = Arc::clone(&r);
        let panicker = thread::spawn(move || {
            let _g = r_clone.try_enter_sync_writer().unwrap();
            panic!("simulated writer panic mid-write");
        });
        // We expect the thread to panic; suppress the propagated
        // panic + verify inflight returned to 0.
        let res = panicker.join();
        assert!(res.is_err(), "writer thread should have panicked");
        assert_eq!(
            r.inflight(),
            0,
            "guard Drop must run even on panic — RAII invariant"
        );
    }
}
