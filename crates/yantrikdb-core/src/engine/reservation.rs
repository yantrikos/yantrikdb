//! The reserve → commit → publish protocol's RAII guard (v0.10 Item 4a.6a
//! finding 2).
//!
//! Every writer that puts a vector into the delta index BEFORE its SQL is
//! durable owes one of two obligations, and which one it owes INVERTS at the
//! commit point. Both halves must survive an unwinding panic. This module is the
//! single owner of that rule.
//!
//! ## Why it must be RAII, and why a kill proof cannot cover it
//!
//! - **Before commit** — nothing durable exists, so the reservation must be
//!   REMOVED. `append_reserved` consumes delta capacity with an entry search
//!   skips and compaction deliberately RETAINS rather than seals (the property
//!   that makes rollback safe). A leaked reservation is therefore PERMANENT:
//!   nobody else will ever remove it, it holds capacity until restart, and
//!   enough of them wedge every writer into `Backpressure`.
//! - **After commit** — the row is durable, so the reservation must be PUBLISHED
//!   (and any pending op counted). A panic here leaves a durable row whose vector
//!   is invisible until an index rebuild.
//!
//! Restart repairs both — the index rebuilds from `memories`, the counter
//! re-seeds — which is exactly why the kill proofs do NOT cover this. The engine
//! deliberately uses non-poisoning locks, so a `catch_unwind` process CONTINUES
//! carrying the leak. That continuing process is the whole exposure.
//!
//! ## Why the pending-op count is a parameter and not part of the rule
//!
//! `record()` owes `publish + count` because its transaction enqueues a PENDING
//! op (`applied = 0`, via `log_op_pending_in_tx`) that `pending_op_count` caches.
//! The correction paths owe `publish` ALONE: `log_op_in_tx` commits `applied = 1`,
//! so they never create a pending op. Counting there would inflate the cache
//! against zero pending rows — monotonic drift with nothing to bring it back down
//! (`mark_op_applied` only decrements rows it actually transitions), and at
//! `MAX_PENDING_OPS` of drift the admission check wedges EVERY foreground write
//! into `Backpressure` forever. That is the v0.7.1 counter-leak class.
//!
//! So the obligation is shared; the counting is not. Copying `record()`'s guard
//! wholesale onto a correction would have introduced precisely the bug the guard
//! exists to prevent, in the other direction.

use std::sync::atomic::{AtomicI64, Ordering};

use super::reembed::SearchState;

/// Which obligation the write currently owes.
enum ResPhase {
    /// Reserved, nothing durable. Owe: remove the reservation.
    Reserved,
    /// Durable. Owe: publish the vector (and count the pending op, if any).
    Committed,
    /// All obligations discharged.
    Done,
}

/// PHASE-AWARE RAII for `reserve → commit → publish`.
///
/// Construct immediately after `append_reserved`. Call [`Self::mark_committed`]
/// the instant the transaction commits — that is the point the obligation
/// inverts, and nothing fallible may sit between the commit and that call.
/// [`Self::complete`] discharges the post-commit work on the happy path; `Drop`
/// discharges whichever obligation is still owed on any unwind.
pub(crate) struct ReservationGuard<'a> {
    state: &'a SearchState,
    /// `Some` only for writers whose committed transaction enqueued a pending
    /// op. See the module doc: this is per-site because the RATIONALE is
    /// per-site, not because the rule is.
    pending_op_count: Option<&'a AtomicI64>,
    rid: &'a str,
    seq: u64,
    /// Chunked embeddings: synthetic window keys (`{rid}#c{idx}`) reserved
    /// at the SAME seq as the parent, owing the same obligation at the same
    /// moment — a chunked write is ONE write with one commit point. Owned
    /// because the keys are minted after the guard's parent borrow.
    chunk_keys: Vec<String>,
    phase: ResPhase,
}

impl<'a> ReservationGuard<'a> {
    /// A writer whose transaction enqueues a pending op: post-commit it owes
    /// `publish` AND the `pending_op_count` increment. (`record()`.)
    pub(crate) fn with_pending_op(
        state: &'a SearchState,
        pending_op_count: &'a AtomicI64,
        rid: &'a str,
        seq: u64,
    ) -> Self {
        Self {
            state,
            pending_op_count: Some(pending_op_count),
            rid,
            seq,
            chunk_keys: Vec::new(),
            phase: ResPhase::Reserved,
        }
    }

    /// A writer whose transaction commits an ALREADY-APPLIED op: post-commit it
    /// owes `publish` only. Counting here would inflate `pending_op_count`
    /// against zero pending rows. (The correction paths.)
    pub(crate) fn publish_only(state: &'a SearchState, rid: &'a str, seq: u64) -> Self {
        Self {
            state,
            pending_op_count: None,
            rid,
            seq,
            chunk_keys: Vec::new(),
            phase: ResPhase::Reserved,
        }
    }

    /// Track a chunk-window key reserved at this write's seq. Call after
    /// each successful `append_reserved` of a chunk vector, BEFORE anything
    /// fallible follows — from that instant the guard owns the entry's
    /// obligation (removal pre-commit, publish post-commit).
    pub(crate) fn add_chunk_key(&mut self, key: String) {
        self.chunk_keys.push(key);
    }

    /// The transaction committed: from here the obligation is publish (+count).
    pub(crate) fn mark_committed(&mut self) {
        self.phase = ResPhase::Committed;
    }

    /// Upgrade a `publish_only` guard to ALSO owe the pending-op count —
    /// called at the moment the surrounding transaction actually ENQUEUES a
    /// pending row (4a.6d-3: `record_with_rid` only learns whether it will
    /// enqueue after `was_new_row` resolves INSIDE the transaction, so the
    /// obligation cannot be chosen at construction). Safe at any pre-commit
    /// point: an unwind before `mark_committed` still takes the Reserved arm
    /// (removal, no count — the rollback removed the pending row too), and
    /// after commit both discharge paths publish AND count.
    pub(crate) fn count_pending_op_on_completion(&mut self, counter: &'a AtomicI64) {
        self.pending_op_count = Some(counter);
    }

    /// Discharge the post-commit obligations exactly once. Returns whether
    /// `publish` found the reservation.
    pub(crate) fn complete(&mut self) -> bool {
        let published = self.discharge_committed();
        self.phase = ResPhase::Done;
        published
    }

    /// publish + count. Idempotent-by-phase: only ever runs once, either from
    /// `complete()` or from `Drop` on a post-commit unwind.
    ///
    /// The PARENT publishes first, then its chunk keys: a chunk hit collapses
    /// to the parent rid at search time, so the parent must be findable
    /// before any window is — the other order would be a (momentary) result
    /// for a record the delta does not yet serve.
    fn discharge_committed(&self) -> bool {
        let published = self.state.vec_index.publish(self.rid, self.seq);
        for key in &self.chunk_keys {
            self.state.vec_index.publish(key, self.seq);
        }
        if let Some(counter) = self.pending_op_count {
            counter.fetch_add(1, Ordering::Relaxed);
        }
        published
    }
}

impl Drop for ReservationGuard<'_> {
    fn drop(&mut self) {
        match self.phase {
            // Always succeeds: reserved entries are never sealed. A removal, NOT
            // a tombstone — a tombstone would suppress the rid and hide a
            // still-valid older vector.
            ResPhase::Reserved => {
                self.state.vec_index.remove_appended(self.rid, self.seq);
                for key in &self.chunk_keys {
                    self.state.vec_index.remove_appended(key, self.seq);
                }
            }
            // Unwound after commit. The write IS durable, so finish the job
            // rather than strand it.
            ResPhase::Committed => {
                self.discharge_committed();
            }
            ResPhase::Done => {}
        }
    }
}

/// The batch analogue of [`ReservationGuard`] (v0.10 Item 4a.6d-2a, #92):
/// one phase, N `(rid, seq)` reservations, all owing the SAME obligation at
/// the same moment — a batch is one write with one commit point.
///
/// `record_batch` reserves delta capacity for EVERY item before its savepoint
/// opens, so capacity exhaustion (the old post-RELEASE append failure)
/// surfaces before a single durable byte — there is nothing to compensate,
/// which is what actually closes #92: the compensating DELETE reversed rows
/// and session counts but could never reverse `entities` upserts,
/// `memory_entities` links, or the in-memory graph_index.
///
/// Always publish-only, and deliberately WITHOUT a `with_pending_op`
/// constructor: `record_batch`'s ops commit `applied = 1` (in-savepoint
/// `log_op_in_tx` since 4a.6d-2b), so it never enqueues a pending op. Counting here
/// would inflate `pending_op_count` against zero pending rows — the v0.7.1
/// counter-leak class the module doc describes. If a queued-batch primitive
/// ever exists, add the constructor WITH its rationale; do not default to
/// counting.
pub(crate) struct BatchReservationGuard<'a> {
    state: &'a SearchState,
    /// Owned because the rids outlive no caller borrow this early: the guard
    /// is constructed before the batch's SQL loop mints its row parameters.
    entries: Vec<(String, u64)>,
    phase: ResPhase,
}

impl<'a> BatchReservationGuard<'a> {
    pub(crate) fn new(state: &'a SearchState, capacity: usize) -> Self {
        Self {
            state,
            entries: Vec::with_capacity(capacity),
            phase: ResPhase::Reserved,
        }
    }

    /// Reserve one item. On `Err` the FAILED item was never reserved (the
    /// delta rejects before inserting), and every PRIOR reservation is still
    /// held by this guard — the caller just returns and `Drop` removes them
    /// all. Partial reservation must never leak: that capacity is
    /// unreclaimable by anyone else (compaction retains unpublished entries).
    pub(crate) fn reserve(
        &mut self,
        rid: String,
        embedding: Vec<f32>,
        seq: u64,
    ) -> crate::error::Result<()> {
        // Batch rids are freshly minted, so an already-present (rid, seq)
        // is an invariant violation — and it must NOT be pushed as an entry
        // (this guard would then publish/remove a vector it doesn't own).
        if self
            .state
            .vec_index
            .append_reserved(rid.clone(), embedding, seq)?
            == crate::vector::delta_index::ReservedAppend::AlreadyPresent
        {
            return Err(crate::error::YantrikDbError::InvalidInput(format!(
                "freshly minted rid {rid} already present in the delta at seq \
                 {seq} — engine invariant violation"
            )));
        }
        self.entries.push((rid, seq));
        Ok(())
    }

    /// The batch's savepoint RELEASEd: all N obligations invert to publish.
    /// Nothing fallible may sit between the RELEASE and this call.
    pub(crate) fn mark_committed(&mut self) {
        self.phase = ResPhase::Committed;
    }

    /// Publish all N exactly once. Returns whether every publish found its
    /// reservation.
    pub(crate) fn complete(&mut self) -> bool {
        let all_published = self.discharge_committed();
        self.phase = ResPhase::Done;
        all_published
    }

    fn discharge_committed(&self) -> bool {
        let mut all_published = true;
        for (rid, seq) in &self.entries {
            all_published &= self.state.vec_index.publish(rid, *seq);
        }
        all_published
    }
}

impl Drop for BatchReservationGuard<'_> {
    fn drop(&mut self) {
        match self.phase {
            ResPhase::Reserved => {
                for (rid, seq) in &self.entries {
                    self.state.vec_index.remove_appended(rid, *seq);
                }
            }
            ResPhase::Committed => {
                self.discharge_committed();
            }
            ResPhase::Done => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::YantrikDB;

    /// These pin the RULE the guard owns, in the DEFAULT test suite — no
    /// `testing` feature, so CI runs them on every PR regardless of the
    /// failpoint seam.
    ///
    /// They exercise the property a kill proof structurally cannot: the engine
    /// uses non-poisoning locks, so a caught panic leaves the PROCESS ALIVE
    /// holding whatever the unwind skipped. A killed process is repaired by
    /// restart; this one is not.
    fn db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn drop_before_commit_removes_the_reservation() {
        let db = db();
        let state = db.search_state.load_full();
        let _ = state
            .vec_index
            .append_reserved("r1".into(), vec![0.1; 8], 7)
            .unwrap();

        {
            let _g = ReservationGuard::publish_only(&state, "r1", 7);
            // falls out of scope still Reserved
        }

        assert!(
            !state.vec_index.remove_appended("r1", 7),
            "guard's Drop must have removed the reservation (nothing left to remove)"
        );
    }

    #[test]
    fn unwind_before_commit_removes_the_reservation() {
        let db = db();
        let state = db.search_state.load_full();
        let _ = state
            .vec_index
            .append_reserved("r2".into(), vec![0.1; 8], 8)
            .unwrap();

        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = ReservationGuard::publish_only(&state, "r2", 8);
            panic!("simulated panic between reserve and commit");
        }));
        assert!(res.is_err(), "panic must propagate");

        // THE bug this guard exists for: a leaked reservation is permanent —
        // compaction retains unpublished entries, so nobody ever reclaims the
        // capacity, and enough of them wedge every writer into Backpressure.
        assert!(
            !state.vec_index.remove_appended("r2", 8),
            "unwind must not leak the reservation"
        );
    }

    #[test]
    fn unwind_after_commit_publishes_rather_than_stranding() {
        let db = db();
        let state = db.search_state.load_full();
        let _ = state
            .vec_index
            .append_reserved("r3".into(), vec![0.1; 8], 9)
            .unwrap();

        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut g = ReservationGuard::publish_only(&state, "r3", 9);
            g.mark_committed();
            panic!("simulated panic between commit and publish");
        }));
        assert!(res.is_err(), "panic must propagate");

        // The write IS durable at this point, so the obligation inverted: the
        // vector must be published, not removed. Otherwise the row exists with an
        // invisible vector until an index rebuild.
        assert!(
            !state.vec_index.publish("r3", 9),
            "Drop must have published it already (a second publish finds nothing unpublished)"
        );
        assert!(
            state.vec_index.remove_appended("r3", 9),
            "the entry must still EXIST — post-commit it is published, never removed"
        );
    }

    #[test]
    fn complete_then_drop_discharges_exactly_once() {
        let db = db();
        let state = db.search_state.load_full();
        let _ = state
            .vec_index
            .append_reserved("r4".into(), vec![0.1; 8], 10)
            .unwrap();
        let before = db.pending_op_count.load(Ordering::Relaxed);

        {
            let mut g = ReservationGuard::with_pending_op(&state, &db.pending_op_count, "r4", 10);
            g.mark_committed();
            assert!(g.complete(), "complete() publishes and reports it");
            // Drop runs here in Done phase and must NOT count again.
        }

        assert_eq!(
            db.pending_op_count.load(Ordering::Relaxed),
            before + 1,
            "exactly one increment — complete() then Drop must not double-count"
        );
    }

    /// The counting is per-SITE because the rationale is per-site. The
    /// correction paths commit `applied = 1` ops (log_op_in_tx) and enqueue
    /// nothing pending, so counting there would inflate the cache against zero
    /// pending rows — monotonic drift with nothing to bring it down, which at
    /// MAX_PENDING_OPS wedges every foreground write into Backpressure forever.
    /// That is the v0.7.1 counter-leak class, and it is exactly what copying
    /// record()'s guard wholesale onto a correction would have introduced.
    #[test]
    fn publish_only_never_touches_the_pending_counter() {
        let db = db();
        let state = db.search_state.load_full();
        let before = db.pending_op_count.load(Ordering::Relaxed);

        // committed-and-completed
        let _ = state
            .vec_index
            .append_reserved("r5".into(), vec![0.1; 8], 11)
            .unwrap();
        {
            let mut g = ReservationGuard::publish_only(&state, "r5", 11);
            g.mark_committed();
            g.complete();
        }
        // committed-and-unwound (Drop discharges)
        let _ = state
            .vec_index
            .append_reserved("r6".into(), vec![0.1; 8], 12)
            .unwrap();
        {
            let mut g = ReservationGuard::publish_only(&state, "r6", 12);
            g.mark_committed();
        }

        assert_eq!(
            db.pending_op_count.load(Ordering::Relaxed),
            before,
            "publish_only must never move pending_op_count, on either discharge path"
        );
    }

    // ── BatchReservationGuard (4a.6d-2a) ──

    /// THE #92 shape: some items reserve, a later one fails (capacity/dim),
    /// the batch returns Err — every reservation already taken must be
    /// removed, or its capacity is held until restart.
    #[test]
    fn batch_drop_before_commit_removes_every_reservation() {
        let db = db();
        let state = db.search_state.load_full();

        {
            let mut g = BatchReservationGuard::new(&state, 3);
            g.reserve("b1".into(), vec![0.1; 8], 21).unwrap();
            g.reserve("b2".into(), vec![0.2; 8], 22).unwrap();
            // A third item WOULD have failed here; the caller `?`-returns and
            // the guard falls out of scope still Reserved.
        }

        assert!(
            !state.vec_index.remove_appended("b1", 21),
            "first reservation must have been removed by Drop"
        );
        assert!(
            !state.vec_index.remove_appended("b2", 22),
            "second reservation must have been removed by Drop"
        );
    }

    #[test]
    fn batch_unwind_after_commit_publishes_all_rather_than_stranding() {
        let db = db();
        let state = db.search_state.load_full();

        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut g = BatchReservationGuard::new(&state, 2);
            g.reserve("b3".into(), vec![0.1; 8], 23).unwrap();
            g.reserve("b4".into(), vec![0.2; 8], 24).unwrap();
            g.mark_committed();
            panic!("simulated panic between the batch RELEASE and publish");
        }));
        assert!(res.is_err(), "panic must propagate");

        // Post-commit the rows are durable: every vector must be published,
        // none removed.
        for (rid, seq) in [("b3", 23u64), ("b4", 24u64)] {
            assert!(
                !state.vec_index.publish(rid, seq),
                "{rid} must already be published by Drop"
            );
            assert!(
                state.vec_index.remove_appended(rid, seq),
                "{rid} must still EXIST — post-commit it is published, never removed"
            );
        }
    }

    #[test]
    fn batch_complete_then_drop_discharges_exactly_once_and_never_counts() {
        let db = db();
        let state = db.search_state.load_full();
        let before = db.pending_op_count.load(Ordering::Relaxed);

        {
            let mut g = BatchReservationGuard::new(&state, 2);
            g.reserve("b5".into(), vec![0.1; 8], 25).unwrap();
            g.reserve("b6".into(), vec![0.2; 8], 26).unwrap();
            g.mark_committed();
            assert!(g.complete(), "complete() publishes all and reports it");
            // Drop runs in Done phase: no second publish, no removal.
        }

        assert!(
            state.vec_index.remove_appended("b5", 25),
            "published entry must survive the Done-phase Drop"
        );
        // record_batch commits applied=1 ops (in-savepoint log_op_in_tx), so the
        // batch guard has NO counting mode at all — the v0.7.1 counter-leak
        // class, kept impossible by construction.
        assert_eq!(
            db.pending_op_count.load(Ordering::Relaxed),
            before,
            "the batch guard must never move pending_op_count"
        );
    }
}
