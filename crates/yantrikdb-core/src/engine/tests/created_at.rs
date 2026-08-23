//! Caller-supplied `created_at` (historical import, 0.14).
//!
//! The contract under test is `RecordInput::created_at`'s: `Some(ts)` lands
//! the event time in `created_at`, `updated_at`, AND `last_access` (decay
//! runs from the event, not the import), flows into the replicated op
//! payload, joins the idempotency digest (a re-dated write is a different
//! write), and makes `recall_as_of`/`time_window` meaningful on bulk-loaded
//! corpora. `None` is byte-for-byte the pre-field behavior. Motivated by the
//! BEAM/AMB finding that on a bulk-loaded conversation corpus every temporal
//! surface collapsed onto the ingest wall-clock — `recall_as_of` was inert,
//! `time_window` was inert, and decay/recency were pure insertion-order
//! noise (the 2026-06-08 benchmark diagnosis, this time fixed at the write
//! path instead of worked around with pinned weights).

use super::*;

/// The row triple: `Some(ts)` stamps created_at, updated_at AND last_access
/// with the event time — an imported record was last touched when it
/// happened, so decay runs from then.
#[test]
fn record_with_created_at_stamps_the_full_row_triple() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let event_time = 1_600_000_000.0; // 2020-09-13, unambiguously "the past"
    let rid = db
        .record_with_idempotency(
            "joined the observatory team",
            "episodic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "work",
            "user",
            None,
            None,
            Some(event_time),
        )
        .unwrap();

    let (created, updated, last_access): (f64, f64, f64) = db
        .conn()
        .query_row(
            "SELECT created_at, updated_at, last_access FROM memories WHERE rid = ?1",
            params![rid],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(created, event_time);
    assert_eq!(updated, event_time);
    assert_eq!(
        last_access, event_time,
        "last_access must be the event time — decay from the import instant \
         would give every imported record a fresh-memory decay curve"
    );
}

/// `None` still stamps now() — the pre-field behavior, byte-for-byte.
#[test]
fn record_without_created_at_still_stamps_now() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let before = crate::engine::now();
    let rid = db
        .record(
            "an ordinary present-day write",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(2.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    let after = crate::engine::now();
    let created: f64 = db
        .conn()
        .query_row(
            "SELECT created_at FROM memories WHERE rid = ?1",
            params![rid],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        created >= before && created <= after,
        "None must stamp the engine clock: {created} outside [{before}, {after}]"
    );
}

/// The point of the feature: on a bulk-loaded corpus with real event times,
/// `recall_as_of(t)` distinguishes what existed at `t` — instead of every
/// record sharing the ingest wall-clock and the filter being inert.
#[test]
fn recall_as_of_respects_imported_event_times() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let january = 1_735_700_000.0;
    let march = 1_740_800_000.0;
    let april = 1_743_500_000.0;

    let rid_old = db
        .record_with_idempotency(
            "works at the observatory",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "work",
            "user",
            None,
            None,
            Some(january),
        )
        .unwrap();
    let rid_new = db
        .record_with_idempotency(
            "moved to the planetarium",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.05, 8),
            "default",
            0.8,
            "work",
            "user",
            None,
            None,
            Some(april),
        )
        .unwrap();

    // As of March, only the January record existed.
    let hits = db
        .recall_as_of(&vec_seed(1.0, 8), 10, march, None, None)
        .unwrap();
    let rids: Vec<&str> = hits.iter().map(|r| r.rid.as_str()).collect();
    assert!(
        rids.contains(&rid_old.as_str()),
        "January record missing at t=March"
    );
    assert!(
        !rids.contains(&rid_new.as_str()),
        "April record visible at t=March — created_at did not reach the as-of filter"
    );

    // Today, both exist.
    let now_hits = db
        .recall_as_of(&vec_seed(1.0, 8), 10, crate::engine::now(), None, None)
        .unwrap();
    let now_rids: Vec<&str> = now_hits.iter().map(|r| r.rid.as_str()).collect();
    assert!(now_rids.contains(&rid_old.as_str()));
    assert!(now_rids.contains(&rid_new.as_str()));
}

/// record_batch: per-item event times land per-row, mixed with None items
/// that stamp now().
#[test]
fn record_batch_carries_per_item_created_at() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let event_time = 1_650_000_000.0;
    let mk = |text: &str, seed: f32, created_at: Option<f64>| RecordInput {
        text: text.into(),
        memory_type: "episodic".into(),
        importance: 0.5,
        valence: 0.0,
        half_life: 604800.0,
        metadata: empty_meta(),
        embedding: vec_seed(seed, 8),
        namespace: "default".into(),
        certainty: 0.8,
        domain: "general".into(),
        source: "user".into(),
        emotional_state: None,
        idempotency_key: None,
        created_at,
    };
    let before = crate::engine::now();
    let rids = db
        .record_batch(&[
            mk("historical item", 1.0, Some(event_time)),
            mk("present item", 2.0, None),
        ])
        .unwrap();

    let created_hist: f64 = db
        .conn()
        .query_row(
            "SELECT created_at FROM memories WHERE rid = ?1",
            params![rids[0]],
            |r| r.get(0),
        )
        .unwrap();
    let created_now: f64 = db
        .conn()
        .query_row(
            "SELECT created_at FROM memories WHERE rid = ?1",
            params![rids[1]],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(created_hist, event_time);
    assert!(
        created_now >= before,
        "None item must stamp the engine clock"
    );
}

/// The event time must reach the RECALL path, not just the SQL row.
///
/// Recall scores from the in-memory `scoring_cache`, not from `memories`, so
/// the two can disagree: `record_batch` shipped the correct `created_at` to
/// SQL while its cache insert still stamped `now()`, and every through-recall
/// consumer (decay, recency, `recall_as_of`, `order="recency"`) saw today for
/// a record whose row said 2020. The SQL-reading tests above all passed —
/// only a through-recall assertion catches it. `record_with_rid` had this
/// right from the start; the batch path was the copied-without-the-rationale
/// sibling.
#[test]
fn imported_event_time_reaches_recall_not_just_the_row() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let event_time = 1_600_000_000.0;
    let mk = |text: &str, seed: f32, created_at: Option<f64>| RecordInput {
        text: text.into(),
        memory_type: "episodic".into(),
        importance: 0.5,
        valence: 0.0,
        half_life: 604800.0,
        metadata: empty_meta(),
        embedding: vec_seed(seed, 8),
        namespace: "default".into(),
        certainty: 0.8,
        domain: "general".into(),
        source: "user".into(),
        emotional_state: None,
        idempotency_key: None,
        created_at,
    };
    let rids = db
        .record_batch(&[mk("batch historical record", 1.0, Some(event_time))])
        .unwrap();

    let hits = db
        .recall(
            &vec_seed(1.0, 8),
            5,
            None,
            None,
            false,
            false,
            None,
            true,
            None,
            None,
            None,
            None,
            None,
            false,
            None, // event_after (#149)
            None, // event_before (#149)
        )
        .unwrap();
    let hit = hits
        .iter()
        .find(|h| h.rid == rids[0])
        .expect("batch record must be recallable");
    assert_eq!(
        hit.created_at, event_time,
        "recall reported {} for a record imported at {event_time} — the \
         scoring cache disagrees with the row",
        hit.created_at
    );
}

/// A non-finite event time is refused in prevalidation — before any side
/// effect, failing the WHOLE batch like every other scalar gate.
#[test]
fn non_finite_created_at_is_refused_before_any_side_effect() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let mk = |created_at: Option<f64>| RecordInput {
        text: "t".into(),
        memory_type: "episodic".into(),
        importance: 0.5,
        valence: 0.0,
        half_life: 604800.0,
        metadata: empty_meta(),
        embedding: vec_seed(1.0, 8),
        namespace: "default".into(),
        certainty: 0.8,
        domain: "general".into(),
        source: "user".into(),
        emotional_state: None,
        idempotency_key: None,
        created_at,
    };
    // Batch: a bad element late in the batch rejects the whole batch.
    let err = db
        .record_batch(&[mk(None), mk(Some(f64::NAN))])
        .unwrap_err();
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::InvalidScalar {
                field: "created_at",
                ..
            }
        ),
        "expected InvalidScalar on created_at, got {err:?}"
    );
    let count: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0, "refused batch left rows behind");

    // Single-record surface, same gate.
    let err = db
        .record_with_idempotency(
            "t",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
            None,
            Some(f64::INFINITY),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        crate::error::YantrikDbError::InvalidScalar {
            field: "created_at",
            ..
        }
    ));
}

/// The digest half of the contract: same key + same payload + same
/// created_at is a retry (original rid, nothing written); same key with a
/// DIFFERENT created_at is a different write — typed conflict, first write
/// stands. A re-dated record decays and recall_as_ofs differently, exactly
/// like a re-vectored one.
#[test]
fn idempotency_treats_a_redated_write_as_a_different_write() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let write = |ts: f64| {
        db.record_with_idempotency(
            "the launch happened",
            "episodic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "work",
            "user",
            None,
            Some("launch-evt-1"),
            Some(ts),
        )
    };
    let rid1 = write(1_600_000_000.0).unwrap();

    // Honest retry: same everything → original rid, one row.
    let rid2 = write(1_600_000_000.0).unwrap();
    assert_eq!(rid1, rid2);
    let count: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 1);

    // Re-dated: different write reusing the key → typed conflict.
    let err = write(1_600_000_001.0).unwrap_err();
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::IdempotencyConflict { .. }
        ),
        "a re-dated keyed write must conflict, got {err:?}"
    );
}

/// Scoring safety: a FUTURE event time scores as "brand new" (clamped), not
/// as an unbounded amplifier. Before the clamp, decay = importance·2^(+x)
/// and recency = e^(+x) grew without bound for future-dated records — the
/// recency wall rebuilt in the other direction, reachable the moment
/// created_at became caller-supplied.
#[test]
fn future_created_at_scores_as_new_not_amplified() {
    use crate::base::scoring::{decay_score, recency_score};
    // Direct unit contract on the clamp:
    assert_eq!(decay_score(0.5, 604800.0, -3_000_000.0), 0.5);
    assert!(recency_score(-3_000_000.0) <= 1.0);
    assert_eq!(recency_score(-3_000_000.0), 1.0);

    // End-to-end: a future-dated decoy must not leapfrog a more similar
    // present-day record purely on its timestamp.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let target = db
        .record(
            "the target memory about the launch",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    let one_year_ahead = crate::engine::now() + 365.0 * 86400.0;
    db.record_with_idempotency(
        "an unrelated future-dated decoy",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(4.0, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
        None,
        Some(one_year_ahead),
    )
    .unwrap();

    let hits = db
        .recall(
            &vec_seed(1.0, 8),
            1,
            None,
            None,
            false,
            false,
            None,
            true,
            None,
            None,
            None,
            None,
            None,
            false,
            None, // event_after (#149)
            None, // event_before (#149)
        )
        .unwrap();
    assert_eq!(
        hits[0].rid, target,
        "future-dated decoy outranked the similar record — the clamp is not \
         holding at the recall path"
    );
}

/// The replicated op payload carries the caller's event time, so followers
/// and the queued-route materializer stamp the same row the origin did.
#[test]
fn record_op_payload_carries_the_imported_event_time() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let event_time = 1_600_000_000.0;
    let rid = db
        .record_with_idempotency(
            "replicated historical record",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
            None,
            Some(event_time),
        )
        .unwrap();
    let payload: String = db
        .conn()
        .query_row(
            "SELECT payload FROM oplog WHERE op_type = 'record' AND target_rid = ?1",
            params![rid],
            |r| r.get(0),
        )
        .unwrap();
    let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
    assert_eq!(
        v["created_at"].as_f64(),
        Some(event_time),
        "op payload must carry the event time — a follower would otherwise \
         materialize a different row than the origin's"
    );
}
