//! rebuild_vec_index() and the background compactor exclude each other.
//!
//! A compaction clones the cold tier, merges the delta into the clone and
//! installs it; a rebuild reads SQL and installs a tier built from it.
//! Interleaved, either overwrote the other: a compaction installed its
//! merge over a rebuilt tier, or a compaction finishing mid-rebuild retired
//! delta entries the rebuild's earlier SQL read did not contain. The
//! rebuild now takes the compaction gate before its SQL read.

use super::*;
use std::time::Duration;

#[test]
fn rebuild_waits_for_an_in_flight_compaction() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    for i in 0..5 {
        db.record(
            &format!("row {i}"),
            "semantic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(i as f32, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    }

    // Stand in for a compaction in flight.
    let vec_index = std::sync::Arc::clone(&db.search_state.load().vec_index);
    let gate = vec_index.pause_compaction();

    let rebuilt = std::thread::scope(|s| {
        let rebuild = s.spawn(|| db.rebuild_vec_index());
        std::thread::sleep(Duration::from_millis(150));
        assert!(
            !rebuild.is_finished(),
            "a rebuild must not read SQL or install while a compaction holds the gate"
        );
        drop(gate);
        rebuild.join().unwrap()
    });
    assert_eq!(rebuilt.unwrap(), 5);
}

/// Where the rebuild holds the gate is the whole fix: from before its SQL
/// read (or a compaction can retire entries the read missed) through the
/// install (or one can land between them). Observed from inside the
/// rebuild through its test hooks, so the check needs no timing.
#[test]
fn rebuild_holds_the_gate_from_before_its_read_through_the_install() {
    use crate::engine::indices::rebuild_hooks;
    use std::sync::{Arc, Mutex};

    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record(
        "a row",
        "semantic",
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

    let seen: Arc<Mutex<Vec<(&'static str, bool)>>> = Arc::default();
    let log = Arc::clone(&seen);
    rebuild_hooks::set(Box::new(move |stage, idx| {
        log.lock()
            .unwrap()
            .push((stage, idx.compaction_gate_is_held()));
    }));
    let rebuilt = db.rebuild_vec_index();
    rebuild_hooks::clear();
    rebuilt.unwrap();

    assert_eq!(
        *seen.lock().unwrap(),
        vec![("before_sql_read", true), ("before_install", true)],
        "the compaction gate must be held at both points"
    );
}

/// A delta entry published before the rebuild's SQL read is subsumed by
/// the rebuilt tier. Left in the delta it shadowed the rebuilt vector, and
/// the next compaction merged the stale one back over it — e.g. a SQL-only
/// repair of a row whose original vector was still in the delta.
#[test]
fn rebuild_retires_the_delta_entries_its_read_subsumes() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let stale = vec_seed(1.0, 8);
    let rid = db
        .record(
            "a row repaired in SQL",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &stale,
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    // The repair writes SQL only; the original vector stays in the delta.
    let mut repaired = vec![0.0f32; 8];
    repaired[7] = 1.0;
    db.conn()
        .execute(
            "UPDATE memories SET embedding = ?1 WHERE rid = ?2",
            params![crate::serde_helpers::serialize_f32(&repaired), rid],
        )
        .unwrap();

    db.rebuild_vec_index().unwrap();
    let vec_index = std::sync::Arc::clone(&db.search_state.load().vec_index);
    let nearest = |q: &[f32]| vec_index.search(q, 1).unwrap()[0].clone();
    let (hit, dist) = nearest(&repaired);
    assert_eq!(hit, rid);
    assert!(
        dist < 1e-3,
        "the index must serve the repaired vector: {dist}"
    );

    vec_index.compact().unwrap();
    let (hit, dist) = nearest(&repaired);
    assert_eq!(hit, rid);
    assert!(
        dist < 1e-3,
        "a compaction must not merge the stale vector back: {dist}"
    );
}
