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
