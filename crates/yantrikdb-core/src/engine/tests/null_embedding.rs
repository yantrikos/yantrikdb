//! Rows with no vector are a normal state, not corruption.
//!
//! A replication follower inserts memories without their embedding (the op
//! carries only a hash), so they arrive active and hot with
//! `embedding IS NULL` until `refresh_embeddings` fills them in. Every path
//! that reads `memories.embedding` must cope: `think()` (consolidation's
//! page query), `archive()`, `hydrate()` and `evict()` used to fail with
//! "Invalid column type Null ... name: embedding".

use super::*;

fn record(db: &YantrikDB, text: &str, seed: f32) -> String {
    db.record(
        text,
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(seed, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap()
}

/// A store holding near-duplicate memories (so consolidation has clusters
/// to form) plus `n_bare` rows whose vector is missing, as a follower's
/// replicated rows are.
fn store_with_bare_rows(n_bare: usize) -> (YantrikDB, Vec<String>) {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    for i in 0..12 {
        record(
            &db,
            &format!("Alice met Bob at the harbour, visit {i}"),
            1.0 + (i % 3) as f32 * 0.001,
        );
    }
    let mut bare = Vec::new();
    for i in 0..n_bare {
        let rid = record(&db, &format!("replicated note {i} about the harbour"), 1.0);
        bare.push(rid);
    }
    {
        let conn = db.conn();
        for rid in &bare {
            conn.execute(
                "UPDATE memories SET embedding = NULL WHERE rid = ?1",
                params![rid],
            )
            .unwrap();
        }
    }
    (db, bare)
}

fn tier(db: &YantrikDB, rid: &str) -> String {
    db.conn()
        .query_row(
            "SELECT storage_tier FROM memories WHERE rid = ?1",
            params![rid],
            |r| r.get(0),
        )
        .unwrap()
}

#[test]
fn think_tolerates_rows_without_a_vector() {
    let (db, _bare) = store_with_bare_rows(5);
    db.think(&ThinkConfig::default())
        .expect("think() must skip rows with no vector, not fail on them");
}

#[test]
fn archive_and_hydrate_a_row_without_a_vector() {
    let (db, bare) = store_with_bare_rows(1);
    let rid = &bare[0];

    assert!(db.archive(rid).expect("archive() of a row with no vector"));
    assert_eq!(tier(&db, rid), "cold");
    let still_null: bool = db
        .conn()
        .query_row(
            "SELECT embedding IS NULL FROM memories WHERE rid = ?1",
            params![rid],
            |r| r.get(0),
        )
        .unwrap();
    assert!(still_null, "archive must not invent a vector");

    assert!(db.hydrate(rid).expect("hydrate() of a row with no vector"));
    assert_eq!(tier(&db, rid), "hot");

    // A row that had a vector still round-trips as before.
    let with_vec = record(&db, "a row with its vector", 3.0);
    assert!(db.archive(&with_vec).unwrap());
    assert!(db.hydrate(&with_vec).unwrap());
    let len: i64 = db
        .conn()
        .query_row(
            "SELECT length(embedding) FROM memories WHERE rid = ?1",
            params![with_vec],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(len, 8 * 4, "hydrate must restore the uncompressed vector");
}

#[test]
fn evict_tolerates_rows_without_a_vector() {
    let (db, bare) = store_with_bare_rows(4);
    let evicted = db
        .evict(3)
        .expect("evict() must not fail on rows with no vector");
    assert!(!evicted.is_empty());
    for rid in &bare {
        let t = tier(&db, rid);
        assert!(t == "hot" || t == "cold", "{rid}: {t}");
    }
}
