//! Rows with no vector are a normal state, not corruption.
//!
//! A replication follower inserts memories without their embedding (the op
//! carries only a hash), so they arrive active and hot with
//! `embedding IS NULL` and absent from the vector index, until
//! `refresh_embeddings` fills them in. Every path that reads
//! `memories.embedding` must cope: `think()` (consolidation's page query),
//! `archive()`, `hydrate()` and `evict()` used to fail with
//! "Invalid column type Null ... name: embedding".
//!
//! Cold rows hold their vector compressed. `refresh_embeddings` used to
//! treat every compressed cold blob as unusable and write a hot-format blob
//! into the cold row, and `hydrate()` decompressed unconditionally, so it
//! panicked inside zstd on such a row.

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

/// Blank the vector of `rids` and rebuild the index, so the rows look like
/// a follower's replicated rows: no vector in SQL, none in the index.
fn strip_vectors(db: &YantrikDB, rids: &[String]) {
    {
        let conn = db.conn();
        for rid in rids {
            conn.execute(
                "UPDATE memories SET embedding = NULL WHERE rid = ?1",
                params![rid],
            )
            .unwrap();
        }
    }
    db.rebuild_vec_index().unwrap();
}

/// A store holding near-duplicate memories (so consolidation has clusters
/// to form) plus `n_bare` rows with no vector.
fn store_with_bare_rows(n_bare: usize) -> (YantrikDB, Vec<String>) {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    for i in 0..12 {
        record(
            &db,
            &format!("Alice met Bob at the harbour, visit {i}"),
            1.0 + (i % 3) as f32 * 0.001,
        );
    }
    let bare: Vec<String> = (0..n_bare)
        .map(|i| record(&db, &format!("replicated note {i} about the harbour"), 1.0))
        .collect();
    strip_vectors(&db, &bare);
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

fn embedding_blob(db: &YantrikDB, rid: &str) -> Option<Vec<u8>> {
    db.conn()
        .query_row(
            "SELECT embedding FROM memories WHERE rid = ?1",
            params![rid],
            |r| r.get(0),
        )
        .unwrap()
}

#[test]
fn think_tolerates_rows_without_a_vector() {
    let (db, _bare) = store_with_bare_rows(5);
    let result = db
        .think(&ThinkConfig::default())
        .expect("think() must skip rows with no vector, not fail on them");
    assert!(
        result.consolidation_count >= 1,
        "the near-duplicates with vectors still consolidate"
    );
}

#[test]
fn archive_and_hydrate_a_row_without_a_vector() {
    let (db, bare) = store_with_bare_rows(1);
    let rid = &bare[0];

    assert!(db.archive(rid).expect("archive() of a row with no vector"));
    assert_eq!(tier(&db, rid), "cold");
    assert!(
        embedding_blob(&db, rid).is_none(),
        "archive must not invent a vector"
    );

    assert!(db.hydrate(rid).expect("hydrate() of a row with no vector"));
    assert_eq!(tier(&db, rid), "hot");
    assert!(
        embedding_blob(&db, rid).is_none(),
        "hydrate must not invent a vector"
    );

    // A row that had a vector still round-trips as before.
    let with_vec = record(&db, "a row with its vector", 3.0);
    assert!(db.archive(&with_vec).unwrap());
    assert!(crate::compression::is_compressed(
        &embedding_blob(&db, &with_vec).unwrap()
    ));
    assert!(db.hydrate(&with_vec).unwrap());
    assert_eq!(
        embedding_blob(&db, &with_vec).unwrap().len(),
        8 * 4,
        "hydrate must restore the uncompressed vector"
    );
}

#[test]
fn evict_tolerates_rows_without_a_vector() {
    let (db, bare) = store_with_bare_rows(4);
    let evicted = db
        .evict(3)
        .expect("evict() must not fail on rows with no vector");
    assert!(
        evicted.iter().any(|rid| bare.contains(rid)),
        "the test must actually evict a row with no vector: {evicted:?}"
    );
    for rid in evicted.iter().filter(|rid| bare.contains(rid)) {
        assert_eq!(tier(&db, rid), "cold");
    }
}

/// The scan cursor's wrap test must count the rows the page query can
/// return. Counting rows with no vector too made it walk pages that do not
/// exist: one empty think() tick per 100 such rows before new duplicates
/// were looked at.
#[test]
fn consolidation_cursor_ignores_rows_without_a_vector() {
    // 12 + 120 rows: more than one 100-row window if the rows with no
    // vector are counted, one window if they are not. (Under 256 writes:
    // these tests run no compactor.)
    let (db, _bare) = store_with_bare_rows(120);
    // limit 5 → a 100-row scan window; 12 rows have vectors.
    crate::consolidate::consolidate(&db, 0.9, 30.0, 2, 5, false, false).unwrap();
    let offset: String = db
        .conn()
        .query_row(
            "SELECT value FROM meta WHERE key = 'consolidation_scan_offset'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        offset, "0",
        "every row with a vector fits in one window, so the cursor wraps"
    );
}

/// A cold row holding an uncompressed blob (a text correct() or an old
/// refresh_embeddings wrote hot format into it) hydrates instead of
/// panicking inside zstd.
#[test]
fn hydrate_reads_an_uncompressed_blob_in_a_cold_row() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = record(&db, "a cold row with a hot-format blob", 5.0);
    assert!(db.archive(&rid).unwrap());
    db.conn()
        .execute(
            "UPDATE memories SET embedding = ?1 WHERE rid = ?2",
            params![crate::serde_helpers::serialize_f32(&vec_seed(5.0, 8)), rid],
        )
        .unwrap();

    assert!(db.hydrate(&rid).expect("hydrate must not panic"));
    assert_eq!(tier(&db, &rid), "hot");
    assert_eq!(embedding_blob(&db, &rid).unwrap().len(), 8 * 4);
}

#[cfg(feature = "bundled-embedder")]
mod with_engine_embedder {
    use super::*;

    fn store() -> (YantrikDB, Vec<String>) {
        let db = YantrikDB::with_default(":memory:").unwrap();
        let rids = (0..6)
            .map(|i| {
                db.record_text(
                    &format!("note {i} about the harbour and the ferry timetable"),
                    "episodic",
                    0.5,
                    0.0,
                    604800.0,
                    &empty_meta(),
                    "default",
                    0.8,
                    "general",
                    "user",
                    None,
                )
                .unwrap()
            })
            .collect();
        (db, rids)
    }

    /// The reviewer's chain: a replicated row is archived, refreshed while
    /// cold, then hydrated. Refresh must store it compressed (cold
    /// convention) and hydrate must bring back a usable vector.
    #[test]
    fn refresh_while_cold_then_hydrate() {
        let (db, rids) = store();
        let rid = &rids[0];
        strip_vectors(&db, std::slice::from_ref(rid));
        assert!(db.archive(rid).unwrap());

        let report = db.refresh_embeddings(None, false).unwrap();
        assert_eq!(report.refreshed, 1, "{report:?}");
        let blob = embedding_blob(&db, rid).expect("refresh filled the vector");
        assert!(
            crate::compression::is_compressed(&blob),
            "a cold row's refreshed vector is stored compressed"
        );
        assert_eq!(tier(&db, rid), "cold");

        assert!(db.hydrate(rid).unwrap());
        assert_eq!(
            embedding_blob(&db, rid).unwrap().len(),
            db.embedding_dim() * 4
        );
    }

    /// Compressed cold vectors are usable; refresh must not flag them, nor
    /// overwrite them with a hot-format blob.
    #[test]
    fn refresh_leaves_compressed_cold_vectors_alone() {
        let (db, rids) = store();
        assert!(db.archive(&rids[1]).unwrap());
        let before = embedding_blob(&db, &rids[1]).unwrap();

        let report = db.refresh_embeddings(None, false).unwrap();
        assert_eq!(report.unusable_found, 0, "{report:?}");
        assert_eq!(embedding_blob(&db, &rids[1]).unwrap(), before);
        assert!(db.hydrate(&rids[1]).unwrap());
    }
}

/// archive() tombstones the index entry even when SQL holds no vector: the
/// index's content is not derivable from the column, and a stale entry
/// would keep an archived row vector-searchable.
#[test]
fn archive_tombstones_an_index_entry_the_column_no_longer_shows() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    record(&db, "a neighbour row", 7.0);
    let rid = record(&db, "indexed, then its column blanked", 6.0);
    db.conn()
        .execute(
            "UPDATE memories SET embedding = NULL WHERE rid = ?1",
            params![rid],
        )
        .unwrap();
    assert!(db.archive(&rid).unwrap());

    let hits = db
        .recall(
            &vec_seed(6.0, 8),
            10,
            None,
            None,
            true,
            false,
            None,
            true,
            None,
            None,
            None,
            None,
            None,
            false,
            None,
            None,
        )
        .unwrap();
    assert!(
        hits.iter().all(|h| h.rid != rid),
        "an archived row must not stay vector-searchable"
    );
}

/// The same paths on an encrypted store, where every stored vector is
/// sealed and a NULL must not reach decryption.
#[test]
fn encrypted_store_tolerates_rows_without_a_vector() {
    let key = [7u8; 32];
    let db = YantrikDB::new_encrypted(":memory:", 8, &key).unwrap();
    for i in 0..12 {
        record(
            &db,
            &format!("Alice met Bob at the harbour, visit {i}"),
            1.0 + (i % 3) as f32 * 0.001,
        );
    }
    let bare: Vec<String> = (0..3)
        .map(|i| record(&db, &format!("replicated note {i}"), 1.0))
        .collect();
    strip_vectors(&db, &bare);

    db.think(&ThinkConfig::default())
        .expect("think() on an encrypted store with rows lacking a vector");
    assert!(db.archive(&bare[0]).unwrap());
    assert!(embedding_blob(&db, &bare[0]).is_none());
    assert!(db.hydrate(&bare[0]).unwrap());
    assert!(embedding_blob(&db, &bare[0]).is_none());

    // A sealed vector still round-trips through the cold tier.
    let with_vec = record(&db, "an encrypted row with its vector", 3.0);
    assert!(db.archive(&with_vec).unwrap());
    assert!(db.hydrate(&with_vec).unwrap());
    assert!(db.get(&with_vec).unwrap().is_some());
}
