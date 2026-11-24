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
/// a follower's replicated rows: no vector in SQL, none in the index. The
/// delta is compacted first: a rebuild replaces the cold tier but leaves
/// the delta, where the rows' original entries would otherwise survive.
fn strip_vectors(db: &YantrikDB, rids: &[String]) {
    db.search_state.load().vec_index.compact().unwrap();
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

    /// archive() tombstones the row's index entry; hydrate() of a row with
    /// no vector appends nothing; a later refresh fills the vector. The
    /// rebuild that follows does not touch the delta, so without an append
    /// at a fresh seq the old tombstone kept the row out of vector recall.
    #[test]
    fn archive_hydrate_then_refresh_is_vector_searchable() {
        let (db, rids) = store();
        let rid = &rids[2];
        strip_vectors(&db, std::slice::from_ref(rid));
        assert!(db.archive(rid).unwrap());
        assert!(db.hydrate(rid).unwrap());
        assert!(embedding_blob(&db, rid).is_none());

        let report = db.refresh_embeddings(None, false).unwrap();
        assert_eq!(report.refreshed, 1, "{report:?}");

        let text = db.get(rid).unwrap().unwrap().text;
        let query = db.embed(&text).unwrap();
        let hits = db
            .recall(
                &query, 10, None, None, false, false, None, true, None, None, None, None, None,
                false, None, None,
            )
            .unwrap();
        assert!(
            hits.iter().any(|h| &h.rid == rid),
            "the refreshed row must be back in vector recall"
        );
    }

    /// A store whose first row was archived reopens: the dimension probe
    /// must not read a compressed blob's byte length as a dimension.
    #[test]
    fn store_reopens_after_its_first_row_is_archived() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reopen.db");
        let path = path.to_str().unwrap();
        let first = {
            let db = YantrikDB::with_default(path).unwrap();
            let rids: Vec<String> = (0..8)
                .map(|i| {
                    db.record_text(
                        &format!("reopen probe {i} about the harbour"),
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
            let first: String = db
                .conn()
                .query_row(
                    "SELECT rid FROM memories WHERE embedding IS NOT NULL LIMIT 1",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert!(rids.contains(&first));
            assert!(db.archive(&first).unwrap());
            db.close().unwrap();
            first
        };
        let db = YantrikDB::with_default(path).expect("the store must reopen");
        assert!(db.hydrate(&first).unwrap());
    }

    fn recalls(db: &YantrikDB, rid: &str) -> bool {
        let text = db.get(rid).unwrap().unwrap().text;
        let query = db.embed(&text).unwrap();
        db.recall(
            &query, 10, None, None, false, false, None, true, None, None, None, None, None, false,
            None, None,
        )
        .unwrap()
        .iter()
        .any(|h| h.rid == rid)
    }

    fn pending_for(db: &YantrikDB, rid: &str) -> crate::engine::repair::PendingRefresh {
        let (ciphertext, tier): (String, String) = db
            .conn()
            .query_row(
                "SELECT text, storage_tier FROM memories WHERE rid = ?1",
                params![rid],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        let vector = db.embed(&db.get(rid).unwrap().unwrap().text).unwrap();
        let blob = if tier == "cold" {
            crate::compression::compress_embedding(&vector)
        } else {
            crate::serde_helpers::serialize_f32(&vector)
        };
        crate::engine::repair::PendingRefresh {
            rid: rid.to_string(),
            scanned_ciphertext: ciphertext,
            scanned_tier: tier,
            new_embedding_blob: blob,
        }
    }
    /// A follower with more rows than the delta tier holds (256 by
    /// default), workers running: the repair fills every vector and the
    /// rows come back in vector recall. (An earlier take on #305 re-indexed
    /// by appending each vector to the delta and failed here with
    /// Backpressure after its SQL had committed.)
    #[test]
    fn refresh_at_follower_scale() {
        let db = std::sync::Arc::new(YantrikDB::with_default(":memory:").unwrap());
        let _workers = crate::engine::materializer::spawn_all_workers(&db, 2);
        // The setup's own writes outrun the compactor too; a caller is
        // expected to retry Backpressure, so the setup does.
        let rids: Vec<String> = (0..700)
            .map(|i| loop {
                match db.record_text(
                    &format!("replicated note {i} about topic {} and the ferry", i % 17),
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
                ) {
                    Ok(rid) => break rid,
                    Err(crate::error::YantrikDbError::Backpressure { retry_after_ms, .. }) => {
                        std::thread::sleep(std::time::Duration::from_millis(retry_after_ms));
                    }
                    Err(e) => panic!("{e}"),
                }
            })
            .collect();
        strip_vectors(&db, &rids);

        let report = db.refresh_embeddings(None, false).unwrap();
        assert_eq!(report.refreshed, rids.len(), "{report:?}");
        for rid in rids.iter().step_by(97) {
            assert!(recalls(&db, rid), "{rid} must be back in vector recall");
        }
    }

    /// The tier guard, cold to hot: a compressed vector encoded for a cold
    /// row is not written once a hydrate has made the row hot.
    #[test]
    fn refresh_skips_a_row_hydrated_after_the_scan() {
        let (db, rids) = store();
        let rid = &rids[5];
        assert!(db.archive(rid).unwrap());
        let pending = vec![pending_for(&db, rid)];
        assert_eq!(pending[0].scanned_tier, "cold");
        assert!(db.hydrate(rid).unwrap());
        let before = embedding_blob(&db, rid).unwrap();

        let mut report = crate::engine::repair::RefreshReport::default();
        db.apply_refreshes(&pending, &mut report).unwrap();
        assert_eq!(report.skipped_concurrent_modification, 1, "{report:?}");
        assert_eq!(
            embedding_blob(&db, rid).unwrap(),
            before,
            "a compressed vector must not land in a hot row"
        );
    }

    /// The tier guard: a vector re-encoded for the tier it was scanned in
    /// is not written if an archive moved the row meanwhile.
    #[test]
    fn refresh_skips_a_row_whose_tier_changed_after_the_scan() {
        let (db, rids) = store();
        let rid = &rids[3];
        let text: String = db
            .conn()
            .query_row(
                "SELECT text FROM memories WHERE rid = ?1",
                params![rid],
                |r| r.get(0),
            )
            .unwrap();
        let vector = db.embed(&db.get(rid).unwrap().unwrap().text).unwrap();
        // Scanned hot, encoded raw...
        let pending = vec![crate::engine::repair::PendingRefresh {
            rid: rid.clone(),
            scanned_ciphertext: text,
            scanned_tier: "hot".to_string(),
            new_embedding_blob: crate::serde_helpers::serialize_f32(&vector),
        }];
        // ...then archived before the write.
        assert!(db.archive(rid).unwrap());
        let before = embedding_blob(&db, rid).unwrap();

        let mut report = crate::engine::repair::RefreshReport::default();
        db.apply_refreshes(&pending, &mut report).unwrap();
        assert_eq!(report.skipped_concurrent_modification, 1, "{report:?}");
        assert_eq!(report.refreshed, 0);
        assert_eq!(
            embedding_blob(&db, rid).unwrap(),
            before,
            "a raw vector must not land in a cold row"
        );
    }

    /// A cold row whose blob does not decode is not "usable" just because
    /// it is non-empty: the repair re-encodes it, and it hydrates after.
    #[test]
    fn refresh_repairs_a_corrupt_cold_vector() {
        let (db, rids) = store();
        let rid = &rids[0];
        assert!(db.archive(rid).unwrap());
        db.conn()
            .execute(
                "UPDATE memories SET embedding = ?1 WHERE rid = ?2",
                params![vec![0x28u8, 0xB5, 0x2F, 0xFD, 1, 2, 3, 4], rid],
            )
            .unwrap();
        assert!(db.hydrate(rid).is_err(), "precondition: it cannot hydrate");

        let report = db.refresh_embeddings(None, false).unwrap();
        assert_eq!(report.unusable_found, 1, "{report:?}");
        assert_eq!(report.refreshed, 1, "{report:?}");
        assert!(db.hydrate(rid).unwrap());
        assert_eq!(
            embedding_blob(&db, rid).unwrap().len(),
            db.embedding_dim() * 4
        );
    }
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

fn record_vector(db: &YantrikDB, text: &str, vector: &[f32]) -> String {
    db.record(
        text,
        "semantic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        vector,
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap()
}

/// The redundancy scan reads every tier. Compressed cold blobs read as raw
/// floats are garbage vectors that look similar to each other.
#[test]
fn redundancy_scan_decodes_cold_vectors() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids: Vec<String> = (0..8)
        .map(|i| {
            let mut v = vec![0.0f32; 8];
            v[i] = 1.0;
            record_vector(&db, &format!("orthogonal row {i}"), &v)
        })
        .collect();
    let hot = crate::triggers::check_redundancy(&db, 0.9).unwrap().len();
    assert_eq!(hot, 0, "orthogonal vectors are not redundant");
    for rid in &rids {
        assert!(db.archive(rid).unwrap());
    }
    let cold = crate::triggers::check_redundancy(&db, 0.9).unwrap().len();
    assert_eq!(cold, hot, "archiving must not change what is redundant");
}

/// A cold blob that carries the zstd magic but does not decode: hydrate()
/// refuses before touching the row, which stays cold.
#[test]
fn hydrate_refuses_a_corrupt_cold_vector_without_moving_the_row() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = record(&db, "a cold row with a corrupt vector", 9.0);
    assert!(db.archive(&rid).unwrap());
    db.conn()
        .execute(
            "UPDATE memories SET embedding = ?1 WHERE rid = ?2",
            params![vec![0x28u8, 0xB5, 0x2F, 0xFD, 1, 2, 3, 4], rid],
        )
        .unwrap();

    assert!(
        db.hydrate(&rid).is_err(),
        "a corrupt vector must not hydrate"
    );
    assert_eq!(tier(&db, &rid), "cold", "the row must stay where it was");
}

fn vector_hits(db: &YantrikDB, query: &[f32], rid: &str) -> bool {
    db.recall(
        query, 10, None, None, true, false, None, true, None, None, None, None, None, false, None,
        None,
    )
    .unwrap()
    .iter()
    .any(|h| h.rid == rid)
}

/// The invariant archive() relies on: a row in the index has its vector in
/// SQL. record_with_rid on an existing row with no vector (a replicated row
/// re-delivered through the apply path) used to index the vector and leave
/// the column NULL, so archive() skipped the tombstone and the cold row
/// stayed vector-searchable.
#[test]
fn record_with_rid_fills_a_missing_vector() {
    let (db, bare) = store_with_bare_rows(1);
    let rid = &bare[0];
    let v = vec_seed(11.0, 8);
    db.record_with_rid(
        rid,
        "replicated note 0 about the harbour",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &v,
        "default",
        0.8,
        "general",
        "user",
        None,
        1_700_000_000_000_000,
        &[],
        "test-embedder",
        None,
        crate::provenance::WriteAdmission::Admitted,
    )
    .unwrap();
    assert!(
        embedding_blob(&db, rid).is_some(),
        "the indexed vector must land in SQL too"
    );
    assert!(vector_hits(&db, &v, rid));
    assert!(db.archive(rid).unwrap());
    assert!(
        !vector_hits(&db, &v, rid),
        "an archived row must not stay vector-searchable"
    );
}

/// Same invariant through the public backfill helper.
#[test]
fn insert_vector_fills_a_missing_vector() {
    let (db, bare) = store_with_bare_rows(1);
    let rid = &bare[0];
    let v = vec_seed(12.0, 8);
    db.insert_vector(rid, &v).unwrap();
    assert!(
        embedding_blob(&db, rid).is_some(),
        "the indexed vector must land in SQL too"
    );
    assert!(db.archive(rid).unwrap());
    assert!(
        !vector_hits(&db, &v, rid),
        "an archived row must not stay vector-searchable"
    );
}

/// The fills cover cold rows too, in the cold-tier (compressed) format:
/// a cold row with no vector that one of these APIs indexes must not stay
/// NULL in SQL, or a later archive skips its tombstone.
#[test]
fn record_with_rid_and_insert_vector_fill_a_cold_row_compressed() {
    let (db, bare) = store_with_bare_rows(2);
    for rid in &bare {
        assert!(db.archive(rid).unwrap());
        assert!(embedding_blob(&db, rid).is_none());
    }

    let v0 = vec_seed(13.0, 8);
    db.record_with_rid(
        &bare[0],
        "replicated note 0 about the harbour",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &v0,
        "default",
        0.8,
        "general",
        "user",
        None,
        1_700_000_000_000_000,
        &[],
        "test-embedder",
        None,
        crate::provenance::WriteAdmission::Admitted,
    )
    .unwrap();
    let v1 = vec_seed(14.0, 8);
    db.insert_vector(&bare[1], &v1).unwrap();

    for (rid, v) in [(&bare[0], &v0), (&bare[1], &v1)] {
        let blob = embedding_blob(&db, rid).expect("the vector must land in SQL");
        assert!(
            crate::compression::is_compressed(&blob),
            "a cold row's vector is stored compressed"
        );
        assert!(db.hydrate(rid).unwrap());
        assert!(db.archive(rid).unwrap());
        assert!(
            !vector_hits(&db, v, rid),
            "an archived row must not stay vector-searchable"
        );
    }
}

/// insert_vector reads the generation, fills SQL and appends to the index
/// under the sync-writer guard: during a reembed cutover it is refused
/// rather than stamping an old-space vector as current.
#[test]
fn insert_vector_is_refused_during_a_cutover() {
    let (db, bare) = store_with_bare_rows(1);
    db.write_router.switch_to_queueing();
    let refused = db.insert_vector(&bare[0], &vec_seed(15.0, 8));
    db.write_router.switch_to_normal();

    assert!(
        matches!(
            refused,
            Err(crate::error::YantrikDbError::VectorInsertDeferredDuringReembed { .. })
        ),
        "{refused:?}"
    );
    assert!(
        embedding_blob(&db, &bare[0]).is_none(),
        "nothing may be written while refused"
    );
    db.insert_vector(&bare[0], &vec_seed(15.0, 8)).unwrap();
    assert!(embedding_blob(&db, &bare[0]).is_some());
}
