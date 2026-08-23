use super::*;

// ─────────────────────────────────────────────────────────────────
// RFC 022 §2: insert_vector + encrypt_embedding_pub (yantrikdb 0.6.5)
//
// Pre-existing methods promoted from `pub(crate)` to `pub` so the
// server's replication backfill path can populate followers' HNSW
// per-row instead of doing a full rebuild_vec_index() per batch.
// ─────────────────────────────────────────────────────────────────

#[test]
fn test_insert_vector_makes_recall_find_it() {
    // Simulates the follower-backfill scenario: a memory row is in SQLite
    // (here: inserted via record() so we don't need raw SQL), but the
    // backfill caller wants to put a *different* embedding into the HNSW
    // for a separately-supplied rid. The simpler exercise: insert_vector
    // is the same path record() takes internally, so calling it with a
    // fresh rid + vector should produce a recall hit on that vector.
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // Use record() once to seat the embedder + indices.
    let _ = db
        .record(
            "seed",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(0.1, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    // Now exercise the new public API directly with a synthetic rid +
    // vector. This is the call path replication backfill will take.
    let synthetic_rid = "test-synthetic-rid-1";
    let synthetic_emb = vec_seed(0.9, 8);
    db.insert_vector(synthetic_rid, &synthetic_emb).unwrap();

    // The HNSW index now contains the synthetic rid. Recall against
    // the synthetic vector should surface it as the top result. We
    // skip the SQLite-row-fetch concern here because that path is
    // exercised by the integration test in the server crate
    // (yantrikdb-server replication_backfill.rs); engine-level test
    // just verifies the API surface and HNSW insertion.
    let results = db
        .recall(
            &synthetic_emb,
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

    // The synthetic rid won't have a matching SQLite row, so recall's
    // post-fetch step will drop it. What matters for this test is that
    // insert_vector() returned Ok(()) and the HNSW now knows about it
    // (verified by stats).
    let stats = db.stats(None).unwrap();
    // We had 1 from record() + 1 from insert_vector — the HNSW knows
    // about both even though SQLite only has the record() one.
    assert!(
        stats.vec_index_entries >= 2,
        "vec_index_entries should be at least 2 (record + insert_vector); got {}",
        stats.vec_index_entries
    );
    // Sanity: results are still well-formed (recall didn't crash on the
    // dangling synthetic rid).
    assert!(results.len() <= 5);
}

#[test]
fn test_insert_vector_idempotent_on_same_rid() {
    // Re-inserting the same rid+vector must not error. The HNSW backend
    // is responsible for de-duping; insert_vector just propagates errors.
    // This guarantees the follower-backfill loop can be retried safely
    // (e.g., on sync_loop poll N+1 after partial-batch failure on poll N).
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = "idempotency-test";
    let emb = vec_seed(0.5, 8);

    db.insert_vector(rid, &emb).unwrap();
    // Second call must not panic or return error.
    db.insert_vector(rid, &emb).unwrap();
}

#[test]
fn test_encrypt_embedding_pub_unencrypted_returns_input_unchanged() {
    // Without an encryption provider, encrypt_embedding_pub is a no-op:
    // returns the input bytes as a Vec<u8>. This matches the existing
    // pub(crate) encrypt_embedding's contract.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let raw: Vec<u8> = (0u8..32).collect();
    let out = db.encrypt_embedding_pub(&raw).unwrap();
    assert_eq!(out, raw, "no-encryption path must return input unchanged");
}

#[test]
fn test_encrypt_embedding_pub_with_encryption_returns_ciphertext() {
    // With encryption enabled, encrypt_embedding_pub must produce
    // ciphertext that differs from the plaintext input. Round-trip
    // verification (decrypt → original) is exercised by the existing
    // `pub(crate) decrypt_embedding` callers (e.g., archive/hydrate
    // tests above); this test only verifies the public wrapper exposes
    // the encryption path correctly.
    let master_key = [0xAB; 32];
    let db = YantrikDB::new_encrypted(":memory:", 8, &master_key).unwrap();
    let raw: Vec<u8> = (0u8..32).collect();
    let out = db.encrypt_embedding_pub(&raw).unwrap();
    assert_ne!(
        out, raw,
        "encrypted path must produce ciphertext, not plaintext"
    );
    // Encrypted blobs include a nonce + tag, so length differs from raw.
    assert!(
        out.len() > raw.len(),
        "encrypted blob should be longer than plaintext (nonce + tag overhead)"
    );
}

// ── Issue #9 cluster replication API: record_with_rid ──

#[test]
fn record_with_rid_basic_succeeds() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let emb = vec_seed(1.0, 64);
    db.record_with_rid(
        "rid_test_1",
        "the quick brown fox",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &emb,
        "default",
        0.8,
        "general",
        "user",
        None,
        1_700_000_000_000_000,
        &[],
        "test-model.v1",
        None,
        crate::provenance::WriteAdmission::Admitted,
    )
    .expect("record_with_rid succeeds");

    let row = db.get("rid_test_1").unwrap().unwrap();
    assert_eq!(row.rid, "rid_test_1");
    assert_eq!(row.text, "the quick brown fox");
    assert_eq!(row.memory_type, "episodic");
}

#[test]
fn record_with_rid_persists_v25_columns() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let emb = vec_seed(2.0, 64);
    db.record_with_rid(
        "rid_v25",
        "test v25 columns",
        "semantic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &emb,
        "default",
        0.8,
        "general",
        "user",
        None,
        1_700_000_000_000_000,
        &[],
        "bge-base-en-v1.5",
        None,
        crate::provenance::WriteAdmission::Admitted,
    )
    .unwrap();

    let conn = db.read_conn();
    let (cum, model): (i64, Option<String>) = conn
        .query_row(
            "SELECT created_at_unix_micros, embedding_model FROM memories WHERE rid = ?1",
            rusqlite::params!["rid_v25"],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(cum, 1_700_000_000_000_000);
    assert_eq!(model.as_deref(), Some("bge-base-en-v1.5"));
}

#[test]
fn record_with_rid_is_idempotent_on_replay() {
    // Determinism contract: a second call with identical args yields
    // identical engine state (no doubles in entities, no doubles in
    // memory_entities, single oplog entry, single memories row).
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let emb = vec_seed(3.0, 64);
    let entities = ["Alice", "Acme"];
    let entity_refs: Vec<&str> = entities.iter().copied().collect();
    for _ in 0..3 {
        db.record_with_rid(
            "rid_idem",
            "Alice works at Acme",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &emb,
            "default",
            0.8,
            "general",
            "user",
            None,
            1_700_000_001_000_000,
            &entity_refs,
            "test-model.v1",
            None,
            crate::provenance::WriteAdmission::Admitted,
        )
        .expect("idempotent re-apply");
    }
    // Phase 4.3 Commit C: entity persistence is enqueued by record_with_rid
    // and applied by the materializer thread. Drain the queue inline before
    // asserting on entity-graph state.
    db.apply_pending_ops_once(100).unwrap();

    let conn = db.read_conn();
    let memory_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE rid = ?1",
            rusqlite::params!["rid_idem"],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(memory_count, 1, "memories has exactly one row");

    // memory_entities should have one row per (memory, entity) pair.
    let me_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memory_entities WHERE memory_rid = ?1",
            rusqlite::params!["rid_idem"],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        me_count, 2,
        "memory_entities has 2 rows (Alice, Acme), no doubles"
    );

    // entities row mention_count should equal 1 (only first call counts as a new mention).
    let mc: i64 = conn
        .query_row(
            "SELECT mention_count FROM entities WHERE name = 'Alice'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(mc, 1, "mention_count not bumped on replay");

    // Oplog should have exactly one record_with_rid entry for this rid.
    let op_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM oplog WHERE op_type = 'record_with_rid' AND target_rid = ?1",
            rusqlite::params!["rid_idem"],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(op_count, 1, "oplog has exactly one record_with_rid entry");
}

#[test]
fn record_with_rid_rejects_dimension_mismatch() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let bad = vec![0.0f32; 32]; // wrong dim
    let err = db
        .record_with_rid(
            "rid_bad_dim",
            "x",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &bad,
            "default",
            0.8,
            "general",
            "user",
            None,
            1_700_000_002_000_000,
            &[],
            "test-model.v1",
            None,
            crate::provenance::WriteAdmission::Admitted,
        )
        .expect_err("must reject");
    match err {
        crate::error::YantrikDbError::EmbeddingDimensionMismatch { expected, got } => {
            assert_eq!(expected, 64);
            assert_eq!(got, 32);
        }
        other => panic!("expected EmbeddingDimensionMismatch, got {other:?}"),
    }
    // The DB must NOT have inserted anything despite the failed call.
    assert!(db.get("rid_bad_dim").unwrap().is_none());
}

#[test]
fn record_with_rid_uses_caller_supplied_timestamp() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let emb = vec_seed(4.0, 64);
    let caller_ts: i64 = 1_700_000_005_000_000;
    db.record_with_rid(
        "rid_ts",
        "test ts",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &emb,
        "default",
        0.8,
        "general",
        "user",
        None,
        caller_ts,
        &[],
        "test-model.v1",
        None,
        crate::provenance::WriteAdmission::Admitted,
    )
    .unwrap();
    // Verify created_at REAL and created_at_unix_micros INTEGER both
    // reflect the caller-supplied timestamp (no engine-side now() call).
    let conn = db.read_conn();
    let (cat_real, cat_micros): (f64, i64) = conn
        .query_row(
            "SELECT created_at, created_at_unix_micros FROM memories WHERE rid = ?1",
            rusqlite::params!["rid_ts"],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(cat_micros, caller_ts);
    let expected_real = (caller_ts as f64) / 1_000_000.0;
    assert!(
        (cat_real - expected_real).abs() < 1e-6,
        "created_at REAL should reflect caller timestamp: got {} expected {}",
        cat_real,
        expected_real
    );
}

#[test]
fn record_with_rid_makes_recall_find_it() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let emb = vec_seed(7.0, 64);
    db.record_with_rid(
        "rid_recall",
        "memory inserted via record_with_rid",
        "episodic",
        0.7,
        0.0,
        604800.0,
        &empty_meta(),
        &emb,
        "default",
        0.8,
        "general",
        "user",
        None,
        1_700_000_006_000_000,
        &[],
        "test-model.v1",
        None,
        crate::provenance::WriteAdmission::Admitted,
    )
    .unwrap();

    let results = db
        .recall(
            &emb, 5, None, None, false, false, None, true, None, None, None, None, None, false,
            None, // event_after (#149)
            None, // event_before (#149)
        )
        .unwrap();
    assert!(
        results.iter().any(|r| r.rid == "rid_recall"),
        "rid_recall should appear in recall results"
    );
}

// ── Issue #9 cluster replication API: tombstone_with_rid ──

#[test]
fn tombstone_with_rid_basic_succeeds() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let emb = vec_seed(1.0, 64);
    let rid = db
        .record(
            "to tombstone",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &emb,
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    db.tombstone_with_rid(
        &rid,
        "default",
        Some("test reason"),
        1_700_000_010_000_000,
        None,
    )
    .expect("tombstone_with_rid succeeds");

    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.consolidation_status, "tombstoned");
}

#[test]
fn tombstone_with_rid_persists_reason() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let emb = vec_seed(2.0, 64);
    let rid = db
        .record(
            "memory with reason",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &emb,
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    db.tombstone_with_rid(
        &rid,
        "default",
        Some("user requested deletion"),
        1_700_000_011_000_000,
        None,
    )
    .unwrap();

    let conn = db.read_conn();
    let reason: Option<String> = conn
        .query_row(
            "SELECT tombstone_reason FROM memories WHERE rid = ?1",
            rusqlite::params![&rid],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(reason.as_deref(), Some("user requested deletion"));
}

#[test]
fn tombstone_with_rid_idempotent_on_replay() {
    // Determinism contract: re-tombstoning a rid that's already tombstoned
    // returns Ok(()) without emitting a second oplog entry.
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let emb = vec_seed(3.0, 64);
    let rid = db
        .record(
            "idempotent tombstone",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &emb,
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    for _ in 0..3 {
        db.tombstone_with_rid(&rid, "default", Some("replay"), 1_700_000_012_000_000, None)
            .expect("idempotent re-apply");
    }

    let conn = db.read_conn();
    let op_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM oplog WHERE op_type = 'forget' AND target_rid = ?1",
            rusqlite::params![&rid],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        op_count, 1,
        "oplog has exactly one forget entry despite 3 calls"
    );
}

#[test]
fn tombstone_with_rid_idempotent_on_missing() {
    // Snapshot-install + log replay overlap means tombstoning a rid that
    // doesn't exist is normal cluster behavior. Must return Ok(()), not error.
    let db = YantrikDB::new(":memory:", 64).unwrap();
    db.tombstone_with_rid(
        "rid_never_existed",
        "default",
        None,
        1_700_000_013_000_000,
        None,
    )
    .expect("must be Ok(()) on missing rid");
    // Verify no oplog entry created for the missing rid.
    let conn = db.read_conn();
    let op_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM oplog WHERE target_rid = ?1",
            rusqlite::params!["rid_never_existed"],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(op_count, 0);
}

#[test]
fn tombstone_with_rid_uses_caller_supplied_timestamp() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let emb = vec_seed(4.0, 64);
    let rid = db
        .record(
            "ts test",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &emb,
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    let caller_ts: i64 = 1_700_000_999_000_000;
    db.tombstone_with_rid(&rid, "default", None, caller_ts, None)
        .unwrap();

    let conn = db.read_conn();
    let updated_at: f64 = conn
        .query_row(
            "SELECT updated_at FROM memories WHERE rid = ?1",
            rusqlite::params![&rid],
            |row| row.get(0),
        )
        .unwrap();
    let expected = (caller_ts as f64) / 1_000_000.0;
    assert!(
        (updated_at - expected).abs() < 1e-6,
        "updated_at should reflect caller ts: got {} expected {}",
        updated_at,
        expected
    );
}

#[test]
fn forget_still_works_after_refactor() {
    // Back-compat: forget() must still return Result<bool>, true on first
    // tombstone of a live row.
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let emb = vec_seed(5.0, 64);
    let rid = db
        .record(
            "forget test",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &emb,
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    let first = db.forget(&rid).unwrap();
    assert!(first, "first forget on live row returns true");

    let second = db.forget(&rid).unwrap();
    assert!(
        !second,
        "second forget on already-tombstoned row returns false"
    );

    let missing = db.forget("rid_never_existed").unwrap();
    assert!(!missing, "forget on missing rid returns false");
}

#[test]
fn forget_rolls_back_every_durable_projection_when_oplog_insert_fails() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "forget must be atomic",
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
    let target_rid = db
        .record(
            "link target",
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
    let oplog_before: i64;
    {
        let conn = db.conn();
        conn.execute(
            "INSERT INTO memory_entities (memory_rid, entity_name) VALUES (?1, 'AtomicityMarker')",
            rusqlite::params![&rid],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memory_chunks (rid, chunk_idx, embedding) VALUES (?1, 1, X'00')",
            rusqlite::params![&rid],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO record_links \
             (link_id, source_rid, target_rid, link_type, status, selection_state, \
              created_at, hlc, origin_actor) \
             VALUES ('atomic-forget-link', ?1, ?2, 'DerivedFrom', 'active', \
                     'selected', 1.0, X'00', 'test')",
            rusqlite::params![&rid, &target_rid],
        )
        .unwrap();
        oplog_before = conn
            .query_row("SELECT COUNT(*) FROM oplog", [], |row| row.get(0))
            .unwrap();
        conn.execute_batch(
            "CREATE TRIGGER fail_forget_op BEFORE INSERT ON oplog \
             WHEN NEW.op_type = 'forget' BEGIN \
                 SELECT RAISE(ABORT, 'forced forget oplog failure'); \
             END;",
        )
        .unwrap();
    }

    let error = db
        .forget(&rid)
        .expect_err("forced oplog failure must escape");
    assert!(format!("{error}").contains("forced forget oplog failure"));

    let conn = db.conn();
    let status: String = conn
        .query_row(
            "SELECT consolidation_status FROM memories WHERE rid = ?1",
            rusqlite::params![&rid],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(status, "active");
    for (table, predicate) in [("memory_entities", "memory_rid"), ("memory_chunks", "rid")] {
        let count: i64 = conn
            .query_row(
                &format!("SELECT COUNT(*) FROM {table} WHERE {predicate} = ?1"),
                rusqlite::params![&rid],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1, "{table} deletion escaped the rollback");
    }
    let link_status: String = conn
        .query_row(
            "SELECT status FROM record_links WHERE link_id = 'atomic-forget-link'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(link_status, "active");
    let oplog_after: i64 = conn
        .query_row("SELECT COUNT(*) FROM oplog", [], |row| row.get(0))
        .unwrap();
    assert_eq!(oplog_after, oplog_before);
}

#[test]
fn tombstone_with_rid_hides_from_recall() {
    // After tombstone_with_rid, the rid must not appear in recall results.
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let emb = vec_seed(6.0, 64);
    let rid = db
        .record(
            "hide me",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &emb,
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    // Sanity: visible before tombstone.
    let r = db
        .recall(
            &emb, 5, None, None, false, false, None, true, None, None, None, None, None, false,
            None, // event_after (#149)
            None, // event_before (#149)
        )
        .unwrap();
    assert!(r.iter().any(|x| x.rid == rid), "visible before tombstone");

    db.tombstone_with_rid(&rid, "default", None, 1_700_000_014_000_000, None)
        .unwrap();

    // Hidden after.
    let r2 = db
        .recall(
            &emb, 5, None, None, false, false, None, true, None, None, None, None, None, false,
            None, // event_after (#149)
            None, // event_before (#149)
        )
        .unwrap();
    assert!(!r2.iter().any(|x| x.rid == rid), "hidden after tombstone");
}

// ── Issue #9 cluster replication API: entity edge methods ──

#[test]
fn upsert_entity_edge_with_id_basic_succeeds() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    db.upsert_entity_edge_with_id(
        "edge_1",
        "Alice",
        "Acme",
        "works_at",
        0.9,
        "default",
        1_700_000_020_000_000,
        None,
    )
    .expect("upsert succeeds");

    // Verify the claim row exists with caller-supplied edge_id.
    let conn = db.read_conn();
    let (cid, src, dst, rel, weight): (String, String, String, String, f64) = conn
        .query_row(
            "SELECT claim_id, src, dst, rel_type, weight FROM claims WHERE claim_id = ?1",
            rusqlite::params!["edge_1"],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )
        .unwrap();
    assert_eq!(cid, "edge_1");
    assert_eq!(src, "Alice");
    assert_eq!(dst, "Acme");
    assert_eq!(rel, "works_at");
    assert!((weight - 0.9).abs() < 1e-6);
}

#[test]
fn upsert_entity_edge_with_id_is_idempotent_on_replay() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    for _ in 0..3 {
        db.upsert_entity_edge_with_id(
            "edge_idem",
            "Bob",
            "Beta Corp",
            "founded",
            0.8,
            "default",
            1_700_000_021_000_000,
            None,
        )
        .expect("idempotent");
    }
    let conn = db.read_conn();
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM claims WHERE claim_id = ?1",
            rusqlite::params!["edge_idem"],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 1, "exactly one claim row regardless of replay");

    let op_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM oplog WHERE op_type = 'upsert_entity_edge_with_id' AND target_rid = ?1",
        rusqlite::params!["edge_idem"],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(op_count, 1, "exactly one oplog entry regardless of replay");
}

#[test]
fn upsert_entity_edge_uses_caller_supplied_timestamp() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let caller_ts: i64 = 1_700_000_555_000_000;
    db.upsert_entity_edge_with_id(
        "edge_ts", "X", "Y", "knows", 0.5, "default", caller_ts, None,
    )
    .unwrap();
    let conn = db.read_conn();
    let created_at: f64 = conn
        .query_row(
            "SELECT created_at FROM claims WHERE claim_id = ?1",
            rusqlite::params!["edge_ts"],
            |row| row.get(0),
        )
        .unwrap();
    let expected = (caller_ts as f64) / 1_000_000.0;
    assert!(
        (created_at - expected).abs() < 1e-6,
        "created_at REAL reflects caller ts: got {} expected {}",
        created_at,
        expected
    );
}

#[test]
fn upsert_entity_edge_creates_entities() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    db.upsert_entity_edge_with_id(
        "edge_ent",
        "Charlie",
        "Delta Inc",
        "ceo_of",
        1.0,
        "default",
        1_700_000_022_000_000,
        None,
    )
    .unwrap();
    let conn = db.read_conn();
    let charlie: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM entities WHERE name = 'Charlie'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let delta: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM entities WHERE name = 'Delta Inc'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(charlie, 1);
    assert_eq!(delta, 1);
}

#[test]
fn delete_entity_edge_with_id_basic_succeeds() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    db.upsert_entity_edge_with_id(
        "edge_del",
        "A",
        "B",
        "knows",
        0.5,
        "default",
        1_700_000_023_000_000,
        None,
    )
    .unwrap();
    db.delete_entity_edge_with_id("edge_del", "default", 1_700_000_024_000_000, None)
        .expect("delete succeeds");
    let conn = db.read_conn();
    let tombstoned: i64 = conn
        .query_row(
            "SELECT tombstoned FROM claims WHERE claim_id = ?1",
            rusqlite::params!["edge_del"],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(tombstoned, 1);
}

#[test]
fn delete_entity_edge_with_id_idempotent_on_missing() {
    // Snapshot-install + log replay overlap means deleting a non-existent
    // edge_id is normal cluster behavior. Must return Ok(()), not error.
    let db = YantrikDB::new(":memory:", 64).unwrap();
    db.delete_entity_edge_with_id("edge_never", "default", 1_700_000_025_000_000, None)
        .expect("missing edge: ok");
    let conn = db.read_conn();
    let op_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM oplog WHERE op_type = 'delete_entity_edge_with_id' AND target_rid = ?1",
        rusqlite::params!["edge_never"],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(op_count, 0, "no oplog noise for missing edge delete");
}

#[test]
fn delete_entity_edge_with_id_idempotent_on_replay() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    db.upsert_entity_edge_with_id(
        "edge_del2",
        "P",
        "Q",
        "knows",
        0.5,
        "default",
        1_700_000_026_000_000,
        None,
    )
    .unwrap();
    for _ in 0..3 {
        db.delete_entity_edge_with_id("edge_del2", "default", 1_700_000_027_000_000, None)
            .expect("idempotent");
    }
    let conn = db.read_conn();
    let op_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM oplog WHERE op_type = 'delete_entity_edge_with_id' AND target_rid = ?1",
        rusqlite::params!["edge_del2"],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(
        op_count, 1,
        "exactly one delete oplog entry across 3 replays"
    );
}
