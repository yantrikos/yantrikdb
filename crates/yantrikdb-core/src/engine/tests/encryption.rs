use super::*;

// ── V5: Encryption at rest tests ──

fn test_key() -> [u8; 32] {
    let mut key = [0u8; 32];
    for (i, b) in key.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(7).wrapping_add(42);
    }
    key
}

#[test]
fn test_encrypted_record_and_get() {
    let key = test_key();
    let db = YantrikDB::new_encrypted(":memory:", 8, &key).unwrap();
    assert!(db.is_encrypted());

    let meta = serde_json::json!({"source": "test", "topic": "encryption"});
    let emb = vec_seed(1.0, 8);
    let rid = db
        .record(
            "secret memory",
            "episodic",
            0.8,
            0.3,
            604800.0,
            &meta,
            &emb,
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.text, "secret memory");
    assert_eq!(mem.memory_type, "episodic");
    assert_eq!(mem.importance, 0.8);
    assert_eq!(mem.metadata["source"], "test");
    assert_eq!(mem.metadata["topic"], "encryption");
}

#[test]
fn test_encrypted_data_not_plaintext_in_db() {
    let key = test_key();
    let db = YantrikDB::new_encrypted(":memory:", 8, &key).unwrap();

    let rid = db
        .record(
            "secret memory",
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

    // Read raw stored text — should NOT be plaintext
    let stored_text: String = db
        .conn()
        .query_row(
            "SELECT text FROM memories WHERE rid = ?1",
            params![rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_ne!(
        stored_text, "secret memory",
        "text should be encrypted in DB"
    );

    // Read raw stored metadata — should NOT be plaintext
    let stored_meta: String = db
        .conn()
        .query_row(
            "SELECT metadata FROM memories WHERE rid = ?1",
            params![rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_ne!(stored_meta, "{}", "metadata should be encrypted in DB");
}

#[test]
fn test_encrypted_recall_roundtrip() {
    let key = test_key();
    let db = YantrikDB::new_encrypted(":memory:", 8, &key).unwrap();

    db.record(
        "cat sat on mat",
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
    db.record(
        "dog ran in park",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(5.0, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();
    db.record(
        "cats love warmth",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(1.1, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();

    let results = db
        .recall(
            &vec_seed(1.0, 8),
            2,
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
        )
        .unwrap();
    assert_eq!(results.len(), 2);
    // Text should be decrypted in results
    assert!(results.iter().any(|r| r.text.contains("cat")));
}

#[test]
fn test_encrypted_record_batch() {
    let key = test_key();
    let db = YantrikDB::new_encrypted(":memory:", 8, &key).unwrap();

    let inputs: Vec<RecordInput> = (0..5)
        .map(|i| RecordInput {
            created_at: None,
            idempotency_key: None,
            text: format!("encrypted batch {i}"),
            memory_type: "episodic".to_string(),
            importance: 0.5,
            valence: 0.0,
            half_life: 604800.0,
            metadata: serde_json::json!({"idx": i}),
            embedding: vec_seed(i as f32, 8),
            namespace: "default".to_string(),
            certainty: 0.8,
            domain: "general".to_string(),
            source: "user".to_string(),
            emotional_state: None,
        })
        .collect();

    let rids = db.record_batch(&inputs).unwrap();
    assert_eq!(rids.len(), 5);

    for (i, rid) in rids.iter().enumerate() {
        let mem = db.get(rid).unwrap().unwrap();
        assert_eq!(mem.text, format!("encrypted batch {i}"));
        assert_eq!(mem.metadata["idx"], i);
    }
}

#[test]
fn test_encrypted_archive_hydrate() {
    let key = test_key();
    let db = YantrikDB::new_encrypted(":memory:", 8, &key).unwrap();

    let emb = vec_seed(2.0, 8);
    let rid = db
        .record(
            "to archive encrypted",
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

    // Archive (encrypt compressed)
    assert!(db.archive(&rid).unwrap());
    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.storage_tier, "cold");
    assert_eq!(mem.text, "to archive encrypted"); // text still decryptable

    // Hydrate (decrypt compressed, re-encrypt raw)
    assert!(db.hydrate(&rid).unwrap());
    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.storage_tier, "hot");

    // Should be findable in recall after hydration
    let results = db
        .recall(
            &emb, 10, None, None, false, false, None, true, None, None, None, None, None, false,
        )
        .unwrap();
    assert!(results.iter().any(|r| r.rid == rid));
}

#[test]
fn test_encrypted_correct_memory() {
    let key = test_key();
    let db = YantrikDB::new_encrypted(":memory:", 8, &key).unwrap();

    let rid = db
        .record(
            "color is green",
            "semantic",
            0.7,
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
    // v0.9.3: importance-only correction (text corrections refused).
    let result = db
        .correct(
            &rid,
            None,
            None,
            Some(0.9),
            None,
            "fixed", // reason (required, #47)
        )
        .unwrap();

    // Issue #47 (v0.7.20): in-place mutation; rid preserved.
    assert_eq!(result.corrected_rid, rid);
    assert!(!result.original_tombstoned);
    let updated = db.get(&rid).unwrap().unwrap();
    assert!((updated.importance - 0.9).abs() < 1e-9);
}

#[test]
fn test_unencrypted_db_unaffected() {
    // Verify existing unencrypted path still works identically
    let db = YantrikDB::new(":memory:", 8).unwrap();
    assert!(!db.is_encrypted());

    let rid = db
        .record(
            "plaintext memory",
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
    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.text, "plaintext memory");

    // Text should be stored as plaintext
    let stored_text: String = db
        .conn()
        .query_row(
            "SELECT text FROM memories WHERE rid = ?1",
            params![rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(stored_text, "plaintext memory");
}

#[test]
fn test_encrypted_db_wrong_key_fails() {
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    // Create with key A
    let key_a = test_key();
    {
        let db = YantrikDB::new_encrypted(path, 8, &key_a).unwrap();
        db.record(
            "secret",
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
        db.close().unwrap();
    }

    // Re-open with key B should fail (wrong DEK unwrap)
    let mut key_b = [0u8; 32];
    key_b[0] = 99;
    let result = YantrikDB::new_encrypted(path, 8, &key_b);
    assert!(
        result.is_err(),
        "Opening encrypted DB with wrong key should fail"
    );
}

#[test]
fn test_encrypted_db_reopen_same_key() {
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    let key = test_key();
    let rid;
    {
        let db = YantrikDB::new_encrypted(path, 8, &key).unwrap();
        rid = db
            .record(
                "persistent secret",
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
        db.close().unwrap();
    }

    // Re-open with same key — should decrypt successfully
    {
        let db = YantrikDB::new_encrypted(path, 8, &key).unwrap();
        let mem = db.get(&rid).unwrap().unwrap();
        assert_eq!(mem.text, "persistent secret");
    }
}

#[test]
fn test_open_encrypted_db_without_key_fails() {
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    let key = test_key();
    {
        let db = YantrikDB::new_encrypted(path, 8, &key).unwrap();
        db.record(
            "data",
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
        db.close().unwrap();
    }

    // Open without key — should detect encryption_enabled and refuse
    let result = YantrikDB::new(path, 8);
    assert!(
        result.is_err(),
        "Opening encrypted DB without key should fail"
    );
}
