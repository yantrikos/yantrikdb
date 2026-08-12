use super::*;

// ── V4: Storage & Performance tests ──

#[test]
fn test_schema_v6_has_storage_tier() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "tier test",
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
    assert_eq!(mem.storage_tier, "hot");
}

#[test]
fn test_schema_v7_has_fts5_and_join_tables() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // FTS5 virtual table exists — insert then search.
    // Must record BEFORE acquiring conn — db.record() internally takes
    // conn, so holding conn across record() would self-deadlock.
    let _rid = db
        .record(
            "The quick brown fox jumps over the lazy dog",
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

    let conn = db.conn();

    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'quick brown'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 1, "FTS5 should index inserted memory");

    // Join tables exist
    let _: i64 = conn
        .query_row("SELECT COUNT(*) FROM trigger_source_rids", [], |row| {
            row.get(0)
        })
        .unwrap();
    let _: i64 = conn
        .query_row("SELECT COUNT(*) FROM pattern_evidence", [], |row| {
            row.get(0)
        })
        .unwrap();
    let _: i64 = conn
        .query_row("SELECT COUNT(*) FROM pattern_entities", [], |row| {
            row.get(0)
        })
        .unwrap();

    // Schema version is current
    let ver: String = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(ver, crate::schema::SCHEMA_VERSION.to_string());
}

#[test]
fn test_fts5_search_multiple_memories() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    db.record(
        "Alice loves Rust programming",
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
    db.record(
        "Bob prefers Python scripting",
        "semantic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(0.5, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();
    db.record(
        "Alice and Bob work on Rust projects",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(0.3, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();

    // Acquire conn AFTER records are written — db.record() internally takes
    // conn, and holding it across db.record() would self-deadlock (the
    // Mutex<Connection> is non-reentrant). See CONCURRENCY.md Rule 4.
    let conn = db.conn();

    // Search for "Rust" should match 2 memories
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'rust'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 2, "FTS5 should find 2 memories containing 'rust'");

    // Search for "Alice" should match 2 memories
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'alice'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 2, "FTS5 should find 2 memories containing 'alice'");

    // Search for "Python" should match 1
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'python'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_archive_memory() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "to archive",
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

    // Archive
    assert!(db.archive(&rid).unwrap());
    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.storage_tier, "cold");

    // Verify removed from vec_memories (recall should not find it)
    let results = db
        .recall(
            &vec_seed(1.0, 8),
            10,
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
    assert!(
        results.iter().all(|r| r.rid != rid),
        "archived memory should not appear in recall"
    );

    // Stats should show archived
    assert_eq!(db.stats(None).unwrap().archived_memories, 1);
}

#[test]
fn test_hydrate_memory() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(2.0, 8);
    let rid = db
        .record(
            "to hydrate",
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

    // Archive then hydrate
    db.archive(&rid).unwrap();
    assert!(db.hydrate(&rid).unwrap());
    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.storage_tier, "hot");

    // Should be back in recall
    let results = db
        .recall(
            &emb, 10, None, None, false, false, None, true, None, None, None, None, None, false,
        )
        .unwrap();
    assert!(
        results.iter().any(|r| r.rid == rid),
        "hydrated memory should appear in recall"
    );

    // Stats
    assert_eq!(db.stats(None).unwrap().archived_memories, 0);
}

#[test]
fn test_archive_idempotent() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "idempotent",
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

    assert!(db.archive(&rid).unwrap());
    assert!(!db.archive(&rid).unwrap()); // Already cold
}

#[test]
fn test_record_batch() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let inputs: Vec<RecordInput> = (0..10)
        .map(|i| RecordInput {
            created_at: None,
            idempotency_key: None,
            text: format!("batch memory {i}"),
            memory_type: "episodic".to_string(),
            importance: 0.5,
            valence: 0.0,
            half_life: 604800.0,
            metadata: serde_json::json!({}),
            embedding: vec_seed(i as f32, 8),
            namespace: "default".to_string(),
            certainty: 0.8,
            domain: "general".to_string(),
            source: "user".to_string(),
            emotional_state: None,
        })
        .collect();

    let rids = db.record_batch(&inputs).unwrap();
    assert_eq!(rids.len(), 10);

    // All retrievable
    for rid in &rids {
        assert!(db.get(rid).unwrap().is_some());
    }
    assert_eq!(db.stats(None).unwrap().active_memories, 10);
}

#[test]
fn test_record_batch_empty() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = db.record_batch(&[]).unwrap();
    assert!(rids.is_empty());
}

#[test]
fn test_evict() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // Seed 20 memories
    for i in 0..20 {
        db.record(
            &format!("evict mem {i}"),
            "episodic",
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
    assert_eq!(db.stats(None).unwrap().active_memories, 20);

    // Evict to keep 10
    let archived = db.evict(10).unwrap();
    assert_eq!(archived.len(), 10);

    let stats = db.stats(None).unwrap();
    assert_eq!(stats.archived_memories, 10);

    // Recall should only find hot memories
    let results = db
        .recall(
            &vec_seed(0.0, 8),
            20,
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
    for r in &results {
        assert!(
            !archived.contains(&r.rid),
            "evicted memory should not be in recall"
        );
    }
}

#[test]
fn test_evict_no_action_when_under_limit() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    for i in 0..5 {
        db.record(
            &format!("small db {i}"),
            "episodic",
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
    let archived = db.evict(10).unwrap();
    assert!(archived.is_empty());
}

#[test]
fn test_query_builder_basic() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    for i in 0..10 {
        db.record(
            &format!("memory {i}"),
            "episodic",
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

    let results = db
        .query(RecallQuery::new(vec_seed(0.0, 8)).top_k(3).skip_reinforce())
        .unwrap();
    assert_eq!(results.len(), 3);
}

#[test]
fn test_query_builder_with_filters() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record(
        "episodic one",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(1.0, 8),
        "work",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();
    db.record(
        "semantic one",
        "semantic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(2.0, 8),
        "work",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();
    db.record(
        "episodic two",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(3.0, 8),
        "personal",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();

    // Filter by type + namespace
    let results = db
        .query(
            RecallQuery::new(vec_seed(1.0, 8))
                .top_k(10)
                .memory_type("episodic")
                .namespace("work")
                .skip_reinforce(),
        )
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].memory_type, "episodic");
    assert_eq!(results[0].namespace, "work");
}

#[test]
fn test_query_builder_contributions_present() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record(
        "test mem",
        "episodic",
        0.8,
        0.5,
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

    let results = db
        .query(RecallQuery::new(vec_seed(1.0, 8)).top_k(1).skip_reinforce())
        .unwrap();
    assert_eq!(results.len(), 1);

    let r = &results[0];
    // Verify explainability fields
    assert!(r.scores.valence_multiplier >= 1.0);
    assert!(r.scores.contributions.similarity >= 0.0);
    assert!(r.scores.contributions.decay >= 0.0);
    assert!(r.scores.contributions.recency >= 0.0);
    assert!(r.scores.contributions.importance >= 0.0);
}
