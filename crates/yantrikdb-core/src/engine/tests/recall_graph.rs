use super::*;

#[test]
fn test_think_empty_db() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let config = ThinkConfig {
        run_consolidation: false,
        run_conflict_scan: false,
        run_pattern_mining: false,
        ..Default::default()
    };
    let result = db.think(&config).unwrap();
    assert!(result.triggers.is_empty());
    assert_eq!(result.consolidation_count, 0);
    assert_eq!(result.conflicts_found, 0);
    assert!(result.duration_ms < 5000);
}

#[test]
fn test_think_with_decayed_memories() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "important deadline",
            "episodic",
            0.9,
            0.0,
            100.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    // Backdate last_access
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    db.conn()
        .execute(
            "UPDATE memories SET last_access = ?1 WHERE rid = ?2",
            rusqlite::params![ts - 10000.0, rid],
        )
        .unwrap();

    let config = ThinkConfig {
        run_consolidation: false,
        run_conflict_scan: false,
        run_pattern_mining: false,
        ..Default::default()
    };
    let result = db.think(&config).unwrap();
    assert!(!result.triggers.is_empty());
    assert_eq!(result.triggers[0].trigger_type, "decay_review");
}

#[test]
fn test_think_records_last_think_at() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let config = ThinkConfig {
        run_consolidation: false,
        run_conflict_scan: false,
        run_pattern_mining: false,
        ..Default::default()
    };
    db.think(&config).unwrap();

    let val: String = db
        .conn()
        .query_row(
            "SELECT value FROM meta WHERE key = 'last_think_at'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let ts: f64 = val.parse().unwrap();
    assert!(ts > 0.0);
}

#[test]
fn test_trigger_lifecycle() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // Create a trigger via persistence
    let trigger = crate::types::Trigger {
        trigger_type: "decay_review".to_string(),
        reason: "test".to_string(),
        urgency: 0.8,
        source_rids: vec!["rid-1".to_string()],
        suggested_action: "test".to_string(),
        context: std::collections::HashMap::new(),
    };
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let tid = crate::triggers::persist_trigger(&db, &trigger, ts)
        .unwrap()
        .unwrap();

    // Verify pending
    let pending = db.get_pending_triggers(10).unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].status, "pending");

    // Deliver
    assert!(db.deliver_trigger(&tid).unwrap());
    let history = db.get_trigger_history(None, 10).unwrap();
    assert_eq!(history[0].status, "delivered");

    // Acknowledge
    assert!(db.acknowledge_trigger(&tid).unwrap());

    // Act
    assert!(db.act_on_trigger(&tid).unwrap());
    let history = db.get_trigger_history(None, 10).unwrap();
    assert_eq!(history[0].status, "acted");
}

#[test]
fn test_stats_include_triggers_and_patterns() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let s = db.stats(None).unwrap();
    assert_eq!(s.pending_triggers, 0);
    assert_eq!(s.active_patterns, 0);
}

// ── Graph-augmented recall: invariant & regression tests ──

#[test]
fn test_entity_type_stored_on_relate() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // "knows" is a person-person relationship
    db.relate("Sarah", "Mike", "knows", 1.0).unwrap();
    // "works_at" → src=person, dst=organization
    db.relate("Sarah", "Flipkart", "works_at", 1.0).unwrap();
    // "lives_in" → src=person, dst=place
    db.relate("Sarah", "Bangalore", "lives_in", 1.0).unwrap();
    // Tech blocklist still works
    db.relate("FAISS", "recommendation engine", "used_in", 1.0)
        .unwrap();

    let etype: String = db
        .conn()
        .query_row(
            "SELECT entity_type FROM entities WHERE name = 'Sarah'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(etype, "person");

    let etype: String = db
        .conn()
        .query_row(
            "SELECT entity_type FROM entities WHERE name = 'Mike'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(etype, "person");

    let etype: String = db
        .conn()
        .query_row(
            "SELECT entity_type FROM entities WHERE name = 'Flipkart'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(etype, "organization");

    let etype: String = db
        .conn()
        .query_row(
            "SELECT entity_type FROM entities WHERE name = 'Bangalore'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(etype, "place");

    let etype: String = db
        .conn()
        .query_row(
            "SELECT entity_type FROM entities WHERE name = 'FAISS'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(etype, "tech");

    let etype: String = db
        .conn()
        .query_row(
            "SELECT entity_type FROM entities WHERE name = 'recommendation engine'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(etype, "unknown");
}

#[test]
fn test_recall_deterministic_with_skip_reinforce() {
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
    let query = vec_seed(3.0, 8);

    let r1 = db
        .recall(
            &query, 5, None, None, false, false, None, true, None, None, None, None, None, false,
        )
        .unwrap();
    let r2 = db
        .recall(
            &query, 5, None, None, false, false, None, true, None, None, None, None, None, false,
        )
        .unwrap();
    let r3 = db
        .recall(
            &query, 5, None, None, false, false, None, true, None, None, None, None, None, false,
        )
        .unwrap();

    // Same RIDs in same order every time
    let rids1: Vec<&str> = r1.iter().map(|r| r.rid.as_str()).collect();
    let rids2: Vec<&str> = r2.iter().map(|r| r.rid.as_str()).collect();
    let rids3: Vec<&str> = r3.iter().map(|r| r.rid.as_str()).collect();
    assert_eq!(rids1, rids2);
    assert_eq!(rids2, rids3);

    // Scores very close (tiny drift from wall-clock recency between calls)
    for i in 0..5 {
        assert!(
            (r1[i].score - r2[i].score).abs() < 1e-4,
            "score drift too large between calls: {} vs {}",
            r1[i].score,
            r2[i].score
        );
    }
}

#[test]
fn test_reinforce_mutates_but_skip_does_not() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "test",
            "episodic",
            0.5,
            0.0,
            1000.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    let original_hl = db.get(&rid).unwrap().unwrap().half_life;

    // skip_reinforce=true should NOT change half_life
    db.recall(
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
    )
    .unwrap();
    let after_skip = db.get(&rid).unwrap().unwrap().half_life;
    assert!((original_hl - after_skip).abs() < 1e-10);

    // skip_reinforce=false SHOULD change half_life
    db.recall(
        &vec_seed(1.0, 8),
        1,
        None,
        None,
        false,
        false,
        None,
        false,
        None,
        None,
        None,
        None,
        None,
        false,
    )
    .unwrap();
    let after_reinforce = db.get(&rid).unwrap().unwrap().half_life;
    assert!(after_reinforce > original_hl);
}

#[test]
fn test_graph_expansion_off_no_graph_results() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let r1 = db
        .record(
            "Alice discussed plan",
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
    let r2 = db
        .record(
            "Bob reviewed code",
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
    db.relate("Alice", "Bob", "knows", 1.0).unwrap();
    db.link_memory_entity(&r1, "Alice").unwrap();
    db.link_memory_entity(&r2, "Bob").unwrap();

    // expand_entities=false: no graph_proximity should be set
    let results = db
        .recall(
            &vec_seed(1.0, 8),
            10,
            None,
            None,
            false,
            false,
            Some("Alice"),
            false,
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
            (r.scores.graph_proximity - 0.0).abs() < 1e-10,
            "graph_proximity should be 0.0 when expansion is disabled"
        );
    }
}

#[test]
fn test_graph_expansion_on_boosts_connected_memory() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let r1 = db
        .record(
            "Alice discussed the project plan",
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
    let r2 = db
        .record(
            "Bob reviewed the code",
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
    db.relate("Alice", "Bob", "knows", 1.0).unwrap();
    db.link_memory_entity(&r1, "Alice").unwrap();
    db.link_memory_entity(&r2, "Bob").unwrap();

    // expand_entities=true with query mentioning "Alice"
    let results = db
        .recall(
            &vec_seed(1.0, 8),
            10,
            None,
            None,
            false,
            true,
            Some("What is Alice working on?"),
            true,
            None,
            None,
            None,
            None,
            None,
            false,
        )
        .unwrap();

    // The Alice memory should have graph_proximity > 0
    let alice_result = results.iter().find(|r| r.rid == r1).unwrap();
    assert!(
        alice_result.scores.graph_proximity > 0.0,
        "Alice memory should have graph proximity when expansion is on"
    );
}

#[test]
fn test_backfill_uses_word_boundaries() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // Create an entity "data"
    db.relate("data", "pipeline", "part_of", 1.0).unwrap();

    // Create memories: one with "data" as a word, one with "database" (contains "data")
    let r1 = db
        .record(
            "the data is clean",
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
    let r2 = db
        .record(
            "the database is fast",
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

    let _count = db.backfill_memory_entities().unwrap();

    // Check: r1 should be linked to "data", r2 should NOT
    let linked_to_data: Vec<String> = db
        .conn()
        .prepare("SELECT memory_rid FROM memory_entities WHERE entity_name = 'data'")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();

    assert!(
        linked_to_data.contains(&r1),
        "memory with 'data' as word should be linked"
    );
    assert!(
        !linked_to_data.contains(&r2),
        "memory with 'database' should NOT be linked (word boundary)"
    );
}

#[test]
fn test_recall_scores_bounded() {
    // All recall scores should be non-negative and reasonably bounded
    let db = YantrikDB::new(":memory:", 8).unwrap();
    for i in 0..10 {
        db.record(
            &format!("memory {i}"),
            "episodic",
            (i as f64) * 0.1,         // importance 0..0.9
            ((i as f64) - 5.0) * 0.2, // valence -1.0..0.8
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
        .recall(
            &vec_seed(5.0, 8),
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
    for r in &results {
        assert!(
            r.score >= 0.0,
            "score should be non-negative, got {}",
            r.score
        );
        assert!(
            r.score < 5.0,
            "score should be reasonably bounded, got {}",
            r.score
        );
        assert!(r.scores.similarity >= -1.0 && r.scores.similarity <= 1.0);
        assert!(r.scores.decay >= 0.0 && r.scores.decay <= 1.0);
        assert!(r.scores.recency >= 0.0 && r.scores.recency <= 1.0);
    }
}

#[test]
fn test_link_memory_entity_idempotent() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "test",
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
    db.relate("Alice", "Bob", "knows", 1.0).unwrap();

    // Link twice — should not error or create duplicates
    db.link_memory_entity(&rid, "Alice").unwrap();
    db.link_memory_entity(&rid, "Alice").unwrap();

    let count: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memory_entities WHERE memory_rid = ?1 AND entity_name = 'Alice'",
            params![rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_schema_v5_has_memory_entities() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let count: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memory_entities'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_recall_top_k_respected_with_graph_expansion() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // Create a web of interconnected memories
    for i in 0..20 {
        let rid = db
            .record(
                &format!("memory about topic {i}"),
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
        let entity = format!("Entity{i}");
        db.relate(
            &entity,
            &format!("Entity{}", (i + 1) % 20),
            "related_to",
            1.0,
        )
        .unwrap();
        db.link_memory_entity(&rid, &entity).unwrap();
    }

    let results = db
        .recall(
            &vec_seed(0.0, 8),
            5,
            None,
            None,
            false,
            true,
            Some("Entity0 topic"),
            true,
            None,
            None,
            None,
            None,
            None,
            false,
        )
        .unwrap();

    // top_k=5 must be respected even with graph expansion
    assert!(
        results.len() <= 5,
        "results should not exceed top_k=5, got {}",
        results.len()
    );
}
