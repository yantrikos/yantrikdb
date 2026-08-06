use super::*;

// ════════════════════════════════════════════════════════════════════════════════
// Phase 3: Richer Dimensions (certainty, domain, source, emotional_state)
// ════════════════════════════════════════════════════════════════════════════════

#[test]
fn test_record_with_dimensions() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);
    let rid = db
        .record(
            "meeting notes for Q1 planning",
            "episodic",
            0.7,
            0.2,
            604800.0,
            &empty_meta(),
            &emb,
            "default",
            0.9,
            "work",
            "document",
            Some("joy"),
        )
        .unwrap();

    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.text, "meeting notes for Q1 planning");
    assert!(
        (mem.certainty - 0.9).abs() < 1e-6,
        "certainty should be 0.9, got {}",
        mem.certainty
    );
    assert_eq!(mem.domain, "work");
    assert_eq!(mem.source, "document");
    assert_eq!(mem.emotional_state, Some("joy".to_string()));
}

#[test]
fn test_domain_filter() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // Record 3 memories: 2 in "work" domain, 1 in "health"
    db.record(
        "work task A",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(1.0, 8),
        "default",
        0.8,
        "work",
        "user",
        None,
    )
    .unwrap();
    db.record(
        "health checkup",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(2.0, 8),
        "default",
        0.8,
        "health",
        "user",
        None,
    )
    .unwrap();
    db.record(
        "work task B",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(3.0, 8),
        "default",
        0.8,
        "work",
        "user",
        None,
    )
    .unwrap();

    // Recall with domain="work" should return only the 2 work memories
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
            Some("work"),
            None,
            None,
            None,
            false,
        )
        .unwrap();
    assert_eq!(
        results.len(),
        2,
        "Expected 2 work-domain memories, got {}",
        results.len()
    );
    for r in &results {
        assert_eq!(r.domain, "work");
    }
}

#[test]
fn test_source_filter() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // Record 3 memories: 2 from "user" source, 1 from "system"
    db.record(
        "user input A",
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
        "system log",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(2.0, 8),
        "default",
        0.8,
        "general",
        "system",
        None,
    )
    .unwrap();
    db.record(
        "user input B",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(3.0, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();

    // Recall with source="user" should return only the 2 user-sourced memories
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
            Some("user"),
            None,
            None,
            false,
        )
        .unwrap();
    assert_eq!(
        results.len(),
        2,
        "Expected 2 user-source memories, got {}",
        results.len()
    );
    for r in &results {
        assert_eq!(r.source, "user");
    }
}

#[test]
fn test_domain_and_source_combined_filter() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // Record 4 memories with different domain/source combinations
    db.record(
        "work from user",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(1.0, 8),
        "default",
        0.8,
        "work",
        "user",
        None,
    )
    .unwrap();
    db.record(
        "work from system",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(2.0, 8),
        "default",
        0.8,
        "work",
        "system",
        None,
    )
    .unwrap();
    db.record(
        "health from user",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(3.0, 8),
        "default",
        0.8,
        "health",
        "user",
        None,
    )
    .unwrap();
    db.record(
        "health from system",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(4.0, 8),
        "default",
        0.8,
        "health",
        "system",
        None,
    )
    .unwrap();

    // Filter by domain="work" AND source="user" — should return only 1
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
            Some("work"),
            Some("user"),
            None,
            None,
            false,
        )
        .unwrap();
    assert_eq!(
        results.len(),
        1,
        "Expected 1 work+user memory, got {}",
        results.len()
    );
    assert_eq!(results[0].domain, "work");
    assert_eq!(results[0].source, "user");

    // Filter by domain="health" AND source="system" — should return only 1
    let results = db
        .recall(
            &vec_seed(4.0, 8),
            10,
            None,
            None,
            false,
            false,
            None,
            true,
            None,
            Some("health"),
            Some("system"),
            None,
            None,
            false,
        )
        .unwrap();
    assert_eq!(
        results.len(),
        1,
        "Expected 1 health+system memory, got {}",
        results.len()
    );
    assert_eq!(results[0].domain, "health");
    assert_eq!(results[0].source, "system");
}

#[test]
fn test_dimensions_preserved_on_correct() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "the sky is green",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "default",
            0.6,
            "work",
            "document",
            Some("surprise"),
        )
        .unwrap();

    // Correct the memory (importance; text corrections refused in v0.9.3).
    // Issue #47 (v0.7.20): in-place mutation, rid preserved, reason required.
    let result = db
        .correct(
            &rid,
            None,
            None,
            Some(0.9),
            None,
            "importance fix", // reason
        )
        .unwrap();
    assert!(!result.original_tombstoned);
    assert_eq!(result.corrected_rid, rid);

    // Verify the in-place mutation preserves domain, source, and emotional_state
    // (these aren't touched by correct() since they're not in the signature).
    let updated = db.get(&rid).unwrap().unwrap();
    assert_eq!(
        updated.domain, "work",
        "domain should be preserved after correction"
    );
    assert_eq!(
        updated.source, "document",
        "source should be preserved after correction"
    );
}

#[test]
fn test_batch_record_with_dimensions() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    let inputs: Vec<RecordInput> = vec![
        RecordInput {
            idempotency_key: None,
            text: "batch work meeting".to_string(),
            memory_type: "episodic".to_string(),
            importance: 0.6,
            valence: 0.1,
            half_life: 604800.0,
            metadata: serde_json::json!({"batch": true}),
            embedding: vec_seed(1.0, 8),
            namespace: "default".to_string(),
            certainty: 0.95,
            domain: "work".to_string(),
            source: "document".to_string(),
            emotional_state: Some("focus".to_string()),
        },
        RecordInput {
            idempotency_key: None,
            text: "batch health jog".to_string(),
            memory_type: "episodic".to_string(),
            importance: 0.4,
            valence: 0.3,
            half_life: 604800.0,
            metadata: serde_json::json!({"batch": true}),
            embedding: vec_seed(2.0, 8),
            namespace: "default".to_string(),
            certainty: 0.7,
            domain: "health".to_string(),
            source: "user".to_string(),
            emotional_state: None,
        },
        RecordInput {
            idempotency_key: None,
            text: "batch personal diary".to_string(),
            memory_type: "semantic".to_string(),
            importance: 0.3,
            valence: -0.1,
            half_life: 604800.0,
            metadata: serde_json::json!({"batch": true}),
            embedding: vec_seed(3.0, 8),
            namespace: "default".to_string(),
            certainty: 0.5,
            domain: "personal".to_string(),
            source: "system".to_string(),
            emotional_state: Some("calm".to_string()),
        },
    ];

    let rids = db.record_batch(&inputs).unwrap();
    assert_eq!(rids.len(), 3);

    // Verify first memory
    let m0 = db.get(&rids[0]).unwrap().unwrap();
    assert_eq!(m0.text, "batch work meeting");
    assert!((m0.certainty - 0.95).abs() < 1e-6);
    assert_eq!(m0.domain, "work");
    assert_eq!(m0.source, "document");
    assert_eq!(m0.emotional_state, Some("focus".to_string()));

    // Verify second memory
    let m1 = db.get(&rids[1]).unwrap().unwrap();
    assert_eq!(m1.domain, "health");
    assert_eq!(m1.source, "user");
    assert_eq!(m1.emotional_state, None);

    // Verify third memory
    let m2 = db.get(&rids[2]).unwrap().unwrap();
    assert_eq!(m2.domain, "personal");
    assert_eq!(m2.source, "system");
    assert_eq!(m2.emotional_state, Some("calm".to_string()));
}
