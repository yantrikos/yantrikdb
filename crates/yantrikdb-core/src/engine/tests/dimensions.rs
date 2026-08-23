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
            None, // event_after (#149)
            None, // event_before (#149)
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
            None, // event_after (#149)
            None, // event_before (#149)
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
            None, // event_after (#149)
            None, // event_before (#149)
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
            None, // event_after (#149)
            None, // event_before (#149)
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
            created_at: None,
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
            created_at: None,
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
            created_at: None,
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

#[test]
fn recall_top_k_above_legacy_candidate_cap_is_not_underfilled() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let inputs = (0..620)
        .map(|index| RecordInput {
            created_at: None,
            idempotency_key: None,
            text: format!("large recall item {index}"),
            memory_type: "episodic".to_string(),
            importance: 0.5,
            valence: 0.0,
            half_life: 604800.0,
            metadata: serde_json::json!({}),
            embedding: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            namespace: "default".to_string(),
            certainty: 0.8,
            domain: "general".to_string(),
            source: "user".to_string(),
            emotional_state: None,
        })
        .collect::<Vec<_>>();
    for chunk in inputs.chunks(256) {
        db.record_batch(chunk).unwrap();
        db.search_state.load().vec_index.compact().unwrap();
    }

    let results = db
        .recall(
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            600,
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
    assert_eq!(results.len(), 600);
}

#[test]
fn explicit_filter_exhausts_small_index_past_nearer_decoys() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let mut inputs = (0..520)
        .map(|index| RecordInput {
            created_at: None,
            idempotency_key: None,
            text: format!("near system decoy {index}"),
            memory_type: "episodic".to_string(),
            importance: 0.5,
            valence: 0.0,
            half_life: 604800.0,
            metadata: serde_json::json!({}),
            embedding: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            namespace: "default".to_string(),
            certainty: 0.8,
            domain: "general".to_string(),
            source: "system".to_string(),
            emotional_state: None,
        })
        .collect::<Vec<_>>();
    inputs.extend((0..20).map(|index| RecordInput {
        created_at: None,
        idempotency_key: None,
        text: format!("eligible user item {index}"),
        memory_type: "episodic".to_string(),
        importance: 0.5,
        valence: 0.0,
        half_life: 604800.0,
        metadata: serde_json::json!({}),
        embedding: vec![0.99, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        namespace: "default".to_string(),
        certainty: 0.8,
        domain: "general".to_string(),
        source: "user".to_string(),
        emotional_state: None,
    }));
    for chunk in inputs.chunks(256) {
        db.record_batch(chunk).unwrap();
        db.search_state.load().vec_index.compact().unwrap();
    }

    let results = db
        .recall(
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            20,
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
            None, // event_after (#149)
            None, // event_before (#149)
        )
        .unwrap();
    assert_eq!(results.len(), 20);
    assert!(results.iter().all(|result| result.source == "user"));
}

#[test]
fn stats_distinguish_verified_provenance_from_legacy_user_source() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let input = |text: &str, source: &str, verified: bool| RecordInput {
        created_at: None,
        idempotency_key: None,
        text: text.to_string(),
        memory_type: "episodic".to_string(),
        importance: 0.5,
        valence: 0.0,
        half_life: 604800.0,
        metadata: if verified {
            serde_json::json!({
                "provenance_verified": true,
                "provenance_method": "direct_turn_v1",
            })
        } else {
            serde_json::json!({})
        },
        embedding: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        namespace: "default".to_string(),
        certainty: 0.8,
        domain: "general".to_string(),
        source: source.to_string(),
        emotional_state: None,
    };
    let verified = input("verified user memory", "user", true);
    let legacy = input("legacy user memory", "user", false);
    let assistant = input("verified assistant memory", "assistant", true);
    let mut other_namespace = input("verified document memory", "document", true);
    other_namespace.namespace = "other".to_string();

    db.record_batch(&[verified, legacy, assistant, other_namespace])
        .unwrap();
    let stats = db.stats(Some("default")).unwrap();

    assert_eq!(stats.provenance_verified_records, 2);
    assert_eq!(stats.unverified_user_source_records, 1);
    assert_eq!(stats.provenance_source_counts.get("user"), Some(&2));
    assert_eq!(stats.provenance_source_counts.get("assistant"), Some(&1));
    assert!(!stats.provenance_source_counts.contains_key("document"));
    assert_eq!(
        stats.provenance_method_counts.get("direct_turn_v1"),
        Some(&2)
    );
    assert_eq!(stats.provenance_method_counts.get("unmarked"), Some(&1));
    assert_eq!(stats.unverified_source_counts.get("user"), Some(&1));
}

#[test]
fn stats_scope_recall_cap_bindings_by_namespace() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.recall_candidate_cap_bound_since_boot.lock().extend(
        [("default", 2), ("other", 3), ("*", 1)]
            .into_iter()
            .map(|(namespace, count)| (namespace.to_string(), count)),
    );

    let global = db.stats(None).unwrap();
    assert_eq!(global.recall_candidate_cap_bound_since_boot, 6);
    assert_eq!(
        global
            .recall_candidate_cap_bound_by_namespace_since_boot
            .get("other"),
        Some(&3)
    );

    let scoped = db.stats(Some("default")).unwrap();
    assert_eq!(scoped.recall_candidate_cap_bound_since_boot, 2);
    assert_eq!(
        scoped.recall_candidate_cap_bound_by_namespace_since_boot,
        std::collections::HashMap::from([("default".to_string(), 2)])
    );
    assert_eq!(global.recall_candidate_cap_namespace_capacity, 1_024);
    assert!(!global.recall_candidate_cap_namespace_stats_truncated_since_boot);
}

#[test]
fn recall_cap_namespace_stats_are_bounded_and_observable() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    for index in 0..=super::super::recall::MAX_TRACKED_RECALL_LIMIT_NAMESPACES {
        db.note_recall_candidate_cap_bound(Some(&format!("tenant-{index}")));
    }

    let stats = db.stats(None).unwrap();
    assert!(stats.recall_candidate_cap_namespace_stats_truncated_since_boot);
    assert_eq!(stats.recall_candidate_cap_bound_since_boot, 1_025);
    assert_eq!(
        stats
            .recall_candidate_cap_bound_by_namespace_since_boot
            .get("<other>"),
        Some(&1)
    );
    assert_eq!(
        stats
            .recall_candidate_cap_bound_by_namespace_since_boot
            .len(),
        1_025,
        "1,024 named namespaces plus one bounded overflow bucket"
    );
}
