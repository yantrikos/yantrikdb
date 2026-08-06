use super::*;

// ── v0.10 4a.7 — trace-contract tests (docs/traces/T06.toml, T07.toml) ──
// These two tests ARE the `test_path` targets of the release-blocking trace
// contracts. Each walks its contract's fixture and asserts every line of its
// `assertions` list, in the default suite (explicit vectors, no embedder, no
// wall-clock asserts). Renaming or weakening them breaks the contract: the
// registry test pins test_path, and implemented -> pending is illegal.

/// TRACE T06 — "anti-laundering-chokepoint" (docs/traces/T06.toml).
///
/// Fixture: a source=inference record; derived write attempts kind=fact/
/// observation; and the consistency matrix (source=inference AND
/// basis=observation). Assertions: typed refusal at WRITE time via the
/// central mutation gate; recall returns source verbatim; bypass paths
/// covered — correct(metadata_merge), record_with_rid, batch,
/// links-after-record, replication.
#[test]
fn t06_anti_laundering_chokepoint() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    assert_eq!(
        db.provenance_gate_mode(),
        crate::provenance::GateMode::Enforce,
        "fresh DB defaults to enforce — the chokepoint is on by default"
    );
    let is_refusal = |e: &crate::error::YantrikDbError| {
        matches!(
            e,
            crate::error::YantrikDbError::ProvenanceInconsistent { .. }
        )
    };
    let rec = |source: &str, meta: serde_json::Value, seed: f32| {
        db.record(
            "t06 probe text",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &meta,
            &vec_seed(seed, 8),
            "t06_ns",
            0.8,
            "general",
            source,
            None,
        )
    };

    // 1) The core laundering shapes: an inference claiming EITHER
    //    authoritative kind — fact or observation — is a TYPED refusal at
    //    write time (the fixture names both; sol 4a.7 r1 finding 1).
    let err = rec("inference", serde_json::json!({"kind": "fact"}), 1.0).unwrap_err();
    assert!(is_refusal(&err), "record() kind=fact: got {err:?}");
    let err = rec("inference", serde_json::json!({"kind": "observation"}), 1.5).unwrap_err();
    assert!(is_refusal(&err), "record() kind=observation: got {err:?}");

    // 2) The consistency matrix: you did not OBSERVE an inference.
    let err = rec(
        "inference",
        serde_json::json!({"kind": "inference", "confidence_basis": "observation"}),
        2.0,
    )
    .unwrap_err();
    assert!(is_refusal(&err), "matrix: got {err:?}");

    // 3) RECALL returns source VERBATIM — asserted through the actual recall
    //    projection, not the point-read (sol 4a.7 r1 finding 2: RecallResult
    //    has its own `source` field, so a recall-path rewrite would slip past
    //    a get()-based check). The gate refuses lies, not lineage.
    let rid = rec("inference", serde_json::json!({"kind": "inference"}), 3.0).unwrap();
    let results = db
        .recall(
            &vec_seed(3.0, 8),
            5,
            None,
            None,
            false,
            false,
            None,
            true, // skip_reinforce: this trace asserts state, not usage
            Some("t06_ns"),
            None,
            None,
            None,
            None,
            false,
        )
        .unwrap();
    let hit = results
        .iter()
        .find(|r| r.rid == rid)
        .expect("the inference record must be recallable");
    assert_eq!(
        hit.source, "inference",
        "recall must return source verbatim"
    );

    // 4) Bypass path: batch. One inconsistent element refuses the WHOLE
    //    batch before any side effect.
    let mk = |source: &str, meta: serde_json::Value, seed: f32| RecordInput {
        idempotency_key: None,
        text: "t06 batch probe".into(),
        memory_type: "semantic".into(),
        importance: 0.5,
        valence: 0.0,
        half_life: 604800.0,
        metadata: meta,
        embedding: vec_seed(seed, 8),
        namespace: "t06_batch_ns".into(),
        certainty: 0.8,
        domain: "general".into(),
        source: source.into(),
        emotional_state: None,
    };
    let err = db
        .record_batch(&[
            mk("user", serde_json::json!({}), 4.0),
            mk("inference", serde_json::json!({"kind": "fact"}), 5.0),
        ])
        .unwrap_err();
    assert!(is_refusal(&err), "batch: got {err:?}");
    let rows: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE namespace = 't06_batch_ns'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(rows, 0, "a refused batch writes nothing");

    // 5) Bypass path: record_with_rid at ORIGIN admission is gated exactly
    //    like record(); at ADMITTED it is deliberately NOT re-gated — the
    //    op was gated at its leader's origin ingress, and re-gating the
    //    apply path would make followers reject consensus-committed writes.
    //    That pair IS the replication coverage: one chokepoint, at origin.
    let err = db
        .record_with_rid(
            "0198c1c2-0000-7000-8000-000000000d60",
            "t06 rwr probe",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({"kind": "fact"}),
            &vec_seed(6.0, 8),
            "t06_ns",
            0.8,
            "general",
            "inference",
            None,
            1_750_000_000_000_000,
            &[],
            "test-embedder",
            None,
            crate::provenance::WriteAdmission::Origin,
        )
        .unwrap_err();
    assert!(is_refusal(&err), "record_with_rid(Origin): got {err:?}");
    db.record_with_rid(
        "0198c1c2-0000-7000-8000-000000000d61",
        "t06 admitted replica probe",
        "semantic",
        0.5,
        0.0,
        604800.0,
        &serde_json::json!({"kind": "fact"}),
        &vec_seed(7.0, 8),
        "t06_ns",
        0.8,
        "general",
        "inference",
        None,
        1_750_000_000_000_000,
        &[],
        "test-embedder",
        None,
        crate::provenance::WriteAdmission::Admitted,
    )
    .expect("Admitted apply is not re-gated (leader gated at origin)");

    // 6) Bypass path: correct(metadata_merge) gates the FINAL merged
    //    metadata against the record's source — no post-hoc promotion.
    let err = db
        .correct(
            &rid,
            None,
            Some(&serde_json::json!({"kind": "fact"})),
            None,
            None,
            "promote",
        )
        .unwrap_err();
    assert!(is_refusal(&err), "correct(metadata_merge): got {err:?}");
    assert_eq!(db.history(&rid).unwrap().len(), 0, "no revision on refusal");

    // 7) Bypass path: links-after-record (record_with_links delegates
    //    through record()'s gate; refused before any link effect).
    let target = rec("user", serde_json::json!({}), 8.0).unwrap();
    let links = [RecordLink {
        target_rid: target.clone(),
        link_type: LinkType::Supports,
    }];
    let err = db
        .record_with_links(
            "t06 laundered via wrapper",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({"kind": "fact"}),
            &vec_seed(9.0, 8),
            "t06_ns",
            0.8,
            "general",
            "inference",
            None,
            &links,
        )
        .unwrap_err();
    assert!(is_refusal(&err), "record_with_links: got {err:?}");
    let inbound = db
        .linked_records(&target, LinkDirection::Inbound, None)
        .unwrap();
    assert!(
        inbound.is_empty(),
        "refused write applied links: {inbound:?}"
    );
}

/// TRACE T07 — "repetition-not-corroboration" (docs/traces/T07.toml).
///
/// Fixture: the same content written 3x with the same idempotency key; and
/// without a key. Assertions: one record with certainty UNCHANGED (retries
/// must not inflate confidence); same key + different payload = typed
/// conflict; no silent near-dup merge without a key.
#[test]
fn t07_repetition_is_not_corroboration() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let write = |text: &str, key: Option<&str>| {
        db.record_with_idempotency(
            text,
            "semantic",
            0.7,
            0.0,
            604800.0,
            &serde_json::json!({}),
            &vec_seed(1.0, 8),
            "t07_ns",
            0.8,
            "general",
            "user",
            None,
            key,
        )
    };

    // Same content, same key, three times.
    let rid1 = write("the same assertion", Some("t07-key")).unwrap();
    let rid2 = write("the same assertion", Some("t07-key")).unwrap();
    let rid3 = write("the same assertion", Some("t07-key")).unwrap();
    assert_eq!(rid1, rid2);
    assert_eq!(rid1, rid3, "every retry returns the ORIGINAL rid");

    // ONE record…
    let rows: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE namespace = 't07_ns'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(rows, 1, "three keyed writes, one record");
    // …with certainty UNCHANGED: repetition must not inflate confidence.
    let (certainty, importance): (f64, f64) = db
        .conn()
        .query_row(
            "SELECT certainty, importance FROM memories WHERE rid = ?1",
            rusqlite::params![rid1],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(certainty, 0.8, "certainty untouched by retries");
    // …and the namespace observed exactly ONE importance sample (a retry
    // that advanced the EWMA would be corroboration through the back door).
    let stats_count: i64 = db
        .conn()
        .query_row(
            "SELECT count FROM namespace_importance_stats WHERE namespace = 't07_ns'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(stats_count, 1, "one write, one calibration observation");
    let _ = importance; // stored calibrated value; pinned by 4a.6b tests

    // Same key, DIFFERENT payload: a typed conflict, never a silent merge.
    let err = write("a different assertion", Some("t07-key")).unwrap_err();
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::IdempotencyConflict { .. }
        ),
        "expected IdempotencyConflict, got {err:?}"
    );

    // WITHOUT a key: the same content three times is three records — the
    // engine must not silently near-dup-merge unkeyed writes.
    let ka = write("unkeyed repeated content", None).unwrap();
    let kb = write("unkeyed repeated content", None).unwrap();
    let kc = write("unkeyed repeated content", None).unwrap();
    assert_ne!(ka, kb);
    assert_ne!(kb, kc);
    let rows: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE namespace = 't07_ns' AND text = 'unkeyed repeated content'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(rows, 3, "no key, no dedup: three distinct records");
}

// ── Relationship-Based Entity Type Tests ──

#[test]
fn test_relate_infers_entity_types() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.relate("MyApp", "React", "built_with", 1.0).unwrap();

    let entities = db.search_entities(Some("MyApp"), None, 1).unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].entity_type, "project");

    let entities = db.search_entities(Some("React"), None, 1).unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].entity_type, "tech");
}

#[test]
fn test_relate_infers_infrastructure() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.relate("Backend", "AWS", "deployed_on", 1.0).unwrap();

    let entities = db.search_entities(Some("AWS"), None, 1).unwrap();
    assert_eq!(entities[0].entity_type, "infrastructure");
}

// ── Confidence-Calibrated Recall Tests ──

#[test]
fn test_recall_with_response_has_certainty_reasons() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);
    db.record(
        "Important architecture decision about microservices",
        "semantic",
        0.8,
        0.0,
        604800.0,
        &empty_meta(),
        &emb,
        "default",
        0.8,
        "work",
        "user",
        None,
    )
    .unwrap();

    let response = db
        .recall_with_response(
            &emb,
            5,
            None,
            None,
            false,
            false,
            Some("architecture decision"),
            false,
            None,
            None,
            None,
        )
        .unwrap();

    assert!(
        !response.certainty_reasons.is_empty(),
        "should have certainty reasons"
    );
    assert!(response.confidence >= 0.0 && response.confidence <= 1.0);
}

#[test]
fn test_recall_empty_db_low_confidence() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);

    let response = db
        .recall_with_response(
            &emb,
            5,
            None,
            None,
            false,
            false,
            Some("anything"),
            false,
            None,
            None,
            None,
        )
        .unwrap();

    assert!(
        response.confidence < 0.5,
        "empty DB should have low confidence"
    );
    assert!(
        response
            .certainty_reasons
            .iter()
            .any(|r| r.contains("No") || r.contains("Sparse") || r.contains("Weak")),
        "should explain low confidence"
    );
}

// ── Relationship Depth Tests ──

#[test]
fn test_relationship_depth_basic() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);

    // Create an entity with some relationships and memories
    db.relate("Alice", "Bob", "knows", 1.0).unwrap();
    db.relate("Alice", "ProjectX", "works_on", 1.0).unwrap();

    db.record(
        "Alice presented the quarterly report",
        "episodic",
        0.5,
        0.3,
        604800.0,
        &empty_meta(),
        &emb,
        "default",
        0.8,
        "work",
        "user",
        None,
    )
    .unwrap();
    db.record(
        "Alice prefers async communication",
        "semantic",
        0.6,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(2.0, 8),
        "default",
        0.8,
        "preference",
        "user",
        None,
    )
    .unwrap();
    // Phase 4.3: drain the post-record materialization queue so the
    // memory_entities link is visible to relationship_depth.
    db.apply_pending_ops_once(100).unwrap();

    let depth = db.relationship_depth("Alice", None).unwrap();
    assert_eq!(depth.entity, "Alice");
    assert_eq!(depth.entity_type, "person");
    assert!(
        depth.connection_count >= 2,
        "Alice connected to Bob and ProjectX"
    );
    assert!(
        depth.memories_mentioning >= 2,
        "at least 2 memories mention Alice"
    );
    assert!(depth.depth_score > 0.0, "should have positive depth score");
    assert!(depth.depth_score <= 1.0);
}

#[test]
fn test_relationship_depth_not_found() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let result = db.relationship_depth("NonexistentEntity", None);
    assert!(result.is_err(), "should error for unknown entity");
}

// ── Procedural Memory Tests ──

#[test]
fn test_record_and_surface_procedural() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(3.0, 8);

    let rid = db
        .record_procedural(
            "Use Agent tool with Explore subtype for architectural questions in this codebase",
            &emb,
            "work",
            "code search",
            0.8,
            "default",
        )
        .unwrap();

    // Verify it was stored as procedural type
    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.memory_type, "procedural");
    assert!((mem.importance - 0.8).abs() < 0.01);

    // Surface it with a similar query
    let results = db
        .surface_procedural(&emb, Some("how to search code"), Some("work"), 5, None)
        .unwrap();
    assert!(!results.is_empty(), "should surface the procedural memory");
    assert_eq!(results[0].memory_type, "procedural");
}

#[test]
fn test_reinforce_procedural() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(4.0, 8);

    let rid = db
        .record_procedural(
            "Always run tests before pushing",
            &emb,
            "work",
            "git workflow",
            0.5,
            "default",
        )
        .unwrap();

    // Reinforce with high outcome
    let reinforced = db.reinforce_procedural(&rid, 1.0).unwrap();
    assert!(reinforced);

    // Check importance increased
    let mem = db.get(&rid).unwrap().unwrap();
    assert!(
        mem.importance > 0.5,
        "importance should increase after positive reinforcement"
    );
}

#[test]
fn test_procedural_stats() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record_procedural(
        "proc 1",
        &vec_seed(1.0, 8),
        "work",
        "task A",
        0.7,
        "default",
    )
    .unwrap();
    db.record_procedural(
        "proc 2",
        &vec_seed(2.0, 8),
        "work",
        "task B",
        0.9,
        "default",
    )
    .unwrap();
    db.record_procedural(
        "proc 3",
        &vec_seed(3.0, 8),
        "health",
        "exercise",
        0.5,
        "default",
    )
    .unwrap();

    let stats = db.procedural_stats(None).unwrap();
    assert!(
        stats.len() >= 2,
        "should have stats for work and health domains"
    );
    let work_stats = stats.iter().find(|(d, _, _)| d == "work");
    assert!(work_stats.is_some());
    let (_, count, _) = work_stats.unwrap();
    assert_eq!(*count, 2);
}

// ── Session + Think Integration Tests ──

#[test]
fn test_session_awareness_trigger() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);

    // Start and end a session
    let sid = db
        .session_start("default", "claude", &serde_json::json!({}))
        .unwrap();
    db.record(
        "Worked on battle testing the MCP server",
        "episodic",
        0.7,
        0.5,
        604800.0,
        &empty_meta(),
        &emb,
        "default",
        0.8,
        "work",
        "user",
        None,
    )
    .unwrap();
    let _summary = db
        .session_end(&sid, Some("Battle tested MCP server v0.2.8"))
        .unwrap();

    // Simulate time passing by backdating the session
    db.conn().execute(
        "UPDATE sessions SET ended_at = ended_at - 86400 * 3, started_at = started_at - 86400 * 3 WHERE session_id = ?1",
        params![sid],
    ).unwrap();

    // Run think — should generate a session_awareness trigger
    let config = ThinkConfig {
        run_consolidation: false,
        run_conflict_scan: false,
        run_pattern_mining: false,
        run_personality: false,
        ..Default::default()
    };
    let result = db.think(&config).unwrap();

    let session_triggers: Vec<_> = result
        .triggers
        .iter()
        .filter(|t| t.trigger_type == "session_awareness")
        .collect();
    assert!(
        !session_triggers.is_empty(),
        "should generate session awareness trigger after 3-day gap"
    );
    assert!(
        session_triggers[0].reason.contains("hours"),
        "reason should mention time gap"
    );
}
