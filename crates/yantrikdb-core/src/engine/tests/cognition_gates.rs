use super::*;

// ── V3 Cognition tests ──

#[test]
fn test_schema_v4_has_trigger_log_and_patterns() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let count: i64 = db.conn().query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('trigger_log', 'patterns')",
        [], |row| row.get(0),
    ).unwrap();
    assert_eq!(count, 2);
}

// RFC 007 Phase 0: the five-layer reasoning substrate tables exist on fresh install.
#[test]
fn test_schema_v19_has_rfc007_tables() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let count: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN \
         ('propositions', 'variables', 'state_assertions', 'rule_edges', 'scenario_specs')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        count, 5,
        "RFC 007 Phase 0 should create all five new tables"
    );
}

#[test]
fn test_schema_v19_claims_has_proposition_id() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // PRAGMA table_info returns a row per column; we assert proposition_id is among them.
    let conn = db.conn();
    let mut stmt = conn.prepare("PRAGMA table_info(claims)").unwrap();
    let cols: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();
    assert!(
        cols.contains(&"proposition_id".to_string()),
        "claims table should have a proposition_id column after V19. Got: {:?}",
        cols
    );
}

#[test]
fn test_schema_v19_rule_edge_whitelist_enforced() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // Seed a variable so the FK is valid.
    db.conn()
        .execute(
            "INSERT INTO variables (variable_id, name, namespace, value_space, scope, created_at) \
         VALUES ('v1', 'var_a', 'default', '{}', 'generic', 0.0)",
            [],
        )
        .unwrap();
    db.conn()
        .execute(
            "INSERT INTO variables (variable_id, name, namespace, value_space, scope, created_at) \
         VALUES ('v2', 'var_b', 'default', '{}', 'generic', 0.0)",
            [],
        )
        .unwrap();
    // Disallowed edge_type — CHECK constraint should reject.
    let result = db.conn().execute(
        "INSERT INTO rule_edges (rule_id, parent_variable_id, child_variable_id, edge_type, \
         direction_confidence, persistence, scope, source, namespace, created_at) \
         VALUES ('r1', 'v1', 'v2', 'implies', 'high', 'instantaneous', 'generic', 'test', 'default', 0.0)",
        [],
    );
    assert!(
        result.is_err(),
        "rule_edges should reject edge_type='implies' — only whitelist (causal_promotes, causal_inhibits, requires) allowed"
    );
    // Allowed edge_type — should succeed.
    db.conn().execute(
        "INSERT INTO rule_edges (rule_id, parent_variable_id, child_variable_id, edge_type, \
         direction_confidence, persistence, scope, source, namespace, created_at) \
         VALUES ('r2', 'v1', 'v2', 'causal_promotes', 'high', 'instantaneous', 'generic', 'test', 'default', 0.0)",
        [],
    ).unwrap();
}

// RFC 008 Phase 1: Warrant Flow — mobility_state + actor_profile + compression_artifact
// on fresh install, plus write-time mobility signal columns on claims.
#[test]
fn test_schema_v20_has_rfc008_phase1_tables() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let count: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN \
         ('mobility_state', 'actor_profile', 'compression_artifact')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        count, 3,
        "RFC 008 Phase 1 should create mobility_state, actor_profile, compression_artifact"
    );
}

#[test]
fn test_schema_v37_item4a_columns_table_and_index() {
    // v0.10 Item 4a.2: memories gains confidence_basis / idempotency_key /
    // origin_actor; the idempotency_claims table + actor-scoped partial unique
    // index exist; schema version is stamped 37.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let conn = db.conn();
    for col in ["confidence_basis", "idempotency_key", "origin_actor"] {
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name = ?1",
                params![col],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1, "memories.{col} must exist at v37");
    }
    let has_table: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='idempotency_claims'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(has_table, 1, "idempotency_claims table must exist");
    let has_index: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_memories_idempotency'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(has_index, 1, "actor-scoped idempotency index must exist");
    let version: i64 = conn
        .query_row(
            "SELECT CAST(value AS INTEGER) FROM meta WHERE key = 'schema_version'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    // `>=`, not `==`: this test's subject is that the v37 Item-4a surfaces
    // EXIST (asserted above), not that v37 is the newest schema. An equality
    // pin here fails on every later bump for no reason — it broke on the v38
    // materializer-index migration (#113) despite every Item-4a surface being
    // intact — which trains the next person to edit the number rather than
    // read the failure. Forward-only stamping (see `open`'s MAX stamp) makes
    // `>=` the correct assertion.
    assert!(
        version >= 37,
        "schema version must be at least 37 (Item 4a); got {version}"
    );
}

// ── Item 4a.4: anti-laundering gate wiring + backward-compat modes ──

#[cfg(feature = "bundled-embedder")]
fn gate_rec(db: &YantrikDB, source: &str, meta: serde_json::Value) -> crate::error::Result<String> {
    db.record(
        "some memory text",
        "semantic",
        0.5,
        0.0,
        604800.0,
        &meta,
        &vec_seed(1.0, 8),
        "default",
        0.8,
        "general",
        source,
        None,
    )
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn gate_fresh_db_defaults_enforce_and_refuses_inference_claiming_fact() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    assert_eq!(
        db.provenance_gate_mode(),
        crate::provenance::GateMode::Enforce,
        "fresh DB defaults to enforce"
    );
    assert_eq!(db.stats(None).unwrap().provenance_gate_mode, "enforce");
    // source=inference claiming kind=fact -> refused at write time
    let err = gate_rec(&db, "inference", serde_json::json!({"kind": "fact"})).unwrap_err();
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::ProvenanceInconsistent { .. }
        ),
        "got {err:?}"
    );
    // a consistent inference record (kind=inference) is fine
    gate_rec(&db, "inference", serde_json::json!({"kind": "inference"})).unwrap();
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn gate_confirmation_allowance_and_override_escape() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // raise-your-basis: confirmation lets an inference claim a fact
    gate_rec(
        &db,
        "inference",
        serde_json::json!({"kind": "fact", "confidence_basis": "confirmation"}),
    )
    .unwrap();
    // explicit override_kind escape
    gate_rec(
        &db,
        "inference",
        serde_json::json!({"kind": "fact", "override_kind": true}),
    )
    .unwrap();
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn gate_warn_mode_counts_but_allows() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.set_provenance_gate_mode(crate::provenance::GateMode::Warn)
        .unwrap();
    let rid = gate_rec(&db, "inference", serde_json::json!({"kind": "fact"})).unwrap();
    assert!(
        db.get(&rid).unwrap().is_some(),
        "warn mode allows the write"
    );
    assert_eq!(
        db.stats(None).unwrap().provenance_flagged_since_boot,
        1,
        "warn mode increments the nudge counter"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn gate_off_mode_skips_entirely() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.set_provenance_gate_mode(crate::provenance::GateMode::Off)
        .unwrap();
    gate_rec(&db, "inference", serde_json::json!({"kind": "fact"})).unwrap();
    assert_eq!(
        db.stats(None).unwrap().provenance_flagged_since_boot,
        0,
        "off mode does not even count"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn gate_source_is_free_form_matrix_binds_only_recognized_inference() {
    // `source` is a FREE-FORM public dimension — tests/test_phases.py records
    // source="manager" and asserts it round-trips verbatim, alongside domain /
    // emotional_state. So an UNRECOGNIZED source is a label the engine takes no
    // position on: the matrix does not bind it, even under enforce.
    //
    // sol 4a.4 wanted strict parsing (to block a source="inference_v2" alias).
    // We decline: (1) it breaks that documented contract for every caller
    // labelling records manager/slack/paper, and (2) it buys nothing — sol's own
    // analysis concedes the internally-consistent lie source="user"+kind="fact"
    // is undetectable, so a caller willing to alias is equally willing to write
    // "user". The gate catches DECLARED CONTRADICTIONS, never lies.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    assert_eq!(
        db.provenance_gate_mode(),
        crate::provenance::GateMode::Enforce
    );
    // Free-form sources are accepted verbatim, even claiming kind=fact.
    for src in ["manager", "paper", "experiment", "inference_v2"] {
        let rid = gate_rec(&db, src, serde_json::json!({"kind": "fact"})).unwrap();
        assert_eq!(
            db.get(&rid).unwrap().unwrap().source,
            src,
            "free-form source must round-trip verbatim"
        );
    }
    // The matrix binds the RECOGNIZED inference source, and still refuses.
    let err = gate_rec(&db, "inference", serde_json::json!({"kind": "fact"})).unwrap_err();
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::ProvenanceInconsistent { .. }
        ),
        "recognized source=inference claiming kind=fact must still be refused, got {err:?}"
    );
    // Nothing was flagged: free-form sources are not violations.
    assert_eq!(db.stats(None).unwrap().provenance_flagged_since_boot, 0);
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn gate_batch_rejects_inconsistent_element_atomically() {
    // Item 4a.4b: the gate runs in record_batch's PREVALIDATION loop, so an
    // inconsistent element LATE in the batch rejects the whole batch before any
    // side effect — earlier elements must not be half-committed.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let mk = |text: &str, source: &str, meta: serde_json::Value| RecordInput {
        idempotency_key: None,
        text: text.to_string(),
        memory_type: "semantic".to_string(),
        importance: 0.5,
        valence: 0.0,
        half_life: 604800.0,
        metadata: meta,
        embedding: vec_seed(1.0, 8),
        namespace: "default".to_string(),
        certainty: 0.8,
        domain: "general".to_string(),
        source: source.to_string(),
        emotional_state: None,
    };
    let batch = vec![
        mk("good one", "user", empty_meta()),
        // late bad element: inference claiming fact
        mk(
            "laundered",
            "inference",
            serde_json::json!({"kind": "fact"}),
        ),
    ];
    let err = db.record_batch(&batch).unwrap_err();
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::ProvenanceInconsistent { .. }
        ),
        "batch with an inconsistent element must be refused, got {err:?}"
    );
    // Atomic: the GOOD element must not have been written either.
    let count: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0, "a rejected batch must write nothing");
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn gate_covers_links_after_record() {
    // T06 names "links-after-record" as a bypass path that must be covered.
    // record_with_links (links.rs:118) delegates to record() with `?`, so the
    // gate refusal propagates BEFORE the link loop at :132 and no edge is
    // applied. That is the right behavior, but it is a TRANSITIVE guarantee —
    // it holds only as long as the wrapper keeps going through record(). If
    // someone later inlines the insert here, the gate silently stops covering
    // this path, so pin it.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let target = db
        .record(
            "an existing target",
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
    let links = [RecordLink {
        target_rid: target.clone(),
        link_type: LinkType::Supports,
    }];
    let err = db
        .record_with_links(
            "laundered via the wrapper",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({"kind": "fact"}),
            &vec_seed(2.0, 8),
            "default",
            0.8,
            "general",
            "inference",
            None,
            &links,
        )
        .unwrap_err();
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::ProvenanceInconsistent { .. }
        ),
        "record_with_links must refuse an inference claiming kind=fact, got {err:?}"
    );
    // Refused before ANY link effect: the target gained no inbound edge.
    let inbound = db
        .linked_records(&target, LinkDirection::Inbound, None)
        .unwrap();
    assert!(
        inbound.is_empty(),
        "a refused record_with_links must apply no links, found {inbound:?}"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn gate_correct_metadata_merge_cannot_flip_kind_to_fact() {
    // Item 4a.4b: a metadata_merge can flip `kind` on an existing record, so
    // correct() gates the FINAL MERGED metadata against the record's source —
    // otherwise correct() is a trivial laundering bypass of record()'s gate.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let rid = db
        .record_text(
            "an inferred conclusion",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({"kind": "inference"}),
            "default",
            0.8,
            "general",
            "inference",
            None,
        )
        .unwrap();
    // Laundering attempt: promote the inference to a fact via metadata_merge.
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
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::ProvenanceInconsistent { .. }
        ),
        "correct(metadata_merge) must not flip an inference to kind=fact, got {err:?}"
    );
    // Refused before any side effect: no revision, kind unchanged.
    assert_eq!(db.history(&rid).unwrap().len(), 0, "no revision recorded");
    // Raising the basis is the documented escape and IS allowed.
    db.correct(
        &rid,
        None,
        Some(&serde_json::json!({"kind": "fact", "confidence_basis": "verification"})),
        None,
        None,
        "verified independently",
    )
    .unwrap();
}

#[test]
fn gate_mode_parse_is_fail_closed() {
    // A malformed persisted mode must NOT silently disable the gate (the
    // fail-open class): only an exact "off" yields Off.
    use crate::provenance::GateMode;
    assert_eq!(GateMode::parse("off").unwrap(), GateMode::Off);
    assert_eq!(GateMode::parse("  Enforce ").unwrap(), GateMode::Enforce);
    assert_eq!(GateMode::parse("warn").unwrap(), GateMode::Warn);
    assert!(
        GateMode::parse("enforc").is_err(),
        "a typo must be a loud error, never a silent Off"
    );
    assert!(GateMode::parse("").is_err());
}

#[test]
fn test_schema_v37_idempotency_claims_actor_scoped_pk() {
    // The claims PK (origin_actor, namespace, idempotency_key) is the
    // serialization point: same (actor, ns, key) conflicts; a DIFFERENT actor
    // with the same ns+key is allowed (actor-scoped, forward-compat for 4b).
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let conn = db.conn();
    let ins = |actor: &str, rid: &str, op: &str| {
        conn.execute(
            "INSERT INTO idempotency_claims \
             (origin_actor, namespace, idempotency_key, rid, payload_digest, op_id, route, generation, state, created_at) \
             VALUES (?1, 'ns', 'k', ?2, X'00', ?3, 'sync', 0, 'pending', 1.0)",
            params![actor, rid, op],
        )
    };
    ins("actor-A", "rid1", "op1").unwrap();
    assert!(
        ins("actor-A", "rid2", "op2").is_err(),
        "same (actor, ns, key) must conflict"
    );
    ins("actor-B", "rid3", "op3").unwrap(); // different actor: allowed
}

#[test]
fn test_schema_v20_claims_has_mobility_signals() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let conn = db.conn();
    let mut stmt = conn.prepare("PRAGMA table_info(claims)").unwrap();
    let cols: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();
    for expected in &[
        "regime_tag",
        "self_generated",
        "source_lineage",
        "modality_signal",
    ] {
        assert!(
            cols.contains(&expected.to_string()),
            "claims table should have column {} after V20. Got: {:?}",
            expected,
            cols
        );
    }
}

#[test]
fn test_schema_v20_actor_profile_whitelist_enforced() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // Allowed actor_type — should succeed.
    db.conn()
        .execute(
            "INSERT INTO actor_profile (actor_id, actor_type, regime, last_updated) \
         VALUES ('ext_medical', 'extractor', 'medical', 0.0)",
            [],
        )
        .unwrap();
    // Disallowed actor_type — should be rejected.
    let bad = db.conn().execute(
        "INSERT INTO actor_profile (actor_id, actor_type, regime, last_updated) \
         VALUES ('weird', 'hallucinator', 'default', 0.0)",
        [],
    );
    assert!(
        bad.is_err(),
        "actor_profile should reject actor_type not in the whitelist"
    );
}

#[test]
fn test_schema_v20_compression_artifact_status_whitelist() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // Allowed status.
    db.conn().execute(
        "INSERT INTO compression_artifact (artifact_id, source_span_json, abstraction_operator, \
         reversibility_pointer, namespace, created_at, status) \
         VALUES ('a1', '[]', 'consolidate_v1', 'raw:1-100', 'default', 0.0, 'active')",
        [],
    ).unwrap();
    // Disallowed status.
    let bad = db.conn().execute(
        "INSERT INTO compression_artifact (artifact_id, source_span_json, abstraction_operator, \
         reversibility_pointer, namespace, created_at, status) \
         VALUES ('a2', '[]', 'x', 'y', 'default', 0.0, 'freshly_minted')",
        [],
    );
    assert!(
        bad.is_err(),
        "compression_artifact.status whitelist should reject unknown values"
    );
}

#[test]
fn test_schema_v20_mobility_state_roundtrip() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // Seed a proposition so the FK holds.
    db.conn()
        .execute(
            "INSERT INTO propositions (proposition_id, src, rel_type, dst, namespace, created_at) \
         VALUES ('p1', 'Alice', 'works_at', 'Acme', 'default', 0.0)",
            [],
        )
        .unwrap();
    // Insert a partial mobility state — only write-tier components populated.
    db.conn()
        .execute(
            "INSERT INTO mobility_state (proposition_id, regime, snapshot_ts, \
         support_mass, attack_mass, self_gen_local, modality_consilience, \
         tier_write_components) \
         VALUES ('p1', 'default', 100.0, 2.0, 0.5, 0.0, 1.0, \
         '[\"support_mass\",\"attack_mass\",\"self_gen_local\",\"modality_consilience\"]')",
            [],
        )
        .unwrap();
    // Read it back.
    let (s, a, psi_l, chi): (f64, f64, f64, f64) = db
        .conn()
        .query_row(
            "SELECT support_mass, attack_mass, self_gen_local, modality_consilience \
         FROM mobility_state WHERE proposition_id='p1'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert_eq!(s, 2.0);
    assert_eq!(a, 0.5);
    assert_eq!(psi_l, 0.0);
    assert_eq!(chi, 1.0);
    // Background-tier components should be NULL since we didn't populate them.
    let ancestral: Option<f64> = db
        .conn()
        .query_row(
            "SELECT self_gen_ancestral FROM mobility_state WHERE proposition_id='p1'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(
        ancestral.is_none(),
        "untouched background components should remain NULL"
    );
}
