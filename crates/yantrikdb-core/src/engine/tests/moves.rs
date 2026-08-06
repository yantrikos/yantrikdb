use super::*;

// ──────────────────────────────────────────────────────────────────
// RFC 008 Phase 1 M5b: Cognitive moves substrate.
// Schema V23, append-only event log, normalized edges, corrections as
// events, adversarial candidate/confirmed staging. Locked spec: Saga 19.
// ──────────────────────────────────────────────────────────────────

use crate::engine::moves::{
    adversarial_status, observability, posthoc_outcome, ClaimRef, RecordMoveEventInput,
    SideEffectRef,
};

fn mk_move_input(move_type: &str, inputs: &[&str], outputs: &[&str]) -> RecordMoveEventInput {
    RecordMoveEventInput {
        move_type: move_type.to_string(),
        operator_version: "v1".to_string(),
        context_regime: None,
        observability: observability::OBSERVED.to_string(),
        inputs: inputs
            .iter()
            .enumerate()
            .map(|(i, c)| ClaimRef {
                claim_id: c.to_string(),
                role: "input".to_string(),
                ordinal: i as i64,
            })
            .collect(),
        outputs: outputs
            .iter()
            .enumerate()
            .map(|(i, c)| ClaimRef {
                claim_id: c.to_string(),
                role: "output".to_string(),
                ordinal: i as i64,
            })
            .collect(),
        ..Default::default()
    }
}

#[test]
fn test_m5b_schema_v23_tables_present() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let conn = db.conn();
    for table in [
        "move_events",
        "move_input_edge",
        "move_output_edge",
        "move_side_effect_edge",
        "move_correction_event",
        "move_adversarial_instance",
        "move_type_registry",
        "inference_basis_registry",
        "move_composition_rule",
        "move_type_profile",
    ] {
        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1",
                rusqlite::params![table],
                |_| Ok(true),
            )
            .unwrap_or(false);
        assert!(exists, "V23 table {} must exist", table);
    }
}

#[test]
fn test_m5b_registries_seeded_at_bootstrap() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let conn = db.conn();
    // move_type_registry should have all 13 seed entries (12 from M5a +
    // aggregate_back added in M8 to complete the decomposition axiom pair).
    let mt_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM move_type_registry WHERE status = 'active'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(mt_count, 13, "move_type_registry seed vocabulary count");

    // inference_basis_registry should have all 5 seed entries.
    let ib_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM inference_basis_registry WHERE status = 'active'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        ib_count, 5,
        "inference_basis_registry seed vocabulary count"
    );

    // Spot-check that core vocabulary is present.
    for mt in [
        "analogy",
        "decomposition",
        "source_audit",
        "hypothesis_generation",
    ] {
        let found: bool = conn
            .query_row(
                "SELECT 1 FROM move_type_registry WHERE move_type = ?1",
                rusqlite::params![mt],
                |_| Ok(true),
            )
            .unwrap_or(false);
        assert!(found, "seed vocabulary missing: {}", mt);
    }
}

#[test]
fn test_m5b_record_move_event_basic() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let move_id = db
        .record_move_event(mk_move_input(
            "analogy",
            &["claim_a", "claim_b"],
            &["claim_c"],
        ))
        .unwrap();
    assert!(!move_id.is_empty());

    // Read it back.
    let ev = db.get_move_event(&move_id).unwrap().unwrap();
    assert_eq!(ev.move_type, "analogy");
    assert_eq!(ev.operator_version, "v1");
    assert_eq!(ev.observability, "observed");
    assert_eq!(ev.context_regime, "default");
    assert!(ev.posthoc_outcome.is_none());
    // Default horizon from registry should be applied (analogy = 60s).
    assert_eq!(ev.expected_evaluation_horizon_ms, Some(60_000));

    // Input edges.
    let inputs = db.get_move_inputs(&move_id).unwrap();
    assert_eq!(inputs.len(), 2);
    assert_eq!(inputs[0].claim_id, "claim_a");
    assert_eq!(inputs[1].claim_id, "claim_b");

    // Output edges.
    let outputs = db.get_move_outputs(&move_id).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].claim_id, "claim_c");
}

#[test]
fn test_m5b_record_move_event_with_side_effects() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let mut inp = mk_move_input("quarantine", &["claim_suspect"], &[]);
    inp.side_effects = vec![
        SideEffectRef {
            claim_id: "downstream_a".into(),
            effect_kind: "quarantine".into(),
        },
        SideEffectRef {
            claim_id: "downstream_b".into(),
            effect_kind: "quarantine".into(),
        },
    ];
    let move_id = db.record_move_event(inp).unwrap();
    let side_effects = db.get_move_side_effects(&move_id).unwrap();
    assert_eq!(side_effects.len(), 2);
}

#[test]
fn test_m5b_list_moves_consuming_and_producing() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // Two moves, both consume claim_x; one produces claim_y.
    let m1 = db
        .record_move_event(mk_move_input("analogy", &["claim_x"], &["claim_y"]))
        .unwrap();
    let m2 = db
        .record_move_event(mk_move_input("decomposition", &["claim_x"], &["claim_z"]))
        .unwrap();

    let consumers = db.list_moves_consuming_claim("claim_x", 10).unwrap();
    assert_eq!(consumers.len(), 2);
    let consumer_ids: std::collections::HashSet<_> =
        consumers.iter().map(|m| m.move_id.clone()).collect();
    assert!(consumer_ids.contains(&m1));
    assert!(consumer_ids.contains(&m2));

    let producers = db.list_moves_producing_claim("claim_y", 10).unwrap();
    assert_eq!(producers.len(), 1);
    assert_eq!(producers[0].move_id, m1);
}

#[test]
fn test_m5b_record_move_rejects_invalid_observability_fields() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // observed + inference_confidence set → reject
    let mut inp = mk_move_input("analogy", &["a"], &["b"]);
    inp.inference_confidence = Some(0.7);
    let r = db.record_move_event(inp);
    assert!(
        r.is_err(),
        "should reject inference_confidence on non-inferred"
    );

    // observed + inference_basis non-empty → reject
    let mut inp2 = mk_move_input("analogy", &["a"], &["b"]);
    inp2.inference_basis = Some(vec!["structural_pattern_match".into()]);
    let r2 = db.record_move_event(inp2);
    assert!(r2.is_err(), "should reject inference_basis on non-inferred");
}

#[test]
fn test_m5b_inferred_move_carries_confidence() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let mut inp = mk_move_input("analogy", &["a"], &["b"]);
    inp.observability = observability::INFERRED.to_string();
    inp.inference_confidence = Some(0.8);
    inp.inference_basis = Some(vec!["structural_pattern_match".into()]);
    let move_id = db.record_move_event(inp).unwrap();
    let ev = db.get_move_event(&move_id).unwrap().unwrap();
    assert_eq!(ev.observability, "inferred");
    assert_eq!(ev.inference_confidence, Some(0.8));
    assert!(ev
        .inference_basis_json
        .as_deref()
        .unwrap_or("")
        .contains("structural_pattern_match"));
}

#[test]
fn test_m5b_record_move_outcome_narrow_mutation() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let move_id = db
        .record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();

    db.record_move_outcome(
        &move_id,
        posthoc_outcome::CORROBORATED,
        Some(serde_json::json!({"predictive_gain": 0.3}).to_string()),
    )
    .unwrap();

    let ev = db.get_move_event(&move_id).unwrap().unwrap();
    assert_eq!(ev.posthoc_outcome.as_deref(), Some("corroborated"));
    assert!(ev.posthoc_recorded_at.is_some());
    assert!(ev.yield_json.contains("predictive_gain"));

    // Second call on the same move should reject (already set).
    let second = db.record_move_outcome(&move_id, posthoc_outcome::RETRACTED, None);
    assert!(
        second.is_err(),
        "should reject overwriting an existing posthoc_outcome"
    );
}

#[test]
fn test_m5b_record_move_outcome_rejects_invalid_label() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let move_id = db
        .record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();
    let r = db.record_move_outcome(&move_id, "totally_made_up", None);
    assert!(r.is_err());
}

#[test]
fn test_m5b_correction_never_mutates_original() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let move_id = db
        .record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();

    let correction_id = db
        .submit_move_correction(
            &move_id,
            Some("decomposition".to_string()),
            None,
            None,
            "initial categorization was wrong".to_string(),
            "curator_alice".to_string(),
        )
        .unwrap();
    assert!(!correction_id.is_empty());

    // The original row still has move_type='analogy'.
    let original = db.get_move_event(&move_id).unwrap().unwrap();
    assert_eq!(
        original.move_type, "analogy",
        "original row must not be mutated"
    );

    // Canonical view reflects the correction.
    let canonical = db.get_move_event_canonical(&move_id).unwrap().unwrap();
    assert_eq!(
        canonical.move_type, "decomposition",
        "canonical reflects latest correction"
    );

    // Correction is readable via list_move_corrections.
    let corrections = db.list_move_corrections(&move_id).unwrap();
    assert_eq!(corrections.len(), 1);
    assert_eq!(corrections[0].correction_id, correction_id);
}

#[test]
fn test_m5b_correction_latest_wins() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let move_id = db
        .record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();

    db.submit_move_correction(
        &move_id,
        Some("decomposition".into()),
        None,
        None,
        "first correction".into(),
        "curator_a".into(),
    )
    .unwrap();
    // Need non-zero time delta between corrections for deterministic ordering.
    std::thread::sleep(std::time::Duration::from_millis(10));
    db.submit_move_correction(
        &move_id,
        Some("ladder_up".into()),
        None,
        None,
        "second correction, first was also wrong".into(),
        "curator_b".into(),
    )
    .unwrap();

    let canonical = db.get_move_event_canonical(&move_id).unwrap().unwrap();
    assert_eq!(
        canonical.move_type, "ladder_up",
        "latest correction should win"
    );
}

#[test]
fn test_m5b_correction_rejects_empty_reason_and_no_fields() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let move_id = db
        .record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();

    // No fields changed → reject.
    let r = db.submit_move_correction(
        &move_id,
        None,
        None,
        None,
        "reason".into(),
        "curator".into(),
    );
    assert!(r.is_err());

    // Empty reason → reject.
    let r2 = db.submit_move_correction(
        &move_id,
        Some("decomposition".into()),
        None,
        None,
        "".into(),
        "curator".into(),
    );
    assert!(r2.is_err());
}

#[test]
fn test_m5b_adversarial_candidate_lifecycle() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let move_id = db
        .record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();

    // Create candidate from an automatic signal.
    let instance_id = db
        .create_adversarial_candidate(
            &move_id,
            "contradiction",
            Some("output claim was retracted within 24h".to_string()),
        )
        .unwrap();
    let candidate = db.get_adversarial_instance(&instance_id).unwrap().unwrap();
    assert_eq!(candidate.status, "candidate");
    assert_eq!(candidate.discovered_via, "contradiction");
    assert!(candidate.traced_root_cause.is_some());
    // Governance invariant: candidate must NOT have generalized_lesson.
    assert!(candidate.generalized_lesson.is_none());
    assert!(candidate.lesson_scope_json.is_none());

    // Promote.
    db.promote_adversarial_candidate(
        &instance_id,
        "analogy over cross-domain claims with dim≠ input modalities often produces false corroborations".into(),
        serde_json::json!({"regimes": ["default"], "move_types": ["analogy"]}).to_string(),
        "curator_alice".into(),
    ).unwrap();

    let confirmed = db.get_adversarial_instance(&instance_id).unwrap().unwrap();
    assert_eq!(confirmed.status, "confirmed");
    assert!(confirmed.generalized_lesson.is_some());
    assert!(confirmed.lesson_scope_json.is_some());
    assert_eq!(
        confirmed.curation_actor_id.as_deref(),
        Some("curator_alice")
    );
}

#[test]
fn test_m5b_adversarial_promote_rejects_non_candidate() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let move_id = db
        .record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();
    let instance_id = db
        .create_adversarial_candidate(&move_id, "contradiction", None)
        .unwrap();
    db.promote_adversarial_candidate(&instance_id, "lesson".into(), "{}".into(), "c".into())
        .unwrap();
    // Second promotion attempt → reject.
    let r =
        db.promote_adversarial_candidate(&instance_id, "lesson2".into(), "{}".into(), "c".into());
    assert!(r.is_err(), "cannot promote non-candidate");
}

#[test]
fn test_m5b_adversarial_promote_requires_non_empty_lesson() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let move_id = db
        .record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();
    let instance_id = db
        .create_adversarial_candidate(&move_id, "retraction", None)
        .unwrap();

    // Empty lesson → reject.
    let r = db.promote_adversarial_candidate(&instance_id, "".into(), "{}".into(), "c".into());
    assert!(r.is_err());

    // Empty scope → reject.
    let r2 = db.promote_adversarial_candidate(&instance_id, "lesson".into(), "".into(), "c".into());
    assert!(r2.is_err());
}

#[test]
fn test_m5b_adversarial_reject_candidate() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let move_id = db
        .record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();
    let instance_id = db
        .create_adversarial_candidate(&move_id, "calibration_signal", None)
        .unwrap();

    db.reject_adversarial_candidate(&instance_id, "curator_bob".into())
        .unwrap();
    let rejected = db.get_adversarial_instance(&instance_id).unwrap().unwrap();
    assert_eq!(rejected.status, "rejected");

    // Cannot reject again (no candidate).
    let r = db.reject_adversarial_candidate(&instance_id, "curator_bob".into());
    assert!(r.is_err());
}

#[test]
fn test_m5b_unknown_move_type_warns_but_does_not_reject() {
    // The soft registry policy: unknown move_type logs a warning but the
    // insert still succeeds. This is the "preserve evidence shape" rule.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let inp = RecordMoveEventInput {
        move_type: "entirely_novel_move_type_not_in_registry".into(),
        operator_version: "v0".into(),
        observability: observability::OBSERVED.into(),
        ..Default::default()
    };
    let r = db.record_move_event(inp);
    assert!(r.is_ok(), "unknown move_type must not reject");
    let ev = db.get_move_event(&r.unwrap()).unwrap().unwrap();
    assert_eq!(ev.move_type, "entirely_novel_move_type_not_in_registry");
    assert!(
        ev.expected_evaluation_horizon_ms.is_none(),
        "unregistered move_type has no default horizon"
    );
}

#[test]
fn test_m5b_dependencies_stored_as_json() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let m1 = db
        .record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();
    let mut inp = mk_move_input("decomposition", &["b"], &["c"]);
    inp.dependencies = vec![m1.clone()];
    let m2 = db.record_move_event(inp).unwrap();
    let ev = db.get_move_event(&m2).unwrap().unwrap();
    assert!(
        ev.dependencies_json.contains(&m1),
        "dependencies_json should reference the upstream move_id"
    );
}

#[test]
fn test_m5b_append_only_preserves_original_after_correction() {
    // Reconstruct from the original row directly — corrections go in a
    // separate table. The structural fields on move_events are never
    // touched by submit_move_correction.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let move_id = db
        .record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();
    db.submit_move_correction(
        &move_id,
        Some("decomposition".into()),
        Some("v2".into()),
        None,
        "reason".into(),
        "curator".into(),
    )
    .unwrap();

    let raw: (String, String) = db
        .conn()
        .query_row(
            "SELECT move_type, operator_version FROM move_events WHERE move_id = ?1",
            rusqlite::params![move_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(
        raw.0, "analogy",
        "move_events row must keep original move_type"
    );
    assert_eq!(
        raw.1, "v1",
        "move_events row must keep original operator_version"
    );
}

#[test]
fn test_m5b_edge_tables_have_fk_integrity() {
    // FK constraint: move_input_edge.move_id REFERENCES move_events(move_id).
    // Attempting to insert an edge for a non-existent move should fail.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let conn = db.conn();
    // Ensure foreign keys are enabled (SQLite defaults to off per-connection).
    conn.execute("PRAGMA foreign_keys = ON", []).unwrap();
    let r = conn.execute(
        "INSERT INTO move_input_edge (move_id, claim_id, input_role, ordinal) \
         VALUES ('nonexistent_move', 'claim_x', 'input', 0)",
        [],
    );
    assert!(r.is_err(), "FK constraint should reject orphan edge");
}

#[test]
fn test_m5b_adversarial_discovered_via_whitelist() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let move_id = db
        .record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();
    let r = db.create_adversarial_candidate(&move_id, "invalid_source", None);
    assert!(r.is_err(), "discovered_via CHECK enforces the whitelist");
}

#[test]
fn test_m5b_seed_registries_idempotent() {
    // Calling seed_move_registries a second time should not duplicate or error.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let before: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM move_type_registry", [], |r| r.get(0))
        .unwrap();
    db.seed_move_registries().unwrap();
    let after: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM move_type_registry", [], |r| r.get(0))
        .unwrap();
    assert_eq!(before, after, "INSERT OR IGNORE should not add duplicates");
}

// ──────────────────────────────────────────────────────────────────
// RFC 008 Phase 1 M6: Background-tier mobility components (τ, λ, ψ_a).
// ──────────────────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────────────
// RFC 008 Phase 1 M8: Axiomatic composition rules.
// ──────────────────────────────────────────────────────────────────

#[test]
fn test_m8_axiom_registry_has_core_entries() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let axioms = db.composition_axioms();
    assert!(
        axioms.len() >= 3,
        "axiom registry should have at least 3 entries"
    );
    let names: Vec<&str> = axioms.iter().map(|a| a.name).collect();
    assert!(names.contains(&"decompose_aggregate_identity"));
    assert!(names.contains(&"negate_analogize_non_commutative"));
    assert!(names.contains(&"source_audit_requires_external_ancestry"));
}

#[test]
fn test_m8_check_composition_non_commutative_match() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let left = db
        .record_move_event(mk_move_input("negate_and_test", &["a"], &["b"]))
        .unwrap();
    let right = db
        .record_move_event(mk_move_input("analogy", &["b"], &["c"]))
        .unwrap();
    let matches = db.check_composition_axioms(&left, &right).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].name, "negate_analogize_non_commutative");
}

#[test]
fn test_m8_check_composition_approx_identity_match() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let d = db
        .record_move_event(mk_move_input("decomposition", &["a"], &["b"]))
        .unwrap();
    let ag = db
        .record_move_event(mk_move_input("aggregate_back", &["b"], &["c"]))
        .unwrap();
    let matches = db.check_composition_axioms(&d, &ag).unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].name, "decompose_aggregate_identity");
}

#[test]
fn test_m8_check_composition_no_match() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let m1 = db
        .record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();
    let m2 = db
        .record_move_event(mk_move_input("ladder_up", &["b"], &["c"]))
        .unwrap();
    let matches = db.check_composition_axioms(&m1, &m2).unwrap();
    assert!(
        matches.is_empty(),
        "unrelated move pair should match no axiom"
    );
}

#[test]
fn test_m8_source_audit_precondition_violation() {
    // source_audit requires self_gen_ancestral < 1.0 (external source must exist).
    // Seed a proposition with its single claim AND a producing move where
    // the input is self-generated, pushing ψ_a toward 1.0.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    seed_proposition(&db, "p_self_audit");
    // Seed final claim of the proposition (self-generated).
    let conn = db.conn();
    for (cid, self_gen, prop) in [
        ("self_input", 1, None),
        ("final_out", 1, Some("p_self_audit")),
    ] {
        conn.execute(
            "INSERT INTO claims (claim_id, src, dst, rel_type, created_at, \
             extractor, polarity, namespace, proposition_id, regime_tag, \
             self_generated, source_lineage, modality_signal, weight) \
             VALUES (?1, ?2, ?3, 'rel', 0.0, 'ext', 1, 'default', ?4, 'default', \
             ?5, '[]', 'text', 1.0)",
            rusqlite::params![
                cid,
                format!("src_{}", cid),
                format!("dst_{}", cid),
                prop,
                self_gen
            ],
        )
        .unwrap();
    }
    drop(conn);
    // A move producing final_out with self_input as its upstream.
    db.record_move_event(mk_move_input(
        "hypothesis_generation",
        &["self_input"],
        &["final_out"],
    ))
    .unwrap();
    // Run write + background tiers so ψ_a is populated.
    db.compute_write_tier_mobility("p_self_audit", "default")
        .unwrap();
    db.compute_background_mobility("p_self_audit", "default")
        .unwrap();

    let violations = db
        .check_move_preconditions("source_audit", "p_self_audit", "default")
        .unwrap();
    assert!(
        !violations.is_empty(),
        "source_audit on ψ_a=1.0 should violate"
    );
    assert_eq!(
        violations[0].axiom_name,
        "source_audit_requires_external_ancestry"
    );
    assert!(violations[0].observation.contains("self_gen_ancestral"));
}

#[test]
fn test_m8_source_audit_passes_with_external_ancestry() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    seed_proposition(&db, "p_ext_audit");
    let conn = db.conn();
    // Mix: one self-gen, one external in the ancestry.
    for (cid, self_gen, prop) in [
        ("ext_src", 0, None),
        ("self_src", 1, None),
        ("final_out", 0, Some("p_ext_audit")),
    ] {
        conn.execute(
            "INSERT INTO claims (claim_id, src, dst, rel_type, created_at, \
             extractor, polarity, namespace, proposition_id, regime_tag, \
             self_generated, source_lineage, modality_signal, weight) \
             VALUES (?1, ?2, ?3, 'rel', 0.0, 'ext', 1, 'default', ?4, 'default', \
             ?5, '[]', 'text', 1.0)",
            rusqlite::params![
                cid,
                format!("src_{}", cid),
                format!("dst_{}", cid),
                prop,
                self_gen
            ],
        )
        .unwrap();
    }
    drop(conn);
    db.record_move_event(mk_move_input(
        "decomposition",
        &["ext_src", "self_src"],
        &["final_out"],
    ))
    .unwrap();
    db.compute_write_tier_mobility("p_ext_audit", "default")
        .unwrap();
    db.compute_background_mobility("p_ext_audit", "default")
        .unwrap();

    let state = db
        .get_mobility_state("p_ext_audit", "default")
        .unwrap()
        .unwrap();
    // ψ_a should be 0.5 (1 self-gen out of 2 ancestral claims).
    assert!((state.self_gen_ancestral.unwrap() - 0.5).abs() < 1e-9);

    let violations = db
        .check_move_preconditions("source_audit", "p_ext_audit", "default")
        .unwrap();
    assert!(
        violations.is_empty(),
        "source_audit should be fine when external ancestry exists"
    );
}

#[test]
fn test_m8_compression_requires_support() {
    // Seed a proposition with only negative polarity claims → σ = 0.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    seed_proposition(&db, "p_no_support");
    seed_contest_claim(
        &db,
        "p_no_support",
        "c1",
        "ext_a",
        -1,
        "[\"s\"]",
        None,
        "default",
        None,
        None,
    );
    db.compute_write_tier_mobility("p_no_support", "default")
        .unwrap();

    let violations = db
        .check_move_preconditions("compression", "p_no_support", "default")
        .unwrap();
    assert!(
        !violations.is_empty(),
        "compression should violate on zero support_mass"
    );
    assert_eq!(violations[0].axiom_name, "compression_requires_support");
}

#[test]
fn test_m8_hypothesis_generation_blocked_on_present_tense_conflict() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    seed_proposition(&db, "p_contested");
    // Opposite-polarity claims with overlapping validity intervals → PRESENT_TENSE_CONFLICT.
    seed_contest_claim(
        &db,
        "p_contested",
        "c_sup",
        "ext_a",
        1,
        "[\"s\"]",
        None,
        "default",
        Some(0.0),
        Some(20.0),
    );
    seed_contest_claim(
        &db,
        "p_contested",
        "c_att",
        "ext_b",
        -1,
        "[\"s\"]",
        None,
        "default",
        Some(10.0),
        Some(30.0),
    );
    db.compute_write_tier_mobility("p_contested", "default")
        .unwrap();
    db.compute_contest_state("p_contested", "default").unwrap();

    let violations = db
        .check_move_preconditions("hypothesis_generation", "p_contested", "default")
        .unwrap();
    assert!(
        !violations.is_empty(),
        "hypothesis_generation should be blocked by PRESENT_TENSE_CONFLICT"
    );
    assert_eq!(
        violations[0].axiom_name,
        "hypothesis_generation_skips_present_tense_conflict"
    );
}

#[test]
fn test_m8_preconditions_ignore_unrelated_move_types() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    seed_proposition(&db, "p_any");
    seed_contest_claim(
        &db, "p_any", "c1", "ext_a", 1, "[\"s\"]", None, "default", None, None,
    );
    db.compute_write_tier_mobility("p_any", "default").unwrap();

    // analogy has no precondition axiom → no violations.
    let violations = db
        .check_move_preconditions("analogy", "p_any", "default")
        .unwrap();
    assert!(violations.is_empty());
}

// ──────────────────────────────────────────────────────────────────
// RFC 008 Phase 1 M9: move_type_profile derivation.
// ──────────────────────────────────────────────────────────────────

#[test]
fn test_m9_profile_counts_uses_and_resolutions() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // Record three moves of the same type; resolve two (corroborated, retracted).
    let m1 = db
        .record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();
    let m2 = db
        .record_move_event(mk_move_input("analogy", &["c"], &["d"]))
        .unwrap();
    db.record_move_event(mk_move_input("analogy", &["e"], &["f"]))
        .unwrap();
    db.record_move_outcome(&m1, "corroborated", None).unwrap();
    db.record_move_outcome(&m2, "retracted", None).unwrap();

    let profile = db
        .recompute_move_type_profile("analogy", "v1", "default")
        .unwrap();
    assert_eq!(profile.uses_count, 3);
    assert_eq!(profile.corroborated_count, 1);
    assert_eq!(profile.retracted_count, 1);
    assert_eq!(profile.harmful_side_effect_count, 0);
    // resolved = 2 (two have posthoc_outcome set). The third is within its
    // horizon (default 60s) so not counted as past-horizon.
    assert_eq!(profile.resolved_count, 2);
    // Contradiction rate = (retracted + harmful) / resolved = 1/2 = 0.5.
    assert!((profile.contradiction_introduction_rate.unwrap() - 0.5).abs() < 1e-9);
}

#[test]
fn test_m9_profile_round_trip_via_get() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();
    let computed = db
        .recompute_move_type_profile("analogy", "v1", "default")
        .unwrap();
    let read = db
        .get_move_type_profile("analogy", "v1", "default")
        .unwrap()
        .unwrap();
    assert_eq!(read.uses_count, computed.uses_count);
    assert_eq!(read.resolved_count, computed.resolved_count);
}

#[test]
fn test_m9_recompute_all_keys_present() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();
    db.record_move_event(mk_move_input("decomposition", &["c"], &["d"]))
        .unwrap();
    let count = db.recompute_all_move_type_profiles().unwrap();
    assert_eq!(count, 2, "two distinct (type, version, regime) triples");
    assert!(db
        .get_move_type_profile("analogy", "v1", "default")
        .unwrap()
        .is_some());
    assert!(db
        .get_move_type_profile("decomposition", "v1", "default")
        .unwrap()
        .is_some());
}

#[test]
fn test_m9_profile_missing_returns_none() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let p = db
        .get_move_type_profile("analogy", "v1", "default")
        .unwrap();
    assert!(p.is_none());
}

#[test]
fn test_m9_contradiction_rate_none_when_no_resolutions() {
    // All moves still pending within horizon → resolved = 0 → rate is None.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();
    let profile = db
        .recompute_move_type_profile("analogy", "v1", "default")
        .unwrap();
    assert_eq!(profile.resolved_count, 0);
    assert!(profile.contradiction_introduction_rate.is_none());
}

// ──────────────────────────────────────────────────────────────────
// RFC 008 Phase 1 M10: auto-adversarial-candidate generation.
// ──────────────────────────────────────────────────────────────────

#[test]
fn test_m10_retraction_auto_files_candidate() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let move_id = db
        .record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();

    // Before outcome — no candidate.
    let before = db.list_adversarial_for_move(&move_id).unwrap();
    assert!(before.is_empty());

    db.record_move_outcome(&move_id, "retracted", None).unwrap();

    let after = db.list_adversarial_for_move(&move_id).unwrap();
    assert_eq!(after.len(), 1, "retraction must auto-file a candidate");
    assert_eq!(after[0].status, "candidate");
    assert_eq!(after[0].discovered_via, "retraction");
    assert!(after[0].traced_root_cause.is_some());
    // Governance: no generalized_lesson on auto-candidates.
    assert!(after[0].generalized_lesson.is_none());
    assert!(after[0].lesson_scope_json.is_none());
}

#[test]
fn test_m10_harmful_side_effect_auto_files_candidate() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let move_id = db
        .record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();
    db.record_move_outcome(&move_id, "harmful_side_effect", None)
        .unwrap();
    let after = db.list_adversarial_for_move(&move_id).unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].discovered_via, "calibration_signal");
}

#[test]
fn test_m10_corroborated_outcome_does_not_file_candidate() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let move_id = db
        .record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();
    db.record_move_outcome(&move_id, "corroborated", None)
        .unwrap();
    let after = db.list_adversarial_for_move(&move_id).unwrap();
    assert!(
        after.is_empty(),
        "corroborated outcome must not file an adversarial candidate"
    );
}

#[test]
fn test_m10_contest_flag_transition_auto_files_for_producing_move() {
    // Record a move producing a claim. Then ingest a conflicting claim to
    // the same proposition so SAME_SOURCE_CONFLICT flips on — the
    // auto-file hook should create an adversarial candidate for the move.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    seed_proposition(&db, "p_flag_auto");
    // First claim of the proposition.
    seed_contest_claim(
        &db,
        "p_flag_auto",
        "claim_out",
        "ext_a",
        1,
        "[\"s\"]",
        None,
        "default",
        None,
        None,
    );
    // Move that produced claim_out.
    let move_id = db
        .record_move_event(mk_move_input(
            "hypothesis_generation",
            &["evidence"],
            &["claim_out"],
        ))
        .unwrap();
    // Initial write-tier + contest recompute: no conflict yet.
    db.compute_write_tier_mobility("p_flag_auto", "default")
        .unwrap();
    db.compute_contest_state("p_flag_auto", "default").unwrap();
    assert!(db.list_adversarial_for_move(&move_id).unwrap().is_empty());

    // Add an opposite-polarity claim with identical source_lineage → SAME_SOURCE_CONFLICT.
    seed_contest_claim(
        &db,
        "p_flag_auto",
        "conflicting",
        "ext_b",
        -1,
        "[\"s\"]",
        None,
        "default",
        None,
        None,
    );
    db.compute_write_tier_mobility("p_flag_auto", "default")
        .unwrap();
    db.compute_contest_state("p_flag_auto", "default").unwrap();

    let after = db.list_adversarial_for_move(&move_id).unwrap();
    assert!(
        !after.is_empty(),
        "SAME_SOURCE_CONFLICT should auto-file adversarial candidate for producing move"
    );
    assert_eq!(after[0].discovered_via, "contradiction");
    assert_eq!(after[0].status, "candidate");
}

#[test]
fn test_m10_contest_flag_transition_dedups_on_repeat_recompute() {
    // Same setup as above but recompute contest multiple times — the
    // (move_id, discovered_via) dedup guard should prevent duplicate rows.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    seed_proposition(&db, "p_dedup");
    seed_contest_claim(
        &db,
        "p_dedup",
        "claim_out",
        "ext_a",
        1,
        "[\"s\"]",
        None,
        "default",
        None,
        None,
    );
    let move_id = db
        .record_move_event(mk_move_input("analogy", &["e"], &["claim_out"]))
        .unwrap();
    db.compute_write_tier_mobility("p_dedup", "default")
        .unwrap();
    db.compute_contest_state("p_dedup", "default").unwrap();
    // Set up the conflict.
    seed_contest_claim(
        &db, "p_dedup", "conflict", "ext_b", -1, "[\"s\"]", None, "default", None, None,
    );
    db.compute_write_tier_mobility("p_dedup", "default")
        .unwrap();
    db.compute_contest_state("p_dedup", "default").unwrap();

    // Multiple recomputes must not create duplicates.
    db.compute_contest_state("p_dedup", "default").unwrap();
    db.compute_contest_state("p_dedup", "default").unwrap();
    db.compute_contest_state("p_dedup", "default").unwrap();

    let candidates = db.list_adversarial_for_move(&move_id).unwrap();
    let contradiction_candidates: Vec<_> = candidates
        .iter()
        .filter(|c| c.discovered_via == "contradiction")
        .collect();
    assert_eq!(
        contradiction_candidates.len(),
        1,
        "repeat contest recompute must not duplicate contradiction candidates"
    );
}

#[test]
fn test_m10_m9_feedback_loop_retraction_updates_profile() {
    // End-to-end: record two moves of the same type, retract one, recompute
    // the profile, verify the retraction count increments. This is the
    // full M5b → M10 → M9 feedback loop.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let m1 = db
        .record_move_event(mk_move_input("analogy", &["a"], &["b"]))
        .unwrap();
    db.record_move_event(mk_move_input("analogy", &["c"], &["d"]))
        .unwrap();
    db.record_move_outcome(&m1, "retracted", None).unwrap();

    // M10 should have filed an adversarial candidate.
    let candidates = db.list_adversarial_for_move(&m1).unwrap();
    assert_eq!(candidates.len(), 1);

    // M9 profile recompute should pick up the retraction.
    let profile = db
        .recompute_move_type_profile("analogy", "v1", "default")
        .unwrap();
    assert_eq!(profile.retracted_count, 1);
    assert_eq!(profile.uses_count, 2);
}

#[test]
fn test_m6_temporal_coherence_stable_polarity_is_one() {
    // All supports on the same proposition → no flips → τ = 1.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    seed_proposition(&db, "p_coh");
    seed_contest_claim(
        &db, "p_coh", "c1", "ext_a", 1, "[\"s\"]", None, "default", None, None,
    );
    seed_contest_claim(
        &db, "p_coh", "c2", "ext_b", 1, "[\"s\"]", None, "default", None, None,
    );
    seed_contest_claim(
        &db, "p_coh", "c3", "ext_c", 1, "[\"s\"]", None, "default", None, None,
    );
    db.compute_write_tier_mobility("p_coh", "default").unwrap();
    let state = db
        .compute_background_mobility("p_coh", "default")
        .unwrap()
        .unwrap();
    assert_eq!(state.temporal_coherence, Some(1.0));
}

#[test]
fn test_m6_temporal_coherence_flips_reduce_score() {
    // Alternating polarities: support, attack, support → two flips out of
    // two transitions → τ = 1 - 2/2 = 0.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    seed_proposition(&db, "p_flip");
    // Use explicit created_at ordering: set created_at monotonically to
    // ensure the sort order is what we expect.
    let conn = db.conn();
    let triples = [
        ("c_p1", "ext_a", 1, 1.0),
        ("c_a1", "ext_b", -1, 2.0),
        ("c_p2", "ext_c", 1, 3.0),
    ];
    for (cid, ext, pol, ts) in triples {
        conn.execute(
            "INSERT INTO claims (claim_id, src, dst, rel_type, created_at, \
             extractor, polarity, namespace, proposition_id, regime_tag, \
             self_generated, source_lineage, modality_signal, weight) \
             VALUES (?1, 'src_p_flip', 'dst_p_flip', 'rel_p_flip', ?2, ?3, ?4, \
             'default', 'p_flip', 'default', 0, '[\"s\"]', 'text', 1.0)",
            rusqlite::params![cid, ts, ext, pol],
        )
        .unwrap();
    }
    drop(conn);
    db.compute_write_tier_mobility("p_flip", "default").unwrap();
    let state = db
        .compute_background_mobility("p_flip", "default")
        .unwrap()
        .unwrap();
    let tau = state.temporal_coherence.unwrap();
    assert!(
        (tau - 0.0).abs() < 1e-9,
        "expected τ=0 for maximally flipping polarity, got {}",
        tau
    );
}

#[test]
fn test_m6_temporal_coherence_single_claim_is_one_by_convention() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    seed_proposition(&db, "p_solo");
    seed_contest_claim(
        &db, "p_solo", "c1", "ext_a", 1, "[\"s\"]", None, "default", None, None,
    );
    db.compute_write_tier_mobility("p_solo", "default").unwrap();
    let state = db
        .compute_background_mobility("p_solo", "default")
        .unwrap()
        .unwrap();
    // Convention: too few claims to judge → default to 1.0 coherence.
    assert_eq!(state.temporal_coherence, Some(1.0));
}

#[test]
fn test_m6_load_bearingness_counts_downstream_moves() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    seed_proposition(&db, "p_load");
    seed_contest_claim(
        &db,
        "p_load",
        "upstream_a",
        "ext_a",
        1,
        "[\"s\"]",
        None,
        "default",
        None,
        None,
    );
    seed_contest_claim(
        &db,
        "p_load",
        "upstream_b",
        "ext_b",
        1,
        "[\"s\"]",
        None,
        "default",
        None,
        None,
    );
    db.compute_write_tier_mobility("p_load", "default").unwrap();

    // No moves yet → λ = 0.
    let before = db
        .compute_background_mobility("p_load", "default")
        .unwrap()
        .unwrap();
    assert_eq!(before.load_bearingness, Some(0.0));

    // Record two moves that consume the upstream claims.
    db.record_move_event(mk_move_input("analogy", &["upstream_a"], &["derived_1"]))
        .unwrap();
    db.record_move_event(mk_move_input(
        "decomposition",
        &["upstream_a", "upstream_b"],
        &["derived_2"],
    ))
    .unwrap();
    // A third move that consumes an unrelated claim — should NOT count.
    db.record_move_event(mk_move_input(
        "analogy",
        &["unrelated_claim"],
        &["derived_3"],
    ))
    .unwrap();

    let after = db
        .compute_background_mobility("p_load", "default")
        .unwrap()
        .unwrap();
    assert_eq!(
        after.load_bearingness,
        Some(2.0),
        "expected 2 downstream moves consuming this proposition's claims"
    );
}

#[test]
fn test_m6_self_gen_ancestral_traces_backward() {
    // Build a two-step ancestry:
    //   evidence_a (self_gen=true), evidence_b (self_gen=false)
    //     → move_m1 → intermediate_c
    //     → move_m2 (inputs: intermediate_c) → final_d
    // Proposition P owns final_d. ψ_a should reflect the ancestry.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // Seed proposition and its claim.
    seed_proposition(&db, "p_anc");
    // Add all the claims we'll reference. final_d belongs to p_anc;
    // the others are referenced by move edges only.
    let conn = db.conn();
    for (cid, self_gen) in [
        ("evidence_a", 1),
        ("evidence_b", 0),
        ("intermediate_c", 0),
        ("final_d", 0),
    ] {
        let prop = if cid == "final_d" {
            Some("p_anc")
        } else {
            None
        };
        conn.execute(
            "INSERT INTO claims (claim_id, src, dst, rel_type, created_at, \
             extractor, polarity, namespace, proposition_id, regime_tag, \
             self_generated, source_lineage, modality_signal, weight) \
             VALUES (?1, ?2, ?3, 'rel_anc', 0.0, 'ext', 1, 'default', ?4, \
             'default', ?5, '[]', 'text', 1.0)",
            rusqlite::params![
                cid,
                format!("src_{}", cid),
                format!("dst_{}", cid),
                prop,
                self_gen
            ],
        )
        .unwrap();
    }
    drop(conn);
    // m1: evidence_a + evidence_b → intermediate_c
    let mut inp1 = mk_move_input(
        "decomposition",
        &["evidence_a", "evidence_b"],
        &["intermediate_c"],
    );
    inp1.operator_version = "v1".to_string();
    db.record_move_event(inp1).unwrap();
    // m2: intermediate_c → final_d
    let mut inp2 = mk_move_input("ladder_up", &["intermediate_c"], &["final_d"]);
    inp2.operator_version = "v1".to_string();
    db.record_move_event(inp2).unwrap();

    db.compute_write_tier_mobility("p_anc", "default").unwrap();
    let state = db
        .compute_background_mobility("p_anc", "default")
        .unwrap()
        .unwrap();
    // BFS from final_d (depth=2): layer 1 finds intermediate_c, layer 2
    // finds evidence_a and evidence_b. Ancestry = {intermediate_c, evidence_a, evidence_b}.
    // self_generated: evidence_a = true. → ψ_a = 1/3.
    let psi_a = state.self_gen_ancestral.unwrap();
    assert!(
        (psi_a - 1.0 / 3.0).abs() < 1e-9,
        "expected ψ_a = 1/3, got {}",
        psi_a
    );
}

#[test]
fn test_m6_self_gen_ancestral_no_moves_is_zero() {
    // Proposition with claims but no upstream moves → no ancestry → ψ_a = 0.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    seed_proposition(&db, "p_leaf");
    seed_contest_claim(
        &db, "p_leaf", "c1", "ext_a", 1, "[\"s\"]", None, "default", None, None,
    );
    db.compute_write_tier_mobility("p_leaf", "default").unwrap();
    let state = db
        .compute_background_mobility("p_leaf", "default")
        .unwrap()
        .unwrap();
    assert_eq!(state.self_gen_ancestral, Some(0.0));
}

#[test]
fn test_m6_compute_background_missing_returns_none() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // No mobility_state row yet → returns None, does not insert one.
    let res = db.compute_background_mobility("nope", "default").unwrap();
    assert!(res.is_none());
}

#[test]
fn test_m6_background_idempotent() {
    // Calling twice with unchanged data produces the same values.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    seed_proposition(&db, "p_idem_bg");
    seed_contest_claim(
        &db,
        "p_idem_bg",
        "c1",
        "ext_a",
        1,
        "[\"s\"]",
        None,
        "default",
        None,
        None,
    );
    db.compute_write_tier_mobility("p_idem_bg", "default")
        .unwrap();
    let first = db
        .compute_background_mobility("p_idem_bg", "default")
        .unwrap()
        .unwrap();
    let second = db
        .compute_background_mobility("p_idem_bg", "default")
        .unwrap()
        .unwrap();
    assert_eq!(first.temporal_coherence, second.temporal_coherence);
    assert_eq!(first.load_bearingness, second.load_bearingness);
    assert_eq!(first.self_gen_ancestral, second.self_gen_ancestral);
}

#[test]
fn test_m6_background_batch_scan() {
    // Seed three propositions, compute write-tier for all, then run the
    // batch recompute and verify all three get their background fields
    // populated.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    for pid in ["p_batch_1", "p_batch_2", "p_batch_3"] {
        seed_proposition(&db, pid);
        seed_contest_claim(
            &db,
            pid,
            &format!("c_{}", pid),
            "ext_a",
            1,
            "[\"s\"]",
            None,
            "default",
            None,
            None,
        );
        db.compute_write_tier_mobility(pid, "default").unwrap();
    }
    // All three should have NULL background fields pre-scan.
    let pending_before: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM mobility_state WHERE temporal_coherence IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(pending_before, 3);

    let processed = db.recompute_background_mobility_batch(10).unwrap();
    assert_eq!(processed, 3);

    let pending_after: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM mobility_state WHERE temporal_coherence IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(pending_after, 0);
}

#[test]
fn test_m6_background_marks_tier_components() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    seed_proposition(&db, "p_tier");
    seed_contest_claim(
        &db, "p_tier", "c1", "ext_a", 1, "[\"s\"]", None, "default", None, None,
    );
    db.compute_write_tier_mobility("p_tier", "default").unwrap();
    let state = db
        .compute_background_mobility("p_tier", "default")
        .unwrap()
        .unwrap();
    for expected in [
        "temporal_coherence",
        "load_bearingness",
        "self_gen_ancestral",
    ] {
        assert!(
            state.tier_bg_components.iter().any(|c| c == expected),
            "tier_bg_components should include {}, got {:?}",
            expected,
            state.tier_bg_components
        );
    }
    // Repeat call shouldn't duplicate entries.
    let state2 = db
        .compute_background_mobility("p_tier", "default")
        .unwrap()
        .unwrap();
    let tc_count = state2
        .tier_bg_components
        .iter()
        .filter(|c| *c == "temporal_coherence")
        .count();
    assert_eq!(
        tc_count, 1,
        "repeated calls must not duplicate tier_bg_components entries"
    );
}

#[test]
fn test_m5b_full_lifecycle_observed_to_retracted_with_adversarial() {
    // End-to-end: record an observed move, enrich with posthoc retraction,
    // file an adversarial candidate, curator promotes to confirmed, verify
    // the canonical view and all pieces are readable.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let move_id = db
        .record_move_event(mk_move_input(
            "hypothesis_generation",
            &["evidence_a", "evidence_b"],
            &["hypothesis_h"],
        ))
        .unwrap();

    // Something goes wrong downstream — retract. Per M10, this auto-files
    // an adversarial candidate with discovered_via='retraction'.
    db.record_move_outcome(
        &move_id,
        posthoc_outcome::RETRACTED,
        Some(
            serde_json::json!({"retraction_cause": "contradiction with high-weight claim"})
                .to_string(),
        ),
    )
    .unwrap();

    // The M10 auto-candidate should exist.
    let auto_candidates = db.list_adversarial_for_move(&move_id).unwrap();
    assert_eq!(auto_candidates.len(), 1);
    assert_eq!(auto_candidates[0].status, adversarial_status::CANDIDATE);
    assert_eq!(auto_candidates[0].discovered_via, "retraction");

    // Curator reviews the auto-candidate and promotes with a generalized lesson.
    db.promote_adversarial_candidate(
        &auto_candidates[0].instance_id,
        "hypothesis_generation from ≤2 evidence sources is prone to retraction".into(),
        serde_json::json!({
            "regimes": ["default"],
            "move_types": ["hypothesis_generation"],
            "input_signatures": {"min_inputs": 2}
        })
        .to_string(),
        "curator_dana".into(),
    )
    .unwrap();

    // Verify everything.
    let ev = db.get_move_event(&move_id).unwrap().unwrap();
    assert_eq!(ev.posthoc_outcome.as_deref(), Some("retracted"));
    let instance = db
        .get_adversarial_instance(&auto_candidates[0].instance_id)
        .unwrap()
        .unwrap();
    assert_eq!(instance.status, adversarial_status::CONFIRMED);
    let instances = db.list_adversarial_for_move(&move_id).unwrap();
    assert_eq!(instances.len(), 1);
}
