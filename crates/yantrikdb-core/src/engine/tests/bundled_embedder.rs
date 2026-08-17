use super::*;

// ── Saga task 20: bundled-embedder auto-attach ──
//
// These tests pin the contract that, on default builds (feature
// `bundled-embedder` is on), `record_text()` and `recall_text()` work
// out of the box — no `set_embedder()` call required. The
// architectural decision (memory rid 019e0686) was that the engine
// ships a default embedder so the user-facing API contract isn't
// "engine plus required side-installs."

#[cfg(feature = "bundled-embedder")]
#[test]
fn bundled_embedder_auto_attaches_on_default_dim() {
    // dim=64 matches BUNDLED_EMBEDDER_DIM (potion-base-2M), so the
    // auto-attach fires. Updated for Slice B (saga task 20, 2026-05-08):
    // bundled embedder switched from hash-trick dim=384 to potion-2M dim=64.
    use crate::embedder::BUNDLED_EMBEDDER_DIM;
    let db = YantrikDB::new(":memory:", BUNDLED_EMBEDDER_DIM).unwrap();
    assert!(
        db.has_embedder(),
        "default-build YantrikDB::new with bundled dim must auto-attach BundledEmbedder"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn with_default_constructor_attaches_bundled_embedder() {
    // YantrikDB::with_default(path) is the constructor that lets callers
    // stay agnostic to the bundled model's dimension. Stays in sync if
    // a future Slice C swaps the bundle to a different-dim variant.
    let db = YantrikDB::with_default(":memory:").unwrap();
    assert!(
        db.has_embedder(),
        "with_default must auto-attach BundledEmbedder"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn bundled_embedder_does_not_attach_on_mismatched_dim() {
    // dim=384 != BUNDLED_EMBEDDER_DIM (64). Auto-attach is silently
    // skipped — caller must set their own embedder. The skip avoids
    // silent dim-mismatch corruption when a caller is intentionally
    // running with a non-default dim (e.g. for an external MiniLM).
    let db = YantrikDB::new(":memory:", 384).unwrap();
    assert!(
        !db.has_embedder(),
        "dim mismatch should NOT auto-attach (avoids silent corruption)"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn bundled_embedder_record_text_round_trip() {
    // The integration shape that actually matters: pip install yantrikdb;
    // YantrikDB::with_default(...); record_text(...); recall_text(...).
    // All works without configuration on default builds.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let _rid = db
        .record_text(
            "Alice met Acme yesterday",
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
        .expect("record_text should work without explicit set_embedder");

    let results = db.recall_text("Alice", 5).expect("recall_text should work");
    assert!(!results.is_empty(), "recall finds the recorded memory");
    assert!(
        results[0].text.contains("Alice"),
        "potion-2M finds the recorded memory; got: {:?}",
        results[0].text
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn record_text_strips_leaked_tool_call_artifact_end_to_end() {
    // Task 29 (Ingest Integrity) wiring regression. Proves the sanitizer is
    // actually invoked on the `record_text` path — not just unit-correct —
    // by storing the exact corpus-signature artifact and asserting the
    // persisted text is clean. The leaked tail must never reach storage or
    // the embedding.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let mangled = "Decision: adopt keyset cursors for list_records.</text>\n\
         <parameter name=\"memory_type\">episodic";
    let rid = db
        .record_text(
            mangled,
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
        .expect("record_text stores sanitized text");

    let results = db.recall_text("keyset cursors list_records", 5).unwrap();
    let hit = results
        .iter()
        .find(|r| r.rid == rid)
        .expect("the recorded memory is retrievable");
    assert!(
        hit.text.contains("keyset cursors"),
        "real content is preserved; got: {:?}",
        hit.text
    );
    assert!(
        !hit.text.contains("</text>"),
        "the leaked closing tag must be stripped; got: {:?}",
        hit.text
    );
    assert!(
        !hit.text.contains("<parameter name="),
        "the leaked parameter fragment must be stripped; got: {:?}",
        hit.text
    );
    assert_eq!(
        hit.text, "Decision: adopt keyset cursors for list_records.",
        "stored text is exactly the cleaned content"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn repair_tool_call_artifacts_cleans_legacy_corpus() {
    // Task 30 end-to-end. Simulates a row corrupted BEFORE the write-time
    // sanitizer existed, then proves the repair migration detects it
    // (dry-run, no mutation), cleans + re-embeds it (apply), preserves the
    // original for recovery, is idempotent, and leaves recall working.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let clean = "Postgres was chosen for the metadata store";
    let rid = db
        .record_text(
            clean,
            "semantic",
            0.6,
            0.0,
            604800.0,
            &empty_meta(),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    // Inject a legacy artifact directly into storage, bypassing record_text
    // (which would now sanitize it). The :memory: db has no encryption, so
    // the stored text is plaintext.
    let dirty = "Postgres was chosen for the metadata store</text>\n\
                 <parameter name=\"memory_type\">semantic";
    {
        let conn = db.conn();
        conn.execute(
            "UPDATE memories SET text = ?1 WHERE rid = ?2",
            rusqlite::params![dirty, rid],
        )
        .unwrap();
    }

    // Dry run detects but does not mutate.
    let dry = db.repair_tool_call_artifacts(true).unwrap();
    assert!(dry.dry_run);
    assert_eq!(dry.artifacts_found, 1);
    assert_eq!(dry.repaired, 0);
    assert!(dry.stripped_bytes > 0);
    {
        let conn = db.conn();
        let t: String = conn
            .query_row(
                "SELECT text FROM memories WHERE rid = ?1",
                rusqlite::params![rid],
                |r| r.get(0),
            )
            .unwrap();
        assert!(t.contains("</text>"), "dry run must NOT mutate");
    }

    // Apply: clean + re-embed + update.
    let applied = db.repair_tool_call_artifacts(false).unwrap();
    assert!(!applied.dry_run);
    assert_eq!(applied.artifacts_found, 1);
    assert_eq!(applied.repaired, 1);
    assert_eq!(applied.skipped_concurrent_modification, 0);
    assert!(applied.errors.is_empty(), "errors: {:?}", applied.errors);

    // The row is now clean.
    {
        let conn = db.conn();
        let t: String = conn
            .query_row(
                "SELECT text FROM memories WHERE rid = ?1",
                rusqlite::params![rid],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(t, clean);
    }

    // The original was preserved for recovery.
    {
        let conn = db.conn();
        let orig: String = conn
            .query_row(
                "SELECT original_text FROM artifact_repair_audit WHERE rid = ?1",
                rusqlite::params![rid],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            orig.contains("</text>"),
            "audit preserves the dirty original"
        );
    }

    // Idempotent: a second apply finds nothing.
    let again = db.repair_tool_call_artifacts(false).unwrap();
    assert_eq!(again.artifacts_found, 0);
    assert_eq!(again.repaired, 0);

    // Recall still works — the vector index was rebuilt consistently.
    let hits = db.recall_text("database for metadata", 5).unwrap();
    assert!(
        hits.iter().any(|h| h.rid == rid),
        "repaired memory is still retrievable"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn importance_calibration_deflates_saturated_namespace() {
    // Task 31 end-to-end. A fresh namespace preserves importance exactly
    // (identity — this is why existing exact-importance tests still pass);
    // a namespace saturated with max-importance writes deflates further
    // high marks below 1.0 while keeping them in the high band.
    let db = YantrikDB::with_default(":memory:").unwrap();

    let read_importance = |rid: &str| -> f64 {
        let conn = db.conn();
        conn.query_row(
            "SELECT importance FROM memories WHERE rid = ?1",
            rusqlite::params![rid],
            |r| r.get(0),
        )
        .unwrap()
    };

    // Fresh namespace: a single max mark is stored exactly.
    let rid0 = db
        .record_text(
            "first genuinely critical fact",
            "semantic",
            1.0,
            0.0,
            604800.0,
            &empty_meta(),
            "fresh",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    assert!(
        (read_importance(&rid0) - 1.0).abs() < 1e-9,
        "fresh namespace preserves importance exactly: {}",
        read_importance(&rid0)
    );

    // Saturate a different namespace with max-importance writes.
    for i in 0..12 {
        db.record_text(
            &format!("everything here is marked critical {i}"),
            "semantic",
            1.0,
            0.0,
            604800.0,
            &empty_meta(),
            "saturated",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    }

    // The next max-importance write is deflated.
    let rid = db
        .record_text(
            "yet another self-declared critical fact",
            "semantic",
            1.0,
            0.0,
            604800.0,
            &empty_meta(),
            "saturated",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    let imp = read_importance(&rid);
    assert!(imp < 1.0, "saturated namespace deflates importance: {imp}");
    assert!(imp >= 0.70, "but keeps it in the high band: {imp}");

    // The deflated memory is still retrievable.
    let hits = db.recall_text("self-declared critical fact", 5).unwrap();
    assert!(hits.iter().any(|h| h.rid == rid));
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn recalibrate_unused_importance_reverts_stale_high_marks() {
    // Task 32 end-to-end. A high-importance memory that is never accessed
    // reverts toward baseline; a recently-written one is untouched; and the
    // pass is idempotent (re-running does not compound the reversion).
    let db = YantrikDB::with_default(":memory:").unwrap();
    let read_imp = |rid: &str| -> f64 {
        let conn = db.conn();
        conn.query_row(
            "SELECT importance FROM memories WHERE rid = ?1",
            rusqlite::params![rid],
            |r| r.get(0),
        )
        .unwrap()
    };

    let stale = db
        .record_text(
            "a once-critical fact nobody revisits",
            "semantic",
            1.0,
            0.0,
            604800.0,
            &empty_meta(),
            "ns",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    let fresh = db
        .record_text(
            "a fact that was just written",
            "semantic",
            1.0,
            0.0,
            604800.0,
            &empty_meta(),
            "ns",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    // Age the first far into the past, never re-accessed.
    {
        let conn = db.conn();
        conn.execute(
            "UPDATE memories SET last_access = 1000.0, access_count = 0 WHERE rid = ?1",
            rusqlite::params![stale],
        )
        .unwrap();
    }

    // Dry run detects exactly the stale candidate, mutating nothing.
    let dry = db.recalibrate_unused_importance(true).unwrap();
    assert!(dry.dry_run);
    assert_eq!(dry.adjusted, 1);
    assert!(
        (read_imp(&stale) - 1.0).abs() < 1e-9,
        "dry run must not mutate"
    );

    // Apply: the stale mark reverts; the fresh one is untouched.
    let applied = db.recalibrate_unused_importance(false).unwrap();
    assert_eq!(applied.adjusted, 1);
    assert!(applied.total_drift > 0.0);
    let reverted = read_imp(&stale);
    assert!(
        reverted < 1.0,
        "stale unused high mark reverted: {reverted}"
    );
    assert!(reverted >= 0.5, "but not below baseline: {reverted}");
    assert!(
        (read_imp(&fresh) - 1.0).abs() < 1e-9,
        "a freshly-written memory is untouched"
    );

    // Idempotent: re-running at the same staleness changes nothing further.
    let again = db.recalibrate_unused_importance(false).unwrap();
    assert_eq!(
        again.adjusted, 0,
        "reversion does not compound across passes"
    );
    assert!((read_imp(&stale) - reverted).abs() < 1e-9);
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn split_oversized_episodes_extracts_linked_atomic_facts() {
    // Task 33 end-to-end. An oversized episodic dump is split into atomic
    // facts, each linked back to the source episode; the parent is demoted
    // out of primary recall; and a query for a specific fact returns the
    // atomic child, not the wall-of-text parent.
    let db = YantrikDB::with_default(":memory:").unwrap();

    let episode = "Session recap. Alice was promoted to engineering lead this week. \
                   The team chose Postgres for the metadata store after benchmarking. \
                   The production launch slipped to March 30 because of the migration. \
                   Bob will own the on-call rotation starting next sprint. \
                   We agreed to cap importance writes so the signal stays meaningful.";
    let parent = db
        .record_text(
            episode,
            "episodic",
            1.0,
            0.0,
            604800.0,
            &empty_meta(),
            "recap",
            0.9,
            "work",
            "user",
            None,
        )
        .unwrap();

    // Dry run reports the split without performing it.
    let dry = db.split_oversized_episodes(true, 120).unwrap();
    assert_eq!(dry.episodes_scanned, 1);
    assert_eq!(dry.episodes_split, 0);
    assert!(dry.atomic_facts_created >= 2);

    // Apply.
    let applied = db.split_oversized_episodes(false, 120).unwrap();
    assert_eq!(applied.episodes_split, 1);
    assert!(applied.atomic_facts_created >= 2, "{applied:?}");
    assert!(applied.errors.is_empty(), "errors: {:?}", applied.errors);

    // The parent is demoted to consolidated (retained, out of primary recall).
    {
        let conn = db.conn();
        let status: String = conn
            .query_row(
                "SELECT consolidation_status FROM memories WHERE rid = ?1",
                rusqlite::params![parent],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(status, "consolidated", "parent episode demoted from recall");
    }

    // Atomic-fact children exist, linked back to the parent.
    let children = db
        .linked_records(&parent, crate::types::LinkDirection::Inbound, None)
        .unwrap();
    assert!(
        children.len() >= 2,
        "parent has atomic-fact children linked back: {}",
        children.len()
    );
    assert!(children.iter().all(|c| c.link_type == "derived_from"));

    // A query for a specific fact returns the atomic child, not the parent.
    let hits = db.recall_text("who owns the on-call rotation", 5).unwrap();
    assert!(!hits.is_empty());
    assert_ne!(
        hits[0].rid, parent,
        "top hit is an atomic fact, not the dump"
    );
    assert!(
        hits[0].text.chars().count() < episode.chars().count(),
        "the returned fact is shorter than the original dump"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn conflict_stamping_and_auto_resolution() {
    // Tasks 25 + 26. An open conflict between two memories is surfaced on
    // recall hits (stamp), then auto-resolved by newer-supersedes when it is
    // an unambiguous low/medium type.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let older = db
        .record_text(
            "The launch date is March 15",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            "ns",
            0.8,
            "work",
            "user",
            None,
        )
        .unwrap();
    // Force the first memory to be strictly older than the second.
    {
        let conn = db.conn();
        conn.execute(
            "UPDATE memories SET created_at = 1000.0 WHERE rid = ?1",
            rusqlite::params![older],
        )
        .unwrap();
    }
    let newer = db
        .record_text(
            "The launch date is March 30",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            "ns",
            0.8,
            "work",
            "user",
            None,
        )
        .unwrap();

    // Insert an open, auto-resolvable (temporal, medium) conflict.
    {
        let conn = db.conn();
        conn.execute(
            "INSERT INTO conflicts \
             (conflict_id, conflict_type, priority, status, memory_a, memory_b, \
              detected_at, detected_by, detection_reason, hlc, origin_actor) \
             VALUES ('cf1', 'temporal', 'medium', 'open', ?1, ?2, 2000.0, 'test', \
                     'same attribute, different value', X'00', 'test')",
            rusqlite::params![older, newer],
        )
        .unwrap();
    }

    // Task 25: the conflict is surfaced on the affected recall hits.
    let hits = db.recall_text("when is the launch date", 5).unwrap();
    let flagged = hits.iter().any(|h| {
        (h.rid == older || h.rid == newer)
            && h.why_retrieved
                .iter()
                .any(|w| w.contains("unresolved") && w.contains("conflict"))
    });
    assert!(flagged, "recall hits carry the conflict warning");

    // Task 26: dry-run reports it as auto-resolvable, mutating nothing.
    let dry = db.auto_resolve_conflicts(true).unwrap();
    assert_eq!(dry.open_before, 1);
    assert_eq!(dry.auto_resolved, 1);
    assert_eq!(dry.routed_to_operator, 0);

    // Apply: newer wins, older is tombstoned, the conflict is resolved.
    let applied = db.auto_resolve_conflicts(false).unwrap();
    assert_eq!(applied.auto_resolved, 1);
    {
        let conn = db.conn();
        let status: String = conn
            .query_row(
                "SELECT status FROM conflicts WHERE conflict_id = 'cf1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(status, "resolved");
        let older_status: String = conn
            .query_row(
                "SELECT consolidation_status FROM memories WHERE rid = ?1",
                rusqlite::params![older],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            older_status, "tombstoned",
            "the older, superseded memory is tombstoned"
        );
    }
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn auto_resolve_routes_identity_conflicts_to_operator() {
    // High-stakes conflicts are never auto-resolved.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let a = db
        .record_text(
            "Pranab lives in Seattle",
            "semantic",
            0.9,
            0.0,
            604800.0,
            &empty_meta(),
            "ns",
            0.9,
            "people",
            "user",
            None,
        )
        .unwrap();
    let b = db
        .record_text(
            "Pranab lives in Austin",
            "semantic",
            0.9,
            0.0,
            604800.0,
            &empty_meta(),
            "ns",
            0.9,
            "people",
            "user",
            None,
        )
        .unwrap();
    {
        let conn = db.conn();
        conn.execute(
            "INSERT INTO conflicts \
             (conflict_id, conflict_type, priority, status, memory_a, memory_b, \
              detected_at, detected_by, detection_reason, hlc, origin_actor) \
             VALUES ('cf2', 'identity_fact', 'high', 'open', ?1, ?2, 1.0, 'test', \
                     'identity conflict', X'00', 'test')",
            rusqlite::params![a, b],
        )
        .unwrap();
    }
    let report = db.auto_resolve_conflicts(false).unwrap();
    assert_eq!(
        report.auto_resolved, 0,
        "identity/high conflicts are not auto-resolved"
    );
    assert_eq!(report.routed_to_operator, 1);
    let conn = db.conn();
    let status: String = conn
        .query_row(
            "SELECT status FROM conflicts WHERE conflict_id = 'cf2'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(status, "open", "left open for an operator");
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn trigger_prune_bounds_pending_backlog() {
    // Task 27. Overdue triggers expire (TTL); the remaining pending backlog
    // is bounded to max_pending by evicting the lowest-urgency excess;
    // acknowledge removes a trigger from pending. Idempotent.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let insert = |id: &str, urgency: f64, expires_at: Option<f64>| {
        let conn = db.conn();
        conn.execute(
            "INSERT INTO trigger_log \
             (trigger_id, trigger_type, urgency, status, reason, suggested_action, \
              source_rids, context, created_at, expires_at, hlc, origin_actor) \
             VALUES (?1, 'decay_review', ?2, 'pending', 'r', 'a', '[]', '{}', 100.0, ?3, \
                     X'00', 'test')",
            rusqlite::params![id, urgency, expires_at],
        )
        .unwrap();
    };
    insert("t_overdue1", 0.9, Some(1.0));
    insert("t_overdue2", 0.9, Some(1.0));
    insert("t_live_lo", 0.1, None);
    insert("t_live_mid", 0.5, None);
    insert("t_live_hi1", 0.8, None);
    insert("t_live_hi2", 0.9, None);
    insert("t_live_hi3", 0.95, None);

    let count_pending = || -> i64 {
        let conn = db.conn();
        conn.query_row(
            "SELECT COUNT(*) FROM trigger_log WHERE status = 'pending'",
            [],
            |r| r.get(0),
        )
        .unwrap()
    };

    // Dry run: 7 pending, 2 overdue, 5 live capped to 3 → 2 over-cap.
    let dry = db.prune_triggers(true, 3).unwrap();
    assert_eq!(dry.pending_before, 7);
    assert_eq!(dry.expired_overdue, 2);
    assert_eq!(dry.expired_over_cap, 2);
    assert_eq!(dry.pending_after, 3);
    assert_eq!(count_pending(), 7, "dry run mutates nothing");

    // Apply: bound to 3.
    let applied = db.prune_triggers(false, 3).unwrap();
    assert_eq!(applied.pending_after, 3);
    assert_eq!(count_pending(), 3);
    {
        let conn = db.conn();
        let lo: String = conn
            .query_row(
                "SELECT status FROM trigger_log WHERE trigger_id = 't_live_lo'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(lo, "expired", "lowest-urgency evicted");
        let hi: String = conn
            .query_row(
                "SELECT status FROM trigger_log WHERE trigger_id = 't_live_hi3'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(hi, "pending", "highest-urgency retained");
    }

    // Re-running is stable now that the backlog is at the cap.
    let again = db.prune_triggers(false, 3).unwrap();
    assert_eq!(again.expired_overdue, 0);
    assert_eq!(again.expired_over_cap, 0);
    assert_eq!(again.pending_after, 3);
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn skill_outcomes_are_recorded_durably() {
    // Task 28. Each real skill outcome appends to the durable timeline so the
    // count rises; outcomes against a non-existent skill record nothing.
    let db = YantrikDB::with_default(":memory:").unwrap();
    assert_eq!(db.skill_outcome_count().unwrap(), 0);

    let taught = db
        .teach_skill(
            "deploy the staging build".to_string(),
            "k1".to_string(),
            vec![],
            crate::skills::SkillTrigger::default(),
        )
        .unwrap();
    assert!(taught);

    assert!(db.skill_succeeded("k1").unwrap());
    assert!(db.skill_failed("k1").unwrap());
    assert!(db.skill_accepted("k1").unwrap());
    assert!(!db.skill_succeeded("does_not_exist").unwrap());

    assert_eq!(
        db.skill_outcome_count().unwrap(),
        3,
        "one durable event per real outcome, none for the missing skill"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn maintenance_cycle_runs_passes_and_records_last_run() {
    // Task 24. The cycle runs the default hygiene passes with per-pass error
    // isolation, leaves the opt-in heavy passes off, and persists a last-run
    // summary for stats / the boot digest.
    let db = YantrikDB::with_default(":memory:").unwrap();
    db.record_text(
        "fact one about the project",
        "semantic",
        0.6,
        0.0,
        604800.0,
        &empty_meta(),
        "ns",
        0.8,
        "work",
        "user",
        None,
    )
    .unwrap();
    db.record_text(
        "fact two about the project",
        "semantic",
        0.6,
        0.0,
        604800.0,
        &empty_meta(),
        "ns",
        0.8,
        "work",
        "user",
        None,
    )
    .unwrap();

    assert!(
        db.last_maintenance_cycle().unwrap().is_none(),
        "no cycle yet"
    );

    let report = db
        .run_maintenance_cycle(&crate::MaintenanceCycleConfig::default())
        .unwrap();
    assert!(report.errors.is_empty(), "errors: {:?}", report.errors);
    assert!(report.ran_at > 0.0);
    // Default config: think + entities + relations + conflicts + triggers + importance ran.
    assert!(report.think_consolidations.is_some());
    assert!(report.entities_linked.is_some());
    assert!(report.relations_upserted.is_some());
    assert!(report.conflicts.is_some());
    assert!(report.triggers.is_some());
    assert!(report.importance.is_some());
    // Heavy passes are opt-in.
    assert!(report.split.is_none());
    assert!(report.repair.is_none());

    // The last-run summary is persisted and retrievable.
    let last = db
        .last_maintenance_cycle()
        .unwrap()
        .expect("last run recorded");
    assert!(last.contains("ran_at"));

    // Idempotent: a second cycle also succeeds with no errors.
    let again = db
        .run_maintenance_cycle(&crate::MaintenanceCycleConfig::default())
        .unwrap();
    assert!(again.errors.is_empty());
}

/// A dry cycle must leave ZERO fingerprints. The 2026-08-15 incident: the
/// MCP layer accepted dry_run=true, dropped it, and a "preview" auto-resolved
/// 15 conflicts and tombstoned 13 live records on a production store. This
/// pins the whole invariant at the engine layer: no status changes, no
/// importance drift, no conflict resolutions, no persisted summary.
#[cfg(feature = "bundled-embedder")]
#[test]
fn dry_run_maintenance_cycle_mutates_nothing() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    for t in [
        "fact one about the project",
        "fact two about the project",
        "an old important note",
        "another note entirely",
    ] {
        db.record_text(
            t,
            "semantic",
            0.9,
            0.0,
            604800.0,
            &empty_meta(),
            "ns",
            0.8,
            "work",
            "user",
            None,
        )
        .unwrap();
    }
    let snapshot = |db: &crate::YantrikDB| {
        let conn = db.conn();
        let row = |q: &str| -> i64 { conn.query_row(q, [], |r| r.get(0)).unwrap() };
        (
            row("SELECT COUNT(*) FROM memories WHERE consolidation_status='active'"),
            row("SELECT COUNT(*) FROM memories WHERE consolidation_status='tombstoned'"),
            row("SELECT COUNT(*) FROM conflicts WHERE status='resolved'"),
            format!(
                "{:?}",
                conn.prepare("SELECT rid, importance FROM memories ORDER BY rid")
                    .unwrap()
                    .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?)))
                    .unwrap()
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .unwrap()
            ),
        )
    };
    let before = snapshot(&db);

    let cfg = crate::MaintenanceCycleConfig {
        dry_run: true,
        ..Default::default()
    };
    let report = db.run_maintenance_cycle(&cfg).unwrap();
    assert!(report.errors.is_empty(), "errors: {:?}", report.errors);
    // Passes with no dry form are skipped, not quietly run wet.
    assert!(
        report.think_consolidations.is_none(),
        "think ran in a dry cycle"
    );
    assert!(
        report.entities_linked.is_none(),
        "backfill ran in a dry cycle"
    );

    assert_eq!(snapshot(&db), before, "dry cycle mutated the store");
    assert!(
        db.last_maintenance_cycle().unwrap().is_none(),
        "a preview must not masquerade as the last real cycle"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn recall_emits_structural_intent_hint() {
    // Task 35. A recency-intent query gets a hint pointing at the exact
    // structural path instead of silently returning a similarity-ranked guess.
    let db = YantrikDB::with_default(":memory:").unwrap();
    db.record_text(
        "entry one of the narrative",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        "chain",
        0.8,
        "self",
        "user",
        None,
    )
    .unwrap();

    let emb = db.embed("the most recent entry in the chain").unwrap();
    let response = db
        .recall_with_response(
            &emb,
            5,
            None,
            None,
            false,
            true,
            Some("what is the most recent entry in the chain"),
            true,
            None,
            None,
            None,
        )
        .unwrap();
    assert!(
        response
            .hints
            .iter()
            .any(|h| h.hint_type == "structural" && h.suggestion.contains("chain_head")),
        "a recency query yields a structural hint: {:?}",
        response.hints
    );

    // A plain semantic query gets no structural hint.
    let emb2 = db.embed("tell me about the narrative").unwrap();
    let plain = db
        .recall_with_response(
            &emb2,
            5,
            None,
            None,
            false,
            true,
            Some("tell me about the narrative content"),
            true,
            None,
            None,
            None,
        )
        .unwrap();
    assert!(
        !plain.hints.iter().any(|h| h.hint_type == "structural"),
        "no structural hint for a non-structural query"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn draft_memories_from_summary_atomizes_and_flags_provisional() {
    // Task 40. An agent's end-of-session summary is atomized into provisional,
    // retrievable candidate memories without the agent calling remember.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let summary = "We decided to use keyset cursors for list_records. \
                   Alice will own the database migration next sprint. \
                   The production launch slipped to March 30 because of it.";
    let rids = db
        .draft_memories_from_summary(summary, "session", "work")
        .unwrap();
    assert!(
        rids.len() >= 2,
        "summary atomized into facts: {}",
        rids.len()
    );

    for rid in &rids {
        let conn = db.conn();
        let meta: String = conn
            .query_row(
                "SELECT metadata FROM memories WHERE rid = ?1",
                rusqlite::params![rid],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            meta.contains("provisional"),
            "drafted memory is flagged provisional"
        );
    }

    let hits = db
        .recall_text("who owns the database migration", 5)
        .unwrap();
    assert!(hits.iter().any(|h| h.text.contains("migration")));
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn recall_stamps_trust_metadata() {
    // Task 41. An aged, rarely-confirmed memory and a superseded memory each
    // arrive on recall with a trust hedge in why_retrieved.
    //
    // v0.10 Item 1: serving a superseded result at all is LEGACY-policy
    // behavior (fresh DBs exclude it from eligibility), so this test pins
    // the stamped-hedge contract for pre-v0.10 databases.
    let db = YantrikDB::with_default(":memory:").unwrap();
    db.set_status_read_policy(false).unwrap();

    let aged = db
        .record_text(
            "an old fact about the deployment process",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            "ns",
            0.8,
            "work",
            "user",
            None,
        )
        .unwrap();
    {
        let conn = db.conn();
        conn.execute(
            "UPDATE memories SET created_at = ?1, access_count = 0 WHERE rid = ?2",
            rusqlite::params![crate::time::now_secs() - 200.0 * 86_400.0, aged],
        )
        .unwrap();
    }
    let hits = db.recall_text("deployment process fact", 5).unwrap();
    let h = hits
        .iter()
        .find(|h| h.rid == aged)
        .expect("aged hit present");
    assert!(
        h.why_retrieved
            .iter()
            .any(|w| w.contains("old") && w.contains("verify")),
        "aged-unconfirmed hedge present: {:?}",
        h.why_retrieved
    );

    // Supersession hedge.
    let old_v = db
        .record_text(
            "the API key rotates monthly",
            "semantic",
            0.6,
            0.0,
            604800.0,
            &empty_meta(),
            "ns2",
            0.8,
            "work",
            "user",
            None,
        )
        .unwrap();
    let new_v = db
        .record_text(
            "the API key rotates weekly now",
            "semantic",
            0.6,
            0.0,
            604800.0,
            &empty_meta(),
            "ns2",
            0.8,
            "work",
            "user",
            None,
        )
        .unwrap();
    db.link(
        &new_v,
        &crate::types::RecordLink {
            target_rid: old_v.clone(),
            link_type: crate::types::LinkType::Supersedes,
        },
    )
    .unwrap();
    let hits2 = db
        .recall_text("how often does the API key rotate", 5)
        .unwrap();
    let ho = hits2
        .iter()
        .find(|h| h.rid == old_v)
        .expect("superseded hit present");
    assert!(
        ho.why_retrieved.iter().any(|w| w.contains("superseded")),
        "superseded hedge present: {:?}",
        ho.why_retrieved
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn auto_relate_creates_cooccurrence_edges() {
    // Task 44. Entities that co-occur in a memory get linked, raising graph
    // density from plain writes. Idempotent.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let r1 = db
        .record_text(
            "Alice and Acme launched the Falcon project",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            "ns",
            0.8,
            "work",
            "user",
            None,
        )
        .unwrap();
    let r2 = db
        .record_text(
            "Alice and Acme shipped Falcon version two",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            "ns",
            0.8,
            "work",
            "user",
            None,
        )
        .unwrap();
    // Simulate the entity extraction (async materializer in production) having
    // linked entities to these memories, so auto-relate has co-occurrences.
    {
        let conn = db.conn();
        for (rid, ent) in [(&r1, "Alice"), (&r1, "Acme"), (&r2, "Alice"), (&r2, "Acme")] {
            conn.execute(
                "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name) VALUES (?1, ?2)",
                rusqlite::params![rid, ent],
            )
            .unwrap();
        }
    }

    let dry = db.auto_relate(true, 100).unwrap();
    assert!(
        dry.pairs_considered >= 1,
        "co-occurring pairs: {}",
        dry.pairs_considered
    );
    assert_eq!(dry.edges_upserted, 0, "dry run upserts nothing");

    let applied = db.auto_relate(false, 100).unwrap();
    assert!(
        applied.edges_upserted >= 1,
        "edges created: {}",
        applied.edges_upserted
    );

    // Idempotent: re-running considers the same pairs and errors-free.
    let again = db.auto_relate(false, 100).unwrap();
    assert_eq!(again.pairs_considered, applied.pairs_considered);
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn session_digest_assembles_boot_briefing() {
    // Task 38. One call returns the narrative head (latest, not
    // highest-importance), the top live decisions (high importance only), and
    // the open-conflict / pending-trigger counts.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let _n1 = db
        .record_text(
            "narrative entry one",
            "episodic",
            0.9,
            0.0,
            604800.0,
            &empty_meta(),
            "narr",
            0.9,
            "self",
            "user",
            None,
        )
        .unwrap();
    let n2 = db
        .record_text(
            "narrative entry two, the latest self-state",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            "narr",
            0.9,
            "self",
            "user",
            None,
        )
        .unwrap();
    db.record_text(
        "decided to adopt keyset cursors for enumeration",
        "semantic",
        0.95,
        0.0,
        604800.0,
        &empty_meta(),
        "work",
        0.9,
        "work",
        "user",
        None,
    )
    .unwrap();
    db.record_text(
        "a trivial passing aside",
        "semantic",
        0.2,
        0.0,
        604800.0,
        &empty_meta(),
        "work",
        0.5,
        "work",
        "user",
        None,
    )
    .unwrap();

    let cfg = crate::SessionDigestConfig {
        narrative_namespace: Some("narr".to_string()),
        ..Default::default()
    };
    let digest = db.session_digest(&cfg).unwrap();

    // Head is the latest entry, not the higher-importance one.
    let head = digest.narrative_head.expect("narrative head present");
    assert_eq!(head.rid, n2);
    assert!(head.snippet.contains("latest self-state"));

    // Top decisions: high-importance only.
    assert!(digest
        .top_decisions
        .iter()
        .any(|d| d.snippet.contains("keyset cursors")));
    assert!(!digest
        .top_decisions
        .iter()
        .any(|d| d.snippet.contains("trivial passing aside")));

    assert_eq!(digest.open_conflict_count, 0);
    assert_eq!(digest.pending_trigger_count, 0);
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn chain_head_returns_exact_latest_entry() {
    // Task 36. The chain head is exactly the latest write, independent of
    // importance — proving it is not the recall lottery.
    let db = YantrikDB::with_default(":memory:").unwrap();
    assert!(
        db.chain_head("chain").unwrap().is_none(),
        "empty chain has no head"
    );

    let _e1 = db
        .record_text(
            "entry one of the narrative",
            "episodic",
            1.0,
            0.0,
            604800.0,
            &empty_meta(),
            "chain",
            0.8,
            "self",
            "user",
            None,
        )
        .unwrap();
    let _e2 = db
        .record_text(
            "entry two of the narrative",
            "episodic",
            0.6,
            0.0,
            604800.0,
            &empty_meta(),
            "chain",
            0.8,
            "self",
            "user",
            None,
        )
        .unwrap();
    // The most recent entry is given the LOWEST importance, so a recall would
    // rank it last — chain_head must still return it.
    let e3 = db
        .record_text(
            "entry three, the most recent",
            "episodic",
            0.3,
            0.0,
            604800.0,
            &empty_meta(),
            "chain",
            0.8,
            "self",
            "user",
            None,
        )
        .unwrap();

    let head = db.chain_head("chain").unwrap().expect("head exists");
    assert_eq!(
        head.rid, e3,
        "head is the latest write, not the highest-importance"
    );
    assert!(head.text.contains("most recent"));

    // A different namespace is unaffected.
    assert!(db.chain_head("other").unwrap().is_none());
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn evict_protects_frequently_recalled_memories() {
    // Feature A (v0.9.0): hot/cold tiering uses recall frequency — a stale but
    // frequently-recalled memory is NOT evicted just for being old, while
    // equally-stale never-recalled peers are.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let mut rids = Vec::new();
    for i in 0..5 {
        rids.push(
            db.record_text(
                &format!("memory number {i} about assorted unrelated topics"),
                "semantic",
                0.5,
                0.0,
                604800.0,
                &empty_meta(),
                "ns",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap(),
        );
    }
    let hot = rids[0].clone();
    {
        let conn = db.conn();
        // Make ALL equally old / stale / never-recalled...
        conn.execute(
            "UPDATE memories SET created_at = 1000.0, last_access = 1000.0, access_count = 0",
            [],
        )
        .unwrap();
        // ...except one, which has been recalled many times.
        conn.execute(
            "UPDATE memories SET access_count = 50 WHERE rid = ?1",
            rusqlite::params![hot],
        )
        .unwrap();
    }

    let evicted = db.evict(2).unwrap();
    assert_eq!(evicted.len(), 3, "evicts down to max_active = 2");
    assert!(
        !evicted.contains(&hot),
        "the frequently-recalled memory survives"
    );

    let tier: String = {
        let conn = db.conn();
        conn.query_row(
            "SELECT storage_tier FROM memories WHERE rid = ?1",
            rusqlite::params![hot],
            |r| r.get(0),
        )
        .unwrap()
    };
    assert_eq!(tier, "hot", "the hot memory stays hot");
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn recall_logs_demand_and_surfaces_gaps() {
    // Feature B (v0.9.0): a user-facing recall auto-logs demand; a frequently
    // asked, poorly-answered query surfaces as a knowledge gap.
    let db = YantrikDB::with_default(":memory:").unwrap();
    // One unrelated memory so recall reaches the demand-logging tail (an empty
    // corpus short-circuits before it). The query stays poorly answered.
    db.record_text(
        "the orchard wall was painted blue last spring",
        "semantic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        "ns",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();
    for _ in 0..4 {
        let _ = db
            .recall_text("how do I rotate the encryption keys", 5)
            .unwrap();
    }
    // recall_text issues an unscoped recall, so its demand lands in the
    // global bucket (namespace = None) per the v0.9.3 isolation contract.
    let (count, avg_top) = db
        .recall_demand_for(None, "how do I rotate the encryption keys")
        .unwrap()
        .expect("the query was logged as demand");
    assert_eq!(count, 4, "asked four times");

    // Surfaces as a gap at a threshold just above its (low) answer quality.
    let gaps = db.knowledge_gaps(None, 3, avg_top + 0.01, 10).unwrap();
    assert!(
        gaps.iter()
            .any(|g| g.query.contains("rotate the encryption keys")),
        "frequent poorly-answered query surfaces as a gap: {gaps:?}"
    );

    // An internal recall (skip_reinforce) must NOT pollute the demand log.
    let emb = db.embed("a different internal probe query").unwrap();
    let _ = db
        .recall(
            &emb,
            5,
            None,
            None,
            false,
            true,
            Some("a different internal probe query"),
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
        db.recall_demand_for(None, "a different internal probe query")
            .unwrap()
            .is_none(),
        "internal (skip_reinforce) recalls are not logged as demand"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn migration_v33_purges_unscopable_demand_rows() {
    // v0.9.3 isolation repair (sol converged plan item 2). Databases written
    // by v0.9.0–v0.9.2 have a GLOBAL-keyed recall_demand table with raw
    // query text; those legacy rows are unscopable (no namespace recorded),
    // so the v32→v33 migration PURGES them and SCHEMA_SQL recreates the
    // namespace-keyed shape. This simulates such a populated pre-fix DB.
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    // Current-schema DB first (so all OTHER tables are in place)...
    {
        let _db = YantrikDB::new(path, 8).unwrap();
    }
    // ...then regress recall_demand to the v0.9.0 shape with a legacy row
    // and rewind the version stamp to 32.
    {
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.execute_batch(
            "DROP TABLE recall_demand;
             CREATE TABLE recall_demand (
                 query_norm TEXT PRIMARY KEY,
                 sample_text TEXT NOT NULL,
                 count INTEGER NOT NULL,
                 sum_top_score REAL NOT NULL,
                 sum_results INTEGER NOT NULL,
                 last_seen REAL NOT NULL
             );
             INSERT INTO recall_demand VALUES
                 ('legacy unscopable query', 'Legacy Unscopable Query?', 7, 0.4, 3, 1.0);
             INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '32');",
        )
        .unwrap();
    }

    // Reopen: V32_TO_V33 drops the legacy table; SCHEMA_SQL recreates the
    // namespace-keyed shape. Legacy rows are gone (purged, not guessed).
    let db = YantrikDB::new(path, 8).expect("v33 migration must succeed on a populated v32 DB");
    {
        let conn = db.conn();
        let rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM recall_demand", [], |r| r.get(0))
            .unwrap();
        assert_eq!(rows, 0, "unscopable legacy demand rows are purged");
        // The new shape is namespace-keyed (this errors if the column is absent).
        conn.query_row("SELECT namespace FROM recall_demand LIMIT 1", [], |r| {
            r.get::<_, String>(0)
        })
        .ok();
    }
    // Demand logging works post-migration under the new key.
    db.record_recall_demand(Some("ns-a"), "post migration question", 0, 0.0)
        .unwrap();
    assert!(db
        .recall_demand_for(Some("ns-a"), "post migration question")
        .unwrap()
        .is_some());
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn session_digest_scopes_decisions_and_conflicts_to_namespace() {
    // v0.9.3: a namespace-scoped digest must not mix another tenant's
    // high-importance memories into top_decisions.
    let db = YantrikDB::with_default(":memory:").unwrap();
    db.record_text(
        "tenant A signed the enterprise contract",
        "semantic",
        0.95,
        0.0,
        604800.0,
        &empty_meta(),
        "tenant-a",
        0.9,
        "work",
        "user",
        None,
    )
    .unwrap();
    db.record_text(
        "tenant B is migrating to postgres",
        "semantic",
        0.95,
        0.0,
        604800.0,
        &empty_meta(),
        "tenant-b",
        0.9,
        "work",
        "user",
        None,
    )
    .unwrap();

    let scoped = db
        .session_digest(&crate::SessionDigestConfig {
            namespace: Some("tenant-a".into()),
            ..Default::default()
        })
        .unwrap();
    assert!(
        !scoped.top_decisions.is_empty(),
        "tenant-a's own decision is present"
    );
    assert!(
        scoped
            .top_decisions
            .iter()
            .all(|d| d.namespace == "tenant-a"),
        "no cross-tenant decisions in a scoped digest: {:?}",
        scoped.top_decisions
    );

    // Unscoped (explicit-global) digest still sees both — unchanged behavior.
    let global = db
        .session_digest(&crate::SessionDigestConfig::default())
        .unwrap();
    let namespaces: std::collections::HashSet<_> = global
        .top_decisions
        .iter()
        .map(|d| d.namespace.clone())
        .collect();
    assert!(namespaces.contains("tenant-a") && namespaces.contains("tenant-b"));
}

#[test]
fn digest_packet_is_status_led_and_reports_changes_since() {
    // v0.10 Item 1c / trace T10 "packet-correctness". Fixture: decisions
    // A (superseded), B (head), C (open question), D (disputed, vs E) in
    // one namespace. The packet main view carries B, C, D-with-flag; A
    // exists only behind include_superseded; what_changed_since(T)
    // returns exactly the records and status transitions after T.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rec = |text: &str, seed: f32| {
        db.record(
            text,
            "semantic",
            0.9,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(seed, 8),
            "t10",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap()
    };
    let a = rec("decision A: deploy to staging", 1.0);
    let b = rec("decision B: deploy to production (corrects A)", 1.05);
    let c = rec("open question C: which region", 2.0);
    let d = rec("decision D: use postgres", 3.0);
    let e = rec("decision E: use sqlite (rival of D)", 3.05);

    db.link(
        &b,
        &crate::types::RecordLink {
            target_rid: a.clone(),
            link_type: crate::types::LinkType::Supersedes,
        },
    )
    .unwrap();
    crate::create_conflict(
        &db,
        &crate::types::ConflictType::Preference,
        &d,
        &e,
        None,
        None,
        "T10 fixture: D vs E",
    )
    .unwrap();

    // Deterministic timeline (no wall-clock asserts): A predates T=1500,
    // everything else follows it. The supersedes link keeps its real
    // (post-T) commit time.
    {
        let conn = db.conn();
        conn.execute(
            "UPDATE memories SET created_at = 1000.0 WHERE rid = ?1",
            rusqlite::params![a],
        )
        .unwrap();
        conn.execute(
            "UPDATE memories SET created_at = 2000.0 WHERE rid IN (?1, ?2, ?3, ?4)",
            rusqlite::params![b, c, d, e],
        )
        .unwrap();
    }

    // Main view: status-led (fresh DB → policy active).
    let digest = db
        .session_digest(&crate::SessionDigestConfig {
            namespace: Some("t10".into()),
            ..Default::default()
        })
        .unwrap();
    let rids: Vec<&str> = digest
        .top_decisions
        .iter()
        .map(|x| x.rid.as_str())
        .collect();
    assert!(rids.contains(&b.as_str()), "head B in main view");
    assert!(rids.contains(&c.as_str()), "open question C in main view");
    assert!(
        rids.contains(&d.as_str()),
        "disputed D in main view (not dropped)"
    );
    assert!(
        !rids.contains(&a.as_str()),
        "superseded A absent from main view"
    );
    let d_entry = digest.top_decisions.iter().find(|x| x.rid == d).unwrap();
    assert!(d_entry.disputed, "D carries the typed disputed flag");
    let b_entry = digest.top_decisions.iter().find(|x| x.rid == b).unwrap();
    assert!(!b_entry.disputed);
    assert_eq!(b_entry.current_status, crate::types::RecordStatus::Active);

    // Expansion: A re-admitted, stamped.
    let expanded = db
        .session_digest(&crate::SessionDigestConfig {
            namespace: Some("t10".into()),
            include_superseded: true,
            ..Default::default()
        })
        .unwrap();
    let a_entry = expanded
        .top_decisions
        .iter()
        .find(|x| x.rid == a)
        .expect("A only behind include_superseded");
    assert_eq!(
        a_entry.current_status,
        crate::types::RecordStatus::Superseded
    );
    assert_eq!(a_entry.superseded_by.as_deref(), Some(b.as_str()));

    // what_changed_since(T=1500): B/C/D/E are new, A is not; exactly one
    // status transition (A → Superseded by B, committed after T).
    let changes = db.what_changed_since(1500.0, Some("t10"), 240).unwrap();
    let new_rids: Vec<&str> = changes.new_records.iter().map(|x| x.rid.as_str()).collect();
    for rid in [&b, &c, &d, &e] {
        assert!(new_rids.contains(&rid.as_str()), "{rid} is new since T");
    }
    assert!(!new_rids.contains(&a.as_str()), "A predates T");
    assert_eq!(
        changes.status_transitions.len(),
        1,
        "exactly one transition"
    );
    let tr = &changes.status_transitions[0];
    assert_eq!(tr.rid, a);
    assert_eq!(tr.from, crate::types::RecordStatus::Active);
    assert_eq!(tr.to, crate::types::RecordStatus::Superseded);
    assert_eq!(tr.by_rid.as_deref(), Some(b.as_str()));
    assert!(tr.at > 1500.0, "transition committed after T");

    // Nothing changed since a T after everything.
    let quiet = db
        .what_changed_since(crate::time::now_secs() + 10.0, Some("t10"), 240)
        .unwrap();
    assert!(quiet.new_records.is_empty());
    assert!(quiet.status_transitions.is_empty());
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn explicit_set_embedder_overrides_bundled() {
    // Slim-build path or custom-model path: set_embedder() after new()
    // takes precedence. The bundled embedder gets dropped; the user's
    // takes over.
    struct DummyEmbedder;
    impl crate::types::Embedder for DummyEmbedder {
        fn embed(
            &self,
            _t: &str,
        ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
            // Distinct sentinel value so we can detect this implementation was used.
            let mut v = vec![0.0; 64];
            v[0] = 0.7777;
            Ok(v)
        }
        fn dim(&self) -> usize {
            64
        }
    }

    let mut db = YantrikDB::with_default(":memory:").unwrap();
    assert!(db.has_embedder(), "starts with bundled");
    // Issue #41 layer 2 / brainstorm-3: set_embedder is now mode-aware
    // and returns Result. For an empty DB (no memories indexed yet)
    // the call accepts ANY embedder regardless of fingerprint match,
    // updating provenance based on candidate.fingerprint().
    // DummyEmbedder returns None from fingerprint() so provenance
    // stays ExternalOrUnknown; runtime_embedder slot updates.
    db.set_embedder(Box::new(DummyEmbedder)).unwrap();
    let v = db.embed("anything").unwrap();
    assert!(
        (v[0] - 0.7777).abs() < 1e-6,
        "DummyEmbedder's sentinel must be visible — set_embedder overrode bundled"
    );
}

// ── 2026-08-13: default for NEW file-backed stores moved to
// potion-base-8M (256d), downloaded on first use.
//
// The tests below pin the property that makes that switch safe to ship:
// an EXISTING database is reopened at the dimension it already holds.
// The vector index is built from the dimension passed to the
// constructor, so a store created at 64 and reopened at 256 would build
// a mismatched index over its existing vectors — data present, and
// unfindable. None of these touch the network: they assert the
// protection path, which resolves before any fetch is attempted.

#[cfg(feature = "bundled-embedder")]
#[test]
fn existing_store_keeps_its_dimension_when_the_default_changes() {
    use crate::embedder::BUNDLED_EMBEDDER_DIM;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy.db");
    let p = path.to_str().unwrap();

    // A store from before the default changed: 64-dim, with a real
    // vector in it so the embedder identity gets stamped.
    {
        let db = YantrikDB::new(p, BUNDLED_EMBEDDER_DIM).unwrap();
        db.record_text(
            "the deploy key is id_yantrikdb_web_deploy",
            "semantic",
            0.9,
            0.0,
            604800.0,
            &serde_json::json!({}),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    }

    // Reopening through the constructor whose default is now 256 must
    // still land on 64 — otherwise every existing user's store breaks.
    let reopened = YantrikDB::with_default(p).unwrap();
    assert_eq!(
        reopened.embedding_dim(),
        BUNDLED_EMBEDDER_DIM,
        "with_default reopened a {BUNDLED_EMBEDDER_DIM}-dim store at {} — this strands \
         every database created before the default changed",
        reopened.embedding_dim()
    );
    assert_eq!(
        reopened.stats(None).unwrap().active_memories,
        1,
        "the pre-existing record must still be there and readable"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn existing_dimension_is_detected_without_an_embedder_identity() {
    // Databases whose vectors were supplied by the caller never stamp an
    // embedder identity — `record()` takes the vector directly. Those are
    // precisely the deployments a dimension change would corrupt
    // silently, so detection must fall back to measuring a stored vector
    // rather than trusting the identity row alone.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("external-vectors.db");
    let p = path.to_str().unwrap();
    {
        let db = YantrikDB::new(p, 384).unwrap();
        db.record(
            "vector supplied by an external MiniLM",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(0.3, 384),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    }
    let reopened = YantrikDB::with_default(p).unwrap();
    assert_eq!(
        reopened.embedding_dim(),
        384,
        "a store holding 384-dim external vectors must reopen at 384"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn in_memory_default_stays_on_the_bundled_embedder() {
    // Deliberate: in-memory stores are ephemeral and the test suite opens
    // many of them, so they must not require a 28 MB download. If this
    // ever flips, `cargo test` starts depending on the network.
    use crate::embedder::BUNDLED_EMBEDDER_DIM;
    let db = YantrikDB::with_default(":memory:").unwrap();
    assert_eq!(db.embedding_dim(), BUNDLED_EMBEDDER_DIM);
    assert!(db.has_embedder());
}

#[cfg(all(feature = "bundled-embedder", feature = "embedder-download"))]
#[test]
#[ignore = "touches the model cache / network; run explicitly with --ignored"]
fn new_file_store_gets_the_downloadable_default() {
    // The end-to-end claim of the 2026-08-13 default change. Ignored by
    // default so `cargo test` stays hermetic — the whole reason in-memory
    // stores were kept on the bundled model.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fresh.db");
    let db = YantrikDB::with_default(path.to_str().unwrap()).unwrap();
    assert_eq!(
        db.embedding_dim(),
        256,
        "a NEW file-backed store must open at potion-base-8M's 256 dims"
    );
    assert!(db.has_embedder(), "and must have that embedder attached");

    let rid = db
        .record_text(
            "the website deploys via a git post-receive hook",
            "semantic",
            0.9,
            0.0,
            604800.0,
            &empty_meta(),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    let hits = db.recall_text("how do I deploy the site", 5).unwrap();
    assert!(
        hits.iter().any(|h| h.rid == rid),
        "the store must actually embed and retrieve with the downloaded model"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn stored_vectors_outrank_a_disagreeing_embedder_identity() {
    // Taken from a real store, not imagined: a 5,050-record production
    // database recorded `embedder_dim = 64 / potion-base-2M` in `meta`
    // while holding 1536-byte (384-dim) MiniLM vectors, because the
    // server embedded python-side and passed vectors to `record()` while
    // an incidental engine-side `embed()` stamped the attached bundled
    // model. Believing the identity row would reopen that database at 64
    // dims and build a 64-dim index over 384-dim vectors.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lying-identity.db");
    let p = path.to_str().unwrap();
    {
        let db = YantrikDB::new(p, 384).unwrap();
        db.record(
            "vectors produced outside the engine",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(0.25, 384),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
        // Forge the disagreement the production store actually had.
        let conn = db.conn();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('embedder_dim', '64')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('embedder_name', 'potion-base-2M')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('embedder_digest', 'blake3:forged')",
            [],
        )
        .unwrap();
    }

    let reopened = YantrikDB::with_default(p).unwrap();
    assert_eq!(
        reopened.embedding_dim(),
        384,
        "the stored vectors are 384-dim; the meta row claiming 64 must not win, or the \
         engine builds a 64-dim index over 384-dim vectors"
    );
}

#[cfg(all(feature = "bundled-embedder", feature = "embedder-download"))]
#[test]
#[ignore = "needs the model cache; run with --features embedder-download --ignored"]
fn a_downloaded_embedder_can_prove_its_embedding_space_to_a_pack() {
    // Before DownloadedEmbedder implemented fingerprint()/name(), a store
    // on potion-base-8M could not mount ANY pack — not on a dimension
    // mismatch, but because the engine had no identity to record, so
    // mount refused with "this database has no recorded embedder
    // identity, so compatibility cannot be proven". That made the pack
    // system bundled-embedder-only, which only surfaced when a 256-dim
    // pack was built and mounted into a 256-dim host.
    use crate::engine::pack::{PackEmbedder, PackManifest};
    let dir = tempfile::tempdir().unwrap();
    let author_path = dir.path().join("author.db");
    let pack_path = dir.path().join("proof.ydbpack");

    let dim = crate::embedder::DownloadedEmbedder::registry_dim("potion-base-8M").unwrap();
    let mut author = YantrikDB::new(author_path.to_str().unwrap(), dim).unwrap();
    author.set_embedder_named("potion-base-8M").unwrap();
    for text in [
        "importance 0.8 to 1.0 marks a decision that changes what we build",
        "a correction preserves history where a second remember does not",
    ] {
        author
            .record_text(
                text,
                "semantic",
                0.6,
                0.0,
                604800.0,
                &empty_meta(),
                "default",
                0.8,
                "general",
                "document",
                None,
            )
            .unwrap();
    }

    // THE FIX'S ACTUAL CONTRACT: a store on a downloaded model must now
    // have a durable embedder identity. Without it the Python binding's
    // seal_pack has nothing to default the manifest's embedder to, which
    // is how packs built on 8M ended up declaring name=null digest=null.
    let identity = author
        .embedder_identity()
        .expect("reading identity must not error")
        .expect(
            "a store that recorded vectors with a downloaded embedder must have a durable              embedder identity — without one it can neither seal a mountable pack nor mount one",
        );

    let manifest = PackManifest {
        name: "identity-proof".into(),
        version: "0.1.0".into(),
        origin: "yantrik/identity-proof".into(),
        description: None,
        // Mirrors what the Python binding does when the caller omits
        // these: default them to what the database actually is.
        embedder: PackEmbedder {
            name: identity.0.clone(),
            digest: Some(identity.1.clone()),
            dim,
        },
        content_digest: None,
        corpus_rows: 0,
        namespace: None,
        publisher_pubkey: None,
        signature: None,
        reembedded_from: None,
        constitution: vec![],
        coverage: vec![],
        recommended_top_k: None,
        recommended_min_similarity: None,
    };
    let sealed = author
        .seal_pack(pack_path.to_str().unwrap(), &manifest, None)
        .expect("sealing from a downloaded-embedder store must work");
    assert!(
        sealed.embedder.digest.is_some(),
        "the sealed pack must carry an embedder digest, or no host can ever prove \
         compatibility with it; got {:?}",
        sealed.embedder
    );
    assert_eq!(sealed.embedder.name.as_deref(), Some("potion-base-8M"));

    // A different, empty host on the same model must accept it.
    let host_path = dir.path().join("host.db");
    let mut host = YantrikDB::new(host_path.to_str().unwrap(), dim).unwrap();
    host.set_embedder_named("potion-base-8M").unwrap();
    host.mount_pack(pack_path.to_str().unwrap())
        .expect("a host on the same downloaded model must mount the pack");
}

#[cfg(all(feature = "bundled-embedder", feature = "embedder-download"))]
#[test]
#[ignore = "needs the model cache; run with --features embedder-download --ignored"]
fn a_64_dim_pack_converts_and_mounts_into_a_256_dim_host() {
    // The whole point of convert_pack: ONE published artifact, usable by
    // hosts in different embedding spaces. Every pack in the wild today
    // is 64-dim/potion-base-2M, and mount treats a dimension mismatch as
    // unconditionally fatal, so without conversion the engine's new
    // 256-dim default would strand the entire pack catalogue.
    use crate::embedder::BUNDLED_EMBEDDER_DIM;
    use crate::engine::pack::{PackEmbedder, PackManifest};
    let dir = tempfile::tempdir().unwrap();
    let author_path = dir.path().join("author.db");
    let pack64 = dir.path().join("original-64.ydbpack");
    let pack256 = dir.path().join("converted-256.ydbpack");

    // Author a pack exactly as the catalogue was built: bundled 2M, 64 dims.
    let author = YantrikDB::new(author_path.to_str().unwrap(), BUNDLED_EMBEDDER_DIM).unwrap();
    for text in [
        "the deploy key for the website is id_yantrikdb_web_deploy",
        "importance 0.8 to 1.0 marks a decision that changes what we build",
        "a correction preserves history where a second remember does not",
    ] {
        author
            .record_text(
                text,
                "semantic",
                0.6,
                0.0,
                604800.0,
                &empty_meta(),
                "default",
                0.9,
                "general",
                "document",
                None,
            )
            .unwrap();
    }
    let identity = author.embedder_identity().unwrap().unwrap();
    let manifest = PackManifest {
        name: "convert-proof".into(),
        version: "0.1.0".into(),
        origin: "yantrik/convert-proof".into(),
        description: None,
        embedder: PackEmbedder {
            name: identity.0.clone(),
            digest: Some(identity.1.clone()),
            dim: BUNDLED_EMBEDDER_DIM,
        },
        content_digest: None,
        corpus_rows: 0,
        namespace: None,
        publisher_pubkey: None,
        signature: None,
        reembedded_from: None,
        constitution: vec!["Answer from the asker's side.".into()],
        coverage: vec!["memory discipline".into()],
        recommended_top_k: Some(6),
        recommended_min_similarity: Some(0.5),
    };
    let sealed = author
        .seal_pack(pack64.to_str().unwrap(), &manifest, None)
        .unwrap();
    drop(author);

    // A 256-dim host cannot mount it — this is the failure being solved.
    let host_path = dir.path().join("host.db");
    let dim256 = crate::embedder::DownloadedEmbedder::registry_dim("potion-base-8M").unwrap();
    let mut host = YantrikDB::new(host_path.to_str().unwrap(), dim256).unwrap();
    host.set_embedder_named("potion-base-8M").unwrap();
    let refused = host.mount_pack(pack64.to_str().unwrap());
    assert!(
        refused.is_err(),
        "a 64-dim pack must NOT mount into a 256-dim host; if this starts passing the          dimension guard has been weakened, not the conversion path fixed"
    );

    // Convert, then mount.
    let converted = YantrikDB::convert_pack(
        pack64.to_str().unwrap(),
        pack256.to_str().unwrap(),
        "potion-base-8M",
    )
    .expect("conversion must succeed");
    assert_eq!(converted.embedder.dim, dim256);
    assert_eq!(converted.embedder.name.as_deref(), Some("potion-base-8M"));
    assert_eq!(
        converted.content_digest, sealed.content_digest,
        "conversion must not change the content digest — the rows are still the publisher's,          only the vectors are ours"
    );
    assert_eq!(
        converted.reembedded_from, sealed.embedder.digest,
        "the original embedder digest must be recorded so the conversion is visible"
    );
    assert!(
        converted.signature.is_none() && converted.publisher_pubkey.is_none(),
        "a publisher signature covers the embedder identity and cannot survive re-embedding"
    );
    assert_eq!(
        converted.constitution.len(),
        1,
        "the constitution must survive conversion — it is what the pack DOES"
    );

    host.mount_pack(pack256.to_str().unwrap())
        .expect("the converted pack must mount into the 256-dim host");

    // And it must actually retrieve, not merely mount.
    let hits = host
        .recall_text("which ssh key deploys the website", 5)
        .unwrap();
    assert!(
        hits.iter()
            .any(|h| h.text.contains("id_yantrikdb_web_deploy")),
        "the converted pack's rows must be retrievable in the host's space; got {:?}",
        hits.iter().map(|h| &h.text).collect::<Vec<_>>()
    );
}

#[cfg(all(feature = "bundled-embedder", feature = "embedder-download"))]
#[test]
#[ignore = "needs the model cache; run with --features embedder-download --ignored"]
fn install_pack_converts_a_foreign_dimension_pack_automatically() {
    // The convenience half: a user should not have to know what embedding
    // space a pack was published in. install_pack is where the conversion
    // belongs — durable, once, explicit — while mount_pack stays a
    // read-only attach that writes nothing.
    use crate::embedder::BUNDLED_EMBEDDER_DIM;
    use crate::engine::pack::{PackEmbedder, PackManifest};
    let dir = tempfile::tempdir().unwrap();
    let pack64 = dir.path().join("catalogue-64.ydbpack");

    let author = YantrikDB::new(
        dir.path().join("author.db").to_str().unwrap(),
        BUNDLED_EMBEDDER_DIM,
    )
    .unwrap();
    author
        .record_text(
            "the deploy key for the website is id_yantrikdb_web_deploy",
            "semantic",
            0.6,
            0.0,
            604800.0,
            &empty_meta(),
            "default",
            0.9,
            "general",
            "document",
            None,
        )
        .unwrap();
    let ident = author.embedder_identity().unwrap().unwrap();
    author
        .seal_pack(
            pack64.to_str().unwrap(),
            &PackManifest {
                name: "auto-convert".into(),
                version: "0.1.0".into(),
                origin: "yantrik/auto-convert".into(),
                description: None,
                embedder: PackEmbedder {
                    name: ident.0.clone(),
                    digest: Some(ident.1.clone()),
                    dim: BUNDLED_EMBEDDER_DIM,
                },
                content_digest: None,
                corpus_rows: 0,
                namespace: None,
                publisher_pubkey: None,
                signature: None,
                reembedded_from: None,
                constitution: vec![],
                coverage: vec![],
                recommended_top_k: None,
                recommended_min_similarity: None,
            },
            None,
        )
        .unwrap();
    drop(author);

    let dim256 = crate::embedder::DownloadedEmbedder::registry_dim("potion-base-8M").unwrap();
    let host_path = dir.path().join("host.db");
    let mut host = YantrikDB::new(host_path.to_str().unwrap(), dim256).unwrap();
    host.set_embedder_named("potion-base-8M").unwrap();
    // Give the host an identity to convert INTO — conversion needs to
    // name the target space.
    host.record_text(
        "host's own memory",
        "semantic",
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
    .unwrap();

    host.install_pack(pack64.to_str().unwrap())
        .expect("install_pack must convert a 64-dim pack into this 256-dim host");

    let hits = host
        .recall_text("which ssh key deploys the website", 5)
        .unwrap();
    assert!(
        hits.iter()
            .any(|h| h.text.contains("id_yantrikdb_web_deploy")),
        "the auto-converted pack's rows must be retrievable; got {:?}",
        hits.iter().map(|h| &h.text).collect::<Vec<_>>()
    );

    // And it must survive a reopen, which is the point of installing:
    // the CONVERTED file is what remounts, not the original.
    drop(host);
    let mut reopened = YantrikDB::new(host_path.to_str().unwrap(), dim256).unwrap();
    reopened.set_embedder_named("potion-base-8M").unwrap();
    reopened.remount_installed();
    let after = reopened
        .recall_text("which ssh key deploys the website", 5)
        .unwrap();
    assert!(
        after
            .iter()
            .any(|h| h.text.contains("id_yantrikdb_web_deploy")),
        "the converted pack must remount on reopen; got {:?}",
        after.iter().map(|h| &h.text).collect::<Vec<_>>()
    );
}

#[test]
fn refresh_embeddings_heals_rows_that_have_no_vector() {
    // The replication shape, reproduced locally: a row that is present and
    // ACTIVE in SQL but carries no embedding, so it is invisible to semantic
    // recall while looking perfectly healthy to `get()` and to any count.
    // replication.rs materializes exactly this — its INSERT omits embedding,
    // embedding_model and embedding_generation — and then declines to add the
    // record to HNSW because it has no vector to add.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let target = "The reconciliation service settles invoices against bank statements.";
    let rid = db
        .record_text(
            target,
            "semantic",
            0.6,
            0.0,
            86400.0,
            &serde_json::json!({}),
            "repl",
            0.9,
            "general",
            "user",
            None,
        )
        .unwrap();
    db.record_text(
        "Unrelated note about the office coffee machine.",
        "semantic",
        0.5,
        0.0,
        86400.0,
        &serde_json::json!({}),
        "repl",
        0.9,
        "general",
        "user",
        None,
    )
    .unwrap();

    let indexed_before = db.rebuild_vec_index().unwrap();
    assert_eq!(indexed_before, 2, "both rows start with usable vectors");

    // Strip the vector, the way a replicated row arrives.
    db.conn()
        .execute(
            "UPDATE memories SET embedding = NULL WHERE rid = ?1",
            [&rid],
        )
        .unwrap();

    // Assert on INDEX MEMBERSHIP, not on a recall result. A vector-less row
    // is still reachable through the FTS lane whenever the query happens to
    // share tokens with the text — the damage is confined to the vector lane,
    // so that is where it has to be measured. (Worth knowing on its own: the
    // replication gap degrades SEMANTIC recall specifically; it does not make
    // a record wholly unfindable, which is part of why it stayed quiet.)
    let indexed_damaged = db.rebuild_vec_index().unwrap();
    assert_eq!(
        indexed_damaged, 1,
        "precondition: a row with no vector cannot be in the vector index"
    );

    // Dry run reports the damage and changes nothing.
    let dry = db.refresh_embeddings(Some("repl"), true).unwrap();
    assert!(dry.dry_run);
    assert_eq!(
        dry.unusable_found, 1,
        "exactly one row lacks a usable vector"
    );
    assert_eq!(dry.refreshed, 0, "a dry run must not write");
    assert!(dry.sample_rids.contains(&rid));

    // Apply heals it.
    let applied = db.refresh_embeddings(Some("repl"), false).unwrap();
    assert_eq!(applied.refreshed, 1, "the damaged row must be re-encoded");
    assert!(applied.errors.is_empty(), "errors: {:?}", applied.errors);

    let indexed_after = db.rebuild_vec_index().unwrap();
    assert_eq!(
        indexed_after, 2,
        "after refresh the healed row must be back in the vector index"
    );
    let back = db
        .recall_text("how are invoices settled against statements", 5)
        .unwrap();
    assert!(
        back.iter().any(|h| h.text == target),
        "and reachable again; got {:?}",
        back.iter().map(|h| &h.text).collect::<Vec<_>>()
    );

    // Idempotent: nothing is unusable now, so a second sweep is a no-op.
    let again = db.refresh_embeddings(Some("repl"), false).unwrap();
    assert_eq!(again.unusable_found, 0, "refresh must be idempotent");
    assert_eq!(again.refreshed, 0);

    // Namespace scoping is real: a sweep of another namespace sees nothing
    // here, and the whole-store sweep still finds nothing to do.
    let other = db.refresh_embeddings(Some("does-not-exist"), true).unwrap();
    assert_eq!(other.scanned, 0, "namespace scope must actually filter");
}
