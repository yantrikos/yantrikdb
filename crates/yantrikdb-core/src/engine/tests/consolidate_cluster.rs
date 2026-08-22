//! Caller-supplied consolidation (`consolidate_cluster`).
//!
//! The default `consolidate()` writes an extractive join carrying the
//! cluster's MEAN embedding. Measured on BEAM (2026-08-11) that costs
//! accuracy through EMBEDDING DILUTION: no content is lost (all texts are
//! joined), but N precise vectors become one average that matches no
//! specific query well. This path lets a caller supply a real synthesis and
//! — the load-bearing part — embeds it FROM ITS OWN TEXT.

use super::*;

struct FixedTestEmbedder;

impl crate::types::Embedder for FixedTestEmbedder {
    fn embed(
        &self,
        _text: &str,
    ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(vec_seed(7.0, 8))
    }

    fn dim(&self) -> usize {
        8
    }

    fn fingerprint(&self) -> Option<String> {
        Some("fixed-test-embedder-v1".to_string())
    }

    fn name(&self) -> Option<String> {
        Some("fixed-test-embedder".to_string())
    }
}

fn seed(db: &YantrikDB, texts: &[&str]) -> Vec<String> {
    texts
        .iter()
        .enumerate()
        .map(|(i, t)| {
            db.record(
                t,
                "episodic",
                0.5,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(1.0 + i as f32 * 0.01, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap()
        })
        .collect()
}

fn durable_write_counts(db: &YantrikDB) -> (i64, i64, i64, i64) {
    let conn = db.conn();
    (
        conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .unwrap(),
        conn.query_row("SELECT COUNT(*) FROM oplog", [], |row| row.get(0))
            .unwrap(),
        conn.query_row("SELECT COUNT(*) FROM idempotency_claims", [], |row| {
            row.get(0)
        })
        .unwrap(),
        conn.query_row("SELECT COUNT(*) FROM consolidation_members", [], |row| {
            row.get(0)
        })
        .unwrap(),
    )
}

/// The point of the feature: the consolidated record's vector must describe
/// the SYNTHESIS, not the average of the members it replaced.
#[test]
fn caller_text_is_embedded_from_itself_not_the_cluster_mean() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["alpha one", "alpha two", "alpha three"]);
    let members: Vec<String> = rids.clone();

    let out = crate::consolidate::consolidate_cluster(
        &db,
        &members,
        "the alpha project shipped",
        Some(&vec_seed(9.0, 8)),
    )
    .unwrap();
    assert_eq!(out["cluster_size"], 3);
    assert_eq!(
        out["embedded_from_text"], false,
        "flag reports engine-side embedding; this call supplied its own vector"
    );

    let new_rid = out["consolidated_rid"].as_str().unwrap();
    let stored: String = db
        .conn()
        .query_row(
            "SELECT text FROM memories WHERE rid = ?1",
            params![new_rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        db.decrypt_text(&stored).unwrap(),
        "the alpha project shipped",
        "the caller owns the text; the engine must not rewrite it"
    );
}

/// Bookkeeping must match the default path exactly — sources retired,
/// membership recorded — or the two paths drift and replication diverges.
#[test]
fn sources_are_retired_and_membership_recorded() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["beta one", "beta two"]);
    let out = crate::consolidate::consolidate_cluster(
        &db,
        &rids,
        "beta summary",
        Some(&vec_seed(9.0, 8)),
    )
    .unwrap();
    let new_rid = out["consolidated_rid"].as_str().unwrap();

    for rid in &rids {
        let (status, into): (String, Option<String>) = db
            .conn()
            .query_row(
                "SELECT consolidation_status, consolidated_into FROM memories WHERE rid = ?1",
                params![rid],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(status, "consolidated", "source {rid} not retired");
        assert_eq!(into.as_deref(), Some(new_rid));
    }
    let members: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM consolidation_members WHERE consolidation_rid = ?1",
            params![new_rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(members, 2, "consolidation_members CRDT rows missing");
}

/// Double-consolidating a member would orphan the first consolidation's
/// membership rows, so it is refused rather than silently accepted.
#[test]
fn already_consolidated_members_are_refused() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["gamma one", "gamma two"]);
    crate::consolidate::consolidate_cluster(&db, &rids, "gamma summary", Some(&vec_seed(9.0, 8)))
        .unwrap();
    let err =
        crate::consolidate::consolidate_cluster(&db, &rids, "gamma again", Some(&vec_seed(9.0, 8)))
            .unwrap_err();
    assert!(
        format!("{err}").contains("not active"),
        "expected an active-status refusal, got: {err}"
    );
}

#[test]
fn empty_cluster_and_empty_text_are_refused() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    assert!(
        crate::consolidate::consolidate_cluster(&db, &[], "x", Some(&vec_seed(9.0, 8))).is_err()
    );
    let rids = seed(&db, &["delta one"]);
    let err = crate::consolidate::consolidate_cluster(&db, &rids, "   ", Some(&vec_seed(9.0, 8)))
        .unwrap_err();
    assert!(format!("{err}").contains("text is empty"), "{err}");
}

#[test]
fn unknown_rid_is_refused_before_anything_is_written() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["eps one"]);
    let mut with_bogus = rids.clone();
    with_bogus.push("does-not-exist".to_string());
    assert!(crate::consolidate::consolidate_cluster(
        &db,
        &with_bogus,
        "s",
        Some(&vec_seed(9.0, 8))
    )
    .is_err());
    let status: String = db
        .conn()
        .query_row(
            "SELECT consolidation_status FROM memories WHERE rid = ?1",
            params![rids[0]],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(status, "active", "a refused call must write nothing");
}

/// A caller-supplied vector of the wrong dimension must be refused before
/// anything is written — a diverged dim is undetectable until a query
/// silently misses.
#[test]
fn wrong_dimension_embedding_is_refused() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["zeta one"]);
    let err = crate::consolidate::consolidate_cluster(
        &db,
        &rids,
        "zeta summary",
        Some(&vec_seed(1.0, 4)),
    )
    .unwrap_err();
    assert!(format!("{err}").contains("consolidate_cluster"), "{err}");
    let status: String = db
        .conn()
        .query_row(
            "SELECT consolidation_status FROM memories WHERE rid = ?1",
            params![rids[0]],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(status, "active", "a refused call must write nothing");
}

/// The synthesis is an abstraction of a SPAN, and its event time must be
/// the span's end — not the wall clock of whenever the maintenance pass
/// happened to run. Stamping now() made every summary invisible to
/// `recall_as_of`/time-window reads on a backdated corpus (historical
/// import, BEAM), which is exactly where the abstraction earns its keep.
#[test]
fn summary_created_at_is_the_span_end_not_the_wall_clock() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids: Vec<String> = [(1_000.0, "iota one"), (2_000.0, "iota two")]
        .iter()
        .map(|(ts, t)| {
            db.record_with_idempotency(
                t,
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
                None,
                Some(*ts),
            )
            .unwrap()
        })
        .collect();

    let result =
        crate::consolidate::summarize_cluster(&db, &rids, "iota summary", Some(&vec_seed(5.0, 8)))
            .unwrap();
    let summary_rid = result["consolidated_rid"].as_str().unwrap().to_string();
    let created_at: f64 = db
        .conn()
        .query_row(
            "SELECT created_at FROM memories WHERE rid = ?1",
            params![summary_rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        created_at, 2_000.0,
        "summary event time must be the newest member's, not now()"
    );
}

#[test]
fn record_synthesis_persists_dual_clocks_and_grounded_provenance() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid_a = db
        .record_with_idempotency(
            "Bryan shared storytelling advice",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({
                "first_mention_at": 500.0,
                "evidence_ids": ["raw-a"],
            }),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
            None,
            Some(1_000.0),
        )
        .unwrap();
    let rid_b = db
        .record_with_idempotency(
            "Shawn expanded on the storytelling impact",
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
            None,
            Some(2_000.0),
        )
        .unwrap();

    let out = crate::consolidate::record_synthesis(
        &db,
        &[rid_b.clone(), rid_a.clone()],
        "Storytelling input from Bryan and Shawn",
        Some(&vec_seed(3.0, 8)),
        "contributed",
        "atomic",
        &serde_json::json!({
            "custom": "kept",
            "first_mention_at": 9_999.0,
            "evidence_ids": ["forged"],
            "synthesis_axis": "forged",
        }),
        "synth:contributed:storytelling-v1",
    )
    .unwrap();
    let synthesis_rid = out["consolidated_rid"].as_str().unwrap();
    let memory = db.get_memory(synthesis_rid).unwrap().unwrap();

    assert_eq!(
        memory.created_at, 2_000.0,
        "availability is newest evidence"
    );
    assert_eq!(memory.source, "inference");
    assert_eq!(
        memory.metadata["first_mention_at"], 1_000.0,
        "caller metadata cannot forge the engine-grounded evidence clock"
    );
    assert_eq!(memory.metadata["evidence_span_end_at"], 2_000.0);
    assert_eq!(memory.metadata["synthesis_available_at"], 2_000.0);
    assert_eq!(memory.metadata["synthesis_axis"], "contributed");
    assert_eq!(memory.metadata["granularity"], "atomic");
    assert_eq!(memory.metadata["custom"], "kept");
    let evidence: std::collections::HashSet<&str> = memory.metadata["evidence_ids"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect();
    assert_eq!(
        evidence,
        std::collections::HashSet::from([rid_a.as_str(), rid_b.as_str()])
    );
    let descriptor: (String, String, String, String, String) = db
        .conn()
        .query_row(
            "SELECT synthesis_axis, synthesis_granularity, synthesis_logical_key, \
                    synthesis_evidence_version, synthesis_state \
             FROM memories WHERE rid = ?1",
            rusqlite::params![synthesis_rid],
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
    assert_eq!(descriptor.0, "contributed");
    assert_eq!(descriptor.1, "atomic");
    assert_eq!(descriptor.2, "synth:contributed:storytelling-v1");
    assert_eq!(descriptor.4, "verified");
    assert_eq!(descriptor.3.len(), 64);
    let dependency_count: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM synthesis_dependencies WHERE synthesis_rid = ?1",
            rusqlite::params![synthesis_rid],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(dependency_count, 2);
    for source_rid in [&rid_a, &rid_b] {
        assert_eq!(
            db.get_memory(source_rid)
                .unwrap()
                .unwrap()
                .consolidation_status,
            "active"
        );
    }

    let before_available = db
        .recall_as_of(&vec_seed(3.0, 8), 10, 1_500.0, None, None)
        .unwrap();
    assert!(
        !before_available
            .iter()
            .any(|memory| memory.rid == synthesis_rid),
        "first mention must not make a synthesis visible before all evidence existed"
    );
    let after_available = db
        .recall_as_of(&vec_seed(3.0, 8), 10, 2_500.0, None, None)
        .unwrap();
    assert!(
        after_available
            .iter()
            .any(|memory| memory.rid == synthesis_rid),
        "synthesis must become visible at its evidence-span end"
    );
}

#[test]
fn record_synthesis_retry_is_idempotent_and_changed_output_conflicts() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["one source", "another source"]);
    let write = |text: &str, source_rids: &[String]| {
        crate::consolidate::record_synthesis(
            &db,
            source_rids,
            text,
            Some(&vec_seed(4.0, 8)),
            "asked",
            "atomic",
            &serde_json::json!({"generator": "test-v1"}),
            "synth:asked:item-1",
        )
    };

    let first = write("The user asked about source setup", &rids).unwrap();
    let reversed = [rids[1].clone(), rids[0].clone()];
    let retry = write("The user asked about source setup", &reversed).unwrap();
    assert_eq!(first["consolidated_rid"], retry["consolidated_rid"]);
    assert_eq!(first["idempotent_replay"], false);
    assert_eq!(retry["idempotent_replay"], true);

    let memory_count: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
        .unwrap();
    let member_count: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM consolidation_members", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(memory_count, 3, "retry created a duplicate synthesis");
    assert_eq!(member_count, 2, "retry duplicated provenance members");

    let error = write("A different stochastic interpretation", &rids).unwrap_err();
    assert!(
        format!("{error}")
            .to_ascii_lowercase()
            .contains("idempotency"),
        "changed output must fail as an idempotency conflict: {error}"
    );
}

#[test]
fn record_synthesis_new_evidence_supersedes_the_previous_logical_generation() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["first observation", "later observation"]);
    let write = |text: &str, source_rids: &[String]| {
        crate::consolidate::record_synthesis(
            &db,
            source_rids,
            text,
            Some(&vec_seed(4.0, 8)),
            "topic",
            "rollup",
            &serde_json::json!({"generator": "test-v1"}),
            "synth:topic:logical-item-1",
        )
    };

    let first = write("The topic began with one observation", &rids[..1]).unwrap();
    let first_rid_owned = first["consolidated_rid"].as_str().unwrap().to_string();
    let dependent = crate::consolidate::record_synthesis(
        &db,
        std::slice::from_ref(&first_rid_owned),
        "A higher-level summary based on the first topic generation",
        Some(&vec_seed(5.0, 8)),
        "summary",
        "rollup",
        &serde_json::json!({"generator": "test-v1"}),
        "synth:summary:dependent-1",
    )
    .unwrap();
    let second = write("The topic now includes a later observation", &rids).unwrap();
    let first_rid = first["consolidated_rid"].as_str().unwrap();
    let second_rid = second["consolidated_rid"].as_str().unwrap();
    assert_ne!(
        first_rid, second_rid,
        "changed evidence is a new generation"
    );

    let states: Vec<(String, String)> = {
        let conn = db.conn();
        let mut stmt = conn
            .prepare(
                "SELECT rid, synthesis_state FROM memories \
                 WHERE synthesis_logical_key = 'synth:topic:logical-item-1' \
                 ORDER BY rid",
            )
            .unwrap();
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap();
        rows.collect::<std::result::Result<_, _>>().unwrap()
    };
    assert_eq!(states.len(), 2);
    assert!(states.contains(&(first_rid.to_string(), "superseded".to_string())));
    assert!(states.contains(&(second_rid.to_string(), "verified".to_string())));
    let dependent_state: String = db
        .conn()
        .query_row(
            "SELECT synthesis_state FROM memories WHERE rid = ?1",
            params![dependent["consolidated_rid"].as_str().unwrap()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        dependent_state, "invalidated",
        "rollups based on a retired generation must not remain current"
    );

    let hits = db
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
            None,
            None,
            None,
            None,
            false,
        )
        .unwrap();
    assert!(!hits.iter().any(|memory| memory.rid == first_rid));
    assert!(hits.iter().any(|memory| memory.rid == second_rid));

    let audit = db.audit_synthesis_evidence(None, 10).unwrap();
    assert_eq!(audit.duplicate_logical_key_group_count, 0);

    let retry = write("The topic now includes a later observation", &rids).unwrap();
    assert_eq!(retry["consolidated_rid"], second["consolidated_rid"]);
    assert_eq!(retry["idempotent_replay"], true);
}

#[test]
fn synthesis_fanout_cap_admits_boundary_refuses_next_and_rolls_back() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.set_synthesis_fanout_cap(2).unwrap();
    let source_rid = seed(&db, &["one authoritative source"]).remove(0);
    let write = |logical_key: &str, text: &str| {
        crate::consolidate::record_synthesis(
            &db,
            std::slice::from_ref(&source_rid),
            text,
            Some(&vec_seed(4.0, 8)),
            "asked",
            "atomic",
            &empty_meta(),
            logical_key,
        )
    };

    let first = write("synth:cap:item-1", "first generation").unwrap();
    write("synth:cap:item-2", "second generation").unwrap();
    let retry = write("synth:cap:item-1", "first generation").unwrap();
    assert_eq!(first["consolidated_rid"], retry["consolidated_rid"]);
    assert_eq!(retry["idempotent_replay"], true);
    assert_eq!(
        db.stats(None).unwrap().synthesis_fanout_refused_since_boot,
        0
    );

    let memories_before: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
        .unwrap();
    let dependencies_before: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM synthesis_dependencies", [], |row| {
            row.get(0)
        })
        .unwrap();
    let claims_before: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM idempotency_claims", [], |row| {
            row.get(0)
        })
        .unwrap();
    let operations_before: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM oplog", [], |row| row.get(0))
        .unwrap();

    let error = write("synth:cap:item-3", "must be refused").unwrap_err();
    assert!(matches!(
        error,
        crate::error::YantrikDbError::SynthesisFanoutLimit {
            source_rid: ref rejected_source,
            current: 2,
            limit: 2,
        } if rejected_source == &source_rid
    ));

    let conn = db.conn();
    assert_eq!(
        conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        memories_before
    );
    assert_eq!(
        conn.query_row("SELECT COUNT(*) FROM synthesis_dependencies", [], |row| {
            row.get::<_, i64>(0)
        })
        .unwrap(),
        dependencies_before
    );
    assert_eq!(
        conn.query_row("SELECT COUNT(*) FROM idempotency_claims", [], |row| {
            row.get::<_, i64>(0)
        })
        .unwrap(),
        claims_before
    );
    assert_eq!(
        conn.query_row("SELECT COUNT(*) FROM oplog", [], |row| row.get::<_, i64>(0))
            .unwrap(),
        operations_before
    );
    drop(conn);

    let stats = db.stats(None).unwrap();
    assert_eq!(stats.synthesis_fanout_cap, 2);
    assert_eq!(stats.synthesis_fanout_refused_since_boot, 1);
    assert_eq!(stats.synthesis_fanout_current_high_water, 2);
    assert_eq!(stats.synthesis_fanout_sources_at_cap, 1);
    assert_eq!(stats.synthesis_fanout_sources_over_cap, 0);

    db.correct(
        &source_rid,
        None,
        Some(&serde_json::json!({"corrected": true})),
        None,
        None,
        "advance evidence revision",
    )
    .unwrap();
    assert_eq!(
        db.stats(None).unwrap().synthesis_fanout_current_high_water,
        0
    );
    write("synth:cap:item-4", "new evidence generation").unwrap();
    assert_eq!(
        db.stats(None).unwrap().synthesis_fanout_current_high_water,
        1
    );
}

#[test]
fn synthesis_fanout_cap_is_positive_and_persists() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fanout.db");
    let path = path.to_str().unwrap();
    {
        let db = YantrikDB::new(path, 8).unwrap();
        assert_eq!(
            db.synthesis_fanout_cap().unwrap(),
            crate::engine::DEFAULT_SYNTHESIS_FANOUT_CAP
        );
        assert!(matches!(
            db.set_synthesis_fanout_cap(0),
            Err(crate::error::YantrikDbError::InvalidInput(_))
        ));
        db.set_synthesis_fanout_cap(7).unwrap();
        assert_eq!(db.synthesis_fanout_cap().unwrap(), 7);
    }
    let reopened = YantrikDB::new(path, 8).unwrap();
    assert_eq!(reopened.synthesis_fanout_cap().unwrap(), 7);
    assert_eq!(reopened.stats(None).unwrap().synthesis_fanout_cap, 7);
}

#[test]
fn synthesis_fanout_cap_changes_reach_already_open_handles() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fanout-shared.db");
    let path = path.to_str().unwrap();
    let controller = YantrikDB::new(path, 8).unwrap();
    let writer = YantrikDB::new(path, 8).unwrap();

    controller.set_synthesis_fanout_cap(1).unwrap();
    assert_eq!(writer.synthesis_fanout_cap().unwrap(), 1);

    let source_rid = seed(&writer, &["shared-handle evidence"]).remove(0);
    crate::consolidate::record_synthesis(
        &writer,
        std::slice::from_ref(&source_rid),
        "first item",
        Some(&vec_seed(5.0, 8)),
        "asked",
        "atomic",
        &empty_meta(),
        "synth:shared-cap:first",
    )
    .unwrap();
    let error = crate::consolidate::record_synthesis(
        &writer,
        std::slice::from_ref(&source_rid),
        "second item",
        Some(&vec_seed(6.0, 8)),
        "asked",
        "atomic",
        &empty_meta(),
        "synth:shared-cap:second",
    )
    .unwrap_err();
    assert!(matches!(
        error,
        crate::error::YantrikDbError::SynthesisFanoutLimit {
            current: 1,
            limit: 1,
            ..
        }
    ));
}

#[test]
fn record_synthesis_defers_without_side_effects_during_reembed() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let source_rids = seed(&db, &["one source", "another source"]);
    let before = durable_write_counts(&db);

    db.write_router.switch_to_queueing();
    let result = crate::consolidate::record_synthesis(
        &db,
        &source_rids,
        "a synthesis that must wait for re-embedding",
        Some(&vec_seed(4.0, 8)),
        "asked",
        "atomic",
        &serde_json::json!({"generator": "test-v1"}),
        "synth:asked:queue-deferral",
    );
    db.write_router.switch_to_normal();

    assert!(matches!(
        result,
        Err(crate::error::YantrikDbError::ConsolidationDeferredDuringReembed)
    ));
    assert_eq!(
        durable_write_counts(&db),
        before,
        "deferral must not change durable tables"
    );
    for source_rid in &source_rids {
        let source = db.get_memory(source_rid).unwrap().unwrap();
        assert_eq!(source.consolidation_status, "active");
        assert!(source.consolidated_into.is_none());
    }
}

#[test]
fn engine_embedded_synthesis_deferral_does_not_stamp_identity() {
    let mut db = YantrikDB::new(":memory:", 8).unwrap();
    db.set_embedder(Box::new(FixedTestEmbedder)).unwrap();
    let source_rids = seed(&db, &["one source", "another source"]);
    let before_counts = durable_write_counts(&db);
    let before_identity = db.embedder_identity().unwrap();

    db.write_router.switch_to_queueing();
    let result = crate::consolidate::record_synthesis(
        &db,
        &source_rids,
        "an engine-embedded synthesis that must wait",
        None,
        "asked",
        "atomic",
        &empty_meta(),
        "synth:asked:embedded-queue-deferral",
    );
    db.write_router.switch_to_normal();

    assert!(matches!(
        result,
        Err(crate::error::YantrikDbError::ConsolidationDeferredDuringReembed)
    ));
    assert_eq!(durable_write_counts(&db), before_counts);
    assert_eq!(
        db.embedder_identity().unwrap(),
        before_identity,
        "a deferred synthesis must not stamp durable embedder identity"
    );
}

#[test]
fn synthesis_rollup_inherits_first_mention_and_flattens_raw_evidence() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let raw: Vec<String> = [(1_000.0, "first detail"), (2_000.0, "second detail")]
        .iter()
        .map(|(ts, text)| {
            db.record_with_idempotency(
                text,
                "episodic",
                0.5,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(*ts as f32, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
                None,
                Some(*ts),
            )
            .unwrap()
        })
        .collect();
    let child = crate::consolidate::record_synthesis(
        &db,
        &raw,
        "fine child",
        Some(&vec_seed(5.0, 8)),
        "contributed",
        "atomic",
        &empty_meta(),
        "synth:child",
    )
    .unwrap();
    let child_rid = child["consolidated_rid"].as_str().unwrap().to_string();
    let rollup = crate::consolidate::record_synthesis(
        &db,
        &[child_rid],
        "coarse rollup",
        Some(&vec_seed(6.0, 8)),
        "contributed",
        "rollup",
        &empty_meta(),
        "synth:rollup",
    )
    .unwrap();
    let rollup_memory = db
        .get_memory(rollup["consolidated_rid"].as_str().unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(rollup_memory.metadata["first_mention_at"], 1_000.0);
    assert_eq!(rollup_memory.metadata["evidence_span_end_at"], 2_000.0);
    let evidence: std::collections::HashSet<&str> = rollup_memory.metadata["evidence_ids"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect();
    assert_eq!(
        evidence,
        raw.iter().map(String::as_str).collect(),
        "rollup provenance must point to raw evidence, not only the child"
    );
}

#[test]
fn recall_prefers_matching_synthesis_representation_and_survives_reopen() {
    fn recall_for(db: &YantrikDB, query: &str) -> Vec<RecallResult> {
        db.recall(
            &vec_seed(5.0, 8),
            10,
            None,
            None,
            false,
            false,
            Some(query),
            true,
            None,
            None,
            None,
            None,
            None,
            false,
        )
        .unwrap()
    }

    fn assert_preference(
        db: &YantrikDB,
        query: &str,
        expected_rid: &str,
        other_rids: &[&str],
        expected_reason: &str,
    ) {
        let results = recall_for(db, query);
        let rank = |rid: &str| {
            results
                .iter()
                .position(|result| result.rid == rid)
                .unwrap_or_else(|| panic!("{rid} missing from recall"))
        };
        for other in other_rids {
            assert!(
                rank(expected_rid) < rank(other),
                "expected {expected_rid} before {other} for {query:?}: {:?}",
                results
                    .iter()
                    .map(|result| (&result.rid, result.score, &result.why_retrieved))
                    .collect::<Vec<_>>()
            );
        }
        let expected = results
            .iter()
            .find(|result| result.rid == expected_rid)
            .unwrap();
        assert!(
            expected
                .why_retrieved
                .iter()
                .any(|reason| reason == expected_reason),
            "missing {expected_reason:?}: {:?}",
            expected.why_retrieved
        );
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("representation-ranking.db");
    let path = path.to_str().unwrap();
    let (atomic_contributed, rollup_contributed, atomic_asked);
    {
        let db = YantrikDB::new(path, 8).unwrap();
        let source_rids = seed(&db, &["raw evidence alpha", "raw evidence beta"]);
        let synth = |axis: &str, granularity: &str, key: &str| {
            crate::consolidate::record_synthesis(
                &db,
                &source_rids,
                "shared project chronology",
                Some(&vec_seed(5.0, 8)),
                axis,
                granularity,
                &empty_meta(),
                key,
            )
            .unwrap()["consolidated_rid"]
                .as_str()
                .unwrap()
                .to_string()
        };
        atomic_contributed = synth("contributed", "atomic", "synth:representation:atomic");
        rollup_contributed = synth("contributed", "rollup", "synth:representation:rollup");
        atomic_asked = synth("asked", "atomic", "synth:representation:asked");

        assert_preference(
            &db,
            "List the order in which I brought up the shared project chronology",
            &atomic_contributed,
            &[&rollup_contributed, &atomic_asked],
            "representation_match:axis=contributed",
        );
        assert_preference(
            &db,
            "Give me an overview of the shared project chronology",
            &rollup_contributed,
            &[&atomic_contributed, &atomic_asked],
            "representation_match:granularity=rollup",
        );
        assert_preference(
            &db,
            "What did I ask about the shared project chronology?",
            &atomic_asked,
            &[&atomic_contributed, &rollup_contributed],
            "representation_match:axis=asked",
        );

        let neutral = recall_for(&db, "shared project chronology");
        let synthesis_results: Vec<&RecallResult> = neutral
            .iter()
            .filter(|result| {
                [
                    atomic_contributed.as_str(),
                    rollup_contributed.as_str(),
                    atomic_asked.as_str(),
                ]
                .contains(&result.rid.as_str())
            })
            .collect();
        assert_eq!(synthesis_results.len(), 3);
        assert!(synthesis_results.iter().all(|result| result
            .why_retrieved
            .iter()
            .all(|reason| !reason.starts_with("representation_match:"))));
        assert!(synthesis_results
            .windows(2)
            .all(|pair| (pair[0].score - pair[1].score).abs() < 1e-12));

        let conflict = recall_for(
            &db,
            "Summarize the shared project chronology items as a list",
        );
        let conflict_syntheses: Vec<&RecallResult> = conflict
            .iter()
            .filter(|result| {
                [
                    atomic_contributed.as_str(),
                    rollup_contributed.as_str(),
                    atomic_asked.as_str(),
                ]
                .contains(&result.rid.as_str())
            })
            .collect();
        assert_eq!(conflict_syntheses.len(), 3);
        assert!(conflict_syntheses.iter().all(|result| result
            .why_retrieved
            .iter()
            .all(|reason| !reason.starts_with("representation_match:granularity="))));
        assert!(conflict_syntheses
            .windows(2)
            .all(|pair| (pair[0].score - pair[1].score).abs() < 1e-12));
    }

    // Cold-open cache hydration must preserve the typed columns too.
    let reopened = YantrikDB::new(path, 8).unwrap();
    assert_preference(
        &reopened,
        "List the order in which I brought up the shared project chronology",
        &atomic_contributed,
        &[&rollup_contributed, &atomic_asked],
        "representation_match:granularity=atomic",
    );
}

#[test]
fn synthesis_generations_invalidate_on_source_correction_and_forget() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["first source", "second source"]);
    let write = || {
        crate::consolidate::record_synthesis(
            &db,
            &rids,
            "grounded item",
            Some(&vec_seed(7.0, 8)),
            "asked",
            "atomic",
            &empty_meta(),
            "synth:lifecycle:item-1",
        )
        .unwrap()
    };

    let first = write();
    let first_rid = first["consolidated_rid"].as_str().unwrap().to_string();
    assert_eq!(
        db.stats(None).unwrap().synthesis_fanout_current_high_water,
        1
    );
    db.correct(
        &rids[0],
        None,
        Some(&serde_json::json!({"corrected": true})),
        None,
        None,
        "source evidence changed",
    )
    .unwrap();
    assert_eq!(
        db.stats(None).unwrap().synthesis_fanout_current_high_water,
        0,
        "source correction must release current fan-out"
    );

    let first_state: String = db
        .conn()
        .query_row(
            "SELECT synthesis_state FROM memories WHERE rid = ?1",
            rusqlite::params![first_rid],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(first_state, "invalidated");
    assert!(
        db.recall_as_of(
            &vec_seed(7.0, 8),
            10,
            crate::time::now_secs() + 1.0,
            None,
            None,
        )
        .unwrap()
        .iter()
        .all(|memory| memory.rid != first_rid),
        "invalidated synthesis must be filtered from recall"
    );

    let second = write();
    let second_rid = second["consolidated_rid"].as_str().unwrap().to_string();
    assert_ne!(
        first_rid, second_rid,
        "a new evidence revision needs a new generation"
    );
    assert_eq!(
        db.stats(None).unwrap().synthesis_fanout_current_high_water,
        1,
        "the replacement generation must consume current fan-out"
    );
    assert!(db.forget(&rids[1]).unwrap());
    assert_eq!(
        db.stats(None).unwrap().synthesis_fanout_current_high_water,
        0,
        "source forget must release current fan-out"
    );
    let second_state: String = db
        .conn()
        .query_row(
            "SELECT synthesis_state FROM memories WHERE rid = ?1",
            rusqlite::params![second_rid],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(second_state, "invalidated");
    assert!(
        db.recall_as_of(
            &vec_seed(7.0, 8),
            10,
            crate::time::now_secs() + 1.0,
            None,
            None,
        )
        .unwrap()
        .iter()
        .all(|memory| memory.rid != second_rid),
        "forget must invalidate and filter dependent syntheses"
    );
}

/// Namespaces are tenant boundaries: a cluster spanning two of them would
/// store tenant B's synthesized content under tenant A's namespace. The
/// incoherent input is refused before anything is written.
#[test]
fn cross_namespace_cluster_is_refused() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid_a = seed(&db, &["kappa one"]).remove(0);
    let rid_b = db
        .record(
            "kappa two",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(2.0, 8),
            "tenant_b",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    let err = crate::consolidate::consolidate_cluster(
        &db,
        &[rid_a.clone(), rid_b],
        "kappa summary",
        Some(&vec_seed(3.0, 8)),
    )
    .unwrap_err();
    assert!(
        format!("{err}").contains("namespace"),
        "refusal must name the boundary: {err}"
    );
    let status: String = db
        .conn()
        .query_row(
            "SELECT consolidation_status FROM memories WHERE rid = ?1",
            params![rid_a],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(status, "active", "a refused call must write nothing");
}

/// `get_memory` — the rid point-read the binding lacked entirely.
/// Found while wiring LLM conflict resolution: `get_conflicts` hands back
/// `memory_a`/`memory_b` as bare rids and there was no way to resolve them
/// to text short of paging the namespace.
#[test]
fn get_memory_reads_by_rid_including_consolidated() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["theta one", "theta two"]);

    let m = db.get_memory(&rids[0]).unwrap().expect("memory must exist");
    assert_eq!(m.text, "theta one");
    assert_eq!(m.consolidation_status, "active");
    assert!(db.get_memory("no-such-rid").unwrap().is_none());

    // Naming a rid asks for THAT record — currency is not the caller's
    // question here, so a consolidated source must still be readable.
    crate::consolidate::consolidate_cluster(&db, &rids, "theta summary", Some(&vec_seed(9.0, 8)))
        .unwrap();
    let after = db.get_memory(&rids[0]).unwrap().expect("still readable");
    assert_eq!(after.text, "theta one");
    assert_eq!(after.consolidation_status, "consolidated");
}
