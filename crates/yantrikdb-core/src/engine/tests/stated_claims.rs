//! Cooperative claims: the writer states, the engine grounds.

use crate::{StatedClaim, YantrikDB, STATED_CLAIM_EXTRACTOR};

fn db_with(text: &str) -> (YantrikDB, String) {
    let db = YantrikDB::with_default(":memory:").unwrap();
    let rid = db
        .record_text(
            text,
            "semantic",
            0.5,
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
    (db, rid)
}

fn claim(src: &str, rel: &str, dst: &str) -> StatedClaim {
    StatedClaim {
        src: src.into(),
        rel_type: rel.into(),
        dst: dst.into(),
        polarity: 1,
        modality: None,
        span_start: None,
        span_end: None,
        valid_from: None,
        valid_to: None,
    }
}

fn claim_modal(src: &str, rel: &str, dst: &str, modality: &str) -> StatedClaim {
    StatedClaim {
        modality: Some(modality.into()),
        ..claim(src, rel, dst)
    }
}

fn record(db: &YantrikDB, text: &str) -> String {
    db.record_text(
        text,
        "semantic",
        0.5,
        0.0,
        604800.0,
        &serde_json::json!({}),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap()
}

#[test]
fn grounded_claims_are_stored_with_provenance_and_ungrounded_ones_are_refused() {
    let (db, rid) = db_with("Pranab prefers Vim for editing Rust and reviews with Maria.");
    let report = db
        .attach_claims(
            &rid,
            &[
                claim("Pranab", "prefers", "Vim"),
                claim("Pranab", "prefers", "Emacs"), // not in the text
                claim("Pranab", "reviews with", "Maria"),
                claim("Pranab", "prefers", "Vim"), // duplicate in batch
                claim("The", "leads", "Vim"),      // phantom subject
                claim("Pranab", "??", "Vim"),      // no relation token survives
            ],
        )
        .unwrap();
    let accepted: Vec<(&str, &str, &str)> = report
        .accepted
        .iter()
        .map(|a| (a.src.as_str(), a.rel_type.as_str(), a.dst.as_str()))
        .collect();
    assert_eq!(
        accepted,
        vec![
            ("Pranab", "prefers", "Vim"),
            ("Pranab", "reviews_with", "Maria")
        ]
    );
    let reasons: Vec<&str> = report.rejected.iter().map(|r| r.reason.as_str()).collect();
    assert_eq!(report.rejected.len(), 4, "{reasons:?}");
    assert!(
        reasons[0].contains("not grounded") && reasons[0].contains("Emacs"),
        "{reasons:?}"
    );
    assert!(reasons[1].contains("duplicate"), "{reasons:?}");
    assert!(reasons[2].contains("admissible"), "{reasons:?}");
    assert!(reasons[3].contains("relation"), "{reasons:?}");

    // Stored with provenance and the stated extractor label.
    let edges = db.get_edges("Pranab").unwrap();
    assert!(
        edges
            .iter()
            .any(|e| e.rel_type == "prefers" && e.dst == "Vim"),
        "{edges:?}"
    );
    let conn = db.conn();
    let (extractor, source): (String, Option<String>) = conn
        .query_row(
            "SELECT extractor, source_memory_rid FROM claims WHERE src='Pranab' AND rel_type='prefers' AND dst='Vim'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(extractor, STATED_CLAIM_EXTRACTOR);
    assert_eq!(source.as_deref(), Some(rid.as_str()));
}

#[test]
fn stated_claims_link_entities_synchronously() {
    // No drain, no think(): the writer's own claim closes the RYW gap for
    // exactly the entities it named.
    let (db, rid) = db_with("Pranab prefers Vim.");
    db.attach_claims(&rid, &[claim("Pranab", "prefers", "Vim")])
        .unwrap();
    let thread = db.recall_thread("default", &["Pranab"], 10).unwrap();
    assert_eq!(thread.total, 1);
    assert_eq!(thread.items[0].rid, rid);
}

#[test]
fn unknown_or_inactive_memory_is_an_error_not_a_silent_no_op() {
    let (db, rid) = db_with("Pranab prefers Vim.");
    assert!(db
        .attach_claims("no-such-rid", &[claim("Pranab", "prefers", "Vim")])
        .is_err());
    db.forget(&rid).unwrap();
    assert!(db
        .attach_claims(&rid, &[claim("Pranab", "prefers", "Vim")])
        .is_err());
}

#[test]
fn two_stated_preferences_about_one_subject_surface_a_conflict() {
    // `prefers` is on the preference whitelist but the extractor must never
    // mint it (multi-valued across domains). Two WRITER-STATED preferences
    // are a deliberate assertion pair, and the edge scan must be able to
    // tie each edge back to its memory through the claim's provenance.
    let (db, a) = db_with("Pranab prefers Vim as his editor.");
    db.attach_claims(&a, &[claim("Pranab", "prefers", "Vim")])
        .unwrap();
    let b = db
        .record_text(
            "Pranab prefers Emacs as his editor now.",
            "semantic",
            0.5,
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
    db.attach_claims(&b, &[claim("Pranab", "prefers", "Emacs")])
        .unwrap();
    let found = crate::conflict::scan_conflicts_limited(&db, 50).unwrap();
    let pair = found
        .iter()
        .find(|c| (c.memory_a == a && c.memory_b == b) || (c.memory_a == b && c.memory_b == a))
        .expect("stated preference pair must surface");
    let kind = format!("{:?}", pair.conflict_type).to_lowercase();
    assert!(kind.contains("preference"), "{kind}");
}

#[test]
fn oversized_batch_is_refused_whole() {
    let (db, rid) = db_with("Pranab prefers Vim.");
    let batch: Vec<StatedClaim> = (0..=crate::MAX_STATED_CLAIMS)
        .map(|_| claim("Pranab", "prefers", "Vim"))
        .collect();
    assert!(db.attach_claims(&rid, &batch).is_err());
}

/// The temporal tag: a record's event time flows into the claims it
/// states and the claims the extractor mints, and an explicit window on
/// a stated claim wins over it.
#[test]
fn claims_inherit_the_records_event_time_unless_the_writer_gives_a_window() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    let t_2024 = 1_704_067_200.0_f64; // 2024-01-01T00:00:00Z
    let rid = db
        .record_text(
            "Alice Moreau works at Fennwick Labs, and Alice Moreau lives in Berlin.",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({"event_time_min": t_2024, "event_time_max": t_2024}),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    db.apply_pending_ops_once(100).unwrap();
    let window = |src: &str, rel: &str, dst: &str| -> (Option<f64>, Option<f64>) {
        db.conn()
            .query_row(
                "SELECT valid_from, valid_to FROM claims WHERE src = ?1 AND rel_type = ?2                  AND dst = ?3 AND tombstoned = 0",
                rusqlite::params![src, rel, dst],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap()
    };
    // Extracted claim inherits the record's event time.
    assert_eq!(
        window("Alice Moreau", "works_at", "Fennwick Labs"),
        (Some(t_2024), None)
    );
    // Stated claim without a window inherits it too.
    let rep = db
        .attach_claims(
            &rid,
            &[StatedClaim {
                src: "Alice Moreau".into(),
                rel_type: "based_in".into(),
                dst: "Berlin".into(),
                ..StatedClaim::default()
            }],
        )
        .unwrap();
    assert_eq!(rep.accepted.len(), 1, "{rep:?}");
    assert_eq!(
        window("Alice Moreau", "based_in", "Berlin"),
        (Some(t_2024), None)
    );
    // An explicit window wins.
    let t_2025 = 1_735_689_600.0_f64;
    let rep = db
        .attach_claims(
            &rid,
            &[StatedClaim {
                src: "Alice Moreau".into(),
                rel_type: "visited".into(),
                dst: "Berlin".into(),
                valid_from: Some(t_2025),
                valid_to: Some(t_2025 + 86_400.0),
                ..StatedClaim::default()
            }],
        )
        .unwrap();
    assert_eq!(rep.accepted.len(), 1, "{rep:?}");
    assert_eq!(
        window("Alice Moreau", "visited", "Berlin"),
        (Some(t_2025), Some(t_2025 + 86_400.0))
    );
    // A record without a tag yields claims without a window.
    let rid2 = db
        .record_text(
            "Bob Lin works at Globex.",
            "semantic",
            0.5,
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
    db.apply_pending_ops_once(100).unwrap();
    let _ = rid2;
    assert_eq!(window("Bob Lin", "works_at", "Globex"), (None, None));
}

// ---------------------------------------------------------------------------
// Modality — the qualifier the cooperative path used to discard.
// ---------------------------------------------------------------------------

#[test]
fn quoted_claim_keeps_its_modality() {
    let (db, rid) = db_with("Vela said that Hugo leads the UNIT team.");
    let report = db
        .attach_claims(&rid, &[claim_modal("Hugo", "leads", "UNIT team", "quoted")])
        .unwrap();
    assert_eq!(report.accepted.len(), 1, "grounded quote must be accepted");

    let stored: String = db
        .conn()
        .query_row(
            "SELECT modality FROM claims WHERE source_memory_rid = ?1",
            rusqlite::params![rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(stored, "quoted", "the writer's modality must survive");
}

#[test]
fn unknown_modality_is_rejected() {
    let (db, rid) = db_with("Hugo leitet das UNIT Team.");
    let report = db
        .attach_claims(
            &rid,
            &[claim_modal("Hugo", "leads", "UNIT Team", "probably")],
        )
        .unwrap();
    assert!(report.accepted.is_empty());
    assert_eq!(report.rejected.len(), 1);
    assert!(
        report.rejected[0]
            .reason
            .contains("modality must be one of"),
        "got {:?}",
        report.rejected[0].reason
    );
}

#[test]
fn span_is_stored_as_the_local_part_of_the_observation() {
    // The span records WHERE in the text a statement sits, so a reader can
    // locate it. It does NOT let two statements coexist: the stored identity
    // key has no span, so a second claim with a different span is refused
    // rather than merged (see `two_statements_differing_only_by_span_...`).
    // REVIEW5 F1: the earlier comment here claimed the span kept the two
    // statements apart, which the code never did.
    let (db, rid) =
        db_with("Hugo leitet das UNIT Team. Spaeter sagt Vela, Hugo leitet das UNIT Team.");
    let mut c = claim("Hugo", "leads", "UNIT Team");
    c.span_start = Some(0);
    c.span_end = Some(24);
    db.attach_claims(&rid, &[c]).unwrap();

    let (s, e): (Option<i32>, Option<i32>) = db
        .conn()
        .query_row(
            "SELECT span_start, span_end FROM claims WHERE source_memory_rid = ?1",
            rusqlite::params![rid],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!((s, e), (Some(0), Some(24)));
}

#[test]
fn missing_span_stays_null_and_does_not_invent_a_location() {
    let (db, rid) = db_with("Hugo leitet das UNIT Team.");
    db.attach_claims(&rid, &[claim("Hugo", "leads", "UNIT Team")])
        .unwrap();
    let (s, e): (Option<i32>, Option<i32>) = db
        .conn()
        .query_row(
            "SELECT span_start, span_end FROM claims WHERE source_memory_rid = ?1",
            rusqlite::params![rid],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!((s, e), (None, None), "unknown locality must stay unknown");
}

#[test]
fn differing_modality_on_the_same_triple_is_refused_not_overwritten() {
    // The stored identity key is (src, dst, rel_type, extractor, polarity,
    // namespace) — it cannot hold both statements. Before this guard the
    // second attach silently replaced the first, qualifiers and all.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let asserted = record(&db, "Hugo leitet das UNIT Team.");
    let quoted = record(&db, "Vela sagte, Hugo leitet das UNIT Team.");

    db.attach_claims(&asserted, &[claim("Hugo", "leads", "UNIT Team")])
        .unwrap();
    let report = db
        .attach_claims(
            &quoted,
            &[claim_modal("Hugo", "leads", "UNIT Team", "quoted")],
        )
        .unwrap();

    assert!(
        report.accepted.is_empty(),
        "a quoted statement must not replace an asserted one"
    );
    assert_eq!(report.rejected.len(), 1);
    assert!(
        report.rejected[0].reason.contains("identity collision"),
        "got {:?}",
        report.rejected[0].reason
    );

    let stored: String = db
        .conn()
        .query_row(
            "SELECT modality FROM claims WHERE src = 'Hugo' AND dst = 'UNIT Team'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(stored, "asserted", "the first qualifier must be intact");
}

#[test]
fn batch_dedup_must_not_swallow_a_validity_collision() {
    // REVIEW4 G4, remaining half. The in-batch dedup key gained `modality`,
    // but not the validity window. Two statements about the same triple with
    // the same modality and different windows are two statements — they must
    // reach the identity guard and be refused as a collision there. Today the
    // second is discarded as a batch duplicate first, so the reason reported
    // to the writer is wrong and the guard never sees the case.
    let (db, rid) = db_with("Hugo leitete 2024 das UNIT Team. Hugo leitet 2026 das UNIT Team.");
    let mut first = claim("Hugo", "leads", "UNIT Team");
    first.valid_from = Some(1_704_067_200.0); // 2024-01-01
    let mut second = claim("Hugo", "leads", "UNIT Team");
    second.valid_from = Some(1_767_225_600.0); // 2026-01-01

    let report = db.attach_claims(&rid, &[first, second]).unwrap();

    assert_eq!(report.accepted.len(), 1, "{report:?}");
    assert_eq!(report.rejected.len(), 1, "{report:?}");
    assert!(
        report.rejected[0].reason.contains("identity collision"),
        "the second window must be refused by the guard, not dropped as a batch duplicate: {:?}",
        report.rejected[0].reason
    );
    let stored: Option<f64> = db
        .conn()
        .query_row(
            "SELECT valid_from FROM claims WHERE src = 'Hugo' AND dst = 'UNIT Team'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        stored,
        Some(1_704_067_200.0),
        "the first window must survive"
    );
}

fn record_at(db: &YantrikDB, text: &str, event_ts: f64) -> String {
    db.record_text(
        text,
        "semantic",
        0.5,
        0.0,
        604800.0,
        &serde_json::json!({"event_time_min": event_ts, "event_time_max": event_ts}),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap()
}

#[test]
fn a_different_validity_window_alone_is_a_collision() {
    // REVIEW4 G1: the guard compares three qualifiers, but only the modality
    // branch had a test. The window branch is the one most likely to fire in
    // real data, because a stated claim inherits the record's event time.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let t_2024 = 1_704_067_200.0_f64;
    let t_2026 = 1_767_225_600.0_f64;
    let a = record_at(&db, "Hugo leitet das UNIT Team.", t_2024);
    let b = record_at(&db, "Hugo leitet das UNIT Team.", t_2026);

    db.attach_claims(&a, &[claim("Hugo", "leads", "UNIT Team")])
        .unwrap();
    let report = db
        .attach_claims(&b, &[claim("Hugo", "leads", "UNIT Team")])
        .unwrap();

    assert_eq!(report.accepted.len(), 0, "{report:?}");
    assert_eq!(report.rejected.len(), 1, "{report:?}");
    assert!(
        report.rejected[0].reason.contains("identity collision"),
        "same modality, different window must still collide: {:?}",
        report.rejected[0].reason
    );
    let stored: Option<f64> = db
        .conn()
        .query_row(
            "SELECT valid_from FROM claims WHERE src = 'Hugo' AND dst = 'UNIT Team'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(stored, Some(t_2024), "the first window must survive");
}

#[test]
fn an_identical_resubmission_is_still_accepted() {
    // The guard must not lock a writer out of its own row: the same statement
    // resubmitted is a refinement, not a collision. Without this test the
    // refusal could silently turn every retry into a rejection.
    let (db, rid) = db_with("Hugo leitet das UNIT Team.");
    db.attach_claims(&rid, &[claim("Hugo", "leads", "UNIT Team")])
        .unwrap();
    let report = db
        .attach_claims(&rid, &[claim("Hugo", "leads", "UNIT Team")])
        .unwrap();

    assert_eq!(report.accepted.len(), 1, "{report:?}");
    assert!(report.rejected.is_empty(), "{report:?}");
    let rows: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM claims WHERE src = 'Hugo' AND dst = 'UNIT Team' AND tombstoned = 0",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(rows, 1, "an update must not create a second row");
}

#[test]
fn an_alias_folded_endpoint_is_detected_as_the_same_identity() {
    // REVIEW4 R4-2 / G2: the guard queries resolved names because it runs
    // inside `ingest_claim_grounded`, after `resolve_alias`. A claim written
    // under an alias must collide with the canonical row — otherwise the
    // UPSERT overwrites the canonical qualifiers through the alias door.
    let db = YantrikDB::with_default(":memory:").unwrap();
    db.add_entity_alias("Hugo Gundlach", "Hugo", "default", "test")
        .unwrap();

    let canonical = record(&db, "Hugo leitet das UNIT Team.");
    db.attach_claims(&canonical, &[claim("Hugo", "leads", "UNIT Team")])
        .unwrap();

    let through_alias = record(&db, "Hugo Gundlach leitet das UNIT Team.");
    let report = db
        .attach_claims(
            &through_alias,
            &[claim_modal("Hugo Gundlach", "leads", "UNIT Team", "quoted")],
        )
        .unwrap();

    assert_eq!(report.accepted.len(), 0, "{report:?}");
    assert_eq!(report.rejected.len(), 1, "{report:?}");
    assert!(
        report.rejected[0].reason.contains("identity collision"),
        "an alias endpoint must reach the same identity: {:?}",
        report.rejected[0].reason
    );
    let stored: String = db
        .conn()
        .query_row(
            "SELECT modality FROM claims WHERE src = 'Hugo' AND dst = 'UNIT Team'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(stored, "asserted", "the canonical qualifier must survive");

    let alias_rows: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM claims WHERE src = 'Hugo Gundlach'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(alias_rows, 0, "the claim row stores resolved endpoints");
}

#[test]
fn two_statements_differing_only_by_span_do_not_silently_merge() {
    // REVIEW5 F1. The comment on `span_is_stored_as_the_local_part_of_the_
    // observation` says the span keeps two statements in one memory apart. It
    // does not: the identity key has no span, so they cannot coexist. Until the
    // identity migration (§31.1) lands, the honest behaviour is to REFUSE the
    // second observation instead of accepting it as a "refinement" and
    // replacing the first one's span.
    let (db, rid) =
        db_with("Hugo leitet das UNIT Team. Spaeter sagt Vela, Hugo leitet das UNIT Team.");
    let mut first = claim("Hugo", "leads", "UNIT Team");
    first.span_start = Some(0);
    first.span_end = Some(24);
    let r1 = db.attach_claims(&rid, &[first]).unwrap();
    assert_eq!(r1.accepted.len(), 1, "{r1:?}");

    let mut second = claim("Hugo", "leads", "UNIT Team");
    second.span_start = Some(33);
    second.span_end = Some(57);
    let r2 = db.attach_claims(&rid, &[second]).unwrap();

    assert_eq!(r2.accepted.len(), 0, "{r2:?}");
    assert_eq!(r2.rejected.len(), 1, "{r2:?}");
    assert!(
        r2.rejected[0].reason.contains("identity collision"),
        "a second observation must be refused, not merged: {:?}",
        r2.rejected[0].reason
    );
    let (s, e): (Option<i32>, Option<i32>) = db
        .conn()
        .query_row(
            "SELECT span_start, span_end FROM claims WHERE src = 'Hugo' AND dst = 'UNIT Team'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!((s, e), (Some(0), Some(24)), "the first span must survive");
}

#[test]
fn a_stored_null_window_collides_with_an_inherited_one() {
    // REVIEW5 F4.1. A row written without a window and a statement whose
    // record carries an event time describe the same statement to a reader,
    // but `None != Some(t)`, so the second is refused. That is fail-closed on
    // purpose — the identity has no notion of "unknown vs inherited" yet. This
    // test pins the behaviour so a future change to that notion is visible.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let plain = record(&db, "Hugo leitet das UNIT Team.");
    let timed = record_at(&db, "Hugo leitet das UNIT Team.", 1_704_067_200.0);

    db.attach_claims(&plain, &[claim("Hugo", "leads", "UNIT Team")])
        .unwrap();
    let report = db
        .attach_claims(&timed, &[claim("Hugo", "leads", "UNIT Team")])
        .unwrap();

    assert_eq!(report.accepted.len(), 0, "{report:?}");
    assert_eq!(report.rejected.len(), 1, "{report:?}");
    assert!(
        report.rejected[0].reason.contains("identity collision"),
        "unknown and inherited are two states: {:?}",
        report.rejected[0].reason
    );
}

#[test]
fn a_resubmission_that_drops_the_span_is_refused() {
    // REVIEW6 C2/Q1. Under the one-observation rule a retry that omits the span
    // is a *different* statement (`None != Some(0)`), so it is refused where
    // the older code accepted it as a refinement. That is deliberate, but it is
    // a behaviour change — pinned here because the retry test only covers
    // None -> None, and a writer cannot see the difference until it is rejected.
    let (db, rid) = db_with("Hugo leitet das UNIT Team.");
    let mut first = claim("Hugo", "leads", "UNIT Team");
    first.span_start = Some(0);
    first.span_end = Some(24);
    db.attach_claims(&rid, &[first]).unwrap();

    let report = db
        .attach_claims(&rid, &[claim("Hugo", "leads", "UNIT Team")])
        .unwrap();

    assert_eq!(report.accepted.len(), 0, "{report:?}");
    assert_eq!(report.rejected.len(), 1, "{report:?}");
    assert!(
        report.rejected[0].reason.contains("identity collision"),
        "a retry without the span is a different statement: {:?}",
        report.rejected[0].reason
    );
}

#[test]
fn two_spans_in_one_batch_reach_the_guard() {
    // Same finding, batch path: the second statement must not be discarded as
    // an in-batch duplicate before the guard can refuse it.
    let (db, rid) =
        db_with("Hugo leitet das UNIT Team. Spaeter sagt Vela, Hugo leitet das UNIT Team.");
    let mut first = claim("Hugo", "leads", "UNIT Team");
    first.span_start = Some(0);
    first.span_end = Some(24);
    let mut second = claim("Hugo", "leads", "UNIT Team");
    second.span_start = Some(33);
    second.span_end = Some(57);

    let report = db.attach_claims(&rid, &[first, second]).unwrap();

    assert_eq!(report.accepted.len(), 1, "{report:?}");
    assert_eq!(report.rejected.len(), 1, "{report:?}");
    assert!(
        report.rejected[0].reason.contains("identity collision"),
        "the second span must reach the guard: {:?}",
        report.rejected[0].reason
    );
}
