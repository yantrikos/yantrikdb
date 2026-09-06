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
        valid_from: None,
        valid_to: None,
    }
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
