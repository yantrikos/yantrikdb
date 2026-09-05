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
