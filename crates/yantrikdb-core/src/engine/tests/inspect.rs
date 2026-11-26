//! Inspection reads for one memory (`engine::inspect`): the reads an
//! explorer needs and could previously get only from raw SQL.

use super::*;
#[cfg(feature = "bundled-embedder")]
use crate::{StatedClaim, STATED_CLAIM_EXTRACTOR};

fn test_key() -> [u8; 32] {
    let mut key = [0u8; 32];
    for (i, b) in key.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(7).wrapping_add(42);
    }
    key
}

fn record_plain(db: &YantrikDB, text: &str, meta: serde_json::Value) -> String {
    db.record(
        text,
        "episodic",
        0.5,
        0.0,
        604800.0,
        &meta,
        &vec_seed(1.0, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap()
}

// ── revision_history ────────────────────────────────────────────────

#[test]
fn revision_history_lists_prior_states_oldest_first() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = record_plain(&db, "original text", serde_json::json!({"k": "v"}));
    assert!(db.revision_history(&rid).unwrap().is_empty());

    db.correct(&rid, None, None, Some(0.9), None, "first")
        .unwrap();
    db.correct(&rid, None, None, Some(0.7), None, "second")
        .unwrap();

    let revs = db.revision_history(&rid).unwrap();
    assert_eq!(revs.len(), 2);
    assert_eq!(revs[0].revision_num, 1);
    assert_eq!(revs[0].reason, "first");
    assert_eq!(revs[0].rid, rid);
    assert_eq!(revs[0].prior_text, "original text");
    assert!(
        (revs[0].prior_importance - 0.5).abs() < 1e-9,
        "revision 1 archives the ORIGINAL importance"
    );
    assert_eq!(
        revs[0].prior_metadata["k"], "v",
        "prior metadata is parsed, not a string"
    );
    assert_eq!(revs[1].revision_num, 2);
    assert_eq!(revs[1].reason, "second");
    assert!(
        (revs[1].prior_importance - 0.9).abs() < 1e-9,
        "revision 2 archives the state revision 1 produced"
    );
    assert!(revs[0].applied_at <= revs[1].applied_at);
    assert!(!revs[0].revision_id.is_empty());
}

#[test]
fn revision_history_is_empty_for_unknown_rids() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    assert!(db.revision_history("no-such-rid").unwrap().is_empty());
}

#[test]
fn revision_history_decrypts_prior_state_on_encrypted_stores() {
    let key = test_key();
    let db = YantrikDB::new_encrypted(":memory:", 8, &key).unwrap();
    assert!(db.is_encrypted());
    let rid = record_plain(&db, "secret original", serde_json::json!({"topic": "enc"}));
    db.correct(&rid, None, None, Some(0.95), None, "bump")
        .unwrap();

    // The archived row is ciphertext; the API must hand back plaintext.
    let stored: String = db
        .conn()
        .query_row(
            "SELECT prior_text FROM record_revisions WHERE rid = ?1",
            rusqlite::params![rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_ne!(
        stored, "secret original",
        "revision rows are archived encrypted"
    );

    let revs = db.revision_history(&rid).unwrap();
    assert_eq!(revs.len(), 1);
    assert_eq!(revs[0].prior_text, "secret original");
    assert_eq!(revs[0].prior_metadata["topic"], "enc");
}

// ── claims_for_memory ───────────────────────────────────────────────

#[cfg(feature = "bundled-embedder")]
fn claim(src: &str, rel: &str, dst: &str) -> StatedClaim {
    StatedClaim {
        src: src.into(),
        rel_type: rel.into(),
        dst: dst.into(),
        polarity: 1,
        modality: None,
        valid_from: None,
        valid_to: None,
    }
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn claims_for_memory_returns_the_memory_s_stated_claims_with_status() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    let rid = db
        .record_text(
            "Pranab prefers Vim for editing Rust and reviews with Maria.",
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
    let other = db
        .record_text(
            "Unrelated note about the weather in Lisbon.",
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
    let report = db
        .attach_claims(
            &rid,
            &[
                claim("Pranab", "prefers", "Vim"),
                claim("Pranab", "reviews with", "Maria"),
            ],
        )
        .unwrap();
    assert_eq!(
        report.accepted.len(),
        2,
        "both claims are grounded in the text"
    );

    let claims = db.claims_for_memory(&rid).unwrap();
    assert_eq!(claims.len(), 2);
    for c in &claims {
        assert_eq!(c["source_memory_rid"], rid);
        assert_eq!(c["status_suggestion"], "active");
        assert_eq!(c["extractor"], STATED_CLAIM_EXTRACTOR);
        assert_eq!(c["polarity"], 1);
    }
    let dsts: std::collections::HashSet<String> = claims
        .iter()
        .map(|c| c["dst"].as_str().unwrap().to_string())
        .collect();
    assert!(dsts.contains("Vim") && dsts.contains("Maria"));

    // Scoped to the memory: another memory sees none of them.
    assert!(db.claims_for_memory(&other).unwrap().is_empty());
    assert!(db.claims_for_memory("no-such-rid").unwrap().is_empty());
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn claims_for_memory_marks_a_denied_claim_negative() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    let rid = db
        .record_text(
            "Pranab does not use Emacs for Rust.",
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
    let mut denied = claim("Pranab", "use", "Emacs");
    denied.polarity = -1;
    let report = db.attach_claims(&rid, &[denied]).unwrap();
    assert_eq!(report.accepted.len(), 1);
    let claims = db.claims_for_memory(&rid).unwrap();
    assert_eq!(claims.len(), 1);
    assert_eq!(claims[0]["polarity"], -1);
    assert_eq!(claims[0]["status_suggestion"], "negative");
}

// ── memory_entities ─────────────────────────────────────────────────

#[test]
fn memory_entities_lists_linked_names_sorted() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = record_plain(&db, "plain record", serde_json::json!({}));
    assert!(db.memory_entities(&rid).unwrap().is_empty());
    db.link_memory_entity(&rid, "Zeta Corp").unwrap();
    db.link_memory_entity(&rid, "Acme").unwrap();
    db.link_memory_entity(&rid, "Acme").unwrap(); // idempotent
    assert_eq!(db.memory_entities(&rid).unwrap(), vec!["Acme", "Zeta Corp"]);
    assert!(db.memory_entities("no-such-rid").unwrap().is_empty());
}
