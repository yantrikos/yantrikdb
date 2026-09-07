//! The bound extractor on a live engine: evidence spans and grounding on
//! the claims it mints, the refusal ledger it writes, and silver recall.

use crate::{StatedClaim, YantrikDB};

fn rec(db: &YantrikDB, text: &str) -> String {
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

fn stated(src: &str, rel: &str, dst: &str) -> StatedClaim {
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
fn minted_claims_carry_spans_and_the_bound_grounding() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    let text = "CT128 is the memory host. After the deploy, CT128 runs 0.19.0 for good.";
    let rid = rec(&db, text);
    db.reextract_claims(None, false).unwrap();
    let claims = db.get_claims("CT128", None).unwrap();
    let runs = claims
        .iter()
        .find(|c| c["rel_type"] == "runs" && c["dst"] == "0.19.0")
        .unwrap_or_else(|| panic!("runs claim minted from the LATER mention: {claims:?}"));
    assert_eq!(runs["grounding"], 2, "{runs}");
    assert_eq!(runs["source_memory_rid"], rid.as_str());
    // The span points at the second CT128, not the heading.
    let (s, e): (i64, i64) = {
        let conn = db.conn();
        conn.query_row(
            "SELECT span_start, span_end FROM claims WHERE claim_id = ?1",
            rusqlite::params![runs["claim_id"].as_str().unwrap()],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap()
    };
    assert_eq!(&text[s as usize..e as usize], "CT128 runs 0.19.0");
    assert!(s as usize > text.find("After").unwrap());
}

#[test]
fn the_ledger_records_why_a_trigger_did_not_bind() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    rec(
        &db,
        "PyPI (trusted publishing) → swarm ping core+server → core runs CT128 dogfood",
    );
    rec(
        &db,
        "Alice Moreau, an engineer from Berlin, works at Fennwick Labs.",
    );
    db.reextract_claims(None, false).unwrap();
    let counts = db.extraction_refusal_counts(None).unwrap();
    assert_eq!(
        counts.get("runs:lowercase_subject").copied(),
        Some(1),
        "{counts:?}"
    );
    assert_eq!(
        counts.get("works_at:subject_not_adjacent").copied(),
        Some(1),
        "{counts:?}"
    );
    let rows = db.extraction_refusals(None, 10).unwrap();
    let core = rows
        .iter()
        .find(|r| r.rel_type == "runs")
        .expect("runs refusal row");
    assert_eq!(
        (core.left_token.as_str(), core.right_token.as_str()),
        ("core", "CT128")
    );
    assert_eq!(core.extractor_version, "2.0");
    // No claim was minted for either — the ledger is where the miss went.
    assert!(db
        .get_claims("PyPI", None)
        .unwrap()
        .iter()
        .all(|c| c["rel_type"] != "runs"));
    assert!(db.get_claims("Berlin", None).unwrap().is_empty());
    // Re-running the heal rewrites, never duplicates.
    db.reextract_claims(None, false).unwrap();
    assert_eq!(db.extraction_refusal_counts(None).unwrap(), counts);
}

#[test]
fn silver_recall_scores_stated_claims_against_the_extractor() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    let r1 = rec(&db, "Alice Moreau works at Fennwick Labs and prefers Vim.");
    let r2 = rec(&db, "Bob Lin, an old friend from Lisbon, works at Globex.");
    let r3 = rec(&db, "Carol Vance works at Globex and lives in Berlin.");
    db.attach_claims(
        &r1,
        &[
            stated("Alice Moreau", "works_at", "Fennwick Labs"),
            stated("Alice Moreau", "prefers", "Vim"),
        ],
    )
    .unwrap();
    db.attach_claims(&r2, &[stated("Bob Lin", "works_at", "Globex")])
        .unwrap();
    db.attach_claims(&r3, &[stated("Carol Vance", "lives_in", "Berlin")])
        .unwrap();
    let report = db.extraction_silver_recall(None).unwrap();
    assert_eq!(report.stated_claims, 4, "{report:?}");
    assert_eq!(
        report.unsupported_relation, 1,
        "prefers is outside the pattern table: {report:?}"
    );
    assert_eq!(report.recovered, 2, "Alice and Carol re-derive: {report:?}");
    assert_eq!(
        report.missed_by_reason.get("subject_not_adjacent").copied(),
        Some(1),
        "Bob's appositive is the miss, with its reason: {report:?}"
    );
    assert!((report.recall - 2.0 / 3.0).abs() < 1e-9, "{report:?}");
}
