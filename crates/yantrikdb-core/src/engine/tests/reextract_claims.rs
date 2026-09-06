//! The re-extraction heal: legacy extractor claims go, assertions stay.

use crate::{StatedClaim, YantrikDB};

fn rec(db: &YantrikDB, text: &str, ns: &str) -> String {
    db.record_text(
        text,
        "semantic",
        0.5,
        0.0,
        604800.0,
        &serde_json::json!({}),
        ns,
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap()
}

fn rels(db: &YantrikDB, extractor: &str) -> Vec<(String, String, String)> {
    let conn = db.conn();
    let mut stmt = conn
        .prepare("SELECT src, rel_type, dst FROM claims WHERE tombstoned = 0 AND extractor = ?1 ORDER BY src, rel_type, dst")
        .unwrap();
    stmt.query_map(rusqlite::params![extractor], |r| {
        Ok((r.get(0)?, r.get(1)?, r.get(2)?))
    })
    .unwrap()
    .map(|r| r.unwrap())
    .collect()
}

fn plant_legacy_junk(db: &YantrikDB, rid: &str, ns: &str) {
    // What the pre-anchoring extractor left behind: a verb bridging two
    // unrelated entities, labelled heuristic_v1 with real provenance.
    let conn = db.conn();
    conn.execute(
        "INSERT INTO claims (claim_id, src, dst, rel_type, weight, created_at, extractor, \
         source_memory_rid, namespace) VALUES ('legacy1', 'Pranab', 'UTC', 'leads', 1.0, 1.0, \
         'heuristic_v1', ?1, ?2)",
        rusqlite::params![rid, ns],
    )
    .unwrap();
}

#[test]
fn heal_replaces_extractor_claims_and_preserves_assertions() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    let a = rec(
        &db,
        "Alice Moreau works at Fennwick Labs as a data engineer.",
        "default",
    );
    let b = rec(
        &db,
        "Pranab confirmed the Materializer runs the loop every tick at UTC midnight.",
        "default",
    );
    db.apply_pending_ops_once(100).unwrap();
    plant_legacy_junk(&db, &b, "default");
    db.relate("Pranab", "Acme", "works_at", 1.0).unwrap();
    db.attach_claims(
        &a,
        &[StatedClaim {
            src: "Alice Moreau".into(),
            rel_type: "mentors".into(),
            dst: "Fennwick Labs".into(),
            polarity: 1,
        }],
    )
    .unwrap();
    let before = rels(&db, "heuristic_v1");
    assert!(
        before
            .iter()
            .any(|r| r == &("Pranab".into(), "leads".into(), "UTC".into())),
        "{before:?}"
    );

    let dry = db.reextract_claims(None, true).unwrap();
    assert_eq!((dry.claims_removed, dry.claims_written), (0, 0));
    assert!(
        rels(&db, "heuristic_v1").iter().any(|r| r.1 == "leads"),
        "dry run must touch nothing"
    );

    let report = db.reextract_claims(None, false).unwrap();
    assert_eq!(report.memories_scanned, 2);
    assert!(report.claims_removed >= 2, "{report:?}");
    let after = rels(&db, "heuristic_v1");
    assert!(
        !after.iter().any(|r| r.1 == "leads"),
        "legacy junk must be gone: {after:?}"
    );
    assert!(
        after.iter().any(|r| r
            == &(
                "Alice Moreau".into(),
                "works_at".into(),
                "Fennwick Labs".into()
            )),
        "anchored extraction must regenerate the real fact: {after:?}"
    );
    assert_eq!(report.claims_written, after.len());
    assert!(
        report.before_by_rel.contains_key("leads") && !report.after_by_rel.contains_key("leads")
    );
    // Assertions survive untouched.
    assert!(rels(&db, "manual")
        .iter()
        .any(|r| r == &("Pranab".into(), "works_at".into(), "Acme".into())));
    assert!(rels(&db, "agent_stated").iter().any(|r| r.1 == "mentors"));
    // Graph index no longer carries the deleted edge.
    assert!(!db
        .get_edges("UTC")
        .unwrap()
        .iter()
        .any(|e| e.rel_type == "leads"));
}

#[test]
fn heal_is_namespace_scoped() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    let a = rec(&db, "Alice works at Acme.", "tenant_a");
    let b = rec(&db, "Bob works at Globex.", "tenant_b");
    db.apply_pending_ops_once(100).unwrap();
    plant_legacy_junk(&db, &a, "tenant_a");
    {
        let conn = db.conn();
        conn.execute(
            "INSERT INTO claims (claim_id, src, dst, rel_type, weight, created_at, extractor, \
             source_memory_rid, namespace) VALUES ('legacy2', 'Bob', 'UTC', 'leads', 1.0, 1.0, \
             'heuristic_v1', ?1, 'tenant_b')",
            rusqlite::params![b],
        )
        .unwrap();
    }
    let report = db.reextract_claims(Some("tenant_a"), false).unwrap();
    assert_eq!(report.memories_scanned, 1);
    let remaining = rels(&db, "heuristic_v1");
    assert!(
        !remaining.iter().any(|r| r.0 == "Pranab" && r.1 == "leads"),
        "tenant_a junk healed"
    );
    assert!(
        remaining.iter().any(|r| r.0 == "Bob" && r.1 == "leads"),
        "tenant_b untouched: {remaining:?}"
    );
}
