//! The entity-table heal (issue #213): inadmissible nodes go, asserted ones stay.

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

fn entity_names(db: &YantrikDB) -> Vec<String> {
    let conn = db.conn();
    let mut stmt = conn
        .prepare("SELECT name FROM entities ORDER BY name")
        .unwrap();
    stmt.query_map([], |r| r.get::<_, String>(0))
        .unwrap()
        .map(|r| r.unwrap())
        .collect()
}

fn count(db: &YantrikDB, sql: &str) -> i64 {
    db.conn().query_row(sql, [], |r| r.get(0)).unwrap()
}

/// What an older extractor left behind: headings, bare numbers and a
/// possessive straggler minted as nodes, with links and a heuristic claim
/// riding on them.
fn plant_legacy_entities(db: &YantrikDB, rid: &str) {
    let conn = db.conn();
    for (name, ty) in [
        ("STRATEGIC POINT", "tech"),
        ("MASTERING", "tech"),
        ("1348", "tech"),
        ("0.15.2", "tech"),
        ("Pranab's", "unknown"),
        ("Recall Return Unrelated Records Root Cause", "unknown"),
    ] {
        conn.execute(
            "INSERT OR IGNORE INTO entities (name, entity_type, first_seen, last_seen, mention_count) \
             VALUES (?1, ?2, 1.0, 1.0, 1)",
            rusqlite::params![name, ty],
        )
        .unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name, entity_name_norm) \
             VALUES (?1, ?2, lower(?2))",
            rusqlite::params![rid, name],
        )
        .unwrap();
    }
    conn.execute(
        "INSERT INTO claims (claim_id, src, dst, rel_type, weight, created_at, extractor, \
         source_memory_rid, namespace) VALUES ('legacy_e1', 'STRATEGIC POINT', 'MASTERING', \
         'lives_in', 1.0, 1.0, 'heuristic_v1', ?1, 'default')",
        rusqlite::params![rid],
    )
    .unwrap();
}

#[test]
fn heal_removes_inadmissible_nodes_and_their_links_but_keeps_asserted_ones() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    let a = rec(&db, "Alice Moreau works at Fennwick Labs in Berlin.");
    db.apply_pending_ops_once(100).unwrap();
    plant_legacy_entities(&db, &a);
    // A writer asserted a claim on one shouted name: that node must survive.
    db.conn()
        .execute(
            "INSERT OR IGNORE INTO entities (name, entity_type, first_seen, last_seen, mention_count) \
             VALUES ('ACME HOLDINGS INTERNATIONAL', 'unknown', 1.0, 1.0, 1)",
            [],
        )
        .unwrap();
    db.relate(
        "Alice Moreau",
        "ACME HOLDINGS INTERNATIONAL",
        "works_at",
        1.0,
    )
    .unwrap();

    let before = entity_names(&db);
    assert!(before.iter().any(|n| n == "STRATEGIC POINT"), "{before:?}");
    assert!(before.iter().any(|n| n == "1348"), "{before:?}");

    let dry = db.reextract_entities(true).unwrap();
    assert!(dry.dry_run);
    assert_eq!(dry.entities_removed, 0);
    assert_eq!(dry.kept_by_claims, 1, "the asserted shouted name is kept");
    assert_eq!(dry.inadmissible, 7);
    assert_eq!(entity_names(&db), before, "a dry run changes nothing");

    let report = db.reextract_entities(false).unwrap();
    assert_eq!(report.entities_removed, 6, "{report:?}");
    assert_eq!(report.kept_by_claims, 1);
    assert_eq!(
        report.claims_removed, 1,
        "the heuristic claim on a dropped node goes too"
    );
    assert!(report.links_removed >= 6, "{report:?}");
    let after = entity_names(&db);
    for gone in ["STRATEGIC POINT", "MASTERING", "1348", "0.15.2", "Pranab's"] {
        assert!(
            !after.iter().any(|n| n == gone),
            "{gone:?} survived: {after:?}"
        );
    }
    for kept in [
        "Alice Moreau",
        "Fennwick Labs",
        "Berlin",
        "ACME HOLDINGS INTERNATIONAL",
    ] {
        assert!(after.iter().any(|n| n == kept), "{kept:?} lost: {after:?}");
    }
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM memory_entities WHERE entity_name = 'STRATEGIC POINT'"
        ),
        0
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM claims WHERE claim_id = 'legacy_e1'"
        ),
        0
    );
    assert_eq!(report.after_classes["no_letters"], 0);
    assert!(report.before_classes["no_letters"] >= 2);

    // Idempotent: a second pass finds nothing to do.
    let again = db.reextract_entities(false).unwrap();
    assert_eq!(again.entities_removed, 0);
    assert_eq!(again.claims_removed, 0);
}

#[test]
fn new_writes_never_mint_values_or_headings_but_values_still_serve_as_objects() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    let rid = rec(
        &db,
        "STRATEGIC POINT: CT128 runs 0.19.0 since 2026. Alice Moreau was born in 1985.",
    );
    db.apply_pending_ops_once(100).unwrap();
    let names = entity_names(&db);
    for bad in ["STRATEGIC POINT", "0.19.0", "2026", "1985"] {
        assert!(!names.iter().any(|n| n == bad), "{bad:?} minted: {names:?}");
    }
    assert!(names.iter().any(|n| n == "CT128"), "{names:?}");
    let edges: Vec<(String, String, String)> = {
        let conn = db.conn();
        let mut stmt = conn
            .prepare("SELECT src, rel_type, dst FROM claims WHERE tombstoned = 0 AND source_memory_rid = ?1")
            .unwrap();
        stmt.query_map(rusqlite::params![rid], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .unwrap()
        .map(|r| r.unwrap())
        .collect()
    };
    assert!(
        edges.contains(&("CT128".into(), "runs".into(), "0.19.0".into())),
        "value object lost as a relation object: {edges:?}"
    );
    assert!(
        edges.contains(&("Alice Moreau".into(), "born_in".into(), "1985".into())),
        "{edges:?}"
    );
    // A value never rides a relation that cannot take one.
    let rid3 = rec(&db, "Carol Vance leads 2 teams and Dana Ito works at 2026.");
    db.apply_pending_ops_once(100).unwrap();
    let junk: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM claims WHERE tombstoned = 0 AND source_memory_rid = ?1 \
             AND rel_type IN ('leads', 'works_at')",
            rusqlite::params![rid3],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(junk, 0, "a value object rode a generic relation");
    // A stated claim on a value object is still grounded and stored.
    let rep = db
        .attach_claims(
            &rid,
            &[StatedClaim {
                src: "CT128".into(),
                rel_type: "runs".into(),
                dst: "0.19.0".into(),
                polarity: 1,
                valid_from: None,
                valid_to: None,
            }],
        )
        .unwrap();
    assert_eq!(rep.accepted.len(), 1, "{rep:?}");
}
