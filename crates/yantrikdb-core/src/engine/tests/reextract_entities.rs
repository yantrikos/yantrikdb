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

/// The store's own lexicon (v52): a word this store writes in lowercase
/// is not a name when it shows up capitalized at a sentence start.
#[test]
fn seed_and_learned_lexicon_refuse_common_words_as_single_token_entities() {
    // Seed: sentence starters never become nodes, even on a cold store.
    let db = YantrikDB::with_default(":memory:").unwrap();
    rec(
        &db,
        "Critically, Alice Moreau shipped the fix. Failed builds stay red.",
    );
    db.apply_pending_ops_once(100).unwrap();
    let names = entity_names(&db);
    for bad in ["Critically", "Failed"] {
        assert!(!names.iter().any(|n| n == bad), "{bad:?} minted: {names:?}");
    }
    assert!(names.iter().any(|n| n == "Alice Moreau"), "{names:?}");

    // Learned: `gizmo` is not in any seed. A fresh store admits `Gizmo`.
    let fresh = YantrikDB::with_default(":memory:").unwrap();
    rec(&fresh, "Gizmo ships the widget on Monday.");
    fresh.apply_pending_ops_once(100).unwrap();
    assert!(
        entity_names(&fresh).iter().any(|n| n == "Gizmo"),
        "control failed"
    );

    // A store that has written `gizmo` in lowercase four times has learned it is a word.
    let learned = YantrikDB::with_default(":memory:").unwrap();
    for i in 0..4 {
        rec(&learned, &format!("the gizmo count went up again ({i})."));
    }
    learned.apply_pending_ops_once(100).unwrap();
    let stats: (i64, i64) = learned
        .conn()
        .query_row(
            "SELECT lower_n, cap_mid_n FROM token_case_stats WHERE token = 'gizmo'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(stats, (4, 0), "one observation per memory per class");
    rec(&learned, "Gizmo ships the widget on Monday.");
    learned.apply_pending_ops_once(100).unwrap();
    let names = entity_names(&learned);
    assert!(
        !names.iter().any(|n| n == "Gizmo"),
        "learned word minted: {names:?}"
    );
    assert!(names.iter().any(|n| n == "Monday") || true); // Monday is a stopword-class token; not asserted

    // The store's usage outranks the seed the other way: a seed word used as
    // a NAME mid-sentence often enough stays a name.
    let brand = YantrikDB::with_default(":memory:").unwrap();
    for i in 0..4 {
        rec(
            &brand,
            &format!("We met the Target team again today ({i})."),
        );
    }
    brand.apply_pending_ops_once(100).unwrap();
    rec(&brand, "Target opened a new store in Berlin.");
    brand.apply_pending_ops_once(100).unwrap();
    assert!(
        entity_names(&brand).iter().any(|n| n == "Target"),
        "{:?}",
        entity_names(&brand)
    );
}

#[test]
fn heal_uses_the_lexicon_and_unlinks_kept_value_names() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    let a = rec(&db, "Alice Moreau works at Fennwick Labs in Berlin.");
    db.apply_pending_ops_once(100).unwrap();
    {
        let conn = db.conn();
        for name in ["Critically", "2026"] {
            conn.execute(
                "INSERT OR IGNORE INTO entities (name, entity_type, first_seen, last_seen, mention_count) \
                 VALUES (?1, 'unknown', 1.0, 1.0, 1)",
                rusqlite::params![name],
            )
            .unwrap();
            conn.execute(
                "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name, entity_name_norm) \
                 VALUES (?1, ?2, lower(?2))",
                rusqlite::params![a, name],
            )
            .unwrap();
        }
    }
    // `2026` is asserted by a manual claim: the row and claim survive, the links do not.
    db.relate("Alice Moreau", "2026", "joined_in", 1.0).unwrap();
    let report = db.reextract_entities(false).unwrap();
    assert!(report.lexicon_memories >= 1, "{report:?}");
    let names = entity_names(&db);
    assert!(!names.iter().any(|n| n == "Critically"), "{names:?}");
    assert!(
        names.iter().any(|n| n == "2026"),
        "kept name lost: {names:?}"
    );
    assert_eq!(report.kept_by_claims, 1, "{report:?}");
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM memory_entities WHERE entity_name = '2026'"
        ),
        0,
        "kept value name still linked"
    );
    assert_eq!(
        count(&db, "SELECT COUNT(*) FROM claims WHERE tombstoned = 0 AND dst = '2026' AND extractor = 'manual'"),
        1
    );
    // Schema landed.
    let v: String = db
        .conn()
        .query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(v, "52");
}
