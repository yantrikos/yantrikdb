//! Self-mined relation templates: stated claims teach the extractor.

use crate::{StatedClaim, YantrikDB, LEARNED_CLAIM_EXTRACTOR};

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

fn state(db: &YantrikDB, rid: &str, src: &str, rel: &str, dst: &str) {
    let report = db
        .attach_claims(
            rid,
            &[StatedClaim {
                src: src.into(),
                rel_type: rel.into(),
                dst: dst.into(),
                polarity: 1,
            }],
        )
        .unwrap();
    assert_eq!(report.accepted.len(), 1, "{:?}", report.rejected);
}

fn learned_claims(db: &YantrikDB, src: &str) -> Vec<(String, String, String)> {
    let conn = db.conn();
    let mut stmt = conn
        .prepare("SELECT src, rel_type, dst FROM claims WHERE src = ?1 AND extractor = ?2 AND tombstoned = 0")
        .unwrap();
    stmt.query_map(rusqlite::params![src, LEARNED_CLAIM_EXTRACTOR], |r| {
        Ok((r.get(0)?, r.get(1)?, r.get(2)?))
    })
    .unwrap()
    .map(|r| r.unwrap())
    .collect()
}

#[test]
fn two_distinct_stated_pairs_promote_a_template_that_extracts_from_plain_writes() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    let a = rec(&db, "Dana mentors Priya on the data platform.");
    state(&db, &a, "Dana", "mentors", "Priya");
    let after_one = db.learned_relation_patterns("default").unwrap();
    assert_eq!(after_one.len(), 1);
    assert_eq!(
        (
            after_one[0].phrase.as_str(),
            after_one[0].pair_count,
            after_one[0].active
        ),
        ("mentors", 1, false)
    );

    let b = rec(&db, "Kim mentors Alex during onboarding.");
    state(&db, &b, "Kim", "mentors", "Alex");
    let after_two = db.learned_relation_patterns("default").unwrap();
    assert_eq!(
        (after_two[0].pair_count, after_two[0].active),
        (2, true),
        "{after_two:?}"
    );

    // A PLAIN write — no stated claim — now gets the relation from the
    // learned template once the materializer runs.
    rec(&db, "Sam mentors Jordan on release engineering.");
    db.apply_pending_ops_once(100).unwrap();
    assert_eq!(
        learned_claims(&db, "Sam"),
        vec![(
            "Sam".to_string(),
            "mentors".to_string(),
            "Jordan".to_string()
        )]
    );
    // The built-in extractor knows no `mentors` template: the claim can
    // only have come from the learned one.
    let conn = db.conn();
    let heuristic: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM claims WHERE src = 'Sam' AND extractor = 'heuristic_v1'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(heuristic, 0);
}

#[test]
fn restating_one_pair_never_promotes() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    for _ in 0..3 {
        let rid = rec(&db, "Dana mentors Priya on the data platform.");
        state(&db, &rid, "Dana", "mentors", "Priya");
    }
    let patterns = db.learned_relation_patterns("default").unwrap();
    assert_eq!((patterns[0].pair_count, patterns[0].active), (1, false));
    rec(&db, "Sam mentors Jordan on release engineering.");
    db.apply_pending_ops_once(100).unwrap();
    assert!(learned_claims(&db, "Sam").is_empty());
}

#[test]
fn templates_are_namespace_scoped_and_forgettable() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    for (s, o, text) in [
        ("Dana", "Priya", "Dana mentors Priya."),
        ("Kim", "Alex", "Kim mentors Alex."),
    ] {
        let rid = db
            .record_text(
                text,
                "semantic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                "tenant_a",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        state(&db, &rid, s, "mentors", o);
    }
    assert!(db.learned_relation_patterns("tenant_a").unwrap()[0].active);
    assert!(db.learned_relation_patterns("default").unwrap().is_empty());
    // A plain write in ANOTHER namespace learns nothing from tenant_a.
    rec(&db, "Sam mentors Jordan.");
    db.apply_pending_ops_once(100).unwrap();
    assert!(learned_claims(&db, "Sam").is_empty());

    assert_eq!(db.forget_learned_relation_patterns("tenant_a").unwrap(), 1);
    assert!(db.learned_relation_patterns("tenant_a").unwrap().is_empty());
}

#[test]
fn function_word_windows_teach_nothing() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    for (s, o, text) in [
        ("Pranab", "Vim", "Pranab is a Vim user."),
        ("Maria", "Emacs", "Maria is a Emacs user."),
    ] {
        let rid = rec(&db, text);
        state(&db, &rid, s, "uses", o);
    }
    assert!(db.learned_relation_patterns("default").unwrap().is_empty());
}

#[test]
fn v51_migration_creates_the_template_tables_on_an_upgraded_store() {
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();
    {
        let _db = YantrikDB::with_default(path).unwrap();
    }
    {
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.execute_batch(
            "DROP TABLE learned_relation_pattern_support; DROP TABLE learned_relation_patterns; \
             INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '50');",
        )
        .unwrap();
    }
    let db = YantrikDB::with_default(path).unwrap();
    let rid = rec(&db, "Dana mentors Priya.");
    state(&db, &rid, "Dana", "mentors", "Priya");
    assert_eq!(db.learned_relation_patterns("default").unwrap().len(), 1);
}

#[test]
fn encrypted_stores_mine_nothing() {
    // A template phrase is a plaintext fragment of the memory; an
    // encrypted store must keep its text at rest and learn no templates.
    let key = [7u8; 32];
    let db = YantrikDB::new_encrypted(":memory:", 8, &key).unwrap();
    let emb: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) / 8.0).collect();
    for (s, o, text) in [
        ("Dana", "Priya", "Dana mentors Priya."),
        ("Kim", "Alex", "Kim mentors Alex."),
    ] {
        let rid = db
            .record(
                text,
                "semantic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &emb,
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        state(&db, &rid, s, "mentors", o);
    }
    assert!(db.learned_relation_patterns("default").unwrap().is_empty());
}
