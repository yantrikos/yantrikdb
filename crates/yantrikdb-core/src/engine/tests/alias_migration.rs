//! C5b — possessive alias migration + graph-index fold (wheel piece 2).
//!
//! The tokenizer exemption (fixed as C5a) persisted phantom entities:
//! the production census found `Pranab's` holding 748 mentions — 35%
//! of that person's references — mistyped and unreachable, plus
//! contractions promoted to entities (`Don't`, 96 mentions). The
//! migration writes reversible alias rows for terminal possessives
//! whose canonical exists; the index build folds aliased names into
//! their canonical at load. Contractions get NO alias (aliasing
//! `Don't` onto a real `Don` would be a false merge).

use super::*;

fn seed_polluted_graph(db: &YantrikDB) {
    let conn = db.conn.lock();
    let ts = 1_700_000_000.0f64;
    for (name, etype, mc) in [
        ("Pranab", "person", 1412i64),
        ("Pranab's", "project", 748),
        ("Hermes", "person", 10),
        ("Hermes's", "org", 2),
        ("Don't", "concept", 96),
    ] {
        conn.execute(
            "INSERT INTO entities (name, entity_type, mention_count, first_seen, last_seen) \
             VALUES (?1, ?2, ?3, ?4, ?4)",
            params![name, etype, mc, ts],
        )
        .unwrap();
    }
    for (rid, entity) in [
        ("rid-x", "Pranab"),
        ("rid-x", "Pranab's"), // must dedupe to ONE link post-fold
        ("rid-y", "Pranab's"), // must repoint to Pranab
    ] {
        conn.execute(
            "INSERT INTO memory_entities (memory_rid, entity_name) VALUES (?1, ?2)",
            params![rid, entity],
        )
        .unwrap();
    }
}

#[test]
fn possessive_phantoms_fold_into_their_canonicals() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    seed_polluted_graph(&db);

    db.rebuild_graph_index().unwrap();

    // Alias rows: possessives yes, contraction no.
    {
        let conn = db.conn.lock();
        let aliases: Vec<(String, String)> = conn
            .prepare("SELECT alias, canonical_name FROM entity_aliases ORDER BY alias")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        assert_eq!(
            aliases,
            vec![
                ("Hermes's".to_string(), "Hermes".to_string()),
                ("Pranab's".to_string(), "Pranab".to_string()),
            ],
            "possessives alias to canonicals; Don't must NOT alias"
        );
    }

    // The fold: canonical carries merged mentions and its OWN type; the
    // phantom is no longer independently resolvable.
    let gi = db.graph_index.read();
    let hits = gi.entity_matches_query(&[String::from("pranab")]);
    assert_eq!(hits.len(), 1, "{hits:?}");
    let (name, etype, mentions) = &hits[0];
    assert_eq!(name, "Pranab");
    assert_eq!(
        etype, "person",
        "canonical type wins; 'project' was the misparse"
    );
    assert_eq!(*mentions, 1412 + 748, "phantom mentions merge additively");

    // Memory links repoint and dedupe through the fold.
    let mems = gi.memories_for_entities(&["Pranab"]);
    assert_eq!(
        mems.len(),
        2,
        "rid-x once (deduped) + rid-y (repointed): {mems:?}"
    );
    assert!(mems.contains("rid-x") && mems.contains("rid-y"));
    drop(gi);

    // Idempotent: a second rebuild writes nothing new and folds the same.
    db.rebuild_graph_index().unwrap();
    {
        let conn = db.conn.lock();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM entity_aliases", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 2, "migration is idempotent");
    }
    let gi = db.graph_index.read();
    let hits = gi.entity_matches_query(&[String::from("pranab")]);
    assert_eq!(hits[0].2, 1412 + 748, "no double-fold on rebuild");
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn claims_lane_surfaces_the_direction_cosine_destroys() {
    // C4, the hermes stress scenario in miniature: many frame-siblings
    // with Taylor as OBJECT, one record with Taylor as SUBJECT. Cosine
    // cannot tell them apart (direction is exactly what mean-pooled
    // embeddings destroy); the claims table knows. The lane must admit
    // the claim's source record with the direction spelled out.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let rec = |text: &str| {
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
    };
    for name in ["Pat", "Sam", "Alex", "Jo", "Max", "Kim", "Lee", "Ada"] {
        rec(&format!("{name} reports to Taylor."));
    }
    let target = rec("Taylor reports to Carol.");

    {
        let conn = db.conn.lock();
        conn.execute(
            "INSERT INTO entities (name, entity_type, mention_count, first_seen, last_seen) \
             VALUES ('Taylor', 'person', 9, 1.0, 1.0)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO claims (claim_id, src, dst, rel_type, weight, created_at, \
             source_memory_rid) VALUES ('c1', 'Taylor', 'Carol', 'reports_to', 1.0, 1.0, ?1)",
            params![target],
        )
        .unwrap();
    }
    db.rebuild_graph_index().unwrap();

    let query = "Who does Taylor report to?";
    let resp = db
        .recall_with_response(
            &db.embed(query).unwrap(),
            5,
            None,
            None,
            false,
            true,
            Some(query),
            true,
            None,
            None,
            None,
        )
        .unwrap();
    let hit = resp
        .results
        .iter()
        .find(|r| r.rid == target)
        .expect("the claim's source record must reach top-5");
    assert!(
        hit.why_retrieved
            .iter()
            .any(|w| w.contains("claims_match: Taylor -reports_to-> Carol")),
        "direction provenance must be spelled out, got {:?}",
        hit.why_retrieved
    );
}

#[test]
fn fold_reverses_when_alias_rows_are_deleted() {
    // The reversibility contract: persisted rows are untouched, so
    // removing the alias rows restores the pre-migration projection.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    seed_polluted_graph(&db);
    db.rebuild_graph_index().unwrap();

    {
        let conn = db.conn.lock();
        conn.execute(
            "DELETE FROM entity_aliases WHERE source = 'possessive_migration_v1'",
            [],
        )
        .unwrap();
        // Rebuild WITHOUT re-running the migration: build directly.
        let rebuilt = crate::graph_index::GraphIndex::build_from_db(&conn).unwrap();
        let hits = rebuilt.entity_matches_query(&[String::from("pranab")]);
        assert_eq!(
            hits[0].2, 1412,
            "with aliases gone the phantom un-folds; canonical keeps only its own mentions"
        );
    }
}
