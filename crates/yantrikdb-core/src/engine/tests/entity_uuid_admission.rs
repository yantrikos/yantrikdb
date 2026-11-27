//! #322 PR a — a record id is an endpoint, never an entity.
//!
//! `relate()` accepts rids and the claim it writes is real. The id itself
//! must not become a node (`entities`, graph index, `memory_entities`), and
//! the two heals must reach stores that already hold rid entities without
//! deleting the claims that name them.

use crate::YantrikDB;

const LOWER: &str = "01a1ee35-1c2d-7a00-9f00-0123456789ab";
const UPPER: &str = "01A1EE35-1C2D-7A00-9F00-0123456789AB";

fn count_one(db: &YantrikDB, sql: &str, name: &str) -> i64 {
    db.conn()
        .query_row(sql, rusqlite::params![name], |r| r.get(0))
        .unwrap()
}

fn entity_rows(db: &YantrikDB, name: &str) -> i64 {
    count_one(db, "SELECT COUNT(*) FROM entities WHERE name = ?1", name)
}

fn link_rows(db: &YantrikDB, name: &str) -> i64 {
    count_one(
        db,
        "SELECT COUNT(*) FROM memory_entities WHERE entity_name = ?1",
        name,
    )
}

fn claim_rows(db: &YantrikDB, claim_id: &str) -> i64 {
    count_one(
        db,
        "SELECT COUNT(*) FROM claims WHERE claim_id = ?1",
        claim_id,
    )
}

#[test]
fn relate_between_record_ids_keeps_the_claim_but_mints_no_entity() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    db.relate(LOWER, UPPER, "related_to", 1.0).unwrap();

    let claims: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM claims WHERE src = ?1 AND dst = ?2 \
             AND rel_type = 'related_to' AND extractor = 'manual' AND tombstoned = 0",
            rusqlite::params![LOWER, UPPER],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(claims, 1, "the relation is real and keeps its claim");

    for rid in [LOWER, UPPER] {
        assert_eq!(entity_rows(&db, rid), 0, "{rid:?} became an entity");
        assert_eq!(link_rows(&db, rid), 0, "{rid:?} got a memory link");
    }
    let gi = db.graph_index.read();
    assert_eq!(gi.entity_count(), 0, "a record id became a graph node");
    assert_eq!(gi.edge_count(), 0, "the index kept an edge through an id");
}

#[test]
fn relate_keeps_the_real_endpoint_next_to_a_record_id() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    db.relate(LOWER, "Alice Moreau", "mentions", 1.0).unwrap();

    assert_eq!(entity_rows(&db, LOWER), 0);
    assert_eq!(entity_rows(&db, "Alice Moreau"), 1);
    let gi = db.graph_index.read();
    assert_eq!(gi.entity_count(), 1);
    assert_eq!(gi.edge_count(), 0, "no hop through a non-node");
}

/// Plant what an older engine left behind: a rid in `entities`, a memory
/// link, and two `manual` claims (one is `related_to`, which the heal's
/// assertion query deliberately ignores; the other makes the row asserted).
fn plant_rid_entity(db: &YantrikDB, memory_rid: &str, rid: &str) {
    let conn = db.conn();
    conn.execute(
        "INSERT INTO entities (name, entity_type, first_seen, last_seen, mention_count) \
         VALUES (?1, 'tech', 1.0, 1.0, 1)",
        rusqlite::params![rid],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO memory_entities (memory_rid, entity_name, entity_name_norm) \
         VALUES (?1, ?2, lower(?2))",
        rusqlite::params![memory_rid, rid],
    )
    .unwrap();
    for (id, rel) in [("rid_e1", "related_to"), ("rid_e2", "works_at")] {
        conn.execute(
            "INSERT INTO claims (claim_id, src, dst, rel_type, weight, created_at, extractor, namespace) \
             VALUES (?1, ?2, 'Alice Moreau', ?3, 1.0, 1.0, 'manual', 'default')",
            rusqlite::params![id, rid, rel],
        )
        .unwrap();
    }
}

#[test]
fn the_entity_heal_drops_rid_rows_but_keeps_their_claims() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    let memory_rid = db
        .record_text(
            "Alice Moreau works at Fennwick Labs.",
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
    plant_rid_entity(&db, &memory_rid, LOWER);

    let dry = db.reextract_entities(true).unwrap();
    assert!(dry.inadmissible >= 1, "{dry:?}");
    assert_eq!(
        dry.kept_by_claims, 0,
        "an id is not kept as an asserted name"
    );
    assert_eq!(dry.entities_removed, 0);
    assert_eq!(entity_rows(&db, LOWER), 1, "a dry run changes nothing");

    let report = db.reextract_entities(false).unwrap();
    assert_eq!(report.kept_by_claims, 0, "{report:?}");
    assert!(report.entities_removed >= 1, "{report:?}");
    assert_eq!(entity_rows(&db, LOWER), 0, "the id row is gone");
    assert_eq!(link_rows(&db, LOWER), 0, "its links are gone");
    assert_eq!(
        claim_rows(&db, "rid_e1"),
        1,
        "the relation survives the heal"
    );
    assert_eq!(claim_rows(&db, "rid_e2"), 1, "the assertion survives too");
    assert_eq!(report.claims_removed, 0, "{report:?}");

    let again = db.reextract_entities(false).unwrap();
    assert_eq!(again.entities_removed, 0, "idempotent");
}

#[test]
fn reopening_a_store_stops_indexing_rid_entities_without_deleting_them() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rids.db");
    let path = path.to_str().unwrap();
    {
        let db = YantrikDB::with_default(path).unwrap();
        plant_rid_entity(&db, "memory-rid", LOWER);
    }
    let db = YantrikDB::with_default(path).unwrap();
    let names = db.graph_index.read().all_entity_names();
    assert!(!names.iter().any(|n| n == LOWER), "{names:?}");
    assert_eq!(
        entity_rows(&db, LOWER),
        1,
        "load suppression is a read-time projection, not a migration"
    );
    assert_eq!(claim_rows(&db, "rid_e1"), 1, "the claim is not deleted");
    assert_eq!(
        db.graph_index.read().edge_count(),
        0,
        "no indexed hop through an id"
    );
}

#[test]
fn the_cluster_applier_applies_the_same_endpoint_rule() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    db.upsert_entity_edge_with_id(
        "edge-rid-1",
        LOWER,
        UPPER,
        "related_to",
        1.0,
        "default",
        1_000_000,
        None,
    )
    .unwrap();

    assert_eq!(
        claim_rows(&db, "edge-rid-1"),
        1,
        "the replicated claim lands"
    );
    for rid in [LOWER, UPPER] {
        assert_eq!(
            entity_rows(&db, rid),
            0,
            "{rid:?} became an entity on replay"
        );
        assert_eq!(link_rows(&db, rid), 0);
    }
    let gi = db.graph_index.read();
    assert_eq!(gi.entity_count(), 0, "a replay minted a graph node");
    assert_eq!(gi.edge_count(), 0);
}
