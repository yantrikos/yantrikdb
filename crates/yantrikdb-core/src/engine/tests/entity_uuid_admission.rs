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

#[test]
fn a_backfill_does_not_relink_a_planted_rid_entity() {
    // Review PR322A-02: `backfill_memory_entities` scanned `entities` with no
    // admission filter, so a rid row that the load-time suppression leaves in
    // place (by design, non-destructive) regained a `memory_entities` link on
    // the next maintenance or replication pass — undoing the heal.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // The rid must appear IN the text, or the backfill would never match it
    // and the test would pass with or without the guard.
    let body = format!("Log entry tied to record {LOWER} during the sweep.");
    let memory = db
        .record(
            &body,
            "episodic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({}),
            &vec![0.1f32; 8],
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    // Plant the rid ENTITY ROW only — no `memory_entities` link, no claim. That
    // is the state the load-time suppression leaves behind, and the state the
    // backfill used to re-link.
    db.conn()
        .execute(
            "INSERT INTO entities (name, entity_type, first_seen, last_seen, mention_count) \
             VALUES (?1, 'tech', 1.0, 1.0, 1)",
            rusqlite::params![LOWER],
        )
        .unwrap();
    assert_eq!(
        link_rows(&db, LOWER),
        0,
        "fixture: no link before the backfill"
    );

    db.backfill_memory_entities().unwrap();

    assert_eq!(
        link_rows(&db, LOWER),
        0,
        "a backfill must not create a memory_entities link for a record id"
    );
}

#[test]
fn ingest_claim_with_a_rid_endpoint_mints_no_index_node_or_edge() {
    // Review PR322A-01: `ingest_claim` is the documented primary claim write
    // path. Its SQL side skipped a UUID endpoint (admit_entity rejects it),
    // but its graph-index side added the node and the hop unconditionally —
    // exactly what `relate()` was changed not to do.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let memory = db
        .record(
            "Alice Moreau signed the supply agreement.",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({}),
            &vec![0.2f32; 8],
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    db.ingest_claim(
        LOWER,
        "related_to",
        "Alice Moreau",
        "default",
        1,
        "asserted",
        None,
        None,
        "manual",
        None,
        "high",
        Some(&memory),
        None,
        None,
        1.0,
    )
    .unwrap();

    assert_eq!(entity_rows(&db, LOWER), 0, "no entity row for a record id");
    let gi = db.graph_index.read();
    assert!(
        !gi.all_entity_names().iter().any(|n| n == LOWER),
        "no graph node for a record id"
    );
    assert_eq!(gi.edge_count(), 0, "no hop through a record id");
}

/// The shape rule is a **whole-name** rule, and that boundary is deliberate.
///
/// #322 asks for "UUID-shaped names only — 8-4-4-4-12 or 32 contiguous hex
/// digits, whole name". A decorated id (`{…}`, one character of prefix or
/// suffix) is a different name, so it is admitted today and would become an
/// entity. Pinned here so the boundary cannot drift silently: if it should be
/// stricter, the rule changes here first, with its own tests.
///
/// URN spellings fail for an unrelated reason — `ENTITY_MAX_CHARS` is 40 and
/// `urn:uuid:` plus 36 characters is 45 — so their refusal is *not* evidence
/// about this rule and is deliberately not asserted.
#[test]
fn the_shape_rule_is_whole_name_only() {
    use crate::knowledge::graph::{is_rejected_entity_name, is_uuid_shaped};

    assert!(is_uuid_shaped(LOWER), "a bare record id is uuid-shaped");
    assert!(
        is_rejected_entity_name(LOWER),
        "a bare record id must be refused as an entity name"
    );

    for decorated in [
        format!("{{{LOWER}}}"),
        format!("x{LOWER}"),
        format!("{LOWER}x"),
    ] {
        assert!(
            !is_uuid_shaped(&decorated),
            "{decorated} is not a whole-name uuid, by the written rule"
        );
        assert!(
            !is_rejected_entity_name(&decorated),
            "{decorated} is admitted — that is the documented scope, not an accident"
        );
    }
}
