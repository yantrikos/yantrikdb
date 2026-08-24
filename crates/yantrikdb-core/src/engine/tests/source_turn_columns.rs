use super::*;

// ── v50: the conversational turn as a first-class column ─────────────
//
// `memories.source_turn` mirrors metadata `source_turn` / `turn_id`
// (engine::thread::extract_source_turn is the single source), stamped by
// EVERY metadata-persisting writer — the same nine sites as the v48
// event_time work. The census test is the category enforcer (the
// event_time_columns.rs pattern): it asserts that NO row anywhere carries
// a json-visible valid turn without the column, after driving several
// distinct write paths, on leader AND follower. The marker/trigger tests
// pin the completeness contract; the replication tests pin the
// canonical-scalar transport; the migration test pins the v49→v50 path.
// =====================================================================

const MARKER: &str = "source_turn_backfill_complete";

fn marker_value(db: &YantrikDB) -> Option<String> {
    db.conn()
        .query_row(
            "SELECT value FROM meta WHERE key = ?1",
            params![MARKER],
            |r| r.get(0),
        )
        .ok()
}

fn column_turn(db: &YantrikDB, rid: &str) -> Option<i64> {
    db.conn()
        .query_row(
            "SELECT source_turn FROM memories WHERE rid = ?1",
            params![rid],
            |r| r.get(0),
        )
        .unwrap()
}

/// Rows whose (json-valid) metadata carries a valid turn — under the
/// EXACT extractor semantics: integer-typed, non-negative, `source_turn`
/// preferred with fall-through to `turn_id` when source_turn is absent
/// OR invalid — while the column is NULL. Zero is the invariant.
fn census_violations(conn: &rusqlite::Connection) -> i64 {
    conn.query_row(
        "SELECT COUNT(*) FROM memories \
         WHERE json_valid(metadata) \
           AND source_turn IS NULL \
           AND (CASE \
                 WHEN json_type(metadata, '$.source_turn') = 'integer' \
                      AND json_extract(metadata, '$.source_turn') >= 0 \
                 THEN json_extract(metadata, '$.source_turn') \
                 WHEN json_type(metadata, '$.turn_id') = 'integer' \
                      AND json_extract(metadata, '$.turn_id') >= 0 \
                 THEN json_extract(metadata, '$.turn_id') \
                 ELSE NULL END) IS NOT NULL",
        [],
        |r| r.get(0),
    )
    .unwrap()
}

fn drain(db: &YantrikDB) {
    for _ in 0..50 {
        if db.apply_pending_ops_once(500).unwrap() == 0 {
            return;
        }
    }
    panic!("pending ops did not drain");
}

/// CENSUS (contract clause 8): after driving record (all turn shapes),
/// both correction paths, and replication apply, no row on either engine
/// may carry a json-visible valid turn with a NULL column — and where the
/// column is set it must EQUAL the extractor's read of the JSON.
#[test]
fn census_every_memories_writer_stamps_the_source_turn_column() {
    use crate::replication::{apply_ops, extract_ops_since};

    let leader = YantrikDB::new(":memory:", 8).unwrap();
    let mut record = |text: &str, meta: serde_json::Value| -> String {
        leader
            .record(
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
    };
    // The turn shapes: preferred key, fallback key, invalid-preferred
    // falling through to fallback, invalid-only (never invented).
    let rid_turn = record("plain turn", serde_json::json!({"source_turn": 5}));
    let rid_fallback = record("fallback turn", serde_json::json!({"turn_id": 2}));
    let rid_fallthrough = record(
        "invalid preferred falls through",
        serde_json::json!({"source_turn": "bogus", "turn_id": 7}),
    );
    let rid_invalid = record("invalid only", serde_json::json!({"source_turn": -3}));
    drain(&leader);

    assert_eq!(column_turn(&leader, &rid_turn), Some(5));
    assert_eq!(column_turn(&leader, &rid_fallback), Some(2));
    assert_eq!(
        column_turn(&leader, &rid_fallthrough),
        Some(7),
        "or_else fallback: invalid source_turn falls through to turn_id"
    );
    assert_eq!(
        column_turn(&leader, &rid_invalid),
        None,
        "never invented from an invalid value"
    );

    // Metadata-only correction (scalar correct path) changes the turn —
    // the column must be re-stamped in the same transaction.
    leader
        .correct(
            &rid_turn,
            None,
            Some(&serde_json::json!({"source_turn": 9})),
            None,
            None,
            "turn was wrong",
        )
        .unwrap();
    assert_eq!(column_turn(&leader, &rid_turn), Some(9));

    // Text-changing correction (re-embedding correct path) re-stamps too.
    let generation = leader.search_generation();
    leader
        .correct_with_embedding(
            &rid_fallback,
            Some("fallback turn, reworded"),
            &vec_seed(3.0, 8),
            generation,
            Some(&serde_json::json!({"turn_id": 4})),
            None,
            None,
            "reworded",
        )
        .unwrap();
    assert_eq!(column_turn(&leader, &rid_fallback), Some(4));

    // Replication apply into a second engine — materialize_record for all
    // rows plus the replicated metadata-only correct arm. The text-changing
    // correct op is excluded, exactly as the event_time census excludes it
    // (embedder-less engines cannot replay its vector).
    let follower = YantrikDB::new(":memory:", 8).unwrap();
    let ops: Vec<_> = extract_ops_since(&leader.conn(), None, None, None, 100)
        .unwrap()
        .into_iter()
        .filter(|o| o.op_type != "correct" || o.embedding.is_none())
        .collect();
    apply_ops(&follower, &ops).unwrap();

    // THE CENSUS: zero violations on both engines, whoever wrote the row.
    for (name, db) in [("leader", &leader), ("follower", &follower)] {
        let conn = db.conn();
        assert_eq!(
            census_violations(&conn),
            0,
            "{name}: a writer persisted turn-bearing metadata without stamping \
             the v50 column; every memories writer must bind extract_source_turn()"
        );
    }
    // Canonical-scalar transport: the follower's columns match the leader's.
    for rid in [&rid_turn, &rid_fallthrough, &rid_invalid] {
        assert_eq!(
            column_turn(&follower, rid),
            column_turn(&leader, rid),
            "follower column equals leader column for {rid}"
        );
    }
}

/// Replication canonical-scalar semantics: a payload-carried scalar is
/// used DIRECTLY (even when it disagrees with the payload metadata); a
/// legacy payload without the key falls back to parsing metadata through
/// the shared extractor.
#[test]
fn replication_uses_canonical_scalar_with_legacy_fallback() {
    use crate::replication::{apply_ops, extract_ops_since};

    let leader = YantrikDB::new(":memory:", 8).unwrap();
    let rid = leader
        .record(
            "canonical scalar row",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({"source_turn": 5}),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    let ops = extract_ops_since(&leader.conn(), None, None, None, 100).unwrap();
    let record_op = ops
        .iter()
        .find(|o| o.op_type == "record" && o.target_rid.as_deref() == Some(rid.as_str()))
        .expect("the record op exists")
        .clone();
    assert_eq!(
        record_op.payload["source_turn"], 5,
        "the leader-derived canonical scalar rides the op payload"
    );

    // (a) Canonical beats metadata parse: force scalar=9 vs metadata=5.
    let mut canonical = record_op.clone();
    canonical.payload["source_turn"] = serde_json::json!(9);
    let f1 = YantrikDB::new(":memory:", 8).unwrap();
    apply_ops(&f1, &[canonical]).unwrap();
    assert_eq!(
        column_turn(&f1, &rid),
        Some(9),
        "the canonical scalar is used directly, never re-derived"
    );

    // (b) Legacy payload (no key): extractor fallback over payload metadata.
    let mut legacy = record_op.clone();
    legacy
        .payload
        .as_object_mut()
        .unwrap()
        .remove("source_turn");
    let f2 = YantrikDB::new(":memory:", 8).unwrap();
    apply_ops(&f2, &[legacy]).unwrap();
    assert_eq!(
        column_turn(&f2, &rid),
        Some(5),
        "legacy fallback parses metadata through the shared extractor"
    );
}

/// Replay repair: re-delivery of a record op may FILL a NULL column but
/// never overwrites a non-NULL value (which came from a write at least as
/// new as the replay).
#[test]
fn replication_replay_repair_fills_null_never_overwrites() {
    use crate::replication::{apply_ops, extract_ops_since};

    let leader = YantrikDB::new(":memory:", 8).unwrap();
    let rid = leader
        .record(
            "repair target",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({"source_turn": 5}),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    let ops = extract_ops_since(&leader.conn(), None, None, None, 100).unwrap();
    let record_op = ops
        .iter()
        .find(|o| o.op_type == "record" && o.target_rid.as_deref() == Some(rid.as_str()))
        .unwrap()
        .clone();

    let follower = YantrikDB::new(":memory:", 8).unwrap();
    apply_ops(&follower, &[record_op.clone()]).unwrap();
    assert_eq!(column_turn(&follower, &rid), Some(5));

    // NULL the column (raw SQL — stales the marker, which is fine here),
    // then replay under a fresh op id: the repair FILLS the NULL.
    follower
        .conn()
        .execute(
            "UPDATE memories SET source_turn = NULL WHERE rid = ?1",
            params![rid],
        )
        .unwrap();
    let mut replay = record_op.clone();
    replay.op_id = crate::id::new_id();
    apply_ops(&follower, &[replay]).unwrap();
    assert_eq!(
        column_turn(&follower, &rid),
        Some(5),
        "replay repair fills a NULL column"
    );

    // A non-NULL (newer) value is NEVER overwritten by a replay.
    follower
        .conn()
        .execute(
            "UPDATE memories SET source_turn = 8 WHERE rid = ?1",
            params![rid],
        )
        .unwrap();
    let mut replay2 = record_op.clone();
    replay2.op_id = crate::id::new_id();
    apply_ops(&follower, &[replay2]).unwrap();
    assert_eq!(
        column_turn(&follower, &rid),
        Some(8),
        "replay repair must not overwrite a non-NULL newer value"
    );
}

/// MARKER TEST (a), verbatim from the contract: a stamped write AFTER
/// completed maintenance leaves the marker true and the next strict query
/// succeeds. (Encrypted store — the strongest form.)
#[test]
fn stamped_write_after_completion_preserves_marker_true() {
    let db = YantrikDB::new_encrypted(":memory:", 8, &[7u8; 32]).unwrap();
    assert_eq!(
        marker_value(&db).as_deref(),
        Some("1"),
        "fresh store: complete immediately"
    );

    let rid = db
        .record(
            "Alpha stamped write",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({"source_turn": 3}),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    db.link_memory_entity(&rid, "Alpha").unwrap();
    assert_eq!(
        marker_value(&db).as_deref(),
        Some("1"),
        "an engine-supported stamped write preserves marker=true \
         (its own trigger fire is not staleness)"
    );

    let query = crate::engine::thread::ThreadQuery {
        entities: vec!["Alpha".to_string()],
        phrases: Vec::new(),
        topic_rids: Vec::new(),
    };
    let out = db
        .recall_thread_v2("default", &query, 10)
        .expect("strict query must succeed: no error after a stamped write");
    assert_eq!(out.total, 1);
    assert_eq!(out.items[0].source_turn, Some(3));
}

/// MARKER TEST (b), verbatim: a stamped write while the backfill is
/// incomplete leaves the marker false, and the strict query still errors —
/// until maintain_source_turn_backfill completes (decrypt-and-stamp heals
/// the raw-NULLed column too).
#[test]
fn stamped_write_while_incomplete_keeps_marker_false_and_strict_query_errors() {
    use crate::error::YantrikDbError;
    let db = YantrikDB::new_encrypted(":memory:", 8, &[7u8; 32]).unwrap();
    let rid1 = db
        .record(
            "Alpha first encrypted row",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({"source_turn": 3}),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    db.link_memory_entity(&rid1, "Alpha").unwrap();

    // Raw SQL breaks the column (an incomplete/staled store): the schema
    // trigger flips the marker to false.
    db.conn()
        .execute(
            "UPDATE memories SET source_turn = NULL WHERE rid = ?1",
            params![rid1],
        )
        .unwrap();
    assert_eq!(marker_value(&db).as_deref(), Some("0"), "trigger staled it");

    // A NEW stamped engine write does NOT waive the false marker.
    let rid2 = db
        .record(
            "Alpha second encrypted row",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({"source_turn": 4}),
            &vec_seed(2.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    db.link_memory_entity(&rid2, "Alpha").unwrap();
    assert_eq!(
        marker_value(&db).as_deref(),
        Some("0"),
        "a stamped write PRESERVES the pre-write state: false stays false"
    );

    let query = crate::engine::thread::ThreadQuery {
        entities: vec!["Alpha".to_string()],
        phrases: Vec::new(),
        topic_rids: Vec::new(),
    };
    let err = db.recall_thread_v2("default", &query, 10).unwrap_err();
    assert!(
        matches!(err, YantrikDbError::MaintenanceRequired { ref operation, .. }
                 if operation == "maintain_source_turn_backfill"),
        "strict query refuses while incomplete: {err:?}"
    );

    // Decrypt-and-stamp maintenance heals it (batched, resumable).
    let mut rounds = 0;
    loop {
        let progress = db.maintain_source_turn_backfill(1_000).unwrap();
        rounds += 1;
        if progress.complete {
            break;
        }
        assert!(rounds < 100, "maintenance must terminate");
    }
    assert_eq!(marker_value(&db).as_deref(), Some("1"));
    let out = db.recall_thread_v2("default", &query, 10).unwrap();
    assert_eq!(out.total, 2);
    assert_eq!(
        out.items.iter().map(|i| i.source_turn).collect::<Vec<_>>(),
        vec![Some(3), Some(4)],
        "the decrypt-and-stamp pass restored the raw-NULLed column"
    );
}

/// KILL TESTS (reviewer blocker 1): raw SQL CHANGES metadata — turn 5→7,
/// and 5→absent — and the maintenance pass must RECOMPUTE (including back
/// to NULL), never NULL-fill; wrong values are never served under a true
/// marker. Plaintext store: the universal gate applies there too.
#[test]
fn raw_metadata_change_is_recomputed_never_null_filled() {
    use crate::error::YantrikDbError;
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "Alpha recompute target",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({"source_turn": 5}),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    db.link_memory_entity(&rid, "Alpha").unwrap();
    assert_eq!(column_turn(&db, &rid), Some(5));
    assert_eq!(marker_value(&db).as_deref(), Some("1"));

    let query = crate::engine::thread::ThreadQuery {
        entities: vec!["Alpha".to_string()],
        phrases: Vec::new(),
        topic_rids: Vec::new(),
    };

    // (1) Raw SQL changes the turn 5 -> 7. The stale NON-NULL scalar (5)
    // must not survive a completing maintenance pass.
    db.conn()
        .execute(
            "UPDATE memories SET metadata = json_set(metadata, '$.source_turn', 7) \
             WHERE rid = ?1",
            params![rid],
        )
        .unwrap();
    assert_eq!(marker_value(&db).as_deref(), Some("0"), "trigger staled it");
    let err = db.recall_thread_v2("default", &query, 10).unwrap_err();
    assert!(
        matches!(err, YantrikDbError::MaintenanceRequired { .. }),
        "the PLAINTEXT store's strict query also refuses before repair: {err:?}"
    );
    loop {
        if db.maintain_source_turn_backfill(1_000).unwrap().complete {
            break;
        }
    }
    assert_eq!(marker_value(&db).as_deref(), Some("1"));
    assert_eq!(
        column_turn(&db, &rid),
        Some(7),
        "maintenance RECOMPUTED the stale non-NULL scalar to the current \
         metadata's value — a NULL-only fill would have left 5"
    );
    let out = db.recall_thread_v2("default", &query, 10).unwrap();
    assert_eq!(
        out.items[0].source_turn,
        Some(7),
        "never a wrong value under marker=true"
    );

    // (2) Raw SQL REMOVES the turn. The column must go back to NULL.
    db.conn()
        .execute(
            "UPDATE memories SET metadata = json_remove(metadata, '$.source_turn') \
             WHERE rid = ?1",
            params![rid],
        )
        .unwrap();
    assert_eq!(marker_value(&db).as_deref(), Some("0"));
    loop {
        if db.maintain_source_turn_backfill(1_000).unwrap().complete {
            break;
        }
    }
    assert_eq!(marker_value(&db).as_deref(), Some("1"));
    assert_eq!(
        column_turn(&db, &rid),
        None,
        "metadata no longer carries a valid turn: the recompute must CLEAR \
         the column, not keep the stale scalar"
    );
    let out = db.recall_thread_v2("default", &query, 10).unwrap();
    assert_eq!(out.items[0].source_turn, None);
}

/// COMPATIBILITY PIN (final reviewer decision): on a stale store
/// (marker=false) v1 recall_thread still answers, correct via its
/// decrypt-derived path, while v2 on the SAME store refuses with
/// MaintenanceRequired.
#[test]
fn v1_stays_correct_on_stale_store_while_v2_requires_maintenance() {
    use crate::error::YantrikDbError;
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let t = 1_700_000_000.0_f64;
    let mut seed = |text: &str, turn: i64, seedv: f32| -> String {
        let rid = db
            .record_with_idempotency(
                text,
                "episodic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({"source_turn": turn}),
                &vec_seed(seedv, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
                None,
                Some(t), // equal created_at: the turn is the tie-break
            )
            .unwrap();
        db.link_memory_entity(&rid, "Alpha").unwrap();
        rid
    };
    let rid5 = seed("Alpha event five", 5, 1.0);
    let rid3 = seed("Alpha event three", 3, 2.0);

    // Raw SQL rewrites rid5's turn 5 -> 1: marker goes false.
    db.conn()
        .execute(
            "UPDATE memories SET metadata = json_set(metadata, '$.source_turn', 1) \
             WHERE rid = ?1",
            params![rid5],
        )
        .unwrap();
    assert_eq!(marker_value(&db).as_deref(), Some("0"));

    // v1: correct order from the CURRENT metadata (decrypt-derived), no
    // MaintenanceRequired ever.
    let v1 = db.recall_thread("default", &["Alpha"], 10).unwrap();
    assert_eq!(
        v1.items.iter().map(|i| i.rid.as_str()).collect::<Vec<_>>(),
        vec![rid5.as_str(), rid3.as_str()],
        "v1 orders by the rewritten turn (1 before 3) — its own decrypt path"
    );
    assert_eq!(
        v1.items.iter().map(|i| i.source_turn).collect::<Vec<_>>(),
        vec![Some(1), Some(3)]
    );

    // v2: strict gate refuses the same store.
    let query = crate::engine::thread::ThreadQuery {
        entities: vec!["Alpha".to_string()],
        phrases: Vec::new(),
        topic_rids: Vec::new(),
    };
    let err = db.recall_thread_v2("default", &query, 10).unwrap_err();
    assert!(matches!(err, YantrikDbError::MaintenanceRequired { .. }));

    // After maintenance the two paths agree.
    loop {
        if db.maintain_source_turn_backfill(1_000).unwrap().complete {
            break;
        }
    }
    let v2 = db.recall_thread_v2("default", &query, 10).unwrap();
    assert_eq!(
        v2.items.iter().map(|i| i.rid.as_str()).collect::<Vec<_>>(),
        v1.items.iter().map(|i| i.rid.as_str()).collect::<Vec<_>>(),
        "healed v2 equals v1"
    );
}

/// Maintenance batch hardening (reviewer item 8): batch=0 and
/// batch>MAX (10_000) are typed InvalidInput; batch=MAX works.
#[test]
fn maintenance_batch_caps_are_typed() {
    use crate::error::YantrikDbError;
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let err = db.maintain_source_turn_backfill(0).unwrap_err();
    assert!(matches!(err, YantrikDbError::InvalidInput(_)), "{err:?}");
    let err = db.maintain_source_turn_backfill(10_001).unwrap_err();
    assert!(matches!(err, YantrikDbError::InvalidInput(_)), "{err:?}");
    let progress = db.maintain_source_turn_backfill(10_000).unwrap();
    assert!(
        progress.complete,
        "empty store completes immediately at MAX"
    );
}

/// MIGRATION PATH (reviewer blocker 2): a v49 store upgraded to v50 gets
/// the column, the Rust backfill, AND the invalidation triggers FROM THE
/// MIGRATION — verified BEHAVIORALLY: raw metadata writes (5→7 and
/// 5→absent) stale the marker on the upgraded store.
#[test]
fn migration_from_v49_installs_column_backfill_and_triggers() {
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();

    // Build a current store with turn-bearing rows, then rewind it to a
    // faithful v49 shape: no column, no index, no v50 triggers, no marker.
    let rid = {
        let db = YantrikDB::new(&path, 8).unwrap();
        let rid = db
            .record(
                "migration row",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({"source_turn": 5}),
                &vec_seed(1.0, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        drop(db);
        rid
    };
    {
        let conn = rusqlite::Connection::open(&path).unwrap();
        conn.execute_batch(
            "DROP TRIGGER memories_source_turn_marker_insert; \
             DROP TRIGGER memories_source_turn_marker_update; \
             DROP INDEX idx_memories_source_turn; \
             ALTER TABLE memories DROP COLUMN source_turn;",
        )
        .unwrap();
        conn.execute(
            "DELETE FROM meta WHERE key IN \
             ('source_turn_backfill_complete', 'source_turn_invalidation_epoch', \
              'source_turn_repair_cursor')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '49')",
            [],
        )
        .unwrap();
    }

    // Reopen: MIGRATE_V49_TO_V50 + the open()-time recompute run.
    let db = YantrikDB::new(&path, 8).expect("v49 -> v50 upgrade must succeed");
    assert_eq!(
        column_turn(&db, &rid),
        Some(5),
        "the open()-time backfill stamped the pre-existing row"
    );
    assert_eq!(
        marker_value(&db).as_deref(),
        Some("1"),
        "unencrypted store: complete after the backfill drains"
    );

    // BEHAVIORAL trigger check 1: raw metadata write 5 -> 7 stales the
    // marker (the trigger exists and fires — not just a name in
    // sqlite_master).
    db.conn()
        .execute(
            "UPDATE memories SET metadata = json_set(metadata, '$.source_turn', 7) \
             WHERE rid = ?1",
            params![rid],
        )
        .unwrap();
    assert_eq!(
        marker_value(&db).as_deref(),
        Some("0"),
        "the migrated store's UPDATE trigger fires on a raw metadata write"
    );
    loop {
        if db.maintain_source_turn_backfill(1_000).unwrap().complete {
            break;
        }
    }
    assert_eq!(column_turn(&db, &rid), Some(7), "recomputed to 7");
    assert_eq!(marker_value(&db).as_deref(), Some("1"));

    // BEHAVIORAL trigger check 2: removing the turn stales it too.
    db.conn()
        .execute(
            "UPDATE memories SET metadata = json_remove(metadata, '$.source_turn') \
             WHERE rid = ?1",
            params![rid],
        )
        .unwrap();
    assert_eq!(
        marker_value(&db).as_deref(),
        Some("0"),
        "5→absent stales too"
    );
    loop {
        if db.maintain_source_turn_backfill(1_000).unwrap().complete {
            break;
        }
    }
    assert_eq!(column_turn(&db, &rid), None, "recomputed to NULL");

    // BEHAVIORAL trigger check 3: the INSERT trigger fires on raw inserts.
    db.conn()
        .execute(
            "INSERT INTO memories (rid, type, text, created_at, updated_at, importance, \
             half_life, last_access, valence, metadata, namespace) \
             VALUES ('raw-insert', 'episodic', 'x', 1.0, 1.0, 0.5, 604800.0, 1.0, 0.0, \
                     '{\"source_turn\": 2}', 'default')",
            [],
        )
        .unwrap();
    assert_eq!(
        marker_value(&db).as_deref(),
        Some("0"),
        "the migrated store's INSERT trigger fires on a raw insert"
    );
}
