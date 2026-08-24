use super::*;

// =====================================================================
// v0.7.3 — migration replay resilience (regression test for the v0.8.13
// homelab cluster upgrade incident, swarm msg 3467c556).
//
// Reproduction: a deployment whose meta.schema_version was rewound (e.g. by
// an old binary briefly running against a newer DB) ends up in a state where
// the on-disk schema is at a higher version than the meta stamp. On the next
// forward upgrade, the migration loop re-runs already-applied migrations and
// trips on `ALTER TABLE ... ADD COLUMN` (not idempotent in SQLite).
//
// The fix has two halves:
//   1. run_migration_idempotent — swallows "duplicate column name" / "already
//      exists" so any migration is replay-safe at statement level.
//   2. MAX-stamp meta.schema_version on every open — stops downgrades from
//      ever rewinding the version stamp going forward.
//
// These tests cover both halves directly.
// =====================================================================

#[test]
fn migration_replay_does_not_trip_on_already_present_column() {
    // Reproduces the yantrikdb-server v0.8.13 cluster upgrade failure:
    // open a current-schema DB, manually rewind meta.schema_version to 23
    // (simulating the rewind-then-upgrade path), then re-open. With the
    // v0.7.3 fix the second open() succeeds; without it, V23_TO_V24's
    // `ALTER TABLE oplog ADD COLUMN embedding BLOB` trips on duplicate.
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    // First open: creates DB at current SCHEMA_VERSION with all columns.
    {
        let _db = YantrikDB::new(path, 8).unwrap();
        // db drops here, conn closed
    }

    // Simulate a rewound meta stamp (the precondition that turns a forward
    // upgrade into a re-run of an already-applied migration). Direct SQL —
    // we explicitly need the corruption.
    {
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '23')",
            [],
        )
        .unwrap();
    }

    // Second open: migration loop sees existing_version=23 and tries to
    // re-run V23_TO_V24, which ADDs a column that already exists. Pre-fix
    // this returns Err("duplicate column name: embedding"). Post-fix it
    // succeeds because run_migration_idempotent swallows that specific
    // error class.
    let db = YantrikDB::new(path, 8)
        .expect("v0.7.3 idempotent migration runner must heal rewound-meta deployments");

    // Sanity: a write through the freshly healed DB still works end-to-end
    // (the column is reachable, the migration didn't leave the schema in a
    // partial state).
    db.record(
        "post-heal smoke",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(1.0, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();
}

#[test]
fn migration_replay_does_not_trip_on_alter_table_against_view() {
    // Regression test for issue #10 (2026-05-09): a DB with meta rewound to
    // v14 hits "Cannot add a column to a view" when MIGRATE_V14_TO_V15 runs
    // `ALTER TABLE edges ADD COLUMN ...` against the backward-compat VIEW
    // that V16_TO_V17 created when it renamed edges to claims. Same root
    // cause as issue closed by v0.7.3 (rewound-meta DBs replay migrations
    // against on-disk schema that's already past them) but a different
    // error class from "duplicate column name".
    //
    // Pre-v0.7.8 the runner only swallowed "duplicate column name" and
    // "already exists"; the view error propagated and broke open(). Post-fix
    // run_migration_idempotent also swallows "Cannot add a column to a view"
    // since it definitionally means the schema is already past where those
    // columns mattered (else edges wouldn't be a view).
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    // First open: creates DB at current SCHEMA_VERSION. Edges exists as a
    // VIEW (renamed from table by V16_V17 migration) and claims is the
    // real backing table.
    {
        let _db = YantrikDB::new(path, 8).unwrap();
    }

    // Sanity check: edges is indeed a view in the current-schema state.
    {
        let conn = rusqlite::Connection::open(path).unwrap();
        let kind: String = conn
            .query_row(
                "SELECT type FROM sqlite_master WHERE name = 'edges'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            kind, "view",
            "fixture precondition: at current schema, edges should be a backward-compat view"
        );
    }

    // Rewind meta to 14 — simulates the issue #10 production state where
    // an older binary briefly ran against a newer DB (or any other path
    // that rewound meta while disk schema stayed advanced).
    {
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '14')",
            [],
        )
        .unwrap();
    }

    // Re-open: existing_version=14, migration loop runs V14_V15 which does
    // `ALTER TABLE edges ADD COLUMN polarity ...` against the view. Pre-fix
    // returns Err("Cannot add a column to a view"); post-fix succeeds
    // because run_migration_idempotent now swallows that specific error
    // class as already-applied.
    let db = YantrikDB::new(path, 8)
        .expect("v0.7.8 idempotent runner must heal rewound-meta DBs that hit ALTER-on-view");

    // Sanity: post-heal write still works end-to-end.
    db.record(
        "post-heal smoke (issue 10)",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(1.0, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();
}

#[test]
fn migration_meta_stamp_does_not_downgrade() {
    // Forward arm of the same fix. Without the MAX guard, a binary whose
    // SCHEMA_VERSION constant is *behind* the on-disk DB silently rewinds
    // the meta stamp — re-creating the precondition for the previous test.
    // This locks the invariant at the meta-write site.
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    // Pre-stamp the DB at a version GREATER than current SCHEMA_VERSION.
    // Direct SQL because we're simulating "ran a future binary first".
    {
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
             INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '999');",
        )
        .unwrap();
    }

    // Open with the current binary — should NOT rewind 999 down to
    // SCHEMA_VERSION.
    let _db = YantrikDB::new(path, 8).unwrap();

    let stamped: String = {
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |row| row.get(0),
        )
        .unwrap()
    };
    assert_eq!(
        stamped, "999",
        "MAX-stamp invariant: open() must never rewind meta.schema_version below the on-disk value"
    );
}

// =====================================================================
// v0.8.x — schema v26 conflict-aware-write provenance columns
// (issue yantrikos/yantrikdb#29, RFC 026 umbrella issue #28).
//
// v26 introduces four additive columns on `memories` that the WriteResolution
// API (issue #30) will populate at write time:
//   - prior_rid, resolution_kind, dismissal_reason, confidence_at_write
//
// Plus normalizes the existing `source` field to the enum {user, inference,
// document, source}; non-conforming rows are coerced to 'user' and the
// count is logged via meta.source_normalization_log_v26.
//
// Tests cover three paths:
//   1. Fresh-install DB has all four columns and both partial indexes.
//   2. Pre-v26 DB upgrades cleanly: columns appear, indexes appear, source
//      normalization runs and the meta log is written.
//   3. Replay-resilience: migration is safe to re-run on an already-v26 DB
//      (per v0.7.3 idempotent runner contract — the same property #16, #22
//      and the cluster-replication incident locked).
// =====================================================================

#[test]
fn schema_v26_fresh_install_has_provenance_columns_and_indexes() {
    // Fresh DB takes the SCHEMA_SQL path (not the migration chain), so this
    // locks the invariant that SCHEMA_SQL stays in sync with the
    // MIGRATE_V25_TO_V26 column set. If someone adds a column to one but
    // not the other (the classic migration drift bug), this test catches it.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let conn = db.conn();

    let cols = table_columns(&conn, "memories");
    for required in [
        "prior_rid",
        "resolution_kind",
        "dismissal_reason",
        "confidence_at_write",
    ] {
        assert!(
            cols.iter().any(|c| c == required),
            "v26: fresh-install memories table missing column {required}, got: {cols:?}"
        );
    }

    assert!(
        index_exists(&conn, "idx_memories_prior_rid"),
        "v26: fresh-install missing partial index idx_memories_prior_rid"
    );
    assert!(
        index_exists(&conn, "idx_memories_resolution_kind"),
        "v26: fresh-install missing partial index idx_memories_resolution_kind"
    );
}

#[test]
fn schema_v26_migration_from_v25_adds_columns_and_normalizes_source() {
    // Simulate a pre-v26 DB: open at current schema, write some rows with
    // both enum-valid and enum-invalid source values, rewind meta to 25,
    // re-open to trigger MIGRATE_V25_TO_V26. After re-open:
    //   - new columns must exist
    //   - new indexes must exist
    //   - rows with non-enum source must have been coerced to 'user'
    //   - meta.source_normalization_log_v26 must report the affected count
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    // First open: creates DB at current SCHEMA_VERSION with all columns.
    {
        let db = YantrikDB::new(path, 8).unwrap();
        // Plant 3 rows: 2 with enum-valid source, 1 with non-enum source.
        // We bypass record() because record()'s contract doesn't yet enforce
        // the enum (that's issue #30's job); using direct SQL is the honest
        // way to simulate a pre-v26 DB that contains legacy free-text source
        // values. The migration's job is to clean those up.
        let conn = db.conn();
        for (rid, src) in [
            ("01900000-0000-7000-8000-000000000001", "user"),
            ("01900000-0000-7000-8000-000000000002", "inference"),
            ("01900000-0000-7000-8000-000000000003", "legacy-freetext"),
        ] {
            conn.execute(
                "INSERT INTO memories (rid, type, text, created_at, updated_at, last_access, source) \
                 VALUES (?1, 'episodic', 'test', 0.0, 0.0, 0.0, ?2)",
                params![rid, src],
            )
            .unwrap();
        }
    }

    // Rewind meta to 25 to force re-run of MIGRATE_V25_TO_V26. The
    // idempotent runner swallows the "duplicate column" errors that
    // the ALTER TABLE statements would raise on the second pass — that
    // property is what makes this rewind-then-reopen test legitimate.
    {
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '25')",
            [],
        )
        .unwrap();
        // Also drop the v26 indexes so we can verify the migration
        // recreates them (without this, IF NOT EXISTS would skip).
        conn.execute("DROP INDEX IF EXISTS idx_memories_prior_rid", [])
            .unwrap();
        conn.execute("DROP INDEX IF EXISTS idx_memories_resolution_kind", [])
            .unwrap();
    }

    // Re-open: existing_version=25 triggers MIGRATE_V25_TO_V26.
    let db = YantrikDB::new(path, 8)
        .expect("v26 migration must run cleanly against a rewound-meta v25 DB");
    let conn = db.conn();

    // Columns present.
    let cols = table_columns(&conn, "memories");
    for required in [
        "prior_rid",
        "resolution_kind",
        "dismissal_reason",
        "confidence_at_write",
    ] {
        assert!(
            cols.iter().any(|c| c == required),
            "v26 migration: missing column {required} after re-open"
        );
    }

    // Indexes recreated.
    assert!(
        index_exists(&conn, "idx_memories_prior_rid"),
        "v26 migration: missing partial index idx_memories_prior_rid"
    );
    assert!(
        index_exists(&conn, "idx_memories_resolution_kind"),
        "v26 migration: missing partial index idx_memories_resolution_kind"
    );

    // Source normalization: legacy-freetext row should be 'user' now.
    let normalized: String = conn
        .query_row(
            "SELECT source FROM memories WHERE rid = '01900000-0000-7000-8000-000000000003'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        normalized, "user",
        "v26 migration must coerce non-enum source to 'user'"
    );

    // Enum-valid rows preserved.
    let preserved: String = conn
        .query_row(
            "SELECT source FROM memories WHERE rid = '01900000-0000-7000-8000-000000000002'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        preserved, "inference",
        "v26 migration must preserve enum-valid source values"
    );

    // Normalization log written to meta — count includes 1 affected row.
    let log: String = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'source_normalization_log_v26'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(
        log.contains("normalized 1 rows"),
        "v26 migration must log normalization count, got: {log}"
    );
}

#[test]
fn schema_v26_migration_replay_is_idempotent() {
    // Replay-resilience: rewinding meta to 25 on a DB that's already at v26
    // schema must not break the second open. This is the same shape as the
    // v0.7.3 / v0.7.8 replay tests above, repeated for the v26 migration so
    // the property is locked at this specific migration boundary too.
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    // First open: fresh v26.
    {
        let _db = YantrikDB::new(path, 8).unwrap();
    }
    // Rewind meta.
    {
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '25')",
            [],
        )
        .unwrap();
    }
    // Second open: MIGRATE_V25_TO_V26 re-runs against already-v26 schema.
    // run_migration_idempotent swallows the duplicate-column errors.
    let db = YantrikDB::new(path, 8)
        .expect("v26 migration runner must heal rewound-meta deployments on a v26-schema DB");

    // Sanity: a write still works end-to-end after the heal.
    db.record(
        "post-v26-heal smoke",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(1.0, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();
}

#[test]
fn schema_v43_fresh_install_has_typed_synthesis_lifecycle() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let conn = db.conn();
    let cols = table_columns(&conn, "memories");
    for required in [
        "synthesis_axis",
        "synthesis_granularity",
        "synthesis_logical_key",
        "synthesis_evidence_version",
        "synthesis_generation_hlc",
        "synthesis_state",
    ] {
        assert!(
            cols.iter().any(|col| col == required),
            "v43 fresh schema missing memories.{required}"
        );
    }
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master \
             WHERE type = 'table' AND name = 'synthesis_dependencies'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap(),
        1
    );
    for index in [
        "idx_synthesis_dependencies_source",
        "idx_synthesis_dependencies_synthesis",
    ] {
        assert!(
            index_exists(&conn, index),
            "v42 fresh schema missing {index}"
        );
    }
}

#[test]
fn schema_v46_fresh_install_has_rollup_outcome_ledger() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let conn = db.conn();
    for table in [
        "rollup_impressions",
        "rollup_impression_children",
        "rollup_impression_outcomes",
        "rollup_impression_additions",
    ] {
        assert_eq!(
            conn.query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
                [table],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
            1,
            "v46 fresh schema missing {table}"
        );
    }
    for index in [
        "idx_rollup_impressions_rollup",
        "idx_rollup_impressions_query",
        "idx_rollup_impression_children_child",
        "idx_rollup_impression_outcomes_created",
        "idx_rollup_impression_additions_child",
    ] {
        assert!(
            index_exists(&conn, index),
            "v46 fresh schema missing {index}"
        );
    }
    for expected in ["requested_count", "query_shape"] {
        assert!(
            table_columns(&conn, "rollup_impressions")
                .iter()
                .any(|column| column == expected),
            "v46 fresh schema missing rollup_impressions.{expected}"
        );
    }
    assert!(
        table_columns(&conn, "rollup_impression_children")
            .iter()
            .any(|column| column == "score"),
        "v46 fresh schema missing rollup_impression_children.score"
    );
}

#[test]
fn schema_v46_migration_adds_omission_features() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE rollup_impressions (impression_id TEXT PRIMARY KEY); \
         CREATE TABLE rollup_impression_children ( \
             impression_id TEXT NOT NULL, child_rid TEXT NOT NULL, rank INTEGER NOT NULL, \
             PRIMARY KEY (impression_id, child_rid) \
         );",
    )
    .unwrap();
    conn.execute_batch(crate::base::schema::MIGRATE_V45_TO_V46)
        .unwrap();

    let impression_cols = table_columns(&conn, "rollup_impressions");
    assert!(impression_cols.iter().any(|col| col == "requested_count"));
    assert!(impression_cols.iter().any(|col| col == "query_shape"));
    assert!(table_columns(&conn, "rollup_impression_children")
        .iter()
        .any(|col| col == "score"));
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
            ["rollup_impression_additions"],
            |row| row.get::<_, i64>(0),
        )
        .unwrap(),
        1
    );
    assert!(index_exists(&conn, "idx_rollup_impression_additions_child"));
}

#[test]
fn schema_v46_migration_bootstraps_child_ledger_skipped_by_v43() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute_batch(crate::base::schema::MIGRATE_V44_TO_V45)
        .unwrap();
    conn.execute_batch(crate::base::schema::MIGRATE_V45_TO_V46)
        .unwrap();

    assert!(table_columns(&conn, "rollup_impression_children")
        .iter()
        .any(|column| column == "score"));
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
            ["rollup_impression_additions"],
            |row| row.get::<_, i64>(0),
        )
        .unwrap(),
        1
    );
}

#[test]
fn schema_v45_migration_adds_rollup_outcome_finalization() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute(
        "CREATE TABLE rollup_impressions (impression_id TEXT PRIMARY KEY)",
        [],
    )
    .unwrap();
    conn.execute_batch(crate::base::schema::MIGRATE_V44_TO_V45)
        .unwrap();

    let cols = table_columns(&conn, "rollup_impressions");
    assert!(cols.iter().any(|col| col == "outcome_payload_hash"));
    assert!(cols.iter().any(|col| col == "outcome_finalized_at"));
}

#[test]
fn schema_v45_migration_bootstraps_ledger_skipped_by_v43() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute_batch(crate::base::schema::MIGRATE_V44_TO_V45)
        .unwrap();

    let cols = table_columns(&conn, "rollup_impressions");
    for expected in [
        "rollup_rid",
        "expansion_payload_hash",
        "outcome_payload_hash",
        "outcome_finalized_at",
    ] {
        assert!(cols.iter().any(|col| col == expected), "missing {expected}");
    }
}

#[test]
fn schema_v43_migration_adds_synthesis_generation_clock() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE memories (rid TEXT PRIMARY KEY)", [])
        .unwrap();
    conn.execute_batch(crate::base::schema::MIGRATE_V42_TO_V43)
        .unwrap();

    let cols = table_columns(&conn, "memories");
    assert!(cols.iter().any(|col| col == "synthesis_generation_hlc"));
}

#[test]
fn schema_v42_migration_adds_the_same_synthesis_surface() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE memories (rid TEXT PRIMARY KEY)", [])
        .unwrap();
    conn.execute_batch(crate::base::schema::MIGRATE_V41_TO_V42)
        .unwrap();

    let cols = table_columns(&conn, "memories");
    for required in [
        "synthesis_axis",
        "synthesis_granularity",
        "synthesis_logical_key",
        "synthesis_evidence_version",
        "synthesis_state",
    ] {
        assert!(cols.iter().any(|col| col == required));
    }
    assert!(index_exists(&conn, "idx_synthesis_dependencies_source"));
    assert!(index_exists(&conn, "idx_synthesis_dependencies_synthesis"));

    conn.execute(
        "INSERT INTO memories (rid, synthesis_granularity, synthesis_state) \
         VALUES ('ordinary', NULL, NULL), ('synth', 'atomic', 'verified')",
        [],
    )
    .unwrap();
    assert!(conn
        .execute(
            "INSERT INTO memories (rid, synthesis_granularity) VALUES ('bad-g', 'session')",
            [],
        )
        .is_err());
    assert!(conn
        .execute(
            "INSERT INTO memories (rid, synthesis_state) VALUES ('bad-s', 'active')",
            [],
        )
        .is_err());
}

// =====================================================================
// Issue #146 — a failing migration statement must name itself.
//
// CI produced `database error: incomplete input` from inside the
// constructor, once, on one platform. That message names nothing: SQLite
// reports a truncated statement at the END of the input, so
// `sqlite3_error_offset()` returns -1 and rusqlite falls back from the
// SQL-carrying SqlInputError to a bare SqliteFailure. The migration
// runner is the one open-path site where SQL is DERIVED (split on `;`,
// line comments stripped) rather than constant — the one place a
// truncated statement could be of our own making — so a propagated
// error there must carry the statement text.
// =====================================================================

#[test]
fn failing_migration_statement_names_itself_in_the_error() {
    use rusqlite::Connection;
    let conn = Connection::open_in_memory().unwrap();
    // The table must EXIST for the truncation to be the reported error:
    // on a bare connection the same statement fails with "no such table:
    // memories" — SQLite resolves the ALTER target before finishing the
    // parse — and that message is in the idempotent-replay swallow list,
    // so the runner silently succeeds. (First version of this test found
    // that out the hard way. It also means the swallow list can mask a
    // genuinely broken statement whose table is absent — acceptable for
    // replay-resilience, but worth knowing.)
    conn.execute("CREATE TABLE memories (rid TEXT PRIMARY KEY)", [])
        .unwrap();
    // `ALTER TABLE memories ADD` is a prefix of a valid statement —
    // prepare fails with exactly the "incomplete input" from #146, and
    // that message is not in the swallow list.
    let err = YantrikDB::run_migration_idempotent(&conn, "ALTER TABLE memories ADD")
        .expect_err("a truncated statement must not succeed");
    let msg = err.to_string();
    assert!(
        msg.contains("migration statement"),
        "error must be stage-tagged, got: {msg}"
    );
    assert!(
        msg.contains("ALTER TABLE memories ADD"),
        "error must carry the statement it choked on, got: {msg}"
    );
}

#[test]
fn swallowed_replay_errors_still_do_not_leak_a_stage_error() {
    // The other direction: the idempotent-replay swallow list must be
    // unaffected by the stage-tagging change. "no such table" is on the
    // list; the batch must succeed even though its statement fails.
    use rusqlite::Connection;
    let conn = Connection::open_in_memory().unwrap();
    YantrikDB::run_migration_idempotent(&conn, "DROP TABLE definitely_not_a_table;")
        .expect("swallowed replay errors must not become failures");
}

// =====================================================================
// Issue #146, SOLVED HALF — the migration splitter must not cut trigger
// bodies. The first stage-tagged recurrence (PR #159 CI) named the exact
// statement: `CREATE TRIGGER ... BEGIN INSERT ...` truncated before its
// `; END` because run_migration_idempotent splits batches on bare `;`.
// The runner's own doc flagged `;`-in-string-literals as the hazard;
// trigger bodies were the case the caveat didn't name. Until this fix,
// ANY migration replay crossing a trigger-bearing migration failed.
// =====================================================================

#[test]
fn migration_splitter_keeps_trigger_bodies_intact() {
    use rusqlite::Connection;
    let conn = Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, txt TEXT)", [])
        .unwrap();
    conn.execute("CREATE TABLE t_log (txt TEXT)", []).unwrap();
    // A batch in exactly the shape that failed in CI: statements before and
    // after a trigger whose body contains its own semicolons.
    let batch = "
        CREATE INDEX IF NOT EXISTS idx_t_txt ON t(txt);
        CREATE TRIGGER IF NOT EXISTS t_insert AFTER INSERT ON t BEGIN
            INSERT INTO t_log(txt) VALUES (new.txt);
            UPDATE t SET txt = txt WHERE id = new.id;
        END;
        CREATE INDEX IF NOT EXISTS idx_t_id ON t(id);
    ";
    YantrikDB::run_migration_idempotent(&conn, batch)
        .expect("a trigger body must survive the splitter");
    // The trigger must actually work — not merely have parsed.
    conn.execute("INSERT INTO t (txt) VALUES ('hello')", [])
        .unwrap();
    let logged: String = conn
        .query_row("SELECT txt FROM t_log LIMIT 1", [], |r| r.get(0))
        .unwrap();
    assert_eq!(logged, "hello");
    // Replay safety unchanged: running the same batch again succeeds.
    YantrikDB::run_migration_idempotent(&conn, batch)
        .expect("replay of a trigger-bearing batch must stay idempotent");
}

// =====================================================================
// #160 cold-review adversarials (found by Codex, empirically reproduced
// against the first depth-scanner fix — both failed it). A quoted
// identifier `"begin"` jammed the scanner's depth counter, grouping
// statements so that a swallowed already-exists SILENTLY DROPPED the
// rest of the group; and an identifier containing a semicolon split
// mid-name. sqlite3_complete understands both because it is SQLite.
// =====================================================================

#[test]
fn quoted_begin_identifier_does_not_group_statements() {
    use rusqlite::Connection;
    let conn = Connection::open_in_memory().unwrap();
    // Precreate "begin" so the first statement hits already-exists and the
    // swallow list eats it — the second statement must STILL run alone.
    conn.execute("CREATE TABLE \"begin\" (id INTEGER)", [])
        .unwrap();
    let batch =
        "CREATE TABLE \"begin\" (id INTEGER); CREATE TABLE applied_after_replay (id INTEGER);";
    YantrikDB::run_migration_idempotent(&conn, batch).unwrap();
    let n: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'applied_after_replay'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        n, 1,
        "statement after a swallowed error must not be silently lost"
    );
}

#[test]
fn semicolon_inside_quoted_identifier_does_not_split() {
    use rusqlite::Connection;
    let conn = Connection::open_in_memory().unwrap();
    let batch = "CREATE TABLE \"semi;colon\" (id INTEGER); CREATE TABLE after_semi (id INTEGER);";
    YantrikDB::run_migration_idempotent(&conn, batch).unwrap();
    for t in ["semi;colon", "after_semi"] {
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = ?1",
                rusqlite::params![t],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1, "table {t:?} must exist");
    }
}

#[test]
fn block_comments_with_semicolons_do_not_split() {
    use rusqlite::Connection;
    let conn = Connection::open_in_memory().unwrap();
    let batch =
        "CREATE TABLE c1 (id INTEGER) /* note; with; semis */; CREATE TABLE c2 (id INTEGER);";
    YantrikDB::run_migration_idempotent(&conn, batch).unwrap();
    let n: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE name IN ('c1','c2')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 2);
}

// =====================================================================
// v48 (#149) — valid time as first-class indexed columns.
//
// event_time_min / event_time_max move out of the metadata JSON into REAL
// columns on `memories`, plus the partial index recall's range scans use.
// Three paths, mirroring the v26/v46 suites:
//   1. Fresh install has the columns and the index via SCHEMA_SQL.
//   2. A v47 store upgrades: columns backfilled from the JSON, index created.
//   3. The backfill's json_valid guard skips ciphertext metadata rows
//      (encrypted stores) instead of erroring the migration.
// =====================================================================

#[test]
fn schema_v48_fresh_install_has_event_time_columns_and_index() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let conn = db.conn();

    let cols = table_columns(&conn, "memories");
    for required in ["event_time_min", "event_time_max"] {
        assert!(
            cols.iter().any(|c| c == required),
            "v48: fresh-install memories table missing column {required}"
        );
    }
    assert!(
        index_exists(&conn, "idx_memories_event_time"),
        "v48: fresh-install missing partial index idx_memories_event_time"
    );
}

#[test]
fn schema_v48_migration_backfills_event_time_and_creates_index() {
    // Simulate a v47 store carrying event time only in the metadata JSON:
    // open at current schema, record a date-bearing memory (which stamps
    // both the JSON and the columns), then NULL the columns (the exact
    // post-ALTER state a real v47 table reaches), drop the index, rewind
    // meta to 47, and reopen. MIGRATE_V47_TO_V48's ALTERs replay as no-ops
    // (idempotent runner), the backfill must copy the JSON values into the
    // columns, and the index must be recreated.
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    let rid;
    {
        let db = YantrikDB::new(path, 8).unwrap();
        rid = db
            .record(
                "met Alice on 2024-03-15",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(1.0, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
    }

    {
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.execute(
            "UPDATE memories SET event_time_min = NULL, event_time_max = NULL",
            [],
        )
        .unwrap();
        conn.execute("DROP INDEX idx_memories_event_time", [])
            .unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '47')",
            [],
        )
        .unwrap();
    }

    {
        let db = YantrikDB::new(path, 8).unwrap();
        let conn = db.conn();
        let (col_min, json_min, col_max, json_max): (
            Option<f64>,
            Option<f64>,
            Option<f64>,
            Option<f64>,
        ) = conn
            .query_row(
                "SELECT event_time_min, json_extract(metadata, '$.event_time_min'), \
                        event_time_max, json_extract(metadata, '$.event_time_max') \
                 FROM memories WHERE rid = ?1",
                [rid.as_str()],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .unwrap();
        assert!(
            json_min.is_some(),
            "precondition: the date-bearing record's metadata JSON carries event_time_min"
        );
        assert_eq!(
            col_min, json_min,
            "v48 backfill must copy event_time_min out of the metadata JSON"
        );
        assert_eq!(
            col_max, json_max,
            "v48 backfill must copy event_time_max out of the metadata JSON"
        );
        let idx: String = conn
            .query_row(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_memories_event_time'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(idx, "idx_memories_event_time");
    }
}

#[test]
fn schema_v48_backfill_skips_ciphertext_metadata() {
    // The json_valid guard is load-bearing: on encrypted stores the metadata
    // column holds ciphertext, and an unguarded json_extract would error the
    // whole migration. Ciphertext rows must keep NULL columns; valid-JSON
    // rows must backfill; and the batch must not error.
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE memories (rid TEXT PRIMARY KEY, \
             namespace TEXT NOT NULL DEFAULT 'default', metadata TEXT);",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO memories (rid, metadata) VALUES \
         ('plain', '{\"event_time_min\": 100.5, \"event_time_max\": 200.5}'), \
         ('cipher', 'AGEv1:definitely-not-json'), \
         ('absent', NULL)",
        [],
    )
    .unwrap();

    conn.execute_batch(crate::base::schema::MIGRATE_V47_TO_V48)
        .unwrap();

    let get = |rid: &str| -> (Option<f64>, Option<f64>) {
        conn.query_row(
            "SELECT event_time_min, event_time_max FROM memories WHERE rid = ?1",
            [rid],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap()
    };
    assert_eq!(get("plain"), (Some(100.5), Some(200.5)));
    assert_eq!(
        get("cipher"),
        (None, None),
        "ciphertext metadata must be skipped, not extracted"
    );
    assert_eq!(get("absent"), (None, None));
    assert!(index_exists(&conn, "idx_memories_event_time"));
}

// =====================================================================
// v49 — the persisted entity normalization key (entity_name_norm).
//
// Reviewer finding (#188 follow-up): recall_thread resolved requested
// entity names by scanning SELECT DISTINCT entity_name FROM
// memory_entities and Unicode-lowercasing EVERY name in Rust per request
// — O(V) over the global entity vocabulary on every call. v49 persists
// the normalized key; the backfill must run in RUST because SQL LOWER()
// is ASCII-only and would corrupt non-ASCII names.
// =====================================================================

#[test]
fn schema_v49_fresh_install_has_entity_norm_column_and_index() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let conn = db.conn();
    let cols = table_columns(&conn, "memory_entities");
    assert!(
        cols.iter().any(|c| c == "entity_name_norm"),
        "v49: fresh-install memory_entities table missing column entity_name_norm"
    );
    assert!(
        index_exists(&conn, "idx_memory_entities_norm"),
        "v49: fresh-install missing index idx_memory_entities_norm"
    );
}

#[test]
fn schema_v49_migration_backfills_unicode_lowercase_in_rust() {
    // A store built at v48 has memory_entities rows with no
    // entity_name_norm. SQL cannot backfill them: LOWER() is ASCII-only
    // ('MÜNSTER' -> 'mÜnster'), so open() must do it in Rust (the
    // entity_norm_backfill stage). Simulate: open at current schema,
    // insert rows with NULL norm (the exact post-ALTER state a real v48
    // table reaches), drop the index, rewind meta to 48, and reopen. The
    // ALTER replays as a no-op (idempotent runner), the index is
    // recreated, and the backfill must produce the correct Unicode
    // lowercase.
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    {
        let _db = YantrikDB::new(path, 8).unwrap();
    }
    {
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.execute(
            "INSERT INTO memory_entities (memory_rid, entity_name) VALUES \
             ('m1', 'Münster'), ('m2', 'MÜNSTER'), ('m3', 'Alpha')",
            [],
        )
        .unwrap();
        conn.execute("DROP INDEX idx_memory_entities_norm", [])
            .unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '48')",
            [],
        )
        .unwrap();
    }
    {
        let db = YantrikDB::new(path, 8).unwrap();
        let conn = db.conn();
        let get_norm = |rid: &str| -> Option<String> {
            conn.query_row(
                "SELECT entity_name_norm FROM memory_entities WHERE memory_rid = ?1",
                [rid],
                |r| r.get(0),
            )
            .unwrap()
        };
        assert_eq!(
            get_norm("m1").as_deref(),
            Some("münster"),
            "backfill must be Rust to_lowercase, not ASCII-only SQL LOWER()"
        );
        assert_eq!(
            get_norm("m2").as_deref(),
            Some("münster"),
            "'MÜNSTER' must fold to 'münster' — SQL LOWER() would give 'mÜnster'"
        );
        assert_eq!(get_norm("m3").as_deref(), Some("alpha"));
        assert!(
            index_exists(&conn, "idx_memory_entities_norm"),
            "v49 migration replay must recreate idx_memory_entities_norm"
        );
    }
}

#[test]
fn schema_v49_backfill_skips_already_stamped_rows() {
    // The open()-time backfill is guarded on entity_name_norm IS NULL:
    // rows already carrying a norm value are never rewritten, and a
    // rewind-then-reopen replay must not error or clobber them.
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();
    {
        let _db = YantrikDB::new(path, 8).unwrap();
    }
    {
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.execute(
            "INSERT INTO memory_entities (memory_rid, entity_name, entity_name_norm) \
             VALUES ('m1', 'Kept', 'kept')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '48')",
            [],
        )
        .unwrap();
    }
    {
        let _db = YantrikDB::new(path, 8).unwrap();
    }
    let conn = rusqlite::Connection::open(path).unwrap();
    let norm: Option<String> = conn
        .query_row(
            "SELECT entity_name_norm FROM memory_entities WHERE memory_rid = 'm1'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(norm.as_deref(), Some("kept"));
}
