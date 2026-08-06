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
