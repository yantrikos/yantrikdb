use rusqlite::params;

use crate::error::Result;
use crate::types::Stats;

use super::{now, YantrikDB};

impl YantrikDB {
    /// Get engine statistics. Optionally filter memory counts by namespace.
    pub fn stats(&self, namespace: Option<&str>) -> Result<Stats> {
        let conn = self.conn.lock();
        let ns_filter = namespace.map(|ns| format!(" AND namespace = '{}'", ns.replace('\'', "''"))).unwrap_or_default();
        let active = conn.query_row(
            &format!("SELECT COUNT(*) FROM memories WHERE consolidation_status = 'active'{}", ns_filter),
            [], |row| row.get(0),
        )?;
        let consolidated = conn.query_row(
            &format!("SELECT COUNT(*) FROM memories WHERE consolidation_status = 'consolidated'{}", ns_filter),
            [], |row| row.get(0),
        )?;
        let tombstoned = conn.query_row(
            &format!("SELECT COUNT(*) FROM memories WHERE consolidation_status = 'tombstoned'{}", ns_filter),
            [], |row| row.get(0),
        )?;
        let archived = conn.query_row(
            &format!("SELECT COUNT(*) FROM memories WHERE storage_tier = 'cold'{}", ns_filter),
            [], |row| row.get(0),
        )?;
        let edges = conn.query_row(
            "SELECT COUNT(*) FROM edges WHERE tombstoned = 0",
            [], |row| row.get(0),
        )?;
        let entities = conn.query_row(
            "SELECT COUNT(*) FROM entities",
            [], |row| row.get(0),
        )?;
        let operations = conn.query_row(
            "SELECT COUNT(*) FROM oplog",
            [], |row| row.get(0),
        )?;
        let open_conflicts = conn.query_row(
            "SELECT COUNT(*) FROM conflicts WHERE status = 'open'",
            [], |row| row.get(0),
        )?;
        let resolved_conflicts = conn.query_row(
            "SELECT COUNT(*) FROM conflicts WHERE status IN ('resolved', 'dismissed')",
            [], |row| row.get(0),
        )?;
        let pending_triggers = conn.query_row(
            "SELECT COUNT(*) FROM trigger_log WHERE status = 'pending'",
            [], |row| row.get(0),
        )?;
        let active_patterns = conn.query_row(
            "SELECT COUNT(*) FROM patterns WHERE status = 'active'",
            [], |row| row.get(0),
        )?;
        drop(conn);

        Ok(Stats {
            active_memories: active,
            consolidated_memories: consolidated,
            tombstoned_memories: tombstoned,
            archived_memories: archived,
            edges,
            entities,
            operations,
            open_conflicts,
            resolved_conflicts,
            pending_triggers,
            active_patterns,
            scoring_cache_entries: self.scoring_cache.read().len(),
            vec_index_entries: self.vec_index.len(),
            graph_index_entities: self.graph_index.read().entity_count(),
            graph_index_edges: self.graph_index.read().edge_count(),
        })
    }

    /// Append an operation to the oplog with HLC and optional embedding hash.
    pub fn log_op(
        &self,
        op_type: &str,
        target_rid: Option<&str>,
        payload: &serde_json::Value,
        emb_hash: Option<&[u8]>,
    ) -> Result<String> {
        let op_id = crate::id::new_id();
        let hlc_ts = self.tick_hlc();
        let hlc_bytes = hlc_ts.to_bytes().to_vec();
        let payload_str = serde_json::to_string(payload)?;

        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO oplog (op_id, op_type, timestamp, target_rid, payload, \
             actor_id, hlc, embedding_hash, origin_actor, applied) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 1)",
            params![
                op_id,
                op_type,
                now(),
                target_rid,
                payload_str,
                self.actor_id,
                hlc_bytes,
                emb_hash,
                self.actor_id,
            ],
        )?;
        Ok(op_id)
    }
    /// **Decoupled write path RFC, Phase 1.**
    ///
    /// Append a *pending* operation to the oplog (applied=0) carrying the
    /// full embedding bytes. Background materializer workers will later drain
    /// these and apply them to the in-memory indexes (memories table,
    /// vec_index, graph_index, scoring_cache), flipping `applied` to 1.
    ///
    /// This is the "WAL append" step from the RFC's freeway diagram. Foreground
    /// `record()` does not call this yet — Phase 4 of the RFC flips that. For
    /// Phase 1 (this version), the API is exposed for tests and for Phase 3
    /// worker scaffolding.
    ///
    /// Idempotent on op_id: if the same op_id is appended twice (e.g., via
    /// crash-restart replay), the second INSERT is silently skipped.
    pub fn log_op_pending(
        &self,
        op_type: &str,
        target_rid: Option<&str>,
        payload: &serde_json::Value,
        emb_hash: Option<&[u8]>,
        embedding: Option<&[u8]>,
    ) -> Result<String> {
        let op_id = crate::id::new_id();
        let hlc_ts = self.tick_hlc();
        let hlc_bytes = hlc_ts.to_bytes().to_vec();
        let payload_str = serde_json::to_string(payload)?;

        // Backpressure: bound the pending-op set so an unbounded writer
        // burst can't blow up RSS or starve the materializer.
        // The bound is intentionally permissive in Phase 1; Phase 3 will
        // add tunable per-namespace partitioning.
        const MAX_PENDING_OPS: i64 = 10_000;
        let pending_now: i64 = {
            let conn = self.read_conn();
            conn.query_row(
                "SELECT COUNT(*) FROM oplog WHERE applied = 0",
                [],
                |row| row.get(0),
            )?
        };
        if pending_now >= MAX_PENDING_OPS {
            return Err(crate::error::YantrikDbError::Backpressure {
                pending: pending_now,
                max: MAX_PENDING_OPS,
                retry_after_ms: 50,
            });
        }

        let conn = self.conn.lock();
        conn.execute(
            "INSERT OR IGNORE INTO oplog \
             (op_id, op_type, timestamp, target_rid, payload, \
              actor_id, hlc, embedding_hash, origin_actor, applied, embedding) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 0, ?10)",
            params![
                op_id,
                op_type,
                now(),
                target_rid,
                payload_str,
                self.actor_id,
                hlc_bytes,
                emb_hash,
                self.actor_id,
                embedding,
            ],
        )?;
        Ok(op_id)
    }

    /// **Decoupled write path RFC, Phase 1.**
    ///
    /// Count of pending oplog entries (applied=0). Used by tests and by the
    /// background materializer to decide whether to wake up.
    pub fn count_pending_ops(&self) -> Result<i64> {
        let conn = self.read_conn();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM oplog WHERE applied = 0",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// **Decoupled write path RFC, Phase 1.**
    ///
    /// Mark a pending oplog entry as materialized. Called by the background
    /// worker after it has applied the op to the in-memory indexes.
    /// Idempotent: marking an already-applied op is a no-op.
    pub fn mark_op_applied(&self, op_id: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE oplog SET applied = 1 WHERE op_id = ?1",
            params![op_id],
        )?;
        Ok(())
    }

    /// **Decoupled write path RFC, Phase 3 scaffolding.**
    ///
    /// Drain up to `limit` pending oplog entries (applied=0) and apply each
    /// to the engine's in-memory indexes. Returns the number of ops actually
    /// applied this pass. Idempotent on re-entry — already-applied ops are
    /// skipped via the `applied = 0` filter.
    ///
    /// This is the worker's main-loop body as a sync function. Phase 3.5
    /// will wrap it in a thread spawn + condvar wake + Drop-based shutdown.
    /// Phase 4 will switch foreground `record()` to call `log_op_pending()`
    /// instead of materializing inline, at which point this drain becomes
    /// the production write-completion path.
    ///
    /// Op-type dispatch in Phase 3 is intentionally a stub: each op type
    /// has a placeholder materializer that just marks the op applied. Phase 4
    /// fills in the actual application logic (memories INSERT, vec_index
    /// update, graph_index update, scoring_cache insert) — and at that point
    /// foreground record() can stop doing it inline.
    pub fn apply_pending_ops_once(&self, limit: usize) -> Result<usize> {
        let pending: Vec<(String, String)> = {
            let conn = self.read_conn();
            let mut stmt = conn.prepare(
                "SELECT op_id, op_type FROM oplog \
                 WHERE applied = 0 \
                 ORDER BY hlc, op_id \
                 LIMIT ?1",
            )?;
            let rows = stmt.query_map(params![limit as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };

        let mut applied = 0usize;
        for (op_id, op_type) in &pending {
            match op_type.as_str() {
                "record" | "forget" | "relate" | "correct" | "consolidate" => {
                    tracing::trace!(
                        target: "yantrikdb::ingest::materialize",
                        op_id = %op_id,
                        op_type = %op_type,
                        "phase 3 stub: marking pending op as applied without inline materialization"
                    );
                    self.mark_op_applied(op_id)?;
                    applied += 1;
                }
                other => {
                    tracing::warn!(
                        target: "yantrikdb::ingest::materialize",
                        op_id = %op_id,
                        op_type = %other,
                        "unknown op_type in pending oplog — skipping"
                    );
                }
            }
        }

        Ok(applied)
    }
}

#[cfg(test)]
mod pending_ops_tests {
    use super::*;
    use crate::YantrikDB;

    fn open_test_db() -> YantrikDB {
        // Use :memory: so tests don't touch disk and migrations are fresh.
        YantrikDB::new(":memory:", 64).expect("open test db")
    }

    fn fake_embedding(seed: f32, dim: usize) -> Vec<u8> {
        let raw: Vec<f32> = (0..dim).map(|i| (seed + i as f32) * 0.1).collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        let normalized: Vec<f32> = raw.iter().map(|x| x / norm).collect();
        crate::serde_helpers::serialize_f32(&normalized)
    }

    #[test]
    fn pending_op_round_trip() {
        let db = open_test_db();
        assert_eq!(db.count_pending_ops().unwrap(), 0, "fresh db has no pending");

        let payload = serde_json::json!({
            "rid": "test_rid_1",
            "type": "episodic",
            "text": "first pending op",
        });
        let emb_bytes = fake_embedding(1.0, 64);
        let op_id = db
            .log_op_pending("record", Some("test_rid_1"), &payload, None, Some(&emb_bytes))
            .expect("log_op_pending");

        assert_eq!(db.count_pending_ops().unwrap(), 1, "one pending op after append");

        db.mark_op_applied(&op_id).expect("mark applied");
        assert_eq!(db.count_pending_ops().unwrap(), 0, "no pending after mark");
    }

    #[test]
    fn pending_op_idempotent_on_double_append() {
        let db = open_test_db();
        let payload = serde_json::json!({"rid": "rid_idem"});
        let emb_bytes = fake_embedding(2.0, 64);

        let op_id_a = db
            .log_op_pending("record", Some("rid_idem"), &payload, None, Some(&emb_bytes))
            .unwrap();
        let op_id_b = db
            .log_op_pending("record", Some("rid_idem"), &payload, None, Some(&emb_bytes))
            .unwrap();

        // Two distinct op_ids generated (uuid7), but each is a separate row.
        assert_ne!(op_id_a, op_id_b, "each call generates a distinct op_id");
        assert_eq!(db.count_pending_ops().unwrap(), 2);
    }

    #[test]
    fn pending_op_persists_embedding_blob() {
        let db = open_test_db();
        let emb_bytes = fake_embedding(3.0, 64);
        let op_id = db
            .log_op_pending(
                "record",
                Some("rid_emb"),
                &serde_json::json!({}),
                None,
                Some(&emb_bytes),
            )
            .unwrap();

        let conn = db.read_conn();
        let stored: Option<Vec<u8>> = conn
            .query_row(
                "SELECT embedding FROM oplog WHERE op_id = ?1",
                params![op_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            stored.as_deref(),
            Some(emb_bytes.as_slice()),
            "embedding bytes round-trip exactly"
        );
    }

    #[test]
    fn mark_op_applied_idempotent() {
        let db = open_test_db();
        let op_id = db
            .log_op_pending("record", None, &serde_json::json!({}), None, None)
            .unwrap();
        db.mark_op_applied(&op_id).unwrap();
        // Calling again on already-applied op is a no-op.
        db.mark_op_applied(&op_id).unwrap();
        assert_eq!(db.count_pending_ops().unwrap(), 0);
    }

    #[test]
    fn count_pending_ignores_applied_ops() {
        let db = open_test_db();
        // Old log_op writes applied=1 directly.
        db.log_op("record", Some("rid_old"), &serde_json::json!({}), None)
            .unwrap();
        assert_eq!(
            db.count_pending_ops().unwrap(),
            0,
            "log_op (applied=1) is not pending"
        );

        // log_op_pending writes applied=0.
        db.log_op_pending("record", Some("rid_new"), &serde_json::json!({}), None, None)
            .unwrap();
        assert_eq!(db.count_pending_ops().unwrap(), 1);
    }

    #[test]
    fn backpressure_engages_at_max_pending() {
        // Saturate the queue with 10_000 pending ops, then verify the
        // 10_001st returns Error::Backpressure with sane fields.
        let db = open_test_db();
        for i in 0..10_000 {
            db.log_op_pending(
                "record",
                Some(&format!("rid_{i}")),
                &serde_json::json!({}),
                None,
                None,
            )
            .expect("first 10k succeed");
        }
        assert_eq!(db.count_pending_ops().unwrap(), 10_000);

        let err = db
            .log_op_pending("record", Some("rid_overflow"), &serde_json::json!({}), None, None)
            .expect_err("11k must fail with backpressure");
        match err {
            crate::error::YantrikDbError::Backpressure { pending, max, retry_after_ms } => {
                assert_eq!(max, 10_000);
                assert_eq!(pending, 10_000);
                assert!(retry_after_ms > 0, "retry hint must be non-zero");
            }
            other => panic!("expected Backpressure, got {other:?}"),
        }

        // After draining one, the next push must succeed (proves backpressure
        // is reactive, not sticky).
        let conn = db.conn.lock();
        conn.execute(
            "UPDATE oplog SET applied = 1 WHERE op_id IN (SELECT op_id FROM oplog WHERE applied = 0 LIMIT 1)",
            [],
        ).unwrap();
        drop(conn);
        db.log_op_pending("record", Some("rid_after_drain"), &serde_json::json!({}), None, None)
            .expect("succeeds after one drained");
    }

    #[test]
    fn apply_pending_drains_then_marks() {
        let db = open_test_db();
        // Seed 3 pending ops of various types.
        for (op_type, target) in [("record", "rid_1"), ("forget", "rid_2"), ("relate", "rid_3")] {
            db.log_op_pending(op_type, Some(target), &serde_json::json!({}), None, None)
                .unwrap();
        }
        assert_eq!(db.count_pending_ops().unwrap(), 3);

        let applied = db.apply_pending_ops_once(10).unwrap();
        assert_eq!(applied, 3, "all 3 pending ops drained in one pass");
        assert_eq!(db.count_pending_ops().unwrap(), 0);
    }

    #[test]
    fn apply_pending_respects_limit() {
        let db = open_test_db();
        for i in 0..5 {
            db.log_op_pending("record", Some(&format!("rid_{i}")), &serde_json::json!({}), None, None)
                .unwrap();
        }
        let applied = db.apply_pending_ops_once(2).unwrap();
        assert_eq!(applied, 2, "only 2 of 5 drained when limit=2");
        assert_eq!(db.count_pending_ops().unwrap(), 3);

        // Subsequent drain picks up the rest.
        let applied2 = db.apply_pending_ops_once(10).unwrap();
        assert_eq!(applied2, 3);
        assert_eq!(db.count_pending_ops().unwrap(), 0);
    }

    #[test]
    fn apply_pending_idempotent_when_empty() {
        let db = open_test_db();
        // No pending ops — drain returns 0 cleanly.
        assert_eq!(db.apply_pending_ops_once(100).unwrap(), 0);
        assert_eq!(db.apply_pending_ops_once(100).unwrap(), 0);
    }

    #[test]
    fn apply_pending_skips_unknown_op_type() {
        let db = open_test_db();
        // Direct INSERT bypassing log_op_pending so we can use a synthetic op_type.
        let conn = db.conn.lock();
        conn.execute(
            "INSERT INTO oplog (op_id, op_type, timestamp, payload, applied) \
             VALUES ('synth_unknown', 'made_up_op', 0.0, '{}', 0)",
            [],
        ).unwrap();
        drop(conn);
        assert_eq!(db.count_pending_ops().unwrap(), 1);

        // Drain doesn't apply unknown op types — they stay pending so a
        // future runtime that knows the op type can drain them.
        let applied = db.apply_pending_ops_once(10).unwrap();
        assert_eq!(applied, 0);
        assert_eq!(db.count_pending_ops().unwrap(), 1, "unknown op_type stays pending");
    }

    #[test]
    fn schema_v25_columns_present() {
        // Open a fresh DB so the canonical SCHEMA_SQL runs and creates
        // memories with the v25 columns. Then verify column metadata.
        let db = open_test_db();
        let conn = db.read_conn();
        let mut stmt = conn.prepare("PRAGMA table_info(memories)").unwrap();
        let cols: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert!(
            cols.contains(&"tombstone_reason".to_string()),
            "tombstone_reason missing — schema v25 not applied"
        );
        assert!(
            cols.contains(&"created_at_unix_micros".to_string()),
            "created_at_unix_micros missing — schema v25 not applied"
        );
        assert!(
            cols.contains(&"embedding_model".to_string()),
            "embedding_model missing — schema v25 not applied"
        );
    }

    #[test]
    fn schema_version_meta_at_25() {
        let db = open_test_db();
        let conn = db.read_conn();
        let v: String = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(v, "25");
    }

    #[test]
    fn schema_v25_indexes_present() {
        let db = open_test_db();
        let conn = db.read_conn();
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='memories'")
            .unwrap();
        let names: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert!(
            names.iter().any(|n| n == "idx_memories_created_at_micros"),
            "idx_memories_created_at_micros missing"
        );
        assert!(
            names.iter().any(|n| n == "idx_memories_embedding_model"),
            "idx_memories_embedding_model missing"
        );
    }
}
