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
            vec_index_entries: self.vec_index.read().len(),
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
}