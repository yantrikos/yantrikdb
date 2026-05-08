use rusqlite::params;

use crate::error::{YantrikDbError, Result};
use crate::scoring;
use crate::types::*;

use super::{now, embedding_hash, YantrikDB};

impl YantrikDB {
    /// Get a single memory by RID.
    #[tracing::instrument(skip(self))]
    pub fn get(&self, rid: &str) -> Result<Option<Memory>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT * FROM memories WHERE rid = ?1",
        )?;

        let result = stmt.query_row(params![rid], |row| {
            Ok((
                row.get::<_, String>("rid")?,
                row.get::<_, String>("type")?,
                row.get::<_, String>("text")?,
                row.get::<_, f64>("created_at")?,
                row.get::<_, f64>("importance")?,
                row.get::<_, f64>("valence")?,
                row.get::<_, f64>("half_life")?,
                row.get::<_, f64>("last_access")?,
                row.get::<_, i64>("access_count")?,
                row.get::<_, String>("consolidation_status")?,
                row.get::<_, String>("storage_tier")?,
                row.get::<_, Option<String>>("consolidated_into")?,
                row.get::<_, String>("metadata")?,
                row.get::<_, String>("namespace")?,
                row.get::<_, f64>("certainty")?,
                row.get::<_, String>("domain")?,
                row.get::<_, String>("source")?,
                row.get::<_, Option<String>>("emotional_state")?,
                row.get::<_, Option<String>>("session_id")?,
                row.get::<_, Option<f64>>("due_at")?,
                row.get::<_, Option<String>>("temporal_kind")?,
            ))
        });

        match result {
            Ok(row) => {
                let text = self.decrypt_text(&row.2)?;
                let meta_str = self.decrypt_text(&row.12)?;
                let metadata: serde_json::Value = serde_json::from_str(&meta_str)
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                Ok(Some(Memory {
                    rid: row.0,
                    memory_type: row.1,
                    text,
                    created_at: row.3,
                    importance: row.4,
                    valence: row.5,
                    half_life: row.6,
                    last_access: row.7,
                    access_count: row.8 as u32,
                    consolidation_status: row.9,
                    storage_tier: row.10,
                    consolidated_into: row.11,
                    metadata,
                    namespace: row.13,
                    certainty: row.14,
                    domain: row.15,
                    source: row.16,
                    emotional_state: row.17,
                    session_id: row.18,
                    due_at: row.19,
                    temporal_kind: row.20,
                }))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Browse memories with optional filters. Returns active memories sorted by the
    /// given field. Useful for auditing stored data without a search query.
    pub fn list_memories(
        &self,
        limit: usize,
        offset: usize,
        domain: Option<&str>,
        memory_type: Option<&str>,
        namespace: Option<&str>,
        sort_by: &str,
    ) -> Result<(Vec<Memory>, usize)> {
        let order = match sort_by {
            "importance" => "importance DESC",
            "last_access" => "last_access DESC",
            _ => "created_at DESC",
        };

        let mut conditions = vec!["consolidation_status = 'active'".to_string()];
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut idx = 1;

        if let Some(d) = domain {
            conditions.push(format!("domain = ?{idx}"));
            param_values.push(Box::new(d.to_string()));
            idx += 1;
        }
        if let Some(mt) = memory_type {
            conditions.push(format!("type = ?{idx}"));
            param_values.push(Box::new(mt.to_string()));
            idx += 1;
        }
        if let Some(ns) = namespace {
            conditions.push(format!("namespace = ?{idx}"));
            param_values.push(Box::new(ns.to_string()));
            idx += 1;
        }

        let where_clause = conditions.join(" AND ");

        // Get total count
        let count_sql = format!("SELECT COUNT(*) FROM memories WHERE {where_clause}");
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();
        let conn = self.conn();
        let total: usize = conn.query_row(&count_sql, params_ref.as_slice(), |row| row.get(0))?;

        // Fetch page
        let sql = format!(
            "SELECT rid, type, text, created_at, importance, valence, half_life, \
             last_access, access_count, consolidation_status, storage_tier, \
             consolidated_into, metadata, namespace, certainty, domain, source, \
             emotional_state, session_id, due_at, temporal_kind \
             FROM memories WHERE {where_clause} ORDER BY {order} LIMIT ?{idx} OFFSET ?{}",
            idx + 1
        );
        param_values.push(Box::new(limit as i64));
        param_values.push(Box::new(offset as i64));
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_ref.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, f64>(3)?,
                row.get::<_, f64>(4)?,
                row.get::<_, f64>(5)?,
                row.get::<_, f64>(6)?,
                row.get::<_, f64>(7)?,
                row.get::<_, i64>(8)?,
                row.get::<_, String>(9)?,
                row.get::<_, String>(10)?,
                row.get::<_, Option<String>>(11)?,
                row.get::<_, String>(12)?,
                row.get::<_, String>(13)?,
                row.get::<_, f64>(14)?,
                row.get::<_, String>(15)?,
                row.get::<_, String>(16)?,
                row.get::<_, Option<String>>(17)?,
                row.get::<_, Option<String>>(18)?,
                row.get::<_, Option<f64>>(19)?,
                row.get::<_, Option<String>>(20)?,
            ))
        })?;

        let mut memories = Vec::new();
        for row in rows {
            let row = row?;
            let text = self.decrypt_text(&row.2)?;
            let meta_str = self.decrypt_text(&row.12)?;
            let metadata: serde_json::Value = serde_json::from_str(&meta_str)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            memories.push(Memory {
                rid: row.0,
                memory_type: row.1,
                text,
                created_at: row.3,
                importance: row.4,
                valence: row.5,
                half_life: row.6,
                last_access: row.7,
                access_count: row.8 as u32,
                consolidation_status: row.9,
                storage_tier: row.10,
                consolidated_into: row.11,
                metadata,
                namespace: row.13,
                certainty: row.14,
                domain: row.15,
                source: row.16,
                emotional_state: row.17,
                session_id: row.18,
                due_at: row.19,
                temporal_kind: row.20,
            });
        }

        Ok((memories, total))
    }

    /// Find memories that have decayed below a threshold.
    #[tracing::instrument(skip(self))]
    pub fn decay(&self, threshold: f64) -> Result<Vec<DecayedMemory>> {
        let ts = now();
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT rid, text, importance, half_life, last_access, type FROM memories \
             WHERE consolidation_status = 'active'",
        )?;

        let mut decayed = Vec::new();
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>("rid")?,
                row.get::<_, String>("text")?,
                row.get::<_, f64>("importance")?,
                row.get::<_, f64>("half_life")?,
                row.get::<_, f64>("last_access")?,
                row.get::<_, String>("type")?,
            ))
        })?;

        for row in rows {
            let (rid, stored_text, importance, half_life, last_access, mem_type) = row?;
            let elapsed = ts - last_access;
            let score = scoring::decay_score(importance, half_life, elapsed);
            if score < threshold {
                let text = self.decrypt_text(&stored_text)?;
                decayed.push(DecayedMemory {
                    rid,
                    text,
                    memory_type: mem_type,
                    original_importance: importance,
                    current_score: score,
                    days_since_access: elapsed / 86400.0,
                });
            }
        }

        Ok(decayed)
    }

    /// **Issue #9 — deterministic tombstone primitive for cluster replication.**
    ///
    /// Sibling of `forget()` that takes caller-supplied timestamp + optional
    /// reason for byte-deterministic follower replay. Used by yantrikdb-server's
    /// cluster-mode applier so replicated tombstones converge to identical
    /// engine state across leader + followers.
    ///
    /// # Contract
    ///
    /// - **Idempotent on missing**: tombstoning a rid that does not exist
    ///   returns `Ok(())` (NOT an error and NOT a `false` flag — different
    ///   from `forget()`). Snapshot-install + log replay overlap means
    ///   double-delete is normal cluster behavior.
    /// - **Idempotent on already-tombstoned**: re-tombstoning a row that
    ///   is already tombstoned returns `Ok(())` without emitting a new
    ///   oplog entry or re-bumping cache state. Replay-safe.
    /// - **Caller-supplied timestamp**: `requested_at_unix_micros` materialized
    ///   into `updated_at` (REAL seconds). No engine `now()` call on this path.
    /// - **Optional reason**: stored in `tombstone_reason TEXT` column (v25).
    ///   NULL when caller passes None.
    ///
    /// Always emits a tombstone marker into the DeltaIndex regardless of
    /// whether the SQL row was newly tombstoned — followers may have the
    /// rid in their delta even if SQL is absent.
    pub fn tombstone_with_rid(
        &self,
        rid: &str,
        reason: Option<&str>,
        requested_at_unix_micros: i64,
    ) -> Result<()> {
        self.tombstone_inner(rid, reason, requested_at_unix_micros)?;
        Ok(())
    }

    /// Internal helper shared by `tombstone_with_rid` and `forget`. Returns
    /// `true` iff the row was newly tombstoned (was active or consolidated
    /// before this call). Returns `false` if rid is missing or already
    /// tombstoned — both treated as idempotent successful no-ops.
    fn tombstone_inner(
        &self,
        rid: &str,
        reason: Option<&str>,
        ts_micros: i64,
    ) -> Result<bool> {
        let ts_secs = (ts_micros as f64) / 1_000_000.0;

        // UPDATE with state filter so changes==1 means "was newly tombstoned".
        // changes==0 means rid missing OR already tombstoned (both: no-op).
        let was_newly_tombstoned = {
            let conn = self.conn();
            let changes = conn.execute(
                "UPDATE memories SET consolidation_status = 'tombstoned', \
                 updated_at = ?1, tombstone_reason = ?2 \
                 WHERE rid = ?3 AND consolidation_status != 'tombstoned'",
                params![ts_secs, reason, rid],
            )?;
            changes > 0
        };

        // Always emit a delta tombstone so search() filters it out even
        // before SQL has applied. Cluster followers may have the rid in
        // their delta from a recent record_with_rid that has not yet
        // compacted into cold; the tombstone marker covers that window.
        let seq = self.vec_seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.vec_index.tombstone(rid, seq);

        // Engine-internal index updates only when the row was newly tombstoned
        // (replay-safe: no double-emit on idempotent re-apply).
        if was_newly_tombstoned {
            self.graph_index.write().unlink_memory(rid);
            self.cache_remove(rid);
            self.log_op(
                "forget",
                Some(rid),
                &serde_json::json!({
                    "rid": rid,
                    "updated_at_unix_micros": ts_micros,
                    "reason": reason,
                }),
                None,
            )?;
        }

        Ok(was_newly_tombstoned)
    }

    /// Tombstone a memory. Returns `true` if the memory was found in a live
    /// state and newly tombstoned; `false` if rid was missing or already
    /// tombstoned (both treated as no-ops).
    ///
    /// Stamped with engine-supplied `now()` — for byte-deterministic
    /// cluster-replicated tombstones use [`tombstone_with_rid`] instead.
    /// `forget()` delegates to `tombstone_inner` with no reason; the bool
    /// return is the only behavioral difference.
    #[tracing::instrument(skip(self))]
    pub fn forget(&self, rid: &str) -> Result<bool> {
        let ts_micros = (now() * 1_000_000.0) as i64;
        self.tombstone_inner(rid, None, ts_micros)
    }

    /// User-initiated memory correction.
    ///
    /// Creates a new corrected memory and tombstones the original.
    #[tracing::instrument(skip(self, new_embedding))]
    pub fn correct(
        &self,
        rid: &str,
        new_text: &str,
        new_importance: Option<f64>,
        new_valence: Option<f64>,
        new_embedding: &[f32],
        correction_note: Option<&str>,
    ) -> Result<CorrectionResult> {
        let original = self
            .get(rid)?
            .ok_or_else(|| YantrikDbError::NotFound(format!("memory: {}", rid)))?;

        let ts = now();
        let importance = new_importance.unwrap_or(original.importance);
        let valence = new_valence.unwrap_or(original.valence);
        let meta = serde_json::json!({
            "corrected_from": rid,
            "correction_note": correction_note,
            "original_text": original.text,
        });

        // Create the corrected memory (logs a "record" op)
        let new_rid = self.record(
            new_text,
            &original.memory_type,
            importance,
            valence,
            original.half_life,
            &meta,
            new_embedding,
            &original.namespace,
            original.certainty,
            &original.domain,
            &original.source,
            original.emotional_state.as_deref(),
        )?;

        // Tombstone the original (logs a "forget" op)
        self.forget(rid)?;

        // Transfer edges from original to corrected memory
        let edges = self.get_edges(rid)?;
        for edge in &edges {
            if edge.src == rid {
                self.relate(&new_rid, &edge.dst, &edge.rel_type, edge.weight)?;
            } else if edge.dst == rid {
                self.relate(&edge.src, &new_rid, &edge.rel_type, edge.weight)?;
            }
        }

        // Log a "correct" op that bundles the correction semantics
        let emb_hash = embedding_hash(new_embedding);
        self.log_op(
            "correct",
            Some(&new_rid),
            &serde_json::json!({
                "original_rid": rid,
                "new_rid": new_rid,
                "text": new_text,
                "type": original.memory_type,
                "importance": importance,
                "valence": valence,
                "half_life": original.half_life,
                "created_at": ts,
                "metadata": meta,
                "correction_note": correction_note,
            }),
            Some(&emb_hash),
        )?;

        // Auto-resolve any open conflicts involving the original rid
        let related_conflicts: Vec<String> = {
            let conn = self.conn();
            let mut stmt = conn.prepare(
                "SELECT conflict_id FROM conflicts
                 WHERE status = 'open' AND (memory_a = ?1 OR memory_b = ?1)",
            )?;
            let rows = stmt.query_map(params![rid], |row| row.get::<_, String>(0))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        }; // drop conn before resolve_conflict which re-acquires it

        for cid in related_conflicts {
            let _ = self.resolve_conflict(
                &cid,
                "keep_both",
                Some(&new_rid),
                None,
                Some(&format!("Auto-resolved: original memory corrected to '{}'", new_rid)),
            );
        }

        Ok(CorrectionResult {
            original_rid: rid.to_string(),
            corrected_rid: new_rid,
            original_tombstoned: true,
        })
    }
}
