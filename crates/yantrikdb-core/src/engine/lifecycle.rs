use rusqlite::params;

use crate::error::{Result, YantrikDbError};
use crate::scoring;
use crate::types::*;

use super::{embedding_hash, now, YantrikDB};

impl YantrikDB {
    /// Get a single memory by RID.
    ///
    /// v0.10 Item 2: a successful consumer `get` is an outcome anchor —
    /// an independent, caller-initiated action targeting the rid — and
    /// records a weak-positive ranking label bound to the rid's most
    /// recent impression (no-op if the ranker never served it). Engine-
    /// internal reads must use [`Self::get_untracked`] instead so the
    /// learner never labels its own traversals.
    #[tracing::instrument(skip(self))]
    pub fn get(&self, rid: &str) -> Result<Option<Memory>> {
        let found = self.get_untracked(rid)?;
        if found.is_some() {
            self.note_caller_used(rid);
        }
        Ok(found)
    }

    /// `get` without the caller_used label — for engine-internal reads
    /// (conflict resolution, link hydration, maintenance).
    pub(crate) fn get_untracked(&self, rid: &str) -> Result<Option<Memory>> {
        let conn = self.conn();
        let mut stmt = conn.prepare("SELECT * FROM memories WHERE rid = ?1")?;

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

    /// Return the head of a chain-shaped namespace — its most recent entry.
    ///
    /// Identity / narrative chains (e.g. a `claude_self_narrative` namespace
    /// whose entries are appended over time) need *exact* "latest entry"
    /// semantics. `recall` ranks by similarity × importance × decay, so it can
    /// never guarantee it returns the chain head; this does. Because rids are
    /// UUIDv7 (lexically chronological), the greatest active rid in the
    /// namespace is exactly the latest write — an O(log n) index seek on the
    /// primary key, not a retrieval lottery.
    ///
    /// Walk backwards from the head with
    /// `list_records(namespace, order="desc", since_rid=head.rid, ...)`.
    pub fn chain_head(&self, namespace: &str) -> Result<Option<Memory>> {
        let (mut records, _) =
            self.list_records(Some(namespace), None, None, None, None, None, 1, "desc")?;
        Ok(records.pop())
    }

    /// **v0.7.24 — structural query path.** Typed enumeration by indexed
    /// metadata fields with a stable keyset cursor — the relational counterpart
    /// to similarity `recall`. All filters are optional and AND-compose. The
    /// `kind` / `drive_id` predicates ride the v32 generated-column indexes
    /// (`idx_memories_kind` / `idx_memories_drive_id`), and pagination is a
    /// keyset walk over the UUIDv7 `rid` primary key (lexically = chronological),
    /// so it stays O(log n) and is stable under concurrent inserts — unlike the
    /// `LIMIT/OFFSET` path in `list_memories`.
    ///
    /// `order`: `"desc"` (newest-first) walks `rid < since_rid`; anything else
    /// (default `"asc"`, oldest-first) walks `rid > since_rid`. Returns the page
    /// plus `next_cursor` = the last rid of a *full* page (pass it back as the
    /// next `since_rid`), or `None` when the page was not filled (end reached).
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip(self))]
    pub fn list_records(
        &self,
        namespace: Option<&str>,
        kind: Option<&str>,
        drive_id: Option<&str>,
        memory_type: Option<&str>,
        domain: Option<&str>,
        since_rid: Option<&str>,
        limit: usize,
        order: &str,
    ) -> Result<(Vec<Memory>, Option<String>)> {
        let desc = order.eq_ignore_ascii_case("desc");
        let cursor_op = if desc { "<" } else { ">" };
        let order_sql = if desc { "DESC" } else { "ASC" };

        let mut conditions = vec!["consolidation_status = 'active'".to_string()];
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut idx = 1;

        // All optional, AND-composed. Order chosen so the most selective
        // indexed predicates (kind/drive_id) appear first.
        for (col, val) in [
            ("kind", kind),
            ("drive_id", drive_id),
            ("namespace", namespace),
            ("type", memory_type),
            ("domain", domain),
        ] {
            if let Some(v) = val {
                conditions.push(format!("{col} = ?{idx}"));
                param_values.push(Box::new(v.to_string()));
                idx += 1;
            }
        }
        if let Some(cursor) = since_rid {
            conditions.push(format!("rid {cursor_op} ?{idx}"));
            param_values.push(Box::new(cursor.to_string()));
            idx += 1;
        }

        let where_clause = conditions.join(" AND ");
        let sql = format!(
            "SELECT rid, type, text, created_at, importance, valence, half_life, \
             last_access, access_count, consolidation_status, storage_tier, \
             consolidated_into, metadata, namespace, certainty, domain, source, \
             emotional_state, session_id, due_at, temporal_kind \
             FROM memories WHERE {where_clause} ORDER BY rid {order_sql} LIMIT ?{idx}"
        );
        param_values.push(Box::new(limit as i64));

        let conn = self.conn();
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

        // next_cursor = last rid only when the page was filled (more may exist).
        let next_cursor = if memories.len() == limit {
            memories.last().map(|m| m.rid.clone())
        } else {
            None
        };
        Ok((memories, next_cursor))
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
    /// Sibling of `forget()` that takes caller-supplied namespace +
    /// timestamp + optional reason + optional seq for byte-deterministic
    /// follower replay. Used by yantrikdb-server's cluster-mode applier so
    /// replicated tombstones converge to identical engine state across
    /// leader + followers.
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
    /// - **Caller-supplied namespace**: required for the visible_seq bump
    ///   regardless of whether the SQL row exists locally — followers
    ///   apply log entries before the corresponding `record_with_rid` may
    ///   have arrived (snapshot lag), but the bump must still happen so
    ///   the cluster-wide visible_seq[ns] is monotonic with the openraft
    ///   commit-log index.
    /// - **Caller-supplied timestamp**: `requested_at_unix_micros` materialized
    ///   into `updated_at` (REAL seconds). No engine `now()` call on this path.
    /// - **Optional reason**: stored in `tombstone_reason TEXT` column (v25).
    ///   NULL when caller passes None.
    /// - **Caller-supplied `seq`** (cluster mode): when `Some(n)`, used
    ///   as the delta-tombstone seq + visible_seq bump value; engine
    ///   ratchets `vec_seq` to at least `n`. `None` lets the engine
    ///   allocate (single-node).
    ///
    /// Always emits a tombstone marker into the DeltaIndex regardless of
    /// whether the SQL row was newly tombstoned — followers may have the
    /// rid in their delta even if SQL is absent.
    pub fn tombstone_with_rid(
        &self,
        rid: &str,
        namespace: &str,
        reason: Option<&str>,
        requested_at_unix_micros: i64,
        seq: Option<u64>,
    ) -> Result<()> {
        self.tombstone_inner(rid, Some(namespace), reason, requested_at_unix_micros, seq)?;
        Ok(())
    }

    /// Internal helper shared by `tombstone_with_rid` and `forget`. Returns
    /// `true` iff the row was newly tombstoned (was active or consolidated
    /// before this call). Returns `false` if rid is missing or already
    /// tombstoned — both treated as idempotent successful no-ops.
    ///
    /// `namespace`:
    ///   - `Some(ns)`: cluster path — caller has the namespace from the
    ///     replication payload; we bump `visible_seq[ns]` even if the rid
    ///     is missing locally (snapshot-lag determinism).
    ///   - `None`: `forget()` path — we SELECT the namespace from the row.
    ///     If the row is missing, `visible_seq` is not bumped (no reader
    ///     would be waiting on a non-existent rid in single-node mode).
    fn tombstone_inner(
        &self,
        rid: &str,
        namespace: Option<&str>,
        reason: Option<&str>,
        ts_micros: i64,
        seq: Option<u64>,
    ) -> Result<bool> {
        let ts_secs = (ts_micros as f64) / 1_000_000.0;

        // Resolve namespace + execute the UPDATE in a single conn block.
        // forget() lookup case: SELECT before UPDATE (the UPDATE may zero
        // changes when the row is missing or already tombstoned, so we
        // can't rely on RETURNING — and namespace doesn't change on
        // tombstone, so a separate SELECT is correct and cheap).
        let (was_newly_tombstoned, ns_to_bump): (bool, Option<String>) = {
            let conn = self.conn();
            let resolved_ns: Option<String> = match namespace {
                Some(ns) => Some(ns.to_string()),
                None => conn
                    .query_row(
                        "SELECT namespace FROM memories WHERE rid = ?1",
                        params![rid],
                        |r| r.get::<_, String>(0),
                    )
                    .ok(),
            };
            let changes = conn.execute(
                "UPDATE memories SET consolidation_status = 'tombstoned', \
                 updated_at = ?1, tombstone_reason = ?2 \
                 WHERE rid = ?3 AND consolidation_status != 'tombstoned'",
                params![ts_secs, reason, rid],
            )?;
            (changes > 0, resolved_ns)
        };

        // Always emit a delta tombstone so search() filters it out even
        // before SQL has applied. Cluster followers may have the rid in
        // their delta from a recent record_with_rid that has not yet
        // compacted into cold; the tombstone marker covers that window.
        //
        // **Issue #41 brainstorm-4 §1.** Snapshot SearchState — the
        // tombstone lands on the active generation's DeltaIndex.
        let seq = self.assign_seq(seq);
        self.search_state.load().vec_index.tombstone(rid, seq);
        if let Some(ns) = &ns_to_bump {
            self.bump_visible_seq(ns, seq);
        }

        // Engine-internal index updates only when the row was newly tombstoned
        // (replay-safe: no double-emit on idempotent re-apply).
        if was_newly_tombstoned {
            self.graph_index.write().unlink_memory(rid);
            self.cache_remove(rid);

            // **Issue #48.** Mark record_links touching this rid as broken
            // rather than deleting them — the audit trail (this link
            // existed before the endpoint was forgotten) is retained, and
            // traversal/recall filter on status='active'. Outbound links
            // from the forgotten rid => broken_source_forgotten; inbound
            // links to it => broken_target_forgotten. No oplog op is
            // emitted: the existing 'forget' op replays this same status
            // transition on followers via the deterministic SQL below.
            {
                let conn = self.conn();
                conn.execute(
                    "UPDATE record_links SET status = 'broken_source_forgotten' \
                     WHERE source_rid = ?1 AND status = 'active'",
                    params![rid],
                )?;
                conn.execute(
                    "UPDATE record_links SET status = 'broken_target_forgotten' \
                     WHERE target_rid = ?1 AND status = 'active'",
                    params![rid],
                )?;
            }

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
    /// `forget()` delegates to `tombstone_inner` with namespace lookup
    /// (the namespace is read from the row); the bool return is the only
    /// behavioral difference vs the cluster primitive.
    #[tracing::instrument(skip(self))]
    pub fn forget(&self, rid: &str) -> Result<bool> {
        let ts_micros = (now() * 1_000_000.0) as i64;
        self.tombstone_inner(rid, None, None, ts_micros, None)
    }

    /// User-initiated memory correction (Issue #47, v0.7.20).
    ///
    /// **In-place mutation with audit trail.** Updates the memory at
    /// `rid` to reflect the supplied text / metadata-merge / importance
    /// / valence changes, while:
    /// - preserving `rid` (no new memory minted)
    /// - preserving `created_at` (the memory's timeline anchor)
    /// - appending a row to `record_revisions` capturing the prior state
    /// - leaving inbound link integrity intact (graph edges, replication
    ///   audit log entries, knowledge graph references all continue to
    ///   resolve because the rid is unchanged)
    /// - logging a "correct" op for replication
    ///
    /// **Embedding NOT supported.** HNSW does not support in-place update
    /// of an existing vector, and rebuilding the cold tier on every
    /// correction is too expensive. Callers needing to change the
    /// embedding still use `forget()` + `record()` (the v0.7.19-and-
    /// earlier behaviour). The new `correct()` is for text + metadata
    /// + importance + valence only.
    ///
    /// **`reason` is required and must be non-empty.** The audit trail
    /// is load-bearing: it is what gives `correct()` its semantic value
    /// over a bare UPDATE. Empty / whitespace-only reasons are rejected
    /// with `YantrikDbError::InvalidInput`.
    ///
    /// **At least one mutation field must be supplied.** Passing all
    /// `None` for `new_text` / `metadata_merge` / `new_importance` /
    /// `new_valence` is a no-op correction and is rejected with
    /// `YantrikDbError::InvalidInput`.
    ///
    /// **Atomic.** The revision insert + memories UPDATE happen in a
    /// single SQL transaction. Either both succeed or neither does.
    #[tracing::instrument(skip(self))]
    pub fn correct(
        &self,
        rid: &str,
        new_text: Option<&str>,
        metadata_merge: Option<&serde_json::Value>,
        new_importance: Option<f64>,
        new_valence: Option<f64>,
        reason: &str,
    ) -> Result<CorrectionResult> {
        // Validate reason non-empty (load-bearing audit field).
        let reason_trimmed = reason.trim();
        if reason_trimmed.is_empty() {
            return Err(YantrikDbError::InvalidInput(
                "correct: `reason` is required and must be non-empty; \
                 the audit trail is load-bearing"
                    .to_string(),
            ));
        }

        // Validate at least one mutation field is supplied.
        if new_text.is_none()
            && metadata_merge.is_none()
            && new_importance.is_none()
            && new_valence.is_none()
        {
            return Err(YantrikDbError::InvalidInput(
                "correct: at least one of `new_text` / `metadata_merge` / \
                 `new_importance` / `new_valence` must be supplied; \
                 a correction with no changes is a no-op"
                    .to_string(),
            ));
        }

        // Load without minting an outcome-anchor label — a correction that
        // hasn't happened yet is not evidence. The single note_caller_used
        // fires at the end of a SUCCESSFUL correction (both paths).
        let original = self
            .get_untracked(rid)?
            .ok_or_else(|| YantrikDbError::NotFound(format!("memory: {}", rid)))?;

        // **v0.10 Item 3: vector-coherent correction.** When the text
        // actually changes, the retrieval vector must change with it —
        // otherwise the durable "current truth" and the embedding disagree
        // forever ("Alice owns service A" corrected to "Bob owns service B"
        // keeps being retrieved for Alice/service-A queries). Route to the
        // staged re-embed protocol. A no-op text change (same bytes) and
        // metadata/scalar-only corrections skip re-embedding entirely.
        let text_changes = new_text.is_some_and(|t| t != original.text);
        if text_changes {
            return self.correct_with_reembed(
                rid,
                &original,
                new_text.expect("text_changes implies Some"),
                metadata_merge,
                new_importance,
                new_valence,
                reason_trimmed,
            );
        }

        let ts = now();
        let hlc_bytes = self.tick_hlc().to_bytes().to_vec();
        let revision_id = crate::id::new_id();

        // Compute the new field values. For metadata, merge the supplied
        // patch into the existing metadata (existing keys overwritten by
        // the patch; un-mentioned keys retained).
        let new_importance_val = new_importance.unwrap_or(original.importance);
        let new_valence_val = new_valence.unwrap_or(original.valence);
        let new_text_val: String = match new_text {
            Some(t) => t.to_string(),
            None => original.text.clone(),
        };
        let prior_metadata_str = serde_json::to_string(&original.metadata)?;
        let new_metadata_val: serde_json::Value = match metadata_merge {
            Some(patch) => {
                let mut merged = original.metadata.clone();
                if let (Some(obj), Some(patch_obj)) = (merged.as_object_mut(), patch.as_object()) {
                    for (k, v) in patch_obj {
                        obj.insert(k.clone(), v.clone());
                    }
                    merged
                } else {
                    // Either original metadata isn't an object or patch
                    // isn't an object — fall back to "patch replaces".
                    patch.clone()
                }
            }
            None => original.metadata.clone(),
        };
        let new_metadata_str = serde_json::to_string(&new_metadata_val)?;

        // Encryption pass-through. The `memories` table stores text +
        // metadata in encrypted form when an encryption provider is
        // attached; bypassing this leaves the table with mixed plaintext
        // + ciphertext rows that later decrypt() calls fail on.
        // `prior_text` and `prior_metadata` in `record_revisions` are
        // stored in the SAME representation (encrypted-or-plain) so
        // history() returns rows that the same decrypt path can handle.
        let stored_new_text = self.encrypt_text(&new_text_val)?;
        let stored_new_metadata = self.encrypt_text(&new_metadata_str)?;
        // The original row read above returned decrypted text + metadata
        // (db.get() decrypts on the way out); re-encrypt for storage in
        // the revision row so it sits next to memories.text in the same
        // representation.
        let stored_prior_text = self.encrypt_text(&original.text)?;
        let stored_prior_metadata = self.encrypt_text(&prior_metadata_str)?;

        // The revision_num is the next sequential number for this rid.
        // Use SQL MAX + 1 inside the same transaction we'll UPDATE in,
        // so concurrent correct() calls on the same rid serialise on
        // the connection lock and each gets a distinct revision_num.
        let conn = self.conn.lock();
        let tx = conn.unchecked_transaction()?;

        let next_revision_num: i64 = tx
            .query_row(
                "SELECT COALESCE(MAX(revision_num), 0) + 1 \
                 FROM record_revisions WHERE rid = ?1",
                params![rid],
                |row| row.get(0),
            )
            .unwrap_or(1);

        // Insert the revision row capturing the prior state.
        tx.execute(
            "INSERT INTO record_revisions \
             (revision_id, rid, revision_num, prior_text, prior_metadata, \
              prior_importance, prior_valence, reason, applied_at, hlc, origin_actor) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                revision_id,
                rid,
                next_revision_num,
                stored_prior_text,
                stored_prior_metadata,
                original.importance,
                original.valence,
                reason_trimmed,
                ts,
                hlc_bytes,
                self.actor_id,
            ],
        )?;

        // UPDATE the memory in place. rid + created_at + embedding are
        // not touched (metadata/scalar-only path — the text is unchanged,
        // so the vector stays coherent). last_access is bumped.
        tx.execute(
            "UPDATE memories \
             SET text = ?1, metadata = ?2, importance = ?3, valence = ?4, \
                 last_access = ?5 \
             WHERE rid = ?6",
            params![
                stored_new_text,
                stored_new_metadata,
                new_importance_val,
                new_valence_val,
                ts,
                rid,
            ],
        )?;

        // **v0.10 Item 3 (sol boundary #6):** the replication op is written
        // INSIDE this transaction. A kill between the memories mutation and
        // the oplog append can no longer lose the correction's replication
        // event (the old path called log_op post-commit in its own conn).
        // No embedding bytes here — this path leaves the vector untouched.
        let applied_generation = self.search_state.load().generation as i64;
        self.insert_correct_op_in_tx(
            &tx,
            rid,
            &serde_json::json!({
                "rid": rid,
                "revision_num": next_revision_num,
                "new_text": new_text,
                "metadata_merge": metadata_merge,
                "new_importance": new_importance,
                "new_valence": new_valence,
                "reason": reason_trimmed,
                "applied_at": ts,
            }),
            None,
            None,
            applied_generation,
        )?;

        tx.commit()?;
        drop(conn);

        // Refresh the scoring_cache so subsequent recall() calls see the
        // new metadata + importance + valence. The cache is keyed by rid;
        // we update in place.
        {
            let mut cache = self.scoring_cache.write();
            if let Some(row) = cache.get_mut(rid) {
                row.importance = new_importance_val;
                // Note: cache doesn't hold raw text; recall hydrates that
                // from SQLite. The importance update is what affects
                // ranking; text + metadata get re-hydrated on next read.
            }
        }

        // v0.10 Item 2 outcome anchor: a correction is an independent
        // caller action targeting the rid — the ranker surfaced a memory
        // the caller cared enough to fix. The label measures RETRIEVAL
        // utility, not content truth (a wrong-but-found memory was still
        // the right retrieval).
        self.note_caller_used(rid);

        Ok(CorrectionResult {
            original_rid: rid.to_string(),
            corrected_rid: rid.to_string(),
            original_tombstoned: false,
            revision_num: next_revision_num,
        })
    }

    /// **v0.10 Item 3 (sol boundary #6).** Insert the `correct` replication
    /// op INSIDE the caller's correction transaction. `log_op` runs in its
    /// own connection AFTER commit, so a kill in the gap loses the
    /// replication event; writing the op in the same transaction makes the
    /// mutation and its intent atomic. `embedding` carries the EXACT
    /// re-embedded bytes (encrypted like `memories.embedding`) for a text
    /// correction so a follower applies them verbatim rather than
    /// re-embedding — follower re-embedding diverges by model
    /// version/quantization. `None` for metadata/scalar corrections.
    #[allow(clippy::too_many_arguments)]
    fn insert_correct_op_in_tx(
        &self,
        tx: &rusqlite::Transaction<'_>,
        rid: &str,
        payload: &serde_json::Value,
        emb_hash: Option<&[u8]>,
        embedding: Option<&[u8]>,
        applied_generation: i64,
    ) -> Result<()> {
        let op_id = crate::id::new_id();
        let hlc_bytes = self.tick_hlc().to_bytes().to_vec();
        let payload_str = serde_json::to_string(payload)?;
        tx.execute(
            "INSERT INTO oplog \
             (op_id, op_type, timestamp, target_rid, payload, actor_id, hlc, \
              embedding_hash, origin_actor, applied, applied_generation, embedding) \
             VALUES (?1, 'correct', ?2, ?3, ?4, ?5, ?6, ?7, ?8, 1, ?9, ?10)",
            params![
                op_id,
                now(),
                rid,
                payload_str,
                self.actor_id,
                hlc_bytes,
                emb_hash,
                self.actor_id,
                applied_generation,
                embedding,
            ],
        )?;
        Ok(())
    }

    /// **v0.10 Item 3 — RID-stable, vector-coherent correction.**
    ///
    /// The staged protocol for a text-changing correction (design: sol's
    /// 8-boundary audit, docs/V0.10_PLAN.md Item 3). The rid, created_at,
    /// and revision chain are preserved; only the content and its retrieval
    /// vector change.
    ///
    /// Ordering and why:
    /// 1. **Embed OUTSIDE the write-router guard** (slow step), mirroring
    ///    `record_text`'s revalidation loop — reembed throughput stays
    ///    bounded by the index rebuild, not by in-flight corrections.
    /// 2. **Guard + revalidate**: if a `reembed()` swap landed between the
    ///    embed and the guard, the vector is in the wrong space — retry.
    ///    If the router is mid-cutover (Queueing), return the typed
    ///    `CorrectionDeferredDuringReembed` (retryable; no state touched).
    /// 3. **Backpressure BEFORE any visible mutation**: under the guard the
    ///    delta can only shrink (compactor) — never grow (other sync
    ///    writers and reembed are excluded) — so a capacity check here
    ///    makes the later tombstone+append infallible. If the delta is
    ///    full, return `Backpressure` having changed nothing.
    /// 4. **One SQL transaction**: revision row (with prior embedding
    ///    model+hash) + memories UPDATE (text, metadata, scalars,
    ///    embedding, embedding_generation) + the `correct` oplog op with
    ///    exact new bytes (boundary #6). Kill before commit → nothing
    ///    changed; kill after → SQL is fully consistent and the index
    ///    rebuilds from it on reopen.
    /// 5. **Delta tombstone+append as one sealed op** after commit, then
    ///    scoring-cache + visible_seq in the same critical section.
    #[allow(clippy::too_many_arguments)]
    fn correct_with_reembed(
        &self,
        rid: &str,
        original: &Memory,
        new_text: &str,
        metadata_merge: Option<&serde_json::Value>,
        new_importance: Option<f64>,
        new_valence: Option<f64>,
        reason_trimmed: &str,
    ) -> Result<CorrectionResult> {
        use crate::serde_helpers::{deserialize_f32, serialize_f32};

        // Prior embedding provenance for the revision audit row: the old
        // model name and a hash of the old vector, so history() can explain
        // WHY the retrieval vector changed and a replica can verify it.
        let (old_emb_stored, prior_embedding_model): (Option<Vec<u8>>, Option<String>) = {
            let conn = self.conn();
            conn.query_row(
                "SELECT embedding, embedding_model FROM memories WHERE rid = ?1",
                params![rid],
                |r| {
                    Ok((
                        r.get::<_, Option<Vec<u8>>>(0)?,
                        r.get::<_, Option<String>>(1)?,
                    ))
                },
            )?
        };
        let prior_embedding_hash: Option<Vec<u8>> = match &old_emb_stored {
            Some(blob) => {
                let plain = self.decrypt_embedding(blob)?;
                Some(embedding_hash(&deserialize_f32(&plain)).to_vec())
            }
            None => None,
        };

        // Resolve the new field values (same merge semantics as the
        // metadata/scalar path).
        let new_importance_val = new_importance.unwrap_or(original.importance);
        let new_valence_val = new_valence.unwrap_or(original.valence);
        let prior_metadata_str = serde_json::to_string(&original.metadata)?;
        let new_metadata_val: serde_json::Value = match metadata_merge {
            Some(patch) => {
                let mut merged = original.metadata.clone();
                if let (Some(obj), Some(patch_obj)) = (merged.as_object_mut(), patch.as_object()) {
                    for (k, v) in patch_obj {
                        obj.insert(k.clone(), v.clone());
                    }
                    merged
                } else {
                    patch.clone()
                }
            }
            None => original.metadata.clone(),
        };
        let new_metadata_str = serde_json::to_string(&new_metadata_val)?;

        // Storage representation (encrypted-or-plain), mirroring the
        // metadata path so history() + memories decrypt uniformly.
        let stored_new_text = self.encrypt_text(new_text)?;
        let stored_new_metadata = self.encrypt_text(&new_metadata_str)?;
        let stored_prior_text = self.encrypt_text(&original.text)?;
        let stored_prior_metadata = self.encrypt_text(&prior_metadata_str)?;

        // Revalidation loop — mirrors record_text (mod.rs). Bounded in
        // expectation: reembed advances the generation monotonically and
        // completes at most once per call.
        loop {
            // Step 1: snapshot for the embed.
            let state_for_embed = self.search_state.load_full();
            let gen_pre = state_for_embed.generation;
            let digest_pre = state_for_embed.runtime_embedder_digest.clone();
            let embedder = state_for_embed
                .embedder
                .as_ref()
                .ok_or(YantrikDbError::NoEmbedder)?
                .clone();
            let dim_pre = state_for_embed.dim();
            drop(state_for_embed);

            // Step 2: embed the new text OUTSIDE any guard (slow).
            let new_embedding = embedder
                .embed(new_text)
                .map_err(|e| YantrikDbError::Inference(e.to_string()))?;
            crate::validate::validate_embedding("correct", &new_embedding, dim_pre)?;

            // Step 3: enter the sync path. A reembed cutover in flight →
            // typed, retryable, nothing touched.
            let sync_guard = match self.write_router.try_enter_sync_writer() {
                Some(g) => g,
                None => {
                    return Err(YantrikDbError::CorrectionDeferredDuringReembed {
                        rid: rid.to_string(),
                    });
                }
            };

            // Step 4: re-snapshot under the guard. From here reembed cannot
            // complete its swap until the guard drops.
            let state_for_commit = self.search_state.load_full();

            // Step 5: revalidate — a swap between step 1 and step 4 means
            // the embedding is in the wrong vector space; retry.
            if state_for_commit.generation != gen_pre
                || state_for_commit.runtime_embedder_digest != digest_pre
            {
                drop(sync_guard);
                tracing::info!(
                    gen_pre,
                    gen_post = state_for_commit.generation,
                    "correct_with_reembed: SearchState advanced mid-embed, retrying",
                );
                continue;
            }
            let generation = state_for_commit.generation as i64;

            // Step 6: backpressure BEFORE any visible mutation. Under the
            // guard the delta only shrinks; a tombstone(cold) + append is at
            // most +2 entries, so this check makes the later mutation
            // infallible.
            let delta_len = state_for_commit.vec_index.delta_len();
            let delta_max = state_for_commit.vec_index.delta_max();
            if delta_len + 2 > delta_max {
                drop(sync_guard);
                return Err(YantrikDbError::Backpressure {
                    pending: delta_len as i64,
                    max: delta_max as i64,
                    retry_after_ms: 50,
                });
            }

            // Step 7: one transaction — revision + memories UPDATE + oplog
            // intent (exact bytes). runtime name recorded as the new
            // embedding's model for the audit trail.
            let ts = now();
            let hlc_bytes = self.tick_hlc().to_bytes().to_vec();
            let revision_id = crate::id::new_id();
            let stored_new_emb = self.encrypt_embedding(&serialize_f32(&new_embedding))?;
            let new_emb_hash = embedding_hash(&new_embedding).to_vec();

            let next_revision_num: i64 = {
                let conn = self.conn.lock();
                let tx = conn.unchecked_transaction()?;
                let n: i64 = tx
                    .query_row(
                        "SELECT COALESCE(MAX(revision_num), 0) + 1 \
                         FROM record_revisions WHERE rid = ?1",
                        params![rid],
                        |row| row.get(0),
                    )
                    .unwrap_or(1);
                tx.execute(
                    "INSERT INTO record_revisions \
                     (revision_id, rid, revision_num, prior_text, prior_metadata, \
                      prior_importance, prior_valence, reason, applied_at, hlc, \
                      origin_actor, prior_embedding_model, prior_embedding_hash) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                    params![
                        revision_id,
                        rid,
                        n,
                        stored_prior_text,
                        stored_prior_metadata,
                        original.importance,
                        original.valence,
                        reason_trimmed,
                        ts,
                        hlc_bytes,
                        self.actor_id,
                        prior_embedding_model,
                        prior_embedding_hash,
                    ],
                )?;
                // rid + created_at preserved; text, metadata, scalars, AND
                // the embedding (+ its generation stamp) change together.
                tx.execute(
                    "UPDATE memories \
                     SET text = ?1, metadata = ?2, importance = ?3, valence = ?4, \
                         embedding = ?5, embedding_generation = ?6, updated_at = ?7, \
                         last_access = ?7 \
                     WHERE rid = ?8",
                    params![
                        stored_new_text,
                        stored_new_metadata,
                        new_importance_val,
                        new_valence_val,
                        stored_new_emb,
                        generation,
                        ts,
                        rid,
                    ],
                )?;
                self.insert_correct_op_in_tx(
                    &tx,
                    rid,
                    &serde_json::json!({
                        "rid": rid,
                        "revision_num": n,
                        "new_text": new_text,
                        "metadata_merge": metadata_merge,
                        "new_importance": new_importance,
                        "new_valence": new_valence,
                        "reason": reason_trimmed,
                        "applied_at": ts,
                        "reembedded": true,
                    }),
                    Some(&new_emb_hash),
                    Some(&stored_new_emb),
                    generation,
                )?;
                tx.commit()?;
                n
            };

            // Step 8: kill boundary. A crash HERE leaves SQL fully
            // consistent (new text + new embedding + oplog intent all
            // committed) and only the in-memory delta stale — the index
            // rebuilds from SQL on reopen. The kill-proof test parks here.
            crate::testing::fail_point("correct.between_commit_and_delta");

            // Step 9: delta tombstone + append as one sealed operation
            // (capacity guaranteed by step 6). Old vector shadowed, new
            // vector visible.
            let seq_tomb = self
                .vec_seq
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            state_for_commit.vec_index.tombstone(rid, seq_tomb);
            let seq_new = self
                .vec_seq
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            // Infallible in practice (capacity checked under the guard); on
            // the theoretical dim-mismatch path this errors after commit,
            // but SQL is already correct and a rebuild recovers the index.
            state_for_commit
                .vec_index
                .append(rid.to_string(), new_embedding.clone(), seq_new)?;
            self.bump_visible_seq(&original.namespace, seq_new);

            // Step 10: scoring cache — importance is the ranking-relevant
            // field the cache holds; text/metadata re-hydrate from SQL.
            {
                let mut cache = self.scoring_cache.write();
                if let Some(row) = cache.get_mut(rid) {
                    row.importance = new_importance_val;
                    row.valence = new_valence_val;
                    row.last_access = ts;
                }
            }

            drop(sync_guard);

            // Outcome anchor (Item 2): a successful correction is an
            // independent caller action targeting the rid.
            self.note_caller_used(rid);

            return Ok(CorrectionResult {
                original_rid: rid.to_string(),
                corrected_rid: rid.to_string(),
                original_tombstoned: false,
                revision_num: next_revision_num,
            });
        }
    }

    /// Query the revision history for a single record (Issue #47).
    ///
    /// Returns revisions ordered by `revision_num` ascending (oldest
    /// first). Empty vec if the record has never been corrected.
    /// Prior text + metadata are decrypted before return (mirrors
    /// `db.get()`'s contract on the `memories` table).
    pub fn history(&self, rid: &str) -> Result<Vec<RecordRevision>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT revision_id, rid, revision_num, prior_text, prior_metadata, \
                    prior_importance, prior_valence, reason, applied_at, origin_actor \
             FROM record_revisions \
             WHERE rid = ?1 \
             ORDER BY revision_num ASC",
        )?;
        let rows = stmt.query_map(params![rid], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, f64>(5)?,
                row.get::<_, f64>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, f64>(8)?,
                row.get::<_, String>(9)?,
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            let (
                revision_id,
                rid,
                revision_num,
                stored_text,
                stored_metadata,
                prior_importance,
                prior_valence,
                reason,
                applied_at,
                origin_actor,
            ) = r?;
            let prior_text = self.decrypt_text(&stored_text)?;
            let prior_metadata_str = self.decrypt_text(&stored_metadata)?;
            let prior_metadata: serde_json::Value = serde_json::from_str(&prior_metadata_str)
                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
            out.push(RecordRevision {
                revision_id,
                rid,
                revision_num,
                prior_text,
                prior_metadata,
                prior_importance,
                prior_valence,
                reason,
                applied_at,
                origin_actor,
            });
        }
        Ok(out)
    }
}
