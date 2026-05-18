use rusqlite::params;

use crate::error::Result;
use crate::types::Stats;

use super::{now, YantrikDB};

impl YantrikDB {
    /// Get engine statistics. Optionally filter memory counts by namespace.
    pub fn stats(&self, namespace: Option<&str>) -> Result<Stats> {
        let conn = self.conn.lock();
        let ns_filter = namespace
            .map(|ns| format!(" AND namespace = '{}'", ns.replace('\'', "''")))
            .unwrap_or_default();
        let active = conn.query_row(
            &format!(
                "SELECT COUNT(*) FROM memories WHERE consolidation_status = 'active'{}",
                ns_filter
            ),
            [],
            |row| row.get(0),
        )?;
        let consolidated = conn.query_row(
            &format!(
                "SELECT COUNT(*) FROM memories WHERE consolidation_status = 'consolidated'{}",
                ns_filter
            ),
            [],
            |row| row.get(0),
        )?;
        let tombstoned = conn.query_row(
            &format!(
                "SELECT COUNT(*) FROM memories WHERE consolidation_status = 'tombstoned'{}",
                ns_filter
            ),
            [],
            |row| row.get(0),
        )?;
        let archived = conn.query_row(
            &format!(
                "SELECT COUNT(*) FROM memories WHERE storage_tier = 'cold'{}",
                ns_filter
            ),
            [],
            |row| row.get(0),
        )?;
        let edges = conn.query_row(
            "SELECT COUNT(*) FROM edges WHERE tombstoned = 0",
            [],
            |row| row.get(0),
        )?;
        let entities = conn.query_row("SELECT COUNT(*) FROM entities", [], |row| row.get(0))?;
        let operations = conn.query_row("SELECT COUNT(*) FROM oplog", [], |row| row.get(0))?;
        let open_conflicts = conn.query_row(
            "SELECT COUNT(*) FROM conflicts WHERE status = 'open'",
            [],
            |row| row.get(0),
        )?;
        let resolved_conflicts = conn.query_row(
            "SELECT COUNT(*) FROM conflicts WHERE status IN ('resolved', 'dismissed')",
            [],
            |row| row.get(0),
        )?;
        let pending_triggers = conn.query_row(
            "SELECT COUNT(*) FROM trigger_log WHERE status = 'pending'",
            [],
            |row| row.get(0),
        )?;
        let active_patterns = conn.query_row(
            "SELECT COUNT(*) FROM patterns WHERE status = 'active'",
            [],
            |row| row.get(0),
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
            vec_index_entries: self.search_state.load().vec_index.len(),
            graph_index_entities: self.graph_index.read().entity_count(),
            graph_index_edges: self.graph_index.read().edge_count(),
        })
    }

    /// Append an operation to the oplog with HLC and optional embedding hash.
    ///
    /// **Issue #41 brainstorm-2 §1 / brainstorm-4 §6.** Stamps the
    /// v27 `applied_generation` column with the active SearchState
    /// generation. The post-swap materializer (Layer 5) uses this
    /// column to discriminate ops already applied under generation G
    /// (skip them — they're durably indexed) from queued-during-
    /// reembed ops (`applied_generation IS NULL` — need re-encode
    /// under the new embedder). Sync writers call this AFTER their
    /// `vec_index.append` so the generation read here is the same
    /// generation the index entry was written against (the
    /// `SyncWriteGuard` held by the caller prevents reembed from
    /// completing its swap while we read).
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
        let applied_generation: i64 = self.search_state.load().generation as i64;

        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO oplog (op_id, op_type, timestamp, target_rid, payload, \
             actor_id, hlc, embedding_hash, origin_actor, applied, applied_generation) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 1, ?10)",
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
                applied_generation,
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
        use std::sync::atomic::Ordering;
        let op_id = crate::id::new_id();
        let hlc_ts = self.tick_hlc();
        let hlc_bytes = hlc_ts.to_bytes().to_vec();
        let payload_str = serde_json::to_string(payload)?;

        // **v0.7.1 perf hotfix.** Backpressure check is now an atomic
        // load against `pending_op_count` instead of `SELECT COUNT(*) FROM
        // oplog WHERE applied = 0`. The previous SQL pattern dominated
        // foreground latency when v0.7.0 wired log_op_pending into the
        // record() hot path — every write paid a Mutex<Connection> acquire
        // + index scan + drop just to check the bound. Cached counter
        // is maintained by `log_op_pending` (fetch_add on insert) and
        // `mark_op_applied` (fetch_sub on apply-win).
        const MAX_PENDING_OPS: i64 = 10_000;
        let pending_now = self.pending_op_count.load(Ordering::Relaxed);
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
        // **v0.7.1**: maintain the cached counter. INSERT OR IGNORE on
        // op_id PK means the row may not actually have been added (caller
        // re-using an op_id is the cluster-replay shape). Check the
        // changes count to only increment on a real insert.
        if conn.changes() > 0 {
            self.pending_op_count.fetch_add(1, Ordering::Relaxed);
        }
        Ok(op_id)
    }

    /// **Decoupled write path RFC, Phase 1.**
    ///
    /// Count of pending oplog entries (applied=0). Used by tests and by the
    /// background materializer to decide whether to wake up.
    ///
    /// **v0.7.1 hotfix:** returns the cached `pending_op_count` atomic
    /// instead of running `SELECT COUNT(*)`. The atomic is maintained by
    /// `log_op_pending` (`fetch_add` on insert) and `mark_op_applied`
    /// (`fetch_sub` on apply-win) so it's always coherent with the SQL
    /// state. Tests that mutate oplog by hand (rare, only in test
    /// helpers) can still use the SQL form via `count_pending_ops_sql`.
    pub fn count_pending_ops(&self) -> Result<i64> {
        use std::sync::atomic::Ordering;
        Ok(self.pending_op_count.load(Ordering::Relaxed))
    }

    /// SQL-backed count for tests / debug. Same shape as v0.7.0's
    /// `count_pending_ops` but routed through `read_conn`. Kept as a
    /// reconciliation oracle for the cached counter; in production paths
    /// use `count_pending_ops`.
    #[doc(hidden)]
    pub fn count_pending_ops_sql(&self) -> Result<i64> {
        let conn = self.read_conn();
        let count: i64 =
            conn.query_row("SELECT COUNT(*) FROM oplog WHERE applied = 0", [], |row| {
                row.get(0)
            })?;
        Ok(count)
    }

    /// **Decoupled write path RFC, Phase 1.**
    ///
    /// Mark a pending oplog entry as materialized. Called by the background
    /// worker after it has applied the op to the in-memory indexes.
    ///
    /// **Returns** `Ok(true)` iff this caller transitioned the row from
    /// `applied=0` to `applied=1`. `Ok(false)` means another worker
    /// already applied it (race on shared oplog, normal under N workers).
    /// The race-safety filter `WHERE applied = 0` is what makes
    /// `apply_pending_ops_once` exactly-once across N concurrent workers
    /// — the work is idempotent so double-execution is safe; this filter
    /// just decides which worker gets to claim the apply count.
    pub fn mark_op_applied(&self, op_id: &str) -> Result<bool> {
        use std::sync::atomic::Ordering;
        let conn = self.conn.lock();
        let changed = conn.execute(
            "UPDATE oplog SET applied = 1 WHERE op_id = ?1 AND applied = 0",
            params![op_id],
        )?;
        let won = changed > 0;
        // **v0.7.1**: decrement the cached counter only when this caller
        // actually transitioned the row. Mirrors log_op_pending's
        // increment-only-on-real-insert pattern; keeps the atomic
        // coherent with SQL applied-state under N concurrent workers.
        if won {
            self.pending_op_count.fetch_sub(1, Ordering::Relaxed);
        }
        Ok(won)
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
        // **Issue #41 Layer 5.** Pull `embedding_model` alongside the
        // standard tuple so we can dispatch queued-during-reembed
        // writes through the re-encode path. embedding_model IS NOT
        // NULL is the v27 signature for record_queued ops (sync
        // record() leaves it NULL).
        let pending: Vec<(String, String, String, Option<String>)> = {
            let conn = self.read_conn();
            let mut stmt = conn.prepare(
                "SELECT op_id, op_type, payload, embedding_model FROM oplog \
                 WHERE applied = 0 \
                 ORDER BY hlc, op_id \
                 LIMIT ?1",
            )?;
            let rows = stmt.query_map(params![limit as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                ))
            })?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };

        let mut applied = 0usize;
        for (op_id, op_type, payload, embedding_model) in &pending {
            match op_type.as_str() {
                // **Phase 4.3 — saga task 3.** This is the only op_type
                // whose dispatch is *real* materialization work (the
                // unbounded entity/relation loops that used to be on the
                // foreground request path). Foreground enqueues; worker
                // applies. See docs/phase_4_3_design.md for the contract.
                crate::engine::op_types::OP_MATERIALIZE_RECORD_POST => {
                    match self.apply_materialize_record_post(payload) {
                        Ok(()) => {
                            // Only count this apply if THIS worker won
                            // the race to flip applied=0 -> applied=1.
                            // Other workers may have done duplicate work
                            // (idempotent), but exactly one gets the count.
                            if self.mark_op_applied(op_id)? {
                                applied += 1;
                            }
                        }
                        Err(e) => {
                            // Don't mark applied — leave pending for retry
                            // on next tick. Workers race-safe via the
                            // applied=0 filter, so a transient failure
                            // doesn't lose the op.
                            tracing::warn!(
                                target: "yantrikdb::ingest::materialize",
                                op_id = %op_id,
                                op_type = %op_type,
                                error = %e,
                                "post-record materialization failed; leaving pending for retry"
                            );
                        }
                    }
                }
                // **Phase 4.3 Commit C — saga task 19.** Cluster-mode
                // sibling of OP_MATERIALIZE_RECORD_POST. Same race-safety
                // semantics; the difference is the dispatch logic uses
                // caller-supplied entity list with no extraction.
                crate::engine::op_types::OP_MATERIALIZE_RECORD_WITH_RID_POST => {
                    match self.apply_materialize_record_with_rid_post(payload) {
                        Ok(()) => {
                            if self.mark_op_applied(op_id)? {
                                applied += 1;
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                target: "yantrikdb::ingest::materialize",
                                op_id = %op_id,
                                op_type = %op_type,
                                error = %e,
                                "post-record-with-rid materialization failed; leaving pending for retry"
                            );
                        }
                    }
                }
                // **Issue #41 Layer 5 — Queue-mode record drain.** The
                // signature embedding_model IS NOT NULL identifies an
                // op that was queued by `record_queued` during a
                // reembed cutover. It carries TEXT (not an embedding).
                // After the swap completes, the materializer drains
                // these ops by re-encoding the text under the active
                // generation's embedder + applying directly into the
                // memories table + new vec_index at the new gen.
                //
                // While reembed is still in flight (meta.reembed_state
                // set), defer: leave applied=0 so the next tick after
                // completion picks it up. This matches brainstorm-2 §5
                // "materializer fully PAUSED during reembed" (we pause
                // only this op class — other op types still drain).
                "record" if embedding_model.is_some() => {
                    if self.reembed_status().is_some() {
                        tracing::trace!(
                            target: "yantrikdb::ingest::materialize",
                            op_id = %op_id,
                            "Layer 5: reembed in flight; deferring queued record"
                        );
                        // No mark_op_applied — leave for next tick.
                        continue;
                    }
                    match self.apply_queued_reembed_record(payload) {
                        Ok(()) => {
                            if self.mark_op_applied(op_id)? {
                                applied += 1;
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                target: "yantrikdb::ingest::materialize",
                                op_id = %op_id,
                                op_type = %op_type,
                                error = %e,
                                "Layer 5 queued reembed record apply failed; leaving pending for retry"
                            );
                        }
                    }
                }
                "record" | "forget" | "relate" | "correct" | "consolidate" => {
                    tracing::trace!(
                        target: "yantrikdb::ingest::materialize",
                        op_id = %op_id,
                        op_type = %op_type,
                        "phase 3 stub: marking pending op as applied without inline materialization"
                    );
                    if self.mark_op_applied(op_id)? {
                        applied += 1;
                    }
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

    /// **Phase 4.3 — apply a queued `materialize_record_post` op.**
    ///
    /// Mirrors the post-INSERT entity/relation extraction loop that used to
    /// live on the foreground `record()` path. Now runs on the materializer
    /// thread so the foreground caller is not blocked on the unbounded
    /// loop count (5-15 entities + 0-3 relations per typical record).
    ///
    /// Idempotent: every SQL operation here is `INSERT OR IGNORE` on a
    /// natural key (entity name, memory_entities pair, edge tuple). A
    /// double-apply across worker restarts produces identical state.
    ///
    /// Payload shape (see `docs/phase_4_3_design.md`):
    ///
    /// ```json
    /// {
    ///   "rid":       "01HX...",
    ///   "text":      "<plaintext OR engine-encrypted>",
    ///   "namespace": "default",
    ///   "ts_secs":   1715184000.0,
    ///   "domain":    "general",
    ///   "source":    "user"
    /// }
    /// ```
    /// **Issue #41 Layer 5 — apply a queued-during-reembed record.**
    ///
    /// `record_queued` writes oplog rows with applied=0,
    /// embedding_model=<old_name>, and a payload carrying the
    /// memory's TEXT (not pre-encoded embedding). After reembed
    /// completes, this materializer drain re-encodes the text under
    /// the ACTIVE SearchState's embedder (the new one) and applies
    /// the row to memories + vec_index at the new generation.
    ///
    /// Idempotent: INSERT OR IGNORE on rid + the
    /// search_state-snapshot-once pattern guarantees a re-apply
    /// (from worker race or restart) produces identical state.
    ///
    /// Caller (apply_pending_ops_once) has already verified the op
    /// type is "record" AND embedding_model IS NOT NULL AND no
    /// reembed is in flight.
    fn apply_queued_reembed_record(&self, payload_json: &str) -> Result<()> {
        use crate::serde_helpers::serialize_f32;

        let payload: serde_json::Value = serde_json::from_str(payload_json).map_err(|e| {
            crate::error::YantrikDbError::InvalidInput(format!(
                "Layer 5 record drain: payload parse failed: {e}"
            ))
        })?;

        let rid = payload.get("rid").and_then(|v| v.as_str()).ok_or_else(|| {
            crate::error::YantrikDbError::InvalidInput("Layer 5 record drain: missing rid".into())
        })?;
        let memory_type = payload
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("episodic");
        let text = payload
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                crate::error::YantrikDbError::InvalidInput(
                    "Layer 5 record drain: missing text".into(),
                )
            })?;
        let importance = payload
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);
        let valence = payload
            .get("valence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let half_life = payload
            .get("half_life")
            .and_then(|v| v.as_f64())
            .unwrap_or(604800.0);
        let metadata = payload
            .get("metadata")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        let ts = payload
            .get("created_at")
            .and_then(|v| v.as_f64())
            .unwrap_or_else(super::now);
        let namespace = payload
            .get("namespace")
            .and_then(|v| v.as_str())
            .unwrap_or("default");
        let certainty = payload
            .get("certainty")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.8);
        let domain = payload
            .get("domain")
            .and_then(|v| v.as_str())
            .unwrap_or("general");
        let source = payload
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("user");
        let emotional_state = payload
            .get("emotional_state")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Snapshot the active SearchState. The embedder + generation
        // here are the post-swap ones (caller verified no reembed in
        // flight; the in-memory state is durable).
        let state = self.search_state.load_full();
        let embedder = state
            .embedder
            .as_ref()
            .ok_or_else(|| {
                crate::error::YantrikDbError::Inference(
                    "Layer 5 record drain: active SearchState has no embedder (cannot re-encode)"
                        .into(),
                )
            })?
            .clone();

        let new_emb = embedder.embed(text).map_err(|e| {
            crate::error::YantrikDbError::Inference(format!(
                "Layer 5 record drain: embedder failed on rid {rid:?}: {e}"
            ))
        })?;
        if new_emb.len() != state.dim() {
            return Err(crate::error::YantrikDbError::Inference(format!(
                "Layer 5 record drain: embedder returned len {} but SearchState dim {}",
                new_emb.len(),
                state.dim(),
            )));
        }
        let emb_blob = serialize_f32(&new_emb);
        let stored_emb = self.encrypt_embedding(&emb_blob)?;

        // The payload's text/metadata are already in engine-stored
        // form (record_queued passed them through). Don't double-
        // encrypt; just hand back as stored.
        let stored_text = text.to_string();
        let stored_meta = serde_json::to_string(&metadata)?;

        let embedding_generation: i64 = state.generation as i64;

        // INSERT OR IGNORE on rid for idempotency. If a prior worker
        // already inserted this row (race + retry), the OR IGNORE
        // makes this a no-op AND mark_op_applied lets one worker win
        // the count.
        {
            let conn = self.conn();
            conn.execute(
                "INSERT OR IGNORE INTO memories \
                 (rid, type, text, embedding, created_at, updated_at, importance, \
                  half_life, last_access, valence, metadata, namespace, \
                  certainty, domain, source, emotional_state, embedding_generation) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
                params![
                    rid,
                    memory_type,
                    stored_text,
                    stored_emb,
                    ts,
                    ts,
                    importance,
                    half_life,
                    ts,
                    valence,
                    stored_meta,
                    namespace,
                    certainty,
                    domain,
                    source,
                    emotional_state,
                    embedding_generation,
                ],
            )?;
        }

        // Append into the active vec_index. Idempotent: DeltaIndex
        // de-dupes on rid+seq.
        let seq = self
            .vec_seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        state.vec_index.append(rid.to_string(), new_emb, seq)?;

        // Bump visible_seq for RYW. Layer 6 will refine the
        // generation-aware semantics; for now bump under the
        // current generation's covers_through_seq logic.
        self.bump_visible_seq(namespace, seq);

        Ok(())
    }

    fn apply_materialize_record_post(&self, payload_json: &str) -> Result<()> {
        let payload: serde_json::Value = serde_json::from_str(payload_json).map_err(|e| {
            crate::error::YantrikDbError::InvalidInput(format!(
                "materialize_record_post: payload parse failed: {e}"
            ))
        })?;

        let rid = payload.get("rid").and_then(|v| v.as_str()).ok_or_else(|| {
            crate::error::YantrikDbError::InvalidInput(
                "materialize_record_post: missing rid".into(),
            )
        })?;
        let text_stored = payload
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                crate::error::YantrikDbError::InvalidInput(
                    "materialize_record_post: missing text".into(),
                )
            })?;
        let namespace = payload
            .get("namespace")
            .and_then(|v| v.as_str())
            .unwrap_or("default");
        let ts_secs = payload
            .get("ts_secs")
            .and_then(|v| v.as_f64())
            .unwrap_or_else(super::now);
        let domain = payload
            .get("domain")
            .and_then(|v| v.as_str())
            .unwrap_or("general");
        let source = payload
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("user");

        // Decrypt the text field if engine encrypted at rest. The payload
        // stored the engine-encrypted form, so the worker decrypts before
        // running the heuristic extractor on plaintext.
        let text_owned: String = self.decrypt_text(text_stored)?;
        let text = text_owned.as_str();

        let text_tokens = crate::graph::tokenize(text);
        let heuristic_entities = crate::graph::extract_heuristic_entities(text);

        // Loop A: seed heuristic entities (idempotent INSERT ... ON CONFLICT).
        if !heuristic_entities.is_empty() {
            let conn = self.conn();
            for entity in &heuristic_entities {
                let entity_type = crate::graph::classify_entity_type(entity);
                conn.execute(
                    "INSERT INTO entities (name, entity_type, first_seen, last_seen, mention_count) \
                     VALUES (?1, ?2, ?3, ?3, 1) \
                     ON CONFLICT(name) DO UPDATE SET \
                        last_seen = ?3, \
                        mention_count = mention_count + 1, \
                        entity_type = CASE \
                            WHEN entity_type = 'unknown' AND ?2 != 'unknown' THEN ?2 \
                            ELSE entity_type END",
                    params![entity, entity_type, ts_secs],
                )?;
            }
        }

        // Compose candidate set: heuristic + already-known entities.
        let mut candidates: std::collections::HashSet<String> =
            heuristic_entities.iter().cloned().collect();
        for known in self.graph_index.read().all_entity_names() {
            if crate::graph::entity_matches_text(&known, &text_tokens) {
                candidates.insert(known);
            }
        }

        if !candidates.is_empty() {
            // Loop B: memory_entities INSERT OR IGNORE.
            {
                let conn = self.conn();
                for entity in &candidates {
                    conn.execute(
                        "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name) VALUES (?1, ?2)",
                        params![rid, entity],
                    )?;
                }
            }
            // graph_index in-memory update (idempotent — add_entity/link dedupe).
            let mut gi = self.graph_index.write();
            for entity in &candidates {
                let entity_type = crate::graph::classify_entity_type(entity);
                gi.add_entity(entity, entity_type);
                gi.link_memory(rid, entity);
            }
        }

        // Loop C+D: relation extraction + claim ingestion.
        let heuristic_vec: Vec<String> = heuristic_entities.iter().cloned().collect();
        let relations = crate::graph::extract_heuristic_relations(text, &heuristic_vec);
        for rel in &relations {
            let already_exists = {
                let conn = self.conn();
                conn.query_row(
                    "SELECT COUNT(*) FROM edges WHERE src = ?1 AND rel_type = ?2 AND dst = ?3 \
                     AND namespace = ?4 AND extractor = 'heuristic_v1' AND tombstoned = 0",
                    params![rel.src, rel.rel_type, rel.dst, namespace],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap_or(0)
                    > 0
            };
            if already_exists {
                continue;
            }
            let _ = self.ingest_claim(
                &rel.src,
                &rel.rel_type,
                &rel.dst,
                namespace,
                rel.polarity,
                &rel.modality,
                None,
                None,
                "heuristic_v1",
                Some("1.0"),
                &rel.confidence_band,
                Some(rid),
                None,
                None,
                1.0,
            );
        }

        // Audit telemetry — same shape as the foreground path emitted before.
        let features = crate::graph::analyze_text_features(text, &heuristic_vec);
        tracing::info!(
            target: "yantrikdb::audit::extraction",
            namespace = %namespace,
            memory_rid = %rid,
            domain = %domain,
            source = %source,
            extractor_version = "heuristic_v1",
            char_length = features.char_length,
            sentence_count = features.sentence_count,
            entity_count = features.entity_count,
            entities_matched_in_graph = candidates.len().saturating_sub(heuristic_entities.len()),
            negation_cue_count = features.negation_cue_count,
            temporal_cue_count = features.temporal_cue_count,
            modality_cue_count = features.modality_cue_count,
            has_compound_markers = features.has_compound_markers,
            likely_assertion = features.likely_assertion,
            "extraction audit (materialized post-record)"
        );

        Ok(())
    }

    /// **Phase 4.3 Commit C — apply a queued `materialize_record_with_rid_post` op.**
    ///
    /// Cluster-mode sibling of [`Self::apply_materialize_record_post`]. Runs the
    /// post-INSERT entity / memory_entities / graph_index updates that used
    /// to live inside the foreground SAVEPOINT block of `record_with_rid()`.
    ///
    /// **Why it differs from the heuristic path.** `record_with_rid` is the
    /// cluster determinism primitive — the leader's apply emits a payload
    /// containing the explicit `extracted_entities` slice; followers must
    /// converge to byte-identical SQL state by replaying that same slice.
    /// Running heuristic extraction on the materializer would risk
    /// divergence (extractor versions or text edge cases differing across
    /// nodes). So this dispatch arm uses the caller's entity list verbatim,
    /// no `extract_heuristic_entities` / `extract_heuristic_relations`
    /// calls.
    ///
    /// `was_new_row` payload field controls whether `entities.mention_count`
    /// gets bumped on insert. Mirrors the original SQL conditional:
    ///
    ///   `mention_count = CASE WHEN ?was_new THEN mention_count + 1 ELSE mention_count END`
    ///
    /// This preserves the contract that replayed-but-not-newly-inserted
    /// memories don't double-count entity mentions.
    ///
    /// Idempotent on every loop step (INSERT OR IGNORE, ON CONFLICT DO UPDATE
    /// with mention_count guarded by `was_new_row`).
    fn apply_materialize_record_with_rid_post(&self, payload_json: &str) -> Result<()> {
        let payload: serde_json::Value = serde_json::from_str(payload_json).map_err(|e| {
            crate::error::YantrikDbError::InvalidInput(format!(
                "materialize_record_with_rid_post: payload parse failed: {e}"
            ))
        })?;

        let rid = payload.get("rid").and_then(|v| v.as_str()).ok_or_else(|| {
            crate::error::YantrikDbError::InvalidInput(
                "materialize_record_with_rid_post: missing rid".into(),
            )
        })?;
        let _namespace = payload
            .get("namespace")
            .and_then(|v| v.as_str())
            .unwrap_or("default");
        let ts_secs = payload
            .get("ts_secs")
            .and_then(|v| v.as_f64())
            .unwrap_or_else(super::now);
        let was_new_row = payload
            .get("was_new_row")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let entities: Vec<String> = payload
            .get("extracted_entities")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        if entities.is_empty() {
            // Nothing to materialize — early return is the cheap idempotent path.
            return Ok(());
        }

        // Loop A: entities INSERT (mirrors the inline savepoint block exactly).
        {
            let conn = self.conn();
            for entity in &entities {
                let entity_type = crate::graph::classify_entity_type(entity);
                conn.execute(
                    "INSERT INTO entities (name, entity_type, first_seen, last_seen, mention_count) \
                     VALUES (?1, ?2, ?3, ?3, 1) \
                     ON CONFLICT(name) DO UPDATE SET \
                        last_seen = ?3, \
                        mention_count = CASE WHEN ?4 THEN mention_count + 1 ELSE mention_count END, \
                        entity_type = CASE \
                            WHEN entity_type = 'unknown' AND ?2 != 'unknown' THEN ?2 \
                            ELSE entity_type END",
                    params![entity, entity_type, ts_secs, was_new_row],
                )?;
                conn.execute(
                    "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name) VALUES (?1, ?2)",
                    params![rid, entity],
                )?;
            }
        }

        // graph_index in-memory update (idempotent — add_entity/link_memory dedupe).
        {
            let mut gi = self.graph_index.write();
            for entity in &entities {
                let entity_type = crate::graph::classify_entity_type(entity);
                gi.add_entity(entity, entity_type);
                gi.link_memory(rid, entity);
            }
        }

        Ok(())
    }

    /// **Phase 6 RYW** — allocate or accept a seq for a write primitive.
    ///
    /// Single-node mode: callers pass `None` and the engine allocates a
    /// fresh seq via `vec_seq.fetch_add` (1-indexed via `+ 1`).
    ///
    /// Cluster mode (RFC 010, design lock 2026-05-07): the applier passes
    /// `Some(commit_log_index)` so the seq IS the openraft commit-log
    /// index — leader and followers thereby agree on a single global
    /// monotonic stream. The engine ratchets `vec_seq` up to at least the
    /// supplied value (via `fetch_max`) so any future single-node writes
    /// against the same engine never produce seqs that collide with the
    /// cluster-supplied stream.
    ///
    /// Returns the seq the caller should use to tag the delta entry, the
    /// oplog row, and the visible_seq bump.
    pub(crate) fn assign_seq(&self, requested: Option<u64>) -> u64 {
        use std::sync::atomic::Ordering;
        match requested {
            Some(n) => {
                self.vec_seq.fetch_max(n, Ordering::Relaxed);
                n
            }
            None => self.vec_seq.fetch_add(1, Ordering::Relaxed) + 1,
        }
    }

    /// **Phase 6 RYW** — bump the visible_seq high-water mark for a
    /// namespace. Called by record/record_with_rid and siblings after the
    /// write has been materialized into the in-memory delta. Idempotent:
    /// only advances the watermark via `fetch_max`; same-or-lower seqs
    /// are no-ops.
    ///
    /// Wakes any threads in ``recall_with_seq`` waiting on this namespace
    /// via the paired condvar.
    pub(crate) fn bump_visible_seq(&self, namespace: &str, seq: u64) {
        use std::sync::atomic::Ordering;
        // Fast path: namespace already present — single fetch_max, no
        // hashmap mutation.
        if let Some(entry) = self.visible_seq.get(namespace) {
            entry.fetch_max(seq, Ordering::Release);
        } else {
            // First write for this namespace: insert. The DashMap entry
            // API gives us insert-or-existing semantics atomically per
            // shard. If two threads race to insert the same namespace
            // for the first time, one wins and the other's fetch_max
            // converges anyway.
            self.visible_seq
                .entry(namespace.to_string())
                .or_insert_with(|| std::sync::atomic::AtomicU64::new(0))
                .fetch_max(seq, Ordering::Release);
        }
        self.visible_seq_cv.notify_all();
    }

    /// **Phase 6 RYW** — current visible_seq high-water mark for a namespace.
    /// Returns 0 for namespaces that have never been bumped.
    ///
    /// Lock-free in steady state: a DashMap shard read + an atomic load.
    pub fn visible_seq_for(&self, namespace: &str) -> u64 {
        use std::sync::atomic::Ordering;
        self.visible_seq
            .get(namespace)
            .map(|e| e.load(Ordering::Acquire))
            .unwrap_or(0)
    }

    /// **Phase 6 RYW** — wait until visible_seq[namespace] >= min_seq or
    /// the timeout expires. Returns ``Ok(())`` on watermark reached;
    /// ``Err(Error::RyWaitTimeout)`` on timeout.
    ///
    /// Callers requesting strict read-your-writes pass a seq from a prior
    /// write to gate a subsequent recall. Default ``recall()`` does not
    /// call this — the delta is always visible by virtue of being scanned
    /// during search; this primitive is only needed when the caller wants
    /// to wait through a compaction-in-progress window or a cluster
    /// follower-apply-lag window.
    pub fn wait_for_visible_seq(
        &self,
        namespace: &str,
        min_seq: u64,
        timeout: std::time::Duration,
    ) -> Result<()> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let current = self.visible_seq_for(namespace);
            if current >= min_seq {
                return Ok(());
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                return Err(crate::error::YantrikDbError::RyWaitTimeout {
                    namespace: namespace.to_string(),
                    requested_seq: min_seq,
                    observed_seq: current,
                    waited_ms: timeout.as_millis() as u64,
                });
            }
            let remaining = deadline - now;
            // The sentinel mutex is a no-data lock pair for the Condvar.
            // Critical race-avoidance pattern: re-check the watermark AFTER
            // acquiring the mutex but BEFORE waiting, because the writer
            // may have bumped + notified between our outer check and here.
            let mut guard = self.visible_seq_wait_mu.lock();
            let recheck = self.visible_seq_for(namespace);
            if recheck >= min_seq {
                return Ok(());
            }
            let result = self.visible_seq_cv.wait_for(&mut guard, remaining);
            drop(guard);
            if result.timed_out() {
                let final_current = self.visible_seq_for(namespace);
                if final_current >= min_seq {
                    return Ok(());
                }
                return Err(crate::error::YantrikDbError::RyWaitTimeout {
                    namespace: namespace.to_string(),
                    requested_seq: min_seq,
                    observed_seq: final_current,
                    waited_ms: timeout.as_millis() as u64,
                });
            }
            // Spurious wakeup or notify_all — re-check the watermark.
        }
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
        assert_eq!(
            db.count_pending_ops().unwrap(),
            0,
            "fresh db has no pending"
        );

        let payload = serde_json::json!({
            "rid": "test_rid_1",
            "type": "episodic",
            "text": "first pending op",
        });
        let emb_bytes = fake_embedding(1.0, 64);
        let op_id = db
            .log_op_pending(
                "record",
                Some("test_rid_1"),
                &payload,
                None,
                Some(&emb_bytes),
            )
            .expect("log_op_pending");

        assert_eq!(
            db.count_pending_ops().unwrap(),
            1,
            "one pending op after append"
        );

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
        db.log_op_pending(
            "record",
            Some("rid_new"),
            &serde_json::json!({}),
            None,
            None,
        )
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
            .log_op_pending(
                "record",
                Some("rid_overflow"),
                &serde_json::json!({}),
                None,
                None,
            )
            .expect_err("11k must fail with backpressure");
        match err {
            crate::error::YantrikDbError::Backpressure {
                pending,
                max,
                retry_after_ms,
            } => {
                assert_eq!(max, 10_000);
                assert_eq!(pending, 10_000);
                assert!(retry_after_ms > 0, "retry hint must be non-zero");
            }
            other => panic!("expected Backpressure, got {other:?}"),
        }

        // After draining one, the next push must succeed (proves backpressure
        // is reactive, not sticky). v0.7.1: drain via the public
        // `mark_op_applied` API so the cached `pending_op_count` atomic
        // stays coherent. Bypassing it with raw SQL (the v0.7.0 shape)
        // wouldn't decrement the counter and would falsely keep
        // backpressure engaged — the new test path verifies the
        // atomic counter contract end-to-end.
        let one_op_id: String = {
            let conn = db.read_conn();
            conn.query_row(
                "SELECT op_id FROM oplog WHERE applied = 0 LIMIT 1",
                [],
                |row| row.get(0),
            )
            .unwrap()
        };
        let was_unset = db.mark_op_applied(&one_op_id).unwrap();
        assert!(was_unset, "mark_op_applied should win the transition");
        db.log_op_pending(
            "record",
            Some("rid_after_drain"),
            &serde_json::json!({}),
            None,
            None,
        )
        .expect("succeeds after one drained");
    }

    #[test]
    fn apply_pending_drains_then_marks() {
        let db = open_test_db();
        // Seed 3 pending ops of various types.
        for (op_type, target) in [
            ("record", "rid_1"),
            ("forget", "rid_2"),
            ("relate", "rid_3"),
        ] {
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
            db.log_op_pending(
                "record",
                Some(&format!("rid_{i}")),
                &serde_json::json!({}),
                None,
                None,
            )
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
        // v0.7.1: enqueue via log_op_pending with a synthetic op_type so
        // the cached `pending_op_count` atomic increments. v0.7.0 used
        // direct SQL INSERT here, but that bypasses the counter and
        // mismatches the public-API count_pending_ops contract — fixed
        // alongside the perf hotfix.
        db.log_op_pending(
            "made_up_op",
            Some("synth_unknown"),
            &serde_json::json!({}),
            None,
            None,
        )
        .unwrap();
        assert_eq!(db.count_pending_ops().unwrap(), 1);

        // Drain doesn't apply unknown op types — they stay pending so a
        // future runtime that knows the op type can drain them.
        let applied = db.apply_pending_ops_once(10).unwrap();
        assert_eq!(applied, 0);
        assert_eq!(
            db.count_pending_ops().unwrap(),
            1,
            "unknown op_type stays pending"
        );
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
    fn schema_version_meta_at_current() {
        // Locks the meta-stamp to the SCHEMA_VERSION constant so a future
        // bump (e.g. v26 → v27 from RFC 026's next phase) automatically
        // moves this test forward without a manual literal edit.
        // Previously hard-coded "25"; v26 (issue #29) made the brittleness
        // obvious so the renaming + constant reference go together.
        let db = open_test_db();
        let conn = db.read_conn();
        let v: String = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(v, crate::schema::SCHEMA_VERSION.to_string());
    }

    // ── Perf regression tests (added v0.7.1 after the SELECT COUNT incident) ──
    //
    // The v0.7.0 → v0.7.1 hotfix story (yantrikdb-server msg b951a2de):
    // log_op_pending was running `SELECT COUNT(*) FROM oplog WHERE applied=0`
    // on every foreground call. At 8 writers that's 16 conn acquisitions/sec
    // just for the backpressure check. v0.7.1 replaced it with an
    // AtomicI64 cached counter. These tests structurally pin both
    // properties (atomic stays coherent with SQL truth + backpressure check
    // is O(1) under load) so a future regression of the same class is
    // caught BEFORE it ships, not by yantrikdb-server's homelab bench.

    /// Atomic counter must never drift from SQL truth across mixed
    /// log_op_pending / mark_op_applied / apply_pending_ops_once
    /// operations. If this drifts, the backpressure check is wrong AND
    /// `count_pending_ops()` lies to the materializer. Drift = silent
    /// data loss possibility.
    #[test]
    fn pending_op_count_atomic_matches_sql_after_workload() {
        let db = open_test_db();

        // Phase 1: pure inserts.
        for i in 0..50 {
            db.log_op_pending(
                "record",
                Some(&format!("rid_w1_{i}")),
                &serde_json::json!({}),
                None,
                None,
            )
            .unwrap();
            assert_eq!(
                db.count_pending_ops().unwrap(),
                db.count_pending_ops_sql().unwrap(),
                "drift after insert #{i}"
            );
        }

        // Phase 2: mixed inserts + applies via apply_pending_ops_once
        // (which exercises the full mark_op_applied → atomic.fetch_sub path).
        let drained = db.apply_pending_ops_once(20).unwrap();
        assert!(drained > 0, "drained at least one");
        assert_eq!(
            db.count_pending_ops().unwrap(),
            db.count_pending_ops_sql().unwrap(),
            "drift after apply_pending_ops_once"
        );

        // Phase 3: more inserts after partial drain.
        for i in 0..30 {
            db.log_op_pending(
                "record",
                Some(&format!("rid_w3_{i}")),
                &serde_json::json!({}),
                None,
                None,
            )
            .unwrap();
        }
        assert_eq!(
            db.count_pending_ops().unwrap(),
            db.count_pending_ops_sql().unwrap(),
            "drift after second-wave inserts"
        );

        // Phase 4: drain all the rest.
        loop {
            let n = db.apply_pending_ops_once(100).unwrap();
            if n == 0 {
                break;
            }
        }
        assert_eq!(db.count_pending_ops().unwrap(), 0);
        assert_eq!(db.count_pending_ops_sql().unwrap(), 0);

        // Phase 5: idempotent re-mark must not double-decrement.
        let extra_op_id = db
            .log_op_pending(
                "record",
                Some("rid_extra"),
                &serde_json::json!({}),
                None,
                None,
            )
            .unwrap();
        let first_mark = db.mark_op_applied(&extra_op_id).unwrap();
        let second_mark = db.mark_op_applied(&extra_op_id).unwrap();
        assert!(first_mark, "first mark wins the transition");
        assert!(!second_mark, "second mark is a no-op (idempotent)");
        assert_eq!(
            db.count_pending_ops().unwrap(),
            0,
            "double-mark must NOT push counter negative"
        );
    }

    /// **Regression guard for the v0.7.0 → v0.7.1 hotfix.** Before the
    /// fix, `log_op_pending` did a SELECT COUNT(*) WHERE applied=0 per
    /// call. That made the foreground hot path scale O(pending_count)
    /// despite the partial index. The fix made it a single atomic load.
    ///
    /// This test pins the property: insert N pending ops, then time the
    /// next log_op_pending call. Even with 5000 pending ops in the
    /// oplog, the call should complete in <10ms (the actual production
    /// number is ~1µs; the threshold is ~10000× looser to absorb CI
    /// noise without false positives). If a regression re-introduces
    /// SELECT COUNT, the call time grows with the count and the
    /// assertion fires.
    #[test]
    fn log_op_pending_is_o1_under_pending_load() {
        use std::time::Instant;

        let db = open_test_db();
        // Seed 5000 pending ops. Each insert is a real conn acquire +
        // INSERT — this is the setup cost, not the test of interest.
        for i in 0..5000 {
            db.log_op_pending(
                "record",
                Some(&format!("rid_seed_{i}")),
                &serde_json::json!({}),
                None,
                None,
            )
            .expect("seed insert");
        }
        assert_eq!(db.count_pending_ops().unwrap(), 5000);

        // Time the next call. Pre-v0.7.1 this scanned 5000 oplog rows
        // via the partial index AND did a separate read_conn acquire.
        // Post-fix it's an atomic load. We give 10ms headroom for CI
        // noise; the actual cost should be sub-millisecond.
        let t0 = Instant::now();
        let _id = db
            .log_op_pending(
                "record",
                Some("rid_under_load"),
                &serde_json::json!({}),
                None,
                None,
            )
            .expect("call under load");
        let elapsed = t0.elapsed();
        assert!(
            elapsed.as_millis() < 10,
            "log_op_pending should be O(1); 5001st call took {elapsed:?} \
             (regression: probably SELECT COUNT re-introduced)"
        );
    }

    /// Atomic counter must remain non-negative and stable when many
    /// concurrent workers race on mark_op_applied. The applied=0 filter
    /// in the SQL guarantees exactly-once SQL transition; the
    /// `if won { fetch_sub }` guard inside mark_op_applied must
    /// preserve that into the atomic.
    #[test]
    fn pending_op_count_under_concurrent_mark() {
        use std::sync::Arc;
        use std::thread;

        let db = Arc::new(open_test_db());
        for i in 0..100 {
            db.log_op_pending(
                "record",
                Some(&format!("rid_conc_{i}")),
                &serde_json::json!({}),
                None,
                None,
            )
            .unwrap();
        }
        assert_eq!(db.count_pending_ops().unwrap(), 100);

        // Snapshot all op_ids so each worker can race to mark them.
        let op_ids: Vec<String> = {
            let conn = db.read_conn();
            let mut stmt = conn
                .prepare("SELECT op_id FROM oplog WHERE applied = 0")
                .unwrap();
            stmt.query_map([], |row| row.get::<_, String>(0))
                .unwrap()
                .filter_map(|r| r.ok())
                .collect()
        };

        // 4 workers each try to mark every op. The applied=0 filter +
        // bool return + `if won` guard means each op is decremented
        // exactly once across all workers.
        let mut handles = Vec::new();
        for _ in 0..4 {
            let db_c = Arc::clone(&db);
            let ids_c = op_ids.clone();
            handles.push(thread::spawn(move || {
                for id in ids_c {
                    let _ = db_c.mark_op_applied(&id);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(
            db.count_pending_ops().unwrap(),
            0,
            "concurrent racing on mark_op_applied must converge atomic to 0"
        );
        assert_eq!(
            db.count_pending_ops_sql().unwrap(),
            0,
            "and SQL truth must agree"
        );
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

    // ── Phase 4.3 (saga task 3): materialize_record_post dispatch arm ──
    //
    // These tests exercise the worker-side materialization path WITHOUT
    // changing foreground behavior. They enqueue an op directly via
    // log_op_pending and drain via apply_pending_ops_once, asserting the
    // resulting SQL/graph state matches what the foreground inline loop
    // would have produced. Commit B will flip foreground to enqueue;
    // these tests act as the contract pin so the flip is provably safe.

    fn enqueue_post_record(db: &YantrikDB, rid: &str, text: &str, namespace: &str) -> String {
        // First INSERT a stub memories row so memory_entities + claims
        // FK references are valid. Foreground (Commit B) will INSERT the
        // memories row before enqueuing; tests mirror that ordering.
        let conn = db.conn();
        let stored_text = db.encrypt_text(text).unwrap();
        let ts = super::super::now();
        conn.execute(
            "INSERT INTO memories \
             (rid, type, text, embedding, created_at, updated_at, importance, \
              half_life, last_access, valence, metadata, namespace, \
              certainty, domain, source, emotional_state) \
             VALUES (?1, 'episodic', ?2, NULL, ?3, ?3, 0.5, 604800.0, ?3, 0.0, '{}', ?4, 0.8, 'general', 'user', NULL)",
            params![rid, stored_text, ts, namespace],
        ).unwrap();
        drop(conn);

        let payload = serde_json::json!({
            "rid": rid,
            "text": stored_text,
            "namespace": namespace,
            "ts_secs": ts,
            "domain": "general",
            "source": "user",
        });
        db.log_op_pending(
            crate::engine::op_types::OP_MATERIALIZE_RECORD_POST,
            Some(rid),
            &payload,
            None,
            None,
        )
        .expect("log_op_pending")
    }

    #[test]
    fn materialize_record_post_inserts_entities() {
        let db = open_test_db();
        let _op_id = enqueue_post_record(&db, "r1", "Alice met Acme yesterday", "default");
        assert_eq!(db.count_pending_ops().unwrap(), 1);

        let n = db.apply_pending_ops_once(10).unwrap();
        assert_eq!(n, 1, "one op drained");
        assert_eq!(db.count_pending_ops().unwrap(), 0);

        // Entities table should now contain Alice and Acme.
        let conn = db.read_conn();
        let alice: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM entities WHERE name = 'Alice'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let acme: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM entities WHERE name = 'Acme'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(alice, 1, "Alice seeded by heuristic");
        assert_eq!(acme, 1, "Acme seeded by heuristic");
    }

    #[test]
    fn materialize_record_post_inserts_memory_entities() {
        let db = open_test_db();
        let _op_id = enqueue_post_record(&db, "r2", "Bob works at Beta Corp", "default");
        let _ = db.apply_pending_ops_once(10).unwrap();

        let conn = db.read_conn();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memory_entities WHERE memory_rid = 'r2'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            count >= 2,
            "memory_entities has at least 2 rows for Bob+Beta Corp; got {count}"
        );
    }

    #[test]
    fn materialize_record_post_idempotent_on_double_drain() {
        // Drain twice — second drain must be a no-op (op already marked
        // applied by first drain). entities mention_count must stay at 1.
        let db = open_test_db();
        let _op_id = enqueue_post_record(&db, "r3", "Charlie went to Delta", "default");
        let n1 = db.apply_pending_ops_once(10).unwrap();
        let n2 = db.apply_pending_ops_once(10).unwrap();
        assert_eq!(n1, 1, "first drain applies");
        assert_eq!(n2, 0, "second drain finds nothing pending");

        let conn = db.read_conn();
        let mc: i64 = conn
            .query_row(
                "SELECT mention_count FROM entities WHERE name = 'Charlie'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(mc, 1, "mention_count not double-bumped");
    }

    #[test]
    fn materialize_record_post_updates_graph_index() {
        // graph_index in-memory state must reflect the worker's apply
        // — this is what makes recall-by-entity find the new memory.
        let db = open_test_db();
        let _op_id = enqueue_post_record(&db, "r4", "Eve climbed Everest", "default");
        let _ = db.apply_pending_ops_once(10).unwrap();

        let gi = db.graph_index.read();
        let names = gi.all_entity_names();
        assert!(names.iter().any(|n| n == "Eve"), "Eve in graph_index");
        assert!(
            names.iter().any(|n| n == "Everest"),
            "Everest in graph_index"
        );
    }

    #[test]
    fn materialize_record_post_concurrent_workers_no_double_apply() {
        // 4 worker threads + 20 ops → exactly-once semantics on the
        // applied=0 filter. Same race-safety as the existing materializer
        // tests, but exercising the new dispatch path.
        use std::sync::Arc;
        use std::thread;

        let db = Arc::new(open_test_db());
        for i in 0..20 {
            let _ = enqueue_post_record(
                &db,
                &format!("rcc_{i}"),
                &format!("Person{i} met Place{i}"),
                "default",
            );
        }
        assert_eq!(db.count_pending_ops().unwrap(), 20);

        let mut handles = Vec::new();
        for _ in 0..4 {
            let db_c = Arc::clone(&db);
            handles.push(thread::spawn(move || {
                let mut total = 0;
                while db_c.count_pending_ops().unwrap() > 0 {
                    total += db_c.apply_pending_ops_once(50).unwrap();
                    if total >= 20 {
                        break;
                    }
                }
                total
            }));
        }
        let totals: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(
            totals.iter().sum::<usize>(),
            20,
            "exactly 20 applies across all workers, no double-counting; got {totals:?}"
        );
        assert_eq!(db.count_pending_ops().unwrap(), 0);

        // entities should have 20 distinct Person* rows, 20 distinct Place* rows.
        let conn = db.read_conn();
        let person_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM entities WHERE name LIKE 'Person%'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(person_count, 20);
    }

    #[test]
    fn materialize_record_post_invalid_payload_leaves_op_pending() {
        // Malformed payload → worker logs warning, leaves op pending for
        // retry. Must not advance applied flag (otherwise we'd silently
        // lose data on a transient parse failure).
        let db = open_test_db();
        let bad_payload = serde_json::json!({"not_a_rid": "oops"});
        let _ = db
            .log_op_pending(
                crate::engine::op_types::OP_MATERIALIZE_RECORD_POST,
                Some("r_bad"),
                &bad_payload,
                None,
                None,
            )
            .unwrap();
        let n = db.apply_pending_ops_once(10).unwrap();
        assert_eq!(n, 0, "malformed op not applied");
        assert_eq!(
            db.count_pending_ops().unwrap(),
            1,
            "still pending for retry"
        );
    }
}
