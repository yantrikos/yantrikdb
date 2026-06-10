use rusqlite::params;
use std::sync::Arc;

use crate::error::Result;
use crate::serde_helpers::serialize_f32;
use crate::types::*;

use super::reembed::SearchState;
use super::write_router::SyncWriteGuard;
use super::{embedding_hash, now, sanitize, YantrikDB};

/// Coerce a blank namespace to the canonical default (v0.7.23).
///
/// The schema column default and the Python/MCP bindings all use
/// `"default"`; an empty or whitespace-only namespace is virtually always
/// a caller-side defaulting accident (e.g. a server gateway doing
/// `unwrap_or("")`). Normalizing it at the engine boundary keeps a single
/// canonical value so writes, point reads, list filters, and recall all
/// agree across every consumer instead of silently persisting an unscoped
/// `""` partition that no reader queries for.
pub(crate) fn normalize_namespace(ns: &str) -> &str {
    if ns.trim().is_empty() {
        "default"
    } else {
        ns
    }
}

impl YantrikDB {
    /// Store a new memory and return its RID.
    ///
    /// **Issue #41 layer 3 — WriteRouter gating.** At entry, the writer
    /// attempts to acquire a `SyncWriteGuard`. If the engine's
    /// `write_router` is in `Normal` state (no reembed in progress),
    /// the guard is acquired and the synchronous path runs: INSERT
    /// memories + vec_index.append + log_op (applied=1). The guard is
    /// held for the full critical section and drops via RAII when
    /// `record` returns, decrementing the inflight-writer counter.
    /// This is the brainstorm-2 invariant that prevents in-flight
    /// writers from committing `applied=1` against an about-to-be-
    /// discarded old generation during reembed cutover.
    ///
    /// If the router is in `Queueing` state (reembed has flipped the
    /// gate and is waiting for writers to drain before capturing
    /// `build_hwm`), `try_enter_sync_writer()` returns None and this
    /// call routes through the queued path: the op is appended to
    /// `oplog` with `applied=0`, `embedding_model = old_embedder_name`,
    /// the full record payload (text + metadata) — the post-swap
    /// materializer re-encodes under the new embedder + applies to
    /// the new generation. The caller's return value (rid + seq) is
    /// the same shape; read-after-write requires `recall_with_seq` to
    /// wait for the new generation's `visible_seq` to advance.
    #[tracing::instrument(skip(self, metadata, embedding), fields(memory_type, namespace))]
    pub fn record(
        &self,
        text: &str,
        memory_type: &str,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: &serde_json::Value,
        embedding: &[f32],
        namespace: &str,
        certainty: f64,
        domain: &str,
        source: &str,
        emotional_state: Option<&str>,
    ) -> Result<String> {
        // Task 29 (Ingest Integrity): strip any leaked tool-call
        // serialization tail from the stored text. On this entry point the
        // caller supplies the embedding, so the vector may still reflect the
        // pre-clean text; that minor staleness is strictly better than
        // persisting the artifact, and the dominant ingest paths
        // (`record_text`, MCP/HTTP) embed engine-side on the cleaned text.
        let sanitized = sanitize::sanitize_tool_call_artifacts(text);
        let text = sanitized.as_ref();
        // v0.7.23: coerce a blank namespace to the canonical default so no
        // consumer persists an unscoped "" partition. Shadows the param so
        // both the sync and queued paths below see the normalized value.
        let namespace = normalize_namespace(namespace);
        // Task 31 (Ingest Integrity): calibrate importance against this
        // namespace's running distribution before it is stored OR replicated
        // (the sync, queued, and oplog paths all flow from the value below).
        let importance = self.calibrate_importance(namespace, importance)?;
        // Issue #41 layer 3: route on write_router state. The guard
        // (if acquired) is held for the full sync path and drops via
        // RAII at function return, panic-safe.
        let sync_guard = self.write_router.try_enter_sync_writer();
        if sync_guard.is_none() {
            // Queueing state — take the queued path. Reembed cutover
            // is in flight; writes go to oplog and the post-swap
            // materializer applies them under the new embedder.
            return self.record_queued(
                text,
                memory_type,
                importance,
                valence,
                half_life,
                metadata,
                embedding,
                namespace,
                certainty,
                domain,
                source,
                emotional_state,
            );
        }
        // guard is held; RAII Drop at function exit decrements inflight.
        let guard = sync_guard.unwrap();

        // **Issue #41 brainstorm-4 §1.** Load SearchState AFTER the
        // guard is acquired. With the guard held, reembed cannot
        // complete its swap, so the loaded state is the published
        // active generation for the entire critical section. Note:
        // for `record()` (caller-supplied embedding), the engine
        // cannot verify the embedding's generation provenance — the
        // caller is responsible for using the embedder consistent
        // with the active generation. `record_text()` (engine-
        // supplied embedding) has a revalidation loop that ensures
        // the embedding and the active generation match.
        let state = self.search_state.load_full();

        self.record_under_guard_and_state(
            state,
            guard,
            text,
            memory_type,
            importance,
            valence,
            half_life,
            metadata,
            embedding,
            namespace,
            certainty,
            domain,
            source,
            emotional_state,
        )
    }

    /// **Issue #41 brainstorm-4 §2.** The post-guard, post-load
    /// critical section shared by `record()` and `record_text()`.
    ///
    /// Caller MUST hold the `SyncWriteGuard` — this is the contract
    /// that prevents reembed from completing its SearchState swap
    /// while we are mid-commit, and the contract that makes
    /// `state.generation` the durable answer to "what generation am I
    /// committing under." The guard is moved in by value and drops
    /// via RAII at function exit, decrementing the in-flight counter.
    ///
    /// Caller MUST also pre-load `state` from `self.search_state` and
    /// pass it in — this commit path uses the snapshot rather than
    /// re-loading, so writer revalidation logic in `record_text()`
    /// (which re-loads after embed to detect a generation advance)
    /// is the single source of truth for generation safety on the
    /// text-embed path.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_under_guard_and_state(
        &self,
        state: Arc<SearchState>,
        _guard: SyncWriteGuard<'_>,
        text: &str,
        memory_type: &str,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: &serde_json::Value,
        embedding: &[f32],
        namespace: &str,
        certainty: f64,
        domain: &str,
        source: &str,
        emotional_state: Option<&str>,
    ) -> Result<String> {
        let rid = crate::id::new_id();
        let ts = now();
        let emb_blob = serialize_f32(embedding);
        let meta_str = serde_json::to_string(metadata)?;

        // Encrypt fields if encryption is enabled
        let stored_text = self.encrypt_text(text)?;
        let stored_meta = self.encrypt_text(&meta_str)?;
        let stored_emb = self.encrypt_embedding(&emb_blob)?;

        // Read active session for this namespace into a local before acquiring conn
        let session_id = self.active_sessions.read().get(namespace).cloned();

        // **Issue #41 brainstorm-4 §6.** Stamp the v28
        // embedding_generation column with the snapshot's generation
        // so the post-swap materializer can discriminate "this row
        // was indexed under the active generation — skip" from "this
        // row was inserted under an old generation — needs re-encode."
        // Read from `state.generation` (not a fresh load) because we
        // hold the SyncWriteGuard for the entire sync path:
        // search_state cannot advance under us until the guard drops.
        let embedding_generation: i64 = state.generation as i64;

        // Acquire conn, do all SQL, then drop before other locks
        {
            let conn = self.conn();
            conn.execute(
                "INSERT INTO memories \
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

            // Auto-link to active session for this namespace
            if let Some(session_id) = &session_id {
                conn.execute(
                    "UPDATE memories SET session_id = ?1 WHERE rid = ?2",
                    params![session_id, rid],
                )?;
                conn.execute(
                    "UPDATE sessions SET memory_count = memory_count + 1 WHERE session_id = ?1",
                    params![session_id],
                )?;
            }
        }
        // conn dropped here

        // Insert into vector index (lock ordering: conn already dropped)
        let seq = self
            .vec_seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        // **Issue surfaced in CT 132 bench post-v0.7.18 (2026-05-20):
        // orphan-on-Backpressure pattern.** If the vec_index.append
        // returns Err (Backpressure when delta is full, dim mismatch,
        // etc.), the memories row inserted above is already committed
        // but the rest of the critical section (vec_index, oplog,
        // visible_seq, materializer enqueue) is skipped. The caller
        // sees an Err and assumes the write failed — but the row
        // exists in SQL with no oplog provenance. Over 39 days on
        // trader's `default` DB, this leaked 23k rows.
        //
        // The compensating DELETE here runs only on the rare failure
        // path. On Backpressure (the common error), the DELETE
        // immediately reclaims the row so SQL state matches what the
        // caller observes (write rejected, no row created).
        if let Err(e) = state.vec_index.append(rid.clone(), embedding.to_vec(), seq) {
            let conn = self.conn();
            let _ = conn.execute("DELETE FROM memories WHERE rid = ?1", params![rid]);
            // **v0.7.23 residual fix.** The compensating DELETE above
            // reclaims the orphaned memories row, but the session
            // `memory_count` bumped in the conn block above survives the
            // delete and over-counts by 1 per backpressure-rejected
            // record. Reverse it so the session stat matches the rows
            // that actually exist.
            if let Some(session_id) = &session_id {
                let _ = conn.execute(
                    "UPDATE sessions SET memory_count = memory_count - 1 WHERE session_id = ?1",
                    params![session_id],
                );
            }
            return Err(e);
        }
        self.bump_visible_seq(namespace, seq);

        // Insert into scoring cache (conn and vec_index dropped)
        self.cache_insert(
            rid.clone(),
            ScoringRow {
                created_at: ts,
                importance,
                half_life,
                last_access: ts,
                access_count: 0,
                valence,
                consolidation_status: "active".to_string(),
                memory_type: memory_type.to_string(),
                namespace: namespace.to_string(),
                certainty,
                domain: domain.to_string(),
                source: source.to_string(),
                emotional_state: emotional_state.map(|s| s.to_string()),
            },
        );

        // Log the user-facing "record" op FIRST so external consumers
        // (replication extract_ops_since, oplog inspectors) see records
        // in their natural causal order: the record came before any
        // post-record materialization queued in its wake.
        let emb_hash = embedding_hash(embedding);
        self.log_op(
            "record",
            Some(&rid),
            &serde_json::json!({
                "rid": rid,
                "type": memory_type,
                "text": text,
                "importance": importance,
                "valence": valence,
                "half_life": half_life,
                "metadata": metadata,
                "created_at": ts,
                "updated_at": ts,
                "namespace": namespace,
                "certainty": certainty,
                "domain": domain,
                "source": source,
                "emotional_state": emotional_state,
            }),
            Some(&emb_hash),
        )?;

        // **Phase 4.3 Commit B (saga task 3, 2026-05-08).** The
        // unbounded entity / memory_entities / claims loops that used
        // to live here are now enqueued for the materializer thread to
        // run off the request path. See docs/phase_4_3_design.md for
        // the contract change (synchronous read-after-write of
        // entity-graph queries shifts from immediate to ms-scale; the
        // delta-recall path is unaffected since DeltaIndex.append
        // happened above on the foreground thread).
        {
            let post_payload = serde_json::json!({
                "rid": rid,
                "text": stored_text,
                "namespace": namespace,
                "ts_secs": ts,
                "domain": domain,
                "source": source,
            });
            self.log_op_pending(
                crate::engine::op_types::OP_MATERIALIZE_RECORD_POST,
                Some(&rid),
                &post_payload,
                None,
                None,
            )?;
        }

        Ok(rid)
    }

    /// Record multiple memories in a single transaction.
    /// Uses SAVEPOINT for atomicity while keeping `&self` (no `&mut self`).
    #[tracing::instrument(skip(self, inputs), fields(batch_size = inputs.len()))]
    pub fn record_batch(&self, inputs: &[RecordInput]) -> Result<Vec<String>> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }

        // Task 29 (Ingest Integrity): strip any leaked tool-call
        // serialization tail from every input's text once, up front. The
        // same cleaned text feeds entity extraction, the stored row, and the
        // audit features below (indexed positionally — `rids` preserves input
        // order). Borrowed (no allocation) on the clean path; the
        // caller-supplied embedding is left as-is, as in `record`.
        let sanitized_texts: Vec<std::borrow::Cow<'_, str>> = inputs
            .iter()
            .map(|i| sanitize::sanitize_tool_call_artifacts(&i.text))
            .collect();

        // Task 31 (Ingest Integrity): calibrate each input's importance
        // against its namespace distribution (positionally aligned with
        // `inputs`). Calls run in order, so later items in the batch see the
        // running-mean effect of earlier ones in the same namespace.
        let calibrated_importances: Vec<f64> = inputs
            .iter()
            .map(|i| self.calibrate_importance(&i.namespace, i.importance))
            .collect::<Result<Vec<_>>>()?;

        // **Issue #41 brainstorm-4 §1.** SearchState snapshot for the
        // batch — every append in this batch lands on the same
        // generation-anchored DeltaIndex.
        let state = self.search_state.load_full();

        // Clone active sessions map before acquiring conn
        let sessions = self.active_sessions.read().clone();

        // Precompute entity candidates per memory before touching conn/graph_index.
        // Two sources:
        //   (a) heuristic extraction from text (capitalized proper-nouns)
        //   (b) match against already-known entities in graph_index
        let known_entities = self.graph_index.read().all_entity_names();
        let per_memory_linkage: Vec<(Vec<String>, std::collections::HashSet<String>)> =
            sanitized_texts
            .iter()
            .map(|text| {
                let text = text.as_ref();
                let text_tokens = crate::graph::tokenize(text);
                let heuristic = crate::graph::extract_heuristic_entities(text);
                let mut candidates: std::collections::HashSet<String> =
                    heuristic.iter().cloned().collect();
                for known in &known_entities {
                    if crate::graph::entity_matches_text(known, &text_tokens) {
                        candidates.insert(known.clone());
                    }
                }
                (heuristic, candidates)
            })
            .collect();

        let mut rids = Vec::with_capacity(inputs.len());

        // Lock conn once for the entire batch SQL work
        {
            let conn = self.conn();
            conn.execute_batch("SAVEPOINT batch_record")?;

            for (idx, input) in inputs.iter().enumerate() {
                let rid = crate::id::new_id();
                let ts = now();
                let emb_blob = serialize_f32(&input.embedding);
                let meta_str = serde_json::to_string(&input.metadata)?;

                // Encrypt fields if encryption is enabled. Task 29: store the
                // sanitized text (positionally aligned with `inputs`).
                let stored_text = self.encrypt_text(sanitized_texts[idx].as_ref())?;
                let stored_meta = self.encrypt_text(&meta_str)?;
                let stored_emb = self.encrypt_embedding(&emb_blob)?;

                // **Issue #41 brainstorm-4 §6.** v28 embedding_generation
                // stamped from the batch's snapshot.
                let embedding_generation: i64 = state.generation as i64;
                let result = conn.execute(
                    "INSERT INTO memories \
                     (rid, type, text, embedding, created_at, updated_at, importance, \
                      half_life, last_access, valence, metadata, namespace, \
                      certainty, domain, source, emotional_state, embedding_generation) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
                    params![rid, input.memory_type, stored_text, stored_emb, ts, ts,
                            calibrated_importances[idx], input.half_life, ts, input.valence, stored_meta,
                            input.namespace, input.certainty, input.domain, input.source,
                            input.emotional_state, embedding_generation],
                );

                if let Err(e) = result {
                    conn.execute_batch("ROLLBACK TO batch_record")?;
                    return Err(e.into());
                }

                rids.push(rid);
            }

            // Auto-link batch to active sessions
            for (rid, input) in rids.iter().zip(inputs.iter()) {
                if let Some(session_id) = sessions.get(&input.namespace) {
                    conn.execute(
                        "UPDATE memories SET session_id = ?1 WHERE rid = ?2",
                        params![session_id, rid],
                    )?;
                    conn.execute(
                        "UPDATE sessions SET memory_count = memory_count + 1 WHERE session_id = ?1",
                        params![session_id],
                    )?;
                }
            }

            // Persist entity linkage (SQL side). graph_index in-memory update
            // happens after conn is dropped to avoid holding two write locks.
            let batch_ts = now();
            for (rid, (heuristic, candidates)) in rids.iter().zip(per_memory_linkage.iter()) {
                for entity in heuristic {
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
                        params![entity, entity_type, batch_ts],
                    )?;
                }
                for entity in candidates {
                    conn.execute(
                        "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name) VALUES (?1, ?2)",
                        params![rid, entity],
                    )?;
                }
            }

            conn.execute_batch("RELEASE batch_record")?;
        }
        // conn dropped; now update graph_index in-memory.
        {
            let mut gi = self.graph_index.write();
            for (rid, (_, candidates)) in rids.iter().zip(per_memory_linkage.iter()) {
                for entity in candidates {
                    let entity_type = crate::graph::classify_entity_type(entity);
                    gi.add_entity(entity, entity_type);
                    gi.link_memory(rid, entity);
                }
            }
        }

        // RFC 006 Phase 0: emit one audit event per memory in the batch.
        for (idx, (rid, (input, (heuristic_entities, candidates)))) in rids
            .iter()
            .zip(inputs.iter().zip(per_memory_linkage.iter()))
            .enumerate()
        {
            let heuristic_vec: Vec<String> = heuristic_entities.iter().cloned().collect();
            let features =
                crate::graph::analyze_text_features(sanitized_texts[idx].as_ref(), &heuristic_vec);
            tracing::info!(
                target: "yantrikdb::audit::extraction",
                namespace = %input.namespace,
                memory_rid = %rid,
                domain = %input.domain,
                source = %input.source,
                extractor_version = "heuristic_v1",
                batch = true,
                char_length = features.char_length,
                sentence_count = features.sentence_count,
                entity_count = features.entity_count,
                entities_matched_in_graph = candidates.len().saturating_sub(heuristic_entities.len()),
                negation_cue_count = features.negation_cue_count,
                temporal_cue_count = features.temporal_cue_count,
                modality_cue_count = features.modality_cue_count,
                has_compound_markers = features.has_compound_markers,
                likely_assertion = features.likely_assertion,
                "extraction audit"
            );
        }

        // Append to vec_index (DeltaIndex) after SQL commit.
        // **v0.7.19 orphan-on-Backpressure fix.** If any append in
        // the batch fails (delta saturation, dim mismatch), the
        // SAVEPOINT above has already committed all N memories
        // rows. Compensating DELETE clears the entire batch so the
        // caller sees an atomic batch-fail outcome rather than
        // partial-commit state. See record() for the rationale on
        // single-row writes.
        for (idx, (rid, input)) in rids.iter().zip(inputs.iter()).enumerate() {
            let seq = self
                .vec_seq
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            if let Err(e) = state
                .vec_index
                .append(rid.clone(), input.embedding.clone(), seq)
            {
                // Roll back all N rows from memories (DELETE is fast
                // under a single conn lock; idempotent via WHERE).
                let conn = self.conn();
                for r in &rids {
                    let _ = conn.execute("DELETE FROM memories WHERE rid = ?1", params![r]);
                }
                // **v0.7.23 residual fix.** Reverse the per-session
                // `memory_count` bumps committed in the SAVEPOINT above,
                // once per memory that was session-linked — mirrors the
                // increment loop exactly so the stat matches the rows the
                // compensating DELETE just removed.
                for input in inputs.iter() {
                    if let Some(session_id) = sessions.get(&input.namespace) {
                        let _ = conn.execute(
                            "UPDATE sessions SET memory_count = memory_count - 1 WHERE session_id = ?1",
                            params![session_id],
                        );
                    }
                }
                let _ = idx; // index of the failing entry, kept for future logging
                return Err(e);
            }
            self.bump_visible_seq(&input.namespace, seq);
        }
        // vec_index dropped, now scoring_cache
        {
            let mut cache = self.scoring_cache.write();
            for (idx, (rid, input)) in rids.iter().zip(inputs.iter()).enumerate() {
                let ts = now();
                cache.insert(
                    rid.clone(),
                    ScoringRow {
                        created_at: ts,
                        importance: calibrated_importances[idx],
                        half_life: input.half_life,
                        last_access: ts,
                        access_count: 0,
                        valence: input.valence,
                        consolidation_status: "active".to_string(),
                        memory_type: input.memory_type.clone(),
                        namespace: input.namespace.clone(),
                        certainty: input.certainty,
                        domain: input.domain.clone(),
                        source: input.source.clone(),
                        emotional_state: input.emotional_state.clone(),
                    },
                );
            }
        }

        // Log a single batch op (log_op locks conn internally)
        self.log_op(
            "record_batch",
            None,
            &serde_json::json!({
                "count": rids.len(),
                "rids": rids,
            }),
            None,
        )?;

        Ok(rids)
    }

    /// **Issue #9 — deterministic mutation primitive for cluster replication.**
    ///
    /// Sibling of `record()` that takes a caller-assigned rid + caller-supplied
    /// embedding + materialized extracted_entities + caller-supplied
    /// timestamp + embedding_model. Engine does NOT call its own embedder
    /// or NER. Used by yantrikdb-server's cluster-mode applier so
    /// replicated writes are byte-deterministic across leader + followers.
    ///
    /// # Contract
    ///
    /// - **Idempotent on rid**: a second call with the same rid + identical
    ///   other fields succeeds without error and produces identical engine
    ///   state (INSERT OR IGNORE on memories, INSERT OR IGNORE on entities,
    ///   INSERT OR IGNORE on memory_entities, DeltaIndex.append idempotent
    ///   on rid+seq).
    /// - **Caller supplies the embedding.** Engine validates dim and rejects
    ///   `Error::EmbeddingDimensionMismatch` on mismatch — diverged dim is
    ///   undetectable until a query notices, so we fail loudly.
    /// - **Caller supplies created_at_unix_micros.** Materialized into both
    ///   `created_at REAL` (for back-compat scoring) and the v25
    ///   `created_at_unix_micros INTEGER` column. No engine-side `now()`
    ///   call on this path — leader stamps once, followers replay verbatim.
    /// - **Caller supplies extracted_entities.** Engine writes entity_edges
    ///   accordingly. Empty slice = no edges; engine does NOT fall back to
    ///   its own NER. (Heuristic NER lives in `crate::knowledge::graph` and
    ///   is callable directly by the leader if needed — see issue #9 thread.)
    /// - **Caller supplies embedding_model.** Stored on the row as the
    ///   engine-deterministic-surface version pin. RFC 013 may swap the
    ///   field type later behind the same column name.
    /// - **Caller-supplied `seq`** (cluster mode): when `Some(n)`, the
    ///   engine uses `n` as the delta-entry seq and the visible_seq bump
    ///   value, and ratchets `vec_seq` up to at least `n`. Per design
    ///   lock 2026-05-07, the seq IS the openraft commit-log index in
    ///   cluster mode, giving byte-deterministic per-namespace
    ///   visible_seq across leader + followers. Single-node callers pass
    ///   `None` and the engine allocates the seq itself.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success or idempotent re-apply. The rid is the input,
    /// not the output — caller already owns it.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(
        skip(self, metadata, embedding, extracted_entities),
        fields(rid, memory_type, namespace, embedding_model)
    )]
    pub fn record_with_rid(
        &self,
        rid: &str,
        text: &str,
        memory_type: &str,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: &serde_json::Value,
        embedding: &[f32],
        namespace: &str,
        certainty: f64,
        domain: &str,
        source: &str,
        emotional_state: Option<&str>,
        created_at_unix_micros: i64,
        extracted_entities: &[&str],
        embedding_model: &str,
        seq: Option<u64>,
    ) -> Result<()> {
        // v0.7.23: coerce a blank namespace to the canonical default. This is
        // the path the server's commit applier uses (record_with_rid on every
        // node), so it closes the gateway `unwrap_or("")` footgun at the
        // engine boundary for all replicas.
        let namespace = normalize_namespace(namespace);
        // Determinism gate: dim must match. Diverged dim = silent corruption.
        if embedding.len() != self.embedding_dim {
            return Err(crate::error::YantrikDbError::EmbeddingDimensionMismatch {
                expected: self.embedding_dim,
                got: embedding.len(),
            });
        }

        // **Issue #41 brainstorm-4 §1.** SearchState snapshot for the
        // determinstic-replay path. The replicated write lands on the
        // currently-active generation's DeltaIndex.
        let state = self.search_state.load_full();

        // Caller-supplied timestamp — NEVER call now() on this path.
        let ts_secs = (created_at_unix_micros as f64) / 1_000_000.0;
        let emb_blob = serialize_f32(embedding);
        let meta_str = serde_json::to_string(metadata)?;

        // Encryption is engine-side and deterministic given the same DEK +
        // same plaintext bytes (AES-GCM is non-deterministic across IVs but
        // the encrypt-once-on-leader model means each follower receives the
        // already-encrypted bytes via the WAL replication path — Phase 4
        // wires that. For now we encrypt locally; cluster-mode follower
        // apply will skip this step in a follow-up patch.)
        let stored_text = self.encrypt_text(text)?;
        let stored_meta = self.encrypt_text(&meta_str)?;
        let stored_emb = self.encrypt_embedding(&emb_blob)?;

        let session_id = self.active_sessions.read().get(namespace).cloned();

        // Single conn block: INSERT OR IGNORE on memories (idempotent on rid),
        // session links, entity persistence. SAVEPOINT for atomicity within
        // the call.
        let was_new_row: bool = {
            let conn = self.conn();
            conn.execute_batch("SAVEPOINT record_with_rid")?;

            let result: Result<bool> = (|| {
                // **Issue #41 brainstorm-4 §6.** v28 embedding_generation
                // stamp from the SearchState snapshot loaded above.
                let embedding_generation: i64 = state.generation as i64;
                let inserted = conn.execute(
                    "INSERT OR IGNORE INTO memories \
                     (rid, type, text, embedding, created_at, updated_at, importance, \
                      half_life, last_access, valence, metadata, namespace, \
                      certainty, domain, source, emotional_state, \
                      created_at_unix_micros, embedding_model, embedding_generation) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?5, ?6, ?7, ?5, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
                    params![
                        rid, memory_type, stored_text, stored_emb,
                        ts_secs,
                        importance, half_life, valence, stored_meta, namespace,
                        certainty, domain, source, emotional_state,
                        created_at_unix_micros, embedding_model,
                        embedding_generation,
                    ],
                )?;
                let was_new_row = inserted == 1;

                if was_new_row {
                    // Auto-link only on first insert. Replay should not
                    // re-bump session memory_count.
                    if let Some(session_id) = &session_id {
                        conn.execute(
                            "UPDATE memories SET session_id = ?1 WHERE rid = ?2",
                            params![session_id, rid],
                        )?;
                        conn.execute(
                            "UPDATE sessions SET memory_count = memory_count + 1 WHERE session_id = ?1",
                            params![session_id],
                        )?;
                    }
                }

                // **Phase 4.3 Commit C (saga task 19, 2026-05-08).** The
                // entity / memory_entities INSERT loop was previously here
                // inside the SAVEPOINT, holding `db.conn().lock()` for
                // O(extracted_entities.len()) statements. Now enqueued as
                // OP_MATERIALIZE_RECORD_WITH_RID_POST after the SAVEPOINT
                // releases. See docs/phase_4_3_design.md for the contract.

                Ok(was_new_row)
            })();

            match result {
                Ok(b) => {
                    conn.execute_batch("RELEASE record_with_rid")?;
                    b
                }
                Err(e) => {
                    let _ = conn.execute_batch("ROLLBACK TO record_with_rid");
                    let _ = conn.execute_batch("RELEASE record_with_rid");
                    return Err(e);
                }
            }
        };
        // conn dropped

        // DeltaIndex append. The seq is either caller-supplied (cluster
        // mode: openraft commit-log index for byte-deterministic replay)
        // or engine-allocated (single-node). On idempotent replay the rid
        // is the same and the seq is identical (cluster) or fresh
        // (single-node retry); the compactor's highest-seq-wins rule
        // converges state identically on both paths.
        let seq = self.assign_seq(seq);
        // **v0.7.19 orphan-on-Backpressure fix.** The trader's
        // `trader_ledger` DB shows `record_with_rid` pinned at
        // exactly 256 (the v0.7.17 delta_max wedge ceiling) — every
        // additional call after that left a memories row from the
        // INSERT OR IGNORE above with no oplog provenance because
        // the vec_index.append Err short-circuited the log_op below.
        // Compensating DELETE on failure. Skip the delete when
        // was_new_row=false (replay path: the row pre-existed; we
        // shouldn't yank it).
        if let Err(e) = state
            .vec_index
            .append(rid.to_string(), embedding.to_vec(), seq)
        {
            if was_new_row {
                let conn = self.conn();
                let _ = conn.execute("DELETE FROM memories WHERE rid = ?1", params![rid]);
                // **v0.7.23 residual fix.** The session `memory_count`
                // bumped inside the (already-RELEASEd) SAVEPOINT survives
                // the compensating DELETE. Reverse it so the stat matches
                // the surviving rows. Mirrors the was_new_row guard on the
                // original bump — replay (was_new_row=false) never bumped.
                if let Some(session_id) = &session_id {
                    let _ = conn.execute(
                        "UPDATE sessions SET memory_count = memory_count - 1 WHERE session_id = ?1",
                        params![session_id],
                    );
                }
            }
            return Err(e);
        }
        self.bump_visible_seq(namespace, seq);

        // Scoring cache (engine-internal; replay safe since insert is
        // overwrite-on-rid).
        if was_new_row {
            self.cache_insert(
                rid.to_string(),
                ScoringRow {
                    created_at: ts_secs,
                    importance,
                    half_life,
                    last_access: ts_secs,
                    access_count: 0,
                    valence,
                    consolidation_status: "active".to_string(),
                    memory_type: memory_type.to_string(),
                    namespace: namespace.to_string(),
                    certainty,
                    domain: domain.to_string(),
                    source: source.to_string(),
                    emotional_state: emotional_state.map(|s| s.to_string()),
                },
            );
        }

        // Op log entry — applied=1 since leader has materialized inline.
        // Followers will receive a separate replicated entry via the
        // cluster sync path; this path never logs applied=0.
        //
        // Logged BEFORE the post-record materialization enqueue so
        // extract_ops_since reports the user-data op in causal order
        // (record_with_rid arrived, then its entity-link materialization
        // was queued).
        let emb_hash = embedding_hash(embedding);
        if was_new_row {
            self.log_op(
                "record_with_rid",
                Some(rid),
                &serde_json::json!({
                    "rid": rid,
                    "type": memory_type,
                    "text": text,
                    "importance": importance,
                    "valence": valence,
                    "half_life": half_life,
                    "metadata": metadata,
                    "created_at_unix_micros": created_at_unix_micros,
                    "namespace": namespace,
                    "certainty": certainty,
                    "domain": domain,
                    "source": source,
                    "emotional_state": emotional_state,
                    "embedding_model": embedding_model,
                    "extracted_entities": extracted_entities,
                }),
                Some(&emb_hash),
            )?;
        }

        // **Phase 4.3 Commit C (saga task 19, 2026-05-08).** Enqueue the
        // entity / memory_entities / graph_index materialization for the
        // worker thread. Skip when there are no entities to apply — the
        // dispatch arm short-circuits the same way, but skipping avoids
        // a wasteful oplog row in the common no-entity case.
        //
        // Cluster determinism: the leader and each follower will both
        // enqueue + apply this op against their local state. Convergence
        // on entities + memory_entities is guaranteed by the same
        // INSERT OR IGNORE / ON CONFLICT idempotency the inline path
        // had. The convergence *time* differs by the materializer-lag
        // window (ms-scale), but the converged final state is identical.
        if !extracted_entities.is_empty() {
            let entities_json: Vec<&str> = extracted_entities.to_vec();
            let post_payload = serde_json::json!({
                "rid": rid,
                "namespace": namespace,
                "ts_secs": ts_secs,
                "extracted_entities": entities_json,
                "was_new_row": was_new_row,
            });
            self.log_op_pending(
                crate::engine::op_types::OP_MATERIALIZE_RECORD_WITH_RID_POST,
                Some(rid),
                &post_payload,
                None,
                None,
            )?;
        }

        Ok(())
    }

    /// **Issue #41 layer 3 — queued write path.** Called from `record()`
    /// when `write_router.try_enter_sync_writer()` returned None
    /// (router is in `Queueing` state during reembed cutover). The op
    /// is logged to `oplog` with `applied=0` and the v27 columns
    /// (`embedding_model = current_runtime_embedder_name`,
    /// `applied_generation = NULL`). The post-swap materializer drains
    /// these ops, re-encodes the text under the new embedder, and
    /// applies to the new generation's memories table + HNSW.
    ///
    /// Important invariants from brainstorm-2/3 enforced here:
    /// - DO NOT write to `memories` table (would mix old+new dim under
    ///   the rebuild snapshot)
    /// - DO NOT call `vec_index.append` (same reason)
    /// - DO NOT bump `visible_seq` (active generation doesn't yet
    ///   cover this seq; the post-swap materializer bumps it after
    ///   applying)
    /// - DO assign a `vec_seq` for the caller's RYW use
    ///   (`recall_with_seq(min_seq=N)` waits for the new generation to
    ///   advance past N)
    ///
    /// The pre-computed `embedding` argument is intentionally
    /// IGNORED. Per brainstorm-3 invariant 8 (queued payload
    /// correctness), the oplog stores logical text and the materializer
    /// re-encodes under the NEW embedder at replay time. Storing a
    /// pre-encoded old-embedder vector in oplog would race against
    /// post-swap replay and produce dim mismatch when the new HNSW is
    /// at a different dim.
    pub(crate) fn record_queued(
        &self,
        text: &str,
        memory_type: &str,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: &serde_json::Value,
        _embedding: &[f32],
        namespace: &str,
        certainty: f64,
        domain: &str,
        source: &str,
        emotional_state: Option<&str>,
    ) -> Result<String> {
        let rid = crate::id::new_id();
        let ts = now();

        // Assign a seq for caller's RYW use. Note we do NOT bump
        // visible_seq — the active generation doesn't yet cover this
        // op; the post-swap materializer is responsible for advancing
        // visible_seq as it drains queued ops.
        let _seq = self
            .vec_seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;

        // Capture the current runtime embedder name (the one active
        // before reembed flipped the router). The post-swap materializer
        // uses this to discriminate ops queued under the old embedder
        // (need re-encode) from ops produced by the new generation's
        // own writers (apply embedding bytes directly).
        let current_embedder_name = self.search_state.load().runtime_embedder_name.clone();

        // Full record payload — what the materializer needs to
        // reconstruct the row.
        let payload = serde_json::json!({
            "rid": rid,
            "type": memory_type,
            "text": text,
            "importance": importance,
            "valence": valence,
            "half_life": half_life,
            "metadata": metadata,
            "created_at": ts,
            "updated_at": ts,
            "namespace": namespace,
            "certainty": certainty,
            "domain": domain,
            "source": source,
            "emotional_state": emotional_state,
        });

        // Write to oplog with applied=0. The v27 `embedding_model`
        // column carries the OLD embedder name so the post-swap
        // materializer knows this needs re-encoding (vs being a
        // legacy pre-v27 op where embedding_model IS NULL and the
        // materializer trusts the embedding bytes as-is).
        self.log_op_pending_for_reembed_queue(
            "record",
            Some(&rid),
            &payload,
            current_embedder_name.as_deref(),
        )?;

        Ok(rid)
    }

    /// **Issue #41 layer 3 — variant of `log_op_pending` that populates
    /// the v27 `oplog.embedding_model` column.** Used by the queued
    /// write path during reembed; lets the post-swap materializer
    /// discriminate queued-during-reembed ops (which need re-encoding
    /// under the new embedder) from legacy pre-v27 ops (which have
    /// NULL `embedding_model` and trust their stored embedding bytes).
    pub(crate) fn log_op_pending_for_reembed_queue(
        &self,
        op_type: &str,
        target_rid: Option<&str>,
        payload: &serde_json::Value,
        embedding_model: Option<&str>,
    ) -> Result<String> {
        use rusqlite::params;
        use std::sync::atomic::Ordering;

        let op_id = crate::id::new_id();
        let hlc_ts = self.tick_hlc();
        let hlc_bytes = hlc_ts.to_bytes().to_vec();
        let payload_str = serde_json::to_string(payload)?;

        // Backpressure check (mirrors log_op_pending's contract).
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
              actor_id, hlc, embedding_hash, origin_actor, applied, \
              embedding, embedding_model, applied_generation) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 0, NULL, ?10, NULL)",
            params![
                op_id,
                op_type,
                now(),
                target_rid,
                payload_str,
                self.actor_id,
                hlc_bytes,
                None::<Vec<u8>>,
                self.actor_id,
                embedding_model,
            ],
        )?;
        if conn.changes() > 0 {
            self.pending_op_count.fetch_add(1, Ordering::Relaxed);
        }
        Ok(op_id)
    }
}
