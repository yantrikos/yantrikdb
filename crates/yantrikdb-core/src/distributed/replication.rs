//! Replication: oplog extraction, CRDT materialization, and operation application.
//!
//! Core principle: replicate operations, not index structures.
//! - Memories: Add-Wins Set (UUIDv7 uniqueness, INSERT OR IGNORE)
//! - Edges: LWW on (src, dst, rel_type), higher HLC wins
//! - Entities: Derived state, recomputed from edges
//! - Forget: Tombstone always wins (irreversible)
//! - Consolidation: Set-union via consolidation_members table

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::engine::YantrikDB;
use crate::error::Result;

/// Unseal a stored oplog payload given a provider (0.13.2). Mirrors
/// `YantrikDB::decode_oplog_payload` for the free-function readers that
/// hold a `Connection` rather than an engine.
pub(crate) fn decode_oplog_payload_with(
    enc: Option<&crate::encryption::EncryptionProvider>,
    stored: &str,
) -> Result<String> {
    match stored.strip_prefix(YantrikDB::OPLOG_ENC_PREFIX) {
        Some(b64) => match enc {
            Some(e) => e.decrypt_string(b64),
            None => Err(crate::error::YantrikDbError::Encryption(
                "oplog payload is encrypted but no key was provided".into(),
            )),
        },
        None => Ok(stored.to_string()),
    }
}
use crate::hlc::HLCTimestamp;
use crate::types::{ScoringRow, SynthesisAdmission};

/// An oplog entry for replication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OplogEntry {
    pub op_id: String,
    pub op_type: String,
    pub timestamp: f64,
    pub target_rid: Option<String>,
    pub payload: serde_json::Value,
    pub actor_id: String,
    pub hlc: Vec<u8>,
    pub embedding_hash: Option<Vec<u8>>,
    pub origin_actor: String,
    /// v0.10 Item 3: exact embedding bytes for a re-embedding `correct`
    /// op, so a follower applies the same vector rather than re-embedding
    /// (which diverges by model version/quantization). Present only on
    /// text-changing corrections; `None` for every other op. Encrypted
    /// under the origin DEK — usable on a same-DEK follower, which is the
    /// same assumption the whole encrypted-replication path already makes
    /// for text/metadata.
    #[serde(default)]
    pub embedding: Option<Vec<u8>>,
}

/// Result of a sync operation.
#[derive(Debug, Clone)]
pub struct SyncStats {
    pub ops_applied: usize,
    pub ops_skipped: usize,
}

/// Extract ops from the oplog since a given cursor.
///
/// Cursor semantics:
/// - **(Some(hlc), Some(op_id))** — compound cursor for exact resumption:
///   `hlc > since_hlc OR (hlc = since_hlc AND op_id > since_op_id)`.
///   Use this when you saved both fields from the last op you processed.
/// - **(Some(hlc), None)** — strict HLC boundary: `hlc > since_hlc`. Returns
///   ops with strictly greater HLC, dropping any op_id ties at the boundary.
/// - **(None, Some(op_id))** — resume from a known op by id. Looks up the
///   op's hlc, then applies the same compound cursor as above so iteration
///   is loss-free even when op_ids tie at the same HLC.
/// - **(None, None)** — no cursor; return everything from the start.
///
/// Fixes #13: previously, only the (Some, Some) arm filtered. Single-watermark
/// callers silently received the entire oplog because the `_` arm built SQL
/// with no boundary clause.
pub fn extract_ops_since(
    conn: &Connection,
    since_hlc: Option<&[u8]>,
    since_op_id: Option<&str>,
    exclude_actor: Option<&str>,
    limit: usize,
) -> Result<Vec<OplogEntry>> {
    extract_ops_since_enc(conn, None, since_hlc, since_op_id, exclude_actor, limit)
}

/// 0.13.2 — `extract_ops_since` for encrypted databases.
///
/// Oplog payloads are sealed at rest when the database has a key, so a
/// reader must present the provider to get JSON back. Callers on
/// plaintext databases pass `None` and get byte-identical behavior to
/// before. Replication peers of an encrypted database share the DEK,
/// so the shipped payload is plaintext JSON on the wire exactly as it
/// was — the seal is an at-rest property, not a transport change.
pub fn extract_ops_since_enc(
    conn: &Connection,
    enc: Option<&crate::encryption::EncryptionProvider>,
    since_hlc: Option<&[u8]>,
    since_op_id: Option<&str>,
    exclude_actor: Option<&str>,
    limit: usize,
) -> Result<Vec<OplogEntry>> {
    // Exclude engine-internal materialization op_types — those exist
    // only to deflect work off the foreground request path on the local
    // node (Phase 4.3) and have no cross-node replication semantics.
    // Each node generates its own materialization queue from its own
    // user-data ops; replicating these would double-do work and was
    // never the cluster sync contract. The `materialize_` prefix is a
    // soft namespace for future siblings (saga task 3 follow-ons).
    let select_cols = "SELECT op_id, op_type, timestamp, target_rid, payload, \
                       actor_id, hlc, embedding_hash, origin_actor, embedding \
                       FROM oplog \
                       WHERE hlc IS NOT NULL \
                         AND op_type NOT LIKE 'materialize\\_%' ESCAPE '\\'";

    let (sql, param_values) = match (since_hlc, since_op_id) {
        (Some(hlc), Some(op_id)) => {
            // Exact compound cursor: skip ops at-or-before (hlc, op_id).
            let mut sql = format!(
                "{select_cols} \
                 AND ((hlc > ?1) OR (hlc = ?1 AND op_id > ?2))"
            );
            let mut params: Vec<Box<dyn rusqlite::types::ToSql>> =
                vec![Box::new(hlc.to_vec()), Box::new(op_id.to_string())];

            if let Some(actor) = exclude_actor {
                sql.push_str(" AND origin_actor != ?3");
                params.push(Box::new(actor.to_string()));
            }

            sql.push_str(" ORDER BY hlc, op_id");
            sql.push_str(&format!(" LIMIT {limit}"));
            (sql, params)
        }
        (Some(hlc), None) => {
            // HLC-only watermark: strictly greater. May skip op_id ties at
            // the boundary HLC; pass the matching op_id too if you need
            // exact dedup.
            let mut sql = format!("{select_cols} AND hlc > ?1");
            let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(hlc.to_vec())];

            if let Some(actor) = exclude_actor {
                sql.push_str(" AND origin_actor != ?2");
                params.push(Box::new(actor.to_string()));
            }

            sql.push_str(" ORDER BY hlc, op_id");
            sql.push_str(&format!(" LIMIT {limit}"));
            (sql, params)
        }
        (None, Some(op_id)) => {
            // op_id-only watermark: look up the op's hlc inline, then apply
            // the compound cursor so we don't lose op_id-ties at the same
            // HLC. Subquery is constant — SQLite plans it once.
            let mut sql = format!(
                "{select_cols} \
                 AND ((hlc > (SELECT hlc FROM oplog WHERE op_id = ?1)) \
                   OR (hlc = (SELECT hlc FROM oplog WHERE op_id = ?1) \
                       AND op_id > ?1))"
            );
            let mut params: Vec<Box<dyn rusqlite::types::ToSql>> =
                vec![Box::new(op_id.to_string())];

            if let Some(actor) = exclude_actor {
                sql.push_str(" AND origin_actor != ?2");
                params.push(Box::new(actor.to_string()));
            }

            sql.push_str(" ORDER BY hlc, op_id");
            sql.push_str(&format!(" LIMIT {limit}"));
            (sql, params)
        }
        (None, None) => {
            // No cursor — full scan from the start of the log.
            let mut sql = String::from(select_cols);
            let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![];

            if let Some(actor) = exclude_actor {
                sql.push_str(" AND origin_actor != ?1");
                params.push(Box::new(actor.to_string()));
            }

            sql.push_str(" ORDER BY hlc, op_id");
            sql.push_str(&format!(" LIMIT {limit}"));
            (sql, params)
        }
    };

    let params_ref: Vec<&dyn rusqlite::types::ToSql> =
        param_values.iter().map(|p| p.as_ref()).collect();

    let mut stmt = conn.prepare(&sql)?;
    let entries = stmt
        .query_map(params_ref.as_slice(), |row| {
            let payload_str: String = row.get("payload")?;
            // 0.13.2: oplog payloads are sealed on encrypted databases.
            // Unsealing here matters for CORRECTNESS as much as for
            // readability — the parse below falls back to `{}`, so a
            // sealed row read raw would replicate an EMPTY payload and
            // lose the record silently. `enc` is None on plaintext
            // databases and the row passes through unchanged.
            let payload_str = match enc {
                Some(e) => decode_oplog_payload_with(Some(e), &payload_str)
                    .unwrap_or_else(|_| payload_str.clone()),
                None => payload_str,
            };
            let payload: serde_json::Value =
                serde_json::from_str(&payload_str).unwrap_or(serde_json::json!({}));

            Ok(OplogEntry {
                op_id: row.get("op_id")?,
                op_type: row.get("op_type")?,
                timestamp: row.get("timestamp")?,
                target_rid: row.get("target_rid")?,
                payload,
                actor_id: row.get("actor_id")?,
                hlc: row.get("hlc")?,
                embedding_hash: row.get("embedding_hash")?,
                origin_actor: row.get("origin_actor")?,
                embedding: row.get("embedding")?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    Ok(entries)
}

/// **Item 4a single-origin guard.** Protected write op types create or mutate
/// durable records carrying provenance (`source` / `kind` / `confidence_basis`).
/// In the single-writer Item 4a model these may originate at exactly ONE
/// authority; a foreign-origin protected write arriving via replication would
/// launder provenance past the local gate. Non-record ops (relate / link /
/// forget / trigger / …) are not provenance-bearing and are not guarded here.
fn is_protected_write(op_type: &str) -> bool {
    matches!(
        op_type,
        "record" | "record_with_rid" | "correct" | "consolidate"
    )
}

impl YantrikDB {
    /// **Item 4a single-origin guard.** The actor whose writes this database
    /// admits as authoritative, or `None` when the guard is inactive (legacy
    /// multi-origin behavior — the default). A single-writer deployment sets
    /// this to its own `actor_id` to reject foreign-origin provenance writes.
    ///
    /// **Fail-CLOSED (sol 4a.1 finding 1):** returns `Result` and distinguishes
    /// "no authority configured" (`Ok(None)`) from a real read failure (`Err`).
    /// A security guard must never be disabled by a malformed value or a query
    /// error, so `apply_ops` propagates the `Err` and applies nothing rather
    /// than silently falling open.
    pub fn authoritative_origin(&self) -> Result<Option<String>> {
        use rusqlite::OptionalExtension;
        Ok(self
            .conn()
            .query_row(
                "SELECT value FROM meta WHERE key = 'authoritative_origin_actor'",
                [],
                |r| r.get::<_, String>(0),
            )
            .optional()?)
    }

    /// Designate the authoritative origin actor (Item 4a). Pass this database's
    /// own `actor_id` on a single-writer deployment to activate the ingress
    /// guard so foreign-origin `record` / `record_with_rid` / `correct` /
    /// `consolidate` ops are rejected by [`apply_ops`].
    ///
    /// **Set this to the AUTHORITATIVE WRITER's actor id — not blindly to
    /// `self.actor_id()`.** On the writer itself those coincide, but a FOLLOWER
    /// must configure the *writer's* id (configuring its own would reject every
    /// op the writer sends). (sol 4a.4.)
    ///
    /// **Configuration is init/quiescent-time only (sol 4a.1 finding 2).** The
    /// authoritative origin is a deployment identity; set it before sync begins.
    /// Nothing sets it automatically — v37 does NOT seed it, because a fresh DB
    /// may legitimately be joining a multi-writer cluster. Changing it
    /// concurrently with an in-flight `apply_ops` is not linearizable (the
    /// preflight read and the apply are separate conn acquisitions) and is
    /// unsupported — a mid-flight change could let one batch straddle the
    /// old/new authority.
    pub fn set_authoritative_origin(&self, actor_id: &str) -> Result<()> {
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('authoritative_origin_actor', ?1)",
            params![actor_id],
        )?;
        Ok(())
    }
}

/// Apply remote ops to a local YantrikDB instance. Idempotent via INSERT OR IGNORE on op_id.
/// Returns the number of ops actually applied (newly inserted).
pub fn apply_ops(db: &YantrikDB, ops: &[OplogEntry]) -> Result<SyncStats> {
    // **Item 4a single-origin ingress guard.** If an authoritative origin is
    // configured, PREFLIGHT the whole batch before any HLC merge / oplog /
    // materialization: a single foreign-origin protected write rejects the
    // ENTIRE batch, leaving the engine byte-for-byte unchanged. Guard is
    // inactive (no-op) when no authority is set.
    if let Some(authority) = db.authoritative_origin()? {
        for op in ops {
            if is_protected_write(&op.op_type) && op.origin_actor != authority {
                return Err(crate::error::YantrikDbError::ForeignOriginRejected {
                    op_type: op.op_type.clone(),
                    origin_actor: op.origin_actor.clone(),
                    authority,
                });
            }
        }
    }

    let mut applied = 0;
    let mut skipped = 0;
    let mut has_relate_or_record = false;

    for op in ops {
        // Check if we already have this op (idempotent)
        let exists: bool = db.conn().query_row(
            "SELECT COUNT(*) > 0 FROM oplog WHERE op_id = ?1",
            params![op.op_id],
            |row| row.get(0),
        )?;

        if exists {
            skipped += 1;
            continue;
        }

        // Merge HLC
        if let Some(remote_ts) = HLCTimestamp::from_bytes(&op.hlc) {
            db.merge_hlc(remote_ts);
        }

        // Track if we need to backfill memory_entities after. Includes
        // "record_with_rid" (now materialized, same as "record") so its
        // entity join rows are backfilled too.
        if op.op_type == "relate" || op.op_type == "record" || op.op_type == "record_with_rid" {
            has_relate_or_record = true;
        }

        // Materialize the operation's side effects
        materialize_op(db, op)?;

        // Insert the op into our local oplog
        let payload_str = serde_json::to_string(&op.payload)?;
        db.conn().execute(
            "INSERT OR IGNORE INTO oplog \
             (op_id, op_type, timestamp, target_rid, payload, \
              actor_id, hlc, embedding_hash, origin_actor, applied, embedding) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 1, ?10)",
            params![
                op.op_id,
                op.op_type,
                op.timestamp,
                op.target_rid,
                payload_str,
                op.actor_id,
                op.hlc,
                op.embedding_hash,
                op.origin_actor,
                op.embedding,
            ],
        )?;

        applied += 1;
    }

    // Backfill memory_entities if any relate/record ops were applied.
    // This ensures the join table stays current after sync.
    if has_relate_or_record && applied > 0 {
        let _ = db.backfill_memory_entities();
    }

    Ok(SyncStats {
        ops_applied: applied,
        ops_skipped: skipped,
    })
}

/// Materialize a single op — replay its side effects on the local DB.
fn materialize_op(db: &YantrikDB, op: &OplogEntry) -> Result<()> {
    match op.op_type.as_str() {
        // "record_with_rid" (cluster/replica apply path) logs a full-payload
        // op whose op_type differs from "record"; without this arm it fell
        // through to the silent unknown-op branch, so records created via the
        // cluster path never peer-replicated (sol Item 4 design review). Its
        // payload is materialize_record-compatible (created_at handled in
        // materialize_record), and INSERT OR IGNORE keeps re-apply idempotent.
        "record" | "record_with_rid" => {
            let synthesis_changes = materialize_record(
                &*db.conn(),
                &op.payload,
                db.embedding_dim(),
                &op.origin_actor,
                &op.hlc,
            )?;
            // Update scoring cache with new record. created_at is carried as
            // `created_at` (record) or `created_at_unix_micros`
            // (record_with_rid) — accept either so the cache matches the
            // durable row.
            let created_at = op.payload["created_at"].as_f64().or_else(|| {
                op.payload["created_at_unix_micros"]
                    .as_f64()
                    .map(|micros| micros / 1_000_000.0)
            });
            let rid = op.payload["rid"].as_str().unwrap_or_default();
            if !rid.is_empty() {
                let synthesis_descriptor: Option<(Option<String>, Option<String>, Option<String>)> =
                    db.conn()
                        .query_row(
                            "SELECT synthesis_state, synthesis_axis, synthesis_granularity \
                         FROM memories WHERE rid = ?1",
                            params![rid],
                            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                        )
                        .optional()?;
                if let Some((synthesis_state, synthesis_axis, synthesis_granularity)) =
                    synthesis_descriptor
                {
                    db.cache_insert(
                        rid.to_string(),
                        ScoringRow {
                            created_at: created_at.unwrap_or(0.0),
                            importance: op.payload["importance"].as_f64().unwrap_or(0.5),
                            half_life: op.payload["half_life"].as_f64().unwrap_or(604800.0),
                            last_access: created_at.unwrap_or(0.0),
                            access_count: 0,
                            valence: op.payload["valence"].as_f64().unwrap_or(0.0),
                            consolidation_status: "active".to_string(),
                            synthesis_state,
                            synthesis_axis,
                            synthesis_granularity,
                            memory_type: op.payload["type"]
                                .as_str()
                                .unwrap_or("episodic")
                                .to_string(),
                            namespace: op.payload["namespace"]
                                .as_str()
                                .unwrap_or("default")
                                .to_string(),
                            certainty: op.payload["certainty"].as_f64().unwrap_or(0.8),
                            domain: op.payload["domain"]
                                .as_str()
                                .unwrap_or("general")
                                .to_string(),
                            source: op.payload["source"].as_str().unwrap_or("user").to_string(),
                            emotional_state: op.payload["emotional_state"]
                                .as_str()
                                .map(|s| s.to_string()),
                        },
                    );
                }
                db.cache_verify_syntheses(&synthesis_changes.verified);
                db.cache_supersede_syntheses(&synthesis_changes.superseded);
                db.cache_invalidate_syntheses(&synthesis_changes.invalidated);
            }
        }
        "relate" => {
            let changed = materialize_relate(&*db.conn(), &op.payload, &op.hlc)?;
            let src = op.payload["src"].as_str().unwrap_or_default();
            let dst = op.payload["dst"].as_str().unwrap_or_default();
            let rel_type = op.payload["rel_type"].as_str().unwrap_or_default();
            let weight = op.payload["weight"].as_f64().unwrap_or(1.0);
            if !src.is_empty() && !dst.is_empty() {
                // The SQL row and graph index are one logical projection. A
                // stale/equal HLC op that loses LWW must not overwrite the
                // causal winner in the in-memory index.
                if changed {
                    let mut gi = db.graph_index.write();
                    let (src_type, dst_type) =
                        crate::graph::classify_with_relationship(src, dst, rel_type);
                    gi.add_entity(src, src_type);
                    gi.add_entity(dst, dst_type);
                    gi.add_edge(src, dst, weight as f32);
                }
                // Conflict detection intentionally stays outside the `changed` gate.
                // An operation that loses LWW must not mutate the row or cache,
                // but a losing concurrent edge may still reveal a conflict.
                let _ = crate::conflict::detect_edge_conflicts(
                    db,
                    src,
                    dst,
                    rel_type,
                    op.target_rid.as_deref(),
                );
            }
        }
        "forget" => {
            let invalidated_syntheses = materialize_forget(&*db.conn(), &op.payload)?;
            // Remove from scoring cache + vec index + graph index
            let rid = op.payload["rid"].as_str().unwrap_or_default();
            if !rid.is_empty() {
                // 2026-08-17: a replicated tombstone is a delete, and a
                // delete applied to an index that is about to be discarded
                // resurrects the record — the same F1/F2 shape closed at
                // `tombstone_inner`, reached through the replication
                // applier instead of forget(). Guard before touching
                // SearchState so a cutover cannot land between the load and
                // the tombstone.
                //
                // Returning Err here is the RETRYABLE outcome by
                // construction: `apply_ops` calls `materialize_op` BEFORE
                // marking the oplog row applied, so a deferred op stays
                // pending and the next sync replays it. Guarded at the arm
                // rather than at `purge_chunks` itself, because
                // `tombstone_inner` already holds a guard when it calls
                // that helper and a nested acquisition would fail the
                // moment a cutover began mid-forget.
                let Some(_sync_guard) = db.write_router.try_enter_sync_writer() else {
                    return Err(crate::error::YantrikDbError::ForgetDeferredDuringReembed {
                        rid: rid.to_string(),
                    });
                };
                db.cache_remove(rid);
                let _seq = db
                    .vec_seq
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    + 1;
                // **Issue #41 brainstorm-4 §1.** Replication-applied
                // tombstones land on the active SearchState's
                // DeltaIndex.
                db.search_state.load().vec_index.tombstone(rid, _seq);
                // Chunked embeddings: window keys need their own markers
                // (exact-string matching), same as the leader's forget.
                db.purge_chunks(rid, _seq)?;
                db.graph_index.write().unlink_memory(rid);
                db.cache_invalidate_syntheses(&invalidated_syntheses);
            }
        }
        "consolidate" => {
            materialize_consolidate(&*db.conn(), &op.payload, &op.hlc, &op.origin_actor)?;
            // Cache: insert consolidated memory + mark sources
            let consolidated_rid = op.payload["consolidated_rid"].as_str().unwrap_or_default();
            let text = op.payload["text"].as_str().unwrap_or("");
            let typed_synthesis = op
                .payload
                .get("synthesis")
                .is_some_and(|value| !value.is_null());
            let additive = op.payload["additive"].as_bool().unwrap_or(false);
            if !typed_synthesis && !consolidated_rid.is_empty() && !text.is_empty() {
                let synthesis_descriptor: Option<(Option<String>, Option<String>, Option<String>)> =
                    db.conn()
                        .query_row(
                            "SELECT synthesis_state, synthesis_axis, synthesis_granularity \
                         FROM memories WHERE rid = ?1",
                            params![consolidated_rid],
                            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                        )
                        .optional()?;
                if let Some((synthesis_state, synthesis_axis, synthesis_granularity)) =
                    synthesis_descriptor
                {
                    db.cache_insert(
                        consolidated_rid.to_string(),
                        ScoringRow {
                            created_at: op.timestamp,
                            importance: op.payload["importance"].as_f64().unwrap_or(0.5),
                            half_life: op.payload["half_life"].as_f64().unwrap_or(604800.0),
                            last_access: op.timestamp,
                            access_count: 0,
                            valence: op.payload["valence"].as_f64().unwrap_or(0.0),
                            consolidation_status: "active".to_string(),
                            synthesis_state,
                            synthesis_axis,
                            synthesis_granularity,
                            memory_type: "semantic".to_string(),
                            namespace: op.payload["namespace"]
                                .as_str()
                                .unwrap_or("default")
                                .to_string(),
                            certainty: 0.8,
                            domain: "general".to_string(),
                            source: "user".to_string(),
                            emotional_state: None,
                        },
                    );
                }
            }
            if !additive {
                if let Some(source_rids) = op.payload["source_rids"].as_array() {
                    for rid_val in source_rids {
                        if let Some(rid) = rid_val.as_str() {
                            db.cache_mark_consolidated(rid, 0.3);
                        }
                    }
                }
            }
        }
        "conflict_detect" => {
            materialize_conflict_detect(&*db.conn(), &op.payload, &op.hlc, &op.origin_actor)?
        }
        "conflict_resolve" => {
            materialize_conflict_resolve(&*db.conn(), &op.payload)?;
            // If keep_a or keep_b, remove the loser from cache + vec index
            let strategy = op.payload["strategy"].as_str().unwrap_or("");
            if strategy == "keep_a" || strategy == "keep_b" {
                if let Some(loser) = op.payload["loser_rid"].as_str() {
                    // Conflict-loser suppression is a delete too — same
                    // cutover hazard as the replicated forget above, and a
                    // resurrected loser is worse than a resurrected forget
                    // because the winner is still present: the store would
                    // serve both sides of a resolved conflict.
                    let Some(_sync_guard) = db.write_router.try_enter_sync_writer() else {
                        return Err(crate::error::YantrikDbError::ForgetDeferredDuringReembed {
                            rid: loser.to_string(),
                        });
                    };
                    db.cache_remove(loser);
                    let _seq = db
                        .vec_seq
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        + 1;
                    // **Issue #41 brainstorm-4 §1.** Active-generation
                    // SearchState tombstone.
                    db.search_state.load().vec_index.tombstone(loser, _seq);
                    // The loser's window keys go with it.
                    db.purge_chunks(loser, _seq)?;
                }
            }
        }
        "correct" => {
            // v0.10 Item 3 finding 5: RID-stable correction applied
            // coherently on the follower — SQL + reserve-append + publish +
            // cache under one conn-lock critical section, exact bytes when
            // the vector space matches, append failures PROPAGATED so the op
            // is retried rather than leaving SQL applied against a stale
            // index. Replaces the old materialize_correct + inline apply.
            db.apply_replicated_correct(&op.payload, op.embedding.as_deref(), &op.origin_actor)?;
        }
        "trigger_fire" => {
            materialize_trigger_fire(&*db.conn(), &op.payload, &op.hlc, &op.origin_actor)?
        }
        "trigger_deliver" | "trigger_ack" | "trigger_act" | "trigger_dismiss" => {
            materialize_trigger_lifecycle(&*db.conn(), &op.payload)?;
        }
        "pattern_upsert" => {
            materialize_pattern(&*db.conn(), &op.payload, &op.hlc, &op.origin_actor)?
        }
        // **Issue #48 — record-to-record links.**
        "link" => {
            materialize_link(&*db.conn(), &op.payload, &op.hlc, &op.origin_actor)?;
        }
        "unlink" => {
            materialize_unlink(&*db.conn(), &op.payload)?;
        }
        "reinforce" | "think" => {
            // Local-only ops; skip during replication
        }
        _ => {
            // Unknown op types are silently skipped (forward compatibility)
        }
    }

    Ok(())
}

/// Materialize a "record" op: INSERT OR IGNORE into memories.
#[derive(Default)]
struct SynthesisStateChanges {
    verified: Vec<String>,
    superseded: Vec<String>,
    invalidated: Vec<String>,
}

impl SynthesisStateChanges {
    fn merge(&mut self, mut other: Self) {
        self.verified.append(&mut other.verified);
        self.superseded.append(&mut other.superseded);
        self.invalidated.append(&mut other.invalidated);
    }
}

fn reverify_synthesis_dependents_in_tx(
    tx: &rusqlite::Transaction<'_>,
    source_rid: &str,
) -> Result<SynthesisStateChanges> {
    let mut frontier = std::collections::VecDeque::from([source_rid.to_string()]);
    let mut visited = std::collections::HashSet::new();
    let mut changes = SynthesisStateChanges::default();

    while let Some(changed_rid) = frontier.pop_front() {
        if !visited.insert(changed_rid.clone()) {
            continue;
        }
        let candidates: Vec<(String, String, String, String)> = {
            let mut stmt = tx.prepare(
                "SELECT DISTINCT m.rid, m.namespace, m.synthesis_evidence_version, \
                                 m.synthesis_logical_key \
                 FROM memories m \
                 JOIN synthesis_dependencies d ON d.synthesis_rid = m.rid \
                 WHERE d.source_rid = ?1 AND m.synthesis_state = 'unverified' \
                 ORDER BY m.rid",
            )?;
            let rows = stmt.query_map(params![changed_rid], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?;
            rows.collect::<std::result::Result<_, _>>()?
        };

        for (synthesis_rid, namespace, evidence_version, logical_key) in candidates {
            let dependencies: Vec<(String, i64, bool)> = {
                let mut stmt = tx.prepare(
                    "SELECT source_rid, source_revision_num, is_direct \
                     FROM synthesis_dependencies WHERE synthesis_rid = ?1 \
                     ORDER BY source_rid",
                )?;
                let rows = stmt.query_map(params![synthesis_rid], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get::<_, i64>(2)? != 0))
                })?;
                rows.collect::<std::result::Result<_, _>>()?
            };
            if dependencies.is_empty() || !dependencies.iter().any(|dependency| dependency.2) {
                continue;
            }

            let mut hasher = blake3::Hasher::new();
            hasher.update(b"yantrikdb:synthesis-evidence:v1\0");
            let mut sources_match = true;
            let mut has_leaf_dependency = false;
            for (dependency_rid, expected_revision, _) in &dependencies {
                hasher.update(dependency_rid.as_bytes());
                hasher.update(b"\0");
                hasher.update(&expected_revision.to_le_bytes());
                let current: Option<(String, String, Option<String>, i64)> = tx
                    .query_row(
                        "SELECT m.namespace, m.consolidation_status, m.synthesis_state, \
                                COALESCE((SELECT MAX(r.revision_num) FROM record_revisions r \
                                          WHERE r.rid = m.rid), 0) \
                         FROM memories m WHERE m.rid = ?1",
                        params![dependency_rid],
                        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                    )
                    .optional()?;
                has_leaf_dependency |= current
                    .as_ref()
                    .is_some_and(|(_, _, state, _)| state.is_none());
                if !matches!(
                    current,
                    Some((ref source_namespace, ref status, ref state, revision_num))
                        if source_namespace == &namespace
                            && status == "active"
                            && state.as_deref().is_none_or(|value| value == "verified")
                            && revision_num == *expected_revision
                ) {
                    sources_match = false;
                    break;
                }
            }
            if !sources_match
                || !has_leaf_dependency
                || hasher.finalize().to_hex().as_str() != evidence_version
            {
                continue;
            }
            if tx.execute(
                "UPDATE memories SET synthesis_state = 'verified' \
                 WHERE rid = ?1 AND synthesis_state = 'unverified'",
                params![synthesis_rid],
            )? > 0
            {
                let (winner, superseded) =
                    YantrikDB::refold_synthesis_generations_in_tx(tx, &namespace, &logical_key)?;
                for previous_rid in &superseded {
                    changes
                        .invalidated
                        .extend(YantrikDB::invalidate_synthesis_dependents_in_tx(
                            tx,
                            previous_rid,
                        )?);
                }
                changes.superseded.extend(superseded);
                if winner.as_deref() == Some(synthesis_rid.as_str()) {
                    frontier.push_back(synthesis_rid.clone());
                    changes.verified.push(synthesis_rid);
                }
            }
        }
    }
    Ok(changes)
}

fn materialize_record(
    conn: &Connection,
    payload: &serde_json::Value,
    _embedding_dim: usize,
    source_actor: &str,
    generation_hlc: &[u8],
) -> Result<SynthesisStateChanges> {
    let rid = payload["rid"].as_str().unwrap_or_default();
    let mem_type = payload["type"].as_str().unwrap_or("episodic");
    let text = payload["text"].as_str().unwrap_or("");
    let importance = payload["importance"].as_f64().unwrap_or(0.5);
    let valence = payload["valence"].as_f64().unwrap_or(0.0);
    let half_life = payload["half_life"].as_f64().unwrap_or(604800.0);
    // The "record" op carries `created_at` (secs); the "record_with_rid" op
    // carries `created_at_unix_micros`. Accept either so both op types
    // materialize with the correct timestamp instead of falling to epoch 0.
    let created_at = payload["created_at"]
        .as_f64()
        .or_else(|| {
            payload["created_at_unix_micros"]
                .as_f64()
                .map(|micros| micros / 1_000_000.0)
        })
        .unwrap_or(0.0);
    let updated_at = payload["updated_at"].as_f64().unwrap_or(created_at);
    let metadata = payload
        .get("metadata")
        .map(|m| serde_json::to_string(m).unwrap_or_else(|_| "{}".to_string()))
        .unwrap_or_else(|| "{}".to_string());
    // v48 (#149): event-time columns from the SAME payload value the
    // metadata column text was serialized from, so column and JSON agree.
    let (event_time_min, event_time_max) = payload
        .get("metadata")
        .map(crate::base::datetext::event_time_bounds)
        .unwrap_or((None, None));

    if rid.is_empty() {
        return Ok(SynthesisStateChanges::default()); // Can't materialize without a rid
    }

    let namespace = payload["namespace"].as_str().unwrap_or("default");
    // **Replication provenance-integrity fix (sol Item 4 design review,
    // 2026-07-14).** These four fields were previously dropped here, so a
    // replicated record's DURABLE row fell to the schema defaults —
    // critically `source='user'` — even when the origin recorded
    // `source='inference'`. Meanwhile the scoring-cache insert below reads
    // them from the payload, so the durable row and the cache disagreed and
    // `get()` returned a laundered `source='user'`. The oplog "record"
    // payload carries all four (engine/record.rs log_op), so read them here
    // with the SAME defaults the cache uses and persist them verbatim. This
    // is the T06 anti-laundering contract at the replication boundary.
    let certainty = payload["certainty"].as_f64().unwrap_or(0.8);
    let domain = payload["domain"].as_str().unwrap_or("general");
    let source = payload["source"].as_str().unwrap_or("user");
    let emotional_state = payload["emotional_state"].as_str();
    // 4a.6c: the v37 idempotency columns, mirrored from the origin (same
    // extend-the-#69-pattern as source above). Absent/null on pre-4a.6c ops
    // and keyless writes -> NULL, identical to the old row shape.
    let idempotency_key = payload["idempotency_key"].as_str();
    let claim_origin_actor = payload["origin_actor"].as_str();

    let synthesis_present = payload
        .get("synthesis")
        .is_some_and(|value| !value.is_null());
    let synthesis = payload
        .get("synthesis")
        .filter(|value| !value.is_null())
        .and_then(|value| serde_json::from_value::<SynthesisAdmission>(value.clone()).ok());
    let synthesis_shape_valid = synthesis.as_ref().is_some_and(|descriptor| {
        !descriptor.axis.trim().is_empty()
            && matches!(descriptor.granularity.as_str(), "atomic" | "rollup")
            && !descriptor.logical_key.trim().is_empty()
            && !descriptor.evidence_version.trim().is_empty()
            && !descriptor.dependencies.is_empty()
            && descriptor
                .dependencies
                .iter()
                .any(|dependency| dependency.is_direct)
            && descriptor
                .dependencies
                .iter()
                .map(|dependency| dependency.source_rid.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len()
                == descriptor.dependencies.len()
    });
    let synthesis_verified = if synthesis_shape_valid {
        let descriptor = synthesis.as_ref().expect("shape-valid descriptor");
        let mut dependencies = descriptor.dependencies.clone();
        dependencies.sort_by(|a, b| a.source_rid.cmp(&b.source_rid));
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"yantrikdb:synthesis-evidence:v1\0");
        for dependency in &dependencies {
            hasher.update(dependency.source_rid.as_bytes());
            hasher.update(b"\0");
            hasher.update(&dependency.source_revision_num.to_le_bytes());
        }
        let version_matches = hasher.finalize().to_hex().as_str() == descriptor.evidence_version;
        let mut sources_match = true;
        let mut has_leaf_dependency = false;
        for dependency in &dependencies {
            let current: Option<(String, String, Option<String>, i64)> = conn
                .query_row(
                    "SELECT m.namespace, m.consolidation_status, m.synthesis_state, \
                            COALESCE((SELECT MAX(r.revision_num) FROM record_revisions r \
                                      WHERE r.rid = m.rid), 0) \
                     FROM memories m WHERE m.rid = ?1",
                    params![dependency.source_rid],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .optional()?;
            has_leaf_dependency |= current
                .as_ref()
                .is_some_and(|(_, _, state, _)| state.is_none());
            if !matches!(
                current,
                Some((ref source_namespace, ref status, ref state, revision_num))
                    if source_namespace == namespace
                        && status == "active"
                        && state.as_deref().is_none_or(|value| value == "verified")
                        && revision_num == dependency.source_revision_num
            ) {
                sources_match = false;
                break;
            }
        }
        version_matches && sources_match && has_leaf_dependency
    } else {
        false
    };
    let synthesis_state = synthesis_present.then_some(if synthesis_verified {
        "verified"
    } else {
        "unverified"
    });

    // Add-Wins Set: INSERT OR IGNORE means first writer wins (UUIDv7 = no collisions)
    let tx = conn.unchecked_transaction()?;
    let inserted = tx.execute(
        "INSERT OR IGNORE INTO memories \
         (rid, type, text, created_at, updated_at, importance, \
          half_life, last_access, valence, metadata, namespace, \
          certainty, domain, source, emotional_state, idempotency_key, origin_actor, \
          synthesis_axis, synthesis_granularity, synthesis_logical_key, \
          synthesis_evidence_version, synthesis_generation_hlc, synthesis_state, \
          event_time_min, event_time_max) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, \
                 ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25)",
        params![
            rid,
            mem_type,
            text,
            created_at,
            updated_at,
            importance,
            half_life,
            created_at,
            valence,
            metadata,
            namespace,
            certainty,
            domain,
            source,
            emotional_state,
            idempotency_key,
            claim_origin_actor,
            synthesis
                .as_ref()
                .filter(|_| synthesis_shape_valid)
                .map(|s| s.axis.as_str()),
            synthesis
                .as_ref()
                .filter(|_| synthesis_shape_valid)
                .map(|s| s.granularity.as_str()),
            synthesis
                .as_ref()
                .filter(|_| synthesis_shape_valid)
                .map(|s| s.logical_key.as_str()),
            synthesis
                .as_ref()
                .filter(|_| synthesis_shape_valid)
                .map(|s| s.evidence_version.as_str()),
            synthesis_shape_valid.then_some(generation_hlc),
            synthesis_state,
            // v48 (#149) event time.
            event_time_min,
            event_time_max,
        ],
    )?;

    if inserted > 0 && synthesis_shape_valid {
        for dependency in &synthesis
            .as_ref()
            .expect("shape-valid descriptor")
            .dependencies
        {
            tx.execute(
                "INSERT INTO synthesis_dependencies \
                 (synthesis_rid, source_rid, source_revision_num, namespace, is_direct) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    rid,
                    dependency.source_rid,
                    dependency.source_revision_num,
                    namespace,
                    i64::from(dependency.is_direct),
                ],
            )?;
        }
    }

    // **v0.7.19 replication audit (postmortem 2026-05-20).** Stamp a
    // row in replication_apply_log so audit queries can distinguish
    // "received via replication" from "true orphan". See
    // base/schema.rs::MIGRATE_V28_TO_V29 for the three-population
    // audit query shape.
    let applied_at = crate::time::now_secs();
    let _ = tx.execute(
        "INSERT OR IGNORE INTO replication_apply_log (rid, op_type, source_actor, applied_at) \
         VALUES (?1, 'record', ?2, ?3)",
        params![rid, source_actor, applied_at],
    );
    let mut synthesis_changes = SynthesisStateChanges::default();
    if inserted > 0 && synthesis_verified {
        let descriptor = synthesis.as_ref().expect("verified synthesis descriptor");
        let (winner, superseded) =
            YantrikDB::refold_synthesis_generations_in_tx(&tx, namespace, &descriptor.logical_key)?;
        for previous_rid in &superseded {
            synthesis_changes
                .invalidated
                .extend(YantrikDB::invalidate_synthesis_dependents_in_tx(
                    &tx,
                    previous_rid,
                )?);
        }
        synthesis_changes.superseded.extend(superseded);
        if winner.as_deref() == Some(rid) {
            synthesis_changes.verified.push(rid.to_string());
        }
    }
    synthesis_changes.merge(reverify_synthesis_dependents_in_tx(&tx, rid)?);
    tx.commit()?;

    // Note: we can't insert into the HNSW vec index without the actual embedding data.
    // The oplog only stores the embedding_hash. The rebuild_vec_index() function
    // can be used as fallback to rebuild the index from the memories table.

    Ok(synthesis_changes)
}

/// Materialize a "relate" op: LWW on (src, dst, rel_type), higher HLC wins.
///
/// **#148:** the comparison is the edge's HLC, not the payload's wall-clock
/// `created_at` — between nodes with clock skew, wall-clock LWW lets a stale
/// edge from a fast clock beat a newer edge from a slow clock (silent
/// resurrection of overwritten values, the exact anomaly HLC exists to
/// prevent). The edge HLC is the leader's `edge_hlc_hex` carried VERBATIM in
/// the payload (the record_links edge-identity pattern); legacy payloads
/// (pre-#148 leaders) fall back to the op envelope's HLC. A local row with no
/// HLC (pre-v47, or written by a non-replicating writer) is compared by
/// `created_at`, preserving the old behavior for exactly the rows the old
/// behavior wrote.
fn materialize_relate(conn: &Connection, payload: &serde_json::Value, hlc: &[u8]) -> Result<bool> {
    let edge_id = payload["edge_id"].as_str().unwrap_or_default();
    let src = payload["src"].as_str().unwrap_or_default();
    let dst = payload["dst"].as_str().unwrap_or_default();
    let rel_type = payload["rel_type"].as_str().unwrap_or_default();
    let weight = payload["weight"].as_f64().unwrap_or(1.0);
    let created_at = payload["created_at"].as_f64().unwrap_or(0.0);
    let edge_hlc: Vec<u8> = payload["edge_hlc_hex"]
        .as_str()
        .and_then(crate::serde_helpers::hex_decode)
        .unwrap_or_else(|| hlc.to_vec());

    if src.is_empty() || dst.is_empty() {
        return Ok(false);
    }

    // LWW: one WHERE predicate controls the complete row update and exposes
    // whether the op won to cache materialization. HLC bytes are big-endian,
    // so BLOB memcmp = causal order. Equal HLC deliberately does not update.
    let changed = conn.execute(
        "INSERT INTO claims (claim_id, src, dst, rel_type, weight, created_at, hlc) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7) \
         ON CONFLICT(src, dst, rel_type, extractor, polarity, namespace) DO UPDATE SET \
         weight = excluded.weight, \
         created_at = excluded.created_at, \
         claim_id = excluded.claim_id, \
         hlc = excluded.hlc \
         WHERE (claims.hlc IS NOT NULL AND excluded.hlc > claims.hlc) \
            OR (claims.hlc IS NULL AND excluded.created_at > claims.created_at)",
        params![edge_id, src, dst, rel_type, weight, created_at, edge_hlc],
    )?;

    // Ensure entities exist
    let ts = created_at;
    for entity in [src, dst] {
        conn.execute(
            "INSERT INTO entities (name, first_seen, last_seen) \
             VALUES (?1, ?2, ?3) \
             ON CONFLICT(name) DO UPDATE SET \
             last_seen = MAX(last_seen, ?3), \
             mention_count = mention_count + 1",
            params![entity, ts, ts],
        )?;
    }

    Ok(changed != 0)
}

/// Materialize a "forget" op: tombstone always wins.
fn materialize_forget(conn: &Connection, payload: &serde_json::Value) -> Result<Vec<String>> {
    let rid = payload["rid"].as_str().unwrap_or_default();
    let updated_at = payload["updated_at"]
        .as_f64()
        .or_else(|| {
            payload["updated_at_unix_micros"]
                .as_f64()
                .map(|micros| micros / 1_000_000.0)
        })
        .unwrap_or(0.0);

    if rid.is_empty() {
        return Ok(Vec::new());
    }

    let tx = conn.unchecked_transaction()?;
    // Tombstone always wins — even if the memory doesn't exist locally yet
    tx.execute(
        "UPDATE memories SET consolidation_status = 'tombstoned', updated_at = ?1 WHERE rid = ?2",
        params![updated_at, rid],
    )?;

    let invalidated_syntheses: Vec<String> = {
        let mut stmt = tx.prepare(
            "UPDATE memories SET synthesis_state = 'invalidated' \
             WHERE synthesis_state = 'verified' \
               AND rid IN ( \
                   SELECT synthesis_rid FROM synthesis_dependencies \
                   WHERE namespace = (SELECT namespace FROM memories WHERE rid = ?1) \
                     AND source_rid = ?1 \
               ) \
             RETURNING rid",
        )?;
        let rows = stmt.query_map(params![rid], |row| row.get(0))?;
        rows.collect::<std::result::Result<_, _>>()?
    };

    // **Issue #48.** Replay the link-status transition the leader applied
    // in tombstone_inner so followers' record_links stay in lockstep.
    tx.execute(
        "UPDATE record_links SET status = 'broken_source_forgotten' \
         WHERE source_rid = ?1 AND status = 'active'",
        params![rid],
    )?;
    tx.execute(
        "UPDATE record_links SET status = 'broken_target_forgotten' \
         WHERE target_rid = ?1 AND status = 'active'",
        params![rid],
    )?;
    tx.commit()?;

    // HNSW vec index removal is handled by the materialize_op dispatcher

    Ok(invalidated_syntheses)
}

/// Materialize a "consolidate" op: insert into consolidation_members (set-union).
fn materialize_consolidate(
    conn: &Connection,
    payload: &serde_json::Value,
    hlc: &[u8],
    actor_id: &str,
) -> Result<()> {
    let consolidated_rid = payload["consolidated_rid"].as_str().unwrap_or_default();
    let source_rids = payload["source_rids"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let additive = payload["additive"].as_bool().unwrap_or(false);

    if consolidated_rid.is_empty() || source_rids.is_empty() {
        return Ok(());
    }

    // Also materialize the consolidated memory itself if present in payload
    let text = payload["text"].as_str().unwrap_or("");
    if !text.is_empty()
        && !payload
            .get("synthesis")
            .is_some_and(|value| !value.is_null())
    {
        let importance = payload["importance"].as_f64().unwrap_or(0.5);
        let valence = payload["valence"].as_f64().unwrap_or(0.0);
        let half_life = payload["half_life"].as_f64().unwrap_or(604800.0);
        let metadata = payload
            .get("metadata")
            .map(|m| serde_json::to_string(m).unwrap_or_else(|_| "{}".to_string()))
            .unwrap_or_else(|| "{}".to_string());
        // v48 (#149): event-time columns from the SAME payload value the
        // metadata column text was serialized from.
        let (event_time_min, event_time_max) = payload
            .get("metadata")
            .map(crate::base::datetext::event_time_bounds)
            .unwrap_or((None, None));
        let ts = crate::time::now_secs();

        let namespace = payload["namespace"].as_str().unwrap_or("default");
        conn.execute(
            "INSERT OR IGNORE INTO memories \
             (rid, type, text, created_at, updated_at, importance, \
              half_life, last_access, valence, metadata, namespace, \
              event_time_min, event_time_max) \
             VALUES (?1, 'semantic', ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                consolidated_rid,
                text,
                ts,
                ts,
                importance,
                half_life,
                ts,
                valence,
                metadata,
                namespace,
                // v48 (#149) event time.
                event_time_min,
                event_time_max,
            ],
        )?;

        // **v0.7.19 replication audit (postmortem 2026-05-20).**
        let _ = conn.execute(
            "INSERT OR IGNORE INTO replication_apply_log (rid, op_type, source_actor, applied_at) \
             VALUES (?1, 'consolidate', ?2, ?3)",
            params![consolidated_rid, actor_id, ts],
        );
    }

    // Insert consolidation_members entries (set-union CRDT: INSERT OR IGNORE)
    for source_rid in &source_rids {
        conn.execute(
            "INSERT OR IGNORE INTO consolidation_members \
             (consolidation_rid, source_rid, hlc, actor_id) \
             VALUES (?1, ?2, ?3, ?4)",
            params![consolidated_rid, source_rid, hlc, actor_id],
        )?;

        // Mark source memories as consolidated (if they exist locally)
        if !additive {
            conn.execute(
                "UPDATE memories \
                 SET consolidation_status = 'consolidated', \
                     consolidated_into = ?1, \
                     importance = importance * 0.3 \
                 WHERE rid = ?2 AND consolidation_status = 'active'",
                params![consolidated_rid, source_rid],
            )?;
        }
    }

    Ok(())
}

// ── V2: Conflict materializers ──

/// Materialize a "conflict_detect" op: INSERT OR IGNORE into conflicts.
fn materialize_conflict_detect(
    conn: &Connection,
    payload: &serde_json::Value,
    hlc: &[u8],
    origin_actor: &str,
) -> Result<()> {
    let conflict_id = payload["conflict_id"].as_str().unwrap_or_default();
    if conflict_id.is_empty() {
        return Ok(());
    }

    conn.execute(
        "INSERT OR IGNORE INTO conflicts
         (conflict_id, conflict_type, priority, status, memory_a, memory_b,
          entity, rel_type, detected_at, detected_by, detection_reason,
          hlc, origin_actor)
         VALUES (?1, ?2, ?3, 'open', ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            conflict_id,
            payload["conflict_type"].as_str().unwrap_or("minor"),
            payload["priority"].as_str().unwrap_or("medium"),
            payload["memory_a"].as_str().unwrap_or_default(),
            payload["memory_b"].as_str().unwrap_or_default(),
            payload["entity"].as_str(),
            payload["rel_type"].as_str(),
            payload["detected_at"].as_f64().unwrap_or(0.0),
            payload["detected_by"].as_str().unwrap_or_default(),
            payload["detection_reason"].as_str().unwrap_or_default(),
            hlc,
            origin_actor,
        ],
    )?;
    Ok(())
}

/// Materialize a "conflict_resolve" op: update the conflict record.
fn materialize_conflict_resolve(conn: &Connection, payload: &serde_json::Value) -> Result<()> {
    let conflict_id = payload["conflict_id"].as_str().unwrap_or_default();
    if conflict_id.is_empty() {
        return Ok(());
    }

    let status = if payload["dismissed"].as_bool().unwrap_or(false) {
        "dismissed"
    } else {
        "resolved"
    };

    conn.execute(
        "UPDATE conflicts SET
         status = ?1,
         resolved_at = ?2,
         resolved_by = ?3,
         strategy = ?4,
         winner_rid = ?5,
         resolution_note = ?6
         WHERE conflict_id = ?7 AND status = 'open'",
        params![
            status,
            payload["resolved_at"].as_f64().unwrap_or(0.0),
            payload["resolved_by"].as_str().unwrap_or_default(),
            payload["strategy"].as_str().unwrap_or_default(),
            payload["winner_rid"].as_str(),
            payload["resolution_note"].as_str(),
            conflict_id,
        ],
    )?;

    // If strategy is keep_a or keep_b, tombstone the loser
    let strategy = payload["strategy"].as_str().unwrap_or("");
    let loser_rid = payload["loser_rid"].as_str();
    if strategy == "keep_a" || strategy == "keep_b" {
        if let Some(loser) = loser_rid {
            let ts = payload["resolved_at"].as_f64().unwrap_or(0.0);
            conn.execute(
                "UPDATE memories SET consolidation_status = 'tombstoned', updated_at = ?1
                 WHERE rid = ?2 AND consolidation_status = 'active'",
                params![ts, loser],
            )?;
            // HNSW vec index removal is handled by the materialize_op dispatcher
        }
    }

    Ok(())
}

/// Materialize a "link" op (Issue #48) on a replica. Idempotent via the
/// UNIQUE(source_rid, target_rid, link_type) constraint + INSERT OR
/// IGNORE — re-applying the same link op across re-syncs is a no-op.
///
/// **v0.10 Phase 0 (deterministic projection):**
/// - The leader's canonical edge identity (`edge_id`, `edge_hlc_hex` in
///   the payload since Phase 0) is persisted VERBATIM, so every replica's
///   row sorts identically under the `max(hlc, id)` total order. Legacy
///   payloads (pre-Phase-0 leaders) fall back to the op envelope's HLC
///   plus a minted id.
/// - Supersedes edges are durably accepted as CANDIDATES and the selected
///   projection is then recomputed by the canonical descending-total-key
///   fold — the result is independent of arrival order, and a losing
///   concurrent edge is retained as `rejected_conflict` (never discarded,
///   never re-typed).
/// - A multi-candidate fold surfaces a `supersede_merge` structural
///   conflict row with a DERIVED deterministic conflict_id (no follower-
///   minted randomness, no oplog echo) — excluded from auto-resolution;
///   Item 1 derives `disputed_with` from the open row.
fn materialize_link(
    conn: &Connection,
    payload: &serde_json::Value,
    hlc: &[u8],
    source_actor: &str,
) -> Result<()> {
    let source_rid = payload["source_rid"].as_str().unwrap_or_default();
    let target_rid = payload["target_rid"].as_str().unwrap_or_default();
    let link_type = payload["link_type"].as_str().unwrap_or_default();
    if source_rid.is_empty() || target_rid.is_empty() || link_type.is_empty() {
        return Ok(());
    }
    let created_at = payload["created_at"]
        .as_f64()
        .unwrap_or_else(crate::time::now_secs);
    // Canonical identity: prefer the leader's carried values.
    let link_id = payload["edge_id"]
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(crate::id::new_id);
    let edge_hlc: Vec<u8> = payload["edge_hlc_hex"]
        .as_str()
        .and_then(crate::serde_helpers::hex_decode)
        .unwrap_or_else(|| hlc.to_vec());

    let is_supersedes = link_type == "supersedes";
    // Supersedes candidates enter unselected; the fold below decides.
    let initial_state = if is_supersedes {
        "rejected_conflict"
    } else {
        "selected"
    };

    conn.execute(
        "INSERT OR IGNORE INTO record_links \
         (link_id, source_rid, target_rid, link_type, status, selection_state, \
          created_at, hlc, origin_actor) \
         VALUES (?1, ?2, ?3, ?4, 'active', ?5, ?6, ?7, ?8)",
        params![
            link_id,
            source_rid,
            target_rid,
            link_type,
            initial_state,
            created_at,
            edge_hlc,
            source_actor
        ],
    )?;

    if is_supersedes {
        let fold = crate::engine::YantrikDB::refold_supersedes_target(conn, target_rid)?;
        if let (Some((winner_edge, winner_src)), false) = (&fold.winner, fold.losers.is_empty()) {
            // Deterministic structural conflict: same id derived on every
            // replica from the contested predecessor + sorted edge ids.
            let mut edge_ids: Vec<&str> = fold
                .losers
                .iter()
                .map(|(e, _)| e.as_str())
                .chain(std::iter::once(winner_edge.as_str()))
                .collect();
            edge_ids.sort_unstable();
            let conflict_id = format!("supersede_merge:{}:{}", target_rid, edge_ids.join("+"));
            let loser_src = &fold.losers[0].1;
            let reason = serde_json::json!({
                "kind": "supersede_merge",
                "predecessor": target_rid,
                "edges": edge_ids,
                "selected_edge": winner_edge,
            })
            .to_string();
            let _ = conn.execute(
                "INSERT OR IGNORE INTO conflicts \
                 (conflict_id, conflict_type, priority, status, memory_a, memory_b, \
                  detected_at, detected_by, detection_reason) \
                 VALUES (?1, 'supersede_merge', 'high', 'open', ?2, ?3, ?4, 'structural', ?5)",
                params![conflict_id, winner_src, loser_src, created_at, reason],
            );
        }
    }

    let applied_at = crate::time::now_secs();
    let _ = conn.execute(
        "INSERT OR IGNORE INTO replication_apply_log (rid, op_type, source_actor, applied_at) \
         VALUES (?1, 'link', ?2, ?3)",
        params![source_rid, source_actor, applied_at],
    );

    Ok(())
}

/// Materialize an "unlink" op (Issue #48) on a replica. A user retraction.
///
/// **v0.10 Phase 0:** supersedes edges are retracted (replayable
/// `selection_state='retracted'`) and the target's projection re-folded —
/// hard-deleting them made concurrent link/unlink arrival-order-dependent
/// across replicas. Other link types keep hard-delete semantics.
fn materialize_unlink(conn: &Connection, payload: &serde_json::Value) -> Result<()> {
    let source_rid = payload["source_rid"].as_str().unwrap_or_default();
    let target_rid = payload["target_rid"].as_str().unwrap_or_default();
    let link_type = payload["link_type"].as_str().unwrap_or_default();
    if source_rid.is_empty() || target_rid.is_empty() || link_type.is_empty() {
        return Ok(());
    }
    if link_type == "supersedes" {
        let n = conn.execute(
            "UPDATE record_links SET selection_state = 'retracted' \
             WHERE source_rid = ?1 AND target_rid = ?2 AND link_type = ?3 \
             AND selection_state != 'retracted'",
            params![source_rid, target_rid, link_type],
        )?;
        if n > 0 {
            crate::engine::YantrikDB::refold_supersedes_target(conn, target_rid)?;
        }
    } else {
        conn.execute(
            "DELETE FROM record_links \
             WHERE source_rid = ?1 AND target_rid = ?2 AND link_type = ?3",
            params![source_rid, target_rid, link_type],
        )?;
    }
    Ok(())
}

// ── Watermark tracking for delta sync ──

/// Get the watermark for a specific peer (last synced HLC + op_id).
pub fn get_peer_watermark(
    conn: &Connection,
    peer_actor: &str,
) -> Result<Option<(Vec<u8>, String)>> {
    match conn.query_row(
        "SELECT last_synced_hlc, last_synced_op_id FROM sync_peers WHERE peer_actor = ?1",
        params![peer_actor],
        |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, String>(1)?)),
    ) {
        Ok(wm) => Ok(Some(wm)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Update the watermark for a specific peer.
pub fn set_peer_watermark(
    conn: &Connection,
    peer_actor: &str,
    hlc: &[u8],
    op_id: &str,
) -> Result<()> {
    let ts = crate::time::now_secs();

    conn.execute(
        "INSERT INTO sync_peers (peer_actor, last_synced_hlc, last_synced_op_id, last_sync_time) \
         VALUES (?1, ?2, ?3, ?4) \
         ON CONFLICT(peer_actor) DO UPDATE SET \
         last_synced_hlc = ?2, last_synced_op_id = ?3, last_sync_time = ?4",
        params![peer_actor, hlc, op_id, ts],
    )?;

    Ok(())
}

/// Rebuild the vector index from memories table (disaster recovery).
/// Delegates to YantrikDB::rebuild_vec_index which builds a new HnswIndex.
pub fn rebuild_vec_index(db: &YantrikDB) -> Result<usize> {
    db.rebuild_vec_index()
}

// ── V3 materializers: triggers and patterns ──

/// Materialize a "trigger_fire" op: INSERT OR IGNORE into trigger_log.
fn materialize_trigger_fire(
    conn: &Connection,
    payload: &serde_json::Value,
    hlc: &[u8],
    origin_actor: &str,
) -> Result<()> {
    let trigger_id = payload["trigger_id"].as_str().unwrap_or_default();
    if trigger_id.is_empty() {
        return Ok(());
    }

    conn.execute(
        "INSERT OR IGNORE INTO trigger_log \
         (trigger_id, trigger_type, urgency, status, reason, suggested_action, \
          source_rids, context, created_at, expires_at, cooldown_key, hlc, origin_actor) \
         VALUES (?1, ?2, ?3, 'pending', ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            trigger_id,
            payload["trigger_type"].as_str().unwrap_or(""),
            payload["urgency"].as_f64().unwrap_or(0.0),
            payload["reason"].as_str().unwrap_or(""),
            payload["suggested_action"].as_str().unwrap_or(""),
            payload
                .get("source_rids")
                .map(|v| v.to_string())
                .unwrap_or("[]".to_string()),
            payload
                .get("context")
                .map(|v| v.to_string())
                .unwrap_or("{}".to_string()),
            payload["created_at"].as_f64().unwrap_or(0.0),
            payload["expires_at"].as_f64(),
            payload["cooldown_key"].as_str().unwrap_or(""),
            hlc,
            origin_actor,
        ],
    )?;

    // Dual-write to join table
    if let Some(rids) = payload.get("source_rids").and_then(|v| v.as_array()) {
        for rid_val in rids {
            if let Some(rid) = rid_val.as_str() {
                conn.execute(
                    "INSERT OR IGNORE INTO trigger_source_rids (trigger_id, rid) VALUES (?1, ?2)",
                    params![trigger_id, rid],
                )?;
            }
        }
    }

    Ok(())
}

/// Materialize a trigger lifecycle transition (deliver/ack/act/dismiss).
fn materialize_trigger_lifecycle(conn: &Connection, payload: &serde_json::Value) -> Result<()> {
    let trigger_id = payload["trigger_id"].as_str().unwrap_or_default();
    if trigger_id.is_empty() {
        return Ok(());
    }

    // Determine which status to set based on the payload keys
    if let Some(ts) = payload["dismissed_at"].as_f64() {
        conn.execute(
            "UPDATE trigger_log SET status = 'dismissed', acted_at = ?1 \
             WHERE trigger_id = ?2 AND status IN ('pending', 'delivered', 'acknowledged')",
            params![ts, trigger_id],
        )?;
    } else if let Some(ts) = payload["acted_at"].as_f64() {
        conn.execute(
            "UPDATE trigger_log SET status = 'acted', acted_at = ?1 \
             WHERE trigger_id = ?2 AND status IN ('delivered', 'acknowledged')",
            params![ts, trigger_id],
        )?;
    } else if let Some(ts) = payload["acknowledged_at"].as_f64() {
        conn.execute(
            "UPDATE trigger_log SET status = 'acknowledged', acknowledged_at = ?1 \
             WHERE trigger_id = ?2 AND status = 'delivered'",
            params![ts, trigger_id],
        )?;
    } else if let Some(ts) = payload["delivered_at"].as_f64() {
        conn.execute(
            "UPDATE trigger_log SET status = 'delivered', delivered_at = ?1 \
             WHERE trigger_id = ?2 AND status = 'pending'",
            params![ts, trigger_id],
        )?;
    }

    Ok(())
}

/// Materialize a "pattern_upsert" op: convergent merge into patterns table.
fn materialize_pattern(
    conn: &Connection,
    payload: &serde_json::Value,
    hlc: &[u8],
    origin_actor: &str,
) -> Result<()> {
    let pattern_id = payload["pattern_id"].as_str().unwrap_or_default();
    if pattern_id.is_empty() {
        return Ok(());
    }

    conn.execute(
        "INSERT INTO patterns \
         (pattern_id, pattern_type, status, confidence, description, \
          evidence_rids, entity_names, context, first_seen, last_confirmed, \
          occurrence_count, hlc, origin_actor) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13) \
         ON CONFLICT(pattern_id) DO UPDATE SET \
         confidence = MAX(confidence, excluded.confidence), \
         last_confirmed = MAX(last_confirmed, excluded.last_confirmed), \
         occurrence_count = MAX(occurrence_count, excluded.occurrence_count), \
         status = CASE WHEN excluded.last_confirmed > last_confirmed \
                  THEN excluded.status ELSE status END",
        params![
            pattern_id,
            payload["pattern_type"].as_str().unwrap_or(""),
            payload["status"].as_str().unwrap_or("active"),
            payload["confidence"].as_f64().unwrap_or(0.0),
            payload["description"].as_str().unwrap_or(""),
            payload
                .get("evidence_rids")
                .map(|v| v.to_string())
                .unwrap_or("[]".to_string()),
            payload
                .get("entity_names")
                .map(|v| v.to_string())
                .unwrap_or("[]".to_string()),
            payload
                .get("context")
                .map(|v| v.to_string())
                .unwrap_or("{}".to_string()),
            payload["first_seen"].as_f64().unwrap_or(0.0),
            payload["last_confirmed"].as_f64().unwrap_or(0.0),
            payload["occurrence_count"].as_i64().unwrap_or(1),
            hlc,
            origin_actor,
        ],
    )?;

    // Dual-write to join tables
    if let Some(rids) = payload.get("evidence_rids").and_then(|v| v.as_array()) {
        for rid_val in rids {
            if let Some(rid) = rid_val.as_str() {
                conn.execute(
                    "INSERT OR IGNORE INTO pattern_evidence (pattern_id, rid) VALUES (?1, ?2)",
                    params![pattern_id, rid],
                )?;
            }
        }
    }
    if let Some(names) = payload.get("entity_names").and_then(|v| v.as_array()) {
        for name_val in names {
            if let Some(name) = name_val.as_str() {
                conn.execute(
                    "INSERT OR IGNORE INTO pattern_entities (pattern_id, entity_name) VALUES (?1, ?2)",
                    params![pattern_id, name],
                )?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::YantrikDB;

    fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim).map(|i| (seed + i as f32) * 0.1).collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    }

    fn empty_meta() -> serde_json::Value {
        serde_json::json!({})
    }

    #[test]
    fn test_extract_ops_empty() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let ops = extract_ops_since(&*db.conn(), None, None, None, 100).unwrap();
        assert!(ops.is_empty());
    }

    #[test]
    fn test_extract_ops_after_record() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        db.record(
            "hello",
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

        let ops = extract_ops_since(&*db.conn(), None, None, None, 100).unwrap();
        // record + reinforce (from recall? no — just record op)
        assert!(!ops.is_empty());
        assert_eq!(ops[0].op_type, "record");
        assert_eq!(ops[0].payload["text"], "hello");
    }

    #[test]
    fn test_apply_ops_idempotent() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        a.record(
            "from A",
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

        let ops = extract_ops_since(&*a.conn(), None, None, None, 100).unwrap();

        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();

        // Apply once
        let r1 = apply_ops(&b, &ops).unwrap();
        assert_eq!(r1.ops_applied, ops.len());

        // Apply again — all skipped
        let r2 = apply_ops(&b, &ops).unwrap();
        assert_eq!(r2.ops_applied, 0);
        assert_eq!(r2.ops_skipped, ops.len());
    }

    #[test]
    fn test_materialize_record() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        let rid = a
            .record(
                "test mem",
                "semantic",
                0.8,
                0.2,
                1000.0,
                &serde_json::json!({"k": "v"}),
                &vec_seed(1.0, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        let ops = extract_ops_since(&*a.conn(), None, None, None, 100).unwrap();
        let record_op = ops.iter().find(|o| o.op_type == "record").unwrap();

        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();
        apply_ops(&b, &[record_op.clone()]).unwrap();

        // Check the memory was materialized
        let mem = b.get(&rid).unwrap();
        assert!(mem.is_some());
        let mem = mem.unwrap();
        assert_eq!(mem.text, "test mem");
        assert_eq!(mem.memory_type, "semantic");
        assert_eq!(mem.importance, 0.8);
    }

    #[test]
    fn test_replicated_record_preserves_provenance() {
        // **T06 anti-laundering at the replication boundary (sol Item 4 design
        // review, 2026-07-14).** materialize_record used to drop source,
        // certainty, domain, and emotional_state, so a replicated
        // source="inference" record silently became source="user" (the schema
        // default) on the follower's durable row. Prove all four survive the
        // hop verbatim.
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        let rid = a
            .record(
                "the sky is green",
                "semantic",
                0.7,
                0.1,
                1000.0,
                &serde_json::json!({"kind": "inference"}),
                &vec_seed(1.0, 8),
                "work",
                0.42,        // non-default certainty
                "science",   // non-default domain
                "inference", // the field that was being laundered to "user"
                Some("concern"),
            )
            .unwrap();

        let ops = extract_ops_since(&*a.conn(), None, None, None, 100).unwrap();
        let record_op = ops.iter().find(|o| o.op_type == "record").unwrap();

        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();
        apply_ops(&b, &[record_op.clone()]).unwrap();

        let mem = b
            .get(&rid)
            .unwrap()
            .expect("record materialized on follower");
        assert_eq!(
            mem.source, "inference",
            "source must NOT be laundered to 'user'"
        );
        assert_eq!(mem.certainty, 0.42, "certainty must survive replication");
        assert_eq!(mem.domain, "science", "domain must survive replication");
        assert_eq!(
            mem.emotional_state.as_deref(),
            Some("concern"),
            "emotional_state must survive replication"
        );
        assert_eq!(mem.namespace, "work");
    }

    // ── Item 4a.1 single-origin ingress guard ──

    fn rec(db: &YantrikDB, text: &str) -> String {
        db.record(
            text,
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
        .unwrap()
    }
    fn ops_of(db: &YantrikDB) -> Vec<OplogEntry> {
        extract_ops_since(&db.conn(), None, None, None, 100).unwrap()
    }

    #[test]
    fn synthesis_replication_verifies_complete_evidence_and_fails_closed_out_of_order() {
        let origin = YantrikDB::new_with_actor(":memory:", 8, "origin").unwrap();
        let source_rids = [
            rec(&origin, "first evidence"),
            rec(&origin, "second evidence"),
        ];
        let synthesis = crate::consolidate::record_synthesis(
            &origin,
            &source_rids,
            "combined item",
            Some(&vec_seed(3.0, 8)),
            "asked",
            "atomic",
            &empty_meta(),
            "synth:replication:item-1",
        )
        .unwrap();
        let synthesis_rid = synthesis["consolidated_rid"].as_str().unwrap();
        let origin_ops = ops_of(&origin);
        let consolidate_op = origin_ops
            .iter()
            .find(|op| op.op_type == "consolidate")
            .unwrap()
            .clone();
        let record_ops: Vec<OplogEntry> = origin_ops
            .into_iter()
            .filter(|op| op.op_type == "record")
            .collect();
        let synthesis_op = record_ops
            .iter()
            .find(|op| op.payload["rid"] == synthesis_rid)
            .unwrap()
            .clone();

        let complete = YantrikDB::new_with_actor(":memory:", 8, "complete").unwrap();
        apply_ops(&complete, &record_ops).unwrap();
        let complete_state: String = complete
            .conn()
            .query_row(
                "SELECT synthesis_state FROM memories WHERE rid = ?1",
                params![synthesis_rid],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(complete_state, "verified");
        {
            let cache = complete.scoring_cache.read();
            let row = cache.get(synthesis_rid).unwrap();
            assert_eq!(row.synthesis_axis.as_deref(), Some("asked"));
            assert_eq!(row.synthesis_granularity.as_deref(), Some("atomic"));
        }
        assert_eq!(
            complete
                .stats(None)
                .unwrap()
                .synthesis_fanout_current_high_water,
            1
        );
        let dependency_count: i64 = complete
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM synthesis_dependencies WHERE synthesis_rid = ?1",
                params![synthesis_rid],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(dependency_count, 2);
        apply_ops(&complete, &[consolidate_op]).unwrap();
        for source_rid in &source_rids {
            let status: String = complete
                .conn()
                .query_row(
                    "SELECT consolidation_status FROM memories WHERE rid = ?1",
                    params![source_rid],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(
                status, "active",
                "additive synthesis must not retire evidence"
            );
        }

        let out_of_order = YantrikDB::new_with_actor(":memory:", 8, "out-of-order").unwrap();
        apply_ops(&out_of_order, &[synthesis_op.clone()]).unwrap();
        let out_of_order_state: String = out_of_order
            .conn()
            .query_row(
                "SELECT synthesis_state FROM memories WHERE rid = ?1",
                params![synthesis_rid],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(out_of_order_state, "unverified");
        assert_eq!(
            out_of_order
                .stats(None)
                .unwrap()
                .synthesis_fanout_current_high_water,
            0,
            "unverified out-of-order synthesis must not consume current fan-out"
        );
        let source_ops: Vec<OplogEntry> = record_ops
            .iter()
            .filter(|op| op.payload["rid"] != synthesis_rid)
            .cloned()
            .collect();
        apply_ops(&out_of_order, &source_ops).unwrap();
        let promoted_state: String = out_of_order
            .conn()
            .query_row(
                "SELECT synthesis_state FROM memories WHERE rid = ?1",
                params![synthesis_rid],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(promoted_state, "verified");
        {
            let cache = out_of_order.scoring_cache.read();
            let row = cache.get(synthesis_rid).unwrap();
            assert_eq!(row.synthesis_state.as_deref(), Some("verified"));
            assert_eq!(row.synthesis_axis.as_deref(), Some("asked"));
            assert_eq!(row.synthesis_granularity.as_deref(), Some("atomic"));
        }
        assert_eq!(
            out_of_order
                .stats(None)
                .unwrap()
                .synthesis_fanout_current_high_water,
            1,
            "promotion after evidence arrival must consume current fan-out"
        );

        let malformed_target = YantrikDB::new_with_actor(":memory:", 8, "malformed").unwrap();
        let malformed_rid = crate::id::new_id();
        let mut malformed_op = synthesis_op;
        malformed_op.op_id = crate::id::new_id();
        malformed_op.target_rid = Some(malformed_rid.clone());
        malformed_op.payload["rid"] = serde_json::json!(malformed_rid);
        malformed_op.payload["synthesis"]["dependencies"] = serde_json::json!([]);
        apply_ops(&malformed_target, &[malformed_op]).unwrap();
        let malformed_state: String = malformed_target
            .conn()
            .query_row(
                "SELECT synthesis_state FROM memories WHERE rid = ?1",
                params![malformed_rid],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(malformed_state, "unverified");
        let malformed_dependency_count: i64 = malformed_target
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM synthesis_dependencies WHERE synthesis_rid = ?1",
                params![malformed_rid],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(malformed_dependency_count, 0);

        let corrected = YantrikDB::new_with_actor(":memory:", 8, "corrected").unwrap();
        apply_ops(&corrected, &record_ops).unwrap();
        assert_eq!(
            corrected
                .stats(None)
                .unwrap()
                .synthesis_fanout_current_high_water,
            1
        );
        origin
            .correct(
                &source_rids[1],
                None,
                Some(&serde_json::json!({"corrected": true})),
                None,
                None,
                "replicated source correction",
            )
            .unwrap();
        let correct_op = ops_of(&origin)
            .into_iter()
            .find(|op| op.op_type == "correct" && op.payload["rid"] == source_rids[1])
            .unwrap();
        apply_ops(&corrected, &[correct_op]).unwrap();
        assert_eq!(
            corrected
                .stats(None)
                .unwrap()
                .synthesis_fanout_current_high_water,
            0,
            "replicated correction must release current fan-out"
        );

        origin.forget(&source_rids[0]).unwrap();
        let forget_op = ops_of(&origin)
            .into_iter()
            .find(|op| op.op_type == "forget" && op.payload["rid"] == source_rids[0])
            .unwrap();
        apply_ops(&complete, &[forget_op]).unwrap();
        let invalidated_state: String = complete
            .conn()
            .query_row(
                "SELECT synthesis_state FROM memories WHERE rid = ?1",
                params![synthesis_rid],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(invalidated_state, "invalidated");
        assert_eq!(
            complete
                .stats(None)
                .unwrap()
                .synthesis_fanout_current_high_water,
            0,
            "replicated forget must release current fan-out"
        );
    }

    #[test]
    fn synthesis_replication_refolds_logical_generations_independent_of_arrival_order() {
        let origin = YantrikDB::new_with_actor(":memory:", 8, "origin").unwrap();
        let source_rids = [
            rec(&origin, "first evidence"),
            rec(&origin, "later evidence"),
        ];
        let first = crate::consolidate::record_synthesis(
            &origin,
            &source_rids[..1],
            "topic from first evidence",
            Some(&vec_seed(3.0, 8)),
            "topic",
            "rollup",
            &empty_meta(),
            "synth:replication:logical-topic",
        )
        .unwrap();
        let second = crate::consolidate::record_synthesis(
            &origin,
            &source_rids,
            "topic with later evidence",
            Some(&vec_seed(3.0, 8)),
            "topic",
            "rollup",
            &empty_meta(),
            "synth:replication:logical-topic",
        )
        .unwrap();
        let first_rid = first["consolidated_rid"].as_str().unwrap();
        let second_rid = second["consolidated_rid"].as_str().unwrap();
        let ops = ops_of(&origin);
        let source_ops: Vec<_> = ops
            .iter()
            .filter(|op| {
                op.op_type == "record"
                    && source_rids
                        .iter()
                        .any(|rid| op.payload["rid"].as_str() == Some(rid))
            })
            .cloned()
            .collect();
        let first_op = ops
            .iter()
            .find(|op| op.op_type == "record" && op.payload["rid"] == first_rid)
            .unwrap()
            .clone();
        let second_op = ops
            .iter()
            .find(|op| op.op_type == "record" && op.payload["rid"] == second_rid)
            .unwrap()
            .clone();

        for synthesis_ops in [
            vec![first_op.clone(), second_op.clone()],
            vec![second_op, first_op],
        ] {
            let replica = YantrikDB::new_with_actor(":memory:", 8, "replica").unwrap();
            apply_ops(&replica, &source_ops).unwrap();
            apply_ops(&replica, &synthesis_ops).unwrap();

            let state = |rid: &str| -> String {
                replica
                    .conn()
                    .query_row(
                        "SELECT synthesis_state FROM memories WHERE rid = ?1",
                        params![rid],
                        |row| row.get(0),
                    )
                    .unwrap()
            };
            assert_eq!(state(first_rid), "superseded");
            assert_eq!(state(second_rid), "verified");
            assert_eq!(
                replica
                    .audit_synthesis_evidence(None, 10)
                    .unwrap()
                    .duplicate_logical_key_group_count,
                0
            );
        }
    }

    #[test]
    fn replicated_synthesis_converges_over_local_fanout_cap_and_is_observable() {
        let origin = YantrikDB::new_with_actor(":memory:", 8, "fanout-origin").unwrap();
        origin.set_synthesis_fanout_cap(2).unwrap();
        let source_rid = rec(&origin, "shared evidence");
        for index in 1..=2 {
            crate::consolidate::record_synthesis(
                &origin,
                std::slice::from_ref(&source_rid),
                &format!("origin synthesis {index}"),
                Some(&vec_seed(index as f32 + 1.0, 8)),
                "asked",
                "atomic",
                &empty_meta(),
                &format!("synth:replicated-fanout:{index}"),
            )
            .unwrap();
        }

        let follower = YantrikDB::new_with_actor(":memory:", 8, "fanout-follower").unwrap();
        follower.set_synthesis_fanout_cap(1).unwrap();
        apply_ops(&follower, &ops_of(&origin)).unwrap();

        let stats = follower.stats(None).unwrap();
        assert_eq!(stats.synthesis_fanout_cap, 1);
        assert_eq!(stats.synthesis_fanout_current_high_water, 2);
        assert_eq!(stats.synthesis_fanout_sources_at_cap, 0);
        assert_eq!(stats.synthesis_fanout_sources_over_cap, 1);
        assert_eq!(stats.synthesis_fanout_refused_since_boot, 0);

        let error = crate::consolidate::record_synthesis(
            &follower,
            std::slice::from_ref(&source_rid),
            "local synthesis must wait",
            Some(&vec_seed(5.0, 8)),
            "asked",
            "atomic",
            &empty_meta(),
            "synth:replicated-fanout:local",
        )
        .unwrap_err();
        assert!(matches!(
            error,
            crate::error::YantrikDbError::SynthesisFanoutLimit {
                current: 2,
                limit: 1,
                ..
            }
        ));
        assert_eq!(
            follower
                .stats(None)
                .unwrap()
                .synthesis_fanout_refused_since_boot,
            1
        );
    }

    #[test]
    fn is_protected_write_classification() {
        for t in ["record", "record_with_rid", "correct", "consolidate"] {
            assert!(is_protected_write(t), "{t} must be protected");
        }
        for t in [
            "relate",
            "link",
            "unlink",
            "forget",
            "think",
            "trigger_fire",
        ] {
            assert!(!is_protected_write(t), "{t} must NOT be guarded");
        }
    }

    #[test]
    fn origin_guard_inactive_by_default_allows_foreign() {
        // Backward-compat: no authority configured -> foreign ops apply (legacy).
        let a = YantrikDB::new_with_actor(":memory:", 8, "actor-A").unwrap();
        let rid = rec(&a, "from A");
        let b = YantrikDB::new_with_actor(":memory:", 8, "actor-B").unwrap();
        apply_ops(&b, &ops_of(&a)).unwrap();
        assert!(
            b.get(&rid).unwrap().is_some(),
            "no authority set -> foreign-origin op applies"
        );
    }

    #[test]
    fn origin_guard_rejects_foreign_protected_write() {
        let c = YantrikDB::new_with_actor(":memory:", 8, "actor-C").unwrap();
        let c_rid = rec(&c, "from C");
        let c_ops = ops_of(&c);
        let b = YantrikDB::new_with_actor(":memory:", 8, "actor-B").unwrap();
        b.set_authoritative_origin("actor-A").unwrap(); // B trusts only A
        let err = apply_ops(&b, &c_ops).unwrap_err();
        assert!(
            matches!(
                err,
                crate::error::YantrikDbError::ForeignOriginRejected { .. }
            ),
            "foreign-origin protected write must be rejected, got {err:?}"
        );
        // State unchanged: no memory, and the op did not enter the oplog.
        assert!(
            b.get(&c_rid).unwrap().is_none(),
            "rejected write leaves no memory"
        );
        let inserted: bool = b
            .conn()
            .query_row(
                "SELECT COUNT(*) > 0 FROM oplog WHERE op_id = ?1",
                params![c_ops[0].op_id],
                |r| r.get(0),
            )
            .unwrap();
        assert!(!inserted, "rejected op must not enter the oplog");
    }

    #[test]
    fn origin_guard_batch_atomic_rejects_whole_mixed_batch() {
        // A C-origin op in a batch that also carries an authority (A) op: the
        // whole batch is rejected and NEITHER op applies (batch-atomicity).
        let a = YantrikDB::new_with_actor(":memory:", 8, "actor-A").unwrap();
        let a_rid = rec(&a, "from A");
        let mut batch = ops_of(&a);
        let c = YantrikDB::new_with_actor(":memory:", 8, "actor-C").unwrap();
        let c_rid = rec(&c, "from C");
        batch.extend(ops_of(&c));

        let b = YantrikDB::new_with_actor(":memory:", 8, "actor-B").unwrap();
        b.set_authoritative_origin("actor-A").unwrap();
        let err = apply_ops(&b, &batch).unwrap_err();
        assert!(matches!(
            err,
            crate::error::YantrikDbError::ForeignOriginRejected { .. }
        ));
        assert!(
            b.get(&a_rid).unwrap().is_none(),
            "batch-atomic: the authority op must also NOT apply when the batch is rejected"
        );
        assert!(
            b.get(&c_rid).unwrap().is_none(),
            "the foreign op must not apply"
        );
    }

    #[test]
    fn origin_guard_rejects_c_relayed_through_b() {
        // TRUE relay (sol 4a.1 finding 3): C originates, B (unguarded) applies
        // then re-exports it, and the guarded node G receives B's RELAYED
        // batch. G must reject by the op's ORIGIN (C), not the deliverer (B) —
        // which requires origin_actor to survive the relay hop.
        let c = YantrikDB::new_with_actor(":memory:", 8, "actor-C").unwrap();
        let c_rid = rec(&c, "from C");
        let b = YantrikDB::new_with_actor(":memory:", 8, "actor-B").unwrap();
        apply_ops(&b, &ops_of(&c)).unwrap(); // B relays (no authority set)
        let relayed = ops_of(&b); // extracted FROM B
        assert!(
            relayed
                .iter()
                .any(|o| o.op_type == "record" && o.origin_actor == "actor-C"),
            "relay must preserve origin_actor=C, got {:?}",
            relayed.iter().map(|o| &o.origin_actor).collect::<Vec<_>>()
        );

        let g = YantrikDB::new_with_actor(":memory:", 8, "actor-G").unwrap();
        g.set_authoritative_origin("actor-A").unwrap(); // G trusts only A
        let err = apply_ops(&g, &relayed).unwrap_err();
        assert!(
            matches!(
                err,
                crate::error::YantrikDbError::ForeignOriginRejected { .. }
            ),
            "a C-origin op relayed through B must be rejected by G, got {err:?}"
        );
        assert!(
            g.get(&c_rid).unwrap().is_none(),
            "relayed foreign-origin op must not apply on the guarded node"
        );
    }

    #[test]
    fn origin_guard_allows_authority_origin() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "actor-A").unwrap();
        let a_rid = rec(&a, "from A");
        let b = YantrikDB::new_with_actor(":memory:", 8, "actor-B").unwrap();
        b.set_authoritative_origin("actor-A").unwrap();
        apply_ops(&b, &ops_of(&a)).unwrap();
        assert!(
            b.get(&a_rid).unwrap().is_some(),
            "authority-origin op must apply"
        );
    }

    #[test]
    fn test_replicated_record_with_rid_materializes_with_provenance() {
        // **sol Item 4 design review, 2026-07-14.** "record_with_rid" ops had
        // no materialize_op arm, so records created via the cluster path
        // silently hit the unknown-op branch and never peer-replicated. Prove
        // the op now materializes with provenance intact AND that its
        // `created_at_unix_micros` field is converted to seconds (not dropped
        // to epoch 0 like the "record" op's `created_at` reader would).
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        let created_micros: i64 = 1_700_000_000_000_000; // = 1_700_000_000 s
        a.record_with_rid(
            "01900000-0000-7000-8000-0000000000aa",
            "cluster-path fact",
            "semantic",
            0.6,
            0.0,
            1000.0,
            &serde_json::json!({"kind": "inference"}),
            &vec_seed(1.0, 8),
            "work",
            0.33,
            "science",
            "inference",
            Some("concern"),
            created_micros,
            &[],
            "test-model",
            None,
            crate::provenance::WriteAdmission::Admitted,
        )
        .unwrap();

        let ops = extract_ops_since(&*a.conn(), None, None, None, 100).unwrap();
        let op = ops
            .iter()
            .find(|o| o.op_type == "record_with_rid")
            .expect("record_with_rid op present in oplog");

        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();
        apply_ops(&b, &[op.clone()]).unwrap();

        let mem = b
            .get("01900000-0000-7000-8000-0000000000aa")
            .unwrap()
            .expect("record_with_rid op materialized on follower (was silently dropped before)");
        assert_eq!(mem.text, "cluster-path fact");
        assert_eq!(mem.source, "inference");
        assert_eq!(mem.certainty, 0.33);
        assert_eq!(mem.domain, "science");
        assert_eq!(
            mem.created_at, 1_700_000_000.0,
            "created_at_unix_micros must be converted to seconds"
        );
    }

    #[test]
    fn test_materialize_record_writes_replication_apply_log() {
        // **v0.7.19 audit-table verification (postmortem 2026-05-20).**
        // When a record op arrives via replication apply, the
        // replication_apply_log table gets a row stamping op_type +
        // source_actor + applied_at. Three-population audit query
        // can then distinguish:
        //   - locally originated: in oplog with origin_actor = self
        //   - received via replication: in replication_apply_log
        //   - true orphan (Backpressure-orphan or bug): in neither
        let a = YantrikDB::new_with_actor(":memory:", 8, "actor-A").unwrap();
        let rid = a
            .record(
                "from A",
                "semantic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &vec_seed(1.0, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        let ops = extract_ops_since(&*a.conn(), None, None, None, 100).unwrap();
        let record_op = ops.iter().find(|o| o.op_type == "record").unwrap();

        // B is a separate engine instance — apply the remote op.
        let b = YantrikDB::new_with_actor(":memory:", 8, "actor-B").unwrap();
        apply_ops(&b, &[record_op.clone()]).unwrap();

        // On B: memories row exists, AND replication_apply_log row exists.
        let conn = b.conn();
        let mem_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE rid = ?1",
                params![&rid],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(mem_count, 1, "B materialized the row");

        let (op_type, source_actor): (String, String) = conn
            .query_row(
                "SELECT op_type, source_actor FROM replication_apply_log WHERE rid = ?1",
                params![&rid],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap_or_else(|_| {
                panic!("v0.7.19: replication_apply_log must have a row for replicated rid {rid}")
            });
        assert_eq!(op_type, "record");
        assert_eq!(
            source_actor, "actor-A",
            "source_actor records the originator's actor_id"
        );

        // Audit-query: on B, the row is NOT in B's local oplog (B
        // didn't originate it) but IS in replication_apply_log. So
        // the three-population shape works:
        let received_via_replication: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories \
                 WHERE rid IN (SELECT rid FROM replication_apply_log)",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(received_via_replication, 1);

        let true_orphans: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories \
                 WHERE rid NOT IN (SELECT target_rid FROM oplog WHERE target_rid IS NOT NULL) \
                   AND rid NOT IN (SELECT rid FROM replication_apply_log)",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            true_orphans, 0,
            "no true orphans on B — every memories row is accounted for by replication_apply_log"
        );
    }

    #[test]
    fn test_tombstone_wins() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        let rid = a
            .record(
                "doomed",
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
        a.forget(&rid).unwrap();

        let ops = extract_ops_since(&*a.conn(), None, None, None, 100).unwrap();

        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();
        apply_ops(&b, &ops).unwrap();

        let mem = b.get(&rid).unwrap().unwrap();
        assert_eq!(mem.consolidation_status, "tombstoned");
    }

    #[test]
    fn test_materialize_relate() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        a.relate("Alice", "Bob", "knows", 0.9).unwrap();

        let ops = extract_ops_since(&*a.conn(), None, None, None, 100).unwrap();
        let relate_op = ops.iter().find(|o| o.op_type == "relate").unwrap();

        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();
        apply_ops(&b, &[relate_op.clone()]).unwrap();

        let edges = b.get_edges("Alice").unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].dst, "Bob");
        assert_eq!(edges[0].weight, 0.9);
    }

    #[test]
    fn test_lww_edge_merge() {
        // Both create same (src,dst,rel_type) with different weights
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        a.relate("X", "Y", "linked", 0.3).unwrap();

        // B creates same edge but later (higher timestamp)
        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        b.relate("X", "Y", "linked", 0.9).unwrap();

        // Apply A's ops to B
        let a_ops = extract_ops_since(&*a.conn(), None, None, None, 100).unwrap();
        apply_ops(&b, &a_ops).unwrap();

        // B should keep its own weight (0.9) since it's newer
        let edges = b.get_edges("X").unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].weight, 0.9);

        // Apply B's ops to A
        let b_ops = extract_ops_since(&*b.conn(), None, None, None, 100).unwrap();
        apply_ops(&a, &b_ops).unwrap();

        // A should now have B's weight (0.9) since it's newer
        let edges = a.get_edges("X").unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].weight, 0.9);
    }

    // ------------------------------------------------------------------
    // #148 — relate LWW must compare HLC, not wall-clock created_at.
    // Every test here constructs ops where the two orderings DISAGREE:
    // under the old created_at comparison each assertion fails, which is
    // how the tests were verified before the fix was applied.
    // ------------------------------------------------------------------

    /// Hand-build a relate op with independently chosen wall clock and HLC —
    /// the skew that only multi-node deployments produce.
    fn relate_op_skewed(
        edge_id: &str,
        weight: f64,
        created_at: f64,
        hlc_millis: u64,
        actor: &str,
    ) -> OplogEntry {
        let hlc = HLCTimestamp {
            millis: hlc_millis,
            logical: 0,
            node_id: 1,
        }
        .to_bytes()
        .to_vec();
        OplogEntry {
            op_id: format!("op-{edge_id}"),
            op_type: "relate".to_string(),
            timestamp: created_at,
            target_rid: Some(edge_id.to_string()),
            payload: serde_json::json!({
                "edge_id": edge_id,
                "src": "X",
                "dst": "Y",
                "rel_type": "linked",
                "weight": weight,
                "created_at": created_at,
                "edge_hlc_hex": hex::encode(&hlc),
            }),
            actor_id: actor.to_string(),
            hlc,
            embedding_hash: None,
            origin_actor: actor.to_string(),
            embedding: None,
        }
    }

    fn edge_row(db: &YantrikDB) -> (String, f64, Option<Vec<u8>>) {
        db.conn()
            .query_row(
                "SELECT claim_id, weight, hlc FROM claims \
                 WHERE src = 'X' AND dst = 'Y' AND rel_type = 'linked'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap()
    }

    fn indexed_edge_weight(db: &YantrikDB) -> f64 {
        db.graph_index
            .read()
            .expand_bfs(&["X"], 1, 8)
            .into_iter()
            .find_map(|(name, hops, weight)| (name == "Y" && hops == 1).then_some(weight))
            .expect("X -> Y must be present in the graph index")
    }

    #[test]
    fn relate_lww_higher_hlc_wins_despite_older_wall_clock() {
        // new-edge writer's clock is SLOW: causally later (hlc 2000) but a
        // smaller created_at (100 < 200). Under wall-clock LWW the stale op
        // from the fast clock would silently resurrect the old weight.
        let newer_causal = relate_op_skewed("e-new", 0.9, 100.0, 2000, "slow-clock");
        let stale_fast = relate_op_skewed("e-old", 0.1, 200.0, 1000, "fast-clock");

        let db = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();
        apply_ops(&db, &[newer_causal.clone()]).unwrap();
        apply_ops(&db, &[stale_fast.clone()]).unwrap();
        let (claim_id, weight, hlc) = edge_row(&db);
        assert_eq!(claim_id, "e-new", "stale op must not overwrite claim_id");
        assert_eq!(weight, 0.9, "stale op must not overwrite weight");
        assert_eq!(hlc.as_deref(), Some(&newer_causal.hlc[..]));

        // Arrival-order independence: reversed application converges to the
        // same row.
        let db2 = YantrikDB::new_with_actor(":memory:", 8, "C").unwrap();
        apply_ops(&db2, &[stale_fast]).unwrap();
        apply_ops(&db2, &[newer_causal]).unwrap();
        assert_eq!(edge_row(&db2), edge_row(&db), "order must not matter");
    }

    #[test]
    fn relate_lww_stale_op_does_not_poison_graph_index() {
        let newer_causal = relate_op_skewed("e-new", 0.9, 100.0, 2000, "slow-clock");
        let stale_fast = relate_op_skewed("e-old", 0.1, 200.0, 1000, "fast-clock");
        let db = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();

        apply_ops(&db, &[newer_causal]).unwrap();
        apply_ops(&db, &[stale_fast]).unwrap();

        assert_eq!(edge_row(&db).1, 0.9, "SQL row keeps the causal winner");
        assert!(
            (indexed_edge_weight(&db) - 0.9).abs() < 1e-6,
            "a losing op must not replace the causal winner in graph_index"
        );
    }

    #[test]
    fn repeated_local_relate_keeps_claim_identity_equal_on_follower() {
        let leader = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        let first = leader.relate("X", "Y", "linked", 0.1).unwrap();
        let second = leader.relate("X", "Y", "linked", 0.9).unwrap();
        assert_ne!(first, second);

        let follower = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();
        apply_ops(&follower, &ops_of(&leader)).unwrap();

        assert_eq!(
            edge_row(&leader),
            edge_row(&follower),
            "leader and follower must agree on claim_id, weight, and HLC"
        );
    }

    #[test]
    fn relate_lww_equal_hlc_is_a_complete_payload_noop() {
        let winner = relate_op_skewed("e-winner", 0.9, 100.0, 2000, "A");
        let mut mutated = winner.clone();
        mutated.target_rid = Some("e-mutated".to_string());
        mutated.payload["edge_id"] = serde_json::json!("e-mutated");
        mutated.payload["weight"] = serde_json::json!(0.1);
        mutated.payload["created_at"] = serde_json::json!(999.0);

        let db = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();
        apply_ops(&db, &[winner.clone()]).unwrap();
        apply_ops(&db, &[mutated]).unwrap();

        assert_eq!(
            edge_row(&db),
            ("e-winner".to_string(), 0.9, Some(winner.hlc)),
            "equal HLC must not partially mutate any persisted payload field"
        );
        assert!(
            (indexed_edge_weight(&db) - 0.9).abs() < 1e-6,
            "equal HLC must not mutate the graph index"
        );
    }

    #[test]
    fn relate_lww_legacy_payload_falls_back_to_envelope_hlc() {
        // Pre-#148 leaders send no edge_hlc_hex; the envelope HLC decides.
        let mut newer = relate_op_skewed("e-new", 0.9, 100.0, 2000, "slow-clock");
        let mut stale = relate_op_skewed("e-old", 0.1, 200.0, 1000, "fast-clock");
        for op in [&mut newer, &mut stale] {
            op.payload.as_object_mut().unwrap().remove("edge_hlc_hex");
        }

        let db = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();
        apply_ops(&db, &[newer]).unwrap();
        apply_ops(&db, &[stale]).unwrap();
        let (claim_id, weight, _) = edge_row(&db);
        assert_eq!(claim_id, "e-new");
        assert_eq!(weight, 0.9);
    }

    #[test]
    fn relate_lww_legacy_row_without_hlc_compares_created_at() {
        // A pre-v47 row (hlc NULL) keeps the old wall-clock semantics: an
        // incoming op with older created_at loses, newer created_at wins —
        // exactly what the old code did for exactly the rows it wrote.
        let db = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();
        db.conn()
            .execute(
                "INSERT INTO claims (claim_id, src, dst, rel_type, weight, created_at) \
                 VALUES ('e-legacy', 'X', 'Y', 'linked', 0.5, 150.0)",
                [],
            )
            .unwrap();

        // Older wall clock (100 < 150): must NOT overwrite, despite carrying
        // an HLC (any HLC beats a NULL under naive COALESCE semantics — this
        // pins the created_at fallback instead).
        apply_ops(&db, &[relate_op_skewed("e-older", 0.1, 100.0, 2000, "A")]).unwrap();
        let (claim_id, weight, _) = edge_row(&db);
        assert_eq!(
            claim_id, "e-legacy",
            "older created_at must lose to legacy row"
        );
        assert_eq!(weight, 0.5);

        // Newer wall clock (200 > 150): wins, and the row is HLC-stamped
        // from here on.
        let winner = relate_op_skewed("e-newer", 0.9, 200.0, 1000, "A");
        apply_ops(&db, &[winner.clone()]).unwrap();
        let (claim_id, weight, hlc) = edge_row(&db);
        assert_eq!(claim_id, "e-newer");
        assert_eq!(weight, 0.9);
        assert_eq!(hlc.as_deref(), Some(&winner.hlc[..]));
    }

    #[test]
    fn relate_replication_carries_edge_hlc_verbatim() {
        // Leader mints the edge HLC once; the follower's row must hold the
        // SAME bytes (record_links edge-identity pattern), so every replica
        // sorts identically.
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        a.relate("X", "Y", "linked", 0.7).unwrap();
        let (_, _, leader_hlc) = edge_row(&a);
        let leader_hlc = leader_hlc.expect("leader relate() must stamp claims.hlc");

        let ops = ops_of(&a);
        let relate_op = ops.iter().find(|o| o.op_type == "relate").unwrap();
        assert_eq!(
            relate_op.payload["edge_hlc_hex"].as_str(),
            Some(hex::encode(&leader_hlc).as_str()),
            "payload must carry the stamped HLC verbatim"
        );

        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();
        apply_ops(&b, &[relate_op.clone()]).unwrap();
        let (_, _, follower_hlc) = edge_row(&b);
        assert_eq!(
            follower_hlc.as_deref(),
            Some(&leader_hlc[..]),
            "follower row must hold the leader's exact HLC bytes"
        );
    }

    #[test]
    fn test_watermark_tracking() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let conn = db.conn();

        // No watermark initially
        let wm = get_peer_watermark(&*conn, "peer-1").unwrap();
        assert!(wm.is_none());

        // Set watermark
        let hlc_bytes = vec![0u8; 16];
        set_peer_watermark(&*conn, "peer-1", &hlc_bytes, "op-123").unwrap();

        let wm = get_peer_watermark(&*conn, "peer-1").unwrap().unwrap();
        assert_eq!(wm.0, hlc_bytes);
        assert_eq!(wm.1, "op-123");

        // Update watermark
        let new_hlc = vec![1u8; 16];
        set_peer_watermark(&*conn, "peer-1", &new_hlc, "op-456").unwrap();

        let wm = get_peer_watermark(&*conn, "peer-1").unwrap().unwrap();
        assert_eq!(wm.0, new_hlc);
        assert_eq!(wm.1, "op-456");
    }

    #[test]
    fn test_extract_with_exclude_actor() {
        let db = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        db.record(
            "from A",
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

        // Extracting while excluding actor "A" should return nothing
        let ops = extract_ops_since(&*db.conn(), None, None, Some("A"), 100).unwrap();
        assert!(ops.is_empty());

        // Extracting without exclusion should return the op
        let ops = extract_ops_since(&*db.conn(), None, None, None, 100).unwrap();
        assert!(!ops.is_empty());
    }

    #[test]
    fn test_consolidation_members_replicate() {
        let a = YantrikDB::new_with_actor(":memory:", 8, "A").unwrap();
        a.record(
            "mem1",
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
        a.record(
            "mem2",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.1, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

        // Consolidate on A
        let consolidated =
            crate::consolidate::consolidate(&a, 0.0, 365.0, 2, 10000, false, false).unwrap();
        assert!(!consolidated.is_empty());

        // Extract all ops and apply to B
        let ops = extract_ops_since(&*a.conn(), None, None, None, 1000).unwrap();
        let b = YantrikDB::new_with_actor(":memory:", 8, "B").unwrap();
        apply_ops(&b, &ops).unwrap();

        // Check that B has the consolidation_members entries
        let count: i64 = b
            .conn()
            .query_row("SELECT COUNT(*) FROM consolidation_members", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert!(count >= 2); // At least 2 source_rids
    }

    /// Regression test for issue #13 (yantrikos/yantrikdb-server).
    ///
    /// Before the fix, calling extract_ops_since with only since_op_id OR
    /// only since_hlc silently returned ALL ops from the start of the log,
    /// because the match expression's `_` arm built SQL with no boundary.
    /// Both watermarks alone must independently filter.
    ///
    /// Reproduction adapted from @mbseid's report.
    #[test]
    fn test_extract_ops_since_single_watermark() {
        let db = YantrikDB::new(":memory:", 8).unwrap();

        // Batch 1: two ops, save the second as the watermark.
        db.record(
            "I love coffee",
            "semantic",
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
        db.record(
            "I go hiking on weekends",
            "episodic",
            0.6,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(2.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

        let batch1 = extract_ops_since(&*db.conn(), None, None, None, 100).unwrap();
        assert_eq!(
            batch1.len(),
            2,
            "batch1 should contain exactly the 2 record ops"
        );
        let wm = batch1.last().unwrap().clone();
        let wm_op_id = wm.op_id.clone();
        let wm_hlc = wm.hlc.clone();

        // Batch 2: two more ops.
        db.record(
            "I work in software",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(3.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
        db.record(
            "Saturdays are for chores",
            "episodic",
            0.4,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(4.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

        // 1. op_id-only watermark: should return ONLY the 2 new ops.
        let after_op_id =
            extract_ops_since(&*db.conn(), None, Some(wm_op_id.as_str()), None, 100).unwrap();
        assert_eq!(
            after_op_id.len(),
            2,
            "since_op_id alone must filter; got {} ops, expected 2",
            after_op_id.len()
        );
        for op in &after_op_id {
            assert!(op.op_id != wm_op_id, "watermark op must not be returned");
        }

        // 2. hlc-only watermark: should return ONLY the 2 new ops.
        let after_hlc =
            extract_ops_since(&*db.conn(), Some(wm_hlc.as_slice()), None, None, 100).unwrap();
        assert_eq!(
            after_hlc.len(),
            2,
            "since_hlc alone must filter; got {} ops, expected 2",
            after_hlc.len()
        );

        // 3. Both watermarks together (the originally-working path): same result.
        let after_both = extract_ops_since(
            &*db.conn(),
            Some(wm_hlc.as_slice()),
            Some(wm_op_id.as_str()),
            None,
            100,
        )
        .unwrap();
        assert_eq!(
            after_both.len(),
            2,
            "compound cursor must filter equivalently"
        );

        // 4. No watermark: should return all 4 ops.
        let all = extract_ops_since(&*db.conn(), None, None, None, 100).unwrap();
        assert_eq!(all.len(), 4, "no cursor returns full log");
    }
}
