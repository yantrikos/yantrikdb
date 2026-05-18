//! **Issue #41 brainstorm-4 §5 — SQL embedding access boundary.**
//!
//! The `memories.embedding` BLOB column is the **third** durable
//! truth in the engine (alongside `SearchState.vec_index` and
//! `meta.active_generation`). brainstorm-4 §5 caught that without a
//! module/trait boundary around it, any future caller can grow a
//! raw `SELECT embedding FROM memories WHERE rid = ?` — and during
//! a reembed, those bytes might be from the OLD vector space while
//! the active SearchState is at the NEW one. Silent corruption when
//! dimensions match; loud-but-late dim-mismatch when they don't.
//!
//! This module is the only sanctioned path through which engine code
//! reads durable embedding bytes. Every call returns the v28
//! `embedding_generation` stamp alongside the bytes so the caller
//! can decide what to do with stale-generation rows (skip, queue for
//! re-encode, refuse).
//!
//! ## Boundary audit
//!
//! [`tests::recall_rs_has_no_raw_sql_embedding_reads`] grep-scans the
//! source of `engine/recall.rs` at test time and fails the build if
//! a future refactor reintroduces a `SELECT ... embedding FROM
//! memories` outside this module. Recall is the hot user-facing
//! query path; recall.rs is the file the audit guards. The
//! cognition/* + storage/* + materializer paths have legitimate
//! reasons to read embedding bytes (compress for archive, score
//! similarity in background analytics) and are allowed to call this
//! module's API directly — they tolerate generation lag because
//! they read durable state and inherently see the SQL-committed
//! generation, not the in-memory one.
//!
//! ## Forward-compatibility with Phase 2
//!
//! Layer 4 Phase 2's Encoding phase reads embeddings from rows that
//! need re-encoding. It will use [`DurableEmbeddingStore::
//! read_embeddings_under_generation`] with the OLD generation, get
//! back the (bytes, gen) pairs, re-encode under the new embedder,
//! and write to `embedding_new` + `embedding_new_model`. The
//! Swapping phase's SAVEPOINT atomically promotes embedding_new ->
//! embedding and bumps both `embedding_generation` (row level) and
//! `meta.active_generation` (engine level).

use crate::error::Result;
use rusqlite::params;
use std::collections::HashMap;

use super::YantrikDB;

/// One row's embedding bytes plus the generation stamp.
///
/// `generation` is the v28 `memories.embedding_generation` column.
/// On pre-v28 rows the column is NULL — interpreted as `Some(0)`
/// (the initial pre-reembed generation, brainstorm-4 §6
/// convention).
///
/// Callers compare `generation` against the active
/// `SearchState.generation` to decide whether the bytes are
/// in-vector-space for the current index. A mismatch means a
/// reembed swap landed after this row was written; the bytes are
/// encoded under the prior embedder.
#[derive(Debug, Clone)]
pub struct EmbeddingWithGeneration {
    /// Decrypted f32-serialized embedding bytes (post-decrypt
    /// blob — the same shape callers expect from the old
    /// `fetch_embeddings_by_rids` API).
    pub bytes: Vec<u8>,
    /// Active generation at the time this row was written. NULL on
    /// pre-v28 rows is mapped to `0` (the initial generation).
    pub generation: u64,
}

/// Engine-internal sanctioned reader for `memories.embedding`.
///
/// Constructed per-call with `&YantrikDB`; holds no state of its
/// own. The struct exists so future call sites use a typed entry
/// point that's audit-discoverable; the methods could be free
/// functions but the wrapper makes the boundary explicit at every
/// call site.
pub(crate) struct DurableEmbeddingStore<'a> {
    db: &'a YantrikDB,
}

impl<'a> DurableEmbeddingStore<'a> {
    /// Construct a reader bound to the engine instance.
    pub fn new(db: &'a YantrikDB) -> Self {
        DurableEmbeddingStore { db }
    }

    /// Batch-read embeddings for an explicit set of rids.
    ///
    /// Returns `(rid, EmbeddingWithGeneration)` for each rid that
    /// has a non-NULL embedding column. Rids without an embedding
    /// (storage_tier='cold' before hydration, tombstoned rows, or
    /// rids that don't exist) are omitted from the result map.
    ///
    /// Caller policy on generation:
    /// - `entry.generation == active_search_state.generation`:
    ///   bytes are in the active vector space. Safe to use for
    ///   similarity scoring.
    /// - `entry.generation < active_search_state.generation`:
    ///   bytes are from an earlier generation. Phase 2 Encoding
    ///   path consumes these for re-encode; recall paths SHOULD
    ///   skip them (or queue them — depends on RYW semantics).
    /// - `entry.generation > active_search_state.generation`:
    ///   impossible under the durable-linearization invariant
    ///   (would mean SQL committed a later gen than `open()`
    ///   restored). Caller may treat as engine bug + panic, or
    ///   return error.
    pub fn read_embeddings_for_rids(
        &self,
        rids: &[&str],
    ) -> Result<HashMap<String, EmbeddingWithGeneration>> {
        if rids.is_empty() {
            return Ok(HashMap::new());
        }
        let placeholders: String = (0..rids.len())
            .map(|i| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(",");
        // The `embedding_generation IS NULL` -> 0 mapping is the
        // brainstorm-4 §6 convention for pre-v28 rows. Done in SQL
        // via COALESCE so callers always receive a u64.
        let sql = format!(
            "SELECT rid, embedding, COALESCE(embedding_generation, 0) \
             FROM memories WHERE rid IN ({placeholders})"
        );
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        for r in rids {
            param_values.push(Box::new(r.to_string()));
        }
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let conn = self.db.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows: Vec<(String, Vec<u8>, i64)> = stmt
            .query_map(params_ref.as_slice(), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        drop(stmt);
        drop(conn);

        let mut map = HashMap::new();
        for (rid, stored_emb, generation) in rows {
            // Decrypt under the engine's at-rest encryption (no-op
            // if encryption is disabled). The brainstorm-4 §5
            // contract: this module is the sole caller of
            // decrypt_embedding for durable rows on read paths
            // (record/storage are write paths and stay separate).
            let bytes = self.db.decrypt_embedding(&stored_emb)?;
            map.insert(
                rid,
                EmbeddingWithGeneration {
                    bytes,
                    generation: generation as u64,
                },
            );
        }
        Ok(map)
    }

    /// **Phase-2 Encoding helper (forward-compatible).** Stream
    /// embedding rows whose `embedding_generation` is strictly less
    /// than the supplied `active_generation`. These are the rows
    /// the Encoding phase needs to re-encode under the new
    /// embedder. Caller is responsible for paginating large result
    /// sets (Phase 2 batches at ReembedOptions::batch_size).
    ///
    /// Bounded to `active` rows
    /// (`consolidation_status = 'active'`). Tombstoned / archived
    /// rows are out of scope for re-encode.
    ///
    /// Returns `(rid, EmbeddingWithGeneration)` pairs in arbitrary
    /// order. The post-swap materializer (Layer 5) uses a similar
    /// scan but with `<= covers_through_seq` semantics; that lives
    /// in the materializer module.
    pub fn read_embeddings_under_generation(
        &self,
        active_generation: u64,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<(String, EmbeddingWithGeneration)>> {
        let conn = self.db.read_conn();
        let mut stmt = conn.prepare(
            "SELECT rid, embedding, COALESCE(embedding_generation, 0) \
             FROM memories \
             WHERE consolidation_status = 'active' \
               AND embedding IS NOT NULL \
               AND COALESCE(embedding_generation, 0) < ?1 \
             ORDER BY rid \
             LIMIT ?2 OFFSET ?3",
        )?;
        let rows: Vec<(String, Vec<u8>, i64)> = stmt
            .query_map(
                params![active_generation as i64, limit as i64, offset as i64],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                },
            )?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        drop(stmt);
        drop(conn);

        let mut out = Vec::with_capacity(rows.len());
        for (rid, stored_emb, generation) in rows {
            let bytes = self.db.decrypt_embedding(&stored_emb)?;
            out.push((
                rid,
                EmbeddingWithGeneration {
                    bytes,
                    generation: generation as u64,
                },
            ));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::YantrikDB;

    fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
        (0..dim).map(|i| seed + (i as f32) * 0.001).collect()
    }

    #[test]
    fn read_embeddings_for_rids_returns_bytes_and_generation() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let rid = db
            .record(
                "durable embed read",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &vec_seed(0.5, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        let store = DurableEmbeddingStore::new(&db);
        let map = store.read_embeddings_for_rids(&[&rid]).unwrap();
        assert_eq!(map.len(), 1, "exact match");
        let entry = map.get(&rid).unwrap();
        assert!(!entry.bytes.is_empty(), "bytes returned");
        assert_eq!(entry.generation, 0, "fresh engine writes rows at gen 0");
    }

    #[test]
    fn read_embeddings_for_rids_reflects_generation_advance() {
        // Lock the brainstorm-4 §6 row-level stamping contract from
        // the consumer side: after a manual generation advance, new
        // rows carry the new stamp.
        let db = YantrikDB::new(":memory:", 8).unwrap();

        // Manually advance the engine to gen 11 via the CAS helper.
        let old_state = db.search_state.load_full();
        let advanced = crate::engine::reembed::SearchState {
            index_embedding: old_state.index_embedding.clone(),
            embedder: old_state.embedder.clone(),
            runtime_embedder_name: old_state.runtime_embedder_name.clone(),
            runtime_embedder_digest: old_state.runtime_embedder_digest.clone(),
            generation: 11,
            covers_through_seq: old_state.covers_through_seq,
            hnsw_m: old_state.hnsw_m,
            hnsw_ef_construction: old_state.hnsw_ef_construction,
            hnsw_ef_search: old_state.hnsw_ef_search,
            vec_index: std::sync::Arc::clone(&old_state.vec_index),
        };
        db.try_publish_search_state(advanced).unwrap();

        let rid = db
            .record(
                "post-advance",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &vec_seed(0.5, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        let store = DurableEmbeddingStore::new(&db);
        let map = store.read_embeddings_for_rids(&[&rid]).unwrap();
        assert_eq!(map.get(&rid).unwrap().generation, 11);
    }

    #[test]
    fn read_embeddings_under_generation_filters_by_strict_less_than() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        // Record under gen 0.
        let rid_old = db
            .record(
                "old gen",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &vec_seed(0.1, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        // Advance gen to 5 and record one more row.
        let old_state = db.search_state.load_full();
        let advanced = crate::engine::reembed::SearchState {
            index_embedding: old_state.index_embedding.clone(),
            embedder: old_state.embedder.clone(),
            runtime_embedder_name: old_state.runtime_embedder_name.clone(),
            runtime_embedder_digest: old_state.runtime_embedder_digest.clone(),
            generation: 5,
            covers_through_seq: old_state.covers_through_seq,
            hnsw_m: old_state.hnsw_m,
            hnsw_ef_construction: old_state.hnsw_ef_construction,
            hnsw_ef_search: old_state.hnsw_ef_search,
            vec_index: std::sync::Arc::clone(&old_state.vec_index),
        };
        db.try_publish_search_state(advanced).unwrap();
        let rid_new = db
            .record(
                "new gen",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &vec_seed(0.2, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        let store = DurableEmbeddingStore::new(&db);
        // Asking for "rows under gen 5" returns rid_old only.
        let rows = store.read_embeddings_under_generation(5, 1000, 0).unwrap();
        let rids: Vec<&str> = rows.iter().map(|(r, _)| r.as_str()).collect();
        assert!(
            rids.contains(&rid_old.as_str()),
            "rid_old (gen 0 < 5) must appear"
        );
        assert!(
            !rids.contains(&rid_new.as_str()),
            "rid_new (gen 5 not strictly less than 5) must NOT appear"
        );

        // Asking for "rows under gen 6" returns both.
        let rows = store.read_embeddings_under_generation(6, 1000, 0).unwrap();
        let rids: Vec<&str> = rows.iter().map(|(r, _)| r.as_str()).collect();
        assert!(rids.contains(&rid_old.as_str()));
        assert!(rids.contains(&rid_new.as_str()));
    }

    /// **brainstorm-4 §5 / §10.7 boundary audit.** Grep-scans
    /// `engine/recall.rs` for raw `SELECT ... embedding FROM
    /// memories` patterns. A non-empty match means a future
    /// refactor reintroduced direct SQL embedding access in the
    /// hot recall path; the failure forces the refactor through
    /// this module's typed API.
    ///
    /// Limitation: regex match catches `embedding` and
    /// `embedding_*` (e.g. embedding_generation, embedding_new) —
    /// the test allowlist permits suffixed names; only bare
    /// `embedding` is flagged.
    #[test]
    fn recall_rs_has_no_raw_sql_embedding_reads() {
        let src = include_str!("recall.rs");

        let mut offenders: Vec<&str> = Vec::new();
        for line in src.lines() {
            let trimmed = line.trim();
            // Heuristic: a SELECT/FROM clause that names bare
            // `embedding` (no underscore suffix) as a column. The
            // patterns we want to forbid are: `SELECT embedding`,
            // `, embedding`, `SELECT rid, embedding`, and
            // similar shapes.
            //
            // Permitted: `embedding_hash`, `embedding_new`,
            // `embedding_model`, `embedding_generation` — those are
            // metadata columns that don't carry the durable vector
            // bytes.
            if !(trimmed.starts_with("//") || trimmed.starts_with("///")) {
                let lower = trimmed.to_ascii_lowercase();
                // A match must look like a SQL string-literal
                // fragment that names `embedding` followed by a
                // boundary (comma, space, end-quote, or FROM
                // keyword) — NOT followed by an underscore.
                for cand in [
                    "select embedding ",
                    "select embedding,",
                    "select embedding\"",
                    "select embedding\\",
                    ", embedding ",
                    ", embedding,",
                    ", embedding\"",
                    ", embedding\\",
                    ", embedding\n",
                ] {
                    if lower.contains(cand) {
                        offenders.push(line);
                        break;
                    }
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "brainstorm-4 §5 audit: recall.rs grew {} raw SQL embedding read(s); \
             route through engine::durable_embeddings::DurableEmbeddingStore. \
             Offending line(s):\n{}",
            offenders.len(),
            offenders.join("\n")
        );
    }
}
