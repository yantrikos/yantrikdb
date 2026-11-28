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

/// **Issue #62.** `archive()` stores cold-tier embeddings zstd-compressed
/// in the SAME column hot rows use for raw f32 bytes. Every durable read
/// must therefore check the zstd magic and decompress (mirroring
/// `hydrate()`) — otherwise scoring paths reinterpret compressed bytes as
/// f32 garbage: wrong similarities always, and a chance NaN that poisons
/// the whole response's confidence. Cheap: a 4-byte magic comparison on
/// the hot path; decompression only ever runs for genuinely cold rows.
fn decompress_if_cold(bytes: Vec<u8>) -> Vec<u8> {
    if crate::compression::is_compressed(&bytes) {
        crate::serde_helpers::serialize_f32(&crate::compression::decompress_embedding(&bytes))
    } else {
        bytes
    }
}

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
    /// has a non-NULL embedding column; rids without one (tombstoned
    /// rows, or rids that don't exist) are omitted from the result map.
    ///
    /// **Issue #62:** cold-tier rows DO have an embedding — `archive()`
    /// rewrites the column with a zstd-COMPRESSED blob (an earlier version
    /// of this doc claimed cold rows were omitted; they never were). This
    /// reader therefore decompresses after decrypting, exactly mirroring
    /// `hydrate()` — before this fix, every recall scoring path that
    /// touched a cold record reinterpreted compressed bytes as raw f32,
    /// producing garbage similarity scores (and, ~50-70% of the time per
    /// record, a NaN that poisoned the response's confidence field).
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
        // SQLite binds at most SQLITE_LIMIT_VARIABLE_NUMBER parameters per
        // statement (32_766 on current builds). The caller on a recall path
        // passes the whole scoped candidate set — on a store above that size
        // one `IN (?1,…,?N)` is not executable at all: the read fails with
        // "variable number must be between ?1 and ?32766" instead of returning
        // rows, and the whole recall errors out. Split the lookup into chunks
        // the same way the supersedes lookup in `recall.rs` and the pack
        // reader already do (`engine::pack::SQL_IN_CHUNK`).
        //
        // This is a pure split of one lookup, not a change of semantics: the
        // set of rows read, their content, the absent-rid rule (no map entry)
        // and the caller's generation policy below are all unchanged. The map
        // is keyed by rid, so chunk order cannot influence the result.
        let mut map = HashMap::with_capacity(rids.len());
        let conn = self.db.read_conn();
        for chunk in rids.chunks(crate::engine::pack::SQL_IN_CHUNK) {
            let placeholders: String = (0..chunk.len())
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
            for r in chunk {
                param_values.push(Box::new(r.to_string()));
            }
            let params_ref: Vec<&dyn rusqlite::types::ToSql> =
                param_values.iter().map(|p| p.as_ref()).collect();

            let mut stmt = conn.prepare(&sql)?;
            // `embedding` is read as Option: a NULL column (e.g. a replication
            // follower materialized from record ops, whose oplog carries only
            // the embedding HASH — replication.rs materialize_op note) means
            // "this rid has no durable vector yet". Before #149 phase 2 that
            // row was unreachable by every caller of this reader, so the NULL
            // was never observed; the valid-time universe lane now reads
            // eligible rows the vector index has never seen, and one NULL must
            // mean an absent map key (the miss every caller already handles),
            // not a hard error that fails the whole batched read.
            let rows: Vec<(String, Option<Vec<u8>>, i64)> = stmt
                .query_map(params_ref.as_slice(), |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<Vec<u8>>>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            drop(stmt);

            for (rid, stored_emb, generation) in rows {
                let Some(stored_emb) = stored_emb else {
                    continue;
                };
                // Decrypt under the engine's at-rest encryption (no-op
                // if encryption is disabled). The brainstorm-4 §5
                // contract: this module is the sole caller of
                // decrypt_embedding for durable rows on read paths
                // (record/storage are write paths and stay separate).
                let bytes = self.db.decrypt_embedding(&stored_emb)?;
                let bytes = decompress_if_cold(bytes);
                map.insert(
                    rid,
                    EmbeddingWithGeneration {
                        bytes,
                        generation: generation as u64,
                    },
                );
            }
        }
        drop(conn);
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
            // Issue #62: see read_embeddings_for_rids — cold rows are
            // compressed at rest and must be decompressed on read.
            let bytes = decompress_if_cold(bytes);
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

/// Regression: the candidate set of a recall can exceed SQLite's bind limit.
///
/// `read_embeddings_for_rids` used to emit ONE statement with one placeholder
/// per rid. Above SQLITE_LIMIT_VARIABLE_NUMBER (32_766 on the bundled build)
/// that statement cannot be prepared at all, so a whole recall over a store of
/// that size failed with `variable number must be between ?1 and ?32766`
/// instead of returning rows. These tests pin the three sizes around the
/// limit: one below, exactly at it, one above.
#[cfg(test)]
mod chunking_tests {
    use super::*;
    use rusqlite::{params, Connection};

    /// Seed `n` rows outside the engine: 33k `record()` calls would take
    /// minutes, and the property under test is the bind count of the reader,
    /// not the write path.
    fn seed_bulk(path: &std::path::Path, n: usize, dim: usize) -> Vec<String> {
        let conn = Connection::open(path).unwrap();
        let blob = vec![0u8; dim * 4];
        conn.execute(
            "INSERT INTO memories (rid, text, type, created_at, updated_at, last_access, embedding, namespace)
             WITH RECURSIVE seq(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM seq WHERE i < ?1)
             SELECT printf('chunk-test-%06d', i), printf('row %d', i), 'episodic',
                    1.0, 1.0, 1.0, ?2, 'default' FROM seq",
            params![n as i64, blob],
        )
        .unwrap();
        // The read under test is reached from the graph lane, whose candidate
        // list is the records sharing an entity. Without these rows the
        // candidate set stays tiny and the bind limit is never approached —
        // the test would pass on the unfixed code.
        conn.execute(
            "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name) \
             SELECT rid, 'shared-entity' FROM memories",
            [],
        )
        .unwrap();
        (1..=n).map(|i| format!("chunk-test-{i:06}")).collect()
    }

    fn boundary_case(n: usize) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.db");
        // Create the schema through the engine, then release the file before
        // the bulk insert.
        drop(YantrikDB::new(path.to_str().unwrap(), 8).unwrap());
        let rids = seed_bulk(&path, n, 8);
        let db = YantrikDB::new(path.to_str().unwrap(), 8).unwrap();
        let store = DurableEmbeddingStore::new(&db);
        let refs: Vec<&str> = rids.iter().map(|s| s.as_str()).collect();
        let map = store.read_embeddings_for_rids(&refs).unwrap();
        assert_eq!(map.len(), n, "every seeded row must come back at {n} rids");
    }

    // Opt-in: seeds 33k rows and pays a full open per case (minutes in a
    // debug build). Run with `cargo test --lib chunking_tests -- --ignored`.
    #[test]
    #[ignore = "seeds 33k rows; minutes in a debug build"]
    fn one_below_the_variable_limit() {
        boundary_case(32_765);
    }

    // Opt-in: seeds 33k rows and pays a full open per case (minutes in a
    // debug build). Run with `cargo test --lib chunking_tests -- --ignored`.
    #[test]
    #[ignore = "seeds 33k rows; minutes in a debug build"]
    fn exactly_at_the_variable_limit() {
        boundary_case(32_766);
    }

    // Opt-in: seeds 33k rows and pays a full open per case (minutes in a
    // debug build). Run with `cargo test --lib chunking_tests -- --ignored`.
    #[test]
    #[ignore = "seeds 33k rows; minutes in a debug build"]
    fn one_above_the_variable_limit() {
        boundary_case(32_767);
    }

    /// The same guard for the ordinary case: a set spanning several chunks
    /// must come back complete, and a rid that does not exist stays absent
    /// instead of failing the whole read.
    #[test]
    fn chunk_boundary_keeps_every_row_and_the_absent_rid_rule() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.db");
        drop(YantrikDB::new(path.to_str().unwrap(), 8).unwrap());
        let n = crate::engine::pack::SQL_IN_CHUNK + 7;
        let mut rids = seed_bulk(&path, n, 8);
        rids.push("does-not-exist".to_string());
        let db = YantrikDB::new(path.to_str().unwrap(), 8).unwrap();
        let store = DurableEmbeddingStore::new(&db);
        let refs: Vec<&str> = rids.iter().map(|s| s.as_str()).collect();
        let map = store.read_embeddings_for_rids(&refs).unwrap();
        assert_eq!(map.len(), n, "the absent rid must not create an entry");
        assert!(!map.contains_key("does-not-exist"));
    }

    /// Discriminating and cheap: lower the connection's bind limit, then read more
    /// rids than that limit allows. A single `IN (...)` cannot be prepared (the old
    /// code), chunking can. `SQL_IN_CHUNK` is 500, so the lowered limit equals the
    /// chunk size and each statement binds exactly at the boundary.
    #[test]
    fn chunking_is_what_makes_an_oversized_read_possible() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.db");
        drop(YantrikDB::new(path.to_str().unwrap(), 8).unwrap());
        let n = crate::engine::pack::SQL_IN_CHUNK * 2 + 300;
        let rids = seed_bulk(&path, n, 8);
        let db = YantrikDB::new(path.to_str().unwrap(), 8).unwrap();
        let default = lower_bind_limit_on_every_read_connection(&db);
        // Premises, stated so a green run cannot come from the wrong reason:
        // without chunking, `n` rids fit into one statement on this build, and
        // with the limit lowered to `SQL_IN_CHUNK` they no longer do.
        assert!(
            default >= n as i32,
            "premise: {n} rids must fit the default bind limit ({default}) in one statement, \
             otherwise the unfixed reader would fail for a different reason"
        );
        assert!(
            n as i32 > crate::engine::pack::SQL_IN_CHUNK as i32,
            "premise: {n} rids must exceed the lowered limit ({})",
            crate::engine::pack::SQL_IN_CHUNK
        );
        let store = DurableEmbeddingStore::new(&db);
        let refs: Vec<&str> = rids.iter().map(|s| s.as_str()).collect();
        let map = store.read_embeddings_for_rids(&refs).unwrap();
        assert_eq!(
            map.len(),
            n,
            "{n} rids with a {}-parameter limit",
            crate::engine::pack::SQL_IN_CHUNK
        );
    }

    /// Lower `SQLITE_LIMIT_VARIABLE_NUMBER` to `SQL_IN_CHUNK` on **every**
    /// connection the reader can draw from, and return the build's default
    /// limit.
    ///
    /// `read_conn()` hands the pool out round-robin. An earlier version of the
    /// test above lowered the limit on the one connection its own `read_conn()`
    /// call returned; the reader then took the *next* slot, still at the
    /// default limit, and 1,300 rids went through a single statement with the
    /// unfixed reader — the test was green on `main` and proved nothing. Since
    /// `set_limit` persists on the connection, taking every slot once and
    /// releasing it again is enough.
    fn lower_bind_limit_on_every_read_connection(db: &YantrikDB) -> i32 {
        use rusqlite::limits::Limit;
        let lowered = crate::engine::pack::SQL_IN_CHUNK as i32;
        let slots = db.read_conns.len();
        let mut guards = Vec::with_capacity(slots.max(1));
        // An empty pool (`YANTRIKDB_READ_POOL=0`) falls back to the write
        // connection, which is then the one connection the reader uses.
        for _ in 0..slots.max(1) {
            let conn = db.read_conn();
            let default = conn.limit(Limit::SQLITE_LIMIT_VARIABLE_NUMBER).unwrap();
            conn.set_limit(Limit::SQLITE_LIMIT_VARIABLE_NUMBER, lowered)
                .unwrap();
            guards.push((conn, default));
        }
        let default = guards[0].1;
        for (conn, _) in &guards {
            assert_eq!(
                conn.limit(Limit::SQLITE_LIMIT_VARIABLE_NUMBER).unwrap(),
                lowered,
                "every pooled connection must carry the lowered limit, \
                 otherwise the reader may use one that does not"
            );
        }
        default
    }
}
