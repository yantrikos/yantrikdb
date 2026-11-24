use rusqlite::{params, OptionalExtension};

use crate::error::Result;
use crate::serde_helpers::serialize_f32;

use super::{now, YantrikDB};

impl YantrikDB {
    /// Archive a hot memory to cold storage (compress embedding, remove from vec index).
    /// Returns true if the memory was archived, false if not found or already cold.
    ///
    /// # KNOWN SPLIT-BRAIN (2026-08-15 surface audit — policy undecided)
    ///
    /// Archiving tombstones the VECTOR index entry but leaves the scoring
    /// cache untouched, so an archived record is unfindable by similarity
    /// yet still fully admissible to the FTS/keyword, claims, valence and
    /// importance-fallback lanes — "archive" means something different per
    /// lane. `forget()` by contrast removes the cache entry. If the intent
    /// is "cold but keyword-findable", that is a defensible policy — but it
    /// must be chosen, documented and tested, not inherited from whichever
    /// lane happened to read which structure. Also: no `visible_seq` bump,
    /// and the raw `vec_seq.fetch_add` below predates `assign_seq`. Do not
    /// "fix" any half of this in isolation — decide the policy first.
    #[tracing::instrument(skip(self))]
    pub fn archive(&self, rid: &str) -> Result<bool> {
        let (ts, had_vector) = {
            let conn = self.conn();
            // A row can have no vector: replication's follower insert omits
            // the embedding (see `refresh_embeddings`). It still archives;
            // there is just nothing to compress or tombstone.
            let row = conn.query_row(
                "SELECT embedding FROM memories WHERE rid = ?1 AND storage_tier = 'hot' AND consolidation_status = 'active'",
                params![rid],
                |row| row.get::<_, Option<Vec<u8>>>(0),
            );

            let stored_blob = match row {
                Ok(blob) => blob,
                Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(false),
                Err(e) => return Err(e.into()),
            };

            // Decrypt if encrypted, then compress, then re-encrypt for cold storage
            let stored_compressed = match &stored_blob {
                Some(blob) => {
                    let raw_blob = self.decrypt_embedding(blob)?;
                    let embedding = crate::serde_helpers::deserialize_f32(&raw_blob);
                    let compressed = crate::compression::compress_embedding(&embedding);
                    Some(self.encrypt_embedding(&compressed)?)
                }
                None => None,
            };
            let ts = now();

            conn.execute(
                "UPDATE memories SET storage_tier = 'cold', embedding = ?1, updated_at = ?2 WHERE rid = ?3",
                params![stored_compressed, ts, rid],
            )?;

            (ts, stored_blob.is_some())
        }; // drop conn before acquiring vec_index write lock

        // Chunked embeddings: window vectors follow the parent's tier —
        // compress the rows (the rebuild's is_compressed defense handles
        // either format, but cold means compressed by convention) and
        // remember the idxs so their index keys get tombstoned below.
        let chunk_idxs: Vec<i64> = {
            let conn = self.conn();
            let mut stmt =
                conn.prepare("SELECT chunk_idx, embedding FROM memory_chunks WHERE rid = ?1")?;
            let rows: Vec<(i64, Vec<u8>)> = stmt
                .query_map(params![rid], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<std::result::Result<_, _>>()?;
            let mut idxs = Vec::with_capacity(rows.len());
            for (idx, stored) in rows {
                let raw = self.decrypt_embedding(&stored)?;
                if !crate::compression::is_compressed(&raw) {
                    let emb = crate::serde_helpers::deserialize_f32(&raw);
                    let stored_c =
                        self.encrypt_embedding(&crate::compression::compress_embedding(&emb))?;
                    conn.execute(
                        "UPDATE memory_chunks SET embedding = ?1 WHERE rid = ?2 AND chunk_idx = ?3",
                        params![stored_c, rid, idx],
                    )?;
                }
                idxs.push(idx);
            }
            idxs
        };

        let seq = self
            .vec_seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        // **Issue #41 brainstorm-4 §1.** Snapshot through SearchState.
        //
        // Only a row with a vector is tombstoned. A row with none is not in
        // the index: no writer ever clears the vector of an indexed row
        // (every UPDATE of `embedding` stores a vector, archive/hydrate keep
        // a NULL as NULL, and the reembed swap is guarded on
        // `embedding_new IS NOT NULL`), and nothing indexes a row without
        // giving SQL its vector (`record_with_rid` on an existing row and
        // `insert_vector` fill a row's missing one). A tombstone for
        // it would outlive the row's return to the hot tier and hide the
        // vector a later
        // `refresh_embeddings` gives it, because the index rebuild that
        // follows a refresh does not touch the delta.
        if had_vector {
            self.search_state.load().vec_index.tombstone(rid, seq);
        }
        // Exact-string matching: the parent's tombstone does not cover
        // its window keys.
        for idx in &chunk_idxs {
            let key = crate::vector::chunk::chunk_key(rid, *idx as usize);
            self.search_state.load().vec_index.tombstone(&key, seq);
        }

        self.log_op(
            "archive",
            Some(rid),
            &serde_json::json!({
                "rid": rid,
                "updated_at": ts,
            }),
            None,
        )?;

        Ok(true)
    }

    /// Hydrate a cold memory back to hot storage (decompress embedding, re-insert into vec index).
    /// Returns true if the memory was hydrated, false if not found or already hot.
    #[tracing::instrument(skip(self))]
    pub fn hydrate(&self, rid: &str) -> Result<bool> {
        let (ts, embedding) = {
            let conn = self.conn();
            // No vector (a replicated row never repaired): hydrate the
            // tier and skip the index, as archive() does.
            let row = conn.query_row(
                "SELECT embedding FROM memories WHERE rid = ?1 AND storage_tier = 'cold'",
                params![rid],
                |row| row.get::<_, Option<Vec<u8>>>(0),
            );

            let stored_blob = match row {
                Ok(blob) => blob,
                Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(false),
                Err(e) => return Err(e.into()),
            };

            // Decrypt if encrypted, decompress, then re-encrypt for hot storage.
            // A cold row can hold an uncompressed blob: refresh_embeddings()
            // and a text correct() write hot format without knowing the
            // tier. Decompressing that panicked inside zstd; read it as is.
            let (embedding, stored_raw) = match &stored_blob {
                Some(blob) => {
                    let stored_plain = self.decrypt_embedding(blob)?;
                    let embedding = crate::compression::decode_embedding_blob(&stored_plain);
                    // Refuse before the UPDATE: a blob that does not decode
                    // to this store's dimension must not leave the row hot
                    // with an unusable vector and no index entry.
                    if embedding.len() != self.embedding_dim {
                        return Err(crate::error::YantrikDbError::EmbeddingDimensionMismatch {
                            expected: self.embedding_dim,
                            got: embedding.len(),
                        });
                    }
                    let raw_blob = serialize_f32(&embedding);
                    let stored_raw = self.encrypt_embedding(&raw_blob)?;
                    (Some(embedding), Some(stored_raw))
                }
                None => (None, None),
            };
            let ts = now();

            conn.execute(
                "UPDATE memories SET storage_tier = 'hot', embedding = ?1, updated_at = ?2 WHERE rid = ?3",
                params![stored_raw, ts, rid],
            )?;

            (ts, embedding)
        }; // drop conn before acquiring vec_index write lock

        // Chunked embeddings: decompress the window rows back to hot
        // format and collect their vectors for re-append.
        let chunk_vecs: Vec<(i64, Vec<f32>)> = {
            let conn = self.conn();
            let mut stmt =
                conn.prepare("SELECT chunk_idx, embedding FROM memory_chunks WHERE rid = ?1")?;
            let rows: Vec<(i64, Vec<u8>)> = stmt
                .query_map(params![rid], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<std::result::Result<_, _>>()?;
            let mut out = Vec::with_capacity(rows.len());
            for (idx, stored) in rows {
                let raw = self.decrypt_embedding(&stored)?;
                let emb = if crate::compression::is_compressed(&raw) {
                    let emb = crate::compression::decompress_embedding(&raw);
                    let stored_raw = self.encrypt_embedding(&serialize_f32(&emb))?;
                    conn.execute(
                        "UPDATE memory_chunks SET embedding = ?1 WHERE rid = ?2 AND chunk_idx = ?3",
                        params![stored_raw, rid, idx],
                    )?;
                    emb
                } else {
                    crate::serde_helpers::deserialize_f32(&raw)
                };
                out.push((idx, emb));
            }
            out
        };

        let seq = self
            .vec_seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        // **Issue #41 brainstorm-4 §1.** Hydration writes land on the
        // active-generation DeltaIndex via SearchState.
        if let Some(embedding) = &embedding {
            self.search_state
                .load()
                .vec_index
                .append(rid.to_string(), embedding.clone(), seq)?;
        }
        // Windows re-enter at the same seq, AFTER the parent (a chunk
        // hit collapses to the parent, which must be findable first).
        for (idx, emb) in &chunk_vecs {
            let key = crate::vector::chunk::chunk_key(rid, *idx as usize);
            self.search_state
                .load()
                .vec_index
                .append(key, emb.clone(), seq)?;
        }

        self.log_op(
            "hydrate",
            Some(rid),
            &serde_json::json!({
                "rid": rid,
                "updated_at": ts,
            }),
            None,
        )?;

        Ok(true)
    }

    /// Insert a single (rid, embedding) pair into the in-memory HNSW vector
    /// index. The matching SQLite memory row must already exist (this is a
    /// backfill helper for replication followers and similar callers that
    /// receive memory rows out-of-band of `record()` and need to bring the
    /// HNSW index up to date piecewise).
    ///
    /// Idempotent: re-inserting an rid that's already present in the index
    /// is a no-op (the underlying HNSW layer is responsible for de-duping).
    /// Errors propagate from the HNSW backend.
    ///
    /// If the row has no vector in SQL (a replicated row), the vector is
    /// written there too, compressed for a cold row, so a row in the index
    /// always has its vector in SQL; `archive()` relies on that. SQL is
    /// authoritative: a cold row is not indexed (it rejoins the index when
    /// hydrated), and neither is a vector that differs from the one the row
    /// already holds (it is logged and skipped; write the column first if
    /// the vector is meant to replace it). Refused
    /// with `VectorInsertDeferredDuringReembed` while a reembed cutover is
    /// in progress; retry after it.
    ///
    /// The vector is stamped with the generation active when this call
    /// runs. Pass one from the embedder active now: a vector computed
    /// before a cutover that has since completed is from the old space,
    /// and nothing here can tell.
    ///
    /// Lock ordering: takes the `conn` briefly for that write, then
    /// `vec_index.write()` — caller must NOT hold a `conn` guard across
    /// this call, per the engine-wide ordering rule (conn → … → vec_index).
    ///
    /// Added in yantrikdb 0.6.5 (RFC 022 §2): exposes the previously
    /// `pub(crate)` HNSW insert path so the server's replication backfill
    /// (`yantrikdb-server crates/yantrikdb-server/src/cluster/sync_loop.rs`)
    /// can populate followers' HNSW per-row instead of doing a full
    /// `rebuild_vec_index()` at the end of every batch. That rebuild was
    /// the cause of the multi-hour follower-recall lag reported by
    /// yantrikdb-agi 2026-05-01.
    #[tracing::instrument(skip(self, embedding), fields(rid = %rid))]
    pub fn insert_vector(&self, rid: &str, embedding: &[f32]) -> Result<()> {
        // v0.9.3 contract gate: this path feeds the index directly (HNSW
        // insert asserts on dim — a panic, not an error — and a NaN element
        // poisons distance math). Reject with a typed error instead, and do
        // it BEFORE the seq counter bumps so a rejected call is a no-op.
        crate::validate::validate_embedding("insert_vector", embedding, self.embedding_dim)?;
        // The generation read, the SQL fill and the index append happen
        // under the sync-writer guard, so a reembed cutover cannot land
        // between them and leave an old-space vector stamped as current.
        let Some(_sync_guard) = self.write_router.try_enter_sync_writer() else {
            return Err(
                crate::error::YantrikDbError::VectorInsertDeferredDuringReembed {
                    rid: rid.to_string(),
                },
            );
        };
        let state = self.search_state.load_full();
        // SQL is authoritative, and the index must agree with it:
        // - a row with no vector gets this one (compressed for a cold row),
        //   so a row in the index always has its vector in SQL; archive()
        //   relies on that to decide whether to tombstone;
        // - a cold row is never indexed (archive tombstoned it; it rejoins
        //   the index when hydrated);
        // - a hot row that already holds a vector is indexed only if it is
        //   this one. The server's backfill writes the column first, so
        //   there this is the whole cost of the check. A different vector
        //   would be a stray index entry that the next rebuild or reopen
        //   drops; it is not appended.
        // The seq is minted while the conn is still held: an archive() that
        // lands after this check takes the conn after us, so its tombstone
        // gets a later seq and outranks the append.
        let seq = {
            let conn = self.conn();
            let row: Option<(String, Option<Vec<u8>>)> = conn
                .query_row(
                    "SELECT storage_tier, embedding FROM memories WHERE rid = ?1 \
                     AND consolidation_status IN ('active', 'consolidated')",
                    params![rid],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()?;
            let forgotten = row.is_none()
                && conn
                    .query_row(
                        "SELECT 1 FROM memories WHERE rid = ?1",
                        params![rid],
                        |_| Ok(()),
                    )
                    .optional()?
                    .is_some();
            let index_it = match row {
                // A row that exists but is not active or consolidated
                // (forgotten, tombstoned) is not indexed: build_vec_index
                // would not index it either.
                None if forgotten => false,
                // No row: nothing to fill or check against (the documented
                // contract is that the row exists).
                None => true,
                Some((tier, None)) => {
                    let cold = tier == "cold";
                    let encoded = if cold {
                        crate::compression::compress_embedding(embedding)
                    } else {
                        serialize_f32(embedding)
                    };
                    let stored = self.encrypt_embedding(&encoded)?;
                    conn.execute(
                        "UPDATE memories SET embedding = ?1, embedding_generation = ?2 \
                         WHERE rid = ?3 AND embedding IS NULL AND storage_tier = ?4",
                        params![stored, state.generation as i64, rid, tier],
                    )?;
                    !cold
                }
                Some((tier, Some(_))) if tier == "cold" => false,
                Some((_, Some(blob))) => {
                    let held =
                        crate::compression::decode_embedding_blob(&self.decrypt_embedding(&blob)?);
                    if held.as_slice() == embedding {
                        true
                    } else {
                        tracing::warn!(
                            rid,
                            "insert_vector: SQL holds a different vector for this row; SQL is \
                             authoritative, so the index is left as it is"
                        );
                        false
                    }
                }
            };
            if !index_it {
                return Ok(());
            }
            self.vec_seq
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1
        };
        // **Issue #41 brainstorm-4 §1.** Replication-backfill writes
        // route through the active SearchState's DeltaIndex.
        state
            .vec_index
            .append(rid.to_string(), embedding.to_vec(), seq)
            .map(|_| ())
    }

    /// Encrypt an embedding blob if at-rest encryption is enabled on this
    /// engine; otherwise return the input unchanged. Used by replication
    /// followers' backfill path to encrypt locally-re-embedded vectors
    /// before writing them to the SQLite `embedding` column, so encrypted
    /// clusters maintain ciphertext-only persistence.
    ///
    /// Added in yantrikdb 0.6.5 (RFC 022 §2): public wrapper over the
    /// existing `pub(crate) encrypt_embedding`. Without this, the server's
    /// `backfill_embeddings()` could not encrypt vectors and skipped
    /// encrypted-cluster writes entirely (see TODO in sync_loop.rs).
    pub fn encrypt_embedding_pub(&self, emb_blob: &[u8]) -> Result<Vec<u8>> {
        self.encrypt_embedding(emb_blob)
    }

    /// Evict memories to cold storage based on decay scores.
    /// Archives the lowest-scoring memories until at most `max_active` hot memories remain.
    /// Returns the list of archived RIDs.
    #[tracing::instrument(skip(self))]
    pub fn evict(&self, max_active: usize) -> Result<Vec<String>> {
        let (mut scored, to_evict) = {
            let conn = self.conn();
            let hot_count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM memories WHERE consolidation_status = 'active' AND storage_tier = 'hot'",
                [],
                |row| row.get(0),
            )?;

            if hot_count as usize <= max_active {
                return Ok(vec![]);
            }

            let to_evict = hot_count as usize - max_active;
            let ts = now();

            let mut stmt = conn.prepare(
                "SELECT rid, importance, half_life, last_access, created_at, access_count \
                 FROM memories \
                 WHERE consolidation_status = 'active' AND storage_tier = 'hot'",
            )?;

            let scored: Vec<(String, f64)> = stmt
                .query_map([], |row| {
                    let rid: String = row.get("rid")?;
                    let importance: f64 = row.get("importance")?;
                    let half_life: f64 = row.get("half_life")?;
                    let last_access: f64 = row.get("last_access")?;
                    let created_at: f64 = row.get("created_at")?;
                    let access_count: i64 = row.get("access_count")?;
                    let elapsed = ts - last_access;
                    let decay = crate::scoring::decay_score(importance, half_life, elapsed);
                    let age = ts - created_at;
                    let recency = crate::scoring::recency_score(age);
                    // Recall frequency resists eviction (hot memories stay hot).
                    let score =
                        crate::scoring::eviction_score(decay, recency, access_count.max(0) as u32);
                    Ok((rid, score))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;

            (scored, to_evict)
        }; // drop conn before archive() which re-acquires it

        // Sort ascending — lowest score = most evictable
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));

        let mut archived_rids = Vec::new();
        for (rid, _) in scored.into_iter().take(to_evict) {
            if self.archive(&rid)? {
                archived_rids.push(rid);
            }
        }

        Ok(archived_rids)
    }
}
