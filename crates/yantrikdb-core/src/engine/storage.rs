use rusqlite::params;

use crate::error::Result;
use crate::serde_helpers::serialize_f32;

use super::{now, YantrikDB};

impl YantrikDB {
    /// Archive a hot memory to cold storage (compress embedding, remove from vec index).
    /// Returns true if the memory was archived, false if not found or already cold.
    #[tracing::instrument(skip(self))]
    pub fn archive(&self, rid: &str) -> Result<bool> {
        let ts = {
            let conn = self.conn();
            let row = conn.query_row(
                "SELECT embedding FROM memories WHERE rid = ?1 AND storage_tier = 'hot' AND consolidation_status = 'active'",
                params![rid],
                |row| row.get::<_, Vec<u8>>(0),
            );

            let stored_blob = match row {
                Ok(blob) => blob,
                Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(false),
                Err(e) => return Err(e.into()),
            };

            // Decrypt if encrypted, then compress, then re-encrypt for cold storage
            let raw_blob = self.decrypt_embedding(&stored_blob)?;
            let embedding = crate::serde_helpers::deserialize_f32(&raw_blob);
            let compressed = crate::compression::compress_embedding(&embedding);
            let stored_compressed = self.encrypt_embedding(&compressed)?;
            let ts = now();

            conn.execute(
                "UPDATE memories SET storage_tier = 'cold', embedding = ?1, updated_at = ?2 WHERE rid = ?3",
                params![stored_compressed, ts, rid],
            )?;

            ts
        }; // drop conn before acquiring vec_index write lock

        let seq = self
            .vec_seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        // **Issue #41 brainstorm-4 §1.** Snapshot through SearchState.
        self.search_state.load().vec_index.tombstone(rid, seq);

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
            let row = conn.query_row(
                "SELECT embedding FROM memories WHERE rid = ?1 AND storage_tier = 'cold'",
                params![rid],
                |row| row.get::<_, Vec<u8>>(0),
            );

            let stored_blob = match row {
                Ok(blob) => blob,
                Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(false),
                Err(e) => return Err(e.into()),
            };

            // Decrypt if encrypted, decompress, then re-encrypt for hot storage
            let compressed_blob = self.decrypt_embedding(&stored_blob)?;
            let embedding = crate::compression::decompress_embedding(&compressed_blob);
            let raw_blob = serialize_f32(&embedding);
            let stored_raw = self.encrypt_embedding(&raw_blob)?;
            let ts = now();

            conn.execute(
                "UPDATE memories SET storage_tier = 'hot', embedding = ?1, updated_at = ?2 WHERE rid = ?3",
                params![stored_raw, ts, rid],
            )?;

            (ts, embedding)
        }; // drop conn before acquiring vec_index write lock

        let seq = self
            .vec_seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        // **Issue #41 brainstorm-4 §1.** Hydration writes land on the
        // active-generation DeltaIndex via SearchState.
        self.search_state
            .load()
            .vec_index
            .append(rid.to_string(), embedding.clone(), seq)?;

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
    /// Lock ordering: takes `vec_index.write()` only — caller must NOT hold
    /// a `conn` guard across this call, per the engine-wide ordering rule
    /// (conn → … → vec_index).
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
        let seq = self
            .vec_seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        // **Issue #41 brainstorm-4 §1.** Replication-backfill writes
        // route through the active SearchState's DeltaIndex.
        self.search_state
            .load()
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
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut archived_rids = Vec::new();
        for (rid, _) in scored.into_iter().take(to_evict) {
            if self.archive(&rid)? {
                archived_rids.push(rid);
            }
        }

        Ok(archived_rids)
    }
}
