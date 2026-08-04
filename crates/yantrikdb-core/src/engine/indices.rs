use crate::encryption::EncryptionProvider;
use crate::error::Result;
use crate::hnsw::HnswIndex;
use crate::serde_helpers::deserialize_f32;

use super::YantrikDB;

impl YantrikDB {
    /// Build the HNSW vector index, optionally decrypting embeddings.
    pub(crate) fn build_vec_index_with_enc(
        conn: &rusqlite::Connection,
        embedding_dim: usize,
        enc: Option<&EncryptionProvider>,
    ) -> Result<HnswIndex> {
        let mut index = HnswIndex::new(embedding_dim);
        let mut stmt = conn.prepare(
            // v0.10 Phase 0 determinism seam: ORDER BY rid so the rebuild
            // inserts rows in a stable order — without it, reopening the
            // same DB could produce a different HNSW graph (and therefore
            // different approximate result sets) from unordered scans.
            "SELECT rid, embedding FROM memories \
             WHERE consolidation_status IN ('active', 'consolidated') \
             AND storage_tier = 'hot' \
             AND embedding IS NOT NULL \
             ORDER BY rid",
        )?;
        let rows = stmt.query_map([], |row| {
            let rid: String = row.get(0)?;
            let emb_blob: Vec<u8> = row.get(1)?;
            Ok((rid, emb_blob))
        })?;
        for row in rows {
            let (rid, emb_blob) = row?;
            let raw_blob = if let Some(e) = enc {
                e.decrypt_bytes(&emb_blob)?
            } else {
                emb_blob
            };
            // Issue #62 defense: this scan is hot-tier-only, but if a
            // compressed (cold-format) blob ever appears here — tier-column
            // drift, partial hydrate, manual SQL — decompress instead of
            // building the index from reinterpreted zstd bytes. One 4-byte
            // magic check per row.
            let embedding = if crate::compression::is_compressed(&raw_blob) {
                crate::compression::decompress_embedding(&raw_blob)
            } else {
                deserialize_f32(&raw_blob)
            };
            if embedding.len() == embedding_dim {
                index.insert(&rid, &embedding)?;
            }
        }
        // Chunked embeddings: window vectors for long records, indexed
        // under synthetic '{rid}#c{idx}' keys. Without this loop, every
        // reopen / rebuild / reembed / pack mount would silently drop
        // them — recall quality would differ before and after a restart,
        // which is the stored-active-unfindable failure class again.
        //
        // The join carries the parent's status/tier filters so a cold or
        // tombstoned parent's windows never reappear here, and the probe
        // for the table itself tolerates packs sealed by pre-chunk
        // engines (structural vetting does not enumerate tables, and a
        // mounted pack is read-only, so the table cannot be created on
        // the fly).
        let have_chunks: bool = conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_chunks'",
                [],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if have_chunks {
            let mut stmt = conn.prepare(
                "SELECT c.rid, c.chunk_idx, c.embedding \
                 FROM memory_chunks c JOIN memories m ON m.rid = c.rid \
                 WHERE m.consolidation_status IN ('active', 'consolidated') \
                 AND m.storage_tier = 'hot' \
                 ORDER BY c.rid, c.chunk_idx",
            )?;
            let rows = stmt.query_map([], |row| {
                let rid: String = row.get(0)?;
                let idx: i64 = row.get(1)?;
                let emb_blob: Vec<u8> = row.get(2)?;
                Ok((rid, idx, emb_blob))
            })?;
            for row in rows {
                let (rid, idx, emb_blob) = row?;
                let raw_blob = if let Some(e) = enc {
                    e.decrypt_bytes(&emb_blob)?
                } else {
                    emb_blob
                };
                let embedding = if crate::compression::is_compressed(&raw_blob) {
                    crate::compression::decompress_embedding(&raw_blob)
                } else {
                    deserialize_f32(&raw_blob)
                };
                if embedding.len() == embedding_dim && idx >= 1 {
                    let key = crate::vector::chunk::chunk_key(&rid, idx as usize);
                    index.insert(&key, &embedding)?;
                }
            }
        }
        // Distance-only pruning can leave a node with no incoming layer-0
        // edges — stored, active, and unfindable by any search. Found
        // live: a mounted 65-record pack lost a different record per
        // mount. Every bulk build ends with the connectivity repair so
        // the guarantee holds at the one point all rebuild paths share.
        let rescued = index.ensure_all_reachable();
        if rescued > 0 {
            tracing::warn!(rescued, "vec index rebuild reconnected unreachable nodes");
        }
        Ok(index)
    }

    /// Build without encryption (backward-compatible helper).
    pub(crate) fn build_vec_index(
        conn: &rusqlite::Connection,
        embedding_dim: usize,
    ) -> Result<HnswIndex> {
        Self::build_vec_index_with_enc(conn, embedding_dim, None)
    }

    /// Rebuild the HNSW vector index from scratch. Called after replication.
    pub fn rebuild_vec_index(&self) -> Result<usize> {
        let conn = self.conn.lock();
        let new_index =
            Self::build_vec_index_with_enc(&conn, self.embedding_dim, self.enc.as_ref())?;
        let count = new_index.len();
        drop(conn);
        // **Issue #41 brainstorm-4 §1.** Install the rebuilt cold tier
        // into the active-generation DeltaIndex via SearchState.
        self.search_state.load().vec_index.install_cold(new_index);
        Ok(count)
    }

    pub fn rebuild_graph_index(&self) -> Result<usize> {
        let conn = self.conn.lock();
        let new_index = crate::graph_index::GraphIndex::build_from_db(&conn)?;
        let count = new_index.entity_count();
        drop(conn);
        *self.graph_index.write() = new_index;
        Ok(count)
    }
}
