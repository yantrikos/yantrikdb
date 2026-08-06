//! Chunked embeddings, engine side — the fix half of the silent-
//! truncation defect (`embedder_window.rs` is the detection half).
//!
//! When a record's text exceeds the embedder's known window, the write
//! path embeds overlapping windows of it (geometry in
//! `crate::vector::chunk`) and indexes each under `{rid}#c{idx}`; the
//! index's search collapses those keys back to the parent. Measured on
//! the corpus that surfaced the defect: MRR 0.057 → 0.395 (7×). See
//! docs/chunked_embeddings_design.md.
//!
//! This module owns:
//! - window persistence: the probed window survives restarts in `meta`,
//!   keyed to the embedder digest it was probed under (a different
//!   embedder has a different window);
//! - the chunk plan + chunk-embed helpers the write paths share;
//! - `rechunk_long_records`, the backfill that makes EXISTING corpora
//!   (the 73%-over-window production case) findable without rewriting
//!   every record.

use std::sync::atomic::Ordering;

use rusqlite::params;

use crate::error::Result;
use crate::vector::chunk;

use super::embedder_window::NO_TRUNCATION;
use super::YantrikDB;

/// meta key: the probed window, in chars (`"none"` = probed, no
/// truncation found). Only meaningful while [`META_WINDOW_DIGEST`]
/// matches the runtime embedder.
pub(crate) const META_WINDOW_CHARS: &str = "embedder_window_chars";
/// meta key: the embedder digest the window was probed under.
pub(crate) const META_WINDOW_DIGEST: &str = "embedder_window_digest";

/// A record's chunk vectors, ready for the reserve/insert/publish
/// protocol: `(chunk_idx, embedding)`, idx 1-based.
pub(crate) type ChunkVectors = Vec<(usize, Vec<f32>)>;

impl YantrikDB {
    /// Persist the probed window so a restart does not silently
    /// deactivate chunking. Best-effort: the window is an optimization
    /// fact, never a reason to fail the probe that found it.
    pub(crate) fn persist_embedder_window(&self, window: usize) {
        let state = self.search_state.load();
        let Some(digest) = state.runtime_embedder_digest.clone() else {
            // An embedder without identity cannot key a persisted
            // window — the next attach could be a different model.
            return;
        };
        let value = if window == NO_TRUNCATION {
            "none".to_string()
        } else {
            window.to_string()
        };
        let conn = self.conn.lock();
        let _ = conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            params![META_WINDOW_CHARS, value],
        );
        let _ = conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            params![META_WINDOW_DIGEST, digest],
        );
    }

    /// Sync the in-memory window with the CURRENT embedder: reset it
    /// to unprobed, then adopt the persisted window if (and only if)
    /// it was probed under this embedder's digest. Called after every
    /// embedder attach/swap — the reset half matters as much as the
    /// adopt half, because a window probed under the PREVIOUS embedder
    /// must never govern chunk geometry for the next one.
    pub(crate) fn adopt_persisted_window(&self) {
        // Reset first: from here the window is unknown until proven.
        self.embedder_window_chars.store(0, Ordering::Relaxed);
        let state = self.search_state.load();
        let Some(current) = state.runtime_embedder_digest.clone() else {
            return;
        };
        drop(state);
        let (stored_digest, stored_window) = {
            let conn = self.conn.lock();
            (
                Self::get_meta(&conn, META_WINDOW_DIGEST).ok().flatten(),
                Self::get_meta(&conn, META_WINDOW_CHARS).ok().flatten(),
            )
        };
        if stored_digest.as_deref() != Some(current.as_str()) {
            return;
        }
        let Some(w) = stored_window else { return };
        let value = if w == "none" {
            NO_TRUNCATION
        } else {
            match w.parse::<usize>() {
                Ok(n) if n > 0 => n,
                _ => return,
            }
        };
        self.embedder_window_chars.store(value, Ordering::Relaxed);
    }

    /// The chunk plan for `text`: byte ranges for windows 1.. when the
    /// embedder's window is known and the text overflows it; `None`
    /// when chunking does not apply (no window known, text fits, or
    /// geometry yields nothing).
    pub(crate) fn chunk_plan(&self, text: &str) -> Option<Vec<(usize, usize)>> {
        let window = match self.embedder_window_chars.load(Ordering::Relaxed) {
            0 | NO_TRUNCATION => return None,
            n => n,
        };
        if text.len() <= window {
            return None;
        }
        let ranges = chunk::chunk_ranges(text, window);
        if ranges.is_empty() {
            None
        } else {
            Some(ranges)
        }
    }

    /// The window used for snippet-span geometry (engine/snippet.rs):
    /// the probed embedder window when one is known, else the measured
    /// default. Snippets only need "a screenful of the right region",
    /// so an unprobed install still gets useful spans.
    pub(crate) fn snippet_window(&self) -> usize {
        match self.embedder_window_chars.load(Ordering::Relaxed) {
            0 | NO_TRUNCATION => super::snippet::DEFAULT_SNIPPET_WINDOW,
            n => n,
        }
    }

    /// Which of `rids` have chunk vectors — the set for which the
    /// vector layer's winning-window ordinal is meaningful. Infallible
    /// by design: an old database without the `memory_chunks` table
    /// (or any read error) yields the empty set, and span stamping
    /// falls back to the query-term scan.
    pub(crate) fn rids_with_chunks(&self, rids: &[&str]) -> std::collections::HashSet<String> {
        if rids.is_empty() {
            return Default::default();
        }
        let conn = self.read_conn();
        let placeholders = vec!["?"; rids.len()].join(",");
        let sql = format!("SELECT DISTINCT rid FROM memory_chunks WHERE rid IN ({placeholders})");
        let Ok(mut stmt) = conn.prepare_cached(&sql) else {
            return Default::default();
        };
        stmt.query_map(rusqlite::params_from_iter(rids.iter()), |row| {
            row.get::<_, String>(0)
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    /// Count a write that overflowed the window and was CHUNKED — the
    /// overflow is handled, not lost, so it must not inflate the
    /// truncation warning. Surfaced in `stats()`.
    pub(crate) fn note_chunked_write(&self) {
        self.embedder_chunked_writes.fetch_add(1, Ordering::Relaxed);
    }

    /// Writes since boot whose overflow was covered by chunk vectors.
    pub fn embedder_chunked_write_count(&self) -> u64 {
        self.embedder_chunked_writes.load(Ordering::Relaxed)
    }

    /// Remove a record's window vectors from index and table: tombstone
    /// every `{rid}#c{idx}` key at `seq` (the index matches keys by
    /// exact string — a parent tombstone does not cover its windows),
    /// then drop the rows so no rebuild re-derives them. The shared
    /// fan-out for forget, replicated forget, and conflict-loser
    /// suppression. Idempotent: no rows ⇒ no-op.
    pub(crate) fn purge_chunks(&self, rid: &str, seq: u64) -> Result<()> {
        let idxs: Vec<i64> = {
            let conn = self.conn.lock();
            let mut stmt = conn.prepare("SELECT chunk_idx FROM memory_chunks WHERE rid = ?1")?;
            let rows = stmt.query_map(params![rid], |r| r.get(0))?;
            rows.collect::<std::result::Result<_, _>>()?
        };
        if idxs.is_empty() {
            return Ok(());
        }
        for idx in &idxs {
            let key = chunk::chunk_key(rid, *idx as usize);
            self.search_state.load().vec_index.tombstone(&key, seq);
        }
        let conn = self.conn.lock();
        conn.execute("DELETE FROM memory_chunks WHERE rid = ?1", params![rid])?;
        Ok(())
    }

    /// Backfill chunk vectors for records written BEFORE chunking was
    /// active (or before the window was probed) — the deployment story
    /// for an existing corpus: probe the window, then call this once.
    ///
    /// For every active hot record with a vector whose (decrypted) text
    /// overflows the window and which has no chunk rows yet: embed its
    /// windows and insert the rows, guarded on the text being unchanged
    /// inside each row's transaction (the reembed staging-write
    /// pattern). The vectors reach the LIVE index via one
    /// `rebuild_vec_index` at the end rather than N delta appends — a
    /// bulk backfill through the delta would just churn backpressure.
    ///
    /// Idempotent: records that already have chunk rows are skipped, so
    /// an interrupted run continues where it stopped. Returns
    /// `(records_chunked, chunk_vectors_written)`.
    pub fn rechunk_long_records(&self) -> Result<(usize, usize)> {
        let window = match self.embedder_window_chars.load(Ordering::Relaxed) {
            0 | NO_TRUNCATION => return Ok((0, 0)),
            n => n,
        };
        let state = self.search_state.load_full();
        let embedder = state
            .embedder
            .as_ref()
            .ok_or(crate::error::YantrikDbError::NoEmbedder)?
            .clone();
        let generation = state.generation;
        let dim = state.dim();
        drop(state);

        // Candidates: active + hot + vectored, no chunk rows yet. Text
        // length cannot be filtered in SQL (it may be encrypted), so
        // every candidate is read and measured client-side.
        let candidates: Vec<(String, String)> = {
            let conn = self.conn.lock();
            let mut stmt = conn.prepare(
                "SELECT m.rid, m.text FROM memories m \
                 WHERE m.consolidation_status = 'active' \
                 AND m.storage_tier = 'hot' \
                 AND m.embedding IS NOT NULL \
                 AND NOT EXISTS (SELECT 1 FROM memory_chunks c WHERE c.rid = m.rid) \
                 ORDER BY m.rid",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            rows.collect::<std::result::Result<_, _>>()?
        };

        let mut records = 0usize;
        let mut vectors = 0usize;
        for (rid, stored_text) in candidates {
            let text = self.decrypt_text(&stored_text)?;
            let ranges = chunk::chunk_ranges(&text, window);
            if ranges.is_empty() {
                continue;
            }
            // Embed OUTSIDE the conn lock — the slow step. If a reembed
            // swap lands mid-run the generation check below stops the
            // backfill: the new embedder has a different (unprobed)
            // window and these vectors belong to the old space.
            if self.search_state.load().generation != generation {
                break;
            }
            let mut chunk_vecs: ChunkVectors = Vec::with_capacity(ranges.len());
            for (i, (a, b)) in ranges.iter().enumerate() {
                let v = embedder
                    .embed(&text[*a..*b])
                    .map_err(|e| crate::error::YantrikDbError::Inference(e.to_string()))?;
                crate::validate::validate_embedding("rechunk", &v, dim)?;
                chunk_vecs.push((i + 1, v));
            }
            // Commit this record's rows, guarded on the text we embedded
            // still being the text on disk (a concurrent correction wins).
            let conn = self.conn.lock();
            let tx = conn.unchecked_transaction()?;
            let current: Option<String> = tx
                .query_row(
                    "SELECT text FROM memories WHERE rid = ?1 \
                     AND consolidation_status = 'active' AND storage_tier = 'hot'",
                    params![rid],
                    |row| row.get(0),
                )
                .ok();
            let unchanged = match current {
                Some(ref stored) => self.decrypt_text(stored)? == text,
                None => false,
            };
            if unchanged {
                for (idx, v) in &chunk_vecs {
                    let blob = self.encrypt_embedding(&crate::serde_helpers::serialize_f32(v))?;
                    tx.execute(
                        "INSERT OR REPLACE INTO memory_chunks (rid, chunk_idx, embedding) \
                         VALUES (?1, ?2, ?3)",
                        params![rid, *idx as i64, blob],
                    )?;
                    vectors += 1;
                }
                tx.commit()?;
                records += 1;
            }
        }

        if records > 0 {
            // One rebuild carries every backfilled vector into the live
            // index (the rebuild loop reads memory_chunks).
            self.rebuild_vec_index()?;
        }
        Ok((records, vectors))
    }
}
