//! Corpus repair for leaked tool-call serialization artifacts.
//!
//! Task 29 ([`super::sanitize`]) stops *new* artifacts at the write boundary.
//! This module repairs the rows already corrupted before that fix existed —
//! the ≥200 memories the 2026-06-10 ingest audit found carrying a leaked
//! tool-call tail in their `text`.
//!
//! For each affected memory the text is cleaned with the exact same sanitizer
//! used on the write path, the cleaned text is RE-EMBEDDED under the active
//! embedder (so the vector matches the real content, not the artifact), and
//! the row is updated in place. The vector index is rebuilt once at the end.
//!
//! ## Safety properties (per the 2026-06-09 wipe lesson: verify before mutate)
//!
//! - **Dry-run first.** With `dry_run = true` nothing is mutated; the report
//!   tells the operator exactly what *would* change. This is the intended
//!   first invocation.
//! - **Recoverable.** Before any row is changed, its original (still
//!   encrypted) text is written to an `artifact_repair_audit` table in the
//!   SAME transaction as the update, so every change is reversible.
//! - **Concurrency-guarded.** The UPDATE matches on the exact stored
//!   ciphertext that was scanned (`WHERE rid = ? AND text = ?`). If a
//!   concurrent write changed the row between scan and apply, the update is a
//!   no-op and the memory is reported as skipped — never clobbered.
//! - **Fault-tolerant.** A failure on one memory is recorded in the report
//!   and repair continues; one bad row never blocks the rest.
//! - **Embedder required in apply mode.** Re-embedding clean text is the
//!   whole point, so apply mode fails fast if no embedder is configured
//!   rather than persisting clean text with a stale vector. Dry-run needs no
//!   embedder.

use rusqlite::params;

use crate::error::{Result, YantrikDbError};
use crate::serde_helpers::serialize_f32;

use super::{now, sanitize, YantrikDB};

/// Per-memory failure encountered during repair. Collected rather than
/// thrown so a single bad row never aborts the whole sweep.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RepairError {
    pub rid: String,
    pub message: String,
}

/// Outcome of a [`YantrikDB::repair_tool_call_artifacts`] sweep.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct RepairReport {
    /// Whether this was a dry run (no mutations performed).
    pub dry_run: bool,
    /// Total memories scanned (active + consolidated — the recall-visible set).
    pub scanned: usize,
    /// Memories found to contain a tool-call artifact.
    pub artifacts_found: usize,
    /// Memories actually repaired (cleaned + re-embedded + row updated).
    /// Always zero on a dry run.
    pub repaired: usize,
    /// Memories found dirty but whose row changed underneath us between scan
    /// and apply (concurrent write), so the guarded UPDATE was a no-op.
    pub skipped_concurrent_modification: usize,
    /// Total bytes stripped across all affected memories.
    pub stripped_bytes: usize,
    /// Per-memory errors. Repair continues past individual failures.
    pub errors: Vec<RepairError>,
    /// Sample of affected rids (capped) for operator spot-checking.
    pub sample_rids: Vec<String>,
}

/// Cap on how many rids the report echoes back, so a large sweep doesn't
/// return an unbounded payload.
const SAMPLE_CAP: usize = 50;

/// A pending in-place repair, computed outside the connection lock.
struct PendingRepair {
    rid: String,
    /// The exact stored ciphertext we scanned — used as the concurrency guard.
    original_ciphertext: String,
    /// The cleaned text, encrypted for storage.
    cleaned_ciphertext: String,
    /// The re-embedded vector, serialized + encrypted for storage.
    new_embedding_blob: Vec<u8>,
    stripped_bytes: usize,
}

impl YantrikDB {
    /// Repair memories whose stored `text` carries a leaked tool-call
    /// serialization artifact. See the module docs for the safety contract.
    ///
    /// Run with `dry_run = true` first to see the scope, then `false` to
    /// apply. Returns a [`RepairReport`] either way.
    pub fn repair_tool_call_artifacts(&self, dry_run: bool) -> Result<RepairReport> {
        let mut report = RepairReport {
            dry_run,
            ..Default::default()
        };

        // Apply mode re-embeds, which needs an embedder. Fail fast and clear
        // rather than persisting clean text against a stale vector.
        if !dry_run && !self.has_embedder() {
            return Err(YantrikDbError::NoEmbedder);
        }

        // ── Phase 1: scan. Hold the conn only for the read, then release it
        // before any (slow) embedding work. Matches the build_vec_index
        // visible set (active + consolidated) so text and index stay in step.
        let scanned_rows: Vec<(String, String)> = {
            let conn = self.conn();
            let mut stmt = conn.prepare(
                "SELECT rid, text FROM memories \
                 WHERE consolidation_status IN ('active', 'consolidated') \
                 ORDER BY rid",
            )?;
            let rows = stmt
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            rows
        };
        report.scanned = scanned_rows.len();

        // ── Phase 2: detect and, in apply mode, compute cleaned text + new
        // embedding OUTSIDE the conn lock (the embed is the slow step).
        let mut pending: Vec<PendingRepair> = Vec::new();
        for (rid, original_ciphertext) in scanned_rows {
            let plaintext = match self.decrypt_text(&original_ciphertext) {
                Ok(t) => t,
                Err(e) => {
                    report.errors.push(RepairError {
                        rid,
                        message: format!("decrypt failed: {e}"),
                    });
                    continue;
                }
            };
            if !sanitize::has_tool_call_artifact(&plaintext) {
                continue;
            }
            report.artifacts_found += 1;

            let cleaned = sanitize::sanitize_tool_call_artifacts(&plaintext);
            let stripped_bytes = plaintext.len().saturating_sub(cleaned.len());

            if report.sample_rids.len() < SAMPLE_CAP {
                report.sample_rids.push(rid.clone());
            }

            if dry_run {
                report.stripped_bytes += stripped_bytes;
                continue;
            }

            // Apply mode: re-embed the cleaned text and encrypt both fields.
            let embedding = match self.embed(&cleaned) {
                Ok(v) => v,
                Err(e) => {
                    report.errors.push(RepairError {
                        rid,
                        message: format!("re-embed failed: {e}"),
                    });
                    continue;
                }
            };
            let new_embedding_blob = match self.encrypt_embedding(&serialize_f32(&embedding)) {
                Ok(b) => b,
                Err(e) => {
                    report.errors.push(RepairError {
                        rid,
                        message: format!("encrypt embedding failed: {e}"),
                    });
                    continue;
                }
            };
            let cleaned_ciphertext = match self.encrypt_text(&cleaned) {
                Ok(c) => c,
                Err(e) => {
                    report.errors.push(RepairError {
                        rid,
                        message: format!("encrypt text failed: {e}"),
                    });
                    continue;
                }
            };
            pending.push(PendingRepair {
                rid,
                original_ciphertext,
                cleaned_ciphertext,
                new_embedding_blob,
                stripped_bytes,
            });
        }

        // Dry run, or nothing to do: report and return without mutating.
        if dry_run || pending.is_empty() {
            return Ok(report);
        }

        // ── Phase 3: apply all pending updates in one transaction. Each
        // memory's audit row and UPDATE commit together; the UPDATE is
        // concurrency-guarded on the scanned ciphertext.
        let generation = self.search_state.load_full().generation as i64;
        let repaired_at = now();
        {
            let conn = self.conn();
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS artifact_repair_audit (\
                   id INTEGER PRIMARY KEY AUTOINCREMENT, \
                   rid TEXT NOT NULL, \
                   original_text TEXT NOT NULL, \
                   cleaned_text TEXT NOT NULL, \
                   stripped_bytes INTEGER NOT NULL, \
                   repaired_at REAL NOT NULL)",
            )?;
            conn.execute_batch("SAVEPOINT artifact_repair")?;
            let apply: Result<()> = (|| {
                for p in &pending {
                    // Only update if the row still holds the exact ciphertext
                    // we scanned — otherwise a concurrent write changed it and
                    // we must not clobber it.
                    let updated = conn.execute(
                        "UPDATE memories \
                         SET text = ?1, embedding = ?2, embedding_generation = ?3, updated_at = ?4 \
                         WHERE rid = ?5 AND text = ?6",
                        params![
                            p.cleaned_ciphertext,
                            p.new_embedding_blob,
                            generation,
                            repaired_at,
                            p.rid,
                            p.original_ciphertext,
                        ],
                    )?;
                    if updated == 0 {
                        report.skipped_concurrent_modification += 1;
                        continue;
                    }
                    // Preserve the original for recoverability (same txn).
                    conn.execute(
                        "INSERT INTO artifact_repair_audit \
                         (rid, original_text, cleaned_text, stripped_bytes, repaired_at) \
                         VALUES (?1, ?2, ?3, ?4, ?5)",
                        params![
                            p.rid,
                            p.original_ciphertext,
                            p.cleaned_ciphertext,
                            p.stripped_bytes as i64,
                            repaired_at,
                        ],
                    )?;
                    report.repaired += 1;
                    report.stripped_bytes += p.stripped_bytes;
                }
                Ok(())
            })();
            match apply {
                Ok(()) => conn.execute_batch("RELEASE artifact_repair")?,
                Err(e) => {
                    let _ =
                        conn.execute_batch("ROLLBACK TO artifact_repair; RELEASE artifact_repair");
                    return Err(e);
                }
            }
        }

        // ── Phase 4: rebuild the vector index once so it reflects the updated
        // embeddings. rebuild_vec_index re-reads memories.embedding for every
        // row, so a single call after the transaction is correct and cheap
        // relative to per-row index surgery.
        if report.repaired > 0 {
            self.rebuild_vec_index()?;
        }

        tracing::info!(
            target: "yantrikdb::audit::repair",
            scanned = report.scanned,
            artifacts_found = report.artifacts_found,
            repaired = report.repaired,
            skipped_concurrent = report.skipped_concurrent_modification,
            stripped_bytes = report.stripped_bytes,
            errors = report.errors.len(),
            "corpus tool-call artifact repair complete",
        );

        Ok(report)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Same-space embedding refresh.
// ─────────────────────────────────────────────────────────────────────

/// Outcome of a [`YantrikDB::refresh_embeddings`] sweep.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct RefreshReport {
    /// Whether this was a dry run (no mutations performed).
    pub dry_run: bool,
    /// Namespace the sweep was limited to, or `None` for the whole store.
    pub namespace: Option<String>,
    /// Recall-visible rows examined (active + consolidated).
    pub scanned: usize,
    /// Rows that cannot participate in vector search under the active space:
    /// a NULL embedding, or a blob whose length is not `dim * 4` bytes.
    pub unusable_found: usize,
    /// Rows re-encoded and written back. Always zero on a dry run.
    pub refreshed: usize,
    /// Rows whose text changed between scan and apply — skipped, never
    /// clobbered.
    pub skipped_concurrent_modification: usize,
    /// Up to `SAMPLE_CAP` rids that were found unusable.
    pub sample_rids: Vec<String>,
    /// Per-row failures. One bad row never aborts the sweep.
    pub errors: Vec<RepairError>,
}

struct PendingRefresh {
    rid: String,
    scanned_ciphertext: String,
    new_embedding_blob: Vec<u8>,
}

impl YantrikDB {
    /// Re-encode records whose vectors are missing or unusable, using the
    /// embedder that is **already active**.
    ///
    /// # Why this is not a re-embed
    ///
    /// The embedding space belongs to the ENGINE, not to a namespace
    /// (architecture decision, 2026-08-17), so `reembed` — which exists to
    /// change that model — is necessarily store-wide and rejects a namespace
    /// scope outright. This is the operation callers usually want instead: it
    /// changes no model, no fingerprint, no dimension and no generation. It
    /// only fills in vectors that are absent or malformed, in the space the
    /// engine is already using.
    ///
    /// Because nothing about the space changes, none of the re-embed
    /// machinery applies: no staging columns, no cutover, no shadow index, no
    /// `SearchState` swap.
    ///
    /// # What it repairs
    ///
    /// A row is "unusable" when it is recall-visible but cannot participate
    /// in vector search: `embedding IS NULL`, or a blob whose length is not
    /// `dim * 4` bytes. The most important source of the first case is
    /// REPLICATION — the follower insert omits `embedding`,
    /// `embedding_model` and `embedding_generation` entirely, so replicated
    /// rows arrive active in SQL and invisible to semantic recall. This is
    /// the repair for them.
    ///
    /// # Safety
    ///
    /// Mirrors [`Self::repair_tool_call_artifacts`], the precedent for bulk
    /// same-space work: dry-run first, the slow embedding happens outside the
    /// connection lock, the UPDATE is concurrency-guarded on the exact
    /// ciphertext that was scanned, and one failing row is recorded rather
    /// than thrown. The index is rebuilt once at the end — and
    /// `rebuild_vec_index` now refuses to install across a re-embed cutover,
    /// so a refresh that races one is reported rather than mixed into the
    /// wrong space.
    pub fn refresh_embeddings(
        &self,
        namespace: Option<&str>,
        dry_run: bool,
    ) -> Result<RefreshReport> {
        let mut report = RefreshReport {
            dry_run,
            namespace: namespace.map(|s| s.to_string()),
            ..Default::default()
        };

        // Apply mode re-encodes; fail fast rather than reporting success
        // while leaving the rows exactly as unusable as they were.
        if !dry_run && !self.has_embedder() {
            return Err(YantrikDbError::NoEmbedder);
        }

        let expected_len = self.embedding_dim() * 4;

        // ── Phase 1: scan. Read-only, conn released before any embedding.
        // Matches build_vec_index's visible set so text and index stay in
        // step, exactly as the sibling repair does.
        let scanned: Vec<(String, String, Option<usize>)> = {
            let conn = self.conn();
            let sql = if namespace.is_some() {
                "SELECT rid, text, length(embedding) FROM memories \
                 WHERE consolidation_status IN ('active', 'consolidated') \
                   AND namespace = ?1 \
                 ORDER BY rid"
            } else {
                "SELECT rid, text, length(embedding) FROM memories \
                 WHERE consolidation_status IN ('active', 'consolidated') \
                 ORDER BY rid"
            };
            let mut stmt = conn.prepare(sql)?;
            fn map_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<(String, String, Option<usize>)> {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, Option<i64>>(2)?.map(|n| n as usize),
                ))
            }
            match namespace {
                Some(ns) => stmt
                    .query_map(params![ns], map_row)?
                    .collect::<std::result::Result<Vec<_>, _>>()?,
                None => stmt
                    .query_map([], map_row)?
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            }
        };
        report.scanned = scanned.len();

        // ── Phase 2: select the unusable rows and, in apply mode, encode
        // them OUTSIDE the conn lock.
        let mut pending: Vec<PendingRefresh> = Vec::new();
        for (rid, scanned_ciphertext, blob_len) in scanned {
            // Encrypted stores pad the blob, so a strict length check would
            // false-positive on every row; there, only a NULL vector is
            // provably unusable. Better to under-repair than to churn a
            // whole corpus on a bad predicate.
            let usable = match blob_len {
                None => false,
                Some(n) if self.is_encrypted() => n > 0,
                Some(n) => n == expected_len,
            };
            if usable {
                continue;
            }
            report.unusable_found += 1;
            if report.sample_rids.len() < SAMPLE_CAP {
                report.sample_rids.push(rid.clone());
            }
            if dry_run {
                continue;
            }

            let plaintext = match self.decrypt_text(&scanned_ciphertext) {
                Ok(t) => t,
                Err(e) => {
                    report.errors.push(RepairError {
                        rid,
                        message: format!("decrypt failed: {e}"),
                    });
                    continue;
                }
            };
            let embedding = match self.embed(&plaintext) {
                Ok(v) => v,
                Err(e) => {
                    report.errors.push(RepairError {
                        rid,
                        message: format!("embed failed: {e}"),
                    });
                    continue;
                }
            };
            let new_embedding_blob = match self.encrypt_embedding(&serialize_f32(&embedding)) {
                Ok(b) => b,
                Err(e) => {
                    report.errors.push(RepairError {
                        rid,
                        message: format!("encrypt embedding failed: {e}"),
                    });
                    continue;
                }
            };
            pending.push(PendingRefresh {
                rid,
                scanned_ciphertext,
                new_embedding_blob,
            });
        }

        if dry_run || pending.is_empty() {
            return Ok(report);
        }

        // ── Phase 3: apply in one transaction. The generation is READ, never
        // bumped: this is the same space the store was already in, and
        // advancing the store-wide watermark for a repair would make every
        // untouched row look stale.
        let state = self.search_state.load_full();
        let generation = state.generation as i64;
        let model = state.runtime_embedder_name.clone();
        let updated_at = now();
        {
            let conn = self.conn();
            conn.execute_batch("SAVEPOINT refresh_embeddings")?;
            let apply: Result<()> = (|| {
                for p in &pending {
                    // Guarded on the exact ciphertext scanned: if a
                    // concurrent write changed the text, the vector just
                    // computed describes content that no longer exists.
                    let updated = conn.execute(
                        "UPDATE memories \
                         SET embedding = ?1, embedding_model = ?2, \
                             embedding_generation = ?3, updated_at = ?4 \
                         WHERE rid = ?5 AND text = ?6",
                        params![
                            p.new_embedding_blob,
                            model,
                            generation,
                            updated_at,
                            p.rid,
                            p.scanned_ciphertext,
                        ],
                    )?;
                    if updated == 0 {
                        report.skipped_concurrent_modification += 1;
                        continue;
                    }
                    report.refreshed += 1;
                }
                Ok(())
            })();
            match apply {
                Ok(()) => conn.execute_batch("RELEASE refresh_embeddings")?,
                Err(e) => {
                    let _ = conn.execute_batch(
                        "ROLLBACK TO refresh_embeddings; RELEASE refresh_embeddings",
                    );
                    return Err(e);
                }
            }
        }

        // ── Phase 4: one rebuild, so the newly-filled vectors become
        // searchable. Refuses to install across a cutover.
        if report.refreshed > 0 {
            self.rebuild_vec_index()?;
        }

        tracing::info!(
            target: "yantrikdb::audit::repair",
            namespace = ?report.namespace,
            scanned = report.scanned,
            unusable_found = report.unusable_found,
            refreshed = report.refreshed,
            skipped_concurrent = report.skipped_concurrent_modification,
            errors = report.errors.len(),
            "same-space embedding refresh complete",
        );

        Ok(report)
    }
}
