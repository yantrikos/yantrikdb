//! v0.10 Item 2 — impression logging and typed ranking labels.
//!
//! The self-sufficient learning layer's data plane. Three rules from the
//! validity review govern everything here:
//!
//! 1. **Impressions are persisted with their feature values at serve
//!    time**, before reinforcement mutates `last_access`/`access_count`.
//!    The learner NEVER rebuilds historical features from current
//!    mutable state — that reconstruction is exposure-confounded.
//! 2. **Being served is not a label.** `recall()` returning a rid is
//!    the ranker's own decision; treating it (or its downstream
//!    reinforcement) as evidence teaches the model to reproduce itself.
//!    Labels come only from explicit feedback, explicit rejection, or
//!    an independent caller-initiated action targeting the rid.
//! 3. **One label per (episode, rid, source)** — enforced by the table's
//!    UNIQUE constraint; repeats are idempotent, not amplifying.
//!
//! On a read-only workload with none of those signals, the valid
//! outcome is that the database never learns ("abstain from learning
//! rather than teach itself that its own answers were correct").

use rusqlite::{params, OptionalExtension};

use crate::error::Result;
use crate::types::{
    RecallResult, RollupMembershipExample, RollupMembershipReport, RollupOutcomeExample,
    RollupOutcomeReport, RollupRankOutcomeStats,
};

use super::YantrikDB;

/// Label weight for explicit relevant/irrelevant feedback.
pub(crate) const WEIGHT_EXPLICIT: f64 = 1.0;
/// Label weight for an explicit rejection in a refine call.
pub(crate) const WEIGHT_REJECTED_REFINE: f64 = 0.5;
/// Label weight for an independent caller action targeting the rid
/// (the outcome anchor). Weak positive by design.
pub(crate) const WEIGHT_CALLER_USED: f64 = 0.3;

/// How far back a label may bind to an impression of its rid. Beyond
/// this, the action is treated as unrelated to any specific serving.
const LABEL_BINDING_HORIZON_SECS: f64 = 7.0 * 86_400.0;

/// Deterministic FNV-1a over the query embedding bytes — the
/// distinct-query-episode grouping key. Hand-rolled so the hash is
/// stable across runs, platforms, and std hasher changes.
pub(crate) fn query_hash(embedding: &[f32]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in embedding {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{h:016x}")
}

impl YantrikDB {
    /// Persist one recall call's served results as impression rows.
    /// Called from `recall()` after the final top_k is assembled and
    /// BEFORE reinforcement. Returns the episode id. Best-effort caller
    /// contract: recall treats a logging failure as fatal only in tests
    /// — the read path must not fail because the learner's ledger
    /// hiccuped (callers use `let _ =` and the learning loop's
    /// diagnostics expose the gap instead).
    pub(crate) fn log_recall_impressions(
        &self,
        results: &[RecallResult],
        query_embedding: &[f32],
        namespace: Option<&str>,
        weight_generation: i64,
    ) -> Result<String> {
        let episode_id = crate::id::new_id();
        if results.is_empty() {
            return Ok(episode_id);
        }
        let ts = crate::time::now_secs();
        let qhash = query_hash(query_embedding);
        let conn = self.conn();
        let mut stmt = conn.prepare_cached(
            "INSERT OR IGNORE INTO recall_impressions \
             (episode_id, rid, rank, f_similarity, f_decay, f_recency, f_importance, \
              f_valence, keyword_boosted, score, weight_generation, namespace, \
              query_hash, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        )?;
        for (rank, r) in results.iter().enumerate() {
            let keyword_boosted = r
                .why_retrieved
                .iter()
                .any(|w| w == "keyword_match" || w == "keyword_reserved");
            stmt.execute(params![
                episode_id,
                r.rid,
                rank as i64,
                r.scores.similarity,
                r.scores.decay,
                r.scores.recency,
                r.scores.importance,
                r.valence,
                keyword_boosted as i64,
                r.score,
                weight_generation,
                namespace,
                qhash,
                ts,
            ])?;
        }
        Ok(episode_id)
    }

    /// Most recent impression episode that served `rid` within the
    /// binding horizon, if any.
    fn latest_impression_episode(&self, rid: &str) -> Result<Option<String>> {
        use rusqlite::OptionalExtension;
        let horizon = crate::time::now_secs() - LABEL_BINDING_HORIZON_SECS;
        let conn = self.conn();
        let episode: Option<String> = conn
            .query_row(
                "SELECT episode_id FROM recall_impressions \
                 WHERE rid = ?1 AND created_at >= ?2 \
                 ORDER BY created_at DESC LIMIT 1",
                params![rid, horizon],
                |row| row.get(0),
            )
            .optional()?;
        Ok(episode)
    }

    /// Insert a typed ranking label bound to the most recent impression
    /// of `rid`. No impression in the horizon → no label (an action on
    /// a record the ranker never served says nothing about the ranker).
    /// Idempotent per (episode, rid, source). Returns whether a label
    /// was recorded.
    pub(crate) fn record_ranking_label(
        &self,
        rid: &str,
        source: &str,
        polarity: i32,
        weight: f64,
    ) -> Result<bool> {
        let Some(episode) = self.latest_impression_episode(rid)? else {
            return Ok(false);
        };
        let conn = self.conn();
        let inserted = conn.execute(
            "INSERT OR IGNORE INTO ranking_labels \
             (label_id, episode_id, rid, source, polarity, weight, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                crate::id::new_id(),
                episode,
                rid,
                source,
                polarity,
                weight,
                crate::time::now_secs(),
            ],
        )?;
        Ok(inserted > 0)
    }

    /// Record one rollup surfaced by a caller-side organization layer.
    /// `impression_id` is optional for convenience; callers that retry across
    /// a transport boundary should generate and reuse one.
    pub fn note_rollup_impression(
        &self,
        rollup_rid: &str,
        query_text: &str,
        namespace: Option<&str>,
        rank: usize,
        score: f64,
        impression_id: Option<&str>,
    ) -> Result<String> {
        self.note_rollup_impression_with_features(
            rollup_rid,
            query_text,
            namespace,
            rank,
            score,
            None,
            None,
            impression_id,
        )
    }

    /// Feature-aware rollup impression. Query shape is intentionally a small
    /// non-text vocabulary so telemetry can distinguish exact/list requests
    /// without persisting the query itself.
    #[allow(clippy::too_many_arguments)]
    pub fn note_rollup_impression_with_features(
        &self,
        rollup_rid: &str,
        query_text: &str,
        namespace: Option<&str>,
        rank: usize,
        score: f64,
        requested_count: Option<usize>,
        query_shape: Option<&str>,
        impression_id: Option<&str>,
    ) -> Result<String> {
        if rollup_rid.trim().is_empty() || query_text.trim().is_empty() || !score.is_finite() {
            return Err(crate::error::YantrikDbError::InvalidInput(
                "rollup impression requires non-empty rollup_rid/query_text and a finite score"
                    .to_string(),
            ));
        }
        if requested_count == Some(0) {
            return Err(crate::error::YantrikDbError::InvalidInput(
                "rollup impression requested_count must be positive".to_string(),
            ));
        }
        let query_shape = query_shape
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_ascii_lowercase);
        if query_shape.as_deref().is_some_and(|value| {
            !matches!(
                value,
                "point" | "list" | "ordered_list" | "summary" | "other"
            )
        }) {
            return Err(crate::error::YantrikDbError::InvalidInput(
                "rollup impression query_shape must be point, list, ordered_list, summary, or other"
                    .to_string(),
            ));
        }
        let impression_id = impression_id
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string)
            .unwrap_or_else(crate::id::new_id);
        let query_hash = stable_text_hash(query_text);
        let ts = crate::time::now_secs();
        let conn = self.conn();
        let stored_namespace: String = conn
            .query_row(
                "SELECT namespace FROM memories WHERE rid = ?1",
                params![rollup_rid],
                |row| row.get(0),
            )
            .optional()?
            .ok_or_else(|| {
                crate::error::YantrikDbError::NotFound(format!("rollup: {rollup_rid}"))
            })?;
        if namespace.is_some_and(|value| value != stored_namespace) {
            return Err(crate::error::YantrikDbError::InvalidInput(format!(
                "rollup {rollup_rid:?} belongs to namespace {stored_namespace:?}, not {namespace:?}"
            )));
        }
        conn.execute(
            "INSERT OR IGNORE INTO rollup_impressions \
             (impression_id, rollup_rid, query_hash, namespace, rank, score, \
              requested_count, query_shape, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                impression_id,
                rollup_rid,
                query_hash,
                stored_namespace,
                rank as i64,
                score,
                requested_count.map(|value| value as i64),
                query_shape,
                ts,
            ],
        )?;
        let existing: (
            String,
            String,
            String,
            i64,
            f64,
            Option<i64>,
            Option<String>,
        ) = conn.query_row(
            "SELECT rollup_rid, query_hash, namespace, rank, score, \
                    requested_count, query_shape \
             FROM rollup_impressions WHERE impression_id = ?1",
            params![impression_id],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                ))
            },
        )?;
        if existing
            != (
                rollup_rid.to_string(),
                query_hash,
                stored_namespace,
                rank as i64,
                score,
                requested_count.map(|value| value as i64),
                query_shape,
            )
        {
            return Err(crate::error::YantrikDbError::InvalidInput(format!(
                "impression_id {impression_id:?} was already used with a different payload"
            )));
        }
        Ok(impression_id)
    }

    /// Bind the exact ordered children returned by an expansion. Reusing an
    /// impression id with a different child payload is rejected.
    pub fn note_rollup_expansion(
        &self,
        impression_id: &str,
        returned_child_rids: &[&str],
    ) -> Result<usize> {
        let children: Vec<(&str, Option<f64>)> = returned_child_rids
            .iter()
            .copied()
            .map(|rid| (rid, None))
            .collect();
        self.note_rollup_expansion_with_scores(impression_id, &children)
    }

    /// Bind returned children together with their immutable serve-time scores.
    pub fn note_rollup_expansion_with_scores(
        &self,
        impression_id: &str,
        returned_children: &[(&str, Option<f64>)],
    ) -> Result<usize> {
        let mut clean: Vec<(String, Option<f64>)> = Vec::new();
        for (rid, score) in returned_children.iter().copied() {
            let rid = rid.trim();
            if score.is_some_and(|value| !value.is_finite()) {
                return Err(crate::error::YantrikDbError::InvalidInput(
                    "rollup expansion child scores must be finite".to_string(),
                ));
            }
            if !rid.is_empty() && !clean.iter().any(|(existing, _)| existing == rid) {
                clean.push((rid.to_string(), score));
            }
        }
        let payload = if clean.iter().all(|(_, score)| score.is_none()) {
            clean
                .iter()
                .map(|(rid, _)| rid.as_str())
                .collect::<Vec<_>>()
                .join("\u{0}")
        } else {
            clean
                .iter()
                .map(|(rid, score)| {
                    format!(
                        "{rid}:{}",
                        score
                            .map(|value| format!("{:016x}", value.to_bits()))
                            .unwrap_or_else(|| "none".to_string())
                    )
                })
                .collect::<Vec<_>>()
                .join("\u{0}")
        };
        let payload_hash = stable_text_hash(&payload);
        let conn = self.conn();
        let tx = conn.unchecked_transaction()?;
        let existing_hash: Option<String> = tx
            .query_row(
                "SELECT expansion_payload_hash FROM rollup_impressions \
                 WHERE impression_id = ?1",
                params![impression_id],
                |row| row.get(0),
            )
            .optional()?
            .ok_or_else(|| {
                crate::error::YantrikDbError::NotFound(format!(
                    "rollup impression: {impression_id}"
                ))
            })?;
        if existing_hash
            .as_deref()
            .is_some_and(|hash| hash != payload_hash)
        {
            return Err(crate::error::YantrikDbError::InvalidInput(format!(
                "rollup impression {impression_id:?} was expanded with a different child payload"
            )));
        }
        tx.execute(
            "UPDATE rollup_impressions SET expansion_payload_hash = ?2, \
             expanded_at = COALESCE(expanded_at, ?3) WHERE impression_id = ?1",
            params![impression_id, payload_hash, crate::time::now_secs()],
        )?;
        let mut stmt = tx.prepare_cached(
            "INSERT OR IGNORE INTO rollup_impression_children \
             (impression_id, child_rid, rank, score) VALUES (?1, ?2, ?3, ?4)",
        )?;
        for (rank, (child_rid, score)) in clean.iter().enumerate() {
            stmt.execute(params![impression_id, child_rid, rank as i64, score])?;
        }
        drop(stmt);
        tx.commit()?;
        Ok(clean.len())
    }

    /// Explicitly record that a caller selected or corrected a returned child.
    pub fn note_rollup_selection(
        &self,
        impression_id: &str,
        child_rid: &str,
        source: &str,
    ) -> Result<bool> {
        if !matches!(source, "selected" | "corrected") {
            return Err(crate::error::YantrikDbError::InvalidInput(
                "rollup selection source must be 'selected' or 'corrected'".to_string(),
            ));
        }
        let conn = self.conn();
        let finalized: Option<Option<f64>> = conn
            .query_row(
                "SELECT outcome_finalized_at FROM rollup_impressions \
                 WHERE impression_id = ?1",
                params![impression_id],
                |row| row.get(0),
            )
            .optional()?;
        let Some(finalized) = finalized else {
            return Ok(false);
        };
        if finalized.is_some() {
            let exists: bool = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM rollup_impression_outcomes \
                 WHERE impression_id = ?1 AND child_rid = ?2 AND source = ?3)",
                params![impression_id, child_rid, source],
                |row| row.get(0),
            )?;
            if exists {
                return Ok(false);
            }
            return Err(crate::error::YantrikDbError::InvalidInput(format!(
                "rollup impression {impression_id:?} already has a finalized outcome"
            )));
        }
        let inserted = conn.execute(
            "INSERT OR IGNORE INTO rollup_impression_outcomes \
             (outcome_id, impression_id, child_rid, source, created_at) \
             SELECT ?1, ?2, ?3, ?4, ?5 \
             WHERE EXISTS (SELECT 1 FROM rollup_impression_children \
                           WHERE impression_id = ?2 AND child_rid = ?3)",
            params![
                crate::id::new_id(),
                impression_id,
                child_rid,
                source,
                crate::time::now_secs(),
            ],
        )?;
        Ok(inserted > 0)
    }

    /// Close the telemetry loop with the complete set of children used by the
    /// caller. Only after this call may omitted returned children be treated as
    /// explicit non-selections in offline measurement.
    pub fn finalize_rollup_outcome(
        &self,
        impression_id: &str,
        selected_child_rids: &[&str],
        corrected_child_rids: &[&str],
    ) -> Result<usize> {
        self.finalize_rollup_outcome_with_omissions(
            impression_id,
            selected_child_rids,
            corrected_child_rids,
            &[],
        )
    }

    /// Finalize an exact outcome while preserving caller-added omissions as a
    /// separate class. Omitted children are explicit false-negative evidence;
    /// they are never inserted into the served expansion.
    pub fn finalize_rollup_outcome_with_omissions(
        &self,
        impression_id: &str,
        selected_child_rids: &[&str],
        corrected_child_rids: &[&str],
        added_child_rids: &[&str],
    ) -> Result<usize> {
        let mut selected: Vec<String> = selected_child_rids
            .iter()
            .map(|rid| rid.trim())
            .filter(|rid| !rid.is_empty())
            .map(str::to_string)
            .collect();
        let corrected: Vec<String> = corrected_child_rids
            .iter()
            .map(|rid| rid.trim())
            .filter(|rid| !rid.is_empty())
            .map(str::to_string)
            .collect();
        selected.extend(corrected.iter().cloned());
        selected.sort();
        selected.dedup();
        let mut corrected = corrected;
        corrected.sort();
        corrected.dedup();
        let mut added: Vec<String> = added_child_rids
            .iter()
            .map(|rid| rid.trim())
            .filter(|rid| !rid.is_empty())
            .map(str::to_string)
            .collect();
        added.sort();
        added.dedup();
        if added.iter().any(|rid| selected.contains(rid)) {
            return Err(crate::error::YantrikDbError::InvalidInput(
                "a rollup child cannot be both returned-selected and caller-omitted".to_string(),
            ));
        }
        let legacy_payload = format!(
            "selected:{}\u{0}corrected:{}",
            selected.join("\u{0}"),
            corrected.join("\u{0}")
        );
        let payload = if added.is_empty() {
            legacy_payload
        } else {
            format!("{legacy_payload}\u{0}omitted:{}", added.join("\u{0}"))
        };
        let payload_hash = stable_text_hash(&payload);

        let conn = self.conn();
        let tx = conn.unchecked_transaction()?;
        let state: Option<(Option<String>, Option<f64>, String, f64)> = tx
            .query_row(
                "SELECT outcome_payload_hash, expanded_at, namespace, created_at \
                 FROM rollup_impressions \
                 WHERE impression_id = ?1",
                params![impression_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;
        let Some((existing_hash, expanded_at, namespace, impression_created_at)) = state else {
            return Err(crate::error::YantrikDbError::NotFound(format!(
                "rollup impression: {impression_id}"
            )));
        };
        if expanded_at.is_none() {
            return Err(crate::error::YantrikDbError::InvalidInput(format!(
                "rollup impression {impression_id:?} has not been expanded"
            )));
        }
        if existing_hash
            .as_deref()
            .is_some_and(|hash| hash != payload_hash)
        {
            return Err(crate::error::YantrikDbError::InvalidInput(format!(
                "rollup impression {impression_id:?} was finalized with a different outcome"
            )));
        }
        if existing_hash.as_deref() == Some(payload_hash.as_str()) {
            return Ok(selected.len() + added.len());
        }

        for child_rid in &selected {
            let returned: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM rollup_impression_children \
                 WHERE impression_id = ?1 AND child_rid = ?2)",
                params![impression_id, child_rid],
                |row| row.get(0),
            )?;
            if !returned {
                return Err(crate::error::YantrikDbError::InvalidInput(format!(
                    "child {child_rid:?} was not returned by rollup impression {impression_id:?}"
                )));
            }
        }
        for child_rid in &added {
            let returned: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM rollup_impression_children \
                 WHERE impression_id = ?1 AND child_rid = ?2)",
                params![impression_id, child_rid],
                |row| row.get(0),
            )?;
            if returned {
                return Err(crate::error::YantrikDbError::InvalidInput(format!(
                    "omitted child {child_rid:?} was already returned by rollup impression {impression_id:?}"
                )));
            }
            let child_state: Option<(String, String, Option<f64>)> = tx
                .query_row(
                    "SELECT m.namespace, m.consolidation_status, \
                            COALESCE( \
                                (SELECT MIN(r.applied_at) FROM replication_apply_log r \
                                 WHERE r.rid = m.rid), \
                                (SELECT MIN(o.timestamp) FROM oplog o \
                                 WHERE o.target_rid = m.rid \
                                   AND o.op_type IN ('record', 'record_with_rid')) \
                            ) \
                     FROM memories m WHERE m.rid = ?1",
                    params![child_rid],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .optional()?;
            let Some((child_namespace, child_status, available_at)) = child_state else {
                return Err(crate::error::YantrikDbError::NotFound(format!(
                    "omitted rollup child: {child_rid}"
                )));
            };
            if child_namespace != namespace {
                return Err(crate::error::YantrikDbError::InvalidInput(format!(
                    "omitted child {child_rid:?} belongs to namespace {child_namespace:?}, not {namespace:?}"
                )));
            }
            if child_status != "active" {
                return Err(crate::error::YantrikDbError::InvalidInput(format!(
                    "omitted child {child_rid:?} is not active"
                )));
            }
            if available_at.is_none_or(|value| value > impression_created_at) {
                return Err(crate::error::YantrikDbError::InvalidInput(format!(
                    "omitted child {child_rid:?} was not available when rollup impression {impression_id:?} was served"
                )));
            }
        }

        let mut existing_stmt = tx.prepare(
            "SELECT child_rid, source FROM rollup_impression_outcomes \
             WHERE impression_id = ?1",
        )?;
        let existing: Vec<(String, String)> = existing_stmt
            .query_map(params![impression_id], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<std::result::Result<_, _>>()?;
        drop(existing_stmt);
        for (child_rid, source) in existing {
            let declared = match source.as_str() {
                "selected" => selected.contains(&child_rid),
                "corrected" => corrected.contains(&child_rid),
                _ => false,
            };
            if !declared {
                return Err(crate::error::YantrikDbError::InvalidInput(format!(
                    "existing {source} outcome for child {child_rid:?} is absent from the finalized payload"
                )));
            }
        }

        let now = crate::time::now_secs();
        let mut insert = tx.prepare_cached(
            "INSERT OR IGNORE INTO rollup_impression_outcomes \
             (outcome_id, impression_id, child_rid, source, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        for child_rid in &selected {
            insert.execute(params![
                crate::id::new_id(),
                impression_id,
                child_rid,
                "selected",
                now,
            ])?;
        }
        for child_rid in &corrected {
            insert.execute(params![
                crate::id::new_id(),
                impression_id,
                child_rid,
                "corrected",
                now,
            ])?;
        }
        drop(insert);
        let mut insert_addition = tx.prepare_cached(
            "INSERT OR IGNORE INTO rollup_impression_additions \
             (impression_id, child_rid, source, created_at) \
             VALUES (?1, ?2, 'caller_false_negative', ?3)",
        )?;
        for child_rid in &added {
            insert_addition.execute(params![impression_id, child_rid, now])?;
        }
        drop(insert_addition);
        tx.execute(
            "UPDATE rollup_impressions SET outcome_payload_hash = ?2, \
             outcome_finalized_at = COALESCE(outcome_finalized_at, ?3) \
             WHERE impression_id = ?1",
            params![impression_id, payload_hash, now],
        )?;
        tx.commit()?;
        Ok(selected.len() + added.len())
    }

    /// Summarize explicit rollup outcomes without mutating the ledger. Missing
    /// finalization is unknown telemetry and never contributes a negative.
    pub fn rollup_outcome_report(
        &self,
        namespace: Option<&str>,
        since: Option<f64>,
    ) -> Result<RollupOutcomeReport> {
        if since.is_some_and(|value| !value.is_finite()) {
            return Err(crate::error::YantrikDbError::InvalidInput(
                "rollup outcome report requires a finite since timestamp".to_string(),
            ));
        }
        let conn = self.conn();
        let (
            total_impressions,
            distinct_queries,
            distinct_rollups,
            expanded_impressions,
            finalized_impressions,
            finalized_distinct_queries,
            finalized_distinct_rollups,
        ): (i64, i64, i64, i64, i64, i64, i64) = conn.query_row(
            "SELECT COUNT(*), COUNT(DISTINCT query_hash), COUNT(DISTINCT rollup_rid), \
                    COALESCE(SUM(expanded_at IS NOT NULL), 0), \
                    COALESCE(SUM(outcome_finalized_at IS NOT NULL), 0), \
                    COUNT(DISTINCT CASE WHEN outcome_finalized_at IS NOT NULL \
                                        THEN query_hash END), \
                    COUNT(DISTINCT CASE WHEN outcome_finalized_at IS NOT NULL \
                                        THEN rollup_rid END) \
             FROM rollup_impressions \
             WHERE (?1 IS NULL OR namespace = ?1) \
               AND (?2 IS NULL OR created_at >= ?2)",
            params![namespace, since],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                ))
            },
        )?;
        let (
            finalized_returned_children,
            finalized_selected_children,
            finalized_corrected_children,
        ): (i64, i64, i64) = conn.query_row(
            "SELECT COUNT(*), \
                    COALESCE(SUM(EXISTS(SELECT 1 FROM rollup_impression_outcomes o \
                                        WHERE o.impression_id = i.impression_id \
                                          AND o.child_rid = c.child_rid \
                                          AND o.source IN ('selected', 'corrected'))), 0), \
                    COALESCE(SUM(EXISTS(SELECT 1 FROM rollup_impression_outcomes o \
                                        WHERE o.impression_id = i.impression_id \
                                          AND o.child_rid = c.child_rid \
                                          AND o.source = 'corrected')), 0) \
             FROM rollup_impressions i \
             JOIN rollup_impression_children c USING (impression_id) \
             WHERE i.outcome_finalized_at IS NOT NULL \
               AND (?1 IS NULL OR i.namespace = ?1) \
               AND (?2 IS NULL OR i.created_at >= ?2)",
            params![namespace, since],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        let max_group_count = |column: &str| -> Result<i64> {
            let sql = format!(
                "SELECT COALESCE(MAX(group_count), 0) FROM (\
                     SELECT COUNT(*) AS group_count FROM rollup_impressions \
                     WHERE outcome_finalized_at IS NOT NULL \
                       AND (?1 IS NULL OR namespace = ?1) \
                       AND (?2 IS NULL OR created_at >= ?2) \
                     GROUP BY {column})"
            );
            Ok(conn.query_row(&sql, params![namespace, since], |row| row.get(0))?)
        };
        let max_query_count = max_group_count("query_hash")?;
        let max_rollup_count = max_group_count("rollup_rid")?;

        let mut rank_stmt = conn.prepare(
            "SELECT i.rank, COUNT(*), \
                    COALESCE(SUM(i.expanded_at IS NOT NULL), 0), \
                    COALESCE(SUM(i.outcome_finalized_at IS NOT NULL), 0), \
                    COALESCE(SUM(CASE WHEN i.outcome_finalized_at IS NOT NULL THEN \
                        (SELECT COUNT(*) FROM rollup_impression_children c \
                         WHERE c.impression_id = i.impression_id) ELSE 0 END), 0), \
                    COALESCE(SUM(CASE WHEN i.outcome_finalized_at IS NOT NULL THEN \
                        (SELECT COUNT(DISTINCT o.child_rid) \
                         FROM rollup_impression_outcomes o \
                         WHERE o.impression_id = i.impression_id \
                           AND o.source IN ('selected', 'corrected')) ELSE 0 END), 0), \
                    COALESCE(SUM(CASE WHEN i.outcome_finalized_at IS NOT NULL THEN \
                        (SELECT COUNT(*) FROM rollup_impression_outcomes o \
                         WHERE o.impression_id = i.impression_id \
                           AND o.source = 'corrected') ELSE 0 END), 0) \
             FROM rollup_impressions i \
             WHERE (?1 IS NULL OR i.namespace = ?1) \
               AND (?2 IS NULL OR i.created_at >= ?2) \
             GROUP BY i.rank ORDER BY i.rank",
        )?;
        let per_rank = rank_stmt
            .query_map(params![namespace, since], |row| {
                let returned: i64 = row.get(4)?;
                let selected: i64 = row.get(5)?;
                Ok(RollupRankOutcomeStats {
                    rank: row.get::<_, i64>(0)? as usize,
                    impressions: row.get(1)?,
                    expanded_impressions: row.get(2)?,
                    finalized_impressions: row.get(3)?,
                    finalized_returned_children: returned,
                    finalized_selected_children: selected,
                    finalized_corrected_children: row.get(6)?,
                    explicit_child_selection_rate: ratio(selected, returned),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let explicitly_unselected_children =
            finalized_returned_children - finalized_selected_children;
        let expansion_rate = ratio(expanded_impressions, total_impressions);
        let telemetry_completion_rate = ratio(finalized_impressions, expanded_impressions);
        let explicit_child_selection_rate =
            ratio(finalized_selected_children, finalized_returned_children);
        let max_finalized_query_share = ratio(max_query_count, finalized_impressions);
        let max_finalized_rollup_share = ratio(max_rollup_count, finalized_impressions);

        const MIN_FINALIZED_IMPRESSIONS: i64 = 200;
        const MIN_FINALIZED_QUERIES: i64 = 30;
        const MIN_FINALIZED_ROLLUPS: i64 = 20;
        const MIN_SELECTED_CHILDREN: i64 = 50;
        const MIN_UNSELECTED_CHILDREN: i64 = 50;
        const MIN_TELEMETRY_COMPLETION_RATE: f64 = 0.80;
        const MAX_GROUP_SHARE: f64 = 0.25;
        let mut readiness_failures = Vec::new();
        if finalized_impressions < MIN_FINALIZED_IMPRESSIONS {
            readiness_failures.push(format!(
                "finalized_impressions requires at least {MIN_FINALIZED_IMPRESSIONS} (observed {finalized_impressions})"
            ));
        }
        if finalized_distinct_queries < MIN_FINALIZED_QUERIES {
            readiness_failures.push(format!(
                "finalized_distinct_queries requires at least {MIN_FINALIZED_QUERIES} (observed {finalized_distinct_queries})"
            ));
        }
        if finalized_distinct_rollups < MIN_FINALIZED_ROLLUPS {
            readiness_failures.push(format!(
                "finalized_distinct_rollups requires at least {MIN_FINALIZED_ROLLUPS} (observed {finalized_distinct_rollups})"
            ));
        }
        if finalized_selected_children < MIN_SELECTED_CHILDREN {
            readiness_failures.push(format!(
                "finalized_selected_children requires at least {MIN_SELECTED_CHILDREN} (observed {finalized_selected_children})"
            ));
        }
        if explicitly_unselected_children < MIN_UNSELECTED_CHILDREN {
            readiness_failures.push(format!(
                "explicitly_unselected_children requires at least {MIN_UNSELECTED_CHILDREN} (observed {explicitly_unselected_children})"
            ));
        }
        if telemetry_completion_rate.is_none_or(|rate| rate < MIN_TELEMETRY_COMPLETION_RATE) {
            readiness_failures.push(format!(
                "telemetry_completion_rate requires at least {MIN_TELEMETRY_COMPLETION_RATE:.2} (observed {:.4})",
                telemetry_completion_rate.unwrap_or_default()
            ));
        }
        if max_finalized_query_share.is_some_and(|share| share > MAX_GROUP_SHARE) {
            readiness_failures.push(format!(
                "max_finalized_query_share must be at most {MAX_GROUP_SHARE:.2} (observed {:.4})",
                max_finalized_query_share.unwrap_or_default()
            ));
        }
        if max_finalized_rollup_share.is_some_and(|share| share > MAX_GROUP_SHARE) {
            readiness_failures.push(format!(
                "max_finalized_rollup_share must be at most {MAX_GROUP_SHARE:.2} (observed {:.4})",
                max_finalized_rollup_share.unwrap_or_default()
            ));
        }
        let evidence_status = if total_impressions == 0 {
            "no_data"
        } else if readiness_failures.is_empty() {
            "ready_for_offline_evaluation"
        } else {
            "insufficient_evidence"
        };

        Ok(RollupOutcomeReport {
            namespace: namespace.map(str::to_string),
            since,
            total_impressions,
            distinct_queries,
            distinct_rollups,
            expanded_impressions,
            finalized_impressions,
            finalized_distinct_queries,
            finalized_distinct_rollups,
            finalized_returned_children,
            finalized_selected_children,
            finalized_corrected_children,
            explicitly_unselected_children,
            expansion_rate,
            telemetry_completion_rate,
            explicit_child_selection_rate,
            max_finalized_query_share,
            max_finalized_rollup_share,
            per_rank,
            evidence_status: evidence_status.to_string(),
            readiness_failures,
        })
    }

    /// Export finalized per-child examples for offline calibration.
    ///
    /// Every feature was frozen when the rollup or expansion was served. The
    /// export never joins mutable memory attributes and never exposes query
    /// text. An omitted child is a negative only after exact finalization.
    pub fn rollup_outcome_examples(
        &self,
        namespace: Option<&str>,
        since: Option<f64>,
        until: Option<f64>,
        limit: usize,
    ) -> Result<Vec<RollupOutcomeExample>> {
        if since.is_some_and(|value| !value.is_finite())
            || until.is_some_and(|value| !value.is_finite())
        {
            return Err(crate::error::YantrikDbError::InvalidInput(
                "rollup outcome examples require finite time bounds".to_string(),
            ));
        }
        if since.zip(until).is_some_and(|(start, end)| start > end) {
            return Err(crate::error::YantrikDbError::InvalidInput(
                "rollup outcome examples require since <= until".to_string(),
            ));
        }
        if limit == 0 {
            return Ok(Vec::new());
        }
        let bounded_limit = limit.min(10_000) as i64;
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT i.impression_id, i.query_hash, i.namespace, i.rollup_rid, \
                    i.rank, i.score, c.child_rid, c.rank, \
                    (SELECT COUNT(*) FROM rollup_impression_children all_children \
                     WHERE all_children.impression_id = i.impression_id), \
                    EXISTS(SELECT 1 FROM rollup_impression_outcomes selected \
                           WHERE selected.impression_id = i.impression_id \
                             AND selected.child_rid = c.child_rid \
                             AND selected.source IN ('selected', 'corrected')), \
                    EXISTS(SELECT 1 FROM rollup_impression_outcomes corrected \
                           WHERE corrected.impression_id = i.impression_id \
                             AND corrected.child_rid = c.child_rid \
                             AND corrected.source = 'corrected'), \
                    i.created_at, i.outcome_finalized_at \
             FROM rollup_impressions i \
             JOIN rollup_impression_children c USING (impression_id) \
             WHERE i.outcome_finalized_at IS NOT NULL \
               AND (?1 IS NULL OR i.namespace = ?1) \
               AND (?2 IS NULL OR i.created_at >= ?2) \
               AND (?3 IS NULL OR i.created_at <= ?3) \
             ORDER BY i.created_at, i.impression_id, c.rank, c.child_rid \
             LIMIT ?4",
        )?;
        let examples = stmt
            .query_map(params![namespace, since, until, bounded_limit], |row| {
                Ok(RollupOutcomeExample {
                    export_schema_version: 1,
                    impression_id: row.get(0)?,
                    query_hash: row.get(1)?,
                    namespace: row.get(2)?,
                    rollup_rid: row.get(3)?,
                    rollup_rank: row.get::<_, i64>(4)? as usize,
                    rollup_score: row.get(5)?,
                    child_rid: row.get(6)?,
                    child_rank: row.get::<_, i64>(7)? as usize,
                    returned_child_count: row.get::<_, i64>(8)? as usize,
                    selected: row.get::<_, i64>(9)? != 0,
                    corrected: row.get::<_, i64>(10)? != 0,
                    created_at: row.get(11)?,
                    outcome_finalized_at: row.get(12)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(examples)
    }

    /// Report whether explicit false-negative telemetry is sufficiently broad
    /// for an offline membership-rescue evaluation. This does not authorize a
    /// production policy change.
    pub fn rollup_membership_report(
        &self,
        namespace: Option<&str>,
        since: Option<f64>,
    ) -> Result<RollupMembershipReport> {
        if since.is_some_and(|value| !value.is_finite()) {
            return Err(crate::error::YantrikDbError::InvalidInput(
                "rollup membership report since must be finite".to_string(),
            ));
        }
        let conn = self.conn();
        let (total_impressions, expanded_impressions, finalized_impressions): (i64, i64, i64) =
            conn.query_row(
                "SELECT COUNT(*), COALESCE(SUM(expanded_at IS NOT NULL), 0), \
                        COALESCE(SUM(outcome_finalized_at IS NOT NULL), 0) \
                 FROM rollup_impressions \
                 WHERE (?1 IS NULL OR namespace = ?1) \
                   AND (?2 IS NULL OR created_at >= ?2)",
                params![namespace, since],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?;
        let (
            finalized_added_children,
            finalized_impressions_with_additions,
            finalized_distinct_queries_with_additions,
        ): (i64, i64, i64) = conn.query_row(
            "SELECT COUNT(*), COUNT(DISTINCT a.impression_id), \
                    COUNT(DISTINCT i.namespace || char(0) || i.query_hash) \
             FROM rollup_impression_additions a \
             JOIN rollup_impressions i USING (impression_id) \
             WHERE i.outcome_finalized_at IS NOT NULL \
               AND (?1 IS NULL OR i.namespace = ?1) \
               AND (?2 IS NULL OR i.created_at >= ?2)",
            params![namespace, since],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        let finalized_returned_children: i64 = conn.query_row(
            "SELECT COUNT(*) FROM rollup_impressions i \
             JOIN rollup_impression_children c USING (impression_id) \
             WHERE i.outcome_finalized_at IS NOT NULL \
               AND (?1 IS NULL OR i.namespace = ?1) \
               AND (?2 IS NULL OR i.created_at >= ?2)",
            params![namespace, since],
            |row| row.get(0),
        )?;
        let telemetry_completion_rate = ratio(finalized_impressions, expanded_impressions);
        let added_child_rate = ratio(
            finalized_added_children,
            finalized_returned_children + finalized_added_children,
        );

        const MIN_ADDED_CHILDREN: i64 = 50;
        const MIN_IMPRESSIONS_WITH_ADDITIONS: i64 = 30;
        const MIN_QUERIES_WITH_ADDITIONS: i64 = 15;
        const MIN_TELEMETRY_COMPLETION_RATE: f64 = 0.80;
        let mut readiness_failures = Vec::new();
        if finalized_added_children < MIN_ADDED_CHILDREN {
            readiness_failures.push(format!(
                "finalized_added_children requires at least {MIN_ADDED_CHILDREN} (observed {finalized_added_children})"
            ));
        }
        if finalized_impressions_with_additions < MIN_IMPRESSIONS_WITH_ADDITIONS {
            readiness_failures.push(format!(
                "finalized_impressions_with_additions requires at least {MIN_IMPRESSIONS_WITH_ADDITIONS} (observed {finalized_impressions_with_additions})"
            ));
        }
        if finalized_distinct_queries_with_additions < MIN_QUERIES_WITH_ADDITIONS {
            readiness_failures.push(format!(
                "finalized_distinct_queries_with_additions requires at least {MIN_QUERIES_WITH_ADDITIONS} (observed {finalized_distinct_queries_with_additions})"
            ));
        }
        if telemetry_completion_rate.is_none_or(|rate| rate < MIN_TELEMETRY_COMPLETION_RATE) {
            readiness_failures.push(format!(
                "telemetry_completion_rate requires at least {MIN_TELEMETRY_COMPLETION_RATE:.2} (observed {:.4})",
                telemetry_completion_rate.unwrap_or_default()
            ));
        }
        let evidence_status = if total_impressions == 0 {
            "no_data"
        } else if readiness_failures.is_empty() {
            "ready_for_offline_evaluation"
        } else {
            "insufficient_evidence"
        };

        Ok(RollupMembershipReport {
            namespace: namespace.map(str::to_string),
            since,
            total_impressions,
            expanded_impressions,
            finalized_impressions,
            finalized_added_children,
            finalized_impressions_with_additions,
            finalized_distinct_queries_with_additions,
            telemetry_completion_rate,
            added_child_rate,
            evidence_status: evidence_status.to_string(),
            readiness_failures,
        })
    }

    /// Export complete finalized impression groups for membership calibration.
    /// Bounds apply to finalization time, so labels created later cannot leak
    /// into an earlier evaluation window. `limit_impressions` never splits a
    /// group even when one impression has many children.
    pub fn rollup_membership_examples(
        &self,
        namespace: Option<&str>,
        finalized_since: Option<f64>,
        finalized_until: Option<f64>,
        limit_impressions: usize,
    ) -> Result<Vec<RollupMembershipExample>> {
        if finalized_since.is_some_and(|value| !value.is_finite())
            || finalized_until.is_some_and(|value| !value.is_finite())
        {
            return Err(crate::error::YantrikDbError::InvalidInput(
                "rollup membership examples require finite finalization bounds".to_string(),
            ));
        }
        if finalized_since
            .zip(finalized_until)
            .is_some_and(|(start, end)| start > end)
        {
            return Err(crate::error::YantrikDbError::InvalidInput(
                "rollup membership examples require finalized_since <= finalized_until".to_string(),
            ));
        }
        if limit_impressions == 0 {
            return Ok(Vec::new());
        }
        let bounded_limit = limit_impressions.min(10_000) as i64;
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "WITH bounded AS ( \
                SELECT * FROM rollup_impressions i \
                WHERE i.outcome_finalized_at IS NOT NULL \
                  AND (?1 IS NULL OR i.namespace = ?1) \
                  AND (?2 IS NULL OR i.outcome_finalized_at >= ?2) \
                  AND (?3 IS NULL OR i.outcome_finalized_at <= ?3) \
                ORDER BY i.outcome_finalized_at, i.impression_id \
                LIMIT ?4 \
             ), examples AS ( \
                SELECT i.impression_id, i.query_hash, i.namespace, i.rollup_rid, \
                    i.rank AS rollup_rank, i.score AS rollup_score, \
                    i.requested_count, i.query_shape, c.child_rid, \
                    c.rank AS child_rank, c.score AS child_score, \
                    (SELECT COUNT(*) FROM rollup_impression_children all_children \
                     WHERE all_children.impression_id = i.impression_id), \
                    1 AS returned, 0 AS omitted_positive, \
                    EXISTS(SELECT 1 FROM rollup_impression_outcomes selected \
                           WHERE selected.impression_id = i.impression_id \
                             AND selected.child_rid = c.child_rid \
                             AND selected.source IN ('selected', 'corrected')), \
                    EXISTS(SELECT 1 FROM rollup_impression_outcomes corrected \
                           WHERE corrected.impression_id = i.impression_id \
                             AND corrected.child_rid = c.child_rid \
                             AND corrected.source = 'corrected'), \
                    NULL AS omission_source, i.created_at, i.outcome_finalized_at \
                FROM bounded i \
                JOIN rollup_impression_children c USING (impression_id) \
                UNION ALL \
                SELECT i.impression_id, i.query_hash, i.namespace, i.rollup_rid, \
                    i.rank, i.score, i.requested_count, i.query_shape, \
                    a.child_rid, NULL, NULL, \
                    (SELECT COUNT(*) FROM rollup_impression_children all_children \
                     WHERE all_children.impression_id = i.impression_id), \
                    0, 1, 1, 0, a.source, i.created_at, i.outcome_finalized_at \
                FROM bounded i \
                JOIN rollup_impression_additions a USING (impression_id) \
             ) \
             SELECT * FROM examples \
             ORDER BY outcome_finalized_at, impression_id, returned DESC, child_rank, child_rid",
        )?;
        let examples = stmt
            .query_map(
                params![namespace, finalized_since, finalized_until, bounded_limit],
                |row| {
                    let query_hash: String = row.get(1)?;
                    let namespace: String = row.get(2)?;
                    Ok(RollupMembershipExample {
                        export_schema_version: 1,
                        impression_id: row.get(0)?,
                        query_key: stable_text_hash(&format!("{namespace}\u{0}{query_hash}")),
                        namespace,
                        rollup_rid: row.get(3)?,
                        rollup_rank: row.get::<_, i64>(4)? as usize,
                        rollup_score: row.get(5)?,
                        requested_count: row.get::<_, Option<i64>>(6)?.map(|value| value as usize),
                        query_shape: row.get(7)?,
                        child_rid: row.get(8)?,
                        child_rank: row.get::<_, Option<i64>>(9)?.map(|value| value as usize),
                        child_score: row.get(10)?,
                        returned_child_count: row.get::<_, i64>(11)? as usize,
                        returned: row.get::<_, i64>(12)? != 0,
                        omitted_positive: row.get::<_, i64>(13)? != 0,
                        positive: row.get::<_, i64>(14)? != 0,
                        corrected: row.get::<_, i64>(15)? != 0,
                        omission_source: row.get(16)?,
                        impression_created_at: row.get(17)?,
                        outcome_finalized_at: row.get(18)?,
                    })
                },
            )?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(examples)
    }

    /// v0.10 Item 2 — explicit rejection: the caller states that these
    /// served results were IRRELEVANT to the query they were served for
    /// (typically alongside a refine). This is deliberately a separate
    /// call from `recall_refine`'s `original_rids`, which only means
    /// "already seen" — seeking more results is not evidence the first
    /// page was wrong (sol ruling 1.2). Consumers with richer exclusion
    /// reasons (redundant, wrong-granularity, duplicate) must filter to
    /// genuine irrelevance BEFORE calling — only that reason is a valid
    /// negative (nuron's exclusion-reason review). Returns how many
    /// labels were recorded (rids without a recent impression bind
    /// nothing).
    pub fn reject_recalled(&self, rids: &[&str]) -> Result<usize> {
        let mut recorded = 0;
        for rid in rids {
            if self.record_ranking_label(rid, "rejected_refine", -1, WEIGHT_REJECTED_REFINE)? {
                recorded += 1;
            }
        }
        Ok(recorded)
    }

    /// v0.10 Item 2 — pick up to 2 served rids worth asking the
    /// consumer to grade: nearest the relevance gate (most informative
    /// for the fit), excluding any (query, rid) ever proposed before
    /// and any rid already labeled for this query. Proposals are
    /// recorded so a skip is never re-asked. Best-effort at the call
    /// site (the read path never fails on the rider).
    pub(crate) fn propose_label_requests(
        &self,
        results: &[RecallResult],
        query_embedding: &[f32],
        threshold_tau: f64,
    ) -> Result<Vec<String>> {
        if results.is_empty() {
            return Ok(Vec::new());
        }
        let qhash = query_hash(query_embedding);
        let mut ranked: Vec<(&str, f64)> = results
            .iter()
            .map(|r| (r.rid.as_str(), (r.scores.similarity - threshold_tau).abs()))
            .collect();
        ranked.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(b.0)));

        let ts = crate::time::now_secs();
        let conn = self.conn();
        let mut asked_stmt = conn.prepare_cached(
            "SELECT EXISTS(SELECT 1 FROM label_requests WHERE query_hash = ?1 AND rid = ?2)",
        )?;
        let mut labeled_stmt = conn.prepare_cached(
            "SELECT EXISTS(SELECT 1 FROM ranking_labels l \
             JOIN recall_impressions i \
               ON i.episode_id = l.episode_id AND i.rid = l.rid \
             WHERE i.query_hash = ?1 AND l.rid = ?2)",
        )?;
        let mut insert_stmt = conn.prepare_cached(
            "INSERT OR IGNORE INTO label_requests (query_hash, rid, requested_at) \
             VALUES (?1, ?2, ?3)",
        )?;
        let mut picked = Vec::with_capacity(2);
        for (rid, _) in ranked {
            if picked.len() == 2 {
                break;
            }
            let asked: bool = asked_stmt.query_row(params![qhash, rid], |r| r.get(0))?;
            if asked {
                continue;
            }
            let labeled: bool = labeled_stmt.query_row(params![qhash, rid], |r| r.get(0))?;
            if labeled {
                continue;
            }
            insert_stmt.execute(params![qhash, rid, ts])?;
            picked.push(rid.to_string());
        }
        Ok(picked)
    }

    /// The outcome anchor: an INDEPENDENT caller-initiated action
    /// targeted this rid (get-by-rid, link creation, correction).
    /// At most one weak positive per (impression, rid) via the UNIQUE
    /// constraint; a rid the ranker never served yields nothing. Called
    /// from consumer-facing mutation/read-by-id paths — NEVER from
    /// recall or any engine-internal traversal (rule 2: served events
    /// and resurfacing are categorically ineligible).
    pub(crate) fn note_caller_used(&self, rid: &str) {
        // Best-effort: a labeling failure must never fail the caller's
        // actual operation.
        let _ = self.record_ranking_label(rid, "caller_used", 1, WEIGHT_CALLER_USED);
    }
}

fn stable_text_hash(value: &str) -> String {
    blake3::hash(value.as_bytes()).to_hex().to_string()
}

fn ratio(numerator: i64, denominator: i64) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim).map(|i| (seed + i as f32) * 0.1).collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    }

    fn rec(db: &YantrikDB, text: &str, seed: f32) -> String {
        rec_in(db, text, seed, "default")
    }

    fn rec_in(db: &YantrikDB, text: &str, seed: f32, namespace: &str) -> String {
        db.record(
            text,
            "semantic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({}),
            &vec_seed(seed, 8),
            namespace,
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap()
    }

    fn rec_with_rid(db: &YantrikDB, rid: &str, text: &str, seed: f32) {
        db.record_with_rid(
            rid,
            text,
            "semantic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({}),
            &vec_seed(seed, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
            1_700_000_000_000_000,
            &[],
            "test-model.v1",
            None,
            crate::provenance::WriteAdmission::Origin,
        )
        .unwrap();
    }

    fn recall_all(db: &YantrikDB, seed: f32) -> Vec<RecallResult> {
        db.recall(
            &vec_seed(seed, 8),
            10,
            None,
            None,
            false,
            false,
            None,
            false, // reinforce ON: impressions must be logged on real recalls
            None,
            None,
            None,
            None,
            None,
            false,
        )
        .unwrap()
    }

    #[test]
    fn recall_logs_impressions_with_serve_time_features() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let a = rec(&db, "fact a", 1.0);
        let _b = rec(&db, "fact b", 1.05);
        let results = recall_all(&db, 1.0);
        assert!(!results.is_empty());

        let conn = db.conn();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM recall_impressions", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            n as usize,
            results.len(),
            "one impression row per served result"
        );
        let (rank, sim, generation): (i64, f64, i64) = conn
            .query_row(
                "SELECT rank, f_similarity, weight_generation FROM recall_impressions \
                 WHERE rid = ?1",
                params![a],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert!(rank >= 0);
        assert!((0.0..=1.0).contains(&sim));
        assert_eq!(generation, 0, "factory weights are generation 0");
        drop(conn);

        // skip_reinforce (engine-internal) recalls must NOT log.
        let before: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM recall_impressions", [], |r| r.get(0))
            .unwrap();
        db.recall(
            &vec_seed(1.0, 8),
            10,
            None,
            None,
            false,
            false,
            None,
            true,
            None,
            None,
            None,
            None,
            None,
            false,
        )
        .unwrap();
        let after: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM recall_impressions", [], |r| r.get(0))
            .unwrap();
        assert_eq!(before, after, "internal recalls leave no impressions");
    }

    #[test]
    fn caller_used_binds_one_weak_positive_to_latest_impression() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let a = rec(&db, "fact a", 1.0);
        recall_all(&db, 1.0);

        // get() is a caller-initiated rid-targeting action → outcome anchor.
        let _ = db.get(&a).unwrap();
        let conn = db.conn();
        let (n, polarity, weight): (i64, i32, f64) = conn
            .query_row(
                "SELECT COUNT(*), polarity, weight FROM ranking_labels \
                 WHERE rid = ?1 AND source = 'caller_used'",
                params![a],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(n, 1);
        assert_eq!(polarity, 1);
        assert!((weight - WEIGHT_CALLER_USED).abs() < 1e-12);
        drop(conn);

        // Repeat gets do not amplify (idempotent per impression/source).
        let _ = db.get(&a).unwrap();
        let n: i64 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM ranking_labels WHERE rid = ?1",
                params![a],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1, "one weak positive per impression, not per access");
    }

    #[test]
    fn unserved_rid_actions_yield_no_label() {
        // An action on a record the ranker never served says nothing
        // about the ranker.
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let a = rec(&db, "never recalled", 1.0);
        let _ = db.get(&a).unwrap();
        let n: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM ranking_labels", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn rollup_outcomes_are_explicit_exact_and_idempotent() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let rollup = rec(&db, "topic rollup", 1.0);
        let child = rec(&db, "answer-sized child", 20.0);
        let first = db
            .note_rollup_impression(
                &rollup,
                "topic question",
                None,
                0,
                0.8,
                Some("impression-1"),
            )
            .unwrap()
            .to_string();
        let retry = db
            .note_rollup_impression(
                &rollup,
                "topic question",
                None,
                0,
                0.8,
                Some("impression-1"),
            )
            .unwrap();
        assert_eq!(first, retry);
        assert!(db
            .note_rollup_impression(
                &rollup,
                "different question",
                None,
                0,
                0.8,
                Some("impression-1"),
            )
            .is_err());

        assert_eq!(
            db.note_rollup_selection(&first, &child, "selected")
                .unwrap(),
            false,
            "a child must have been returned before it can be selected"
        );
        assert_eq!(
            db.note_rollup_expansion(&first, &[&child, &child]).unwrap(),
            1
        );
        assert_eq!(db.note_rollup_expansion(&first, &[&child]).unwrap(), 1);
        let conn = db.conn();
        let impressions: i64 = conn
            .query_row("SELECT COUNT(*) FROM rollup_impressions", [], |r| r.get(0))
            .unwrap();
        let children: i64 = conn
            .query_row("SELECT COUNT(*) FROM rollup_impression_children", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(impressions, 1);
        assert_eq!(children, 1);
        drop(conn);

        // A generic point-read carries no hidden rollup semantics.
        let _ = db.get(&child).unwrap();
        assert_eq!(
            db.conn()
                .query_row("SELECT COUNT(*) FROM rollup_impression_outcomes", [], |r| {
                    r.get::<_, i64>(0)
                })
                .unwrap(),
            0
        );
        assert!(db
            .note_rollup_selection(&first, &child, "selected")
            .unwrap());
        assert!(!db
            .note_rollup_selection(&first, &child, "selected")
            .unwrap());
        assert!(db
            .note_rollup_selection(&first, &child, "corrected")
            .unwrap());
        assert_eq!(
            db.finalize_rollup_outcome(&first, &[&child], &[&child])
                .unwrap(),
            1
        );
        assert_eq!(
            db.finalize_rollup_outcome(&first, &[&child], &[&child])
                .unwrap(),
            1,
            "an exact retry must be idempotent"
        );
        assert!(db.finalize_rollup_outcome(&first, &[], &[]).is_err());
        assert!(!db
            .note_rollup_selection(&first, &child, "selected")
            .unwrap());
        let conn = db.conn();
        let outcomes: i64 = conn
            .query_row("SELECT COUNT(*) FROM rollup_impression_outcomes", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(outcomes, 2);
        drop(conn);
        let report = db.rollup_outcome_report(None, None).unwrap();
        assert_eq!(report.finalized_returned_children, 1);
        assert_eq!(report.finalized_selected_children, 1);
        assert_eq!(report.finalized_corrected_children, 1);
        assert_eq!(report.explicitly_unselected_children, 0);

        let conn = db.conn();
        conn.execute(
            "DELETE FROM rollup_impression_outcomes \
             WHERE impression_id = ?1 AND source = 'selected'",
            params![first],
        )
        .unwrap();
        drop(conn);
        let corrected_only = db.rollup_outcome_report(None, None).unwrap();
        assert_eq!(corrected_only.finalized_selected_children, 1);
        assert_eq!(corrected_only.explicitly_unselected_children, 0);

        let conn = db.conn();
        conn.execute(
            "DELETE FROM rollup_impressions WHERE impression_id = ?1",
            params![first],
        )
        .unwrap();
        let outcomes_after_delete: i64 = conn
            .query_row("SELECT COUNT(*) FROM rollup_impression_outcomes", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(outcomes_after_delete, 0);
    }

    #[test]
    fn rollup_omissions_are_exact_positive_and_not_served_children() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let rollup = rec(&db, "topic rollup", 1.0);
        let returned = rec(&db, "returned child", 2.0);
        let added = rec(&db, "omitted positive child", 3.0);
        let other_namespace = rec_in(&db, "other namespace child", 4.0, "other");
        let inactive = rec(&db, "inactive child", 5.0);
        db.conn()
            .execute(
                "UPDATE memories SET consolidation_status = 'tombstoned' WHERE rid = ?1",
                [&inactive],
            )
            .unwrap();

        let impression = db
            .note_rollup_impression_with_features(
                &rollup,
                "List exactly two topic items",
                None,
                0,
                0.9,
                Some(2),
                Some("list"),
                Some("addition-example"),
            )
            .unwrap();
        let late = rec(&db, "created after impression", 6.0);
        db.conn()
            .execute(
                "UPDATE oplog SET timestamp = timestamp + 60.0 \
                 WHERE target_rid = ?1 AND op_type = 'record'",
                [&late],
            )
            .unwrap();
        assert_eq!(
            db.note_rollup_impression_with_features(
                &rollup,
                "List exactly two topic items",
                None,
                0,
                0.9,
                Some(2),
                Some("LIST"),
                Some("addition-example"),
            )
            .unwrap(),
            impression
        );
        assert!(db
            .note_rollup_impression_with_features(
                &rollup,
                "List exactly two topic items",
                None,
                0,
                0.9,
                Some(3),
                Some("list"),
                Some("addition-example"),
            )
            .is_err());
        assert!(db
            .note_rollup_impression_with_features(
                &rollup,
                "query",
                None,
                0,
                0.9,
                Some(0),
                Some("unknown"),
                None,
            )
            .is_err());

        let returned_with_score = [(&returned[..], Some(0.73))];
        assert_eq!(
            db.note_rollup_expansion_with_scores(&impression, &returned_with_score)
                .unwrap(),
            1
        );
        assert_eq!(
            db.note_rollup_expansion_with_scores(&impression, &returned_with_score)
                .unwrap(),
            1
        );
        assert!(db
            .finalize_rollup_outcome_with_omissions(&impression, &[&returned], &[], &[&returned],)
            .is_err());
        assert!(db
            .finalize_rollup_outcome_with_omissions(
                &impression,
                &[&returned],
                &[],
                &["missing-child"],
            )
            .is_err());
        assert!(db
            .finalize_rollup_outcome_with_omissions(
                &impression,
                &[&returned],
                &[],
                &[&other_namespace],
            )
            .is_err());
        assert!(db
            .finalize_rollup_outcome_with_omissions(&impression, &[&returned], &[], &[&inactive],)
            .is_err());
        assert!(db
            .finalize_rollup_outcome_with_omissions(&impression, &[&returned], &[], &[&late],)
            .is_err());
        assert_eq!(
            db.finalize_rollup_outcome_with_omissions(
                &impression,
                &[&returned],
                &[],
                &[&added, &added],
            )
            .unwrap(),
            2
        );
        assert_eq!(
            db.finalize_rollup_outcome_with_omissions(&impression, &[&returned], &[], &[&added],)
                .unwrap(),
            2,
            "an exact addition retry must be idempotent"
        );
        db.conn()
            .execute(
                "UPDATE memories SET consolidation_status = 'tombstoned' WHERE rid = ?1",
                [&added],
            )
            .unwrap();
        assert_eq!(
            db.finalize_rollup_outcome_with_omissions(&impression, &[&returned], &[], &[&added],)
                .unwrap(),
            2,
            "an exact retry must not be invalidated by later memory state"
        );
        assert!(db
            .finalize_rollup_outcome(&impression, &[&returned], &[])
            .is_err());

        let conn = db.conn();
        let served: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM rollup_impression_children WHERE impression_id = ?1",
                [&impression],
                |row| row.get(0),
            )
            .unwrap();
        let additions: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM rollup_impression_additions WHERE impression_id = ?1",
                [&impression],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(served, 1, "an omission must not rewrite served history");
        assert_eq!(additions, 1);
        assert!(conn
            .execute(
                "INSERT INTO rollup_impression_additions \
                 (impression_id, child_rid, source, created_at) \
                 VALUES (?1, ?2, 'caller_false_negative', 0.0)",
                params![impression, returned],
            )
            .is_err());
        assert!(conn
            .execute(
                "INSERT INTO rollup_impression_children \
                 (impression_id, child_rid, rank) VALUES (?1, ?2, 9)",
                params![impression, added],
            )
            .is_err());
        drop(conn);

        let report = db.rollup_membership_report(None, None).unwrap();
        assert_eq!(report.finalized_added_children, 1);
        assert_eq!(report.finalized_impressions_with_additions, 1);
        assert_eq!(report.finalized_distinct_queries_with_additions, 1);
        assert_eq!(report.added_child_rate, Some(0.5));
    }

    #[test]
    fn rollup_expansion_requires_an_exact_impression() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        assert!(db.note_rollup_expansion("missing", &[]).is_err());
        assert!(!db
            .note_rollup_selection("missing", "child", "selected")
            .unwrap());
        assert!(db
            .note_rollup_selection("missing", "child", "implicit")
            .is_err());
        assert!(db.finalize_rollup_outcome("missing", &[], &[]).is_err());
    }

    #[test]
    fn rollup_omission_accepts_record_with_rid_only_when_preexisting() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let rollup = rec(&db, "topic rollup", 1.0);
        let returned = rec(&db, "returned child", 2.0);
        rec_with_rid(&db, "caller-rid-before", "preexisting omitted child", 3.0);

        let impression = db
            .note_rollup_impression(&rollup, "topic list", None, 0, 0.9, None)
            .unwrap();
        db.note_rollup_expansion(&impression, &[&returned]).unwrap();

        rec_with_rid(&db, "caller-rid-after", "later omitted child", 4.0);
        db.conn()
            .execute(
                "UPDATE oplog SET timestamp = timestamp + 60.0 \
                 WHERE target_rid = 'caller-rid-after' AND op_type = 'record_with_rid'",
                [],
            )
            .unwrap();
        assert!(db
            .finalize_rollup_outcome_with_omissions(
                &impression,
                &[&returned],
                &[],
                &["caller-rid-after"],
            )
            .is_err());
        assert_eq!(
            db.finalize_rollup_outcome_with_omissions(
                &impression,
                &[&returned],
                &[],
                &["caller-rid-before"],
            )
            .unwrap(),
            2
        );
    }

    #[test]
    fn rollup_outcome_report_counts_only_finalized_absence_as_negative() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let rollup = rec(&db, "topic rollup", 1.0);
        let first_child = rec(&db, "first child", 20.0);
        let second_child = rec(&db, "second child", 21.0);

        let unfinished = db
            .note_rollup_impression(&rollup, "unfinished", None, 1, 0.7, None)
            .unwrap();
        db.note_rollup_expansion(&unfinished, &[&first_child, &second_child])
            .unwrap();
        let before = db.rollup_outcome_report(None, None).unwrap();
        assert_eq!(before.finalized_impressions, 0);
        assert_eq!(before.finalized_returned_children, 0);
        assert_eq!(before.explicitly_unselected_children, 0);
        assert_eq!(before.telemetry_completion_rate, Some(0.0));

        let finalized = db
            .note_rollup_impression(&rollup, "finalized", None, 0, 0.9, None)
            .unwrap();
        db.note_rollup_expansion(&finalized, &[&first_child, &second_child])
            .unwrap();
        assert_eq!(db.finalize_rollup_outcome(&finalized, &[], &[]).unwrap(), 0);
        assert!(db
            .note_rollup_selection(&finalized, &first_child, "selected")
            .is_err());

        let count_before: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM rollup_impressions", [], |row| {
                row.get(0)
            })
            .unwrap();
        let report = db.rollup_outcome_report(Some("default"), None).unwrap();
        let count_after: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM rollup_impressions", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count_before, count_after, "the report must be read-only");
        assert_eq!(report.total_impressions, 2);
        assert_eq!(report.expanded_impressions, 2);
        assert_eq!(report.finalized_impressions, 1);
        assert_eq!(report.finalized_returned_children, 2);
        assert_eq!(report.finalized_selected_children, 0);
        assert_eq!(report.explicitly_unselected_children, 2);
        assert_eq!(report.telemetry_completion_rate, Some(0.5));
        assert_eq!(report.explicit_child_selection_rate, Some(0.0));
        assert_eq!(report.evidence_status, "insufficient_evidence");
        assert_eq!(report.per_rank.len(), 2);
        assert_eq!(report.per_rank[0].rank, 0);
        assert_eq!(report.per_rank[0].finalized_impressions, 1);
        assert_eq!(report.per_rank[1].rank, 1);
        assert_eq!(report.per_rank[1].finalized_impressions, 0);
        assert_eq!(
            db.rollup_outcome_report(Some("other"), None)
                .unwrap()
                .evidence_status,
            "no_data"
        );
        assert_eq!(
            db.rollup_outcome_report(None, Some(crate::time::now_secs() + 1.0))
                .unwrap()
                .evidence_status,
            "no_data"
        );
        assert!(db.rollup_outcome_report(None, Some(f64::NAN)).is_err());
    }

    #[test]
    fn rollup_outcome_examples_export_only_frozen_finalized_children() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let rollup = rec(&db, "topic rollup", 1.0);
        let selected = rec(&db, "selected child", 20.0);
        let corrected = rec(&db, "corrected child", 21.0);
        let unselected = rec(&db, "unselected child", 22.0);
        let added = rec(&db, "omitted positive child", 23.0);

        let unfinished = db
            .note_rollup_impression(&rollup, "unfinished topic", None, 2, 0.4, None)
            .unwrap();
        db.note_rollup_expansion(&unfinished, &[&unselected])
            .unwrap();

        let finalized = db
            .note_rollup_impression_with_features(
                &rollup,
                "finalized topic",
                None,
                1,
                0.75,
                Some(4),
                Some("ordered_list"),
                Some("offline-example"),
            )
            .unwrap();
        db.note_rollup_expansion_with_scores(
            &finalized,
            &[
                (&selected, Some(0.8)),
                (&corrected, Some(0.7)),
                (&unselected, None),
            ],
        )
        .unwrap();
        db.finalize_rollup_outcome_with_omissions(
            &finalized,
            &[&selected],
            &[&corrected],
            &[&added],
        )
        .unwrap();

        let count_before: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM rollup_impressions", [], |row| {
                row.get(0)
            })
            .unwrap();
        let examples = db
            .rollup_outcome_examples(Some("default"), None, None, 100)
            .unwrap();
        let count_after: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM rollup_impressions", [], |row| {
                row.get(0)
            })
            .unwrap();

        assert_eq!(count_before, count_after, "the export must be read-only");
        assert_eq!(examples.len(), 3, "the v1 export remains returned-only");
        assert_eq!(examples[0].export_schema_version, 1);
        assert_eq!(examples[0].impression_id, "offline-example");
        assert_eq!(examples[0].query_hash, stable_text_hash("finalized topic"));
        assert_eq!(examples[0].rollup_rid, rollup);
        assert_eq!(examples[0].rollup_rank, 1);
        assert_eq!(examples[0].rollup_score, 0.75);
        assert_eq!(examples[0].returned_child_count, 3);
        assert_eq!(examples[0].child_rid, selected);
        assert_eq!(examples[0].child_rank, 0);
        assert!(examples[0].selected);
        assert!(!examples[0].corrected);
        assert_eq!(examples[1].child_rid, corrected);
        assert!(examples[1].selected, "correction implies selection");
        assert!(examples[1].corrected);
        assert_eq!(examples[2].child_rid, unselected);
        assert!(!examples[2].selected);
        assert!(!examples[2].corrected);
        let membership = db
            .rollup_membership_examples(Some("default"), None, None, 1)
            .unwrap();
        assert_eq!(
            membership.len(),
            4,
            "the impression limit keeps groups whole"
        );
        assert!(membership.iter().all(|row| row.impression_id == finalized));
        assert_eq!(membership[0].requested_count, Some(4));
        assert_eq!(membership[0].query_shape.as_deref(), Some("ordered_list"));
        assert_eq!(membership[0].child_rank, Some(0));
        assert_eq!(membership[0].child_score, Some(0.8));
        assert!(membership[0].returned);
        assert!(!membership[0].omitted_positive);
        assert!(membership[0].positive);
        let omitted = membership.iter().find(|row| row.omitted_positive).unwrap();
        assert_eq!(omitted.child_rid, added);
        assert_eq!(omitted.child_rank, None);
        assert_eq!(omitted.child_score, None);
        assert!(!omitted.returned);
        assert!(omitted.positive);
        assert_eq!(
            omitted.omission_source.as_deref(),
            Some("caller_false_negative")
        );
        assert!(!omitted.corrected);
        assert!(!db
            .rollup_membership_examples(
                None,
                None,
                Some(omitted.outcome_finalized_at - 0.000_001),
                100,
            )
            .unwrap()
            .iter()
            .any(|row| row.impression_id == finalized));
        assert_eq!(
            db.rollup_outcome_examples(None, None, None, 2)
                .unwrap()
                .len(),
            2
        );
        assert!(db
            .rollup_outcome_examples(None, None, None, 0)
            .unwrap()
            .is_empty());
        assert!(db
            .rollup_outcome_examples(None, Some(crate::time::now_secs() + 1.0), None, 100)
            .unwrap()
            .is_empty());
        assert!(db
            .rollup_outcome_examples(None, None, Some(0.0), 100)
            .unwrap()
            .is_empty());
        assert!(db
            .rollup_outcome_examples(None, Some(f64::NAN), None, 100)
            .is_err());
        assert!(db
            .rollup_outcome_examples(None, Some(2.0), Some(1.0), 100)
            .is_err());
    }

    #[test]
    fn rollup_outcome_report_reaches_offline_readiness_with_diverse_complete_data() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let rollups: Vec<String> = (0..20)
            .map(|index| rec(&db, &format!("rollup {index}"), index as f32 + 1.0))
            .collect();
        let selected = rec(&db, "selected child", 30.0);
        let unselected = rec(&db, "unselected child", 31.0);

        for index in 0..200 {
            let impression = db
                .note_rollup_impression(
                    &rollups[index % rollups.len()],
                    &format!("question {}", index % 40),
                    None,
                    index % 4,
                    0.8,
                    None,
                )
                .unwrap();
            db.note_rollup_expansion(&impression, &[&selected, &unselected])
                .unwrap();
            db.finalize_rollup_outcome(&impression, &[&selected], &[])
                .unwrap();
        }

        let report = db.rollup_outcome_report(None, None).unwrap();
        assert_eq!(report.finalized_impressions, 200);
        assert_eq!(report.finalized_distinct_queries, 40);
        assert_eq!(report.finalized_distinct_rollups, 20);
        assert_eq!(report.finalized_selected_children, 200);
        assert_eq!(report.explicitly_unselected_children, 200);
        assert_eq!(report.telemetry_completion_rate, Some(1.0));
        assert_eq!(report.evidence_status, "ready_for_offline_evaluation");
        assert!(report.readiness_failures.is_empty());
    }

    #[test]
    fn label_request_rider_asks_once_per_query_rid() {
        // nuron's labeling economics: ≤2 rids per response, nearest the
        // relevance gate, and a (query, rid) pair is proposed at most
        // once EVER — skipping is free because a skip is never re-asked.
        let db = YantrikDB::new(":memory:", 8).unwrap();
        for i in 0..4 {
            rec(&db, &format!("fact {i}"), 1.0 + i as f32 * 0.02);
        }
        let respond = || {
            db.recall_with_response(
                &vec_seed(1.0, 8),
                5,
                None,
                None,
                false,
                false,
                None,
                false,
                None,
                None,
                None,
            )
            .unwrap()
        };
        let first = respond().coverage.label_request;
        assert!(!first.is_empty() && first.len() <= 2, "{first:?}");

        let second = respond().coverage.label_request;
        for rid in &first {
            assert!(
                !second.contains(rid),
                "same (query, rid) must never be re-asked: {second:?}"
            );
        }
        // Two more calls exhaust the 4-record pool; a further identical
        // query has nothing left to ask.
        let _ = respond();
        let exhausted = respond().coverage.label_request;
        assert!(
            exhausted.is_empty(),
            "pool exhausted — no repeat requests: {exhausted:?}"
        );
    }

    #[test]
    fn explicit_feedback_creates_bound_label() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let a = rec(&db, "fact a", 1.0);
        recall_all(&db, 1.0);
        db.recall_feedback(Some("q"), None, &a, "irrelevant", Some(0.4), Some(0))
            .unwrap();
        let (source, polarity, weight): (String, i32, f64) = db
            .conn()
            .query_row(
                "SELECT source, polarity, weight FROM ranking_labels WHERE rid = ?1",
                params![a],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(source, "explicit");
        assert_eq!(polarity, -1);
        assert!((weight - WEIGHT_EXPLICIT).abs() < 1e-12);
    }
}
