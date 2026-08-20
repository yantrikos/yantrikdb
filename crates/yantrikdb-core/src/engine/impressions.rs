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
use crate::types::RecallResult;

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
        if rollup_rid.trim().is_empty() || query_text.trim().is_empty() || !score.is_finite() {
            return Err(crate::error::YantrikDbError::InvalidInput(
                "rollup impression requires non-empty rollup_rid/query_text and a finite score"
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
             (impression_id, rollup_rid, query_hash, namespace, rank, score, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                impression_id,
                rollup_rid,
                query_hash,
                stored_namespace,
                rank as i64,
                score,
                ts,
            ],
        )?;
        let existing: (String, String, String, i64, f64) = conn.query_row(
            "SELECT rollup_rid, query_hash, namespace, rank, score \
             FROM rollup_impressions WHERE impression_id = ?1",
            params![impression_id],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
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
        let mut clean = Vec::new();
        for rid in returned_child_rids.iter().copied() {
            if !rid.trim().is_empty() && !clean.contains(&rid) {
                clean.push(rid);
            }
        }
        let payload_hash = stable_text_hash(&clean.join("\u{0}"));
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
             (impression_id, child_rid, rank) VALUES (?1, ?2, ?3)",
        )?;
        for (rank, child_rid) in clean.iter().enumerate() {
            stmt.execute(params![impression_id, child_rid, rank as i64])?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim).map(|i| (seed + i as f32) * 0.1).collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    }

    fn rec(db: &YantrikDB, text: &str, seed: f32) -> String {
        db.record(
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
        )
        .unwrap()
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
        let conn = db.conn();
        let outcomes: i64 = conn
            .query_row("SELECT COUNT(*) FROM rollup_impression_outcomes", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(outcomes, 2);

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
    fn rollup_expansion_requires_an_exact_impression() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        assert!(db.note_rollup_expansion("missing", &[]).is_err());
        assert!(!db
            .note_rollup_selection("missing", "child", "selected")
            .unwrap());
        assert!(db
            .note_rollup_selection("missing", "child", "implicit")
            .is_err());
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
