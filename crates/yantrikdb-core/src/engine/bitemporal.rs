//! Bitemporal recall — `recall_as_of(query, t)`: what did this database
//! believe at time `t`?
//!
//! The engine never silently loses state: `correct()` mutates in place
//! but archives the prior text/metadata/importance/valence in
//! `record_revisions` (with `applied_at`), and supersession is an edge
//! in `record_links` (with `created_at`). Those two ledgers are enough
//! to answer an as-of query WITHOUT a new storage model:
//!
//!   1. a record that did not exist at `t` is excluded
//!      (`memories.created_at > t`);
//!   2. a record superseded at `t` is excluded — but only by edges that
//!      already existed at `t` (`record_links.created_at <= t`); a later
//!      supersession does not rewrite what was believed then;
//!   3. a record corrected after `t` is rolled back to its `t`-state by
//!      walking `record_revisions` for the earliest revision applied
//!      after `t` — its `prior_*` columns ARE the state at `t`.
//!
//! HONEST LIMITS, documented rather than hidden:
//!
//! - **Ranking is present-day.** Similarity is computed against today's
//!   vectors (a corrected record's embedding cannot be un-embedded;
//!   `prior_embedding_hash` lets history explain the drift, not undo
//!   it), and decay/importance components are today's. `recall_as_of`
//!   reconstructs CONTENT and CURRENCY at `t`, not the exact ranking a
//!   query would have produced at `t`.
//! - **Forgotten records stay forgotten.** A tombstoned record is
//!   physically out of the recall path (cache and index), so an as-of
//!   query cannot resurface something forgotten since `t`. The oplog
//!   holds that history for explicit forensics; recall does not.
//! - **Access stats are untouched.** The pool query runs with
//!   `skip_reinforce` — archaeology must not masquerade as usage.
//!
//! Results keep their present-day `current_status`: a hit can honestly
//! read "this was current at `t`; today it is superseded", which is
//! exactly what an audit wants to see.

use rusqlite::{params, OptionalExtension};

use crate::error::Result;
use crate::types::RecallResult;

use super::YantrikDB;

/// Over-fetch multiplier for the candidate pool: as-of filtering can
/// only shrink the pool, so the underlying recall casts a wider net.
const AS_OF_POOL_FACTOR: usize = 4;
/// Upper bound on the pool a single as-of query may hydrate.
const AS_OF_POOL_CAP: usize = 100;

impl YantrikDB {
    /// Recall against the state of belief at `as_of` (epoch seconds).
    ///
    /// See the module docs for exact semantics and honest limits.
    pub fn recall_as_of(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        as_of: f64,
        namespace: Option<&str>,
        memory_type: Option<&str>,
    ) -> Result<Vec<RecallResult>> {
        let fetch_k = top_k
            .saturating_mul(AS_OF_POOL_FACTOR)
            .clamp(top_k, AS_OF_POOL_CAP.max(top_k));
        let mut pool = self.recall(
            query_embedding,
            fetch_k,
            None, // time_window — as-of does its own temporal filtering
            memory_type,
            false, // include_consolidated
            // Aligned 2026-08-15: graph expansion measured −0.24 MRR and is
            // off by default on every other surface; this pre-decision
            // hardcode was never revisited (no comment defended it), and
            // as_of passes query_text=None — the weakest seeding mode.
            false, // expand_entities
            None,  // query_text
            true,  // skip_reinforce: archaeology is not usage
            namespace,
            None, // domain
            None, // source
            None, // certainty_min
            None, // order
            true, // include_superseded: the as-of filter decides, not today's
        )?;

        // 1. Nothing that did not exist yet.
        pool.retain(|r| r.created_at <= as_of);

        // 2. Nothing already superseded at `as_of`. Edges created after
        //    `as_of` do not suppress what was still current then.
        if !pool.is_empty() {
            let rids: Vec<&str> = pool.iter().map(|r| r.rid.as_str()).collect();
            let superseded = self.superseded_rids_as_of(&rids, as_of)?;
            if !superseded.is_empty() {
                pool.retain(|r| !superseded.contains(&r.rid));
            }
        }

        // 3. Roll corrected records back to their state at `as_of`.
        for result in pool.iter_mut() {
            self.rollback_result_to(result, as_of)?;
        }

        pool.truncate(top_k);
        Ok(pool)
    }

    /// Which of `rids` were superseded by an edge that already existed
    /// at `as_of`. Same shape as `superseded_rids_among`, with the
    /// edge-creation clock applied.
    fn superseded_rids_as_of(
        &self,
        rids: &[&str],
        as_of: f64,
    ) -> Result<std::collections::HashSet<String>> {
        if rids.is_empty() {
            return Ok(std::collections::HashSet::new());
        }
        let placeholders: String = (0..rids.len())
            .map(|i| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT DISTINCT target_rid FROM record_links \
             WHERE link_type = 'supersedes' \
             AND status = 'active' AND selection_state = 'selected' \
             AND created_at <= ?{} \
             AND target_rid IN ({placeholders})",
            rids.len() + 1
        );
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        for r in rids {
            param_values.push(Box::new(r.to_string()));
        }
        param_values.push(Box::new(as_of));
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params_ref.as_slice(), |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<std::collections::HashSet<_>, _>>()?;
        Ok(rows)
    }

    /// If `result`'s record was corrected after `as_of`, swap in the
    /// state it had at `as_of`. The earliest revision applied AFTER
    /// `as_of` archived exactly the pre-correction state we want.
    fn rollback_result_to(&self, result: &mut RecallResult, as_of: f64) -> Result<()> {
        let conn = self.read_conn();
        let prior = conn
            .query_row(
                "SELECT prior_text, prior_metadata, prior_importance, \
                        prior_valence \
                 FROM record_revisions \
                 WHERE rid = ?1 AND applied_at > ?2 \
                 ORDER BY revision_num ASC LIMIT 1",
                params![result.rid, as_of],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, f64>(2)?,
                        row.get::<_, f64>(3)?,
                    ))
                },
            )
            .optional()?;
        if let Some((text, metadata, importance, valence)) = prior {
            // Revisions are archived in STORED (encrypted-or-plain) form by
            // design (see the correct() revision insert). Every other read
            // path decrypts on hydration; this one assigned the stored form
            // straight into the result — on encrypted DBs the caller got
            // ciphertext text and Null metadata (serde on ciphertext),
            // silently. 2026-08-15 surface audit.
            result.text = self.decrypt_text(&text)?;
            let metadata_plain = self.decrypt_text(&metadata)?;
            result.metadata =
                serde_json::from_str(&metadata_plain).unwrap_or(serde_json::Value::Null);
            result.importance = importance;
            result.valence = valence;
            result
                .why_retrieved
                .push("as_of: rolled back to pre-correction state".to_string());
        }
        Ok(())
    }
}
