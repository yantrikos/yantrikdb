//! Recall demand log + knowledge-gap surface (v0.9.0).
//!
//! The substrate's "known unknowns." Every user-facing recall records, in a
//! cheap O(1) aggregate keyed by the *normalized* query, how often that query
//! is asked and how well it's answered (result count + best-hit score). From
//! that, [`YantrikDB::knowledge_gaps`] surfaces the queries people keep asking
//! that return little or nothing — the demand the memory *should* satisfy but
//! doesn't.
//!
//! This closes the retrieval-demand loop: no other memory layer tells you what
//! it's missing. Bounded by distinct-query cardinality (not total recalls);
//! the write is one indexed UPSERT, in the same spirit as the per-recall
//! `reinforce` the engine already does, and gated off for internal/eval recalls
//! (`skip_reinforce`).

use rusqlite::{params, OptionalExtension};

use crate::error::Result;

use super::{now, YantrikDB};

/// A frequently-asked, poorly-answered query — a demand the substrate should
/// satisfy but can't.
#[derive(Debug, Clone, serde::Serialize)]
pub struct KnowledgeGap {
    /// A recent raw form of the query.
    pub query: String,
    /// How many times it has been asked.
    pub count: u64,
    /// Mean best-hit score across those asks (low = poorly answered).
    pub avg_top_score: f64,
    /// Mean number of results returned.
    pub avg_results: f64,
    pub last_seen: f64,
}

/// Normalize a query into a cluster key: lowercase, whitespace-collapsed,
/// trailing terminal punctuation stripped. So "Who owns the rotation?" and
/// "who owns the rotation" aggregate together.
pub(crate) fn normalize_query(q: &str) -> String {
    q.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_end_matches(['?', '.', '!', ' '])
        .to_string()
}

impl YantrikDB {
    /// Record one recall's outcome into the demand aggregate. Best-effort and
    /// cheap; called automatically on user-facing recalls (not internal ones).
    pub(crate) fn record_recall_demand(
        &self,
        query_text: &str,
        result_count: usize,
        top_score: f64,
    ) -> Result<()> {
        let norm = normalize_query(query_text);
        if norm.is_empty() {
            return Ok(());
        }
        let conn = self.conn();
        conn.execute(
            "INSERT INTO recall_demand \
             (query_norm, sample_text, count, sum_top_score, sum_results, last_seen) \
             VALUES (?1, ?2, 1, ?3, ?4, ?5) \
             ON CONFLICT(query_norm) DO UPDATE SET \
               count = count + 1, \
               sample_text = ?2, \
               sum_top_score = sum_top_score + ?3, \
               sum_results = sum_results + ?4, \
               last_seen = ?5",
            params![norm, query_text, top_score, result_count as i64, now()],
        )?;
        Ok(())
    }

    /// Surface knowledge gaps: queries asked at least `min_count` times whose
    /// mean best-hit score is at or below `max_avg_top_score` (i.e. frequently
    /// asked, poorly answered), most-asked first.
    pub fn knowledge_gaps(
        &self,
        min_count: u64,
        max_avg_top_score: f64,
        limit: usize,
    ) -> Result<Vec<KnowledgeGap>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT sample_text, count, sum_top_score, sum_results, last_seen \
             FROM recall_demand \
             WHERE count >= ?1 AND (sum_top_score / count) <= ?2 \
             ORDER BY count DESC, (sum_top_score / count) ASC \
             LIMIT ?3",
        )?;
        let rows = stmt
            .query_map(
                params![min_count as i64, max_avg_top_score, limit as i64],
                |r| {
                    let count: i64 = r.get(1)?;
                    let sum_top: f64 = r.get(2)?;
                    let sum_res: i64 = r.get(3)?;
                    let n = count.max(1) as f64;
                    Ok(KnowledgeGap {
                        query: r.get(0)?,
                        count: count.max(0) as u64,
                        avg_top_score: sum_top / n,
                        avg_results: sum_res as f64 / n,
                        last_seen: r.get(4)?,
                    })
                },
            )?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Point lookup of the demand stats for one query (mainly for tests/ops):
    /// returns (count, avg_top_score) or None.
    pub fn recall_demand_for(&self, query_text: &str) -> Result<Option<(u64, f64)>> {
        let norm = normalize_query(query_text);
        let conn = self.conn();
        let row = conn
            .query_row(
                "SELECT count, sum_top_score FROM recall_demand WHERE query_norm = ?1",
                params![norm],
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?)),
            )
            .optional()?;
        Ok(row.map(|(c, s)| (c.max(0) as u64, s / c.max(1) as f64)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_clusters_punctuation_and_case() {
        assert_eq!(
            normalize_query("Who Owns  the rotation?"),
            "who owns the rotation"
        );
        assert_eq!(
            normalize_query("who owns the rotation"),
            "who owns the rotation"
        );
    }

    #[cfg(feature = "bundled-embedder")]
    #[test]
    fn knowledge_gaps_surface_frequent_low_yield_queries() {
        let db = YantrikDB::with_default(":memory:").unwrap();
        // A frequently-asked, poorly-answered query.
        for _ in 0..5 {
            db.record_recall_demand("how do I configure the widget", 0, 0.0)
                .unwrap();
        }
        // A frequently-asked, WELL-answered query (should NOT be a gap).
        for _ in 0..5 {
            db.record_recall_demand("who is alice", 3, 1.1).unwrap();
        }
        // A rare poorly-answered query (below min_count, not yet a gap).
        db.record_recall_demand("obscure one-off question", 0, 0.0)
            .unwrap();

        let gaps = db.knowledge_gaps(3, 0.4, 10).unwrap();
        assert_eq!(gaps.len(), 1, "only the frequent low-yield query: {gaps:?}");
        assert_eq!(gaps[0].query, "how do I configure the widget");
        assert_eq!(gaps[0].count, 5);
        assert!(gaps[0].avg_top_score <= 0.4);
    }
}
