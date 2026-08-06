//! C4 — the claims lane (wheel piece 3): retrieval finally reads the
//! store that knows direction.
//!
//! The write path extracts directional relations losslessly into
//! `claims` (src, rel_type, dst, polarity, source_memory_rid) — and
//! until this lane, NO retrieval code consulted them ("the substrate
//! stores the answer; retrieval never reads it"). Measured cost on the
//! stress gate: every query whose answer existed in claims with
//! correct direction was missed, because cosine destroys
//! subject/object direction (`Taylor reports to Carol` scored BELOW
//! `Pat reports to Taylor` for the query "taylor") and the co-mention
//! entity graph is undirected.
//!
//! Shape: resolve query entities with the (post-C5a, alias-folded)
//! graph index, look up their claims by src OR dst, and admit each
//! claim's SOURCE RECORD into the candidate pool with a why that
//! carries the full directional provenance — "claims_match:
//! Taylor -reports_to-> Carol (anchor Taylor)". The lane is exact
//! evidence (an index lookup, not a heuristic), so admitted candidates
//! also get keyword-reserve eligibility at full lexical strength: the
//! same rescue guarantee that flipped the exact-phrase repro.
//!
//! One definition, both recall twins — the copy-a-pattern law.

use rusqlite::{params, Connection};

use crate::graph_index::GraphIndex;

/// Cap on query entities consulted — a query rarely names more.
const MAX_ANCHOR_ENTITIES: usize = 4;
/// Cap on claims admitted per anchor entity; keeps the lane an
/// index lookup, never a scan.
const MAX_CLAIMS_PER_ENTITY: usize = 24;

/// A claims-lane candidate: the claim's source record plus the
/// directional provenance that justifies its admission.
pub(crate) struct ClaimCandidate {
    pub rid: String,
    /// e.g. `claims_match: Taylor -reports_to-> Carol (anchor Taylor)`
    pub why: String,
}

/// Resolve `query_tokens` to entities and return the source records of
/// their claims. Best-effort by design: a missing `claims` table (old
/// packs) or any read error yields an empty lane, never a failed
/// recall. Duplicate rids keep their first (best-anchored) why.
pub(crate) fn claims_candidates(
    conn: &Connection,
    graph_index: &GraphIndex,
    query_tokens: &[String],
    namespace: Option<&str>,
) -> Vec<ClaimCandidate> {
    let mut anchors = graph_index.entity_matches_query(query_tokens);
    if anchors.is_empty() {
        return Vec::new();
    }
    // Strongest anchors first (mention count), bounded.
    anchors.sort_by(|a, b| b.2.cmp(&a.2));
    anchors.truncate(MAX_ANCHOR_ENTITIES);

    let mut out: Vec<ClaimCandidate> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (entity, _etype, _mentions) in &anchors {
        let sql = format!(
            "SELECT src, rel_type, dst, source_memory_rid, polarity FROM claims \
             WHERE (src = ?1 OR dst = ?1) AND tombstoned = 0 \
             AND source_memory_rid IS NOT NULL {} \
             ORDER BY created_at DESC LIMIT {}",
            if namespace.is_some() {
                "AND namespace = ?2"
            } else {
                ""
            },
            MAX_CLAIMS_PER_ENTITY,
        );
        let Ok(mut stmt) = conn.prepare_cached(&sql) else {
            return out; // no claims table — empty lane, never an error
        };
        let rows: Vec<(String, String, String, String, i64)> = {
            let mapper = |row: &rusqlite::Row| -> rusqlite::Result<_> {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            };
            let res = if let Some(ns) = namespace {
                stmt.query_map(params![entity, ns], mapper)
                    .map(|r| r.filter_map(|x| x.ok()).collect())
            } else {
                stmt.query_map(params![entity], mapper)
                    .map(|r| r.filter_map(|x| x.ok()).collect())
            };
            res.unwrap_or_default()
        };
        for (src, rel, dst, rid, polarity) in rows {
            if !seen.insert(rid.clone()) {
                continue;
            }
            let neg = if polarity < 0 { "NOT " } else { "" };
            out.push(ClaimCandidate {
                why: format!("claims_match: {src} -{neg}{rel}-> {dst} (anchor {entity})"),
                rid,
            });
        }
    }
    out
}

impl super::YantrikDB {
    /// Apply the claims lane to a recall candidate pool: boost pool
    /// members whose records back a claim about a query entity, and
    /// admit source records the vector/FTS lanes missed. The boost is
    /// keyword-strength at full lexical weight (lex = 1.0) — a claim is
    /// exact evidence, stronger than any term statistic — and the
    /// `claims_match:` why makes the candidate keyword-reserve-eligible
    /// (see `lexical::apply_keyword_reserve`), the same rescue
    /// guarantee that flipped the exact-phrase repro.
    ///
    /// Shared by `recall_inner` and `recall_profiled_inner` — one
    /// definition, two callers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_claims_lane(
        &self,
        scored: &mut Vec<crate::types::RecallResult>,
        query_embedding: &[f32],
        query_text: Option<&str>,
        namespace: Option<&str>,
        time_window: Option<(f64, f64)>,
        include_consolidated: bool,
        learned_weights: &crate::types::LearnedWeights,
        ts: f64,
        query_sentiment: f64,
    ) -> crate::error::Result<()> {
        use crate::base::scoring;

        let Some(qt) = query_text else {
            return Ok(());
        };
        let cands = {
            let gi = self.graph_index.read();
            let tokens = crate::graph::tokenize(qt);
            let conn = self.read_conn();
            claims_candidates(&conn, &gi, &tokens, namespace)
        };
        if cands.is_empty() {
            return Ok(());
        }

        let mut by_rid: std::collections::HashMap<&str, &str> = cands
            .iter()
            .map(|c| (c.rid.as_str(), c.why.as_str()))
            .collect();

        // Members already in the pool: boost + stamp provenance.
        for result in scored.iter_mut() {
            if let Some(why) = by_rid.remove(result.rid.as_str()) {
                if !result
                    .why_retrieved
                    .iter()
                    .any(|w| w.starts_with("claims_match"))
                {
                    result.score += super::lexical::keyword_lane_boost(
                        learned_weights.keyword_boost,
                        result.scores.similarity,
                        1.0,
                    );
                    result.why_retrieved.push(why.to_string());
                }
            }
        }

        // Source records no lane admitted: score them in. No cosine
        // floor here — the lane's whole point is that the record's
        // embedding may be arbitrarily far from the query while the
        // claim is exact.
        let new_rids: Vec<(&str, &str)> = by_rid.into_iter().collect();
        if new_rids.is_empty() {
            return Ok(());
        }
        let rid_refs: Vec<&str> = new_rids.iter().map(|(r, _)| *r).collect();
        let emb_map = self.fetch_embeddings_by_rids(&rid_refs)?;
        let cache = self.scoring_cache.read();
        for (rid, claim_why) in new_rids {
            let Some(row) = cache.get(rid) else { continue };
            let status_ok = if include_consolidated {
                row.consolidation_status == "active" || row.consolidation_status == "consolidated"
            } else {
                row.consolidation_status == "active"
            };
            if !status_ok {
                continue;
            }
            if let Some((start, end)) = time_window {
                if row.created_at < start || row.created_at > end {
                    continue;
                }
            }
            let Some(emb_blob) = emb_map.get(rid) else {
                continue;
            };
            let mem_emb = crate::serde_helpers::deserialize_f32(emb_blob);
            let sim_score = crate::consolidate::cosine_similarity(query_embedding, &mem_emb) as f64;
            let elapsed = ts - row.last_access;
            let decay = scoring::decay_score(row.importance, row.half_life, elapsed);
            let age = ts - row.created_at;
            let recency = scoring::recency_score(age);
            let composite = scoring::adaptive_composite_score(
                sim_score,
                decay,
                recency,
                row.importance,
                row.valence,
                query_sentiment,
                learned_weights,
            );
            let boost =
                super::lexical::keyword_lane_boost(learned_weights.keyword_boost, sim_score, 1.0);
            let mut why = scoring::build_why(sim_score, recency, decay, row.valence);
            why.push(claim_why.to_string());
            let contributions = scoring::adaptive_contributions(
                sim_score,
                decay,
                recency,
                row.importance,
                learned_weights,
            );
            let valence_multiplier = scoring::query_valence_boost(row.valence, query_sentiment);
            scored.push(crate::types::RecallResult {
                rid: rid.to_string(),
                memory_type: row.memory_type.clone(),
                text: String::new(),
                created_at: row.created_at,
                importance: row.importance,
                valence: row.valence,
                score: composite + boost,
                scores: crate::types::ScoreBreakdown {
                    similarity: sim_score,
                    decay,
                    recency,
                    importance: row.importance,
                    graph_proximity: 0.0,
                    contributions,
                    valence_multiplier,
                },
                why_retrieved: why,
                metadata: serde_json::Value::Null,
                namespace: row.namespace.clone(),
                certainty: row.certainty,
                domain: row.domain.clone(),
                source: row.source.clone(),
                emotional_state: row.emotional_state.clone(),
                current_status: Default::default(),
                superseded_by: None,
                disputed_with: Vec::new(),
                aged_last_verified: None,
                best_span: None,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_provenance_is_spelled_out() {
        // Pure formatting contract — the why must let a consumer see
        // WHO is subject without opening the record.
        let c = ClaimCandidate {
            rid: "r".into(),
            why: format!(
                "claims_match: {} -{}{}-> {} (anchor {})",
                "Taylor", "", "reports_to", "Carol", "Taylor"
            ),
        };
        assert_eq!(
            c.why,
            "claims_match: Taylor -reports_to-> Carol (anchor Taylor)"
        );
    }
}
