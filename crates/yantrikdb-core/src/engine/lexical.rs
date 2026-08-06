//! BM25 lexical fusion — the keyword lane learns to tell a rare-term
//! exact match from common-term frame noise.
//!
//! The defect this fixes (reproduced from a production report,
//! tests/repros/exact_phrase_starvation.py): a record containing the
//! query phrase VERBATIM ranked nowhere in top-10 while 150
//! frame-shaped distractors filled the results. Three stacked causes,
//! all measured on the repro:
//!
//! 1. **Dilution** — mean-pooled static embedders push a long record's
//!    vector toward its dominant topic; cos(query, verbatim record)
//!    measured 0.057 at dim 64. The lexical lane is the only route in.
//! 2. **Flat boost** — the old boost `keyword_boost * (1 - sim)` paid
//!    every FTS candidate alike, so common-term noise ("failure",
//!    "class" in every frame record) got the same credit as an
//!    all-terms-including-rare match. FTS5 computes bm25() for every
//!    match and the engine discarded it.
//! 3. **Expired calibration** — KEYWORD_RESERVE_MIN_SIM = 0.25 was
//!    calibrated on 384-dim MiniLM cosines; at dim 64 the rescue lane
//!    refused sim 0.057–0.20, i.e. exactly the records it was built to
//!    rescue.
//!
//! The fusion: capture the raw bm25 rank FTS5 already computes,
//! normalize per query to a strength in (0, 1] (best match = 1), and
//! (a) scale the keyword boost by that strength — the query's best
//! lexical match keeps the old boost magnitude (whose +0.034 MRR
//! contribution on the production clone is the measured baseline to
//! preserve), noise pays its own discount; (b) admit reserve slots by
//! lexical strength OR cosine, so an exact match is rescuable at any
//! similarity; (c) let a top-tier lexical match bypass the FTS_MIN_SIM
//! cosine floor, which is itself a dim-calibrated constant.
//!
//! Everything here is shared by `recall_inner` and
//! `recall_profiled_inner` — the two keyword lanes are copies, and a
//! formula patched in one and not the other is a silent divergence
//! (the copy-a-pattern failure class). One definition, two callers.

use crate::base::types::RecallResult;
use std::collections::HashMap;

/// Reserve up to this many top_k slots for keyword-matched candidates
/// buried below the score cutoff.
pub(crate) const KEYWORD_RESERVE_SLOTS: usize = 3;

/// Cosine admission floor for a reserve slot. A 384-dim calibration —
/// kept as the OR-branch for high-dim installs where cosine is still
/// informative, but no longer the only door (see
/// [`KEYWORD_RESERVE_MIN_LEX`]).
pub(crate) const KEYWORD_RESERVE_MIN_SIM: f64 = 0.25;

/// Lexical admission floor for a reserve slot: at least half the
/// query's best bm25 strength. Dimension-independent — bm25 is
/// computed on the text, not the embedding.
pub(crate) const KEYWORD_RESERVE_MIN_LEX: f64 = 0.5;

/// A candidate at or above this lexical strength may enter the pool
/// even below the FTS_MIN_SIM cosine floor. Deliberately stricter than
/// [`KEYWORD_RESERVE_MIN_LEX`]: bypassing the noise floor entirely is
/// reserved for near-best lexical matches (the verbatim-phrase case),
/// not everything that clears reserve admission.
pub(crate) const LEX_STRONG: f64 = 0.75;

/// Per-query lexical strength from raw FTS5 bm25 ranks.
///
/// Input: `(rid, rank)` rows as FTS5 returns them — rank is NEGATIVE,
/// more negative = better match. Duplicates (a rid seen by both the
/// AND and OR pass, or by phase 1 and phase 2) keep their best rank.
///
/// Output: rid → strength in (0, 1], computed as `rank / best_rank`.
/// The query's best match gets exactly 1.0; a doc matching only the
/// common terms of a five-term query lands well below
/// [`KEYWORD_RESERVE_MIN_LEX`]. When bm25 does NOT discriminate (all
/// matches near-equal, e.g. a single-keyword query over uniform text)
/// every strength is ~1.0 and the lane behaves exactly as it did
/// before fusion — the discount only exists where the evidence does.
pub(crate) fn lexical_strengths(ranked: &[(String, f64)]) -> HashMap<String, f64> {
    let mut best: HashMap<String, f64> = HashMap::new();
    for (rid, rank) in ranked {
        best.entry(rid.clone())
            .and_modify(|r| *r = r.min(*rank))
            .or_insert(*rank);
    }
    if best.is_empty() {
        return best;
    }
    let top = best.values().fold(f64::INFINITY, |a, &r| a.min(r));
    if top >= -1e-9 {
        // Degenerate: no negative ranks (bm25 never returns this for a
        // real match). Treat everything as equally strong rather than
        // dividing by ~zero.
        for v in best.values_mut() {
            *v = 1.0;
        }
        return best;
    }
    for v in best.values_mut() {
        *v = (*v / top).clamp(0.0, 1.0);
    }
    best
}

/// The keyword-lane boost: the pre-fusion `keyword_boost * (1 - sim)`
/// scaled by lexical strength. `lex = 1.0` reproduces the old formula
/// bit-for-bit, so the measured +0.034 contribution of the lane's
/// genuine matches is preserved; weaker matches pay `lex` as a direct
/// discount.
pub(crate) fn keyword_lane_boost(keyword_boost: f64, sim: f64, lex: f64) -> f64 {
    keyword_boost * lex.clamp(0.0, 1.0) * (1.0 - sim).max(0.2)
}

/// Keyword slot reservation (step 3.5 of recall): sort by score, then
/// lift the best keyword-matched candidates stranded below the top_k
/// cutoff to just above it. Admission is lexical strength OR cosine;
/// candidates are ranked by strength first, cosine as tie-break —
/// an exact-phrase match with a diluted embedding beats a paraphrase
/// with a middling one for the rescue slots.
pub(crate) fn apply_keyword_reserve(
    scored: &mut [RecallResult],
    lex_by_rid: &HashMap<String, f64>,
    top_k: usize,
) {
    scored.sort_by(|a, b| b.score.total_cmp(&a.score));

    let cutoff_idx = top_k.min(scored.len()).saturating_sub(1);
    let cutoff_score = scored.get(cutoff_idx).map(|r| r.score).unwrap_or(0.0);

    let mut kw_below: Vec<(usize, f64, f64)> = scored
        .iter()
        .enumerate()
        .filter(|(_, r)| {
            r.why_retrieved.iter().any(|w| w == "keyword_match") && r.score < cutoff_score
        })
        .map(|(i, r)| {
            let lex = lex_by_rid.get(r.rid.as_str()).copied().unwrap_or(0.0);
            (i, lex, r.scores.similarity)
        })
        .filter(|(_, lex, sim)| *lex >= KEYWORD_RESERVE_MIN_LEX || *sim >= KEYWORD_RESERVE_MIN_SIM)
        .collect();
    kw_below.sort_by(|a, b| b.1.total_cmp(&a.1).then(b.2.total_cmp(&a.2)));

    for (idx, _, _) in kw_below.into_iter().take(KEYWORD_RESERVE_SLOTS) {
        scored[idx].score = cutoff_score + 0.001;
        scored[idx]
            .why_retrieved
            .push("keyword_reserved".to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::types::{RecallResult, ScoreBreakdown, ScoreContributions};

    fn hit(rid: &str, score: f64, sim: f64, why: &[&str]) -> RecallResult {
        RecallResult {
            rid: rid.to_string(),
            memory_type: "semantic".to_string(),
            text: String::new(),
            created_at: 0.0,
            importance: 0.5,
            valence: 0.0,
            score,
            scores: ScoreBreakdown {
                similarity: sim,
                decay: 0.0,
                recency: 0.0,
                importance: 0.5,
                graph_proximity: 0.0,
                contributions: ScoreContributions {
                    similarity: 0.0,
                    decay: 0.0,
                    recency: 0.0,
                    importance: 0.0,
                    graph_proximity: 0.0,
                },
                valence_multiplier: 1.0,
            },
            why_retrieved: why.iter().map(|s| s.to_string()).collect(),
            metadata: serde_json::Value::Null,
            namespace: "default".to_string(),
            certainty: 0.8,
            domain: "general".to_string(),
            source: "user".to_string(),
            emotional_state: None,
            current_status: Default::default(),
            superseded_by: None,
            disputed_with: Vec::new(),
            aged_last_verified: None,
            best_span: None,
        }
    }

    #[test]
    fn strengths_normalize_best_to_one_and_noise_below_reserve_floor() {
        // A five-term query: the verbatim record matches all terms
        // (bm25 -12.4), frame noise matches the two common ones (-2.1).
        let ranked = vec![
            ("verbatim".to_string(), -12.4),
            ("noise1".to_string(), -2.1),
            ("noise2".to_string(), -1.7),
        ];
        let lex = lexical_strengths(&ranked);
        assert_eq!(lex["verbatim"], 1.0);
        assert!(lex["noise1"] < KEYWORD_RESERVE_MIN_LEX);
        assert!(lex["noise2"] < lex["noise1"]);
    }

    #[test]
    fn strengths_keep_best_rank_across_duplicate_rows() {
        // Same rid from the AND pass (-9.0) and the OR pass (-9.0) and
        // an importance-phase row (-4.0 from a partial-group query):
        // the best rank wins, never the last-seen.
        let ranked = vec![
            ("a".to_string(), -4.0),
            ("a".to_string(), -9.0),
            ("b".to_string(), -9.0),
        ];
        let lex = lexical_strengths(&ranked);
        assert_eq!(lex["a"], 1.0);
        assert_eq!(lex["b"], 1.0);
    }

    #[test]
    fn uniform_ranks_reproduce_the_pre_fusion_lane() {
        // When bm25 does not discriminate, every strength is 1.0 and
        // keyword_lane_boost equals the old flat formula exactly.
        let ranked = vec![("x".to_string(), -3.0), ("y".to_string(), -3.0)];
        let lex = lexical_strengths(&ranked);
        for rid in ["x", "y"] {
            let old = 0.31 * (1.0f64 - 0.4).max(0.2);
            assert_eq!(keyword_lane_boost(0.31, 0.4, lex[rid]), old);
        }
    }

    #[test]
    fn degenerate_non_negative_ranks_do_not_divide_by_zero() {
        let ranked = vec![("a".to_string(), 0.0), ("b".to_string(), 0.0)];
        let lex = lexical_strengths(&ranked);
        assert_eq!(lex["a"], 1.0);
        assert_eq!(lex["b"], 1.0);
        assert!(lexical_strengths(&[]).is_empty());
    }

    #[test]
    fn reserve_admits_diluted_exact_match_by_lexical_strength() {
        // The hermes case: verbatim-phrase record at sim 0.057 — the
        // old sim>=0.25 door refused it; the lexical door admits it.
        let mut scored = vec![
            hit("top1", 0.90, 0.60, &[]),
            hit("top2", 0.80, 0.55, &[]),
            hit("top3", 0.70, 0.50, &[]),
            hit("verbatim", 0.10, 0.057, &["keyword_match"]),
        ];
        let mut lex = HashMap::new();
        lex.insert("verbatim".to_string(), 1.0);
        apply_keyword_reserve(&mut scored, &lex, 3);
        let v = scored.iter().find(|r| r.rid == "verbatim").unwrap();
        assert!(
            v.why_retrieved.iter().any(|w| w == "keyword_reserved"),
            "lexical-top match must win a reserve slot at any cosine"
        );
        assert!(v.score > 0.70, "lifted just above the cutoff");
    }

    #[test]
    fn reserve_still_admits_by_cosine_and_refuses_weak_noise() {
        // A keyword match with decent cosine but no lexical entry
        // (grandfathered door) is admitted; a weak-both candidate is not.
        let mut scored = vec![
            hit("top1", 0.90, 0.60, &[]),
            hit("top2", 0.80, 0.55, &[]),
            hit("top3", 0.70, 0.50, &[]),
            hit("cosine_ok", 0.20, 0.30, &["keyword_match"]),
            hit("weak_noise", 0.15, 0.10, &["keyword_match"]),
        ];
        let mut lex = HashMap::new();
        lex.insert("cosine_ok".to_string(), 0.2);
        lex.insert("weak_noise".to_string(), 0.2);
        apply_keyword_reserve(&mut scored, &lex, 3);
        assert!(scored
            .iter()
            .find(|r| r.rid == "cosine_ok")
            .unwrap()
            .why_retrieved
            .iter()
            .any(|w| w == "keyword_reserved"));
        assert!(!scored
            .iter()
            .find(|r| r.rid == "weak_noise")
            .unwrap()
            .why_retrieved
            .iter()
            .any(|w| w == "keyword_reserved"));
    }

    #[test]
    fn reserve_ranks_by_lexical_strength_before_cosine() {
        // Four admissible candidates, three slots: the lowest lexical
        // strength (cosine-door admit) loses the contest.
        let mut scored = vec![
            hit("top1", 0.90, 0.60, &[]),
            hit("top2", 0.85, 0.55, &[]),
            hit("top3", 0.80, 0.50, &[]),
            hit("lex_hi", 0.10, 0.05, &["keyword_match"]),
            hit("lex_mid", 0.10, 0.06, &["keyword_match"]),
            hit("lex_low_cos_hi", 0.10, 0.40, &["keyword_match"]),
            hit("lex_hi2", 0.10, 0.07, &["keyword_match"]),
        ];
        let mut lex = HashMap::new();
        lex.insert("lex_hi".to_string(), 1.0);
        lex.insert("lex_mid".to_string(), 0.8);
        lex.insert("lex_hi2".to_string(), 0.9);
        lex.insert("lex_low_cos_hi".to_string(), 0.1);
        apply_keyword_reserve(&mut scored, &lex, 3);
        let reserved: Vec<&str> = scored
            .iter()
            .filter(|r| r.why_retrieved.iter().any(|w| w == "keyword_reserved"))
            .map(|r| r.rid.as_str())
            .collect();
        assert_eq!(reserved.len(), KEYWORD_RESERVE_SLOTS);
        assert!(reserved.contains(&"lex_hi"));
        assert!(reserved.contains(&"lex_hi2"));
        assert!(reserved.contains(&"lex_mid"));
        assert!(!reserved.contains(&"lex_low_cos_hi"));
    }
}
