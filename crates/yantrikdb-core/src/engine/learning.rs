//! v0.10 Item 2 — the self-sufficient learning loop (v2).
//!
//! Rewrites the v0.4-era coordinate-descent loop around the impression
//! ledger, per the three-seat validity review. The prior loop had two
//! disqualifying defects: it reconstructed historical features from
//! CURRENT mutable memory state (exposure-confounded — the engine
//! reinforces what it returns), and it re-trained on the same cumulative
//! feedback every cognition tick (MAX_DELTA bounded one generation, not
//! total drift on reused evidence).
//!
//! v2 rules (sol's rulings, nuron's consumer review):
//!
//! - **Labels**: only `explicit`, `rejected_refine`, `caller_used` —
//!   bound to impressions. Being served is never evidence. A read-only
//!   workload stays at generation 0 forever: the database abstains from
//!   learning rather than teach itself that its own answers were correct.
//! - **Features**: read from `recall_impressions` (frozen at serve
//!   time), never rebuilt from mutable state.
//! - **Pairs**: only within one impression episode, only between a
//!   positively- and a negatively-labeled rid. Non-action is not a
//!   negative; non-rejection is not a positive. Pair mass is normalized
//!   per episode so long result lists cannot dominate.
//! - **Optimizer**: deterministic bounded coordinate search over the
//!   REAL scorer (`adaptive_composite_score` + recorded keyword boost)
//!   with pairwise logistic loss — not a linear proxy. Fixed parameter
//!   order, fixed passes, no RNG. (`base::fitting` holds the generic
//!   linear primitives + PAV; this loop needs the scorer-aware variant.)
//! - **Gates**: distinct preference-bearing query EPISODES, not rows.
//!   Explicit path ≥ 20, implicit-only ≥ 50 (with a smaller drift
//!   allowance), rid diversity, fresh evidence beyond the last fit's
//!   watermark.
//! - **Champion/challenger**: query-grouped FORWARD validation (train
//!   on the older episodes, validate on the newest; never row k-fold).
//!   Swap only if held-out loss improves ≥ 0.02 absolute AND the
//!   candidate is no worse on ≥ 60% of validation episodes AND any
//!   explicit-labeled slice does not regress AND every parameter stays
//!   inside the per-generation drift bound.
//! - **Watermark**: every fit attempt (accepted or rejected) records
//!   the newest label it consumed; the next fit requires genuinely new
//!   evidence.
//! - **Rollback**: post-swap, the previous weights shadow-score new
//!   preference episodes; if the champion is worse by ≥ 0.02 over ≥ 20
//!   distinct post-swap episodes, roll back to last-good.

use std::collections::{BTreeMap, HashMap};

use rusqlite::params;

use crate::error::Result;
use crate::scoring;
use crate::types::LearnedWeights;

use super::{now, YantrikDB};

/// Minimum distinct preference-bearing episodes when at least one side
/// of some pair is explicit feedback.
const MIN_EPISODES_EXPLICIT: usize = 20;
/// Minimum distinct preference-bearing episodes when ALL evidence is
/// implicit (rejected_refine / caller_used only).
const MIN_EPISODES_IMPLICIT_ONLY: usize = 50;
/// Minimum distinct validation episodes for a swap decision.
const MIN_VALIDATION_EPISODES: usize = 10;
/// Minimum distinct rids across the training evidence.
const MIN_RID_DIVERSITY: usize = 8;
/// Distinct NEW preference-bearing episodes required beyond the last
/// fit's evidence watermark before another fit may run.
const MIN_NEW_EPISODES: usize = 5;
/// Required absolute held-out loss improvement for a swap.
const SWAP_MARGIN: f64 = 0.02;
/// Fraction of validation episodes on which the candidate must be no
/// worse than the champion.
const MIN_NO_WORSE_FRACTION: f64 = 0.60;
/// Hard per-generation drift bound per parameter (explicit evidence).
const MAX_DELTA: f64 = 0.05;
/// Tighter drift bound when the fit is implicit-only.
const MAX_DELTA_IMPLICIT: f64 = 0.03;
/// Episode weight multiplier when an episode carries no explicit label.
const IMPLICIT_EPISODE_WEIGHT: f64 = 0.5;
/// Post-swap shadow evaluation: distinct new episodes before a rollback
/// decision, and the regression margin that triggers it.
const ROLLBACK_MIN_EPISODES: usize = 20;
const ROLLBACK_MARGIN: f64 = 0.02;

/// One labeled, feature-frozen impression row.
#[derive(Debug, Clone)]
struct LabeledImpression {
    episode_id: String,
    first_seen: f64, // episode's earliest impression created_at
    f_similarity: f64,
    f_decay: f64,
    f_recency: f64,
    f_importance: f64,
    f_valence: f64,
    keyword_boosted: bool,
    polarity: i32,
    label_weight: f64,
    source: String,
    label_created_at: f64,
}

/// A within-episode preference: `pos` should outscore `neg`.
#[derive(Debug, Clone)]
struct PreferencePair {
    pos: usize, // indices into the episode's rows
    neg: usize,
    weight: f64,
}

/// One query episode's evidence.
#[derive(Debug, Clone)]
struct Episode {
    rows: Vec<LabeledImpression>,
    pairs: Vec<PreferencePair>,
    has_explicit: bool,
    first_seen: f64,
    newest_label: f64,
}

/// Typed outcome of one learning-loop invocation. Persisted (JSON) to
/// `meta.last_learning_report` so a learner failure is never silently
/// indistinguishable from "not enough evidence".
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct LearningReport {
    /// What happened: "swapped" | "rejected" | "rolled_back" |
    /// "insufficient_evidence" | "no_new_evidence".
    pub outcome: String,
    /// Human-readable specifics (which gate failed, by how much).
    pub detail: String,
    /// Distinct preference-bearing episodes available.
    pub distinct_episodes: usize,
    /// Labels by source over the considered evidence.
    pub label_counts: BTreeMap<String, i64>,
    /// Accepted generation after this run (unchanged if no swap).
    pub generation: i64,
    pub train_loss: Option<f64>,
    pub validation_loss: Option<f64>,
    pub champion_validation_loss: Option<f64>,
    /// Safety observable (sol ruling 7): fraction of validation episodes
    /// where the candidate drops the highest-similarity impression from
    /// its top-3 while the champion kept it. Early warning, not a veto.
    pub semantic_anchor_drop_rate: Option<f64>,
    /// Standing invariant (asserted every run): labels can only carry
    /// the three caller-driven sources — engine resurfacing can never
    /// mint a positive. Always 0 by construction; exposed so consumer
    /// seats can assert it.
    pub engine_resurface_positive_count: i64,
    /// How much potential signal the label-binding horizon discarded:
    /// impressions older than the horizon that never received a label
    /// (nuron's review — lets the 7-day value be tuned from data
    /// instead of argued).
    pub expired_unlabeled_impressions: i64,
}

/// Score an impression's frozen features under candidate weights,
/// through the REAL composite scorer (neutral query sentiment — the
/// impression predates sentiment capture) plus the recorded keyword
/// boost, mirroring the serve-time score assembly.
fn score_impression(w: &LearnedWeights, r: &LabeledImpression) -> f64 {
    let mut s = scoring::adaptive_composite_score(
        r.f_similarity,
        r.f_decay,
        r.f_recency,
        r.f_importance,
        r.f_valence,
        0.0,
        w,
    );
    if r.keyword_boosted {
        s += w.keyword_boost * (1.0 - r.f_similarity).max(0.2);
    }
    s
}

/// Weighted mean pairwise logistic loss of `w` over `episodes`.
/// Returns `None` when there are no usable pairs — the caller must
/// treat that as "no evaluation", never as a zero-loss win.
fn episodes_loss(w: &LearnedWeights, episodes: &[&Episode]) -> Option<f64> {
    let mut total_w = 0.0;
    let mut total_loss = 0.0;
    for ep in episodes {
        for p in &ep.pairs {
            let z = score_impression(w, &ep.rows[p.pos]) - score_impression(w, &ep.rows[p.neg]);
            // −ln σ(z), computed stably.
            let loss = if z > 0.0 {
                (1.0 + (-z).exp()).ln()
            } else {
                -z + (1.0 + z.exp()).ln()
            };
            total_loss += p.weight * loss;
            total_w += p.weight;
        }
    }
    if total_w > 0.0 {
        Some(total_loss / total_w)
    } else {
        None
    }
}

/// Legal ranges per parameter (mirrors the historical clamps).
const BOUND_W: (f64, f64) = (0.05, 0.90);
const BOUND_TAU: (f64, f64) = (0.10, 0.50);
const BOUND_ALPHA: (f64, f64) = (0.10, 1.50);
const BOUND_KW: (f64, f64) = (0.0, 1.0);

fn apply_param(w: &mut LearnedWeights, idx: usize, value: f64) {
    match idx {
        0 => w.w_sim = value.clamp(BOUND_W.0, BOUND_W.1),
        1 => w.w_decay = value.clamp(BOUND_W.0, BOUND_W.1),
        2 => w.w_recency = value.clamp(BOUND_W.0, BOUND_W.1),
        3 => w.gate_tau = value.clamp(BOUND_TAU.0, BOUND_TAU.1),
        4 => w.alpha_imp = value.clamp(BOUND_ALPHA.0, BOUND_ALPHA.1),
        5 => w.keyword_boost = value.clamp(BOUND_KW.0, BOUND_KW.1),
        _ => unreachable!(),
    }
}

fn get_param(w: &LearnedWeights, idx: usize) -> f64 {
    match idx {
        0 => w.w_sim,
        1 => w.w_decay,
        2 => w.w_recency,
        3 => w.gate_tau,
        4 => w.alpha_imp,
        5 => w.keyword_boost,
        _ => unreachable!(),
    }
}

/// Normalize the three base blend weights to sum 1 (five effective
/// degrees of freedom — same convention the scorer assumes).
fn normalize_base(w: &mut LearnedWeights) {
    let sum = w.w_sim + w.w_decay + w.w_recency;
    if sum > 0.0 {
        w.w_sim /= sum;
        w.w_decay /= sum;
        w.w_recency /= sum;
    }
}

/// Deterministic bounded coordinate search over the six scorer
/// parameters: fixed order, fixed passes, halving step, accept only
/// strict improvements of training loss. No RNG, no data-dependent
/// iteration counts.
fn fit_candidate(train: &[&Episode], champion: &LearnedWeights) -> LearnedWeights {
    let mut best = champion.clone();
    normalize_base(&mut best);
    let Some(mut best_loss) = episodes_loss(&best, train) else {
        return best;
    };
    let mut step = 0.16;
    for _pass in 0..10 {
        for idx in 0..6 {
            for dir in [1.0, -1.0] {
                let mut cand = best.clone();
                apply_param(&mut cand, idx, get_param(&best, idx) + dir * step);
                if idx < 3 {
                    normalize_base(&mut cand);
                }
                if let Some(loss) = episodes_loss(&cand, train) {
                    if loss < best_loss {
                        best = cand;
                        best_loss = loss;
                    }
                }
            }
        }
        step *= 0.5;
    }
    best
}

/// Clamp every parameter of `cand` to within `max_delta` of `champion`
/// — the hard per-generation drift bound. Deliberately the FINAL
/// operation, with NO re-normalization after it: renormalizing would
/// scale parameters back outside the bound (the historical loop's
/// clamp-then-renormalize overshoot). The base blend's sum may deviate
/// from 1 by at most 3·max_delta for one generation; the next fit's
/// normalize step re-anchors it. The clamped candidate is what gets
/// EVALUATED — never evaluate weights you wouldn't ship.
fn clamp_drift(cand: &LearnedWeights, champion: &LearnedWeights, max_delta: f64) -> LearnedWeights {
    let mut out = cand.clone();
    for idx in 0..6 {
        let c = get_param(champion, idx);
        let v = get_param(cand, idx).clamp(c - max_delta, c + max_delta);
        apply_param(&mut out, idx, v);
    }
    out
}

/// Fraction of validation episodes where `cand` drops the episode's
/// highest-similarity impression from its top-3 while `champ` retained
/// it — the semantic-anchor early-warning observable (sol ruling 7).
fn semantic_anchor_drop_rate(
    cand: &LearnedWeights,
    champ: &LearnedWeights,
    validation: &[&Episode],
) -> Option<f64> {
    let mut considered = 0usize;
    let mut dropped = 0usize;
    for ep in validation {
        if ep.rows.len() < 4 {
            continue; // top-3 membership is vacuous on tiny episodes
        }
        let anchor = ep
            .rows
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.f_similarity.total_cmp(&b.1.f_similarity))
            .map(|(i, _)| i)?;
        let in_top3 = |w: &LearnedWeights| {
            let mut scores: Vec<(usize, f64)> = ep
                .rows
                .iter()
                .enumerate()
                .map(|(i, r)| (i, score_impression(w, r)))
                .collect();
            scores.sort_by(|a, b| b.1.total_cmp(&a.1));
            scores.iter().take(3).any(|(i, _)| *i == anchor)
        };
        considered += 1;
        if in_top3(champ) && !in_top3(cand) {
            dropped += 1;
        }
    }
    if considered > 0 {
        Some(dropped as f64 / considered as f64)
    } else {
        None
    }
}

impl YantrikDB {
    /// Run the self-sufficient learning loop. Returns a typed report;
    /// persists it to `meta.last_learning_report` so failures and
    /// abstentions are observable, not silent.
    pub fn run_learning(&self) -> Result<LearningReport> {
        let report = self.run_learning_inner();
        if let Ok(ref r) = report {
            if let Ok(json) = serde_json::to_string(r) {
                let _ = self.conn().execute(
                    "INSERT OR REPLACE INTO meta (key, value) \
                     VALUES ('last_learning_report', ?1)",
                    params![json],
                );
            }
        }
        report
    }

    fn run_learning_inner(&self) -> Result<LearningReport> {
        let champion = self.load_learned_weights()?;
        let mut report = LearningReport {
            generation: champion.generation,
            ..Default::default()
        };

        // Standing invariant: the schema CHECK constraint makes a
        // 'served' label source unrepresentable; count anything outside
        // the three caller-driven sources (must be zero).
        report.engine_resurface_positive_count = self.conn().query_row(
            "SELECT COUNT(*) FROM ranking_labels \
             WHERE source NOT IN ('explicit', 'rejected_refine', 'caller_used')",
            [],
            |r| r.get(0),
        )?;

        // Horizon-discard observable: impressions past the label-binding
        // horizon that never got a label (tunes the horizon from data).
        report.expired_unlabeled_impressions = self.conn().query_row(
            "SELECT COUNT(*) FROM recall_impressions i \
             WHERE i.created_at < ?1 AND NOT EXISTS \
               (SELECT 1 FROM ranking_labels l \
                WHERE l.episode_id = i.episode_id AND l.rid = i.rid)",
            params![crate::time::now_secs() - 7.0 * 86_400.0],
            |r| r.get(0),
        )?;

        // Post-swap shadow check FIRST: an accepted generation must
        // prove itself on evidence it has never seen before we consider
        // fitting the next one.
        if let Some(rollback) = self.check_rollback(&champion)? {
            return Ok(rollback);
        }

        let episodes = self.load_preference_episodes()?;
        for ep in &episodes {
            for row in &ep.rows {
                *report.label_counts.entry(row.source.clone()).or_insert(0) += 1;
            }
        }
        report.distinct_episodes = episodes.len();

        // Gates: distinct preference-bearing episodes, not rows.
        let has_explicit = episodes.iter().any(|e| e.has_explicit);
        let min_episodes = if has_explicit {
            MIN_EPISODES_EXPLICIT
        } else {
            MIN_EPISODES_IMPLICIT_ONLY
        };
        if episodes.len() < min_episodes {
            report.outcome = "insufficient_evidence".into();
            report.detail = format!(
                "{} preference-bearing episodes < {} required ({})",
                episodes.len(),
                min_episodes,
                if has_explicit {
                    "explicit path"
                } else {
                    "implicit-only path"
                },
            );
            return Ok(report);
        }
        let distinct_rids: std::collections::HashSet<&str> = episodes
            .iter()
            .flat_map(|e| e.rows.iter().map(|r| r.episode_id.as_str()))
            .collect();
        // (episode_id above is a typo guard: rid diversity below)
        drop(distinct_rids);
        let rid_diversity: std::collections::HashSet<String> = {
            let conn = self.conn();
            let mut stmt = conn.prepare("SELECT DISTINCT rid FROM ranking_labels")?;
            let rows = stmt
                .query_map([], |r| r.get::<_, String>(0))?
                .collect::<std::result::Result<_, _>>()?;
            rows
        };
        if rid_diversity.len() < MIN_RID_DIVERSITY {
            report.outcome = "insufficient_evidence".into();
            report.detail = format!(
                "rid diversity {} < {MIN_RID_DIVERSITY}",
                rid_diversity.len()
            );
            return Ok(report);
        }

        // Fresh-evidence watermark: require genuinely new episodes since
        // the last fit ATTEMPT (accepted or rejected) — the same
        // cumulative evidence must not drive repeated fitting.
        let watermark: f64 = self.conn().query_row(
            "SELECT COALESCE(MAX(evidence_watermark), 0) FROM learned_weights_history",
            [],
            |r| r.get(0),
        )?;
        let new_episodes = episodes
            .iter()
            .filter(|e| e.newest_label > watermark)
            .count();
        if new_episodes < MIN_NEW_EPISODES {
            report.outcome = "no_new_evidence".into();
            report.detail = format!(
                "{new_episodes} episodes with labels beyond watermark {watermark:.0} < {MIN_NEW_EPISODES}"
            );
            return Ok(report);
        }

        // Query-grouped FORWARD split: train on the oldest episodes,
        // validate on the newest (durable first-seen order, episode id
        // tie-break — never row-level k-fold).
        let mut ordered: Vec<&Episode> = episodes.iter().collect();
        ordered.sort_by(|a, b| {
            a.first_seen
                .total_cmp(&b.first_seen)
                .then_with(|| a.rows[0].episode_id.cmp(&b.rows[0].episode_id))
        });
        let val_count = MIN_VALIDATION_EPISODES.max(ordered.len() / 4);
        if ordered.len() < val_count + MIN_VALIDATION_EPISODES {
            report.outcome = "insufficient_evidence".into();
            report.detail = format!(
                "{} episodes cannot fund a {val_count}-episode validation split",
                ordered.len()
            );
            return Ok(report);
        }
        let (train, validation) = ordered.split_at(ordered.len() - val_count);

        // Fit on train; clamp to the drift bound BEFORE evaluation.
        let max_delta = if has_explicit {
            MAX_DELTA
        } else {
            MAX_DELTA_IMPLICIT
        };
        let fitted = fit_candidate(train, &champion);
        let mut candidate = clamp_drift(&fitted, &champion, max_delta);
        candidate.generation = champion.generation + 1;

        report.train_loss = episodes_loss(&candidate, train);
        let cand_val = episodes_loss(&candidate, validation);
        let champ_val = episodes_loss(&champion, validation);
        report.validation_loss = cand_val;
        report.champion_validation_loss = champ_val;
        report.semantic_anchor_drop_rate =
            semantic_anchor_drop_rate(&candidate, &champion, validation);

        let consumed_watermark = episodes
            .iter()
            .map(|e| e.newest_label)
            .fold(watermark, f64::max);

        // Swap decision.
        let (Some(cand_val), Some(champ_val)) = (cand_val, champ_val) else {
            report.outcome = "rejected".into();
            report.detail = "validation slice produced no usable pairs".into();
            self.record_fit_attempt(&candidate, &report, "rejected", consumed_watermark)?;
            return Ok(report);
        };
        let improves = champ_val - cand_val >= SWAP_MARGIN;
        let mut no_worse = 0usize;
        let mut comparable = 0usize;
        let mut explicit_regressed = false;
        for ep in validation {
            let slice = [*ep];
            if let (Some(c), Some(ch)) = (
                episodes_loss(&candidate, &slice),
                episodes_loss(&champion, &slice),
            ) {
                comparable += 1;
                if c <= ch + 1e-12 {
                    no_worse += 1;
                }
                if ep.has_explicit && c > ch + 1e-12 {
                    explicit_regressed = true;
                }
            }
        }
        let no_worse_ok =
            comparable > 0 && (no_worse as f64 / comparable as f64) >= MIN_NO_WORSE_FRACTION;

        if improves && no_worse_ok && !explicit_regressed {
            self.swap_champion(&candidate, &report, consumed_watermark)?;
            report.outcome = "swapped".into();
            report.detail = format!(
                "held-out loss {champ_val:.4} -> {cand_val:.4} ({} of {} validation episodes no worse)",
                no_worse, comparable
            );
            report.generation = candidate.generation;
        } else {
            report.outcome = "rejected".into();
            report.detail = format!(
                "improves={improves} (Δ={:.4} vs {SWAP_MARGIN}), no_worse={no_worse}/{comparable}, explicit_regressed={explicit_regressed}",
                champ_val - cand_val
            );
            self.record_fit_attempt(&candidate, &report, "rejected", consumed_watermark)?;
        }
        Ok(report)
    }

    /// Post-swap shadow evaluation: compare the active generation
    /// against last-good on preference episodes that arrived AFTER the
    /// swap. Regression ≥ ROLLBACK_MARGIN over ≥ ROLLBACK_MIN_EPISODES
    /// distinct episodes → restore last-good.
    fn check_rollback(&self, champion: &LearnedWeights) -> Result<Option<LearningReport>> {
        if champion.generation == 0 {
            return Ok(None);
        }
        use rusqlite::OptionalExtension;
        let prior: Option<(String, f64)> = self
            .conn()
            .query_row(
                "SELECT weights_json, fitted_at FROM learned_weights_history \
                 WHERE generation = ?1 AND status = 'active'",
                params![champion.generation],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let Some((_, fitted_at)) = prior else {
            return Ok(None);
        };
        let last_good: Option<String> = self
            .conn()
            .query_row(
                "SELECT weights_json FROM learned_weights_history \
                 WHERE generation < ?1 AND status IN ('active', 'superseded') \
                 ORDER BY generation DESC LIMIT 1",
                params![champion.generation],
                |r| r.get(0),
            )
            .optional()?;
        let last_good: LearnedWeights = match last_good {
            Some(json) => serde_json::from_str(&json).unwrap_or_default(),
            // Generation 1's predecessor is the shipped defaults.
            None => LearnedWeights::default(),
        };

        let episodes = self.load_preference_episodes()?;
        let post_swap: Vec<&Episode> = episodes
            .iter()
            .filter(|e| e.newest_label > fitted_at)
            .collect();
        if post_swap.len() < ROLLBACK_MIN_EPISODES {
            return Ok(None);
        }
        let (Some(current), Some(good)) = (
            episodes_loss(champion, &post_swap),
            episodes_loss(&last_good, &post_swap),
        ) else {
            return Ok(None);
        };
        if current - good < ROLLBACK_MARGIN {
            return Ok(None);
        }

        // Regression confirmed on unseen evidence: restore last-good.
        let mut restored = last_good.clone();
        restored.generation = champion.generation; // history keys stay unique
        self.save_learned_weights(&restored)?;
        self.conn().execute(
            "UPDATE learned_weights_history SET status = 'rolled_back' \
             WHERE generation = ?1",
            params![champion.generation],
        )?;
        let mut report = LearningReport {
            outcome: "rolled_back".into(),
            detail: format!(
                "post-swap loss {current:.4} vs last-good {good:.4} over {} episodes",
                post_swap.len()
            ),
            generation: restored.generation,
            distinct_episodes: post_swap.len(),
            ..Default::default()
        };
        report.validation_loss = Some(current);
        report.champion_validation_loss = Some(good);
        Ok(Some(report))
    }

    /// Load preference-bearing episodes: labeled impressions joined to
    /// their frozen serve-time features, grouped by episode, with pairs
    /// formed only between positive and negative labels of the SAME
    /// episode and pair mass normalized per episode.
    fn load_preference_episodes(&self) -> Result<Vec<Episode>> {
        let conn = self.conn();
        // Impressions recorded before `ranking_feature_epoch` were frozen
        // under a DIFFERENT meaning of `f_decay` — "time since last read"
        // rather than "the record's own age" (see MIGRATE_V40_TO_V41). The
        // column name is identical, so nothing but this boundary
        // distinguishes them, and fitting across it trains one weight on two
        // quantities. Fresh stores stamp the epoch at 0 and filter nothing.
        let epoch: f64 = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'ranking_feature_epoch'",
                [],
                |r| r.get::<_, String>(0),
            )
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0);
        let mut stmt = conn.prepare(
            "SELECT i.episode_id, i.rid, i.f_similarity, i.f_decay, i.f_recency, \
                    i.f_importance, i.f_valence, i.keyword_boosted, i.created_at, \
                    l.polarity, l.weight, l.source, l.created_at \
             FROM ranking_labels l \
             JOIN recall_impressions i \
               ON i.episode_id = l.episode_id AND i.rid = l.rid \
             WHERE i.created_at >= ?1 \
             ORDER BY i.episode_id, i.rank",
        )?;
        let rows = stmt
            .query_map(params![epoch], |r| {
                Ok(LabeledImpression {
                    episode_id: r.get(0)?,
                    first_seen: r.get(8)?,
                    f_similarity: r.get(2)?,
                    f_decay: r.get(3)?,
                    f_recency: r.get(4)?,
                    f_importance: r.get(5)?,
                    f_valence: r.get(6)?,
                    keyword_boosted: r.get::<_, i64>(7)? != 0,
                    polarity: r.get(9)?,
                    label_weight: r.get(10)?,
                    source: r.get(11)?,
                    label_created_at: r.get(12)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        drop(stmt);
        drop(conn);

        let mut by_episode: HashMap<String, Vec<LabeledImpression>> = HashMap::new();
        for row in rows {
            by_episode
                .entry(row.episode_id.clone())
                .or_default()
                .push(row);
        }

        let mut episodes = Vec::new();
        // Deterministic episode order (BTreeMap semantics via sort).
        let mut keys: Vec<String> = by_episode.keys().cloned().collect();
        keys.sort();
        for key in keys {
            let rows = by_episode.remove(&key).expect("key from map");
            let pos: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].polarity > 0).collect();
            let neg: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].polarity < 0).collect();
            if pos.is_empty() || neg.is_empty() {
                continue; // not preference-bearing
            }
            let has_explicit = rows.iter().any(|r| r.source == "explicit");
            let episode_scale = if has_explicit {
                1.0
            } else {
                IMPLICIT_EPISODE_WEIGHT
            };
            let mut pairs = Vec::with_capacity(pos.len() * neg.len());
            let mut mass = 0.0;
            for &p in &pos {
                for &n in &neg {
                    let w = rows[p].label_weight * rows[n].label_weight;
                    mass += w;
                    pairs.push(PreferencePair {
                        pos: p,
                        neg: n,
                        weight: w,
                    });
                }
            }
            // Normalize per episode so long lists cannot dominate.
            if mass > 0.0 {
                for p in &mut pairs {
                    p.weight = p.weight / mass * episode_scale;
                }
            }
            let first_seen = rows
                .iter()
                .map(|r| r.first_seen)
                .fold(f64::INFINITY, f64::min);
            let newest_label = rows.iter().map(|r| r.label_created_at).fold(0.0, f64::max);
            episodes.push(Episode {
                rows,
                pairs,
                has_explicit,
                first_seen,
                newest_label,
            });
        }
        Ok(episodes)
    }

    /// Persist a fit attempt to history WITHOUT touching the live row.
    fn record_fit_attempt(
        &self,
        candidate: &LearnedWeights,
        report: &LearningReport,
        status: &str,
        watermark: f64,
    ) -> Result<()> {
        let conn = self.conn();
        conn.execute(
            "INSERT OR REPLACE INTO learned_weights_history \
             (generation, weights_json, fitted_at, train_loss, validation_loss, \
              champion_validation_loss, label_counts_json, distinct_queries, \
              swap_reason, status, evidence_watermark) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                candidate.generation,
                serde_json::to_string(candidate).unwrap_or_default(),
                now(),
                report.train_loss,
                report.validation_loss,
                report.champion_validation_loss,
                serde_json::to_string(&report.label_counts).unwrap_or_default(),
                report.distinct_episodes as i64,
                report.detail,
                status,
                watermark,
            ],
        )?;
        Ok(())
    }

    /// Atomically accept a challenger: mark the previous active history
    /// row superseded, insert the new active row, update the live
    /// weights — one transaction.
    fn swap_champion(
        &self,
        candidate: &LearnedWeights,
        report: &LearningReport,
        watermark: f64,
    ) -> Result<()> {
        {
            let conn = self.conn();
            let sp = crate::engine::savepoint::SavepointGuard::new(&conn, "weight_swap")?;

            conn.execute(
                "UPDATE learned_weights_history SET status = 'superseded' \
                     WHERE status = 'active'",
                [],
            )?;
            conn.execute(
                "INSERT OR REPLACE INTO learned_weights_history \
                     (generation, weights_json, fitted_at, train_loss, validation_loss, \
                      champion_validation_loss, label_counts_json, distinct_queries, \
                      swap_reason, status, evidence_watermark) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'active', ?10)",
                params![
                    candidate.generation,
                    serde_json::to_string(candidate).unwrap_or_default(),
                    now(),
                    report.train_loss,
                    report.validation_loss,
                    report.champion_validation_loss,
                    serde_json::to_string(&report.label_counts).unwrap_or_default(),
                    report.distinct_episodes as i64,
                    "held-out improvement",
                    watermark,
                ],
            )?;
            conn.execute(
                "UPDATE learned_weights SET \
                     w_sim = ?1, w_decay = ?2, w_recency = ?3, \
                     gate_tau = ?4, alpha_imp = ?5, keyword_boost = ?6, \
                     updated_at = ?7, generation = ?8 \
                     WHERE id = 1",
                params![
                    candidate.w_sim,
                    candidate.w_decay,
                    candidate.w_recency,
                    candidate.gate_tau,
                    candidate.alpha_imp,
                    candidate.keyword_boost,
                    now(),
                    candidate.generation,
                ],
            )?;

            sp.release()?;
        }
        Ok(())
    }

    /// Save updated weights to the live singleton row.
    fn save_learned_weights(&self, weights: &LearnedWeights) -> Result<()> {
        let ts = now();
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE learned_weights SET \
             w_sim = ?1, w_decay = ?2, w_recency = ?3, \
             gate_tau = ?4, alpha_imp = ?5, keyword_boost = ?6, \
             updated_at = ?7, generation = ?8 \
             WHERE id = 1",
            params![
                weights.w_sim,
                weights.w_decay,
                weights.w_recency,
                weights.gate_tau,
                weights.alpha_imp,
                weights.keyword_boost,
                ts,
                weights.generation,
            ],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim).map(|i| (seed + i as f32) * 0.1).collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    }

    /// Inject one synthetic preference-bearing episode straight into the
    /// ledger: a "good" impression (high similarity, no boosts) labeled
    /// positive and a "bad" impression the DEFAULT weights still
    /// over-rank, labeled negative.
    ///
    /// The bad row's shape moved with the 2026-08-05 recency-wall fix:
    /// the old shape (low sim, high decay/recency) is now ranked
    /// correctly by the DEFAULT formula — freshness became a bounded
    /// multiplier, so there is almost nothing left for the fitter to
    /// learn from it (Δ-loss fell under the swap margin, which is the
    /// structural fix working, not a learning regression). The remaining
    /// ADDITIVE channel the defaults can over-rank is the keyword lane:
    /// a keyword-boosted, barely-similar, fresh record (sim 0.15 + 0.31
    /// boost ≈ 0.39) still edges out a plainly relevant one (0.5 · 0.75
    /// = 0.375). Consistent explicit evidence against that is exactly
    /// what the fitter can fix inside one MAX_DELTA generation (lower
    /// keyword_boost / decay / recency interior weights, raise w_sim).
    /// Deterministic timestamps so the forward split is stable.
    fn inject_episode(db: &YantrikDB, n: usize, source: &str) {
        let conn = db.conn();
        let episode = format!("ep-{n:04}");
        let ts = 1_000_000.0 + n as f64;
        for (rid, rank, sim, decay, recency, kw, polarity) in [
            (
                format!("good-{n}"),
                1,
                0.75_f64,
                0.0_f64,
                0.0_f64,
                0_i64,
                1_i32,
            ),
            (format!("bad-{n}"), 0, 0.15, 0.9, 0.9, 1, -1),
        ] {
            conn.execute(
                "INSERT INTO recall_impressions \
                 (episode_id, rid, rank, f_similarity, f_decay, f_recency, \
                  f_importance, f_valence, keyword_boosted, score, \
                  weight_generation, namespace, query_hash, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0.0, 0.0, ?7, 0.5, 0, 'default', ?8, ?9)",
                params![
                    episode,
                    rid,
                    rank,
                    sim,
                    decay,
                    recency,
                    kw,
                    format!("q{n}"),
                    ts
                ],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO ranking_labels \
                 (label_id, episode_id, rid, source, polarity, weight, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, 1.0, ?6)",
                params![
                    format!("lbl-{n}-{polarity}"),
                    episode,
                    rid,
                    source,
                    polarity,
                    ts + 0.5,
                ],
            )
            .unwrap();
        }
    }

    #[test]
    fn served_only_workload_never_learns() {
        // THE thesis test: a database that is only ever queried — no
        // feedback, no rejection, no caller action — must stay at
        // generation 0 forever. Being served is not evidence.
        let db = YantrikDB::new(":memory:", 8).unwrap();
        for i in 0..30 {
            db.record(
                &format!("fact {i}"),
                "semantic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &vec_seed(i as f32, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        }
        for i in 0..60 {
            db.recall(
                &vec_seed((i % 30) as f32, 8),
                5,
                None,
                None,
                false,
                false,
                None,
                false, // real consumer recalls: impressions ARE logged
                None,
                None,
                None,
                None,
                None,
                false,
                None, // event_after (#149)
                None, // event_before (#149)
            )
            .unwrap();
        }
        let impressions: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM recall_impressions", [], |r| r.get(0))
            .unwrap();
        assert!(impressions > 0, "impressions were logged");

        let report = db.run_learning().unwrap();
        assert_eq!(report.outcome, "insufficient_evidence");
        assert_eq!(report.distinct_episodes, 0, "no preference pairs exist");
        assert_eq!(db.load_learned_weights().unwrap().generation, 0);
        assert_eq!(report.engine_resurface_positive_count, 0);
    }

    #[test]
    fn consistent_explicit_preferences_swap_the_champion() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        for n in 0..40 {
            inject_episode(&db, n, "explicit");
        }
        let report = db.run_learning().unwrap();
        assert_eq!(report.outcome, "swapped", "report: {report:?}");
        assert_eq!(report.generation, 1);

        let w = db.load_learned_weights().unwrap();
        assert_eq!(w.generation, 1);
        // The evidence says similarity is under-weighted relative to
        // decay/recency; the fitted weights must move that way, inside
        // the drift clamp.
        let d = LearnedWeights::default();
        assert!(w.w_sim > d.w_sim, "w_sim rose: {w:?}");
        assert!(
            (w.w_sim - d.w_sim).abs() <= MAX_DELTA + 1e-9,
            "drift bound is exact — clamp is the final operation: {w:?}"
        );
        // History has exactly one active row at generation 1.
        let (status, gen): (String, i64) = db
            .conn()
            .query_row(
                "SELECT status, generation FROM learned_weights_history \
                 WHERE status = 'active'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!((status.as_str(), gen), ("active", 1));

        // Report persisted for diagnostics.
        let report_json: String = db
            .conn()
            .query_row(
                "SELECT value FROM meta WHERE key = 'last_learning_report'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(report_json.contains("\"outcome\":\"swapped\""));
    }

    #[test]
    fn watermark_blocks_refit_on_same_evidence() {
        // The same cumulative evidence must not drive repeated updates
        // every cognition tick.
        let db = YantrikDB::new(":memory:", 8).unwrap();
        for n in 0..40 {
            inject_episode(&db, n, "explicit");
        }
        let first = db.run_learning().unwrap();
        assert_eq!(first.outcome, "swapped");
        let second = db.run_learning().unwrap();
        assert_eq!(
            second.outcome, "no_new_evidence",
            "second tick on identical evidence must abstain: {second:?}"
        );
        assert_eq!(db.load_learned_weights().unwrap().generation, 1);
    }

    #[test]
    fn implicit_only_gate_is_stricter() {
        // 40 preference-bearing episodes is enough for the explicit path
        // but NOT the implicit-only path (>= 50).
        let db = YantrikDB::new(":memory:", 8).unwrap();
        for n in 0..40 {
            inject_episode(&db, n, "caller_used");
        }
        let report = db.run_learning().unwrap();
        assert_eq!(report.outcome, "insufficient_evidence");
        assert!(report.detail.contains("implicit-only"), "{report:?}");

        for n in 40..60 {
            inject_episode(&db, n, "caller_used");
        }
        let report = db.run_learning().unwrap();
        // 60 episodes clears the implicit gate; whether the challenger
        // wins is a swap-rule question, but the loop must at least FIT.
        assert!(
            report.outcome == "swapped" || report.outcome == "rejected",
            "implicit path fits at >= 50 episodes: {report:?}"
        );
        if report.outcome == "swapped" {
            // Implicit-only drift allowance is tighter.
            let w = db.load_learned_weights().unwrap();
            let d = LearnedWeights::default();
            assert!(
                (w.w_sim - d.w_sim).abs() <= MAX_DELTA_IMPLICIT + 1e-9,
                "implicit drift bound is exact: {w:?}"
            );
        }
    }
}
