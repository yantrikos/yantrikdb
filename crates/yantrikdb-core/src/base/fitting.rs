//! v0.10 Item 2 — deterministic, dependency-free fitting primitives.
//!
//! Two hand-rolled estimators, each small enough to audit line-by-line
//! (the three-seat review rejected linfa/smartcore as dependency-tree
//! drag on a SQLite-weight engine):
//!
//! - **Pairwise ranking fit**: Bradley–Terry / pairwise logistic LOSS
//!   over within-query "A should rank above B" judgments, minimized by
//!   deterministic bounded COORDINATE SEARCH (sol's validity ruling:
//!   the loss and the optimizer are separate choices; for ~6 bounded
//!   parameters a fixed-order, fixed-pass coordinate search is adequate
//!   and easier to audit than gradient descent's learning-rate and
//!   convergence knobs). No randomness, no data-dependent iteration
//!   counts — the same input always produces the same weights, so
//!   fitted generations are reproducible in the reliability gate.
//!
//! - **PAV isotonic regression** (pool-adjacent-violators): calibrates
//!   raw ranking scores to observed relevance frequencies without
//!   assuming a parametric shape. Ships as a PRIMITIVE ONLY in v0.10 —
//!   per-DB calibration stays inactive until independent two-class
//!   outcome data with a separate calibration holdout exists (sol:
//!   "calibration does not improve rank order"; activating it on ~50
//!   biased implicit observations would launder bias into confidence).
//!
//! Both are pure functions of their inputs. Policy — which labels are
//! trustworthy, when to fire, whether a fitted candidate replaces the
//! champion — lives in the engine layer, not here.

/// One ranking judgment: within a single query, `preferred` should rank
/// above `other`. `weight` scales this pair's gradient contribution
/// (explicit feedback = 1.0; implicit sources get capped lower weights
/// at the call site so they can never outvote explicit labels).
#[derive(Debug, Clone)]
pub struct RankPair {
    pub preferred: Vec<f64>,
    pub other: Vec<f64>,
    pub weight: f64,
}

/// Per-parameter box bound for the coordinate search: fitted weights
/// stay inside `[lo, hi]` regardless of what the data says. The engine
/// passes the same legal ranges it already enforces on stored weights.
pub type Bound = (f64, f64);

/// Configuration for [`fit_pairwise_ranker`]. Defaults are tuned for
/// the recall-weight problem: ~6 bounded parameters, 50–500 noisy pairs.
#[derive(Debug, Clone)]
pub struct RankerFitConfig {
    /// Fixed number of full coordinate passes. Each pass halves the
    /// step, so the final resolution is `initial_step / 2^(passes-1)`.
    /// Fixed count (no convergence test) keeps iteration behavior
    /// data-independent and bit-reproducible across platforms.
    pub passes: usize,
    /// Step size for the first pass, as an absolute weight delta.
    pub initial_step: f64,
    /// L2 penalty toward `anchor` (NOT toward zero): the objective is
    /// mean pairwise loss + (λ / total_pair_weight)·½‖w − anchor‖², so
    /// the champion-shaped prior RELAXES as labeled evidence
    /// accumulates — one noisy pair yields "champion, barely adjusted",
    /// fifty consistent pairs let the data dominate. The engine layer
    /// additionally applies its hard per-generation drift clamp.
    pub l2_toward_anchor: f64,
}

impl Default for RankerFitConfig {
    fn default() -> Self {
        Self {
            // Cumulative one-direction range is Σ step/2^k ≈ 2·initial,
            // so initial_step = 0.5 lets the search traverse a full
            // [0, 1] parameter range; the anchor prior (not the step
            // schedule) is what bounds drift under sparse evidence.
            passes: 12,
            initial_step: 0.5,
            l2_toward_anchor: 1.0,
        }
    }
}

/// Fit linear ranking weights from pairwise judgments by deterministic
/// bounded coordinate search over the pairwise logistic loss.
///
/// Model: P(preferred ranks above other) = σ(w · (x_pref − x_other)).
/// Optimizer (sol's validity ruling): fixed parameter order, fixed pass
/// count, step halved each pass, each move accepted only if it strictly
/// improves the regularized objective. No gradients, no learning rate,
/// no RNG, no data-dependent iteration counts — the same input always
/// produces the same weights.
///
/// Returns `anchor` clamped to `bounds` when `pairs` is empty or every
/// pair has non-positive weight (no signal → no movement).
pub fn fit_pairwise_ranker(
    pairs: &[RankPair],
    anchor: &[f64],
    bounds: &[Bound],
    config: &RankerFitConfig,
) -> Vec<f64> {
    let dim = anchor.len();
    assert_eq!(bounds.len(), dim, "one bound per parameter");
    let clamp = |i: usize, v: f64| v.clamp(bounds[i].0, bounds[i].1);

    let mut w: Vec<f64> = anchor
        .iter()
        .enumerate()
        .map(|(i, &v)| clamp(i, v))
        .collect();

    let usable: Vec<RankPair> = pairs
        .iter()
        .filter(|p| p.weight > 0.0 && p.preferred.len() == dim && p.other.len() == dim)
        .cloned()
        .collect();
    if usable.is_empty() {
        return w;
    }
    let total_weight: f64 = usable.iter().map(|p| p.weight).sum();
    let prior_scale = config.l2_toward_anchor / total_weight;

    let objective = |w: &[f64]| -> f64 {
        // usable is non-empty with positive weight, so loss is Some.
        let data = pairwise_logistic_loss(&usable, w).unwrap_or(f64::INFINITY);
        let mut prior = 0.0;
        for i in 0..dim {
            let d = w[i] - anchor[i];
            prior += d * d;
        }
        data + prior_scale * 0.5 * prior
    };

    let mut best = objective(&w);
    let mut step = config.initial_step;
    for _pass in 0..config.passes {
        for i in 0..dim {
            for dir in [1.0, -1.0] {
                let candidate_v = clamp(i, w[i] + dir * step);
                if candidate_v == w[i] {
                    continue;
                }
                let old = w[i];
                w[i] = candidate_v;
                let cand = objective(&w);
                if cand < best {
                    best = cand;
                } else {
                    w[i] = old;
                }
            }
        }
        step *= 0.5;
    }
    w
}

/// Weighted mean pairwise logistic loss (negative log-likelihood per
/// unit weight) of `w` on `pairs`. The champion/challenger gate compares
/// this on held-out pairs. Returns `None` when no usable pairs exist —
/// the caller must treat "no evaluation data" as "no swap", never as a
/// zero-loss win.
pub fn pairwise_logistic_loss(pairs: &[RankPair], w: &[f64]) -> Option<f64> {
    let dim = w.len();
    let mut total_w = 0.0;
    let mut total_loss = 0.0;
    for p in pairs {
        if p.weight <= 0.0 || p.preferred.len() != dim || p.other.len() != dim {
            continue;
        }
        let mut z = 0.0;
        for i in 0..dim {
            z += w[i] * (p.preferred[i] - p.other[i]);
        }
        // −ln σ(z), computed stably.
        let loss = if z > 0.0 {
            (1.0 + (-z).exp()).ln()
        } else {
            -z + (1.0 + z.exp()).ln()
        };
        total_loss += p.weight * loss;
        total_w += p.weight;
    }
    if total_w > 0.0 {
        Some(total_loss / total_w)
    } else {
        None
    }
}

/// One calibration observation: a raw score `x` and the outcome `y`
/// (1.0 = relevant, 0.0 = irrelevant, or any bounded target), with a
/// weight.
#[derive(Debug, Clone, Copy)]
pub struct CalPoint {
    pub x: f64,
    pub y: f64,
    pub weight: f64,
}

/// A fitted isotonic (monotone non-decreasing) step function, evaluated
/// by interpolation between block means.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct IsotonicMap {
    /// Block boundaries: (x at block center, fitted y), ascending in x.
    pub knots: Vec<(f64, f64)>,
}

impl IsotonicMap {
    /// Evaluate the calibration map at `x`: linear interpolation between
    /// knots, clamped to the fitted range at the extremes. An empty map
    /// is the identity clamped to [0, 1] (no calibration data → pass
    /// scores through rather than inventing structure).
    pub fn eval(&self, x: f64) -> f64 {
        if self.knots.is_empty() {
            return x.clamp(0.0, 1.0);
        }
        let first = self.knots[0];
        let last = self.knots[self.knots.len() - 1];
        if x <= first.0 {
            return first.1;
        }
        if x >= last.0 {
            return last.1;
        }
        for pair in self.knots.windows(2) {
            let (x0, y0) = pair[0];
            let (x1, y1) = pair[1];
            if x <= x1 {
                if (x1 - x0).abs() < f64::EPSILON {
                    return y1;
                }
                let t = (x - x0) / (x1 - x0);
                return y0 + t * (y1 - y0);
            }
        }
        last.1
    }
}

/// Pool-adjacent-violators isotonic regression.
///
/// Sorts points by `x` (stable; ties keep input order) and pools
/// adjacent blocks whose weighted means violate monotonicity, until the
/// fitted sequence is non-decreasing. O(n log n) for the sort, O(n)
/// pooling. Deterministic. Zero- and negative-weight points are dropped.
pub fn fit_pav_isotonic(points: &[CalPoint]) -> IsotonicMap {
    let mut pts: Vec<CalPoint> = points.iter().copied().filter(|p| p.weight > 0.0).collect();
    if pts.is_empty() {
        return IsotonicMap::default();
    }
    pts.sort_by(|a, b| a.x.total_cmp(&b.x));

    // Each block: (weighted y-sum, weight, weighted x-sum).
    struct Block {
        y_sum: f64,
        w: f64,
        x_sum: f64,
    }
    let mut blocks: Vec<Block> = Vec::with_capacity(pts.len());
    for p in &pts {
        blocks.push(Block {
            y_sum: p.y * p.weight,
            w: p.weight,
            x_sum: p.x * p.weight,
        });
        // Pool while the last two blocks violate monotonicity.
        while blocks.len() >= 2 {
            let n = blocks.len();
            let mean_last = blocks[n - 1].y_sum / blocks[n - 1].w;
            let mean_prev = blocks[n - 2].y_sum / blocks[n - 2].w;
            if mean_prev <= mean_last {
                break;
            }
            let last = blocks.pop().expect("len >= 2");
            let prev = blocks.last_mut().expect("len >= 1");
            prev.y_sum += last.y_sum;
            prev.w += last.w;
            prev.x_sum += last.x_sum;
        }
    }

    IsotonicMap {
        knots: blocks
            .iter()
            .map(|b| (b.x_sum / b.w, b.y_sum / b.w))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    const UNIT: Bound = (0.0, 1.0);

    #[test]
    fn ranker_no_pairs_returns_anchor_unchanged() {
        let anchor = vec![0.5, 0.2, 0.3];
        let bounds = [UNIT; 3];
        let w = fit_pairwise_ranker(&[], &anchor, &bounds, &RankerFitConfig::default());
        assert_eq!(w, anchor);
        // Non-positive weights are "no signal" too.
        let dead = vec![RankPair {
            preferred: vec![1.0, 0.0, 0.0],
            other: vec![0.0, 1.0, 0.0],
            weight: 0.0,
        }];
        let w = fit_pairwise_ranker(&dead, &anchor, &bounds, &RankerFitConfig::default());
        assert_eq!(w, anchor);
    }

    #[test]
    fn ranker_learns_the_discriminating_feature() {
        // Preferred items consistently have higher feature 0; feature 1
        // is anti-correlated; feature 2 is noise (identical). The fit
        // must raise w[0] relative to anchor and lower w[1].
        let anchor = vec![0.3, 0.3, 0.3];
        let bounds = [UNIT; 3];
        let pairs: Vec<RankPair> = (0..40)
            .map(|i| {
                let bump = 0.1 + 0.01 * (i % 5) as f64;
                RankPair {
                    preferred: vec![0.6 + bump, 0.2, 0.5],
                    other: vec![0.3, 0.5 + bump, 0.5],
                    weight: 1.0,
                }
            })
            .collect();
        let w = fit_pairwise_ranker(&pairs, &anchor, &bounds, &RankerFitConfig::default());
        assert!(w[0] > anchor[0], "w0 rose: {w:?}");
        assert!(w[1] < anchor[1], "w1 fell: {w:?}");
        // Identical feature: moving w[2] changes only the prior term,
        // which any move strictly worsens — it must stay at the anchor.
        assert!(
            approx(w[2], anchor[2], 1e-9),
            "identical feature cannot move: {w:?}"
        );
        // And the fit genuinely improves pairwise loss.
        let before = pairwise_logistic_loss(&pairs, &anchor).unwrap();
        let after = pairwise_logistic_loss(&pairs, &w).unwrap();
        assert!(after < before, "loss improved: {before} -> {after}");
    }

    #[test]
    fn ranker_is_deterministic_and_respects_bounds() {
        let anchor = vec![0.5, 0.2];
        let bounds = [(0.05, 0.9), (0.05, 0.9)];
        let pairs = vec![
            RankPair {
                preferred: vec![0.9, 0.1],
                other: vec![0.2, 0.8],
                weight: 1.0,
            },
            RankPair {
                preferred: vec![0.8, 0.3],
                other: vec![0.4, 0.6],
                weight: 0.5,
            },
        ];
        let cfg = RankerFitConfig::default();
        let w1 = fit_pairwise_ranker(&pairs, &anchor, &bounds, &cfg);
        let w2 = fit_pairwise_ranker(&pairs, &anchor, &bounds, &cfg);
        assert_eq!(w1, w2, "bitwise identical across runs");
        for (i, v) in w1.iter().enumerate() {
            assert!(
                *v >= bounds[i].0 && *v <= bounds[i].1,
                "w{i}={v} inside bounds"
            );
        }
    }

    #[test]
    fn ranker_anchor_regularization_bounds_drift_on_sparse_data() {
        // ONE noisy pair must not yank the weights: at minimal evidence
        // the anchor pull dominates and keeps the fit near the champion.
        // The same contradiction repeated 100x IS evidence, and may move
        // the weights much further — the prior must lose to data.
        let anchor = vec![0.5, 0.2, 0.3];
        let bounds = [UNIT; 3];
        let contradiction = RankPair {
            preferred: vec![0.0, 0.0, 1.0],
            other: vec![1.0, 1.0, 0.0],
            weight: 1.0,
        };
        let sparse = fit_pairwise_ranker(
            &[contradiction.clone()],
            &anchor,
            &bounds,
            &RankerFitConfig::default(),
        );
        let mut max_drift_sparse: f64 = 0.0;
        for i in 0..3 {
            max_drift_sparse = max_drift_sparse.max((sparse[i] - anchor[i]).abs());
        }
        assert!(
            max_drift_sparse < 0.5,
            "single pair moved weights too far: {sparse:?}"
        );

        let heavy: Vec<RankPair> = std::iter::repeat_n(contradiction, 100).collect();
        let bulk = fit_pairwise_ranker(&heavy, &anchor, &bounds, &RankerFitConfig::default());
        let mut max_drift_bulk: f64 = 0.0;
        for i in 0..3 {
            max_drift_bulk = max_drift_bulk.max((bulk[i] - anchor[i]).abs());
        }
        assert!(
            max_drift_bulk > max_drift_sparse,
            "accumulating evidence must relax the anchor pull: sparse={max_drift_sparse} bulk={max_drift_bulk}"
        );
    }

    #[test]
    fn ranker_starts_from_clamped_anchor_when_anchor_out_of_bounds() {
        // Defensive: a corrupted champion row outside the legal range
        // must not survive the fit.
        let anchor = vec![1.7, -0.4];
        let bounds = [UNIT; 2];
        let w = fit_pairwise_ranker(&[], &anchor, &bounds, &RankerFitConfig::default());
        assert_eq!(w, vec![1.0, 0.0]);
    }

    #[test]
    fn loss_returns_none_without_usable_pairs() {
        assert!(pairwise_logistic_loss(&[], &[0.5, 0.5]).is_none());
    }

    #[test]
    fn pav_pools_violators_to_monotone_blocks() {
        // Classic PAV example: y = [1, 0] at ascending x must pool to a
        // single block at the weighted mean.
        let pts = vec![
            CalPoint {
                x: 0.1,
                y: 1.0,
                weight: 1.0,
            },
            CalPoint {
                x: 0.9,
                y: 0.0,
                weight: 1.0,
            },
        ];
        let map = fit_pav_isotonic(&pts);
        assert_eq!(map.knots.len(), 1);
        assert!(approx(map.knots[0].1, 0.5, 1e-12));
    }

    #[test]
    fn pav_preserves_already_monotone_data_and_interpolates() {
        let pts: Vec<CalPoint> = [(0.1, 0.0), (0.4, 0.5), (0.9, 1.0)]
            .iter()
            .map(|&(x, y)| CalPoint { x, y, weight: 1.0 })
            .collect();
        let map = fit_pav_isotonic(&pts);
        assert_eq!(map.knots.len(), 3, "no pooling needed");
        assert!(approx(map.eval(0.4), 0.5, 1e-12));
        assert!(approx(map.eval(0.25), 0.25, 1e-9), "midpoint interpolates");
        assert!(approx(map.eval(-5.0), 0.0, 1e-12), "clamped low");
        assert!(approx(map.eval(5.0), 1.0, 1e-12), "clamped high");
    }

    #[test]
    fn pav_weighted_pooling_respects_weights() {
        // Violating pair with unequal weights pools to the WEIGHTED mean.
        let pts = vec![
            CalPoint {
                x: 0.2,
                y: 1.0,
                weight: 3.0,
            },
            CalPoint {
                x: 0.8,
                y: 0.0,
                weight: 1.0,
            },
        ];
        let map = fit_pav_isotonic(&pts);
        assert_eq!(map.knots.len(), 1);
        assert!(approx(map.knots[0].1, 0.75, 1e-12));
    }

    #[test]
    fn pav_empty_is_identity_clamped() {
        let map = fit_pav_isotonic(&[]);
        assert!(approx(map.eval(0.3), 0.3, 1e-12));
        assert!(approx(map.eval(1.7), 1.0, 1e-12));
        assert!(approx(map.eval(-0.2), 0.0, 1e-12));
    }

    #[test]
    fn pav_output_is_monotone_on_noisy_input() {
        // Noisy sawtooth in, monotone out — the defining invariant.
        let pts: Vec<CalPoint> = (0..50)
            .map(|i| {
                let x = i as f64 / 50.0;
                let noise = if i % 3 == 0 { -0.3 } else { 0.2 };
                CalPoint {
                    x,
                    y: (x + noise).clamp(0.0, 1.0),
                    weight: 1.0,
                }
            })
            .collect();
        let map = fit_pav_isotonic(&pts);
        for pair in map.knots.windows(2) {
            assert!(pair[0].1 <= pair[1].1, "monotone: {:?}", map.knots);
        }
    }
}
