//! Runtime-tunable retrieval parameters.
//!
//! # Why these are not constants
//!
//! Every scoring bound in this engine was a compiled-in constant, which had
//! two costs that only became obvious once we tried to measure them.
//!
//! The cheap cost: testing one value meant a full rebuild (~90s) plus a wheel
//! install, so a five-point sweep was a ten-minute round trip and nobody ever
//! ran one. The constants were therefore never swept — they were guessed once
//! and inherited forever.
//!
//! The expensive cost: because no sweep existed, a *systematically wrong*
//! calibration survived indefinitely. Measured 2026-08-13 on 5,035 real
//! memories: the bounds are RAW-COSINE RATIOS, but this embedding space is
//! compressed (mean random-pair cosine 0.424 for potion-8M), so the top 100
//! results spanned only 1.286x. A budget documented as "reorders near-equals,
//! reverses at most a 30% relevance gap" could in practice move a record
//! across 120-252 ranks. Every constant was enforced exactly as written and
//! meant something entirely different from what its comment claimed.
//!
//! A constant you cannot sweep is a constant you cannot validate. These are
//! now runtime values with the previous constants as defaults, so a sweep is
//! an env var rather than a build.
//!
//! # Usage
//!
//! ```text
//! YANTRIKDB_POLICY_BUDGET=1.05 YANTRIKDB_GATE_TAU=0.45 <run the eval>
//! ```
//!
//! Read once per process on first use. That is deliberate: retrieval must be
//! deterministic within a run, so a mid-run change cannot silently split a
//! measurement into two configurations. Tests that need a specific
//! configuration should construct [`Tuning`] directly rather than mutating
//! the environment.

use std::sync::OnceLock;

/// All runtime-tunable retrieval parameters, with the historical constants as
/// defaults so behaviour is unchanged unless something is explicitly set.
#[derive(Debug, Clone, PartialEq)]
pub struct Tuning {
    /// Total ratio the policy layer may apply, as a MULTIPLIER (not a log).
    /// `1.30` was the shipped default; the measurement above suggests it is
    /// far too generous for compressed spaces, which is exactly the sort of
    /// claim this knob exists to test.
    pub policy_budget: f64,

    /// Prior weights. Should sum to <= 1.0; the engine renormalizes if they
    /// do not, so a careless sweep degrades gracefully instead of silently
    /// widening the ceiling.
    pub pw_freshness: f64,
    pub pw_importance: f64,
    pub pw_graph: f64,
    pub pw_agreement: f64,
    pub pw_usage: f64,

    /// Importance-gate midpoint, in raw cosine. Measured to be badly
    /// miscalibrated: at `0.25` the gate is already 0.89 open at the
    /// RANDOM-PAIR similarity of this corpus (0.424), so importance is
    /// effectively ungated everywhere it matters.
    pub gate_tau: f64,
    /// Gate sharpness.
    pub gate_k: f64,

    /// Ceiling for an exploration lane's multiplicative lift.
    pub lane_lift_max: f64,

    /// Lane admission floors, all in raw cosine and all suspect for the same
    /// reason as `gate_tau`.
    pub fts_min_sim: f64,
    pub cold_min_sim: f64,
    pub valence_min_sim: f64,

    /// MMR relevance/diversity trade-off.
    pub mmr_lambda: f64,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            policy_budget: 1.30,
            pw_freshness: 0.22,
            pw_importance: 0.40,
            pw_graph: 0.13,
            pw_agreement: 0.13,
            pw_usage: 0.12,
            gate_tau: 0.25,
            gate_k: 12.0,
            lane_lift_max: 0.10,
            fts_min_sim: 0.05,
            cold_min_sim: 0.10,
            valence_min_sim: 0.02,
            mmr_lambda: 0.9,
        }
    }
}

fn env_f64(key: &str, fallback: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(fallback)
}

impl Tuning {
    /// Read from `YANTRIKDB_*` environment variables, falling back to the
    /// shipped defaults. Non-numeric or non-finite values are ignored rather
    /// than fatal: a typo in a sweep script must not take retrieval down.
    pub fn from_env() -> Self {
        let d = Self::default();
        Self {
            policy_budget: env_f64("YANTRIKDB_POLICY_BUDGET", d.policy_budget).max(1.0),
            pw_freshness: env_f64("YANTRIKDB_PW_FRESHNESS", d.pw_freshness).max(0.0),
            pw_importance: env_f64("YANTRIKDB_PW_IMPORTANCE", d.pw_importance).max(0.0),
            pw_graph: env_f64("YANTRIKDB_PW_GRAPH", d.pw_graph).max(0.0),
            pw_agreement: env_f64("YANTRIKDB_PW_AGREEMENT", d.pw_agreement).max(0.0),
            pw_usage: env_f64("YANTRIKDB_PW_USAGE", d.pw_usage).max(0.0),
            gate_tau: env_f64("YANTRIKDB_GATE_TAU", d.gate_tau).clamp(0.0, 1.0),
            gate_k: env_f64("YANTRIKDB_GATE_K", d.gate_k).max(0.0),
            lane_lift_max: env_f64("YANTRIKDB_LANE_LIFT_MAX", d.lane_lift_max).max(0.0),
            fts_min_sim: env_f64("YANTRIKDB_FTS_MIN_SIM", d.fts_min_sim),
            cold_min_sim: env_f64("YANTRIKDB_COLD_MIN_SIM", d.cold_min_sim),
            valence_min_sim: env_f64("YANTRIKDB_VALENCE_MIN_SIM", d.valence_min_sim),
            mmr_lambda: env_f64("YANTRIKDB_MMR_LAMBDA", d.mmr_lambda).clamp(0.0, 1.0),
        }
    }

    /// `ln(policy_budget)` — the exponent scale the policy layer applies.
    #[inline]
    pub fn policy_budget_ln(&self) -> f64 {
        self.policy_budget.max(1.0).ln()
    }

    /// Prior weights, renormalized to sum to at most 1.0.
    ///
    /// The shared-budget guarantee depends on this sum, so it is enforced
    /// here rather than trusted from the environment. A sweep that sets five
    /// weights to 1.0 each gets them scaled down, not a 5x ceiling.
    pub fn normalized_weights(&self) -> (f64, f64, f64, f64, f64) {
        let sum = self.pw_freshness
            + self.pw_importance
            + self.pw_graph
            + self.pw_agreement
            + self.pw_usage;
        let k = if sum > 1.0 { 1.0 / sum } else { 1.0 };
        (
            self.pw_freshness * k,
            self.pw_importance * k,
            self.pw_graph * k,
            self.pw_agreement * k,
            self.pw_usage * k,
        )
    }

    /// A stable one-line description, for stamping into run metadata so a
    /// result can never be separated from the configuration that produced it.
    pub fn fingerprint(&self) -> String {
        let (f, i, g, a, u) = self.normalized_weights();
        format!(
            "budget={:.3} w=[f{:.3},i{:.3},g{:.3},a{:.3},u{:.3}] gate=({:.3},{:.1}) \
             lane={:.3} floors=[fts{:.3},cold{:.3},val{:.3}] mmr={:.2}",
            self.policy_budget,
            f,
            i,
            g,
            a,
            u,
            self.gate_tau,
            self.gate_k,
            self.lane_lift_max,
            self.fts_min_sim,
            self.cold_min_sim,
            self.valence_min_sim,
            self.mmr_lambda,
        )
    }
}

static TUNING: OnceLock<Tuning> = OnceLock::new();

/// The process-wide tuning, read from the environment on first use.
#[inline]
pub fn tuning() -> &'static Tuning {
    TUNING.get_or_init(Tuning::from_env)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_shipped_constants() {
        let d = Tuning::default();
        assert_eq!(d.policy_budget, 1.30);
        assert_eq!(d.gate_tau, 0.25);
        assert_eq!(d.mmr_lambda, 0.9);
        // The partition invariant, now checked on the DEFAULTS rather than on
        // hand-listed constants that a future signal could forget to include.
        let (f, i, g, a, u) = d.normalized_weights();
        assert!(f + i + g + a + u <= 1.0 + 1e-12);
    }

    #[test]
    fn careless_weights_are_renormalized_not_obeyed() {
        // A sweep script setting every weight to 1.0 must not buy a 5x
        // ceiling — the shared-budget property is enforced by the engine,
        // not by the operator getting it right.
        let t = Tuning {
            pw_freshness: 1.0,
            pw_importance: 1.0,
            pw_graph: 1.0,
            pw_agreement: 1.0,
            pw_usage: 1.0,
            ..Tuning::default()
        };
        let (f, i, g, a, u) = t.normalized_weights();
        let sum = f + i + g + a + u;
        assert!((sum - 1.0).abs() < 1e-12, "weights must renormalize, got {sum}");
    }

    #[test]
    fn a_budget_below_one_cannot_invert_the_multiplier() {
        // policy_budget < 1.0 would make exp(ln(x)) shrink scores and turn
        // every prior into a PENALTY — a plausible typo in a sweep.
        let t = Tuning { policy_budget: 0.5, ..Tuning::default() };
        assert!(t.policy_budget_ln() >= 0.0);
    }

    #[test]
    fn fingerprint_is_stable_and_descriptive() {
        let s = Tuning::default().fingerprint();
        assert!(s.contains("budget=1.300"));
        assert!(s.contains("gate=(0.250,12.0)"));
    }
}
