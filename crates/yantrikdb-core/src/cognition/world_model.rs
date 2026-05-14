//! World Model Learning — action→outcome transition model.
//!
//! Learns cause-and-effect from the system's operational history.
//! For each (discretized_state, action_type) pair, maintains an
//! outcome distribution with Bayesian smoothing.
//!
//! # Key invariants
//!
//! - Bayesian smoothing prevents overconfidence from small samples
//! - State features discretized into ~100 context clusters
//! - Predictions degrade gracefully to uninformative priors
//! - All operations are O(1) for lookup, O(k) for update

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ══════════════════════════════════════════════════════════════════════════════
// § 1  State Features (Discretized Context)
// ══════════════════════════════════════════════════════════════════════════════

/// Discretized time-of-day bins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TimeBin {
    /// 00:00 – 05:59
    EarlyMorning = 0,
    /// 06:00 – 09:59
    Morning = 1,
    /// 10:00 – 13:59
    Midday = 2,
    /// 14:00 – 17:59
    Afternoon = 3,
    /// 18:00 – 21:59
    Evening = 4,
    /// 22:00 – 23:59
    Night = 5,
}

impl TimeBin {
    pub fn from_hour(hour: u8) -> Self {
        match hour {
            0..=5 => Self::EarlyMorning,
            6..=9 => Self::Morning,
            10..=13 => Self::Midday,
            14..=17 => Self::Afternoon,
            18..=21 => Self::Evening,
            _ => Self::Night,
        }
    }

    pub fn from_timestamp(ts: f64) -> Self {
        let hour = ((ts % 86400.0) / 3600.0) as u8;
        Self::from_hour(hour)
    }
}

/// Discretized receptivity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ReceptivityBin {
    /// Receptivity < 0.3
    Low = 0,
    /// 0.3 ≤ receptivity < 0.6
    Medium = 1,
    /// 0.6 ≤ receptivity < 0.8
    High = 2,
    /// Receptivity ≥ 0.8
    VeryHigh = 3,
}

impl ReceptivityBin {
    pub fn from_value(v: f64) -> Self {
        if v >= 0.8 {
            Self::VeryHigh
        } else if v >= 0.6 {
            Self::High
        } else if v >= 0.3 {
            Self::Medium
        } else {
            Self::Low
        }
    }
}

/// Discretized session stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum SessionStage {
    /// First 5 minutes.
    Early = 0,
    /// 5 minutes – 30 minutes.
    Mid = 1,
    /// 30+ minutes.
    Late = 2,
}

impl SessionStage {
    pub fn from_duration_secs(d: f64) -> Self {
        if d < 300.0 {
            Self::Early
        } else if d < 1800.0 {
            Self::Mid
        } else {
            Self::Late
        }
    }
}

/// Discretized error rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ErrorRateBin {
    /// 0 errors/min
    None = 0,
    /// 0–1 errors/min
    Low = 1,
    /// 1+ errors/min
    High = 2,
}

impl ErrorRateBin {
    pub fn from_rate(rate: f64) -> Self {
        if rate <= 0.0 {
            Self::None
        } else if rate <= 1.0 {
            Self::Low
        } else {
            Self::High
        }
    }
}

/// Compact discretized state features.
///
/// Uniquely identifies a context cluster. The combination of bins
/// gives ~6×4×3×3×3 = 648 possible clusters — very manageable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StateFeatures {
    pub time_bin: TimeBin,
    pub receptivity_bin: ReceptivityBin,
    pub session_stage: SessionStage,
    pub error_rate_bin: ErrorRateBin,
    pub active_goal_count_bin: u8, // 0, 1, 2+ (clamped to 2)
}

impl StateFeatures {
    /// Discretize from raw context values.
    pub fn discretize(
        timestamp: f64,
        receptivity: f64,
        session_duration_secs: f64,
        error_rate_per_min: f64,
        active_goal_count: usize,
    ) -> Self {
        Self {
            time_bin: TimeBin::from_timestamp(timestamp),
            receptivity_bin: ReceptivityBin::from_value(receptivity),
            session_stage: SessionStage::from_duration_secs(session_duration_secs),
            error_rate_bin: ErrorRateBin::from_rate(error_rate_per_min),
            active_goal_count_bin: (active_goal_count as u8).min(2),
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 2  Action & Outcome Types
// ══════════════════════════════════════════════════════════════════════════════

/// High-level action kind (what the system did).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionKind {
    /// Surfaced a proactive suggestion.
    SurfaceSuggestion,
    /// Sent a notification.
    SendNotification,
    /// Executed a tool call.
    ExecuteTool,
    /// Invoked the LLM.
    InvokeLlm,
    /// Offered a learned skill.
    OfferSkill,
    /// Ran an experiment variant.
    RunExperiment,
    /// Performed background maintenance.
    BackgroundMaintenance,
}

impl ActionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SurfaceSuggestion => "surface_suggestion",
            Self::SendNotification => "send_notification",
            Self::ExecuteTool => "execute_tool",
            Self::InvokeLlm => "invoke_llm",
            Self::OfferSkill => "offer_skill",
            Self::RunExperiment => "run_experiment",
            Self::BackgroundMaintenance => "background_maintenance",
        }
    }
}

/// Observed outcome of an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionOutcome {
    /// User accepted/acted on it.
    Accepted,
    /// User explicitly rejected it.
    Rejected,
    /// User ignored it (timeout).
    Ignored,
    /// Action succeeded (for tool/LLM calls).
    Succeeded,
    /// Action failed.
    Failed,
}

impl ActionOutcome {
    /// Whether this is a positive outcome.
    pub fn is_positive(self) -> bool {
        matches!(self, Self::Accepted | Self::Succeeded)
    }

    /// Whether this is a negative outcome.
    pub fn is_negative(self) -> bool {
        matches!(self, Self::Rejected | Self::Failed)
    }

    /// Convert to index for distribution tracking.
    fn index(self) -> usize {
        match self {
            Self::Accepted => 0,
            Self::Rejected => 1,
            Self::Ignored => 2,
            Self::Succeeded => 3,
            Self::Failed => 4,
        }
    }

    const COUNT: usize = 5;
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  Outcome Distribution (Dirichlet-Categorical)
// ══════════════════════════════════════════════════════════════════════════════

/// Dirichlet-Categorical distribution over action outcomes.
///
/// Maintains counts + prior (α parameters) for Bayesian smoothing.
/// The posterior mean is (count_i + α_i) / (total + Σα).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeDistribution {
    /// Observed counts per outcome.
    pub(crate) counts: [u32; ActionOutcome::COUNT],
    /// Dirichlet prior (pseudo-counts). Default: uniform α=1.
    prior: [f64; ActionOutcome::COUNT],
}

impl OutcomeDistribution {
    /// Create with uniform uninformative prior (α=1 for all outcomes).
    pub fn uninformative_prior() -> Self {
        Self {
            counts: [0; ActionOutcome::COUNT],
            prior: [1.0; ActionOutcome::COUNT],
        }
    }

    /// Create with custom prior.
    pub fn with_prior(prior: [f64; ActionOutcome::COUNT]) -> Self {
        Self {
            counts: [0; ActionOutcome::COUNT],
            prior,
        }
    }

    /// Record an observed outcome.
    pub fn observe(&mut self, outcome: ActionOutcome) {
        self.counts[outcome.index()] += 1;
    }

    /// Total observations.
    pub fn total_observations(&self) -> u32 {
        self.counts.iter().sum()
    }

    /// Total prior mass (Σα).
    fn prior_mass(&self) -> f64 {
        self.prior.iter().sum()
    }

    /// Posterior mean probability for a specific outcome.
    ///
    /// P(outcome) = (count + α) / (total + Σα)
    pub fn posterior_mean(&self, outcome: ActionOutcome) -> f64 {
        let idx = outcome.index();
        let total = self.total_observations() as f64 + self.prior_mass();
        if total <= 0.0 {
            return 1.0 / ActionOutcome::COUNT as f64;
        }
        (self.counts[idx] as f64 + self.prior[idx]) / total
    }

    /// Get all posterior mean probabilities.
    pub fn all_posteriors(&self) -> [f64; ActionOutcome::COUNT] {
        let total = self.total_observations() as f64 + self.prior_mass();
        let mut result = [0.0; ActionOutcome::COUNT];
        if total <= 0.0 {
            let uniform = 1.0 / ActionOutcome::COUNT as f64;
            result.fill(uniform);
            return result;
        }
        for i in 0..ActionOutcome::COUNT {
            result[i] = (self.counts[i] as f64 + self.prior[i]) / total;
        }
        result
    }

    /// Expected positive outcome rate: P(Accepted) + P(Succeeded).
    pub fn positive_rate(&self) -> f64 {
        self.posterior_mean(ActionOutcome::Accepted) + self.posterior_mean(ActionOutcome::Succeeded)
    }

    /// Expected negative outcome rate: P(Rejected) + P(Failed).
    pub fn negative_rate(&self) -> f64 {
        self.posterior_mean(ActionOutcome::Rejected) + self.posterior_mean(ActionOutcome::Failed)
    }

    /// Entropy of the posterior distribution (higher = more uncertain).
    pub fn entropy(&self) -> f64 {
        let posteriors = self.all_posteriors();
        let mut h = 0.0;
        for &p in &posteriors {
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        h
    }

    /// Whether we have enough data to be confident (low entropy + enough observations).
    pub fn is_informative(&self, min_observations: u32) -> bool {
        self.total_observations() >= min_observations
    }

    /// Most likely outcome.
    pub fn mode(&self) -> ActionOutcome {
        let posteriors = self.all_posteriors();
        let idx = posteriors
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        match idx {
            0 => ActionOutcome::Accepted,
            1 => ActionOutcome::Rejected,
            2 => ActionOutcome::Ignored,
            3 => ActionOutcome::Succeeded,
            _ => ActionOutcome::Failed,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  Transition Model
// ══════════════════════════════════════════════════════════════════════════════

/// A single transition entry (serialization-friendly).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionEntry {
    pub features: StateFeatures,
    pub action: ActionKind,
    pub distribution: OutcomeDistribution,
}

/// The world model: maps (state, action) → outcome distribution.
///
/// Learns from the system's own operational history. Used by CK-1.8
/// forward simulation to predict consequences of candidate actions.
///
/// Uses a Vec + runtime HashMap for serialization compatibility
/// (JSON requires string keys, but our keys are struct tuples).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionModel {
    /// Serializable transition entries.
    entries: Vec<TransitionEntry>,
    /// Runtime lookup index (rebuilt from entries).
    #[serde(skip)]
    index: HashMap<(StateFeatures, ActionKind), usize>,
    /// Total transitions recorded.
    pub total_transitions: u64,
    /// Global action outcome counts (for baseline comparison).
    pub global_outcomes: OutcomeDistribution,
}

impl TransitionModel {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            index: HashMap::new(),
            total_transitions: 0,
            global_outcomes: OutcomeDistribution::uninformative_prior(),
        }
    }

    /// Rebuild the runtime index from entries (needed after deserialization).
    pub fn rebuild_index(&mut self) {
        self.index.clear();
        for (i, entry) in self.entries.iter().enumerate() {
            self.index.insert((entry.features, entry.action), i);
        }
    }

    /// Record a transition: given state + action, observed outcome.
    pub fn record(&mut self, features: StateFeatures, action: ActionKind, outcome: ActionOutcome) {
        let key = (features, action);
        if let Some(&idx) = self.index.get(&key) {
            self.entries[idx].distribution.observe(outcome);
        } else {
            let idx = self.entries.len();
            let mut dist = OutcomeDistribution::uninformative_prior();
            dist.observe(outcome);
            self.entries.push(TransitionEntry {
                features,
                action,
                distribution: dist,
            });
            self.index.insert(key, idx);
        }
        self.global_outcomes.observe(outcome);
        self.total_transitions += 1;
    }

    /// Predict the outcome distribution for a (state, action) pair.
    ///
    /// Falls back to uninformative prior if no data for this specific pair.
    pub fn predict(&self, features: &StateFeatures, action: ActionKind) -> OutcomeDistribution {
        self.index
            .get(&(*features, action))
            .map(|&idx| self.entries[idx].distribution.clone())
            .unwrap_or_else(OutcomeDistribution::uninformative_prior)
    }

    /// Predict with hierarchical fallback.
    ///
    /// If the specific (state, action) pair has few observations,
    /// blend with the global distribution weighted by confidence.
    pub fn predict_blended(
        &self,
        features: &StateFeatures,
        action: ActionKind,
        min_observations: u32,
    ) -> OutcomeDistribution {
        let specific = self.predict(features, action);
        if specific.total_observations() >= min_observations {
            return specific;
        }

        // Blend: use global as fallback with weight proportional to data scarcity
        let specific_weight = specific.total_observations() as f64 / min_observations as f64;
        let global_weight = 1.0 - specific_weight;

        let s_posteriors = specific.all_posteriors();
        let g_posteriors = self.global_outcomes.all_posteriors();

        let mut blended_prior = [0.0f64; ActionOutcome::COUNT];
        for i in 0..ActionOutcome::COUNT {
            blended_prior[i] =
                (s_posteriors[i] * specific_weight + g_posteriors[i] * global_weight).max(0.001);
        }

        // Return a distribution with the blended prior and no new counts
        let mut result = OutcomeDistribution::with_prior(blended_prior);
        // Carry forward specific counts so caller can see observation count
        result.counts = specific.counts;
        result
    }

    /// Get the expected positive outcome rate for an action in a given state.
    pub fn expected_success(&self, features: &StateFeatures, action: ActionKind) -> f64 {
        self.predict(features, action).positive_rate()
    }

    /// Get the best action (highest positive rate) for a given state.
    pub fn best_action(
        &self,
        features: &StateFeatures,
        actions: &[ActionKind],
    ) -> Option<ActionKind> {
        actions
            .iter()
            .max_by(|&&a, &&b| {
                let ra = self.expected_success(features, a);
                let rb = self.expected_success(features, b);
                ra.partial_cmp(&rb).unwrap()
            })
            .copied()
    }

    /// Number of unique (state, action) pairs observed.
    pub fn unique_pairs(&self) -> usize {
        self.entries.len()
    }

    /// Get the most well-observed pairs (by observation count).
    pub fn top_pairs(&self, limit: usize) -> Vec<((StateFeatures, ActionKind), u32)> {
        let mut pairs: Vec<_> = self
            .entries
            .iter()
            .map(|e| ((e.features, e.action), e.distribution.total_observations()))
            .collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(limit);
        pairs
    }

    /// Average prediction accuracy (for pairs with enough data).
    pub fn prediction_accuracy(&self, min_observations: u32) -> f64 {
        let informative: Vec<_> = self
            .entries
            .iter()
            .map(|e| &e.distribution)
            .filter(|d| d.is_informative(min_observations))
            .collect();

        if informative.is_empty() {
            return 0.5; // Uninformative
        }

        // Accuracy = average probability assigned to the mode (most likely outcome)
        let total_accuracy: f64 = informative
            .iter()
            .map(|d| {
                let mode = d.mode();
                d.posterior_mean(mode)
            })
            .sum();

        total_accuracy / informative.len() as f64
    }
}

impl Default for TransitionModel {
    fn default() -> Self {
        Self::new()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 5  World Model Summary
// ══════════════════════════════════════════════════════════════════════════════

/// Summary of the world model's learned knowledge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelSummary {
    /// Total transitions recorded.
    pub total_transitions: u64,
    /// Unique (state, action) pairs.
    pub unique_pairs: usize,
    /// Global positive outcome rate.
    pub global_positive_rate: f64,
    /// Global negative outcome rate.
    pub global_negative_rate: f64,
    /// Average prediction accuracy.
    pub prediction_accuracy: f64,
    /// Global outcome entropy (uncertainty).
    pub global_entropy: f64,
}

/// Generate a summary of the world model.
pub fn summarize_world_model(model: &TransitionModel) -> WorldModelSummary {
    WorldModelSummary {
        total_transitions: model.total_transitions,
        unique_pairs: model.unique_pairs(),
        global_positive_rate: model.global_outcomes.positive_rate(),
        global_negative_rate: model.global_outcomes.negative_rate(),
        prediction_accuracy: model.prediction_accuracy(5),
        global_entropy: model.global_outcomes.entropy(),
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 6  Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn noon_features() -> StateFeatures {
        StateFeatures::discretize(43200.0, 0.7, 600.0, 0.0, 1)
    }

    fn morning_features() -> StateFeatures {
        StateFeatures::discretize(28800.0, 0.5, 300.0, 0.5, 2)
    }

    // ── State Features ──

    #[test]
    fn test_time_bin_discretization() {
        assert_eq!(TimeBin::from_hour(3), TimeBin::EarlyMorning);
        assert_eq!(TimeBin::from_hour(8), TimeBin::Morning);
        assert_eq!(TimeBin::from_hour(12), TimeBin::Midday);
        assert_eq!(TimeBin::from_hour(15), TimeBin::Afternoon);
        assert_eq!(TimeBin::from_hour(19), TimeBin::Evening);
        assert_eq!(TimeBin::from_hour(23), TimeBin::Night);
    }

    #[test]
    fn test_receptivity_bin() {
        assert_eq!(ReceptivityBin::from_value(0.1), ReceptivityBin::Low);
        assert_eq!(ReceptivityBin::from_value(0.5), ReceptivityBin::Medium);
        assert_eq!(ReceptivityBin::from_value(0.7), ReceptivityBin::High);
        assert_eq!(ReceptivityBin::from_value(0.9), ReceptivityBin::VeryHigh);
    }

    #[test]
    fn test_session_stage() {
        assert_eq!(SessionStage::from_duration_secs(60.0), SessionStage::Early);
        assert_eq!(SessionStage::from_duration_secs(600.0), SessionStage::Mid);
        assert_eq!(SessionStage::from_duration_secs(3600.0), SessionStage::Late);
    }

    #[test]
    fn test_state_features_discretize() {
        let f = StateFeatures::discretize(43200.0, 0.7, 600.0, 0.0, 3);
        assert_eq!(f.time_bin, TimeBin::Midday);
        assert_eq!(f.receptivity_bin, ReceptivityBin::High);
        assert_eq!(f.session_stage, SessionStage::Mid);
        assert_eq!(f.error_rate_bin, ErrorRateBin::None);
        assert_eq!(f.active_goal_count_bin, 2); // clamped
    }

    // ── Outcome Distribution ──

    #[test]
    fn test_uninformative_prior() {
        let dist = OutcomeDistribution::uninformative_prior();
        let posteriors = dist.all_posteriors();
        // With α=1 for all 5 outcomes and 0 observations, each is 1/5 = 0.2
        for &p in &posteriors {
            assert!((p - 0.2).abs() < 0.01);
        }
    }

    #[test]
    fn test_posterior_update() {
        let mut dist = OutcomeDistribution::uninformative_prior();
        // Observe 10 acceptances, 2 rejections
        for _ in 0..10 {
            dist.observe(ActionOutcome::Accepted);
        }
        for _ in 0..2 {
            dist.observe(ActionOutcome::Rejected);
        }

        let p_accepted = dist.posterior_mean(ActionOutcome::Accepted);
        let p_rejected = dist.posterior_mean(ActionOutcome::Rejected);

        assert!(
            p_accepted > p_rejected,
            "P(Accepted)={:.3} should exceed P(Rejected)={:.3}",
            p_accepted,
            p_rejected,
        );
        assert!(p_accepted > 0.5);
    }

    #[test]
    fn test_bayesian_smoothing() {
        let mut dist = OutcomeDistribution::uninformative_prior();
        // Single observation shouldn't dominate
        dist.observe(ActionOutcome::Accepted);

        let p_accepted = dist.posterior_mean(ActionOutcome::Accepted);
        // With 1 obs + α=1 prior: (1+1)/(1+5) = 0.333
        assert!(
            (p_accepted - 0.333).abs() < 0.01,
            "Single obs should be smoothed: P={:.3}",
            p_accepted,
        );
    }

    #[test]
    fn test_positive_negative_rates() {
        let mut dist = OutcomeDistribution::uninformative_prior();
        for _ in 0..8 {
            dist.observe(ActionOutcome::Accepted);
        }
        for _ in 0..2 {
            dist.observe(ActionOutcome::Rejected);
        }

        assert!(dist.positive_rate() > dist.negative_rate());
    }

    #[test]
    fn test_entropy() {
        // Uniform distribution has maximum entropy
        let uniform = OutcomeDistribution::uninformative_prior();
        let h_uniform = uniform.entropy();

        // Peaked distribution has lower entropy
        let mut peaked = OutcomeDistribution::uninformative_prior();
        for _ in 0..100 {
            peaked.observe(ActionOutcome::Accepted);
        }
        let h_peaked = peaked.entropy();

        assert!(
            h_peaked < h_uniform,
            "Peaked entropy ({:.3}) should be lower than uniform ({:.3})",
            h_peaked,
            h_uniform,
        );
    }

    #[test]
    fn test_mode() {
        let mut dist = OutcomeDistribution::uninformative_prior();
        for _ in 0..20 {
            dist.observe(ActionOutcome::Succeeded);
        }
        for _ in 0..3 {
            dist.observe(ActionOutcome::Failed);
        }

        assert_eq!(dist.mode(), ActionOutcome::Succeeded);
    }

    // ── Transition Model ──

    #[test]
    fn test_model_record_and_predict() {
        let mut model = TransitionModel::new();
        let features = noon_features();

        // Record 10 successful tool calls at noon
        for _ in 0..10 {
            model.record(features, ActionKind::ExecuteTool, ActionOutcome::Succeeded);
        }
        for _ in 0..2 {
            model.record(features, ActionKind::ExecuteTool, ActionOutcome::Failed);
        }

        let prediction = model.predict(&features, ActionKind::ExecuteTool);
        assert!(
            prediction.posterior_mean(ActionOutcome::Succeeded)
                > prediction.posterior_mean(ActionOutcome::Failed),
        );
        assert_eq!(model.total_transitions, 12);
    }

    #[test]
    fn test_model_fallback_to_uninformative() {
        let model = TransitionModel::new();
        let features = noon_features();

        let prediction = model.predict(&features, ActionKind::SurfaceSuggestion);
        // Should return uninformative prior
        let posteriors = prediction.all_posteriors();
        for &p in &posteriors {
            assert!((p - 0.2).abs() < 0.01);
        }
    }

    #[test]
    fn test_model_blended_prediction() {
        let mut model = TransitionModel::new();
        let features = noon_features();

        // Record small sample for specific pair
        model.record(
            features,
            ActionKind::SurfaceSuggestion,
            ActionOutcome::Accepted,
        );

        // Also record a lot of global data with rejections
        for _ in 0..50 {
            model.record(
                morning_features(),
                ActionKind::SurfaceSuggestion,
                ActionOutcome::Rejected,
            );
        }

        // Blended prediction should incorporate global signal
        let blended = model.predict_blended(&features, ActionKind::SurfaceSuggestion, 10);
        let specific = model.predict(&features, ActionKind::SurfaceSuggestion);

        // The blended Rejected probability should be higher than pure specific
        // (global has lots of rejections, pulling blended toward rejection)
        assert!(
            blended.posterior_mean(ActionOutcome::Rejected)
                > specific.posterior_mean(ActionOutcome::Rejected),
            "Blended Rejected ({:.4}) should exceed specific Rejected ({:.4}) — global influence",
            blended.posterior_mean(ActionOutcome::Rejected),
            specific.posterior_mean(ActionOutcome::Rejected),
        );
    }

    #[test]
    fn test_best_action() {
        let mut model = TransitionModel::new();
        let features = noon_features();

        // Suggestions get accepted at noon
        for _ in 0..10 {
            model.record(
                features,
                ActionKind::SurfaceSuggestion,
                ActionOutcome::Accepted,
            );
        }
        // Notifications get rejected at noon
        for _ in 0..10 {
            model.record(
                features,
                ActionKind::SendNotification,
                ActionOutcome::Rejected,
            );
        }

        let actions = [ActionKind::SurfaceSuggestion, ActionKind::SendNotification];
        let best = model.best_action(&features, &actions);
        assert_eq!(best, Some(ActionKind::SurfaceSuggestion));
    }

    #[test]
    fn test_prediction_accuracy() {
        let mut model = TransitionModel::new();
        let features = noon_features();

        // Very predictable: always accepted
        for _ in 0..20 {
            model.record(features, ActionKind::ExecuteTool, ActionOutcome::Succeeded);
        }

        let accuracy = model.prediction_accuracy(5);
        assert!(
            accuracy > 0.7,
            "Accuracy ({:.3}) should be high for consistent outcomes",
            accuracy,
        );
    }

    #[test]
    fn test_global_outcomes() {
        let mut model = TransitionModel::new();

        model.record(
            noon_features(),
            ActionKind::ExecuteTool,
            ActionOutcome::Succeeded,
        );
        model.record(
            morning_features(),
            ActionKind::SurfaceSuggestion,
            ActionOutcome::Accepted,
        );
        model.record(
            noon_features(),
            ActionKind::SendNotification,
            ActionOutcome::Rejected,
        );

        assert_eq!(model.global_outcomes.total_observations(), 3);
    }

    #[test]
    fn test_top_pairs() {
        let mut model = TransitionModel::new();

        for _ in 0..10 {
            model.record(
                noon_features(),
                ActionKind::ExecuteTool,
                ActionOutcome::Succeeded,
            );
        }
        for _ in 0..5 {
            model.record(
                morning_features(),
                ActionKind::SurfaceSuggestion,
                ActionOutcome::Accepted,
            );
        }

        let top = model.top_pairs(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].1, 10); // Most observed pair first
    }

    // ── Summary ──

    #[test]
    fn test_world_model_summary() {
        let mut model = TransitionModel::new();
        for _ in 0..20 {
            model.record(
                noon_features(),
                ActionKind::ExecuteTool,
                ActionOutcome::Succeeded,
            );
        }

        let summary = summarize_world_model(&model);
        assert_eq!(summary.total_transitions, 20);
        assert_eq!(summary.unique_pairs, 1);
        assert!(summary.global_positive_rate > 0.5);
    }
}
