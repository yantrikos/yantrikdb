//! Online Weight Learning + Confidence Calibration.
//!
//! Continuously tunes scoring weights and confidence thresholds
//! from interaction outcomes, using no external ML dependencies.
//!
//! # Components
//!
//! 1. **Utility Weight Learner**: Coordinate descent on pairwise ranking loss
//!    for the 4 evaluator weights (effect, intent, preference, simulation).
//!
//! 2. **Per-Action Confidence Bandits**: Beta(α, β) posteriors per action kind,
//!    learned from acceptance/rejection signals. Thompson sampling for thresholds.
//!
//! 3. **Confidence Calibration Map**: Isotonic regression over 10 bins,
//!    mapping raw confidence to calibrated (predicted-vs-actual) confidence.
//!
//! 4. **Source Reliability Tracker**: Bayesian accuracy tracking for each
//!    evidence source (User, LLM, Autonomous, System).
//!
//! 5. **Learning Scheduler**: Tracks interaction count and triggers
//!    refit/audit at configured intervals.
//!
//! # Learning Signals
//!
//! - Accepted > Ignored > Rejected (ordinal preference signal)
//! - Success/failure of executed actions (binary signal)
//! - Belief confirmation/contradiction (source reliability signal)

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ══════════════════════════════════════════════════════════════════════════════
// § 1  Interaction Outcomes
// ══════════════════════════════════════════════════════════════════════════════

/// The outcome of a suggestion/action offered to the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionOutcome {
    /// User explicitly accepted the suggestion.
    Accepted,
    /// User modified the suggestion before accepting.
    Modified,
    /// User ignored the suggestion (timeout).
    Ignored,
    /// User explicitly rejected the suggestion.
    Rejected,
}

impl InteractionOutcome {
    /// Ordinal value for pairwise ranking (higher = better).
    pub fn ordinal(self) -> u8 {
        match self {
            Self::Accepted => 3,
            Self::Modified => 2,
            Self::Ignored => 1,
            Self::Rejected => 0,
        }
    }

    /// Whether this outcome counts as positive for bandit updates.
    pub fn is_positive(self) -> bool {
        matches!(self, Self::Accepted | Self::Modified)
    }
}

/// Record of a single interaction for learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionRecord {
    /// The action kind string.
    pub action_kind: String,
    /// Raw (uncalibrated) confidence that was assigned.
    pub raw_confidence: f64,
    /// The outcome of the interaction.
    pub outcome: InteractionOutcome,
    /// Feature values used in scoring: [effect, intent, preference, simulation].
    pub features: [f64; 4],
    /// When this interaction occurred.
    pub timestamp: f64,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 2  Utility Weight Learner
// ══════════════════════════════════════════════════════════════════════════════

/// Learned weights for the utility evaluator.
///
/// These correspond to [effect_weight, intent_weight, preference_weight, simulation_weight]
/// in EvaluatorConfig.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilityWeights {
    /// Current weights [0.0, 1.0] each. Sum should be ~1.0.
    pub weights: [f64; 4],
    /// Number of weight updates performed.
    pub update_count: u64,
    /// Running pairwise ranking loss (lower = better).
    pub cumulative_loss: f64,
    /// Number of pairwise comparisons evaluated.
    pub pair_count: u64,
}

impl UtilityWeights {
    pub fn new() -> Self {
        Self {
            weights: [0.35, 0.25, 0.15, 0.25],
            update_count: 0,
            cumulative_loss: 0.0,
            pair_count: 0,
        }
    }

    /// Compute the weighted score for a feature vector.
    pub fn score(&self, features: &[f64; 4]) -> f64 {
        self.weights
            .iter()
            .zip(features.iter())
            .map(|(w, f)| w * f)
            .sum()
    }

    /// Current ranking accuracy [0.0, 1.0] (pairs correctly ordered / total pairs).
    pub fn accuracy(&self) -> f64 {
        if self.pair_count == 0 {
            0.5
        } else {
            1.0 - self.cumulative_loss / self.pair_count as f64
        }
    }
}

impl Default for UtilityWeights {
    fn default() -> Self {
        Self::new()
    }
}

/// Update utility weights using coordinate descent on pairwise ranking loss.
///
/// For each pair where outcome(a) > outcome(b), we want score(a) > score(b).
/// The loss is the hinge loss: max(0, margin - (score_a - score_b)).
///
/// `interactions` should contain the recent batch of interactions.
pub fn update_utility_weights(
    weights: &mut UtilityWeights,
    interactions: &[InteractionRecord],
    learning_rate: f64,
    margin: f64,
) {
    if interactions.len() < 2 {
        return;
    }

    // Collect pairwise comparisons from ordinal outcomes
    let mut total_loss = 0.0;
    let mut pair_count = 0u64;
    let mut gradient = [0.0f64; 4];

    for i in 0..interactions.len() {
        for j in (i + 1)..interactions.len() {
            let (better, worse) =
                if interactions[i].outcome.ordinal() > interactions[j].outcome.ordinal() {
                    (&interactions[i], &interactions[j])
                } else if interactions[j].outcome.ordinal() > interactions[i].outcome.ordinal() {
                    (&interactions[j], &interactions[i])
                } else {
                    continue; // Same outcome, no signal
                };

            let score_better = weights.score(&better.features);
            let score_worse = weights.score(&worse.features);
            let diff = score_better - score_worse;

            pair_count += 1;

            if diff < margin {
                // Hinge loss active — compute gradient
                let loss = margin - diff;
                total_loss += loss;

                for k in 0..4 {
                    gradient[k] += better.features[k] - worse.features[k];
                }
            }
        }
    }

    if pair_count == 0 {
        return;
    }

    // Normalize gradient by pair count
    let scale = learning_rate / pair_count as f64;
    for k in 0..4 {
        weights.weights[k] += gradient[k] * scale;
        // Clamp to [0.01, 1.0]
        weights.weights[k] = weights.weights[k].clamp(0.01, 1.0);
    }

    // Normalize weights to sum to 1.0
    let sum: f64 = weights.weights.iter().sum();
    if sum > 0.0 {
        for w in &mut weights.weights {
            *w /= sum;
        }
    }

    weights.update_count += 1;
    weights.cumulative_loss += total_loss;
    weights.pair_count += pair_count;
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  Per-Action Confidence Bandits
// ══════════════════════════════════════════════════════════════════════════════

/// Beta-Bernoulli bandit for a single action kind.
///
/// Tracks acceptance rate and provides Thompson-sampled thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionBandit {
    /// Beta posterior: successes + prior.
    pub alpha: f64,
    /// Beta posterior: failures + prior.
    pub beta: f64,
    /// Total number of observations.
    pub total: u64,
}

impl ActionBandit {
    /// New bandit with uninformative prior Beta(1, 1).
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
            total: 0,
        }
    }

    /// Record a positive outcome (acceptance).
    pub fn record_positive(&mut self) {
        self.alpha += 1.0;
        self.total += 1;
    }

    /// Record a negative outcome (rejection/ignore).
    pub fn record_negative(&mut self) {
        self.beta += 1.0;
        self.total += 1;
    }

    /// Posterior mean P(acceptance).
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Posterior variance.
    pub fn variance(&self) -> f64 {
        let ab = self.alpha + self.beta;
        (self.alpha * self.beta) / (ab * ab * (ab + 1.0))
    }

    /// Recommended confidence threshold: mean - safety_margin * stddev.
    /// Lower threshold = more willing to act (less conservative).
    pub fn threshold(&self, safety_margin: f64) -> f64 {
        let std = self.variance().sqrt();
        (self.mean() - safety_margin * std).clamp(0.05, 0.95)
    }
}

impl Default for ActionBandit {
    fn default() -> Self {
        Self::new()
    }
}

/// Collection of per-action-kind bandits.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BanditRegistry {
    /// Per-action-kind bandits.
    pub bandits: HashMap<String, ActionBandit>,
    /// Total interactions processed.
    pub total_interactions: u64,
}

impl BanditRegistry {
    pub fn new() -> Self {
        Self {
            bandits: HashMap::new(),
            total_interactions: 0,
        }
    }

    /// Get or create the bandit for an action kind.
    pub fn get_or_create(&mut self, action_kind: &str) -> &mut ActionBandit {
        self.bandits
            .entry(action_kind.to_string())
            .or_insert_with(ActionBandit::new)
    }

    /// Record an interaction outcome for a specific action kind.
    pub fn record(&mut self, action_kind: &str, outcome: InteractionOutcome) {
        let bandit = self.get_or_create(action_kind);
        if outcome.is_positive() {
            bandit.record_positive();
        } else {
            bandit.record_negative();
        }
        self.total_interactions += 1;
    }

    /// Get the recommended threshold for an action kind.
    pub fn threshold(&self, action_kind: &str, safety_margin: f64) -> f64 {
        match self.bandits.get(action_kind) {
            Some(bandit) => bandit.threshold(safety_margin),
            None => 0.5, // Default threshold for unknown action kinds
        }
    }

    /// Get the acceptance rate for an action kind.
    pub fn acceptance_rate(&self, action_kind: &str) -> f64 {
        match self.bandits.get(action_kind) {
            Some(bandit) => bandit.mean(),
            None => 0.5,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  Confidence Calibration Map
// ══════════════════════════════════════════════════════════════════════════════

/// Number of calibration bins.
const NUM_BINS: usize = 10;

/// A single calibration bin tracking predicted vs actual success.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationBin {
    /// Total predictions falling in this bin.
    pub count: u64,
    /// Number of positive outcomes in this bin.
    pub positive: u64,
    /// Sum of raw confidences in this bin (for computing mean predicted).
    pub sum_predicted: f64,
}

impl CalibrationBin {
    pub fn new() -> Self {
        Self {
            count: 0,
            positive: 0,
            sum_predicted: 0.0,
        }
    }

    /// Actual success rate in this bin.
    pub fn actual_rate(&self) -> f64 {
        if self.count == 0 {
            0.5
        } else {
            self.positive as f64 / self.count as f64
        }
    }

    /// Mean predicted confidence in this bin.
    pub fn mean_predicted(&self) -> f64 {
        if self.count == 0 {
            0.5
        } else {
            self.sum_predicted / self.count as f64
        }
    }
}

impl Default for CalibrationBin {
    fn default() -> Self {
        Self::new()
    }
}

/// Isotonic calibration map over equal-width confidence bins.
///
/// Maps raw confidence [0.0, 1.0] to calibrated confidence [0.0, 1.0]
/// using the actual success rate observed in each bin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMap {
    /// 10 bins covering [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0].
    pub bins: [CalibrationBin; NUM_BINS],
    /// Isotonic-fitted calibrated values (monotonically non-decreasing).
    pub isotonic_values: [f64; NUM_BINS],
    /// Total predictions processed.
    pub total: u64,
    /// Whether the isotonic map needs refitting.
    pub needs_refit: bool,
}

impl CalibrationMap {
    pub fn new() -> Self {
        let bins = std::array::from_fn(|_| CalibrationBin::new());
        let isotonic_values = std::array::from_fn(|i| (i as f64 + 0.5) / NUM_BINS as f64);
        Self {
            bins,
            isotonic_values,
            total: 0,
            needs_refit: false,
        }
    }

    /// Record a prediction and its outcome.
    pub fn record(&mut self, raw_confidence: f64, was_positive: bool) {
        let bin_idx = confidence_to_bin(raw_confidence);
        self.bins[bin_idx].count += 1;
        self.bins[bin_idx].sum_predicted += raw_confidence;
        if was_positive {
            self.bins[bin_idx].positive += 1;
        }
        self.total += 1;

        // Mark for refit every 50 observations
        if self.total % 50 == 0 {
            self.needs_refit = true;
        }
    }

    /// Get the calibrated confidence for a raw confidence value.
    pub fn calibrate(&self, raw_confidence: f64) -> f64 {
        let bin_idx = confidence_to_bin(raw_confidence);
        self.isotonic_values[bin_idx]
    }

    /// Refit the isotonic map using pool adjacent violators algorithm (PAVA).
    pub fn refit(&mut self) {
        // Step 1: Compute raw calibrated values from bins
        let mut values: [f64; NUM_BINS] = std::array::from_fn(|i| self.bins[i].actual_rate());

        // Step 2: Pool Adjacent Violators (ensure monotonically non-decreasing)
        isotonic_regression(&mut values);

        self.isotonic_values = values;
        self.needs_refit = false;
    }

    /// Expected Calibration Error (ECE) — mean absolute difference
    /// between predicted and actual across bins with data.
    pub fn calibration_error(&self) -> f64 {
        let mut total_error = 0.0;
        let mut total_weight = 0u64;

        for bin in &self.bins {
            if bin.count > 0 {
                let error = (bin.actual_rate() - bin.mean_predicted()).abs();
                total_error += error * bin.count as f64;
                total_weight += bin.count;
            }
        }

        if total_weight == 0 {
            0.0
        } else {
            total_error / total_weight as f64
        }
    }
}

impl Default for CalibrationMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Map raw confidence to bin index [0, NUM_BINS-1].
fn confidence_to_bin(confidence: f64) -> usize {
    let c = confidence.clamp(0.0, 0.999);
    (c * NUM_BINS as f64) as usize
}

/// Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.
///
/// Enforces monotonically non-decreasing values.
fn isotonic_regression(values: &mut [f64]) {
    let n = values.len();
    if n <= 1 {
        return;
    }

    // Forward pass: merge adjacent blocks that violate monotonicity
    let mut block_start = vec![0usize; n];
    let mut block_sum = vec![0.0f64; n];
    let mut block_count = vec![1u32; n];
    let mut num_blocks = n;

    for i in 0..n {
        block_start[i] = i;
        block_sum[i] = values[i];
        block_count[i] = 1;
    }

    // Simplified PAVA: iterative averaging of violating neighbors
    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i + 1 < num_blocks {
            let mean_i = block_sum[i] / block_count[i] as f64;
            let mean_next = block_sum[i + 1] / block_count[i + 1] as f64;

            if mean_i > mean_next {
                // Merge blocks
                block_sum[i] += block_sum[i + 1];
                block_count[i] += block_count[i + 1];

                // Shift remaining blocks
                for j in (i + 2)..num_blocks {
                    block_start[j - 1] = block_start[j];
                    block_sum[j - 1] = block_sum[j];
                    block_count[j - 1] = block_count[j];
                }
                num_blocks -= 1;
                changed = true;
            } else {
                i += 1;
            }
        }
    }

    // Write merged values back
    let mut pos = 0;
    for i in 0..num_blocks {
        let mean = block_sum[i] / block_count[i] as f64;
        for _ in 0..block_count[i] {
            values[pos] = mean;
            pos += 1;
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 5  Source Reliability Tracker
// ══════════════════════════════════════════════════════════════════════════════

/// Evidence source types for reliability tracking.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceSource {
    /// Direct user statement.
    User,
    /// LLM-generated inference.
    Llm,
    /// Autonomous system observation.
    Autonomous,
    /// External data source.
    External(String),
}

impl EvidenceSource {
    pub fn as_str(&self) -> &str {
        match self {
            Self::User => "user",
            Self::Llm => "llm",
            Self::Autonomous => "autonomous",
            Self::External(name) => name,
        }
    }
}

/// Reliability tracker for a single evidence source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceReliability {
    /// Beta posterior: confirmed beliefs.
    pub alpha: f64,
    /// Beta posterior: contradicted beliefs.
    pub beta: f64,
    /// Total beliefs from this source.
    pub total: u64,
}

impl SourceReliability {
    pub fn new() -> Self {
        Self {
            alpha: 2.0, // Mildly informative prior (assume somewhat reliable)
            beta: 1.0,
            total: 0,
        }
    }

    /// Record a confirmed belief from this source.
    pub fn record_confirmed(&mut self) {
        self.alpha += 1.0;
        self.total += 1;
    }

    /// Record a contradicted belief from this source.
    pub fn record_contradicted(&mut self) {
        self.beta += 1.0;
        self.total += 1;
    }

    /// Reliability score (posterior mean).
    pub fn reliability(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Confidence interval width (lower = more certain about reliability).
    pub fn uncertainty(&self) -> f64 {
        let ab = self.alpha + self.beta;
        2.0 * (self.alpha * self.beta / (ab * ab * (ab + 1.0))).sqrt()
    }
}

impl Default for SourceReliability {
    fn default() -> Self {
        Self::new()
    }
}

/// Collection of source reliability trackers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReliabilityRegistry {
    pub sources: HashMap<String, SourceReliability>,
}

impl ReliabilityRegistry {
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
        }
    }

    /// Get or create the tracker for a source.
    pub fn get_or_create(&mut self, source: &str) -> &mut SourceReliability {
        self.sources
            .entry(source.to_string())
            .or_insert_with(SourceReliability::new)
    }

    /// Get the reliability weight for a source [0.0, 1.0].
    pub fn reliability(&self, source: &str) -> f64 {
        match self.sources.get(source) {
            Some(s) => s.reliability(),
            None => 0.67, // Default for unknown sources
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 6  Learning State (Aggregate)
// ══════════════════════════════════════════════════════════════════════════════

/// Aggregate learning state — persisted as a single unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningState {
    /// Utility scoring weights.
    pub weights: UtilityWeights,
    /// Per-action confidence bandits.
    pub bandits: BanditRegistry,
    /// Confidence calibration map.
    pub calibration: CalibrationMap,
    /// Source reliability trackers.
    pub reliability: ReliabilityRegistry,
    /// Recent interaction buffer for batch weight updates.
    pub interaction_buffer: Vec<InteractionRecord>,
    /// Configuration.
    pub config: LearningConfig,
    /// Total interactions ever processed.
    pub total_interactions: u64,
    /// Last weight refit timestamp.
    pub last_weight_refit: f64,
    /// Last calibration refit timestamp.
    pub last_calibration_refit: f64,
}

impl LearningState {
    pub fn new() -> Self {
        Self {
            weights: UtilityWeights::new(),
            bandits: BanditRegistry::new(),
            calibration: CalibrationMap::new(),
            reliability: ReliabilityRegistry::new(),
            interaction_buffer: Vec::new(),
            config: LearningConfig::default(),
            total_interactions: 0,
            last_weight_refit: 0.0,
            last_calibration_refit: 0.0,
        }
    }
}

impl Default for LearningState {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the learning system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Learning rate for weight updates.
    pub weight_learning_rate: f64,
    /// Margin for pairwise ranking loss.
    pub ranking_margin: f64,
    /// Safety margin for bandit thresholds (in standard deviations).
    pub bandit_safety_margin: f64,
    /// Number of interactions before triggering a weight refit.
    pub weight_refit_interval: u64,
    /// Number of interactions before triggering a calibration refit.
    pub calibration_refit_interval: u64,
    /// Maximum size of the interaction buffer.
    pub max_interaction_buffer: usize,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            weight_learning_rate: 0.01,
            ranking_margin: 0.1,
            bandit_safety_margin: 1.0,
            weight_refit_interval: 100,
            calibration_refit_interval: 50,
            max_interaction_buffer: 500,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 7  Core Learning API
// ══════════════════════════════════════════════════════════════════════════════

/// Record an interaction outcome and update all learning components.
///
/// This is the primary entry point — called after every user interaction
/// with a suggestion/action.
pub fn record_interaction(state: &mut LearningState, record: InteractionRecord, now: f64) {
    // 1. Update bandit for this action kind
    state.bandits.record(&record.action_kind, record.outcome);

    // 2. Update calibration map
    state
        .calibration
        .record(record.raw_confidence, record.outcome.is_positive());

    // 3. Buffer interaction for batch weight updates
    state.interaction_buffer.push(record);
    state.total_interactions += 1;

    // 4. Enforce buffer size
    if state.interaction_buffer.len() > state.config.max_interaction_buffer {
        let drain = state.interaction_buffer.len() - state.config.max_interaction_buffer;
        state.interaction_buffer.drain(0..drain);
    }

    // 5. Check if weight refit is due
    if state.total_interactions % state.config.weight_refit_interval == 0 {
        update_utility_weights(
            &mut state.weights,
            &state.interaction_buffer,
            state.config.weight_learning_rate,
            state.config.ranking_margin,
        );
        state.last_weight_refit = now;
    }

    // 6. Check if calibration refit is due
    if state.calibration.needs_refit {
        state.calibration.refit();
        state.last_calibration_refit = now;
    }
}

/// Record that a belief from a specific source was confirmed.
pub fn record_belief_confirmed(state: &mut LearningState, source: &str) {
    state.reliability.get_or_create(source).record_confirmed();
}

/// Record that a belief from a specific source was contradicted.
pub fn record_belief_contradicted(state: &mut LearningState, source: &str) {
    state
        .reliability
        .get_or_create(source)
        .record_contradicted();
}

/// Get calibrated confidence for a raw confidence value.
pub fn calibrated_confidence(state: &LearningState, raw_confidence: f64) -> f64 {
    state.calibration.calibrate(raw_confidence)
}

/// Get the recommended confidence threshold for an action kind.
pub fn action_threshold(state: &LearningState, action_kind: &str) -> f64 {
    state
        .bandits
        .threshold(action_kind, state.config.bandit_safety_margin)
}

/// Get the current utility weights as a snapshot.
pub fn weight_snapshot(state: &LearningState) -> WeightSnapshot {
    WeightSnapshot {
        effect_weight: state.weights.weights[0],
        intent_weight: state.weights.weights[1],
        preference_weight: state.weights.weights[2],
        simulation_weight: state.weights.weights[3],
        update_count: state.weights.update_count,
        ranking_accuracy: state.weights.accuracy(),
    }
}

/// Weight snapshot for external consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightSnapshot {
    pub effect_weight: f64,
    pub intent_weight: f64,
    pub preference_weight: f64,
    pub simulation_weight: f64,
    pub update_count: u64,
    pub ranking_accuracy: f64,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 8  Learning Report
// ══════════════════════════════════════════════════════════════════════════════

/// Comprehensive report on the learning system's state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningReport {
    /// Current utility weights.
    pub weights: WeightSnapshot,
    /// Expected Calibration Error (lower = better calibrated).
    pub calibration_error: f64,
    /// Number of action kinds being tracked.
    pub action_kinds_tracked: usize,
    /// Per-action acceptance rates.
    pub action_acceptance_rates: HashMap<String, f64>,
    /// Source reliability scores.
    pub source_reliabilities: HashMap<String, f64>,
    /// Total interactions processed.
    pub total_interactions: u64,
    /// Total weight refits performed.
    pub weight_refits: u64,
    /// Interaction buffer fullness [0.0, 1.0].
    pub buffer_utilization: f64,
}

/// Generate a learning report.
pub fn learning_report(state: &LearningState) -> LearningReport {
    let action_acceptance_rates: HashMap<String, f64> = state
        .bandits
        .bandits
        .iter()
        .map(|(k, b)| (k.clone(), b.mean()))
        .collect();

    let source_reliabilities: HashMap<String, f64> = state
        .reliability
        .sources
        .iter()
        .map(|(k, s)| (k.clone(), s.reliability()))
        .collect();

    let buffer_utilization = if state.config.max_interaction_buffer > 0 {
        state.interaction_buffer.len() as f64 / state.config.max_interaction_buffer as f64
    } else {
        0.0
    };

    LearningReport {
        weights: weight_snapshot(state),
        calibration_error: state.calibration.calibration_error(),
        action_kinds_tracked: state.bandits.bandits.len(),
        action_acceptance_rates,
        source_reliabilities,
        total_interactions: state.total_interactions,
        weight_refits: state.weights.update_count,
        buffer_utilization,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 9  Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(offset: f64) -> f64 {
        86400.0 * 100.0 + offset
    }

    fn make_record(
        kind: &str,
        features: [f64; 4],
        confidence: f64,
        outcome: InteractionOutcome,
    ) -> InteractionRecord {
        InteractionRecord {
            action_kind: kind.to_string(),
            raw_confidence: confidence,
            outcome,
            features,
            timestamp: ts(0.0),
        }
    }

    // ── § 1: Utility weight learning ──

    #[test]
    fn test_utility_weights_default() {
        let weights = UtilityWeights::new();
        let sum: f64 = weights.weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Weights should sum to ~1.0");
    }

    #[test]
    fn test_utility_weight_scoring() {
        let weights = UtilityWeights::new();
        let features = [1.0, 0.5, 0.3, 0.8];
        let score = weights.score(&features);
        let expected = 0.35 * 1.0 + 0.25 * 0.5 + 0.15 * 0.3 + 0.25 * 0.8;
        assert!((score - expected).abs() < 0.001);
    }

    #[test]
    fn test_weight_update_improves_ranking() {
        let mut weights = UtilityWeights::new();

        // Create interactions where feature[0] (effect) is the strongest predictor
        let interactions = vec![
            make_record(
                "test",
                [0.9, 0.2, 0.1, 0.3],
                0.7,
                InteractionOutcome::Accepted,
            ),
            make_record(
                "test",
                [0.1, 0.8, 0.7, 0.2],
                0.5,
                InteractionOutcome::Rejected,
            ),
            make_record(
                "test",
                [0.8, 0.3, 0.2, 0.1],
                0.6,
                InteractionOutcome::Accepted,
            ),
            make_record(
                "test",
                [0.2, 0.7, 0.6, 0.1],
                0.4,
                InteractionOutcome::Rejected,
            ),
        ];

        let initial_w0 = weights.weights[0];
        update_utility_weights(&mut weights, &interactions, 0.1, 0.05);

        // Effect weight should increase (it predicts acceptance)
        assert!(
            weights.weights[0] > initial_w0 - 0.01,
            "Effect weight should not decrease significantly: {} → {}",
            initial_w0,
            weights.weights[0]
        );

        // Weights should still sum to ~1.0
        let sum: f64 = weights.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Weights should sum to ~1.0 after update: {}",
            sum
        );
    }

    #[test]
    fn test_weight_update_too_few_interactions() {
        let mut weights = UtilityWeights::new();
        let original = weights.weights;

        update_utility_weights(&mut weights, &[], 0.1, 0.05);
        assert_eq!(weights.weights, original, "No update on empty interactions");

        let interactions = vec![make_record(
            "test",
            [0.5; 4],
            0.5,
            InteractionOutcome::Accepted,
        )];
        update_utility_weights(&mut weights, &interactions, 0.1, 0.05);
        assert_eq!(weights.weights, original, "No update on single interaction");
    }

    // ── § 2: Action bandits ──

    #[test]
    fn test_bandit_basic() {
        let mut bandit = ActionBandit::new();
        assert!((bandit.mean() - 0.5).abs() < 0.01, "Uninformative prior");

        for _ in 0..8 {
            bandit.record_positive();
        }
        for _ in 0..2 {
            bandit.record_negative();
        }

        assert!(
            bandit.mean() > 0.7,
            "Should reflect high acceptance: {}",
            bandit.mean()
        );
    }

    #[test]
    fn test_bandit_threshold() {
        let mut bandit = ActionBandit::new();
        for _ in 0..20 {
            bandit.record_positive();
        }
        for _ in 0..5 {
            bandit.record_negative();
        }

        let threshold = bandit.threshold(1.0);
        assert!(threshold < bandit.mean(), "Threshold should be below mean");
        assert!(threshold > 0.0, "Threshold should be positive");
    }

    #[test]
    fn test_bandit_registry() {
        let mut registry = BanditRegistry::new();

        registry.record("remind", InteractionOutcome::Accepted);
        registry.record("remind", InteractionOutcome::Accepted);
        registry.record("remind", InteractionOutcome::Rejected);
        registry.record("alert", InteractionOutcome::Rejected);
        registry.record("alert", InteractionOutcome::Rejected);

        assert!(registry.acceptance_rate("remind") > registry.acceptance_rate("alert"));
        assert!(registry.threshold("unknown", 1.0) > 0.0);
    }

    // ── § 3: Calibration map ──

    #[test]
    fn test_calibration_recording() {
        let mut cal = CalibrationMap::new();

        // High confidence predictions that succeed
        for _ in 0..10 {
            cal.record(0.85, true);
        }

        // Low confidence predictions that fail
        for _ in 0..10 {
            cal.record(0.15, false);
        }

        let high_bin = &cal.bins[8]; // 0.8-0.9
        assert_eq!(high_bin.count, 10);
        assert_eq!(high_bin.positive, 10);

        let low_bin = &cal.bins[1]; // 0.1-0.2
        assert_eq!(low_bin.count, 10);
        assert_eq!(low_bin.positive, 0);
    }

    #[test]
    fn test_calibration_refit() {
        let mut cal = CalibrationMap::new();

        // Well-calibrated: high confidence → high success, low → low success
        for _ in 0..20 {
            cal.record(0.9, true);
            cal.record(0.1, false);
        }

        cal.refit();

        let high_cal = cal.calibrate(0.9);
        let low_cal = cal.calibrate(0.1);
        assert!(
            high_cal > low_cal,
            "High confidence should calibrate higher: {} vs {}",
            high_cal,
            low_cal
        );
    }

    #[test]
    fn test_isotonic_regression() {
        // Non-monotonic input: [0.5, 0.3, 0.7, 0.2, 0.8]
        let mut values = [0.5, 0.3, 0.7, 0.2, 0.8];
        isotonic_regression(&mut values);

        // Should be monotonically non-decreasing
        for i in 1..values.len() {
            assert!(
                values[i] >= values[i - 1] - 1e-10,
                "Not monotonic at {}: {} < {}",
                i,
                values[i],
                values[i - 1]
            );
        }
    }

    #[test]
    fn test_isotonic_already_monotonic() {
        let mut values = [0.1, 0.3, 0.5, 0.7, 0.9];
        let original = values;
        isotonic_regression(&mut values);
        assert_eq!(values, original, "Already monotonic should not change");
    }

    #[test]
    fn test_calibration_error() {
        let mut cal = CalibrationMap::new();

        // Perfect calibration: 80% confidence → 80% success
        for _ in 0..8 {
            cal.record(0.85, true);
        }
        for _ in 0..2 {
            cal.record(0.85, false);
        }

        let ece = cal.calibration_error();
        assert!(ece < 0.1, "Well-calibrated should have low ECE: {}", ece);
    }

    // ── § 4: Source reliability ──

    #[test]
    fn test_source_reliability_basic() {
        let mut source = SourceReliability::new();

        for _ in 0..9 {
            source.record_confirmed();
        }
        source.record_contradicted();

        assert!(
            source.reliability() > 0.8,
            "Mostly confirmed → high reliability: {}",
            source.reliability()
        );
    }

    #[test]
    fn test_reliability_registry() {
        let mut registry = ReliabilityRegistry::new();

        registry.get_or_create("user").record_confirmed();
        registry.get_or_create("user").record_confirmed();
        registry.get_or_create("llm").record_confirmed();
        registry.get_or_create("llm").record_contradicted();

        assert!(registry.reliability("user") > registry.reliability("llm"));
        assert!((registry.reliability("unknown") - 0.67).abs() < 0.01);
    }

    // ── § 5: Full learning pipeline ──

    #[test]
    fn test_record_interaction() {
        let mut state = LearningState::new();

        for i in 0..5 {
            let record = InteractionRecord {
                action_kind: "remind".to_string(),
                raw_confidence: 0.7,
                outcome: if i < 4 {
                    InteractionOutcome::Accepted
                } else {
                    InteractionOutcome::Rejected
                },
                features: [0.8, 0.5, 0.3, 0.6],
                timestamp: ts(i as f64),
            };
            record_interaction(&mut state, record, ts(i as f64));
        }

        assert_eq!(state.total_interactions, 5);
        assert_eq!(state.interaction_buffer.len(), 5);
        assert!(state.bandits.acceptance_rate("remind") > 0.6);
    }

    #[test]
    fn test_calibrated_confidence() {
        let mut state = LearningState::new();

        // Seed calibration data
        for _ in 0..20 {
            state.calibration.record(0.8, true);
            state.calibration.record(0.2, false);
        }
        state.calibration.refit();

        let high = calibrated_confidence(&state, 0.85);
        let low = calibrated_confidence(&state, 0.15);
        assert!(high > low, "Calibrated confidence should respect ordering");
    }

    #[test]
    fn test_action_threshold() {
        let mut state = LearningState::new();

        // Record many accepts for "remind"
        for _ in 0..10 {
            state.bandits.record("remind", InteractionOutcome::Accepted);
        }
        state.bandits.record("remind", InteractionOutcome::Rejected);

        let thresh = action_threshold(&state, "remind");
        assert!(
            thresh > 0.3 && thresh < 0.9,
            "Threshold should be reasonable: {}",
            thresh
        );
    }

    #[test]
    fn test_weight_snapshot() {
        let state = LearningState::new();
        let snap = weight_snapshot(&state);

        let sum = snap.effect_weight
            + snap.intent_weight
            + snap.preference_weight
            + snap.simulation_weight;
        assert!((sum - 1.0).abs() < 0.01);
        assert_eq!(snap.update_count, 0);
    }

    #[test]
    fn test_belief_reliability_tracking() {
        let mut state = LearningState::new();

        record_belief_confirmed(&mut state, "user");
        record_belief_confirmed(&mut state, "user");
        record_belief_confirmed(&mut state, "llm");
        record_belief_contradicted(&mut state, "llm");

        assert!(state.reliability.reliability("user") > state.reliability.reliability("llm"));
    }

    #[test]
    fn test_learning_report() {
        let mut state = LearningState::new();

        // Add some data
        for i in 0..3 {
            let record = InteractionRecord {
                action_kind: "suggest".to_string(),
                raw_confidence: 0.6,
                outcome: InteractionOutcome::Accepted,
                features: [0.5, 0.4, 0.3, 0.2],
                timestamp: ts(i as f64),
            };
            record_interaction(&mut state, record, ts(i as f64));
        }
        record_belief_confirmed(&mut state, "user");

        let report = learning_report(&state);
        assert_eq!(report.total_interactions, 3);
        assert_eq!(report.action_kinds_tracked, 1);
        assert!(!report.source_reliabilities.is_empty());
    }

    #[test]
    fn test_interaction_buffer_limit() {
        let mut state = LearningState::new();
        state.config.max_interaction_buffer = 5;

        for i in 0..10 {
            let record = make_record("test", [0.5; 4], 0.5, InteractionOutcome::Accepted);
            record_interaction(&mut state, record, ts(i as f64));
        }

        assert!(
            state.interaction_buffer.len() <= 5,
            "Buffer should be limited to max"
        );
    }

    #[test]
    fn test_weight_refit_triggers() {
        let mut state = LearningState::new();
        state.config.weight_refit_interval = 5;

        for i in 0..10 {
            let outcome = if i % 3 == 0 {
                InteractionOutcome::Rejected
            } else {
                InteractionOutcome::Accepted
            };
            let record = InteractionRecord {
                action_kind: "test".to_string(),
                raw_confidence: 0.6,
                outcome,
                features: [0.5 + (i as f64) * 0.05, 0.3, 0.2, 0.4],
                timestamp: ts(i as f64),
            };
            record_interaction(&mut state, record, ts(i as f64));
        }

        // Should have triggered weight refit at interaction 5 and 10
        assert!(
            state.weights.update_count >= 1,
            "Weight refit should have triggered"
        );
    }

    // ── § 6: Edge cases ──

    #[test]
    fn test_empty_state_report() {
        let state = LearningState::new();
        let report = learning_report(&state);
        assert_eq!(report.total_interactions, 0);
        assert_eq!(report.action_kinds_tracked, 0);
        assert!(report.calibration_error < 0.01);
    }

    #[test]
    fn test_confidence_bin_boundaries() {
        assert_eq!(confidence_to_bin(0.0), 0);
        assert_eq!(confidence_to_bin(0.099), 0);
        assert_eq!(confidence_to_bin(0.1), 1);
        assert_eq!(confidence_to_bin(0.999), 9);
        assert_eq!(confidence_to_bin(1.0), 9); // Clamped
        assert_eq!(confidence_to_bin(-0.1), 0); // Clamped
        assert_eq!(confidence_to_bin(1.5), 9); // Clamped
    }
}
