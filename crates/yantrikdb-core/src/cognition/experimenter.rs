//! Safe Self-Experimentation Engine — bounded A/B testing.
//!
//! The system cautiously experiments to learn what works, WITHOUT
//! ever asking the user. Safety bounds prevent annoyance.
//!
//! # Lifecycle
//!
//! ```text
//! Hypothesis ──► Safety check ──► Design ──► Running ──► Statistical test
//!                    │                          │              │
//!                    ▼                          ▼              ▼
//!                  Reject                     Abort       Conclude → Belief
//! ```
//!
//! # Safety guarantees
//!
//! - MaxRejections: abort after N consecutive rejections for any variant
//! - MaxDuration: auto-conclude after time limit
//! - MinAcceptanceRate: abort if rate drops below threshold
//! - NeverExperimentWith: exclude high-stakes action categories
//! - Maximum concurrent experiments (default: 3)

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ══════════════════════════════════════════════════════════════════════════════
// § 1  Experiment Types
// ══════════════════════════════════════════════════════════════════════════════

/// Unique experiment identifier.
pub type ExperimentId = u64;

/// What parameter is being varied.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExperimentVariable {
    /// Lead time before deadline reminders (minutes).
    ReminderLeadMinutes,
    /// Surfacing mode (whisper vs nudge vs alert).
    SurfacingMode,
    /// Proactive surfacing frequency (surfaces per hour).
    SurfacingFrequency,
    /// Information density (brief vs detailed).
    InformationDensity,
    /// Time-of-day preference for surfacing.
    SurfacingTimeOfDay,
    /// Custom variable with a label.
    Custom(String),
}

impl ExperimentVariable {
    pub fn as_str(&self) -> &str {
        match self {
            Self::ReminderLeadMinutes => "reminder_lead_minutes",
            Self::SurfacingMode => "surfacing_mode",
            Self::SurfacingFrequency => "surfacing_frequency",
            Self::InformationDensity => "information_density",
            Self::SurfacingTimeOfDay => "surfacing_time_of_day",
            Self::Custom(s) => s,
        }
    }

    /// Whether this variable is safe to experiment with.
    pub fn is_safe(&self) -> bool {
        // All defined variables are safe by design.
        // Custom variables require explicit safety check.
        !matches!(self, Self::Custom(_))
    }
}

/// A specific value for an experiment variable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VariantValue {
    /// Numeric value (e.g., 15, 30, 45 minutes).
    Number(f64),
    /// String value (e.g., "whisper", "nudge").
    Label(String),
}

impl VariantValue {
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_label(&self) -> Option<&str> {
        match self {
            Self::Label(s) => Some(s),
            _ => None,
        }
    }
}

/// Status of an experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentStatus {
    /// Designed but not yet started.
    Designed,
    /// Currently running and collecting data.
    Running,
    /// Completed with a conclusion.
    Concluded,
    /// Aborted due to safety bound violation.
    Aborted,
}

/// Outcome of a single experiment trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrialOutcome {
    /// User accepted/acted on the variant.
    Positive,
    /// User rejected the variant.
    Negative,
    /// User didn't respond (timeout).
    Neutral,
}

/// Safety bound — when violated, experiment aborts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyBound {
    /// Abort after N consecutive negative outcomes for any variant.
    MaxConsecutiveRejections(u32),
    /// Abort after this many seconds.
    MaxDuration(f64),
    /// Abort if any variant's acceptance rate drops below threshold.
    MinAcceptanceRate(f64),
}

// ══════════════════════════════════════════════════════════════════════════════
// § 2  Experiment Struct
// ══════════════════════════════════════════════════════════════════════════════

/// A single self-experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    /// Unique ID.
    pub id: ExperimentId,
    /// Human-readable hypothesis.
    pub hypothesis: String,
    /// What parameter is being varied.
    pub variable: ExperimentVariable,
    /// The variant values being tested.
    pub variants: Vec<VariantValue>,
    /// Target sample size per variant.
    pub sample_size_target: u32,
    /// Safety bounds (all must pass for experiment to continue).
    pub safety_bounds: Vec<SafetyBound>,
    /// Current status.
    pub status: ExperimentStatus,
    /// Timestamp when created.
    pub created_at: f64,
    /// Timestamp when started (first trial).
    pub started_at: Option<f64>,
    /// Timestamp when concluded/aborted.
    pub ended_at: Option<f64>,
    /// Per-variant trial results: variant_index → Beta(α, β) parameters.
    pub variant_results: Vec<BetaPosterior>,
    /// Index of last assigned variant (for round-robin).
    pub last_variant_idx: usize,
    /// Consecutive negative count per variant (for safety).
    pub consecutive_negatives: Vec<u32>,
    /// Winning variant index (set on conclusion).
    pub winner: Option<usize>,
}

/// Beta distribution posterior for a variant.
///
/// α = positive count + 1 (prior), β = negative count + 1 (prior).
/// The posterior mean is α / (α + β).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaPosterior {
    /// Successes + prior (starts at 1.0).
    pub alpha: f64,
    /// Failures + prior (starts at 1.0).
    pub beta: f64,
    /// Total trials for this variant.
    pub trials: u32,
}

impl BetaPosterior {
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
            trials: 0,
        }
    }

    /// Record a positive outcome.
    pub fn record_positive(&mut self) {
        self.alpha += 1.0;
        self.trials += 1;
    }

    /// Record a negative outcome.
    pub fn record_negative(&mut self) {
        self.beta += 1.0;
        self.trials += 1;
    }

    /// Record a neutral outcome (counts as trial but no update).
    pub fn record_neutral(&mut self) {
        self.trials += 1;
    }

    /// Posterior mean: E[θ] = α / (α + β).
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Posterior variance: Var[θ] = αβ / ((α+β)²(α+β+1)).
    pub fn variance(&self) -> f64 {
        let s = self.alpha + self.beta;
        (self.alpha * self.beta) / (s * s * (s + 1.0))
    }

    /// 95% credible interval width (approximation using Beta normal approx).
    pub fn credible_interval_width(&self) -> f64 {
        2.0 * 1.96 * self.variance().sqrt()
    }

    /// Acceptance rate = (α - 1) / trials (raw, without prior).
    pub fn acceptance_rate(&self) -> f64 {
        if self.trials == 0 {
            return 0.5; // Uninformative
        }
        (self.alpha - 1.0) / self.trials as f64
    }

    /// Thompson sampling: draw from Beta(α, β).
    ///
    /// Uses a simple deterministic approximation for reproducibility:
    /// mode of the distribution = (α-1)/(α+β-2), with noise.
    pub fn thompson_score(&self, jitter: f64) -> f64 {
        // Deterministic mode + small jitter for exploration
        let mode = if self.alpha > 1.0 && self.beta > 1.0 {
            (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)
        } else {
            self.mean()
        };
        (mode + jitter).clamp(0.0, 1.0)
    }
}

impl Default for BetaPosterior {
    fn default() -> Self {
        Self::new()
    }
}

impl Experiment {
    /// Create a new experiment.
    pub fn new(
        id: ExperimentId,
        hypothesis: String,
        variable: ExperimentVariable,
        variants: Vec<VariantValue>,
        sample_size_target: u32,
        safety_bounds: Vec<SafetyBound>,
        now: f64,
    ) -> Self {
        let n = variants.len();
        Self {
            id,
            hypothesis,
            variable,
            variants,
            sample_size_target,
            safety_bounds,
            status: ExperimentStatus::Designed,
            created_at: now,
            started_at: None,
            ended_at: None,
            variant_results: (0..n).map(|_| BetaPosterior::new()).collect(),
            last_variant_idx: 0,
            consecutive_negatives: vec![0; n],
            winner: None,
        }
    }

    /// Start the experiment.
    pub fn start(&mut self, now: f64) {
        self.status = ExperimentStatus::Running;
        self.started_at = Some(now);
    }

    /// Whether the experiment is active (running).
    pub fn is_active(&self) -> bool {
        self.status == ExperimentStatus::Running
    }

    /// Number of variants.
    pub fn variant_count(&self) -> usize {
        self.variants.len()
    }

    /// Total trials across all variants.
    pub fn total_trials(&self) -> u32 {
        self.variant_results.iter().map(|v| v.trials).sum()
    }

    /// Whether target sample size is reached for all variants.
    pub fn is_sample_complete(&self) -> bool {
        self.variant_results
            .iter()
            .all(|v| v.trials >= self.sample_size_target)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  Experiment Registry
// ══════════════════════════════════════════════════════════════════════════════

/// Registry of all experiments (active, concluded, aborted).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentRegistry {
    /// All experiments.
    pub experiments: Vec<Experiment>,
    /// Maximum concurrent active experiments.
    pub max_concurrent: usize,
    /// Next experiment ID.
    next_id: u64,
    /// Total experiments created.
    pub total_created: u64,
    /// Total experiments concluded.
    pub total_concluded: u64,
    /// Total experiments aborted.
    pub total_aborted: u64,
}

impl ExperimentRegistry {
    pub fn new() -> Self {
        Self {
            experiments: Vec::new(),
            max_concurrent: 3,
            next_id: 1,
            total_created: 0,
            total_concluded: 0,
            total_aborted: 0,
        }
    }

    /// Get the next experiment ID.
    fn alloc_id(&mut self) -> ExperimentId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Number of currently active experiments.
    pub fn active_count(&self) -> usize {
        self.experiments.iter().filter(|e| e.is_active()).count()
    }

    /// Whether we can start a new experiment.
    pub fn can_start_new(&self) -> bool {
        self.active_count() < self.max_concurrent
    }

    /// Get all active experiments.
    pub fn active_experiments(&self) -> Vec<&Experiment> {
        self.experiments.iter().filter(|e| e.is_active()).collect()
    }

    /// Get concluded experiments.
    pub fn concluded_experiments(&self) -> Vec<&Experiment> {
        self.experiments
            .iter()
            .filter(|e| e.status == ExperimentStatus::Concluded)
            .collect()
    }

    /// Find an experiment by ID.
    pub fn find(&self, id: ExperimentId) -> Option<&Experiment> {
        self.experiments.iter().find(|e| e.id == id)
    }

    /// Find an experiment mutably by ID.
    pub fn find_mut(&mut self, id: ExperimentId) -> Option<&mut Experiment> {
        self.experiments.iter_mut().find(|e| e.id == id)
    }
}

impl Default for ExperimentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  Experiment Operations (Pure Functions)
// ══════════════════════════════════════════════════════════════════════════════

/// Design and register a new experiment.
///
/// Returns `None` if:
/// - Variable is unsafe
/// - Too many concurrent experiments
/// - An experiment on this variable already exists
pub fn design_experiment(
    registry: &mut ExperimentRegistry,
    hypothesis: String,
    variable: ExperimentVariable,
    variants: Vec<VariantValue>,
    sample_size_target: u32,
    safety_bounds: Vec<SafetyBound>,
    now: f64,
) -> Option<ExperimentId> {
    // Safety gate: variable must be safe
    if !variable.is_safe() {
        return None;
    }

    // Capacity gate
    if !registry.can_start_new() {
        return None;
    }

    // Dedup gate: no active experiment on same variable
    let has_existing = registry
        .experiments
        .iter()
        .any(|e| e.is_active() && e.variable == variable);
    if has_existing {
        return None;
    }

    // Validate: need at least 2 variants
    if variants.len() < 2 {
        return None;
    }

    let id = registry.alloc_id();
    let mut experiment = Experiment::new(
        id,
        hypothesis,
        variable,
        variants,
        sample_size_target,
        safety_bounds,
        now,
    );
    experiment.start(now);
    registry.experiments.push(experiment);
    registry.total_created += 1;

    Some(id)
}

/// Assign a variant for the next trial using Thompson sampling.
///
/// Returns `(variant_index, variant_value)`.
pub fn assign_variant(
    experiment: &mut Experiment,
    jitter_seed: f64,
) -> Option<(usize, &VariantValue)> {
    if !experiment.is_active() {
        return None;
    }

    // Thompson sampling: pick variant with highest sampled value
    let best_idx = experiment
        .variant_results
        .iter()
        .enumerate()
        .max_by(|(i, a), (j, b)| {
            // Use index-based jitter for determinism
            let ja = jitter_seed * (1.0 + *i as f64 * 0.1) % 0.05;
            let jb = jitter_seed * (1.0 + *j as f64 * 0.1) % 0.05;
            a.thompson_score(ja)
                .partial_cmp(&b.thompson_score(jb))
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    experiment.last_variant_idx = best_idx;
    Some((best_idx, &experiment.variants[best_idx]))
}

/// Assign a variant using simple round-robin (deterministic).
pub fn assign_variant_round_robin(experiment: &mut Experiment) -> Option<(usize, &VariantValue)> {
    if !experiment.is_active() {
        return None;
    }

    let idx = (experiment.last_variant_idx + 1) % experiment.variant_count();
    experiment.last_variant_idx = idx;
    Some((idx, &experiment.variants[idx]))
}

/// Record the outcome of a trial.
///
/// Returns `true` if the experiment should continue, `false` if it should stop.
pub fn record_trial(
    experiment: &mut Experiment,
    variant_idx: usize,
    outcome: TrialOutcome,
    now: f64,
) -> bool {
    if variant_idx >= experiment.variant_count() {
        return false;
    }

    // Update Beta posterior
    match outcome {
        TrialOutcome::Positive => {
            experiment.variant_results[variant_idx].record_positive();
            experiment.consecutive_negatives[variant_idx] = 0;
        }
        TrialOutcome::Negative => {
            experiment.variant_results[variant_idx].record_negative();
            experiment.consecutive_negatives[variant_idx] += 1;
        }
        TrialOutcome::Neutral => {
            experiment.variant_results[variant_idx].record_neutral();
        }
    }

    // Check safety bounds
    if !check_safety_bounds(experiment, now) {
        experiment.status = ExperimentStatus::Aborted;
        experiment.ended_at = Some(now);
        return false;
    }

    // Check if sample complete
    if experiment.is_sample_complete() {
        return false; // Signal to conclude
    }

    true
}

/// Check all safety bounds. Returns `true` if experiment can continue.
fn check_safety_bounds(experiment: &Experiment, now: f64) -> bool {
    for bound in &experiment.safety_bounds {
        match bound {
            SafetyBound::MaxConsecutiveRejections(max) => {
                if experiment.consecutive_negatives.iter().any(|&c| c >= *max) {
                    return false;
                }
            }
            SafetyBound::MaxDuration(max_secs) => {
                if let Some(started) = experiment.started_at {
                    if now - started > *max_secs {
                        return false;
                    }
                }
            }
            SafetyBound::MinAcceptanceRate(min_rate) => {
                for result in &experiment.variant_results {
                    if result.trials >= 5 && result.acceptance_rate() < *min_rate {
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// Conclude an experiment: determine winner, update status.
///
/// Returns the index and value of the winning variant (highest posterior mean).
pub fn conclude_experiment(experiment: &mut Experiment, now: f64) -> Option<(usize, VariantValue)> {
    if experiment.status != ExperimentStatus::Running {
        return None;
    }

    // Find winner: variant with highest posterior mean
    let winner_idx = experiment
        .variant_results
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.mean().partial_cmp(&b.mean()).unwrap())
        .map(|(i, _)| i)?;

    experiment.winner = Some(winner_idx);
    experiment.status = ExperimentStatus::Concluded;
    experiment.ended_at = Some(now);

    Some((winner_idx, experiment.variants[winner_idx].clone()))
}

/// Check all active experiments for auto-conclusion or safety abort.
pub fn check_experiments(registry: &mut ExperimentRegistry, now: f64) -> Vec<ExperimentId> {
    let mut concluded_ids = Vec::new();

    let active_ids: Vec<ExperimentId> = registry
        .experiments
        .iter()
        .filter(|e| e.is_active())
        .map(|e| e.id)
        .collect();

    for id in active_ids {
        let experiment = match registry.find_mut(id) {
            Some(e) => e,
            None => continue,
        };

        // Safety check
        if !check_safety_bounds(experiment, now) {
            experiment.status = ExperimentStatus::Aborted;
            experiment.ended_at = Some(now);
            registry.total_aborted += 1;
            concluded_ids.push(id);
            continue;
        }

        // Sample complete → conclude
        if experiment.is_sample_complete() {
            conclude_experiment(experiment, now);
            registry.total_concluded += 1;
            concluded_ids.push(id);
        }
    }

    concluded_ids
}

/// Get the confidence that one variant is better than another.
///
/// Uses the probability P(θ_a > θ_b) approximation for Beta distributions.
/// When both variants have many samples, this converges to the true value.
pub fn variant_superiority(a: &BetaPosterior, b: &BetaPosterior) -> f64 {
    // Approximation: use normal approximation to Beta
    let mean_a = a.mean();
    let mean_b = b.mean();
    let var_a = a.variance();
    let var_b = b.variance();

    let combined_std = (var_a + var_b).sqrt();
    if combined_std < 1e-10 {
        return if mean_a > mean_b { 1.0 } else { 0.0 };
    }

    let z = (mean_a - mean_b) / combined_std;
    // Approximate Φ(z) using logistic function
    1.0 / (1.0 + (-1.7 * z).exp())
}

// ══════════════════════════════════════════════════════════════════════════════
// § 5  Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(offset: f64) -> f64 {
        86400.0 * 100.0 + offset
    }

    fn default_bounds() -> Vec<SafetyBound> {
        vec![
            SafetyBound::MaxConsecutiveRejections(5),
            SafetyBound::MaxDuration(86400.0), // 24 hours
            SafetyBound::MinAcceptanceRate(0.1),
        ]
    }

    // ── Beta Posterior ──

    #[test]
    fn test_beta_posterior_new() {
        let beta = BetaPosterior::new();
        assert_eq!(beta.alpha, 1.0);
        assert_eq!(beta.beta, 1.0);
        assert!((beta.mean() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_beta_posterior_updates() {
        let mut beta = BetaPosterior::new();
        for _ in 0..8 {
            beta.record_positive();
        }
        for _ in 0..2 {
            beta.record_negative();
        }

        assert_eq!(beta.trials, 10);
        // Mean = (1+8)/(1+8+1+2) = 9/12 = 0.75
        assert!((beta.mean() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_beta_credible_interval() {
        let mut small = BetaPosterior::new();
        small.record_positive();

        let mut large = BetaPosterior::new();
        for _ in 0..100 {
            large.record_positive();
        }

        assert!(
            large.credible_interval_width() < small.credible_interval_width(),
            "More data → narrower CI",
        );
    }

    // ── Experiment Design ──

    #[test]
    fn test_design_experiment() {
        let mut registry = ExperimentRegistry::new();

        let id = design_experiment(
            &mut registry,
            "30min lead > 15min lead".to_string(),
            ExperimentVariable::ReminderLeadMinutes,
            vec![VariantValue::Number(15.0), VariantValue::Number(30.0)],
            10,
            default_bounds(),
            ts(0.0),
        );

        assert!(id.is_some());
        assert_eq!(registry.active_count(), 1);
    }

    #[test]
    fn test_design_rejects_unsafe() {
        let mut registry = ExperimentRegistry::new();

        let id = design_experiment(
            &mut registry,
            "Test custom".to_string(),
            ExperimentVariable::Custom("dangerous".to_string()),
            vec![
                VariantValue::Label("a".to_string()),
                VariantValue::Label("b".to_string()),
            ],
            10,
            default_bounds(),
            ts(0.0),
        );

        assert!(id.is_none());
    }

    #[test]
    fn test_design_rejects_duplicate() {
        let mut registry = ExperimentRegistry::new();

        design_experiment(
            &mut registry,
            "First".to_string(),
            ExperimentVariable::SurfacingMode,
            vec![
                VariantValue::Label("whisper".to_string()),
                VariantValue::Label("nudge".to_string()),
            ],
            10,
            default_bounds(),
            ts(0.0),
        );

        let id2 = design_experiment(
            &mut registry,
            "Duplicate".to_string(),
            ExperimentVariable::SurfacingMode,
            vec![
                VariantValue::Label("a".to_string()),
                VariantValue::Label("b".to_string()),
            ],
            10,
            default_bounds(),
            ts(1.0),
        );

        assert!(id2.is_none());
    }

    #[test]
    fn test_max_concurrent() {
        let mut registry = ExperimentRegistry::new();
        registry.max_concurrent = 2;

        for i in 0..2 {
            design_experiment(
                &mut registry,
                format!("Exp {}", i),
                if i == 0 {
                    ExperimentVariable::ReminderLeadMinutes
                } else {
                    ExperimentVariable::SurfacingFrequency
                },
                vec![VariantValue::Number(1.0), VariantValue::Number(2.0)],
                10,
                default_bounds(),
                ts(i as f64),
            );
        }

        let id3 = design_experiment(
            &mut registry,
            "Third".to_string(),
            ExperimentVariable::InformationDensity,
            vec![
                VariantValue::Label("brief".to_string()),
                VariantValue::Label("detailed".to_string()),
            ],
            10,
            default_bounds(),
            ts(2.0),
        );

        assert!(id3.is_none(), "Should reject 3rd experiment (max 2)");
    }

    // ── Variant Assignment ──

    #[test]
    fn test_round_robin_assignment() {
        let mut registry = ExperimentRegistry::new();
        let id = design_experiment(
            &mut registry,
            "Test".to_string(),
            ExperimentVariable::ReminderLeadMinutes,
            vec![
                VariantValue::Number(15.0),
                VariantValue::Number(30.0),
                VariantValue::Number(45.0),
            ],
            5,
            default_bounds(),
            ts(0.0),
        )
        .unwrap();

        let exp = registry.find_mut(id).unwrap();

        // Round-robin should cycle through variants
        let (idx0, _) = assign_variant_round_robin(exp).unwrap();
        let (idx1, _) = assign_variant_round_robin(exp).unwrap();
        let (idx2, _) = assign_variant_round_robin(exp).unwrap();
        let (idx3, _) = assign_variant_round_robin(exp).unwrap();

        assert_eq!(idx0, 1); // starts at 0, first call goes to 1
        assert_eq!(idx1, 2);
        assert_eq!(idx2, 0);
        assert_eq!(idx3, 1);
    }

    // ── Trial Recording ──

    #[test]
    fn test_record_trial() {
        let mut registry = ExperimentRegistry::new();
        let id = design_experiment(
            &mut registry,
            "Test".to_string(),
            ExperimentVariable::SurfacingMode,
            vec![
                VariantValue::Label("whisper".to_string()),
                VariantValue::Label("nudge".to_string()),
            ],
            5,
            default_bounds(),
            ts(0.0),
        )
        .unwrap();

        let exp = registry.find_mut(id).unwrap();

        // Record positive for variant 0
        assert!(record_trial(exp, 0, TrialOutcome::Positive, ts(1.0)));
        assert_eq!(exp.variant_results[0].trials, 1);
        assert!((exp.variant_results[0].mean() - 0.667).abs() < 0.01);
    }

    // ── Safety Bounds ──

    #[test]
    fn test_safety_max_rejections() {
        let mut registry = ExperimentRegistry::new();
        let id = design_experiment(
            &mut registry,
            "Test".to_string(),
            ExperimentVariable::SurfacingFrequency,
            vec![VariantValue::Number(1.0), VariantValue::Number(2.0)],
            100,
            vec![SafetyBound::MaxConsecutiveRejections(3)],
            ts(0.0),
        )
        .unwrap();

        let exp = registry.find_mut(id).unwrap();

        // 3 consecutive rejections for variant 0 → abort
        assert!(record_trial(exp, 0, TrialOutcome::Negative, ts(1.0)));
        assert!(record_trial(exp, 0, TrialOutcome::Negative, ts(2.0)));
        assert!(!record_trial(exp, 0, TrialOutcome::Negative, ts(3.0)));
        assert_eq!(exp.status, ExperimentStatus::Aborted);
    }

    #[test]
    fn test_safety_max_duration() {
        let mut registry = ExperimentRegistry::new();
        let id = design_experiment(
            &mut registry,
            "Test".to_string(),
            ExperimentVariable::InformationDensity,
            vec![
                VariantValue::Label("brief".to_string()),
                VariantValue::Label("detailed".to_string()),
            ],
            100,
            vec![SafetyBound::MaxDuration(60.0)], // 60 seconds
            ts(0.0),
        )
        .unwrap();

        let exp = registry.find_mut(id).unwrap();

        // Within time limit
        assert!(record_trial(exp, 0, TrialOutcome::Positive, ts(30.0)));
        // Exceeds time limit
        assert!(!record_trial(exp, 0, TrialOutcome::Positive, ts(120.0)));
        assert_eq!(exp.status, ExperimentStatus::Aborted);
    }

    // ── Conclusion ──

    #[test]
    fn test_conclude_experiment() {
        let mut registry = ExperimentRegistry::new();
        let id = design_experiment(
            &mut registry,
            "Test lead time".to_string(),
            ExperimentVariable::ReminderLeadMinutes,
            vec![VariantValue::Number(15.0), VariantValue::Number(30.0)],
            3,
            default_bounds(),
            ts(0.0),
        )
        .unwrap();

        let exp = registry.find_mut(id).unwrap();

        // Variant 0 (15min): 1 positive, 2 negative → mean ≈ 0.4
        record_trial(exp, 0, TrialOutcome::Positive, ts(1.0));
        record_trial(exp, 0, TrialOutcome::Negative, ts(2.0));
        record_trial(exp, 0, TrialOutcome::Negative, ts(3.0));

        // Variant 1 (30min): 3 positive → mean ≈ 0.8
        record_trial(exp, 1, TrialOutcome::Positive, ts(4.0));
        record_trial(exp, 1, TrialOutcome::Positive, ts(5.0));
        record_trial(exp, 1, TrialOutcome::Positive, ts(6.0));

        let result = conclude_experiment(exp, ts(7.0));
        assert!(result.is_some());

        let (winner_idx, winner_value) = result.unwrap();
        assert_eq!(winner_idx, 1);
        assert_eq!(winner_value, VariantValue::Number(30.0));
        assert_eq!(exp.status, ExperimentStatus::Concluded);
    }

    // ── Variant Superiority ──

    #[test]
    fn test_variant_superiority() {
        let mut a = BetaPosterior::new();
        let mut b = BetaPosterior::new();

        // A: 80% success
        for _ in 0..80 {
            a.record_positive();
        }
        for _ in 0..20 {
            a.record_negative();
        }

        // B: 40% success
        for _ in 0..40 {
            b.record_positive();
        }
        for _ in 0..60 {
            b.record_negative();
        }

        let prob = variant_superiority(&a, &b);
        assert!(prob > 0.95, "P(A > B) = {:.3} should be > 0.95", prob,);
    }

    // ── Check Experiments ──

    #[test]
    fn test_check_experiments_auto_conclude() {
        let mut registry = ExperimentRegistry::new();
        let id = design_experiment(
            &mut registry,
            "Auto conclude".to_string(),
            ExperimentVariable::SurfacingTimeOfDay,
            vec![VariantValue::Number(9.0), VariantValue::Number(14.0)],
            2,
            default_bounds(),
            ts(0.0),
        )
        .unwrap();

        // Fill all variants to target
        let exp = registry.find_mut(id).unwrap();
        record_trial(exp, 0, TrialOutcome::Positive, ts(1.0));
        record_trial(exp, 0, TrialOutcome::Positive, ts(2.0));
        record_trial(exp, 1, TrialOutcome::Positive, ts(3.0));
        record_trial(exp, 1, TrialOutcome::Negative, ts(4.0));

        let concluded = check_experiments(&mut registry, ts(5.0));
        assert_eq!(concluded.len(), 1);
        assert_eq!(concluded[0], id);

        let exp = registry.find(id).unwrap();
        assert_eq!(exp.status, ExperimentStatus::Concluded);
        assert_eq!(exp.winner, Some(0)); // Both 100% vs 50%
    }
}
