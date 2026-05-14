//! CK-4.4 — Meta-Cognition: Knowing What You Don't Know.
//!
//! The system's ability to assess its own reasoning quality,
//! detect when it's operating outside its competence, and decide
//! when to defer to an LLM or ask for clarification.
//!
//! # Design principles
//! - Pure functions only — no DB access
//! - All signals are quantifiable and auditable
//! - Conservative: prefer escalation over silent failure
//! - Every abstain decision includes a human-readable reason

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::calibration::{CalibrationMap, LearningState, ReliabilityRegistry};
use crate::experimenter::ExperimentRegistry;
use crate::flywheel::{BeliefStage, BeliefStore};
use crate::observer::{EventBuffer, EventKind};
use crate::skills::SkillRegistry;
use crate::world_model::TransitionModel;

// ── §1: Meta-Cognitive Report ──────────────────────────────────────

/// Comprehensive meta-cognitive assessment of reasoning quality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaCognitiveReport {
    /// Evidence sparsity — how thin is the data underlying decisions?
    /// ∈ [0.0, 1.0]. 0.0 = rich evidence, 1.0 = nearly blind.
    pub evidence_sparsity: f64,

    /// Model disagreement — do internal subsystems agree?
    /// ∈ [0.0, 1.0]. 0.0 = consensus, 1.0 = complete disagreement.
    pub model_disagreement: f64,

    /// Contradiction density — ratio of unresolved belief conflicts.
    /// ∈ [0.0, 1.0]. Higher = more contradictions per active belief.
    pub contradiction_density: f64,

    /// Rolling prediction accuracy over recent predictions.
    /// ∈ [0.0, 1.0]. Based on world model + calibration accuracy.
    pub prediction_accuracy: f64,

    /// Calibration error — are confidence estimates well-calibrated?
    /// ∈ [0.0, 1.0]. Lower = better. Expected Calibration Error (ECE).
    pub calibration_error: f64,

    /// Coverage — proportion of (state, action) space with observations.
    /// ∈ [0.0, 1.0]. Higher = fewer blind spots.
    pub coverage: f64,

    /// Source reliability — aggregate trustworthiness of evidence sources.
    /// ∈ [0.0, 1.0]. Based on reliability registry.
    pub source_reliability: f64,

    /// Skill maturity — proportion of skills that are Validated or better.
    /// ∈ [0.0, 1.0]. Higher = more reliable action repertoire.
    pub skill_maturity: f64,

    /// Belief maturity — proportion of beliefs that are Established or better.
    /// ∈ [0.0, 1.0]. Higher = more stable world understanding.
    pub belief_maturity: f64,

    /// Overall meta-cognitive confidence — composite of all signals.
    /// ∈ [0.0, 1.0]. Below threshold → system should be cautious.
    pub overall_confidence: f64,

    /// Per-signal breakdown for auditability.
    pub signal_details: Vec<SignalDetail>,

    /// Coverage gaps detected — blind spots where the system lacks data.
    pub coverage_gaps: Vec<CoverageGap>,

    /// When the assessment was made (unix seconds).
    pub assessed_at: f64,
}

/// A single meta-cognitive signal with its value and weight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalDetail {
    pub name: String,
    pub value: f64,
    pub weight: f64,
    /// How this value contributes to overall confidence.
    /// Positive = increases confidence, negative = decreases.
    pub contribution: f64,
    pub status: SignalStatus,
}

/// Health status of a meta-cognitive signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalStatus {
    /// Signal is healthy, no concerns.
    Healthy,
    /// Signal is degraded, warrants attention.
    Warning,
    /// Signal is critically low, may need intervention.
    Critical,
}

/// A specific area where the system lacks coverage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageGap {
    pub kind: CoverageGapKind,
    pub description: String,
    /// How severe this gap is ∈ [0.0, 1.0].
    pub severity: f64,
}

/// Types of coverage gaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoverageGapKind {
    /// No observations for this (state, action) combination.
    UnexploredStateAction,
    /// An event kind has never been observed.
    UnobservedEventKind,
    /// A node kind has too few instances in the graph.
    SparseNodeKind,
    /// A belief category has no established beliefs.
    WeakBeliefDomain,
    /// Skills in this category have low confidence.
    UnreliableSkillDomain,
}

// ── §2: Abstain Decision ───────────────────────────────────────────

/// Decision about whether to proceed with autonomous action or defer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstainDecision {
    pub action: AbstainAction,
    /// Reasons contributing to this decision.
    pub reasons: Vec<AbstainReason>,
    /// The meta-cognitive confidence at decision time.
    pub meta_confidence: f64,
}

/// What action to take when the system is uncertain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AbstainAction {
    /// Proceed with the autonomous action.
    Proceed,
    /// Wait and gather more evidence before acting.
    Wait,
    /// Escalate to an LLM for higher-quality reasoning.
    EscalateToLlm,
    /// Ask the user for clarification.
    AskClarification,
    /// Defer this decision entirely — too uncertain.
    Defer,
}

/// A reason contributing to an abstain decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstainReason {
    pub signal: String,
    pub description: String,
    pub severity: f64,
}

// ── §3: Reasoning Health ───────────────────────────────────────────

/// Overall reasoning health report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningHealthReport {
    /// Grade: A (>0.8), B (>0.6), C (>0.4), D (>0.2), F (≤0.2).
    pub grade: char,
    /// ∈ [0.0, 1.0] overall reasoning health.
    pub health_score: f64,
    /// Specific subsystem health assessments.
    pub subsystem_health: Vec<SubsystemHealth>,
    /// Actionable recommendations for improvement.
    pub recommendations: Vec<Recommendation>,
}

/// Health of a specific reasoning subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsystemHealth {
    pub name: String,
    pub score: f64,
    pub status: SignalStatus,
    pub detail: String,
}

/// A recommendation for improving reasoning quality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub priority: RecommendationPriority,
    pub category: String,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

// ── §4: Confidence Report ──────────────────────────────────────────

/// Detailed confidence and calibration report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceReport {
    /// Expected Calibration Error ∈ [0.0, 1.0]. Lower = better.
    pub calibration_error: f64,
    /// Rolling prediction accuracy ∈ [0.0, 1.0].
    pub prediction_accuracy: f64,
    /// Number of predictions in the accuracy window.
    pub prediction_count: u64,
    /// Per-bin calibration details.
    pub bin_details: Vec<CalibrationBinDetail>,
    /// Per-source reliability scores.
    pub source_reliabilities: Vec<SourceReliabilityDetail>,
    /// Coverage statistics.
    pub coverage_gaps: Vec<CoverageGap>,
}

/// Calibration detail for a single confidence bin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationBinDetail {
    /// Predicted confidence range, e.g., "0.70-0.80".
    pub range: String,
    /// Average predicted confidence in this bin.
    pub predicted: f64,
    /// Actual success rate in this bin.
    pub actual: f64,
    /// Number of predictions in this bin.
    pub count: u64,
    /// Gap between predicted and actual.
    pub gap: f64,
}

/// Reliability score for a single evidence source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceReliabilityDetail {
    pub source: String,
    pub reliability: f64,
    pub observation_count: u64,
}

// ── §5: Meta-Cognitive Configuration ───────────────────────────────

/// Configuration for meta-cognitive thresholds and weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaCognitiveConfig {
    // ── Signal weights (sum to 1.0) ──
    /// Weight for evidence sparsity in overall confidence.
    pub w_evidence: f64,
    /// Weight for model agreement.
    pub w_agreement: f64,
    /// Weight for contradiction density.
    pub w_contradiction: f64,
    /// Weight for prediction accuracy.
    pub w_accuracy: f64,
    /// Weight for calibration quality.
    pub w_calibration: f64,
    /// Weight for coverage.
    pub w_coverage: f64,

    // ── Abstain thresholds ──
    /// Evidence sparsity above this → escalate.
    pub sparsity_escalate: f64,
    /// Model disagreement above this → ask clarification.
    pub disagreement_clarify: f64,
    /// Prediction accuracy below this → escalate.
    pub accuracy_escalate: f64,
    /// Minimum candidate confidence to proceed.
    pub min_candidate_confidence: f64,
    /// Overall meta confidence below this → defer.
    pub defer_threshold: f64,

    // ── Warning/critical thresholds ──
    /// Signal value below this is "warning".
    pub warning_threshold: f64,
    /// Signal value below this is "critical".
    pub critical_threshold: f64,

    // ── Coverage ──
    /// Minimum observations per (state, action) pair to count as covered.
    pub min_coverage_observations: u64,
    /// Minimum beliefs per category to count as covered.
    pub min_beliefs_per_category: usize,
}

impl Default for MetaCognitiveConfig {
    fn default() -> Self {
        Self {
            w_evidence: 0.20,
            w_agreement: 0.15,
            w_contradiction: 0.10,
            w_accuracy: 0.25,
            w_calibration: 0.15,
            w_coverage: 0.15,

            sparsity_escalate: 0.8,
            disagreement_clarify: 0.7,
            accuracy_escalate: 0.5,
            min_candidate_confidence: 0.4,
            defer_threshold: 0.25,

            warning_threshold: 0.5,
            critical_threshold: 0.3,

            min_coverage_observations: 5,
            min_beliefs_per_category: 2,
        }
    }
}

// ── §6: Meta-Cognitive Inputs ──────────────────────────────────────

/// Everything needed for a meta-cognitive assessment.
pub struct MetaCognitiveInputs<'a> {
    pub learning_state: &'a LearningState,
    pub belief_store: &'a BeliefStore,
    pub event_buffer: &'a EventBuffer,
    pub skill_registry: &'a SkillRegistry,
    pub experiment_registry: &'a ExperimentRegistry,
    pub transition_model: &'a TransitionModel,
    pub config: &'a MetaCognitiveConfig,
    pub now: f64,
}

// ── §7: Core Assessment ────────────────────────────────────────────

/// Perform a comprehensive meta-cognitive assessment.
///
/// Analyzes all subsystems to determine how reliable the system's
/// reasoning currently is. Returns a detailed report with per-signal
/// breakdowns and coverage gap analysis.
pub fn metacognitive_assessment(inputs: &MetaCognitiveInputs) -> MetaCognitiveReport {
    let config = inputs.config;

    // 1. Evidence sparsity.
    let evidence_sparsity = compute_evidence_sparsity(inputs);

    // 2. Model disagreement.
    let model_disagreement = compute_model_disagreement(inputs);

    // 3. Contradiction density.
    let contradiction_density = compute_contradiction_density(inputs);

    // 4. Prediction accuracy (invert: high accuracy → low sparsity).
    let prediction_accuracy = compute_prediction_accuracy(inputs);

    // 5. Calibration error.
    let calibration_error = compute_calibration_error(inputs);

    // 6. Coverage.
    let coverage = compute_coverage(inputs);

    // 7. Source reliability.
    let source_reliability = compute_source_reliability(inputs);

    // 8. Skill maturity.
    let skill_maturity = compute_skill_maturity(inputs);

    // 9. Belief maturity.
    let belief_maturity = compute_belief_maturity(inputs);

    // 10. Coverage gaps.
    let coverage_gaps = detect_coverage_gaps(inputs);

    // Compute signal details.
    let signals = vec![
        make_signal(
            "evidence",
            1.0 - evidence_sparsity,
            config.w_evidence,
            config,
        ),
        make_signal(
            "agreement",
            1.0 - model_disagreement,
            config.w_agreement,
            config,
        ),
        make_signal(
            "consistency",
            1.0 - contradiction_density,
            config.w_contradiction,
            config,
        ),
        make_signal("accuracy", prediction_accuracy, config.w_accuracy, config),
        make_signal(
            "calibration",
            1.0 - calibration_error,
            config.w_calibration,
            config,
        ),
        make_signal("coverage", coverage, config.w_coverage, config),
    ];

    // Weighted composite confidence.
    let overall_confidence: f64 = signals
        .iter()
        .map(|s| s.contribution)
        .sum::<f64>()
        .clamp(0.0, 1.0);

    MetaCognitiveReport {
        evidence_sparsity,
        model_disagreement,
        contradiction_density,
        prediction_accuracy,
        calibration_error,
        coverage,
        source_reliability,
        skill_maturity,
        belief_maturity,
        overall_confidence,
        signal_details: signals,
        coverage_gaps,
        assessed_at: inputs.now,
    }
}

// ── §8: Abstain Decision Logic ─────────────────────────────────────

/// An action candidate being evaluated for abstain logic.
pub struct MetaActionCandidate {
    pub description: String,
    pub confidence: f64,
}

/// Determine whether the system should proceed or defer.
///
/// Evaluates the meta-cognitive state against thresholds and
/// candidate action confidences. Returns a decision with reasons.
pub fn should_abstain(
    report: &MetaCognitiveReport,
    candidates: &[MetaActionCandidate],
    config: &MetaCognitiveConfig,
) -> AbstainDecision {
    let mut reasons = Vec::new();

    // Check evidence sparsity.
    if report.evidence_sparsity > config.sparsity_escalate {
        reasons.push(AbstainReason {
            signal: "evidence_sparsity".to_string(),
            description: format!(
                "Evidence sparsity {:.2} exceeds threshold {:.2} — novel situation with insufficient data",
                report.evidence_sparsity, config.sparsity_escalate
            ),
            severity: report.evidence_sparsity,
        });
    }

    // Check model disagreement.
    if report.model_disagreement > config.disagreement_clarify {
        reasons.push(AbstainReason {
            signal: "model_disagreement".to_string(),
            description: format!(
                "Model disagreement {:.2} exceeds threshold {:.2} — internal subsystems conflict",
                report.model_disagreement, config.disagreement_clarify
            ),
            severity: report.model_disagreement,
        });
    }

    // Check prediction accuracy.
    if report.prediction_accuracy < config.accuracy_escalate {
        reasons.push(AbstainReason {
            signal: "prediction_accuracy".to_string(),
            description: format!(
                "Prediction accuracy {:.2} below threshold {:.2} — recent predictions unreliable",
                report.prediction_accuracy, config.accuracy_escalate
            ),
            severity: 1.0 - report.prediction_accuracy,
        });
    }

    // Check candidate confidence.
    let all_low = !candidates.is_empty()
        && candidates
            .iter()
            .all(|c| c.confidence < config.min_candidate_confidence);
    if all_low {
        let max_conf = candidates
            .iter()
            .map(|c| c.confidence)
            .fold(0.0_f64, f64::max);
        reasons.push(AbstainReason {
            signal: "candidate_confidence".to_string(),
            description: format!(
                "All {} candidates below confidence threshold {:.2} (max: {:.2})",
                candidates.len(),
                config.min_candidate_confidence,
                max_conf
            ),
            severity: config.min_candidate_confidence - max_conf,
        });
    }

    // Determine action based on reasons.
    let action = if reasons.is_empty() {
        AbstainAction::Proceed
    } else if report.overall_confidence < config.defer_threshold {
        // Extremely low confidence → full defer.
        AbstainAction::Defer
    } else if report.evidence_sparsity > config.sparsity_escalate {
        AbstainAction::EscalateToLlm
    } else if report.model_disagreement > config.disagreement_clarify {
        AbstainAction::AskClarification
    } else if report.prediction_accuracy < config.accuracy_escalate {
        AbstainAction::EscalateToLlm
    } else if all_low {
        AbstainAction::Wait
    } else {
        // Multiple mild concerns but none critical → proceed with caution.
        AbstainAction::Proceed
    };

    AbstainDecision {
        action,
        reasons,
        meta_confidence: report.overall_confidence,
    }
}

// ── §9: Confidence Report ──────────────────────────────────────────

/// Generate a detailed confidence and calibration report.
pub fn confidence_report(inputs: &MetaCognitiveInputs) -> ConfidenceReport {
    let cal = &inputs.learning_state.calibration;
    let calibration_error = cal.calibration_error();
    let prediction_accuracy = compute_prediction_accuracy(inputs);

    // Per-bin calibration details.
    let bin_details = extract_calibration_bins(cal);

    // Source reliabilities.
    let source_reliabilities = extract_source_reliabilities(&inputs.learning_state.reliability);

    // Coverage gaps.
    let coverage_gaps = detect_coverage_gaps(inputs);

    ConfidenceReport {
        calibration_error,
        prediction_accuracy,
        prediction_count: inputs.transition_model.total_transitions,
        bin_details,
        source_reliabilities,
        coverage_gaps,
    }
}

// ── §10: Reasoning Health ──────────────────────────────────────────

/// Produce a reasoning health report with grade and recommendations.
pub fn reasoning_health(inputs: &MetaCognitiveInputs) -> ReasoningHealthReport {
    let report = metacognitive_assessment(inputs);

    // Subsystem assessments.
    let mut subsystems = Vec::new();

    // Calibration subsystem.
    let cal_score = 1.0 - report.calibration_error;
    subsystems.push(SubsystemHealth {
        name: "Calibration".to_string(),
        score: cal_score,
        status: classify_signal(cal_score, inputs.config),
        detail: format!(
            "ECE: {:.3}, {} total predictions",
            report.calibration_error, inputs.learning_state.calibration.total
        ),
    });

    // World model subsystem.
    let wm_score = report.prediction_accuracy;
    subsystems.push(SubsystemHealth {
        name: "World Model".to_string(),
        score: wm_score,
        status: classify_signal(wm_score, inputs.config),
        detail: format!(
            "{} transitions, {:.1}% accuracy",
            inputs.transition_model.total_transitions,
            wm_score * 100.0
        ),
    });

    // Belief system.
    let belief_score = report.belief_maturity;
    subsystems.push(SubsystemHealth {
        name: "Belief System".to_string(),
        score: belief_score,
        status: classify_signal(belief_score, inputs.config),
        detail: format!(
            "{} formed, {:.0}% established",
            inputs.belief_store.total_formed,
            belief_score * 100.0
        ),
    });

    // Skill system.
    let skill_score = report.skill_maturity;
    subsystems.push(SubsystemHealth {
        name: "Skill System".to_string(),
        score: skill_score,
        status: classify_signal(skill_score, inputs.config),
        detail: format!(
            "{} skills, {:.0}% mature",
            inputs.skill_registry.skills.len(),
            skill_score * 100.0
        ),
    });

    // Evidence collection.
    let evidence_score = 1.0 - report.evidence_sparsity;
    subsystems.push(SubsystemHealth {
        name: "Evidence Collection".to_string(),
        score: evidence_score,
        status: classify_signal(evidence_score, inputs.config),
        detail: format!("{} events ingested", inputs.event_buffer.total_ingested),
    });

    // Experimentation.
    let exp_score = compute_experiment_health(inputs);
    subsystems.push(SubsystemHealth {
        name: "Experimentation".to_string(),
        score: exp_score,
        status: classify_signal(exp_score, inputs.config),
        detail: format!(
            "{} concluded, {} aborted",
            inputs.experiment_registry.total_concluded, inputs.experiment_registry.total_aborted
        ),
    });

    // Overall health = mean of subsystem scores.
    let health_score = if subsystems.is_empty() {
        0.5
    } else {
        subsystems.iter().map(|s| s.score).sum::<f64>() / subsystems.len() as f64
    };

    let grade = score_to_grade(health_score);

    // Generate recommendations.
    let recommendations = generate_recommendations(&report, &subsystems, inputs);

    ReasoningHealthReport {
        grade,
        health_score,
        subsystem_health: subsystems,
        recommendations,
    }
}

// ── §11: Meta-Cognitive History ────────────────────────────────────

/// A timestamped snapshot of meta-cognitive state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaCognitiveSnapshot {
    pub timestamp: f64,
    pub overall_confidence: f64,
    pub evidence_sparsity: f64,
    pub prediction_accuracy: f64,
    pub calibration_error: f64,
    pub coverage: f64,
    pub abstain_action: String,
}

/// Rolling history of meta-cognitive assessments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaCognitiveHistory {
    snapshots: Vec<MetaCognitiveSnapshot>,
    max_snapshots: usize,
    pub total_assessments: u64,
    pub total_escalations: u64,
    pub total_deferrals: u64,
    pub total_proceeds: u64,
}

impl MetaCognitiveHistory {
    pub fn new(max_snapshots: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            max_snapshots,
            total_assessments: 0,
            total_escalations: 0,
            total_deferrals: 0,
            total_proceeds: 0,
        }
    }

    /// Record a meta-cognitive assessment.
    pub fn record(&mut self, report: &MetaCognitiveReport, decision: &AbstainDecision) {
        self.total_assessments += 1;
        match &decision.action {
            AbstainAction::Proceed => self.total_proceeds += 1,
            AbstainAction::EscalateToLlm => self.total_escalations += 1,
            AbstainAction::Defer => self.total_deferrals += 1,
            _ => {}
        }

        self.snapshots.push(MetaCognitiveSnapshot {
            timestamp: report.assessed_at,
            overall_confidence: report.overall_confidence,
            evidence_sparsity: report.evidence_sparsity,
            prediction_accuracy: report.prediction_accuracy,
            calibration_error: report.calibration_error,
            coverage: report.coverage,
            abstain_action: format!("{:?}", decision.action),
        });

        if self.snapshots.len() > self.max_snapshots {
            self.snapshots.remove(0);
        }
    }

    /// Average meta-cognitive confidence over last N snapshots.
    pub fn recent_confidence(&self, n: usize) -> f64 {
        let recent: Vec<f64> = self
            .snapshots
            .iter()
            .rev()
            .take(n)
            .map(|s| s.overall_confidence)
            .collect();
        if recent.is_empty() {
            return 0.5;
        }
        recent.iter().sum::<f64>() / recent.len() as f64
    }

    /// Confidence trend over last N snapshots (positive = improving).
    pub fn confidence_trend(&self, n: usize) -> f64 {
        let scores: Vec<f64> = self
            .snapshots
            .iter()
            .rev()
            .take(n)
            .map(|s| s.overall_confidence)
            .collect();
        if scores.len() < 2 {
            return 0.0;
        }
        let first = scores.last().unwrap();
        let last = scores.first().unwrap();
        (last - first) / (scores.len() as f64 - 1.0)
    }

    /// Escalation rate over last N assessments.
    pub fn escalation_rate(&self, n: usize) -> f64 {
        let recent: Vec<&MetaCognitiveSnapshot> = self.snapshots.iter().rev().take(n).collect();
        if recent.is_empty() {
            return 0.0;
        }
        let escalations = recent
            .iter()
            .filter(|s| s.abstain_action == "EscalateToLlm")
            .count();
        escalations as f64 / recent.len() as f64
    }

    pub fn latest(&self) -> Option<&MetaCognitiveSnapshot> {
        self.snapshots.last()
    }

    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }
}

// ── §12: Internal — Evidence Sparsity ──────────────────────────────

/// Compute evidence sparsity ∈ [0.0, 1.0].
///
/// Based on total observations, event diversity, and data recency.
/// A system that has seen many diverse events recently has low sparsity.
fn compute_evidence_sparsity(inputs: &MetaCognitiveInputs) -> f64 {
    let total = inputs.event_buffer.total_ingested;

    // Volume component: approaches 0 as observations grow.
    // sigmoid: 1 / (1 + total/500)
    let volume_sparsity = 1.0 / (1.0 + total as f64 / 500.0);

    // Diversity component: how many event kinds have been observed?
    let all_kinds = [
        EventKind::AppOpened,
        EventKind::AppClosed,
        EventKind::AppSequence,
        EventKind::NotificationReceived,
        EventKind::NotificationDismissed,
        EventKind::NotificationActedOn,
        EventKind::SuggestionAccepted,
        EventKind::SuggestionRejected,
        EventKind::SuggestionIgnored,
        EventKind::SuggestionModified,
        EventKind::QueryRepeated,
        EventKind::UserTyping,
        EventKind::UserIdle,
        EventKind::ToolCallCompleted,
        EventKind::LlmCalled,
        EventKind::ErrorOccurred,
    ];
    let observed_kinds = all_kinds
        .iter()
        .filter(|k| inputs.event_buffer.by_kind(**k, 1).len() > 0)
        .count();
    let diversity = observed_kinds as f64 / all_kinds.len() as f64;
    let diversity_sparsity = 1.0 - diversity;

    // Belief grounding: how many beliefs have ≥3 observations?
    let grounded_beliefs = inputs
        .belief_store
        .iter()
        .filter(|b| b.confirming_observations + b.contradicting_observations >= 3)
        .count();
    let total_beliefs = inputs.belief_store.len().max(1);
    let grounding_sparsity = 1.0 - (grounded_beliefs as f64 / total_beliefs as f64);

    // Weighted average.
    (0.4 * volume_sparsity + 0.3 * diversity_sparsity + 0.3 * grounding_sparsity).clamp(0.0, 1.0)
}

// ── §13: Internal — Model Disagreement ─────────────────────────────

/// Compute model disagreement ∈ [0.0, 1.0].
///
/// Measures how much different internal models disagree.
/// Based on experiment variance, outcome distribution entropy,
/// and calibration-vs-accuracy gap.
fn compute_model_disagreement(inputs: &MetaCognitiveInputs) -> f64 {
    let mut disagreement_signals = Vec::new();

    // 1. Experiment ambiguity: active experiments without clear winners.
    let active = inputs.experiment_registry.active_experiments();
    if !active.is_empty() {
        let ambiguous = active
            .iter()
            .filter(|e| {
                // An experiment is ambiguous if no variant has >70% posterior.
                e.variant_results.iter().all(|v| v.mean() < 0.7)
            })
            .count();
        disagreement_signals.push(ambiguous as f64 / active.len() as f64);
    }

    // 2. Global outcome entropy (normalized to [0, 1]).
    let max_entropy = (5.0_f64).ln(); // 5 outcome types → max entropy = ln(5).
    let global_entropy = inputs.transition_model.global_outcomes.entropy();
    if max_entropy > 0.0 {
        disagreement_signals.push((global_entropy / max_entropy).clamp(0.0, 1.0));
    }

    // 3. Calibration-vs-accuracy gap.
    let cal_error = inputs.learning_state.calibration.calibration_error();
    let accuracy = inputs.learning_state.weights.accuracy();
    // If calibration and accuracy diverge significantly, models disagree.
    let gap = (cal_error - (1.0 - accuracy)).abs();
    disagreement_signals.push(gap.clamp(0.0, 1.0));

    if disagreement_signals.is_empty() {
        return 0.5; // No data → moderate uncertainty.
    }

    disagreement_signals.iter().sum::<f64>() / disagreement_signals.len() as f64
}

// ── §14: Internal — Contradiction Density ──────────────────────────

/// Compute contradiction density ∈ [0.0, 1.0].
///
/// Ratio of beliefs with contradicting observations to total beliefs.
fn compute_contradiction_density(inputs: &MetaCognitiveInputs) -> f64 {
    let total = inputs.belief_store.len();
    if total == 0 {
        return 0.0; // No beliefs → no contradictions.
    }

    let contradicted = inputs
        .belief_store
        .iter()
        .filter(|b| {
            b.contradicting_observations > 0
                && b.contradicting_observations as f64
                    / (b.confirming_observations + b.contradicting_observations).max(1) as f64
                    > 0.3
        })
        .count();

    (contradicted as f64 / total as f64).clamp(0.0, 1.0)
}

// ── §15: Internal — Prediction Accuracy ────────────────────────────

/// Compute prediction accuracy ∈ [0.0, 1.0].
///
/// Blends world model prediction accuracy with learning state accuracy.
fn compute_prediction_accuracy(inputs: &MetaCognitiveInputs) -> f64 {
    let wm_accuracy = inputs
        .transition_model
        .prediction_accuracy(inputs.config.min_coverage_observations as u32);
    let learning_accuracy = inputs.learning_state.weights.accuracy();

    // If world model has few data points, rely more on learning accuracy.
    if inputs.transition_model.total_transitions < 10 {
        learning_accuracy
    } else {
        0.6 * wm_accuracy + 0.4 * learning_accuracy
    }
}

// ── §16: Internal — Calibration Error ──────────────────────────────

/// Compute calibration error ∈ [0.0, 1.0].
fn compute_calibration_error(inputs: &MetaCognitiveInputs) -> f64 {
    inputs.learning_state.calibration.calibration_error()
}

// ── §17: Internal — Coverage ───────────────────────────────────────

/// Compute coverage ∈ [0.0, 1.0].
///
/// How much of the possible (state, action) space has been observed?
fn compute_coverage(inputs: &MetaCognitiveInputs) -> f64 {
    let unique_pairs = inputs.transition_model.unique_pairs();
    // There are 6 time bins × 4 receptivity × 3 session × 3 error × 3 goals × 7 actions = 4536 possible.
    // But in practice many are impossible. Use a realistic denominator.
    let realistic_space = 200.0; // Typical system sees ~200 distinct state-action pairs.
    let coverage_ratio = (unique_pairs as f64 / realistic_space).clamp(0.0, 1.0);

    // Also consider skill coverage.
    let total_skills = inputs.skill_registry.skills.len();
    let reliable_skills = inputs
        .skill_registry
        .skills
        .values()
        .filter(|s| s.confidence >= 0.6 && !s.deprecated)
        .count();
    let skill_coverage = if total_skills == 0 {
        0.0
    } else {
        reliable_skills as f64 / total_skills as f64
    };

    0.6 * coverage_ratio + 0.4 * skill_coverage
}

// ── §18: Internal — Source Reliability ──────────────────────────────

/// Compute aggregate source reliability ∈ [0.0, 1.0].
fn compute_source_reliability(inputs: &MetaCognitiveInputs) -> f64 {
    let sources = &inputs.learning_state.reliability.sources;
    if sources.is_empty() {
        return 0.5; // No data → moderate assumption.
    }

    let total_weight: f64 = sources.values().map(|s| s.total as f64).sum();
    if total_weight < 1.0 {
        return 0.5;
    }

    // Weighted average by observation count.
    let weighted_sum: f64 = sources
        .values()
        .map(|s| s.reliability() * s.total as f64)
        .sum();

    (weighted_sum / total_weight).clamp(0.0, 1.0)
}

// ── §19: Internal — Skill Maturity ─────────────────────────────────

/// Compute skill maturity ∈ [0.0, 1.0].
///
/// Proportion of non-deprecated skills that are Validated or better
/// (confidence ≥ 0.5).
fn compute_skill_maturity(inputs: &MetaCognitiveInputs) -> f64 {
    let active_skills: Vec<_> = inputs
        .skill_registry
        .skills
        .values()
        .filter(|s| !s.deprecated)
        .collect();

    if active_skills.is_empty() {
        return 0.0;
    }

    let mature = active_skills.iter().filter(|s| s.confidence >= 0.5).count();

    mature as f64 / active_skills.len() as f64
}

// ── §20: Internal — Belief Maturity ────────────────────────────────

/// Compute belief maturity ∈ [0.0, 1.0].
///
/// Proportion of beliefs that are Established or Certain.
fn compute_belief_maturity(inputs: &MetaCognitiveInputs) -> f64 {
    let total = inputs.belief_store.len();
    if total == 0 {
        return 0.0;
    }

    let established = inputs
        .belief_store
        .iter()
        .filter(|b| matches!(b.stage, BeliefStage::Established | BeliefStage::Certain))
        .count();

    established as f64 / total as f64
}

// ── §21: Internal — Coverage Gap Detection ─────────────────────────

/// Detect specific coverage gaps in the system's knowledge.
fn detect_coverage_gaps(inputs: &MetaCognitiveInputs) -> Vec<CoverageGap> {
    let mut gaps = Vec::new();

    // 1. Unobserved event kinds.
    let all_kinds = [
        EventKind::AppOpened,
        EventKind::AppClosed,
        EventKind::AppSequence,
        EventKind::NotificationReceived,
        EventKind::NotificationDismissed,
        EventKind::NotificationActedOn,
        EventKind::SuggestionAccepted,
        EventKind::SuggestionRejected,
        EventKind::SuggestionIgnored,
        EventKind::SuggestionModified,
        EventKind::QueryRepeated,
        EventKind::UserTyping,
        EventKind::UserIdle,
        EventKind::ToolCallCompleted,
        EventKind::LlmCalled,
        EventKind::ErrorOccurred,
    ];
    for kind in &all_kinds {
        if inputs.event_buffer.by_kind(*kind, 1).is_empty() {
            gaps.push(CoverageGap {
                kind: CoverageGapKind::UnobservedEventKind,
                description: format!("No {:?} events observed", kind),
                severity: 0.3,
            });
        }
    }

    // 2. Weak belief domains.
    use crate::flywheel::BeliefCategory;
    let categories = [
        BeliefCategory::Temporal,
        BeliefCategory::Preference,
        BeliefCategory::Behavioral,
        BeliefCategory::Productivity,
        BeliefCategory::Need,
        BeliefCategory::System,
    ];
    for cat in &categories {
        let count = inputs
            .belief_store
            .iter()
            .filter(|b| {
                b.category == *cat
                    && matches!(b.stage, BeliefStage::Established | BeliefStage::Certain)
            })
            .count();
        if count < inputs.config.min_beliefs_per_category {
            gaps.push(CoverageGap {
                kind: CoverageGapKind::WeakBeliefDomain,
                description: format!(
                    "{:?} has only {} established beliefs (need {})",
                    cat, count, inputs.config.min_beliefs_per_category
                ),
                severity: 0.5,
            });
        }
    }

    // 3. Low world model coverage.
    let unique = inputs.transition_model.unique_pairs();
    if unique < 10 {
        gaps.push(CoverageGap {
            kind: CoverageGapKind::UnexploredStateAction,
            description: format!("Only {} state-action pairs explored (need ≥10)", unique),
            severity: 0.7,
        });
    }

    // 4. Unreliable skill domains.
    let unreliable: Vec<_> = inputs
        .skill_registry
        .skills
        .values()
        .filter(|s| !s.deprecated && s.confidence < 0.3 && s.offer_count > 3)
        .collect();
    if !unreliable.is_empty() {
        gaps.push(CoverageGap {
            kind: CoverageGapKind::UnreliableSkillDomain,
            description: format!(
                "{} skills with low confidence despite multiple offers",
                unreliable.len()
            ),
            severity: 0.4,
        });
    }

    gaps
}

// ── §22: Internal — Helpers ────────────────────────────────────────

fn make_signal(name: &str, value: f64, weight: f64, config: &MetaCognitiveConfig) -> SignalDetail {
    let contribution = value * weight;
    SignalDetail {
        name: name.to_string(),
        value,
        weight,
        contribution,
        status: classify_signal(value, config),
    }
}

fn classify_signal(value: f64, config: &MetaCognitiveConfig) -> SignalStatus {
    if value < config.critical_threshold {
        SignalStatus::Critical
    } else if value < config.warning_threshold {
        SignalStatus::Warning
    } else {
        SignalStatus::Healthy
    }
}

fn score_to_grade(score: f64) -> char {
    if score > 0.8 {
        'A'
    } else if score > 0.6 {
        'B'
    } else if score > 0.4 {
        'C'
    } else if score > 0.2 {
        'D'
    } else {
        'F'
    }
}

fn compute_experiment_health(inputs: &MetaCognitiveInputs) -> f64 {
    let reg = &inputs.experiment_registry;
    let total = reg.total_concluded + reg.total_aborted;
    if total == 0 {
        return 0.5; // No experiments → neutral.
    }

    // Health = concluded / (concluded + aborted).
    let conclude_rate = reg.total_concluded as f64 / total as f64;

    // Penalize if too many active experiments (overloaded).
    let active_penalty = if reg.active_experiments().len() > reg.max_concurrent {
        0.2
    } else {
        0.0
    };

    (conclude_rate - active_penalty).clamp(0.0, 1.0)
}

fn extract_calibration_bins(cal: &CalibrationMap) -> Vec<CalibrationBinDetail> {
    cal.bins
        .iter()
        .enumerate()
        .map(|(i, bin)| {
            let lo = i as f64 * 0.1;
            let hi = lo + 0.1;
            let predicted = if bin.count > 0 {
                bin.sum_predicted / bin.count as f64
            } else {
                (lo + hi) / 2.0
            };
            let actual = bin.actual_rate();
            CalibrationBinDetail {
                range: format!("{:.1}-{:.1}", lo, hi),
                predicted,
                actual,
                count: bin.count,
                gap: (predicted - actual).abs(),
            }
        })
        .collect()
}

fn extract_source_reliabilities(reg: &ReliabilityRegistry) -> Vec<SourceReliabilityDetail> {
    reg.sources
        .iter()
        .map(|(name, src)| SourceReliabilityDetail {
            source: name.clone(),
            reliability: src.reliability(),
            observation_count: src.total,
        })
        .collect()
}

fn generate_recommendations(
    report: &MetaCognitiveReport,
    subsystems: &[SubsystemHealth],
    inputs: &MetaCognitiveInputs,
) -> Vec<Recommendation> {
    let mut recs = Vec::new();

    // Critical subsystems first.
    for sub in subsystems {
        if sub.status == SignalStatus::Critical {
            recs.push(Recommendation {
                priority: RecommendationPriority::Critical,
                category: sub.name.clone(),
                description: format!(
                    "{} is critically degraded ({:.0}%): {}",
                    sub.name,
                    sub.score * 100.0,
                    sub.detail
                ),
            });
        }
    }

    // Evidence sparsity.
    if report.evidence_sparsity > 0.7 {
        recs.push(Recommendation {
            priority: RecommendationPriority::High,
            category: "Evidence".to_string(),
            description: format!(
                "Evidence sparsity is {:.0}%. Increase user interaction or data collection to reduce uncertainty.",
                report.evidence_sparsity * 100.0
            ),
        });
    }

    // Calibration drift.
    if report.calibration_error > 0.15 {
        recs.push(Recommendation {
            priority: if report.calibration_error > 0.3 {
                RecommendationPriority::High
            } else {
                RecommendationPriority::Medium
            },
            category: "Calibration".to_string(),
            description: format!(
                "Calibration error is {:.1}%. Confidence estimates may be unreliable. Consider recalibration.",
                report.calibration_error * 100.0
            ),
        });
    }

    // Low coverage.
    if report.coverage < 0.3 {
        recs.push(Recommendation {
            priority: RecommendationPriority::Medium,
            category: "Coverage".to_string(),
            description: format!(
                "Only {:.0}% state-action coverage. System has significant blind spots.",
                report.coverage * 100.0
            ),
        });
    }

    // Experiment overload.
    if inputs.experiment_registry.active_experiments().len()
        > inputs.experiment_registry.max_concurrent
    {
        recs.push(Recommendation {
            priority: RecommendationPriority::Medium,
            category: "Experimentation".to_string(),
            description: "Too many concurrent experiments. Consider concluding or aborting some."
                .to_string(),
        });
    }

    // Low skill maturity.
    if report.skill_maturity < 0.3 && inputs.skill_registry.skills.len() > 3 {
        recs.push(Recommendation {
            priority: RecommendationPriority::Low,
            category: "Skills".to_string(),
            description: format!(
                "Only {:.0}% of skills are mature. More usage data needed to validate skill patterns.",
                report.skill_maturity * 100.0
            ),
        });
    }

    // Sort by priority (Critical first).
    recs.sort_by(|a, b| b.priority.cmp(&a.priority));
    recs
}

// ── §23: Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{
        BanditRegistry, CalibrationBin, CalibrationMap, LearningState, ReliabilityRegistry,
        SourceReliability, UtilityWeights,
    };
    use crate::experimenter::{BetaPosterior, Experiment, ExperimentRegistry, ExperimentStatus};
    use crate::flywheel::{
        AutonomousBelief, BeliefCategory, BeliefEvidence, BeliefStage, BeliefStore,
    };
    use crate::observer::{EventBuffer, EventKind};
    use crate::skills::{LearnedSkill, SkillOrigin, SkillRegistry, SkillStage};
    use crate::world_model::{OutcomeDistribution, TransitionModel};

    fn empty_learning_state() -> LearningState {
        LearningState {
            weights: UtilityWeights::default(),
            calibration: CalibrationMap::new(),
            bandits: BanditRegistry::new(),
            reliability: ReliabilityRegistry::new(),
            interaction_buffer: Vec::new(),
            config: crate::calibration::LearningConfig::default(),
            total_interactions: 0,
            last_weight_refit: 0.0,
            last_calibration_refit: 0.0,
        }
    }

    fn empty_inputs<'a>(
        ls: &'a LearningState,
        bs: &'a BeliefStore,
        eb: &'a EventBuffer,
        sr: &'a SkillRegistry,
        er: &'a ExperimentRegistry,
        tm: &'a TransitionModel,
        config: &'a MetaCognitiveConfig,
    ) -> MetaCognitiveInputs<'a> {
        MetaCognitiveInputs {
            learning_state: ls,
            belief_store: bs,
            event_buffer: eb,
            skill_registry: sr,
            experiment_registry: er,
            transition_model: tm,
            config,
            now: 1_000_000.0,
        }
    }

    #[test]
    fn test_empty_assessment() {
        let ls = empty_learning_state();
        let bs = BeliefStore::new();
        let eb = EventBuffer::new(1000);
        let sr = SkillRegistry::new();
        let er = ExperimentRegistry::new();
        let tm = TransitionModel::new();
        let config = MetaCognitiveConfig::default();

        let inputs = empty_inputs(&ls, &bs, &eb, &sr, &er, &tm, &config);
        let report = metacognitive_assessment(&inputs);

        // Empty system should have high sparsity.
        assert!(report.evidence_sparsity > 0.5);
        // Should have low belief/skill maturity.
        assert_eq!(report.belief_maturity, 0.0);
        assert_eq!(report.skill_maturity, 0.0);
        // Overall confidence should be moderate-to-low.
        assert!(report.overall_confidence < 0.7);
    }

    #[test]
    fn test_should_abstain_proceed() {
        let report = MetaCognitiveReport {
            evidence_sparsity: 0.2,
            model_disagreement: 0.1,
            contradiction_density: 0.0,
            prediction_accuracy: 0.8,
            calibration_error: 0.05,
            coverage: 0.7,
            source_reliability: 0.9,
            skill_maturity: 0.8,
            belief_maturity: 0.7,
            overall_confidence: 0.85,
            signal_details: Vec::new(),
            coverage_gaps: Vec::new(),
            assessed_at: 1000.0,
        };

        let candidates = vec![MetaActionCandidate {
            description: "Send notification".to_string(),
            confidence: 0.8,
        }];
        let config = MetaCognitiveConfig::default();

        let decision = should_abstain(&report, &candidates, &config);
        assert_eq!(decision.action, AbstainAction::Proceed);
        assert!(decision.reasons.is_empty());
    }

    #[test]
    fn test_should_abstain_escalate_sparsity() {
        let report = MetaCognitiveReport {
            evidence_sparsity: 0.9,
            model_disagreement: 0.1,
            contradiction_density: 0.0,
            prediction_accuracy: 0.7,
            calibration_error: 0.1,
            coverage: 0.5,
            source_reliability: 0.8,
            skill_maturity: 0.5,
            belief_maturity: 0.5,
            overall_confidence: 0.5,
            signal_details: Vec::new(),
            coverage_gaps: Vec::new(),
            assessed_at: 1000.0,
        };

        let candidates = vec![MetaActionCandidate {
            description: "Act".to_string(),
            confidence: 0.7,
        }];
        let config = MetaCognitiveConfig::default();

        let decision = should_abstain(&report, &candidates, &config);
        assert_eq!(decision.action, AbstainAction::EscalateToLlm);
        assert!(!decision.reasons.is_empty());
    }

    #[test]
    fn test_should_abstain_clarify_disagreement() {
        let report = MetaCognitiveReport {
            evidence_sparsity: 0.3,
            model_disagreement: 0.85,
            contradiction_density: 0.2,
            prediction_accuracy: 0.7,
            calibration_error: 0.1,
            coverage: 0.6,
            source_reliability: 0.8,
            skill_maturity: 0.6,
            belief_maturity: 0.5,
            overall_confidence: 0.5,
            signal_details: Vec::new(),
            coverage_gaps: Vec::new(),
            assessed_at: 1000.0,
        };

        let candidates = vec![MetaActionCandidate {
            description: "Act".to_string(),
            confidence: 0.6,
        }];
        let config = MetaCognitiveConfig::default();

        let decision = should_abstain(&report, &candidates, &config);
        assert_eq!(decision.action, AbstainAction::AskClarification);
    }

    #[test]
    fn test_should_abstain_wait_low_candidates() {
        let report = MetaCognitiveReport {
            evidence_sparsity: 0.3,
            model_disagreement: 0.2,
            contradiction_density: 0.0,
            prediction_accuracy: 0.7,
            calibration_error: 0.1,
            coverage: 0.6,
            source_reliability: 0.8,
            skill_maturity: 0.6,
            belief_maturity: 0.5,
            overall_confidence: 0.7,
            signal_details: Vec::new(),
            coverage_gaps: Vec::new(),
            assessed_at: 1000.0,
        };

        let candidates = vec![
            MetaActionCandidate {
                description: "A".to_string(),
                confidence: 0.2,
            },
            MetaActionCandidate {
                description: "B".to_string(),
                confidence: 0.1,
            },
        ];
        let config = MetaCognitiveConfig::default();

        let decision = should_abstain(&report, &candidates, &config);
        assert_eq!(decision.action, AbstainAction::Wait);
    }

    #[test]
    fn test_should_abstain_defer_extreme() {
        let report = MetaCognitiveReport {
            evidence_sparsity: 0.95,
            model_disagreement: 0.9,
            contradiction_density: 0.8,
            prediction_accuracy: 0.2,
            calibration_error: 0.5,
            coverage: 0.1,
            source_reliability: 0.3,
            skill_maturity: 0.1,
            belief_maturity: 0.1,
            overall_confidence: 0.15,
            signal_details: Vec::new(),
            coverage_gaps: Vec::new(),
            assessed_at: 1000.0,
        };

        let candidates = vec![MetaActionCandidate {
            description: "Act".to_string(),
            confidence: 0.3,
        }];
        let config = MetaCognitiveConfig::default();

        let decision = should_abstain(&report, &candidates, &config);
        assert_eq!(decision.action, AbstainAction::Defer);
    }

    #[test]
    fn test_reasoning_health_empty() {
        let ls = empty_learning_state();
        let bs = BeliefStore::new();
        let eb = EventBuffer::new(1000);
        let sr = SkillRegistry::new();
        let er = ExperimentRegistry::new();
        let tm = TransitionModel::new();
        let config = MetaCognitiveConfig::default();

        let inputs = empty_inputs(&ls, &bs, &eb, &sr, &er, &tm, &config);
        let health = reasoning_health(&inputs);

        // Empty system should have low health.
        assert!(health.health_score < 0.7);
        assert!(!health.subsystem_health.is_empty());
    }

    #[test]
    fn test_confidence_report_empty() {
        let ls = empty_learning_state();
        let bs = BeliefStore::new();
        let eb = EventBuffer::new(1000);
        let sr = SkillRegistry::new();
        let er = ExperimentRegistry::new();
        let tm = TransitionModel::new();
        let config = MetaCognitiveConfig::default();

        let inputs = empty_inputs(&ls, &bs, &eb, &sr, &er, &tm, &config);
        let conf = confidence_report(&inputs);

        assert_eq!(conf.bin_details.len(), 10);
        assert!(conf.prediction_count == 0 || conf.prediction_accuracy >= 0.0);
    }

    #[test]
    fn test_evidence_sparsity_decreases_with_data() {
        let ls = empty_learning_state();
        let mut bs = BeliefStore::new();
        let sr = SkillRegistry::new();
        let er = ExperimentRegistry::new();
        let tm = TransitionModel::new();
        let config = MetaCognitiveConfig::default();

        // Empty buffer → high sparsity.
        let eb1 = EventBuffer::new(1000);
        let inputs1 = empty_inputs(&ls, &bs, &eb1, &sr, &er, &tm, &config);
        let sparsity1 = compute_evidence_sparsity(&inputs1);

        // Buffer with many events → lower sparsity.
        let mut eb2 = EventBuffer::new(10000);
        // Simulate ingestion.
        eb2.total_ingested = 2000;
        let inputs2 = empty_inputs(&ls, &bs, &eb2, &sr, &er, &tm, &config);
        let sparsity2 = compute_evidence_sparsity(&inputs2);

        assert!(sparsity2 < sparsity1);
    }

    #[test]
    fn test_contradiction_density_with_conflicts() {
        let ls = empty_learning_state();
        let sr = SkillRegistry::new();
        let er = ExperimentRegistry::new();
        let tm = TransitionModel::new();
        let config = MetaCognitiveConfig::default();
        let eb = EventBuffer::new(1000);

        // No contradictions.
        let bs1 = BeliefStore::new();
        let inputs1 = empty_inputs(&ls, &bs1, &eb, &sr, &er, &tm, &config);
        assert_eq!(compute_contradiction_density(&inputs1), 0.0);

        // Add beliefs with contradictions.
        let mut bs2 = BeliefStore::new();
        let mut b1 = AutonomousBelief::new(
            "Morning".to_string(),
            BeliefCategory::Temporal,
            "t:morning".to_string(),
            BeliefEvidence::Temporal {
                event_kind: EventKind::AppOpened,
                peak_hour: 9,
                quiet_hours: vec![],
                distribution_skew: 0.8,
            },
            86400.0 * 100.0,
        );
        b1.confirming_observations = 3;
        b1.contradicting_observations = 5; // >30% contradicting
        bs2.upsert(b1);

        let inputs2 = empty_inputs(&ls, &bs2, &eb, &sr, &er, &tm, &config);
        let density = compute_contradiction_density(&inputs2);
        assert!(density > 0.0);
    }

    #[test]
    fn test_metacognitive_history() {
        let mut history = MetaCognitiveHistory::new(5);

        let report = MetaCognitiveReport {
            evidence_sparsity: 0.3,
            model_disagreement: 0.2,
            contradiction_density: 0.0,
            prediction_accuracy: 0.8,
            calibration_error: 0.1,
            coverage: 0.6,
            source_reliability: 0.8,
            skill_maturity: 0.7,
            belief_maturity: 0.6,
            overall_confidence: 0.75,
            signal_details: Vec::new(),
            coverage_gaps: Vec::new(),
            assessed_at: 1000.0,
        };

        let decision = AbstainDecision {
            action: AbstainAction::Proceed,
            reasons: Vec::new(),
            meta_confidence: 0.75,
        };

        history.record(&report, &decision);
        assert_eq!(history.total_assessments, 1);
        assert_eq!(history.total_proceeds, 1);
        assert_eq!(history.snapshot_count(), 1);

        // Record an escalation.
        let decision2 = AbstainDecision {
            action: AbstainAction::EscalateToLlm,
            reasons: vec![AbstainReason {
                signal: "test".to_string(),
                description: "test".to_string(),
                severity: 0.9,
            }],
            meta_confidence: 0.3,
        };
        history.record(&report, &decision2);
        assert_eq!(history.total_escalations, 1);

        // Test capacity.
        for i in 0..10 {
            history.record(&report, &decision);
        }
        assert_eq!(history.snapshot_count(), 5); // Max retained.
        assert_eq!(history.total_assessments, 12);
    }

    #[test]
    fn test_coverage_gap_detection() {
        let ls = empty_learning_state();
        let bs = BeliefStore::new();
        let eb = EventBuffer::new(1000);
        let sr = SkillRegistry::new();
        let er = ExperimentRegistry::new();
        let tm = TransitionModel::new();
        let config = MetaCognitiveConfig::default();

        let inputs = empty_inputs(&ls, &bs, &eb, &sr, &er, &tm, &config);
        let gaps = detect_coverage_gaps(&inputs);

        // Should detect unobserved event kinds and weak belief domains.
        assert!(!gaps.is_empty());
        assert!(gaps
            .iter()
            .any(|g| g.kind == CoverageGapKind::UnobservedEventKind));
        assert!(gaps
            .iter()
            .any(|g| g.kind == CoverageGapKind::WeakBeliefDomain));
    }

    #[test]
    fn test_score_to_grade() {
        assert_eq!(score_to_grade(0.95), 'A');
        assert_eq!(score_to_grade(0.75), 'B');
        assert_eq!(score_to_grade(0.55), 'C');
        assert_eq!(score_to_grade(0.35), 'D');
        assert_eq!(score_to_grade(0.1), 'F');
    }

    #[test]
    fn test_signal_classification() {
        let config = MetaCognitiveConfig::default();
        assert_eq!(classify_signal(0.8, &config), SignalStatus::Healthy);
        assert_eq!(classify_signal(0.4, &config), SignalStatus::Warning);
        assert_eq!(classify_signal(0.2, &config), SignalStatus::Critical);
    }
}
