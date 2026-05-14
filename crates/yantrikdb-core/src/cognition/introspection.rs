//! Self-Awareness Introspection Report.
//!
//! Aggregates learning state from all cognition subsystems into a unified
//! "what have you learned" report. This is the showstopper feature — the
//! system can explain its own cognitive growth to the user.
//!
//! # Subsystems aggregated
//!
//! 1. **Flywheel** — Autonomous beliefs formed, confirmed, pruned
//! 2. **Observer** — Observation volume and event rates
//! 3. **Skills** — Discovered and promoted skills
//! 4. **Experimenter** — Hypothesis testing results
//! 5. **Extractor** — NLP template learning efficiency
//! 6. **Calibration** — Confidence calibration and weight stability
//! 7. **World Model** — Transition learning and prediction accuracy
//!
//! # Design
//!
//! Pure functions only — no DB access. The engine layer loads each subsystem's
//! state and passes it here for aggregation. This keeps the cognition layer
//! testable without SQLite.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::calibration::{self, LearningState};
use super::experimenter::{ExperimentRegistry, ExperimentStatus};
use super::extractor::{self, TemplateStore};
use super::flywheel::{AutonomousBelief, BeliefStage, BeliefStore};
use super::observer::EventBuffer;
use super::skills::{self, SkillRegistry, SkillStage};
use super::world_model::{self, TransitionModel};

// ══════════════════════════════════════════════════════════════════════════════
// § 1  Core Report Types
// ══════════════════════════════════════════════════════════════════════════════

/// Comprehensive introspection report — everything the system has learned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrospectionReport {
    // ── Volume ──
    /// Total observations ever ingested by the event observer.
    pub total_observations: u64,
    /// Observations in the last 24 hours.
    pub observations_last_24h: u32,
    /// Observations in the last 7 days.
    pub observations_last_7d: u32,

    // ── Beliefs ──
    /// Total beliefs formed autonomously (all-time).
    pub beliefs_formed: u64,
    /// Beliefs currently at Established or Certain stage.
    pub beliefs_established: u32,
    /// Beliefs pruned (confidence fell too low).
    pub beliefs_pruned: u64,
    /// Active beliefs (Hypothesis + Emerging + Established + Certain).
    pub beliefs_active: u32,
    /// Belief stage breakdown.
    pub belief_stages: BeliefStageBreakdown,

    // ── Skills ──
    /// Total skills ever discovered.
    pub skills_discovered: u64,
    /// Skills approved/promoted by user interaction.
    pub skills_promoted: u64,
    /// Skills currently active and offerable.
    pub skills_active: usize,
    /// Skills pending offer (candidate stage).
    pub skills_pending: usize,
    /// Skills deprecated.
    pub skills_deprecated: u64,
    /// Skill origin breakdown.
    pub skill_origins: HashMap<String, usize>,

    // ── World Model ──
    /// Unique (state, action) pairs learned.
    pub transition_pairs_learned: usize,
    /// World model prediction accuracy [0.0, 1.0].
    pub world_model_accuracy: f64,
    /// Total transitions observed.
    pub world_model_transitions: u64,

    // ── Experiments ──
    /// Total experiments completed (concluded normally).
    pub experiments_completed: u64,
    /// Experiments aborted (safety bound triggered).
    pub experiments_aborted: u64,
    /// Currently active experiments.
    pub experiments_active: usize,
    /// Total experiments ever created.
    pub experiments_total: u64,

    // ── Extraction Efficiency ──
    /// Templates learned by the extractor flywheel.
    pub extraction_templates: usize,
    /// Total template matches (times a template saved an LLM call).
    pub extraction_template_matches: u64,
    /// Average template confidence.
    pub extraction_avg_confidence: f64,

    // ── Calibration ──
    /// Expected Calibration Error (lower = better calibrated).
    pub calibration_error: f64,
    /// Number of weight refits performed.
    pub weight_refits: u64,
    /// Ranking accuracy of learned weights [0.0, 1.0].
    pub weight_accuracy: f64,
    /// Total learning interactions processed.
    pub learning_interactions: u64,
    /// Per-action acceptance rates.
    pub action_acceptance_rates: HashMap<String, f64>,
    /// Source reliability scores.
    pub source_reliabilities: HashMap<String, f64>,

    // ── Surprising Discoveries ──
    /// Notable beliefs the system has formed with high confidence.
    pub discoveries: Vec<Discovery>,

    // ── Timeline ──
    /// Key learning milestones in chronological order.
    pub milestones: Vec<LearningMilestone>,

    // ── Meta ──
    /// Timestamp when this report was generated.
    pub generated_at: f64,
}

/// Breakdown of beliefs by lifecycle stage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BeliefStageBreakdown {
    pub hypothesis: u32,
    pub emerging: u32,
    pub established: u32,
    pub certain: u32,
    pub dying: u32,
}

/// A notable discovery — a belief that crossed the "interesting" threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Discovery {
    /// Human-readable description (e.g., "You're more productive in the morning").
    pub description: String,
    /// Belief category as string.
    pub category: String,
    /// Current confidence [0.0, 1.0].
    pub confidence: f64,
    /// Total evidence observations (confirming + contradicting).
    pub evidence_count: u32,
    /// When the belief was first formed (Unix seconds).
    pub discovered_at: f64,
    /// Current belief stage.
    pub stage: String,
    /// The dedup key for cross-referencing.
    pub dedup_key: String,
}

/// A learning milestone — a significant event in the system's cognitive growth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningMilestone {
    /// Unix timestamp of the milestone.
    pub timestamp: f64,
    /// Type of milestone.
    pub kind: MilestoneKind,
    /// Human-readable description.
    pub description: String,
}

/// Types of learning milestones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MilestoneKind {
    /// First belief formed.
    FirstBelief,
    /// A belief reached Established stage.
    BeliefEstablished,
    /// A belief reached Certain stage.
    BeliefCertain,
    /// A skill was discovered.
    SkillDiscovered,
    /// A skill was promoted.
    SkillPromoted,
    /// An experiment concluded.
    ExperimentConcluded,
    /// Calibration milestone (e.g., first refit).
    CalibrationRefit,
    /// Observation volume milestone.
    ObservationMilestone,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 2  Report Inputs
// ══════════════════════════════════════════════════════════════════════════════

/// All subsystem states needed to generate the introspection report.
///
/// The engine layer loads each component and passes it here as a bundle.
/// This avoids the cognition layer needing to know about persistence.
pub struct IntrospectionInputs<'a> {
    pub belief_store: &'a BeliefStore,
    pub event_buffer: &'a EventBuffer,
    pub skill_registry: &'a SkillRegistry,
    pub experiment_registry: &'a ExperimentRegistry,
    pub template_store: &'a TemplateStore,
    pub learning_state: &'a LearningState,
    pub transition_model: &'a TransitionModel,
    pub now: f64,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  Report Generation
// ══════════════════════════════════════════════════════════════════════════════

/// Generate the full introspection report.
///
/// This is the main entry point — called by the engine layer after loading
/// all subsystem states.
pub fn what_have_you_learned(inputs: &IntrospectionInputs) -> IntrospectionReport {
    let now = inputs.now;

    // ── Observations ──
    let total_observations = inputs.event_buffer.total_ingested;
    let observations_last_24h = count_recent_events(inputs.event_buffer, now, 86400.0);
    let observations_last_7d = count_recent_events(inputs.event_buffer, now, 604800.0);

    // ── Beliefs ──
    let belief_stages = count_belief_stages(inputs.belief_store);
    let beliefs_active = belief_stages.hypothesis
        + belief_stages.emerging
        + belief_stages.established
        + belief_stages.certain;
    let beliefs_established = belief_stages.established + belief_stages.certain;

    // ── Skills ──
    let skill_summary = skills::summarize_skills(inputs.skill_registry);

    // ── World Model ──
    let world_summary = world_model::summarize_world_model(inputs.transition_model);

    // ── Extractor ──
    let extractor_summary = extractor::summarize_extractor(inputs.template_store);

    // ── Calibration ──
    let cal_report = calibration::learning_report(inputs.learning_state);

    // ── Experiments ──
    let exp_active = inputs
        .experiment_registry
        .experiments
        .iter()
        .filter(|e| {
            matches!(
                e.status,
                ExperimentStatus::Running | ExperimentStatus::Designed
            )
        })
        .count();

    // ── Discoveries ──
    let discoveries = extract_discoveries(inputs.belief_store);

    // ── Milestones ──
    let milestones = build_milestones(inputs);

    IntrospectionReport {
        // Volume
        total_observations,
        observations_last_24h,
        observations_last_7d,

        // Beliefs
        beliefs_formed: inputs.belief_store.total_formed,
        beliefs_established,
        beliefs_pruned: inputs.belief_store.total_pruned,
        beliefs_active,
        belief_stages,

        // Skills
        skills_discovered: skill_summary.total_discovered,
        skills_promoted: skill_summary.total_promoted,
        skills_active: skill_summary.active,
        skills_pending: skill_summary.candidates,
        skills_deprecated: skill_summary.total_deprecated,
        skill_origins: skill_summary.by_origin,

        // World Model
        transition_pairs_learned: world_summary.unique_pairs,
        world_model_accuracy: world_summary.prediction_accuracy,
        world_model_transitions: world_summary.total_transitions,

        // Experiments
        experiments_completed: inputs.experiment_registry.total_concluded,
        experiments_aborted: inputs.experiment_registry.total_aborted,
        experiments_active: exp_active,
        experiments_total: inputs.experiment_registry.total_created,

        // Extraction
        extraction_templates: extractor_summary.total_templates,
        extraction_template_matches: extractor_summary.total_template_matches,
        extraction_avg_confidence: extractor_summary.avg_template_confidence,

        // Calibration
        calibration_error: cal_report.calibration_error,
        weight_refits: cal_report.weight_refits,
        weight_accuracy: cal_report.weights.ranking_accuracy,
        learning_interactions: cal_report.total_interactions,
        action_acceptance_rates: cal_report.action_acceptance_rates,
        source_reliabilities: cal_report.source_reliabilities,

        // Discoveries
        discoveries,

        // Milestones
        milestones,

        // Meta
        generated_at: now,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  Discovery Extraction
// ══════════════════════════════════════════════════════════════════════════════

/// Minimum confidence for a belief to be considered a "discovery".
const DISCOVERY_MIN_CONFIDENCE: f64 = 0.6;

/// Minimum evidence count (confirming + contradicting) for a discovery.
const DISCOVERY_MIN_EVIDENCE: u32 = 3;

/// Maximum number of discoveries to include in the report.
const MAX_DISCOVERIES: usize = 20;

/// Extract notable discoveries from the belief store.
///
/// Selects beliefs that are sufficiently confident and well-evidenced,
/// sorted by confidence (highest first), then by evidence count.
fn extract_discoveries(store: &BeliefStore) -> Vec<Discovery> {
    let mut candidates: Vec<Discovery> = store
        .iter()
        .filter(|b| {
            b.confidence >= DISCOVERY_MIN_CONFIDENCE
                && (b.confirming_observations + b.contradicting_observations)
                    >= DISCOVERY_MIN_EVIDENCE
                && !matches!(b.stage, BeliefStage::Dying)
        })
        .map(|b| belief_to_discovery(b))
        .collect();

    // Sort by confidence descending, then evidence count descending
    candidates.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.evidence_count.cmp(&a.evidence_count))
    });

    candidates.truncate(MAX_DISCOVERIES);
    candidates
}

/// Convert an AutonomousBelief into a Discovery.
fn belief_to_discovery(belief: &AutonomousBelief) -> Discovery {
    Discovery {
        description: belief.description.clone(),
        category: format!("{:?}", belief.category),
        confidence: belief.confidence,
        evidence_count: belief.confirming_observations + belief.contradicting_observations,
        discovered_at: belief.formed_at,
        stage: format!("{:?}", belief.stage),
        dedup_key: belief.dedup_key.clone(),
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 5  Milestone Construction
// ══════════════════════════════════════════════════════════════════════════════

/// Maximum number of milestones in the timeline.
const MAX_MILESTONES: usize = 50;

/// Build a chronological timeline of learning milestones.
fn build_milestones(inputs: &IntrospectionInputs) -> Vec<LearningMilestone> {
    let mut milestones = Vec::new();

    // ── Belief milestones ──
    for belief in inputs.belief_store.iter() {
        match belief.stage {
            BeliefStage::Established => {
                milestones.push(LearningMilestone {
                    timestamp: belief.last_updated,
                    kind: MilestoneKind::BeliefEstablished,
                    description: format!(
                        "Belief established: {}",
                        truncate(&belief.description, 80)
                    ),
                });
            }
            BeliefStage::Certain => {
                milestones.push(LearningMilestone {
                    timestamp: belief.last_updated,
                    kind: MilestoneKind::BeliefCertain,
                    description: format!(
                        "Belief confirmed with certainty: {}",
                        truncate(&belief.description, 80)
                    ),
                });
            }
            _ => {}
        }
    }

    // ── First belief milestone ──
    if let Some(earliest) = inputs.belief_store.iter().min_by(|a, b| {
        a.formed_at
            .partial_cmp(&b.formed_at)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        milestones.push(LearningMilestone {
            timestamp: earliest.formed_at,
            kind: MilestoneKind::FirstBelief,
            description: format!(
                "First belief formed: {}",
                truncate(&earliest.description, 80)
            ),
        });
    }

    // ── Skill milestones ──
    for skill in inputs.skill_registry.skills.values() {
        match skill.stage {
            SkillStage::Promoted => {
                milestones.push(LearningMilestone {
                    timestamp: if skill.last_seen_at > 0.0 {
                        skill.last_seen_at
                    } else {
                        skill.discovered_at
                    },
                    kind: MilestoneKind::SkillPromoted,
                    description: format!("Skill promoted: {}", truncate(&skill.description, 80)),
                });
            }
            SkillStage::Candidate | SkillStage::Validated | SkillStage::Reliable => {
                milestones.push(LearningMilestone {
                    timestamp: skill.discovered_at,
                    kind: MilestoneKind::SkillDiscovered,
                    description: format!("Skill discovered: {}", truncate(&skill.description, 80)),
                });
            }
            _ => {}
        }
    }

    // ── Experiment milestones ──
    for exp in &inputs.experiment_registry.experiments {
        if matches!(exp.status, ExperimentStatus::Concluded) {
            milestones.push(LearningMilestone {
                timestamp: exp.created_at,
                kind: MilestoneKind::ExperimentConcluded,
                description: format!("Experiment concluded: {}", truncate(&exp.hypothesis, 80)),
            });
        }
    }

    // Sort chronologically and truncate
    milestones.sort_by(|a, b| {
        a.timestamp
            .partial_cmp(&b.timestamp)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    milestones.truncate(MAX_MILESTONES);
    milestones
}

// ══════════════════════════════════════════════════════════════════════════════
// § 6  Helpers
// ══════════════════════════════════════════════════════════════════════════════

/// Count belief stages.
fn count_belief_stages(store: &BeliefStore) -> BeliefStageBreakdown {
    let mut breakdown = BeliefStageBreakdown::default();
    for belief in store.iter() {
        match belief.stage {
            BeliefStage::Hypothesis => breakdown.hypothesis += 1,
            BeliefStage::Emerging => breakdown.emerging += 1,
            BeliefStage::Established => breakdown.established += 1,
            BeliefStage::Certain => breakdown.certain += 1,
            BeliefStage::Dying => breakdown.dying += 1,
        }
    }
    breakdown
}

/// Count events in the buffer within a time window.
///
/// Scans the event buffer (newest-first via `recent()`) and counts
/// events with timestamp >= (now - window_secs).
fn count_recent_events(buffer: &EventBuffer, now: f64, window_secs: f64) -> u32 {
    let cutoff = now - window_secs;
    let events = buffer.recent(buffer.len());
    events.iter().filter(|e| e.timestamp >= cutoff).count() as u32
}

/// Truncate a string to a maximum character length, appending "..." if truncated.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 7  Discovery Explanation
// ══════════════════════════════════════════════════════════════════════════════

/// Detailed explanation of a single discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryExplanation {
    /// The discovery itself.
    pub discovery: Discovery,
    /// How many observations confirmed this belief.
    pub confirming_observations: u32,
    /// How many observations contradicted this belief.
    pub contradicting_observations: u32,
    /// The confirmation ratio.
    pub confirmation_ratio: f64,
    /// Stage progression description.
    pub stage_history: String,
    /// Category-specific evidence summary.
    pub evidence_summary: String,
}

/// Explain a specific discovery by its dedup_key.
pub fn explain_discovery(store: &BeliefStore, dedup_key: &str) -> Option<DiscoveryExplanation> {
    let belief = store.iter().find(|b| b.dedup_key == dedup_key)?;

    let total = belief.confirming_observations + belief.contradicting_observations;
    let confirmation_ratio = if total > 0 {
        belief.confirming_observations as f64 / total as f64
    } else {
        0.0
    };

    let stage_history = format!(
        "Currently at {:?} stage (confidence: {:.1}%)",
        belief.stage,
        belief.confidence * 100.0
    );

    let evidence_summary = format!(
        "{} confirming vs {} contradicting observations. Category: {:?}. Evidence type: {:?}.",
        belief.confirming_observations,
        belief.contradicting_observations,
        belief.category,
        std::mem::discriminant(&belief.evidence)
    );

    Some(DiscoveryExplanation {
        discovery: belief_to_discovery(belief),
        confirming_observations: belief.confirming_observations,
        contradicting_observations: belief.contradicting_observations,
        confirmation_ratio,
        stage_history,
        evidence_summary,
    })
}

// ══════════════════════════════════════════════════════════════════════════════
// § 8  Learning Timeline Query
// ══════════════════════════════════════════════════════════════════════════════

/// Get learning milestones within a time range.
pub fn milestones_in_range(
    inputs: &IntrospectionInputs,
    since: f64,
    until: f64,
) -> Vec<LearningMilestone> {
    build_milestones(inputs)
        .into_iter()
        .filter(|m| m.timestamp >= since && m.timestamp <= until)
        .collect()
}

/// Get a compact summary string suitable for display.
pub fn learning_summary_text(report: &IntrospectionReport) -> String {
    let mut parts = Vec::new();

    if report.beliefs_formed > 0 {
        parts.push(format!(
            "{} beliefs formed ({} established)",
            report.beliefs_formed, report.beliefs_established
        ));
    }

    if report.skills_discovered > 0 {
        parts.push(format!(
            "{} skills discovered ({} active)",
            report.skills_discovered, report.skills_active
        ));
    }

    if report.world_model_transitions > 0 {
        parts.push(format!(
            "{} transitions learned ({:.0}% accuracy)",
            report.transition_pairs_learned,
            report.world_model_accuracy * 100.0
        ));
    }

    if report.experiments_total > 0 {
        parts.push(format!(
            "{} experiments ({} completed)",
            report.experiments_total, report.experiments_completed
        ));
    }

    if report.extraction_templates > 0 {
        parts.push(format!(
            "{} extraction templates ({} matches)",
            report.extraction_templates, report.extraction_template_matches
        ));
    }

    if report.learning_interactions > 0 {
        parts.push(format!(
            "calibration: {:.3} ECE, {:.0}% ranking accuracy",
            report.calibration_error,
            report.weight_accuracy * 100.0
        ));
    }

    if parts.is_empty() {
        "No learning data yet. The system will learn from interactions over time.".to_string()
    } else {
        parts.join(" | ")
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 9  Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::LearningState;
    use crate::experimenter::ExperimentRegistry;
    use crate::extractor::TemplateStore;
    use crate::flywheel::{BeliefCategory, BeliefEvidence, BeliefStore};
    use crate::observer::{EventBuffer, EventKind};
    use crate::skills::SkillRegistry;
    use crate::world_model::TransitionModel;

    fn ts(offset: f64) -> f64 {
        86400.0 * 100.0 + offset
    }

    fn temporal_evidence() -> BeliefEvidence {
        BeliefEvidence::Temporal {
            event_kind: EventKind::AppOpened,
            peak_hour: 9,
            quiet_hours: vec![],
            distribution_skew: 0.8,
        }
    }

    fn preference_evidence(preferred: &str) -> BeliefEvidence {
        BeliefEvidence::Preference {
            preferred_value: preferred.to_string(),
            accept_rate: 0.8,
            reject_rate: 0.2,
            sample_size: 10,
        }
    }

    fn behavioral_evidence() -> BeliefEvidence {
        BeliefEvidence::Behavioral {
            from_app: 14,
            to_app: 20,
            transition_count: 15,
            avg_gap_ms: 2000.0,
        }
    }

    fn empty_inputs() -> (
        BeliefStore,
        EventBuffer,
        SkillRegistry,
        ExperimentRegistry,
        TemplateStore,
        LearningState,
        TransitionModel,
    ) {
        (
            BeliefStore::new(),
            EventBuffer::new(1000),
            SkillRegistry::new(),
            ExperimentRegistry::new(),
            TemplateStore::new(),
            LearningState::new(),
            TransitionModel::new(),
        )
    }

    fn make_inputs<'a>(
        bs: &'a BeliefStore,
        eb: &'a EventBuffer,
        sr: &'a SkillRegistry,
        er: &'a ExperimentRegistry,
        ts_store: &'a TemplateStore,
        ls: &'a LearningState,
        tm: &'a TransitionModel,
        now: f64,
    ) -> IntrospectionInputs<'a> {
        IntrospectionInputs {
            belief_store: bs,
            event_buffer: eb,
            skill_registry: sr,
            experiment_registry: er,
            template_store: ts_store,
            learning_state: ls,
            transition_model: tm,
            now,
        }
    }

    #[test]
    fn test_empty_report() {
        let (bs, eb, sr, er, ts_store, ls, tm) = empty_inputs();
        let inputs = make_inputs(&bs, &eb, &sr, &er, &ts_store, &ls, &tm, ts(0.0));

        let report = what_have_you_learned(&inputs);
        assert_eq!(report.total_observations, 0);
        assert_eq!(report.beliefs_formed, 0);
        assert_eq!(report.skills_discovered, 0);
        assert_eq!(report.experiments_total, 0);
        assert!(report.discoveries.is_empty());
        assert!(report.milestones.is_empty());
    }

    #[test]
    fn test_empty_summary_text() {
        let (bs, eb, sr, er, ts_store, ls, tm) = empty_inputs();
        let inputs = make_inputs(&bs, &eb, &sr, &er, &ts_store, &ls, &tm, ts(0.0));

        let report = what_have_you_learned(&inputs);
        let text = learning_summary_text(&report);
        assert!(text.contains("No learning data"), "Empty report: {}", text);
    }

    #[test]
    fn test_belief_stage_breakdown() {
        let mut store = BeliefStore::new();

        let mut b1 = AutonomousBelief::new(
            "Morning person".to_string(),
            BeliefCategory::Temporal,
            "temporal:morning".to_string(),
            temporal_evidence(),
            ts(0.0),
        );
        b1.confidence = 0.75;
        b1.stage = BeliefStage::Established;
        store.upsert(b1);

        let mut b2 = AutonomousBelief::new(
            "Likes coffee".to_string(),
            BeliefCategory::Preference,
            "pref:coffee".to_string(),
            preference_evidence("coffee"),
            ts(1.0),
        );
        b2.confidence = 0.9;
        b2.stage = BeliefStage::Certain;
        store.upsert(b2);

        let b3 = AutonomousBelief::new(
            "Maybe night owl".to_string(),
            BeliefCategory::Temporal,
            "temporal:night".to_string(),
            temporal_evidence(),
            ts(2.0),
        );
        store.upsert(b3);

        let breakdown = count_belief_stages(&store);
        assert_eq!(breakdown.established, 1);
        assert_eq!(breakdown.certain, 1);
        assert_eq!(breakdown.hypothesis, 1);
    }

    #[test]
    fn test_discovery_extraction() {
        let mut store = BeliefStore::new();

        // High-confidence, well-evidenced belief → should be a discovery
        let mut b1 = AutonomousBelief::new(
            "You're most productive between 9am and 11am".to_string(),
            BeliefCategory::Productivity,
            "prod:morning_peak".to_string(),
            behavioral_evidence(),
            ts(0.0),
        );
        b1.confidence = 0.85;
        b1.confirming_observations = 12;
        b1.contradicting_observations = 2;
        b1.stage = BeliefStage::Established;
        store.upsert(b1);

        // Low-confidence belief → should NOT be a discovery
        let b2 = AutonomousBelief::new(
            "Maybe likes dark theme".to_string(),
            BeliefCategory::Preference,
            "pref:dark".to_string(),
            preference_evidence("dark"),
            ts(1.0),
        );
        store.upsert(b2);

        let discoveries = extract_discoveries(&store);
        assert_eq!(discoveries.len(), 1);
        assert!(discoveries[0].description.contains("productive"));
        assert_eq!(discoveries[0].evidence_count, 14);
    }

    #[test]
    fn test_discovery_sorting() {
        let mut store = BeliefStore::new();

        let mut b1 = AutonomousBelief::new(
            "Prefers Rust".to_string(),
            BeliefCategory::Preference,
            "pref:rust".to_string(),
            preference_evidence("rust"),
            ts(0.0),
        );
        b1.confidence = 0.8;
        b1.confirming_observations = 5;
        b1.stage = BeliefStage::Established;
        store.upsert(b1);

        let mut b2 = AutonomousBelief::new(
            "Morning worker".to_string(),
            BeliefCategory::Temporal,
            "temp:morning".to_string(),
            temporal_evidence(),
            ts(1.0),
        );
        b2.confidence = 0.9;
        b2.confirming_observations = 18;
        b2.contradicting_observations = 2;
        b2.stage = BeliefStage::Certain;
        store.upsert(b2);

        let discoveries = extract_discoveries(&store);
        assert_eq!(discoveries.len(), 2);
        assert!(discoveries[0].confidence >= discoveries[1].confidence);
    }

    #[test]
    fn test_discovery_excludes_dying() {
        let mut store = BeliefStore::new();

        let mut b1 = AutonomousBelief::new(
            "Dying belief".to_string(),
            BeliefCategory::Preference,
            "dying:1".to_string(),
            preference_evidence("a"),
            ts(0.0),
        );
        b1.confidence = 0.7;
        b1.confirming_observations = 5;
        b1.stage = BeliefStage::Dying;
        store.upsert(b1);

        let discoveries = extract_discoveries(&store);
        assert!(
            discoveries.is_empty(),
            "Dying beliefs should not be discoveries"
        );
    }

    #[test]
    fn test_explain_discovery() {
        let mut store = BeliefStore::new();

        let mut b1 = AutonomousBelief::new(
            "Likes morning meetings".to_string(),
            BeliefCategory::Preference,
            "pref:morning_meetings".to_string(),
            preference_evidence("morning"),
            ts(0.0),
        );
        b1.confidence = 0.75;
        b1.confirming_observations = 8;
        b1.contradicting_observations = 3;
        b1.stage = BeliefStage::Established;
        store.upsert(b1);

        let explanation = explain_discovery(&store, "pref:morning_meetings").unwrap();
        assert_eq!(explanation.confirming_observations, 8);
        assert_eq!(explanation.contradicting_observations, 3);
        assert!(explanation.confirmation_ratio > 0.7);
        assert!(explanation.stage_history.contains("Established"));

        assert!(explain_discovery(&store, "nonexistent").is_none());
    }

    #[test]
    fn test_milestone_construction() {
        let mut store = BeliefStore::new();

        let mut b1 = AutonomousBelief::new(
            "First belief".to_string(),
            BeliefCategory::Temporal,
            "temp:first".to_string(),
            temporal_evidence(),
            ts(100.0),
        );
        b1.confidence = 0.8;
        b1.stage = BeliefStage::Established;
        b1.last_updated = ts(200.0);
        store.upsert(b1);

        let (eb, sr, er, ts_store, ls, tm) = (
            EventBuffer::new(1000),
            SkillRegistry::new(),
            ExperimentRegistry::new(),
            TemplateStore::new(),
            LearningState::new(),
            TransitionModel::new(),
        );
        let inputs = make_inputs(&store, &eb, &sr, &er, &ts_store, &ls, &tm, ts(300.0));

        let milestones = build_milestones(&inputs);
        assert!(!milestones.is_empty());

        let kinds: Vec<_> = milestones.iter().map(|m| format!("{:?}", m.kind)).collect();
        assert!(kinds.iter().any(|k| k.contains("FirstBelief")));
        assert!(kinds.iter().any(|k| k.contains("BeliefEstablished")));
    }

    #[test]
    fn test_milestones_sorted_chronologically() {
        let mut store = BeliefStore::new();

        let mut b1 = AutonomousBelief::new(
            "Later belief".to_string(),
            BeliefCategory::Temporal,
            "temp:later".to_string(),
            temporal_evidence(),
            ts(500.0),
        );
        b1.confidence = 0.8;
        b1.stage = BeliefStage::Established;
        b1.last_updated = ts(600.0);
        store.upsert(b1);

        let mut b2 = AutonomousBelief::new(
            "Earlier belief".to_string(),
            BeliefCategory::Preference,
            "pref:earlier".to_string(),
            preference_evidence("a"),
            ts(100.0),
        );
        b2.confidence = 0.9;
        b2.stage = BeliefStage::Certain;
        b2.last_updated = ts(200.0);
        store.upsert(b2);

        let (eb, sr, er, ts_store, ls, tm) = (
            EventBuffer::new(1000),
            SkillRegistry::new(),
            ExperimentRegistry::new(),
            TemplateStore::new(),
            LearningState::new(),
            TransitionModel::new(),
        );
        let inputs = make_inputs(&store, &eb, &sr, &er, &ts_store, &ls, &tm, ts(1000.0));

        let milestones = build_milestones(&inputs);
        for i in 1..milestones.len() {
            assert!(
                milestones[i].timestamp >= milestones[i - 1].timestamp,
                "Milestones not sorted: {} at index {} < {} at index {}",
                milestones[i].timestamp,
                i,
                milestones[i - 1].timestamp,
                i - 1
            );
        }
    }

    #[test]
    fn test_report_with_beliefs() {
        let mut store = BeliefStore::new();

        let mut b1 = AutonomousBelief::new(
            "Test belief".to_string(),
            BeliefCategory::Behavioral,
            "beh:test".to_string(),
            behavioral_evidence(),
            ts(0.0),
        );
        b1.confidence = 0.75;
        b1.stage = BeliefStage::Established;
        b1.confirming_observations = 8;
        store.upsert(b1); // total_formed becomes 1

        // Adjust counters after upsert to simulate prior history
        store.total_formed = 5;
        store.total_pruned = 1;

        let (eb, sr, er, ts_store, ls, tm) = (
            EventBuffer::new(1000),
            SkillRegistry::new(),
            ExperimentRegistry::new(),
            TemplateStore::new(),
            LearningState::new(),
            TransitionModel::new(),
        );
        let inputs = make_inputs(&store, &eb, &sr, &er, &ts_store, &ls, &tm, ts(100.0));

        let report = what_have_you_learned(&inputs);
        assert_eq!(report.beliefs_formed, 5);
        assert_eq!(report.beliefs_pruned, 1);
        assert_eq!(report.beliefs_established, 1);
        assert_eq!(report.beliefs_active, 1);
    }

    #[test]
    fn test_summary_text_with_data() {
        let mut store = BeliefStore::new();

        let mut b = AutonomousBelief::new(
            "Test".to_string(),
            BeliefCategory::Temporal,
            "t:1".to_string(),
            temporal_evidence(),
            ts(0.0),
        );
        b.confidence = 0.8;
        b.stage = BeliefStage::Established;
        store.upsert(b); // total_formed = 1

        // Set to desired value after upsert
        store.total_formed = 10;

        let (eb, sr, er, ts_store, ls, tm) = (
            EventBuffer::new(1000),
            SkillRegistry::new(),
            ExperimentRegistry::new(),
            TemplateStore::new(),
            LearningState::new(),
            TransitionModel::new(),
        );
        let inputs = make_inputs(&store, &eb, &sr, &er, &ts_store, &ls, &tm, ts(100.0));

        let report = what_have_you_learned(&inputs);
        let text = learning_summary_text(&report);
        assert!(text.contains("10 beliefs formed"), "Summary: {}", text);
        assert!(text.contains("1 established"), "Summary: {}", text);
    }

    #[test]
    fn test_truncate_helper() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("this is a longer string", 10), "this is...");
        assert_eq!(truncate("exact len!", 10), "exact len!");
    }

    #[test]
    fn test_milestones_in_range() {
        let mut store = BeliefStore::new();

        let mut b1 = AutonomousBelief::new(
            "In range".to_string(),
            BeliefCategory::Temporal,
            "temp:in".to_string(),
            temporal_evidence(),
            ts(50.0),
        );
        b1.confidence = 0.8;
        b1.stage = BeliefStage::Established;
        b1.last_updated = ts(60.0);
        store.upsert(b1);

        let mut b2 = AutonomousBelief::new(
            "Out of range".to_string(),
            BeliefCategory::Temporal,
            "temp:out".to_string(),
            temporal_evidence(),
            ts(500.0),
        );
        b2.confidence = 0.8;
        b2.stage = BeliefStage::Established;
        b2.last_updated = ts(600.0);
        store.upsert(b2);

        let (eb, sr, er, ts_store, ls, tm) = (
            EventBuffer::new(1000),
            SkillRegistry::new(),
            ExperimentRegistry::new(),
            TemplateStore::new(),
            LearningState::new(),
            TransitionModel::new(),
        );
        let inputs = make_inputs(&store, &eb, &sr, &er, &ts_store, &ls, &tm, ts(1000.0));

        let in_range = milestones_in_range(&inputs, ts(0.0), ts(100.0));
        for m in &in_range {
            assert!(m.timestamp >= ts(0.0) && m.timestamp <= ts(100.0));
        }
    }
}
