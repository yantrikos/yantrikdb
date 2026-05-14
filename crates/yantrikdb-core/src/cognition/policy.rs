//! CK-1.9: Policy Engine + Constraint Filtering
//!
//! The final selection layer in the cognitive pipeline. Applies hard and soft
//! constraints to evaluated action candidates, computes diversity penalties,
//! and produces a single chosen action with a full reasoning trace.
//!
//! ## Pipeline
//!
//! 1. **Hard constraint filter** — binary pass/fail, removes candidates
//! 2. **Soft constraint penalties** — adjusts utility scores
//! 3. **Diversity penalty** — suppresses repetitive suggestions
//! 4. **Selection** — argmax on adjusted utility
//! 5. **Trace generation** — explainable reasoning for every decision
//!
//! ## Hard Constraints
//!
//! - Privacy: don't surface private memories in shared context
//! - Quiet hours: no proactive actions during do-not-disturb
//! - Cooldown: same action type within cooldown period
//! - Confidence floor: per-action-type minimum confidence
//! - Device capability: action requires capabilities not present
//!
//! ## Soft Constraints
//!
//! - Repetition suppression (diminishing returns)
//! - Emotional sensitivity (reduce proactivity during distress)
//! - Cognitive load awareness (attention budget depleted)
//! - Anti-spam (max N suggestions per time window)

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::evaluator::EvaluatedAction;
use super::state::*;

// ── Policy Configuration ──

/// Configuration for the policy engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    /// Enable quiet-hours enforcement.
    pub enforce_quiet_hours: bool,

    /// Quiet hours start (0-23, local time).
    pub quiet_hours_start: u8,

    /// Quiet hours end (0-23, local time).
    pub quiet_hours_end: u8,

    /// Per-action-kind confidence floors.
    /// Actions below their kind's floor are rejected.
    pub confidence_floors: HashMap<String, f64>,

    /// Cooldown in seconds: minimum time between same action kind.
    pub action_cooldown_secs: f64,

    /// Maximum suggestions per window (anti-spam).
    pub max_suggestions_per_window: usize,

    /// Anti-spam window duration in seconds.
    pub suggestion_window_secs: f64,

    /// Repetition suppression factor [0.0, 1.0].
    /// Higher = stronger suppression of repeated action kinds.
    pub repetition_suppression: f64,

    /// Diversity penalty: how much to penalize consecutive same-kind actions.
    pub diversity_penalty: f64,

    /// Emotional sensitivity multiplier.
    /// When user distress is detected, proactivity is scaled by (1.0 - this * distress).
    pub emotional_sensitivity: f64,

    /// Cognitive load threshold [0.0, 1.0].
    /// When estimated load exceeds this, only low-cost actions pass.
    pub cognitive_load_threshold: f64,

    /// Minimum adjusted utility to select (below → Wait).
    pub selection_threshold: f64,

    /// Whether to escalate to LLM when all candidates are below threshold.
    pub escalate_to_llm_on_empty: bool,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        let mut confidence_floors = HashMap::new();
        // Conservative defaults: high-cost actions need higher confidence
        // Keys must match ActionKind::as_str() (lowercase)
        confidence_floors.insert("execute".to_string(), 0.70);
        confidence_floors.insert("warn".to_string(), 0.55);
        confidence_floors.insert("schedule".to_string(), 0.50);
        confidence_floors.insert("communicate".to_string(), 0.45);
        confidence_floors.insert("suggest".to_string(), 0.35);
        confidence_floors.insert("organize".to_string(), 0.30);
        confidence_floors.insert("inform".to_string(), 0.25);
        confidence_floors.insert("abstain".to_string(), 0.0);

        Self {
            enforce_quiet_hours: true,
            quiet_hours_start: 22,
            quiet_hours_end: 7,
            confidence_floors,
            action_cooldown_secs: 300.0, // 5 minutes
            max_suggestions_per_window: 10,
            suggestion_window_secs: 3600.0, // 1 hour
            repetition_suppression: 0.3,
            diversity_penalty: 0.15,
            emotional_sensitivity: 0.6,
            cognitive_load_threshold: 0.8,
            selection_threshold: 0.05,
            escalate_to_llm_on_empty: true,
        }
    }
}

// ── Context Snapshot ──

/// Snapshot of the user's current context for policy decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyContext {
    /// Current hour (0-23, local time).
    pub current_hour: u8,

    /// Whether do-not-disturb is explicitly enabled.
    pub dnd_active: bool,

    /// Recent action history: (action_kind_name, timestamp_secs).
    pub recent_actions: Vec<(String, f64)>,

    /// Estimated user distress level [0.0, 1.0].
    /// Derived from recent emotional episodes or valence.
    pub distress_level: f64,

    /// Estimated cognitive load [0.0, 1.0].
    /// Derived from active task count, urgency, attention usage.
    pub cognitive_load: f64,

    /// Number of suggestions already made in the current window.
    pub suggestions_in_window: usize,

    /// Device capabilities available (e.g., "notifications", "audio", "display").
    pub device_capabilities: Vec<String>,

    /// Whether the current context is shared (e.g., screen sharing).
    pub is_shared_context: bool,
}

impl Default for PolicyContext {
    fn default() -> Self {
        Self {
            current_hour: 12,
            dnd_active: false,
            recent_actions: vec![],
            distress_level: 0.0,
            cognitive_load: 0.0,
            suggestions_in_window: 0,
            device_capabilities: vec![
                "notifications".to_string(),
                "audio".to_string(),
                "display".to_string(),
            ],
            is_shared_context: false,
        }
    }
}

// ── Reasoning Trace ──

/// Why a candidate was rejected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectedCandidate {
    /// Schema name of the rejected action.
    pub schema_name: String,

    /// Action kind.
    pub action_kind: String,

    /// Original utility before policy adjustments.
    pub original_utility: f64,

    /// The constraint that caused rejection.
    pub rejection_reason: RejectionReason,
}

/// Specific reason a candidate was rejected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RejectionReason {
    /// Quiet hours / DND active.
    QuietHours,
    /// Below confidence floor for this action kind.
    ConfidenceFloor { required: f64, actual: f64 },
    /// Cooldown period not elapsed.
    Cooldown { remaining_secs: f64 },
    /// Anti-spam: too many suggestions in window.
    AntiSpam { count: usize, max: usize },
    /// Privacy: action would expose private data in shared context.
    PrivacyFilter,
    /// Adjusted utility below selection threshold.
    BelowThreshold { adjusted: f64, threshold: f64 },
    /// Hard constraint from cognitive graph violated.
    HardConstraint { description: String },
}

/// Contribution of a specific factor to the final decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorContribution {
    /// Name of the factor (e.g., "emotional_sensitivity", "diversity_penalty").
    pub factor: String,
    /// How much this factor affected the utility.
    pub delta: f64,
    /// Explanation.
    pub description: String,
}

/// The full reasoning trace for a policy decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTrace {
    /// Active factors that influenced the final score.
    pub active_factors: Vec<FactorContribution>,

    /// Candidates that were rejected and why.
    pub rejected_candidates: Vec<RejectedCandidate>,

    /// Number of candidates that passed hard constraints.
    pub passed_hard_filter: usize,

    /// Number of candidates after soft penalty adjustment.
    pub passed_soft_filter: usize,

    /// Total execution time in microseconds.
    pub execution_time_us: u64,
}

// ── Policy Decision Output ──

/// The outcome of policy selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyDecision {
    /// A specific action was selected.
    Act(SelectedAction),
    /// All candidates were filtered or below threshold — wait.
    Wait {
        /// Why we decided to wait.
        reason: String,
    },
    /// All candidates below threshold, escalate to LLM for creative response.
    EscalateToLlm {
        /// Context to pass to the LLM.
        context_summary: String,
    },
}

/// A selected action with adjusted score and trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectedAction {
    /// The evaluated action that was selected.
    pub action: EvaluatedAction,

    /// Utility after policy adjustments.
    pub adjusted_utility: f64,

    /// Alternative actions (ranked by adjusted utility).
    pub alternatives: Vec<AdjustedCandidate>,
}

/// A candidate with its policy-adjusted utility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjustedCandidate {
    /// Schema name.
    pub schema_name: String,
    /// Action kind.
    pub action_kind: ActionKind,
    /// Adjusted utility after all policy penalties.
    pub adjusted_utility: f64,
    /// Original utility before policy.
    pub original_utility: f64,
}

/// Complete policy engine result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyResult {
    /// The decision made.
    pub decision: PolicyDecision,

    /// Full reasoning trace.
    pub trace: ReasoningTrace,

    /// Total candidates input.
    pub total_input: usize,
}

// ── Hard Constraint Checks ──

/// Check quiet hours constraint.
fn check_quiet_hours(
    config: &PolicyConfig,
    ctx: &PolicyContext,
    action: &EvaluatedAction,
) -> Option<RejectionReason> {
    if !config.enforce_quiet_hours {
        return None;
    }

    // DND always blocks proactive actions
    if ctx.dnd_active && action.candidate.action_kind != ActionKind::Abstain {
        return Some(RejectionReason::QuietHours);
    }

    // Check time-based quiet hours (handle wrap-around midnight)
    let in_quiet = if config.quiet_hours_start > config.quiet_hours_end {
        // e.g., 22:00 - 07:00
        ctx.current_hour >= config.quiet_hours_start || ctx.current_hour < config.quiet_hours_end
    } else {
        // e.g., 01:00 - 06:00
        ctx.current_hour >= config.quiet_hours_start && ctx.current_hour < config.quiet_hours_end
    };

    if in_quiet && action.candidate.action_kind != ActionKind::Abstain {
        // Allow high-urgency Warn actions through (safety override)
        if action.candidate.action_kind == ActionKind::Warn && action.utility > 0.7 {
            return None;
        }
        return Some(RejectionReason::QuietHours);
    }

    None
}

/// Check per-action-kind confidence floor.
fn check_confidence_floor(
    config: &PolicyConfig,
    action: &EvaluatedAction,
) -> Option<RejectionReason> {
    let kind_str = action.candidate.action_kind.as_str();
    if let Some(&floor) = config.confidence_floors.get(kind_str) {
        if action.confidence < floor {
            return Some(RejectionReason::ConfidenceFloor {
                required: floor,
                actual: action.confidence,
            });
        }
    }
    None
}

/// Check action cooldown.
fn check_cooldown(
    config: &PolicyConfig,
    ctx: &PolicyContext,
    action: &EvaluatedAction,
    now: f64,
) -> Option<RejectionReason> {
    let kind_str = action.candidate.action_kind.as_str().to_string();

    for (recent_kind, timestamp) in &ctx.recent_actions {
        if *recent_kind == kind_str {
            let elapsed = now - timestamp;
            if elapsed < config.action_cooldown_secs {
                return Some(RejectionReason::Cooldown {
                    remaining_secs: config.action_cooldown_secs - elapsed,
                });
            }
        }
    }
    None
}

/// Check anti-spam limit.
fn check_anti_spam(config: &PolicyConfig, ctx: &PolicyContext) -> Option<RejectionReason> {
    if ctx.suggestions_in_window >= config.max_suggestions_per_window {
        return Some(RejectionReason::AntiSpam {
            count: ctx.suggestions_in_window,
            max: config.max_suggestions_per_window,
        });
    }
    None
}

/// Check privacy filter for shared contexts.
fn check_privacy(ctx: &PolicyContext, action: &EvaluatedAction) -> Option<RejectionReason> {
    if !ctx.is_shared_context {
        return None;
    }

    // In shared context, filter actions that might expose personal data.
    // Heuristic: actions involving personal needs, emotional states, or health
    // should not surface in shared contexts.
    let sensitive_schemas = [
        "emotional_check_in",
        "health_reminder",
        "personal_reflection",
        "mood_journaling",
        "suggest_self_care",
    ];

    if sensitive_schemas.contains(&action.candidate.schema_name.as_str()) {
        return Some(RejectionReason::PrivacyFilter);
    }

    None
}

/// Check hard constraints from the cognitive graph (Constraint nodes).
fn check_graph_constraints(
    action: &EvaluatedAction,
    nodes: &[&CognitiveNode],
) -> Option<RejectionReason> {
    for node in nodes {
        if node.id.kind() != NodeKind::Constraint {
            continue;
        }
        if let NodePayload::Constraint(ref c) = node.payload {
            if c.constraint_type != ConstraintType::Hard {
                continue;
            }
            // Check if any bound node in the action's preconditions references
            // a node that the constraint also references
            let constraint_active = is_constraint_applicable(c, action);
            if constraint_active {
                return Some(RejectionReason::HardConstraint {
                    description: c.description.clone(),
                });
            }
        }
    }
    None
}

/// Determine if a hard constraint applies to a given action.
///
/// A constraint applies if:
/// - Its condition matches the action kind (e.g., "no Execute during focus")
/// - Its condition mentions the action's schema name
fn is_constraint_applicable(constraint: &ConstraintPayload, action: &EvaluatedAction) -> bool {
    let condition_lower = constraint.condition.to_lowercase();
    let kind_lower = action.candidate.action_kind.as_str().to_lowercase();
    let schema_lower = action.candidate.schema_name.to_lowercase();

    // Check if constraint condition mentions the action kind or schema
    condition_lower.contains(&kind_lower) || condition_lower.contains(&schema_lower)
}

// ── Soft Constraint Penalties ──

/// Compute all soft constraint penalties and return (total_penalty, factors).
fn compute_soft_penalties(
    action: &EvaluatedAction,
    config: &PolicyConfig,
    ctx: &PolicyContext,
    nodes: &[&CognitiveNode],
) -> (f64, Vec<FactorContribution>) {
    let mut total_penalty = 0.0;
    let mut factors = Vec::new();

    // 1. Repetition suppression
    let rep_penalty = compute_repetition_penalty(action, config, ctx);
    if rep_penalty > 0.0 {
        total_penalty += rep_penalty;
        factors.push(FactorContribution {
            factor: "repetition_suppression".to_string(),
            delta: -rep_penalty,
            description: format!(
                "Same action kind '{}' used recently, penalty {:.3}",
                action.candidate.action_kind.as_str(),
                rep_penalty,
            ),
        });
    }

    // 2. Emotional sensitivity
    let emo_penalty = compute_emotional_penalty(action, config, ctx);
    if emo_penalty > 0.0 {
        total_penalty += emo_penalty;
        factors.push(FactorContribution {
            factor: "emotional_sensitivity".to_string(),
            delta: -emo_penalty,
            description: format!(
                "User distress {:.2} dampens proactivity, penalty {:.3}",
                ctx.distress_level, emo_penalty,
            ),
        });
    }

    // 3. Cognitive load awareness
    let load_penalty = compute_load_penalty(action, config, ctx);
    if load_penalty > 0.0 {
        total_penalty += load_penalty;
        factors.push(FactorContribution {
            factor: "cognitive_load".to_string(),
            delta: -load_penalty,
            description: format!(
                "User cognitive load {:.2} exceeds threshold, penalty {:.3}",
                ctx.cognitive_load, load_penalty,
            ),
        });
    }

    // 4. Graph-based soft constraints
    let soft_penalty = compute_graph_soft_penalties(action, nodes);
    if soft_penalty > 0.0 {
        total_penalty += soft_penalty;
        factors.push(FactorContribution {
            factor: "soft_constraint".to_string(),
            delta: -soft_penalty,
            description: format!(
                "Soft constraints from cognitive graph, penalty {:.3}",
                soft_penalty,
            ),
        });
    }

    (total_penalty, factors)
}

/// Repetition suppression: penalize if the same action kind was used recently.
fn compute_repetition_penalty(
    action: &EvaluatedAction,
    config: &PolicyConfig,
    ctx: &PolicyContext,
) -> f64 {
    let kind_str = action.candidate.action_kind.as_str().to_string();
    let recent_same = ctx
        .recent_actions
        .iter()
        .filter(|(k, _)| k == &kind_str)
        .count();

    if recent_same == 0 {
        return 0.0;
    }

    // Diminishing returns: each repetition adds less penalty
    config.repetition_suppression * (1.0 - 1.0 / (1.0 + recent_same as f64))
}

/// Emotional sensitivity: reduce proactivity during user distress.
fn compute_emotional_penalty(
    action: &EvaluatedAction,
    config: &PolicyConfig,
    ctx: &PolicyContext,
) -> f64 {
    if ctx.distress_level < 0.3 {
        return 0.0; // no penalty for low distress
    }

    // Only penalize proactive actions (not Abstain or Inform)
    if action.candidate.action_kind == ActionKind::Abstain
        || action.candidate.action_kind == ActionKind::Inform
    {
        return 0.0;
    }

    // Scale penalty by distress level and action intrusiveness
    let intrusiveness = action.candidate.action_kind.base_cost();
    config.emotional_sensitivity * ctx.distress_level * intrusiveness
}

/// Cognitive load: penalize disruptive actions when user attention is depleted.
fn compute_load_penalty(
    action: &EvaluatedAction,
    config: &PolicyConfig,
    ctx: &PolicyContext,
) -> f64 {
    if ctx.cognitive_load < config.cognitive_load_threshold {
        return 0.0;
    }

    // Over-threshold: penalize proportional to excess load and action cost
    let excess = ctx.cognitive_load - config.cognitive_load_threshold;
    let cost = action.candidate.action_kind.base_cost();

    excess * cost * 2.0 // amplified for high-cost actions
}

/// Soft constraint penalties from Constraint nodes in the cognitive graph.
fn compute_graph_soft_penalties(action: &EvaluatedAction, nodes: &[&CognitiveNode]) -> f64 {
    let mut penalty = 0.0;

    for node in nodes {
        if node.id.kind() != NodeKind::Constraint {
            continue;
        }
        if let NodePayload::Constraint(ref c) = node.payload {
            if c.constraint_type != ConstraintType::Soft {
                continue;
            }
            if is_constraint_applicable(c, action) {
                // Soft constraints add a moderate penalty (scaled by node activation)
                penalty += 0.1 * node.attrs.activation.max(0.3);
            }
        }
    }

    penalty.min(0.3) // cap graph soft penalties
}

// ── Diversity Penalty ──

/// Apply diversity penalty to prevent N-in-a-row of the same action kind.
fn compute_diversity_penalty(
    action: &EvaluatedAction,
    config: &PolicyConfig,
    ctx: &PolicyContext,
) -> f64 {
    let kind_str = action.candidate.action_kind.as_str().to_string();

    // Count consecutive recent actions of the same kind (from most recent)
    let mut consecutive = 0;
    for (k, _) in ctx.recent_actions.iter().rev() {
        if k == &kind_str {
            consecutive += 1;
        } else {
            break;
        }
    }

    if consecutive == 0 {
        return 0.0;
    }

    // Exponential penalty for consecutive same-kind
    config.diversity_penalty * (consecutive as f64).powi(2)
}

// ── Main Selection Pipeline ──

/// Run the full policy engine on evaluated action candidates.
///
/// This is the main entry point for CK-1.9.
pub fn select_action(
    candidates: &[EvaluatedAction],
    nodes: &[&CognitiveNode],
    config: &PolicyConfig,
    ctx: &PolicyContext,
    now: f64,
) -> PolicyResult {
    let start = std::time::Instant::now();
    let total_input = candidates.len();
    let mut rejected = Vec::new();
    let mut all_factors = Vec::new();

    // ── Phase 1: Hard constraint filter ──

    // Global hard checks (apply once, reject all if triggered)
    let global_spam = check_anti_spam(config, ctx);

    let mut passed_hard: Vec<&EvaluatedAction> = Vec::new();

    for candidate in candidates {
        // Global anti-spam
        if let Some(ref reason) = global_spam {
            rejected.push(RejectedCandidate {
                schema_name: candidate.candidate.schema_name.clone(),
                action_kind: candidate.candidate.action_kind.as_str().to_string(),
                original_utility: candidate.utility,
                rejection_reason: reason.clone(),
            });
            continue;
        }

        // Quiet hours
        if let Some(reason) = check_quiet_hours(config, ctx, candidate) {
            rejected.push(RejectedCandidate {
                schema_name: candidate.candidate.schema_name.clone(),
                action_kind: candidate.candidate.action_kind.as_str().to_string(),
                original_utility: candidate.utility,
                rejection_reason: reason,
            });
            continue;
        }

        // Confidence floor
        if let Some(reason) = check_confidence_floor(config, candidate) {
            rejected.push(RejectedCandidate {
                schema_name: candidate.candidate.schema_name.clone(),
                action_kind: candidate.candidate.action_kind.as_str().to_string(),
                original_utility: candidate.utility,
                rejection_reason: reason,
            });
            continue;
        }

        // Cooldown
        if let Some(reason) = check_cooldown(config, ctx, candidate, now) {
            rejected.push(RejectedCandidate {
                schema_name: candidate.candidate.schema_name.clone(),
                action_kind: candidate.candidate.action_kind.as_str().to_string(),
                original_utility: candidate.utility,
                rejection_reason: reason,
            });
            continue;
        }

        // Privacy
        if let Some(reason) = check_privacy(ctx, candidate) {
            rejected.push(RejectedCandidate {
                schema_name: candidate.candidate.schema_name.clone(),
                action_kind: candidate.candidate.action_kind.as_str().to_string(),
                original_utility: candidate.utility,
                rejection_reason: reason,
            });
            continue;
        }

        // Graph hard constraints
        if let Some(reason) = check_graph_constraints(candidate, nodes) {
            rejected.push(RejectedCandidate {
                schema_name: candidate.candidate.schema_name.clone(),
                action_kind: candidate.candidate.action_kind.as_str().to_string(),
                original_utility: candidate.utility,
                rejection_reason: reason,
            });
            continue;
        }

        passed_hard.push(candidate);
    }

    let passed_hard_count = passed_hard.len();

    // ── Phase 2: Soft constraint penalties ──

    let mut adjusted: Vec<(f64, &EvaluatedAction)> = Vec::new();

    for candidate in &passed_hard {
        let mut penalty = 0.0;

        // Soft constraint penalties
        let (soft_penalty, factors) = compute_soft_penalties(candidate, config, ctx, nodes);
        penalty += soft_penalty;
        all_factors.extend(factors);

        // Diversity penalty
        let div_penalty = compute_diversity_penalty(candidate, config, ctx);
        if div_penalty > 0.0 {
            penalty += div_penalty;
            all_factors.push(FactorContribution {
                factor: "diversity_penalty".to_string(),
                delta: -div_penalty,
                description: format!(
                    "Consecutive same-kind '{}' actions, penalty {:.3}",
                    candidate.candidate.action_kind.as_str(),
                    div_penalty,
                ),
            });
        }

        let adjusted_utility = candidate.utility - penalty;
        adjusted.push((adjusted_utility, candidate));
    }

    // Sort by adjusted utility descending
    adjusted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // ── Phase 3: Threshold filter ──

    let above_threshold: Vec<(f64, &EvaluatedAction)> = adjusted
        .iter()
        .filter(|(u, _)| *u >= config.selection_threshold)
        .cloned()
        .collect();

    // Reject below-threshold
    for (adj_u, candidate) in &adjusted {
        if *adj_u < config.selection_threshold {
            rejected.push(RejectedCandidate {
                schema_name: candidate.candidate.schema_name.clone(),
                action_kind: candidate.candidate.action_kind.as_str().to_string(),
                original_utility: candidate.utility,
                rejection_reason: RejectionReason::BelowThreshold {
                    adjusted: *adj_u,
                    threshold: config.selection_threshold,
                },
            });
        }
    }

    let passed_soft_count = above_threshold.len();

    let trace = ReasoningTrace {
        active_factors: all_factors,
        rejected_candidates: rejected,
        passed_hard_filter: passed_hard_count,
        passed_soft_filter: passed_soft_count,
        execution_time_us: start.elapsed().as_micros() as u64,
    };

    // ── Phase 4: Selection ──

    let decision = if let Some((adj_utility, best)) = above_threshold.first() {
        let alternatives: Vec<AdjustedCandidate> = above_threshold
            .iter()
            .skip(1)
            .take(6)
            .map(|(u, a)| AdjustedCandidate {
                schema_name: a.candidate.schema_name.clone(),
                action_kind: a.candidate.action_kind,
                adjusted_utility: *u,
                original_utility: a.utility,
            })
            .collect();

        PolicyDecision::Act(SelectedAction {
            action: (*best).clone(),
            adjusted_utility: *adj_utility,
            alternatives,
        })
    } else if config.escalate_to_llm_on_empty && total_input > 0 {
        // Had candidates but all filtered — escalate
        PolicyDecision::EscalateToLlm {
            context_summary: format!(
                "{} candidates evaluated, {} passed hard filter, none above threshold {:.2}",
                total_input, passed_hard_count, config.selection_threshold,
            ),
        }
    } else {
        PolicyDecision::Wait {
            reason: if total_input == 0 {
                "No action candidates available".to_string()
            } else {
                format!(
                    "All {} candidates filtered by policy constraints",
                    total_input,
                )
            },
        }
    };

    PolicyResult {
        decision,
        trace,
        total_input,
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::super::action::ActionCandidate;
    use super::super::evaluator::EvaluatedAction;
    use super::*;

    fn make_evaluated(
        name: &str,
        kind: ActionKind,
        utility: f64,
        confidence: f64,
    ) -> EvaluatedAction {
        EvaluatedAction {
            candidate: ActionCandidate {
                schema_name: name.to_string(),
                action_kind: kind,
                description: format!("{name}: test"),
                source_intent: "test intent".to_string(),
                precondition_bindings: vec![],
                satisfied_required: 0,
                total_required: 0,
                satisfied_soft: 0,
                relevance_score: utility * 0.8,
                schema_node: None,
            },
            effect_utility: utility * 0.5,
            base_cost: kind.base_cost(),
            timing_penalty: 0.0,
            intent_alignment: utility * 0.3,
            preference_alignment: 0.0,
            simulation_delta: utility * 0.2,
            utility,
            confidence,
        }
    }

    #[test]
    fn test_select_empty_candidates() {
        let config = PolicyConfig::default();
        let ctx = PolicyContext::default();
        let result = select_action(&[], &[], &config, &ctx, 1000.0);

        assert!(matches!(result.decision, PolicyDecision::Wait { .. }));
        assert_eq!(result.total_input, 0);
    }

    #[test]
    fn test_select_single_candidate() {
        let candidates = vec![make_evaluated(
            "send_reminder",
            ActionKind::Communicate,
            0.5,
            0.8,
        )];
        let config = PolicyConfig::default();
        let ctx = PolicyContext::default();
        let result = select_action(&candidates, &[], &config, &ctx, 1000.0);

        assert!(matches!(result.decision, PolicyDecision::Act(_)));
        if let PolicyDecision::Act(ref selected) = result.decision {
            assert_eq!(selected.action.candidate.schema_name, "send_reminder");
        }
    }

    #[test]
    fn test_quiet_hours_filter() {
        let candidates = vec![
            make_evaluated("send_reminder", ActionKind::Communicate, 0.5, 0.8),
            make_evaluated("abstain", ActionKind::Abstain, 0.1, 1.0),
        ];
        let config = PolicyConfig::default(); // quiet hours 22-07
        let mut ctx = PolicyContext::default();
        ctx.current_hour = 23; // in quiet hours

        let result = select_action(&candidates, &[], &config, &ctx, 1000.0);

        // send_reminder should be rejected, abstain should pass
        let rejected_names: Vec<&str> = result
            .trace
            .rejected_candidates
            .iter()
            .filter(|r| matches!(r.rejection_reason, RejectionReason::QuietHours))
            .map(|r| r.schema_name.as_str())
            .collect();
        assert!(rejected_names.contains(&"send_reminder"));
    }

    #[test]
    fn test_dnd_blocks_proactive() {
        let candidates = vec![make_evaluated(
            "send_reminder",
            ActionKind::Communicate,
            0.5,
            0.8,
        )];
        let config = PolicyConfig::default();
        let mut ctx = PolicyContext::default();
        ctx.dnd_active = true;

        let result = select_action(&candidates, &[], &config, &ctx, 1000.0);
        assert!(result
            .trace
            .rejected_candidates
            .iter()
            .any(|r| { matches!(r.rejection_reason, RejectionReason::QuietHours) }));
    }

    #[test]
    fn test_confidence_floor() {
        // Execute requires 0.70 confidence by default
        let candidates = vec![
            make_evaluated("run_script", ActionKind::Execute, 0.6, 0.50), // below floor
        ];
        let config = PolicyConfig::default();
        let ctx = PolicyContext::default();
        let result = select_action(&candidates, &[], &config, &ctx, 1000.0);

        assert!(result
            .trace
            .rejected_candidates
            .iter()
            .any(|r| { matches!(r.rejection_reason, RejectionReason::ConfidenceFloor { .. }) }));
    }

    #[test]
    fn test_cooldown() {
        let candidates = vec![make_evaluated(
            "send_reminder",
            ActionKind::Communicate,
            0.5,
            0.8,
        )];
        let config = PolicyConfig::default(); // 300s cooldown
        let mut ctx = PolicyContext::default();
        // Same kind executed 100s ago (must match ActionKind::as_str())
        ctx.recent_actions.push(("communicate".to_string(), 900.0));

        let result = select_action(&candidates, &[], &config, &ctx, 1000.0);
        assert!(result
            .trace
            .rejected_candidates
            .iter()
            .any(|r| { matches!(r.rejection_reason, RejectionReason::Cooldown { .. }) }));
    }

    #[test]
    fn test_anti_spam() {
        let candidates = vec![make_evaluated(
            "send_reminder",
            ActionKind::Communicate,
            0.5,
            0.8,
        )];
        let config = PolicyConfig::default(); // max 10 per window
        let mut ctx = PolicyContext::default();
        ctx.suggestions_in_window = 10; // at limit

        let result = select_action(&candidates, &[], &config, &ctx, 1000.0);
        assert!(result
            .trace
            .rejected_candidates
            .iter()
            .any(|r| { matches!(r.rejection_reason, RejectionReason::AntiSpam { .. }) }));
    }

    #[test]
    fn test_privacy_in_shared_context() {
        let candidates = vec![
            make_evaluated("emotional_check_in", ActionKind::Communicate, 0.5, 0.8),
            make_evaluated("send_reminder", ActionKind::Communicate, 0.4, 0.8),
        ];
        let config = PolicyConfig::default();
        let mut ctx = PolicyContext::default();
        ctx.is_shared_context = true;

        let result = select_action(&candidates, &[], &config, &ctx, 1000.0);
        assert!(result.trace.rejected_candidates.iter().any(|r| {
            r.schema_name == "emotional_check_in"
                && matches!(r.rejection_reason, RejectionReason::PrivacyFilter)
        }));
    }

    #[test]
    fn test_emotional_sensitivity_penalty() {
        let candidates = vec![make_evaluated(
            "send_reminder",
            ActionKind::Communicate,
            0.5,
            0.8,
        )];
        let config = PolicyConfig::default();

        // No distress
        let ctx_calm = PolicyContext::default();
        let result_calm = select_action(&candidates, &[], &config, &ctx_calm, 1000.0);

        // High distress
        let mut ctx_distressed = PolicyContext::default();
        ctx_distressed.distress_level = 0.8;
        let result_distressed = select_action(&candidates, &[], &config, &ctx_distressed, 1000.0);

        // Distressed result should have emotional sensitivity factor
        assert!(result_distressed
            .trace
            .active_factors
            .iter()
            .any(|f| { f.factor == "emotional_sensitivity" }));

        // If both selected, distressed utility should be lower
        if let (PolicyDecision::Act(ref calm), PolicyDecision::Act(ref distressed)) =
            (&result_calm.decision, &result_distressed.decision)
        {
            assert!(
                distressed.adjusted_utility <= calm.adjusted_utility,
                "distressed utility should be lower"
            );
        }
    }

    #[test]
    fn test_diversity_penalty() {
        let candidates = vec![make_evaluated(
            "send_reminder",
            ActionKind::Communicate,
            0.5,
            0.8,
        )];
        let config = PolicyConfig::default();
        let mut ctx = PolicyContext::default();
        // 3 consecutive communicate actions (must match ActionKind::as_str())
        ctx.recent_actions = vec![
            ("inform".to_string(), 100.0), // different kind — breaks streak
            ("communicate".to_string(), 200.0),
            ("communicate".to_string(), 300.0),
            ("communicate".to_string(), 400.0), // 3 consecutive
        ];

        let result = select_action(&candidates, &[], &config, &ctx, 1000.0);
        assert!(result
            .trace
            .active_factors
            .iter()
            .any(|f| { f.factor == "diversity_penalty" }));
    }

    #[test]
    fn test_escalate_to_llm_when_all_filtered() {
        let candidates = vec![make_evaluated(
            "send_reminder",
            ActionKind::Communicate,
            0.5,
            0.8,
        )];
        let mut config = PolicyConfig::default();
        config.selection_threshold = 10.0; // impossibly high
        config.escalate_to_llm_on_empty = true;
        let ctx = PolicyContext::default();

        let result = select_action(&candidates, &[], &config, &ctx, 1000.0);
        assert!(matches!(
            result.decision,
            PolicyDecision::EscalateToLlm { .. }
        ));
    }

    #[test]
    fn test_wait_when_escalation_disabled() {
        let candidates = vec![make_evaluated(
            "send_reminder",
            ActionKind::Communicate,
            0.5,
            0.8,
        )];
        let mut config = PolicyConfig::default();
        config.selection_threshold = 10.0; // impossibly high
        config.escalate_to_llm_on_empty = false;
        let ctx = PolicyContext::default();

        let result = select_action(&candidates, &[], &config, &ctx, 1000.0);
        assert!(matches!(result.decision, PolicyDecision::Wait { .. }));
    }

    #[test]
    fn test_ranking_preserved() {
        let candidates = vec![
            make_evaluated("send_reminder", ActionKind::Communicate, 0.7, 0.8),
            make_evaluated("risk_alert", ActionKind::Warn, 0.5, 0.8),
            make_evaluated("organize_tasks", ActionKind::Organize, 0.3, 0.8),
        ];
        let config = PolicyConfig::default();
        let ctx = PolicyContext::default();

        let result = select_action(&candidates, &[], &config, &ctx, 1000.0);
        if let PolicyDecision::Act(ref selected) = result.decision {
            // Best should be selected
            assert_eq!(selected.action.candidate.schema_name, "send_reminder");
            // Alternatives should be in order
            if selected.alternatives.len() >= 2 {
                assert!(
                    selected.alternatives[0].adjusted_utility
                        >= selected.alternatives[1].adjusted_utility
                );
            }
        }
    }

    #[test]
    fn test_hard_constraint_from_graph() {
        let mut alloc = NodeIdAllocator::new();
        let constraint_id = alloc.alloc(NodeKind::Constraint);
        let constraint_node = CognitiveNode::new(
            constraint_id,
            "No execute during focus".to_string(),
            NodePayload::Constraint(ConstraintPayload {
                description: "No execute during focus".to_string(),
                constraint_type: ConstraintType::Hard,
                condition: "no execute actions".to_string(),
                imposed_by: "user".to_string(),
            }),
        );

        let candidates = vec![
            make_evaluated("run_script", ActionKind::Execute, 0.6, 0.9),
            make_evaluated("send_reminder", ActionKind::Communicate, 0.5, 0.8),
        ];
        let nodes: Vec<&CognitiveNode> = vec![&constraint_node];
        let config = PolicyConfig::default();
        let ctx = PolicyContext::default();

        let result = select_action(&candidates, &nodes, &config, &ctx, 1000.0);

        // Execute should be rejected by hard constraint
        assert!(result.trace.rejected_candidates.iter().any(|r| {
            r.schema_name == "run_script"
                && matches!(r.rejection_reason, RejectionReason::HardConstraint { .. })
        }));

        // Communicate should still be selected
        if let PolicyDecision::Act(ref selected) = result.decision {
            assert_eq!(selected.action.candidate.schema_name, "send_reminder");
        }
    }

    #[test]
    fn test_warn_overrides_quiet_hours_when_urgent() {
        let candidates = vec![make_evaluated("risk_alert", ActionKind::Warn, 0.8, 0.9)];
        let config = PolicyConfig::default();
        let mut ctx = PolicyContext::default();
        ctx.current_hour = 23; // quiet hours

        let result = select_action(&candidates, &[], &config, &ctx, 1000.0);

        // High-urgency Warn should pass through quiet hours
        assert!(matches!(result.decision, PolicyDecision::Act(_)));
    }

    #[test]
    fn test_cognitive_load_penalty() {
        let candidates = vec![make_evaluated("run_script", ActionKind::Execute, 0.5, 0.9)];
        let config = PolicyConfig::default(); // threshold 0.8
        let mut ctx = PolicyContext::default();
        ctx.cognitive_load = 0.95; // well above threshold

        let result = select_action(&candidates, &[], &config, &ctx, 1000.0);
        assert!(result
            .trace
            .active_factors
            .iter()
            .any(|f| { f.factor == "cognitive_load" }));
    }

    #[test]
    fn test_trace_completeness() {
        let candidates = vec![
            make_evaluated("send_reminder", ActionKind::Communicate, 0.5, 0.8),
            make_evaluated("run_script", ActionKind::Execute, 0.3, 0.50), // below floor
        ];
        let config = PolicyConfig::default();
        let ctx = PolicyContext::default();

        let result = select_action(&candidates, &[], &config, &ctx, 1000.0);

        // Trace should account for all candidates
        let accounted = result.trace.passed_hard_filter + result.trace.rejected_candidates.len();
        assert_eq!(
            accounted, result.total_input,
            "all candidates should be accounted for in trace"
        );
        assert!(result.trace.execution_time_us > 0);
    }
}
