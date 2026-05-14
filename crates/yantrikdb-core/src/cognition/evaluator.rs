//! CK-1.8: Utility Scoring + Bounded Forward Simulation
//!
//! Ranks action candidates by computing expected utility — the anticipated
//! value of taking an action, considering both benefits and costs. Uses a
//! bounded forward simulation (1-step lookahead) to estimate how the
//! cognitive graph would change if an action were taken.
//!
//! ## Utility Model
//!
//! ```text
//! U(action) = Σ(effect.probability × effect.utility × context_multiplier)
//!           - action_kind.base_cost
//!           - timing_penalty
//!           + intent_alignment_bonus
//!           + preference_alignment_bonus
//! ```
//!
//! ## Bounded Forward Simulation
//!
//! For each candidate, we simulate 1 step forward:
//! 1. Apply expected effects to a snapshot of relevant nodes
//! 2. Check if any goals advance, needs are met, or constraints violated
//! 3. Compute delta-utility from the simulated state change
//!
//! This is bounded to O(candidates × effects × nodes) — no recursion.

use serde::{Deserialize, Serialize};

use super::action::ActionCandidate;
use super::state::*;

// ── Configuration ──

/// Configuration for utility evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluatorConfig {
    /// Weight for effect-based utility [0.0, 1.0].
    pub effect_weight: f64,

    /// Weight for intent alignment bonus [0.0, 1.0].
    pub intent_weight: f64,

    /// Weight for preference alignment [0.0, 1.0].
    pub preference_weight: f64,

    /// Weight for forward simulation delta [0.0, 1.0].
    pub simulation_weight: f64,

    /// Penalty multiplier for acting at a bad time (e.g., user is busy).
    /// 0.0 = no timing penalty, 1.0 = full penalty.
    pub timing_penalty_scale: f64,

    /// How much to penalize high-cost action kinds.
    pub cost_penalty_scale: f64,

    /// Minimum utility to include in results (filter noise).
    pub min_utility: f64,

    /// Maximum candidates to return.
    pub max_results: usize,
}

impl Default for EvaluatorConfig {
    fn default() -> Self {
        Self {
            effect_weight: 0.35,
            intent_weight: 0.25,
            preference_weight: 0.15,
            simulation_weight: 0.25,
            timing_penalty_scale: 0.5,
            cost_penalty_scale: 1.0,
            min_utility: -0.5, // allow slightly negative (Abstain is near 0)
            max_results: 10,
        }
    }
}

// ── Evaluated Action ──

/// An action candidate with computed utility score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluatedAction {
    /// The original candidate.
    pub candidate: ActionCandidate,

    /// Computed expected utility from effects.
    pub effect_utility: f64,

    /// Base cost of the action kind.
    pub base_cost: f64,

    /// Timing penalty (0 = good timing, higher = bad timing).
    pub timing_penalty: f64,

    /// Bonus for aligning with the intent.
    pub intent_alignment: f64,

    /// Bonus for aligning with user preferences.
    pub preference_alignment: f64,

    /// Utility delta from forward simulation.
    pub simulation_delta: f64,

    /// Final composite utility score.
    pub utility: f64,

    /// Confidence in this utility estimate [0.0, 1.0].
    /// Lower when effects have low probability or sparse evidence.
    pub confidence: f64,
}

/// Result of evaluating a set of action candidates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// Evaluated actions sorted by utility descending.
    pub actions: Vec<EvaluatedAction>,

    /// Total candidates evaluated.
    pub total_evaluated: usize,

    /// Number filtered by min_utility.
    pub filtered_count: usize,

    /// Duration in microseconds.
    pub duration_us: u64,
}

// ── Effect Utility Computation ──

/// Compute expected utility from an action schema's effects.
///
/// EU = Σ(probability × utility) for all effects.
fn compute_effect_utility(candidate: &ActionCandidate) -> f64 {
    // We don't have the full schema effects in ActionCandidate,
    // so we use the schema's built-in info via lookup.
    if let Some(schema) = super::action::lookup_builtin(&candidate.schema_name) {
        schema
            .effects
            .iter()
            .map(|e| e.probability * e.utility)
            .sum()
    } else {
        // For persisted/custom schemas, use relevance_score as proxy
        candidate.relevance_score * 0.5
    }
}

/// Compute confidence in the utility estimate.
///
/// Higher when:
/// - All required preconditions are satisfied
/// - Effects have high probability
/// - Schema has been executed many times
fn compute_confidence(candidate: &ActionCandidate) -> f64 {
    let precondition_confidence = if candidate.total_required > 0 {
        candidate.satisfied_required as f64 / candidate.total_required as f64
    } else {
        1.0
    };

    // Look up schema for execution count
    let experience_confidence =
        if let Some(schema) = super::action::lookup_builtin(&candidate.schema_name) {
            // Confidence grows with execution count, plateaus at 20
            (schema.execution_count as f64 / 20.0).min(1.0)
        } else {
            0.5 // neutral for unknown schemas
        };

    // Average effect probability as confidence factor
    let effect_confidence = if let Some(schema) =
        super::action::lookup_builtin(&candidate.schema_name)
    {
        if schema.effects.is_empty() {
            0.5
        } else {
            schema.effects.iter().map(|e| e.probability).sum::<f64>() / schema.effects.len() as f64
        }
    } else {
        0.5
    };

    // Weighted average
    0.50 * precondition_confidence + 0.25 * effect_confidence + 0.25 * experience_confidence
}

// ── Timing Analysis ──

/// Compute timing penalty based on context nodes.
///
/// Higher penalty when:
/// - User has high-urgency tasks they're working on (don't interrupt)
/// - Recent high-valence episodes suggest the user is focused
/// - Time of day suggests the user is sleeping/busy
fn compute_timing_penalty(candidate: &ActionCandidate, nodes: &[&CognitiveNode]) -> f64 {
    // If action is Abstain, no timing penalty
    if candidate.action_kind == ActionKind::Abstain {
        return 0.0;
    }

    let mut penalty = 0.0;

    // Check for active high-urgency tasks (suggests user is focused)
    let max_task_urgency = nodes
        .iter()
        .filter(|n| n.id.kind() == NodeKind::Task)
        .map(|n| n.attrs.urgency)
        .fold(0.0_f64, f64::max);

    if max_task_urgency > 0.7 {
        // User is in deep work — penalize disruptive actions
        let disruptiveness = candidate.action_kind.base_cost();
        penalty += disruptiveness * max_task_urgency;
    }

    // Check conversation thread activity (don't interrupt active conversations)
    let has_active_thread = nodes
        .iter()
        .any(|n| n.id.kind() == NodeKind::ConversationThread && n.attrs.activation > 0.5);
    if has_active_thread && candidate.action_kind != ActionKind::Communicate {
        penalty += 0.1;
    }

    penalty.min(0.5) // cap penalty at 0.5
}

// ── Preference Alignment ──

/// Compute how well an action aligns with user preferences.
///
/// Checks if any preference nodes are connected to the action's context.
fn compute_preference_alignment(
    candidate: &ActionCandidate,
    nodes: &[&CognitiveNode],
    edges: &[CognitiveEdge],
) -> f64 {
    let preferences: Vec<&CognitiveNode> = nodes
        .iter()
        .filter(|n| n.id.kind() == NodeKind::Preference)
        .copied()
        .collect();

    if preferences.is_empty() {
        return 0.0; // no preferences to align with
    }

    // Check for Prefers/Avoids edges involving the candidate's bound nodes
    let bound_nodes: std::collections::HashSet<u32> = candidate
        .precondition_bindings
        .iter()
        .filter_map(|b| b.bound_node.map(|n| n.to_raw()))
        .collect();

    if bound_nodes.is_empty() {
        return 0.0;
    }

    let mut alignment = 0.0;
    let mut count = 0;

    for edge in edges {
        let src = edge.src.to_raw();
        let dst = edge.dst.to_raw();

        // Preference → bound node via Prefers edge
        if edge.kind == CognitiveEdgeKind::Prefers
            && preferences.iter().any(|p| p.id.to_raw() == src)
            && bound_nodes.contains(&dst)
        {
            alignment += edge.weight.abs();
            count += 1;
        }

        // Preference → bound node via Avoids edge (negative alignment)
        if edge.kind == CognitiveEdgeKind::Avoids
            && preferences.iter().any(|p| p.id.to_raw() == src)
            && bound_nodes.contains(&dst)
        {
            alignment -= edge.weight.abs();
            count += 1;
        }
    }

    if count > 0 {
        (alignment / count as f64).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

// ── Forward Simulation ──

/// Bounded 1-step forward simulation.
///
/// Estimates the utility delta if this action were taken:
/// - Goals that would advance → positive delta
/// - Needs that would be satisfied → positive delta
/// - Constraints that would be violated → negative delta
/// - Risks that would be mitigated → positive delta
fn simulate_forward(candidate: &ActionCandidate, nodes: &[&CognitiveNode]) -> f64 {
    let mut delta = 0.0;

    // Look up schema effects for simulation
    let effects = if let Some(schema) = super::action::lookup_builtin(&candidate.schema_name) {
        schema.effects
    } else {
        return 0.0; // can't simulate unknown schemas
    };

    // For each effect, estimate impact on the cognitive graph
    for effect in &effects {
        let expected_value = effect.probability * effect.utility;

        // Goal advancement: if effect is positive and goals exist
        if expected_value > 0.0 {
            let active_goals = nodes.iter()
                .filter(|n| n.id.kind() == NodeKind::Goal)
                .filter(|n| matches!(&n.payload, NodePayload::Goal(g) if g.status == GoalStatus::Active))
                .count();

            if active_goals > 0 {
                // Positive effects on an active-goal context amplify delta
                delta += expected_value * (1.0 + 0.1 * active_goals as f64).min(1.5);
            } else {
                delta += expected_value;
            }
        }

        // Negative effects: risk of harm
        if expected_value < 0.0 {
            delta += expected_value; // already negative
        }
    }

    // Constraint violation check: if action is Execute or high-cost,
    // and constraints exist, apply a dampening factor
    let constraint_count = nodes
        .iter()
        .filter(|n| n.id.kind() == NodeKind::Constraint)
        .count();

    if constraint_count > 0 && candidate.action_kind.base_cost() >= 0.3 {
        delta -= 0.05 * constraint_count as f64;
    }

    // Need satisfaction: if action addresses a need, bonus
    let unmet_needs = nodes
        .iter()
        .filter(|n| n.id.kind() == NodeKind::Need)
        .filter(|n| n.attrs.urgency > 0.3)
        .count();

    if unmet_needs > 0 && candidate.action_kind != ActionKind::Abstain {
        delta += 0.05 * unmet_needs as f64;
    }

    delta.clamp(-1.0, 1.0)
}

// ── Main Evaluation Pipeline ──

/// Evaluate a set of action candidates and compute utility scores.
///
/// This is the main entry point for CK-1.8.
pub fn evaluate_candidates(
    candidates: &[ActionCandidate],
    nodes: &[&CognitiveNode],
    edges: &[CognitiveEdge],
    config: &EvaluatorConfig,
) -> EvaluationResult {
    let start = std::time::Instant::now();
    let total_evaluated = candidates.len();

    let mut evaluated: Vec<EvaluatedAction> = candidates
        .iter()
        .map(|candidate| {
            let effect_utility = compute_effect_utility(candidate);
            let base_cost = candidate.action_kind.base_cost();
            let timing_penalty = compute_timing_penalty(candidate, nodes);
            let intent_alignment = candidate.relevance_score; // from CK-1.7
            let preference_alignment = compute_preference_alignment(candidate, nodes, edges);
            let simulation_delta = simulate_forward(candidate, nodes);
            let confidence = compute_confidence(candidate);

            // Composite utility
            let utility = config.effect_weight * effect_utility
                - config.cost_penalty_scale * base_cost
                - config.timing_penalty_scale * timing_penalty
                + config.intent_weight * intent_alignment
                + config.preference_weight * preference_alignment
                + config.simulation_weight * simulation_delta;

            EvaluatedAction {
                candidate: candidate.clone(),
                effect_utility,
                base_cost,
                timing_penalty,
                intent_alignment,
                preference_alignment,
                simulation_delta,
                utility,
                confidence,
            }
        })
        .collect();

    // Sort by utility descending
    evaluated.sort_by(|a, b| {
        b.utility
            .partial_cmp(&a.utility)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Filter by min_utility
    let filtered: Vec<EvaluatedAction> = evaluated
        .into_iter()
        .filter(|e| e.utility >= config.min_utility)
        .take(config.max_results)
        .collect();

    let filtered_count = total_evaluated - filtered.len();

    EvaluationResult {
        actions: filtered,
        total_evaluated,
        filtered_count,
        duration_us: start.elapsed().as_micros() as u64,
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::super::action::{generate_candidates, ActionConfig, PreconditionBinding};
    use super::super::intent::{IntentSource, ScoredIntent};
    use super::*;

    fn make_candidate(name: &str, kind: ActionKind, relevance: f64) -> ActionCandidate {
        ActionCandidate {
            schema_name: name.to_string(),
            action_kind: kind,
            description: format!("{name}: test"),
            source_intent: "test intent".to_string(),
            precondition_bindings: vec![],
            satisfied_required: 0,
            total_required: 0,
            satisfied_soft: 0,
            relevance_score: relevance,
            schema_node: None,
        }
    }

    fn make_goal(alloc: &mut NodeIdAllocator, desc: &str, urgency: f64) -> CognitiveNode {
        let id = alloc.alloc(NodeKind::Goal);
        let mut node = CognitiveNode::new(
            id,
            desc.to_string(),
            NodePayload::Goal(GoalPayload {
                description: desc.to_string(),
                status: GoalStatus::Active,
                progress: 0.3,
                deadline: None,
                priority: Priority::High,
                parent_goal: None,
                completion_criteria: "Done".to_string(),
            }),
        );
        node.attrs.urgency = urgency;
        node
    }

    #[test]
    fn test_effect_utility_computation() {
        let candidate = make_candidate("send_reminder", ActionKind::Communicate, 0.5);
        let eu = compute_effect_utility(&candidate);
        // send_reminder: (0.85 * 0.6) + (0.30 * -0.2) = 0.51 - 0.06 = 0.45
        assert!((eu - 0.45).abs() < 0.01, "expected ~0.45, got {eu}");
    }

    #[test]
    fn test_abstain_utility() {
        let candidate = make_candidate("abstain", ActionKind::Abstain, 0.3);
        let eu = compute_effect_utility(&candidate);
        // abstain: (0.95 * 0.1) = 0.095
        assert!(eu > 0.0, "abstain should have small positive utility");
        assert!(eu < 0.2, "abstain utility should be small");
    }

    #[test]
    fn test_timing_penalty_during_focus() {
        let mut alloc = NodeIdAllocator::new();
        let task_id = alloc.alloc(NodeKind::Task);
        let mut task = CognitiveNode::new(
            task_id,
            "Urgent task".to_string(),
            NodePayload::Task(TaskPayload {
                description: "Urgent work".to_string(),
                status: TaskStatus::InProgress,
                goal_id: None,
                deadline: None,
                priority: Priority::Critical,
                estimated_minutes: None,
                prerequisites: vec![],
            }),
        );
        task.attrs.urgency = 0.9;

        let nodes: Vec<&CognitiveNode> = vec![&task];
        let disruptive = make_candidate("send_reminder", ActionKind::Communicate, 0.5);
        let penalty = compute_timing_penalty(&disruptive, &nodes);

        assert!(
            penalty > 0.0,
            "should have timing penalty during focus work"
        );

        let abstain = make_candidate("abstain", ActionKind::Abstain, 0.3);
        let abstain_penalty = compute_timing_penalty(&abstain, &nodes);
        assert_eq!(
            abstain_penalty, 0.0,
            "abstain should have no timing penalty"
        );
    }

    #[test]
    fn test_evaluate_candidates_basic() {
        let candidates = vec![
            make_candidate("send_reminder", ActionKind::Communicate, 0.7),
            make_candidate("abstain", ActionKind::Abstain, 0.3),
            make_candidate("risk_alert", ActionKind::Warn, 0.5),
        ];

        let config = EvaluatorConfig::default();
        let result = evaluate_candidates(&candidates, &[], &[], &config);

        assert_eq!(result.total_evaluated, 3);
        assert!(!result.actions.is_empty());

        // Should be sorted by utility descending
        for w in result.actions.windows(2) {
            assert!(w[0].utility >= w[1].utility, "should be sorted descending");
        }
    }

    #[test]
    fn test_evaluate_with_goals_boosts_utility() {
        let mut alloc = NodeIdAllocator::new();
        let goal = make_goal(&mut alloc, "Ship feature", 0.8);

        let candidates = vec![make_candidate(
            "send_reminder",
            ActionKind::Communicate,
            0.7,
        )];

        let config = EvaluatorConfig::default();

        // Without goals
        let result_no_goals = evaluate_candidates(&candidates, &[], &[], &config);

        // With goals
        let nodes: Vec<&CognitiveNode> = vec![&goal];
        let result_with_goals = evaluate_candidates(&candidates, &nodes, &[], &config);

        // Forward simulation should give a boost when goals exist
        assert!(
            result_with_goals.actions[0].simulation_delta
                >= result_no_goals.actions[0].simulation_delta,
            "goals should boost forward simulation delta"
        );
    }

    #[test]
    fn test_confidence_computation() {
        let full_satisfaction = ActionCandidate {
            schema_name: "send_reminder".to_string(),
            action_kind: ActionKind::Communicate,
            description: "test".to_string(),
            source_intent: "test".to_string(),
            precondition_bindings: vec![PreconditionBinding {
                description: "test".to_string(),
                required: true,
                satisfied: true,
                bound_node: None,
            }],
            satisfied_required: 1,
            total_required: 1,
            satisfied_soft: 0,
            relevance_score: 0.5,
            schema_node: None,
        };

        let partial_satisfaction = ActionCandidate {
            satisfied_required: 0,
            total_required: 1,
            ..full_satisfaction.clone()
        };

        let full_conf = compute_confidence(&full_satisfaction);
        let partial_conf = compute_confidence(&partial_satisfaction);

        assert!(
            full_conf > partial_conf,
            "full precondition satisfaction should yield higher confidence"
        );
    }

    #[test]
    fn test_min_utility_filter() {
        let candidates = vec![
            make_candidate("send_reminder", ActionKind::Communicate, 0.7),
            make_candidate("abstain", ActionKind::Abstain, 0.1),
        ];

        let mut config = EvaluatorConfig::default();
        config.min_utility = 0.3; // high threshold

        let result = evaluate_candidates(&candidates, &[], &[], &config);

        // Some candidates may be filtered out by high threshold
        assert!(result.actions.len() <= candidates.len());
        for action in &result.actions {
            assert!(
                action.utility >= config.min_utility,
                "all returned actions should be above min_utility"
            );
        }
    }

    #[test]
    fn test_high_cost_actions_penalized() {
        let low_cost = make_candidate("abstain", ActionKind::Abstain, 0.5);
        let high_cost = make_candidate("decay_stale_activations", ActionKind::Execute, 0.5);

        let config = EvaluatorConfig::default();
        let result = evaluate_candidates(&[low_cost, high_cost], &[], &[], &config);

        if result.actions.len() >= 2 {
            let abstain = result
                .actions
                .iter()
                .find(|a| a.candidate.schema_name == "abstain");
            let execute = result
                .actions
                .iter()
                .find(|a| a.candidate.schema_name == "decay_stale_activations");

            if let (Some(a), Some(e)) = (abstain, execute) {
                assert!(
                    a.base_cost < e.base_cost,
                    "Execute should have higher base cost than Abstain"
                );
            }
        }
    }
}
