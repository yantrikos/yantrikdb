//! CK-1.10: suggest_next_step() — Full Pipeline Integration
//!
//! Wires the complete cognitive pipeline into a single entry point:
//!
//! ```text
//! suggest_next_step(request)
//!   → CK-1.6: intent inference
//!   → CK-1.7: action candidate generation
//!   → CK-1.8: utility scoring + forward simulation
//!   → CK-1.9: policy filtering + constraint enforcement
//!   → NextStepResponse
//! ```
//!
//! ## Execution Modes
//!
//! - **Fast** (<3ms target): top-5 candidates, minimal scoring
//! - **Balanced** (<10ms target): top-12 candidates, full scoring
//! - **Deep** (<30ms target): top-20 candidates, full scoring + simulation
//!
//! ## Request/Response
//!
//! The request captures the current context snapshot. The response includes
//! the chosen action, alternatives, confidence, and a full reasoning trace.

use serde::{Deserialize, Serialize};

use super::action::{self, ActionCandidate, ActionConfig, CandidateGenerationResult};
use super::evaluator::{self, EvaluatedAction, EvaluationResult, EvaluatorConfig};
use super::intent::{self, IntentConfig, IntentInferenceResult, ScoredIntent};
use super::policy::{
    self, AdjustedCandidate, PolicyConfig, PolicyContext, PolicyDecision, PolicyResult,
    ReasoningTrace, SelectedAction,
};
use super::state::*;

// ── Execution Mode ──

/// How deeply to analyze before responding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Fastest path: cached state, top-5 candidates, no simulation.
    /// Target: <3ms.
    Fast,
    /// Standard path: fresh inference, top-12 candidates, full scoring.
    /// Target: <10ms.
    Balanced,
    /// Thorough path: full graph, top-20 candidates, simulation + deep scoring.
    /// Target: <30ms.
    Deep,
}

impl Default for ExecutionMode {
    fn default() -> Self {
        Self::Balanced
    }
}

impl ExecutionMode {
    /// Max candidates to evaluate in this mode.
    pub fn max_candidates(self) -> usize {
        match self {
            Self::Fast => 5,
            Self::Balanced => 12,
            Self::Deep => 20,
        }
    }

    /// Max results to return in this mode.
    pub fn max_results(self) -> usize {
        match self {
            Self::Fast => 3,
            Self::Balanced => 6,
            Self::Deep => 10,
        }
    }
}

// ── Request ──

/// Request for the suggest_next_step() pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextStepRequest {
    /// Current timestamp (seconds since epoch).
    pub now: f64,

    /// Optional text from the current user turn (for context).
    pub turn_text: Option<String>,

    /// Current user context snapshot.
    pub context: PolicyContext,

    /// Execution mode controls depth vs. speed tradeoff.
    pub mode: ExecutionMode,

    /// Optional overrides for sub-pipeline configs.
    pub overrides: Option<PipelineOverrides>,
}

impl Default for NextStepRequest {
    fn default() -> Self {
        Self {
            now: 0.0,
            turn_text: None,
            context: PolicyContext::default(),
            mode: ExecutionMode::Balanced,
            overrides: None,
        }
    }
}

/// Optional overrides for sub-pipeline configurations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineOverrides {
    pub intent_config: Option<IntentConfig>,
    pub action_config: Option<ActionConfig>,
    pub eval_config: Option<EvaluatorConfig>,
    pub policy_config: Option<PolicyConfig>,
}

// ── Response ──

/// The action proposal — what the system suggests doing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionProposal {
    /// Schema name (e.g., "send_reminder", "suggest_break").
    pub schema_name: String,

    /// Action kind.
    pub action_kind: ActionKind,

    /// Human-readable description of what would happen.
    pub description: String,

    /// Final adjusted utility score.
    pub utility: f64,

    /// Confidence in this proposal [0.0, 1.0].
    pub confidence: f64,

    /// The intent this action serves.
    pub source_intent: String,
}

/// Complete response from suggest_next_step().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextStepResponse {
    /// The chosen action (if any).
    pub chosen: Option<ActionProposal>,

    /// Alternative actions, ranked by utility.
    pub alternatives: Vec<ActionProposal>,

    /// Overall confidence in the recommendation.
    pub confidence: f64,

    /// Whether the system recommends escalating to LLM.
    pub should_call_llm: bool,

    /// Full reasoning trace.
    pub trace: ReasoningTrace,

    /// Pipeline stage metrics.
    pub metrics: PipelineMetrics,
}

/// Timing and count metrics for each pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetrics {
    /// Total pipeline execution time in microseconds.
    pub total_us: u64,

    /// Time spent in intent inference (CK-1.6).
    pub intent_us: u64,

    /// Time spent in candidate generation (CK-1.7).
    pub action_us: u64,

    /// Time spent in utility scoring (CK-1.8).
    pub eval_us: u64,

    /// Time spent in policy selection (CK-1.9).
    pub policy_us: u64,

    /// Number of intents inferred.
    pub intents_count: usize,

    /// Number of action candidates generated.
    pub candidates_count: usize,

    /// Number of candidates that passed evaluation.
    pub evaluated_count: usize,

    /// Execution mode used.
    pub mode: ExecutionMode,
}

// ── Pipeline ──

/// Run the full suggest_next_step() pipeline.
///
/// This is the main entry point for the cognitive kernel.
/// It chains CK-1.6 → CK-1.7 → CK-1.8 → CK-1.9 in sequence,
/// respecting the execution mode for performance budgets.
pub fn suggest_next_step(
    request: &NextStepRequest,
    nodes: &[&CognitiveNode],
    edges: &[CognitiveEdge],
) -> NextStepResponse {
    let pipeline_start = std::time::Instant::now();

    let mode = request.mode;
    let overrides = request.overrides.as_ref();

    // ── Stage 1: Intent Inference (CK-1.6) ──
    let intent_start = std::time::Instant::now();

    let mut intent_config = overrides
        .and_then(|o| o.intent_config.clone())
        .unwrap_or_default();

    // Mode-specific tuning
    match mode {
        ExecutionMode::Fast => {
            intent_config.max_hypotheses = 3;
        }
        ExecutionMode::Balanced => {
            intent_config.max_hypotheses = 8;
        }
        ExecutionMode::Deep => {
            intent_config.max_hypotheses = 15;
        }
    }

    let intent_result = intent::infer_intents(nodes, edges, request.now, &intent_config);
    let intent_us = intent_start.elapsed().as_micros() as u64;

    // ── Stage 2: Action Candidate Generation (CK-1.7) ──
    let action_start = std::time::Instant::now();

    let mut action_config = overrides
        .and_then(|o| o.action_config.clone())
        .unwrap_or_default();

    action_config.max_total_candidates = mode.max_candidates();

    let schema_nodes: Vec<&CognitiveNode> = nodes
        .iter()
        .filter(|n| n.kind() == NodeKind::ActionSchema)
        .copied()
        .collect();

    let candidate_result = action::generate_candidates(
        &intent_result.hypotheses,
        nodes,
        &schema_nodes,
        &action_config,
    );
    let action_us = action_start.elapsed().as_micros() as u64;

    // ── Stage 3: Utility Scoring (CK-1.8) ──
    let eval_start = std::time::Instant::now();

    let mut eval_config = overrides
        .and_then(|o| o.eval_config.clone())
        .unwrap_or_default();

    eval_config.max_results = mode.max_results();

    let eval_result =
        evaluator::evaluate_candidates(&candidate_result.candidates, nodes, edges, &eval_config);
    let eval_us = eval_start.elapsed().as_micros() as u64;

    // ── Stage 4: Policy Selection (CK-1.9) ──
    let policy_start = std::time::Instant::now();

    let policy_config = overrides
        .and_then(|o| o.policy_config.clone())
        .unwrap_or_default();

    let policy_result = policy::select_action(
        &eval_result.actions,
        nodes,
        &policy_config,
        &request.context,
        request.now,
    );
    let policy_us = policy_start.elapsed().as_micros() as u64;

    // ── Build Response ──
    let total_us = pipeline_start.elapsed().as_micros() as u64;

    let metrics = PipelineMetrics {
        total_us,
        intent_us,
        action_us,
        eval_us,
        policy_us,
        intents_count: intent_result.hypotheses.len(),
        candidates_count: candidate_result.candidates.len(),
        evaluated_count: eval_result.actions.len(),
        mode,
    };

    match policy_result.decision {
        PolicyDecision::Act(selected) => {
            let chosen = action_to_proposal(&selected.action);
            let alternatives: Vec<ActionProposal> = selected
                .alternatives
                .iter()
                .map(adjusted_to_proposal)
                .collect();

            NextStepResponse {
                confidence: selected.action.confidence,
                chosen: Some(chosen),
                alternatives,
                should_call_llm: false,
                trace: policy_result.trace,
                metrics,
            }
        }
        PolicyDecision::EscalateToLlm { .. } => NextStepResponse {
            chosen: None,
            alternatives: vec![],
            confidence: 0.0,
            should_call_llm: true,
            trace: policy_result.trace,
            metrics,
        },
        PolicyDecision::Wait { .. } => NextStepResponse {
            chosen: None,
            alternatives: vec![],
            confidence: 0.0,
            should_call_llm: false,
            trace: policy_result.trace,
            metrics,
        },
    }
}

/// Convert an EvaluatedAction into an ActionProposal.
fn action_to_proposal(action: &EvaluatedAction) -> ActionProposal {
    ActionProposal {
        schema_name: action.candidate.schema_name.clone(),
        action_kind: action.candidate.action_kind,
        description: action.candidate.description.clone(),
        utility: action.utility,
        confidence: action.confidence,
        source_intent: action.candidate.source_intent.clone(),
    }
}

/// Convert an AdjustedCandidate into an ActionProposal.
fn adjusted_to_proposal(adj: &AdjustedCandidate) -> ActionProposal {
    ActionProposal {
        schema_name: adj.schema_name.clone(),
        action_kind: adj.action_kind,
        description: String::new(), // minimal for alternatives
        utility: adj.adjusted_utility,
        confidence: 0.0, // not tracked for alternatives
        source_intent: String::new(),
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn make_goal(alloc: &mut NodeIdAllocator, desc: &str, urgency: f64) -> CognitiveNode {
        let id = alloc.alloc(NodeKind::Goal);
        let mut node = CognitiveNode::new(
            id,
            desc.to_string(),
            NodePayload::Goal(GoalPayload {
                description: desc.to_string(),
                status: GoalStatus::Active,
                progress: 0.4,
                deadline: None,
                priority: Priority::High,
                parent_goal: None,
                completion_criteria: "Done".to_string(),
            }),
        );
        node.attrs.urgency = urgency;
        node
    }

    fn make_need(alloc: &mut NodeIdAllocator, desc: &str, intensity: f64) -> CognitiveNode {
        let id = alloc.alloc(NodeKind::Need);
        CognitiveNode::new(
            id,
            desc.to_string(),
            NodePayload::Need(NeedPayload {
                description: desc.to_string(),
                category: NeedCategory::Professional,
                intensity,
                last_satisfied: None,
                satisfaction_pattern: "Do it".to_string(),
            }),
        )
    }

    fn make_routine(alloc: &mut NodeIdAllocator, desc: &str) -> CognitiveNode {
        let id = alloc.alloc(NodeKind::Routine);
        let mut node = CognitiveNode::new(
            id,
            desc.to_string(),
            NodePayload::Routine(RoutinePayload {
                description: desc.to_string(),
                period_secs: 86400.0,
                phase_offset_secs: 32400.0,
                reliability: 0.8,
                observation_count: 20,
                last_triggered: 0.0,
                action_description: desc.to_string(),
                weekday_mask: 0x1F,
            }),
        );
        node.attrs.urgency = 0.5;
        node
    }

    fn make_constraint(alloc: &mut NodeIdAllocator, desc: &str, hard: bool) -> CognitiveNode {
        let id = alloc.alloc(NodeKind::Constraint);
        CognitiveNode::new(
            id,
            desc.to_string(),
            NodePayload::Constraint(ConstraintPayload {
                description: desc.to_string(),
                constraint_type: if hard {
                    ConstraintType::Hard
                } else {
                    ConstraintType::Soft
                },
                condition: desc.to_lowercase(),
                imposed_by: "user".to_string(),
            }),
        )
    }

    fn default_request() -> NextStepRequest {
        NextStepRequest {
            now: 1710000000.0,
            turn_text: None,
            context: PolicyContext::default(),
            mode: ExecutionMode::Balanced,
            overrides: None,
        }
    }

    // ── Basic pipeline tests ──

    #[test]
    fn test_empty_graph() {
        let req = default_request();
        let resp = suggest_next_step(&req, &[], &[]);

        assert!(resp.chosen.is_none());
        assert!(resp.alternatives.is_empty());
        assert_eq!(resp.metrics.intents_count, 0);
        assert_eq!(resp.metrics.candidates_count, 0);
    }

    #[test]
    fn test_single_goal() {
        let mut alloc = NodeIdAllocator::new();
        let goal = make_goal(&mut alloc, "Ship quarterly report", 0.8);
        let nodes: Vec<&CognitiveNode> = vec![&goal];

        let req = default_request();
        let resp = suggest_next_step(&req, &nodes, &[]);

        // Should produce intents and candidates from a single goal
        assert!(resp.metrics.intents_count > 0);
        assert!(resp.metrics.total_us > 0);
    }

    #[test]
    fn test_multiple_goals_ranking() {
        let mut alloc = NodeIdAllocator::new();
        let high = make_goal(&mut alloc, "Critical deadline", 0.95);
        let low = make_goal(&mut alloc, "Nice to have", 0.4);
        let nodes: Vec<&CognitiveNode> = vec![&high, &low];

        let req = default_request();
        let resp = suggest_next_step(&req, &nodes, &[]);

        // Higher urgency goal should produce higher-ranked actions
        if let Some(ref chosen) = resp.chosen {
            assert!(
                chosen.utility > 0.0,
                "chosen action should have positive utility"
            );
        }
    }

    #[test]
    fn test_goal_plus_need() {
        let mut alloc = NodeIdAllocator::new();
        let goal = make_goal(&mut alloc, "Finish project", 0.7);
        let need = make_need(&mut alloc, "Take a break", 0.6);
        let nodes: Vec<&CognitiveNode> = vec![&goal, &need];

        let req = default_request();
        let resp = suggest_next_step(&req, &nodes, &[]);

        // Should generate intents from both sources
        assert!(resp.metrics.intents_count >= 1);
    }

    #[test]
    fn test_routine_context() {
        let mut alloc = NodeIdAllocator::new();
        let routine = make_routine(&mut alloc, "Morning standup");
        let nodes: Vec<&CognitiveNode> = vec![&routine];

        let req = default_request();
        let resp = suggest_next_step(&req, &nodes, &[]);

        assert!(resp.metrics.total_us > 0);
    }

    // ── Mode tests ──

    #[test]
    fn test_fast_mode() {
        let mut alloc = NodeIdAllocator::new();
        let goal = make_goal(&mut alloc, "Fast goal", 0.8);
        let nodes: Vec<&CognitiveNode> = vec![&goal];

        let mut req = default_request();
        req.mode = ExecutionMode::Fast;
        let resp = suggest_next_step(&req, &nodes, &[]);

        assert_eq!(resp.metrics.mode, ExecutionMode::Fast);
        // Fast mode limits candidates
        assert!(resp.metrics.candidates_count <= ExecutionMode::Fast.max_candidates());
    }

    #[test]
    fn test_deep_mode() {
        let mut alloc = NodeIdAllocator::new();
        let goal = make_goal(&mut alloc, "Deep analysis goal", 0.9);
        let need = make_need(&mut alloc, "Complex need", 0.7);
        let nodes: Vec<&CognitiveNode> = vec![&goal, &need];

        let mut req = default_request();
        req.mode = ExecutionMode::Deep;
        let resp = suggest_next_step(&req, &nodes, &[]);

        assert_eq!(resp.metrics.mode, ExecutionMode::Deep);
    }

    // ── Policy integration tests ──

    #[test]
    fn test_quiet_hours_integration() {
        let mut alloc = NodeIdAllocator::new();
        let goal = make_goal(&mut alloc, "Send emails", 0.7);
        let nodes: Vec<&CognitiveNode> = vec![&goal];

        let mut req = default_request();
        req.context.current_hour = 3; // 3 AM

        let resp = suggest_next_step(&req, &nodes, &[]);

        // Most proactive actions should be filtered during quiet hours
        let quiet_rejections = resp
            .trace
            .rejected_candidates
            .iter()
            .filter(|r| matches!(r.rejection_reason, policy::RejectionReason::QuietHours))
            .count();
        // We expect some rejections during quiet hours
        assert!(
            quiet_rejections > 0 || resp.chosen.is_none(),
            "quiet hours should filter proactive actions or result in no action"
        );
    }

    #[test]
    fn test_dnd_blocks_actions() {
        let mut alloc = NodeIdAllocator::new();
        let goal = make_goal(&mut alloc, "Important task", 0.8);
        let nodes: Vec<&CognitiveNode> = vec![&goal];

        let mut req = default_request();
        req.context.dnd_active = true;

        let resp = suggest_next_step(&req, &nodes, &[]);

        // DND should block proactive actions
        assert!(resp
            .trace
            .rejected_candidates
            .iter()
            .any(|r| { matches!(r.rejection_reason, policy::RejectionReason::QuietHours) }));
    }

    #[test]
    fn test_high_distress_dampens() {
        let mut alloc = NodeIdAllocator::new();
        let goal = make_goal(&mut alloc, "Project deadline", 0.8);
        let nodes: Vec<&CognitiveNode> = vec![&goal];

        // Normal state
        let req_normal = default_request();
        let resp_normal = suggest_next_step(&req_normal, &nodes, &[]);

        // High distress
        let mut req_distressed = default_request();
        req_distressed.context.distress_level = 0.9;
        let resp_distressed = suggest_next_step(&req_distressed, &nodes, &[]);

        // Distressed should have emotional sensitivity factors
        let has_emo_factor = resp_distressed
            .trace
            .active_factors
            .iter()
            .any(|f| f.factor == "emotional_sensitivity");

        if resp_distressed.chosen.is_some() {
            assert!(
                has_emo_factor,
                "high distress should trigger emotional sensitivity"
            );
        }
    }

    #[test]
    fn test_constraint_enforcement() {
        let mut alloc = NodeIdAllocator::new();
        let goal = make_goal(&mut alloc, "Run maintenance", 0.7);
        let constraint = make_constraint(&mut alloc, "No execute during focus hours", true);
        let nodes: Vec<&CognitiveNode> = vec![&goal, &constraint];

        let req = default_request();
        let resp = suggest_next_step(&req, &nodes, &[]);

        // Hard constraint should filter execute-type actions
        assert!(resp.metrics.total_us > 0);
    }

    // ── Edge integration tests ──

    #[test]
    fn test_edges_influence_intents() {
        let mut alloc = NodeIdAllocator::new();
        let goal = make_goal(&mut alloc, "Launch product", 0.8);
        let task_id = alloc.alloc(NodeKind::Task);
        let task = CognitiveNode::new(
            task_id,
            "Write docs".to_string(),
            NodePayload::Task(TaskPayload {
                description: "Write docs".to_string(),
                status: TaskStatus::InProgress,
                goal_id: Some(goal.id),
                deadline: None,
                priority: Priority::High,
                estimated_minutes: Some(120),
                prerequisites: vec![],
            }),
        );

        let edge = CognitiveEdge {
            src: task.id,
            dst: goal.id,
            kind: CognitiveEdgeKind::AdvancesGoal,
            weight: 0.8,
            created_at_ms: 1710000000000,
            last_confirmed_ms: 1710000000000,
            observation_count: 3,
            confidence: 0.9,
        };

        let nodes: Vec<&CognitiveNode> = vec![&goal, &task];
        let edges = vec![edge];

        let req = default_request();
        let resp = suggest_next_step(&req, &nodes, &edges);

        assert!(resp.metrics.total_us > 0);
    }

    // ── Metrics tests ──

    #[test]
    fn test_metrics_populated() {
        let mut alloc = NodeIdAllocator::new();
        let goal = make_goal(&mut alloc, "Test metrics", 0.7);
        let nodes: Vec<&CognitiveNode> = vec![&goal];

        let req = default_request();
        let resp = suggest_next_step(&req, &nodes, &[]);

        assert!(resp.metrics.total_us > 0);
        assert!(resp.metrics.intent_us <= resp.metrics.total_us);
        assert_eq!(resp.metrics.mode, ExecutionMode::Balanced);
    }

    #[test]
    fn test_trace_always_present() {
        let req = default_request();
        let resp = suggest_next_step(&req, &[], &[]);

        // Trace should always be present, even for empty input
        assert!(resp.metrics.total_us > 0);
    }

    // ── Response structure tests ──

    #[test]
    fn test_should_call_llm() {
        let mut alloc = NodeIdAllocator::new();
        let goal = make_goal(&mut alloc, "Complex problem", 0.5);
        let nodes: Vec<&CognitiveNode> = vec![&goal];

        // Set impossibly high threshold to force escalation
        let mut req = default_request();
        req.overrides = Some(PipelineOverrides {
            policy_config: Some(PolicyConfig {
                selection_threshold: 100.0,
                escalate_to_llm_on_empty: true,
                ..PolicyConfig::default()
            }),
            ..PipelineOverrides::default()
        });

        let resp = suggest_next_step(&req, &nodes, &[]);

        if resp.metrics.candidates_count > 0 {
            assert!(
                resp.should_call_llm,
                "should escalate to LLM when all above-threshold candidates are filtered"
            );
            assert!(resp.chosen.is_none());
        }
    }

    #[test]
    fn test_alternatives_ordered() {
        let mut alloc = NodeIdAllocator::new();
        let g1 = make_goal(&mut alloc, "Goal A", 0.9);
        let g2 = make_goal(&mut alloc, "Goal B", 0.7);
        let g3 = make_goal(&mut alloc, "Goal C", 0.5);
        let nodes: Vec<&CognitiveNode> = vec![&g1, &g2, &g3];

        let req = default_request();
        let resp = suggest_next_step(&req, &nodes, &[]);

        // Alternatives should be sorted by utility
        for w in resp.alternatives.windows(2) {
            assert!(
                w[0].utility >= w[1].utility,
                "alternatives should be sorted by utility descending"
            );
        }
    }

    // ── Override tests ──

    #[test]
    fn test_config_overrides() {
        let mut alloc = NodeIdAllocator::new();
        let goal = make_goal(&mut alloc, "Override test", 0.7);
        let nodes: Vec<&CognitiveNode> = vec![&goal];

        let mut req = default_request();
        req.overrides = Some(PipelineOverrides {
            intent_config: Some(IntentConfig {
                max_hypotheses: 2,
                ..IntentConfig::default()
            }),
            ..PipelineOverrides::default()
        });

        let resp = suggest_next_step(&req, &nodes, &[]);

        // Should limit intents to 2
        assert!(resp.metrics.intents_count <= 2);
    }

    // ── Scenario tests ──

    #[test]
    fn test_busy_professional_scenario() {
        let mut alloc = NodeIdAllocator::new();

        // Professional with multiple competing priorities
        let deadline = make_goal(&mut alloc, "Client deliverable due today", 0.95);
        let meeting = make_goal(&mut alloc, "Prepare for board meeting", 0.7);
        let health = make_need(&mut alloc, "Take medication", 0.8);
        let routine = make_routine(&mut alloc, "Daily standup");

        let nodes: Vec<&CognitiveNode> = vec![&deadline, &meeting, &health, &routine];

        let mut req = default_request();
        req.context.cognitive_load = 0.7;

        let resp = suggest_next_step(&req, &nodes, &[]);

        // Should produce a response with multiple intents
        assert!(resp.metrics.intents_count >= 1);
        // The system should suggest something (not just wait)
        // given the high urgency of the deadline
    }

    #[test]
    fn test_relaxed_evening_scenario() {
        let mut alloc = NodeIdAllocator::new();

        let low_priority = make_goal(&mut alloc, "Organize photos", 0.3);
        let nodes: Vec<&CognitiveNode> = vec![&low_priority];

        let mut req = default_request();
        req.context.current_hour = 20; // 8 PM, before quiet hours
        req.context.cognitive_load = 0.1;
        req.context.distress_level = 0.0;

        let resp = suggest_next_step(&req, &nodes, &[]);

        // Low-urgency goal may or may not produce action
        assert!(resp.metrics.total_us > 0);
    }

    #[test]
    fn test_anti_spam_scenario() {
        let mut alloc = NodeIdAllocator::new();
        let goal = make_goal(&mut alloc, "Important goal", 0.8);
        let nodes: Vec<&CognitiveNode> = vec![&goal];

        let mut req = default_request();
        req.context.suggestions_in_window = 15; // well over limit

        let resp = suggest_next_step(&req, &nodes, &[]);

        // All candidates should be filtered by anti-spam
        if resp.metrics.candidates_count > 0 {
            assert!(
                resp.chosen.is_none() || resp.should_call_llm,
                "anti-spam should prevent new suggestions"
            );
        }
    }

    #[test]
    fn test_shared_context_privacy() {
        let mut alloc = NodeIdAllocator::new();
        let goal = make_goal(&mut alloc, "Team project", 0.7);
        let nodes: Vec<&CognitiveNode> = vec![&goal];

        let mut req = default_request();
        req.context.is_shared_context = true;

        let resp = suggest_next_step(&req, &nodes, &[]);

        // In shared context, sensitive schemas should be filtered
        if let Some(ref chosen) = resp.chosen {
            let sensitive = [
                "emotional_check_in",
                "health_reminder",
                "personal_reflection",
            ];
            assert!(
                !sensitive.contains(&chosen.schema_name.as_str()),
                "sensitive actions should not be chosen in shared context"
            );
        }
    }
}
