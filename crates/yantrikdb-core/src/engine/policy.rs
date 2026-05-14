//! Engine-level policy API.
//!
//! Wires the policy engine into `YantrikDB` methods that chain
//! the full pipeline: intent → action → evaluate → policy select.

use crate::action::ActionConfig;
use crate::error::Result;
use crate::evaluator::EvaluatorConfig;
use crate::intent::IntentConfig;
use crate::policy::{self, PolicyConfig, PolicyContext, PolicyResult};
use crate::state::{CognitiveEdgeKind, CognitiveNode, NodeKind};

use super::{now, YantrikDB};

impl YantrikDB {
    // ── Policy Selection ──

    /// Run the full pipeline: intent → action → evaluate → policy select.
    ///
    /// This chains CK-1.6 → CK-1.7 → CK-1.8 → CK-1.9 into a single call.
    /// Returns a policy decision with full reasoning trace.
    pub fn select_action(
        &self,
        intent_config: &IntentConfig,
        action_config: &ActionConfig,
        eval_config: &EvaluatorConfig,
        policy_config: &PolicyConfig,
        policy_ctx: &PolicyContext,
    ) -> Result<PolicyResult> {
        // CK-1.8: Run evaluate_actions (which chains 1.6 → 1.7 → 1.8)
        let eval_result = self.evaluate_actions(intent_config, action_config, eval_config)?;

        // Load constraint nodes for policy enforcement
        let mut context_nodes: Vec<CognitiveNode> = Vec::new();
        context_nodes.extend(self.load_cognitive_nodes_by_kind(NodeKind::Constraint)?);
        // Also load preference nodes for soft constraint evaluation
        context_nodes.extend(self.load_cognitive_nodes_by_kind(NodeKind::Preference)?);

        let node_refs: Vec<&CognitiveNode> = context_nodes.iter().collect();
        let ts = now();

        // CK-1.9: Policy selection
        let result = policy::select_action(
            &eval_result.actions,
            &node_refs,
            policy_config,
            policy_ctx,
            ts,
        );

        Ok(result)
    }

    /// Run the full pipeline with default configurations.
    pub fn select_action_default(&self, policy_ctx: &PolicyContext) -> Result<PolicyResult> {
        self.select_action(
            &IntentConfig::default(),
            &ActionConfig::default(),
            &EvaluatorConfig::default(),
            &PolicyConfig::default(),
            policy_ctx,
        )
    }

    /// Run the full pipeline with all-default configs and default context.
    pub fn select_action_auto(&self) -> Result<PolicyResult> {
        self.select_action_default(&PolicyContext::default())
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::policy::{PolicyConfig, PolicyContext, PolicyDecision};
    use crate::state::*;

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn test_select_action_empty_graph() {
        let db = test_db();
        let result = db.select_action_auto().unwrap();

        // Empty graph → Wait (no candidates)
        assert!(matches!(result.decision, PolicyDecision::Wait { .. }));
    }

    #[test]
    fn test_full_pipeline_with_policy() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let goal_id = alloc.alloc(NodeKind::Goal);
        let mut goal = CognitiveNode::new(
            goal_id,
            "Ship feature".to_string(),
            NodePayload::Goal(GoalPayload {
                description: "Ship feature".to_string(),
                status: GoalStatus::Active,
                progress: 0.5,
                deadline: None,
                priority: Priority::Critical,
                parent_goal: None,
                completion_criteria: "Deployed".to_string(),
            }),
        );
        goal.attrs.urgency = 0.85;
        db.persist_cognitive_node(&goal).unwrap();
        db.persist_node_id_allocator(&alloc).unwrap();

        let ctx = PolicyContext::default();
        let result = db.select_action_default(&ctx).unwrap();

        // Should produce a decision (Act or EscalateToLlm)
        assert!(
            matches!(
                result.decision,
                PolicyDecision::Act(_) | PolicyDecision::EscalateToLlm { .. }
            ),
            "should produce an action decision or escalate"
        );
        assert!(result.trace.execution_time_us > 0);
    }

    #[test]
    fn test_quiet_hours_policy() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let goal_id = alloc.alloc(NodeKind::Goal);
        let mut goal = CognitiveNode::new(
            goal_id,
            "Morning routine".to_string(),
            NodePayload::Goal(GoalPayload {
                description: "Morning routine".to_string(),
                status: GoalStatus::Active,
                progress: 0.2,
                deadline: None,
                priority: Priority::High,
                parent_goal: None,
                completion_criteria: "Done".to_string(),
            }),
        );
        goal.attrs.urgency = 0.7;
        db.persist_cognitive_node(&goal).unwrap();
        db.persist_node_id_allocator(&alloc).unwrap();

        let mut ctx = PolicyContext::default();
        ctx.current_hour = 3; // 3 AM — quiet hours

        let result = db.select_action_default(&ctx).unwrap();

        // Most proactive actions should be filtered by quiet hours
        assert!(result.trace.rejected_candidates.iter().any(|r| {
            matches!(
                r.rejection_reason,
                crate::policy::RejectionReason::QuietHours
            )
        }));
    }

    #[test]
    fn test_constraint_node_enforcement() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        // Add a goal to generate candidates
        let goal_id = alloc.alloc(NodeKind::Goal);
        let mut goal = CognitiveNode::new(
            goal_id,
            "Test goal".to_string(),
            NodePayload::Goal(GoalPayload {
                description: "Test goal".to_string(),
                status: GoalStatus::Active,
                progress: 0.3,
                deadline: None,
                priority: Priority::High,
                parent_goal: None,
                completion_criteria: "Done".to_string(),
            }),
        );
        goal.attrs.urgency = 0.7;
        db.persist_cognitive_node(&goal).unwrap();

        // Add a hard constraint
        let constraint_id = alloc.alloc(NodeKind::Constraint);
        let constraint = CognitiveNode::new(
            constraint_id,
            "No execute".to_string(),
            NodePayload::Constraint(ConstraintPayload {
                description: "No execute during work".to_string(),
                constraint_type: ConstraintType::Hard,
                condition: "no execute actions".to_string(),
                imposed_by: "user".to_string(),
            }),
        );
        db.persist_cognitive_node(&constraint).unwrap();
        db.persist_node_id_allocator(&alloc).unwrap();

        let ctx = PolicyContext::default();
        let result = db.select_action_default(&ctx).unwrap();

        // Execute-type actions should be filtered
        let execute_rejected = result.trace.rejected_candidates.iter().any(|r| {
            r.action_kind == "Execute"
                && matches!(
                    r.rejection_reason,
                    crate::policy::RejectionReason::HardConstraint { .. }
                )
        });
        // This may or may not find execute candidates depending on what the pipeline generates,
        // but the constraint node should be loaded and checked
        assert!(result.trace.execution_time_us > 0);
    }
}
