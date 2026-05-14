//! Engine-level utility evaluation API.
//!
//! Wires the evaluator module into `YantrikDB` methods that chain
//! intent inference → candidate generation → utility evaluation.

use crate::action::ActionConfig;
use crate::error::Result;
use crate::evaluator::{self, EvaluatedAction, EvaluationResult, EvaluatorConfig};
use crate::intent::IntentConfig;
use crate::state::{CognitiveEdgeKind, CognitiveNode, NodeKind};

use super::YantrikDB;

impl YantrikDB {
    // ── Utility Evaluation ──

    /// Run the full intent → action → evaluation pipeline.
    ///
    /// This chains CK-1.6 → CK-1.7 → CK-1.8 into a single call.
    /// Returns evaluated actions ranked by utility.
    pub fn evaluate_actions(
        &self,
        intent_config: &IntentConfig,
        action_config: &ActionConfig,
        eval_config: &EvaluatorConfig,
    ) -> Result<EvaluationResult> {
        // CK-1.6: Infer intents
        let intent_result = self.infer_intents(intent_config)?;

        // CK-1.7: Generate candidates
        let candidate_result =
            self.generate_action_candidates(&intent_result.hypotheses, action_config)?;

        // Load context for evaluation
        let mut all_nodes: Vec<CognitiveNode> = Vec::new();
        for kind in NodeKind::ALL.iter() {
            all_nodes.extend(self.load_cognitive_nodes_by_kind(*kind)?);
        }

        // Load preference-relevant edges
        let mut edges = Vec::new();
        for kind in &[CognitiveEdgeKind::Prefers, CognitiveEdgeKind::Avoids] {
            edges.extend(self.load_cognitive_edges_by_kind(*kind)?);
        }

        let node_refs: Vec<&CognitiveNode> = all_nodes.iter().collect();

        // CK-1.8: Evaluate candidates
        let result = evaluator::evaluate_candidates(
            &candidate_result.candidates,
            &node_refs,
            &edges,
            eval_config,
        );

        Ok(result)
    }

    /// Get the best action from the full pipeline.
    pub fn best_action(
        &self,
        intent_config: &IntentConfig,
        action_config: &ActionConfig,
        eval_config: &EvaluatorConfig,
    ) -> Result<Option<EvaluatedAction>> {
        let result = self.evaluate_actions(intent_config, action_config, eval_config)?;
        Ok(result.actions.into_iter().next())
    }

    /// Evaluate with all default configurations.
    pub fn evaluate_actions_default(&self) -> Result<EvaluationResult> {
        self.evaluate_actions(
            &IntentConfig::default(),
            &ActionConfig::default(),
            &EvaluatorConfig::default(),
        )
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::action::ActionConfig;
    use crate::engine::YantrikDB;
    use crate::evaluator::EvaluatorConfig;
    use crate::intent::IntentConfig;
    use crate::state::*;

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn test_evaluate_empty_graph() {
        let db = test_db();
        let result = db.evaluate_actions_default().unwrap();
        assert!(result.actions.is_empty() || result.total_evaluated == 0);
    }

    #[test]
    fn test_full_pipeline_with_goal() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let goal_id = alloc.alloc(NodeKind::Goal);
        let mut node = CognitiveNode::new(
            goal_id,
            "Deliver project".to_string(),
            NodePayload::Goal(GoalPayload {
                description: "Deliver project".to_string(),
                status: GoalStatus::Active,
                progress: 0.6,
                deadline: None,
                priority: Priority::Critical,
                parent_goal: None,
                completion_criteria: "Shipped".to_string(),
            }),
        );
        node.attrs.urgency = 0.85;
        db.persist_cognitive_node(&node).unwrap();
        db.persist_node_id_allocator(&alloc).unwrap();

        let result = db
            .evaluate_actions(
                &IntentConfig::default(),
                &ActionConfig::default(),
                &EvaluatorConfig::default(),
            )
            .unwrap();

        assert!(
            !result.actions.is_empty(),
            "should produce evaluated actions"
        );

        // Should be sorted by utility
        for w in result.actions.windows(2) {
            assert!(w[0].utility >= w[1].utility);
        }
    }

    #[test]
    fn test_best_action() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let goal_id = alloc.alloc(NodeKind::Goal);
        let mut node = CognitiveNode::new(
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
        node.attrs.urgency = 0.7;
        db.persist_cognitive_node(&node).unwrap();
        db.persist_node_id_allocator(&alloc).unwrap();

        let best = db
            .best_action(
                &IntentConfig::default(),
                &ActionConfig::default(),
                &EvaluatorConfig::default(),
            )
            .unwrap();

        assert!(best.is_some());
    }
}
