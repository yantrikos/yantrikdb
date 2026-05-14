//! Engine-level action schema API.
//!
//! Wires the action schema library and candidate generator into `YantrikDB`
//! methods that operate on the persistent cognitive graph.

use crate::action::{self, ActionCandidate, ActionConfig, CandidateGenerationResult};
use crate::error::Result;
use crate::intent::{IntentConfig, ScoredIntent};
use crate::state::{CognitiveNode, NodeKind};

use super::YantrikDB;

impl YantrikDB {
    // ── Action Candidate Generation ──

    /// Generate action candidates for a set of intent hypotheses.
    ///
    /// Loads all relevant nodes and persisted action schemas, then runs
    /// the candidate generator against built-in + persisted schemas.
    pub fn generate_action_candidates(
        &self,
        intents: &[ScoredIntent],
        config: &ActionConfig,
    ) -> Result<CandidateGenerationResult> {
        // Load all node kinds for precondition matching
        let mut all_nodes: Vec<CognitiveNode> = Vec::new();
        for kind in NodeKind::ALL.iter() {
            all_nodes.extend(self.load_cognitive_nodes_by_kind(*kind)?);
        }

        // Load persisted action schemas separately (for custom/learned schemas)
        let persisted_schemas: Vec<CognitiveNode> = all_nodes
            .iter()
            .filter(|n| n.id.kind() == NodeKind::ActionSchema)
            .cloned()
            .collect();

        let node_refs: Vec<&CognitiveNode> = all_nodes.iter().collect();
        let schema_refs: Vec<&CognitiveNode> = persisted_schemas.iter().collect();

        let result = action::generate_candidates(intents, &node_refs, &schema_refs, config);
        Ok(result)
    }

    /// Run the full intent→action pipeline: infer intents, then generate candidates.
    ///
    /// This is the convenience method that chains CK-1.6 → CK-1.7.
    pub fn infer_and_generate_actions(
        &self,
        intent_config: &IntentConfig,
        action_config: &ActionConfig,
    ) -> Result<(Vec<ScoredIntent>, CandidateGenerationResult)> {
        let intent_result = self.infer_intents(intent_config)?;
        let action_result =
            self.generate_action_candidates(&intent_result.hypotheses, action_config)?;
        Ok((intent_result.hypotheses, action_result))
    }

    /// Get the top action candidate from the full pipeline.
    pub fn top_action(
        &self,
        intent_config: &IntentConfig,
        action_config: &ActionConfig,
    ) -> Result<Option<ActionCandidate>> {
        let (_, result) = self.infer_and_generate_actions(intent_config, action_config)?;
        Ok(result.candidates.into_iter().next())
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::action::ActionConfig;
    use crate::engine::YantrikDB;
    use crate::intent::IntentConfig;
    use crate::state::*;

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn test_generate_candidates_empty_graph() {
        let db = test_db();
        let config = ActionConfig::default();
        let result = db.generate_action_candidates(&[], &config).unwrap();

        // No intents → no candidates
        assert!(result.candidates.is_empty());
    }

    #[test]
    fn test_full_intent_to_action_pipeline() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        // Create a goal with urgency
        let goal_id = alloc.alloc(NodeKind::Goal);
        let mut node = CognitiveNode::new(
            goal_id,
            "Ship v1.0".to_string(),
            NodePayload::Goal(GoalPayload {
                description: "Ship v1.0".to_string(),
                status: GoalStatus::Active,
                progress: 0.7,
                deadline: None,
                priority: Priority::Critical,
                parent_goal: None,
                completion_criteria: "Released".to_string(),
            }),
        );
        node.attrs.urgency = 0.9;
        db.persist_cognitive_node(&node).unwrap();
        db.persist_node_id_allocator(&alloc).unwrap();

        let intent_config = IntentConfig::default();
        let action_config = ActionConfig::default();

        let (intents, actions) = db
            .infer_and_generate_actions(&intent_config, &action_config)
            .unwrap();

        assert!(!intents.is_empty(), "should generate intents");
        assert!(
            !actions.candidates.is_empty(),
            "should generate action candidates"
        );

        // Candidates should be sorted by relevance
        for w in actions.candidates.windows(2) {
            assert!(w[0].relevance_score >= w[1].relevance_score);
        }
    }

    #[test]
    fn test_top_action() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let goal_id = alloc.alloc(NodeKind::Goal);
        let mut node = CognitiveNode::new(
            goal_id,
            "Complete task".to_string(),
            NodePayload::Goal(GoalPayload {
                description: "Complete task".to_string(),
                status: GoalStatus::Active,
                progress: 0.5,
                deadline: None,
                priority: Priority::High,
                parent_goal: None,
                completion_criteria: "Done".to_string(),
            }),
        );
        node.attrs.urgency = 0.7;
        db.persist_cognitive_node(&node).unwrap();
        db.persist_node_id_allocator(&alloc).unwrap();

        let top = db
            .top_action(&IntentConfig::default(), &ActionConfig::default())
            .unwrap();
        assert!(top.is_some());
    }
}
