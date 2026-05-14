//! Engine-level intent inference API.
//!
//! Wires the pure-function intent inference module into `YantrikDB` methods
//! that operate on the persistent cognitive graph.

use crate::error::Result;
use crate::intent::{self, IntentConfig, IntentInferenceResult, ScoredIntent};
use crate::state::{CognitiveEdge, CognitiveEdgeKind, CognitiveNode, NodeKind};

use super::{now, YantrikDB};

impl YantrikDB {
    // ── Intent Inference ──

    /// Run intent inference on the full cognitive graph.
    ///
    /// Loads all relevant node kinds (goals, routines, needs, episodes,
    /// opportunities), loads edges, and runs the inference pipeline.
    ///
    /// Returns ranked hypotheses with posterior probabilities.
    pub fn infer_intents(&self, config: &IntentConfig) -> Result<IntentInferenceResult> {
        let ts = now();

        // Load nodes from all signal-source kinds
        let mut all_nodes: Vec<CognitiveNode> = Vec::new();
        for kind in &[
            NodeKind::Goal,
            NodeKind::Routine,
            NodeKind::Need,
            NodeKind::Episode,
            NodeKind::Opportunity,
        ] {
            all_nodes.extend(self.load_cognitive_nodes_by_kind(*kind)?);
        }

        // Load edges relevant to intent inference
        let edges = self.load_intent_relevant_edges()?;

        let refs: Vec<&CognitiveNode> = all_nodes.iter().collect();
        let result = intent::infer_intents(&refs, &edges, ts, config);

        Ok(result)
    }

    /// Run intent inference and materialize the top hypotheses as
    /// IntentHypothesis nodes in the cognitive graph.
    ///
    /// This is the "full pipeline" version: infer → materialize → return.
    /// Materialized nodes can then participate in spreading activation
    /// and be picked up by the action schema generator (CK-1.7).
    ///
    /// Returns the inference result plus the materialized node IDs.
    pub fn infer_and_materialize_intents(
        &self,
        config: &IntentConfig,
    ) -> Result<(IntentInferenceResult, Vec<CognitiveNode>)> {
        let result = self.infer_intents(config)?;

        let mut alloc = self.load_node_id_allocator()?;

        // Materialize top hypotheses as cognitive nodes
        let mut materialized = Vec::new();
        for hypothesis in &result.hypotheses {
            let node = intent::intent_to_node(hypothesis, &mut alloc);
            self.persist_cognitive_node(&node)?;
            materialized.push(node);
        }

        self.persist_node_id_allocator(&alloc)?;

        Ok((result, materialized))
    }

    /// Run intent inference with default configuration.
    pub fn infer_intents_default(&self) -> Result<IntentInferenceResult> {
        self.infer_intents(&IntentConfig::default())
    }

    /// Get the top intent hypothesis (highest posterior), if any.
    pub fn top_intent(&self, config: &IntentConfig) -> Result<Option<ScoredIntent>> {
        let result = self.infer_intents(config)?;
        Ok(result.hypotheses.into_iter().next())
    }

    // ── Helper: load intent-relevant edges ──

    /// Load edges relevant to intent inference.
    ///
    /// We only need edge kinds that affect intent scoring:
    /// AdvancesGoal, BlocksGoal, Triggers, Supports, Contradicts, AssociatedWith.
    fn load_intent_relevant_edges(&self) -> Result<Vec<CognitiveEdge>> {
        let relevant_kinds = [
            CognitiveEdgeKind::AdvancesGoal,
            CognitiveEdgeKind::BlocksGoal,
            CognitiveEdgeKind::Triggers,
            CognitiveEdgeKind::Supports,
            CognitiveEdgeKind::Contradicts,
            CognitiveEdgeKind::AssociatedWith,
        ];

        let mut all_edges = Vec::new();
        for kind in &relevant_kinds {
            all_edges.extend(self.load_cognitive_edges_by_kind(*kind)?);
        }
        Ok(all_edges)
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::intent::IntentConfig;
    use crate::state::*;

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    fn make_goal_node(alloc: &mut NodeIdAllocator, desc: &str, urgency: f64) -> CognitiveNode {
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

    fn make_need_node(alloc: &mut NodeIdAllocator, desc: &str, intensity: f64) -> CognitiveNode {
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

    #[test]
    fn test_infer_intents_empty_graph() {
        let db = test_db();
        let config = IntentConfig::default();
        let result = db.infer_intents(&config).unwrap();

        assert_eq!(result.total_generated, 0);
        assert!(result.hypotheses.is_empty());
    }

    #[test]
    fn test_infer_intents_with_goals() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let goal = make_goal_node(&mut alloc, "Finish quarterly report", 0.8);
        db.persist_cognitive_node(&goal).unwrap();
        db.persist_node_id_allocator(&alloc).unwrap();

        let config = IntentConfig::default();
        let result = db.infer_intents(&config).unwrap();

        assert!(result.total_generated >= 1);
        assert!(result
            .hypotheses
            .iter()
            .any(|h| h.description.contains("quarterly report")));
    }

    #[test]
    fn test_infer_and_materialize() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let goal = make_goal_node(&mut alloc, "Learn Rust", 0.7);
        let need = make_need_node(&mut alloc, "Practice coding", 0.6);

        db.persist_cognitive_node(&goal).unwrap();
        db.persist_cognitive_node(&need).unwrap();
        db.persist_node_id_allocator(&alloc).unwrap();

        let config = IntentConfig::default();
        let (result, materialized) = db.infer_and_materialize_intents(&config).unwrap();

        assert!(!result.hypotheses.is_empty());
        assert_eq!(materialized.len(), result.hypotheses.len());

        for node in &materialized {
            assert_eq!(node.id.kind(), NodeKind::IntentHypothesis);
            assert!(matches!(node.payload, NodePayload::IntentHypothesis(_)));
        }
    }

    #[test]
    fn test_top_intent() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let goal = make_goal_node(&mut alloc, "Ship feature", 0.9);
        db.persist_cognitive_node(&goal).unwrap();
        db.persist_node_id_allocator(&alloc).unwrap();

        let config = IntentConfig::default();
        let top = db.top_intent(&config).unwrap();

        assert!(top.is_some());
        assert!(top.unwrap().description.contains("Ship feature"));
    }

    #[test]
    fn test_infer_intents_default() {
        let db = test_db();
        let result = db.infer_intents_default().unwrap();
        assert_eq!(result.total_generated, 0);
    }
}
