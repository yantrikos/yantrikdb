//! Engine-level belief revision API.
//!
//! Wires the pure-function belief revision, contradiction detection, and
//! belief query modules into `YantrikDB` methods that operate on the
//! persistent cognitive graph.

use std::collections::HashMap;

use crate::belief::{
    self, apply_staleness_decay, assert_evidence, propagate_evidence, BeliefRevisionConfig,
    Evidence, EvidenceResult, RevisionSummary,
};
use crate::belief_query::{self, BeliefExplanation, BeliefInventory, BeliefPattern};
use crate::contradiction::{self, ContradictionConfig, ContradictionScanResult};
use crate::error::Result;
use crate::state::{CognitiveEdgeKind, CognitiveNode, NodeId, NodeKind, NodePayload};

use super::{now, YantrikDB};

impl YantrikDB {
    // ── Evidence Assertion ──

    /// Assert a piece of evidence against a belief in the persistent graph.
    ///
    /// Loads the belief node, applies Bayesian updating, optionally propagates
    /// through epistemic edges, and persists the updated nodes.
    ///
    /// Returns `None` if the target node doesn't exist or isn't a belief.
    pub fn assert_belief_evidence(
        &self,
        evidence: &Evidence,
        config: &BeliefRevisionConfig,
    ) -> Result<Option<EvidenceResult>> {
        // Load the target belief node
        let mut node = match self.load_cognitive_node(evidence.target_belief)? {
            Some(n) => n,
            None => return Ok(None),
        };

        // Apply Bayesian update
        let mut result = match assert_evidence(&mut node, evidence, config) {
            Some(r) => r,
            None => return Ok(None),
        };

        // Persist the updated belief
        self.persist_cognitive_node(&node)?;

        // Handle evidence propagation if requested
        if evidence.propagate {
            let edges = self.load_cognitive_edges_from(evidence.target_belief)?;

            // Filter to epistemic edges
            let epistemic_edges: Vec<_> = edges
                .iter()
                .filter(|e| e.kind.is_epistemic())
                .cloned()
                .collect();

            if !epistemic_edges.is_empty() {
                // Load downstream belief nodes
                let dst_ids: Vec<NodeId> = epistemic_edges.iter().map(|e| e.dst).collect();

                let mut downstream: HashMap<NodeId, CognitiveNode> = HashMap::new();
                for &dst_id in &dst_ids {
                    if let Some(dst_node) = self.load_cognitive_node(dst_id)? {
                        if dst_node.kind() == NodeKind::Belief {
                            downstream.insert(dst_id, dst_node);
                        }
                    }
                }

                // Propagate
                let effects = propagate_evidence(
                    evidence.target_belief,
                    result.effective_weight,
                    &epistemic_edges,
                    &mut downstream,
                    config,
                    0,
                );

                // Persist updated downstream nodes
                for (id, updated_node) in &downstream {
                    if effects.iter().any(|e| e.belief_id == *id) {
                        self.persist_cognitive_node(updated_node)?;
                    }
                }

                result.propagated_to = effects;
            }
        }

        Ok(Some(result))
    }

    /// Assert multiple pieces of evidence in a batch.
    ///
    /// More efficient than calling `assert_belief_evidence` in a loop
    /// because it batches the persistence operations.
    pub fn assert_belief_evidence_batch(
        &self,
        evidence_batch: &[Evidence],
        config: &BeliefRevisionConfig,
    ) -> Result<Vec<EvidenceResult>> {
        let mut results = Vec::with_capacity(evidence_batch.len());
        let mut updated_nodes: HashMap<NodeId, CognitiveNode> = HashMap::new();

        for evidence in evidence_batch {
            // Check if we already loaded this node in this batch
            let node = if let Some(n) = updated_nodes.get_mut(&evidence.target_belief) {
                n
            } else if let Some(n) = self.load_cognitive_node(evidence.target_belief)? {
                updated_nodes.insert(evidence.target_belief, n);
                updated_nodes.get_mut(&evidence.target_belief).unwrap()
            } else {
                continue;
            };

            if let Some(result) = assert_evidence(node, evidence, config) {
                results.push(result);
            }
        }

        // Batch persist all updated nodes
        let nodes_to_persist: Vec<CognitiveNode> = updated_nodes.into_values().collect();
        if !nodes_to_persist.is_empty() {
            self.persist_cognitive_nodes(&nodes_to_persist)?;
        }

        Ok(results)
    }

    // ── Belief Revision (batch staleness + ceiling) ──

    /// Run belief revision on all persistent beliefs: staleness decay + ceiling.
    ///
    /// This should be called during the `think()` loop to keep beliefs current.
    pub fn revise_all_beliefs(&self, config: &BeliefRevisionConfig) -> Result<RevisionSummary> {
        let now_secs = now();

        // Load all belief nodes
        let mut beliefs = self.load_cognitive_nodes_by_kind(NodeKind::Belief)?;
        if beliefs.is_empty() {
            return Ok(RevisionSummary::default());
        }

        let mut summary = RevisionSummary::default();
        let mut updated = Vec::new();

        for node in beliefs.iter_mut() {
            let decayed = apply_staleness_decay(node, now_secs, config);
            if decayed > 1e-6 {
                summary.staleness_decays += 1;
                summary.beliefs_revised += 1;
                updated.push(node.clone());
            }
        }

        // Persist updated beliefs
        if !updated.is_empty() {
            self.persist_cognitive_nodes(&updated)?;
        }

        Ok(summary)
    }

    // ── Contradiction Detection ──

    /// Scan the persistent belief graph for contradictions.
    pub fn detect_belief_contradictions(
        &self,
        config: &ContradictionConfig,
    ) -> Result<ContradictionScanResult> {
        let now_secs = now();

        // Load all belief and preference nodes
        let beliefs = self.load_cognitive_nodes_by_kind(NodeKind::Belief)?;
        let preferences = self.load_cognitive_nodes_by_kind(NodeKind::Preference)?;

        // Build node map
        let mut node_map: HashMap<NodeId, CognitiveNode> = HashMap::new();
        for n in beliefs.into_iter().chain(preferences.into_iter()) {
            node_map.insert(n.id, n);
        }

        let ref_map: HashMap<NodeId, &CognitiveNode> =
            node_map.iter().map(|(id, n)| (*id, n)).collect();

        // Load all Contradicts edges
        let contradicts_edges =
            self.load_cognitive_edges_by_kind(CognitiveEdgeKind::Contradicts)?;

        let result =
            contradiction::scan_contradictions(&ref_map, &contradicts_edges, config, now_secs);

        Ok(result)
    }

    // ── Belief Queries ──

    /// Query beliefs from persistent storage using a pattern.
    pub fn query_beliefs(&self, pattern: &BeliefPattern) -> Result<Vec<CognitiveNode>> {
        let beliefs = self.load_cognitive_nodes_by_kind(NodeKind::Belief)?;
        let results: Vec<CognitiveNode> = belief_query::query_beliefs(beliefs.iter(), pattern)
            .into_iter()
            .cloned()
            .collect();
        Ok(results)
    }

    /// Explain why a specific belief is held.
    ///
    /// Returns the full provenance chain including evidence trail,
    /// supporting/contradicting beliefs, and confidence trend.
    pub fn explain_belief(&self, belief_id: NodeId) -> Result<Option<BeliefExplanation>> {
        let node = match self.load_cognitive_node(belief_id)? {
            Some(n) => n,
            None => return Ok(None),
        };

        let now_secs = now();

        // Load edges pointing TO this belief (for supporting/contradicting beliefs)
        let edges_to = self.load_cognitive_edges_to(belief_id)?;

        // Load the source nodes of those edges
        let mut neighbor_map: HashMap<NodeId, CognitiveNode> = HashMap::new();
        for edge in &edges_to {
            if edge.kind.is_epistemic() {
                if let Some(src) = self.load_cognitive_node(edge.src)? {
                    neighbor_map.insert(edge.src, src);
                }
            }
        }

        let neighbor_refs: HashMap<NodeId, &CognitiveNode> =
            neighbor_map.iter().map(|(id, n)| (*id, n)).collect();

        let explanation = belief_query::explain_belief(&node, &edges_to, &neighbor_refs, now_secs);
        Ok(explanation)
    }

    /// Get a comprehensive inventory of the belief landscape.
    pub fn belief_inventory(&self) -> Result<BeliefInventory> {
        let beliefs = self.load_cognitive_nodes_by_kind(NodeKind::Belief)?;
        let refs: Vec<&CognitiveNode> = beliefs.iter().collect();
        Ok(belief_query::belief_inventory(&refs))
    }

    /// Confirm a belief (mark as user-confirmed).
    ///
    /// User-confirmed beliefs get higher reliability priors and are
    /// preferred during contradiction resolution.
    pub fn confirm_belief(&self, belief_id: NodeId) -> Result<bool> {
        let mut node = match self.load_cognitive_node(belief_id)? {
            Some(n) => n,
            None => return Ok(false),
        };

        if let NodePayload::Belief(ref mut belief) = node.payload {
            belief.user_confirmed = true;
            node.attrs.provenance = crate::state::Provenance::Told;
            // Boost confidence for confirmed beliefs
            let boost = 1.0; // +1.0 log-odds ≈ +0.27 probability at P=0.5
            belief.log_odds += boost;
            node.attrs.confidence = crate::state::sigmoid(belief.log_odds);
            self.persist_cognitive_node(&node)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Refute a belief (user explicitly says it's wrong).
    ///
    /// Sets log-odds strongly negative and marks as user-confirmed
    /// (confirmed to be false).
    pub fn refute_belief(&self, belief_id: NodeId) -> Result<bool> {
        let mut node = match self.load_cognitive_node(belief_id)? {
            Some(n) => n,
            None => return Ok(false),
        };

        if let NodePayload::Belief(ref mut belief) = node.payload {
            belief.user_confirmed = true; // Confirmed as FALSE
            belief.log_odds = -4.0; // Strong disbelief
            node.attrs.confidence = crate::state::sigmoid(belief.log_odds);
            node.attrs.provenance = crate::state::Provenance::Told;
            self.persist_cognitive_node(&node)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{
        BeliefPayload, CognitiveEdge, CognitiveNode, NodeIdAllocator, NodePayload,
        PreferencePayload, Provenance,
    };

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 4).unwrap()
    }

    fn make_belief_node(alloc: &mut NodeIdAllocator, prop: &str, log_odds: f64) -> CognitiveNode {
        let id = alloc.alloc(NodeKind::Belief);
        let mut node = CognitiveNode::new(
            id,
            prop.to_string(),
            NodePayload::Belief(BeliefPayload {
                proposition: prop.to_string(),
                log_odds,
                domain: "test".to_string(),
                evidence_trail: vec![],
                user_confirmed: false,
            }),
        );
        node.attrs.confidence = crate::state::sigmoid(log_odds);
        node
    }

    #[test]
    fn test_assert_evidence_persists() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let node = make_belief_node(&mut alloc, "User likes coffee", 0.0);
        db.persist_cognitive_node(&node).unwrap();

        let evidence = Evidence {
            target_belief: node.id,
            weight: 2.0,
            source: "observed coffee purchase".to_string(),
            provenance: Provenance::Observed,
            propagate: false,
            timestamp: 1000.0,
        };

        let config = BeliefRevisionConfig::default();
        let result = db
            .assert_belief_evidence(&evidence, &config)
            .unwrap()
            .unwrap();

        assert!(result.posterior_probability > 0.5);

        // Verify persistence
        let loaded = db.load_cognitive_node(node.id).unwrap().unwrap();
        if let NodePayload::Belief(b) = &loaded.payload {
            assert!(b.log_odds > 0.0);
            assert_eq!(b.evidence_trail.len(), 1);
        } else {
            panic!("expected belief");
        }
    }

    #[test]
    fn test_assert_evidence_with_propagation() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let a = make_belief_node(&mut alloc, "Coffee is healthy", 1.0);
        let b = make_belief_node(&mut alloc, "Caffeine is safe", 0.0);

        db.persist_cognitive_node(&a).unwrap();
        db.persist_cognitive_node(&b).unwrap();

        // A --Supports--> B
        let edge = CognitiveEdge::new(a.id, b.id, CognitiveEdgeKind::Supports, 0.8);
        db.persist_cognitive_edge(&edge).unwrap();

        let evidence = Evidence {
            target_belief: a.id,
            weight: 2.0,
            source: "new study".to_string(),
            provenance: Provenance::Extracted,
            propagate: true,
            timestamp: 1000.0,
        };

        let config = BeliefRevisionConfig::default();
        let result = db
            .assert_belief_evidence(&evidence, &config)
            .unwrap()
            .unwrap();

        // Should have propagated to B
        assert!(!result.propagated_to.is_empty());

        // Verify B was updated
        let loaded_b = db.load_cognitive_node(b.id).unwrap().unwrap();
        if let NodePayload::Belief(bel) = &loaded_b.payload {
            assert!(
                bel.log_odds > 0.0,
                "B should have received propagated evidence"
            );
        }
    }

    #[test]
    fn test_batch_evidence() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let n1 = make_belief_node(&mut alloc, "Belief 1", 0.0);
        let n2 = make_belief_node(&mut alloc, "Belief 2", 0.0);

        db.persist_cognitive_node(&n1).unwrap();
        db.persist_cognitive_node(&n2).unwrap();

        let batch = vec![
            Evidence {
                target_belief: n1.id,
                weight: 1.5,
                source: "src1".to_string(),
                provenance: Provenance::Observed,
                propagate: false,
                timestamp: 1000.0,
            },
            Evidence {
                target_belief: n2.id,
                weight: -1.0,
                source: "src2".to_string(),
                provenance: Provenance::Inferred,
                propagate: false,
                timestamp: 1000.0,
            },
        ];

        let config = BeliefRevisionConfig::default();
        let results = db.assert_belief_evidence_batch(&batch, &config).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].posterior_probability > 0.5);
        assert!(results[1].posterior_probability < 0.5);
    }

    #[test]
    fn test_detect_contradictions() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let a = make_belief_node(&mut alloc, "Earth is flat", 2.0);
        let b = make_belief_node(&mut alloc, "Earth is round", 3.0);

        db.persist_cognitive_node(&a).unwrap();
        db.persist_cognitive_node(&b).unwrap();

        let edge = CognitiveEdge::new(a.id, b.id, CognitiveEdgeKind::Contradicts, 0.9);
        db.persist_cognitive_edge(&edge).unwrap();

        let config = ContradictionConfig::default();
        let result = db.detect_belief_contradictions(&config).unwrap();

        assert!(result.epistemic_conflicts >= 1);
        assert!(!result.conflicts.is_empty());
    }

    #[test]
    fn test_query_beliefs() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        for i in 0..5 {
            let lo = if i < 3 { 3.0 } else { 0.1 };
            let node = make_belief_node(&mut alloc, &format!("Belief {i}"), lo);
            db.persist_cognitive_node(&node).unwrap();
        }

        let pattern = BeliefPattern {
            min_probability: Some(0.8),
            limit: 10,
            ..Default::default()
        };

        let results = db.query_beliefs(&pattern).unwrap();
        assert_eq!(results.len(), 3); // The 3 with high log-odds
    }

    #[test]
    fn test_explain_belief() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let node = make_belief_node(&mut alloc, "User prefers dark mode", 2.0);
        db.persist_cognitive_node(&node).unwrap();

        // Add evidence via assert
        let evidence = Evidence {
            target_belief: node.id,
            weight: 1.5,
            source: "settings observation".to_string(),
            provenance: Provenance::Observed,
            propagate: false,
            timestamp: 1000.0,
        };

        let config = BeliefRevisionConfig::default();
        db.assert_belief_evidence(&evidence, &config).unwrap();

        let explanation = db.explain_belief(node.id).unwrap().unwrap();

        assert_eq!(explanation.proposition, "User prefers dark mode");
        assert!(explanation.probability > 0.8);
        assert!(!explanation.supporting_evidence.is_empty());
    }

    #[test]
    fn test_confirm_belief() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let node = make_belief_node(&mut alloc, "User's birthday is March 15", 1.0);
        db.persist_cognitive_node(&node).unwrap();

        assert!(db.confirm_belief(node.id).unwrap());

        let loaded = db.load_cognitive_node(node.id).unwrap().unwrap();
        if let NodePayload::Belief(b) = &loaded.payload {
            assert!(b.user_confirmed);
            assert!(b.log_odds > 1.0); // Boosted
        }
    }

    #[test]
    fn test_refute_belief() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let node = make_belief_node(&mut alloc, "User lives in NYC", 2.0);
        db.persist_cognitive_node(&node).unwrap();

        assert!(db.refute_belief(node.id).unwrap());

        let loaded = db.load_cognitive_node(node.id).unwrap().unwrap();
        if let NodePayload::Belief(b) = &loaded.payload {
            assert!(b.user_confirmed);
            assert!(b.log_odds < -3.0); // Strongly refuted
        }
    }

    #[test]
    fn test_belief_inventory() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        for i in 0..10 {
            let mut node =
                make_belief_node(&mut alloc, &format!("Belief {i}"), (i as f64 - 5.0) * 0.5);
            if let NodePayload::Belief(b) = &mut node.payload {
                b.domain = if i < 4 {
                    "health".to_string()
                } else {
                    "work".to_string()
                };
            }
            db.persist_cognitive_node(&node).unwrap();
        }

        let inv = db.belief_inventory().unwrap();
        assert_eq!(inv.total_beliefs, 10);
        assert_eq!(*inv.by_domain.get("health").unwrap(), 4);
        assert_eq!(*inv.by_domain.get("work").unwrap(), 6);
    }
}
