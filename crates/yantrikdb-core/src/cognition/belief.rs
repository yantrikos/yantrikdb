//! Belief Revision Engine — Bayesian Truth Maintenance
//!
//! This module implements principled belief revision using log-odds representation.
//! It provides the core algorithms for:
//! - **Evidence assertion**: Add new evidence (supporting or contradicting) to beliefs
//! - **Bayesian updating**: Update log-odds with source reliability priors
//! - **Staleness decay**: Beliefs lose confidence when not re-evidenced
//! - **Confidence ceiling enforcement**: Volatile beliefs can't reach certainty
//! - **Evidence propagation**: Evidence flows through epistemic edges (Supports/Contradicts)
//!
//! ## Log-Odds Refresher
//!
//! Log-odds is the natural representation for Bayesian updating:
//! ```text
//! P(true) = sigmoid(log_odds) = 1 / (1 + exp(-log_odds))
//! log_odds_new = log_odds_old + evidence_weight * source_reliability
//! ```
//!
//! This is additive (no clamping at boundaries), handles conflicting evidence
//! naturally, and is numerically stable for extreme values.

use std::collections::HashMap;

use crate::state::{
    BeliefPayload, CognitiveAttrs, CognitiveEdge, CognitiveEdgeKind, CognitiveNode, EvidenceEntry,
    NodeId, NodeKind, NodePayload, Provenance,
};

// ── Evidence Types ──

/// A piece of evidence to be asserted into the belief system.
#[derive(Debug, Clone)]
pub struct Evidence {
    /// Which belief this evidence applies to.
    pub target_belief: NodeId,
    /// Log-likelihood ratio: positive = supports, negative = contradicts.
    /// Typical range: [-3.0, 3.0] for normal evidence. >4.0 for very strong.
    pub weight: f64,
    /// Human-readable source description.
    pub source: String,
    /// Provenance of this evidence (affects reliability scaling).
    pub provenance: Provenance,
    /// Optional: propagate this evidence through epistemic edges?
    pub propagate: bool,
    /// When this evidence was observed (unix timestamp seconds).
    pub timestamp: f64,
}

/// Result of asserting a piece of evidence.
#[derive(Debug, Clone)]
pub struct EvidenceResult {
    /// The belief that was updated.
    pub belief_id: NodeId,
    /// Log-odds before the update.
    pub prior_log_odds: f64,
    /// Log-odds after the update.
    pub posterior_log_odds: f64,
    /// Probability before.
    pub prior_probability: f64,
    /// Probability after.
    pub posterior_probability: f64,
    /// Effective weight applied (after reliability scaling).
    pub effective_weight: f64,
    /// Whether the belief crossed a significance threshold.
    pub crossed_threshold: bool,
    /// Direction of the threshold crossing (if any).
    pub threshold_direction: Option<ThresholdDirection>,
    /// Beliefs affected by propagation (if propagate=true).
    pub propagated_to: Vec<PropagationEffect>,
}

/// Direction of a threshold crossing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThresholdDirection {
    /// Belief became confident (crossed from uncertain to believed).
    BecameConfident,
    /// Belief became doubtful (crossed from believed to uncertain).
    BecameDoubtful,
    /// Belief flipped (crossed from believed to disbelieved or vice versa).
    Flipped,
}

/// Effect of evidence propagation to a downstream belief.
#[derive(Debug, Clone)]
pub struct PropagationEffect {
    pub belief_id: NodeId,
    pub edge_kind: CognitiveEdgeKind,
    pub propagated_weight: f64,
    pub new_log_odds: f64,
}

/// Summary of a batch belief revision pass.
#[derive(Debug, Clone, Default)]
pub struct RevisionSummary {
    /// Number of beliefs whose log-odds changed.
    pub beliefs_revised: usize,
    /// Number of beliefs that crossed a significance threshold.
    pub threshold_crossings: usize,
    /// Number of beliefs that decayed due to staleness.
    pub staleness_decays: usize,
    /// Number of beliefs that hit a confidence ceiling.
    pub ceiling_enforcements: usize,
    /// Contradictions detected during this revision pass.
    pub contradictions_found: usize,
    /// Total evidence entries processed.
    pub evidence_processed: usize,
}

// ── Configuration ──

/// Configuration for the belief revision engine.
#[derive(Debug, Clone)]
pub struct BeliefRevisionConfig {
    /// Probability thresholds for significance detection.
    /// Default: 0.75 (believed), 0.25 (disbelieved).
    pub believed_threshold: f64,
    pub disbelieved_threshold: f64,

    /// Staleness decay rate: log-odds per day of no evidence.
    /// Applied as: log_odds *= (1 - staleness_rate * days_since_last_evidence)
    /// Default: 0.02 (very slow — beliefs are sticky).
    pub staleness_rate_per_day: f64,

    /// Minimum days before staleness decay kicks in.
    /// Freshly evidenced beliefs don't decay. Default: 7.
    pub staleness_grace_period_days: f64,

    /// Maximum absolute log-odds (prevents runaway confidence).
    /// Default: 6.0 (~99.75% / 0.25% probability).
    pub max_abs_log_odds: f64,

    /// Confidence ceiling for volatile beliefs (volatility > threshold).
    /// Beliefs that change often can't reach full certainty.
    /// Default: max_abs=4.0 when volatility > 0.5.
    pub volatile_ceiling_log_odds: f64,
    pub volatility_threshold: f64,

    /// Propagation decay: how much evidence weight diminishes per hop.
    /// Default: 0.5 (each hop halves the weight).
    pub propagation_decay: f64,

    /// Maximum propagation hops. Default: 2.
    pub max_propagation_hops: u32,

    /// Minimum evidence weight to bother propagating. Default: 0.1.
    pub min_propagation_weight: f64,
}

impl Default for BeliefRevisionConfig {
    fn default() -> Self {
        Self {
            believed_threshold: 0.75,
            disbelieved_threshold: 0.25,
            staleness_rate_per_day: 0.02,
            staleness_grace_period_days: 7.0,
            max_abs_log_odds: 6.0,
            volatile_ceiling_log_odds: 4.0,
            volatility_threshold: 0.5,
            propagation_decay: 0.5,
            max_propagation_hops: 2,
            min_propagation_weight: 0.1,
        }
    }
}

// ── Core Belief Revision Functions ──

/// Assert a single piece of evidence against a belief node.
///
/// Updates the belief's log-odds using Bayesian updating with source reliability priors.
/// Returns the full result including threshold crossings and propagation effects.
///
/// # Arguments
/// * `node` - The belief node to update (must be a Belief payload)
/// * `evidence` - The evidence to assert
/// * `config` - Revision configuration
/// * `epistemic_edges` - Edges from this belief to other beliefs (for propagation)
/// * `downstream_nodes` - Mutable map of downstream belief nodes (for propagation writes)
pub fn assert_evidence(
    node: &mut CognitiveNode,
    evidence: &Evidence,
    config: &BeliefRevisionConfig,
) -> Option<EvidenceResult> {
    let belief = match &mut node.payload {
        NodePayload::Belief(b) => b,
        _ => return None,
    };

    let prior_log_odds = belief.log_odds;
    let prior_probability = crate::state::sigmoid(prior_log_odds);

    // Scale evidence by source reliability
    let reliability = evidence.provenance.reliability_prior();
    let effective_weight = evidence.weight * reliability;

    // Apply the Bayesian update
    belief.update(
        evidence.weight,
        reliability,
        &evidence.source,
        evidence.timestamp,
    );

    // Enforce confidence ceiling for volatile beliefs
    enforce_confidence_ceiling(belief, &node.attrs, config);

    // Enforce absolute log-odds cap
    belief.log_odds = belief
        .log_odds
        .clamp(-config.max_abs_log_odds, config.max_abs_log_odds);

    let posterior_log_odds = belief.log_odds;
    let posterior_probability = crate::state::sigmoid(posterior_log_odds);

    // Detect threshold crossings
    let threshold_direction =
        detect_threshold_crossing(prior_probability, posterior_probability, config);

    // Update cognitive attributes
    node.attrs.confidence = posterior_probability;
    node.attrs.evidence_count = belief.evidence_trail.len() as u32;
    node.attrs.touch(0.1); // Mild activation boost for updated beliefs

    // Increase volatility if evidence contradicts current direction
    if (prior_log_odds > 0.0 && effective_weight < 0.0)
        || (prior_log_odds < 0.0 && effective_weight > 0.0)
    {
        node.attrs.volatility = (node.attrs.volatility + 0.05).min(1.0);
    } else {
        // Confirming evidence reduces volatility slightly
        node.attrs.volatility = (node.attrs.volatility - 0.01).max(0.0);
    }

    Some(EvidenceResult {
        belief_id: evidence.target_belief,
        prior_log_odds,
        posterior_log_odds,
        prior_probability,
        posterior_probability,
        effective_weight,
        crossed_threshold: threshold_direction.is_some(),
        threshold_direction,
        propagated_to: vec![], // Propagation handled separately
    })
}

/// Propagate evidence through epistemic edges to downstream beliefs.
///
/// When belief A is updated with evidence, and A --Supports--> B exists,
/// the evidence propagates to B (attenuated by propagation_decay * edge_weight).
/// Contradicts edges flip the sign.
///
/// Returns a list of (downstream_node_id, propagated_weight) pairs.
pub fn propagate_evidence(
    source_id: NodeId,
    effective_weight: f64,
    edges: &[CognitiveEdge],
    downstream_nodes: &mut HashMap<NodeId, CognitiveNode>,
    config: &BeliefRevisionConfig,
    hop: u32,
) -> Vec<PropagationEffect> {
    if hop >= config.max_propagation_hops {
        return vec![];
    }

    let mut effects = Vec::new();

    for edge in edges {
        if edge.src != source_id {
            continue;
        }

        // Only propagate through epistemic edges
        let sign_multiplier = match edge.kind {
            CognitiveEdgeKind::Supports => 1.0,
            CognitiveEdgeKind::Contradicts => -1.0,
            _ => continue,
        };

        // Attenuate: weight * edge_weight * edge_confidence * decay
        let propagated_weight = effective_weight
            * sign_multiplier
            * edge.weight.abs()
            * edge.confidence
            * config.propagation_decay;

        if propagated_weight.abs() < config.min_propagation_weight {
            continue;
        }

        // Apply to downstream node
        if let Some(downstream) = downstream_nodes.get_mut(&edge.dst) {
            if let NodePayload::Belief(belief) = &mut downstream.payload {
                let old_lo = belief.log_odds;
                belief.log_odds += propagated_weight;
                belief.log_odds = belief
                    .log_odds
                    .clamp(-config.max_abs_log_odds, config.max_abs_log_odds);

                downstream.attrs.confidence = crate::state::sigmoid(belief.log_odds);

                effects.push(PropagationEffect {
                    belief_id: edge.dst,
                    edge_kind: edge.kind,
                    propagated_weight,
                    new_log_odds: belief.log_odds,
                });

                // Don't record in evidence_trail for propagated evidence
                // (it's derived, not primary). But do log the change.
                let _ = old_lo; // suppress unused warning — change is tracked in effects
            }
        }
    }

    effects
}

/// Apply staleness decay to a belief node.
///
/// Beliefs that haven't received new evidence for a while gradually
/// regress toward uncertainty (log_odds → 0). This prevents ancient
/// beliefs from dominating when they may no longer be valid.
///
/// Returns the amount of log-odds decayed (positive = moved toward 0).
pub fn apply_staleness_decay(
    node: &mut CognitiveNode,
    now_secs: f64,
    config: &BeliefRevisionConfig,
) -> f64 {
    let belief = match &mut node.payload {
        NodePayload::Belief(b) => b,
        _ => return 0.0,
    };

    // Find the most recent evidence timestamp
    let last_evidence_time = belief
        .evidence_trail
        .iter()
        .map(|e| e.timestamp)
        .fold(f64::NEG_INFINITY, f64::max);

    if last_evidence_time == f64::NEG_INFINITY {
        return 0.0; // No evidence at all — nothing to decay
    }

    let days_since_evidence = (now_secs - last_evidence_time) / 86400.0;

    if days_since_evidence < config.staleness_grace_period_days {
        return 0.0; // Within grace period
    }

    let effective_days = days_since_evidence - config.staleness_grace_period_days;
    let decay_factor = 1.0 - (config.staleness_rate_per_day * effective_days).min(0.95);

    let old_log_odds = belief.log_odds;
    belief.log_odds *= decay_factor;

    // Enforce ceiling after decay too
    enforce_confidence_ceiling(belief, &node.attrs, config);

    // Update confidence
    node.attrs.confidence = crate::state::sigmoid(belief.log_odds);

    (old_log_odds - belief.log_odds).abs()
}

/// Batch-revise all belief nodes: apply staleness decay and ceiling enforcement.
///
/// This is called during the `think()` loop to keep beliefs current.
/// Returns a summary of what changed.
pub fn revise_beliefs(
    beliefs: &mut [&mut CognitiveNode],
    now_secs: f64,
    config: &BeliefRevisionConfig,
) -> RevisionSummary {
    let mut summary = RevisionSummary::default();

    for node in beliefs.iter_mut() {
        if node.kind() != NodeKind::Belief {
            continue;
        }

        let decayed = apply_staleness_decay(node, now_secs, config);
        if decayed > 1e-6 {
            summary.staleness_decays += 1;
            summary.beliefs_revised += 1;
        }

        // Check ceiling enforcement
        if let NodePayload::Belief(belief) = &mut node.payload {
            let old_lo = belief.log_odds;
            enforce_confidence_ceiling(belief, &node.attrs, config);
            if (old_lo - belief.log_odds).abs() > 1e-6 {
                summary.ceiling_enforcements += 1;
                if decayed <= 1e-6 {
                    summary.beliefs_revised += 1;
                }
            }
        }
    }

    summary
}

// ── Internal Helpers ──

/// Enforce confidence ceiling for volatile beliefs.
fn enforce_confidence_ceiling(
    belief: &mut BeliefPayload,
    attrs: &CognitiveAttrs,
    config: &BeliefRevisionConfig,
) {
    if attrs.volatility > config.volatility_threshold {
        belief.log_odds = belief.log_odds.clamp(
            -config.volatile_ceiling_log_odds,
            config.volatile_ceiling_log_odds,
        );
    }
}

/// Detect if a probability crossed a significance threshold.
fn detect_threshold_crossing(
    prior: f64,
    posterior: f64,
    config: &BeliefRevisionConfig,
) -> Option<ThresholdDirection> {
    let was_believed = prior >= config.believed_threshold;
    let was_disbelieved = prior <= config.disbelieved_threshold;
    let is_believed = posterior >= config.believed_threshold;
    let is_disbelieved = posterior <= config.disbelieved_threshold;

    if was_believed && is_disbelieved || was_disbelieved && is_believed {
        Some(ThresholdDirection::Flipped)
    } else if !was_believed && is_believed {
        Some(ThresholdDirection::BecameConfident)
    } else if was_believed && !is_believed {
        Some(ThresholdDirection::BecameDoubtful)
    } else {
        None
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{sigmoid, NodeIdAllocator};

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
        node.attrs.confidence = sigmoid(log_odds);
        node
    }

    fn make_evidence(target: NodeId, weight: f64) -> Evidence {
        Evidence {
            target_belief: target,
            weight,
            source: "test observation".to_string(),
            provenance: Provenance::Observed,
            propagate: false,
            timestamp: 1000.0,
        }
    }

    #[test]
    fn test_assert_positive_evidence() {
        let mut alloc = NodeIdAllocator::new();
        let mut node = make_belief_node(&mut alloc, "User likes coffee", 0.0);
        let config = BeliefRevisionConfig::default();
        let evidence = make_evidence(node.id, 2.0);

        let result = assert_evidence(&mut node, &evidence, &config).unwrap();

        assert!(result.posterior_log_odds > result.prior_log_odds);
        assert!(result.posterior_probability > 0.5);
        assert!(result.effective_weight > 0.0);
        // Observed reliability = 0.90, so effective = 2.0 * 0.90 = 1.8
        assert!((result.effective_weight - 1.8).abs() < 0.01);
    }

    #[test]
    fn test_assert_negative_evidence() {
        let mut alloc = NodeIdAllocator::new();
        let mut node = make_belief_node(&mut alloc, "User dislikes tea", 2.0);
        let config = BeliefRevisionConfig::default();
        let evidence = make_evidence(node.id, -1.5);

        let result = assert_evidence(&mut node, &evidence, &config).unwrap();

        assert!(result.posterior_log_odds < result.prior_log_odds);
        assert!(result.posterior_probability < result.prior_probability);
    }

    #[test]
    fn test_threshold_crossing_became_confident() {
        let mut alloc = NodeIdAllocator::new();
        // Start at 0.5 probability (log_odds = 0.0)
        let mut node = make_belief_node(&mut alloc, "User exercises daily", 0.0);
        let config = BeliefRevisionConfig::default();
        // Strong positive evidence should push past 0.75 threshold
        let evidence = make_evidence(node.id, 3.0);

        let result = assert_evidence(&mut node, &evidence, &config).unwrap();

        assert!(result.crossed_threshold);
        assert_eq!(
            result.threshold_direction,
            Some(ThresholdDirection::BecameConfident)
        );
    }

    #[test]
    fn test_threshold_flip() {
        let mut alloc = NodeIdAllocator::new();
        // Start strongly believed (log_odds = 3.0, P ≈ 0.95)
        let mut node = make_belief_node(&mut alloc, "User prefers email", 3.0);
        let config = BeliefRevisionConfig::default();
        // Massive contradicting evidence
        let evidence = make_evidence(node.id, -8.0);

        let result = assert_evidence(&mut node, &evidence, &config).unwrap();

        assert!(result.crossed_threshold);
        assert_eq!(
            result.threshold_direction,
            Some(ThresholdDirection::Flipped)
        );
        assert!(result.posterior_probability < 0.25);
    }

    #[test]
    fn test_confidence_ceiling_volatile_belief() {
        let mut alloc = NodeIdAllocator::new();
        let mut node = make_belief_node(&mut alloc, "User mood is positive", 0.0);
        node.attrs.volatility = 0.8; // Highly volatile
        let config = BeliefRevisionConfig::default();

        // Try to push very high
        let evidence = Evidence {
            target_belief: node.id,
            weight: 10.0,
            source: "test".to_string(),
            provenance: Provenance::Told, // highest reliability = 0.95
            propagate: false,
            timestamp: 1000.0,
        };

        let result = assert_evidence(&mut node, &evidence, &config).unwrap();

        // Should be capped at volatile_ceiling_log_odds (4.0)
        assert!(result.posterior_log_odds <= config.volatile_ceiling_log_odds + 0.01);
    }

    #[test]
    fn test_staleness_decay_within_grace_period() {
        let mut alloc = NodeIdAllocator::new();
        let mut node = make_belief_node(&mut alloc, "Recent belief", 3.0);

        // Add evidence from 3 days ago (within default 7-day grace)
        if let NodePayload::Belief(b) = &mut node.payload {
            b.evidence_trail.push(EvidenceEntry {
                source: "test".to_string(),
                weight: 2.0,
                timestamp: 1000.0 - 3.0 * 86400.0,
            });
        }

        let config = BeliefRevisionConfig::default();
        let decayed = apply_staleness_decay(&mut node, 1000.0, &config);

        assert!(decayed < 1e-6, "should not decay within grace period");
    }

    #[test]
    fn test_staleness_decay_after_grace_period() {
        let mut alloc = NodeIdAllocator::new();
        let mut node = make_belief_node(&mut alloc, "Old belief", 3.0);

        // Evidence from 30 days ago
        if let NodePayload::Belief(b) = &mut node.payload {
            b.evidence_trail.push(EvidenceEntry {
                source: "test".to_string(),
                weight: 2.0,
                timestamp: 1000.0 - 30.0 * 86400.0,
            });
        }

        let config = BeliefRevisionConfig::default();
        let decayed = apply_staleness_decay(&mut node, 1000.0, &config);

        assert!(decayed > 0.1, "should decay after grace period");
        if let NodePayload::Belief(b) = &node.payload {
            assert!(b.log_odds < 3.0, "log-odds should have decreased");
            assert!(
                b.log_odds > 0.0,
                "should still be positive (not overcorrected)"
            );
        }
    }

    #[test]
    fn test_batch_revision() {
        let mut alloc = NodeIdAllocator::new();

        // Create beliefs with old evidence
        let mut nodes: Vec<CognitiveNode> = (0..5)
            .map(|i| {
                let mut n = make_belief_node(&mut alloc, &format!("Belief {i}"), 2.0);
                if let NodePayload::Belief(b) = &mut n.payload {
                    b.evidence_trail.push(EvidenceEntry {
                        source: "old".to_string(),
                        weight: 1.5,
                        timestamp: 1000.0 - 60.0 * 86400.0, // 60 days ago
                    });
                }
                n
            })
            .collect();

        let config = BeliefRevisionConfig::default();
        let mut refs: Vec<&mut CognitiveNode> = nodes.iter_mut().collect();
        let summary = revise_beliefs(&mut refs, 1000.0, &config);

        assert_eq!(summary.staleness_decays, 5);
        assert_eq!(summary.beliefs_revised, 5);
    }

    #[test]
    fn test_provenance_reliability_affects_update() {
        let mut alloc = NodeIdAllocator::new();

        // Same weight, different provenances
        let mut node_told = make_belief_node(&mut alloc, "Told belief", 0.0);
        let mut node_inferred = make_belief_node(&mut alloc, "Inferred belief", 0.0);

        let config = BeliefRevisionConfig::default();

        let ev_told = Evidence {
            target_belief: node_told.id,
            weight: 2.0,
            source: "user said".to_string(),
            provenance: Provenance::Told,
            propagate: false,
            timestamp: 1000.0,
        };

        let ev_inferred = Evidence {
            target_belief: node_inferred.id,
            weight: 2.0,
            source: "pattern detected".to_string(),
            provenance: Provenance::Inferred,
            propagate: false,
            timestamp: 1000.0,
        };

        let r_told = assert_evidence(&mut node_told, &ev_told, &config).unwrap();
        let r_inferred = assert_evidence(&mut node_inferred, &ev_inferred, &config).unwrap();

        // Told (0.95 reliability) should produce stronger update than Inferred (0.60)
        assert!(r_told.effective_weight > r_inferred.effective_weight);
        assert!(r_told.posterior_probability > r_inferred.posterior_probability);
    }

    #[test]
    fn test_evidence_propagation() {
        let mut alloc = NodeIdAllocator::new();
        let config = BeliefRevisionConfig::default();

        let id_a = alloc.alloc(NodeKind::Belief);
        let id_b = alloc.alloc(NodeKind::Belief);
        let id_c = alloc.alloc(NodeKind::Belief);

        // B is downstream of A via Supports
        // C is downstream of A via Contradicts
        let mut downstream = HashMap::new();
        downstream.insert(id_b, make_belief_node_with_id(id_b, "B supports A", 0.0));
        downstream.insert(id_c, make_belief_node_with_id(id_c, "C contradicts A", 0.0));

        let edges = vec![
            CognitiveEdge::new(id_a, id_b, CognitiveEdgeKind::Supports, 0.8),
            CognitiveEdge::new(id_a, id_c, CognitiveEdgeKind::Contradicts, 0.7),
        ];

        let effects = propagate_evidence(id_a, 2.0, &edges, &mut downstream, &config, 0);

        assert_eq!(effects.len(), 2);

        // B should have been pushed positive (Supports → same direction)
        let b = downstream.get(&id_b).unwrap();
        if let NodePayload::Belief(bel) = &b.payload {
            assert!(bel.log_odds > 0.0, "B should be positive");
        }

        // C should have been pushed negative (Contradicts → flipped direction)
        let c = downstream.get(&id_c).unwrap();
        if let NodePayload::Belief(bel) = &c.payload {
            assert!(bel.log_odds < 0.0, "C should be negative");
        }
    }

    #[test]
    fn test_max_log_odds_cap() {
        let mut alloc = NodeIdAllocator::new();
        let mut node = make_belief_node(&mut alloc, "Extreme belief", 5.0);
        let config = BeliefRevisionConfig::default();

        // Push way past cap
        let evidence = Evidence {
            target_belief: node.id,
            weight: 20.0,
            source: "extreme".to_string(),
            provenance: Provenance::Told,
            propagate: false,
            timestamp: 1000.0,
        };

        let result = assert_evidence(&mut node, &evidence, &config).unwrap();
        assert!(result.posterior_log_odds <= config.max_abs_log_odds);
    }

    #[test]
    fn test_contradicting_evidence_increases_volatility() {
        let mut alloc = NodeIdAllocator::new();
        let mut node = make_belief_node(&mut alloc, "Flip-flopping belief", 2.0);
        let initial_volatility = node.attrs.volatility;
        let config = BeliefRevisionConfig::default();

        // Contradicting evidence (belief is positive, evidence is negative)
        let evidence = make_evidence(node.id, -1.0);
        assert_evidence(&mut node, &evidence, &config);

        assert!(node.attrs.volatility > initial_volatility);
    }

    #[test]
    fn test_confirming_evidence_decreases_volatility() {
        let mut alloc = NodeIdAllocator::new();
        let mut node = make_belief_node(&mut alloc, "Stable belief", 2.0);
        node.attrs.volatility = 0.5;
        let config = BeliefRevisionConfig::default();

        // Confirming evidence (positive belief, positive evidence)
        let evidence = make_evidence(node.id, 1.0);
        assert_evidence(&mut node, &evidence, &config);

        assert!(node.attrs.volatility < 0.5);
    }

    #[test]
    fn test_non_belief_node_returns_none() {
        let mut alloc = NodeIdAllocator::new();
        let id = alloc.alloc(NodeKind::Entity);
        let mut node = CognitiveNode::new(
            id,
            "Not a belief".to_string(),
            NodePayload::Entity(crate::state::EntityPayload {
                name: "test".to_string(),
                entity_type: "person".to_string(),
                memory_rids: vec![],
            }),
        );

        let config = BeliefRevisionConfig::default();
        let evidence = Evidence {
            target_belief: id,
            weight: 1.0,
            source: "test".to_string(),
            provenance: Provenance::Observed,
            propagate: false,
            timestamp: 1000.0,
        };

        assert!(assert_evidence(&mut node, &evidence, &config).is_none());
    }

    // Helper: create belief node with specific ID
    fn make_belief_node_with_id(id: NodeId, prop: &str, log_odds: f64) -> CognitiveNode {
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
        node.attrs.confidence = sigmoid(log_odds);
        node
    }
}
