//! Belief Query & Explanation — Pattern matching and provenance chains.
//!
//! This module provides structured querying over the belief graph:
//! - **Pattern matching**: Find beliefs by domain, confidence range, provenance
//! - **Explanation**: Full provenance chain for why a belief is held
//! - **Evidence summary**: Aggregate support/contradiction statistics
//! - **Belief inventory**: Summary of the entire belief landscape

use std::collections::HashMap;

use crate::state::{
    BeliefPayload, CognitiveEdge, CognitiveEdgeKind, CognitiveNode, EvidenceEntry, NodeId,
    NodeKind, NodePayload, Provenance,
};

// ── Query Types ──

/// Pattern for querying beliefs.
#[derive(Debug, Clone, Default)]
pub struct BeliefPattern {
    /// Filter by domain (None = all domains).
    pub domain: Option<String>,
    /// Minimum probability threshold.
    pub min_probability: Option<f64>,
    /// Maximum probability threshold.
    pub max_probability: Option<f64>,
    /// Filter by provenance type.
    pub provenance: Option<Provenance>,
    /// Only user-confirmed beliefs.
    pub user_confirmed_only: bool,
    /// Minimum evidence count.
    pub min_evidence_count: Option<u32>,
    /// Maximum results.
    pub limit: usize,
    /// Sort order.
    pub order: BeliefOrder,
}

/// Sort order for belief queries.
#[derive(Debug, Clone, Copy, Default)]
pub enum BeliefOrder {
    /// Highest confidence first.
    #[default]
    ByConfidence,
    /// Most recent evidence first.
    ByRecency,
    /// Most evidence entries first.
    ByEvidenceCount,
    /// Highest salience first.
    BySalience,
    /// Most volatile first.
    ByVolatility,
}

/// Full explanation of why a belief is held.
#[derive(Debug, Clone)]
pub struct BeliefExplanation {
    /// The belief being explained.
    pub belief_id: NodeId,
    /// The proposition.
    pub proposition: String,
    /// Current probability.
    pub probability: f64,
    /// Current log-odds.
    pub log_odds: f64,
    /// Domain.
    pub domain: String,
    /// How this belief was originally created.
    pub provenance: Provenance,
    /// Whether the user explicitly confirmed it.
    pub user_confirmed: bool,
    /// All evidence supporting this belief.
    pub supporting_evidence: Vec<EvidenceSummary>,
    /// All evidence contradicting this belief.
    pub contradicting_evidence: Vec<EvidenceSummary>,
    /// Net support strength.
    pub net_support: f64,
    /// Beliefs that support this one (via Supports edges).
    pub supporting_beliefs: Vec<(NodeId, String, f64)>, // (id, label, edge_weight)
    /// Beliefs that contradict this one (via Contradicts edges).
    pub contradicting_beliefs: Vec<(NodeId, String, f64)>,
    /// Confidence trend: positive = getting more confident, negative = weakening.
    pub confidence_trend: f64,
    /// How volatile this belief has been.
    pub volatility: f64,
    /// Age of this belief in seconds.
    pub age_secs: f64,
}

/// Summary of a single piece of evidence.
#[derive(Debug, Clone)]
pub struct EvidenceSummary {
    /// Source description.
    pub source: String,
    /// Weight (positive = supports, negative = contradicts).
    pub weight: f64,
    /// When recorded.
    pub timestamp: f64,
    /// Age in seconds from now.
    pub age_secs: f64,
}

/// Aggregate statistics about the belief landscape.
#[derive(Debug, Clone, Default)]
pub struct BeliefInventory {
    /// Total number of beliefs.
    pub total_beliefs: usize,
    /// Beliefs grouped by domain.
    pub by_domain: HashMap<String, usize>,
    /// Number of high-confidence beliefs (P > 0.8).
    pub high_confidence: usize,
    /// Number of uncertain beliefs (0.3 < P < 0.7).
    pub uncertain: usize,
    /// Number of user-confirmed beliefs.
    pub user_confirmed: usize,
    /// Number of beliefs with contradicting evidence.
    pub contested: usize,
    /// Average evidence count per belief.
    pub avg_evidence_count: f64,
    /// Most volatile beliefs (top 5).
    pub most_volatile: Vec<(NodeId, String, f64)>, // (id, label, volatility)
    /// Strongest beliefs (top 5 by |log_odds|).
    pub strongest: Vec<(NodeId, String, f64)>, // (id, proposition, probability)
    /// Weakest contested beliefs (top 5 — high evidence but low confidence).
    pub most_contested: Vec<(NodeId, String, f64)>,
}

// ── Query Functions ──

/// Query beliefs matching a pattern from a set of cognitive nodes.
pub fn query_beliefs<'a>(
    nodes: impl Iterator<Item = &'a CognitiveNode>,
    pattern: &BeliefPattern,
) -> Vec<&'a CognitiveNode> {
    let mut results: Vec<&CognitiveNode> = nodes
        .filter(|n| n.kind() == NodeKind::Belief)
        .filter(|n| {
            let belief = match &n.payload {
                NodePayload::Belief(b) => b,
                _ => return false,
            };

            // Domain filter
            if let Some(ref domain) = pattern.domain {
                if &belief.domain != domain {
                    return false;
                }
            }

            // Probability range
            let prob = crate::state::sigmoid(belief.log_odds);
            if let Some(min_p) = pattern.min_probability {
                if prob < min_p {
                    return false;
                }
            }
            if let Some(max_p) = pattern.max_probability {
                if prob > max_p {
                    return false;
                }
            }

            // Provenance filter
            if let Some(prov) = pattern.provenance {
                if n.attrs.provenance != prov {
                    return false;
                }
            }

            // User confirmed filter
            if pattern.user_confirmed_only && !belief.user_confirmed {
                return false;
            }

            // Evidence count filter
            if let Some(min_ev) = pattern.min_evidence_count {
                if (belief.evidence_trail.len() as u32) < min_ev {
                    return false;
                }
            }

            true
        })
        .collect();

    // Sort
    match pattern.order {
        BeliefOrder::ByConfidence => {
            results.sort_by(|a, b| {
                b.attrs
                    .confidence
                    .partial_cmp(&a.attrs.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        BeliefOrder::ByRecency => {
            results.sort_by(|a, b| b.attrs.last_updated_ms.cmp(&a.attrs.last_updated_ms));
        }
        BeliefOrder::ByEvidenceCount => {
            results.sort_by(|a, b| b.attrs.evidence_count.cmp(&a.attrs.evidence_count));
        }
        BeliefOrder::BySalience => {
            results.sort_by(|a, b| {
                b.attrs
                    .salience
                    .partial_cmp(&a.attrs.salience)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        BeliefOrder::ByVolatility => {
            results.sort_by(|a, b| {
                b.attrs
                    .volatility
                    .partial_cmp(&a.attrs.volatility)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }

    results.truncate(pattern.limit.max(1));
    results
}

/// Generate a full explanation of a belief, including its provenance chain.
pub fn explain_belief(
    node: &CognitiveNode,
    edges: &[CognitiveEdge],
    neighbor_nodes: &HashMap<NodeId, &CognitiveNode>,
    now_secs: f64,
) -> Option<BeliefExplanation> {
    let belief = match &node.payload {
        NodePayload::Belief(b) => b,
        _ => return None,
    };

    let probability = crate::state::sigmoid(belief.log_odds);

    // Classify evidence
    let mut supporting = Vec::new();
    let mut contradicting = Vec::new();
    for entry in &belief.evidence_trail {
        let summary = EvidenceSummary {
            source: entry.source.clone(),
            weight: entry.weight,
            timestamp: entry.timestamp,
            age_secs: (now_secs - entry.timestamp).max(0.0),
        };
        if entry.weight >= 0.0 {
            supporting.push(summary);
        } else {
            contradicting.push(summary);
        }
    }

    // Net support
    let net_support = belief.support_strength() - belief.contradiction_strength();

    // Find supporting and contradicting beliefs via edges
    let mut supporting_beliefs = Vec::new();
    let mut contradicting_beliefs = Vec::new();

    for edge in edges {
        if edge.dst != node.id {
            continue;
        }

        if let Some(src_node) = neighbor_nodes.get(&edge.src) {
            match edge.kind {
                CognitiveEdgeKind::Supports => {
                    supporting_beliefs.push((edge.src, src_node.label.clone(), edge.weight));
                }
                CognitiveEdgeKind::Contradicts => {
                    contradicting_beliefs.push((edge.src, src_node.label.clone(), edge.weight));
                }
                _ => {}
            }
        }
    }

    // Confidence trend: compare recent evidence weights to older ones
    let confidence_trend = compute_confidence_trend(&belief.evidence_trail);

    Some(BeliefExplanation {
        belief_id: node.id,
        proposition: belief.proposition.clone(),
        probability,
        log_odds: belief.log_odds,
        domain: belief.domain.clone(),
        provenance: node.attrs.provenance,
        user_confirmed: belief.user_confirmed,
        supporting_evidence: supporting,
        contradicting_evidence: contradicting,
        net_support,
        supporting_beliefs,
        contradicting_beliefs,
        confidence_trend,
        volatility: node.attrs.volatility,
        age_secs: node.attrs.age_secs(),
    })
}

/// Build a comprehensive inventory of the belief landscape.
pub fn belief_inventory(nodes: &[&CognitiveNode]) -> BeliefInventory {
    let mut inv = BeliefInventory::default();
    let mut total_evidence = 0usize;
    let mut volatile_heap: Vec<(NodeId, String, f64)> = Vec::new();
    let mut strongest_heap: Vec<(NodeId, String, f64)> = Vec::new();
    let mut contested_heap: Vec<(NodeId, String, f64)> = Vec::new();

    for &node in nodes {
        if node.kind() != NodeKind::Belief {
            continue;
        }

        let belief = match &node.payload {
            NodePayload::Belief(b) => b,
            _ => continue,
        };

        inv.total_beliefs += 1;

        // Domain grouping
        *inv.by_domain.entry(belief.domain.clone()).or_insert(0) += 1;

        let prob = crate::state::sigmoid(belief.log_odds);

        // Confidence buckets
        if prob > 0.8 || prob < 0.2 {
            inv.high_confidence += 1;
        }
        if prob > 0.3 && prob < 0.7 {
            inv.uncertain += 1;
        }

        if belief.user_confirmed {
            inv.user_confirmed += 1;
        }

        let ev_count = belief.evidence_trail.len();
        total_evidence += ev_count;

        // Contested = has both supporting and contradicting evidence
        if belief.support_strength() > 0.0 && belief.contradiction_strength() > 0.0 {
            inv.contested += 1;
            contested_heap.push((node.id, belief.proposition.clone(), prob));
        }

        volatile_heap.push((node.id, node.label.clone(), node.attrs.volatility));
        strongest_heap.push((node.id, belief.proposition.clone(), prob));
    }

    if inv.total_beliefs > 0 {
        inv.avg_evidence_count = total_evidence as f64 / inv.total_beliefs as f64;
    }

    // Top-5 most volatile
    volatile_heap.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    inv.most_volatile = volatile_heap.into_iter().take(5).collect();

    // Top-5 strongest (by distance from 0.5)
    strongest_heap.sort_by(|a, b| {
        let dist_a = (a.2 - 0.5).abs();
        let dist_b = (b.2 - 0.5).abs();
        dist_b
            .partial_cmp(&dist_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    inv.strongest = strongest_heap.into_iter().take(5).collect();

    // Top-5 most contested (sorted by evidence count, lowest confidence)
    contested_heap.sort_by(|a, b| {
        let uncertainty_a = (a.2 - 0.5).abs();
        let uncertainty_b = (b.2 - 0.5).abs();
        uncertainty_a
            .partial_cmp(&uncertainty_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    inv.most_contested = contested_heap.into_iter().take(5).collect();

    inv
}

// ── Internal Helpers ──

/// Compute confidence trend from evidence trail.
/// Positive = evidence increasingly supporting, negative = increasingly contradicting.
fn compute_confidence_trend(evidence: &[EvidenceEntry]) -> f64 {
    if evidence.len() < 2 {
        return 0.0;
    }

    // Split evidence into halves by time
    let mut sorted: Vec<&EvidenceEntry> = evidence.iter().collect();
    sorted.sort_by(|a, b| {
        a.timestamp
            .partial_cmp(&b.timestamp)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mid = sorted.len() / 2;
    let early_avg: f64 = sorted[..mid].iter().map(|e| e.weight).sum::<f64>() / mid as f64;
    let late_avg: f64 =
        sorted[mid..].iter().map(|e| e.weight).sum::<f64>() / (sorted.len() - mid) as f64;

    late_avg - early_avg
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{sigmoid, NodeIdAllocator};

    fn make_belief(
        alloc: &mut NodeIdAllocator,
        prop: &str,
        log_odds: f64,
        domain: &str,
    ) -> CognitiveNode {
        let id = alloc.alloc(NodeKind::Belief);
        let mut node = CognitiveNode::new(
            id,
            prop.to_string(),
            NodePayload::Belief(BeliefPayload {
                proposition: prop.to_string(),
                log_odds,
                domain: domain.to_string(),
                evidence_trail: vec![],
                user_confirmed: false,
            }),
        );
        node.attrs.confidence = sigmoid(log_odds);
        node
    }

    #[test]
    fn test_query_by_domain() {
        let mut alloc = NodeIdAllocator::new();
        let a = make_belief(&mut alloc, "Coffee is good", 2.0, "food");
        let b = make_belief(&mut alloc, "Tea is better", 1.5, "food");
        let c = make_belief(&mut alloc, "Rain is nice", 1.0, "weather");

        let nodes = vec![a, b, c];
        let pattern = BeliefPattern {
            domain: Some("food".to_string()),
            limit: 10,
            ..Default::default()
        };

        let results = query_beliefs(nodes.iter(), &pattern);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_by_probability_range() {
        let mut alloc = NodeIdAllocator::new();
        let strong = make_belief(&mut alloc, "Strong", 3.0, "test"); // P ≈ 0.95
        let weak = make_belief(&mut alloc, "Weak", 0.2, "test"); // P ≈ 0.55
        let negative = make_belief(&mut alloc, "No", -2.0, "test"); // P ≈ 0.12

        let nodes = vec![strong, weak, negative];
        let pattern = BeliefPattern {
            min_probability: Some(0.8),
            limit: 10,
            ..Default::default()
        };

        let results = query_beliefs(nodes.iter(), &pattern);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].label, "Strong");
    }

    #[test]
    fn test_query_user_confirmed_only() {
        let mut alloc = NodeIdAllocator::new();
        let mut confirmed = make_belief(&mut alloc, "Confirmed", 2.0, "test");
        if let NodePayload::Belief(b) = &mut confirmed.payload {
            b.user_confirmed = true;
        }
        let unconfirmed = make_belief(&mut alloc, "Unconfirmed", 2.0, "test");

        let nodes = vec![confirmed, unconfirmed];
        let pattern = BeliefPattern {
            user_confirmed_only: true,
            limit: 10,
            ..Default::default()
        };

        let results = query_beliefs(nodes.iter(), &pattern);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].label, "Confirmed");
    }

    #[test]
    fn test_explain_belief() {
        let mut alloc = NodeIdAllocator::new();
        let mut node = make_belief(&mut alloc, "Coffee helps focus", 2.0, "health");

        // Add evidence trail
        if let NodePayload::Belief(b) = &mut node.payload {
            b.evidence_trail.push(EvidenceEntry {
                source: "morning observation".to_string(),
                weight: 1.5,
                timestamp: 500.0,
            });
            b.evidence_trail.push(EvidenceEntry {
                source: "afternoon crash".to_string(),
                weight: -0.3,
                timestamp: 700.0,
            });
            b.evidence_trail.push(EvidenceEntry {
                source: "productivity data".to_string(),
                weight: 0.8,
                timestamp: 900.0,
            });
        }

        let explanation = explain_belief(&node, &[], &HashMap::new(), 1000.0).unwrap();

        assert_eq!(explanation.proposition, "Coffee helps focus");
        assert!(explanation.probability > 0.8);
        assert_eq!(explanation.supporting_evidence.len(), 2);
        assert_eq!(explanation.contradicting_evidence.len(), 1);
        assert!(explanation.net_support > 0.0);
    }

    #[test]
    fn test_belief_inventory() {
        let mut alloc = NodeIdAllocator::new();
        let mut nodes = Vec::new();

        for i in 0..10 {
            let lo = if i < 5 { 3.0 } else { 0.1 };
            let domain = if i < 3 { "health" } else { "work" };
            nodes.push(make_belief(&mut alloc, &format!("Belief {i}"), lo, domain));
        }

        let refs: Vec<&CognitiveNode> = nodes.iter().collect();
        let inv = belief_inventory(&refs);

        assert_eq!(inv.total_beliefs, 10);
        assert_eq!(*inv.by_domain.get("health").unwrap(), 3);
        assert_eq!(*inv.by_domain.get("work").unwrap(), 7);
        assert!(inv.high_confidence >= 5); // The 5 with log_odds=3.0
    }

    #[test]
    fn test_confidence_trend() {
        let evidence = vec![
            EvidenceEntry {
                source: "old".into(),
                weight: -0.5,
                timestamp: 100.0,
            },
            EvidenceEntry {
                source: "old2".into(),
                weight: -0.3,
                timestamp: 200.0,
            },
            EvidenceEntry {
                source: "new".into(),
                weight: 1.0,
                timestamp: 800.0,
            },
            EvidenceEntry {
                source: "new2".into(),
                weight: 0.8,
                timestamp: 900.0,
            },
        ];

        let trend = compute_confidence_trend(&evidence);
        assert!(trend > 0.5, "trend should be positive (supporting later)");
    }
}
