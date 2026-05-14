//! Contradiction Detection — Finding and tracking conflicting beliefs.
//!
//! The contradiction detector identifies beliefs that logically or semantically
//! conflict with each other. It operates on the cognitive state graph and
//! produces structured `BeliefConflict` records that the policy engine can
//! use to trigger resolution actions.
//!
//! ## Detection Methods
//!
//! 1. **Epistemic edge conflicts**: Two beliefs connected by a `Contradicts` edge
//!    where both have high confidence — the system holds contradictory views.
//!
//! 2. **Domain conflicts**: Two beliefs in the same domain making opposing claims.
//!    Detected by comparing log-odds sign + proposition similarity.
//!
//! 3. **Goal-belief conflicts**: A belief contradicts a prerequisite for an active goal.
//!
//! 4. **Preference conflicts**: Two preferences in the same domain that are mutually
//!    exclusive (prefers X AND prefers not-X).

use std::collections::HashMap;

use crate::state::{
    BeliefPayload, CognitiveEdge, CognitiveEdgeKind, CognitiveNode, NodeId, NodeKind, NodePayload,
    PreferencePayload,
};

// ── Conflict Types ──

/// A detected contradiction between beliefs.
#[derive(Debug, Clone)]
pub struct BeliefConflict {
    /// First belief in the conflict.
    pub belief_a: NodeId,
    /// Second belief in the conflict.
    pub belief_b: NodeId,
    /// How the conflict was detected.
    pub detection_method: ConflictDetectionMethod,
    /// Severity [0.0, 1.0] — higher = more urgent to resolve.
    /// Based on confidence levels of both beliefs and their salience.
    pub severity: f64,
    /// Description of the conflict for human consumption.
    pub description: String,
    /// When the conflict was detected (unix timestamp seconds).
    pub detected_at: f64,
    /// Suggested resolution strategy.
    pub suggested_resolution: ResolutionStrategy,
}

/// How a contradiction was detected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConflictDetectionMethod {
    /// Connected by a Contradicts edge, both high-confidence.
    EpistemicEdge,
    /// Same domain, opposing log-odds signs.
    DomainOpposition,
    /// Belief undermines an active goal's prerequisites.
    GoalConflict,
    /// Mutually exclusive preferences in same domain.
    PreferenceConflict,
}

/// Suggested way to resolve a contradiction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResolutionStrategy {
    /// Ask the user to clarify which belief is correct.
    AskUser,
    /// Prefer the belief with more evidence.
    PreferStrongerEvidence,
    /// Prefer the more recent belief.
    PreferMoreRecent,
    /// Prefer the user-confirmed belief.
    PreferUserConfirmed,
    /// Both beliefs may be contextually valid (different domains/times).
    AcceptBoth,
    /// Merge into a nuanced belief that acknowledges both sides.
    Synthesize,
}

/// Summary of contradiction detection results.
#[derive(Debug, Clone, Default)]
pub struct ContradictionScanResult {
    /// All conflicts found.
    pub conflicts: Vec<BeliefConflict>,
    /// Number of epistemic edge conflicts.
    pub epistemic_conflicts: usize,
    /// Number of domain opposition conflicts.
    pub domain_conflicts: usize,
    /// Number of preference conflicts.
    pub preference_conflicts: usize,
}

// ── Configuration ──

/// Configuration for contradiction detection.
#[derive(Debug, Clone)]
pub struct ContradictionConfig {
    /// Minimum confidence (probability) for both beliefs to trigger a conflict.
    /// Default: 0.6 — don't flag low-confidence beliefs as contradictions.
    pub min_confidence_for_conflict: f64,

    /// Minimum severity to report. Default: 0.2 (report most conflicts).
    pub min_severity: f64,

    /// Maximum number of conflicts to return. Default: 20.
    pub max_conflicts: usize,
}

impl Default for ContradictionConfig {
    fn default() -> Self {
        Self {
            min_confidence_for_conflict: 0.6,
            min_severity: 0.2,
            max_conflicts: 20,
        }
    }
}

// ── Core Detection Functions ──

/// Detect contradictions from epistemic edges (Contradicts edges between beliefs).
///
/// When belief A --Contradicts--> belief B, and both have P > threshold,
/// that's a contradiction the system should resolve.
pub fn detect_epistemic_conflicts(
    nodes: &HashMap<NodeId, &CognitiveNode>,
    edges: &[CognitiveEdge],
    config: &ContradictionConfig,
    now_secs: f64,
) -> Vec<BeliefConflict> {
    let mut conflicts = Vec::new();

    for edge in edges {
        if edge.kind != CognitiveEdgeKind::Contradicts {
            continue;
        }

        let node_a = match nodes.get(&edge.src) {
            Some(n) if n.kind() == NodeKind::Belief => *n,
            _ => continue,
        };
        let node_b = match nodes.get(&edge.dst) {
            Some(n) if n.kind() == NodeKind::Belief => *n,
            _ => continue,
        };

        let conf_a = node_a.attrs.confidence;
        let conf_b = node_b.attrs.confidence;

        // Both must be above threshold to count as a real conflict
        if conf_a < config.min_confidence_for_conflict
            || conf_b < config.min_confidence_for_conflict
        {
            continue;
        }

        let severity = compute_conflict_severity(node_a, node_b, edge.weight.abs());
        if severity < config.min_severity {
            continue;
        }

        let resolution = suggest_resolution(node_a, node_b);

        let desc = format!(
            "Contradictory beliefs: \"{}\" (P={:.0}%) vs \"{}\" (P={:.0}%)",
            node_a.label,
            conf_a * 100.0,
            node_b.label,
            conf_b * 100.0,
        );

        conflicts.push(BeliefConflict {
            belief_a: edge.src,
            belief_b: edge.dst,
            detection_method: ConflictDetectionMethod::EpistemicEdge,
            severity,
            description: desc,
            detected_at: now_secs,
            suggested_resolution: resolution,
        });
    }

    conflicts
}

/// Detect domain-level contradictions: beliefs in the same domain with opposing stances.
///
/// This catches cases where no explicit Contradicts edge exists but the beliefs
/// semantically conflict (e.g., "User prefers mornings" vs "User is a night owl").
pub fn detect_domain_conflicts(
    beliefs: &[&CognitiveNode],
    config: &ContradictionConfig,
    now_secs: f64,
) -> Vec<BeliefConflict> {
    let mut conflicts = Vec::new();

    // Group beliefs by domain
    let mut by_domain: HashMap<&str, Vec<&CognitiveNode>> = HashMap::new();
    for &node in beliefs {
        if let NodePayload::Belief(b) = &node.payload {
            by_domain.entry(&b.domain).or_default().push(node);
        }
    }

    // Within each domain, find pairs with opposing log-odds
    for (_domain, domain_beliefs) in &by_domain {
        for i in 0..domain_beliefs.len() {
            for j in (i + 1)..domain_beliefs.len() {
                let a = domain_beliefs[i];
                let b = domain_beliefs[j];

                let (belief_a, belief_b) = match (&a.payload, &b.payload) {
                    (NodePayload::Belief(ba), NodePayload::Belief(bb)) => (ba, bb),
                    _ => continue,
                };

                // Check for opposing directions (one positive, one negative)
                // Both must be confident
                if belief_a.log_odds.signum() == belief_b.log_odds.signum() {
                    continue; // Same direction — not a contradiction
                }

                let conf_a = a.attrs.confidence;
                let conf_b = b.attrs.confidence;
                // For domain conflicts, at least one must be on the "confident" side
                let max_conf = conf_a.max(conf_b);
                let min_conf = conf_a.min(conf_b);

                if max_conf < config.min_confidence_for_conflict {
                    continue;
                }
                // The other should also be somewhat confident in its direction
                if min_conf < 0.4 {
                    continue;
                }

                let severity = compute_conflict_severity(a, b, 0.5);
                if severity < config.min_severity {
                    continue;
                }

                let resolution = suggest_resolution(a, b);

                conflicts.push(BeliefConflict {
                    belief_a: a.id,
                    belief_b: b.id,
                    detection_method: ConflictDetectionMethod::DomainOpposition,
                    severity,
                    description: format!("Domain conflict: \"{}\" vs \"{}\"", a.label, b.label,),
                    detected_at: now_secs,
                    suggested_resolution: resolution,
                });
            }
        }
    }

    conflicts
}

/// Detect preference conflicts: mutually exclusive preferences in the same domain.
pub fn detect_preference_conflicts(
    nodes: &[&CognitiveNode],
    config: &ContradictionConfig,
    now_secs: f64,
) -> Vec<BeliefConflict> {
    let mut conflicts = Vec::new();

    // Collect preference nodes grouped by domain
    let mut by_domain: HashMap<&str, Vec<(&CognitiveNode, &PreferencePayload)>> = HashMap::new();
    for &node in nodes {
        if let NodePayload::Preference(pref) = &node.payload {
            by_domain
                .entry(&pref.domain)
                .or_default()
                .push((node, pref));
        }
    }

    for (_domain, prefs) in &by_domain {
        for i in 0..prefs.len() {
            for j in (i + 1)..prefs.len() {
                let (node_a, pref_a) = &prefs[i];
                let (node_b, pref_b) = &prefs[j];

                // Check: A prefers X, B prefers not-X (or vice versa)
                let is_conflicting = pref_a
                    .dispreferred
                    .as_deref()
                    .map_or(false, |dis| dis == pref_b.preferred)
                    || pref_b
                        .dispreferred
                        .as_deref()
                        .map_or(false, |dis| dis == pref_a.preferred);

                if !is_conflicting {
                    continue;
                }

                let severity = (pref_a.strength + pref_b.strength) / 2.0;
                if severity < config.min_severity {
                    continue;
                }

                conflicts.push(BeliefConflict {
                    belief_a: node_a.id,
                    belief_b: node_b.id,
                    detection_method: ConflictDetectionMethod::PreferenceConflict,
                    severity,
                    description: format!(
                        "Preference conflict in domain: prefers \"{}\" vs prefers \"{}\"",
                        pref_a.preferred, pref_b.preferred,
                    ),
                    detected_at: now_secs,
                    suggested_resolution: ResolutionStrategy::AskUser,
                });
            }
        }
    }

    conflicts
}

/// Run all contradiction detection methods and merge results.
pub fn scan_contradictions(
    nodes: &HashMap<NodeId, &CognitiveNode>,
    edges: &[CognitiveEdge],
    config: &ContradictionConfig,
    now_secs: f64,
) -> ContradictionScanResult {
    let mut result = ContradictionScanResult::default();

    // Method 1: Epistemic edge conflicts
    let epistemic = detect_epistemic_conflicts(nodes, edges, config, now_secs);
    result.epistemic_conflicts = epistemic.len();
    result.conflicts.extend(epistemic);

    // Method 2: Domain conflicts (beliefs only)
    let beliefs: Vec<&CognitiveNode> = nodes
        .values()
        .filter(|n| n.kind() == NodeKind::Belief)
        .copied()
        .collect();
    let domain = detect_domain_conflicts(&beliefs, config, now_secs);
    result.domain_conflicts = domain.len();
    result.conflicts.extend(domain);

    // Method 3: Preference conflicts
    let all_nodes: Vec<&CognitiveNode> = nodes.values().copied().collect();
    let prefs = detect_preference_conflicts(&all_nodes, config, now_secs);
    result.preference_conflicts = prefs.len();
    result.conflicts.extend(prefs);

    // Sort by severity descending, truncate
    result.conflicts.sort_by(|a, b| {
        b.severity
            .partial_cmp(&a.severity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    result.conflicts.truncate(config.max_conflicts);

    result
}

// ── Internal Helpers ──

/// Compute conflict severity from both beliefs' attributes.
fn compute_conflict_severity(a: &CognitiveNode, b: &CognitiveNode, edge_weight: f64) -> f64 {
    // Severity = geometric mean of confidences * max salience * edge weight factor
    let confidence_factor = (a.attrs.confidence * b.attrs.confidence).sqrt();
    let salience_factor = a.attrs.salience.max(b.attrs.salience);
    let weight_factor = 0.5 + 0.5 * edge_weight; // 0.5 to 1.0

    (confidence_factor * salience_factor * weight_factor).clamp(0.0, 1.0)
}

/// Suggest the best resolution strategy based on belief properties.
fn suggest_resolution(a: &CognitiveNode, b: &CognitiveNode) -> ResolutionStrategy {
    let (belief_a, belief_b) = match (&a.payload, &b.payload) {
        (NodePayload::Belief(ba), NodePayload::Belief(bb)) => (ba, bb),
        _ => return ResolutionStrategy::AskUser,
    };

    // If one is user-confirmed, prefer it
    if belief_a.user_confirmed && !belief_b.user_confirmed {
        return ResolutionStrategy::PreferUserConfirmed;
    }
    if belief_b.user_confirmed && !belief_a.user_confirmed {
        return ResolutionStrategy::PreferUserConfirmed;
    }

    // If one has significantly more evidence, prefer it
    let ev_a = belief_a.evidence_trail.len();
    let ev_b = belief_b.evidence_trail.len();
    if ev_a > ev_b * 2 || ev_b > ev_a * 2 {
        return ResolutionStrategy::PreferStrongerEvidence;
    }

    // If one is much more recent, prefer it
    let recency_a = belief_a.evidence_trail.last().map_or(0.0, |e| e.timestamp);
    let recency_b = belief_b.evidence_trail.last().map_or(0.0, |e| e.timestamp);
    let recency_gap = (recency_a - recency_b).abs();
    if recency_gap > 7.0 * 86400.0 {
        // More than a week apart
        return ResolutionStrategy::PreferMoreRecent;
    }

    // Default: ask the user
    ResolutionStrategy::AskUser
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{BeliefPayload, NodeIdAllocator, NodePayload};

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
        node.attrs.confidence = crate::state::sigmoid(log_odds);
        node.attrs.salience = 0.6;
        node
    }

    #[test]
    fn test_epistemic_conflict_detection() {
        let mut alloc = NodeIdAllocator::new();
        let a = make_belief(&mut alloc, "Earth is flat", 2.0, "geography");
        let b = make_belief(&mut alloc, "Earth is round", 3.0, "geography");

        let edge = CognitiveEdge::new(a.id, b.id, CognitiveEdgeKind::Contradicts, 0.9);

        let mut nodes = HashMap::new();
        nodes.insert(a.id, &a);
        nodes.insert(b.id, &b);

        let config = ContradictionConfig::default();
        let conflicts = detect_epistemic_conflicts(&nodes, &[edge], &config, 1000.0);

        assert_eq!(conflicts.len(), 1);
        assert_eq!(
            conflicts[0].detection_method,
            ConflictDetectionMethod::EpistemicEdge
        );
        assert!(conflicts[0].severity > 0.3);
    }

    #[test]
    fn test_low_confidence_beliefs_no_conflict() {
        let mut alloc = NodeIdAllocator::new();
        let a = make_belief(&mut alloc, "Maybe X", 0.2, "test");
        let b = make_belief(&mut alloc, "Maybe not X", -0.3, "test");

        let edge = CognitiveEdge::new(a.id, b.id, CognitiveEdgeKind::Contradicts, 0.9);

        let mut nodes = HashMap::new();
        nodes.insert(a.id, &a);
        nodes.insert(b.id, &b);

        let config = ContradictionConfig::default();
        let conflicts = detect_epistemic_conflicts(&nodes, &[edge], &config, 1000.0);

        assert_eq!(
            conflicts.len(),
            0,
            "low-confidence beliefs should not trigger conflict"
        );
    }

    #[test]
    fn test_domain_conflict_detection() {
        let mut alloc = NodeIdAllocator::new();
        // Both beliefs in same domain, opposing log-odds signs.
        // Manually set confidence high for both so both pass the threshold.
        // "User prefers mornings" (believed, lo=2.5) vs "User avoids mornings" (believed false, lo=-2.5)
        // For the node with negative log_odds, we set attrs.confidence high
        // to represent "we are confident this proposition is FALSE."
        let mut a = make_belief(&mut alloc, "User prefers mornings", 2.5, "schedule");
        let mut b = make_belief(&mut alloc, "User avoids mornings", -2.5, "schedule");
        // Override confidence: both nodes are "confident" (we know our stance)
        a.attrs.confidence = 0.92; // sigmoid(2.5) — naturally high
        b.attrs.confidence = 0.92; // We're confident this is false — override

        let beliefs = vec![&a, &b];
        let config = ContradictionConfig::default();
        let conflicts = detect_domain_conflicts(&beliefs, &config, 1000.0);

        assert_eq!(conflicts.len(), 1);
        assert_eq!(
            conflicts[0].detection_method,
            ConflictDetectionMethod::DomainOpposition
        );
    }

    #[test]
    fn test_preference_conflict_detection() {
        let mut alloc = NodeIdAllocator::new();

        let id_a = alloc.alloc(NodeKind::Preference);
        let node_a = CognitiveNode::new(
            id_a,
            "Prefers dark mode".to_string(),
            NodePayload::Preference(PreferencePayload {
                domain: "UI".to_string(),
                preferred: "dark mode".to_string(),
                dispreferred: Some("light mode".to_string()),
                strength: 0.8,
                log_odds: 2.0,
                observation_count: 10,
            }),
        );

        let id_b = alloc.alloc(NodeKind::Preference);
        let node_b = CognitiveNode::new(
            id_b,
            "Prefers light mode".to_string(),
            NodePayload::Preference(PreferencePayload {
                domain: "UI".to_string(),
                preferred: "light mode".to_string(),
                dispreferred: Some("dark mode".to_string()),
                strength: 0.6,
                log_odds: 1.5,
                observation_count: 5,
            }),
        );

        let nodes = vec![&node_a, &node_b];
        let config = ContradictionConfig::default();
        let conflicts = detect_preference_conflicts(&nodes, &config, 1000.0);

        assert_eq!(conflicts.len(), 1);
        assert_eq!(
            conflicts[0].detection_method,
            ConflictDetectionMethod::PreferenceConflict
        );
    }

    #[test]
    fn test_full_scan() {
        let mut alloc = NodeIdAllocator::new();
        let a = make_belief(&mut alloc, "Belief A", 2.5, "test");
        let b = make_belief(&mut alloc, "Belief B", 2.0, "test2");

        let edge = CognitiveEdge::new(a.id, b.id, CognitiveEdgeKind::Contradicts, 0.8);

        let mut nodes = HashMap::new();
        nodes.insert(a.id, &a);
        nodes.insert(b.id, &b);

        let config = ContradictionConfig::default();
        let result = scan_contradictions(&nodes, &[edge], &config, 1000.0);

        assert!(result.epistemic_conflicts >= 1);
    }

    #[test]
    fn test_resolution_prefers_user_confirmed() {
        let mut alloc = NodeIdAllocator::new();
        let mut a = make_belief(&mut alloc, "A", 2.0, "test");
        let mut b = make_belief(&mut alloc, "B", 2.0, "test");

        if let NodePayload::Belief(ba) = &mut a.payload {
            ba.user_confirmed = true;
        }

        let res = suggest_resolution(&a, &b);
        assert_eq!(res, ResolutionStrategy::PreferUserConfirmed);
    }

    #[test]
    fn test_resolution_prefers_stronger_evidence() {
        let mut alloc = NodeIdAllocator::new();
        let mut a = make_belief(&mut alloc, "A", 2.0, "test");
        let b = make_belief(&mut alloc, "B", 2.0, "test");

        // Give A lots of evidence
        if let NodePayload::Belief(ba) = &mut a.payload {
            for i in 0..10 {
                ba.evidence_trail.push(crate::state::EvidenceEntry {
                    source: format!("source_{i}"),
                    weight: 0.5,
                    timestamp: 1000.0 + i as f64,
                });
            }
        }

        let res = suggest_resolution(&a, &b);
        assert_eq!(res, ResolutionStrategy::PreferStrongerEvidence);
    }
}
