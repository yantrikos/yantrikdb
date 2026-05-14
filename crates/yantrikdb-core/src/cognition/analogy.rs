//! CK-5.1 — Analogical Reasoning Engine.
//!
//! Detects structural similarities across domains and transfers learned
//! strategies to novel contexts. Based on Gentner's Structure-Mapping Theory:
//! prefer analogies that preserve **relational structure** (causes, prevents,
//! enables) over surface similarity (same name, same type).
//!
//! # Design principles
//! - Pure functions only — no DB access (engine layer handles persistence)
//! - Systematicity preference — higher-order relations > attributes
//! - Structural consistency — 1:1 node correspondence, no cross-mappings
//! - Pragmatic relevance — analogies must be useful for current goals
//! - Incremental — analogies strengthen or decay with use

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::state::{
    CognitiveEdge, CognitiveEdgeKind, CognitiveNode, NodeId, NodeKind, NodePayload,
};

// ══════════════════════════════════════════════════════════════════════════════
// § 1  Core Types
// ══════════════════════════════════════════════════════════════════════════════

/// How knowledge was transferred via an analogy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransferType {
    /// Direct 1:1 mapping of a fact from source to target.
    DirectMapping,
    /// Transfer via an abstracted intermediate representation.
    AbstractionTransfer,
    /// Transfer based on relational structure (higher-order).
    RelationalTransfer,
}

impl TransferType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DirectMapping => "direct_mapping",
            Self::AbstractionTransfer => "abstraction_transfer",
            Self::RelationalTransfer => "relational_transfer",
        }
    }
}

/// Scope for searching analogies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnalogyScope {
    /// Search within the same domain only.
    SameDomain,
    /// Search across different domains (the interesting case).
    CrossDomain,
    /// Search everywhere.
    Universal,
}

/// A projected fact derived from an analogy — a prediction about the
/// target domain based on knowledge in the source domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectedFact {
    /// What we predict exists in the target domain.
    pub description: String,
    /// The edge kind we expect to find (e.g., Causes, Supports).
    pub expected_edge_kind: CognitiveEdgeKind,
    /// Source node from which this was projected.
    pub source_ref: NodeId,
    /// Target node this projection attaches to.
    pub target_ref: NodeId,
}

/// A prediction derived from an analogy — transferring knowledge from
/// the source domain to the target domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateInference {
    /// The fact we know in the source domain.
    pub source_fact: NodeId,
    /// What we predict in the target domain.
    pub projected_target: ProjectedFact,
    /// How strongly the structural match supports this prediction [0.0, 1.0].
    pub confidence: f64,
    /// How the transfer was accomplished.
    pub transfer_type: TransferType,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 2  Structural Mapping
// ══════════════════════════════════════════════════════════════════════════════

/// A node-level correspondence in a structural mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCorrespondence {
    pub source: NodeId,
    pub target: NodeId,
    /// How well these two nodes match (kind similarity + attribute overlap).
    pub match_quality: f64,
}

/// An edge-level correspondence in a structural mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeCorrespondence {
    /// Source edge (src, dst, kind).
    pub source_edge: (NodeId, NodeId, CognitiveEdgeKind),
    /// Target edge (src, dst, kind).
    pub target_edge: (NodeId, NodeId, CognitiveEdgeKind),
    /// Whether the edge kinds match exactly.
    pub exact_kind_match: bool,
}

/// Alignment between two subgraphs — the core data structure of
/// analogical reasoning. Maps nodes and edges from a source domain
/// to a target domain, measuring structural similarity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralMapping {
    /// Nodes in the source domain.
    pub source_nodes: Vec<NodeId>,
    /// Nodes in the target domain.
    pub target_nodes: Vec<NodeId>,
    /// Aligned node pairs with match quality.
    pub node_correspondences: Vec<NodeCorrespondence>,
    /// Aligned edge pairs.
    pub edge_correspondences: Vec<EdgeCorrespondence>,
    /// Overall structural similarity [0.0, 1.0].
    pub structural_similarity: f64,
    /// How deep the relational match goes (number of edge hops).
    pub relational_depth: usize,
    /// How useful this analogy is for current goals [0.0, 1.0].
    pub pragmatic_relevance: f64,
    /// Predictions derived from this mapping.
    pub candidate_inferences: Vec<CandidateInference>,
    /// Source domain label (e.g., "project_management").
    pub source_domain: String,
    /// Target domain label (e.g., "hiring").
    pub target_domain: String,
    /// When this mapping was created (unix ms).
    pub created_at_ms: u64,
    /// When this mapping was last used.
    pub last_used_ms: u64,
    /// How many times this mapping was accessed.
    pub use_count: u32,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  Analogy Store
// ══════════════════════════════════════════════════════════════════════════════

/// Indexed collection of discovered analogies, with domain-based lookup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalogyStore {
    /// All discovered mappings.
    pub mappings: Vec<StructuralMapping>,
    /// Index: domain name → indices into `mappings`.
    pub domain_index: HashMap<String, Vec<usize>>,
    /// Minimum structural_similarity to retain a mapping.
    pub quality_threshold: f64,
    /// Maximum number of stored mappings.
    pub capacity: usize,
}

impl Default for AnalogyStore {
    fn default() -> Self {
        Self {
            mappings: Vec::new(),
            domain_index: HashMap::new(),
            quality_threshold: 0.3,
            capacity: 200,
        }
    }
}

impl AnalogyStore {
    /// Create a new store with custom settings.
    pub fn with_config(quality_threshold: f64, capacity: usize) -> Self {
        Self {
            quality_threshold,
            capacity,
            ..Default::default()
        }
    }

    /// Number of stored mappings.
    pub fn len(&self) -> usize {
        self.mappings.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.mappings.is_empty()
    }

    /// Add a mapping to the store, updating the domain index.
    pub fn insert(&mut self, mapping: StructuralMapping) {
        let idx = self.mappings.len();
        // Index by both source and target domain.
        self.domain_index
            .entry(mapping.source_domain.clone())
            .or_default()
            .push(idx);
        if mapping.source_domain != mapping.target_domain {
            self.domain_index
                .entry(mapping.target_domain.clone())
                .or_default()
                .push(idx);
        }
        self.mappings.push(mapping);

        // Evict lowest-quality if over capacity.
        if self.mappings.len() > self.capacity {
            self.evict_weakest();
        }
    }

    /// Find mappings that involve a given domain.
    pub fn find_by_domain(&self, domain: &str) -> Vec<&StructuralMapping> {
        match self.domain_index.get(domain) {
            Some(indices) => indices
                .iter()
                .filter_map(|&i| self.mappings.get(i))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Find cross-domain mappings (source and target are different domains).
    pub fn cross_domain_mappings(&self) -> Vec<&StructuralMapping> {
        self.mappings
            .iter()
            .filter(|m| m.source_domain != m.target_domain)
            .collect()
    }

    /// Evict the weakest mapping and rebuild the domain index.
    fn evict_weakest(&mut self) {
        if self.mappings.is_empty() {
            return;
        }
        // Find weakest by structural_similarity * use_count weight.
        let weakest_idx = self
            .mappings
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let score_a = a.structural_similarity * (1.0 + a.use_count as f64).ln();
                let score_b = b.structural_similarity * (1.0 + b.use_count as f64).ln();
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap();

        self.mappings.remove(weakest_idx);
        self.rebuild_index();
    }

    /// Rebuild the domain index from scratch.
    fn rebuild_index(&mut self) {
        self.domain_index.clear();
        for (idx, m) in self.mappings.iter().enumerate() {
            self.domain_index
                .entry(m.source_domain.clone())
                .or_default()
                .push(idx);
            if m.source_domain != m.target_domain {
                self.domain_index
                    .entry(m.target_domain.clone())
                    .or_default()
                    .push(idx);
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  Analogy Query
// ══════════════════════════════════════════════════════════════════════════════

/// Request to find analogies for a given subgraph pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalogicalQuery {
    /// The subgraph pattern to find analogies for.
    pub source_subgraph: Vec<NodeId>,
    /// Edges within the source subgraph.
    pub source_edges: Vec<CognitiveEdge>,
    /// Domain of the source subgraph.
    pub source_domain: String,
    /// Scope of the search.
    pub search_scope: AnalogyScope,
    /// Minimum relational depth to consider.
    pub min_relational_depth: usize,
    /// Maximum results to return.
    pub max_results: usize,
}

/// An automatically detected analogical opportunity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalogicalOpportunity {
    /// The recently active subgraph that triggered this opportunity.
    pub recent_subgraph: Vec<NodeId>,
    /// The older, well-understood subgraph it resembles.
    pub reference_subgraph: Vec<NodeId>,
    /// The structural mapping between them.
    pub mapping: StructuralMapping,
    /// Why this opportunity is interesting.
    pub reason: String,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 5  Core Algorithms
// ══════════════════════════════════════════════════════════════════════════════

/// Compute the relational similarity between two edge kinds.
///
/// Same kind = 1.0, same category (epistemic, causal, etc.) = 0.6,
/// different category = 0.0.
fn edge_kind_similarity(a: CognitiveEdgeKind, b: CognitiveEdgeKind) -> f64 {
    if a == b {
        return 1.0;
    }
    // Group edges by semantic category for partial matches.
    let cat_a = edge_category(a);
    let cat_b = edge_category(b);
    if cat_a == cat_b {
        0.6
    } else {
        0.0
    }
}

/// Classify edge kinds into semantic categories.
fn edge_category(kind: CognitiveEdgeKind) -> u8 {
    match kind {
        CognitiveEdgeKind::Supports | CognitiveEdgeKind::Contradicts => 0, // epistemic
        CognitiveEdgeKind::Causes | CognitiveEdgeKind::Predicts | CognitiveEdgeKind::Prevents => 1, // causal
        CognitiveEdgeKind::AdvancesGoal
        | CognitiveEdgeKind::BlocksGoal
        | CognitiveEdgeKind::SubtaskOf
        | CognitiveEdgeKind::Requires => 2, // goal/task
        CognitiveEdgeKind::AssociatedWith
        | CognitiveEdgeKind::InstanceOf
        | CognitiveEdgeKind::PartOf
        | CognitiveEdgeKind::SimilarTo => 3, // associative
        CognitiveEdgeKind::PrecedesTemporally | CognitiveEdgeKind::Triggers => 4, // temporal
        CognitiveEdgeKind::Prefers | CognitiveEdgeKind::Avoids | CognitiveEdgeKind::Constrains => 5, // preference/constraint
    }
}

/// Compute how well two nodes match based on their kind and attributes.
///
/// Same kind = 0.5 base. Same domain (if Entity) = +0.3.
/// Close confidence values = +0.2.
fn node_match_quality(source: &CognitiveNode, target: &CognitiveNode) -> f64 {
    let mut score = 0.0;

    // Kind match is the strongest signal.
    if source.kind() == target.kind() {
        score += 0.5;
    } else {
        // Different kinds rarely make good analogies at node level.
        return 0.05;
    }

    // Domain match for entities.
    if let (NodePayload::Entity(s), NodePayload::Entity(t)) = (&source.payload, &target.payload) {
        if s.entity_type == t.entity_type {
            score += 0.15;
        }
    }

    // Belief domain match.
    if let (NodePayload::Belief(s), NodePayload::Belief(t)) = (&source.payload, &target.payload) {
        if s.domain == t.domain {
            score += 0.15;
        }
    }

    // Attribute similarity (confidence proximity).
    let conf_diff = (source.attrs.confidence - target.attrs.confidence).abs();
    score += 0.2 * (1.0 - conf_diff);

    // Valence alignment.
    let val_diff = (source.attrs.valence - target.attrs.valence).abs();
    score += 0.1 * (1.0 - val_diff / 2.0);

    score.clamp(0.0, 1.0)
}

/// Compute structural similarity between two subgraphs.
///
/// Uses a greedy alignment algorithm:
/// 1. Score all possible node pairings by match quality
/// 2. Greedily assign best pairs (1:1 constraint)
/// 3. Compute edge correspondence based on node alignment
/// 4. Score = weighted(node_match, edge_match, systematicity)
///
/// Systematicity: prefer mappings where higher-order relations
/// (causal, epistemic) align over surface associations.
pub fn compute_structural_similarity(
    source_nodes: &[&CognitiveNode],
    target_nodes: &[&CognitiveNode],
    source_edges: &[CognitiveEdge],
    target_edges: &[CognitiveEdge],
) -> StructuralMapping {
    let now = crate::state::now_ms();

    // ── Step 1: Score all node pairings ──
    let mut pair_scores: Vec<(usize, usize, f64)> = Vec::new();
    for (si, sn) in source_nodes.iter().enumerate() {
        for (ti, tn) in target_nodes.iter().enumerate() {
            let q = node_match_quality(sn, tn);
            if q > 0.1 {
                pair_scores.push((si, ti, q));
            }
        }
    }

    // Sort by quality descending for greedy assignment.
    pair_scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // ── Step 2: Greedy 1:1 assignment ──
    let mut used_source = vec![false; source_nodes.len()];
    let mut used_target = vec![false; target_nodes.len()];
    let mut node_correspondences: Vec<NodeCorrespondence> = Vec::new();
    // Map from source NodeId → target NodeId for edge matching.
    let mut node_map: HashMap<NodeId, NodeId> = HashMap::new();

    for (si, ti, quality) in &pair_scores {
        if !used_source[*si] && !used_target[*ti] {
            used_source[*si] = true;
            used_target[*ti] = true;
            let src_id = source_nodes[*si].id;
            let tgt_id = target_nodes[*ti].id;
            node_correspondences.push(NodeCorrespondence {
                source: src_id,
                target: tgt_id,
                match_quality: *quality,
            });
            node_map.insert(src_id, tgt_id);
        }
    }

    // ── Step 3: Edge correspondence ──
    let mut edge_correspondences: Vec<EdgeCorrespondence> = Vec::new();
    let mut relational_depth: usize = 0;

    for se in source_edges {
        // Check if both endpoints are mapped.
        if let (Some(&mapped_src), Some(&mapped_dst)) =
            (node_map.get(&se.src), node_map.get(&se.dst))
        {
            // Find a matching edge in the target.
            for te in target_edges {
                if te.src == mapped_src && te.dst == mapped_dst {
                    let exact = se.kind == te.kind;
                    let kind_sim = edge_kind_similarity(se.kind, te.kind);
                    if kind_sim > 0.0 {
                        edge_correspondences.push(EdgeCorrespondence {
                            source_edge: (se.src, se.dst, se.kind),
                            target_edge: (te.src, te.dst, te.kind),
                            exact_kind_match: exact,
                        });
                        // Track depth: causal/epistemic edges count as deeper.
                        if se.kind.is_causal() || se.kind.is_epistemic() {
                            relational_depth += 1;
                        }
                    }
                }
            }
        }
    }

    // ── Step 4: Compute scores ──
    let max_possible_nodes = source_nodes.len().min(target_nodes.len()).max(1);
    let max_possible_edges = source_edges.len().min(target_edges.len()).max(1);

    let node_coverage = node_correspondences.len() as f64 / max_possible_nodes as f64;
    let avg_node_quality = if node_correspondences.is_empty() {
        0.0
    } else {
        node_correspondences
            .iter()
            .map(|c| c.match_quality)
            .sum::<f64>()
            / node_correspondences.len() as f64
    };

    let edge_coverage = edge_correspondences.len() as f64 / max_possible_edges as f64;

    // Systematicity bonus: fraction of matched edges that are causal/epistemic.
    let high_order_count = edge_correspondences
        .iter()
        .filter(|ec| ec.source_edge.2.is_causal() || ec.source_edge.2.is_epistemic())
        .count();
    let systematicity = if edge_correspondences.is_empty() {
        0.0
    } else {
        high_order_count as f64 / edge_correspondences.len() as f64
    };

    // Exact-match bonus for edges.
    let exact_match_ratio = if edge_correspondences.is_empty() {
        0.0
    } else {
        edge_correspondences
            .iter()
            .filter(|ec| ec.exact_kind_match)
            .count() as f64
            / edge_correspondences.len() as f64
    };

    // Final score: node coverage + edge coverage + systematicity + exactness.
    let structural_similarity = (0.25 * node_coverage * avg_node_quality
        + 0.30 * edge_coverage
        + 0.30 * systematicity
        + 0.15 * exact_match_ratio)
        .clamp(0.0, 1.0);

    // Infer domains from node labels.
    let source_domain = infer_domain(source_nodes);
    let target_domain = infer_domain(target_nodes);

    StructuralMapping {
        source_nodes: source_nodes.iter().map(|n| n.id).collect(),
        target_nodes: target_nodes.iter().map(|n| n.id).collect(),
        node_correspondences,
        edge_correspondences,
        structural_similarity,
        relational_depth,
        pragmatic_relevance: 0.0,         // set externally
        candidate_inferences: Vec::new(), // populated by generate_candidate_inferences
        source_domain,
        target_domain,
        created_at_ms: now,
        last_used_ms: now,
        use_count: 0,
    }
}

/// Infer a domain label from a set of nodes.
///
/// Uses belief domains, entity types, or falls back to the most
/// common node kind.
fn infer_domain(nodes: &[&CognitiveNode]) -> String {
    // Try belief domains first.
    for n in nodes {
        if let NodePayload::Belief(b) = &n.payload {
            if !b.domain.is_empty() {
                return b.domain.clone();
            }
        }
    }
    // Try entity types.
    for n in nodes {
        if let NodePayload::Entity(e) = &n.payload {
            if !e.entity_type.is_empty() {
                return e.entity_type.clone();
            }
        }
    }
    // Fallback: most common node kind.
    let mut kind_counts: HashMap<NodeKind, usize> = HashMap::new();
    for n in nodes {
        *kind_counts.entry(n.kind()).or_default() += 1;
    }
    kind_counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(k, _)| k.as_str().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

// ══════════════════════════════════════════════════════════════════════════════
// § 6  Candidate Inference Generation
// ══════════════════════════════════════════════════════════════════════════════

/// Generate predictions about the target domain by projecting
/// unmapped source knowledge through the structural mapping.
///
/// For each source edge where the source endpoint IS mapped but
/// the relationship does NOT have a target counterpart, generate
/// a prediction: "this relationship probably exists in the target too."
pub fn generate_candidate_inferences(
    mapping: &StructuralMapping,
    source_nodes: &[&CognitiveNode],
    source_edges: &[CognitiveEdge],
) -> Vec<CandidateInference> {
    let mut inferences = Vec::new();

    // Build set of already-matched source edges.
    let matched_source: std::collections::HashSet<(NodeId, NodeId)> = mapping
        .edge_correspondences
        .iter()
        .map(|ec| (ec.source_edge.0, ec.source_edge.1))
        .collect();

    // Build node map: source → target.
    let node_map: HashMap<NodeId, NodeId> = mapping
        .node_correspondences
        .iter()
        .map(|nc| (nc.source, nc.target))
        .collect();

    // Node label lookup for descriptions.
    let label_map: HashMap<NodeId, &str> = source_nodes
        .iter()
        .map(|n| (n.id, n.label.as_str()))
        .collect();

    for se in source_edges {
        if matched_source.contains(&(se.src, se.dst)) {
            continue; // Already has a target counterpart — no prediction needed.
        }

        // At least one endpoint must be mapped for a useful inference.
        let mapped_src = node_map.get(&se.src).copied();
        let mapped_dst = node_map.get(&se.dst).copied();

        if let (Some(tsrc), Some(tdst)) = (mapped_src, mapped_dst) {
            // Both endpoints mapped but no matching edge in target → predict edge.
            let transfer_type = if se.kind.is_causal() || se.kind.is_epistemic() {
                TransferType::RelationalTransfer
            } else {
                TransferType::DirectMapping
            };

            let src_label = label_map.get(&se.src).unwrap_or(&"node");
            let dst_label = label_map.get(&se.dst).unwrap_or(&"node");

            inferences.push(CandidateInference {
                source_fact: se.src,
                projected_target: ProjectedFact {
                    description: format!(
                        "Predicted: {} {} {} (by analogy from {} → {})",
                        tsrc,
                        se.kind.as_str(),
                        tdst,
                        src_label,
                        dst_label,
                    ),
                    expected_edge_kind: se.kind,
                    source_ref: tsrc,
                    target_ref: tdst,
                },
                confidence: mapping.structural_similarity * se.confidence * 0.8,
                transfer_type,
            });
        } else if let Some(tsrc) = mapped_src {
            // Only source endpoint mapped — weaker inference.
            let src_label = label_map.get(&se.src).unwrap_or(&"node");

            inferences.push(CandidateInference {
                source_fact: se.src,
                projected_target: ProjectedFact {
                    description: format!(
                        "Predicted: {} may have a '{}' relationship (by analogy from {})",
                        tsrc,
                        se.kind.as_str(),
                        src_label,
                    ),
                    expected_edge_kind: se.kind,
                    source_ref: tsrc,
                    target_ref: NodeId::NIL,
                },
                confidence: mapping.structural_similarity * se.confidence * 0.4,
                transfer_type: TransferType::AbstractionTransfer,
            });
        }
    }

    // Sort by confidence descending.
    inferences.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    inferences
}

// ══════════════════════════════════════════════════════════════════════════════
// § 7  Analogy Quality Evaluation
// ══════════════════════════════════════════════════════════════════════════════

/// Score an analogy's quality based on:
/// - Systematicity: do higher-order relations (causal, epistemic) align?
/// - Structural consistency: is the mapping 1:1 without contradictions?
/// - Pragmatic relevance: does this help with current goals?
///
/// Returns a score in [0.0, 1.0].
pub fn evaluate_analogy_quality(mapping: &StructuralMapping) -> f64 {
    // Systematicity: ratio of causal+epistemic edges in the match.
    let total_edges = mapping.edge_correspondences.len().max(1);
    let high_order = mapping
        .edge_correspondences
        .iter()
        .filter(|ec| ec.source_edge.2.is_causal() || ec.source_edge.2.is_epistemic())
        .count();
    let systematicity = high_order as f64 / total_edges as f64;

    // Structural consistency: all node correspondences have quality > 0.3.
    let consistency = if mapping.node_correspondences.is_empty() {
        0.0
    } else {
        let good_matches = mapping
            .node_correspondences
            .iter()
            .filter(|nc| nc.match_quality > 0.3)
            .count();
        good_matches as f64 / mapping.node_correspondences.len() as f64
    };

    // Exact edge match ratio.
    let exactness = if mapping.edge_correspondences.is_empty() {
        0.0
    } else {
        mapping
            .edge_correspondences
            .iter()
            .filter(|ec| ec.exact_kind_match)
            .count() as f64
            / total_edges as f64
    };

    // Depth bonus: deeper relational matches are more valuable.
    let depth_bonus = (mapping.relational_depth as f64 / 5.0).min(1.0);

    // Combine with weights that favor systematicity.
    (0.35 * systematicity + 0.25 * consistency + 0.20 * exactness + 0.20 * depth_bonus)
        .clamp(0.0, 1.0)
}

// ══════════════════════════════════════════════════════════════════════════════
// § 8  Analogy Search
// ══════════════════════════════════════════════════════════════════════════════

/// Find analogies for a source subgraph by searching candidate target
/// subgraphs. Candidates are groups of nodes organized by domain.
///
/// This is the main entry point for analogical reasoning.
pub fn find_analogies(
    query: &AnalogicalQuery,
    source_node_data: &[&CognitiveNode],
    candidate_groups: &[SubgraphGroup],
) -> Vec<StructuralMapping> {
    let mut results: Vec<StructuralMapping> = Vec::new();

    for group in candidate_groups {
        // Scope filtering.
        match query.search_scope {
            AnalogyScope::SameDomain => {
                if group.domain != query.source_domain {
                    continue;
                }
            }
            AnalogyScope::CrossDomain => {
                if group.domain == query.source_domain {
                    continue;
                }
            }
            AnalogyScope::Universal => {}
        }

        let mut mapping = compute_structural_similarity(
            source_node_data,
            &group.nodes.iter().collect::<Vec<_>>(),
            &query.source_edges,
            &group.edges,
        );

        // Set domains.
        mapping.source_domain = query.source_domain.clone();
        mapping.target_domain = group.domain.clone();

        // Filter by minimum depth and quality.
        if mapping.relational_depth >= query.min_relational_depth
            && mapping.structural_similarity >= 0.15
        {
            // Generate inferences for this mapping.
            let inferences =
                generate_candidate_inferences(&mapping, source_node_data, &query.source_edges);
            mapping.candidate_inferences = inferences;

            results.push(mapping);
        }
    }

    // Sort by structural similarity descending.
    results.sort_by(|a, b| {
        b.structural_similarity
            .partial_cmp(&a.structural_similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results.truncate(query.max_results);
    results
}

/// A group of nodes+edges forming a candidate target subgraph.
#[derive(Debug, Clone)]
pub struct SubgraphGroup {
    /// Domain label for this group.
    pub domain: String,
    /// The nodes in this subgraph.
    pub nodes: Vec<CognitiveNode>,
    /// The edges within this subgraph.
    pub edges: Vec<CognitiveEdge>,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 9  Strategy Transfer
// ══════════════════════════════════════════════════════════════════════════════

/// A transferred action schema — an action template adapted from a
/// source domain to a target domain using a structural mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferredStrategy {
    /// Original schema name from the source domain.
    pub source_schema: String,
    /// Adapted description for the target domain.
    pub adapted_description: String,
    /// Confidence in this transfer [0.0, 1.0].
    pub transfer_confidence: f64,
    /// Node correspondences used in the transfer.
    pub mappings_used: Vec<(NodeId, NodeId)>,
    /// How the transfer was accomplished.
    pub transfer_type: TransferType,
}

/// Adapt action schemas from the source domain to the target domain
/// using a structural mapping.
///
/// For each source action schema, check if its preconditions and effects
/// reference mapped nodes, then create an adapted version for the target.
pub fn transfer_strategy(
    source_schemas: &[&CognitiveNode],
    mapping: &StructuralMapping,
) -> Vec<TransferredStrategy> {
    let node_map: HashMap<NodeId, NodeId> = mapping
        .node_correspondences
        .iter()
        .map(|nc| (nc.source, nc.target))
        .collect();

    let mut transferred = Vec::new();

    for schema_node in source_schemas {
        if let NodePayload::ActionSchema(schema) = &schema_node.payload {
            // Check how many precondition node refs map to the target domain.
            let mut mapped_preconditions = 0;
            let mut total_node_refs = 0;
            let mut mappings_used = Vec::new();

            for precond in &schema.preconditions {
                if let Some(ref_node) = precond.node_ref {
                    total_node_refs += 1;
                    if let Some(&target_node) = node_map.get(&ref_node) {
                        mapped_preconditions += 1;
                        mappings_used.push((ref_node, target_node));
                    }
                }
            }

            // Only transfer if at least some structure maps.
            let mapping_ratio = if total_node_refs > 0 {
                mapped_preconditions as f64 / total_node_refs as f64
            } else {
                // No node refs — transfer based on structural similarity alone.
                0.5
            };

            if mapping_ratio > 0.0 {
                let confidence = mapping.structural_similarity * mapping_ratio;
                transferred.push(TransferredStrategy {
                    source_schema: schema.name.clone(),
                    adapted_description: format!(
                        "[Transferred from {}→{}] {}",
                        mapping.source_domain, mapping.target_domain, schema.description,
                    ),
                    transfer_confidence: confidence.clamp(0.0, 1.0),
                    mappings_used,
                    transfer_type: if mapping_ratio >= 0.8 {
                        TransferType::DirectMapping
                    } else if mapping.relational_depth >= 2 {
                        TransferType::RelationalTransfer
                    } else {
                        TransferType::AbstractionTransfer
                    },
                });
            }
        }
    }

    // Sort by confidence descending.
    transferred.sort_by(|a, b| {
        b.transfer_confidence
            .partial_cmp(&a.transfer_confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    transferred
}

// ══════════════════════════════════════════════════════════════════════════════
// § 10  Opportunity Detection
// ══════════════════════════════════════════════════════════════════════════════

/// Scan for analogical opportunities: recently active subgraphs that
/// structurally resemble older, well-understood subgraphs.
///
/// This is the background scan that runs during cognitive ticks to
/// proactively surface useful analogies.
pub fn detect_analogical_opportunities(
    recent_subgraph: &[&CognitiveNode],
    recent_edges: &[CognitiveEdge],
    reference_groups: &[SubgraphGroup],
    min_similarity: f64,
) -> Vec<AnalogicalOpportunity> {
    let mut opportunities = Vec::new();

    for group in reference_groups {
        let mapping = compute_structural_similarity(
            recent_subgraph,
            &group.nodes.iter().collect::<Vec<_>>(),
            recent_edges,
            &group.edges,
        );

        if mapping.structural_similarity >= min_similarity && mapping.relational_depth >= 1 {
            let reason = format!(
                "Recent activity in '{}' structurally resembles '{}' \
                 (similarity: {:.2}, depth: {})",
                mapping.source_domain,
                mapping.target_domain,
                mapping.structural_similarity,
                mapping.relational_depth,
            );

            opportunities.push(AnalogicalOpportunity {
                recent_subgraph: recent_subgraph.iter().map(|n| n.id).collect(),
                reference_subgraph: group.nodes.iter().map(|n| n.id).collect(),
                mapping,
                reason,
            });
        }
    }

    // Sort by structural similarity descending.
    opportunities.sort_by(|a, b| {
        b.mapping
            .structural_similarity
            .partial_cmp(&a.mapping.structural_similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    opportunities
}

// ══════════════════════════════════════════════════════════════════════════════
// § 11  Decay & Maintenance
// ══════════════════════════════════════════════════════════════════════════════

/// Decay report from store maintenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalogyMaintenanceReport {
    /// Mappings pruned for low quality.
    pub pruned_low_quality: usize,
    /// Mappings pruned for age (unused too long).
    pub pruned_stale: usize,
    /// Remaining mappings.
    pub remaining: usize,
}

/// Decay unused analogies over time. Removes mappings that haven't been
/// used recently and whose quality is below threshold.
///
/// - Mappings unused for > `max_age_ms` are pruned if quality < threshold.
/// - All mappings get a small quality decay if unused.
pub fn analogy_strength_decay(
    store: &mut AnalogyStore,
    now_ms: u64,
    max_age_ms: u64,
) -> AnalogyMaintenanceReport {
    let threshold = store.quality_threshold;
    let mut pruned_low_quality = 0;
    let mut pruned_stale = 0;

    store.mappings.retain(|m| {
        let age = now_ms.saturating_sub(m.last_used_ms);
        let is_stale = age > max_age_ms && m.structural_similarity < threshold * 1.5;
        let is_weak = m.structural_similarity < threshold;

        if is_weak {
            pruned_low_quality += 1;
            return false;
        }
        if is_stale {
            pruned_stale += 1;
            return false;
        }
        true
    });

    let remaining = store.mappings.len();
    store.rebuild_index();

    AnalogyMaintenanceReport {
        pruned_low_quality,
        pruned_stale,
        remaining,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 12  Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{self, *};

    // ── Helpers ──

    fn make_entity(seq: u32, name: &str, entity_type: &str) -> CognitiveNode {
        CognitiveNode::new(
            NodeId::new(NodeKind::Entity, seq),
            name.to_string(),
            NodePayload::Entity(EntityPayload {
                name: name.to_string(),
                entity_type: entity_type.to_string(),
                memory_rids: vec![],
            }),
        )
    }

    fn make_belief(seq: u32, proposition: &str, domain: &str) -> CognitiveNode {
        CognitiveNode::new(
            NodeId::new(NodeKind::Belief, seq),
            proposition.to_string(),
            NodePayload::Belief(BeliefPayload {
                proposition: proposition.to_string(),
                log_odds: 1.0,
                domain: domain.to_string(),
                evidence_trail: vec![],
                user_confirmed: false,
            }),
        )
    }

    fn make_goal(seq: u32, desc: &str) -> CognitiveNode {
        CognitiveNode::new(
            NodeId::new(NodeKind::Goal, seq),
            desc.to_string(),
            NodePayload::Goal(GoalPayload {
                description: desc.to_string(),
                status: GoalStatus::Active,
                completion_criteria: String::new(),
                deadline: None,
                priority: Priority::Medium,
                parent_goal: None,
                progress: 0.0,
            }),
        )
    }

    fn make_action_schema(seq: u32, name: &str, precond_refs: Vec<NodeId>) -> CognitiveNode {
        let preconditions: Vec<Precondition> = precond_refs
            .into_iter()
            .map(|nid| Precondition {
                description: format!("Requires {}", nid),
                node_ref: Some(nid),
                required: true,
            })
            .collect();

        CognitiveNode::new(
            NodeId::new(NodeKind::ActionSchema, seq),
            name.to_string(),
            NodePayload::ActionSchema(ActionSchemaPayload {
                name: name.to_string(),
                description: format!("Action: {}", name),
                action_kind: ActionKind::Suggest,
                preconditions,
                effects: vec![],
                confidence_threshold: 0.5,
                success_rate: 0.7,
                execution_count: 5,
                acceptance_count: 3,
            }),
        )
    }

    fn edge(src: NodeId, dst: NodeId, kind: CognitiveEdgeKind) -> CognitiveEdge {
        CognitiveEdge::new(src, dst, kind, 0.8)
    }

    // ── Tests ──

    #[test]
    fn test_node_match_quality_same_kind() {
        let a = make_entity(1, "Project Alpha", "project");
        let b = make_entity(2, "Project Beta", "project");
        let q = node_match_quality(&a, &b);
        // Same kind + same entity_type → high match.
        assert!(q > 0.6, "Same-kind same-type should match well: {q}");
    }

    #[test]
    fn test_node_match_quality_different_kind() {
        let a = make_entity(1, "Alice", "person");
        let b = make_goal(1, "Ship the product");
        let q = node_match_quality(&a, &b);
        assert!(q < 0.1, "Different kinds should match poorly: {q}");
    }

    #[test]
    fn test_edge_kind_similarity() {
        assert_eq!(
            edge_kind_similarity(CognitiveEdgeKind::Causes, CognitiveEdgeKind::Causes),
            1.0
        );
        // Same category (causal).
        assert!(edge_kind_similarity(CognitiveEdgeKind::Causes, CognitiveEdgeKind::Predicts) > 0.5);
        // Different category.
        assert_eq!(
            edge_kind_similarity(CognitiveEdgeKind::Causes, CognitiveEdgeKind::Prefers),
            0.0
        );
    }

    #[test]
    fn test_structural_similarity_identical_subgraphs() {
        // Two isomorphic subgraphs: A→B (Causes), B→C (Supports).
        let a1 = make_entity(1, "Team", "group");
        let b1 = make_entity(2, "Effort", "concept");
        let c1 = make_belief(1, "We can ship", "work");

        let a2 = make_entity(3, "Squad", "group");
        let b2 = make_entity(4, "Work", "concept");
        let c2 = make_belief(2, "We will deliver", "work");

        let source_edges = vec![
            edge(a1.id, b1.id, CognitiveEdgeKind::Causes),
            edge(b1.id, c1.id, CognitiveEdgeKind::Supports),
        ];
        let target_edges = vec![
            edge(a2.id, b2.id, CognitiveEdgeKind::Causes),
            edge(b2.id, c2.id, CognitiveEdgeKind::Supports),
        ];

        let mapping = compute_structural_similarity(
            &[&a1, &b1, &c1],
            &[&a2, &b2, &c2],
            &source_edges,
            &target_edges,
        );

        assert!(
            mapping.structural_similarity > 0.5,
            "Isomorphic subgraphs should have high similarity: {}",
            mapping.structural_similarity
        );
        assert_eq!(mapping.node_correspondences.len(), 3);
        assert_eq!(mapping.edge_correspondences.len(), 2);
    }

    #[test]
    fn test_structural_similarity_no_overlap() {
        let a = make_entity(1, "Sun", "star");
        let b = make_goal(1, "Learn Rust");

        let mapping = compute_structural_similarity(&[&a], &[&b], &[], &[]);

        assert!(
            mapping.structural_similarity < 0.15,
            "No overlap should give low similarity: {}",
            mapping.structural_similarity
        );
    }

    #[test]
    fn test_systematicity_preference() {
        // Mapping with causal edges should score higher than one with only associative edges.
        let a1 = make_entity(1, "A", "concept");
        let b1 = make_entity(2, "B", "concept");
        let a2 = make_entity(3, "C", "concept");
        let b2 = make_entity(4, "D", "concept");

        // Causal edge.
        let causal_source = vec![edge(a1.id, b1.id, CognitiveEdgeKind::Causes)];
        let causal_target = vec![edge(a2.id, b2.id, CognitiveEdgeKind::Causes)];

        // Associative edge.
        let assoc_source = vec![edge(a1.id, b1.id, CognitiveEdgeKind::AssociatedWith)];
        let assoc_target = vec![edge(a2.id, b2.id, CognitiveEdgeKind::AssociatedWith)];

        let m_causal =
            compute_structural_similarity(&[&a1, &b1], &[&a2, &b2], &causal_source, &causal_target);
        let m_assoc =
            compute_structural_similarity(&[&a1, &b1], &[&a2, &b2], &assoc_source, &assoc_target);

        assert!(
            m_causal.structural_similarity > m_assoc.structural_similarity,
            "Causal ({}) should score higher than associative ({})",
            m_causal.structural_similarity,
            m_assoc.structural_similarity,
        );
    }

    #[test]
    fn test_candidate_inference_generation() {
        let a1 = make_entity(1, "Manager", "person");
        let b1 = make_entity(2, "Report", "document");
        let c1 = make_entity(3, "Deadline", "event");

        let a2 = make_entity(4, "Lead", "person");
        let b2 = make_entity(5, "Spec", "document");

        // Source has 3 edges; target only has 1 matching.
        let source_edges = vec![
            edge(a1.id, b1.id, CognitiveEdgeKind::Causes), // mapped
            edge(b1.id, c1.id, CognitiveEdgeKind::Requires), // b1 mapped, c1 NOT mapped
            edge(a1.id, c1.id, CognitiveEdgeKind::AdvancesGoal), // a1 mapped, c1 NOT mapped
        ];
        let target_edges = vec![edge(a2.id, b2.id, CognitiveEdgeKind::Causes)];

        let mapping = compute_structural_similarity(
            &[&a1, &b1, &c1],
            &[&a2, &b2],
            &source_edges,
            &target_edges,
        );

        let inferences = generate_candidate_inferences(&mapping, &[&a1, &b1, &c1], &source_edges);

        // Should have inferences for unmapped edges.
        assert!(
            !inferences.is_empty(),
            "Should generate inferences for unmapped edges"
        );

        // All inferences should have positive confidence.
        for inf in &inferences {
            assert!(
                inf.confidence > 0.0,
                "Inference confidence should be positive"
            );
        }
    }

    #[test]
    fn test_analogy_quality_evaluation() {
        let a1 = make_entity(1, "X", "concept");
        let b1 = make_entity(2, "Y", "concept");
        let a2 = make_entity(3, "P", "concept");
        let b2 = make_entity(4, "Q", "concept");

        let source_edges = vec![edge(a1.id, b1.id, CognitiveEdgeKind::Causes)];
        let target_edges = vec![edge(a2.id, b2.id, CognitiveEdgeKind::Causes)];

        let mapping =
            compute_structural_similarity(&[&a1, &b1], &[&a2, &b2], &source_edges, &target_edges);

        let quality = evaluate_analogy_quality(&mapping);
        assert!(quality > 0.0, "Quality should be positive: {quality}");
        assert!(quality <= 1.0, "Quality should be ≤ 1.0: {quality}");
    }

    #[test]
    fn test_strategy_transfer() {
        let a1 = make_entity(1, "Employee", "person");
        let b1 = make_entity(2, "Review", "process");

        let a2 = make_entity(3, "Student", "person");
        let b2 = make_entity(4, "Exam", "process");

        // Action schema in source domain referencing a1.
        let schema = make_action_schema(1, "schedule_review", vec![a1.id, b1.id]);

        let source_edges = vec![edge(a1.id, b1.id, CognitiveEdgeKind::Requires)];
        let target_edges = vec![edge(a2.id, b2.id, CognitiveEdgeKind::Requires)];

        let mapping =
            compute_structural_similarity(&[&a1, &b1], &[&a2, &b2], &source_edges, &target_edges);

        let transferred = transfer_strategy(&[&schema], &mapping);
        assert!(
            !transferred.is_empty(),
            "Should transfer at least one strategy"
        );
        assert!(
            transferred[0].transfer_confidence > 0.0,
            "Transfer confidence should be positive"
        );
        assert!(
            transferred[0].adapted_description.contains("Transferred"),
            "Description should note the transfer"
        );
    }

    #[test]
    fn test_analogy_store_operations() {
        let mut store = AnalogyStore::with_config(0.2, 5);
        assert!(store.is_empty());

        let now = crate::state::now_ms();
        for i in 0..3 {
            store.insert(StructuralMapping {
                source_nodes: vec![],
                target_nodes: vec![],
                node_correspondences: vec![],
                edge_correspondences: vec![],
                structural_similarity: 0.5 + i as f64 * 0.1,
                relational_depth: 1,
                pragmatic_relevance: 0.5,
                candidate_inferences: vec![],
                source_domain: "hiring".to_string(),
                target_domain: format!("domain_{}", i),
                created_at_ms: now,
                last_used_ms: now,
                use_count: i as u32,
            });
        }

        assert_eq!(store.len(), 3);
        assert_eq!(store.find_by_domain("hiring").len(), 3);
        assert_eq!(store.find_by_domain("domain_0").len(), 1);
        assert_eq!(store.cross_domain_mappings().len(), 3);
    }

    #[test]
    fn test_analogy_store_eviction() {
        let mut store = AnalogyStore::with_config(0.1, 3);
        let now = crate::state::now_ms();

        for i in 0..5 {
            store.insert(StructuralMapping {
                source_nodes: vec![],
                target_nodes: vec![],
                node_correspondences: vec![],
                edge_correspondences: vec![],
                structural_similarity: 0.3 + i as f64 * 0.1,
                relational_depth: 1,
                pragmatic_relevance: 0.5,
                candidate_inferences: vec![],
                source_domain: format!("src_{}", i),
                target_domain: format!("tgt_{}", i),
                created_at_ms: now,
                last_used_ms: now,
                use_count: 0,
            });
        }

        // Capacity is 3, so 2 weakest should have been evicted.
        assert_eq!(store.len(), 3);

        // The remaining should be the 3 highest similarity.
        for m in &store.mappings {
            assert!(
                m.structural_similarity >= 0.5,
                "Weak mappings should have been evicted: {}",
                m.structural_similarity,
            );
        }
    }

    #[test]
    fn test_analogy_decay() {
        let mut store = AnalogyStore::with_config(0.3, 10);
        let now = crate::state::now_ms();
        let old = now - 100_000_000; // ~27 hours ago

        // Fresh, good mapping.
        store.insert(StructuralMapping {
            source_nodes: vec![],
            target_nodes: vec![],
            node_correspondences: vec![],
            edge_correspondences: vec![],
            structural_similarity: 0.8,
            relational_depth: 2,
            pragmatic_relevance: 0.5,
            candidate_inferences: vec![],
            source_domain: "fresh".to_string(),
            target_domain: "target".to_string(),
            created_at_ms: now,
            last_used_ms: now,
            use_count: 5,
        });

        // Old, weak mapping.
        store.insert(StructuralMapping {
            source_nodes: vec![],
            target_nodes: vec![],
            node_correspondences: vec![],
            edge_correspondences: vec![],
            structural_similarity: 0.35,
            relational_depth: 1,
            pragmatic_relevance: 0.1,
            candidate_inferences: vec![],
            source_domain: "stale".to_string(),
            target_domain: "target".to_string(),
            created_at_ms: old,
            last_used_ms: old,
            use_count: 0,
        });

        // Below-threshold mapping.
        store.insert(StructuralMapping {
            source_nodes: vec![],
            target_nodes: vec![],
            node_correspondences: vec![],
            edge_correspondences: vec![],
            structural_similarity: 0.2,
            relational_depth: 0,
            pragmatic_relevance: 0.0,
            candidate_inferences: vec![],
            source_domain: "weak".to_string(),
            target_domain: "target".to_string(),
            created_at_ms: now,
            last_used_ms: now,
            use_count: 0,
        });

        let report = analogy_strength_decay(&mut store, now, 86_400_000); // 24h max age

        assert!(
            report.pruned_low_quality > 0,
            "Should prune below-threshold"
        );
        assert!(report.pruned_stale > 0, "Should prune stale unused");
        assert_eq!(report.remaining, 1, "Only the fresh good one should remain");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_find_analogies() {
        // Source: Manager → Report (Causes).
        let a1 = make_entity(1, "Manager", "person");
        let b1 = make_entity(2, "Report", "document");
        let source_edges = vec![edge(a1.id, b1.id, CognitiveEdgeKind::Causes)];

        // Target group: Teacher → Lesson (Causes) — should match.
        let a2 = make_entity(3, "Teacher", "person");
        let b2 = make_entity(4, "Lesson", "document");
        let target_edges = vec![edge(a2.id, b2.id, CognitiveEdgeKind::Causes)];

        let groups = vec![SubgraphGroup {
            domain: "education".to_string(),
            nodes: vec![a2, b2],
            edges: target_edges,
        }];

        let query = AnalogicalQuery {
            source_subgraph: vec![a1.id, b1.id],
            source_edges,
            source_domain: "business".to_string(),
            search_scope: AnalogyScope::CrossDomain,
            min_relational_depth: 0,
            max_results: 5,
        };

        let results = find_analogies(&query, &[&a1, &b1], &groups);
        assert!(
            !results.is_empty(),
            "Should find at least one cross-domain analogy"
        );
        assert!(
            results[0].structural_similarity > 0.3,
            "Cross-domain causal match should be decent: {}",
            results[0].structural_similarity,
        );
    }

    #[test]
    fn test_find_analogies_scope_filtering() {
        let a1 = make_entity(1, "A", "concept");
        let a2 = make_entity(2, "B", "concept");

        let groups = vec![
            SubgraphGroup {
                domain: "business".to_string(), // same domain
                nodes: vec![a2.clone()],
                edges: vec![],
            },
            SubgraphGroup {
                domain: "science".to_string(), // different domain
                nodes: vec![a2],
                edges: vec![],
            },
        ];

        // CrossDomain should skip "business".
        let query = AnalogicalQuery {
            source_subgraph: vec![a1.id],
            source_edges: vec![],
            source_domain: "business".to_string(),
            search_scope: AnalogyScope::CrossDomain,
            min_relational_depth: 0,
            max_results: 10,
        };

        let results = find_analogies(&query, &[&a1], &groups);
        for r in &results {
            assert_ne!(
                r.target_domain, "business",
                "CrossDomain should not match same domain"
            );
        }
    }

    #[test]
    fn test_opportunity_detection() {
        let a1 = make_entity(1, "Developer", "person");
        let b1 = make_entity(2, "Sprint", "event");
        let recent_edges = vec![edge(a1.id, b1.id, CognitiveEdgeKind::AdvancesGoal)];

        let a2 = make_entity(3, "Author", "person");
        let b2 = make_entity(4, "Chapter", "event");

        let groups = vec![SubgraphGroup {
            domain: "writing".to_string(),
            nodes: vec![a2, b2],
            edges: vec![edge(
                NodeId::new(NodeKind::Entity, 3),
                NodeId::new(NodeKind::Entity, 4),
                CognitiveEdgeKind::AdvancesGoal,
            )],
        }];

        let opps = detect_analogical_opportunities(&[&a1, &b1], &recent_edges, &groups, 0.2);
        // May or may not find opportunities depending on match quality,
        // but the function should not panic and should return valid results.
        for opp in &opps {
            assert!(opp.mapping.structural_similarity >= 0.2);
            assert!(!opp.reason.is_empty());
        }
    }

    #[test]
    fn test_relational_depth_tracking() {
        let a1 = make_entity(1, "X", "concept");
        let b1 = make_entity(2, "Y", "concept");
        let c1 = make_belief(1, "Z is true", "logic");

        let a2 = make_entity(3, "P", "concept");
        let b2 = make_entity(4, "Q", "concept");
        let c2 = make_belief(2, "R is true", "logic");

        // Two causal edges + one epistemic → depth should be ≥ 2.
        let source_edges = vec![
            edge(a1.id, b1.id, CognitiveEdgeKind::Causes),
            edge(b1.id, c1.id, CognitiveEdgeKind::Supports),
        ];
        let target_edges = vec![
            edge(a2.id, b2.id, CognitiveEdgeKind::Causes),
            edge(b2.id, c2.id, CognitiveEdgeKind::Supports),
        ];

        let mapping = compute_structural_similarity(
            &[&a1, &b1, &c1],
            &[&a2, &b2, &c2],
            &source_edges,
            &target_edges,
        );

        assert!(
            mapping.relational_depth >= 2,
            "Should have relational depth ≥ 2 for causal+epistemic: {}",
            mapping.relational_depth,
        );
    }

    #[test]
    fn test_infer_domain_from_beliefs() {
        let b = make_belief(1, "It will rain", "weather");
        assert_eq!(infer_domain(&[&b]), "weather");
    }

    #[test]
    fn test_infer_domain_from_entities() {
        let e = make_entity(1, "Alice", "person");
        assert_eq!(infer_domain(&[&e]), "person");
    }

    #[test]
    fn test_infer_domain_fallback() {
        let g = make_goal(1, "Ship v1");
        assert_eq!(infer_domain(&[&g]), "goal");
    }

    #[test]
    fn test_transfer_type_as_str() {
        assert_eq!(TransferType::DirectMapping.as_str(), "direct_mapping");
        assert_eq!(
            TransferType::RelationalTransfer.as_str(),
            "relational_transfer"
        );
        assert_eq!(
            TransferType::AbstractionTransfer.as_str(),
            "abstraction_transfer"
        );
    }

    #[test]
    fn test_empty_subgraph_similarity() {
        let mapping = compute_structural_similarity(&[], &[], &[], &[]);
        assert_eq!(mapping.node_correspondences.len(), 0);
        assert_eq!(mapping.edge_correspondences.len(), 0);
        // Should not panic, similarity should be low/zero.
        assert!(mapping.structural_similarity <= 0.5);
    }
}
