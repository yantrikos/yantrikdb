//! Working Set & Spreading Activation
//!
//! The WorkingSet is a bounded in-memory projection of the persistent cognitive
//! state graph. It holds the ~32 most relevant nodes at any moment, manages
//! activation spreading through typed edges, and handles eviction when the
//! set is full.
//!
//! ## Design Principles
//!
//! - **Bounded**: Hard cap on active nodes (default 32, configurable). This isn't
//!   a limitation — it's a feature. Human working memory holds ~7±2 items.
//!   Our system operates with ~32 for richer reasoning while staying fast.
//!
//! - **Non-neural**: Spreading activation uses typed edge weights, NOT gradient
//!   descent. It's interpretable: you can trace exactly why a node is active.
//!
//! - **SIMD-friendly**: Activation vectors are contiguous f64 arrays for cache
//!   locality. Top-K selection uses partial sort, not full sort.
//!
//! - **Inhibitory**: Contradicts/Prevents/BlocksGoal edges suppress activation.
//!   This prevents contradictory beliefs from being simultaneously active.
//!
//! - **Anytime**: Three execution modes (Fast <3ms, Balanced <10ms, Deep <30ms)
//!   controlled by hop depth and top-K truncation per hop.
//!
//! ## How Spreading Activation Works
//!
//! 1. **Seed**: External stimulus activates one or more nodes (e.g., user mentions
//!    an entity, a routine fires, a deadline approaches).
//!
//! 2. **Spread**: For each active node, transfer activation to neighbors through
//!    typed edges. Transfer amount = `source_activation * edge_weight * edge_confidence
//!    * edge_kind.activation_transfer()`. Inhibitory edges subtract.
//!
//! 3. **Decay**: All activations decay by a factor per hop (default 0.7).
//!    This ensures activation attenuates with graph distance.
//!
//! 4. **Truncate**: After each hop, only the top-K most activated nodes survive.
//!    This prevents activation explosion in dense graphs.
//!
//! 5. **Evict**: If the working set exceeds capacity, evict nodes with the
//!    lowest relevance_score() (combines activation, salience, persistence,
//!    urgency, recency).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use super::state::*;

// ── Configuration ──

/// Controls spreading activation behavior.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Maximum nodes in the working set.
    pub capacity: usize,
    /// Maximum hops for spreading activation.
    pub max_hops: u32,
    /// Top-K nodes retained per hop (prevents explosion in dense graphs).
    pub top_k_per_hop: usize,
    /// Decay factor per hop [0.0, 1.0]. Lower = faster attenuation.
    pub hop_decay: f64,
    /// Minimum activation to propagate (below this, stop spreading).
    pub activation_threshold: f64,
    /// Global inhibition factor: when an inhibitory edge fires, it also
    /// reduces activation of all nodes that support the inhibited node.
    pub lateral_inhibition: f64,
    /// Activation boost given to newly inserted nodes.
    pub insertion_boost: f64,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self::balanced()
    }
}

impl AttentionConfig {
    /// Fast mode: 1 hop, small top-K. Target: <3ms.
    pub fn fast() -> Self {
        Self {
            capacity: 32,
            max_hops: 1,
            top_k_per_hop: 8,
            hop_decay: 0.6,
            activation_threshold: 0.05,
            lateral_inhibition: 0.15,
            insertion_boost: 0.5,
        }
    }

    /// Balanced mode: 2 hops, moderate top-K. Target: <10ms.
    pub fn balanced() -> Self {
        Self {
            capacity: 32,
            max_hops: 2,
            top_k_per_hop: 12,
            hop_decay: 0.7,
            activation_threshold: 0.03,
            lateral_inhibition: 0.20,
            insertion_boost: 0.5,
        }
    }

    /// Deep mode: 3 hops, large top-K. Target: <30ms.
    pub fn deep() -> Self {
        Self {
            capacity: 48,
            max_hops: 3,
            top_k_per_hop: 16,
            hop_decay: 0.75,
            activation_threshold: 0.02,
            lateral_inhibition: 0.25,
            insertion_boost: 0.5,
        }
    }
}

// ── Working Set ──

/// The bounded in-memory cognitive working set.
///
/// Holds up to `capacity` cognitive nodes and their interconnecting edges.
/// Provides spreading activation, decay, and eviction.
pub struct WorkingSet {
    /// Active cognitive nodes, keyed by NodeId.
    nodes: HashMap<u32, CognitiveNode>,
    /// Edges between nodes in the working set.
    /// Stored as adjacency list: src_raw → Vec<(dst_raw, edge)>.
    adjacency: HashMap<u32, Vec<(u32, CognitiveEdge)>>,
    /// Reverse adjacency for inhibition lookups: dst_raw → Vec<src_raw>.
    reverse_adj: HashMap<u32, Vec<u32>>,
    /// Configuration for attention behavior.
    config: AttentionConfig,
    /// Node ID allocator for creating new nodes.
    allocator: NodeIdAllocator,
    /// Nodes evicted in the last operation (for persistence).
    last_evicted: Vec<CognitiveNode>,
    /// Activation deltas from the last spread (for debugging/tracing).
    last_spread_deltas: HashMap<u32, f64>,
    /// Total number of spread operations performed.
    spread_count: u64,
}

impl WorkingSet {
    /// Create a new empty working set with default config.
    pub fn new() -> Self {
        Self::with_config(AttentionConfig::default())
    }

    /// Create with specific configuration.
    pub fn with_config(config: AttentionConfig) -> Self {
        Self {
            nodes: HashMap::with_capacity(config.capacity),
            adjacency: HashMap::with_capacity(config.capacity * 2),
            reverse_adj: HashMap::with_capacity(config.capacity * 2),
            config,
            allocator: NodeIdAllocator::new(),
            last_evicted: Vec::new(),
            last_spread_deltas: HashMap::new(),
            spread_count: 0,
        }
    }

    /// Create with a pre-initialized allocator (for persistence restore).
    pub fn with_allocator(config: AttentionConfig, allocator: NodeIdAllocator) -> Self {
        Self {
            nodes: HashMap::with_capacity(config.capacity),
            adjacency: HashMap::with_capacity(config.capacity * 2),
            reverse_adj: HashMap::with_capacity(config.capacity * 2),
            config,
            allocator,
            last_evicted: Vec::new(),
            last_spread_deltas: HashMap::new(),
            spread_count: 0,
        }
    }

    // ── Accessors ──

    /// Number of nodes currently in the working set.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the working set is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Whether the working set is at capacity.
    pub fn is_full(&self) -> bool {
        self.nodes.len() >= self.config.capacity
    }

    /// Get a node by its ID.
    pub fn get(&self, id: NodeId) -> Option<&CognitiveNode> {
        self.nodes.get(&id.to_raw())
    }

    /// Get a mutable reference to a node.
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut CognitiveNode> {
        self.nodes.get_mut(&id.to_raw())
    }

    /// Get the configuration.
    pub fn config(&self) -> &AttentionConfig {
        &self.config
    }

    /// Get a mutable reference to the allocator.
    pub fn allocator_mut(&mut self) -> &mut NodeIdAllocator {
        &mut self.allocator
    }

    /// Get the allocator.
    pub fn allocator(&self) -> &NodeIdAllocator {
        &self.allocator
    }

    /// Get nodes evicted during the last insertion/spread.
    pub fn last_evicted(&self) -> &[CognitiveNode] {
        &self.last_evicted
    }

    /// Get activation deltas from the last spread (node_raw → delta).
    pub fn last_spread_deltas(&self) -> &HashMap<u32, f64> {
        &self.last_spread_deltas
    }

    /// Total spread operations performed.
    pub fn spread_count(&self) -> u64 {
        self.spread_count
    }

    /// Iterate over all nodes.
    pub fn iter(&self) -> impl Iterator<Item = &CognitiveNode> {
        self.nodes.values()
    }

    /// Iterate over all nodes mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut CognitiveNode> {
        self.nodes.values_mut()
    }

    /// Get all edges from a source node.
    pub fn edges_from(&self, id: NodeId) -> &[(u32, CognitiveEdge)] {
        self.adjacency
            .get(&id.to_raw())
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all nodes sorted by activation (descending).
    pub fn by_activation(&self) -> Vec<&CognitiveNode> {
        let mut nodes: Vec<_> = self.nodes.values().collect();
        nodes.sort_by(|a, b| {
            b.attrs
                .activation
                .partial_cmp(&a.attrs.activation)
                .unwrap_or(Ordering::Equal)
        });
        nodes
    }

    /// Get all nodes sorted by relevance_score (descending).
    pub fn by_relevance(&self) -> Vec<&CognitiveNode> {
        let mut nodes: Vec<_> = self.nodes.values().collect();
        nodes.sort_by(|a, b| {
            b.attrs
                .relevance_score()
                .partial_cmp(&a.attrs.relevance_score())
                .unwrap_or(Ordering::Equal)
        });
        nodes
    }

    /// Get the top-K most activated nodes.
    pub fn top_k(&self, k: usize) -> Vec<&CognitiveNode> {
        if k >= self.nodes.len() {
            return self.by_activation();
        }

        // Use a min-heap of size K for O(n log k) selection
        let mut heap: BinaryHeap<ActivationEntry> = BinaryHeap::new();
        for node in self.nodes.values() {
            let entry = ActivationEntry {
                raw_id: node.id.to_raw(),
                activation: node.attrs.activation,
            };
            if heap.len() < k {
                heap.push(entry);
            } else if let Some(min) = heap.peek() {
                if entry.activation > min.activation {
                    heap.pop();
                    heap.push(entry);
                }
            }
        }

        // Collect IDs and look up nodes
        let ids: Vec<u32> = heap.into_iter().map(|e| e.raw_id).collect();
        let mut result: Vec<&CognitiveNode> =
            ids.iter().filter_map(|id| self.nodes.get(id)).collect();
        result.sort_by(|a, b| {
            b.attrs
                .activation
                .partial_cmp(&a.attrs.activation)
                .unwrap_or(Ordering::Equal)
        });
        result
    }

    /// Get all nodes of a specific kind.
    pub fn nodes_of_kind(&self, kind: NodeKind) -> Vec<&CognitiveNode> {
        self.nodes.values().filter(|n| n.kind() == kind).collect()
    }

    // ── Mutation ──

    /// Insert a node into the working set.
    /// If the set is full, evicts the least relevant node first.
    /// Returns the evicted node (if any).
    pub fn insert(&mut self, mut node: CognitiveNode) -> Option<CognitiveNode> {
        self.last_evicted.clear();

        // Apply insertion boost
        node.attrs.touch(self.config.insertion_boost);

        let evicted = if self.is_full() {
            let victim = self.find_eviction_candidate(&node);
            victim.and_then(|id| self.remove(id))
        } else {
            None
        };

        let raw = node.id.to_raw();
        self.nodes.insert(raw, node);

        if let Some(ref e) = evicted {
            self.last_evicted.push(e.clone());
        }

        evicted
    }

    /// Insert a node and its edges together.
    pub fn insert_with_edges(
        &mut self,
        node: CognitiveNode,
        edges: Vec<CognitiveEdge>,
    ) -> Option<CognitiveNode> {
        let evicted = self.insert(node);
        for edge in edges {
            self.add_edge(edge);
        }
        evicted
    }

    /// Remove a node and all its edges from the working set.
    /// Returns the removed node.
    pub fn remove(&mut self, id: NodeId) -> Option<CognitiveNode> {
        let raw = id.to_raw();
        let node = self.nodes.remove(&raw)?;

        // Remove outgoing edges
        self.adjacency.remove(&raw);

        // Remove from reverse adjacency
        self.reverse_adj.remove(&raw);

        // Remove incoming edges (from other nodes' adjacency lists)
        for adj in self.adjacency.values_mut() {
            adj.retain(|(dst, _)| *dst != raw);
        }

        // Remove from reverse adjacency of other nodes
        for rev in self.reverse_adj.values_mut() {
            rev.retain(|src| *src != raw);
        }

        Some(node)
    }

    /// Add an edge between nodes in the working set.
    /// Both endpoints must exist in the set.
    pub fn add_edge(&mut self, edge: CognitiveEdge) -> bool {
        let src_raw = edge.src.to_raw();
        let dst_raw = edge.dst.to_raw();

        if !self.nodes.contains_key(&src_raw) || !self.nodes.contains_key(&dst_raw) {
            return false;
        }

        // Forward adjacency
        self.adjacency
            .entry(src_raw)
            .or_insert_with(Vec::new)
            .push((dst_raw, edge));

        // Reverse adjacency
        self.reverse_adj
            .entry(dst_raw)
            .or_insert_with(Vec::new)
            .push(src_raw);

        true
    }

    /// Remove all edges between two specific nodes.
    pub fn remove_edges_between(&mut self, src: NodeId, dst: NodeId) {
        let src_raw = src.to_raw();
        let dst_raw = dst.to_raw();

        if let Some(adj) = self.adjacency.get_mut(&src_raw) {
            adj.retain(|(d, _)| *d != dst_raw);
        }
        if let Some(rev) = self.reverse_adj.get_mut(&dst_raw) {
            rev.retain(|s| *s != src_raw);
        }
    }

    /// Clear all nodes and edges.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.adjacency.clear();
        self.reverse_adj.clear();
        self.last_evicted.clear();
        self.last_spread_deltas.clear();
    }

    // ── Spreading Activation ──

    /// Seed activation on a node and spread through the graph.
    ///
    /// This is the core attention mechanism. It:
    /// 1. Sets activation on the seed node
    /// 2. Spreads activation through typed edges for `max_hops`
    /// 3. Applies inhibition from contradicts/prevents/blocks edges
    /// 4. Truncates to top-K per hop
    /// 5. Applies lateral inhibition
    ///
    /// Returns the number of nodes whose activation changed.
    pub fn activate_and_spread(&mut self, seed_id: NodeId, seed_activation: f64) -> usize {
        self.last_spread_deltas.clear();
        self.spread_count += 1;

        // Set seed activation
        if let Some(node) = self.nodes.get_mut(&seed_id.to_raw()) {
            let old = node.attrs.activation;
            node.attrs.activation = (node.attrs.activation + seed_activation).min(1.0);
            node.attrs.last_updated_ms = now_ms();
            self.last_spread_deltas
                .insert(seed_id.to_raw(), node.attrs.activation - old);
        } else {
            return 0;
        }

        // Collect current frontier (nodes to spread from)
        let mut frontier: Vec<(u32, f64)> = vec![(seed_id.to_raw(), seed_activation)];
        let mut changed_count = 1usize;

        for hop in 0..self.config.max_hops {
            let decay = self.config.hop_decay.powi(hop as i32 + 1);
            let mut next_frontier: Vec<(u32, f64)> = Vec::new();
            let mut inhibition_targets: Vec<(u32, f64)> = Vec::new();

            for &(src_raw, src_activation) in &frontier {
                let edges = match self.adjacency.get(&src_raw) {
                    Some(e) => e.clone(), // Clone to avoid borrow conflict
                    None => continue,
                };

                for (dst_raw, edge) in &edges {
                    let transfer = edge.effective_activation_transfer();
                    let delta = src_activation * transfer * decay;

                    if delta.abs() < self.config.activation_threshold {
                        continue;
                    }

                    if edge.kind.is_inhibitory() {
                        // Inhibitory: reduce target activation
                        inhibition_targets.push((*dst_raw, delta)); // delta is already negative
                    } else {
                        // Excitatory: boost target activation
                        next_frontier.push((*dst_raw, delta));
                    }
                }
            }

            // Apply excitatory deltas
            for (dst_raw, delta) in &next_frontier {
                if let Some(node) = self.nodes.get_mut(dst_raw) {
                    let old = node.attrs.activation;
                    node.attrs.activation = (node.attrs.activation + delta).clamp(0.0, 1.0);
                    node.attrs.last_updated_ms = now_ms();
                    let actual_delta = node.attrs.activation - old;
                    if actual_delta.abs() > 1e-6 {
                        *self.last_spread_deltas.entry(*dst_raw).or_insert(0.0) += actual_delta;
                        changed_count += 1;
                    }
                }
            }

            // Apply inhibitory deltas
            for (dst_raw, delta) in &inhibition_targets {
                if let Some(node) = self.nodes.get_mut(dst_raw) {
                    let old = node.attrs.activation;
                    // delta is negative (from is_inhibitory edges)
                    node.attrs.activation = (node.attrs.activation + delta).clamp(0.0, 1.0);
                    let actual_delta = node.attrs.activation - old;
                    if actual_delta.abs() > 1e-6 {
                        *self.last_spread_deltas.entry(*dst_raw).or_insert(0.0) += actual_delta;
                        changed_count += 1;
                    }
                }

                // Lateral inhibition: also reduce nodes that support the inhibited target
                if self.config.lateral_inhibition > 0.0 {
                    let supporters: Vec<u32> =
                        self.reverse_adj.get(dst_raw).cloned().unwrap_or_default();

                    for supporter_raw in supporters {
                        // Check if the edge from supporter to dst is a Supports edge
                        let is_supporter = self
                            .adjacency
                            .get(&supporter_raw)
                            .map(|edges| {
                                edges.iter().any(|(d, e)| {
                                    *d == *dst_raw && e.kind == CognitiveEdgeKind::Supports
                                })
                            })
                            .unwrap_or(false);

                        if is_supporter {
                            if let Some(node) = self.nodes.get_mut(&supporter_raw) {
                                let lateral = delta * self.config.lateral_inhibition;
                                let old = node.attrs.activation;
                                node.attrs.activation =
                                    (node.attrs.activation + lateral).clamp(0.0, 1.0);
                                let actual_delta = node.attrs.activation - old;
                                if actual_delta.abs() > 1e-6 {
                                    *self.last_spread_deltas.entry(supporter_raw).or_insert(0.0) +=
                                        actual_delta;
                                }
                            }
                        }
                    }
                }
            }

            // Top-K truncation: only keep the most activated nodes in the frontier
            if next_frontier.len() > self.config.top_k_per_hop {
                next_frontier.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                next_frontier.truncate(self.config.top_k_per_hop);
            }

            // If nothing spread, stop early
            if next_frontier.is_empty() && inhibition_targets.is_empty() {
                break;
            }

            frontier = next_frontier;
        }

        changed_count
    }

    /// Activate multiple seed nodes simultaneously and spread.
    /// Useful for multi-entity queries or routine co-activations.
    pub fn activate_seeds(&mut self, seeds: &[(NodeId, f64)]) -> usize {
        let mut total_changed = 0;

        // Set all seed activations first
        for &(id, activation) in seeds {
            if let Some(node) = self.nodes.get_mut(&id.to_raw()) {
                let old = node.attrs.activation;
                node.attrs.activation = (node.attrs.activation + activation).min(1.0);
                node.attrs.last_updated_ms = now_ms();
                if (node.attrs.activation - old).abs() > 1e-6 {
                    total_changed += 1;
                }
            }
        }

        // Then spread from all seeds
        // We collect frontier from all seeds and do combined spreading
        let mut frontier: Vec<(u32, f64)> = seeds
            .iter()
            .filter_map(|&(id, act)| {
                if self.nodes.contains_key(&id.to_raw()) {
                    Some((id.to_raw(), act))
                } else {
                    None
                }
            })
            .collect();

        self.spread_count += 1;

        for hop in 0..self.config.max_hops {
            let decay = self.config.hop_decay.powi(hop as i32 + 1);
            let mut next_frontier: Vec<(u32, f64)> = Vec::new();

            for &(src_raw, src_activation) in &frontier {
                let edges = match self.adjacency.get(&src_raw) {
                    Some(e) => e.clone(),
                    None => continue,
                };

                for (dst_raw, edge) in &edges {
                    let transfer = edge.effective_activation_transfer();
                    let delta = src_activation * transfer * decay;

                    if delta.abs() < self.config.activation_threshold {
                        continue;
                    }

                    if let Some(node) = self.nodes.get_mut(dst_raw) {
                        let old = node.attrs.activation;
                        node.attrs.activation = (node.attrs.activation + delta).clamp(0.0, 1.0);
                        if (node.attrs.activation - old).abs() > 1e-6 {
                            total_changed += 1;
                            next_frontier.push((*dst_raw, delta.abs()));
                        }
                    }
                }
            }

            if next_frontier.len() > self.config.top_k_per_hop {
                next_frontier.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                next_frontier.truncate(self.config.top_k_per_hop);
            }

            if next_frontier.is_empty() {
                break;
            }

            frontier = next_frontier;
        }

        total_changed
    }

    /// Apply temporal decay to all nodes in the working set.
    /// Call this periodically (e.g., every cognitive tick).
    pub fn decay_all(&mut self, elapsed_secs: f64) {
        for node in self.nodes.values_mut() {
            node.attrs.decay(elapsed_secs);
        }
    }

    /// Evict nodes below an activation threshold.
    /// Returns evicted nodes (for persistence if they're persistent).
    pub fn evict_below_threshold(&mut self, threshold: f64) -> Vec<CognitiveNode> {
        let to_evict: Vec<u32> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.attrs.activation < threshold && n.attrs.persistence < 0.9)
            .map(|(raw, _)| *raw)
            .collect();

        let mut evicted = Vec::new();
        for raw in to_evict {
            let id = NodeId::from_raw(raw);
            if let Some(node) = self.remove(id) {
                evicted.push(node);
            }
        }

        self.last_evicted = evicted.clone();
        evicted
    }

    // ── Queries ──

    /// Find the most activated node of a specific kind.
    pub fn most_active_of_kind(&self, kind: NodeKind) -> Option<&CognitiveNode> {
        self.nodes
            .values()
            .filter(|n| n.kind() == kind)
            .max_by(|a, b| {
                a.attrs
                    .activation
                    .partial_cmp(&b.attrs.activation)
                    .unwrap_or(Ordering::Equal)
            })
    }

    /// Find all active beliefs (activation > threshold).
    pub fn active_beliefs(&self, threshold: f64) -> Vec<&CognitiveNode> {
        self.nodes
            .values()
            .filter(|n| n.kind() == NodeKind::Belief && n.attrs.activation > threshold)
            .collect()
    }

    /// Find all urgent nodes (urgency > threshold).
    pub fn urgent_nodes(&self, threshold: f64) -> Vec<&CognitiveNode> {
        let mut nodes: Vec<_> = self
            .nodes
            .values()
            .filter(|n| n.attrs.urgency > threshold)
            .collect();
        nodes.sort_by(|a, b| {
            b.attrs
                .urgency
                .partial_cmp(&a.attrs.urgency)
                .unwrap_or(Ordering::Equal)
        });
        nodes
    }

    /// Find nodes that support a given node (via Supports edges).
    pub fn supporters_of(&self, id: NodeId) -> Vec<(&CognitiveNode, &CognitiveEdge)> {
        let raw = id.to_raw();
        let sources = match self.reverse_adj.get(&raw) {
            Some(s) => s,
            None => return vec![],
        };

        let mut result = Vec::new();
        for &src_raw in sources {
            if let Some(edges) = self.adjacency.get(&src_raw) {
                for (dst, edge) in edges {
                    if *dst == raw && edge.kind == CognitiveEdgeKind::Supports {
                        if let Some(node) = self.nodes.get(&src_raw) {
                            result.push((node, edge));
                        }
                    }
                }
            }
        }
        result
    }

    /// Find nodes that contradict a given node (via Contradicts edges).
    pub fn contradictors_of(&self, id: NodeId) -> Vec<(&CognitiveNode, &CognitiveEdge)> {
        let raw = id.to_raw();
        let sources = match self.reverse_adj.get(&raw) {
            Some(s) => s,
            None => return vec![],
        };

        let mut result = Vec::new();
        for &src_raw in sources {
            if let Some(edges) = self.adjacency.get(&src_raw) {
                for (dst, edge) in edges {
                    if *dst == raw && edge.kind == CognitiveEdgeKind::Contradicts {
                        if let Some(node) = self.nodes.get(&src_raw) {
                            result.push((node, edge));
                        }
                    }
                }
            }
        }
        result
    }

    /// Get goals that are blocked (have BlocksGoal edges from active nodes).
    pub fn blocked_goals(&self) -> Vec<(&CognitiveNode, Vec<&CognitiveNode>)> {
        let goals: Vec<_> = self.nodes_of_kind(NodeKind::Goal);
        let mut blocked = Vec::new();

        for goal in goals {
            let raw = goal.id.to_raw();
            let blockers: Vec<&CognitiveNode> = self
                .reverse_adj
                .get(&raw)
                .map(|sources| {
                    sources
                        .iter()
                        .filter_map(|src_raw| {
                            let edges = self.adjacency.get(src_raw)?;
                            let is_blocker = edges
                                .iter()
                                .any(|(d, e)| *d == raw && e.kind == CognitiveEdgeKind::BlocksGoal);
                            if is_blocker {
                                self.nodes.get(src_raw)
                            } else {
                                None
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();

            if !blockers.is_empty() {
                blocked.push((goal, blockers));
            }
        }

        blocked
    }

    /// Compute a snapshot of the working set state for debugging.
    pub fn snapshot(&self) -> WorkingSetSnapshot {
        let mut node_summaries: Vec<NodeSummary> = self
            .nodes
            .values()
            .map(|n| NodeSummary {
                id: n.id,
                kind: n.kind(),
                label: n.label.clone(),
                activation: n.attrs.activation,
                relevance: n.attrs.relevance_score(),
                urgency: n.attrs.urgency,
            })
            .collect();

        node_summaries.sort_by(|a, b| {
            b.activation
                .partial_cmp(&a.activation)
                .unwrap_or(Ordering::Equal)
        });

        let edge_count: usize = self.adjacency.values().map(|v| v.len()).sum();

        WorkingSetSnapshot {
            node_count: self.nodes.len(),
            edge_count,
            capacity: self.config.capacity,
            spread_count: self.spread_count,
            nodes: node_summaries,
        }
    }

    // ── Internal ──

    /// Find the best eviction candidate — the node with lowest relevance
    /// that isn't the node being inserted.
    fn find_eviction_candidate(&self, incoming: &CognitiveNode) -> Option<NodeId> {
        self.nodes
            .values()
            .filter(|n| n.id != incoming.id)
            // Don't evict high-persistence nodes (core beliefs, constraints)
            .filter(|n| n.attrs.persistence < 0.95)
            .min_by(|a, b| {
                a.attrs
                    .relevance_score()
                    .partial_cmp(&b.attrs.relevance_score())
                    .unwrap_or(Ordering::Equal)
            })
            .map(|n| n.id)
    }
}

impl Default for WorkingSet {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helper Types ──

/// Min-heap entry for top-K selection (reversed ordering for BinaryHeap).
struct ActivationEntry {
    raw_id: u32,
    activation: f64,
}

impl PartialEq for ActivationEntry {
    fn eq(&self, other: &Self) -> bool {
        self.activation == other.activation
    }
}

impl Eq for ActivationEntry {}

impl PartialOrd for ActivationEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ActivationEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // REVERSED: min-heap behavior — lowest activation at top
        other
            .activation
            .partial_cmp(&self.activation)
            .unwrap_or(Ordering::Equal)
    }
}

/// Summary of a node in the working set (for debugging).
#[derive(Debug, Clone)]
pub struct NodeSummary {
    pub id: NodeId,
    pub kind: NodeKind,
    pub label: String,
    pub activation: f64,
    pub relevance: f64,
    pub urgency: f64,
}

/// Snapshot of the working set state.
#[derive(Debug, Clone)]
pub struct WorkingSetSnapshot {
    pub node_count: usize,
    pub edge_count: usize,
    pub capacity: usize,
    pub spread_count: u64,
    pub nodes: Vec<NodeSummary>,
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a simple node with given activation.
    fn make_node(
        alloc: &mut NodeIdAllocator,
        kind: NodeKind,
        label: &str,
        activation: f64,
    ) -> CognitiveNode {
        let id = alloc.alloc(kind);
        let mut attrs = CognitiveAttrs::default_for(kind);
        attrs.activation = activation;
        CognitiveNode::with_attrs(id, label.to_string(), make_payload(kind, label), attrs)
    }

    fn make_payload(kind: NodeKind, label: &str) -> NodePayload {
        match kind {
            NodeKind::Entity => NodePayload::Entity(EntityPayload {
                name: label.into(),
                entity_type: "test".into(),
                memory_rids: vec![],
            }),
            NodeKind::Belief => NodePayload::Belief(BeliefPayload {
                proposition: label.into(),
                log_odds: 0.0,
                domain: "test".into(),
                evidence_trail: vec![],
                user_confirmed: false,
            }),
            NodeKind::Goal => NodePayload::Goal(GoalPayload {
                description: label.into(),
                status: GoalStatus::Active,
                progress: 0.0,
                deadline: None,
                priority: Priority::Medium,
                parent_goal: None,
                completion_criteria: "test".into(),
            }),
            NodeKind::Task => NodePayload::Task(TaskPayload {
                description: label.into(),
                status: TaskStatus::Pending,
                goal_id: None,
                deadline: None,
                priority: Priority::Medium,
                estimated_minutes: None,
                prerequisites: vec![],
            }),
            NodeKind::Risk => NodePayload::Risk(RiskPayload {
                description: label.into(),
                severity: 0.5,
                likelihood: 0.5,
                mitigation: "test".into(),
                threatened_goals: vec![],
            }),
            _ => NodePayload::Entity(EntityPayload {
                name: label.into(),
                entity_type: "test".into(),
                memory_rids: vec![],
            }),
        }
    }

    #[test]
    fn test_working_set_basic() {
        let mut ws = WorkingSet::new();
        let mut alloc = NodeIdAllocator::new();

        let node = make_node(&mut alloc, NodeKind::Entity, "Alice", 0.5);
        let id = node.id;
        ws.insert(node);

        assert_eq!(ws.len(), 1);
        assert!(!ws.is_empty());
        assert!(!ws.is_full());

        let retrieved = ws.get(id).unwrap();
        assert_eq!(retrieved.label, "Alice");
        // Activation should be boosted by insertion_boost
        assert!(retrieved.attrs.activation > 0.5);
    }

    #[test]
    fn test_working_set_capacity_eviction() {
        let config = AttentionConfig {
            capacity: 3,
            ..AttentionConfig::fast()
        };
        let mut ws = WorkingSet::with_config(config);
        let mut alloc = NodeIdAllocator::new();

        // Insert 3 nodes — should fit
        let n1 = make_node(&mut alloc, NodeKind::Entity, "A", 0.3);
        let n2 = make_node(&mut alloc, NodeKind::Entity, "B", 0.8);
        let n3 = make_node(&mut alloc, NodeKind::Entity, "C", 0.6);

        ws.insert(n1);
        ws.insert(n2);
        ws.insert(n3);
        assert_eq!(ws.len(), 3);

        // 4th node should evict the least relevant
        let n4 = make_node(&mut alloc, NodeKind::Entity, "D", 0.9);
        let evicted = ws.insert(n4);
        assert!(evicted.is_some());
        assert_eq!(ws.len(), 3);
    }

    #[test]
    fn test_spreading_activation_basic() {
        let mut ws = WorkingSet::with_config(AttentionConfig::fast());
        let mut alloc = NodeIdAllocator::new();

        let entity = make_node(&mut alloc, NodeKind::Entity, "Alice", 0.0);
        let belief = make_node(&mut alloc, NodeKind::Belief, "Alice is kind", 0.0);
        let entity_id = entity.id;
        let belief_id = belief.id;

        ws.insert(entity);
        ws.insert(belief);

        // Add a Supports edge: entity → belief
        ws.add_edge(CognitiveEdge::new(
            entity_id,
            belief_id,
            CognitiveEdgeKind::Supports,
            0.8,
        ));

        // Activate entity and spread
        let changed = ws.activate_and_spread(entity_id, 1.0);
        assert!(changed >= 1);

        // Belief should have received activation
        let belief_node = ws.get(belief_id).unwrap();
        assert!(
            belief_node.attrs.activation > 0.0,
            "Belief should receive activation via Supports edge, got {}",
            belief_node.attrs.activation
        );
    }

    #[test]
    fn test_spreading_activation_inhibition() {
        let mut ws = WorkingSet::with_config(AttentionConfig::fast());
        let mut alloc = NodeIdAllocator::new();

        let evidence = make_node(&mut alloc, NodeKind::Episode, "Evidence", 0.0);
        let belief = make_node(&mut alloc, NodeKind::Belief, "Wrong belief", 0.8);
        let evidence_id = evidence.id;
        let belief_id = belief.id;

        ws.insert(evidence);
        ws.insert(belief);

        // Contradicts edge: evidence → belief (should inhibit)
        ws.add_edge(CognitiveEdge::new(
            evidence_id,
            belief_id,
            CognitiveEdgeKind::Contradicts,
            0.9,
        ));

        let before = ws.get(belief_id).unwrap().attrs.activation;
        ws.activate_and_spread(evidence_id, 1.0);
        let after = ws.get(belief_id).unwrap().attrs.activation;

        assert!(
            after < before,
            "Contradicts edge should inhibit belief: before={before}, after={after}"
        );
    }

    #[test]
    fn test_spreading_multi_hop() {
        let config = AttentionConfig {
            max_hops: 2,
            ..AttentionConfig::balanced()
        };
        let mut ws = WorkingSet::with_config(config);
        let mut alloc = NodeIdAllocator::new();

        // Chain: A → B → C
        let a = make_node(&mut alloc, NodeKind::Entity, "A", 0.0);
        let b = make_node(&mut alloc, NodeKind::Entity, "B", 0.0);
        let c = make_node(&mut alloc, NodeKind::Entity, "C", 0.0);
        let a_id = a.id;
        let b_id = b.id;
        let c_id = c.id;

        ws.insert(a);
        ws.insert(b);
        ws.insert(c);

        ws.add_edge(CognitiveEdge::new(
            a_id,
            b_id,
            CognitiveEdgeKind::Causes,
            0.9,
        ));
        ws.add_edge(CognitiveEdge::new(
            b_id,
            c_id,
            CognitiveEdgeKind::Causes,
            0.9,
        ));

        ws.activate_and_spread(a_id, 1.0);

        let b_act = ws.get(b_id).unwrap().attrs.activation;
        let c_act = ws.get(c_id).unwrap().attrs.activation;

        assert!(b_act > 0.0, "B should receive activation from A");
        assert!(c_act > 0.0, "C should receive activation from B (2nd hop)");
        assert!(
            b_act > c_act,
            "B should have more activation than C (decay per hop)"
        );
    }

    #[test]
    fn test_decay_all() {
        let mut ws = WorkingSet::new();
        let mut alloc = NodeIdAllocator::new();

        let node = make_node(&mut alloc, NodeKind::Belief, "test", 1.0);
        let id = node.id;
        ws.insert(node);

        // Force the activation to a known value (bypassing insertion boost)
        ws.get_mut(id).unwrap().attrs.activation = 1.0;

        ws.decay_all(600.0); // 10 minutes

        let after = ws.get(id).unwrap().attrs.activation;
        assert!(after < 1.0, "Activation should decay after 10 minutes");
        assert!(after > 0.0, "Activation shouldn't be zero after 10 minutes");
    }

    #[test]
    fn test_evict_below_threshold() {
        let mut ws = WorkingSet::new();
        let mut alloc = NodeIdAllocator::new();

        let high = make_node(&mut alloc, NodeKind::Entity, "Active", 0.8);
        let low = make_node(&mut alloc, NodeKind::Entity, "Dormant", 0.01);
        let high_id = high.id;

        ws.insert(high);
        ws.insert(low);

        // Force activations
        ws.get_mut(high_id).unwrap().attrs.activation = 0.8;
        let low_raw: u32 = ws
            .iter()
            .find(|n| n.label == "Dormant")
            .unwrap()
            .id
            .to_raw();
        ws.nodes.get_mut(&low_raw).unwrap().attrs.activation = 0.01;
        ws.nodes.get_mut(&low_raw).unwrap().attrs.persistence = 0.1;

        let evicted = ws.evict_below_threshold(0.05);
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].label, "Dormant");
        assert_eq!(ws.len(), 1);
    }

    #[test]
    fn test_top_k() {
        let mut ws = WorkingSet::new();
        let mut alloc = NodeIdAllocator::new();

        for i in 0..10 {
            let mut node = make_node(&mut alloc, NodeKind::Entity, &format!("N{i}"), 0.0);
            node.attrs.activation = i as f64 * 0.1;
            ws.insert(node);
        }

        // Force activations (insertion boost messes with exact values)
        for node in ws.iter_mut() {
            let idx: f64 = node.label[1..].parse().unwrap();
            node.attrs.activation = idx * 0.1;
        }

        let top3 = ws.top_k(3);
        assert_eq!(top3.len(), 3);
        assert!(top3[0].attrs.activation >= top3[1].attrs.activation);
        assert!(top3[1].attrs.activation >= top3[2].attrs.activation);
    }

    #[test]
    fn test_nodes_of_kind() {
        let mut ws = WorkingSet::new();
        let mut alloc = NodeIdAllocator::new();

        ws.insert(make_node(&mut alloc, NodeKind::Entity, "E1", 0.5));
        ws.insert(make_node(&mut alloc, NodeKind::Entity, "E2", 0.5));
        ws.insert(make_node(&mut alloc, NodeKind::Belief, "B1", 0.5));
        ws.insert(make_node(&mut alloc, NodeKind::Goal, "G1", 0.5));

        assert_eq!(ws.nodes_of_kind(NodeKind::Entity).len(), 2);
        assert_eq!(ws.nodes_of_kind(NodeKind::Belief).len(), 1);
        assert_eq!(ws.nodes_of_kind(NodeKind::Goal).len(), 1);
        assert_eq!(ws.nodes_of_kind(NodeKind::Task).len(), 0);
    }

    #[test]
    fn test_supporters_and_contradictors() {
        let mut ws = WorkingSet::new();
        let mut alloc = NodeIdAllocator::new();

        let belief = make_node(&mut alloc, NodeKind::Belief, "Main belief", 0.5);
        let support = make_node(&mut alloc, NodeKind::Episode, "Supporting evidence", 0.5);
        let contra = make_node(&mut alloc, NodeKind::Episode, "Contradicting evidence", 0.5);
        let belief_id = belief.id;
        let support_id = support.id;
        let contra_id = contra.id;

        ws.insert(belief);
        ws.insert(support);
        ws.insert(contra);

        ws.add_edge(CognitiveEdge::new(
            support_id,
            belief_id,
            CognitiveEdgeKind::Supports,
            0.8,
        ));
        ws.add_edge(CognitiveEdge::new(
            contra_id,
            belief_id,
            CognitiveEdgeKind::Contradicts,
            0.7,
        ));

        let supporters = ws.supporters_of(belief_id);
        assert_eq!(supporters.len(), 1);
        assert_eq!(supporters[0].0.label, "Supporting evidence");

        let contradictors = ws.contradictors_of(belief_id);
        assert_eq!(contradictors.len(), 1);
        assert_eq!(contradictors[0].0.label, "Contradicting evidence");
    }

    #[test]
    fn test_blocked_goals() {
        let mut ws = WorkingSet::new();
        let mut alloc = NodeIdAllocator::new();

        let goal = make_node(&mut alloc, NodeKind::Goal, "Important goal", 0.7);
        let risk = make_node(&mut alloc, NodeKind::Risk, "Blocking risk", 0.6);
        let goal_id = goal.id;
        let risk_id = risk.id;

        ws.insert(goal);
        ws.insert(risk);
        ws.add_edge(CognitiveEdge::new(
            risk_id,
            goal_id,
            CognitiveEdgeKind::BlocksGoal,
            0.8,
        ));

        let blocked = ws.blocked_goals();
        assert_eq!(blocked.len(), 1);
        assert_eq!(blocked[0].0.label, "Important goal");
        assert_eq!(blocked[0].1.len(), 1);
    }

    #[test]
    fn test_snapshot() {
        let mut ws = WorkingSet::new();
        let mut alloc = NodeIdAllocator::new();

        ws.insert(make_node(&mut alloc, NodeKind::Entity, "A", 0.8));
        ws.insert(make_node(&mut alloc, NodeKind::Belief, "B", 0.3));

        let snap = ws.snapshot();
        assert_eq!(snap.node_count, 2);
        assert_eq!(snap.capacity, 32);
        assert_eq!(snap.nodes.len(), 2);
    }

    #[test]
    fn test_activate_seeds() {
        // Use balanced config with low threshold to ensure spreading works
        let config = AttentionConfig {
            activation_threshold: 0.01,
            ..AttentionConfig::balanced()
        };
        let mut ws = WorkingSet::with_config(config);
        let mut alloc = NodeIdAllocator::new();

        let a = make_node(&mut alloc, NodeKind::Entity, "A", 0.0);
        let b = make_node(&mut alloc, NodeKind::Entity, "B", 0.0);
        let c = make_node(&mut alloc, NodeKind::Entity, "C", 0.0);
        let a_id = a.id;
        let b_id = b.id;
        let c_id = c.id;

        ws.insert(a);
        ws.insert(b);
        ws.insert(c);

        // Force zero activation after insertion boost
        for n in ws.iter_mut() {
            n.attrs.activation = 0.0;
        }

        // Strong Causes edges (0.8 transfer factor) for reliable spreading
        ws.add_edge(CognitiveEdge::new(
            a_id,
            c_id,
            CognitiveEdgeKind::Causes,
            0.9,
        ));
        ws.add_edge(CognitiveEdge::new(
            b_id,
            c_id,
            CognitiveEdgeKind::Causes,
            0.9,
        ));

        let changed = ws.activate_seeds(&[(a_id, 0.8), (b_id, 0.8)]);
        assert!(changed >= 2, "At least seeds should change, got {changed}");

        // C should receive activation from both A and B
        let c_act = ws.get(c_id).unwrap().attrs.activation;
        assert!(
            c_act > 0.0,
            "C should receive combined activation from A and B, got {c_act}"
        );
    }

    #[test]
    fn test_remove_node_cleans_edges() {
        let mut ws = WorkingSet::new();
        let mut alloc = NodeIdAllocator::new();

        let a = make_node(&mut alloc, NodeKind::Entity, "A", 0.5);
        let b = make_node(&mut alloc, NodeKind::Entity, "B", 0.5);
        let c = make_node(&mut alloc, NodeKind::Entity, "C", 0.5);
        let a_id = a.id;
        let b_id = b.id;
        let c_id = c.id;

        ws.insert(a);
        ws.insert(b);
        ws.insert(c);

        ws.add_edge(CognitiveEdge::new(
            a_id,
            b_id,
            CognitiveEdgeKind::Supports,
            0.8,
        ));
        ws.add_edge(CognitiveEdge::new(
            b_id,
            c_id,
            CognitiveEdgeKind::Causes,
            0.7,
        ));

        // Remove B — should clean all edges involving B
        ws.remove(b_id);
        assert_eq!(ws.len(), 2);
        assert!(
            ws.edges_from(a_id).is_empty()
                || ws.edges_from(a_id).iter().all(|(d, _)| *d != b_id.to_raw())
        );
    }

    #[test]
    fn test_lateral_inhibition() {
        let config = AttentionConfig {
            lateral_inhibition: 0.5, // Strong lateral inhibition
            ..AttentionConfig::fast()
        };
        let mut ws = WorkingSet::with_config(config);
        let mut alloc = NodeIdAllocator::new();

        // Setup: supporter → belief ← contradicting_evidence
        let supporter = make_node(&mut alloc, NodeKind::Episode, "Support", 0.5);
        let belief = make_node(&mut alloc, NodeKind::Belief, "Contested", 0.5);
        let evidence = make_node(&mut alloc, NodeKind::Episode, "Contradiction", 0.0);
        let supporter_id = supporter.id;
        let belief_id = belief.id;
        let evidence_id = evidence.id;

        ws.insert(supporter);
        ws.insert(belief);
        ws.insert(evidence);

        // Force known activations
        ws.get_mut(supporter_id).unwrap().attrs.activation = 0.5;
        ws.get_mut(belief_id).unwrap().attrs.activation = 0.5;
        ws.get_mut(evidence_id).unwrap().attrs.activation = 0.0;

        ws.add_edge(CognitiveEdge::new(
            supporter_id,
            belief_id,
            CognitiveEdgeKind::Supports,
            0.8,
        ));
        ws.add_edge(CognitiveEdge::new(
            evidence_id,
            belief_id,
            CognitiveEdgeKind::Contradicts,
            0.9,
        ));

        let supporter_before = ws.get(supporter_id).unwrap().attrs.activation;

        // Activate contradiction
        ws.activate_and_spread(evidence_id, 1.0);

        let supporter_after = ws.get(supporter_id).unwrap().attrs.activation;

        // Lateral inhibition should reduce the supporter's activation too
        assert!(supporter_after < supporter_before,
            "Lateral inhibition should reduce supporter: before={supporter_before}, after={supporter_after}");
    }

    #[test]
    fn test_urgent_nodes() {
        let mut ws = WorkingSet::new();
        let mut alloc = NodeIdAllocator::new();

        let mut urgent_task = make_node(&mut alloc, NodeKind::Task, "Deadline!", 0.5);
        urgent_task.attrs.urgency = 0.9;
        let mut calm_task = make_node(&mut alloc, NodeKind::Task, "Whenever", 0.5);
        calm_task.attrs.urgency = 0.1;

        ws.insert(urgent_task);
        ws.insert(calm_task);

        let urgent = ws.urgent_nodes(0.5);
        assert_eq!(urgent.len(), 1);
        assert_eq!(urgent[0].label, "Deadline!");
    }

    #[test]
    fn test_insert_with_edges() {
        let mut ws = WorkingSet::new();
        let mut alloc = NodeIdAllocator::new();

        let a = make_node(&mut alloc, NodeKind::Entity, "A", 0.5);
        let a_id = a.id;
        ws.insert(a);

        let b = make_node(&mut alloc, NodeKind::Entity, "B", 0.5);
        let b_id = b.id;

        ws.insert_with_edges(
            b,
            vec![CognitiveEdge::new(
                a_id,
                b_id,
                CognitiveEdgeKind::AssociatedWith,
                0.7,
            )],
        );

        assert_eq!(ws.len(), 2);
        assert_eq!(ws.edges_from(a_id).len(), 1);
    }

    #[test]
    fn test_config_modes() {
        let fast = AttentionConfig::fast();
        let balanced = AttentionConfig::balanced();
        let deep = AttentionConfig::deep();

        assert!(fast.max_hops < balanced.max_hops);
        assert!(balanced.max_hops < deep.max_hops);
        assert!(fast.top_k_per_hop < deep.top_k_per_hop);
    }

    #[test]
    fn test_clear() {
        let mut ws = WorkingSet::new();
        let mut alloc = NodeIdAllocator::new();

        ws.insert(make_node(&mut alloc, NodeKind::Entity, "A", 0.5));
        ws.insert(make_node(&mut alloc, NodeKind::Belief, "B", 0.5));
        assert_eq!(ws.len(), 2);

        ws.clear();
        assert!(ws.is_empty());
        assert_eq!(ws.len(), 0);
    }

    #[test]
    fn test_edge_not_added_for_missing_node() {
        let mut ws = WorkingSet::new();
        let mut alloc = NodeIdAllocator::new();

        let a = make_node(&mut alloc, NodeKind::Entity, "A", 0.5);
        let a_id = a.id;
        ws.insert(a);

        // B doesn't exist in the working set
        let b_id = alloc.alloc(NodeKind::Entity);
        let added = ws.add_edge(CognitiveEdge::new(
            a_id,
            b_id,
            CognitiveEdgeKind::Supports,
            0.5,
        ));
        assert!(!added, "Edge should not be added when endpoint is missing");
    }

    #[test]
    fn test_spreading_activation_respects_threshold() {
        let config = AttentionConfig {
            activation_threshold: 0.5, // Very high threshold
            max_hops: 1,
            ..AttentionConfig::fast()
        };
        let mut ws = WorkingSet::with_config(config);
        let mut alloc = NodeIdAllocator::new();

        let a = make_node(&mut alloc, NodeKind::Entity, "A", 0.0);
        let b = make_node(&mut alloc, NodeKind::Entity, "B", 0.0);
        let a_id = a.id;
        let b_id = b.id;

        ws.insert(a);
        ws.insert(b);

        // Weak edge — transfer will be below threshold
        ws.add_edge(CognitiveEdge::new(
            a_id,
            b_id,
            CognitiveEdgeKind::SimilarTo,
            0.1,
        ));

        // Force zero
        ws.get_mut(b_id).unwrap().attrs.activation = 0.0;

        ws.activate_and_spread(a_id, 0.3);

        // B should NOT receive activation because transfer < threshold
        let b_act = ws.get(b_id).unwrap().attrs.activation;
        assert!(
            b_act < 0.01,
            "Weak edge below threshold shouldn't spread: got {b_act}"
        );
    }
}
