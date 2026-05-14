//! CK-4.3 — Coherence Monitor + Cognitive Health.
//!
//! The "sanity checker" that keeps the reasoning engine internally
//! consistent. Detects goal conflicts, belief contradictions, attention
//! fragmentation, zombie activations, circular dependencies, and orphans.
//!
//! # Design principles
//! - Pure functions only — no DB access
//! - Conservative enforcement — prefer demoting over deleting
//! - Explainable — every enforcement action has rationale
//! - Incremental — runs each cognitive tick, not batch

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::attention::WorkingSet;
use crate::contradiction::BeliefConflict;
use crate::state::{
    CognitiveEdge, CognitiveEdgeKind, CognitiveNode, GoalPayload, GoalStatus, NodeId, NodeKind,
    NodePayload, TaskPayload, TaskStatus,
};

// ── §1: Coherence report ────────────────────────────────────────────

/// Comprehensive cognitive health report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceReport {
    /// Pairs of active goals that conflict with each other.
    pub goal_conflicts: Vec<GoalConflict>,
    /// Belief contradictions where both sides have high confidence.
    pub belief_contradictions: Vec<BeliefContradiction>,
    /// How fragmented attention is ∈ [0.0, 1.0].
    /// 0.0 = focused on few items, 1.0 = scattered across many.
    pub attention_fragmentation: f64,
    /// Nodes that are active but have not progressed.
    pub stale_activations: Vec<StaleNode>,
    /// Working set load ∈ [0.0, 1.0+].
    pub cognitive_load: f64,
    /// Tasks/beliefs with no supporting context.
    pub orphaned_items: Vec<OrphanedItem>,
    /// Circular dependency cycles in goal/task graph.
    pub circular_dependencies: Vec<DependencyCycle>,
    /// Goals with deadline pressure.
    pub deadline_pressure: Vec<DeadlineAlert>,
    /// Overall coherence score ∈ [0.0, 1.0].
    pub coherence_score: f64,
    /// When the check was performed (unix seconds).
    pub checked_at: f64,
}

/// Two active goals that work against each other.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalConflict {
    pub goal_a: NodeId,
    pub goal_a_desc: String,
    pub goal_b: NodeId,
    pub goal_b_desc: String,
    /// How severe the conflict is ∈ [0.0, 1.0].
    pub severity: f64,
    /// Explanation of why they conflict.
    pub reason: String,
}

/// Two beliefs that contradict each other, both with non-trivial confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefContradiction {
    pub belief_a: NodeId,
    pub belief_b: NodeId,
    pub severity: f64,
    pub description: String,
}

/// A node that is "active" but hasn't been updated recently.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaleNode {
    pub node_id: NodeId,
    pub label: String,
    pub kind: NodeKind,
    pub activation: f64,
    /// How long since last update (seconds).
    pub age_secs: f64,
}

/// An item with no supporting edges or parent context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrphanedItem {
    pub node_id: NodeId,
    pub label: String,
    pub kind: NodeKind,
    /// Why it's considered orphaned.
    pub reason: String,
}

/// A cycle in the dependency graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyCycle {
    pub nodes: Vec<NodeId>,
    /// The weakest edge in the cycle (candidate for breaking).
    pub weakest_edge_weight: f64,
}

/// A goal approaching or past its deadline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadlineAlert {
    pub goal_id: NodeId,
    pub description: String,
    pub deadline: f64,
    /// Seconds remaining (negative = overdue).
    pub remaining_secs: f64,
    pub progress: f64,
}

// ── §2: Coherence configuration ─────────────────────────────────────

/// Configuration for coherence checking and enforcement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceConfig {
    /// Weight for goal conflicts in coherence score.
    pub w_conflict: f64,
    /// Weight for belief contradictions.
    pub w_contradiction: f64,
    /// Weight for attention fragmentation.
    pub w_fragmentation: f64,
    /// Weight for zombie/stale nodes.
    pub w_zombie: f64,
    /// Weight for cognitive overload.
    pub w_overload: f64,
    /// Threshold below which emergency consolidation triggers.
    pub emergency_threshold: f64,
    /// Seconds since last update before a node is considered stale.
    pub stale_threshold_secs: f64,
    /// Minimum activation for a node to be considered "active".
    pub active_threshold: f64,
    /// Maximum cognitive load before overload penalties.
    pub load_target: f64,
    /// Activation reduction for zombie nodes during enforcement.
    pub zombie_demotion_factor: f64,
    /// Minimum confidence for a belief contradiction to be reported.
    pub min_contradiction_confidence: f64,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            w_conflict: 0.15,
            w_contradiction: 0.10,
            w_fragmentation: 0.20,
            w_zombie: 0.10,
            w_overload: 0.15,
            emergency_threshold: 0.3,
            stale_threshold_secs: 86400.0 * 3.0, // 3 days
            active_threshold: 0.3,
            load_target: 0.8,
            zombie_demotion_factor: 0.5,
            min_contradiction_confidence: 0.5,
        }
    }
}

// ── §3: Coherence inputs ────────────────────────────────────────────

/// Everything needed for a coherence check — passed by the engine layer.
pub struct CoherenceInputs<'a> {
    pub working_set: &'a WorkingSet,
    pub edges: &'a [CognitiveEdge],
    pub belief_conflicts: &'a [BeliefConflict],
    pub config: &'a CoherenceConfig,
    pub now: f64,
}

// ── §4: Main coherence check ────────────────────────────────────────

/// Perform a comprehensive coherence check.
///
/// This is the main entry point. Analyzes the cognitive state for
/// conflicts, contradictions, fragmentation, staleness, orphans,
/// and circular dependencies. Returns a scored report.
pub fn check_coherence(inputs: &CoherenceInputs) -> CoherenceReport {
    let ws = inputs.working_set;
    let config = inputs.config;
    let now = inputs.now;

    // 1. Goal conflicts.
    let goal_conflicts = detect_goal_conflicts(ws, inputs.edges);

    // 2. Belief contradictions.
    let belief_contradictions = filter_contradictions(inputs.belief_conflicts, config);

    // 3. Attention fragmentation.
    let attention_fragmentation = compute_fragmentation(ws);

    // 4. Stale activations (zombies).
    let stale_activations = detect_stale_nodes(ws, now, config);

    // 5. Cognitive load.
    let cognitive_load = ws.len() as f64 / ws.config().capacity.max(1) as f64;

    // 6. Orphaned items.
    let orphaned_items = detect_orphans(ws, inputs.edges);

    // 7. Circular dependencies.
    let circular_dependencies = detect_cycles(ws, inputs.edges);

    // 8. Deadline pressure.
    let deadline_pressure = detect_deadline_pressure(ws, now);

    // 9. Compute coherence score.
    let total_active = ws.len().max(1) as f64;
    let coherence_score = compute_coherence_score(
        &goal_conflicts,
        &belief_contradictions,
        attention_fragmentation,
        &stale_activations,
        cognitive_load,
        total_active,
        config,
    );

    CoherenceReport {
        goal_conflicts,
        belief_contradictions,
        attention_fragmentation,
        stale_activations,
        cognitive_load,
        orphaned_items,
        circular_dependencies,
        deadline_pressure,
        coherence_score,
        checked_at: now,
    }
}

// ── §5: Enforcement ─────────────────────────────────────────────────

/// Actions taken during enforcement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementReport {
    /// What was done.
    pub actions: Vec<EnforcementAction>,
    /// Total items affected.
    pub items_affected: usize,
    /// Conflicts resolved.
    pub conflicts_resolved: usize,
    /// Zombies demoted.
    pub zombies_demoted: usize,
    /// Items pruned from working set.
    pub items_pruned: usize,
    /// Coherence score before enforcement.
    pub score_before: f64,
    /// Whether emergency consolidation was triggered.
    pub emergency_triggered: bool,
}

/// A single enforcement action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcementAction {
    pub kind: EnforcementKind,
    pub target: NodeId,
    pub description: String,
}

/// Types of enforcement actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnforcementKind {
    /// Reduced activation of a stale node.
    DemoteZombie,
    /// Removed lowest-salience items from working set.
    PruneLowSalience,
    /// Broke a circular dependency.
    BreakCycle,
    /// Demoted a contradicted belief.
    ResolveContradiction,
    /// Archived a stale goal.
    ArchiveStaleGoal,
}

/// Generate enforcement actions based on a coherence report.
///
/// Returns the actions that SHOULD be taken — the engine layer
/// actually applies them to the working set.
pub fn plan_enforcement(
    report: &CoherenceReport,
    ws: &WorkingSet,
    config: &CoherenceConfig,
) -> EnforcementReport {
    let mut actions = Vec::new();
    let score_before = report.coherence_score;
    let emergency = score_before < config.emergency_threshold;

    // 1. Demote zombie nodes.
    let mut zombies_demoted = 0usize;
    for stale in &report.stale_activations {
        if stale.activation > config.active_threshold {
            actions.push(EnforcementAction {
                kind: EnforcementKind::DemoteZombie,
                target: stale.node_id,
                description: format!(
                    "Demote stale {} '{}' (inactive {:.0}h, activation {:.2} → {:.2})",
                    kind_name(stale.kind),
                    stale.label,
                    stale.age_secs / 3600.0,
                    stale.activation,
                    stale.activation * config.zombie_demotion_factor,
                ),
            });
            zombies_demoted += 1;
        }
    }

    // 2. Prune low-salience items when overloaded.
    let mut items_pruned = 0usize;
    if report.cognitive_load > config.load_target {
        let excess = ws.len() as f64 - (ws.config().capacity as f64 * config.load_target);
        let to_prune = excess.ceil() as usize;

        // Get nodes sorted by salience (lowest first).
        let mut candidates: Vec<&CognitiveNode> = ws
            .iter()
            .filter(|n| {
                // Don't prune active goals or in-progress tasks.
                match &n.payload {
                    NodePayload::Goal(g) => g.status != GoalStatus::Active,
                    NodePayload::Task(t) => t.status != TaskStatus::InProgress,
                    _ => true,
                }
            })
            .collect();
        candidates.sort_by(|a, b| {
            a.attrs
                .salience
                .partial_cmp(&b.attrs.salience)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for node in candidates.into_iter().take(to_prune) {
            actions.push(EnforcementAction {
                kind: EnforcementKind::PruneLowSalience,
                target: node.id,
                description: format!(
                    "Prune low-salience {} '{}' (salience={:.2})",
                    kind_name(node.id.kind()),
                    node.label,
                    node.attrs.salience,
                ),
            });
            items_pruned += 1;
        }
    }

    // 3. Break circular dependencies (weakest link).
    for cycle in &report.circular_dependencies {
        if cycle.nodes.len() >= 2 {
            // Target the first node in the cycle for simplicity.
            // The engine layer will remove the weakest edge.
            actions.push(EnforcementAction {
                kind: EnforcementKind::BreakCycle,
                target: cycle.nodes[0],
                description: format!(
                    "Break cycle involving {} nodes (weakest edge weight={:.2})",
                    cycle.nodes.len(),
                    cycle.weakest_edge_weight,
                ),
            });
        }
    }

    // 4. Resolve belief contradictions (highest-severity first).
    let mut conflicts_resolved = 0usize;
    let mut sorted_contradictions = report.belief_contradictions.clone();
    sorted_contradictions.sort_by(|a, b| {
        b.severity
            .partial_cmp(&a.severity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let max_resolutions = if emergency { 5 } else { 2 };
    for contradiction in sorted_contradictions.iter().take(max_resolutions) {
        // Pick the belief with lower evidence count to demote.
        let (target, description) = pick_contradiction_loser(contradiction, ws);
        actions.push(EnforcementAction {
            kind: EnforcementKind::ResolveContradiction,
            target,
            description,
        });
        conflicts_resolved += 1;
    }

    // 5. Archive stale goals (no progress, very old).
    for stale in &report.stale_activations {
        if stale.kind == NodeKind::Goal && stale.age_secs > config.stale_threshold_secs * 2.0 {
            actions.push(EnforcementAction {
                kind: EnforcementKind::ArchiveStaleGoal,
                target: stale.node_id,
                description: format!(
                    "Archive stale goal '{}' (inactive {:.0} days)",
                    stale.label,
                    stale.age_secs / 86400.0,
                ),
            });
        }
    }

    let items_affected = actions.len();

    EnforcementReport {
        actions,
        items_affected,
        conflicts_resolved,
        zombies_demoted,
        items_pruned,
        score_before,
        emergency_triggered: emergency,
    }
}

// ── §6: Coherence history ───────────────────────────────────────────

/// A timestamped coherence score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceSnapshot {
    pub timestamp: f64,
    pub score: f64,
    pub goal_conflicts: usize,
    pub belief_contradictions: usize,
    pub cognitive_load: f64,
    pub stale_count: usize,
}

/// Rolling history of coherence scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceHistory {
    snapshots: Vec<CoherenceSnapshot>,
    max_snapshots: usize,
    pub total_checks: u64,
    pub total_enforcements: u64,
    pub total_emergency_triggers: u64,
}

impl CoherenceHistory {
    pub fn new(max_snapshots: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            max_snapshots,
            total_checks: 0,
            total_enforcements: 0,
            total_emergency_triggers: 0,
        }
    }

    /// Record a coherence check result.
    pub fn record(&mut self, report: &CoherenceReport) {
        self.total_checks += 1;
        self.snapshots.push(CoherenceSnapshot {
            timestamp: report.checked_at,
            score: report.coherence_score,
            goal_conflicts: report.goal_conflicts.len(),
            belief_contradictions: report.belief_contradictions.len(),
            cognitive_load: report.cognitive_load,
            stale_count: report.stale_activations.len(),
        });
        if self.snapshots.len() > self.max_snapshots {
            self.snapshots.remove(0);
        }
    }

    /// Record an enforcement.
    pub fn record_enforcement(&mut self, emergency: bool) {
        self.total_enforcements += 1;
        if emergency {
            self.total_emergency_triggers += 1;
        }
    }

    /// Get snapshots within a time window.
    pub fn in_window(&self, since: f64, until: f64) -> Vec<&CoherenceSnapshot> {
        self.snapshots
            .iter()
            .filter(|s| s.timestamp >= since && s.timestamp <= until)
            .collect()
    }

    /// Get the most recent snapshot.
    pub fn latest(&self) -> Option<&CoherenceSnapshot> {
        self.snapshots.last()
    }

    /// Average coherence score over the last N snapshots.
    pub fn recent_average(&self, n: usize) -> f64 {
        let recent: Vec<f64> = self
            .snapshots
            .iter()
            .rev()
            .take(n)
            .map(|s| s.score)
            .collect();
        if recent.is_empty() {
            return 1.0;
        }
        recent.iter().sum::<f64>() / recent.len() as f64
    }

    /// Trend: is coherence improving or degrading?
    /// Returns slope of last N scores (positive = improving).
    pub fn trend(&self, n: usize) -> f64 {
        let scores: Vec<f64> = self
            .snapshots
            .iter()
            .rev()
            .take(n)
            .map(|s| s.score)
            .collect();
        if scores.len() < 2 {
            return 0.0;
        }
        // Simple slope from first to last.
        let first = scores.last().unwrap();
        let last = scores.first().unwrap();
        (last - first) / (scores.len() as f64 - 1.0)
    }

    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }
}

// ── §7: Internal — goal conflict detection ──────────────────────────

/// Detect conflicting active goals.
fn detect_goal_conflicts(ws: &WorkingSet, edges: &[CognitiveEdge]) -> Vec<GoalConflict> {
    let mut conflicts = Vec::new();

    let goals: Vec<&CognitiveNode> = ws.nodes_of_kind(NodeKind::Goal);
    let active_goals: Vec<&CognitiveNode> = goals
        .into_iter()
        .filter(|n| {
            if let NodePayload::Goal(g) = &n.payload {
                g.status == GoalStatus::Active
            } else {
                false
            }
        })
        .collect();

    // Check for BlocksGoal edges between active goals.
    for i in 0..active_goals.len() {
        for j in (i + 1)..active_goals.len() {
            let a = active_goals[i];
            let b = active_goals[j];

            let blocks = edges.iter().any(|e| {
                (e.src == a.id && e.dst == b.id && e.kind == CognitiveEdgeKind::BlocksGoal)
                    || (e.src == b.id && e.dst == a.id && e.kind == CognitiveEdgeKind::BlocksGoal)
                    || (e.src == a.id && e.dst == b.id && e.kind == CognitiveEdgeKind::Contradicts)
                    || (e.src == b.id && e.dst == a.id && e.kind == CognitiveEdgeKind::Contradicts)
            });

            if blocks {
                let desc_a = goal_desc(a);
                let desc_b = goal_desc(b);
                let severity = (a.attrs.urgency + b.attrs.urgency) / 2.0;

                conflicts.push(GoalConflict {
                    goal_a: a.id,
                    goal_a_desc: desc_a.clone(),
                    goal_b: b.id,
                    goal_b_desc: desc_b.clone(),
                    severity,
                    reason: format!("'{}' blocks '{}'", desc_a, desc_b),
                });
            }
        }
    }

    conflicts
}

/// Extract goal description from a cognitive node.
fn goal_desc(node: &CognitiveNode) -> String {
    if let NodePayload::Goal(g) = &node.payload {
        g.description.clone()
    } else {
        node.label.clone()
    }
}

// ── §8: Internal — belief contradictions ────────────────────────────

/// Filter belief conflicts to only high-severity ones.
fn filter_contradictions(
    conflicts: &[BeliefConflict],
    config: &CoherenceConfig,
) -> Vec<BeliefContradiction> {
    conflicts
        .iter()
        .filter(|c| c.severity >= config.min_contradiction_confidence)
        .map(|c| BeliefContradiction {
            belief_a: c.belief_a,
            belief_b: c.belief_b,
            severity: c.severity,
            description: c.description.clone(),
        })
        .collect()
}

/// Pick which belief to demote in a contradiction.
fn pick_contradiction_loser(
    contradiction: &BeliefContradiction,
    ws: &WorkingSet,
) -> (NodeId, String) {
    let node_a = ws.get(contradiction.belief_a);
    let node_b = ws.get(contradiction.belief_b);

    let (loser, winner_label) = match (node_a, node_b) {
        (Some(a), Some(b)) => {
            // Prefer keeping the one with more evidence.
            if a.attrs.evidence_count >= b.attrs.evidence_count {
                (b.id, a.label.clone())
            } else {
                (a.id, b.label.clone())
            }
        }
        (None, Some(_)) => (contradiction.belief_a, "unknown".to_string()),
        (Some(_), None) => (contradiction.belief_b, "unknown".to_string()),
        (None, None) => (contradiction.belief_a, "unknown".to_string()),
    };

    (
        loser,
        format!(
            "Demote contradicted belief in favor of '{}' (higher evidence)",
            winner_label
        ),
    )
}

// ── §9: Internal — fragmentation ────────────────────────────────────

/// Compute attention fragmentation.
///
/// High fragmentation = many nodes with similar activation levels
/// (no clear focus). Low fragmentation = few nodes dominate.
fn compute_fragmentation(ws: &WorkingSet) -> f64 {
    if ws.len() <= 1 {
        return 0.0;
    }

    let activations: Vec<f64> = ws.iter().map(|n| n.attrs.activation).collect();
    let total: f64 = activations.iter().sum();

    if total <= 0.0 {
        return 0.0;
    }

    // Normalized entropy (Shannon entropy / max entropy).
    let n = activations.len() as f64;
    let max_entropy = n.ln();
    if max_entropy <= 0.0 {
        return 0.0;
    }

    let entropy: f64 = activations
        .iter()
        .filter(|&&a| a > 0.0)
        .map(|&a| {
            let p = a / total;
            -p * p.ln()
        })
        .sum();

    (entropy / max_entropy).clamp(0.0, 1.0)
}

// ── §10: Internal — stale detection ─────────────────────────────────

/// Detect nodes with high activation but no recent updates.
fn detect_stale_nodes(ws: &WorkingSet, now: f64, config: &CoherenceConfig) -> Vec<StaleNode> {
    let now_ms = (now * 1000.0) as u64;

    ws.iter()
        .filter(|n| {
            n.attrs.activation >= config.active_threshold
                && now_ms.saturating_sub(n.attrs.last_updated_ms) as f64 / 1000.0
                    > config.stale_threshold_secs
        })
        .map(|n| StaleNode {
            node_id: n.id,
            label: n.label.clone(),
            kind: n.id.kind(),
            activation: n.attrs.activation,
            age_secs: now_ms.saturating_sub(n.attrs.last_updated_ms) as f64 / 1000.0,
        })
        .collect()
}

// ── §11: Internal — orphan detection ────────────────────────────────

/// Detect items with no supporting edges.
fn detect_orphans(ws: &WorkingSet, edges: &[CognitiveEdge]) -> Vec<OrphanedItem> {
    let connected: HashSet<u32> = edges
        .iter()
        .flat_map(|e| [e.src.to_raw(), e.dst.to_raw()])
        .collect();

    ws.iter()
        .filter(|n| {
            // Only check tasks, beliefs, and intents (goals can exist standalone).
            matches!(
                n.id.kind(),
                NodeKind::Task | NodeKind::Belief | NodeKind::IntentHypothesis
            ) && !connected.contains(&n.id.to_raw())
        })
        .map(|n| {
            let reason = match n.id.kind() {
                NodeKind::Task => "Task has no associated goal or prerequisite edges",
                NodeKind::Belief => "Belief has no supporting or contradicting evidence edges",
                NodeKind::IntentHypothesis => "Intent hypothesis has no supporting context",
                _ => "No connected edges",
            };
            OrphanedItem {
                node_id: n.id,
                label: n.label.clone(),
                kind: n.id.kind(),
                reason: reason.to_string(),
            }
        })
        .collect()
}

// ── §12: Internal — cycle detection ─────────────────────────────────

/// Detect circular dependencies in the goal/task graph.
///
/// Uses DFS-based cycle detection on Requires and SubtaskOf edges.
fn detect_cycles(ws: &WorkingSet, edges: &[CognitiveEdge]) -> Vec<DependencyCycle> {
    // Build adjacency for dependency edges only.
    let mut adj: HashMap<u32, Vec<(u32, f64)>> = HashMap::new();
    for e in edges {
        if matches!(
            e.kind,
            CognitiveEdgeKind::Requires | CognitiveEdgeKind::SubtaskOf
        ) {
            adj.entry(e.src.to_raw())
                .or_default()
                .push((e.dst.to_raw(), e.weight));
        }
    }

    let mut visited = HashSet::new();
    let mut in_stack = HashSet::new();
    let mut cycles = Vec::new();
    let mut path = Vec::new();

    let node_ids: Vec<u32> = ws.iter().map(|n| n.id.to_raw()).collect();

    for &start in &node_ids {
        if !visited.contains(&start) {
            dfs_cycle(
                start,
                &adj,
                &mut visited,
                &mut in_stack,
                &mut path,
                &mut cycles,
            );
        }
    }

    cycles
}

/// DFS helper for cycle detection.
fn dfs_cycle(
    node: u32,
    adj: &HashMap<u32, Vec<(u32, f64)>>,
    visited: &mut HashSet<u32>,
    in_stack: &mut HashSet<u32>,
    path: &mut Vec<(u32, f64)>,
    cycles: &mut Vec<DependencyCycle>,
) {
    visited.insert(node);
    in_stack.insert(node);
    path.push((node, 0.0));

    if let Some(neighbors) = adj.get(&node) {
        for &(next, weight) in neighbors {
            if !visited.contains(&next) {
                path.last_mut().unwrap().1 = weight;
                dfs_cycle(next, adj, visited, in_stack, path, cycles);
            } else if in_stack.contains(&next) {
                // Found a cycle — extract it.
                let cycle_start = path.iter().position(|&(n, _)| n == next);
                if let Some(start_idx) = cycle_start {
                    let cycle_nodes: Vec<NodeId> = path[start_idx..]
                        .iter()
                        .map(|&(n, _)| NodeId::from_raw(n))
                        .collect();
                    let weakest = path[start_idx..]
                        .iter()
                        .map(|&(_, w)| w)
                        .fold(f64::INFINITY, f64::min);

                    if cycle_nodes.len() >= 2 {
                        cycles.push(DependencyCycle {
                            nodes: cycle_nodes,
                            weakest_edge_weight: if weakest.is_infinite() { 0.0 } else { weakest },
                        });
                    }
                }
            }
        }
    }

    path.pop();
    in_stack.remove(&node);
}

// ── §13: Internal — deadline pressure ───────────────────────────────

/// Detect goals with deadline pressure.
fn detect_deadline_pressure(ws: &WorkingSet, now: f64) -> Vec<DeadlineAlert> {
    ws.nodes_of_kind(NodeKind::Goal)
        .into_iter()
        .filter_map(|n| {
            if let NodePayload::Goal(g) = &n.payload {
                if g.status != GoalStatus::Active {
                    return None;
                }
                let deadline = g.deadline?;
                let remaining = deadline - now;
                // Alert if < 24h remaining or overdue.
                if remaining < 86400.0 {
                    Some(DeadlineAlert {
                        goal_id: n.id,
                        description: g.description.clone(),
                        deadline,
                        remaining_secs: remaining,
                        progress: g.progress,
                    })
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect()
}

// ── §14: Internal — coherence scoring ───────────────────────────────

/// Compute the composite coherence score.
fn compute_coherence_score(
    goal_conflicts: &[GoalConflict],
    belief_contradictions: &[BeliefContradiction],
    fragmentation: f64,
    stale_nodes: &[StaleNode],
    cognitive_load: f64,
    total_active: f64,
    config: &CoherenceConfig,
) -> f64 {
    let conflict_penalty = config.w_conflict * goal_conflicts.len() as f64;
    let contradiction_penalty = config.w_contradiction * belief_contradictions.len() as f64;
    let fragmentation_penalty = config.w_fragmentation * fragmentation;
    let zombie_penalty = if total_active > 0.0 {
        config.w_zombie * (stale_nodes.len() as f64 / total_active)
    } else {
        0.0
    };
    let overload_penalty = config.w_overload * (cognitive_load - config.load_target).max(0.0);

    let score = 1.0
        - conflict_penalty
        - contradiction_penalty
        - fragmentation_penalty
        - zombie_penalty
        - overload_penalty;

    score.clamp(0.0, 1.0)
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Human-readable name for a node kind.
fn kind_name(kind: NodeKind) -> &'static str {
    match kind {
        NodeKind::Entity => "entity",
        NodeKind::Episode => "episode",
        NodeKind::Belief => "belief",
        NodeKind::Goal => "goal",
        NodeKind::Task => "task",
        NodeKind::IntentHypothesis => "intent",
        NodeKind::Routine => "routine",
        NodeKind::Need => "need",
        NodeKind::Opportunity => "opportunity",
        NodeKind::Risk => "risk",
        NodeKind::Constraint => "constraint",
        NodeKind::Preference => "preference",
        NodeKind::ConversationThread => "thread",
        NodeKind::ActionSchema => "schema",
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::{AttentionConfig, WorkingSet};
    use crate::state::{
        BeliefPayload, CognitiveAttrs, CognitiveEdge, CognitiveEdgeKind, CognitiveNode,
        GoalPayload, GoalStatus, NodeId, NodeKind, NodePayload, Priority, Provenance, TaskPayload,
        TaskStatus,
    };

    fn default_attrs(activation: f64, last_updated_ms: u64) -> CognitiveAttrs {
        CognitiveAttrs {
            confidence: 0.7,
            activation,
            salience: 0.5,
            persistence: 0.7,
            valence: 0.3,
            urgency: 0.5,
            novelty: 0.2,
            last_updated_ms,
            volatility: 0.1,
            provenance: Provenance::Inferred,
            evidence_count: 5,
        }
    }

    fn make_ws() -> WorkingSet {
        WorkingSet::with_config(AttentionConfig {
            capacity: 20,
            max_hops: 2,
            top_k_per_hop: 5,
            hop_decay: 0.5,
            activation_threshold: 0.1,
            lateral_inhibition: 0.3,
            insertion_boost: 0.1,
        })
    }

    fn goal_node(
        id: u32,
        desc: &str,
        status: GoalStatus,
        activation: f64,
        ts_ms: u64,
    ) -> CognitiveNode {
        CognitiveNode {
            id: NodeId::new(NodeKind::Goal, id),
            attrs: default_attrs(activation, ts_ms),
            payload: NodePayload::Goal(GoalPayload {
                description: desc.to_string(),
                status,
                progress: 0.0,
                deadline: None,
                priority: Priority::Medium,
                parent_goal: None,
                completion_criteria: "Done".to_string(),
            }),
            label: desc.to_string(),
            metadata: Default::default(),
        }
    }

    fn task_node(id: u32, desc: &str, goal_id: Option<NodeId>) -> CognitiveNode {
        CognitiveNode {
            id: NodeId::new(NodeKind::Task, id),
            attrs: default_attrs(0.5, 1_000_000),
            payload: NodePayload::Task(TaskPayload {
                description: desc.to_string(),
                status: TaskStatus::Pending,
                goal_id,
                deadline: None,
                priority: Priority::Medium,
                estimated_minutes: None,
                prerequisites: Vec::new(),
            }),
            label: desc.to_string(),
            metadata: Default::default(),
        }
    }

    fn belief_node(
        id: u32,
        label: &str,
        activation: f64,
        evidence: u32,
        ts_ms: u64,
    ) -> CognitiveNode {
        CognitiveNode {
            id: NodeId::new(NodeKind::Belief, id),
            attrs: {
                let mut a = default_attrs(activation, ts_ms);
                a.evidence_count = evidence;
                a
            },
            payload: NodePayload::Belief(BeliefPayload {
                proposition: label.to_string(),
                log_odds: 1.0,
                domain: "test".to_string(),
                evidence_trail: Vec::new(),
                user_confirmed: false,
            }),
            label: label.to_string(),
            metadata: Default::default(),
        }
    }

    #[test]
    fn test_empty_coherence() {
        let ws = make_ws();
        let config = CoherenceConfig::default();
        let inputs = CoherenceInputs {
            working_set: &ws,
            edges: &[],
            belief_conflicts: &[],
            config: &config,
            now: 1000.0,
        };

        let report = check_coherence(&inputs);
        assert_eq!(report.coherence_score, 1.0);
        assert!(report.goal_conflicts.is_empty());
        assert!(report.stale_activations.is_empty());
    }

    #[test]
    fn test_goal_conflict_detection() {
        let mut ws = make_ws();
        let g1 = goal_node(1, "Go fast", GoalStatus::Active, 0.8, 1_000_000);
        let g2 = goal_node(2, "Go slow", GoalStatus::Active, 0.7, 1_000_000);
        ws.insert(g1.clone());
        ws.insert(g2.clone());

        let edges = vec![CognitiveEdge {
            src: g1.id,
            dst: g2.id,
            kind: CognitiveEdgeKind::BlocksGoal,
            weight: 0.8,
            created_at_ms: 1000,
            last_confirmed_ms: 2000,
            observation_count: 5,
            confidence: 0.7,
        }];

        let config = CoherenceConfig::default();
        let inputs = CoherenceInputs {
            working_set: &ws,
            edges: &edges,
            belief_conflicts: &[],
            config: &config,
            now: 1000.0,
        };

        let report = check_coherence(&inputs);
        assert_eq!(report.goal_conflicts.len(), 1);
        assert!(report.coherence_score < 1.0);
    }

    #[test]
    fn test_stale_node_detection() {
        let mut ws = make_ws();
        let node = goal_node(1, "Old goal", GoalStatus::Active, 0.8, 1_000_000);
        let node_id = node.id;
        ws.insert(node);
        // Force old timestamp AFTER insert (insert calls touch() which resets it).
        ws.get_mut(node_id).unwrap().attrs.last_updated_ms = 1_000_000;

        let config = CoherenceConfig {
            stale_threshold_secs: 86400.0, // 1 day
            ..Default::default()
        };

        let now = 1000.0 + 86400.0 * 5.0; // 5 days later
        let inputs = CoherenceInputs {
            working_set: &ws,
            edges: &[],
            belief_conflicts: &[],
            config: &config,
            now,
        };

        let report = check_coherence(&inputs);
        assert_eq!(report.stale_activations.len(), 1);
        assert!(report.stale_activations[0].age_secs > 86400.0);
    }

    #[test]
    fn test_orphan_detection() {
        let mut ws = make_ws();
        // Task with no edges.
        let orphan_task = task_node(1, "Orphan task", None);
        ws.insert(orphan_task);

        let config = CoherenceConfig::default();
        let inputs = CoherenceInputs {
            working_set: &ws,
            edges: &[], // No edges!
            belief_conflicts: &[],
            config: &config,
            now: 1000.0,
        };

        let report = check_coherence(&inputs);
        assert_eq!(report.orphaned_items.len(), 1);
    }

    #[test]
    fn test_cognitive_load() {
        let mut ws = WorkingSet::with_config(AttentionConfig {
            capacity: 5, // Small capacity.
            max_hops: 2,
            top_k_per_hop: 5,
            hop_decay: 0.5,
            activation_threshold: 0.1,
            lateral_inhibition: 0.3,
            insertion_boost: 0.1,
        });

        // Add 4 nodes (80% load).
        for i in 0..4 {
            ws.insert(goal_node(
                i,
                &format!("Goal {}", i),
                GoalStatus::Active,
                0.5,
                1_000_000,
            ));
        }

        let config = CoherenceConfig::default();
        let inputs = CoherenceInputs {
            working_set: &ws,
            edges: &[],
            belief_conflicts: &[],
            config: &config,
            now: 1000.0,
        };

        let report = check_coherence(&inputs);
        assert!((report.cognitive_load - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_fragmentation() {
        let mut ws = make_ws();

        // All nodes with similar activation → high fragmentation.
        for i in 0..5 {
            ws.insert(goal_node(
                i,
                &format!("G{}", i),
                GoalStatus::Active,
                0.5,
                1_000_000,
            ));
        }

        let frag = compute_fragmentation(&ws);
        // Should be relatively high (near 1.0) since activations are equal.
        assert!(frag > 0.9);

        // Now with one dominant node.
        let mut ws2 = make_ws();
        ws2.insert(goal_node(
            10,
            "Dominant",
            GoalStatus::Active,
            0.9,
            1_000_000,
        ));
        ws2.insert(goal_node(11, "Minor1", GoalStatus::Active, 0.05, 1_000_000));
        ws2.insert(goal_node(12, "Minor2", GoalStatus::Active, 0.05, 1_000_000));

        let frag2 = compute_fragmentation(&ws2);
        // Should be lower (more focused).
        assert!(frag2 < frag);
    }

    #[test]
    fn test_enforcement_zombie_demotion() {
        let mut ws = make_ws();
        let old_ts = 1_000u64;
        let node_id = NodeId::new(NodeKind::Goal, 1);
        ws.insert(goal_node(1, "Zombie goal", GoalStatus::Active, 0.8, old_ts));
        // insert() calls touch() which resets last_updated_ms — restore old value.
        ws.get_mut(node_id).unwrap().attrs.last_updated_ms = old_ts;

        let config = CoherenceConfig {
            stale_threshold_secs: 1.0, // Very short for testing.
            ..Default::default()
        };

        let inputs = CoherenceInputs {
            working_set: &ws,
            edges: &[],
            belief_conflicts: &[],
            config: &config,
            now: 100.0,
        };

        let report = check_coherence(&inputs);
        let enforcement = plan_enforcement(&report, &ws, &config);

        assert!(enforcement.zombies_demoted > 0);
        assert!(enforcement
            .actions
            .iter()
            .any(|a| a.kind == EnforcementKind::DemoteZombie));
    }

    #[test]
    fn test_coherence_history() {
        let mut history = CoherenceHistory::new(5);

        for i in 0..7 {
            let report = CoherenceReport {
                goal_conflicts: Vec::new(),
                belief_contradictions: Vec::new(),
                attention_fragmentation: 0.0,
                stale_activations: Vec::new(),
                cognitive_load: 0.5,
                orphaned_items: Vec::new(),
                circular_dependencies: Vec::new(),
                deadline_pressure: Vec::new(),
                coherence_score: 0.8 + (i as f64 * 0.02),
                checked_at: 1000.0 + i as f64 * 60.0,
            };
            history.record(&report);
        }

        // Should only keep last 5.
        assert_eq!(history.snapshot_count(), 5);
        assert_eq!(history.total_checks, 7);

        // Average should be close to recent values.
        let avg = history.recent_average(3);
        assert!(avg > 0.8);

        // Trend should be positive (scores increasing).
        let trend = history.trend(5);
        assert!(trend > 0.0);
    }

    #[test]
    fn test_deadline_pressure() {
        let mut ws = make_ws();
        let mut node = goal_node(1, "Urgent goal", GoalStatus::Active, 0.8, 1_000_000);
        if let NodePayload::Goal(ref mut g) = node.payload {
            g.deadline = Some(1050.0); // 50 seconds from now.
        }
        ws.insert(node);

        let alerts = detect_deadline_pressure(&ws, 1000.0);
        assert_eq!(alerts.len(), 1);
        assert!((alerts[0].remaining_secs - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_contradiction_resolution() {
        let mut ws = make_ws();
        let b1 = belief_node(1, "Belief A", 0.8, 10, 1_000_000);
        let b2 = belief_node(2, "Belief B", 0.7, 3, 1_000_000);
        ws.insert(b1.clone());
        ws.insert(b2.clone());

        let contradictions = vec![BeliefContradiction {
            belief_a: b1.id,
            belief_b: b2.id,
            severity: 0.8,
            description: "A contradicts B".to_string(),
        }];

        let report = CoherenceReport {
            goal_conflicts: Vec::new(),
            belief_contradictions: contradictions,
            attention_fragmentation: 0.0,
            stale_activations: Vec::new(),
            cognitive_load: 0.5,
            orphaned_items: Vec::new(),
            circular_dependencies: Vec::new(),
            deadline_pressure: Vec::new(),
            coherence_score: 0.7,
            checked_at: 1000.0,
        };

        let config = CoherenceConfig::default();
        let enforcement = plan_enforcement(&report, &ws, &config);

        assert_eq!(enforcement.conflicts_resolved, 1);
        // Should demote b2 (fewer evidence: 3 vs 10).
        let resolve_action = enforcement
            .actions
            .iter()
            .find(|a| a.kind == EnforcementKind::ResolveContradiction);
        assert!(resolve_action.is_some());
        assert_eq!(resolve_action.unwrap().target, b2.id);
    }

    #[test]
    fn test_cycle_detection() {
        let mut ws = make_ws();
        let g1 = goal_node(1, "Goal A", GoalStatus::Active, 0.5, 1_000_000);
        let g2 = goal_node(2, "Goal B", GoalStatus::Active, 0.5, 1_000_000);
        ws.insert(g1.clone());
        ws.insert(g2.clone());

        // A requires B, B requires A → cycle.
        let edges = vec![
            CognitiveEdge {
                src: g1.id,
                dst: g2.id,
                kind: CognitiveEdgeKind::Requires,
                weight: 0.8,
                created_at_ms: 1000,
                last_confirmed_ms: 2000,
                observation_count: 3,
                confidence: 0.6,
            },
            CognitiveEdge {
                src: g2.id,
                dst: g1.id,
                kind: CognitiveEdgeKind::Requires,
                weight: 0.5,
                created_at_ms: 1000,
                last_confirmed_ms: 2000,
                observation_count: 2,
                confidence: 0.4,
            },
        ];

        let cycles = detect_cycles(&ws, &edges);
        assert!(!cycles.is_empty());
        assert!(cycles[0].nodes.len() >= 2);
    }

    #[test]
    fn test_emergency_trigger() {
        let config = CoherenceConfig {
            emergency_threshold: 0.5,
            ..Default::default()
        };

        let report = CoherenceReport {
            goal_conflicts: Vec::new(),
            belief_contradictions: Vec::new(),
            attention_fragmentation: 0.0,
            stale_activations: Vec::new(),
            cognitive_load: 0.5,
            orphaned_items: Vec::new(),
            circular_dependencies: Vec::new(),
            deadline_pressure: Vec::new(),
            coherence_score: 0.3, // Below emergency threshold.
            checked_at: 1000.0,
        };

        let ws = make_ws();
        let enforcement = plan_enforcement(&report, &ws, &config);
        assert!(enforcement.emergency_triggered);
    }
}
