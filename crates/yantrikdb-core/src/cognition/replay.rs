//! CK-5.6 — Experience Replay / Dream Consolidation.
//!
//! Prioritized experience replay that actively refines memories during
//! idle time. The system "dreams" — replaying surprising episodes to
//! update beliefs, refine schemas, and strengthen causal models.
//!
//! # Design principles
//! - Pure functions only — no DB access (engine layer handles persistence)
//! - Prioritized replay (Schaul et al., 2015) — surprising episodes first
//! - TD-error driven: large prediction errors → more replay priority
//! - Cross-domain association discovery during replay
//! - Budget-controlled to avoid consuming too many resources

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::state::NodeId;

// ══════════════════════════════════════════════════════════════════════════════
// § 1  Core Types
// ══════════════════════════════════════════════════════════════════════════════

/// Sampling strategy for the replay buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Uniform random sampling.
    Uniform,
    /// Prioritize episodes with high TD error (most surprising).
    PrioritizedByTDError,
    /// Prioritize recent episodes.
    PrioritizedByRecency,
    /// Prioritize episodes with high surprise × recency.
    PrioritizedBySurprise,
}

/// What action was taken in an episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRecord {
    /// Description of the action.
    pub description: String,
    /// Domain/category of the action.
    pub domain: String,
    /// Nodes involved in the action.
    pub involved_nodes: Vec<NodeId>,
}

/// Outcome data from an episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeData {
    /// Observed utility of the outcome ∈ [-1.0, 1.0].
    pub utility: f64,
    /// Whether the outcome matched expectations.
    pub expected: bool,
    /// Domain tags for cross-referencing.
    pub domains: Vec<String>,
    /// Nodes affected by the outcome.
    pub affected_nodes: Vec<NodeId>,
}

/// A single episode queued for replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayEntry {
    /// The Episode node ID in the cognitive graph.
    pub episode_id: NodeId,
    /// Expected utility at decision time.
    pub expected_utility: f64,
    /// Action taken.
    pub action: ActionRecord,
    /// What actually happened.
    pub outcome: OutcomeData,
    /// Temporal difference error: |expected - actual| utility.
    pub td_error: f64,
    /// How many times this episode has been replayed.
    pub replay_count: u32,
    /// When this episode was last replayed (unix ms).
    pub last_replayed_at: u64,
    /// When the episode originally occurred (unix ms).
    pub created_at: u64,
    /// Computed priority for sampling (higher = more likely to be replayed).
    pub priority: f64,
}

/// A change to a belief discovered during replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefDelta {
    /// The belief node.
    pub belief_id: NodeId,
    /// Change in confidence.
    pub confidence_delta: f64,
    /// Reason for the change.
    pub reason: String,
}

/// A change to a causal edge discovered during replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalDelta {
    /// Cause node.
    pub cause: NodeId,
    /// Effect node.
    pub effect: NodeId,
    /// Change in causal strength.
    pub strength_delta: f64,
}

/// Result of replaying a single episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayOutcome {
    /// Which episode was replayed.
    pub episode_id: NodeId,
    /// Updated TD error after re-evaluation.
    pub new_td_error: f64,
    /// Belief changes discovered.
    pub belief_updates: Vec<BeliefDelta>,
    /// Causal edges to strengthen or weaken.
    pub causal_updates: Vec<CausalDelta>,
    /// Cross-domain associations found: (node_a, node_b, similarity).
    pub associations: Vec<(NodeId, NodeId, f64)>,
    /// Insights generated.
    pub insights: Vec<String>,
}

/// Summary of a complete replay cycle ("dream report").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamReport {
    /// How many episodes were replayed.
    pub replays_executed: usize,
    /// How many beliefs were updated.
    pub beliefs_updated: usize,
    /// How many causal edges were modified.
    pub causal_updates: usize,
    /// New cross-domain associations discovered.
    pub new_associations: usize,
    /// Notable insights from the dream cycle.
    pub insights: Vec<String>,
    /// Total time spent (ms).
    pub duration_ms: u64,
    /// Average TD error of replayed episodes.
    pub avg_td_error: f64,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 2  Replay Budget
// ══════════════════════════════════════════════════════════════════════════════

/// Controls how much replay happens per cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBudget {
    /// Maximum episodes to replay per cycle.
    pub max_replays_per_cycle: usize,
    /// Minimum idle time before replaying (ms).
    pub min_idle_ms: u64,
    /// Maximum age of replayable episodes (ms). Default: 30 days.
    pub max_replay_age_ms: u64,
    /// Priority exponent: higher = more skewed toward high-TD-error episodes.
    pub priority_exponent: f64,
    /// Minimum TD error to be eligible for replay.
    pub min_td_error: f64,
    /// Maximum times a single episode can be replayed.
    pub max_replays_per_episode: u32,
}

impl Default for ReplayBudget {
    fn default() -> Self {
        Self {
            max_replays_per_cycle: 10,
            min_idle_ms: 30_000,
            max_replay_age_ms: 30 * 24 * 3600 * 1000, // 30 days
            priority_exponent: 0.6,
            min_td_error: 0.05,
            max_replays_per_episode: 5,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  Replay Buffer
// ══════════════════════════════════════════════════════════════════════════════

/// Prioritized collection of replayable episodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBuffer {
    /// Episodes eligible for replay.
    pub entries: Vec<ReplayEntry>,
    /// Maximum buffer size.
    pub capacity: usize,
    /// How to sample episodes.
    pub strategy: SamplingStrategy,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, strategy: SamplingStrategy) -> Self {
        Self {
            entries: Vec::new(),
            capacity,
            strategy,
        }
    }

    /// Number of entries in the buffer.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  Replay Engine
// ══════════════════════════════════════════════════════════════════════════════

/// Replay statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReplayStats {
    /// Total replay cycles executed.
    pub total_cycles: u64,
    /// Total episodes replayed across all cycles.
    pub total_replays: u64,
    /// Total belief updates from replay.
    pub total_belief_updates: u64,
    /// Total causal updates from replay.
    pub total_causal_updates: u64,
    /// Total cross-domain associations discovered.
    pub total_associations: u64,
}

/// The replay engine: manages the experience replay process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayEngine {
    /// The prioritized replay buffer.
    pub buffer: ReplayBuffer,
    /// Budget configuration.
    pub budget: ReplayBudget,
    /// When the last replay cycle occurred (unix ms).
    pub last_replay_at: u64,
    /// Cumulative statistics.
    pub stats: ReplayStats,
}

impl ReplayEngine {
    /// Create a new replay engine with default settings.
    pub fn new() -> Self {
        Self {
            buffer: ReplayBuffer::new(500, SamplingStrategy::PrioritizedByTDError),
            budget: ReplayBudget::default(),
            last_replay_at: 0,
            stats: ReplayStats::default(),
        }
    }

    /// Create with custom budget and capacity.
    pub fn with_config(capacity: usize, strategy: SamplingStrategy, budget: ReplayBudget) -> Self {
        Self {
            buffer: ReplayBuffer::new(capacity, strategy),
            budget,
            last_replay_at: 0,
            stats: ReplayStats::default(),
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 5  Core Operations
// ══════════════════════════════════════════════════════════════════════════════

/// Compute temporal difference error.
pub fn compute_td_error(expected_utility: f64, actual_utility: f64) -> f64 {
    (actual_utility - expected_utility).abs()
}

/// Add a new episode to the replay buffer.
///
/// If the buffer is at capacity, evicts the lowest-priority entry.
pub fn add_to_buffer(
    engine: &mut ReplayEngine,
    episode_id: NodeId,
    expected_utility: f64,
    action: ActionRecord,
    outcome: OutcomeData,
    now_ms: u64,
) {
    let td_error = compute_td_error(expected_utility, outcome.utility);

    // Only add if TD error exceeds minimum threshold.
    if td_error < engine.budget.min_td_error {
        return;
    }

    let priority = compute_priority(
        td_error,
        now_ms,
        now_ms, // just created
        0,      // never replayed
        engine.budget.priority_exponent,
    );

    let entry = ReplayEntry {
        episode_id,
        expected_utility,
        action,
        outcome,
        td_error,
        replay_count: 0,
        last_replayed_at: 0,
        created_at: now_ms,
        priority,
    };

    if engine.buffer.entries.len() >= engine.buffer.capacity {
        // Find and remove lowest-priority entry.
        if let Some(min_idx) = engine
            .buffer
            .entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.priority
                    .partial_cmp(&b.priority)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
        {
            if engine.buffer.entries[min_idx].priority < priority {
                engine.buffer.entries.swap_remove(min_idx);
            } else {
                return; // new entry is lower priority than everything in buffer
            }
        }
    }

    engine.buffer.entries.push(entry);
}

/// Compute priority for a replay entry.
fn compute_priority(
    td_error: f64,
    now_ms: u64,
    created_at: u64,
    replay_count: u32,
    exponent: f64,
) -> f64 {
    // Priority = td_error^exponent × recency_factor × novelty_factor
    let td_component = td_error.abs().powf(exponent);

    // Recency: newer episodes get higher priority.
    let age_hours = (now_ms.saturating_sub(created_at)) as f64 / 3_600_000.0;
    let recency = 1.0 / (1.0 + age_hours / 24.0); // half-life of 24 hours

    // Novelty: less-replayed episodes get higher priority.
    let novelty = 1.0 / (1.0 + replay_count as f64);

    td_component * recency * novelty
}

/// Check whether the engine should run a replay cycle.
pub fn should_replay(engine: &ReplayEngine, now_ms: u64) -> bool {
    if engine.buffer.is_empty() {
        return false;
    }

    let time_since_last = now_ms.saturating_sub(engine.last_replay_at);
    time_since_last >= engine.budget.min_idle_ms
}

/// Recompute priorities for all entries in the buffer.
pub fn reprioritize_buffer(engine: &mut ReplayEngine, now_ms: u64) {
    for entry in &mut engine.buffer.entries {
        entry.priority = compute_priority(
            entry.td_error,
            now_ms,
            entry.created_at,
            entry.replay_count,
            engine.budget.priority_exponent,
        );
    }

    // Sort by priority descending.
    engine.buffer.entries.sort_by(|a, b| {
        b.priority
            .partial_cmp(&a.priority)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Sample episodes from the buffer for replay.
fn sample_episodes(engine: &ReplayEngine, count: usize) -> Vec<usize> {
    let n = engine.buffer.entries.len().min(count);
    let budget = &engine.budget;

    match engine.buffer.strategy {
        SamplingStrategy::Uniform => {
            // Simple: take first N eligible entries.
            (0..engine.buffer.entries.len())
                .filter(|&i| {
                    let e = &engine.buffer.entries[i];
                    e.replay_count < budget.max_replays_per_episode
                })
                .take(n)
                .collect()
        }
        SamplingStrategy::PrioritizedByTDError | SamplingStrategy::PrioritizedBySurprise => {
            // Already sorted by priority — take top N eligible.
            let mut indices = Vec::new();
            for (i, entry) in engine.buffer.entries.iter().enumerate() {
                if entry.replay_count < budget.max_replays_per_episode {
                    indices.push(i);
                    if indices.len() >= n {
                        break;
                    }
                }
            }
            indices
        }
        SamplingStrategy::PrioritizedByRecency => {
            // Sort by created_at descending, take top N.
            let mut sorted_indices: Vec<usize> = (0..engine.buffer.entries.len())
                .filter(|&i| engine.buffer.entries[i].replay_count < budget.max_replays_per_episode)
                .collect();
            sorted_indices.sort_by(|&a, &b| {
                engine.buffer.entries[b]
                    .created_at
                    .cmp(&engine.buffer.entries[a].created_at)
            });
            sorted_indices.into_iter().take(n).collect()
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 6  Re-evaluation & Replay
// ══════════════════════════════════════════════════════════════════════════════

/// Re-evaluate an episode with current knowledge.
///
/// Compares the original expected utility with what we now know
/// about the outcome and related beliefs.
pub fn re_evaluate_episode(
    entry: &ReplayEntry,
    current_belief_confidences: &[(NodeId, f64)],
) -> ReplayOutcome {
    let mut belief_updates = Vec::new();
    let mut insights = Vec::new();

    // Check how beliefs about involved nodes have changed.
    let belief_map: HashMap<NodeId, f64> = current_belief_confidences.iter().cloned().collect();

    for &affected in &entry.outcome.affected_nodes {
        if let Some(&current_conf) = belief_map.get(&affected) {
            // If the outcome was surprising and this belief is still uncertain,
            // suggest updating it.
            let should_increase = entry.outcome.utility > entry.expected_utility;
            let delta = if should_increase { 0.1 } else { -0.1 };

            if (current_conf < 0.8 && should_increase) || (current_conf > 0.2 && !should_increase) {
                belief_updates.push(BeliefDelta {
                    belief_id: affected,
                    confidence_delta: delta * entry.td_error,
                    reason: format!(
                        "Replay of episode {:?}: outcome was {} than expected (TD error {:.2})",
                        entry.episode_id,
                        if should_increase { "better" } else { "worse" },
                        entry.td_error
                    ),
                });
            }
        }
    }

    // Generate causal updates based on action → outcome relationship.
    let mut causal_updates = Vec::new();
    for &involved in &entry.action.involved_nodes {
        for &affected in &entry.outcome.affected_nodes {
            if involved != affected {
                let strength_delta = if entry.outcome.expected {
                    0.05 // confirmed: strengthen
                } else {
                    -0.05 // surprised: weaken
                };
                causal_updates.push(CausalDelta {
                    cause: involved,
                    effect: affected,
                    strength_delta,
                });
            }
        }
    }

    // Generate insight if TD error is high.
    if entry.td_error > 0.3 {
        let direction = if entry.outcome.utility > entry.expected_utility {
            "positively"
        } else {
            "negatively"
        };
        insights.push(format!(
            "Episode {:?} {} surprised us: expected utility {:.2}, got {:.2}. Domain: {}.",
            entry.episode_id,
            direction,
            entry.expected_utility,
            entry.outcome.utility,
            entry.action.domain
        ));
    }

    // Compute updated TD error (may be lower now with more knowledge).
    let new_td_error = entry.td_error * 0.9; // slightly decay on each replay

    ReplayOutcome {
        episode_id: entry.episode_id,
        new_td_error,
        belief_updates,
        causal_updates,
        associations: Vec::new(), // computed at cycle level
        insights,
    }
}

/// Discover cross-domain associations among replayed episodes.
///
/// Finds episodes from different domains that share structural patterns
/// (same affected nodes, similar TD errors, related outcomes).
pub fn discover_cross_associations(
    outcomes: &[ReplayOutcome],
    entries: &[&ReplayEntry],
) -> Vec<(NodeId, NodeId, f64)> {
    let mut associations = Vec::new();

    for i in 0..entries.len() {
        for j in (i + 1)..entries.len() {
            let a = &entries[i];
            let b = &entries[j];

            // Skip same-domain pairs.
            if a.action.domain == b.action.domain {
                continue;
            }

            // Check for shared affected nodes.
            let shared_nodes: Vec<NodeId> = a
                .outcome
                .affected_nodes
                .iter()
                .filter(|n| b.outcome.affected_nodes.contains(n))
                .cloned()
                .collect();

            if !shared_nodes.is_empty() {
                // Similarity based on shared structure and TD error proximity.
                let td_similarity = 1.0 - (a.td_error - b.td_error).abs().min(1.0);
                let node_overlap = shared_nodes.len() as f64
                    / (a.outcome.affected_nodes.len() + b.outcome.affected_nodes.len()) as f64;
                let similarity = 0.5 * td_similarity + 0.5 * node_overlap;

                if similarity > 0.2 {
                    for node in &shared_nodes {
                        associations.push((a.episode_id, b.episode_id, similarity));
                    }
                }
            }
        }
    }

    // Deduplicate.
    associations.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    associations.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

    associations
}

// ══════════════════════════════════════════════════════════════════════════════
// § 7  Replay Cycle
// ══════════════════════════════════════════════════════════════════════════════

/// Run a complete replay cycle ("dream session").
///
/// Samples episodes from the buffer, re-evaluates each with current
/// knowledge, discovers cross-domain associations, and produces a
/// dream report summarizing all findings.
pub fn run_replay_cycle(
    engine: &mut ReplayEngine,
    current_beliefs: &[(NodeId, f64)],
    now_ms: u64,
) -> DreamReport {
    let start_ms = now_ms;

    // Reprioritize before sampling.
    reprioritize_buffer(engine, now_ms);

    // Sample episodes.
    let sample_indices = sample_episodes(engine, engine.budget.max_replays_per_cycle);

    if sample_indices.is_empty() {
        return DreamReport {
            replays_executed: 0,
            beliefs_updated: 0,
            causal_updates: 0,
            new_associations: 0,
            insights: Vec::new(),
            duration_ms: 0,
            avg_td_error: 0.0,
        };
    }

    // Collect entries for replay (clone to avoid borrow issues).
    let entries: Vec<ReplayEntry> = sample_indices
        .iter()
        .map(|&i| engine.buffer.entries[i].clone())
        .collect();

    // Re-evaluate each episode.
    let mut outcomes = Vec::new();
    let mut total_td_error = 0.0;

    for entry in &entries {
        let outcome = re_evaluate_episode(entry, current_beliefs);
        total_td_error += outcome.new_td_error;
        outcomes.push(outcome);
    }

    // Discover cross-domain associations.
    let entry_refs: Vec<&ReplayEntry> = entries.iter().collect();
    let associations = discover_cross_associations(&outcomes, &entry_refs);

    // Aggregate results.
    let total_belief_updates: usize = outcomes.iter().map(|o| o.belief_updates.len()).sum();
    let total_causal_updates: usize = outcomes.iter().map(|o| o.causal_updates.len()).sum();
    let all_insights: Vec<String> = outcomes.iter().flat_map(|o| o.insights.clone()).collect();

    // Update entries in buffer (new TD error, replay count).
    for (idx_pos, &buf_idx) in sample_indices.iter().enumerate() {
        if buf_idx < engine.buffer.entries.len() && idx_pos < outcomes.len() {
            engine.buffer.entries[buf_idx].td_error = outcomes[idx_pos].new_td_error;
            engine.buffer.entries[buf_idx].replay_count += 1;
            engine.buffer.entries[buf_idx].last_replayed_at = now_ms;
        }
    }

    // Update engine state.
    engine.last_replay_at = now_ms;
    engine.stats.total_cycles += 1;
    engine.stats.total_replays += entries.len() as u64;
    engine.stats.total_belief_updates += total_belief_updates as u64;
    engine.stats.total_causal_updates += total_causal_updates as u64;
    engine.stats.total_associations += associations.len() as u64;

    let replays_executed = entries.len();
    let avg_td_error = if replays_executed > 0 {
        total_td_error / replays_executed as f64
    } else {
        0.0
    };

    DreamReport {
        replays_executed,
        beliefs_updated: total_belief_updates,
        causal_updates: total_causal_updates,
        new_associations: associations.len(),
        insights: all_insights,
        duration_ms: now_ms.saturating_sub(start_ms),
        avg_td_error,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 8  Buffer Maintenance
// ══════════════════════════════════════════════════════════════════════════════

/// Remove expired and fully-replayed entries from the buffer.
///
/// Returns the number of entries removed.
pub fn buffer_maintenance(engine: &mut ReplayEngine, now_ms: u64) -> usize {
    let max_age = engine.budget.max_replay_age_ms;
    let max_replays = engine.budget.max_replays_per_episode;
    let initial_len = engine.buffer.entries.len();

    engine.buffer.entries.retain(|entry| {
        let age = now_ms.saturating_sub(entry.created_at);
        let not_expired = age < max_age;
        let not_exhausted = entry.replay_count < max_replays;
        let still_surprising = entry.td_error > 0.01;
        not_expired && not_exhausted && still_surprising
    });

    // Reprioritize remaining entries.
    reprioritize_buffer(engine, now_ms);

    // Trim to capacity.
    if engine.buffer.entries.len() > engine.buffer.capacity {
        engine.buffer.entries.truncate(engine.buffer.capacity);
    }

    initial_len - engine.buffer.entries.len()
}

/// Get a summary of the replay engine state.
pub fn replay_summary(engine: &ReplayEngine) -> ReplaySummary {
    let avg_td = if engine.buffer.entries.is_empty() {
        0.0
    } else {
        engine
            .buffer
            .entries
            .iter()
            .map(|e| e.td_error)
            .sum::<f64>()
            / engine.buffer.entries.len() as f64
    };

    let max_td = engine
        .buffer
        .entries
        .iter()
        .map(|e| e.td_error)
        .fold(0.0f64, f64::max);

    // Domain distribution.
    let mut domain_counts: HashMap<String, usize> = HashMap::new();
    for entry in &engine.buffer.entries {
        *domain_counts
            .entry(entry.action.domain.clone())
            .or_insert(0) += 1;
    }

    ReplaySummary {
        buffer_size: engine.buffer.entries.len(),
        buffer_capacity: engine.buffer.capacity,
        avg_td_error: avg_td,
        max_td_error: max_td,
        total_cycles: engine.stats.total_cycles,
        total_replays: engine.stats.total_replays,
        domain_distribution: domain_counts,
    }
}

/// Summary of the replay engine state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaySummary {
    pub buffer_size: usize,
    pub buffer_capacity: usize,
    pub avg_td_error: f64,
    pub max_td_error: f64,
    pub total_cycles: u64,
    pub total_replays: u64,
    pub domain_distribution: HashMap<String, usize>,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 9  Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{NodeId, NodeKind};

    fn episode(seq: u32) -> NodeId {
        NodeId::new(NodeKind::Episode, seq)
    }

    fn entity(seq: u32) -> NodeId {
        NodeId::new(NodeKind::Entity, seq)
    }

    fn belief(seq: u32) -> NodeId {
        NodeId::new(NodeKind::Belief, seq)
    }

    fn make_action(domain: &str, nodes: &[NodeId]) -> ActionRecord {
        ActionRecord {
            description: format!("test action in {}", domain),
            domain: domain.to_string(),
            involved_nodes: nodes.to_vec(),
        }
    }

    fn make_outcome(
        utility: f64,
        expected: bool,
        domains: &[&str],
        nodes: &[NodeId],
    ) -> OutcomeData {
        OutcomeData {
            utility,
            expected,
            domains: domains.iter().map(|s| s.to_string()).collect(),
            affected_nodes: nodes.to_vec(),
        }
    }

    // ── TD error tests ──────────────────────────────────────────────────

    #[test]
    fn test_td_error_computation() {
        assert!((compute_td_error(0.5, 0.8) - 0.3).abs() < 0.01);
        assert!((compute_td_error(0.8, 0.2) - 0.6).abs() < 0.01);
        assert!((compute_td_error(0.5, 0.5) - 0.0).abs() < 0.01);
    }

    // ── Buffer operations ───────────────────────────────────────────────

    #[test]
    fn test_add_to_buffer() {
        let mut engine = ReplayEngine::new();

        add_to_buffer(
            &mut engine,
            episode(1),
            0.5,
            make_action("work", &[entity(1)]),
            make_outcome(0.9, false, &["work"], &[belief(1)]),
            1000,
        );

        assert_eq!(engine.buffer.len(), 1);
        assert!(engine.buffer.entries[0].td_error > 0.0);
    }

    #[test]
    fn test_add_below_threshold_ignored() {
        let mut engine = ReplayEngine::new();

        // TD error = |0.5 - 0.51| = 0.01 < min_td_error (0.05)
        add_to_buffer(
            &mut engine,
            episode(1),
            0.50,
            make_action("work", &[entity(1)]),
            make_outcome(0.51, true, &["work"], &[belief(1)]),
            1000,
        );

        assert!(engine.buffer.is_empty());
    }

    #[test]
    fn test_buffer_capacity_eviction() {
        let mut engine = ReplayEngine::with_config(
            3, // capacity = 3
            SamplingStrategy::PrioritizedByTDError,
            ReplayBudget::default(),
        );

        // Add 4 entries — 4th should evict the lowest priority.
        for i in 0..4 {
            let td = 0.1 + i as f64 * 0.2; // increasing TD error
            add_to_buffer(
                &mut engine,
                episode(i),
                0.5,
                make_action("work", &[entity(i)]),
                make_outcome(0.5 + td, false, &["work"], &[belief(i)]),
                1000 + i as u64 * 1000,
            );
        }

        assert_eq!(engine.buffer.len(), 3);
    }

    // ── Priority computation ────────────────────────────────────────────

    #[test]
    fn test_priority_td_error_dominates() {
        let high_td = compute_priority(0.8, 1000, 900, 0, 0.6);
        let low_td = compute_priority(0.1, 1000, 900, 0, 0.6);
        assert!(high_td > low_td);
    }

    #[test]
    fn test_priority_recency_matters() {
        let recent = compute_priority(0.5, 10000, 9000, 0, 0.6);
        let old = compute_priority(0.5, 10000, 1000, 0, 0.6);
        assert!(recent > old);
    }

    #[test]
    fn test_priority_novelty_matters() {
        let fresh = compute_priority(0.5, 1000, 900, 0, 0.6);
        let stale = compute_priority(0.5, 1000, 900, 5, 0.6);
        assert!(fresh > stale);
    }

    // ── Should replay ───────────────────────────────────────────────────

    #[test]
    fn test_should_replay_empty_buffer() {
        let engine = ReplayEngine::new();
        assert!(!should_replay(&engine, 100000));
    }

    #[test]
    fn test_should_replay_after_idle() {
        let mut engine = ReplayEngine::new();
        add_to_buffer(
            &mut engine,
            episode(1),
            0.5,
            make_action("work", &[entity(1)]),
            make_outcome(0.9, false, &["work"], &[belief(1)]),
            1000,
        );
        engine.last_replay_at = 1000;

        // Not enough idle time.
        assert!(!should_replay(&engine, 2000));

        // Enough idle time.
        assert!(should_replay(&engine, 50000));
    }

    // ── Re-evaluation ───────────────────────────────────────────────────

    #[test]
    fn test_re_evaluate_surprising_episode() {
        let entry = ReplayEntry {
            episode_id: episode(1),
            expected_utility: 0.3,
            action: make_action("work", &[entity(1)]),
            outcome: make_outcome(0.8, false, &["work"], &[belief(1), belief(2)]),
            td_error: 0.5,
            replay_count: 0,
            last_replayed_at: 0,
            created_at: 1000,
            priority: 1.0,
        };

        let beliefs = vec![(belief(1), 0.5), (belief(2), 0.6)];
        let outcome = re_evaluate_episode(&entry, &beliefs);

        assert!(!outcome.belief_updates.is_empty());
        assert!(!outcome.insights.is_empty()); // TD error > 0.3
        assert!(outcome.new_td_error < entry.td_error); // decayed
    }

    #[test]
    fn test_re_evaluate_expected_episode() {
        let entry = ReplayEntry {
            episode_id: episode(1),
            expected_utility: 0.5,
            action: make_action("work", &[entity(1)]),
            outcome: make_outcome(0.55, true, &["work"], &[belief(1)]),
            td_error: 0.05,
            replay_count: 0,
            last_replayed_at: 0,
            created_at: 1000,
            priority: 0.5,
        };

        let beliefs = vec![(belief(1), 0.7)];
        let outcome = re_evaluate_episode(&entry, &beliefs);

        // Low TD error → no insights.
        assert!(outcome.insights.is_empty());
    }

    // ── Cross-domain associations ───────────────────────────────────────

    #[test]
    fn test_cross_domain_association_discovery() {
        let shared_node = belief(99);

        let entries = vec![
            ReplayEntry {
                episode_id: episode(1),
                expected_utility: 0.5,
                action: make_action("work", &[entity(1)]),
                outcome: make_outcome(0.8, false, &["work"], &[shared_node, belief(1)]),
                td_error: 0.4,
                replay_count: 0,
                last_replayed_at: 0,
                created_at: 1000,
                priority: 1.0,
            },
            ReplayEntry {
                episode_id: episode(2),
                expected_utility: 0.6,
                action: make_action("health", &[entity(2)]),
                outcome: make_outcome(0.9, false, &["health"], &[shared_node, belief(2)]),
                td_error: 0.35,
                replay_count: 0,
                last_replayed_at: 0,
                created_at: 2000,
                priority: 0.9,
            },
        ];

        let outcomes: Vec<ReplayOutcome> = entries
            .iter()
            .map(|e| re_evaluate_episode(e, &[]))
            .collect();

        let entry_refs: Vec<&ReplayEntry> = entries.iter().collect();
        let assocs = discover_cross_associations(&outcomes, &entry_refs);

        // Should find association through shared_node.
        assert!(!assocs.is_empty());
    }

    // ── Replay cycle ────────────────────────────────────────────────────

    #[test]
    fn test_full_replay_cycle() {
        let mut engine = ReplayEngine::new();

        // Add several episodes.
        for i in 0..5 {
            add_to_buffer(
                &mut engine,
                episode(i),
                0.5,
                make_action("work", &[entity(i)]),
                make_outcome(0.9 - i as f64 * 0.1, false, &["work"], &[belief(i)]),
                1000 + i as u64 * 1000,
            );
        }

        let beliefs: Vec<(NodeId, f64)> = (0..5).map(|i| (belief(i), 0.5)).collect();
        let report = run_replay_cycle(&mut engine, &beliefs, 100000);

        assert!(report.replays_executed > 0);
        assert_eq!(engine.stats.total_cycles, 1);
    }

    // ── Buffer maintenance ──────────────────────────────────────────────

    #[test]
    fn test_maintenance_removes_expired() {
        let mut engine = ReplayEngine::new();

        add_to_buffer(
            &mut engine,
            episode(1),
            0.5,
            make_action("work", &[entity(1)]),
            make_outcome(0.9, false, &["work"], &[belief(1)]),
            1000,
        );

        // Way in the future — episode should be expired.
        let far_future = 1000 + engine.budget.max_replay_age_ms + 1;
        let removed = buffer_maintenance(&mut engine, far_future);

        assert_eq!(removed, 1);
        assert!(engine.buffer.is_empty());
    }

    #[test]
    fn test_maintenance_keeps_recent() {
        let mut engine = ReplayEngine::new();

        add_to_buffer(
            &mut engine,
            episode(1),
            0.5,
            make_action("work", &[entity(1)]),
            make_outcome(0.9, false, &["work"], &[belief(1)]),
            1000,
        );

        let removed = buffer_maintenance(&mut engine, 2000);
        assert_eq!(removed, 0);
        assert_eq!(engine.buffer.len(), 1);
    }

    // ── Summary ─────────────────────────────────────────────────────────

    #[test]
    fn test_replay_summary() {
        let mut engine = ReplayEngine::new();

        add_to_buffer(
            &mut engine,
            episode(1),
            0.3,
            make_action("work", &[entity(1)]),
            make_outcome(0.9, false, &["work"], &[belief(1)]),
            1000,
        );
        add_to_buffer(
            &mut engine,
            episode(2),
            0.4,
            make_action("health", &[entity(2)]),
            make_outcome(0.8, false, &["health"], &[belief(2)]),
            2000,
        );

        let summary = replay_summary(&engine);
        assert_eq!(summary.buffer_size, 2);
        assert!(summary.avg_td_error > 0.0);
        assert_eq!(summary.domain_distribution.len(), 2);
    }
}
