//! CK-1.6: Intent Inference Engine
//!
//! Bayesian intent inference without LLM — detects what the user likely wants
//! from the cognitive state graph alone. Intent hypotheses are generated from
//! multiple signal sources, scored via a linear feature model, and ranked by
//! posterior probability.
//!
//! ## Signal Sources (hypothesis generators)
//!
//! 1. **Goal-driven**: Active goals with urgency → "user wants to advance goal X"
//! 2. **Routine-driven**: Detected routines near trigger time → "user is about to do Y"
//! 3. **Need-driven**: Unmet needs with rising intensity → "user needs Z"
//! 4. **Episode-driven**: Recent episodes suggest continuation → "user wants to continue A"
//! 5. **Opportunity-driven**: Time-bounded chances → "user should seize opportunity B"
//!
//! ## Scoring
//!
//! Each hypothesis gets a 10-dimensional feature vector:
//! - [0] goal_alignment: how well this intent advances active goals
//! - [1] temporal_match: routine phase proximity / deadline urgency
//! - [2] recency: how recently supporting evidence was observed
//! - [3] frequency: how often this intent has been acted on historically
//! - [4] entity_overlap: shared entities between intent and recent context
//! - [5] valence_match: emotional alignment with user's current state
//! - [6] evidence_strength: number and weight of supporting beliefs
//! - [7] need_intensity: urgency of the underlying need (if any)
//! - [8] confidence: average confidence of supporting nodes
//! - [9] novelty: inverse of how routine this intent is (surprise factor)
//!
//! Features are combined via learned weights (initially hand-tuned) to produce
//! a raw score, then softmax-normalized across competing hypotheses to get
//! posterior probabilities.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::state::*;

// ── Feature Dimensions ──

/// Number of dimensions in the intent feature vector.
pub const FEATURE_DIM: usize = 10;

/// Feature indices for readability.
pub mod feature {
    pub const GOAL_ALIGNMENT: usize = 0;
    pub const TEMPORAL_MATCH: usize = 1;
    pub const RECENCY: usize = 2;
    pub const FREQUENCY: usize = 3;
    pub const ENTITY_OVERLAP: usize = 4;
    pub const VALENCE_MATCH: usize = 5;
    pub const EVIDENCE_STRENGTH: usize = 6;
    pub const NEED_INTENSITY: usize = 7;
    pub const CONFIDENCE: usize = 8;
    pub const NOVELTY: usize = 9;
}

// ── Configuration ──

/// Configuration for intent inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentConfig {
    /// Feature weights for the linear scoring model.
    /// Length must equal FEATURE_DIM.
    pub weights: Vec<f64>,

    /// Minimum posterior probability to surface an intent hypothesis.
    pub min_posterior: f64,

    /// Maximum number of hypotheses to return.
    pub max_hypotheses: usize,

    /// Softmax temperature — lower = more decisive, higher = more exploratory.
    /// 1.0 = standard softmax, 0.5 = sharper, 2.0 = flatter.
    pub temperature: f64,

    /// Time window (seconds) for "recent" episodes.
    pub recency_window_secs: f64,

    /// Routine proximity window: how close to trigger time counts as "imminent".
    pub routine_proximity_secs: f64,

    /// Minimum goal urgency to generate goal-driven hypotheses.
    pub min_goal_urgency: f64,

    /// Minimum need intensity to generate need-driven hypotheses.
    pub min_need_intensity: f64,

    /// Whether to include opportunity-driven hypotheses.
    pub include_opportunities: bool,
}

impl Default for IntentConfig {
    fn default() -> Self {
        Self {
            // Hand-tuned initial weights — will be refined by adaptive learning (CK-3).
            // Goal alignment and temporal match are strongest signals.
            weights: vec![
                0.25, // goal_alignment  — strongest: goals drive proactive action
                0.20, // temporal_match  — strong: time context is very predictive
                0.12, // recency         — moderate: recent activity predicts intent
                0.10, // frequency       — moderate: habits predict behavior
                0.08, // entity_overlap  — moderate: shared context signals relevance
                0.05, // valence_match   — weak: emotional alignment matters less
                0.08, // evidence_strength — moderate: well-evidenced intents are real
                0.05, // need_intensity  — weak: needs are softer signals
                0.04, // confidence      — weak: confidence of supporting nodes
                0.03, // novelty         — weakest: surprise is a tiebreaker
            ],
            min_posterior: 0.05,
            max_hypotheses: 8,
            temperature: 1.0,
            recency_window_secs: 3600.0 * 4.0, // 4 hours
            routine_proximity_secs: 1800.0,    // 30 minutes
            min_goal_urgency: 0.3,
            min_need_intensity: 0.4,
            include_opportunities: true,
        }
    }
}

// ── Hypothesis Source ──

/// What signal source generated this intent hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentSource {
    /// An active goal with sufficient urgency.
    GoalDriven,
    /// A routine near its trigger time.
    RoutineDriven,
    /// An unmet need with rising intensity.
    NeedDriven,
    /// Recent episodes suggest continuation or follow-up.
    EpisodeDriven,
    /// A time-bounded opportunity about to expire.
    OpportunityDriven,
}

impl IntentSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GoalDriven => "goal_driven",
            Self::RoutineDriven => "routine_driven",
            Self::NeedDriven => "need_driven",
            Self::EpisodeDriven => "episode_driven",
            Self::OpportunityDriven => "opportunity_driven",
        }
    }
}

// ── Scored Hypothesis ──

/// A scored intent hypothesis ready for ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredIntent {
    /// Natural language description of the intent.
    pub description: String,

    /// What signal source generated this hypothesis.
    pub source: IntentSource,

    /// The 10-dimensional feature vector.
    pub features: Vec<f64>,

    /// Raw score from the linear model (before softmax).
    pub raw_score: f64,

    /// Posterior probability after softmax normalization [0.0, 1.0].
    pub posterior: f64,

    /// NodeIds of the cognitive nodes that support this hypothesis.
    pub supporting_nodes: Vec<NodeId>,

    /// NodeId of the source node (goal, routine, need, etc.) that spawned this.
    pub source_node: NodeId,
}

// ── Inference Result ──

/// Result of running intent inference on the cognitive graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentInferenceResult {
    /// Ranked hypotheses (highest posterior first).
    pub hypotheses: Vec<ScoredIntent>,

    /// Total number of hypotheses generated before filtering.
    pub total_generated: usize,

    /// Number filtered out by min_posterior threshold.
    pub filtered_count: usize,

    /// Inference duration in microseconds.
    pub duration_us: u64,
}

// ── Feature Extraction ──

/// Extract features for a goal-driven intent hypothesis.
pub fn extract_goal_features(
    goal: &CognitiveNode,
    goals: &[&CognitiveNode],
    episodes: &[&CognitiveNode],
    edges: &[CognitiveEdge],
    now: f64,
    config: &IntentConfig,
) -> Vec<f64> {
    let mut f = vec![0.0; FEATURE_DIM];

    // [0] goal_alignment: urgency of this specific goal
    f[feature::GOAL_ALIGNMENT] = goal.attrs.urgency;

    // [1] temporal_match: deadline proximity if present
    if let NodePayload::Goal(ref gp) = goal.payload {
        if let Some(deadline) = gp.deadline {
            let time_left = (deadline - now).max(0.0);
            let hours_left = time_left / 3600.0;
            // Sigmoid-like urgency curve: peaks as deadline approaches
            f[feature::TEMPORAL_MATCH] = 1.0 / (1.0 + (hours_left / 24.0));
        }
    }

    // [2] recency: how recently this goal was updated
    let age_secs = goal.attrs.age_secs();
    f[feature::RECENCY] = (-age_secs / config.recency_window_secs).exp();

    // [3] frequency: evidence count as proxy for how often this goal is touched
    f[feature::FREQUENCY] = (goal.attrs.evidence_count as f64 / 10.0).min(1.0);

    // [4] entity_overlap: count edges connecting this goal to recent episodes
    let goal_raw = goal.id.to_raw();
    let overlap = edges
        .iter()
        .filter(|e| {
            let src = e.src.to_raw();
            let tgt = e.dst.to_raw();
            (src == goal_raw || tgt == goal_raw)
                && episodes
                    .iter()
                    .any(|ep| ep.id.to_raw() == src || ep.id.to_raw() == tgt)
        })
        .count();
    f[feature::ENTITY_OVERLAP] = (overlap as f64 / 5.0).min(1.0);

    // [5] valence_match: positive goals feel good
    f[feature::VALENCE_MATCH] = (goal.attrs.valence + 1.0) / 2.0; // normalize [-1,1] → [0,1]

    // [6] evidence_strength: number of edges supporting this goal
    let support = edges
        .iter()
        .filter(|e| e.dst.to_raw() == goal_raw && e.kind == CognitiveEdgeKind::AdvancesGoal)
        .count();
    f[feature::EVIDENCE_STRENGTH] = (support as f64 / 5.0).min(1.0);

    // [7] need_intensity: 0 for goal-driven (no need involved)
    f[feature::NEED_INTENSITY] = 0.0;

    // [8] confidence: goal confidence
    f[feature::CONFIDENCE] = goal.attrs.confidence;

    // [9] novelty: how novel this goal is (new goals are more interesting)
    f[feature::NOVELTY] = goal.attrs.novelty;

    clamp_features(&mut f);
    f
}

/// Extract features for a routine-driven intent hypothesis.
pub fn extract_routine_features(
    routine: &CognitiveNode,
    edges: &[CognitiveEdge],
    now: f64,
    config: &IntentConfig,
) -> Vec<f64> {
    let mut f = vec![0.0; FEATURE_DIM];

    if let NodePayload::Routine(ref rp) = routine.payload {
        // [0] goal_alignment: check if any edge links this routine to a goal
        let routine_raw = routine.id.to_raw();
        let goal_edges = edges
            .iter()
            .filter(|e| e.src.to_raw() == routine_raw && e.kind == CognitiveEdgeKind::AdvancesGoal)
            .count();
        f[feature::GOAL_ALIGNMENT] = (goal_edges as f64 / 3.0).min(1.0);

        // [1] temporal_match: how close we are to the next trigger time
        let time_until = rp.time_until_next(now);
        if time_until <= config.routine_proximity_secs {
            // Closer = higher score: 1.0 at trigger time, decaying outward
            f[feature::TEMPORAL_MATCH] = 1.0 - (time_until / config.routine_proximity_secs);
        }

        // [2] recency: last triggered recency
        if rp.last_triggered > 0.0 {
            let since_last = now - rp.last_triggered;
            f[feature::RECENCY] = (-since_last / config.recency_window_secs).exp();
        }

        // [3] frequency: observation count as proxy
        f[feature::FREQUENCY] = (rp.observation_count as f64 / 20.0).min(1.0);

        // [5] valence_match: routines are emotionally neutral by default
        f[feature::VALENCE_MATCH] = (routine.attrs.valence + 1.0) / 2.0;

        // [6] evidence_strength: reliability of the routine
        f[feature::EVIDENCE_STRENGTH] = rp.reliability;

        // [8] confidence: routine confidence
        f[feature::CONFIDENCE] = routine.attrs.confidence;

        // [9] novelty: inverse of reliability — unreliable routines are more "novel"
        f[feature::NOVELTY] = 1.0 - rp.reliability;
    }

    clamp_features(&mut f);
    f
}

/// Extract features for a need-driven intent hypothesis.
pub fn extract_need_features(
    need: &CognitiveNode,
    edges: &[CognitiveEdge],
    now: f64,
    config: &IntentConfig,
) -> Vec<f64> {
    let mut f = vec![0.0; FEATURE_DIM];

    if let NodePayload::Need(ref np) = need.payload {
        // [0] goal_alignment: check if need links to any goals
        let need_raw = need.id.to_raw();
        let goal_edges = edges
            .iter()
            .filter(|e| e.src.to_raw() == need_raw && e.kind == CognitiveEdgeKind::AdvancesGoal)
            .count();
        f[feature::GOAL_ALIGNMENT] = (goal_edges as f64 / 3.0).min(1.0);

        // [1] temporal_match: time since last satisfied (longer = more urgent)
        if let Some(last_sat) = np.last_satisfied {
            let hours_since = (now - last_sat) / 3600.0;
            f[feature::TEMPORAL_MATCH] = (hours_since / 24.0).min(1.0);
        } else {
            f[feature::TEMPORAL_MATCH] = 0.8; // never satisfied = fairly urgent
        }

        // [2] recency: how recently this need was identified
        let age_secs = need.attrs.age_secs();
        f[feature::RECENCY] = (-age_secs / config.recency_window_secs).exp();

        // [5] valence_match: unmet needs have negative valence
        f[feature::VALENCE_MATCH] = (need.attrs.valence + 1.0) / 2.0;

        // [7] need_intensity: the primary signal for need-driven intents
        f[feature::NEED_INTENSITY] = np.intensity;

        // [8] confidence: need confidence
        f[feature::CONFIDENCE] = need.attrs.confidence;

        // [9] novelty: new needs are more novel
        f[feature::NOVELTY] = need.attrs.novelty;
    }

    clamp_features(&mut f);
    f
}

/// Extract features for an episode-driven intent hypothesis.
///
/// Episode-driven intents detect that the user was recently engaged in an
/// activity and likely wants to continue or follow up.
pub fn extract_episode_features(
    episode: &CognitiveNode,
    related_episodes: &[&CognitiveNode],
    edges: &[CognitiveEdge],
    now: f64,
    config: &IntentConfig,
) -> Vec<f64> {
    let mut f = vec![0.0; FEATURE_DIM];

    // [0] goal_alignment: does this episode connect to an active goal?
    let ep_raw = episode.id.to_raw();
    let goal_edges = edges
        .iter()
        .filter(|e| e.src.to_raw() == ep_raw && e.kind == CognitiveEdgeKind::AdvancesGoal)
        .count();
    f[feature::GOAL_ALIGNMENT] = (goal_edges as f64 / 3.0).min(1.0);

    // [1] temporal_match: recency is the key signal for episode continuation
    if let NodePayload::Episode(ref ep) = episode.payload {
        let age_hours = (now - ep.occurred_at) / 3600.0;
        f[feature::TEMPORAL_MATCH] = (-age_hours / 4.0).exp(); // 4-hour half-life
    }

    // [2] recency: same as temporal for episodes
    let age_secs = episode.attrs.age_secs();
    f[feature::RECENCY] = (-age_secs / config.recency_window_secs).exp();

    // [3] frequency: how many related episodes exist (activity cluster)
    f[feature::FREQUENCY] = (related_episodes.len() as f64 / 5.0).min(1.0);

    // [4] entity_overlap: shared edges between this episode and others
    let overlap = edges
        .iter()
        .filter(|e| {
            let src = e.src.to_raw();
            let tgt = e.dst.to_raw();
            (src == ep_raw || tgt == ep_raw)
                && related_episodes
                    .iter()
                    .any(|r| r.id.to_raw() == src || r.id.to_raw() == tgt)
        })
        .count();
    f[feature::ENTITY_OVERLAP] = (overlap as f64 / 5.0).min(1.0);

    // [5] valence_match
    f[feature::VALENCE_MATCH] = (episode.attrs.valence + 1.0) / 2.0;

    // [6] evidence_strength: activation level (how much attention this has)
    f[feature::EVIDENCE_STRENGTH] = episode.attrs.activation;

    // [8] confidence
    f[feature::CONFIDENCE] = episode.attrs.confidence;

    // [9] novelty
    f[feature::NOVELTY] = episode.attrs.novelty;

    clamp_features(&mut f);
    f
}

/// Extract features for an opportunity-driven intent hypothesis.
pub fn extract_opportunity_features(
    opportunity: &CognitiveNode,
    edges: &[CognitiveEdge],
    now: f64,
    _config: &IntentConfig,
) -> Vec<f64> {
    let mut f = vec![0.0; FEATURE_DIM];

    if let NodePayload::Opportunity(ref op) = opportunity.payload {
        // [0] goal_alignment: how many goals does this opportunity advance?
        f[feature::GOAL_ALIGNMENT] = (op.relevant_goals.len() as f64 / 3.0).min(1.0);

        // [1] temporal_match: urgency based on expiry
        let time_left = (op.expires_at - now).max(0.0);
        let hours_left = time_left / 3600.0;
        f[feature::TEMPORAL_MATCH] = if hours_left < 24.0 {
            1.0 - (hours_left / 24.0) // urgent when expiring soon
        } else {
            0.2 // low urgency for distant opportunities
        };

        // [2] recency: how new this opportunity is
        let age_secs = opportunity.attrs.age_secs();
        f[feature::RECENCY] = (-age_secs / 14400.0).exp(); // 4-hour half-life

        // [4] entity_overlap: edges to other active nodes
        let op_raw = opportunity.id.to_raw();
        let connections = edges
            .iter()
            .filter(|e| e.src.to_raw() == op_raw || e.dst.to_raw() == op_raw)
            .count();
        f[feature::ENTITY_OVERLAP] = (connections as f64 / 5.0).min(1.0);

        // [6] evidence_strength: expected benefit
        f[feature::EVIDENCE_STRENGTH] = op.expected_benefit;

        // [8] confidence
        f[feature::CONFIDENCE] = opportunity.attrs.confidence;

        // [9] novelty: opportunities are inherently novel
        f[feature::NOVELTY] = opportunity.attrs.novelty.max(0.5);
    }

    clamp_features(&mut f);
    f
}

// ── Linear Scoring ──

/// Compute raw score from feature vector and weights via dot product.
pub fn linear_score(features: &[f64], weights: &[f64]) -> f64 {
    features
        .iter()
        .zip(weights.iter())
        .map(|(f, w)| f * w)
        .sum()
}

/// Softmax normalization with temperature.
///
/// Returns a vector of probabilities summing to 1.0.
/// Temperature < 1.0 sharpens the distribution (more decisive).
/// Temperature > 1.0 flattens it (more exploratory).
pub fn softmax(scores: &[f64], temperature: f64) -> Vec<f64> {
    if scores.is_empty() {
        return vec![];
    }

    let temp = temperature.max(0.01); // prevent division by zero
    let scaled: Vec<f64> = scores.iter().map(|s| s / temp).collect();

    // Subtract max for numerical stability
    let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|s| (s - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();

    if sum <= 0.0 {
        // Degenerate case: uniform distribution
        let n = scores.len() as f64;
        return vec![1.0 / n; scores.len()];
    }

    exps.iter().map(|e| e / sum).collect()
}

// ── Hypothesis Generation ──

/// Generate intent hypotheses from goal nodes.
pub fn generate_goal_hypotheses(
    goals: &[&CognitiveNode],
    episodes: &[&CognitiveNode],
    edges: &[CognitiveEdge],
    now: f64,
    config: &IntentConfig,
) -> Vec<ScoredIntent> {
    goals
        .iter()
        .filter(|g| {
            g.attrs.urgency >= config.min_goal_urgency
                && matches!(g.payload, NodePayload::Goal(ref gp) if gp.status == GoalStatus::Active)
        })
        .map(|g| {
            let features = extract_goal_features(g, goals, episodes, edges, now, config);
            let raw_score = linear_score(&features, &config.weights);
            let description = match &g.payload {
                NodePayload::Goal(gp) => format!("Advance goal: {}", gp.description),
                _ => "Advance unknown goal".to_string(),
            };
            ScoredIntent {
                description,
                source: IntentSource::GoalDriven,
                features,
                raw_score,
                posterior: 0.0, // set after softmax
                supporting_nodes: vec![g.id],
                source_node: g.id,
            }
        })
        .collect()
}

/// Generate intent hypotheses from routine nodes near their trigger time.
pub fn generate_routine_hypotheses(
    routines: &[&CognitiveNode],
    edges: &[CognitiveEdge],
    now: f64,
    config: &IntentConfig,
) -> Vec<ScoredIntent> {
    routines
        .iter()
        .filter(|r| {
            if let NodePayload::Routine(ref rp) = r.payload {
                rp.time_until_next(now) <= config.routine_proximity_secs
            } else {
                false
            }
        })
        .map(|r| {
            let features = extract_routine_features(r, edges, now, config);
            let raw_score = linear_score(&features, &config.weights);
            let description = match &r.payload {
                NodePayload::Routine(rp) => format!("Routine: {}", rp.description),
                _ => "Unknown routine".to_string(),
            };
            ScoredIntent {
                description,
                source: IntentSource::RoutineDriven,
                features,
                raw_score,
                posterior: 0.0,
                supporting_nodes: vec![r.id],
                source_node: r.id,
            }
        })
        .collect()
}

/// Generate intent hypotheses from unmet need nodes.
pub fn generate_need_hypotheses(
    needs: &[&CognitiveNode],
    edges: &[CognitiveEdge],
    now: f64,
    config: &IntentConfig,
) -> Vec<ScoredIntent> {
    needs
        .iter()
        .filter(|n| {
            if let NodePayload::Need(ref np) = n.payload {
                np.intensity >= config.min_need_intensity
            } else {
                false
            }
        })
        .map(|n| {
            let features = extract_need_features(n, edges, now, config);
            let raw_score = linear_score(&features, &config.weights);
            let description = match &n.payload {
                NodePayload::Need(np) => format!("Address need: {}", np.description),
                _ => "Address unknown need".to_string(),
            };
            ScoredIntent {
                description,
                source: IntentSource::NeedDriven,
                features,
                raw_score,
                posterior: 0.0,
                supporting_nodes: vec![n.id],
                source_node: n.id,
            }
        })
        .collect()
}

/// Generate intent hypotheses from recent episodes.
///
/// Only the most recent episode is used as a "continuation" seed.
/// Older episodes within the recency window serve as supporting context.
pub fn generate_episode_hypotheses(
    episodes: &[&CognitiveNode],
    edges: &[CognitiveEdge],
    now: f64,
    config: &IntentConfig,
) -> Vec<ScoredIntent> {
    // Only generate from episodes within the recency window
    let recent: Vec<&CognitiveNode> = episodes
        .iter()
        .filter(|e| {
            if let NodePayload::Episode(ref ep) = e.payload {
                (now - ep.occurred_at) <= config.recency_window_secs
            } else {
                e.attrs.age_secs() <= config.recency_window_secs
            }
        })
        .copied()
        .collect();

    if recent.is_empty() {
        return vec![];
    }

    // Use the most recent episode as the primary seed
    let seed = recent.iter().max_by(|a, b| {
        let a_time = match &a.payload {
            NodePayload::Episode(ep) => ep.occurred_at,
            _ => 0.0,
        };
        let b_time = match &b.payload {
            NodePayload::Episode(ep) => ep.occurred_at,
            _ => 0.0,
        };
        a_time
            .partial_cmp(&b_time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    match seed {
        Some(s) => {
            let features = extract_episode_features(s, &recent, edges, now, config);
            let raw_score = linear_score(&features, &config.weights);
            let description = match &s.payload {
                NodePayload::Episode(ep) => format!("Continue: {}", ep.summary),
                _ => "Continue recent activity".to_string(),
            };
            let supporting: Vec<NodeId> = recent.iter().map(|e| e.id).collect();
            vec![ScoredIntent {
                description,
                source: IntentSource::EpisodeDriven,
                features,
                raw_score,
                posterior: 0.0,
                supporting_nodes: supporting,
                source_node: s.id,
            }]
        }
        None => vec![],
    }
}

/// Generate intent hypotheses from opportunity nodes.
pub fn generate_opportunity_hypotheses(
    opportunities: &[&CognitiveNode],
    edges: &[CognitiveEdge],
    now: f64,
    config: &IntentConfig,
) -> Vec<ScoredIntent> {
    if !config.include_opportunities {
        return vec![];
    }

    opportunities
        .iter()
        .filter(|o| {
            if let NodePayload::Opportunity(ref op) = o.payload {
                op.expires_at > now // not yet expired
            } else {
                false
            }
        })
        .map(|o| {
            let features = extract_opportunity_features(o, edges, now, config);
            let raw_score = linear_score(&features, &config.weights);
            let description = match &o.payload {
                NodePayload::Opportunity(op) => format!("Seize opportunity: {}", op.description),
                _ => "Unknown opportunity".to_string(),
            };
            let mut supporting = vec![o.id];
            if let NodePayload::Opportunity(ref op) = o.payload {
                supporting.extend_from_slice(&op.relevant_goals);
            }
            ScoredIntent {
                description,
                source: IntentSource::OpportunityDriven,
                features,
                raw_score,
                posterior: 0.0,
                supporting_nodes: supporting,
                source_node: o.id,
            }
        })
        .collect()
}

// ── Full Inference Pipeline ──

/// Run the full intent inference pipeline on a set of cognitive nodes and edges.
///
/// This is the core function — it generates hypotheses from all signal sources,
/// scores them via the linear model, normalizes with softmax, filters by
/// min_posterior, and returns ranked results.
pub fn infer_intents(
    nodes: &[&CognitiveNode],
    edges: &[CognitiveEdge],
    now: f64,
    config: &IntentConfig,
) -> IntentInferenceResult {
    let start = std::time::Instant::now();

    // Partition nodes by kind
    let goals: Vec<&CognitiveNode> = nodes
        .iter()
        .filter(|n| n.id.kind() == NodeKind::Goal)
        .copied()
        .collect();
    let routines: Vec<&CognitiveNode> = nodes
        .iter()
        .filter(|n| n.id.kind() == NodeKind::Routine)
        .copied()
        .collect();
    let needs: Vec<&CognitiveNode> = nodes
        .iter()
        .filter(|n| n.id.kind() == NodeKind::Need)
        .copied()
        .collect();
    let episodes: Vec<&CognitiveNode> = nodes
        .iter()
        .filter(|n| n.id.kind() == NodeKind::Episode)
        .copied()
        .collect();
    let opportunities: Vec<&CognitiveNode> = nodes
        .iter()
        .filter(|n| n.id.kind() == NodeKind::Opportunity)
        .copied()
        .collect();

    // Generate hypotheses from all sources
    let mut all_hypotheses: Vec<ScoredIntent> = Vec::new();
    all_hypotheses.extend(generate_goal_hypotheses(
        &goals, &episodes, edges, now, config,
    ));
    all_hypotheses.extend(generate_routine_hypotheses(&routines, edges, now, config));
    all_hypotheses.extend(generate_need_hypotheses(&needs, edges, now, config));
    all_hypotheses.extend(generate_episode_hypotheses(&episodes, edges, now, config));
    all_hypotheses.extend(generate_opportunity_hypotheses(
        &opportunities,
        edges,
        now,
        config,
    ));

    let total_generated = all_hypotheses.len();

    if all_hypotheses.is_empty() {
        return IntentInferenceResult {
            hypotheses: vec![],
            total_generated: 0,
            filtered_count: 0,
            duration_us: start.elapsed().as_micros() as u64,
        };
    }

    // Softmax normalization across all hypotheses
    let raw_scores: Vec<f64> = all_hypotheses.iter().map(|h| h.raw_score).collect();
    let posteriors = softmax(&raw_scores, config.temperature);

    for (h, &p) in all_hypotheses.iter_mut().zip(posteriors.iter()) {
        h.posterior = p;
    }

    // Sort by posterior descending
    all_hypotheses.sort_by(|a, b| {
        b.posterior
            .partial_cmp(&a.posterior)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Filter by min_posterior and truncate
    let filtered: Vec<ScoredIntent> = all_hypotheses
        .into_iter()
        .filter(|h| h.posterior >= config.min_posterior)
        .take(config.max_hypotheses)
        .collect();

    let filtered_count = total_generated - filtered.len();

    IntentInferenceResult {
        hypotheses: filtered,
        total_generated,
        filtered_count,
        duration_us: start.elapsed().as_micros() as u64,
    }
}

/// Convert a ScoredIntent into an IntentPayload for storage as a CognitiveNode.
pub fn intent_to_payload(intent: &ScoredIntent) -> IntentPayload {
    IntentPayload {
        description: intent.description.clone(),
        features: intent.features.clone(),
        posterior: intent.posterior,
        candidate_actions: vec![], // filled by action schema (CK-1.7)
        source_context: intent.source.as_str().to_string(),
    }
}

/// Create a CognitiveNode for an intent hypothesis.
pub fn intent_to_node(intent: &ScoredIntent, alloc: &mut NodeIdAllocator) -> CognitiveNode {
    let id = alloc.alloc(NodeKind::IntentHypothesis);
    let mut attrs = CognitiveAttrs::default_for(NodeKind::IntentHypothesis);
    attrs.confidence = intent.posterior;
    attrs.activation = intent.raw_score.clamp(0.0, 1.0);
    attrs.salience = intent
        .features
        .get(feature::GOAL_ALIGNMENT)
        .copied()
        .unwrap_or(0.0)
        .max(attrs.salience);
    attrs.urgency = intent
        .features
        .get(feature::TEMPORAL_MATCH)
        .copied()
        .unwrap_or(0.0);
    attrs.provenance = Provenance::Inferred;

    CognitiveNode {
        id,
        label: intent.description.clone(),
        attrs,
        payload: NodePayload::IntentHypothesis(intent_to_payload(intent)),
        metadata: HashMap::new(),
    }
}

// ── Helpers ──

/// Clamp all feature values to [0.0, 1.0].
fn clamp_features(features: &mut [f64]) {
    for f in features.iter_mut() {
        *f = f.clamp(0.0, 1.0);
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn make_goal(alloc: &mut NodeIdAllocator, desc: &str, urgency: f64) -> CognitiveNode {
        let id = alloc.alloc(NodeKind::Goal);
        let mut attrs = CognitiveAttrs::default_for(NodeKind::Goal);
        attrs.urgency = urgency;
        CognitiveNode {
            id,
            label: desc.to_string(),
            attrs,
            payload: NodePayload::Goal(GoalPayload {
                description: desc.to_string(),
                status: GoalStatus::Active,
                progress: 0.3,
                deadline: None,
                priority: Priority::High,
                parent_goal: None,
                completion_criteria: "Done when complete".to_string(),
            }),
            metadata: HashMap::new(),
        }
    }

    fn make_routine(alloc: &mut NodeIdAllocator, desc: &str, now: f64) -> CognitiveNode {
        let id = alloc.alloc(NodeKind::Routine);
        let attrs = CognitiveAttrs::default_for(NodeKind::Routine);
        CognitiveNode {
            id,
            label: desc.to_string(),
            attrs,
            payload: NodePayload::Routine(RoutinePayload {
                description: desc.to_string(),
                period_secs: 86400.0,           // daily
                phase_offset_secs: now + 600.0, // triggers in 10 minutes
                reliability: 0.85,
                observation_count: 15,
                last_triggered: now - 86400.0,
                action_description: "Do the thing".to_string(),
                weekday_mask: 0x7F,
            }),
            metadata: HashMap::new(),
        }
    }

    fn make_need(alloc: &mut NodeIdAllocator, desc: &str, intensity: f64) -> CognitiveNode {
        let id = alloc.alloc(NodeKind::Need);
        let attrs = CognitiveAttrs::default_for(NodeKind::Need);
        CognitiveNode {
            id,
            label: desc.to_string(),
            attrs,
            payload: NodePayload::Need(NeedPayload {
                description: desc.to_string(),
                category: NeedCategory::Informational,
                intensity,
                last_satisfied: None,
                satisfaction_pattern: "Search for information".to_string(),
            }),
            metadata: HashMap::new(),
        }
    }

    fn make_episode(alloc: &mut NodeIdAllocator, summary: &str, occurred_at: f64) -> CognitiveNode {
        let id = alloc.alloc(NodeKind::Episode);
        let attrs = CognitiveAttrs::default_for(NodeKind::Episode);
        CognitiveNode {
            id,
            label: summary.to_string(),
            attrs,
            payload: NodePayload::Episode(EpisodePayload {
                memory_rid: format!("rid_{}", id.to_raw()),
                summary: summary.to_string(),
                occurred_at,
                participants: vec!["user".to_string()],
            }),
            metadata: HashMap::new(),
        }
    }

    fn make_opportunity(alloc: &mut NodeIdAllocator, desc: &str, expires_at: f64) -> CognitiveNode {
        let id = alloc.alloc(NodeKind::Opportunity);
        let attrs = CognitiveAttrs::default_for(NodeKind::Opportunity);
        CognitiveNode {
            id,
            label: desc.to_string(),
            attrs,
            payload: NodePayload::Opportunity(OpportunityPayload {
                description: desc.to_string(),
                expires_at,
                expected_benefit: 0.8,
                required_action: "Act now".to_string(),
                relevant_goals: vec![],
            }),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_softmax_basic() {
        let scores = vec![1.0, 2.0, 3.0];
        let probs = softmax(&scores, 1.0);
        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "softmax should sum to 1.0");
        assert!(probs[2] > probs[1], "higher score → higher probability");
        assert!(probs[1] > probs[0], "higher score → higher probability");
    }

    #[test]
    fn test_softmax_temperature() {
        let scores = vec![1.0, 2.0, 3.0];
        let sharp = softmax(&scores, 0.5);
        let flat = softmax(&scores, 2.0);
        // Sharper temperature → winner takes more
        assert!(
            sharp[2] > flat[2],
            "lower temperature should sharpen distribution"
        );
    }

    #[test]
    fn test_softmax_empty() {
        let probs = softmax(&[], 1.0);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_linear_score() {
        let features = vec![1.0, 0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let weights = vec![0.25, 0.20, 0.12, 0.10, 0.08, 0.05, 0.08, 0.05, 0.04, 0.03];
        let score = linear_score(&features, &weights);
        let expected = 1.0 * 0.25 + 0.5 * 0.20 + 0.3 * 0.10;
        assert!((score - expected).abs() < 1e-10);
    }

    #[test]
    fn test_goal_hypothesis_generation() {
        let mut alloc = NodeIdAllocator::new();
        let g1 = make_goal(&mut alloc, "Finish report", 0.8);
        let g2 = make_goal(&mut alloc, "Buy groceries", 0.2); // below threshold
        let g3 = make_goal(&mut alloc, "Exercise", 0.5);

        let goals = vec![&g1, &g2, &g3];
        let config = IntentConfig::default();
        let now = 1710000000.0;

        let hypotheses = generate_goal_hypotheses(&goals, &[], &[], now, &config);

        // g2 should be filtered out (urgency 0.2 < min_goal_urgency 0.3)
        assert_eq!(hypotheses.len(), 2);
        assert!(hypotheses[0].description.contains("Finish report"));
        assert!(hypotheses[1].description.contains("Exercise"));
    }

    #[test]
    fn test_routine_hypothesis_generation() {
        let mut alloc = NodeIdAllocator::new();
        let now = 1710000000.0;
        let r1 = make_routine(&mut alloc, "Check email", now);
        // This routine triggers in 10 minutes — within proximity window

        let routines = vec![&r1];
        let config = IntentConfig::default();

        let hypotheses = generate_routine_hypotheses(&routines, &[], now, &config);

        assert_eq!(hypotheses.len(), 1);
        assert!(hypotheses[0].description.contains("Check email"));
        assert!(hypotheses[0].features[feature::TEMPORAL_MATCH] > 0.5);
    }

    #[test]
    fn test_need_hypothesis_generation() {
        let mut alloc = NodeIdAllocator::new();
        let n1 = make_need(&mut alloc, "Research Rust async patterns", 0.7);
        let n2 = make_need(&mut alloc, "Social interaction", 0.2); // below threshold

        let needs = vec![&n1, &n2];
        let config = IntentConfig::default();
        let now = 1710000000.0;

        let hypotheses = generate_need_hypotheses(&needs, &[], now, &config);

        assert_eq!(hypotheses.len(), 1);
        assert!(hypotheses[0].description.contains("Research"));
    }

    #[test]
    fn test_episode_hypothesis_generation() {
        let mut alloc = NodeIdAllocator::new();
        let now = 1710000000.0;
        let e1 = make_episode(&mut alloc, "Working on database migration", now - 1800.0);
        let e2 = make_episode(&mut alloc, "Debugging test failures", now - 600.0);

        let episodes = vec![&e1, &e2];
        let config = IntentConfig::default();

        let hypotheses = generate_episode_hypotheses(&episodes, &[], now, &config);

        // Should pick the most recent episode as the seed
        assert_eq!(hypotheses.len(), 1);
        assert!(hypotheses[0]
            .description
            .contains("Debugging test failures"));
        assert_eq!(hypotheses[0].source, IntentSource::EpisodeDriven);
    }

    #[test]
    fn test_opportunity_hypothesis_generation() {
        let mut alloc = NodeIdAllocator::new();
        let now = 1710000000.0;
        let o1 = make_opportunity(&mut alloc, "Conference early-bird ends", now + 7200.0);
        let o2 = make_opportunity(&mut alloc, "Expired sale", now - 100.0); // expired

        let opportunities = vec![&o1, &o2];
        let config = IntentConfig::default();

        let hypotheses = generate_opportunity_hypotheses(&opportunities, &[], now, &config);

        assert_eq!(hypotheses.len(), 1);
        assert!(hypotheses[0].description.contains("Conference"));
    }

    #[test]
    fn test_full_inference_pipeline() {
        let mut alloc = NodeIdAllocator::new();
        let now = 1710000000.0;

        let g1 = make_goal(&mut alloc, "Finish report", 0.8);
        let r1 = make_routine(&mut alloc, "Check email", now);
        let n1 = make_need(&mut alloc, "Research async", 0.6);
        let e1 = make_episode(&mut alloc, "Writing code", now - 600.0);
        let o1 = make_opportunity(&mut alloc, "Conference", now + 3600.0);

        let nodes: Vec<&CognitiveNode> = vec![&g1, &r1, &n1, &e1, &o1];
        let config = IntentConfig::default();

        let result = infer_intents(&nodes, &[], now, &config);

        assert_eq!(result.total_generated, 5);
        assert!(
            !result.hypotheses.is_empty(),
            "should produce at least one hypothesis"
        );

        // Posteriors should sum to ≤ 1.0 (some may be filtered)
        let sum: f64 = result.hypotheses.iter().map(|h| h.posterior).sum();
        assert!(sum <= 1.01, "posteriors should sum to ≤ 1.0, got {}", sum);

        // Should be sorted by posterior descending
        for w in result.hypotheses.windows(2) {
            assert!(
                w[0].posterior >= w[1].posterior,
                "should be sorted descending"
            );
        }
    }

    #[test]
    fn test_intent_to_node() {
        let mut alloc = NodeIdAllocator::new();
        let intent = ScoredIntent {
            description: "Test intent".to_string(),
            source: IntentSource::GoalDriven,
            features: vec![0.8, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            raw_score: 0.4,
            posterior: 0.35,
            supporting_nodes: vec![],
            source_node: alloc.alloc(NodeKind::Goal),
        };

        let node = intent_to_node(&intent, &mut alloc);

        assert_eq!(node.id.kind(), NodeKind::IntentHypothesis);
        assert_eq!(node.label, "Test intent");
        assert!((node.attrs.confidence - 0.35).abs() < 1e-10); // posterior → confidence
        assert_eq!(node.attrs.provenance, Provenance::Inferred);
        if let NodePayload::IntentHypothesis(ref ip) = node.payload {
            assert_eq!(ip.features.len(), 10);
            assert!((ip.posterior - 0.35).abs() < 1e-10);
        } else {
            panic!("expected IntentHypothesis payload");
        }
    }

    #[test]
    fn test_empty_graph_produces_no_hypotheses() {
        let config = IntentConfig::default();
        let result = infer_intents(&[], &[], 1710000000.0, &config);

        assert_eq!(result.total_generated, 0);
        assert!(result.hypotheses.is_empty());
        assert_eq!(result.filtered_count, 0);
    }

    #[test]
    fn test_clamp_features() {
        let mut f = vec![-0.5, 1.5, 0.5, -0.1, 2.0];
        clamp_features(&mut f);
        assert_eq!(f, vec![0.0, 1.0, 0.5, 0.0, 1.0]);
    }

    #[test]
    fn test_feature_dimensions() {
        // Ensure default config has correct number of weights
        let config = IntentConfig::default();
        assert_eq!(config.weights.len(), FEATURE_DIM);

        // Ensure all feature extractors produce correct-length vectors
        let mut alloc = NodeIdAllocator::new();
        let now = 1710000000.0;

        let g = make_goal(&mut alloc, "Test", 0.5);
        let f = extract_goal_features(&g, &[], &[], &[], now, &config);
        assert_eq!(f.len(), FEATURE_DIM);

        let r = make_routine(&mut alloc, "Test", now);
        let f = extract_routine_features(&r, &[], now, &config);
        assert_eq!(f.len(), FEATURE_DIM);

        let n = make_need(&mut alloc, "Test", 0.5);
        let f = extract_need_features(&n, &[], now, &config);
        assert_eq!(f.len(), FEATURE_DIM);

        let e = make_episode(&mut alloc, "Test", now - 100.0);
        let f = extract_episode_features(&e, &[], &[], now, &config);
        assert_eq!(f.len(), FEATURE_DIM);

        let o = make_opportunity(&mut alloc, "Test", now + 3600.0);
        let f = extract_opportunity_features(&o, &[], now, &config);
        assert_eq!(f.len(), FEATURE_DIM);
    }
}
