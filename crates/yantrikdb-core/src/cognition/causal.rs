//! CK-4.1 — Causal-Lite Inference Engine.
//!
//! Local causal reasoning from temporal patterns, intervention outcomes,
//! and graph structure. NOT full causal discovery (no d-separation, no
//! global DAG fitting). Instead: practical, incremental, explainable
//! causal edges suitable for a personal AI companion.
//!
//! # Design principles
//! - Pure functions only — no DB access (engine layer handles persistence)
//! - Online / incremental — edges strengthen or weaken with each observation
//! - Explainable — every causal claim carries a `CausalTrace` showing why
//! - Conservative — high bar for claiming causation vs. correlation
//! - Context-aware — causal strength can vary by `StateFeatures` context

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::observer::{EventBuffer, EventKind, SystemEvent, SystemEventData};
use crate::state::{CognitiveEdge, CognitiveEdgeKind, NodeId};
use crate::world_model::{ActionKind, ActionOutcome, StateFeatures, TransitionModel};

// ── §1: Core types ──────────────────────────────────────────────────

/// A directed causal hypothesis: cause → effect.
///
/// Stronger than correlation — requires temporal precedence, repeated
/// co-occurrence, and ideally intervention evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEdge {
    /// What we believe causes the effect.
    pub cause: CausalNode,
    /// What we believe is caused.
    pub effect: CausalNode,
    /// Overall causal strength ∈ [-1.0, 1.0].
    /// Positive = promotes, negative = inhibits/prevents.
    pub strength: f64,
    /// Confidence in the causal relationship ∈ [0.0, 1.0].
    pub confidence: f64,
    /// How many times we've observed this pattern.
    pub observation_count: u32,
    /// How many times an intervention confirmed causality.
    pub intervention_count: u32,
    /// How many times the cause occurred WITHOUT the effect following.
    pub non_occurrence_count: u32,
    /// Lag statistics: median delay from cause to effect (seconds).
    pub median_lag_secs: f64,
    /// Lag statistics: interquartile range of delays (seconds).
    pub lag_iqr_secs: f64,
    /// Context conditions under which the causal link holds most strongly.
    pub context_strengths: Vec<ContextualStrength>,
    /// Evidence chain explaining why we believe this is causal.
    pub trace: CausalTrace,
    /// When this edge was first hypothesized (unix seconds).
    pub created_at: f64,
    /// When this edge was last updated (unix seconds).
    pub updated_at: f64,
    /// Current lifecycle stage.
    pub stage: CausalStage,
}

/// Identifies a cause or effect node.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CausalNode {
    /// A cognitive graph node (belief, goal, task, etc.).
    GraphNode(NodeId),
    /// An event kind (app opened, suggestion accepted, etc.).
    Event(EventKind),
    /// An action the system took.
    Action(ActionKind),
    /// An abstract named signal (e.g. "high_error_rate", "morning_routine").
    Signal(String),
}

/// Lifecycle stage of a causal hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalStage {
    /// Just observed temporal correlation, not yet tested.
    Hypothesized,
    /// Repeated observations above threshold, promoting to candidate.
    Candidate,
    /// Intervention evidence or strong statistical support.
    Established,
    /// Contradicted by recent evidence, being weakened.
    Weakening,
    /// Confidence below threshold, marked for removal.
    Refuted,
}

/// Causal strength modulated by context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualStrength {
    pub features: StateFeatures,
    pub strength: f64,
    pub observation_count: u32,
}

// ── §2: Causal trace (explainability) ───────────────────────────────

/// Full evidence chain for a causal hypothesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalTrace {
    /// Individual pieces of evidence.
    pub evidence: Vec<CausalEvidence>,
    /// Overall reasoning method that produced this edge.
    pub primary_method: DiscoveryMethod,
    /// Human-readable summary.
    pub summary: String,
}

/// A single piece of causal evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CausalEvidence {
    /// Temporal precedence: cause consistently precedes effect.
    TemporalPrecedence {
        /// How many times cause preceded effect within the lag window.
        co_occurrences: u32,
        /// Average lag (seconds).
        avg_lag_secs: f64,
        /// Standard deviation of lag.
        lag_stddev_secs: f64,
    },
    /// Granger-style: past cause values improve prediction of effect.
    GrangerPrecedence {
        /// Improvement in prediction accuracy when including cause.
        prediction_lift: f64,
        /// Baseline prediction accuracy without cause.
        baseline_accuracy: f64,
        /// Number of test windows evaluated.
        test_windows: u32,
    },
    /// Intervention: we took an action and observed the expected effect.
    InterventionOutcome {
        /// The action we took.
        action: ActionKind,
        /// How many interventions produced the expected effect.
        successes: u32,
        /// How many interventions did NOT produce the expected effect.
        failures: u32,
        /// Posterior probability of causal link (Beta distribution).
        posterior_mean: f64,
    },
    /// Conditional independence test (simplified).
    /// If cause and effect become independent when conditioning on Z,
    /// the link is likely spurious.
    ConditionalTest {
        /// The conditioning variable.
        conditioner: CausalNode,
        /// Mutual information with conditioner.
        mi_with_conditioner: f64,
        /// Mutual information without conditioner.
        mi_without_conditioner: f64,
        /// Whether the edge survived the test.
        survived: bool,
    },
    /// Dose-response: stronger cause → stronger effect.
    DoseResponse {
        /// Correlation between cause intensity and effect magnitude.
        correlation: f64,
        /// Number of graded observations.
        sample_size: u32,
    },
    /// Graph structure: existing causal paths in the cognitive graph.
    GraphPath {
        /// Intermediate nodes in the causal path.
        intermediates: Vec<NodeId>,
        /// Combined edge weight along the path.
        path_weight: f64,
    },
}

/// How the causal hypothesis was discovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiscoveryMethod {
    /// Temporal co-occurrence mining.
    TemporalAssociation,
    /// Granger-style predictive precedence.
    GrangerPrecedence,
    /// Intervention/experiment result.
    Intervention,
    /// Cognitive graph structure analysis.
    GraphStructure,
    /// User explicitly told us.
    UserProvided,
    /// Multiple methods converged.
    Converged,
}

// ── §3: Causal store ────────────────────────────────────────────────

/// The causal knowledge base — all known causal hypotheses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalStore {
    /// All causal edges, keyed by (cause, effect) for fast lookup.
    edges: Vec<CausalEdge>,
    /// Index: cause → vec of edge indices.
    #[serde(skip)]
    cause_index: HashMap<CausalNode, Vec<usize>>,
    /// Index: effect → vec of edge indices.
    #[serde(skip)]
    effect_index: HashMap<CausalNode, Vec<usize>>,
    /// Configuration.
    pub config: CausalConfig,
    /// Statistics.
    pub total_hypothesized: u64,
    pub total_established: u64,
    pub total_refuted: u64,
}

/// Configuration for causal discovery and maintenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalConfig {
    /// Minimum co-occurrences before promoting from Hypothesized → Candidate.
    pub min_observations_candidate: u32,
    /// Minimum co-occurrences before promoting from Candidate → Established.
    pub min_observations_established: u32,
    /// Minimum confidence to remain Established.
    pub min_confidence_established: f64,
    /// Maximum lag window for temporal association (seconds).
    pub max_lag_window_secs: f64,
    /// Minimum lag (seconds) — ignore near-simultaneous events.
    pub min_lag_secs: f64,
    /// Confidence below which an edge is marked Refuted.
    pub refutation_threshold: f64,
    /// Decay factor per day for unconfirmed edges.
    pub daily_decay: f64,
    /// Maximum edges to keep (prune weakest when exceeded).
    pub max_edges: usize,
    /// How many intervention successes needed for strong evidence.
    pub intervention_evidence_threshold: u32,
    /// Minimum prediction lift for Granger evidence.
    pub min_granger_lift: f64,
}

impl Default for CausalConfig {
    fn default() -> Self {
        Self {
            min_observations_candidate: 5,
            min_observations_established: 15,
            min_confidence_established: 0.6,
            max_lag_window_secs: 300.0, // 5 minutes
            min_lag_secs: 0.5,
            refutation_threshold: 0.2,
            daily_decay: 0.98,
            max_edges: 500,
            intervention_evidence_threshold: 3,
            min_granger_lift: 0.1,
        }
    }
}

impl CausalStore {
    pub fn new() -> Self {
        Self::with_config(CausalConfig::default())
    }

    pub fn with_config(config: CausalConfig) -> Self {
        Self {
            edges: Vec::new(),
            cause_index: HashMap::new(),
            effect_index: HashMap::new(),
            config,
            total_hypothesized: 0,
            total_established: 0,
            total_refuted: 0,
        }
    }

    /// Rebuild runtime indices after deserialization.
    pub fn rebuild_indices(&mut self) {
        self.cause_index.clear();
        self.effect_index.clear();
        for (i, edge) in self.edges.iter().enumerate() {
            self.cause_index
                .entry(edge.cause.clone())
                .or_default()
                .push(i);
            self.effect_index
                .entry(edge.effect.clone())
                .or_default()
                .push(i);
        }
    }

    /// Get all causal edges (read-only).
    pub fn edges(&self) -> &[CausalEdge] {
        &self.edges
    }

    /// Number of active edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Find a specific edge by cause and effect.
    pub fn find_edge(&self, cause: &CausalNode, effect: &CausalNode) -> Option<&CausalEdge> {
        self.cause_index.get(cause).and_then(|indices| {
            indices.iter().find_map(|&i| {
                let e = &self.edges[i];
                if &e.effect == effect {
                    Some(e)
                } else {
                    None
                }
            })
        })
    }

    /// Find a specific edge by cause and effect (mutable).
    fn find_edge_mut(
        &mut self,
        cause: &CausalNode,
        effect: &CausalNode,
    ) -> Option<&mut CausalEdge> {
        let idx = self.cause_index.get(cause).and_then(|indices| {
            indices
                .iter()
                .find(|&&i| &self.edges[i].effect == effect)
                .copied()
        });
        idx.map(|i| &mut self.edges[i])
    }

    /// Get all effects caused by a given node.
    pub fn effects_of(&self, cause: &CausalNode) -> Vec<&CausalEdge> {
        self.cause_index
            .get(cause)
            .map(|indices| indices.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    /// Get all causes of a given effect.
    pub fn causes_of(&self, effect: &CausalNode) -> Vec<&CausalEdge> {
        self.effect_index
            .get(effect)
            .map(|indices| indices.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    /// Insert or update a causal edge.
    pub fn upsert(&mut self, edge: CausalEdge) {
        if let Some(existing) = self.find_edge_mut(&edge.cause, &edge.effect) {
            *existing = edge;
        } else {
            let idx = self.edges.len();
            self.cause_index
                .entry(edge.cause.clone())
                .or_default()
                .push(idx);
            self.effect_index
                .entry(edge.effect.clone())
                .or_default()
                .push(idx);
            self.edges.push(edge);
            self.total_hypothesized += 1;
        }
    }

    /// Remove all Refuted edges and rebuild indices.
    pub fn prune_refuted(&mut self) -> usize {
        let before = self.edges.len();
        self.edges.retain(|e| e.stage != CausalStage::Refuted);
        let pruned = before - self.edges.len();
        if pruned > 0 {
            self.rebuild_indices();
        }
        pruned
    }

    /// Prune to max_edges, removing weakest (lowest confidence) first.
    pub fn enforce_capacity(&mut self) -> usize {
        if self.edges.len() <= self.config.max_edges {
            return 0;
        }
        // Sort by confidence ascending → remove from front.
        let mut indexed: Vec<(usize, f64)> = self
            .edges
            .iter()
            .enumerate()
            .map(|(i, e)| (i, e.confidence))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let to_remove = self.edges.len() - self.config.max_edges;
        let remove_set: std::collections::HashSet<usize> =
            indexed.iter().take(to_remove).map(|&(i, _)| i).collect();

        let before = self.edges.len();
        let mut kept = Vec::with_capacity(self.config.max_edges);
        for (i, edge) in self.edges.drain(..).enumerate() {
            if !remove_set.contains(&i) {
                kept.push(edge);
            }
        }
        self.edges = kept;
        self.rebuild_indices();
        before - self.edges.len()
    }
}

// ── §4: Temporal association discovery ──────────────────────────────

/// Result of scanning events for temporal co-occurrence patterns.
#[derive(Debug, Clone)]
pub struct TemporalPattern {
    pub cause: CausalNode,
    pub effect: CausalNode,
    pub co_occurrences: u32,
    pub avg_lag_secs: f64,
    pub lag_stddev_secs: f64,
    pub lags: Vec<f64>,
}

/// Discover temporal association patterns from event history.
///
/// For each pair of event kinds, counts how often event A precedes
/// event B within the configured lag window. Returns patterns that
/// meet the minimum observation threshold.
pub fn discover_temporal_patterns(
    events: &[&SystemEvent],
    config: &CausalConfig,
    min_count: u32,
) -> Vec<TemporalPattern> {
    // Group events by kind for efficient pairwise comparison.
    let mut by_kind: HashMap<EventKind, Vec<f64>> = HashMap::new();
    for ev in events {
        let kind = event_kind_of(&ev.data);
        by_kind.entry(kind).or_default().push(ev.timestamp);
    }

    // Sort timestamps within each kind.
    for timestamps in by_kind.values_mut() {
        timestamps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }

    let mut patterns = Vec::new();
    let kinds: Vec<EventKind> = by_kind.keys().copied().collect();

    for &cause_kind in &kinds {
        for &effect_kind in &kinds {
            if cause_kind == effect_kind {
                continue;
            }

            let cause_times = &by_kind[&cause_kind];
            let effect_times = &by_kind[&effect_kind];

            let mut lags = Vec::new();

            // For each cause event, find the closest following effect event.
            for &ct in cause_times {
                // Binary search for the first effect after ct + min_lag.
                let min_t = ct + config.min_lag_secs;
                let max_t = ct + config.max_lag_window_secs;

                let start = effect_times.partition_point(|&t| t < min_t);

                if start < effect_times.len() && effect_times[start] <= max_t {
                    lags.push(effect_times[start] - ct);
                }
            }

            if lags.len() >= min_count as usize {
                let n = lags.len() as f64;
                let avg = lags.iter().sum::<f64>() / n;
                let variance = if n > 1.0 {
                    lags.iter().map(|l| (l - avg).powi(2)).sum::<f64>() / (n - 1.0)
                } else {
                    0.0
                };

                patterns.push(TemporalPattern {
                    cause: CausalNode::Event(cause_kind),
                    effect: CausalNode::Event(effect_kind),
                    co_occurrences: lags.len() as u32,
                    avg_lag_secs: avg,
                    lag_stddev_secs: variance.sqrt(),
                    lags,
                });
            }
        }
    }

    patterns
}

/// Integrate discovered temporal patterns into the causal store.
pub fn integrate_temporal_patterns(
    store: &mut CausalStore,
    patterns: Vec<TemporalPattern>,
    now: f64,
) {
    let config = store.config.clone();

    for pattern in patterns {
        let lags = &pattern.lags;
        let median_lag = median_f64(lags);
        let (q1, q3) = quartiles_f64(lags);
        let iqr = q3 - q1;

        // Compute a preliminary confidence from consistency of lags.
        // Low variance relative to mean → more confident.
        let cv = if pattern.avg_lag_secs > 0.0 {
            pattern.lag_stddev_secs / pattern.avg_lag_secs
        } else {
            1.0
        };
        // CV < 0.3 → very consistent → confidence boost.
        // CV > 1.0 → noisy → low confidence.
        let consistency_score = (1.0 - cv.min(1.0)).max(0.0);
        let count_score = (pattern.co_occurrences as f64 / 20.0).min(1.0);
        let confidence = 0.3 * consistency_score + 0.7 * count_score;

        if let Some(existing) = store.find_edge_mut(&pattern.cause, &pattern.effect) {
            // Update existing edge.
            existing.observation_count += pattern.co_occurrences;
            existing.median_lag_secs = median_lag;
            existing.lag_iqr_secs = iqr;
            existing.updated_at = now;

            // Blend confidence with existing.
            existing.confidence = 0.7 * existing.confidence + 0.3 * confidence;

            // Add temporal evidence.
            existing
                .trace
                .evidence
                .push(CausalEvidence::TemporalPrecedence {
                    co_occurrences: pattern.co_occurrences,
                    avg_lag_secs: pattern.avg_lag_secs,
                    lag_stddev_secs: pattern.lag_stddev_secs,
                });

            // Stage transitions.
            update_stage(existing, &config);
        } else {
            let mut edge = CausalEdge {
                cause: pattern.cause,
                effect: pattern.effect,
                strength: confidence.min(0.5), // Conservative initial strength.
                confidence,
                observation_count: pattern.co_occurrences,
                intervention_count: 0,
                non_occurrence_count: 0,
                median_lag_secs: median_lag,
                lag_iqr_secs: iqr,
                context_strengths: Vec::new(),
                trace: CausalTrace {
                    evidence: vec![CausalEvidence::TemporalPrecedence {
                        co_occurrences: pattern.co_occurrences,
                        avg_lag_secs: pattern.avg_lag_secs,
                        lag_stddev_secs: pattern.lag_stddev_secs,
                    }],
                    primary_method: DiscoveryMethod::TemporalAssociation,
                    summary: format!(
                        "Observed {} times with avg lag {:.1}s (σ={:.1}s)",
                        pattern.co_occurrences, pattern.avg_lag_secs, pattern.lag_stddev_secs,
                    ),
                },
                created_at: now,
                updated_at: now,
                stage: CausalStage::Hypothesized,
            };
            update_stage(&mut edge, &config);
            store.upsert(edge);
        }
    }
}

// ── §5: Intervention tracking ───────────────────────────────────────

/// Record the outcome of a deliberate intervention (action).
///
/// If we took action A expecting effect E, record whether E followed.
/// This is the strongest form of causal evidence.
pub fn record_intervention(
    store: &mut CausalStore,
    action: ActionKind,
    expected_effect: CausalNode,
    effect_observed: bool,
    now: f64,
) {
    let cause = CausalNode::Action(action);
    let config = store.config.clone();

    if let Some(existing) = store.find_edge_mut(&cause, &expected_effect) {
        if effect_observed {
            existing.intervention_count += 1;
            existing.observation_count += 1;
        } else {
            existing.non_occurrence_count += 1;
        }

        // Beta posterior for intervention evidence.
        let successes = existing.intervention_count as f64;
        let failures = existing.non_occurrence_count as f64;
        let posterior_mean = (successes + 1.0) / (successes + failures + 2.0);

        existing
            .trace
            .evidence
            .push(CausalEvidence::InterventionOutcome {
                action,
                successes: existing.intervention_count,
                failures: existing.non_occurrence_count,
                posterior_mean,
            });

        // Intervention evidence strongly updates confidence.
        existing.confidence = 0.4 * existing.confidence + 0.6 * posterior_mean;
        existing.strength = existing.confidence * if existing.strength >= 0.0 { 1.0 } else { -1.0 };
        existing.updated_at = now;

        update_stage(existing, &config);
    } else {
        // First intervention for this pair — create a new hypothesis.
        let posterior_mean = if effect_observed {
            2.0 / 3.0
        } else {
            1.0 / 3.0
        };
        let edge = CausalEdge {
            cause: cause.clone(),
            effect: expected_effect,
            strength: if effect_observed { 0.4 } else { -0.2 },
            confidence: posterior_mean,
            observation_count: 1,
            intervention_count: if effect_observed { 1 } else { 0 },
            non_occurrence_count: if effect_observed { 0 } else { 1 },
            median_lag_secs: 0.0,
            lag_iqr_secs: 0.0,
            context_strengths: Vec::new(),
            trace: CausalTrace {
                evidence: vec![CausalEvidence::InterventionOutcome {
                    action,
                    successes: if effect_observed { 1 } else { 0 },
                    failures: if effect_observed { 0 } else { 1 },
                    posterior_mean,
                }],
                primary_method: DiscoveryMethod::Intervention,
                summary: format!(
                    "Intervention: action {:?} → effect {}observed",
                    action,
                    if effect_observed { "" } else { "NOT " },
                ),
            },
            created_at: now,
            updated_at: now,
            stage: CausalStage::Hypothesized,
        };
        store.upsert(edge);
    }
}

// ── §6: Granger-style predictive precedence ─────────────────────────

/// Simplified Granger test: does knowing about the cause improve our
/// ability to predict the effect?
///
/// Compares prediction accuracy of effect timing with and without
/// cause history. Returns the prediction lift.
pub fn granger_test(
    cause_times: &[f64],
    effect_times: &[f64],
    window_secs: f64,
    n_windows: usize,
) -> Option<GrangerResult> {
    if cause_times.is_empty() || effect_times.is_empty() || n_windows == 0 {
        return None;
    }

    // Find the time range.
    let all_times: Vec<f64> = cause_times
        .iter()
        .chain(effect_times.iter())
        .copied()
        .collect();
    let t_min = all_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = all_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = t_max - t_min;

    if range < window_secs * 2.0 {
        return None; // Not enough data.
    }

    let step = range / n_windows as f64;

    // Baseline: predict "effect happens in next window" based only on
    // effect base rate.
    let effect_rate = effect_times.len() as f64 / (range / window_secs);

    let mut correct_baseline = 0u32;
    let mut correct_with_cause = 0u32;
    let mut total_windows = 0u32;

    for w in 0..n_windows {
        let w_start = t_min + w as f64 * step;
        let w_end = w_start + step;
        let next_start = w_end;
        let next_end = next_start + step;

        if next_end > t_max {
            break;
        }

        total_windows += 1;

        let effect_in_next = effect_times
            .iter()
            .any(|&t| t >= next_start && t < next_end);

        // Baseline: predict based on rate.
        let baseline_predicts = effect_rate > 0.5;
        if baseline_predicts == effect_in_next {
            correct_baseline += 1;
        }

        // With cause: predict effect if cause occurred in current window.
        let cause_in_window = cause_times.iter().any(|&t| t >= w_start && t < w_end);
        if cause_in_window == effect_in_next {
            correct_with_cause += 1;
        }
    }

    if total_windows < 3 {
        return None;
    }

    let baseline_accuracy = correct_baseline as f64 / total_windows as f64;
    let with_cause_accuracy = correct_with_cause as f64 / total_windows as f64;
    let lift = with_cause_accuracy - baseline_accuracy;

    Some(GrangerResult {
        prediction_lift: lift,
        baseline_accuracy,
        with_cause_accuracy,
        test_windows: total_windows,
    })
}

/// Result of a simplified Granger test.
#[derive(Debug, Clone)]
pub struct GrangerResult {
    pub prediction_lift: f64,
    pub baseline_accuracy: f64,
    pub with_cause_accuracy: f64,
    pub test_windows: u32,
}

/// Apply Granger evidence to a causal edge.
pub fn apply_granger_evidence(
    store: &mut CausalStore,
    cause: CausalNode,
    effect: CausalNode,
    result: &GrangerResult,
    now: f64,
    config: &CausalConfig,
) {
    if result.prediction_lift < config.min_granger_lift {
        return; // Not significant enough.
    }

    let evidence = CausalEvidence::GrangerPrecedence {
        prediction_lift: result.prediction_lift,
        baseline_accuracy: result.baseline_accuracy,
        test_windows: result.test_windows,
    };

    if let Some(existing) = store.find_edge_mut(&cause, &effect) {
        existing.trace.evidence.push(evidence);
        // Granger evidence provides moderate confidence boost.
        let granger_confidence = 0.5 + 0.5 * result.prediction_lift.min(1.0);
        existing.confidence = 0.6 * existing.confidence + 0.4 * granger_confidence;
        existing.updated_at = now;
        update_stage(existing, config);
    } else {
        let confidence = 0.3 + 0.3 * result.prediction_lift.min(1.0);
        let edge = CausalEdge {
            cause,
            effect,
            strength: confidence * 0.5,
            confidence,
            observation_count: result.test_windows,
            intervention_count: 0,
            non_occurrence_count: 0,
            median_lag_secs: 0.0,
            lag_iqr_secs: 0.0,
            context_strengths: Vec::new(),
            trace: CausalTrace {
                evidence: vec![evidence],
                primary_method: DiscoveryMethod::GrangerPrecedence,
                summary: format!(
                    "Granger test: {:.1}% prediction lift over {:.1}% baseline ({} windows)",
                    result.prediction_lift * 100.0,
                    result.baseline_accuracy * 100.0,
                    result.test_windows,
                ),
            },
            created_at: now,
            updated_at: now,
            stage: CausalStage::Hypothesized,
        };
        store.upsert(edge);
    }
}

// ── §7: Graph-based causal inference ────────────────────────────────

/// Extract causal edges from the cognitive graph.
///
/// Looks for existing `Causes`, `Predicts`, `Prevents` edges and
/// converts them to CausalEdge entries.
pub fn extract_graph_causal_edges(graph_edges: &[CognitiveEdge], now: f64) -> Vec<CausalEdge> {
    let mut results = Vec::new();

    for ge in graph_edges {
        if !ge.kind.is_causal() {
            continue;
        }

        let (strength_sign, description) = match ge.kind {
            CognitiveEdgeKind::Causes => (1.0, "causes"),
            CognitiveEdgeKind::Predicts => (0.7, "predicts"),
            CognitiveEdgeKind::Prevents => (-1.0, "prevents"),
            _ => continue,
        };

        let edge = CausalEdge {
            cause: CausalNode::GraphNode(ge.src),
            effect: CausalNode::GraphNode(ge.dst),
            strength: ge.weight * strength_sign,
            confidence: ge.confidence,
            observation_count: ge.observation_count,
            intervention_count: 0,
            non_occurrence_count: 0,
            median_lag_secs: 0.0,
            lag_iqr_secs: 0.0,
            context_strengths: Vec::new(),
            trace: CausalTrace {
                evidence: vec![CausalEvidence::GraphPath {
                    intermediates: Vec::new(),
                    path_weight: ge.weight,
                }],
                primary_method: DiscoveryMethod::GraphStructure,
                summary: format!(
                    "Graph edge: {:?} {} {:?} (weight={:.2}, obs={})",
                    ge.src, description, ge.dst, ge.weight, ge.observation_count,
                ),
            },
            created_at: ge.created_at_ms as f64 / 1000.0,
            updated_at: now,
            stage: if ge.confidence >= 0.6 && ge.observation_count >= 15 {
                CausalStage::Established
            } else if ge.observation_count >= 5 {
                CausalStage::Candidate
            } else {
                CausalStage::Hypothesized
            },
        };
        results.push(edge);
    }

    results
}

// ── §8: Online query API ────────────────────────────────────────────

/// Estimate the causal effect of a cause on a specific effect.
///
/// Returns None if no causal link is known.
pub fn estimate_effect(
    store: &CausalStore,
    cause: &CausalNode,
    effect: &CausalNode,
) -> Option<EffectEstimate> {
    let edge = store.find_edge(cause, effect)?;

    // Weigh evidence types differently.
    let has_intervention = edge.intervention_count > 0;
    let has_granger = edge
        .trace
        .evidence
        .iter()
        .any(|e| matches!(e, CausalEvidence::GrangerPrecedence { .. }));
    let evidence_quality = if has_intervention {
        EvidenceQuality::Interventional
    } else if has_granger {
        EvidenceQuality::Predictive
    } else {
        EvidenceQuality::Correlational
    };

    Some(EffectEstimate {
        strength: edge.strength,
        confidence: edge.confidence,
        evidence_quality,
        observation_count: edge.observation_count,
        intervention_count: edge.intervention_count,
        median_lag_secs: edge.median_lag_secs,
        stage: edge.stage,
    })
}

/// Predict all downstream effects of activating a cause.
///
/// Follows causal chains up to `max_depth` hops. Returns effects
/// sorted by expected impact (strength × confidence, decaying with depth).
pub fn predict_effects(
    store: &CausalStore,
    cause: &CausalNode,
    max_depth: u32,
) -> Vec<PredictedEffect> {
    let mut visited: HashMap<CausalNode, PredictedEffect> = HashMap::new();
    let mut frontier: Vec<(CausalNode, f64, f64, u32)> = vec![(
        cause.clone(),
        1.0, // strength multiplier
        1.0, // confidence multiplier
        0,   // depth
    )];

    while let Some((current, str_mult, conf_mult, depth)) = frontier.pop() {
        if depth > max_depth {
            continue;
        }

        for edge in store.effects_of(&current) {
            if edge.stage == CausalStage::Refuted {
                continue;
            }

            let propagated_strength = edge.strength * str_mult * 0.8; // 20% decay per hop
            let propagated_confidence = edge.confidence * conf_mult * 0.9;

            // Only propagate if signal is still meaningful.
            if propagated_confidence < 0.1 || propagated_strength.abs() < 0.05 {
                continue;
            }

            let effect_node = &edge.effect;

            let update = visited
                .entry(effect_node.clone())
                .or_insert_with(|| PredictedEffect {
                    node: effect_node.clone(),
                    expected_strength: 0.0,
                    confidence: 0.0,
                    hops: depth + 1,
                    path_count: 0,
                });

            // Combine multiple paths: use max confidence, sum strengths.
            update.expected_strength += propagated_strength;
            update.confidence = update.confidence.max(propagated_confidence);
            update.hops = update.hops.min(depth + 1);
            update.path_count += 1;

            if depth + 1 < max_depth {
                frontier.push((
                    effect_node.clone(),
                    propagated_strength,
                    propagated_confidence,
                    depth + 1,
                ));
            }
        }
    }

    // Remove the cause itself from results.
    visited.remove(cause);

    let mut results: Vec<PredictedEffect> = visited.into_values().collect();
    // Sort by |strength| × confidence descending.
    results.sort_by(|a, b| {
        let score_a = a.expected_strength.abs() * a.confidence;
        let score_b = b.expected_strength.abs() * b.confidence;
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

/// What-if analysis: if cause were activated, what would happen?
///
/// Combines causal prediction with world model transition data
/// for a richer picture.
pub fn what_if(
    store: &CausalStore,
    transition_model: &TransitionModel,
    cause: &CausalNode,
    context: &StateFeatures,
    max_depth: u32,
) -> WhatIfResult {
    let predicted = predict_effects(store, cause, max_depth);

    // Check transition model for action-based causes.
    let transition_info = if let CausalNode::Action(action_kind) = cause {
        let dist = transition_model.predict(context, *action_kind);
        Some(TransitionPrediction {
            accept_prob: dist.posterior_mean(ActionOutcome::Accepted),
            reject_prob: dist.posterior_mean(ActionOutcome::Rejected),
            ignore_prob: dist.posterior_mean(ActionOutcome::Ignored),
            succeed_prob: dist.posterior_mean(ActionOutcome::Succeeded),
            fail_prob: dist.posterior_mean(ActionOutcome::Failed),
        })
    } else {
        None
    };

    // Separate positive and negative effects.
    let positive_effects: Vec<PredictedEffect> = predicted
        .iter()
        .filter(|e| e.expected_strength > 0.0)
        .cloned()
        .collect();
    let negative_effects: Vec<PredictedEffect> = predicted
        .iter()
        .filter(|e| e.expected_strength < 0.0)
        .cloned()
        .collect();

    let net_expected_impact: f64 = predicted
        .iter()
        .map(|e| e.expected_strength * e.confidence)
        .sum();

    WhatIfResult {
        cause: cause.clone(),
        predicted_effects: predicted,
        positive_effects,
        negative_effects,
        net_expected_impact,
        transition_prediction: transition_info,
        max_depth_reached: max_depth,
    }
}

/// The result of an effect estimation query.
#[derive(Debug, Clone)]
pub struct EffectEstimate {
    pub strength: f64,
    pub confidence: f64,
    pub evidence_quality: EvidenceQuality,
    pub observation_count: u32,
    pub intervention_count: u32,
    pub median_lag_secs: f64,
    pub stage: CausalStage,
}

/// Quality tier of causal evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceQuality {
    /// Only temporal correlation.
    Correlational,
    /// Granger-style predictive precedence.
    Predictive,
    /// Actual intervention/experiment evidence.
    Interventional,
}

/// A predicted downstream effect.
#[derive(Debug, Clone)]
pub struct PredictedEffect {
    pub node: CausalNode,
    pub expected_strength: f64,
    pub confidence: f64,
    pub hops: u32,
    pub path_count: u32,
}

/// Transition model prediction for action outcomes.
#[derive(Debug, Clone)]
pub struct TransitionPrediction {
    pub accept_prob: f64,
    pub reject_prob: f64,
    pub ignore_prob: f64,
    pub succeed_prob: f64,
    pub fail_prob: f64,
}

/// Complete what-if analysis result.
#[derive(Debug, Clone)]
pub struct WhatIfResult {
    pub cause: CausalNode,
    pub predicted_effects: Vec<PredictedEffect>,
    pub positive_effects: Vec<PredictedEffect>,
    pub negative_effects: Vec<PredictedEffect>,
    pub net_expected_impact: f64,
    pub transition_prediction: Option<TransitionPrediction>,
    pub max_depth_reached: u32,
}

// ── §9: Decay and maintenance ───────────────────────────────────────

/// Apply time-based decay to all causal edges.
///
/// Edges that haven't been confirmed recently lose confidence.
/// Returns the number of edges that transitioned to Refuted.
pub fn apply_decay(store: &mut CausalStore, now: f64) -> u32 {
    let daily_decay = store.config.daily_decay;
    let refutation_threshold = store.config.refutation_threshold;
    let mut newly_refuted = 0u32;

    for edge in &mut store.edges {
        let days_since_update = (now - edge.updated_at) / 86400.0;
        if days_since_update <= 0.0 {
            continue;
        }

        let decay_factor = daily_decay.powf(days_since_update);
        edge.confidence *= decay_factor;

        if edge.confidence < refutation_threshold && edge.stage != CausalStage::Refuted {
            edge.stage = CausalStage::Refuted;
            newly_refuted += 1;
            store.total_refuted += 1;
        } else if edge.confidence < store.config.min_confidence_established
            && edge.stage == CausalStage::Established
        {
            edge.stage = CausalStage::Weakening;
        }
    }

    newly_refuted
}

/// Full maintenance pass: decay, prune refuted, enforce capacity.
pub fn maintain_causal_store(store: &mut CausalStore, now: f64) -> MaintenanceReport {
    let decayed = apply_decay(store, now);
    let pruned_refuted = store.prune_refuted();
    let pruned_capacity = store.enforce_capacity();

    MaintenanceReport {
        edges_decayed_to_refuted: decayed,
        refuted_edges_pruned: pruned_refuted,
        capacity_pruned: pruned_capacity,
        remaining_edges: store.edges.len(),
    }
}

/// Report from a maintenance pass.
#[derive(Debug, Clone)]
pub struct MaintenanceReport {
    pub edges_decayed_to_refuted: u32,
    pub refuted_edges_pruned: usize,
    pub capacity_pruned: usize,
    pub remaining_edges: usize,
}

// ── §10: Discovery orchestrator ─────────────────────────────────────

/// Run a full causal discovery pass over recent events.
///
/// This is the main entry point for the engine layer to call
/// during a cognitive tick.
pub fn discover_local_causality(
    store: &mut CausalStore,
    events: &EventBuffer,
    graph_edges: &[CognitiveEdge],
    now: f64,
) -> DiscoveryReport {
    let config = store.config.clone();

    // 1. Temporal pattern discovery from recent events.
    let recent = events.recent(1000);
    let temporal_patterns =
        discover_temporal_patterns(&recent, &config, config.min_observations_candidate);
    let temporal_count = temporal_patterns.len();
    integrate_temporal_patterns(store, temporal_patterns, now);

    // 2. Granger tests on event kind pairs with sufficient data.
    let mut granger_count = 0u32;
    let mut by_kind: HashMap<EventKind, Vec<f64>> = HashMap::new();
    for ev in &recent {
        let kind = event_kind_of(&ev.data);
        by_kind.entry(kind).or_default().push(ev.timestamp);
    }
    let kinds: Vec<EventKind> = by_kind.keys().copied().collect();
    for &ck in &kinds {
        for &ek in &kinds {
            if ck == ek {
                continue;
            }
            let ct = &by_kind[&ck];
            let et = &by_kind[&ek];
            if ct.len() < 10 || et.len() < 10 {
                continue; // Need reasonable sample.
            }
            if let Some(result) = granger_test(ct, et, config.max_lag_window_secs, 20) {
                if result.prediction_lift >= config.min_granger_lift {
                    apply_granger_evidence(
                        store,
                        CausalNode::Event(ck),
                        CausalNode::Event(ek),
                        &result,
                        now,
                        &config,
                    );
                    granger_count += 1;
                }
            }
        }
    }

    // 3. Graph structure extraction.
    let graph_causal = extract_graph_causal_edges(graph_edges, now);
    let graph_count = graph_causal.len();
    for edge in graph_causal {
        store.upsert(edge);
    }

    // 4. Maintenance.
    let maintenance = maintain_causal_store(store, now);

    DiscoveryReport {
        temporal_patterns_found: temporal_count,
        granger_edges_added: granger_count,
        graph_edges_imported: graph_count,
        maintenance,
        total_edges: store.edges.len(),
    }
}

/// Report from a discovery pass.
#[derive(Debug, Clone)]
pub struct DiscoveryReport {
    pub temporal_patterns_found: usize,
    pub granger_edges_added: u32,
    pub graph_edges_imported: usize,
    pub maintenance: MaintenanceReport,
    pub total_edges: usize,
}

// ── §11: Summary and explanation ────────────────────────────────────

/// Human-readable summary of the causal store.
pub fn causal_summary(store: &CausalStore) -> CausalSummary {
    let mut by_stage = [0u32; 5];
    let mut by_method = HashMap::new();
    let mut strongest: Option<&CausalEdge> = None;
    let mut most_observed: Option<&CausalEdge> = None;

    for edge in &store.edges {
        let stage_idx = match edge.stage {
            CausalStage::Hypothesized => 0,
            CausalStage::Candidate => 1,
            CausalStage::Established => 2,
            CausalStage::Weakening => 3,
            CausalStage::Refuted => 4,
        };
        by_stage[stage_idx] += 1;

        *by_method.entry(edge.trace.primary_method).or_insert(0u32) += 1;

        if strongest.map_or(true, |s| edge.confidence > s.confidence) {
            strongest = Some(edge);
        }
        if most_observed.map_or(true, |s| edge.observation_count > s.observation_count) {
            most_observed = Some(edge);
        }
    }

    CausalSummary {
        total_edges: store.edges.len(),
        hypothesized: by_stage[0],
        candidates: by_stage[1],
        established: by_stage[2],
        weakening: by_stage[3],
        refuted: by_stage[4],
        by_discovery_method: by_method,
        strongest_edge: strongest.map(|e| EdgeSummary {
            cause: format!("{:?}", e.cause),
            effect: format!("{:?}", e.effect),
            strength: e.strength,
            confidence: e.confidence,
            observations: e.observation_count,
        }),
        most_observed_edge: most_observed.map(|e| EdgeSummary {
            cause: format!("{:?}", e.cause),
            effect: format!("{:?}", e.effect),
            strength: e.strength,
            confidence: e.confidence,
            observations: e.observation_count,
        }),
        lifetime_hypothesized: store.total_hypothesized,
        lifetime_established: store.total_established,
        lifetime_refuted: store.total_refuted,
    }
}

/// Summary of the causal knowledge base.
#[derive(Debug, Clone)]
pub struct CausalSummary {
    pub total_edges: usize,
    pub hypothesized: u32,
    pub candidates: u32,
    pub established: u32,
    pub weakening: u32,
    pub refuted: u32,
    pub by_discovery_method: HashMap<DiscoveryMethod, u32>,
    pub strongest_edge: Option<EdgeSummary>,
    pub most_observed_edge: Option<EdgeSummary>,
    pub lifetime_hypothesized: u64,
    pub lifetime_established: u64,
    pub lifetime_refuted: u64,
}

/// Compact edge summary for reporting.
#[derive(Debug, Clone)]
pub struct EdgeSummary {
    pub cause: String,
    pub effect: String,
    pub strength: f64,
    pub confidence: f64,
    pub observations: u32,
}

/// Explain a specific causal edge in detail.
pub fn explain_causal_edge(
    store: &CausalStore,
    cause: &CausalNode,
    effect: &CausalNode,
) -> Option<CausalExplanation> {
    let edge = store.find_edge(cause, effect)?;

    let evidence_summary: Vec<String> = edge
        .trace
        .evidence
        .iter()
        .map(|e| match e {
            CausalEvidence::TemporalPrecedence {
                co_occurrences,
                avg_lag_secs,
                ..
            } => format!(
                "Temporal: {} co-occurrences, avg lag {:.1}s",
                co_occurrences, avg_lag_secs
            ),
            CausalEvidence::GrangerPrecedence {
                prediction_lift,
                test_windows,
                ..
            } => format!(
                "Granger: {:.1}% prediction lift ({} windows)",
                prediction_lift * 100.0,
                test_windows
            ),
            CausalEvidence::InterventionOutcome {
                successes,
                failures,
                posterior_mean,
                ..
            } => format!(
                "Intervention: {}/{} successes (posterior={:.2})",
                successes,
                successes + failures,
                posterior_mean
            ),
            CausalEvidence::ConditionalTest { survived, .. } => format!(
                "Conditional independence test: {}",
                if *survived { "SURVIVED" } else { "FAILED" }
            ),
            CausalEvidence::DoseResponse {
                correlation,
                sample_size,
            } => format!("Dose-response: r={:.2} (n={})", correlation, sample_size),
            CausalEvidence::GraphPath {
                intermediates,
                path_weight,
            } => format!(
                "Graph path: {} hops, weight={:.2}",
                intermediates.len(),
                path_weight
            ),
        })
        .collect();

    Some(CausalExplanation {
        cause: format!("{:?}", edge.cause),
        effect: format!("{:?}", edge.effect),
        strength: edge.strength,
        confidence: edge.confidence,
        stage: edge.stage,
        observation_count: edge.observation_count,
        intervention_count: edge.intervention_count,
        non_occurrence_count: edge.non_occurrence_count,
        median_lag_secs: edge.median_lag_secs,
        evidence_summary,
        primary_method: edge.trace.primary_method,
        trace_summary: edge.trace.summary.clone(),
        context_count: edge.context_strengths.len(),
    })
}

/// Detailed explanation of a causal edge.
#[derive(Debug, Clone)]
pub struct CausalExplanation {
    pub cause: String,
    pub effect: String,
    pub strength: f64,
    pub confidence: f64,
    pub stage: CausalStage,
    pub observation_count: u32,
    pub intervention_count: u32,
    pub non_occurrence_count: u32,
    pub median_lag_secs: f64,
    pub evidence_summary: Vec<String>,
    pub primary_method: DiscoveryMethod,
    pub trace_summary: String,
    pub context_count: usize,
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Update causal stage based on current observation count and confidence.
fn update_stage(edge: &mut CausalEdge, config: &CausalConfig) {
    match edge.stage {
        CausalStage::Hypothesized => {
            if edge.observation_count >= config.min_observations_candidate {
                edge.stage = CausalStage::Candidate;
            }
        }
        CausalStage::Candidate => {
            if edge.observation_count >= config.min_observations_established
                && edge.confidence >= config.min_confidence_established
            {
                edge.stage = CausalStage::Established;
            }
        }
        CausalStage::Established => {
            if edge.confidence < config.min_confidence_established {
                edge.stage = CausalStage::Weakening;
            }
        }
        CausalStage::Weakening => {
            if edge.confidence >= config.min_confidence_established {
                edge.stage = CausalStage::Established; // Recovered.
            } else if edge.confidence < config.refutation_threshold {
                edge.stage = CausalStage::Refuted;
            }
        }
        CausalStage::Refuted => {
            // No recovery from refuted — prune it.
        }
    }
}

/// Extract EventKind from SystemEventData.
fn event_kind_of(data: &SystemEventData) -> EventKind {
    match data {
        SystemEventData::AppOpened { .. } => EventKind::AppOpened,
        SystemEventData::AppClosed { .. } => EventKind::AppClosed,
        SystemEventData::AppSequence { .. } => EventKind::AppSequence,
        SystemEventData::NotificationReceived { .. } => EventKind::NotificationReceived,
        SystemEventData::NotificationDismissed { .. } => EventKind::NotificationDismissed,
        SystemEventData::NotificationActedOn { .. } => EventKind::NotificationActedOn,
        SystemEventData::SuggestionAccepted { .. } => EventKind::SuggestionAccepted,
        SystemEventData::SuggestionRejected { .. } => EventKind::SuggestionRejected,
        SystemEventData::SuggestionIgnored { .. } => EventKind::SuggestionIgnored,
        SystemEventData::SuggestionModified { .. } => EventKind::SuggestionModified,
        SystemEventData::QueryRepeated { .. } => EventKind::QueryRepeated,
        SystemEventData::UserTyping { .. } => EventKind::UserTyping,
        SystemEventData::UserIdle { .. } => EventKind::UserIdle,
        SystemEventData::ToolCallCompleted { .. } => EventKind::ToolCallCompleted,
        SystemEventData::LlmCalled { .. } => EventKind::LlmCalled,
        SystemEventData::ErrorOccurred { .. } => EventKind::ErrorOccurred,
    }
}

/// Compute median of a slice (non-empty).
fn median_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Compute Q1 and Q3 of a slice.
fn quartiles_f64(values: &[f64]) -> (f64, f64) {
    if values.len() < 4 {
        return (
            values.first().copied().unwrap_or(0.0),
            values.last().copied().unwrap_or(0.0),
        );
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let q1 = sorted[n / 4];
    let q3 = sorted[3 * n / 4];
    (q1, q3)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observer::{EventBuffer, EventKind, SystemEvent, SystemEventData};

    fn make_event(timestamp: f64, data: SystemEventData) -> SystemEvent {
        SystemEvent { timestamp, data }
    }

    fn app_opened(ts: f64, app_id: u16) -> SystemEvent {
        make_event(ts, SystemEventData::AppOpened { app_id })
    }

    fn app_closed(ts: f64, app_id: u16) -> SystemEvent {
        make_event(
            ts,
            SystemEventData::AppClosed {
                app_id,
                duration_ms: 5000,
            },
        )
    }

    fn suggestion_accepted(ts: f64) -> SystemEvent {
        make_event(
            ts,
            SystemEventData::SuggestionAccepted {
                suggestion_id: 1,
                action_kind: "test".to_string(),
                latency_ms: 500,
            },
        )
    }

    fn error_event(ts: f64) -> SystemEvent {
        make_event(
            ts,
            SystemEventData::ErrorOccurred {
                app_id: 1,
                error_type: "test_error".to_string(),
            },
        )
    }

    #[test]
    fn test_causal_store_new() {
        let store = CausalStore::new();
        assert_eq!(store.edge_count(), 0);
        assert_eq!(store.total_hypothesized, 0);
    }

    #[test]
    fn test_causal_store_upsert_and_find() {
        let mut store = CausalStore::new();

        let edge = CausalEdge {
            cause: CausalNode::Event(EventKind::AppOpened),
            effect: CausalNode::Event(EventKind::SuggestionAccepted),
            strength: 0.5,
            confidence: 0.7,
            observation_count: 10,
            intervention_count: 0,
            non_occurrence_count: 2,
            median_lag_secs: 5.0,
            lag_iqr_secs: 2.0,
            context_strengths: Vec::new(),
            trace: CausalTrace {
                evidence: vec![],
                primary_method: DiscoveryMethod::TemporalAssociation,
                summary: "test".to_string(),
            },
            created_at: 1000.0,
            updated_at: 1000.0,
            stage: CausalStage::Candidate,
        };

        store.upsert(edge);
        assert_eq!(store.edge_count(), 1);
        assert_eq!(store.total_hypothesized, 1);

        let found = store.find_edge(
            &CausalNode::Event(EventKind::AppOpened),
            &CausalNode::Event(EventKind::SuggestionAccepted),
        );
        assert!(found.is_some());
        assert_eq!(found.unwrap().strength, 0.5);

        // Upsert same pair updates, doesn't add.
        let mut updated = store.edges[0].clone();
        updated.strength = 0.8;
        store.upsert(updated);
        assert_eq!(store.edge_count(), 1);
        assert_eq!(store.edges[0].strength, 0.8);
    }

    #[test]
    fn test_effects_of_and_causes_of() {
        let mut store = CausalStore::new();

        let edge1 = CausalEdge {
            cause: CausalNode::Event(EventKind::AppOpened),
            effect: CausalNode::Event(EventKind::SuggestionAccepted),
            strength: 0.5,
            confidence: 0.7,
            observation_count: 10,
            intervention_count: 0,
            non_occurrence_count: 0,
            median_lag_secs: 3.0,
            lag_iqr_secs: 1.0,
            context_strengths: Vec::new(),
            trace: CausalTrace {
                evidence: vec![],
                primary_method: DiscoveryMethod::TemporalAssociation,
                summary: "test".to_string(),
            },
            created_at: 1000.0,
            updated_at: 1000.0,
            stage: CausalStage::Candidate,
        };

        let edge2 = CausalEdge {
            cause: CausalNode::Event(EventKind::AppOpened),
            effect: CausalNode::Event(EventKind::UserTyping),
            strength: 0.3,
            confidence: 0.5,
            observation_count: 5,
            intervention_count: 0,
            non_occurrence_count: 0,
            median_lag_secs: 2.0,
            lag_iqr_secs: 0.5,
            context_strengths: Vec::new(),
            trace: CausalTrace {
                evidence: vec![],
                primary_method: DiscoveryMethod::TemporalAssociation,
                summary: "test".to_string(),
            },
            created_at: 1000.0,
            updated_at: 1000.0,
            stage: CausalStage::Hypothesized,
        };

        store.upsert(edge1);
        store.upsert(edge2);

        let effects = store.effects_of(&CausalNode::Event(EventKind::AppOpened));
        assert_eq!(effects.len(), 2);

        let causes = store.causes_of(&CausalNode::Event(EventKind::SuggestionAccepted));
        assert_eq!(causes.len(), 1);
    }

    #[test]
    fn test_temporal_pattern_discovery() {
        // Create events: AppOpened consistently followed by SuggestionAccepted ~5s later.
        let mut events = Vec::new();
        for i in 0..10 {
            let base = 1000.0 + i as f64 * 60.0;
            events.push(app_opened(base, 1));
            events.push(suggestion_accepted(base + 5.0));
        }

        let refs: Vec<&SystemEvent> = events.iter().collect();
        let config = CausalConfig::default();
        let patterns = discover_temporal_patterns(&refs, &config, 5);

        assert!(!patterns.is_empty());

        let app_to_suggestion = patterns.iter().find(|p| {
            p.cause == CausalNode::Event(EventKind::AppOpened)
                && p.effect == CausalNode::Event(EventKind::SuggestionAccepted)
        });
        assert!(app_to_suggestion.is_some());

        let pat = app_to_suggestion.unwrap();
        assert!(pat.co_occurrences >= 5);
        assert!((pat.avg_lag_secs - 5.0).abs() < 1.0);
    }

    #[test]
    fn test_integrate_temporal_patterns() {
        let mut store = CausalStore::new();
        let pattern = TemporalPattern {
            cause: CausalNode::Event(EventKind::AppOpened),
            effect: CausalNode::Event(EventKind::UserTyping),
            co_occurrences: 8,
            avg_lag_secs: 3.0,
            lag_stddev_secs: 0.5,
            lags: vec![2.5, 2.8, 3.0, 3.1, 3.0, 2.9, 3.2, 3.5],
        };

        integrate_temporal_patterns(&mut store, vec![pattern], 2000.0);
        assert_eq!(store.edge_count(), 1);

        let edge = &store.edges[0];
        assert_eq!(edge.stage, CausalStage::Candidate); // 8 >= 5 (min_observations_candidate)
        assert!(edge.confidence > 0.0);
        assert!(edge.median_lag_secs > 0.0);
    }

    #[test]
    fn test_record_intervention_new() {
        let mut store = CausalStore::new();

        record_intervention(
            &mut store,
            ActionKind::SurfaceSuggestion,
            CausalNode::Event(EventKind::SuggestionAccepted),
            true,
            1000.0,
        );

        assert_eq!(store.edge_count(), 1);
        let edge = &store.edges[0];
        assert_eq!(edge.intervention_count, 1);
        assert_eq!(edge.non_occurrence_count, 0);
        assert!(edge.confidence > 0.5); // Successful intervention → positive posterior.
    }

    #[test]
    fn test_record_intervention_updates() {
        let mut store = CausalStore::new();
        let cause = CausalNode::Action(ActionKind::ExecuteTool);
        let effect = CausalNode::Event(EventKind::ToolCallCompleted);

        // 3 successes, 1 failure.
        record_intervention(
            &mut store,
            ActionKind::ExecuteTool,
            effect.clone(),
            true,
            1000.0,
        );
        record_intervention(
            &mut store,
            ActionKind::ExecuteTool,
            effect.clone(),
            true,
            1001.0,
        );
        record_intervention(
            &mut store,
            ActionKind::ExecuteTool,
            effect.clone(),
            true,
            1002.0,
        );
        record_intervention(
            &mut store,
            ActionKind::ExecuteTool,
            effect.clone(),
            false,
            1003.0,
        );

        assert_eq!(store.edge_count(), 1);
        let edge = store.find_edge(&cause, &effect).unwrap();
        assert_eq!(edge.intervention_count, 3);
        assert_eq!(edge.non_occurrence_count, 1);
        assert!(edge.confidence > 0.5); // 3/4 success rate.
    }

    #[test]
    fn test_granger_test_basic() {
        // Cause events at regular intervals.
        let cause_times: Vec<f64> = (0..20).map(|i| i as f64 * 10.0).collect();
        // Effect events follow cause with ~2s lag.
        let effect_times: Vec<f64> = (0..20).map(|i| i as f64 * 10.0 + 2.0).collect();

        let result = granger_test(&cause_times, &effect_times, 10.0, 15);
        assert!(result.is_some());
    }

    #[test]
    fn test_granger_test_insufficient_data() {
        let result = granger_test(&[1.0], &[2.0], 10.0, 5);
        assert!(result.is_none());
    }

    #[test]
    fn test_predict_effects_chain() {
        let mut store = CausalStore::new();

        // A → B → C chain.
        let a = CausalNode::Event(EventKind::AppOpened);
        let b = CausalNode::Event(EventKind::UserTyping);
        let c = CausalNode::Event(EventKind::SuggestionAccepted);

        let make_edge = |cause: CausalNode, effect: CausalNode, strength: f64| CausalEdge {
            cause,
            effect,
            strength,
            confidence: 0.8,
            observation_count: 20,
            intervention_count: 0,
            non_occurrence_count: 0,
            median_lag_secs: 3.0,
            lag_iqr_secs: 1.0,
            context_strengths: Vec::new(),
            trace: CausalTrace {
                evidence: vec![],
                primary_method: DiscoveryMethod::TemporalAssociation,
                summary: "test".to_string(),
            },
            created_at: 1000.0,
            updated_at: 1000.0,
            stage: CausalStage::Established,
        };

        store.upsert(make_edge(a.clone(), b.clone(), 0.7));
        store.upsert(make_edge(b.clone(), c.clone(), 0.6));

        let effects = predict_effects(&store, &a, 3);
        assert!(effects.len() >= 2);

        // B should be first (direct, stronger).
        assert_eq!(effects[0].node, b);
        assert_eq!(effects[0].hops, 1);

        // C should be second (indirect).
        assert_eq!(effects[1].node, c);
        assert_eq!(effects[1].hops, 2);
    }

    #[test]
    fn test_estimate_effect() {
        let mut store = CausalStore::new();

        let edge = CausalEdge {
            cause: CausalNode::Action(ActionKind::SurfaceSuggestion),
            effect: CausalNode::Event(EventKind::SuggestionAccepted),
            strength: 0.6,
            confidence: 0.8,
            observation_count: 15,
            intervention_count: 5,
            non_occurrence_count: 2,
            median_lag_secs: 4.0,
            lag_iqr_secs: 1.5,
            context_strengths: Vec::new(),
            trace: CausalTrace {
                evidence: vec![CausalEvidence::InterventionOutcome {
                    action: ActionKind::SurfaceSuggestion,
                    successes: 5,
                    failures: 2,
                    posterior_mean: 0.75,
                }],
                primary_method: DiscoveryMethod::Intervention,
                summary: "test".to_string(),
            },
            created_at: 1000.0,
            updated_at: 1000.0,
            stage: CausalStage::Established,
        };

        store.upsert(edge);

        let estimate = estimate_effect(
            &store,
            &CausalNode::Action(ActionKind::SurfaceSuggestion),
            &CausalNode::Event(EventKind::SuggestionAccepted),
        );
        assert!(estimate.is_some());

        let est = estimate.unwrap();
        assert_eq!(est.evidence_quality, EvidenceQuality::Interventional);
        assert_eq!(est.intervention_count, 5);
        assert_eq!(est.stage, CausalStage::Established);
    }

    #[test]
    fn test_decay_and_refutation() {
        let mut store = CausalStore::new();

        let edge = CausalEdge {
            cause: CausalNode::Event(EventKind::AppOpened),
            effect: CausalNode::Event(EventKind::ErrorOccurred),
            strength: 0.3,
            confidence: 0.25, // Just above refutation threshold.
            observation_count: 5,
            intervention_count: 0,
            non_occurrence_count: 3,
            median_lag_secs: 10.0,
            lag_iqr_secs: 5.0,
            context_strengths: Vec::new(),
            trace: CausalTrace {
                evidence: vec![],
                primary_method: DiscoveryMethod::TemporalAssociation,
                summary: "test".to_string(),
            },
            created_at: 1000.0,
            updated_at: 1000.0, // 100 days ago if now=8641000
            stage: CausalStage::Hypothesized,
        };

        store.upsert(edge);

        // Apply decay — 100 days of decay at 0.98/day.
        let now = 1000.0 + 86400.0 * 100.0;
        let refuted = apply_decay(&mut store, now);
        assert_eq!(refuted, 1);
        assert_eq!(store.edges[0].stage, CausalStage::Refuted);
    }

    #[test]
    fn test_prune_refuted() {
        let mut store = CausalStore::new();

        let make = |kind: EventKind, stage: CausalStage| CausalEdge {
            cause: CausalNode::Event(EventKind::AppOpened),
            effect: CausalNode::Event(kind),
            strength: 0.5,
            confidence: 0.5,
            observation_count: 5,
            intervention_count: 0,
            non_occurrence_count: 0,
            median_lag_secs: 1.0,
            lag_iqr_secs: 0.5,
            context_strengths: Vec::new(),
            trace: CausalTrace {
                evidence: vec![],
                primary_method: DiscoveryMethod::TemporalAssociation,
                summary: "test".to_string(),
            },
            created_at: 1000.0,
            updated_at: 1000.0,
            stage,
        };

        store.upsert(make(EventKind::UserTyping, CausalStage::Established));
        store.upsert(make(EventKind::ErrorOccurred, CausalStage::Refuted));
        store.upsert(make(EventKind::SuggestionAccepted, CausalStage::Candidate));

        assert_eq!(store.edge_count(), 3);
        let pruned = store.prune_refuted();
        assert_eq!(pruned, 1);
        assert_eq!(store.edge_count(), 2);
    }

    #[test]
    fn test_enforce_capacity() {
        let mut store = CausalStore::with_config(CausalConfig {
            max_edges: 2,
            ..CausalConfig::default()
        });

        let kinds = [
            EventKind::AppOpened,
            EventKind::AppClosed,
            EventKind::UserTyping,
        ];
        for (i, &kind) in kinds.iter().enumerate() {
            store.upsert(CausalEdge {
                cause: CausalNode::Event(EventKind::ErrorOccurred),
                effect: CausalNode::Event(kind),
                strength: 0.5,
                confidence: (i + 1) as f64 * 0.2,
                observation_count: 5,
                intervention_count: 0,
                non_occurrence_count: 0,
                median_lag_secs: 1.0,
                lag_iqr_secs: 0.5,
                context_strengths: Vec::new(),
                trace: CausalTrace {
                    evidence: vec![],
                    primary_method: DiscoveryMethod::TemporalAssociation,
                    summary: "test".to_string(),
                },
                created_at: 1000.0,
                updated_at: 1000.0,
                stage: CausalStage::Candidate,
            });
        }

        assert_eq!(store.edge_count(), 3);
        let pruned = store.enforce_capacity();
        assert_eq!(pruned, 1);
        assert_eq!(store.edge_count(), 2);

        // Weakest (lowest confidence) should be removed.
        let remaining_confs: Vec<f64> = store.edges.iter().map(|e| e.confidence).collect();
        assert!(remaining_confs.iter().all(|&c| c >= 0.4));
    }

    #[test]
    fn test_causal_summary() {
        let mut store = CausalStore::new();

        let make = |kind: EventKind, stage: CausalStage, conf: f64, obs: u32| CausalEdge {
            cause: CausalNode::Event(EventKind::AppOpened),
            effect: CausalNode::Event(kind),
            strength: 0.5,
            confidence: conf,
            observation_count: obs,
            intervention_count: 0,
            non_occurrence_count: 0,
            median_lag_secs: 1.0,
            lag_iqr_secs: 0.5,
            context_strengths: Vec::new(),
            trace: CausalTrace {
                evidence: vec![],
                primary_method: DiscoveryMethod::TemporalAssociation,
                summary: "test".to_string(),
            },
            created_at: 1000.0,
            updated_at: 1000.0,
            stage,
        };

        store.upsert(make(
            EventKind::UserTyping,
            CausalStage::Established,
            0.9,
            20,
        ));
        store.upsert(make(
            EventKind::ErrorOccurred,
            CausalStage::Hypothesized,
            0.3,
            3,
        ));
        store.upsert(make(
            EventKind::SuggestionAccepted,
            CausalStage::Candidate,
            0.6,
            8,
        ));

        let summary = causal_summary(&store);
        assert_eq!(summary.total_edges, 3);
        assert_eq!(summary.established, 1);
        assert_eq!(summary.hypothesized, 1);
        assert_eq!(summary.candidates, 1);
        assert!(summary.strongest_edge.is_some());
        assert_eq!(summary.strongest_edge.as_ref().unwrap().confidence, 0.9);
    }

    #[test]
    fn test_explain_causal_edge() {
        let mut store = CausalStore::new();

        let cause = CausalNode::Action(ActionKind::SurfaceSuggestion);
        let effect = CausalNode::Event(EventKind::SuggestionAccepted);

        let edge = CausalEdge {
            cause: cause.clone(),
            effect: effect.clone(),
            strength: 0.6,
            confidence: 0.8,
            observation_count: 15,
            intervention_count: 3,
            non_occurrence_count: 1,
            median_lag_secs: 4.0,
            lag_iqr_secs: 1.5,
            context_strengths: Vec::new(),
            trace: CausalTrace {
                evidence: vec![
                    CausalEvidence::TemporalPrecedence {
                        co_occurrences: 12,
                        avg_lag_secs: 4.0,
                        lag_stddev_secs: 1.0,
                    },
                    CausalEvidence::InterventionOutcome {
                        action: ActionKind::SurfaceSuggestion,
                        successes: 3,
                        failures: 1,
                        posterior_mean: 0.67,
                    },
                ],
                primary_method: DiscoveryMethod::Converged,
                summary: "test edge".to_string(),
            },
            created_at: 1000.0,
            updated_at: 2000.0,
            stage: CausalStage::Established,
        };

        store.upsert(edge);

        let explanation = explain_causal_edge(&store, &cause, &effect);
        assert!(explanation.is_some());

        let exp = explanation.unwrap();
        assert_eq!(exp.evidence_summary.len(), 2);
        assert_eq!(exp.primary_method, DiscoveryMethod::Converged);
        assert_eq!(exp.intervention_count, 3);
    }

    #[test]
    fn test_median_and_quartiles() {
        assert_eq!(median_f64(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0);
        assert_eq!(median_f64(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median_f64(&[]), 0.0);

        let (q1, q3) = quartiles_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(q1, 3.0);
        assert_eq!(q3, 7.0);
    }

    #[test]
    fn test_stage_transitions() {
        let config = CausalConfig::default();
        let mut edge = CausalEdge {
            cause: CausalNode::Event(EventKind::AppOpened),
            effect: CausalNode::Event(EventKind::UserTyping),
            strength: 0.5,
            confidence: 0.7,
            observation_count: 3,
            intervention_count: 0,
            non_occurrence_count: 0,
            median_lag_secs: 2.0,
            lag_iqr_secs: 0.5,
            context_strengths: Vec::new(),
            trace: CausalTrace {
                evidence: vec![],
                primary_method: DiscoveryMethod::TemporalAssociation,
                summary: "test".to_string(),
            },
            created_at: 1000.0,
            updated_at: 1000.0,
            stage: CausalStage::Hypothesized,
        };

        // Not enough observations → stays Hypothesized.
        update_stage(&mut edge, &config);
        assert_eq!(edge.stage, CausalStage::Hypothesized);

        // 5 observations → Candidate.
        edge.observation_count = 5;
        update_stage(&mut edge, &config);
        assert_eq!(edge.stage, CausalStage::Candidate);

        // 15 observations + confidence ≥ 0.6 → Established.
        edge.observation_count = 15;
        update_stage(&mut edge, &config);
        assert_eq!(edge.stage, CausalStage::Established);

        // Low confidence → Weakening.
        edge.confidence = 0.4;
        update_stage(&mut edge, &config);
        assert_eq!(edge.stage, CausalStage::Weakening);

        // Recovery.
        edge.confidence = 0.7;
        update_stage(&mut edge, &config);
        assert_eq!(edge.stage, CausalStage::Established);

        // Very low confidence → Refuted.
        edge.confidence = 0.1;
        edge.stage = CausalStage::Weakening;
        update_stage(&mut edge, &config);
        assert_eq!(edge.stage, CausalStage::Refuted);
    }

    #[test]
    fn test_graph_causal_extraction() {
        let edges = vec![
            CognitiveEdge {
                src: NodeId::new(crate::state::NodeKind::Belief, 1),
                dst: NodeId::new(crate::state::NodeKind::Goal, 1),
                kind: CognitiveEdgeKind::Causes,
                weight: 0.8,
                created_at_ms: 1000000,
                last_confirmed_ms: 2000000,
                observation_count: 10,
                confidence: 0.7,
            },
            CognitiveEdge {
                src: NodeId::new(crate::state::NodeKind::Belief, 2),
                dst: NodeId::new(crate::state::NodeKind::Risk, 1),
                kind: CognitiveEdgeKind::Prevents,
                weight: 0.6,
                created_at_ms: 1000000,
                last_confirmed_ms: 2000000,
                observation_count: 5,
                confidence: 0.5,
            },
            // Non-causal edge — should be skipped.
            CognitiveEdge {
                src: NodeId::new(crate::state::NodeKind::Entity, 1),
                dst: NodeId::new(crate::state::NodeKind::Entity, 2),
                kind: CognitiveEdgeKind::AssociatedWith,
                weight: 0.9,
                created_at_ms: 1000000,
                last_confirmed_ms: 2000000,
                observation_count: 20,
                confidence: 0.9,
            },
        ];

        let causal = extract_graph_causal_edges(&edges, 3000.0);
        assert_eq!(causal.len(), 2);
        assert!(causal[0].strength > 0.0); // Causes → positive
        assert!(causal[1].strength < 0.0); // Prevents → negative
    }

    #[test]
    fn test_discover_local_causality() {
        let mut store = CausalStore::new();
        let mut buffer = EventBuffer::new(1000);

        // Add patterned events: AppOpened → UserTyping ~3s later, 10 times.
        for i in 0..10 {
            let base = 1000.0 + i as f64 * 30.0;
            buffer.push(app_opened(base, 1));
            buffer.push(make_event(
                base + 3.0,
                SystemEventData::UserTyping {
                    app_id: 1,
                    duration_ms: 5000,
                    characters: 100,
                },
            ));
        }

        let graph_edges = vec![];
        let report = discover_local_causality(&mut store, &buffer, &graph_edges, 2000.0);

        assert!(report.total_edges > 0);
    }
}
