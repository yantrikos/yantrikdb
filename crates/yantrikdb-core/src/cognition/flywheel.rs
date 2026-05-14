//! Autonomous Learning Flywheel — forms beliefs from observed correlations.
//!
//! The flywheel operates on pure statistical inference from the passive
//! event observer (CK-3.1). No explicit user feedback required.
//!
//! # Belief Formation Pipeline
//!
//! ```text
//! Observer events ──► Pattern detection ──► Belief candidates (0.3)
//!                                              │
//!                          ┌───────────────────┤
//!                          ▼                   ▼
//!                  Confirming obs.      Contradicting obs.
//!                  (confidence ↑)      (confidence ↓)
//!                          │                   │
//!                          ▼                   ▼
//!                  Established (0.7)    Pruned (<0.2)
//!                          │
//!                          ▼
//!                  Certain (0.9) — used in reasoning
//! ```
//!
//! # Belief Categories
//!
//! - **Temporal**: "User dismisses after 10pm" (from circadian histogram)
//! - **Preference**: "Prefers 30min reminder lead" (from accept/reject rates)
//! - **Behavioral**: "Opens music after terminal" (from app sequences)
//! - **Productivity**: "Reviews take longer on Fridays" (from duration patterns)
//! - **Need**: "Repeatedly searches for X" (from query repetition)
//! - **System**: "Tool X fails 40% of time" (from tool call outcomes)

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::observer::{
    CircadianHistogram, DerivedSignals, EventBuffer, EventKind, ObserverState, SystemEvent,
    SystemEventData,
};

// ══════════════════════════════════════════════════════════════════════════════
// § 1  Autonomous Belief Types
// ══════════════════════════════════════════════════════════════════════════════

/// Unique identifier for an autonomous belief.
pub type BeliefId = u64;

/// Category of autonomously formed belief.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BeliefCategory {
    /// Time-of-day pattern (e.g., "less responsive after 10pm").
    Temporal,
    /// User preference (e.g., "prefers nudge over alert").
    Preference,
    /// App usage correlation (e.g., "music after terminal").
    Behavioral,
    /// Duration/productivity pattern (e.g., "slower on Fridays").
    Productivity,
    /// Repeated unresolved need (e.g., "keeps searching for X").
    Need,
    /// System reliability pattern (e.g., "tool X fails often").
    System,
}

impl BeliefCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Temporal => "temporal",
            Self::Preference => "preference",
            Self::Behavioral => "behavioral",
            Self::Productivity => "productivity",
            Self::Need => "need",
            Self::System => "system",
        }
    }
}

/// Confidence lifecycle stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeliefStage {
    /// Just formed, very tentative (confidence 0.2–0.5).
    Hypothesis,
    /// Gaining evidence (confidence 0.5–0.7).
    Emerging,
    /// Enough evidence to be used in reasoning (confidence 0.7–0.9).
    Established,
    /// Very strong evidence, no longer needs confirmation (confidence 0.9+).
    Certain,
    /// Contradicted, pending removal (confidence <0.2).
    Dying,
}

/// An autonomously formed belief atom.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomousBelief {
    /// Unique ID (hash-based, stable across re-derivation).
    pub id: BeliefId,
    /// Human-readable description of what was learned.
    pub description: String,
    /// Category of this belief.
    pub category: BeliefCategory,
    /// Current confidence [0.0, 1.0].
    pub confidence: f64,
    /// Current lifecycle stage (derived from confidence).
    pub stage: BeliefStage,
    /// Number of confirming observations.
    pub confirming_observations: u32,
    /// Number of contradicting observations.
    pub contradicting_observations: u32,
    /// Timestamp when first formed.
    pub formed_at: f64,
    /// Timestamp of last update.
    pub last_updated: f64,
    /// The dedup key — prevents duplicate beliefs about the same pattern.
    pub dedup_key: String,
    /// Quantitative evidence (category-specific).
    pub evidence: BeliefEvidence,
}

/// Category-specific evidence backing a belief.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BeliefEvidence {
    /// Temporal: hour-of-day distribution anomaly.
    Temporal {
        event_kind: EventKind,
        peak_hour: u8,
        quiet_hours: Vec<u8>,
        distribution_skew: f64,
    },
    /// Preference: accept/reject rate difference.
    Preference {
        preferred_value: String,
        accept_rate: f64,
        reject_rate: f64,
        sample_size: u32,
    },
    /// Behavioral: app sequence correlation.
    Behavioral {
        from_app: u16,
        to_app: u16,
        transition_count: u32,
        avg_gap_ms: f64,
    },
    /// Productivity: duration pattern.
    Productivity {
        app_id: u16,
        avg_duration_ms: f64,
        sample_size: u32,
    },
    /// Need: repeated query.
    Need { query_hash: u64, repeat_count: u32 },
    /// System: tool/LLM reliability.
    System {
        component: String,
        success_rate: f64,
        sample_size: u32,
    },
}

impl AutonomousBelief {
    /// Create a new belief at hypothesis stage.
    pub fn new(
        description: String,
        category: BeliefCategory,
        dedup_key: String,
        evidence: BeliefEvidence,
        now: f64,
    ) -> Self {
        let id = compute_belief_id(&dedup_key);
        Self {
            id,
            description,
            category,
            confidence: 0.3,
            stage: BeliefStage::Hypothesis,
            confirming_observations: 1,
            contradicting_observations: 0,
            formed_at: now,
            last_updated: now,
            dedup_key,
            evidence,
        }
    }

    /// Update confidence with a confirming observation.
    pub fn confirm(&mut self, now: f64) {
        self.confirming_observations += 1;
        // Bayesian-inspired: each confirmation moves confidence toward 1.0
        // with diminishing returns. Formula: c' = c + (1-c) * learning_rate
        let learning_rate = 0.05 / (1.0 + self.confirming_observations as f64 * 0.01);
        self.confidence = (self.confidence + (1.0 - self.confidence) * learning_rate).min(0.99);
        self.last_updated = now;
        self.update_stage();
    }

    /// Update confidence with a contradicting observation.
    pub fn contradict(&mut self, now: f64) {
        self.contradicting_observations += 1;
        // Each contradiction moves confidence toward 0.0
        let decay_rate = 0.08 / (1.0 + self.contradicting_observations as f64 * 0.01);
        self.confidence = (self.confidence - self.confidence * decay_rate).max(0.0);
        self.last_updated = now;
        self.update_stage();
    }

    /// Recalculate stage from current confidence.
    fn update_stage(&mut self) {
        self.stage = confidence_to_stage(self.confidence);
    }

    /// Whether this belief is established enough for reasoning.
    pub fn is_established(&self) -> bool {
        matches!(self.stage, BeliefStage::Established | BeliefStage::Certain)
    }

    /// Whether this belief should be pruned.
    pub fn should_prune(&self) -> bool {
        self.confidence < 0.15
    }

    /// Total observations (confirming + contradicting).
    pub fn total_observations(&self) -> u32 {
        self.confirming_observations + self.contradicting_observations
    }

    /// Agreement ratio: confirming / total.
    pub fn agreement_ratio(&self) -> f64 {
        let total = self.total_observations();
        if total == 0 {
            return 0.5;
        }
        self.confirming_observations as f64 / total as f64
    }

    /// Age in seconds.
    pub fn age_secs(&self, now: f64) -> f64 {
        now - self.formed_at
    }
}

/// Map confidence to lifecycle stage.
fn confidence_to_stage(confidence: f64) -> BeliefStage {
    if confidence >= 0.9 {
        BeliefStage::Certain
    } else if confidence >= 0.7 {
        BeliefStage::Established
    } else if confidence >= 0.5 {
        BeliefStage::Emerging
    } else if confidence >= 0.2 {
        BeliefStage::Hypothesis
    } else {
        BeliefStage::Dying
    }
}

/// Compute a stable ID from a dedup key.
fn compute_belief_id(dedup_key: &str) -> BeliefId {
    // Use FNV-1a for fast, stable hashing
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in dedup_key.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ══════════════════════════════════════════════════════════════════════════════
// § 2  Belief Store (In-Memory)
// ══════════════════════════════════════════════════════════════════════════════

/// Persistent store of all autonomously formed beliefs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefStore {
    /// Beliefs indexed by dedup_key.
    beliefs: HashMap<String, AutonomousBelief>,
    /// Total beliefs ever formed (including pruned).
    pub total_formed: u64,
    /// Total beliefs pruned.
    pub total_pruned: u64,
    /// Total beliefs that reached established stage.
    pub total_established: u64,
}

impl BeliefStore {
    pub fn new() -> Self {
        Self {
            beliefs: HashMap::new(),
            total_formed: 0,
            total_pruned: 0,
            total_established: 0,
        }
    }

    /// Insert or update a belief.
    pub fn upsert(&mut self, belief: AutonomousBelief) {
        if !self.beliefs.contains_key(&belief.dedup_key) {
            self.total_formed += 1;
        }
        if belief.is_established() {
            self.total_established += 1;
        }
        self.beliefs.insert(belief.dedup_key.clone(), belief);
    }

    /// Find a belief by its dedup key.
    pub fn find(&self, dedup_key: &str) -> Option<&AutonomousBelief> {
        self.beliefs.get(dedup_key)
    }

    /// Find a belief mutably by its dedup key.
    pub fn find_mut(&mut self, dedup_key: &str) -> Option<&mut AutonomousBelief> {
        self.beliefs.get_mut(dedup_key)
    }

    /// Get all beliefs above a minimum confidence.
    pub fn beliefs_above(&self, min_confidence: f64) -> Vec<&AutonomousBelief> {
        self.beliefs
            .values()
            .filter(|b| b.confidence >= min_confidence)
            .collect()
    }

    /// Get all established beliefs (confidence >= 0.7).
    pub fn established(&self) -> Vec<&AutonomousBelief> {
        self.beliefs_above(0.7)
    }

    /// Get all belief candidates (hypothesis + emerging, not yet established).
    pub fn candidates(&self) -> Vec<&AutonomousBelief> {
        self.beliefs
            .values()
            .filter(|b| matches!(b.stage, BeliefStage::Hypothesis | BeliefStage::Emerging))
            .collect()
    }

    /// Get beliefs by category.
    pub fn by_category(&self, category: BeliefCategory) -> Vec<&AutonomousBelief> {
        self.beliefs
            .values()
            .filter(|b| b.category == category)
            .collect()
    }

    /// Prune dying beliefs. Returns count removed.
    pub fn prune(&mut self) -> usize {
        let before = self.beliefs.len();
        self.beliefs.retain(|_, b| !b.should_prune());
        let pruned = before - self.beliefs.len();
        self.total_pruned += pruned as u64;
        pruned
    }

    /// Total active beliefs.
    pub fn len(&self) -> usize {
        self.beliefs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.beliefs.is_empty()
    }

    /// Iterate over all beliefs.
    pub fn iter(&self) -> impl Iterator<Item = &AutonomousBelief> {
        self.beliefs.values()
    }
}

impl Default for BeliefStore {
    fn default() -> Self {
        Self::new()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  Flywheel Configuration
// ══════════════════════════════════════════════════════════════════════════════

/// Configuration for the autonomous belief formation pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlywheelConfig {
    /// Minimum observations before a temporal pattern becomes a belief.
    pub min_temporal_observations: u32,
    /// Minimum distribution skew for temporal beliefs (z-score).
    pub min_temporal_skew: f64,
    /// Minimum observations for preference beliefs.
    pub min_preference_observations: u32,
    /// Minimum rate difference for preference beliefs.
    pub min_preference_rate_diff: f64,
    /// Minimum transition count for behavioral beliefs.
    pub min_behavioral_transitions: u32,
    /// Minimum repeat count for need detection.
    pub min_need_repeats: u32,
    /// Minimum observations for system reliability beliefs.
    pub min_system_observations: u32,
    /// Maximum active beliefs (prevents unbounded growth).
    pub max_beliefs: usize,
    /// Prune beliefs older than this (seconds) if not established.
    pub candidate_ttl_secs: f64,
}

impl Default for FlywheelConfig {
    fn default() -> Self {
        Self {
            min_temporal_observations: 20,
            min_temporal_skew: 2.0,
            min_preference_observations: 10,
            min_preference_rate_diff: 0.2,
            min_behavioral_transitions: 5,
            min_need_repeats: 3,
            min_system_observations: 10,
            max_beliefs: 500,
            candidate_ttl_secs: 7.0 * 86400.0, // 7 days
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  Belief Formation Pipeline (Pure Functions)
// ══════════════════════════════════════════════════════════════════════════════

/// Result of a belief formation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormationResult {
    /// Newly created beliefs.
    pub new_beliefs: Vec<AutonomousBelief>,
    /// Existing beliefs that were confirmed.
    pub confirmed: u32,
    /// Existing beliefs that were contradicted.
    pub contradicted: u32,
    /// Beliefs pruned for low confidence.
    pub pruned: u32,
    /// Beliefs pruned for expiry (old candidates).
    pub expired: u32,
}

/// Run the full belief formation pipeline.
///
/// Analyzes the event buffer and observer state to detect patterns,
/// then creates or updates beliefs in the store.
pub fn form_beliefs(
    buffer: &EventBuffer,
    state: &ObserverState,
    store: &mut BeliefStore,
    config: &FlywheelConfig,
    now: f64,
) -> FormationResult {
    let mut result = FormationResult {
        new_beliefs: Vec::new(),
        confirmed: 0,
        contradicted: 0,
        pruned: 0,
        expired: 0,
    };

    // Phase 1: Detect temporal patterns from circadian histogram
    detect_temporal_beliefs(&state.histogram, store, config, now, &mut result);

    // Phase 2: Detect preference patterns from suggestion outcomes
    detect_preference_beliefs(buffer, store, config, now, &mut result);

    // Phase 3: Detect behavioral patterns from app sequences
    detect_behavioral_beliefs(buffer, store, config, now, &mut result);

    // Phase 4: Detect need patterns from repeated queries
    detect_need_beliefs(buffer, store, config, now, &mut result);

    // Phase 5: Detect system reliability patterns
    detect_system_beliefs(buffer, store, config, now, &mut result);

    // Phase 6: Prune dying and expired beliefs
    result.pruned = store.prune() as u32;
    result.expired = expire_old_candidates(store, config, now) as u32;

    // Phase 7: Enforce max beliefs (evict lowest confidence)
    enforce_max_beliefs(store, config);

    result
}

// ══════════════════════════════════════════════════════════════════════════════
// § 5  Pattern Detectors
// ══════════════════════════════════════════════════════════════════════════════

/// Detect temporal patterns from the circadian histogram.
fn detect_temporal_beliefs(
    histogram: &CircadianHistogram,
    store: &mut BeliefStore,
    config: &FlywheelConfig,
    now: f64,
    result: &mut FormationResult,
) {
    for &kind in &EventKind::ALL {
        let dist = histogram.distribution(kind);
        let total: u32 = dist.iter().sum();
        if total < config.min_temporal_observations {
            continue;
        }

        let mean = total as f64 / 24.0;
        if mean < 0.5 {
            continue;
        }

        // Compute standard deviation
        let variance: f64 = dist
            .iter()
            .map(|&c| {
                let diff = c as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / 24.0;
        let std_dev = variance.sqrt();

        if std_dev < 0.1 {
            continue; // Uniform — no pattern
        }

        let skew = std_dev / mean;
        if skew < config.min_temporal_skew * 0.1 {
            continue;
        }

        // Find peak and quiet hours
        let peak_hour = dist
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(h, _)| h as u8)
            .unwrap_or(0);
        let quiet_hours: Vec<u8> = dist
            .iter()
            .enumerate()
            .filter(|(_, &c)| (c as f64) < mean * 0.3)
            .map(|(h, _)| h as u8)
            .collect();

        let dedup_key = format!("temporal:{}:peak{}", kind.as_str(), peak_hour);
        let description = format!(
            "{} peaks at hour {} ({}× avg)",
            kind.as_str(),
            peak_hour,
            format_ratio(dist[peak_hour as usize] as f64, mean),
        );

        upsert_or_confirm(
            store,
            &dedup_key,
            || {
                AutonomousBelief::new(
                    description.clone(),
                    BeliefCategory::Temporal,
                    dedup_key.clone(),
                    BeliefEvidence::Temporal {
                        event_kind: kind,
                        peak_hour,
                        quiet_hours,
                        distribution_skew: skew,
                    },
                    now,
                )
            },
            now,
            result,
        );
    }
}

/// Detect preference patterns from suggestion accept/reject rates.
fn detect_preference_beliefs(
    buffer: &EventBuffer,
    store: &mut BeliefStore,
    config: &FlywheelConfig,
    now: f64,
    result: &mut FormationResult,
) {
    // Count accept/reject per action_kind
    let mut accept_counts: HashMap<String, u32> = HashMap::new();
    let mut reject_counts: HashMap<String, u32> = HashMap::new();

    for event in buffer.iter() {
        match &event.data {
            SystemEventData::SuggestionAccepted { action_kind, .. } => {
                *accept_counts.entry(action_kind.clone()).or_insert(0) += 1;
            }
            SystemEventData::SuggestionRejected { action_kind, .. } => {
                *reject_counts.entry(action_kind.clone()).or_insert(0) += 1;
            }
            _ => {}
        }
    }

    // Find action kinds with strong preference signal
    let all_kinds: Vec<String> = accept_counts
        .keys()
        .chain(reject_counts.keys())
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    for kind in all_kinds {
        let accepted = *accept_counts.get(&kind).unwrap_or(&0);
        let rejected = *reject_counts.get(&kind).unwrap_or(&0);
        let total = accepted + rejected;

        if total < config.min_preference_observations {
            continue;
        }

        let accept_rate = accepted as f64 / total as f64;
        let reject_rate = rejected as f64 / total as f64;

        // Strong preference: accept rate > 0.7 or < 0.3
        if (accept_rate - 0.5).abs() < config.min_preference_rate_diff {
            continue;
        }

        let (description, preferred) = if accept_rate > 0.5 {
            (
                format!(
                    "User accepts '{}' suggestions {:.0}% of time",
                    kind,
                    accept_rate * 100.0
                ),
                kind.clone(),
            )
        } else {
            (
                format!(
                    "User rejects '{}' suggestions {:.0}% of time",
                    kind,
                    reject_rate * 100.0
                ),
                kind.clone(),
            )
        };

        let dedup_key = format!("preference:suggestion:{}", kind);

        upsert_or_confirm(
            store,
            &dedup_key,
            || {
                AutonomousBelief::new(
                    description.clone(),
                    BeliefCategory::Preference,
                    dedup_key.clone(),
                    BeliefEvidence::Preference {
                        preferred_value: preferred.clone(),
                        accept_rate,
                        reject_rate,
                        sample_size: total,
                    },
                    now,
                )
            },
            now,
            result,
        );
    }
}

/// Detect behavioral patterns from app transition sequences.
fn detect_behavioral_beliefs(
    buffer: &EventBuffer,
    store: &mut BeliefStore,
    config: &FlywheelConfig,
    now: f64,
    result: &mut FormationResult,
) {
    // Count transitions: (from_app, to_app) → (count, total_gap_ms)
    let mut transitions: HashMap<(u16, u16), (u32, u64)> = HashMap::new();

    for event in buffer.iter() {
        if let SystemEventData::AppSequence {
            from_app,
            to_app,
            gap_ms,
        } = &event.data
        {
            let entry = transitions.entry((*from_app, *to_app)).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += gap_ms;
        }
    }

    for ((from_app, to_app), (count, total_gap)) in &transitions {
        if *count < config.min_behavioral_transitions {
            continue;
        }

        let avg_gap_ms = *total_gap as f64 / *count as f64;
        let dedup_key = format!("behavioral:app_seq:{}→{}", from_app, to_app);
        let description = format!(
            "After app {} user opens app {} ({} times, avg {:.1}s gap)",
            from_app,
            to_app,
            count,
            avg_gap_ms / 1000.0
        );

        upsert_or_confirm(
            store,
            &dedup_key,
            || {
                AutonomousBelief::new(
                    description.clone(),
                    BeliefCategory::Behavioral,
                    dedup_key.clone(),
                    BeliefEvidence::Behavioral {
                        from_app: *from_app,
                        to_app: *to_app,
                        transition_count: *count,
                        avg_gap_ms,
                    },
                    now,
                )
            },
            now,
            result,
        );
    }
}

/// Detect repeated queries (unresolved needs).
fn detect_need_beliefs(
    buffer: &EventBuffer,
    store: &mut BeliefStore,
    config: &FlywheelConfig,
    now: f64,
    result: &mut FormationResult,
) {
    let mut query_counts: HashMap<u64, u32> = HashMap::new();

    for event in buffer.iter() {
        if let SystemEventData::QueryRepeated {
            query_hash, count, ..
        } = &event.data
        {
            *query_counts.entry(*query_hash).or_insert(0) += count;
        }
    }

    for (hash, count) in &query_counts {
        if *count < config.min_need_repeats {
            continue;
        }

        let dedup_key = format!("need:query:{:x}", hash);
        let description = format!(
            "Repeated query (hash {:x}) {} times — unresolved need",
            hash, count
        );

        upsert_or_confirm(
            store,
            &dedup_key,
            || {
                AutonomousBelief::new(
                    description.clone(),
                    BeliefCategory::Need,
                    dedup_key.clone(),
                    BeliefEvidence::Need {
                        query_hash: *hash,
                        repeat_count: *count,
                    },
                    now,
                )
            },
            now,
            result,
        );
    }
}

/// Detect system reliability patterns from tool call outcomes.
fn detect_system_beliefs(
    buffer: &EventBuffer,
    store: &mut BeliefStore,
    config: &FlywheelConfig,
    now: f64,
    result: &mut FormationResult,
) {
    // Track per-tool success rates
    let mut tool_stats: HashMap<String, (u32, u32)> = HashMap::new(); // (success, total)

    for event in buffer.iter() {
        if let SystemEventData::ToolCallCompleted {
            tool_name, success, ..
        } = &event.data
        {
            let entry = tool_stats.entry(tool_name.clone()).or_insert((0, 0));
            if *success {
                entry.0 += 1;
            }
            entry.1 += 1;
        }
    }

    for (tool, (success, total)) in &tool_stats {
        if *total < config.min_system_observations {
            continue;
        }

        let success_rate = *success as f64 / *total as f64;

        // Only form beliefs about notable reliability patterns
        if success_rate > 0.95 || (success_rate > 0.5 && success_rate < 0.95) {
            // Reliable or moderately unreliable — not interesting enough
            // Only flag significantly unreliable tools
            if success_rate > 0.5 {
                continue;
            }
        }

        let dedup_key = format!("system:tool_reliability:{}", tool);
        let description = format!(
            "Tool '{}' has {:.0}% success rate ({}/{})",
            tool,
            success_rate * 100.0,
            success,
            total
        );

        upsert_or_confirm(
            store,
            &dedup_key,
            || {
                AutonomousBelief::new(
                    description.clone(),
                    BeliefCategory::System,
                    dedup_key.clone(),
                    BeliefEvidence::System {
                        component: tool.clone(),
                        success_rate,
                        sample_size: *total,
                    },
                    now,
                )
            },
            now,
            result,
        );
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 6  Helper Functions
// ══════════════════════════════════════════════════════════════════════════════

/// Upsert or confirm an existing belief.
fn upsert_or_confirm<F>(
    store: &mut BeliefStore,
    dedup_key: &str,
    create: F,
    now: f64,
    result: &mut FormationResult,
) where
    F: FnOnce() -> AutonomousBelief,
{
    if let Some(existing) = store.find_mut(dedup_key) {
        existing.confirm(now);
        result.confirmed += 1;
    } else {
        let belief = create();
        result.new_beliefs.push(belief.clone());
        store.upsert(belief);
    }
}

/// Remove old candidates that never reached established stage.
fn expire_old_candidates(store: &mut BeliefStore, config: &FlywheelConfig, now: f64) -> usize {
    let before = store.len();
    store.beliefs.retain(|_, b| {
        if b.is_established() {
            return true; // Never expire established beliefs
        }
        b.age_secs(now) < config.candidate_ttl_secs
    });
    before - store.len()
}

/// Enforce maximum belief count by evicting lowest confidence.
fn enforce_max_beliefs(store: &mut BeliefStore, config: &FlywheelConfig) {
    if store.len() <= config.max_beliefs {
        return;
    }

    let mut entries: Vec<(String, f64)> = store
        .beliefs
        .iter()
        .map(|(k, b)| (k.clone(), b.confidence))
        .collect();
    entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let to_remove = store.len() - config.max_beliefs;
    for (key, _) in entries.into_iter().take(to_remove) {
        store.beliefs.remove(&key);
    }
}

/// Format a ratio for display (e.g., "2.5×").
fn format_ratio(value: f64, baseline: f64) -> String {
    if baseline <= 0.0 {
        return "∞".to_string();
    }
    format!("{:.1}", value / baseline)
}

// ══════════════════════════════════════════════════════════════════════════════
// § 7  Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observer::{
        EventBuffer, ObserverConfig, ObserverState, SystemEvent, SystemEventData,
    };

    fn ts(offset: f64) -> f64 {
        86400.0 * 100.0 + offset
    }

    // ── Belief Lifecycle ──

    #[test]
    fn test_belief_creation() {
        let belief = AutonomousBelief::new(
            "Test belief".to_string(),
            BeliefCategory::Temporal,
            "test:key".to_string(),
            BeliefEvidence::Temporal {
                event_kind: EventKind::AppOpened,
                peak_hour: 10,
                quiet_hours: vec![2, 3, 4],
                distribution_skew: 1.5,
            },
            ts(0.0),
        );

        assert_eq!(belief.confidence, 0.3);
        assert_eq!(belief.stage, BeliefStage::Hypothesis);
        assert!(!belief.is_established());
        assert!(!belief.should_prune());
    }

    #[test]
    fn test_belief_confirm_lifecycle() {
        let mut belief = AutonomousBelief::new(
            "Test".to_string(),
            BeliefCategory::Preference,
            "test".to_string(),
            BeliefEvidence::Preference {
                preferred_value: "nudge".to_string(),
                accept_rate: 0.8,
                reject_rate: 0.2,
                sample_size: 20,
            },
            ts(0.0),
        );

        // Confirm many times — confidence should rise
        for i in 1..=50 {
            belief.confirm(ts(i as f64));
        }

        assert!(
            belief.confidence > 0.7,
            "After 50 confirmations, confidence ({:.3}) should exceed 0.7",
            belief.confidence,
        );
        assert!(belief.is_established());
    }

    #[test]
    fn test_belief_contradict_lifecycle() {
        let mut belief = AutonomousBelief::new(
            "Test".to_string(),
            BeliefCategory::Behavioral,
            "test".to_string(),
            BeliefEvidence::Behavioral {
                from_app: 1,
                to_app: 2,
                transition_count: 5,
                avg_gap_ms: 3000.0,
            },
            ts(0.0),
        );

        // Contradict many times — confidence should drop
        for i in 1..=30 {
            belief.contradict(ts(i as f64));
        }

        assert!(
            belief.confidence < 0.2,
            "After 30 contradictions, confidence ({:.3}) should be below 0.2",
            belief.confidence,
        );
        assert!(belief.should_prune());
    }

    // ── Belief Store ──

    #[test]
    fn test_store_upsert_and_query() {
        let mut store = BeliefStore::new();

        let b1 = AutonomousBelief::new(
            "Belief A".to_string(),
            BeliefCategory::Temporal,
            "a".to_string(),
            BeliefEvidence::Temporal {
                event_kind: EventKind::AppOpened,
                peak_hour: 10,
                quiet_hours: vec![],
                distribution_skew: 1.0,
            },
            ts(0.0),
        );
        store.upsert(b1);
        assert_eq!(store.len(), 1);
        assert_eq!(store.total_formed, 1);

        let found = store.find("a").unwrap();
        assert_eq!(found.description, "Belief A");
    }

    #[test]
    fn test_store_established_filter() {
        let mut store = BeliefStore::new();

        let mut high = AutonomousBelief::new(
            "High".to_string(),
            BeliefCategory::Temporal,
            "high".to_string(),
            BeliefEvidence::Temporal {
                event_kind: EventKind::AppOpened,
                peak_hour: 10,
                quiet_hours: vec![],
                distribution_skew: 1.0,
            },
            ts(0.0),
        );
        // Boost to established
        for i in 0..50 {
            high.confirm(ts(i as f64));
        }
        store.upsert(high);

        let low = AutonomousBelief::new(
            "Low".to_string(),
            BeliefCategory::Temporal,
            "low".to_string(),
            BeliefEvidence::Temporal {
                event_kind: EventKind::AppClosed,
                peak_hour: 15,
                quiet_hours: vec![],
                distribution_skew: 0.5,
            },
            ts(0.0),
        );
        store.upsert(low);

        assert_eq!(store.established().len(), 1);
        assert_eq!(store.candidates().len(), 1);
    }

    #[test]
    fn test_store_prune() {
        let mut store = BeliefStore::new();

        let mut dying = AutonomousBelief::new(
            "Dying".to_string(),
            BeliefCategory::Need,
            "dying".to_string(),
            BeliefEvidence::Need {
                query_hash: 42,
                repeat_count: 3,
            },
            ts(0.0),
        );
        // Contradict until dying
        for i in 0..30 {
            dying.contradict(ts(i as f64));
        }
        store.upsert(dying);

        let pruned = store.prune();
        assert_eq!(pruned, 1);
        assert_eq!(store.len(), 0);
    }

    // ── Formation Pipeline ──

    #[test]
    fn test_temporal_belief_formation() {
        let mut store = BeliefStore::new();
        let config = FlywheelConfig {
            min_temporal_observations: 10,
            ..Default::default()
        };

        // Build a histogram with strong peak at hour 10
        let mut histogram = CircadianHistogram::new();
        for _ in 0..20 {
            histogram.record(EventKind::AppOpened, 10.0 * 3600.0); // 10am
        }
        for _ in 0..2 {
            histogram.record(EventKind::AppOpened, 15.0 * 3600.0); // 3pm (low)
        }

        let mut result = FormationResult {
            new_beliefs: Vec::new(),
            confirmed: 0,
            contradicted: 0,
            pruned: 0,
            expired: 0,
        };
        detect_temporal_beliefs(&histogram, &mut store, &config, ts(0.0), &mut result);

        assert!(
            !result.new_beliefs.is_empty(),
            "Should detect temporal peak at hour 10",
        );
        assert_eq!(result.new_beliefs[0].category, BeliefCategory::Temporal);
    }

    #[test]
    fn test_preference_belief_formation() {
        let mut store = BeliefStore::new();
        let config = FlywheelConfig {
            min_preference_observations: 5,
            ..Default::default()
        };

        let mut buffer = EventBuffer::new(100);
        // Strong acceptance for "nudge"
        for i in 0..8 {
            buffer.push(SystemEvent::new(
                ts(i as f64),
                SystemEventData::SuggestionAccepted {
                    suggestion_id: i,
                    action_kind: "nudge".to_string(),
                    latency_ms: 500,
                },
            ));
        }
        // Strong rejection for "nudge" — wait, let's keep it simple
        for i in 0..2 {
            buffer.push(SystemEvent::new(
                ts(10.0 + i as f64),
                SystemEventData::SuggestionRejected {
                    suggestion_id: 100 + i,
                    action_kind: "nudge".to_string(),
                },
            ));
        }

        let mut result = FormationResult {
            new_beliefs: Vec::new(),
            confirmed: 0,
            contradicted: 0,
            pruned: 0,
            expired: 0,
        };
        detect_preference_beliefs(&buffer, &mut store, &config, ts(20.0), &mut result);

        assert!(
            !result.new_beliefs.is_empty(),
            "Should detect preference for 'nudge' (80% accept)",
        );
    }

    #[test]
    fn test_behavioral_belief_formation() {
        let mut store = BeliefStore::new();
        let config = FlywheelConfig {
            min_behavioral_transitions: 3,
            ..Default::default()
        };

        let mut buffer = EventBuffer::new(100);
        // Repeated: terminal (14) → music (20) sequence
        for i in 0..5 {
            buffer.push(SystemEvent::new(
                ts(i as f64 * 10.0),
                SystemEventData::AppSequence {
                    from_app: 14,
                    to_app: 20,
                    gap_ms: 5000,
                },
            ));
        }

        let mut result = FormationResult {
            new_beliefs: Vec::new(),
            confirmed: 0,
            contradicted: 0,
            pruned: 0,
            expired: 0,
        };
        detect_behavioral_beliefs(&buffer, &mut store, &config, ts(100.0), &mut result);

        assert!(
            !result.new_beliefs.is_empty(),
            "Should detect terminal→music behavioral pattern",
        );
        assert_eq!(result.new_beliefs[0].category, BeliefCategory::Behavioral);
    }

    #[test]
    fn test_need_belief_formation() {
        let mut store = BeliefStore::new();
        let config = FlywheelConfig::default();

        let mut buffer = EventBuffer::new(100);
        for i in 0..5 {
            buffer.push(SystemEvent::new(
                ts(i as f64),
                SystemEventData::QueryRepeated {
                    query_hash: 0xDEAD,
                    count: 1,
                },
            ));
        }

        let mut result = FormationResult {
            new_beliefs: Vec::new(),
            confirmed: 0,
            contradicted: 0,
            pruned: 0,
            expired: 0,
        };
        detect_need_beliefs(&buffer, &mut store, &config, ts(10.0), &mut result);

        assert!(
            !result.new_beliefs.is_empty(),
            "Should detect repeated query as unresolved need",
        );
    }

    #[test]
    fn test_system_belief_formation() {
        let mut store = BeliefStore::new();
        let config = FlywheelConfig {
            min_system_observations: 5,
            ..Default::default()
        };

        let mut buffer = EventBuffer::new(100);
        // Tool "flaky_api" fails 80% of time
        for i in 0..10 {
            buffer.push(SystemEvent::new(
                ts(i as f64),
                SystemEventData::ToolCallCompleted {
                    tool_name: "flaky_api".to_string(),
                    success: i < 2, // Only 2 successes out of 10
                    duration_ms: 500,
                },
            ));
        }

        let mut result = FormationResult {
            new_beliefs: Vec::new(),
            confirmed: 0,
            contradicted: 0,
            pruned: 0,
            expired: 0,
        };
        detect_system_beliefs(&buffer, &mut store, &config, ts(20.0), &mut result);

        assert!(
            !result.new_beliefs.is_empty(),
            "Should detect unreliable tool 'flaky_api'",
        );
        assert_eq!(result.new_beliefs[0].category, BeliefCategory::System);
    }

    // ── Full Pipeline ──

    #[test]
    fn test_full_formation_pipeline() {
        let mut buffer = EventBuffer::new(1000);
        let mut state = ObserverState::new();
        let mut store = BeliefStore::new();
        let config = FlywheelConfig {
            min_temporal_observations: 5,
            min_preference_observations: 3,
            min_behavioral_transitions: 2,
            ..Default::default()
        };

        // Seed the histogram with temporal data
        for _ in 0..10 {
            state.histogram.record(EventKind::AppOpened, 9.0 * 3600.0);
        }

        // Seed buffer with preference data
        for i in 0..5 {
            buffer.push(SystemEvent::new(
                ts(i as f64),
                SystemEventData::SuggestionAccepted {
                    suggestion_id: i,
                    action_kind: "whisper".to_string(),
                    latency_ms: 300,
                },
            ));
        }

        // Seed buffer with behavioral data
        for i in 0..3 {
            buffer.push(SystemEvent::new(
                ts(10.0 + i as f64),
                SystemEventData::AppSequence {
                    from_app: 12,
                    to_app: 15,
                    gap_ms: 2000,
                },
            ));
        }

        let result = form_beliefs(&buffer, &state, &mut store, &config, ts(20.0));

        assert!(
            !result.new_beliefs.is_empty(),
            "Full pipeline should produce beliefs",
        );
        assert!(store.len() > 0);
    }

    #[test]
    fn test_confirm_existing_belief() {
        let mut buffer = EventBuffer::new(100);
        let state = ObserverState::new();
        let mut store = BeliefStore::new();
        let config = FlywheelConfig {
            min_behavioral_transitions: 2,
            ..Default::default()
        };

        // First run: create belief
        for i in 0..3 {
            buffer.push(SystemEvent::new(
                ts(i as f64),
                SystemEventData::AppSequence {
                    from_app: 1,
                    to_app: 2,
                    gap_ms: 1000,
                },
            ));
        }
        let r1 = form_beliefs(&buffer, &state, &mut store, &config, ts(10.0));
        assert_eq!(r1.new_beliefs.len(), 1);

        // Second run: should confirm, not create new
        let r2 = form_beliefs(&buffer, &state, &mut store, &config, ts(20.0));
        assert_eq!(r2.new_beliefs.len(), 0);
        assert!(r2.confirmed > 0);

        // Belief confidence should have increased
        let belief = store.find("behavioral:app_seq:1→2").unwrap();
        assert!(belief.confidence > 0.3);
    }

    #[test]
    fn test_expire_old_candidates() {
        let mut store = BeliefStore::new();
        let config = FlywheelConfig {
            candidate_ttl_secs: 100.0, // 100 seconds for testing
            ..Default::default()
        };

        let belief = AutonomousBelief::new(
            "Old".to_string(),
            BeliefCategory::Need,
            "old".to_string(),
            BeliefEvidence::Need {
                query_hash: 1,
                repeat_count: 3,
            },
            ts(0.0),
        );
        store.upsert(belief);

        let expired = expire_old_candidates(&mut store, &config, ts(200.0));
        assert_eq!(expired, 1);
        assert!(store.is_empty());
    }

    #[test]
    fn test_enforce_max_beliefs() {
        let mut store = BeliefStore::new();
        let config = FlywheelConfig {
            max_beliefs: 3,
            ..Default::default()
        };

        for i in 0..5 {
            let mut b = AutonomousBelief::new(
                format!("Belief {}", i),
                BeliefCategory::Temporal,
                format!("b{}", i),
                BeliefEvidence::Temporal {
                    event_kind: EventKind::AppOpened,
                    peak_hour: i as u8,
                    quiet_hours: vec![],
                    distribution_skew: 1.0,
                },
                ts(0.0),
            );
            b.confidence = 0.3 + (i as f64 * 0.1); // 0.3, 0.4, 0.5, 0.6, 0.7
            store.upsert(b);
        }

        enforce_max_beliefs(&mut store, &config);
        assert_eq!(store.len(), 3);

        // Lowest confidence beliefs should be evicted
        assert!(store.find("b0").is_none()); // 0.3
        assert!(store.find("b1").is_none()); // 0.4
        assert!(store.find("b4").is_some()); // 0.7 — kept
    }

    // ── Dedup Key ──

    #[test]
    fn test_belief_id_stability() {
        let id1 = compute_belief_id("temporal:app_opened:peak10");
        let id2 = compute_belief_id("temporal:app_opened:peak10");
        let id3 = compute_belief_id("temporal:app_opened:peak11");

        assert_eq!(id1, id2); // Same key → same ID
        assert_ne!(id1, id3); // Different key → different ID
    }
}
