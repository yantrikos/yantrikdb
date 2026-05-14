//! Autonomous Skill Acquisition — learns reusable multi-step capabilities
//! from observed user action patterns.
//!
//! # Architecture
//!
//! ```text
//! Observer events ──► Sequence mining ──► SkillCandidate (confidence 0.3)
//!                          │                     │
//!                          │            Trigger context inference
//!                          │                     │
//!                          ▼                     ▼
//!                   Repeated use         LearnedSkill (confidence ↑)
//!                          │                     │
//!                          ▼                     ▼
//!                   Reliable (0.7)        SkillTemplate (reusable)
//!                          │
//!                          ▼
//!                   ActionSchema node persisted in graph
//! ```
//!
//! # Skill Lifecycle
//!
//! 1. **Discovery**: Detect repeated action sequences from observer events
//! 2. **Candidacy**: Form SkillCandidate with inferred trigger context
//! 3. **Validation**: Track success/failure when skill is offered and executed
//! 4. **Promotion**: Reliable skills become ActionSchema nodes in the graph
//! 5. **Deprecation**: Skills unused for extended periods are marked stale
//!
//! # Privacy
//! - Skills reference action_kind strings and app_ids — never raw user content
//! - Query hashes used for need-based skills (BLAKE3, irreversible)
//! - All data local, never transmitted

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::observer::{EventBuffer, EventKind, SystemEvent, SystemEventData};

// ══════════════════════════════════════════════════════════════════════════════
// § 1  Skill Types
// ══════════════════════════════════════════════════════════════════════════════

/// Unique identifier for a learned skill.
pub type SkillId = u64;

/// How the skill was discovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SkillOrigin {
    /// Mined from repeated app transition sequences.
    AppSequence,
    /// Mined from repeated tool call chains.
    ToolChain,
    /// Mined from repeated suggestion-accept patterns.
    SuggestionPattern,
    /// Mined from notification → action response patterns.
    NotificationResponse,
    /// Manually taught by the user.
    UserTaught,
}

impl SkillOrigin {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AppSequence => "app_sequence",
            Self::ToolChain => "tool_chain",
            Self::SuggestionPattern => "suggestion_pattern",
            Self::NotificationResponse => "notification_response",
            Self::UserTaught => "user_taught",
        }
    }
}

/// Lifecycle stage of a learned skill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkillStage {
    /// Just discovered from pattern mining (confidence 0.2–0.5).
    Candidate,
    /// Validated through repeated observation (confidence 0.5–0.7).
    Validated,
    /// Reliable enough to proactively offer (confidence 0.7–0.9).
    Reliable,
    /// Promoted to an ActionSchema in the cognitive graph (confidence 0.9+).
    Promoted,
    /// Not observed recently, may be outdated.
    Stale,
    /// Explicitly deprecated (user rejected or low success rate).
    Deprecated,
}

impl SkillStage {
    pub fn from_confidence(confidence: f64, last_used_age_secs: f64, deprecated: bool) -> Self {
        if deprecated {
            return Self::Deprecated;
        }
        // Stale after 30 days of non-use
        if last_used_age_secs > 30.0 * 86400.0 && confidence < 0.9 {
            return Self::Stale;
        }
        match confidence {
            c if c >= 0.9 => Self::Promoted,
            c if c >= 0.7 => Self::Reliable,
            c if c >= 0.5 => Self::Validated,
            _ => Self::Candidate,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 2  Skill Steps and Triggers
// ══════════════════════════════════════════════════════════════════════════════

/// A single step within a learned multi-step skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillStep {
    /// Zero-based position in the sequence.
    pub ordinal: u16,
    /// The action kind string (matches observer event action_kind).
    pub action_kind: String,
    /// Optional app_id if this step involves an app transition.
    pub app_id: Option<u16>,
    /// Optional tool name if this step is a tool call.
    pub tool_name: Option<String>,
    /// Expected duration of this step in milliseconds (learned from observations).
    pub expected_duration_ms: u64,
    /// Whether this step is optional (can be skipped without breaking the skill).
    pub optional: bool,
}

/// Context conditions under which a skill should be triggered.
///
/// All fields are optional — the more that match, the stronger the trigger signal.
/// A trigger with no fields set matches any context (universal skill).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillTrigger {
    /// Time-of-day bins when this skill is typically used (0-23 hours).
    /// Empty = any time.
    pub time_bins: Vec<u8>,
    /// App that was open when the skill was typically initiated.
    pub initiating_app_id: Option<u16>,
    /// App transition that precedes this skill (from → to).
    pub preceding_transition: Option<(u16, u16)>,
    /// Day-of-week preference (0=Mon, 6=Sun). Empty = any day.
    pub day_of_week: Vec<u8>,
    /// Minimum session duration (seconds) before this skill is relevant.
    pub min_session_duration_secs: Option<f64>,
    /// Whether the user was idle before skill initiation.
    pub preceded_by_idle: Option<bool>,
}

impl SkillTrigger {
    pub fn new() -> Self {
        Self {
            time_bins: Vec::new(),
            initiating_app_id: None,
            preceding_transition: None,
            day_of_week: Vec::new(),
            min_session_duration_secs: None,
            preceded_by_idle: None,
        }
    }

    /// How many trigger conditions are specified (non-empty).
    pub fn specificity(&self) -> u8 {
        let mut s = 0u8;
        if !self.time_bins.is_empty() {
            s += 1;
        }
        if self.initiating_app_id.is_some() {
            s += 1;
        }
        if self.preceding_transition.is_some() {
            s += 1;
        }
        if !self.day_of_week.is_empty() {
            s += 1;
        }
        if self.min_session_duration_secs.is_some() {
            s += 1;
        }
        if self.preceded_by_idle.is_some() {
            s += 1;
        }
        s
    }

    /// Score how well the current context matches this trigger [0.0, 1.0].
    ///
    /// Returns None if no conditions are specified (universal match → 1.0).
    pub fn match_score(
        &self,
        hour: u8,
        day: u8,
        current_app: Option<u16>,
        recent_transition: Option<(u16, u16)>,
        session_duration_secs: f64,
        was_idle: bool,
    ) -> f64 {
        let spec = self.specificity();
        if spec == 0 {
            return 1.0;
        }

        let mut matches = 0u8;
        let mut total = 0u8;

        if !self.time_bins.is_empty() {
            total += 1;
            if self.time_bins.contains(&hour) {
                matches += 1;
            }
        }

        if let Some(app) = self.initiating_app_id {
            total += 1;
            if current_app == Some(app) {
                matches += 1;
            }
        }

        if let Some(expected) = self.preceding_transition {
            total += 1;
            if recent_transition == Some(expected) {
                matches += 1;
            }
        }

        if !self.day_of_week.is_empty() {
            total += 1;
            if self.day_of_week.contains(&day) {
                matches += 1;
            }
        }

        if let Some(min_dur) = self.min_session_duration_secs {
            total += 1;
            if session_duration_secs >= min_dur {
                matches += 1;
            }
        }

        if let Some(expect_idle) = self.preceded_by_idle {
            total += 1;
            if was_idle == expect_idle {
                matches += 1;
            }
        }

        if total == 0 {
            1.0
        } else {
            matches as f64 / total as f64
        }
    }
}

impl Default for SkillTrigger {
    fn default() -> Self {
        Self::new()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  Learned Skill
// ══════════════════════════════════════════════════════════════════════════════

/// A learned multi-step skill with trigger context and reliability tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedSkill {
    /// Unique identifier (derived from dedup_key hash).
    pub id: SkillId,
    /// Human-readable description of what this skill does.
    pub description: String,
    /// How this skill was discovered.
    pub origin: SkillOrigin,
    /// Current lifecycle stage.
    pub stage: SkillStage,
    /// Deduplication key (e.g., "app_seq:14→20→15").
    pub dedup_key: String,
    /// The ordered sequence of steps.
    pub steps: Vec<SkillStep>,
    /// When this skill should be triggered.
    pub trigger: SkillTrigger,
    /// Current confidence [0.0, 1.0].
    pub confidence: f64,
    /// Number of times this exact sequence was observed.
    pub observation_count: u32,
    /// Number of times the skill was offered to the user.
    pub offer_count: u32,
    /// Number of times the user accepted the offered skill.
    pub acceptance_count: u32,
    /// Number of times the skill executed successfully.
    pub success_count: u32,
    /// Number of times the skill execution failed.
    pub failure_count: u32,
    /// When this skill was first discovered (Unix seconds).
    pub discovered_at: f64,
    /// When this skill was last observed or used (Unix seconds).
    pub last_seen_at: f64,
    /// Whether explicitly deprecated by user or system.
    pub deprecated: bool,
    /// Optional NodeId of the promoted ActionSchema (set after promotion).
    pub promoted_schema_id: Option<u32>,
}

impl LearnedSkill {
    /// Create a new skill candidate from a discovered pattern.
    pub fn new(
        description: String,
        origin: SkillOrigin,
        dedup_key: String,
        steps: Vec<SkillStep>,
        trigger: SkillTrigger,
        observation_count: u32,
        now: f64,
    ) -> Self {
        let id = hash_dedup_key(&dedup_key);
        Self {
            id,
            description,
            origin,
            stage: SkillStage::Candidate,
            dedup_key,
            steps,
            trigger,
            confidence: 0.3,
            observation_count,
            offer_count: 0,
            acceptance_count: 0,
            success_count: 0,
            failure_count: 0,
            discovered_at: now,
            last_seen_at: now,
            deprecated: false,
            promoted_schema_id: None,
        }
    }

    /// Record another observation of this action sequence.
    pub fn observe(&mut self, now: f64) {
        self.observation_count += 1;
        self.last_seen_at = now;
        // Confidence grows with repeated observation (diminishing returns).
        let learning_rate = 0.05;
        self.confidence += (1.0 - self.confidence) * learning_rate;
        self.update_stage(now);
    }

    /// Record that the skill was offered to the user.
    pub fn record_offer(&mut self) {
        self.offer_count += 1;
    }

    /// Record that the user accepted the offered skill.
    pub fn record_acceptance(&mut self, now: f64) {
        self.acceptance_count += 1;
        self.last_seen_at = now;
        let learning_rate = 0.10;
        self.confidence += (1.0 - self.confidence) * learning_rate;
        self.update_stage(now);
    }

    /// Record a successful skill execution.
    pub fn record_success(&mut self, now: f64) {
        self.success_count += 1;
        self.last_seen_at = now;
        let learning_rate = 0.08;
        self.confidence += (1.0 - self.confidence) * learning_rate;
        self.update_stage(now);
    }

    /// Record a failed skill execution.
    pub fn record_failure(&mut self, now: f64) {
        self.failure_count += 1;
        self.last_seen_at = now;
        let decay_rate = 0.15;
        self.confidence -= self.confidence * decay_rate;
        self.update_stage(now);
    }

    /// Record that the user rejected the offered skill.
    pub fn record_rejection(&mut self, now: f64) {
        self.last_seen_at = now;
        let decay_rate = 0.12;
        self.confidence -= self.confidence * decay_rate;
        self.update_stage(now);
    }

    /// Explicitly deprecate this skill.
    pub fn deprecate(&mut self) {
        self.deprecated = true;
        self.stage = SkillStage::Deprecated;
    }

    /// Success rate [0.0, 1.0] (NaN-safe).
    pub fn success_rate(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            0.5 // Neutral prior
        } else {
            self.success_count as f64 / total as f64
        }
    }

    /// Acceptance rate [0.0, 1.0] (NaN-safe).
    pub fn acceptance_rate(&self) -> f64 {
        if self.offer_count == 0 {
            0.5 // Neutral prior
        } else {
            self.acceptance_count as f64 / self.offer_count as f64
        }
    }

    /// Whether this skill is ready to be proactively offered.
    pub fn is_offerable(&self) -> bool {
        matches!(self.stage, SkillStage::Reliable | SkillStage::Promoted)
            && !self.deprecated
            && self.confidence >= 0.7
    }

    /// Whether this skill is eligible for promotion to an ActionSchema.
    pub fn is_promotable(&self) -> bool {
        self.confidence >= 0.9
            && self.observation_count >= 10
            && self.success_rate() >= 0.75
            && self.promoted_schema_id.is_none()
            && !self.deprecated
    }

    /// Number of steps in this skill.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    fn update_stage(&mut self, now: f64) {
        let age = now - self.last_seen_at;
        self.stage = SkillStage::from_confidence(self.confidence, age, self.deprecated);
    }
}

/// FNV-1a hash of the dedup key → SkillId.
fn hash_dedup_key(key: &str) -> SkillId {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in key.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  Skill Registry
// ══════════════════════════════════════════════════════════════════════════════

/// Configuration for the skill acquisition engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillConfig {
    /// Minimum times a sequence must be observed to become a candidate.
    pub min_observations: u32,
    /// Maximum number of learned skills to retain.
    pub max_skills: usize,
    /// Maximum steps in a single skill (prevents overly long chains).
    pub max_steps_per_skill: usize,
    /// Minimum confidence to offer a skill proactively.
    pub min_offer_confidence: f64,
    /// Minimum confidence and observations for ActionSchema promotion.
    pub promotion_confidence: f64,
    /// Minimum observations for promotion.
    pub promotion_min_observations: u32,
    /// Minimum success rate for promotion.
    pub promotion_min_success_rate: f64,
    /// Age in seconds after which an unused candidate is pruned.
    pub candidate_ttl_secs: f64,
    /// Age in seconds after which an unused validated skill becomes stale.
    pub stale_threshold_secs: f64,
    /// Maximum gap between events to consider them part of the same sequence (ms).
    pub max_sequence_gap_ms: u64,
    /// Whether to auto-deprecate skills with very low acceptance rates.
    pub auto_deprecate: bool,
    /// Acceptance rate below which auto-deprecation triggers.
    pub auto_deprecate_threshold: f64,
    /// Minimum offers before auto-deprecation can trigger.
    pub auto_deprecate_min_offers: u32,
}

impl Default for SkillConfig {
    fn default() -> Self {
        Self {
            min_observations: 3,
            max_skills: 200,
            max_steps_per_skill: 8,
            min_offer_confidence: 0.7,
            promotion_confidence: 0.9,
            promotion_min_observations: 10,
            promotion_min_success_rate: 0.75,
            candidate_ttl_secs: 14.0 * 86400.0,   // 14 days
            stale_threshold_secs: 30.0 * 86400.0, // 30 days
            max_sequence_gap_ms: 30_000,          // 30 seconds
            auto_deprecate: true,
            auto_deprecate_threshold: 0.15,
            auto_deprecate_min_offers: 5,
        }
    }
}

/// The skill registry — stores all learned skills.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillRegistry {
    /// All learned skills, keyed by dedup_key.
    pub skills: HashMap<String, LearnedSkill>,
    /// Total skills ever discovered.
    pub total_discovered: u64,
    /// Total skills promoted to ActionSchema.
    pub total_promoted: u64,
    /// Total skills deprecated.
    pub total_deprecated: u64,
    /// Last time skill discovery was run (Unix seconds).
    pub last_discovery_at: f64,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self {
            skills: HashMap::new(),
            total_discovered: 0,
            total_promoted: 0,
            total_deprecated: 0,
            last_discovery_at: 0.0,
        }
    }

    /// Get a skill by dedup_key.
    pub fn find(&self, dedup_key: &str) -> Option<&LearnedSkill> {
        self.skills.get(dedup_key)
    }

    /// Get a mutable skill by dedup_key.
    pub fn find_mut(&mut self, dedup_key: &str) -> Option<&mut LearnedSkill> {
        self.skills.get_mut(dedup_key)
    }

    /// Get a skill by SkillId.
    pub fn find_by_id(&self, id: SkillId) -> Option<&LearnedSkill> {
        self.skills.values().find(|s| s.id == id)
    }

    /// Number of active (non-deprecated) skills.
    pub fn active_count(&self) -> usize {
        self.skills.values().filter(|s| !s.deprecated).count()
    }

    /// Skills that can currently be offered.
    pub fn offerable(&self) -> Vec<&LearnedSkill> {
        self.skills.values().filter(|s| s.is_offerable()).collect()
    }

    /// Skills eligible for promotion.
    pub fn promotable(&self) -> Vec<&LearnedSkill> {
        self.skills.values().filter(|s| s.is_promotable()).collect()
    }

    /// Skills by origin.
    pub fn by_origin(&self, origin: SkillOrigin) -> Vec<&LearnedSkill> {
        self.skills
            .values()
            .filter(|s| s.origin == origin && !s.deprecated)
            .collect()
    }

    /// Skills by stage.
    pub fn by_stage(&self, stage: SkillStage) -> Vec<&LearnedSkill> {
        self.skills
            .values()
            .filter(|s| std::mem::discriminant(&s.stage) == std::mem::discriminant(&stage))
            .collect()
    }

    /// Total skill count.
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 5  Skill Discovery — Sequence Mining
// ══════════════════════════════════════════════════════════════════════════════

/// Result of a skill discovery run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryResult {
    /// Newly discovered skill candidates.
    pub new_skills: Vec<String>,
    /// Existing skills that were re-observed (dedup_keys).
    pub reinforced: Vec<String>,
    /// Skills auto-deprecated this run.
    pub auto_deprecated: Vec<String>,
    /// Skills pruned (expired candidates).
    pub pruned: Vec<String>,
    /// Total skills in registry after this run.
    pub total_skills: usize,
}

/// Run the skill discovery pipeline.
///
/// Analyzes recent events in the buffer, mines repeated sequences,
/// updates or creates skill candidates, handles lifecycle transitions.
pub fn discover_skills(
    buffer: &EventBuffer,
    registry: &mut SkillRegistry,
    config: &SkillConfig,
    now: f64,
) -> DiscoveryResult {
    let mut result = DiscoveryResult {
        new_skills: Vec::new(),
        reinforced: Vec::new(),
        auto_deprecated: Vec::new(),
        pruned: Vec::new(),
        total_skills: 0,
    };

    // Phase 1: Mine app transition sequences
    discover_app_sequence_skills(buffer, registry, config, now, &mut result);

    // Phase 2: Mine tool call chains
    discover_tool_chain_skills(buffer, registry, config, now, &mut result);

    // Phase 3: Mine suggestion acceptance patterns
    discover_suggestion_pattern_skills(buffer, registry, config, now, &mut result);

    // Phase 4: Mine notification response patterns
    discover_notification_response_skills(buffer, registry, config, now, &mut result);

    // Phase 5: Auto-deprecate low-performing skills
    if config.auto_deprecate {
        auto_deprecate_skills(registry, config, &mut result);
    }

    // Phase 6: Prune expired candidates
    prune_expired_candidates(registry, config, now, &mut result);

    // Phase 7: Enforce max skills limit
    enforce_max_skills(registry, config);

    registry.last_discovery_at = now;
    result.total_skills = registry.len();
    result
}

// ══════════════════════════════════════════════════════════════════════════════
// § 6  Sequence Mining Algorithms
// ══════════════════════════════════════════════════════════════════════════════

/// Mine app transition sequences (A→B→C patterns).
///
/// Scans the event buffer for AppOpened/AppSequence events and finds
/// repeated multi-step sequences within the configured gap threshold.
fn discover_app_sequence_skills(
    buffer: &EventBuffer,
    registry: &mut SkillRegistry,
    config: &SkillConfig,
    now: f64,
    result: &mut DiscoveryResult,
) {
    // Extract app transition chains from recent events
    let chains = extract_app_chains(
        buffer,
        config.max_sequence_gap_ms,
        config.max_steps_per_skill,
    );

    // Count occurrences of each chain
    let mut chain_counts: HashMap<Vec<u16>, u32> = HashMap::new();
    let mut chain_times: HashMap<Vec<u16>, Vec<f64>> = HashMap::new();
    for (chain, timestamp) in &chains {
        if chain.len() >= 2 {
            *chain_counts.entry(chain.clone()).or_insert(0) += 1;
            chain_times
                .entry(chain.clone())
                .or_default()
                .push(*timestamp);
        }
    }

    // Create or reinforce skills for frequent chains
    for (chain, count) in &chain_counts {
        if *count < config.min_observations {
            continue;
        }

        let dedup_key = format!(
            "app_seq:{}",
            chain
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join("→")
        );

        if let Some(skill) = registry.find_mut(&dedup_key) {
            skill.observe(now);
            result.reinforced.push(dedup_key);
        } else {
            // Build steps
            let steps: Vec<SkillStep> = chain
                .iter()
                .enumerate()
                .map(|(i, &app_id)| {
                    SkillStep {
                        ordinal: i as u16,
                        action_kind: "app_open".to_string(),
                        app_id: Some(app_id),
                        tool_name: None,
                        expected_duration_ms: 0, // Unknown from app opens alone
                        optional: false,
                    }
                })
                .collect();

            // Infer trigger context from observation timestamps
            let trigger = infer_trigger_from_timestamps(
                chain_times.get(chain).unwrap_or(&Vec::new()),
                Some(chain[0]),
            );

            let description = format!(
                "App workflow: {}",
                chain
                    .iter()
                    .map(|id| format!("app#{}", id))
                    .collect::<Vec<_>>()
                    .join(" → ")
            );

            let skill = LearnedSkill::new(
                description,
                SkillOrigin::AppSequence,
                dedup_key.clone(),
                steps,
                trigger,
                *count,
                now,
            );

            registry.skills.insert(dedup_key.clone(), skill);
            registry.total_discovered += 1;
            result.new_skills.push(dedup_key);
        }
    }
}

/// Mine tool call chain patterns (tool_A → tool_B → tool_C).
fn discover_tool_chain_skills(
    buffer: &EventBuffer,
    registry: &mut SkillRegistry,
    config: &SkillConfig,
    now: f64,
    result: &mut DiscoveryResult,
) {
    let chains = extract_tool_chains(
        buffer,
        config.max_sequence_gap_ms,
        config.max_steps_per_skill,
    );

    let mut chain_counts: HashMap<Vec<String>, u32> = HashMap::new();
    let mut chain_times: HashMap<Vec<String>, Vec<f64>> = HashMap::new();
    for (chain, timestamp) in &chains {
        if chain.len() >= 2 {
            *chain_counts.entry(chain.clone()).or_insert(0) += 1;
            chain_times
                .entry(chain.clone())
                .or_default()
                .push(*timestamp);
        }
    }

    for (chain, count) in &chain_counts {
        if *count < config.min_observations {
            continue;
        }

        let dedup_key = format!("tool_chain:{}", chain.join("→"));

        if let Some(skill) = registry.find_mut(&dedup_key) {
            skill.observe(now);
            result.reinforced.push(dedup_key);
        } else {
            let steps: Vec<SkillStep> = chain
                .iter()
                .enumerate()
                .map(|(i, tool)| SkillStep {
                    ordinal: i as u16,
                    action_kind: "tool_call".to_string(),
                    app_id: None,
                    tool_name: Some(tool.clone()),
                    expected_duration_ms: 0,
                    optional: false,
                })
                .collect();

            let trigger =
                infer_trigger_from_timestamps(chain_times.get(chain).unwrap_or(&Vec::new()), None);

            let description = format!("Tool chain: {}", chain.join(" → "));

            let skill = LearnedSkill::new(
                description,
                SkillOrigin::ToolChain,
                dedup_key.clone(),
                steps,
                trigger,
                *count,
                now,
            );

            registry.skills.insert(dedup_key.clone(), skill);
            registry.total_discovered += 1;
            result.new_skills.push(dedup_key);
        }
    }
}

/// Mine suggestion acceptance patterns.
///
/// Detects when the same action_kind is consistently accepted,
/// suggesting the system can proactively take that action.
fn discover_suggestion_pattern_skills(
    buffer: &EventBuffer,
    registry: &mut SkillRegistry,
    config: &SkillConfig,
    now: f64,
    result: &mut DiscoveryResult,
) {
    // Count accepts and rejects per action_kind
    let mut accept_counts: HashMap<String, u32> = HashMap::new();
    let mut reject_counts: HashMap<String, u32> = HashMap::new();
    let mut accept_times: HashMap<String, Vec<f64>> = HashMap::new();

    for event in buffer.recent(buffer.len()) {
        match &event.data {
            SystemEventData::SuggestionAccepted { action_kind, .. } => {
                *accept_counts.entry(action_kind.clone()).or_insert(0) += 1;
                accept_times
                    .entry(action_kind.clone())
                    .or_default()
                    .push(event.timestamp);
            }
            SystemEventData::SuggestionRejected { action_kind, .. } => {
                *reject_counts.entry(action_kind.clone()).or_insert(0) += 1;
            }
            _ => {}
        }
    }

    for (action_kind, accepts) in &accept_counts {
        let rejects = reject_counts.get(action_kind).copied().unwrap_or(0);
        let total = accepts + rejects;
        if total < config.min_observations {
            continue;
        }

        let acceptance_rate = *accepts as f64 / total as f64;
        // Only create skill for highly accepted suggestion types
        if acceptance_rate < 0.75 {
            continue;
        }

        let dedup_key = format!("suggestion_pattern:{}", action_kind);

        if let Some(skill) = registry.find_mut(&dedup_key) {
            skill.observe(now);
            result.reinforced.push(dedup_key);
        } else {
            let steps = vec![SkillStep {
                ordinal: 0,
                action_kind: action_kind.clone(),
                app_id: None,
                tool_name: None,
                expected_duration_ms: 0,
                optional: false,
            }];

            let trigger = infer_trigger_from_timestamps(
                accept_times.get(action_kind).unwrap_or(&Vec::new()),
                None,
            );

            let description = format!(
                "Auto-action: {} ({}% acceptance rate)",
                action_kind,
                (acceptance_rate * 100.0) as u32,
            );

            let mut skill = LearnedSkill::new(
                description,
                SkillOrigin::SuggestionPattern,
                dedup_key.clone(),
                steps,
                trigger,
                *accepts,
                now,
            );
            // Start with higher confidence since we have accept/reject data
            skill.confidence = 0.3 + acceptance_rate * 0.3;

            registry.skills.insert(dedup_key.clone(), skill);
            registry.total_discovered += 1;
            result.new_skills.push(dedup_key);
        }
    }
}

/// Mine notification → action response patterns.
///
/// Detects when a specific notification type consistently triggers
/// the same user action, suggesting the system can automate the response.
fn discover_notification_response_skills(
    buffer: &EventBuffer,
    registry: &mut SkillRegistry,
    config: &SkillConfig,
    now: f64,
    result: &mut DiscoveryResult,
) {
    // Build notification_type → action_taken frequency map
    let mut response_counts: HashMap<(String, String), u32> = HashMap::new();
    let mut response_times: HashMap<(String, String), Vec<f64>> = HashMap::new();

    for event in buffer.recent(buffer.len()) {
        if let SystemEventData::NotificationActedOn {
            notif_type,
            action_taken,
        } = &event.data
        {
            let key = (notif_type.clone(), action_taken.clone());
            *response_counts.entry(key.clone()).or_insert(0) += 1;
            response_times.entry(key).or_default().push(event.timestamp);
        }
    }

    for ((notif_type, action_taken), count) in &response_counts {
        if *count < config.min_observations {
            continue;
        }

        let dedup_key = format!("notif_response:{}→{}", notif_type, action_taken);

        if let Some(skill) = registry.find_mut(&dedup_key) {
            skill.observe(now);
            result.reinforced.push(dedup_key);
        } else {
            let steps = vec![
                SkillStep {
                    ordinal: 0,
                    action_kind: format!("receive_{}", notif_type),
                    app_id: None,
                    tool_name: None,
                    expected_duration_ms: 0,
                    optional: false,
                },
                SkillStep {
                    ordinal: 1,
                    action_kind: action_taken.clone(),
                    app_id: None,
                    tool_name: None,
                    expected_duration_ms: 0,
                    optional: false,
                },
            ];

            let key = (notif_type.clone(), action_taken.clone());
            let trigger = infer_trigger_from_timestamps(
                response_times.get(&key).unwrap_or(&Vec::new()),
                None,
            );

            let description = format!(
                "Auto-respond: {} notification → {}",
                notif_type, action_taken
            );

            let skill = LearnedSkill::new(
                description,
                SkillOrigin::NotificationResponse,
                dedup_key.clone(),
                steps,
                trigger,
                *count,
                now,
            );

            registry.skills.insert(dedup_key.clone(), skill);
            registry.total_discovered += 1;
            result.new_skills.push(dedup_key);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 7  Event Chain Extraction
// ══════════════════════════════════════════════════════════════════════════════

/// Extract app transition chains from the event buffer.
///
/// Returns (chain_of_app_ids, first_timestamp_of_chain).
fn extract_app_chains(
    buffer: &EventBuffer,
    max_gap_ms: u64,
    max_steps: usize,
) -> Vec<(Vec<u16>, f64)> {
    let mut events = buffer.recent(buffer.len());
    events.reverse(); // recent() returns newest-first; we need chronological order
    let mut chains: Vec<(Vec<u16>, f64)> = Vec::new();
    let mut current_chain: Vec<u16> = Vec::new();
    let mut chain_start: f64 = 0.0;
    let mut last_ts: f64 = 0.0;

    for event in events {
        match &event.data {
            SystemEventData::AppOpened { app_id } => {
                let gap_ms = ((event.timestamp - last_ts) * 1000.0) as u64;
                if !current_chain.is_empty()
                    && (gap_ms > max_gap_ms || current_chain.len() >= max_steps)
                {
                    // End current chain, start new one
                    if current_chain.len() >= 2 {
                        chains.push((current_chain.clone(), chain_start));
                    }
                    current_chain.clear();
                }
                if current_chain.is_empty() {
                    chain_start = event.timestamp;
                }
                // Avoid consecutive duplicates
                if current_chain.last() != Some(app_id) {
                    current_chain.push(*app_id);
                }
                last_ts = event.timestamp;
            }
            SystemEventData::AppSequence {
                from_app, to_app, ..
            } => {
                // Use actual timestamp gap between events to detect chain boundaries
                let inter_event_gap_ms = if last_ts > 0.0 {
                    ((event.timestamp - last_ts) * 1000.0) as u64
                } else {
                    0
                };
                if !current_chain.is_empty()
                    && (inter_event_gap_ms > max_gap_ms || current_chain.len() >= max_steps)
                {
                    if current_chain.len() >= 2 {
                        chains.push((current_chain.clone(), chain_start));
                    }
                    current_chain.clear();
                }
                if current_chain.is_empty() {
                    chain_start = event.timestamp;
                    current_chain.push(*from_app);
                }
                if current_chain.last() != Some(to_app) {
                    current_chain.push(*to_app);
                }
                if current_chain.len() >= max_steps {
                    chains.push((current_chain.clone(), chain_start));
                    current_chain.clear();
                }
                last_ts = event.timestamp;
            }
            _ => {}
        }
    }

    // Don't forget the trailing chain
    if current_chain.len() >= 2 {
        chains.push((current_chain, chain_start));
    }

    chains
}

/// Extract tool call chains from the event buffer.
///
/// Returns (chain_of_tool_names, first_timestamp_of_chain).
fn extract_tool_chains(
    buffer: &EventBuffer,
    max_gap_ms: u64,
    max_steps: usize,
) -> Vec<(Vec<String>, f64)> {
    let mut events = buffer.recent(buffer.len());
    events.reverse(); // recent() returns newest-first; we need chronological order
    let mut chains: Vec<(Vec<String>, f64)> = Vec::new();
    let mut current_chain: Vec<String> = Vec::new();
    let mut chain_start: f64 = 0.0;
    let mut last_ts: f64 = 0.0;

    for event in events {
        if let SystemEventData::ToolCallCompleted {
            tool_name, success, ..
        } = &event.data
        {
            let gap_ms = ((event.timestamp - last_ts) * 1000.0) as u64;
            if !current_chain.is_empty()
                && (gap_ms > max_gap_ms || current_chain.len() >= max_steps)
            {
                if current_chain.len() >= 2 {
                    chains.push((current_chain.clone(), chain_start));
                }
                current_chain.clear();
            }
            if current_chain.is_empty() {
                chain_start = event.timestamp;
            }
            // Only include successful tool calls in chains
            if *success {
                current_chain.push(tool_name.clone());
            } else {
                // Failed tool call breaks the chain
                if current_chain.len() >= 2 {
                    chains.push((current_chain.clone(), chain_start));
                }
                current_chain.clear();
            }
            last_ts = event.timestamp;
        }
    }

    if current_chain.len() >= 2 {
        chains.push((current_chain, chain_start));
    }

    chains
}

// ══════════════════════════════════════════════════════════════════════════════
// § 8  Trigger Inference
// ══════════════════════════════════════════════════════════════════════════════

/// Infer trigger context from observation timestamps and optional initiating app.
///
/// Uses statistical analysis of when the pattern was observed to determine
/// time-of-day and day-of-week preferences.
fn infer_trigger_from_timestamps(timestamps: &[f64], initiating_app: Option<u16>) -> SkillTrigger {
    let mut trigger = SkillTrigger::new();
    trigger.initiating_app_id = initiating_app;

    if timestamps.is_empty() {
        return trigger;
    }

    // Extract hours and days from timestamps
    let mut hour_counts = [0u32; 24];
    let mut day_counts = [0u32; 7];

    for &ts in timestamps {
        let secs = ts as u64;
        let hour = ((secs % 86400) / 3600) as u8;
        let day = ((secs / 86400 + 3) % 7) as u8; // Unix epoch was Thursday
        hour_counts[hour as usize] += 1;
        day_counts[day as usize] += 1;
    }

    // Find peak hours (within 2 standard deviations of max)
    let total_obs = timestamps.len() as f64;
    let uniform_rate = total_obs / 24.0;

    for (hour, &count) in hour_counts.iter().enumerate() {
        // Hour is a peak if it has significantly more observations than uniform
        if count as f64 > uniform_rate * 2.0 && count >= 2 {
            trigger.time_bins.push(hour as u8);
        }
    }

    // Find preferred days (significantly above uniform rate)
    let day_uniform = total_obs / 7.0;
    for (day, &count) in day_counts.iter().enumerate() {
        if count as f64 > day_uniform * 1.5 && count >= 2 {
            trigger.day_of_week.push(day as u8);
        }
    }

    trigger
}

// ══════════════════════════════════════════════════════════════════════════════
// § 9  Lifecycle Management
// ══════════════════════════════════════════════════════════════════════════════

/// Auto-deprecate skills with persistently low acceptance rates.
fn auto_deprecate_skills(
    registry: &mut SkillRegistry,
    config: &SkillConfig,
    result: &mut DiscoveryResult,
) {
    let keys: Vec<String> = registry
        .skills
        .iter()
        .filter(|(_, s)| {
            !s.deprecated
                && s.offer_count >= config.auto_deprecate_min_offers
                && s.acceptance_rate() < config.auto_deprecate_threshold
        })
        .map(|(k, _)| k.clone())
        .collect();

    for key in keys {
        if let Some(skill) = registry.skills.get_mut(&key) {
            skill.deprecate();
            registry.total_deprecated += 1;
            result.auto_deprecated.push(key);
        }
    }
}

/// Prune expired candidate skills that never gained enough confidence.
fn prune_expired_candidates(
    registry: &mut SkillRegistry,
    config: &SkillConfig,
    now: f64,
    result: &mut DiscoveryResult,
) {
    let expired: Vec<String> = registry
        .skills
        .iter()
        .filter(|(_, s)| {
            matches!(s.stage, SkillStage::Candidate)
                && (now - s.last_seen_at) > config.candidate_ttl_secs
        })
        .map(|(k, _)| k.clone())
        .collect();

    for key in expired {
        registry.skills.remove(&key);
        result.pruned.push(key);
    }
}

/// Enforce maximum skill count by removing lowest-confidence deprecated/stale skills.
fn enforce_max_skills(registry: &mut SkillRegistry, config: &SkillConfig) {
    if registry.skills.len() <= config.max_skills {
        return;
    }

    // Collect skills sorted by priority (deprecated < stale < candidate < validated)
    let mut ranked: Vec<(String, f64, u8)> = registry
        .skills
        .iter()
        .map(|(k, s)| {
            let priority = match s.stage {
                SkillStage::Deprecated => 0,
                SkillStage::Stale => 1,
                SkillStage::Candidate => 2,
                SkillStage::Validated => 3,
                SkillStage::Reliable => 4,
                SkillStage::Promoted => 5,
            };
            (k.clone(), s.confidence, priority)
        })
        .collect();

    // Sort by priority ascending, then confidence ascending (remove lowest first)
    ranked.sort_by(|a, b| {
        a.2.cmp(&b.2)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    let to_remove = registry.skills.len() - config.max_skills;
    for (key, _, _) in ranked.into_iter().take(to_remove) {
        registry.skills.remove(&key);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 10  Skill Matching — Find Relevant Skills for Current Context
// ══════════════════════════════════════════════════════════════════════════════

/// A scored skill match for the current context.
#[derive(Debug, Clone)]
pub struct SkillMatch {
    /// The matched skill's dedup key.
    pub dedup_key: String,
    /// The matched skill's ID.
    pub skill_id: SkillId,
    /// How well the current context matches the trigger [0.0, 1.0].
    pub trigger_score: f64,
    /// The skill's confidence.
    pub confidence: f64,
    /// Combined relevance score = trigger_score * confidence.
    pub relevance: f64,
    /// Human-readable description.
    pub description: String,
    /// Number of steps.
    pub step_count: usize,
}

/// Find skills relevant to the current context, ranked by relevance.
///
/// Returns up to `max_results` matches above `min_trigger_score`.
pub fn find_matching_skills(
    registry: &SkillRegistry,
    hour: u8,
    day: u8,
    current_app: Option<u16>,
    recent_transition: Option<(u16, u16)>,
    session_duration_secs: f64,
    was_idle: bool,
    min_trigger_score: f64,
    max_results: usize,
) -> Vec<SkillMatch> {
    let mut matches: Vec<SkillMatch> = registry
        .skills
        .iter()
        .filter(|(_, s)| s.is_offerable())
        .filter_map(|(key, skill)| {
            let trigger_score = skill.trigger.match_score(
                hour,
                day,
                current_app,
                recent_transition,
                session_duration_secs,
                was_idle,
            );
            if trigger_score < min_trigger_score {
                return None;
            }
            let relevance = trigger_score * skill.confidence;
            Some(SkillMatch {
                dedup_key: key.clone(),
                skill_id: skill.id,
                trigger_score,
                confidence: skill.confidence,
                relevance,
                description: skill.description.clone(),
                step_count: skill.step_count(),
            })
        })
        .collect();

    // Sort by relevance descending
    matches.sort_by(|a, b| {
        b.relevance
            .partial_cmp(&a.relevance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    matches.truncate(max_results);
    matches
}

/// Summary statistics for the skill registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillSummary {
    pub total: usize,
    pub active: usize,
    pub candidates: usize,
    pub validated: usize,
    pub reliable: usize,
    pub promoted: usize,
    pub stale: usize,
    pub deprecated: usize,
    pub total_discovered: u64,
    pub total_promoted: u64,
    pub total_deprecated: u64,
    pub by_origin: HashMap<String, usize>,
}

/// Generate a summary of the skill registry.
pub fn summarize_skills(registry: &SkillRegistry) -> SkillSummary {
    let mut by_origin: HashMap<String, usize> = HashMap::new();
    let mut candidates = 0usize;
    let mut validated = 0usize;
    let mut reliable = 0usize;
    let mut promoted = 0usize;
    let mut stale = 0usize;
    let mut deprecated = 0usize;

    for skill in registry.skills.values() {
        *by_origin
            .entry(skill.origin.as_str().to_string())
            .or_insert(0) += 1;
        match skill.stage {
            SkillStage::Candidate => candidates += 1,
            SkillStage::Validated => validated += 1,
            SkillStage::Reliable => reliable += 1,
            SkillStage::Promoted => promoted += 1,
            SkillStage::Stale => stale += 1,
            SkillStage::Deprecated => deprecated += 1,
        }
    }

    SkillSummary {
        total: registry.len(),
        active: registry.active_count(),
        candidates,
        validated,
        reliable,
        promoted,
        stale,
        deprecated,
        total_discovered: registry.total_discovered,
        total_promoted: registry.total_promoted,
        total_deprecated: registry.total_deprecated,
        by_origin,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 11  Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(offset: f64) -> f64 {
        86400.0 * 100.0 + offset
    }

    fn make_buffer(events: Vec<SystemEvent>) -> EventBuffer {
        let mut buffer = EventBuffer::new(10_000);
        for event in events {
            buffer.push(event);
        }
        buffer
    }

    fn app_open(time: f64, app_id: u16) -> SystemEvent {
        SystemEvent::new(time, SystemEventData::AppOpened { app_id })
    }

    fn app_seq(time: f64, from: u16, to: u16, gap: u64) -> SystemEvent {
        SystemEvent::new(
            time,
            SystemEventData::AppSequence {
                from_app: from,
                to_app: to,
                gap_ms: gap,
            },
        )
    }

    fn tool_call(time: f64, name: &str, success: bool) -> SystemEvent {
        SystemEvent::new(
            time,
            SystemEventData::ToolCallCompleted {
                tool_name: name.to_string(),
                success,
                duration_ms: 100,
            },
        )
    }

    fn suggestion_accept(time: f64, kind: &str) -> SystemEvent {
        SystemEvent::new(
            time,
            SystemEventData::SuggestionAccepted {
                suggestion_id: 1,
                action_kind: kind.to_string(),
                latency_ms: 200,
            },
        )
    }

    fn suggestion_reject(time: f64, kind: &str) -> SystemEvent {
        SystemEvent::new(
            time,
            SystemEventData::SuggestionRejected {
                suggestion_id: 2,
                action_kind: kind.to_string(),
            },
        )
    }

    fn notif_acted(time: f64, notif_type: &str, action: &str) -> SystemEvent {
        SystemEvent::new(
            time,
            SystemEventData::NotificationActedOn {
                notif_type: notif_type.to_string(),
                action_taken: action.to_string(),
            },
        )
    }

    // ── § 1: Basic skill creation ──

    #[test]
    fn test_skill_creation() {
        let skill = LearnedSkill::new(
            "Test skill".to_string(),
            SkillOrigin::AppSequence,
            "test:key".to_string(),
            vec![SkillStep {
                ordinal: 0,
                action_kind: "app_open".to_string(),
                app_id: Some(14),
                tool_name: None,
                expected_duration_ms: 0,
                optional: false,
            }],
            SkillTrigger::new(),
            3,
            ts(0.0),
        );

        assert_eq!(skill.stage, SkillStage::Candidate);
        assert!((skill.confidence - 0.3).abs() < 0.01);
        assert_eq!(skill.observation_count, 3);
        assert!(!skill.is_offerable());
    }

    #[test]
    fn test_skill_confidence_lifecycle() {
        let mut skill = LearnedSkill::new(
            "Test".to_string(),
            SkillOrigin::ToolChain,
            "test:lifecycle".to_string(),
            vec![],
            SkillTrigger::new(),
            1,
            ts(0.0),
        );

        // Repeated observations increase confidence
        for i in 1..30 {
            skill.observe(ts(i as f64));
        }
        assert!(
            skill.confidence > 0.5,
            "Should gain confidence: {}",
            skill.confidence
        );

        // Acceptances boost faster
        for i in 30..40 {
            skill.record_acceptance(ts(i as f64));
        }
        assert!(
            skill.confidence > 0.7,
            "Should be reliable: {}",
            skill.confidence
        );

        // Failures reduce confidence
        for i in 40..50 {
            skill.record_failure(ts(i as f64));
        }
        assert!(
            skill.confidence < 0.7,
            "Failures should reduce confidence: {}",
            skill.confidence
        );
    }

    #[test]
    fn test_skill_success_rate() {
        let mut skill = LearnedSkill::new(
            "Test".to_string(),
            SkillOrigin::AppSequence,
            "test:rate".to_string(),
            vec![],
            SkillTrigger::new(),
            1,
            ts(0.0),
        );

        // Neutral prior when no data
        assert!((skill.success_rate() - 0.5).abs() < 0.01);

        skill.success_count = 7;
        skill.failure_count = 3;
        assert!((skill.success_rate() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_skill_deprecation() {
        let mut skill = LearnedSkill::new(
            "Test".to_string(),
            SkillOrigin::AppSequence,
            "test:deprecate".to_string(),
            vec![],
            SkillTrigger::new(),
            1,
            ts(0.0),
        );

        skill.confidence = 0.8;
        skill.stage = SkillStage::Reliable;
        assert!(skill.is_offerable());

        skill.deprecate();
        assert_eq!(skill.stage, SkillStage::Deprecated);
        assert!(!skill.is_offerable());
    }

    #[test]
    fn test_skill_promotability() {
        let mut skill = LearnedSkill::new(
            "Test".to_string(),
            SkillOrigin::AppSequence,
            "test:promote".to_string(),
            vec![],
            SkillTrigger::new(),
            12,
            ts(0.0),
        );

        skill.confidence = 0.95;
        skill.success_count = 10;
        skill.failure_count = 1;
        assert!(skill.is_promotable());

        // Already promoted → not promotable
        skill.promoted_schema_id = Some(42);
        assert!(!skill.is_promotable());
    }

    // ── § 2: Trigger matching ──

    #[test]
    fn test_trigger_universal() {
        let trigger = SkillTrigger::new();
        let score = trigger.match_score(12, 3, None, None, 0.0, false);
        assert!(
            (score - 1.0).abs() < 0.01,
            "Universal trigger should always match"
        );
    }

    #[test]
    fn test_trigger_specificity() {
        let mut trigger = SkillTrigger::new();
        assert_eq!(trigger.specificity(), 0);

        trigger.time_bins = vec![9, 10, 11];
        assert_eq!(trigger.specificity(), 1);

        trigger.initiating_app_id = Some(14);
        assert_eq!(trigger.specificity(), 2);
    }

    #[test]
    fn test_trigger_matching() {
        let trigger = SkillTrigger {
            time_bins: vec![9, 10, 11],
            initiating_app_id: Some(14),
            preceding_transition: None,
            day_of_week: vec![0, 1, 2, 3, 4], // Weekdays
            min_session_duration_secs: None,
            preceded_by_idle: None,
        };

        // Perfect match
        let score = trigger.match_score(10, 2, Some(14), None, 0.0, false);
        assert!((score - 1.0).abs() < 0.01, "All conditions met: {}", score);

        // Partial match (wrong hour)
        let score = trigger.match_score(22, 2, Some(14), None, 0.0, false);
        assert!((score - 2.0 / 3.0).abs() < 0.01, "2/3 match: {}", score);

        // No match
        let score = trigger.match_score(22, 5, Some(99), None, 0.0, false);
        assert!((score).abs() < 0.01, "0/3 match: {}", score);
    }

    // ── § 3: App sequence discovery ──

    #[test]
    fn test_discover_app_sequences() {
        let mut events = Vec::new();

        // Repeat the sequence 14→20→15 three times
        for round in 0..3 {
            let base = round as f64 * 100.0;
            events.push(app_seq(ts(base + 1.0), 14, 20, 2000));
            events.push(app_seq(ts(base + 3.0), 20, 15, 3000));
        }

        let buffer = make_buffer(events);
        let mut registry = SkillRegistry::new();
        let config = SkillConfig::default();

        let result = discover_skills(&buffer, &mut registry, &config, ts(500.0));

        assert!(
            !result.new_skills.is_empty(),
            "Should discover app sequence skills"
        );

        // Verify the skill was created
        let key = result.new_skills.iter().find(|k| k.starts_with("app_seq:"));
        assert!(key.is_some(), "Should have app_seq skill");

        let skill = registry.find(key.unwrap()).unwrap();
        assert_eq!(skill.origin, SkillOrigin::AppSequence);
        assert!(skill.steps.len() >= 2);
    }

    #[test]
    fn test_discover_reinforces_existing() {
        let mut events = Vec::new();
        for round in 0..5 {
            let base = round as f64 * 100.0;
            events.push(app_seq(ts(base + 1.0), 1, 2, 1000));
            events.push(app_seq(ts(base + 2.0), 2, 3, 1000));
        }

        let buffer = make_buffer(events);
        let mut registry = SkillRegistry::new();
        let config = SkillConfig::default();

        // First run: discover
        let result1 = discover_skills(&buffer, &mut registry, &config, ts(600.0));
        assert!(!result1.new_skills.is_empty());
        let initial_confidence = registry.skills.values().next().unwrap().confidence;

        // Second run: reinforce
        let result2 = discover_skills(&buffer, &mut registry, &config, ts(700.0));
        assert!(!result2.reinforced.is_empty());
        let reinforced_confidence = registry.skills.values().next().unwrap().confidence;

        assert!(
            reinforced_confidence > initial_confidence,
            "Reinforcement should increase confidence"
        );
    }

    // ── § 4: Tool chain discovery ──

    #[test]
    fn test_discover_tool_chains() {
        let mut events = Vec::new();

        // Repeat tool chain: fetch_data → transform → store, 3 times
        for round in 0..3 {
            let base = round as f64 * 100.0;
            events.push(tool_call(ts(base + 1.0), "fetch_data", true));
            events.push(tool_call(ts(base + 2.0), "transform", true));
            events.push(tool_call(ts(base + 3.0), "store", true));
        }

        let buffer = make_buffer(events);
        let mut registry = SkillRegistry::new();
        let config = SkillConfig::default();

        let result = discover_skills(&buffer, &mut registry, &config, ts(500.0));

        let tool_skills: Vec<_> = result
            .new_skills
            .iter()
            .filter(|k| k.starts_with("tool_chain:"))
            .collect();
        assert!(!tool_skills.is_empty(), "Should discover tool chain skills");
    }

    #[test]
    fn test_tool_chain_broken_by_failure() {
        let mut events = Vec::new();

        // Chain with failure in middle (should break into separate chains)
        for round in 0..3 {
            let base = round as f64 * 100.0;
            events.push(tool_call(ts(base + 1.0), "fetch", true));
            events.push(tool_call(ts(base + 2.0), "broken", false)); // breaks chain
            events.push(tool_call(ts(base + 3.0), "store", true));
        }

        let buffer = make_buffer(events);
        let mut registry = SkillRegistry::new();
        let config = SkillConfig::default();

        let result = discover_skills(&buffer, &mut registry, &config, ts(500.0));

        // Should NOT find any 3-step chain because failure breaks it
        for key in &result.new_skills {
            if key.starts_with("tool_chain:") {
                let skill = registry.find(key).unwrap();
                assert!(skill.steps.len() <= 2, "Failure should break chain");
            }
        }
    }

    // ── § 5: Suggestion pattern discovery ──

    #[test]
    fn test_discover_suggestion_patterns() {
        let mut events = Vec::new();

        // 8 accepts, 2 rejects for "remind" → 80% acceptance
        for i in 0..8 {
            events.push(suggestion_accept(ts(i as f64), "remind"));
        }
        for i in 0..2 {
            events.push(suggestion_reject(ts(10.0 + i as f64), "remind"));
        }

        let buffer = make_buffer(events);
        let mut registry = SkillRegistry::new();
        let config = SkillConfig::default();

        let result = discover_skills(&buffer, &mut registry, &config, ts(500.0));

        let pattern_skills: Vec<_> = result
            .new_skills
            .iter()
            .filter(|k| k.starts_with("suggestion_pattern:"))
            .collect();
        assert!(
            !pattern_skills.is_empty(),
            "Should discover suggestion pattern"
        );

        let skill = registry.find("suggestion_pattern:remind").unwrap();
        assert!(
            skill.confidence > 0.3,
            "High acceptance should boost initial confidence"
        );
    }

    #[test]
    fn test_suggestion_pattern_low_acceptance_ignored() {
        let mut events = Vec::new();

        // 2 accepts, 8 rejects for "nag" → 20% acceptance
        for i in 0..2 {
            events.push(suggestion_accept(ts(i as f64), "nag"));
        }
        for i in 0..8 {
            events.push(suggestion_reject(ts(10.0 + i as f64), "nag"));
        }

        let buffer = make_buffer(events);
        let mut registry = SkillRegistry::new();
        let config = SkillConfig::default();

        let result = discover_skills(&buffer, &mut registry, &config, ts(500.0));

        assert!(
            !result
                .new_skills
                .contains(&"suggestion_pattern:nag".to_string()),
            "Low acceptance should not create skill"
        );
    }

    // ── § 6: Notification response discovery ──

    #[test]
    fn test_discover_notification_responses() {
        let mut events = Vec::new();

        // User consistently marks email notifications as "archive"
        for i in 0..5 {
            events.push(notif_acted(ts(i as f64), "email", "archive"));
        }

        let buffer = make_buffer(events);
        let mut registry = SkillRegistry::new();
        let config = SkillConfig::default();

        let result = discover_skills(&buffer, &mut registry, &config, ts(500.0));

        let notif_skills: Vec<_> = result
            .new_skills
            .iter()
            .filter(|k| k.starts_with("notif_response:"))
            .collect();
        assert!(
            !notif_skills.is_empty(),
            "Should discover notification response skill"
        );

        let skill = registry.find("notif_response:email→archive").unwrap();
        assert_eq!(skill.origin, SkillOrigin::NotificationResponse);
        assert_eq!(skill.steps.len(), 2);
    }

    // ── § 7: Auto-deprecation ──

    #[test]
    fn test_auto_deprecation() {
        let mut registry = SkillRegistry::new();
        let config = SkillConfig::default();

        let mut skill = LearnedSkill::new(
            "Bad skill".to_string(),
            SkillOrigin::AppSequence,
            "test:bad".to_string(),
            vec![],
            SkillTrigger::new(),
            10,
            ts(0.0),
        );
        skill.confidence = 0.8;
        skill.stage = SkillStage::Reliable;
        skill.offer_count = 10;
        skill.acceptance_count = 1; // 10% acceptance → below threshold
        registry.skills.insert("test:bad".to_string(), skill);

        let buffer = make_buffer(vec![]);
        let result = discover_skills(&buffer, &mut registry, &config, ts(500.0));

        assert!(
            result.auto_deprecated.contains(&"test:bad".to_string()),
            "Low acceptance should trigger auto-deprecation"
        );
        assert!(registry.find("test:bad").unwrap().deprecated);
    }

    // ── § 8: Candidate pruning ──

    #[test]
    fn test_candidate_pruning() {
        let mut registry = SkillRegistry::new();
        let config = SkillConfig::default();

        let skill = LearnedSkill::new(
            "Old candidate".to_string(),
            SkillOrigin::AppSequence,
            "test:old".to_string(),
            vec![],
            SkillTrigger::new(),
            1,
            ts(0.0),
        );
        registry.skills.insert("test:old".to_string(), skill);

        // Run discovery far in the future
        let buffer = make_buffer(vec![]);
        let future = ts(0.0) + config.candidate_ttl_secs + 1000.0;
        let result = discover_skills(&buffer, &mut registry, &config, future);

        assert!(
            result.pruned.contains(&"test:old".to_string()),
            "Expired candidates should be pruned"
        );
        assert!(registry.find("test:old").is_none());
    }

    // ── § 9: Max skills enforcement ──

    #[test]
    fn test_enforce_max_skills() {
        let mut registry = SkillRegistry::new();
        let mut config = SkillConfig::default();
        config.max_skills = 3;

        // Add 5 skills with varying confidence
        for i in 0..5 {
            let mut skill = LearnedSkill::new(
                format!("Skill {}", i),
                SkillOrigin::AppSequence,
                format!("test:max_{}", i),
                vec![],
                SkillTrigger::new(),
                1,
                ts(0.0),
            );
            skill.confidence = 0.3 + i as f64 * 0.1;
            registry.skills.insert(format!("test:max_{}", i), skill);
        }

        enforce_max_skills(&mut registry, &config);

        assert_eq!(registry.len(), 3, "Should enforce max_skills limit");

        // Highest confidence skills should survive
        assert!(registry.find("test:max_4").is_some());
        assert!(registry.find("test:max_3").is_some());
        assert!(registry.find("test:max_2").is_some());
    }

    // ── § 10: Skill matching ──

    #[test]
    fn test_find_matching_skills() {
        let mut registry = SkillRegistry::new();

        // Reliable skill that matches morning hours
        let mut skill1 = LearnedSkill::new(
            "Morning workflow".to_string(),
            SkillOrigin::AppSequence,
            "test:morning".to_string(),
            vec![SkillStep {
                ordinal: 0,
                action_kind: "app_open".to_string(),
                app_id: Some(14),
                tool_name: None,
                expected_duration_ms: 0,
                optional: false,
            }],
            SkillTrigger {
                time_bins: vec![8, 9, 10],
                initiating_app_id: Some(14),
                preceding_transition: None,
                day_of_week: vec![],
                min_session_duration_secs: None,
                preceded_by_idle: None,
            },
            10,
            ts(0.0),
        );
        skill1.confidence = 0.85;
        skill1.stage = SkillStage::Reliable;

        // Reliable skill for evening
        let mut skill2 = LearnedSkill::new(
            "Evening routine".to_string(),
            SkillOrigin::AppSequence,
            "test:evening".to_string(),
            vec![],
            SkillTrigger {
                time_bins: vec![20, 21, 22],
                initiating_app_id: None,
                preceding_transition: None,
                day_of_week: vec![],
                min_session_duration_secs: None,
                preceded_by_idle: None,
            },
            10,
            ts(0.0),
        );
        skill2.confidence = 0.75;
        skill2.stage = SkillStage::Reliable;

        registry.skills.insert("test:morning".to_string(), skill1);
        registry.skills.insert("test:evening".to_string(), skill2);

        // Query in the morning with app 14 open
        let matches = find_matching_skills(&registry, 9, 2, Some(14), None, 3600.0, false, 0.3, 10);

        assert!(!matches.is_empty());
        assert_eq!(
            matches[0].dedup_key, "test:morning",
            "Morning skill should rank highest"
        );
        assert!(matches[0].relevance > 0.5);
    }

    #[test]
    fn test_find_matching_excludes_non_offerable() {
        let mut registry = SkillRegistry::new();

        // Candidate skill (not offerable)
        let skill = LearnedSkill::new(
            "Candidate".to_string(),
            SkillOrigin::AppSequence,
            "test:candidate".to_string(),
            vec![],
            SkillTrigger::new(),
            1,
            ts(0.0),
        );
        registry.skills.insert("test:candidate".to_string(), skill);

        let matches = find_matching_skills(&registry, 12, 3, None, None, 0.0, false, 0.0, 10);

        assert!(matches.is_empty(), "Candidates should not be matched");
    }

    // ── § 11: Summary ──

    #[test]
    fn test_skill_summary() {
        let mut registry = SkillRegistry::new();
        registry.total_discovered = 5;
        registry.total_promoted = 1;
        registry.total_deprecated = 1;

        let mut s1 = LearnedSkill::new(
            "A".to_string(),
            SkillOrigin::AppSequence,
            "a".to_string(),
            vec![],
            SkillTrigger::new(),
            1,
            ts(0.0),
        );
        s1.stage = SkillStage::Reliable;
        registry.skills.insert("a".to_string(), s1);

        let mut s2 = LearnedSkill::new(
            "B".to_string(),
            SkillOrigin::ToolChain,
            "b".to_string(),
            vec![],
            SkillTrigger::new(),
            1,
            ts(0.0),
        );
        s2.stage = SkillStage::Candidate;
        registry.skills.insert("b".to_string(), s2);

        let mut s3 = LearnedSkill::new(
            "C".to_string(),
            SkillOrigin::AppSequence,
            "c".to_string(),
            vec![],
            SkillTrigger::new(),
            1,
            ts(0.0),
        );
        s3.deprecated = true;
        s3.stage = SkillStage::Deprecated;
        registry.skills.insert("c".to_string(), s3);

        let summary = summarize_skills(&registry);
        assert_eq!(summary.total, 3);
        assert_eq!(summary.active, 2);
        assert_eq!(summary.reliable, 1);
        assert_eq!(summary.candidates, 1);
        assert_eq!(summary.deprecated, 1);
        assert_eq!(summary.by_origin.get("app_sequence"), Some(&2));
        assert_eq!(summary.by_origin.get("tool_chain"), Some(&1));
    }

    // ── § 12: Registry operations ──

    #[test]
    fn test_registry_queries() {
        let mut registry = SkillRegistry::new();

        let mut s1 = LearnedSkill::new(
            "A".to_string(),
            SkillOrigin::AppSequence,
            "a".to_string(),
            vec![],
            SkillTrigger::new(),
            5,
            ts(0.0),
        );
        s1.confidence = 0.8;
        s1.stage = SkillStage::Reliable;
        registry.skills.insert("a".to_string(), s1);

        let s2 = LearnedSkill::new(
            "B".to_string(),
            SkillOrigin::ToolChain,
            "b".to_string(),
            vec![],
            SkillTrigger::new(),
            1,
            ts(0.0),
        );
        registry.skills.insert("b".to_string(), s2);

        assert_eq!(registry.active_count(), 2);
        assert_eq!(registry.offerable().len(), 1);
        assert_eq!(registry.by_origin(SkillOrigin::AppSequence).len(), 1);
        assert_eq!(registry.by_origin(SkillOrigin::ToolChain).len(), 1);
        assert!(registry.find("a").is_some());
        assert!(registry.find("nonexistent").is_none());
    }

    // ── § 13: Hash stability ──

    #[test]
    fn test_dedup_key_hash_stable() {
        let h1 = hash_dedup_key("app_seq:14→20→15");
        let h2 = hash_dedup_key("app_seq:14→20→15");
        let h3 = hash_dedup_key("app_seq:14→20→16");

        assert_eq!(h1, h2, "Same key should produce same hash");
        assert_ne!(h1, h3, "Different keys should produce different hashes");
    }

    // ── § 14: Trigger inference ──

    #[test]
    fn test_trigger_inference() {
        // Events clustered around 9am (hour 9 = second 32400)
        let timestamps: Vec<f64> = (0..10)
            .map(|i| {
                86400.0 * (100 + i) as f64 + 32400.0 // 9:00 each day
            })
            .collect();

        let trigger = infer_trigger_from_timestamps(&timestamps, Some(14));

        assert!(
            trigger.time_bins.contains(&9),
            "Should detect 9am peak: {:?}",
            trigger.time_bins
        );
        assert_eq!(trigger.initiating_app_id, Some(14));
    }

    #[test]
    fn test_trigger_empty_timestamps() {
        let trigger = infer_trigger_from_timestamps(&[], None);
        assert_eq!(trigger.specificity(), 0);
    }

    // ── § 15: Edge cases ──

    #[test]
    fn test_empty_buffer_discovery() {
        let buffer = make_buffer(vec![]);
        let mut registry = SkillRegistry::new();
        let config = SkillConfig::default();

        let result = discover_skills(&buffer, &mut registry, &config, ts(0.0));

        assert!(result.new_skills.is_empty());
        assert!(result.reinforced.is_empty());
        assert_eq!(result.total_skills, 0);
    }

    #[test]
    fn test_below_minimum_observations() {
        let mut events = Vec::new();
        // Only 2 occurrences (below default min of 3)
        for round in 0..2 {
            let base = round as f64 * 100.0;
            events.push(app_seq(ts(base + 1.0), 14, 20, 2000));
        }

        let buffer = make_buffer(events);
        let mut registry = SkillRegistry::new();
        let config = SkillConfig::default();

        let result = discover_skills(&buffer, &mut registry, &config, ts(500.0));
        assert!(
            result.new_skills.is_empty(),
            "Below min_observations should not create skill"
        );
    }

    #[test]
    fn test_skill_stage_from_confidence() {
        assert_eq!(
            SkillStage::from_confidence(0.3, 0.0, false),
            SkillStage::Candidate
        );
        assert_eq!(
            SkillStage::from_confidence(0.6, 0.0, false),
            SkillStage::Validated
        );
        assert_eq!(
            SkillStage::from_confidence(0.8, 0.0, false),
            SkillStage::Reliable
        );
        assert_eq!(
            SkillStage::from_confidence(0.95, 0.0, false),
            SkillStage::Promoted
        );
        assert_eq!(
            SkillStage::from_confidence(0.8, 31.0 * 86400.0, false),
            SkillStage::Stale
        );
        assert_eq!(
            SkillStage::from_confidence(0.8, 0.0, true),
            SkillStage::Deprecated
        );
    }
}
