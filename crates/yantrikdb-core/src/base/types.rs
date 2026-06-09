use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Embedder trait ──

/// Trait for converting text to embedding vectors.
/// Implementations can use any embedding model (sentence-transformers, candle, etc.).
pub trait Embedder: Send + Sync {
    /// Embed a single text string into a vector.
    fn embed(
        &self,
        text: &str,
    ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>>;

    /// Embed multiple texts. Default implementation calls embed() in a loop.
    fn embed_batch(
        &self,
        texts: &[&str],
    ) -> std::result::Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// The dimensionality of produced embeddings.
    fn dim(&self) -> usize;

    /// Stable identity of this embedder. SHA-256 of model weights, or
    /// equivalent fingerprint that distinguishes one model from another
    /// even when they share dim. Default `None` for back-compat with
    /// existing third-party Embedder impls.
    ///
    /// Used by `db.set_embedder*` (issue #41) to distinguish:
    /// - same-model-replacement (matching fingerprint, matching dim) — safe Arc swap
    /// - different-model-same-dim (different fingerprint, same dim) —
    ///   silent-corruption risk, rejected on populated `Known`-provenance
    ///   DBs with `ChangeEmbedderDigestRequiresReembed`
    ///
    /// Embedders returning `None` are treated as `ExternalOrUnknown`
    /// provenance — they may attach to empty or unknown-provenance DBs
    /// but cannot attach to a `Known`-provenance populated DB without
    /// going through `db.reembed()`. Conservative-correct: a custom
    /// embedder without identity cannot prove compatibility with
    /// previously-indexed vectors.
    ///
    /// Bundled embedders + the `set_embedder_named` named-download path
    /// override this with real fingerprints (SHA-256 from the embedder-
    /// download registry).
    fn fingerprint(&self) -> Option<String> {
        None
    }

    /// Human-readable name of this embedder (e.g. "potion-base-2M").
    /// Independent of fingerprint identity; used for observability and
    /// status reporting. Default `None` for back-compat. Named-download
    /// embedders override with the registry name.
    fn name(&self) -> Option<String> {
        None
    }
}

/// A memory record returned by get() and recall().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub rid: String,
    pub memory_type: String,
    pub text: String,
    pub created_at: f64,
    pub importance: f64,
    pub valence: f64,
    pub half_life: f64,
    pub last_access: f64,
    pub access_count: u32,
    pub consolidation_status: String,
    pub storage_tier: String,
    pub consolidated_into: Option<String>,
    pub metadata: serde_json::Value,
    pub namespace: String,
    // Cognitive dimensions (V10)
    pub certainty: f64,
    pub domain: String,
    pub source: String,
    pub emotional_state: Option<String>,
    // Session & temporal (V13)
    pub session_id: Option<String>,
    pub due_at: Option<f64>,
    pub temporal_kind: Option<String>,
}

/// Score breakdown for a recall result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub similarity: f64,
    pub decay: f64,
    pub recency: f64,
    pub importance: f64,
    pub graph_proximity: f64,
    /// Weighted contribution of each signal to the final score.
    pub contributions: ScoreContributions,
    /// Valence multiplier applied to the raw score.
    pub valence_multiplier: f64,
}

/// Weighted contributions of each signal (signal_value * weight).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreContributions {
    pub similarity: f64,
    pub decay: f64,
    pub recency: f64,
    pub importance: f64,
    pub graph_proximity: f64,
}

/// A recall result with scoring information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    pub rid: String,
    pub memory_type: String,
    pub text: String,
    pub created_at: f64,
    pub importance: f64,
    pub valence: f64,
    pub score: f64,
    pub scores: ScoreBreakdown,
    pub why_retrieved: Vec<String>,
    pub metadata: serde_json::Value,
    pub namespace: String,
    // Cognitive dimensions (V10)
    pub certainty: f64,
    pub domain: String,
    pub source: String,
    pub emotional_state: Option<String>,
}

/// Response from recall with confidence and hints for interactive retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResponse {
    pub results: Vec<RecallResult>,
    pub confidence: f64,
    /// Human-readable explanation of what drove the confidence score.
    pub certainty_reasons: Vec<String>,
    pub retrieval_summary: RetrievalSummary,
    pub hints: Vec<RefinementHint>,
}

/// Summary of how retrieval was performed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalSummary {
    pub top_similarity: f64,
    pub score_spread: f64,
    pub sources_used: Vec<String>,
    pub candidate_count: usize,
}

/// A hint for refining a query when confidence is low.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementHint {
    pub hint_type: String,
    pub suggestion: String,
    pub related_entities: Vec<String>,
}

/// An edge in the entity graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub edge_id: String,
    pub src: String,
    pub dst: String,
    pub rel_type: String,
    pub weight: f64,
}

/// An entity in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    pub entity_type: String,
    pub first_seen: f64,
    pub last_seen: f64,
    pub mention_count: i64,
}

/// Engine statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stats {
    pub active_memories: i64,
    pub consolidated_memories: i64,
    pub tombstoned_memories: i64,
    pub archived_memories: i64,
    pub edges: i64,
    pub entities: i64,
    pub operations: i64,
    pub open_conflicts: i64,
    pub resolved_conflicts: i64,
    pub pending_triggers: i64,
    pub active_patterns: i64,
    pub scoring_cache_entries: usize,
    pub vec_index_entries: usize,
    pub graph_index_entities: usize,
    pub graph_index_edges: usize,
}

/// A proactive trigger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trigger {
    pub trigger_type: String,
    pub reason: String,
    pub urgency: f64,
    pub source_rids: Vec<String>,
    pub suggested_action: String,
    pub context: HashMap<String, serde_json::Value>,
}

/// Consolidation result (after consolidation runs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationResult {
    pub consolidated_rid: String,
    pub source_rids: Vec<String>,
    pub cluster_size: usize,
    pub summary: String,
    pub importance: f64,
    pub entities_linked: usize,
}

/// Dry run consolidation preview.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationPreview {
    pub cluster_size: usize,
    pub texts: Vec<String>,
    pub preview_summary: String,
    pub source_rids: Vec<String>,
}

/// Internal struct with embedding data for clustering.
#[derive(Debug, Clone)]
pub struct MemoryWithEmbedding {
    pub rid: String,
    pub memory_type: String,
    pub text: String,
    pub embedding: Vec<f32>,
    pub created_at: f64,
    pub importance: f64,
    pub valence: f64,
    pub half_life: f64,
    pub last_access: f64,
    pub metadata: serde_json::Value,
    pub namespace: String,
}

/// A decayed memory candidate from decay().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayedMemory {
    pub rid: String,
    pub text: String,
    pub memory_type: String,
    pub original_importance: f64,
    pub current_score: f64,
    pub days_since_access: f64,
}

/// Lightweight scoring fields cached in memory for fast recall scoring.
/// These are the only fields needed to compute composite_score() during recall.
#[derive(Debug, Clone)]
pub struct ScoringRow {
    pub created_at: f64,
    pub importance: f64,
    pub half_life: f64,
    pub last_access: f64,
    pub access_count: u32,
    pub valence: f64,
    pub consolidation_status: String,
    pub memory_type: String,
    pub namespace: String,
    // Cognitive dimensions (V10)
    pub certainty: f64,
    pub domain: String,
    pub source: String,
    pub emotional_state: Option<String>,
}

/// Input for batch record operations.
#[derive(Debug, Clone)]
pub struct RecordInput {
    pub text: String,
    pub memory_type: String,
    pub importance: f64,
    pub valence: f64,
    pub half_life: f64,
    pub metadata: serde_json::Value,
    pub embedding: Vec<f32>,
    pub namespace: String,
    // Cognitive dimensions (V10)
    pub certainty: f64,
    pub domain: String,
    pub source: String,
    pub emotional_state: Option<String>,
}

// ── Conflict types (V2) ──

/// The type of semantic conflict between two memories.
#[derive(Debug, Clone, PartialEq)]
pub enum ConflictType {
    IdentityFact,
    Preference,
    Temporal,
    Consolidation,
    Minor,
}

impl ConflictType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ConflictType::IdentityFact => "identity_fact",
            ConflictType::Preference => "preference",
            ConflictType::Temporal => "temporal",
            ConflictType::Consolidation => "consolidation",
            ConflictType::Minor => "minor",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "identity_fact" => ConflictType::IdentityFact,
            "preference" => ConflictType::Preference,
            "temporal" => ConflictType::Temporal,
            "consolidation" => ConflictType::Consolidation,
            _ => ConflictType::Minor,
        }
    }

    pub fn default_priority(&self) -> &'static str {
        match self {
            ConflictType::IdentityFact => "critical",
            ConflictType::Preference => "high",
            ConflictType::Temporal => "high",
            ConflictType::Consolidation => "medium",
            ConflictType::Minor => "low",
        }
    }
}

/// A conflict between two memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    pub conflict_id: String,
    pub conflict_type: String,
    pub priority: String,
    pub status: String,
    pub memory_a: String,
    pub memory_b: String,
    pub entity: Option<String>,
    pub rel_type: Option<String>,
    pub detected_at: f64,
    pub detected_by: String,
    pub detection_reason: String,
    pub resolved_at: Option<f64>,
    pub resolved_by: Option<String>,
    pub strategy: Option<String>,
    pub winner_rid: Option<String>,
    pub resolution_note: Option<String>,
}

/// Result of a conflict resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictResolutionResult {
    pub conflict_id: String,
    pub strategy: String,
    pub winner_rid: Option<String>,
    pub loser_tombstoned: bool,
    pub new_memory_rid: Option<String>,
}

/// Result of a user-initiated correction.
///
/// **Issue #47 (v0.7.20):** `correct()` now mutates the memory in place,
/// preserving `rid` and `created_at`. `original_rid` and `corrected_rid`
/// are therefore always equal; `original_tombstoned` is always `false`.
/// The fields are kept for back-compat with v0.7.19-and-earlier consumers
/// that destructured this struct. `revision_num` is new — it tells the
/// caller which revision number the correction wrote (1-indexed,
/// monotonically increasing per-rid).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionResult {
    pub original_rid: String,
    pub corrected_rid: String,
    pub original_tombstoned: bool,
    pub revision_num: i64,
}

/// A single entry from a `correct()` revision history query.
/// See `YantrikDB::history()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordRevision {
    pub revision_id: String,
    pub rid: String,
    pub revision_num: i64,
    pub prior_text: String,
    pub prior_metadata: serde_json::Value,
    pub prior_importance: f64,
    pub prior_valence: f64,
    pub reason: String,
    pub applied_at: f64,
    pub origin_actor: String,
}

// ── Issue #48 — record-to-record link model (schema v31) ──

/// Closed set of record-to-record link types + a `Custom` escape hatch.
///
/// See docs/record_link_model_rfc.md. The recall-time proximity polarity
/// (whether surfacing one endpoint should boost or demote the other) is a
/// pure function of the variant — see [`LinkType::recall_polarity`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkType {
    /// This record builds on / improves the target (newer is better).
    Advances,
    /// This record replaces the target. Does NOT auto-tombstone the
    /// target (semantic claim, not a lifecycle action). Recall demotes
    /// the superseded predecessor.
    Supersedes,
    /// This record disagrees with the target. Quasi-symmetric — queried
    /// bidirectionally.
    Contradicts,
    /// This record provides evidence for the target.
    Supports,
    /// This record raises a question about the target.
    Questions,
    /// This record was derived from / inspired by the target (provenance,
    /// no correctness claim — distinct from Advances).
    DerivedFrom,
    /// Extensibility hatch. Stored as `custom:<name>`; engine does not
    /// interpret the name (treated as a weak positive at recall time).
    Custom(String),
}

/// How a link affects recall when traversing from one endpoint to the
/// other. Multiplier applied to the graph-proximity contribution of the
/// linked record; <1.0 demotes, >0 with <1 dampens, 1.0 is neutral-positive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkRecallPolarity {
    /// Multiplier for the LINKED (target/other-endpoint) record's
    /// proximity contribution.
    pub neighbor_factor: f64,
    /// Multiplier applied to the SEED record's own score when it is the
    /// target of this link type (used by Supersedes to demote a
    /// superseded predecessor). 1.0 = no self-demotion.
    pub demote_self_as_target: f64,
}

impl LinkType {
    /// Serialize to the canonical string stored in `record_links.link_type`.
    pub fn as_str(&self) -> String {
        match self {
            LinkType::Advances => "advances".to_string(),
            LinkType::Supersedes => "supersedes".to_string(),
            LinkType::Contradicts => "contradicts".to_string(),
            LinkType::Supports => "supports".to_string(),
            LinkType::Questions => "questions".to_string(),
            LinkType::DerivedFrom => "derived_from".to_string(),
            LinkType::Custom(name) => format!("custom:{name}"),
        }
    }

    /// Parse from the stored string. `custom:<name>` round-trips to
    /// `Custom(name)`; any other unrecognized string also becomes
    /// `Custom(...)` so forward-compat strings from a newer node don't
    /// hard-error on an older one.
    pub fn from_str_lenient(s: &str) -> LinkType {
        match s {
            "advances" => LinkType::Advances,
            "supersedes" => LinkType::Supersedes,
            "contradicts" => LinkType::Contradicts,
            "supports" => LinkType::Supports,
            "questions" => LinkType::Questions,
            "derived_from" => LinkType::DerivedFrom,
            other => {
                let name = other.strip_prefix("custom:").unwrap_or(other);
                LinkType::Custom(name.to_string())
            }
        }
    }

    /// Whether this link type is treated as undirected at traversal /
    /// recall time. Only `Contradicts` is symmetric.
    pub fn is_symmetric(&self) -> bool {
        matches!(self, LinkType::Contradicts)
    }

    /// Recall-time polarity. Pure function of the variant. See the
    /// relation-aware-scoring table in the RFC.
    pub fn recall_polarity(&self) -> LinkRecallPolarity {
        match self {
            // Positive: pull the neighbor in at full proximity.
            LinkType::Supports | LinkType::Advances => LinkRecallPolarity {
                neighbor_factor: 1.0,
                demote_self_as_target: 1.0,
            },
            // Provenance: weaker positive.
            LinkType::DerivedFrom => LinkRecallPolarity {
                neighbor_factor: 0.6,
                demote_self_as_target: 1.0,
            },
            // Supersedes: boost the successor (neighbor) AND demote the
            // superseded predecessor when IT is the surfaced record.
            LinkType::Supersedes => LinkRecallPolarity {
                neighbor_factor: 1.0,
                demote_self_as_target: 0.5,
            },
            // Relevant-but-not-endorsing: surface the contradictor so the
            // caller sees the conflict, but no positive relevance halo.
            LinkType::Contradicts => LinkRecallPolarity {
                neighbor_factor: 0.3,
                demote_self_as_target: 1.0,
            },
            // Weak relevance / uncertainty.
            LinkType::Questions => LinkRecallPolarity {
                neighbor_factor: 0.3,
                demote_self_as_target: 1.0,
            },
            // Engine doesn't interpret custom — weak positive.
            LinkType::Custom(_) => LinkRecallPolarity {
                neighbor_factor: 0.5,
                demote_self_as_target: 1.0,
            },
        }
    }
}

/// A record-to-record link to create alongside (or after) a record write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordLink {
    pub target_rid: String,
    pub link_type: LinkType,
}

/// Direction filter for [`YantrikDB::linked_records`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkDirection {
    Outbound,
    Inbound,
    Both,
}

/// A single result from a `linked_records` traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkedRecord {
    pub rid: String,
    pub link_type: String,
    pub created_at: f64,
    /// "outbound" or "inbound" relative to the queried rid.
    pub direction: String,
}

/// Per-link outcome from `record_with_links_partial` (issue #48). The
/// record always commits first (durable via the oplog); each link is
/// then attempted independently — a `Failed` on one does not abort the
/// others or fail the call. Callers get the rid + a full per-link vec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LinkResult {
    /// Link row was newly inserted.
    Inserted {
        target_rid: String,
        link_type: String,
    },
    /// Link already existed (UNIQUE(source,target,type) hit; INSERT OR
    /// IGNORE was a no-op). Retry-safe; algo treats it like `Inserted`
    /// but it's distinguished for telemetry.
    AlreadyExists {
        target_rid: String,
        link_type: String,
    },
    /// Link insert failed; the record itself is still durable.
    Failed {
        target_rid: String,
        link_type: String,
        error: String,
    },
}

/// Result of [`YantrikDB::audit_leak_candidates`] (issue #48 follow-up).
///
/// Replaces the broken "orphan" metric (memories lacking oplog presence),
/// which the trader postmortem proved is a benign **oplog-compaction
/// artifact**, not a leak: locally-originated memories whose oplog rows
/// aged out of the retention window look orphan-shaped but are healthy.
///
/// The windowed check only flags memories created *within* the surviving
/// oplog window (`created_at >= window_floor`, where `window_floor =
/// MIN(oplog.timestamp)`) that nonetheless have no oplog op and no
/// `replication_apply_log` entry. Inside the window the oplog row SHOULD
/// still exist, so its absence is a genuine signal (write-path bug /
/// direct-SQL injection). Outside the window, absence is expected
/// compaction and is NOT counted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakAuditReport {
    /// `MIN(oplog.timestamp)` — the compaction boundary. `None` if the
    /// oplog is empty (no window can be established; `candidate_count` 0).
    pub window_floor: Option<f64>,
    /// Exact count of in-window leak candidates.
    pub candidate_count: usize,
    /// Up to `max_rids` candidate rids (the count is exact; this is a
    /// bounded sample for investigation).
    pub candidate_rids: Vec<String>,
}

// ── Cognition types (V3) ──

/// Trigger type classification.
#[derive(Debug, Clone, PartialEq)]
pub enum TriggerType {
    DecayReview,
    ConsolidationReady,
    ConflictEscalation,
    TemporalDrift,
    Redundancy,
    RelationshipInsight,
    ValenceTrend,
    EntityAnomaly,
    PatternDiscovered,
}

impl TriggerType {
    pub fn as_str(&self) -> &'static str {
        match self {
            TriggerType::DecayReview => "decay_review",
            TriggerType::ConsolidationReady => "consolidation_ready",
            TriggerType::ConflictEscalation => "conflict_escalation",
            TriggerType::TemporalDrift => "temporal_drift",
            TriggerType::Redundancy => "redundancy",
            TriggerType::RelationshipInsight => "relationship_insight",
            TriggerType::ValenceTrend => "valence_trend",
            TriggerType::EntityAnomaly => "entity_anomaly",
            TriggerType::PatternDiscovered => "pattern_discovered",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "decay_review" => TriggerType::DecayReview,
            "consolidation_ready" => TriggerType::ConsolidationReady,
            "conflict_escalation" => TriggerType::ConflictEscalation,
            "temporal_drift" => TriggerType::TemporalDrift,
            "redundancy" => TriggerType::Redundancy,
            "relationship_insight" => TriggerType::RelationshipInsight,
            "valence_trend" => TriggerType::ValenceTrend,
            "entity_anomaly" => TriggerType::EntityAnomaly,
            "pattern_discovered" => TriggerType::PatternDiscovered,
            _ => TriggerType::DecayReview,
        }
    }

    pub fn default_cooldown_secs(&self) -> f64 {
        match self {
            TriggerType::DecayReview => 86400.0 * 3.0,
            TriggerType::ConsolidationReady => 86400.0,
            TriggerType::ConflictEscalation => 86400.0 * 2.0,
            TriggerType::TemporalDrift => 86400.0 * 14.0,
            TriggerType::Redundancy => 86400.0,
            TriggerType::RelationshipInsight => 86400.0 * 7.0,
            TriggerType::ValenceTrend => 86400.0 * 7.0,
            TriggerType::EntityAnomaly => 86400.0 * 7.0,
            TriggerType::PatternDiscovered => 86400.0 * 7.0,
        }
    }

    pub fn default_expiry_secs(&self) -> f64 {
        match self {
            TriggerType::DecayReview => 86400.0 * 7.0,
            TriggerType::ConsolidationReady => 86400.0 * 3.0,
            TriggerType::ConflictEscalation => 86400.0 * 14.0,
            _ => 86400.0 * 7.0,
        }
    }
}

/// Configuration for the think() cognition loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkConfig {
    pub importance_threshold: f64,
    pub decay_threshold: f64,
    pub max_triggers: usize,
    pub run_consolidation: bool,
    pub run_conflict_scan: bool,
    pub run_pattern_mining: bool,
    pub consolidation_sim_threshold: f64,
    pub consolidation_time_window_days: f64,
    pub consolidation_min_cluster: usize,
    pub consolidation_limit: usize,
    /// When true, consolidation requires candidate pairs to share at least one
    /// extracted entity (falls back to cosine-only when either side has no
    /// entities). Guards against merging semantically similar sentences that
    /// refer to different subjects.
    pub consolidation_require_entity_overlap: bool,
    pub min_active_memories: i64,
    pub run_personality: bool,
    /// **v0.7.23 prototype.** When true, `think()` extracts simple
    /// "<subject> is <value>" assertions from free-text memories into the
    /// claim layer so `scan_claim_conflicts` can detect attribute-value
    /// updates made via plain `record()` (e.g. "brand color is blue" →
    /// "brand color is now green"). Off by default — opt-in because the
    /// copular heuristic can flag non-exclusive values for review. Skipped
    /// when encryption is enabled (stored text is ciphertext).
    pub extract_attribute_claims: bool,
}

impl Default for ThinkConfig {
    fn default() -> Self {
        Self {
            importance_threshold: 0.5,
            decay_threshold: 0.1,
            max_triggers: 10,
            run_consolidation: true,
            run_conflict_scan: true,
            run_pattern_mining: false,
            consolidation_sim_threshold: 0.6,
            consolidation_time_window_days: 7.0,
            consolidation_min_cluster: 2,
            consolidation_limit: 5,
            consolidation_require_entity_overlap: true,
            min_active_memories: 10,
            run_personality: true,
            extract_attribute_claims: false,
        }
    }
}

/// Result of a think() pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkResult {
    pub triggers: Vec<Trigger>,
    pub consolidation_count: usize,
    pub conflicts_found: usize,
    pub patterns_new: usize,
    pub patterns_updated: usize,
    pub expired_triggers: usize,
    pub personality_updated: bool,
    pub duration_ms: u64,
}

/// A persisted trigger with lifecycle state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedTrigger {
    pub trigger_id: String,
    pub trigger_type: String,
    pub urgency: f64,
    pub status: String,
    pub reason: String,
    pub suggested_action: String,
    pub source_rids: Vec<String>,
    pub context: serde_json::Value,
    pub created_at: f64,
    pub delivered_at: Option<f64>,
    pub acknowledged_at: Option<f64>,
    pub acted_at: Option<f64>,
    pub expires_at: Option<f64>,
}

/// A detected pattern across memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub pattern_id: String,
    pub pattern_type: String,
    pub status: String,
    pub confidence: f64,
    pub description: String,
    pub evidence_rids: Vec<String>,
    pub entity_names: Vec<String>,
    pub context: serde_json::Value,
    pub first_seen: f64,
    pub last_confirmed: f64,
    pub occurrence_count: i64,
}

/// Result of pattern mining.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMiningResult {
    pub new_patterns: usize,
    pub updated_patterns: usize,
    pub stale_patterns: usize,
}

/// Configuration for pattern mining.
#[derive(Debug, Clone)]
pub struct PatternConfig {
    pub co_occurrence_min_count: usize,
    pub temporal_cluster_min_events: usize,
    pub valence_trend_delta_threshold: f64,
    pub topic_cluster_sim_threshold: f64,
    pub topic_cluster_time_window_days: f64,
    pub entity_hub_min_degree: usize,
    pub max_patterns: usize,
    // Cross-domain mining (V13)
    pub cross_domain_candidates_per_domain: usize,
    pub cross_domain_sim_threshold: f64,
    pub cross_domain_max_per_pair: usize,
    pub entity_bridge_min_domains: usize,
    pub entity_bridge_min_mentions_per_domain: usize,
    pub run_cross_domain: bool,
}

// ── Profiling types (feature-gated) ──

/// Timing breakdown for a single recall() invocation.
#[cfg(feature = "profiling")]
#[derive(Debug, Clone)]
pub struct RecallTimings {
    pub vec_search_ms: f64,
    pub cache_score_ms: f64,
    pub fetch_ms: f64,
    pub scoring_ms: f64,
    pub graph_ms: f64,
    pub reinforce_ms: f64,
    pub sort_truncate_ms: f64,
    pub total_ms: f64,
    pub candidate_count: usize,
    pub graph_expansion_count: usize,
}

/// Result of recall_profiled() — recall results plus timing breakdown.
#[cfg(feature = "profiling")]
#[derive(Debug, Clone)]
pub struct RecallProfiledResult {
    pub results: Vec<RecallResult>,
    pub timings: RecallTimings,
}

/// Builder for composable recall queries.
///
/// ```rust,ignore
/// let results = db.query(embedding)
///     .top_k(10)
///     .memory_type("episodic")
///     .namespace("work")
///     .expand_entities("tell me about Alice")
///     .time_window(start, end)
///     .execute()?;
/// ```
#[derive(Debug, Clone)]
pub struct RecallQuery {
    pub embedding: Vec<f32>,
    pub top_k: usize,
    pub time_window: Option<(f64, f64)>,
    pub memory_type: Option<String>,
    pub include_consolidated: bool,
    pub expand_entities: bool,
    pub query_text: Option<String>,
    pub skip_reinforce: bool,
    pub namespace: Option<String>,
    // V10 filters
    pub domain: Option<String>,
    pub source: Option<String>,
    // Issue #46: confidence first-class on recall.
    pub certainty_min: Option<f64>,
    pub order: Option<String>,
}

impl RecallQuery {
    /// Create a new query builder with the given embedding vector.
    pub fn new(embedding: Vec<f32>) -> Self {
        Self {
            embedding,
            top_k: 10,
            time_window: None,
            memory_type: None,
            include_consolidated: false,
            expand_entities: false,
            query_text: None,
            skip_reinforce: false,
            namespace: None,
            domain: None,
            source: None,
            certainty_min: None,
            order: None,
        }
    }

    /// Issue #46: drop results whose certainty falls below `min`.
    pub fn certainty_min(mut self, min: f64) -> Self {
        self.certainty_min = Some(min);
        self
    }

    /// Issue #46: re-sort the top_k by `"relevance"` (default),
    /// `"certainty"`, or `"recency"`. Invalid strings surface as
    /// `YantrikDbError::InvalidInput` on `execute()`.
    pub fn order(mut self, order: &str) -> Self {
        self.order = Some(order.to_string());
        self
    }

    /// Set maximum number of results to return.
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Filter by memory type (e.g., "episodic", "semantic", "procedural").
    pub fn memory_type(mut self, mt: &str) -> Self {
        self.memory_type = Some(mt.to_string());
        self
    }

    /// Filter by namespace.
    pub fn namespace(mut self, ns: &str) -> Self {
        self.namespace = Some(ns.to_string());
        self
    }

    /// Restrict results to a time window (created_at between start and end).
    pub fn time_window(mut self, start: f64, end: f64) -> Self {
        self.time_window = Some((start, end));
        self
    }

    /// Enable graph expansion with the given query text for entity extraction.
    pub fn expand_entities(mut self, query_text: &str) -> Self {
        self.expand_entities = true;
        self.query_text = Some(query_text.to_string());
        self
    }

    /// Include consolidated (merged) memories in results.
    pub fn include_consolidated(mut self) -> Self {
        self.include_consolidated = true;
        self
    }

    /// Skip spaced-repetition reinforcement on accessed memories.
    pub fn skip_reinforce(mut self) -> Self {
        self.skip_reinforce = true;
        self
    }

    /// Filter by domain (e.g., "work", "health", "family").
    pub fn domain(mut self, d: &str) -> Self {
        self.domain = Some(d.to_string());
        self
    }

    /// Filter by source (e.g., "user", "system", "document", "inference").
    pub fn source(mut self, s: &str) -> Self {
        self.source = Some(s.to_string());
        self
    }
}

/// Learned scoring weights stored per-database for adaptive recall.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedWeights {
    pub w_sim: f64,
    pub w_decay: f64,
    pub w_recency: f64,
    pub gate_tau: f64,
    pub alpha_imp: f64,
    pub keyword_boost: f64,
    pub generation: i64,
}

impl Default for LearnedWeights {
    fn default() -> Self {
        Self {
            w_sim: 0.50,
            w_decay: 0.20,
            w_recency: 0.30,
            gate_tau: 0.25,
            alpha_imp: 0.80,
            keyword_boost: 0.31,
            generation: 0,
        }
    }
}

// ── Personality types (V11) ──

/// A single personality trait with its current score and derivation metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityTrait {
    pub trait_name: String,
    pub score: f64,
    pub confidence: f64,
    pub sample_count: i64,
    pub updated_at: f64,
}

/// Aggregated personality profile across all traits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityProfile {
    pub traits: Vec<PersonalityTrait>,
    pub updated_at: f64,
}

// ── Session types (V13) ──

/// A session tracks a conversation or interaction period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub session_id: String,
    pub namespace: String,
    pub client_id: String,
    pub status: String,
    pub started_at: f64,
    pub ended_at: Option<f64>,
    pub summary: Option<String>,
    pub avg_valence: Option<f64>,
    pub memory_count: i64,
    pub topics: Vec<String>,
    pub metadata: serde_json::Value,
}

/// Summary returned when ending a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub duration_secs: f64,
    pub memory_count: i64,
    pub avg_valence: f64,
    pub topics: Vec<String>,
}

// ── Temporal & Entity Profile types (V13) ──

/// Rich profile of an entity across time, domains, and sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityProfile {
    pub entity: String,
    pub entity_type: String,
    pub mention_count: i64,
    pub session_count: i64,
    pub domains: Vec<DomainCount>,
    pub avg_valence: f64,
    pub valence_trend: f64,
    pub dominant_emotion: Option<String>,
    pub interaction_frequency: f64,
    pub last_mentioned_at: f64,
    pub first_seen: f64,
    pub window_days: f64,
}

/// Count of mentions within a domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainCount {
    pub domain: String,
    pub count: i64,
}

// ── Cross-domain mining types (V13) ──

/// A link between memories in different domains discovered by cross-domain mining.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDomainLink {
    pub rid_a: String,
    pub rid_b: String,
    pub domain_a: String,
    pub domain_b: String,
    pub similarity: f64,
    pub text_a: String,
    pub text_b: String,
    pub score: f64,
}

/// An entity that bridges multiple domains.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityBridge {
    pub entity: String,
    pub domains: Vec<DomainCount>,
    pub bridge_score: f64,
    pub total_mentions: i64,
}

// ── Relationship depth types (V14) ──

/// Rich interaction metrics for an entity, measuring depth of knowledge.
/// This goes beyond simple mention counts to capture how deeply the system
/// knows about an entity across sessions, domains, and time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipDepth {
    /// The entity name.
    pub entity: String,
    /// The entity type (person, organization, tech, etc.).
    pub entity_type: String,
    /// Number of distinct sessions where this entity appeared.
    pub sessions_together: i64,
    /// Total memories mentioning this entity.
    pub memories_mentioning: i64,
    /// Average valence of memories involving this entity.
    pub avg_valence: f64,
    /// Domains this entity spans (e.g., ["work", "health", "family"]).
    pub domains_spanning: Vec<String>,
    /// Distinct relationship types connected to this entity.
    pub relationship_types: Vec<String>,
    /// Number of distinct entities this entity is connected to in the graph.
    pub connection_count: i64,
    /// Composite depth score (0.0-1.0): higher = deeper relationship.
    /// Combines sessions, memories, domain breadth, connection count.
    pub depth_score: f64,
    /// When this entity was first seen.
    pub first_seen: f64,
    /// When this entity was last seen.
    pub last_seen: f64,
    /// Mentions per day since first seen.
    pub interaction_frequency: f64,
}

// ── Substitution categories (V14) ──

/// A substitution category (e.g., "databases", "cloud_providers").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstitutionCategory {
    pub id: String,
    pub name: String,
    pub conflict_mode: String,
    pub status: String,
    pub member_count: i64,
}

/// A member of a substitution category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstitutionMember {
    pub id: String,
    pub category_name: String,
    pub token_normalized: String,
    pub token_display: String,
    pub confidence: f64,
    pub source: String,
    pub status: String,
}

/// Result of reclassifying a conflict (learning entry point).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReclassifyResult {
    pub conflict_id: String,
    pub old_type: String,
    pub new_type: String,
    pub learned_members: Vec<LearnedMember>,
    pub category_created: Option<String>,
}

/// A member learned during conflict reclassification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedMember {
    pub token: String,
    pub category_name: String,
    pub is_new: bool,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            co_occurrence_min_count: 3,
            temporal_cluster_min_events: 3,
            valence_trend_delta_threshold: 0.3,
            topic_cluster_sim_threshold: 0.55,
            topic_cluster_time_window_days: 30.0,
            entity_hub_min_degree: 5,
            max_patterns: 50,
            cross_domain_candidates_per_domain: 15,
            cross_domain_sim_threshold: 0.50,
            cross_domain_max_per_pair: 3,
            entity_bridge_min_domains: 2,
            entity_bridge_min_mentions_per_domain: 3,
            run_cross_domain: true,
        }
    }
}
