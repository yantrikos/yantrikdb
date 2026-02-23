use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    pub consolidation_status: String,
    pub consolidated_into: Option<String>,
    pub metadata: serde_json::Value,
}

/// Score breakdown for a recall result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub similarity: f64,
    pub decay: f64,
    pub recency: f64,
    pub importance: f64,
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

/// Engine statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stats {
    pub active_memories: i64,
    pub consolidated_memories: i64,
    pub tombstoned_memories: i64,
    pub edges: i64,
    pub entities: i64,
    pub operations: i64,
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
