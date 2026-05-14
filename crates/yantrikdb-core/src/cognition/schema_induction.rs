//! CK-5.2 — Schema Induction Engine.
//!
//! Observes concrete (state → action → outcome) sequences and extracts
//! generalizable decision schemas. Uses **anti-unification** (least general
//! generalization) to find the most specific rule that covers all positive
//! examples. Start specific, widen only with evidence.
//!
//! # Design principles
//! - Pure functions only — no DB access (engine layer handles persistence)
//! - Conservative generalization — start specific, widen with evidence
//! - Bidirectional refinement — success widens, failure narrows
//! - Composable — schemas can merge when overlap detected
//! - Measurable — confidence = support / (support + contradiction)

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::state::{CognitiveEdgeKind, NodeId, NodeKind};

// ══════════════════════════════════════════════════════════════════════════════
// § 1  Core Types
// ══════════════════════════════════════════════════════════════════════════════

/// Unique identifier for an induced schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SchemaId(pub u64);

/// Direction of an edge in a precondition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Direction {
    /// Edge from the focal node outward.
    Outgoing,
    /// Edge pointing into the focal node.
    Incoming,
    /// Either direction.
    Any,
}

/// An abstract precondition — not tied to specific entities.
///
/// These are the generalized conditions under which a schema applies.
/// Extracted from concrete episodes via anti-unification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SchemaCondition {
    /// A node of this kind must be present in context.
    NodeKindPresent(NodeKind),

    /// An edge of this kind must exist in the specified direction.
    EdgeExists {
        kind: CognitiveEdgeKind,
        direction: Direction,
    },

    /// A numeric attribute must be within a range.
    AttributeInRange { key: String, min: f64, max: f64 },

    /// The episode must be within a temporal window of the context.
    TemporalWindow { recency_ms: u64 },

    /// A belief in a category must exceed a confidence threshold.
    BeliefAboveThreshold {
        category: String,
        min_confidence: f64,
    },
}

impl SchemaCondition {
    /// Check if two conditions are "compatible" — same variant and overlapping.
    pub fn is_compatible(&self, other: &SchemaCondition) -> bool {
        match (self, other) {
            (SchemaCondition::NodeKindPresent(a), SchemaCondition::NodeKindPresent(b)) => a == b,
            (
                SchemaCondition::EdgeExists {
                    kind: ka,
                    direction: da,
                },
                SchemaCondition::EdgeExists {
                    kind: kb,
                    direction: db,
                },
            ) => ka == kb && (da == db || *da == Direction::Any || *db == Direction::Any),
            (
                SchemaCondition::AttributeInRange { key: ka, .. },
                SchemaCondition::AttributeInRange { key: kb, .. },
            ) => ka == kb,
            (SchemaCondition::TemporalWindow { .. }, SchemaCondition::TemporalWindow { .. }) => {
                true
            }
            (
                SchemaCondition::BeliefAboveThreshold { category: ca, .. },
                SchemaCondition::BeliefAboveThreshold { category: cb, .. },
            ) => ca == cb,
            _ => false,
        }
    }

    /// Generalize two compatible conditions into their least general generalization.
    ///
    /// Widens ranges, relaxes thresholds.
    pub fn generalize_with(&self, other: &SchemaCondition) -> Option<SchemaCondition> {
        match (self, other) {
            (SchemaCondition::NodeKindPresent(a), SchemaCondition::NodeKindPresent(b)) => {
                if a == b {
                    Some(SchemaCondition::NodeKindPresent(*a))
                } else {
                    None
                }
            }
            (
                SchemaCondition::EdgeExists {
                    kind: ka,
                    direction: da,
                },
                SchemaCondition::EdgeExists {
                    kind: kb,
                    direction: db,
                },
            ) => {
                if ka == kb {
                    let dir = if da == db { *da } else { Direction::Any };
                    Some(SchemaCondition::EdgeExists {
                        kind: *ka,
                        direction: dir,
                    })
                } else {
                    None
                }
            }
            (
                SchemaCondition::AttributeInRange {
                    key: ka,
                    min: min_a,
                    max: max_a,
                },
                SchemaCondition::AttributeInRange {
                    key: kb,
                    min: min_b,
                    max: max_b,
                },
            ) => {
                if ka == kb {
                    Some(SchemaCondition::AttributeInRange {
                        key: ka.clone(),
                        min: min_a.min(*min_b),
                        max: max_a.max(*max_b),
                    })
                } else {
                    None
                }
            }
            (
                SchemaCondition::TemporalWindow { recency_ms: a },
                SchemaCondition::TemporalWindow { recency_ms: b },
            ) => Some(SchemaCondition::TemporalWindow {
                recency_ms: (*a).max(*b), // widen the window
            }),
            (
                SchemaCondition::BeliefAboveThreshold {
                    category: ca,
                    min_confidence: ma,
                },
                SchemaCondition::BeliefAboveThreshold {
                    category: cb,
                    min_confidence: mb,
                },
            ) => {
                if ca == cb {
                    Some(SchemaCondition::BeliefAboveThreshold {
                        category: ca.clone(),
                        min_confidence: ma.min(*mb), // relax threshold
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 2  Action Template
// ══════════════════════════════════════════════════════════════════════════════

/// A parameterized action — generalized from concrete action instances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionTemplate {
    /// Action type identifier (e.g., "send_reminder", "suggest_break").
    pub base_action: String,
    /// Abstract parameter slots that get filled at execution time.
    pub parameter_slots: Vec<ParameterSlot>,
    /// Constraints on parameter values.
    pub constraints: Vec<TemplateConstraint>,
}

/// An abstract parameter that gets bound at execution time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSlot {
    /// Parameter name (e.g., "target_person", "delay_minutes").
    pub name: String,
    /// Expected type of the parameter.
    pub param_type: ParamType,
    /// Whether this parameter is required.
    pub required: bool,
}

/// Types of parameters in an action template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParamType {
    /// A reference to a cognitive node.
    NodeRef,
    /// A numeric value.
    Numeric,
    /// A text string.
    Text,
    /// A duration in milliseconds.
    Duration,
    /// A boolean flag.
    Boolean,
}

/// Constraints on parameter values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateConstraint {
    /// Which parameter this constrains.
    pub param_name: String,
    /// The constraint expression.
    pub constraint: ConstraintExpr,
}

/// Constraint expressions for template parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintExpr {
    /// Value must be in a numeric range.
    Range { min: f64, max: f64 },
    /// Value must be one of these options.
    OneOf(Vec<String>),
    /// Node must be of this kind.
    NodeKindIs(NodeKind),
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  Expected Outcome
// ══════════════════════════════════════════════════════════════════════════════

/// Statistical outcome from applying a schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedOutcome {
    /// What typically happens.
    pub description: String,
    /// How often this outcome occurs [0.0, 1.0].
    pub probability: f64,
    /// Positive or negative valence [-1.0, 1.0].
    pub valence: f64,
    /// How many times this outcome has been observed.
    pub observation_count: u32,
}

impl ExpectedOutcome {
    /// Update outcome statistics with a new observation.
    pub fn observe(&mut self, matched: bool) {
        self.observation_count += 1;
        if matched {
            // Move probability toward 1.0.
            self.probability += (1.0 - self.probability) * 0.1;
        } else {
            // Move probability toward 0.0.
            self.probability -= self.probability * 0.1;
        }
        self.probability = self.probability.clamp(0.0, 1.0);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  Induced Schema
// ══════════════════════════════════════════════════════════════════════════════

/// An abstract decision rule extracted from observations.
///
/// Lifecycle: Hypothesis → Emerging → Established → (Revised or Deprecated)
/// as evidence accumulates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InducedSchema {
    /// Unique identifier.
    pub id: SchemaId,
    /// Human-readable label.
    pub name: String,
    /// When this schema applies.
    pub preconditions: Vec<SchemaCondition>,
    /// What to do (parameterized action).
    pub action_template: ActionTemplate,
    /// What usually happens when this schema is applied.
    pub expected_outcomes: Vec<ExpectedOutcome>,
    /// Concrete episodes that support this schema.
    pub supporting_episodes: Vec<NodeId>,
    /// Episodes where the schema was applied but failed.
    pub contradicting_episodes: Vec<NodeId>,
    /// Confidence = support / (support + contradiction).
    pub confidence: f64,
    /// How many distinct contexts this applies to [0.0, 1.0].
    /// Higher = more general.
    pub generality: f64,
    /// When this schema was first created (unix ms).
    pub created_at: u64,
    /// When this schema was last refined (unix ms).
    pub last_refined_at: u64,
    /// How many times the schema has been refined.
    pub refinement_count: u32,
}

impl InducedSchema {
    /// Recompute confidence from supporting/contradicting counts.
    pub fn recompute_confidence(&mut self) {
        let support = self.supporting_episodes.len() as f64;
        let contra = self.contradicting_episodes.len() as f64;
        let total = support + contra;
        self.confidence = if total > 0.0 { support / total } else { 0.5 };
    }

    /// Whether this schema has enough evidence to be considered reliable.
    pub fn is_reliable(&self) -> bool {
        self.confidence >= 0.7 && self.supporting_episodes.len() >= 3
    }

    /// Total evidence count.
    pub fn evidence_count(&self) -> usize {
        self.supporting_episodes.len() + self.contradicting_episodes.len()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 5  Episode Data (Input)
// ══════════════════════════════════════════════════════════════════════════════

/// A concrete (state, action, outcome) triple observed from the system.
/// This is the raw input that drives schema induction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeData {
    /// The episode node id.
    pub episode_id: NodeId,
    /// Conditions that were true at the time of the episode.
    pub conditions: Vec<SchemaCondition>,
    /// The action that was taken.
    pub action_type: String,
    /// Whether the outcome was positive.
    pub outcome_positive: bool,
    /// Outcome description.
    pub outcome_description: String,
    /// Outcome valence [-1.0, 1.0].
    pub outcome_valence: f64,
    /// When this episode occurred (unix ms).
    pub timestamp_ms: u64,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 6  Context Snapshot (for matching)
// ══════════════════════════════════════════════════════════════════════════════

/// A snapshot of the current cognitive context for schema matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSnapshot {
    /// Node kinds present in the current working set.
    pub node_kinds_present: Vec<NodeKind>,
    /// Edge kinds present in the current context.
    pub edge_kinds_present: Vec<(CognitiveEdgeKind, Direction)>,
    /// Attribute values for range checks.
    pub attributes: HashMap<String, f64>,
    /// Active belief categories with their confidence.
    pub belief_confidences: HashMap<String, f64>,
    /// Current timestamp (unix ms).
    pub now_ms: u64,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 7  Schema Store
// ══════════════════════════════════════════════════════════════════════════════

/// Collection of induced schemas with indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaStore {
    /// All schemas.
    pub schemas: Vec<InducedSchema>,
    /// Index: node kind → schema indices whose preconditions require that kind.
    pub precondition_index: HashMap<NodeKind, Vec<usize>>,
    /// Next schema id.
    next_id: u64,
}

impl Default for SchemaStore {
    fn default() -> Self {
        Self {
            schemas: Vec::new(),
            precondition_index: HashMap::new(),
            next_id: 1,
        }
    }
}

impl SchemaStore {
    /// Number of schemas.
    pub fn len(&self) -> usize {
        self.schemas.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.schemas.is_empty()
    }

    /// Allocate a new schema id.
    pub fn alloc_id(&mut self) -> SchemaId {
        let id = SchemaId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Add a schema to the store.
    pub fn insert(&mut self, schema: InducedSchema) {
        let idx = self.schemas.len();
        // Index by node kinds in preconditions.
        for cond in &schema.preconditions {
            if let SchemaCondition::NodeKindPresent(kind) = cond {
                self.precondition_index.entry(*kind).or_default().push(idx);
            }
        }
        self.schemas.push(schema);
    }

    /// Find a schema by id.
    pub fn find(&self, id: SchemaId) -> Option<&InducedSchema> {
        self.schemas.iter().find(|s| s.id == id)
    }

    /// Find a mutable schema by id.
    pub fn find_mut(&mut self, id: SchemaId) -> Option<&mut InducedSchema> {
        self.schemas.iter_mut().find(|s| s.id == id)
    }

    /// Find schemas whose action_template.base_action matches.
    pub fn find_by_action(&self, action_type: &str) -> Vec<&InducedSchema> {
        self.schemas
            .iter()
            .filter(|s| s.action_template.base_action == action_type)
            .collect()
    }

    /// Rebuild the precondition index from scratch.
    pub fn rebuild_index(&mut self) {
        self.precondition_index.clear();
        for (idx, schema) in self.schemas.iter().enumerate() {
            for cond in &schema.preconditions {
                if let SchemaCondition::NodeKindPresent(kind) = cond {
                    self.precondition_index.entry(*kind).or_default().push(idx);
                }
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 8  Core Algorithms
// ══════════════════════════════════════════════════════════════════════════════

/// Process a new episode. Match against existing schemas:
/// - If a matching schema exists → refine it.
/// - If no match → check if enough similar episodes exist to induce a new one.
pub fn observe_episode(episode: &EpisodeData, store: &mut SchemaStore) {
    // Find schemas with the same action type.
    let matching_ids: Vec<SchemaId> = store
        .find_by_action(&episode.action_type)
        .iter()
        .map(|s| s.id)
        .collect();

    let mut refined_any = false;

    for id in matching_ids {
        if let Some(schema) = store.find_mut(id) {
            // Check if the episode's conditions are compatible with the schema.
            let match_score = condition_match_score(&schema.preconditions, &episode.conditions);
            if match_score > 0.5 {
                refine_schema(schema, episode);
                refined_any = true;
            }
        }
    }

    if !refined_any {
        // Collect episodes with the same action type for potential induction.
        // We use the contradicting/supporting episodes from existing schemas
        // plus the current episode to see if we have enough for a new schema.
        // For now, seed a new hypothesis schema from this single episode.
        let now = crate::state::now_ms();
        let id = store.alloc_id();
        let mut schema = InducedSchema {
            id,
            name: format!("schema_{}", id.0),
            preconditions: episode.conditions.clone(),
            action_template: ActionTemplate {
                base_action: episode.action_type.clone(),
                parameter_slots: Vec::new(),
                constraints: Vec::new(),
            },
            expected_outcomes: vec![ExpectedOutcome {
                description: episode.outcome_description.clone(),
                probability: if episode.outcome_positive { 0.8 } else { 0.2 },
                valence: episode.outcome_valence,
                observation_count: 1,
            }],
            supporting_episodes: if episode.outcome_positive {
                vec![episode.episode_id]
            } else {
                Vec::new()
            },
            contradicting_episodes: if !episode.outcome_positive {
                vec![episode.episode_id]
            } else {
                Vec::new()
            },
            confidence: if episode.outcome_positive { 0.6 } else { 0.3 },
            generality: 0.1, // Very specific initially.
            created_at: now,
            last_refined_at: now,
            refinement_count: 0,
        };
        schema.recompute_confidence();
        store.insert(schema);
    }
}

/// Compute how well an episode's conditions match a schema's preconditions.
/// Returns a score in [0.0, 1.0].
fn condition_match_score(
    schema_conds: &[SchemaCondition],
    episode_conds: &[SchemaCondition],
) -> f64 {
    if schema_conds.is_empty() {
        return 1.0; // No preconditions → matches everything.
    }

    let mut matched = 0;
    for sc in schema_conds {
        if episode_conds.iter().any(|ec| sc.is_compatible(ec)) {
            matched += 1;
        }
    }

    matched as f64 / schema_conds.len() as f64
}

/// Refine a schema with a new episode.
///
/// If the episode was successful: widen preconditions to cover new context.
/// If the episode failed: narrow preconditions to exclude failure context.
pub fn refine_schema(schema: &mut InducedSchema, episode: &EpisodeData) {
    let now = crate::state::now_ms();
    schema.last_refined_at = now;
    schema.refinement_count += 1;

    if episode.outcome_positive {
        schema.supporting_episodes.push(episode.episode_id);

        // Widen preconditions via generalization.
        schema.preconditions =
            generalize_preconditions(&[schema.preconditions.clone(), episode.conditions.clone()]);

        // Update generality: more diverse contexts = more general.
        let unique_contexts = schema.supporting_episodes.len();
        schema.generality = (unique_contexts as f64 / 20.0).min(1.0);
    } else {
        schema.contradicting_episodes.push(episode.episode_id);

        // Specialize: add discriminating conditions from failure.
        specialize_on_failure(schema, episode);
    }

    // Update outcome statistics.
    if let Some(outcome) = schema.expected_outcomes.first_mut() {
        outcome.observe(episode.outcome_positive);
    }

    schema.recompute_confidence();
}

/// Given 3+ similar episodes, extract the common abstract structure as a schema.
///
/// Uses anti-unification: find the most specific generalization that covers
/// all provided episodes.
pub fn induce_schema(episodes: &[EpisodeData], store: &mut SchemaStore) -> Option<InducedSchema> {
    if episodes.len() < 3 {
        return None;
    }

    // All episodes must share the same action type.
    let action_type = &episodes[0].action_type;
    if !episodes.iter().all(|e| e.action_type == *action_type) {
        return None;
    }

    // Collect all condition sets.
    let condition_sets: Vec<Vec<SchemaCondition>> =
        episodes.iter().map(|e| e.conditions.clone()).collect();

    // Anti-unify: find common conditions.
    let generalized = generalize_preconditions(&condition_sets);

    // Compute outcome statistics.
    let positive_count = episodes.iter().filter(|e| e.outcome_positive).count();
    let total = episodes.len();
    let avg_valence = episodes.iter().map(|e| e.outcome_valence).sum::<f64>() / total as f64;

    let now = crate::state::now_ms();
    let id = store.alloc_id();

    let schema = InducedSchema {
        id,
        name: format!("induced_{}", action_type),
        preconditions: generalized,
        action_template: ActionTemplate {
            base_action: action_type.clone(),
            parameter_slots: Vec::new(),
            constraints: Vec::new(),
        },
        expected_outcomes: vec![ExpectedOutcome {
            description: format!("Typical outcome of {}", action_type),
            probability: positive_count as f64 / total as f64,
            valence: avg_valence,
            observation_count: total as u32,
        }],
        supporting_episodes: episodes
            .iter()
            .filter(|e| e.outcome_positive)
            .map(|e| e.episode_id)
            .collect(),
        contradicting_episodes: episodes
            .iter()
            .filter(|e| !e.outcome_positive)
            .map(|e| e.episode_id)
            .collect(),
        confidence: positive_count as f64 / total as f64,
        generality: (episodes.len() as f64 / 20.0).min(1.0),
        created_at: now,
        last_refined_at: now,
        refinement_count: 0,
    };

    Some(schema)
}

// ══════════════════════════════════════════════════════════════════════════════
// § 9  Precondition Generalization (Anti-Unification)
// ══════════════════════════════════════════════════════════════════════════════

/// Find the minimal set of conditions that covers all provided condition sets.
///
/// Anti-unification / least general generalization: for each condition that
/// appears in ALL sets (or most), include the generalized version.
pub fn generalize_preconditions(condition_sets: &[Vec<SchemaCondition>]) -> Vec<SchemaCondition> {
    if condition_sets.is_empty() {
        return Vec::new();
    }
    if condition_sets.len() == 1 {
        return condition_sets[0].clone();
    }

    // Start with the first set and progressively generalize.
    let mut result = condition_sets[0].clone();

    for other_set in &condition_sets[1..] {
        let mut generalized = Vec::new();

        for cond in &result {
            // Find a compatible condition in the other set.
            if let Some(partner) = other_set.iter().find(|oc| cond.is_compatible(oc)) {
                // Generalize the two into one.
                if let Some(gen) = cond.generalize_with(partner) {
                    generalized.push(gen);
                }
            }
            // If no compatible partner, this condition is dropped (too specific).
        }

        result = generalized;
    }

    result
}

/// Add discriminating conditions that separate success from failure.
///
/// When a schema fails, look at what conditions the failure episode has
/// that successful episodes don't, and add the NEGATION as a precondition.
fn specialize_on_failure(schema: &mut InducedSchema, failure: &EpisodeData) {
    // Find conditions in the failure that aren't in the schema's preconditions.
    for fc in &failure.conditions {
        let already_covered = schema.preconditions.iter().any(|sc| sc.is_compatible(fc));
        if !already_covered {
            // This condition was present during failure but not in our schema.
            // Add a more restrictive version to the preconditions.
            match fc {
                SchemaCondition::AttributeInRange { key, min, max } => {
                    // Exclude this range by narrowing.
                    schema
                        .preconditions
                        .push(SchemaCondition::AttributeInRange {
                            key: key.clone(),
                            min: *max, // Set min above the failure range.
                            max: f64::INFINITY,
                        });
                }
                SchemaCondition::BeliefAboveThreshold {
                    category,
                    min_confidence,
                } => {
                    // Require higher confidence than what failed.
                    schema
                        .preconditions
                        .push(SchemaCondition::BeliefAboveThreshold {
                            category: category.clone(),
                            min_confidence: min_confidence + 0.1,
                        });
                }
                _ => {
                    // For other types, we can't easily negate, so we note the
                    // failure but don't add a new condition.
                }
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 10  Schema Matching
// ══════════════════════════════════════════════════════════════════════════════

/// Find applicable schemas for the current context, ranked by
/// (confidence × generality × recency).
pub fn match_schemas(context: &ContextSnapshot, store: &SchemaStore) -> Vec<(SchemaId, f64)> {
    let mut matches: Vec<(SchemaId, f64)> = Vec::new();

    for schema in &store.schemas {
        let satisfaction = evaluate_satisfaction(&schema.preconditions, context);
        if satisfaction >= 0.5 {
            // Recency factor: schemas refined recently get a small boost.
            let age_days =
                (context.now_ms.saturating_sub(schema.last_refined_at)) as f64 / (86_400_000.0);
            let recency = (-age_days / 30.0).exp(); // 30-day half-life

            let score =
                satisfaction * schema.confidence * (0.5 + 0.3 * schema.generality + 0.2 * recency);
            matches.push((schema.id, score));
        }
    }

    matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    matches
}

/// Evaluate how well a context satisfies a set of preconditions.
/// Returns a score in [0.0, 1.0].
fn evaluate_satisfaction(preconditions: &[SchemaCondition], context: &ContextSnapshot) -> f64 {
    if preconditions.is_empty() {
        return 1.0;
    }

    let mut satisfied = 0;
    for cond in preconditions {
        let met = match cond {
            SchemaCondition::NodeKindPresent(kind) => context.node_kinds_present.contains(kind),
            SchemaCondition::EdgeExists { kind, direction } => context
                .edge_kinds_present
                .iter()
                .any(|(ek, ed)| ek == kind && (*direction == Direction::Any || ed == direction)),
            SchemaCondition::AttributeInRange { key, min, max } => context
                .attributes
                .get(key)
                .map_or(false, |v| v >= min && v <= max),
            SchemaCondition::TemporalWindow { .. } => {
                true // Temporal conditions are always "met" for matching.
            }
            SchemaCondition::BeliefAboveThreshold {
                category,
                min_confidence,
            } => context
                .belief_confidences
                .get(category)
                .map_or(false, |c| c >= min_confidence),
        };
        if met {
            satisfied += 1;
        }
    }

    satisfied as f64 / preconditions.len() as f64
}

// ══════════════════════════════════════════════════════════════════════════════
// § 11  Schema Merging
// ══════════════════════════════════════════════════════════════════════════════

/// If two schemas have overlapping preconditions and the same action template,
/// merge them into one more general schema.
pub fn merge_schemas(a: &InducedSchema, b: &InducedSchema) -> Option<InducedSchema> {
    // Must have the same base action.
    if a.action_template.base_action != b.action_template.base_action {
        return None;
    }

    // Preconditions must overlap significantly.
    let overlap = condition_match_score(&a.preconditions, &b.preconditions);
    if overlap < 0.6 {
        return None;
    }

    let now = crate::state::now_ms();

    // Generalize preconditions.
    let merged_preconditions =
        generalize_preconditions(&[a.preconditions.clone(), b.preconditions.clone()]);

    // Merge outcomes.
    let mut outcomes = a.expected_outcomes.clone();
    for bo in &b.expected_outcomes {
        if let Some(existing) = outcomes
            .iter_mut()
            .find(|o| o.description == bo.description)
        {
            existing.observation_count += bo.observation_count;
            existing.probability = (existing.probability + bo.probability) / 2.0;
            existing.valence = (existing.valence + bo.valence) / 2.0;
        } else {
            outcomes.push(bo.clone());
        }
    }

    // Combine episode references.
    let mut supporting = a.supporting_episodes.clone();
    supporting.extend(&b.supporting_episodes);
    let mut contradicting = a.contradicting_episodes.clone();
    contradicting.extend(&b.contradicting_episodes);

    let mut merged = InducedSchema {
        id: a.id, // Keep the older id.
        name: format!("merged_{}_{}", a.id.0, b.id.0),
        preconditions: merged_preconditions,
        action_template: a.action_template.clone(),
        expected_outcomes: outcomes,
        supporting_episodes: supporting,
        contradicting_episodes: contradicting,
        confidence: 0.0, // Recomputed below.
        generality: a.generality.max(b.generality),
        created_at: a.created_at.min(b.created_at),
        last_refined_at: now,
        refinement_count: a.refinement_count + b.refinement_count + 1,
    };
    merged.recompute_confidence();

    Some(merged)
}

// ══════════════════════════════════════════════════════════════════════════════
// § 12  Schema Maintenance
// ══════════════════════════════════════════════════════════════════════════════

/// Report from schema maintenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaMaintenanceReport {
    /// Schemas pruned for low confidence.
    pub pruned_low_confidence: usize,
    /// Schemas merged.
    pub merged: usize,
    /// Schemas decayed (unused).
    pub decayed: usize,
    /// Remaining schemas.
    pub remaining: usize,
}

/// Prune low-confidence schemas, merge near-duplicates, decay unused ones.
pub fn schema_maintenance(
    store: &mut SchemaStore,
    now_ms: u64,
    max_age_ms: u64,
) -> SchemaMaintenanceReport {
    let mut pruned_low_confidence = 0;
    let mut decayed = 0;

    // Prune schemas with very low confidence and enough evidence.
    store.schemas.retain(|s| {
        let has_enough_evidence = s.evidence_count() >= 3;
        let is_low_confidence = s.confidence < 0.25;
        let is_old_unused =
            now_ms.saturating_sub(s.last_refined_at) > max_age_ms && s.confidence < 0.5;

        if has_enough_evidence && is_low_confidence {
            pruned_low_confidence += 1;
            return false;
        }
        if is_old_unused {
            decayed += 1;
            return false;
        }
        true
    });

    // Try to merge near-duplicate schemas.
    let mut merged_count = 0;
    let mut i = 0;
    while i < store.schemas.len() {
        let mut j = i + 1;
        while j < store.schemas.len() {
            let can_merge = store.schemas[i].action_template.base_action
                == store.schemas[j].action_template.base_action;
            if can_merge {
                if let Some(merged) = merge_schemas(&store.schemas[i], &store.schemas[j]) {
                    store.schemas[i] = merged;
                    store.schemas.remove(j);
                    merged_count += 1;
                    continue; // Don't increment j.
                }
            }
            j += 1;
        }
        i += 1;
    }

    let remaining = store.schemas.len();
    store.rebuild_index();

    SchemaMaintenanceReport {
        pruned_low_confidence,
        merged: merged_count,
        decayed,
        remaining,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 13  Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_episode(
        seq: u32,
        action: &str,
        positive: bool,
        conditions: Vec<SchemaCondition>,
    ) -> EpisodeData {
        EpisodeData {
            episode_id: NodeId::new(NodeKind::Episode, seq),
            conditions,
            action_type: action.to_string(),
            outcome_positive: positive,
            outcome_description: if positive {
                "success".to_string()
            } else {
                "failure".to_string()
            },
            outcome_valence: if positive { 0.8 } else { -0.5 },
            timestamp_ms: crate::state::now_ms(),
        }
    }

    fn basic_conditions() -> Vec<SchemaCondition> {
        vec![
            SchemaCondition::NodeKindPresent(NodeKind::Goal),
            SchemaCondition::AttributeInRange {
                key: "urgency".to_string(),
                min: 0.5,
                max: 1.0,
            },
        ]
    }

    fn variant_conditions() -> Vec<SchemaCondition> {
        vec![
            SchemaCondition::NodeKindPresent(NodeKind::Goal),
            SchemaCondition::AttributeInRange {
                key: "urgency".to_string(),
                min: 0.3,
                max: 0.9,
            },
        ]
    }

    #[test]
    fn test_single_episode_observation() {
        let mut store = SchemaStore::default();
        let ep = make_episode(1, "send_reminder", true, basic_conditions());
        observe_episode(&ep, &mut store);

        assert_eq!(store.len(), 1);
        let schema = &store.schemas[0];
        assert_eq!(schema.action_template.base_action, "send_reminder");
        assert!(!schema.preconditions.is_empty());
        assert_eq!(schema.supporting_episodes.len(), 1);
    }

    #[test]
    fn test_refinement_on_success() {
        let mut store = SchemaStore::default();
        let ep1 = make_episode(1, "send_reminder", true, basic_conditions());
        let ep2 = make_episode(2, "send_reminder", true, variant_conditions());

        observe_episode(&ep1, &mut store);
        observe_episode(&ep2, &mut store);

        let schema = &store.schemas[0];
        assert_eq!(schema.supporting_episodes.len(), 2);
        assert!(schema.confidence > 0.5);
        assert!(schema.refinement_count >= 1);
    }

    #[test]
    fn test_refinement_on_failure() {
        let mut store = SchemaStore::default();
        let ep1 = make_episode(1, "send_reminder", true, basic_conditions());
        observe_episode(&ep1, &mut store);

        // Failure with a different condition.
        let failure_conditions = vec![
            SchemaCondition::NodeKindPresent(NodeKind::Goal),
            SchemaCondition::AttributeInRange {
                key: "urgency".to_string(),
                min: 0.1,
                max: 0.3,
            },
            SchemaCondition::BeliefAboveThreshold {
                category: "risk".to_string(),
                min_confidence: 0.2,
            },
        ];
        let ep2 = make_episode(2, "send_reminder", false, failure_conditions);
        observe_episode(&ep2, &mut store);

        let schema = &store.schemas[0];
        assert_eq!(schema.contradicting_episodes.len(), 1);
        assert!(schema.confidence < 1.0);
    }

    #[test]
    fn test_three_episode_induction() {
        let mut store = SchemaStore::default();
        let episodes = vec![
            make_episode(1, "suggest_break", true, basic_conditions()),
            make_episode(2, "suggest_break", true, variant_conditions()),
            make_episode(
                3,
                "suggest_break",
                true,
                vec![
                    SchemaCondition::NodeKindPresent(NodeKind::Goal),
                    SchemaCondition::AttributeInRange {
                        key: "urgency".to_string(),
                        min: 0.4,
                        max: 0.8,
                    },
                ],
            ),
        ];

        let schema = induce_schema(&episodes, &mut store);
        assert!(schema.is_some(), "Should induce a schema from 3 episodes");

        let s = schema.unwrap();
        assert_eq!(s.supporting_episodes.len(), 3);
        assert!(s.confidence > 0.9);
        // Preconditions should be generalized (wider range).
        let has_urgency = s.preconditions.iter().any(|c| {
            matches!(c, SchemaCondition::AttributeInRange { key, min, max }
                if key == "urgency" && *min <= 0.3 && *max >= 0.9)
        });
        assert!(has_urgency, "Should have generalized urgency range");
    }

    #[test]
    fn test_induction_requires_three_episodes() {
        let mut store = SchemaStore::default();
        let episodes = vec![
            make_episode(1, "act", true, basic_conditions()),
            make_episode(2, "act", true, basic_conditions()),
        ];
        assert!(induce_schema(&episodes, &mut store).is_none());
    }

    #[test]
    fn test_induction_requires_same_action() {
        let mut store = SchemaStore::default();
        let episodes = vec![
            make_episode(1, "act_a", true, basic_conditions()),
            make_episode(2, "act_b", true, basic_conditions()),
            make_episode(3, "act_a", true, basic_conditions()),
        ];
        assert!(induce_schema(&episodes, &mut store).is_none());
    }

    #[test]
    fn test_precondition_generalization() {
        let sets = vec![
            vec![SchemaCondition::AttributeInRange {
                key: "urgency".to_string(),
                min: 0.5,
                max: 0.8,
            }],
            vec![SchemaCondition::AttributeInRange {
                key: "urgency".to_string(),
                min: 0.3,
                max: 0.9,
            }],
        ];
        let result = generalize_preconditions(&sets);
        assert_eq!(result.len(), 1);
        if let SchemaCondition::AttributeInRange { min, max, .. } = &result[0] {
            assert!(*min <= 0.3, "min should widen: {min}");
            assert!(*max >= 0.9, "max should widen: {max}");
        } else {
            panic!("Expected AttributeInRange");
        }
    }

    #[test]
    fn test_precondition_generalization_drops_unshared() {
        let sets = vec![
            vec![
                SchemaCondition::NodeKindPresent(NodeKind::Goal),
                SchemaCondition::NodeKindPresent(NodeKind::Task),
            ],
            vec![SchemaCondition::NodeKindPresent(NodeKind::Goal)],
        ];
        let result = generalize_preconditions(&sets);
        // Task condition should be dropped (not in second set).
        assert_eq!(result.len(), 1);
        assert!(matches!(
            result[0],
            SchemaCondition::NodeKindPresent(NodeKind::Goal)
        ));
    }

    #[test]
    fn test_schema_matching() {
        let mut store = SchemaStore::default();
        let ep = make_episode(1, "remind", true, basic_conditions());
        observe_episode(&ep, &mut store);

        let context = ContextSnapshot {
            node_kinds_present: vec![NodeKind::Goal, NodeKind::Task],
            edge_kinds_present: vec![],
            attributes: {
                let mut m = HashMap::new();
                m.insert("urgency".to_string(), 0.7);
                m
            },
            belief_confidences: HashMap::new(),
            now_ms: crate::state::now_ms(),
        };

        let matches = match_schemas(&context, &store);
        assert!(!matches.is_empty(), "Should match the schema");
        assert!(matches[0].1 > 0.0);
    }

    #[test]
    fn test_schema_matching_no_match() {
        let mut store = SchemaStore::default();
        let ep = make_episode(1, "remind", true, basic_conditions());
        observe_episode(&ep, &mut store);

        // Context that doesn't satisfy preconditions.
        let context = ContextSnapshot {
            node_kinds_present: vec![NodeKind::Entity],
            edge_kinds_present: vec![],
            attributes: {
                let mut m = HashMap::new();
                m.insert("urgency".to_string(), 0.1); // Below threshold.
                m
            },
            belief_confidences: HashMap::new(),
            now_ms: crate::state::now_ms(),
        };

        let matches = match_schemas(&context, &store);
        assert!(matches.is_empty(), "Should not match with wrong context");
    }

    #[test]
    fn test_schema_merging() {
        let now = crate::state::now_ms();
        let a = InducedSchema {
            id: SchemaId(1),
            name: "schema_1".to_string(),
            preconditions: basic_conditions(),
            action_template: ActionTemplate {
                base_action: "remind".to_string(),
                parameter_slots: vec![],
                constraints: vec![],
            },
            expected_outcomes: vec![ExpectedOutcome {
                description: "reminded".to_string(),
                probability: 0.8,
                valence: 0.5,
                observation_count: 5,
            }],
            supporting_episodes: vec![NodeId::new(NodeKind::Episode, 1)],
            contradicting_episodes: vec![],
            confidence: 0.9,
            generality: 0.3,
            created_at: now,
            last_refined_at: now,
            refinement_count: 2,
        };

        let b = InducedSchema {
            id: SchemaId(2),
            name: "schema_2".to_string(),
            preconditions: variant_conditions(),
            action_template: ActionTemplate {
                base_action: "remind".to_string(),
                parameter_slots: vec![],
                constraints: vec![],
            },
            expected_outcomes: vec![ExpectedOutcome {
                description: "reminded".to_string(),
                probability: 0.7,
                valence: 0.4,
                observation_count: 3,
            }],
            supporting_episodes: vec![NodeId::new(NodeKind::Episode, 2)],
            contradicting_episodes: vec![],
            confidence: 0.8,
            generality: 0.2,
            created_at: now,
            last_refined_at: now,
            refinement_count: 1,
        };

        let merged = merge_schemas(&a, &b);
        assert!(merged.is_some(), "Should merge compatible schemas");
        let m = merged.unwrap();
        assert_eq!(m.supporting_episodes.len(), 2);
        assert_eq!(m.expected_outcomes[0].observation_count, 8);
    }

    #[test]
    fn test_schema_merging_different_actions() {
        let now = crate::state::now_ms();
        let a = InducedSchema {
            id: SchemaId(1),
            name: "a".to_string(),
            preconditions: basic_conditions(),
            action_template: ActionTemplate {
                base_action: "action_a".to_string(),
                parameter_slots: vec![],
                constraints: vec![],
            },
            expected_outcomes: vec![],
            supporting_episodes: vec![],
            contradicting_episodes: vec![],
            confidence: 0.8,
            generality: 0.5,
            created_at: now,
            last_refined_at: now,
            refinement_count: 0,
        };
        let b = InducedSchema {
            id: SchemaId(2),
            name: "b".to_string(),
            preconditions: basic_conditions(),
            action_template: ActionTemplate {
                base_action: "action_b".to_string(),
                parameter_slots: vec![],
                constraints: vec![],
            },
            expected_outcomes: vec![],
            supporting_episodes: vec![],
            contradicting_episodes: vec![],
            confidence: 0.8,
            generality: 0.5,
            created_at: now,
            last_refined_at: now,
            refinement_count: 0,
        };
        assert!(
            merge_schemas(&a, &b).is_none(),
            "Different actions should not merge"
        );
    }

    #[test]
    fn test_schema_maintenance() {
        let mut store = SchemaStore::default();
        let now = crate::state::now_ms();
        let old = now - 100_000_000;

        // Good schema.
        let good_id = store.alloc_id();
        store.insert(InducedSchema {
            id: good_id,
            name: "good".to_string(),
            preconditions: basic_conditions(),
            action_template: ActionTemplate {
                base_action: "act".to_string(),
                parameter_slots: vec![],
                constraints: vec![],
            },
            expected_outcomes: vec![],
            supporting_episodes: vec![
                NodeId::new(NodeKind::Episode, 1),
                NodeId::new(NodeKind::Episode, 2),
                NodeId::new(NodeKind::Episode, 3),
            ],
            contradicting_episodes: vec![],
            confidence: 0.9,
            generality: 0.5,
            created_at: now,
            last_refined_at: now,
            refinement_count: 3,
        });

        // Low confidence schema with enough evidence.
        let weak_id = store.alloc_id();
        store.insert(InducedSchema {
            id: weak_id,
            name: "weak".to_string(),
            preconditions: basic_conditions(),
            action_template: ActionTemplate {
                base_action: "other".to_string(),
                parameter_slots: vec![],
                constraints: vec![],
            },
            expected_outcomes: vec![],
            supporting_episodes: vec![NodeId::new(NodeKind::Episode, 4)],
            contradicting_episodes: vec![
                NodeId::new(NodeKind::Episode, 5),
                NodeId::new(NodeKind::Episode, 6),
                NodeId::new(NodeKind::Episode, 7),
            ],
            confidence: 0.2,
            generality: 0.1,
            created_at: old,
            last_refined_at: old,
            refinement_count: 0,
        });

        let report = schema_maintenance(&mut store, now, 86_400_000);
        assert!(report.pruned_low_confidence > 0, "Should prune weak schema");
        assert_eq!(report.remaining, 1);
    }

    #[test]
    fn test_condition_compatibility() {
        let a = SchemaCondition::NodeKindPresent(NodeKind::Goal);
        let b = SchemaCondition::NodeKindPresent(NodeKind::Goal);
        let c = SchemaCondition::NodeKindPresent(NodeKind::Task);
        assert!(a.is_compatible(&b));
        assert!(!a.is_compatible(&c));
    }

    #[test]
    fn test_condition_generalization() {
        let a = SchemaCondition::AttributeInRange {
            key: "x".to_string(),
            min: 0.3,
            max: 0.7,
        };
        let b = SchemaCondition::AttributeInRange {
            key: "x".to_string(),
            min: 0.5,
            max: 0.9,
        };
        let gen = a.generalize_with(&b).unwrap();
        if let SchemaCondition::AttributeInRange { min, max, .. } = gen {
            assert_eq!(min, 0.3);
            assert_eq!(max, 0.9);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_expected_outcome_observe() {
        let mut outcome = ExpectedOutcome {
            description: "test".to_string(),
            probability: 0.5,
            valence: 0.0,
            observation_count: 0,
        };
        outcome.observe(true);
        assert!(outcome.probability > 0.5);
        assert_eq!(outcome.observation_count, 1);

        outcome.observe(false);
        assert_eq!(outcome.observation_count, 2);
    }

    #[test]
    fn test_schema_is_reliable() {
        let now = crate::state::now_ms();
        let mut schema = InducedSchema {
            id: SchemaId(1),
            name: "test".to_string(),
            preconditions: vec![],
            action_template: ActionTemplate {
                base_action: "act".to_string(),
                parameter_slots: vec![],
                constraints: vec![],
            },
            expected_outcomes: vec![],
            supporting_episodes: vec![
                NodeId::new(NodeKind::Episode, 1),
                NodeId::new(NodeKind::Episode, 2),
                NodeId::new(NodeKind::Episode, 3),
            ],
            contradicting_episodes: vec![],
            confidence: 0.0,
            generality: 0.5,
            created_at: now,
            last_refined_at: now,
            refinement_count: 0,
        };
        schema.recompute_confidence();
        assert!(schema.is_reliable());

        schema.contradicting_episodes = vec![
            NodeId::new(NodeKind::Episode, 4),
            NodeId::new(NodeKind::Episode, 5),
            NodeId::new(NodeKind::Episode, 6),
            NodeId::new(NodeKind::Episode, 7),
        ];
        schema.recompute_confidence();
        assert!(!schema.is_reliable());
    }

    #[test]
    fn test_store_find_by_action() {
        let mut store = SchemaStore::default();
        let ep1 = make_episode(1, "remind", true, basic_conditions());
        let ep2 = make_episode(2, "warn", true, basic_conditions());
        observe_episode(&ep1, &mut store);
        observe_episode(&ep2, &mut store);

        assert_eq!(store.find_by_action("remind").len(), 1);
        assert_eq!(store.find_by_action("warn").len(), 1);
        assert_eq!(store.find_by_action("other").len(), 0);
    }
}
