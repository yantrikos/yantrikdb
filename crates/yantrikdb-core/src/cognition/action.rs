//! CK-1.7: Action Schema Library + Candidate Generator
//!
//! Provides a built-in library of action schemas and a candidate generator
//! that maps intent hypotheses to concrete action candidates. The system
//! reasons about *what it can do* without calling an LLM.
//!
//! ## Built-in Action Library
//!
//! 30 action schemas across 8 ActionKind categories, each with:
//! - Typed preconditions referencing cognitive node kinds
//! - Expected effects with probability and utility
//! - Learned confidence thresholds and success rates
//!
//! ## Candidate Generation
//!
//! Given an intent hypothesis + cognitive graph context:
//! 1. **Match**: Find schemas whose preconditions are satisfiable
//! 2. **Bind**: Bind precondition node_refs to actual graph nodes
//! 3. **Score**: Preliminary relevance score (refined by CK-1.8 utility)
//! 4. **Rank**: Return sorted candidates with bindings

use serde::{Deserialize, Serialize};

use super::intent::ScoredIntent;
use super::state::*;

// ── Action Candidate ──

/// A concrete action candidate: a schema bound to specific graph context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionCandidate {
    /// The action schema this candidate is based on.
    pub schema_name: String,

    /// Action category.
    pub action_kind: ActionKind,

    /// Natural language description of *this specific* action instance.
    pub description: String,

    /// The intent hypothesis that triggered this candidate.
    pub source_intent: String,

    /// Precondition satisfaction: (description, satisfied, bound_node).
    pub precondition_bindings: Vec<PreconditionBinding>,

    /// How many required preconditions are satisfied.
    pub satisfied_required: usize,

    /// Total required preconditions.
    pub total_required: usize,

    /// How many soft preconditions are satisfied.
    pub satisfied_soft: usize,

    /// Preliminary relevance score [0.0, 1.0].
    /// Combines precondition satisfaction, schema success rate, and intent posterior.
    pub relevance_score: f64,

    /// NodeId of the schema (if persisted), or None for built-in schemas.
    pub schema_node: Option<NodeId>,
}

/// A precondition bound to a specific graph node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreconditionBinding {
    pub description: String,
    pub required: bool,
    pub satisfied: bool,
    pub bound_node: Option<NodeId>,
}

// ── Generation Result ──

/// Result of candidate generation for a set of intents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateGenerationResult {
    /// All generated candidates, sorted by relevance_score descending.
    pub candidates: Vec<ActionCandidate>,

    /// Total schemas evaluated.
    pub schemas_evaluated: usize,

    /// Total candidates before filtering.
    pub total_before_filter: usize,

    /// Duration in microseconds.
    pub duration_us: u64,
}

// ── Configuration ──

/// Configuration for candidate generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionConfig {
    /// Minimum fraction of required preconditions that must be satisfied.
    /// 1.0 = all required preconditions must be met (strict).
    /// 0.5 = at least half (lenient, for exploration).
    pub min_precondition_ratio: f64,

    /// Maximum candidates to return per intent.
    pub max_candidates_per_intent: usize,

    /// Maximum total candidates across all intents.
    pub max_total_candidates: usize,

    /// Whether to include the Abstain action (always available).
    pub include_abstain: bool,

    /// Minimum schema success rate to consider.
    pub min_success_rate: f64,
}

impl Default for ActionConfig {
    fn default() -> Self {
        Self {
            min_precondition_ratio: 0.8,
            max_candidates_per_intent: 5,
            max_total_candidates: 15,
            include_abstain: true,
            min_success_rate: 0.0, // allow untried schemas
        }
    }
}

// ── Built-in Action Library ──

/// A schema template used to build ActionSchemaPayload instances.
/// Compact representation for the built-in library.
struct SchemaTemplate {
    name: &'static str,
    description: &'static str,
    kind: ActionKind,
    /// Preconditions as (description, required, required_node_kind).
    preconditions: &'static [(&'static str, bool, Option<NodeKind>)],
    /// Effects as (description, probability, utility).
    effects: &'static [(&'static str, f64, f64)],
    /// Default confidence threshold.
    confidence_threshold: f64,
}

/// The built-in action schema library.
/// 30 schemas across 8 ActionKind categories.
const BUILTIN_SCHEMAS: &[SchemaTemplate] = &[
    // ── Communicate (4) ──
    SchemaTemplate {
        name: "send_reminder",
        description: "Remind the user about a pending task or goal",
        kind: ActionKind::Communicate,
        preconditions: &[
            (
                "User has an active goal or task",
                true,
                Some(NodeKind::Goal),
            ),
            ("Goal or task has urgency above threshold", true, None),
        ],
        effects: &[
            ("User is reminded and may take action", 0.85, 0.6),
            ("User may feel interrupted", 0.30, -0.2),
        ],
        confidence_threshold: 0.4,
    },
    SchemaTemplate {
        name: "proactive_greeting",
        description: "Greet the user based on detected routine start",
        kind: ActionKind::Communicate,
        preconditions: &[(
            "A routine is near its trigger time",
            true,
            Some(NodeKind::Routine),
        )],
        effects: &[("User feels acknowledged", 0.90, 0.3)],
        confidence_threshold: 0.3,
    },
    SchemaTemplate {
        name: "check_in",
        description: "Ask the user how a goal or need is progressing",
        kind: ActionKind::Communicate,
        preconditions: &[
            ("User has an active goal", true, Some(NodeKind::Goal)),
            ("Goal hasn't been updated recently", false, None),
        ],
        effects: &[
            ("User provides progress update", 0.70, 0.4),
            ("User feels monitored", 0.15, -0.15),
        ],
        confidence_threshold: 0.5,
    },
    SchemaTemplate {
        name: "share_encouragement",
        description: "Encourage the user about progress on a goal",
        kind: ActionKind::Communicate,
        preconditions: &[(
            "User has a goal with positive progress",
            true,
            Some(NodeKind::Goal),
        )],
        effects: &[("User feels motivated", 0.80, 0.5)],
        confidence_threshold: 0.3,
    },
    // ── Inform (5) ──
    SchemaTemplate {
        name: "surface_relevant_memory",
        description: "Recall a related past experience or decision",
        kind: ActionKind::Inform,
        preconditions: &[(
            "Related episode exists in memory",
            true,
            Some(NodeKind::Episode),
        )],
        effects: &[("User gains useful context", 0.75, 0.5)],
        confidence_threshold: 0.4,
    },
    SchemaTemplate {
        name: "present_belief_summary",
        description: "Summarize what the system believes about a topic",
        kind: ActionKind::Inform,
        preconditions: &[("Relevant belief exists", true, Some(NodeKind::Belief))],
        effects: &[("User validates or corrects beliefs", 0.80, 0.4)],
        confidence_threshold: 0.5,
    },
    SchemaTemplate {
        name: "explain_pattern",
        description: "Explain a detected behavioral pattern to the user",
        kind: ActionKind::Inform,
        preconditions: &[(
            "A routine or pattern is detected",
            true,
            Some(NodeKind::Routine),
        )],
        effects: &[
            ("User gains self-awareness", 0.85, 0.5),
            ("User may dispute the pattern", 0.20, -0.1),
        ],
        confidence_threshold: 0.5,
    },
    SchemaTemplate {
        name: "present_opportunity",
        description: "Inform the user about a time-bounded opportunity",
        kind: ActionKind::Inform,
        preconditions: &[(
            "An unexpired opportunity exists",
            true,
            Some(NodeKind::Opportunity),
        )],
        effects: &[
            ("User seizes the opportunity", 0.50, 0.7),
            ("User ignores it", 0.40, 0.0),
        ],
        confidence_threshold: 0.3,
    },
    SchemaTemplate {
        name: "share_entity_insight",
        description: "Share an insight about a person or concept the user cares about",
        kind: ActionKind::Inform,
        preconditions: &[(
            "An entity with connections exists",
            true,
            Some(NodeKind::Entity),
        )],
        effects: &[("User learns something useful", 0.65, 0.4)],
        confidence_threshold: 0.4,
    },
    // ── Organize (4) ──
    SchemaTemplate {
        name: "prioritize_tasks",
        description: "Suggest a reordering of the user's active tasks",
        kind: ActionKind::Organize,
        preconditions: &[
            ("Multiple active tasks exist", true, Some(NodeKind::Task)),
            ("At least one task has urgency", false, None),
        ],
        effects: &[("User focuses on highest-value work", 0.70, 0.6)],
        confidence_threshold: 0.5,
    },
    SchemaTemplate {
        name: "suggest_goal_decomposition",
        description: "Suggest breaking a large goal into sub-goals",
        kind: ActionKind::Organize,
        preconditions: &[(
            "A goal with low progress exists",
            true,
            Some(NodeKind::Goal),
        )],
        effects: &[("Goal becomes more actionable", 0.75, 0.5)],
        confidence_threshold: 0.5,
    },
    SchemaTemplate {
        name: "consolidate_related_notes",
        description: "Suggest merging related memory entries",
        kind: ActionKind::Organize,
        preconditions: &[(
            "Multiple related episodes exist",
            true,
            Some(NodeKind::Episode),
        )],
        effects: &[("Memory is cleaner and more useful", 0.80, 0.3)],
        confidence_threshold: 0.6,
    },
    SchemaTemplate {
        name: "archive_completed_goal",
        description: "Suggest archiving a completed or abandoned goal",
        kind: ActionKind::Organize,
        preconditions: &[(
            "A completed or abandoned goal exists",
            true,
            Some(NodeKind::Goal),
        )],
        effects: &[("Cognitive graph is tidier", 0.90, 0.2)],
        confidence_threshold: 0.3,
    },
    // ── Schedule (3) ──
    SchemaTemplate {
        name: "suggest_time_block",
        description: "Suggest blocking time for a high-urgency task",
        kind: ActionKind::Schedule,
        preconditions: &[("A task with deadline exists", true, Some(NodeKind::Task))],
        effects: &[("User allocates time and makes progress", 0.60, 0.6)],
        confidence_threshold: 0.5,
    },
    SchemaTemplate {
        name: "routine_adjustment",
        description: "Suggest adjusting a routine's timing based on drift",
        kind: ActionKind::Schedule,
        preconditions: &[(
            "A routine with low reliability exists",
            true,
            Some(NodeKind::Routine),
        )],
        effects: &[("Routine becomes more reliable", 0.55, 0.3)],
        confidence_threshold: 0.5,
    },
    SchemaTemplate {
        name: "deadline_warning",
        description: "Warn about an approaching deadline",
        kind: ActionKind::Schedule,
        preconditions: &[(
            "A task or goal has an imminent deadline",
            true,
            Some(NodeKind::Task),
        )],
        effects: &[("User takes timely action", 0.75, 0.5)],
        confidence_threshold: 0.3,
    },
    // ── Suggest (5) ──
    SchemaTemplate {
        name: "suggest_break",
        description: "Suggest the user take a break based on activity patterns",
        kind: ActionKind::Suggest,
        preconditions: &[
            (
                "User has been active for extended period",
                false,
                Some(NodeKind::Episode),
            ),
            (
                "Health or wellbeing need detected",
                false,
                Some(NodeKind::Need),
            ),
        ],
        effects: &[("User takes a break and feels refreshed", 0.50, 0.5)],
        confidence_threshold: 0.4,
    },
    SchemaTemplate {
        name: "suggest_delegation",
        description: "Suggest delegating a task that's blocking progress",
        kind: ActionKind::Suggest,
        preconditions: &[("A blocked task exists", true, Some(NodeKind::Task))],
        effects: &[("Task gets unblocked", 0.40, 0.6)],
        confidence_threshold: 0.6,
    },
    SchemaTemplate {
        name: "suggest_learning",
        description: "Suggest learning material based on detected knowledge gaps",
        kind: ActionKind::Suggest,
        preconditions: &[("An informational need exists", true, Some(NodeKind::Need))],
        effects: &[("User fills knowledge gap", 0.55, 0.5)],
        confidence_threshold: 0.4,
    },
    SchemaTemplate {
        name: "suggest_connection",
        description: "Suggest reaching out to a relevant person",
        kind: ActionKind::Suggest,
        preconditions: &[
            ("A social need exists", true, Some(NodeKind::Need)),
            (
                "A relevant entity (person) exists",
                false,
                Some(NodeKind::Entity),
            ),
        ],
        effects: &[("Social need is addressed", 0.45, 0.5)],
        confidence_threshold: 0.5,
    },
    SchemaTemplate {
        name: "suggest_alternative_approach",
        description: "Suggest a different approach to a stuck goal",
        kind: ActionKind::Suggest,
        preconditions: &[(
            "A goal with low progress exists",
            true,
            Some(NodeKind::Goal),
        )],
        effects: &[("User tries new approach and makes progress", 0.40, 0.6)],
        confidence_threshold: 0.5,
    },
    // ── Warn (3) ──
    SchemaTemplate {
        name: "risk_alert",
        description: "Alert the user about a detected risk",
        kind: ActionKind::Warn,
        preconditions: &[(
            "A risk with significant expected impact",
            true,
            Some(NodeKind::Risk),
        )],
        effects: &[
            ("User takes preventive action", 0.60, 0.7),
            ("User feels anxious", 0.25, -0.2),
        ],
        confidence_threshold: 0.4,
    },
    SchemaTemplate {
        name: "contradiction_alert",
        description: "Alert about conflicting beliefs that need resolution",
        kind: ActionKind::Warn,
        preconditions: &[("Contradicting beliefs exist", true, Some(NodeKind::Belief))],
        effects: &[("User resolves the contradiction", 0.55, 0.5)],
        confidence_threshold: 0.5,
    },
    SchemaTemplate {
        name: "constraint_violation_warning",
        description: "Warn that a planned action may violate a constraint",
        kind: ActionKind::Warn,
        preconditions: &[("A constraint exists", true, Some(NodeKind::Constraint))],
        effects: &[("User avoids the violation", 0.80, 0.4)],
        confidence_threshold: 0.3,
    },
    // ── Execute (3) ──
    SchemaTemplate {
        name: "auto_consolidate",
        description: "Automatically consolidate redundant memories",
        kind: ActionKind::Execute,
        preconditions: &[(
            "Multiple similar episodes exist",
            true,
            Some(NodeKind::Episode),
        )],
        effects: &[("Memory is cleaner", 0.90, 0.3)],
        confidence_threshold: 0.7,
    },
    SchemaTemplate {
        name: "update_belief_confidence",
        description: "Update a belief's confidence based on new evidence",
        kind: ActionKind::Execute,
        preconditions: &[(
            "A belief with new evidence exists",
            true,
            Some(NodeKind::Belief),
        )],
        effects: &[("Belief accuracy improves", 0.85, 0.3)],
        confidence_threshold: 0.5,
    },
    SchemaTemplate {
        name: "decay_stale_activations",
        description: "Decay activation of nodes that haven't been accessed",
        kind: ActionKind::Execute,
        preconditions: &[], // always valid — system maintenance
        effects: &[("Working set stays focused", 0.95, 0.2)],
        confidence_threshold: 0.2,
    },
    // ── Abstain (1) ──
    SchemaTemplate {
        name: "abstain",
        description: "Explicitly decide to take no action right now",
        kind: ActionKind::Abstain,
        preconditions: &[], // always satisfiable
        effects: &[("User is not disturbed", 0.95, 0.1)],
        confidence_threshold: 0.0,
    },
];

/// Get the number of built-in schemas.
pub fn builtin_schema_count() -> usize {
    BUILTIN_SCHEMAS.len()
}

/// Convert a SchemaTemplate to an ActionSchemaPayload.
fn template_to_payload(t: &SchemaTemplate) -> ActionSchemaPayload {
    ActionSchemaPayload {
        name: t.name.to_string(),
        description: t.description.to_string(),
        action_kind: t.kind,
        preconditions: t
            .preconditions
            .iter()
            .map(|(desc, req, _kind)| {
                Precondition {
                    description: desc.to_string(),
                    node_ref: None, // bound at generation time
                    required: *req,
                }
            })
            .collect(),
        effects: t
            .effects
            .iter()
            .map(|(desc, prob, util)| Effect {
                description: desc.to_string(),
                probability: *prob,
                utility: *util,
            })
            .collect(),
        confidence_threshold: t.confidence_threshold,
        success_rate: 0.5, // neutral prior for untried schemas
        execution_count: 0,
        acceptance_count: 0,
    }
}

/// Get all built-in schemas as ActionSchemaPayload instances.
pub fn builtin_schemas() -> Vec<ActionSchemaPayload> {
    BUILTIN_SCHEMAS.iter().map(template_to_payload).collect()
}

/// Look up a built-in schema by name.
pub fn lookup_builtin(name: &str) -> Option<ActionSchemaPayload> {
    BUILTIN_SCHEMAS
        .iter()
        .find(|t| t.name == name)
        .map(template_to_payload)
}

// ── Precondition Matching ──

/// Check if a schema's preconditions are satisfiable given available nodes.
///
/// Returns bindings for each precondition: which node (if any) satisfies it.
fn match_preconditions(
    template: &SchemaTemplate,
    nodes: &[&CognitiveNode],
) -> Vec<PreconditionBinding> {
    template
        .preconditions
        .iter()
        .map(|(desc, required, kind_opt)| {
            let bound =
                kind_opt.and_then(|kind| nodes.iter().find(|n| n.id.kind() == kind).map(|n| n.id));

            // Preconditions without a required node kind are context-based
            // (e.g., "goal hasn't been updated recently") — check via heuristics
            let satisfied = if let Some(kind) = kind_opt {
                nodes.iter().any(|n| n.id.kind() == *kind)
            } else {
                true // context preconditions are optimistically satisfied
            };

            PreconditionBinding {
                description: desc.to_string(),
                required: *required,
                satisfied,
                bound_node: bound,
            }
        })
        .collect()
}

/// Calculate precondition satisfaction ratio.
fn precondition_satisfaction(bindings: &[PreconditionBinding]) -> (usize, usize, usize) {
    let required_total = bindings.iter().filter(|b| b.required).count();
    let required_sat = bindings
        .iter()
        .filter(|b| b.required && b.satisfied)
        .count();
    let soft_sat = bindings
        .iter()
        .filter(|b| !b.required && b.satisfied)
        .count();
    (required_sat, required_total, soft_sat)
}

// ── Relevance Scoring ──

/// Compute preliminary relevance score for a candidate.
///
/// Combines:
/// - Precondition satisfaction ratio (40%)
/// - Schema success rate (25%)
/// - Intent posterior probability (25%)
/// - Soft precondition bonus (10%)
fn compute_relevance(
    required_sat: usize,
    required_total: usize,
    soft_sat: usize,
    total_soft: usize,
    schema: &SchemaTemplate,
    intent_posterior: f64,
) -> f64 {
    let req_ratio = if required_total > 0 {
        required_sat as f64 / required_total as f64
    } else {
        1.0 // no required preconditions = fully satisfied
    };

    let soft_ratio = if total_soft > 0 {
        soft_sat as f64 / total_soft as f64
    } else {
        0.0
    };

    // Neutral prior for untried schemas
    let success = 0.5_f64;
    let _ = schema; // success_rate would come from persisted schema

    0.40 * req_ratio + 0.25 * success + 0.25 * intent_posterior + 0.10 * soft_ratio
}

// ── Candidate Generation ──

/// Generate action candidates for a single intent hypothesis.
pub fn generate_candidates_for_intent(
    intent: &ScoredIntent,
    nodes: &[&CognitiveNode],
    persisted_schemas: &[&CognitiveNode],
    config: &ActionConfig,
) -> Vec<ActionCandidate> {
    let mut candidates = Vec::new();

    // Evaluate built-in schemas
    for template in BUILTIN_SCHEMAS {
        if template.kind == ActionKind::Abstain && !config.include_abstain {
            continue;
        }

        let bindings = match_preconditions(template, nodes);
        let (req_sat, req_total, soft_sat) = precondition_satisfaction(&bindings);

        // Check minimum precondition ratio
        let ratio = if req_total > 0 {
            req_sat as f64 / req_total as f64
        } else {
            1.0
        };

        if ratio < config.min_precondition_ratio {
            continue;
        }

        let total_soft = bindings.iter().filter(|b| !b.required).count();
        let relevance = compute_relevance(
            req_sat,
            req_total,
            soft_sat,
            total_soft,
            template,
            intent.posterior,
        );

        candidates.push(ActionCandidate {
            schema_name: template.name.to_string(),
            action_kind: template.kind,
            description: format!("{}: {}", template.name, template.description),
            source_intent: intent.description.clone(),
            precondition_bindings: bindings,
            satisfied_required: req_sat,
            total_required: req_total,
            satisfied_soft: soft_sat,
            relevance_score: relevance,
            schema_node: None,
        });
    }

    // Evaluate persisted (custom/learned) schemas
    for schema_node in persisted_schemas {
        if let NodePayload::ActionSchema(ref schema) = schema_node.payload {
            if schema.success_rate < config.min_success_rate {
                continue;
            }

            let bindings: Vec<PreconditionBinding> = schema
                .preconditions
                .iter()
                .map(|p| {
                    let satisfied = if let Some(node_ref) = p.node_ref {
                        nodes.iter().any(|n| n.id == node_ref)
                    } else {
                        true // no specific node required
                    };
                    PreconditionBinding {
                        description: p.description.clone(),
                        required: p.required,
                        satisfied,
                        bound_node: p.node_ref,
                    }
                })
                .collect();

            let (req_sat, req_total, soft_sat) = precondition_satisfaction(&bindings);
            let ratio = if req_total > 0 {
                req_sat as f64 / req_total as f64
            } else {
                1.0
            };

            if ratio < config.min_precondition_ratio {
                continue;
            }

            let total_soft = bindings.iter().filter(|b| !b.required).count();
            let soft_ratio = if total_soft > 0 {
                soft_sat as f64 / total_soft as f64
            } else {
                0.0
            };
            let relevance = 0.40 * ratio
                + 0.25 * schema.success_rate
                + 0.25 * intent.posterior
                + 0.10 * soft_ratio;

            candidates.push(ActionCandidate {
                schema_name: schema.name.clone(),
                action_kind: schema.action_kind,
                description: format!("{}: {}", schema.name, schema.description),
                source_intent: intent.description.clone(),
                precondition_bindings: bindings,
                satisfied_required: req_sat,
                total_required: req_total,
                satisfied_soft: soft_sat,
                relevance_score: relevance,
                schema_node: Some(schema_node.id),
            });
        }
    }

    // Sort by relevance descending, truncate
    candidates.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(config.max_candidates_per_intent);

    candidates
}

/// Generate action candidates for multiple intent hypotheses.
///
/// This is the main entry point for the candidate generator.
pub fn generate_candidates(
    intents: &[ScoredIntent],
    nodes: &[&CognitiveNode],
    persisted_schemas: &[&CognitiveNode],
    config: &ActionConfig,
) -> CandidateGenerationResult {
    let start = std::time::Instant::now();
    let schemas_evaluated = BUILTIN_SCHEMAS.len() + persisted_schemas.len();

    let mut all_candidates: Vec<ActionCandidate> = Vec::new();

    for intent in intents {
        let candidates = generate_candidates_for_intent(intent, nodes, persisted_schemas, config);
        all_candidates.extend(candidates);
    }

    let total_before_filter = all_candidates.len();

    // Deduplicate: same schema_name appearing for different intents — keep highest relevance
    all_candidates.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut seen = std::collections::HashSet::new();
    all_candidates.retain(|c| seen.insert(c.schema_name.clone()));

    all_candidates.truncate(config.max_total_candidates);

    CandidateGenerationResult {
        candidates: all_candidates,
        schemas_evaluated,
        total_before_filter,
        duration_us: start.elapsed().as_micros() as u64,
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_goal(alloc: &mut NodeIdAllocator, desc: &str, urgency: f64) -> CognitiveNode {
        let id = alloc.alloc(NodeKind::Goal);
        let mut node = CognitiveNode::new(
            id,
            desc.to_string(),
            NodePayload::Goal(GoalPayload {
                description: desc.to_string(),
                status: GoalStatus::Active,
                progress: 0.3,
                deadline: None,
                priority: Priority::High,
                parent_goal: None,
                completion_criteria: "Done".to_string(),
            }),
        );
        node.attrs.urgency = urgency;
        node
    }

    fn make_episode_node(alloc: &mut NodeIdAllocator, summary: &str) -> CognitiveNode {
        let id = alloc.alloc(NodeKind::Episode);
        CognitiveNode::new(
            id,
            summary.to_string(),
            NodePayload::Episode(EpisodePayload {
                memory_rid: format!("rid_{}", id.to_raw()),
                summary: summary.to_string(),
                occurred_at: now_secs(),
                participants: vec!["user".to_string()],
            }),
        )
    }

    fn make_need_node(alloc: &mut NodeIdAllocator, desc: &str, intensity: f64) -> CognitiveNode {
        let id = alloc.alloc(NodeKind::Need);
        let mut node = CognitiveNode::new(
            id,
            desc.to_string(),
            NodePayload::Need(NeedPayload {
                description: desc.to_string(),
                category: NeedCategory::Informational,
                intensity,
                last_satisfied: None,
                satisfaction_pattern: "search".to_string(),
            }),
        );
        node.attrs.urgency = intensity;
        node
    }

    fn make_intent(desc: &str, posterior: f64) -> ScoredIntent {
        ScoredIntent {
            description: desc.to_string(),
            source: super::super::intent::IntentSource::GoalDriven,
            features: vec![0.5; 10],
            raw_score: 0.3,
            posterior,
            supporting_nodes: vec![],
            source_node: NodeId::new(NodeKind::Goal, 0),
        }
    }

    #[test]
    fn test_builtin_schema_count() {
        assert_eq!(builtin_schema_count(), 28);
    }

    #[test]
    fn test_builtin_schemas_valid() {
        let schemas = builtin_schemas();
        assert_eq!(schemas.len(), 28);

        // Every schema should have a name and description
        for s in &schemas {
            assert!(!s.name.is_empty());
            assert!(!s.description.is_empty());
        }

        // Check all 8 action kinds are represented
        let kinds: std::collections::HashSet<_> = schemas.iter().map(|s| s.action_kind).collect();
        assert_eq!(
            kinds.len(),
            8,
            "all 8 ActionKind variants should be represented"
        );
    }

    #[test]
    fn test_lookup_builtin() {
        let schema = lookup_builtin("send_reminder").unwrap();
        assert_eq!(schema.action_kind, ActionKind::Communicate);
        assert!(!schema.preconditions.is_empty());

        assert!(lookup_builtin("nonexistent").is_none());
    }

    #[test]
    fn test_generate_candidates_with_goal() {
        let mut alloc = NodeIdAllocator::new();
        let goal = make_goal(&mut alloc, "Finish report", 0.8);
        let episode = make_episode_node(&mut alloc, "Working on report");

        let nodes: Vec<&CognitiveNode> = vec![&goal, &episode];
        let intent = make_intent("Advance goal: Finish report", 0.6);
        let config = ActionConfig::default();

        let result = generate_candidates(&[intent], &nodes, &[], &config);

        assert!(
            !result.candidates.is_empty(),
            "should find matching schemas"
        );
        assert!(result.schemas_evaluated >= 28);

        // Should include at least send_reminder (needs Goal)
        let has_reminder = result
            .candidates
            .iter()
            .any(|c| c.schema_name == "send_reminder");
        assert!(has_reminder, "send_reminder should match (Goal exists)");
    }

    #[test]
    fn test_generate_candidates_no_nodes() {
        let intent = make_intent("Some intent", 0.5);
        let config = ActionConfig::default();

        let result = generate_candidates(&[intent], &[], &[], &config);

        // Only schemas with no required preconditions should match
        // (abstain, decay_stale_activations, suggest_break)
        for c in &result.candidates {
            assert_eq!(
                c.total_required, 0,
                "without nodes, only no-precondition schemas should match"
            );
        }
    }

    #[test]
    fn test_abstain_always_available() {
        let intent = make_intent("Any intent", 0.3);
        let config = ActionConfig::default();

        let result = generate_candidates(&[intent], &[], &[], &config);
        let has_abstain = result.candidates.iter().any(|c| c.schema_name == "abstain");
        assert!(has_abstain, "abstain should always be available");
    }

    #[test]
    fn test_abstain_can_be_disabled() {
        let intent = make_intent("Any intent", 0.3);
        let mut config = ActionConfig::default();
        config.include_abstain = false;

        let result = generate_candidates(&[intent], &[], &[], &config);
        let has_abstain = result.candidates.iter().any(|c| c.schema_name == "abstain");
        assert!(!has_abstain, "abstain should be excluded when disabled");
    }

    #[test]
    fn test_candidates_sorted_by_relevance() {
        let mut alloc = NodeIdAllocator::new();
        let goal = make_goal(&mut alloc, "Test goal", 0.8);
        let episode = make_episode_node(&mut alloc, "Test episode");
        let need = make_need_node(&mut alloc, "Learn something", 0.7);

        let nodes: Vec<&CognitiveNode> = vec![&goal, &episode, &need];
        let intent = make_intent("Test intent", 0.7);
        let config = ActionConfig::default();

        let result = generate_candidates(&[intent], &nodes, &[], &config);

        for w in result.candidates.windows(2) {
            assert!(
                w[0].relevance_score >= w[1].relevance_score,
                "candidates should be sorted by relevance descending"
            );
        }
    }

    #[test]
    fn test_deduplication_across_intents() {
        let mut alloc = NodeIdAllocator::new();
        let goal = make_goal(&mut alloc, "Goal A", 0.8);
        let nodes: Vec<&CognitiveNode> = vec![&goal];

        let intent1 = make_intent("Intent 1", 0.6);
        let intent2 = make_intent("Intent 2", 0.4);
        let config = ActionConfig::default();

        let result = generate_candidates(&[intent1, intent2], &nodes, &[], &config);

        // No duplicate schema names
        let names: Vec<&str> = result
            .candidates
            .iter()
            .map(|c| c.schema_name.as_str())
            .collect();
        let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
        assert_eq!(
            names.len(),
            unique.len(),
            "no duplicate schema names after dedup"
        );
    }

    #[test]
    fn test_persisted_schema_evaluation() {
        let mut alloc = NodeIdAllocator::new();
        let goal = make_goal(&mut alloc, "Custom goal", 0.8);

        // Create a custom persisted schema
        let schema_id = alloc.alloc(NodeKind::ActionSchema);
        let custom_schema = CognitiveNode::new(
            schema_id,
            "custom_action".to_string(),
            NodePayload::ActionSchema(ActionSchemaPayload {
                name: "custom_action".to_string(),
                description: "A custom learned action".to_string(),
                action_kind: ActionKind::Suggest,
                preconditions: vec![],
                effects: vec![Effect {
                    description: "Something good happens".to_string(),
                    probability: 0.8,
                    utility: 0.5,
                }],
                confidence_threshold: 0.3,
                success_rate: 0.75,
                execution_count: 10,
                acceptance_count: 8,
            }),
        );

        let nodes: Vec<&CognitiveNode> = vec![&goal];
        let schemas: Vec<&CognitiveNode> = vec![&custom_schema];
        let intent = make_intent("Test", 0.5);
        let config = ActionConfig::default();

        let result = generate_candidates(&[intent], &nodes, &schemas, &config);

        let has_custom = result
            .candidates
            .iter()
            .any(|c| c.schema_name == "custom_action");
        assert!(has_custom, "persisted schema should be included");
    }

    #[test]
    fn test_max_candidates_limit() {
        let mut alloc = NodeIdAllocator::new();
        // Create nodes of many kinds to match many schemas
        let goal = make_goal(&mut alloc, "Goal", 0.8);
        let ep = make_episode_node(&mut alloc, "Episode");
        let need = make_need_node(&mut alloc, "Need", 0.6);

        let belief_id = alloc.alloc(NodeKind::Belief);
        let belief = CognitiveNode::new(
            belief_id,
            "Belief".to_string(),
            NodePayload::Belief(BeliefPayload {
                proposition: "Something is true".to_string(),
                log_odds: 2.0,
                domain: "test".to_string(),
                evidence_trail: vec![],
                user_confirmed: false,
            }),
        );

        let task_id = alloc.alloc(NodeKind::Task);
        let task = CognitiveNode::new(
            task_id,
            "Task".to_string(),
            NodePayload::Task(TaskPayload {
                description: "Do something".to_string(),
                status: TaskStatus::Pending,
                goal_id: None,
                deadline: None,
                priority: Priority::Medium,
                estimated_minutes: Some(60),
                prerequisites: vec![],
            }),
        );

        let risk_id = alloc.alloc(NodeKind::Risk);
        let risk = CognitiveNode::new(
            risk_id,
            "Risk".to_string(),
            NodePayload::Risk(RiskPayload {
                description: "Something bad".to_string(),
                severity: 0.7,
                likelihood: 0.5,
                mitigation: "Be careful".to_string(),
                threatened_goals: vec![],
            }),
        );

        let constraint_id = alloc.alloc(NodeKind::Constraint);
        let constraint = CognitiveNode::new(
            constraint_id,
            "Constraint".to_string(),
            NodePayload::Constraint(ConstraintPayload {
                description: "No late notifications".to_string(),
                constraint_type: ConstraintType::Hard,
                condition: "always".to_string(),
                imposed_by: "user".to_string(),
            }),
        );

        let routine_id = alloc.alloc(NodeKind::Routine);
        let routine = CognitiveNode::new(
            routine_id,
            "Routine".to_string(),
            NodePayload::Routine(RoutinePayload {
                description: "Daily check".to_string(),
                period_secs: 86400.0,
                phase_offset_secs: 0.0,
                reliability: 0.3, // low reliability
                observation_count: 5,
                last_triggered: 0.0,
                action_description: "check".to_string(),
                weekday_mask: 0x7F,
            }),
        );

        let opp_id = alloc.alloc(NodeKind::Opportunity);
        let opportunity = CognitiveNode::new(
            opp_id,
            "Opportunity".to_string(),
            NodePayload::Opportunity(OpportunityPayload {
                description: "Sale".to_string(),
                expires_at: now_secs() + 3600.0,
                expected_benefit: 0.8,
                required_action: "Buy".to_string(),
                relevant_goals: vec![],
            }),
        );

        let entity_id = alloc.alloc(NodeKind::Entity);
        let entity = CognitiveNode::new(
            entity_id,
            "Entity".to_string(),
            NodePayload::Entity(EntityPayload {
                name: "Alice".to_string(),
                entity_type: "person".to_string(),
                memory_rids: vec![],
            }),
        );

        let nodes: Vec<&CognitiveNode> = vec![
            &goal,
            &ep,
            &need,
            &belief,
            &task,
            &risk,
            &constraint,
            &routine,
            &opportunity,
            &entity,
        ];

        let intent = make_intent("Big intent", 0.8);
        let mut config = ActionConfig::default();
        config.max_total_candidates = 10;

        let result = generate_candidates(&[intent], &nodes, &[], &config);
        assert!(
            result.candidates.len() <= 10,
            "should respect max_total_candidates"
        );
    }
}
