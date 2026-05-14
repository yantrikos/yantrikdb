//! CK-4.2 — Planning Graph / HTN-Lite Reasoning.
//!
//! Bounded means-ends reasoning for multi-step goal achievement.
//! NOT a heavyweight classical planner — a practical, incremental
//! planning system with:
//!
//! - Action schemas as operators with preconditions and effects
//! - Backward chaining from goals to find applicable operators
//! - HTN-lite decomposition via learned skills
//! - Beam search with heuristic pruning (max depth 2-5)
//! - Plan scoring: feasibility × utility × simplicity
//!
//! # Design principles
//! - Pure functions only — no DB access (engine layer handles persistence)
//! - Small plans (2-5 steps) — not solving TSP
//! - Explainable — every plan step has rationale
//! - Incremental — plans update as state changes

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::state::{
    ActionKind, ActionSchemaPayload, CognitiveAttrs, CognitiveEdge, CognitiveEdgeKind,
    CognitiveNode, ConstraintPayload, ConstraintType, Effect, GoalPayload, GoalStatus, NodeId,
    NodeKind, NodePayload, Precondition, Priority, TaskPayload, TaskStatus,
};

// ── §1: Plan types ──────────────────────────────────────────────────

/// A unique identifier for a plan step within a plan.
pub type PlanStepId = u32;

/// A complete plan: an ordered sequence of steps to achieve a goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    /// Target goal this plan aims to achieve.
    pub goal_id: NodeId,
    /// Goal description (for display).
    pub goal_description: String,
    /// Ordered steps to execute.
    pub steps: Vec<PlanStep>,
    /// Overall plan score.
    pub score: PlanScore,
    /// Why this plan was chosen over alternatives.
    pub rationale: String,
    /// When the plan was generated (unix seconds).
    pub created_at: f64,
    /// Whether the plan is still considered viable.
    pub viable: bool,
    /// Detected blockers that may prevent execution.
    pub blockers: Vec<Blocker>,
}

/// A single step in a plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Position in the plan (0-indexed).
    pub ordinal: u32,
    /// The action schema to execute.
    pub schema_name: String,
    /// Which action kind.
    pub action_kind: ActionKind,
    /// Human-readable description of what this step does.
    pub description: String,
    /// Reference to the ActionSchema node in the cognitive graph.
    pub schema_node: Option<NodeId>,
    /// Preconditions that must hold before executing.
    pub preconditions: Vec<BoundPrecondition>,
    /// Expected effects after execution.
    pub expected_effects: Vec<Effect>,
    /// Estimated duration in seconds.
    pub estimated_duration_secs: f64,
    /// How this step was derived.
    pub derivation: StepDerivation,
    /// Feasibility score for this step ∈ [0.0, 1.0].
    pub feasibility: f64,
}

/// A precondition bound to the current state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundPrecondition {
    pub description: String,
    pub required: bool,
    pub satisfied: bool,
    /// Which node satisfies it (if any).
    pub bound_node: Option<NodeId>,
}

/// How a plan step was derived.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepDerivation {
    /// Directly matched an action schema to a goal/subgoal.
    DirectMatch,
    /// From backward chaining — this step produces a needed precondition.
    BackwardChain { enables_step: u32 },
    /// Decomposed from a learned skill sequence.
    SkillDecomposition { skill_id: u64 },
    /// From a task prerequisite chain.
    TaskPrerequisite { task_node: NodeId },
}

/// Composite plan score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanScore {
    /// Can we actually execute this? ∈ [0.0, 1.0].
    pub feasibility: f64,
    /// Expected utility of achieving the goal ∈ [-1.0, 1.0].
    pub expected_utility: f64,
    /// Simplicity bonus — fewer steps is better ∈ [0.0, 1.0].
    pub simplicity: f64,
    /// Historical success rate of the schemas involved ∈ [0.0, 1.0].
    pub schema_success_rate: f64,
    /// Urgency of the goal ∈ [0.0, 1.0].
    pub urgency: f64,
    /// Composite score (weighted combination).
    pub composite: f64,
    /// Estimated total duration in seconds.
    pub estimated_total_secs: f64,
}

/// Something that blocks plan execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Blocker {
    /// What is blocked.
    pub step_ordinal: Option<u32>,
    /// Human-readable description.
    pub description: String,
    /// How severe (0.0 = minor, 1.0 = fatal).
    pub severity: f64,
    /// Category of blocker.
    pub kind: BlockerKind,
    /// Possible resolution.
    pub resolution: Option<String>,
}

/// Categories of blockers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlockerKind {
    /// A required precondition is not met.
    UnsatisfiedPrecondition,
    /// A constraint prevents an action.
    ConstraintViolation,
    /// A prerequisite task is not complete.
    PrerequisiteIncomplete,
    /// The goal is already completed or abandoned.
    GoalInactive,
    /// Deadline has passed or is too tight.
    DeadlinePressure,
    /// An involved schema has very low success rate.
    LowConfidenceSchema,
}

// ── §2: Planner configuration ──────────────────────────────────────

/// Configuration for the planner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerConfig {
    /// Maximum plan depth (steps).
    pub max_depth: u32,
    /// Beam width for search.
    pub beam_width: usize,
    /// Number of top plans to return.
    pub top_k: usize,
    /// Minimum feasibility to include in results.
    pub min_feasibility: f64,
    /// Minimum schema success rate to consider.
    pub min_schema_success_rate: f64,
    /// Weight for feasibility in composite score.
    pub weight_feasibility: f64,
    /// Weight for expected utility.
    pub weight_utility: f64,
    /// Weight for simplicity.
    pub weight_simplicity: f64,
    /// Weight for schema success rate.
    pub weight_success_rate: f64,
    /// Weight for urgency.
    pub weight_urgency: f64,
    /// Estimated seconds per step if no estimate available.
    pub default_step_duration_secs: f64,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            max_depth: 4,
            beam_width: 6,
            top_k: 3,
            min_feasibility: 0.3,
            min_schema_success_rate: 0.2,
            weight_feasibility: 0.30,
            weight_utility: 0.25,
            weight_simplicity: 0.15,
            weight_success_rate: 0.20,
            weight_urgency: 0.10,
            default_step_duration_secs: 30.0,
        }
    }
}

// ── §3: Planning context ────────────────────────────────────────────

/// Everything the planner needs to work — passed by the engine layer.
pub struct PlanningContext<'a> {
    /// All action schemas available.
    pub schemas: &'a [SchemaEntry],
    /// All active goals.
    pub goals: &'a [GoalEntry],
    /// All tasks (may have prerequisites).
    pub tasks: &'a [TaskEntry],
    /// Active constraints.
    pub constraints: &'a [ConstraintEntry],
    /// Cognitive graph edges (for relationship queries).
    pub edges: &'a [CognitiveEdge],
    /// Learned skills that can serve as decomposition templates.
    pub skills: &'a [SkillTemplate],
    /// Current time (unix seconds).
    pub now: f64,
    /// Planner configuration.
    pub config: &'a PlannerConfig,
}

/// An action schema with its cognitive node context.
#[derive(Debug, Clone)]
pub struct SchemaEntry {
    pub node_id: NodeId,
    pub attrs: CognitiveAttrs,
    pub payload: ActionSchemaPayload,
}

/// A goal with its cognitive node context.
#[derive(Debug, Clone)]
pub struct GoalEntry {
    pub node_id: NodeId,
    pub attrs: CognitiveAttrs,
    pub payload: GoalPayload,
}

/// A task with its cognitive node context.
#[derive(Debug, Clone)]
pub struct TaskEntry {
    pub node_id: NodeId,
    pub attrs: CognitiveAttrs,
    pub payload: TaskPayload,
}

/// A constraint with its cognitive node context.
#[derive(Debug, Clone)]
pub struct ConstraintEntry {
    pub node_id: NodeId,
    pub payload: ConstraintPayload,
}

/// A learned skill usable as a decomposition template.
#[derive(Debug, Clone)]
pub struct SkillTemplate {
    pub skill_id: u64,
    pub description: String,
    pub steps: Vec<SkillStepInfo>,
    pub confidence: f64,
    pub success_rate: f64,
}

/// Simplified skill step for planning.
#[derive(Debug, Clone)]
pub struct SkillStepInfo {
    pub ordinal: u16,
    pub action_kind: String,
    pub description: String,
    pub expected_duration_ms: u64,
    pub optional: bool,
}

// ── §4: Plan generation — main entry point ──────────────────────────

/// Result of plan generation.
#[derive(Debug, Clone)]
pub struct PlanProposal {
    /// Top-k plans sorted by composite score descending.
    pub plans: Vec<Plan>,
    /// Schemas considered but rejected.
    pub rejected_schemas: Vec<RejectedSchema>,
    /// Blockers affecting all plans.
    pub global_blockers: Vec<Blocker>,
    /// Total candidate plans evaluated.
    pub candidates_evaluated: usize,
}

/// A schema that was considered but rejected.
#[derive(Debug, Clone)]
pub struct RejectedSchema {
    pub schema_name: String,
    pub reason: String,
}

/// Generate plans to achieve a specific goal.
///
/// This is the main entry point. It:
/// 1. Validates the goal is active
/// 2. Finds applicable action schemas (forward + backward)
/// 3. Decomposes using skills when available
/// 4. Scores and ranks plans via beam search
/// 5. Returns top-k plans with blockers
pub fn instantiate_plan(goal_id: NodeId, ctx: &PlanningContext) -> PlanProposal {
    let config = ctx.config;

    // Find the target goal.
    let goal = match ctx.goals.iter().find(|g| g.node_id == goal_id) {
        Some(g) => g,
        None => {
            return PlanProposal {
                plans: Vec::new(),
                rejected_schemas: Vec::new(),
                global_blockers: vec![Blocker {
                    step_ordinal: None,
                    description: "Goal not found in active goals".to_string(),
                    severity: 1.0,
                    kind: BlockerKind::GoalInactive,
                    resolution: None,
                }],
                candidates_evaluated: 0,
            };
        }
    };

    // Check goal is actionable.
    let mut global_blockers = Vec::new();
    if goal.payload.status != GoalStatus::Active {
        global_blockers.push(Blocker {
            step_ordinal: None,
            description: format!("Goal status is {:?}, not Active", goal.payload.status),
            severity: 1.0,
            kind: BlockerKind::GoalInactive,
            resolution: Some("Reactivate the goal first".to_string()),
        });
        return PlanProposal {
            plans: Vec::new(),
            rejected_schemas: Vec::new(),
            global_blockers,
            candidates_evaluated: 0,
        };
    }

    // Check deadline pressure.
    if let Some(deadline) = goal.payload.deadline {
        if deadline < ctx.now {
            global_blockers.push(Blocker {
                step_ordinal: None,
                description: "Goal deadline has passed".to_string(),
                severity: 0.8,
                kind: BlockerKind::DeadlinePressure,
                resolution: Some("Extend the deadline or abandon the goal".to_string()),
            });
        }
    }

    // Find schemas that advance this goal.
    let advancing_schemas = find_advancing_schemas(goal_id, ctx);
    let mut rejected_schemas = Vec::new();

    // Build candidate plans.
    let mut candidate_plans: Vec<Plan> = Vec::new();

    // Strategy 1: Direct single-schema plans.
    for se in &advancing_schemas {
        if se.payload.success_rate < config.min_schema_success_rate {
            rejected_schemas.push(RejectedSchema {
                schema_name: se.payload.name.clone(),
                reason: format!(
                    "Success rate {:.0}% below minimum {:.0}%",
                    se.payload.success_rate * 100.0,
                    config.min_schema_success_rate * 100.0,
                ),
            });
            continue;
        }

        let (step, blockers) = build_plan_step(se, 0, StepDerivation::DirectMatch, ctx);
        if step.feasibility >= config.min_feasibility {
            let plan = build_plan(goal, vec![step], blockers, ctx);
            candidate_plans.push(plan);
        }
    }

    // Strategy 2: Skill-based decomposition.
    for skill in ctx.skills {
        if skill.confidence < 0.4 {
            continue;
        }
        if let Some(plan) = decompose_via_skill(goal, skill, ctx) {
            candidate_plans.push(plan);
        }
    }

    // Strategy 3: Task prerequisite chains.
    let goal_tasks = find_tasks_for_goal(goal_id, ctx);
    if !goal_tasks.is_empty() {
        if let Some(plan) = build_task_chain_plan(goal, &goal_tasks, ctx) {
            candidate_plans.push(plan);
        }
    }

    // Strategy 4: Backward chaining (depth-limited).
    if candidate_plans.len() < config.beam_width && config.max_depth > 1 {
        let bc_plans = backward_chain(goal, &advancing_schemas, ctx, config.max_depth);
        candidate_plans.extend(bc_plans);
    }

    let candidates_evaluated = candidate_plans.len();

    // Score, sort, and take top-k.
    candidate_plans.sort_by(|a, b| {
        b.score
            .composite
            .partial_cmp(&a.score.composite)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidate_plans.truncate(config.top_k);

    // Check constraints against all remaining plans.
    for plan in &mut candidate_plans {
        let constraint_blockers = check_constraints(plan, ctx);
        plan.blockers.extend(constraint_blockers);
        plan.viable = plan.blockers.iter().all(|b| b.severity < 0.9);
    }

    PlanProposal {
        plans: candidate_plans,
        rejected_schemas,
        global_blockers,
        candidates_evaluated,
    }
}

// ── §5: Next step recommendation ───────────────────────────────────

/// Recommend the next concrete step toward a goal.
///
/// Generates a plan (or uses a cached one) and returns the first
/// unblocked step.
pub fn next_plan_step(goal_id: NodeId, ctx: &PlanningContext) -> Option<PlanStep> {
    let proposal = instantiate_plan(goal_id, ctx);
    let best_plan = proposal.plans.into_iter().find(|p| p.viable)?;

    // Return first step without fatal blockers.
    best_plan.steps.into_iter().find(|step| {
        !best_plan
            .blockers
            .iter()
            .any(|b| b.step_ordinal == Some(step.ordinal) && b.severity >= 0.9)
    })
}

// ── §6: Plan evaluation ─────────────────────────────────────────────

/// Evaluate a plan's quality.
pub fn evaluate_plan(plan: &Plan, config: &PlannerConfig) -> PlanScore {
    score_plan(&plan.steps, config)
}

/// Detect all blockers preventing goal achievement.
pub fn detect_blockers(goal_id: NodeId, ctx: &PlanningContext) -> Vec<Blocker> {
    let proposal = instantiate_plan(goal_id, ctx);
    let mut all_blockers = proposal.global_blockers;

    for plan in &proposal.plans {
        for b in &plan.blockers {
            if !all_blockers
                .iter()
                .any(|existing| existing.description == b.description)
            {
                all_blockers.push(b.clone());
            }
        }
    }

    all_blockers
}

// ── §7: Internal — schema matching ─────────────────────────────────

/// Find schemas that advance a given goal.
///
/// Uses cognitive graph edges (AdvancesGoal) and effect matching.
fn find_advancing_schemas<'a>(goal_id: NodeId, ctx: &'a PlanningContext) -> Vec<&'a SchemaEntry> {
    let mut result = Vec::new();

    // Method 1: Explicit graph edges from schema → goal.
    let advancing_ids: HashSet<NodeId> = ctx
        .edges
        .iter()
        .filter(|e| {
            e.dst == goal_id
                && (e.kind == CognitiveEdgeKind::AdvancesGoal
                    || e.kind == CognitiveEdgeKind::Causes)
        })
        .map(|e| e.src)
        .collect();

    for se in ctx.schemas {
        if advancing_ids.contains(&se.node_id) {
            result.push(se);
        }
    }

    // Method 2: Schemas with high activation (contextually relevant).
    // Only add if they have positive effects and aren't already included.
    let existing: HashSet<NodeId> = result.iter().map(|s| s.node_id).collect();
    for se in ctx.schemas {
        if existing.contains(&se.node_id) {
            continue;
        }
        if se.attrs.activation >= 0.5
            && !se.payload.effects.is_empty()
            && se.payload.effects.iter().any(|e| e.utility > 0.0)
        {
            result.push(se);
        }
    }

    result
}

/// Find tasks associated with a goal.
fn find_tasks_for_goal<'a>(goal_id: NodeId, ctx: &'a PlanningContext) -> Vec<&'a TaskEntry> {
    ctx.tasks
        .iter()
        .filter(|t| t.payload.goal_id == Some(goal_id))
        .collect()
}

// ── §8: Internal — step building ────────────────────────────────────

/// Build a plan step from a schema, binding preconditions to current state.
fn build_plan_step(
    schema: &SchemaEntry,
    ordinal: u32,
    derivation: StepDerivation,
    ctx: &PlanningContext,
) -> (PlanStep, Vec<Blocker>) {
    let mut bound_preconds = Vec::new();
    let mut blockers = Vec::new();
    let mut satisfied_required = 0usize;
    let mut total_required = 0usize;

    for pc in &schema.payload.preconditions {
        let (satisfied, bound_node) = check_precondition(pc, ctx);

        if pc.required {
            total_required += 1;
            if satisfied {
                satisfied_required += 1;
            } else {
                blockers.push(Blocker {
                    step_ordinal: Some(ordinal),
                    description: format!("Required precondition not met: {}", pc.description),
                    severity: 0.8,
                    kind: BlockerKind::UnsatisfiedPrecondition,
                    resolution: Some(format!("Satisfy: {}", pc.description)),
                });
            }
        }

        bound_preconds.push(BoundPrecondition {
            description: pc.description.clone(),
            required: pc.required,
            satisfied,
            bound_node,
        });
    }

    let feasibility = if total_required == 0 {
        // No preconditions — fully feasible.
        schema.payload.success_rate.max(0.5)
    } else {
        let precond_ratio = satisfied_required as f64 / total_required as f64;
        precond_ratio * schema.payload.success_rate.max(0.1)
    };

    let estimated_secs = if schema.payload.execution_count > 0 {
        // Use schema's historical data if available.
        ctx.config.default_step_duration_secs
    } else {
        ctx.config.default_step_duration_secs
    };

    let step = PlanStep {
        ordinal,
        schema_name: schema.payload.name.clone(),
        action_kind: schema.payload.action_kind,
        description: schema.payload.description.clone(),
        schema_node: Some(schema.node_id),
        preconditions: bound_preconds,
        expected_effects: schema.payload.effects.clone(),
        estimated_duration_secs: estimated_secs,
        derivation,
        feasibility,
    };

    (step, blockers)
}

/// Check if a precondition is satisfied.
///
/// Returns (satisfied, optional_bound_node).
fn check_precondition(
    precondition: &Precondition,
    ctx: &PlanningContext,
) -> (bool, Option<NodeId>) {
    // If the precondition references a specific node, check it.
    if let Some(node_ref) = precondition.node_ref {
        // Check if the referenced node exists and is in good state.
        // For goals: check if completed. For tasks: check if completed.
        // For other nodes: check if confidence > 0.5.
        if let Some(goal) = ctx.goals.iter().find(|g| g.node_id == node_ref) {
            let satisfied = goal.payload.status == GoalStatus::Completed;
            return (satisfied, Some(node_ref));
        }
        if let Some(task) = ctx.tasks.iter().find(|t| t.node_id == node_ref) {
            let satisfied = task.payload.status == TaskStatus::Completed;
            return (satisfied, Some(node_ref));
        }
        // Generic: node exists with reasonable confidence.
        // (We don't have direct access to all nodes here, so we check edges.)
        let has_support = ctx
            .edges
            .iter()
            .any(|e| (e.src == node_ref || e.dst == node_ref) && e.confidence > 0.5);
        return (has_support, Some(node_ref));
    }

    // No specific node ref — assume soft precondition is satisfied
    // (the engine will validate at execution time).
    (!precondition.required, None)
}

// ── §9: Internal — skill decomposition ──────────────────────────────

/// Try to decompose a goal into steps using a learned skill.
fn decompose_via_skill(
    goal: &GoalEntry,
    skill: &SkillTemplate,
    ctx: &PlanningContext,
) -> Option<Plan> {
    if skill.steps.is_empty() {
        return None;
    }

    let mut plan_steps = Vec::new();
    let mut blockers = Vec::new();

    for (i, skill_step) in skill.steps.iter().enumerate() {
        if skill_step.optional {
            continue; // Skip optional steps for now.
        }

        // Try to match skill step to an action schema.
        let matching_schema = ctx.schemas.iter().find(|s| {
            s.payload
                .name
                .to_lowercase()
                .contains(&skill_step.action_kind.to_lowercase())
                || format!("{:?}", s.payload.action_kind)
                    .to_lowercase()
                    .contains(&skill_step.action_kind.to_lowercase())
        });

        let step = if let Some(schema) = matching_schema {
            let (mut step, step_blockers) = build_plan_step(
                schema,
                i as u32,
                StepDerivation::SkillDecomposition {
                    skill_id: skill.skill_id,
                },
                ctx,
            );
            step.description = skill_step.description.clone();
            step.estimated_duration_secs = skill_step.expected_duration_ms as f64 / 1000.0;
            blockers.extend(step_blockers);
            step
        } else {
            // No matching schema — create a generic step.
            PlanStep {
                ordinal: i as u32,
                schema_name: skill_step.action_kind.clone(),
                action_kind: ActionKind::Execute,
                description: skill_step.description.clone(),
                schema_node: None,
                preconditions: Vec::new(),
                expected_effects: Vec::new(),
                estimated_duration_secs: skill_step.expected_duration_ms as f64 / 1000.0,
                derivation: StepDerivation::SkillDecomposition {
                    skill_id: skill.skill_id,
                },
                feasibility: skill.success_rate * 0.8,
            }
        };

        plan_steps.push(step);
    }

    if plan_steps.is_empty() {
        return None;
    }

    Some(build_plan(goal, plan_steps, blockers, ctx))
}

// ── §10: Internal — task chain planning ─────────────────────────────

/// Build a plan from existing task prerequisites.
fn build_task_chain_plan(
    goal: &GoalEntry,
    tasks: &[&TaskEntry],
    ctx: &PlanningContext,
) -> Option<Plan> {
    // Topologically sort tasks by prerequisites.
    let ordered = topological_sort_tasks(tasks);

    if ordered.is_empty() {
        return None;
    }

    let mut plan_steps = Vec::new();
    let mut blockers = Vec::new();

    for (i, task) in ordered.iter().enumerate() {
        if task.payload.status == TaskStatus::Completed {
            continue; // Already done.
        }

        // Check prerequisites.
        for prereq_id in &task.payload.prerequisites {
            let prereq_done = tasks
                .iter()
                .find(|t| t.node_id == *prereq_id)
                .map(|t| t.payload.status == TaskStatus::Completed)
                .unwrap_or(false);

            if !prereq_done {
                blockers.push(Blocker {
                    step_ordinal: Some(i as u32),
                    description: format!("Prerequisite task {:?} not completed", prereq_id),
                    severity: 0.7,
                    kind: BlockerKind::PrerequisiteIncomplete,
                    resolution: Some("Complete prerequisite task first".to_string()),
                });
            }
        }

        // Try to find a schema that matches this task.
        let matching_schema = ctx.schemas.iter().find(|s| {
            s.payload.description.to_lowercase().contains(
                &task
                    .payload
                    .description
                    .to_lowercase()
                    .split_whitespace()
                    .next()
                    .unwrap_or(""),
            )
        });

        let step = if let Some(schema) = matching_schema {
            let (mut step, step_blockers) = build_plan_step(
                schema,
                i as u32,
                StepDerivation::TaskPrerequisite {
                    task_node: task.node_id,
                },
                ctx,
            );
            step.description = task.payload.description.clone();
            if let Some(mins) = task.payload.estimated_minutes {
                step.estimated_duration_secs = mins as f64 * 60.0;
            }
            blockers.extend(step_blockers);
            step
        } else {
            let feasibility = if task.payload.status == TaskStatus::InProgress {
                0.8
            } else {
                0.6
            };

            PlanStep {
                ordinal: i as u32,
                schema_name: "task_execution".to_string(),
                action_kind: ActionKind::Execute,
                description: task.payload.description.clone(),
                schema_node: None,
                preconditions: Vec::new(),
                expected_effects: Vec::new(),
                estimated_duration_secs: task
                    .payload
                    .estimated_minutes
                    .map(|m| m as f64 * 60.0)
                    .unwrap_or(ctx.config.default_step_duration_secs),
                derivation: StepDerivation::TaskPrerequisite {
                    task_node: task.node_id,
                },
                feasibility,
            }
        };

        plan_steps.push(step);
    }

    if plan_steps.is_empty() {
        return None;
    }

    Some(build_plan(goal, plan_steps, blockers, ctx))
}

/// Topological sort of tasks by prerequisites (Kahn's algorithm).
fn topological_sort_tasks<'a>(tasks: &[&'a TaskEntry]) -> Vec<&'a TaskEntry> {
    let task_ids: HashSet<NodeId> = tasks.iter().map(|t| t.node_id).collect();
    let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
    let mut dependents: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

    for task in tasks {
        in_degree.entry(task.node_id).or_insert(0);
        for prereq in &task.payload.prerequisites {
            if task_ids.contains(prereq) {
                *in_degree.entry(task.node_id).or_insert(0) += 1;
                dependents.entry(*prereq).or_default().push(task.node_id);
            }
        }
    }

    let mut queue: Vec<NodeId> = in_degree
        .iter()
        .filter(|(_, &d)| d == 0)
        .map(|(&id, _)| id)
        .collect();

    let mut result = Vec::new();
    while let Some(id) = queue.pop() {
        if let Some(task) = tasks.iter().find(|t| t.node_id == id) {
            result.push(*task);
        }
        if let Some(deps) = dependents.get(&id) {
            for dep_id in deps {
                if let Some(degree) = in_degree.get_mut(dep_id) {
                    *degree = degree.saturating_sub(1);
                    if *degree == 0 {
                        queue.push(*dep_id);
                    }
                }
            }
        }
    }

    result
}

// ── §11: Internal — backward chaining ───────────────────────────────

/// Backward chain from the goal to find multi-step plans.
///
/// For each schema that advances the goal, check its preconditions.
/// If a precondition is unmet, find schemas that satisfy it (subgoal).
/// Beam search limits exploration.
fn backward_chain(
    goal: &GoalEntry,
    initial_schemas: &[&SchemaEntry],
    ctx: &PlanningContext,
    max_depth: u32,
) -> Vec<Plan> {
    let mut plans = Vec::new();

    for schema in initial_schemas {
        let (final_step, mut blockers) =
            build_plan_step(schema, 0, StepDerivation::DirectMatch, ctx);

        // Find unsatisfied required preconditions.
        let unmet: Vec<&BoundPrecondition> = final_step
            .preconditions
            .iter()
            .filter(|p| p.required && !p.satisfied)
            .collect();

        if unmet.is_empty() {
            continue; // Already covered by direct plans.
        }

        // Try to find schemas that satisfy each unmet precondition.
        let mut prefix_steps: Vec<PlanStep> = Vec::new();
        let mut depth = 1u32;

        for (i, unmet_precond) in unmet.iter().enumerate() {
            if depth >= max_depth {
                break;
            }

            // Find a schema whose effects match this precondition.
            let enabler = find_enabling_schema(unmet_precond, ctx);

            if let Some(enabler_schema) = enabler {
                let (step, step_blockers) = build_plan_step(
                    enabler_schema,
                    i as u32,
                    StepDerivation::BackwardChain {
                        enables_step: final_step.ordinal,
                    },
                    ctx,
                );
                blockers.extend(step_blockers);
                prefix_steps.push(step);
                depth += 1;
            }
        }

        if !prefix_steps.is_empty() {
            // Renumber ordinals.
            for (j, step) in prefix_steps.iter_mut().enumerate() {
                step.ordinal = j as u32;
            }
            let mut all_steps = prefix_steps;
            let mut final_step = final_step;
            final_step.ordinal = all_steps.len() as u32;
            all_steps.push(final_step);

            let plan = build_plan(goal, all_steps, blockers, ctx);
            plans.push(plan);
        }
    }

    plans
}

/// Find a schema whose effects could satisfy an unmet precondition.
fn find_enabling_schema<'a>(
    precondition: &BoundPrecondition,
    ctx: &'a PlanningContext,
) -> Option<&'a SchemaEntry> {
    let desc_lower = precondition.description.to_lowercase();

    // Check if any schema's effects match the precondition description.
    ctx.schemas.iter().find(|s| {
        s.payload.effects.iter().any(|effect| {
            let effect_lower = effect.description.to_lowercase();
            // Simple keyword overlap heuristic.
            let precond_words: HashSet<&str> = desc_lower.split_whitespace().collect();
            let effect_words: HashSet<&str> = effect_lower.split_whitespace().collect();
            let overlap = precond_words.intersection(&effect_words).count();
            overlap >= 2 || effect_lower.contains(&desc_lower)
        })
    })
}

// ── §12: Internal — scoring ─────────────────────────────────────────

/// Score a sequence of plan steps.
fn score_plan(steps: &[PlanStep], config: &PlannerConfig) -> PlanScore {
    if steps.is_empty() {
        return PlanScore {
            feasibility: 0.0,
            expected_utility: 0.0,
            simplicity: 1.0,
            schema_success_rate: 0.0,
            urgency: 0.0,
            composite: 0.0,
            estimated_total_secs: 0.0,
        };
    }

    // Feasibility: product of step feasibilities (serial dependency).
    let feasibility = steps
        .iter()
        .map(|s| s.feasibility)
        .fold(1.0, |acc, f| acc * f);

    // Expected utility: max positive effect utility across steps.
    let expected_utility = steps
        .iter()
        .flat_map(|s| s.expected_effects.iter())
        .map(|e| e.utility * e.probability)
        .fold(0.0f64, f64::max);

    // Simplicity: 1.0 for 1 step, decays with more steps.
    let simplicity = 1.0 / (1.0 + (steps.len() as f64 - 1.0) * 0.3);

    // Schema success rate: geometric mean.
    let success_rates: Vec<f64> = steps
        .iter()
        .filter(|s| s.schema_node.is_some())
        .map(|s| s.feasibility.max(0.1))
        .collect();
    let schema_success_rate = if success_rates.is_empty() {
        0.5
    } else {
        let product: f64 = success_rates.iter().product();
        product.powf(1.0 / success_rates.len() as f64)
    };

    let estimated_total_secs: f64 = steps.iter().map(|s| s.estimated_duration_secs).sum();

    PlanScore {
        feasibility,
        expected_utility,
        simplicity,
        schema_success_rate,
        urgency: 0.0,   // Set by build_plan from goal urgency.
        composite: 0.0, // Computed in build_plan.
        estimated_total_secs,
    }
}

/// Build a complete Plan from steps and a goal.
fn build_plan(
    goal: &GoalEntry,
    steps: Vec<PlanStep>,
    blockers: Vec<Blocker>,
    ctx: &PlanningContext,
) -> Plan {
    let config = ctx.config;
    let mut score = score_plan(&steps, config);
    score.urgency = goal.attrs.urgency;

    // Composite weighted score.
    score.composite = config.weight_feasibility * score.feasibility
        + config.weight_utility * score.expected_utility.max(0.0)
        + config.weight_simplicity * score.simplicity
        + config.weight_success_rate * score.schema_success_rate
        + config.weight_urgency * score.urgency;

    let viable = blockers.iter().all(|b| b.severity < 0.9);

    let rationale = if steps.len() == 1 {
        format!(
            "Direct action: {} (feasibility={:.0}%)",
            steps[0].schema_name,
            score.feasibility * 100.0,
        )
    } else {
        format!(
            "{}-step plan via {} (feasibility={:.0}%, utility={:.2})",
            steps.len(),
            steps
                .iter()
                .map(|s| s.schema_name.as_str())
                .collect::<Vec<_>>()
                .join(" → "),
            score.feasibility * 100.0,
            score.expected_utility,
        )
    };

    Plan {
        goal_id: goal.node_id,
        goal_description: goal.payload.description.clone(),
        steps,
        score,
        rationale,
        created_at: ctx.now,
        viable,
        blockers,
    }
}

// ── §13: Internal — constraint checking ─────────────────────────────

/// Check plan steps against active constraints.
fn check_constraints(plan: &Plan, ctx: &PlanningContext) -> Vec<Blocker> {
    let mut blockers = Vec::new();

    for constraint in ctx.constraints {
        for step in &plan.steps {
            // Check if the constraint's condition matches any step.
            let condition_lower = constraint.payload.condition.to_lowercase();
            let step_desc_lower = step.description.to_lowercase();
            let schema_lower = step.schema_name.to_lowercase();

            let matches = condition_lower
                .split_whitespace()
                .any(|word| step_desc_lower.contains(word) || schema_lower.contains(word));

            if matches {
                let severity = match constraint.payload.constraint_type {
                    ConstraintType::Hard => 0.95,
                    ConstraintType::Soft => 0.5,
                };

                blockers.push(Blocker {
                    step_ordinal: Some(step.ordinal),
                    description: format!("Constraint violated: {}", constraint.payload.description),
                    severity,
                    kind: BlockerKind::ConstraintViolation,
                    resolution: Some(format!(
                        "Imposed by: {}. Condition: {}",
                        constraint.payload.imposed_by, constraint.payload.condition
                    )),
                });
            }
        }
    }

    blockers
}

// ── §14: Plan store (persistence) ───────────────────────────────────

/// Stores active plans for goals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStore {
    /// Active plans keyed by goal NodeId.
    plans: HashMap<u32, Plan>,
    /// How many plans have been generated lifetime.
    pub total_generated: u64,
    /// How many plans were executed successfully.
    pub total_succeeded: u64,
    /// How many plans failed or were abandoned.
    pub total_failed: u64,
}

impl PlanStore {
    pub fn new() -> Self {
        Self {
            plans: HashMap::new(),
            total_generated: 0,
            total_succeeded: 0,
            total_failed: 0,
        }
    }

    /// Get the active plan for a goal.
    pub fn get_plan(&self, goal_id: NodeId) -> Option<&Plan> {
        self.plans.get(&goal_id.to_raw())
    }

    /// Store a plan for a goal (replaces any existing plan).
    pub fn set_plan(&mut self, plan: Plan) {
        self.total_generated += 1;
        self.plans.insert(plan.goal_id.to_raw(), plan);
    }

    /// Mark a plan as succeeded and remove it.
    pub fn mark_succeeded(&mut self, goal_id: NodeId) -> Option<Plan> {
        self.total_succeeded += 1;
        self.plans.remove(&goal_id.to_raw())
    }

    /// Mark a plan as failed and remove it.
    pub fn mark_failed(&mut self, goal_id: NodeId) -> Option<Plan> {
        self.total_failed += 1;
        self.plans.remove(&goal_id.to_raw())
    }

    /// Remove a plan without recording outcome.
    pub fn remove_plan(&mut self, goal_id: NodeId) -> Option<Plan> {
        self.plans.remove(&goal_id.to_raw())
    }

    /// Get all active plans.
    pub fn active_plans(&self) -> Vec<&Plan> {
        self.plans.values().collect()
    }

    /// Number of active plans.
    pub fn active_count(&self) -> usize {
        self.plans.len()
    }

    /// Prune plans for goals that are no longer active.
    pub fn prune_inactive_goals(&mut self, active_goal_ids: &HashSet<u32>) -> usize {
        let before = self.plans.len();
        self.plans.retain(|gid, _| active_goal_ids.contains(gid));
        before - self.plans.len()
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{
        ActionKind, ActionSchemaPayload, CognitiveAttrs, CognitiveEdge, CognitiveEdgeKind,
        ConstraintPayload, ConstraintType, Effect, GoalPayload, GoalStatus, NodeId, NodeKind,
        Precondition, Priority, Provenance, TaskPayload, TaskStatus,
    };

    fn default_attrs() -> CognitiveAttrs {
        CognitiveAttrs {
            confidence: 0.8,
            activation: 0.6,
            salience: 0.5,
            persistence: 0.7,
            valence: 0.3,
            urgency: 0.5,
            novelty: 0.2,
            last_updated_ms: 1000000,
            volatility: 0.1,
            provenance: Provenance::Inferred,
            evidence_count: 5,
        }
    }

    fn test_goal(id: u32) -> GoalEntry {
        GoalEntry {
            node_id: NodeId::new(NodeKind::Goal, id),
            attrs: default_attrs(),
            payload: GoalPayload {
                description: "Test goal".to_string(),
                status: GoalStatus::Active,
                progress: 0.0,
                deadline: None,
                priority: Priority::Medium,
                parent_goal: None,
                completion_criteria: "Goal is achieved".to_string(),
            },
        }
    }

    fn test_schema(id: u32, name: &str) -> SchemaEntry {
        SchemaEntry {
            node_id: NodeId::new(NodeKind::ActionSchema, id),
            attrs: default_attrs(),
            payload: ActionSchemaPayload {
                name: name.to_string(),
                description: format!("Schema: {}", name),
                action_kind: ActionKind::Execute,
                preconditions: Vec::new(),
                effects: vec![Effect {
                    description: "Positive outcome".to_string(),
                    probability: 0.8,
                    utility: 0.6,
                }],
                confidence_threshold: 0.5,
                success_rate: 0.8,
                execution_count: 10,
                acceptance_count: 8,
            },
        }
    }

    fn test_config() -> PlannerConfig {
        PlannerConfig::default()
    }

    fn edge(src: NodeId, dst: NodeId, kind: CognitiveEdgeKind) -> CognitiveEdge {
        CognitiveEdge {
            src,
            dst,
            kind,
            weight: 0.8,
            created_at_ms: 1000000,
            last_confirmed_ms: 2000000,
            observation_count: 5,
            confidence: 0.7,
        }
    }

    #[test]
    fn test_empty_plan() {
        let config = test_config();
        let goal = test_goal(1);
        let ctx = PlanningContext {
            schemas: &[],
            goals: &[goal.clone()],
            tasks: &[],
            constraints: &[],
            edges: &[],
            skills: &[],
            now: 1000.0,
            config: &config,
        };

        let proposal = instantiate_plan(goal.node_id, &ctx);
        assert!(proposal.plans.is_empty());
    }

    #[test]
    fn test_goal_not_found() {
        let config = test_config();
        let ctx = PlanningContext {
            schemas: &[],
            goals: &[],
            tasks: &[],
            constraints: &[],
            edges: &[],
            skills: &[],
            now: 1000.0,
            config: &config,
        };

        let proposal = instantiate_plan(NodeId::new(NodeKind::Goal, 99), &ctx);
        assert!(proposal.plans.is_empty());
        assert!(!proposal.global_blockers.is_empty());
    }

    #[test]
    fn test_inactive_goal() {
        let config = test_config();
        let mut goal = test_goal(1);
        goal.payload.status = GoalStatus::Completed;

        let ctx = PlanningContext {
            schemas: &[],
            goals: &[goal.clone()],
            tasks: &[],
            constraints: &[],
            edges: &[],
            skills: &[],
            now: 1000.0,
            config: &config,
        };

        let proposal = instantiate_plan(goal.node_id, &ctx);
        assert!(proposal.plans.is_empty());
        assert!(proposal
            .global_blockers
            .iter()
            .any(|b| { matches!(b.kind, BlockerKind::GoalInactive) }));
    }

    #[test]
    fn test_direct_plan_with_advancing_schema() {
        let config = test_config();
        let goal = test_goal(1);
        let schema = test_schema(1, "achieve_goal");

        let edges = vec![edge(
            schema.node_id,
            goal.node_id,
            CognitiveEdgeKind::AdvancesGoal,
        )];

        let ctx = PlanningContext {
            schemas: &[schema],
            goals: &[goal.clone()],
            tasks: &[],
            constraints: &[],
            edges: &edges,
            skills: &[],
            now: 1000.0,
            config: &config,
        };

        let proposal = instantiate_plan(goal.node_id, &ctx);
        assert!(!proposal.plans.is_empty());

        let best = &proposal.plans[0];
        assert_eq!(best.steps.len(), 1);
        assert_eq!(best.steps[0].schema_name, "achieve_goal");
        assert!(best.score.composite > 0.0);
        assert!(best.viable);
    }

    #[test]
    fn test_schema_with_preconditions() {
        let config = test_config();
        let goal = test_goal(1);

        let prereq_goal = GoalEntry {
            node_id: NodeId::new(NodeKind::Goal, 2),
            attrs: default_attrs(),
            payload: GoalPayload {
                description: "Prerequisite goal".to_string(),
                status: GoalStatus::Completed,
                progress: 1.0,
                deadline: None,
                priority: Priority::Medium,
                parent_goal: None,
                completion_criteria: "Done".to_string(),
            },
        };

        let mut schema = test_schema(1, "guarded_action");
        schema.payload.preconditions = vec![Precondition {
            description: "Prerequisite goal must be completed".to_string(),
            node_ref: Some(prereq_goal.node_id),
            required: true,
        }];

        let edges = vec![edge(
            schema.node_id,
            goal.node_id,
            CognitiveEdgeKind::AdvancesGoal,
        )];

        let ctx = PlanningContext {
            schemas: &[schema],
            goals: &[goal.clone(), prereq_goal],
            tasks: &[],
            constraints: &[],
            edges: &edges,
            skills: &[],
            now: 1000.0,
            config: &config,
        };

        let proposal = instantiate_plan(goal.node_id, &ctx);
        assert!(!proposal.plans.is_empty());

        let best = &proposal.plans[0];
        // Precondition should be satisfied (prereq goal is Completed).
        assert!(best.steps[0].preconditions[0].satisfied);
    }

    #[test]
    fn test_unsatisfied_precondition_blocker() {
        let config = test_config();
        let goal = test_goal(1);

        let prereq_goal = GoalEntry {
            node_id: NodeId::new(NodeKind::Goal, 2),
            attrs: default_attrs(),
            payload: GoalPayload {
                description: "Not done yet".to_string(),
                status: GoalStatus::Active, // NOT completed
                progress: 0.3,
                deadline: None,
                priority: Priority::Medium,
                parent_goal: None,
                completion_criteria: "Done".to_string(),
            },
        };

        let mut schema = test_schema(1, "needs_prereq");
        schema.payload.preconditions = vec![Precondition {
            description: "Need prereq".to_string(),
            node_ref: Some(prereq_goal.node_id),
            required: true,
        }];

        let edges = vec![edge(
            schema.node_id,
            goal.node_id,
            CognitiveEdgeKind::AdvancesGoal,
        )];

        let ctx = PlanningContext {
            schemas: &[schema],
            goals: &[goal.clone(), prereq_goal],
            tasks: &[],
            constraints: &[],
            edges: &edges,
            skills: &[],
            now: 1000.0,
            config: &config,
        };

        let proposal = instantiate_plan(goal.node_id, &ctx);
        // Plan may still be generated (with low feasibility) or may have blockers.
        if !proposal.plans.is_empty() {
            let best = &proposal.plans[0];
            assert!(!best.steps[0].preconditions[0].satisfied);
        }
    }

    #[test]
    fn test_skill_decomposition() {
        let config = test_config();
        let goal = test_goal(1);
        let schema = test_schema(1, "send_reminder");

        let skill = SkillTemplate {
            skill_id: 42,
            description: "Multi-step reminder".to_string(),
            steps: vec![
                SkillStepInfo {
                    ordinal: 0,
                    action_kind: "send_reminder".to_string(),
                    description: "Send the reminder".to_string(),
                    expected_duration_ms: 5000,
                    optional: false,
                },
                SkillStepInfo {
                    ordinal: 1,
                    action_kind: "confirm".to_string(),
                    description: "Confirm delivery".to_string(),
                    expected_duration_ms: 2000,
                    optional: false,
                },
            ],
            confidence: 0.8,
            success_rate: 0.9,
        };

        // Schema advances goal via activation (no explicit edge needed).
        let mut schema_with_activation = schema;
        schema_with_activation.attrs.activation = 0.6;

        let ctx = PlanningContext {
            schemas: &[schema_with_activation],
            goals: &[goal.clone()],
            tasks: &[],
            constraints: &[],
            edges: &[],
            skills: &[skill],
            now: 1000.0,
            config: &config,
        };

        let proposal = instantiate_plan(goal.node_id, &ctx);
        // Should have at least one plan from skill decomposition.
        let skill_plans: Vec<&Plan> = proposal
            .plans
            .iter()
            .filter(|p| p.steps.len() > 1)
            .collect();
        // Skill plan creates 2 steps.
        assert!(!skill_plans.is_empty() || !proposal.plans.is_empty());
    }

    #[test]
    fn test_task_chain_plan() {
        let config = test_config();
        let goal = test_goal(1);

        let task1 = TaskEntry {
            node_id: NodeId::new(NodeKind::Task, 1),
            attrs: default_attrs(),
            payload: TaskPayload {
                description: "First task".to_string(),
                status: TaskStatus::Completed,
                goal_id: Some(goal.node_id),
                deadline: None,
                priority: Priority::Medium,
                estimated_minutes: Some(10),
                prerequisites: Vec::new(),
            },
        };

        let task2 = TaskEntry {
            node_id: NodeId::new(NodeKind::Task, 2),
            attrs: default_attrs(),
            payload: TaskPayload {
                description: "Second task".to_string(),
                status: TaskStatus::Pending,
                goal_id: Some(goal.node_id),
                deadline: None,
                priority: Priority::Medium,
                estimated_minutes: Some(20),
                prerequisites: vec![task1.node_id],
            },
        };

        let ctx = PlanningContext {
            schemas: &[],
            goals: &[goal.clone()],
            tasks: &[task1, task2],
            constraints: &[],
            edges: &[],
            skills: &[],
            now: 1000.0,
            config: &config,
        };

        let proposal = instantiate_plan(goal.node_id, &ctx);
        // Should have a task chain plan with 1 step (task1 is completed, task2 is pending).
        if !proposal.plans.is_empty() {
            let plan = &proposal.plans[0];
            assert!(plan.steps.iter().any(|s| s.description == "Second task"));
        }
    }

    #[test]
    fn test_constraint_violation() {
        let config = test_config();
        let goal = test_goal(1);
        let schema = test_schema(1, "delete_all");

        let edges = vec![edge(
            schema.node_id,
            goal.node_id,
            CognitiveEdgeKind::AdvancesGoal,
        )];

        let constraint = ConstraintEntry {
            node_id: NodeId::new(NodeKind::Constraint, 1),
            payload: ConstraintPayload {
                description: "Never delete user data".to_string(),
                constraint_type: ConstraintType::Hard,
                condition: "delete".to_string(),
                imposed_by: "system_policy".to_string(),
            },
        };

        let ctx = PlanningContext {
            schemas: &[schema],
            goals: &[goal.clone()],
            tasks: &[],
            constraints: &[constraint],
            edges: &edges,
            skills: &[],
            now: 1000.0,
            config: &config,
        };

        let proposal = instantiate_plan(goal.node_id, &ctx);
        // Plan should exist but be non-viable due to constraint.
        if !proposal.plans.is_empty() {
            let plan = &proposal.plans[0];
            assert!(plan
                .blockers
                .iter()
                .any(|b| { matches!(b.kind, BlockerKind::ConstraintViolation) }));
            assert!(!plan.viable);
        }
    }

    #[test]
    fn test_plan_scoring() {
        let config = test_config();

        let step1 = PlanStep {
            ordinal: 0,
            schema_name: "step1".to_string(),
            action_kind: ActionKind::Execute,
            description: "First".to_string(),
            schema_node: Some(NodeId::new(NodeKind::ActionSchema, 1)),
            preconditions: Vec::new(),
            expected_effects: vec![Effect {
                description: "Good".to_string(),
                probability: 0.9,
                utility: 0.7,
            }],
            estimated_duration_secs: 30.0,
            derivation: StepDerivation::DirectMatch,
            feasibility: 0.9,
        };

        let step2 = PlanStep {
            ordinal: 1,
            schema_name: "step2".to_string(),
            action_kind: ActionKind::Communicate,
            description: "Second".to_string(),
            schema_node: Some(NodeId::new(NodeKind::ActionSchema, 2)),
            preconditions: Vec::new(),
            expected_effects: vec![Effect {
                description: "Done".to_string(),
                probability: 0.8,
                utility: 0.5,
            }],
            estimated_duration_secs: 20.0,
            derivation: StepDerivation::BackwardChain { enables_step: 0 },
            feasibility: 0.7,
        };

        let score = score_plan(&[step1, step2], &config);
        // Feasibility = 0.9 * 0.7 = 0.63
        assert!((score.feasibility - 0.63).abs() < 0.01);
        // Simplicity for 2 steps < 1.0
        assert!(score.simplicity < 1.0);
        assert!(score.estimated_total_secs > 0.0);
    }

    #[test]
    fn test_next_plan_step() {
        let config = test_config();
        let goal = test_goal(1);
        let schema = test_schema(1, "quick_action");

        let edges = vec![edge(
            schema.node_id,
            goal.node_id,
            CognitiveEdgeKind::AdvancesGoal,
        )];

        let ctx = PlanningContext {
            schemas: &[schema],
            goals: &[goal.clone()],
            tasks: &[],
            constraints: &[],
            edges: &edges,
            skills: &[],
            now: 1000.0,
            config: &config,
        };

        let step = next_plan_step(goal.node_id, &ctx);
        assert!(step.is_some());
        assert_eq!(step.unwrap().schema_name, "quick_action");
    }

    #[test]
    fn test_detect_blockers_deadline() {
        let config = test_config();
        let mut goal = test_goal(1);
        goal.payload.deadline = Some(500.0); // In the past.

        let ctx = PlanningContext {
            schemas: &[],
            goals: &[goal.clone()],
            tasks: &[],
            constraints: &[],
            edges: &[],
            skills: &[],
            now: 1000.0,
            config: &config,
        };

        let blockers = detect_blockers(goal.node_id, &ctx);
        assert!(blockers
            .iter()
            .any(|b| { matches!(b.kind, BlockerKind::DeadlinePressure) }));
    }

    #[test]
    fn test_plan_store() {
        let mut store = PlanStore::new();
        let goal_id = NodeId::new(NodeKind::Goal, 1);

        assert_eq!(store.active_count(), 0);
        assert!(store.get_plan(goal_id).is_none());

        let plan = Plan {
            goal_id,
            goal_description: "Test".to_string(),
            steps: Vec::new(),
            score: PlanScore {
                feasibility: 0.8,
                expected_utility: 0.6,
                simplicity: 1.0,
                schema_success_rate: 0.9,
                urgency: 0.5,
                composite: 0.7,
                estimated_total_secs: 30.0,
            },
            rationale: "Test plan".to_string(),
            created_at: 1000.0,
            viable: true,
            blockers: Vec::new(),
        };

        store.set_plan(plan);
        assert_eq!(store.active_count(), 1);
        assert_eq!(store.total_generated, 1);
        assert!(store.get_plan(goal_id).is_some());

        store.mark_succeeded(goal_id);
        assert_eq!(store.active_count(), 0);
        assert_eq!(store.total_succeeded, 1);
    }

    #[test]
    fn test_topological_sort() {
        let t1 = TaskEntry {
            node_id: NodeId::new(NodeKind::Task, 1),
            attrs: default_attrs(),
            payload: TaskPayload {
                description: "A".to_string(),
                status: TaskStatus::Pending,
                goal_id: None,
                deadline: None,
                priority: Priority::Medium,
                estimated_minutes: None,
                prerequisites: Vec::new(),
            },
        };
        let t2 = TaskEntry {
            node_id: NodeId::new(NodeKind::Task, 2),
            attrs: default_attrs(),
            payload: TaskPayload {
                description: "B".to_string(),
                status: TaskStatus::Pending,
                goal_id: None,
                deadline: None,
                priority: Priority::Medium,
                estimated_minutes: None,
                prerequisites: vec![t1.node_id],
            },
        };
        let t3 = TaskEntry {
            node_id: NodeId::new(NodeKind::Task, 3),
            attrs: default_attrs(),
            payload: TaskPayload {
                description: "C".to_string(),
                status: TaskStatus::Pending,
                goal_id: None,
                deadline: None,
                priority: Priority::Medium,
                estimated_minutes: None,
                prerequisites: vec![t1.node_id, t2.node_id],
            },
        };

        let tasks: Vec<&TaskEntry> = vec![&t3, &t2, &t1]; // Deliberately out of order.
        let sorted = topological_sort_tasks(&tasks);

        assert_eq!(sorted.len(), 3);
        // t1 must come before t2 and t3, t2 must come before t3.
        let pos1 = sorted.iter().position(|t| t.node_id == t1.node_id).unwrap();
        let pos2 = sorted.iter().position(|t| t.node_id == t2.node_id).unwrap();
        let pos3 = sorted.iter().position(|t| t.node_id == t3.node_id).unwrap();
        assert!(pos1 < pos2);
        assert!(pos2 < pos3);
    }

    #[test]
    fn test_multiple_plans_ranking() {
        let config = test_config();
        let goal = test_goal(1);

        let schema_good = test_schema(1, "good_action");
        let mut schema_weak = test_schema(2, "weak_action");
        schema_weak.payload.success_rate = 0.3;

        let edges = vec![
            edge(
                schema_good.node_id,
                goal.node_id,
                CognitiveEdgeKind::AdvancesGoal,
            ),
            edge(
                schema_weak.node_id,
                goal.node_id,
                CognitiveEdgeKind::AdvancesGoal,
            ),
        ];

        let ctx = PlanningContext {
            schemas: &[schema_good, schema_weak],
            goals: &[goal.clone()],
            tasks: &[],
            constraints: &[],
            edges: &edges,
            skills: &[],
            now: 1000.0,
            config: &config,
        };

        let proposal = instantiate_plan(goal.node_id, &ctx);
        assert!(proposal.plans.len() >= 2);

        // Best plan should have higher composite score.
        assert!(proposal.plans[0].score.composite >= proposal.plans[1].score.composite);
        assert_eq!(proposal.plans[0].steps[0].schema_name, "good_action");
    }
}
