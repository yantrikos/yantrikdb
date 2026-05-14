//! CK-4.6 — Cognitive Query DSL: Composable Reasoning Operators.
//!
//! A builder-pattern API for constructing custom reasoning pipelines
//! from cognitive primitives. Each operator transforms the cognitive
//! state, and operators can be freely chained.
//!
//! # Design principles
//! - Pure data structures — pipeline definition is DB-free
//! - Lazy by default — operators are recorded, then executed by engine
//! - Composable — each operator returns the pipeline for chaining
//! - Explainable — every step produces a reasoning trace entry
//! - Budgeted — execution can be time-bounded

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::state::NodeId;

// ── §1: Pipeline Definition ───────────────────────────────────────

/// A composable reasoning pipeline built from cognitive operators.
///
/// Constructed via the builder pattern, then executed against the
/// database by the engine layer.
#[derive(Debug, Clone)]
pub struct CognitivePipeline {
    /// Ordered list of operators to execute.
    pub(crate) operators: Vec<CognitiveOperator>,
    /// Optional time budget for execution.
    pub(crate) budget: Option<Duration>,
    /// Whether to generate an explanation trace.
    pub(crate) explain: bool,
    /// Execution mode.
    pub(crate) mode: ExecutionMode,
    /// Optional context for the pipeline.
    pub(crate) context: Option<PipelineContext>,
}

impl CognitivePipeline {
    /// Create a new empty pipeline.
    pub fn new() -> Self {
        Self {
            operators: Vec::new(),
            budget: None,
            explain: false,
            mode: ExecutionMode::Lazy,
            context: None,
        }
    }

    // ── Builder methods ──

    /// Set context for the pipeline (e.g., user turn, event trigger).
    pub fn with_context(mut self, ctx: PipelineContext) -> Self {
        self.context = Some(ctx);
        self
    }

    /// Set a time budget. Operators are skipped if budget is exhausted.
    pub fn with_budget(mut self, budget: Duration) -> Self {
        self.budget = Some(budget);
        self.mode = ExecutionMode::Budgeted;
        self
    }

    /// Set execution mode.
    pub fn with_mode(mut self, mode: ExecutionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Spread activation from seed nodes, narrowing focus.
    pub fn attend(mut self, seeds: Vec<NodeId>) -> Self {
        self.operators.push(CognitiveOperator::Attend(AttendOp {
            seeds,
            max_hops: 2,
            decay: 0.5,
        }));
        self
    }

    /// Spread activation with custom parameters.
    pub fn attend_with(mut self, seeds: Vec<NodeId>, max_hops: u32, decay: f64) -> Self {
        self.operators.push(CognitiveOperator::Attend(AttendOp {
            seeds,
            max_hops,
            decay,
        }));
        self
    }

    /// Retrieve relevant memories into the working context.
    pub fn recall(mut self, top_k: usize) -> Self {
        self.operators.push(CognitiveOperator::Recall(RecallOp {
            top_k,
            query: None,
            domain: None,
        }));
        self
    }

    /// Retrieve with a text query and optional domain filter.
    pub fn recall_query(mut self, query: String, top_k: usize, domain: Option<String>) -> Self {
        self.operators.push(CognitiveOperator::Recall(RecallOp {
            top_k,
            query: Some(query),
            domain,
        }));
        self
    }

    /// Revise beliefs based on new evidence.
    pub fn believe(mut self, evidence: EvidenceInput) -> Self {
        self.operators
            .push(CognitiveOperator::Believe(BelieveOp { evidence }));
        self
    }

    /// Forward-simulate consequences of the current state.
    pub fn project(mut self, horizon: ProjectionHorizon) -> Self {
        self.operators.push(CognitiveOperator::Project(ProjectOp {
            horizon,
            include_causal: true,
        }));
        self
    }

    /// Evaluate and rank candidate actions.
    pub fn compare(mut self, candidates: Vec<CandidateAction>) -> Self {
        self.operators.push(CognitiveOperator::Compare(CompareOp {
            candidates,
            apply_personality: true,
        }));
        self
    }

    /// Filter candidates by policy constraints.
    pub fn constrain(mut self, policies: Vec<PolicyConstraint>) -> Self {
        self.operators
            .push(CognitiveOperator::Constrain(ConstrainOp { policies }));
        self
    }

    /// Check anticipatory items within a timeframe.
    pub fn anticipate(mut self, horizon_secs: f64) -> Self {
        self.operators
            .push(CognitiveOperator::Anticipate(AnticipateOp { horizon_secs }));
        self
    }

    /// Plan toward a goal using means-ends reasoning.
    pub fn plan(mut self, goal_id: NodeId, max_depth: u32) -> Self {
        self.operators
            .push(CognitiveOperator::Plan(PlanOp { goal_id, max_depth }));
        self
    }

    /// Run a meta-cognitive assessment.
    pub fn assess(mut self) -> Self {
        self.operators.push(CognitiveOperator::Assess);
        self
    }

    /// Check coherence of the current cognitive state.
    pub fn coherence_check(mut self) -> Self {
        self.operators.push(CognitiveOperator::CoherenceCheck);
        self
    }

    /// Enable explanation trace generation.
    pub fn explain(mut self) -> Self {
        self.explain = true;
        self
    }

    /// Number of operators in the pipeline.
    pub fn len(&self) -> usize {
        self.operators.len()
    }

    /// Whether the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.operators.is_empty()
    }
}

impl Default for CognitivePipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ── §2: Operators ──────────────────────────────────────────────────

/// A single cognitive operator in the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CognitiveOperator {
    /// Spread activation from seed nodes.
    Attend(AttendOp),
    /// Retrieve relevant memories.
    Recall(RecallOp),
    /// Revise beliefs with evidence.
    Believe(BelieveOp),
    /// Forward-simulate consequences.
    Project(ProjectOp),
    /// Evaluate candidate actions.
    Compare(CompareOp),
    /// Apply policy constraints.
    Constrain(ConstrainOp),
    /// Check anticipatory items.
    Anticipate(AnticipateOp),
    /// Plan toward a goal.
    Plan(PlanOp),
    /// Meta-cognitive assessment.
    Assess,
    /// Coherence check.
    CoherenceCheck,
}

impl CognitiveOperator {
    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Attend(_) => "attend",
            Self::Recall(_) => "recall",
            Self::Believe(_) => "believe",
            Self::Project(_) => "project",
            Self::Compare(_) => "compare",
            Self::Constrain(_) => "constrain",
            Self::Anticipate(_) => "anticipate",
            Self::Plan(_) => "plan",
            Self::Assess => "assess",
            Self::CoherenceCheck => "coherence_check",
        }
    }

    /// Priority for budget-based execution (higher = more important).
    pub fn priority(&self) -> u8 {
        match self {
            Self::Attend(_) => 10,     // Foundation — always run.
            Self::Recall(_) => 9,      // Critical for context.
            Self::Believe(_) => 8,     // Evidence integration.
            Self::Compare(_) => 7,     // Action selection.
            Self::Constrain(_) => 7,   // Safety — always run if comparing.
            Self::Plan(_) => 6,        // Means-ends reasoning.
            Self::Project(_) => 5,     // Forward simulation.
            Self::Anticipate(_) => 4,  // Proactive — nice to have.
            Self::Assess => 3,         // Meta — can skip under pressure.
            Self::CoherenceCheck => 2, // Maintenance — skip if budget tight.
        }
    }
}

/// Attend operator parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttendOp {
    pub seeds: Vec<NodeId>,
    pub max_hops: u32,
    pub decay: f64,
}

/// Recall operator parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallOp {
    pub top_k: usize,
    pub query: Option<String>,
    pub domain: Option<String>,
}

/// Believe operator parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BelieveOp {
    pub evidence: EvidenceInput,
}

/// New evidence to integrate into beliefs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceInput {
    /// Target belief node (or creates new if absent).
    pub target: Option<NodeId>,
    /// Observation that either confirms or contradicts.
    pub observation: String,
    /// Direction: positive = confirming, negative = contradicting.
    pub direction: f64,
    /// Strength of this evidence ∈ [0.0, 1.0].
    pub strength: f64,
    /// Source of the evidence.
    pub source: String,
}

/// Project operator parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectOp {
    pub horizon: ProjectionHorizon,
    pub include_causal: bool,
}

/// How far forward to project.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ProjectionHorizon {
    /// Immediate consequences only.
    OneStep,
    /// Short-term (minutes).
    ShortTerm,
    /// Medium-term (hours).
    MediumTerm,
    /// Long-term (days).
    LongTerm,
}

impl ProjectionHorizon {
    /// Approximate seconds for this horizon.
    pub fn seconds(&self) -> f64 {
        match self {
            Self::OneStep => 60.0,
            Self::ShortTerm => 600.0,   // 10 minutes
            Self::MediumTerm => 7200.0, // 2 hours
            Self::LongTerm => 86400.0,  // 1 day
        }
    }
}

/// Compare operator parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareOp {
    pub candidates: Vec<CandidateAction>,
    pub apply_personality: bool,
}

/// A candidate action to evaluate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateAction {
    pub description: String,
    pub action_kind: String,
    pub confidence: f64,
    /// Properties for personality biasing.
    pub properties: crate::personality_bias::ActionProperties,
}

/// Constrain operator parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstrainOp {
    pub policies: Vec<PolicyConstraint>,
}

/// A policy constraint applied to filter candidates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConstraint {
    pub name: String,
    pub kind: ConstraintKind,
}

/// Types of policy constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintKind {
    /// Minimum confidence to pass.
    MinConfidence(f64),
    /// Maximum risk tolerance.
    MaxRisk(f64),
    /// Require meta-cognitive confidence above threshold.
    MetaCogThreshold(f64),
    /// Require user consent for this action type.
    RequireConsent(String),
    /// Block specific action kinds.
    BlockActionKind(String),
    /// Time-of-day restriction (quiet hours).
    QuietHours { start_hour: u8, end_hour: u8 },
}

/// Anticipate operator parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnticipateOp {
    pub horizon_secs: f64,
}

/// Plan operator parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanOp {
    pub goal_id: NodeId,
    pub max_depth: u32,
}

// ── §3: Execution Mode ─────────────────────────────────────────────

/// How the pipeline is executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Operators are recorded then executed all at once.
    Lazy,
    /// Execute within a time budget, skipping low-priority operators.
    Budgeted,
}

// ── §4: Pipeline Context ───────────────────────────────────────────

/// Context provided to the pipeline at construction time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineContext {
    /// Description of what triggered this reasoning.
    pub trigger: String,
    /// User message text (if applicable).
    pub user_input: Option<String>,
    /// Mentioned entities from the user's message.
    pub mentioned_entities: Vec<NodeId>,
    /// Current timestamp.
    pub timestamp: f64,
}

// ── §5: Pipeline Result ────────────────────────────────────────────

/// Result of executing a cognitive pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    /// Per-operator results in execution order.
    pub steps: Vec<StepResult>,
    /// How many operators were executed (may be less than pipeline length if budgeted).
    pub operators_executed: usize,
    /// How many operators were skipped (due to budget).
    pub operators_skipped: usize,
    /// Total wall-clock execution time.
    pub elapsed_ms: u64,
    /// Whether the budget was exhausted.
    pub budget_exhausted: bool,
    /// Explanation trace (if `.explain()` was called).
    pub explanation: Option<ExplanationTrace>,
    /// Overall pipeline status.
    pub status: PipelineStatus,
}

/// Status of the pipeline execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineStatus {
    /// All operators completed successfully.
    Complete,
    /// Some operators were skipped due to budget.
    Partial,
    /// Pipeline failed at a specific step.
    Failed,
    /// Pipeline was empty (no operators).
    Empty,
}

/// Result of a single operator execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    /// Operator name.
    pub operator: String,
    /// Whether this step succeeded.
    pub success: bool,
    /// Execution time for this step.
    pub elapsed_ms: u64,
    /// Operator-specific output.
    pub output: StepOutput,
    /// Explanation for this step (if explain mode).
    pub trace: Option<String>,
}

/// Operator-specific output variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepOutput {
    /// Attend: nodes activated, activation deltas.
    Attend {
        nodes_activated: usize,
        top_activated: Vec<(NodeId, f64)>,
    },
    /// Recall: memories retrieved.
    Recall {
        memories_retrieved: usize,
        top_matches: Vec<RecallMatch>,
    },
    /// Believe: beliefs updated.
    Believe {
        beliefs_updated: usize,
        confidence_delta: f64,
    },
    /// Project: predicted consequences.
    Project { predictions: Vec<Prediction> },
    /// Compare: ranked candidates.
    Compare { ranked: Vec<RankedCandidate> },
    /// Constrain: filtered candidates.
    Constrain {
        passed: usize,
        filtered_out: usize,
        violations: Vec<ConstraintViolation>,
    },
    /// Anticipate: upcoming items.
    Anticipate { items: Vec<AnticipatedItem> },
    /// Plan: generated plan.
    Plan {
        plan_found: bool,
        steps: usize,
        score: f64,
    },
    /// Assess: meta-cognitive result.
    Assess {
        overall_confidence: f64,
        coverage_gaps: usize,
    },
    /// Coherence: health check.
    Coherence {
        score: f64,
        conflicts: usize,
        stale_nodes: usize,
    },
    /// Operator was skipped (budget).
    Skipped { reason: String },
    /// Operator failed.
    Error { message: String },
}

/// A recalled memory match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallMatch {
    pub text: String,
    pub similarity: f64,
    pub memory_type: String,
}

/// A predicted consequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub description: String,
    pub probability: f64,
    pub valence: f64,
}

/// A ranked candidate action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedCandidate {
    pub description: String,
    pub score: f64,
    pub personality_bias: f64,
    pub rank: usize,
}

/// A constraint violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    pub constraint_name: String,
    pub candidate: String,
    pub reason: String,
}

/// An anticipated item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnticipatedItem {
    pub description: String,
    pub expected_at: f64,
    pub confidence: f64,
}

// ── §6: Explanation Trace ──────────────────────────────────────────

/// A human-readable trace of the reasoning process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationTrace {
    /// Step-by-step reasoning narrative.
    pub steps: Vec<ExplanationStep>,
    /// Overall reasoning summary.
    pub summary: String,
}

/// A single step in the explanation trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationStep {
    pub step_number: usize,
    pub operator: String,
    pub description: String,
    pub key_insight: Option<String>,
}

// ── §7: Pipeline Execution (Pure Logic) ────────────────────────────

/// Execute a pipeline step and produce a StepResult.
///
/// This function is the dispatcher — it matches the operator type
/// and calls the appropriate handler. The engine layer provides
/// implementations via the `PipelineExecutor` trait.
pub fn execute_pipeline(
    pipeline: &CognitivePipeline,
    executor: &dyn PipelineExecutor,
) -> PipelineResult {
    if pipeline.is_empty() {
        return PipelineResult {
            steps: Vec::new(),
            operators_executed: 0,
            operators_skipped: 0,
            elapsed_ms: 0,
            budget_exhausted: false,
            explanation: None,
            status: PipelineStatus::Empty,
        };
    }

    let start = Instant::now();
    let mut steps = Vec::with_capacity(pipeline.operators.len());
    let mut executed = 0usize;
    let mut skipped = 0usize;
    let mut budget_exhausted = false;
    let mut any_failed = false;

    // Sort operators by priority if budgeted.
    let execution_order: Vec<usize> = if pipeline.mode == ExecutionMode::Budgeted {
        let mut indices: Vec<usize> = (0..pipeline.operators.len()).collect();
        indices.sort_by(|a, b| {
            pipeline.operators[*b]
                .priority()
                .cmp(&pipeline.operators[*a].priority())
        });
        indices
    } else {
        (0..pipeline.operators.len()).collect()
    };

    for &idx in &execution_order {
        // Check budget.
        if let Some(budget) = pipeline.budget {
            if start.elapsed() >= budget {
                budget_exhausted = true;
                skipped += execution_order.len() - executed;
                // Record remaining as skipped.
                for &remaining_idx in &execution_order[executed..] {
                    let _ = remaining_idx; // Already counted in skipped.
                }
                break;
            }
        }

        let op = &pipeline.operators[idx];
        let step_start = Instant::now();

        let output = executor.execute_operator(op, &pipeline.context);
        let step_elapsed = step_start.elapsed().as_millis() as u64;

        let success = !matches!(output, StepOutput::Error { .. });
        if !success {
            any_failed = true;
        }

        let trace = if pipeline.explain {
            Some(format_step_trace(op, &output))
        } else {
            None
        };

        steps.push(StepResult {
            operator: op.name().to_string(),
            success,
            elapsed_ms: step_elapsed,
            output,
            trace,
        });
        executed += 1;
    }

    let total_elapsed = start.elapsed().as_millis() as u64;

    let status = if any_failed {
        PipelineStatus::Failed
    } else if budget_exhausted {
        PipelineStatus::Partial
    } else {
        PipelineStatus::Complete
    };

    // Build explanation trace if requested.
    let explanation = if pipeline.explain {
        Some(build_explanation(&steps))
    } else {
        None
    };

    PipelineResult {
        steps,
        operators_executed: executed,
        operators_skipped: skipped,
        elapsed_ms: total_elapsed,
        budget_exhausted,
        explanation,
        status,
    }
}

// ── §8: Pipeline Executor Trait ────────────────────────────────────

/// Trait implemented by the engine layer to execute individual operators.
///
/// The cognition layer defines what the operators mean; the engine
/// layer provides the implementation that accesses the database.
pub trait PipelineExecutor {
    /// Execute a single cognitive operator.
    fn execute_operator(
        &self,
        operator: &CognitiveOperator,
        context: &Option<PipelineContext>,
    ) -> StepOutput;
}

// ── §9: Helpers ────────────────────────────────────────────────────

fn format_step_trace(op: &CognitiveOperator, output: &StepOutput) -> String {
    match (op, output) {
        (
            CognitiveOperator::Attend(a),
            StepOutput::Attend {
                nodes_activated, ..
            },
        ) => {
            format!(
                "Spread activation from {} seeds → {} nodes activated",
                a.seeds.len(),
                nodes_activated
            )
        }
        (
            CognitiveOperator::Recall(r),
            StepOutput::Recall {
                memories_retrieved, ..
            },
        ) => {
            format!(
                "Retrieved {} memories (top_k={})",
                memories_retrieved, r.top_k
            )
        }
        (
            CognitiveOperator::Believe(_),
            StepOutput::Believe {
                beliefs_updated,
                confidence_delta,
                ..
            },
        ) => {
            format!(
                "Updated {} beliefs (Δconf={:+.3})",
                beliefs_updated, confidence_delta
            )
        }
        (CognitiveOperator::Project(p), StepOutput::Project { predictions, .. }) => {
            format!(
                "Projected {:?} → {} predictions",
                p.horizon,
                predictions.len()
            )
        }
        (CognitiveOperator::Compare(_), StepOutput::Compare { ranked, .. }) => {
            format!(
                "Compared {} candidates → top: {}",
                ranked.len(),
                ranked
                    .first()
                    .map(|r| r.description.as_str())
                    .unwrap_or("none")
            )
        }
        (
            CognitiveOperator::Constrain(_),
            StepOutput::Constrain {
                passed,
                filtered_out,
                ..
            },
        ) => {
            format!(
                "{} passed, {} filtered by constraints",
                passed, filtered_out
            )
        }
        (
            CognitiveOperator::Plan(_),
            StepOutput::Plan {
                plan_found,
                steps,
                score,
                ..
            },
        ) => {
            if *plan_found {
                format!("Plan found: {} steps, score {:.2}", steps, score)
            } else {
                "No viable plan found".to_string()
            }
        }
        (
            CognitiveOperator::Assess,
            StepOutput::Assess {
                overall_confidence,
                coverage_gaps,
                ..
            },
        ) => {
            format!(
                "Meta-cognitive confidence: {:.2}, {} coverage gaps",
                overall_confidence, coverage_gaps
            )
        }
        (
            CognitiveOperator::CoherenceCheck,
            StepOutput::Coherence {
                score,
                conflicts,
                stale_nodes,
                ..
            },
        ) => {
            format!(
                "Coherence: {:.2}, {} conflicts, {} stale",
                score, conflicts, stale_nodes
            )
        }
        (_, StepOutput::Skipped { reason }) => {
            format!("Skipped: {}", reason)
        }
        (_, StepOutput::Error { message }) => {
            format!("Error: {}", message)
        }
        _ => format!("{}: completed", op.name()),
    }
}

fn build_explanation(steps: &[StepResult]) -> ExplanationTrace {
    let explanation_steps: Vec<ExplanationStep> = steps
        .iter()
        .enumerate()
        .filter(|(_, s)| s.success)
        .map(|(i, s)| ExplanationStep {
            step_number: i + 1,
            operator: s.operator.clone(),
            description: s
                .trace
                .clone()
                .unwrap_or_else(|| format!("{}: completed", s.operator)),
            key_insight: extract_key_insight(&s.output),
        })
        .collect();

    let summary = if explanation_steps.is_empty() {
        "No reasoning steps executed.".to_string()
    } else {
        let op_names: Vec<&str> = explanation_steps
            .iter()
            .map(|s| s.operator.as_str())
            .collect();
        format!(
            "Reasoning pipeline: {} steps ({})",
            explanation_steps.len(),
            op_names.join(" → ")
        )
    };

    ExplanationTrace {
        steps: explanation_steps,
        summary,
    }
}

fn extract_key_insight(output: &StepOutput) -> Option<String> {
    match output {
        StepOutput::Compare { ranked, .. } if !ranked.is_empty() => Some(format!(
            "Best candidate: {} (score {:.2})",
            ranked[0].description, ranked[0].score
        )),
        StepOutput::Assess {
            overall_confidence, ..
        } if *overall_confidence < 0.5 => {
            Some("Low meta-cognitive confidence — consider escalating".to_string())
        }
        StepOutput::Coherence { score, .. } if *score < 0.5 => {
            Some("Coherence degraded — enforcement may be needed".to_string())
        }
        StepOutput::Constrain { filtered_out, .. } if *filtered_out > 0 => Some(format!(
            "{} candidates filtered by policy constraints",
            filtered_out
        )),
        _ => None,
    }
}

// ── §10: Common Pipeline Patterns ──────────────────────────────────

/// Pre-built pipeline patterns for common reasoning scenarios.
pub struct PipelinePatterns;

impl PipelinePatterns {
    /// Standard reasoning for a user turn.
    /// attend → recall → believe → compare → constrain → explain
    pub fn user_turn(seeds: Vec<NodeId>, candidates: Vec<CandidateAction>) -> CognitivePipeline {
        CognitivePipeline::new()
            .attend(seeds)
            .recall(5)
            .compare(candidates)
            .constrain(vec![PolicyConstraint {
                name: "min_confidence".to_string(),
                kind: ConstraintKind::MinConfidence(0.3),
            }])
            .explain()
    }

    /// Proactive reasoning (no user input).
    /// anticipate → assess → compare → constrain → explain
    pub fn proactive(candidates: Vec<CandidateAction>, horizon_secs: f64) -> CognitivePipeline {
        CognitivePipeline::new()
            .anticipate(horizon_secs)
            .assess()
            .compare(candidates)
            .constrain(vec![
                PolicyConstraint {
                    name: "min_confidence".to_string(),
                    kind: ConstraintKind::MinConfidence(0.5),
                },
                PolicyConstraint {
                    name: "meta_cog".to_string(),
                    kind: ConstraintKind::MetaCogThreshold(0.4),
                },
            ])
            .explain()
    }

    /// Deep reasoning for complex decisions.
    /// attend → recall → project → plan → compare → constrain → assess → explain
    pub fn deep_reasoning(
        seeds: Vec<NodeId>,
        goal_id: NodeId,
        candidates: Vec<CandidateAction>,
    ) -> CognitivePipeline {
        CognitivePipeline::new()
            .attend(seeds)
            .recall(10)
            .project(ProjectionHorizon::MediumTerm)
            .plan(goal_id, 4)
            .compare(candidates)
            .constrain(vec![PolicyConstraint {
                name: "min_confidence".to_string(),
                kind: ConstraintKind::MinConfidence(0.4),
            }])
            .assess()
            .explain()
    }

    /// Quick health check.
    /// assess → coherence_check → explain
    pub fn health_check() -> CognitivePipeline {
        CognitivePipeline::new()
            .assess()
            .coherence_check()
            .explain()
    }

    /// Budgeted reasoning with a time limit.
    pub fn budgeted(
        seeds: Vec<NodeId>,
        candidates: Vec<CandidateAction>,
        budget_ms: u64,
    ) -> CognitivePipeline {
        CognitivePipeline::new()
            .with_budget(Duration::from_millis(budget_ms))
            .attend(seeds)
            .recall(5)
            .compare(candidates)
            .constrain(vec![PolicyConstraint {
                name: "min_confidence".to_string(),
                kind: ConstraintKind::MinConfidence(0.3),
            }])
            .assess()
            .explain()
    }
}

// ── §10.5: Stub Executor (for benchmarks + tests) ──────────────────

/// A stub executor that returns synthetic results for all operators.
///
/// Useful for benchmarking pipeline mechanics without a database.
pub struct StubExecutor;

impl PipelineExecutor for StubExecutor {
    fn execute_operator(
        &self,
        operator: &CognitiveOperator,
        _context: &Option<PipelineContext>,
    ) -> StepOutput {
        match operator {
            CognitiveOperator::Attend(a) => StepOutput::Attend {
                nodes_activated: a.seeds.len() * 3,
                top_activated: a.seeds.iter().map(|id| (*id, 0.8)).collect(),
            },
            CognitiveOperator::Recall(r) => StepOutput::Recall {
                memories_retrieved: r.top_k.min(3),
                top_matches: vec![RecallMatch {
                    text: "Test memory".to_string(),
                    similarity: 0.85,
                    memory_type: "episodic".to_string(),
                }],
            },
            CognitiveOperator::Believe(_) => StepOutput::Believe {
                beliefs_updated: 1,
                confidence_delta: 0.1,
            },
            CognitiveOperator::Project(_) => StepOutput::Project {
                predictions: vec![Prediction {
                    description: "User will check email".to_string(),
                    probability: 0.7,
                    valence: 0.3,
                }],
            },
            CognitiveOperator::Compare(c) => {
                let ranked: Vec<RankedCandidate> = c
                    .candidates
                    .iter()
                    .enumerate()
                    .map(|(i, cand)| RankedCandidate {
                        description: cand.description.clone(),
                        score: cand.confidence * 0.9,
                        personality_bias: 0.05,
                        rank: i + 1,
                    })
                    .collect();
                StepOutput::Compare { ranked }
            }
            CognitiveOperator::Constrain(_) => StepOutput::Constrain {
                passed: 2,
                filtered_out: 1,
                violations: vec![ConstraintViolation {
                    constraint_name: "min_confidence".to_string(),
                    candidate: "Low confidence action".to_string(),
                    reason: "Confidence 0.2 < threshold 0.3".to_string(),
                }],
            },
            CognitiveOperator::Anticipate(_) => StepOutput::Anticipate {
                items: vec![AnticipatedItem {
                    description: "Meeting in 30 minutes".to_string(),
                    expected_at: 1800.0,
                    confidence: 0.9,
                }],
            },
            CognitiveOperator::Plan(_) => StepOutput::Plan {
                plan_found: true,
                steps: 3,
                score: 0.75,
            },
            CognitiveOperator::Assess => StepOutput::Assess {
                overall_confidence: 0.72,
                coverage_gaps: 2,
            },
            CognitiveOperator::CoherenceCheck => StepOutput::Coherence {
                score: 0.85,
                conflicts: 0,
                stale_nodes: 1,
            },
        }
    }
}

// ── §11: Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::NodeKind;

    fn seed_ids() -> Vec<NodeId> {
        vec![
            NodeId::new(NodeKind::Entity, 1),
            NodeId::new(NodeKind::Entity, 2),
        ]
    }

    fn test_candidates() -> Vec<CandidateAction> {
        vec![
            CandidateAction {
                description: "Send notification".to_string(),
                action_kind: "notify".to_string(),
                confidence: 0.8,
                properties: Default::default(),
            },
            CandidateAction {
                description: "Wait and observe".to_string(),
                action_kind: "wait".to_string(),
                confidence: 0.6,
                properties: Default::default(),
            },
        ]
    }

    #[test]
    fn test_empty_pipeline() {
        let pipeline = CognitivePipeline::new();
        let result = execute_pipeline(&pipeline, &StubExecutor);

        assert_eq!(result.status, PipelineStatus::Empty);
        assert_eq!(result.operators_executed, 0);
    }

    #[test]
    fn test_simple_pipeline() {
        let pipeline = CognitivePipeline::new()
            .attend(seed_ids())
            .recall(5)
            .compare(test_candidates());

        assert_eq!(pipeline.len(), 3);

        let result = execute_pipeline(&pipeline, &StubExecutor);
        assert_eq!(result.status, PipelineStatus::Complete);
        assert_eq!(result.operators_executed, 3);
        assert_eq!(result.operators_skipped, 0);
    }

    #[test]
    fn test_pipeline_with_explanation() {
        let pipeline = CognitivePipeline::new()
            .attend(seed_ids())
            .recall(5)
            .explain();

        let result = execute_pipeline(&pipeline, &StubExecutor);

        assert!(result.explanation.is_some());
        let trace = result.explanation.unwrap();
        assert_eq!(trace.steps.len(), 2);
        assert!(trace.summary.contains("2 steps"));
    }

    #[test]
    fn test_user_turn_pattern() {
        let pipeline = PipelinePatterns::user_turn(seed_ids(), test_candidates());

        assert!(pipeline.len() >= 4);
        assert!(pipeline.explain);

        let result = execute_pipeline(&pipeline, &StubExecutor);
        assert_eq!(result.status, PipelineStatus::Complete);
    }

    #[test]
    fn test_health_check_pattern() {
        let pipeline = PipelinePatterns::health_check();

        assert_eq!(pipeline.len(), 2);

        let result = execute_pipeline(&pipeline, &StubExecutor);
        assert_eq!(result.status, PipelineStatus::Complete);

        // Check that coherence result is present.
        let coherence_step = result
            .steps
            .iter()
            .find(|s| s.operator == "coherence_check")
            .unwrap();
        assert!(coherence_step.success);
    }

    #[test]
    fn test_deep_reasoning_pattern() {
        let goal_id = NodeId::new(NodeKind::Goal, 1);
        let pipeline = PipelinePatterns::deep_reasoning(seed_ids(), goal_id, test_candidates());

        assert!(pipeline.len() >= 6);
        let result = execute_pipeline(&pipeline, &StubExecutor);
        assert_eq!(result.status, PipelineStatus::Complete);
    }

    #[test]
    fn test_operator_priority_ordering() {
        assert!(
            CognitiveOperator::Attend(AttendOp {
                seeds: vec![],
                max_hops: 1,
                decay: 0.5,
            })
            .priority()
                > CognitiveOperator::Assess.priority()
        );

        assert!(
            CognitiveOperator::Recall(RecallOp {
                top_k: 5,
                query: None,
                domain: None,
            })
            .priority()
                > CognitiveOperator::CoherenceCheck.priority()
        );
    }

    #[test]
    fn test_operator_names() {
        assert_eq!(
            CognitiveOperator::Attend(AttendOp {
                seeds: vec![],
                max_hops: 1,
                decay: 0.5,
            })
            .name(),
            "attend"
        );
        assert_eq!(CognitiveOperator::Assess.name(), "assess");
        assert_eq!(CognitiveOperator::CoherenceCheck.name(), "coherence_check");
    }

    #[test]
    fn test_projection_horizons() {
        assert!(ProjectionHorizon::OneStep.seconds() < ProjectionHorizon::ShortTerm.seconds());
        assert!(ProjectionHorizon::ShortTerm.seconds() < ProjectionHorizon::MediumTerm.seconds());
        assert!(ProjectionHorizon::MediumTerm.seconds() < ProjectionHorizon::LongTerm.seconds());
    }

    #[test]
    fn test_budgeted_pipeline_short_budget() {
        // With a very small budget, the budgeted mode should prioritize operators.
        let pipeline = CognitivePipeline::new()
            .with_budget(Duration::from_millis(500))
            .attend(seed_ids())
            .recall(5)
            .assess()
            .coherence_check();

        assert_eq!(pipeline.mode, ExecutionMode::Budgeted);

        let result = execute_pipeline(&pipeline, &StubExecutor);
        // Stub executor is instant, so all should complete.
        assert_eq!(result.status, PipelineStatus::Complete);
        assert_eq!(result.operators_executed, 4);
    }

    #[test]
    fn test_pipeline_context() {
        let pipeline = CognitivePipeline::new()
            .with_context(PipelineContext {
                trigger: "user_message".to_string(),
                user_input: Some("Hello".to_string()),
                mentioned_entities: seed_ids(),
                timestamp: 1_000_000.0,
            })
            .attend(seed_ids());

        assert!(pipeline.context.is_some());
        assert_eq!(pipeline.context.as_ref().unwrap().trigger, "user_message");
    }

    #[test]
    fn test_evidence_input() {
        let evidence = EvidenceInput {
            target: Some(NodeId::new(NodeKind::Belief, 1)),
            observation: "User confirmed preference".to_string(),
            direction: 1.0,
            strength: 0.8,
            source: "user".to_string(),
        };

        let pipeline = CognitivePipeline::new().believe(evidence);

        assert_eq!(pipeline.len(), 1);

        let result = execute_pipeline(&pipeline, &StubExecutor);
        assert_eq!(result.status, PipelineStatus::Complete);
    }

    #[test]
    fn test_constraint_kinds() {
        let constraints = vec![
            PolicyConstraint {
                name: "confidence".to_string(),
                kind: ConstraintKind::MinConfidence(0.5),
            },
            PolicyConstraint {
                name: "risk".to_string(),
                kind: ConstraintKind::MaxRisk(0.3),
            },
            PolicyConstraint {
                name: "quiet_hours".to_string(),
                kind: ConstraintKind::QuietHours {
                    start_hour: 22,
                    end_hour: 7,
                },
            },
        ];

        let pipeline = CognitivePipeline::new().constrain(constraints);

        assert_eq!(pipeline.len(), 1);
    }

    #[test]
    fn test_key_insight_extraction() {
        let output = StepOutput::Assess {
            overall_confidence: 0.3,
            coverage_gaps: 5,
        };
        let insight = extract_key_insight(&output);
        assert!(insight.is_some());
        assert!(insight.unwrap().contains("escalating"));
    }

    #[test]
    fn test_proactive_pattern() {
        let pipeline = PipelinePatterns::proactive(test_candidates(), 3600.0);
        assert!(pipeline.len() >= 4);

        let result = execute_pipeline(&pipeline, &StubExecutor);
        assert_eq!(result.status, PipelineStatus::Complete);
    }
}
