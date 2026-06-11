// ── Directory modules ──
mod base;
mod cognition;
mod distributed;
pub mod embedder;
pub mod engine;
pub(crate) mod id;
mod knowledge;
pub(crate) mod time;
mod vector;

// ── Re-exports at original crate paths ──
pub use base::{
    bench_utils, compression, encryption, error, hlc, schema, scoring, serde_helpers, types, vault,
};
pub use cognition::{
    action, agenda, analogy, attention, belief, belief_network, belief_query, benchmark,
    benchmark_ck4, calibration, causal, coherence, consolidate, contradiction, counterfactual,
    evaluator, experimenter, extractor, flywheel, hawkes, intent, introspection, metacognition,
    narrative, observer, patterns, personality, personality_bias, perspective, planner, policy,
    query_dsl, receptivity, replay, schema_induction, skills, state, suggest, surfacing, temporal,
    tick, triggers, world_model,
};
pub use distributed::{conflict, replication, sync};
pub use knowledge::{graph, graph_index};
pub use vector::hnsw;

// ── Convenience re-exports ──
pub use attention::{AttentionConfig, WorkingSet};
pub use conflict::{
    create_conflict, detect_edge_conflicts, scan_claim_conflicts, scan_conflicts,
    scan_conflicts_limited,
};
pub use consolidate::{consolidate, find_consolidation_candidates};
pub use cognition::triggers::TriggerPruneReport;
pub use engine::conflict::ConflictBurndownReport;
pub use engine::digest::{SessionDigest, SessionDigestConfig};
pub use engine::graph_ops::AutoRelateReport;
pub use engine::importance::ImportanceRecalibrationReport;
pub use engine::maintenance::{MaintenanceCycleConfig, MaintenanceCycleReport};
pub use engine::repair::{RepairError, RepairReport};
pub use engine::split::SplitReport;
pub use engine::tenant::{TenantConfig, TenantManager};
pub use engine::YantrikDB;
pub use error::YantrikDbError;
pub use patterns::mine_patterns;
pub use personality::{derive_personality, get_personality, set_personality_trait};
pub use state::{
    CognitiveAttrs, CognitiveEdge, CognitiveEdgeKind, CognitiveNode, NodeId, NodeIdAllocator,
    NodeKind, NodePayload, Provenance,
};
pub use triggers::{check_all_triggers, check_consolidation_triggers, check_decay_triggers};
pub use types::*;
// V13: sessions, temporal helpers, entity profile, cross-domain types are re-exported via `pub use types::*;`
pub use action::{ActionCandidate, ActionConfig, CandidateGenerationResult};
pub use agenda::{
    Agenda, AgendaConfig, AgendaId, AgendaItem, AgendaKind, AgendaStatus, TickResult, UrgencyFn,
};
pub use analogy::{
    AnalogicalOpportunity, AnalogicalQuery, AnalogyMaintenanceReport, AnalogyScope, AnalogyStore,
    CandidateInference, EdgeCorrespondence, NodeCorrespondence, ProjectedFact, StructuralMapping,
    SubgraphGroup, TransferType, TransferredStrategy,
};
pub use belief::{
    BeliefRevisionConfig, Evidence, EvidenceResult, RevisionSummary, ThresholdDirection,
};
pub use belief_network::{
    BPConfig, BPResult, BeliefNetwork, BeliefVariable, Distribution, EdgeRelation,
    EvidenceContribution, Factor as NetworkFactor, FactorId, FactorType, InferenceQuery,
    InferenceResult, InferenceType, NetworkHealth, PotentialFunction, VariableId,
};
pub use belief_query::{BeliefExplanation, BeliefInventory, BeliefOrder, BeliefPattern};
pub use calibration::{
    ActionBandit, BanditRegistry, CalibrationMap, EvidenceSource, InteractionOutcome,
    InteractionRecord, LearningConfig, LearningReport, LearningState, ReliabilityRegistry,
    SourceReliability, UtilityWeights, WeightSnapshot,
};
pub use causal::{
    CausalConfig, CausalEdge, CausalEvidence, CausalExplanation, CausalNode, CausalStage,
    CausalStore, CausalSummary, CausalTrace, DiscoveryMethod,
    DiscoveryReport as CausalDiscoveryReport, EffectEstimate, EvidenceQuality, PredictedEffect,
    WhatIfResult,
};
pub use coherence::{
    BeliefContradiction, CoherenceConfig, CoherenceHistory, CoherenceReport, CoherenceSnapshot,
    DeadlineAlert, DependencyCycle, EnforcementAction, EnforcementKind, EnforcementReport,
    GoalConflict, OrphanedItem, StaleNode,
};
pub use contradiction::{
    BeliefConflict, ConflictDetectionMethod, ContradictionConfig, ContradictionScanResult,
    ResolutionStrategy,
};
pub use counterfactual::{
    CounterfactualConfig, CounterfactualQuery, CounterfactualResult, CounterfactualType,
    DecisionRecord, DeltaDirection, Intervention, NodeDelta, Observation, OutcomeDifference,
    RegretReport, SensitivityEntry, SimulatedStep, StateSnapshot,
};
pub use engine::graph_state::{
    CognitiveGraphSaveResult, CognitiveGraphStats, CognitiveNodeFilter, CognitiveNodeOrder,
};
pub use evaluator::{EvaluatedAction, EvaluationResult, EvaluatorConfig};
pub use experimenter::{
    BetaPosterior, Experiment, ExperimentId, ExperimentRegistry, ExperimentStatus,
    ExperimentVariable, SafetyBound, TrialOutcome, VariantValue,
};
pub use extractor::{
    CognitiveUpdate, ExtractionContext, ExtractionResponse, ExtractorConfig, ExtractorSummary,
    ExtractorTier, LlmExtractionRequest, SerializableOpTemplate, TemplateStore, UpdateOp,
};
pub use flywheel::{
    AutonomousBelief, BeliefCategory, BeliefEvidence, BeliefStage, BeliefStore, FlywheelConfig,
    FormationResult,
};
pub use hawkes::{
    AnticipatedEvent, CircadianProfile, EventPrediction, EventTypeModel, HawkesParams,
    HawkesRegistry, HawkesRegistryConfig, ModelSummary,
};
pub use intent::{IntentConfig, IntentInferenceResult, IntentSource, ScoredIntent};
pub use introspection::{
    BeliefStageBreakdown, Discovery, DiscoveryExplanation, IntrospectionReport, LearningMilestone,
    MilestoneKind,
};
pub use metacognition::{
    AbstainAction, AbstainDecision, AbstainReason, ConfidenceReport, CoverageGap, CoverageGapKind,
    MetaCognitiveConfig, MetaCognitiveHistory, MetaCognitiveReport, MetaCognitiveSnapshot,
    ReasoningHealthReport, SignalDetail, SignalStatus,
};
pub use narrative::{
    ArcAlert, ArcAlertType, ArcId, ArcStatus, ArcTheme, AutobiographicalTimeline, Chapter,
    ChapterType, DirectionChange, Milestone, NarrativeArc, NarrativeEpisode, NarrativeQuery,
    NarrativeResult, TurningPoint,
};
pub use observer::{
    CircadianHistogram, DerivedSignals, EventBuffer, EventCounters, EventFilter, EventKind,
    ObserverConfig, ObserverState, ObserverSummary, SystemEvent, SystemEventData,
};
pub use personality_bias::{
    ActionProperties, BiasConfig, BiasContribution, BondLevel, EvolutionConfig, LearnedPreferences,
    PersonalityBiasResult, PersonalityBiasStore, PersonalityBiasVector, PersonalityImpactReport,
    PersonalityPreset,
};
pub use perspective::{
    ActivationCondition, ActivationContext, CognitiveStyle, ConflictType, EdgeWeightModifier,
    Perspective, PerspectiveConflict, PerspectiveId, PerspectiveStack, PerspectiveStore,
    PerspectiveTransition, PerspectiveType, SalienceOverride, SalienceTarget, TemporalFocus,
};
pub use planner::{
    Blocker, BlockerKind, BoundPrecondition, Plan, PlanProposal, PlanScore, PlanStep, PlanStore,
    PlannerConfig, StepDerivation,
};
pub use policy::{
    PolicyConfig, PolicyContext, PolicyDecision, PolicyResult, ReasoningTrace, SelectedAction,
};
pub use query_dsl::{
    CandidateAction as PipelineCandidateAction, CognitiveOperator, CognitivePipeline,
    ConstraintKind, EvidenceInput, ExecutionMode as PipelineExecutionMode, ExplanationTrace,
    PipelineContext, PipelineExecutor, PipelinePatterns, PipelineResult, PipelineStatus,
    PolicyConstraint, ProjectionHorizon, StepOutput, StepResult,
};
pub use receptivity::{
    ActivityState, AttentionBudgetConfig, ContextSnapshot, NotificationMode, QuietHoursConfig,
    ReceptivityEstimate, ReceptivityFactor, ReceptivityModel, SuggestionOutcome,
};
pub use replay::{
    ActionRecord, BeliefDelta, CausalDelta, DreamReport, OutcomeData, ReplayBudget, ReplayBuffer,
    ReplayEngine, ReplayEntry, ReplayOutcome, ReplayStats, ReplaySummary, SamplingStrategy,
};
pub use schema_induction::{
    ActionTemplate, ConstraintExpr, ContextSnapshot as SchemaContext, Direction as SchemaDirection,
    EpisodeData, ExpectedOutcome, InducedSchema, ParamType, ParameterSlot, SchemaCondition,
    SchemaId, SchemaMaintenanceReport, SchemaStore as InducedSchemaStore, TemplateConstraint,
};
pub use skills::{
    DiscoveryResult, LearnedSkill, SkillConfig, SkillId, SkillMatch, SkillOrigin, SkillRegistry,
    SkillStage, SkillStep, SkillSummary, SkillTrigger,
};
pub use suggest::{
    ActionProposal, ExecutionMode, NextStepRequest, NextStepResponse, PipelineMetrics,
};
pub use surfacing::{
    ProactiveSuggestion, SuppressedItem, SuppressionCause, SurfaceMode, SurfaceOutcome,
    SurfaceRateLimiter, SurfaceReason, SurfacingConfig, SurfacingPreferences, SurfacingResult,
};
pub use temporal::{
    BurstConfig, BurstResult, DeadlineUrgencyConfig, EwmaTracker, IntervalRelation,
    PeriodicityConfig, PeriodicityResult, RecencyConfig, SeasonalHistogram, TemporalMotif,
    TemporalOrder, TemporalRelevanceConfig, TimeInterval,
};
pub use tick::{
    Anomaly, AnomalyKind, CachedSuggestion, TickConfig, TickPhase, TickReport, TickState,
};
pub use world_model::{
    ActionKind as WorldActionKind, ActionOutcome as WorldActionOutcome, OutcomeDistribution,
    StateFeatures, TransitionModel, WorldModelSummary,
};
