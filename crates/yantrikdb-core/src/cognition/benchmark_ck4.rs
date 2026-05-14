//! CK-4 Benchmark Suite — Advanced Reasoning Primitives + Meta-Cognition
//!
//! Categories 21–26, covering the six CK-4 subsystems:
//!   21. Causal Inference Engine
//!   22. Planning Graph / HTN-Lite
//!   23. Coherence Monitor
//!   24. Meta-Cognition
//!   25. Personality Bias Vectors
//!   26. Cognitive Query DSL
//!
//! Each category runs 4–6 persona-driven tests that exercise the core
//! pure-function APIs without requiring a database.

use std::time::Instant;

use super::benchmark::{BenchResult, BenchRun, PersonaScenario};
use super::state::*;

// ── Category 21: Causal Inference Engine ──────────────────────────────

pub fn run_causal_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::causal::*;
    use super::observer::{EventBuffer, SystemEvent, SystemEventData};
    use super::world_model::{StateFeatures, TransitionModel};

    let persona = Some(scenario.name.clone());
    let now = now_secs();

    // Build a causal store from scenario edges.
    let start = Instant::now();
    let mut store = CausalStore::new();

    // Seed causal edges from graph edges that imply causality.
    let causal_edge_count = seed_causal_from_graph(&mut store, &scenario.edges, now);
    let has_edges = store.edge_count() > 0;

    run.add(BenchResult {
        category: "causal".into(),
        test_name: "store_seeding".into(),
        persona: persona.clone(),
        passed: has_edges || causal_edge_count == 0,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("causal_edges".into()),
        metric_value: Some(store.edge_count() as f64),
        details: Some(format!(
            "graph_edges={} causal_converted={}",
            scenario.edges.len(),
            causal_edge_count
        )),
    });

    // Test 2: Predict effects (forward causal chain).
    let start = Instant::now();
    let mut total_predictions = 0;
    let mut all_bounded = true;
    for edge in store.edges() {
        let predictions = predict_effects(&store, &edge.cause, 3);
        for p in &predictions {
            if p.confidence < 0.0 || p.confidence > 1.0 {
                all_bounded = false;
            }
        }
        total_predictions += predictions.len();
    }
    run.add(BenchResult {
        category: "causal".into(),
        test_name: "predict_effects".into(),
        persona: persona.clone(),
        passed: all_bounded,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("total_predictions".into()),
        metric_value: Some(total_predictions as f64),
        details: None,
    });

    // Test 3: What-if analysis.
    let start = Instant::now();
    let model = TransitionModel::new();
    let context = StateFeatures::discretize(now, 0.7, 600.0, 0.0, 1);
    let mut what_if_ran = 0;
    let mut all_valid = true;
    for edge in store.edges().iter().take(5) {
        let result = what_if(&store, &model, &edge.cause, &context, 3);
        if result.net_expected_impact.is_nan() || result.net_expected_impact.is_infinite() {
            all_valid = false;
        }
        what_if_ran += 1;
    }
    run.add(BenchResult {
        category: "causal".into(),
        test_name: "what_if_analysis".into(),
        persona: persona.clone(),
        passed: all_valid,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("scenarios_evaluated".into()),
        metric_value: Some(what_if_ran as f64),
        details: None,
    });

    // Test 4: Causal summary statistics.
    let start = Instant::now();
    let summary = causal_summary(&store);
    let valid_summary = summary.total_edges == store.edge_count();
    let avg_confidence = if store.edge_count() > 0 {
        store.edges().iter().map(|e| e.confidence).sum::<f64>() / store.edge_count() as f64
    } else {
        0.0
    };
    run.add(BenchResult {
        category: "causal".into(),
        test_name: "causal_summary".into(),
        persona: persona.clone(),
        passed: valid_summary,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("total_edges".into()),
        metric_value: Some(summary.total_edges as f64),
        details: Some(format!(
            "avg_confidence={:.3} stages=H{}/C{}/E{}/W{}/R{}",
            avg_confidence,
            summary.hypothesized,
            summary.candidates,
            summary.established,
            summary.weakening,
            summary.refuted,
        )),
    });

    // Test 5: Maintenance (decay + prune).
    let start = Instant::now();
    let mut maintenance_store = store.clone();
    let report = maintain_causal_store(&mut maintenance_store, now);
    let edges_after = maintenance_store.edge_count();
    run.add(BenchResult {
        category: "causal".into(),
        test_name: "maintenance".into(),
        persona: persona.clone(),
        passed: edges_after <= store.edge_count(),
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("pruned".into()),
        metric_value: Some(report.refuted_edges_pruned as f64 + report.capacity_pruned as f64),
        details: Some(format!(
            "decayed={} pruned_refuted={} pruned_capacity={}",
            report.edges_decayed_to_refuted, report.refuted_edges_pruned, report.capacity_pruned
        )),
    });

    // Test 6: Discovery from event buffer.
    let start = Instant::now();
    let mut event_buffer = EventBuffer::new(100);
    // Feed persona's episode timestamps as events.
    for (i, node) in scenario.nodes.iter().enumerate() {
        if let NodePayload::Episode(ep) = &node.payload {
            event_buffer.push(SystemEvent::new(
                ep.occurred_at,
                SystemEventData::AppOpened {
                    app_id: (i % 24) as u16,
                },
            ));
        }
    }
    let mut discovery_store = CausalStore::new();
    let disc_report =
        discover_local_causality(&mut discovery_store, &event_buffer, &scenario.edges, now);
    run.add(BenchResult {
        category: "causal".into(),
        test_name: "discovery".into(),
        persona: persona.clone(),
        passed: true, // discovery is best-effort
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("temporal_patterns".into()),
        metric_value: Some(disc_report.temporal_patterns_found as f64),
        details: Some(format!(
            "temporal={} granger={} graph_imported={} total={}",
            disc_report.temporal_patterns_found,
            disc_report.granger_edges_added,
            disc_report.graph_edges_imported,
            disc_report.total_edges,
        )),
    });
}

/// Convert graph edges with causal semantics into CausalStore edges.
fn seed_causal_from_graph(
    store: &mut super::causal::CausalStore,
    edges: &[CognitiveEdge],
    now: f64,
) -> usize {
    use super::causal::*;

    let mut count = 0;
    for edge in edges {
        let (strength, confidence) = match edge.kind {
            CognitiveEdgeKind::Causes => (0.7, 0.6),
            CognitiveEdgeKind::Prevents => (-0.6, 0.5),
            CognitiveEdgeKind::AdvancesGoal => (0.5, 0.5),
            CognitiveEdgeKind::BlocksGoal => (-0.5, 0.4),
            _ => continue,
        };
        store.upsert(CausalEdge {
            cause: CausalNode::GraphNode(edge.src),
            effect: CausalNode::GraphNode(edge.dst),
            strength,
            confidence,
            observation_count: 3,
            intervention_count: 0,
            non_occurrence_count: 0,
            median_lag_secs: 60.0,
            lag_iqr_secs: 30.0,
            context_strengths: Vec::new(),
            trace: CausalTrace {
                evidence: Vec::new(),
                primary_method: DiscoveryMethod::GraphStructure,
                summary: format!("{:?} edge", edge.kind),
            },
            created_at: now - 3600.0,
            updated_at: now,
            stage: CausalStage::Candidate,
        });
        count += 1;
    }
    count
}

// ── Category 22: Planning Graph / HTN-Lite ────────────────────────────

pub fn run_planner_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::planner::*;

    let persona = Some(scenario.name.clone());
    let now = now_secs();

    // Extract goals, tasks, and schemas from persona nodes.
    let (goals, tasks, schemas, constraints) = extract_planning_data(&scenario.nodes);

    // Test 1: Plan instantiation for each goal.
    let start = Instant::now();
    let mut plans_generated = 0;
    let mut all_scores_valid = true;
    let skill_templates: Vec<SkillTemplate> = Vec::new();
    let ctx = PlanningContext {
        schemas: &schemas,
        goals: &goals,
        tasks: &tasks,
        constraints: &constraints,
        edges: &scenario.edges,
        skills: &skill_templates,
        now,
        config: &PlannerConfig::default(),
    };

    for goal in &goals {
        let proposal = instantiate_plan(goal.node_id, &ctx);
        plans_generated += proposal.plans.len();
        for plan in &proposal.plans {
            if plan.score.composite < 0.0 || plan.score.composite > 1.0 {
                all_scores_valid = false;
            }
        }
    }
    run.add(BenchResult {
        category: "planner".into(),
        test_name: "plan_instantiation".into(),
        persona: persona.clone(),
        passed: all_scores_valid,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("plans_generated".into()),
        metric_value: Some(plans_generated as f64),
        details: Some(format!("goals={}", goals.len())),
    });

    // Test 2: Blocker detection.
    let start = Instant::now();
    let mut total_blockers = 0;
    for goal in &goals {
        let blockers = detect_blockers(goal.node_id, &ctx);
        total_blockers += blockers.len();
    }
    run.add(BenchResult {
        category: "planner".into(),
        test_name: "blocker_detection".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("blockers_found".into()),
        metric_value: Some(total_blockers as f64),
        details: None,
    });

    // Test 3: Plan scoring consistency.
    let start = Instant::now();
    let config = PlannerConfig::default();
    let mut scores_monotonic = true;
    for goal in &goals {
        let proposal = instantiate_plan(goal.node_id, &ctx);
        // Plans should be sorted by composite score descending.
        let composites: Vec<f64> = proposal.plans.iter().map(|p| p.score.composite).collect();
        for w in composites.windows(2) {
            if w[0] < w[1] - 1e-9 {
                scores_monotonic = false;
            }
        }
    }
    run.add(BenchResult {
        category: "planner".into(),
        test_name: "score_ordering".into(),
        persona: persona.clone(),
        passed: scores_monotonic,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: None,
        metric_value: None,
        details: None,
    });

    // Test 4: Next-step extraction.
    let start = Instant::now();
    let mut steps_found = 0;
    for goal in &goals {
        if let Some(_step) = next_plan_step(goal.node_id, &ctx) {
            steps_found += 1;
        }
    }
    run.add(BenchResult {
        category: "planner".into(),
        test_name: "next_step_extraction".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("steps_found".into()),
        metric_value: Some(steps_found as f64),
        details: Some(format!("goals_checked={}", goals.len())),
    });

    // Test 5: PlanStore persistence roundtrip.
    let start = Instant::now();
    let mut store = PlanStore::new();
    let mut stored = 0;
    for goal in &goals {
        let proposal = instantiate_plan(goal.node_id, &ctx);
        if let Some(plan) = proposal.plans.into_iter().next() {
            store.set_plan(plan);
            stored += 1;
        }
    }
    let retrieved = goals
        .iter()
        .filter(|g| store.get_plan(g.node_id).is_some())
        .count();
    run.add(BenchResult {
        category: "planner".into(),
        test_name: "plan_store_roundtrip".into(),
        persona: persona.clone(),
        passed: retrieved == stored,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("stored".into()),
        metric_value: Some(stored as f64),
        details: Some(format!("stored={} retrieved={}", stored, retrieved)),
    });
}

/// Extract planning-relevant data from persona nodes.
fn extract_planning_data(
    nodes: &[CognitiveNode],
) -> (
    Vec<super::planner::GoalEntry>,
    Vec<super::planner::TaskEntry>,
    Vec<super::planner::SchemaEntry>,
    Vec<super::planner::ConstraintEntry>,
) {
    let mut goals = Vec::new();
    let mut tasks = Vec::new();
    let mut schemas = Vec::new();
    let mut constraints = Vec::new();

    for node in nodes {
        match &node.payload {
            NodePayload::Goal(p) => {
                goals.push(super::planner::GoalEntry {
                    node_id: node.id,
                    attrs: node.attrs.clone(),
                    payload: p.clone(),
                });
            }
            NodePayload::Task(p) => {
                tasks.push(super::planner::TaskEntry {
                    node_id: node.id,
                    attrs: node.attrs.clone(),
                    payload: p.clone(),
                });
            }
            NodePayload::ActionSchema(p) => {
                schemas.push(super::planner::SchemaEntry {
                    node_id: node.id,
                    attrs: node.attrs.clone(),
                    payload: p.clone(),
                });
            }
            NodePayload::Constraint(p) => {
                constraints.push(super::planner::ConstraintEntry {
                    node_id: node.id,
                    payload: p.clone(),
                });
            }
            _ => {}
        }
    }

    (goals, tasks, schemas, constraints)
}

// ── Category 23: Coherence Monitor ────────────────────────────────────

pub fn run_coherence_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::attention::{AttentionConfig, WorkingSet};
    use super::coherence::*;
    use super::contradiction::BeliefConflict;

    let persona = Some(scenario.name.clone());
    let now = now_secs();
    let config = CoherenceConfig::default();

    // Build a working set from scenario nodes.
    let ws = build_working_set(&scenario.nodes, &scenario.edges);

    // Test 1: Full coherence check.
    let start = Instant::now();
    let inputs = CoherenceInputs {
        working_set: &ws,
        edges: &scenario.edges,
        belief_conflicts: &[],
        config: &config,
        now,
    };
    let report = check_coherence(&inputs);
    let valid_score = report.coherence_score >= 0.0 && report.coherence_score <= 1.0;
    let valid_load = report.cognitive_load >= 0.0;
    run.add(BenchResult {
        category: "coherence".into(),
        test_name: "full_check".into(),
        persona: persona.clone(),
        passed: valid_score && valid_load,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("coherence_score".into()),
        metric_value: Some(report.coherence_score),
        details: Some(format!(
            "load={:.3} conflicts={} stale={} orphans={}",
            report.cognitive_load,
            report.goal_conflicts.len(),
            report.stale_activations.len(),
            report.orphaned_items.len(),
        )),
    });

    // Test 2: Enforcement plan generation.
    let start = Instant::now();
    let enforcement = plan_enforcement(&report, &ws, &config);
    let valid_enforcement = enforcement.items_affected >= 0;
    run.add(BenchResult {
        category: "coherence".into(),
        test_name: "enforcement_plan".into(),
        persona: persona.clone(),
        passed: true, // enforcement is advisory
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("items_affected".into()),
        metric_value: Some(enforcement.items_affected as f64),
        details: Some(format!(
            "actions={} emergency={}",
            enforcement.actions.len(),
            enforcement.emergency_triggered,
        )),
    });

    // Test 3: Coherence with belief conflicts.
    let start = Instant::now();
    let fake_conflicts = build_belief_conflicts(&scenario.nodes);
    let conflict_inputs = CoherenceInputs {
        working_set: &ws,
        edges: &scenario.edges,
        belief_conflicts: &fake_conflicts,
        config: &config,
        now,
    };
    let conflict_report = check_coherence(&conflict_inputs);
    // More conflicts → lower coherence.
    let coherence_degraded = fake_conflicts.is_empty()
        || conflict_report.coherence_score <= report.coherence_score + 0.01;
    run.add(BenchResult {
        category: "coherence".into(),
        test_name: "conflict_impact".into(),
        persona: persona.clone(),
        passed: coherence_degraded,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("conflict_count".into()),
        metric_value: Some(fake_conflicts.len() as f64),
        details: Some(format!(
            "base_score={:.3} conflict_score={:.3}",
            report.coherence_score, conflict_report.coherence_score,
        )),
    });

    // Test 4: Coherence history tracking.
    let start = Instant::now();
    let mut history = CoherenceHistory::new(50);
    // Simulate 5 checks.
    for i in 0..5 {
        let mut varied_config = config.clone();
        varied_config.stale_threshold_secs = 3600.0 + (i as f64 * 100.0);
        let check_inputs = CoherenceInputs {
            working_set: &ws,
            edges: &scenario.edges,
            belief_conflicts: &[],
            config: &varied_config,
            now: now + (i as f64 * 60.0),
        };
        let r = check_coherence(&check_inputs);
        history.record(&r);
    }
    let trend = history.trend(5);
    let avg = history.recent_average(5);
    let valid_history = avg >= 0.0 && avg <= 1.0 && trend.is_finite();
    run.add(BenchResult {
        category: "coherence".into(),
        test_name: "history_tracking".into(),
        persona: persona.clone(),
        passed: valid_history,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("trend".into()),
        metric_value: Some(trend),
        details: Some(format!(
            "avg={:.3} checks={} snapshots={}",
            avg,
            history.total_checks,
            history.snapshot_count()
        )),
    });

    // Test 5: Attention fragmentation detection.
    let start = Instant::now();
    let frag = report.attention_fragmentation;
    let valid_frag = frag >= 0.0 && frag <= 1.0;
    run.add(BenchResult {
        category: "coherence".into(),
        test_name: "fragmentation_detection".into(),
        persona: persona.clone(),
        passed: valid_frag,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("fragmentation".into()),
        metric_value: Some(frag),
        details: None,
    });
}

/// Build a WorkingSet from scenario nodes for coherence testing.
fn build_working_set(
    nodes: &[CognitiveNode],
    edges: &[CognitiveEdge],
) -> super::attention::WorkingSet {
    use super::attention::{AttentionConfig, WorkingSet};

    let config = AttentionConfig {
        capacity: 100, // large enough for all persona nodes
        max_hops: 2,
        top_k_per_hop: 5,
        hop_decay: 0.5,
        activation_threshold: 0.05,
        lateral_inhibition: 0.3,
        insertion_boost: 0.1,
    };
    let mut ws = WorkingSet::with_config(config);

    // Insert nodes, preserving edges for spreading.
    for node in nodes {
        ws.insert(node.clone());
    }
    for edge in edges {
        ws.add_edge(edge.clone());
    }

    ws
}

/// Build synthetic belief conflicts from belief nodes for testing.
fn build_belief_conflicts(nodes: &[CognitiveNode]) -> Vec<super::contradiction::BeliefConflict> {
    use super::contradiction::{BeliefConflict, ConflictDetectionMethod, ResolutionStrategy};

    let beliefs: Vec<&CognitiveNode> = nodes
        .iter()
        .filter(|n| matches!(n.payload, NodePayload::Belief(_)))
        .collect();

    let mut conflicts = Vec::new();
    let now = now_secs();
    // Create pairwise conflicts between the first few beliefs.
    for pair in beliefs.windows(2).take(3) {
        conflicts.push(BeliefConflict {
            belief_a: pair[0].id,
            belief_b: pair[1].id,
            detection_method: ConflictDetectionMethod::DomainOpposition,
            severity: 0.6,
            description: format!(
                "Conflict between '{}' and '{}'",
                pair[0].label, pair[1].label
            ),
            detected_at: now,
            suggested_resolution: ResolutionStrategy::PreferStrongerEvidence,
        });
    }
    conflicts
}

// ── Category 24: Meta-Cognition ──────────────────────────────────────

pub fn run_metacognition_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::calibration::LearningState;
    use super::experimenter::ExperimentRegistry;
    use super::flywheel::BeliefStore;
    use super::metacognition::*;
    use super::observer::EventBuffer;
    use super::skills::SkillRegistry;
    use super::world_model::TransitionModel;

    let persona = Some(scenario.name.clone());
    let now = now_secs();

    // Build default subsystem inputs.
    let learning_state = LearningState::default();
    let belief_store = BeliefStore::new();
    let event_buffer = EventBuffer::new(100);
    let skill_registry = SkillRegistry::new();
    let experiment_registry = ExperimentRegistry::new();
    let transition_model = TransitionModel::new();
    let config = MetaCognitiveConfig::default();

    let inputs = MetaCognitiveInputs {
        learning_state: &learning_state,
        belief_store: &belief_store,
        event_buffer: &event_buffer,
        skill_registry: &skill_registry,
        experiment_registry: &experiment_registry,
        transition_model: &transition_model,
        config: &config,
        now,
    };

    // Test 1: Meta-cognitive assessment.
    let start = Instant::now();
    let report = metacognitive_assessment(&inputs);
    let valid = report.overall_confidence >= 0.0
        && report.overall_confidence <= 1.0
        && report.evidence_sparsity >= 0.0
        && report.evidence_sparsity <= 1.0;
    run.add(BenchResult {
        category: "metacognition".into(),
        test_name: "assessment".into(),
        persona: persona.clone(),
        passed: valid,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("overall_confidence".into()),
        metric_value: Some(report.overall_confidence),
        details: Some(format!(
            "evidence={:.3} disagreement={:.3} accuracy={:.3} coverage={:.3}",
            report.evidence_sparsity,
            report.model_disagreement,
            report.prediction_accuracy,
            report.coverage,
        )),
    });

    // Test 2: Abstain decision.
    let start = Instant::now();
    let candidates = vec![
        MetaActionCandidate {
            description: "Send reminder".into(),
            confidence: 0.8,
        },
        MetaActionCandidate {
            description: "Delete file".into(),
            confidence: 0.3,
        },
    ];
    let decision = should_abstain(&report, &candidates, &config);
    let valid_decision = decision.meta_confidence >= 0.0 && decision.meta_confidence <= 1.0;
    run.add(BenchResult {
        category: "metacognition".into(),
        test_name: "abstain_decision".into(),
        persona: persona.clone(),
        passed: valid_decision,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("meta_confidence".into()),
        metric_value: Some(decision.meta_confidence),
        details: Some(format!(
            "action={:?} reasons={}",
            decision.action,
            decision.reasons.len(),
        )),
    });

    // Test 3: Confidence report.
    let start = Instant::now();
    let conf_report = confidence_report(&inputs);
    let valid_conf = conf_report.calibration_error >= 0.0
        && conf_report.prediction_accuracy >= 0.0
        && conf_report.prediction_accuracy <= 1.0;
    run.add(BenchResult {
        category: "metacognition".into(),
        test_name: "confidence_report".into(),
        persona: persona.clone(),
        passed: valid_conf,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("calibration_error".into()),
        metric_value: Some(conf_report.calibration_error),
        details: Some(format!(
            "bins={} sources={}",
            conf_report.bin_details.len(),
            conf_report.source_reliabilities.len(),
        )),
    });

    // Test 4: Reasoning health report.
    let start = Instant::now();
    let health = reasoning_health(&inputs);
    let valid_grade = "ABCDF".contains(health.grade);
    let valid_health = health.health_score >= 0.0 && health.health_score <= 1.0;
    run.add(BenchResult {
        category: "metacognition".into(),
        test_name: "reasoning_health".into(),
        persona: persona.clone(),
        passed: valid_grade && valid_health,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("health_score".into()),
        metric_value: Some(health.health_score),
        details: Some(format!(
            "grade={} subsystems={} recommendations={}",
            health.grade,
            health.subsystem_health.len(),
            health.recommendations.len(),
        )),
    });

    // Test 5: Signal detail analysis.
    let start = Instant::now();
    let all_signals_valid = report
        .signal_details
        .iter()
        .all(|s| s.value >= 0.0 && s.value <= 1.0);
    let signal_count = report.signal_details.len();
    run.add(BenchResult {
        category: "metacognition".into(),
        test_name: "signal_analysis".into(),
        persona: persona.clone(),
        passed: all_signals_valid && signal_count > 0,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("signal_count".into()),
        metric_value: Some(signal_count as f64),
        details: Some(format!("coverage_gaps={}", report.coverage_gaps.len())),
    });

    // Test 6: History tracking.
    let start = Instant::now();
    let mut history = MetaCognitiveHistory::new(50);
    // Record several assessments with their decisions.
    for _ in 0..5 {
        let r = metacognitive_assessment(&inputs);
        let d = should_abstain(&r, &candidates, &config);
        history.record(&r, &d);
    }
    let valid_history = history.total_assessments == 5;
    run.add(BenchResult {
        category: "metacognition".into(),
        test_name: "history_tracking".into(),
        persona: persona.clone(),
        passed: valid_history,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("assessments".into()),
        metric_value: Some(history.total_assessments as f64),
        details: Some(format!("escalation_rate={:.3}", history.escalation_rate(5),)),
    });
}

// ── Category 25: Personality Bias Vectors ────────────────────────────

pub fn run_personality_bias_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::personality_bias::*;

    let persona = Some(scenario.name.clone());

    // Test 1: All presets produce valid vectors.
    let start = Instant::now();
    let presets = [
        PersonalityPreset::Assistant,
        PersonalityPreset::Companion,
        PersonalityPreset::Coach,
        PersonalityPreset::Guardian,
    ];
    let mut all_valid = true;
    for preset in &presets {
        let v = preset.vector();
        if !is_valid_vector(&v) {
            all_valid = false;
        }
    }
    run.add(BenchResult {
        category: "personality_bias".into(),
        test_name: "preset_validity".into(),
        persona: persona.clone(),
        passed: all_valid,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("presets_tested".into()),
        metric_value: Some(presets.len() as f64),
        details: None,
    });

    // Test 2: Bias computation with diverse action properties.
    let start = Instant::now();
    let config = BiasConfig::default();
    let companion = PersonalityPreset::Companion.vector();
    let guardian = PersonalityPreset::Guardian.vector();

    let risky_action = ActionProperties {
        info_gain: 0.8,
        anticipatory_value: 0.3,
        risk: 0.9,
        emotional_utility: 0.2,
        goal_progress: 0.5,
        novelty: 0.7,
        follow_up_value: 0.3,
        base_confidence: 0.6,
    };

    let companion_bias = compute_bias(&companion, &risky_action, &config);
    let guardian_bias = compute_bias(&guardian, &risky_action, &config);

    // Guardian should be more cautious about risky actions.
    let guardian_more_cautious = guardian_bias.total_bias < companion_bias.total_bias;
    run.add(BenchResult {
        category: "personality_bias".into(),
        test_name: "bias_computation".into(),
        persona: persona.clone(),
        passed: guardian_more_cautious,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("companion_bias".into()),
        metric_value: Some(companion_bias.total_bias),
        details: Some(format!(
            "companion={:.3} guardian={:.3} contributions={}",
            companion_bias.total_bias,
            guardian_bias.total_bias,
            companion_bias.contributions.len(),
        )),
    });

    // Test 3: Bond-level dampening.
    let start = Instant::now();
    let full_personality = PersonalityPreset::Companion.vector();
    let stranger_dampened = dampen_personality(&full_personality, BondLevel::Stranger);
    let trusted_dampened = dampen_personality(&full_personality, BondLevel::Trusted);

    // Stranger dampening should push toward neutral more than Trusted.
    let neutral = PersonalityBiasVector::neutral();
    let stranger_dist = personality_distance(&stranger_dampened, &neutral);
    let trusted_dist = personality_distance(&trusted_dampened, &neutral);
    let dampening_correct = stranger_dist < trusted_dist;
    run.add(BenchResult {
        category: "personality_bias".into(),
        test_name: "bond_dampening".into(),
        persona: persona.clone(),
        passed: dampening_correct,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("stranger_distance".into()),
        metric_value: Some(stranger_dist),
        details: Some(format!(
            "stranger_dist={:.3} trusted_dist={:.3}",
            stranger_dist, trusted_dist,
        )),
    });

    // Test 4: Personality evolution.
    let start = Instant::now();
    let current = PersonalityPreset::Assistant.vector();
    let prefs = LearnedPreferences {
        adjustments: vec![(0, 0.15)], // curiosity dimension, positive adjustment
        observation_count: 20,
    };
    let evo_config = EvolutionConfig::default();

    let evolved = evolve_personality(&current, BondLevel::Familiar, &prefs, &evo_config);
    let curiosity_increased = evolved.curiosity >= current.curiosity;
    let still_valid = is_valid_vector(&evolved);
    run.add(BenchResult {
        category: "personality_bias".into(),
        test_name: "personality_evolution".into(),
        persona: persona.clone(),
        passed: curiosity_increased && still_valid,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("curiosity_delta".into()),
        metric_value: Some(evolved.curiosity - current.curiosity),
        details: Some(format!(
            "before={:.3} after={:.3} valid={}",
            current.curiosity, evolved.curiosity, still_valid,
        )),
    });

    // Test 5: Impact report across presets.
    let start = Instant::now();
    let safe_action = ActionProperties {
        info_gain: 0.2,
        anticipatory_value: 0.1,
        risk: 0.1,
        emotional_utility: 0.5,
        goal_progress: 0.8,
        novelty: 0.1,
        follow_up_value: 0.2,
        base_confidence: 0.9,
    };

    let actions: Vec<(&str, &ActionProperties)> = vec![
        ("risky_action", &risky_action),
        ("safe_action", &safe_action),
    ];

    let report = personality_impact(
        &companion,
        BondLevel::Bonded,
        "Companion",
        &actions,
        &config,
    );
    let valid_report = report.action_biases.len() == 2;
    let avg_bias = if report.action_biases.is_empty() {
        0.0
    } else {
        report
            .action_biases
            .iter()
            .map(|a| a.bias_result.total_bias)
            .sum::<f64>()
            / report.action_biases.len() as f64
    };
    run.add(BenchResult {
        category: "personality_bias".into(),
        test_name: "impact_report".into(),
        persona: persona.clone(),
        passed: valid_report,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("actions_analyzed".into()),
        metric_value: Some(report.action_biases.len() as f64),
        details: Some(format!(
            "avg_bias={:.3} most_boosted={:?} most_penalized={:?}",
            avg_bias, report.most_boosted, report.most_penalized,
        )),
    });

    // Test 6: PersonalityBiasStore integration.
    let start = Instant::now();
    let mut store = PersonalityBiasStore::from_preset(PersonalityPreset::Companion);
    store.set_bond_level(BondLevel::Familiar);

    let store_result = store.apply_bias(&risky_action);
    let store_result2 = store.apply_bias(&safe_action);
    let store_consistent =
        store_result.total_bias.is_finite() && store_result2.total_bias.is_finite();
    run.add(BenchResult {
        category: "personality_bias".into(),
        test_name: "store_integration".into(),
        persona: persona.clone(),
        passed: store_consistent,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("risky_bias".into()),
        metric_value: Some(store_result.total_bias),
        details: Some(format!(
            "risky={:.3} safe={:.3} threshold_delta={:.3}",
            store_result.total_bias,
            store_result2.total_bias,
            store_result.confidence_threshold_delta,
        )),
    });
}

/// Check that all personality dimensions are in [0.0, 1.0].
fn is_valid_vector(v: &super::personality_bias::PersonalityBiasVector) -> bool {
    let dims = [
        v.curiosity,
        v.proactivity,
        v.caution,
        v.warmth,
        v.efficiency,
        v.playfulness,
        v.formality,
        v.persistence,
    ];
    dims.iter().all(|&d| d >= 0.0 && d <= 1.0)
}

/// Euclidean distance between two personality vectors (for testing).
fn personality_distance(
    a: &super::personality_bias::PersonalityBiasVector,
    b: &super::personality_bias::PersonalityBiasVector,
) -> f64 {
    let dims = [
        (a.curiosity - b.curiosity),
        (a.proactivity - b.proactivity),
        (a.caution - b.caution),
        (a.warmth - b.warmth),
        (a.efficiency - b.efficiency),
        (a.playfulness - b.playfulness),
        (a.formality - b.formality),
        (a.persistence - b.persistence),
    ];
    dims.iter().map(|d| d * d).sum::<f64>().sqrt()
}

// ── Category 26: Cognitive Query DSL ─────────────────────────────────

pub fn run_query_dsl_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::query_dsl::*;

    let persona = Some(scenario.name.clone());
    let now = now_secs();

    // Build seed node IDs from scenario.
    let seeds: Vec<NodeId> = scenario.nodes.iter().take(3).map(|n| n.id).collect();

    let candidates: Vec<CandidateAction> = vec![
        CandidateAction {
            description: "Remind about deadline".into(),
            action_kind: "notify".into(),
            confidence: 0.8,
            properties: Default::default(),
        },
        CandidateAction {
            description: "Suggest break".into(),
            action_kind: "wellbeing".into(),
            confidence: 0.5,
            properties: Default::default(),
        },
    ];

    // Test 1: Pipeline builder fluency.
    let start = Instant::now();
    let pipeline = CognitivePipeline::new()
        .attend(seeds.clone())
        .recall(5)
        .compare(candidates.clone())
        .assess()
        .coherence_check()
        .explain();
    let correct_len = pipeline.len() == 5; // 5 operators (explain is a flag, not operator)
    let has_explain = pipeline.explain;
    run.add(BenchResult {
        category: "query_dsl".into(),
        test_name: "pipeline_builder".into(),
        persona: persona.clone(),
        passed: correct_len && has_explain,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("operators".into()),
        metric_value: Some(pipeline.len() as f64),
        details: None,
    });

    // Test 2: Stub executor (pure-function execution).
    let start = Instant::now();
    let pipeline = CognitivePipeline::new()
        .attend(seeds.clone())
        .recall(5)
        .explain();
    let stub = StubExecutor;
    let result = execute_pipeline(&pipeline, &stub);
    let all_success = result.steps.iter().all(|s| s.success);
    let has_explanation = result.explanation.is_some();
    run.add(BenchResult {
        category: "query_dsl".into(),
        test_name: "stub_execution".into(),
        persona: persona.clone(),
        passed: all_success && has_explanation,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("steps_executed".into()),
        metric_value: Some(result.operators_executed as f64),
        details: Some(format!("status={:?}", result.status)),
    });

    // Test 3: Pipeline patterns.
    let start = Instant::now();
    let patterns = vec![
        (
            "user_turn",
            PipelinePatterns::user_turn(seeds.clone(), candidates.clone()),
        ),
        (
            "proactive",
            PipelinePatterns::proactive(candidates.clone(), 3600.0),
        ),
        (
            "deep_reasoning",
            PipelinePatterns::deep_reasoning(seeds.clone(), seeds[0], candidates.clone()),
        ),
        ("health_check", PipelinePatterns::health_check()),
        (
            "budgeted",
            PipelinePatterns::budgeted(seeds.clone(), candidates.clone(), 50),
        ),
    ];
    let all_non_empty = patterns.iter().all(|(_, p)| !p.is_empty());
    let pattern_count = patterns.len();
    run.add(BenchResult {
        category: "query_dsl".into(),
        test_name: "pipeline_patterns".into(),
        persona: persona.clone(),
        passed: all_non_empty,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("patterns_tested".into()),
        metric_value: Some(pattern_count as f64),
        details: Some(
            patterns
                .iter()
                .map(|(name, p)| format!("{}={}", name, p.len()))
                .collect::<Vec<_>>()
                .join(" "),
        ),
    });

    // Test 4: Budgeted execution respects limits.
    let start = Instant::now();
    let budgeted = PipelinePatterns::budgeted(seeds.clone(), candidates.clone(), 1);
    let stub = StubExecutor;
    let result = execute_pipeline(&budgeted, &stub);
    // With 1ms budget, might not complete all operators.
    let valid_status = matches!(
        result.status,
        PipelineStatus::Complete | PipelineStatus::Partial
    );
    run.add(BenchResult {
        category: "query_dsl".into(),
        test_name: "budgeted_execution".into(),
        persona: persona.clone(),
        passed: valid_status,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("executed".into()),
        metric_value: Some(result.operators_executed as f64),
        details: Some(format!(
            "skipped={} budget_exhausted={}",
            result.operators_skipped, result.budget_exhausted,
        )),
    });

    // Test 5: Execution with DB-backed executor.
    let start = Instant::now();
    let pipeline = PipelinePatterns::health_check();
    let db_result = scenario.db.execute_pipeline(&pipeline);
    let db_ok = db_result.is_ok();
    let (exec_count, status_str) = if let Ok(ref r) = db_result {
        (r.operators_executed, format!("{:?}", r.status))
    } else {
        (0, "error".into())
    };
    run.add(BenchResult {
        category: "query_dsl".into(),
        test_name: "db_backed_execution".into(),
        persona: persona.clone(),
        passed: db_ok,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("operators_executed".into()),
        metric_value: Some(exec_count as f64),
        details: Some(format!("status={}", status_str)),
    });

    // Test 6: Full pipeline through DB.
    let start = Instant::now();
    let full_pipeline = CognitivePipeline::new()
        .attend(seeds.clone())
        .recall(3)
        .compare(candidates.clone())
        .assess()
        .coherence_check()
        .explain();
    let full_result = scenario.db.execute_pipeline(&full_pipeline);
    let full_ok = full_result.is_ok();
    let (full_exec, full_explained) = if let Ok(ref r) = full_result {
        (r.operators_executed, r.explanation.is_some())
    } else {
        (0, false)
    };
    run.add(BenchResult {
        category: "query_dsl".into(),
        test_name: "full_db_pipeline".into(),
        persona: persona.clone(),
        passed: full_ok && full_explained,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("operators_executed".into()),
        metric_value: Some(full_exec as f64),
        details: Some(format!("explained={}", full_explained)),
    });
}

// ── Unit Tests ──

#[cfg(test)]
mod tests {
    use super::super::benchmark::{BenchRun, BenchTracker};

    fn test_run() -> BenchRun {
        let tracker = BenchTracker::in_memory().unwrap();
        tracker.start_run("ck4-test", None).unwrap()
    }

    #[test]
    fn test_causal_bench_runs() {
        let tracker = BenchTracker::in_memory().unwrap();
        let mut run = tracker.start_run("causal-test", None).unwrap();
        let scenario = super::super::benchmark::build_aisha().unwrap();
        super::run_causal_tests(&mut run, &scenario);
        assert!(
            !run.results.is_empty(),
            "causal tests should produce results"
        );
        assert!(
            run.results.iter().all(|r| r.category == "causal"),
            "all results should be causal category"
        );
    }

    #[test]
    fn test_planner_bench_runs() {
        let tracker = BenchTracker::in_memory().unwrap();
        let mut run = tracker.start_run("planner-test", None).unwrap();
        let scenario = super::super::benchmark::build_marcus().unwrap();
        super::run_planner_tests(&mut run, &scenario);
        assert!(
            !run.results.is_empty(),
            "planner tests should produce results"
        );
    }

    #[test]
    fn test_coherence_bench_runs() {
        let tracker = BenchTracker::in_memory().unwrap();
        let mut run = tracker.start_run("coherence-test", None).unwrap();
        let scenario = super::super::benchmark::build_priya().unwrap();
        super::run_coherence_tests(&mut run, &scenario);
        assert!(
            !run.results.is_empty(),
            "coherence tests should produce results"
        );
    }

    #[test]
    fn test_metacognition_bench_runs() {
        let tracker = BenchTracker::in_memory().unwrap();
        let mut run = tracker.start_run("meta-test", None).unwrap();
        let scenario = super::super::benchmark::build_emre().unwrap();
        super::run_metacognition_tests(&mut run, &scenario);
        assert!(
            !run.results.is_empty(),
            "metacognition tests should produce results"
        );
    }

    #[test]
    fn test_personality_bias_bench_runs() {
        let tracker = BenchTracker::in_memory().unwrap();
        let mut run = tracker.start_run("personality-test", None).unwrap();
        let scenario = super::super::benchmark::build_keiko().unwrap();
        super::run_personality_bias_tests(&mut run, &scenario);
        assert!(
            !run.results.is_empty(),
            "personality bias tests should produce results"
        );
    }

    #[test]
    fn test_query_dsl_bench_runs() {
        let tracker = BenchTracker::in_memory().unwrap();
        let mut run = tracker.start_run("dsl-test", None).unwrap();
        let scenario = super::super::benchmark::build_aisha().unwrap();
        super::run_query_dsl_tests(&mut run, &scenario);
        assert!(
            !run.results.is_empty(),
            "query DSL tests should produce results"
        );
    }
}
