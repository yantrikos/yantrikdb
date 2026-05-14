//! CK-5 Benchmark Suite — Generative Understanding Primitives
//!
//! Categories 27–33, covering the seven CK-5 subsystems:
//!   27. Analogical Reasoning
//!   28. Schema Induction
//!   29. Episodic Narrative Memory
//!   30. Counterfactual Simulator
//!   31. Probabilistic Belief Network
//!   32. Experience Replay / Dream Consolidation
//!   33. Perspective Engine
//!
//! Each category runs 4–7 persona-driven tests that exercise the core
//! pure-function APIs without requiring a database.

use std::collections::HashMap;
use std::time::Instant;

use super::benchmark::{BenchResult, BenchRun, PersonaScenario};
use super::state::*;

// ── Category 27: Analogical Reasoning ────────────────────────────────

pub fn run_analogy_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::analogy::*;

    let persona = Some(scenario.name.clone());

    // Build subgraph groups from scenario nodes by metadata domain.
    let groups = build_subgraph_groups(&scenario.nodes, &scenario.edges);

    // Test 1: Subgraph group construction.
    let start = Instant::now();
    let group_count = groups.len();
    run.add(BenchResult {
        category: "analogy".into(),
        test_name: "subgraph_grouping".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("groups".into()),
        metric_value: Some(group_count as f64),
        details: Some(format!(
            "nodes={} edges={} groups={}",
            scenario.nodes.len(),
            scenario.edges.len(),
            group_count,
        )),
    });

    // Test 2: Analogical opportunity detection.
    let start = Instant::now();
    let node_refs: Vec<&CognitiveNode> = scenario.nodes.iter().collect();
    let opportunities = detect_analogical_opportunities(&node_refs, &scenario.edges, &groups, 0.3);
    run.add(BenchResult {
        category: "analogy".into(),
        test_name: "opportunity_detection".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("opportunities".into()),
        metric_value: Some(opportunities.len() as f64),
        details: None,
    });

    // Test 3: Find analogies with query.
    let start = Instant::now();
    let mut mappings_found = 0;
    let mut all_quality_valid = true;
    if !groups.is_empty() {
        let query = AnalogicalQuery {
            source_subgraph: groups[0].nodes.iter().map(|n| n.id).collect(),
            source_edges: groups[0].edges.clone(),
            source_domain: groups[0].domain.clone(),
            search_scope: AnalogyScope::CrossDomain,
            min_relational_depth: 1,
            max_results: 10,
        };
        let mappings = find_analogies(&query, &node_refs, &groups);
        for m in &mappings {
            if m.structural_similarity < 0.0 || m.structural_similarity > 1.0 {
                all_quality_valid = false;
            }
        }
        mappings_found = mappings.len();
    }
    run.add(BenchResult {
        category: "analogy".into(),
        test_name: "find_analogies".into(),
        persona: persona.clone(),
        passed: all_quality_valid,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("mappings".into()),
        metric_value: Some(mappings_found as f64),
        details: None,
    });

    // Test 4: Analogy store serialization roundtrip.
    let start = Instant::now();
    let store = AnalogyStore::default();
    let json = serde_json::to_string(&store).unwrap_or_default();
    let restored: Result<AnalogyStore, _> = serde_json::from_str(&json);
    run.add(BenchResult {
        category: "analogy".into(),
        test_name: "store_roundtrip".into(),
        persona: persona.clone(),
        passed: restored.is_ok(),
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: None,
        metric_value: None,
        details: None,
    });

    // Test 5: Strength decay on empty store.
    let start = Instant::now();
    let now_ms = (now_secs() * 1000.0) as u64;
    let mut store_mut = AnalogyStore::default();
    let report = analogy_strength_decay(&mut store_mut, now_ms, 86_400_000 * 30);
    run.add(BenchResult {
        category: "analogy".into(),
        test_name: "strength_decay".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("remaining".into()),
        metric_value: Some(report.remaining as f64),
        details: Some(format!(
            "pruned_quality={} pruned_stale={}",
            report.pruned_low_quality, report.pruned_stale,
        )),
    });
}

/// Build subgraph groups by clustering nodes by their metadata "domain" tag.
fn build_subgraph_groups(
    nodes: &[CognitiveNode],
    edges: &[CognitiveEdge],
) -> Vec<super::analogy::SubgraphGroup> {
    let mut domain_map: HashMap<String, Vec<&CognitiveNode>> = HashMap::new();
    for node in nodes {
        let domain = node
            .metadata
            .get("domain")
            .and_then(|v| v.as_str())
            .unwrap_or("general")
            .to_string();
        domain_map.entry(domain).or_default().push(node);
    }

    domain_map
        .into_iter()
        .map(|(domain, domain_nodes)| {
            let node_id_set: std::collections::HashSet<NodeId> =
                domain_nodes.iter().map(|n| n.id).collect();
            let domain_edges: Vec<CognitiveEdge> = edges
                .iter()
                .filter(|e| node_id_set.contains(&e.src) && node_id_set.contains(&e.dst))
                .cloned()
                .collect();
            super::analogy::SubgraphGroup {
                domain,
                nodes: domain_nodes.into_iter().cloned().collect(),
                edges: domain_edges,
            }
        })
        .collect()
}

// ── Category 28: Schema Induction ────────────────────────────────────

pub fn run_schema_induction_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::schema_induction::*;

    let persona = Some(scenario.name.clone());
    let now_ms = (now_secs() * 1000.0) as u64;

    // Build episodes from persona action-schema and task nodes.
    let episodes = build_episodes_from_scenario(&scenario.nodes, now_ms);

    // Test 1: Episode extraction.
    let start = Instant::now();
    let ep_count = episodes.len();
    run.add(BenchResult {
        category: "schema_induction".into(),
        test_name: "episode_extraction".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("episodes".into()),
        metric_value: Some(ep_count as f64),
        details: None,
    });

    // Test 2: Observe episodes into schema store.
    let start = Instant::now();
    let mut store = SchemaStore::default();
    for ep in &episodes {
        observe_episode(ep, &mut store);
    }
    let schema_count = store.len();
    run.add(BenchResult {
        category: "schema_induction".into(),
        test_name: "episode_observation".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("schemas_induced".into()),
        metric_value: Some(schema_count as f64),
        details: Some(format!("episodes_observed={}", ep_count)),
    });

    // Test 3: Schema matching against context.
    let start = Instant::now();
    let mut all_scores_bounded = true;
    let ctx = ContextSnapshot {
        node_kinds_present: scenario
            .nodes
            .iter()
            .map(|n| n.kind())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect(),
        edge_kinds_present: scenario
            .edges
            .iter()
            .map(|e| (e.kind, Direction::Any))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect(),
        attributes: HashMap::new(),
        belief_confidences: scenario
            .nodes
            .iter()
            .filter(|n| n.kind() == NodeKind::Belief)
            .map(|n| (n.label.clone(), n.attrs.confidence))
            .collect(),
        now_ms,
    };
    let matches = match_schemas(&ctx, &store);
    for (_, score) in &matches {
        if *score < 0.0 || *score > 1.0 {
            all_scores_bounded = false;
        }
    }
    let total_matches = matches.len();
    run.add(BenchResult {
        category: "schema_induction".into(),
        test_name: "schema_matching".into(),
        persona: persona.clone(),
        passed: all_scores_bounded,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("total_matches".into()),
        metric_value: Some(total_matches as f64),
        details: None,
    });

    // Test 4: Schema maintenance.
    let start = Instant::now();
    let report = schema_maintenance(&mut store, now_ms, 86_400_000 * 90);
    let post_count = store.len();
    run.add(BenchResult {
        category: "schema_induction".into(),
        test_name: "schema_maintenance".into(),
        persona: persona.clone(),
        passed: post_count <= schema_count,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("pruned".into()),
        metric_value: Some(report.pruned_low_confidence as f64),
        details: Some(format!(
            "before={} after={} merged={}",
            schema_count, post_count, report.merged,
        )),
    });

    // Test 5: Store serialization roundtrip.
    let start = Instant::now();
    let json = serde_json::to_string(&store).unwrap_or_default();
    let restored: Result<SchemaStore, _> = serde_json::from_str(&json);
    run.add(BenchResult {
        category: "schema_induction".into(),
        test_name: "store_roundtrip".into(),
        persona: persona.clone(),
        passed: restored.is_ok(),
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("json_bytes".into()),
        metric_value: Some(json.len() as f64),
        details: None,
    });
}

/// Build synthetic EpisodeData from persona action-schema and task nodes.
fn build_episodes_from_scenario(
    nodes: &[CognitiveNode],
    now_ms: u64,
) -> Vec<super::schema_induction::EpisodeData> {
    use super::schema_induction::{EpisodeData, SchemaCondition};

    let mut episodes = Vec::new();

    for (i, node) in nodes.iter().enumerate() {
        if node.kind() == NodeKind::ActionSchema || node.kind() == NodeKind::Task {
            episodes.push(EpisodeData {
                episode_id: node.id,
                conditions: vec![SchemaCondition::NodeKindPresent(node.kind())],
                action_type: node.label.clone(),
                outcome_positive: i % 3 != 0,
                outcome_description: format!("outcome for {}", node.label),
                outcome_valence: if i % 3 != 0 { 0.7 } else { -0.2 },
                timestamp_ms: now_ms - (i as u64) * 3_600_000,
            });
        }
    }

    episodes
}

// ── Category 29: Episodic Narrative Memory ────────────────────────────

pub fn run_narrative_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::narrative::*;

    let persona = Some(scenario.name.clone());
    let now_ms = (now_secs() * 1000.0) as u64;

    // Build narrative episodes from persona episode nodes.
    let episodes = build_narrative_episodes(&scenario.nodes, now_ms);

    // Test 1: Episode construction.
    let start = Instant::now();
    let ep_count = episodes.len();
    run.add(BenchResult {
        category: "narrative".into(),
        test_name: "episode_construction".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("episodes".into()),
        metric_value: Some(ep_count as f64),
        details: None,
    });

    // Test 2: Arc assignment.
    let start = Instant::now();
    let mut timeline = AutobiographicalTimeline::default();
    let mut arc_ids = Vec::new();
    for ep in &episodes {
        let arc_id = assign_to_arc(ep, &mut timeline);
        arc_ids.push(arc_id);
    }
    let unique_arcs: std::collections::HashSet<_> = arc_ids.iter().collect();
    run.add(BenchResult {
        category: "narrative".into(),
        test_name: "arc_assignment".into(),
        persona: persona.clone(),
        passed: !arc_ids.is_empty() || episodes.is_empty(),
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("unique_arcs".into()),
        metric_value: Some(unique_arcs.len() as f64),
        details: Some(format!("episodes={} arcs={}", ep_count, unique_arcs.len())),
    });

    // Test 3: Arc health check.
    let start = Instant::now();
    let alerts = arc_health_check(&timeline, now_ms);
    run.add(BenchResult {
        category: "narrative".into(),
        test_name: "arc_health_check".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("alerts".into()),
        metric_value: Some(alerts.len() as f64),
        details: None,
    });

    // Test 4: Arc summary generation.
    let start = Instant::now();
    let mut summaries_generated = 0;
    for arc in &timeline.arcs {
        let summary = generate_arc_summary(arc);
        if !summary.is_empty() {
            summaries_generated += 1;
        }
    }
    run.add(BenchResult {
        category: "narrative".into(),
        test_name: "arc_summary_generation".into(),
        persona: persona.clone(),
        passed: summaries_generated == timeline.arcs.len(),
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("summaries".into()),
        metric_value: Some(summaries_generated as f64),
        details: None,
    });

    // Test 5: Timeline query — active arcs.
    let start = Instant::now();
    let result = query_timeline(&timeline, &NarrativeQuery::ActiveArcs);
    run.add(BenchResult {
        category: "narrative".into(),
        test_name: "query_active_arcs".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("active_arcs".into()),
        metric_value: Some(result.arcs.len() as f64),
        details: None,
    });

    // Test 6: Timeline serialization roundtrip.
    let start = Instant::now();
    let json = serde_json::to_string(&timeline).unwrap_or_default();
    let restored: Result<AutobiographicalTimeline, _> = serde_json::from_str(&json);
    run.add(BenchResult {
        category: "narrative".into(),
        test_name: "timeline_roundtrip".into(),
        persona: persona.clone(),
        passed: restored.is_ok(),
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("json_bytes".into()),
        metric_value: Some(json.len() as f64),
        details: None,
    });
}

/// Build narrative episodes from persona Episode-type nodes.
fn build_narrative_episodes(
    nodes: &[CognitiveNode],
    now_ms: u64,
) -> Vec<super::narrative::NarrativeEpisode> {
    use super::narrative::NarrativeEpisode;

    let mut episodes = Vec::new();

    for (i, node) in nodes.iter().enumerate() {
        if node.kind() == NodeKind::Episode {
            let domain = node
                .metadata
                .get("domain")
                .and_then(|v| v.as_str())
                .unwrap_or("general")
                .to_string();
            episodes.push(NarrativeEpisode {
                episode_id: node.id,
                summary: node.label.clone(),
                participants: vec![node.id],
                domains: vec![domain],
                sentiment: node.attrs.valence,
                timestamp_ms: now_ms - (i as u64) * 86_400_000,
                related_goal: None,
            });
        }
    }

    episodes
}

// ── Category 30: Counterfactual Simulator ─────────────────────────────

pub fn run_counterfactual_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::causal::*;
    use super::counterfactual::*;

    let persona = Some(scenario.name.clone());
    let now = now_secs();

    // Build a causal store from scenario edges.
    let mut store = CausalStore::new();
    seed_causal_from_graph(&mut store, &scenario.edges, now);
    let config = CounterfactualConfig::default();

    // Test 1: Config defaults.
    let start = Instant::now();
    let horizon_ok = config.max_horizon > 0 && config.max_horizon <= 20;
    let decay_ok = config.confidence_decay > 0.0 && config.confidence_decay < 1.0;
    run.add(BenchResult {
        category: "counterfactual".into(),
        test_name: "config_defaults".into(),
        persona: persona.clone(),
        passed: horizon_ok && decay_ok,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("max_horizon".into()),
        metric_value: Some(config.max_horizon as f64),
        details: Some(format!(
            "horizon={} conf_decay={:.2} hop_decay={:.2}",
            config.max_horizon, config.confidence_decay, config.hop_decay,
        )),
    });

    // Test 2: Simulate counterfactual queries.
    let start = Instant::now();
    let mut simulations_run = 0;
    let mut all_results_valid = true;
    let edge_list = store.edges();
    for edge in edge_list.iter().take(5) {
        let query = CounterfactualQuery {
            intervention: Intervention::ForceActivation {
                node: edge.cause.clone(),
                strength: 1.0,
            },
            observation: None,
            horizon_steps: 3,
            query_type: CounterfactualType::WhatIf,
        };
        let result = simulate_counterfactual(&query, &store, &config);
        if result.confidence < 0.0 || result.confidence > 1.0 {
            all_results_valid = false;
        }
        simulations_run += 1;
    }
    run.add(BenchResult {
        category: "counterfactual".into(),
        test_name: "simulate_counterfactual".into(),
        persona: persona.clone(),
        passed: all_results_valid,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("simulations".into()),
        metric_value: Some(simulations_run as f64),
        details: None,
    });

    // Test 3: Sensitivity analysis.
    let start = Instant::now();
    let mut sensitivity_entries = 0;
    if let Some(edge) = edge_list.first() {
        let query = CounterfactualQuery {
            intervention: Intervention::ForceActivation {
                node: edge.cause.clone(),
                strength: 1.0,
            },
            observation: None,
            horizon_steps: 3,
            query_type: CounterfactualType::WhatIf,
        };
        let entries = sensitivity_analysis(&query, &store, &config);
        sensitivity_entries = entries.len();
    }
    run.add(BenchResult {
        category: "counterfactual".into(),
        test_name: "sensitivity_analysis".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("entries".into()),
        metric_value: Some(sensitivity_entries as f64),
        details: None,
    });

    // Test 4: Compare alternatives.
    let start = Instant::now();
    let mut comparison_valid = true;
    if edge_list.len() >= 2 {
        let diff = compare_alternatives(&edge_list[0].cause, &edge_list[1].cause, &store, &config);
        if diff.is_nan() {
            comparison_valid = false;
        }
    }
    run.add(BenchResult {
        category: "counterfactual".into(),
        test_name: "compare_alternatives".into(),
        persona: persona.clone(),
        passed: comparison_valid,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: None,
        metric_value: None,
        details: Some(format!("edges_available={}", edge_list.len())),
    });

    // Test 5: Config serialization roundtrip.
    let start = Instant::now();
    let json = serde_json::to_string(&config).unwrap_or_default();
    let restored: Result<CounterfactualConfig, _> = serde_json::from_str(&json);
    run.add(BenchResult {
        category: "counterfactual".into(),
        test_name: "config_roundtrip".into(),
        persona: persona.clone(),
        passed: restored.is_ok(),
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: None,
        metric_value: None,
        details: None,
    });
}

/// Seed causal store from cognitive graph edges that imply causality.
fn seed_causal_from_graph(
    store: &mut super::causal::CausalStore,
    edges: &[CognitiveEdge],
    now: f64,
) -> usize {
    use super::causal::*;

    let mut count = 0;
    for edge in edges {
        let strength = match edge.kind {
            CognitiveEdgeKind::Causes => 0.8,
            CognitiveEdgeKind::Predicts => 0.6,
            CognitiveEdgeKind::Triggers => 0.7,
            CognitiveEdgeKind::Prevents => -0.5,
            _ => continue,
        };
        let cause = CausalNode::GraphNode(edge.src);
        let effect = CausalNode::GraphNode(edge.dst);
        let trace = CausalTrace {
            evidence: vec![CausalEvidence::TemporalPrecedence {
                co_occurrences: 5,
                avg_lag_secs: 60.0,
                lag_stddev_secs: 10.0,
            }],
            primary_method: DiscoveryMethod::TemporalAssociation,
            summary: format!("{:?} edge", edge.kind),
        };
        let ce = CausalEdge {
            cause,
            effect,
            strength,
            confidence: 0.7,
            observation_count: 5,
            intervention_count: 0,
            non_occurrence_count: 1,
            median_lag_secs: 60.0,
            lag_iqr_secs: 20.0,
            context_strengths: Vec::new(),
            trace,
            created_at: now,
            updated_at: now,
            stage: CausalStage::Established,
        };
        store.upsert(ce);
        count += 1;
    }
    count
}

// ── Category 31: Probabilistic Belief Network ─────────────────────────

pub fn run_belief_network_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::belief_network::*;

    let persona = Some(scenario.name.clone());

    // Build a belief network from persona belief nodes and edges.
    let (network, var_ids) = build_belief_network_from_scenario(&scenario.nodes, &scenario.edges);

    // Test 1: Network construction.
    let start = Instant::now();
    let var_count = network.variable_count();
    let fac_count = network.factor_count();
    run.add(BenchResult {
        category: "belief_network".into(),
        test_name: "network_construction".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("variables".into()),
        metric_value: Some(var_count as f64),
        details: Some(format!("vars={} factors={}", var_count, fac_count)),
    });

    // Test 2: Belief propagation.
    let start = Instant::now();
    let mut net = network.clone();
    let config = BPConfig::default();
    let bp_result = loopy_belief_propagation(&mut net, &config);
    run.add(BenchResult {
        category: "belief_network".into(),
        test_name: "belief_propagation".into(),
        persona: persona.clone(),
        passed: bp_result.converged || var_count == 0,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("iterations".into()),
        metric_value: Some(bp_result.iterations as f64),
        details: Some(format!(
            "converged={} max_change={:.6}",
            bp_result.converged, bp_result.max_change,
        )),
    });

    // Test 3: Network diagnostics.
    let start = Instant::now();
    let health = network_diagnostics(&net);
    run.add(BenchResult {
        category: "belief_network".into(),
        test_name: "network_diagnostics".into(),
        persona: persona.clone(),
        passed: health.healthy || var_count == 0,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("avg_confidence".into()),
        metric_value: Some(health.avg_confidence),
        details: Some(format!(
            "components={} extreme_priors={} instabilities={}",
            health.components,
            health.extreme_priors.len(),
            health.potential_instabilities.len(),
        )),
    });

    // Test 4: Information gain.
    let start = Instant::now();
    if var_ids.len() >= 2 {
        let mut net2 = network.clone();
        let ig = information_gain(&mut net2, var_ids[0], var_ids[1], &config);
        run.add(BenchResult {
            category: "belief_network".into(),
            test_name: "information_gain".into(),
            persona: persona.clone(),
            passed: ig.is_finite(),
            duration_us: start.elapsed().as_micros() as f64,
            metric_name: Some("info_gain".into()),
            metric_value: Some(ig),
            details: None,
        });
    } else {
        run.add(BenchResult {
            category: "belief_network".into(),
            test_name: "information_gain".into(),
            persona: persona.clone(),
            passed: true,
            duration_us: start.elapsed().as_micros() as f64,
            metric_name: None,
            metric_value: None,
            details: Some("insufficient_variables".into()),
        });
    }

    // Test 5: Network serialization roundtrip.
    let start = Instant::now();
    let json = serde_json::to_string(&network).unwrap_or_default();
    let restored: Result<BeliefNetwork, _> = serde_json::from_str(&json);
    let mut roundtrip_ok = restored.is_ok();
    if let Ok(mut restored_net) = restored {
        restored_net.rebuild_indices();
        roundtrip_ok = restored_net.variable_count() == var_count;
    }
    run.add(BenchResult {
        category: "belief_network".into(),
        test_name: "network_roundtrip".into(),
        persona: persona.clone(),
        passed: roundtrip_ok,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("json_bytes".into()),
        metric_value: Some(json.len() as f64),
        details: None,
    });
}

/// Build a BeliefNetwork from persona belief nodes and edges.
fn build_belief_network_from_scenario(
    nodes: &[CognitiveNode],
    edges: &[CognitiveEdge],
) -> (
    super::belief_network::BeliefNetwork,
    Vec<super::belief_network::VariableId>,
) {
    use super::belief_network::*;

    // Extract beliefs: (node_id, label, log_odds from confidence).
    let mut beliefs: Vec<(NodeId, String, f64)> = Vec::new();
    for node in nodes {
        if node.kind() == NodeKind::Belief {
            let p = node.attrs.confidence.clamp(0.01, 0.99);
            let log_odds = (p / (1.0 - p)).ln();
            beliefs.push((node.id, node.label.clone(), log_odds));
        }
    }

    let belief_id_set: std::collections::HashSet<NodeId> =
        beliefs.iter().map(|(id, _, _)| *id).collect();

    let mut edge_data: Vec<(NodeId, NodeId, EdgeRelation, f64)> = Vec::new();
    for edge in edges {
        if belief_id_set.contains(&edge.src) && belief_id_set.contains(&edge.dst) {
            let relation = match edge.kind {
                CognitiveEdgeKind::Supports => EdgeRelation::Supports,
                CognitiveEdgeKind::Contradicts => EdgeRelation::Contradicts,
                CognitiveEdgeKind::Causes => EdgeRelation::Causes,
                CognitiveEdgeKind::Predicts => EdgeRelation::Predicts,
                _ => EdgeRelation::Correlates,
            };
            edge_data.push((edge.src, edge.dst, relation, edge.weight));
        }
    }

    let belief_refs: Vec<(NodeId, &str, f64)> = beliefs
        .iter()
        .map(|(id, label, lo)| (*id, label.as_str(), *lo))
        .collect();

    let network = build_network_from_edges(&belief_refs, &edge_data);
    let var_ids: Vec<VariableId> = (0..beliefs.len()).map(|i| VariableId(i as u64)).collect();

    (network, var_ids)
}

// ── Category 32: Experience Replay / Dream Consolidation ──────────────

pub fn run_replay_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::replay::*;

    let persona = Some(scenario.name.clone());
    let now_ms = (now_secs() * 1000.0) as u64;

    // Test 1: Engine initialization.
    let start = Instant::now();
    let mut engine = ReplayEngine::new();
    let summary = replay_summary(&engine);
    run.add(BenchResult {
        category: "replay".into(),
        test_name: "engine_init".into(),
        persona: persona.clone(),
        passed: summary.buffer_size == 0 && summary.total_replays == 0,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: None,
        metric_value: None,
        details: None,
    });

    // Test 2: Add experiences to buffer.
    let start = Instant::now();
    let mut attempted = 0;
    for (i, node) in scenario.nodes.iter().enumerate().take(20) {
        if node.kind() == NodeKind::Episode || node.kind() == NodeKind::Task {
            let domain = node
                .metadata
                .get("domain")
                .and_then(|v| v.as_str())
                .unwrap_or("general")
                .to_string();
            let action = ActionRecord {
                description: node.label.clone(),
                domain: domain.clone(),
                involved_nodes: vec![node.id],
            };
            let outcome = OutcomeData {
                utility: 0.3 + (i as f64 % 7.0) * 0.1,
                expected: i % 3 != 0,
                domains: vec![domain],
                affected_nodes: vec![node.id],
            };
            add_to_buffer(
                &mut engine,
                node.id,
                0.5, // expected utility
                action,
                outcome,
                now_ms - (i as u64) * 600_000,
            );
            attempted += 1;
        }
    }
    let post_summary = replay_summary(&engine);
    run.add(BenchResult {
        category: "replay".into(),
        test_name: "buffer_population".into(),
        persona: persona.clone(),
        passed: post_summary.buffer_size <= attempted,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("buffer_size".into()),
        metric_value: Some(post_summary.buffer_size as f64),
        details: Some(format!("attempted={}", attempted)),
    });

    // Test 3: Should-replay check.
    let start = Instant::now();
    let should = should_replay(&engine, now_ms);
    run.add(BenchResult {
        category: "replay".into(),
        test_name: "should_replay_check".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("should_replay".into()),
        metric_value: Some(if should { 1.0 } else { 0.0 }),
        details: None,
    });

    // Test 4: Reprioritize buffer.
    let start = Instant::now();
    reprioritize_buffer(&mut engine, now_ms);
    run.add(BenchResult {
        category: "replay".into(),
        test_name: "reprioritize".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("buffer_size".into()),
        metric_value: Some(post_summary.buffer_size as f64),
        details: None,
    });

    // Test 5: Dream cycle.
    let start = Instant::now();
    let beliefs: Vec<(NodeId, f64)> = scenario
        .nodes
        .iter()
        .filter(|n| n.kind() == NodeKind::Belief)
        .map(|n| (n.id, n.attrs.confidence))
        .collect();
    let report = run_replay_cycle(&mut engine, &beliefs, now_ms);
    run.add(BenchResult {
        category: "replay".into(),
        test_name: "dream_cycle".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("replays_executed".into()),
        metric_value: Some(report.replays_executed as f64),
        details: Some(format!(
            "beliefs={} causal={} assoc={}",
            report.beliefs_updated, report.causal_updates, report.new_associations,
        )),
    });

    // Test 6: Engine serialization roundtrip.
    let start = Instant::now();
    let json = serde_json::to_string(&engine).unwrap_or_default();
    let restored: Result<ReplayEngine, _> = serde_json::from_str(&json);
    run.add(BenchResult {
        category: "replay".into(),
        test_name: "engine_roundtrip".into(),
        persona: persona.clone(),
        passed: restored.is_ok(),
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("json_bytes".into()),
        metric_value: Some(json.len() as f64),
        details: None,
    });
}

// ── Category 33: Perspective Engine ───────────────────────────────────

pub fn run_perspective_tests(run: &mut BenchRun, scenario: &PersonaScenario) {
    use super::perspective::*;

    let persona = Some(scenario.name.clone());
    let now_ms = (now_secs() * 1000.0) as u64;

    // Test 1: Preset creation.
    let start = Instant::now();
    let mut store = PerspectiveStore::new();
    let presets = ["creative", "deadline", "reflective"];
    let mut preset_ids = Vec::new();
    for name in &presets {
        let id = store.alloc_id();
        if let Some(p) = create_preset(name, id, now_ms) {
            store.insert(p);
            preset_ids.push(id);
        }
    }
    run.add(BenchResult {
        category: "perspective".into(),
        test_name: "preset_creation".into(),
        persona: persona.clone(),
        passed: preset_ids.len() == 3,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("presets_created".into()),
        metric_value: Some(preset_ids.len() as f64),
        details: None,
    });

    // Test 2: Perspective activation and stack.
    let start = Instant::now();
    let mut stack = PerspectiveStack::new();
    for &id in &preset_ids {
        activate_perspective(&mut stack, id, &store);
    }
    let stack_depth = stack.depth();
    run.add(BenchResult {
        category: "perspective".into(),
        test_name: "stack_activation".into(),
        persona: persona.clone(),
        passed: stack_depth == preset_ids.len(),
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("stack_depth".into()),
        metric_value: Some(stack_depth as f64),
        details: None,
    });

    // Test 3: Salience resolution.
    let start = Instant::now();
    let mut all_valid = true;
    for node in &scenario.nodes {
        let domain = node.metadata.get("domain").and_then(|v| v.as_str());
        let tags: Vec<String> = node
            .metadata
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let resolved = resolve_salience(&stack, node.id, node.kind(), domain, &tags, 1.0, &store);
        if resolved < 0.0 || resolved > 10.0 {
            all_valid = false;
        }
    }
    run.add(BenchResult {
        category: "perspective".into(),
        test_name: "salience_resolution".into(),
        persona: persona.clone(),
        passed: all_valid,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("nodes_resolved".into()),
        metric_value: Some(scenario.nodes.len() as f64),
        details: None,
    });

    // Test 4: Edge weight resolution.
    let start = Instant::now();
    let mut edge_resolutions = 0;
    let mut all_edge_valid = true;
    for edge in &scenario.edges {
        let resolved = resolve_edge_weight(&stack, edge.kind, edge.weight, &store);
        if resolved.is_nan() || resolved.is_infinite() {
            all_edge_valid = false;
        }
        edge_resolutions += 1;
    }
    run.add(BenchResult {
        category: "perspective".into(),
        test_name: "edge_weight_resolution".into(),
        persona: persona.clone(),
        passed: all_edge_valid,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("edges_resolved".into()),
        metric_value: Some(edge_resolutions as f64),
        details: None,
    });

    // Test 5: Cognitive style blending.
    let start = Instant::now();
    let style = resolve_cognitive_style(&stack, &store);
    let style_valid = style.exploration_vs_exploitation >= 0.0
        && style.exploration_vs_exploitation <= 1.0
        && style.risk_tolerance >= 0.0
        && style.risk_tolerance <= 1.0;
    run.add(BenchResult {
        category: "perspective".into(),
        test_name: "cognitive_style_blend".into(),
        persona: persona.clone(),
        passed: style_valid,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("exploration".into()),
        metric_value: Some(style.exploration_vs_exploitation),
        details: Some(format!(
            "risk={:.2} abstraction={:.2} social={:.2}",
            style.risk_tolerance, style.abstraction_level, style.social_weight,
        )),
    });

    // Test 6: Conflict detection.
    let start = Instant::now();
    let conflicts = perspective_conflict_check(&stack, &store);
    run.add(BenchResult {
        category: "perspective".into(),
        test_name: "conflict_detection".into(),
        persona: persona.clone(),
        passed: true,
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("conflicts".into()),
        metric_value: Some(conflicts.len() as f64),
        details: None,
    });

    // Test 7: Store serialization roundtrip.
    let start = Instant::now();
    let json = serde_json::to_string(&store).unwrap_or_default();
    let restored: Result<PerspectiveStore, _> = serde_json::from_str(&json);
    run.add(BenchResult {
        category: "perspective".into(),
        test_name: "store_roundtrip".into(),
        persona: persona.clone(),
        passed: restored.is_ok(),
        duration_us: start.elapsed().as_micros() as f64,
        metric_name: Some("json_bytes".into()),
        metric_value: Some(json.len() as f64),
        details: None,
    });
}
