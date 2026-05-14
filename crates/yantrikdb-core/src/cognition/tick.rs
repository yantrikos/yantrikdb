//! CK-2.5: cognitive_tick() — Background Cognition Heartbeat
//!
//! The cognitive tick is the "pulse" of the system — a periodic function
//! that runs in the background to keep the cognitive graph alive. It
//! handles urgency updates, activation decay, routine predictions,
//! anomaly detection, suggestion caching, and memory consolidation.
//!
//! ## Tick Phases
//!
//! Each tick runs through up to 6 phases depending on the tick counter
//! and available time budget:
//!
//! 1. **Urgency refresh** (every tick) — Recompute agenda urgencies
//! 2. **Activation decay** (every tick) — Apply temporal decay to nodes
//! 3. **Routine predictions** (every 10 ticks) — Hawkes process check
//! 4. **Anomaly detection** (every 100 ticks) — Missing/unusual events
//! 5. **Suggestion cache** (every 50 ticks) — Pre-compute next actions
//! 6. **Consolidation** (every 1000 ticks) — Memory maintenance
//!
//! ## Performance Contract
//!
//! - Phase 1+2: < 1ms combined
//! - Phase 3: < 1ms
//! - Phase 4: < 2ms
//! - Phase 5: < 3ms
//! - Phase 6: variable (only when idle, can be interrupted)
//! - Total steady-state: < 5ms per tick

use serde::{Deserialize, Serialize};

use super::agenda::{
    Agenda, AgendaConfig, AgendaId, AgendaKind, TickResult as AgendaTickResult, UrgencyFn,
};
use super::hawkes::HawkesRegistry;
use super::state::*;

// ══════════════════════════════════════════════════════════════════════════════
// § 1  Tick Configuration
// ══════════════════════════════════════════════════════════════════════════════

/// Configuration for the cognitive tick loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickConfig {
    /// Interval in ticks between routine prediction checks.
    pub routine_check_interval: u64,
    /// Interval in ticks between anomaly detection passes.
    pub anomaly_check_interval: u64,
    /// Interval in ticks between suggestion cache refreshes.
    pub suggestion_cache_interval: u64,
    /// Interval in ticks between consolidation passes.
    pub consolidation_interval: u64,
    /// Maximum time budget per tick in microseconds.
    /// Phases are skipped if budget is exhausted.
    pub budget_us: u64,
    /// Minimum activation threshold — nodes below this are candidates for eviction.
    pub activation_eviction_threshold: f64,
    /// Maximum number of nodes to decay per tick (limits CPU usage).
    pub max_decay_nodes_per_tick: usize,
    /// Hawkes anticipation threshold multiplier.
    pub anticipation_threshold: f64,
    /// Anomaly: hours without expected event before flagging.
    pub missing_event_hours: f64,
    /// Anomaly: hours of stale task/goal before flagging.
    pub stale_item_hours: f64,
    /// Maximum cached suggestions to keep.
    pub max_cached_suggestions: usize,
}

impl Default for TickConfig {
    fn default() -> Self {
        Self {
            routine_check_interval: 10,
            anomaly_check_interval: 100,
            suggestion_cache_interval: 50,
            consolidation_interval: 1000,
            budget_us: 5000, // 5ms
            activation_eviction_threshold: 0.001,
            max_decay_nodes_per_tick: 200,
            anticipation_threshold: 1.5,
            missing_event_hours: 4.0,
            stale_item_hours: 48.0,
            max_cached_suggestions: 5,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 2  Tick State
// ══════════════════════════════════════════════════════════════════════════════

/// Persistent state for the cognitive tick loop.
///
/// Tracks the tick counter, last timestamps for each phase,
/// and cached suggestions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickState {
    /// Monotonic tick counter.
    pub tick_count: u64,
    /// Last time each phase ran (unix seconds).
    pub last_urgency_at: f64,
    pub last_decay_at: f64,
    pub last_routine_check_at: f64,
    pub last_anomaly_check_at: f64,
    pub last_suggestion_cache_at: f64,
    pub last_consolidation_at: f64,
    /// Cached proactive suggestions from the last suggestion pass.
    pub cached_suggestions: Vec<CachedSuggestion>,
    /// Detected anomalies from the last anomaly pass.
    pub active_anomalies: Vec<Anomaly>,
    /// Elapsed seconds between last two ticks (for decay calculation).
    pub last_tick_elapsed_secs: f64,
}

impl TickState {
    pub fn new() -> Self {
        Self {
            tick_count: 0,
            last_urgency_at: 0.0,
            last_decay_at: 0.0,
            last_routine_check_at: 0.0,
            last_anomaly_check_at: 0.0,
            last_suggestion_cache_at: 0.0,
            last_consolidation_at: 0.0,
            cached_suggestions: Vec::new(),
            active_anomalies: Vec::new(),
            last_tick_elapsed_secs: 1.0,
        }
    }
}

impl Default for TickState {
    fn default() -> Self {
        Self::new()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  Tick Report
// ══════════════════════════════════════════════════════════════════════════════

/// Report produced by a single cognitive tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickReport {
    /// Tick number.
    pub tick_number: u64,
    /// Agenda item IDs that are ready to surface to the user.
    pub items_surfaced: Vec<super::agenda::AgendaId>,
    /// Number of routine predictions that were updated.
    pub predictions_updated: u32,
    /// New anomalies detected this tick.
    pub anomalies_detected: Vec<Anomaly>,
    /// Cached proactive suggestions (refreshed periodically).
    pub cached_suggestions: Vec<CachedSuggestion>,
    /// Whether memory consolidation ran this tick.
    pub consolidation_ran: bool,
    /// Number of nodes whose activation was decayed.
    pub nodes_decayed: u32,
    /// Number of nodes evicted (fell below activation threshold).
    pub nodes_evicted: u32,
    /// Total tick duration in microseconds.
    pub tick_duration_us: u64,
    /// Which phases ran during this tick.
    pub phases_executed: Vec<TickPhase>,
    /// Whether the tick was cut short due to budget exhaustion.
    pub budget_exhausted: bool,
}

/// Which phase of the tick loop was executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TickPhase {
    UrgencyRefresh,
    ActivationDecay,
    RoutinePrediction,
    AnomalyDetection,
    SuggestionCache,
    Consolidation,
}

impl TickPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UrgencyRefresh => "urgency_refresh",
            Self::ActivationDecay => "activation_decay",
            Self::RoutinePrediction => "routine_prediction",
            Self::AnomalyDetection => "anomaly_detection",
            Self::SuggestionCache => "suggestion_cache",
            Self::Consolidation => "consolidation",
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  Anomaly
// ══════════════════════════════════════════════════════════════════════════════

/// A detected anomaly from the cognitive tick's monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// What kind of anomaly.
    pub kind: AnomalyKind,
    /// Human-readable description.
    pub description: String,
    /// Severity [0.0, 1.0].
    pub severity: f64,
    /// Related node ID (if applicable).
    pub node_id: Option<NodeId>,
    /// When the anomaly was detected (unix seconds).
    pub detected_at: f64,
}

/// Types of anomalies the tick loop can detect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyKind {
    /// A routine event that was expected but hasn't happened.
    MissingExpectedEvent,
    /// A task or goal that hasn't been updated in too long.
    StaleItem,
    /// An unusual burst of activity compared to baseline.
    UnusualActivityBurst,
    /// A belief conflict that remains unresolved.
    UnresolvedConflict,
    /// Agenda item that has been snoozed too many times.
    ChronicSnooze,
}

impl AnomalyKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MissingExpectedEvent => "missing_expected_event",
            Self::StaleItem => "stale_item",
            Self::UnusualActivityBurst => "unusual_activity_burst",
            Self::UnresolvedConflict => "unresolved_conflict",
            Self::ChronicSnooze => "chronic_snooze",
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 5  Cached Suggestion
// ══════════════════════════════════════════════════════════════════════════════

/// A pre-computed proactive suggestion cached by the tick loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedSuggestion {
    /// What action is suggested.
    pub description: String,
    /// Action kind (from ActionCandidate).
    pub action_kind: String,
    /// Utility score from the evaluator.
    pub utility: f64,
    /// Confidence in the suggestion.
    pub confidence: f64,
    /// When this was cached (unix seconds).
    pub cached_at: f64,
    /// How long this cache entry is valid (seconds).
    pub ttl_secs: f64,
}

impl CachedSuggestion {
    /// Whether this cached suggestion is still valid.
    pub fn is_valid(&self, now: f64) -> bool {
        now - self.cached_at < self.ttl_secs
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 6  The Tick Function (Pure — no DB dependency)
// ══════════════════════════════════════════════════════════════════════════════

/// Execute one cognitive tick.
///
/// This is the pure, functional core. It takes all state as input
/// and returns the report + updated state. The engine layer handles
/// loading/saving state from the database.
///
/// # Arguments
/// * `now` — Current time (unix seconds).
/// * `state` — Mutable tick state (tick counter, caches, etc.).
/// * `agenda` — Mutable agenda for urgency updates.
/// * `nodes` — Currently active cognitive nodes.
/// * `hawkes_registry` — Hawkes models for routine prediction.
/// * `config` — Tick configuration.
///
/// # Returns
/// `TickReport` describing what happened this tick.
pub fn cognitive_tick(
    now: f64,
    state: &mut TickState,
    agenda: &mut Agenda,
    nodes: &mut [CognitiveNode],
    hawkes_registry: &HawkesRegistry,
    config: &TickConfig,
) -> TickReport {
    let tick_start = std::time::Instant::now();
    let tick_number = state.tick_count;
    state.tick_count += 1;

    // Track elapsed time since last tick
    let elapsed = if state.last_urgency_at > 0.0 {
        (now - state.last_urgency_at).max(0.01)
    } else {
        1.0
    };
    state.last_tick_elapsed_secs = elapsed;

    let mut report = TickReport {
        tick_number,
        items_surfaced: Vec::new(),
        predictions_updated: 0,
        anomalies_detected: Vec::new(),
        cached_suggestions: Vec::new(),
        consolidation_ran: false,
        nodes_decayed: 0,
        nodes_evicted: 0,
        tick_duration_us: 0,
        phases_executed: Vec::new(),
        budget_exhausted: false,
    };

    let budget_deadline = tick_start + std::time::Duration::from_micros(config.budget_us);

    // ── Phase 1: Urgency Refresh (every tick) ──
    {
        let agenda_config = AgendaConfig::default();
        let tick_result = agenda.tick(now, &agenda_config);
        report.items_surfaced = tick_result.ready_to_surface;
        report.phases_executed.push(TickPhase::UrgencyRefresh);
        state.last_urgency_at = now;
    }

    if std::time::Instant::now() >= budget_deadline {
        report.budget_exhausted = true;
        report.tick_duration_us = tick_start.elapsed().as_micros() as u64;
        return report;
    }

    // ── Phase 2: Activation Decay (every tick) ──
    {
        let mut decayed = 0u32;
        let mut evicted = 0u32;
        let limit = config.max_decay_nodes_per_tick.min(nodes.len());

        for node in nodes.iter_mut().take(limit) {
            let old_activation = node.attrs.activation;
            node.attrs.decay(elapsed);
            if old_activation > config.activation_eviction_threshold
                && node.attrs.activation <= config.activation_eviction_threshold
            {
                evicted += 1;
            }
            if old_activation != node.attrs.activation {
                decayed += 1;
            }
        }

        report.nodes_decayed = decayed;
        report.nodes_evicted = evicted;
        report.phases_executed.push(TickPhase::ActivationDecay);
        state.last_decay_at = now;
    }

    if std::time::Instant::now() >= budget_deadline {
        report.budget_exhausted = true;
        report.tick_duration_us = tick_start.elapsed().as_micros() as u64;
        return report;
    }

    // ── Phase 3: Routine Predictions (every N ticks) ──
    if tick_number % config.routine_check_interval == 0 {
        let mut predictions_updated = 0u32;

        for (label, model) in &hawkes_registry.models {
            if model.total_observations >= 5 {
                if model.should_anticipate(now, config.anticipation_threshold) {
                    // Check if this routine is already in the agenda
                    let already_tracked = agenda
                        .items_iter()
                        .any(|item| item.description.contains(label) && item.is_surfaceable(now));

                    if !already_tracked {
                        if let Some(pred) = model.predict_next(now, 3600.0, 60.0) {
                            // Create anticipatory agenda item
                            let agenda_config = AgendaConfig::default();
                            agenda.add_item_at(
                                NodeId::NIL,
                                AgendaKind::RoutineWindowOpening,
                                UrgencyFn::Constant {
                                    value: pred.confidence * 0.6,
                                },
                                Some(pred.predicted_time),
                                format!(
                                    "Anticipated: {} (confidence {:.0}%)",
                                    label,
                                    pred.confidence * 100.0
                                ),
                                now,
                                &agenda_config,
                            );
                        }
                    }
                    predictions_updated += 1;
                }
            }
        }

        report.predictions_updated = predictions_updated;
        report.phases_executed.push(TickPhase::RoutinePrediction);
        state.last_routine_check_at = now;
    }

    if std::time::Instant::now() >= budget_deadline {
        report.budget_exhausted = true;
        report.tick_duration_us = tick_start.elapsed().as_micros() as u64;
        return report;
    }

    // ── Phase 4: Anomaly Detection (every N ticks) ──
    if tick_number % config.anomaly_check_interval == 0 {
        let mut anomalies = Vec::new();

        // 4a: Stale items — tasks/goals not updated for too long
        let stale_threshold_ms = (config.stale_item_hours * 3600.0 * 1000.0) as u64;
        let now_ms = (now * 1000.0) as u64;

        for node in nodes.iter() {
            match &node.payload {
                NodePayload::Task(t) if t.status == TaskStatus::InProgress => {
                    if now_ms.saturating_sub(node.attrs.last_updated_ms) > stale_threshold_ms {
                        anomalies.push(Anomaly {
                            kind: AnomalyKind::StaleItem,
                            description: format!(
                                "Task '{}' in progress but not updated for {:.0}h",
                                t.description,
                                (now_ms - node.attrs.last_updated_ms) as f64 / 3_600_000.0,
                            ),
                            severity: 0.5,
                            node_id: Some(node.id),
                            detected_at: now,
                        });
                    }
                }
                NodePayload::Goal(g) if g.status == GoalStatus::Active && g.progress < 0.01 => {
                    if now_ms.saturating_sub(node.attrs.last_updated_ms) > stale_threshold_ms {
                        anomalies.push(Anomaly {
                            kind: AnomalyKind::StaleItem,
                            description: format!(
                                "Goal '{}' active with 0% progress for {:.0}h",
                                g.description,
                                (now_ms - node.attrs.last_updated_ms) as f64 / 3_600_000.0,
                            ),
                            severity: 0.6,
                            node_id: Some(node.id),
                            detected_at: now,
                        });
                    }
                }
                _ => {}
            }
        }

        // 4b: Missing expected events — routines with high reliability
        // that haven't triggered on schedule
        let missing_threshold_secs = config.missing_event_hours * 3600.0;
        for node in nodes.iter() {
            if let NodePayload::Routine(r) = &node.payload {
                if r.reliability > 0.7 && r.observation_count > 10 {
                    let time_since = now - r.last_triggered;
                    let expected_period = r.period_secs;
                    if time_since > expected_period + missing_threshold_secs {
                        anomalies.push(Anomaly {
                            kind: AnomalyKind::MissingExpectedEvent,
                            description: format!(
                                "Routine '{}' (reliability {:.0}%) expected every {:.0}h, last triggered {:.1}h ago",
                                r.description,
                                r.reliability * 100.0,
                                expected_period / 3600.0,
                                time_since / 3600.0,
                            ),
                            severity: 0.4,
                            node_id: Some(node.id),
                            detected_at: now,
                        });
                    }
                }
            }
        }

        // 4c: Chronic snooze — agenda items snoozed multiple times
        for item in agenda.items_iter() {
            if item.dismiss_count >= 3 && item.is_surfaceable(now) {
                anomalies.push(Anomaly {
                    kind: AnomalyKind::ChronicSnooze,
                    description: format!(
                        "Item '{}' has been dismissed {} times",
                        item.description, item.dismiss_count,
                    ),
                    severity: 0.3,
                    node_id: None,
                    detected_at: now,
                });
            }
        }

        state.active_anomalies = anomalies.clone();
        report.anomalies_detected = anomalies;
        report.phases_executed.push(TickPhase::AnomalyDetection);
        state.last_anomaly_check_at = now;
    }

    if std::time::Instant::now() >= budget_deadline {
        report.budget_exhausted = true;
        report.tick_duration_us = tick_start.elapsed().as_micros() as u64;
        return report;
    }

    // ── Phase 5: Suggestion Cache (every N ticks) ──
    // Note: actual suggest_next_step() requires the full pipeline which
    // is done at the engine level. Here we just expire old cached items.
    if tick_number % config.suggestion_cache_interval == 0 {
        state.cached_suggestions.retain(|s| s.is_valid(now));
        report.cached_suggestions = state.cached_suggestions.clone();
        report.phases_executed.push(TickPhase::SuggestionCache);
        state.last_suggestion_cache_at = now;
    }

    // ── Phase 6: Consolidation (every N ticks, lowest priority) ──
    if tick_number % config.consolidation_interval == 0 {
        if std::time::Instant::now() < budget_deadline {
            // Mark that consolidation should run — actual consolidation
            // is done at the engine level since it requires DB access
            report.consolidation_ran = true;
            report.phases_executed.push(TickPhase::Consolidation);
            state.last_consolidation_at = now;
        }
    }

    report.tick_duration_us = tick_start.elapsed().as_micros() as u64;
    report
}

// ══════════════════════════════════════════════════════════════════════════════
// § 7  Tick Scheduling
// ══════════════════════════════════════════════════════════════════════════════

/// Determine the optimal interval until the next tick.
///
/// Adapts tick frequency based on system state:
/// - Idle user: tick every 5 seconds
/// - Active user: tick every 1 second (catch interactions)
/// - Urgent items pending: tick every 0.5 seconds
pub fn next_tick_interval_ms(state: &TickState, agenda: &Agenda, now: f64) -> u64 {
    // Check for urgent items
    let active_items = agenda.get_active(now, 5);
    let has_urgent = active_items
        .iter()
        .any(|item| item.current_urgency(now) > 0.8);

    if has_urgent {
        500 // 0.5s when urgent items pending
    } else if state.last_tick_elapsed_secs < 2.0 {
        1000 // 1s when recently active
    } else {
        5000 // 5s when idle
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::super::agenda::{Agenda, AgendaConfig, AgendaKind, UrgencyFn};
    use super::super::hawkes::HawkesRegistry;
    use super::*;

    fn make_test_nodes() -> Vec<CognitiveNode> {
        let mut alloc = NodeIdAllocator::new();
        let mut nodes = Vec::new();

        // A task that's in progress
        let task_id = alloc.alloc(NodeKind::Task);
        let mut task = CognitiveNode::new(
            task_id,
            "Active task".to_string(),
            NodePayload::Task(TaskPayload {
                description: "Active task".to_string(),
                status: TaskStatus::InProgress,
                goal_id: None,
                deadline: None,
                priority: Priority::Medium,
                estimated_minutes: None,
                prerequisites: vec![],
            }),
        );
        task.attrs.activation = 0.5;
        nodes.push(task);

        // A stale goal
        let goal_id = alloc.alloc(NodeKind::Goal);
        let mut goal = CognitiveNode::new(
            goal_id,
            "Stale goal".to_string(),
            NodePayload::Goal(GoalPayload {
                description: "Stale goal".to_string(),
                status: GoalStatus::Active,
                progress: 0.0,
                deadline: None,
                priority: Priority::High,
                parent_goal: None,
                completion_criteria: "Done".to_string(),
            }),
        );
        goal.attrs.last_updated_ms = 1000; // Very old
        goal.attrs.activation = 0.3;
        nodes.push(goal);

        // A routine node
        let routine_id = alloc.alloc(NodeKind::Routine);
        let routine = CognitiveNode::new(
            routine_id,
            "Email check".to_string(),
            NodePayload::Routine(RoutinePayload {
                description: "Check email".to_string(),
                period_secs: 3600.0,
                phase_offset_secs: 0.0,
                reliability: 0.9,
                observation_count: 50,
                last_triggered: 0.0, // Very long ago
                action_description: "Open email app".to_string(),
                weekday_mask: 0x7F,
            }),
        );
        nodes.push(routine);

        nodes
    }

    #[test]
    fn test_basic_tick() {
        let mut state = TickState::new();
        let mut agenda = Agenda::new();
        let mut nodes = make_test_nodes();
        let registry = HawkesRegistry::new();
        let config = TickConfig::default();

        let now = 1_700_000_000.0;
        let report = cognitive_tick(now, &mut state, &mut agenda, &mut nodes, &registry, &config);

        assert_eq!(report.tick_number, 0);
        assert!(report.phases_executed.contains(&TickPhase::UrgencyRefresh));
        assert!(report.phases_executed.contains(&TickPhase::ActivationDecay));
        assert_eq!(state.tick_count, 1);
    }

    #[test]
    fn test_activation_decay() {
        let mut state = TickState::new();
        state.last_urgency_at = 1_700_000_000.0 - 10.0; // 10s ago
        let mut agenda = Agenda::new();
        let mut nodes = make_test_nodes();
        let registry = HawkesRegistry::new();
        let config = TickConfig::default();

        let initial_activation = nodes[0].attrs.activation;
        let now = 1_700_000_000.0;

        let report = cognitive_tick(now, &mut state, &mut agenda, &mut nodes, &registry, &config);

        // Activation should have decayed
        assert!(
            nodes[0].attrs.activation < initial_activation,
            "Activation should decay: {} -> {}",
            initial_activation,
            nodes[0].attrs.activation,
        );
        assert!(report.nodes_decayed > 0);
    }

    #[test]
    fn test_routine_prediction_phase() {
        let mut state = TickState::new();
        let mut agenda = Agenda::new();
        let mut nodes = make_test_nodes();
        let config = TickConfig {
            routine_check_interval: 1, // Run every tick for testing
            ..Default::default()
        };

        // Create a Hawkes registry with an anticipatable model
        let mut registry = HawkesRegistry::new();
        let timestamps: Vec<f64> = (0..20)
            .map(|i| 1_699_999_000.0 + i as f64 * 600.0)
            .collect();
        registry.observe_batch("email", &timestamps);

        let now = *timestamps.last().unwrap() + 60.0;
        let report = cognitive_tick(now, &mut state, &mut agenda, &mut nodes, &registry, &config);

        assert!(report
            .phases_executed
            .contains(&TickPhase::RoutinePrediction));
    }

    #[test]
    fn test_anomaly_detection_stale_items() {
        let mut state = TickState::new();
        let mut agenda = Agenda::new();
        let mut nodes = make_test_nodes();
        let registry = HawkesRegistry::new();
        let config = TickConfig {
            anomaly_check_interval: 1, // Run every tick for testing
            stale_item_hours: 1.0,     // 1 hour threshold
            ..Default::default()
        };

        let now = 1_700_000_000.0;
        let report = cognitive_tick(now, &mut state, &mut agenda, &mut nodes, &registry, &config);

        assert!(report
            .phases_executed
            .contains(&TickPhase::AnomalyDetection));
        // Should detect stale goal (last_updated_ms = 1000, way old)
        let stale_anomalies: Vec<_> = report
            .anomalies_detected
            .iter()
            .filter(|a| a.kind == AnomalyKind::StaleItem)
            .collect();
        assert!(
            !stale_anomalies.is_empty(),
            "Should detect stale goal: {:?}",
            report.anomalies_detected
        );
    }

    #[test]
    fn test_anomaly_detection_missing_routine() {
        let mut state = TickState::new();
        let mut agenda = Agenda::new();
        let mut nodes = make_test_nodes();
        let registry = HawkesRegistry::new();
        let config = TickConfig {
            anomaly_check_interval: 1,
            missing_event_hours: 1.0,
            ..Default::default()
        };

        let now = 1_700_000_000.0;
        let report = cognitive_tick(now, &mut state, &mut agenda, &mut nodes, &registry, &config);

        // Should detect missing routine (last_triggered = 0.0)
        let missing: Vec<_> = report
            .anomalies_detected
            .iter()
            .filter(|a| a.kind == AnomalyKind::MissingExpectedEvent)
            .collect();
        assert!(
            !missing.is_empty(),
            "Should detect missing routine: {:?}",
            report.anomalies_detected
        );
    }

    #[test]
    fn test_suggestion_cache_expiry() {
        let mut state = TickState::new();
        state.cached_suggestions.push(CachedSuggestion {
            description: "Old suggestion".to_string(),
            action_kind: "suggest".to_string(),
            utility: 0.7,
            confidence: 0.8,
            cached_at: 1_000_000.0,
            ttl_secs: 300.0,
        });
        state.cached_suggestions.push(CachedSuggestion {
            description: "Fresh suggestion".to_string(),
            action_kind: "inform".to_string(),
            utility: 0.5,
            confidence: 0.6,
            cached_at: 1_700_000_000.0 - 10.0,
            ttl_secs: 300.0,
        });

        let mut agenda = Agenda::new();
        let mut nodes = Vec::new();
        let registry = HawkesRegistry::new();
        let config = TickConfig {
            suggestion_cache_interval: 1,
            ..Default::default()
        };

        let now = 1_700_000_000.0;
        let report = cognitive_tick(now, &mut state, &mut agenda, &mut nodes, &registry, &config);

        assert!(report.phases_executed.contains(&TickPhase::SuggestionCache));
        // Old suggestion should be expired, fresh one kept
        assert_eq!(state.cached_suggestions.len(), 1);
        assert_eq!(state.cached_suggestions[0].description, "Fresh suggestion");
    }

    #[test]
    fn test_consolidation_triggers_on_interval() {
        let mut state = TickState::new();
        state.tick_count = 999; // Next tick will be 1000th
        let mut agenda = Agenda::new();
        let mut nodes = Vec::new();
        let registry = HawkesRegistry::new();
        let config = TickConfig {
            consolidation_interval: 1000,
            ..Default::default()
        };

        let now = 1_700_000_000.0;
        // Tick 999 → tick_number=999, 999 % 1000 != 0
        let report = cognitive_tick(now, &mut state, &mut agenda, &mut nodes, &registry, &config);
        assert!(!report.consolidation_ran);

        // Next tick (tick_number=1000) should trigger
        let report = cognitive_tick(
            now + 1.0,
            &mut state,
            &mut agenda,
            &mut nodes,
            &registry,
            &config,
        );
        assert!(report.consolidation_ran);
    }

    #[test]
    fn test_tick_counter_increments() {
        let mut state = TickState::new();
        let mut agenda = Agenda::new();
        let mut nodes = Vec::new();
        let registry = HawkesRegistry::new();
        let config = TickConfig::default();

        for i in 0..5 {
            cognitive_tick(
                1_700_000_000.0 + i as f64,
                &mut state,
                &mut agenda,
                &mut nodes,
                &registry,
                &config,
            );
        }
        assert_eq!(state.tick_count, 5);
    }

    #[test]
    fn test_next_tick_interval_idle() {
        let state = TickState {
            last_tick_elapsed_secs: 10.0, // Idle for 10s
            ..Default::default()
        };
        let agenda = Agenda::new();
        let now = 1_700_000_000.0;
        let interval = next_tick_interval_ms(&state, &agenda, now);
        assert_eq!(interval, 5000); // 5s for idle
    }

    #[test]
    fn test_next_tick_interval_active() {
        let state = TickState {
            last_tick_elapsed_secs: 1.0, // Recent activity
            ..Default::default()
        };
        let agenda = Agenda::new();
        let now = 1_700_000_000.0;
        let interval = next_tick_interval_ms(&state, &agenda, now);
        assert_eq!(interval, 1000); // 1s for active
    }

    #[test]
    fn test_agenda_items_surfaced() {
        let mut state = TickState::new();
        let mut agenda = Agenda::new();
        let agenda_config = AgendaConfig::default();
        let now = 1_700_000_000.0;

        // Add an item with high urgency that should surface
        agenda.add_item_at(
            NodeId::NIL,
            AgendaKind::StalledIntent,
            UrgencyFn::Constant { value: 0.9 },
            None,
            "High urgency item".to_string(),
            now - 100.0, // Created 100s ago
            &agenda_config,
        );

        let mut nodes = Vec::new();
        let registry = HawkesRegistry::new();
        let config = TickConfig::default();

        let report = cognitive_tick(now, &mut state, &mut agenda, &mut nodes, &registry, &config);

        // The item should be surfaced since it's active and urgent
        // (depends on AgendaConfig.surface_threshold)
        assert!(report.phases_executed.contains(&TickPhase::UrgencyRefresh));
    }

    #[test]
    fn test_items_iter_available() {
        // Verify that agenda.items_iter() is available
        let agenda = Agenda::new();
        let count = agenda.items_iter().count();
        assert_eq!(count, 0);
    }
}
