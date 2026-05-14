//! CK-2.1: Agenda / Open Loops Engine
//!
//! Maintains a persistent sense of "what is unfinished" — carries unresolved
//! items across sessions and resurfaces them at the right moment using
//! parameterized urgency functions.
//!
//! ## Urgency Functions
//!
//! Each agenda item has a time-dependent urgency function `u(t)`:
//!
//! - **Linear**: `u(t) = base + slope × (t - t₀)` — steady escalation
//! - **Sigmoid**: `u(t) = 1 / (1 + exp(-k × (deadline - t - τ)))` — deadline ramp
//! - **StepAtDeadline**: 0 until `deadline - buffer`, then 1.0
//! - **DecayIfIgnored**: starts at `initial`, decays by `decay_factor` per dismissal
//! - **Constant**: fixed urgency, never changes
//!
//! ## Open Loop Detection
//!
//! Scans the cognitive graph for items that became "open loops":
//! - Tasks with no progress for > stale_threshold
//! - Goals with active status but 0 progress change
//! - Commitments (episodes referencing future actions) without resolution
//! - Belief conflicts without resolution

use serde::{Deserialize, Serialize};

use super::state::*;

// ── Agenda Item Identity ──

/// Unique identifier for an agenda item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgendaId(pub u64);

/// Monotonic agenda ID allocator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgendaIdAllocator {
    next: u64,
}

impl AgendaIdAllocator {
    pub fn new() -> Self {
        Self { next: 1 }
    }

    pub fn alloc(&mut self) -> AgendaId {
        let id = AgendaId(self.next);
        self.next += 1;
        id
    }

    pub fn high_water_mark(&self) -> u64 {
        self.next
    }

    pub fn from_high_water_mark(hwm: u64) -> Self {
        Self { next: hwm.max(1) }
    }
}

impl Default for AgendaIdAllocator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Urgency Functions ──

/// Parameterized urgency function that computes urgency from time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UrgencyFn {
    /// Linear ramp: `base + slope × (now - created_at)`.
    /// Useful for items that steadily become more important.
    Linear { base: f64, slope: f64 },

    /// Sigmoid approaching deadline:
    /// `1 / (1 + exp(-steepness × (deadline - now - offset)))`.
    /// Low urgency until near deadline, then rapid ramp.
    Sigmoid {
        deadline: f64,
        steepness: f64,
        offset: f64,
    },

    /// Step function: 0 until `deadline - buffer_secs`, then 1.0.
    /// Binary urgency for hard deadlines.
    StepAtDeadline { deadline: f64, buffer_secs: f64 },

    /// Starts at `initial`, multiplied by `decay_factor` per dismissal.
    /// Items the user keeps dismissing become less urgent.
    DecayIfIgnored { initial: f64, decay_factor: f64 },

    /// Fixed urgency, never changes with time.
    Constant { value: f64 },
}

impl UrgencyFn {
    /// Evaluate urgency at a given time.
    ///
    /// `created_at` — when the agenda item was created (seconds).
    /// `now` — current time (seconds).
    /// `dismiss_count` — number of times this item was dismissed.
    pub fn evaluate(&self, created_at: f64, now: f64, dismiss_count: u32) -> f64 {
        let raw = match self {
            Self::Linear { base, slope } => {
                let elapsed = (now - created_at).max(0.0);
                base + slope * elapsed
            }

            Self::Sigmoid {
                deadline,
                steepness,
                offset,
            } => {
                // Urgency increases as now approaches deadline.
                // time_until > 0 means deadline is in the future.
                let time_until = deadline - now - offset;
                1.0 / (1.0 + (steepness * time_until).exp())
            }

            Self::StepAtDeadline {
                deadline,
                buffer_secs,
            } => {
                if now >= deadline - buffer_secs {
                    1.0
                } else {
                    0.0
                }
            }

            Self::DecayIfIgnored {
                initial,
                decay_factor,
            } => initial * decay_factor.powi(dismiss_count as i32),

            Self::Constant { value } => *value,
        };

        raw.clamp(0.0, 1.0)
    }
}

// ── Agenda Item ──

/// What kind of open loop this represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgendaKind {
    /// A question was asked but never answered.
    UnresolvedQuestion,
    /// User made a commitment ("I'll do X") not yet fulfilled.
    PendingCommitment,
    /// Something needs follow-up (e.g., sent email, awaiting reply).
    FollowUpNeeded,
    /// A routine window is opening (predicted from patterns).
    RoutineWindowOpening,
    /// A deadline is approaching.
    DeadlineApproaching,
    /// An anomaly was detected that needs user confirmation.
    AnomalyRequiresConfirmation,
    /// Two beliefs conflict and need resolution.
    BeliefConflictNeedsResolution,
    /// A task was started but abandoned (no progress for threshold).
    AbandonedTask,
    /// An intent was inferred but never acted on.
    StalledIntent,
}

impl AgendaKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnresolvedQuestion => "unresolved_question",
            Self::PendingCommitment => "pending_commitment",
            Self::FollowUpNeeded => "follow_up_needed",
            Self::RoutineWindowOpening => "routine_window_opening",
            Self::DeadlineApproaching => "deadline_approaching",
            Self::AnomalyRequiresConfirmation => "anomaly_requires_confirmation",
            Self::BeliefConflictNeedsResolution => "belief_conflict_needs_resolution",
            Self::AbandonedTask => "abandoned_task",
            Self::StalledIntent => "stalled_intent",
        }
    }
}

/// Lifecycle status of an agenda item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgendaStatus {
    /// Active and may surface to user.
    Active,
    /// Temporarily suppressed until `snoozed_until`.
    Snoozed,
    /// The open loop was resolved (task completed, question answered, etc.).
    Resolved,
    /// Past its relevance window — no longer applicable.
    Expired,
    /// User explicitly dismissed; won't resurface (unless urgency spikes).
    Dismissed,
}

/// A rule that suppresses surfacing in specific contexts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuppressionRule {
    /// Human-readable description.
    pub description: String,
    /// Suppress when this condition matches the context.
    pub condition: SuppressionCondition,
}

/// Conditions that can suppress an agenda item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuppressionCondition {
    /// Suppress during specific hours.
    TimeRange { start_hour: u8, end_hour: u8 },
    /// Suppress when user is in a specific activity.
    DuringActivity { activity: String },
    /// Suppress when cognitive load exceeds threshold.
    HighCognitiveLoad { threshold: f64 },
    /// Suppress in shared/public contexts.
    SharedContext,
}

/// A single agenda item — an unresolved thing the system is tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgendaItem {
    /// Unique identifier.
    pub id: AgendaId,

    /// Which cognitive graph node spawned this item.
    pub source_node: NodeId,

    /// What kind of open loop this is.
    pub kind: AgendaKind,

    /// When this item was created (seconds since epoch).
    pub created_at: f64,

    /// Optional hard deadline (seconds since epoch).
    pub due_at: Option<f64>,

    /// Parameterized urgency function.
    pub urgency_fn: UrgencyFn,

    /// Current lifecycle status.
    pub status: AgendaStatus,

    /// Contextual suppression rules.
    pub suppression_rules: Vec<SuppressionRule>,

    /// When this item was last surfaced to the user.
    pub last_surfaced_at: Option<f64>,

    /// How many times this item has been surfaced.
    pub surface_count: u32,

    /// Maximum times to surface before auto-expiring (anti-nag).
    pub max_surfaces: u8,

    /// How many times the user dismissed this item.
    pub dismiss_count: u32,

    /// If snoozed, when to wake up.
    pub snoozed_until: Option<f64>,

    /// Human-readable description.
    pub description: String,
}

impl AgendaItem {
    /// Compute current urgency for this item.
    pub fn current_urgency(&self, now: f64) -> f64 {
        self.urgency_fn
            .evaluate(self.created_at, now, self.dismiss_count)
    }

    /// Whether this item is surfaceable right now.
    pub fn is_surfaceable(&self, now: f64) -> bool {
        match self.status {
            AgendaStatus::Active => true,
            AgendaStatus::Snoozed => self.snoozed_until.map_or(false, |until| now >= until),
            _ => false,
        }
    }

    /// Whether this item has exceeded its max surface count.
    pub fn is_nagging(&self) -> bool {
        self.surface_count >= self.max_surfaces as u32
    }

    /// Check suppression rules against context.
    pub fn is_suppressed(&self, hour: u8, cognitive_load: f64, is_shared: bool) -> bool {
        for rule in &self.suppression_rules {
            match &rule.condition {
                SuppressionCondition::TimeRange {
                    start_hour,
                    end_hour,
                } => {
                    let in_range = if start_hour > end_hour {
                        hour >= *start_hour || hour < *end_hour
                    } else {
                        hour >= *start_hour && hour < *end_hour
                    };
                    if in_range {
                        return true;
                    }
                }
                SuppressionCondition::HighCognitiveLoad { threshold } => {
                    if cognitive_load > *threshold {
                        return true;
                    }
                }
                SuppressionCondition::SharedContext => {
                    if is_shared {
                        return true;
                    }
                }
                SuppressionCondition::DuringActivity { .. } => {
                    // Activity matching requires richer context — skip for now
                }
            }
        }
        false
    }
}

// ── Agenda Configuration ──

/// Configuration for the agenda engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgendaConfig {
    /// Urgency threshold above which items are ready to surface.
    pub surface_threshold: f64,

    /// Default max surfaces before auto-expiring.
    pub default_max_surfaces: u8,

    /// How long a snoozed item stays snoozed (default, seconds).
    pub default_snooze_secs: f64,

    /// Stale threshold: tasks with no progress for this many seconds
    /// are flagged as open loops.
    pub stale_task_threshold_secs: f64,

    /// Stale threshold for goals.
    pub stale_goal_threshold_secs: f64,

    /// Maximum active agenda items (prevents unbounded growth).
    pub max_active_items: usize,

    /// Minimum time between surfacing the same item (seconds).
    pub min_resurface_interval_secs: f64,
}

impl Default for AgendaConfig {
    fn default() -> Self {
        Self {
            surface_threshold: 0.5,
            default_max_surfaces: 5,
            default_snooze_secs: 3600.0,         // 1 hour
            stale_task_threshold_secs: 172800.0, // 48 hours
            stale_goal_threshold_secs: 259200.0, // 72 hours
            max_active_items: 100,
            min_resurface_interval_secs: 1800.0, // 30 minutes
        }
    }
}

// ── Agenda Engine ──

/// Result of a tick operation — what changed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickResult {
    /// Items that crossed the urgency threshold and are ready to surface.
    pub ready_to_surface: Vec<AgendaId>,

    /// Items that auto-expired (exceeded max surfaces or past deadline).
    pub auto_expired: Vec<AgendaId>,

    /// Items that were un-snoozed (snooze time elapsed).
    pub unsnoozed: Vec<AgendaId>,

    /// Total active items after tick.
    pub active_count: usize,
}

/// Result of open loop detection scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenLoopScanResult {
    /// Newly detected open loops.
    pub new_loops: Vec<DetectedLoop>,

    /// Total nodes scanned.
    pub nodes_scanned: usize,
}

/// A newly detected open loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedLoop {
    /// The source node that represents the open loop.
    pub node_id: NodeId,

    /// What kind of open loop.
    pub kind: AgendaKind,

    /// Why it was detected.
    pub reason: String,

    /// Suggested urgency function.
    pub suggested_urgency: UrgencyFn,

    /// Suggested description.
    pub description: String,
}

/// The in-memory agenda — a collection of tracked items.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Agenda {
    /// All tracked items.
    pub items: Vec<AgendaItem>,

    /// ID allocator.
    pub allocator: AgendaIdAllocator,
}

impl Agenda {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new item to the agenda.
    pub fn add_item(
        &mut self,
        source_node: NodeId,
        kind: AgendaKind,
        urgency_fn: UrgencyFn,
        due_at: Option<f64>,
        description: String,
        config: &AgendaConfig,
    ) -> AgendaId {
        // Check capacity
        if self.active_count() >= config.max_active_items {
            self.evict_lowest_urgency(0.0); // evict with current time unknown, use 0
        }

        let id = self.allocator.alloc();
        self.items.push(AgendaItem {
            id,
            source_node,
            kind,
            created_at: 0.0, // caller should set via set_created_at
            due_at,
            urgency_fn,
            status: AgendaStatus::Active,
            suppression_rules: vec![],
            last_surfaced_at: None,
            surface_count: 0,
            max_surfaces: config.default_max_surfaces,
            dismiss_count: 0,
            snoozed_until: None,
            description,
        });
        id
    }

    /// Add a new item with explicit created_at timestamp.
    pub fn add_item_at(
        &mut self,
        source_node: NodeId,
        kind: AgendaKind,
        urgency_fn: UrgencyFn,
        due_at: Option<f64>,
        description: String,
        created_at: f64,
        config: &AgendaConfig,
    ) -> AgendaId {
        let id = self.add_item(source_node, kind, urgency_fn, due_at, description, config);
        if let Some(item) = self.items.last_mut() {
            item.created_at = created_at;
        }
        id
    }

    /// Tick: recompute urgencies, handle snooze expiry, auto-expire stale items.
    pub fn tick(&mut self, now: f64, config: &AgendaConfig) -> TickResult {
        let mut ready = Vec::new();
        let mut expired = Vec::new();
        let mut unsnoozed = Vec::new();

        for item in &mut self.items {
            match item.status {
                AgendaStatus::Active => {
                    // Check max surfaces (anti-nag)
                    if item.is_nagging() {
                        item.status = AgendaStatus::Expired;
                        expired.push(item.id);
                        continue;
                    }

                    // Check deadline expiry
                    if let Some(due) = item.due_at {
                        if now > due && item.kind != AgendaKind::DeadlineApproaching {
                            item.status = AgendaStatus::Expired;
                            expired.push(item.id);
                            continue;
                        }
                    }

                    // Compute urgency
                    let urgency = item.current_urgency(now);
                    if urgency >= config.surface_threshold {
                        // Check min resurface interval
                        let can_resurface = item.last_surfaced_at.map_or(true, |last| {
                            now - last >= config.min_resurface_interval_secs
                        });
                        if can_resurface {
                            ready.push(item.id);
                        }
                    }
                }

                AgendaStatus::Snoozed => {
                    if let Some(until) = item.snoozed_until {
                        if now >= until {
                            item.status = AgendaStatus::Active;
                            item.snoozed_until = None;
                            unsnoozed.push(item.id);
                        }
                    }
                }

                _ => {} // Resolved, Expired, Dismissed — skip
            }
        }

        TickResult {
            ready_to_surface: ready,
            auto_expired: expired,
            unsnoozed,
            active_count: self.active_count(),
        }
    }

    /// Mark an item as resolved.
    pub fn resolve(&mut self, id: AgendaId) -> bool {
        if let Some(item) = self.find_mut(id) {
            item.status = AgendaStatus::Resolved;
            true
        } else {
            false
        }
    }

    /// Snooze an item for a duration.
    pub fn snooze(&mut self, id: AgendaId, now: f64, duration_secs: f64) -> bool {
        if let Some(item) = self.find_mut(id) {
            item.status = AgendaStatus::Snoozed;
            item.snoozed_until = Some(now + duration_secs);
            true
        } else {
            false
        }
    }

    /// Dismiss an item. Increments dismiss_count (affects DecayIfIgnored urgency).
    pub fn dismiss(&mut self, id: AgendaId) -> bool {
        if let Some(item) = self.find_mut(id) {
            item.dismiss_count += 1;
            item.status = AgendaStatus::Dismissed;
            true
        } else {
            false
        }
    }

    /// Record that an item was surfaced to the user.
    pub fn mark_surfaced(&mut self, id: AgendaId, now: f64) {
        if let Some(item) = self.find_mut(id) {
            item.surface_count += 1;
            item.last_surfaced_at = Some(now);
        }
    }

    /// Get active items sorted by current urgency (descending).
    pub fn get_active(&self, now: f64, limit: usize) -> Vec<&AgendaItem> {
        let mut active: Vec<&AgendaItem> = self
            .items
            .iter()
            .filter(|i| i.is_surfaceable(now))
            .collect();

        active.sort_by(|a, b| {
            let ua = a.current_urgency(now);
            let ub = b.current_urgency(now);
            ub.partial_cmp(&ua).unwrap_or(std::cmp::Ordering::Equal)
        });

        active.into_iter().take(limit).collect()
    }

    /// Count active items.
    pub fn active_count(&self) -> usize {
        self.items
            .iter()
            .filter(|i| matches!(i.status, AgendaStatus::Active | AgendaStatus::Snoozed))
            .count()
    }

    /// Find an item by ID.
    pub fn find(&self, id: AgendaId) -> Option<&AgendaItem> {
        self.items.iter().find(|i| i.id == id)
    }

    /// Find an item by ID (mutable).
    fn find_mut(&mut self, id: AgendaId) -> Option<&mut AgendaItem> {
        self.items.iter_mut().find(|i| i.id == id)
    }

    /// Iterate over all items (read-only).
    pub fn items_iter(&self) -> impl Iterator<Item = &AgendaItem> {
        self.items.iter()
    }

    /// Check if an item exists for a given source node.
    pub fn has_item_for_node(&self, node_id: NodeId) -> bool {
        self.items.iter().any(|i| {
            i.source_node == node_id
                && matches!(i.status, AgendaStatus::Active | AgendaStatus::Snoozed)
        })
    }

    /// Evict the lowest-urgency active item.
    fn evict_lowest_urgency(&mut self, now: f64) {
        let mut lowest_idx = None;
        let mut lowest_urgency = f64::MAX;

        for (idx, item) in self.items.iter().enumerate() {
            if matches!(item.status, AgendaStatus::Active) {
                let u = item.current_urgency(now);
                if u < lowest_urgency {
                    lowest_urgency = u;
                    lowest_idx = Some(idx);
                }
            }
        }

        if let Some(idx) = lowest_idx {
            self.items[idx].status = AgendaStatus::Expired;
        }
    }
}

// ── Open Loop Detection ──

/// Scan cognitive nodes for new open loops.
///
/// Detects:
/// - Tasks that are InProgress but haven't been updated recently
/// - Goals that are Active with no progress change
/// - Belief conflicts pending resolution
/// - Intent hypotheses that were never acted on
pub fn detect_open_loops(
    nodes: &[&CognitiveNode],
    existing_agenda: &Agenda,
    now: f64,
    config: &AgendaConfig,
) -> OpenLoopScanResult {
    let mut new_loops = Vec::new();

    for node in nodes {
        // Skip if we already track this node
        if existing_agenda.has_item_for_node(node.id) {
            continue;
        }

        match (&node.payload, node.id.kind()) {
            // Stale tasks
            (NodePayload::Task(task), NodeKind::Task) => {
                if task.status == TaskStatus::InProgress {
                    let age = now - (node.attrs.last_updated_ms as f64 / 1000.0);
                    if age > config.stale_task_threshold_secs {
                        new_loops.push(DetectedLoop {
                            node_id: node.id,
                            kind: AgendaKind::AbandonedTask,
                            reason: format!(
                                "Task '{}' in progress but no activity for {:.0}h",
                                task.description,
                                age / 3600.0,
                            ),
                            suggested_urgency: UrgencyFn::Linear {
                                base: 0.3,
                                slope: 0.00001, // ~0.036/hour
                            },
                            description: format!("Stale task: {}", task.description),
                        });
                    }
                }
            }

            // Stale goals
            (NodePayload::Goal(goal), NodeKind::Goal) => {
                if goal.status == GoalStatus::Active && goal.progress < 0.05 {
                    let age = now - (node.attrs.last_updated_ms as f64 / 1000.0);
                    if age > config.stale_goal_threshold_secs {
                        new_loops.push(DetectedLoop {
                            node_id: node.id,
                            kind: AgendaKind::StalledIntent,
                            reason: format!(
                                "Goal '{}' is active but has {:.0}% progress and no activity for {:.0}h",
                                goal.description,
                                goal.progress * 100.0,
                                age / 3600.0,
                            ),
                            suggested_urgency: UrgencyFn::Linear {
                                base: 0.2,
                                slope: 0.000005,
                            },
                            description: format!("Stalled goal: {}", goal.description),
                        });
                    }
                }

                // Approaching deadline
                if let Some(deadline) = goal.deadline {
                    let time_until = deadline - now;
                    if time_until > 0.0 && time_until < 86400.0 * 3.0 {
                        // Within 3 days of deadline
                        new_loops.push(DetectedLoop {
                            node_id: node.id,
                            kind: AgendaKind::DeadlineApproaching,
                            reason: format!(
                                "Goal '{}' deadline in {:.1} hours",
                                goal.description,
                                time_until / 3600.0,
                            ),
                            suggested_urgency: UrgencyFn::Sigmoid {
                                deadline,
                                steepness: 0.0001,
                                offset: 3600.0, // ramp starts 1h before
                            },
                            description: format!("Deadline: {}", goal.description),
                        });
                    }
                }
            }

            // Tasks with approaching deadlines
            (NodePayload::Task(task), NodeKind::Task) => {
                if let Some(deadline) = task.deadline {
                    let time_until = deadline - now;
                    if time_until > 0.0
                        && time_until < 86400.0
                        && task.status != TaskStatus::Completed
                        && task.status != TaskStatus::Cancelled
                    {
                        new_loops.push(DetectedLoop {
                            node_id: node.id,
                            kind: AgendaKind::DeadlineApproaching,
                            reason: format!(
                                "Task '{}' due in {:.1} hours",
                                task.description,
                                time_until / 3600.0,
                            ),
                            suggested_urgency: UrgencyFn::StepAtDeadline {
                                deadline,
                                buffer_secs: 3600.0,
                            },
                            description: format!("Due soon: {}", task.description),
                        });
                    }
                }
            }

            _ => {}
        }
    }

    OpenLoopScanResult {
        nodes_scanned: nodes.len(),
        new_loops,
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> AgendaConfig {
        AgendaConfig::default()
    }

    fn make_goal_node(alloc: &mut NodeIdAllocator, desc: &str) -> CognitiveNode {
        let id = alloc.alloc(NodeKind::Goal);
        CognitiveNode::new(
            id,
            desc.to_string(),
            NodePayload::Goal(GoalPayload {
                description: desc.to_string(),
                status: GoalStatus::Active,
                progress: 0.0,
                deadline: None,
                priority: Priority::High,
                parent_goal: None,
                completion_criteria: "Done".to_string(),
            }),
        )
    }

    fn make_task_node(
        alloc: &mut NodeIdAllocator,
        desc: &str,
        status: TaskStatus,
    ) -> CognitiveNode {
        let id = alloc.alloc(NodeKind::Task);
        CognitiveNode::new(
            id,
            desc.to_string(),
            NodePayload::Task(TaskPayload {
                description: desc.to_string(),
                status,
                goal_id: None,
                deadline: None,
                priority: Priority::Medium,
                estimated_minutes: None,
                prerequisites: vec![],
            }),
        )
    }

    // ── Urgency function tests ──

    #[test]
    fn test_linear_urgency() {
        let f = UrgencyFn::Linear {
            base: 0.1,
            slope: 0.0001,
        };
        // At creation
        assert!((f.evaluate(1000.0, 1000.0, 0) - 0.1).abs() < 0.001);
        // After 1 hour
        assert!((f.evaluate(1000.0, 4600.0, 0) - 0.46).abs() < 0.01);
        // Clamped to 1.0
        let high_slope = UrgencyFn::Linear {
            base: 0.5,
            slope: 1.0,
        };
        assert_eq!(high_slope.evaluate(0.0, 100.0, 0), 1.0);
    }

    #[test]
    fn test_sigmoid_urgency() {
        let deadline = 10000.0;
        let f = UrgencyFn::Sigmoid {
            deadline,
            steepness: 0.001,
            offset: 1000.0,
        };
        // Far from deadline — low urgency
        let far = f.evaluate(0.0, 1000.0, 0);
        // Near deadline — high urgency
        let near = f.evaluate(0.0, 9500.0, 0);
        assert!(
            near > far,
            "urgency should increase near deadline: near={near} far={far}"
        );
    }

    #[test]
    fn test_step_at_deadline() {
        let f = UrgencyFn::StepAtDeadline {
            deadline: 10000.0,
            buffer_secs: 3600.0,
        };
        // Before buffer
        assert_eq!(f.evaluate(0.0, 5000.0, 0), 0.0);
        // Within buffer
        assert_eq!(f.evaluate(0.0, 7000.0, 0), 1.0);
        // Past deadline
        assert_eq!(f.evaluate(0.0, 11000.0, 0), 1.0);
    }

    #[test]
    fn test_decay_if_ignored() {
        let f = UrgencyFn::DecayIfIgnored {
            initial: 0.8,
            decay_factor: 0.5,
        };
        assert!((f.evaluate(0.0, 0.0, 0) - 0.8).abs() < 0.001);
        assert!((f.evaluate(0.0, 0.0, 1) - 0.4).abs() < 0.001);
        assert!((f.evaluate(0.0, 0.0, 2) - 0.2).abs() < 0.001);
        assert!((f.evaluate(0.0, 0.0, 3) - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_constant_urgency() {
        let f = UrgencyFn::Constant { value: 0.7 };
        assert_eq!(f.evaluate(0.0, 0.0, 0), 0.7);
        assert_eq!(f.evaluate(0.0, 99999.0, 5), 0.7);
    }

    // ── Agenda lifecycle tests ──

    #[test]
    fn test_add_and_retrieve() {
        let mut agenda = Agenda::new();
        let mut alloc = NodeIdAllocator::new();
        let node = make_goal_node(&mut alloc, "Test goal");
        let config = default_config();

        let id = agenda.add_item_at(
            node.id,
            AgendaKind::StalledIntent,
            UrgencyFn::Constant { value: 0.6 },
            None,
            "Test item".to_string(),
            1000.0,
            &config,
        );

        assert_eq!(agenda.active_count(), 1);
        let found = agenda.find(id).unwrap();
        assert_eq!(found.kind, AgendaKind::StalledIntent);
        assert_eq!(found.description, "Test item");
    }

    #[test]
    fn test_resolve() {
        let mut agenda = Agenda::new();
        let mut alloc = NodeIdAllocator::new();
        let node = make_goal_node(&mut alloc, "Goal");
        let config = default_config();

        let id = agenda.add_item(
            node.id,
            AgendaKind::PendingCommitment,
            UrgencyFn::Constant { value: 0.5 },
            None,
            "Commitment".to_string(),
            &config,
        );

        assert!(agenda.resolve(id));
        assert_eq!(agenda.active_count(), 0);
        assert_eq!(agenda.find(id).unwrap().status, AgendaStatus::Resolved);
    }

    #[test]
    fn test_snooze_and_unsnooze() {
        let mut agenda = Agenda::new();
        let mut alloc = NodeIdAllocator::new();
        let node = make_goal_node(&mut alloc, "Goal");
        let config = default_config();

        let id = agenda.add_item_at(
            node.id,
            AgendaKind::FollowUpNeeded,
            UrgencyFn::Constant { value: 0.7 },
            None,
            "Follow up".to_string(),
            1000.0,
            &config,
        );

        // Snooze for 1 hour
        agenda.snooze(id, 1000.0, 3600.0);
        assert_eq!(agenda.find(id).unwrap().status, AgendaStatus::Snoozed);

        // Tick before snooze expires
        let result1 = agenda.tick(2000.0, &config);
        assert!(result1.unsnoozed.is_empty());

        // Tick after snooze expires
        let result2 = agenda.tick(5000.0, &config);
        assert!(result2.unsnoozed.contains(&id));
        assert_eq!(agenda.find(id).unwrap().status, AgendaStatus::Active);
    }

    #[test]
    fn test_dismiss_decays_urgency() {
        let mut agenda = Agenda::new();
        let mut alloc = NodeIdAllocator::new();
        let node = make_goal_node(&mut alloc, "Goal");
        let config = default_config();

        let id = agenda.add_item_at(
            node.id,
            AgendaKind::UnresolvedQuestion,
            UrgencyFn::DecayIfIgnored {
                initial: 0.8,
                decay_factor: 0.5,
            },
            None,
            "Question".to_string(),
            1000.0,
            &config,
        );

        let u_before = agenda.find(id).unwrap().current_urgency(2000.0);
        agenda.dismiss(id);

        // Re-activate to check urgency
        agenda.find_mut(id).unwrap().status = AgendaStatus::Active;
        let u_after = agenda.find(id).unwrap().current_urgency(2000.0);

        assert!(
            u_after < u_before,
            "urgency should decrease after dismissal"
        );
    }

    #[test]
    fn test_tick_surfaces_urgent_items() {
        let mut agenda = Agenda::new();
        let mut alloc = NodeIdAllocator::new();
        let node = make_goal_node(&mut alloc, "Goal");
        let config = default_config(); // threshold 0.5

        // Item below threshold
        agenda.add_item_at(
            node.id,
            AgendaKind::StalledIntent,
            UrgencyFn::Constant { value: 0.3 },
            None,
            "Low urgency".to_string(),
            1000.0,
            &config,
        );

        let node2 = make_goal_node(&mut alloc, "Goal 2");
        // Item above threshold
        agenda.add_item_at(
            node2.id,
            AgendaKind::DeadlineApproaching,
            UrgencyFn::Constant { value: 0.8 },
            None,
            "High urgency".to_string(),
            1000.0,
            &config,
        );

        let result = agenda.tick(2000.0, &config);
        assert_eq!(result.ready_to_surface.len(), 1);
        assert_eq!(result.active_count, 2);
    }

    #[test]
    fn test_anti_nag_auto_expire() {
        let mut agenda = Agenda::new();
        let mut alloc = NodeIdAllocator::new();
        let node = make_goal_node(&mut alloc, "Goal");
        let mut config = default_config();
        config.default_max_surfaces = 3;

        let id = agenda.add_item_at(
            node.id,
            AgendaKind::FollowUpNeeded,
            UrgencyFn::Constant { value: 0.9 },
            None,
            "Nagging item".to_string(),
            1000.0,
            &config,
        );

        // Surface 3 times
        for _ in 0..3 {
            agenda.mark_surfaced(id, 2000.0);
        }

        // Tick should auto-expire
        let result = agenda.tick(3000.0, &config);
        assert!(result.auto_expired.contains(&id));
        assert_eq!(agenda.find(id).unwrap().status, AgendaStatus::Expired);
    }

    #[test]
    fn test_get_active_sorted() {
        let mut agenda = Agenda::new();
        let mut alloc = NodeIdAllocator::new();
        let config = default_config();

        let n1 = make_goal_node(&mut alloc, "Low");
        let n2 = make_goal_node(&mut alloc, "High");
        let n3 = make_goal_node(&mut alloc, "Mid");

        agenda.add_item_at(
            n1.id,
            AgendaKind::AbandonedTask,
            UrgencyFn::Constant { value: 0.3 },
            None,
            "Low".into(),
            1000.0,
            &config,
        );
        agenda.add_item_at(
            n2.id,
            AgendaKind::DeadlineApproaching,
            UrgencyFn::Constant { value: 0.9 },
            None,
            "High".into(),
            1000.0,
            &config,
        );
        agenda.add_item_at(
            n3.id,
            AgendaKind::FollowUpNeeded,
            UrgencyFn::Constant { value: 0.6 },
            None,
            "Mid".into(),
            1000.0,
            &config,
        );

        let active = agenda.get_active(2000.0, 10);
        assert_eq!(active.len(), 3);
        assert_eq!(active[0].description, "High");
        assert_eq!(active[1].description, "Mid");
        assert_eq!(active[2].description, "Low");
    }

    #[test]
    fn test_suppression_rules() {
        let mut item = AgendaItem {
            id: AgendaId(1),
            source_node: NodeId::from_raw(0),
            kind: AgendaKind::FollowUpNeeded,
            created_at: 1000.0,
            due_at: None,
            urgency_fn: UrgencyFn::Constant { value: 0.7 },
            status: AgendaStatus::Active,
            suppression_rules: vec![
                SuppressionRule {
                    description: "No during sleep".to_string(),
                    condition: SuppressionCondition::TimeRange {
                        start_hour: 22,
                        end_hour: 7,
                    },
                },
                SuppressionRule {
                    description: "Not when busy".to_string(),
                    condition: SuppressionCondition::HighCognitiveLoad { threshold: 0.8 },
                },
            ],
            last_surfaced_at: None,
            surface_count: 0,
            max_surfaces: 5,
            dismiss_count: 0,
            snoozed_until: None,
            description: "Test".to_string(),
        };

        // Normal hours, low load
        assert!(!item.is_suppressed(14, 0.3, false));
        // Sleep hours
        assert!(item.is_suppressed(23, 0.3, false));
        // High load
        assert!(item.is_suppressed(14, 0.9, false));
        // Shared context (no rule for it)
        assert!(!item.is_suppressed(14, 0.3, true));

        // Add shared context rule
        item.suppression_rules.push(SuppressionRule {
            description: "Not when sharing".to_string(),
            condition: SuppressionCondition::SharedContext,
        });
        assert!(item.is_suppressed(14, 0.3, true));
    }

    // ── Open loop detection tests ──

    #[test]
    fn test_detect_stale_task() {
        let mut alloc = NodeIdAllocator::new();
        let mut task = make_task_node(&mut alloc, "Write tests", TaskStatus::InProgress);
        task.attrs.last_updated_ms = 1000; // very old (1 second in ms)

        let agenda = Agenda::new();
        let config = AgendaConfig {
            stale_task_threshold_secs: 3600.0, // 1 hour for testing
            ..default_config()
        };

        let now = 100000.0; // well past threshold
        let nodes: Vec<&CognitiveNode> = vec![&task];
        let result = detect_open_loops(&nodes, &agenda, now, &config);

        assert_eq!(result.new_loops.len(), 1);
        assert_eq!(result.new_loops[0].kind, AgendaKind::AbandonedTask);
    }

    #[test]
    fn test_detect_stale_goal() {
        let mut alloc = NodeIdAllocator::new();
        let mut goal = make_goal_node(&mut alloc, "Learn Rust");
        goal.attrs.last_updated_ms = 1000;

        let agenda = Agenda::new();
        let config = AgendaConfig {
            stale_goal_threshold_secs: 3600.0,
            ..default_config()
        };

        let now = 100000.0;
        let nodes: Vec<&CognitiveNode> = vec![&goal];
        let result = detect_open_loops(&nodes, &agenda, now, &config);

        assert_eq!(result.new_loops.len(), 1);
        assert_eq!(result.new_loops[0].kind, AgendaKind::StalledIntent);
    }

    #[test]
    fn test_detect_approaching_deadline() {
        let mut alloc = NodeIdAllocator::new();
        let id = alloc.alloc(NodeKind::Goal);
        let now = 100000.0;
        let deadline = now + 3600.0 * 12.0; // 12 hours away

        let goal = CognitiveNode::new(
            id,
            "Ship feature".to_string(),
            NodePayload::Goal(GoalPayload {
                description: "Ship feature".to_string(),
                status: GoalStatus::Active,
                progress: 0.5,
                deadline: Some(deadline),
                priority: Priority::Critical,
                parent_goal: None,
                completion_criteria: "Deployed".to_string(),
            }),
        );

        let agenda = Agenda::new();
        let config = default_config();
        let nodes: Vec<&CognitiveNode> = vec![&goal];
        let result = detect_open_loops(&nodes, &agenda, now, &config);

        assert!(result
            .new_loops
            .iter()
            .any(|l| l.kind == AgendaKind::DeadlineApproaching));
    }

    #[test]
    fn test_no_duplicate_detection() {
        let mut alloc = NodeIdAllocator::new();
        let mut task = make_task_node(&mut alloc, "Stale task", TaskStatus::InProgress);
        task.attrs.last_updated_ms = 1000;

        let mut agenda = Agenda::new();
        let config = AgendaConfig {
            stale_task_threshold_secs: 3600.0,
            ..default_config()
        };

        // Add existing item for this node
        agenda.add_item_at(
            task.id,
            AgendaKind::AbandonedTask,
            UrgencyFn::Constant { value: 0.5 },
            None,
            "Already tracked".to_string(),
            1000.0,
            &config,
        );

        let now = 100000.0;
        let nodes: Vec<&CognitiveNode> = vec![&task];
        let result = detect_open_loops(&nodes, &agenda, now, &config);

        // Should not create duplicate
        assert!(result.new_loops.is_empty());
    }

    #[test]
    fn test_min_resurface_interval() {
        let mut agenda = Agenda::new();
        let mut alloc = NodeIdAllocator::new();
        let node = make_goal_node(&mut alloc, "Goal");
        let config = AgendaConfig {
            min_resurface_interval_secs: 1800.0, // 30 min
            ..default_config()
        };

        let id = agenda.add_item_at(
            node.id,
            AgendaKind::DeadlineApproaching,
            UrgencyFn::Constant { value: 0.9 },
            None,
            "Urgent".to_string(),
            1000.0,
            &config,
        );

        // First tick — should surface
        let result1 = agenda.tick(2000.0, &config);
        assert!(result1.ready_to_surface.contains(&id));

        // Mark surfaced
        agenda.mark_surfaced(id, 2000.0);

        // Tick 10 minutes later — too soon
        let result2 = agenda.tick(2600.0, &config);
        assert!(!result2.ready_to_surface.contains(&id));

        // Tick 35 minutes later — OK
        let result3 = agenda.tick(4100.0, &config);
        assert!(result3.ready_to_surface.contains(&id));
    }
}
