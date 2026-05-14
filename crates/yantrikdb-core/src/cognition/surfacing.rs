//! CK-2.6: Anticipatory Action Surfacing + Contextual Suppression
//!
//! The intelligence that decides **when** and **how** to surface anticipatory
//! actions to the user — the difference between helpful and annoying.
//!
//! ## Surfacing Decision Pipeline
//!
//! For each agenda item that crosses the urgency threshold:
//!
//! 1. **Urgency gate**: item urgency ≥ threshold
//! 2. **Receptivity gate**: user receptivity model says "open to this"
//! 3. **Suppression gate**: contextual rules say "not now"
//! 4. **Anti-spam gate**: rate limiter says "not too many"
//! 5. **Relevance boost/penalty**: context match adjusts priority
//! 6. **Mode selection**: whisper / nudge / alert / preempt
//!
//! ## Surfacing Modes
//!
//! - **Whisper**: ambient notification (badge, subtle indicator). User in flow.
//! - **Nudge**: brief notification with one-tap action. User between tasks.
//! - **Alert**: prominent notification. Deadline or high importance.
//! - **Preempt**: interrupt current activity. Critical / imminent deadline.
//!
//! ## Anti-Nag Contract
//!
//! - Exponential backoff on dismissed items (via DecayIfIgnored urgency)
//! - Per-session and per-hour rate limits
//! - Items that are nagging are auto-muted
//! - User preference learning adjusts future surfacing thresholds

use serde::{Deserialize, Serialize};

use super::agenda::{AgendaId, AgendaItem, AgendaKind, AgendaStatus};
use super::receptivity::{
    ActivityState, ContextSnapshot, NotificationMode, ReceptivityEstimate, ReceptivityModel,
};

// ══════════════════════════════════════════════════════════════════════════════
// § 1  Surfacing Mode
// ══════════════════════════════════════════════════════════════════════════════

/// How prominently to surface an action to the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SurfaceMode {
    /// Ambient indicator — badge count, subtle dot. Lowest disruption.
    Whisper,
    /// Brief notification with dismiss/act buttons.
    Nudge,
    /// Prominent notification that demands attention.
    Alert,
    /// Interrupts current activity. Reserved for critical/imminent items.
    Preempt,
}

impl SurfaceMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Whisper => "whisper",
            Self::Nudge => "nudge",
            Self::Alert => "alert",
            Self::Preempt => "preempt",
        }
    }

    /// Disruption cost [0, 1] for this mode.
    pub fn disruption_cost(self) -> f64 {
        match self {
            Self::Whisper => 0.05,
            Self::Nudge => 0.25,
            Self::Alert => 0.6,
            Self::Preempt => 0.95,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 2  Surfacing Reason
// ══════════════════════════════════════════════════════════════════════════════

/// Why an item is being surfaced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SurfaceReason {
    /// Urgency exceeded threshold.
    UrgencyThreshold,
    /// Deadline is imminent.
    DeadlineImminent,
    /// Routine window is opening (predicted).
    RoutineWindowOpening,
    /// Anomaly requires user attention.
    AnomalyDetected,
    /// Belief conflict needs resolution.
    ConflictNeedsResolution,
    /// Task has been stalled too long.
    StalledTask,
    /// Follow-up needed.
    FollowUpDue,
    /// User commitment not yet fulfilled.
    UnfulfilledCommitment,
}

impl SurfaceReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UrgencyThreshold => "urgency_threshold",
            Self::DeadlineImminent => "deadline_imminent",
            Self::RoutineWindowOpening => "routine_window_opening",
            Self::AnomalyDetected => "anomaly_detected",
            Self::ConflictNeedsResolution => "conflict_needs_resolution",
            Self::StalledTask => "stalled_task",
            Self::FollowUpDue => "follow_up_due",
            Self::UnfulfilledCommitment => "unfulfilled_commitment",
        }
    }

    /// Base importance of this reason type [0, 1].
    pub fn base_importance(self) -> f64 {
        match self {
            Self::DeadlineImminent => 0.95,
            Self::ConflictNeedsResolution => 0.7,
            Self::AnomalyDetected => 0.65,
            Self::UnfulfilledCommitment => 0.6,
            Self::StalledTask => 0.5,
            Self::FollowUpDue => 0.5,
            Self::RoutineWindowOpening => 0.4,
            Self::UrgencyThreshold => 0.5,
        }
    }
}

/// Derive the most fitting reason from an agenda item's kind.
fn reason_from_kind(kind: AgendaKind) -> SurfaceReason {
    match kind {
        AgendaKind::DeadlineApproaching => SurfaceReason::DeadlineImminent,
        AgendaKind::RoutineWindowOpening => SurfaceReason::RoutineWindowOpening,
        AgendaKind::AnomalyRequiresConfirmation => SurfaceReason::AnomalyDetected,
        AgendaKind::BeliefConflictNeedsResolution => SurfaceReason::ConflictNeedsResolution,
        AgendaKind::AbandonedTask | AgendaKind::StalledIntent => SurfaceReason::StalledTask,
        AgendaKind::FollowUpNeeded => SurfaceReason::FollowUpDue,
        AgendaKind::PendingCommitment => SurfaceReason::UnfulfilledCommitment,
        AgendaKind::UnresolvedQuestion => SurfaceReason::UrgencyThreshold,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  Suppression Gate
// ══════════════════════════════════════════════════════════════════════════════

/// Why an item was suppressed (not surfaced).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SuppressionCause {
    /// User is not receptive (deep focus, DND, etc.).
    LowReceptivity,
    /// Item-level suppression rule matched.
    ItemSuppressionRule,
    /// Quiet hours active.
    QuietHours,
    /// Rate limit exceeded (too many surfaces recently).
    RateLimited,
    /// Item has been dismissed too many times (anti-nag).
    AntiNag,
    /// Item has exceeded its max surface count.
    MaxSurfaces,
    /// Minimum resurface interval not elapsed.
    TooSoon,
    /// Notification mode blocks this priority level.
    NotificationModeBlock,
}

impl SuppressionCause {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LowReceptivity => "low_receptivity",
            Self::ItemSuppressionRule => "item_suppression_rule",
            Self::QuietHours => "quiet_hours",
            Self::RateLimited => "rate_limited",
            Self::AntiNag => "anti_nag",
            Self::MaxSurfaces => "max_surfaces",
            Self::TooSoon => "too_soon",
            Self::NotificationModeBlock => "notification_mode_block",
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  Proactive Suggestion (Output)
// ══════════════════════════════════════════════════════════════════════════════

/// A proactive suggestion ready to present to the user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProactiveSuggestion {
    /// The agenda item ID being surfaced.
    pub agenda_id: AgendaId,
    /// Human-readable description.
    pub description: String,
    /// What kind of open loop this is.
    pub kind: AgendaKind,
    /// How prominently to surface.
    pub mode: SurfaceMode,
    /// Why it's being surfaced.
    pub reason: SurfaceReason,
    /// Composite confidence/relevance score [0, 1].
    pub confidence: f64,
    /// Urgency of the underlying item [0, 1].
    pub urgency: f64,
    /// Estimated disruption cost of surfacing [0, 1].
    pub disruption_cost: f64,
    /// Net utility: benefit of surfacing minus disruption cost.
    pub net_utility: f64,
}

/// A suppressed item — for debugging and learning why things didn't surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuppressedItem {
    /// The agenda item ID that was suppressed.
    pub agenda_id: AgendaId,
    /// Why it was suppressed.
    pub cause: SuppressionCause,
    /// The urgency it had when suppressed.
    pub urgency: f64,
}

/// Result of the surfacing pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfacingResult {
    /// Items to surface, sorted by net utility (highest first).
    pub suggestions: Vec<ProactiveSuggestion>,
    /// Items that were considered but suppressed.
    pub suppressed: Vec<SuppressedItem>,
    /// Total items evaluated.
    pub items_evaluated: usize,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 5  Surfacing Configuration
// ══════════════════════════════════════════════════════════════════════════════

/// Configuration for the surfacing pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfacingConfig {
    /// Minimum urgency to consider for surfacing.
    pub urgency_threshold: f64,
    /// Minimum receptivity score to allow surfacing.
    pub receptivity_threshold: f64,
    /// Maximum suggestions per call.
    pub max_suggestions: usize,
    /// Maximum surfaces per hour (anti-spam).
    pub max_surfaces_per_hour: u32,
    /// Maximum surfaces per session (anti-fatigue).
    pub max_surfaces_per_session: u32,
    /// Minimum seconds between surfacing the same item.
    pub min_resurface_interval_secs: f64,
    /// Dismiss count threshold for anti-nag suppression.
    pub anti_nag_dismiss_threshold: u32,
    /// Urgency threshold for preempt mode.
    pub preempt_urgency_threshold: f64,
    /// Urgency threshold for alert mode.
    pub alert_urgency_threshold: f64,
    /// Urgency threshold for nudge mode (below this = whisper).
    pub nudge_urgency_threshold: f64,
    /// Whether to allow preempt mode at all.
    pub allow_preempt: bool,
    /// Weight for urgency in net utility calculation.
    pub urgency_weight: f64,
    /// Weight for reason importance in net utility.
    pub importance_weight: f64,
    /// Weight for receptivity in net utility.
    pub receptivity_weight: f64,
    /// Penalty weight for disruption cost.
    pub disruption_penalty_weight: f64,
}

impl Default for SurfacingConfig {
    fn default() -> Self {
        Self {
            urgency_threshold: 0.4,
            receptivity_threshold: 0.3,
            max_suggestions: 3,
            max_surfaces_per_hour: 10,
            max_surfaces_per_session: 30,
            min_resurface_interval_secs: 1800.0, // 30 minutes
            anti_nag_dismiss_threshold: 3,
            preempt_urgency_threshold: 0.95,
            alert_urgency_threshold: 0.8,
            nudge_urgency_threshold: 0.5,
            allow_preempt: true,
            urgency_weight: 0.4,
            importance_weight: 0.25,
            receptivity_weight: 0.2,
            disruption_penalty_weight: 0.15,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 6  Rate Limiter
// ══════════════════════════════════════════════════════════════════════════════

/// Tracks recent surfacing events for rate limiting.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SurfaceRateLimiter {
    /// Timestamps of recent surfaces (unix seconds).
    pub recent_surface_times: Vec<f64>,
    /// Total surfaces in the current session.
    pub session_surface_count: u32,
}

impl SurfaceRateLimiter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a surfacing event.
    pub fn record_surface(&mut self, now: f64) {
        self.recent_surface_times.push(now);
        self.session_surface_count += 1;
        // Prune entries older than 2 hours
        self.recent_surface_times.retain(|&t| now - t < 7200.0);
    }

    /// Count surfaces in the last hour.
    pub fn surfaces_last_hour(&self, now: f64) -> u32 {
        self.recent_surface_times
            .iter()
            .filter(|&&t| now - t < 3600.0)
            .count() as u32
    }

    /// Check if rate limit is exceeded.
    pub fn is_rate_limited(&self, now: f64, config: &SurfacingConfig) -> bool {
        self.surfaces_last_hour(now) >= config.max_surfaces_per_hour
            || self.session_surface_count >= config.max_surfaces_per_session
    }

    /// Reset session counter (e.g., on new session).
    pub fn reset_session(&mut self) {
        self.session_surface_count = 0;
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 7  Mode Selection
// ══════════════════════════════════════════════════════════════════════════════

/// Select the appropriate surfacing mode for an item.
///
/// Decision matrix:
/// - Preempt: urgency ≥ 0.95 AND deadline within 5 min AND preempt allowed
/// - Alert: urgency ≥ 0.8 OR deadline within 1 hour
/// - Nudge: urgency ≥ 0.5 OR user is between tasks
/// - Whisper: everything else
pub fn select_surface_mode(
    urgency: f64,
    reason: SurfaceReason,
    activity: ActivityState,
    config: &SurfacingConfig,
) -> SurfaceMode {
    // Preempt: only for imminent deadlines with very high urgency
    if config.allow_preempt
        && urgency >= config.preempt_urgency_threshold
        && reason == SurfaceReason::DeadlineImminent
    {
        return SurfaceMode::Preempt;
    }

    // Alert: high urgency or important reasons
    if urgency >= config.alert_urgency_threshold {
        return SurfaceMode::Alert;
    }

    // Alert for conflicts regardless of urgency
    if reason == SurfaceReason::ConflictNeedsResolution && urgency >= 0.6 {
        return SurfaceMode::Alert;
    }

    // Nudge: medium urgency, or user is in a natural break
    if urgency >= config.nudge_urgency_threshold {
        return SurfaceMode::Nudge;
    }

    // Nudge if user is at a natural break point
    if matches!(activity, ActivityState::Idle | ActivityState::TaskSwitching)
        && urgency >= config.urgency_threshold
    {
        return SurfaceMode::Nudge;
    }

    // Everything else: whisper
    SurfaceMode::Whisper
}

// ══════════════════════════════════════════════════════════════════════════════
// § 8  Notification Mode Filter
// ══════════════════════════════════════════════════════════════════════════════

/// Check if a surfacing mode is allowed given the user's notification preference.
fn is_mode_allowed(mode: SurfaceMode, notification_mode: NotificationMode) -> bool {
    match notification_mode {
        NotificationMode::All => true,
        NotificationMode::ImportantOnly => mode >= SurfaceMode::Alert,
        NotificationMode::DoNotDisturb => false,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 9  Net Utility Calculation
// ══════════════════════════════════════════════════════════════════════════════

/// Compute the net utility of surfacing an item.
///
/// net_utility = w_u × urgency + w_i × importance + w_r × receptivity
///             − w_d × disruption_cost
///
/// Higher net utility means more beneficial to surface.
pub fn compute_net_utility(
    urgency: f64,
    importance: f64,
    receptivity: f64,
    disruption_cost: f64,
    config: &SurfacingConfig,
) -> f64 {
    let benefit = config.urgency_weight * urgency
        + config.importance_weight * importance
        + config.receptivity_weight * receptivity;
    let cost = config.disruption_penalty_weight * disruption_cost;
    (benefit - cost).clamp(0.0, 1.0)
}

// ══════════════════════════════════════════════════════════════════════════════
// § 10  The Surfacing Pipeline (Pure — no DB dependency)
// ══════════════════════════════════════════════════════════════════════════════

/// Run the surfacing decision pipeline.
///
/// Takes agenda items, receptivity estimate, rate limiter state,
/// and returns which items to surface, how to surface them,
/// and which were suppressed (for learning).
///
/// # Arguments
/// * `items` — Candidate agenda items (usually from `agenda.get_active()`).
/// * `now` — Current time (unix seconds).
/// * `receptivity` — Current user receptivity estimate.
/// * `rate_limiter` — Rate limiter state.
/// * `context` — Current user context snapshot.
/// * `config` — Surfacing configuration.
pub fn run_surfacing_pipeline(
    items: &[AgendaItem],
    now: f64,
    receptivity: &ReceptivityEstimate,
    rate_limiter: &SurfaceRateLimiter,
    context: &ContextSnapshot,
    config: &SurfacingConfig,
) -> SurfacingResult {
    let mut suggestions = Vec::new();
    let mut suppressed = Vec::new();
    let items_evaluated = items.len();

    for item in items {
        // ── Gate 0: Status check ──
        if !item.is_surfaceable(now) {
            continue;
        }

        let urgency = item.current_urgency(now);

        // ── Gate 1: Urgency threshold ──
        if urgency < config.urgency_threshold {
            continue; // Below threshold — not even a candidate
        }

        // ── Gate 2: Anti-nag ──
        if item.dismiss_count >= config.anti_nag_dismiss_threshold {
            suppressed.push(SuppressedItem {
                agenda_id: item.id,
                cause: SuppressionCause::AntiNag,
                urgency,
            });
            continue;
        }

        // ── Gate 3: Max surfaces ──
        if item.is_nagging() {
            suppressed.push(SuppressedItem {
                agenda_id: item.id,
                cause: SuppressionCause::MaxSurfaces,
                urgency,
            });
            continue;
        }

        // ── Gate 4: Minimum resurface interval ──
        if let Some(last) = item.last_surfaced_at {
            if now - last < config.min_resurface_interval_secs {
                suppressed.push(SuppressedItem {
                    agenda_id: item.id,
                    cause: SuppressionCause::TooSoon,
                    urgency,
                });
                continue;
            }
        }

        // ── Gate 5: Rate limit ──
        if rate_limiter.is_rate_limited(now, config) {
            // Exception: preempt-worthy items bypass rate limit
            if urgency < config.preempt_urgency_threshold {
                suppressed.push(SuppressedItem {
                    agenda_id: item.id,
                    cause: SuppressionCause::RateLimited,
                    urgency,
                });
                continue;
            }
        }

        // ── Gate 6: Quiet hours (unless critical urgency) ──
        if receptivity.is_quiet_hours && urgency < config.preempt_urgency_threshold {
            suppressed.push(SuppressedItem {
                agenda_id: item.id,
                cause: SuppressionCause::QuietHours,
                urgency,
            });
            continue;
        }

        // ── Gate 7: Receptivity check (with urgency override) ──
        // High-urgency items get a pass even with low receptivity
        let effective_receptivity_threshold = if urgency >= config.alert_urgency_threshold {
            config.receptivity_threshold * 0.5 // Halved threshold for alerts
        } else {
            config.receptivity_threshold
        };

        if receptivity.score < effective_receptivity_threshold
            && urgency < config.preempt_urgency_threshold
        {
            suppressed.push(SuppressedItem {
                agenda_id: item.id,
                cause: SuppressionCause::LowReceptivity,
                urgency,
            });
            continue;
        }

        // ── Gate 8: Item-level suppression rules ──
        let hour = ((now % 86400.0) / 3600.0) as u8;
        let cognitive_load = match context.activity {
            ActivityState::DeepFocus => 0.95,
            ActivityState::FocusedWork => 0.75,
            ActivityState::Communicating => 0.6,
            ActivityState::TaskSwitching => 0.4,
            ActivityState::Browsing => 0.3,
            ActivityState::JustReturned => 0.2,
            ActivityState::Idle => 0.1,
        };
        let is_shared = false; // TODO: derive from context when screen-sharing detection exists

        if item.is_suppressed(hour, cognitive_load, is_shared) {
            suppressed.push(SuppressedItem {
                agenda_id: item.id,
                cause: SuppressionCause::ItemSuppressionRule,
                urgency,
            });
            continue;
        }

        // ── Mode selection ──
        let reason = reason_from_kind(item.kind);
        let mode = select_surface_mode(urgency, reason, context.activity, config);

        // ── Gate 9: Notification mode filter ──
        if !is_mode_allowed(mode, context.notification_mode) {
            suppressed.push(SuppressedItem {
                agenda_id: item.id,
                cause: SuppressionCause::NotificationModeBlock,
                urgency,
            });
            continue;
        }

        // ── Compute net utility ──
        let importance = reason.base_importance();
        let disruption = mode.disruption_cost();
        let net_utility =
            compute_net_utility(urgency, importance, receptivity.score, disruption, config);

        suggestions.push(ProactiveSuggestion {
            agenda_id: item.id,
            description: item.description.clone(),
            kind: item.kind,
            mode,
            reason,
            confidence: receptivity.score * urgency,
            urgency,
            disruption_cost: disruption,
            net_utility,
        });
    }

    // Sort by net utility (highest first)
    suggestions.sort_by(|a, b| {
        b.net_utility
            .partial_cmp(&a.net_utility)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Limit to max_suggestions
    suggestions.truncate(config.max_suggestions);

    SurfacingResult {
        suggestions,
        suppressed,
        items_evaluated,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 11  Feedback Learning
// ══════════════════════════════════════════════════════════════════════════════

/// Outcome of a surfaced suggestion — used to update surfacing thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurfaceOutcome {
    /// User acted on the suggestion.
    Acted,
    /// User acknowledged but deferred.
    Deferred,
    /// User explicitly dismissed.
    Dismissed,
    /// Suggestion expired without user interaction.
    Expired,
    /// User found it annoying (explicit negative feedback).
    Annoyed,
}

/// Persistent surfacing preferences learned from user behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfacingPreferences {
    /// Per-kind acceptance rates (kind_str → (accepted, total)).
    pub kind_acceptance: std::collections::HashMap<String, (u32, u32)>,
    /// Per-mode acceptance rates (mode_str → (accepted, total)).
    pub mode_acceptance: std::collections::HashMap<String, (u32, u32)>,
    /// Per-hour acceptance rates (hour → (accepted, total)).
    pub hourly_acceptance: [AcceptanceRate; 24],
    /// Overall learned urgency threshold adjustment.
    /// Positive = user wants fewer suggestions (raise threshold).
    /// Negative = user wants more (lower threshold).
    pub threshold_adjustment: f64,
    /// Total feedback events received.
    pub total_feedback: u64,
}

/// Simple acceptance rate tracker.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct AcceptanceRate {
    pub accepted: u32,
    pub total: u32,
}

impl AcceptanceRate {
    /// Acceptance rate with Laplace smoothing.
    pub fn rate(&self) -> f64 {
        (self.accepted as f64 + 1.0) / (self.total as f64 + 2.0)
    }
}

impl Default for SurfacingPreferences {
    fn default() -> Self {
        Self {
            kind_acceptance: std::collections::HashMap::new(),
            mode_acceptance: std::collections::HashMap::new(),
            hourly_acceptance: [AcceptanceRate::default(); 24],
            threshold_adjustment: 0.0,
            total_feedback: 0,
        }
    }
}

impl SurfacingPreferences {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record feedback for a surfaced suggestion.
    pub fn observe(
        &mut self,
        kind: AgendaKind,
        mode: SurfaceMode,
        hour: u8,
        outcome: SurfaceOutcome,
    ) {
        let acted = matches!(outcome, SurfaceOutcome::Acted);
        let negative = matches!(outcome, SurfaceOutcome::Dismissed | SurfaceOutcome::Annoyed);

        // Update kind acceptance
        let kind_entry = self
            .kind_acceptance
            .entry(kind.as_str().to_string())
            .or_insert((0, 0));
        if acted {
            kind_entry.0 += 1;
        }
        kind_entry.1 += 1;

        // Update mode acceptance
        let mode_entry = self
            .mode_acceptance
            .entry(mode.as_str().to_string())
            .or_insert((0, 0));
        if acted {
            mode_entry.0 += 1;
        }
        mode_entry.1 += 1;

        // Update hourly acceptance
        let h = (hour as usize).min(23);
        if acted {
            self.hourly_acceptance[h].accepted += 1;
        }
        self.hourly_acceptance[h].total += 1;

        // Adjust threshold
        let lr = 0.01; // Learning rate
        if negative {
            self.threshold_adjustment += lr; // Raise threshold
        } else if acted {
            self.threshold_adjustment -= lr * 0.5; // Cautiously lower
        }
        self.threshold_adjustment = self.threshold_adjustment.clamp(-0.2, 0.3);

        self.total_feedback += 1;
    }

    /// Get the learned effective urgency threshold.
    pub fn effective_threshold(&self, base_threshold: f64) -> f64 {
        (base_threshold + self.threshold_adjustment).clamp(0.1, 0.9)
    }

    /// Get acceptance rate for a specific kind.
    pub fn kind_rate(&self, kind: AgendaKind) -> f64 {
        self.kind_acceptance
            .get(kind.as_str())
            .map(|&(a, t)| (a as f64 + 1.0) / (t as f64 + 2.0))
            .unwrap_or(0.5) // Prior: 50%
    }

    /// Get acceptance rate for a specific hour.
    pub fn hour_rate(&self, hour: u8) -> f64 {
        self.hourly_acceptance[(hour as usize).min(23)].rate()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 12  Convenience — apply preferences to pipeline
// ══════════════════════════════════════════════════════════════════════════════

/// Run the surfacing pipeline with learned preferences applied.
///
/// Adjusts urgency threshold and re-ranks based on per-kind acceptance rates.
pub fn run_surfacing_with_preferences(
    items: &[AgendaItem],
    now: f64,
    receptivity: &ReceptivityEstimate,
    rate_limiter: &SurfaceRateLimiter,
    context: &ContextSnapshot,
    config: &SurfacingConfig,
    preferences: &SurfacingPreferences,
) -> SurfacingResult {
    // Adjust config with learned preferences
    let mut adjusted_config = config.clone();
    adjusted_config.urgency_threshold = preferences.effective_threshold(config.urgency_threshold);

    let mut result = run_surfacing_pipeline(
        items,
        now,
        receptivity,
        rate_limiter,
        context,
        &adjusted_config,
    );

    // Re-score with acceptance rate boost
    let hour = ((now % 86400.0) / 3600.0) as u8;
    let hour_rate = preferences.hour_rate(hour);

    for suggestion in &mut result.suggestions {
        let kind_rate = preferences.kind_rate(suggestion.kind);
        // Blend acceptance rate into net utility
        let acceptance_boost = (kind_rate + hour_rate) / 2.0 - 0.5; // [-0.5, 0.5]
        suggestion.net_utility = (suggestion.net_utility + acceptance_boost * 0.1).clamp(0.0, 1.0);
    }

    // Re-sort after adjustment
    result.suggestions.sort_by(|a, b| {
        b.net_utility
            .partial_cmp(&a.net_utility)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    result
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::super::agenda::{Agenda, AgendaConfig, AgendaKind, UrgencyFn};
    use super::super::state::NodeId;
    use super::*;

    fn make_receptivity(score: f64, quiet: bool) -> ReceptivityEstimate {
        ReceptivityEstimate {
            score,
            factors: vec![],
            is_quiet_hours: quiet,
            budget_remaining: 10,
        }
    }

    fn make_context(activity: ActivityState, mode: NotificationMode) -> ContextSnapshot {
        ContextSnapshot {
            now: 1_700_000_000.0,
            activity,
            recent_interactions_15min: 5,
            recent_outcomes: (3, 0, 0),
            secs_since_last_interaction: 30.0,
            session_duration_secs: 600.0,
            emotional_valence: 0.0,
            session_suggestions_accepted: 2,
            session_suggestion_budget: 20,
            notification_mode: mode,
        }
    }

    fn make_test_items() -> (Agenda, Vec<AgendaItem>) {
        let mut agenda = Agenda::new();
        let config = AgendaConfig::default();
        let now = 1_700_000_000.0;

        // High urgency deadline
        agenda.add_item_at(
            NodeId::NIL,
            AgendaKind::DeadlineApproaching,
            UrgencyFn::Constant { value: 0.9 },
            Some(now + 300.0),
            "Submit report".to_string(),
            now - 3600.0,
            &config,
        );

        // Medium urgency follow-up
        agenda.add_item_at(
            NodeId::NIL,
            AgendaKind::FollowUpNeeded,
            UrgencyFn::Constant { value: 0.6 },
            None,
            "Follow up with vendor".to_string(),
            now - 7200.0,
            &config,
        );

        // Low urgency routine
        agenda.add_item_at(
            NodeId::NIL,
            AgendaKind::RoutineWindowOpening,
            UrgencyFn::Constant { value: 0.45 },
            None,
            "Check email".to_string(),
            now - 600.0,
            &config,
        );

        // Below threshold
        agenda.add_item_at(
            NodeId::NIL,
            AgendaKind::StalledIntent,
            UrgencyFn::Constant { value: 0.2 },
            None,
            "Organize bookmarks".to_string(),
            now - 86400.0,
            &config,
        );

        let items = agenda.items.clone();
        (agenda, items)
    }

    #[test]
    fn test_basic_pipeline() {
        let (_agenda, items) = make_test_items();
        let receptivity = make_receptivity(0.7, false);
        let rate_limiter = SurfaceRateLimiter::new();
        let context = make_context(ActivityState::TaskSwitching, NotificationMode::All);
        let config = SurfacingConfig::default();
        let now = 1_700_000_000.0;

        let result =
            run_surfacing_pipeline(&items, now, &receptivity, &rate_limiter, &context, &config);

        assert_eq!(result.items_evaluated, 4);
        // Should surface at least the high urgency item
        assert!(!result.suggestions.is_empty());
        // Should NOT surface the below-threshold item
        assert!(
            !result
                .suggestions
                .iter()
                .any(|s| s.description == "Organize bookmarks"),
            "Below-threshold item should not surface"
        );
    }

    #[test]
    fn test_mode_selection_urgency_levels() {
        let config = SurfacingConfig::default();

        // Critical urgency deadline → Preempt
        let mode = select_surface_mode(
            0.96,
            SurfaceReason::DeadlineImminent,
            ActivityState::Idle,
            &config,
        );
        assert_eq!(mode, SurfaceMode::Preempt);

        // High urgency → Alert
        let mode = select_surface_mode(
            0.85,
            SurfaceReason::UrgencyThreshold,
            ActivityState::Idle,
            &config,
        );
        assert_eq!(mode, SurfaceMode::Alert);

        // Medium urgency → Nudge
        let mode = select_surface_mode(
            0.55,
            SurfaceReason::FollowUpDue,
            ActivityState::Idle,
            &config,
        );
        assert_eq!(mode, SurfaceMode::Nudge);

        // Low urgency but at natural break → Nudge
        let mode = select_surface_mode(
            0.42,
            SurfaceReason::RoutineWindowOpening,
            ActivityState::Idle,
            &config,
        );
        assert_eq!(mode, SurfaceMode::Nudge);

        // Low urgency in focus → Whisper
        let mode = select_surface_mode(
            0.42,
            SurfaceReason::RoutineWindowOpening,
            ActivityState::FocusedWork,
            &config,
        );
        assert_eq!(mode, SurfaceMode::Whisper);
    }

    #[test]
    fn test_quiet_hours_suppression() {
        let (_agenda, items) = make_test_items();
        let receptivity = make_receptivity(0.7, true); // Quiet hours ON
        let rate_limiter = SurfaceRateLimiter::new();
        let context = make_context(ActivityState::Idle, NotificationMode::All);
        let config = SurfacingConfig::default();
        let now = 1_700_000_000.0;

        let result =
            run_surfacing_pipeline(&items, now, &receptivity, &rate_limiter, &context, &config);

        // All non-critical items should be suppressed
        let quiet_suppressed: Vec<_> = result
            .suppressed
            .iter()
            .filter(|s| s.cause == SuppressionCause::QuietHours)
            .collect();
        assert!(
            !quiet_suppressed.is_empty(),
            "Items should be suppressed during quiet hours"
        );
    }

    #[test]
    fn test_dnd_mode_blocks_all() {
        let (_agenda, items) = make_test_items();
        let receptivity = make_receptivity(0.7, false);
        let rate_limiter = SurfaceRateLimiter::new();
        let context = make_context(ActivityState::Idle, NotificationMode::DoNotDisturb);
        let config = SurfacingConfig::default();
        let now = 1_700_000_000.0;

        let result =
            run_surfacing_pipeline(&items, now, &receptivity, &rate_limiter, &context, &config);

        assert!(
            result.suggestions.is_empty(),
            "DND should block all suggestions"
        );
        let mode_blocked: Vec<_> = result
            .suppressed
            .iter()
            .filter(|s| s.cause == SuppressionCause::NotificationModeBlock)
            .collect();
        assert!(!mode_blocked.is_empty());
    }

    #[test]
    fn test_rate_limiting() {
        let (_agenda, items) = make_test_items();
        let receptivity = make_receptivity(0.7, false);
        let mut rate_limiter = SurfaceRateLimiter::new();
        let context = make_context(ActivityState::Idle, NotificationMode::All);
        let config = SurfacingConfig {
            max_surfaces_per_hour: 2,
            ..Default::default()
        };
        let now = 1_700_000_000.0;

        // Record 2 recent surfaces
        rate_limiter.record_surface(now - 60.0);
        rate_limiter.record_surface(now - 30.0);

        let result =
            run_surfacing_pipeline(&items, now, &receptivity, &rate_limiter, &context, &config);

        // Most items should be rate-limited
        let rate_limited: Vec<_> = result
            .suppressed
            .iter()
            .filter(|s| s.cause == SuppressionCause::RateLimited)
            .collect();
        assert!(!rate_limited.is_empty(), "Should hit rate limit");
    }

    #[test]
    fn test_anti_nag_suppression() {
        let mut agenda = Agenda::new();
        let config = AgendaConfig::default();
        let now = 1_700_000_000.0;

        let id = agenda.add_item_at(
            NodeId::NIL,
            AgendaKind::FollowUpNeeded,
            UrgencyFn::Constant { value: 0.7 },
            None,
            "Nagging item".to_string(),
            now - 3600.0,
            &config,
        );

        // Dismiss 3 times
        for _ in 0..3 {
            agenda.dismiss(id);
            agenda.items.last_mut().unwrap().status = AgendaStatus::Active;
        }

        let items = agenda.items.clone();
        let receptivity = make_receptivity(0.7, false);
        let rate_limiter = SurfaceRateLimiter::new();
        let context = make_context(ActivityState::Idle, NotificationMode::All);
        let surf_config = SurfacingConfig::default();

        let result = run_surfacing_pipeline(
            &items,
            now,
            &receptivity,
            &rate_limiter,
            &context,
            &surf_config,
        );

        let anti_nag: Vec<_> = result
            .suppressed
            .iter()
            .filter(|s| s.cause == SuppressionCause::AntiNag)
            .collect();
        assert!(!anti_nag.is_empty(), "Should suppress nagging items");
    }

    #[test]
    fn test_low_receptivity_suppression() {
        let (_agenda, items) = make_test_items();
        let receptivity = make_receptivity(0.1, false); // Very low receptivity
        let rate_limiter = SurfaceRateLimiter::new();
        let context = make_context(ActivityState::DeepFocus, NotificationMode::All);
        let config = SurfacingConfig::default();
        let now = 1_700_000_000.0;

        let result =
            run_surfacing_pipeline(&items, now, &receptivity, &rate_limiter, &context, &config);

        let low_recep: Vec<_> = result
            .suppressed
            .iter()
            .filter(|s| s.cause == SuppressionCause::LowReceptivity)
            .collect();
        assert!(
            !low_recep.is_empty(),
            "Should suppress when user is not receptive"
        );
    }

    #[test]
    fn test_net_utility_ordering() {
        let config = SurfacingConfig::default();

        // High urgency + high receptivity = high utility
        let u1 = compute_net_utility(0.9, 0.8, 0.9, 0.25, &config);
        // Low urgency + low receptivity = low utility
        let u2 = compute_net_utility(0.3, 0.4, 0.3, 0.25, &config);

        assert!(
            u1 > u2,
            "High urgency should have higher net utility: {} vs {}",
            u1,
            u2
        );
    }

    #[test]
    fn test_net_utility_disruption_penalty() {
        let config = SurfacingConfig::default();

        // Same benefit, different disruption costs
        let u_low = compute_net_utility(0.7, 0.6, 0.7, 0.05, &config); // Whisper
        let u_high = compute_net_utility(0.7, 0.6, 0.7, 0.95, &config); // Preempt

        assert!(
            u_low > u_high,
            "Higher disruption should lower utility: {} vs {}",
            u_low,
            u_high
        );
    }

    #[test]
    fn test_suggestions_sorted_by_utility() {
        let (_agenda, items) = make_test_items();
        let receptivity = make_receptivity(0.7, false);
        let rate_limiter = SurfaceRateLimiter::new();
        let context = make_context(ActivityState::TaskSwitching, NotificationMode::All);
        let config = SurfacingConfig {
            max_suggestions: 10,
            ..Default::default()
        };
        let now = 1_700_000_000.0;

        let result =
            run_surfacing_pipeline(&items, now, &receptivity, &rate_limiter, &context, &config);

        // Verify sorted by net utility descending
        for window in result.suggestions.windows(2) {
            assert!(
                window[0].net_utility >= window[1].net_utility,
                "Should be sorted by net utility: {} >= {}",
                window[0].net_utility,
                window[1].net_utility,
            );
        }
    }

    #[test]
    fn test_max_suggestions_limit() {
        let mut agenda = Agenda::new();
        let config = AgendaConfig::default();
        let now = 1_700_000_000.0;

        // Add 10 high-urgency items
        for i in 0..10 {
            agenda.add_item_at(
                NodeId::NIL,
                AgendaKind::FollowUpNeeded,
                UrgencyFn::Constant { value: 0.7 },
                None,
                format!("Item {}", i),
                now - 3600.0,
                &config,
            );
        }

        let items = agenda.items.clone();
        let receptivity = make_receptivity(0.7, false);
        let rate_limiter = SurfaceRateLimiter::new();
        let context = make_context(ActivityState::Idle, NotificationMode::All);
        let surf_config = SurfacingConfig {
            max_suggestions: 3,
            ..Default::default()
        };

        let result = run_surfacing_pipeline(
            &items,
            now,
            &receptivity,
            &rate_limiter,
            &context,
            &surf_config,
        );

        assert!(
            result.suggestions.len() <= 3,
            "Should respect max_suggestions: got {}",
            result.suggestions.len()
        );
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = SurfaceRateLimiter::new();
        let now = 1_700_000_000.0;
        let config = SurfacingConfig::default(); // 10 per hour

        assert_eq!(limiter.surfaces_last_hour(now), 0);
        assert!(!limiter.is_rate_limited(now, &config));

        for i in 0..10 {
            limiter.record_surface(now - i as f64 * 60.0);
        }

        assert_eq!(limiter.surfaces_last_hour(now), 10);
        assert!(limiter.is_rate_limited(now, &config));
    }

    #[test]
    fn test_preferences_learning() {
        let mut prefs = SurfacingPreferences::new();

        // Simulate: user likes follow-up nudges at 10am, hates routine whispers at 3am
        for _ in 0..20 {
            prefs.observe(
                AgendaKind::FollowUpNeeded,
                SurfaceMode::Nudge,
                10,
                SurfaceOutcome::Acted,
            );
        }
        for _ in 0..10 {
            prefs.observe(
                AgendaKind::RoutineWindowOpening,
                SurfaceMode::Whisper,
                3,
                SurfaceOutcome::Dismissed,
            );
        }

        let follow_up_rate = prefs.kind_rate(AgendaKind::FollowUpNeeded);
        let routine_rate = prefs.kind_rate(AgendaKind::RoutineWindowOpening);

        assert!(
            follow_up_rate > routine_rate,
            "Follow-up rate ({:.2}) should exceed routine rate ({:.2})",
            follow_up_rate,
            routine_rate,
        );

        assert!(
            prefs.hour_rate(10) > prefs.hour_rate(3),
            "10am rate ({:.2}) should exceed 3am rate ({:.2})",
            prefs.hour_rate(10),
            prefs.hour_rate(3),
        );

        // Test threshold direction with only dismissals
        let mut dismiss_prefs = SurfacingPreferences::new();
        for _ in 0..10 {
            dismiss_prefs.observe(
                AgendaKind::RoutineWindowOpening,
                SurfaceMode::Whisper,
                3,
                SurfaceOutcome::Dismissed,
            );
        }
        assert!(
            dismiss_prefs.threshold_adjustment > 0.0,
            "Pure dismissals should raise threshold: {}",
            dismiss_prefs.threshold_adjustment,
        );
    }

    #[test]
    fn test_notification_mode_filtering() {
        assert!(is_mode_allowed(SurfaceMode::Whisper, NotificationMode::All));
        assert!(is_mode_allowed(SurfaceMode::Alert, NotificationMode::All));
        assert!(is_mode_allowed(
            SurfaceMode::Alert,
            NotificationMode::ImportantOnly
        ));
        assert!(!is_mode_allowed(
            SurfaceMode::Nudge,
            NotificationMode::ImportantOnly
        ));
        assert!(!is_mode_allowed(
            SurfaceMode::Preempt,
            NotificationMode::DoNotDisturb
        ));
    }

    #[test]
    fn test_surface_mode_ordering() {
        // Modes should have a total order for notification filtering
        assert!(SurfaceMode::Whisper < SurfaceMode::Nudge);
        assert!(SurfaceMode::Nudge < SurfaceMode::Alert);
        assert!(SurfaceMode::Alert < SurfaceMode::Preempt);
    }

    #[test]
    fn test_disruption_cost_ordering() {
        assert!(SurfaceMode::Whisper.disruption_cost() < SurfaceMode::Nudge.disruption_cost());
        assert!(SurfaceMode::Nudge.disruption_cost() < SurfaceMode::Alert.disruption_cost());
        assert!(SurfaceMode::Alert.disruption_cost() < SurfaceMode::Preempt.disruption_cost());
    }

    #[test]
    fn test_preempt_bypasses_rate_limit() {
        let mut agenda = Agenda::new();
        let aconfig = AgendaConfig::default();
        let now = 1_700_000_000.0;

        // Critical urgency deadline
        agenda.add_item_at(
            NodeId::NIL,
            AgendaKind::DeadlineApproaching,
            UrgencyFn::Constant { value: 0.96 },
            Some(now + 60.0),
            "Critical deadline in 1 min".to_string(),
            now - 3600.0,
            &aconfig,
        );

        let items = agenda.items.clone();
        let receptivity = make_receptivity(0.7, false);
        let mut rate_limiter = SurfaceRateLimiter::new();
        // Saturate rate limiter
        for i in 0..20 {
            rate_limiter.record_surface(now - i as f64 * 60.0);
        }
        let context = make_context(ActivityState::Idle, NotificationMode::All);
        let config = SurfacingConfig {
            max_surfaces_per_hour: 5,
            ..Default::default()
        };

        let result =
            run_surfacing_pipeline(&items, now, &receptivity, &rate_limiter, &context, &config);

        // Critical item should bypass rate limit
        assert!(
            result
                .suggestions
                .iter()
                .any(|s| s.mode == SurfaceMode::Preempt),
            "Critical deadline should bypass rate limit and surface as preempt"
        );
    }

    #[test]
    fn test_with_preferences() {
        let (_agenda, items) = make_test_items();
        let receptivity = make_receptivity(0.7, false);
        let rate_limiter = SurfaceRateLimiter::new();
        let context = make_context(ActivityState::TaskSwitching, NotificationMode::All);
        let config = SurfacingConfig::default();
        let prefs = SurfacingPreferences::new();
        let now = 1_700_000_000.0;

        let result = run_surfacing_with_preferences(
            &items,
            now,
            &receptivity,
            &rate_limiter,
            &context,
            &config,
            &prefs,
        );

        // Should still work with default preferences
        assert_eq!(result.items_evaluated, 4);
    }
}
