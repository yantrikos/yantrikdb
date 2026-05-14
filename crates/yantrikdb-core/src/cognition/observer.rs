//! Passive Event Observer — the foundation of autonomous learning.
//!
//! Watches system-wide events without requiring explicit feedback.
//! All data is structured metadata — NO raw text content is stored.
//!
//! # Architecture
//!
//! ```text
//! Producers ──(observe)──► EventBuffer (circular, 10K capacity)
//!                              │
//!                              ├──► EventCounters  (per-kind totals + sliding windows)
//!                              ├──► EventHistogram (24-hour circadian distribution)
//!                              └──► EventRateTracker (per-kind events/sec)
//! ```
//!
//! # Privacy guarantees
//! - No raw text stored — only structured metadata
//! - Event types individually disableable via `ObserverConfig`
//! - All data local, never transmitted
//! - Configurable retention period

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

// ══════════════════════════════════════════════════════════════════════════════
// § 1  Event Kind Discriminator
// ══════════════════════════════════════════════════════════════════════════════

/// Discriminator for event types — used as hash map keys and filter predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum EventKind {
    AppOpened = 0,
    AppClosed = 1,
    AppSequence = 2,
    NotificationReceived = 3,
    NotificationDismissed = 4,
    NotificationActedOn = 5,
    SuggestionAccepted = 6,
    SuggestionRejected = 7,
    SuggestionIgnored = 8,
    SuggestionModified = 9,
    QueryRepeated = 10,
    UserTyping = 11,
    UserIdle = 12,
    ToolCallCompleted = 13,
    LlmCalled = 14,
    ErrorOccurred = 15,
}

impl EventKind {
    /// All event kinds for iteration.
    pub const ALL: [EventKind; 16] = [
        Self::AppOpened,
        Self::AppClosed,
        Self::AppSequence,
        Self::NotificationReceived,
        Self::NotificationDismissed,
        Self::NotificationActedOn,
        Self::SuggestionAccepted,
        Self::SuggestionRejected,
        Self::SuggestionIgnored,
        Self::SuggestionModified,
        Self::QueryRepeated,
        Self::UserTyping,
        Self::UserIdle,
        Self::ToolCallCompleted,
        Self::LlmCalled,
        Self::ErrorOccurred,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::AppOpened => "app_opened",
            Self::AppClosed => "app_closed",
            Self::AppSequence => "app_sequence",
            Self::NotificationReceived => "notification_received",
            Self::NotificationDismissed => "notification_dismissed",
            Self::NotificationActedOn => "notification_acted_on",
            Self::SuggestionAccepted => "suggestion_accepted",
            Self::SuggestionRejected => "suggestion_rejected",
            Self::SuggestionIgnored => "suggestion_ignored",
            Self::SuggestionModified => "suggestion_modified",
            Self::QueryRepeated => "query_repeated",
            Self::UserTyping => "user_typing",
            Self::UserIdle => "user_idle",
            Self::ToolCallCompleted => "tool_call_completed",
            Self::LlmCalled => "llm_called",
            Self::ErrorOccurred => "error_occurred",
        }
    }

    /// Whether this event kind carries a learning signal about user preferences.
    pub fn is_preference_signal(self) -> bool {
        matches!(
            self,
            Self::SuggestionAccepted
                | Self::SuggestionRejected
                | Self::SuggestionIgnored
                | Self::SuggestionModified
                | Self::NotificationDismissed
                | Self::NotificationActedOn
        )
    }

    /// Whether this event kind indicates user activity (vs system events).
    pub fn is_user_activity(self) -> bool {
        matches!(
            self,
            Self::AppOpened
                | Self::AppClosed
                | Self::AppSequence
                | Self::UserTyping
                | Self::UserIdle
                | Self::NotificationActedOn
                | Self::SuggestionAccepted
                | Self::SuggestionModified
                | Self::QueryRepeated
        )
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 2  System Event
// ══════════════════════════════════════════════════════════════════════════════

/// A system-wide event observed by the passive observer.
///
/// All fields are structured metadata — no raw text content.
/// Timestamps are Unix seconds (f64) for consistency with the cognitive tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemEvent {
    /// When this event occurred (Unix seconds, f64).
    pub timestamp: f64,
    /// The event payload.
    pub data: SystemEventData,
}

/// Event payload — typed variants for each observable event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemEventData {
    // ── App lifecycle ──
    AppOpened {
        app_id: u16,
    },
    AppClosed {
        app_id: u16,
        duration_ms: u64,
    },
    AppSequence {
        from_app: u16,
        to_app: u16,
        gap_ms: u64,
    },

    // ── Notifications ──
    NotificationReceived {
        notif_type: String,
        source: String,
    },
    NotificationDismissed {
        notif_type: String,
        time_to_dismiss_ms: u64,
    },
    NotificationActedOn {
        notif_type: String,
        action_taken: String,
    },

    // ── Suggestions (learning signals) ──
    SuggestionAccepted {
        suggestion_id: u64,
        action_kind: String,
        latency_ms: u64,
    },
    SuggestionRejected {
        suggestion_id: u64,
        action_kind: String,
    },
    SuggestionIgnored {
        suggestion_id: u64,
        action_kind: String,
        timeout_ms: u64,
    },
    SuggestionModified {
        suggestion_id: u64,
        /// BLAKE3 hash of original suggestion (privacy: no raw text).
        original_hash: u64,
        /// BLAKE3 hash of modified version.
        modified_hash: u64,
    },

    // ── User behavior ──
    QueryRepeated {
        /// BLAKE3 hash of query text (privacy: no raw text).
        query_hash: u64,
        count: u32,
    },
    UserTyping {
        app_id: u16,
        duration_ms: u64,
        characters: u32,
    },
    UserIdle {
        duration_ms: u64,
    },

    // ── System operations ──
    ToolCallCompleted {
        tool_name: String,
        success: bool,
        duration_ms: u64,
    },
    LlmCalled {
        reason: String,
        tokens_used: u32,
        latency_ms: u64,
    },
    ErrorOccurred {
        app_id: u16,
        error_type: String,
    },
}

impl SystemEvent {
    /// Create a new event with the current timestamp placeholder.
    /// Callers should set `timestamp` to `now()` from the engine layer.
    pub fn new(timestamp: f64, data: SystemEventData) -> Self {
        Self { timestamp, data }
    }

    /// Get the event kind discriminator.
    pub fn kind(&self) -> EventKind {
        match &self.data {
            SystemEventData::AppOpened { .. } => EventKind::AppOpened,
            SystemEventData::AppClosed { .. } => EventKind::AppClosed,
            SystemEventData::AppSequence { .. } => EventKind::AppSequence,
            SystemEventData::NotificationReceived { .. } => EventKind::NotificationReceived,
            SystemEventData::NotificationDismissed { .. } => EventKind::NotificationDismissed,
            SystemEventData::NotificationActedOn { .. } => EventKind::NotificationActedOn,
            SystemEventData::SuggestionAccepted { .. } => EventKind::SuggestionAccepted,
            SystemEventData::SuggestionRejected { .. } => EventKind::SuggestionRejected,
            SystemEventData::SuggestionIgnored { .. } => EventKind::SuggestionIgnored,
            SystemEventData::SuggestionModified { .. } => EventKind::SuggestionModified,
            SystemEventData::QueryRepeated { .. } => EventKind::QueryRepeated,
            SystemEventData::UserTyping { .. } => EventKind::UserTyping,
            SystemEventData::UserIdle { .. } => EventKind::UserIdle,
            SystemEventData::ToolCallCompleted { .. } => EventKind::ToolCallCompleted,
            SystemEventData::LlmCalled { .. } => EventKind::LlmCalled,
            SystemEventData::ErrorOccurred { .. } => EventKind::ErrorOccurred,
        }
    }

    /// Extract the app_id if this event involves an app.
    pub fn app_id(&self) -> Option<u16> {
        match &self.data {
            SystemEventData::AppOpened { app_id }
            | SystemEventData::AppClosed { app_id, .. }
            | SystemEventData::UserTyping { app_id, .. }
            | SystemEventData::ErrorOccurred { app_id, .. } => Some(*app_id),
            SystemEventData::AppSequence { to_app, .. } => Some(*to_app),
            _ => None,
        }
    }

    /// Whether this is an annoyance signal (notification dismissed in <500ms).
    pub fn is_annoyance_signal(&self) -> bool {
        matches!(
            &self.data,
            SystemEventData::NotificationDismissed {
                time_to_dismiss_ms, ..
            } if *time_to_dismiss_ms < 500
        )
    }

    /// Extract duration_ms if this event carries one.
    pub fn duration_ms(&self) -> Option<u64> {
        match &self.data {
            SystemEventData::AppClosed { duration_ms, .. }
            | SystemEventData::UserTyping { duration_ms, .. }
            | SystemEventData::UserIdle { duration_ms, .. }
            | SystemEventData::ToolCallCompleted { duration_ms, .. }
            | SystemEventData::LlmCalled {
                latency_ms: duration_ms,
                ..
            } => Some(*duration_ms),
            _ => None,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  Observer Configuration
// ══════════════════════════════════════════════════════════════════════════════

/// Configuration for the passive event observer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObserverConfig {
    /// Maximum events in the circular buffer.
    pub buffer_capacity: usize,
    /// Event kinds that are disabled (not observed).
    pub disabled_kinds: Vec<EventKind>,
    /// Retention period in seconds — events older than this are pruned on persistence.
    pub retention_secs: f64,
    /// Batch persistence threshold — persist after this many new events.
    pub batch_persist_threshold: usize,
    /// Batch persistence interval — persist after this many seconds.
    pub batch_persist_interval_secs: f64,
    /// Maximum events per second across all kinds (rate limiting).
    pub max_events_per_sec: f64,
    /// Sliding window size for rate calculations (seconds).
    pub rate_window_secs: f64,
}

impl Default for ObserverConfig {
    fn default() -> Self {
        Self {
            buffer_capacity: 10_000,
            disabled_kinds: Vec::new(),
            retention_secs: 30.0 * 86400.0, // 30 days
            batch_persist_threshold: 100,
            batch_persist_interval_secs: 60.0,
            max_events_per_sec: 100.0,
            rate_window_secs: 300.0, // 5-minute window
        }
    }
}

impl ObserverConfig {
    /// Check whether an event kind is enabled.
    pub fn is_enabled(&self, kind: EventKind) -> bool {
        !self.disabled_kinds.contains(&kind)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  Event Buffer (Circular)
// ══════════════════════════════════════════════════════════════════════════════

/// Circular buffer for recent events. O(1) push, O(n) scan.
///
/// Backed by `VecDeque` with enforced capacity. When full, oldest events
/// are evicted. Memory: ~100 bytes/event × 10K = ~1MB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventBuffer {
    events: VecDeque<SystemEvent>,
    capacity: usize,
    /// Total events ever ingested (monotonic, never resets).
    pub total_ingested: u64,
    /// Count of events dropped due to rate limiting.
    pub total_dropped: u64,
}

impl EventBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(capacity.min(1024)), // lazy alloc
            capacity,
            total_ingested: 0,
            total_dropped: 0,
        }
    }

    /// Push an event into the buffer. Evicts oldest if full.
    pub fn push(&mut self, event: SystemEvent) {
        if self.events.len() >= self.capacity {
            self.events.pop_front();
        }
        self.events.push_back(event);
        self.total_ingested += 1;
    }

    /// Record a dropped event (rate limited).
    pub fn record_drop(&mut self) {
        self.total_dropped += 1;
    }

    /// Number of events currently in the buffer.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Get the most recent N events (newest first).
    pub fn recent(&self, limit: usize) -> Vec<&SystemEvent> {
        self.events.iter().rev().take(limit).collect()
    }

    /// Get events matching a filter predicate (newest first).
    pub fn filter<F>(&self, limit: usize, predicate: F) -> Vec<&SystemEvent>
    where
        F: Fn(&SystemEvent) -> bool,
    {
        self.events
            .iter()
            .rev()
            .filter(|e| predicate(e))
            .take(limit)
            .collect()
    }

    /// Get events of a specific kind (newest first).
    pub fn by_kind(&self, kind: EventKind, limit: usize) -> Vec<&SystemEvent> {
        self.filter(limit, move |e| e.kind() == kind)
    }

    /// Get events within a time window [since, until].
    pub fn in_window(&self, since: f64, until: f64) -> Vec<&SystemEvent> {
        self.events
            .iter()
            .rev()
            .take_while(|e| e.timestamp >= since)
            .filter(|e| e.timestamp <= until)
            .collect()
    }

    /// Get events since a timestamp (newest first).
    pub fn since(&self, since_ts: f64) -> Vec<&SystemEvent> {
        self.events
            .iter()
            .rev()
            .take_while(|e| e.timestamp >= since_ts)
            .collect()
    }

    /// Drain events older than `cutoff` timestamp. Returns count removed.
    pub fn prune_before(&mut self, cutoff: f64) -> usize {
        let before = self.events.len();
        self.events.retain(|e| e.timestamp >= cutoff);
        before - self.events.len()
    }

    /// Drain all events for batch persistence. Returns owned events.
    pub fn drain_all(&mut self) -> Vec<SystemEvent> {
        self.events.drain(..).collect()
    }

    /// Get the timestamp of the oldest event, if any.
    pub fn oldest_timestamp(&self) -> Option<f64> {
        self.events.front().map(|e| e.timestamp)
    }

    /// Get the timestamp of the newest event, if any.
    pub fn newest_timestamp(&self) -> Option<f64> {
        self.events.back().map(|e| e.timestamp)
    }

    /// Iterate over all events (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &SystemEvent> {
        self.events.iter()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 5  Event Counters
// ══════════════════════════════════════════════════════════════════════════════

/// Per-event-kind counters with sliding window support.
///
/// Maintains both all-time totals and recent timestamps for rate calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventCounters {
    /// All-time count per event kind.
    totals: HashMap<EventKind, u64>,
    /// Recent event timestamps per kind (for sliding window rate calculation).
    /// Bounded: keeps only timestamps within `window_secs`.
    recent_timestamps: HashMap<EventKind, VecDeque<f64>>,
    /// Window size for rate calculations.
    window_secs: f64,
}

impl EventCounters {
    pub fn new(window_secs: f64) -> Self {
        Self {
            totals: HashMap::new(),
            recent_timestamps: HashMap::new(),
            window_secs,
        }
    }

    /// Record an event occurrence.
    pub fn record(&mut self, kind: EventKind, timestamp: f64) {
        *self.totals.entry(kind).or_insert(0) += 1;
        let timestamps = self.recent_timestamps.entry(kind).or_default();
        timestamps.push_back(timestamp);
        // Evict stale timestamps
        let cutoff = timestamp - self.window_secs;
        while timestamps.front().map_or(false, |&t| t < cutoff) {
            timestamps.pop_front();
        }
    }

    /// Get total count for a kind (all-time).
    pub fn total(&self, kind: EventKind) -> u64 {
        self.totals.get(&kind).copied().unwrap_or(0)
    }

    /// Get event rate (events/second) within the sliding window.
    pub fn rate(&self, kind: EventKind, now: f64) -> f64 {
        let timestamps = match self.recent_timestamps.get(&kind) {
            Some(ts) => ts,
            None => return 0.0,
        };
        let cutoff = now - self.window_secs;
        let count = timestamps.iter().filter(|&&t| t >= cutoff).count();
        count as f64 / self.window_secs
    }

    /// Get event count within a specific time window.
    pub fn count_in_window(&self, kind: EventKind, since: f64, until: f64) -> usize {
        self.recent_timestamps
            .get(&kind)
            .map(|ts| ts.iter().filter(|&&t| t >= since && t <= until).count())
            .unwrap_or(0)
    }

    /// Get aggregate rate across all event kinds.
    pub fn total_rate(&self, now: f64) -> f64 {
        EventKind::ALL.iter().map(|&k| self.rate(k, now)).sum()
    }

    /// Prune stale timestamps across all kinds.
    pub fn prune(&mut self, now: f64) {
        let cutoff = now - self.window_secs;
        for timestamps in self.recent_timestamps.values_mut() {
            while timestamps.front().map_or(false, |&t| t < cutoff) {
                timestamps.pop_front();
            }
        }
    }

    /// Get all kinds that have been observed, with their totals.
    pub fn all_totals(&self) -> Vec<(EventKind, u64)> {
        let mut result: Vec<_> = self.totals.iter().map(|(&k, &v)| (k, v)).collect();
        result.sort_by(|a, b| b.1.cmp(&a.1)); // descending by count
        result
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 6  Circadian Histogram
// ══════════════════════════════════════════════════════════════════════════════

/// 24-hour histogram for circadian distribution of events.
///
/// Tracks how events distribute across hours of the day,
/// enabling time-of-day pattern detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircadianHistogram {
    /// Counts per hour (0..23) per event kind.
    buckets: HashMap<EventKind, [u32; 24]>,
}

impl CircadianHistogram {
    pub fn new() -> Self {
        Self {
            buckets: HashMap::new(),
        }
    }

    /// Record an event at a given timestamp.
    pub fn record(&mut self, kind: EventKind, timestamp: f64) {
        let hour = ((timestamp % 86400.0) / 3600.0) as usize;
        let hour = hour.min(23);
        let bins = self.buckets.entry(kind).or_insert([0u32; 24]);
        bins[hour] += 1;
    }

    /// Get the distribution for a specific event kind.
    pub fn distribution(&self, kind: EventKind) -> [u32; 24] {
        self.buckets.get(&kind).copied().unwrap_or([0u32; 24])
    }

    /// Get the peak hour for a specific event kind.
    pub fn peak_hour(&self, kind: EventKind) -> Option<u8> {
        let dist = self.distribution(kind);
        let total: u32 = dist.iter().sum();
        if total == 0 {
            return None;
        }
        let (hour, _) = dist.iter().enumerate().max_by_key(|(_, &count)| count)?;
        Some(hour as u8)
    }

    /// Get the hourly rate normalized to mean=1.0 (for Hawkes circadian modulation).
    pub fn normalized_distribution(&self, kind: EventKind) -> [f64; 24] {
        let dist = self.distribution(kind);
        let total: f64 = dist.iter().map(|&c| c as f64).sum();
        if total == 0.0 {
            return [1.0; 24]; // Uniform prior
        }
        let mean = total / 24.0;
        let mut normalized = [0.0f64; 24];
        for i in 0..24 {
            normalized[i] = dist[i] as f64 / mean;
        }
        normalized
    }

    /// Get the total count for a kind across all hours.
    pub fn total(&self, kind: EventKind) -> u32 {
        self.distribution(kind).iter().sum()
    }

    /// Check if hour `h` is a quiet period for this event kind
    /// (below 0.3× the mean rate).
    pub fn is_quiet_hour(&self, kind: EventKind, h: u8) -> bool {
        let normalized = self.normalized_distribution(kind);
        normalized[h as usize % 24] < 0.3
    }

    /// Check if hour `h` is a peak period for this event kind
    /// (above 2.0× the mean rate).
    pub fn is_peak_hour(&self, kind: EventKind, h: u8) -> bool {
        let normalized = self.normalized_distribution(kind);
        normalized[h as usize % 24] > 2.0
    }
}

impl Default for CircadianHistogram {
    fn default() -> Self {
        Self::new()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 7  Observer State (Serializable Aggregate)
// ══════════════════════════════════════════════════════════════════════════════

/// The complete observer state — serializable for persistence.
///
/// This is what gets saved to and loaded from the database.
/// The event buffer itself may be persisted separately (larger payload).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObserverState {
    /// Per-event-kind counters.
    pub counters: EventCounters,
    /// 24-hour circadian distribution.
    pub histogram: CircadianHistogram,
    /// Configuration.
    pub config: ObserverConfig,
    /// Session start timestamp.
    pub session_start: f64,
    /// Number of events in the current persistence batch (not yet persisted).
    pub pending_batch_count: usize,
    /// Timestamp of last persistence flush.
    pub last_flush_at: f64,
}

impl ObserverState {
    pub fn new() -> Self {
        Self {
            counters: EventCounters::new(300.0), // 5-minute window
            histogram: CircadianHistogram::new(),
            config: ObserverConfig::default(),
            session_start: 0.0,
            pending_batch_count: 0,
            last_flush_at: 0.0,
        }
    }

    pub fn with_config(config: ObserverConfig) -> Self {
        let window = config.rate_window_secs;
        Self {
            counters: EventCounters::new(window),
            histogram: CircadianHistogram::new(),
            config,
            session_start: 0.0,
            pending_batch_count: 0,
            last_flush_at: 0.0,
        }
    }
}

impl Default for ObserverState {
    fn default() -> Self {
        Self::new()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 8  Core Observer Logic (Pure Functions)
// ══════════════════════════════════════════════════════════════════════════════

/// Observe an event: validate, buffer, update counters and histogram.
///
/// Returns `true` if the event was accepted, `false` if filtered/rate-limited.
pub fn observe_event(
    event: SystemEvent,
    buffer: &mut EventBuffer,
    state: &mut ObserverState,
) -> bool {
    let kind = event.kind();

    // Gate 1: Event kind enabled?
    if !state.config.is_enabled(kind) {
        return false;
    }

    // Gate 2: Rate limiting — global events/sec
    let current_rate = state.counters.total_rate(event.timestamp);
    if current_rate > state.config.max_events_per_sec {
        buffer.record_drop();
        return false;
    }

    // Accept: update all tracking structures
    state.counters.record(kind, event.timestamp);
    state.histogram.record(kind, event.timestamp);
    state.pending_batch_count += 1;
    buffer.push(event);

    true
}

/// Check whether a persistence flush is needed.
pub fn needs_flush(state: &ObserverState, now: f64) -> bool {
    if state.pending_batch_count >= state.config.batch_persist_threshold {
        return true;
    }
    if state.pending_batch_count > 0
        && (now - state.last_flush_at) >= state.config.batch_persist_interval_secs
    {
        return true;
    }
    false
}

/// Mark the state as flushed (reset batch counter, update timestamp).
pub fn mark_flushed(state: &mut ObserverState, now: f64) {
    state.pending_batch_count = 0;
    state.last_flush_at = now;
}

// ══════════════════════════════════════════════════════════════════════════════
// § 9  Query API (Pure Functions)
// ══════════════════════════════════════════════════════════════════════════════

/// Filter predicate for querying events.
#[derive(Debug, Clone)]
pub struct EventFilter {
    /// Only include these event kinds (empty = all).
    pub kinds: Vec<EventKind>,
    /// Only events after this timestamp.
    pub since: Option<f64>,
    /// Only events before this timestamp.
    pub until: Option<f64>,
    /// Only events involving this app.
    pub app_id: Option<u16>,
    /// Only preference-signal events.
    pub preference_signals_only: bool,
    /// Only user-activity events.
    pub user_activity_only: bool,
}

impl Default for EventFilter {
    fn default() -> Self {
        Self {
            kinds: Vec::new(),
            since: None,
            until: None,
            app_id: None,
            preference_signals_only: false,
            user_activity_only: false,
        }
    }
}

impl EventFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kind(mut self, kind: EventKind) -> Self {
        self.kinds.push(kind);
        self
    }

    pub fn since(mut self, ts: f64) -> Self {
        self.since = Some(ts);
        self
    }

    pub fn until(mut self, ts: f64) -> Self {
        self.until = Some(ts);
        self
    }

    pub fn app(mut self, app_id: u16) -> Self {
        self.app_id = Some(app_id);
        self
    }

    pub fn preferences_only(mut self) -> Self {
        self.preference_signals_only = true;
        self
    }

    pub fn user_activity(mut self) -> Self {
        self.user_activity_only = true;
        self
    }

    /// Test whether an event matches this filter.
    pub fn matches(&self, event: &SystemEvent) -> bool {
        let kind = event.kind();

        if !self.kinds.is_empty() && !self.kinds.contains(&kind) {
            return false;
        }
        if let Some(since) = self.since {
            if event.timestamp < since {
                return false;
            }
        }
        if let Some(until) = self.until {
            if event.timestamp > until {
                return false;
            }
        }
        if let Some(app_id) = self.app_id {
            if event.app_id() != Some(app_id) {
                return false;
            }
        }
        if self.preference_signals_only && !kind.is_preference_signal() {
            return false;
        }
        if self.user_activity_only && !kind.is_user_activity() {
            return false;
        }

        true
    }
}

/// Query recent events from the buffer with filtering.
pub fn query_events<'a>(
    buffer: &'a EventBuffer,
    filter: &EventFilter,
    limit: usize,
) -> Vec<&'a SystemEvent> {
    buffer.filter(limit, |e| filter.matches(e))
}

/// Compute event rate for a specific kind over a custom window.
pub fn event_rate(counters: &EventCounters, kind: EventKind, now: f64) -> f64 {
    counters.rate(kind, now)
}

/// Get a summary of observer activity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObserverSummary {
    /// Total events ever observed.
    pub total_events: u64,
    /// Events currently in the buffer.
    pub buffer_size: usize,
    /// Events dropped by rate limiting.
    pub total_dropped: u64,
    /// Per-kind totals (sorted descending).
    pub kind_totals: Vec<(EventKind, u64)>,
    /// Current aggregate event rate (events/sec).
    pub current_rate: f64,
    /// Most active event kind.
    pub most_active_kind: Option<EventKind>,
    /// Session duration in seconds.
    pub session_duration_secs: f64,
    /// Events pending persistence.
    pub pending_batch_count: usize,
}

/// Generate an observer summary.
pub fn summarize(buffer: &EventBuffer, state: &ObserverState, now: f64) -> ObserverSummary {
    let kind_totals = state.counters.all_totals();
    let most_active_kind = kind_totals.first().map(|&(k, _)| k);

    ObserverSummary {
        total_events: buffer.total_ingested,
        buffer_size: buffer.len(),
        total_dropped: buffer.total_dropped,
        kind_totals,
        current_rate: state.counters.total_rate(now),
        most_active_kind,
        session_duration_secs: if state.session_start > 0.0 {
            now - state.session_start
        } else {
            0.0
        },
        pending_batch_count: state.pending_batch_count,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 10  Derived Signals (Computed from Observations)
// ══════════════════════════════════════════════════════════════════════════════

/// Signals derived from recent observations — used by higher CK layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedSignals {
    /// Suggestion acceptance rate over recent window.
    pub suggestion_acceptance_rate: f64,
    /// Suggestion rejection rate over recent window.
    pub suggestion_rejection_rate: f64,
    /// Average notification dismiss time (ms).
    pub avg_notification_dismiss_ms: f64,
    /// Fraction of notifications dismissed in <500ms (annoyance signal).
    pub annoyance_rate: f64,
    /// Number of repeated queries (unresolved needs).
    pub unresolved_query_count: u32,
    /// Average app session duration (ms).
    pub avg_app_session_ms: f64,
    /// User idle fraction (idle_time / total_time in window).
    pub idle_fraction: f64,
    /// LLM call count in window.
    pub llm_calls_in_window: u32,
    /// LLM total tokens in window.
    pub llm_tokens_in_window: u32,
    /// Tool call success rate in window.
    pub tool_success_rate: f64,
    /// Error rate (errors/minute) in window.
    pub error_rate_per_min: f64,
}

/// Compute derived signals from the event buffer.
pub fn compute_derived_signals(buffer: &EventBuffer, since: f64, now: f64) -> DerivedSignals {
    let window = buffer.since(since);

    let mut suggestion_accepted = 0u32;
    let mut suggestion_total = 0u32;
    let mut suggestion_rejected = 0u32;
    let mut dismiss_times: Vec<u64> = Vec::new();
    let mut annoyance_count = 0u32;
    let mut notification_count = 0u32;
    let mut query_hashes: HashMap<u64, u32> = HashMap::new();
    let mut app_durations: Vec<u64> = Vec::new();
    let mut idle_total_ms = 0u64;
    let mut llm_calls = 0u32;
    let mut llm_tokens = 0u32;
    let mut tool_success = 0u32;
    let mut tool_total = 0u32;
    let mut error_count = 0u32;

    for event in &window {
        match &event.data {
            SystemEventData::SuggestionAccepted { .. } => {
                suggestion_accepted += 1;
                suggestion_total += 1;
            }
            SystemEventData::SuggestionRejected { .. } => {
                suggestion_rejected += 1;
                suggestion_total += 1;
            }
            SystemEventData::SuggestionIgnored { .. } => {
                suggestion_total += 1;
            }
            SystemEventData::SuggestionModified { .. } => {
                // Modified counts as partial acceptance
                suggestion_accepted += 1;
                suggestion_total += 1;
            }
            SystemEventData::NotificationDismissed {
                time_to_dismiss_ms, ..
            } => {
                dismiss_times.push(*time_to_dismiss_ms);
                notification_count += 1;
                if *time_to_dismiss_ms < 500 {
                    annoyance_count += 1;
                }
            }
            SystemEventData::NotificationActedOn { .. } => {
                notification_count += 1;
            }
            SystemEventData::QueryRepeated {
                query_hash, count, ..
            } => {
                *query_hashes.entry(*query_hash).or_insert(0) += count;
            }
            SystemEventData::AppClosed { duration_ms, .. } => {
                app_durations.push(*duration_ms);
            }
            SystemEventData::UserIdle { duration_ms, .. } => {
                idle_total_ms += duration_ms;
            }
            SystemEventData::LlmCalled { tokens_used, .. } => {
                llm_calls += 1;
                llm_tokens += tokens_used;
            }
            SystemEventData::ToolCallCompleted { success, .. } => {
                tool_total += 1;
                if *success {
                    tool_success += 1;
                }
            }
            SystemEventData::ErrorOccurred { .. } => {
                error_count += 1;
            }
            _ => {}
        }
    }

    let window_secs = (now - since).max(1.0);

    DerivedSignals {
        suggestion_acceptance_rate: if suggestion_total > 0 {
            suggestion_accepted as f64 / suggestion_total as f64
        } else {
            0.5 // uninformative prior
        },
        suggestion_rejection_rate: if suggestion_total > 0 {
            suggestion_rejected as f64 / suggestion_total as f64
        } else {
            0.0
        },
        avg_notification_dismiss_ms: if dismiss_times.is_empty() {
            0.0
        } else {
            dismiss_times.iter().sum::<u64>() as f64 / dismiss_times.len() as f64
        },
        annoyance_rate: if notification_count > 0 {
            annoyance_count as f64 / notification_count as f64
        } else {
            0.0
        },
        unresolved_query_count: query_hashes.values().filter(|&&c| c >= 3).count() as u32,
        avg_app_session_ms: if app_durations.is_empty() {
            0.0
        } else {
            app_durations.iter().sum::<u64>() as f64 / app_durations.len() as f64
        },
        idle_fraction: (idle_total_ms as f64 / 1000.0) / window_secs,
        llm_calls_in_window: llm_calls,
        llm_tokens_in_window: llm_tokens,
        tool_success_rate: if tool_total > 0 {
            tool_success as f64 / tool_total as f64
        } else {
            1.0 // uninformative prior
        },
        error_rate_per_min: error_count as f64 / (window_secs / 60.0),
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 11  App Sequence Detection
// ══════════════════════════════════════════════════════════════════════════════

/// Detect app transition sequences from a series of AppOpened events.
///
/// Emits AppSequence events for consecutive app opens within `max_gap_ms`.
pub fn detect_app_sequences(buffer: &EventBuffer, since: f64, max_gap_ms: u64) -> Vec<SystemEvent> {
    let opens: Vec<&SystemEvent> = buffer.filter(usize::MAX, |e| {
        e.timestamp >= since && matches!(e.data, SystemEventData::AppOpened { .. })
    });

    // Opens are newest-first, reverse for chronological order
    let mut chronological: Vec<_> = opens.into_iter().collect();
    chronological.reverse();

    let mut sequences = Vec::new();

    for pair in chronological.windows(2) {
        let (prev, curr) = (pair[0], pair[1]);
        if let (
            SystemEventData::AppOpened { app_id: from },
            SystemEventData::AppOpened { app_id: to },
        ) = (&prev.data, &curr.data)
        {
            let gap_ms = ((curr.timestamp - prev.timestamp) * 1000.0) as u64;
            if gap_ms <= max_gap_ms && from != to {
                sequences.push(SystemEvent::new(
                    curr.timestamp,
                    SystemEventData::AppSequence {
                        from_app: *from,
                        to_app: *to,
                        gap_ms,
                    },
                ));
            }
        }
    }

    sequences
}

// ══════════════════════════════════════════════════════════════════════════════
// § 12  Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(offset: f64) -> f64 {
        86400.0 * 100.0 + offset // Day 100 + offset seconds
    }

    fn make_app_open(app_id: u16, offset: f64) -> SystemEvent {
        SystemEvent::new(ts(offset), SystemEventData::AppOpened { app_id })
    }

    fn make_app_close(app_id: u16, duration_ms: u64, offset: f64) -> SystemEvent {
        SystemEvent::new(
            ts(offset),
            SystemEventData::AppClosed {
                app_id,
                duration_ms,
            },
        )
    }

    fn make_suggestion_accepted(offset: f64) -> SystemEvent {
        SystemEvent::new(
            ts(offset),
            SystemEventData::SuggestionAccepted {
                suggestion_id: 1,
                action_kind: "test".to_string(),
                latency_ms: 500,
            },
        )
    }

    fn make_suggestion_rejected(offset: f64) -> SystemEvent {
        SystemEvent::new(
            ts(offset),
            SystemEventData::SuggestionRejected {
                suggestion_id: 2,
                action_kind: "test".to_string(),
            },
        )
    }

    // ── Event Kind ──

    #[test]
    fn test_event_kind_discriminator() {
        let event = make_app_open(5, 0.0);
        assert_eq!(event.kind(), EventKind::AppOpened);
        assert_eq!(event.app_id(), Some(5));
    }

    #[test]
    fn test_event_kind_classification() {
        assert!(EventKind::SuggestionAccepted.is_preference_signal());
        assert!(!EventKind::AppOpened.is_preference_signal());
        assert!(EventKind::AppOpened.is_user_activity());
        assert!(!EventKind::LlmCalled.is_user_activity());
    }

    #[test]
    fn test_all_kinds_enumerated() {
        assert_eq!(EventKind::ALL.len(), 16);
    }

    // ── Event Buffer ──

    #[test]
    fn test_buffer_push_and_query() {
        let mut buf = EventBuffer::new(100);
        buf.push(make_app_open(1, 0.0));
        buf.push(make_app_open(2, 1.0));
        buf.push(make_app_open(3, 2.0));

        assert_eq!(buf.len(), 3);
        assert_eq!(buf.total_ingested, 3);

        let recent = buf.recent(2);
        assert_eq!(recent.len(), 2);
        // Newest first
        assert_eq!(recent[0].app_id(), Some(3));
        assert_eq!(recent[1].app_id(), Some(2));
    }

    #[test]
    fn test_buffer_capacity_eviction() {
        let mut buf = EventBuffer::new(3);
        for i in 0..5 {
            buf.push(make_app_open(i, i as f64));
        }
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.total_ingested, 5);
        // Oldest (0, 1) evicted — remaining: 2, 3, 4
        assert_eq!(buf.recent(3)[2].app_id(), Some(2));
    }

    #[test]
    fn test_buffer_filter_by_kind() {
        let mut buf = EventBuffer::new(100);
        buf.push(make_app_open(1, 0.0));
        buf.push(make_suggestion_accepted(1.0));
        buf.push(make_app_open(2, 2.0));

        let opens = buf.by_kind(EventKind::AppOpened, 10);
        assert_eq!(opens.len(), 2);

        let suggestions = buf.by_kind(EventKind::SuggestionAccepted, 10);
        assert_eq!(suggestions.len(), 1);
    }

    #[test]
    fn test_buffer_time_window() {
        let mut buf = EventBuffer::new(100);
        buf.push(make_app_open(1, 0.0));
        buf.push(make_app_open(2, 10.0));
        buf.push(make_app_open(3, 20.0));

        let window = buf.in_window(ts(5.0), ts(15.0));
        assert_eq!(window.len(), 1);
        assert_eq!(window[0].app_id(), Some(2));
    }

    #[test]
    fn test_buffer_prune() {
        let mut buf = EventBuffer::new(100);
        buf.push(make_app_open(1, 0.0));
        buf.push(make_app_open(2, 10.0));
        buf.push(make_app_open(3, 20.0));

        let pruned = buf.prune_before(ts(15.0));
        assert_eq!(pruned, 2);
        assert_eq!(buf.len(), 1);
    }

    // ── Event Counters ──

    #[test]
    fn test_counters_total_and_rate() {
        let mut counters = EventCounters::new(60.0);

        for i in 0..10 {
            counters.record(EventKind::AppOpened, ts(i as f64));
        }
        assert_eq!(counters.total(EventKind::AppOpened), 10);

        let rate = counters.rate(EventKind::AppOpened, ts(10.0));
        assert!((rate - 10.0 / 60.0).abs() < 0.01);
    }

    #[test]
    fn test_counters_window_eviction() {
        let mut counters = EventCounters::new(10.0);

        counters.record(EventKind::AppOpened, ts(0.0));
        counters.record(EventKind::AppOpened, ts(5.0));
        counters.record(EventKind::AppOpened, ts(15.0)); // evicts ts(0.0)

        assert_eq!(counters.total(EventKind::AppOpened), 3); // all-time
        let rate = counters.rate(EventKind::AppOpened, ts(15.0));
        // Only 2 events in window [5.0, 15.0]
        assert!((rate - 2.0 / 10.0).abs() < 0.01);
    }

    // ── Circadian Histogram ──

    #[test]
    fn test_histogram_distribution() {
        let mut hist = CircadianHistogram::new();

        // 10am events (hour 10)
        for i in 0..5 {
            hist.record(EventKind::AppOpened, 10.0 * 3600.0 + i as f64);
        }
        // 3pm events (hour 15)
        for i in 0..3 {
            hist.record(EventKind::AppOpened, 15.0 * 3600.0 + i as f64);
        }

        let dist = hist.distribution(EventKind::AppOpened);
        assert_eq!(dist[10], 5);
        assert_eq!(dist[15], 3);
        assert_eq!(hist.peak_hour(EventKind::AppOpened), Some(10));
        assert_eq!(hist.total(EventKind::AppOpened), 8);
    }

    #[test]
    fn test_histogram_normalized() {
        let mut hist = CircadianHistogram::new();
        // All events at hour 12 (noon)
        for _ in 0..24 {
            hist.record(EventKind::AppOpened, 12.0 * 3600.0);
        }

        let norm = hist.normalized_distribution(EventKind::AppOpened);
        assert!(norm[12] > 20.0); // 24/1.0 = 24× mean
        assert_eq!(norm[0], 0.0); // nothing at midnight
    }

    // ── Observe Function ──

    #[test]
    fn test_observe_event_accepted() {
        let mut buffer = EventBuffer::new(100);
        let mut state = ObserverState::new();

        let event = make_app_open(1, 0.0);
        let accepted = observe_event(event, &mut buffer, &mut state);

        assert!(accepted);
        assert_eq!(buffer.len(), 1);
        assert_eq!(state.counters.total(EventKind::AppOpened), 1);
        assert_eq!(state.pending_batch_count, 1);
    }

    #[test]
    fn test_observe_disabled_kind() {
        let mut buffer = EventBuffer::new(100);
        let mut config = ObserverConfig::default();
        config.disabled_kinds.push(EventKind::UserTyping);
        let mut state = ObserverState::with_config(config);

        let event = SystemEvent::new(
            ts(0.0),
            SystemEventData::UserTyping {
                app_id: 1,
                duration_ms: 1000,
                characters: 50,
            },
        );
        let accepted = observe_event(event, &mut buffer, &mut state);

        assert!(!accepted);
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_observe_rate_limiting() {
        let mut buffer = EventBuffer::new(10000);
        let mut config = ObserverConfig::default();
        config.max_events_per_sec = 2.0; // Very low for testing
        config.rate_window_secs = 1.0;
        let mut state = ObserverState::with_config(config);

        // Flood with events at same timestamp
        let mut accepted_count = 0;
        for i in 0..10 {
            let event = make_app_open(i, 0.0);
            if observe_event(event, &mut buffer, &mut state) {
                accepted_count += 1;
            }
        }

        // Some should be rate-limited
        assert!(accepted_count < 10);
        assert!(buffer.total_dropped > 0);
    }

    // ── Flush Logic ──

    #[test]
    fn test_needs_flush_by_count() {
        let mut state = ObserverState::new();
        state.config.batch_persist_threshold = 5;
        state.pending_batch_count = 5;

        assert!(needs_flush(&state, ts(0.0)));
    }

    #[test]
    fn test_needs_flush_by_time() {
        let mut state = ObserverState::new();
        state.config.batch_persist_interval_secs = 30.0;
        state.pending_batch_count = 1;
        state.last_flush_at = ts(0.0);

        assert!(!needs_flush(&state, ts(10.0))); // too soon
        assert!(needs_flush(&state, ts(31.0))); // time elapsed
    }

    // ── Event Filter ──

    #[test]
    fn test_filter_by_kind() {
        let mut buffer = EventBuffer::new(100);
        buffer.push(make_app_open(1, 0.0));
        buffer.push(make_suggestion_accepted(1.0));
        buffer.push(make_app_open(2, 2.0));

        let filter = EventFilter::new().kind(EventKind::SuggestionAccepted);
        let results = query_events(&buffer, &filter, 10);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_filter_by_app_id() {
        let mut buffer = EventBuffer::new(100);
        buffer.push(make_app_open(1, 0.0));
        buffer.push(make_app_open(2, 1.0));
        buffer.push(make_app_close(1, 5000, 2.0));

        let filter = EventFilter::new().app(1);
        let results = query_events(&buffer, &filter, 10);
        assert_eq!(results.len(), 2); // open + close for app 1
    }

    #[test]
    fn test_filter_preference_signals() {
        let mut buffer = EventBuffer::new(100);
        buffer.push(make_app_open(1, 0.0));
        buffer.push(make_suggestion_accepted(1.0));
        buffer.push(make_suggestion_rejected(2.0));

        let filter = EventFilter::new().preferences_only();
        let results = query_events(&buffer, &filter, 10);
        assert_eq!(results.len(), 2); // accepted + rejected
    }

    // ── Derived Signals ──

    #[test]
    fn test_derived_signals_acceptance_rate() {
        let mut buffer = EventBuffer::new(100);
        buffer.push(make_suggestion_accepted(0.0));
        buffer.push(make_suggestion_accepted(1.0));
        buffer.push(make_suggestion_rejected(2.0));

        let signals = compute_derived_signals(&buffer, ts(-1.0), ts(3.0));
        assert!((signals.suggestion_acceptance_rate - 2.0 / 3.0).abs() < 0.01);
        assert!((signals.suggestion_rejection_rate - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_derived_signals_annoyance() {
        let mut buffer = EventBuffer::new(100);
        // Fast dismiss = annoyance
        buffer.push(SystemEvent::new(
            ts(0.0),
            SystemEventData::NotificationDismissed {
                notif_type: "alert".to_string(),
                time_to_dismiss_ms: 200,
            },
        ));
        // Slow dismiss = intentional
        buffer.push(SystemEvent::new(
            ts(1.0),
            SystemEventData::NotificationDismissed {
                notif_type: "info".to_string(),
                time_to_dismiss_ms: 3000,
            },
        ));

        let signals = compute_derived_signals(&buffer, ts(-1.0), ts(2.0));
        assert!((signals.annoyance_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_derived_signals_idle_fraction() {
        let mut buffer = EventBuffer::new(100);
        // 30s idle out of 60s window
        buffer.push(SystemEvent::new(
            ts(10.0),
            SystemEventData::UserIdle { duration_ms: 30000 },
        ));

        let signals = compute_derived_signals(&buffer, ts(0.0), ts(60.0));
        assert!((signals.idle_fraction - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_derived_signals_tool_success() {
        let mut buffer = EventBuffer::new(100);
        buffer.push(SystemEvent::new(
            ts(0.0),
            SystemEventData::ToolCallCompleted {
                tool_name: "search".to_string(),
                success: true,
                duration_ms: 100,
            },
        ));
        buffer.push(SystemEvent::new(
            ts(1.0),
            SystemEventData::ToolCallCompleted {
                tool_name: "execute".to_string(),
                success: false,
                duration_ms: 200,
            },
        ));

        let signals = compute_derived_signals(&buffer, ts(-1.0), ts(2.0));
        assert!((signals.tool_success_rate - 0.5).abs() < 0.01);
    }

    // ── App Sequence Detection ──

    #[test]
    fn test_app_sequence_detection() {
        let mut buffer = EventBuffer::new(100);
        buffer.push(make_app_open(1, 0.0));
        buffer.push(make_app_open(2, 2.0)); // 2s gap
        buffer.push(make_app_open(3, 5.0)); // 3s gap

        let sequences = detect_app_sequences(&buffer, ts(-1.0), 10_000);
        assert_eq!(sequences.len(), 2);

        match &sequences[0].data {
            SystemEventData::AppSequence {
                from_app, to_app, ..
            } => {
                assert_eq!(*from_app, 1);
                assert_eq!(*to_app, 2);
            }
            _ => panic!("expected AppSequence"),
        }
    }

    #[test]
    fn test_app_sequence_gap_filter() {
        let mut buffer = EventBuffer::new(100);
        buffer.push(make_app_open(1, 0.0));
        buffer.push(make_app_open(2, 120.0)); // 120s gap — too long

        let sequences = detect_app_sequences(&buffer, ts(-1.0), 60_000); // 60s max
        assert_eq!(sequences.len(), 0);
    }

    #[test]
    fn test_app_sequence_same_app() {
        let mut buffer = EventBuffer::new(100);
        buffer.push(make_app_open(1, 0.0));
        buffer.push(make_app_open(1, 2.0)); // same app — no transition

        let sequences = detect_app_sequences(&buffer, ts(-1.0), 10_000);
        assert_eq!(sequences.len(), 0);
    }

    // ── Observer Summary ──

    #[test]
    fn test_observer_summary() {
        let mut buffer = EventBuffer::new(100);
        let mut state = ObserverState::new();
        state.session_start = ts(0.0);

        for i in 0..5 {
            observe_event(make_app_open(i, i as f64), &mut buffer, &mut state);
        }

        let summary = summarize(&buffer, &state, ts(10.0));
        assert_eq!(summary.total_events, 5);
        assert_eq!(summary.buffer_size, 5);
        assert_eq!(summary.most_active_kind, Some(EventKind::AppOpened));
        assert!((summary.session_duration_secs - 10.0).abs() < 0.01);
    }

    // ── Annoyance Signal ──

    #[test]
    fn test_annoyance_signal() {
        let fast = SystemEvent::new(
            ts(0.0),
            SystemEventData::NotificationDismissed {
                notif_type: "x".to_string(),
                time_to_dismiss_ms: 200,
            },
        );
        let slow = SystemEvent::new(
            ts(0.0),
            SystemEventData::NotificationDismissed {
                notif_type: "x".to_string(),
                time_to_dismiss_ms: 3000,
            },
        );
        assert!(fast.is_annoyance_signal());
        assert!(!slow.is_annoyance_signal());
    }
}
