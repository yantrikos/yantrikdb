//! CK-2.2: Temporal Reasoning Primitives
//!
//! Provides the temporal intelligence layer for YantrikDB's cognitive engine.
//! This module is purely functional — no database dependency — enabling
//! deterministic testing and composition.
//!
//! ## Capabilities
//!
//! 1. **Interval Algebra** — Allen's interval relations (before, after, during,
//!    overlaps, meets, starts, finishes, equals) for reasoning about event
//!    relationships and scheduling conflicts.
//!
//! 2. **Recency-Weighted Relevance** — Exponential decay functions that score
//!    how relevant a past event is to the current moment, with configurable
//!    half-life per domain.
//!
//! 3. **Periodicity Detection** — Infer daily/weekly/monthly patterns from
//!    sparse event timestamps using autocorrelation and peak detection.
//!
//! 4. **EWMA (Exponentially Weighted Moving Average)** — Smoothed tracking of
//!    continuous signals (response times, activity levels, mood) with
//!    configurable smoothing factor.
//!
//! 5. **Burst Detection** — Identifies sudden increases in event frequency
//!    using a sliding window model with z-score threshold.
//!
//! 6. **Temporal Motif Mining** — Discovers recurring sequences of event types
//!    (e.g., "email check → calendar review → standup") from episode streams.
//!
//! 7. **Deadline Urgency** — Sigmoid-based urgency curves for deadline-driven
//!    nodes, computing how urgently attention should be directed.
//!
//! 8. **Event Ordering** — Causal/temporal sequencing using topological sort
//!    over PrecedesTemporally and Triggers edges.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ══════════════════════════════════════════════════════════════════════════════
// § 1  Interval Algebra
// ══════════════════════════════════════════════════════════════════════════════

/// A closed time interval `[start, end]` in seconds (unix timestamp).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeInterval {
    pub start: f64,
    pub end: f64,
}

impl TimeInterval {
    /// Create a new interval. Panics if `start > end`.
    pub fn new(start: f64, end: f64) -> Self {
        debug_assert!(start <= end, "TimeInterval: start ({start}) > end ({end})");
        Self { start, end }
    }

    /// Point interval (instantaneous event).
    pub fn point(t: f64) -> Self {
        Self { start: t, end: t }
    }

    /// Duration of the interval in seconds.
    #[inline]
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }

    /// Midpoint of the interval.
    #[inline]
    pub fn midpoint(&self) -> f64 {
        (self.start + self.end) / 2.0
    }

    /// Whether this interval contains a point in time.
    #[inline]
    pub fn contains_point(&self, t: f64) -> bool {
        t >= self.start && t <= self.end
    }

    /// Whether this interval fully contains another interval.
    #[inline]
    pub fn contains_interval(&self, other: &TimeInterval) -> bool {
        self.start <= other.start && self.end >= other.end
    }

    /// Classify the relationship between two intervals using Allen's algebra.
    pub fn relation_to(&self, other: &TimeInterval) -> IntervalRelation {
        // Tolerance for "meets" — intervals within ε of touching are considered meeting
        const EPSILON: f64 = 0.5;

        if (self.end - other.start).abs() < EPSILON && self.start < other.start {
            return IntervalRelation::Meets;
        }
        if (other.end - self.start).abs() < EPSILON && other.start < self.start {
            return IntervalRelation::MetBy;
        }

        if self.end < other.start {
            IntervalRelation::Before
        } else if other.end < self.start {
            IntervalRelation::After
        } else if self.start == other.start && self.end == other.end {
            IntervalRelation::Equals
        } else if self.start == other.start && self.end < other.end {
            IntervalRelation::Starts
        } else if self.start == other.start && self.end > other.end {
            IntervalRelation::StartedBy
        } else if self.end == other.end && self.start > other.start {
            IntervalRelation::Finishes
        } else if self.end == other.end && self.start < other.start {
            IntervalRelation::FinishedBy
        } else if self.start > other.start && self.end < other.end {
            IntervalRelation::During
        } else if other.start > self.start && other.end < self.end {
            IntervalRelation::Contains
        } else if self.start < other.start && self.end > other.start && self.end < other.end {
            IntervalRelation::Overlaps
        } else if other.start < self.start && other.end > self.start && other.end < self.end {
            IntervalRelation::OverlappedBy
        } else {
            // Fallback — should not reach here with proper interval semantics
            IntervalRelation::Overlaps
        }
    }

    /// Compute the overlap duration with another interval (0 if no overlap).
    pub fn overlap_duration(&self, other: &TimeInterval) -> f64 {
        let overlap_start = self.start.max(other.start);
        let overlap_end = self.end.min(other.end);
        (overlap_end - overlap_start).max(0.0)
    }

    /// Merge two overlapping or adjacent intervals into one.
    /// Returns `None` if they are disjoint (gap > tolerance).
    pub fn merge(&self, other: &TimeInterval, tolerance: f64) -> Option<TimeInterval> {
        if self.start <= other.end + tolerance && other.start <= self.end + tolerance {
            Some(TimeInterval::new(
                self.start.min(other.start),
                self.end.max(other.end),
            ))
        } else {
            None
        }
    }
}

/// Allen's 13 interval relations — the complete set for temporal reasoning.
///
/// These relations are mutually exclusive and exhaustive: any two intervals
/// stand in exactly one of these 13 relations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntervalRelation {
    /// Self ends before other starts: `[s1,e1] < [s2,e2]`
    Before,
    /// Self starts after other ends: `[s2,e2] < [s1,e1]`
    After,
    /// Self meets other: `e1 ≈ s2` (within epsilon)
    Meets,
    /// Self is met by other: `e2 ≈ s1`
    MetBy,
    /// Self overlaps other: `s1 < s2 < e1 < e2`
    Overlaps,
    /// Self is overlapped by other: `s2 < s1 < e2 < e1`
    OverlappedBy,
    /// Self starts other: same start, self ends first
    Starts,
    /// Self is started by other: same start, other ends first
    StartedBy,
    /// Self finishes other: same end, self starts later
    Finishes,
    /// Self is finished by other: same end, other starts later
    FinishedBy,
    /// Self during other: fully contained within
    During,
    /// Self contains other: fully contains
    Contains,
    /// Identical intervals
    Equals,
}

impl IntervalRelation {
    /// Whether the two intervals have any temporal overlap.
    pub fn has_overlap(self) -> bool {
        !matches!(self, Self::Before | Self::After | Self::Meets | Self::MetBy)
    }

    /// Inverse relation (swap perspectives).
    pub fn inverse(self) -> Self {
        match self {
            Self::Before => Self::After,
            Self::After => Self::Before,
            Self::Meets => Self::MetBy,
            Self::MetBy => Self::Meets,
            Self::Overlaps => Self::OverlappedBy,
            Self::OverlappedBy => Self::Overlaps,
            Self::Starts => Self::StartedBy,
            Self::StartedBy => Self::Starts,
            Self::Finishes => Self::FinishedBy,
            Self::FinishedBy => Self::Finishes,
            Self::During => Self::Contains,
            Self::Contains => Self::During,
            Self::Equals => Self::Equals,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 2  Recency-Weighted Relevance
// ══════════════════════════════════════════════════════════════════════════════

/// Configuration for recency-weighted relevance scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecencyConfig {
    /// Half-life in seconds — the time for relevance to decay to 50%.
    /// Default: 86400.0 (1 day).
    pub half_life_secs: f64,
    /// Minimum relevance floor — events never decay below this.
    /// Default: 0.01.
    pub floor: f64,
}

impl Default for RecencyConfig {
    fn default() -> Self {
        Self {
            half_life_secs: 86400.0,
            floor: 0.01,
        }
    }
}

/// Domain-specific half-lives for different types of information.
/// Some things stay relevant longer than others.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainRecencyMap {
    /// Domain name → half-life in seconds.
    pub entries: HashMap<String, f64>,
    /// Fallback half-life for unknown domains.
    pub default_half_life: f64,
}

impl Default for DomainRecencyMap {
    fn default() -> Self {
        let mut entries = HashMap::new();
        // Conversations decay fast — yesterday's chat is less relevant
        entries.insert("conversation".to_string(), 3600.0 * 4.0);
        // Tasks have medium persistence
        entries.insert("task".to_string(), 86400.0 * 3.0);
        // Goals persist much longer
        entries.insert("goal".to_string(), 86400.0 * 30.0);
        // Beliefs are very persistent
        entries.insert("belief".to_string(), 86400.0 * 90.0);
        // Routines have long memory (need history to detect patterns)
        entries.insert("routine".to_string(), 86400.0 * 60.0);
        // Preferences are quasi-permanent
        entries.insert("preference".to_string(), 86400.0 * 180.0);
        // Episodes (events) decay at medium rate
        entries.insert("episode".to_string(), 86400.0 * 7.0);

        Self {
            entries,
            default_half_life: 86400.0,
        }
    }
}

impl DomainRecencyMap {
    /// Get the half-life for a domain.
    pub fn half_life_for(&self, domain: &str) -> f64 {
        self.entries
            .get(domain)
            .copied()
            .unwrap_or(self.default_half_life)
    }
}

/// Compute recency-weighted relevance score for an event.
///
/// Uses exponential decay: `relevance = max(floor, 2^(-age / half_life))`.
///
/// # Arguments
/// * `event_time` — When the event occurred (unix seconds).
/// * `now` — Current time (unix seconds).
/// * `config` — Recency configuration.
///
/// # Returns
/// Relevance in `[floor, 1.0]` — 1.0 means "just happened", floor for very old.
pub fn recency_relevance(event_time: f64, now: f64, config: &RecencyConfig) -> f64 {
    let age = (now - event_time).max(0.0);
    let decay = f64::powf(2.0, -age / config.half_life_secs);
    decay.max(config.floor)
}

/// Compute recency-weighted relevance with domain-specific half-life.
pub fn recency_relevance_domain(
    event_time: f64,
    now: f64,
    domain: &str,
    domain_map: &DomainRecencyMap,
) -> f64 {
    let half_life = domain_map.half_life_for(domain);
    let config = RecencyConfig {
        half_life_secs: half_life,
        floor: 0.01,
    };
    recency_relevance(event_time, now, &config)
}

/// Score a batch of events by recency and return them sorted (most relevant first).
///
/// Each event is `(event_id, event_time)`. Returns `(event_id, relevance_score)`.
pub fn rank_by_recency(events: &[(u64, f64)], now: f64, config: &RecencyConfig) -> Vec<(u64, f64)> {
    let mut scored: Vec<(u64, f64)> = events
        .iter()
        .map(|&(id, time)| (id, recency_relevance(time, now, config)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  Periodicity Detection
// ══════════════════════════════════════════════════════════════════════════════

/// Configuration for periodicity detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodicityConfig {
    /// Candidate periods to test (in seconds).
    /// Default: common human-scale periods.
    pub candidate_periods: Vec<f64>,
    /// Minimum number of events required for reliable detection.
    /// Default: 5.
    pub min_events: usize,
    /// Autocorrelation threshold for a period to be considered detected.
    /// Default: 0.3 (moderate correlation).
    pub correlation_threshold: f64,
    /// Tolerance window as a fraction of the period.
    /// Events within `period * tolerance` of a predicted time count as hits.
    /// Default: 0.15 (15% of period).
    pub tolerance_fraction: f64,
}

impl Default for PeriodicityConfig {
    fn default() -> Self {
        Self {
            candidate_periods: vec![
                3600.0,         // hourly
                3600.0 * 4.0,   // 4-hourly
                86400.0,        // daily
                86400.0 * 7.0,  // weekly
                86400.0 * 14.0, // biweekly
                86400.0 * 30.0, // monthly
                86400.0 * 90.0, // quarterly
            ],
            min_events: 5,
            correlation_threshold: 0.3,
            tolerance_fraction: 0.15,
        }
    }
}

/// Result of periodicity detection for a single candidate period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPeriod {
    /// The period in seconds.
    pub period_secs: f64,
    /// Autocorrelation score [0.0, 1.0].
    pub correlation: f64,
    /// Phase offset: seconds into the period when events cluster.
    pub phase_offset_secs: f64,
    /// Fraction of expected events that actually occurred.
    pub hit_rate: f64,
    /// Human-readable label (e.g., "daily", "weekly").
    pub label: String,
}

/// Full periodicity analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodicityResult {
    /// All detected periods (above threshold), sorted by correlation descending.
    pub detected: Vec<DetectedPeriod>,
    /// Number of events analyzed.
    pub event_count: usize,
    /// Time span of the data (seconds from first to last event).
    pub span_secs: f64,
}

/// Detect periodicities in a set of event timestamps.
///
/// Uses a correlation-based approach:
/// 1. For each candidate period `T`, compute phase offsets `t_i mod T`
/// 2. Compute circular concentration (Rayleigh test analog) of phase offsets
/// 3. Periods with high concentration indicate periodicity
///
/// # Arguments
/// * `timestamps` — Unix timestamps of events (need not be sorted).
/// * `config` — Periodicity detection parameters.
///
/// # Returns
/// `PeriodicityResult` with detected periods sorted by correlation.
pub fn detect_periodicity(timestamps: &[f64], config: &PeriodicityConfig) -> PeriodicityResult {
    if timestamps.len() < config.min_events {
        return PeriodicityResult {
            detected: vec![],
            event_count: timestamps.len(),
            span_secs: 0.0,
        };
    }

    let mut sorted = timestamps.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let span = sorted.last().unwrap() - sorted.first().unwrap();
    if span <= 0.0 {
        return PeriodicityResult {
            detected: vec![],
            event_count: timestamps.len(),
            span_secs: 0.0,
        };
    }

    let n = sorted.len() as f64;
    let mut detected = Vec::new();

    for &period in &config.candidate_periods {
        // Skip if we don't have enough span to observe this period
        if span < period * 1.5 {
            continue;
        }

        // Compute phase offsets normalized to [0, 2π]
        let two_pi = std::f64::consts::TAU;
        let phases: Vec<f64> = sorted
            .iter()
            .map(|&t| ((t % period) / period) * two_pi)
            .collect();

        // Rayleigh test: compute mean resultant length
        // R = sqrt((sum cos θ)² + (sum sin θ)²) / n
        let cos_sum: f64 = phases.iter().map(|&p| p.cos()).sum();
        let sin_sum: f64 = phases.iter().map(|&p| p.sin()).sum();
        let r = (cos_sum * cos_sum + sin_sum * sin_sum).sqrt() / n;

        // Mean phase angle → phase offset in seconds
        let mean_phase = sin_sum.atan2(cos_sum);
        let phase_offset = ((mean_phase / two_pi) * period + period) % period;

        // Hit rate: fraction of events falling within tolerance of predicted times
        let tolerance = period * config.tolerance_fraction;
        let hits = sorted
            .iter()
            .filter(|&&t| {
                let offset = (t - phase_offset) % period;
                let offset = if offset < 0.0 {
                    offset + period
                } else {
                    offset
                };
                offset < tolerance || (period - offset) < tolerance
            })
            .count();
        let hit_rate = hits as f64 / n;

        if r >= config.correlation_threshold {
            detected.push(DetectedPeriod {
                period_secs: period,
                correlation: r,
                phase_offset_secs: phase_offset,
                hit_rate,
                label: period_label(period),
            });
        }
    }

    // Sort by correlation descending
    detected.sort_by(|a, b| {
        b.correlation
            .partial_cmp(&a.correlation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    PeriodicityResult {
        detected,
        event_count: timestamps.len(),
        span_secs: span,
    }
}

/// Human-readable label for a period.
fn period_label(period_secs: f64) -> String {
    let hours = period_secs / 3600.0;
    if hours < 2.0 {
        format!("{:.0}min", period_secs / 60.0)
    } else if hours < 48.0 {
        format!("{:.0}h", hours)
    } else {
        let days = hours / 24.0;
        if (days - 7.0).abs() < 1.0 {
            "weekly".to_string()
        } else if (days - 14.0).abs() < 2.0 {
            "biweekly".to_string()
        } else if (days - 30.0).abs() < 5.0 {
            "monthly".to_string()
        } else if (days - 90.0).abs() < 10.0 {
            "quarterly".to_string()
        } else {
            format!("{:.0}d", days)
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  EWMA — Exponentially Weighted Moving Average
// ══════════════════════════════════════════════════════════════════════════════

/// Exponentially Weighted Moving Average tracker.
///
/// Smooths a stream of observations with configurable responsiveness.
/// α close to 1.0 = highly responsive to new values (noisy).
/// α close to 0.0 = highly smoothed (sluggish).
///
/// Update rule: `ewma = α × new_value + (1 - α) × ewma`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EwmaTracker {
    /// Smoothing factor α ∈ (0, 1].
    pub alpha: f64,
    /// Current smoothed value.
    pub value: f64,
    /// Current smoothed variance (for anomaly detection).
    pub variance: f64,
    /// Number of observations ingested.
    pub count: u64,
    /// Last observation timestamp.
    pub last_time: f64,
}

impl EwmaTracker {
    /// Create a new EWMA tracker with the given smoothing factor.
    ///
    /// # Panics
    /// Panics if `alpha` is not in (0, 1].
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0, 1]");
        Self {
            alpha,
            value: 0.0,
            variance: 0.0,
            count: 0,
            last_time: 0.0,
        }
    }

    /// Create a tracker with initial value (avoids cold-start bias).
    pub fn with_initial(alpha: f64, initial_value: f64) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0, 1]");
        Self {
            alpha,
            value: initial_value,
            variance: 0.0,
            count: 1,
            last_time: 0.0,
        }
    }

    /// Ingest a new observation.
    pub fn update(&mut self, observation: f64, timestamp: f64) {
        if self.count == 0 {
            // First observation — initialize directly
            self.value = observation;
            self.variance = 0.0;
        } else {
            let diff = observation - self.value;
            self.value += self.alpha * diff;
            // Welford-style online variance with exponential weighting
            self.variance = (1.0 - self.alpha) * (self.variance + self.alpha * diff * diff);
        }
        self.count += 1;
        self.last_time = timestamp;
    }

    /// Current smoothed value.
    #[inline]
    pub fn current(&self) -> f64 {
        self.value
    }

    /// Estimated standard deviation.
    #[inline]
    pub fn std_dev(&self) -> f64 {
        self.variance.sqrt()
    }

    /// Z-score of a new observation relative to the current EWMA.
    /// Useful for anomaly detection.
    pub fn z_score(&self, observation: f64) -> f64 {
        let sd = self.std_dev();
        if sd < 1e-10 || self.count < 3 {
            0.0
        } else {
            (observation - self.value) / sd
        }
    }

    /// Whether a new observation would be anomalous (|z| > threshold).
    pub fn is_anomaly(&self, observation: f64, z_threshold: f64) -> bool {
        self.z_score(observation).abs() > z_threshold
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 5  Burst Detection
// ══════════════════════════════════════════════════════════════════════════════

/// Configuration for burst detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurstConfig {
    /// Window size in seconds for computing event rate.
    /// Default: 3600.0 (1 hour).
    pub window_secs: f64,
    /// Z-score threshold for declaring a burst.
    /// Default: 2.5.
    pub z_threshold: f64,
    /// Minimum number of windows required before detection is reliable.
    /// Default: 5.
    pub min_windows: usize,
    /// Smoothing factor for baseline rate EWMA.
    /// Default: 0.2.
    pub ewma_alpha: f64,
}

impl Default for BurstConfig {
    fn default() -> Self {
        Self {
            window_secs: 3600.0,
            z_threshold: 2.5,
            min_windows: 5,
            ewma_alpha: 0.2,
        }
    }
}

/// A detected burst of activity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedBurst {
    /// Start of the burst window (unix seconds).
    pub window_start: f64,
    /// End of the burst window (unix seconds).
    pub window_end: f64,
    /// Event count in the burst window.
    pub event_count: usize,
    /// Z-score of this window's rate vs. baseline.
    pub z_score: f64,
    /// Baseline event rate (events per window).
    pub baseline_rate: f64,
}

/// Burst detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BurstResult {
    /// Detected bursts, sorted chronologically.
    pub bursts: Vec<DetectedBurst>,
    /// Overall event rate (events per window).
    pub overall_rate: f64,
    /// Total windows analyzed.
    pub windows_analyzed: usize,
}

/// Detect bursts in a stream of event timestamps.
///
/// Divides the time range into fixed windows, computes event counts per window,
/// then uses EWMA to establish a baseline rate and flags windows that exceed
/// the baseline by more than `z_threshold` standard deviations.
///
/// # Arguments
/// * `timestamps` — Unix timestamps of events (need not be sorted).
/// * `config` — Burst detection parameters.
pub fn detect_bursts(timestamps: &[f64], config: &BurstConfig) -> BurstResult {
    if timestamps.is_empty() {
        return BurstResult {
            bursts: vec![],
            overall_rate: 0.0,
            windows_analyzed: 0,
        };
    }

    let mut sorted = timestamps.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let first = sorted[0];
    let last = sorted[sorted.len() - 1];
    let span = last - first;

    if span < config.window_secs {
        return BurstResult {
            bursts: vec![],
            overall_rate: sorted.len() as f64,
            windows_analyzed: 1,
        };
    }

    // Divide into windows and count events per window
    let num_windows = ((span / config.window_secs).ceil() as usize).max(1);
    let mut window_counts = vec![0usize; num_windows];

    for &t in &sorted {
        let idx = ((t - first) / config.window_secs).floor() as usize;
        let idx = idx.min(num_windows - 1);
        window_counts[idx] += 1;
    }

    let overall_rate = sorted.len() as f64 / num_windows as f64;

    // Use EWMA to build adaptive baseline
    let mut tracker = EwmaTracker::new(config.ewma_alpha);
    let mut bursts = Vec::new();

    for (i, &count) in window_counts.iter().enumerate() {
        let rate = count as f64;
        let window_start = first + i as f64 * config.window_secs;
        let window_end = window_start + config.window_secs;

        if tracker.count >= config.min_windows as u64 {
            let z = tracker.z_score(rate);
            if z > config.z_threshold {
                bursts.push(DetectedBurst {
                    window_start,
                    window_end,
                    event_count: count,
                    z_score: z,
                    baseline_rate: tracker.current(),
                });
            }
        }

        tracker.update(rate, window_start);
    }

    BurstResult {
        bursts,
        overall_rate,
        windows_analyzed: num_windows,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 6  Temporal Motif Mining
// ══════════════════════════════════════════════════════════════════════════════

/// A labeled temporal event for motif mining.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledEvent {
    /// Event type label (e.g., "email_check", "calendar_view", "task_complete").
    pub label: String,
    /// When the event occurred (unix seconds).
    pub timestamp: f64,
}

/// A discovered temporal motif (recurring sequence of event types).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalMotif {
    /// Ordered sequence of event labels.
    pub sequence: Vec<String>,
    /// Number of times this sequence was observed.
    pub occurrences: u32,
    /// Average time span of the motif (seconds from first to last event).
    pub avg_span_secs: f64,
    /// Average inter-event gaps within the motif (seconds).
    pub avg_gaps: Vec<f64>,
    /// Confidence score [0.0, 1.0] based on consistency of timing.
    pub confidence: f64,
}

/// Configuration for motif mining.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifConfig {
    /// Maximum time gap between consecutive events in a motif (seconds).
    /// Default: 7200.0 (2 hours).
    pub max_gap_secs: f64,
    /// Minimum motif length (number of events).
    /// Default: 2.
    pub min_length: usize,
    /// Maximum motif length.
    /// Default: 5.
    pub max_length: usize,
    /// Minimum number of occurrences to report a motif.
    /// Default: 3.
    pub min_occurrences: u32,
}

impl Default for MotifConfig {
    fn default() -> Self {
        Self {
            max_gap_secs: 7200.0,
            min_length: 2,
            max_length: 5,
            min_occurrences: 3,
        }
    }
}

/// Mine temporal motifs from a stream of labeled events.
///
/// Finds recurring subsequences where:
/// 1. Events occur within `max_gap_secs` of each other
/// 2. The same sequence appears at least `min_occurrences` times
/// 3. Timing is reasonably consistent (measured by confidence)
///
/// # Arguments
/// * `events` — Labeled events (need not be sorted).
/// * `config` — Mining parameters.
pub fn mine_temporal_motifs(events: &[LabeledEvent], config: &MotifConfig) -> Vec<TemporalMotif> {
    if events.len() < config.min_length {
        return vec![];
    }

    // Sort by timestamp
    let mut sorted: Vec<&LabeledEvent> = events.iter().collect();
    sorted.sort_by(|a, b| {
        a.timestamp
            .partial_cmp(&b.timestamp)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Extract all subsequences of valid length within max_gap constraint
    // Key: sequence of labels → list of (span_secs, gap_secs_list)
    let mut pattern_instances: HashMap<Vec<String>, Vec<(f64, Vec<f64>)>> = HashMap::new();

    for start_idx in 0..sorted.len() {
        let mut sequence = vec![sorted[start_idx].label.clone()];
        let mut gaps = Vec::new();

        for end_idx in (start_idx + 1)..sorted.len() {
            let gap = sorted[end_idx].timestamp - sorted[end_idx - 1].timestamp;
            if gap > config.max_gap_secs {
                break;
            }
            if gap < 0.0 {
                break; // Should not happen after sorting
            }

            gaps.push(gap);
            sequence.push(sorted[end_idx].label.clone());

            if sequence.len() >= config.min_length && sequence.len() <= config.max_length {
                let span = sorted[end_idx].timestamp - sorted[start_idx].timestamp;
                pattern_instances
                    .entry(sequence.clone())
                    .or_default()
                    .push((span, gaps.clone()));
            }

            if sequence.len() >= config.max_length {
                break;
            }
        }
    }

    // Convert to motifs, filtering by min_occurrences
    let mut motifs: Vec<TemporalMotif> = pattern_instances
        .into_iter()
        .filter(|(_, instances)| instances.len() >= config.min_occurrences as usize)
        .map(|(sequence, instances)| {
            let occurrences = instances.len() as u32;

            // Average span
            let avg_span = instances.iter().map(|(s, _)| s).sum::<f64>() / instances.len() as f64;

            // Average gaps per position
            let gap_count = sequence.len() - 1;
            let mut avg_gaps = vec![0.0; gap_count];
            let mut gap_variances = vec![0.0; gap_count];

            for (_, gaps) in &instances {
                for (i, &g) in gaps.iter().enumerate() {
                    if i < gap_count {
                        avg_gaps[i] += g;
                    }
                }
            }
            for g in &mut avg_gaps {
                *g /= instances.len() as f64;
            }

            // Compute gap variance for confidence
            for (_, gaps) in &instances {
                for (i, &g) in gaps.iter().enumerate() {
                    if i < gap_count {
                        let diff = g - avg_gaps[i];
                        gap_variances[i] += diff * diff;
                    }
                }
            }
            for v in &mut gap_variances {
                *v /= instances.len() as f64;
            }

            // Confidence: based on coefficient of variation of gaps
            // Low variance relative to mean gap → high confidence
            let confidence = if gap_count == 0 {
                1.0
            } else {
                let mean_cv: f64 = avg_gaps
                    .iter()
                    .zip(gap_variances.iter())
                    .map(|(&mean, &var)| {
                        if mean < 1.0 {
                            1.0 // Negligible gap — perfect confidence
                        } else {
                            let cv = var.sqrt() / mean;
                            (1.0 - cv).max(0.0) // Low CV → high confidence
                        }
                    })
                    .sum::<f64>()
                    / gap_count as f64;
                mean_cv.clamp(0.0, 1.0)
            };

            TemporalMotif {
                sequence,
                occurrences,
                avg_span_secs: avg_span,
                avg_gaps,
                confidence,
            }
        })
        .collect();

    // Sort by occurrences × confidence (most significant first)
    motifs.sort_by(|a, b| {
        let score_a = a.occurrences as f64 * a.confidence;
        let score_b = b.occurrences as f64 * b.confidence;
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    motifs
}

// ══════════════════════════════════════════════════════════════════════════════
// § 7  Deadline Urgency
// ══════════════════════════════════════════════════════════════════════════════

/// Configuration for deadline urgency computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadlineUrgencyConfig {
    /// Steepness of the normalized sigmoid curve (higher = sharper transition).
    /// Default: 10.0 (smooth S-curve over the ramp window).
    pub steepness: f64,
    /// How far before the deadline urgency begins to rise (seconds).
    /// Default: 86400.0 (1 day before).
    pub ramp_start_before: f64,
    /// Urgency value when exactly at the deadline.
    /// Default: 0.95 (not quite 1.0 to leave room for overdue escalation).
    pub at_deadline: f64,
    /// Urgency value when past the deadline.
    /// Default: 1.0.
    pub overdue: f64,
    /// Base urgency when far from deadline.
    /// Default: 0.02.
    pub floor: f64,
}

impl Default for DeadlineUrgencyConfig {
    fn default() -> Self {
        Self {
            steepness: 10.0,
            ramp_start_before: 86400.0,
            at_deadline: 0.95,
            overdue: 1.0,
            floor: 0.02,
        }
    }
}

/// Compute deadline urgency for a node.
///
/// Uses a normalized sigmoid over the ramp window:
/// 1. Compute `fraction = 1 - time_remaining / ramp_start_before` (0 at ramp start, 1 at deadline)
/// 2. Apply sigmoid: `sig(k * (fraction - 0.5))`, normalized to [0, 1]
/// 3. Scale to `[floor, at_deadline]`
///
/// Guarantees:
/// - Monotonically increasing as `now` approaches `deadline`
/// - Returns `floor` when far from deadline
/// - Returns `at_deadline` at the deadline
/// - Returns `overdue` when past the deadline
///
/// # Arguments
/// * `deadline` — When the task/goal is due (unix seconds).
/// * `now` — Current time (unix seconds).
/// * `config` — Urgency curve parameters.
///
/// # Returns
/// Urgency in [0.0, 1.0].
pub fn deadline_urgency(deadline: f64, now: f64, config: &DeadlineUrgencyConfig) -> f64 {
    let time_remaining = deadline - now;

    if time_remaining <= 0.0 {
        return config.overdue;
    }

    if time_remaining > config.ramp_start_before {
        return config.floor;
    }

    // fraction: 0.0 at ramp start, 1.0 at deadline
    let fraction = 1.0 - (time_remaining / config.ramp_start_before);

    // Normalized sigmoid: maps [0, 1] → [0, 1]
    let k = config.steepness;
    let raw = 1.0 / (1.0 + (-k * (fraction - 0.5)).exp());
    let sig_low = 1.0 / (1.0 + (k * 0.5).exp()); // sig at fraction=0
    let sig_high = 1.0 / (1.0 + (-k * 0.5).exp()); // sig at fraction=1
    let normalized = (raw - sig_low) / (sig_high - sig_low);

    // Scale to [floor, at_deadline]
    let urgency = config.floor + (config.at_deadline - config.floor) * normalized;
    urgency.clamp(0.0, 1.0)
}

/// Compute compound urgency considering both deadline and priority.
///
/// Higher-priority items become urgent earlier by expanding the ramp window.
/// A Critical task (weight=1.0) starts ramping at 2× the base window.
/// A Low task (weight=0.25) starts ramping at ~0.875× the base window.
pub fn compound_urgency(
    deadline: f64,
    now: f64,
    priority_weight: f64,
    config: &DeadlineUrgencyConfig,
) -> f64 {
    // Priority scales the ramp_start_before: Critical items become urgent earlier
    let scale = 0.5 + 1.5 * priority_weight; // Low=0.875, Med=1.25, High=1.625, Crit=2.0
    let adjusted_config = DeadlineUrgencyConfig {
        ramp_start_before: config.ramp_start_before * scale,
        ..*config
    };
    deadline_urgency(deadline, now, &adjusted_config)
}

// ══════════════════════════════════════════════════════════════════════════════
// § 8  Event Ordering & Causal Sequencing
// ══════════════════════════════════════════════════════════════════════════════

/// An event in a temporal sequence, referenced by node ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEvent {
    /// The cognitive node this event represents.
    pub node_id: super::state::NodeId,
    /// Timestamp of the event (unix seconds).
    pub timestamp: f64,
    /// Node IDs that this event causally/temporally depends on.
    pub predecessors: Vec<super::state::NodeId>,
}

/// Result of topological ordering of temporal events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalOrder {
    /// Events in causal/temporal order (predecessors before dependents).
    pub ordered: Vec<super::state::NodeId>,
    /// Events involved in dependency cycles (if any).
    pub cycles: Vec<super::state::NodeId>,
    /// Depth of each node in the dependency graph (0 = no predecessors).
    pub depths: HashMap<u32, usize>,
}

/// Compute a topological ordering of temporal events.
///
/// Events are ordered such that predecessors appear before dependents.
/// Falls back to timestamp ordering when no explicit dependencies exist.
/// Detects cycles and reports them separately.
///
/// # Arguments
/// * `events` — Temporal events with predecessor relationships.
pub fn topological_order(events: &[TemporalEvent]) -> TemporalOrder {
    if events.is_empty() {
        return TemporalOrder {
            ordered: vec![],
            cycles: vec![],
            depths: HashMap::new(),
        };
    }

    // Build adjacency and in-degree maps
    let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut in_degree: HashMap<u32, usize> = HashMap::new();
    let mut all_nodes: Vec<u32> = Vec::new();

    for event in events {
        let nid = event.node_id.to_raw();
        all_nodes.push(nid);
        in_degree.entry(nid).or_insert(0);
        adj.entry(nid).or_default();

        for &pred in &event.predecessors {
            let pid = pred.to_raw();
            adj.entry(pid).or_default().push(nid);
            *in_degree.entry(nid).or_insert(0) += 1;
            in_degree.entry(pid).or_insert(0);
        }
    }

    // Kahn's algorithm for topological sort
    let mut queue: Vec<u32> = in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(&nid, _)| nid)
        .collect();

    // Sort initial queue by timestamp for deterministic ordering among peers
    let ts_map: HashMap<u32, f64> = events
        .iter()
        .map(|e| (e.node_id.to_raw(), e.timestamp))
        .collect();
    queue.sort_by(|a, b| {
        let ta = ts_map.get(a).copied().unwrap_or(0.0);
        let tb = ts_map.get(b).copied().unwrap_or(0.0);
        ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ordered = Vec::new();
    let mut depths: HashMap<u32, usize> = HashMap::new();
    let mut processed = std::collections::HashSet::new();

    while let Some(nid) = queue.first().copied() {
        queue.remove(0);
        if processed.contains(&nid) {
            continue;
        }
        processed.insert(nid);
        ordered.push(super::state::NodeId::from_raw(nid));

        let depth = *depths.get(&nid).unwrap_or(&0);

        if let Some(successors) = adj.get(&nid) {
            for &succ in successors {
                if let Some(deg) = in_degree.get_mut(&succ) {
                    *deg = deg.saturating_sub(1);
                    let succ_depth = depth + 1;
                    let existing = depths.entry(succ).or_insert(0);
                    *existing = (*existing).max(succ_depth);
                    if *deg == 0 {
                        queue.push(succ);
                    }
                }
            }
        }

        // Re-sort queue by timestamp for deterministic peer ordering
        queue.sort_by(|a, b| {
            let ta = ts_map.get(a).copied().unwrap_or(0.0);
            let tb = ts_map.get(b).copied().unwrap_or(0.0);
            ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Any nodes not in ordered are part of cycles
    let ordered_set: std::collections::HashSet<u32> = ordered.iter().map(|n| n.to_raw()).collect();
    let cycles: Vec<super::state::NodeId> = all_nodes
        .iter()
        .filter(|n| !ordered_set.contains(n))
        .map(|&n| super::state::NodeId::from_raw(n))
        .collect();

    TemporalOrder {
        ordered,
        cycles,
        depths,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 9  Seasonal Histogram
// ══════════════════════════════════════════════════════════════════════════════

/// A histogram that bins events by time-of-day or day-of-week.
///
/// Useful for understanding when events cluster (e.g., "user is most active
/// between 9am-11am on weekdays").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalHistogram {
    /// Number of bins (e.g., 24 for hourly, 7 for daily).
    pub num_bins: usize,
    /// Count per bin.
    pub counts: Vec<u32>,
    /// Total events ingested.
    pub total: u32,
    /// Label for the histogram (e.g., "hour_of_day", "day_of_week").
    pub label: String,
}

impl SeasonalHistogram {
    /// Create a new histogram with the given number of bins.
    pub fn new(num_bins: usize, label: &str) -> Self {
        Self {
            num_bins,
            counts: vec![0; num_bins],
            total: 0,
            label: label.to_string(),
        }
    }

    /// Create an hour-of-day histogram (24 bins, 0=midnight, 23=11pm).
    pub fn hour_of_day() -> Self {
        Self::new(24, "hour_of_day")
    }

    /// Create a day-of-week histogram (7 bins, 0=Monday, 6=Sunday).
    pub fn day_of_week() -> Self {
        Self::new(7, "day_of_week")
    }

    /// Add an event at the given bin index.
    pub fn add(&mut self, bin: usize) {
        if bin < self.num_bins {
            self.counts[bin] += 1;
            self.total += 1;
        }
    }

    /// Add events from unix timestamps using a bin extraction function.
    ///
    /// `bin_fn` maps a timestamp to a bin index.
    pub fn add_timestamps(&mut self, timestamps: &[f64], bin_fn: impl Fn(f64) -> usize) {
        for &t in timestamps {
            let bin = bin_fn(t);
            self.add(bin);
        }
    }

    /// Normalized distribution (each bin as fraction of total).
    pub fn distribution(&self) -> Vec<f64> {
        if self.total == 0 {
            return vec![0.0; self.num_bins];
        }
        self.counts
            .iter()
            .map(|&c| c as f64 / self.total as f64)
            .collect()
    }

    /// Peak bin (most events). Returns (bin_index, count).
    pub fn peak(&self) -> (usize, u32) {
        self.counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, &c)| (i, c))
            .unwrap_or((0, 0))
    }

    /// Top-k bins by count.
    pub fn top_k(&self, k: usize) -> Vec<(usize, u32)> {
        let mut indexed: Vec<(usize, u32)> = self
            .counts
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c))
            .collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));
        indexed.truncate(k);
        indexed
    }

    /// Entropy of the distribution (bits). Low entropy = concentrated pattern.
    pub fn entropy(&self) -> f64 {
        let dist = self.distribution();
        dist.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum()
    }

    /// Concentration ratio: fraction of events in the top-k bins.
    pub fn concentration(&self, top_k: usize) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let top_sum: u32 = self.top_k(top_k).iter().map(|(_, c)| c).sum();
        top_sum as f64 / self.total as f64
    }
}

/// Extract hour-of-day from a unix timestamp.
///
/// Note: This uses UTC. For local time, offset must be applied before calling.
pub fn hour_of_day_utc(timestamp: f64) -> usize {
    let secs_in_day = timestamp % 86400.0;
    let hour = (secs_in_day / 3600.0).floor() as usize;
    hour.min(23)
}

/// Extract day-of-week from a unix timestamp (0=Thursday for epoch, adjusted to 0=Monday).
pub fn day_of_week_utc(timestamp: f64) -> usize {
    // Unix epoch (1970-01-01) was a Thursday (day 4 in ISO, 0-indexed Monday=0)
    let days_since_epoch = (timestamp / 86400.0).floor() as i64;
    let dow = ((days_since_epoch % 7 + 3) % 7) as usize; // 0=Monday
    dow.min(6)
}

// ══════════════════════════════════════════════════════════════════════════════
// § 10  Temporal Relevance Composite
// ══════════════════════════════════════════════════════════════════════════════

/// Configuration for composite temporal relevance scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalRelevanceConfig {
    /// Weight for recency component.
    pub recency_weight: f64,
    /// Weight for periodicity alignment.
    pub periodicity_weight: f64,
    /// Weight for deadline urgency.
    pub deadline_weight: f64,
    /// Recency half-life in seconds.
    pub recency_half_life: f64,
}

impl Default for TemporalRelevanceConfig {
    fn default() -> Self {
        Self {
            recency_weight: 0.4,
            periodicity_weight: 0.3,
            deadline_weight: 0.3,
            recency_half_life: 86400.0,
        }
    }
}

/// Compute a composite temporal relevance score for a cognitive node.
///
/// Combines:
/// - **Recency**: How recently was this node updated?
/// - **Periodicity alignment**: If the node is a routine, how close are we
///   to its next occurrence?
/// - **Deadline urgency**: If the node has a deadline, how urgent is it?
///
/// # Arguments
/// * `last_updated` — When the node was last updated (unix seconds).
/// * `now` — Current time (unix seconds).
/// * `next_occurrence` — Next predicted occurrence if periodic (seconds), or `None`.
/// * `deadline` — Deadline if applicable (seconds), or `None`.
/// * `config` — Scoring weights.
///
/// # Returns
/// Composite relevance in [0.0, 1.0].
pub fn temporal_relevance_composite(
    last_updated: f64,
    now: f64,
    next_occurrence: Option<f64>,
    deadline: Option<f64>,
    config: &TemporalRelevanceConfig,
) -> f64 {
    let recency_config = RecencyConfig {
        half_life_secs: config.recency_half_life,
        floor: 0.01,
    };
    let recency_score = recency_relevance(last_updated, now, &recency_config);

    let periodicity_score = match next_occurrence {
        Some(next) => {
            let time_until = (next - now).max(0.0);
            // Close to next occurrence → high relevance
            // Uses inverse sigmoid: high when time_until is small
            1.0 / (1.0 + (time_until / 3600.0)) // Decays over hours
        }
        None => 0.0,
    };

    let deadline_score = match deadline {
        Some(dl) => deadline_urgency(dl, now, &DeadlineUrgencyConfig::default()),
        None => 0.0,
    };

    // Weighted combination, normalized
    let total_weight = config.recency_weight + config.periodicity_weight + config.deadline_weight;
    if total_weight < 1e-10 {
        return 0.0;
    }

    let composite = (config.recency_weight * recency_score
        + config.periodicity_weight * periodicity_score
        + config.deadline_weight * deadline_score)
        / total_weight;

    composite.clamp(0.0, 1.0)
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Interval Algebra ──

    #[test]
    fn test_interval_before_after() {
        let a = TimeInterval::new(10.0, 20.0);
        let b = TimeInterval::new(30.0, 40.0);
        assert_eq!(a.relation_to(&b), IntervalRelation::Before);
        assert_eq!(b.relation_to(&a), IntervalRelation::After);
        assert!(!a.relation_to(&b).has_overlap());
    }

    #[test]
    fn test_interval_meets() {
        let a = TimeInterval::new(10.0, 20.0);
        let b = TimeInterval::new(20.0, 30.0);
        assert_eq!(a.relation_to(&b), IntervalRelation::Meets);
        assert_eq!(b.relation_to(&a), IntervalRelation::MetBy);
    }

    #[test]
    fn test_interval_overlaps() {
        let a = TimeInterval::new(10.0, 25.0);
        let b = TimeInterval::new(20.0, 35.0);
        assert_eq!(a.relation_to(&b), IntervalRelation::Overlaps);
        assert_eq!(b.relation_to(&a), IntervalRelation::OverlappedBy);
        assert!(a.relation_to(&b).has_overlap());
    }

    #[test]
    fn test_interval_during_contains() {
        let a = TimeInterval::new(15.0, 25.0);
        let b = TimeInterval::new(10.0, 30.0);
        assert_eq!(a.relation_to(&b), IntervalRelation::During);
        assert_eq!(b.relation_to(&a), IntervalRelation::Contains);
    }

    #[test]
    fn test_interval_starts_finishes() {
        let a = TimeInterval::new(10.0, 20.0);
        let b = TimeInterval::new(10.0, 30.0);
        assert_eq!(a.relation_to(&b), IntervalRelation::Starts);
        assert_eq!(b.relation_to(&a), IntervalRelation::StartedBy);

        let c = TimeInterval::new(20.0, 30.0);
        let d = TimeInterval::new(10.0, 30.0);
        assert_eq!(c.relation_to(&d), IntervalRelation::Finishes);
        assert_eq!(d.relation_to(&c), IntervalRelation::FinishedBy);
    }

    #[test]
    fn test_interval_equals() {
        let a = TimeInterval::new(10.0, 20.0);
        let b = TimeInterval::new(10.0, 20.0);
        assert_eq!(a.relation_to(&b), IntervalRelation::Equals);
    }

    #[test]
    fn test_interval_overlap_duration() {
        let a = TimeInterval::new(10.0, 30.0);
        let b = TimeInterval::new(20.0, 40.0);
        assert!((a.overlap_duration(&b) - 10.0).abs() < 1e-10);

        let c = TimeInterval::new(50.0, 60.0);
        assert!((a.overlap_duration(&c) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_interval_merge() {
        let a = TimeInterval::new(10.0, 25.0);
        let b = TimeInterval::new(20.0, 35.0);
        let merged = a.merge(&b, 0.0).unwrap();
        assert!((merged.start - 10.0).abs() < 1e-10);
        assert!((merged.end - 35.0).abs() < 1e-10);

        let c = TimeInterval::new(50.0, 60.0);
        assert!(a.merge(&c, 0.0).is_none());
    }

    #[test]
    fn test_interval_inverse_symmetry() {
        for rel in [
            IntervalRelation::Before,
            IntervalRelation::After,
            IntervalRelation::Meets,
            IntervalRelation::MetBy,
            IntervalRelation::Overlaps,
            IntervalRelation::OverlappedBy,
            IntervalRelation::Starts,
            IntervalRelation::StartedBy,
            IntervalRelation::Finishes,
            IntervalRelation::FinishedBy,
            IntervalRelation::During,
            IntervalRelation::Contains,
            IntervalRelation::Equals,
        ] {
            assert_eq!(rel.inverse().inverse(), rel);
        }
    }

    // ── Recency ──

    #[test]
    fn test_recency_just_happened() {
        let config = RecencyConfig::default();
        let score = recency_relevance(1000.0, 1000.0, &config);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_recency_one_halflife_ago() {
        let config = RecencyConfig {
            half_life_secs: 100.0,
            floor: 0.0,
        };
        let score = recency_relevance(0.0, 100.0, &config);
        assert!((score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_recency_floor() {
        let config = RecencyConfig {
            half_life_secs: 100.0,
            floor: 0.05,
        };
        // Very old event
        let score = recency_relevance(0.0, 1_000_000.0, &config);
        assert!((score - 0.05).abs() < 0.01);
    }

    #[test]
    fn test_recency_ranking() {
        let config = RecencyConfig::default();
        let events = vec![(1, 100.0), (2, 500.0), (3, 300.0)];
        let ranked = rank_by_recency(&events, 1000.0, &config);
        assert_eq!(ranked[0].0, 2); // Most recent
        assert_eq!(ranked[1].0, 3);
        assert_eq!(ranked[2].0, 1); // Oldest
    }

    #[test]
    fn test_domain_recency() {
        let map = DomainRecencyMap::default();
        // Conversations decay faster than goals
        let conv_half = map.half_life_for("conversation");
        let goal_half = map.half_life_for("goal");
        assert!(conv_half < goal_half);
    }

    // ── EWMA ──

    #[test]
    fn test_ewma_converges() {
        let mut tracker = EwmaTracker::new(0.3);
        for i in 0..100 {
            tracker.update(10.0, i as f64);
        }
        assert!((tracker.current() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_ewma_tracks_change() {
        let mut tracker = EwmaTracker::new(0.5);
        for i in 0..20 {
            tracker.update(5.0, i as f64);
        }
        // Now shift to 15.0
        for i in 20..40 {
            tracker.update(15.0, i as f64);
        }
        // Should be close to 15.0
        assert!((tracker.current() - 15.0).abs() < 0.5);
    }

    #[test]
    fn test_ewma_z_score() {
        let mut tracker = EwmaTracker::new(0.2);
        // Establish baseline at ~10
        for i in 0..50 {
            tracker.update(10.0 + (i as f64 * 0.01), i as f64);
        }
        // Spike should have high z-score
        let z = tracker.z_score(50.0);
        assert!(z > 2.0);
    }

    #[test]
    fn test_ewma_anomaly_detection() {
        let mut tracker = EwmaTracker::new(0.1);
        for i in 0..100 {
            tracker.update(5.0 + (i % 2) as f64, i as f64);
        }
        assert!(!tracker.is_anomaly(5.5, 3.0)); // Normal
        assert!(tracker.is_anomaly(100.0, 3.0)); // Anomalous
    }

    // ── Periodicity ──

    #[test]
    fn test_detect_daily_periodicity() {
        let config = PeriodicityConfig {
            min_events: 3,
            correlation_threshold: 0.3,
            ..Default::default()
        };
        // Generate events at ~9am daily for 2 weeks
        let base = 1000000.0;
        let daily = 86400.0;
        let timestamps: Vec<f64> = (0..14)
            .map(|d| base + d as f64 * daily + 100.0) // Small jitter
            .collect();

        let result = detect_periodicity(&timestamps, &config);
        assert!(!result.detected.is_empty());
        // Should detect daily period
        let has_daily = result
            .detected
            .iter()
            .any(|p| (p.period_secs - 86400.0).abs() < 1.0);
        assert!(
            has_daily,
            "Should detect daily period: {:?}",
            result.detected
        );
    }

    #[test]
    fn test_periodicity_insufficient_data() {
        let config = PeriodicityConfig {
            min_events: 10,
            ..Default::default()
        };
        let timestamps = vec![100.0, 200.0, 300.0];
        let result = detect_periodicity(&timestamps, &config);
        assert!(result.detected.is_empty());
    }

    #[test]
    fn test_periodicity_random_noise() {
        let config = PeriodicityConfig {
            min_events: 5,
            correlation_threshold: 0.5, // Higher threshold
            ..Default::default()
        };
        // Random-ish timestamps (no real periodicity)
        let timestamps = vec![
            100.0, 357.0, 981.0, 1234.0, 2001.0, 3500.0, 5600.0, 8900.0, 14000.0, 22000.0,
        ];
        let result = detect_periodicity(&timestamps, &config);
        // Should find few or no strong periodicities
        // (may find weak ones by chance, but correlation should be lower)
        for p in &result.detected {
            assert!(
                p.correlation < 0.9,
                "Unexpected strong periodicity in noise: {:?}",
                p
            );
        }
    }

    // ── Burst Detection ──

    #[test]
    fn test_detect_burst() {
        let config = BurstConfig {
            window_secs: 100.0,
            z_threshold: 2.0,
            min_windows: 3,
            ewma_alpha: 0.3,
        };
        // Normal rate: ~2 events per window, then a burst of 20
        let mut timestamps = Vec::new();
        // Normal windows (0-600s, ~2 per window)
        for i in 0..6 {
            timestamps.push(i as f64 * 100.0 + 10.0);
            timestamps.push(i as f64 * 100.0 + 50.0);
        }
        // Burst window (600-700s: 20 events)
        for j in 0..20 {
            timestamps.push(600.0 + j as f64 * 4.0);
        }

        let result = detect_bursts(&timestamps, &config);
        assert!(!result.bursts.is_empty(), "Should detect burst");
        // The burst should be in the 600-700 window
        let burst = &result.bursts[0];
        assert!(burst.event_count > 5);
        assert!(burst.z_score > 2.0);
    }

    #[test]
    fn test_no_burst_uniform() {
        let config = BurstConfig::default();
        // Uniform rate
        let timestamps: Vec<f64> = (0..100).map(|i| i as f64 * 100.0).collect();
        let result = detect_bursts(&timestamps, &config);
        assert!(
            result.bursts.is_empty(),
            "Uniform rate should not trigger burst"
        );
    }

    // ── Deadline Urgency ──

    #[test]
    fn test_urgency_overdue() {
        let config = DeadlineUrgencyConfig::default();
        let urgency = deadline_urgency(1000.0, 2000.0, &config);
        assert!((urgency - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_urgency_far_away() {
        let config = DeadlineUrgencyConfig::default();
        let urgency = deadline_urgency(1_000_000.0, 0.0, &config);
        assert!(
            (urgency - 0.02).abs() < 1e-10,
            "Far-away deadline should equal floor: {urgency}"
        );
    }

    #[test]
    fn test_urgency_monotonic_increase() {
        let config = DeadlineUrgencyConfig::default();
        let deadline = 100_000.0;
        let mut prev = 0.0;
        for t in (0..100_000).step_by(10_000) {
            let u = deadline_urgency(deadline, t as f64, &config);
            assert!(
                u >= prev - 1e-10,
                "Urgency should increase: {prev} -> {u} at t={t}"
            );
            prev = u;
        }
    }

    #[test]
    fn test_compound_urgency_priority_scaling() {
        let config = DeadlineUrgencyConfig::default();
        let deadline = 100_000.0;
        let now = 50_000.0;

        let low = compound_urgency(deadline, now, 0.25, &config);
        let high = compound_urgency(deadline, now, 1.0, &config);
        // Higher priority should cause higher urgency at the same time
        assert!(
            high >= low,
            "Higher priority should mean higher urgency: low={low}, high={high}"
        );
    }

    // ── Temporal Motifs ──

    #[test]
    fn test_mine_recurring_sequence() {
        let config = MotifConfig {
            max_gap_secs: 3600.0,
            min_length: 2,
            max_length: 3,
            min_occurrences: 3,
        };

        // Repeat pattern: email → calendar → standup, 3 times
        let mut events = Vec::new();
        for day in 0..3 {
            let base = day as f64 * 86400.0;
            events.push(LabeledEvent {
                label: "email".to_string(),
                timestamp: base + 100.0,
            });
            events.push(LabeledEvent {
                label: "calendar".to_string(),
                timestamp: base + 600.0,
            });
            events.push(LabeledEvent {
                label: "standup".to_string(),
                timestamp: base + 1200.0,
            });
        }

        let motifs = mine_temporal_motifs(&events, &config);
        assert!(!motifs.is_empty());
        // Should find the email→calendar pattern
        let has_email_cal = motifs.iter().any(|m| {
            m.sequence.len() >= 2 && m.sequence[0] == "email" && m.sequence[1] == "calendar"
        });
        assert!(
            has_email_cal,
            "Should find email→calendar motif: {:?}",
            motifs
        );
    }

    #[test]
    fn test_motif_gap_constraint() {
        let config = MotifConfig {
            max_gap_secs: 100.0,
            min_length: 2,
            max_length: 3,
            min_occurrences: 2,
        };

        // Events too far apart to form motifs
        let events: Vec<LabeledEvent> = (0..6)
            .map(|i| {
                LabeledEvent {
                    label: if i % 2 == 0 {
                        "a".to_string()
                    } else {
                        "b".to_string()
                    },
                    timestamp: i as f64 * 1000.0, // 1000s gaps > 100s max
                }
            })
            .collect();

        let motifs = mine_temporal_motifs(&events, &config);
        assert!(
            motifs.is_empty(),
            "Large gaps should prevent motif detection"
        );
    }

    // ── Topological Order ──

    #[test]
    fn test_topological_order_simple() {
        use super::super::state::{NodeId, NodeKind};

        let a = NodeId::new(NodeKind::Episode, 1);
        let b = NodeId::new(NodeKind::Episode, 2);
        let c = NodeId::new(NodeKind::Episode, 3);

        let events = vec![
            TemporalEvent {
                node_id: a,
                timestamp: 100.0,
                predecessors: vec![],
            },
            TemporalEvent {
                node_id: b,
                timestamp: 200.0,
                predecessors: vec![a],
            },
            TemporalEvent {
                node_id: c,
                timestamp: 300.0,
                predecessors: vec![b],
            },
        ];

        let order = topological_order(&events);
        assert_eq!(order.ordered.len(), 3);
        assert_eq!(order.ordered[0], a);
        assert_eq!(order.ordered[1], b);
        assert_eq!(order.ordered[2], c);
        assert!(order.cycles.is_empty());
    }

    #[test]
    fn test_topological_order_with_cycle() {
        use super::super::state::{NodeId, NodeKind};

        let a = NodeId::new(NodeKind::Episode, 1);
        let b = NodeId::new(NodeKind::Episode, 2);

        let events = vec![
            TemporalEvent {
                node_id: a,
                timestamp: 100.0,
                predecessors: vec![b],
            },
            TemporalEvent {
                node_id: b,
                timestamp: 200.0,
                predecessors: vec![a],
            },
        ];

        let order = topological_order(&events);
        // Both nodes should be in cycles since they mutually depend
        assert!(!order.cycles.is_empty());
    }

    #[test]
    fn test_topological_depth() {
        use super::super::state::{NodeId, NodeKind};

        let a = NodeId::new(NodeKind::Episode, 1);
        let b = NodeId::new(NodeKind::Episode, 2);
        let c = NodeId::new(NodeKind::Episode, 3);

        let events = vec![
            TemporalEvent {
                node_id: a,
                timestamp: 100.0,
                predecessors: vec![],
            },
            TemporalEvent {
                node_id: b,
                timestamp: 200.0,
                predecessors: vec![a],
            },
            TemporalEvent {
                node_id: c,
                timestamp: 300.0,
                predecessors: vec![a, b],
            },
        ];

        let order = topological_order(&events);
        assert_eq!(*order.depths.get(&b.to_raw()).unwrap_or(&0), 1);
        assert_eq!(*order.depths.get(&c.to_raw()).unwrap_or(&0), 2);
    }

    // ── Seasonal Histogram ──

    #[test]
    fn test_seasonal_histogram_peak() {
        let mut hist = SeasonalHistogram::hour_of_day();
        // Most events at 9am
        for _ in 0..20 {
            hist.add(9);
        }
        for _ in 0..5 {
            hist.add(14);
        }
        for _ in 0..3 {
            hist.add(22);
        }

        let (peak_bin, peak_count) = hist.peak();
        assert_eq!(peak_bin, 9);
        assert_eq!(peak_count, 20);
    }

    #[test]
    fn test_seasonal_entropy() {
        // Uniform distribution → maximum entropy
        let mut uniform = SeasonalHistogram::new(4, "test");
        for _ in 0..25 {
            uniform.add(0);
        }
        for _ in 0..25 {
            uniform.add(1);
        }
        for _ in 0..25 {
            uniform.add(2);
        }
        for _ in 0..25 {
            uniform.add(3);
        }

        // Concentrated → low entropy
        let mut concentrated = SeasonalHistogram::new(4, "test");
        for _ in 0..100 {
            concentrated.add(0);
        }

        assert!(uniform.entropy() > concentrated.entropy());
    }

    #[test]
    fn test_concentration_ratio() {
        let mut hist = SeasonalHistogram::new(10, "test");
        for _ in 0..50 {
            hist.add(0);
        }
        for _ in 0..30 {
            hist.add(1);
        }
        for _ in 0..20 {
            hist.add(2);
        }

        let c2 = hist.concentration(2);
        assert!((c2 - 0.80).abs() < 1e-10); // Top 2 bins = 80%
    }

    // ── Composite Relevance ──

    #[test]
    fn test_composite_all_recent() {
        let config = TemporalRelevanceConfig::default();
        let now = 1000.0;
        let score = temporal_relevance_composite(now, now, None, None, &config);
        // Only recency contributes (1.0), periodicity and deadline are 0
        // score = (0.4 * 1.0 + 0.3 * 0.0 + 0.3 * 0.0) / 1.0 = 0.4
        assert!((score - 0.4).abs() < 0.05);
    }

    #[test]
    fn test_composite_with_deadline() {
        let config = TemporalRelevanceConfig::default();
        let now = 1000.0;
        let deadline = 1001.0; // 1 second away
        let score = temporal_relevance_composite(now, now, None, Some(deadline), &config);
        // Recency = 1.0 (just now), deadline urgency should be high
        assert!(score > 0.5);
    }

    #[test]
    fn test_day_of_week_utc() {
        // 2024-01-01 (Monday) = 1704067200
        let monday = 1704067200.0;
        assert_eq!(day_of_week_utc(monday), 0); // Monday = 0
    }

    #[test]
    fn test_hour_of_day_utc() {
        // Midnight UTC = hour 0
        let midnight = 86400.0 * 100.0; // Some midnight
        assert_eq!(hour_of_day_utc(midnight), 0);
        // 9am UTC
        let nine_am = midnight + 9.0 * 3600.0;
        assert_eq!(hour_of_day_utc(nine_am), 9);
    }
}
