//! CK-2.3: Routine Prediction via Hawkes Processes
//!
//! Implements self-exciting point process models for predicting when recurring
//! events will happen next. Each event type gets its own lightweight model that
//! learns from observation history and produces calibrated intensity forecasts.
//!
//! ## Hawkes Process Model
//!
//! The conditional intensity function:
//!
//! ```text
//! λ(t) = μ + Σᵢ α · exp(-β · (t - tᵢ))
//! ```
//!
//! Where:
//! - `μ` = base rate (events per second, learned per event type)
//! - `α` = excitation parameter (how much past events boost future likelihood)
//! - `β` = decay parameter (how quickly excitation fades)
//! - `tᵢ` = timestamps of past events
//!
//! The model captures **self-excitation**: the occurrence of an event makes
//! future events of the same type temporarily more likely (clustering effect).
//! This is perfect for human behavioral patterns — checking email once makes
//! you likely to check again soon.
//!
//! ## Circadian Modulation
//!
//! Raw Hawkes processes don't capture time-of-day effects. We add a circadian
//! multiplier `c(t)` that modulates the base rate:
//!
//! ```text
//! λ(t) = μ · c(t) + Σᵢ α · exp(-β · (t - tᵢ))
//! ```
//!
//! Where `c(t)` is learned from a 24-bin hour-of-day histogram, normalized
//! so that `mean(c) = 1.0`.
//!
//! ## Parameter Learning
//!
//! Online MLE via simplified EM:
//! 1. After each observation, update sufficient statistics
//! 2. Periodically re-estimate (μ, α, β) from statistics
//! 3. Warm-start from observed event rate and inter-event times
//!
//! ## Performance
//!
//! - O(k) per intensity evaluation where k = recent event count
//! - Only recent events within the excitation window (5/β seconds) matter
//! - Typically k < 100 even for frequent events

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::temporal::SeasonalHistogram;

// ══════════════════════════════════════════════════════════════════════════════
// § 1  Hawkes Process Core
// ══════════════════════════════════════════════════════════════════════════════

/// Parameters for a single Hawkes process model.
///
/// These three numbers fully define the intensity dynamics for one event type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HawkesParams {
    /// Base rate μ (events per second). Always > 0.
    pub mu: f64,
    /// Excitation magnitude α. How much each past event boosts intensity.
    /// α ∈ [0, β) for stability (ensures process doesn't explode).
    pub alpha: f64,
    /// Excitation decay rate β (1/seconds). Higher = faster decay.
    /// Must be > α for stability (branching ratio α/β < 1).
    pub beta: f64,
}

impl HawkesParams {
    /// Create new params with stability validation.
    ///
    /// # Panics
    /// Panics if `mu <= 0`, `alpha < 0`, `beta <= 0`, or `alpha >= beta` (unstable).
    pub fn new(mu: f64, alpha: f64, beta: f64) -> Self {
        debug_assert!(mu > 0.0, "mu must be positive: {mu}");
        debug_assert!(alpha >= 0.0, "alpha must be non-negative: {alpha}");
        debug_assert!(beta > 0.0, "beta must be positive: {beta}");
        debug_assert!(
            alpha < beta,
            "branching ratio alpha/beta must be < 1 for stability: {alpha}/{beta}"
        );
        Self { mu, alpha, beta }
    }

    /// Create conservative default params for a given observed event rate.
    ///
    /// `observed_rate` is events per second.
    pub fn from_rate(observed_rate: f64) -> Self {
        let mu = observed_rate.max(1e-8);
        // Conservative: mild self-excitation
        let beta = 1.0 / 300.0; // 5-minute decay half-life
        let alpha = beta * 0.3; // Branching ratio 0.3
        Self { mu, alpha, beta }
    }

    /// Branching ratio α/β. Must be < 1 for stability.
    /// Higher ratio = stronger clustering effect.
    #[inline]
    pub fn branching_ratio(&self) -> f64 {
        if self.beta > 0.0 {
            self.alpha / self.beta
        } else {
            0.0
        }
    }

    /// Expected stationary rate: μ / (1 - α/β).
    /// This is the long-term average event rate including self-excitation.
    pub fn stationary_rate(&self) -> f64 {
        let br = self.branching_ratio();
        if br >= 1.0 {
            f64::INFINITY
        } else {
            self.mu / (1.0 - br)
        }
    }

    /// Effective excitation window: events older than this contribute < 1%
    /// of their peak excitation. Equals `ln(100) / β ≈ 4.6 / β`.
    pub fn excitation_window(&self) -> f64 {
        if self.beta > 0.0 {
            4.6 / self.beta
        } else {
            f64::INFINITY
        }
    }
}

impl Default for HawkesParams {
    fn default() -> Self {
        Self {
            mu: 1.0 / 3600.0,  // 1 event per hour
            alpha: 0.001,      // Mild excitation
            beta: 1.0 / 300.0, // 5-minute decay
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 2  Circadian Modulation
// ══════════════════════════════════════════════════════════════════════════════

/// Circadian (time-of-day) modulation of the base rate.
///
/// 24 bins (one per hour), normalized so mean = 1.0.
/// A value of 2.0 at hour 9 means the base rate is doubled at 9am.
/// A value of 0.1 at hour 3 means the base rate is 10% at 3am.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircadianProfile {
    /// Multiplier per hour [0..23]. Normalized so mean ≈ 1.0.
    pub hourly_multipliers: [f64; 24],
    /// Total events observed (for learning weight).
    pub observation_count: u32,
}

impl CircadianProfile {
    /// Flat profile (no time-of-day modulation).
    pub fn flat() -> Self {
        Self {
            hourly_multipliers: [1.0; 24],
            observation_count: 0,
        }
    }

    /// Learn circadian profile from a seasonal histogram.
    ///
    /// Normalizes the histogram so that mean multiplier = 1.0.
    /// Uses Laplace smoothing to handle empty bins.
    pub fn from_histogram(hist: &SeasonalHistogram) -> Self {
        assert!(hist.num_bins == 24, "Expected 24-bin hour histogram");

        let total = hist.total.max(1) as f64;
        let smoothing = 1.0; // Laplace smoothing: add 1 pseudo-count per bin
        let smoothed_total = total + 24.0 * smoothing;

        let mut multipliers = [0.0; 24];
        for i in 0..24 {
            let smoothed_count = hist.counts[i] as f64 + smoothing;
            // Raw probability × 24 gives multiplier (mean = 1.0)
            multipliers[i] = (smoothed_count / smoothed_total) * 24.0;
        }

        Self {
            hourly_multipliers: multipliers,
            observation_count: hist.total,
        }
    }

    /// Get the circadian multiplier for a given unix timestamp.
    ///
    /// Interpolates between adjacent hours for smooth transitions.
    pub fn multiplier_at(&self, timestamp: f64) -> f64 {
        let secs_in_day = timestamp % 86400.0;
        let fractional_hour = secs_in_day / 3600.0;
        let hour_low = fractional_hour.floor() as usize % 24;
        let hour_high = (hour_low + 1) % 24;
        let frac = fractional_hour - fractional_hour.floor();

        // Linear interpolation between adjacent hours
        let m_low = self.hourly_multipliers[hour_low];
        let m_high = self.hourly_multipliers[hour_high];
        m_low + frac * (m_high - m_low)
    }

    /// Update the profile with a new observation at the given timestamp.
    ///
    /// Uses exponential moving average to gradually adapt.
    pub fn observe(&mut self, timestamp: f64, learning_rate: f64) {
        let hour = super::temporal::hour_of_day_utc(timestamp);
        self.observation_count += 1;

        // Boost the observed hour, decay others slightly
        for i in 0..24 {
            if i == hour {
                self.hourly_multipliers[i] += learning_rate * (2.0 - self.hourly_multipliers[i]);
            } else {
                // Slight pull toward 1.0 for unobserved hours
                self.hourly_multipliers[i] +=
                    learning_rate * 0.1 * (1.0 - self.hourly_multipliers[i]);
            }
        }

        // Re-normalize to mean = 1.0
        let sum: f64 = self.hourly_multipliers.iter().sum();
        if sum > 0.0 {
            let scale = 24.0 / sum;
            for m in &mut self.hourly_multipliers {
                *m *= scale;
            }
        }
    }
}

impl Default for CircadianProfile {
    fn default() -> Self {
        Self::flat()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  Event Type Model
// ══════════════════════════════════════════════════════════════════════════════

/// A complete Hawkes process model for one event type.
///
/// Combines the core Hawkes parameters with circadian modulation
/// and maintains the event history needed for intensity computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventTypeModel {
    /// The event type label (e.g., "email_check", "terminal_open").
    pub label: String,
    /// Core Hawkes parameters.
    pub params: HawkesParams,
    /// Circadian modulation profile.
    pub circadian: CircadianProfile,
    /// Recent event timestamps (sorted ascending).
    /// Only events within the excitation window are kept.
    pub recent_events: Vec<f64>,
    /// Maximum number of events to retain.
    pub max_history: usize,
    /// Total observations ever recorded (for learning).
    pub total_observations: u64,
    /// Sum of inter-event times (for online μ estimation).
    pub sum_inter_event: f64,
    /// Sum of squared inter-event times (for variance estimation).
    pub sum_inter_event_sq: f64,
    /// Count of inter-event intervals recorded.
    pub inter_event_count: u64,
    /// Last time parameters were re-estimated.
    pub last_refit_at: f64,
    /// Minimum interval between parameter re-estimations (seconds).
    pub refit_interval: f64,
}

impl EventTypeModel {
    /// Create a new model for an event type with default parameters.
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            params: HawkesParams::default(),
            circadian: CircadianProfile::flat(),
            recent_events: Vec::new(),
            max_history: 500,
            total_observations: 0,
            sum_inter_event: 0.0,
            sum_inter_event_sq: 0.0,
            inter_event_count: 0,
            last_refit_at: 0.0,
            refit_interval: 3600.0, // Re-estimate params at most once per hour
        }
    }

    /// Create a model warm-started from observed timestamps.
    pub fn from_observations(label: &str, timestamps: &[f64]) -> Self {
        let mut model = Self::new(label);
        if timestamps.is_empty() {
            return model;
        }

        let mut sorted = timestamps.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute inter-event statistics
        let mut intervals = Vec::new();
        for w in sorted.windows(2) {
            let dt = w[1] - w[0];
            if dt > 0.0 {
                intervals.push(dt);
            }
        }

        if !intervals.is_empty() {
            let mean_interval: f64 = intervals.iter().sum::<f64>() / intervals.len() as f64;
            let var_interval: f64 = intervals
                .iter()
                .map(|&dt| (dt - mean_interval).powi(2))
                .sum::<f64>()
                / intervals.len() as f64;

            // Estimate μ from mean rate
            let mu = if mean_interval > 0.0 {
                1.0 / mean_interval
            } else {
                1.0 / 3600.0
            };

            // Estimate clustering: if variance >> mean², there's clustering
            let cv_squared = if mean_interval > 0.0 {
                var_interval / (mean_interval * mean_interval)
            } else {
                0.0
            };

            // Higher CV² → stronger self-excitation
            let branching_ratio = (cv_squared / (1.0 + cv_squared)).clamp(0.05, 0.85);

            // β from median inter-event time
            let mut sorted_intervals = intervals.clone();
            sorted_intervals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = sorted_intervals[sorted_intervals.len() / 2];
            let beta = if median > 0.0 {
                1.0 / median
            } else {
                1.0 / 300.0
            };
            let alpha = branching_ratio * beta;

            model.params = HawkesParams { mu, alpha, beta };

            model.sum_inter_event = intervals.iter().sum();
            model.sum_inter_event_sq = intervals.iter().map(|dt| dt * dt).sum();
            model.inter_event_count = intervals.len() as u64;
        }

        // Build circadian profile
        let mut hist = super::temporal::SeasonalHistogram::hour_of_day();
        for &t in &sorted {
            hist.add(super::temporal::hour_of_day_utc(t));
        }
        model.circadian = CircadianProfile::from_histogram(&hist);

        // Keep recent events
        let window = model.params.excitation_window();
        let cutoff = sorted.last().unwrap() - window;
        model.recent_events = sorted.into_iter().filter(|&t| t >= cutoff).collect();
        if model.recent_events.len() > model.max_history {
            let start = model.recent_events.len() - model.max_history;
            model.recent_events = model.recent_events[start..].to_vec();
        }
        model.total_observations = timestamps.len() as u64;

        model
    }

    /// Record a new event observation and update the model.
    pub fn observe(&mut self, timestamp: f64) {
        // Update inter-event statistics
        if let Some(&last) = self.recent_events.last() {
            let dt = timestamp - last;
            if dt > 0.0 {
                self.sum_inter_event += dt;
                self.sum_inter_event_sq += dt * dt;
                self.inter_event_count += 1;
            }
        }

        self.recent_events.push(timestamp);
        self.total_observations += 1;

        // Prune old events outside excitation window
        let window = self.params.excitation_window();
        let cutoff = timestamp - window;
        self.recent_events.retain(|&t| t >= cutoff);

        // Enforce max history
        if self.recent_events.len() > self.max_history {
            let start = self.recent_events.len() - self.max_history;
            self.recent_events = self.recent_events[start..].to_vec();
        }

        // Update circadian profile
        self.circadian.observe(timestamp, 0.05);

        // Periodically re-estimate parameters
        if timestamp - self.last_refit_at >= self.refit_interval && self.inter_event_count >= 10 {
            self.refit_parameters();
            self.last_refit_at = timestamp;
        }
    }

    /// Re-estimate Hawkes parameters from accumulated statistics.
    ///
    /// Uses method-of-moments estimation:
    /// - μ estimated from mean event rate
    /// - α/β estimated from clustering (coefficient of variation)
    fn refit_parameters(&mut self) {
        if self.inter_event_count < 5 {
            return;
        }

        let n = self.inter_event_count as f64;
        let mean_dt = self.sum_inter_event / n;
        let var_dt = (self.sum_inter_event_sq / n) - (mean_dt * mean_dt);

        if mean_dt <= 0.0 {
            return;
        }

        // New μ estimate (smoothed with prior)
        let new_mu = 1.0 / mean_dt;
        self.params.mu = 0.7 * self.params.mu + 0.3 * new_mu;
        self.params.mu = self.params.mu.max(1e-8);

        // Clustering estimate from CV²
        let cv_sq = (var_dt / (mean_dt * mean_dt)).max(0.0);
        let new_br = (cv_sq / (1.0 + cv_sq)).clamp(0.05, 0.85);

        // New β from mean inter-event time
        let new_beta = (1.0 / mean_dt).max(1e-6);
        self.params.beta = 0.7 * self.params.beta + 0.3 * new_beta;

        // α from branching ratio
        let old_br = self.params.branching_ratio();
        let smoothed_br = 0.7 * old_br + 0.3 * new_br;
        self.params.alpha = smoothed_br * self.params.beta;

        // Safety: ensure stability
        if self.params.alpha >= self.params.beta {
            self.params.alpha = self.params.beta * 0.85;
        }
    }

    /// Compute the conditional intensity λ(t) at a given time.
    ///
    /// This is the instantaneous event rate: higher values mean the event
    /// is more likely to happen "right now".
    pub fn intensity(&self, t: f64) -> f64 {
        let circadian = self.circadian.multiplier_at(t);
        let base = self.params.mu * circadian;

        let excitation: f64 = self
            .recent_events
            .iter()
            .filter(|&&ti| ti < t)
            .map(|&ti| self.params.alpha * (-self.params.beta * (t - ti)).exp())
            .sum();

        (base + excitation).max(0.0)
    }

    /// Compute intensity curve over a time range.
    ///
    /// Returns `(timestamp, intensity)` pairs sampled at `step_secs` intervals.
    pub fn intensity_curve(&self, start: f64, end: f64, step_secs: f64) -> Vec<(f64, f64)> {
        let mut curve = Vec::new();
        let mut t = start;
        while t <= end {
            curve.push((t, self.intensity(t)));
            t += step_secs;
        }
        curve
    }

    /// Predict the next event time using intensity-based sampling.
    ///
    /// Scans forward from `now` in small steps, returning the first time
    /// where cumulative intensity exceeds the threshold (expected event).
    ///
    /// Returns `(predicted_time, confidence)` where confidence is derived
    /// from the peak intensity relative to base rate.
    pub fn predict_next(
        &self,
        now: f64,
        horizon_secs: f64,
        step_secs: f64,
    ) -> Option<EventPrediction> {
        if self.total_observations < 3 {
            return None; // Not enough data
        }

        let mut cumulative = 0.0;
        let mut peak_intensity = 0.0_f64;
        let mut peak_time = now;
        let mut t = now;
        let end = now + horizon_secs;

        while t <= end {
            let lambda = self.intensity(t);
            cumulative += lambda * step_secs;
            if lambda > peak_intensity {
                peak_intensity = lambda;
                peak_time = t;
            }

            // Expected event: cumulative intensity reaches 1.0
            // (Poisson: expected count = integral of intensity)
            if cumulative >= 1.0 {
                let stationary = self.params.stationary_rate();
                let confidence = if stationary > 0.0 {
                    (peak_intensity / stationary).clamp(0.0, 1.0)
                } else {
                    0.5
                };

                return Some(EventPrediction {
                    predicted_time: t,
                    peak_intensity_time: peak_time,
                    peak_intensity,
                    confidence,
                    time_until: t - now,
                });
            }

            t += step_secs;
        }

        // No event predicted within horizon — return peak time as best guess
        if peak_intensity > 0.0 {
            let stationary = self.params.stationary_rate();
            let confidence = if stationary > 0.0 {
                (peak_intensity / stationary * 0.3).clamp(0.0, 0.5)
            } else {
                0.1
            };

            Some(EventPrediction {
                predicted_time: peak_time,
                peak_intensity_time: peak_time,
                peak_intensity,
                confidence,
                time_until: peak_time - now,
            })
        } else {
            None
        }
    }

    /// Whether an event should be anticipated right now.
    ///
    /// Returns true if the current intensity exceeds the threshold multiplied
    /// by the stationary rate. A threshold of 1.5 means "50% more likely than
    /// average".
    pub fn should_anticipate(&self, now: f64, threshold_multiplier: f64) -> bool {
        let lambda = self.intensity(now);
        let threshold = self.params.stationary_rate() * threshold_multiplier;
        lambda > threshold
    }

    /// Summary statistics for this model.
    pub fn summary(&self) -> ModelSummary {
        let (peak_hour, peak_mult) = self
            .circadian
            .hourly_multipliers
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(h, &m)| (h, m))
            .unwrap_or((0, 1.0));

        ModelSummary {
            label: self.label.clone(),
            base_rate_per_hour: self.params.mu * 3600.0,
            stationary_rate_per_hour: self.params.stationary_rate() * 3600.0,
            branching_ratio: self.params.branching_ratio(),
            excitation_window_mins: self.params.excitation_window() / 60.0,
            peak_hour,
            peak_circadian_multiplier: peak_mult,
            total_observations: self.total_observations,
        }
    }
}

/// A prediction for when an event will next occur.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventPrediction {
    /// Predicted event time (unix seconds).
    pub predicted_time: f64,
    /// Time of peak intensity within the prediction window.
    pub peak_intensity_time: f64,
    /// Peak intensity value.
    pub peak_intensity: f64,
    /// Confidence in the prediction [0.0, 1.0].
    pub confidence: f64,
    /// Seconds until the predicted event.
    pub time_until: f64,
}

/// Summary statistics for an event type model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSummary {
    pub label: String,
    pub base_rate_per_hour: f64,
    pub stationary_rate_per_hour: f64,
    pub branching_ratio: f64,
    pub excitation_window_mins: f64,
    pub peak_hour: usize,
    pub peak_circadian_multiplier: f64,
    pub total_observations: u64,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  Multi-Event Registry
// ══════════════════════════════════════════════════════════════════════════════

/// Registry of Hawkes process models for multiple event types.
///
/// Maintains one `EventTypeModel` per observed event label.
/// Serializable for persistence via the meta table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HawkesRegistry {
    /// Models keyed by event label.
    pub models: HashMap<String, EventTypeModel>,
    /// Configuration for new models.
    pub config: HawkesRegistryConfig,
}

/// Configuration for the Hawkes registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HawkesRegistryConfig {
    /// Maximum number of event types to track.
    /// Least-recently-observed models are evicted when exceeded.
    pub max_event_types: usize,
    /// Minimum observations before a model's predictions are trusted.
    pub min_observations_for_prediction: u64,
    /// Prediction horizon (seconds).
    pub prediction_horizon_secs: f64,
    /// Prediction step size (seconds).
    pub prediction_step_secs: f64,
    /// Threshold multiplier for anticipation (e.g., 1.5 = 50% above average).
    pub anticipation_threshold: f64,
}

impl Default for HawkesRegistryConfig {
    fn default() -> Self {
        Self {
            max_event_types: 100,
            min_observations_for_prediction: 5,
            prediction_horizon_secs: 3600.0 * 4.0, // 4 hours
            prediction_step_secs: 60.0,            // 1-minute steps
            anticipation_threshold: 1.5,
        }
    }
}

impl HawkesRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            config: HawkesRegistryConfig::default(),
        }
    }

    /// Create a registry with custom config.
    pub fn with_config(config: HawkesRegistryConfig) -> Self {
        Self {
            models: HashMap::new(),
            config,
        }
    }

    /// Record an event observation, creating or updating the model.
    pub fn observe(&mut self, label: &str, timestamp: f64) {
        if let Some(model) = self.models.get_mut(label) {
            model.observe(timestamp);
        } else {
            // Evict least-recently-observed if at capacity
            if self.models.len() >= self.config.max_event_types {
                self.evict_oldest();
            }
            let mut model = EventTypeModel::new(label);
            model.observe(timestamp);
            self.models.insert(label.to_string(), model);
        }
    }

    /// Batch-observe historical events for an event type.
    pub fn observe_batch(&mut self, label: &str, timestamps: &[f64]) {
        if timestamps.is_empty() {
            return;
        }
        // Evict if needed
        if !self.models.contains_key(label) && self.models.len() >= self.config.max_event_types {
            self.evict_oldest();
        }
        let model = EventTypeModel::from_observations(label, timestamps);
        self.models.insert(label.to_string(), model);
    }

    /// Get predictions for all event types that should be anticipated now.
    pub fn anticipate_all(&self, now: f64) -> Vec<AnticipatedEvent> {
        let mut anticipated = Vec::new();

        for (label, model) in &self.models {
            if model.total_observations < self.config.min_observations_for_prediction {
                continue;
            }

            if model.should_anticipate(now, self.config.anticipation_threshold) {
                if let Some(pred) = model.predict_next(
                    now,
                    self.config.prediction_horizon_secs,
                    self.config.prediction_step_secs,
                ) {
                    anticipated.push(AnticipatedEvent {
                        label: label.clone(),
                        prediction: pred,
                        current_intensity: model.intensity(now),
                        stationary_rate: model.params.stationary_rate(),
                    });
                }
            }
        }

        // Sort by confidence descending
        anticipated.sort_by(|a, b| {
            b.prediction
                .confidence
                .partial_cmp(&a.prediction.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        anticipated
    }

    /// Predict next occurrence for a specific event type.
    pub fn predict(&self, label: &str, now: f64) -> Option<EventPrediction> {
        let model = self.models.get(label)?;
        if model.total_observations < self.config.min_observations_for_prediction {
            return None;
        }
        model.predict_next(
            now,
            self.config.prediction_horizon_secs,
            self.config.prediction_step_secs,
        )
    }

    /// Get summaries for all tracked event types.
    pub fn summaries(&self) -> Vec<ModelSummary> {
        let mut summaries: Vec<ModelSummary> = self.models.values().map(|m| m.summary()).collect();
        summaries.sort_by(|a, b| b.total_observations.cmp(&a.total_observations));
        summaries
    }

    /// Number of tracked event types.
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Evict the least-recently-observed model.
    fn evict_oldest(&mut self) {
        let oldest = self
            .models
            .iter()
            .min_by(|(_, a), (_, b)| {
                let last_a = a.recent_events.last().copied().unwrap_or(0.0);
                let last_b = b.recent_events.last().copied().unwrap_or(0.0);
                last_a
                    .partial_cmp(&last_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, _)| k.clone());

        if let Some(key) = oldest {
            self.models.remove(&key);
        }
    }
}

impl Default for HawkesRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// An event type that the system anticipates happening soon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnticipatedEvent {
    /// Event type label.
    pub label: String,
    /// Prediction details.
    pub prediction: EventPrediction,
    /// Current intensity λ(now).
    pub current_intensity: f64,
    /// Stationary (average) rate for comparison.
    pub stationary_rate: f64,
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── HawkesParams ──

    #[test]
    fn test_params_stability() {
        let p = HawkesParams::new(0.001, 0.002, 0.01);
        assert!(p.branching_ratio() < 1.0);
        assert!(p.stationary_rate() > p.mu);
        assert!(p.excitation_window() > 0.0);
    }

    #[test]
    fn test_params_from_rate() {
        let p = HawkesParams::from_rate(0.001); // 1 event per 1000s
        assert!(p.mu > 0.0);
        assert!(p.branching_ratio() < 1.0);
    }

    #[test]
    fn test_stationary_rate() {
        let p = HawkesParams::new(0.001, 0.005, 0.01);
        // Branching ratio = 0.5, stationary rate = μ/(1-0.5) = 2μ
        let expected = 0.001 / (1.0 - 0.5);
        assert!((p.stationary_rate() - expected).abs() < 1e-10);
    }

    // ── CircadianProfile ──

    #[test]
    fn test_flat_profile() {
        let profile = CircadianProfile::flat();
        assert!((profile.multiplier_at(0.0) - 1.0).abs() < 1e-10);
        assert!((profile.multiplier_at(43200.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_circadian_from_histogram() {
        let mut hist = super::super::temporal::SeasonalHistogram::hour_of_day();
        // All events at 9am
        for _ in 0..100 {
            hist.add(9);
        }
        let profile = CircadianProfile::from_histogram(&hist);
        // 9am should have the highest multiplier
        let mut peak_hour = 0;
        let mut peak_val = 0.0;
        for (i, &m) in profile.hourly_multipliers.iter().enumerate() {
            if m > peak_val {
                peak_val = m;
                peak_hour = i;
            }
        }
        assert_eq!(peak_hour, 9);
        assert!(peak_val > 1.0, "Peak should be above average: {peak_val}");
    }

    #[test]
    fn test_circadian_mean_normalized() {
        let mut hist = super::super::temporal::SeasonalHistogram::hour_of_day();
        for _ in 0..50 {
            hist.add(9);
        }
        for _ in 0..30 {
            hist.add(14);
        }
        for _ in 0..20 {
            hist.add(21);
        }
        let profile = CircadianProfile::from_histogram(&hist);
        let mean: f64 = profile.hourly_multipliers.iter().sum::<f64>() / 24.0;
        assert!(
            (mean - 1.0).abs() < 0.01,
            "Mean multiplier should be ~1.0: {mean}"
        );
    }

    // ── EventTypeModel ──

    #[test]
    fn test_model_intensity_base() {
        let model = EventTypeModel::new("test");
        // No recent events → intensity = base rate × circadian
        let lambda = model.intensity(1000.0);
        assert!(lambda > 0.0);
        assert!((lambda - model.params.mu).abs() < 0.01); // Flat circadian ≈ 1.0
    }

    #[test]
    fn test_model_intensity_after_event() {
        let mut model = EventTypeModel::new("test");
        model.params = HawkesParams::new(0.001, 0.005, 0.01);
        model.recent_events.push(1000.0);
        model.total_observations = 1;

        // Just after event: intensity should be boosted
        let lambda_just_after = model.intensity(1001.0);
        let lambda_later = model.intensity(2000.0);

        assert!(
            lambda_just_after > lambda_later,
            "Intensity should decay after event: {lambda_just_after} vs {lambda_later}"
        );
    }

    #[test]
    fn test_model_excitation_decay() {
        let mut model = EventTypeModel::new("test");
        model.params = HawkesParams::new(0.001, 0.005, 0.01);
        model.recent_events.push(0.0);
        model.total_observations = 1;

        // Intensity should decay toward base rate
        let lambda_0 = model.intensity(1.0);
        let lambda_100 = model.intensity(100.0);
        let lambda_1000 = model.intensity(1000.0);

        assert!(lambda_0 > lambda_100);
        assert!(lambda_100 > lambda_1000);
        // At t=1000, excitation should be nearly gone
        let base = model.params.mu; // Circadian ≈ 1.0
        assert!(
            (lambda_1000 - base).abs() < 0.001,
            "Should converge to base: {lambda_1000} vs {base}"
        );
    }

    #[test]
    fn test_model_from_observations() {
        // Simulate daily events for 2 weeks at a consistent hour
        let daily = 86400.0;
        // Use a base that is exactly midnight UTC, then add 9h offset
        let base = 1_700_006_400.0; // 2023-11-15 00:00:00 UTC
        let timestamps: Vec<f64> = (0..14)
            .map(|d| base + d as f64 * daily + 32400.0) // +9h = 9am UTC
            .collect();

        // Verify the hour
        let hour = super::super::temporal::hour_of_day_utc(timestamps[0]);

        let model = EventTypeModel::from_observations("daily_check", &timestamps);
        assert_eq!(model.total_observations, 14);
        assert!(model.params.mu > 0.0);
        assert!(model.params.branching_ratio() < 1.0);

        // Circadian should peak at the computed hour
        let summary = model.summary();
        assert_eq!(summary.peak_hour, hour);
    }

    #[test]
    fn test_model_observe_updates() {
        let mut model = EventTypeModel::new("test");
        for i in 0..20 {
            model.observe(1000.0 + i as f64 * 100.0);
        }
        assert_eq!(model.total_observations, 20);
        assert!(!model.recent_events.is_empty());
    }

    #[test]
    fn test_model_predict_next() {
        // Create a model with regular events
        let timestamps: Vec<f64> = (0..50)
            .map(|i| 1_000_000.0 + i as f64 * 600.0) // Every 10 minutes
            .collect();
        let model = EventTypeModel::from_observations("frequent", &timestamps);

        let now = *timestamps.last().unwrap() + 60.0;
        let pred = model.predict_next(now, 3600.0, 30.0);
        assert!(pred.is_some(), "Should predict a next event");

        let pred = pred.unwrap();
        assert!(pred.time_until > 0.0);
        assert!(pred.confidence > 0.0);
    }

    #[test]
    fn test_model_should_anticipate() {
        let mut model = EventTypeModel::new("test");
        model.params = HawkesParams::new(0.001, 0.005, 0.01);
        // Add a recent burst of events
        for i in 0..5 {
            model.recent_events.push(1000.0 + i as f64 * 10.0);
        }
        model.total_observations = 5;

        // Right after the burst, intensity should be elevated
        let should = model.should_anticipate(1060.0, 1.5);
        // With 5 recent events within 40s, excitation should be high
        assert!(should, "Should anticipate after burst");
    }

    #[test]
    fn test_intensity_curve() {
        let mut model = EventTypeModel::new("test");
        model.params = HawkesParams::new(0.001, 0.005, 0.01);
        model.recent_events.push(1000.0);
        model.total_observations = 1;

        let curve = model.intensity_curve(1000.0, 2000.0, 100.0);
        assert!(!curve.is_empty());
        // Should be decreasing (excitation decaying)
        let first_lambda = curve[1].1; // Skip t=1000 (event time)
        let last_lambda = curve.last().unwrap().1;
        assert!(first_lambda >= last_lambda);
    }

    // ── HawkesRegistry ──

    #[test]
    fn test_registry_observe() {
        let mut registry = HawkesRegistry::new();
        registry.observe("email_check", 1000.0);
        registry.observe("email_check", 1600.0);
        registry.observe("terminal_open", 1200.0);

        assert_eq!(registry.model_count(), 2);
        assert_eq!(
            registry
                .models
                .get("email_check")
                .unwrap()
                .total_observations,
            2
        );
    }

    #[test]
    fn test_registry_batch_observe() {
        let mut registry = HawkesRegistry::new();
        let timestamps: Vec<f64> = (0..30).map(|i| 1_000_000.0 + i as f64 * 3600.0).collect();
        registry.observe_batch("hourly_task", &timestamps);

        assert_eq!(registry.model_count(), 1);
        let model = registry.models.get("hourly_task").unwrap();
        assert_eq!(model.total_observations, 30);
    }

    #[test]
    fn test_registry_eviction() {
        let config = HawkesRegistryConfig {
            max_event_types: 3,
            ..Default::default()
        };
        let mut registry = HawkesRegistry::with_config(config);
        registry.observe("a", 100.0);
        registry.observe("b", 200.0);
        registry.observe("c", 300.0);
        assert_eq!(registry.model_count(), 3);

        // Adding a 4th should evict the oldest
        registry.observe("d", 400.0);
        assert_eq!(registry.model_count(), 3);
        assert!(!registry.models.contains_key("a")); // "a" was oldest
    }

    #[test]
    fn test_registry_predict() {
        let mut registry = HawkesRegistry::new();
        // Need enough observations for prediction
        let timestamps: Vec<f64> = (0..20).map(|i| 1_000_000.0 + i as f64 * 600.0).collect();
        registry.observe_batch("check", &timestamps);

        let now = *timestamps.last().unwrap() + 60.0;
        let pred = registry.predict("check", now);
        assert!(pred.is_some());
    }

    #[test]
    fn test_registry_anticipate_all() {
        let mut registry = HawkesRegistry::new();

        // Create a frequent event type with burst
        for i in 0..10 {
            registry.observe("active_event", 1_000_000.0 + i as f64 * 60.0);
        }

        // Create an infrequent event type
        for i in 0..6 {
            registry.observe("rare_event", 1_000_000.0 + i as f64 * 86400.0);
        }

        let now = 1_000_000.0 + 660.0; // Just after the active_event burst
        let anticipated = registry.anticipate_all(now);
        // active_event should be anticipated due to recent burst
        // (may or may not depending on exact intensity vs threshold)
        // At minimum, the function should not panic
        for ae in &anticipated {
            assert!(ae.prediction.confidence >= 0.0);
            assert!(ae.prediction.confidence <= 1.0);
        }
    }

    #[test]
    fn test_registry_summaries() {
        let mut registry = HawkesRegistry::new();
        registry.observe("email", 1000.0);
        registry.observe("calendar", 2000.0);

        let summaries = registry.summaries();
        assert_eq!(summaries.len(), 2);
        for s in &summaries {
            assert!(s.base_rate_per_hour > 0.0);
            assert!(s.branching_ratio < 1.0);
        }
    }

    // ── Parameter Learning ──

    #[test]
    fn test_refit_converges_to_true_rate() {
        // Generate events at ~1 per 300s (12/hour)
        let mut model = EventTypeModel::new("regular");
        let interval = 300.0;
        for i in 0..100 {
            model.observe(1_000_000.0 + i as f64 * interval);
        }

        // After 100 observations, μ should be close to 1/300
        let expected_mu = 1.0 / interval;
        let ratio = model.params.mu / expected_mu;
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "μ should be within 2× of true rate: got {}, expected {}",
            model.params.mu,
            expected_mu
        );
    }

    #[test]
    fn test_clustered_events_higher_branching() {
        // Clustered events: bursts of 5 with long gaps
        let mut timestamps = Vec::new();
        for burst in 0..10 {
            let base = 1_000_000.0 + burst as f64 * 3600.0;
            for j in 0..5 {
                timestamps.push(base + j as f64 * 10.0); // 10s apart within burst
            }
        }
        let clustered = EventTypeModel::from_observations("clustered", &timestamps);

        // Regular events: evenly spaced
        let regular_ts: Vec<f64> = (0..50).map(|i| 1_000_000.0 + i as f64 * 720.0).collect();
        let regular = EventTypeModel::from_observations("regular", &regular_ts);

        // Clustered should have higher branching ratio
        assert!(
            clustered.params.branching_ratio() > regular.params.branching_ratio(),
            "Clustered should have higher BR: {} vs {}",
            clustered.params.branching_ratio(),
            regular.params.branching_ratio()
        );
    }
}
