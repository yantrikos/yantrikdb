//! CK-2.4: User Receptivity / Interruptibility Model
//!
//! Predicts WHEN the user is open to suggestions — a smart companion knows
//! when NOT to act. This module answers the question: "If I interrupt the user
//! right now with this action, what is the probability they'll welcome it?"
//!
//! ## Architecture
//!
//! 1. **Feature Vector** — 10-dimensional signal space capturing temporal,
//!    behavioral, and contextual state.
//!
//! 2. **Logistic Regression** — Online-learned classifier:
//!    `P(receptive | x) = σ(w · x + b)`
//!    Updated from accept/dismiss/ignore feedback.
//!
//! 3. **Interruption Cost** — Activity-aware cost estimate that modulates
//!    the utility score from CK-1.8.
//!
//! 4. **Attention Budget** — Session-level fatigue tracking that limits
//!    how many suggestions per session.
//!
//! 5. **Quiet Hours** — Time-of-day windows where interruptions are blocked.
//!
//! ## Integration Points
//!
//! - `estimate_receptivity()` feeds into `PolicyConfig` decision threshold
//! - `interruption_cost()` feeds into `EvaluatorConfig` cost penalty
//! - `attention_budget_remaining()` gates `suggest_next_step()` output

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ══════════════════════════════════════════════════════════════════════════════
// § 1  Feature Space
// ══════════════════════════════════════════════════════════════════════════════

/// The 10 features used by the receptivity model.
///
/// All features are normalized to approximately [0, 1] (or [-1, 1] for valence)
/// before being fed to the logistic regression.
pub const FEATURE_COUNT: usize = 10;

/// Named indices into the feature vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(usize)]
pub enum FeatureIndex {
    /// Circadian signal: time-of-day receptivity from learned pattern [0, 1].
    CircadianReceptivity = 0,
    /// Current activity state encoded as a cost level [0, 1].
    /// 0 = idle, 0.5 = light activity, 1.0 = deep focus.
    ActivityLevel = 1,
    /// Interaction frequency in the last 15 minutes, normalized [0, 1].
    /// High = user is engaged and present.
    InteractionFrequency = 2,
    /// Recent dismissal rate over last 10 suggestions [0, 1].
    /// High = user is rejecting suggestions (not receptive).
    DismissalRate = 3,
    /// Time since last interaction, sigmoid-normalized [0, 1].
    /// 0 = just interacted, 1 = long idle (might have left).
    IdleDuration = 4,
    /// Session duration fatigue factor [0, 1].
    /// 0 = fresh session, 1 = extended session (fatigued).
    SessionFatigue = 5,
    /// Day-of-week signal [0, 1].
    /// Learned from historical accept patterns per day.
    DayOfWeekReceptivity = 6,
    /// Emotional valence estimate [-1, 1] → mapped to [0, 1].
    /// Negative = stressed/frustrated, positive = calm/happy.
    EmotionalValence = 7,
    /// Suggestions accepted in current session / budget [0, 1].
    /// High = attention budget nearing depletion.
    BudgetUtilization = 8,
    /// Notification mode: 0 = all, 0.5 = important only, 1.0 = do not disturb.
    NotificationMode = 9,
}

impl FeatureIndex {
    /// All feature indices in order.
    pub const ALL: [FeatureIndex; FEATURE_COUNT] = [
        Self::CircadianReceptivity,
        Self::ActivityLevel,
        Self::InteractionFrequency,
        Self::DismissalRate,
        Self::IdleDuration,
        Self::SessionFatigue,
        Self::DayOfWeekReceptivity,
        Self::EmotionalValence,
        Self::BudgetUtilization,
        Self::NotificationMode,
    ];

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::CircadianReceptivity => "circadian_receptivity",
            Self::ActivityLevel => "activity_level",
            Self::InteractionFrequency => "interaction_frequency",
            Self::DismissalRate => "dismissal_rate",
            Self::IdleDuration => "idle_duration",
            Self::SessionFatigue => "session_fatigue",
            Self::DayOfWeekReceptivity => "day_of_week_receptivity",
            Self::EmotionalValence => "emotional_valence",
            Self::BudgetUtilization => "budget_utilization",
            Self::NotificationMode => "notification_mode",
        }
    }
}

/// A receptivity feature vector.
pub type FeatureVector = [f64; FEATURE_COUNT];

/// Create a default feature vector (neutral/idle state).
pub fn default_features() -> FeatureVector {
    [
        0.5, // Circadian: average
        0.0, // Activity: idle
        0.0, // Interaction: none
        0.0, // Dismissal: none
        0.5, // Idle: moderate
        0.0, // Fatigue: fresh
        0.5, // Day: average
        0.5, // Emotion: neutral (mapped from 0.0)
        0.0, // Budget: unused
        0.0, // Notification: all allowed
    ]
}

// ══════════════════════════════════════════════════════════════════════════════
// § 2  Activity State
// ══════════════════════════════════════════════════════════════════════════════

/// The user's current activity state, used to estimate interruption cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActivityState {
    /// User is idle — screen on but no input for > 2 minutes.
    Idle,
    /// User just returned — opened device or switched to app.
    JustReturned,
    /// Light browsing — switching between apps, reading.
    Browsing,
    /// Active communication — typing messages, on a call.
    Communicating,
    /// Task switching — between activities, natural interrupt point.
    TaskSwitching,
    /// Focused work — sustained typing, coding, writing.
    FocusedWork,
    /// Deep focus — extended uninterrupted work (> 15 minutes).
    DeepFocus,
}

impl ActivityState {
    /// Interruption cost for this activity state [0.0, 1.0].
    /// Higher cost = worse time to interrupt.
    pub fn interruption_cost(self) -> f64 {
        match self {
            Self::Idle => 0.05,
            Self::JustReturned => 0.10,
            Self::TaskSwitching => 0.15,
            Self::Browsing => 0.30,
            Self::Communicating => 0.55,
            Self::FocusedWork => 0.75,
            Self::DeepFocus => 0.95,
        }
    }

    /// Activity level for the feature vector [0.0, 1.0].
    pub fn activity_level(self) -> f64 {
        self.interruption_cost()
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::JustReturned => "just_returned",
            Self::Browsing => "browsing",
            Self::Communicating => "communicating",
            Self::TaskSwitching => "task_switching",
            Self::FocusedWork => "focused_work",
            Self::DeepFocus => "deep_focus",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "idle" => Self::Idle,
            "just_returned" => Self::JustReturned,
            "browsing" => Self::Browsing,
            "communicating" => Self::Communicating,
            "task_switching" => Self::TaskSwitching,
            "focused_work" => Self::FocusedWork,
            "deep_focus" => Self::DeepFocus,
            _ => Self::Idle,
        }
    }
}

/// Notification mode — user's current notification preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotificationMode {
    /// All notifications allowed.
    All,
    /// Only important notifications.
    ImportantOnly,
    /// Do not disturb — block everything.
    DoNotDisturb,
}

impl NotificationMode {
    /// Feature value for this mode [0.0, 1.0].
    pub fn feature_value(self) -> f64 {
        match self {
            Self::All => 0.0,
            Self::ImportantOnly => 0.5,
            Self::DoNotDisturb => 1.0,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  Context Snapshot
// ══════════════════════════════════════════════════════════════════════════════

/// A snapshot of the user's current context, used to build the feature vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSnapshot {
    /// Current time (unix seconds).
    pub now: f64,
    /// Current activity state.
    pub activity: ActivityState,
    /// Number of interactions in the last 15 minutes.
    pub recent_interactions_15min: u32,
    /// Recent suggestion outcomes: (accepted, dismissed, ignored).
    pub recent_outcomes: (u32, u32, u32),
    /// Seconds since the last user interaction.
    pub secs_since_last_interaction: f64,
    /// Seconds since the session started.
    pub session_duration_secs: f64,
    /// Emotional valence estimate [-1.0, 1.0].
    pub emotional_valence: f64,
    /// Number of suggestions accepted this session.
    pub session_suggestions_accepted: u32,
    /// Suggestion budget for this session.
    pub session_suggestion_budget: u32,
    /// Current notification mode.
    pub notification_mode: NotificationMode,
}

impl ContextSnapshot {
    /// Extract the feature vector from this context snapshot.
    pub fn to_features(&self, model: &ReceptivityModel) -> FeatureVector {
        let mut features = default_features();

        // F0: Circadian receptivity from learned hour-of-day pattern
        let hour = super::temporal::hour_of_day_utc(self.now);
        features[FeatureIndex::CircadianReceptivity as usize] = model.circadian_receptivity[hour];

        // F1: Activity level
        features[FeatureIndex::ActivityLevel as usize] = self.activity.activity_level();

        // F2: Interaction frequency (normalize: 20+ interactions = 1.0)
        features[FeatureIndex::InteractionFrequency as usize] =
            (self.recent_interactions_15min as f64 / 20.0).clamp(0.0, 1.0);

        // F3: Dismissal rate
        let total_outcomes =
            self.recent_outcomes.0 + self.recent_outcomes.1 + self.recent_outcomes.2;
        features[FeatureIndex::DismissalRate as usize] = if total_outcomes > 0 {
            self.recent_outcomes.1 as f64 / total_outcomes as f64
        } else {
            0.0
        };

        // F4: Idle duration (sigmoid: 0 at 0s, ~0.5 at 5min, ~1.0 at 30min)
        features[FeatureIndex::IdleDuration as usize] =
            1.0 / (1.0 + (-0.005 * (self.secs_since_last_interaction - 300.0)).exp());

        // F5: Session fatigue (sigmoid: 0 at start, ~0.5 at 2h, ~1.0 at 6h)
        features[FeatureIndex::SessionFatigue as usize] =
            1.0 / (1.0 + (-0.0003 * (self.session_duration_secs - 7200.0)).exp());

        // F6: Day-of-week receptivity
        let dow = super::temporal::day_of_week_utc(self.now);
        features[FeatureIndex::DayOfWeekReceptivity as usize] = model.dow_receptivity[dow];

        // F7: Emotional valence → [0, 1] (from [-1, 1])
        features[FeatureIndex::EmotionalValence as usize] = (self.emotional_valence + 1.0) / 2.0;

        // F8: Budget utilization
        features[FeatureIndex::BudgetUtilization as usize] = if self.session_suggestion_budget > 0 {
            self.session_suggestions_accepted as f64 / self.session_suggestion_budget as f64
        } else {
            0.0
        };

        // F9: Notification mode
        features[FeatureIndex::NotificationMode as usize] = self.notification_mode.feature_value();

        features
    }
}

impl Default for ContextSnapshot {
    fn default() -> Self {
        Self {
            now: 0.0,
            activity: ActivityState::Idle,
            recent_interactions_15min: 0,
            recent_outcomes: (0, 0, 0),
            secs_since_last_interaction: 0.0,
            session_duration_secs: 0.0,
            emotional_valence: 0.0,
            session_suggestions_accepted: 0,
            session_suggestion_budget: 20,
            notification_mode: NotificationMode::All,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  Logistic Regression Model
// ══════════════════════════════════════════════════════════════════════════════

/// Online logistic regression model for receptivity prediction.
///
/// Learns from user feedback (accept/dismiss/ignore) and adapts to
/// individual patterns over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceptivityModel {
    /// Weight vector for the 10 features.
    pub weights: [f64; FEATURE_COUNT],
    /// Bias term.
    pub bias: f64,
    /// Learning rate for online updates.
    pub learning_rate: f64,
    /// L2 regularization strength (prevents overfitting).
    pub l2_lambda: f64,
    /// Total training examples seen.
    pub training_count: u64,
    /// Hour-of-day receptivity pattern [0..23], learned from feedback.
    pub circadian_receptivity: [f64; 24],
    /// Day-of-week receptivity pattern [0..6] (Mon=0).
    pub dow_receptivity: [f64; 7],
    /// Quiet hours configuration.
    pub quiet_hours: QuietHoursConfig,
    /// Attention budget configuration.
    pub attention_budget: AttentionBudgetConfig,
}

impl ReceptivityModel {
    /// Create a new model with sensible default weights.
    ///
    /// Default weights encode common-sense priors:
    /// - Activity level has strong negative weight (busy = less receptive)
    /// - Dismissal rate has strong negative weight
    /// - Interaction frequency has positive weight (engaged = receptive)
    /// - Circadian/DOW are neutral until learned
    pub fn new() -> Self {
        let mut weights = [0.0; FEATURE_COUNT];
        weights[FeatureIndex::CircadianReceptivity as usize] = 1.5; // Positive: good time → receptive
        weights[FeatureIndex::ActivityLevel as usize] = -3.0; // Negative: busy → not receptive
        weights[FeatureIndex::InteractionFrequency as usize] = 2.0; // Positive: engaged → receptive
        weights[FeatureIndex::DismissalRate as usize] = -4.0; // Strong negative: dismissing → not receptive
        weights[FeatureIndex::IdleDuration as usize] = -1.0; // Mildly negative: too idle → might have left
        weights[FeatureIndex::SessionFatigue as usize] = -1.5; // Negative: fatigued → less receptive
        weights[FeatureIndex::DayOfWeekReceptivity as usize] = 1.0; // Positive: good day → receptive
        weights[FeatureIndex::EmotionalValence as usize] = 1.5; // Positive: good mood → receptive
        weights[FeatureIndex::BudgetUtilization as usize] = -2.5; // Negative: budget depleted → stop
        weights[FeatureIndex::NotificationMode as usize] = -5.0; // Strong negative: DND → don't interrupt

        Self {
            weights,
            bias: 0.5, // Slight positive bias (lean toward engaging)
            learning_rate: 0.05,
            l2_lambda: 0.001,
            training_count: 0,
            circadian_receptivity: [0.5; 24], // Neutral until learned
            dow_receptivity: [0.5; 7],        // Neutral until learned
            quiet_hours: QuietHoursConfig::default(),
            attention_budget: AttentionBudgetConfig::default(),
        }
    }

    /// Compute the sigmoid function.
    #[inline]
    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    /// Compute receptivity probability from a feature vector.
    ///
    /// Returns `P(receptive | features)` in [0.0, 1.0].
    pub fn predict(&self, features: &FeatureVector) -> f64 {
        let z: f64 = self
            .weights
            .iter()
            .zip(features.iter())
            .map(|(&w, &x)| w * x)
            .sum::<f64>()
            + self.bias;
        Self::sigmoid(z)
    }

    /// Full receptivity estimate with factor decomposition.
    pub fn estimate(&self, context: &ContextSnapshot) -> ReceptivityEstimate {
        // Check quiet hours first
        if self.quiet_hours.is_quiet(context.now) {
            return ReceptivityEstimate {
                score: 0.0,
                factors: vec![ReceptivityFactor {
                    name: "quiet_hours".to_string(),
                    value: 1.0,
                    contribution: -1.0,
                    description: "Quiet hours active — all interruptions blocked".to_string(),
                }],
                is_quiet_hours: true,
                budget_remaining: 0,
            };
        }

        // Check DND mode
        if context.notification_mode == NotificationMode::DoNotDisturb {
            return ReceptivityEstimate {
                score: 0.0,
                factors: vec![ReceptivityFactor {
                    name: "do_not_disturb".to_string(),
                    value: 1.0,
                    contribution: -1.0,
                    description: "Do Not Disturb mode active".to_string(),
                }],
                is_quiet_hours: false,
                budget_remaining: 0,
            };
        }

        let features = context.to_features(self);
        let score = self.predict(&features);

        // Decompose into factors
        let factors: Vec<ReceptivityFactor> = FeatureIndex::ALL
            .iter()
            .map(|&idx| {
                let i = idx as usize;
                let contribution = self.weights[i] * features[i];
                ReceptivityFactor {
                    name: idx.name().to_string(),
                    value: features[i],
                    contribution,
                    description: factor_description(idx, features[i]),
                }
            })
            .collect();

        let budget_remaining = self
            .attention_budget
            .remaining(context.session_suggestions_accepted);

        ReceptivityEstimate {
            score,
            factors,
            is_quiet_hours: false,
            budget_remaining,
        }
    }

    /// Update the model from user feedback.
    ///
    /// `outcome` — What the user did with the suggestion.
    /// `features` — Feature vector at the time of the suggestion.
    pub fn learn(&mut self, features: &FeatureVector, outcome: SuggestionOutcome) {
        let target = match outcome {
            SuggestionOutcome::Accepted => 1.0,
            SuggestionOutcome::Dismissed => 0.0,
            SuggestionOutcome::Ignored => 0.3, // Partial negative signal
        };

        let prediction = self.predict(features);
        let error = target - prediction;

        // Gradient descent update with L2 regularization
        for i in 0..FEATURE_COUNT {
            let gradient = error * features[i] - self.l2_lambda * self.weights[i];
            self.weights[i] += self.learning_rate * gradient;
        }
        self.bias += self.learning_rate * error;

        self.training_count += 1;

        // Decay learning rate slightly over time (minimum 0.001)
        if self.training_count % 100 == 0 {
            self.learning_rate = (self.learning_rate * 0.95).max(0.001);
        }
    }

    /// Update circadian and day-of-week patterns from feedback.
    pub fn learn_temporal_pattern(&mut self, timestamp: f64, outcome: SuggestionOutcome) {
        let hour = super::temporal::hour_of_day_utc(timestamp);
        let dow = super::temporal::day_of_week_utc(timestamp);
        let lr = 0.05;

        let target = match outcome {
            SuggestionOutcome::Accepted => 0.8,
            SuggestionOutcome::Dismissed => 0.2,
            SuggestionOutcome::Ignored => 0.4,
        };

        // Exponential moving average update
        self.circadian_receptivity[hour] += lr * (target - self.circadian_receptivity[hour]);
        self.dow_receptivity[dow] += lr * (target - self.dow_receptivity[dow]);
    }

    /// Combined learn: update both the weights and the temporal patterns.
    pub fn observe_outcome(&mut self, context: &ContextSnapshot, outcome: SuggestionOutcome) {
        let features = context.to_features(self);
        self.learn(&features, outcome);
        self.learn_temporal_pattern(context.now, outcome);
    }
}

impl Default for ReceptivityModel {
    fn default() -> Self {
        Self::new()
    }
}

/// What the user did with a suggestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionOutcome {
    /// User accepted/acted on the suggestion.
    Accepted,
    /// User explicitly dismissed the suggestion.
    Dismissed,
    /// Suggestion expired without user action.
    Ignored,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 5  Receptivity Estimate
// ══════════════════════════════════════════════════════════════════════════════

/// Full receptivity estimate with explainable factors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceptivityEstimate {
    /// Overall receptivity score [0.0, 1.0].
    /// 0.0 = do not interrupt, 1.0 = highly receptive.
    pub score: f64,
    /// Factor decomposition showing what contributed to the score.
    pub factors: Vec<ReceptivityFactor>,
    /// Whether quiet hours are currently active.
    pub is_quiet_hours: bool,
    /// Remaining suggestions in the attention budget.
    pub budget_remaining: u32,
}

impl ReceptivityEstimate {
    /// Whether the user is likely receptive (score > threshold).
    pub fn is_receptive(&self, threshold: f64) -> bool {
        self.score >= threshold && !self.is_quiet_hours && self.budget_remaining > 0
    }

    /// The single most negative factor (biggest reason not to interrupt).
    pub fn top_blocker(&self) -> Option<&ReceptivityFactor> {
        self.factors.iter().min_by(|a, b| {
            a.contribution
                .partial_cmp(&b.contribution)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// The single most positive factor (biggest reason to interrupt).
    pub fn top_enabler(&self) -> Option<&ReceptivityFactor> {
        self.factors.iter().max_by(|a, b| {
            a.contribution
                .partial_cmp(&b.contribution)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

/// A single factor contributing to the receptivity score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceptivityFactor {
    /// Feature name.
    pub name: String,
    /// Feature value [0, 1].
    pub value: f64,
    /// Contribution to the logit (weight × value).
    pub contribution: f64,
    /// Human-readable description.
    pub description: String,
}

/// Generate a human-readable description for a feature value.
fn factor_description(idx: FeatureIndex, value: f64) -> String {
    match idx {
        FeatureIndex::CircadianReceptivity => {
            if value > 0.7 {
                "Peak receptivity time of day".into()
            } else if value < 0.3 {
                "Low receptivity time of day".into()
            } else {
                "Average time of day".into()
            }
        }
        FeatureIndex::ActivityLevel => {
            if value > 0.7 {
                "User in deep focus — avoid interrupting".into()
            } else if value > 0.4 {
                "User moderately active".into()
            } else {
                "User idle or lightly active".into()
            }
        }
        FeatureIndex::InteractionFrequency => {
            if value > 0.5 {
                "User actively engaged".into()
            } else if value > 0.1 {
                "Some recent interaction".into()
            } else {
                "No recent interaction".into()
            }
        }
        FeatureIndex::DismissalRate => {
            if value > 0.5 {
                "High dismissal rate — user rejecting suggestions".into()
            } else if value > 0.2 {
                "Moderate dismissal rate".into()
            } else {
                "Low dismissal rate — user open to suggestions".into()
            }
        }
        FeatureIndex::IdleDuration => {
            if value > 0.7 {
                "User has been idle for a while — may have left".into()
            } else if value > 0.3 {
                "User recently active".into()
            } else {
                "User just interacted".into()
            }
        }
        FeatureIndex::SessionFatigue => {
            if value > 0.7 {
                "Extended session — user may be fatigued".into()
            } else if value > 0.3 {
                "Moderate session length".into()
            } else {
                "Fresh session".into()
            }
        }
        FeatureIndex::DayOfWeekReceptivity => {
            if value > 0.7 {
                "Historically receptive day".into()
            } else if value < 0.3 {
                "Historically unreceptive day".into()
            } else {
                "Average day".into()
            }
        }
        FeatureIndex::EmotionalValence => {
            if value > 0.7 {
                "Positive emotional state".into()
            } else if value < 0.3 {
                "Negative emotional state".into()
            } else {
                "Neutral emotional state".into()
            }
        }
        FeatureIndex::BudgetUtilization => {
            if value > 0.8 {
                "Attention budget nearly depleted".into()
            } else if value > 0.5 {
                "Attention budget half used".into()
            } else {
                "Attention budget available".into()
            }
        }
        FeatureIndex::NotificationMode => {
            if value > 0.7 {
                "Do Not Disturb mode".into()
            } else if value > 0.3 {
                "Important notifications only".into()
            } else {
                "All notifications allowed".into()
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 6  Quiet Hours
// ══════════════════════════════════════════════════════════════════════════════

/// Quiet hours configuration — time windows where interruptions are blocked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuietHoursConfig {
    /// Whether quiet hours are enabled.
    pub enabled: bool,
    /// Start hour (0-23, UTC). E.g., 22 for 10pm.
    pub start_hour: u8,
    /// End hour (0-23, UTC). E.g., 7 for 7am.
    pub end_hour: u8,
    /// Weekday mask: bit 0 = Monday, bit 6 = Sunday. 0x7F = every day.
    pub weekday_mask: u8,
    /// Whether to allow critical notifications during quiet hours.
    pub allow_critical: bool,
}

impl Default for QuietHoursConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            start_hour: 22,
            end_hour: 7,
            weekday_mask: 0x7F, // Every day
            allow_critical: true,
        }
    }
}

impl QuietHoursConfig {
    /// Whether the given timestamp falls within quiet hours.
    pub fn is_quiet(&self, timestamp: f64) -> bool {
        if !self.enabled {
            return false;
        }

        let hour = super::temporal::hour_of_day_utc(timestamp) as u8;
        let dow = super::temporal::day_of_week_utc(timestamp);

        // Check weekday mask
        if self.weekday_mask & (1 << dow) == 0 {
            return false; // Quiet hours not active on this day
        }

        // Handle wrap-around (e.g., 22:00 to 07:00)
        if self.start_hour > self.end_hour {
            // Wraps past midnight
            hour >= self.start_hour || hour < self.end_hour
        } else if self.start_hour < self.end_hour {
            hour >= self.start_hour && hour < self.end_hour
        } else {
            false // start == end: no quiet hours
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 7  Attention Budget
// ══════════════════════════════════════════════════════════════════════════════

/// Attention budget — limits how many suggestions per session.
///
/// Prevents suggestion fatigue by tracking and capping the number of
/// proactive interruptions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionBudgetConfig {
    /// Maximum suggestions per session.
    pub max_per_session: u32,
    /// Minimum interval between suggestions (seconds).
    pub min_interval_secs: f64,
    /// Budget recovery rate (suggestions per hour of idle time).
    pub recovery_rate_per_hour: f64,
}

impl Default for AttentionBudgetConfig {
    fn default() -> Self {
        Self {
            max_per_session: 20,
            min_interval_secs: 120.0, // At least 2 minutes between suggestions
            recovery_rate_per_hour: 5.0,
        }
    }
}

impl AttentionBudgetConfig {
    /// How many suggestions remain in the budget.
    pub fn remaining(&self, used: u32) -> u32 {
        self.max_per_session.saturating_sub(used)
    }

    /// Whether enough time has passed since the last suggestion.
    pub fn interval_ok(&self, secs_since_last_suggestion: f64) -> bool {
        secs_since_last_suggestion >= self.min_interval_secs
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 8  Interruption Cost Estimator
// ══════════════════════════════════════════════════════════════════════════════

/// Estimate the cost of interrupting the user with a specific action kind.
///
/// Combines the activity-based cost with action-specific urgency discounting.
/// High urgency actions get a cost reduction (urgent things are worth interrupting for).
///
/// # Arguments
/// * `activity` — Current user activity state.
/// * `action_urgency` — How urgent is the action [0.0, 1.0].
/// * `action_importance` — How important is the action [0.0, 1.0].
///
/// # Returns
/// Interruption cost in [0.0, 1.0]. Higher = worse time to interrupt.
pub fn estimate_interruption_cost(
    activity: ActivityState,
    action_urgency: f64,
    action_importance: f64,
) -> f64 {
    let base_cost = activity.interruption_cost();

    // Urgency discount: very urgent actions reduce the perceived cost
    // At urgency=1.0: cost reduced by 50%
    // At urgency=0.0: no reduction
    let urgency_discount = 1.0 - 0.5 * action_urgency;

    // Importance discount: very important actions also reduce cost
    // At importance=1.0: cost reduced by 30%
    let importance_discount = 1.0 - 0.3 * action_importance;

    let adjusted = base_cost * urgency_discount * importance_discount;
    adjusted.clamp(0.0, 1.0)
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Feature Space ──

    #[test]
    fn test_default_features_bounds() {
        let features = default_features();
        for (i, &f) in features.iter().enumerate() {
            assert!(f >= 0.0 && f <= 1.0, "Feature {i} out of bounds: {f}");
        }
    }

    #[test]
    fn test_feature_index_count() {
        assert_eq!(FeatureIndex::ALL.len(), FEATURE_COUNT);
    }

    // ── Activity State ──

    #[test]
    fn test_activity_cost_ordering() {
        // Cost should increase from idle to deep focus
        let states = [
            ActivityState::Idle,
            ActivityState::JustReturned,
            ActivityState::TaskSwitching,
            ActivityState::Browsing,
            ActivityState::Communicating,
            ActivityState::FocusedWork,
            ActivityState::DeepFocus,
        ];
        for w in states.windows(2) {
            assert!(
                w[0].interruption_cost() <= w[1].interruption_cost(),
                "{:?} ({}) should cost <= {:?} ({})",
                w[0],
                w[0].interruption_cost(),
                w[1],
                w[1].interruption_cost()
            );
        }
    }

    // ── Model Prediction ──

    #[test]
    fn test_model_idle_user_receptive() {
        let model = ReceptivityModel::new();
        let context = ContextSnapshot {
            now: 50000.0, // Some daytime hour
            activity: ActivityState::Idle,
            recent_interactions_15min: 5,
            recent_outcomes: (3, 0, 0), // All accepted
            secs_since_last_interaction: 30.0,
            session_duration_secs: 600.0,
            emotional_valence: 0.3,
            session_suggestions_accepted: 2,
            session_suggestion_budget: 20,
            notification_mode: NotificationMode::All,
        };

        let estimate = model.estimate(&context);
        assert!(
            estimate.score > 0.5,
            "Idle user should be receptive: {}",
            estimate.score
        );
        assert!(!estimate.is_quiet_hours);
    }

    #[test]
    fn test_model_focused_user_not_receptive() {
        let model = ReceptivityModel::new();
        let context = ContextSnapshot {
            now: 50000.0,
            activity: ActivityState::DeepFocus,
            recent_interactions_15min: 15,
            recent_outcomes: (1, 5, 2), // High dismissal
            secs_since_last_interaction: 10.0,
            session_duration_secs: 14400.0, // 4 hours
            emotional_valence: -0.3,
            session_suggestions_accepted: 15,
            session_suggestion_budget: 20,
            notification_mode: NotificationMode::ImportantOnly,
        };

        let estimate = model.estimate(&context);
        assert!(
            estimate.score < 0.3,
            "Focused user should not be receptive: {}",
            estimate.score
        );
    }

    #[test]
    fn test_model_dnd_blocks() {
        let model = ReceptivityModel::new();
        let context = ContextSnapshot {
            now: 50000.0,
            activity: ActivityState::Idle,
            notification_mode: NotificationMode::DoNotDisturb,
            ..Default::default()
        };

        let estimate = model.estimate(&context);
        assert_eq!(estimate.score, 0.0, "DND should block all suggestions");
    }

    #[test]
    fn test_prediction_bounds() {
        let model = ReceptivityModel::new();
        // Test with extreme feature values
        let low_features = [0.0; FEATURE_COUNT];
        let high_features = [1.0; FEATURE_COUNT];
        let p_low = model.predict(&low_features);
        let p_high = model.predict(&high_features);

        assert!(p_low >= 0.0 && p_low <= 1.0);
        assert!(p_high >= 0.0 && p_high <= 1.0);
    }

    // ── Online Learning ──

    #[test]
    fn test_learning_moves_prediction() {
        let mut model = ReceptivityModel::new();

        // Feature vector where user was receptive
        let receptive_features: FeatureVector = [0.5, 0.1, 0.8, 0.0, 0.2, 0.1, 0.5, 0.7, 0.1, 0.0];
        let pred_before = model.predict(&receptive_features);

        // Train: user accepted in this context
        for _ in 0..20 {
            model.learn(&receptive_features, SuggestionOutcome::Accepted);
        }

        let pred_after = model.predict(&receptive_features);
        assert!(
            pred_after > pred_before,
            "Training on accepts should increase prediction: {} -> {}",
            pred_before,
            pred_after
        );
    }

    #[test]
    fn test_learning_dismissals_decrease() {
        let mut model = ReceptivityModel::new();
        let features: FeatureVector = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0];
        let pred_before = model.predict(&features);

        for _ in 0..20 {
            model.learn(&features, SuggestionOutcome::Dismissed);
        }

        let pred_after = model.predict(&features);
        assert!(
            pred_after < pred_before,
            "Training on dismissals should decrease prediction: {} -> {}",
            pred_before,
            pred_after
        );
    }

    #[test]
    fn test_temporal_pattern_learning() {
        let mut model = ReceptivityModel::new();

        // Simulate: user accepts at 9am, dismisses at 3am
        let nine_am = 86400.0 + 9.0 * 3600.0; // UTC 9am
        let three_am = 86400.0 + 3.0 * 3600.0; // UTC 3am

        for _ in 0..20 {
            model.learn_temporal_pattern(nine_am, SuggestionOutcome::Accepted);
            model.learn_temporal_pattern(three_am, SuggestionOutcome::Dismissed);
        }

        let morning_receptivity = model.circadian_receptivity[9];
        let night_receptivity = model.circadian_receptivity[3];
        assert!(
            morning_receptivity > night_receptivity,
            "Morning should be more receptive than 3am: {} vs {}",
            morning_receptivity,
            night_receptivity
        );
    }

    // ── Quiet Hours ──

    #[test]
    fn test_quiet_hours_wrap_around() {
        let config = QuietHoursConfig {
            enabled: true,
            start_hour: 22,
            end_hour: 7,
            weekday_mask: 0x7F,
            allow_critical: true,
        };

        // 23:00 UTC → in quiet hours
        let eleven_pm = 86400.0 + 23.0 * 3600.0;
        assert!(config.is_quiet(eleven_pm));

        // 03:00 UTC → in quiet hours
        let three_am = 86400.0 + 3.0 * 3600.0;
        assert!(config.is_quiet(three_am));

        // 12:00 UTC → not in quiet hours
        let noon = 86400.0 + 12.0 * 3600.0;
        assert!(!config.is_quiet(noon));
    }

    #[test]
    fn test_quiet_hours_disabled() {
        let config = QuietHoursConfig {
            enabled: false,
            ..Default::default()
        };
        // Should never be quiet when disabled
        assert!(!config.is_quiet(86400.0 + 23.0 * 3600.0));
    }

    #[test]
    fn test_quiet_hours_weekday_mask() {
        let config = QuietHoursConfig {
            enabled: true,
            start_hour: 22,
            end_hour: 7,
            weekday_mask: 0x1F, // Mon-Fri only
            allow_critical: true,
        };

        // Find a Saturday (dow=5) at 23:00 UTC
        // 2024-01-06 was a Saturday → unix = 1704499200
        let saturday_night = 1704499200.0 + 23.0 * 3600.0;
        let dow = super::super::temporal::day_of_week_utc(saturday_night);
        if dow >= 5 {
            // Weekend → quiet hours not active (mask doesn't include Sat/Sun)
            assert!(!config.is_quiet(saturday_night));
        }
    }

    // ── Attention Budget ──

    #[test]
    fn test_budget_remaining() {
        let config = AttentionBudgetConfig::default();
        assert_eq!(config.remaining(0), 20);
        assert_eq!(config.remaining(15), 5);
        assert_eq!(config.remaining(25), 0); // Can't go negative
    }

    #[test]
    fn test_budget_interval() {
        let config = AttentionBudgetConfig::default();
        assert!(!config.interval_ok(60.0)); // Too soon
        assert!(config.interval_ok(120.0)); // Just right
        assert!(config.interval_ok(300.0)); // Well past
    }

    // ── Interruption Cost ──

    #[test]
    fn test_interruption_cost_urgency_discount() {
        let base = estimate_interruption_cost(ActivityState::FocusedWork, 0.0, 0.0);
        let urgent = estimate_interruption_cost(ActivityState::FocusedWork, 1.0, 0.0);
        assert!(
            urgent < base,
            "Urgent actions should have lower cost: {} vs {}",
            urgent,
            base
        );
    }

    #[test]
    fn test_interruption_cost_importance_discount() {
        let base = estimate_interruption_cost(ActivityState::Browsing, 0.0, 0.0);
        let important = estimate_interruption_cost(ActivityState::Browsing, 0.0, 1.0);
        assert!(
            important < base,
            "Important actions should have lower cost: {} vs {}",
            important,
            base
        );
    }

    #[test]
    fn test_interruption_cost_bounds() {
        for activity in [
            ActivityState::Idle,
            ActivityState::DeepFocus,
            ActivityState::Communicating,
        ] {
            for urgency in [0.0, 0.5, 1.0] {
                for importance in [0.0, 0.5, 1.0] {
                    let cost = estimate_interruption_cost(activity, urgency, importance);
                    assert!(
                        cost >= 0.0 && cost <= 1.0,
                        "Cost out of bounds: {} for {:?}, u={}, i={}",
                        cost,
                        activity,
                        urgency,
                        importance
                    );
                }
            }
        }
    }

    // ── Estimate Helpers ──

    #[test]
    fn test_estimate_top_blocker_enabler() {
        let model = ReceptivityModel::new();
        let context = ContextSnapshot {
            now: 50000.0,
            activity: ActivityState::FocusedWork,
            recent_interactions_15min: 10,
            recent_outcomes: (5, 2, 1),
            secs_since_last_interaction: 30.0,
            session_duration_secs: 3600.0,
            emotional_valence: 0.0,
            session_suggestions_accepted: 5,
            session_suggestion_budget: 20,
            notification_mode: NotificationMode::All,
        };

        let estimate = model.estimate(&context);
        assert!(estimate.top_blocker().is_some());
        assert!(estimate.top_enabler().is_some());
        assert!(
            estimate.top_blocker().unwrap().contribution
                <= estimate.top_enabler().unwrap().contribution
        );
    }

    #[test]
    fn test_is_receptive_threshold() {
        let estimate = ReceptivityEstimate {
            score: 0.6,
            factors: vec![],
            is_quiet_hours: false,
            budget_remaining: 10,
        };
        assert!(estimate.is_receptive(0.5));
        assert!(!estimate.is_receptive(0.7));

        // Quiet hours override
        let quiet_estimate = ReceptivityEstimate {
            score: 0.9,
            factors: vec![],
            is_quiet_hours: true,
            budget_remaining: 10,
        };
        assert!(!quiet_estimate.is_receptive(0.5));
    }
}
