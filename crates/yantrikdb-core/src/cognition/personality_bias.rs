//! CK-4.5 — Personality Bias Vectors.
//!
//! Structured bias vectors that modulate action scoring based on
//! the companion's personality. Same cognitive engine, different
//! character depending on the active personality profile.
//!
//! # Design principles
//! - Pure functions only — no DB access
//! - Bias is additive, not multiplicative (transparent contribution)
//! - Personality evolves gradually via EMA blending
//! - Every bias contribution is explainable

use serde::{Deserialize, Serialize};

// ── §1: Personality Bias Vector ────────────────────────────────────

/// An 8-dimensional personality bias vector.
///
/// Each dimension is ∈ [0.0, 1.0] and modulates action scoring
/// in a specific way. The vector is applied as an additive bias
/// during action evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityBiasVector {
    /// Boosts information-seeking, exploration, learning actions.
    pub curiosity: f64,
    /// Boosts anticipatory actions, volunteering help before asked.
    pub proactivity: f64,
    /// Raises confidence thresholds, prefers waiting over acting.
    pub caution: f64,
    /// Boosts emotional support, wellbeing checks, empathy.
    pub warmth: f64,
    /// Penalizes low-utility actions, prefers directness and brevity.
    pub efficiency: f64,
    /// Boosts novelty, surprise, creative and unexpected suggestions.
    pub playfulness: f64,
    /// Affects suggestion framing style (higher = more formal).
    pub formality: f64,
    /// Affects reminder frequency, follow-up tenacity, goal tracking.
    pub persistence: f64,
}

impl PersonalityBiasVector {
    /// Create a neutral personality (all 0.5).
    pub fn neutral() -> Self {
        Self {
            curiosity: 0.5,
            proactivity: 0.5,
            caution: 0.5,
            warmth: 0.5,
            efficiency: 0.5,
            playfulness: 0.5,
            formality: 0.5,
            persistence: 0.5,
        }
    }

    /// Get a dimension by index (0..8).
    pub fn dimension(&self, idx: usize) -> f64 {
        match idx {
            0 => self.curiosity,
            1 => self.proactivity,
            2 => self.caution,
            3 => self.warmth,
            4 => self.efficiency,
            5 => self.playfulness,
            6 => self.formality,
            7 => self.persistence,
            _ => 0.5,
        }
    }

    /// Set a dimension by index.
    pub fn set_dimension(&mut self, idx: usize, value: f64) {
        let v = value.clamp(0.0, 1.0);
        match idx {
            0 => self.curiosity = v,
            1 => self.proactivity = v,
            2 => self.caution = v,
            3 => self.warmth = v,
            4 => self.efficiency = v,
            5 => self.playfulness = v,
            6 => self.formality = v,
            7 => self.persistence = v,
            _ => {}
        }
    }

    /// Number of dimensions.
    pub const DIMENSIONS: usize = 8;

    /// Dimension names for display and serialization.
    pub const DIMENSION_NAMES: [&'static str; 8] = [
        "curiosity",
        "proactivity",
        "caution",
        "warmth",
        "efficiency",
        "playfulness",
        "formality",
        "persistence",
    ];

    /// Cosine similarity with another vector ∈ [-1.0, 1.0].
    pub fn similarity(&self, other: &Self) -> f64 {
        let mut dot = 0.0;
        let mut mag_a = 0.0;
        let mut mag_b = 0.0;
        for i in 0..Self::DIMENSIONS {
            let a = self.dimension(i);
            let b = other.dimension(i);
            dot += a * b;
            mag_a += a * a;
            mag_b += b * b;
        }
        let denom = mag_a.sqrt() * mag_b.sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            dot / denom
        }
    }

    /// Euclidean distance to another vector.
    pub fn distance(&self, other: &Self) -> f64 {
        let mut sum = 0.0;
        for i in 0..Self::DIMENSIONS {
            let d = self.dimension(i) - other.dimension(i);
            sum += d * d;
        }
        sum.sqrt()
    }

    /// Clamp all dimensions to [0.0, 1.0].
    pub fn clamp(&mut self) {
        for i in 0..Self::DIMENSIONS {
            let v = self.dimension(i).clamp(0.0, 1.0);
            self.set_dimension(i, v);
        }
    }
}

impl Default for PersonalityBiasVector {
    fn default() -> Self {
        Self::neutral()
    }
}

// ── §2: Preset Profiles ───────────────────────────────────────────

/// Named personality presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PersonalityPreset {
    /// High efficiency, medium caution, low playfulness.
    Assistant,
    /// High warmth, medium proactivity, medium playfulness.
    Companion,
    /// High persistence, medium curiosity, low caution.
    Coach,
    /// High caution, high proactivity, low playfulness.
    Guardian,
}

impl PersonalityPreset {
    /// Get the bias vector for this preset.
    pub fn vector(&self) -> PersonalityBiasVector {
        match self {
            Self::Assistant => PersonalityBiasVector {
                curiosity: 0.4,
                proactivity: 0.3,
                caution: 0.6,
                warmth: 0.4,
                efficiency: 0.9,
                playfulness: 0.2,
                formality: 0.7,
                persistence: 0.5,
            },
            Self::Companion => PersonalityBiasVector {
                curiosity: 0.6,
                proactivity: 0.5,
                caution: 0.4,
                warmth: 0.9,
                efficiency: 0.4,
                playfulness: 0.6,
                formality: 0.3,
                persistence: 0.5,
            },
            Self::Coach => PersonalityBiasVector {
                curiosity: 0.6,
                proactivity: 0.6,
                caution: 0.3,
                warmth: 0.5,
                efficiency: 0.7,
                playfulness: 0.3,
                formality: 0.5,
                persistence: 0.9,
            },
            Self::Guardian => PersonalityBiasVector {
                curiosity: 0.4,
                proactivity: 0.8,
                caution: 0.9,
                warmth: 0.5,
                efficiency: 0.6,
                playfulness: 0.2,
                formality: 0.7,
                persistence: 0.7,
            },
        }
    }

    /// All preset variants.
    pub const ALL: [PersonalityPreset; 4] = [
        Self::Assistant,
        Self::Companion,
        Self::Coach,
        Self::Guardian,
    ];
}

// ── §3: Action Properties for Bias Calculation ─────────────────────

/// Properties of an action that personality biases act upon.
///
/// Each field represents a dimension of the action that a personality
/// trait can boost or penalize. All values ∈ [0.0, 1.0].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActionProperties {
    /// How much information this action is expected to yield.
    pub info_gain: f64,
    /// How anticipatory this action is (acting before asked).
    pub anticipatory_value: f64,
    /// How risky this action is.
    pub risk: f64,
    /// How much emotional support this provides.
    pub emotional_utility: f64,
    /// How much goal progress this action drives.
    pub goal_progress: f64,
    /// How novel or surprising this action is.
    pub novelty: f64,
    /// How much follow-up value this creates.
    pub follow_up_value: f64,
    /// Base confidence before personality modulation.
    pub base_confidence: f64,
}

// ── §4: Bias Calculation ───────────────────────────────────────────

/// Result of applying personality bias to an action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityBiasResult {
    /// The total additive bias (can be negative).
    pub total_bias: f64,
    /// Per-dimension contributions for explainability.
    pub contributions: Vec<BiasContribution>,
    /// Confidence threshold modifier (caution raises thresholds).
    pub confidence_threshold_delta: f64,
}

/// A single dimension's contribution to the bias.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasContribution {
    pub dimension: String,
    pub personality_value: f64,
    pub action_property: f64,
    pub contribution: f64,
}

/// Compute the personality bias for an action.
///
/// Returns a signed bias value and per-dimension breakdown.
/// Positive bias = personality favors this action.
/// Negative bias = personality disfavors this action.
pub fn compute_bias(
    personality: &PersonalityBiasVector,
    action: &ActionProperties,
    config: &BiasConfig,
) -> PersonalityBiasResult {
    let mut contributions = Vec::with_capacity(8);
    let mut total = 0.0;

    // Curiosity → info gain.
    let c = centered(personality.curiosity) * action.info_gain * config.w_curiosity;
    contributions.push(BiasContribution {
        dimension: "curiosity".to_string(),
        personality_value: personality.curiosity,
        action_property: action.info_gain,
        contribution: c,
    });
    total += c;

    // Proactivity → anticipatory value.
    let c = centered(personality.proactivity) * action.anticipatory_value * config.w_proactivity;
    contributions.push(BiasContribution {
        dimension: "proactivity".to_string(),
        personality_value: personality.proactivity,
        action_property: action.anticipatory_value,
        contribution: c,
    });
    total += c;

    // Caution → penalizes risk.
    let c = -centered(personality.caution) * action.risk * config.w_caution;
    contributions.push(BiasContribution {
        dimension: "caution".to_string(),
        personality_value: personality.caution,
        action_property: action.risk,
        contribution: c,
    });
    total += c;

    // Warmth → emotional utility.
    let c = centered(personality.warmth) * action.emotional_utility * config.w_warmth;
    contributions.push(BiasContribution {
        dimension: "warmth".to_string(),
        personality_value: personality.warmth,
        action_property: action.emotional_utility,
        contribution: c,
    });
    total += c;

    // Efficiency → goal progress.
    let c = centered(personality.efficiency) * action.goal_progress * config.w_efficiency;
    contributions.push(BiasContribution {
        dimension: "efficiency".to_string(),
        personality_value: personality.efficiency,
        action_property: action.goal_progress,
        contribution: c,
    });
    total += c;

    // Playfulness → novelty.
    let c = centered(personality.playfulness) * action.novelty * config.w_playfulness;
    contributions.push(BiasContribution {
        dimension: "playfulness".to_string(),
        personality_value: personality.playfulness,
        action_property: action.novelty,
        contribution: c,
    });
    total += c;

    // Persistence → follow-up value.
    let c = centered(personality.persistence) * action.follow_up_value * config.w_persistence;
    contributions.push(BiasContribution {
        dimension: "persistence".to_string(),
        personality_value: personality.persistence,
        action_property: action.follow_up_value,
        contribution: c,
    });
    total += c;

    // Confidence threshold delta: caution raises required confidence.
    let confidence_threshold_delta = centered(personality.caution) * config.caution_threshold_scale;

    PersonalityBiasResult {
        total_bias: total,
        contributions,
        confidence_threshold_delta,
    }
}

/// Center a [0.0, 1.0] value around 0 → [-0.5, 0.5].
/// Neutral (0.5) contributes zero bias.
#[inline]
fn centered(value: f64) -> f64 {
    value - 0.5
}

// ── §5: Bias Configuration ─────────────────────────────────────────

/// Weights and scaling for personality bias application.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasConfig {
    /// Weight for curiosity dimension.
    pub w_curiosity: f64,
    /// Weight for proactivity dimension.
    pub w_proactivity: f64,
    /// Weight for caution dimension.
    pub w_caution: f64,
    /// Weight for warmth dimension.
    pub w_warmth: f64,
    /// Weight for efficiency dimension.
    pub w_efficiency: f64,
    /// Weight for playfulness dimension.
    pub w_playfulness: f64,
    /// Weight for persistence dimension.
    pub w_persistence: f64,
    /// How much caution raises the confidence threshold.
    pub caution_threshold_scale: f64,
    /// Overall bias scaling factor (controls how strongly personality affects scoring).
    pub bias_scale: f64,
}

impl Default for BiasConfig {
    fn default() -> Self {
        Self {
            w_curiosity: 1.0,
            w_proactivity: 1.0,
            w_caution: 1.0,
            w_warmth: 1.0,
            w_efficiency: 1.0,
            w_playfulness: 0.8,
            w_persistence: 0.8,
            caution_threshold_scale: 0.15,
            bias_scale: 0.3,
        }
    }
}

// ── §6: Personality Evolution ──────────────────────────────────────

/// Bond level affects how much personality is expressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BondLevel {
    /// New relationship — cautious, formal.
    Stranger,
    /// Early interactions — slightly warmer.
    Acquaintance,
    /// Established rapport — moderate expression.
    Familiar,
    /// Strong relationship — full expression.
    Bonded,
    /// Deep trust — personality fully expressed.
    Trusted,
}

impl BondLevel {
    /// Expression multiplier ∈ [0.3, 1.0].
    /// At low bond, personality differences from neutral are dampened.
    pub fn expression_factor(&self) -> f64 {
        match self {
            Self::Stranger => 0.3,
            Self::Acquaintance => 0.5,
            Self::Familiar => 0.7,
            Self::Bonded => 0.9,
            Self::Trusted => 1.0,
        }
    }
}

/// Apply bond-level dampening to a personality vector.
///
/// At low bond levels, personality traits are pulled toward neutral (0.5).
/// At high bond levels, the full personality is expressed.
pub fn dampen_personality(
    personality: &PersonalityBiasVector,
    bond: BondLevel,
) -> PersonalityBiasVector {
    let factor = bond.expression_factor();
    let mut result = personality.clone();
    for i in 0..PersonalityBiasVector::DIMENSIONS {
        let raw = personality.dimension(i);
        // Lerp between neutral (0.5) and actual value.
        let dampened = 0.5 + (raw - 0.5) * factor;
        result.set_dimension(i, dampened);
    }
    result
}

/// Learned preference adjustments from user feedback.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LearnedPreferences {
    /// Per-dimension adjustments learned from user feedback.
    /// Each entry: (dimension_index, adjustment ∈ [-0.2, 0.2]).
    pub adjustments: Vec<(usize, f64)>,
    /// Total feedback observations.
    pub observation_count: u64,
}

/// Evolve personality based on bond level and learned preferences.
///
/// Uses EMA blending to gradually shift personality toward what
/// the user seems to prefer, bounded by the evolution config.
pub fn evolve_personality(
    current: &PersonalityBiasVector,
    bond: BondLevel,
    preferences: &LearnedPreferences,
    config: &EvolutionConfig,
) -> PersonalityBiasVector {
    let mut evolved = current.clone();

    // Only evolve if enough observations.
    if preferences.observation_count < config.min_observations {
        return evolved;
    }

    // Bond gates evolution rate.
    let rate = config.base_learning_rate * bond.expression_factor();

    // Apply learned adjustments via EMA.
    for &(dim_idx, adjustment) in &preferences.adjustments {
        if dim_idx < PersonalityBiasVector::DIMENSIONS {
            let current_val = evolved.dimension(dim_idx);
            let target = (current_val + adjustment).clamp(0.0, 1.0);
            let new_val = current_val + rate * (target - current_val);
            evolved.set_dimension(
                dim_idx,
                new_val.clamp(config.min_trait_value, config.max_trait_value),
            );
        }
    }

    evolved
}

/// Configuration for personality evolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    /// Base EMA learning rate for personality evolution.
    pub base_learning_rate: f64,
    /// Minimum observations before evolution begins.
    pub min_observations: u64,
    /// Hard floor for any trait value.
    pub min_trait_value: f64,
    /// Hard ceiling for any trait value.
    pub max_trait_value: f64,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            base_learning_rate: 0.1,
            min_observations: 10,
            min_trait_value: 0.05,
            max_trait_value: 0.95,
        }
    }
}

// ── §7: Personality Impact Report ──────────────────────────────────

/// Human-readable report of how personality affected a decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityImpactReport {
    /// The active personality profile name/preset.
    pub profile_name: String,
    /// Bond level at decision time.
    pub bond_level: BondLevel,
    /// The effective (dampened) personality vector.
    pub effective_personality: PersonalityBiasVector,
    /// Per-action bias results.
    pub action_biases: Vec<ActionBiasEntry>,
    /// Which action was most boosted by personality.
    pub most_boosted: Option<String>,
    /// Which action was most penalized by personality.
    pub most_penalized: Option<String>,
}

/// Bias result for a single action in the impact report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionBiasEntry {
    pub action_description: String,
    pub bias_result: PersonalityBiasResult,
}

/// Generate a personality impact report for a set of actions.
pub fn personality_impact(
    personality: &PersonalityBiasVector,
    bond: BondLevel,
    profile_name: &str,
    actions: &[(&str, &ActionProperties)],
    config: &BiasConfig,
) -> PersonalityImpactReport {
    let effective = dampen_personality(personality, bond);

    let mut action_biases = Vec::with_capacity(actions.len());
    for (desc, props) in actions {
        let result = compute_bias(&effective, props, config);
        action_biases.push(ActionBiasEntry {
            action_description: desc.to_string(),
            bias_result: result,
        });
    }

    let most_boosted = action_biases
        .iter()
        .max_by(|a, b| {
            a.bias_result
                .total_bias
                .partial_cmp(&b.bias_result.total_bias)
                .unwrap()
        })
        .filter(|a| a.bias_result.total_bias > 0.0)
        .map(|a| a.action_description.clone());

    let most_penalized = action_biases
        .iter()
        .min_by(|a, b| {
            a.bias_result
                .total_bias
                .partial_cmp(&b.bias_result.total_bias)
                .unwrap()
        })
        .filter(|a| a.bias_result.total_bias < 0.0)
        .map(|a| a.action_description.clone());

    PersonalityImpactReport {
        profile_name: profile_name.to_string(),
        bond_level: bond,
        effective_personality: effective,
        action_biases,
        most_boosted,
        most_penalized,
    }
}

// ── §8: Personality Store ──────────────────────────────────────────

/// Persistent personality state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalityBiasStore {
    /// The current personality bias vector.
    pub current: PersonalityBiasVector,
    /// The base preset this personality was derived from.
    pub base_preset: Option<PersonalityPreset>,
    /// Current bond level.
    pub bond_level: BondLevel,
    /// Learned preference adjustments from user feedback.
    pub preferences: LearnedPreferences,
    /// Evolution configuration.
    pub evolution_config: EvolutionConfig,
    /// Bias calculation configuration.
    pub bias_config: BiasConfig,
    /// Total evolution steps applied.
    pub evolution_count: u64,
    /// Last evolution timestamp.
    pub last_evolved_at: f64,
}

impl PersonalityBiasStore {
    pub fn new() -> Self {
        Self {
            current: PersonalityBiasVector::neutral(),
            base_preset: None,
            bond_level: BondLevel::Stranger,
            preferences: LearnedPreferences::default(),
            evolution_config: EvolutionConfig::default(),
            bias_config: BiasConfig::default(),
            evolution_count: 0,
            last_evolved_at: 0.0,
        }
    }

    /// Create from a preset.
    pub fn from_preset(preset: PersonalityPreset) -> Self {
        Self {
            current: preset.vector(),
            base_preset: Some(preset),
            ..Self::new()
        }
    }

    /// Apply bias to an action using the current personality and bond level.
    pub fn apply_bias(&self, action: &ActionProperties) -> PersonalityBiasResult {
        let effective = dampen_personality(&self.current, self.bond_level);
        compute_bias(&effective, action, &self.bias_config)
    }

    /// Record user feedback and evolve personality.
    pub fn record_feedback(&mut self, dimension_idx: usize, adjustment: f64, now: f64) {
        let bounded = adjustment.clamp(-0.2, 0.2);
        self.preferences.adjustments.push((dimension_idx, bounded));
        self.preferences.observation_count += 1;

        // Evolve.
        self.current = evolve_personality(
            &self.current,
            self.bond_level,
            &self.preferences,
            &self.evolution_config,
        );
        self.evolution_count += 1;
        self.last_evolved_at = now;
    }

    /// Update bond level.
    pub fn set_bond_level(&mut self, bond: BondLevel) {
        self.bond_level = bond;
    }
}

// ── §9: Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neutral_personality() {
        let neutral = PersonalityBiasVector::neutral();
        for i in 0..PersonalityBiasVector::DIMENSIONS {
            assert!((neutral.dimension(i) - 0.5).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_preset_profiles() {
        for preset in PersonalityPreset::ALL {
            let vec = preset.vector();
            for i in 0..PersonalityBiasVector::DIMENSIONS {
                let v = vec.dimension(i);
                assert!(v >= 0.0 && v <= 1.0, "{:?} dim {} = {}", preset, i, v);
            }
        }
    }

    #[test]
    fn test_similarity_identical() {
        let a = PersonalityPreset::Companion.vector();
        let sim = a.similarity(&a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_different() {
        let a = PersonalityPreset::Guardian.vector();
        let b = PersonalityPreset::Companion.vector();
        let sim = a.similarity(&b);
        // Different but not orthogonal.
        assert!(sim > 0.5 && sim < 1.0);
    }

    #[test]
    fn test_distance_self_zero() {
        let a = PersonalityPreset::Coach.vector();
        assert!(a.distance(&a) < 1e-10);
    }

    #[test]
    fn test_bias_neutral_zero() {
        let neutral = PersonalityBiasVector::neutral();
        let action = ActionProperties {
            info_gain: 0.5,
            anticipatory_value: 0.5,
            risk: 0.5,
            emotional_utility: 0.5,
            goal_progress: 0.5,
            novelty: 0.5,
            follow_up_value: 0.5,
            base_confidence: 0.7,
        };
        let config = BiasConfig::default();
        let result = compute_bias(&neutral, &action, &config);

        // Neutral personality → zero bias.
        assert!(
            result.total_bias.abs() < 1e-10,
            "Expected ~0 bias from neutral personality, got {}",
            result.total_bias
        );
        assert!(result.confidence_threshold_delta.abs() < 1e-10);
    }

    #[test]
    fn test_bias_cautious_penalizes_risk() {
        let cautious = PersonalityPreset::Guardian.vector();
        let risky_action = ActionProperties {
            risk: 0.9,
            ..Default::default()
        };
        let config = BiasConfig::default();
        let result = compute_bias(&cautious, &risky_action, &config);

        // Caution is 0.9 for Guardian → centered = 0.4.
        // Contribution = -0.4 * 0.9 * 1.0 = -0.36.
        assert!(
            result.total_bias < 0.0,
            "High caution should penalize risky actions"
        );

        // Confidence threshold should be raised.
        assert!(result.confidence_threshold_delta > 0.0);
    }

    #[test]
    fn test_bias_warm_boosts_emotional() {
        let warm = PersonalityPreset::Companion.vector();
        let emotional_action = ActionProperties {
            emotional_utility: 0.8,
            ..Default::default()
        };
        let config = BiasConfig::default();
        let result = compute_bias(&warm, &emotional_action, &config);

        // Warmth is 0.9 for Companion → centered = 0.4.
        // Contribution = 0.4 * 0.8 * 1.0 = 0.32.
        assert!(
            result.total_bias > 0.0,
            "High warmth should boost emotional actions"
        );
    }

    #[test]
    fn test_bias_contributions_sum_to_total() {
        let personality = PersonalityPreset::Coach.vector();
        let action = ActionProperties {
            info_gain: 0.7,
            anticipatory_value: 0.3,
            risk: 0.4,
            emotional_utility: 0.2,
            goal_progress: 0.8,
            novelty: 0.5,
            follow_up_value: 0.9,
            base_confidence: 0.6,
        };
        let config = BiasConfig::default();
        let result = compute_bias(&personality, &action, &config);

        let sum: f64 = result.contributions.iter().map(|c| c.contribution).sum();
        assert!((sum - result.total_bias).abs() < 1e-10);
    }

    #[test]
    fn test_bond_dampening() {
        let full = PersonalityPreset::Companion.vector();

        let stranger = dampen_personality(&full, BondLevel::Stranger);
        let trusted = dampen_personality(&full, BondLevel::Trusted);

        // Stranger should be closer to neutral.
        let neutral = PersonalityBiasVector::neutral();
        assert!(stranger.distance(&neutral) < full.distance(&neutral));

        // Trusted should be close to original.
        assert!(trusted.distance(&full) < 1e-10);
    }

    #[test]
    fn test_bond_expression_factors() {
        assert!(BondLevel::Stranger.expression_factor() < BondLevel::Trusted.expression_factor());
        assert!((BondLevel::Trusted.expression_factor() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evolution_no_change_insufficient_data() {
        let personality = PersonalityPreset::Assistant.vector();
        let preferences = LearnedPreferences {
            adjustments: vec![(0, 0.1)],
            observation_count: 3, // Below min_observations (10).
        };
        let config = EvolutionConfig::default();

        let evolved = evolve_personality(&personality, BondLevel::Bonded, &preferences, &config);

        // Should not change — too few observations.
        assert!(evolved.distance(&personality) < 1e-10);
    }

    #[test]
    fn test_evolution_applies_with_sufficient_data() {
        let personality = PersonalityPreset::Assistant.vector();
        let preferences = LearnedPreferences {
            adjustments: vec![(0, 0.15)], // Boost curiosity.
            observation_count: 20,
        };
        let config = EvolutionConfig::default();

        let evolved = evolve_personality(&personality, BondLevel::Bonded, &preferences, &config);

        // Curiosity should increase.
        assert!(evolved.curiosity > personality.curiosity);
    }

    #[test]
    fn test_store_from_preset() {
        let store = PersonalityBiasStore::from_preset(PersonalityPreset::Companion);
        assert_eq!(store.base_preset, Some(PersonalityPreset::Companion));
        assert!((store.current.warmth - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_store_apply_bias() {
        let store = PersonalityBiasStore::from_preset(PersonalityPreset::Guardian);
        let action = ActionProperties {
            risk: 0.8,
            goal_progress: 0.6,
            ..Default::default()
        };
        let result = store.apply_bias(&action);

        // Guardian at Stranger bond → dampened caution.
        // Should still show some penalty for risk.
        assert!(result.contributions.len() == 7);
    }

    #[test]
    fn test_personality_impact_report() {
        let personality = PersonalityPreset::Companion.vector();
        let action_a = ActionProperties {
            emotional_utility: 0.9,
            ..Default::default()
        };
        let action_b = ActionProperties {
            risk: 0.8,
            ..Default::default()
        };

        let report = personality_impact(
            &personality,
            BondLevel::Bonded,
            "Companion",
            &[("Comfort", &action_a), ("Risky move", &action_b)],
            &BiasConfig::default(),
        );

        assert_eq!(report.action_biases.len(), 2);
        assert_eq!(report.profile_name, "Companion");

        // Companion should boost emotional action.
        let comfort_bias = &report.action_biases[0].bias_result;
        assert!(comfort_bias.total_bias > 0.0);
    }

    #[test]
    fn test_record_feedback_evolves() {
        let mut store = PersonalityBiasStore::from_preset(PersonalityPreset::Assistant);
        store.bond_level = BondLevel::Bonded;
        let original_curiosity = store.current.curiosity;

        // Record enough feedback to trigger evolution.
        for _ in 0..15 {
            store.record_feedback(0, 0.1, 1_000_000.0);
        }

        // Curiosity should have increased.
        assert!(store.current.curiosity > original_curiosity);
        assert_eq!(store.evolution_count, 15);
    }
}
