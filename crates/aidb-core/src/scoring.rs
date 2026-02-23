/// Multi-signal scoring for memory recall.
///
/// Matches the Python implementation exactly (engine.py:246-263).

/// Compute the decay score: I(t) = importance * 2^(-t / half_life)
pub fn decay_score(importance: f64, half_life: f64, elapsed: f64) -> f64 {
    if half_life > 0.0 {
        importance * f64::powf(2.0, -elapsed / half_life)
    } else {
        0.0
    }
}

/// Compute the recency score: exp(-age / (7 * 86400))
pub fn recency_score(age: f64) -> f64 {
    f64::exp(-age / (7.0 * 86400.0))
}

/// Compute the valence boost: 1.0 + 0.3 * |valence|
pub fn valence_boost(valence: f64) -> f64 {
    1.0 + 0.3 * valence.abs()
}

/// Compute the composite recall score using multi-signal fusion.
///
/// score = (0.40 * similarity + 0.25 * decay + 0.20 * recency + 0.15 * importance) * valence_boost
pub fn composite_score(
    similarity: f64,
    decay: f64,
    recency: f64,
    importance: f64,
    valence: f64,
) -> f64 {
    let raw = 0.40 * similarity + 0.25 * decay + 0.20 * recency + 0.15 * importance.min(1.0);
    raw * valence_boost(valence)
}

/// Build a human-readable explanation for why a memory was retrieved.
pub fn build_why(similarity: f64, recency: f64, decay: f64, valence: f64) -> Vec<String> {
    let mut why = Vec::new();
    if similarity > 0.5 {
        why.push(format!("semantically similar ({similarity:.2})"));
    }
    if recency > 0.5 {
        why.push("recent".to_string());
    }
    if decay > 0.3 {
        why.push(format!("important (decay={decay:.2})"));
    }
    if valence.abs() > 0.5 {
        why.push(format!("emotionally weighted ({valence:.2})"));
    }
    if why.is_empty() {
        why.push("matched query".to_string());
    }
    why
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay_score_fresh() {
        // Just recorded: elapsed = 0
        let score = decay_score(0.8, 604800.0, 0.0);
        assert!((score - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_decay_score_one_half_life() {
        let score = decay_score(1.0, 100.0, 100.0);
        assert!((score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_decay_score_zero_half_life() {
        let score = decay_score(0.8, 0.0, 100.0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_recency_score_fresh() {
        let score = recency_score(0.0);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_recency_score_seven_days() {
        let score = recency_score(7.0 * 86400.0);
        assert!((score - f64::exp(-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_valence_boost_zero() {
        assert!((valence_boost(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_valence_boost_positive() {
        assert!((valence_boost(1.0) - 1.3).abs() < 1e-10);
    }

    #[test]
    fn test_valence_boost_negative() {
        assert!((valence_boost(-0.5) - 1.15).abs() < 1e-10);
    }

    #[test]
    fn test_composite_score_basic() {
        let score = composite_score(1.0, 1.0, 1.0, 1.0, 0.0);
        // (0.40 + 0.25 + 0.20 + 0.15) * 1.0 = 1.0
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_composite_score_with_valence() {
        let score = composite_score(1.0, 1.0, 1.0, 1.0, 1.0);
        // 1.0 * 1.3 = 1.3
        assert!((score - 1.3).abs() < 1e-10);
    }
}
