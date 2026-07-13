//! Central numeric/vector contract gate (v0.9.3, issue #60 follow-up).
//!
//! One shared validator for every engine entry path that accepts an
//! embedding or scoring scalar. The contract: validation runs BEFORE any
//! side effect — importance calibration, SQL insert, oplog append, counter
//! bump, delta append, or index mutation — so a rejected write leaves the
//! engine byte-for-byte unchanged.
//!
//! Why this exists: v0.9.2 hardened the *consumption* side (NaN can no
//! longer panic recall's sort comparators), but a non-finite embedding or
//! scalar could still be *persisted* — silent corruption that scores as
//! garbage forever (a NaN embedding cosine-guards to distance 1.0 against
//! everything). The gate closes the class at the source, on every path:
//! caller-supplied vectors (`record`, `record_batch`, `record_with_rid`,
//! `insert_vector`), engine-generated vectors (`embed`, and therefore
//! `record_text`/`recall_text`/reembed staging), caller-supplied query
//! vectors (`recall`), and write-path scalars (importance / valence /
//! certainty / half_life, whose NaN/Inf poison decay + scoring math).
//!
//! Scope note (patch-release discipline): finiteness and dimension are
//! hard-rejected — they are unambiguous corruption. Range *policy* (e.g.
//! importance > 1.0) is intentionally NOT enforced here; the engine
//! tolerates and calibrates out-of-range importances today, and tightening
//! that contract is a behavior change reserved for a minor release.

use crate::error::{Result, YantrikDbError};

/// Validate an embedding at an engine entry path: expected dimension and
/// every element finite. `path` names the API for the error message.
pub fn validate_embedding(
    path: &'static str,
    embedding: &[f32],
    expected_dim: usize,
) -> Result<()> {
    if embedding.len() != expected_dim {
        return Err(YantrikDbError::InvalidEmbedding {
            path,
            index: None,
            reason: format!(
                "dimension mismatch: expected {expected_dim}, got {}",
                embedding.len()
            ),
        });
    }
    if let Some(index) = embedding.iter().position(|v| !v.is_finite()) {
        return Err(YantrikDbError::InvalidEmbedding {
            path,
            index: Some(index),
            reason: format!("non-finite element {} at index {index}", embedding[index]),
        });
    }
    Ok(())
}

/// Validate write-path scoring scalars are finite. Pass `(field, value)`
/// pairs; the first non-finite one is rejected with a typed error.
pub fn validate_scalars(path: &'static str, fields: &[(&'static str, f64)]) -> Result<()> {
    for &(field, value) in fields {
        if !value.is_finite() {
            return Err(YantrikDbError::InvalidScalar { path, field, value });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_valid_embedding_and_scalars() {
        assert!(validate_embedding("record", &[0.1, -0.2, 0.3], 3).is_ok());
        assert!(validate_scalars("record", &[("importance", 0.5), ("valence", -1.0)]).is_ok());
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let err = validate_embedding("record", &[0.1, 0.2], 3).unwrap_err();
        match err {
            YantrikDbError::InvalidEmbedding {
                path,
                index,
                reason,
            } => {
                assert_eq!(path, "record");
                assert_eq!(index, None);
                assert!(reason.contains("expected 3, got 2"), "{reason}");
            }
            other => panic!("wrong error: {other}"),
        }
    }

    #[test]
    fn rejects_non_finite_elements_with_index() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = validate_embedding("insert_vector", &[0.1, bad, 0.3], 3).unwrap_err();
            match err {
                YantrikDbError::InvalidEmbedding { index, .. } => assert_eq!(index, Some(1)),
                other => panic!("wrong error: {other}"),
            }
        }
    }

    #[test]
    fn rejects_non_finite_scalars_by_field() {
        let err = validate_scalars("record", &[("importance", 0.5), ("half_life", f64::NAN)])
            .unwrap_err();
        match err {
            YantrikDbError::InvalidScalar { field, .. } => assert_eq!(field, "half_life"),
            other => panic!("wrong error: {other}"),
        }
    }
}
