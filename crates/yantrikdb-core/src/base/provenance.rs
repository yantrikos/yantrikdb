//! **v0.10 Item 4a — typed provenance grammar + the anti-laundering consistency
//! matrix.**
//!
//! PURE and stateless: it parses a record's DECLARED provenance fields and
//! checks their internal consistency with no dependence on engine state, so a
//! leader and a follower reach the same verdict by construction. Wired into the
//! write paths in Item 4a.4.
//!
//! Scope + limitation (nuron/sol): the gate prevents DECLARED contradictions
//! (e.g. `source=inference` claiming `kind=fact`). It cannot verify truthful
//! provenance — a caller that lies (`source=user` for an inference) or omits
//! provenance is undetectable. Omission still cannot yield an
//! inference-claiming-fact because the matrix keys off the declared fields.

use crate::error::{Result, YantrikDbError};

/// Who/how a record was obtained. **Closed** vocabulary: an unparseable source
/// is rejected, so a caller cannot smuggle `source="inferece"` to dodge the
/// matrix. `Source` stays a `&str` at the engine API; the gate parses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    User,
    Inference,
    Document,
    System,
}

impl Source {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "user" => Ok(Self::User),
            "inference" => Ok(Self::Inference),
            "document" => Ok(Self::Document),
            "system" => Ok(Self::System),
            other => Err(YantrikDbError::ProvenanceInconsistent {
                path: "source",
                reason: format!(
                    "unknown source '{other}' (expected user|inference|document|system)"
                ),
            }),
        }
    }

    /// The canonical stored form. Recall returns THIS (T06 "verbatim" means the
    /// canonical accepted value; a value that needed whitespace/case rescue is
    /// stored canonicalized — no provenance information is lost).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Inference => "inference",
            Self::Document => "document",
            Self::System => "system",
        }
    }
}

/// The justification tier. **Closed** vocabulary (unparseable → rejected).
/// `Asserted` — "a named source claimed it; I have not verified it" — is the
/// majority case ("user said X" / "doc states X") and its absence would force a
/// false `observation` stamp (nuron). `Learned` carries the model id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfidenceBasis {
    Observation,
    Asserted,
    Confirmation,
    Verification,
    Inference,
    Assumption,
    Learned { model: String },
}

impl ConfidenceBasis {
    pub fn parse(s: &str) -> Result<Self> {
        let t = s.trim();
        match t.to_ascii_lowercase().as_str() {
            "observation" => return Ok(Self::Observation),
            "asserted" => return Ok(Self::Asserted),
            "confirmation" => return Ok(Self::Confirmation),
            "verification" => return Ok(Self::Verification),
            "inference" => return Ok(Self::Inference),
            "assumption" => return Ok(Self::Assumption),
            _ => {}
        }
        if let Some(model) = parse_learned(t) {
            return Ok(Self::Learned { model });
        }
        Err(YantrikDbError::ProvenanceInconsistent {
            path: "confidence_basis",
            reason: format!(
                "unknown confidence_basis '{t}' (expected observation|asserted|confirmation|\
                 verification|inference|assumption|learned(<model>))"
            ),
        })
    }

    pub fn to_canonical(&self) -> String {
        match self {
            Self::Observation => "observation".into(),
            Self::Asserted => "asserted".into(),
            Self::Confirmation => "confirmation".into(),
            Self::Verification => "verification".into(),
            Self::Inference => "inference".into(),
            Self::Assumption => "assumption".into(),
            Self::Learned { model } => format!("learned({model})"),
        }
    }
}

/// `learned(<model>)` grammar: model is `[A-Za-z0-9._-]{1,64}` — non-empty,
/// bounded, ASCII only, no nested parens/unicode (sol r3 precision). Returns the
/// model string, or `None` if the shape does not match.
fn parse_learned(s: &str) -> Option<String> {
    // Case-INSENSITIVE "learned(" prefix, case-PRESERVING model. (`s` is
    // already trimmed by the caller.)
    let body = s.strip_suffix(')')?;
    if body.len() < 8 || !body.is_char_boundary(8) {
        return None;
    }
    let (prefix, inner) = body.split_at(8); // "learned(" is 8 ASCII bytes
    if !prefix.eq_ignore_ascii_case("learned(") {
        return None;
    }
    if inner.is_empty() || inner.len() > 64 {
        return None;
    }
    if !inner
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    {
        return None;
    }
    Some(inner.to_string())
}

/// The record's `metadata.kind`. **Open** vocabulary: any string is accepted,
/// but only the PROTECTED tier {`fact`, `observation`, `inference`} participates
/// in the matrix. Missing / null / empty / non-string kind → `Unspecified`
/// (sol r3: cannot "reject an unparseable protected kind" under open vocab —
/// `fact.` / `inferece` are simply `Other`, not errors).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimKind {
    Fact,
    Observation,
    Inference,
    Unspecified,
    Other(String),
}

impl ClaimKind {
    /// Parse from the record's final plaintext `metadata.kind` (already merged,
    /// pre-encryption). `None` = key absent or non-string.
    pub fn parse(kind: Option<&str>) -> Self {
        match kind {
            None => Self::Unspecified,
            Some(k) => {
                let t = k.trim();
                if t.is_empty() {
                    return Self::Unspecified;
                }
                match t.to_ascii_lowercase().as_str() {
                    "fact" => Self::Fact,
                    "observation" => Self::Observation,
                    "inference" => Self::Inference,
                    _ => Self::Other(t.to_string()),
                }
            }
        }
    }

    /// The protected "authoritative claim" tier that an inference-source record
    /// may not assert without confirmation/verification or an explicit override.
    fn is_authoritative(&self) -> bool {
        matches!(self, Self::Fact | Self::Observation)
    }
}

/// The anti-laundering consistency matrix (pure). Given already-parsed declared
/// fields, decide whether the record may be admitted. `override_kind` is the
/// deliberate, traced+stamped escape hatch (the DOCUMENTED escape is instead to
/// raise `confidence_basis` to confirmation/verification).
pub fn check_provenance_consistency(
    source: Source,
    basis: &ConfidenceBasis,
    kind: &ClaimKind,
    override_kind: bool,
) -> Result<()> {
    if source == Source::Inference {
        // You cannot have OBSERVED an inference.
        if *basis == ConfidenceBasis::Observation {
            return Err(YantrikDbError::ProvenanceInconsistent {
                path: "source/confidence_basis",
                reason: "source=inference cannot have confidence_basis=observation \
                         (you did not observe an inference)"
                    .into(),
            });
        }
        // An inference cannot CLAIM to be a fact/observation unless it was
        // independently confirmed/verified (the "raise your basis" allowance)
        // or explicitly overridden (traced + stamped).
        if kind.is_authoritative()
            && !override_kind
            && !matches!(
                basis,
                ConfidenceBasis::Confirmation | ConfidenceBasis::Verification
            )
        {
            let kind_str = match kind {
                ClaimKind::Fact => "fact",
                _ => "observation",
            };
            return Err(YantrikDbError::ProvenanceInconsistent {
                path: "source/kind",
                reason: format!(
                    "source=inference cannot claim kind={kind_str} without confidence_basis in \
                     {{confirmation, verification}} or an explicit override_kind"
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_parse_closed_vocab() {
        assert_eq!(Source::parse("user").unwrap(), Source::User);
        assert_eq!(Source::parse("  Inference ").unwrap(), Source::Inference);
        assert_eq!(Source::parse("DOCUMENT").unwrap(), Source::Document);
        // unparseable protected source is rejected (no laundering via typos)
        assert!(Source::parse("inferece").is_err());
        assert!(Source::parse("").is_err());
    }

    #[test]
    fn basis_parse_incl_asserted_and_learned() {
        assert_eq!(
            ConfidenceBasis::parse("asserted").unwrap(),
            ConfidenceBasis::Asserted
        );
        assert_eq!(
            ConfidenceBasis::parse(" Learned(bge-base-en-v1.5) ").unwrap(),
            ConfidenceBasis::Learned {
                model: "bge-base-en-v1.5".into()
            }
        );
        assert!(ConfidenceBasis::parse("learned()").is_err()); // empty model
        assert!(ConfidenceBasis::parse("learned(a b)").is_err()); // space illegal
        assert!(ConfidenceBasis::parse("nonsense").is_err());
        // canonical round-trip
        assert_eq!(
            ConfidenceBasis::parse("learned(m1)")
                .unwrap()
                .to_canonical(),
            "learned(m1)"
        );
    }

    #[test]
    fn kind_open_vocab_and_unspecified() {
        assert_eq!(ClaimKind::parse(Some("fact")), ClaimKind::Fact);
        assert_eq!(
            ClaimKind::parse(Some(" Observation ")),
            ClaimKind::Observation
        );
        // open vocab: non-protected is Other, not an error
        assert_eq!(
            ClaimKind::parse(Some("procedure")),
            ClaimKind::Other("procedure".into())
        );
        // fact. is NOT protected-fact (sol r3): it's Other
        assert_eq!(
            ClaimKind::parse(Some("fact.")),
            ClaimKind::Other("fact.".into())
        );
        assert_eq!(ClaimKind::parse(None), ClaimKind::Unspecified);
        assert_eq!(ClaimKind::parse(Some("")), ClaimKind::Unspecified);
        assert_eq!(ClaimKind::parse(Some("   ")), ClaimKind::Unspecified);
    }

    #[test]
    fn matrix_refuses_inference_observation_basis() {
        let e = check_provenance_consistency(
            Source::Inference,
            &ConfidenceBasis::Observation,
            &ClaimKind::Inference,
            false,
        );
        assert!(e.is_err(), "inference + observation basis must refuse");
    }

    #[test]
    fn matrix_refuses_inference_claiming_fact() {
        let e = check_provenance_consistency(
            Source::Inference,
            &ConfidenceBasis::Asserted,
            &ClaimKind::Fact,
            false,
        );
        assert!(
            e.is_err(),
            "inference claiming fact (no confirm/override) refused"
        );
    }

    #[test]
    fn matrix_confirmation_allowance() {
        // raise-your-basis: inference claiming fact IS allowed with confirmation
        assert!(check_provenance_consistency(
            Source::Inference,
            &ConfidenceBasis::Confirmation,
            &ClaimKind::Fact,
            false,
        )
        .is_ok());
        assert!(check_provenance_consistency(
            Source::Inference,
            &ConfidenceBasis::Verification,
            &ClaimKind::Observation,
            false,
        )
        .is_ok());
    }

    #[test]
    fn matrix_override_allowance() {
        assert!(check_provenance_consistency(
            Source::Inference,
            &ConfidenceBasis::Asserted,
            &ClaimKind::Fact,
            true, // explicit override
        )
        .is_ok());
    }

    #[test]
    fn matrix_allows_consistent_records() {
        // user asserting a fact — fine
        assert!(check_provenance_consistency(
            Source::User,
            &ConfidenceBasis::Asserted,
            &ClaimKind::Fact,
            false,
        )
        .is_ok());
        // inference labeled as an inference — fine
        assert!(check_provenance_consistency(
            Source::Inference,
            &ConfidenceBasis::Inference,
            &ClaimKind::Inference,
            false,
        )
        .is_ok());
        // inference with an unprotected kind — fine
        assert!(check_provenance_consistency(
            Source::Inference,
            &ConfidenceBasis::Asserted,
            &ClaimKind::Other("hypothesis".into()),
            false,
        )
        .is_ok());
        // unspecified kind never triggers the authoritative-claim rule
        assert!(check_provenance_consistency(
            Source::Inference,
            &ConfidenceBasis::Asserted,
            &ClaimKind::Unspecified,
            false,
        )
        .is_ok());
    }
}
