//! Attribute-value claim extraction from free text (v0.7.23 prototype).
//!
//! Bridges the free-text `record()` path to the claim-conflict detector
//! ([`crate::conflict::scan_claim_conflicts`]). The detector already flags
//! `same_subject_same_relation_distinct_object` — but only over claim rows in
//! the `claims`/`edges` layer, which the heuristic entity extractor never
//! populates for plain "<subject> is <value>" statements. As a result the most
//! common memory update ("brand color is blue" → later "brand color is now
//! green") produced *zero* claims and so was never detected as a conflict.
//!
//! This module closes that gap on the **extraction** side, not the detection
//! side: it turns simple copular assertions into `(subject, "is", value)`
//! triples so two values of the same attribute land as distinct objects of the
//! same `(src, rel_type)` and the existing detector fires.
//!
//! ## Scope and limitations (prototype)
//! - Only copular assertions (`is`/`are`/`was`/`were`) are recognised. "has/of"
//!   possessive attributes are future work.
//! - `rel_type` is the normalised copula (`"is"`), so *any* two differing
//!   values asserted of the same subject flag a conflict — including non-
//!   exclusive ones ("X is blue" vs "X is vibrant"). That is acceptable because
//!   a conflict is a *review signal*, not an auto-mutation; tightening it to
//!   known mutually-exclusive value classes is the natural follow-up.
//! - Hex/code annotations (`#1F4E79`) are dropped from the value so
//!   "blue #1F4E79" and "blue" do not spuriously disagree.
//! - English-only, heuristic, allocation-light. Off by default; enabled via
//!   `ThinkConfig.extract_attribute_claims`.

/// One extracted attribute-value assertion: `subject --rel--> value`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttrClaim {
    pub subject: String,
    pub rel: String,
    pub value: String,
}

/// Copulas, each padded with spaces so they only match as whole words.
const COPULAS: &[&str] = &[" is ", " are ", " was ", " were "];

/// Leading filler words stripped from the *value* side so "is now green" and
/// "is green" normalise to the same value.
const VALUE_FILLERS: &[&str] = &[
    "now",
    "currently",
    "presently",
    "still",
    "also",
    "the",
    "a",
    "an",
];

/// Cap on subject/value phrase length (words). Longer spans are almost never
/// clean attribute assertions and inflate false-positive risk.
const MAX_PHRASE_WORDS: usize = 6;

/// Extract attribute-value claims from a block of free text.
///
/// Splits on sentence boundaries and extracts at most one assertion per
/// sentence (the first copula). Returns triples with `rel = "is"`.
pub fn extract_attribute_value_claims(text: &str) -> Vec<AttrClaim> {
    text.split(|c| matches!(c, '.' | '!' | '?' | ';' | '\n'))
        .filter_map(extract_one)
        .collect()
}

/// Extract a single `(subject, "is", value)` from one sentence, or `None`.
fn extract_one(sentence: &str) -> Option<AttrClaim> {
    let trimmed = sentence.trim();
    if trimmed.is_empty() {
        return None;
    }
    // Operate on a space-padded lowercase copy so copulas at the very start or
    // end still match, and so subject/value are case-normalised consistently.
    let padded = format!(" {} ", trimmed.to_lowercase());

    // First copula occurrence (leftmost) splits subject | value.
    let (cop_idx, cop_len) = COPULAS
        .iter()
        .filter_map(|c| padded.find(c).map(|i| (i, c.len())))
        .min_by_key(|(i, _)| *i)?;

    let subject = normalize_phrase(&padded[..cop_idx], &[]);
    let value = normalize_phrase(&padded[cop_idx + cop_len..], VALUE_FILLERS);

    if subject.is_empty() || value.is_empty() {
        return None;
    }
    if subject.split_whitespace().count() > MAX_PHRASE_WORDS
        || value.split_whitespace().count() > MAX_PHRASE_WORDS
    {
        return None;
    }
    // Require at least one alphabetic char in the subject (skip "12 is 5" etc.).
    if !subject.chars().any(|c| c.is_alphabetic()) {
        return None;
    }
    Some(AttrClaim {
        subject,
        rel: "is".to_string(),
        value,
    })
}

/// Normalise a phrase: drop `#`-prefixed code/hex tokens, strip surrounding
/// punctuation, lowercase, and remove leading filler words.
fn normalize_phrase(raw: &str, strip_leading: &[&str]) -> String {
    let mut words: Vec<String> = raw
        .split_whitespace()
        // Drop annotation tokens like "#1F4E79" entirely.
        .filter(|w| !w.starts_with('#'))
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric() && c != '-')
                .to_string()
        })
        .filter(|w| !w.is_empty())
        .collect();

    while let Some(first) = words.first() {
        if strip_leading.contains(&first.as_str()) {
            words.remove(0);
        } else {
            break;
        }
    }
    words.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(text: &str) -> Option<AttrClaim> {
        let v = extract_attribute_value_claims(text);
        v.into_iter().next()
    }

    #[test]
    fn extracts_subject_and_value_dropping_hex() {
        let c = one("Brand color is blue #1F4E79.").unwrap();
        assert_eq!(c.subject, "brand color");
        assert_eq!(c.rel, "is");
        assert_eq!(c.value, "blue");
    }

    #[test]
    fn strips_value_filler_so_updates_share_subject_and_differ_on_value() {
        let blue = one("Brand color is blue #1F4E79.").unwrap();
        let green = one("Brand color is now green #2E7D32.").unwrap();
        // Same subject + rel → groups together; distinct value → conflict.
        assert_eq!(blue.subject, green.subject);
        assert_eq!(blue.rel, green.rel);
        assert_ne!(blue.value, green.value);
        assert_eq!(green.value, "green");
    }

    #[test]
    fn identical_restatement_normalizes_equal() {
        // "blue #1F4E79" and "blue" must NOT look like a conflict.
        let a = one("Brand color is blue #1F4E79.").unwrap();
        let b = one("Brand color is blue.").unwrap();
        assert_eq!(a.value, b.value);
    }

    #[test]
    fn handles_other_copulas() {
        assert_eq!(one("The deadline was March.").unwrap().value, "march");
        assert_eq!(one("Members are five.").unwrap().subject, "members");
    }

    #[test]
    fn no_copula_yields_nothing() {
        assert!(one("Just a passing thought about colors").is_none());
    }

    #[test]
    fn rejects_overlong_subject_and_value() {
        // 7-word subject before the copula → rejected.
        assert!(one("one two three four five six seven is blue").is_none());
        // 7-word value after the copula → rejected.
        assert!(one("color is one two three four five six seven").is_none());
    }

    #[test]
    fn multiple_sentences_each_extracted() {
        let v = extract_attribute_value_claims("Sky is blue. Grass is green.");
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].value, "blue");
        assert_eq!(v[1].value, "green");
    }
}
