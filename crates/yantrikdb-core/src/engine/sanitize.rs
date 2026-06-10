//! Write-time sanitization of leaked tool-call serialization artifacts.
//!
//! ## Why this exists
//!
//! A cross-surface agent/client serialization bug leaked the *tail* of MCP
//! tool calls into the `text` of stored memories. The dominant corpus
//! signature (observed in ≥200 live memories during the 2026-06-10 ingest
//! audit) is a trailing run of the form:
//!
//! ```text
//! ...the real memory text...</text>
//! <parameter name="memory_type">episodic
//! ```
//!
//! i.e. the closing tag of the `text` parameter followed by one or more
//! subsequent `<parameter ...>value` fragments, captured verbatim into the
//! stored text. Left untreated this pollutes the embedding (the trailing XML
//! shifts the vector away from the real content) and surfaces as garbage in
//! recall snippets.
//!
//! ## What it does
//!
//! Strips that trailing artifact at the engine write boundary so every
//! surface (the canonical MCP server, the HTTP server, the Python library)
//! is protected regardless of whether the upstream serialization bug is ever
//! fixed. The engine is the single chokepoint all writers cross, which is
//! why the defense belongs here rather than in any one client.
//!
//! ## Design constraints
//!
//! - **Conservative.** It removes only a recognized tool-call serialization
//!   *tail* and never touches the leading prose. A memory that legitimately
//!   *discusses* `<parameter name=` mid-body (such as this very task's notes)
//!   is left intact — see [`is_short_trailing_tail`].
//! - **Zero-cost on the clean path.** Returns [`Cow::Borrowed`] unchanged
//!   when there is no artifact, so the overwhelming majority of writes pay
//!   only a substring scan, no allocation.
//! - **Clean, don't reject.** The real memory precedes the artifact, so we
//!   truncate the tail and keep the content rather than failing the write.

use std::borrow::Cow;

/// The closing tag of a leaked `text` parameter — the anchor of the dominant
/// corpus signature.
const TEXT_CLOSE: &str = "</text>";

/// Markers that, when they immediately follow a `</text>` (after optional
/// whitespace), identify that `</text>` as the start of a leaked tool-call
/// serialization tail rather than legitimate prose.
const TAIL_FOLLOW_MARKERS: &[&str] = &[
    "<parameter name=",
    "<parameter",
    "<invoke name=",
    "<invoke",
    "</invoke>",
    "</function_calls>",
    "<function_calls>",
];

/// Opening markers for the defensive "bare trailing fragment" rule — the case
/// where only the very tail of the serialization survived with no preceding
/// `</text>` to anchor on.
const BARE_TAIL_MARKERS: &[&str] = &["<parameter name=", "<invoke name=", "<function_calls>"];

/// Upper bound on the length of a bare trailing fragment we will treat as an
/// artifact. A genuine leaked tail (`<parameter name="importance">0.8` and
/// friends) is short; a long region is far more likely to be real prose that
/// merely mentions the marker.
const MAX_BARE_TAIL: usize = 256;

/// Strip a trailing leaked tool-call serialization artifact from `text`.
///
/// Returns [`Cow::Borrowed`] unchanged on the clean path. Returns
/// [`Cow::Owned`] (with trailing whitespace trimmed) when an artifact tail
/// was removed.
pub(crate) fn sanitize_tool_call_artifacts(text: &str) -> Cow<'_, str> {
    match artifact_cut_index(text) {
        Some(cut) => {
            let cleaned = text[..cut].trim_end();
            // Single cross-surface observability point: every ingest path
            // funnels through here, so one grep target
            // (`yantrikdb::audit::ingest`) confirms whether new artifacts are
            // still arriving after deploy. We log only sizes, never the
            // memory content, to keep logs free of stored text.
            tracing::warn!(
                target: "yantrikdb::audit::ingest",
                original_len = text.len(),
                cleaned_len = cleaned.len(),
                stripped_bytes = text.len() - cleaned.len(),
                "stripped leaked tool-call serialization artifact from write",
            );
            Cow::Owned(cleaned.to_string())
        }
        None => Cow::Borrowed(text),
    }
}

/// Whether `text` contains a recognized tool-call serialization artifact.
///
/// Used by callers that want to count / log occurrences (e.g. the corpus
/// repair migration in task 30) without paying for the clean copy.
pub(crate) fn has_tool_call_artifact(text: &str) -> bool {
    artifact_cut_index(text).is_some()
}

/// Return the byte index at which a recognized tool-call artifact tail begins,
/// or `None` if the text is clean. When both rules match, the earliest
/// (smallest) index wins so the entire tail is removed.
fn artifact_cut_index(text: &str) -> Option<usize> {
    let mut best: Option<usize> = None;

    // Rule 1 (primary, proven corpus signature): a `</text>` whose next
    // non-whitespace content is a tool-call follow marker. Everything from
    // that `</text>` to end-of-string is the leaked tail. Scan from the
    // front and cut at the earliest qualifying `</text>`.
    let mut from = 0;
    while let Some(rel) = text[from..].find(TEXT_CLOSE) {
        let idx = from + rel;
        let after = text[idx + TEXT_CLOSE.len()..].trim_start();
        if TAIL_FOLLOW_MARKERS.iter().any(|m| after.starts_with(m)) {
            best = Some(idx);
            break;
        }
        from = idx + TEXT_CLOSE.len();
    }

    // Rule 2 (defensive): the text ends with a bare leaked opening fragment
    // with no preceding `</text>` to anchor on (only the very tail of the
    // serialization survived). Conservative: fires only when the marker
    // begins a *short* trailing region with no blank-line paragraph break,
    // so a multi-paragraph note that merely mentions `<parameter` mid-body
    // is left intact.
    for marker in BARE_TAIL_MARKERS {
        if let Some(idx) = text.find(marker) {
            let earlier_than_best = best.map_or(true, |b| idx < b);
            if earlier_than_best && is_short_trailing_tail(&text[idx..]) {
                best = Some(idx);
            }
        }
    }

    best
}

/// True when `tail` (which begins at a candidate marker and runs to
/// end-of-string) looks like a tool-call serialization remnant: short, and
/// without a blank-line paragraph break that would signal real prose.
fn is_short_trailing_tail(tail: &str) -> bool {
    tail.len() <= MAX_BARE_TAIL && !tail.contains("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_text_is_borrowed_unchanged() {
        let s = "Alice is the engineering lead. Deadline is March 30.";
        let out = sanitize_tool_call_artifacts(s);
        assert!(matches!(out, Cow::Borrowed(_)), "clean text must not allocate");
        assert_eq!(out, s);
        assert!(!has_tool_call_artifact(s));
    }

    #[test]
    fn strips_exact_corpus_signature() {
        // The literal pattern found across ≥200 live memories.
        let s = "Real memory content here.</text>\n<parameter name=\"memory_type\">episodic";
        let out = sanitize_tool_call_artifacts(s);
        assert_eq!(out, "Real memory content here.");
        assert!(has_tool_call_artifact(s));
    }

    #[test]
    fn strips_multiple_trailing_parameters() {
        let s = "Decision: use Postgres.</text>\n\
                 <parameter name=\"memory_type\">semantic<parameter name=\"importance\">0.8";
        let out = sanitize_tool_call_artifacts(s);
        assert_eq!(out, "Decision: use Postgres.");
    }

    #[test]
    fn strips_bare_trailing_parameter_without_text_close() {
        // Only the very tail of the serialization survived.
        let s = "Some note worth keeping\n<parameter name=\"memory_type\">episodic";
        let out = sanitize_tool_call_artifacts(s);
        assert_eq!(out, "Some note worth keeping");
    }

    #[test]
    fn strips_invoke_form() {
        let s = "Content before the leak</text>\n<invoke name=\"remember\">";
        let out = sanitize_tool_call_artifacts(s);
        assert_eq!(out, "Content before the leak");
    }

    #[test]
    fn tolerates_whitespace_between_close_and_marker() {
        let s = "content</text>   \n  <parameter name=\"x\">y";
        let out = sanitize_tool_call_artifacts(s);
        assert_eq!(out, "content");
    }

    #[test]
    fn preserves_legit_midbody_mention_of_marker() {
        // A memory that *discusses* the artifact (e.g. this task's own notes)
        // must survive intact: the marker is followed by a paragraph break,
        // so the defensive rule declines to cut.
        let s = "To fix the bug, strip <parameter name= fragments at write time.\n\n\
                 This second paragraph explains why the engine is the right boundary.";
        let out = sanitize_tool_call_artifacts(s);
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(out, s);
    }

    #[test]
    fn preserves_standalone_text_close_in_prose() {
        // `</text>` not followed by a tool-call marker is legitimate (e.g.
        // discussing HTML/XML) and must not trigger a cut.
        let s = "In the lesson I typed </text> to close the element, then continued.";
        let out = sanitize_tool_call_artifacts(s);
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(out, s);
    }

    #[test]
    fn cuts_at_earliest_qualifying_close() {
        // Two `</text>` occurrences; the first is plain prose, the second
        // begins the real artifact tail. Only the artifact is removed.
        let s = "I wrote </text> earlier as an example.</text>\n<parameter name=\"x\">y";
        let out = sanitize_tool_call_artifacts(s);
        assert_eq!(out, "I wrote </text> earlier as an example.");
    }

    #[test]
    fn fully_mangled_input_collapses_to_empty() {
        // A write that is *only* an artifact has no real content to keep.
        // Documented behavior: the cleaned result is empty.
        let s = "<invoke name=\"remember\">";
        let out = sanitize_tool_call_artifacts(s);
        assert_eq!(out, "");
    }

    #[test]
    fn long_trailing_region_is_not_treated_as_artifact() {
        // A bare marker followed by a long prose region (> MAX_BARE_TAIL) is
        // left intact — it is prose, not a leaked fragment.
        let long_prose = "x".repeat(MAX_BARE_TAIL + 10);
        let s = format!("Notes about <parameter name= in the protocol: {long_prose}");
        let out = sanitize_tool_call_artifacts(&s);
        assert_eq!(out, s);
    }
}
