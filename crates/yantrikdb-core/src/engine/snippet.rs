//! Snippet projection — recall reports WHERE in a record the match
//! lives, so consumers can ship a window instead of the whole text.
//!
//! Motivation (measured on a live agent session, 2026-08-05): one
//! recall returned ~15 KB of JSON across 5 hits; the one relevant hit
//! needed ~600 chars of its 2.3 KB text — a ~25× token overspend that
//! every MCP consumer pays on every call. The engine already knows the
//! matching region for chunked records (the winning window from
//! `DeltaIndex::search_with_windows`); this module turns that — plus a
//! query-term scan fallback for records that entered through the
//! keyword lane or were never chunked — into `RecallResult.best_span`.
//!
//! The span is REPORTED, never applied: `text` stays full in the core
//! result, and bindings slice on request (`snippets=true`). That keeps
//! `top_k`/`text` semantics identical for every existing consumer.

use crate::base::types::RecallResult;
use crate::vector::chunk;
use std::collections::HashMap;

/// Window size used when the embedder's input window is unknown (no
/// probe has run — e.g. installs that set the embedder Python-side).
/// 700 chars is the window the chunked-retrieval result was measured
/// with on the production clone; for snippet purposes it only needs to
/// be "a screenful of the right region", not exact.
pub(crate) const DEFAULT_SNIPPET_WINDOW: usize = 700;

/// Stamp `best_span` on hydrated results.
///
/// Resolution order per result:
/// 1. Text fits the window → `None` (nothing to trim).
/// 2. The record's winning chunk window from the vector search
///    (`win_by_rid`, ordinal 0 = head window).
/// 3. Query-term scan: the window containing the most query-term
///    occurrences (longer terms weighted heavier — a crude rarity
///    proxy that needs no corpus statistics).
/// 4. No terms hit anywhere → the head window, so a consumer trimming
///    by span still gets a deterministic, non-empty region.
///
/// Must run AFTER text hydration — spans index into the final text.
pub(crate) fn stamp_best_spans(
    scored: &mut [RecallResult],
    win_by_rid: &HashMap<String, u32>,
    query_text: Option<&str>,
    window: usize,
) {
    let terms: Vec<String> = query_text
        .map(|qt| {
            qt.split(|c: char| !c.is_alphanumeric())
                .filter(|t| t.len() >= 3)
                .map(|t| t.to_lowercase())
                .collect()
        })
        .unwrap_or_default();

    for r in scored.iter_mut() {
        if r.text.len() <= window {
            r.best_span = None;
            continue;
        }
        let windows = record_windows(&r.text, window);
        r.best_span = match win_by_rid.get(r.rid.as_str()) {
            // The vector layer knows which window matched. Guard the
            // ordinal against geometry drift (window changed since the
            // chunks were minted and rechunk hasn't run yet).
            Some(&w) if (w as usize) < windows.len() => Some(windows[w as usize]),
            _ => Some(best_term_window(&r.text, &windows, &terms)),
        };
    }
}

/// All candidate windows of a text: the head window plus the chunk
/// geometry's overlapping windows — identical ranges to what the
/// chunking write path embeds, so a vector-layer ordinal indexes
/// directly into this list.
fn record_windows(text: &str, window: usize) -> Vec<(usize, usize)> {
    let head_end = floor_char_boundary(text, window.min(text.len()));
    let mut out = vec![(0usize, head_end)];
    out.extend(chunk::chunk_ranges(text, window));
    out
}

/// The window with the highest query-term hit score; the head window
/// on a total miss. Ties break earliest (reading order).
fn best_term_window(text: &str, windows: &[(usize, usize)], terms: &[String]) -> (usize, usize) {
    if terms.is_empty() {
        return windows[0];
    }
    let mut best = windows[0];
    let mut best_score = 0usize;
    for &(a, b) in windows {
        let hay = text[a..b].to_lowercase();
        let mut score = 0usize;
        for t in terms {
            // Weight by term length: "misattributing" says more about
            // the right window than "the class" ever can.
            score += hay.matches(t.as_str()).count() * t.len();
        }
        if score > best_score {
            best_score = score;
            best = (a, b);
        }
    }
    best
}

fn floor_char_boundary(s: &str, i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    let mut j = i;
    while j > 0 && !s.is_char_boundary(j) {
        j -= 1;
    }
    j
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result_with(rid: &str, text: String) -> RecallResult {
        use crate::base::types::{ScoreBreakdown, ScoreContributions};
        RecallResult {
            rid: rid.to_string(),
            memory_type: "semantic".to_string(),
            text,
            created_at: 0.0,
            importance: 0.5,
            valence: 0.0,
            score: 0.5,
            scores: ScoreBreakdown {
                similarity: 0.5,
                decay: 0.0,
                recency: 0.0,
                importance: 0.5,
                graph_proximity: 0.0,
                contributions: ScoreContributions {
                    similarity: 0.0,
                    decay: 0.0,
                    recency: 0.0,
                    importance: 0.0,
                    graph_proximity: 0.0,
                },
                valence_multiplier: 1.0,
            },
            why_retrieved: Vec::new(),
            metadata: serde_json::Value::Null,
            namespace: "default".to_string(),
            certainty: 0.8,
            domain: "general".to_string(),
            source: "user".to_string(),
            emotional_state: None,
            current_status: Default::default(),
            superseded_by: None,
            disputed_with: Vec::new(),
            aged_last_verified: None,
            best_span: None,
        }
    }

    #[test]
    fn short_text_gets_no_span() {
        let mut scored = vec![result_with("a", "short".to_string())];
        stamp_best_spans(&mut scored, &HashMap::new(), Some("short"), 700);
        assert_eq!(scored[0].best_span, None);
    }

    #[test]
    fn vector_winner_ordinal_maps_to_its_window() {
        let text = "x".repeat(2000);
        let mut scored = vec![result_with("a", text.clone())];
        let mut wins = HashMap::new();
        wins.insert("a".to_string(), 2u32);
        stamp_best_spans(&mut scored, &wins, None, 700);
        let expected = chunk::chunk_ranges(&text, 700)[1];
        assert_eq!(scored[0].best_span, Some(expected));
    }

    #[test]
    fn term_scan_finds_the_phrase_buried_mid_text() {
        // The exact-phrase case: phrase far past the head window; no
        // vector winner (FTS-sourced candidate).
        let text = format!(
            "{}the misattributing failure surfaces named class lives here. {}",
            "filler sentence padding out the head window. ".repeat(30),
            "trailing filler. ".repeat(40),
        );
        let mut scored = vec![result_with("a", text.clone())];
        stamp_best_spans(
            &mut scored,
            &HashMap::new(),
            Some("misattributing failure surfaces named class"),
            700,
        );
        let (a, b) = scored[0].best_span.expect("long text must get a span");
        assert!(
            text[a..b].contains("misattributing"),
            "span [{a}, {b}) must cover the matched phrase"
        );
        assert!(a > 0, "the head window does not contain the phrase");
    }

    #[test]
    fn total_miss_falls_back_to_head_window() {
        let text = "unrelated words all the way down. ".repeat(60);
        let mut scored = vec![result_with("a", text)];
        stamp_best_spans(&mut scored, &HashMap::new(), Some("zzz qqq"), 700);
        let (a, _) = scored[0].best_span.unwrap();
        assert_eq!(a, 0);
    }

    #[test]
    fn stale_ordinal_past_geometry_falls_back_to_term_scan() {
        // Ordinal 9 but the current geometry only yields ~3 windows —
        // the guard must not index out of bounds.
        let text = "words ".repeat(300);
        let mut scored = vec![result_with("a", text)];
        let mut wins = HashMap::new();
        wins.insert("a".to_string(), 9u32);
        stamp_best_spans(&mut scored, &wins, Some("words"), 700);
        assert!(scored[0].best_span.is_some());
    }

    #[test]
    fn spans_respect_utf8_boundaries() {
        let text = "é".repeat(1500);
        let mut scored = vec![result_with("a", text.clone())];
        stamp_best_spans(&mut scored, &HashMap::new(), Some("nope"), 700);
        let (a, b) = scored[0].best_span.unwrap();
        let _ = &text[a..b]; // would panic on a non-boundary
    }
}
