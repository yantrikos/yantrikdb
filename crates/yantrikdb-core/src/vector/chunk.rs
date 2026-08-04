//! Chunked embeddings — one record, N vectors.
//!
//! A truncating embedder (all-MiniLM-L6-v2: 256 tokens) silently drops
//! every token past its window; measured on a production install, 73%
//! of records exceeded it and a fragment from a record's END retrieved
//! its parent 8% of the time. Chunking is the fix half of that defect
//! (detection shipped as `detect_embedder_window`): when a record's
//! text exceeds the embedder's known window, the engine embeds
//! overlapping windows of it and indexes each under a synthetic key.
//! Retrieval collapses those keys back to the parent rid, so the
//! record is findable from any part of its text. Measured on the
//! corpus that surfaced the defect: MRR 0.057 → 0.395 (7×), beating an
//! embedder swap on every metric. See docs/chunked_embeddings_design.md.
//!
//! This module owns the two pure pieces both layers share: the chunk
//! KEY encoding (vector layer collapses keys; engine layer mints and
//! tombstones them) and the window GEOMETRY (engine layer slices text;
//! tests reason about coverage).

/// Separator between a parent rid and a chunk ordinal. Rids are UUIDv7
/// strings — `#` cannot appear in one, so `parent_of` is unambiguous.
/// Index keys are opaque strings at every layer (HNSW, delta, cold),
/// which is what makes a synthetic key legal in the first place.
const CHUNK_SEP: &str = "#c";

/// Hard ceiling on chunks per record. Bounds delta-index slot
/// consumption (a chunked write reserves 1+N slots against delta_max)
/// and pathological inputs. The production corpus that motivated this
/// maxes out at ~5 windows.
pub const MAX_CHUNKS: usize = 16;

/// The index key for chunk `idx` (1-based) of `rid`. Chunk 0 is the
/// record's own embedding under the plain rid — it never gets a
/// synthetic key.
pub fn chunk_key(rid: &str, idx: usize) -> String {
    debug_assert!(idx >= 1, "chunk 0 lives under the plain rid");
    format!("{rid}{CHUNK_SEP}{idx}")
}

/// The parent rid of an index key: strips a `#c<digits>` suffix, and
/// returns the key untouched when it is a plain rid.
///
/// Strict about the suffix (`#c` followed by ONLY digits) so a
/// hypothetical user key containing `#c` mid-string is not mangled —
/// collapse must never merge two records that are actually distinct.
pub fn parent_of(key: &str) -> &str {
    if let Some(pos) = key.rfind(CHUNK_SEP) {
        let suffix = &key[pos + CHUNK_SEP.len()..];
        if !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) {
            return &key[..pos];
        }
    }
    key
}

/// Collapse `(key, distance)` search results to at most one entry per
/// PARENT rid, keeping the lowest distance, preserving ascending-
/// distance order of the survivors. Input must be sorted ascending by
/// distance (both HNSW and the delta merge already guarantee it), so
/// the first occurrence of a parent is its best window and later ones
/// drop — one pass, no re-sort.
pub fn collapse_to_parents(results: Vec<(String, f64)>) -> Vec<(String, f64)> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut out: Vec<(String, f64)> = Vec::with_capacity(results.len());
    for (key, dist) in results {
        let parent = parent_of(&key);
        if seen.insert(parent.to_string()) {
            out.push((parent.to_string(), dist));
        }
    }
    out
}

/// The character windows to embed for a text of `len` chars under an
/// embedder window of `window` chars: chunk 0 is `[0, window)` (the
/// existing head embedding — NOT returned here), and this returns the
/// ranges for chunks 1.. — each `window` wide with `window / 5`
/// overlap (20%, mirroring the 200/40-token prototype the 7× result
/// was measured with), until the text is covered or [`MAX_CHUNKS`] is
/// reached.
///
/// Returns an empty vec when the text fits the window. A final short
/// tail window is kept only when it adds at least `overlap` chars of
/// NEW text — a sliver shorter than the overlap is already covered by
/// the previous window.
///
/// Ranges are BYTE offsets aligned down to `char` boundaries, so
/// slicing `&text[a..b]` is always valid UTF-8. Alignment can shift a
/// boundary by at most 3 bytes, which the 20% overlap absorbs.
pub fn chunk_ranges(text: &str, window: usize) -> Vec<(usize, usize)> {
    let len = text.len();
    if window == 0 || len <= window {
        return Vec::new();
    }
    let overlap = (window / 5).max(1);
    let stride = window - overlap;
    let mut out = Vec::new();
    let mut start = stride;
    // Chunk 0 covered [0, window); each next window must add new text.
    let mut covered_to = window;
    while start < len && out.len() < MAX_CHUNKS {
        let end = (start + window).min(len);
        if end <= covered_to + overlap.min(len.saturating_sub(covered_to)) && end < len {
            // Defensive: cannot happen with stride > 0, but never loop.
            break;
        }
        // Keep a tail window only if it adds >= overlap new chars.
        if end == len && end.saturating_sub(covered_to) < overlap {
            break;
        }
        let a = floor_char_boundary(text, start);
        let b = floor_char_boundary(text, end).max(a);
        if b > a {
            out.push((a, b));
        }
        covered_to = end;
        if end == len {
            break;
        }
        start += stride;
    }
    out
}

/// Largest byte index `<= i` that is a char boundary. (Stable-Rust
/// stand-in for `str::floor_char_boundary`.)
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

    #[test]
    fn chunk_key_roundtrips_through_parent_of() {
        let rid = "019fcea7-f941-7d9d-bc96-9882a8704026";
        for idx in [1, 2, 9, 16] {
            assert_eq!(parent_of(&chunk_key(rid, idx)), rid);
        }
    }

    #[test]
    fn plain_rid_is_its_own_parent() {
        let rid = "019fcea7-f941-7d9d-bc96-9882a8704026";
        assert_eq!(parent_of(rid), rid);
    }

    #[test]
    fn non_numeric_suffix_is_not_a_chunk_key() {
        // A user-supplied key containing '#c' mid-string must not be
        // mangled — collapse must never merge distinct records.
        assert_eq!(parent_of("note#cool"), "note#cool");
        assert_eq!(parent_of("x#c"), "x#c");
        assert_eq!(parent_of("x#c1a"), "x#c1a");
    }

    #[test]
    fn collapse_keeps_best_window_per_parent_in_order() {
        let results = vec![
            ("a#c2".to_string(), 0.10),
            ("b".to_string(), 0.20),
            ("a".to_string(), 0.30),
            ("b#c1".to_string(), 0.40),
            ("c#c3".to_string(), 0.50),
        ];
        let collapsed = collapse_to_parents(results);
        assert_eq!(
            collapsed,
            vec![
                ("a".to_string(), 0.10),
                ("b".to_string(), 0.20),
                ("c".to_string(), 0.50),
            ]
        );
    }

    #[test]
    fn short_text_needs_no_chunks() {
        assert!(chunk_ranges("hello", 100).is_empty());
        let exactly = "x".repeat(100);
        assert!(chunk_ranges(&exactly, 100).is_empty());
    }

    #[test]
    fn ranges_cover_the_tail() {
        let text = "y".repeat(1000);
        let ranges = chunk_ranges(&text, 300);
        assert!(!ranges.is_empty());
        // The final range must reach the end of the text — the whole
        // point is that the tail becomes findable.
        assert_eq!(ranges.last().unwrap().1, 1000);
        // Every range is at most one window wide and non-empty.
        for (a, b) in &ranges {
            assert!(b > a && b - a <= 300);
        }
        // Consecutive windows overlap (no gaps).
        let mut covered = 300; // chunk 0
        for (a, b) in &ranges {
            assert!(*a <= covered, "gap before {a} (covered to {covered})");
            covered = covered.max(*b);
        }
        assert_eq!(covered, 1000);
    }

    #[test]
    fn tiny_tail_sliver_is_dropped() {
        // window 300, overlap 60, stride 240. len 310: chunk-1 window
        // would add only 10 new chars (< overlap) — already covered.
        let text = "z".repeat(310);
        assert!(chunk_ranges(&text, 300).is_empty());
    }

    #[test]
    fn chunk_count_is_capped() {
        let text = "w".repeat(1_000_000);
        assert!(chunk_ranges(&text, 300).len() <= MAX_CHUNKS);
    }

    #[test]
    fn ranges_respect_utf8_boundaries() {
        // Multi-byte chars straddling window edges must not panic.
        let text = "é".repeat(500); // 2 bytes each, 1000 bytes
        for (a, b) in chunk_ranges(&text, 300) {
            let _ = &text[a..b]; // would panic on a non-boundary
        }
    }
}
