//! Detecting the embedder's input window — because silent truncation
//! is silent retrieval loss.
//!
//! A transformer embedder has a fixed input window (all-MiniLM-L6-v2:
//! 256 tokens). Text beyond it is dropped before the forward pass. The
//! record is still stored intact and still looks healthy in `stats()`,
//! but the vector describes only its head — so the tail is unfindable.
//! That is the same stored-active-unfindable shape as the HNSW orphan
//! bug, and it was measured on a production install at **73% of
//! records**, where a verbatim fragment from a record's end retrieved
//! its own parent 8% of the time (start of the same records: 28%).
//!
//! The [`Embedder`](crate::types::Embedder) trait cannot simply declare
//! its window: an implementation may be a BYO ONNX model, a hosted API,
//! or a Python callable the engine cannot introspect. So the engine
//! DETECTS one empirically, the same way this defect was found — build
//! two texts identical up to position `n` and differing after it. If
//! the embedder returns (near-)identical vectors, everything past `n`
//! was ignored.
//!
//! The probe is opt-in and one-shot: it costs a handful of embed calls
//! and is never run on the write path automatically. Operators call it,
//! or the value is reported once a caller asks for it.

use std::sync::atomic::Ordering;

use crate::error::Result;

use super::YantrikDB;

/// Two embeddings are "the same vector" when every component matches
/// this closely.
///
/// Deliberately near-exact rather than a cosine threshold. A loose
/// threshold cannot tell TRUNCATION from DILUTION: a three-word suffix
/// appended to a very long prefix barely moves any embedder's output,
/// so `cosine > 0.9999` reports truncation for a model that read every
/// character. Under real truncation the model receives byte-identical
/// input and returns a bit-identical vector, so equality is the honest
/// test. (Found by the mock-embedder test, which reported a window on
/// an embedder that has none.)
const SAME_VECTOR_EPS: f32 = 1e-6;
/// Probe bounds, in characters. The lower bound is below any real
/// window; the upper bound is past every embedder in common use.
const PROBE_MIN: usize = 128;
const PROBE_MAX: usize = 65_536;
/// Sentinel stored when the probe found no truncation at all.
pub(crate) const NO_TRUNCATION: usize = usize::MAX;

/// Did the embedder return the same vector for both inputs — i.e. did
/// it see the same bytes?
fn same_vector(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| (x - y).abs() <= SAME_VECTOR_EPS)
}

impl YantrikDB {
    /// Does content at character offset `n` still reach the embedder?
    ///
    /// Two texts sharing a filler prefix of length `n` and differing
    /// only after it. Near-identical vectors mean the suffix was never
    /// seen.
    fn suffix_is_seen(&self, n: usize) -> Result<bool> {
        let filler = "the quick brown fox jumps over the lazy dog ".repeat(n / 44 + 1);
        let head = &filler[..n.min(filler.len())];
        let a = self.embed(&format!("{head} alpha zulu quasar"))?;
        let b = self.embed(&format!("{head} omega tundra basalt"))?;
        Ok(!same_vector(&a, &b))
    }

    /// Empirically detect the embedder's usable input window, in
    /// characters. `None` means no truncation was detected within the
    /// probe range; `Some(n)` is the approximate budget past which text
    /// stops affecting the vector.
    ///
    /// Characters rather than tokens deliberately: the engine has no
    /// access to a third-party embedder's tokenizer, and a character
    /// budget is the honest unit for the warning it powers. Expect it
    /// to land near `window_tokens * ~4`.
    pub fn detect_embedder_window(&self) -> Result<Option<usize>> {
        // Cheap exit: if content at the far end is still seen, nothing
        // is being dropped anywhere below it.
        if self.suffix_is_seen(PROBE_MAX)? {
            self.embedder_window_chars
                .store(NO_TRUNCATION, Ordering::Relaxed);
            self.persist_embedder_window(NO_TRUNCATION);
            return Ok(None);
        }
        // A truncating embedder: binary search the boundary between
        // "suffix still seen" and "suffix ignored".
        let (mut lo, mut hi) = (PROBE_MIN, PROBE_MAX);
        if !self.suffix_is_seen(lo)? {
            // Truncates below even the smallest probe — report the
            // floor rather than pretending to more precision.
            self.embedder_window_chars.store(lo, Ordering::Relaxed);
            self.persist_embedder_window(lo);
            return Ok(Some(lo));
        }
        while hi - lo > 64 {
            let mid = lo + (hi - lo) / 2;
            if self.suffix_is_seen(mid)? {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        self.embedder_window_chars.store(lo, Ordering::Relaxed);
        // The probe is ~24 embed calls — persist the answer (keyed to
        // the embedder digest) so a restart neither repays that cost
        // nor silently deactivates chunking.
        self.persist_embedder_window(lo);
        Ok(Some(lo))
    }

    /// The detected window, if [`Self::detect_embedder_window`] has run.
    pub fn embedder_window(&self) -> Option<usize> {
        match self.embedder_window_chars.load(Ordering::Relaxed) {
            0 => None,
            NO_TRUNCATION => None,
            n => Some(n),
        }
    }

    /// Writes since boot whose text exceeded the detected window — i.e.
    /// records stored with a vector that describes only their head.
    pub fn embedder_truncated_write_count(&self) -> u64 {
        self.embedder_truncated_writes.load(Ordering::Relaxed)
    }

    /// Count a write whose text cannot fit the embedder's window.
    ///
    /// Called from the engine-side embed path. Silent is the one thing
    /// this must not be: the record is about to be stored intact and
    /// indexed from its head only, and nothing downstream can tell.
    pub(crate) fn note_possible_truncation(&self, text_len: usize) {
        let window = self.embedder_window_chars.load(Ordering::Relaxed);
        if window == 0 || window == NO_TRUNCATION || text_len <= window {
            return;
        }
        let n = self
            .embedder_truncated_writes
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        // Loud once, then on a widening cadence: an operator ingesting a
        // long corpus should see this without the log becoming the
        // corpus.
        if n == 1 || n == 10 || n % 100 == 0 {
            eprintln!(
                "yantrikdb: WARNING — text of {text_len} chars exceeds the detected \
                 embedder window (~{window} chars); the tail is stored but NOT embedded, \
                 so it cannot be retrieved. {n} such write(s) since boot. \
                 Split long records, or attach an embedder with a larger window."
            );
        }
    }
}
