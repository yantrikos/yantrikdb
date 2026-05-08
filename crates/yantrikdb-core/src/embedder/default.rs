//! `BundledEmbedder` — Slice A baseline implementation.
//!
//! Hash-trick TF-IDF in pure Rust. Deterministic, zero-allocation per
//! token, no external models to download. Produces a 384-dim L2-
//! normalized vector that gives reasonable cosine similarity for
//! lexically overlapping short texts.
//!
//! Quality is below sentence-transformer baselines (no semantic
//! understanding — only lexical), but useful enough that
//! `record_text()` / `recall_text()` are not no-ops out of the box.
//! Slice B (saga task 20 follow-up) replaces internals with
//! all-MiniLM-L6-v2 via candle-transformers.

use crate::types::Embedder;
use std::collections::HashMap;

/// Default 384-dim — matches all-MiniLM-L6-v2 (the planned Slice B
/// model), so deployments don't need to change `embedding_dim` at the
/// engine constructor when Slice B lands.
pub const BUNDLED_EMBEDDER_DIM: usize = 384;

/// **Bundled default embedder.** See [`super`] module docs for the
/// Slice A vs Slice B split and the architectural framing.
///
/// Stateless — `Default::default()` is the canonical constructor.
/// Trivially `Send + Sync` since there is no interior state to share.
#[derive(Debug, Default, Clone)]
pub struct BundledEmbedder;

impl BundledEmbedder {
    /// Construct a new bundled embedder. Equivalent to `Default::default()`.
    pub fn new() -> Self {
        Self
    }

    /// Compute the 384-dim embedding for `text` without going through the
    /// `Embedder` trait — useful when the caller already has `&BundledEmbedder`
    /// and wants to skip the boxed-error round-trip.
    pub fn embed_direct(&self, text: &str) -> Vec<f32> {
        embed_hash_tfidf(text, BUNDLED_EMBEDDER_DIM)
    }
}

impl Embedder for BundledEmbedder {
    fn embed(
        &self,
        text: &str,
    ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(self.embed_direct(text))
    }

    fn dim(&self) -> usize {
        BUNDLED_EMBEDDER_DIM
    }
}

// ── implementation ──

/// Hash-trick TF-IDF baseline. Tokenizes on Unicode word boundaries,
/// lowercases, hashes each token to a `dim`-bucket index with a weight
/// derived from log(1 + count), L2-normalizes the result. No
/// stop-word list, no IDF (single-document); the L2 normalization is
/// what makes cosine-similarity behave reasonably across documents
/// of different lengths.
fn embed_hash_tfidf(text: &str, dim: usize) -> Vec<f32> {
    // Token frequencies — count duplicates so the TF weighting is real.
    let mut tf: HashMap<String, u32> = HashMap::new();
    for token in tokenize(text) {
        *tf.entry(token).or_insert(0) += 1;
    }

    // Scatter: each token contributes log(1 + count) to its hashed
    // bucket. Sign comes from a second hash so cancellation can occur
    // (the standard hash-trick "feature_hashing with signs" pattern,
    // reduces collision bias).
    let mut v = vec![0.0f32; dim];
    for (token, count) in &tf {
        let h = stable_hash(token.as_bytes());
        let bucket = (h % dim as u64) as usize;
        // Sign bit derived from a different mixing of the hash.
        let sign: f32 = if (h.rotate_right(33).wrapping_mul(0x9e37_79b9_7f4a_7c15) & 1) == 0 {
            1.0
        } else {
            -1.0
        };
        let weight = (1.0 + *count as f32).ln();
        v[bucket] += sign * weight;
    }

    // L2 normalize so cosine similarity is meaningful across different
    // text lengths. Empty / whitespace-only text yields a zero vector
    // that we leave alone (cosine-of-zero is undefined but the engine's
    // cosine_distance_f64 returns 1.0 in that case, which is the
    // "irrelevant" answer).
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Tokenize text: split on whitespace and ASCII non-alphanumeric, drop
/// empty fragments, lowercase. ASCII-only because BundledEmbedder is
/// the lexical-baseline implementation; Unicode tokenization is
/// candle-territory for Slice B.
fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
}

/// FxHash-shaped 64-bit stable hash. Matches across runs, processes,
/// and platforms — required for embedding determinism. Not cryptographic.
fn stable_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim_is_384() {
        assert_eq!(BundledEmbedder::new().dim(), 384);
    }

    #[test]
    fn embed_returns_l2_normalized_vector() {
        let e = BundledEmbedder::new();
        let v = e.embed("Alice is the engineering lead").unwrap();
        assert_eq!(v.len(), 384);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit-norm; got {norm}");
    }

    #[test]
    fn embed_is_deterministic_across_calls() {
        // Determinism is load-bearing for cluster replication: every
        // node must compute the same vector for the same text.
        let e = BundledEmbedder::new();
        let a = e.embed("Acme Corporation").unwrap();
        let b = e.embed("Acme Corporation").unwrap();
        assert_eq!(a, b, "same input must yield same vector");
    }

    #[test]
    fn embed_empty_string_returns_zero_vector() {
        let e = BundledEmbedder::new();
        let v = e.embed("").unwrap();
        assert!(v.iter().all(|x| *x == 0.0), "empty text yields zero vector");
    }

    #[test]
    fn lexically_similar_texts_are_more_similar_than_unrelated() {
        // Sanity: "Alice met Bob at the cafe" vs "Alice and Bob had
        // coffee" should be closer than vs "the stock market is volatile".
        // Cosine similarity = dot product (since inputs are L2-normalized).
        let e = BundledEmbedder::new();
        let a = e.embed("Alice met Bob at the cafe").unwrap();
        let b = e.embed("Alice and Bob had coffee").unwrap();
        let c = e.embed("the stock market is volatile").unwrap();

        let cos = |x: &[f32], y: &[f32]| -> f32 {
            x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
        };
        let sim_ab = cos(&a, &b);
        let sim_ac = cos(&a, &c);
        assert!(
            sim_ab > sim_ac,
            "lexically similar texts should be more similar than unrelated; \
             sim_ab={sim_ab} sim_ac={sim_ac}"
        );
    }

    #[test]
    fn case_insensitive_tokenization() {
        let e = BundledEmbedder::new();
        let a = e.embed("ALICE Bob").unwrap();
        let b = e.embed("alice bob").unwrap();
        assert_eq!(a, b, "case-insensitive: ALICE and alice hash to same bucket");
    }
}
