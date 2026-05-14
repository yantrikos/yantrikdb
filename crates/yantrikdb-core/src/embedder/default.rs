//! `BundledEmbedder` — Slice B (saga task 20, 2026-05-08).
//!
//! Production default embedder shipped with the engine: a static
//! lookup-table embedding distilled from `baai/bge-base-en-v1.5` via
//! [model2vec](https://github.com/MinishLab/model2vec). Inference =
//! tokenize → embedding-table lookup → mean-pool → L2-normalize. No
//! transformer inference, no GPU, ~500× faster than sentence-
//! transformers on CPU.
//!
//! # Quality
//!
//! Empirical quality eval (yantrikdb memory rid `019e06a7`, 30
//! yantrikdb-shaped memories × 10 queries):
//!
//! | Embedder        | R@5  | R@10 | MRR  |
//! |-----------------|------|------|------|
//! | hash-trick (v1) | 0.75 | 0.85 | 0.71 |
//! | **potion-2M**   | 0.90 | 0.90 | 0.78 |
//! | MiniLM-L6 (gold)| 0.95 | 1.00 | 0.90 |
//!
//! # Why bundled by default
//!
//! See yantrikdb memory rid `019e0686`. The engine's `record_text()` /
//! `recall_text()` API is half-broken without an embedder — bundling
//! one ships the SQLite-shape "just works" contract. Users wanting
//! full sentence-transformer quality call `set_embedder()` with
//! their own implementation; the bundled default stays out of the way.
//!
//! # Model files
//!
//! Bundled via `include_bytes!` from `crates/yantrikdb-core/assets/potion-base-2M/`:
//! `model.safetensors` (~7.2 MB), `tokenizer.json` (~668 KB),
//! `config.json`, `modules.json`. ~7.9 MB total committed footprint.
//!
//! # Why dim=64
//!
//! `potion-base-2M`'s output dimension is 64 (fewer params per vocab
//! entry → smaller model). Engine deployments using the bundled
//! default initialize `YantrikDB::new(path, 64)` (or use the
//! `with_default()` constructor sugar). Existing dim=384 deployments
//! continue to work — they just don't get auto-attached and must
//! call `set_embedder()` themselves (same as today).
//!
//! # Slice C (follow-up)
//!
//! `db.set_embedder_named("potion-base-8M")` — downloads the larger,
//! higher-quality model variants from `yantrikos/yantrikdb-models`
//! GitHub Releases on first call, caches under the user's data dir.
//! Not in this commit; tracked under saga task 20 Slice C.

use crate::types::Embedder;

/// Output dimension of the bundled `potion-base-2M` model. The engine's
/// `with_default()` constructor opens at this dim; auto-attach in
/// `YantrikDB::new(path, dim)` matches on this value.
pub const BUNDLED_EMBEDDER_DIM: usize = 64;

// ── Bundled model files ──
//
// `include_bytes!` paths are relative to the source file. The
// `assets/potion-base-2M/` directory is committed to the repo so
// `cargo build` and downstream `cargo install yantrikdb` both bake
// the bytes into the final binary — no network at install or
// runtime.
const POTION_2M_MODEL: &[u8] = include_bytes!("../../assets/potion-base-2M/model.safetensors");
const POTION_2M_TOKENIZER: &[u8] = include_bytes!("../../assets/potion-base-2M/tokenizer.json");
const POTION_2M_CONFIG: &[u8] = include_bytes!("../../assets/potion-base-2M/config.json");
const POTION_2M_MODULES: &[u8] = include_bytes!("../../assets/potion-base-2M/modules.json");

/// **Bundled default embedder.** See [module docs][super] for the
/// design rationale and the Slice A → B → C arc.
///
/// Lazy: the model is only loaded on first call to [`embed`] or
/// [`embed_batch`]. Subsequent calls reuse the loaded `StaticModel`.
/// Cloning is cheap — clones share the same `Arc<StaticModel>` once
/// loaded.
#[derive(Clone)]
pub struct BundledEmbedder {
    inner: std::sync::Arc<once_cell_lite::Lazy>,
}

/// Tiny private shim so we can hand-roll OnceCell semantics with
/// `std::sync::OnceLock` without depending on `once_cell`. (The std
/// type is what we need; this module exists just to keep the public
/// `BundledEmbedder` struct readable.)
mod once_cell_lite {
    use model2vec_rs::model::StaticModel;
    use std::sync::OnceLock;

    pub struct Lazy {
        // OnceLock<Result<StaticModel, String>> so a load failure is
        // also memoized — we don't retry forever on a broken bundle.
        cell: OnceLock<Result<StaticModel, String>>,
    }

    impl Lazy {
        pub fn new() -> Self {
            Self {
                cell: OnceLock::new(),
            }
        }

        /// Get-or-init the model. Extracts the bundled bytes to a
        /// process-local temp directory on first call, then loads via
        /// `model2vec_rs::StaticModel::from_pretrained`.
        pub fn get(&self) -> Result<&StaticModel, &str> {
            let result = self.cell.get_or_init(|| init_model());
            match result {
                Ok(m) => Ok(m),
                Err(e) => Err(e.as_str()),
            }
        }
    }

    /// Extract the four bundled files into a temp dir and load. Errors
    /// are returned as owned strings so the OnceLock's stored value is
    /// `'static` (no borrows escape the closure).
    fn init_model() -> Result<StaticModel, String> {
        use std::io::Write;
        // Per-process unique temp dir so concurrent yantrikdb processes
        // on the same host don't race on file writes.
        let temp = std::env::temp_dir()
            .join("yantrikdb")
            .join(format!("potion-2M-{}", std::process::id()));
        std::fs::create_dir_all(&temp)
            .map_err(|e| format!("create temp dir {}: {e}", temp.display()))?;

        for (name, bytes) in [
            ("model.safetensors", super::POTION_2M_MODEL),
            ("tokenizer.json", super::POTION_2M_TOKENIZER),
            ("config.json", super::POTION_2M_CONFIG),
            ("modules.json", super::POTION_2M_MODULES),
        ] {
            let path = temp.join(name);
            // Write only if the file doesn't exist with the right
            // length already — covers re-init within the same process
            // (shouldn't happen with OnceLock but is cheap insurance).
            let need_write = match std::fs::metadata(&path) {
                Ok(m) => m.len() != bytes.len() as u64,
                Err(_) => true,
            };
            if need_write {
                let mut f = std::fs::File::create(&path)
                    .map_err(|e| format!("create {}: {e}", path.display()))?;
                f.write_all(bytes)
                    .map_err(|e| format!("write {}: {e}", path.display()))?;
            }
        }

        StaticModel::from_pretrained(&temp, None, None, None)
            .map_err(|e| format!("model2vec_rs load: {e}"))
    }
}

impl BundledEmbedder {
    /// Construct a new bundled embedder. Cheap — the model is not
    /// loaded until the first call to [`embed`] / [`embed_batch`].
    pub fn new() -> Self {
        Self {
            inner: std::sync::Arc::new(once_cell_lite::Lazy::new()),
        }
    }
}

impl Default for BundledEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for BundledEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BundledEmbedder")
            .field("model", &"potion-base-2M")
            .field("dim", &BUNDLED_EMBEDDER_DIM)
            .finish()
    }
}

impl Embedder for BundledEmbedder {
    fn embed(
        &self,
        text: &str,
    ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let model = self
            .inner
            .get()
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.to_string().into() })?;
        Ok(model.encode_single(text))
    }

    fn embed_batch(
        &self,
        texts: &[&str],
    ) -> std::result::Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        let model = self
            .inner
            .get()
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.to_string().into() })?;
        let owned: Vec<String> = texts.iter().map(|s| (*s).to_string()).collect();
        Ok(model.encode(&owned))
    }

    fn dim(&self) -> usize {
        BUNDLED_EMBEDDER_DIM
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim_is_64() {
        assert_eq!(BundledEmbedder::new().dim(), BUNDLED_EMBEDDER_DIM);
        assert_eq!(BUNDLED_EMBEDDER_DIM, 64);
    }

    #[test]
    fn embed_returns_64_dim_vector() {
        let e = BundledEmbedder::new();
        let v = e.embed("Alice is the engineering lead at Acme").unwrap();
        assert_eq!(v.len(), 64);
    }

    #[test]
    fn embed_returns_l2_normalized_vector() {
        // potion-base-2M's modules.json includes a Normalize step, so
        // outputs should be unit norm. Verifies the model.config's
        // `normalize: true` is honored end-to-end.
        let e = BundledEmbedder::new();
        let v = e.embed("Project Atlas launches in March").unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "expected unit-norm; got {norm}");
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
    fn embed_batch_matches_single() {
        let e = BundledEmbedder::new();
        let inputs = ["one fish", "two fish", "red fish"];
        let single: Vec<Vec<f32>> = inputs.iter().map(|t| e.embed(t).unwrap()).collect();
        let batch = e.embed_batch(&inputs).unwrap();
        assert_eq!(single.len(), batch.len());
        for (s, b) in single.iter().zip(batch.iter()) {
            assert_eq!(s.len(), b.len(), "dim mismatch single vs batch");
            for (x, y) in s.iter().zip(b.iter()) {
                assert!(
                    (x - y).abs() < 1e-5,
                    "single vs batch divergence: {x} vs {y}"
                );
            }
        }
    }

    #[test]
    fn semantically_similar_texts_score_higher_than_unrelated() {
        // Beyond the lexical baseline of Slice A — potion-2M does
        // semantic matching. "engineering lead" should be closer to
        // "tech leader" than to "lunch menu".
        let e = BundledEmbedder::new();
        let a = e.embed("Alice is the engineering lead").unwrap();
        let b = e.embed("Alice runs the technical team").unwrap();
        let c = e.embed("the lunch menu has pasta today").unwrap();

        let cos =
            |x: &[f32], y: &[f32]| -> f32 { x.iter().zip(y.iter()).map(|(a, b)| a * b).sum() };
        let sim_ab = cos(&a, &b);
        let sim_ac = cos(&a, &c);
        assert!(
            sim_ab > sim_ac,
            "semantic match should beat unrelated; sim_ab={sim_ab} sim_ac={sim_ac}"
        );
    }
}
