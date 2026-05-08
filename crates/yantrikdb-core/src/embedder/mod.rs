//! Default `Embedder` implementations bundled with the engine.
//!
//! Without these, the engine's `record_text()` / `recall_text()` methods
//! require the user to call `set_embedder()` with an external embedder
//! (e.g. sentence-transformers via PyO3). Bundling a default makes the
//! engine "just work" out of the box, matching the SQLite-shape no-extra-
//! installation contract.
//!
//! See [`docs/phase_4_3_design.md`](../../../../docs/phase_4_3_design.md)
//! and yantrikdb memory rid `019e0686-824c-7e70-9c87-201f8fffbdac` for
//! the architectural decision (Pranab course-correction 2026-05-08).
//!
//! # Slice A vs Slice B
//!
//! **Slice A (current):** A `BundledEmbedder` that uses a deterministic
//! hash-trick TF-IDF projection. Zero ML dependencies, fast, returns a
//! 384-dim vector. Useful for short-text similarity at the API-contract
//! level — proves the wiring. **Not** a replacement for a real sentence
//! embedder for retrieval-quality use cases.
//!
//! **Slice B (saga task 20 follow-up):** Replace the internals with
//! all-MiniLM-L6-v2 inference via candle-transformers, weights bundled.
//! Public API stays the same; quality jumps to MiniLM-baseline.

#[cfg(feature = "bundled-embedder")]
pub mod default;

#[cfg(feature = "bundled-embedder")]
pub use default::{BundledEmbedder, BUNDLED_EMBEDDER_DIM};

#[cfg(feature = "embedder-download")]
pub mod downloaded;

#[cfg(feature = "embedder-download")]
pub use downloaded::DownloadedEmbedder;
