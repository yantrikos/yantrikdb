//! Slice C — runtime-downloaded embedders from
//! [`yantrikos/yantrikdb-models`](https://github.com/yantrikos/yantrikdb-models).
//!
//! Compiled only when the `embedder-download` feature is on (off by
//! default — keeps the slim build slim, keeps WASM builds clean).
//!
//! # Contract
//!
//! ```ignore
//! let mut db = YantrikDB::new("memory.db", 256)?;
//! db.set_embedder_named("potion-base-8M")?;
//! // First call: downloads tarball from GitHub Releases, verifies
//! //             SHA-256, extracts to ~/.cache/yantrikdb/models/...,
//! //             loads via model2vec-rs, calls set_embedder().
//! // Subsequent calls (same process or cold restart): hits the cache.
//! ```
//!
//! The registry below maps `name → (release_tag, sha256, expected_dim)`
//! so a hash mismatch on the wire fails fast and the user falls back to
//! `set_embedder()` with their own implementation.

use crate::error::{Result, YantrikDbError};
use crate::types::Embedder;
use model2vec_rs::model::StaticModel;
use std::path::PathBuf;

/// Static registry of downloadable model variants. Each entry pins the
/// exact GitHub Release tag + SHA-256 + dim the engine was built
/// against, so a deployed engine won't accidentally pick up a
/// different artifact if upstream re-uploads.
struct DownloadableModel {
    /// Release tag in `yantrikos/yantrikdb-models`.
    release_tag: &'static str,
    /// Asset filename within the release.
    asset: &'static str,
    /// Hex-encoded SHA-256 of the `.tar.gz` asset bytes.
    sha256: &'static str,
    /// Output dim of the loaded model. Used for the auto-attach guard
    /// (engine's `embedding_dim` must match).
    dim: usize,
}

/// **v0.7.x Slice C registry.** Hardcoded so the engine binary
/// guarantees provenance of every named model. Adding a new variant
/// requires (1) a release on `yantrikos/yantrikdb-models` and (2) an
/// engine release that includes its constants here.
fn registry(name: &str) -> Option<DownloadableModel> {
    match name {
        // potion-base-8M — 256-dim, ~92% MiniLM. Sweet spot for users
        // who want better quality than the bundled potion-base-2M
        // without paying for a transformer.
        "potion-base-8M" => Some(DownloadableModel {
            release_tag: "v0.1.0",
            asset: "potion-base-8M.tar.gz",
            sha256: "89dd960591c4fa0c7f7a45ed4cb94167ce4e09886f39bae008b8072b42439ac5",
            dim: 256,
        }),
        // potion-base-32M — 512-dim, ~95% MiniLM. Largest English
        // variant before multilingual jumps to 488 MB.
        "potion-base-32M" => Some(DownloadableModel {
            release_tag: "v0.1.0",
            asset: "potion-base-32M.tar.gz",
            sha256: "428163e9aa596b38bf98f6d41ff7cb1b3d7e6d21f58e1edc8124cd9d180f93ad",
            dim: 512,
        }),
        _ => None,
    }
}

/// Compute the URL for a model's release asset.
fn asset_url(model: &DownloadableModel) -> String {
    format!(
        "https://github.com/yantrikos/yantrikdb-models/releases/download/{}/{}",
        model.release_tag, model.asset
    )
}

/// Compute the cache path for an extracted model.
fn cache_dir_for(name: &str, release_tag: &str) -> Result<PathBuf> {
    let base = dirs::cache_dir().ok_or_else(|| {
        YantrikDbError::InvalidInput(
            "could not resolve user cache dir; set XDG_CACHE_HOME or HOME".into(),
        )
    })?;
    Ok(base.join("yantrikdb").join("models").join(format!("{name}-{release_tag}")))
}

/// Returns true iff the cache dir exists with all four expected files
/// at non-zero size. Cheap directory-stat check; the engine doesn't
/// re-verify SHA-256 on each cache hit (we trust the filesystem to
/// preserve what we wrote).
fn cache_is_populated(dir: &std::path::Path) -> bool {
    for f in &["model.safetensors", "tokenizer.json", "config.json", "modules.json"] {
        match std::fs::metadata(dir.join(f)) {
            Ok(m) if m.len() > 0 => continue,
            _ => return false,
        }
    }
    true
}

/// Download the asset, verify SHA-256, extract into the cache dir.
/// Atomic via tmp dir + rename: a partially-extracted cache dir is
/// never left visible.
fn fetch_and_extract(model: &DownloadableModel, name: &str) -> Result<PathBuf> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let final_dir = cache_dir_for(name, model.release_tag)?;
    if cache_is_populated(&final_dir) {
        return Ok(final_dir);
    }

    let url = asset_url(model);
    tracing::info!(
        target: "yantrikdb::embedder::download",
        name = name,
        url = %url,
        sha256 = model.sha256,
        "downloading model artifact"
    );

    // Buffer the full tarball into memory (28-125 MB). Stream-decompress
    // would save peak RSS, but the simpler path is fine for the user-
    // initiated, infrequent case set_embedder_named is called in.
    let resp = ureq::get(&url)
        .set("User-Agent", concat!("yantrikdb/", env!("CARGO_PKG_VERSION")))
        .call()
        .map_err(|e| YantrikDbError::InvalidInput(format!("download {url}: {e}")))?;

    let mut bytes = Vec::with_capacity(64 * 1024 * 1024);
    resp.into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| YantrikDbError::InvalidInput(format!("read body: {e}")))?;

    // SHA-256 verify before we trust the bytes for anything else.
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let actual = hex::encode(hasher.finalize());
    if actual != model.sha256 {
        return Err(YantrikDbError::InvalidInput(format!(
            "model {name}: SHA-256 mismatch — expected {}, got {actual}. Refusing to load \
             (corrupted download or upstream tampering). Try again or fall back to \
             set_embedder() with your own implementation.",
            model.sha256,
        )));
    }

    // Extract under a temp sibling dir, then atomic rename. Avoids
    // exposing a half-written cache to a concurrent process.
    let parent = final_dir.parent().ok_or_else(|| {
        YantrikDbError::InvalidInput("cache dir has no parent".into())
    })?;
    std::fs::create_dir_all(parent)
        .map_err(|e| YantrikDbError::InvalidInput(format!("mkdir {}: {e}", parent.display())))?;
    let tmp_dir = parent.join(format!(
        "{}-{}.tmp.{}",
        name,
        model.release_tag,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp_dir); // cleanup any prior crash
    std::fs::create_dir_all(&tmp_dir)
        .map_err(|e| YantrikDbError::InvalidInput(format!("mkdir tmp {}: {e}", tmp_dir.display())))?;

    let gz = flate2::read::GzDecoder::new(&bytes[..]);
    let mut archive = tar::Archive::new(gz);
    // The tarball entries are scoped under e.g. `potion-base-8M/...`.
    // We want the inner files at the cache dir root, so we strip the
    // leading directory component.
    for entry in archive.entries().map_err(|e| {
        YantrikDbError::InvalidInput(format!("tar open: {e}"))
    })? {
        let mut entry = entry.map_err(|e| {
            YantrikDbError::InvalidInput(format!("tar entry: {e}"))
        })?;
        let path = entry.path().map_err(|e| {
            YantrikDbError::InvalidInput(format!("tar path: {e}"))
        })?;
        // Strip the leading dir component if present.
        let stripped: PathBuf = path
            .components()
            .skip(1)
            .collect();
        if stripped.as_os_str().is_empty() {
            continue;
        }
        let dest = tmp_dir.join(&stripped);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        entry.unpack(&dest).map_err(|e| {
            YantrikDbError::InvalidInput(format!(
                "tar unpack {}: {e}",
                dest.display()
            ))
        })?;
    }

    // Final move: rename tmp_dir → final_dir. If a concurrent process
    // beat us to it, both renames may run; the second one fails and we
    // accept whatever's already there.
    if !cache_is_populated(&final_dir) {
        std::fs::rename(&tmp_dir, &final_dir).map_err(|e| {
            YantrikDbError::InvalidInput(format!(
                "rename {} -> {}: {e}",
                tmp_dir.display(),
                final_dir.display()
            ))
        })?;
    } else {
        // Cleanup our tmp on the rare race-loss path.
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    if !cache_is_populated(&final_dir) {
        return Err(YantrikDbError::InvalidInput(format!(
            "after extract, expected files missing in {}",
            final_dir.display()
        )));
    }

    tracing::info!(
        target: "yantrikdb::embedder::download",
        name = name,
        cache_dir = %final_dir.display(),
        "model ready"
    );
    Ok(final_dir)
}

/// Wrapper `Embedder` impl that delegates to a loaded `StaticModel`.
/// Sister of `BundledEmbedder` but loads from the user's filesystem
/// cache rather than `include_bytes!`.
pub struct DownloadedEmbedder {
    model: std::sync::Arc<StaticModel>,
    dim: usize,
}

impl DownloadedEmbedder {
    /// **Saga task 20 Slice C.** Resolve `name` against the static
    /// registry; download + verify + extract on cache miss; load via
    /// `model2vec-rs`. Returns an `Embedder` impl ready for
    /// `db.set_embedder()`.
    pub fn fetch(name: &str) -> Result<Self> {
        let model = registry(name).ok_or_else(|| {
            YantrikDbError::InvalidInput(format!(
                "unknown embedder name {name:?}; known: potion-base-8M, potion-base-32M"
            ))
        })?;
        let dir = fetch_and_extract(&model, name)?;
        let static_model = StaticModel::from_pretrained(&dir, None, None, None).map_err(|e| {
            YantrikDbError::InvalidInput(format!(
                "model2vec_rs load from {}: {e}",
                dir.display()
            ))
        })?;
        Ok(Self {
            model: std::sync::Arc::new(static_model),
            dim: model.dim,
        })
    }
}

impl Embedder for DownloadedEmbedder {
    fn embed(
        &self,
        text: &str,
    ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(self.model.encode_single(text))
    }

    fn embed_batch(
        &self,
        texts: &[&str],
    ) -> std::result::Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        let owned: Vec<String> = texts.iter().map(|s| (*s).to_string()).collect();
        Ok(self.model.encode(&owned))
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_unknown_name_returns_none() {
        assert!(registry("definitely-not-real").is_none());
    }

    #[test]
    fn registry_known_models_have_consistent_dim() {
        let m8 = registry("potion-base-8M").expect("registered");
        assert_eq!(m8.dim, 256);
        let m32 = registry("potion-base-32M").expect("registered");
        assert_eq!(m32.dim, 512);
        // SHA-256s are 64 hex chars.
        assert_eq!(m8.sha256.len(), 64);
        assert_eq!(m32.sha256.len(), 64);
    }

    #[test]
    fn asset_url_format_matches_release_pattern() {
        let m = registry("potion-base-8M").unwrap();
        let url = asset_url(&m);
        assert_eq!(
            url,
            "https://github.com/yantrikos/yantrikdb-models/releases/download/v0.1.0/potion-base-8M.tar.gz"
        );
    }

    // Live download test: opt-in via `YANTRIKDB_TEST_LIVE_DOWNLOAD=1`
    // env var since CI shouldn't depend on network access. Locally
    // useful as a smoke test when iterating on the download path.
    #[test]
    fn live_download_potion_8m_smoke() {
        if std::env::var_os("YANTRIKDB_TEST_LIVE_DOWNLOAD").is_none() {
            return; // skip silently
        }
        let e = DownloadedEmbedder::fetch("potion-base-8M")
            .expect("fetch potion-base-8M live");
        assert_eq!(e.dim(), 256);
        let v = e.embed("Alice met Acme yesterday").unwrap();
        assert_eq!(v.len(), 256);
        // L2-normalized output (model2vec config has normalize=true).
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "expected unit norm; got {norm}");
    }
}
