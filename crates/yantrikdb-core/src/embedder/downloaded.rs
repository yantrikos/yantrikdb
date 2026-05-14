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
        // variant before multilingual jumps to ~460 MB.
        "potion-base-32M" => Some(DownloadableModel {
            release_tag: "v0.1.0",
            asset: "potion-base-32M.tar.gz",
            sha256: "428163e9aa596b38bf98f6d41ff7cb1b3d7e6d21f58e1edc8124cd9d180f93ad",
            dim: 512,
        }),
        // potion-multilingual-128M — 256-dim, BGE-M3 tokenizer, 101
        // languages. ~460 MB tarball; large but a single download
        // unlocks multilingual semantic recall without needing a
        // transformer or external sentence-transformers install. Same
        // model2vec static-embedding architecture as the English
        // variants, just trained against a multilingual corpus +
        // tokenizer. dim=256 matches potion-base-8M, so existing
        // dim=256 callers can swap to this without reopening the DB.
        //
        // Added in engine v0.7.9 in response to yantrikdb-hermes-plugin
        // Issue #1 (alienos) — first real user request for multilingual
        // support. Coordinated with plugin v0.4.0 env-var embedder
        // selection.
        "potion-multilingual-128M" => Some(DownloadableModel {
            release_tag: "v0.2.0",
            asset: "potion-multilingual-128M.tar.gz",
            sha256: "bbd9b15fa1303538206911f82c85b5d52e3fa0a334479f988e8a370b0e2e7a52",
            dim: 256,
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
    Ok(base
        .join("yantrikdb")
        .join("models")
        .join(format!("{name}-{release_tag}")))
}

/// Returns true iff the cache dir exists with all four expected files
/// at non-zero size. Cheap directory-stat check; the engine doesn't
/// re-verify SHA-256 on each cache hit (we trust the filesystem to
/// preserve what we wrote).
fn cache_is_populated(dir: &std::path::Path) -> bool {
    for f in &[
        "model.safetensors",
        "tokenizer.json",
        "config.json",
        "modules.json",
    ] {
        match std::fs::metadata(dir.join(f)) {
            Ok(m) if m.len() > 0 => continue,
            _ => return false,
        }
    }
    true
}

/// Extract a `.tar.gz` byte slice into `dest_dir`, handling both
/// tarball layouts our model artifacts ship in:
///   - **v0.1.0 layout** (potion-base-8M, potion-base-32M): files
///     nested under a top-level directory matching the model name,
///     e.g. `potion-base-8M/model.safetensors`. The leading directory
///     component must be stripped so files land directly in `dest_dir`.
///   - **v0.2.0 layout** (potion-multilingual-128M): files at the
///     archive root, e.g. `model.safetensors`. No stripping needed.
///
/// Per-entry: strip the leading component only when the path has 2+
/// components. Single-component paths are written as-is. Directory
/// entries are skipped (parent dirs are created as needed when files
/// are unpacked).
///
/// Closes yantrikos/yantrikdb#15 — pre-fix this code unconditionally
/// stripped the leading component, which silently produced an empty
/// cache dir for v0.2.0-layout artifacts because every entry had 1
/// component, `skip(1)` produced an empty path, and the `continue`
/// skipped every file.
fn extract_tarball_to(bytes: &[u8], dest_dir: &std::path::Path) -> Result<()> {
    let gz = flate2::read::GzDecoder::new(bytes);
    let mut archive = tar::Archive::new(gz);
    for entry in archive
        .entries()
        .map_err(|e| YantrikDbError::InvalidInput(format!("tar open: {e}")))?
    {
        let mut entry =
            entry.map_err(|e| YantrikDbError::InvalidInput(format!("tar entry: {e}")))?;
        // Skip pure directory entries — we create dirs as needed when
        // unpacking files.
        if entry.header().entry_type().is_dir() {
            continue;
        }
        let path = entry
            .path()
            .map_err(|e| YantrikDbError::InvalidInput(format!("tar path: {e}")))?;

        let n_components = path.components().count();
        let stripped: PathBuf = if n_components >= 2 {
            path.components().skip(1).collect()
        } else {
            path.into_owned()
        };

        if stripped.as_os_str().is_empty() {
            continue;
        }
        let dest = dest_dir.join(&stripped);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        entry.unpack(&dest).map_err(|e| {
            YantrikDbError::InvalidInput(format!("tar unpack {}: {e}", dest.display()))
        })?;
    }
    Ok(())
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
        .set(
            "User-Agent",
            concat!("yantrikdb/", env!("CARGO_PKG_VERSION")),
        )
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
    let parent = final_dir
        .parent()
        .ok_or_else(|| YantrikDbError::InvalidInput("cache dir has no parent".into()))?;
    std::fs::create_dir_all(parent)
        .map_err(|e| YantrikDbError::InvalidInput(format!("mkdir {}: {e}", parent.display())))?;
    let tmp_dir = parent.join(format!(
        "{}-{}.tmp.{}",
        name,
        model.release_tag,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp_dir); // cleanup any prior crash
    std::fs::create_dir_all(&tmp_dir).map_err(|e| {
        YantrikDbError::InvalidInput(format!("mkdir tmp {}: {e}", tmp_dir.display()))
    })?;

    // **Closes #26.** Restored the call to extract_tarball_to that PR #23's
    // refactor accidentally swallowed. The helper was correctly defined +
    // unit-tested for both v0.1.0-prefix and v0.2.0-files-at-root layouts,
    // but never invoked from fetch_and_extract — so cache_is_populated()
    // always failed and set_embedder_named() returned "expected files
    // missing" for ALL named models in v0.7.13/v0.7.14, not just the
    // multilingual one. Same class of gap as v0.7.9's user-side-smoke
    // failure that filed #15 in the first place. The cleanup-on-error
    // arm leaves no half-written tmp_dir lying around if extraction
    // fails partway.
    extract_tarball_to(&bytes, &tmp_dir).map_err(|e| {
        let _ = std::fs::remove_dir_all(&tmp_dir);
        e
    })?;

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
                "unknown embedder name {name:?}; known: \
                 potion-base-8M, potion-base-32M, potion-multilingual-128M"
            ))
        })?;
        let dir = fetch_and_extract(&model, name)?;
        let static_model = StaticModel::from_pretrained(&dir, None, None, None).map_err(|e| {
            YantrikDbError::InvalidInput(format!("model2vec_rs load from {}: {e}", dir.display()))
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

    #[test]
    fn registry_includes_potion_multilingual_128m() {
        let m = registry("potion-multilingual-128M").expect("registered");
        assert_eq!(m.dim, 256);
        assert_eq!(m.release_tag, "v0.2.0");
        assert_eq!(m.sha256.len(), 64);
    }

    /// Helper: build an in-memory .tar.gz with the given (path, bytes)
    /// entries. Used by the layout-agnostic extractor tests below.
    fn build_tar_gz(entries: &[(&str, &[u8])]) -> Vec<u8> {
        use std::io::Write;
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        {
            let mut builder = tar::Builder::new(&mut gz);
            for (path, content) in entries {
                let mut header = tar::Header::new_gnu();
                header.set_size(content.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                builder
                    .append_data(&mut header, path, *content)
                    .expect("tar append");
            }
            builder.finish().expect("tar finish");
        }
        gz.finish().expect("gz finish")
    }

    /// Closes yantrikos/yantrikdb#15. The v0.2.0 multilingual tarball
    /// ships files at the archive root (`model.safetensors`, not
    /// `potion-multilingual-128M/model.safetensors`). Pre-fix the
    /// extractor unconditionally stripped the leading component, so
    /// every entry's path became empty and was skipped, leaving the
    /// cache dir empty. This test exercises the files-at-root layout
    /// to ensure the fix actually works for the multilingual case.
    #[test]
    fn extract_tarball_files_at_root_layout() {
        let bytes = build_tar_gz(&[
            ("model.safetensors", b"safetensors-bytes" as &[u8]),
            ("tokenizer.json", b"{\"tokenizer\": true}"),
            ("config.json", b"{\"hidden_dim\": 256}"),
            ("modules.json", b"[]"),
        ]);
        let dir = tempfile::tempdir().expect("tmpdir");
        extract_tarball_to(&bytes, dir.path()).expect("extract succeeds");

        for filename in [
            "model.safetensors",
            "tokenizer.json",
            "config.json",
            "modules.json",
        ] {
            let p = dir.path().join(filename);
            assert!(p.exists(), "missing expected file: {}", p.display());
        }
    }

    /// Symmetric: the v0.1.0 base tarballs (potion-base-8M /
    /// potion-base-32M) nest files under a top-level directory. The
    /// extractor must strip that directory so files land at the cache
    /// root for model2vec to find them. This test guards against the
    /// regression-of-the-regression — fixing files-at-root must not
    /// break files-under-prefix.
    #[test]
    fn extract_tarball_files_under_prefix_layout() {
        let bytes = build_tar_gz(&[
            (
                "potion-base-8M/model.safetensors",
                b"safetensors-bytes" as &[u8],
            ),
            ("potion-base-8M/tokenizer.json", b"{\"tokenizer\": true}"),
            ("potion-base-8M/config.json", b"{\"hidden_dim\": 256}"),
            ("potion-base-8M/modules.json", b"[]"),
        ]);
        let dir = tempfile::tempdir().expect("tmpdir");
        extract_tarball_to(&bytes, dir.path()).expect("extract succeeds");

        // After stripping the prefix, files are at the dir root.
        for filename in [
            "model.safetensors",
            "tokenizer.json",
            "config.json",
            "modules.json",
        ] {
            let p = dir.path().join(filename);
            assert!(p.exists(), "missing expected file: {}", p.display());
        }
        // The prefix directory should NOT exist as a literal child of
        // dest_dir — the strip should have eliminated it.
        assert!(
            !dir.path().join("potion-base-8M").exists(),
            "prefix directory should have been stripped"
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
        let e = DownloadedEmbedder::fetch("potion-base-8M").expect("fetch potion-base-8M live");
        assert_eq!(e.dim(), 256);
        let v = e.embed("Alice met Acme yesterday").unwrap();
        assert_eq!(v.len(), 256);
        // L2-normalized output (model2vec config has normalize=true).
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "expected unit norm; got {norm}");
    }
}
