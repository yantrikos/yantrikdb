//! Python surface for mountable knowledge packs.
//!
//! `seal_pack` produces a file; `mount_pack` / `unmount_pack` attach and
//! detach it. See `docs/PACKS.md` for why this is a mount rather than an
//! import — briefly, because unmounting has to leave the host untouched,
//! and a tombstone-based detach cannot.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
type PyObject = pyo3::Py<pyo3::PyAny>;

use pyo3::exceptions::PyValueError;
use yantrikdb_core::{MountOptions, PackEmbedder, PackManifest, PackRecallOptions};

use crate::py_types::recall_result_to_dict;

use super::{map_err, PyYantrikDB};

#[pymethods]
impl PyYantrikDB {
    /// Write a sealed, mountable pack from this database.
    ///
    /// `namespace` scopes the export and is almost always what you want;
    /// `None` exports every namespace, private ones included.
    ///
    /// `embedder_digest` must identify the embedder whose vectors the
    /// pack carries. It defaults to this database's own recorded
    /// identity, which is correct whenever you are sealing a pack you
    /// built here. A pack with no digest can only be mounted with
    /// `allow_unverified_embedder=True`.
    ///
    /// Returns the sealed manifest, including the content digest and row
    /// count computed at seal time.
    #[pyo3(signature = (
        dest_path, name, version, origin, namespace=None, description=None,
        embedder_name=None, embedder_digest=None, embedder_dim=None,
        constitution=None, coverage=None,
        recommended_top_k=None, recommended_min_similarity=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn seal_pack(
        &self,
        py: Python<'_>,
        dest_path: &str,
        name: &str,
        version: &str,
        origin: &str,
        namespace: Option<&str>,
        description: Option<&str>,
        embedder_name: Option<&str>,
        embedder_digest: Option<&str>,
        embedder_dim: Option<usize>,
        constitution: Option<Vec<String>>,
        coverage: Option<Vec<String>>,
        recommended_top_k: Option<u32>,
        recommended_min_similarity: Option<f64>,
    ) -> PyResult<PyObject> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;

        // Default the declared embedder to what this database actually
        // recorded. Sealing with a digest the vectors do not have would
        // produce a pack that mounts cleanly and returns nonsense —
        // exactly the failure the mount check exists to stop — so the
        // safe value is the one on disk, not one the caller guessed.
        let recorded = db.embedder_identity().map_err(map_err)?;
        let embedder = PackEmbedder {
            name: embedder_name
                .map(|s| s.to_string())
                .or_else(|| recorded.as_ref().and_then(|(n, _, _)| n.clone())),
            digest: embedder_digest
                .map(|s| s.to_string())
                .or_else(|| recorded.as_ref().map(|(_, d, _)| d.clone())),
            dim: embedder_dim
                .or_else(|| recorded.as_ref().map(|(_, _, dim)| *dim))
                .unwrap_or_else(|| db.embedding_dim()),
        };

        let manifest = PackManifest {
            name: name.to_string(),
            version: version.to_string(),
            origin: origin.to_string(),
            description: description.map(|s| s.to_string()),
            embedder,
            content_digest: None,
            corpus_rows: 0,
            namespace: None,
            publisher_pubkey: None,
            signature: None,
            reembedded_from: None,
            constitution: constitution.unwrap_or_default(),
            coverage: coverage.unwrap_or_default(),
            // Passed through rather than defaulted: an absent value must
            // stay absent so the host knows the author did not measure
            // one, instead of inheriting a number invented here.
            recommended_top_k,
            recommended_min_similarity,
        };

        let sealed = db
            .seal_pack(dest_path, &manifest, namespace)
            .map_err(map_err)?;
        manifest_to_dict(py, &sealed, dest_path)
    }

    /// Mount a sealed pack read-only. Returns its pack id
    /// (`origin@version`).
    ///
    /// Raises `PackEmbedderMismatch` when the pack's vectors are not
    /// provably in this database's embedding space. That is a refusal to
    /// produce silently wrong results, not a bug — see the exception's
    /// docstring before reaching for `allow_unverified_embedder`, which
    /// covers *unproven* compatibility and never a *proven* mismatch.
    #[pyo3(signature = (path, allow_unverified_embedder=false, skip_content_digest=false))]
    fn mount_pack(
        &self,
        path: &str,
        allow_unverified_embedder: bool,
        skip_content_digest: bool,
    ) -> PyResult<String> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
        let opts = MountOptions {
            allow_unverified_embedder,
            skip_content_digest,
        };
        db.mount_pack_opts(path, &opts).map_err(map_err)
    }

    /// Unmount a pack. Returns `False` if no pack with that id was
    /// mounted. The host database is not modified — not one row, not one
    /// calibration counter.
    fn unmount_pack(&self, pack_id: &str) -> PyResult<bool> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
        db.unmount_pack(pack_id).map_err(map_err)
    }

    /// Unmount every pack. Returns how many were unmounted.
    fn unmount_all_packs(&self) -> PyResult<usize> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
        Ok(db.unmount_all_packs())
    }

    /// Currently mounted packs, in mount order.
    fn mounted_packs(&self, py: Python<'_>) -> PyResult<PyObject> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
        let out = PyList::empty(py);
        for info in db.mounted_packs() {
            let d = PyDict::new(py);
            d.set_item("pack_id", info.pack_id)?;
            d.set_item("name", info.name)?;
            d.set_item("version", info.version)?;
            d.set_item("origin", info.origin)?;
            d.set_item("description", info.description)?;
            d.set_item("path", info.path)?;
            d.set_item("trust", format!("{:?}", info.trust).to_lowercase())?;
            d.set_item("rows", info.rows)?;
            d.set_item("tier_multiplier", info.tier_multiplier)?;
            // 0.18: the namespace (present on the Rust PackInfo since its
            // doc comment described exactly this silent failure, and
            // dropped at this boundary until now) plus the sealed, SIGNED
            // retrieval/coverage facts — so a consumer never reads the
            // unsigned pack.toml off disk to learn a floor the manifest
            // already carries under its signature.
            d.set_item("namespace", info.namespace)?;
            d.set_item("content_digest", info.content_digest)?;
            d.set_item("coverage", info.coverage)?;
            d.set_item("recommended_top_k", info.recommended_top_k)?;
            d.set_item(
                "recommended_min_similarity",
                info.recommended_min_similarity,
            )?;
            d.set_item("publisher_pubkey", info.publisher_pubkey)?;
            d.set_item("signed", info.signed)?;
            out.append(d)?;
        }
        Ok(out.into())
    }

    /// Install a pack: copy it beside the database, mount it, and record
    /// it so it re-mounts automatically on every future open.
    ///
    /// This is the durable counterpart to `mount_pack`, which stays
    /// transient so that merely mounting never writes to your database.
    /// Returns the pack id.
    fn install_pack(&self, path: &str) -> PyResult<String> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
        db.install_pack(path).map_err(map_err)
    }

    /// Uninstall a pack: unmount it, forget it, and delete the copy in
    /// the pack directory. Returns `False` if it was not installed.
    fn uninstall_pack(&self, pack_id: &str) -> PyResult<bool> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
        db.uninstall_pack(pack_id).map_err(map_err)
    }

    /// Packs recorded as installed, whether or not they mounted this
    /// session. Compare with `mounted_packs()` to spot one that failed
    /// to re-mount.
    fn installed_packs(&self, py: Python<'_>) -> PyResult<PyObject> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
        let out = PyList::empty(py);
        for p in db.installed_packs().map_err(map_err)? {
            let d = PyDict::new(py);
            d.set_item("pack_id", p.pack_id)?;
            d.set_item("file_name", p.file_name)?;
            d.set_item("name", p.name)?;
            d.set_item("version", p.version)?;
            d.set_item("content_digest", p.content_digest)?;
            d.set_item("installed_at", p.installed_at)?;
            out.append(d)?;
        }
        Ok(out.into())
    }

    /// Directory where installed packs live, or `None` for an in-memory
    /// database.
    fn pack_dir(&self) -> PyResult<Option<String>> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
        Ok(db.pack_dir().map(|p| p.to_string_lossy().to_string()))
    }

    /// Re-mount every installed pack, returning what happened to each.
    /// Runs automatically at open; exposed for diagnosing a pack that
    /// did not come back.
    fn remount_installed(&self, py: Python<'_>) -> PyResult<PyObject> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
        let out = PyList::empty(py);
        for o in db.remount_installed() {
            let d = PyDict::new(py);
            d.set_item("pack_id", o.pack_id)?;
            d.set_item("mounted", o.mounted)?;
            d.set_item("reason", o.reason)?;
            out.append(d)?;
        }
        Ok(out.into())
    }

    /// Read a pack file's manifest without mounting it.
    #[staticmethod]
    fn read_pack_manifest(py: Python<'_>, path: &str) -> PyResult<PyObject> {
        let m = yantrikdb_core::YantrikDB::read_manifest(path).map_err(map_err)?;
        manifest_to_dict(py, &m, path)
    }

    /// Rewrite a pack's vectors into a different embedding space.
    ///
    /// A pack's vectors only work in the space they were built in, and
    /// mount treats a dimension mismatch as fatal — 64-dim vectors
    /// physically cannot be searched by a 256-dim index. This produces a
    /// converted copy at `dest` so one published artifact can serve hosts
    /// in any space.
    ///
    ///     YantrikDB.convert_pack("mypack.ydbpack",
    ///                            "mypack-256.ydbpack",
    ///                            "potion-base-8M")
    ///
    /// **You usually do not need to call this.** `install_pack()` converts
    /// automatically when the pack's space differs from the database's.
    /// Reach for it when producing artifacts to publish.
    ///
    /// The publisher's content digest is over `(rid, text)`, so it still
    /// verifies afterwards — rows keep their original ids and text, and
    /// only the vectors are regenerated. Any publisher signature IS
    /// dropped, because it covers the embedder identity and would no
    /// longer verify; the original embedder digest is recorded in the new
    /// manifest's `reembedded_from` so the conversion is visible.
    ///
    /// Raises `RuntimeError` if `dest` exists, if the embedder name is
    /// unknown, or if the pack is already in that space.
    #[staticmethod]
    fn convert_pack(
        py: Python<'_>,
        src: &str,
        dest: &str,
        embedder_name: &str,
    ) -> PyResult<PyObject> {
        let m =
            yantrikdb_core::YantrikDB::convert_pack(src, dest, embedder_name).map_err(map_err)?;
        manifest_to_dict(py, &m, dest)
    }

    /// Generate a publisher keypair as `(secret_hex, public_hex)`.
    ///
    /// The secret key IS the publisher identity — whoever holds it can
    /// publish packs as you. The engine never stores it.
    #[staticmethod]
    fn generate_pack_keypair() -> (String, String) {
        yantrikdb_core::engine::pack::generate_pack_keypair()
    }

    /// Sign a sealed pack with a publisher secret key. Returns the
    /// public key hex. The signature covers identity, content digest,
    /// embedder identity, constitution and coverage — everything except
    /// the cosmetic description.
    #[staticmethod]
    fn sign_pack(path: &str, secret_key_hex: &str) -> PyResult<String> {
        yantrikdb_core::YantrikDB::sign_pack(path, secret_key_hex).map_err(map_err)
    }

    /// Sign arbitrary bytes with a secret key (evaluation certificates
    /// etc). Returns the signature hex.
    #[staticmethod]
    fn sign_bytes(secret_key_hex: &str, data: &[u8]) -> PyResult<String> {
        yantrikdb_core::engine::pack::sign_bytes(secret_key_hex, data).map_err(map_err)
    }

    /// Verify a signature over arbitrary bytes. Returns False for a
    /// wrong signature; raises only on malformed key/signature hex.
    #[staticmethod]
    fn verify_bytes(pubkey_hex: &str, data: &[u8], signature_hex: &str) -> PyResult<bool> {
        yantrikdb_core::engine::pack::verify_bytes(pubkey_hex, data, signature_hex).map_err(map_err)
    }

    /// The public key corresponding to a secret key.
    #[staticmethod]
    fn pubkey_of(secret_key_hex: &str) -> PyResult<String> {
        yantrikdb_core::engine::pack::pubkey_of(secret_key_hex).map_err(map_err)
    }

    /// Trust a publisher key: packs validly signed by it mount at the
    /// `signed` tier (higher recall-ranking multiplier) from now on.
    #[pyo3(signature = (pubkey_hex, label=None))]
    fn trust_publisher(&self, pubkey_hex: &str, label: Option<&str>) -> PyResult<()> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
        db.trust_publisher(pubkey_hex, label).map_err(map_err)
    }

    /// Stop trusting a publisher key. Mounted packs keep their tier
    /// until remount. Returns `False` if the key was not trusted.
    fn untrust_publisher(&self, pubkey_hex: &str) -> PyResult<bool> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
        db.untrust_publisher(pubkey_hex).map_err(map_err)
    }

    /// Publisher keys this host trusts, as `[{"pubkey", "label"}]`.
    fn trusted_publishers(&self, py: Python<'_>) -> PyResult<PyObject> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
        let out = PyList::empty(py);
        for (pubkey, label) in db.trusted_publishers().map_err(map_err)? {
            let d = PyDict::new(py);
            d.set_item("pubkey", pubkey)?;
            d.set_item("label", label)?;
            out.append(d)?;
        }
        Ok(out.into())
    }

    /// The unconditional context block for all mounted packs — each
    /// pack's coverage index and constitution rules, assembled by the
    /// engine so every consumer injects the same block. Put it in your
    /// system prompt while packs are mounted. `None` when no mounted
    /// pack declares either tier.
    fn pack_context(&self) -> PyResult<Option<String>> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
        Ok(db.pack_context())
    }

    /// The context block for ONLY the packs in `pack_ids` (0.18), in
    /// MOUNT order regardless of the order given, duplicates collapsed.
    /// Every id must be mounted — an unknown id raises `PackNotMounted`
    /// rather than being skipped. `[]` returns `None`.
    fn pack_context_for(&self, pack_ids: Vec<String>) -> PyResult<Option<String>> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
        let ids: Vec<&str> = pack_ids.iter().map(|s| s.as_str()).collect();
        db.pack_context_for(&ids).map_err(map_err)
    }

    /// Pack-only recall over an explicit allowlist (0.18).
    ///
    /// Searches ONLY the packs in `pack_ids` — host rows are never
    /// candidates and a pack outside the list cannot crowd one inside
    /// it. The allowlist is validated first: an unknown id raises
    /// `PackNotMounted` and nothing is searched.
    ///
    /// Each pack's signed `recommended_min_similarity` gates raw
    /// similarity; `min_similarity` lets the host RAISE every pack's
    /// floor and can never lower a pack's own. A host record that
    /// supersedes a pack rid removes it, as in `recall`. Ties on score
    /// break by mount order then rid, so the result is deterministic.
    /// No MMR, graph expansion or reinforcement. Every hit carries
    /// `hit["pack"] = {pack_id, name, version, trust, content_digest}`.
    #[pyo3(signature = (pack_ids, query=None, query_embedding=None, top_k=10, *, min_similarity=None, namespace=None, memory_type=None, domain=None, source=None, certainty_min=None, include_consolidated=false, time_window=None))]
    #[allow(clippy::too_many_arguments)]
    fn recall_from_packs_for(
        &self,
        py: Python<'_>,
        pack_ids: Vec<String>,
        query: Option<&str>,
        query_embedding: Option<Vec<f32>>,
        top_k: usize,
        min_similarity: Option<f64>,
        namespace: Option<&str>,
        memory_type: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
        certainty_min: Option<f64>,
        include_consolidated: bool,
        time_window: Option<(f64, f64)>,
    ) -> PyResult<Vec<PyObject>> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
        let ids: Vec<&str> = pack_ids.iter().map(|s| s.as_str()).collect();
        // Validated here, as ValueError, rather than by remapping the
        // engine's InvalidInput globally: that variant already crosses
        // this boundary as RuntimeError for older APIs, and changing it
        // would change their exception type under existing handlers.
        if let Some(m) = min_similarity {
            if !m.is_finite() || !(0.0..=1.0).contains(&m) {
                return Err(PyValueError::new_err(format!(
                    "min_similarity must be within [0, 1], got {m}"
                )));
            }
        }
        // Allowlist BEFORE the embedder: an unknown id must fail before a
        // query string is encoded (review of #203).
        db.validate_pack_allowlist(&ids).map_err(map_err)?;
        let emb = match query_embedding {
            Some(e) => e,
            None => match query {
                Some(q) => self.embed_text(py, q)?,
                None => {
                    return Err(PyValueError::new_err(
                        "Must provide either query or query_embedding",
                    ))
                }
            },
        };
        let opts = PackRecallOptions {
            include_consolidated,
            memory_type,
            time_window,
            namespace,
            domain,
            source,
            certainty_min,
            min_similarity,
        };
        let results = db
            .recall_from_packs_for(&ids, &emb, top_k, query, &opts)
            .map_err(map_err)?;
        results
            .iter()
            .map(|r| recall_result_to_dict(py, r))
            .collect()
    }

    /// Assert that this database's existing vectors were built by the
    /// currently-attached embedder, and record that as its identity.
    ///
    /// Identity is normally stamped automatically the first time
    /// `record_text` stores a vector the engine itself produced. Call
    /// this when you populate the database with vectors computed
    /// elsewhere, or to bring a database written before this feature
    /// existed up to date — otherwise it can never mount a pack without
    /// `allow_unverified_embedder=True`.
    ///
    /// It is an assertion, not a measurement: nothing checks the claim,
    /// because nothing can. Returns the adopted digest.
    fn adopt_embedder_identity(&self) -> PyResult<String> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
        db.adopt_embedder_identity().map_err(map_err)
    }

    /// The embedder identity recorded on disk for this database, as
    /// `{"name", "digest", "dim"}`, or `None` if none was ever recorded.
    ///
    /// A database with no recorded identity cannot prove pack
    /// compatibility, which is why `mount_pack` refuses it by default.
    fn embedder_identity(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
        let Some((name, digest, dim)) = db.embedder_identity().map_err(map_err)? else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("name", name)?;
        d.set_item("digest", digest)?;
        d.set_item("dim", dim)?;
        Ok(Some(d.into()))
    }
}

fn manifest_to_dict(py: Python<'_>, m: &PackManifest, path: &str) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("path", path)?;
    d.set_item("pack_id", m.pack_id())?;
    d.set_item("name", &m.name)?;
    d.set_item("version", &m.version)?;
    d.set_item("origin", &m.origin)?;
    d.set_item("description", m.description.clone())?;
    d.set_item("namespace", m.namespace.clone())?;
    d.set_item("content_digest", m.content_digest.clone())?;
    d.set_item("corpus_rows", m.corpus_rows)?;
    d.set_item("constitution", m.constitution.clone())?;
    d.set_item("coverage", m.coverage.clone())?;
    d.set_item("recommended_top_k", m.recommended_top_k)?;
    d.set_item("recommended_min_similarity", m.recommended_min_similarity)?;
    d.set_item("publisher_pubkey", m.publisher_pubkey.clone())?;
    d.set_item("signed", m.signature.is_some())?;
    // Present iff this pack's vectors were regenerated locally: the rows
    // are still the publisher's (the content digest verifies) but the
    // vectors are this host's, and any publisher signature was dropped.
    d.set_item("reembedded_from", m.reembedded_from.clone())?;
    let e = PyDict::new(py);
    e.set_item("name", m.embedder.name.clone())?;
    e.set_item("digest", m.embedder.digest.clone())?;
    e.set_item("dim", m.embedder.dim)?;
    d.set_item("embedder", e)?;
    Ok(d.into())
}
