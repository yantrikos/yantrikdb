mod cognition;
mod graph;
mod memory;
mod pack;
mod session_temporal;
mod sync;

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
type PyObject = pyo3::Py<pyo3::PyAny>;

use yantrikdb_core::engine::materializer::{
    recommended_worker_count, spawn_all_workers, AllWorkerGuards,
};
use yantrikdb_core::YantrikDB;

use crate::py_types::*;

/// Python wrapper for the YantrikDB engine.
#[pyclass(name = "YantrikDB")]
pub struct PyYantrikDB {
    pub(crate) inner: Option<Arc<YantrikDB>>,
    pub(crate) embedder: Option<PyObject>,
    /// Background worker pool (materializers + compactor). MUST be held for the
    /// engine's lifetime: without the compactor the in-memory delta tier fills
    /// to `delta_max` (256) and every subsequent write returns Backpressure
    /// ("ingest queue full"). The pyo3 constructors previously forgot to spawn
    /// these, wedging any long write session (e.g. writing a full novel) after
    /// ~256 records. Dropped with the struct → Weak<YantrikDB> upgrade fails →
    /// workers shut down cleanly.
    pub(crate) _workers: Option<AllWorkerGuards>,
}

impl PyYantrikDB {
    /// Wrap a freshly-constructed engine AND spawn its background workers
    /// (materializer pool + compactor). This is the single place that owns the
    /// "don't forget the compactor" rule for the Python binding.
    pub(crate) fn from_engine(inner: YantrikDB, embedder: Option<PyObject>) -> Self {
        let inner = Arc::new(inner);
        let workers = spawn_all_workers(&inner, recommended_worker_count());
        Self {
            inner: Some(inner),
            embedder,
            _workers: Some(workers),
        }
    }

    /// Compare a Python embedder against the identity the store recorded for
    /// the vectors already on disk (#117).
    ///
    /// The engine refuses a same-dim-different-digest swap in its own
    /// `set_embedder`, but a Python embedder never reaches that path: it is
    /// held on this wrapper, encoding queries and writes for an engine whose
    /// `SearchState` has never seen it. So the comparison has to happen here.
    ///
    /// A Python object has no digest the engine can derive, so it declares
    /// one — `fingerprint` or `digest`, a plain string attribute. Rules:
    ///
    /// * store has no recorded identity -> nothing to contradict, attach.
    /// * recorded dim contradicts the width of the vectors actually stored
    ///   -> the row is provably false (that model does not emit that width),
    ///   so it is evidence of nothing. Attach, warn loudly. This is "cannot
    ///   verify", NOT "verified mismatch" — the two must not collapse, or
    ///   every database written before identities were recorded is bricked.
    /// * declared fingerprint equals the recorded digest -> attach.
    /// * declared fingerprint differs -> refuse. This is the provable
    ///   mismatch, and it is silent corruption if allowed: cosine distance
    ///   between two unrelated spaces still returns a plausible number
    ///   (measured: 0.595 against vectors built by a different model).
    /// * nothing declared -> refuse. "I cannot prove these vectors are mine"
    ///   is the case that produced the 0.595, not a safe default.
    ///
    /// `allow_unverified_embedder` overrides the last two, matching the
    /// convention `pack.rs` already uses for mounting.
    fn check_embedder_identity(
        &self,
        py: Python<'_>,
        embedder: &PyObject,
        allow_unverified: bool,
    ) -> PyResult<()> {
        if allow_unverified {
            return Ok(());
        }
        let Some(inner) = self.inner.as_ref() else {
            return Ok(());
        };
        // (name, digest, dim) of whatever built the vectors already stored.
        let recorded = inner.embedder_identity().map_err(map_err)?;
        let Some((recorded_name, recorded_digest, recorded_dim)) = recorded else {
            return Ok(()); // nothing recorded: nothing to contradict
        };

        // **A stored vector is the authority, not the recorded identity.**
        // The same ordering `detect_existing_dim` applies at open time, for
        // the same reason: the identity row records what was ATTACHED when
        // the engine first embedded, which is not necessarily what produced
        // the vectors in the file. A caller that embeds externally and passes
        // vectors to `record()` never stamps an identity, but any incidental
        // `embed()` call stamps the attached model anyway.
        //
        // `stored_vector_dim()` decodes a durable hot vector and measures its
        // actual width. Do not substitute `embedding_dim()` here: the Python
        // constructor accepts that value from its caller, so treating it as a
        // measurement can turn a wrong open configuration into permission to
        // attach an unrelated embedder.
        //
        // Refusing on it would be worse than useless: it bricks the database
        // (every call fails, lazily, after the service starts clean) on the
        // strength of a value the engine elsewhere already refuses to trust,
        // and the error names a model we can prove did not build the vectors.
        // A gate that misreports provenance is worse than no gate.
        //
        // Found on a real 5,699-record production store recording
        // `potion-base-2M / dim 64` while holding 1536-byte (384-dim) MiniLM
        // vectors — the same store `detect_existing_dim`'s comment cites.
        //
        // This is "cannot verify", which is a DIFFERENT state from "verified
        // mismatch" — collapsing the two is what bricks legacy databases.
        // Unverified is where every pre-gate database already lived, so this
        // is no weaker than the release before it; it simply declines to
        // manufacture a verdict from a row known to be wrong. Loud, because
        // the identity still needs operator verification and repair.
        let measured_dim = inner.stored_vector_dim().map_err(map_err)?;
        if measured_dim.is_some_and(|dim| recorded_dim != dim) {
            let measured_dim = measured_dim.expect("checked as Some above");
            let named = recorded_name
                .clone()
                .unwrap_or_else(|| "<unnamed>".to_string());
            PyErr::warn(
                py,
                &py.get_type::<pyo3::exceptions::PyUserWarning>(),
                &std::ffi::CString::new(format!(
                    "this database records its vectors as built by {named} (dim \
                     {recorded_dim}), but the vectors it holds are {measured_dim}-dim, so \
                     the recorded identity is wrong and does not describe them. Attaching \
                     without an identity check: provenance is UNVERIFIED, which is what it \
                     was before this \
                     check existed. Before writing, independently verify that the attached \
                     embedder reproduces the stored vectors from their source text; this \
                     release does not provide an automated identity repair."
                ))
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
                0,
            )?;
            return Ok(());
        }

        let declared = ["fingerprint", "digest"].iter().find_map(|attr| {
            embedder
                .getattr(py, *attr)
                .ok()
                .and_then(|v| v.extract::<String>(py).ok())
        });

        match declared {
            Some(fp) if fp == recorded_digest => Ok(()),
            other => {
                let named = recorded_name.unwrap_or_else(|| "<unnamed>".to_string());
                let got = match other {
                    Some(fp) => format!("declares fingerprint {fp:?}"),
                    None => "declares no `fingerprint` or `digest`".to_string(),
                };
                Err(PyRuntimeError::new_err(format!(
                    "this database's vectors were built by {named} (digest {recorded_digest}, \
                     dim {recorded_dim}), and the embedder being attached {got}. Queries would \
                     be encoded in a different space than the vectors they are compared against, \
                     and cosine distance still returns a plausible number for unrelated spaces — \
                     so the results would look fine and be wrong. Options: attach the embedder \
                     that built them; set `.fingerprint = \"{recorded_digest}\"` on your \
                     embedder if it IS that model; call reembed() to rebuild the vectors in the \
                     new space; or pass allow_unverified_embedder=True if you accept the risk."
                )))
            }
        }
    }

    /// (Re)spawn the background worker pool against the live engine `Arc`.
    /// Paired with `self._workers = None` (which drops the guards and JOINs
    /// the worker threads) around any operation that needs exclusive
    /// `Arc::get_mut` access — the workers hold `Weak<YantrikDB>` refs, and
    /// `get_mut` requires weak count 0. No-op if the engine is closed.
    #[cfg(feature = "embedder-download")]
    fn respawn_workers(&mut self) {
        if let Some(arc) = self.inner.as_ref() {
            self._workers = Some(spawn_all_workers(arc, recommended_worker_count()));
        }
    }
}

/// A thin proxy exposing execute() and commit() on the underlying connection.
/// This is needed because Python tests access db._conn.execute(...) directly.
/// Stores an Arc<YantrikDB> and acquires the connection lock on each call.
#[pyclass]
pub struct ConnectionProxy {
    db: Arc<YantrikDB>,
}

/// A cursor-like object returned by ConnectionProxy.execute().
/// Holds the result rows so fetchall()/fetchone() can return them.
#[pyclass]
pub struct CursorProxy {
    rows: Vec<PyObject>,
    rowcount: usize,
}

#[pymethods]
impl CursorProxy {
    fn fetchall(&self, py: Python<'_>) -> Vec<PyObject> {
        self.rows.iter().map(|r| r.clone_ref(py)).collect()
    }

    fn fetchone(&self, py: Python<'_>) -> Option<PyObject> {
        self.rows.first().map(|r| r.clone_ref(py))
    }

    #[getter]
    fn rowcount(&self) -> usize {
        self.rowcount
    }
}

#[pymethods]
impl ConnectionProxy {
    #[pyo3(signature = (sql, params=None))]
    fn execute(
        &self,
        py: Python<'_>,
        sql: &str,
        params: Option<&Bound<'_, PyTuple>>,
    ) -> PyResult<CursorProxy> {
        let conn = self.db.conn();

        // Result-shape detection by the STATEMENT's column count, not a
        // string prefix: `WITH … SELECT`, `PRAGMA`, and `INSERT … RETURNING`
        // all produce rows but failed the old starts_with("SELECT") check —
        // their rows were silently discarded with rowcount 0 (2026-08-15
        // binding audit; the lenient shim inflates silently in tests).
        // Probe on the ALREADY-HELD guard — re-acquiring self.db.conn()
        // here while the outer `conn` is held is the same-thread re-lock
        // deadlock (#83). The probe statement drops at the match end.
        let is_select = match conn.prepare(sql) {
            Ok(stmt) => stmt.column_count() > 0,
            Err(_) => false, // real error surfaces from the execute below
        };

        let param_values: Vec<Box<dyn rusqlite::types::ToSql>> = if let Some(p) = params {
            p.iter()
                .map(|item| py_to_sql_value(&item))
                .collect::<PyResult<_>>()?
        } else {
            vec![]
        };
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        if is_select {
            let mut stmt = conn
                .prepare(sql)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let col_count = stmt.column_count();
            let col_names: Vec<String> = (0..col_count)
                .map(|i| stmt.column_name(i).unwrap_or("").to_string())
                .collect();

            let rows_result = stmt
                .query_map(params_ref.as_slice(), |row| {
                    let mut values: Vec<rusqlite::types::Value> = Vec::new();
                    for i in 0..col_count {
                        values.push(row.get::<_, rusqlite::types::Value>(i)?);
                    }
                    Ok(values)
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let mut py_rows: Vec<PyObject> = Vec::new();
            for row_result in rows_result {
                let values = row_result.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let dict = PyDict::new(py);
                for (i, val) in values.iter().enumerate() {
                    let py_val = sqlite_value_to_py(py, val)?;
                    dict.set_item(&col_names[i], py_val)?;
                }
                py_rows.push(dict.into());
            }

            Ok(CursorProxy {
                rows: py_rows,
                rowcount: 0,
            })
        } else {
            let changes = conn
                .execute(sql, params_ref.as_slice())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(CursorProxy {
                rows: vec![],
                rowcount: changes,
            })
        }
    }

    fn executescript(&self, _py: Python<'_>, sql: &str) -> PyResult<()> {
        self.db
            .conn()
            .execute_batch(sql)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    fn commit(&self, _py: Python<'_>) -> PyResult<()> {
        // In rusqlite auto-commit mode, commit is a no-op
        Ok(())
    }
}

fn sqlite_value_to_py(py: Python<'_>, val: &rusqlite::types::Value) -> PyResult<PyObject> {
    match val {
        rusqlite::types::Value::Null => Ok(py.None()),
        rusqlite::types::Value::Integer(i) => {
            Ok((*i).into_pyobject(py)?.to_owned().into_any().unbind())
        }
        rusqlite::types::Value::Real(f) => {
            Ok((*f).into_pyobject(py)?.to_owned().into_any().unbind())
        }
        rusqlite::types::Value::Text(s) => {
            Ok(s.as_str().into_pyobject(py)?.to_owned().into_any().unbind())
        }
        rusqlite::types::Value::Blob(b) => Ok(b
            .as_slice()
            .into_pyobject(py)?
            .to_owned()
            .into_any()
            .unbind()),
    }
}

pub(crate) fn py_to_sql_value(obj: &Bound<'_, PyAny>) -> PyResult<Box<dyn rusqlite::types::ToSql>> {
    if obj.is_none() {
        return Ok(Box::new(rusqlite::types::Null));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Box::new(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Box::new(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Box::new(s));
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Box::new(b));
    }
    Err(PyRuntimeError::new_err(format!(
        "Unsupported SQL parameter type: {}",
        obj.get_type().name()?
    )))
}

pub(crate) fn map_err(e: yantrikdb_core::YantrikDbError) -> PyErr {
    use crate::py_errors;
    use yantrikdb_core::YantrikDbError as E;
    // v0.10 (yantrikdb-mcp friction 2): actionable variants cross the pyo3
    // boundary as TYPED exception classes so hosts branch on type, never on
    // message text. Every class subclasses RuntimeError — pre-v0.10 handlers
    // keep working. The message still carries the details (e.g. the existing
    // rid on a conflict).
    match e {
        E::Backpressure { .. } => py_errors::Backpressure::new_err(e.to_string()),
        E::CorrectionDeferredDuringReembed { .. } => {
            py_errors::CorrectionDeferredDuringReembed::new_err(e.to_string())
        }
        E::BatchDeferredDuringReembed { .. } => {
            py_errors::BatchDeferredDuringReembed::new_err(e.to_string())
        }
        E::IdempotencyConflict { .. } => py_errors::IdempotencyConflict::new_err(e.to_string()),
        E::InvalidIdempotencyKey { .. } => py_errors::InvalidIdempotencyKey::new_err(e.to_string()),
        E::ProvenanceInconsistent { .. } => {
            py_errors::ProvenanceInconsistent::new_err(e.to_string())
        }
        E::RecallContended { .. } => py_errors::RecallContended::new_err(e.to_string()),
        E::PackEmbedderMismatch { .. } => py_errors::PackEmbedderMismatch::new_err(e.to_string()),
        E::PackAlreadyMounted { .. } => py_errors::PackAlreadyMounted::new_err(e.to_string()),
        E::PackSignatureInvalid { .. } => py_errors::PackSignatureInvalid::new_err(e.to_string()),
        E::NoEmbedder => PyRuntimeError::new_err(e.to_string()),
        E::NoQuery => PyValueError::new_err(e.to_string()),
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

#[pymethods]
impl PyYantrikDB {
    #[new]
    #[pyo3(signature = (db_path=":memory:", embedding_dim=384, embedder=None, encryption_key=None, model_dir=None, allow_unverified_embedder=false))]
    fn new(
        py: Python<'_>,
        db_path: &str,
        embedding_dim: usize,
        embedder: Option<PyObject>,
        encryption_key: Option<Vec<u8>>,
        model_dir: Option<&str>,
        allow_unverified_embedder: bool,
    ) -> PyResult<Self> {
        #[allow(unused_mut)]
        let mut inner = if let Some(key_bytes) = encryption_key {
            if key_bytes.len() != 32 {
                return Err(PyValueError::new_err(
                    "encryption_key must be exactly 32 bytes",
                ));
            }
            let mut key = [0u8; 32];
            key.copy_from_slice(&key_bytes);
            YantrikDB::new_encrypted(db_path, embedding_dim, &key).map_err(map_err)?
        } else {
            YantrikDB::new(db_path, embedding_dim).map_err(map_err)?
        };

        // If model_dir provided and candle feature enabled, use CandleEmbedder
        #[cfg(feature = "candle")]
        if let Some(dir) = model_dir {
            let candle_embedder = yantrik_ml::CandleEmbedder::from_dir(std::path::Path::new(dir))
                .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to load candle embedder: {e}"))
            })?;
            // set_embedder returns Result post-#41 (mode-aware refactor).
            // On a freshly-constructed engine (no memories yet) this can
            // only fail on dim mismatch — surface that as a Python error.
            inner
                .set_embedder(Box::new(candle_embedder))
                .map_err(map_err)?;
        }

        #[cfg(not(feature = "candle"))]
        if model_dir.is_some() {
            return Err(PyRuntimeError::new_err(
                "model_dir requires the 'candle' feature. Build with: maturin develop --features candle",
            ));
        }

        // #117: the constructor is the path the original report used. Build
        // the wrapper first so the engine is queryable, then prove the
        // embedder belongs to these vectors BEFORE it is allowed to encode
        // anything. Gating only `set_embedder` left this open.
        let mut this = Self::from_engine(inner, None);
        if let Some(emb) = embedder {
            this.check_embedder_identity(py, &emb, allow_unverified_embedder)?;
            this.embedder = Some(emb);
        }
        Ok(this)
    }

    /// **v0.7.4** — open with a default embedder pre-attached.
    ///
    /// Mirrors `YantrikDB::with_default` on the Rust side, so `record(
    /// text=...)`, `recall(query=...)`, `record_text()` and `recall_text()`
    /// work with no external sentence-transformers install. This is the
    /// "Just Works" entry point for Python.
    ///
    /// **Which model you get (changed 2026-08-13).** A NEW file-backed
    /// store opens at 256 dims on `potion-base-8M`, downloaded once
    /// (~28 MB, SHA-256 pinned, cached under your cache dir). Measured on
    /// 5,035 real agent memories through the engine's own recall, it
    /// leaves the correct record outside the top 100 on 1 question in 12
    /// where the bundled 64-dim `potion-base-2M` misses 4 in 12 — the
    /// difference between a memory that answers and one that appears
    /// empty.
    ///
    /// If the model cannot be fetched (offline), the store is created on
    /// the bundled `potion-base-2M` instead and a warning is logged. That
    /// choice is permanent for that store: a database's dimension is fixed
    /// at creation, so switching later means re-embedding.
    ///
    /// An EXISTING store always reopens at the dimension it already holds,
    /// so upgrading the library never strands a database. In-memory
    /// (`":memory:"`) stores stay on the bundled model deliberately — they
    /// are ephemeral, and defaulting them to a download would make every
    /// test suite that uses one require the network.
    ///
    /// Slim wheels built with `--no-default-features` will raise at the
    /// engine level (the bundled embedder is feature-gated). Default PyPI
    /// wheels include it.
    #[staticmethod]
    fn with_default(db_path: &str) -> PyResult<Self> {
        let inner = YantrikDB::with_default(db_path).map_err(map_err)?;
        Ok(Self::from_engine(inner, None))
    }

    /// Whether this instance has encryption enabled.
    ///
    /// **Not a sufficient check on versions < 0.13.2** (GHSA-84vx-5fgq-5p59):
    /// this returned `True` while record payloads sat in plaintext in
    /// `oplog.payload`. Pair it with `oplog_plaintext_rows()`.
    #[getter]
    fn is_encrypted(&self) -> PyResult<bool> {
        let db = self.get_inner()?;
        Ok(db.is_encrypted())
    }

    /// Unsealed oplog rows — the "am I affected?" number from
    /// GHSA-84vx-5fgq-5p59, exposed so the answer is a count rather
    /// than a judgement.
    ///
    /// On an encrypted 0.13.2+ database this must be `0` after open
    /// (the migration seals pre-0.13.2 rows). On an UNencrypted
    /// database it counts every row, because nothing is sealed there
    /// and nothing should be — read it together with `is_encrypted`.
    fn oplog_plaintext_rows(&self) -> PyResult<usize> {
        let db = self.get_inner()?;
        db.oplog_plaintext_rows().map_err(map_err)
    }

    /// Whether ANY embedder is configured (Rust-native or Python-side).
    ///
    /// **v0.7.10 fix for issue yantrikos/yantrikdb-hermes-plugin#4
    /// (alienos 2026-05-13).** The pyo3 wrapper stores embedders in TWO
    /// places: `self.inner.embedder` (Rust-native, set via the engine's
    /// own `set_embedder()` from `with_default()`, `auto_attach_bundled
    /// _embedder()`, and `set_embedder_named()`) and `self.embedder`
    /// (Python `PyObject` set via the pyo3 `set_embedder(obj)` method
    /// when the caller passes their own Python embedder like a
    /// `SentenceTransformer` or `Model2VecEmbedder`). Both work in
    /// `embed_text()` — it tries the Rust path first, then falls back
    /// to the Python path.
    ///
    /// Pre-v0.7.10 `has_embedder()` only checked the Rust side, so a
    /// caller who attached a Python embedder via `set_embedder(...)`
    /// would see `has_embedder() == False` despite having a fully
    /// functional embedder — and `record_text()` / `recall_text()`
    /// would work correctly. The asymmetry blocked the Hermes plugin's
    /// startup precondition check (its embedded.py raised on
    /// `not has_embedder()`).
    ///
    /// True if EITHER side has an embedder. Functionally equivalent to
    /// "can `embed_text(...)` succeed?", which is the actual question
    /// any caller running this check is trying to answer.
    fn has_embedder(&self) -> PyResult<bool> {
        let db = self.get_inner()?;
        Ok(db.has_embedder() || self.embedder.is_some())
    }

    /// Attach a Python-callable embedder. Must implement
    /// `encode(text: str) -> list[float] | numpy.ndarray`.
    ///
    /// **v0.7.5 hostile-input guard.** Previously this method accepted any
    /// PyObject without validation, then silently failed at first
    /// `record_text()` / `recall_text()` call with a confusing downstream
    /// error. The classic trap (reported by yantrikdb-hermes-agent msg
    /// c8734310) was passing a string by mistake — Python strings have a
    /// `.encode(charset)` method that interprets the argument as a charset
    /// name, raising `LookupError: unknown encoding: <text>` from deep
    /// inside the embed path.
    ///
    /// Now we probe the embedder at set time with a sentinel string and
    /// reject anything that doesn't return a numeric vector. Costs one
    /// extra `encode()` call up front in exchange for a clear,
    /// localized error.
    #[pyo3(signature = (embedder, allow_unverified_embedder = false))]
    fn set_embedder(
        &mut self,
        py: Python<'_>,
        embedder: PyObject,
        allow_unverified_embedder: bool,
    ) -> PyResult<()> {
        // Probe with a sentinel — if encode() doesn't produce a numeric
        // vector, the embedder is bogus and we raise immediately.
        let probe = embedder
            .call_method1(py, "encode", ("__yantrikdb_probe__",))
            .map_err(|e| {
                PyTypeError::new_err(format!(
                    "embedder must implement encode(text: str) -> list[float] \
                 (calling .encode('__yantrikdb_probe__') raised: {e}). \
                 Hint: pass a sentence-transformers SentenceTransformer or \
                 any object with a compatible encode() method, OR use \
                 YantrikDB.with_default(path) for the bundled embedder."
                ))
            })?;

        // Numeric-vector check. Tolerates list[float] and numpy.ndarray
        // (via .tolist()) since both are common embedder output types.
        let numeric_ok = probe.extract::<Vec<f32>>(py).is_ok()
            || probe
                .call_method0(py, "tolist")
                .and_then(|l| l.extract::<Vec<f32>>(py))
                .is_ok();
        if !numeric_ok {
            return Err(PyTypeError::new_err(
                "embedder.encode(text) must return list[float] or numpy.ndarray; \
                 got non-numeric. Common cause: passing a str (str.encode is a \
                 charset codec, not an embedder).",
            ));
        }

        // #117: the probe above proves it is AN embedder. This proves it is
        // THE embedder — the one whose space the stored vectors live in.
        self.check_embedder_identity(py, &embedder, allow_unverified_embedder)?;

        self.embedder = Some(embedder);
        Ok(())
    }

    /// **v0.7.5 — Slice C exposed in Python.** Replace the engine's current
    /// embedder with one downloaded from
    /// [`yantrikos/yantrikdb-models`](https://github.com/yantrikos/yantrikdb-models).
    ///
    /// Known names (registry pinned per release for SHA-256 verification):
    /// - `"potion-base-8M"`  — 256-dim, ~92% MiniLM, ~28 MB tarball
    /// - `"potion-base-32M"` — 512-dim, ~95% MiniLM, ~121 MB tarball
    ///
    /// First call fetches + verifies SHA-256 + extracts to
    /// `dirs::cache_dir() / "yantrikdb" / "models" /`. Subsequent calls
    /// (this process or any other against the same cache dir) hit the
    /// cache and skip the network.
    ///
    /// **Dimension contract.** The engine's `embedding_dim` (set at
    /// construction) must match the named model's output dim, and a store's
    /// dimension is fixed when it is created. So `set_embedder_named` can
    /// only ever swap in a model of the dimension the store already has —
    /// to use a different one, construct at that dim:
    /// `YantrikDB(":memory:", embedding_dim=256)` then
    /// `db.set_embedder_named("potion-base-8M")`.
    ///
    /// Since 2026-08-13 `with_default(path)` already picks the right dim
    /// for you: a NEW file-backed store opens at 256 on `potion-base-8M`
    /// (downloaded once, ~28 MB), falling back to the bundled 64-dim
    /// `potion-base-2M` when it cannot be fetched. An EXISTING store always
    /// reopens at the dimension it already holds, so this never strands
    /// data. In-memory stores stay on the bundled model deliberately.
    ///
    /// Raises `RuntimeError` if the wheel was built with
    /// `--no-default-features` (the `embedder-download` Cargo feature is
    /// off) or if exclusive access to the engine isn't available (e.g.
    /// any `_conn` proxy is still live — drop it first).
    #[cfg(feature = "embedder-download")]
    fn set_embedder_named(&mut self, name: &str) -> PyResult<()> {
        if self.inner.is_none() {
            return Err(PyRuntimeError::new_err("YantrikDB is closed"));
        }
        // The engine swap needs exclusive (`&mut`) access via `Arc::get_mut`,
        // which requires strong count 1 AND weak count 0. Since v0.9.0 the
        // constructors spawn a background worker pool (materializer + compactor)
        // whose threads hold `Weak<YantrikDB>` refs (issue #58) — those alone
        // make `get_mut` return `None`. So stop the workers first: dropping the
        // guards JOINs the threads, releasing their weak refs. Respawn after the
        // swap (success or failure) so the engine keeps its compactor — without
        // it, writes wedge at delta_max. This is an infrequent, operator-
        // initiated call, so the brief worker pause is acceptable.
        self._workers = None;

        let swap = {
            let arc = self.inner.as_mut().expect("checked is_some above");
            match Arc::get_mut(arc) {
                Some(engine) => engine.set_embedder_named(name).map_err(map_err),
                // A real external clone is still live (e.g. a ConnectionProxy or
                // a cloned YantrikDB) — the original, legitimate guard. Preserve
                // its actionable message.
                None => Err(PyRuntimeError::new_err(
                    "set_embedder_named requires exclusive access to the engine; \
                     drop any ConnectionProxy / cloned YantrikDB references before calling",
                )),
            }
        };

        self.respawn_workers();
        swap
    }

    /// Slim-build stub. When the Python crate is built with
    /// `--no-default-features` the embedder-download path compiles out
    /// entirely; this stub raises so callers get a clear actionable error
    /// instead of `AttributeError: 'YantrikDB' object has no attribute
    /// 'set_embedder_named'` (which is the wrong shape — they didn't
    /// mistype, the feature is just absent).
    #[cfg(not(feature = "embedder-download"))]
    fn set_embedder_named(&mut self, _name: &str) -> PyResult<()> {
        Err(PyRuntimeError::new_err(
            "set_embedder_named requires the 'embedder-download' Cargo feature, \
             which is on by default. This wheel was built --no-default-features. \
             Either rebuild with default features or use YantrikDB.with_default() \
             for the bundled potion-base-2M embedder.",
        ))
    }

    /// The _conn property — returns a ConnectionProxy for test compatibility.
    #[getter]
    fn _conn(&self) -> PyResult<ConnectionProxy> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
        Ok(ConnectionProxy { db: Arc::clone(db) })
    }

    /// The actor_id of this YantrikDB instance (read-only).
    #[getter]
    fn actor_id(&self) -> PyResult<String> {
        let db = self.get_inner()?;
        Ok(db.actor_id().to_string())
    }

    #[pyo3(signature = (namespace=None))]
    fn stats(&self, py: Python<'_>, namespace: Option<&str>) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let s = db.stats(namespace).map_err(map_err)?;
        stats_to_dict(py, &s)
    }

    /// v0.10 Item 1 — whether the status-led read path is active
    /// (superseded records excluded from recall eligibility). True by
    /// default on fresh databases; False on migrated pre-v0.10
    /// databases until opted in via `set_status_read_policy(True)`.
    fn status_read_policy(&self) -> PyResult<bool> {
        Ok(self.get_inner()?.status_read_policy())
    }

    /// v0.10 Item 1 — durably set the status read policy
    /// (`meta.status_read_policy`). Legacy-database opt-in: review
    /// `stats()["superseded_served_since_boot"]`, then call
    /// `set_status_read_policy(True)` to exclude superseded records
    /// from recall. `False` returns to legacy include-everything.
    fn set_status_read_policy(&self, exclude_superseded: bool) -> PyResult<()> {
        self.get_inner()?
            .set_status_read_policy(exclude_superseded)
            .map_err(map_err)
    }

    /// Maximum local verified-synthesis fan-out per evidence record.
    fn synthesis_fanout_cap(&self) -> PyResult<usize> {
        self.get_inner()?.synthesis_fanout_cap().map_err(map_err)
    }

    /// Durably set the local synthesis fan-out admission ceiling.
    fn set_synthesis_fanout_cap(&self, cap: usize) -> PyResult<()> {
        self.get_inner()?
            .set_synthesis_fanout_cap(cap)
            .map_err(map_err)
    }

    /// **v0.10 Item 4a.4** — the active anti-laundering gate mode:
    /// `"off"` | `"warn"` | `"enforce"`. Fresh databases default to `enforce`;
    /// MIGRATED databases default to `warn` (the gate counts violations into
    /// `stats()["provenance_flagged_since_boot"]` but never refuses), so an
    /// upgrade cannot break existing callers.
    fn provenance_gate_mode(&self) -> PyResult<String> {
        Ok(self
            .get_inner()?
            .provenance_gate_mode()
            .as_str()
            .to_string())
    }

    /// Durably opt in to a gate mode — the migration path. Review
    /// `stats()["provenance_flagged_since_boot"]` on a migrated database, fix
    /// the flagged callers, then `set_provenance_gate_mode("enforce")`.
    /// Accepts `"off"` | `"warn"` | `"enforce"`; a malformed value is a loud
    /// error (never a silent disable).
    fn set_provenance_gate_mode(&self, mode: &str) -> PyResult<()> {
        let parsed = yantrikdb_core::provenance::GateMode::parse(mode).map_err(map_err)?;
        self.get_inner()?
            .set_provenance_gate_mode(parsed)
            .map_err(map_err)
    }

    /// Exposed for Python consolidate.py compatibility.
    #[pyo3(signature = (op_type, target_rid, payload))]
    fn _log_op(
        &self,
        op_type: &str,
        target_rid: Option<&str>,
        payload: &Bound<'_, PyDict>,
    ) -> PyResult<String> {
        let db = self.get_inner()?;
        let payload_json = py_to_json(&payload.as_any())?;
        db.log_op(op_type, target_rid, &payload_json, None)
            .map_err(map_err)
    }

    /// Close the engine. Returns `True` when THIS call performed the
    /// exclusive close; `False` when other live references (e.g. a `_conn`
    /// proxy still held) kept the database open — it closes when the last
    /// reference drops. The old signature returned bare Ok(()) in both
    /// cases, so "db.close() succeeded" could mean "still open and
    /// writable through the proxy" (2026-08-15 binding audit).
    fn close(&mut self) -> PyResult<bool> {
        if let Some(arc) = self.inner.take() {
            match Arc::try_unwrap(arc) {
                Ok(db) => {
                    db.close().map_err(map_err)?;
                    return Ok(true);
                }
                Err(_arc) => {
                    // Other references still exist; ours is dropped, the
                    // DB closes on last drop — but NOT yet, and the caller
                    // deserves to know.
                    return Ok(false);
                }
            }
        }
        // Already closed: idempotent, and truthfully "no exclusive close
        // performed by this call".
        Ok(false)
    }
}

impl PyYantrikDB {
    /// Get a reference to the inner YantrikDB engine (for use by consolidation/trigger wrappers).
    pub fn get_inner(&self) -> PyResult<&YantrikDB> {
        self.inner
            .as_deref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))
    }

    pub(crate) fn embed_text(&self, py: Python<'_>, text: &str) -> PyResult<Vec<f32>> {
        // Try Rust-native embedder first (candle or any Embedder impl)
        if let Some(db) = &self.inner {
            if db.has_embedder() {
                return db.embed(text).map_err(map_err);
            }
        }

        // Fall back to Python embedder
        match &self.embedder {
            Some(emb) => {
                let result = emb.call_method1(py, "encode", (text,))?;
                // Handle both list and numpy array returns
                if let Ok(list) = result.extract::<Vec<f32>>(py) {
                    Ok(list)
                } else {
                    // Try calling .tolist() for numpy arrays
                    let list = result.call_method0(py, "tolist")?;
                    list.extract::<Vec<f32>>(py)
                }
            }
            None => Err(PyRuntimeError::new_err(
                "No embedder configured. Pass an embedder to YantrikDB() or call set_embedder().",
            )),
        }
    }
}
