mod cognition;
mod graph;
mod inspect;
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
    /// Bumped whenever `embedder` is replaced (`set_embedder`). A binding
    /// that planned its vector with one Python embedder checks it before it
    /// uses the vector (`planned`).
    embedder_epoch: u64,
    /// Background worker pool (materializers + compactor). MUST be held for the
    /// engine's lifetime: without the compactor the in-memory delta tier fills
    /// to `delta_max` (256) and every subsequent write returns Backpressure
    /// ("ingest queue full"). The pyo3 constructors previously forgot to spawn
    /// these, wedging any long write session (e.g. writing a full novel) after
    /// ~256 records. Dropped with the struct → Weak<YantrikDB> upgrade fails →
    /// workers shut down cleanly.
    ///
    /// Held behind its own lock, not the wrapper's borrow: `set_embedder_named`
    /// and `close()` stop and restart it while other threads may hold the
    /// wrapper's borrow, and a restart that needed that borrow could be
    /// refused and skipped, leaving the store without workers (#302 review 3).
    pub(crate) workers: Arc<WorkerPool>,
    /// Set while `set_embedder_named` waits for exclusive access to the
    /// engine. Calls that take an engine snapshot wait (with the GIL
    /// released) while it is set, so steady traffic cannot keep the engine
    /// shared forever; in-flight calls finish normally. See
    /// [`PyYantrikDB::snapshot_borrow`].
    pub(crate) swap_pending: Arc<std::sync::atomic::AtomicBool>,
    /// Held by the one `set_embedder_named` running on this wrapper: swaps
    /// take turns (#302 review 4). Only that method uses it, so the slim
    /// build (`--no-default-features`) has none.
    #[cfg(feature = "embedder-download")]
    swap_turn: Arc<std::sync::atomic::AtomicBool>,
    /// Tests only (`_inject_swap_panic`): the next swap panics just before
    /// it installs the new embedder, where the workers are stopped.
    inject_swap_panic: bool,
}

/// The wrapper's background worker pool (materializers + compactor), with
/// its own lock so it can be stopped and restarted without the wrapper's
/// borrow.
pub(crate) struct WorkerPool {
    guards: std::sync::Mutex<Option<AllWorkerGuards>>,
    /// How many pools have been spawned. Read by the tests
    /// (`_worker_spawns`): a swap that waits must not restart the pool on
    /// every try.
    spawns: std::sync::atomic::AtomicU64,
}

impl WorkerPool {
    /// A pool already running against `engine`.
    fn running(engine: &Arc<YantrikDB>) -> Arc<Self> {
        Arc::new(Self {
            guards: std::sync::Mutex::new(Some(spawn_all_workers(
                engine,
                recommended_worker_count(),
            ))),
            spawns: std::sync::atomic::AtomicU64::new(1),
        })
    }

    fn guards(&self) -> std::sync::MutexGuard<'_, Option<AllWorkerGuards>> {
        // A panic while the lock was held leaves the Option intact.
        self.guards.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Take the running workers out. Dropping what this returns stops them
    /// (it joins the threads, a materializer mid-drain included), so drop it
    /// with the GIL released.
    fn take(&self) -> Option<AllWorkerGuards> {
        self.guards().take()
    }

    /// Whether workers are running (tests only, through `_workers_running`).
    fn is_running(&self) -> bool {
        self.guards().is_some()
    }

    /// Start workers against `engine` unless some are running.
    fn ensure_running(&self, engine: &Arc<YantrikDB>) {
        let mut guards = self.guards();
        if guards.is_none() {
            *guards = Some(spawn_all_workers(engine, recommended_worker_count()));
            self.spawns
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

impl PyYantrikDB {
    /// Wrap a freshly-constructed engine AND spawn its background workers
    /// (materializer pool + compactor). This is the single place that owns the
    /// "don't forget the compactor" rule for the Python binding.
    pub(crate) fn from_engine(inner: YantrikDB, embedder: Option<PyObject>) -> Self {
        let inner = Arc::new(inner);
        let workers = WorkerPool::running(&inner);
        Self {
            inner: Some(inner),
            embedder,
            embedder_epoch: 0,
            workers,
            swap_pending: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            #[cfg(feature = "embedder-download")]
            swap_turn: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            inject_swap_panic: false,
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
        E::ForeignSqliteInstance { .. } => py_errors::ForeignSqliteInstance::new_err(e.to_string()),
        E::IntegrityCheckFailed { .. } => py_errors::IntegrityCheckFailed::new_err(e.to_string()),
        // The commit hook's abort surfaces from SQLite as
        // SQLITE_CONSTRAINT_COMMITHOOK. Public writes are pre-checked and
        // get the typed error for their cause; an abort only reaches here
        // when the guard tripped between the pre-check and the commit, and
        // this mapping cannot tell which cause it was — so it says both.
        E::Database(rusqlite::Error::SqliteFailure(ref err, _))
            if err.extended_code == yantrikdb_core::engine::SQLITE_CONSTRAINT_COMMITHOOK =>
        {
            py_errors::ForeignSqliteInstance::new_err(
                "refusing to write: the engine's write guard aborted the commit. Either \
                 another SQLite library has this store open in this process (issue #225; \
                 stats()[\"foreign_sqlite_tainted\"]) or the store's integrity check failed \
                 (issue #247; stats()[\"integrity_tainted\"], recover with integrity_check()).",
            )
        }
        E::PackEmbedderMismatch { .. } => py_errors::PackEmbedderMismatch::new_err(e.to_string()),
        E::PackAlreadyMounted { .. } => py_errors::PackAlreadyMounted::new_err(e.to_string()),
        E::PackSignatureInvalid { .. } => py_errors::PackSignatureInvalid::new_err(e.to_string()),
        E::PackNotMounted { .. } => py_errors::PackNotMounted::new_err(e.to_string()),
        // v50 thread v2: both variants are actionable — the first names
        // the route to drop, the second the maintenance op to run.
        E::CapabilityUnavailable { .. } => {
            py_errors::PhraseRouteUnavailableError::new_err(e.to_string())
        }
        E::MaintenanceRequired { .. } => {
            py_errors::SourceTurnMaintenanceRequiredError::new_err(e.to_string())
        }
        E::InvalidThreadTopic { .. } => py_errors::InvalidThreadTopicError::new_err(e.to_string()),
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
        slf: &Bound<'_, Self>,
        embedder: PyObject,
        allow_unverified_embedder: bool,
    ) -> PyResult<()> {
        let py = slf.py();
        // #282: this method used to take `&mut self`, so pyo3 held the
        // wrapper's mutable borrow for the whole call — including the
        // probe below, which calls back into Python. Any other thread
        // touching the wrapper during the probe was refused with
        // `RuntimeError: Already mutably borrowed` (measured at 99.998%
        // of concurrent recall() calls). The borrow is only needed for
        // the final field store, so the call now runs in three phases:
        // validate with no borrow held, identity-check under a brief
        // shared borrow, then borrow mutably for the store alone.

        // Phase 1, no borrow held: probe with a sentinel — if encode()
        // doesn't produce a numeric vector, the embedder is bogus and we
        // raise immediately.
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

        // Phase 2, shared borrow only: #117 — the probe above proves it
        // is AN embedder. This proves it is THE embedder — the one whose
        // space the stored vectors live in. The helper takes `&self`; a
        // shared borrow cannot refuse the `&self` crowd #282 is about.
        slf.try_borrow()?
            .check_embedder_identity(py, &embedder, allow_unverified_embedder)?;

        // Phase 3, mutable borrow for the store alone — one field
        // assignment, no Python call inside it. The embedder it replaces is
        // dropped after the borrow ends: dropping it can run its __del__,
        // which may call back into this store. A close() can now land
        // between validation and the store, so re-check the engine is
        // still there. This changes behaviour: on main, set_embedder()
        // on a closed wrapper returned None, because
        // check_embedder_identity returns Ok when inner is None. It
        // now raises instead — storing an embedder onto a closed
        // wrapper must not look like success. (has_embedder() is not
        // the concern here: it goes through get_inner()? and already
        // raises on a closed wrapper.)
        let previous = {
            let mut borrowed = slf.try_borrow_mut()?;
            if borrowed.inner.is_none() {
                return Err(PyRuntimeError::new_err("YantrikDB is closed"));
            }
            borrowed.embedder_epoch += 1;
            borrowed.embedder.replace(embedder)
        };
        drop(previous);
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
    /// off), or if exclusive access to the engine is still unavailable
    /// after 10 seconds: another thread's call on this database never
    /// finished, or a `_conn` proxy or cloned reference is still alive.
    #[cfg(feature = "embedder-download")]
    fn set_embedder_named(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        if slf.try_borrow()?.inner.is_none() {
            return Err(PyRuntimeError::new_err("YantrikDB is closed"));
        }
        // #271: fetch() downloads the model on first use (tens of MB),
        // verifies its SHA-256 and loads it, from the cache on later calls.
        // All of that runs with no borrow of the wrapper and the GIL
        // released, so every other thread keeps running and keeps using
        // this store. The swap below waits for exclusive access with the
        // GIL released too; it holds the GIL only for the engine swap
        // itself and to restart the workers' handles. The workers are
        // stopped (joined) with the GIL released.
        let downloaded = slf
            .py()
            .detach(|| yantrikdb_core::embedder::DownloadedEmbedder::fetch(name))
            .map_err(map_err)?;

        // The engine swap needs exclusive (`&mut`) access via `Arc::get_mut`,
        // which requires strong count 1 AND weak count 0. The background
        // workers (materializers + compactor) hold `Weak<YantrikDB>` refs
        // (issue #58), which alone make `get_mut` return `None`, so they are
        // stopped for the swap itself. Not for the wait: the calls the swap
        // waits for can need them (a write past delta_max waits on the
        // compactor), and stopping and restarting them on every try started
        // a pool every ~1.5 ms.
        //
        // Every detached call holds a strong ref while it runs. Under the GIL
        // those calls had finished before this one started; now they overlap,
        // and steady traffic keeps starting new ones. So raise the swap gate:
        // new calls wait at their engine snapshot (`snapshot_borrow`) while
        // the ones in flight finish, and this call waits, with no borrow held,
        // the GIL released and the workers running, until only the wrapper
        // holds the engine. Then it stops the workers, takes the wrapper's
        // mutable borrow and swaps. If the engine was taken again in between,
        // or another thread holds the wrapper's borrow (a call inside a
        // Python embedder's encode()), it restarts the workers and waits on,
        // trying again at most every 50 ms. Past the deadline the holder is
        // most likely a `_conn` proxy or a cloned reference, which no wait
        // releases.
        //
        // The swap holds no reference to the engine, not even a Weak: it
        // reads the strong count through the wrapper's own Arc on each try.
        // A Weak held across the wait blocks `get_mut` too, so two swaps
        // pending at once each blocked the other until both deadlines ran
        // out. Swaps on one wrapper also take turns (`swap_turn`), the second
        // waiting with the GIL released (#302 review 4).
        //
        // The workers are restarted on every way out (`Restart`), an early
        // `?` and a panic included, from their own pool, which needs no
        // borrow of the wrapper: a store left without workers wedges at
        // delta_max (256 writes) (#302 review 3).
        const WAIT: std::time::Duration = std::time::Duration::from_secs(10);
        const RETRY_EVERY: std::time::Duration = std::time::Duration::from_millis(50);
        const BORROW_FOR: std::time::Duration = std::time::Duration::from_millis(10);
        struct LowerGate(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for LowerGate {
            fn drop(&mut self) {
                self.0.store(false, std::sync::atomic::Ordering::Release);
            }
        }
        /// Gives up this wrapper's swap turn when dropped.
        struct EndTurn(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for EndTurn {
            fn drop(&mut self) {
                self.0.store(false, std::sync::atomic::Ordering::Release);
            }
        }
        /// Restarts stopped workers when dropped, from the wrapper's current
        /// engine, unless the wrapper is closed. It needs no reference of its
        /// own to the engine, so it can restart them from anywhere in the
        /// swap, a panic while the new embedder is installed included.
        struct Restart<'a, 'py> {
            slf: &'a Bound<'py, PyYantrikDB>,
            pool: Arc<WorkerPool>,
        }
        impl Restart<'_, '_> {
            fn now(&self) {
                let this = retry_until(self.slf.py(), std::time::Duration::from_secs(1), || {
                    self.slf.try_borrow()
                });
                if let Some(engine) = this.ok().and_then(|this| this.inner.clone()) {
                    self.pool.ensure_running(&engine);
                }
            }
        }
        impl Drop for Restart<'_, '_> {
            fn drop(&mut self) {
                self.now();
            }
        }
        let (pool, gate_flag, turn) = {
            let this = Self::borrow_briefly(slf)?;
            if this.inner.is_none() {
                return Err(PyRuntimeError::new_err("YantrikDB is closed"));
            }
            (
                this.workers.clone(),
                this.swap_pending.clone(),
                this.swap_turn.clone(),
            )
        };
        // Another swap on this wrapper finishes first; its own deadline
        // bounds this wait.
        while turn
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .is_err()
        {
            slf.py()
                .detach(|| std::thread::sleep(std::time::Duration::from_millis(1)));
        }
        let _turn = EndTurn(turn);
        gate_flag.store(true, std::sync::atomic::Ordering::Release);
        let gate = LowerGate(gate_flag);
        let restart = Restart { slf, pool };

        let mut downloaded = Some(downloaded);
        let deadline = std::time::Instant::now() + WAIT;
        let mut last_try: Option<std::time::Instant> = None;
        // What stood in the way last: the wrapper's borrow (held by another
        // thread) rather than another holder of the engine. The timeout
        // message names it.
        let mut borrow_blocked = false;
        let swap = loop {
            let now = std::time::Instant::now();
            let late = now >= deadline;
            // Look at the engine through the wrapper. A close() that landed
            // meanwhile has settled the workers for whatever still holds the
            // engine (it restarts them when a proxy or a call keeps it
            // alive): stop nothing and restart nothing (#302 review 4). A
            // mutable borrow held elsewhere: not this time.
            let holders = match slf.try_borrow() {
                Ok(this) => match this.inner.as_ref() {
                    None => break Err(PyRuntimeError::new_err("YantrikDB is closed")),
                    Some(arc) => Some(Arc::strong_count(arc)),
                },
                Err(_) => None,
            };
            let quiet = holders.is_some_and(|n| n <= 1);
            match holders {
                None => borrow_blocked = true,
                Some(n) if n > 1 => borrow_blocked = false,
                Some(_) => {}
            }
            let due = last_try.is_none_or(|t| now.duration_since(t) >= RETRY_EVERY);
            if quiet && (due || late) {
                last_try = Some(now);
                let stopped = restart.pool.take();
                slf.py().detach(move || drop(stopped));
                match Self::borrow_mut_within(slf, BORROW_FOR) {
                    Ok(mut this) => {
                        let inject_panic = std::mem::take(&mut this.inject_swap_panic);
                        let Some(arc) = this.inner.as_mut() else {
                            break Err(PyRuntimeError::new_err("YantrikDB is closed"));
                        };
                        if let Some(engine) = Arc::get_mut(arc) {
                            if inject_panic {
                                panic!("injected by _inject_swap_panic (tests only)");
                            }
                            let swapped = engine
                                .set_downloaded_embedder(downloaded.take().expect("swapped once"))
                                .map_err(map_err);
                            restart.pool.ensure_running(arc);
                            break swapped;
                        }
                        restart.pool.ensure_running(arc);
                        borrow_blocked = false;
                    }
                    Err(_) => {
                        borrow_blocked = true;
                        restart.now();
                    }
                }
            }
            if late && borrow_blocked {
                break Err(PyRuntimeError::new_err(format!(
                    "set_embedder_named needs exclusive access to the engine and did not \
                     get it within {}s: the engine was free, but another thread held a \
                     borrow of this YantrikDB each time (a record() inside its Python \
                     embedder's encode(), for example). Retry once that call is done.",
                    WAIT.as_secs()
                )));
            }
            if late {
                break Err(PyRuntimeError::new_err(format!(
                    "set_embedder_named needs exclusive access to the engine and did not \
                     get it within {}s: another thread's call on this database is still \
                     running (waiting on a lock held by another process, for example), \
                     or a ConnectionProxy (`_conn`) or cloned YantrikDB reference is \
                     still alive. Let the call finish or drop the reference, then retry.",
                    WAIT.as_secs()
                )));
            }
            slf.py()
                .detach(|| std::thread::sleep(std::time::Duration::from_millis(1)));
        };
        drop(restart);
        drop(gate);
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

    /// Re-embed every stored memory with a new named embedder, resolving the
    /// model through the embedder-download registry (downloading it on first
    /// use) and returning the reembed report as a dict. Raises if the wheel
    /// was built `--no-default-features`, if the engine refuses the call (a
    /// `Pause` write policy, namespace scoping or `resume_from_checkpoint`),
    /// or if the model cannot be resolved.
    #[cfg(feature = "embedder-download")]
    fn reembed(&self, py: Python<'_>, new_embedder_name: &str) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let report = db
            .reembed(
                new_embedder_name,
                yantrikdb_core::engine::reembed::ReembedOptions::default(),
            )
            .map_err(map_err)?;
        let value =
            serde_json::to_value(&report).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        json_to_py(py, &value)
    }

    /// Slim-build stub, mirroring set_embedder_named.
    #[cfg(not(feature = "embedder-download"))]
    fn reembed(&self, _py: Python<'_>, _new_embedder_name: &str) -> PyResult<PyObject> {
        Err(PyRuntimeError::new_err(
            "reembed requires the 'embedder-download' Cargo feature, \
             which is on by default. This wheel was built --no-default-features.",
        ))
    }

    /// Make the next `set_embedder_named` panic just before it installs
    /// the new embedder, where the workers are stopped (tests only).
    fn _inject_swap_panic(&mut self) {
        self.inject_swap_panic = true;
    }

    /// Whether this wrapper's worker pool is running (tests only).
    #[getter]
    fn _workers_running(&self) -> bool {
        self.workers.is_running()
    }

    /// How many worker pools this wrapper has spawned (tests only).
    #[getter]
    fn _worker_spawns(&self) -> u64 {
        self.workers
            .spawns
            .load(std::sync::atomic::Ordering::Relaxed)
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
    fn stats(slf: &Bound<'_, Self>, namespace: Option<&str>) -> PyResult<PyObject> {
        let py = slf.py();
        // No borrow of `slf` is held across the detached call; see
        // `engine_handle` for why that matters.
        let db = Self::engine_handle(slf)?;
        let namespace = namespace.map(str::to_owned);
        // Release the GIL for the engine call (#271). `stats()` aggregates
        // over the store and, without this, holds the GIL for its whole
        // duration, stalling every other thread in the host while it runs.
        // `detach` is PyO3 0.28's name for what older docs call
        // `allow_threads`. Nothing in the closure touches Python: the
        // namespace was converted to an owned string above, and the result
        // dict is built after reattaching. `Arc<YantrikDB>` is owned and
        // `Send + Sync`, which is what `Ungil` requires.
        let s = py
            .detach(move || db.stats(namespace.as_deref()))
            .map_err(map_err)?;
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

    /// `"off"` | `"shadow"` | `"enforce"`: whether the claims lane admits
    /// only grounded, currently-valid claims. Every database defaults to
    /// `shadow`, which changes no result and counts what `enforce` would
    /// refuse into `stats()["claim_chain_gate_suppressed_since_boot"]`.
    fn claim_chain_gate_mode(&self) -> PyResult<String> {
        Ok(self
            .get_inner()?
            .claim_chain_gate_mode()
            .as_str()
            .to_string())
    }

    /// Durably set the claim-chain gate mode. Read the suppression counters
    /// under `shadow` first; a malformed value is a loud error.
    fn set_claim_chain_gate_mode(&self, mode: &str) -> PyResult<()> {
        let parsed = yantrikdb_core::ChainGateMode::parse(mode).map_err(map_err)?;
        self.get_inner()?
            .set_claim_chain_gate_mode(parsed)
            .map_err(map_err)
    }

    /// `"off"` | `"warn"` | `"refuse"`: what the engine does when another
    /// SQLite library (the stdlib `sqlite3` module, a system libsqlite3)
    /// has this store open in this process — the silent-corruption hazard
    /// of issue #225. Every database defaults to `refuse`: writes fail with
    /// `ForeignSqliteInstance` from the first detection until the engine is
    /// reopened (the foreign library's close may unlink the shm/WAL under it). Detection needs Linux or
    /// macOS and a file-backed store (`stats()["foreign_sqlite_supported"]`).
    /// The mode also governs the refusal after a failed integrity check
    /// (`IntegrityCheckFailed`, every platform), which `integrity_check()`
    /// lifts once the store checks clean.
    fn foreign_sqlite_mode(&self) -> PyResult<String> {
        Ok(self.get_inner()?.foreign_sqlite_mode().as_str().to_string())
    }

    /// Durably set the guard mode. `warn` keeps writing and counts; `off`
    /// never scans. Both opt out of a corruption guard.
    fn set_foreign_sqlite_mode(&self, mode: &str) -> PyResult<()> {
        let parsed = yantrikdb_core::ForeignSqliteMode::parse(mode).map_err(map_err)?;
        self.get_inner()?
            .set_foreign_sqlite_mode(parsed)
            .map_err(map_err)
    }

    /// Scan now: is a second SQLite library holding this store open in this
    /// process? Always False where detection is unsupported.
    fn foreign_sqlite_detected(&self) -> PyResult<bool> {
        Ok(self.get_inner()?.foreign_sqlite_detected())
    }

    /// `PRAGMA quick_check` on a read connection, recorded in
    /// `stats()["last_integrity_check"]`. A failure is re-run once on the
    /// writer connection; a confirmed one makes the engine refuse writes
    /// with `IntegrityCheckFailed` until a later check returns "ok" — so
    /// after repairing the store, call this to resume writing (no reopen).
    /// Queued automatically whenever a commit from outside this engine is
    /// seen.
    fn integrity_check(&self) -> PyResult<String> {
        self.get_inner()?.integrity_check().map_err(map_err)
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
    ///
    /// The background workers are stopped first. They hold only `Weak`
    /// refs, but each one upgrades to a strong ref while it works, and the
    /// compactor used to hold one across its whole idle wait, so this
    /// returned False almost every time with nothing else holding the
    /// engine (#271 review).
    fn close(slf: &Bound<'_, Self>) -> PyResult<bool> {
        // Take the handle and the workers under a brief borrow, then stop
        // the workers (joining a materializer mid-drain) and close SQLite (a
        // WAL checkpoint) with the GIL released: held through both, a thread
        // beside the close froze for its whole length, up to 1.8 s under
        // write load (#271 review).
        //
        // Nothing after the first borrow needs the wrapper again: the
        // workers come from their own pool, so a borrow another thread holds
        // by then cannot cost the store its workers (#302 review 3). That is
        // defensive: since #295 no method holds the wrapper's borrow for any
        // length once the store is closed (set_embedder() no longer holds it
        // across its probe, and every other method reports a closed store
        // at once), so the public API cannot reach that state, and no test
        // drives it.
        let (arc, pool) = {
            let mut this = Self::borrow_mut_briefly(slf)?;
            match this.inner.take() {
                Some(arc) => (arc, this.workers.clone()),
                // Already closed: idempotent, and truthfully "no exclusive
                // close performed by this call".
                None => return Ok(false),
            }
        };
        slf.py()
            .detach(move || -> Result<bool, yantrikdb_core::YantrikDbError> {
                drop(pool.take());
                match Arc::try_unwrap(arc) {
                    Ok(db) => {
                        db.close()?;
                        Ok(true)
                    }
                    // Other references still exist (another thread's call in
                    // flight, or a `_conn` proxy); ours drops here and the DB
                    // closes on the last drop, but NOT yet, and the caller
                    // deserves to know. Until then the engine still takes
                    // writes through those references, so it keeps its
                    // workers: without the compactor, writes wedge at
                    // delta_max. They hold only Weak refs and exit when the
                    // engine is finally dropped.
                    Err(arc) => {
                        pool.ensure_running(&arc);
                        Ok(false)
                    }
                }
            })
            .map_err(map_err)
    }
}

/// Call `attempt` until it succeeds or `within` has passed, sleeping a
/// millisecond with the GIL released between tries. The last refusal is
/// the error.
fn retry_until<T, E: Into<PyErr>>(
    py: Python<'_>,
    within: std::time::Duration,
    mut attempt: impl FnMut() -> Result<T, E>,
) -> PyResult<T> {
    let deadline = std::time::Instant::now() + within;
    loop {
        match attempt() {
            Ok(value) => return Ok(value),
            Err(e) if std::time::Instant::now() >= deadline => return Err(e.into()),
            Err(_) => py.detach(|| std::thread::sleep(std::time::Duration::from_millis(1))),
        }
    }
}

/// Convert a Python embedder's `encode()` output to a vector, accepting
/// `list[float]` and numpy arrays (via `.tolist()`).
///
/// Shared by `embed_text()` and the window probe so both interpret an
/// embedder's return value identically.
pub(crate) fn py_vector(py: Python<'_>, result: &PyObject) -> PyResult<Vec<f32>> {
    if let Ok(list) = result.extract::<Vec<f32>>(py) {
        return Ok(list);
    }
    let list = result.call_method0(py, "tolist")?;
    list.extract::<Vec<f32>>(py)
}

/// What a binding plans its vector with, taken under one brief borrow and
/// holding no reference to the engine (see [`PyYantrikDB::planned`]).
pub(crate) struct EmbedderSnapshot {
    /// The caller-attached Python embedder, which outranks the engine's.
    pub(crate) embedder: Option<PyObject>,
    /// Whether the engine has an embedder of its own.
    pub(crate) native: bool,
    epoch: u64,
}

/// Where a binding's vector comes from, decided once, from the embedder
/// snapshot [`PyYantrikDB::planned`] took (#271).
pub(crate) enum EmbedPlan<'a> {
    /// A vector already in hand: passed by the caller, or produced by the
    /// caller-attached Python embedder (with the GIL held, as any Python
    /// call must be, but with no borrow of the wrapper).
    Ready(Vec<f32>),
    /// The engine's own embedder, run inside the detached closure with the
    /// engine call it feeds: native inference needs no GIL, and one detach
    /// per call keeps the cost to one GIL handoff. Borrows the caller's
    /// text: a `&str` argument outlives the call, detach included.
    Native(&'a str),
}

impl EmbedPlan<'_> {
    /// The vector, inside `py.detach`.
    pub(crate) fn into_vector(
        self,
        db: &YantrikDB,
    ) -> Result<Vec<f32>, yantrikdb_core::YantrikDbError> {
        match self {
            EmbedPlan::Ready(v) => Ok(v),
            EmbedPlan::Native(text) => db.embed(text),
        }
    }
}

/// Run one engine call with the GIL released (#271).
///
/// `db` is an owned handle ([`PyYantrikDB::engine_handle`] or
/// [`PyYantrikDB::planned`]), so no borrow of the wrapper is held
/// while `f` runs. Every binding that reaches SQLite goes through here, not
/// only the slow ones: the engine serialises on one connection mutex, so even
/// a point read can wait out another thread's write (up to `busy_timeout`),
/// and a binding that waits with the GIL held stops every Python thread in
/// the host for that long.
///
/// `f` may borrow the call's `&str` and other plain-Rust arguments: they are
/// `Send`, and the Python objects they come from are kept alive by the call
/// itself. Anything Python-bound (a dict, a callback) must be converted
/// before the call and the result converted after it; the `Send` bounds
/// enforce that.
pub(crate) fn run_detached<T, F>(py: Python<'_>, db: Arc<YantrikDB>, f: F) -> PyResult<T>
where
    F: FnOnce(&YantrikDB) -> Result<T, yantrikdb_core::YantrikDbError> + Send,
    T: Send,
{
    py.detach(move || f(&db)).map_err(map_err)
}

/// `embed_text`'s precedence, with nothing of the store held across
/// `encode()` (see [`PyYantrikDB::planned`]): an embedder the caller
/// ATTACHED wins over one the engine resolved for this store's dimension
/// (yantrikdb-hermes-plugin #84), then the engine's own, else the same
/// error `embed_text` raises.
pub(crate) fn plan_embed<'a>(
    py: Python<'_>,
    snapshot: &EmbedderSnapshot,
    text: &'a str,
) -> PyResult<EmbedPlan<'a>> {
    if let Some(emb) = snapshot.embedder.as_ref() {
        let result = emb.call_method1(py, "encode", (text,))?;
        return Ok(EmbedPlan::Ready(py_vector(py, &result)?));
    }
    if snapshot.native {
        return Ok(EmbedPlan::Native(text));
    }
    Err(PyRuntimeError::new_err(
        "No embedder configured. Pass an embedder to YantrikDB() or call set_embedder().",
    ))
}

/// The usual query shape: a vector the caller passed wins, else `text` is
/// embedded with [`plan_embed`], else `ValueError(missing)`.
pub(crate) fn query_plan<'a>(
    py: Python<'_>,
    snapshot: &EmbedderSnapshot,
    given: Option<Vec<f32>>,
    text: Option<&'a str>,
    missing: &'static str,
) -> PyResult<EmbedPlan<'a>> {
    match (given, text) {
        (Some(v), _) => Ok(EmbedPlan::Ready(v)),
        (None, Some(t)) => plan_embed(py, snapshot, t),
        (None, None) => Err(PyValueError::new_err(missing)),
    }
}

impl PyYantrikDB {
    /// The wrapper's mutable borrow, retried for up to a second instead of
    /// failing at once. Since the bindings release the GIL, another thread
    /// can hold a shared borrow for a moment (taking its engine snapshot,
    /// or around a Python embedder's encode()), and a close() or swap that
    /// arrives then was refused with "Already borrowed" (#271 review). The
    /// GIL is released between tries, so the holder can finish. The second
    /// is a deadline, not a count of tries: 1000 one-millisecond sleeps ran
    /// 1.3-2.4 s (#302 review 3).
    pub(crate) fn borrow_mut_briefly<'py>(slf: &Bound<'py, Self>) -> PyResult<PyRefMut<'py, Self>> {
        Self::borrow_mut_within(slf, std::time::Duration::from_secs(1))
    }

    /// [`borrow_mut_briefly`](Self::borrow_mut_briefly), retried for `within`.
    fn borrow_mut_within<'py>(
        slf: &Bound<'py, Self>,
        within: std::time::Duration,
    ) -> PyResult<PyRefMut<'py, Self>> {
        retry_until(slf.py(), within, || slf.try_borrow_mut())
    }

    /// The wrapper's shared borrow, retried for up to a second: refused only
    /// while another thread holds the mutable one (a `close()`, a swap, or
    /// `set_embedder()` storing its embedder).
    #[cfg(feature = "embedder-download")]
    fn borrow_briefly<'py>(slf: &Bound<'py, Self>) -> PyResult<PyRef<'py, Self>> {
        retry_until(slf.py(), std::time::Duration::from_secs(1), || {
            slf.try_borrow()
        })
    }

    /// An owned engine handle for a call that releases the GIL (#271).
    ///
    /// A `&self` pymethod keeps its `PyRef` borrow for the whole call. Once
    /// the engine call runs inside `py.detach`, other Python threads run
    /// too, and any `&mut self` method they call (`close`,
    /// `set_embedder_named`) fails with `RuntimeError: Already borrowed`
    /// for as long as the detached call lasts. Methods that detach therefore
    /// take `slf: &Bound<Self>`, borrow only long enough to clone the `Arc`,
    /// and run the engine with no borrow held. A concurrent `close()` then
    /// behaves as it always has with live references: it drops the
    /// wrapper's handle and returns `False`; the engine closes when the
    /// detached call releases the last reference.
    pub(crate) fn engine_handle(slf: &Bound<'_, Self>) -> PyResult<Arc<YantrikDB>> {
        Self::snapshot_borrow(slf)?
            .inner
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))
    }

    /// The shared borrow a call takes to snapshot the engine.
    ///
    /// While `set_embedder_named` waits for exclusive access
    /// (`swap_pending`), a new call waits here with the GIL released, so it
    /// does not take a fresh reference that keeps the engine shared; it then
    /// proceeds on the swapped engine.
    ///
    /// It waits for as long as the swap does: until the gate drops or the
    /// store closes. A call that went ahead mid-swap would take exactly the
    /// reference the gate keeps out, so a time limit of its own (this was
    /// 2 s) let steady traffic starve the swap again once it passed (#302
    /// review 3). The wait is still bounded: the swap gives up at its own
    /// deadline (10 s) and the gate drops on every way out of it.
    ///
    /// No call the swap waits for is itself waiting here. A binding runs a
    /// Python embedder's `encode()` before it takes the engine
    /// ([`planned`](Self::planned)), so a call that `encode()` makes back
    /// into the store, on its own thread or by handing it to another one,
    /// holds up nothing the swap waits for: it waits here, then runs on the
    /// swapped engine (#302 review 4). No detached call calls back into
    /// Python on this branch. `record()` and the other methods that embed
    /// through `embed_text` still hold the wrapper's borrow (but no engine
    /// reference) across `encode()`; a swap then waits on that borrow, and
    /// a call `encode()` makes back into the store waits here until the
    /// swap gives up (#297 moves them).
    ///
    /// try_borrow, not borrow: a method holding the mutable borrow
    /// (`close()`, the swap, `set_embedder()` storing its embedder) can be
    /// mid-call when another thread arrives here; that surfaces as
    /// RuntimeError, never as a panic. A refusal is retried for up to 10 ms
    /// first, which covers those brief mutable borrows; a longer one still
    /// gets the error.
    pub(crate) fn snapshot_borrow<'py>(slf: &Bound<'py, Self>) -> PyResult<PyRef<'py, Self>> {
        Self::shared_borrow(slf, true)
    }

    /// [`snapshot_borrow`](Self::snapshot_borrow), or with `gated` false
    /// the same borrow without waiting at the swap gate: for a look that
    /// takes no engine reference.
    fn shared_borrow<'py>(slf: &Bound<'py, Self>, gated: bool) -> PyResult<PyRef<'py, Self>> {
        const BORROW_RETRY: std::time::Duration = std::time::Duration::from_millis(10);
        let mut refused_since: Option<std::time::Instant> = None;
        loop {
            match slf.try_borrow() {
                Ok(this) => {
                    let pending =
                        gated && this.swap_pending.load(std::sync::atomic::Ordering::Acquire);
                    if !pending || this.inner.is_none() {
                        return Ok(this);
                    }
                }
                Err(e) => {
                    let since = *refused_since.get_or_insert_with(std::time::Instant::now);
                    if since.elapsed() >= BORROW_RETRY {
                        return Err(e.into());
                    }
                }
            }
            slf.py()
                .detach(|| std::thread::sleep(std::time::Duration::from_millis(1)));
        }
    }

    /// Get a reference to the inner YantrikDB engine (for use by consolidation/trigger wrappers).
    pub fn get_inner(&self) -> PyResult<&YantrikDB> {
        self.inner
            .as_deref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))
    }

    /// [`run_detached`] on this wrapper's engine: the common case, a binding
    /// that does not embed.
    pub(crate) fn detached<T, F>(slf: &Bound<'_, Self>, f: F) -> PyResult<T>
    where
        F: FnOnce(&YantrikDB) -> Result<T, yantrikdb_core::YantrikDbError> + Send,
        T: Send,
    {
        run_detached(slf.py(), Self::engine_handle(slf)?, f)
    }

    /// The engine handle and a binding's embedding plan, the plan made while
    /// the call holds nothing of the store (#302 review 4).
    ///
    /// A Python embedder's `encode()` can call back into the store, on its
    /// own thread or by handing the call to another thread and waiting for
    /// it. While it runs, this call holds no borrow of the wrapper and no
    /// reference to the engine: the embedder is taken under a brief borrow
    /// (not at the swap gate: it takes no engine reference), `plan` runs,
    /// and only then is the engine handle taken, at the gate. Holding the
    /// engine across `encode()` made a pending `set_embedder_named` wait for
    /// this call while the call's own callback waited at the gate for the
    /// swap, until the swap's 10 s deadline failed it.
    ///
    /// The plan must still fit the engine it is used on:
    /// - A Python vector depends only on the Python embedder that made it.
    ///   `set_embedder_named` replaces the engine's own embedder, which a
    ///   caller-attached Python one outranks, so a swap landing in between
    ///   leaves that vector valid.
    /// - A native plan embeds inside the detached call, on the engine taken
    ///   here.
    /// - What can make a plan stale is `set_embedder()` replacing (or first
    ///   attaching) the Python embedder in between. It bumps
    ///   `embedder_epoch`; the plan is then made again with the new one, at
    ///   most three times.
    pub(crate) fn planned<T>(
        slf: &Bound<'_, Self>,
        mut plan: impl FnMut(&EmbedderSnapshot) -> PyResult<T>,
    ) -> PyResult<(Arc<YantrikDB>, T)> {
        const TRIES: usize = 3;
        let py = slf.py();
        for _ in 0..TRIES {
            let snapshot = {
                let this = Self::shared_borrow(slf, false)?;
                let Some(db) = this.inner.as_deref() else {
                    return Err(PyRuntimeError::new_err("YantrikDB is closed"));
                };
                EmbedderSnapshot {
                    embedder: this.embedder.as_ref().map(|e| e.clone_ref(py)),
                    native: db.has_embedder(),
                    epoch: this.embedder_epoch,
                }
            };
            let made = plan(&snapshot)?;
            let this = Self::snapshot_borrow(slf)?;
            let db = this
                .inner
                .clone()
                .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;
            if this.embedder_epoch == snapshot.epoch {
                return Ok((db, made));
            }
        }
        Err(PyRuntimeError::new_err(
            "this store's Python embedder was replaced three times while this call was \
             embedding with it; retry the call",
        ))
    }

    pub(crate) fn embed_text(&self, py: Python<'_>, text: &str) -> PyResult<Vec<f32>> {
        // **An embedder the caller ATTACHED wins over one the engine merely
        // resolved for this store's dimension** (yantrikdb-hermes-plugin #84).
        //
        // This used to try the native embedder first. The wheel ships
        // `embedder-download`, so a store opened at 256 dims already has the
        // bundled potion-base-8M attached natively — and every
        // `set_embedder(obj)` whose model happened to be 256-d was therefore
        // discarded in silence. The caller's model never ran, nothing raised,
        // and the vectors came from a model they did not choose: someone who
        // picked a multilingual model to hold non-English memories got
        // English-centric vectors with no way to notice.
        //
        // Precedence is now intent-ordered. An explicit `set_embedder(obj)`
        // (or `embedder=` on the constructor) states which space this store's
        // vectors live in, so it outranks a default the engine attached on the
        // caller's behalf. Space-mixing stays guarded: `set_embedder` runs
        // `check_embedder_identity` before accepting the object, so attaching
        // one to a store with a recorded identity is refused as before.
        if let Some(emb) = &self.embedder {
            let result = emb.call_method1(py, "encode", (text,))?;
            return py_vector(py, &result);
        }

        if let Some(db) = &self.inner {
            if db.has_embedder() {
                return db.embed(text).map_err(map_err);
            }
        }

        Err(PyRuntimeError::new_err(
            "No embedder configured. Pass an embedder to YantrikDB() or call set_embedder().",
        ))
    }
}
