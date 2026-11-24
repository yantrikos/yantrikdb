use pyo3::prelude::*;
use pyo3::types::PyDict;
// pyo3 0.28 removed the `PyObject` top-level alias. Reintroduce locally —
// `Py<PyAny>` is the equivalent owned reference.
type PyObject = pyo3::Py<pyo3::PyAny>;

use yantrikdb_core::types::MemoryWithEmbedding;

use crate::py_engine::{py_embed, PyYantrikDB};
use crate::py_types::*;

/// Compute cosine similarity between two vectors.
#[pyfunction]
#[pyo3(name = "_cosine_similarity")]
pub fn py_cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> f64 {
    yantrikdb_core::consolidate::cosine_similarity(&a, &b)
}

/// Generate an extractive summary from a list of memory dicts.
#[pyfunction]
#[pyo3(name = "_extractive_summary")]
pub fn py_extractive_summary(memories: Vec<Bound<'_, PyDict>>) -> PyResult<String> {
    let mems: Vec<MemoryWithEmbedding> = memories
        .iter()
        .map(|d| dict_to_mem_with_embedding(d))
        .collect::<PyResult<_>>()?;
    Ok(yantrikdb_core::consolidate::extractive_summary(&mems))
}

/// Find clusters of related memories.
#[pyfunction]
#[pyo3(name = "_find_clusters", signature = (memories, sim_threshold=0.6, time_window_days=7.0, min_cluster_size=2, max_cluster_size=10))]
pub fn py_find_clusters(
    memories: Vec<Bound<'_, PyDict>>,
    sim_threshold: f64,
    time_window_days: f64,
    min_cluster_size: usize,
    max_cluster_size: usize,
) -> PyResult<Vec<Vec<PyObject>>> {
    let mems: Vec<MemoryWithEmbedding> = memories
        .iter()
        .map(|d| dict_to_mem_with_embedding(d))
        .collect::<PyResult<_>>()?;

    let cluster_indices = yantrikdb_core::consolidate::find_clusters(
        &mems,
        None, // entities_by_rid — not exposed to Python yet
        sim_threshold,
        time_window_days,
        min_cluster_size,
        max_cluster_size,
        // No seed exclusion: a Python caller cannot supply a predicate across
        // the FFI, and this binding is a raw utility. The engine paths the
        // plugin actually consolidates through (`consolidate`,
        // `find_consolidation_candidates`) apply the gist exclusion in Rust.
        None,
    );

    // Convert back: each cluster is a list of the original dicts
    let result: Vec<Vec<PyObject>> = cluster_indices
        .into_iter()
        .map(|indices| {
            indices
                .into_iter()
                .map(|i| memories[i].clone().into_any().unbind())
                .collect()
        })
        .collect();

    Ok(result)
}

/// Find consolidation candidates.
#[pyfunction]
#[pyo3(signature = (db, sim_threshold=0.6, time_window_days=7.0, min_cluster_size=2, limit=100))]
pub fn find_consolidation_candidates(
    py: Python<'_>,
    db: Bound<'_, PyYantrikDB>,
    sim_threshold: f64,
    time_window_days: f64,
    min_cluster_size: usize,
    limit: usize,
) -> PyResult<Vec<Vec<PyObject>>> {
    // Reaches SQLite: with the GIL released and no borrow of `db` held (#271).
    let inner = PyYantrikDB::engine_handle(&db)?;
    let clusters = py
        .detach(|| {
            yantrikdb_core::consolidate::find_consolidation_candidates(
                &inner,
                sim_threshold,
                time_window_days,
                min_cluster_size,
                limit,
                true, // require_entity_overlap — default since v0.6.0
            )
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Convert clusters to Python dicts
    clusters
        .iter()
        .map(|cluster| {
            cluster
                .iter()
                .map(|m| mem_with_emb_to_dict(py, m))
                .collect::<PyResult<Vec<_>>>()
        })
        .collect()
}

/// Run the full consolidation pipeline.
#[pyfunction]
#[pyo3(name = "consolidate", signature = (db, sim_threshold=0.6, time_window_days=7.0, min_cluster_size=2, limit=100, dry_run=false))]
pub fn py_consolidate(
    db: Bound<'_, PyYantrikDB>,
    sim_threshold: f64,
    time_window_days: f64,
    min_cluster_size: usize,
    limit: usize,
    dry_run: bool,
) -> PyResult<Vec<PyObject>> {
    let py = db.py();
    // No borrow of `db` is held across the detached call; see
    // `engine_handle` in py_engine for why that matters.
    let inner = PyYantrikDB::engine_handle(&db)?;
    // Release the GIL for the engine call (#271). One consolidate() pass
    // loads a page of eligible memories, decrypts their embeddings, runs
    // pairwise cosine clustering over the page, builds the extractive
    // summaries, and commits the consolidations; on a loaded store a
    // single call runs for seconds, and without this it holds the GIL
    // for the whole pass, stalling every other thread in the host.
    // `detach` is PyO3 0.28's name for what older docs call
    // `allow_threads`. Nothing in the closure touches Python: it takes
    // an owned `Arc<YantrikDB>` plus plain values, the default path
    // carries each cluster's mean embedding so the engine embedder is
    // never invoked, and the result JSON is converted after
    // reattaching. `Arc<YantrikDB>` is owned and `Send + Sync`, which
    // is what `Ungil` requires.
    let results = py
        .detach(move || {
            yantrikdb_core::consolidate::consolidate(
                &inner,
                sim_threshold,
                time_window_days,
                min_cluster_size,
                limit,
                true, // require_entity_overlap — default behavior since v0.6.0
                dry_run,
            )
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    results.iter().map(|v| json_to_py(py, v)).collect()
}

fn dict_to_mem_with_embedding(d: &Bound<'_, PyDict>) -> PyResult<MemoryWithEmbedding> {
    let rid: String = d
        .get_item("rid")?
        .map(|v| v.extract())
        .unwrap_or(Ok("".to_string()))?;
    let text: String = d
        .get_item("text")?
        .map(|v| v.extract())
        .unwrap_or(Ok("".to_string()))?;
    let memory_type: String = d
        .get_item("type")?
        .map(|v| v.extract())
        .unwrap_or(Ok("episodic".to_string()))?;
    let embedding: Vec<f32> = d
        .get_item("embedding")?
        .map(|v| v.extract())
        .unwrap_or(Ok(vec![]))?;
    let created_at: f64 = d
        .get_item("created_at")?
        .map(|v| v.extract())
        .unwrap_or(Ok(0.0))?;
    let importance: f64 = d
        .get_item("importance")?
        .map(|v| v.extract())
        .unwrap_or(Ok(0.5))?;
    let valence: f64 = d
        .get_item("valence")?
        .map(|v| v.extract())
        .unwrap_or(Ok(0.0))?;
    let half_life: f64 = d
        .get_item("half_life")?
        .map(|v| v.extract())
        .unwrap_or(Ok(604800.0))?;
    let last_access: f64 = d
        .get_item("last_access")?
        .map(|v| v.extract())
        .unwrap_or(Ok(0.0))?;
    let metadata: serde_json::Value = d
        .get_item("metadata")?
        .map(|v| py_to_json(&v))
        .unwrap_or(Ok(serde_json::json!({})))?;

    Ok(MemoryWithEmbedding {
        rid,
        memory_type,
        text,
        embedding,
        created_at,
        importance,
        valence,
        half_life,
        last_access,
        metadata,
        // Read from the dict, defaulting only when absent. The old hardcode
        // was harmless solely because find_clusters ignores namespace — a
        // landmine for any future per-namespace guard (silent cross-tenant
        // merge). Read it now so the guard, when it comes, sees the truth.
        namespace: d
            .get_item("namespace")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_else(|| "default".to_string()),
    })
}

fn mem_with_emb_to_dict(py: Python<'_>, m: &MemoryWithEmbedding) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("rid", &m.rid)?;
    dict.set_item("type", &m.memory_type)?;
    dict.set_item("text", &m.text)?;
    dict.set_item("embedding", &m.embedding)?;
    dict.set_item("created_at", m.created_at)?;
    dict.set_item("importance", m.importance)?;
    dict.set_item("valence", m.valence)?;
    dict.set_item("half_life", m.half_life)?;
    dict.set_item("last_access", m.last_access)?;
    dict.set_item("metadata", json_to_py(py, &m.metadata)?)?;
    Ok(dict.into())
}

/// Consolidate an explicit cluster with CALLER-SUPPLIED text.
///
/// The default `consolidate()` writes an extractive join of the cluster's
/// texts carrying the MEAN of their embeddings. Measured on BEAM
/// (2026-08-11), that costs accuracy through embedding dilution: no content
/// is lost, but N precise vectors collapse into one average that matches no
/// specific query well. This entry point lets the caller supply a real
/// synthesis — e.g. from a small local model — whose vector describes the
/// text that was actually written.
///
/// `embedding`: omit to let the ENGINE embed `text` (requires an engine
/// embedder). Callers whose embedder is Python-side — `YantrikDB(embedder=…)`
/// / `set_embedder()`, where the engine itself has none — must pass the
/// vector they computed for `text`.
///
/// Returns the same dict shape as `consolidate()`'s elements, plus
/// `embedded_from_text` reporting which path produced the vector.
/// Store a synthesis of `source_rids` BESIDE them, leaving every source live.
///
/// The difference from `consolidate_cluster` is the whole point: that one
/// retires its sources (`consolidation_status = 'consolidated'`, which the
/// default recall filter excludes, plus a 0.3x importance cut), this one does
/// not. Use it to add an ABSTRACTION over a cluster without losing the
/// verbatim detail underneath.
///
/// Motivation, measured: 26% of all points lost on BEAM sit on abstract topic
/// labels ('Initial project setup', 'Integration test coverage') that appear
/// NOWHERE in 12.4M characters of the stored conversations. Retrieval cannot
/// surface a phrase nobody wrote — it has to be synthesised and stored. The
/// replacing form was tried and cost -21.6pp on summarization and -17.5pp on
/// preference_following by hiding the detail those categories need.
///
/// `idempotency_key` (#291): without one, two identical calls store two
/// summaries. With one, a retry returns the first summary with
/// `idempotent_replay: True`, and reusing the key with different text or a
/// different source set raises instead of writing a second record. Sources
/// may be named in any order.
#[pyfunction]
#[pyo3(
    name = "summarize_cluster",
    signature = (db, source_rids, text, embedding=None, idempotency_key=None)
)]
pub fn py_summarize_cluster(
    py: Python<'_>,
    db: Bound<'_, PyYantrikDB>,
    source_rids: Vec<String>,
    text: &str,
    embedding: Option<Vec<f32>>,
    idempotency_key: Option<&str>,
) -> PyResult<PyObject> {
    // Reaches SQLite: with the GIL released and no borrow of `db` held (#271).
    let (inner, embedder) = PyYantrikDB::engine_and_embedder(&db)?;
    let from_text = embedder.as_ref().map(py_embed);
    let vector = summary_vector(embedding.as_deref(), from_text.as_ref());
    let out = py
        .detach(|| {
            crate::py_engine::in_engine_call(|| {
                yantrikdb_core::consolidate::summarize_cluster_with(
                    &inner,
                    &source_rids,
                    text,
                    vector,
                    idempotency_key,
                )
            })
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    json_to_py(py, &out)
}

/// Undo a replacing consolidation: return the sources `summary_rid` retired
/// to `active`, at the importance they had before it (#291).
///
/// The summary stays, so the result is what `summarize_cluster` would have
/// written: the synthesis beside its sources. `forget()` it as well if it
/// should go too. Sources consolidated before schema v55 have no stored
/// value; the call then raises and writes nothing, unless
/// `allow_inferred=True`, which restores them at current importance / 0.3.
/// Forgotten sources are never brought back.
///
/// Returns `{"summary_rid", "restored": [{"rid", "importance", "basis"}],
/// "skipped": [{"rid", "reason"}]}`. A second call restores nothing.
#[pyfunction]
#[pyo3(
    name = "restore_from_consolidation",
    signature = (db, summary_rid, allow_inferred=false)
)]
pub fn py_restore_from_consolidation(
    py: Python<'_>,
    db: Bound<'_, PyYantrikDB>,
    summary_rid: &str,
    allow_inferred: bool,
) -> PyResult<PyObject> {
    // Reaches SQLite: with the GIL released and no borrow of `db` held (#271).
    let inner = PyYantrikDB::engine_handle(&db)?;
    let out = py
        .detach(|| {
            yantrikdb_core::consolidate::restore_from_consolidation(
                &inner,
                summary_rid,
                allow_inferred,
            )
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    json_to_py(py, &out)
}

/// Persist a query-independent synthesized item beside its source evidence.
///
/// The engine owns provenance and temporal metadata: `created_at` is the
/// newest source (availability), while `metadata.first_mention_at` is the
/// earliest source mention. `axis` and `granularity` let callers store a
/// lattice of fine children and rollups instead of one irreversible summary.
/// `idempotency_key` must be stable for the logical item. A retry over the
/// same evidence whose text changes is rejected. Changed evidence creates a
/// new generation and atomically supersedes the older logical generation.
#[pyfunction]
#[pyo3(
    name = "record_synthesis",
    signature = (
        db,
        source_rids,
        text,
        axis,
        idempotency_key,
        granularity="atomic",
        embedding=None,
        metadata=None
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn py_record_synthesis(
    py: Python<'_>,
    db: Bound<'_, PyYantrikDB>,
    source_rids: Vec<String>,
    text: &str,
    axis: &str,
    idempotency_key: &str,
    granularity: &str,
    embedding: Option<Vec<f32>>,
    metadata: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyObject> {
    // Reaches SQLite: with the GIL released and no borrow of `db` held (#271).
    let (inner, embedder) = PyYantrikDB::engine_and_embedder(&db)?;
    let from_text = embedder.as_ref().map(py_embed);
    let vector = summary_vector(embedding.as_deref(), from_text.as_ref());
    let metadata = match metadata {
        Some(value) => py_to_json(&value.as_any())?,
        None => serde_json::json!({}),
    };
    let out = py
        .detach(|| {
            crate::py_engine::in_engine_call(|| {
                yantrikdb_core::consolidate::record_synthesis_with(
                    &inner,
                    &source_rids,
                    text,
                    vector,
                    axis,
                    granularity,
                    &metadata,
                    idempotency_key,
                )
            })
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    json_to_py(py, &out)
}

#[pyfunction]
#[pyo3(name = "consolidate_cluster", signature = (db, source_rids, text, embedding=None))]
pub fn py_consolidate_cluster(
    py: Python<'_>,
    db: Bound<'_, PyYantrikDB>,
    source_rids: Vec<String>,
    text: &str,
    embedding: Option<Vec<f32>>,
) -> PyResult<PyObject> {
    // Reaches SQLite: with the GIL released and no borrow of `db` held (#271).
    let (inner, embedder) = PyYantrikDB::engine_and_embedder(&db)?;
    let from_text = embedder.as_ref().map(py_embed);
    let vector = summary_vector(embedding.as_deref(), from_text.as_ref());
    let out = py
        .detach(|| {
            crate::py_engine::in_engine_call(|| {
                yantrikdb_core::consolidate::consolidate_cluster_with(
                    &inner,
                    &source_rids,
                    text,
                    vector,
                )
            })
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    json_to_py(py, &out)
}

/// Where a cluster write's vector comes from, in the precedence every other
/// write follows: a vector the caller passed (part of a keyed write's
/// payload), else the caller-attached Python embedder, else the engine's
/// own. The Python embedder is called on the text inside the detached call,
/// re-taking the GIL just for `encode()`, and only if the write goes ahead:
/// a keyed retry returns at the engine's pre-admission probe without
/// embedding. Such a write is digested on its text and fields, like an
/// engine-embedded one, so a retry is not refused because an embedder did
/// not reproduce its vector bit for bit (#271 review). Without a Python
/// embedder path, a store whose only embedder was a Python one raised "No
/// embedder configured" whenever no vector was passed.
fn summary_vector<'a, F>(
    explicit: Option<&'a [f32]>,
    from_text: Option<&'a F>,
) -> yantrikdb_core::consolidate::SummaryVector<'a>
where
    F: Fn(&str) -> yantrikdb_core::error::Result<Vec<f32>> + Sync,
{
    use yantrikdb_core::consolidate::SummaryVector;
    match (explicit, from_text) {
        (Some(v), _) => SummaryVector::Explicit(v),
        (None, Some(f)) => SummaryVector::FromText(f),
        (None, None) => SummaryVector::Engine,
    }
}
