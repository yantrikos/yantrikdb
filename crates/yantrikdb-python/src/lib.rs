use pyo3::prelude::*;

pub mod py_consolidate;
pub mod py_engine;
pub mod py_errors;
pub mod py_tenant;
pub mod py_triggers;
pub mod py_types;

#[pymodule]
fn _yantrikdb_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Engine
    m.add_class::<py_engine::PyYantrikDB>()?;
    m.add_class::<py_tenant::PyTenantManager>()?;

    // Typed exceptions (v0.10) — all subclass RuntimeError, so pre-v0.10
    // `except RuntimeError:` handlers keep working; new code branches on type.
    m.add("Backpressure", m.py().get_type::<py_errors::Backpressure>())?;
    m.add(
        "CorrectionDeferredDuringReembed",
        m.py()
            .get_type::<py_errors::CorrectionDeferredDuringReembed>(),
    )?;
    m.add(
        "BatchDeferredDuringReembed",
        m.py().get_type::<py_errors::BatchDeferredDuringReembed>(),
    )?;
    m.add(
        "IdempotencyConflict",
        m.py().get_type::<py_errors::IdempotencyConflict>(),
    )?;
    m.add(
        "InvalidIdempotencyKey",
        m.py().get_type::<py_errors::InvalidIdempotencyKey>(),
    )?;
    m.add(
        "ProvenanceInconsistent",
        m.py().get_type::<py_errors::ProvenanceInconsistent>(),
    )?;
    m.add(
        "RecallContended",
        m.py().get_type::<py_errors::RecallContended>(),
    )?;
    m.add(
        "PackEmbedderMismatch",
        m.py().get_type::<py_errors::PackEmbedderMismatch>(),
    )?;
    m.add(
        "PackAlreadyMounted",
        m.py().get_type::<py_errors::PackAlreadyMounted>(),
    )?;
    m.add(
        "PackSignatureInvalid",
        m.py().get_type::<py_errors::PackSignatureInvalid>(),
    )?;

    // Triggers
    m.add_class::<py_triggers::PyTrigger>()?;
    m.add_function(wrap_pyfunction!(py_triggers::check_decay_triggers, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_triggers::check_consolidation_triggers,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_triggers::check_all_triggers, m)?)?;

    // Consolidation
    m.add_function(wrap_pyfunction!(py_consolidate::py_consolidate, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_consolidate::find_consolidation_candidates,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_consolidate::py_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(py_consolidate::py_extractive_summary, m)?)?;
    m.add_function(wrap_pyfunction!(py_consolidate::py_find_clusters, m)?)?;

    Ok(())
}
