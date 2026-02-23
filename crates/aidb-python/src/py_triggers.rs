use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::py_engine::PyAIDB;
use crate::py_types::json_to_py;

/// Python-visible Trigger class.
#[pyclass(name = "Trigger")]
#[derive(Clone)]
pub struct PyTrigger {
    #[pyo3(get)]
    pub trigger_type: String,
    #[pyo3(get)]
    pub reason: String,
    #[pyo3(get)]
    pub urgency: f64,
    #[pyo3(get)]
    pub source_rids: Vec<String>,
    #[pyo3(get)]
    pub suggested_action: String,
    context_data: HashMap<String, serde_json::Value>,
}

#[pymethods]
impl PyTrigger {
    #[getter]
    fn context(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (k, v) in &self.context_data {
            dict.set_item(k, json_to_py(py, v)?)?;
        }
        Ok(dict.into())
    }
}

impl From<aidb_core::Trigger> for PyTrigger {
    fn from(t: aidb_core::Trigger) -> Self {
        PyTrigger {
            trigger_type: t.trigger_type,
            reason: t.reason,
            urgency: t.urgency,
            source_rids: t.source_rids,
            suggested_action: t.suggested_action,
            context_data: t.context,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (db, importance_threshold=0.5, decay_threshold=0.1, max_triggers=5))]
pub fn check_decay_triggers(
    db: &PyAIDB,
    importance_threshold: f64,
    decay_threshold: f64,
    max_triggers: usize,
) -> PyResult<Vec<PyTrigger>> {
    let inner = db.get_inner()?;
    let triggers = aidb_core::check_decay_triggers(
        inner,
        importance_threshold,
        decay_threshold,
        max_triggers,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(triggers.into_iter().map(PyTrigger::from).collect())
}

#[pyfunction]
#[pyo3(signature = (db, min_active_memories=10))]
pub fn check_consolidation_triggers(
    db: &PyAIDB,
    min_active_memories: i64,
) -> PyResult<Vec<PyTrigger>> {
    let inner = db.get_inner()?;
    let triggers = aidb_core::check_consolidation_triggers(inner, min_active_memories)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(triggers.into_iter().map(PyTrigger::from).collect())
}

#[pyfunction]
#[pyo3(signature = (db, importance_threshold=0.5, decay_threshold=0.1, max_triggers=5))]
pub fn check_all_triggers(
    db: &PyAIDB,
    importance_threshold: f64,
    decay_threshold: f64,
    max_triggers: usize,
) -> PyResult<Vec<PyTrigger>> {
    let inner = db.get_inner()?;
    let triggers = aidb_core::check_all_triggers(
        inner,
        importance_threshold,
        decay_threshold,
        max_triggers,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(triggers.into_iter().map(PyTrigger::from).collect())
}
