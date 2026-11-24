use pyo3::prelude::*;
use pyo3::types::PyDict;
type PyObject = pyo3::Py<pyo3::PyAny>;

use crate::py_types::*;

use super::PyYantrikDB;

/// Inspection reads for one memory; see `yantrikdb_core::engine::inspect`.
/// These are the three reads an explorer (atlas, CLI, MCP, TUI) needs and
/// could previously get only from raw SQL, which is forbidden in-process.
#[pymethods]
impl PyYantrikDB {
    /// Active claims whose source memory is `rid`, newest first, in the
    /// same dict shape as entity claims, including `status_suggestion`.
    /// Read-only; leaves no access trace.
    fn claims_for_memory(slf: &Bound<'_, Self>, rid: &str) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        let claims = Self::detached(slf, |db| db.claims_for_memory(rid))?;
        claims.iter().map(|c| json_to_py(py, c)).collect()
    }

    /// Every correction applied to `rid`, oldest first. Each entry carries
    /// the PRIOR state (`prior_text`, `prior_metadata`, `prior_importance`,
    /// `prior_valence`), the `reason` given to `correct()`, `applied_at`
    /// (unix seconds, transaction time) and `origin_actor`. Empty when the
    /// memory was never corrected or the rid is unknown. Decrypted on
    /// encrypted stores.
    fn revision_history(slf: &Bound<'_, Self>, rid: &str) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        let revs = Self::detached(slf, |db| db.revision_history(rid))?;
        revs.iter()
            .map(|r| {
                let d = PyDict::new(py);
                d.set_item("revision_id", &r.revision_id)?;
                d.set_item("rid", &r.rid)?;
                d.set_item("revision_num", r.revision_num)?;
                d.set_item("prior_text", &r.prior_text)?;
                d.set_item("prior_metadata", json_to_py(py, &r.prior_metadata)?)?;
                d.set_item("prior_importance", r.prior_importance)?;
                d.set_item("prior_valence", r.prior_valence)?;
                d.set_item("reason", &r.reason)?;
                d.set_item("applied_at", r.applied_at)?;
                d.set_item("origin_actor", &r.origin_actor)?;
                Ok(d.into())
            })
            .collect()
    }

    /// Stored entity names linked to `rid`, sorted. Empty for an unlinked
    /// or unknown rid.
    fn memory_entities(slf: &Bound<'_, Self>, rid: &str) -> PyResult<Vec<String>> {
        Self::detached(slf, |db| db.memory_entities(rid))
    }
}
