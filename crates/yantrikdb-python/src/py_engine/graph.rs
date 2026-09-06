use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
type PyObject = pyo3::Py<pyo3::PyAny>;

use crate::py_types::*;

use super::{map_err, PyYantrikDB};

#[pymethods]
impl PyYantrikDB {
    /// Ground and store the claims the writer states about a memory it
    /// recorded. Each claim is a dict with `src`/`subject`, `rel_type`/
    /// `relation`, `dst`/`object` and optional `polarity` (1 or -1).
    /// Returns `{"memory_rid", "accepted": [...], "rejected": [{..., "reason"}]}`.
    /// Subject and object must occur in the memory's text; nothing is
    /// inferred. Raises on an unknown/inactive memory or a batch over the
    /// cap; per-claim problems are reported, never raised.
    fn attach_claims(
        &self,
        py: Python<'_>,
        memory_rid: &str,
        claims: Vec<Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let mut stated = Vec::with_capacity(claims.len());
        for d in &claims {
            let pick = |keys: &[&str]| -> PyResult<Option<String>> {
                for k in keys {
                    if let Some(v) = d.get_item(*k)? {
                        return Ok(Some(v.extract::<String>()?));
                    }
                }
                Ok(None)
            };
            let src = pick(&["src", "subject"])?
                .ok_or_else(|| PyValueError::new_err("claim needs 'src' (or 'subject')"))?;
            let rel_type = pick(&["rel_type", "relation", "rel"])?
                .ok_or_else(|| PyValueError::new_err("claim needs 'rel_type' (or 'relation')"))?;
            let dst = pick(&["dst", "object"])?
                .ok_or_else(|| PyValueError::new_err("claim needs 'dst' (or 'object')"))?;
            let polarity: i32 = d
                .get_item("polarity")?
                .map(|v| v.extract::<i32>())
                .transpose()?
                .unwrap_or(1);
            let window = |k: &str| -> PyResult<Option<f64>> {
                d.get_item(k)?
                    .filter(|v| !v.is_none())
                    .map(|v| v.extract::<f64>())
                    .transpose()
            };
            stated.push(yantrikdb_core::StatedClaim {
                src,
                rel_type,
                dst,
                polarity,
                valid_from: window("valid_from")?,
                valid_to: window("valid_to")?,
            });
        }
        let report = db.attach_claims(memory_rid, &stated).map_err(map_err)?;
        let val = serde_json::to_value(&report)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        json_to_py(py, &val)
    }

    /// The relation templates this store has taught itself from stated
    /// claims in `namespace` (active ones are applied to plain writes).
    #[pyo3(signature = (namespace="default"))]
    fn learned_relation_patterns(&self, py: Python<'_>, namespace: &str) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let rows = db.learned_relation_patterns(namespace).map_err(map_err)?;
        let val = serde_json::to_value(&rows)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        json_to_py(py, &val)
    }

    /// Forget every learned template in `namespace`; returns how many were
    /// removed. Claims already minted by them stay.
    #[pyo3(signature = (namespace="default"))]
    fn forget_learned_relation_patterns(&self, namespace: &str) -> PyResult<usize> {
        let db = self.get_inner()?;
        db.forget_learned_relation_patterns(namespace)
            .map_err(map_err)
    }

    #[pyo3(signature = (src, dst, rel_type="related_to", weight=1.0))]
    fn relate(&self, src: &str, dst: &str, rel_type: &str, weight: f64) -> PyResult<String> {
        let db = self.get_inner()?;
        db.relate(src, dst, rel_type, weight).map_err(map_err)
    }

    fn get_edges(&self, py: Python<'_>, entity: &str) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let edges = db.get_edges(entity).map_err(map_err)?;
        edges.iter().map(|e| edge_to_dict(py, e)).collect()
    }

    #[pyo3(signature = (pattern=None, entity_type=None, limit=20))]
    fn search_entities(
        &self,
        py: Python<'_>,
        pattern: Option<&str>,
        entity_type: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let entities = db
            .search_entities(pattern, entity_type, limit)
            .map_err(map_err)?;
        entities.iter().map(|e| entity_to_dict(py, e)).collect()
    }

    fn link_memory_entity(&self, memory_rid: &str, entity_name: &str) -> PyResult<()> {
        let db = self.get_inner()?;
        db.link_memory_entity(memory_rid, entity_name)
            .map_err(map_err)
    }

    fn backfill_memory_entities(&self) -> PyResult<usize> {
        let db = self.get_inner()?;
        db.backfill_memory_entities().map_err(map_err)
    }
}
