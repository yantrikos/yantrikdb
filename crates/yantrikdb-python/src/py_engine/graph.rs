use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
type PyObject = pyo3::Py<pyo3::PyAny>;

use crate::py_types::*;

use super::{run_detached, to_py, PyYantrikDB};

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
        slf: &Bound<'_, Self>,
        memory_rid: &str,
        claims: Vec<Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        let py = slf.py();
        // Closed before anything else, as when the whole call ran under one
        // borrow: a closed store reports that, not a malformed claim.
        let db = Self::engine_handle(slf)?;
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
            let int_field = |d: &Bound<'_, PyDict>, k: &str| -> PyResult<Option<i32>> {
                d.get_item(k)?
                    .filter(|v| !v.is_none())
                    .map(|v| v.extract::<i32>())
                    .transpose()
            };
            stated.push(yantrikdb_core::StatedClaim {
                src,
                rel_type,
                dst,
                polarity,
                // The writer's modality, verbatim. Validating it is the
                // engine's job (attach_claims rejects unknown values); the
                // binding must not invent "asserted" here, which is what made
                // every quotation and hypothesis look like a plain assertion.
                modality: pick(&["modality"])?,
                // Where in the text the statement sits. Stored so a reader can
                // locate it, and part of the statement's qualifiers: a second
                // claim from the same memory with a different span is refused
                // by the engine's identity guard rather than replacing the
                // first. Across memories spans are not comparable, so a
                // restatement from another memory is accepted whatever its
                // span. The span does not let two observations coexist — the
                // stored key has no span.
                span_start: int_field(&d, "span_start")?,
                span_end: int_field(&d, "span_end")?,
                valid_from: window("valid_from")?,
                valid_to: window("valid_to")?,
            });
        }
        let report = run_detached(py, db, |db| db.attach_claims(memory_rid, &stated))?;
        to_py(py, serde_json::to_value(&report))
    }

    /// The relation templates this store has taught itself from stated
    /// claims in `namespace` (active ones are applied to plain writes).
    #[pyo3(signature = (namespace="default"))]
    fn learned_relation_patterns(slf: &Bound<'_, Self>, namespace: &str) -> PyResult<PyObject> {
        let rows = Self::detached(slf, |db| db.learned_relation_patterns(namespace))?;
        to_py(slf.py(), serde_json::to_value(&rows))
    }

    /// Forget every learned template in `namespace`; returns how many were
    /// removed. Claims already minted by them stay.
    #[pyo3(signature = (namespace="default"))]
    fn forget_learned_relation_patterns(slf: &Bound<'_, Self>, namespace: &str) -> PyResult<usize> {
        Self::detached(slf, |db| db.forget_learned_relation_patterns(namespace))
    }

    #[pyo3(signature = (src, dst, rel_type="related_to", weight=1.0))]
    fn relate(
        slf: &Bound<'_, Self>,
        src: &str,
        dst: &str,
        rel_type: &str,
        weight: f64,
    ) -> PyResult<String> {
        Self::detached(slf, |db| db.relate(src, dst, rel_type, weight))
    }

    fn get_edges(slf: &Bound<'_, Self>, entity: &str) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        let edges = Self::detached(slf, |db| db.get_edges(entity))?;
        edges.iter().map(|e| edge_to_dict(py, e)).collect()
    }

    /// Every claim touching an entity with its qualifiers: polarity,
    /// modality, `valid_from`/`valid_to` (the temporal tag), extractor,
    /// confidence band, provenance and a derived status. `get_edges` is
    /// the bare graph view; this is the claims-table view.
    #[pyo3(signature = (entity, namespace=None))]
    fn get_claims(
        slf: &Bound<'_, Self>,
        entity: &str,
        namespace: Option<&str>,
    ) -> PyResult<PyObject> {
        let rows = Self::detached(slf, |db| db.get_claims(entity, namespace))?;
        json_to_py(slf.py(), &serde_json::Value::Array(rows))
    }

    #[pyo3(signature = (pattern=None, entity_type=None, limit=20))]
    fn search_entities(
        slf: &Bound<'_, Self>,
        pattern: Option<&str>,
        entity_type: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        let entities = Self::detached(slf, |db| db.search_entities(pattern, entity_type, limit))?;
        entities.iter().map(|e| entity_to_dict(py, e)).collect()
    }

    fn link_memory_entity(
        slf: &Bound<'_, Self>,
        memory_rid: &str,
        entity_name: &str,
    ) -> PyResult<()> {
        Self::detached(slf, |db| db.link_memory_entity(memory_rid, entity_name))
    }

    fn backfill_memory_entities(slf: &Bound<'_, Self>) -> PyResult<usize> {
        Self::detached(slf, |db| db.backfill_memory_entities())
    }
}
