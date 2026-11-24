//! Python bindings for sessions, temporal helpers, relationship depth,
//! entity profile, and procedural memory APIs.

use pyo3::prelude::*;
use pyo3::types::PyDict;
type PyObject = pyo3::Py<pyo3::PyAny>;

use crate::py_types::*;

use super::{plan_embed, run_detached, EmbedPlan, PyYantrikDB};

#[pymethods]
impl PyYantrikDB {
    // ── Session API (V13) ──

    #[pyo3(signature = (namespace="default", client_id="default", metadata=None))]
    fn session_start(
        slf: &Bound<'_, Self>,
        namespace: &str,
        client_id: &str,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        let meta = match metadata {
            Some(d) => py_to_json(&d.as_any())?,
            None => serde_json::json!({}),
        };
        Self::detached(slf, |db| db.session_start(namespace, client_id, &meta))
    }

    #[pyo3(signature = (session_id, summary=None))]
    fn session_end(
        slf: &Bound<'_, Self>,
        session_id: &str,
        summary: Option<&str>,
    ) -> PyResult<PyObject> {
        let result = Self::detached(slf, |db| db.session_end(session_id, summary))?;
        session_summary_to_dict(slf.py(), &result)
    }

    #[pyo3(signature = (namespace="default", client_id="default"))]
    fn active_session(
        slf: &Bound<'_, Self>,
        namespace: &str,
        client_id: &str,
    ) -> PyResult<Option<PyObject>> {
        match Self::detached(slf, |db| db.active_session(namespace, client_id))? {
            Some(s) => Ok(Some(session_to_dict(slf.py(), &s)?)),
            None => Ok(None),
        }
    }

    #[pyo3(signature = (namespace="default", client_id="default", limit=10))]
    fn session_history(
        slf: &Bound<'_, Self>,
        namespace: &str,
        client_id: &str,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        let sessions = Self::detached(slf, |db| db.session_history(namespace, client_id, limit))?;
        sessions.iter().map(|s| session_to_dict(py, s)).collect()
    }

    #[pyo3(signature = (max_age_hours=24.0))]
    fn session_abandon_stale(slf: &Bound<'_, Self>, max_age_hours: f64) -> PyResult<usize> {
        Self::detached(slf, |db| db.session_abandon_stale(max_age_hours))
    }

    // ── Temporal helpers (V13) ──

    #[pyo3(signature = (days=30.0, limit=50, namespace=None))]
    fn stale(
        slf: &Bound<'_, Self>,
        days: f64,
        limit: usize,
        namespace: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        let memories = Self::detached(slf, |db| db.stale(days, limit, namespace))?;
        memories.iter().map(|m| memory_to_dict(py, m)).collect()
    }

    #[pyo3(signature = (days=7.0, limit=50, namespace=None))]
    fn upcoming(
        slf: &Bound<'_, Self>,
        days: f64,
        limit: usize,
        namespace: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        let memories = Self::detached(slf, |db| db.upcoming(days, limit, namespace))?;
        memories.iter().map(|m| memory_to_dict(py, m)).collect()
    }

    #[pyo3(signature = (entity, days=90.0, namespace=None))]
    fn entity_profile(
        slf: &Bound<'_, Self>,
        entity: &str,
        days: f64,
        namespace: Option<&str>,
    ) -> PyResult<PyObject> {
        let profile = Self::detached(slf, |db| db.entity_profile(entity, days, namespace))?;
        entity_profile_to_dict(slf.py(), &profile)
    }

    // ── Relationship depth (V14) ──

    #[pyo3(signature = (entity, namespace=None))]
    fn relationship_depth(
        slf: &Bound<'_, Self>,
        entity: &str,
        namespace: Option<&str>,
    ) -> PyResult<PyObject> {
        let depth = Self::detached(slf, |db| db.relationship_depth(entity, namespace))?;
        relationship_depth_to_dict(slf.py(), &depth)
    }

    // ── Procedural memory (V14) ──

    #[pyo3(signature = (query_embedding, query_text=None, domain=None, top_k=5, namespace=None))]
    fn surface_procedural(
        slf: &Bound<'_, Self>,
        query_embedding: Vec<f32>,
        query_text: Option<&str>,
        domain: Option<&str>,
        top_k: usize,
        namespace: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        let results = Self::detached(slf, |db| {
            db.surface_procedural(&query_embedding, query_text, domain, top_k, namespace)
        })?;
        results
            .iter()
            .map(|r| recall_result_to_dict(py, r))
            .collect()
    }

    #[pyo3(signature = (text, embedding=None, domain="general", task_context="", effectiveness=0.5, namespace="default"))]
    fn record_procedural(
        slf: &Bound<'_, Self>,
        text: &str,
        embedding: Option<Vec<f32>>,
        domain: &str,
        task_context: &str,
        effectiveness: f64,
        namespace: &str,
    ) -> PyResult<String> {
        let py = slf.py();
        let (db, plan) = Self::planned(slf, |e| match embedding.clone() {
            Some(v) => Ok(EmbedPlan::Ready(v)),
            None => plan_embed(py, e, text),
        })?;
        run_detached(py, db, |db| {
            let emb = plan.into_vector(db)?;
            db.record_procedural(text, &emb, domain, task_context, effectiveness, namespace)
        })
    }

    #[pyo3(signature = (rid, outcome))]
    fn reinforce_procedural(slf: &Bound<'_, Self>, rid: &str, outcome: f64) -> PyResult<bool> {
        Self::detached(slf, |db| db.reinforce_procedural(rid, outcome))
    }

    #[pyo3(signature = (namespace=None))]
    fn procedural_stats(slf: &Bound<'_, Self>, namespace: Option<&str>) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        let stats = Self::detached(slf, |db| db.procedural_stats(namespace))?;
        stats
            .iter()
            .map(|(domain, count, avg_imp)| {
                let dict = PyDict::new(py);
                dict.set_item("domain", domain)?;
                dict.set_item("count", count)?;
                dict.set_item("avg_effectiveness", avg_imp)?;
                Ok(dict.into())
            })
            .collect()
    }

    // ── Substitution category APIs (V14) ──

    #[pyo3(signature = (conflict_id, new_type))]
    fn reclassify_conflict(
        slf: &Bound<'_, Self>,
        conflict_id: &str,
        new_type: &str,
    ) -> PyResult<PyObject> {
        let result = Self::detached(slf, |db| db.reclassify_conflict(conflict_id, new_type))?;
        reclassify_result_to_dict(slf.py(), &result)
    }

    fn substitution_categories(slf: &Bound<'_, Self>) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        let cats = Self::detached(slf, |db| db.substitution_categories())?;
        cats.iter()
            .map(|c| substitution_category_to_dict(py, c))
            .collect()
    }

    #[pyo3(signature = (category_name))]
    fn substitution_members(slf: &Bound<'_, Self>, category_name: &str) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        let members = Self::detached(slf, |db| db.substitution_members(category_name))?;
        members
            .iter()
            .map(|m| substitution_member_to_dict(py, m))
            .collect()
    }

    #[pyo3(signature = (category_name, members, source="llm_suggested"))]
    fn learn_category_members(
        slf: &Bound<'_, Self>,
        category_name: &str,
        members: Vec<(String, f64)>,
        source: &str,
    ) -> PyResult<usize> {
        Self::detached(slf, |db| {
            db.learn_category_members(category_name, &members, source)
        })
    }

    #[pyo3(signature = (category_name))]
    fn reset_category_to_seed(slf: &Bound<'_, Self>, category_name: &str) -> PyResult<usize> {
        Self::detached(slf, |db| db.reset_category_to_seed(category_name))
    }
}
