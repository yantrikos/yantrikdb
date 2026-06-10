use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
type PyObject = pyo3::Py<pyo3::PyAny>;

use crate::py_types::*;

use super::{map_err, PyYantrikDB};

#[pymethods]
impl PyYantrikDB {
    #[pyo3(signature = (text, memory_type="episodic", importance=0.5, valence=0.0, half_life=604800.0, metadata=None, embedding=None, namespace="default", certainty=0.8, domain="general", source="user", emotional_state=None))]
    fn record(
        &self,
        py: Python<'_>,
        text: &str,
        memory_type: &str,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: Option<&Bound<'_, PyDict>>,
        embedding: Option<Vec<f32>>,
        namespace: &str,
        certainty: f64,
        domain: &str,
        source: &str,
        emotional_state: Option<&str>,
    ) -> PyResult<String> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;

        let emb = match embedding {
            Some(e) => e,
            None => self.embed_text(py, text)?,
        };

        let meta = match metadata {
            Some(d) => py_to_json(&d.as_any())?,
            None => serde_json::json!({}),
        };

        db.record(
            text,
            memory_type,
            importance,
            valence,
            half_life,
            &meta,
            &emb,
            namespace,
            certainty,
            domain,
            source,
            emotional_state,
        )
        .map_err(map_err)
    }

    #[pyo3(signature = (query=None, query_embedding=None, top_k=10, time_window=None, memory_type=None, include_consolidated=false, expand_entities=true, skip_reinforce=false, namespace=None, domain=None, source=None, certainty_min=None, order=None))]
    #[allow(clippy::too_many_arguments)]
    fn recall(
        &self,
        py: Python<'_>,
        query: Option<&str>,
        query_embedding: Option<Vec<f32>>,
        top_k: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
        expand_entities: bool,
        skip_reinforce: bool,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
        // Issue #46: confidence first-class on recall. `certainty_min`
        // filters candidates whose `certainty < min`. `order` re-sorts
        // the final top_k: "relevance" (default) | "certainty" | "recency".
        certainty_min: Option<f64>,
        order: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;

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

        let results = db
            .recall(
                &emb,
                top_k,
                time_window,
                memory_type,
                include_consolidated,
                expand_entities,
                query,
                skip_reinforce,
                namespace,
                domain,
                source,
                certainty_min,
                order,
            )
            .map_err(map_err)?;

        results
            .iter()
            .map(|r| recall_result_to_dict(py, r))
            .collect()
    }

    /// **v0.7.4** — record a memory, auto-embedding via the engine's
    /// configured embedder.
    ///
    /// Mirrors `YantrikDB::record_text` on the Rust side. The text is
    /// embedded using whatever Rust-native embedder is attached (bundled
    /// or downloaded via `set_embedder_named`); raises `RuntimeError` if
    /// no embedder is configured. For Python-callable embedders (passed
    /// at construction), prefer `record(text=..., embedding=None)` which
    /// also dispatches via `embed_text()`.
    ///
    /// Identical semantics to `record(text=..., embedding=None)` when an
    /// embedder is attached; provided as an explicit surface that mirrors
    /// the engine's `record_text` for users coming from the Rust API.
    #[pyo3(signature = (text, memory_type="episodic", importance=0.5, valence=0.0, half_life=604800.0, metadata=None, namespace="default", certainty=0.8, domain="general", source="user", emotional_state=None))]
    fn record_text(
        &self,
        py: Python<'_>,
        text: &str,
        memory_type: &str,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: Option<&Bound<'_, PyDict>>,
        namespace: &str,
        certainty: f64,
        domain: &str,
        source: &str,
        emotional_state: Option<&str>,
    ) -> PyResult<String> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;

        let emb = self.embed_text(py, text)?;

        let meta = match metadata {
            Some(d) => py_to_json(&d.as_any())?,
            None => serde_json::json!({}),
        };

        db.record(
            text,
            memory_type,
            importance,
            valence,
            half_life,
            &meta,
            &emb,
            namespace,
            certainty,
            domain,
            source,
            emotional_state,
        )
        .map_err(map_err)
    }

    /// **v0.7.4 / v0.7.7-extended** — recall memories by text query,
    /// auto-embedding via the engine's configured embedder.
    ///
    /// Mirrors `YantrikDB::recall_text` / `YantrikDB::recall_text_filtered`
    /// on the Rust side: same defaults for unfiltered calls (no time
    /// window, no memory_type filter, expand_entities=true,
    /// skip_reinforce=false). v0.7.7 added keyword-only `namespace` /
    /// `domain` / `source` filters so consumers can isolate retrieval
    /// to a specific subspace without reaching for the full `recall()`
    /// surface.
    ///
    /// **Skill-search use case (v0.3.0+ plugin pattern).** YantrikDB
    /// stores skills by convention as records in
    /// `namespace="skill_substrate"` with
    /// `metadata.record_type="skill"`. To search only skills:
    ///
    /// ```python
    /// hits = db.recall_text(
    ///     "how to handle JSON parsing edge cases",
    ///     top_k=5,
    ///     namespace="skill_substrate",
    /// )
    /// ```
    ///
    /// For richer filtering (memory_type, time window,
    /// include_consolidated, etc.) continue to use `recall(query=...)`
    /// which exposes the full engine surface. This method's namespace +
    /// domain + source kwargs cover the subspace-isolation case
    /// specifically.
    ///
    /// **Surface design.** All three filter args are keyword-only
    /// (after the `*`). Adding them as positionals would have shifted
    /// `top_k`'s meaning for callers passing it positionally — this
    /// keeps `recall_text(query, 5)` working unchanged. Coordinated
    /// with yantrikdb-hermes-agent v0.3.0 (swarm msg 8994b0a1) on the
    /// "pyo3 surfaces should be Pythonic, not transcriptions of the
    /// Rust signature" principle.
    #[pyo3(signature = (query, top_k=10, *, namespace=None, domain=None, source=None))]
    fn recall_text(
        &self,
        py: Python<'_>,
        query: &str,
        top_k: usize,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let db = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("YantrikDB is closed"))?;

        let emb = self.embed_text(py, query)?;

        // Dispatch via recall() directly so we can pass all three filter
        // args in one call. Engine's recall_text_filtered only takes
        // domain + source (no namespace) and recall_text takes none —
        // recall() is the one with all three positional. Same defaults
        // as recall_text on the Rust side: time_window=None,
        // memory_type=None, include_consolidated=false,
        // expand_entities=true, skip_reinforce=false.
        let results = db
            .recall(
                &emb,
                top_k,
                None,  // time_window
                None,  // memory_type
                false, // include_consolidated
                true,  // expand_entities
                Some(query),
                false, // skip_reinforce
                namespace,
                domain,
                source,
                None, // certainty_min (#46)
                None, // order (#46) — relevance default
            )
            .map_err(map_err)?;

        results
            .iter()
            .map(|r| recall_result_to_dict(py, r))
            .collect()
    }

    /// Recall with response including confidence scoring and refinement hints.
    #[pyo3(signature = (query=None, query_embedding=None, top_k=10, time_window=None, memory_type=None, include_consolidated=false, expand_entities=true, skip_reinforce=false, namespace=None, domain=None, source=None))]
    fn recall_with_response(
        &self,
        py: Python<'_>,
        query: Option<&str>,
        query_embedding: Option<Vec<f32>>,
        top_k: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
        expand_entities: bool,
        skip_reinforce: bool,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;

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

        let response = db
            .recall_with_response(
                &emb,
                top_k,
                time_window,
                memory_type,
                include_consolidated,
                expand_entities,
                query,
                skip_reinforce,
                namespace,
                domain,
                source,
            )
            .map_err(map_err)?;

        recall_response_to_dict(py, &response)
    }

    /// Refine a previous recall using a follow-up query.
    #[pyo3(signature = (original_query_embedding, refinement_text=None, refinement_embedding=None, original_rids=vec![], top_k=10, namespace=None, domain=None, source=None))]
    fn recall_refine(
        &self,
        py: Python<'_>,
        original_query_embedding: Vec<f32>,
        refinement_text: Option<&str>,
        refinement_embedding: Option<Vec<f32>>,
        original_rids: Vec<String>,
        top_k: usize,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;

        let ref_emb = match refinement_embedding {
            Some(e) => e,
            None => match refinement_text {
                Some(t) => self.embed_text(py, t)?,
                None => {
                    return Err(PyValueError::new_err(
                        "Must provide either refinement_text or refinement_embedding",
                    ))
                }
            },
        };

        let response = db
            .recall_refine(
                &original_query_embedding,
                &ref_emb,
                &original_rids,
                top_k,
                namespace,
                domain,
                source,
            )
            .map_err(map_err)?;

        recall_response_to_dict(py, &response)
    }

    /// Query builder API: composable recall with keyword arguments.
    ///
    /// ```python
    /// results = db.query(
    ///     embedding=emb,
    ///     top_k=10,
    ///     memory_type="episodic",
    ///     namespace="work",
    /// )
    /// ```
    #[pyo3(signature = (
        query=None, embedding=None, top_k=10, memory_type=None, namespace=None,
        time_window=None, expand_entities=false, include_consolidated=false,
        skip_reinforce=false, domain=None, source=None
    ))]
    fn query(
        &self,
        py: Python<'_>,
        query: Option<&str>,
        embedding: Option<Vec<f32>>,
        top_k: usize,
        memory_type: Option<&str>,
        namespace: Option<&str>,
        time_window: Option<(f64, f64)>,
        expand_entities: bool,
        include_consolidated: bool,
        skip_reinforce: bool,
        domain: Option<&str>,
        source: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;

        let emb = match embedding {
            Some(e) => e,
            None => match query {
                Some(q) => self.embed_text(py, q)?,
                None => {
                    return Err(PyValueError::new_err(
                        "Must provide either query or embedding",
                    ))
                }
            },
        };

        let mut q = yantrikdb_core::RecallQuery::new(emb).top_k(top_k);
        if let Some(mt) = memory_type {
            q = q.memory_type(mt);
        }
        if let Some(ns) = namespace {
            q = q.namespace(ns);
        }
        if let Some(tw) = time_window {
            q = q.time_window(tw.0, tw.1);
        }
        if expand_entities {
            q = q.expand_entities(query.unwrap_or(""));
        }
        if include_consolidated {
            q = q.include_consolidated();
        }
        if skip_reinforce {
            q = q.skip_reinforce();
        }
        if let Some(d) = domain {
            q = q.domain(d);
        }
        if let Some(s) = source {
            q = q.source(s);
        }

        let results = db.query(q).map_err(map_err)?;
        results
            .iter()
            .map(|r| recall_result_to_dict(py, r))
            .collect()
    }

    fn get(&self, py: Python<'_>, rid: &str) -> PyResult<Option<PyObject>> {
        let db = self.get_inner()?;
        match db.get(rid).map_err(map_err)? {
            Some(mem) => Ok(Some(memory_to_dict(py, &mem)?)),
            None => Ok(None),
        }
    }

    fn forget(&self, rid: &str) -> PyResult<bool> {
        let db = self.get_inner()?;
        db.forget(rid).map_err(map_err)
    }

    #[pyo3(signature = (limit=50, offset=0, domain=None, memory_type=None, namespace=None, sort_by="created_at"))]
    fn list_memories(
        &self,
        py: Python<'_>,
        limit: usize,
        offset: usize,
        domain: Option<&str>,
        memory_type: Option<&str>,
        namespace: Option<&str>,
        sort_by: &str,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let (memories, total) = db
            .list_memories(limit, offset, domain, memory_type, namespace, sort_by)
            .map_err(map_err)?;
        let dict = pyo3::types::PyDict::new(py);
        let items: Vec<PyObject> = memories
            .iter()
            .map(|m| memory_to_dict(py, m))
            .collect::<PyResult<Vec<_>>>()?;
        dict.set_item("memories", items)?;
        dict.set_item("total", total)?;
        dict.set_item("offset", offset)?;
        Ok(dict.into())
    }

    /// **v0.7.24 — structural query path.** Typed enumeration by indexed
    /// metadata fields (`kind`, `drive_id`) with a keyset cursor over the
    /// UUIDv7 `rid` — the relational counterpart to similarity `recall`. All
    /// filters optional + AND-composed. Returns
    /// `{ "records": [...], "next_cursor": <last_rid|None>, "limit": N }`;
    /// pass `next_cursor` back as `since_rid` to page. `order` is
    /// `"asc"` (oldest-first, default) or `"desc"` (newest-first).
    #[pyo3(signature = (namespace=None, kind=None, drive_id=None, memory_type=None, domain=None, since_rid=None, limit=50, order="asc"))]
    #[allow(clippy::too_many_arguments)]
    fn list_records(
        &self,
        py: Python<'_>,
        namespace: Option<&str>,
        kind: Option<&str>,
        drive_id: Option<&str>,
        memory_type: Option<&str>,
        domain: Option<&str>,
        since_rid: Option<&str>,
        limit: usize,
        order: &str,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let (records, next_cursor) = db
            .list_records(
                namespace,
                kind,
                drive_id,
                memory_type,
                domain,
                since_rid,
                limit,
                order,
            )
            .map_err(map_err)?;
        let dict = pyo3::types::PyDict::new(py);
        let items: Vec<PyObject> = records
            .iter()
            .map(|m| memory_to_dict(py, m))
            .collect::<PyResult<Vec<_>>>()?;
        dict.set_item("records", items)?;
        dict.set_item("next_cursor", next_cursor)?;
        dict.set_item("limit", limit)?;
        Ok(dict.into())
    }

    #[pyo3(signature = (threshold=0.01))]
    fn decay(&self, py: Python<'_>, threshold: f64) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let decayed = db.decay(threshold).map_err(map_err)?;
        decayed.iter().map(|d| decayed_to_dict(py, d)).collect()
    }

    /// Issue #47 (v0.7.20): in-place correct() with revision history.
    /// **Breaking change vs v0.7.19:**
    /// - `reason` is now REQUIRED and must be non-empty
    /// - `embedding` / `correction_note` params REMOVED
    /// - `new_text` is now Optional (was required); pass None to keep
    /// - `metadata_merge` added (Optional dict to merge into existing
    ///   metadata; None = keep existing)
    /// - `corrected_rid` in result == `rid` (in-place mutation)
    /// - `original_tombstoned` always False
    /// - new `revision_num` field on result (1-indexed)
    ///
    /// Embedding-level corrections still go through forget+record (HNSW
    /// does not support in-place vector update).
    #[pyo3(signature = (rid, reason, new_text=None, metadata_merge=None, new_importance=None, new_valence=None))]
    #[allow(clippy::too_many_arguments)]
    fn correct(
        &self,
        py: Python<'_>,
        rid: &str,
        reason: &str,
        new_text: Option<&str>,
        metadata_merge: Option<&Bound<'_, PyDict>>,
        new_importance: Option<f64>,
        new_valence: Option<f64>,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let metadata_json = match metadata_merge {
            Some(d) => Some(py_to_json(&d.as_any())?),
            None => None,
        };
        let result = db
            .correct(
                rid,
                new_text,
                metadata_json.as_ref(),
                new_importance,
                new_valence,
                reason,
            )
            .map_err(map_err)?;
        let dict = PyDict::new(py);
        dict.set_item("original_rid", &result.original_rid)?;
        dict.set_item("corrected_rid", &result.corrected_rid)?;
        dict.set_item("original_tombstoned", result.original_tombstoned)?;
        dict.set_item("revision_num", result.revision_num)?;
        Ok(dict.into())
    }

    /// Issue #47: query the revision history for a record.
    /// Returns a list of dicts, oldest-first.
    fn history(&self, py: Python<'_>, rid: &str) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let revisions = db.history(rid).map_err(map_err)?;
        revisions
            .iter()
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
                Ok::<PyObject, pyo3::PyErr>(d.into())
            })
            .collect()
    }

    // ── Issue #48 — record-to-record links ──

    /// Record a memory and atomically attach record-to-record links.
    /// `links` is a list of dicts: `{"target_rid": "...", "link_type": "supersedes"}`.
    /// link_type is one of advances/supersedes/contradicts/supports/
    /// questions/derived_from, or "custom:<name>".
    #[pyo3(signature = (text, links, memory_type="episodic", importance=0.5, valence=0.0, half_life=604800.0, metadata=None, embedding=None, namespace="default", certainty=0.8, domain="general", source="user", emotional_state=None))]
    #[allow(clippy::too_many_arguments)]
    fn record_with_links(
        &self,
        py: Python<'_>,
        text: &str,
        links: Vec<Bound<'_, PyDict>>,
        memory_type: &str,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: Option<&Bound<'_, PyDict>>,
        embedding: Option<Vec<f32>>,
        namespace: &str,
        certainty: f64,
        domain: &str,
        source: &str,
        emotional_state: Option<&str>,
    ) -> PyResult<String> {
        let db = self.get_inner()?;
        let emb = match embedding {
            Some(e) => e,
            None => self.embed_text(py, text)?,
        };
        let meta = match metadata {
            Some(d) => py_to_json(&d.as_any())?,
            None => serde_json::json!({}),
        };
        let parsed = parse_links(&links)?;
        db.record_with_links(
            text,
            memory_type,
            importance,
            valence,
            half_life,
            &meta,
            &emb,
            namespace,
            certainty,
            domain,
            source,
            emotional_state,
            &parsed,
        )
        .map_err(map_err)
    }

    /// Add a single record-to-record link. Returns the link_id.
    fn link(&self, source_rid: &str, target_rid: &str, link_type: &str) -> PyResult<String> {
        let db = self.get_inner()?;
        let rl = yantrikdb_core::RecordLink {
            target_rid: target_rid.to_string(),
            link_type: yantrikdb_core::LinkType::from_str_lenient(link_type),
        };
        db.link(source_rid, &rl).map_err(map_err)
    }

    /// Remove a single link. Returns True if a row was deleted.
    fn unlink(&self, source_rid: &str, target_rid: &str, link_type: &str) -> PyResult<bool> {
        let db = self.get_inner()?;
        db.unlink(
            source_rid,
            target_rid,
            &yantrikdb_core::LinkType::from_str_lenient(link_type),
        )
        .map_err(map_err)
    }

    /// Traverse links from `rid`. `direction` is "outbound" | "inbound" |
    /// "both"; `link_type` filters to one type (None = all). Returns a list
    /// of dicts {rid, link_type, created_at, direction}.
    #[pyo3(signature = (rid, direction="both", link_type=None))]
    fn linked_records(
        &self,
        py: Python<'_>,
        rid: &str,
        direction: &str,
        link_type: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let dir = match direction {
            "outbound" => yantrikdb_core::LinkDirection::Outbound,
            "inbound" => yantrikdb_core::LinkDirection::Inbound,
            "both" => yantrikdb_core::LinkDirection::Both,
            other => {
                return Err(PyValueError::new_err(format!(
                    "linked_records: invalid direction {other:?}; expected \
                     \"outbound\" | \"inbound\" | \"both\""
                )))
            }
        };
        let lt = link_type.map(yantrikdb_core::LinkType::from_str_lenient);
        let out = db.linked_records(rid, dir, lt.as_ref()).map_err(map_err)?;
        out.iter()
            .map(|l| {
                let d = PyDict::new(py);
                d.set_item("rid", &l.rid)?;
                d.set_item("link_type", &l.link_type)?;
                d.set_item("created_at", l.created_at)?;
                d.set_item("direction", &l.direction)?;
                Ok::<PyObject, pyo3::PyErr>(d.into())
            })
            .collect()
    }

    /// Recall with record-link expansion. `expand_links` is the hop budget
    /// (0 = identical to recall()). Mirrors `recall()` args otherwise.
    #[pyo3(signature = (query=None, query_embedding=None, top_k=10, expand_links=1, time_window=None, memory_type=None, include_consolidated=false, expand_entities=true, skip_reinforce=false, namespace=None, domain=None, source=None, certainty_min=None, order=None))]
    #[allow(clippy::too_many_arguments)]
    fn recall_with_links(
        &self,
        py: Python<'_>,
        query: Option<&str>,
        query_embedding: Option<Vec<f32>>,
        top_k: usize,
        expand_links: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
        expand_entities: bool,
        skip_reinforce: bool,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
        certainty_min: Option<f64>,
        order: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
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
        let results = db
            .recall_with_links(
                &emb,
                top_k,
                time_window,
                memory_type,
                include_consolidated,
                expand_entities,
                query,
                skip_reinforce,
                namespace,
                domain,
                source,
                certainty_min,
                order,
                expand_links,
            )
            .map_err(map_err)?;
        results
            .iter()
            .map(|r| recall_result_to_dict(py, r))
            .collect()
    }

    /// One-shot reification of legacy `metadata.supersedes` strings into
    /// Supersedes links. Returns the count created. Idempotent.
    fn reify_supersedes_links(&self) -> PyResult<usize> {
        let db = self.get_inner()?;
        db.reify_supersedes_links().map_err(map_err)
    }

    /// Like `record_with_links` but returns `(rid, [per_link_result])`
    /// instead of failing fast. Each result dict has `result` in
    /// {"inserted","already_exists","failed"}, `target_rid`, `link_type`,
    /// and (for failed) `error`. The record commits even if a link fails.
    #[pyo3(signature = (text, links, memory_type="episodic", importance=0.5, valence=0.0, half_life=604800.0, metadata=None, embedding=None, namespace="default", certainty=0.8, domain="general", source="user", emotional_state=None))]
    #[allow(clippy::too_many_arguments)]
    fn record_with_links_partial(
        &self,
        py: Python<'_>,
        text: &str,
        links: Vec<Bound<'_, PyDict>>,
        memory_type: &str,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: Option<&Bound<'_, PyDict>>,
        embedding: Option<Vec<f32>>,
        namespace: &str,
        certainty: f64,
        domain: &str,
        source: &str,
        emotional_state: Option<&str>,
    ) -> PyResult<(String, Vec<PyObject>)> {
        let db = self.get_inner()?;
        let emb = match embedding {
            Some(e) => e,
            None => self.embed_text(py, text)?,
        };
        let meta = match metadata {
            Some(d) => py_to_json(&d.as_any())?,
            None => serde_json::json!({}),
        };
        let parsed = parse_links(&links)?;
        let (rid, results) = db
            .record_with_links_partial(
                text,
                memory_type,
                importance,
                valence,
                half_life,
                &meta,
                &emb,
                namespace,
                certainty,
                domain,
                source,
                emotional_state,
                &parsed,
            )
            .map_err(map_err)?;
        let dicts: Vec<PyObject> = results
            .iter()
            .map(|r| {
                let d = PyDict::new(py);
                use yantrikdb_core::LinkResult::*;
                match r {
                    Inserted {
                        target_rid,
                        link_type,
                    } => {
                        d.set_item("result", "inserted")?;
                        d.set_item("target_rid", target_rid)?;
                        d.set_item("link_type", link_type)?;
                    }
                    AlreadyExists {
                        target_rid,
                        link_type,
                    } => {
                        d.set_item("result", "already_exists")?;
                        d.set_item("target_rid", target_rid)?;
                        d.set_item("link_type", link_type)?;
                    }
                    Failed {
                        target_rid,
                        link_type,
                        error,
                    } => {
                        d.set_item("result", "failed")?;
                        d.set_item("target_rid", target_rid)?;
                        d.set_item("link_type", link_type)?;
                        d.set_item("error", error)?;
                    }
                }
                Ok::<PyObject, pyo3::PyErr>(d.into())
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok((rid, dicts))
    }

    /// Windowed leak-candidate audit (issue #48 follow-up). Returns a dict
    /// {window_floor, candidate_count, candidate_rids}. Replaces the
    /// compaction-confused "orphan" metric — only flags in-window memories
    /// that genuinely lack oplog + replication provenance.
    #[pyo3(signature = (max_rids=100))]
    fn audit_leak_candidates(&self, py: Python<'_>, max_rids: usize) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let report = db.audit_leak_candidates(max_rids).map_err(map_err)?;
        let d = PyDict::new(py);
        d.set_item("window_floor", report.window_floor)?;
        d.set_item("candidate_count", report.candidate_count)?;
        d.set_item("candidate_rids", report.candidate_rids)?;
        Ok(d.into())
    }

    fn record_batch(
        &self,
        py: Python<'_>,
        inputs: Vec<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<String>> {
        let db = self.get_inner()?;

        let mut record_inputs = Vec::with_capacity(inputs.len());
        for d in &inputs {
            let text: String = d
                .get_item("text")?
                .ok_or_else(|| PyValueError::new_err("Each input must have a 'text' key"))?
                .extract()?;

            let memory_type: String = d
                .get_item("memory_type")?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_else(|| "episodic".to_string());

            let importance: f64 = d
                .get_item("importance")?
                .map(|v| v.extract::<f64>())
                .transpose()?
                .unwrap_or(0.5);

            let valence: f64 = d
                .get_item("valence")?
                .map(|v| v.extract::<f64>())
                .transpose()?
                .unwrap_or(0.0);

            let half_life: f64 = d
                .get_item("half_life")?
                .map(|v| v.extract::<f64>())
                .transpose()?
                .unwrap_or(604800.0);

            let metadata = d
                .get_item("metadata")?
                .map(|v| py_to_json(&v))
                .transpose()?
                .unwrap_or(serde_json::json!({}));

            let embedding: Vec<f32> = match d.get_item("embedding")? {
                Some(v) => v.extract()?,
                None => self.embed_text(py, &text)?,
            };

            let namespace: String = d
                .get_item("namespace")?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_else(|| "default".to_string());

            let certainty: f64 = d
                .get_item("certainty")?
                .map(|v| v.extract::<f64>())
                .transpose()?
                .unwrap_or(0.8);

            let domain: String = d
                .get_item("domain")?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_else(|| "general".to_string());

            let source: String = d
                .get_item("source")?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_else(|| "user".to_string());

            let emotional_state: Option<String> = d
                .get_item("emotional_state")?
                .map(|v| v.extract::<Option<String>>())
                .transpose()?
                .flatten();

            record_inputs.push(yantrikdb_core::RecordInput {
                text,
                memory_type,
                importance,
                valence,
                half_life,
                metadata,
                embedding,
                namespace,
                certainty,
                domain,
                source,
                emotional_state,
            });
        }

        db.record_batch(&record_inputs).map_err(map_err)
    }

    /// Record feedback on a recall result for adaptive learning.
    #[pyo3(signature = (rid, feedback, query_text=None, query_embedding=None, score_at_retrieval=None, rank_at_retrieval=None))]
    fn recall_feedback(
        &self,
        rid: &str,
        feedback: &str,
        query_text: Option<&str>,
        query_embedding: Option<Vec<f32>>,
        score_at_retrieval: Option<f64>,
        rank_at_retrieval: Option<i32>,
    ) -> PyResult<()> {
        let db = self.get_inner()?;
        db.recall_feedback(
            query_text,
            query_embedding.as_deref(),
            rid,
            feedback,
            score_at_retrieval,
            rank_at_retrieval,
        )
        .map_err(map_err)
    }

    /// Get the current learned scoring weights.
    fn learned_weights(&self, py: Python<'_>) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let w = db.load_learned_weights().map_err(map_err)?;
        let dict = PyDict::new(py);
        dict.set_item("w_sim", w.w_sim)?;
        dict.set_item("w_decay", w.w_decay)?;
        dict.set_item("w_recency", w.w_recency)?;
        dict.set_item("gate_tau", w.gate_tau)?;
        dict.set_item("alpha_imp", w.alpha_imp)?;
        dict.set_item("keyword_boost", w.keyword_boost)?;
        dict.set_item("generation", w.generation)?;
        Ok(dict.into())
    }

    /// Embed text using the configured embedder. Returns a list of floats.
    ///
    /// ```python
    /// embedding = db.embed("some text")
    /// ```
    fn embed(&self, py: Python<'_>, text: &str) -> PyResult<Vec<f32>> {
        self.embed_text(py, text)
    }
}

/// Parse a Python list of `{"target_rid": str, "link_type": str}` dicts
/// into `Vec<RecordLink>` for `record_with_links`. (Issue #48.)
fn parse_links(links: &[Bound<'_, PyDict>]) -> PyResult<Vec<yantrikdb_core::RecordLink>> {
    let mut out = Vec::with_capacity(links.len());
    for d in links {
        let target_rid: String = d
            .get_item("target_rid")?
            .ok_or_else(|| PyValueError::new_err("each link must have a 'target_rid' key"))?
            .extract()?;
        let link_type: String = d
            .get_item("link_type")?
            .ok_or_else(|| PyValueError::new_err("each link must have a 'link_type' key"))?
            .extract()?;
        out.push(yantrikdb_core::RecordLink {
            target_rid,
            link_type: yantrikdb_core::LinkType::from_str_lenient(&link_type),
        });
    }
    Ok(out)
}
