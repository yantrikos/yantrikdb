use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
type PyObject = pyo3::Py<pyo3::PyAny>;

use crate::py_types::*;

use super::{map_err, PyYantrikDB};

#[pymethods]
impl PyYantrikDB {
    // ── Cognition loop (V3) ──

    #[pyo3(signature = (config=None))]
    fn think(&self, py: Python<'_>, config: Option<&Bound<'_, PyDict>>) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let cfg = if let Some(d) = config {
            let mut c = yantrikdb_core::ThinkConfig::default();
            if let Ok(Some(v)) = d.get_item("importance_threshold") {
                c.importance_threshold = v.extract()?;
            }
            if let Ok(Some(v)) = d.get_item("decay_threshold") {
                c.decay_threshold = v.extract()?;
            }
            if let Ok(Some(v)) = d.get_item("max_triggers") {
                c.max_triggers = v.extract()?;
            }
            if let Ok(Some(v)) = d.get_item("run_consolidation") {
                c.run_consolidation = v.extract()?;
            }
            if let Ok(Some(v)) = d.get_item("run_conflict_scan") {
                c.run_conflict_scan = v.extract()?;
            }
            if let Ok(Some(v)) = d.get_item("run_pattern_mining") {
                c.run_pattern_mining = v.extract()?;
            }
            if let Ok(Some(v)) = d.get_item("min_active_memories") {
                c.min_active_memories = v.extract()?;
            }
            if let Ok(Some(v)) = d.get_item("run_personality") {
                c.run_personality = v.extract()?;
            }
            if let Ok(Some(v)) = d.get_item("consolidation_limit") {
                c.consolidation_limit = v.extract()?;
            }
            if let Ok(Some(v)) = d.get_item("consolidation_time_window_days") {
                c.consolidation_time_window_days = v.extract()?;
            }
            if let Ok(Some(v)) = d.get_item("consolidation_sim_threshold") {
                c.consolidation_sim_threshold = v.extract()?;
            }
            if let Ok(Some(v)) = d.get_item("extract_attribute_claims") {
                c.extract_attribute_claims = v.extract()?;
            }
            if let Ok(Some(v)) = d.get_item("consolidation_min_cluster") {
                c.consolidation_min_cluster = v.extract()?;
            }
            if let Ok(Some(v)) = d.get_item("consolidation_require_entity_overlap") {
                c.consolidation_require_entity_overlap = v.extract()?;
            }
            // Reject unrecognized keys. This dict was the ONE binding surface
            // where a misspelled or unsupported knob was silently absorbed —
            // the dry-run incident shape: two documented ThinkConfig fields
            // (the pair added above) were unreachable from Python for months
            // and nothing ever said so. A typo must be an error, not a
            // silently-default run.
            const KNOWN: [&str; 14] = [
                "importance_threshold",
                "decay_threshold",
                "max_triggers",
                "run_consolidation",
                "run_conflict_scan",
                "run_pattern_mining",
                "min_active_memories",
                "run_personality",
                "consolidation_limit",
                "consolidation_time_window_days",
                "consolidation_sim_threshold",
                "extract_attribute_claims",
                "consolidation_min_cluster",
                "consolidation_require_entity_overlap",
            ];
            for key in d.keys() {
                let k: String = key.extract()?;
                if !KNOWN.contains(&k.as_str()) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "think(): unknown config key {k:?}; known keys: {KNOWN:?}"
                    )));
                }
            }
            c
        } else {
            yantrikdb_core::ThinkConfig::default()
        };
        let result = db.think(&cfg).map_err(map_err)?;
        think_result_to_dict(py, &result)
    }

    fn deliver_trigger(&self, trigger_id: &str) -> PyResult<bool> {
        let db = self.get_inner()?;
        db.deliver_trigger(trigger_id).map_err(map_err)
    }

    fn acknowledge_trigger(&self, trigger_id: &str) -> PyResult<bool> {
        let db = self.get_inner()?;
        db.acknowledge_trigger(trigger_id).map_err(map_err)
    }

    fn act_on_trigger(&self, trigger_id: &str) -> PyResult<bool> {
        let db = self.get_inner()?;
        db.act_on_trigger(trigger_id).map_err(map_err)
    }

    fn dismiss_trigger(&self, trigger_id: &str) -> PyResult<bool> {
        let db = self.get_inner()?;
        db.dismiss_trigger(trigger_id).map_err(map_err)
    }

    #[pyo3(signature = (limit=10))]
    fn get_pending_triggers(&self, py: Python<'_>, limit: usize) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let triggers = db.get_pending_triggers(limit).map_err(map_err)?;
        triggers
            .iter()
            .map(|t| persisted_trigger_to_dict(py, t))
            .collect()
    }

    #[pyo3(signature = (trigger_type=None, limit=50))]
    fn get_trigger_history(
        &self,
        py: Python<'_>,
        trigger_type: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let triggers = db
            .get_trigger_history(trigger_type, limit)
            .map_err(map_err)?;
        triggers
            .iter()
            .map(|t| persisted_trigger_to_dict(py, t))
            .collect()
    }

    #[pyo3(signature = (pattern_type=None, status=None, limit=50))]
    fn get_patterns(
        &self,
        py: Python<'_>,
        pattern_type: Option<&str>,
        status: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let patterns = db
            .get_patterns(pattern_type, status, limit)
            .map_err(map_err)?;
        patterns.iter().map(|p| pattern_to_dict(py, p)).collect()
    }

    // ── Personality API (V11) ──

    fn get_personality(&self, py: Python<'_>) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let profile = db.get_personality().map_err(map_err)?;
        personality_profile_to_dict(py, &profile)
    }

    fn derive_personality(&self, py: Python<'_>) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let profile = db.derive_personality().map_err(map_err)?;
        personality_profile_to_dict(py, &profile)
    }

    #[pyo3(signature = (name, score))]
    fn set_personality_trait(&self, name: &str, score: f64) -> PyResult<bool> {
        let db = self.get_inner()?;
        db.set_personality_trait(name, score).map_err(map_err)
    }

    // ── Conflict resolution API (V2) ──

    #[pyo3(signature = (status=None, conflict_type=None, entity=None, priority=None, namespace=None, limit=50))]
    fn get_conflicts(
        &self,
        py: Python<'_>,
        status: Option<&str>,
        conflict_type: Option<&str>,
        entity: Option<&str>,
        priority: Option<&str>,
        namespace: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let conflicts = db
            .get_conflicts(status, conflict_type, entity, priority, namespace, limit)
            .map_err(map_err)?;
        conflicts.iter().map(|c| conflict_to_dict(py, c)).collect()
    }

    fn get_conflict(&self, py: Python<'_>, conflict_id: &str) -> PyResult<Option<PyObject>> {
        let db = self.get_inner()?;
        match db.get_conflict(conflict_id).map_err(map_err)? {
            Some(c) => Ok(Some(conflict_to_dict(py, &c)?)),
            None => Ok(None),
        }
    }

    #[pyo3(signature = (conflict_id, strategy, winner_rid=None, new_text=None, resolution_note=None))]
    fn resolve_conflict(
        &self,
        py: Python<'_>,
        conflict_id: &str,
        strategy: &str,
        winner_rid: Option<&str>,
        new_text: Option<&str>,
        resolution_note: Option<&str>,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let result = db
            .resolve_conflict(conflict_id, strategy, winner_rid, new_text, resolution_note)
            .map_err(map_err)?;
        let dict = PyDict::new(py);
        dict.set_item("conflict_id", &result.conflict_id)?;
        dict.set_item("strategy", &result.strategy)?;
        dict.set_item("winner_rid", &result.winner_rid)?;
        dict.set_item("loser_tombstoned", result.loser_tombstoned)?;
        dict.set_item("new_memory_rid", &result.new_memory_rid)?;
        Ok(dict.into())
    }

    #[pyo3(signature = (conflict_id, note=None))]
    fn dismiss_conflict(&self, conflict_id: &str, note: Option<&str>) -> PyResult<()> {
        let db = self.get_inner()?;
        db.dismiss_conflict(conflict_id, note).map_err(map_err)
    }

    fn scan_conflicts(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let conflicts = yantrikdb_core::scan_conflicts(db).map_err(map_err)?;
        conflicts.iter().map(|c| conflict_to_dict(py, c)).collect()
    }
}

/// Typed facets (contract: docs/standing_instruction_facet_design.md).
/// Names here are the contract's illustrative ones made concrete; behavior
/// is normative and tested engine-side.
#[pymethods]
impl PyYantrikDB {
    /// Extract standing-instruction facets from user-authored records.
    ///
    /// `dry_run=True` audits without writing anything — the contract's
    /// false-fire audit surface. Returns the audit counters as a dict.
    #[pyo3(signature = (namespace="default", dry_run=false))]
    fn extract_standing_instructions(
        &self,
        py: Python<'_>,
        namespace: &str,
        dry_run: bool,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let audit = db
            .extract_standing_instructions(namespace, dry_run)
            .map_err(map_err)?;
        let val =
            serde_json::to_value(&audit).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        crate::py_types::json_to_py(py, &val)
    }

    /// The standing-instruction salience lane: verified facets for a
    /// namespace in first-mention order, complete when within `limit`,
    /// deterministic prefix with an explicit `omitted` count otherwise.
    /// Ordinary `recall` is untouched; callers compose the two.
    #[pyo3(signature = (namespace="default", limit=8))]
    fn recall_facets(&self, py: Python<'_>, namespace: &str, limit: usize) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let out = db.recall_facets(namespace, limit).map_err(map_err)?;
        let val = serde_json::to_value(&out).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        crate::py_types::json_to_py(py, &val)
    }

    /// Coverage-first thread retrieval (opt-in): ALL visible rows in the
    /// namespace matched by ANY requested anchor, in chronological order
    /// with 1-based positions — no similarity ranking, deterministic SQL
    /// only. Truncation keeps the earliest `limit` rows and reports it via
    /// `total`/`omitted`. Ordinary `recall` is untouched.
    ///
    /// Thread v2: optional `phrases` (literal FTS phrases; typed
    /// `PhraseRouteUnavailableError` on an encrypted store) and
    /// `topic_rids` (already-resolved topic synthesis rids) anchors union
    /// with the entity route. Result-shape compatibility rule (documented
    /// choice): an ENTITY-ONLY call — `phrases`/`topic_rids` omitted or
    /// empty — returns the EXACT legacy v1 dict shape (`items` without
    /// route fields, `total`, `omitted`), so pre-v2 callers are unbroken;
    /// any call passing phrases or topic_rids returns the richer v2 shape
    /// (items carry `routes`, matched `phrases`, matched `topic_rids`;
    /// the result adds an explicit `returned`). Only those V2-ROUTED calls
    /// (phrases/topic_rids passed here, or any `recall_thread_v2` call)
    /// can raise `SourceTurnMaintenanceRequiredError` on a store whose
    /// `source_turn` columns are stale — run
    /// `maintain_source_turn_backfill` until complete. An entity-only call
    /// takes the legacy v1 path and NEVER raises it (v1 orders from
    /// decrypt-derived turns, independent of the persisted columns).
    #[pyo3(signature = (namespace, entities, limit=100, phrases=None, topic_rids=None))]
    fn recall_thread(
        &self,
        py: Python<'_>,
        namespace: &str,
        entities: Vec<String>,
        limit: usize,
        phrases: Option<Vec<String>>,
        topic_rids: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let phrases = phrases.unwrap_or_default();
        let topic_rids = topic_rids.unwrap_or_default();
        let val = if phrases.is_empty() && topic_rids.is_empty() {
            // Legacy shape for entity-only calls (v1 compatibility).
            let entity_refs: Vec<&str> = entities.iter().map(String::as_str).collect();
            let out = db
                .recall_thread(namespace, &entity_refs, limit)
                .map_err(map_err)?;
            serde_json::to_value(&out).map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        } else {
            let query = yantrikdb_core::ThreadQuery {
                entities,
                phrases,
                topic_rids,
            };
            let out = db
                .recall_thread_v2(namespace, &query, limit)
                .map_err(map_err)?;
            serde_json::to_value(&out).map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        };
        crate::py_types::json_to_py(py, &val)
    }

    /// The EXPLICIT v2 entry point (final reviewer batch 11): ALWAYS takes
    /// the v2 engine path — entity-only, phrase-only, topic-only, any mix,
    /// or all-empty (all-empty returns an empty result with `returned=0`;
    /// an empty query is valid, not a fault). The method name IS the
    /// version selection: no routing rules, no strict/version kwargs.
    /// Strict source_turn gating applies unconditionally — a stale marker
    /// raises `SourceTurnMaintenanceRequiredError` even for entity-only
    /// queries — and the result is always the richer v2 shape (items carry
    /// `routes` / matched `phrases` / matched `topic_rids`; the result
    /// carries explicit `total` / `returned` / `omitted`). `recall_thread`
    /// above stays byte-for-byte legacy v1 for entity-only calls.
    #[pyo3(signature = (namespace, entities=None, limit=100, phrases=None, topic_rids=None))]
    fn recall_thread_v2(
        &self,
        py: Python<'_>,
        namespace: &str,
        entities: Option<Vec<String>>,
        limit: usize,
        phrases: Option<Vec<String>>,
        topic_rids: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let query = yantrikdb_core::ThreadQuery {
            entities: entities.unwrap_or_default(),
            phrases: phrases.unwrap_or_default(),
            topic_rids: topic_rids.unwrap_or_default(),
        };
        let out = db
            .recall_thread_v2(namespace, &query, limit)
            .map_err(map_err)?;
        let val = serde_json::to_value(&out).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        crate::py_types::json_to_py(py, &val)
    }

    /// One batch of the v50 source_turn recompute/repair pass — the
    /// maintenance operation `SourceTurnMaintenanceRequiredError` names.
    /// Returns `{"processed", "remaining", "complete"}`; call repeatedly
    /// until `complete` is true.
    #[pyo3(signature = (batch=10_000))]
    fn maintain_source_turn_backfill(&self, py: Python<'_>, batch: usize) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let out = db.maintain_source_turn_backfill(batch).map_err(map_err)?;
        let val = serde_json::to_value(&out).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        crate::py_types::json_to_py(py, &val)
    }
}
