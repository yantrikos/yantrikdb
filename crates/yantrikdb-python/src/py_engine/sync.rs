use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
type PyObject = pyo3::Py<pyo3::PyAny>;

use crate::py_types::*;

use super::{map_err, PyYantrikDB};

#[pymethods]
impl PyYantrikDB {
    // ── Storage tier operations ──

    fn archive(&self, rid: &str) -> PyResult<bool> {
        let db = self.get_inner()?;
        db.archive(rid).map_err(map_err)
    }

    fn hydrate(&self, rid: &str) -> PyResult<bool> {
        let db = self.get_inner()?;
        db.hydrate(rid).map_err(map_err)
    }

    #[pyo3(signature = (max_active,))]
    fn evict(&self, max_active: usize) -> PyResult<Vec<String>> {
        let db = self.get_inner()?;
        db.evict(max_active).map_err(map_err)
    }

    // ── Replication API (V5 P2P Sync) ──

    #[pyo3(signature = (since_hlc=None, since_op_id=None, exclude_actor=None, limit=1000))]
    fn extract_ops_since(
        &self,
        py: Python<'_>,
        since_hlc: Option<Vec<u8>>,
        since_op_id: Option<&str>,
        exclude_actor: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let db = self.get_inner()?;
        let ops = yantrikdb_core::replication::extract_ops_since(
            &*db.conn(),
            since_hlc.as_deref(),
            since_op_id,
            exclude_actor,
            limit,
        )
        .map_err(map_err)?;

        let mut result = Vec::with_capacity(ops.len());
        for op in &ops {
            let dict = PyDict::new(py);
            dict.set_item("op_id", &op.op_id)?;
            dict.set_item("op_type", &op.op_type)?;
            dict.set_item("timestamp", op.timestamp)?;
            dict.set_item("target_rid", &op.target_rid)?;
            dict.set_item("payload", json_to_py(py, &op.payload)?)?;
            dict.set_item("actor_id", &op.actor_id)?;
            dict.set_item("hlc", &op.hlc)?;
            dict.set_item("embedding_hash", &op.embedding_hash)?;
            dict.set_item("origin_actor", &op.origin_actor)?;
            result.push(dict.into());
        }
        Ok(result)
    }

    fn apply_ops(&self, py: Python<'_>, ops: Vec<Bound<'_, PyDict>>) -> PyResult<PyObject> {
        let db = self.get_inner()?;

        let mut entries = Vec::with_capacity(ops.len());
        for d in &ops {
            let op_id: String = d
                .get_item("op_id")?
                .ok_or_else(|| PyValueError::new_err("Each op must have 'op_id'"))?
                .extract()?;
            let op_type: String = d
                .get_item("op_type")?
                .ok_or_else(|| PyValueError::new_err("Each op must have 'op_type'"))?
                .extract()?;
            let timestamp: f64 = d
                .get_item("timestamp")?
                .ok_or_else(|| PyValueError::new_err("Each op must have 'timestamp'"))?
                .extract()?;
            let target_rid: Option<String> = d
                .get_item("target_rid")?
                .and_then(|v| if v.is_none() { None } else { Some(v) })
                .map(|v| v.extract())
                .transpose()?;
            let payload = d
                .get_item("payload")?
                .map(|v| py_to_json(&v))
                .transpose()?
                .unwrap_or(serde_json::json!({}));
            let actor_id: String = d
                .get_item("actor_id")?
                .ok_or_else(|| PyValueError::new_err("Each op must have 'actor_id'"))?
                .extract()?;
            let hlc: Vec<u8> = d
                .get_item("hlc")?
                .ok_or_else(|| PyValueError::new_err("Each op must have 'hlc'"))?
                .extract()?;
            let embedding_hash: Option<Vec<u8>> = d
                .get_item("embedding_hash")?
                .and_then(|v| if v.is_none() { None } else { Some(v) })
                .map(|v| v.extract())
                .transpose()?;
            let origin_actor: String = d
                .get_item("origin_actor")?
                .ok_or_else(|| PyValueError::new_err("Each op must have 'origin_actor'"))?
                .extract()?;

            entries.push(yantrikdb_core::replication::OplogEntry {
                op_id,
                op_type,
                timestamp,
                target_rid,
                payload,
                actor_id,
                hlc,
                embedding_hash,
                origin_actor,
            });
        }

        let stats = yantrikdb_core::replication::apply_ops(db, &entries).map_err(map_err)?;
        let dict = PyDict::new(py);
        dict.set_item("ops_applied", stats.ops_applied)?;
        dict.set_item("ops_skipped", stats.ops_skipped)?;
        Ok(dict.into())
    }

    fn get_peer_watermark(&self, py: Python<'_>, peer_actor: &str) -> PyResult<Option<PyObject>> {
        let db = self.get_inner()?;
        match yantrikdb_core::replication::get_peer_watermark(&*db.conn(), peer_actor)
            .map_err(map_err)?
        {
            Some((hlc, op_id)) => {
                let dict = PyDict::new(py);
                dict.set_item("hlc", &hlc)?;
                dict.set_item("op_id", &op_id)?;
                Ok(Some(dict.into()))
            }
            None => Ok(None),
        }
    }

    fn set_peer_watermark(&self, peer_actor: &str, hlc: Vec<u8>, op_id: &str) -> PyResult<()> {
        let db = self.get_inner()?;
        yantrikdb_core::replication::set_peer_watermark(&*db.conn(), peer_actor, &hlc, op_id)
            .map_err(map_err)
    }

    fn rebuild_vec_index(&self) -> PyResult<usize> {
        let db = self.get_inner()?;
        db.rebuild_vec_index().map_err(map_err)
    }

    fn rebuild_graph_index(&self) -> PyResult<usize> {
        let db = self.get_inner()?;
        db.rebuild_graph_index().map_err(map_err)
    }

    /// Repair memories whose stored text carries a leaked tool-call
    /// serialization artifact (task 30). Call with `dry_run=True` first to
    /// see the scope, then `dry_run=False` to apply. Returns a report dict.
    #[pyo3(signature = (dry_run = true))]
    fn repair_tool_call_artifacts(&self, py: Python<'_>, dry_run: bool) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let report = db.repair_tool_call_artifacts(dry_run).map_err(map_err)?;

        let dict = PyDict::new(py);
        dict.set_item("dry_run", report.dry_run)?;
        dict.set_item("scanned", report.scanned)?;
        dict.set_item("artifacts_found", report.artifacts_found)?;
        dict.set_item("repaired", report.repaired)?;
        dict.set_item(
            "skipped_concurrent_modification",
            report.skipped_concurrent_modification,
        )?;
        dict.set_item("stripped_bytes", report.stripped_bytes)?;
        dict.set_item("sample_rids", report.sample_rids)?;

        let errors = pyo3::types::PyList::empty(py);
        for e in &report.errors {
            let ed = PyDict::new(py);
            ed.set_item("rid", &e.rid)?;
            ed.set_item("message", &e.message)?;
            errors.append(ed)?;
        }
        dict.set_item("errors", errors)?;

        Ok(dict.into())
    }

    /// Revert stale, unused, high-importance memories toward baseline
    /// (use-it-or-lose-it, task 32). Background maintenance; call with
    /// `dry_run=True` first. Returns a report dict.
    #[pyo3(signature = (dry_run = true))]
    fn recalibrate_unused_importance(&self, py: Python<'_>, dry_run: bool) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let report = db.recalibrate_unused_importance(dry_run).map_err(map_err)?;

        let dict = PyDict::new(py);
        dict.set_item("dry_run", report.dry_run)?;
        dict.set_item("scanned", report.scanned)?;
        dict.set_item("adjusted", report.adjusted)?;
        dict.set_item("total_drift", report.total_drift)?;
        dict.set_item("sample_rids", report.sample_rids)?;
        Ok(dict.into())
    }

    /// Split oversized episodic memories into atomic semantic facts linked
    /// back to the source episode (task 33). Background maintenance; call
    /// with `dry_run=True` first. `min_chars` is the plaintext length above
    /// which an episode is a candidate. Returns a report dict.
    #[pyo3(signature = (dry_run = true, min_chars = 1500))]
    fn split_oversized_episodes(
        &self,
        py: Python<'_>,
        dry_run: bool,
        min_chars: usize,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let report = db
            .split_oversized_episodes(dry_run, min_chars)
            .map_err(map_err)?;

        let dict = PyDict::new(py);
        dict.set_item("dry_run", report.dry_run)?;
        dict.set_item("episodes_scanned", report.episodes_scanned)?;
        dict.set_item("episodes_split", report.episodes_split)?;
        dict.set_item("atomic_facts_created", report.atomic_facts_created)?;
        dict.set_item("sample_parent_rids", report.sample_parent_rids)?;
        dict.set_item("errors", report.errors)?;
        Ok(dict.into())
    }

    /// Burn down open conflicts: auto-resolve the unambiguous ones
    /// (newer-supersedes) and leave ambiguous/high-stakes ones for an
    /// operator (task 26). Dry-run first. Returns a report dict.
    #[pyo3(signature = (dry_run = true))]
    fn auto_resolve_conflicts(&self, py: Python<'_>, dry_run: bool) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let report = db.auto_resolve_conflicts(dry_run).map_err(map_err)?;

        let dict = PyDict::new(py);
        dict.set_item("dry_run", report.dry_run)?;
        dict.set_item("open_before", report.open_before)?;
        dict.set_item("auto_resolved", report.auto_resolved)?;
        dict.set_item("routed_to_operator", report.routed_to_operator)?;
        dict.set_item("sample_resolved", report.sample_resolved)?;
        dict.set_item("errors", report.errors)?;
        Ok(dict.into())
    }

    /// Bound the pending-trigger backlog: expire overdue triggers and evict
    /// the lowest-urgency excess over `max_pending` (task 27). Dry-run first.
    #[pyo3(signature = (dry_run = true, max_pending = 64))]
    fn prune_triggers(
        &self,
        py: Python<'_>,
        dry_run: bool,
        max_pending: usize,
    ) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let report = db.prune_triggers(dry_run, max_pending).map_err(map_err)?;

        let dict = PyDict::new(py);
        dict.set_item("dry_run", report.dry_run)?;
        dict.set_item("pending_before", report.pending_before)?;
        dict.set_item("expired_overdue", report.expired_overdue)?;
        dict.set_item("expired_over_cap", report.expired_over_cap)?;
        dict.set_item("pending_after", report.pending_after)?;
        Ok(dict.into())
    }

    /// Total skill outcomes recorded in the durable timeline (task 28).
    fn skill_outcome_count(&self) -> PyResult<usize> {
        let db = self.get_inner()?;
        db.skill_outcome_count().map_err(map_err)
    }

    /// Run one maintenance cycle — the sleep cycle (task 24). Drives the
    /// enabled hygiene passes once and returns a JSON summary string. The
    /// heavier corpus-rewriting passes (split, repair) are opt-in. A host
    /// schedules this on a timer; the engine does not own one.
    #[pyo3(signature = (
        run_think = true,
        burn_down_conflicts = true,
        prune_triggers = true,
        max_pending_triggers = 64,
        recalibrate_importance = true,
        backfill_entities = true,
        auto_relate = true,
        max_auto_relate_edges = 500,
        split_oversized = false,
        split_min_chars = 1500,
        repair_artifacts = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn run_maintenance_cycle(
        &self,
        run_think: bool,
        burn_down_conflicts: bool,
        prune_triggers: bool,
        max_pending_triggers: usize,
        recalibrate_importance: bool,
        backfill_entities: bool,
        auto_relate: bool,
        max_auto_relate_edges: usize,
        split_oversized: bool,
        split_min_chars: usize,
        repair_artifacts: bool,
    ) -> PyResult<String> {
        let db = self.get_inner()?;
        let cfg = yantrikdb_core::MaintenanceCycleConfig {
            run_think,
            burn_down_conflicts,
            prune_triggers,
            max_pending_triggers,
            recalibrate_importance,
            backfill_entities,
            auto_relate,
            max_auto_relate_edges,
            split_oversized,
            split_min_chars,
            repair_artifacts,
        };
        let report = db.run_maintenance_cycle(&cfg).map_err(map_err)?;
        Ok(serde_json::to_string(&report).unwrap_or_else(|_| "{}".to_string()))
    }

    /// The last persisted maintenance-cycle summary (JSON), or None.
    fn last_maintenance_cycle(&self) -> PyResult<Option<String>> {
        let db = self.get_inner()?;
        db.last_maintenance_cycle().map_err(map_err)
    }

    /// Materialize the session-start digest (task 38) — narrative chain head,
    /// top live decisions, open conflicts, pending triggers, last maintenance.
    /// One call, host-injected at boot. Returns a JSON string.
    #[pyo3(signature = (
        narrative_namespace = None,
        max_decisions = 8,
        max_conflicts = 5,
        max_triggers = 5,
        snippet_chars = 240
    ))]
    fn session_digest(
        &self,
        narrative_namespace: Option<String>,
        max_decisions: usize,
        max_conflicts: usize,
        max_triggers: usize,
        snippet_chars: usize,
    ) -> PyResult<String> {
        let db = self.get_inner()?;
        let cfg = yantrikdb_core::SessionDigestConfig {
            narrative_namespace,
            max_decisions,
            max_conflicts,
            max_triggers,
            snippet_chars,
        };
        let digest = db.session_digest(&cfg).map_err(map_err)?;
        Ok(serde_json::to_string(&digest).unwrap_or_else(|_| "{}".to_string()))
    }

    /// End-of-session auto-capture (task 40): atomize an agent-provided
    /// session summary into provisional candidate memories. Returns new rids.
    #[pyo3(signature = (summary, namespace = "default", domain = "general"))]
    fn draft_memories_from_summary(
        &self,
        summary: &str,
        namespace: &str,
        domain: &str,
    ) -> PyResult<Vec<String>> {
        let db = self.get_inner()?;
        db.draft_memories_from_summary(summary, namespace, domain)
            .map_err(map_err)
    }

    /// Auto-relate co-occurring entities to raise graph density (task 44).
    /// Dry-run first. Returns a report dict.
    #[pyo3(signature = (dry_run = true, max_edges = 500))]
    fn auto_relate(&self, py: Python<'_>, dry_run: bool, max_edges: usize) -> PyResult<PyObject> {
        let db = self.get_inner()?;
        let report = db.auto_relate(dry_run, max_edges).map_err(map_err)?;
        let dict = PyDict::new(py);
        dict.set_item("dry_run", report.dry_run)?;
        dict.set_item("pairs_considered", report.pairs_considered)?;
        dict.set_item("edges_upserted", report.edges_upserted)?;
        Ok(dict.into())
    }
}
