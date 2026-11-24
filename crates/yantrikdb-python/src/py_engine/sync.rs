use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
type PyObject = pyo3::Py<pyo3::PyAny>;

use crate::py_types::*;

use super::{py_vector, run_detached, PyYantrikDB};

/// Convert a core `Task` into a Python dict.
fn task_to_dict(py: Python<'_>, t: &yantrikdb_core::Task) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("id", &t.id)?;
    d.set_item("namespace", &t.namespace)?;
    d.set_item("title", &t.title)?;
    d.set_item("status", &t.status)?;
    d.set_item("priority", &t.priority)?;
    d.set_item("parent_id", &t.parent_id)?;
    d.set_item("created_at", t.created_at)?;
    d.set_item("updated_at", t.updated_at)?;
    Ok(d.into())
}

#[pymethods]
impl PyYantrikDB {
    // ── Storage tier operations ──

    fn archive(slf: &Bound<'_, Self>, rid: &str) -> PyResult<bool> {
        Self::detached(slf, |db| db.archive(rid))
    }

    fn hydrate(slf: &Bound<'_, Self>, rid: &str) -> PyResult<bool> {
        Self::detached(slf, |db| db.hydrate(rid))
    }

    #[pyo3(signature = (max_active,))]
    fn evict(slf: &Bound<'_, Self>, max_active: usize) -> PyResult<Vec<String>> {
        Self::detached(slf, |db| db.evict(max_active))
    }

    // ── Replication API (V5 P2P Sync) ──

    #[pyo3(signature = (since_hlc=None, since_op_id=None, exclude_actor=None, limit=1000))]
    fn extract_ops_since(
        slf: &Bound<'_, Self>,
        since_hlc: Option<Vec<u8>>,
        since_op_id: Option<&str>,
        exclude_actor: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        // 0.13.2: pass the provider so sealed payloads unseal — without
        // it a sealed row would parse to an EMPTY payload and the op
        // would replicate as a no-op.
        let ops = Self::detached(slf, |db| {
            yantrikdb_core::replication::extract_ops_since_enc(
                &db.conn(),
                db.encryption(),
                since_hlc.as_deref(),
                since_op_id,
                exclude_actor,
                limit,
            )
        })?;

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
            // v0.10 Item 3: exact embedding bytes for re-embedding corrections.
            dict.set_item("embedding", &op.embedding)?;
            result.push(dict.into());
        }
        Ok(result)
    }

    fn apply_ops(slf: &Bound<'_, Self>, ops: Vec<Bound<'_, PyDict>>) -> PyResult<PyObject> {
        // Closed is reported before any argument error, as before the
        // GIL sweep (#271).
        let db = Self::engine_handle(slf)?;
        let py = slf.py();
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
            // v0.10 Item 3: optional exact embedding bytes for re-embedding
            // corrections. Absent on every other op and on pre-v0.10 peers.
            let embedding: Option<Vec<u8>> = d
                .get_item("embedding")?
                .and_then(|v| if v.is_none() { None } else { Some(v) })
                .map(|v| v.extract())
                .transpose()?;

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
                embedding,
            });
        }

        let stats = run_detached(slf.py(), db, |db| {
            yantrikdb_core::replication::apply_ops(db, &entries)
        })?;
        let dict = PyDict::new(py);
        dict.set_item("ops_applied", stats.ops_applied)?;
        dict.set_item("ops_skipped", stats.ops_skipped)?;
        Ok(dict.into())
    }

    fn get_peer_watermark(slf: &Bound<'_, Self>, peer_actor: &str) -> PyResult<Option<PyObject>> {
        let watermark = Self::detached(slf, |db| {
            yantrikdb_core::replication::get_peer_watermark(&db.conn(), peer_actor)
        })?;
        match watermark {
            Some((hlc, op_id)) => {
                let dict = PyDict::new(slf.py());
                dict.set_item("hlc", &hlc)?;
                dict.set_item("op_id", &op_id)?;
                Ok(Some(dict.into()))
            }
            None => Ok(None),
        }
    }

    fn set_peer_watermark(
        slf: &Bound<'_, Self>,
        peer_actor: &str,
        hlc: Vec<u8>,
        op_id: &str,
    ) -> PyResult<()> {
        Self::detached(slf, |db| {
            yantrikdb_core::replication::set_peer_watermark(&db.conn(), peer_actor, &hlc, op_id)
        })
    }

    fn rebuild_vec_index(slf: &Bound<'_, Self>) -> PyResult<usize> {
        Self::detached(slf, |db| db.rebuild_vec_index())
    }

    fn rebuild_graph_index(slf: &Bound<'_, Self>) -> PyResult<usize> {
        Self::detached(slf, |db| db.rebuild_graph_index())
    }

    /// Repair memories whose stored text carries a leaked tool-call
    /// serialization artifact (task 30). Call with `dry_run=True` first to
    /// see the scope, then `dry_run=False` to apply. Returns a report dict.
    #[pyo3(signature = (dry_run = true))]
    fn repair_tool_call_artifacts(slf: &Bound<'_, Self>, dry_run: bool) -> PyResult<PyObject> {
        let py = slf.py();
        let report = Self::detached(slf, |db| db.repair_tool_call_artifacts(dry_run))?;

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

    /// Re-encode records whose vectors are missing or unusable, using the
    /// embedder that is ALREADY ACTIVE.
    ///
    /// This is not a re-embed: it changes no model, fingerprint, dimension or
    /// generation, so none of the cutover machinery applies. It fills in
    /// vectors that are absent or malformed, in the space the engine is
    /// already using.
    ///
    /// The case that matters most is REPLICATION — the follower insert omits
    /// `embedding`, `embedding_model` and `embedding_generation` entirely, so
    /// replicated rows arrive active in SQL and invisible to semantic recall.
    /// Without this binding that repair was reachable only from Rust, which
    /// meant an operator hitting the problem through the server or MCP had no
    /// way to run it.
    ///
    /// `namespace=None` sweeps the whole store. Call with `dry_run=True`
    /// first; it reports what it would touch without writing.
    #[pyo3(signature = (namespace = None, dry_run = true))]
    fn refresh_embeddings(
        slf: &Bound<'_, Self>,
        namespace: Option<&str>,
        dry_run: bool,
    ) -> PyResult<PyObject> {
        let py = slf.py();
        // An attached Python embedder re-encodes the rows, in the precedence
        // every write follows. The engine's refresh_embeddings could only use
        // its own, so on a follower whose only embedder is a Python one the
        // replicated rows (which arrive without vectors) could never be
        // repaired: it raised "No embedder configured". The rows are encoded
        // inside the detached call, re-taking the GIL once per batch of up
        // to 256 texts.
        let (db, embedder) = Self::engine_and_embedder(slf)?;
        // The rows record no embedding_model (NULL): never the engine's
        // embedder, which did not make these vectors (#304 review 3), and
        // not the Python embedder's self-declared `name` either, which the
        // engine cannot vouch for. Every other write a Python embedder makes
        // leaves it NULL too (#304 review 4).
        let report = run_detached(py, db, |db| match embedder.as_ref() {
            Some(emb) => {
                db.refresh_embeddings_with_batch(namespace, dry_run, None, |texts: &[&str]| {
                    Python::attach(|py| {
                        texts
                            .iter()
                            .map(|text| {
                                emb.call_method1(py, "encode", (*text,))
                                    .and_then(|out| py_vector(py, &out))
                                    .map_err(|e| {
                                        yantrikdb_core::YantrikDbError::Inference(e.to_string())
                                    })
                            })
                            .collect()
                    })
                })
            }
            None => db.refresh_embeddings(namespace, dry_run),
        })?;

        let dict = PyDict::new(py);
        dict.set_item("dry_run", report.dry_run)?;
        dict.set_item("namespace", report.namespace.clone())?;
        dict.set_item("scanned", report.scanned)?;
        dict.set_item("unusable_found", report.unusable_found)?;
        dict.set_item("refreshed", report.refreshed)?;
        dict.set_item(
            "skipped_concurrent_modification",
            report.skipped_concurrent_modification,
        )?;
        dict.set_item("sample_rids", report.sample_rids.clone())?;

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
    fn recalibrate_unused_importance(slf: &Bound<'_, Self>, dry_run: bool) -> PyResult<PyObject> {
        let py = slf.py();
        let report = Self::detached(slf, |db| db.recalibrate_unused_importance(dry_run))?;

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
        slf: &Bound<'_, Self>,
        dry_run: bool,
        min_chars: usize,
    ) -> PyResult<PyObject> {
        let py = slf.py();
        let report = Self::detached(slf, |db| db.split_oversized_episodes(dry_run, min_chars))?;

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
    fn auto_resolve_conflicts(slf: &Bound<'_, Self>, dry_run: bool) -> PyResult<PyObject> {
        let py = slf.py();
        let report = Self::detached(slf, |db| db.auto_resolve_conflicts(dry_run))?;

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
        slf: &Bound<'_, Self>,
        dry_run: bool,
        max_pending: usize,
    ) -> PyResult<PyObject> {
        let py = slf.py();
        let report = Self::detached(slf, |db| db.prune_triggers(dry_run, max_pending))?;

        let dict = PyDict::new(py);
        dict.set_item("dry_run", report.dry_run)?;
        dict.set_item("pending_before", report.pending_before)?;
        dict.set_item("expired_overdue", report.expired_overdue)?;
        dict.set_item("expired_stale", report.expired_stale)?;
        dict.set_item("expired_over_cap", report.expired_over_cap)?;
        dict.set_item("pending_after", report.pending_after)?;
        Ok(dict.into())
    }

    /// Total skill outcomes recorded in the durable timeline (task 28).
    fn skill_outcome_count(slf: &Bound<'_, Self>) -> PyResult<usize> {
        Self::detached(slf, |db| db.skill_outcome_count())
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
        repair_artifacts = false,
        dry_run = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn run_maintenance_cycle(
        slf: &Bound<'_, Self>,
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
        dry_run: bool,
    ) -> PyResult<String> {
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
            dry_run,
        };
        let report = Self::detached(slf, |db| db.run_maintenance_cycle(&cfg))?;
        Ok(serde_json::to_string(&report).map_err(|e| {
            // A maintenance cycle that RAN must never report as "{}" — the
            // caller could not tell "ran, report lost" from "did nothing".
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "report serialization failed after the cycle ran: {e}"
            ))
        })?)
    }

    /// The last persisted maintenance-cycle summary (JSON), or None.
    fn last_maintenance_cycle(slf: &Bound<'_, Self>) -> PyResult<Option<String>> {
        Self::detached(slf, |db| db.last_maintenance_cycle())
    }

    /// The maintenance-debt ledger — how overdue is cognition? Returns a dict
    /// with exactly four keys: `writes_since_think` (memory writes committed
    /// since cognition last completed a pass), `last_think_at` (epoch float,
    /// None if cognition has never run), `open_conflicts`, and
    /// `pending_triggers`. Read-only and cheap; a reactive host (the MCP
    /// server) surfaces this to the calling LLM, which acts as the scheduler
    /// the passive engine deliberately does not own.
    fn maintenance_debt(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
        let py = slf.py();
        let debt = Self::detached(slf, |db| Ok(db.maintenance_debt()))?;
        let dict = PyDict::new(py);
        dict.set_item("writes_since_think", debt.writes_since_think)?;
        dict.set_item("last_think_at", debt.last_think_at)?;
        dict.set_item("open_conflicts", debt.open_conflicts)?;
        dict.set_item("pending_triggers", debt.pending_triggers)?;
        Ok(dict.into())
    }

    /// Materialize the session-start digest (task 38) — narrative chain head,
    /// top live decisions, open conflicts, pending triggers, last maintenance.
    /// One call, host-injected at boot. Returns a JSON string.
    #[pyo3(signature = (
        narrative_namespace = None,
        max_decisions = 8,
        max_conflicts = 5,
        max_triggers = 5,
        snippet_chars = 240,
        namespace = None,
        include_superseded = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn session_digest(
        slf: &Bound<'_, Self>,
        narrative_namespace: Option<String>,
        max_decisions: usize,
        max_conflicts: usize,
        max_triggers: usize,
        snippet_chars: usize,
        namespace: Option<String>,
        include_superseded: bool,
    ) -> PyResult<String> {
        let cfg = yantrikdb_core::SessionDigestConfig {
            narrative_namespace,
            // v0.9.3: optional content scope — decisions + conflicts filtered
            // to this namespace so per-tenant digests never mix tenants.
            namespace,
            max_decisions,
            max_conflicts,
            max_triggers,
            snippet_chars,
            // v0.10 Item 1c: history packets re-admit superseded records,
            // stamped with current_status + superseded_by.
            include_superseded,
        };
        let digest = Self::detached(slf, |db| db.session_digest(&cfg))?;
        serde_json::to_string(&digest).map_err(|e| {
            // Never report "{}" for an operation that ran — the caller
            // cannot tell "ran, report lost" from "did nothing".
            pyo3::exceptions::PyRuntimeError::new_err(format!("result serialization failed: {e}"))
        })
    }

    /// v0.10 Item 1c (trace T10) — what changed since `since` (epoch
    /// seconds): records created and status transitions (supersessions)
    /// committed after that instant, optionally scoped to a namespace.
    /// Returns a JSON string with `new_records` and `status_transitions`.
    #[pyo3(signature = (since, namespace = None, snippet_chars = 240))]
    fn what_changed_since(
        slf: &Bound<'_, Self>,
        since: f64,
        namespace: Option<&str>,
        snippet_chars: usize,
    ) -> PyResult<String> {
        let changes = Self::detached(slf, |db| {
            db.what_changed_since(since, namespace, snippet_chars)
        })?;
        serde_json::to_string(&changes).map_err(|e| {
            // Never report "{}" for an operation that ran — the caller
            // cannot tell "ran, report lost" from "did nothing".
            pyo3::exceptions::PyRuntimeError::new_err(format!("result serialization failed: {e}"))
        })
    }

    /// End-of-session auto-capture (task 40): atomize an agent-provided
    /// session summary into provisional candidate memories. Returns new rids.
    #[pyo3(signature = (summary, namespace = "default", domain = "general"))]
    fn draft_memories_from_summary(
        slf: &Bound<'_, Self>,
        summary: &str,
        namespace: &str,
        domain: &str,
    ) -> PyResult<Vec<String>> {
        let py = slf.py();
        let (db, embedder) = Self::engine_and_embedder(slf)?;
        // An embedder the caller attached outranks the engine's own, as in
        // every other write (`embed_text`'s precedence). The engine cannot
        // hold a Python embedder, so its `draft_memories_from_summary` raised
        // "No embedder configured" on a store that had only one.
        let Some(emb) = embedder else {
            return run_detached(py, db, |db| {
                db.draft_memories_from_summary(summary, namespace, domain)
            });
        };
        // The facts go through the engine's record_text route with the
        // caller's embedder: a fact past the known window gets its chunk
        // vectors as it is written, as on the engine path.
        //
        // Every fact is encoded here, before the call detaches, under one
        // hold of the GIL and with the store marked (a call encode() makes
        // back into the store skips a pending swap's gate). Re-taking the
        // GIL inside the detached call for each fact waited about a switch
        // interval per fact whenever another thread was busy: 60 drafts
        // took 1,021 ms beside a CPU-bound thread against 164 ms alone
        // (#304 review 4). The detached call re-takes the GIL only for
        // texts it was not handed, the chunk windows of a long fact.
        let facts = yantrikdb_core::YantrikDB::draft_facts_from_summary(summary);
        let mut vectors: std::collections::HashMap<String, Vec<f32>> =
            std::collections::HashMap::with_capacity(facts.len());
        super::in_engine_call(&db, || -> PyResult<()> {
            for fact in facts {
                let out = emb.call_method1(py, "encode", (fact.as_str(),))?;
                let vector = py_vector(py, &out)?;
                vectors.insert(fact, vector);
            }
            Ok(())
        })?;
        let embed_window = super::py_embed(&emb);
        run_detached(py, db, |db| {
            db.draft_memories_from_summary_embedded_by(summary, namespace, domain, |text: &str| {
                match vectors.get(text) {
                    Some(vector) => Ok(vector.clone()),
                    None => embed_window(text),
                }
            })
        })
    }

    /// Auto-relate co-occurring entities to raise graph density (task 44).
    /// Dry-run first. Returns a report dict.
    #[pyo3(signature = (dry_run = true, max_edges = 500))]
    fn auto_relate(slf: &Bound<'_, Self>, dry_run: bool, max_edges: usize) -> PyResult<PyObject> {
        let py = slf.py();
        let report = Self::detached(slf, |db| db.auto_relate(dry_run, max_edges))?;
        let dict = PyDict::new(py);
        dict.set_item("dry_run", report.dry_run)?;
        dict.set_item("pairs_considered", report.pairs_considered)?;
        dict.set_item("edges_upserted", report.edges_upserted)?;
        Ok(dict.into())
    }

    /// Append a raw conversation turn to the namespace's working-memory ring
    /// buffer, pruned to the last `max_turns` (v0.9.0). Verbatim, not embedded.
    /// Returns the turn id.
    #[pyo3(signature = (namespace, role, content, max_turns = 10))]
    fn record_turn(
        slf: &Bound<'_, Self>,
        namespace: &str,
        role: &str,
        content: &str,
        max_turns: usize,
    ) -> PyResult<i64> {
        Self::detached(slf, |db| {
            db.record_turn(namespace, role, content, max_turns)
        })
    }

    /// The last `limit` turns for a namespace, oldest-first (ready to prepend
    /// to a prompt as recent context).
    #[pyo3(signature = (namespace, limit = 10))]
    fn recent_turns(
        slf: &Bound<'_, Self>,
        namespace: &str,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        let turns = Self::detached(slf, |db| db.recent_turns(namespace, limit))?;
        turns
            .iter()
            .map(|t| {
                let d = PyDict::new(py);
                d.set_item("role", &t.role)?;
                d.set_item("content", &t.content)?;
                d.set_item("created_at", t.created_at)?;
                Ok(d.into())
            })
            .collect()
    }

    /// Clear a namespace's conversation buffer. Returns rows removed.
    fn clear_turns(slf: &Bound<'_, Self>, namespace: &str) -> PyResult<usize> {
        Self::detached(slf, |db| db.clear_turns(namespace))
    }

    /// Surface knowledge gaps — frequently-asked, poorly-answered queries the
    /// substrate should satisfy but can't (v0.9.0). Returns a list of
    /// {query, count, avg_top_score, avg_results, last_seen} dicts.
    ///
    /// v0.9.3: scoped — `namespace=<ns>` reads that namespace's demand;
    /// `namespace=None` reads the global bucket (recalls issued without a
    /// namespace filter). One call never mixes scopes. Always empty on
    /// encrypted databases (demand capture is disabled there).
    #[pyo3(signature = (min_count = 3, max_avg_top_score = 0.4, limit = 20, namespace = None))]
    fn knowledge_gaps(
        slf: &Bound<'_, Self>,
        min_count: u64,
        max_avg_top_score: f64,
        limit: usize,
        namespace: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        let gaps = Self::detached(slf, |db| {
            db.knowledge_gaps(namespace, min_count, max_avg_top_score, limit)
        })?;
        gaps.iter()
            .map(|g| {
                let d = PyDict::new(py);
                d.set_item("query", &g.query)?;
                d.set_item("count", g.count)?;
                d.set_item("avg_top_score", g.avg_top_score)?;
                d.set_item("avg_results", g.avg_results)?;
                d.set_item("last_seen", g.last_seen)?;
                Ok(d.into())
            })
            .collect()
    }

    /// Create a task/chore in a namespace. Returns the new task id. (v0.9.0)
    #[pyo3(signature = (namespace, title, priority = "medium", parent_id = None))]
    fn task_add(
        slf: &Bound<'_, Self>,
        namespace: &str,
        title: &str,
        priority: &str,
        parent_id: Option<&str>,
    ) -> PyResult<String> {
        Self::detached(slf, |db| db.task_add(namespace, title, priority, parent_id))
    }

    /// Update a task's status and/or priority. Returns whether it existed.
    #[pyo3(signature = (id, status = None, priority = None))]
    fn task_update(
        slf: &Bound<'_, Self>,
        id: &str,
        status: Option<&str>,
        priority: Option<&str>,
    ) -> PyResult<bool> {
        Self::detached(slf, |db| db.task_update(id, status, priority))
    }

    /// List tasks in a namespace, optionally filtered by status, priority-ordered.
    #[pyo3(signature = (namespace, status = None))]
    fn task_list(
        slf: &Bound<'_, Self>,
        namespace: &str,
        status: Option<&str>,
    ) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        let tasks = Self::detached(slf, |db| db.task_list(namespace, status))?;
        tasks.iter().map(|t| task_to_dict(py, t)).collect()
    }

    /// Get one task by id, or None.
    fn task_get(slf: &Bound<'_, Self>, id: &str) -> PyResult<Option<PyObject>> {
        let py = slf.py();
        match Self::detached(slf, |db| db.task_get(id))? {
            Some(t) => Ok(Some(task_to_dict(py, &t)?)),
            None => Ok(None),
        }
    }

    /// Delete a task. Returns whether a row was removed.
    fn task_delete(slf: &Bound<'_, Self>, id: &str) -> PyResult<bool> {
        Self::detached(slf, |db| db.task_delete(id))
    }
}
