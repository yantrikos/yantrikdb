use pyo3::prelude::*;
use pyo3::types::PyDict;
type PyObject = pyo3::Py<pyo3::PyAny>;

use crate::py_types::*;

use super::{map_err, to_py, PyYantrikDB};

#[pymethods]
impl PyYantrikDB {
    // ── Cognition loop (V3) ──

    #[pyo3(signature = (config=None))]
    fn think(slf: &Bound<'_, Self>, config: Option<&Bound<'_, PyDict>>) -> PyResult<PyObject> {
        let py = slf.py();
        // No borrow of `slf` is held past this line; see `engine_handle`.
        let db = Self::engine_handle(slf)?;
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
            if let Ok(Some(v)) = d.get_item("redundancy_sim_threshold") {
                c.redundancy_sim_threshold = v.extract()?;
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
            const KNOWN: [&str; 15] = [
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
                "redundancy_sim_threshold",
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
        let result = {
            // Release the GIL for the engine call (#271). `think()` is CPU-bound
            // and, without this, holds the GIL for its whole duration: a 10 ms
            // Python ticker measured 99 increments/s before the call and 0 after
            // it started, for over 110 s on a 23k-record store. Everything else
            // in the host — the async gateway's event loop included — starves
            // while it runs.
            //
            // `detach` is what older PyO3 docs call `allow_threads`; the rename
            // landed in 0.26 and this crate is on 0.28.3, where `allow_threads`
            // no longer exists.
            //
            // Nothing in the closure touches Python: the config dict was already
            // converted to `ThinkConfig` above, and the result dict is built
            // after reattaching. `Arc<YantrikDB>` is owned and `Send + Sync`,
            // which is what `Ungil` requires.
            py.detach(move || db.think(&cfg)).map_err(map_err)?
        };
        think_result_to_dict(py, &result)
    }

    fn deliver_trigger(slf: &Bound<'_, Self>, trigger_id: &str) -> PyResult<bool> {
        Self::detached(slf, |db| db.deliver_trigger(trigger_id))
    }

    fn acknowledge_trigger(slf: &Bound<'_, Self>, trigger_id: &str) -> PyResult<bool> {
        Self::detached(slf, |db| db.acknowledge_trigger(trigger_id))
    }

    fn act_on_trigger(slf: &Bound<'_, Self>, trigger_id: &str) -> PyResult<bool> {
        Self::detached(slf, |db| db.act_on_trigger(trigger_id))
    }

    fn dismiss_trigger(slf: &Bound<'_, Self>, trigger_id: &str) -> PyResult<bool> {
        Self::detached(slf, |db| db.dismiss_trigger(trigger_id))
    }

    #[pyo3(signature = (limit=10))]
    fn get_pending_triggers(slf: &Bound<'_, Self>, limit: usize) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        let triggers = Self::detached(slf, |db| db.get_pending_triggers(limit))?;
        triggers
            .iter()
            .map(|t| persisted_trigger_to_dict(py, t))
            .collect()
    }

    #[pyo3(signature = (trigger_type=None, limit=50))]
    fn get_trigger_history(
        slf: &Bound<'_, Self>,
        trigger_type: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        let triggers = Self::detached(slf, |db| db.get_trigger_history(trigger_type, limit))?;
        triggers
            .iter()
            .map(|t| persisted_trigger_to_dict(py, t))
            .collect()
    }

    #[pyo3(signature = (pattern_type=None, status=None, limit=50))]
    fn get_patterns(
        slf: &Bound<'_, Self>,
        pattern_type: Option<&str>,
        status: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        // Counts and rows in ONE detached section, so the GIL is released
        // once for the whole page instead of once per pattern. It is not a
        // snapshot: `db.conn()` takes the connection mutex per call, so a
        // concurrent `forget` can land between two rows of this page. Each
        // individual (total, tombstoned) pair is one SELECT and is internally
        // consistent; cross-row consistency is not promised here.
        let rows = Self::detached(slf, |db| {
            let patterns = db.get_patterns(pattern_type, status, limit)?;
            let mut out = Vec::with_capacity(patterns.len());
            for p in patterns {
                let counts = db.pattern_evidence_counts(&p.pattern_id)?;
                // Data first, type second: if the pattern HAS evidence rows,
                // the numbers are facts and are reported whatever the type
                // says. `None` is reserved for "this type records no evidence
                // and this pattern has none" — so a miner that starts writing
                // evidence for a previously untracked type cannot be hidden
                // behind the static list.
                let counts = if counts.0 > 0
                    || yantrikdb_core::patterns::evidence_is_tracked(&p.pattern_type)
                {
                    Some(counts)
                } else {
                    None
                };
                out.push((p, counts));
            }
            Ok(out)
        })?;
        rows.iter()
            .map(|(p, counts)| pattern_to_dict(py, p, *counts))
            .collect()
    }

    // ── Personality API (V11) ──

    fn get_personality(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
        let profile = Self::detached(slf, |db| db.get_personality())?;
        personality_profile_to_dict(slf.py(), &profile)
    }

    fn derive_personality(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
        let profile = Self::detached(slf, |db| db.derive_personality())?;
        personality_profile_to_dict(slf.py(), &profile)
    }

    #[pyo3(signature = (name, score))]
    fn set_personality_trait(slf: &Bound<'_, Self>, name: &str, score: f64) -> PyResult<bool> {
        Self::detached(slf, |db| db.set_personality_trait(name, score))
    }

    // ── Conflict resolution API (V2) ──

    #[pyo3(signature = (status=None, conflict_type=None, entity=None, priority=None, namespace=None, limit=50))]
    fn get_conflicts(
        slf: &Bound<'_, Self>,
        status: Option<&str>,
        conflict_type: Option<&str>,
        entity: Option<&str>,
        priority: Option<&str>,
        namespace: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        let conflicts = Self::detached(slf, |db| {
            db.get_conflicts(status, conflict_type, entity, priority, namespace, limit)
        })?;
        conflicts.iter().map(|c| conflict_to_dict(py, c)).collect()
    }

    fn get_conflict(slf: &Bound<'_, Self>, conflict_id: &str) -> PyResult<Option<PyObject>> {
        match Self::detached(slf, |db| db.get_conflict(conflict_id))? {
            Some(c) => Ok(Some(conflict_to_dict(slf.py(), &c)?)),
            None => Ok(None),
        }
    }

    #[pyo3(signature = (conflict_id, strategy, winner_rid=None, new_text=None, resolution_note=None))]
    fn resolve_conflict(
        slf: &Bound<'_, Self>,
        conflict_id: &str,
        strategy: &str,
        winner_rid: Option<&str>,
        new_text: Option<&str>,
        resolution_note: Option<&str>,
    ) -> PyResult<PyObject> {
        let py = slf.py();
        let result = Self::detached(slf, |db| {
            db.resolve_conflict(conflict_id, strategy, winner_rid, new_text, resolution_note)
        })?;
        let dict = PyDict::new(py);
        dict.set_item("conflict_id", &result.conflict_id)?;
        dict.set_item("strategy", &result.strategy)?;
        dict.set_item("winner_rid", &result.winner_rid)?;
        dict.set_item("loser_tombstoned", result.loser_tombstoned)?;
        dict.set_item("new_memory_rid", &result.new_memory_rid)?;
        Ok(dict.into())
    }

    #[pyo3(signature = (conflict_id, note=None))]
    fn dismiss_conflict(
        slf: &Bound<'_, Self>,
        conflict_id: &str,
        note: Option<&str>,
    ) -> PyResult<()> {
        Self::detached(slf, |db| db.dismiss_conflict(conflict_id, note))
    }

    fn scan_conflicts(slf: &Bound<'_, Self>) -> PyResult<Vec<PyObject>> {
        let py = slf.py();
        // No borrow of `slf` is held across the detached call; see
        // `engine_handle` for why that matters.
        let db = Self::engine_handle(slf)?;
        // Release the GIL for the engine call (#271). `scan_conflicts()`
        // walks every contradicting edge group on the identity/preference
        // whitelist plus the concurrent-consolidation pairs, and on a
        // loaded store a single call runs well over half a second;
        // without this it holds the GIL for the whole walk, stalling
        // every other thread in the host. `detach` is PyO3 0.28's name
        // for what older docs call `allow_threads`. Nothing in the
        // closure touches Python: it takes an owned `Arc<YantrikDB>` and
        // returns owned `Conflict` values, and the result dicts are
        // built after reattaching. `Arc<YantrikDB>` is owned and
        // `Send + Sync`, which is what `Ungil` requires.
        let conflicts = py
            .detach(move || yantrikdb_core::scan_conflicts(&db))
            .map_err(map_err)?;
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
        slf: &Bound<'_, Self>,
        namespace: &str,
        dry_run: bool,
    ) -> PyResult<PyObject> {
        let audit = Self::detached(slf, |db| {
            db.extract_standing_instructions(namespace, dry_run)
        })?;
        to_py(slf.py(), serde_json::to_value(&audit))
    }

    /// The standing-instruction salience lane: verified facets for a
    /// namespace in first-mention order, complete when within `limit`,
    /// deterministic prefix with an explicit `omitted` count otherwise.
    /// Ordinary `recall` is untouched; callers compose the two.
    #[pyo3(signature = (namespace="default", limit=8))]
    fn recall_facets(slf: &Bound<'_, Self>, namespace: &str, limit: usize) -> PyResult<PyObject> {
        let out = Self::detached(slf, |db| db.recall_facets(namespace, limit))?;
        to_py(slf.py(), serde_json::to_value(&out))
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
        slf: &Bound<'_, Self>,
        namespace: &str,
        entities: Vec<String>,
        limit: usize,
        phrases: Option<Vec<String>>,
        topic_rids: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let phrases = phrases.unwrap_or_default();
        let topic_rids = topic_rids.unwrap_or_default();
        let val = if phrases.is_empty() && topic_rids.is_empty() {
            // Legacy shape for entity-only calls (v1 compatibility).
            let entity_refs: Vec<&str> = entities.iter().map(String::as_str).collect();
            let out = Self::detached(slf, |db| db.recall_thread(namespace, &entity_refs, limit))?;
            serde_json::to_value(&out)
        } else {
            let query = yantrikdb_core::ThreadQuery {
                entities,
                phrases,
                topic_rids,
            };
            let out = Self::detached(slf, |db| db.recall_thread_v2(namespace, &query, limit))?;
            serde_json::to_value(&out)
        };
        to_py(slf.py(), val)
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
        slf: &Bound<'_, Self>,
        namespace: &str,
        entities: Option<Vec<String>>,
        limit: usize,
        phrases: Option<Vec<String>>,
        topic_rids: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let query = yantrikdb_core::ThreadQuery {
            entities: entities.unwrap_or_default(),
            phrases: phrases.unwrap_or_default(),
            topic_rids: topic_rids.unwrap_or_default(),
        };
        let out = Self::detached(slf, |db| db.recall_thread_v2(namespace, &query, limit))?;
        to_py(slf.py(), serde_json::to_value(&out))
    }

    /// Drop every extractor-minted claim (heuristic_v1 / learned_v1),
    /// optionally in one namespace, and re-extract from every active memory
    /// with the current extractor. relate() and writer-stated claims are
    /// never touched. `dry_run=True` reports and changes nothing. Returns
    /// `{memories_scanned, claims_removed, claims_written, before_by_rel,
    /// after_by_rel, ...}`.
    #[pyo3(signature = (namespace=None, dry_run=false))]
    fn reextract_claims(
        slf: &Bound<'_, Self>,
        namespace: Option<&str>,
        dry_run: bool,
    ) -> PyResult<PyObject> {
        let out = Self::detached(slf, |db| db.reextract_claims(namespace, dry_run))?;
        to_py(slf.py(), serde_json::to_value(&out))
    }

    /// Entity-table heal (issue #213): re-apply today's admission predicate to
    /// every stored entity, drop the refused ones (plus their links and any
    /// extractor claim on them) unless a manual/stated claim asserts them,
    /// then rebuild the graph index. Idempotent; `dry_run=True` only reports.
    #[pyo3(signature = (dry_run=false))]
    fn reextract_entities(slf: &Bound<'_, Self>, dry_run: bool) -> PyResult<PyObject> {
        let out = Self::detached(slf, |db| db.reextract_entities(dry_run))?;
        to_py(slf.py(), serde_json::to_value(&out))
    }

    /// The extraction refusal ledger as a histogram keyed `rel_type:reason`
    /// — why the bound extractor saw a relation trigger and would not bind
    /// it. The instrument the next binding rule is chosen from.
    #[pyo3(signature = (namespace=None))]
    fn extraction_refusal_counts(
        slf: &Bound<'_, Self>,
        namespace: Option<&str>,
    ) -> PyResult<PyObject> {
        let out = Self::detached(slf, |db| db.extraction_refusal_counts(namespace))?;
        to_py(slf.py(), serde_json::to_value(&out))
    }

    /// The ledger rows, newest first: memory_rid, rel_type, trigger, reason,
    /// left_token, right_token, at (byte offset), extractor_version.
    #[pyo3(signature = (namespace=None, limit=200))]
    fn extraction_refusals(
        slf: &Bound<'_, Self>,
        namespace: Option<&str>,
        limit: usize,
    ) -> PyResult<PyObject> {
        let out = Self::detached(slf, |db| db.extraction_refusals(namespace, limit))?;
        to_py(slf.py(), serde_json::to_value(&out))
    }

    /// Silver recall: re-derive every cooperative (`agent_stated`) claim
    /// from its source memory with the bound extractor. Returns
    /// stated_claims, recovered, unsupported_relation, missed_by_reason and
    /// recall (recovered over supported relations). Read-only; judge-free.
    #[pyo3(signature = (namespace=None))]
    fn extraction_silver_recall(
        slf: &Bound<'_, Self>,
        namespace: Option<&str>,
    ) -> PyResult<PyObject> {
        let out = Self::detached(slf, |db| db.extraction_silver_recall(namespace))?;
        to_py(slf.py(), serde_json::to_value(&out))
    }

    /// One batch of the v50 source_turn recompute/repair pass — the
    /// maintenance operation `SourceTurnMaintenanceRequiredError` names.
    /// Returns `{"processed", "remaining", "complete"}`; call repeatedly
    /// until `complete` is true.
    #[pyo3(signature = (batch=10_000))]
    fn maintain_source_turn_backfill(slf: &Bound<'_, Self>, batch: usize) -> PyResult<PyObject> {
        let out = Self::detached(slf, |db| db.maintain_source_turn_backfill(batch))?;
        to_py(slf.py(), serde_json::to_value(out))
    }
}
