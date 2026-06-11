//! The sleep cycle — autonomous maintenance orchestration (task 24).
//!
//! Epic 6's thesis is "close every loop": every hygiene mechanism the engine
//! has (consolidation, conflict resolution, trigger expiry, importance
//! correction) exists, but closing them was voluntary — and the substrate
//! itself documents that voluntary agent protocols don't survive drift. This
//! is the structural fix: a single call that runs the safe, idempotent passes
//! together, so a host (the MCP server, a cron, the future daemon) can drive
//! ongoing hygiene on a timer with no agent in the loop.
//!
//! The engine deliberately does NOT own a timer thread — a storage engine
//! scheduling itself is the wrong boundary. It exposes
//! [`YantrikDB::run_maintenance_cycle`]; the host decides the cadence (idle
//! trigger, cron, slow heartbeat). The last cycle's summary is persisted so
//! `stats`/the boot digest can show when hygiene last ran and what it did.
//!
//! Per-pass failures are isolated: one pass erroring is recorded in the report
//! and the cycle continues, so a single bad pass never blocks the rest.

use crate::error::Result;
use crate::types::ThinkConfig;
use crate::{
    ConflictBurndownReport, ImportanceRecalibrationReport, RepairReport, SplitReport,
    TriggerPruneReport,
};

use super::{now, YantrikDB};

/// Which passes a maintenance cycle runs. The light, idempotent hygiene passes
/// are on by default; the heavier corpus-rewriting passes (mega-blob split,
/// artifact repair) are opt-in so a routine cycle stays cheap and safe.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MaintenanceCycleConfig {
    /// Run the cognition loop (consolidation, conflict scan, trigger expiry,
    /// pattern mining) via [`YantrikDB::think`].
    pub run_think: bool,
    /// Burn down open conflicts (newer-supersedes; ambiguous → operator).
    pub burn_down_conflicts: bool,
    /// Bound the pending-trigger backlog.
    pub prune_triggers: bool,
    /// Cap for the trigger prune.
    pub max_pending_triggers: usize,
    /// Revert stale, unused, high-importance memories toward baseline.
    pub recalibrate_importance: bool,
    /// Backfill missing memory↔entity links so the knowledge graph keeps
    /// pace with the corpus (continuous extraction; task 42).
    pub backfill_entities: bool,
    /// Auto-relate co-occurring entities to raise edge density (task 44).
    pub auto_relate: bool,
    /// Cap on edges upserted per auto-relate pass.
    pub max_auto_relate_edges: usize,
    /// Split oversized episodic dumps into atomic facts (heavier; opt-in).
    pub split_oversized: bool,
    /// Minimum plaintext length for the split pass.
    pub split_min_chars: usize,
    /// Repair leaked tool-call artifacts in the corpus (one-off; opt-in).
    pub repair_artifacts: bool,
}

impl Default for MaintenanceCycleConfig {
    fn default() -> Self {
        Self {
            run_think: true,
            burn_down_conflicts: true,
            prune_triggers: true,
            max_pending_triggers: 64,
            recalibrate_importance: true,
            backfill_entities: true,
            auto_relate: true,
            max_auto_relate_edges: 500,
            split_oversized: false,
            split_min_chars: 1500,
            repair_artifacts: false,
        }
    }
}

/// Summary of one maintenance cycle. Sub-reports are `Some` only for passes
/// that ran.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct MaintenanceCycleReport {
    pub ran_at: f64,
    /// `think` summary (consolidations, conflicts found, triggers expired).
    pub think_consolidations: Option<usize>,
    pub think_conflicts_found: Option<usize>,
    pub think_triggers_expired: Option<usize>,
    /// memory↔entity links created by the continuous backfill (task 42).
    pub entities_linked: Option<usize>,
    /// co-occurrence edges upserted by auto-relate (task 44).
    pub relations_upserted: Option<usize>,
    pub conflicts: Option<ConflictBurndownReport>,
    pub triggers: Option<TriggerPruneReport>,
    pub importance: Option<ImportanceRecalibrationReport>,
    pub split: Option<SplitReport>,
    pub repair: Option<RepairReport>,
    /// Per-pass errors; a failing pass never aborts the cycle.
    pub errors: Vec<String>,
}

impl YantrikDB {
    /// Run one maintenance cycle — the sleep cycle. Runs the enabled passes in
    /// dependency order (detect via `think`, then resolve/prune/recalibrate),
    /// isolates per-pass failures, persists the summary for `stats`/the boot
    /// digest, and returns it. Idempotent: re-running converges (every pass it
    /// drives is itself idempotent).
    pub fn run_maintenance_cycle(
        &self,
        config: &MaintenanceCycleConfig,
    ) -> Result<MaintenanceCycleReport> {
        let mut report = MaintenanceCycleReport {
            ran_at: now(),
            ..Default::default()
        };

        // Detect first: consolidation + conflict scan + trigger expiry.
        if config.run_think {
            match self.think(&ThinkConfig::default()) {
                Ok(tr) => {
                    report.think_consolidations = Some(tr.consolidation_count);
                    report.think_conflicts_found = Some(tr.conflicts_found);
                    report.think_triggers_expired = Some(tr.expired_triggers);
                }
                Err(e) => report.errors.push(format!("think: {e}")),
            }
        }

        // Keep the knowledge graph in pace with the corpus: backfill any
        // memory↔entity links the at-write (materializer) extraction missed,
        // then refresh the in-memory graph index so recall's expand_entities
        // sees them.
        if config.backfill_entities {
            match self.backfill_memory_entities() {
                Ok(n) => {
                    report.entities_linked = Some(n);
                    if n > 0 {
                        let _ = self.rebuild_graph_index();
                    }
                }
                Err(e) => report.errors.push(format!("entities: {e}")),
            }
        }

        // Raise edge density: relate entities that co-occur in a memory.
        if config.auto_relate {
            match self.auto_relate(false, config.max_auto_relate_edges) {
                Ok(r) => report.relations_upserted = Some(r.edges_upserted),
                Err(e) => report.errors.push(format!("auto_relate: {e}")),
            }
        }

        // Then resolve the conflicts think (and prior writes) surfaced.
        if config.burn_down_conflicts {
            match self.auto_resolve_conflicts(false) {
                Ok(r) => report.conflicts = Some(r),
                Err(e) => report.errors.push(format!("conflicts: {e}")),
            }
        }

        if config.prune_triggers {
            match self.prune_triggers(false, config.max_pending_triggers) {
                Ok(r) => report.triggers = Some(r),
                Err(e) => report.errors.push(format!("triggers: {e}")),
            }
        }

        if config.recalibrate_importance {
            match self.recalibrate_unused_importance(false) {
                Ok(r) => report.importance = Some(r),
                Err(e) => report.errors.push(format!("importance: {e}")),
            }
        }

        if config.split_oversized {
            match self.split_oversized_episodes(false, config.split_min_chars) {
                Ok(r) => report.split = Some(r),
                Err(e) => report.errors.push(format!("split: {e}")),
            }
        }

        if config.repair_artifacts {
            match self.repair_tool_call_artifacts(false) {
                Ok(r) => report.repair = Some(r),
                Err(e) => report.errors.push(format!("repair: {e}")),
            }
        }

        // Persist the last-run summary so stats / the boot digest can show
        // when hygiene last ran and what it did.
        if let Ok(summary) = serde_json::to_string(&report) {
            let conn = self.conn();
            let _ = conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_maintenance_cycle', ?1)",
                rusqlite::params![summary],
            );
        }

        tracing::info!(
            target: "yantrikdb::audit::maintenance",
            ran_at = report.ran_at,
            errors = report.errors.len(),
            "maintenance cycle complete",
        );

        Ok(report)
    }

    /// The last persisted maintenance-cycle summary (JSON), or `None` if no
    /// cycle has run. For `stats` / the boot digest.
    pub fn last_maintenance_cycle(&self) -> Result<Option<String>> {
        let conn = self.conn();
        match conn.query_row(
            "SELECT value FROM meta WHERE key = 'last_maintenance_cycle'",
            [],
            |r| r.get::<_, String>(0),
        ) {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}
