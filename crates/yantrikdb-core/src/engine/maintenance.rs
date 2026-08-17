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
    /// Preview mode: dry-capable passes run dry; passes with no dry form
    /// (think's consolidation, entity backfill, split, repair) are SKIPPED
    /// rather than quietly run wet, and the summary is NOT persisted as the
    /// last cycle. Added 2026-08-15 after the MCP layer accepted and
    /// documented a dry_run parameter no layer below implemented — a "dry"
    /// call auto-resolved 15 conflicts and tombstoned 13 live records on a
    /// production store.
    pub dry_run: bool,
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
            dry_run: false,
        }
    }
}

/// The cognitive compactor's ledger (v0.15.x).
///
/// The core is a passive library — it cannot schedule maintenance, but it
/// must always be able to ANSWER "how overdue is maintenance?". A reactive
/// deployment (the MCP server) surfaces this to the calling LLM, which acts
/// as the scheduler. Four cheap numbers, one call:
///
/// - `writes_since_think`: memory writes committed since cognition last
///   completed a pass — new material no conflict scan has seen. Incremented
///   atomically with each origin content write (record / record_text /
///   record_batch per item / correct / origin record_with_rid); NOT moved by
///   access-pattern ops (reinforce, feedback, relate, archive, forget) or by
///   replication apply (a follower's imports get thought about on the
///   leader).
/// - `last_think_at`: when cognition last completed a pass (`think` with its
///   conflict scan, or a non-dry `run_maintenance_cycle`). `None` = never.
/// - `open_conflicts` / `pending_triggers`: the backlog cognition has
///   surfaced but nobody has resolved.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize)]
pub struct MaintenanceDebt {
    pub writes_since_think: u64,
    pub last_think_at: Option<f64>,
    pub open_conflicts: u64,
    pub pending_triggers: u64,
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
    /// Debt-ledger increment: `meta.writes_since_think += n`.
    ///
    /// MUST be called on the SAME connection/transaction as the write it
    /// counts, inside the write's transaction wherever one exists — the
    /// counter is then atomic with the row it counts, so a rollback (or an
    /// idempotent hit that never reaches the tx) leaves the ledger untouched
    /// and the count can never drift from the corpus. Same discipline as
    /// `advance_importance_stats_in_tx`, and it sits next to that call at
    /// every site.
    ///
    /// Associated fn (no `&self`) precisely so a call site holding the conn
    /// lock cannot accidentally re-lock it.
    pub(crate) fn bump_writes_since_think_on(conn: &rusqlite::Connection, n: u64) -> Result<()> {
        // Stored as TEXT like every meta value; CAST round-trips it. A
        // missing key starts the ledger at n; a non-numeric value (never
        // written by the engine) CASTs to 0 rather than erroring.
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('writes_since_think', CAST(?1 AS TEXT)) \
             ON CONFLICT(key) DO UPDATE SET \
                 value = CAST(CAST(value AS INTEGER) + ?1 AS TEXT)",
            rusqlite::params![n as i64],
        )?;
        Ok(())
    }

    /// Debt-ledger reset: cognition completed a pass over the corpus — stamp
    /// `last_think_at` and zero `writes_since_think`, atomically on the
    /// caller's connection. Called from `think()` when its conflict scan ran
    /// and from a non-dry `run_maintenance_cycle` at completion. A dry run
    /// must NEVER reach this (the 0.15.0 dry-run contract: a preview clears
    /// nothing).
    pub(crate) fn clear_maintenance_debt_on(conn: &rusqlite::Connection, ts: f64) -> Result<()> {
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_think_at', ?1)",
            rusqlite::params![ts.to_string()],
        )?;
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('writes_since_think', '0')",
            [],
        )?;
        Ok(())
    }

    /// How overdue is maintenance? See [`MaintenanceDebt`].
    ///
    /// Read-only, served from the read pool, and it never fails the caller:
    /// missing meta keys read as zero/`None`, and each COUNT is best-effort
    /// (a schema too old to have the table reads as zero rather than
    /// erroring). This is the one call a reactive host must always be able
    /// to make, so it degrades to zeros instead of propagating errors.
    pub fn maintenance_debt(&self) -> MaintenanceDebt {
        let conn = self.read_conn();
        let meta_str = |key: &str| -> Option<String> {
            conn.query_row(
                "SELECT value FROM meta WHERE key = ?1",
                rusqlite::params![key],
                |r| r.get::<_, String>(0),
            )
            .ok()
        };
        let writes_since_think = meta_str("writes_since_think")
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(0);
        let last_think_at = meta_str("last_think_at").and_then(|v| v.trim().parse::<f64>().ok());
        let count = |sql: &str| -> u64 {
            conn.query_row(sql, [], |r| r.get::<_, i64>(0))
                .map(|n| n.max(0) as u64)
                .unwrap_or(0)
        };
        // The same predicates stats() and the boot digest use — one
        // definition of "open" and "pending" across every surface.
        let open_conflicts = count("SELECT COUNT(*) FROM conflicts WHERE status = 'open'");
        let pending_triggers = count("SELECT COUNT(*) FROM trigger_log WHERE status = 'pending'");
        MaintenanceDebt {
            writes_since_think,
            last_think_at,
            open_conflicts,
            pending_triggers,
        }
    }

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
        // No dry form exists for think (consolidation writes, conflict scan
        // writes conflict rows), so in preview it is skipped outright.
        if config.run_think && !config.dry_run {
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
        if config.backfill_entities && !config.dry_run {
            match self.backfill_memory_entities() {
                Ok(n) => {
                    report.entities_linked = Some(n);
                    if n > 0 {
                        // Entities were just committed; a failed index
                        // rebuild means recall serves a stale graph while
                        // the report shows entities_linked = n. The errors
                        // vec exists for exactly this.
                        if let Err(e) = self.rebuild_graph_index() {
                            report.errors.push(format!("graph_index: {e}"));
                        }
                    }
                }
                Err(e) => report.errors.push(format!("entities: {e}")),
            }
        }

        // Raise edge density: relate entities that co-occur in a memory.
        if config.auto_relate {
            match self.auto_relate(config.dry_run, config.max_auto_relate_edges) {
                Ok(r) => report.relations_upserted = Some(r.edges_upserted),
                Err(e) => report.errors.push(format!("auto_relate: {e}")),
            }
        }

        // Then resolve the conflicts think (and prior writes) surfaced.
        if config.burn_down_conflicts {
            match self.auto_resolve_conflicts(config.dry_run) {
                Ok(r) => report.conflicts = Some(r),
                Err(e) => report.errors.push(format!("conflicts: {e}")),
            }
        }

        if config.prune_triggers {
            match self.prune_triggers(config.dry_run, config.max_pending_triggers) {
                Ok(r) => report.triggers = Some(r),
                Err(e) => report.errors.push(format!("triggers: {e}")),
            }
        }

        if config.recalibrate_importance {
            match self.recalibrate_unused_importance(config.dry_run) {
                Ok(r) => report.importance = Some(r),
                Err(e) => report.errors.push(format!("importance: {e}")),
            }
        }

        if config.split_oversized {
            match self.split_oversized_episodes(config.dry_run, config.split_min_chars) {
                Ok(r) => report.split = Some(r),
                Err(e) => report.errors.push(format!("split: {e}")),
            }
        }

        if config.repair_artifacts {
            match self.repair_tool_call_artifacts(config.dry_run) {
                Ok(r) => report.repair = Some(r),
                Err(e) => report.errors.push(format!("repair: {e}")),
            }
        }

        // Completion of a REAL cycle settles the debt ledger: stamp
        // last_think_at and zero writes_since_think. A dry run must not —
        // a preview that cleared debt would tell the scheduling host the
        // corpus was thought about when nothing looked at it (the same
        // masquerade the persist guard below exists for). If the think pass
        // above ran, its conflict scan already cleared the ledger; this
        // re-stamp is idempotent and also covers cycles configured with
        // run_think = false — the cycle's other hygiene passes still
        // constitute a completed pass over the corpus. Runs BEFORE the
        // summary persist so a failure here lands in the persisted report.
        if !config.dry_run {
            let conn = self.conn();
            if let Err(e) = Self::clear_maintenance_debt_on(&conn, now()) {
                report.errors.push(format!("debt_ledger: {e}"));
            }
        }

        // Persist the last-run summary so stats / the boot digest can show
        // when hygiene last ran and what it did. A preview is not a cycle:
        // persisting it would let a dry call masquerade as real hygiene.
        if !config.dry_run {
            if let Ok(summary) = serde_json::to_string(&report) {
                let conn = self.conn();
                let _ = conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_maintenance_cycle', ?1)",
                    rusqlite::params![summary],
                );
            }
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
