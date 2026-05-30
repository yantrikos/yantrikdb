//! Issue #48 follow-up — windowed leak-candidate audit.
//!
//! Background: the trader postmortem (2026-05-29) found 107k `memories`
//! rows with no oplog `record`/`record_with_rid` op. The naive "orphan"
//! definition (`in memories AND not in oplog AND not in
//! replication_apply_log`) treated these as suspicious — but they are a
//! benign **oplog-compaction artifact**: the oplog is a transient,
//! compactable change-log, so locally-originated memories whose oplog
//! rows aged out of the retention window look orphan-shaped while being
//! perfectly healthy. The metric was measuring the wrong thing.
//!
//! The correct, zero-extra-storage detector is **windowed**: only flag
//! memories created within the surviving oplog window
//! (`created_at >= MIN(oplog.timestamp)`) that still lack an oplog op and
//! a `replication_apply_log` entry. Inside the window the oplog row
//! SHOULD still exist, so its absence is a genuine signal (a live
//! write-path bug, or a direct-SQL insert that bypassed the engine).
//! Outside the window, absence is expected compaction — not counted.

use rusqlite::params;

use crate::error::Result;
use crate::types::LeakAuditReport;

use super::YantrikDB;

impl YantrikDB {
    /// Windowed leak-candidate audit (issue #48 follow-up). See module
    /// docs for why this replaces the compaction-confused "orphan" metric.
    ///
    /// `max_rids` caps the returned `candidate_rids` sample; the
    /// `candidate_count` is always exact. A healthy node returns
    /// `candidate_count == 0` (every in-window memory has its oplog op).
    pub fn audit_leak_candidates(&self, max_rids: usize) -> Result<LeakAuditReport> {
        let conn = self.conn();

        // Window floor = the oldest surviving oplog row. Memories created
        // before this have had their oplog rows compacted (expected);
        // memories created at/after this should still have them.
        let window_floor: Option<f64> = conn
            .query_row("SELECT MIN(timestamp) FROM oplog", [], |r| {
                r.get::<_, Option<f64>>(0)
            })
            .unwrap_or(None);

        let Some(floor) = window_floor else {
            // Empty oplog — no window to check within. Can't distinguish
            // compaction from leak; report nothing rather than false-alarm.
            return Ok(LeakAuditReport {
                window_floor: None,
                candidate_count: 0,
                candidate_rids: Vec::new(),
            });
        };

        // Shared predicate: in-window memory with no locally-originated
        // oplog op AND not received via replication.
        const PREDICATE: &str = "m.created_at >= ?1 \
             AND m.rid NOT IN ( \
                 SELECT target_rid FROM oplog \
                 WHERE target_rid IS NOT NULL \
                   AND op_type IN ('record','record_with_rid','consolidate','correct')) \
             AND m.rid NOT IN (SELECT rid FROM replication_apply_log)";

        let count: i64 = conn.query_row(
            &format!("SELECT COUNT(*) FROM memories m WHERE {PREDICATE}"),
            params![floor],
            |r| r.get(0),
        )?;

        let candidate_rids: Vec<String> = {
            let sql = format!(
                "SELECT m.rid FROM memories m WHERE {PREDICATE} \
                 ORDER BY m.created_at DESC LIMIT ?2"
            );
            let mut stmt = conn.prepare(&sql)?;
            let rows =
                stmt.query_map(params![floor, max_rids as i64], |r| r.get::<_, String>(0))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };

        Ok(LeakAuditReport {
            window_floor: Some(floor),
            candidate_count: count as usize,
            candidate_rids,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::YantrikDB;

    fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim).map(|i| (seed + i as f32) * 0.1).collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn healthy_node_has_zero_leak_candidates() {
        // Every memory written via record() has an oplog op, so within the
        // window there should be zero leak candidates.
        let db = YantrikDB::new(":memory:", 8).unwrap();
        for i in 0..10 {
            db.record(
                &format!("mem {i}"),
                "semantic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &vec_seed(i as f32, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        }
        let report = db.audit_leak_candidates(100).unwrap();
        assert!(report.window_floor.is_some());
        assert_eq!(
            report.candidate_count, 0,
            "healthy node: every in-window memory has its oplog op"
        );
        assert!(report.candidate_rids.is_empty());
    }

    #[test]
    fn direct_sql_insert_inside_window_is_flagged() {
        // A row inserted directly into `memories` (bypassing record(), so
        // no oplog op) with created_at inside the window IS a real leak
        // candidate — exactly the write-path-bug / injection case we want.
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let anchor = db
            .record(
                "anchor establishes the window",
                "semantic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &vec_seed(1.0, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        let floor: f64 = db
            .conn()
            .query_row("SELECT MIN(timestamp) FROM oplog", [], |r| r.get(0))
            .unwrap();

        // Inject a memories row with NO oplog op, created_at after the floor.
        db.conn()
            .execute(
                "INSERT INTO memories \
                 (rid, type, text, embedding, created_at, updated_at, importance, \
                  half_life, last_access, valence, metadata, namespace, certainty, \
                  domain, source) \
                 VALUES ('leaked-rid', 'semantic', 'x', NULL, ?1, ?1, 0.5, 604800.0, \
                         ?1, 0.0, '{}', 'default', 0.8, 'general', 'user')",
                params![floor + 1.0],
            )
            .unwrap();

        let report = db.audit_leak_candidates(100).unwrap();
        assert_eq!(
            report.candidate_count, 1,
            "injected in-window row must flag"
        );
        assert!(report.candidate_rids.contains(&"leaked-rid".to_string()));
        // The legitimately-recorded anchor is NOT flagged.
        assert!(!report.candidate_rids.contains(&anchor));
    }

    #[test]
    fn pre_window_orphan_is_not_flagged() {
        // A row with created_at BEFORE the window floor (i.e. its oplog row
        // would have been compacted) must NOT be flagged — that's the
        // benign compaction case the postmortem identified.
        let db = YantrikDB::new(":memory:", 8).unwrap();
        db.record(
            "anchor",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({}),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
        let floor: f64 = db
            .conn()
            .query_row("SELECT MIN(timestamp) FROM oplog", [], |r| r.get(0))
            .unwrap();

        // Inject a row created BEFORE the floor (compaction-aged orphan).
        db.conn()
            .execute(
                "INSERT INTO memories \
                 (rid, type, text, embedding, created_at, updated_at, importance, \
                  half_life, last_access, valence, metadata, namespace, certainty, \
                  domain, source) \
                 VALUES ('aged-rid', 'semantic', 'x', NULL, ?1, ?1, 0.5, 604800.0, \
                         ?1, 0.0, '{}', 'default', 0.8, 'general', 'user')",
                params![floor - 1000.0],
            )
            .unwrap();

        let report = db.audit_leak_candidates(100).unwrap();
        assert!(
            !report.candidate_rids.contains(&"aged-rid".to_string()),
            "pre-window compaction-aged row must NOT be flagged as a leak"
        );
    }
}
