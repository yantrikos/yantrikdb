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
use crate::types::{LeakAuditReport, SynthesisEvidenceAuditIssue, SynthesisEvidenceAuditReport};

use super::YantrikDB;

const INVALID_SYNTHESIS_EVIDENCE: &str = "s.synthesis_state = 'verified' \
    AND s.consolidation_status = 'active' \
    AND ( \
        NOT EXISTS (SELECT 1 FROM synthesis_dependencies d \
                    WHERE d.synthesis_rid = s.rid) \
        OR NOT EXISTS (SELECT 1 FROM synthesis_dependencies d \
                       WHERE d.synthesis_rid = s.rid AND d.is_direct = 1) \
        OR NOT EXISTS (SELECT 1 FROM synthesis_dependencies d \
                       JOIN memories leaf ON leaf.rid = d.source_rid \
                       WHERE d.synthesis_rid = s.rid \
                         AND leaf.synthesis_state IS NULL) \
        OR EXISTS ( \
            SELECT 1 FROM synthesis_dependencies d \
            LEFT JOIN memories source ON source.rid = d.source_rid \
            WHERE d.synthesis_rid = s.rid \
              AND (d.namespace <> s.namespace \
                   OR source.rid IS NULL \
                   OR source.namespace <> s.namespace \
                   OR source.consolidation_status <> 'active' \
                   OR (source.synthesis_state IS NOT NULL \
                       AND source.synthesis_state <> 'verified') \
                   OR d.source_revision_num <> COALESCE(( \
                       SELECT MAX(r.revision_num) FROM record_revisions r \
                       WHERE r.rid = d.source_rid), 0)) \
        ) \
        OR EXISTS ( \
            WITH RECURSIVE reachable(rid) AS ( \
                SELECT d.source_rid FROM synthesis_dependencies d \
                WHERE d.synthesis_rid = s.rid \
                UNION \
                SELECT d.source_rid FROM synthesis_dependencies d \
                JOIN reachable prior ON d.synthesis_rid = prior.rid \
            ) \
            SELECT 1 FROM reachable WHERE rid = s.rid \
        ) \
        OR (s.synthesis_logical_key IS NOT NULL AND EXISTS ( \
            SELECT 1 FROM memories duplicate \
            WHERE duplicate.rid <> s.rid \
              AND duplicate.namespace = s.namespace \
              AND duplicate.synthesis_logical_key = s.synthesis_logical_key \
              AND duplicate.synthesis_state = 'verified' \
              AND duplicate.consolidation_status = 'active' \
        )) \
    )";

impl YantrikDB {
    /// Audit active verified syntheses against the same evidence invariants
    /// enforced at admission. This is deliberately report-only: an unexpected
    /// mismatch may indicate direct SQL damage or an old write-path bug, and
    /// automatic mutation would need its own replicated repair operation.
    ///
    /// The counts are exact. `max_issues` only bounds the diagnostic sample.
    pub fn audit_synthesis_evidence(
        &self,
        namespace: Option<&str>,
        max_issues: usize,
    ) -> Result<SynthesisEvidenceAuditReport> {
        let conn = self.conn();
        let verified_active_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories s \
             WHERE s.synthesis_state = 'verified' \
               AND s.consolidation_status = 'active' \
               AND (?1 IS NULL OR s.namespace = ?1)",
            params![namespace],
            |row| row.get(0),
        )?;
        let candidate_sql = format!(
            "SELECT COUNT(*) FROM memories s \
             WHERE (?1 IS NULL OR s.namespace = ?1) AND ({INVALID_SYNTHESIS_EVIDENCE})"
        );
        let candidate_count: i64 =
            conn.query_row(&candidate_sql, params![namespace], |row| row.get(0))?;
        let orphan_dependency_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM synthesis_dependencies d \
             LEFT JOIN memories synthesis ON synthesis.rid = d.synthesis_rid \
             WHERE synthesis.rid IS NULL \
               AND (?1 IS NULL OR d.namespace = ?1)",
            params![namespace],
            |row| row.get(0),
        )?;
        let synthesis_fanout_cap = Self::synthesis_fanout_cap_from_conn(&conn)?;
        let sources_over_fanout_cap: i64 = conn.query_row(
            "WITH fanout AS ( \
                 SELECT d.source_rid, COUNT(DISTINCT d.synthesis_rid) AS n \
                 FROM synthesis_dependencies d \
                 JOIN memories synthesis ON synthesis.rid = d.synthesis_rid \
                 WHERE synthesis.synthesis_state = 'verified' \
                   AND synthesis.consolidation_status = 'active' \
                   AND (?1 IS NULL OR d.namespace = ?1) \
                 GROUP BY d.source_rid \
             ) \
             SELECT COUNT(*) FROM fanout WHERE n > ?2",
            params![namespace, synthesis_fanout_cap as i64],
            |row| row.get(0),
        )?;
        let dependency_cycle_count: i64 = conn.query_row(
            "WITH RECURSIVE paths(root_rid, source_rid) AS ( \
                 SELECT d.synthesis_rid, d.source_rid \
                 FROM synthesis_dependencies d \
                 UNION \
                 SELECT paths.root_rid, d.source_rid \
                 FROM paths \
                 JOIN synthesis_dependencies d \
                   ON d.synthesis_rid = paths.source_rid \
             ), cycle_roots AS ( \
                 SELECT DISTINCT root_rid FROM paths WHERE root_rid = source_rid \
             ) \
             SELECT COUNT(*) FROM cycle_roots cycle \
             JOIN memories synthesis ON synthesis.rid = cycle.root_rid \
             WHERE synthesis.synthesis_state = 'verified' \
               AND synthesis.consolidation_status = 'active' \
               AND (?1 IS NULL OR synthesis.namespace = ?1)",
            params![namespace],
            |row| row.get(0),
        )?;
        let duplicate_logical_key_group_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM ( \
                 SELECT s.namespace, s.synthesis_logical_key \
                 FROM memories s \
                 WHERE s.synthesis_state = 'verified' \
                   AND s.consolidation_status = 'active' \
                   AND s.synthesis_logical_key IS NOT NULL \
                   AND (?1 IS NULL OR s.namespace = ?1) \
                 GROUP BY s.namespace, s.synthesis_logical_key \
                 HAVING COUNT(*) > 1 \
             )",
            params![namespace],
            |row| row.get(0),
        )?;

        let sample_sql = format!(
            "SELECT s.rid, s.namespace FROM memories s \
             WHERE (?1 IS NULL OR s.namespace = ?1) AND ({INVALID_SYNTHESIS_EVIDENCE}) \
             ORDER BY s.namespace, s.created_at, s.rid LIMIT ?2"
        );
        let sample_limit = max_issues.min(i64::MAX as usize) as i64;
        let sampled: Vec<(String, String)> = {
            let mut stmt = conn.prepare(&sample_sql)?;
            let rows = stmt.query_map(params![namespace, sample_limit], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })?;
            rows.collect::<std::result::Result<_, _>>()?
        };

        let mut issues = Vec::with_capacity(sampled.len());
        for (synthesis_rid, synthesis_namespace) in sampled {
            let dependencies: Vec<(
                String,
                String,
                bool,
                Option<String>,
                Option<String>,
                Option<String>,
                Option<String>,
                i64,
                i64,
            )> = {
                let mut stmt = conn.prepare(
                    "SELECT d.source_rid, d.namespace, d.is_direct, \
                            source.rid, source.namespace, \
                            source.consolidation_status, source.synthesis_state, \
                            d.source_revision_num, \
                            COALESCE((SELECT MAX(r.revision_num) \
                                      FROM record_revisions r \
                                      WHERE r.rid = d.source_rid), 0) \
                     FROM synthesis_dependencies d \
                     LEFT JOIN memories source ON source.rid = d.source_rid \
                     WHERE d.synthesis_rid = ?1 \
                     ORDER BY d.source_rid",
                )?;
                let rows = stmt.query_map(params![synthesis_rid], |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get::<_, i64>(2)? != 0,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                        row.get(7)?,
                        row.get(8)?,
                    ))
                })?;
                rows.collect::<std::result::Result<_, _>>()?
            };

            let mut reasons = std::collections::BTreeSet::new();
            let has_cycle: i64 = conn.query_row(
                "WITH RECURSIVE reachable(rid) AS ( \
                     SELECT d.source_rid FROM synthesis_dependencies d \
                     WHERE d.synthesis_rid = ?1 \
                     UNION \
                     SELECT d.source_rid FROM synthesis_dependencies d \
                     JOIN reachable prior ON d.synthesis_rid = prior.rid \
                 ) \
                 SELECT EXISTS(SELECT 1 FROM reachable WHERE rid = ?1)",
                params![synthesis_rid],
                |row| row.get(0),
            )?;
            if has_cycle != 0 {
                reasons.insert("dependency_cycle".to_string());
            }
            let has_duplicate_generation: i64 = conn.query_row(
                "SELECT EXISTS( \
                     SELECT 1 FROM memories current \
                     JOIN memories duplicate \
                       ON duplicate.namespace = current.namespace \
                      AND duplicate.synthesis_logical_key = current.synthesis_logical_key \
                      AND duplicate.rid <> current.rid \
                     WHERE current.rid = ?1 \
                       AND current.synthesis_logical_key IS NOT NULL \
                       AND duplicate.synthesis_state = 'verified' \
                       AND duplicate.consolidation_status = 'active' \
                 )",
                params![synthesis_rid],
                |row| row.get(0),
            )?;
            if has_duplicate_generation != 0 {
                reasons.insert("duplicate_active_logical_key".to_string());
            }
            if dependencies.is_empty() {
                reasons.insert("missing_dependencies".to_string());
            }
            if !dependencies.iter().any(|dependency| dependency.2) {
                reasons.insert("missing_direct_dependency".to_string());
            }
            if !dependencies
                .iter()
                .any(|dependency| dependency.3.is_some() && dependency.6.is_none())
            {
                reasons.insert("missing_raw_leaf_dependency".to_string());
            }
            for (
                source_rid,
                dependency_namespace,
                _,
                source_exists,
                source_namespace,
                source_status,
                source_synthesis_state,
                expected_revision,
                actual_revision,
            ) in dependencies
            {
                if dependency_namespace != synthesis_namespace {
                    reasons.insert(format!("dependency_namespace_mismatch:{source_rid}"));
                }
                if source_exists.is_none() {
                    reasons.insert(format!("missing_source:{source_rid}"));
                    continue;
                }
                if source_namespace.as_deref() != Some(synthesis_namespace.as_str()) {
                    reasons.insert(format!("source_namespace_mismatch:{source_rid}"));
                }
                if source_status.as_deref() != Some("active") {
                    reasons.insert(format!("source_not_active:{source_rid}"));
                }
                if source_synthesis_state
                    .as_deref()
                    .is_some_and(|state| state != "verified")
                {
                    reasons.insert(format!("source_synthesis_not_verified:{source_rid}"));
                }
                if expected_revision != actual_revision {
                    reasons.insert(format!(
                        "revision_mismatch:{source_rid}:{expected_revision}:{actual_revision}"
                    ));
                }
            }
            issues.push(SynthesisEvidenceAuditIssue {
                synthesis_rid,
                reasons: reasons.into_iter().collect(),
            });
        }

        Ok(SynthesisEvidenceAuditReport {
            verified_active_count: verified_active_count.max(0) as usize,
            candidate_count: candidate_count.max(0) as usize,
            orphan_dependency_count: orphan_dependency_count.max(0) as usize,
            sources_over_fanout_cap: sources_over_fanout_cap.max(0) as usize,
            dependency_cycle_count: dependency_cycle_count.max(0) as usize,
            duplicate_logical_key_group_count: duplicate_logical_key_group_count.max(0) as usize,
            issues,
        })
    }

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

    fn source(db: &YantrikDB, text: &str) -> String {
        db.record(
            text,
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
        .unwrap()
    }

    fn synthesis(db: &YantrikDB, source_rid: &str, logical_key: &str) -> String {
        crate::consolidate::record_synthesis(
            db,
            &[source_rid.to_string()],
            logical_key,
            Some(&vec_seed(2.0, 8)),
            "asked",
            "atomic",
            &serde_json::json!({}),
            logical_key,
        )
        .unwrap()["consolidated_rid"]
            .as_str()
            .unwrap()
            .to_string()
    }

    fn reasons(db: &YantrikDB) -> Vec<String> {
        let report = db.audit_synthesis_evidence(None, 10).unwrap();
        assert_eq!(report.candidate_count, 1);
        report.issues[0].reasons.clone()
    }

    #[test]
    fn synthesis_evidence_audit_detects_revision_and_shape_drift_without_mutation() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let source_rid = db
            .record(
                "source evidence",
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
        let synthesis = crate::consolidate::record_synthesis(
            &db,
            std::slice::from_ref(&source_rid),
            "grounded item",
            Some(&vec_seed(2.0, 8)),
            "asked",
            "atomic",
            &serde_json::json!({}),
            "synth:audit:item-1",
        )
        .unwrap();
        let synthesis_rid = synthesis["consolidated_rid"].as_str().unwrap();

        let healthy = db.audit_synthesis_evidence(None, 100).unwrap();
        assert_eq!(healthy.verified_active_count, 1);
        assert_eq!(healthy.candidate_count, 0);
        assert!(healthy.issues.is_empty());

        db.conn()
            .execute(
                "UPDATE synthesis_dependencies SET source_revision_num = 9 \
                 WHERE synthesis_rid = ?1 AND source_rid = ?2",
                params![synthesis_rid, source_rid],
            )
            .unwrap();
        let revision_drift = db.audit_synthesis_evidence(Some("default"), 100).unwrap();
        assert_eq!(revision_drift.candidate_count, 1);
        assert!(revision_drift.issues[0]
            .reasons
            .iter()
            .any(|reason| reason.starts_with("revision_mismatch:")));
        assert_eq!(
            db.conn()
                .query_row(
                    "SELECT synthesis_state FROM memories WHERE rid = ?1",
                    params![synthesis_rid],
                    |row| row.get::<_, String>(0),
                )
                .unwrap(),
            "verified",
            "the audit must remain report-only"
        );

        db.conn()
            .execute(
                "DELETE FROM synthesis_dependencies WHERE synthesis_rid = ?1",
                params![synthesis_rid],
            )
            .unwrap();
        let bounded = db.audit_synthesis_evidence(None, 0).unwrap();
        assert_eq!(bounded.candidate_count, 1);
        assert!(bounded.issues.is_empty());
        let missing = db.audit_synthesis_evidence(None, 1).unwrap();
        assert_eq!(
            missing.issues[0].reasons,
            vec![
                "missing_dependencies",
                "missing_direct_dependency",
                "missing_raw_leaf_dependency",
            ]
        );
        assert_eq!(
            db.audit_synthesis_evidence(Some("other"), 100)
                .unwrap()
                .candidate_count,
            0
        );
    }

    #[test]
    fn synthesis_evidence_audit_classifies_each_source_violation() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let source_rid = source(&db, "source health");
        let synthesis_rid = synthesis(&db, &source_rid, "synth:audit:source-health");

        db.conn()
            .execute(
                "UPDATE memories SET consolidation_status = 'tombstoned' WHERE rid = ?1",
                params![source_rid],
            )
            .unwrap();
        assert!(reasons(&db)
            .iter()
            .any(|reason| reason.starts_with("source_not_active:")));

        db.conn()
            .execute(
                "UPDATE memories SET consolidation_status = 'active', namespace = 'other' \
                 WHERE rid = ?1",
                params![source_rid],
            )
            .unwrap();
        assert!(reasons(&db)
            .iter()
            .any(|reason| reason.starts_with("source_namespace_mismatch:")));

        db.conn()
            .execute(
                "UPDATE memories SET namespace = 'default', synthesis_state = 'invalidated' \
                 WHERE rid = ?1",
                params![source_rid],
            )
            .unwrap();
        assert!(reasons(&db)
            .iter()
            .any(|reason| reason.starts_with("source_synthesis_not_verified:")));

        db.conn()
            .execute(
                "UPDATE memories SET synthesis_state = NULL WHERE rid = ?1",
                params![source_rid],
            )
            .unwrap();
        db.conn()
            .execute(
                "UPDATE synthesis_dependencies SET namespace = 'other' \
                 WHERE synthesis_rid = ?1",
                params![synthesis_rid],
            )
            .unwrap();
        assert!(reasons(&db)
            .iter()
            .any(|reason| reason.starts_with("dependency_namespace_mismatch:")));

        db.conn()
            .execute("DELETE FROM memories WHERE rid = ?1", params![source_rid])
            .unwrap();
        let missing = reasons(&db);
        assert!(missing
            .iter()
            .any(|reason| reason.starts_with("missing_source:")));
        assert!(missing.contains(&"missing_raw_leaf_dependency".to_string()));
    }

    #[test]
    fn synthesis_evidence_audit_isolates_direct_and_raw_leaf_shape_gaps() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let source_rid = source(&db, "shape source");
        let synthesis_rid = synthesis(&db, &source_rid, "synth:audit:shape");

        db.conn()
            .execute(
                "UPDATE synthesis_dependencies SET is_direct = 0 \
                 WHERE synthesis_rid = ?1",
                params![synthesis_rid],
            )
            .unwrap();
        assert_eq!(reasons(&db), vec!["missing_direct_dependency"]);

        db.conn()
            .execute(
                "UPDATE synthesis_dependencies SET is_direct = 1 \
                 WHERE synthesis_rid = ?1",
                params![synthesis_rid],
            )
            .unwrap();
        let parent_rid = synthesis(&db, &synthesis_rid, "synth:audit:shape-parent");
        db.conn()
            .execute(
                "DELETE FROM synthesis_dependencies \
                 WHERE synthesis_rid = ?1 AND source_rid = ?2",
                params![parent_rid, source_rid],
            )
            .unwrap();
        assert_eq!(reasons(&db), vec!["missing_raw_leaf_dependency"]);
    }

    #[test]
    fn synthesis_evidence_audit_reports_orphans_and_replication_style_over_cap() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        db.set_synthesis_fanout_cap(2).unwrap();
        let source_rid = source(&db, "central evidence");
        let first = synthesis(&db, &source_rid, "synth:audit:fanout-1");
        synthesis(&db, &source_rid, "synth:audit:fanout-2");
        db.set_synthesis_fanout_cap(1).unwrap();

        let over_cap = db.audit_synthesis_evidence(None, 10).unwrap();
        assert_eq!(over_cap.sources_over_fanout_cap, 1);
        assert_eq!(over_cap.candidate_count, 0);

        db.conn()
            .execute("DELETE FROM memories WHERE rid = ?1", params![first])
            .unwrap();
        let orphan = db.audit_synthesis_evidence(None, 10).unwrap();
        assert_eq!(orphan.orphan_dependency_count, 1);
        assert_eq!(orphan.sources_over_fanout_cap, 0);
    }

    #[test]
    fn synthesis_evidence_audit_detects_cycles_and_duplicate_generations() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let source_a = source(&db, "cycle source a");
        let source_b = source(&db, "cycle source b");
        let synthesis_a = synthesis(&db, &source_a, "synth:audit:cycle-a");
        let synthesis_b = synthesis(&db, &source_b, "synth:audit:cycle-b");

        db.conn()
            .execute(
                "INSERT INTO synthesis_dependencies \
                 (synthesis_rid, source_rid, source_revision_num, namespace, is_direct) \
                 VALUES (?1, ?2, 0, 'default', 0), (?2, ?1, 0, 'default', 0)",
                params![synthesis_a, synthesis_b],
            )
            .unwrap();
        let cycles = db.audit_synthesis_evidence(None, 10).unwrap();
        assert_eq!(cycles.dependency_cycle_count, 2);
        assert_eq!(cycles.candidate_count, 2);
        assert!(cycles
            .issues
            .iter()
            .all(|issue| issue.reasons.contains(&"dependency_cycle".to_string())));

        db.conn()
            .execute(
                "DELETE FROM synthesis_dependencies \
                 WHERE (synthesis_rid = ?1 AND source_rid = ?2) \
                    OR (synthesis_rid = ?2 AND source_rid = ?1)",
                params![synthesis_a, synthesis_b],
            )
            .unwrap();
        db.conn()
            .execute(
                "UPDATE memories SET synthesis_logical_key = 'synth:audit:duplicate' \
                 WHERE rid IN (?1, ?2)",
                params![synthesis_a, synthesis_b],
            )
            .unwrap();
        let duplicates = db.audit_synthesis_evidence(None, 10).unwrap();
        assert_eq!(duplicates.dependency_cycle_count, 0);
        assert_eq!(duplicates.duplicate_logical_key_group_count, 1);
        assert_eq!(duplicates.candidate_count, 2);
        assert!(duplicates.issues.iter().all(|issue| issue
            .reasons
            .contains(&"duplicate_active_logical_key".to_string())));
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
