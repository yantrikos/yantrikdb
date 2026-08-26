use std::collections::HashMap;

use crate::error::Result;
use crate::types::ScoringRow;

use super::YantrikDB;

impl YantrikDB {
    /// Columns that exist on `table` right now.
    ///
    /// Needed because this loader runs against two very different databases:
    /// the host's (always migrated to the current schema before use) and a
    /// **mounted pack's** (opened read-only from a file some other engine
    /// sealed, and never migratable — see `pack::mount_pack_opts`).
    fn existing_columns(
        conn: &rusqlite::Connection,
        table: &str,
    ) -> Result<std::collections::HashSet<String>> {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
        let mut out = std::collections::HashSet::new();
        for row in rows {
            out.insert(row?);
        }
        Ok(out)
    }

    /// Load scoring-relevant fields for all non-tombstoned memories into a HashMap.
    ///
    /// **The projection is schema-aware, and has to be.** A mounted pack
    /// carries whatever schema its publisher's engine wrote and can never be
    /// migrated (it is opened `query_only`, and rewriting a signed publisher
    /// artifact is not an option). Columns added after that seal are therefore
    /// legitimately absent: the v41→v42 synthesis triplet is missing from every
    /// pack published by an engine at or below 0.15.x. A fixed projection made
    /// all of them fail to mount with `no such column: synthesis_state` — a
    /// break the host path could never surface, because a host database is
    /// always migrated before this runs. Absent columns read as NULL, which is
    /// exactly what v42 migrates existing host rows to.
    pub(crate) fn load_scoring_cache(
        conn: &rusqlite::Connection,
    ) -> Result<HashMap<String, ScoringRow>> {
        let present = Self::existing_columns(conn, "memories")?;
        let project = |name: &str| -> String {
            if present.contains(name) {
                name.to_string()
            } else {
                format!("NULL AS {name}")
            }
        };
        let sql = format!(
            "SELECT rid, created_at, importance, half_life, last_access, \
             valence, consolidation_status, type, namespace, access_count, \
             certainty, domain, source, emotional_state, {}, {}, {} \
             FROM memories \
             WHERE consolidation_status != 'tombstoned'",
            project("synthesis_state"),
            project("synthesis_axis"),
            project("synthesis_granularity"),
        );
        let mut stmt = conn.prepare(&sql)?;

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                ScoringRow {
                    created_at: row.get(1)?,
                    importance: row.get(2)?,
                    half_life: row.get(3)?,
                    last_access: row.get(4)?,
                    valence: row.get(5)?,
                    consolidation_status: row.get(6)?,
                    memory_type: row.get(7)?,
                    namespace: row.get(8)?,
                    access_count: row.get::<_, i64>(9)? as u32,
                    certainty: row.get(10)?,
                    domain: row.get(11)?,
                    source: row.get(12)?,
                    emotional_state: row.get(13)?,
                    synthesis_state: row.get(14)?,
                    synthesis_axis: row.get(15)?,
                    synthesis_granularity: row.get(16)?,
                },
            ))
        })?;

        let mut cache = HashMap::new();
        for row in rows {
            let (rid, scoring_row) = row?;
            cache.insert(rid, scoring_row);
        }
        Ok(cache)
    }

    /// Insert a scoring row into the in-memory cache.
    pub fn cache_insert(&self, rid: String, row: ScoringRow) {
        self.scoring_cache.write().insert(rid, row);
    }

    /// Remove a scoring row from the in-memory cache.
    pub fn cache_remove(&self, rid: &str) {
        self.scoring_cache.write().remove(rid);
    }

    pub(crate) fn cache_invalidate_syntheses(&self, rids: &[String]) {
        let mut cache = self.scoring_cache.write();
        for rid in rids {
            if let Some(row) = cache.get_mut(rid) {
                row.synthesis_state = Some("invalidated".to_string());
            }
        }
    }

    pub(crate) fn cache_verify_syntheses(&self, rids: &[String]) {
        let mut cache = self.scoring_cache.write();
        for rid in rids {
            if let Some(row) = cache.get_mut(rid) {
                row.synthesis_state = Some("verified".to_string());
            }
        }
    }

    pub(crate) fn cache_supersede_syntheses(&self, rids: &[String]) {
        let mut cache = self.scoring_cache.write();
        for rid in rids {
            if let Some(row) = cache.get_mut(rid) {
                row.synthesis_state = Some("superseded".to_string());
            }
        }
    }

    /// Mark a memory as consolidated in the cache and reduce its importance.
    pub fn cache_mark_consolidated(&self, rid: &str, importance_factor: f64) {
        let mut cache = self.scoring_cache.write();
        if let Some(row) = cache.get_mut(rid) {
            row.consolidation_status = "consolidated".to_string();
            row.importance *= importance_factor;
        }
    }
}
