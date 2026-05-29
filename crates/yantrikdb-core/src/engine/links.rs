//! Issue #48 — first-class record-to-record links (schema v31, 0.7.x series).
//!
//! Record links live in their own `record_links` table, distinct from the
//! entity graph (`claims`), so rids don't pollute the entity classifier
//! (see RFC §"Considered alternatives"). This module owns the write +
//! traversal API: [`YantrikDB::record_with_links`],
//! [`YantrikDB::link`], [`YantrikDB::unlink`],
//! [`YantrikDB::linked_records`].
//!
//! **Atomicity boundary (honest).** The engine's `record()` is decoupled
//! (oplog → materializer), so there is no single SQLite transaction that
//! spans "the memories row + the links." `record_with_links` therefore
//! commits the record first (durable via the oplog), then inserts each
//! link via [`YantrikDB::link`], which is itself durable + idempotent.
//! The only non-atomic window is "record committed, a subsequent link
//! insert failed" — recoverable by re-calling `link()` (idempotent on the
//! UNIQUE(source_rid, target_rid, link_type) constraint). This is the
//! same shape as the rest of the decoupled write path and is documented
//! rather than overclaimed.
//!
//! **Replication.** Each link emits a standalone `link` oplog op (and
//! `unlink` emits `unlink`). They replicate independently and apply
//! idempotently via `INSERT OR IGNORE`. This is simpler than threading a
//! links-array through the `record` op payload and is equally correct.

use rusqlite::params;

use crate::error::{Result, YantrikDbError};
use crate::types::{LinkDirection, LinkType, LinkedRecord, RecordLink};

use super::{now, YantrikDB};

impl YantrikDB {
    /// Record a memory and atomically(-ish; see module docs) attach
    /// record-to-record links. `record()`'s signature is intentionally
    /// left unchanged (100+ call sites); this is the link-aware entry
    /// point. Callers with no links should just use `record()`.
    #[allow(clippy::too_many_arguments)]
    pub fn record_with_links(
        &self,
        text: &str,
        memory_type: &str,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: &serde_json::Value,
        embedding: &[f32],
        namespace: &str,
        certainty: f64,
        domain: &str,
        source: &str,
        emotional_state: Option<&str>,
        links: &[RecordLink],
    ) -> Result<String> {
        let rid = self.record(
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
        )?;
        for link in links {
            self.link(&rid, link)?;
        }
        Ok(rid)
    }

    /// Add a single record-to-record link from `source_rid`.
    ///
    /// Validates: `source_rid` non-empty, `target_rid` non-empty, and
    /// `source_rid != target_rid` (a record cannot link to itself).
    /// Idempotent on `UNIQUE(source_rid, target_rid, link_type)` via
    /// `INSERT OR IGNORE`. Returns the `link_id` (freshly minted even if
    /// the row already existed and the insert was ignored).
    pub fn link(&self, source_rid: &str, link: &RecordLink) -> Result<String> {
        if source_rid.is_empty() {
            return Err(YantrikDbError::InvalidInput(
                "link: source_rid must be non-empty".to_string(),
            ));
        }
        if link.target_rid.is_empty() {
            return Err(YantrikDbError::InvalidInput(
                "link: target_rid must be non-empty".to_string(),
            ));
        }
        if source_rid == link.target_rid {
            return Err(YantrikDbError::InvalidInput(
                "link: a record cannot link to itself".to_string(),
            ));
        }

        let link_id = crate::id::new_id();
        let link_type_str = link.link_type.as_str();
        let ts = now();
        let hlc_bytes = self.tick_hlc().to_bytes().to_vec();

        {
            let conn = self.conn.lock();
            conn.execute(
                "INSERT OR IGNORE INTO record_links \
                 (link_id, source_rid, target_rid, link_type, status, \
                  created_at, hlc, origin_actor) \
                 VALUES (?1, ?2, ?3, ?4, 'active', ?5, ?6, ?7)",
                params![
                    link_id,
                    source_rid,
                    link.target_rid,
                    link_type_str,
                    ts,
                    hlc_bytes,
                    self.actor_id,
                ],
            )?;
        }

        self.log_op(
            "link",
            Some(source_rid),
            &serde_json::json!({
                "source_rid": source_rid,
                "target_rid": link.target_rid,
                "link_type": link_type_str,
                "created_at": ts,
            }),
            None,
        )?;

        Ok(link_id)
    }

    /// Remove a single link. Returns `true` if a row was deleted.
    ///
    /// Unlike `forget()` (which marks links broken for audit), explicit
    /// `unlink()` is a user retraction and hard-deletes the row.
    pub fn unlink(&self, source_rid: &str, target_rid: &str, link_type: &LinkType) -> Result<bool> {
        let link_type_str = link_type.as_str();
        let deleted = {
            let conn = self.conn.lock();
            conn.execute(
                "DELETE FROM record_links \
                 WHERE source_rid = ?1 AND target_rid = ?2 AND link_type = ?3",
                params![source_rid, target_rid, link_type_str],
            )?
        };

        if deleted > 0 {
            self.log_op(
                "unlink",
                Some(source_rid),
                &serde_json::json!({
                    "source_rid": source_rid,
                    "target_rid": target_rid,
                    "link_type": link_type_str,
                }),
                None,
            )?;
        }

        Ok(deleted > 0)
    }

    /// Issue #48 — one-shot reification of the legacy
    /// `metadata.supersedes = "<rid>"` string convention into proper
    /// `Supersedes` record links. Returns the number of links created.
    ///
    /// **Why this is an explicit method, not an auto-migration in
    /// `new()`:** auto-running a data migration that emits oplog ops on
    /// every engine open is an idempotency hazard, and `new()`'s struct
    /// construction is not a clean place to thread the HLC clock. Calling
    /// `self.link()` per row gives a correct `tick_hlc()` HLC + a
    /// replicating `link` op + idempotency (UNIQUE → INSERT OR IGNORE)
    /// for free. `origin_actor` is the calling node's actor (a real
    /// owner) rather than a synthetic 'migration_v31' tag — which is more
    /// correct for replication. Operators run this once during the
    /// schema-v31 upgrade. Idempotent: safe to run repeatedly.
    ///
    /// Reads metadata via the decrypt path so it works on encrypted DBs.
    pub fn reify_supersedes_links(&self) -> Result<usize> {
        // Pull rid + stored (possibly encrypted) metadata for all active
        // memories. We decrypt + JSON-parse in Rust rather than relying on
        // SQLite json_extract, which can't see through encrypted metadata.
        let rows: Vec<(String, String)> = {
            let conn = self.conn.lock();
            let mut stmt = conn.prepare(
                "SELECT rid, metadata FROM memories \
                 WHERE consolidation_status = 'active'",
            )?;
            let mapped = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            mapped.collect::<std::result::Result<Vec<_>, _>>()?
        };

        let mut created = 0usize;
        for (rid, stored_meta) in rows {
            let meta_str = self.decrypt_text(&stored_meta)?;
            let Ok(meta) = serde_json::from_str::<serde_json::Value>(&meta_str) else {
                continue;
            };
            let Some(target) = meta.get("supersedes").and_then(|v| v.as_str()) else {
                continue;
            };
            if target.is_empty() || target == rid {
                continue;
            }
            // link() is idempotent on UNIQUE(source,target,type); a
            // re-run simply INSERT OR IGNOREs. Count only fresh inserts by
            // checking row presence before/after would be racy under the
            // lock churn; instead we count attempts that didn't error.
            self.link(
                &rid,
                &RecordLink {
                    target_rid: target.to_string(),
                    link_type: LinkType::Supersedes,
                },
            )?;
            created += 1;
        }
        Ok(created)
    }

    /// Traverse links from `rid`. Only `status='active'` links are
    /// returned. For `Contradicts` (symmetric), `Outbound` and `Inbound`
    /// both surface the partner; otherwise direction is literal.
    ///
    /// `link_type=None` returns all types.
    pub fn linked_records(
        &self,
        rid: &str,
        direction: LinkDirection,
        link_type: Option<&LinkType>,
    ) -> Result<Vec<LinkedRecord>> {
        let type_filter = link_type.map(|lt| lt.as_str());
        let mut out: Vec<LinkedRecord> = Vec::new();
        let conn = self.conn.lock();

        // Outbound: rid is source → return target as the linked record.
        if matches!(direction, LinkDirection::Outbound | LinkDirection::Both) {
            let mut stmt = conn.prepare(
                "SELECT target_rid, link_type, created_at FROM record_links \
                 WHERE source_rid = ?1 AND status = 'active' \
                 AND (?2 IS NULL OR link_type = ?2) \
                 ORDER BY created_at ASC",
            )?;
            let rows = stmt.query_map(params![rid, type_filter], |row| {
                Ok(LinkedRecord {
                    rid: row.get::<_, String>(0)?,
                    link_type: row.get::<_, String>(1)?,
                    created_at: row.get::<_, f64>(2)?,
                    direction: "outbound".to_string(),
                })
            })?;
            for r in rows {
                out.push(r?);
            }
        }

        // Inbound: rid is target → return source as the linked record.
        if matches!(direction, LinkDirection::Inbound | LinkDirection::Both) {
            let mut stmt = conn.prepare(
                "SELECT source_rid, link_type, created_at FROM record_links \
                 WHERE target_rid = ?1 AND status = 'active' \
                 AND (?2 IS NULL OR link_type = ?2) \
                 ORDER BY created_at ASC",
            )?;
            let rows = stmt.query_map(params![rid, type_filter], |row| {
                Ok(LinkedRecord {
                    rid: row.get::<_, String>(0)?,
                    link_type: row.get::<_, String>(1)?,
                    created_at: row.get::<_, f64>(2)?,
                    direction: "inbound".to_string(),
                })
            })?;
            for r in rows {
                out.push(r?);
            }
        }

        // Symmetric link types (Contradicts): when querying one
        // direction, also surface the partner from the OTHER direction so
        // "A contradicts B" is visible from both A and B regardless of
        // which way the row was stored. Only do this when not already
        // querying Both (which covers both directions anyway).
        if !matches!(direction, LinkDirection::Both) {
            let want_symmetric = match link_type {
                Some(lt) => lt.is_symmetric(),
                None => true, // unfiltered: include symmetric partners
            };
            if want_symmetric {
                let (col_match, col_return, dir_label) = match direction {
                    LinkDirection::Outbound => ("target_rid", "source_rid", "inbound"),
                    LinkDirection::Inbound => ("source_rid", "target_rid", "outbound"),
                    LinkDirection::Both => unreachable!(),
                };
                let sql = format!(
                    "SELECT {col_return}, link_type, created_at FROM record_links \
                     WHERE {col_match} = ?1 AND status = 'active' \
                     AND link_type = 'contradicts' \
                     AND (?2 IS NULL OR link_type = ?2) \
                     ORDER BY created_at ASC"
                );
                let mut stmt = conn.prepare(&sql)?;
                let rows = stmt.query_map(params![rid, type_filter], |row| {
                    Ok(LinkedRecord {
                        rid: row.get::<_, String>(0)?,
                        link_type: row.get::<_, String>(1)?,
                        created_at: row.get::<_, f64>(2)?,
                        direction: dir_label.to_string(),
                    })
                })?;
                for r in rows {
                    out.push(r?);
                }
            }
        }

        Ok(out)
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

    fn rec(db: &YantrikDB, text: &str, seed: f32) -> String {
        db.record(
            text,
            "semantic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({}),
            &vec_seed(seed, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap()
    }

    #[test]
    fn record_with_links_creates_links_atomically() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let b = rec(&db, "target", 1.0);
        let c = rec(&db, "another", 2.0);
        let a = db
            .record_with_links(
                "source",
                "semantic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &vec_seed(3.0, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
                &[
                    RecordLink {
                        target_rid: b.clone(),
                        link_type: LinkType::Supersedes,
                    },
                    RecordLink {
                        target_rid: c.clone(),
                        link_type: LinkType::Supports,
                    },
                ],
            )
            .unwrap();

        let out = db
            .linked_records(&a, LinkDirection::Outbound, None)
            .unwrap();
        assert_eq!(out.len(), 2);
        assert!(out
            .iter()
            .any(|l| l.rid == b && l.link_type == "supersedes"));
        assert!(out.iter().any(|l| l.rid == c && l.link_type == "supports"));
    }

    #[test]
    fn link_is_idempotent_on_unique() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let a = rec(&db, "a", 1.0);
        let b = rec(&db, "b", 2.0);
        let link = RecordLink {
            target_rid: b.clone(),
            link_type: LinkType::Advances,
        };
        db.link(&a, &link).unwrap();
        db.link(&a, &link).unwrap(); // second is INSERT OR IGNORE no-op
        let out = db
            .linked_records(&a, LinkDirection::Outbound, None)
            .unwrap();
        assert_eq!(out.len(), 1, "duplicate link must not create a second row");
    }

    #[test]
    fn link_rejects_self_and_empty() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let a = rec(&db, "a", 1.0);
        assert!(db
            .link(
                &a,
                &RecordLink {
                    target_rid: a.clone(),
                    link_type: LinkType::Advances
                }
            )
            .is_err());
        assert!(db
            .link(
                &a,
                &RecordLink {
                    target_rid: String::new(),
                    link_type: LinkType::Advances
                }
            )
            .is_err());
    }

    #[test]
    fn unlink_removes_and_reports() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let a = rec(&db, "a", 1.0);
        let b = rec(&db, "b", 2.0);
        db.link(
            &a,
            &RecordLink {
                target_rid: b.clone(),
                link_type: LinkType::Supports,
            },
        )
        .unwrap();
        assert!(db.unlink(&a, &b, &LinkType::Supports).unwrap());
        assert!(
            !db.unlink(&a, &b, &LinkType::Supports).unwrap(),
            "second unlink is a no-op"
        );
        assert!(db
            .linked_records(&a, LinkDirection::Outbound, None)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn linked_records_inbound_and_typed_filter() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let a = rec(&db, "a", 1.0);
        let b = rec(&db, "b", 2.0);
        db.link(
            &a,
            &RecordLink {
                target_rid: b.clone(),
                link_type: LinkType::Supersedes,
            },
        )
        .unwrap();
        // Inbound on b finds a.
        let inbound = db.linked_records(&b, LinkDirection::Inbound, None).unwrap();
        assert_eq!(inbound.len(), 1);
        assert_eq!(inbound[0].rid, a);
        assert_eq!(inbound[0].direction, "inbound");
        // Typed filter that doesn't match returns empty.
        let none = db
            .linked_records(&b, LinkDirection::Inbound, Some(&LinkType::Supports))
            .unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn contradicts_is_bidirectional() {
        // A contradicts B (stored A->B). Querying from B must surface A
        // even though B is the target, because contradicts is symmetric.
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let a = rec(&db, "a", 1.0);
        let b = rec(&db, "b", 2.0);
        db.link(
            &a,
            &RecordLink {
                target_rid: b.clone(),
                link_type: LinkType::Contradicts,
            },
        )
        .unwrap();

        let from_b = db
            .linked_records(&b, LinkDirection::Outbound, Some(&LinkType::Contradicts))
            .unwrap();
        assert!(
            from_b.iter().any(|l| l.rid == a),
            "contradicts must be visible from the target endpoint too, got {from_b:?}"
        );
    }

    #[test]
    fn forget_marks_links_broken_not_deleted() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let a = rec(&db, "a", 1.0);
        let b = rec(&db, "b", 2.0);
        db.link(
            &a,
            &RecordLink {
                target_rid: b.clone(),
                link_type: LinkType::Supports,
            },
        )
        .unwrap();

        db.forget(&a).unwrap();

        // Active traversal no longer returns it.
        assert!(db
            .linked_records(&a, LinkDirection::Outbound, None)
            .unwrap()
            .is_empty());
        // But the row is retained with a broken status (audit trail).
        let conn = db.conn();
        let status: String = conn
            .query_row(
                "SELECT status FROM record_links WHERE source_rid = ?1 AND target_rid = ?2",
                rusqlite::params![a, b],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(status, "broken_source_forgotten");
    }

    #[test]
    fn correct_preserves_links_via_rid_stability() {
        // v0.7.20 correct() mutates in place (rid preserved), so links
        // keyed on rid survive a correction with no special handling.
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let a = rec(&db, "a v0", 1.0);
        let b = rec(&db, "b", 2.0);
        db.link(
            &a,
            &RecordLink {
                target_rid: b.clone(),
                link_type: LinkType::Advances,
            },
        )
        .unwrap();
        db.correct(&a, Some("a v1"), None, None, None, "fix")
            .unwrap();
        let out = db
            .linked_records(&a, LinkDirection::Outbound, None)
            .unwrap();
        assert_eq!(
            out.len(),
            1,
            "links survive in-place correction (rid preserved)"
        );
        assert_eq!(out[0].rid, b);
    }

    #[test]
    fn reify_supersedes_links_from_metadata() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let old = rec(&db, "old wonder", 1.0);
        // New record carries the legacy metadata.supersedes string.
        let new = db
            .record(
                "new wonder",
                "semantic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({ "supersedes": old }),
                &vec_seed(2.0, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        let n = db.reify_supersedes_links().unwrap();
        assert_eq!(n, 1, "one supersedes link reified");
        let out = db
            .linked_records(&new, LinkDirection::Outbound, Some(&LinkType::Supersedes))
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].rid, old);

        // Idempotent: re-running doesn't duplicate.
        db.reify_supersedes_links().unwrap();
        let out2 = db
            .linked_records(&new, LinkDirection::Outbound, Some(&LinkType::Supersedes))
            .unwrap();
        assert_eq!(out2.len(), 1, "reify is idempotent");
    }

    #[test]
    fn link_type_string_roundtrip() {
        for lt in [
            LinkType::Advances,
            LinkType::Supersedes,
            LinkType::Contradicts,
            LinkType::Supports,
            LinkType::Questions,
            LinkType::DerivedFrom,
            LinkType::Custom("my_link".to_string()),
        ] {
            assert_eq!(LinkType::from_str_lenient(&lt.as_str()), lt);
        }
        // Unknown string is lenient -> Custom.
        assert_eq!(
            LinkType::from_str_lenient("future_type"),
            LinkType::Custom("future_type".to_string())
        );
    }
}
