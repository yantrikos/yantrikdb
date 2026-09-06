//! One-time claim re-extraction — the heal that makes an extractor fix
//! reach a store that already exists.
//!
//! The materializer extracts claims when a memory is WRITTEN. An extractor
//! improvement therefore touches only new writes: the production memory
//! store measured on 2026-09-05 still carried 2,576 claims minted by the
//! old patterns (57% junk `leads`, "Pranab runs UTC") after the anchored
//! extractor shipped, and the claim-chain and conflict scanners kept
//! reading them. `reextract_claims` drops every claim the extractors
//! minted (`heuristic_v1`, `learned_v1`) and re-runs the materializer's
//! own extraction over every active memory — one definition, two callers.
//!
//! Scope, deliberately: extractor-minted claims ONLY. `relate()` rows
//! (`manual`) and writer-stated claims (`agent_stated`) are assertions,
//! not derivations, and are never touched. Extractor claims are node-local
//! derived state (their `hlc` is NULL, they do not replicate), so deleting
//! and regenerating them is safe; nothing else references a heuristic
//! claim by id. The in-memory graph index is rebuilt at the end so
//! expansion stops seeing the deleted edges.

use std::collections::BTreeMap;

use rusqlite::params;

use crate::error::Result;

/// Extractor labels whose claims are derived from text and safe to regenerate.
pub const REEXTRACT_EXTRACTORS: &[&str] = &["heuristic_v1", "learned_v1"];
/// Memories processed per write-lock hold.
const REEXTRACT_BATCH: usize = 500;

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ReextractReport {
    pub namespace: Option<String>,
    pub dry_run: bool,
    pub memories_scanned: usize,
    pub claims_removed: usize,
    pub claims_written: usize,
    /// Extractor-minted claims by relation before the heal.
    pub before_by_rel: BTreeMap<String, i64>,
    /// Extractor-minted claims by relation after the heal (equals `before`
    /// on a dry run).
    pub after_by_rel: BTreeMap<String, i64>,
}

impl super::YantrikDB {
    fn extracted_claims_by_rel(&self, namespace: Option<&str>) -> Result<BTreeMap<String, i64>> {
        let conn = self.conn();
        let sql = format!(
            "SELECT rel_type, COUNT(*) FROM claims WHERE tombstoned = 0 \
             AND extractor IN ('heuristic_v1','learned_v1') {} GROUP BY rel_type",
            if namespace.is_some() {
                "AND namespace = ?1"
            } else {
                ""
            }
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows: Vec<(String, i64)> = if let Some(ns) = namespace {
            stmt.query_map(params![ns], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<std::result::Result<_, _>>()?
        } else {
            stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<std::result::Result<_, _>>()?
        };
        Ok(rows.into_iter().collect())
    }

    /// Drop every extractor-minted claim (optionally in one namespace) and
    /// re-extract from every active memory with the current extractor. See
    /// the module note for scope and safety. `dry_run` reports the current
    /// state and touches nothing.
    pub fn reextract_claims(
        &self,
        namespace: Option<&str>,
        dry_run: bool,
    ) -> Result<ReextractReport> {
        let before_by_rel = self.extracted_claims_by_rel(namespace)?;
        let mut report = ReextractReport {
            namespace: namespace.map(str::to_string),
            dry_run,
            memories_scanned: 0,
            claims_removed: 0,
            claims_written: 0,
            after_by_rel: before_by_rel.clone(),
            before_by_rel,
        };

        // Count what would be scanned even on a dry run.
        {
            let conn = self.conn();
            let sql = format!(
                "SELECT COUNT(*) FROM memories WHERE consolidation_status = 'active' {}",
                if namespace.is_some() {
                    "AND namespace = ?1"
                } else {
                    ""
                }
            );
            report.memories_scanned = if let Some(ns) = namespace {
                conn.query_row(&sql, params![ns], |r| r.get::<_, i64>(0))? as usize
            } else {
                conn.query_row(&sql, [], |r| r.get::<_, i64>(0))? as usize
            };
        }
        if dry_run {
            return Ok(report);
        }

        // 1. Remove the derived rows.
        {
            let conn = self.conn();
            let sql = format!(
                "DELETE FROM claims WHERE extractor IN ('heuristic_v1','learned_v1') {}",
                if namespace.is_some() {
                    "AND namespace = ?1"
                } else {
                    ""
                }
            );
            report.claims_removed = if let Some(ns) = namespace {
                conn.execute(&sql, params![ns])?
            } else {
                conn.execute(&sql, [])?
            };
        }

        // 2. Re-extract, keyset-paged so no single lock hold scans the store.
        let mut last_rowid: i64 = 0;
        let mut scanned = 0usize;
        loop {
            let page: Vec<(i64, String, String, String)> = {
                let conn = self.conn();
                let sql = format!(
                    "SELECT rowid, rid, text, namespace FROM memories \
                     WHERE consolidation_status = 'active' AND rowid > ?1 {} \
                     ORDER BY rowid LIMIT ?2",
                    if namespace.is_some() {
                        "AND namespace = ?3"
                    } else {
                        ""
                    }
                );
                let mut stmt = conn.prepare(&sql)?;
                let mapper =
                    |r: &rusqlite::Row| -> rusqlite::Result<(i64, String, String, String)> {
                        Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
                    };
                if let Some(ns) = namespace {
                    stmt.query_map(params![last_rowid, REEXTRACT_BATCH as i64, ns], mapper)?
                        .collect::<std::result::Result<_, _>>()?
                } else {
                    stmt.query_map(params![last_rowid, REEXTRACT_BATCH as i64], mapper)?
                        .collect::<std::result::Result<_, _>>()?
                }
            };
            if page.is_empty() {
                break;
            }
            for (rowid, rid, stored, ns) in page {
                last_rowid = rowid;
                scanned += 1;
                let text = match self.decrypt_text(&stored) {
                    Ok(t) => t,
                    Err(_) => continue, // unreadable row: leave it claimless rather than fail the heal
                };
                let heuristic = self.extract_entities_for(&text);
                report.claims_written += self.ingest_extracted_claims(&rid, &text, &ns, &heuristic);
            }
        }
        report.memories_scanned = scanned;

        // 3. The in-memory graph index still holds the deleted edges.
        self.rebuild_graph_index()?;
        report.after_by_rel = self.extracted_claims_by_rel(namespace)?;
        Ok(report)
    }
}

/// Report of [`YantrikDB::reextract_entities`].
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct EntityAdmissionReport {
    pub dry_run: bool,
    pub entities_scanned: usize,
    /// Names that fail [`crate::graph::admit_entity`] today.
    pub inadmissible: usize,
    /// Inadmissible names kept because a manual or writer-stated claim
    /// references them — an assertion outranks a heuristic. Their
    /// `memory_entities` links are still removed (counted in
    /// `links_removed`), so a value a claim asserts stops being a hub.
    pub kept_by_claims: usize,
    /// Active memories scanned to rebuild `token_case_stats` first.
    pub lexicon_memories: usize,
    pub entities_removed: usize,
    pub links_removed: usize,
    /// Extractor-minted claims dropped because an endpoint was removed.
    pub claims_removed: usize,
    /// Junk-class counts before the heal (`all_caps`, `has_digit`,
    /// `no_letters`, `four_plus_words`, `long`).
    pub before_classes: BTreeMap<String, i64>,
    /// The same counts after (equals `before` on a dry run).
    pub after_classes: BTreeMap<String, i64>,
}

const ENTITY_HEAL_BATCH: usize = 500;

impl super::YantrikDB {
    fn entity_classes(&self) -> Result<BTreeMap<String, i64>> {
        let conn = self.conn();
        let mut out = BTreeMap::new();
        let q = |sql: &str| -> Result<i64> { Ok(conn.query_row(sql, [], |r| r.get::<_, i64>(0))?) };
        out.insert("total".into(), q("SELECT COUNT(*) FROM entities")?);
        out.insert(
            "all_caps".into(),
            q("SELECT COUNT(*) FROM entities WHERE name = upper(name) AND name <> lower(name) AND length(name) >= 2")?,
        );
        out.insert(
            "has_digit".into(),
            q("SELECT COUNT(*) FROM entities WHERE name GLOB '*[0-9]*'")?,
        );
        out.insert(
            "no_letters".into(),
            q("SELECT COUNT(*) FROM entities WHERE NOT name GLOB '*[A-Za-z]*'")?,
        );
        out.insert(
            "four_plus_words".into(),
            q("SELECT COUNT(*) FROM entities WHERE (length(name) - length(replace(name,' ',''))) >= 3")?,
        );
        out.insert(
            "long".into(),
            q("SELECT COUNT(*) FROM entities WHERE length(name) >= 40")?,
        );
        Ok(out)
    }

    /// One-time heal for the entity table: re-apply today's admission
    /// predicate ([`crate::graph::admit_entity`]) to every stored entity
    /// and remove the ones it refuses — with their `memory_entities` links
    /// and any extractor-minted claim that used them as an endpoint — then
    /// rebuild the graph index so expansion stops seeing the dropped nodes.
    ///
    /// Nothing revisits admitted nodes otherwise: the extractor gate in
    /// #213 protects new writes only, and a store written by an older
    /// extractor keeps every heading and bare number it ever minted as a
    /// hop the claims lane can follow. This is the entity-table twin of
    /// [`Self::reextract_claims`], and like it is idempotent.
    ///
    /// An inadmissible name referenced by a `manual` or `agent_stated`
    /// claim is KEPT: a writer asserted it, and an assertion outranks the
    /// heuristic. The entity table is store-wide (no namespace column), so
    /// the heal is too. `dry_run` reports and changes nothing.
    pub fn reextract_entities(&self, dry_run: bool) -> Result<EntityAdmissionReport> {
        let before_classes = self.entity_classes()?;
        // The store's lexicon first: admission below asks it how each
        // single-token name is written here. Rebuilding is a read of every
        // active text and is idempotent, so a dry run may do it too.
        let lexicon_memories = self.rebuild_token_case_stats()?;
        let mut report = EntityAdmissionReport {
            dry_run,
            entities_scanned: 0,
            inadmissible: 0,
            kept_by_claims: 0,
            lexicon_memories,
            entities_removed: 0,
            links_removed: 0,
            claims_removed: 0,
            after_classes: before_classes.clone(),
            before_classes,
        };
        let names: Vec<String> = {
            let conn = self.conn();
            let mut stmt = conn.prepare("SELECT name FROM entities ORDER BY name")?;
            let rows: Vec<String> = stmt
                .query_map([], |r| r.get::<_, String>(0))?
                .collect::<std::result::Result<_, _>>()?;
            rows
        };
        report.entities_scanned = names.len();
        let mut doomed: Vec<String> = Vec::new();
        let mut kept: Vec<String> = Vec::new();
        {
            let conn = self.conn();
            let mut asserted = conn.prepare(
                "SELECT COUNT(*) FROM claims WHERE tombstoned = 0 \
                 AND extractor NOT IN ('heuristic_v1','learned_v1') \
                 AND (src = ?1 OR dst = ?1)",
            )?;
            for name in &names {
                if crate::graph::admit_entity_with(name, |tok| Self::token_case_stats(&conn, tok)) {
                    continue;
                }
                report.inadmissible += 1;
                let refs: i64 = asserted.query_row(params![name], |r| r.get(0))?;
                if refs > 0 {
                    report.kept_by_claims += 1;
                    kept.push(name.clone());
                    continue;
                }
                doomed.push(name.clone());
            }
        }
        if dry_run {
            return Ok(report);
        }
        // A kept name keeps its row and its asserted claim, but not the
        // links that made `2026` a 49-connection hub for graph expansion.
        for batch in kept.chunks(ENTITY_HEAL_BATCH) {
            let conn = self.conn();
            for name in batch {
                report.links_removed += conn.execute(
                    "DELETE FROM memory_entities WHERE entity_name = ?1",
                    params![name],
                )?;
            }
        }
        for batch in doomed.chunks(ENTITY_HEAL_BATCH) {
            let conn = self.conn();
            for name in batch {
                report.links_removed += conn.execute(
                    "DELETE FROM memory_entities WHERE entity_name = ?1",
                    params![name],
                )?;
                report.claims_removed += conn.execute(
                    "DELETE FROM claims WHERE extractor IN ('heuristic_v1','learned_v1') \
                     AND (src = ?1 OR dst = ?1)",
                    params![name],
                )?;
                report.entities_removed +=
                    conn.execute("DELETE FROM entities WHERE name = ?1", params![name])?;
            }
        }
        self.rebuild_graph_index()?;
        report.after_classes = self.entity_classes()?;
        Ok(report)
    }
}
