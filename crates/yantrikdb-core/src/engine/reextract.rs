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
    /// Extractor-minted claims — and derived co-occurrence edges — dropped
    /// because an endpoint was removed.
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
            // An assertion outranks the heuristic — but a co-occurrence edge
            // is derived, not asserted, whatever extractor label auto-relate
            // stamped on it: on the production store 141 `co_occurs_with`
            // rows labelled `manual` were all that kept `NOT`, `AND` and
            // `10` in the table with a thousand mentions each.
            let mut asserted = conn.prepare(
                "SELECT COUNT(*) FROM claims WHERE tombstoned = 0 \
                 AND extractor NOT IN ('heuristic_v1','learned_v1') \
                 AND rel_type NOT IN ('co_occurs_with','related_to','mentions') \
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
                    "DELETE FROM claims WHERE (extractor IN ('heuristic_v1','learned_v1') \
                     OR rel_type IN ('co_occurs_with','related_to','mentions')) \
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

// ── The refusal ledger and silver recall (2026-09-07) ───────────────────
//
// Precision work on the extractor was blind: every change looked free
// because nothing measured recall without a judge. Two instruments, both
// judge-free and both computed from what the store already holds:
//
// * the REFUSAL LEDGER (`extraction_refusals`, written by the bound
//   extractor on every extraction): a histogram of why triggers did not
//   bind. If most misses are `lowercase_subject`, alias policy is the
//   lever; if `subject_not_adjacent`, the wrapper list; if the relation
//   never appears, the pattern table.
// * SILVER RECALL: every cooperative claim (`agent_stated`) is a labelled
//   example — the writer stated a triple and the engine grounded it in a
//   memory's text. Hide the label, re-run the extractor over that memory,
//   and ask whether it recovers the same triple. Not ground truth (the
//   writer's vocabulary is wider than the pattern table, and a grounded
//   claim need not be a sentence the patterns cover), but it MOVES when a
//   precision change costs recall, which is what was missing.

/// One row of the refusal ledger.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ExtractionRefusalRow {
    pub memory_rid: String,
    pub namespace: String,
    pub rel_type: String,
    pub trigger: String,
    pub reason: String,
    pub left_token: String,
    pub right_token: String,
    pub at: i64,
    pub extractor_version: String,
}

/// Report of [`YantrikDB::extraction_silver_recall`].
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SilverRecallReport {
    pub namespace: Option<String>,
    /// Cooperative claims with a readable source memory.
    pub stated_claims: usize,
    /// Stated triples the bound extractor re-derived from the same text.
    pub recovered: usize,
    /// Stated triples whose relation no built-in pattern can mint at all —
    /// outside the extractor's vocabulary, counted apart from misses.
    pub unsupported_relation: usize,
    /// Missed triples by cause: a refusal reason recorded for the same
    /// relation on that memory, `bound_elsewhere` (the relation fired on
    /// different endpoints), or `no_trigger` (no pattern fired).
    pub missed_by_reason: BTreeMap<String, usize>,
    /// Recovered / (recovered + missed), on the supported relations only.
    pub recall: f64,
}

impl super::YantrikDB {
    /// Refusal counts keyed `rel_type:reason`, the histogram the next
    /// binding rule is chosen from. Empty on a pre-v54 store.
    pub fn extraction_refusal_counts(
        &self,
        namespace: Option<&str>,
    ) -> Result<BTreeMap<String, i64>> {
        let conn = self.conn();
        let sql = format!(
            "SELECT rel_type, reason, COUNT(*) FROM extraction_refusals {} \
             GROUP BY rel_type, reason",
            if namespace.is_some() {
                "WHERE namespace = ?1"
            } else {
                ""
            }
        );
        let Ok(mut stmt) = conn.prepare(&sql) else {
            return Ok(BTreeMap::new());
        };
        let mapper = |r: &rusqlite::Row| -> rusqlite::Result<(String, String, i64)> {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        };
        let rows: Vec<(String, String, i64)> = if let Some(ns) = namespace {
            stmt.query_map(params![ns], mapper)?
                .collect::<std::result::Result<_, _>>()?
        } else {
            stmt.query_map([], mapper)?
                .collect::<std::result::Result<_, _>>()?
        };
        Ok(rows
            .into_iter()
            .map(|(rel, reason, n)| (format!("{rel}:{reason}"), n))
            .collect())
    }

    /// The ledger rows themselves, newest first, for inspection.
    pub fn extraction_refusals(
        &self,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ExtractionRefusalRow>> {
        let conn = self.conn();
        let sql = format!(
            "SELECT memory_rid, namespace, rel_type, trigger, reason, left_token, right_token, \
             at, extractor_version FROM extraction_refusals {} \
             ORDER BY created_at DESC, memory_rid, at LIMIT ?{}",
            if namespace.is_some() {
                "WHERE namespace = ?1"
            } else {
                ""
            },
            if namespace.is_some() { 2 } else { 1 }
        );
        let Ok(mut stmt) = conn.prepare(&sql) else {
            return Ok(Vec::new());
        };
        let mapper = |r: &rusqlite::Row| -> rusqlite::Result<ExtractionRefusalRow> {
            Ok(ExtractionRefusalRow {
                memory_rid: r.get(0)?,
                namespace: r.get(1)?,
                rel_type: r.get(2)?,
                trigger: r.get(3)?,
                reason: r.get(4)?,
                left_token: r.get(5)?,
                right_token: r.get(6)?,
                at: r.get(7)?,
                extractor_version: r.get(8)?,
            })
        };
        let limit = limit as i64;
        let rows = if let Some(ns) = namespace {
            stmt.query_map(params![ns, limit], mapper)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            stmt.query_map(params![limit], mapper)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        Ok(rows)
    }

    /// Silver recall: re-derive every cooperative claim from its source
    /// memory with the bound extractor and count what came back. See the
    /// module note. Read-only.
    pub fn extraction_silver_recall(&self, namespace: Option<&str>) -> Result<SilverRecallReport> {
        let stated: Vec<(String, String, String, String)> = {
            let conn = self.conn();
            let sql = format!(
                "SELECT src, rel_type, dst, source_memory_rid FROM claims \
                 WHERE extractor = ?1 AND tombstoned = 0 AND source_memory_rid IS NOT NULL {} \
                 ORDER BY created_at",
                if namespace.is_some() {
                    "AND namespace = ?2"
                } else {
                    ""
                }
            );
            let mut stmt = conn.prepare(&sql)?;
            let mapper = |r: &rusqlite::Row| -> rusqlite::Result<(String, String, String, String)> {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
            };
            if let Some(ns) = namespace {
                stmt.query_map(
                    params![crate::engine::graph_ops::STATED_CLAIM_EXTRACTOR, ns],
                    mapper,
                )?
                .collect::<std::result::Result<_, _>>()?
            } else {
                stmt.query_map(
                    params![crate::engine::graph_ops::STATED_CLAIM_EXTRACTOR],
                    mapper,
                )?
                .collect::<std::result::Result<_, _>>()?
            }
        };
        let supported: std::collections::HashSet<String> =
            crate::graph::builtin_relation_types().into_iter().collect();
        let mut report = SilverRecallReport {
            namespace: namespace.map(str::to_string),
            stated_claims: 0,
            recovered: 0,
            unsupported_relation: 0,
            missed_by_reason: BTreeMap::new(),
            recall: 0.0,
        };
        // One extraction per memory, shared by its claims.
        let mut cache: std::collections::HashMap<String, Option<crate::graph::RelationExtraction>> =
            std::collections::HashMap::new();
        for (src, rel, dst, rid) in stated {
            let extraction = cache.entry(rid.clone()).or_insert_with(|| {
                let stored: Option<String> = {
                    let conn = self.conn();
                    conn.query_row(
                        "SELECT text FROM memories WHERE rid = ?1",
                        params![rid],
                        |r| r.get(0),
                    )
                    .ok()
                };
                let text = stored.and_then(|t| self.decrypt_text(&t).ok())?;
                let mut candidates = self.extract_entities_for(&text);
                for v in crate::graph::extract_value_candidates(&text) {
                    if !candidates.contains(&v) {
                        candidates.push(v);
                    }
                }
                Some(crate::graph::extract_relations_bound(&text, &candidates))
            });
            let Some(extraction) = extraction else {
                continue; // unreadable source: not a labelled example
            };
            report.stated_claims += 1;
            if !supported.contains(&rel) {
                report.unsupported_relation += 1;
                continue;
            }
            let same = |a: &str, b: &str| a.eq_ignore_ascii_case(b);
            if extraction
                .relations
                .iter()
                .any(|r| r.rel_type == rel && same(&r.src, &src) && same(&r.dst, &dst))
            {
                report.recovered += 1;
                continue;
            }
            let reason =
                if let Some(refusal) = extraction.refusals.iter().find(|r| r.rel_type == rel) {
                    refusal.reason.to_string()
                } else if extraction.relations.iter().any(|r| r.rel_type == rel) {
                    "bound_elsewhere".to_string()
                } else {
                    "no_trigger".to_string()
                };
            *report.missed_by_reason.entry(reason).or_insert(0) += 1;
        }
        let missed: usize = report.missed_by_reason.values().sum();
        let denom = report.recovered + missed;
        report.recall = if denom == 0 {
            0.0
        } else {
            report.recovered as f64 / denom as f64
        };
        Ok(report)
    }
}
