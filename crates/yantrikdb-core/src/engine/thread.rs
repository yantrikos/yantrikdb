//! Coverage-first thread retrieval (opt-in) — the engine half of the
//! event_ordering recovery hypothesis.
//!
//! Event-ordering questions ("in what order did I mention X?") are a
//! SET-COVERAGE + SEQUENCE task: they need EVERY row mentioning an entity,
//! each with its position, and nothing else. Similarity recall — even with
//! `order="chronological"` — only REORDERS its similarity-bounded pool, so
//! coverage is never guaranteed: a thread member the query vector doesn't
//! rank into the pool is silently absent (the same pool-bounded
//! false-negative shape recall's event-time filter closed for valid time
//! in #149/#173). `recall_thread` is the explicit opt-in alternative:
//! deterministic SQL over the `memory_entities` join, no vector search, no
//! ranking. No default behavior changes anywhere.
//!
//! Contract:
//! - **Coverage-first eligibility.** The eligible set is ALL visible
//!   memories in the namespace joined through `memory_entities`
//!   (base/schema.rs: `memory_entities(memory_rid, entity_name)`, indexed
//!   both directions) matching ANY requested entity.
//! - **Entity-name matching.** The engine writes `entity_name` VERBATIM as
//!   extracted or caller-supplied — case-preserved, never lowercased (see
//!   the `INSERT ... INTO memory_entities` sites in `engine/record.rs`,
//!   `engine/stats.rs` and `engine/graph_ops.rs`); entity↔text matching
//!   happens through `crate::graph::tokenize`'s Unicode lowercasing
//!   instead. This lane matches the same way, through the PERSISTED
//!   normalized key: every writer also stamps `entity_name_norm` =
//!   [`normalize_entity_name`] — Unicode lowercase via Rust
//!   `to_lowercase()`, deliberately NOT full Unicode case folding, in
//!   lockstep with `crate::graph::tokenize` (never SQL `LOWER()`, which
//!   is ASCII-only and would diverge from the tokenizer on non-ASCII
//!   names) — so requested names resolve with one indexed lookup on
//!   `idx_memory_entities_norm` instead of Rust-lowercasing the whole
//!   entity vocabulary per call (the pre-v49 O(V) DISTINCT scan).
//! - **Visibility.** Mirrors recall's default read predicates: rows must be
//!   `consolidation_status = 'active'` (recall's default,
//!   `include_consolidated = false`), pass the synthesis lifecycle gate
//!   (`synthesis_state IS NULL OR synthesis_state = 'verified'`, recall's
//!   `synthesis_lifecycle_allows`), and — when the v0.10 status read policy
//!   is active — not be the target of a selected active `supersedes` link
//!   (recall step 3.4's `superseded_rids_among`). Tombstoned rows are
//!   excluded twice over: the status predicate, plus `tombstone_inner`
//!   deleting their `memory_entities` rows.
//! - **Order.** Ascending `(created_at, source_turn NULLS LAST, rid)` — a
//!   deterministic total order. `created_at` is transaction/ingestion time,
//!   which under conversational ingestion tracks conversation order; the
//!   turn tie-break only matters within one ingestion instant.
//! - **Truncation is loud, never sampled.** If eligible > limit the FIRST
//!   `limit` chronological rows are kept (the earliest) and the caller
//!   learns what happened through `total`/`omitted` — the `recall_facets`
//!   omitted precedent. Positions are 1-based over the FULL eligible
//!   thread (pre-truncation); because truncation keeps the earliest
//!   prefix, returned items always carry positions `1..=items.len()`.
//! - **Never-invent.** `source_turn` comes only from metadata `source_turn`
//!   (preferred) or `turn_id` when it is a valid non-negative JSON integer;
//!   anything else is `None`. `position` is computed, never read.

use std::collections::{BTreeSet, HashMap};

use crate::base::error::Result;

/// The single source of the persisted `memory_entities.entity_name_norm`
/// key: Unicode lowercase via Rust `str::to_lowercase()` — deliberately
/// NOT full Unicode case folding (no `ß` -> `ss`, no locale tailoring),
/// in lockstep with `crate::graph::tokenize`, which lowercases the same
/// way.
///
/// This MUST stay in lockstep with `crate::graph::tokenize`'s lowercasing:
/// entity↔text matching is DEFINED by the tokenizer, and the stored key
/// exists precisely so SQL can perform that match through an index —
/// SQL `LOWER()` is ASCII-only and is NOT an acceptable substitute. Every
/// `INSERT INTO memory_entities` writer must bind this value
/// (engine/record.rs, engine/stats.rs, engine/graph_ops.rs) and then call
/// [`repair_entity_norm`]; open()'s `entity_norm_backfill` stage uses it
/// for pre-v49 rows; the enforcement census in `thread_tests` fails if
/// any writer forgets.
pub(crate) fn normalize_entity_name(name: &str) -> String {
    name.to_lowercase()
}

/// Write-time self-heal for the persisted normalized key. `INSERT OR
/// IGNORE` never touches a pre-existing `(memory_rid, entity_name)` row,
/// so a row written before v49 — or carrying a stale norm value — would
/// otherwise NEVER be repaired by later natural writes, and the
/// normalized-key invariant would only hold for stores that started
/// clean. Every `INSERT OR IGNORE INTO memory_entities` writer calls this
/// right after its insert. The UPDATE's predicate makes it a no-op on
/// already-correct rows (including the row the insert itself just
/// wrote), and it deliberately does NOT touch `entities.mention_count` —
/// first-mention accounting stays exactly on the insert's `changes()`
/// result (engine/stats.rs).
pub(crate) fn repair_entity_norm(
    conn: &rusqlite::Connection,
    memory_rid: &str,
    entity_name: &str,
) -> rusqlite::Result<()> {
    let norm = normalize_entity_name(entity_name);
    conn.execute(
        "UPDATE memory_entities SET entity_name_norm = ?3 \
         WHERE memory_rid = ?1 AND entity_name = ?2 \
           AND (entity_name_norm IS NULL OR entity_name_norm != ?3)",
        rusqlite::params![memory_rid, entity_name, norm],
    )?;
    Ok(())
}

/// One row of a coverage-first thread, in chronological order.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ThreadItem {
    pub rid: String,
    /// Decrypted memory text.
    pub text: String,
    /// Transaction/ingestion time (tracks conversation order under
    /// conversational ingestion).
    pub created_at: f64,
    /// From metadata `source_turn` or `turn_id`, only when a valid
    /// non-negative integer; never invented.
    pub source_turn: Option<i64>,
    /// 1-based chronological position within the FULL eligible thread
    /// (pre-truncation).
    pub position: usize,
    /// Which of the REQUESTED entities this row matched (in request order,
    /// case-insensitive duplicates collapsed to the first spelling).
    pub entities: Vec<String>,
}

/// Result of [`crate::YantrikDB::recall_thread`]: the chronological items
/// plus loud truncation accounting — silent truncation would be a coverage
/// contract violation, so `total`/`omitted` are part of the type.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ThreadRecall {
    /// Chronological ascending.
    pub items: Vec<ThreadItem>,
    /// Total eligible before truncation.
    pub total: usize,
    /// `total - items.len()`.
    pub omitted: usize,
}

impl crate::YantrikDB {
    /// Coverage-first thread retrieval over the `memory_entities` join —
    /// see the module docs for the full contract. Opt-in; existing `recall`
    /// behavior is untouched.
    ///
    /// Empty `entities`, an unknown namespace, or entities with no linked
    /// rows all yield an empty `ThreadRecall` (total 0), not an error; an
    /// unknown entity among known ones contributes nothing.
    pub fn recall_thread(
        &self,
        namespace: &str,
        entities: &[&str],
        limit: usize,
    ) -> Result<ThreadRecall> {
        let empty = || ThreadRecall {
            items: Vec::new(),
            total: 0,
            omitted: 0,
        };
        if entities.is_empty() {
            return Ok(empty());
        }

        // Requested-name index, keyed by the persisted normalization (the
        // tokenizer's Unicode lowercase — see normalize_entity_name). Duplicated
        // requests ("Alpha", "alpha") collapse to the first spelling so a
        // row never lists the same entity twice.
        let mut req_by_lower: HashMap<String, usize> = HashMap::new();
        for (i, e) in entities.iter().enumerate() {
            req_by_lower.entry(normalize_entity_name(e)).or_insert(i);
        }

        struct RowAgg {
            text: String,
            created_at: f64,
            metadata: Option<String>,
            matched: BTreeSet<usize>,
        }
        let mut by_rid: HashMap<String, RowAgg> = HashMap::new();

        {
            let conn = self.read_conn();

            // v49: requested names resolve directly against the PERSISTED
            // normalized key (entity_name_norm — stamped by every writer,
            // backfilled at open; see normalize_entity_name). One indexed
            // lookup on idx_memory_entities_norm over a small fixed
            // parameter list — the pre-v49 O(V) DISTINCT vocabulary scan
            // (Unicode-lowercased in Rust per request) is retired.
            let mut norm_names: Vec<&String> = req_by_lower.keys().collect();
            norm_names.sort(); // deterministic parameter order

            let placeholders: String = (0..norm_names.len())
                .map(|i| format!("?{}", i + 2))
                .collect::<Vec<_>>()
                .join(",");
            // Visibility predicates mirrored from recall's default read
            // path (see module docs); the supersedes exclusion runs below,
            // outside the conn guard.
            let sql = format!(
                "SELECT m.rid, m.text, m.created_at, m.metadata, me.entity_name_norm \
                 FROM memories m \
                 JOIN memory_entities me ON me.memory_rid = m.rid \
                 WHERE m.namespace = ?1 \
                   AND m.consolidation_status = 'active' \
                   AND (m.synthesis_state IS NULL OR m.synthesis_state = 'verified') \
                   AND me.entity_name_norm IN ({placeholders})"
            );
            let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
                Vec::with_capacity(norm_names.len() + 1);
            param_values.push(Box::new(namespace.to_string()));
            for name in &norm_names {
                param_values.push(Box::new((*name).clone()));
            }
            let params_ref: Vec<&dyn rusqlite::types::ToSql> =
                param_values.iter().map(|p| p.as_ref()).collect();

            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map(params_ref.as_slice(), |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, f64>(2)?,
                    r.get::<_, Option<String>>(3)?,
                    r.get::<_, String>(4)?,
                ))
            })?;
            for row in rows {
                let (rid, text, created_at, metadata, entity_norm) = row?;
                let idx = req_by_lower[&entity_norm];
                by_rid
                    .entry(rid)
                    .or_insert_with(|| RowAgg {
                        text,
                        created_at,
                        metadata,
                        matched: BTreeSet::new(),
                    })
                    .matched
                    .insert(idx);
            }
        } // conn guard dropped — superseded_rids_among acquires its own.

        // Recall step 3.4's eligibility rule, mirrored: when the status
        // read policy is active, a record with a selected active
        // `supersedes` successor is not visible.
        if self.status_read_policy() && !by_rid.is_empty() {
            let rids: Vec<String> = by_rid.keys().cloned().collect();
            let rid_refs: Vec<&str> = rids.iter().map(String::as_str).collect();
            let superseded = self.superseded_rids_among(&rid_refs)?;
            for rid in superseded {
                by_rid.remove(&rid);
            }
        }

        // Hydrate: decrypt (same self.decrypt_text path the facet lane
        // uses) and read source_turn under the never-invent rule.
        fn valid_turn(meta: &serde_json::Value, key: &str) -> Option<i64> {
            meta.get(key)?.as_i64().filter(|t| *t >= 0)
        }
        let mut eligible: Vec<(String, String, f64, Option<i64>, BTreeSet<usize>)> =
            Vec::with_capacity(by_rid.len());
        for (rid, agg) in by_rid {
            let text = self.decrypt_text(&agg.text)?;
            let source_turn = match agg.metadata.as_deref() {
                None | Some("") => None,
                Some(stored_meta) => {
                    let metadata = self.decrypt_text(stored_meta)?;
                    serde_json::from_str::<serde_json::Value>(&metadata)
                        .ok()
                        .and_then(|meta| {
                            valid_turn(&meta, "source_turn")
                                .or_else(|| valid_turn(&meta, "turn_id"))
                        })
                }
            };
            eligible.push((rid, text, agg.created_at, source_turn, agg.matched));
        }

        // Deterministic total order: created_at asc, then source_turn
        // (NULLS LAST) within equal created_at, then rid.
        eligible.sort_by(|a, b| {
            a.2.total_cmp(&b.2)
                .then_with(|| match (a.3, b.3) {
                    (Some(x), Some(y)) => x.cmp(&y),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                })
                .then_with(|| a.0.cmp(&b.0))
        });

        let total = eligible.len();
        let mut items = Vec::with_capacity(total.min(limit));
        for (pos0, (rid, text, created_at, source_turn, matched)) in
            eligible.into_iter().enumerate()
        {
            if pos0 >= limit {
                break; // earliest-prefix truncation; accounted below.
            }
            items.push(ThreadItem {
                rid,
                text,
                created_at,
                source_turn,
                position: pos0 + 1,
                entities: matched
                    .into_iter()
                    .map(|i| entities[i].to_string())
                    .collect(),
            });
        }
        let omitted = total - items.len();
        Ok(ThreadRecall {
            items,
            total,
            omitted,
        })
    }
}

#[cfg(test)]
mod thread_tests {
    use crate::YantrikDB;

    const NS: &str = "n";
    const BASE_MICROS: i64 = 1_700_000_000_000_000;

    fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim).map(|i| (seed + i as f32) * 0.1).collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    }

    /// Deterministic seeded write: caller-controlled created_at and entity
    /// links via record_with_rid's `extracted_entities` (persisted by the
    /// materializer — drain() makes it synchronous for asserts).
    #[allow(clippy::too_many_arguments)]
    fn seed_row(
        db: &YantrikDB,
        rid: &str,
        text: &str,
        metadata: &serde_json::Value,
        created_at_micros: i64,
        entity_names: &[&str],
        seed: f32,
    ) {
        db.record_with_rid(
            rid,
            text,
            "episodic",
            0.5,
            0.0,
            604800.0,
            metadata,
            &vec_seed(seed, 8),
            NS,
            0.8,
            "general",
            "user",
            None,
            created_at_micros,
            entity_names,
            "test-model.v1",
            None,
            crate::provenance::WriteAdmission::Admitted,
        )
        .unwrap();
    }

    /// Drain the pending materializer queue inline (Phase 4.3: entity
    /// persistence for record_with_rid is applied by the materializer).
    fn drain(db: &YantrikDB) {
        for _ in 0..50 {
            if db.apply_pending_ops_once(500).unwrap() == 0 {
                return;
            }
        }
        panic!("pending ops did not drain");
    }

    fn meta_empty() -> serde_json::Value {
        serde_json::json!({})
    }

    /// (a) COVERAGE PIN — the reason this lane exists. 60 alpha rows
    /// interleaved with 60 beta-only rows: recall_thread must return
    /// EXACTLY the 60 alpha rows, in created_at order, positions 1..60 —
    /// including every row a similarity pool bounded at top_k=10 could
    /// never surface. Coverage is asserted by count + order + positions.
    #[test]
    fn coverage_pin_returns_every_thread_member_in_order() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let mut alpha_rids = Vec::new();
        for i in 0..120 {
            let rid = format!("r{i:03}");
            let micros = BASE_MICROS + (i as i64) * 1_000_000;
            if i % 2 == 0 {
                seed_row(
                    &db,
                    &rid,
                    &format!("Alpha update number {i}"),
                    &meta_empty(),
                    micros,
                    &["Alpha"],
                    i as f32,
                );
                alpha_rids.push(rid);
            } else {
                seed_row(
                    &db,
                    &rid,
                    &format!("Beta update number {i}"),
                    &meta_empty(),
                    micros,
                    &["Beta"],
                    i as f32,
                );
            }
        }
        drain(&db);

        // Requested lowercase against stored "Alpha": the lane must match
        // the tokenizer's case-insensitivity, not the stored spelling.
        let out = db.recall_thread(NS, &["alpha"], 100).unwrap();
        assert_eq!(out.total, 60, "eligible set is ALL 60 alpha rows");
        assert_eq!(out.omitted, 0);
        assert_eq!(out.items.len(), 60);
        for (i, item) in out.items.iter().enumerate() {
            assert_eq!(item.rid, alpha_rids[i], "chronological (insertion) order");
            assert_eq!(item.position, i + 1, "positions 1..=60");
            assert_eq!(item.entities, vec!["alpha".to_string()]);
            assert!(item.text.contains("Alpha update"), "decrypted text");
            if i > 0 {
                assert!(
                    out.items[i - 1].created_at < item.created_at,
                    "created_at strictly ascending"
                );
            }
        }
    }

    /// (b) Equal created_at: source_turn ascending first (from either
    /// `source_turn` or `turn_id`), turn-bearing before turn-less (NULLS
    /// LAST), rid as the final tie-break; invalid (negative / non-integer)
    /// turns are None, never invented.
    #[test]
    fn turn_tie_break_within_equal_created_at() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let t = BASE_MICROS;
        seed_row(
            &db,
            "m_turn7",
            "Gamma event seven",
            &serde_json::json!({"source_turn": 7}),
            t,
            &["Gamma"],
            1.0,
        );
        seed_row(
            &db,
            "a_turn3",
            "Gamma event three",
            &serde_json::json!({"turn_id": 3}),
            t,
            &["Gamma"],
            2.0,
        );
        seed_row(
            &db,
            "a_none",
            "Gamma event with no turn",
            &meta_empty(),
            t,
            &["Gamma"],
            3.0,
        );
        seed_row(
            &db,
            "z_neg",
            "Gamma event negative turn",
            &serde_json::json!({"source_turn": -2}),
            t,
            &["Gamma"],
            4.0,
        );
        seed_row(
            &db,
            "b_str",
            "Gamma event string turn",
            &serde_json::json!({"source_turn": "5"}),
            t,
            &["Gamma"],
            5.0,
        );
        // Later created_at dominates any turn value: turn 0 sorts LAST.
        seed_row(
            &db,
            "later_turn0",
            "Gamma event later",
            &serde_json::json!({"source_turn": 0}),
            t + 1_000_000,
            &["Gamma"],
            6.0,
        );
        drain(&db);

        let out = db.recall_thread(NS, &["Gamma"], 10).unwrap();
        let rids: Vec<&str> = out.items.iter().map(|i| i.rid.as_str()).collect();
        assert_eq!(
            rids,
            vec![
                "a_turn3",
                "m_turn7",
                "a_none",
                "b_str",
                "z_neg",
                "later_turn0"
            ],
            "turn asc, then NULLS LAST by rid, then created_at dominates"
        );
        let turns: Vec<Option<i64>> = out.items.iter().map(|i| i.source_turn).collect();
        assert_eq!(
            turns,
            vec![Some(3), Some(7), None, None, None, Some(0)],
            "invalid turns are None — never invented"
        );
        assert_eq!(
            out.items.iter().map(|i| i.position).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6]
        );
    }

    /// (c) Truncation keeps the EARLIEST rows, exposes omitted, and the
    /// positions still reflect the full-thread numbering (1-based over the
    /// pre-truncation order — an earliest-prefix, so 1..=limit).
    #[test]
    fn truncation_keeps_earliest_and_reports_omitted() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        for i in 0..10 {
            seed_row(
                &db,
                &format!("t{i}"),
                &format!("Alpha step {i}"),
                &meta_empty(),
                BASE_MICROS + (i as i64) * 1_000_000,
                &["Alpha"],
                i as f32,
            );
        }
        drain(&db);

        let out = db.recall_thread(NS, &["Alpha"], 4).unwrap();
        assert_eq!(out.total, 10);
        assert_eq!(out.omitted, 6);
        assert_eq!(
            out.items.iter().map(|i| i.rid.as_str()).collect::<Vec<_>>(),
            vec!["t0", "t1", "t2", "t3"],
            "the earliest are kept, never a similarity sample"
        );
        assert_eq!(
            out.items.iter().map(|i| i.position).collect::<Vec<_>>(),
            vec![1, 2, 3, 4],
            "full-thread numbering"
        );
    }

    /// (d) Multi-entity: a row matched by either entity appears ONCE;
    /// `entities` lists exactly the matched subset of the request, in
    /// request order; duplicate spellings of one entity collapse.
    #[test]
    fn multi_entity_rows_appear_once_with_matched_subset() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        seed_row(
            &db,
            "d1",
            "Alpha alone here",
            &meta_empty(),
            BASE_MICROS,
            &["Alpha"],
            1.0,
        );
        seed_row(
            &db,
            "d2",
            "Alpha met Beta today",
            &meta_empty(),
            BASE_MICROS + 1_000_000,
            &["Alpha", "Beta"],
            2.0,
        );
        seed_row(
            &db,
            "d3",
            "Beta alone here",
            &meta_empty(),
            BASE_MICROS + 2_000_000,
            &["Beta"],
            3.0,
        );
        drain(&db);

        let out = db.recall_thread(NS, &["Alpha", "Beta"], 10).unwrap();
        assert_eq!(out.total, 3);
        assert_eq!(out.items.len(), 3, "d2 appears exactly once");
        assert_eq!(out.items[0].entities, vec!["Alpha".to_string()]);
        assert_eq!(
            out.items[1].entities,
            vec!["Alpha".to_string(), "Beta".to_string()]
        );
        assert_eq!(out.items[2].entities, vec!["Beta".to_string()]);

        // Duplicate spellings collapse to the first requested form.
        let dup = db.recall_thread(NS, &["Alpha", "alpha"], 10).unwrap();
        assert_eq!(dup.items[0].entities, vec!["Alpha".to_string()]);

        // (5) Empty request / unknown entity are empty results, not errors.
        let none = db.recall_thread(NS, &[], 10).unwrap();
        assert_eq!((none.total, none.omitted, none.items.len()), (0, 0, 0));
        let unknown = db.recall_thread(NS, &["Nobody"], 10).unwrap();
        assert_eq!(unknown.total, 0);
        // Unknown namespace: empty, not an error.
        assert_eq!(
            db.recall_thread("other_ns", &["Alpha"], 10).unwrap().total,
            0
        );
    }

    /// (e) Visibility: a forgotten (tombstoned) row leaves the thread, and
    /// a superseded row leaves it under the status read policy — the same
    /// predicates recall applies.
    #[test]
    fn tombstoned_and_superseded_rows_are_excluded() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        for (i, rid) in ["v1", "v2", "v3"].iter().enumerate() {
            seed_row(
                &db,
                rid,
                &format!("Alpha visibility row {i}"),
                &meta_empty(),
                BASE_MICROS + (i as i64) * 1_000_000,
                &["Alpha"],
                i as f32,
            );
        }
        drain(&db);
        assert_eq!(db.recall_thread(NS, &["Alpha"], 10).unwrap().total, 3);

        assert!(db.forget("v2").unwrap());
        let out = db.recall_thread(NS, &["Alpha"], 10).unwrap();
        assert_eq!(out.total, 2, "forgotten row excluded");
        assert_eq!(
            out.items.iter().map(|i| i.rid.as_str()).collect::<Vec<_>>(),
            vec!["v1", "v3"]
        );
        assert_eq!(
            out.items.iter().map(|i| i.position).collect::<Vec<_>>(),
            vec![1, 2],
            "positions renumber over the eligible thread"
        );

        // v3 supersedes v1 (edge direction new→old). Fresh DBs default the
        // status read policy ON, so the target drops from the thread.
        assert!(db.status_read_policy());
        db.link(
            "v3",
            &crate::types::RecordLink {
                target_rid: "v1".to_string(),
                link_type: crate::types::LinkType::Supersedes,
            },
        )
        .unwrap();
        let out = db.recall_thread(NS, &["Alpha"], 10).unwrap();
        assert_eq!(out.total, 1, "superseded row excluded");
        assert_eq!(out.items[0].rid, "v3");
        assert_eq!(out.items[0].position, 1);
    }

    /// (f) Replication parity: after apply_ops the follower answers
    /// recall_thread identically. Follower entity-join rows come from
    /// apply_ops' backfill_memory_entities (distributed/replication.rs),
    /// which scans memory TEXT against the entities table — so entities
    /// must arrive via a replicated `relate` op and row texts must carry
    /// the entity tokens (the production shape).
    #[test]
    fn replication_parity_on_follower() {
        use crate::replication::{apply_ops, extract_ops_since};

        let leader = YantrikDB::new(":memory:", 8).unwrap();
        for i in 0..6 {
            let (name, tag) = if i % 2 == 0 {
                ("Alpha", "milestone")
            } else {
                ("Beta", "sync")
            };
            seed_row(
                &leader,
                &format!("p{i}"),
                &format!("{name} {tag} number {i}"),
                &serde_json::json!({"source_turn": i}),
                BASE_MICROS + (i as i64) * 1_000_000,
                &[name],
                i as f32,
            );
        }
        // The relate op replicates the entities themselves; without it the
        // follower's backfill has no entity vocabulary to scan for.
        leader.relate("Alpha", "Beta", "related_to", 1.0).unwrap();
        drain(&leader);

        let leader_out = leader.recall_thread(NS, &["Alpha"], 100).unwrap();
        assert_eq!(leader_out.total, 3, "leader thread complete");

        let follower = YantrikDB::new(":memory:", 8).unwrap();
        let ops = extract_ops_since(&leader.conn(), None, None, None, 1000).unwrap();
        apply_ops(&follower, &ops).unwrap();

        let follower_out = follower.recall_thread(NS, &["Alpha"], 100).unwrap();
        assert_eq!(
            follower_out, leader_out,
            "same thread on both sides — items, positions, turns, totals"
        );
        // And the multi-entity view converges too.
        assert_eq!(
            follower.recall_thread(NS, &["Alpha", "Beta"], 100).unwrap(),
            leader.recall_thread(NS, &["Alpha", "Beta"], 100).unwrap()
        );
    }

    /// (g) ENFORCEMENT CENSUS — the normalized-key invariant. Rows reach
    /// `memory_entities` through record (heuristic extraction via the
    /// materializer), relate() (entity creation + text backfill), and
    /// replication apply on a follower (apply_ops -> materialize +
    /// backfill_memory_entities). After exercising all three natural
    /// paths, EVERY row on BOTH sides must carry entity_name_norm ==
    /// normalize_entity_name(entity_name). A writer that forgets the
    /// binding leaves NULL (or a diverging value) and fails here.
    #[test]
    fn every_writer_stamps_the_normalized_entity_key() {
        use crate::engine::thread::normalize_entity_name;
        use crate::replication::{apply_ops, extract_ops_since};

        let leader = YantrikDB::new(":memory:", 8).unwrap();
        // Natural path 1: record with entity-bearing text (non-ASCII
        // included — the exact case SQL LOWER() would corrupt).
        for i in 0..4 {
            seed_row(
                &leader,
                &format!("c{i}"),
                &format!("Münster planning with Alpha and Beta round {i}"),
                &meta_empty(),
                BASE_MICROS + (i as i64) * 1_000_000,
                &["Münster", "Alpha"],
                i as f32,
            );
        }
        // Natural path 2: relate() — creates entities and backfills
        // memory_entities from row text (graph_ops.rs).
        leader.relate("Alpha", "Beta", "related_to", 1.0).unwrap();
        leader.relate("Münster", "Beta", "located_in", 0.7).unwrap();
        drain(&leader);

        // Natural path 3: replication apply to a follower.
        let follower = YantrikDB::new(":memory:", 8).unwrap();
        let ops = extract_ops_since(&leader.conn(), None, None, None, 1000).unwrap();
        apply_ops(&follower, &ops).unwrap();

        let census = |db: &YantrikDB, side: &str| {
            let conn = db.conn();
            let rows: Vec<(String, Option<String>)> = conn
                .prepare("SELECT entity_name, entity_name_norm FROM memory_entities")
                .unwrap()
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
                .unwrap()
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap();
            assert!(
                !rows.is_empty(),
                "{side}: census precondition — the natural paths must have \
                 produced memory_entities rows"
            );
            let bad = rows
                .iter()
                .filter(|(name, norm)| {
                    norm.as_deref() != Some(normalize_entity_name(name).as_str())
                })
                .count();
            assert_eq!(
                bad, 0,
                "{side}: a writer inserted memory_entities without the normalized \
                 key; every writer must bind normalize_entity_name()"
            );
        };
        census(&leader, "leader");
        census(&follower, "follower");
    }

    /// (h) Non-ASCII case-insensitive resolution via the indexed path: the
    /// stored spelling is 'MÜNSTER'; the request is 'münster'. Under Rust
    /// to_lowercase both fold to 'münster'; under SQL LOWER() the stored
    /// key would have been 'mÜnster' and the lookup would return nothing.
    #[test]
    fn non_ascii_entity_resolves_case_insensitively_via_indexed_path() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        seed_row(
            &db,
            "m_de",
            "Planning the MÜNSTER rollout",
            &meta_empty(),
            BASE_MICROS,
            &["MÜNSTER"],
            1.0,
        );
        drain(&db);

        let out = db.recall_thread(NS, &["münster"], 10).unwrap();
        assert_eq!(
            out.total, 1,
            "Unicode fold must match, ASCII fold would miss"
        );
        assert_eq!(out.items[0].rid, "m_de");
        assert_eq!(out.items[0].position, 1);
        assert_eq!(out.items[0].entities, vec!["münster".to_string()]);
    }
    /// (i) SELF-HEAL ON WRITE — INSERT OR IGNORE never touches an existing
    /// row, so a pre-v49 (or corrupted) NULL-norm row must be repaired by
    /// the conditional UPDATE every writer runs after its insert
    /// (repair_entity_norm). Seed the bad rows manually, then drive two
    /// natural writer paths over the same (rid, entity) pairs.
    #[test]
    fn natural_writes_self_heal_a_null_normalized_key() {
        use crate::engine::thread::normalize_entity_name;
        let db = YantrikDB::new(":memory:", 8).unwrap();

        // Path A: the record_with_rid materializer (engine/stats.rs). The
        // memory_entities row pre-exists with NULL norm; the recorded
        // memory then claims the same (rid, entity) and must repair it.
        db.conn()
            .execute(
                "INSERT INTO memory_entities (memory_rid, entity_name) \
                 VALUES ('m_pre', 'Münster')",
                [],
            )
            .unwrap();
        seed_row(
            &db,
            "m_pre",
            "Münster status update",
            &meta_empty(),
            BASE_MICROS,
            &["Münster"],
            1.0,
        );
        drain(&db);

        // Path B: link_memory_entity (engine/graph_ops.rs).
        db.conn()
            .execute(
                "INSERT INTO memory_entities (memory_rid, entity_name) \
                 VALUES ('m_pre2', 'Alpha')",
                [],
            )
            .unwrap();
        db.link_memory_entity("m_pre2", "Alpha").unwrap();

        let norm = |rid: &str, name: &str| -> Option<String> {
            db.conn()
                .query_row(
                    "SELECT entity_name_norm FROM memory_entities \
                     WHERE memory_rid = ?1 AND entity_name = ?2",
                    [rid, name],
                    |r| r.get(0),
                )
                .unwrap()
        };
        assert_eq!(
            norm("m_pre", "Münster").as_deref(),
            Some(normalize_entity_name("Münster").as_str()),
            "the record materializer must repair a pre-existing NULL norm"
        );
        assert_eq!(
            norm("m_pre2", "Alpha").as_deref(),
            Some("alpha"),
            "link_memory_entity must repair a pre-existing NULL norm"
        );
    }
}
