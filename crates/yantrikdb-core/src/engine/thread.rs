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
//!
//! Thread v2 (`recall_thread_v2`) extends the lane with three UNION-ed
//! query routes — entities (the v1 route), literal FTS phrases, and
//! already-resolved topic synthesis rids joined through
//! `synthesis_dependencies` — with per-item route provenance, all inside
//! ONE read-transaction snapshot. Cost model, explicitly: the COUNT and
//! positions range over the FULL eligible union (coverage must be exact
//! and truncation loud — anchor counts are capped but one anchor can
//! match arbitrarily many rows, so the union itself is unbounded and
//! SQLite may sort all of it), while DECRYPTION is bounded by the
//! returned page (`LIMIT`) — the union is never decrypted. The ordering
//! reads the persisted v50 `source_turn` column (stamped by every
//! metadata-persisting writer through [`extract_source_turn`]); on
//! encrypted stores the completeness marker + `maintain_source_turn_backfill`
//! guard that order rather than assume it.
//!
//! v1 vs v2 (final compatibility decision): v1 `recall_thread` KEEPS its
//! own pre-v2 implementation path — decrypt-derived turn ordering over
//! the fully materialized eligible set, correct regardless of
//! column/marker state, no input caps, never `MaintenanceRequired`.
//! Strict marker gating and the bounded-page cost model are exclusively
//! v2's contract. Both paths derive turns through the ONE shared
//! [`extract_source_turn`], so on a healthy store they order identically.

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

/// The single source of the persisted `memories.source_turn` column (v50):
/// metadata key `source_turn` (preferred) falling back to `turn_id`, a
/// valid non-negative JSON **integer** only — a float `5.0`, a string
/// `"5"`, or a negative value is `None`, and the fallback applies whenever
/// `source_turn` is absent OR invalid (`.or_else` semantics), exactly the
/// never-invent rule this module has always applied at read time.
///
/// Stamping (all nine metadata-persisting memories writers — the same
/// sites the v48 `event_time_bounds` work marked), the open()-time
/// `source_turn_backfill` stage, `maintain_source_turn_backfill`'s
/// decrypt-and-stamp, replication's legacy-payload fallback, and v2 reads
/// ALL go through this one function; a second implementation of the
/// fallback chain (SQL or Rust) is a drift bug by construction.
pub(crate) fn extract_source_turn(metadata: &serde_json::Value) -> Option<i64> {
    fn valid_turn(meta: &serde_json::Value, key: &str) -> Option<i64> {
        meta.get(key)?.as_i64().filter(|t| *t >= 0)
    }
    valid_turn(metadata, "source_turn").or_else(|| valid_turn(metadata, "turn_id"))
}

/// Meta key of the v50 source_turn completeness marker: `'1'` means every
/// row's `source_turn` column faithfully mirrors its (plaintext) metadata,
/// `'0'` means an encrypted store still owes `maintain_source_turn_backfill`
/// work (or a raw SQL write staled the store — see the
/// `memories_source_turn_marker_*` triggers in base/schema.rs).
pub(crate) const SOURCE_TURN_MARKER_KEY: &str = "source_turn_backfill_complete";

/// Meta key of the invalidation epoch: bumped by the schema triggers on
/// every memories INSERT / metadata-or-source_turn UPDATE. The repair
/// cursor is stamped with the epoch it was taken under; a bump from a raw
/// SQL write makes any in-flight cursor stale, forcing the repair scan to
/// restart from rowid 0 rather than certify rows a raw write mutated
/// behind it. Engine-supported stamped writes restore the epoch together
/// with the marker (they stamped faithfully, so nothing needs rescanning).
pub(crate) const SOURCE_TURN_EPOCH_KEY: &str = "source_turn_invalidation_epoch";

/// Pre-write snapshot of the completeness marker AND the invalidation
/// epoch, taken under the serialized writer lock by every engine-supported
/// write that stamps `source_turn` from the plaintext metadata it
/// serializes. The schema's invalidation triggers flip the marker to `'0'`
/// and bump the epoch on ANY memories INSERT / metadata-or-source_turn
/// UPDATE (raw SQL writes stale the store); a stamping engine write then
/// calls [`marker_restore`] with this snapshot so its own trigger fire
/// does not count as staleness: true stays true, false is never waived by
/// a normal write, and a resumable repair cursor is not needlessly reset.
#[derive(Debug, Clone)]
pub(crate) struct MarkerSnapshot {
    marker: Option<String>,
    epoch: Option<String>,
}

fn meta_get(conn: &rusqlite::Connection, key: &str) -> rusqlite::Result<Option<String>> {
    match conn.query_row(
        "SELECT value FROM meta WHERE key = ?1",
        rusqlite::params![key],
        |r| r.get(0),
    ) {
        Ok(v) => Ok(Some(v)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e),
    }
}

fn meta_put(
    conn: &rusqlite::Connection,
    key: &str,
    value: &Option<String>,
) -> rusqlite::Result<()> {
    match value {
        Some(v) => {
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
                rusqlite::params![key, v],
            )?;
        }
        None => {
            conn.execute("DELETE FROM meta WHERE key = ?1", rusqlite::params![key])?;
        }
    }
    Ok(())
}

/// See [`MarkerSnapshot`].
pub(crate) fn marker_snapshot(conn: &rusqlite::Connection) -> rusqlite::Result<MarkerSnapshot> {
    Ok(MarkerSnapshot {
        marker: meta_get(conn, SOURCE_TURN_MARKER_KEY)?,
        epoch: meta_get(conn, SOURCE_TURN_EPOCH_KEY)?,
    })
}

/// Restore the marker + epoch to their pre-write state (see
/// [`MarkerSnapshot`]). `None` means the key did not exist before the
/// write (a pre-v50 store mid-upgrade) and is removed again rather than
/// invented.
pub(crate) fn marker_restore(
    conn: &rusqlite::Connection,
    prior: &MarkerSnapshot,
) -> rusqlite::Result<()> {
    meta_put(conn, SOURCE_TURN_MARKER_KEY, &prior.marker)?;
    meta_put(conn, SOURCE_TURN_EPOCH_KEY, &prior.epoch)?;
    Ok(())
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

/// One row of a v2 multi-route thread. Deliberately a DISTINCT type from
/// [`ThreadItem`] (reviewer item 7): adding fields to the v1 type would be
/// source-breaking for downstream struct literals/patterns, so v1 stays
/// byte-for-byte unchanged and v2 carries the richer provenance here.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ThreadItemV2 {
    pub rid: String,
    /// Decrypted memory text.
    pub text: String,
    /// Transaction/ingestion time (tracks conversation order under
    /// conversational ingestion).
    pub created_at: f64,
    /// The persisted v50 column (stamped from metadata `source_turn` /
    /// `turn_id` by every writer via [`extract_source_turn`]); never
    /// invented.
    pub source_turn: Option<i64>,
    /// 1-based chronological position within the FULL eligible union
    /// (pre-truncation).
    pub position: usize,
    /// Which of the REQUESTED entities this row matched (in request order,
    /// case-insensitive duplicates collapsed to the first spelling — the
    /// v1 rule, unchanged).
    pub entities: Vec<String>,
    /// Route provenance: ALL of the query's routes this row matched, as a
    /// deterministic, stable-ordered subset of
    /// `["entity", "phrase", "topic"]` — a row found by both an entity
    /// and a phrase lists both, in that fixed order, regardless of
    /// request shape.
    pub routes: Vec<&'static str>,
    /// Per-anchor provenance for the phrase route: the REQUESTED phrases,
    /// verbatim as requested, that matched this row — in REQUEST ORDER,
    /// with duplicate request items deduplicated first-occurrence-wins
    /// BEFORE matching (so a phrase requested twice appears at most once).
    pub phrases: Vec<String>,
    /// Per-anchor provenance for the topic route: the requested topic
    /// synthesis rids whose direct evidence includes this row — same
    /// determinism rule as `phrases` (request order, first-occurrence
    /// dedup).
    pub topic_rids: Vec<String>,
}

/// Result of [`crate::YantrikDB::recall_thread_v2`]: chronological items
/// with per-anchor provenance plus EXPLICIT accounting — `total`,
/// `returned`, and `omitted` are all real serialized fields (the v2
/// contract reports all three; `returned == items.len()` by construction
/// and is pinned by test, never left implied).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ThreadRecallV2 {
    /// Chronological ascending (created_at, source_turn NULLS LAST, rid).
    pub items: Vec<ThreadItemV2>,
    /// Total eligible before truncation (the full union count).
    pub total: usize,
    /// `items.len()` — the page actually returned.
    pub returned: usize,
    /// `total - returned`.
    pub omitted: usize,
}

/// The v2 multi-route thread query: any non-empty route contributes to the
/// eligible set (SQL UNION, deduped); all three empty is a valid query
/// with an empty result, not an error.
///
/// Caps (typed `InvalidInput` on violation): at most 64 entities, 32
/// phrases, 16 topic rids; every item 1..=512 bytes.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct ThreadQuery {
    /// Entity names, matched case-insensitively through the persisted
    /// `entity_name_norm` key (see [`normalize_entity_name`]).
    pub entities: Vec<String>,
    /// Literal phrases for the `memories_fts` route. Each phrase is
    /// escaped as ONE FTS5 string literal (embedded `"` doubled) — never
    /// interpreted as FTS query syntax. Unavailable on encrypted stores
    /// (typed `CapabilityUnavailable`, fail closed).
    pub phrases: Vec<String>,
    /// Already-resolved topic synthesis rids: direct evidence joins
    /// through `synthesis_dependencies` (`is_direct = 1`,
    /// namespace-scoped). A rid that does not resolve to a visible row in
    /// this namespace is a typed error — identical for nonexistent and
    /// cross-namespace rids, so existence never leaks across tenants.
    pub topic_rids: Vec<String>,
}

/// Progress of one [`crate::YantrikDB::maintain_source_turn_backfill`]
/// call: how many candidate rows this batch examined, how many remain
/// beyond the persisted cursor, and whether the store is now complete
/// (the completeness marker was set).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct MaintenanceProgress {
    /// Candidate rows examined (decrypted and, where a valid turn was
    /// found, stamped) by this call.
    pub processed: usize,
    /// Candidate rows still beyond the cursor after this call.
    pub remaining: usize,
    /// True when nothing remains: the marker is now `'1'` and the strict
    /// ordering gate is satisfied.
    pub complete: bool,
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

/// Meta key of the resumable repair cursor: `"{epoch}:{rowid}"` — the
/// invalidation epoch the scan started under plus the last examined
/// rowid. A later epoch (a raw SQL write fired the triggers) makes the
/// cursor stale and the scan restarts from rowid 0; cleared on
/// completion.
pub(crate) const SOURCE_TURN_CURSOR_KEY: &str = "source_turn_repair_cursor";

/// One batch of the v50 source_turn RECOMPUTE pass — the single
/// implementation behind open()'s `source_turn_backfill` stage
/// (unencrypted stores, identity decrypt) and
/// [`crate::YantrikDB::maintain_source_turn_backfill`] (decrypt-and-stamp).
///
/// This is a full recompute, not a NULL-fill (reviewer blocker): raw SQL
/// can CHANGE metadata (turn 5 -> 7, or 5 -> absent), leaving a stale
/// NON-NULL scalar behind — so every row beyond the cursor is compared
/// against the shared extractor's output on its CURRENT metadata, and a
/// mismatch is rewritten in either direction, INCLUDING back to NULL when
/// the metadata no longer carries a valid turn. The completeness marker is
/// set only when a full pass drains (`remaining == 0`), never after a
/// partial or NULL-only sweep.
///
/// Resumability: the cursor persists in meta as `"{epoch}:{rowid}"`, one
/// transaction per call. The epoch half is the trigger-bumped
/// [`SOURCE_TURN_EPOCH_KEY`]: a raw write mid-scan bumps it, the stored
/// cursor no longer matches, and the next call restarts from rowid 0
/// rather than certifying rows mutated behind the scan. The scan's own
/// UPDATEs also fire the trigger, so the cursor is stamped with the epoch
/// read AFTER this batch's updates, inside the same serialized-writer
/// transaction — self-inflicted bumps do not restart the scan, and
/// engine-supported writes restore the epoch (marker_restore) so they do
/// not either.
///
/// `decrypt` failures fall back to parsing the stored bytes as JSON
/// directly (replication applies store payload-plaintext metadata even on
/// encrypted followers); if neither decrypts nor parses, the error
/// propagates — fail loud, never certify a row that could not be read.
pub(crate) fn source_turn_repair_batch<F>(
    conn: &rusqlite::Connection,
    decrypt: F,
    batch: i64,
) -> Result<MaintenanceProgress>
where
    F: Fn(&str) -> Result<String>,
{
    let tx = conn.unchecked_transaction()?;
    let epoch_start = meta_get(&tx, SOURCE_TURN_EPOCH_KEY)?.unwrap_or_else(|| "0".to_string());
    let cursor: i64 = match meta_get(&tx, SOURCE_TURN_CURSOR_KEY)? {
        Some(stored) => match stored.split_once(':') {
            Some((epoch, rowid)) if epoch == epoch_start => rowid.parse::<i64>().unwrap_or(0),
            _ => 0, // stale epoch (raw write since) or unparsable: restart
        },
        None => 0,
    };
    let candidates: Vec<(i64, Option<String>, Option<i64>)> = {
        let mut stmt = tx.prepare(
            "SELECT rowid, metadata, source_turn FROM memories \
             WHERE rowid > ?1 ORDER BY rowid LIMIT ?2",
        )?;
        let rows = stmt.query_map(rusqlite::params![cursor, batch], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()?
    };
    let mut processed = 0usize;
    let mut last = cursor;
    for (rowid, stored_meta, stored_turn) in &candidates {
        let expected: Option<i64> = match stored_meta.as_deref() {
            None | Some("") => None,
            Some(stored) => {
                let plain = match decrypt(stored) {
                    Ok(p) => p,
                    Err(decrypt_err) => {
                        if serde_json::from_str::<serde_json::Value>(stored).is_ok() {
                            stored.to_string()
                        } else {
                            return Err(decrypt_err);
                        }
                    }
                };
                serde_json::from_str::<serde_json::Value>(&plain)
                    .ok()
                    .as_ref()
                    .and_then(extract_source_turn)
            }
        };
        if expected != *stored_turn {
            // Fires the invalidation trigger; the cursor below is stamped
            // with the post-update epoch, and only a DRAINED pass sets the
            // marker back to '1'.
            tx.execute(
                "UPDATE memories SET source_turn = ?1 WHERE rowid = ?2",
                rusqlite::params![expected, rowid],
            )?;
        }
        processed += 1;
        last = *rowid;
    }
    let remaining: i64 = tx.query_row(
        "SELECT COUNT(*) FROM memories WHERE rowid > ?1",
        rusqlite::params![last],
        |r| r.get(0),
    )?;
    let complete = remaining == 0;
    if complete {
        meta_put(&tx, SOURCE_TURN_MARKER_KEY, &Some("1".to_string()))?;
        meta_put(&tx, SOURCE_TURN_CURSOR_KEY, &None)?;
    } else if processed > 0 {
        let epoch_now = meta_get(&tx, SOURCE_TURN_EPOCH_KEY)?.unwrap_or_else(|| "0".to_string());
        meta_put(
            &tx,
            SOURCE_TURN_CURSOR_KEY,
            &Some(format!("{epoch_now}:{last}")),
        )?;
    }
    tx.commit()?;
    // Checked i64 -> usize (reviewer fix 5.3): never a saturating shortcut.
    let remaining = usize::try_from(remaining).map_err(|_| {
        crate::base::error::YantrikDbError::InvalidInput(format!(
            "source_turn repair: remaining count {remaining} does not fit in usize"
        ))
    })?;
    Ok(MaintenanceProgress {
        processed,
        remaining,
        complete,
    })
}

impl crate::YantrikDB {
    /// Coverage-first thread retrieval over the `memory_entities` join —
    /// see the module docs for the full contract. Opt-in; existing `recall`
    /// behavior is untouched.
    ///
    /// v1 compatibility contract (final reviewer decision): this method
    /// PRESERVES ITS EXACT PRE-V2 SEMANTICS on its own implementation
    /// path — per-row decrypt-derived `source_turn` ordering over the full
    /// eligible set, correct REGARDLESS of the v50 column/marker state.
    /// Legacy callers never see `MaintenanceRequired`, and none of v2's
    /// input caps apply here. Strict marker gating (and the bounded-page
    /// cost model) is exclusively [`Self::recall_thread_v2`]'s contract;
    /// full materialization + decrypt was always THIS method's documented
    /// cost model. The turn derivation goes through the ONE shared
    /// [`extract_source_turn`] — same semantics the v50 column is stamped
    /// with, so on a healthy store the two paths order identically.
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
        // tokenizer's Unicode lowercase — see normalize_entity_name).
        // Duplicated requests ("Alpha", "alpha") collapse to the first
        // spelling so a row never lists the same entity twice.
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
            // parameter list.
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
        // uses) and read source_turn under the never-invent rule, through
        // the ONE shared extractor the v50 column is also stamped with.
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
                        .as_ref()
                        .and_then(extract_source_turn)
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

    /// Multi-route coverage-first thread retrieval (thread v2): the
    /// eligible set is the SQL UNION of the entity, phrase, and topic
    /// routes, deduped, under v1's visibility predicates — all expressed
    /// in SQL (including the supersede exclusion) — ordered by
    /// `(created_at ASC, source_turn ASC NULLS LAST, rid ASC)` over the
    /// persisted v50 column.
    ///
    /// Consistency and bounds:
    /// - **Single snapshot**: `total`, positions, the returned page, and
    ///   the route provenance are all computed inside ONE read
    ///   transaction, so a concurrent writer can never skew the count
    ///   against the rows.
    /// - **Full-union count, bounded decrypt**: the COUNT and the
    ///   `ROW_NUMBER()` positions range over the FULL eligible union
    ///   (that is the coverage contract — truncation must be loud and
    ///   positions full-thread), but hydration/decryption touches ONLY
    ///   the returned page (`LIMIT limit`, the earliest prefix). The
    ///   input caps (64 entities / 32 phrases / 16 topic rids, each item
    ///   1..=512 bytes) bound the ANCHOR lists — one anchor can still
    ///   match arbitrarily many rows, and SQLite may sort the full union
    ///   (see the idx_memories_source_turn note in base/schema.rs) — while
    ///   the limit bounds the decrypt work; nothing decrypts the union.
    /// - `limit = 0` is a pure count: `items` is empty, `total` is the
    ///   full eligible count, `omitted == total`. `limit` itself is capped
    ///   at `MAX_LIMIT` (10_000): an uncapped limit would defeat the
    ///   bounded-page decryption promise and push the per-page provenance
    ///   `IN (page rids)` probes past SQLite's bind-variable budget.
    /// - All three routes empty → empty `ThreadRecallV2`, not an error.
    ///
    /// Fail-closed edges:
    /// - phrases on an encrypted store → [`crate::error::YantrikDbError::CapabilityUnavailable`]
    ///   (`capability = "phrase_thread_route"`): the FTS index holds
    ///   ciphertext and a MATCH would return silent emptiness.
    /// - a topic rid that does not resolve to a VISIBLE VERIFIED TOPIC
    ///   SYNTHESIS row in this namespace (`synthesis_state = 'verified'`
    ///   with the organizer descriptor columns present — an ordinary
    ///   memory rid is NOT a topic) → typed
    ///   [`crate::error::YantrikDbError::InvalidThreadTopic`],
    ///   byte-identical across nonexistent / cross-namespace / inactive /
    ///   unverified / ordinary-row rids (no existence leakage).
    /// - completeness marker not `'1'` + non-empty eligible set →
    ///   [`crate::error::YantrikDbError::MaintenanceRequired`] naming
    ///   `maintain_source_turn_backfill` — on EVERY store, encrypted or
    ///   not: a raw SQL metadata write on a plaintext store stales the
    ///   scalar column just the same, and the gate refuses rather than
    ///   silently misorders until the (for plaintext stores, parse-only)
    ///   recompute pass drains.
    pub fn recall_thread_v2(
        &self,
        namespace: &str,
        query: &ThreadQuery,
        limit: usize,
    ) -> Result<ThreadRecallV2> {
        use crate::base::error::YantrikDbError;

        const MAX_ENTITIES: usize = 64;
        const MAX_PHRASES: usize = 32;
        const MAX_TOPIC_RIDS: usize = 16;
        const MAX_ITEM_BYTES: usize = 512;
        /// Reviewer fix 5.3: the page bound is part of the bounded-work
        /// contract — big enough for any real thread page, small enough
        /// that page-rid IN-lists stay within SQLite's bind budget.
        const MAX_LIMIT: usize = 10_000;

        fn check_items(kind: &str, items: &[String], max_count: usize) -> Result<()> {
            if items.len() > max_count {
                return Err(YantrikDbError::InvalidInput(format!(
                    "recall_thread_v2: {} {kind} items exceed the cap of {max_count}",
                    items.len()
                )));
            }
            for item in items {
                if item.is_empty() || item.len() > MAX_ITEM_BYTES {
                    return Err(YantrikDbError::InvalidInput(format!(
                        "recall_thread_v2: every {kind} item must be 1..={MAX_ITEM_BYTES} bytes"
                    )));
                }
            }
            Ok(())
        }
        check_items("entity", &query.entities, MAX_ENTITIES)?;
        check_items("phrase", &query.phrases, MAX_PHRASES)?;
        check_items("topic_rid", &query.topic_rids, MAX_TOPIC_RIDS)?;
        if limit > MAX_LIMIT {
            return Err(YantrikDbError::InvalidInput(format!(
                "recall_thread_v2: limit {limit} exceeds the cap of {MAX_LIMIT}"
            )));
        }
        // Checked cast (audit trap 2): the page bound crosses into SQL as
        // i64 — a usize that does not fit must refuse, not wrap. (Always
        // fits after the MAX_LIMIT check; kept checked on principle.)
        let limit_i64 = i64::try_from(limit).map_err(|_| {
            YantrikDbError::InvalidInput(format!(
                "recall_thread_v2: limit {limit} does not fit in i64"
            ))
        })?;

        let empty = || ThreadRecallV2 {
            items: Vec::new(),
            total: 0,
            returned: 0,
            omitted: 0,
        };
        if query.entities.is_empty() && query.phrases.is_empty() && query.topic_rids.is_empty() {
            return Ok(empty());
        }

        // ENCRYPTION BOUNDARY — fail closed, NEVER silent empty: the FTS
        // index on an encrypted store indexes ciphertext, so a phrase
        // MATCH cannot see plaintext. Entities and topics (relational
        // routes) work under encryption.
        if !query.phrases.is_empty() && self.is_encrypted() {
            return Err(YantrikDbError::CapabilityUnavailable {
                capability: "phrase_thread_route".to_string(),
                reason: "the memories_fts index on an encrypted store holds ciphertext, so \
                         a phrase MATCH would silently match nothing; use the entity or \
                         topic routes, or an unencrypted store"
                    .to_string(),
            });
        }

        // Requested-name index by the persisted normalization (the v1
        // rule): duplicate spellings collapse to the first request index.
        let mut req_by_norm: HashMap<String, usize> = HashMap::new();
        for (i, e) in query.entities.iter().enumerate() {
            req_by_norm.entry(normalize_entity_name(e)).or_insert(i);
        }
        let mut norm_names: Vec<String> = req_by_norm.keys().cloned().collect();
        norm_names.sort(); // deterministic parameter order

        // Anchor determinism rule (reviewer item 6): duplicates in the
        // request are deduplicated FIRST-OCCURRENCE-WINS before matching,
        // and every per-anchor provenance list below is emitted in REQUEST
        // ORDER over these deduped lists.
        fn dedup_first<'a>(items: &'a [String]) -> Vec<&'a str> {
            let mut seen = std::collections::HashSet::new();
            items
                .iter()
                .filter(|s| seen.insert(s.as_str()))
                .map(String::as_str)
                .collect()
        }
        let uniq_phrases: Vec<&str> = dedup_first(&query.phrases);
        let uniq_topics: Vec<&str> = dedup_first(&query.topic_rids);

        // FTS5 string-literal escaping (audit trap 7): each phrase is ONE
        // quoted literal (embedded '"' doubled) — user text is never
        // interpreted as FTS query syntax.
        fn fts_literal(phrase: &str) -> String {
            format!("\"{}\"", phrase.replace('"', "\"\""))
        }
        // The union route ORs all phrase literals into one MATCH.
        let fts_expr: Option<String> = if uniq_phrases.is_empty() {
            None
        } else {
            Some(
                uniq_phrases
                    .iter()
                    .map(|p| fts_literal(p))
                    .collect::<Vec<_>>()
                    .join(" OR "),
            )
        };

        // ── SQL assembly ─────────────────────────────────────────────
        // ?1 is always the namespace; route parameters follow in fixed
        // order (entity norms, FTS expression, topic rids). Each union
        // part carries its own DISTINCT (the one-part case has no UNION
        // to dedupe it); UNION dedupes across parts.
        let mut params_v: Vec<Box<dyn rusqlite::types::ToSql>> =
            vec![Box::new(namespace.to_string())];
        let mut union_parts: Vec<String> = Vec::new();
        if !norm_names.is_empty() {
            let ph: Vec<String> = norm_names
                .iter()
                .map(|n| {
                    params_v.push(Box::new(n.clone()));
                    format!("?{}", params_v.len())
                })
                .collect();
            union_parts.push(format!(
                "SELECT DISTINCT me.memory_rid AS rid FROM memory_entities me \
                 WHERE me.entity_name_norm IN ({})",
                ph.join(",")
            ));
        }
        if let Some(expr) = &fts_expr {
            params_v.push(Box::new(expr.clone()));
            union_parts.push(format!(
                "SELECT DISTINCT fm.rid AS rid FROM memories fm \
                 JOIN memories_fts ON memories_fts.rowid = fm.rowid \
                 WHERE memories_fts MATCH ?{}",
                params_v.len()
            ));
        }
        if !uniq_topics.is_empty() {
            let ph: Vec<String> = uniq_topics
                .iter()
                .map(|t| {
                    params_v.push(Box::new((*t).to_string()));
                    format!("?{}", params_v.len())
                })
                .collect();
            union_parts.push(format!(
                "SELECT DISTINCT d.source_rid AS rid FROM synthesis_dependencies d \
                 WHERE d.namespace = ?1 AND d.is_direct = 1 \
                   AND d.synthesis_rid IN ({})",
                ph.join(",")
            ));
        }
        let union_sql = union_parts.join(" UNION ");

        // Visibility predicates — v1's exact set, EXPRESSED IN SQL. v1 ran
        // the supersede exclusion post-SQL through superseded_rids_among;
        // the NOT EXISTS below is the same record_links predicate
        // (link_type/status/selection_state on target_rid), applied under
        // the same status-read-policy gate — equivalence is pinned by the
        // v1 test suite running unchanged through this path plus the
        // explicit equivalence test in thread_tests.
        let mut vis = String::from(
            "m.namespace = ?1 AND m.consolidation_status = 'active' \
             AND (m.synthesis_state IS NULL OR m.synthesis_state = 'verified')",
        );
        if self.status_read_policy() {
            vis.push_str(
                " AND NOT EXISTS (SELECT 1 FROM record_links rl \
                  WHERE rl.link_type = 'supersedes' AND rl.status = 'active' \
                    AND rl.selection_state = 'selected' AND rl.target_rid = m.rid)",
            );
        }

        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            params_v.iter().map(|p| p.as_ref()).collect();

        // ── SINGLE SNAPSHOT (audit trap 1) ───────────────────────────
        // Topic validation, the full-union COUNT, the marker probe, the
        // page, and the route provenance ALL read inside ONE deferred read
        // transaction: no writer can skew total vs rows vs routes.
        let conn = self.read_conn();
        let tx = conn.unchecked_transaction()?;

        // Topic rids must resolve to a VISIBLE, VERIFIED TOPIC SYNTHESIS
        // row in THIS namespace — not merely a visible memory (reviewer
        // fix 5.2: an ordinary active rid must not pass as a "topic" and
        // silently contribute zero dependencies). The strongest
        // SQL-checkable form without decrypting is the persisted organizer
        // descriptor columns record_synthesis stamps:
        // synthesis_state = 'verified' plus a non-NULL axis / valid
        // granularity / non-NULL logical key. ONE error shape across
        // ordinary / nonexistent / cross-namespace / inactive / unverified
        // rids (audit trap 8: no existence leakage).
        let topic_vis = format!(
            "{vis} AND m.synthesis_state = 'verified' \
             AND m.synthesis_axis IS NOT NULL \
             AND m.synthesis_granularity IN ('atomic', 'rollup') \
             AND m.synthesis_logical_key IS NOT NULL"
        );
        for topic_rid in &uniq_topics {
            let visible: bool = tx.query_row(
                &format!(
                    "SELECT EXISTS(SELECT 1 FROM memories m WHERE m.rid = ?2 AND {topic_vis})"
                ),
                rusqlite::params![namespace, topic_rid],
                |r| r.get(0),
            )?;
            if !visible {
                // Dedicated typed variant (reviewer item 9): one message
                // template for EVERY cause — leakage-free by construction.
                return Err(YantrikDbError::InvalidThreadTopic {
                    topic_rid: (*topic_rid).to_string(),
                });
            }
        }

        // Full-union COUNT — deliberately a SEPARATE statement from the
        // page: a COUNT(*) OVER () window inside a LIMIT-ed page reads
        // only the limited rows and silently reports total=0 at limit=0.
        // Same transaction, so it is the same snapshot the page reads.
        let total: i64 = tx.query_row(
            &format!(
                "SELECT COUNT(*) FROM ({union_sql}) u \
                 JOIN memories m ON m.rid = u.rid WHERE {vis}"
            ),
            params_ref.as_slice(),
            |r| r.get(0),
        )?;
        let total_usize = usize::try_from(total).map_err(|_| {
            YantrikDbError::InvalidInput(format!(
                "recall_thread_v2: eligible count {total} does not fit in usize"
            ))
        })?;
        if total_usize == 0 {
            return Ok(empty());
        }

        // STRICT ORDERING GATE — for EVERY store, not only encrypted ones
        // (reviewer fix 5.1): a live raw-SQL metadata change on a PLAINTEXT
        // store also stales the marker, and serving the stale scalar until
        // the next reopen would silently misorder. Gate on the marker
        // whenever the eligible set is non-empty; plaintext stores clear it
        // with the same maintain_source_turn_backfill (for them a fast
        // parse-only recompute pass), and their open()-time recompute also
        // heals on restart.
        {
            let marker = meta_get(&tx, SOURCE_TURN_MARKER_KEY)?;
            if marker.as_deref() != Some("1") {
                return Err(YantrikDbError::MaintenanceRequired {
                    operation: "maintain_source_turn_backfill".to_string(),
                    reason: "this store's source_turn columns are not known to mirror \
                             their metadata (backfill incomplete, or a raw SQL write \
                             staled them), so the strict (created_at, source_turn, rid) \
                             thread order cannot be guaranteed; call \
                             maintain_source_turn_backfill in batches until it reports \
                             complete"
                        .to_string(),
                });
            }
        }

        // The page: ROW_NUMBER() over the FULL eligible union in the total
        // SQL order — created_at ASC, source_turn ASC NULLS LAST (the
        // `IS NULL` sort key), rid ASC — then the earliest prefix.
        // Positions are full-thread; decryption below touches only these
        // rows.
        let page_sql = format!(
            "SELECT rid, text, created_at, source_turn, pos FROM ( \
               SELECT m.rid AS rid, m.text AS text, m.created_at AS created_at, \
                      m.source_turn AS source_turn, \
                      ROW_NUMBER() OVER (ORDER BY m.created_at ASC, \
                          (m.source_turn IS NULL) ASC, m.source_turn ASC, \
                          m.rid ASC) AS pos \
               FROM ({union_sql}) u JOIN memories m ON m.rid = u.rid \
               WHERE {vis} \
             ) ORDER BY pos ASC LIMIT ?{}",
            params_v.len() + 1
        );
        struct PageRow {
            rid: String,
            stored_text: String,
            created_at: f64,
            source_turn: Option<i64>,
            position: usize,
        }
        let mut page_params: Vec<&dyn rusqlite::types::ToSql> = params_ref.clone();
        page_params.push(&limit_i64);
        let page: Vec<PageRow> = {
            let mut stmt = tx.prepare(&page_sql)?;
            let rows = stmt.query_map(page_params.as_slice(), |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, f64>(2)?,
                    r.get::<_, Option<i64>>(3)?,
                    r.get::<_, i64>(4)?,
                ))
            })?;
            rows.map(|row| {
                let (rid, stored_text, created_at, source_turn, pos) = row?;
                // Checked i64 -> usize (reviewer fix 5.3): never `as`.
                let position = usize::try_from(pos).map_err(|_| {
                    YantrikDbError::InvalidInput(format!(
                        "recall_thread_v2: position {pos} does not fit in usize"
                    ))
                })?;
                Ok(PageRow {
                    rid,
                    stored_text,
                    created_at,
                    source_turn,
                    position,
                })
            })
            .collect::<Result<Vec<_>>>()?
        };

        // Route + per-anchor provenance (audit trap 7 + reviewer item 6):
        // ALL routes that matched each returned row AND the specific
        // requested anchors that matched it, from the same snapshot, in
        // REQUEST ORDER over the first-occurrence-deduped anchor lists.
        // Bounded: every probe is IN (page rids), and the page is capped.
        let page_rids: Vec<&str> = page.iter().map(|p| p.rid.as_str()).collect();
        let mut ent_matches: HashMap<String, BTreeSet<usize>> = HashMap::new();
        // rid -> indexes into uniq_phrases / uniq_topics (BTreeSet keeps
        // them in request order because the deduped lists are).
        let mut phrase_matches: HashMap<String, BTreeSet<usize>> = HashMap::new();
        let mut topic_matches: HashMap<String, BTreeSet<usize>> = HashMap::new();
        if !page_rids.is_empty() {
            if !norm_names.is_empty() {
                let mut p: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
                let n_ph: Vec<String> = norm_names
                    .iter()
                    .map(|n| {
                        p.push(Box::new(n.clone()));
                        format!("?{}", p.len())
                    })
                    .collect();
                let r_ph: Vec<String> = page_rids
                    .iter()
                    .map(|r| {
                        p.push(Box::new((*r).to_string()));
                        format!("?{}", p.len())
                    })
                    .collect();
                let pr: Vec<&dyn rusqlite::types::ToSql> = p.iter().map(|b| b.as_ref()).collect();
                let mut stmt = tx.prepare(&format!(
                    "SELECT me.memory_rid, me.entity_name_norm FROM memory_entities me \
                     WHERE me.entity_name_norm IN ({}) AND me.memory_rid IN ({})",
                    n_ph.join(","),
                    r_ph.join(",")
                ))?;
                let rows = stmt.query_map(pr.as_slice(), |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                })?;
                for row in rows {
                    let (rid, norm) = row?;
                    if let Some(idx) = req_by_norm.get(&norm) {
                        ent_matches.entry(rid).or_default().insert(*idx);
                    }
                }
            }
            // Per-PHRASE membership (reviewer item 6): one bounded MATCH
            // per deduped requested phrase, restricted to the page rids,
            // so each row's provenance names the exact phrase(s) that
            // matched it — no aggregate-OR collapse.
            for (phrase_idx, phrase) in uniq_phrases.iter().enumerate() {
                let literal = fts_literal(phrase);
                let mut p: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
                p.push(Box::new(literal));
                let r_ph: Vec<String> = page_rids
                    .iter()
                    .map(|r| {
                        p.push(Box::new((*r).to_string()));
                        format!("?{}", p.len())
                    })
                    .collect();
                let pr: Vec<&dyn rusqlite::types::ToSql> = p.iter().map(|b| b.as_ref()).collect();
                let mut stmt = tx.prepare(&format!(
                    "SELECT fm.rid FROM memories fm \
                     JOIN memories_fts ON memories_fts.rowid = fm.rowid \
                     WHERE memories_fts MATCH ?1 AND fm.rid IN ({})",
                    r_ph.join(",")
                ))?;
                let rows = stmt.query_map(pr.as_slice(), |r| r.get::<_, String>(0))?;
                for row in rows {
                    phrase_matches.entry(row?).or_default().insert(phrase_idx);
                }
            }
            // Per-TOPIC membership: (synthesis_rid, source_rid) pairs so
            // each row lists exactly the requested topics it evidences.
            if !uniq_topics.is_empty() {
                let mut p: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
                p.push(Box::new(namespace.to_string()));
                let t_ph: Vec<String> = uniq_topics
                    .iter()
                    .map(|t| {
                        p.push(Box::new((*t).to_string()));
                        format!("?{}", p.len())
                    })
                    .collect();
                let r_ph: Vec<String> = page_rids
                    .iter()
                    .map(|r| {
                        p.push(Box::new((*r).to_string()));
                        format!("?{}", p.len())
                    })
                    .collect();
                let pr: Vec<&dyn rusqlite::types::ToSql> = p.iter().map(|b| b.as_ref()).collect();
                let mut stmt = tx.prepare(&format!(
                    "SELECT d.synthesis_rid, d.source_rid FROM synthesis_dependencies d \
                     WHERE d.namespace = ?1 AND d.is_direct = 1 \
                       AND d.synthesis_rid IN ({}) AND d.source_rid IN ({})",
                    t_ph.join(","),
                    r_ph.join(",")
                ))?;
                let rows = stmt.query_map(pr.as_slice(), |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                })?;
                for row in rows {
                    let (synthesis_rid, source_rid) = row?;
                    if let Some(topic_idx) = uniq_topics.iter().position(|t| *t == synthesis_rid) {
                        topic_matches
                            .entry(source_rid)
                            .or_default()
                            .insert(topic_idx);
                    }
                }
            }
        }
        drop(tx);
        drop(conn); // snapshot closed; decryption below needs no SQL.

        let mut items = Vec::with_capacity(page.len());
        for row in page {
            let text = self.decrypt_text(&row.stored_text)?;
            let ent_idx = ent_matches.get(&row.rid);
            let phrase_idx = phrase_matches.get(&row.rid);
            let topic_idx = topic_matches.get(&row.rid);
            let mut routes: Vec<&'static str> = Vec::new();
            if ent_idx.is_some_and(|s| !s.is_empty()) {
                routes.push("entity");
            }
            if phrase_idx.is_some_and(|s| !s.is_empty()) {
                routes.push("phrase");
            }
            if topic_idx.is_some_and(|s| !s.is_empty()) {
                routes.push("topic");
            }
            let entities: Vec<String> = ent_idx
                .map(|s| s.iter().map(|i| query.entities[*i].clone()).collect())
                .unwrap_or_default();
            let phrases: Vec<String> = phrase_idx
                .map(|s| s.iter().map(|i| uniq_phrases[*i].to_string()).collect())
                .unwrap_or_default();
            let topic_rids: Vec<String> = topic_idx
                .map(|s| s.iter().map(|i| uniq_topics[*i].to_string()).collect())
                .unwrap_or_default();
            items.push(ThreadItemV2 {
                rid: row.rid,
                text,
                created_at: row.created_at,
                source_turn: row.source_turn,
                position: row.position,
                entities,
                routes,
                phrases,
                topic_rids,
            });
        }
        let returned = items.len();
        let omitted = total_usize - returned;
        Ok(ThreadRecallV2 {
            items,
            total: total_usize,
            returned,
            omitted,
        })
    }

    /// One batch of the v50 source_turn recompute/repair pass with
    /// decrypt-and-stamp — the maintenance operation an encrypted store
    /// runs (in batches, resumably) to satisfy `recall_thread_v2`'s strict
    /// ordering gate; a harmless recompute sweep on unencrypted stores.
    ///
    /// See [`source_turn_repair_batch`] for the full semantics: every row
    /// beyond the persisted cursor is compared against the shared
    /// extractor's output on its CURRENT (decrypted) metadata and
    /// rewritten on mismatch — including back to NULL when the metadata no
    /// longer carries a valid turn — and the completeness marker is set
    /// `'1'` only when a full pass drains. Raw SQL writes mid-scan bump
    /// the invalidation epoch and restart the scan from rowid 0; lazy
    /// write-time stamping continues independently but NEVER sets the
    /// marker.
    pub fn maintain_source_turn_backfill(&self, batch: usize) -> Result<MaintenanceProgress> {
        use crate::base::error::YantrikDbError;
        /// Matches the open-time backfill's batch size: one transaction
        /// materializes (and on encrypted stores decrypts) at most this
        /// many rows, which is what keeps the pass resumable and the
        /// writer lock's hold time bounded — an unbounded batch would
        /// defeat both.
        const MAX_MAINTENANCE_BATCH: usize = 10_000;
        if batch == 0 {
            return Err(YantrikDbError::InvalidInput(
                "maintain_source_turn_backfill: batch must be >= 1 (a zero batch can \
                 never make progress)"
                    .to_string(),
            ));
        }
        if batch > MAX_MAINTENANCE_BATCH {
            return Err(YantrikDbError::InvalidInput(format!(
                "maintain_source_turn_backfill: batch {batch} exceeds the cap of \
                 {MAX_MAINTENANCE_BATCH}"
            )));
        }
        let batch_i64 = i64::try_from(batch).map_err(|_| {
            YantrikDbError::InvalidInput(format!(
                "maintain_source_turn_backfill: batch {batch} does not fit in i64"
            ))
        })?;
        // The serialized writer lock: stamping is a write.
        let conn = self.conn();
        source_turn_repair_batch(&conn, |stored| self.decrypt_text(stored), batch_i64)
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

    // ═════════════════════ thread v2 (multi-route) ═════════════════════

    use crate::engine::thread::ThreadQuery;

    fn q(entities: &[&str], phrases: &[&str], topics: &[&str]) -> ThreadQuery {
        ThreadQuery {
            entities: entities.iter().map(|s| s.to_string()).collect(),
            phrases: phrases.iter().map(|s| s.to_string()).collect(),
            topic_rids: topics.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// (j) UNION dedup + multi-route provenance: a row matched by BOTH an
    /// entity and a phrase appears ONCE and lists both routes in the fixed
    /// ["entity", "phrase"] order, with per-anchor fields populated.
    #[test]
    fn v2_union_dedup_and_multi_route_provenance() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        seed_row(
            &db,
            "j1",
            "Alpha standup notes",
            &meta_empty(),
            BASE_MICROS,
            &["Alpha"],
            1.0,
        );
        seed_row(
            &db,
            "j2",
            "Alpha ran the quarterly sync today",
            &meta_empty(),
            BASE_MICROS + 1_000_000,
            &["Alpha"],
            2.0,
        );
        seed_row(
            &db,
            "j3",
            "Notes from the quarterly sync recap",
            &meta_empty(),
            BASE_MICROS + 2_000_000,
            &["Beta"],
            3.0,
        );
        drain(&db);

        let out = db
            .recall_thread_v2(NS, &q(&["Alpha"], &["quarterly sync"], &[]), 10)
            .unwrap();
        assert_eq!(out.total, 3, "union of both routes, deduped");
        assert_eq!(out.returned, 3);
        assert_eq!(out.omitted, 0);
        let rids: Vec<&str> = out.items.iter().map(|i| i.rid.as_str()).collect();
        assert_eq!(rids, vec!["j1", "j2", "j3"], "chronological order");
        assert_eq!(out.items[0].routes, vec!["entity"]);
        assert_eq!(
            out.items[1].routes,
            vec!["entity", "phrase"],
            "both routes, stable order"
        );
        assert_eq!(out.items[2].routes, vec!["phrase"]);
        assert_eq!(out.items[1].entities, vec!["Alpha".to_string()]);
        assert_eq!(out.items[1].phrases, vec!["quarterly sync".to_string()]);
        assert!(out.items[0].phrases.is_empty());
        assert!(out.items[2].entities.is_empty());
        assert!(out.items.iter().all(|i| i.topic_rids.is_empty()));
    }

    /// (k) Per-anchor provenance determinism: a row matched by TWO phrases
    /// lists both, in REQUEST ORDER; a duplicate anchor in the request is
    /// deduplicated first-occurrence-wins and appears once.
    #[test]
    fn v2_per_anchor_phrase_provenance_in_request_order() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        seed_row(
            &db,
            "k1",
            "budget review and roadmap planning in one session",
            &meta_empty(),
            BASE_MICROS,
            &["Gamma"],
            1.0,
        );
        seed_row(
            &db,
            "k2",
            "roadmap planning only here",
            &meta_empty(),
            BASE_MICROS + 1_000_000,
            &["Gamma"],
            2.0,
        );
        drain(&db);

        let out = db
            .recall_thread_v2(
                NS,
                &q(
                    &[],
                    &["budget review", "roadmap planning", "budget review"],
                    &[],
                ),
                10,
            )
            .unwrap();
        assert_eq!(out.total, 2);
        assert_eq!(
            out.items[0].phrases,
            vec!["budget review".to_string(), "roadmap planning".to_string()],
            "both matching phrases, request order, duplicate collapsed"
        );
        assert_eq!(
            out.items[1].phrases,
            vec!["roadmap planning".to_string()],
            "only the phrase that actually matched"
        );
        assert_eq!(out.items[0].routes, vec!["phrase"]);
    }

    /// (k2) All three routes on one row: entity + phrase + topic — routes
    /// carries all three bits in fixed order and every per-anchor field is
    /// populated. Topic evidence joins synthesis_dependencies
    /// (is_direct=1); record_synthesis is ADDITIVE, so the source stays
    /// visible.
    #[test]
    fn v2_three_route_row_carries_all_provenance() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        seed_row(
            &db,
            "t1",
            "Alpha closed the vendor contract",
            &serde_json::json!({"source_turn": 4}),
            BASE_MICROS,
            &["Alpha"],
            1.0,
        );
        drain(&db);
        let topic = crate::consolidate::record_synthesis(
            &db,
            &["t1".to_string()],
            "Procurement thread organizer",
            Some(&vec_seed(9.0, 8)),
            "topic",
            "rollup",
            &serde_json::json!({}),
            "topic:procurement-v1",
        )
        .unwrap();
        let topic_rid = topic["consolidated_rid"].as_str().unwrap().to_string();

        let out = db
            .recall_thread_v2(
                NS,
                &q(&["Alpha"], &["vendor contract"], &[topic_rid.as_str()]),
                10,
            )
            .unwrap();
        let item = out
            .items
            .iter()
            .find(|i| i.rid == "t1")
            .expect("the evidence row is eligible");
        assert_eq!(item.routes, vec!["entity", "phrase", "topic"]);
        assert_eq!(item.entities, vec!["Alpha".to_string()]);
        assert_eq!(item.phrases, vec!["vendor contract".to_string()]);
        assert_eq!(item.topic_rids, vec![topic_rid.clone()]);
        assert_eq!(item.source_turn, Some(4), "persisted column served");
    }

    /// (l) FTS literal escaping (audit trap 7): a phrase containing a
    /// double-quote round-trips as ONE literal (no FTS syntax error), and
    /// FTS operators inside a phrase are treated as literal tokens — the
    /// phrase "alpha OR beta" matches only the row containing that token
    /// sequence, never the OR-interpretation.
    #[test]
    fn v2_fts_phrases_are_literals_never_syntax() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        seed_row(
            &db,
            "l1",
            "they said \"hello there\" twice",
            &meta_empty(),
            BASE_MICROS,
            &["Delta"],
            1.0,
        );
        seed_row(
            &db,
            "l2",
            "alpha or beta together",
            &meta_empty(),
            BASE_MICROS + 1_000_000,
            &["Delta"],
            2.0,
        );
        seed_row(
            &db,
            "l3",
            "alpha alone here",
            &meta_empty(),
            BASE_MICROS + 2_000_000,
            &["Delta"],
            3.0,
        );
        drain(&db);

        let quoted = db
            .recall_thread_v2(NS, &q(&[], &["said \"hello there\""], &[]), 10)
            .unwrap();
        assert_eq!(
            quoted
                .items
                .iter()
                .map(|i| i.rid.as_str())
                .collect::<Vec<_>>(),
            vec!["l1"],
            "embedded double-quote is escaped, not FTS syntax"
        );

        let or_phrase = db
            .recall_thread_v2(NS, &q(&[], &["alpha OR beta"], &[]), 10)
            .unwrap();
        assert_eq!(
            or_phrase
                .items
                .iter()
                .map(|i| i.rid.as_str())
                .collect::<Vec<_>>(),
            vec!["l2"],
            "OR inside a phrase is a literal token — l3 (alpha alone) must NOT match"
        );
    }

    /// (m) Typed, leakage-free topic errors: an ordinary memory rid, a
    /// nonexistent rid, and a cross-namespace topic rid ALL fail with the
    /// SAME InvalidThreadTopic variant and the SAME message template.
    #[test]
    fn v2_topic_errors_are_typed_and_leak_nothing() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        seed_row(
            &db,
            "m1",
            "Alpha ordinary row",
            &meta_empty(),
            BASE_MICROS,
            &["Alpha"],
            1.0,
        );
        drain(&db);
        // A real topic, but in ANOTHER namespace.
        let other_rid = db
            .record_with_rid(
                "m_other",
                "Other-ns evidence",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &meta_empty(),
                &vec_seed(5.0, 8),
                "other_ns",
                0.8,
                "general",
                "user",
                None,
                BASE_MICROS,
                &[],
                "test-model.v1",
                None,
                crate::provenance::WriteAdmission::Admitted,
            )
            .map(|_| "m_other".to_string())
            .unwrap();
        drain(&db);
        let cross_topic = crate::consolidate::record_synthesis(
            &db,
            &[other_rid],
            "Other-ns organizer",
            Some(&vec_seed(6.0, 8)),
            "topic",
            "rollup",
            &serde_json::json!({}),
            "topic:otherns-v1",
        )
        .unwrap();
        let cross_rid = cross_topic["consolidated_rid"]
            .as_str()
            .unwrap()
            .to_string();

        let probe = |topic: &str| -> String {
            let err = db
                .recall_thread_v2(NS, &q(&["Alpha"], &[], &[topic]), 10)
                .unwrap_err();
            assert!(
                matches!(err, crate::error::YantrikDbError::InvalidThreadTopic { .. }),
                "typed InvalidThreadTopic, got: {err:?}"
            );
            err.to_string().replace(topic, "<RID>")
        };
        let ordinary = probe("m1"); // an active memory, but NOT a topic
        let nonexistent = probe("no-such-rid");
        let cross_ns = probe(&cross_rid);
        assert_eq!(
            ordinary, nonexistent,
            "ordinary-memory and nonexistent rids: identical error shape"
        );
        assert_eq!(
            nonexistent, cross_ns,
            "nonexistent and cross-namespace rids: identical error shape (no leakage)"
        );
    }

    /// (n) Snapshot accounting + the limit=0 pin: total/positions come from
    /// the FULL eligible union regardless of the page bound; `returned` is
    /// an explicit serialized field; limit=0 is a pure count, distinct from
    /// the all-routes-empty query.
    #[test]
    fn v2_totals_positions_and_limit_zero() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        for i in 0..10 {
            seed_row(
                &db,
                &format!("n{i}"),
                &format!("Alpha step {i}"),
                &meta_empty(),
                BASE_MICROS + (i as i64) * 1_000_000,
                &["Alpha"],
                i as f32,
            );
        }
        drain(&db);

        let page = db
            .recall_thread_v2(NS, &q(&["Alpha"], &[], &[]), 4)
            .unwrap();
        assert_eq!(
            (page.total, page.returned, page.omitted),
            (10, 4, 6),
            "full-union total, page-bounded returned"
        );
        assert_eq!(
            page.items.iter().map(|i| i.position).collect::<Vec<_>>(),
            vec![1, 2, 3, 4],
            "positions contiguous from 1 over the full-thread numbering"
        );
        let val = serde_json::to_value(&page).unwrap();
        assert_eq!(
            val["returned"], 4,
            "returned is an explicit serialized field"
        );
        assert_eq!(val["items"].as_array().unwrap().len(), 4);

        // limit = 0: pure count — items empty, total exact, omitted==total.
        let zero = db
            .recall_thread_v2(NS, &q(&["Alpha"], &[], &[]), 0)
            .unwrap();
        assert_eq!(
            (zero.items.len(), zero.total, zero.returned, zero.omitted),
            (0, 10, 0, 10),
            "limit=0 must still report the exact full-union total"
        );

        // All three routes empty: a DIFFERENT case — empty result, total 0.
        let none = db.recall_thread_v2(NS, &q(&[], &[], &[]), 10).unwrap();
        assert_eq!((none.total, none.returned, none.omitted), (0, 0, 0));
    }

    /// (o) SQL-ORDER PIN: within equal created_at the PERSISTED source_turn
    /// column orders rows entirely in SQL — turn ascending, NULLS LAST,
    /// rid tie-break — and a later created_at dominates any turn. This is
    /// the failing-direction target: ordering by rid alone breaks it.
    #[test]
    fn v2_sql_orders_by_persisted_turn_with_nulls_last() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let t = BASE_MICROS;
        seed_row(
            &db,
            "z_turn2",
            "Epsilon event two",
            &serde_json::json!({"source_turn": 2}),
            t,
            &["Epsilon"],
            1.0,
        );
        seed_row(
            &db,
            "a_turn9",
            "Epsilon event nine",
            &serde_json::json!({"turn_id": 9}),
            t,
            &["Epsilon"],
            2.0,
        );
        seed_row(
            &db,
            "b_none",
            "Epsilon event without a turn",
            &meta_empty(),
            t,
            &["Epsilon"],
            3.0,
        );
        seed_row(
            &db,
            "a_none",
            "Epsilon event also without a turn",
            &serde_json::json!({"source_turn": "not-a-turn"}),
            t,
            &["Epsilon"],
            4.0,
        );
        seed_row(
            &db,
            "later_turn0",
            "Epsilon later event",
            &serde_json::json!({"source_turn": 0}),
            t + 1_000_000,
            &["Epsilon"],
            5.0,
        );
        drain(&db);

        let out = db
            .recall_thread_v2(NS, &q(&["Epsilon"], &[], &[]), 10)
            .unwrap();
        assert_eq!(
            out.items.iter().map(|i| i.rid.as_str()).collect::<Vec<_>>(),
            vec!["z_turn2", "a_turn9", "a_none", "b_none", "later_turn0"],
            "turn asc (2 before 9 despite rid order), then NULLS LAST by rid, \
             then created_at dominates"
        );
        assert_eq!(
            out.items.iter().map(|i| i.source_turn).collect::<Vec<_>>(),
            vec![Some(2), Some(9), None, None, Some(0)],
            "the persisted column's values, never invented"
        );
    }

    /// (p) Input caps: MAX_LIMIT (10_000) works, MAX_LIMIT+1 is a typed
    /// InvalidInput; anchor-count and item-byte caps are typed too.
    #[test]
    fn v2_input_caps_are_typed() {
        use crate::error::YantrikDbError;
        let db = YantrikDB::new(":memory:", 8).unwrap();
        seed_row(
            &db,
            "p1",
            "Alpha row",
            &meta_empty(),
            BASE_MICROS,
            &["Alpha"],
            1.0,
        );
        drain(&db);

        assert!(
            db.recall_thread_v2(NS, &q(&["Alpha"], &[], &[]), 10_000)
                .is_ok(),
            "limit at the cap works"
        );
        let over = db
            .recall_thread_v2(NS, &q(&["Alpha"], &[], &[]), 10_001)
            .unwrap_err();
        assert!(matches!(over, YantrikDbError::InvalidInput(_)), "{over:?}");

        let many: Vec<String> = (0..65).map(|i| format!("e{i}")).collect();
        let many_refs: Vec<&str> = many.iter().map(String::as_str).collect();
        let err = db
            .recall_thread_v2(NS, &q(&many_refs, &[], &[]), 10)
            .unwrap_err();
        assert!(matches!(err, YantrikDbError::InvalidInput(_)), "{err:?}");

        let big = "x".repeat(513);
        let err = db
            .recall_thread_v2(NS, &q(&[], &[big.as_str()], &[]), 10)
            .unwrap_err();
        assert!(matches!(err, YantrikDbError::InvalidInput(_)), "{err:?}");
    }

    /// (q) v1/v2 supersede equivalence: v1 excludes superseded rows
    /// post-SQL (superseded_rids_among); v2 expresses the same predicate
    /// in SQL. Same store, same query → same rid set and totals.
    #[test]
    fn v2_supersede_exclusion_matches_v1() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        for (i, rid) in ["s1", "s2", "s3"].iter().enumerate() {
            seed_row(
                &db,
                rid,
                &format!("Alpha supersede row {i}"),
                &meta_empty(),
                BASE_MICROS + (i as i64) * 1_000_000,
                &["Alpha"],
                i as f32,
            );
        }
        drain(&db);
        assert!(db.status_read_policy());
        db.link(
            "s3",
            &crate::types::RecordLink {
                target_rid: "s1".to_string(),
                link_type: crate::types::LinkType::Supersedes,
            },
        )
        .unwrap();

        let v1 = db.recall_thread(NS, &["Alpha"], 10).unwrap();
        let v2 = db
            .recall_thread_v2(NS, &q(&["Alpha"], &[], &[]), 10)
            .unwrap();
        assert_eq!(v1.total, v2.total, "same eligible count");
        assert_eq!(
            v1.items.iter().map(|i| i.rid.as_str()).collect::<Vec<_>>(),
            v2.items.iter().map(|i| i.rid.as_str()).collect::<Vec<_>>(),
            "same rows in the same order — the SQL NOT EXISTS is equivalent \
             to v1's post-SQL superseded_rids_among"
        );
        assert!(
            !v2.items.iter().any(|i| i.rid == "s1"),
            "superseded row excluded in SQL"
        );
    }

    /// (r) Replication parity for v2: after apply_ops the follower answers
    /// the FULL multi-route v2 query identically — order, source_turn
    /// scalars, routes, per-anchor fields, totals, returned, omitted.
    #[test]
    fn v2_replication_parity_on_follower() {
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
                &format!("r{i}"),
                &format!("{name} {tag} number {i}"),
                &serde_json::json!({"source_turn": i}),
                BASE_MICROS + (i as i64) * 1_000_000,
                &[name],
                i as f32,
            );
        }
        leader.relate("Alpha", "Beta", "related_to", 1.0).unwrap();
        drain(&leader);

        let query = q(&["Alpha"], &["sync"], &[]);
        let leader_out = leader.recall_thread_v2(NS, &query, 100).unwrap();
        assert_eq!(leader_out.total, 6, "3 entity rows + 3 phrase rows");
        assert!(
            leader_out.items.iter().any(|i| i.routes == vec!["phrase"]),
            "phrase-only rows present"
        );

        let follower = YantrikDB::new(":memory:", 8).unwrap();
        let ops = extract_ops_since(&leader.conn(), None, None, None, 1000).unwrap();
        apply_ops(&follower, &ops).unwrap();

        let follower_out = follower.recall_thread_v2(NS, &query, 100).unwrap();
        assert_eq!(
            follower_out, leader_out,
            "identical v2 result on both sides — order, turns, routes, \
             per-anchor fields, totals, returned, omitted"
        );
    }

    /// Encrypted store: the phrase route fails CLOSED with the typed
    /// CapabilityUnavailable — never a silent empty result — while the
    /// entity route keeps working.
    #[test]
    fn v2_encrypted_phrase_route_is_typed_error() {
        use crate::error::YantrikDbError;
        let db = YantrikDB::new_encrypted(":memory:", 8, &[7u8; 32]).unwrap();
        seed_row(
            &db,
            "e1",
            "Alpha encrypted row",
            &serde_json::json!({"source_turn": 1}),
            BASE_MICROS,
            &["Alpha"],
            1.0,
        );
        drain(&db);

        let err = db
            .recall_thread_v2(NS, &q(&["Alpha"], &["anything"], &[]), 10)
            .unwrap_err();
        match err {
            YantrikDbError::CapabilityUnavailable { capability, .. } => {
                assert_eq!(capability, "phrase_thread_route");
            }
            other => panic!("expected CapabilityUnavailable, got {other:?}"),
        }
        // Entities still work under encryption (fresh store: marker complete).
        let ok = db
            .recall_thread_v2(NS, &q(&["Alpha"], &[], &[]), 10)
            .unwrap();
        assert_eq!(ok.total, 1);
        assert_eq!(ok.items[0].source_turn, Some(1));
    }
}
