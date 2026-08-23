use rusqlite::{params, OptionalExtension};

use crate::error::{Result, YantrikDbError};
use crate::scoring;
use crate::types::*;

use super::reservation::ReservationGuard;
use super::{embedding_hash, now, YantrikDB};

/// **v0.10 Item 3 — correction seqlock guard (sol r4).** Bumps the DB-wide
/// `correction_epoch` ODD on construction and back EVEN on Drop (every
/// error/panic path), so a reader can detect a correction that interleaved
/// with its candidate-generation → hydration span. Constructed only under
/// the serialized connection lock, so the even↔odd toggle has a single
/// holder at a time.
pub(crate) struct CorrectionEpochGuard<'a> {
    epoch: &'a std::sync::atomic::AtomicU64,
}

impl<'a> CorrectionEpochGuard<'a> {
    fn new(epoch: &'a std::sync::atomic::AtomicU64) -> Self {
        epoch.fetch_add(1, std::sync::atomic::Ordering::AcqRel); // even -> odd
        Self { epoch }
    }
}

impl Drop for CorrectionEpochGuard<'_> {
    fn drop(&mut self) {
        self.epoch.fetch_add(1, std::sync::atomic::Ordering::AcqRel); // odd -> even
    }
}

impl YantrikDB {
    pub(crate) fn invalidate_synthesis_dependents_in_tx(
        tx: &rusqlite::Transaction<'_>,
        source_rid: &str,
    ) -> Result<Vec<String>> {
        let mut stmt = tx.prepare(
            "UPDATE memories SET synthesis_state = 'invalidated' \
             WHERE synthesis_state = 'verified' \
               AND rid IN ( \
                   SELECT synthesis_rid FROM synthesis_dependencies \
                   WHERE namespace = (SELECT namespace FROM memories WHERE rid = ?1) \
                     AND source_rid = ?1 \
               ) \
             RETURNING rid",
        )?;
        let rows = stmt.query_map(params![source_rid], |row| row.get(0))?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    }

    /// Enter the correction seqlock (v0.10 Item 3). Call AFTER acquiring
    /// the serialized conn lock and BEFORE the reservation/SQL mutation;
    /// hold the returned guard across SQL commit + vector publish + cache
    /// update. The guard restores an even epoch on drop (all paths).
    ///
    /// **The returned guard BORROWS `conn_guard`** (sol r5 finding 1), so
    /// the connection cannot be released while the epoch guard is live —
    /// the compiler forces `drop(_epoch)` before `drop(conn)`. That keeps
    /// the single-holder invariant real: the epoch returns to even only
    /// after the current correction has both finished mutating AND is still
    /// holding conn, so no second correction can overlap into a "falsely
    /// even" window.
    pub(crate) fn enter_correction_epoch<'a, G>(
        &'a self,
        _conn_guard: &'a G,
    ) -> CorrectionEpochGuard<'a> {
        CorrectionEpochGuard::new(&self.correction_epoch)
    }

    /// Read a stable EVEN correction epoch for a recall's before-search
    /// snapshot. Spins while a correction is mid-flight (odd) — that window
    /// is only the commit + publish + cache critical section (microseconds),
    /// never the slow embed. Bounded (sol r5 finding 2): after the spin
    /// budget, returns `None` so the caller surfaces a retryable busy error
    /// rather than busy-waiting forever under a correction storm.
    pub(crate) fn correction_epoch_even(&self) -> Option<u64> {
        const MAX_SPINS: u32 = 1 << 20; // ~1M spins with periodic yields
        let mut spins = 0u32;
        loop {
            let e = self
                .correction_epoch
                .load(std::sync::atomic::Ordering::Acquire);
            if e & 1 == 0 {
                return Some(e);
            }
            spins += 1;
            if spins >= MAX_SPINS {
                return None;
            }
            if spins % 1024 == 0 {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
    }

    /// Validate a recall's before-search epoch snapshot AFTER its reads
    /// (candidate generation + hydration). Uses an **Acquire fence then a
    /// Relaxed load** (sol r5 finding 3 / the crossbeam seqlock pattern):
    /// the fence orders the PRECEDING reads before the version check, which
    /// a plain `load(Acquire)` does not (Acquire orders operations AFTER the
    /// load). Returns true iff no correction started or completed since `e0`
    /// (still even and unchanged).
    pub(crate) fn correction_epoch_validate(&self, e0: u64) -> bool {
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        self.correction_epoch
            .load(std::sync::atomic::Ordering::Relaxed)
            == e0
    }
    /// Get a single memory by RID.
    ///
    /// v0.10 Item 2: a successful consumer `get` is an outcome anchor —
    /// an independent, caller-initiated action targeting the rid — and
    /// records a weak-positive ranking label bound to the rid's most
    /// recent impression (no-op if the ranker never served it). Engine-
    /// internal reads must use [`Self::get_untracked`] instead so the
    /// learner never labels its own traversals.
    #[tracing::instrument(skip(self))]
    pub fn get(&self, rid: &str) -> Result<Option<Memory>> {
        let found = self.get_untracked(rid)?;
        if found.is_some() {
            self.note_caller_used(rid);
        }
        Ok(found)
    }

    /// `get` without the caller_used label — for engine-internal reads
    /// (conflict resolution, link hydration, maintenance).
    pub(crate) fn get_untracked(&self, rid: &str) -> Result<Option<Memory>> {
        let conn = self.conn();
        let mut stmt = conn.prepare("SELECT * FROM memories WHERE rid = ?1")?;

        let result = stmt.query_row(params![rid], |row| {
            Ok((
                row.get::<_, String>("rid")?,
                row.get::<_, String>("type")?,
                row.get::<_, String>("text")?,
                row.get::<_, f64>("created_at")?,
                row.get::<_, f64>("importance")?,
                row.get::<_, f64>("valence")?,
                row.get::<_, f64>("half_life")?,
                row.get::<_, f64>("last_access")?,
                row.get::<_, i64>("access_count")?,
                row.get::<_, String>("consolidation_status")?,
                row.get::<_, String>("storage_tier")?,
                row.get::<_, Option<String>>("consolidated_into")?,
                row.get::<_, String>("metadata")?,
                row.get::<_, String>("namespace")?,
                row.get::<_, f64>("certainty")?,
                row.get::<_, String>("domain")?,
                row.get::<_, String>("source")?,
                row.get::<_, Option<String>>("emotional_state")?,
                row.get::<_, Option<String>>("session_id")?,
                row.get::<_, Option<f64>>("due_at")?,
                row.get::<_, Option<String>>("temporal_kind")?,
            ))
        });

        match result {
            Ok(row) => {
                let text = self.decrypt_text(&row.2)?;
                let meta_str = self.decrypt_text(&row.12)?;
                let metadata: serde_json::Value = serde_json::from_str(&meta_str)
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                Ok(Some(Memory {
                    rid: row.0,
                    memory_type: row.1,
                    text,
                    created_at: row.3,
                    importance: row.4,
                    valence: row.5,
                    half_life: row.6,
                    last_access: row.7,
                    access_count: row.8 as u32,
                    consolidation_status: row.9,
                    storage_tier: row.10,
                    consolidated_into: row.11,
                    metadata,
                    namespace: row.13,
                    certainty: row.14,
                    domain: row.15,
                    source: row.16,
                    emotional_state: row.17,
                    session_id: row.18,
                    due_at: row.19,
                    temporal_kind: row.20,
                }))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Browse memories with optional filters. Returns active memories sorted by the
    /// given field. Useful for auditing stored data without a search query.
    pub fn list_memories(
        &self,
        limit: usize,
        offset: usize,
        domain: Option<&str>,
        memory_type: Option<&str>,
        namespace: Option<&str>,
        sort_by: &str,
    ) -> Result<(Vec<Memory>, usize)> {
        let order = match sort_by {
            "importance" => "importance DESC",
            "last_access" => "last_access DESC",
            _ => "created_at DESC",
        };

        let mut conditions = vec!["consolidation_status = 'active'".to_string()];
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut idx = 1;

        if let Some(d) = domain {
            conditions.push(format!("domain = ?{idx}"));
            param_values.push(Box::new(d.to_string()));
            idx += 1;
        }
        if let Some(mt) = memory_type {
            conditions.push(format!("type = ?{idx}"));
            param_values.push(Box::new(mt.to_string()));
            idx += 1;
        }
        if let Some(ns) = namespace {
            conditions.push(format!("namespace = ?{idx}"));
            param_values.push(Box::new(ns.to_string()));
            idx += 1;
        }

        let where_clause = conditions.join(" AND ");

        // Get total count
        let count_sql = format!("SELECT COUNT(*) FROM memories WHERE {where_clause}");
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();
        let conn = self.conn();
        let total: usize = conn.query_row(&count_sql, params_ref.as_slice(), |row| row.get(0))?;

        // Fetch page
        let sql = format!(
            "SELECT rid, type, text, created_at, importance, valence, half_life, \
             last_access, access_count, consolidation_status, storage_tier, \
             consolidated_into, metadata, namespace, certainty, domain, source, \
             emotional_state, session_id, due_at, temporal_kind \
             FROM memories WHERE {where_clause} ORDER BY {order} LIMIT ?{idx} OFFSET ?{}",
            idx + 1
        );
        param_values.push(Box::new(limit as i64));
        param_values.push(Box::new(offset as i64));
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_ref.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, f64>(3)?,
                row.get::<_, f64>(4)?,
                row.get::<_, f64>(5)?,
                row.get::<_, f64>(6)?,
                row.get::<_, f64>(7)?,
                row.get::<_, i64>(8)?,
                row.get::<_, String>(9)?,
                row.get::<_, String>(10)?,
                row.get::<_, Option<String>>(11)?,
                row.get::<_, String>(12)?,
                row.get::<_, String>(13)?,
                row.get::<_, f64>(14)?,
                row.get::<_, String>(15)?,
                row.get::<_, String>(16)?,
                row.get::<_, Option<String>>(17)?,
                row.get::<_, Option<String>>(18)?,
                row.get::<_, Option<f64>>(19)?,
                row.get::<_, Option<String>>(20)?,
            ))
        })?;

        let mut memories = Vec::new();
        for row in rows {
            let row = row?;
            let text = self.decrypt_text(&row.2)?;
            let meta_str = self.decrypt_text(&row.12)?;
            let metadata: serde_json::Value = serde_json::from_str(&meta_str)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            memories.push(Memory {
                rid: row.0,
                memory_type: row.1,
                text,
                created_at: row.3,
                importance: row.4,
                valence: row.5,
                half_life: row.6,
                last_access: row.7,
                access_count: row.8 as u32,
                consolidation_status: row.9,
                storage_tier: row.10,
                consolidated_into: row.11,
                metadata,
                namespace: row.13,
                certainty: row.14,
                domain: row.15,
                source: row.16,
                emotional_state: row.17,
                session_id: row.18,
                due_at: row.19,
                temporal_kind: row.20,
            });
        }

        Ok((memories, total))
    }

    /// Fetch one memory by rid, decrypted, or `None` if it does not exist.
    ///
    /// The store had no rid point-read: `list_memories` pages and `recall`
    /// is semantic, so a caller holding a rid — which is what `get_conflicts`,
    /// `get_edges`, `record_links` and the consolidation APIs all hand back —
    /// could only resolve it to text by paging the whole namespace. Found
    /// while wiring LLM conflict resolution, where the conflict record gives
    /// `memory_a`/`memory_b` as bare rids and the reconciler needs both texts.
    ///
    /// Returns the memory regardless of `consolidation_status`: a caller
    /// naming a specific rid wants that record, not a currency judgement.
    pub fn get_memory(&self, rid: &str) -> Result<Option<Memory>> {
        let conn = self.conn();
        let row = conn
            .query_row(
                "SELECT rid, type, text, created_at, importance, valence, half_life, \
                 last_access, access_count, consolidation_status, storage_tier, \
                 consolidated_into, metadata, namespace, certainty, domain, source, \
                 emotional_state, session_id, due_at, temporal_kind \
                 FROM memories WHERE rid = ?1",
                rusqlite::params![rid],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, f64>(3)?,
                        row.get::<_, f64>(4)?,
                        row.get::<_, f64>(5)?,
                        row.get::<_, f64>(6)?,
                        row.get::<_, f64>(7)?,
                        row.get::<_, i64>(8)?,
                        row.get::<_, String>(9)?,
                        row.get::<_, String>(10)?,
                        row.get::<_, Option<String>>(11)?,
                        row.get::<_, String>(12)?,
                        row.get::<_, String>(13)?,
                        row.get::<_, f64>(14)?,
                        row.get::<_, String>(15)?,
                        row.get::<_, String>(16)?,
                        row.get::<_, Option<String>>(17)?,
                        row.get::<_, Option<String>>(18)?,
                        row.get::<_, Option<f64>>(19)?,
                        row.get::<_, Option<String>>(20)?,
                    ))
                },
            )
            .optional()?;
        drop(conn); // decrypt_* must not run under the conn lock (CONCURRENCY.md Rule 4)
        let Some(row) = row else { return Ok(None) };
        let text = self.decrypt_text(&row.2)?;
        let meta_str = self.decrypt_text(&row.12)?;
        Ok(Some(Memory {
            rid: row.0,
            memory_type: row.1,
            text,
            created_at: row.3,
            importance: row.4,
            valence: row.5,
            half_life: row.6,
            last_access: row.7,
            access_count: row.8 as u32,
            consolidation_status: row.9,
            storage_tier: row.10,
            consolidated_into: row.11,
            metadata: serde_json::from_str(&meta_str)
                .unwrap_or(serde_json::Value::Object(Default::default())),
            namespace: row.13,
            certainty: row.14,
            domain: row.15,
            source: row.16,
            emotional_state: row.17,
            session_id: row.18,
            due_at: row.19,
            temporal_kind: row.20,
        }))
    }

    /// Return the head of a chain-shaped namespace — its most recent entry.
    ///
    /// Identity / narrative chains (e.g. a `claude_self_narrative` namespace
    /// whose entries are appended over time) need *exact* "latest entry"
    /// semantics. `recall` ranks by similarity × importance × decay, so it can
    /// never guarantee it returns the chain head; this does. Because rids are
    /// UUIDv7 (lexically chronological), the greatest active rid in the
    /// namespace is exactly the latest write — an O(log n) index seek on the
    /// primary key, not a retrieval lottery.
    ///
    /// Walk backwards from the head with
    /// `list_records(namespace, order="desc", since_rid=head.rid, ...)`.
    pub fn chain_head(&self, namespace: &str) -> Result<Option<Memory>> {
        let (mut records, _) =
            self.list_records(Some(namespace), None, None, None, None, None, 1, "desc")?;
        Ok(records.pop())
    }

    /// **v0.7.24 — structural query path.** Typed enumeration by indexed
    /// metadata fields with a stable keyset cursor — the relational counterpart
    /// to similarity `recall`. All filters are optional and AND-compose. The
    /// `kind` / `drive_id` predicates ride the v32 generated-column indexes
    /// (`idx_memories_kind` / `idx_memories_drive_id`), and pagination is a
    /// keyset walk over the UUIDv7 `rid` primary key (lexically = chronological),
    /// so it stays O(log n) and is stable under concurrent inserts — unlike the
    /// `LIMIT/OFFSET` path in `list_memories`.
    ///
    /// `order`: `"desc"` (newest-first) walks `rid < since_rid`; anything else
    /// (default `"asc"`, oldest-first) walks `rid > since_rid`. Returns the page
    /// plus `next_cursor` = the last rid of a *full* page (pass it back as the
    /// next `since_rid`), or `None` when the page was not filled (end reached).
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip(self))]
    pub fn list_records(
        &self,
        namespace: Option<&str>,
        kind: Option<&str>,
        drive_id: Option<&str>,
        memory_type: Option<&str>,
        domain: Option<&str>,
        since_rid: Option<&str>,
        limit: usize,
        order: &str,
    ) -> Result<(Vec<Memory>, Option<String>)> {
        let desc = order.eq_ignore_ascii_case("desc");
        let cursor_op = if desc { "<" } else { ">" };
        let order_sql = if desc { "DESC" } else { "ASC" };

        let mut conditions = vec!["consolidation_status = 'active'".to_string()];
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut idx = 1;

        // All optional, AND-composed. Order chosen so the most selective
        // indexed predicates (kind/drive_id) appear first.
        for (col, val) in [
            ("kind", kind),
            ("drive_id", drive_id),
            ("namespace", namespace),
            ("type", memory_type),
            ("domain", domain),
        ] {
            if let Some(v) = val {
                conditions.push(format!("{col} = ?{idx}"));
                param_values.push(Box::new(v.to_string()));
                idx += 1;
            }
        }
        if let Some(cursor) = since_rid {
            conditions.push(format!("rid {cursor_op} ?{idx}"));
            param_values.push(Box::new(cursor.to_string()));
            idx += 1;
        }

        let where_clause = conditions.join(" AND ");
        let sql = format!(
            "SELECT rid, type, text, created_at, importance, valence, half_life, \
             last_access, access_count, consolidation_status, storage_tier, \
             consolidated_into, metadata, namespace, certainty, domain, source, \
             emotional_state, session_id, due_at, temporal_kind \
             FROM memories WHERE {where_clause} ORDER BY rid {order_sql} LIMIT ?{idx}"
        );
        param_values.push(Box::new(limit as i64));

        let conn = self.conn();
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_ref.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, f64>(3)?,
                row.get::<_, f64>(4)?,
                row.get::<_, f64>(5)?,
                row.get::<_, f64>(6)?,
                row.get::<_, f64>(7)?,
                row.get::<_, i64>(8)?,
                row.get::<_, String>(9)?,
                row.get::<_, String>(10)?,
                row.get::<_, Option<String>>(11)?,
                row.get::<_, String>(12)?,
                row.get::<_, String>(13)?,
                row.get::<_, f64>(14)?,
                row.get::<_, String>(15)?,
                row.get::<_, String>(16)?,
                row.get::<_, Option<String>>(17)?,
                row.get::<_, Option<String>>(18)?,
                row.get::<_, Option<f64>>(19)?,
                row.get::<_, Option<String>>(20)?,
            ))
        })?;

        let mut memories = Vec::new();
        for row in rows {
            let row = row?;
            let text = self.decrypt_text(&row.2)?;
            let meta_str = self.decrypt_text(&row.12)?;
            let metadata: serde_json::Value = serde_json::from_str(&meta_str)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            memories.push(Memory {
                rid: row.0,
                memory_type: row.1,
                text,
                created_at: row.3,
                importance: row.4,
                valence: row.5,
                half_life: row.6,
                last_access: row.7,
                access_count: row.8 as u32,
                consolidation_status: row.9,
                storage_tier: row.10,
                consolidated_into: row.11,
                metadata,
                namespace: row.13,
                certainty: row.14,
                domain: row.15,
                source: row.16,
                emotional_state: row.17,
                session_id: row.18,
                due_at: row.19,
                temporal_kind: row.20,
            });
        }

        // next_cursor = last rid only when the page was filled (more may exist).
        let next_cursor = if memories.len() == limit {
            memories.last().map(|m| m.rid.clone())
        } else {
            None
        };
        Ok((memories, next_cursor))
    }

    /// Find memories that have decayed below a threshold.
    #[tracing::instrument(skip(self))]
    pub fn decay(&self, threshold: f64) -> Result<Vec<DecayedMemory>> {
        let ts = now();
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT rid, text, importance, half_life, last_access, type FROM memories \
             WHERE consolidation_status = 'active'",
        )?;

        let mut decayed = Vec::new();
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>("rid")?,
                row.get::<_, String>("text")?,
                row.get::<_, f64>("importance")?,
                row.get::<_, f64>("half_life")?,
                row.get::<_, f64>("last_access")?,
                row.get::<_, String>("type")?,
            ))
        })?;

        for row in rows {
            let (rid, stored_text, importance, half_life, last_access, mem_type) = row?;
            let elapsed = ts - last_access;
            let score = scoring::decay_score(importance, half_life, elapsed);
            if score < threshold {
                let text = self.decrypt_text(&stored_text)?;
                decayed.push(DecayedMemory {
                    rid,
                    text,
                    memory_type: mem_type,
                    original_importance: importance,
                    current_score: score,
                    days_since_access: elapsed / 86400.0,
                });
            }
        }

        Ok(decayed)
    }

    /// **Issue #9 — deterministic tombstone primitive for cluster replication.**
    ///
    /// Sibling of `forget()` that takes caller-supplied namespace +
    /// timestamp + optional reason + optional seq for byte-deterministic
    /// follower replay. Used by yantrikdb-server's cluster-mode applier so
    /// replicated tombstones converge to identical engine state across
    /// leader + followers.
    ///
    /// # Contract
    ///
    /// - **Idempotent on missing**: tombstoning a rid that does not exist
    ///   returns `Ok(())` (NOT an error and NOT a `false` flag — different
    ///   from `forget()`). Snapshot-install + log replay overlap means
    ///   double-delete is normal cluster behavior.
    /// - **Idempotent on already-tombstoned**: re-tombstoning a row that
    ///   is already tombstoned returns `Ok(())` without emitting a new
    ///   oplog entry or re-bumping cache state. Replay-safe.
    /// - **Caller-supplied namespace**: required for the visible_seq bump
    ///   regardless of whether the SQL row exists locally — followers
    ///   apply log entries before the corresponding `record_with_rid` may
    ///   have arrived (snapshot lag), but the bump must still happen so
    ///   the cluster-wide visible_seq[ns] is monotonic with the openraft
    ///   commit-log index.
    /// - **Caller-supplied timestamp**: `requested_at_unix_micros` materialized
    ///   into `updated_at` (REAL seconds). No engine `now()` call on this path.
    /// - **Optional reason**: stored in `tombstone_reason TEXT` column (v25).
    ///   NULL when caller passes None.
    /// - **Caller-supplied `seq`** (cluster mode): when `Some(n)`, used
    ///   as the delta-tombstone seq + visible_seq bump value; engine
    ///   ratchets `vec_seq` to at least `n`. `None` lets the engine
    ///   allocate (single-node).
    ///
    /// Always emits a tombstone marker into the DeltaIndex regardless of
    /// whether the SQL row was newly tombstoned — followers may have the
    /// rid in their delta even if SQL is absent.
    pub fn tombstone_with_rid(
        &self,
        rid: &str,
        namespace: &str,
        reason: Option<&str>,
        requested_at_unix_micros: i64,
        seq: Option<u64>,
    ) -> Result<()> {
        self.tombstone_inner(rid, Some(namespace), reason, requested_at_unix_micros, seq)?;
        Ok(())
    }

    /// Internal helper shared by `tombstone_with_rid` and `forget`. Returns
    /// `true` iff the row was newly tombstoned (was active or consolidated
    /// before this call). Returns `false` if rid is missing or already
    /// tombstoned — both treated as idempotent successful no-ops.
    ///
    /// `namespace`:
    ///   - `Some(ns)`: cluster path — caller has the namespace from the
    ///     replication payload; we bump `visible_seq[ns]` even if the rid
    ///     is missing locally (snapshot-lag determinism).
    ///   - `None`: `forget()` path — we SELECT the namespace from the row.
    ///     If the row is missing, `visible_seq` is not bumped (no reader
    ///     would be waiting on a non-existent rid in single-node mode).
    fn tombstone_inner(
        &self,
        rid: &str,
        namespace: Option<&str>,
        reason: Option<&str>,
        ts_micros: i64,
        seq: Option<u64>,
    ) -> Result<bool> {
        // THE F1/F2 FORGET-RESURRECTION RACE, closed 2026-08-17.
        //
        // Both entry points — `forget()` and the cluster-deterministic
        // `tombstone_with_rid()` — funnel here, and this body tombstones the
        // rid AND its chunks in `self.search_state.load().vec_index` (see
        // `purge_chunks`). Neither took the sync-writer guard, so a reembed
        // cutover could interleave:
        //
        //   1. forget() tombstones the row in SQL and in the CURRENT delta;
        //   2. reembed publishes a new SearchState whose index was built
        //      from a SQL snapshot taken BEFORE that tombstone committed;
        //   3. the delta tombstone died with the discarded state.
        //
        // The record is then tombstoned in SQL and ALIVE in the live index —
        // a delete that silently un-deletes, visible only to whoever
        // retrieves the thing the user asked to forget. Guarding here means
        // the cutover cannot complete mid-delete: reembed switches to
        // Queueing and then waits for in-flight sync writers to drain, so a
        // forget either finishes before the swap or is refused outright.
        //
        // Guarded at `tombstone_inner` rather than at each caller
        // deliberately — one predicate covering forget(), tombstone_with_rid()
        // and the chunk purge they share, instead of three copies to keep in
        // sync. (`purge_chunks` has two further direct callers on the
        // replication conflict-loser path; those are not covered by this
        // guard and are tracked separately.)
        let Some(_sync_guard) = self.write_router.try_enter_sync_writer() else {
            return Err(crate::error::YantrikDbError::ForgetDeferredDuringReembed {
                rid: rid.to_string(),
            });
        };

        let ts_secs = (ts_micros as f64) / 1_000_000.0;

        // Every durable projection of a forget belongs to one transaction.
        // A crash may leave the in-memory indexes stale until reopen, but SQL
        // can no longer expose a tombstoned memory with live entity/chunk/link
        // projections or without its replication intent.
        let (was_newly_tombstoned, ns_to_bump, chunk_idxs, invalidated_syntheses): (
            bool,
            Option<String>,
            Vec<i64>,
            Vec<String>,
        ) = {
            let conn = self.conn();
            let tx = conn.unchecked_transaction()?;
            let resolved_ns: Option<String> = match namespace {
                Some(ns) => Some(ns.to_string()),
                None => tx
                    .query_row(
                        "SELECT namespace FROM memories WHERE rid = ?1",
                        params![rid],
                        |r| r.get::<_, String>(0),
                    )
                    .ok(),
            };
            let changes = tx.execute(
                "UPDATE memories SET consolidation_status = 'tombstoned', \
                 updated_at = ?1, tombstone_reason = ?2 \
                 WHERE rid = ?3 AND consolidation_status != 'tombstoned'",
                params![ts_secs, reason, rid],
            )?;
            let was_newly_tombstoned = changes > 0;

            let chunk_idxs: Vec<i64> = {
                let mut stmt = tx.prepare("SELECT chunk_idx FROM memory_chunks WHERE rid = ?1")?;
                let rows = stmt.query_map(params![rid], |r| r.get(0))?;
                rows.collect::<std::result::Result<_, _>>()?
            };
            tx.execute("DELETE FROM memory_chunks WHERE rid = ?1", params![rid])?;

            let invalidated_syntheses = if was_newly_tombstoned {
                Self::invalidate_synthesis_dependents_in_tx(&tx, rid)?
            } else {
                Vec::new()
            };

            if was_newly_tombstoned {
                tx.execute(
                    "DELETE FROM memory_entities WHERE memory_rid = ?1",
                    params![rid],
                )?;
                tx.execute(
                    "UPDATE record_links SET status = 'broken_source_forgotten' \
                     WHERE source_rid = ?1 AND status = 'active'",
                    params![rid],
                )?;
                tx.execute(
                    "UPDATE record_links SET status = 'broken_target_forgotten' \
                     WHERE target_rid = ?1 AND status = 'active'",
                    params![rid],
                )?;
                self.log_op_in_tx(
                    &tx,
                    "forget",
                    Some(rid),
                    &serde_json::json!({
                        "rid": rid,
                        "updated_at_unix_micros": ts_micros,
                        "reason": reason,
                    }),
                    None,
                    None,
                    self.search_state.load().generation as i64,
                    None,
                )?;
            }
            tx.commit()?;
            (
                was_newly_tombstoned,
                resolved_ns,
                chunk_idxs,
                invalidated_syntheses,
            )
        };

        // Always emit a delta tombstone so search() filters it out even
        // before SQL has applied. Cluster followers may have the rid in
        // their delta from a recent record_with_rid that has not yet
        // compacted into cold; the tombstone marker covers that window.
        //
        // **Issue #41 brainstorm-4 §1.** Snapshot SearchState — the
        // tombstone lands on the active generation's DeltaIndex.
        let seq = self.assign_seq(seq);
        self.search_state.load().vec_index.tombstone(rid, seq);
        // Chunked embeddings: the index matches keys by exact string, so
        // tombstoning the parent does NOT cover its `{rid}#c{idx}` window
        // keys — each needs its own marker or the windows keep serving a
        // dead record. Their durable rows were deleted in the transaction
        // above; only the infallible index markers remain here.
        for idx in chunk_idxs {
            let key = crate::vector::chunk::chunk_key(rid, idx as usize);
            self.search_state.load().vec_index.tombstone(&key, seq);
        }
        if let Some(ns) = &ns_to_bump {
            self.bump_visible_seq(ns, seq);
        }

        // Engine-internal index updates only when the row was newly tombstoned
        // (replay-safe: no double-emit on idempotent re-apply).
        if was_newly_tombstoned {
            self.graph_index.write().unlink_memory(rid);
            self.cache_remove(rid);
            self.cache_invalidate_syntheses(&invalidated_syntheses);
        }

        Ok(was_newly_tombstoned)
    }

    /// Tombstone a memory. Returns `true` if the memory was found in a live
    /// state and newly tombstoned; `false` if rid was missing or already
    /// tombstoned (both treated as no-ops).
    ///
    /// Stamped with engine-supplied `now()` — for byte-deterministic
    /// cluster-replicated tombstones use [`tombstone_with_rid`] instead.
    /// `forget()` delegates to `tombstone_inner` with namespace lookup
    /// (the namespace is read from the row); the bool return is the only
    /// behavioral difference vs the cluster primitive.
    #[tracing::instrument(skip(self))]
    pub fn forget(&self, rid: &str) -> Result<bool> {
        let ts_micros = (now() * 1_000_000.0) as i64;
        self.tombstone_inner(rid, None, None, ts_micros, None)
    }

    /// Current search-state generation. The generation bumps monotonically on
    /// every reembed cutover (a `set_embedder` that keeps the same digest
    /// deliberately does NOT bump it), and it defines the vector space the
    /// HNSW index lives in. A caller that pre-embeds text (Python-embedder
    /// path) snapshots this BEFORE embedding and passes it to
    /// [`Self::correct_with_embedding`] so a reembed cutover that races the
    /// correction is detected (sol r8): the supplied vector is only accepted
    /// if the index is still at the same generation at commit.
    pub fn search_generation(&self) -> u64 {
        self.search_state.load_full().generation
    }

    /// User-initiated memory correction (Issue #47, v0.7.20).
    ///
    /// **In-place mutation with audit trail.** Updates the memory at
    /// `rid` to reflect the supplied text / metadata-merge / importance
    /// / valence changes, while:
    /// - preserving `rid` (no new memory minted)
    /// - preserving `created_at` (the memory's timeline anchor)
    /// - appending a row to `record_revisions` capturing the prior state
    /// - leaving inbound link integrity intact (graph edges, replication
    ///   audit log entries, knowledge graph references all continue to
    ///   resolve because the rid is unchanged)
    /// - logging a "correct" op for replication
    ///
    /// **Embedding NOT supported.** HNSW does not support in-place update
    /// of an existing vector, and rebuilding the cold tier on every
    /// correction is too expensive. Callers needing to change the
    /// embedding still use `forget()` + `record()` (the v0.7.19-and-
    /// earlier behaviour). The new `correct()` is for text + metadata
    /// + importance + valence only.
    ///
    /// **`reason` is required and must be non-empty.** The audit trail
    /// is load-bearing: it is what gives `correct()` its semantic value
    /// over a bare UPDATE. Empty / whitespace-only reasons are rejected
    /// with `YantrikDbError::InvalidInput`.
    ///
    /// **At least one mutation field must be supplied.** Passing all
    /// `None` for `new_text` / `metadata_merge` / `new_importance` /
    /// `new_valence` is a no-op correction and is rejected with
    /// `YantrikDbError::InvalidInput`.
    ///
    /// **Atomic.** The revision insert + memories UPDATE happen in a
    /// single SQL transaction. Either both succeed or neither does.
    #[tracing::instrument(skip(self))]
    pub fn correct(
        &self,
        rid: &str,
        new_text: Option<&str>,
        metadata_merge: Option<&serde_json::Value>,
        new_importance: Option<f64>,
        new_valence: Option<f64>,
        reason: &str,
    ) -> Result<CorrectionResult> {
        self.correct_impl(
            rid,
            new_text,
            None,
            metadata_merge,
            new_importance,
            new_valence,
            reason,
        )
    }

    /// **v0.10 Item 3 (Python-binding embedder parity).** As [`Self::correct`],
    /// but the caller supplies the embedding for `new_text` instead of the
    /// engine embedding it internally. Used by the Python binding when the
    /// user attached a Python-callable embedder via `set_embedder(obj)`: the
    /// re-embed runs pure-Rust and cannot call back into Python, so the
    /// binding pre-embeds on the Python thread (mirroring `record_text`) and
    /// hands the vector here.
    ///
    /// `embedding_generation` is the [`Self::search_generation`] value the
    /// caller snapshotted BEFORE embedding. It pins the vector to the space it
    /// was computed in: if a reembed cutover advanced the generation between
    /// that snapshot and commit, the vector is stale-space and the correction
    /// is rejected with the retryable `CorrectionDeferredDuringReembed` rather
    /// than durably committing a wrong-space vector (sol r8). `new_embedding`
    /// is validated for dim/finiteness on the text-changing path; it is
    /// ignored for metadata/scalar-only corrections. Coherence machinery is
    /// identical to `correct` — only the source of the vector differs.
    pub fn correct_with_embedding(
        &self,
        rid: &str,
        new_text: Option<&str>,
        new_embedding: &[f32],
        embedding_generation: u64,
        metadata_merge: Option<&serde_json::Value>,
        new_importance: Option<f64>,
        new_valence: Option<f64>,
        reason: &str,
    ) -> Result<CorrectionResult> {
        self.correct_impl(
            rid,
            new_text,
            Some((new_embedding, embedding_generation)),
            metadata_merge,
            new_importance,
            new_valence,
            reason,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn correct_impl(
        &self,
        rid: &str,
        new_text: Option<&str>,
        // Item 3 (sol r8): (vector, generation-it-was-embedded-against). The
        // generation pins the vector to its space so a racing reembed cutover
        // is caught before the wrong-space vector is committed.
        caller_embedding: Option<(&[f32], u64)>,
        metadata_merge: Option<&serde_json::Value>,
        new_importance: Option<f64>,
        new_valence: Option<f64>,
        reason: &str,
    ) -> Result<CorrectionResult> {
        // Validate reason non-empty (load-bearing audit field).
        let reason_trimmed = reason.trim();
        if reason_trimmed.is_empty() {
            return Err(YantrikDbError::InvalidInput(
                "correct: `reason` is required and must be non-empty; \
                 the audit trail is load-bearing"
                    .to_string(),
            ));
        }

        // Validate at least one mutation field is supplied.
        if new_text.is_none()
            && metadata_merge.is_none()
            && new_importance.is_none()
            && new_valence.is_none()
        {
            return Err(YantrikDbError::InvalidInput(
                "correct: at least one of `new_text` / `metadata_merge` / \
                 `new_importance` / `new_valence` must be supplied; \
                 a correction with no changes is a no-op"
                    .to_string(),
            ));
        }

        // Load without minting an outcome-anchor label — a correction that
        // hasn't happened yet is not evidence. The single note_caller_used
        // fires at the end of a SUCCESSFUL correction (both paths).
        let original = self
            .get_untracked(rid)?
            .ok_or_else(|| YantrikDbError::NotFound(format!("memory: {}", rid)))?;

        // **v0.10 Item 3: vector-coherent correction.** When the text
        // actually changes, the retrieval vector must change with it —
        // otherwise the durable "current truth" and the embedding disagree
        // forever ("Alice owns service A" corrected to "Bob owns service B"
        // keeps being retrieved for Alice/service-A queries). Route to the
        // staged re-embed protocol. A no-op text change (same bytes) and
        // metadata/scalar-only corrections skip re-embedding entirely.
        let text_changes = new_text.is_some_and(|t| t != original.text);
        if text_changes {
            return self.correct_with_reembed(
                rid,
                &original,
                new_text.expect("text_changes implies Some"),
                caller_embedding,
                metadata_merge,
                new_importance,
                new_valence,
                reason_trimmed,
            );
        }

        let ts = now();
        let hlc_bytes = self.tick_hlc().to_bytes().to_vec();
        let revision_id = crate::id::new_id();
        let _ = &original; // prior state is re-read inside the tx (below)

        // The tx holds the conn lock, which serialises concurrent
        // correct() calls on the same rid. Prior state is re-read HERE so
        // a second correction records the FIRST's committed state, not a
        // stale pre-loop snapshot (sol finding 7: correct/correct).
        let conn = self.conn.lock();
        // Correction seqlock (r4 finding 3): held across commit + cache so a
        // concurrent recall retries rather than mixing pre/post scalar state.
        let _epoch = self.enter_correction_epoch(&conn);
        let tx = conn.unchecked_transaction()?;

        // Current stored (encrypted-or-plain) state = this correction's
        // prior state. Stored representation is reused directly for the
        // revision row — no decrypt/re-encrypt round trip. Guard on active
        // status (finding 6): a forget() may have tombstoned the row.
        let cur_row: Option<(String, String, f64, f64)> = tx
            .query_row(
                "SELECT text, metadata, importance, valence FROM memories \
                 WHERE rid = ?1 AND consolidation_status = 'active'",
                params![rid],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .optional()?;
        let (cur_stored_text, cur_stored_metadata, cur_importance, cur_valence) = cur_row
            .ok_or_else(|| {
                YantrikDbError::NotFound(format!(
                    "memory {rid} is no longer active (forgotten during correction)"
                ))
            })?;

        // **Review finding 4 (metadata path):** re-classify text change
        // against the SERIALIZED in-tx state. new_text was compared to a
        // pre-lock snapshot in the caller; a concurrent text correction may
        // have changed the row since, so a `new_text` that looked unchanged
        // now actually differs — and writing it here without re-embedding
        // would pair new text with the old vector. Delegate to the
        // re-embedding path in that case.
        if let Some(t) = new_text {
            let cur_text_plain = self.decrypt_text(&cur_stored_text)?;
            if t != cur_text_plain {
                drop(tx);
                // Release THIS path's correction epoch BEFORE conn (the guard
                // borrows conn, so the compiler enforces this order):
                // correct_with_reembed enters its own guard, and two live
                // guards would double-toggle the epoch (odd→even)
                // mid-correction. Nothing was mutated here.
                drop(_epoch);
                drop(conn);
                return self.correct_with_reembed(
                    rid,
                    &original,
                    t,
                    caller_embedding,
                    metadata_merge,
                    new_importance,
                    new_valence,
                    reason_trimmed,
                );
            }
        }

        // Resolve new values by merging onto the CURRENT state.
        let new_importance_val = new_importance.unwrap_or(cur_importance);
        let new_valence_val = new_valence.unwrap_or(cur_valence);
        let new_text_val: String = match new_text {
            // Re-verified equal to current above. Sanitized like every
            // record surface — correct() was the one remaining path that
            // could persist a tool-call artifact tail (2026-08-15 surface
            // audit).
            Some(t) => crate::engine::sanitize::sanitize_tool_call_artifacts(t).into_owned(),
            None => self.decrypt_text(&cur_stored_text)?,
        };
        let cur_metadata_plain = self.decrypt_text(&cur_stored_metadata)?;
        let cur_metadata_val: serde_json::Value =
            serde_json::from_str(&cur_metadata_plain).unwrap_or(serde_json::Value::Null);
        let mut new_metadata_val: serde_json::Value = match metadata_merge {
            Some(patch) => {
                let mut merged = cur_metadata_val.clone();
                if let (Some(obj), Some(patch_obj)) = (merged.as_object_mut(), patch.as_object()) {
                    for (k, v) in patch_obj {
                        obj.insert(k.clone(), v.clone());
                    }
                    merged
                } else {
                    patch.clone()
                }
            }
            None => cur_metadata_val.clone(),
        };
        // EVENT TIME follows the text. Correct "March 15" to "April 20" and
        // the extracted event keys said March forever — metadata
        // contradicting its own text, the exact class 00edc87 closed for
        // partial caller keys, missed on the one surface that rewrites text
        // (2026-08-15 surface audit). On a text change the three keys are
        // re-derived AS ONE UNIT from the corrected prose — unless this very
        // correction's metadata_merge supplies any of them, in which case
        // the caller owns all three (the merge_event_dates ownership rule).
        if text_changes {
            let caller_owns_event_keys =
                metadata_merge.and_then(|p| p.as_object()).is_some_and(|p| {
                    ["event_dates", "event_time_min", "event_time_max"]
                        .iter()
                        .any(|k| p.contains_key(*k))
                });
            if !caller_owns_event_keys {
                if let Some(obj) = new_metadata_val.as_object_mut() {
                    for k in ["event_dates", "event_time_min", "event_time_max"] {
                        obj.remove(k);
                    }
                }
                new_metadata_val =
                    crate::base::datetext::merge_event_dates(&new_metadata_val, &new_text_val);
            }
        }
        // **v0.10 Item 4a.4b — anti-laundering gate on the correction path
        // (T06 fan-out).** A `metadata_merge` can flip `kind` to fact/observation
        // on an inference-sourced record, so the gate must see the FINAL MERGED
        // plaintext metadata (not the caller's patch) paired with the record's
        // existing `source` — `correct()` cannot change `source` itself. Runs
        // before the revision insert / UPDATE below, inside the tx, so a refusal
        // rolls back leaving the row untouched. Warn-mode flags are counted only
        // after the commit below (4a.6b).
        let gate_verdict = self.gate_provenance(&original.source, &new_metadata_val)?;
        let stored_new_text = self.encrypt_text(&new_text_val)?;
        let stored_new_metadata = self.encrypt_text(&serde_json::to_string(&new_metadata_val)?)?;
        // v48 (#149): re-stamp the event-time columns from the SAME plaintext
        // value just serialized (pre-encryption) — a correction that rewrites
        // metadata must never leave the columns describing the old JSON.
        let (event_time_min, event_time_max) =
            crate::base::datetext::event_time_bounds(&new_metadata_val);

        let next_revision_num: i64 = tx
            .query_row(
                "SELECT COALESCE(MAX(revision_num), 0) + 1 \
                 FROM record_revisions WHERE rid = ?1",
                params![rid],
                |row| row.get(0),
            )
            .unwrap_or(1);

        // Insert the revision row capturing the re-read prior state.
        tx.execute(
            "INSERT INTO record_revisions \
             (revision_id, rid, revision_num, prior_text, prior_metadata, \
              prior_importance, prior_valence, reason, applied_at, hlc, origin_actor) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                revision_id,
                rid,
                next_revision_num,
                cur_stored_text,
                cur_stored_metadata,
                cur_importance,
                cur_valence,
                reason_trimmed,
                ts,
                hlc_bytes,
                self.actor_id,
            ],
        )?;

        // UPDATE the memory in place. rid + created_at + embedding are
        // not touched (metadata/scalar-only path — the text is unchanged,
        // so the vector stays coherent). last_access is bumped.
        tx.execute(
            "UPDATE memories \
             SET text = ?1, metadata = ?2, importance = ?3, valence = ?4, \
                 last_access = ?5, event_time_min = ?7, event_time_max = ?8 \
             WHERE rid = ?6",
            params![
                stored_new_text,
                stored_new_metadata,
                new_importance_val,
                new_valence_val,
                ts,
                rid,
                // v48 (#149) event time, from the merged plaintext metadata.
                event_time_min,
                event_time_max,
            ],
        )?;

        // **v0.10 Item 3 (sol boundary #6):** the replication op is written
        // INSIDE this transaction. A kill between the memories mutation and
        // the oplog append can no longer lose the correction's replication
        // event (the old path called log_op post-commit in its own conn).
        // No embedding bytes here — this path leaves the vector untouched.
        let applied_generation = self.search_state.load().generation as i64;
        self.insert_correct_op_in_tx(
            &tx,
            rid,
            &serde_json::json!({
                "rid": rid,
                "revision_num": next_revision_num,
                "new_text": new_text,
                "metadata_merge": metadata_merge,
                "new_importance": new_importance,
                "new_valence": new_valence,
                "reason": reason_trimmed,
                "applied_at": ts,
            }),
            None,
            None,
            applied_generation,
        )?;

        // Maintenance-debt ledger: a correction rewrites content — new
        // material for cognition (the corrected claim may now conflict with
        // records the old text agreed with). Counted inside the tx, atomic
        // with the mutation it counts.
        Self::bump_writes_since_think_on(&tx, 1)?;

        let invalidated_syntheses = Self::invalidate_synthesis_dependents_in_tx(&tx, rid)?;

        tx.commit()?;

        // 4a.6b: the correction is durable — a warn-mode flag counts now.
        self.note_flagged_write_committed(gate_verdict);

        // Refresh the scoring_cache BEFORE dropping conn (finding 4): two
        // serialized corrections must update the cache in commit order, not
        // race after releasing the lock. Both importance AND valence are
        // ranking-relevant; text + metadata re-hydrate from SQL on read.
        {
            let mut cache = self.scoring_cache.write();
            if let Some(row) = cache.get_mut(rid) {
                row.importance = new_importance_val;
                row.valence = new_valence_val;
                row.last_access = ts;
            }
        }
        self.cache_invalidate_syntheses(&invalidated_syntheses);
        // Epoch guard (borrows conn) drops BEFORE conn — restores an even
        // epoch while this correction still holds conn, so no second
        // correction can overlap into a falsely-even window (sol r5 #1).
        drop(_epoch);
        drop(conn);

        // v0.10 Item 2 outcome anchor: a correction is an independent
        // caller action targeting the rid — the ranker surfaced a memory
        // the caller cared enough to fix. The label measures RETRIEVAL
        // utility, not content truth (a wrong-but-found memory was still
        // the right retrieval).
        self.note_caller_used(rid);

        Ok(CorrectionResult {
            original_rid: rid.to_string(),
            corrected_rid: rid.to_string(),
            original_tombstoned: false,
            revision_num: next_revision_num,
        })
    }

    /// **v0.10 Item 3 (sol boundary #6).** Insert the `correct` replication
    /// op INSIDE the caller's correction transaction. `log_op` runs in its
    /// own connection AFTER commit, so a kill in the gap loses the
    /// replication event; writing the op in the same transaction makes the
    /// mutation and its intent atomic. `embedding` carries the EXACT
    /// re-embedded bytes (encrypted like `memories.embedding`) for a text
    /// correction so a follower applies them verbatim rather than
    /// re-embedding — follower re-embedding diverges by model
    /// version/quantization. `None` for metadata/scalar corrections.
    #[allow(clippy::too_many_arguments)]
    /// **Entity-graph coherence — the correction "safety half" (nuron, v0.10
    /// Item-3 follow-up).** A text-changing `correct()` re-embeds the vector
    /// but the `memory_entities` links were extracted from the OLD text. A link
    /// whose entity surface string no longer appears in the corrected text is
    /// STALE and keeps serving this record under its old association through
    /// graph expansion (`why_retrieved = ["graph-connected via <old-entity>"]`)
    /// — the reembed-independent, always-on face of the same "old meaning still
    /// served" problem the vector path already closes. Drop those links.
    ///
    /// Matching reuses the SAME tokenizer/matcher entity EXTRACTION uses
    /// (`graph::entity_matches_text` over `graph::tokenize`) so the two agree —
    /// crucially, that is WORD-BOUNDARY (token-equality, not substring), so a
    /// short entity can't false-KEEP by colliding inside a larger word
    /// (`"Volkan"` tokens ≠ `"Volkanic"` tokens). A false-KEEP would silently
    /// preserve the harm while the code looks like it ran; a false-DROP (entity
    /// referred to by an alias) is benign under-retrieval. Err toward dropping.
    ///
    /// Runs in the correction's transaction so the removal is atomic with the
    /// text change. Adding links for NEW entities (re-extraction) is the
    /// deferrable "completeness half" and is intentionally NOT done here.
    fn drop_stale_memory_entity_links_in_tx(
        tx: &rusqlite::Transaction<'_>,
        rid: &str,
        new_text_plain: &str,
    ) -> Result<()> {
        let linked: Vec<String> = {
            let mut stmt =
                tx.prepare("SELECT entity_name FROM memory_entities WHERE memory_rid = ?1")?;
            let rows = stmt.query_map(params![rid], |r| r.get::<_, String>(0))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        if linked.is_empty() {
            return Ok(());
        }
        let tokens = crate::graph::tokenize(new_text_plain);
        for entity in &linked {
            if !crate::graph::entity_matches_text(entity, &tokens) {
                tx.execute(
                    "DELETE FROM memory_entities WHERE memory_rid = ?1 AND entity_name = ?2",
                    params![rid, entity],
                )?;
            }
        }
        Ok(())
    }

    /// **Entity-graph coherence — in-memory index eviction (nuron live-verify
    /// finding, v0.10 Item-3 follow-up).** `drop_stale_memory_entity_links_in_tx`
    /// deletes the DURABLE `memory_entities` rows, but recall's `expand_entities`
    /// reads the in-memory `graph_index`, which the correction transaction never
    /// touches (graph_ops.rs documents "graph_index retains the edge until the
    /// next engine reload"). So durable-only dropping left the corrected record
    /// still served under its OLD association through graph expansion — a
    /// green-unit-test/red-live-recall index-staleness bug the durable-table
    /// assertion could not catch.
    ///
    /// Re-derive this memory's in-memory links from the now-committed,
    /// authoritative durable rows: clear its links and re-link the survivors.
    /// Cheap (O(links for this one memory)) and atomic to recall readers under
    /// the graph_index write lock. Call AFTER the correction commits and its
    /// conn lock is released — the brief re-lock here keeps `conn` and
    /// `graph_index` from being held simultaneously (record's graph path also
    /// releases conn before taking graph_index, so the lock order is preserved).
    fn resync_memory_graph_links_after_correction(&self, rid: &str) -> Result<()> {
        let survivors: Vec<String> = {
            let conn = self.conn.lock();
            let mut stmt =
                conn.prepare("SELECT entity_name FROM memory_entities WHERE memory_rid = ?1")?;
            let rows = stmt.query_map(params![rid], |r| r.get::<_, String>(0))?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        let mut gi = self.graph_index.write();
        gi.unlink_memory(rid);
        for e in &survivors {
            gi.link_memory(rid, e);
        }
        Ok(())
    }

    /// Thin wrapper over the generalized [`YantrikDB::log_op_in_tx`] (Item 4a.6a).
    ///
    /// This function WAS the only in-transaction oplog writer in the tree; 4a.6a
    /// generalized its body so `record()` could adopt the same protocol. Kept as
    /// a named wrapper because `correct()`'s call site reads better for it, and
    /// because it pins `op_type` to `'correct'` at one place.
    fn insert_correct_op_in_tx(
        &self,
        tx: &rusqlite::Transaction<'_>,
        rid: &str,
        payload: &serde_json::Value,
        emb_hash: Option<&[u8]>,
        embedding: Option<&[u8]>,
        applied_generation: i64,
    ) -> Result<()> {
        self.log_op_in_tx(
            tx,
            "correct",
            Some(rid),
            payload,
            emb_hash,
            embedding,
            applied_generation,
            None,
        )?;
        Ok(())
    }

    /// **v0.10 Item 3 — RID-stable, vector-coherent correction.**
    ///
    /// The staged protocol for a text-changing correction (design: sol's
    /// 8-boundary audit, docs/V0.10_PLAN.md Item 3). The rid, created_at,
    /// and revision chain are preserved; only the content and its retrieval
    /// vector change.
    ///
    /// Ordering and why:
    /// 1. **Embed OUTSIDE the write-router guard** (slow step), mirroring
    ///    `record_text`'s revalidation loop — reembed throughput stays
    ///    bounded by the index rebuild, not by in-flight corrections.
    /// 2. **Guard + revalidate**: if a `reembed()` swap landed between the
    ///    embed and the guard, the vector is in the wrong space — retry.
    ///    If the router is mid-cutover (Queueing), return the typed
    ///    `CorrectionDeferredDuringReembed` (retryable; no state touched).
    /// 3. **Backpressure BEFORE any visible mutation**: under the guard the
    ///    delta can only shrink (compactor) — never grow (other sync
    ///    writers and reembed are excluded) — so a capacity check here
    ///    makes the later tombstone+append infallible. If the delta is
    ///    full, return `Backpressure` having changed nothing.
    /// 4. **One SQL transaction**: revision row (with prior embedding
    ///    model+hash) + memories UPDATE (text, metadata, scalars,
    ///    embedding, embedding_generation) + the `correct` oplog op with
    ///    exact new bytes (boundary #6). Kill before commit → nothing
    ///    changed; kill after → SQL is fully consistent and the index
    ///    rebuilds from it on reopen.
    /// 5. **Delta tombstone+append as one sealed op** after commit, then
    ///    scoring-cache + visible_seq in the same critical section.
    #[allow(clippy::too_many_arguments)]
    fn correct_with_reembed(
        &self,
        rid: &str,
        original: &Memory,
        new_text: &str,
        // Item 3 (Python-binding parity): when `Some((vec, gen))`, the caller
        // already embedded `new_text` (it holds a Python-callable embedder the
        // pure-Rust re-embed cannot reach) and `gen` is the search generation
        // it embedded against. When `None`, the engine embeds using its native
        // `search_state.embedder`. Coherence is identical either way — only the
        // vector's source differs, and `gen` pins the caller's vector to its
        // space (sol r8).
        caller_embedding: Option<(&[f32], u64)>,
        metadata_merge: Option<&serde_json::Value>,
        new_importance: Option<f64>,
        new_valence: Option<f64>,
        reason_trimmed: &str,
    ) -> Result<CorrectionResult> {
        use crate::serde_helpers::{deserialize_f32, serialize_f32};

        // Namespace is immutable across a correction; everything else
        // (text, metadata, scalars, embedding, model) is re-read from SQL
        // INSIDE the serialized critical section below so two concurrent
        // corrections record each other's true prior state (sol finding 7:
        // correct/correct must not both snapshot the same original).
        let namespace = original.namespace.clone();

        // Revalidation loop — mirrors record_text (mod.rs). Bounded in
        // expectation: reembed advances the generation monotonically and
        // completes at most once per call.
        loop {
            // Step 1: snapshot for the embed. Only require a native embedder
            // when the caller did NOT supply one (Item 3 Python parity).
            let state_for_embed = self.search_state.load_full();
            let gen_pre = state_for_embed.generation;
            let digest_pre = state_for_embed.runtime_embedder_digest.clone();

            // sol r8: a caller-supplied vector was embedded against a specific
            // generation. If a reembed cutover advanced the generation between
            // that snapshot and now, the vector is in the OLD space and must
            // NOT be committed against the new index. Reject retryably BEFORE
            // any side effect; the caller re-embeds against the new generation
            // and reissues. (The engine-embedded `None` path re-embeds every
            // iteration, so it is covered by the Step-5 generation recheck
            // instead.)
            if let Some((_, caller_gen)) = caller_embedding {
                if caller_gen != gen_pre {
                    return Err(YantrikDbError::CorrectionDeferredDuringReembed {
                        rid: rid.to_string(),
                    });
                }
            }

            let embedder = match caller_embedding {
                Some(_) => None,
                None => Some(
                    state_for_embed
                        .embedder
                        .as_ref()
                        .ok_or(YantrikDbError::NoEmbedder)?
                        .clone(),
                ),
            };
            let dim_pre = state_for_embed.dim();
            drop(state_for_embed);

            // Step 2: obtain the new embedding OUTSIDE any guard (slow). It
            // depends only on new_text, so it needs no prior state and is safe
            // to compute before the critical section. Either the caller
            // supplied it (Python embedder) or the engine embeds natively.
            let new_embedding = match caller_embedding {
                Some((e, _)) => e.to_vec(),
                None => embedder
                    .as_ref()
                    .expect("embedder present when caller_embedding is None")
                    .embed(new_text)
                    .map_err(|e| YantrikDbError::Inference(e.to_string()))?,
            };
            crate::validate::validate_embedding("correct", &new_embedding, dim_pre)?;

            // Chunked embeddings for the corrected text. Engine-embedded
            // path: re-chunk under the same snapshot embedder (the step-5
            // revalidation covers the whole set). Caller-supplied vector:
            // the engine cannot chunk text it did not embed — the old
            // windows are PURGED below and the record becomes head-only
            // (a recall regression `rechunk_long_records()` can repay;
            // stale windows would keep serving the pre-correction text,
            // which is silent corruption).
            let new_chunks: Vec<(usize, Vec<f32>)> = match (caller_embedding, embedder.as_ref()) {
                (None, Some(embedder)) => match self.chunk_plan(new_text) {
                    Some(ranges) => {
                        let mut cv = Vec::with_capacity(ranges.len());
                        for (i, (a, b)) in ranges.iter().enumerate() {
                            let v = embedder
                                .embed(&new_text[*a..*b])
                                .map_err(|e| YantrikDbError::Inference(e.to_string()))?;
                            crate::validate::validate_embedding("correct#chunk", &v, dim_pre)?;
                            cv.push((i + 1, v));
                        }
                        cv
                    }
                    None => Vec::new(),
                },
                _ => Vec::new(),
            };

            // Step 3: enter the sync path. A reembed SWAP is in flight
            // (router Queueing) → typed, retryable, nothing touched. The
            // Encoding/Rebuilding window does NOT set Queueing; that window
            // is handled by clearing the staging columns in the UPDATE
            // below (sol finding 4).
            let sync_guard = match self.write_router.try_enter_sync_writer() {
                Some(g) => g,
                None => {
                    return Err(YantrikDbError::CorrectionDeferredDuringReembed {
                        rid: rid.to_string(),
                    });
                }
            };

            // Step 4: re-snapshot under the guard. From here reembed cannot
            // complete its swap until the guard drops.
            let state_for_commit = self.search_state.load_full();

            // Step 5: revalidate — a swap between step 1 and step 4 means
            // the embedding is in the wrong vector space.
            if state_for_commit.generation != gen_pre
                || state_for_commit.runtime_embedder_digest != digest_pre
            {
                drop(sync_guard);
                // A caller-supplied embedding (Python embedder) cannot be
                // re-embedded here into the new space — the engine has no
                // native embedder for this text. Surface the retryable
                // deferred error; the caller re-embeds and reissues. Nothing
                // was mutated. (Unreachable in practice for a pure-Python-
                // embedder DB: db.reembed() needs a native embedder, so the
                // generation cannot advance from a cutover here.)
                if caller_embedding.is_some() {
                    return Err(YantrikDbError::CorrectionDeferredDuringReembed {
                        rid: rid.to_string(),
                    });
                }
                tracing::info!(
                    gen_pre,
                    gen_post = state_for_commit.generation,
                    "correct_with_reembed: SearchState advanced mid-embed, retrying",
                );
                continue;
            }
            let generation = state_for_commit.generation as i64;
            let runtime_model = state_for_commit.runtime_embedder_name.clone();

            let ts = now();
            let hlc_bytes = self.tick_hlc().to_bytes().to_vec();
            let revision_id = crate::id::new_id();
            let stored_new_emb = self.encrypt_embedding(&serialize_f32(&new_embedding))?;
            // Chunk blobs + keys prepared before the conn lock (CPU work).
            let stored_new_chunks: Vec<(usize, String, Vec<u8>)> = new_chunks
                .iter()
                .map(|(idx, v)| {
                    let blob = self.encrypt_embedding(&serialize_f32(v))?;
                    Ok((*idx, crate::vector::chunk::chunk_key(rid, *idx), blob))
                })
                .collect::<Result<_>>()?;
            let new_emb_hash = embedding_hash(&new_embedding).to_vec();

            // The conn lock is held across the delta append AND the SQL
            // commit, so corrections to the SAME rid serialize: the append
            // order equals the commit order, and no other correction can
            // interleave its append between our append and our commit (the
            // SyncWriteGuard is a counter, NOT a writer lock, so it cannot
            // provide this ordering on its own).
            let conn = self.conn.lock();

            // **Correction seqlock (r4 finding 3).** Enter the epoch NOW,
            // under the conn lock and before any mutation: the SQL commit +
            // vector publish + cache update below are odd-epoch; a concurrent
            // recall whose candidate generation straddles this span detects
            // the change and retries, so a result can never mix this
            // correction's new text with its old ranking vector.
            let _epoch = self.enter_correction_epoch(&conn);

            // **Allocate seq UNDER the conn lock** (v2-review finding 1):
            // search picks the HIGHEST seq, not the most recently appended.
            // If seq were minted before the lock, a stalled C1 (seq N) could
            // append after C2 (seq N+1) committed, and search would keep
            // C1's older-seq... no — keep C2's higher seq while SQL holds
            // C1's text. Minting seq here ties seq order to the serialized
            // append+commit order, so highest seq == last committed text.
            let seq_new = self
                .vec_seq
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;

            // **RESERVE the vector slot via an unpublished superseding
            // append** (review finding 2). The reserved entry holds delta
            // capacity (so Backpressure is reported here, before any SQL
            // change) but is INVISIBLE to search and SKIPPED by compaction
            // until we publish it post-commit. This closes two races the
            // plain-append version had: (a) compaction could seal the
            // uncommitted vector into cold before a commit-failure could
            // remove it; (b) a reader on a separate connection could see the
            // new vector paired with the not-yet-committed old text. While
            // reserved, search still returns the old durable vector.
            // `false` = an identical (rid, seq_new) already exists. seq_new is
            // freshly minted for this correction, so that is an invariant
            // violation — and constructing the guard below over an entry this
            // call does not own would let a failure path remove another
            // write's published vector (4a.6d-3, sol r1 finding 1).
            match state_for_commit.vec_index.append_reserved(
                rid.to_string(),
                new_embedding.clone(),
                seq_new,
            ) {
                Ok(crate::vector::delta_index::ReservedAppend::Inserted) => {}
                Ok(crate::vector::delta_index::ReservedAppend::AlreadyPresent) => {
                    drop(_epoch);
                    drop(conn);
                    drop(sync_guard);
                    return Err(crate::error::YantrikDbError::InvalidInput(format!(
                        "freshly minted seq {seq_new} for rid {rid} already \
                         present in the delta — engine invariant violation"
                    )));
                }
                Err(e) => {
                    drop(_epoch);
                    drop(conn);
                    drop(sync_guard);
                    return Err(e);
                }
            }

            // From here the reservation is owed back on EVERY exit — including an
            // unwinding panic. The hand-rolled `remove_appended` this replaces
            // could not cover that: a panic inside the commit closure unwound
            // straight past it and leaked the reservation permanently (compaction
            // retains unpublished entries by design, so nobody ever reclaims it,
            // and enough leaks wedge every writer into Backpressure).
            //
            // publish_only, NOT with_pending_op: this transaction's `correct` op
            // commits via log_op_in_tx with applied=1, so it creates no pending
            // op. Incrementing pending_op_count here would inflate the cache
            // against zero pending rows — the v0.7.1 counter-leak class, in the
            // other direction.
            let mut reservation = ReservationGuard::publish_only(&state_for_commit, rid, seq_new);

            // Reserve the corrected text's window keys at the SAME seq —
            // one correction, one commit point, one guard. Highest-seq-wins
            // means each new window supersedes its old published entry by
            // key; SURPLUS old keys (new text has fewer windows) are
            // tombstoned post-commit below.
            for ((_, v), (_, key, _)) in new_chunks.iter().zip(stored_new_chunks.iter()) {
                match state_for_commit
                    .vec_index
                    .append_reserved(key.clone(), v.clone(), seq_new)
                {
                    Ok(crate::vector::delta_index::ReservedAppend::Inserted) => {
                        reservation.add_chunk_key(key.clone());
                    }
                    Ok(crate::vector::delta_index::ReservedAppend::AlreadyPresent) => {
                        drop(reservation);
                        drop(_epoch);
                        drop(conn);
                        drop(sync_guard);
                        return Err(crate::error::YantrikDbError::InvalidInput(format!(
                            "freshly minted seq {seq_new} for chunk key {key} already \
                             present in the delta — engine invariant violation"
                        )));
                    }
                    Err(e) => {
                        drop(reservation);
                        drop(_epoch);
                        drop(conn);
                        drop(sync_guard);
                        return Err(e);
                    }
                }
            }

            // SQL transaction. Prior state is re-read HERE (under the
            // serialized conn lock) so it reflects any correction that
            // committed just before us.
            //
            // 4a.6b: the gate verdict escapes the closure so a warn-mode flag
            // can be counted post-commit. `None` only on a pre-gate error path.
            let mut gate_verdict: Option<crate::provenance::GateVerdict> = None;
            // The OLD text's window idxs escape the closure too: surplus keys
            // (old windows the corrected text no longer has) are tombstoned
            // post-commit, and only a committed tx knows the true prior set.
            let mut old_chunk_idxs: Vec<i64> = Vec::new();
            let mut invalidated_syntheses: Vec<String> = Vec::new();
            let commit_result: Result<i64> =
                (|| {
                    let tx = conn.unchecked_transaction()?;

                    // True current state = this correction's prior state.
                    // **Review finding 6.** Revalidate active status HERE, under
                    // the serialized conn lock, AFTER the slow embed: a forget()
                    // may have tombstoned the row while we were embedding. If so,
                    // bail — the outer match removes our superseding append, so a
                    // correction never resurrects a forgotten record's vector.
                    #[allow(clippy::type_complexity)]
                let row: Option<(String, String, f64, f64, Option<Vec<u8>>, Option<String>)> = tx
                    .query_row(
                        "SELECT text, metadata, importance, valence, embedding, embedding_model \
                         FROM memories WHERE rid = ?1 AND consolidation_status = 'active'",
                        params![rid],
                        |r| {
                            Ok((
                                r.get(0)?,
                                r.get(1)?,
                                r.get(2)?,
                                r.get(3)?,
                                r.get::<_, Option<Vec<u8>>>(4)?,
                                r.get::<_, Option<String>>(5)?,
                            ))
                        },
                    )
                    .optional()?;
                    let (cur_text, cur_meta_str, cur_importance, cur_valence, cur_emb, cur_model) =
                        row.ok_or_else(|| {
                            YantrikDbError::NotFound(format!(
                                "memory {rid} is no longer active (forgotten during correction)"
                            ))
                        })?;
                    let prior_meta_plain = self.decrypt_text(&cur_meta_str)?;
                    let prior_meta_val: serde_json::Value =
                        serde_json::from_str(&prior_meta_plain).unwrap_or(serde_json::Value::Null);
                    let prior_embedding_hash: Option<Vec<u8>> = match &cur_emb {
                        Some(blob) => {
                            let plain = self.decrypt_embedding(blob)?;
                            Some(embedding_hash(&deserialize_f32(&plain)).to_vec())
                        }
                        None => None,
                    };

                    // Merge onto CURRENT metadata; resolve scalars from current.
                    let new_importance_val = new_importance.unwrap_or(cur_importance);
                    let new_valence_val = new_valence.unwrap_or(cur_valence);
                    let new_metadata_val: serde_json::Value = match metadata_merge {
                        Some(patch) => {
                            let mut merged = prior_meta_val.clone();
                            if let (Some(obj), Some(patch_obj)) =
                                (merged.as_object_mut(), patch.as_object())
                            {
                                for (k, v) in patch_obj {
                                    obj.insert(k.clone(), v.clone());
                                }
                                merged
                            } else {
                                patch.clone()
                            }
                        }
                        None => prior_meta_val.clone(),
                    };
                    // EVENT TIME follows the corrected text — on THIS path,
                    // because `text_changes` dispatches here before the
                    // scalar-path block runs (the same dispatch trap the
                    // anti-laundering gate below documents; the scalar-path
                    // twin of this block is defensive only). Re-derive the
                    // three keys as ONE UNIT from the new prose unless this
                    // correction's own metadata_merge supplies any of them.
                    let mut new_metadata_val = new_metadata_val;
                    let caller_owns_event_keys =
                        metadata_merge.and_then(|p| p.as_object()).is_some_and(|p| {
                            ["event_dates", "event_time_min", "event_time_max"]
                                .iter()
                                .any(|k| p.contains_key(*k))
                        });
                    if !caller_owns_event_keys {
                        if let Some(obj) = new_metadata_val.as_object_mut() {
                            for k in ["event_dates", "event_time_min", "event_time_max"] {
                                obj.remove(k);
                            }
                        }
                        new_metadata_val =
                            crate::base::datetext::merge_event_dates(&new_metadata_val, new_text);
                    }
                    // **Anti-laundering gate — the text-changing correction was
                    // a BYPASS (found while wiring 4a.6b).** 4a.4b gated the
                    // scalar-only path (correct_impl), but `text_changes`
                    // dispatches HERE before that gate runs, and this path
                    // merges `metadata_merge` all the same — so
                    // `correct(new_text=<any one-char change>,
                    // metadata_merge={"kind":"fact"})` on an inference-sourced
                    // record laundered straight past Enforce. Verified
                    // empirically before fixing: the scalar flip refused, the
                    // text-change flip committed kind=fact. Gate on the FINAL
                    // merged metadata, inside the tx (a refusal rolls back and
                    // the outer match removes the reserved append), same design
                    // as the scalar site. `original.source` is stable —
                    // correct() cannot change source.
                    let v = self.gate_provenance(&original.source, &new_metadata_val)?;
                    gate_verdict = Some(v);
                    let stored_new_text = self.encrypt_text(new_text)?;
                    let stored_new_metadata =
                        self.encrypt_text(&serde_json::to_string(&new_metadata_val)?)?;
                    // v48 (#149): re-stamp the event-time columns from the
                    // SAME plaintext value just serialized (pre-encryption).
                    let (event_time_min, event_time_max) =
                        crate::base::datetext::event_time_bounds(&new_metadata_val);

                    let n: i64 = tx
                        .query_row(
                            "SELECT COALESCE(MAX(revision_num), 0) + 1 \
                         FROM record_revisions WHERE rid = ?1",
                            params![rid],
                            |row| row.get(0),
                        )
                        .unwrap_or(1);
                    tx.execute(
                        "INSERT INTO record_revisions \
                     (revision_id, rid, revision_num, prior_text, prior_metadata, \
                      prior_importance, prior_valence, reason, applied_at, hlc, \
                      origin_actor, prior_embedding_model, prior_embedding_hash) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                        params![
                            revision_id,
                            rid,
                            n,
                            cur_text,
                            cur_meta_str,
                            cur_importance,
                            cur_valence,
                            reason_trimmed,
                            ts,
                            hlc_bytes,
                            self.actor_id,
                            cur_model,
                            prior_embedding_hash,
                        ],
                    )?;
                    // rid + created_at preserved; text, metadata, scalars, AND
                    // the embedding change together. embedding_model is stamped
                    // with the runtime model. The reembed STAGING columns are
                    // CLEARED (sol finding 4): if a reembed is mid-Encoding, its
                    // swap must not promote a stale staged vector for the OLD
                    // text over this correction — a NULL embedding_new makes the
                    // reembed re-encode this row from the new text instead.
                    tx.execute(
                        "UPDATE memories \
                     SET text = ?1, metadata = ?2, importance = ?3, valence = ?4, \
                         embedding = ?5, embedding_generation = ?6, embedding_model = ?7, \
                         embedding_new = NULL, embedding_new_model = NULL, \
                         updated_at = ?8, last_access = ?8, \
                         event_time_min = ?10, event_time_max = ?11 \
                     WHERE rid = ?9",
                        params![
                            stored_new_text,
                            stored_new_metadata,
                            new_importance_val,
                            new_valence_val,
                            stored_new_emb,
                            generation,
                            runtime_model,
                            ts,
                            rid,
                            // v48 (#149) event time, from the merged
                            // plaintext metadata.
                            event_time_min,
                            event_time_max,
                        ],
                    )?;
                    // **Entity-graph coherence — safety half (nuron finding).**
                    // The vector is re-embedded above, but the memory→entity
                    // links were extracted from the OLD text; a link whose
                    // entity no longer appears in the corrected text keeps
                    // serving this record under its old association via graph
                    // expansion. Drop those stale links IN this tx (atomic with
                    // the text change). Adding links for NEW entities is the
                    // deferrable completeness half, not done here.
                    // Chunked embeddings: replace the window rows atomically
                    // with the text they describe. The old idx set is read
                    // first so the post-commit step can tombstone surplus
                    // keys — without that, a correction that shrinks the
                    // window count leaves the trailing old windows serving
                    // vanished text forever.
                    {
                        let mut idx_stmt =
                            tx.prepare("SELECT chunk_idx FROM memory_chunks WHERE rid = ?1")?;
                        old_chunk_idxs = idx_stmt
                            .query_map(params![rid], |r| r.get(0))?
                            .collect::<std::result::Result<_, _>>()?;
                    }
                    tx.execute("DELETE FROM memory_chunks WHERE rid = ?1", params![rid])?;
                    for (idx, _, blob) in &stored_new_chunks {
                        tx.execute(
                            "INSERT INTO memory_chunks (rid, chunk_idx, embedding) \
                             VALUES (?1, ?2, ?3)",
                            params![rid, *idx as i64, blob],
                        )?;
                    }
                    Self::drop_stale_memory_entity_links_in_tx(&tx, rid, new_text)?;
                    self.insert_correct_op_in_tx(
                        &tx,
                        rid,
                        &serde_json::json!({
                            "rid": rid,
                            "revision_num": n,
                            "new_text": new_text,
                            "metadata_merge": metadata_merge,
                            "new_importance": new_importance,
                            "new_valence": new_valence,
                            "reason": reason_trimmed,
                            "applied_at": ts,
                            "reembedded": true,
                            // v0.10 Item 3 finding 5: the embedding's model,
                            // so a follower validates its vector space matches
                            // before applying the exact bytes (else it falls
                            // back to re-embedding on rebuild).
                            "embedding_model": runtime_model,
                            "prior_embedding_model": cur_model,
                            "prior_embedding_hash": prior_embedding_hash
                                .as_ref()
                                .map(|b| b.iter().map(|x| format!("{x:02x}")).collect::<String>()),
                        }),
                        Some(&new_emb_hash),
                        Some(&stored_new_emb),
                        generation,
                    )?;
                    // Maintenance-debt ledger: the text-changing correction
                    // path — same rationale as the metadata/scalar path's
                    // count, same in-tx atomicity.
                    Self::bump_writes_since_think_on(&tx, 1)?;
                    invalidated_syntheses = Self::invalidate_synthesis_dependents_in_tx(&tx, rid)?;
                    tx.commit()?;
                    let _ = (new_importance_val, new_valence_val); // used in the UPDATE above
                    Ok(n)
                })();

            let next_revision_num = match commit_result {
                Ok(n) => n,
                Err(e) => {
                    // Commit failed (rare IO error) OR the row was forgotten
                    // mid-correction (NotFound from the active-status guard).
                    // The guard is still in Reserved phase, so dropping it here
                    // removes the reservation — the delta returns to its
                    // pre-correction visibility (old vector shows / rid stays
                    // forgotten), matching the unchanged SQL. Compaction could
                    // not have sealed it (reserved entries are skipped by the
                    // seal), so the removal always succeeds.
                    drop(reservation);
                    drop(_epoch);
                    drop(conn);
                    drop(sync_guard);
                    return Err(e);
                }
            };

            // The correction is durable: the obligation INVERTS here from
            // "remove the reservation" to "publish it". Nothing fallible may sit
            // between the commit and this call.
            reservation.mark_committed();

            // **Publish** the reserved append now that the correction is
            // durable: it becomes the visible, compaction-eligible live
            // vector, atomically superseding the old one. Still under the
            // conn lock, so the publish order matches the commit order.
            //
            // Via the guard, so an unwind between the commit and here still
            // publishes rather than stranding a durable row behind an invisible
            // vector (only an index rebuild would have recovered it).
            reservation.complete();

            // Surplus window keys: every old idx the corrected text no longer
            // produces gets a tombstone at seq_new. (Keys the new text kept
            // were superseded by their own higher-seq publish above; the
            // caller-supplied-vector path re-chunks nothing, so ALL old idxs
            // are surplus there.) Still under the conn lock, matching the
            // publish-order discipline.
            {
                let kept: std::collections::HashSet<usize> =
                    stored_new_chunks.iter().map(|(i, _, _)| *i).collect();
                for old in &old_chunk_idxs {
                    if !kept.contains(&(*old as usize)) {
                        let key = crate::vector::chunk::chunk_key(rid, *old as usize);
                        state_for_commit.vec_index.tombstone(&key, seq_new);
                    }
                }
            }
            self.cache_invalidate_syntheses(&invalidated_syntheses);

            // 4a.6b: durable — a warn-mode flag counts now. Set on every path
            // that reaches the commit (the gate runs before any tx write).
            if let Some(v) = gate_verdict {
                self.note_flagged_write_committed(v);
            }

            // Kill boundary. A crash HERE (after the durable commit) leaves
            // SQL wholly-new — new text + embedding + revision + correct op
            // all committed — and the index rebuilds from SQL on reopen.
            crate::testing::fail_point("correct.between_commit_and_delta");

            // Scoring cache — importance/valence are the ranking-relevant
            // fields the cache holds; text/metadata re-hydrate from SQL.
            {
                let mut cache = self.scoring_cache.write();
                if let Some(row) = cache.get_mut(rid) {
                    row.importance = new_importance.unwrap_or(row.importance);
                    row.valence = new_valence.unwrap_or(row.valence);
                    row.last_access = ts;
                }
            }

            // visible_seq is published LAST (sol finding 7): a
            // recall_with_seq waiter must not wake and rank against a
            // half-applied correction (stale scoring cache).
            self.bump_visible_seq(&namespace, seq_new);

            // Epoch guard (borrows conn) drops BEFORE conn — even epoch is
            // restored while this correction still holds conn (sol r5 #1).
            drop(_epoch);
            drop(conn);
            drop(sync_guard);

            // Entity-graph coherence: propagate the durable stale-link drop to
            // the in-memory graph_index recall reads (nuron). Done AFTER conn is
            // released so conn + graph_index are never held together. A failure
            // here degrades to the pre-fix "stale until reload" behavior rather
            // than failing an already-committed correction.
            if let Err(e) = self.resync_memory_graph_links_after_correction(rid) {
                tracing::warn!(
                    rid,
                    error = %e,
                    "graph-index resync after correction failed; stale entity links until reload"
                );
            }

            // Outcome anchor (Item 2): a successful correction is an
            // independent caller action targeting the rid.
            self.note_caller_used(rid);

            return Ok(CorrectionResult {
                original_rid: rid.to_string(),
                corrected_rid: rid.to_string(),
                original_tombstoned: false,
                revision_num: next_revision_num,
            });
        }
    }

    /// **v0.10 Item 3 finding 5 — apply a replicated `correct` op
    /// coherently on a follower.** Mirrors the leader's serialization: the
    /// conn lock is held across the reserve-append + SQL commit + publish,
    /// the memories row is stamped with the follower's active embedding
    /// model + generation, the revision records prior provenance, and
    /// append failures are PROPAGATED (so the op is retried rather than
    /// leaving SQL applied against a stale index).
    ///
    /// The exact leader bytes are applied to the vector index ONLY when the
    /// correction re-embedded AND the leader's embedding model matches this
    /// follower's active model (same vector space) AND the bytes decrypt
    /// (same DEK). Otherwise the SQL row is updated and the vector is left
    /// for `rebuild_vec_index` — the same fallback record replication uses.
    pub(crate) fn apply_replicated_correct(
        &self,
        payload: &serde_json::Value,
        embedding: Option<&[u8]>,
        source_actor: &str,
    ) -> Result<()> {
        use crate::serde_helpers::{deserialize_f32, serialize_f32};

        let rid = payload["rid"].as_str().unwrap_or_default();
        if rid.is_empty() {
            return Ok(());
        }
        let reembedded = payload["reembedded"].as_bool().unwrap_or(false);
        let revision_num = payload["revision_num"].as_i64().unwrap_or(1);
        let reason = payload["reason"].as_str().unwrap_or("");
        let applied_at = payload["applied_at"]
            .as_f64()
            .unwrap_or_else(crate::time::now_secs);
        let op_model = payload["embedding_model"].as_str();
        let new_text = payload["new_text"].as_str();

        // Revalidation loop, mirroring the leader: any local embed happens
        // OUTSIDE the guard, then the guarded critical section revalidates
        // the generation (retry on a reembed swap that landed mid-embed).
        loop {
            let state0 = self.search_state.load_full();
            let gen0 = state0.generation;
            let digest0 = state0.runtime_embedder_digest.clone();
            let runtime_model = state0.runtime_embedder_name.clone();
            let embedder = state0.embedder.clone();
            let dim0 = state0.dim();
            drop(state0);

            // Decide the vector to apply (finding 5 / r3 #2). Exact bytes
            // ONLY when the leader model is present AND matches this
            // follower's active model AND the bytes decrypt. Otherwise, for
            // a re-embedding correction, RE-EMBED the new text locally under
            // the follower's active embedder (the correct follower-space
            // vector) — a plain rebuild would keep the stale vector, so
            // "defer to rebuild" is NOT a valid fallback.
            let model_matches =
                reembedded && op_model.is_some() && op_model == runtime_model.as_deref();
            let exact: Option<Vec<f32>> = if model_matches {
                embedding
                    .and_then(|enc| self.decrypt_embedding(enc).ok())
                    .map(|p| deserialize_f32(&p))
            } else {
                None
            };
            let new_vec: Option<Vec<f32>> = if !reembedded {
                None // metadata/scalar correction — vector unchanged
            } else if let Some(v) = exact {
                Some(v)
            } else if let (Some(emb), Some(t)) = (embedder.as_ref(), new_text) {
                let v = emb
                    .embed(t)
                    .map_err(|e| YantrikDbError::Inference(e.to_string()))?;
                crate::validate::validate_embedding("apply_replicated_correct", &v, dim0)?;
                Some(v)
            } else {
                // Re-embedding correction, but the exact bytes are unusable
                // AND this follower has no embedder to re-encode. Refuse so
                // the op is RETRIED (never leave text/vector incoherent).
                return Err(YantrikDbError::NoEmbedder);
            };

            // Chunked embeddings: chunks are derived state and never ride
            // the op — the follower re-derives them from new_text under its
            // OWN embedder and its OWN probed window. This is coherent in
            // both vector branches (the exact-bytes branch requires the
            // models to match, so a locally embedded window is in the same
            // space as the leader's head vector). No embedder / no window /
            // no text ⇒ empty, and the old windows are purged below.
            let new_chunks: Vec<(usize, Vec<f32>)> = match (reembedded, new_text, embedder.as_ref())
            {
                (true, Some(t), Some(emb)) => match self.chunk_plan(t) {
                    Some(ranges) => {
                        let mut cv = Vec::with_capacity(ranges.len());
                        for (i, (a, b)) in ranges.iter().enumerate() {
                            let v = emb
                                .embed(&t[*a..*b])
                                .map_err(|e| YantrikDbError::Inference(e.to_string()))?;
                            crate::validate::validate_embedding(
                                "apply_replicated_correct#chunk",
                                &v,
                                dim0,
                            )?;
                            cv.push((i + 1, v));
                        }
                        cv
                    }
                    None => Vec::new(),
                },
                _ => Vec::new(),
            };

            // Enter the write router BEFORE loading state / conn (r3 #1):
            // otherwise a reembed cutover can swap SearchState between our
            // state load and our commit, detaching our delta append.
            let _guard = match self.write_router.try_enter_sync_writer() {
                Some(g) => g,
                None => {
                    return Err(YantrikDbError::CorrectionDeferredDuringReembed {
                        rid: rid.to_string(),
                    });
                }
            };
            let state = self.search_state.load_full();
            if state.generation != gen0 || state.runtime_embedder_digest != digest0 {
                drop(_guard);
                continue; // swap landed mid-embed; retry (re-embed under new)
            }
            let generation = state.generation as i64;
            let conn = self.conn.lock();
            // Correction seqlock (r4 finding 3): a follower-applied correction
            // is odd-epoch across its commit + publish + cache, so a
            // concurrent local recall retries rather than pairing new text
            // with an old ranking vector.
            let _epoch = self.enter_correction_epoch(&conn);

            // Prior state = the follower's current ACTIVE row.
            #[allow(clippy::type_complexity)]
            let existing: Option<(
                String,
                String,
                f64,
                f64,
                Option<Vec<u8>>,
                Option<String>,
            )> = conn
                .query_row(
                    "SELECT text, metadata, importance, valence, embedding, embedding_model \
                     FROM memories WHERE rid = ?1 AND consolidation_status = 'active'",
                    params![rid],
                    |r| {
                        Ok((
                            r.get(0)?,
                            r.get(1)?,
                            r.get(2)?,
                            r.get(3)?,
                            r.get::<_, Option<Vec<u8>>>(4)?,
                            r.get::<_, Option<String>>(5)?,
                        ))
                    },
                )
                .optional()?;
            let Some((ex_text, ex_meta, ex_imp, ex_val, ex_emb, ex_model)) = existing else {
                return Ok(()); // absent/forgotten — replay order handles it
            };

            let new_importance = payload["new_importance"].as_f64().unwrap_or(ex_imp);
            let new_valence = payload["new_valence"].as_f64().unwrap_or(ex_val);
            let stored_new_text = match new_text {
                Some(t) => self.encrypt_text(t)?,
                None => ex_text.clone(),
            };
            let ex_meta_plain = self.decrypt_text(&ex_meta)?;
            let ex_meta_val: serde_json::Value =
                serde_json::from_str(&ex_meta_plain).unwrap_or(serde_json::json!({}));
            let new_meta_val: serde_json::Value = match payload.get("metadata_merge") {
                Some(patch) if !patch.is_null() => {
                    let mut merged = ex_meta_val.clone();
                    if let (Some(obj), Some(patch_obj)) =
                        (merged.as_object_mut(), patch.as_object())
                    {
                        for (k, v) in patch_obj {
                            obj.insert(k.clone(), v.clone());
                        }
                        merged
                    } else {
                        patch.clone()
                    }
                }
                _ => ex_meta_val,
            };
            let stored_new_meta = self.encrypt_text(&serde_json::to_string(&new_meta_val)?)?;
            // v48 (#149): re-stamp the event-time columns from the SAME
            // plaintext value just serialized (pre-encryption).
            let (event_time_min, event_time_max) =
                crate::base::datetext::event_time_bounds(&new_meta_val);

            // Store the follower-local encrypted bytes (exact or re-embedded).
            let stored_emb: Option<Vec<u8>> = match &new_vec {
                Some(v) => Some(self.encrypt_embedding(&serialize_f32(v))?),
                None => None,
            };
            let prior_hash: Option<Vec<u8>> = ex_emb.as_ref().and_then(|b| {
                self.decrypt_embedding(b)
                    .ok()
                    .map(|p| embedding_hash(&deserialize_f32(&p)).to_vec())
            });

            // Reserve-append BEFORE commit; propagate failure (op retried).
            // `false` = the freshly minted seq collided with an existing
            // (rid, seq) entry — an invariant violation; constructing the
            // guard over an entry this call does not own would let a failure
            // path remove another write's published vector (4a.6d-3).
            let seq_new = if let Some(v) = &new_vec {
                let s = self
                    .vec_seq
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    + 1;
                if state
                    .vec_index
                    .append_reserved(rid.to_string(), v.clone(), s)?
                    == crate::vector::delta_index::ReservedAppend::AlreadyPresent
                {
                    return Err(crate::error::YantrikDbError::InvalidInput(format!(
                        "freshly minted seq {s} for rid {rid} already present \
                         in the delta — engine invariant violation"
                    )));
                }
                Some(s)
            } else {
                None
            };

            // Same obligation as the leader path, via the same guard: owed back on
            // every exit including an unwind, which the hand-rolled cleanup below
            // could not cover. `None` when this correction carries no vector —
            // there is nothing reserved, so nothing is owed.
            //
            // publish_only: this tx commits an already-applied replicated
            // correction and enqueues no pending op.
            let mut reservation = seq_new.map(|s| ReservationGuard::publish_only(&state, rid, s));

            // Window keys ride the same reservation at the same seq (they
            // exist only when the correction carries a vector, so seq_new
            // is Some whenever new_chunks is non-empty).
            let stored_new_chunks: Vec<(usize, String, Vec<u8>)> = new_chunks
                .iter()
                .map(|(idx, v)| {
                    let blob = self.encrypt_embedding(&serialize_f32(v))?;
                    Ok((*idx, crate::vector::chunk::chunk_key(rid, *idx), blob))
                })
                .collect::<Result<_>>()?;
            if let (Some(s), Some(r)) = (seq_new, reservation.as_mut()) {
                for ((_, v), (_, key, _)) in new_chunks.iter().zip(stored_new_chunks.iter()) {
                    match state.vec_index.append_reserved(key.clone(), v.clone(), s) {
                        Ok(crate::vector::delta_index::ReservedAppend::Inserted) => {
                            r.add_chunk_key(key.clone());
                        }
                        Ok(crate::vector::delta_index::ReservedAppend::AlreadyPresent) => {
                            return Err(crate::error::YantrikDbError::InvalidInput(format!(
                                "freshly minted seq {s} for chunk key {key} already \
                                 present in the delta — engine invariant violation"
                            )));
                        }
                        Err(e) => return Err(e),
                    }
                }
            }

            let mut old_chunk_idxs: Vec<i64> = Vec::new();
            let mut invalidated_syntheses: Vec<String> = Vec::new();
            let commit: Result<()> = (|| {
                let tx = conn.unchecked_transaction()?;
                tx.execute(
                    "INSERT OR IGNORE INTO record_revisions \
                     (revision_id, rid, revision_num, prior_text, prior_metadata, \
                      prior_importance, prior_valence, reason, applied_at, hlc, \
                      origin_actor, prior_embedding_model, prior_embedding_hash) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                    params![
                        crate::id::new_id(),
                        rid,
                        revision_num,
                        ex_text,
                        ex_meta,
                        ex_imp,
                        ex_val,
                        reason,
                        applied_at,
                        Vec::<u8>::new(),
                        source_actor,
                        ex_model,
                        prior_hash,
                    ],
                )?;
                if let Some(enc) = &stored_emb {
                    tx.execute(
                        "UPDATE memories \
                         SET text = ?1, metadata = ?2, importance = ?3, valence = ?4, \
                             embedding = ?5, embedding_model = ?6, embedding_generation = ?7, \
                             embedding_new = NULL, embedding_new_model = NULL, last_access = ?8, \
                             event_time_min = ?10, event_time_max = ?11 \
                         WHERE rid = ?9",
                        params![
                            stored_new_text,
                            stored_new_meta,
                            new_importance,
                            new_valence,
                            enc,
                            runtime_model,
                            generation,
                            applied_at,
                            rid,
                            // v48 (#149) event time, from the merged
                            // plaintext metadata.
                            event_time_min,
                            event_time_max,
                        ],
                    )?;
                } else {
                    tx.execute(
                        "UPDATE memories \
                         SET text = ?1, metadata = ?2, importance = ?3, valence = ?4, \
                             last_access = ?5, event_time_min = ?7, event_time_max = ?8 \
                         WHERE rid = ?6",
                        params![
                            stored_new_text,
                            stored_new_meta,
                            new_importance,
                            new_valence,
                            applied_at,
                            rid,
                            // v48 (#149) event time, from the merged
                            // plaintext metadata.
                            event_time_min,
                            event_time_max,
                        ],
                    )?;
                }
                // Chunked embeddings — mirror the leader: a text-changing
                // correction replaces the window rows atomically with the
                // text, and the old idx set escapes for surplus-key
                // tombstoning post-commit.
                if reembedded {
                    {
                        let mut idx_stmt =
                            tx.prepare("SELECT chunk_idx FROM memory_chunks WHERE rid = ?1")?;
                        old_chunk_idxs = idx_stmt
                            .query_map(params![rid], |r| r.get(0))?
                            .collect::<std::result::Result<_, _>>()?;
                    }
                    tx.execute("DELETE FROM memory_chunks WHERE rid = ?1", params![rid])?;
                    for (idx, _, blob) in &stored_new_chunks {
                        tx.execute(
                            "INSERT INTO memory_chunks (rid, chunk_idx, embedding) \
                             VALUES (?1, ?2, ?3)",
                            params![rid, *idx as i64, blob],
                        )?;
                    }
                }
                // Entity-graph coherence safety half (nuron) — mirror the
                // leader: on a text-changing correction, drop memory→entity
                // links whose entity no longer appears in the corrected text,
                // so a follower's graph expansion stays coherent with the
                // corrected text just like the leader's.
                if let Some(t) = new_text {
                    Self::drop_stale_memory_entity_links_in_tx(&tx, rid, t)?;
                }
                // Provenance stamp (minor r3): every replication-apply site
                // records into replication_apply_log (schema contract).
                tx.execute(
                    "INSERT OR IGNORE INTO replication_apply_log \
                     (rid, op_type, source_actor, applied_at) VALUES (?1, 'correct', ?2, ?3)",
                    params![rid, source_actor, applied_at],
                )?;
                invalidated_syntheses = Self::invalidate_synthesis_dependents_in_tx(&tx, rid)?;
                tx.commit()?;
                Ok(())
            })();
            if let Err(e) = commit {
                // Still Reserved: dropping removes the reservation.
                drop(reservation);
                return Err(e);
            }
            if let Some(r) = reservation.as_mut() {
                // Durable — the obligation inverts to publish.
                r.mark_committed();
                r.complete();
            }
            // Surplus window keys (old idxs the corrected text no longer
            // produces) — same rule as the leader, at the same seq as the
            // correction's publish.
            if let Some(s) = seq_new {
                let kept: std::collections::HashSet<usize> =
                    stored_new_chunks.iter().map(|(i, _, _)| *i).collect();
                for old in &old_chunk_idxs {
                    if !kept.contains(&(*old as usize)) {
                        let key = crate::vector::chunk::chunk_key(rid, *old as usize);
                        state.vec_index.tombstone(&key, s);
                    }
                }
            }
            {
                let mut cache = self.scoring_cache.write();
                if let Some(row) = cache.get_mut(rid) {
                    row.importance = new_importance;
                    row.valence = new_valence;
                }
            }
            self.cache_invalidate_syntheses(&invalidated_syntheses);
            // Release the epoch (restores even) then conn — both held across
            // commit + publish + cache like the leader — BEFORE the graph
            // resync re-locks conn. Then propagate the durable stale-link drop
            // to the in-memory graph_index on the follower too (nuron), so a
            // follower's graph expansion stays coherent with the corrected text.
            drop(_epoch);
            drop(conn);
            if let Err(e) = self.resync_memory_graph_links_after_correction(rid) {
                tracing::warn!(
                    rid,
                    error = %e,
                    "graph-index resync after replicated correction failed; stale links until reload"
                );
            }
            return Ok(());
        }
    }

    /// Query the revision history for a single record (Issue #47).
    ///
    /// Returns revisions ordered by `revision_num` ascending (oldest
    /// first). Empty vec if the record has never been corrected.
    /// Prior text + metadata are decrypted before return (mirrors
    /// `db.get()`'s contract on the `memories` table).
    pub fn history(&self, rid: &str) -> Result<Vec<RecordRevision>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT revision_id, rid, revision_num, prior_text, prior_metadata, \
                    prior_importance, prior_valence, reason, applied_at, origin_actor, \
                    prior_embedding_model, prior_embedding_hash \
             FROM record_revisions \
             WHERE rid = ?1 \
             ORDER BY revision_num ASC",
        )?;
        let rows = stmt.query_map(params![rid], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, f64>(5)?,
                row.get::<_, f64>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, f64>(8)?,
                row.get::<_, String>(9)?,
                row.get::<_, Option<String>>(10)?,
                row.get::<_, Option<Vec<u8>>>(11)?,
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            let (
                revision_id,
                rid,
                revision_num,
                stored_text,
                stored_metadata,
                prior_importance,
                prior_valence,
                reason,
                applied_at,
                origin_actor,
                prior_embedding_model,
                prior_embedding_hash_bytes,
            ) = r?;
            let prior_text = self.decrypt_text(&stored_text)?;
            let prior_metadata_str = self.decrypt_text(&stored_metadata)?;
            let prior_metadata: serde_json::Value = serde_json::from_str(&prior_metadata_str)
                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
            let prior_embedding_hash = prior_embedding_hash_bytes.map(|b| {
                b.iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
            });
            out.push(RecordRevision {
                revision_id,
                rid,
                revision_num,
                prior_text,
                prior_metadata,
                prior_importance,
                prior_valence,
                reason,
                applied_at,
                origin_actor,
                prior_embedding_model,
                prior_embedding_hash,
            });
        }
        Ok(out)
    }
}
