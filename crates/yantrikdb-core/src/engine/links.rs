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

use rusqlite::{params, OptionalExtension};

use crate::error::{Result, YantrikDbError};
use crate::serde_helpers::hex_lower;

/// Bounded-walk cap for supersedes-chain traversals (v0.10 Phase 0).
pub(crate) const CHAIN_WALK_CAP: usize = 1_000;

/// Outcome of a bounded supersedes-graph walk (v0.10 Phase 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WalkOutcome {
    Reached,
    NotReached,
    CapHit,
}

/// Result of the deterministic supersedes projection fold (v0.10 Phase 0):
/// `(edge_id, source_rid)` of the selected winner (None when every
/// candidate would close a cycle) plus the retained rejected candidates.
pub(crate) struct SupersedesFold {
    pub winner: Option<(String, String)>,
    pub losers: Vec<(String, String)>,
}

/// **v0.10 Phase 0 — supersedes-chain audit report** (report-only; the
/// engine never auto-repairs — no-auto-quarantine principle). Produced by
/// [`YantrikDB::verify_chains`]. Item 1's `status_read_policy` enablement
/// refuses on a non-empty report until explicit repair.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ChainAuditReport {
    /// Predecessors with MORE than one selected active inbound successor
    /// (legacy pre-Phase-0 data): `(target_rid, selected edge_ids)`.
    pub multi_successor: Vec<(String, Vec<String>)>,
    /// Rids participating in a supersedes cycle in the selected graph.
    pub cycle_members: Vec<String>,
    /// Selected active edges whose endpoints are in different namespaces:
    /// edge_ids.
    pub cross_namespace: Vec<String>,
    /// Selected active edges with a missing endpoint row: edge_ids.
    /// (Endpoints tombstoned AFTER linking are lifecycle history, handled
    /// by `status`, and are NOT reported here.)
    pub dangling: Vec<String>,
    /// Components whose traversal exceeded the walk cap (audit could not
    /// complete for them).
    pub cap_exceeded: Vec<String>,
}

impl ChainAuditReport {
    /// True when the selected supersedes graph satisfies every Phase-0
    /// invariant.
    pub fn is_clean(&self) -> bool {
        self.multi_successor.is_empty()
            && self.cycle_members.is_empty()
            && self.cross_namespace.is_empty()
            && self.dangling.is_empty()
            && self.cap_exceeded.is_empty()
    }
}
use crate::types::{
    LinkDirection, LinkResult, LinkType, LinkedRecord, RecallResult, RecordLink, ScoreBreakdown,
    ScoreContributions,
};

use super::{now, YantrikDB};

/// Shared candidate-budget cap for link expansion at recall time (RFC §3):
/// `expand_links` and `expand_entities` together must not add more than
/// this many candidates, bounding worst-case fan-out.
const LINK_EXPANSION_BUDGET: usize = 50;

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

    /// Like [`Self::record_with_links`] but returns a per-link outcome
    /// instead of failing fast (issue #48, v0.7.22). The record commits
    /// first (durable via the oplog); then each link is attempted
    /// independently — a failing link is captured as
    /// [`LinkResult::Failed`] and does NOT abort the remaining links or
    /// fail the call. Returns `(rid, per_link_results)`.
    ///
    /// This is the surface the MCP layer wants: it avoids re-querying to
    /// reconstruct which links landed after a fail-fast `?` short-circuit.
    /// `AlreadyExists` (the idempotent UNIQUE hit) is distinguished from
    /// `Inserted` for telemetry; algo's retry path treats them the same.
    #[allow(clippy::too_many_arguments)]
    pub fn record_with_links_partial(
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
    ) -> Result<(String, Vec<LinkResult>)> {
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

        let mut results = Vec::with_capacity(links.len());
        for link in links {
            let target_rid = link.target_rid.clone();
            let link_type = link.link_type.as_str();
            match self.link_core(&rid, link) {
                Ok((_id, true)) => results.push(LinkResult::Inserted {
                    target_rid,
                    link_type,
                }),
                Ok((_id, false)) => results.push(LinkResult::AlreadyExists {
                    target_rid,
                    link_type,
                }),
                Err(e) => results.push(LinkResult::Failed {
                    target_rid,
                    link_type,
                    error: e.to_string(),
                }),
            }
        }
        Ok((rid, results))
    }

    /// Add a single record-to-record link from `source_rid`.
    ///
    /// Validates: `source_rid` non-empty, `target_rid` non-empty, and
    /// `source_rid != target_rid` (a record cannot link to itself).
    /// Idempotent on `UNIQUE(source_rid, target_rid, link_type)` via
    /// `INSERT OR IGNORE`. Returns the `link_id` (freshly minted even if
    /// the row already existed and the insert was ignored).
    pub fn link(&self, source_rid: &str, link: &RecordLink) -> Result<String> {
        let (link_id, _inserted) = self.link_core(source_rid, link)?;
        Ok(link_id)
    }

    /// Core link insert shared by [`Self::link`] and
    /// [`Self::record_with_links_partial`]. Returns `(link_id, inserted)`.
    ///
    /// **v0.10 Phase 0 reshape (sol-converged, rid 019f5e7e):**
    /// - **Idempotency with ORIGINAL identity**: an exact
    ///   (source, target, type) duplicate returns the EXISTING edge id and
    ///   emits NO new oplog op — a retry must not mint a newer identity or
    ///   a newer replication order (T7 discipline applied to links).
    /// - **Canonical identity**: ONE id and ONE HLC are minted for the edge,
    ///   shared verbatim by the record_links row AND the oplog op
    ///   (`op_id == link_id`, `op.hlc == row.hlc`), and carried in the
    ///   payload so followers persist the exact same identity. This is what
    ///   makes `max(hlc, id)` a replayable total order for merge arbitration.
    /// - **Atomicity**: row + oplog op commit in one SAVEPOINT — a crash
    ///   can no longer leave a local edge with no replication event.
    /// - **Supersedes integrity gate**: endpoints must exist, be
    ///   non-tombstoned, and share a namespace; the target (predecessor) must
    ///   not already have a selected active successor (single-INBOUND-edge
    ///   invariant — the edge direction is new→old); the insertion must not
    ///   create a cycle in the predecessor closure. All checks and the
    ///   insert happen under ONE connection lock, so two concurrent callers
    ///   cannot both pass the gate.
    fn link_core(&self, source_rid: &str, link: &RecordLink) -> Result<(String, bool)> {
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

        let link_type_str = link.link_type.as_str();
        let is_supersedes = matches!(link.link_type, crate::types::LinkType::Supersedes);
        let ts = now();
        // Mint the canonical identity BEFORE the transaction (one id, one HLC).
        let edge_id = crate::id::new_id();
        let hlc_bytes = self.tick_hlc().to_bytes().to_vec();
        let applied_generation: i64 = self.search_state.load().generation as i64;

        let conn = self.conn.lock();

        // Idempotent duplicate: return the ORIGINAL identity, no new op.
        if let Some(existing_id) = conn
            .query_row(
                "SELECT link_id FROM record_links \
                 WHERE source_rid = ?1 AND target_rid = ?2 AND link_type = ?3",
                params![source_rid, link.target_rid, link_type_str],
                |r| r.get::<_, String>(0),
            )
            .optional()?
        {
            return Ok((existing_id, false));
        }

        if is_supersedes {
            Self::gate_supersedes(&conn, source_rid, &link.target_rid)?;
        }

        let payload = serde_json::json!({
            "source_rid": source_rid,
            "target_rid": link.target_rid,
            "link_type": link_type_str,
            "created_at": ts,
            // Canonical identity for follower persistence (v0.10 Phase 0).
            "edge_id": edge_id,
            "edge_hlc_hex": hex_lower(&hlc_bytes),
            "selection_state": "selected",
        });
        let payload_str = serde_json::to_string(&payload)?;

        conn.execute_batch("SAVEPOINT link_core_txn")?;
        let txn: Result<()> = (|| {
            conn.execute(
                "INSERT INTO record_links \
                 (link_id, source_rid, target_rid, link_type, status, selection_state, \
                  created_at, hlc, origin_actor) \
                 VALUES (?1, ?2, ?3, ?4, 'active', 'selected', ?5, ?6, ?7)",
                params![
                    edge_id,
                    source_rid,
                    link.target_rid,
                    link_type_str,
                    ts,
                    hlc_bytes,
                    self.actor_id,
                ],
            )?;
            // (Phase 0 failpoint "link.between_row_and_oplog" lands here with
            // the `testing`-gated registry — the kill proof asserts NEITHER
            // row survives a kill inside this savepoint.)
            crate::testing::fail_point("link.between_row_and_oplog");
            conn.execute(
                "INSERT INTO oplog (op_id, op_type, timestamp, target_rid, payload, \
                 actor_id, hlc, embedding_hash, origin_actor, applied, applied_generation) \
                 VALUES (?1, 'link', ?2, ?3, ?4, ?5, ?6, NULL, ?7, 1, ?8)",
                params![
                    edge_id,
                    ts,
                    source_rid,
                    payload_str,
                    self.actor_id,
                    hlc_bytes,
                    self.actor_id,
                    applied_generation,
                ],
            )?;
            Ok(())
        })();
        match txn {
            Ok(()) => {
                conn.execute_batch("RELEASE link_core_txn")?;
                Ok((edge_id, true))
            }
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK TO link_core_txn; RELEASE link_core_txn");
                Err(e)
            }
        }
    }

    /// **v0.10 Phase 0 — the Supersedes integrity gate.** Caller holds the
    /// connection lock; all checks run against that same connection so the
    /// gate + insert are atomic with respect to concurrent writers.
    ///
    /// Edge direction is NEW→OLD (`source` supersedes `target`): the
    /// invariant is one selected active INBOUND edge per target
    /// (predecessor), and the cycle check walks the TARGET's outgoing
    /// predecessor closure looking for the source.
    fn gate_supersedes(
        conn: &rusqlite::Connection,
        source_rid: &str,
        target_rid: &str,
    ) -> Result<()> {
        // Endpoints: exist, non-tombstoned, same namespace.
        let fetch = |rid: &str| -> Result<Option<(String, String)>> {
            Ok(conn
                .query_row(
                    "SELECT namespace, consolidation_status FROM memories WHERE rid = ?1",
                    params![rid],
                    |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
                )
                .optional()?)
        };
        let Some((src_ns, src_status)) = fetch(source_rid)? else {
            return Err(YantrikDbError::InvalidLinkEndpoints {
                reason: format!("supersedes source {source_rid} not found"),
            });
        };
        let Some((tgt_ns, tgt_status)) = fetch(target_rid)? else {
            return Err(YantrikDbError::InvalidLinkEndpoints {
                reason: format!("supersedes target {target_rid} not found"),
            });
        };
        if src_status == "tombstoned" || tgt_status == "tombstoned" {
            return Err(YantrikDbError::InvalidLinkEndpoints {
                reason: format!(
                    "supersedes endpoints must be live (source {src_status}, target {tgt_status})"
                ),
            });
        }
        if src_ns != tgt_ns {
            return Err(YantrikDbError::InvalidLinkEndpoints {
                reason: format!(
                    "supersedes endpoints must share a namespace ({src_ns} vs {tgt_ns})"
                ),
            });
        }

        // Single-successor: at most one selected active inbound edge per
        // predecessor.
        if let Some((edge_id, successor)) = conn
            .query_row(
                "SELECT link_id, source_rid FROM record_links \
                 WHERE target_rid = ?1 AND link_type = 'supersedes' \
                 AND selection_state = 'selected' AND status = 'active' \
                 LIMIT 1",
                params![target_rid],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
            )
            .optional()?
        {
            return Err(YantrikDbError::SupersedeConflict {
                predecessor_rid: target_rid.to_string(),
                existing_successor_rid: successor,
                existing_edge_id: edge_id,
            });
        }

        // Cycle check: walk the target's outgoing predecessor closure
        // (bounded graph walk — multi-predecessor merges are legal, so this
        // is a queue + visited set, not a linked-list walk). Reaching the
        // source means the new edge closes a loop.
        match Self::supersedes_walk_reaches(conn, target_rid, source_rid)? {
            WalkOutcome::Reached => Err(YantrikDbError::SupersedeCycle {
                source_rid: source_rid.to_string(),
                target_rid: target_rid.to_string(),
            }),
            WalkOutcome::CapHit => Err(YantrikDbError::ChainTraversalLimit {
                start_rid: target_rid.to_string(),
                limit: CHAIN_WALK_CAP,
            }),
            WalkOutcome::NotReached => Ok(()),
        }
    }

    /// Bounded walk over the SELECTED supersedes graph: starting from
    /// `from_rid`'s outgoing predecessor closure, does it reach `needle`?
    /// Shared by the local write gate (which converts outcomes to typed
    /// errors) and the replication fold (which treats Reached/CapHit as
    /// "candidate not selectable" — a durable remote candidate is never
    /// discarded at the cap, per the Phase-0 converged design).
    pub(crate) fn supersedes_walk_reaches(
        conn: &rusqlite::Connection,
        from_rid: &str,
        needle: &str,
    ) -> Result<WalkOutcome> {
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut queue: std::collections::VecDeque<String> = std::collections::VecDeque::new();
        queue.push_back(from_rid.to_string());
        while let Some(current) = queue.pop_front() {
            if visited.len() > CHAIN_WALK_CAP {
                return Ok(WalkOutcome::CapHit);
            }
            if !visited.insert(current.clone()) {
                continue;
            }
            let mut stmt = conn.prepare(
                "SELECT target_rid FROM record_links \
                 WHERE source_rid = ?1 AND link_type = 'supersedes' \
                 AND selection_state = 'selected' AND status = 'active'",
            )?;
            let preds = stmt
                .query_map(params![current], |r| r.get::<_, String>(0))?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            for p in preds {
                if p == needle {
                    return Ok(WalkOutcome::Reached);
                }
                queue.push_back(p);
            }
        }
        Ok(WalkOutcome::NotReached)
    }

    /// **v0.10 Phase 0 — audit the selected supersedes graph** against the
    /// chain-integrity invariants. REPORT-ONLY: legacy databases (edges
    /// written before the write gate existed) may violate them; the engine
    /// never repairs automatically. Explicit repair / canonicalization is a
    /// maintenance action; Item 1's `status_read_policy` opt-in refuses on
    /// a dirty report.
    pub fn verify_chains(&self) -> Result<ChainAuditReport> {
        let conn = self.conn.lock();
        let mut report = ChainAuditReport::default();

        // Multi-successor predecessors (selected active inbound > 1).
        {
            let mut stmt = conn.prepare(
                "SELECT target_rid, GROUP_CONCAT(link_id) FROM record_links \
                 WHERE link_type = 'supersedes' AND status = 'active' \
                 AND selection_state = 'selected' \
                 GROUP BY target_rid HAVING COUNT(*) > 1",
            )?;
            let rows = stmt
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            for (target, ids) in rows {
                report
                    .multi_successor
                    .push((target, ids.split(',').map(str::to_string).collect()));
            }
        }

        // Cross-namespace + dangling endpoints, one pass via LEFT JOINs.
        {
            let mut stmt = conn.prepare(
                "SELECT l.link_id, ms.namespace, mt.namespace \
                 FROM record_links l \
                 LEFT JOIN memories ms ON ms.rid = l.source_rid \
                 LEFT JOIN memories mt ON mt.rid = l.target_rid \
                 WHERE l.link_type = 'supersedes' AND l.status = 'active' \
                 AND l.selection_state = 'selected'",
            )?;
            let rows = stmt
                .query_map([], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, Option<String>>(1)?,
                        r.get::<_, Option<String>>(2)?,
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            for (edge_id, src_ns, tgt_ns) in rows {
                match (src_ns, tgt_ns) {
                    (Some(a), Some(b)) if a != b => report.cross_namespace.push(edge_id),
                    (None, _) | (_, None) => report.dangling.push(edge_id),
                    _ => {}
                }
            }
        }

        // Cycles in the selected graph: iterative DFS with an in-stack set,
        // bounded per component by the walk cap.
        {
            let edges: Vec<(String, String)> = {
                let mut stmt = conn.prepare(
                    "SELECT source_rid, target_rid FROM record_links \
                     WHERE link_type = 'supersedes' AND status = 'active' \
                     AND selection_state = 'selected'",
                )?;
                let rows = stmt
                    .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                rows
            };
            let mut adj: std::collections::HashMap<&str, Vec<&str>> =
                std::collections::HashMap::new();
            for (s, t) in &edges {
                adj.entry(s.as_str()).or_default().push(t.as_str());
            }
            let mut color: std::collections::HashMap<&str, u8> = std::collections::HashMap::new(); // 0 unvisited, 1 in-stack, 2 done
            let mut cycle_members: std::collections::BTreeSet<String> =
                std::collections::BTreeSet::new();
            for start in adj.keys().copied().collect::<Vec<_>>() {
                if color.get(start).copied().unwrap_or(0) != 0 {
                    continue;
                }
                // Iterative DFS: stack of (node, next-child-index).
                let mut stack: Vec<(&str, usize)> = vec![(start, 0)];
                color.insert(start, 1);
                let mut steps = 0usize;
                while let Some(&mut (node, ref mut idx)) = stack.last_mut() {
                    steps += 1;
                    if steps > CHAIN_WALK_CAP {
                        report.cap_exceeded.push(start.to_string());
                        for (n, _) in &stack {
                            color.insert(n, 2);
                        }
                        break;
                    }
                    let children = adj.get(node).map(|v| v.as_slice()).unwrap_or(&[]);
                    if *idx < children.len() {
                        let child = children[*idx];
                        *idx += 1;
                        match color.get(child).copied().unwrap_or(0) {
                            0 => {
                                color.insert(child, 1);
                                stack.push((child, 0));
                            }
                            1 => {
                                // Back edge: everything from `child` up the
                                // stack is on the cycle.
                                let pos = stack.iter().position(|(n, _)| *n == child);
                                if let Some(p) = pos {
                                    for (n, _) in &stack[p..] {
                                        cycle_members.insert((*n).to_string());
                                    }
                                }
                            }
                            _ => {}
                        }
                    } else {
                        color.insert(node, 2);
                        stack.pop();
                    }
                }
            }
            report.cycle_members = cycle_members.into_iter().collect();
        }

        Ok(report)
    }

    /// **v0.10 Phase 0 — deterministic supersedes projection fold** for one
    /// predecessor. Used on the REPLICATION apply path (the local write
    /// gate refuses conflicting edges up front; replication must instead
    /// durably accept every remote candidate and then derive the selected
    /// projection from the candidate set, so the result is independent of
    /// arrival order).
    ///
    /// Canonical rule (sol-converged): consider the target's active,
    /// non-retracted candidates in DESCENDING total-key order
    /// (`hlc DESC, link_id DESC` — HLC bytes are memcmp-sortable and the
    /// leader's exact values are persisted verbatim on followers, so every
    /// replica computes the same order). The first candidate whose
    /// selection keeps the selected graph acyclic wins; all others are
    /// retained as `rejected_conflict`. Equivalent, for a cycle, to
    /// dropping the lowest-key edge under the fold — never "whichever
    /// arrived last".
    pub(crate) fn refold_supersedes_target(
        conn: &rusqlite::Connection,
        target_rid: &str,
    ) -> Result<SupersedesFold> {
        // Demote all of this target's candidates first so the cycle checks
        // below run against the rest of the selected graph only (no
        // self-interference from a previously-selected edge we may unseat).
        conn.execute(
            "UPDATE record_links SET selection_state = 'rejected_conflict' \
             WHERE target_rid = ?1 AND link_type = 'supersedes' \
             AND status = 'active' AND selection_state IN ('selected', 'rejected_conflict')",
            params![target_rid],
        )?;

        let candidates: Vec<(String, String)> = {
            let mut stmt = conn.prepare(
                "SELECT link_id, source_rid FROM record_links \
                 WHERE target_rid = ?1 AND link_type = 'supersedes' \
                 AND status = 'active' AND selection_state = 'rejected_conflict' \
                 ORDER BY hlc DESC, link_id DESC",
            )?;
            let rows = stmt
                .query_map(params![target_rid], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            rows
        };

        let mut winner: Option<(String, String)> = None;
        let mut losers: Vec<(String, String)> = Vec::new();
        for (edge_id, source_rid) in candidates {
            if winner.is_none()
                && matches!(
                    Self::supersedes_walk_reaches(conn, target_rid, &source_rid)?,
                    WalkOutcome::NotReached
                )
            {
                conn.execute(
                    "UPDATE record_links SET selection_state = 'selected' WHERE link_id = ?1",
                    params![edge_id],
                )?;
                winner = Some((edge_id, source_rid));
            } else {
                losers.push((edge_id, source_rid));
            }
        }
        Ok(SupersedesFold { winner, losers })
    }

    /// Remove a single link. Returns `true` if a row was affected.
    ///
    /// Unlike `forget()` (which marks links broken for audit), explicit
    /// `unlink()` is a user retraction.
    ///
    /// **v0.10 Phase 0:** for SUPERSEDES edges, retraction is replayable —
    /// the row flips to `selection_state='retracted'` (never hard-deleted;
    /// a hard delete would be arrival-order-dependent under concurrent
    /// link/unlink replication) and the target's projection is re-folded
    /// so the next-best durable candidate is promoted deterministically.
    /// Other link types keep their historical hard-delete semantics.
    pub fn unlink(&self, source_rid: &str, target_rid: &str, link_type: &LinkType) -> Result<bool> {
        let link_type_str = link_type.as_str();
        let is_supersedes = matches!(link_type, LinkType::Supersedes);
        let affected = {
            let conn = self.conn.lock();
            if is_supersedes {
                let n = conn.execute(
                    "UPDATE record_links SET selection_state = 'retracted' \
                     WHERE source_rid = ?1 AND target_rid = ?2 AND link_type = ?3 \
                     AND selection_state != 'retracted'",
                    params![source_rid, target_rid, link_type_str],
                )?;
                if n > 0 {
                    // Promote the next-best candidate (if any) for this
                    // predecessor — same canonical fold replication uses.
                    Self::refold_supersedes_target(&conn, target_rid)?;
                }
                n
            } else {
                conn.execute(
                    "DELETE FROM record_links \
                     WHERE source_rid = ?1 AND target_rid = ?2 AND link_type = ?3",
                    params![source_rid, target_rid, link_type_str],
                )?
            }
        };

        if affected > 0 {
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

        Ok(affected > 0)
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

    /// Issue #48 — recall with record-link expansion.
    ///
    /// Additive sibling of `recall()` (NOT a signature change to it — same
    /// call-site-cascade rationale as `record_with_links`). `expand_links`
    /// is the hop budget; `0` makes this identical to `recall()`.
    ///
    /// **Design — isolated post-pass, not a weave into the recall core.**
    /// Runs the standard `recall()` for a slightly larger base pool, then
    /// applies two link-aware transforms:
    /// 1. **Supersedes demotion** — any base result that is the TARGET of
    ///    an active `supersedes` link is multiplied by its
    ///    `demote_self_as_target` factor (0.5). This is the half of the
    ///    fix that stops a stale, superseded record from dominating.
    /// 2. **Neighbor surfacing** — for each base result, active outbound
    ///    links (and symmetric `contradicts`) surface the linked record
    ///    (if active + not already present), scored at
    ///    `seed.score * neighbor_factor / 4` (1-hop decay mirroring the
    ///    entity graph's `4^hops`). This is the half that pulls the
    ///    superseder/contradictor in even when it isn't semantically near
    ///    the query. Budget-capped at [`LINK_EXPANSION_BUDGET`].
    ///
    /// Then re-sort by score and truncate to `top_k`.
    ///
    /// **Tradeoff (documented):** surfaced neighbors do NOT pass through
    /// MMR diversity (the post-pass runs after `recall()`'s MMR). For v1
    /// this is acceptable — the link set is small and intentional, unlike
    /// the entity graph. A future revision can move expansion pre-MMR by
    /// weaving into the recall core if diversity over linked records
    /// proves to matter empirically.
    #[allow(clippy::too_many_arguments)]
    pub fn recall_with_links(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
        expand_entities: bool,
        query_text: Option<&str>,
        skip_reinforce: bool,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
        certainty_min: Option<f64>,
        order: Option<&str>,
        expand_links: usize,
    ) -> Result<Vec<RecallResult>> {
        // Larger base pool when expanding so demotion/expansion has room
        // to reorder before the final truncate.
        let base_k = if expand_links == 0 {
            top_k
        } else {
            top_k.saturating_add(LINK_EXPANSION_BUDGET)
        };

        let mut base = self.recall(
            query_embedding,
            base_k,
            time_window,
            memory_type,
            include_consolidated,
            expand_entities,
            query_text,
            skip_reinforce,
            namespace,
            domain,
            source,
            certainty_min,
            order,
        )?;

        if expand_links == 0 {
            base.truncate(top_k);
            return Ok(base);
        }

        let mut present: std::collections::HashSet<String> =
            base.iter().map(|r| r.rid.clone()).collect();

        // Phase 1: supersedes demotion.
        let demote = LinkType::Supersedes.recall_polarity().demote_self_as_target;
        for r in base.iter_mut() {
            let superseded_by =
                self.linked_records(&r.rid, LinkDirection::Inbound, Some(&LinkType::Supersedes))?;
            if !superseded_by.is_empty() {
                r.score *= demote;
                r.why_retrieved
                    .push("demoted: superseded by a newer record".to_string());
            }
        }

        // Phase 2: neighbor surfacing (budget-capped).
        let mut added: Vec<RecallResult> = Vec::new();
        let mut budget = LINK_EXPANSION_BUDGET;
        let seeds: Vec<(String, f64)> = base.iter().map(|r| (r.rid.clone(), r.score)).collect();
        'seeds: for (seed_rid, seed_score) in &seeds {
            if budget == 0 {
                break;
            }
            let links = self.linked_records(seed_rid, LinkDirection::Outbound, None)?;
            for l in links {
                if budget == 0 {
                    break 'seeds;
                }
                if present.contains(&l.rid) {
                    continue;
                }
                let lt = LinkType::from_str_lenient(&l.link_type);
                let pol = lt.recall_polarity();
                if pol.neighbor_factor <= 0.0 {
                    continue;
                }
                let Some(mem) = self.get(&l.rid)? else {
                    continue;
                };
                if mem.consolidation_status != "active" {
                    continue;
                }
                // 1-hop proximity decay mirrors the entity graph's 4^hops.
                let nscore = seed_score * pol.neighbor_factor / 4.0;
                present.insert(l.rid.clone());
                budget -= 1;
                added.push(RecallResult {
                    rid: mem.rid,
                    memory_type: mem.memory_type,
                    text: mem.text,
                    created_at: mem.created_at,
                    importance: mem.importance,
                    valence: mem.valence,
                    score: nscore,
                    scores: ScoreBreakdown {
                        similarity: 0.0,
                        decay: 0.0,
                        recency: 0.0,
                        importance: mem.importance,
                        graph_proximity: nscore,
                        contributions: ScoreContributions {
                            similarity: 0.0,
                            decay: 0.0,
                            recency: 0.0,
                            importance: 0.0,
                            graph_proximity: nscore,
                        },
                        valence_multiplier: 1.0,
                    },
                    why_retrieved: vec![format!("linked via {} from {}", l.link_type, seed_rid)],
                    metadata: mem.metadata,
                    namespace: mem.namespace,
                    certainty: mem.certainty,
                    domain: mem.domain,
                    source: mem.source,
                    emotional_state: mem.emotional_state,
                });
            }
        }

        base.extend(added);
        base.sort_by(|a, b| b.score.total_cmp(&a.score));
        base.truncate(top_k);
        Ok(base)
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
                 AND selection_state = 'selected' \
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
                 AND selection_state = 'selected' \
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
                     AND selection_state = 'selected' \
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

    // ── v0.10 Phase 0: chain-integrity gate ──

    fn supersede(db: &YantrikDB, newer: &str, older: &str) -> Result<String> {
        db.link(
            newer,
            &RecordLink {
                target_rid: older.to_string(),
                link_type: LinkType::Supersedes,
            },
        )
    }

    #[test]
    fn supersede_gate_enforces_single_inbound_successor() {
        // Edge direction is NEW→OLD: "one successor per record" means one
        // selected active INBOUND edge per target (sol correction — an
        // outgoing-edge gate would enforce the wrong invariant).
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let old = rec(&db, "v1 of the fact", 1.0);
        let b = rec(&db, "v2 of the fact", 2.0);
        let c = rec(&db, "rival v2 of the fact", 3.0);

        supersede(&db, &b, &old).unwrap();
        let err = supersede(&db, &c, &old).unwrap_err();
        match err {
            YantrikDbError::SupersedeConflict {
                predecessor_rid,
                existing_successor_rid,
                ..
            } => {
                assert_eq!(predecessor_rid, old);
                assert_eq!(existing_successor_rid, b);
            }
            other => panic!("wrong error: {other}"),
        }

        // Multiple OUTGOING edges stay legal: one new record may merge
        // several predecessors (each predecessor still has one successor).
        let old2 = rec(&db, "parallel old fact", 4.0);
        supersede(&db, &b, &old2).unwrap();
    }

    #[test]
    fn supersede_gate_rejects_cycles_and_bad_endpoints() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let a = rec(&db, "a", 1.0);
        let b = rec(&db, "b", 2.0);
        let c = rec(&db, "c", 3.0);

        // Chain c→b→a, then closing a→c must be refused (multi-hop cycle).
        supersede(&db, &b, &a).unwrap();
        supersede(&db, &c, &b).unwrap();
        assert!(matches!(
            supersede(&db, &a, &c).unwrap_err(),
            YantrikDbError::SupersedeCycle { .. }
        ));

        // Missing endpoint.
        assert!(matches!(
            supersede(&db, &a, "no-such-rid").unwrap_err(),
            YantrikDbError::InvalidLinkEndpoints { .. }
        ));

        // Cross-namespace refusal.
        let other_ns = db
            .record(
                "other namespace fact",
                "semantic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &vec_seed(9.0, 8),
                "tenant-b",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        assert!(matches!(
            supersede(&db, &other_ns, &a).unwrap_err(),
            YantrikDbError::InvalidLinkEndpoints { .. }
        ));

        // Tombstoned endpoint refusal.
        let dead = rec(&db, "doomed", 5.0);
        db.forget(&dead).unwrap();
        let live = rec(&db, "live", 6.0);
        assert!(matches!(
            supersede(&db, &live, &dead).unwrap_err(),
            YantrikDbError::InvalidLinkEndpoints { .. }
        ));
    }

    /// Insert a supersedes CANDIDATE row directly (simulating the
    /// replication accept step) with a controlled HLC byte value.
    fn insert_candidate(db: &YantrikDB, edge_id: &str, src: &str, tgt: &str, hlc_byte: u8) {
        let conn = db.conn();
        conn.execute(
            "INSERT INTO record_links \
             (link_id, source_rid, target_rid, link_type, status, selection_state, \
              created_at, hlc, origin_actor) \
             VALUES (?1, ?2, ?3, 'supersedes', 'active', 'rejected_conflict', 1.0, ?4, 'test')",
            params![edge_id, src, tgt, vec![hlc_byte]],
        )
        .unwrap();
    }

    #[test]
    fn supersedes_fold_is_deterministic_regardless_of_arrival_order() {
        // v0.10 Phase 0: two concurrent successors for one predecessor.
        // Whatever order the candidates arrive in, the fold must select the
        // SAME winner (highest total key = hlc DESC, link_id DESC) and
        // retain the loser as rejected_conflict.
        for arrival in [&["e-low", "e-high"][..], &["e-high", "e-low"][..]] {
            let db = YantrikDB::new(":memory:", 8).unwrap();
            let old = rec(&db, "predecessor", 1.0);
            let a = rec(&db, "successor a", 2.0);
            let b = rec(&db, "successor b", 3.0);
            for edge in arrival {
                match *edge {
                    "e-low" => insert_candidate(&db, "e-low", &a, &old, 10),
                    "e-high" => insert_candidate(&db, "e-high", &b, &old, 20),
                    _ => unreachable!(),
                }
            }
            let fold = {
                let conn = db.conn();
                YantrikDB::refold_supersedes_target(&conn, &old).unwrap()
            };
            let (winner_edge, winner_src) = fold.winner.expect("a winner is selected");
            assert_eq!(
                winner_edge, "e-high",
                "higher HLC wins (arrival {arrival:?})"
            );
            assert_eq!(winner_src, b);
            assert_eq!(fold.losers.len(), 1);
            assert_eq!(fold.losers[0].0, "e-low");

            // The projection: only the winner is a selected active edge.
            let conn = db.conn();
            let selected: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM record_links WHERE target_rid = ?1 \
                     AND link_type = 'supersedes' AND selection_state = 'selected'",
                    params![old],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(selected, 1);
            // The loser is retained (audit), not deleted.
            let total: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM record_links WHERE target_rid = ?1",
                    params![old],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(total, 2, "losing candidate retained durably");
        }
    }

    #[test]
    fn retracting_selected_supersedes_promotes_next_candidate() {
        // v0.10 Phase 0: unlink on a supersedes edge is a replayable
        // retraction; the fold then promotes the next-best durable
        // candidate deterministically.
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let old = rec(&db, "predecessor", 1.0);
        let a = rec(&db, "successor a", 2.0);
        let b = rec(&db, "successor b", 3.0);
        insert_candidate(&db, "e-low", &a, &old, 10);
        insert_candidate(&db, "e-high", &b, &old, 20);
        {
            let conn = db.conn();
            YantrikDB::refold_supersedes_target(&conn, &old).unwrap();
        }

        // Retract the winner (b -> old). The loser (a -> old) is promoted.
        assert!(db.unlink(&b, &old, &LinkType::Supersedes).unwrap());
        let conn = db.conn();
        let (state_high,): (String,) = conn
            .query_row(
                "SELECT selection_state FROM record_links WHERE link_id = 'e-high'",
                [],
                |r| Ok((r.get(0)?,)),
            )
            .unwrap();
        assert_eq!(
            state_high, "retracted",
            "retraction is durable, not a delete"
        );
        let (state_low,): (String,) = conn
            .query_row(
                "SELECT selection_state FROM record_links WHERE link_id = 'e-low'",
                [],
                |r| Ok((r.get(0)?,)),
            )
            .unwrap();
        assert_eq!(state_low, "selected", "next candidate promoted");
    }

    #[test]
    fn verify_chains_reports_legacy_violations_and_clean_graphs() {
        // Report-only audit (v0.10 Phase 0): a clean graph is clean; legacy
        // violations injected below the gate are each detected.
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let a = rec(&db, "a", 1.0);
        let b = rec(&db, "b", 2.0);
        supersede(&db, &b, &a).unwrap();
        assert!(
            db.verify_chains().unwrap().is_clean(),
            "gated graph is clean"
        );

        // Legacy violations via direct SQL (pre-Phase-0 databases).
        let c = rec(&db, "c", 3.0);
        let d = rec(&db, "d", 4.0);
        // multi-successor: second selected inbound edge on `a`.
        insert_candidate(&db, "e-multi", &c, &a, 30);
        {
            let conn = db.conn();
            conn.execute(
                "UPDATE record_links SET selection_state = 'selected' WHERE link_id = 'e-multi'",
                [],
            )
            .unwrap();
            // cycle: d→c and c→d both selected.
            conn.execute_batch(
                &format!(
                    "INSERT INTO record_links (link_id, source_rid, target_rid, link_type, status, selection_state, created_at, hlc, origin_actor) \
                     VALUES ('e-cy1', '{d}', '{c}', 'supersedes', 'active', 'selected', 1.0, x'01', 'test');
                     INSERT INTO record_links (link_id, source_rid, target_rid, link_type, status, selection_state, created_at, hlc, origin_actor) \
                     VALUES ('e-cy2', '{c}', '{d}', 'supersedes', 'active', 'selected', 1.0, x'02', 'test');
                     INSERT INTO record_links (link_id, source_rid, target_rid, link_type, status, selection_state, created_at, hlc, origin_actor) \
                     VALUES ('e-dangle', '{c}', 'ghost-rid', 'supersedes', 'active', 'selected', 1.0, x'03', 'test');"
                ),
            )
            .unwrap();
        }

        let report = db.verify_chains().unwrap();
        assert!(!report.is_clean());
        assert!(
            report.multi_successor.iter().any(|(t, _)| t == &a),
            "multi-successor on {a} detected: {report:?}"
        );
        assert!(
            report.cycle_members.contains(&c) && report.cycle_members.contains(&d),
            "cycle members detected: {report:?}"
        );
        assert!(
            report.dangling.contains(&"e-dangle".to_string()),
            "dangling endpoint detected: {report:?}"
        );
    }

    #[test]
    fn link_retry_returns_original_identity_and_mints_no_new_op() {
        // v0.10 Phase 0 canonical identity: a duplicate link returns the
        // ORIGINAL edge id and does not append another oplog op (a retry
        // must not get a newer replication order — T7 applied to links).
        // Also: the edge row and its oplog op share ONE id and ONE HLC.
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let a = rec(&db, "a", 1.0);
        let b = rec(&db, "b", 2.0);

        let first = supersede(&db, &a, &b).unwrap();
        let ops_after_first: i64 = {
            let conn = db.conn();
            conn.query_row(
                "SELECT COUNT(*) FROM oplog WHERE op_type = 'link'",
                [],
                |r| r.get(0),
            )
            .unwrap()
        };

        let second = supersede(&db, &a, &b).unwrap();
        assert_eq!(first, second, "retry returns the original edge id");

        let conn = db.conn();
        let ops_after_second: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM oplog WHERE op_type = 'link'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(ops_after_first, ops_after_second, "no second op minted");

        // Canonical identity: op_id == link_id and the HLC bytes match.
        let (row_hlc,): (Vec<u8>,) = conn
            .query_row(
                "SELECT hlc FROM record_links WHERE link_id = ?1",
                params![first],
                |r| Ok((r.get(0)?,)),
            )
            .unwrap();
        let (op_hlc,): (Vec<u8>,) = conn
            .query_row(
                "SELECT hlc FROM oplog WHERE op_id = ?1 AND op_type = 'link'",
                params![first],
                |r| Ok((r.get(0)?,)),
            )
            .unwrap();
        assert_eq!(row_hlc, op_hlc, "edge row and oplog op share one HLC");
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
        // v0.9.3: importance correction (text corrections refused); the
        // rid-stability property under test is identical.
        db.correct(&a, None, None, Some(0.9), None, "fix").unwrap();
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
    fn expand_links_demotes_superseded_and_surfaces_superseder() {
        // The redteam's motivating correctness scenario. A supersedes B.
        // A query close to B should, with expand_links, demote B and
        // surface A above it.
        let db = YantrikDB::new(":memory:", 8).unwrap();
        // B and the query are near-identical; A is also near but we rely
        // on the link, not similarity, to rank it.
        let b = rec(&db, "old fact about widgets", 5.0);
        let a = rec(&db, "corrected fact about widgets", 5.05);
        db.link(
            &a,
            &RecordLink {
                target_rid: b.clone(),
                link_type: LinkType::Supersedes,
            },
        )
        .unwrap();

        let query = vec_seed(5.0, 8); // closest to B

        // Baseline (expand_links=0): B is present, not demoted.
        let base = db
            .recall_with_links(
                &query, 5, None, None, false, false, None, true, None, None, None, None, None, 0,
            )
            .unwrap();
        let base_b = base.iter().find(|r| r.rid == b).expect("B in baseline");
        let base_b_score = base_b.score;

        // With expansion: B is demoted (score strictly lower than baseline)
        // and A is present.
        let expanded = db
            .recall_with_links(
                &query, 5, None, None, false, false, None, true, None, None, None, None, None, 1,
            )
            .unwrap();
        let exp_b = expanded
            .iter()
            .find(|r| r.rid == b)
            .expect("B still present");
        assert!(
            exp_b.score < base_b_score,
            "superseded B must be demoted: baseline={base_b_score}, expanded={}",
            exp_b.score
        );
        assert!(
            expanded.iter().any(|r| r.rid == a),
            "superseder A must be present in expanded results"
        );
        // A should rank above B after demotion.
        let pos_a = expanded.iter().position(|r| r.rid == a).unwrap();
        let pos_b = expanded.iter().position(|r| r.rid == b).unwrap();
        assert!(pos_a < pos_b, "A (superseder) must rank above demoted B");
    }

    #[test]
    fn expand_links_zero_is_identical_to_recall() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let _a = rec(&db, "alpha", 1.0);
        let _b = rec(&db, "beta", 2.0);
        let query = vec_seed(1.0, 8);
        let via_links = db
            .recall_with_links(
                &query, 5, None, None, false, false, None, true, None, None, None, None, None, 0,
            )
            .unwrap();
        let direct = db
            .recall(
                &query, 5, None, None, false, false, None, true, None, None, None, None, None,
            )
            .unwrap();
        assert_eq!(
            via_links.iter().map(|r| &r.rid).collect::<Vec<_>>(),
            direct.iter().map(|r| &r.rid).collect::<Vec<_>>(),
            "expand_links=0 must match recall() exactly"
        );
    }

    #[test]
    fn expand_links_surfaces_neighbor_excluded_from_base_pool() {
        // B supports A. A is in a different domain, so a domain-filtered
        // recall excludes A from the base pool entirely — yet expand_links
        // must still surface A via B's outbound support link, labelled as
        // link-sourced. (In a tiny DB without the filter, A would already
        // be in the base pool; the domain filter is what forces the
        // genuine neighbor-surfacing path.)
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let b = db
            .record(
                "matches query in default domain",
                "semantic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &vec_seed(3.0, 8),
                "default",
                0.8,
                "default", // domain
                "user",
                None,
            )
            .unwrap();
        let a = db
            .record(
                "supporting evidence in a hidden domain",
                "semantic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &vec_seed(3.05, 8),
                "default",
                0.8,
                "hidden", // different domain -> excluded by the filter below
                "user",
                None,
            )
            .unwrap();
        db.link(
            &b,
            &RecordLink {
                target_rid: a.clone(),
                link_type: LinkType::Supports,
            },
        )
        .unwrap();

        let query = vec_seed(3.0, 8);
        // domain="default" excludes A from the base recall pool.
        let expanded = db
            .recall_with_links(
                &query,
                5,
                None,
                None,
                false,
                false,
                None,
                true,
                None,
                Some("default"),
                None,
                None,
                None,
                1,
            )
            .unwrap();
        let a_res = expanded
            .iter()
            .find(|r| r.rid == a)
            .expect("linked supporter A must surface even though excluded from base pool");
        assert!(
            a_res.why_retrieved.iter().any(|w| w.contains("linked via")),
            "surfaced neighbor must be labelled as link-sourced, got {:?}",
            a_res.why_retrieved
        );
    }

    #[test]
    fn record_with_links_partial_reports_per_link_outcomes() {
        use crate::types::LinkResult;
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let b = rec(&db, "target b", 1.0);

        // links array: valid Advances→b, a DUPLICATE Advances→b (same
        // source/target/type within the call → AlreadyExists on the
        // second), and an empty-target link (Failed). The record must
        // still commit despite the failure.
        let (rid, results) = db
            .record_with_links_partial(
                "partial test",
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
                        link_type: LinkType::Advances,
                    },
                    RecordLink {
                        target_rid: b.clone(),
                        link_type: LinkType::Advances,
                    },
                    RecordLink {
                        target_rid: String::new(),
                        link_type: LinkType::Supports,
                    },
                ],
            )
            .unwrap();

        // Record committed despite the failing link.
        assert!(db.get(&rid).unwrap().is_some());
        assert_eq!(results.len(), 3);
        assert!(
            matches!(results[0], LinkResult::Inserted { .. }),
            "first link inserted, got {:?}",
            results[0]
        );
        assert!(
            matches!(results[1], LinkResult::AlreadyExists { .. }),
            "duplicate link already-exists, got {:?}",
            results[1]
        );
        assert!(
            matches!(results[2], LinkResult::Failed { .. }),
            "empty-target link failed, got {:?}",
            results[2]
        );

        // Net effect: exactly one active Advances link to b.
        let out = db
            .linked_records(&rid, LinkDirection::Outbound, Some(&LinkType::Advances))
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].rid, b);
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
