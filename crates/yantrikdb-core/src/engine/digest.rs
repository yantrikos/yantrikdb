//! Session-start digest — wake up already knowing (task 38).
//!
//! Session start otherwise means a blank agent composing good recall queries
//! from nothing — and boot quality then depends on that session's
//! query-writing skill, which is the documented substrate-underuse drift
//! failure mode by construction. This materializes a single, token-budgeted
//! briefing the host can inject at session start so continuity is ambient:
//! the narrative chain head (who I am, from the last verified entry), the live
//! high-importance decisions, the open conflicts and pending triggers that
//! need attention, and when hygiene last ran.
//!
//! Everything here is assembled from primitives built across this program
//! (chain_head, get_conflicts, get_pending_triggers, last_maintenance_cycle),
//! so the digest is a thin, cheap composition — not a new subsystem.

use rusqlite::params;

use crate::error::Result;

use super::YantrikDB;

/// How much the digest pulls in. Kept small so the briefing fits a tight
/// token budget at session start.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionDigestConfig {
    /// The append-only identity/narrative chain to read the head of.
    pub narrative_namespace: Option<String>,
    /// **v0.9.3 isolation scope (sol converged plan item 2).** When set, the
    /// digest's CONTENT aggregates (top decisions, open conflicts + count)
    /// are filtered to this namespace, so a multi-tenant host composing one
    /// digest per tenant never mixes another tenant's memories in. `None`
    /// keeps the original explicit-global behavior (single-tenant embedded
    /// use, where the caller owns the whole database anyway). Pending
    /// TRIGGERS remain global in both modes — `trigger_log` rows carry
    /// engine-generated operational reasons keyed by rid, not memory text,
    /// and namespace-scoping them requires a source_rids join deferred to
    /// the v0.10 reliability program.
    pub namespace: Option<String>,
    pub max_decisions: usize,
    pub max_conflicts: usize,
    pub max_triggers: usize,
    /// Max characters of each memory's text to include as a snippet.
    pub snippet_chars: usize,
}

impl Default for SessionDigestConfig {
    fn default() -> Self {
        Self {
            narrative_namespace: None,
            namespace: None,
            max_decisions: 8,
            max_conflicts: 5,
            max_triggers: 5,
            snippet_chars: 240,
        }
    }
}

/// A memory rendered for the digest — just enough to orient.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DigestEntry {
    pub rid: String,
    pub snippet: String,
    pub importance: f64,
    pub created_at: f64,
    pub namespace: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DigestConflict {
    pub conflict_id: String,
    pub conflict_type: String,
    pub priority: String,
    pub memory_a: String,
    pub memory_b: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DigestTrigger {
    pub trigger_id: String,
    pub trigger_type: String,
    pub urgency: f64,
    pub reason: String,
}

/// The session-start briefing.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SessionDigest {
    /// Head of the narrative chain (the latest verified self-entry), if a
    /// narrative namespace was configured and non-empty.
    pub narrative_head: Option<DigestEntry>,
    /// Live high-importance memories — the decisions worth inheriting.
    pub top_decisions: Vec<DigestEntry>,
    /// Open conflicts needing resolution (also their total count).
    pub open_conflicts: Vec<DigestConflict>,
    pub open_conflict_count: usize,
    /// Pending triggers due (also their total count).
    pub pending_triggers: Vec<DigestTrigger>,
    pub pending_trigger_count: usize,
    /// JSON summary of the last maintenance cycle, if any has run.
    pub last_maintenance: Option<String>,
}

impl YantrikDB {
    /// Materialize the session-start digest. One call, host-injected at boot.
    pub fn session_digest(&self, config: &SessionDigestConfig) -> Result<SessionDigest> {
        let mut digest = SessionDigest::default();
        let snip = |s: &str, n: usize| -> String { s.chars().take(n).collect::<String>() };

        // Narrative chain head — the latest verified self-entry.
        if let Some(ns) = config.narrative_namespace.as_deref() {
            if let Some(head) = self.chain_head(ns)? {
                digest.narrative_head = Some(DigestEntry {
                    snippet: snip(&head.text, config.snippet_chars),
                    rid: head.rid,
                    importance: head.importance,
                    created_at: head.created_at,
                    namespace: head.namespace,
                });
            }
        }

        // Top live decisions — highest importance, most recent first. Read
        // directly so we can rank by importance (list_records ranks by rid).
        // v0.9.3: scoped to config.namespace when set (isolation contract).
        {
            let conn = self.conn();
            let (sql, ns_param) = match config.namespace.as_deref() {
                Some(ns) => (
                    "SELECT rid, text, importance, created_at, namespace FROM memories \
                     WHERE consolidation_status = 'active' AND importance >= 0.7 \
                     AND namespace = ?2 \
                     ORDER BY importance DESC, created_at DESC LIMIT ?1",
                    Some(ns.to_string()),
                ),
                None => (
                    "SELECT rid, text, importance, created_at, namespace FROM memories \
                     WHERE consolidation_status = 'active' AND importance >= 0.7 \
                     ORDER BY importance DESC, created_at DESC LIMIT ?1",
                    None,
                ),
            };
            let mut stmt = conn.prepare(sql)?;
            let map_row = |r: &rusqlite::Row| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, f64>(2)?,
                    r.get::<_, f64>(3)?,
                    r.get::<_, String>(4)?,
                ))
            };
            let rows = match &ns_param {
                Some(ns) => stmt
                    .query_map(params![config.max_decisions as i64, ns], map_row)?
                    .collect::<std::result::Result<Vec<_>, _>>()?,
                None => stmt
                    .query_map(params![config.max_decisions as i64], map_row)?
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            };
            drop(stmt);
            drop(conn);
            for (rid, enc_text, importance, created_at, namespace) in rows {
                let text = self.decrypt_text(&enc_text).unwrap_or(enc_text);
                digest.top_decisions.push(DigestEntry {
                    snippet: snip(&text, config.snippet_chars),
                    rid,
                    importance,
                    created_at,
                    namespace,
                });
            }
        }

        // Open conflicts needing attention + total count.
        // v0.9.3: when a namespace scope is set, conflicts are filtered via
        // their memory_a's namespace (the conflicts table itself carries no
        // namespace column; memory_a and memory_b share one by construction
        // since conflict detection compares within a namespace).
        match config.namespace.as_deref() {
            Some(ns) => {
                let conn = self.conn();
                let mut stmt = conn.prepare(
                    "SELECT c.conflict_id, c.conflict_type, c.priority, c.memory_a, c.memory_b \
                     FROM conflicts c \
                     WHERE c.status = 'open' AND EXISTS (\
                       SELECT 1 FROM memories m WHERE m.rid = c.memory_a AND m.namespace = ?1) \
                     ORDER BY c.detected_at DESC LIMIT ?2",
                )?;
                digest.open_conflicts = stmt
                    .query_map(params![ns, config.max_conflicts as i64], |r| {
                        Ok(DigestConflict {
                            conflict_id: r.get(0)?,
                            conflict_type: r.get(1)?,
                            priority: r.get(2)?,
                            memory_a: r.get(3)?,
                            memory_b: r.get(4)?,
                        })
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                digest.open_conflict_count = conn.query_row(
                    "SELECT COUNT(*) FROM conflicts c \
                     WHERE c.status = 'open' AND EXISTS (\
                       SELECT 1 FROM memories m WHERE m.rid = c.memory_a AND m.namespace = ?1)",
                    params![ns],
                    |r| r.get::<_, i64>(0),
                )? as usize;
            }
            None => {
                let conflicts =
                    self.get_conflicts(Some("open"), None, None, None, None, config.max_conflicts)?;
                digest.open_conflicts = conflicts
                    .into_iter()
                    .map(|c| DigestConflict {
                        conflict_id: c.conflict_id,
                        conflict_type: c.conflict_type,
                        priority: c.priority,
                        memory_a: c.memory_a,
                        memory_b: c.memory_b,
                    })
                    .collect();
                digest.open_conflict_count = {
                    let conn = self.conn();
                    conn.query_row(
                        "SELECT COUNT(*) FROM conflicts WHERE status = 'open'",
                        [],
                        |r| r.get::<_, i64>(0),
                    )? as usize
                };
            }
        }

        // Pending triggers due + total count.
        let triggers = crate::triggers::get_pending_triggers(self, config.max_triggers)?;
        digest.pending_triggers = triggers
            .into_iter()
            .map(|t| DigestTrigger {
                trigger_id: t.trigger_id,
                trigger_type: t.trigger_type,
                urgency: t.urgency,
                reason: t.reason,
            })
            .collect();
        digest.pending_trigger_count = {
            let conn = self.conn();
            conn.query_row(
                "SELECT COUNT(*) FROM trigger_log WHERE status = 'pending'",
                [],
                |r| r.get::<_, i64>(0),
            )? as usize
        };

        // When hygiene last ran.
        digest.last_maintenance = self.last_maintenance_cycle()?;

        Ok(digest)
    }

    /// Task 40 — end-of-session auto-capture. Takes an agent-provided session
    /// summary and drafts candidate memories from it: atomized into facts
    /// (the same segmenter as the mega-blob split), stored as low-importance
    /// PROVISIONAL semantic memories (`metadata.provisional = true`,
    /// `source = "session_auto_capture"`) for cheap later review / the sleep
    /// cycle to consolidate. Returns the new rids.
    ///
    /// This moves the structuring work off the agent's hot path — a session
    /// that never paused to call `remember` still yields well-formed memories
    /// at the end. (The summary itself is the agent's to produce; the engine
    /// cannot observe the conversation.)
    pub fn draft_memories_from_summary(
        &self,
        summary: &str,
        namespace: &str,
        domain: &str,
    ) -> Result<Vec<String>> {
        // Smaller target than the mega-blob split — auto-capture wants
        // granular candidate facts (one thought each) for review.
        let facts = super::split::segment_into_atomic_facts(summary, 120, 60);
        let meta = serde_json::json!({
            "provisional": true,
            "kind": "session_auto_capture",
        });
        let mut rids = Vec::with_capacity(facts.len());
        for fact in facts {
            let rid = self.record_text(
                &fact,
                "semantic",
                0.5,
                0.0,
                604_800.0,
                &meta,
                namespace,
                0.7,
                domain,
                "session_auto_capture",
                None,
            )?;
            rids.push(rid);
        }
        Ok(rids)
    }
}
