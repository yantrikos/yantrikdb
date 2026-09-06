use rusqlite::{params, Connection};

use crate::error::Result;
use crate::types::{Edge, Entity};

use super::{now, YantrikDB};

/// C5b (wheel piece 2) — heal the possessive pollution the tokenizer
/// exemption persisted. For every entity whose name carries a terminal
/// possessive clitic (`Taylor's`, `Hermes'`, `Hermes's`) and whose bare
/// form already exists as an entity, write a REVERSIBLE alias row
/// (`entity_aliases`, source `possessive_migration_v1`). The alias is
/// consumed by [`crate::graph_index::GraphIndex::build_from_db`], which
/// folds aliased entities into their canonical at load — persisted rows
/// are never rewritten, so deleting the alias rows reverses the merge.
///
/// Deliberately conservative, per the agreed migration design:
/// - only TERMINAL `'s` / `'` are stripped (repeatedly, so `Hermes's` →
///   `Hermes'` → `Hermes`), because those are possessives; contractions
///   (`Don't`, `I'm`, `GC'd`) do not match the rule and are left as
///   unreachable dead nodes for a later prune — aliasing `Don't` to a
///   real entity named `Don` would be a false merge.
/// - the canonical must ALREADY exist (case-insensitive match, exact-
///   case row wins ties); we never mint entities here.
/// - type conflicts resolve to the canonical row's type by construction
///   (the fold keeps the canonical's metadata; the phantom's type was
///   an artifact of misparsing possessive contexts).
///
/// Idempotent (upsert) and cheap (one pass over apostrophe entities),
/// so it runs on every open, healing databases written by pre-C5a
/// engines. Returns `(aliases_written, apostrophe_entities_total)` —
/// the pollution census before/after is the migration's success metric.
pub(crate) fn migrate_possessive_aliases(conn: &Connection) -> Result<(usize, usize)> {
    let names: Vec<String> = {
        let mut stmt = conn.prepare("SELECT name FROM entities")?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        rows
    };
    let mut by_lower: std::collections::HashMap<String, &str> =
        std::collections::HashMap::with_capacity(names.len());
    for n in &names {
        // Exact-case rows win ties: insert lowercase key only if absent,
        // then let an exact-case duplicate overwrite (same value anyway).
        by_lower.entry(n.to_lowercase()).or_insert(n.as_str());
    }

    let ts = now();
    let mut written = 0usize;
    let mut apostrophes = 0usize;
    for name in &names {
        if !name.contains('\'') {
            continue;
        }
        apostrophes += 1;
        let mut stripped = name.as_str();
        loop {
            if let Some(s) = stripped.strip_suffix("'s").or_else(|| {
                stripped
                    .strip_suffix("'S")
                    .or_else(|| stripped.strip_suffix('\''))
            }) {
                stripped = s;
            } else {
                break;
            }
        }
        if stripped.is_empty() || stripped == name || stripped.contains('\'') {
            continue; // not a terminal possessive (contraction, mid-name quote)
        }
        let Some(&canonical) = by_lower.get(&stripped.to_lowercase()) else {
            continue; // no existing canonical — never mint one here
        };
        if canonical.eq_ignore_ascii_case(name) {
            continue;
        }
        let changes = conn.execute(
            "INSERT INTO entity_aliases (alias, canonical_name, namespace, source, created_at) \
             VALUES (?1, ?2, 'default', 'possessive_migration_v1', ?3) \
             ON CONFLICT(alias, namespace) DO NOTHING",
            params![name, canonical, ts],
        )?;
        written += changes;
    }
    Ok((written, apostrophes))
}

/// Outcome of a [`YantrikDB::auto_relate`] pass (task 44).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct AutoRelateReport {
    pub dry_run: bool,
    /// Co-occurring entity pairs considered this pass.
    pub pairs_considered: usize,
    /// Edges upserted (idempotent — re-running refreshes rather than dupes).
    pub edges_upserted: usize,
}

/// Resolve (src, rel_type, dst, namespace) to a proposition_id, creating
/// the proposition row if it doesn't exist. Propositions are the canonical
/// identity under which all claim rows about the same triple aggregate;
/// RFC 008 mobility_state is keyed by (proposition_id, regime).
///
/// UNIQUE(src, rel_type, dst, namespace) is enforced at the schema level,
/// so a concurrent inserter racing on the same triple will produce one
/// proposition row and one duplicate-key violation; we recover from the
/// violation by reading the existing row.
fn ensure_proposition(
    conn: &Connection,
    src: &str,
    rel_type: &str,
    dst: &str,
    namespace: &str,
    created_at: f64,
) -> Result<String> {
    // Fast path: already exists.
    let existing: Option<String> = conn
        .query_row(
            "SELECT proposition_id FROM propositions \
             WHERE src = ?1 AND rel_type = ?2 AND dst = ?3 AND namespace = ?4",
            params![src, rel_type, dst, namespace],
            |row| row.get(0),
        )
        .ok();
    if let Some(id) = existing {
        return Ok(id);
    }

    let proposition_id = crate::id::new_id();
    conn.execute(
        "INSERT INTO propositions (proposition_id, src, rel_type, dst, namespace, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
         ON CONFLICT(src, rel_type, dst, namespace) DO NOTHING",
        params![proposition_id, src, rel_type, dst, namespace, created_at],
    )?;

    // The INSERT may have been a no-op due to the conflict — re-read to
    // get whichever id actually won (ours or the racer's).
    let id: String = conn.query_row(
        "SELECT proposition_id FROM propositions \
         WHERE src = ?1 AND rel_type = ?2 AND dst = ?3 AND namespace = ?4",
        params![src, rel_type, dst, namespace],
        |row| row.get(0),
    )?;
    Ok(id)
}

impl YantrikDB {
    /// Create or update a relationship between entities.
    #[tracing::instrument(skip(self))]
    pub fn relate(&self, src: &str, dst: &str, rel_type: &str, weight: f64) -> Result<String> {
        let edge_id = crate::id::new_id();
        let ts = now();
        // #148: the edge's causal timestamp, minted once and carried VERBATIM
        // into both the claims row and the relate op payload (edge_hlc_hex),
        // so every replica's LWW compares the same value — the record_links
        // edge-identity pattern. Replication LWW compares THIS, not
        // created_at: wall clocks skew between nodes.
        let edge_hlc = self.tick_hlc().to_bytes().to_vec();

        // Classify entity types using relationship semantics
        let (src_type, dst_type) = crate::graph::classify_with_relationship(src, dst, rel_type);

        // Phase 1: Lock conn for all SQL operations, then drop
        {
            let conn = self.conn.lock();
            // Legacy relate() uses default extractor/polarity/namespace so the
            // effective unique key is (src, dst, rel_type, 'manual', 1, 'default').
            // Local overwrite stays unconditional: local calls are
            // serialized, and a fresh tick_hlc() happens-after any merged
            // remote op, so the new edge_hlc is always the causal maximum.
            conn.execute(
                "INSERT INTO claims (claim_id, src, dst, rel_type, weight, created_at, hlc) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7) \
                 ON CONFLICT(src, dst, rel_type, extractor, polarity, namespace) \
                 DO UPDATE SET claim_id = ?1, weight = ?5, created_at = ?6, hlc = ?7",
                params![edge_id, src, dst, rel_type, weight, ts, edge_hlc],
            )?;

            // Ensure entities exist with classified entity_type
            for (entity, etype) in [(src, src_type), (dst, dst_type)] {
                conn.execute(
                    "INSERT INTO entities (name, entity_type, first_seen, last_seen) \
                     VALUES (?1, ?2, ?3, ?4) \
                     ON CONFLICT(name) DO UPDATE SET last_seen = ?4, mention_count = mention_count + 1, \
                     entity_type = CASE WHEN entities.entity_type = 'unknown' THEN ?2 ELSE entities.entity_type END",
                    params![entity, etype, ts, ts],
                )?;
            }
        } // conn dropped

        // Phase 2: Lock graph_index write for in-memory updates, then drop
        {
            let mut gi = self.graph_index.write();
            gi.add_entity(src, src_type);
            gi.add_entity(dst, dst_type);
            gi.add_edge(src, dst, weight as f32);
        } // graph_index dropped

        // Backfill memory_entities for newly-created entities.
        // When remember() runs BEFORE relate(), the memory doesn't get linked
        // because the entity doesn't exist yet. Fix: scan active memories for
        // mentions of the src/dst entities and create links retroactively.
        self.backfill_memory_entities_for(&[src, dst])?;

        self.log_op(
            "relate",
            Some(&edge_id),
            &serde_json::json!({
                "edge_id": edge_id,
                "src": src,
                "dst": dst,
                "rel_type": rel_type,
                "weight": weight,
                "created_at": ts,
                "edge_hlc_hex": hex::encode(&edge_hlc),
            }),
            None,
        )?;

        Ok(edge_id)
    }

    /// Task 44 — auto-relate: raise graph density from plain writes (no agent
    /// `relate` call) by linking entities that co-occur in the same memory.
    /// The graph-lift eval (task 43) showed connectivity improves recall on
    /// connected data; this creates that connectivity continuously.
    ///
    /// Edges are `co_occurs_with`, weighted by co-occurrence count, upserted
    /// idempotently (the claims UNIQUE key), strongest pairs first and capped
    /// at `max_edges` per pass so it stays cheap; the sleep cycle runs it
    /// incrementally. Refreshes the in-memory graph index so recall's
    /// expand_entities sees the new edges.
    pub fn auto_relate(&self, dry_run: bool, max_edges: usize) -> Result<AutoRelateReport> {
        let mut report = AutoRelateReport {
            dry_run,
            ..Default::default()
        };

        // Co-occurring entity pairs (same memory), strongest first. `a < b`
        // dedups the symmetric pair and rules out self-loops.
        let pairs: Vec<(String, String, i64)> = {
            let conn = self.conn.lock();
            let mut stmt = conn.prepare(
                "SELECT a.entity_name, b.entity_name, COUNT(*) AS cooccur \
                 FROM memory_entities a \
                 JOIN memory_entities b \
                   ON a.memory_rid = b.memory_rid AND a.entity_name < b.entity_name \
                 GROUP BY a.entity_name, b.entity_name \
                 ORDER BY cooccur DESC LIMIT ?1",
            )?;
            let collected = stmt
                .query_map(params![max_edges as i64], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, i64>(2)?,
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            collected
        };
        report.pairs_considered = pairs.len();

        if dry_run || pairs.is_empty() {
            return Ok(report);
        }

        let ts = now();
        {
            let conn = self.conn.lock();
            for (src, dst, cooccur) in &pairs {
                let edge_id = crate::id::new_id();
                // Normalize co-occurrence count into a [0.1, 1.0] weight.
                let weight = ((*cooccur as f64) / 10.0).clamp(0.1, 1.0);
                conn.execute(
                    "INSERT INTO claims (claim_id, src, dst, rel_type, weight, created_at) \
                     VALUES (?1, ?2, ?3, 'co_occurs_with', ?4, ?5) \
                     ON CONFLICT(src, dst, rel_type, extractor, polarity, namespace) \
                     DO UPDATE SET weight = ?4, created_at = ?5",
                    params![edge_id, src, dst, weight, ts],
                )?;
                report.edges_upserted += 1;
            }
        }

        // Refresh the in-memory graph index so recall's expand_entities picks
        // up the new edges.
        self.rebuild_graph_index()?;

        tracing::info!(
            target: "yantrikdb::audit::graph",
            pairs = report.pairs_considered,
            edges_upserted = report.edges_upserted,
            "auto-relate pass complete",
        );

        Ok(report)
    }

    /// **Issue #9 — deterministic entity-edge upsert primitive for cluster replication.**
    ///
    /// Sibling of `relate()` that takes a caller-assigned `edge_id` (replaces
    /// today's relate() which generates one) + caller-supplied timestamp.
    /// Used by yantrikdb-server's cluster-mode applier so replicated edges
    /// converge to identical engine state across leader + followers.
    ///
    /// # Contract
    ///
    /// - **Idempotent on edge_id**: a second call with the same edge_id +
    ///   identical other fields succeeds without error and produces
    ///   identical engine state. Implementation: INSERT OR IGNORE on
    ///   the claims primary key (claim_id).
    /// - **Caller-supplied timestamp**: `created_at_unix_micros` materialized
    ///   into `claims.created_at` (REAL seconds). No engine `now()` call.
    /// - **UNIQUE conflict on (src, dst, rel_type, extractor, polarity, namespace)**:
    ///   if a different edge_id already covers the same logical edge, the
    ///   second insert silently no-ops. Caller is responsible for using the
    ///   canonical edge_id chosen by the leader; concurrent leaders are not
    ///   supported (RFC 010 single-leader assumption).
    ///
    /// extractor defaults to 'manual', polarity to 1, modality to 'asserted'
    /// — same defaults `relate()` uses. The `claims` table backs both APIs.
    /// Spec uses the name `entity_edges` for the abstract concept; physical
    /// table is `claims` since RFC 006.
    ///
    /// **Caller-supplied `seq`** (cluster mode): when `Some(n)`, used as
    /// the visible_seq bump value for the namespace; engine ratchets
    /// `vec_seq` to at least `n`. Per the cluster RYW design lock the seq
    /// IS the openraft commit-log index, so a follower replaying the
    /// edge-upsert log entry advances `visible_seq[namespace]` to exactly
    /// the same watermark the leader did. Single-node callers pass `None`.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip(self), fields(edge_id, src, dst, rel_type, namespace))]
    pub fn upsert_entity_edge_with_id(
        &self,
        edge_id: &str,
        src: &str,
        dst: &str,
        rel_type: &str,
        weight: f64,
        namespace: &str,
        created_at_unix_micros: i64,
        seq: Option<u64>,
    ) -> Result<()> {
        let ts_secs = (created_at_unix_micros as f64) / 1_000_000.0;
        let (src_type, dst_type) = crate::graph::classify_with_relationship(src, dst, rel_type);

        // SAVEPOINT-guarded conn block. INSERT OR IGNORE on claim_id PK
        // gives idempotency on edge_id; the UNIQUE(src, dst, rel_type,
        // extractor, polarity, namespace) acts as a secondary filter.
        let was_new_row: bool = {
            let conn = self.conn();

            let sp = crate::engine::savepoint::SavepointGuard::new(&conn, "upsert_edge")?;
            let inserted = conn.execute(
                "INSERT OR IGNORE INTO claims \
                     (claim_id, src, dst, rel_type, weight, created_at, namespace) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![edge_id, src, dst, rel_type, weight, ts_secs, namespace],
            )?;

            let was_new = inserted == 1;

            if was_new {
                // Ensure entities exist with classified entity_type.
                for (entity, etype) in [(src, src_type), (dst, dst_type)] {
                    conn.execute(
                            "INSERT INTO entities (name, entity_type, first_seen, last_seen) \
                             VALUES (?1, ?2, ?3, ?3) \
                             ON CONFLICT(name) DO UPDATE SET last_seen = ?3, mention_count = mention_count + 1, \
                             entity_type = CASE WHEN entities.entity_type = 'unknown' THEN ?2 ELSE entities.entity_type END",
                            params![entity, etype, ts_secs],
                        )?;
                }
            }

            sp.release()?;

            was_new
        };

        // In-memory graph_index update only on first insert (idempotent on
        // replay since add_entity/add_edge dedupe by name).
        if was_new_row {
            let mut gi = self.graph_index.write();
            gi.add_entity(src, src_type);
            gi.add_entity(dst, dst_type);
            gi.add_edge(src, dst, weight as f32);
        }

        // Skip backfill_memory_entities on cluster path — followers will
        // receive memory_entities updates via the record_with_rid log
        // entries that span the same backfill window.

        // Bump visible_seq for cluster RYW determinism. Even on idempotent
        // re-apply (was_new_row == false) we bump — followers must reach
        // the same watermark the leader did regardless of whether the SQL
        // state is novel; this is what makes recall_with_seq work uniformly
        // across leader + followers per the design lock.
        let seq = self.assign_seq(seq);
        self.bump_visible_seq(namespace, seq);

        if was_new_row {
            self.log_op(
                "upsert_entity_edge_with_id",
                Some(edge_id),
                &serde_json::json!({
                    "edge_id": edge_id,
                    "src": src,
                    "dst": dst,
                    "rel_type": rel_type,
                    "weight": weight,
                    "namespace": namespace,
                    "created_at_unix_micros": created_at_unix_micros,
                }),
                None,
            )?;
        }

        Ok(())
    }

    /// **Issue #9 — deterministic entity-edge delete primitive for cluster replication.**
    ///
    /// Tombstones a claim by edge_id. Idempotent on missing — deleting a
    /// non-existent edge_id returns `Ok(())`, not an error. Snapshot-install +
    /// log-replay overlap means double-delete is normal cluster behavior.
    ///
    /// **Caller-supplied namespace** is required for the visible_seq bump
    /// even when the local row is missing (snapshot-lag determinism on
    /// followers). The cluster applier always has it from the
    /// replication payload.
    ///
    /// **Caller-supplied `seq`** (cluster mode): when `Some(n)`, used as
    /// the visible_seq bump value; engine ratchets `vec_seq` to at least
    /// `n`. Single-node callers pass `None`.
    ///
    /// Note: in-memory `graph_index` retains the edge until the next engine
    /// restart (no remove_edge primitive yet). The `claims` row is correctly
    /// tombstoned and the SQL-backed views filter it out. Best-effort recall
    /// via graph_index may include the stale edge until reload — Phase 4.3 +
    /// graph_index reload-on-tombstone is a follow-up.
    #[tracing::instrument(skip(self))]
    pub fn delete_entity_edge_with_id(
        &self,
        edge_id: &str,
        namespace: &str,
        requested_at_unix_micros: i64,
        seq: Option<u64>,
    ) -> Result<()> {
        let ts_secs = (requested_at_unix_micros as f64) / 1_000_000.0;
        let was_newly_tombstoned = {
            let conn = self.conn();
            let changes = conn.execute(
                "UPDATE claims SET tombstoned = 1, created_at = ?1 \
                 WHERE claim_id = ?2 AND tombstoned = 0",
                params![ts_secs, edge_id],
            )?;
            changes > 0
        };

        // Bump visible_seq for cluster RYW determinism — see upsert sibling
        // for rationale. Bump on idempotent re-apply too.
        let seq = self.assign_seq(seq);
        self.bump_visible_seq(namespace, seq);

        if was_newly_tombstoned {
            self.log_op(
                "delete_entity_edge_with_id",
                Some(edge_id),
                &serde_json::json!({
                    "edge_id": edge_id,
                    "namespace": namespace,
                    "requested_at_unix_micros": requested_at_unix_micros,
                }),
                None,
            )?;
        }

        Ok(())
    }

    /// Get all edges connected to an entity.
    pub fn get_edges(&self, entity: &str) -> Result<Vec<Edge>> {
        let conn = self.conn.lock();
        let mut stmt =
            conn.prepare("SELECT * FROM edges WHERE (src = ?1 OR dst = ?1) AND tombstoned = 0")?;

        let edges = stmt
            .query_map(params![entity], |row| {
                Ok(Edge {
                    edge_id: row.get("edge_id")?,
                    src: row.get("src")?,
                    dst: row.get("dst")?,
                    rel_type: row.get("rel_type")?,
                    weight: row.get("weight")?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(edges)
    }

    /// Search entities by name pattern. If pattern is None, returns all entities
    /// ordered by most recently seen. Pattern uses SQL LIKE syntax (% for wildcard).
    pub fn search_entities(
        &self,
        pattern: Option<&str>,
        entity_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Entity>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        // SQL contains legacy rows intentionally retained for auditability.
        // Overfetch before intersecting with the active in-memory graph so
        // suppressed headings and folded possessive aliases cannot occupy the
        // caller's entire result window.
        let query_limit = limit
            .saturating_mul(20)
            .min(10_000.max(limit))
            .min(i64::MAX as usize);
        let (sql, params_vec): (String, Vec<Box<dyn rusqlite::types::ToSql>>) =
            match (pattern, entity_type) {
                (Some(p), Some(t)) => (
                    "SELECT name, entity_type, first_seen, last_seen, mention_count \
                 FROM entities WHERE name LIKE ?1 AND entity_type = ?2 \
                 ORDER BY last_seen DESC LIMIT ?3"
                        .to_string(),
                    vec![
                        Box::new(format!("%{}%", p)) as Box<dyn rusqlite::types::ToSql>,
                        Box::new(t.to_string()),
                        Box::new(query_limit as i64),
                    ],
                ),
                (Some(p), None) => (
                    "SELECT name, entity_type, first_seen, last_seen, mention_count \
                 FROM entities WHERE name LIKE ?1 \
                 ORDER BY last_seen DESC LIMIT ?2"
                        .to_string(),
                    vec![
                        Box::new(format!("%{}%", p)) as Box<dyn rusqlite::types::ToSql>,
                        Box::new(query_limit as i64),
                    ],
                ),
                (None, Some(t)) => (
                    "SELECT name, entity_type, first_seen, last_seen, mention_count \
                 FROM entities WHERE entity_type = ?1 \
                 ORDER BY last_seen DESC LIMIT ?2"
                        .to_string(),
                    vec![
                        Box::new(t.to_string()) as Box<dyn rusqlite::types::ToSql>,
                        Box::new(query_limit as i64),
                    ],
                ),
                (None, None) => (
                    "SELECT name, entity_type, first_seen, last_seen, mention_count \
                 FROM entities ORDER BY last_seen DESC LIMIT ?1"
                        .to_string(),
                    vec![Box::new(query_limit as i64) as Box<dyn rusqlite::types::ToSql>],
                ),
            };

        let conn = self.conn.lock();
        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();
        let mut entities = stmt
            .query_map(param_refs.as_slice(), |row| {
                Ok(Entity {
                    name: row.get("name")?,
                    entity_type: row.get("entity_type")?,
                    first_seen: row.get("first_seen")?,
                    last_seen: row.get("last_seen")?,
                    mention_count: row.get("mention_count")?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let admitted: std::collections::HashSet<String> = self
            .graph_index
            .read()
            .all_entity_names()
            .into_iter()
            .collect();
        entities.retain(|entity| admitted.contains(&entity.name));
        entities.truncate(limit);

        Ok(entities)
    }

    /// Link a memory to an entity for graph-augmented recall.
    pub fn link_memory_entity(&self, memory_rid: &str, entity_name: &str) -> Result<()> {
        // Phase 1: Lock conn for SQL INSERT, then drop
        {
            let conn = self.conn.lock();
            conn.execute(
                "INSERT OR IGNORE INTO memory_entities \
                 (memory_rid, entity_name, entity_name_norm) VALUES (?1, ?2, ?3)",
                params![
                    memory_rid,
                    entity_name,
                    crate::engine::thread::normalize_entity_name(entity_name)
                ],
            )?;
            crate::engine::thread::repair_entity_norm(&conn, memory_rid, entity_name)?;
        } // conn dropped

        // Phase 2: Lock graph_index write for in-memory update
        self.graph_index
            .write()
            .link_memory(memory_rid, entity_name);
        Ok(())
    }

    /// Backfill memory_entities for a specific set of entity names.
    /// Used by relate() to retroactively link memories to newly-created entities.
    fn backfill_memory_entities_for(&self, entity_names: &[&str]) -> Result<()> {
        // Phase 1: Lock conn, query candidate memories for each entity, drop conn
        struct LinkCandidate {
            rid: String,
            entity: String,
        }
        let mut candidates = Vec::new();

        {
            let conn = self.conn.lock();
            let mut stmt = conn.prepare_cached(
                "SELECT rid, text FROM memories \
                 WHERE consolidation_status = 'active' \
                 AND rid NOT IN (SELECT memory_rid FROM memory_entities WHERE entity_name = ?1)",
            )?;
            for &entity in entity_names {
                let entity_tokens = crate::graph::tokenize(entity);
                if entity_tokens.is_empty() || !crate::graph::admit_entity(entity) {
                    continue;
                }
                let rows: Vec<(String, String)> = stmt
                    .query_map(params![entity], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;

                // Phase 2: Compute matches (decrypt_text doesn't need conn)
                for (rid, stored_text) in &rows {
                    let text = self
                        .decrypt_text(stored_text)
                        .unwrap_or_else(|_| stored_text.clone());
                    let text_tokens = crate::graph::tokenize(&text);
                    if crate::graph::entity_matches_text(entity, &text_tokens) {
                        candidates.push(LinkCandidate {
                            rid: rid.clone(),
                            entity: entity.to_string(),
                        });
                    }
                }
            }
        } // conn dropped

        if candidates.is_empty() {
            return Ok(());
        }

        // Phase 3: Lock conn, do INSERT OR IGNORE for each link, drop conn
        {
            let conn = self.conn.lock();
            for c in &candidates {
                conn.execute(
                    "INSERT OR IGNORE INTO memory_entities \
                     (memory_rid, entity_name, entity_name_norm) VALUES (?1, ?2, ?3)",
                    params![
                        c.rid,
                        c.entity,
                        crate::engine::thread::normalize_entity_name(&c.entity)
                    ],
                )?;
                crate::engine::thread::repair_entity_norm(&conn, &c.rid, &c.entity)?;
            }
        } // conn dropped

        // Phase 4: Lock graph_index write, do link_memory for each, drop
        {
            let mut gi = self.graph_index.write();
            for c in &candidates {
                gi.link_memory(&c.rid, &c.entity);
            }
        } // graph_index dropped

        Ok(())
    }

    /// Backfill the memory_entities table by scanning memory text for known entity names.
    /// Uses word-boundary matching to avoid false positives.
    /// Returns the number of links created. Idempotent (uses INSERT OR IGNORE).
    pub fn backfill_memory_entities(&self) -> Result<usize> {
        // Phase 1: Lock conn, query entities and memories, drop conn
        let entities: Vec<String>;
        let raw_memories: Vec<(String, String)>;

        {
            let conn = self.conn.lock();
            entities = conn
                .prepare("SELECT name FROM entities")?
                .query_map([], |row| row.get(0))?
                .collect::<std::result::Result<Vec<_>, _>>()?;

            if entities.is_empty() {
                return Ok(0);
            }

            raw_memories = conn
                .prepare("SELECT rid, text FROM memories WHERE consolidation_status = 'active'")?
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<std::result::Result<Vec<_>, _>>()?;
        } // conn dropped

        // Phase 2: Compute matches (decrypt_text doesn't need conn)
        let memories: Vec<(String, String)> = raw_memories
            .into_iter()
            .map(|(rid, stored_text)| {
                let text = self.decrypt_text(&stored_text)?;
                Ok((rid, text))
            })
            .collect::<crate::error::Result<Vec<_>>>()?;

        struct LinkCandidate {
            rid: String,
            entity: String,
        }
        let mut candidates = Vec::new();

        for (rid, text) in &memories {
            let text_tokens = crate::graph::tokenize(text);
            for entity in &entities {
                if crate::graph::entity_matches_text(entity, &text_tokens) {
                    candidates.push(LinkCandidate {
                        rid: rid.clone(),
                        entity: entity.clone(),
                    });
                }
            }
        }

        let count = candidates.len();

        if count == 0 {
            return Ok(0);
        }

        // Phase 3: Lock conn, do INSERT OR IGNORE for each link, drop conn
        {
            let conn = self.conn.lock();
            for c in &candidates {
                conn.execute(
                    "INSERT OR IGNORE INTO memory_entities \
                     (memory_rid, entity_name, entity_name_norm) VALUES (?1, ?2, ?3)",
                    params![
                        c.rid,
                        c.entity,
                        crate::engine::thread::normalize_entity_name(&c.entity)
                    ],
                )?;
                crate::engine::thread::repair_entity_norm(&conn, &c.rid, &c.entity)?;
            }
        } // conn dropped

        // Phase 4: Lock graph_index write, do link_memory for each, drop
        {
            let mut gi = self.graph_index.write();
            for c in &candidates {
                gi.link_memory(&c.rid, &c.entity);
            }
        } // graph_index dropped

        Ok(count)
    }

    // ── RFC 006 Phase 1: Claims + Entity Aliasing ──

    /// Resolve an entity name through the alias table.
    ///
    /// Prefers namespace-specific aliases over the global default namespace.
    /// Returns the canonical name if an alias exists, or the original name if not.
    pub fn resolve_alias(&self, entity: &str, namespace: &str) -> String {
        let conn = self.conn.lock();
        // Try namespace-specific alias first
        let result: Option<String> = conn
            .query_row(
                "SELECT canonical_name FROM entity_aliases WHERE alias = ?1 AND namespace = ?2",
                params![entity, namespace],
                |row| row.get(0),
            )
            .ok();

        if let Some(canonical) = result {
            return canonical;
        }

        // Fall back to global 'default' namespace
        conn.query_row(
            "SELECT canonical_name FROM entity_aliases WHERE alias = ?1 AND namespace = 'default'",
            params![entity],
            |row| row.get(0),
        )
        .unwrap_or_else(|_| entity.to_string())
    }

    /// Register an explicit entity alias.
    pub fn add_entity_alias(
        &self,
        alias: &str,
        canonical_name: &str,
        namespace: &str,
        source: &str,
    ) -> Result<bool> {
        let ts = now();
        let conn = self.conn.lock();
        let changes = conn.execute(
            "INSERT INTO entity_aliases (alias, canonical_name, namespace, source, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5) \
             ON CONFLICT(alias, namespace) DO UPDATE SET canonical_name = ?2, source = ?4, created_at = ?5",
            params![alias, canonical_name, namespace, source, ts],
        )?;
        Ok(changes > 0)
    }

    /// Ingest a structured claim (RFC 006 Phase 1).
    ///
    /// This is the primary write path for claims. It resolves entity aliases,
    /// inserts into the edges table with full qualifier columns, updates the
    /// entity + graph indexes, and logs to the oplog. `relate()` still works
    /// but will be deprecated in v0.7 (Phase 5) in favor of this method.
    #[tracing::instrument(skip(self))]
    pub fn ingest_claim(
        &self,
        src: &str,
        rel_type: &str,
        dst: &str,
        namespace: &str,
        polarity: i32,
        modality: &str,
        valid_from: Option<f64>,
        valid_to: Option<f64>,
        extractor: &str,
        extractor_version: Option<&str>,
        confidence_band: &str,
        source_memory_rid: Option<&str>,
        span_start: Option<i32>,
        span_end: Option<i32>,
        weight: f64,
    ) -> Result<String> {
        let claim_id = crate::id::new_id();
        let ts = now();

        // Resolve aliases before storage
        let src_resolved = self.resolve_alias(src, namespace);
        let dst_resolved = self.resolve_alias(dst, namespace);

        let (src_type, dst_type) =
            crate::graph::classify_with_relationship(&src_resolved, &dst_resolved, rel_type);

        // RFC 008 M3: default regime for claims produced by the ingestion API.
        // A regime-aware variant can accept this as an argument later; 'default'
        // is the only regime written by the public HTTP endpoint today.
        let regime_tag = "default";

        // Phase 1: SQL inserts + write-tier mobility recompute, all under one
        // parking_lot mutex hold so SQLite's single-writer invariant serializes
        // concurrent ingests on the same proposition.
        let proposition_id: String;
        {
            let conn = self.conn.lock();

            // RFC 007 Phase 0: every claim must point at a canonical proposition.
            // Ensure the (src, rel_type, dst, namespace) proposition exists and
            // get its id — create atomically if missing.
            proposition_id =
                ensure_proposition(&conn, &src_resolved, rel_type, &dst_resolved, namespace, ts)?;

            // RFC 006: uniqueness is scoped to (src, dst, rel_type, extractor, polarity, namespace)
            // so multiple sources can make conflicting claims about the same fact. Only an
            // identical resubmission (same extractor + same polarity + same namespace) is
            // treated as an update — typically a validity window refinement.
            conn.execute(
                "INSERT INTO claims (claim_id, src, dst, rel_type, weight, created_at, \
                 polarity, modality, valid_from, valid_to, extractor, extractor_version, \
                 confidence_band, source_memory_rid, span_start, span_end, namespace, \
                 proposition_id, regime_tag) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19) \
                 ON CONFLICT(src, dst, rel_type, extractor, polarity, namespace) DO UPDATE SET \
                 weight = ?5, created_at = ?6, modality = ?8, \
                 valid_from = ?9, valid_to = ?10, extractor_version = ?12, \
                 confidence_band = ?13, source_memory_rid = ?14, span_start = ?15, span_end = ?16, \
                 proposition_id = ?18, regime_tag = ?19",
                params![
                    claim_id, src_resolved, dst_resolved, rel_type, weight, ts,
                    polarity, modality, valid_from, valid_to, extractor, extractor_version,
                    confidence_band, source_memory_rid, span_start, span_end, namespace,
                    proposition_id, regime_tag
                ],
            )?;

            // Ensure entities exist — for endpoints the entity table admits.
            // A value object (`0.19.0`, `1985`) is a legitimate claim OBJECT
            // but never a node: the claim row carries it, the read side
            // refuses it as an anchor, and issue #213 measured what happens
            // when every endpoint becomes a node (11% bare numbers).
            for (entity, etype) in [(&*src_resolved, src_type), (&*dst_resolved, dst_type)] {
                if !crate::graph::admit_entity(entity) {
                    continue;
                }
                conn.execute(
                    "INSERT INTO entities (name, entity_type, first_seen, last_seen) \
                     VALUES (?1, ?2, ?3, ?4) \
                     ON CONFLICT(name) DO UPDATE SET last_seen = ?4, mention_count = mention_count + 1, \
                     entity_type = CASE WHEN entities.entity_type = 'unknown' THEN ?2 ELSE entities.entity_type END",
                    params![entity, etype, ts, ts],
                )?;
            }

            // RFC 008 M3: recompute write-tier mobility for the claim's
            // (proposition_id, regime). Idempotent via content_hash — if the
            // live set hasn't changed (e.g. ON CONFLICT DO UPDATE with the
            // same extractor/polarity), this is a no-op aside from the
            // content_hash comparison.
            //
            // Failures here do NOT abort the claim insert. M3 keeps claim
            // durability as the authoritative commit; if mobility computation
            // fails (malformed lineage, etc.), the state row is skipped and
            // the background reconciler will recompute it. See Saga note 14
            // for the correctness/availability tradeoff discussion.
            if let Err(e) = crate::engine::warrant::compute_write_tier_mobility_conn(
                &conn,
                &proposition_id,
                regime_tag,
            ) {
                tracing::warn!(
                    proposition_id = %proposition_id,
                    regime = regime_tag,
                    error = %e,
                    "mobility recompute failed during claim ingest; reconciler will retry"
                );
            }
            // RFC 008 M4: contest state Γ(c) — grounded diagnostics from the
            // same live claim set. Separate derivation boundary (own version
            // and content_hash) but same snapshot via the shared lock scope.
            // Same failure policy as mobility: log and continue; claim is
            // authoritative.
            if let Err(e) = crate::engine::warrant::compute_contest_state_conn(
                &conn,
                &proposition_id,
                regime_tag,
            ) {
                tracing::warn!(
                    proposition_id = %proposition_id,
                    regime = regime_tag,
                    error = %e,
                    "contest recompute failed during claim ingest; reconciler will retry"
                );
            }
        } // conn dropped

        // Phase 2: graph_index update
        {
            let mut gi = self.graph_index.write();
            gi.add_entity(&src_resolved, src_type);
            gi.add_entity(&dst_resolved, dst_type);
            gi.add_edge(&src_resolved, &dst_resolved, weight as f32);
        }

        // Phase 3: backfill memory_entities for newly-created entities
        self.backfill_memory_entities_for(&[&src_resolved, &dst_resolved])?;

        // Log to oplog as "claim" operation
        self.log_op(
            "claim",
            Some(&claim_id),
            &serde_json::json!({
                "claim_id": claim_id,
                "src": src_resolved,
                "dst": dst_resolved,
                "rel_type": rel_type,
                "weight": weight,
                "polarity": polarity,
                "modality": modality,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "extractor": extractor,
                "confidence_band": confidence_band,
                "source_memory_rid": source_memory_rid,
                "namespace": namespace,
                "created_at": ts,
            }),
            None,
        )?;

        Ok(claim_id)
    }

    /// Get claims (extended edges) for a specific entity, optionally filtered
    /// by namespace. Includes a computed `status_suggestion` field derived at
    /// read time (RFC 006 Phase 2):
    ///
    /// - `active`: positive polarity, no contradictions, no valid_to set
    /// - `superseded`: valid_to is set (a later claim replaced this one)
    /// - `historical`: valid_to is in the past (explicitly time-bounded)
    /// - `conflicted`: an open conflict references this claim
    /// - `negative`: polarity = -1 (negated claim, preserved for provenance)
    pub fn get_claims(
        &self,
        entity: &str,
        namespace: Option<&str>,
    ) -> Result<Vec<serde_json::Value>> {
        let now = now();
        let conn = self.conn.lock();
        let sql = if let Some(ns) = namespace {
            format!(
                "SELECT edge_id, src, dst, rel_type, weight, created_at, \
                 polarity, modality, valid_from, valid_to, extractor, confidence_band, \
                 source_memory_rid, namespace \
                 FROM edges WHERE (src = ?1 OR dst = ?1) AND namespace = '{}' AND tombstoned = 0 \
                 ORDER BY created_at DESC",
                ns.replace('\'', "''")
            )
        } else {
            "SELECT edge_id, src, dst, rel_type, weight, created_at, \
             polarity, modality, valid_from, valid_to, extractor, confidence_band, \
             source_memory_rid, namespace \
             FROM edges WHERE (src = ?1 OR dst = ?1) AND tombstoned = 0 \
             ORDER BY created_at DESC"
                .to_string()
        };

        // Collect open conflict rids for status derivation
        let conflict_rids: std::collections::HashSet<String> = {
            let mut stmt = conn.prepare(
                "SELECT memory_a FROM conflicts WHERE status = 'open' \
                 UNION SELECT memory_b FROM conflicts WHERE status = 'open'",
            )?;
            let rows: Vec<String> = stmt
                .query_map([], |row| row.get::<_, String>(0))?
                .filter_map(|r| r.ok())
                .collect();
            drop(stmt);
            rows.into_iter().collect()
        };

        let mut stmt = conn.prepare(&sql)?;
        let claims = stmt
            .query_map(params![entity], |row| {
                let claim_id: String = row.get(0)?;
                let polarity: i32 = row.get(6)?;
                let valid_to: Option<f64> = row.get(9)?;
                let source_rid: Option<String> = row.get(12)?;

                // Derive status at read time
                let status = if polarity == -1 {
                    "negative"
                } else if let Some(vt) = valid_to {
                    if vt < now {
                        "historical"
                    } else {
                        "superseded"
                    }
                } else if conflict_rids.contains(&claim_id)
                    || source_rid
                        .as_ref()
                        .map_or(false, |r| conflict_rids.contains(r))
                {
                    "conflicted"
                } else {
                    "active"
                };

                Ok(serde_json::json!({
                    "claim_id": claim_id,
                    "src": row.get::<_, String>(1)?,
                    "dst": row.get::<_, String>(2)?,
                    "rel_type": row.get::<_, String>(3)?,
                    "weight": row.get::<_, f64>(4)?,
                    "created_at": row.get::<_, f64>(5)?,
                    "polarity": polarity,
                    "modality": row.get::<_, String>(7)?,
                    "valid_from": row.get::<_, Option<f64>>(8)?,
                    "valid_to": valid_to,
                    "extractor": row.get::<_, String>(10)?,
                    "confidence_band": row.get::<_, String>(11)?,
                    "source_memory_rid": source_rid,
                    "namespace": row.get::<_, String>(13)?,
                    "status_suggestion": status,
                }))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(claims)
    }
}

#[cfg(test)]
mod entity_search_tests {
    use crate::YantrikDB;

    #[test]
    fn search_entities_hides_legacy_rows_suppressed_from_the_active_graph() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        db.relate("Alice", "Acme", "works_with", 1.0).unwrap();

        // Simulate a heading retained from an older extractor. It is newer
        // than Alice and would occupy a LIMIT 1 raw SQL result.
        db.conn()
            .execute(
                "INSERT INTO entities (name, entity_type, first_seen, last_seen, mention_count) \
                 VALUES ('Alice MUST UPDATE MCP CONFIG', 'unknown', 9e18, 9e18, 1)",
                [],
            )
            .unwrap();

        let found = db.search_entities(Some("Alice"), None, 1).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "Alice");
    }

    #[test]
    fn search_entities_accepts_limits_above_the_overfetch_cap() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        db.relate("Alice", "Acme", "works_with", 1.0).unwrap();

        let above_cap = db.search_entities(None, None, 10_001).unwrap();
        assert_eq!(above_cap.len(), 2);

        let maximum = db.search_entities(None, None, usize::MAX).unwrap();
        assert_eq!(maximum.len(), 2);
    }
}

// ── Cooperative claims (2026-09-05) ────────────────────────────────────
//
// The agent that writes a memory is already a language model; the engine
// is not. The heuristic extractor mints relations for a handful of
// sentence shapes (measured 2026-09-05: five of twenty), so most facts a
// writer states never reach the claims table, and the contradiction and
// succession machinery that reads that table stays blind to them.
//
// `attach_claims` lets the WRITER state the claims a memory makes, and
// the engine grounds each one MECHANICALLY before storing it: subject and
// object must occur in the memory's own text (the same whole-token match
// the extractor uses), the relation must be a bounded snake_case token,
// and phantom names are refused with the same predicate the read lanes
// apply. Nothing is inferred; a claim the text does not support is
// rejected with a reason, never silently dropped. Accepted claims carry
// provenance (`source_memory_rid`), so the claims lane, the claim-chain
// traversal, the conflict scanners and entity threads all see them —
// with zero model calls inside the engine.

/// The writer-stated claim extractor label (`claims.extractor`).
pub const STATED_CLAIM_EXTRACTOR: &str = "agent_stated";
/// Claims accepted per `attach_claims` call.
pub const MAX_STATED_CLAIMS: usize = 32;
const MAX_STATED_ENDPOINT_BYTES: usize = 128;
const MAX_STATED_REL_BYTES: usize = 64;

/// A claim the writer states about a memory it recorded.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StatedClaim {
    pub src: String,
    pub rel_type: String,
    pub dst: String,
    /// `1` asserted (default), `-1` denied.
    #[serde(default = "default_polarity")]
    pub polarity: i32,
    /// World-validity start (unix seconds): when the stated fact became
    /// true, not when it was written. Absent, the memory's own event time
    /// (`event_time_min`, the temporal tag on the record) is used; absent
    /// that too, the claim carries no window and only write order remains.
    /// The conflict scanner treats non-overlapping windows as succession,
    /// not contradiction (RFC 006), so this is what aligns a stated
    /// relation with the time it held.
    #[serde(default)]
    pub valid_from: Option<f64>,
    /// World-validity end; `None` means still true.
    #[serde(default)]
    pub valid_to: Option<f64>,
}

impl Default for StatedClaim {
    fn default() -> Self {
        Self {
            src: String::new(),
            rel_type: String::new(),
            dst: String::new(),
            polarity: 1,
            valid_from: None,
            valid_to: None,
        }
    }
}

fn default_polarity() -> i32 {
    1
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct AcceptedClaim {
    pub claim_id: String,
    pub src: String,
    pub rel_type: String,
    pub dst: String,
    pub polarity: i32,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct RejectedClaim {
    pub src: String,
    pub rel_type: String,
    pub dst: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct AttachClaimsReport {
    pub memory_rid: String,
    pub accepted: Vec<AcceptedClaim>,
    pub rejected: Vec<RejectedClaim>,
}

/// Canonical relation token: lowercase, `[a-z0-9_]`, separators folded
/// to `_`, no leading/trailing/double underscores. `None` when nothing
/// admissible survives.
pub(crate) fn normalize_relation(rel: &str) -> Option<String> {
    let mut out = String::with_capacity(rel.len());
    let mut last_sep = true;
    for c in rel.trim().chars() {
        let c = c.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
            last_sep = false;
        } else if !last_sep {
            out.push('_');
            last_sep = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.is_empty()
        || out.len() > MAX_STATED_REL_BYTES
        || !out.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
    {
        return None;
    }
    Some(out)
}

impl super::YantrikDB {
    /// Ground and store the claims a writer states about `memory_rid`.
    /// See the module note above. Errors only on an unknown or inactive
    /// memory or an oversized batch; per-claim problems are REPORTED in
    /// `rejected`, never raised.
    /// The store's own lexicon: how it has written `token` (lowercased) so
    /// far. `None` on encrypted stores (never populated) and for unseen tokens.
    pub(crate) fn token_case_stats(
        conn: &rusqlite::Connection,
        token: &str,
    ) -> Option<crate::graph::CaseStats> {
        conn.query_row(
            "SELECT lower_n, cap_mid_n, cap_start_n FROM token_case_stats WHERE token = ?1",
            params![token],
            |r| {
                Ok(crate::graph::CaseStats {
                    lower_n: r.get(0)?,
                    cap_mid_n: r.get(1)?,
                    cap_start_n: r.get(2)?,
                })
            },
        )
        .ok()
    }

    /// Fold one memory's case observations into `token_case_stats` (at most
    /// one count per token per class per memory). Plaintext tokens, so an
    /// encrypted store learns nothing — same rule as relation templates.
    pub(crate) fn record_token_case_observations(
        &self,
        conn: &rusqlite::Connection,
        text: &str,
    ) -> Result<()> {
        if self.is_encrypted() {
            return Ok(());
        }
        let mut stmt = conn.prepare_cached(
            "INSERT INTO token_case_stats (token, lower_n, cap_mid_n, cap_start_n) \
             VALUES (?1, ?2, ?3, ?4) \
             ON CONFLICT(token) DO UPDATE SET lower_n = lower_n + ?2, \
             cap_mid_n = cap_mid_n + ?3, cap_start_n = cap_start_n + ?4",
        )?;
        for (token, class) in crate::graph::token_case_observations(text) {
            let (l, m, st) = match class {
                crate::graph::TokenCase::Lower => (1, 0, 0),
                crate::graph::TokenCase::CapMid => (0, 1, 0),
                crate::graph::TokenCase::CapStart => (0, 0, 1),
            };
            stmt.execute(params![token, l, m, st])?;
        }
        Ok(())
    }

    /// Recompute `token_case_stats` from every active memory (the heal's
    /// first step, and the backfill for stores that predate v52).
    pub(crate) fn rebuild_token_case_stats(&self) -> Result<usize> {
        let conn = self.conn();
        conn.execute("DELETE FROM token_case_stats", [])?;
        if self.is_encrypted() {
            return Ok(0);
        }
        let texts: Vec<String> = {
            let mut stmt =
                conn.prepare("SELECT text FROM memories WHERE consolidation_status = 'active'")?;
            let rows: Vec<String> = stmt
                .query_map([], |r| r.get::<_, String>(0))?
                .collect::<std::result::Result<_, _>>()?;
            rows
        };
        let mut n = 0usize;
        for stored in &texts {
            let text = self.decrypt_text(stored).unwrap_or_else(|_| stored.clone());
            self.record_token_case_observations(&conn, &text)?;
            n += 1;
        }
        Ok(n)
    }

    /// The extractor every engine writer uses: heuristic entities admitted
    /// against this store's own lexicon.
    pub(crate) fn extract_entities_for(&self, text: &str) -> Vec<String> {
        let conn = self.conn();
        crate::graph::extract_heuristic_entities_with(text, |tok| {
            Self::token_case_stats(&conn, tok)
        })
    }

    /// The record's temporal tag (`event_time_min`, set from metadata
    /// `event_time_min`/`event_time_max` at write time), if any.
    pub(crate) fn memory_event_time_min(&self, rid: &str) -> Option<f64> {
        let conn = self.conn();
        conn.query_row(
            "SELECT event_time_min FROM memories WHERE rid = ?1",
            params![rid],
            |r| r.get::<_, Option<f64>>(0),
        )
        .ok()
        .flatten()
    }

    pub fn attach_claims(
        &self,
        memory_rid: &str,
        claims: &[StatedClaim],
    ) -> Result<AttachClaimsReport> {
        use crate::error::YantrikDbError;
        if claims.len() > MAX_STATED_CLAIMS {
            return Err(YantrikDbError::InvalidInput(format!(
                "attach_claims: {} claims exceed the cap of {MAX_STATED_CLAIMS}",
                claims.len()
            )));
        }
        let memory = self
            .get(memory_rid)?
            .ok_or_else(|| YantrikDbError::NotFound(format!("memory {memory_rid}")))?;
        if memory.consolidation_status != "active" {
            return Err(YantrikDbError::InvalidInput(format!(
                "attach_claims: memory {memory_rid} is {}, not active",
                memory.consolidation_status
            )));
        }
        let text_tokens = crate::graph::tokenize(&memory.text);
        // The record's temporal tag: a claim stated without its own window
        // inherits the time the memory is ABOUT, so relation and time stay
        // aligned without the writer repeating the date per claim.
        let event_min = self.memory_event_time_min(memory_rid);
        let mut report = AttachClaimsReport {
            memory_rid: memory_rid.to_string(),
            accepted: Vec::new(),
            rejected: Vec::new(),
        };
        let mut seen_in_batch: std::collections::HashSet<(String, String, String, i32)> =
            std::collections::HashSet::new();

        for claim in claims {
            let src = claim.src.trim();
            let dst = claim.dst.trim();
            let reject = |reason: String, report: &mut AttachClaimsReport| {
                report.rejected.push(RejectedClaim {
                    src: src.to_string(),
                    rel_type: claim.rel_type.clone(),
                    dst: dst.to_string(),
                    reason,
                });
            };
            let Some(rel) = normalize_relation(&claim.rel_type) else {
                reject(
                    "relation must normalize to a snake_case token of at most 64 bytes starting with a letter".into(),
                    &mut report,
                );
                continue;
            };
            if claim.polarity != 1 && claim.polarity != -1 {
                reject(
                    "polarity must be 1 (asserted) or -1 (denied)".into(),
                    &mut report,
                );
                continue;
            }
            if src.is_empty() || dst.is_empty() {
                reject("subject and object must be non-empty".into(), &mut report);
                continue;
            }
            if src.len() > MAX_STATED_ENDPOINT_BYTES || dst.len() > MAX_STATED_ENDPOINT_BYTES {
                reject(
                    format!("subject and object must be at most {MAX_STATED_ENDPOINT_BYTES} bytes"),
                    &mut report,
                );
                continue;
            }
            // The subject must be a node the table admits; the object may
            // also be a value (`0.19.0`, `1985`): `CT128 runs 0.19.0` is the
            // succession-bearing claim shape and its object is not a thing.
            if !crate::graph::admit_entity(src) {
                reject("subject is not an admissible entity name (stopword, heading, possessive or bare number)".into(), &mut report);
                continue;
            }
            if !crate::graph::admit_entity(dst) && !crate::graph::is_value_object(dst) {
                reject("object is neither an admissible entity name nor a value (a number, version or year)".into(), &mut report);
                continue;
            }
            if src.eq_ignore_ascii_case(dst) {
                reject("subject and object are the same entity".into(), &mut report);
                continue;
            }
            // GROUNDING — the whole point. Same predicate the extractor and
            // the entity backfill use; whole tokens, case-insensitive.
            let mut missing = Vec::new();
            if !crate::graph::entity_matches_text(src, &text_tokens) {
                missing.push(format!("subject {src:?}"));
            }
            if !crate::graph::entity_matches_text(dst, &text_tokens) {
                missing.push(format!("object {dst:?}"));
            }
            if !missing.is_empty() {
                reject(
                    format!(
                        "not grounded: {} does not occur in the memory text",
                        missing.join(" and ")
                    ),
                    &mut report,
                );
                continue;
            }
            if !seen_in_batch.insert((
                src.to_lowercase(),
                rel.clone(),
                dst.to_lowercase(),
                claim.polarity,
            )) {
                reject(
                    "duplicate of an earlier claim in this batch".into(),
                    &mut report,
                );
                continue;
            }
            let claim_id = self.ingest_claim(
                src,
                &rel,
                dst,
                &memory.namespace,
                claim.polarity,
                "asserted",
                claim.valid_from.or(event_min),
                claim.valid_to,
                STATED_CLAIM_EXTRACTOR,
                Some("1.0"),
                "high",
                Some(memory_rid),
                None,
                None,
                1.0,
            )?;
            // The memory now carries its endpoints as entities SYNCHRONOUSLY —
            // a stated claim closes the materializer's read-your-writes gap
            // for exactly the entities the writer cared about.
            self.link_memory_entity(memory_rid, src)?;
            self.link_memory_entity(memory_rid, dst)?;
            // Self-mined templates: the phrase between subject and object is
            // a candidate template for this relation (see the note above
            // `mine_relation_template`). Best-effort: a mining failure must
            // never fail the claim it learned from.
            if let Err(e) =
                self.mine_relation_template(&memory.text, src, &rel, dst, &memory.namespace)
            {
                tracing::warn!(error = %e, "relation template mining failed; claim kept");
            }
            report.accepted.push(AcceptedClaim {
                claim_id,
                src: src.to_string(),
                rel_type: rel,
                dst: dst.to_string(),
                polarity: claim.polarity,
            });
        }
        Ok(report)
    }
}

#[cfg(test)]
mod stated_claim_tests {
    use super::normalize_relation;

    #[test]
    fn relation_normalization() {
        assert_eq!(normalize_relation("prefers"), Some("prefers".into()));
        assert_eq!(normalize_relation(" Works At "), Some("works_at".into()));
        assert_eq!(normalize_relation("lives-in"), Some("lives_in".into()));
        assert_eq!(
            normalize_relation("is the CEO of"),
            Some("is_the_ceo_of".into())
        );
        assert_eq!(normalize_relation("__"), None);
        assert_eq!(normalize_relation("1st"), None);
        assert_eq!(normalize_relation(&"x".repeat(65)), None);
    }
}

// ── Self-mined relation templates (2026-09-05) ─────────────────────────
//
// The built-in extractor knows a fixed phrase table. Every grounded claim
// a writer states is also a labelled example of how THIS store's authors
// phrase that relation: "Dana mentors Priya" + claim (Dana, mentors,
// Priya) says the phrase "mentors" carries `mentors`. After two distinct
// (src, dst) pairs agree on a phrase, the phrase is promoted to an active
// template and the materializer applies it to plain writes — the
// extractor grows with usage, per namespace, with no model anywhere.
//
// Precision rules (each one closes a measured failure class):
// - subject must precede object within 150 chars, the extractor's window;
// - the phrase is 1..=6 whole tokens after negation stripping;
// - at least one token is >= 3 chars and not a function word, so "is a"
//   can never become a template;
// - a phrase is promoted only by DISTINCT pairs — restating one fact never
//   promotes;
// - never mined on encrypted stores (the phrase is plaintext).

/// Claims minted by learned templates carry this extractor label.
pub const LEARNED_CLAIM_EXTRACTOR: &str = "learned_v1";
/// Distinct (src, dst) pairs required before a phrase becomes a template.
pub const TEMPLATE_PROMOTION_PAIRS: i64 = 2;
/// Active templates loaded per namespace at materialization.
pub const MAX_ACTIVE_TEMPLATES: usize = 256;
const MAX_TEMPLATE_TOKENS: usize = 6;
const TEMPLATE_FUNCTION_WORDS: &[&str] = &[
    "is", "a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or", "with", "as", "his",
    "her", "their", "its", "was", "were", "be", "been", "has", "have", "had", "by", "from", "that",
    "this", "it", "are", "am", "now", "also", "just", "very", "still", "not", "no", "never",
];

/// A learned template as reported by [`YantrikDB::learned_relation_patterns`].
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct LearnedRelationPattern {
    pub namespace: String,
    pub rel_type: String,
    pub phrase: String,
    pub pair_count: i64,
    pub active: bool,
    pub first_seen: f64,
    pub last_seen: f64,
}

/// The between-window phrase a grounded claim teaches, or `None` when the
/// text offers no admissible one. Pure; unit-tested below.
pub(crate) fn template_phrase(text: &str, src: &str, dst: &str) -> Option<String> {
    let lower = text.to_lowercase();
    let src_l = src.to_lowercase();
    let dst_l = dst.to_lowercase();
    let pos_a = lower.find(&src_l)?;
    let start = pos_a + src_l.len();
    let rel = lower.get(start..)?.find(&dst_l)?;
    let pos_b = start + rel;
    if pos_b - pos_a > 150 {
        return None;
    }
    let between = lower.get(start..pos_b)?;
    let tokens: Vec<&str> = between
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|t| !t.is_empty())
        .filter(|t| !crate::graph::negation_cue(t))
        .collect();
    if tokens.is_empty() || tokens.len() > MAX_TEMPLATE_TOKENS {
        return None;
    }
    if !tokens
        .iter()
        .any(|t| t.chars().count() >= 3 && !TEMPLATE_FUNCTION_WORDS.contains(t))
    {
        return None;
    }
    Some(tokens.join(" "))
}

impl super::YantrikDB {
    fn mine_relation_template(
        &self,
        text: &str,
        src: &str,
        rel_type: &str,
        dst: &str,
        namespace: &str,
    ) -> Result<()> {
        if self.is_encrypted() {
            return Ok(());
        }
        let Some(phrase) = template_phrase(text, src, dst) else {
            return Ok(());
        };
        let ts = now();
        let conn = self.conn();
        conn.execute(
            "INSERT OR IGNORE INTO learned_relation_pattern_support \
             (namespace, rel_type, phrase, src_norm, dst_norm) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                namespace,
                rel_type,
                phrase,
                src.to_lowercase(),
                dst.to_lowercase()
            ],
        )?;
        let pairs: i64 = conn.query_row(
            "SELECT COUNT(*) FROM learned_relation_pattern_support \
             WHERE namespace = ?1 AND rel_type = ?2 AND phrase = ?3",
            params![namespace, rel_type, phrase],
            |r| r.get(0),
        )?;
        let active = i64::from(pairs >= TEMPLATE_PROMOTION_PAIRS);
        conn.execute(
            "INSERT INTO learned_relation_patterns \
             (namespace, rel_type, phrase, pair_count, active, first_seen, last_seen) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6) \
             ON CONFLICT(namespace, rel_type, phrase) DO UPDATE SET \
             pair_count = ?4, active = ?5, last_seen = ?6",
            params![namespace, rel_type, phrase, pairs, active, ts],
        )?;
        Ok(())
    }

    /// Active templates for a namespace, strongest first, bounded — what
    /// the materializer applies to a plain write.
    pub(crate) fn active_relation_templates(
        conn: &rusqlite::Connection,
        namespace: &str,
    ) -> Vec<(String, String)> {
        let Ok(mut stmt) = conn.prepare_cached(
            "SELECT phrase, rel_type FROM learned_relation_patterns \
             WHERE namespace = ?1 AND active = 1 \
             ORDER BY pair_count DESC, phrase ASC LIMIT ?2",
        ) else {
            return Vec::new(); // pre-v51 store mid-migration: no templates
        };
        stmt.query_map(params![namespace, MAX_ACTIVE_TEMPLATES as i64], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })
        .map(|rows| rows.filter_map(|x| x.ok()).collect())
        .unwrap_or_default()
    }

    /// Every learned template (active or still gathering support) in a
    /// namespace — the audit surface for what the store has taught itself.
    pub fn learned_relation_patterns(
        &self,
        namespace: &str,
    ) -> Result<Vec<LearnedRelationPattern>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT namespace, rel_type, phrase, pair_count, active, first_seen, last_seen \
             FROM learned_relation_patterns WHERE namespace = ?1 \
             ORDER BY active DESC, pair_count DESC, rel_type ASC, phrase ASC",
        )?;
        let rows = stmt
            .query_map(params![namespace], |r| {
                Ok(LearnedRelationPattern {
                    namespace: r.get(0)?,
                    rel_type: r.get(1)?,
                    phrase: r.get(2)?,
                    pair_count: r.get(3)?,
                    active: r.get::<_, i64>(4)? != 0,
                    first_seen: r.get(5)?,
                    last_seen: r.get(6)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Forget every learned template in a namespace (and its support).
    /// Claims already minted by those templates stay — they are ordinary
    /// claims with `extractor = 'learned_v1'` and their own provenance.
    /// Returns the number of templates removed.
    pub fn forget_learned_relation_patterns(&self, namespace: &str) -> Result<usize> {
        let conn = self.conn();
        conn.execute(
            "DELETE FROM learned_relation_pattern_support WHERE namespace = ?1",
            params![namespace],
        )?;
        let n = conn.execute(
            "DELETE FROM learned_relation_patterns WHERE namespace = ?1",
            params![namespace],
        )?;
        Ok(n)
    }
}

#[cfg(test)]
mod template_phrase_tests {
    use super::template_phrase;

    #[test]
    fn phrase_between_subject_and_object() {
        assert_eq!(
            template_phrase("Dana mentors Priya at Acme.", "Dana", "Priya"),
            Some("mentors".into())
        );
        assert_eq!(
            template_phrase(
                "Alice Moreau is the lead reviewer for Fennwick Labs.",
                "Alice Moreau",
                "Fennwick Labs"
            ),
            Some("is the lead reviewer for".into())
        );
        // Negation cues are stripped so the phrase matches the positive form.
        assert_eq!(
            template_phrase("Dana does not mentor Priya.", "Dana", "Priya"),
            Some("does mentor".into())
        );
    }

    #[test]
    fn function_word_windows_and_bad_shapes_are_refused() {
        assert_eq!(
            template_phrase("Pranab is a Vim user.", "Pranab", "Vim"),
            None
        );
        assert_eq!(
            template_phrase("Priya was mentored by Dana.", "Dana", "Priya"),
            None,
            "object before subject"
        );
        assert_eq!(
            template_phrase("Dana Priya", "Dana", "Priya"),
            None,
            "empty window"
        );
        let long = format!("Dana {} Priya", "word ".repeat(7));
        assert_eq!(
            template_phrase(&long, "Dana", "Priya"),
            None,
            "more than six tokens"
        );
        let far = format!("Dana {} Priya", "x".repeat(200));
        assert_eq!(
            template_phrase(&far, "Dana", "Priya"),
            None,
            "outside the 150-char window"
        );
    }
}
