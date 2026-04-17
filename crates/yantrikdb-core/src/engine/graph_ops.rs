use rusqlite::params;

use crate::error::Result;
use crate::types::{Edge, Entity};

use super::{now, YantrikDB};

impl YantrikDB {
    /// Create or update a relationship between entities.
    #[tracing::instrument(skip(self))]
    pub fn relate(
        &self,
        src: &str,
        dst: &str,
        rel_type: &str,
        weight: f64,
    ) -> Result<String> {
        let edge_id = crate::id::new_id();
        let ts = now();

        // Classify entity types using relationship semantics
        let (src_type, dst_type) =
            crate::graph::classify_with_relationship(src, dst, rel_type);

        // Phase 1: Lock conn for all SQL operations, then drop
        {
            let conn = self.conn.lock();
            conn.execute(
                "INSERT INTO claims (claim_id, src, dst, rel_type, weight, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
                 ON CONFLICT(src, dst, rel_type) DO UPDATE SET weight = ?5, created_at = ?6",
                params![edge_id, src, dst, rel_type, weight, ts],
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
            }),
            None,
        )?;

        Ok(edge_id)
    }

    /// Get all edges connected to an entity.
    pub fn get_edges(&self, entity: &str) -> Result<Vec<Edge>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT * FROM edges WHERE (src = ?1 OR dst = ?1) AND tombstoned = 0",
        )?;

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
        let (sql, params_vec): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = match (pattern, entity_type) {
            (Some(p), Some(t)) => (
                "SELECT name, entity_type, first_seen, last_seen, mention_count \
                 FROM entities WHERE name LIKE ?1 AND entity_type = ?2 \
                 ORDER BY last_seen DESC LIMIT ?3".to_string(),
                vec![
                    Box::new(format!("%{}%", p)) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(t.to_string()),
                    Box::new(limit as i64),
                ],
            ),
            (Some(p), None) => (
                "SELECT name, entity_type, first_seen, last_seen, mention_count \
                 FROM entities WHERE name LIKE ?1 \
                 ORDER BY last_seen DESC LIMIT ?2".to_string(),
                vec![
                    Box::new(format!("%{}%", p)) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(limit as i64),
                ],
            ),
            (None, Some(t)) => (
                "SELECT name, entity_type, first_seen, last_seen, mention_count \
                 FROM entities WHERE entity_type = ?1 \
                 ORDER BY last_seen DESC LIMIT ?2".to_string(),
                vec![
                    Box::new(t.to_string()) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(limit as i64),
                ],
            ),
            (None, None) => (
                "SELECT name, entity_type, first_seen, last_seen, mention_count \
                 FROM entities ORDER BY last_seen DESC LIMIT ?1".to_string(),
                vec![Box::new(limit as i64) as Box<dyn rusqlite::types::ToSql>],
            ),
        };

        let conn = self.conn.lock();
        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();
        let entities = stmt
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

        Ok(entities)
    }

    /// Link a memory to an entity for graph-augmented recall.
    pub fn link_memory_entity(&self, memory_rid: &str, entity_name: &str) -> Result<()> {
        // Phase 1: Lock conn for SQL INSERT, then drop
        {
            let conn = self.conn.lock();
            conn.execute(
                "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name) VALUES (?1, ?2)",
                params![memory_rid, entity_name],
            )?;
        } // conn dropped

        // Phase 2: Lock graph_index write for in-memory update
        self.graph_index.write().link_memory(memory_rid, entity_name);
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
                 AND rid NOT IN (SELECT memory_rid FROM memory_entities WHERE entity_name = ?1)"
            )?;
            for &entity in entity_names {
                let entity_tokens = crate::graph::tokenize(entity);
                if entity_tokens.is_empty() {
                    continue;
                }
                let rows: Vec<(String, String)> = stmt
                    .query_map(params![entity], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;

                // Phase 2: Compute matches (decrypt_text doesn't need conn)
                for (rid, stored_text) in &rows {
                    let text = self.decrypt_text(stored_text).unwrap_or_else(|_| stored_text.clone());
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
                    "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name) VALUES (?1, ?2)",
                    params![c.rid, c.entity],
                )?;
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
            entities = conn.prepare(
                "SELECT name FROM entities",
            )?.query_map([], |row| row.get(0))?.collect::<std::result::Result<Vec<_>, _>>()?;

            if entities.is_empty() {
                return Ok(0);
            }

            raw_memories = conn.prepare(
                "SELECT rid, text FROM memories WHERE consolidation_status = 'active'",
            )?.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?.collect::<std::result::Result<Vec<_>, _>>()?;
        } // conn dropped

        // Phase 2: Compute matches (decrypt_text doesn't need conn)
        let memories: Vec<(String, String)> = raw_memories.into_iter()
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
                    "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name) VALUES (?1, ?2)",
                    params![c.rid, c.entity],
                )?;
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

        // Phase 1: SQL inserts (conn locked)
        {
            let conn = self.conn.lock();
            conn.execute(
                "INSERT INTO claims (claim_id, src, dst, rel_type, weight, created_at, \
                 polarity, modality, valid_from, valid_to, extractor, extractor_version, \
                 confidence_band, source_memory_rid, span_start, span_end, namespace) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17) \
                 ON CONFLICT(src, dst, rel_type) DO UPDATE SET \
                 weight = ?5, created_at = ?6, polarity = ?7, modality = ?8, \
                 valid_from = ?9, valid_to = ?10, extractor = ?11, extractor_version = ?12, \
                 confidence_band = ?13, source_memory_rid = ?14, span_start = ?15, span_end = ?16, \
                 namespace = ?17",
                params![
                    claim_id, src_resolved, dst_resolved, rel_type, weight, ts,
                    polarity, modality, valid_from, valid_to, extractor, extractor_version,
                    confidence_band, source_memory_rid, span_start, span_end, namespace
                ],
            )?;

            // Ensure entities exist
            for (entity, etype) in [(&*src_resolved, src_type), (&*dst_resolved, dst_type)] {
                conn.execute(
                    "INSERT INTO entities (name, entity_type, first_seen, last_seen) \
                     VALUES (?1, ?2, ?3, ?4) \
                     ON CONFLICT(name) DO UPDATE SET last_seen = ?4, mention_count = mention_count + 1, \
                     entity_type = CASE WHEN entities.entity_type = 'unknown' THEN ?2 ELSE entities.entity_type END",
                    params![entity, etype, ts, ts],
                )?;
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
                 UNION SELECT memory_b FROM conflicts WHERE status = 'open'"
            )?;
            let rows: Vec<String> = stmt.query_map([], |row| row.get::<_, String>(0))?
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
                    if vt < now { "historical" } else { "superseded" }
                } else if conflict_rids.contains(&claim_id)
                    || source_rid.as_ref().map_or(false, |r| conflict_rids.contains(r))
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
