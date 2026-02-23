use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};

use crate::error::{AidbError, Result};
use crate::schema::{SCHEMA_SQL, SCHEMA_VERSION};
use crate::scoring;
use crate::serde_helpers::serialize_f32;
use crate::types::*;

/// The AIDB cognitive memory engine.
pub struct AIDB {
    conn: Connection,
    embedding_dim: usize,
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

impl AIDB {
    /// Create a new AIDB instance.
    pub fn new(db_path: &str, embedding_dim: usize) -> Result<Self> {
        // Register sqlite-vec as auto-extension before opening any connection
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }

        let conn = Connection::open(db_path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        conn.execute_batch(SCHEMA_SQL)?;

        // Create virtual table for vector search
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories \
             USING vec0(rid TEXT PRIMARY KEY, embedding float[{embedding_dim}])"
        ))?;

        // Set schema version
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?1)",
            params![SCHEMA_VERSION.to_string()],
        )?;

        Ok(Self {
            conn,
            embedding_dim,
        })
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get a reference to the underlying connection (for test compatibility).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Get a mutable reference to the underlying connection.
    pub fn conn_mut(&mut self) -> &mut Connection {
        &mut self.conn
    }

    // ── record() — store a memory ──

    /// Store a new memory and return its RID.
    pub fn record(
        &self,
        text: &str,
        memory_type: &str,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: &serde_json::Value,
        embedding: &[f32],
    ) -> Result<String> {
        let rid = uuid7::uuid7().to_string();
        let ts = now();
        let emb_blob = serialize_f32(embedding);
        let meta_str = serde_json::to_string(metadata)?;

        self.conn.execute(
            "INSERT INTO memories \
             (rid, type, text, embedding, created_at, updated_at, importance, \
              half_life, last_access, valence, metadata) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![rid, memory_type, text, emb_blob, ts, ts, importance, half_life, ts, valence, meta_str],
        )?;

        // Insert into vector index
        self.conn.execute(
            "INSERT INTO vec_memories (rid, embedding) VALUES (?1, ?2)",
            params![rid, emb_blob],
        )?;

        self.log_op("record", Some(&rid), &serde_json::json!({
            "type": memory_type,
            "text": text,
            "importance": importance,
            "valence": valence,
        }))?;

        Ok(rid)
    }

    // ── recall() — multi-signal retrieval ──

    /// Retrieve memories using multi-signal fusion scoring.
    pub fn recall(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
    ) -> Result<Vec<RecallResult>> {
        let ts = now();
        let emb_blob = serialize_f32(query_embedding);

        // Step 1: Vector candidate generation
        let fetch_k = (top_k * 5).min(200);
        let mut stmt = self.conn.prepare(
            "SELECT rid, distance FROM vec_memories \
             WHERE embedding MATCH ?1 ORDER BY distance LIMIT ?2",
        )?;

        let vec_results: Vec<(String, f64)> = stmt
            .query_map(params![emb_blob, fetch_k as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        if vec_results.is_empty() {
            return Ok(vec![]);
        }

        let rids: Vec<&str> = vec_results.iter().map(|(r, _)| r.as_str()).collect();
        let vec_scores: std::collections::HashMap<&str, f64> = vec_results
            .iter()
            .map(|(r, d)| (r.as_str(), 1.0 - d))
            .collect();

        // Step 2: Fetch full memory records with filtering
        let statuses: Vec<&str> = if include_consolidated {
            vec!["active", "consolidated"]
        } else {
            vec!["active"]
        };

        let rid_placeholders: String = (0..rids.len()).map(|i| format!("?{}", i + 1)).collect::<Vec<_>>().join(",");
        let status_offset = rids.len() + 1;
        let status_placeholders: String = (0..statuses.len())
            .map(|i| format!("?{}", status_offset + i))
            .collect::<Vec<_>>()
            .join(",");

        let mut sql = format!(
            "SELECT * FROM memories WHERE rid IN ({rid_placeholders}) \
             AND consolidation_status IN ({status_placeholders})"
        );

        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        for r in &rids {
            param_values.push(Box::new(r.to_string()));
        }
        for s in &statuses {
            param_values.push(Box::new(s.to_string()));
        }

        if let Some((start, end)) = time_window {
            let n = param_values.len();
            sql.push_str(&format!(" AND created_at BETWEEN ?{} AND ?{}", n + 1, n + 2));
            param_values.push(Box::new(start));
            param_values.push(Box::new(end));
        }

        if let Some(mt) = memory_type {
            let n = param_values.len();
            sql.push_str(&format!(" AND type = ?{}", n + 1));
            param_values.push(Box::new(mt.to_string()));
        }

        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let mut stmt = self.conn.prepare(&sql)?;
        let memories: Vec<_> = stmt
            .query_map(params_ref.as_slice(), |row| {
                Ok(MemoryRow {
                    rid: row.get("rid")?,
                    memory_type: row.get("type")?,
                    text: row.get("text")?,
                    created_at: row.get("created_at")?,
                    importance: row.get("importance")?,
                    valence: row.get("valence")?,
                    half_life: row.get("half_life")?,
                    last_access: row.get("last_access")?,
                    metadata: row.get("metadata")?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // Step 3: Score with multi-signal fusion
        let mut scored: Vec<RecallResult> = Vec::new();
        for mem in &memories {
            let sim_score = *vec_scores.get(mem.rid.as_str()).unwrap_or(&0.0);

            let elapsed = ts - mem.last_access;
            let decay = scoring::decay_score(mem.importance, mem.half_life, elapsed);

            let age = ts - mem.created_at;
            let recency = scoring::recency_score(age);

            let composite = scoring::composite_score(
                sim_score,
                decay,
                recency,
                mem.importance,
                mem.valence,
            );

            let why = scoring::build_why(sim_score, recency, decay, mem.valence);

            let metadata: serde_json::Value =
                serde_json::from_str(&mem.metadata).unwrap_or(serde_json::Value::Object(Default::default()));

            scored.push(RecallResult {
                rid: mem.rid.clone(),
                memory_type: mem.memory_type.clone(),
                text: mem.text.clone(),
                created_at: mem.created_at,
                importance: mem.importance,
                valence: mem.valence,
                score: composite,
                scores: ScoreBreakdown {
                    similarity: sim_score,
                    decay,
                    recency,
                    importance: mem.importance,
                },
                why_retrieved: why,
                metadata,
            });
        }

        // Step 4: Sort and return top_k
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        // Reinforce accessed memories (spaced repetition)
        for r in &scored {
            self.reinforce(&r.rid)?;
        }

        Ok(scored)
    }

    /// Reinforce a memory on access — increase half_life and update last_access.
    fn reinforce(&self, rid: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE memories SET last_access = ?1, half_life = MIN(half_life * 1.2, 31536000.0) WHERE rid = ?2",
            params![now(), rid],
        )?;
        Ok(())
    }

    // ── relate() — create entity links ──

    /// Create or update a relationship between entities.
    pub fn relate(
        &self,
        src: &str,
        dst: &str,
        rel_type: &str,
        weight: f64,
    ) -> Result<String> {
        let edge_id = uuid7::uuid7().to_string();
        let ts = now();

        self.conn.execute(
            "INSERT INTO edges (edge_id, src, dst, rel_type, weight, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
             ON CONFLICT(src, dst, rel_type) DO UPDATE SET weight = ?5, created_at = ?6",
            params![edge_id, src, dst, rel_type, weight, ts],
        )?;

        // Ensure entities exist
        for entity in [src, dst] {
            self.conn.execute(
                "INSERT INTO entities (name, first_seen, last_seen) \
                 VALUES (?1, ?2, ?3) \
                 ON CONFLICT(name) DO UPDATE SET last_seen = ?3, mention_count = mention_count + 1",
                params![entity, ts, ts],
            )?;
        }

        self.log_op("relate", None, &serde_json::json!({
            "src": src, "dst": dst, "rel_type": rel_type, "weight": weight,
        }))?;

        Ok(edge_id)
    }

    // ── decay() — compute current importance scores ──

    /// Find memories that have decayed below a threshold.
    pub fn decay(&self, threshold: f64) -> Result<Vec<DecayedMemory>> {
        let ts = now();
        let mut stmt = self.conn.prepare(
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
            let (rid, text, importance, half_life, last_access, mem_type) = row?;
            let elapsed = ts - last_access;
            let score = scoring::decay_score(importance, half_life, elapsed);
            if score < threshold {
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

    // ── forget() — tombstone a memory ──

    /// Tombstone a memory. Returns true if the memory was found and tombstoned.
    pub fn forget(&self, rid: &str) -> Result<bool> {
        let changes = self.conn.execute(
            "UPDATE memories SET consolidation_status = 'tombstoned', updated_at = ?1 WHERE rid = ?2",
            params![now(), rid],
        )?;

        if changes > 0 {
            self.conn.execute(
                "DELETE FROM vec_memories WHERE rid = ?1",
                params![rid],
            )?;
            self.log_op("forget", Some(rid), &serde_json::json!({}))?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    // ── Utility methods ──

    /// Get a single memory by RID.
    pub fn get(&self, rid: &str) -> Result<Option<Memory>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM memories WHERE rid = ?1",
        )?;

        let result = stmt.query_row(params![rid], |row| {
            let meta_str: String = row.get("metadata")?;
            let metadata: serde_json::Value =
                serde_json::from_str(&meta_str).unwrap_or(serde_json::Value::Object(Default::default()));

            Ok(Memory {
                rid: row.get("rid")?,
                memory_type: row.get("type")?,
                text: row.get("text")?,
                created_at: row.get("created_at")?,
                importance: row.get("importance")?,
                valence: row.get("valence")?,
                half_life: row.get("half_life")?,
                last_access: row.get("last_access")?,
                consolidation_status: row.get("consolidation_status")?,
                consolidated_into: row.get("consolidated_into")?,
                metadata,
            })
        });

        match result {
            Ok(mem) => Ok(Some(mem)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Get all edges connected to an entity.
    pub fn get_edges(&self, entity: &str) -> Result<Vec<Edge>> {
        let mut stmt = self.conn.prepare(
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

    /// Get engine statistics.
    pub fn stats(&self) -> Result<Stats> {
        let active = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE consolidation_status = 'active'",
            [], |row| row.get(0),
        )?;
        let consolidated = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE consolidation_status = 'consolidated'",
            [], |row| row.get(0),
        )?;
        let tombstoned = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE consolidation_status = 'tombstoned'",
            [], |row| row.get(0),
        )?;
        let edges = self.conn.query_row(
            "SELECT COUNT(*) FROM edges WHERE tombstoned = 0",
            [], |row| row.get(0),
        )?;
        let entities = self.conn.query_row(
            "SELECT COUNT(*) FROM entities",
            [], |row| row.get(0),
        )?;
        let operations = self.conn.query_row(
            "SELECT COUNT(*) FROM oplog",
            [], |row| row.get(0),
        )?;

        Ok(Stats {
            active_memories: active,
            consolidated_memories: consolidated,
            tombstoned_memories: tombstoned,
            edges,
            entities,
            operations,
        })
    }

    /// Append an operation to the oplog.
    pub fn log_op(&self, op_type: &str, target_rid: Option<&str>, payload: &serde_json::Value) -> Result<String> {
        let op_id = uuid7::uuid7().to_string();
        let payload_str = serde_json::to_string(payload)?;
        self.conn.execute(
            "INSERT INTO oplog (op_id, op_type, timestamp, target_rid, payload) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![op_id, op_type, now(), target_rid, payload_str],
        )?;
        Ok(op_id)
    }

    /// Close the database connection. After this, the engine cannot be used.
    pub fn close(self) -> Result<()> {
        self.conn.close().map_err(|(_, e)| AidbError::Database(e))
    }
}

/// Internal row struct for recall query results.
struct MemoryRow {
    rid: String,
    memory_type: String,
    text: String,
    created_at: f64,
    importance: f64,
    valence: f64,
    half_life: f64,
    last_access: f64,
    metadata: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim).map(|i| (seed + i as f32) * 0.1).collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    }

    fn empty_meta() -> serde_json::Value {
        serde_json::json!({})
    }

    #[test]
    fn test_new_and_stats() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let s = db.stats().unwrap();
        assert_eq!(s.active_memories, 0);
        assert_eq!(s.edges, 0);
    }

    #[test]
    fn test_record_and_get() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let emb = vec_seed(1.0, 8);
        let rid = db.record("hello world", "episodic", 0.8, 0.0, 604800.0, &empty_meta(), &emb).unwrap();
        assert_eq!(rid.len(), 36);

        let mem = db.get(&rid).unwrap().unwrap();
        assert_eq!(mem.text, "hello world");
        assert_eq!(mem.memory_type, "episodic");
        assert_eq!(mem.importance, 0.8);
        assert_eq!(mem.consolidation_status, "active");
    }

    #[test]
    fn test_record_updates_stats() {
        let db = AIDB::new(":memory:", 8).unwrap();
        db.record("one", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        db.record("two", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8)).unwrap();
        assert_eq!(db.stats().unwrap().active_memories, 2);
    }

    #[test]
    fn test_recall_basic() {
        let db = AIDB::new(":memory:", 8).unwrap();
        db.record("the cat sat on the mat", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        db.record("dogs are loyal friends", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(5.0, 8)).unwrap();
        db.record("cats love warm places", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.1, 8)).unwrap();

        let results = db.recall(&vec_seed(1.0, 8), 2, None, None, false).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_recall_empty() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let results = db.recall(&vec_seed(1.0, 8), 5, None, None, false).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_relate_and_get_edges() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let eid = db.relate("Alice", "Bob", "knows", 1.0).unwrap();
        assert_eq!(eid.len(), 36);

        let edges = db.get_edges("Alice").unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].src, "Alice");
        assert_eq!(edges[0].dst, "Bob");
    }

    #[test]
    fn test_forget() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let rid = db.record("forget me", "episodic", 0.5, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        assert!(db.forget(&rid).unwrap());
        let mem = db.get(&rid).unwrap().unwrap();
        assert_eq!(mem.consolidation_status, "tombstoned");
    }

    #[test]
    fn test_forget_nonexistent() {
        let db = AIDB::new(":memory:", 8).unwrap();
        assert!(!db.forget("nonexistent").unwrap());
    }

    #[test]
    fn test_decay_fresh() {
        let db = AIDB::new(":memory:", 8).unwrap();
        db.record("fresh", "episodic", 0.9, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8)).unwrap();
        let decayed = db.decay(0.01).unwrap();
        assert!(decayed.is_empty());
    }
}
