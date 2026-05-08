mod belief;
pub mod materializer;
pub(crate) mod op_types;
mod cache;
mod intent;
mod action;
mod evaluator;
mod policy;
mod suggest;
mod agenda;
mod temporal;
mod hawkes;
mod receptivity;
mod tick;
mod surfacing;
mod observer;
mod flywheel;
mod world_model;
mod experimenter;
mod skills;
mod extractor;
mod calibration;
mod introspection;
mod causal;
mod planner;
mod cognition;
mod coherence;
mod metacognition;
mod personality_bias;
mod query_dsl;
mod conflict;
mod analogy_engine;
mod schema_induction_engine;
mod narrative_engine;
mod counterfactual_engine;
mod belief_network_engine;
mod replay_engine;
mod perspective_engine;
mod feedback;
pub mod graph_state;
mod graph_ops;
mod indices;
mod learning;
mod lifecycle;
mod recall;
mod record;
mod procedural;
mod session;
mod stats;
mod storage;
mod temporal_helpers;
mod warrant;
pub mod moves;
pub mod tenant;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
// parking_lot::Mutex and RwLock: non-poisoning (no PoisonError on panic),
// smaller, faster, and integrate with parking_lot::deadlock::check_deadlock()
// which the server runs on a background task. Critical property: if a thread
// panics while holding an engine lock, subsequent acquirers do NOT see a
// PoisonError and do NOT themselves panic — we can recover. With std::sync,
// a single panic inside the engine can cascade into every other thread
// panicking on lock(), which cascades the whole process.
use parking_lot::{Mutex, MutexGuard, RwLock};

use base64::Engine;
use rand::Rng;
use rusqlite::{params, Connection};

use crate::encryption::{self, EncryptionProvider};
use crate::error::{YantrikDbError, Result};
use crate::graph_index::GraphIndex;
use crate::hlc::{HLCTimestamp, HLC};
use crate::hnsw::HnswIndex;
use crate::schema::{
    MIGRATE_V1_TO_V2, MIGRATE_V2_TO_V3, MIGRATE_V3_TO_V4, MIGRATE_V4_TO_V5,
    MIGRATE_V5_TO_V6, MIGRATE_V6_TO_V7, MIGRATE_V7_TO_V8, MIGRATE_V8_TO_V9,
    MIGRATE_V9_TO_V10, MIGRATE_V10_TO_V11, MIGRATE_V11_TO_V12, MIGRATE_V12_TO_V13,
    MIGRATE_V13_TO_V14, MIGRATE_V14_TO_V15, MIGRATE_V15_TO_V16, MIGRATE_V16_TO_V17,
    MIGRATE_V17_TO_V18, MIGRATE_V18_TO_V19, MIGRATE_V19_TO_V20, MIGRATE_V20_TO_V21,
    MIGRATE_V21_TO_V22, MIGRATE_V22_TO_V23, MIGRATE_V23_TO_V24, MIGRATE_V24_TO_V25, SCHEMA_SQL, SCHEMA_VERSION,
};
use crate::types::*;

/// The YantrikDB cognitive memory engine.
///
/// Thread-safe: all internal state is protected by `Mutex` or `RwLock`.
/// `conn` uses `Mutex` because `rusqlite::Connection` is `!Sync`.
/// Read-heavy fields (`scoring_cache`, `vec_index`, `graph_index`,
/// `active_sessions`) use `RwLock` for concurrent reader throughput.
///
/// **Lock ordering** (always acquire in this order to prevent deadlocks):
///   conn → hlc → scoring_cache → vec_index → graph_index → active_sessions
///
/// ## Concurrent recall (read pool)
///
/// `read_conns` is a small pool of additional SQLite connections opened
/// in WAL mode against the same database file. Each is wrapped in a
/// `Mutex` (since `Connection` is `!Sync`). Read-heavy paths like
/// `recall()` call [`Self::read_conn`] to acquire any free pooled
/// connection, allowing N concurrent recalls instead of all serialising
/// through the single `conn` mutex. Writes (record/forget/correct) and
/// migrations continue to use `conn` so SQLite's single-writer rule is
/// preserved naturally.
///
/// Pool size is configurable via the `YANTRIKDB_READ_POOL` env var
/// (default 4). Set to 0 to disable the pool — `read_conn()` then
/// returns the write connection, preserving v0.6.3 and earlier
/// behavior.
pub struct YantrikDB {
    pub(crate) conn: Mutex<Connection>,
    /// Pool of additional read-only SQLite connections opened against
    /// the same database file with WAL pragmas. Recall paths acquire a
    /// free connection round-robin to enable concurrent reads.
    pub(crate) read_conns: Vec<Mutex<Connection>>,
    /// Round-robin starting index for read pool acquisition.
    pub(crate) read_idx: std::sync::atomic::AtomicUsize,
    pub(crate) embedding_dim: usize,
    pub(crate) hlc: Mutex<HLC>,
    pub(crate) actor_id: String,
    pub(crate) scoring_cache: RwLock<HashMap<String, ScoringRow>>,
    pub(crate) vec_index: crate::vector::delta_index::DeltaIndex,
    /// Monotonic seq counter for vec_index appends/tombstones.
    /// Used by Phase 6 RYW (recall_with_seq); also feeds DeltaIndex's
    /// per-entry seq tag for compaction ordering.
    pub(crate) vec_seq: std::sync::atomic::AtomicU64,
    /// **Phase 6 RYW**: per-namespace high-water mark of applied seqs.
    /// Updated by record/record_with_rid (and siblings) after the write
    /// has materialized into the in-memory delta. `recall_with_seq` waits
    /// until `visible_seq[ns] >= min_seq` before scanning. Strict
    /// read-your-writes is opt-in; default `recall()` keeps current
    /// "delta is always visible" semantics.
    ///
    /// `DashMap<String, AtomicU64>` so the read path (`visible_seq_for`)
    /// is fully lock-free in steady state — a sharded hashmap shard read
    /// + an atomic load. Writers (`bump_visible_seq`) acquire only the
    /// sharded entry's lock to insert-on-first-use; subsequent bumps for
    /// the same namespace are a single shard-shared `fetch_max`. This
    /// keeps the recall hot path off the global mutex that the previous
    /// `parking_lot::Mutex<HashMap<...>>` design imposed (msg from
    /// yantrikdb-server, 2026-05-07: "DashMap eliminates the lock-on-every-
    /// recall that would dominate at scale").
    pub(crate) visible_seq: dashmap::DashMap<String, std::sync::atomic::AtomicU64>,
    /// **Phase 6 RYW**: Condvar + sentinel mutex paired with `visible_seq`
    /// for wake-on-update semantics in `wait_for_visible_seq`. The mutex
    /// is a `()` sentinel — no data lives behind it; it exists only
    /// because parking_lot::Condvar's `wait_for` API requires a guard.
    /// `record/record_with_rid` notify_all after bumping `visible_seq[ns]`;
    /// waiters re-check the AtomicU64 after each wakeup.
    pub(crate) visible_seq_cv: parking_lot::Condvar,
    pub(crate) visible_seq_wait_mu: parking_lot::Mutex<()>,
    pub(crate) graph_index: RwLock<GraphIndex>,
    pub(crate) enc: Option<EncryptionProvider>,
    /// Optional text-to-embedding converter. When set, enables `record_text()`
    /// and `recall_text()` which auto-embed text without an external server.
    embedder: Option<Box<dyn crate::types::Embedder + Send + Sync>>,
    /// Cache of active sessions: namespace → session_id
    pub(crate) active_sessions: RwLock<HashMap<String, String>>,
}

impl YantrikDB {
    /// Acquire a read connection from the pool. Round-robin across pool
    /// slots, with try_lock fast-path to avoid blocking when any slot
    /// is free. If all are busy, blocks on the round-robin choice.
    ///
    /// If the pool is empty (`YANTRIKDB_READ_POOL=0`), falls back to the
    /// write connection — preserves single-mutex behavior of pre-v0.6.4.
    pub(crate) fn read_conn(&self) -> MutexGuard<'_, Connection> {
        use std::sync::atomic::Ordering;
        let n = self.read_conns.len();
        if n == 0 {
            return self.conn.lock();
        }
        let start = self.read_idx.fetch_add(1, Ordering::Relaxed) % n;
        for i in 0..n {
            let idx = (start + i) % n;
            if let Some(g) = self.read_conns[idx].try_lock() {
                return g;
            }
        }
        // All slots busy — block on the round-robin choice.
        self.read_conns[start].lock()
    }
}

// Static assertion: YantrikDB must be Send + Sync.
const _: () = {
    fn _assert_send<T: Send>() {}
    fn _assert_sync<T: Sync>() {}
    fn _check() {
        _assert_send::<YantrikDB>();
        _assert_sync::<YantrikDB>();
    }
};

pub(crate) fn now() -> f64 {
    crate::time::now_secs()
}

/// Compute BLAKE3 hash of an embedding blob.
pub(crate) fn embedding_hash(embedding: &[f32]) -> Vec<u8> {
    let blob = crate::serde_helpers::serialize_f32(embedding);
    blake3::hash(&blob).as_bytes().to_vec()
}

/// Lightweight struct for fetching only text and metadata (post-scoring hydration).
pub(crate) struct TextMetadataRow {
    pub rid: String,
    pub text: String,
    pub metadata: String,
}

impl YantrikDB {
    /// Create a new YantrikDB instance with auto-generated actor_id.
    pub fn new(db_path: &str, embedding_dim: usize) -> Result<Self> {
        Self::open(db_path, embedding_dim, None, None)
    }

    /// Create a new YantrikDB instance with an explicit actor_id (for sync tests).
    pub fn new_with_actor(db_path: &str, embedding_dim: usize, actor_id: &str) -> Result<Self> {
        Self::open(db_path, embedding_dim, Some(actor_id.to_string()), None)
    }

    /// Create a new encrypted YantrikDB instance.
    ///
    /// The 32-byte `master_key` is used to wrap/unwrap a per-database Data Encryption Key (DEK).
    /// All text, metadata, and embedding fields are encrypted at rest using AES-256-GCM.
    /// In-memory indexes operate on plaintext for full query performance.
    pub fn new_encrypted(db_path: &str, embedding_dim: usize, master_key: &[u8; 32]) -> Result<Self> {
        Self::open(db_path, embedding_dim, None, Some(master_key))
    }

    fn open(
        db_path: &str,
        embedding_dim: usize,
        actor_id: Option<String>,
        master_key: Option<&[u8; 32]>,
    ) -> Result<Self> {
        let conn = Connection::open(db_path)?;

        // Enforce SQLite pragmas for durability + performance.
        // See CONCURRENCY.md and ops/runbooks/disk-full.md.
        //
        // journal_mode=WAL: write-ahead logging for concurrent readers +
        //   crash recovery. Critical for all multi-threaded usage.
        // synchronous=NORMAL: in WAL mode, NORMAL is crash-safe (protects
        //   against corruption on power loss) while avoiding the fsync-per-
        //   commit overhead of FULL. The WAL itself is fsync'd on checkpoint.
        // foreign_keys=ON: enforce referential integrity on conflicts,
        //   sessions, etc.
        // busy_timeout=5000: wait up to 5 seconds for a lock instead of
        //   immediately returning SQLITE_BUSY. Prevents spurious failures
        //   under concurrent access (e.g., oplog GC + consolidation).
        // wal_autocheckpoint=1000: auto-checkpoint after 1000 pages (~4MB).
        //   Prevents unbounded WAL growth under sustained write load.
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; \
             PRAGMA synchronous=NORMAL; \
             PRAGMA foreign_keys=ON; \
             PRAGMA busy_timeout=5000; \
             PRAGMA wal_autocheckpoint=1000;",
        )?;

        // Verify critical pragmas actually took effect. SQLite silently
        // ignores some pragmas in certain modes (e.g. journal_mode on
        // read-only or in-memory databases). Log a warning if any mismatch.
        let actual_journal: String = conn.query_row(
            "PRAGMA journal_mode", [], |row| row.get(0),
        ).unwrap_or_default();
        if actual_journal != "wal" && db_path != ":memory:" {
            tracing::warn!(
                expected = "wal",
                actual = %actual_journal,
                path = %db_path,
                "SQLite journal_mode pragma did not take effect"
            );
        }

        // Check existing schema version for migration
        let existing_version = Self::get_schema_version(&conn);

        // Sequential migration chain — each version cascades.
        let migrations: &[(i32, &str)] = &[
            (1, MIGRATE_V1_TO_V2),
            (2, MIGRATE_V2_TO_V3),
            (3, MIGRATE_V3_TO_V4),
            (4, MIGRATE_V4_TO_V5),
            (5, MIGRATE_V5_TO_V6),
            (6, MIGRATE_V6_TO_V7),
            (7, MIGRATE_V7_TO_V8),
            (8, MIGRATE_V8_TO_V9),
            (9, MIGRATE_V9_TO_V10),
            (10, MIGRATE_V10_TO_V11),
            (11, MIGRATE_V11_TO_V12),
            (12, MIGRATE_V12_TO_V13),
            (13, MIGRATE_V13_TO_V14),
            (14, MIGRATE_V14_TO_V15),
            (15, MIGRATE_V15_TO_V16),
            (16, MIGRATE_V16_TO_V17),
            (17, MIGRATE_V17_TO_V18),
            (18, MIGRATE_V18_TO_V19),
            (19, MIGRATE_V19_TO_V20),
            (20, MIGRATE_V20_TO_V21),
            (21, MIGRATE_V21_TO_V22),
            (22, MIGRATE_V22_TO_V23),
            (23, MIGRATE_V23_TO_V24),
            (24, MIGRATE_V24_TO_V25),
        ];
        if let Some(v) = existing_version {
            for &(from_v, sql) in migrations {
                if v <= from_v {
                    conn.execute_batch(sql)?;
                }
            }
        }

        conn.execute_batch(SCHEMA_SQL)?;

        // Populate seed substitution categories (idempotent)
        crate::distributed::seed_categories::populate_seed_categories(&conn)?;

        // RFC 008 M5b: seed move_type_registry + inference_basis_registry
        // with canonical vocabulary (idempotent INSERT OR IGNORE).
        crate::engine::moves::seed_registries_inner(&conn)?;

        // Set schema version
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?1)",
            params![SCHEMA_VERSION.to_string()],
        )?;

        // Resolve actor_id: explicit > stored in meta > generate new
        let actor_id = if let Some(id) = actor_id {
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('actor_id', ?1)",
                params![id],
            )?;
            id
        } else {
            match Self::get_meta(&conn, "actor_id")? {
                Some(id) => id,
                None => {
                    let id = crate::id::new_id();
                    conn.execute(
                        "INSERT OR REPLACE INTO meta (key, value) VALUES ('actor_id', ?1)",
                        params![id],
                    )?;
                    id
                }
            }
        };

        // Resolve node_id: stored in meta > generate random
        let node_id: u32 = match Self::get_meta(&conn, "node_id")? {
            Some(s) => s.parse().unwrap_or_else(|_| {
                let id: u32 = rand::thread_rng().gen();
                id
            }),
            None => {
                let id: u32 = rand::thread_rng().gen();
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('node_id', ?1)",
                    params![id.to_string()],
                )?;
                id
            }
        };

        // Initialize encryption (envelope pattern: master_key wraps DEK)
        let enc = if let Some(mk) = master_key {
            let provider = match Self::get_meta(&conn, "encrypted_dek")? {
                Some(wrapped_b64) => {
                    // Existing DB: unwrap DEK
                    let wrapped = base64::engine::general_purpose::STANDARD
                        .decode(&wrapped_b64)
                        .map_err(|e| YantrikDbError::Encryption(format!("DEK base64: {e}")))?;
                    let dek = encryption::unwrap_dek(mk, &wrapped)?;
                    EncryptionProvider::from_dek(&dek)
                }
                None => {
                    // New DB: generate and store DEK
                    let dek = encryption::generate_key();
                    let wrapped = encryption::wrap_dek(mk, &dek)?;
                    let wrapped_b64 = base64::engine::general_purpose::STANDARD.encode(&wrapped);
                    conn.execute(
                        "INSERT OR REPLACE INTO meta (key, value) VALUES ('encrypted_dek', ?1)",
                        params![wrapped_b64],
                    )?;
                    conn.execute(
                        "INSERT OR REPLACE INTO meta (key, value) VALUES ('encryption_enabled', '1')",
                        [],
                    )?;
                    EncryptionProvider::from_dek(&dek)
                }
            };
            Some(provider)
        } else {
            // Verify we're not opening an encrypted DB without a key
            if Self::get_meta(&conn, "encryption_enabled")?.as_deref() == Some("1") {
                return Err(YantrikDbError::Encryption(
                    "database is encrypted but no master_key provided".into(),
                ));
            }
            None
        };

        let scoring_cache = Self::load_scoring_cache(&conn)?;
        let vec_index = Self::build_vec_index_with_enc(&conn, embedding_dim, enc.as_ref())?;
        let graph_index = GraphIndex::build_from_db(&conn)?;

        // Load active sessions from DB
        let active_sessions = Self::load_active_sessions(&conn)?;

        // Build the read-connection pool. Each pooled connection opens
        // independently against the same SQLite file with WAL-mode
        // pragmas — WAL allows multiple readers concurrently. Pool
        // size is read from YANTRIKDB_READ_POOL env (default 4).
        //
        // In-memory databases (`:memory:`) are SKIPPED: each
        // `Connection::open(":memory:")` creates a *new* in-memory db,
        // so pooled read connections wouldn't see writes from the main
        // connection. Tests use `:memory:` extensively; falling back to
        // the single write connection for those is correct and matches
        // pre-pool behavior.
        let is_memory = db_path == ":memory:" || db_path.starts_with("file::memory:");
        let pool_size: usize = if is_memory {
            0
        } else {
            std::env::var("YANTRIKDB_READ_POOL")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(4)
        };
        let mut read_conns = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            let rc = Connection::open(db_path)?;
            rc.execute_batch(
                "PRAGMA journal_mode=WAL; \
                 PRAGMA synchronous=NORMAL; \
                 PRAGMA foreign_keys=ON; \
                 PRAGMA busy_timeout=5000;",
            )?;
            read_conns.push(Mutex::new(rc));
        }
        if pool_size > 0 {
            tracing::info!(
                pool_size,
                "yantrikdb-core: read connection pool initialized"
            );
        }

        Ok(Self {
            conn: Mutex::new(conn),
            read_conns,
            read_idx: std::sync::atomic::AtomicUsize::new(0),
            embedding_dim,
            hlc: Mutex::new(HLC::new(node_id)),
            actor_id,
            scoring_cache: RwLock::new(scoring_cache),
            vec_index: {
                let delta_max = std::env::var("YANTRIKDB_DELTA_MAX")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(crate::vector::delta_index::DEFAULT_DELTA_MAX);
                let max_dirty_age = std::env::var("YANTRIKDB_MAX_DIRTY_AGE_SECS")
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
                    .map(std::time::Duration::from_secs)
                    .unwrap_or(crate::vector::delta_index::DEFAULT_MAX_DIRTY_AGE);
                crate::vector::delta_index::DeltaIndex::from_cold_with_age(
                    vec_index,
                    delta_max,
                    max_dirty_age,
                )
            },
            vec_seq: std::sync::atomic::AtomicU64::new(0),
            visible_seq: dashmap::DashMap::new(),
            visible_seq_cv: parking_lot::Condvar::new(),
            visible_seq_wait_mu: parking_lot::Mutex::new(()),
            graph_index: RwLock::new(graph_index),
            enc,
            embedder: None,
            active_sessions: RwLock::new(active_sessions),
        })
    }

    fn get_schema_version(conn: &Connection) -> Option<i32> {
        conn.query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |row| {
                let v: String = row.get(0)?;
                Ok(v.parse::<i32>().unwrap_or(0))
            },
        )
        .ok()
    }

    fn load_active_sessions(conn: &Connection) -> Result<HashMap<String, String>> {
        let mut map = HashMap::new();
        // Table may not exist yet during initial schema creation
        let mut stmt = match conn.prepare(
            "SELECT namespace, session_id FROM sessions WHERE status = 'active'",
        ) {
            Ok(s) => s,
            Err(_) => return Ok(map),
        };
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        for row in rows {
            let (ns, sid) = row?;
            map.insert(ns, sid);
        }
        Ok(map)
    }

    fn get_meta(conn: &Connection, key: &str) -> Result<Option<String>> {
        match conn.query_row(
            "SELECT value FROM meta WHERE key = ?1",
            params![key],
            |row| row.get(0),
        ) {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Get a new HLC timestamp (ticks the clock forward).
    pub fn tick_hlc(&self) -> HLCTimestamp {
        self.hlc.lock().now()
    }

    /// Merge a remote HLC timestamp into the local clock.
    pub fn merge_hlc(&self, remote: HLCTimestamp) -> HLCTimestamp {
        self.hlc.lock().recv(remote)
    }

    /// Get the actor_id of this instance.
    pub fn actor_id(&self) -> &str {
        &self.actor_id
    }

    /// **Engine-pressure surface for external schedulers.**
    ///
    /// Returns the soft cap on the delta tier (i.e. the post-v0.6.7
    /// `DEFAULT_DELTA_MAX` of 256, or whatever the operator set via the
    /// `YANTRIKDB_DELTA_MAX` env var). Used by yantrikdb-server's tick
    /// loop to scale the enrichment-pause threshold proportionally to
    /// engine capacity — see CONCURRENCY.md and the cross-stack rule
    /// "engine pressure suppresses enrichment" (saga task 16).
    pub fn delta_max(&self) -> usize {
        self.vec_index.delta_max()
    }

    /// Current delta-tier length (live entries + tombstone markers).
    /// Pairs with `delta_max()` for pressure-ratio computation.
    pub fn delta_len(&self) -> usize {
        self.vec_index.delta_len()
    }

    /// Current cold-tier length (entries that have been merged into
    /// the HNSW). Useful for ops dashboards that want to see the
    /// hot/cold split — most reads against a healthy engine should
    /// hit cold rather than the linear delta scan.
    pub fn cold_len(&self) -> usize {
        self.vec_index.cold_len()
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Acquire the database connection lock.
    ///
    /// Returns a `MutexGuard` that deref's to `&Connection`.
    /// The lock is released when the guard is dropped.
    pub fn conn(&self) -> MutexGuard<'_, Connection> {
        self.conn.lock()
    }

    /// Whether this instance has encryption enabled.
    pub fn is_encrypted(&self) -> bool {
        self.enc.is_some()
    }

    /// Get a reference to the encryption provider (for vault operations).
    pub fn encryption(&self) -> Option<&EncryptionProvider> {
        self.enc.as_ref()
    }

    // ── Encryption helpers (transparent to callers) ──

    /// Encrypt a string field if encryption is enabled, otherwise pass through.
    pub(crate) fn encrypt_text(&self, plaintext: &str) -> Result<String> {
        match &self.enc {
            Some(e) => e.encrypt_string(plaintext),
            None => Ok(plaintext.to_string()),
        }
    }

    /// Decrypt a string field if encryption is enabled, otherwise pass through.
    pub(crate) fn decrypt_text(&self, stored: &str) -> Result<String> {
        match &self.enc {
            Some(e) => e.decrypt_string(stored),
            None => Ok(stored.to_string()),
        }
    }

    /// Encrypt an embedding blob if encryption is enabled.
    pub(crate) fn encrypt_embedding(&self, emb_blob: &[u8]) -> Result<Vec<u8>> {
        match &self.enc {
            Some(e) => e.encrypt_bytes(emb_blob),
            None => Ok(emb_blob.to_vec()),
        }
    }

    /// Decrypt an embedding blob if encryption is enabled.
    pub(crate) fn decrypt_embedding(&self, stored: &[u8]) -> Result<Vec<u8>> {
        match &self.enc {
            Some(e) => e.decrypt_bytes(stored),
            None => Ok(stored.to_vec()),
        }
    }

    /// Close the database connection. After this, the engine cannot be used.
    ///
    /// parking_lot::Mutex::into_inner returns T directly (no PoisonError),
    /// unlike std::sync::Mutex::into_inner which returns Result.
    pub fn close(self) -> Result<()> {
        self.conn
            .into_inner()
            .close()
            .map_err(|(_, e)| YantrikDbError::Database(e))
    }

    // ── Embedder integration ──

    /// Set the text-to-embedding converter. Enables `embed()`, `record_text()`,
    /// and `recall_text()` which auto-embed text without an external server.
    pub fn set_embedder(&mut self, embedder: Box<dyn crate::types::Embedder + Send + Sync>) {
        self.embedder = Some(embedder);
    }

    /// Whether an embedder is configured.
    pub fn has_embedder(&self) -> bool {
        self.embedder.is_some()
    }

    /// Embed text using the configured embedder.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embedder
            .as_ref()
            .ok_or(YantrikDbError::NoEmbedder)?
            .embed(text)
            .map_err(|e| YantrikDbError::Inference(e.to_string()))
    }

    /// Record a memory with automatic embedding generation.
    pub fn record_text(
        &self,
        text: &str,
        memory_type: &str,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: &serde_json::Value,
        namespace: &str,
        certainty: f64,
        domain: &str,
        source: &str,
        emotional_state: Option<&str>,
    ) -> Result<String> {
        let embedding = self.embed(text)?;
        self.record(
            text,
            memory_type,
            importance,
            valence,
            half_life,
            metadata,
            &embedding,
            namespace,
            certainty,
            domain,
            source,
            emotional_state,
        )
    }

    /// Recall memories by text query with automatic embedding.
    pub fn recall_text(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<RecallResult>> {
        let embedding = self.embed(query)?;
        self.recall(
            &embedding,
            top_k,
            None,  // time_window
            None,  // memory_type
            false, // include_consolidated
            true,  // expand_entities
            Some(query),
            false, // skip_reinforce
            None,  // namespace
            None,  // domain
            None,  // source
        )
    }

    /// Recall memories with domain and source filters.
    ///
    /// Like `recall_text` but restricts results to a specific domain
    /// (e.g. `"session/summary"`, `"audit/tools"`) and/or source
    /// (e.g. `"self"`, `"companion"`, `"system"`).
    pub fn recall_text_filtered(
        &self,
        query: &str,
        top_k: usize,
        domain: Option<&str>,
        source: Option<&str>,
    ) -> Result<Vec<RecallResult>> {
        let embedding = self.embed(query)?;
        self.recall(
            &embedding,
            top_k,
            None,  // time_window
            None,  // memory_type
            false, // include_consolidated
            true,  // expand_entities
            Some(query),
            false, // skip_reinforce
            None,  // namespace
            domain,
            source,
        )
    }
}
