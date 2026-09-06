mod action;
mod agenda;
mod analogy_engine;
mod audit;
mod belief;
mod belief_network_engine;
mod bitemporal;
mod cache;
mod calibration;
mod capture;
mod causal;
mod chunking;
mod claims_lane;
mod cognition;
mod coherence;
pub mod conflict;
pub mod conversation;
mod counterfactual_engine;
pub mod demand;
pub mod digest;
mod durable_embeddings;
mod embedder_window;
mod evaluator;
mod experimenter;
mod extractor;
pub mod facets;
mod feedback;
mod flywheel;
pub mod graph_ops;
pub mod graph_state;
mod hawkes;
mod idempotency;
pub mod importance;
mod impressions;
mod indices;
mod intent;
mod introspection;
mod learning;
mod lexical;
mod lifecycle;
pub mod links;
pub mod maintenance;
pub mod materializer;
mod metacognition;
pub mod moves;
mod narrative_engine;
mod observer;
pub(crate) mod op_types;
pub mod pack;
mod personality_bias;
mod perspective_engine;
mod planner;
mod policy;
mod procedural;
mod query_dsl;
mod recall;
mod receptivity;
mod record;
pub mod reembed;
pub mod reextract;
pub mod repair;
mod replay_engine;
mod reservation;
mod sanitize;
mod savepoint;
mod schema_induction_engine;
mod session;
mod skills;
mod snippet;
pub mod split;
mod stats;
mod storage;
mod suggest;
mod surfacing;
pub mod tasks;
mod temporal;
mod temporal_helpers;
pub mod tenant;
#[cfg(test)]
mod tests;
pub mod thread;
mod tick;
mod warrant;
mod world_model;
pub mod write_router;

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
use crate::error::{Result, YantrikDbError};
use crate::graph_index::GraphIndex;
use crate::hlc::{HLCTimestamp, HLC};
use crate::hnsw::HnswIndex;
use crate::provenance::GateVerdict;
use crate::schema::{
    MIGRATE_V10_TO_V11, MIGRATE_V11_TO_V12, MIGRATE_V12_TO_V13, MIGRATE_V13_TO_V14,
    MIGRATE_V14_TO_V15, MIGRATE_V15_TO_V16, MIGRATE_V16_TO_V17, MIGRATE_V17_TO_V18,
    MIGRATE_V18_TO_V19, MIGRATE_V19_TO_V20, MIGRATE_V1_TO_V2, MIGRATE_V20_TO_V21,
    MIGRATE_V21_TO_V22, MIGRATE_V22_TO_V23, MIGRATE_V23_TO_V24, MIGRATE_V24_TO_V25,
    MIGRATE_V25_TO_V26, MIGRATE_V26_TO_V27, MIGRATE_V27_TO_V28, MIGRATE_V28_TO_V29,
    MIGRATE_V29_TO_V30, MIGRATE_V2_TO_V3, MIGRATE_V30_TO_V31, MIGRATE_V31_TO_V32,
    MIGRATE_V32_TO_V33, MIGRATE_V33_TO_V34, MIGRATE_V34_TO_V35, MIGRATE_V35_TO_V36,
    MIGRATE_V36_TO_V37, MIGRATE_V37_TO_V38, MIGRATE_V3_TO_V4, MIGRATE_V40_TO_V41,
    MIGRATE_V41_TO_V42, MIGRATE_V42_TO_V43, MIGRATE_V44_TO_V45, MIGRATE_V45_TO_V46,
    MIGRATE_V46_TO_V47, MIGRATE_V47_TO_V48, MIGRATE_V48_TO_V49, MIGRATE_V49_TO_V50,
    MIGRATE_V4_TO_V5, MIGRATE_V50_TO_V51, MIGRATE_V5_TO_V6, MIGRATE_V6_TO_V7, MIGRATE_V7_TO_V8,
    MIGRATE_V8_TO_V9, MIGRATE_V9_TO_V10, SCHEMA_SQL, SCHEMA_VERSION,
};
use crate::types::*;

/// The YantrikDB cognitive memory engine.
///
/// Thread-safe: all internal state is protected by `Mutex` or `RwLock`.
/// `conn` uses `Mutex` because `rusqlite::Connection` is `!Sync`.
/// Read-heavy fields (`scoring_cache`, `graph_index`,
/// `active_sessions`) use `RwLock` for concurrent reader throughput.
/// The vector index lives inside `search_state` as `Arc<DeltaIndex>`
/// (issue #41 brainstorm-4 §1) — `DeltaIndex` carries its own
/// internal locks, and `ArcSwap<SearchState>` is the atomic
/// publication wrapper.
///
/// **Lock ordering** (always acquire in this order to prevent deadlocks):
///   conn → hlc → scoring_cache → SearchState.vec_index → graph_index → active_sessions
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
    /// The path this database was opened from. Needed to locate the
    /// sibling `<stem>.packs/` directory where installed packs live.
    /// `":memory:"` for in-memory databases, which cannot host packs.
    pub(crate) db_path: String,
    pub(crate) hlc: Mutex<HLC>,
    pub(crate) actor_id: String,
    pub(crate) scoring_cache: RwLock<HashMap<String, ScoringRow>>,
    // Issue #41 brainstorm-4 §1: standalone `vec_index` field retired.
    // The vector index now lives ONLY inside `search_state` as
    // `Arc<DeltaIndex>`, so `search_state.store(new_state)` becomes the
    // single atomic publication unit for (embedder + provenance + dim
    // + generation + vec_index). Reembed Phase-2 swap can republish a
    // brand-new `DeltaIndex` atomically with the rest of SearchState
    // without any split-brain window. Readers do
    // `self.search_state.load[_full]().vec_index.X(...)`.
    /// Monotonic seq counter for SearchState.vec_index appends/tombstones.
    /// Used by Phase 6 RYW (recall_with_seq); also feeds DeltaIndex's
    /// per-entry seq tag for compaction ordering.
    pub(crate) vec_seq: std::sync::atomic::AtomicU64,
    /// **v0.7.1 perf hotfix.** Cached pending-oplog count for foreground
    /// `log_op_pending` backpressure check. Replaces the per-call
    /// `SELECT COUNT(*) FROM oplog WHERE applied = 0` index scan that
    /// dominated v0.7.0's foreground write path under sustained load
    /// (5× tput drop diagnosed via yantrikdb-server msg `b951a2de`).
    ///
    /// Maintained by:
    /// - `open()`: initialize from one-time SQL `SELECT COUNT(...)` at boot.
    /// - `log_op_pending`: `fetch_add(1)` after a successful insert.
    /// - `mark_op_applied`: `fetch_sub(1)` only when the row transitioned
    ///   from `applied=0` to `applied=1` (the bool the method now returns).
    ///
    /// Backpressure check on the foreground hot path becomes a single
    /// `Relaxed` atomic load instead of a Mutex<Connection> acquire +
    /// index scan + drop.
    pub(crate) pending_op_count: std::sync::atomic::AtomicI64,
    /// **v0.10 Item 1 — status-led read path.** Cached
    /// `meta.status_read_policy`: `true` means recall EXCLUDES superseded
    /// records from result eligibility (the fresh-install default);
    /// `false` is the legacy include-everything behavior for pre-v0.10
    /// databases until the operator opts in via
    /// [`YantrikDB::set_status_read_policy`]. Exclusion is
    /// eligibility-not-demotion: superseded rows never compete for
    /// top_k slots, rather than being score-penalized. Per-call
    /// `include_superseded = true` re-admits them (stamped) for
    /// history/archaeology queries.
    pub(crate) exclude_superseded_reads: std::sync::atomic::AtomicBool,
    /// **v0.10 Item 1 — adoption nudge.** Since-boot count of recall
    /// results served while superseded (only possible on legacy-policy
    /// databases or `include_superseded` calls). Surfaced in `stats()`
    /// so operators of migrated DBs can see what the status read policy
    /// would have excluded before opting in. In-memory by design — a
    /// durable counter would put a write on the recall hot path.
    pub(crate) superseded_served_since_boot: std::sync::atomic::AtomicU64,
    /// Since-boot count of recalls whose desired HNSW candidate pool exceeded
    /// the engine's bounded oversampling ceiling. This makes a quality-relevant
    /// cap visible without adding persistence or a write to the recall path.
    pub(crate) recall_candidate_cap_bound_since_boot:
        parking_lot::Mutex<std::collections::HashMap<String, u64>>,
    /// True once namespace-level recall-cap telemetry has folded a new
    /// namespace into the bounded overflow bucket.
    pub(crate) recall_candidate_cap_namespace_stats_truncated_since_boot:
        std::sync::atomic::AtomicBool,
    /// Local cap refusals since open. In-memory by design: the rejected
    /// transaction must remain side-effect-free; durable current pressure is
    /// independently visible from `synthesis_dependencies` in `stats()`.
    pub(crate) synthesis_fanout_refused_since_boot: std::sync::atomic::AtomicU64,
    /// **Embedder input window, detected empirically** (see
    /// `engine::embedder_window`). The `Embedder` trait cannot declare a
    /// window — a BYO or Python-callable embedder is opaque — so the
    /// engine probes for one: 0 = not probed yet, `usize::MAX` = no
    /// truncation detected, otherwise the approximate character budget
    /// beyond which text stops affecting the vector.
    ///
    /// This exists because silent truncation is silent retrieval loss:
    /// a record longer than the window is stored intact and embedded
    /// only from its head, so its tail becomes unfindable — the same
    /// stored-active-unfindable shape as the HNSW orphan bug, measured
    /// at 73% of records on a production install.
    pub(crate) embedder_window_chars: std::sync::atomic::AtomicUsize,
    /// Since-boot count of writes whose text exceeded the detected
    /// window. In-memory by design, like the counters above.
    pub(crate) embedder_truncated_writes: std::sync::atomic::AtomicU64,
    /// Since-boot count of writes whose overflow was covered by chunk
    /// vectors instead (`engine::chunking`) — handled, not lost, so
    /// they deliberately do NOT count as truncated.
    pub(crate) embedder_chunked_writes: std::sync::atomic::AtomicU64,
    /// **v0.10 Item 4a.4 — anti-laundering gate mode**, cached from
    /// `meta.provenance_gate_mode` (0=off, 1=warn, 2=enforce). Fresh installs
    /// default to enforce; migrated/legacy installs to warn (see open()).
    pub(crate) provenance_gate_mode: std::sync::atomic::AtomicU8,
    /// **v0.10 Item 4a.4 — adoption nudge.** Since-boot count of writes the
    /// provenance gate FLAGGED as internally inconsistent but did NOT refuse
    /// (warn mode). Surfaced in `stats()` so a migrated DB's operator sees what
    /// `enforce` would reject before opting in. In-memory by design.
    pub(crate) provenance_flagged_since_boot: std::sync::atomic::AtomicU64,
    /// **v0.10 Item 3 — correction seqlock (sol r4).** A DB-wide epoch that
    /// makes a text-changing correction's (SQL commit + vector publish +
    /// scoring-cache update) atomic FROM A READER'S PERSPECTIVE, without
    /// versioning cold entries. A correction bumps this ODD before its
    /// mutation and back EVEN (via RAII, on every error/panic path) after.
    /// `recall` reads an even value before candidate generation and rechecks
    /// the identical value after hydration; a change (or odd) means a
    /// correction interleaved — the ranking vector and the hydrated text
    /// could be different content versions — so the recall discards and
    /// retries. Even at boot (0). See [`Self::enter_correction_epoch`].
    pub(crate) correction_epoch: std::sync::atomic::AtomicU64,
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
    /// **Issue #41 reembed primitive.** Synchronized cutover barrier
    /// between synchronous writes (`Normal` state) and queued writes
    /// (`Queueing` state during reembed). Writers acquire via
    /// `try_enter_sync_writer()` and hold the RAII guard for the full
    /// memories INSERT + vec_index.append + oplog write critical
    /// section. Reembed flips state to `Queueing`, waits for
    /// `wait_for_no_sync_writers()`, then can safely capture
    /// `build_hwm` knowing no synchronous writer can still commit to
    /// the old generation. See `engine::write_router` module for the
    /// brainstorm-2 rationale and the cutover-sequence regression
    /// test.
    pub(crate) write_router: crate::engine::write_router::SharedWriteRouter,

    /// **Issue #41 — layer 2 / brainstorm-3.** Atomically-swappable
    /// SearchState carrying the runtime embedder + index_embedding
    /// provenance + generation + HNSW params. Read paths acquire once
    /// via `self.search_state.load_full()` and use the snapshot for
    /// the full request — this prevents observing a mixed embedder /
    /// provenance / dim state mid-set_embedder or mid-reembed.
    ///
    /// Today this co-exists with the legacy `embedder: Option<Box<...>>`
    /// and `embedding_dim: usize` fields above. The migration retires
    /// those in a later checkpoint; until then, search_state mirrors
    /// the legacy fields on every set_embedder / new(). See
    /// `engine::reembed::SearchState` for the field semantics.
    pub(crate) search_state: arc_swap::ArcSwap<crate::engine::reembed::SearchState>,

    /// **Issue #41 — layer 2 / brainstorm-3.** Serializes SearchState
    /// republication. Acquired by:
    /// - `set_embedder` / `set_embedder_named` (mode validation +
    ///    coherent-bundle publication)
    /// - Future `reembed()` cutover (final swap)
    /// - Future empty-index-reset paths
    ///
    /// NOT held by writers — writers serialize via `write_router`. Two
    /// separate primitives for two separate invariants:
    /// - `write_router` = "is this writer allowed to take the sync
    ///    path right now?"
    /// - `index_write_lock` = "is the SearchState mid-republication
    ///    right now?"
    ///
    /// No double-locking risk: set_embedder doesn't acquire
    /// write_router, writers don't acquire index_write_lock.
    pub(crate) index_write_lock: parking_lot::Mutex<()>,

    /// **Packs.** Read-only knowledge packs currently mounted against
    /// this database, in mount order. Each entry owns its own
    /// connection, HNSW and scoring cache; none of them touch host
    /// state, so unmounting is `retain()` and nothing else.
    ///
    /// Recall clones the Arcs under a short read lock
    /// (`pack_snapshot()`) rather than holding the registry for the
    /// request, so mounting or unmounting never blocks a recall in
    /// flight.
    pub(crate) packs: parking_lot::RwLock<Vec<std::sync::Arc<crate::engine::pack::MountedPack>>>,

    /// Whether this database's embedder identity is already on disk.
    /// Keeps `stamp_embedder_identity_once` to a relaxed atomic load on
    /// the `record_text` hot path after the first write.
    pub(crate) embedder_identity_stamped: std::sync::atomic::AtomicBool,
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

/// Default maximum number of verified synthesis generations backed by one
/// evidence record. The measured BEAM write-synthesis cohort normally emits
/// two atomic axes per source; 64 leaves room for additional axes, rollups,
/// and regeneration while bounding correction/forget invalidation work.
pub const DEFAULT_SYNTHESIS_FANOUT_CAP: usize = 64;

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

/// Embedder a NEW store created via [`YantrikDB::with_default`] uses.
///
/// Downloaded on first use (~28 MB, SHA-256 pinned, cached under the
/// user's cache dir) rather than bundled: the crate already ships 7.9 MB
/// of `potion-base-2M` weights via `include_bytes!` and crates.io caps a
/// published crate at 10 MB, so this one cannot be baked in.
#[cfg(feature = "embedder-download")]
pub const DEFAULT_NEW_STORE_EMBEDDER: &str = "potion-base-8M";

impl YantrikDB {
    /// Create a new YantrikDB instance with auto-generated actor_id.
    pub fn new(db_path: &str, embedding_dim: usize) -> Result<Self> {
        let mut db = Self::open(db_path, embedding_dim, None, None)?;
        Self::finish_construction(&mut db);
        Ok(db)
    }

    /// **Saga task 20** — convenience constructor that opens with the
    /// engine's bundled embedder dimension (currently 64 for
    /// `potion-base-2M`). Equivalent to `YantrikDB::new(path, 64)`
    /// when the `bundled-embedder` feature is on. Lets callers stay
    /// agnostic to the bundled model's dimension; if the bundle ever
    /// changes (e.g. Slice C swaps in a 256-dim variant) the
    /// `with_default()` users get the new dim automatically without
    /// having to update their code.
    ///
    /// Slim builds (`--no-default-features`) compile this method out
    /// — there is no bundled embedder to align with.
    ///
    /// # Which embedder a NEW store gets (changed 2026-08-13)
    ///
    /// New stores open at [`DEFAULT_NEW_STORE_EMBEDDER`]'s dimension and
    /// download it on first use; the bundled 64-dim `potion-base-2M` is
    /// the offline fallback. The measurement behind the switch, on 5,035
    /// real production memories with 12 rid-pinned probes, retrieved
    /// through this engine's own `recall()` rather than raw cosine:
    ///
    /// | embedder       | MRR   | correct record absent from top 100 |
    /// |----------------|-------|------------------------------------|
    /// | potion-base-2M | 0.120 | 4 of 12                            |
    /// | potion-base-8M | 0.312 | 1 of 12                            |
    ///
    /// The miss rate is the reason, not the MRR: under the bundled model
    /// a third of real questions had no correct answer anywhere in the
    /// first hundred results, which reads to a user as the memory simply
    /// not being there. (On conversational-paraphrase corpora the two are
    /// indistinguishable at every k from 2 to 80 — the gain is specific to
    /// dense, vocabulary-heavy stores, which is what agent memory is.)
    ///
    /// # Existing stores never change dimension
    ///
    /// An existing database is opened at the dimension it already holds,
    /// so this switch cannot strand anyone's data. That check is the whole
    /// reason this method is not simply `Self::new(path, 256)`: the vector
    /// index is built from the dimension passed here, so opening a 64-dim
    /// store at 256 would build a mismatched index over existing vectors.
    #[cfg(feature = "bundled-embedder")]
    pub fn with_default(db_path: &str) -> Result<Self> {
        Self::new(db_path, Self::default_dim_for(db_path))
    }

    /// Dimension `with_default` should open `db_path` at.
    ///
    /// Existing store → the dimension it already holds. New store → the
    /// downloadable default if it can be obtained, else the bundled dim.
    #[cfg(feature = "bundled-embedder")]
    fn default_dim_for(db_path: &str) -> usize {
        // In-memory databases keep the bundled embedder deliberately.
        // They are ephemeral, so the retrieval quality that motivated the
        // switch cannot accrue to them, and the engine's own test suite
        // opens hundreds of them — defaulting those to a 28 MB fetch would
        // make `cargo test` require the network. Callers who want the
        // larger model in memory ask for it: `new(":memory:", 256)` then
        // `set_embedder_named`.
        if db_path.is_empty() || db_path.starts_with(':') {
            return crate::embedder::BUNDLED_EMBEDDER_DIM;
        }
        if let Some(dim) = Self::detect_existing_dim(db_path) {
            return dim;
        }
        #[cfg(feature = "embedder-download")]
        {
            // Resolve the model BEFORE choosing the dimension. Opening at
            // 256 first and discovering the download failed afterwards
            // would leave a 256-dim store with no embedder that can fill
            // it — a database broken by its own constructor. Cached after
            // the first call, so this is not a per-open network hit.
            match crate::embedder::DownloadedEmbedder::fetch(DEFAULT_NEW_STORE_EMBEDDER) {
                Ok(emb) => return emb.dim(),
                Err(e) => {
                    // Loud, because the alternative is a user believing
                    // they are on the better embedder when they are not.
                    // The store records its own embedder identity in
                    // `meta`, so which one was used stays inspectable
                    // after the fact rather than being guesswork.
                    tracing::warn!(
                        target: "yantrikdb::embedder",
                        error = %e,
                        default_model = DEFAULT_NEW_STORE_EMBEDDER,
                        "could not obtain the default embedder (offline?); creating this \
                         store with the bundled potion-base-2M at {} dims instead. Retrieval \
                         on large stores is measurably worse — to switch later you must \
                         re-embed, since dimension is fixed at creation.",
                        crate::embedder::BUNDLED_EMBEDDER_DIM
                    );
                }
            }
        }
        crate::embedder::BUNDLED_EMBEDDER_DIM
    }

    /// Embedding dimension an existing database already holds, if any.
    ///
    /// **A STORED VECTOR IS THE AUTHORITY, NOT THE RECORDED IDENTITY.**
    /// That ordering is not fussiness — it was measured on a live store.
    /// The `meta` embedder identity records what the engine had ATTACHED
    /// the first time it produced a vector, which is not necessarily what
    /// produced the vectors in the file: a caller that embeds externally
    /// and passes vectors to `record()` never stamps an identity, but any
    /// incidental `embed()` call stamps the attached model anyway. A real
    /// 5,050-record production store was found claiming
    /// `embedder_dim = 64 / potion-base-2M` while holding 1536-byte
    /// (384-dim) MiniLM vectors. Trusting that row would have opened a
    /// 384-dim database at 64 dims — exactly the silent index corruption
    /// this function exists to prevent.
    ///
    /// The identity is still used when the file holds no vectors to
    /// measure, where it is the only evidence available and cannot
    /// contradict anything.
    #[cfg(feature = "bundled-embedder")]
    fn detect_existing_dim(db_path: &str) -> Option<usize> {
        if db_path.is_empty() || db_path.starts_with(':') {
            return None; // in-memory databases are always new
        }
        if !std::path::Path::new(db_path).exists() {
            return None;
        }
        let conn = rusqlite::Connection::open_with_flags(
            db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )
        .ok()?;

        let measured = conn
            .query_row(
                "SELECT length(embedding) FROM memories WHERE embedding IS NOT NULL LIMIT 1",
                [],
                |r| r.get::<_, i64>(0),
            )
            .ok()
            .map(|bytes| bytes as usize / std::mem::size_of::<f32>())
            .filter(|d| *d > 0);
        let claimed = match Self::read_embedder_identity(&conn) {
            Ok(Some((_, _, dim))) if dim > 0 => Some(dim),
            _ => None,
        };

        match (measured, claimed) {
            (Some(m), Some(c)) if m != c => {
                // Surfaced rather than silently reconciled: the store's
                // provenance record is wrong, which also means pack
                // mounting and any provenance gate are reasoning from a
                // false premise. Opening at `m` keeps the data readable.
                tracing::warn!(
                    target: "yantrikdb::embedder",
                    measured_dim = m,
                    recorded_dim = c,
                    "database records an embedder dim that disagrees with its own vectors; \
                     opening at the measured width. The recorded identity is not describing \
                     the vectors in this file — check how they were produced before relying \
                     on provenance checks or mounting packs against it."
                );
                Some(m)
            }
            (Some(m), _) => Some(m),
            (None, c) => c,
        }
    }

    /// **Saga task 20 Slice C** — replace the engine's current embedder
    /// with one downloaded from
    /// [`yantrikos/yantrikdb-models`](https://github.com/yantrikos/yantrikdb-models).
    /// Available in default + `embedder-download` builds; compiles out
    /// when neither feature is on.
    ///
    /// Known names (registry hardcoded per release for SHA-256 pinning):
    /// - `"potion-base-8M"`  — 256-dim, ~92% MiniLM, ~28 MB tarball
    /// - `"potion-base-32M"` — 512-dim, ~95% MiniLM, ~121 MB tarball
    ///
    /// On first call this fetches the tarball, verifies its SHA-256
    /// against a constant pinned at compile time, extracts to
    /// `dirs::cache_dir() / "yantrikdb" / "models" /`, and loads via
    /// `model2vec-rs`. Subsequent calls (this process or any other
    /// against the same cache dir) hit the cache and skip the network.
    ///
    /// **Dimension contract.** The named model's output dim must match
    /// the engine's `embedding_dim` set at `YantrikDB::new(path, dim)`.
    /// Mismatch is rejected to prevent silent vector-index corruption.
    ///
    /// **Errors.** Returns `Error::InvalidInput` for: unknown name,
    /// network failure, SHA-256 mismatch, dim mismatch, or filesystem
    /// errors. The engine's existing embedder (if any) is preserved on
    /// error — `set_embedder_named` is atomic.
    #[cfg(feature = "embedder-download")]
    pub fn set_embedder_named(&mut self, name: &str) -> Result<()> {
        use crate::embedder::DownloadedEmbedder;
        let downloaded = DownloadedEmbedder::fetch(name)?;
        if downloaded.dim() != self.embedding_dim() {
            return Err(crate::error::YantrikDbError::InvalidInput(format!(
                "embedder {name:?} dim={} but engine was opened with dim={}; \
                 either reopen with `YantrikDB::new(path, {})` or pick a \
                 differently-dimensioned named embedder",
                downloaded.dim(),
                self.embedding_dim(),
                downloaded.dim(),
            )));
        }
        self.set_embedder(Box::new(downloaded))?;
        Ok(())
    }

    /// Create a new YantrikDB instance with an explicit actor_id (for sync tests).
    pub fn new_with_actor(db_path: &str, embedding_dim: usize, actor_id: &str) -> Result<Self> {
        let mut db = Self::open(db_path, embedding_dim, Some(actor_id.to_string()), None)?;
        Self::finish_construction(&mut db);
        Ok(db)
    }

    /// Create a new encrypted YantrikDB instance.
    ///
    /// The 32-byte `master_key` is used to wrap/unwrap a per-database Data Encryption Key (DEK).
    /// All text, metadata, and embedding fields are encrypted at rest using AES-256-GCM.
    /// In-memory indexes operate on plaintext for full query performance.
    pub fn new_encrypted(
        db_path: &str,
        embedding_dim: usize,
        master_key: &[u8; 32],
    ) -> Result<Self> {
        let mut db = Self::open(db_path, embedding_dim, None, Some(master_key))?;
        Self::finish_construction(&mut db);
        Ok(db)
    }

    /// **Saga task 20.** When the `bundled-embedder` feature is on (default),
    /// attach the engine's own `BundledEmbedder` so `record_text()` and
    /// `recall_text()` work out of the box. Compiles to a no-op under
    /// `--no-default-features` — slim deployments must call `set_embedder()`
    /// explicitly. The auto-attach is a no-op when the engine's
    /// `embedding_dim` does not match the bundled embedder's dim, so a
    /// caller running with a non-default dim sees `NoEmbedder` until they
    /// wire their own (avoids silent dim-mismatch corruption).
    #[allow(unused_variables)]
    /// Attach the bundled embedder, then re-mount installed packs.
    ///
    /// Order matters and is not incidental: mounting proves a pack shares
    /// this database's embedding space, and on an empty database that
    /// proof comes from the *attached* embedder. Re-mounting before the
    /// embedder is attached would refuse every pack on a fresh install.
    fn finish_construction(db: &mut Self) {
        Self::auto_attach_bundled_embedder(db);
        db.remount_installed();
        // 0.13.2 security migration: seal oplog payloads written before
        // the fix. Runs on every open of an encrypted database, is a
        // no-op once healed (the WHERE clause skips marked rows) and a
        // no-op on plaintext databases. Best-effort at the call site
        // for the same reason every other open-time migration is —
        // a failure must not make an existing database unopenable —
        // but it warns loudly, and `oplog_plaintext_rows()` lets an
        // operator check rather than assume.
        if let Err(e) = db.migrate_oplog_payload_encryption() {
            tracing::error!(
                error = %e,
                "oplog payload encryption migration FAILED — pre-0.13.2 plaintext \
                 may remain on disk; see oplog_plaintext_rows()"
            );
        }
    }

    fn auto_attach_bundled_embedder(db: &mut Self) {
        // A store opened at the downloadable default's dimension gets that
        // model attached here rather than in `with_default`, so the
        // attach-then-remount order above holds for it too: a pack proves
        // it shares this database's embedding space against the ATTACHED
        // embedder, so attaching after remount would refuse every pack on
        // a fresh install. Cached after first fetch, so this is not a
        // per-open network hit; on failure the engine simply comes up
        // without an embedder, which is the pre-existing behaviour for any
        // dimension it cannot serve.
        #[cfg(feature = "embedder-download")]
        {
            use crate::embedder::DownloadedEmbedder;
            let bundled_dim = {
                #[cfg(feature = "bundled-embedder")]
                {
                    crate::embedder::BUNDLED_EMBEDDER_DIM
                }
                #[cfg(not(feature = "bundled-embedder"))]
                {
                    usize::MAX
                }
            };
            // Check the registry's declared dim FIRST — it is a compile-time
            // constant. Fetching to discover the dim would put a network
            // attempt on every open of any store the default cannot serve
            // (a 384-dim MiniLM store, say).
            if db.embedding_dim() != bundled_dim
                && DownloadedEmbedder::registry_dim(DEFAULT_NEW_STORE_EMBEDDER)
                    == Some(db.embedding_dim())
            {
                if let Ok(emb) = DownloadedEmbedder::fetch(DEFAULT_NEW_STORE_EMBEDDER) {
                    let _ = db.set_embedder(Box::new(emb));
                    return;
                }
            }
        }
        #[cfg(feature = "bundled-embedder")]
        {
            use crate::embedder::{BundledEmbedder, BUNDLED_EMBEDDER_DIM};
            if db.embedding_dim() == BUNDLED_EMBEDDER_DIM {
                // set_embedder returns Result post-#41 (mode-aware
                // refactor). Auto-attach is best-effort — if it fails
                // for any reason (currently only dim mismatch, but
                // that's already gated by the if above) we proceed
                // without an embedder and the user can wire one
                // manually. Failure here is not catastrophic.
                let _ = db.set_embedder(Box::new(BundledEmbedder::new()));
            }
        }
    }

    fn open(
        db_path: &str,
        embedding_dim: usize,
        actor_id: Option<String>,
        master_key: Option<&[u8; 32]>,
    ) -> Result<Self> {
        // Stage-tag every SQL call in the open path (issue #146). A
        // truncated-statement parse error reaches us as a bare
        // `SqliteFailure(_, "incomplete input")` — SQLite reports the
        // truncation at the end of input, `sqlite3_error_offset()` is -1
        // there, so rusqlite never builds the SQL-carrying variant. The
        // one observed occurrence therefore named nothing. These tags make
        // the next one name its stage.
        fn at<T>(stage: &str, r: std::result::Result<T, rusqlite::Error>) -> Result<T> {
            r.map_err(|source| YantrikDbError::DatabaseAt {
                stage: stage.to_owned(),
                source,
            })
        }
        // Same, for callees that already return the crate error: re-tag
        // only the untagged `Database` case, pass everything else through.
        fn rewrap<T>(stage: &str, r: Result<T>) -> Result<T> {
            r.map_err(|e| match e {
                YantrikDbError::Database(source) => YantrikDbError::DatabaseAt {
                    stage: stage.to_owned(),
                    source,
                },
                other => other,
            })
        }

        let conn = at("open", Connection::open(db_path))?;

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
        at(
            "pragmas",
            conn.execute_batch(
                "PRAGMA journal_mode=WAL; \
                 PRAGMA synchronous=NORMAL; \
                 PRAGMA foreign_keys=ON; \
                 PRAGMA busy_timeout=5000; \
                 PRAGMA wal_autocheckpoint=1000;",
            ),
        )?;

        // Verify critical pragmas actually took effect. SQLite silently
        // ignores some pragmas in certain modes (e.g. journal_mode on
        // read-only or in-memory databases). Log a warning if any mismatch.
        let actual_journal: String = conn
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap_or_default();
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

        // **v0.10 Item 4a.4 (sol) — an unambiguous "brand new database" signal.**
        // `get_schema_version` collapses a query FAILURE or a missing key into
        // `None`, so an EXISTING database whose `schema_version` row is missing
        // or unreadable would be misclassified as fresh and handed the strict
        // fresh defaults — exactly the upgrade break the migration model exists
        // to prevent. Ask the real question instead ("did this database have any
        // user tables before we initialized it?"), evaluated BEFORE SCHEMA_SQL
        // runs below. On any error, assume NOT empty: an unreadable database is
        // treated as pre-existing, so we fail toward the LENIENT/back-compatible
        // default rather than toward breaking a live caller.
        let db_was_empty: bool = conn
            .query_row(
                "SELECT COUNT(*) = 0 FROM sqlite_master \
                 WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(false);

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
            (25, MIGRATE_V25_TO_V26),
            (26, MIGRATE_V26_TO_V27),
            (27, MIGRATE_V27_TO_V28),
            (28, MIGRATE_V28_TO_V29),
            (29, MIGRATE_V29_TO_V30),
            (30, MIGRATE_V30_TO_V31),
            (31, MIGRATE_V31_TO_V32),
            (32, MIGRATE_V32_TO_V33),
            (33, MIGRATE_V33_TO_V34),
            (34, MIGRATE_V34_TO_V35),
            (35, MIGRATE_V35_TO_V36),
            (36, MIGRATE_V36_TO_V37),
            (37, MIGRATE_V37_TO_V38),
            // v38-v40 were code-only. v41 voids fits made against the old
            // meaning of `f_decay` — see MIGRATE_V40_TO_V41.
            (40, MIGRATE_V40_TO_V41),
            (41, MIGRATE_V41_TO_V42),
            (42, MIGRATE_V42_TO_V43),
            (44, MIGRATE_V44_TO_V45),
            (45, MIGRATE_V45_TO_V46),
            (46, MIGRATE_V46_TO_V47),
            (47, MIGRATE_V47_TO_V48),
            (48, MIGRATE_V48_TO_V49),
            (49, MIGRATE_V49_TO_V50),
            (50, MIGRATE_V50_TO_V51),
        ];
        if let Some(v) = existing_version {
            for &(from_v, sql) in migrations {
                if v <= from_v {
                    rewrap(
                        &format!("migration v{}->v{}", from_v, from_v + 1),
                        Self::run_migration_idempotent(&conn, sql),
                    )?;
                }
            }
        }

        at("schema_sql", conn.execute_batch(SCHEMA_SQL))?;

        // **v49 entity_name_norm backfill — Rust, not SQL.** The reviewer
        // finding behind v49: `recall_thread` resolved requested entity
        // names by scanning `SELECT DISTINCT entity_name FROM
        // memory_entities` and Unicode-lowercasing EVERY name in Rust per
        // request — O(V) over the global entity vocabulary, across
        // namespaces, on every call. The persisted key retires that scan,
        // but MIGRATE_V48_TO_V49 cannot backfill it in SQL: LOWER() is
        // ASCII-only and would diverge from crate::graph::tokenize's
        // Unicode lowercasing on non-ASCII names. So the engine backfills
        // here, post-migration, in KEYSET-PAGED batches: a large upgraded
        // store must never materialize its whole entity join in RAM, so
        // rows are fetched 10k at a time by ascending rowid and each batch
        // commits in its own transaction. Crash-safe and idempotent:
        // committed rows are no longer NULL and are never revisited, and
        // on a store with nothing to do (every open after the first) the
        // loop is a single indexed probe.
        //
        // Index tradeoff, documented deliberately: MIGRATE_V48_TO_V49
        // creates idx_memory_entities_norm BEFORE this backfill runs, so
        // the one-time backfill pays per-row index maintenance. Creating
        // the index after the backfill would save that churn, but would
        // split the index's existence across two owners (migration SQL vs
        // engine code) and complicate the idempotent-replay contract —
        // run_migration_idempotent re-runs the migration wholesale on
        // rewound stores — so the migration keeps the CREATE INDEX.
        {
            const BACKFILL_BATCH: i64 = 10_000;
            let mut last_rowid: i64 = 0;
            loop {
                let batch: Vec<(i64, String)> = at("entity_norm_backfill", {
                    (|| {
                        let mut stmt = conn.prepare(
                            "SELECT rowid, entity_name FROM memory_entities \
                             WHERE entity_name_norm IS NULL AND rowid > ?1 \
                             ORDER BY rowid LIMIT ?2",
                        )?;
                        let rows = stmt.query_map(params![last_rowid, BACKFILL_BATCH], |r| {
                            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
                        })?;
                        rows.collect::<std::result::Result<Vec<_>, _>>()
                    })()
                })?;
                let Some(&(batch_last, _)) = batch.last() else {
                    break;
                };
                let tx = at("entity_norm_backfill", conn.unchecked_transaction())?;
                {
                    let mut update = at(
                        "entity_norm_backfill",
                        tx.prepare(
                            "UPDATE memory_entities SET entity_name_norm = ?1 \
                             WHERE rowid = ?2",
                        ),
                    )?;
                    for (rowid, name) in &batch {
                        at(
                            "entity_norm_backfill",
                            update.execute(params![
                                crate::engine::thread::normalize_entity_name(name),
                                rowid
                            ]),
                        )?;
                    }
                }
                at("entity_norm_backfill", tx.commit())?;
                last_rowid = batch_last;
            }
        }

        // **v50 source_turn backfill — Rust, not SQL** (audit trap 3: no
        // unbounded UPDATE in the migration). MIGRATE_V49_TO_V50 only adds
        // the column + index + triggers; the ENGINE repairs here,
        // post-migration, by looping the SAME full-recompute core the
        // maintenance op uses (thread::source_turn_repair_batch) in 10k-row
        // transactions: EVERY row beyond the epoch-stamped resumable cursor
        // is compared against the ONE shared extractor
        // (engine::thread::extract_source_turn) over its current metadata
        // and rewritten in BOTH directions — including resetting a stale
        // non-NULL column back to NULL. There is no "committed rows are
        // never revisited" shortcut: a raw SQL write can mutate metadata
        // behind any previously-stamped row, which is exactly why the
        // schema triggers bump the invalidation epoch and stale the cursor
        // (a NULL-only fill would certify those rows wrong — reviewer
        // blocker 1 on this PR's build).
        // Rows whose metadata cannot be parsed at rest (encrypted blobs)
        // fall back per repair-core rules; store-level completeness is
        // tracked by the meta 'source_turn_backfill_complete' marker
        // (converged option b):
        //   - fresh stores (no prior schema version): marker set '1'
        //     immediately — every future row is stamped at write.
        //   - unencrypted stores: marker set '1' when a repair pass drains
        //     with its cursor epoch still current; a marker staled to '0'
        //     by raw SQL (the schema triggers) is healed by re-running
        //     this same full recompute on the next open.
        //   - encrypted stores: marker set '0' at (post-)migration; only
        //     maintain_source_turn_backfill's decrypt-and-stamp completion
        //     sets it '1'. Lazy write-time stamping continues but NEVER
        //     sets the marker.
        // The encryption probe checks BOTH the passed key and the persisted
        // 'encryption_enabled' meta: an encrypted DB opened without a key
        // fails later in this constructor, and the marker must not have
        // been set '1' by then on the strength of "no key was passed".
        {
            let is_encrypted_store = master_key.is_some()
                || matches!(
                    rewrap(
                        "source_turn_backfill",
                        Self::get_meta(&conn, "encryption_enabled")
                    )?
                    .as_deref(),
                    Some("1")
                );
            let marker = rewrap(
                "source_turn_backfill",
                Self::get_meta(&conn, crate::engine::thread::SOURCE_TURN_MARKER_KEY),
            )?;
            let set_marker = |value: &str| -> Result<()> {
                at(
                    "source_turn_backfill",
                    conn.execute(
                        "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
                        params![crate::engine::thread::SOURCE_TURN_MARKER_KEY, value],
                    ),
                )?;
                Ok(())
            };
            if existing_version.is_none() {
                // Fresh store: nothing predates the stamping writers.
                if marker.is_none() {
                    set_marker("1")?;
                }
            } else if marker.as_deref() != Some("1") {
                if is_encrypted_store {
                    if marker.is_none() {
                        set_marker("0")?;
                    }
                } else {
                    // Full RECOMPUTE (reviewer blocker — never a NULL-only
                    // fill: raw SQL can change a turn 5->7 or remove it,
                    // leaving a stale NON-NULL scalar a fill would skip),
                    // through the ONE shared repair core the maintenance
                    // op also uses: keyset-paged 10k batches, one
                    // transaction each, resumable via the epoch-stamped
                    // cursor (a crash mid-loop resumes; a raw write
                    // invalidates the cursor and restarts the scan). The
                    // core's completion — a drained full pass — is what
                    // sets the marker '1'; plaintext parse happens through
                    // the identity decrypt (this branch is unencrypted).
                    loop {
                        let progress = rewrap(
                            "source_turn_backfill",
                            crate::engine::thread::source_turn_repair_batch(
                                &conn,
                                |stored| Ok(stored.to_string()),
                                10_000,
                            ),
                        )?;
                        if progress.complete {
                            break;
                        }
                    }
                }
            }
        }

        // Populate seed substitution categories (idempotent)
        rewrap(
            "seed_categories",
            crate::distributed::seed_categories::populate_seed_categories(&conn),
        )?;

        // RFC 008 M5b: seed move_type_registry + inference_basis_registry
        // with canonical vocabulary (idempotent INSERT OR IGNORE).
        rewrap(
            "seed_registries",
            crate::engine::moves::seed_registries_inner(&conn),
        )?;

        // Set schema version — never downgrade.
        //
        // **v0.7.3 migration-resilience fix.** Previously this unconditionally
        // wrote SCHEMA_VERSION, which meant a single accidental run of an
        // older binary against a newer DB (rollback during incident response,
        // testing an older release, container image swap) would silently
        // rewind meta.schema_version while leaving the on-disk schema at the
        // higher version. The next forward upgrade then re-ran already-applied
        // migrations (e.g. ALTER TABLE oplog ADD COLUMN embedding) and
        // tripped on "duplicate column name". Diagnosed via yantrikdb-server
        // homelab v0.8.13 cluster upgrade failure (msg 3467c556).
        //
        // MAX-stamp guarantees forward-only progress on the version meta even
        // if the running binary is older than the on-disk schema. Combined
        // with run_migration_idempotent below, both prevents new occurrences
        // (forward) and heals existing corrupted-meta deployments (replay).
        let stamp = std::cmp::max(existing_version.unwrap_or(0), SCHEMA_VERSION);
        at(
            "version_stamp",
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?1)",
                params![stamp.to_string()],
            ),
        )?;

        // **v0.10 Item 1.** Fresh installs default to the status-led read
        // path: superseded records are excluded from recall eligibility.
        // `existing_version` is None only when the meta table didn't exist
        // before this open — i.e. a brand-new database. Migrated/legacy
        // DBs keep include-everything behavior until the operator opts in
        // (set_status_read_policy); the stats() adoption-nudge counter
        // shows them what the policy would have excluded. INSERT OR
        // IGNORE keeps any operator-set value authoritative.
        if existing_version.is_none() {
            at(
                "fresh_defaults",
                conn.execute(
                    "INSERT OR IGNORE INTO meta (key, value) \
                     VALUES ('status_read_policy', 'exclude_superseded')",
                    [],
                ),
            )?;
        }

        // **v0.10 Item 4a.4 — anti-laundering gate mode, backward-compat by
        // migration path (same shape as Item 1).** FRESH installs default to
        // `enforce` (new users protected: a write with internally inconsistent
        // provenance is refused). MIGRATED/legacy installs default to `warn`:
        // the gate runs and increments the `provenance_flagged_since_boot`
        // stats() nudge, but NEVER refuses — so existing callers are not broken
        // on upgrade. `set_provenance_gate_mode` is the durable opt-in. INSERT
        // OR IGNORE keeps any operator-set value authoritative across opens.
        // Uses `db_was_empty` (a real emptiness check), NOT
        // `existing_version.is_none()` — the latter misclassifies an existing DB
        // with an unreadable/missing schema_version as fresh and would hand it
        // `enforce` (sol 4a.4).
        let default_gate_mode = if db_was_empty { "enforce" } else { "warn" };
        at(
            "fresh_defaults",
            conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('provenance_gate_mode', ?1)",
                params![default_gate_mode],
            ),
        )?;

        // **v28 (issue #41 brainstorm-4 §6).** Seed meta.active_generation
        // on first install. INSERT OR IGNORE preserves the durable
        // value on subsequent opens — reembed Phase-2's swap
        // transaction is the only path that mutates it. If a fresh
        // install runs without ever reembedding, the row stays '0'
        // for the engine's entire lifetime, and pre-v28 rows whose
        // embedding_generation IS NULL are correctly treated as
        // "covered by generation 0."
        at(
            "fresh_defaults",
            conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('active_generation', '0')",
                [],
            ),
        )?;

        // Resolve actor_id: explicit > stored in meta > generate new
        let actor_id = if let Some(id) = actor_id {
            at(
                "actor_id",
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('actor_id', ?1)",
                    params![id],
                ),
            )?;
            id
        } else {
            match rewrap("actor_id", Self::get_meta(&conn, "actor_id"))? {
                Some(id) => id,
                None => {
                    let id = crate::id::new_id();
                    at(
                        "actor_id",
                        conn.execute(
                            "INSERT OR REPLACE INTO meta (key, value) VALUES ('actor_id', ?1)",
                            params![id],
                        ),
                    )?;
                    id
                }
            }
        };

        // **v0.10 Item 4a.4 — origin guard stays OPT-IN.** Unlike the local
        // provenance gate (which defaults to enforce for fresh installs), the
        // replication ingress guard is a deployment-topology declaration: a
        // fresh DB may legitimately be joining a multi-writer cluster, and
        // auto-claiming self-authority would break bidirectional sync. A
        // deployment that has DECLARED itself single-writer calls
        // `set_authoritative_origin(self.actor_id())` to activate the guard
        // (recommended in the single-writer deploy docs; multi-origin is Item
        // 4b). This keeps existing AND new multi-master deployments working.

        // **v28 (issue #41 brainstorm-4 §6).** Read the durable
        // active SearchState generation. Defaults to 0 if missing —
        // covers both fresh installs (the INSERT OR IGNORE above
        // wrote '0') and pre-v28 DBs that haven't been touched by
        // the v28 migration yet (shouldn't happen — migration ran
        // above — but defensive).
        let active_generation: u64 = rewrap(
            "reembed_recovery",
            Self::get_meta(&conn, "active_generation"),
        )?
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

        // **Layer 7 — crash recovery for in-flight reembed.**
        //
        // If `meta.reembed_state` is set, the engine crashed mid-
        // reembed. Decide what to do based on the durable
        // `meta.active_generation`:
        //
        // - If `active_generation < in_flight_generation`: the SQL
        //   swap transaction (Phase 2 step 5) did NOT commit before
        //   the crash. The staging columns (`memories.embedding_new`
        //   + `embedding_new_model`) may be partially populated; we
        //   discard them and clear `meta.reembed_state`. The next
        //   `db.reembed(target_name)` call starts fresh and
        //   overwrites whatever staging survived.
        //
        // - If `active_generation >= in_flight_generation`: the SQL
        //   swap DID commit; the in-memory SearchState publish
        //   (step 6) is what was lost. SQL is durably at the new
        //   generation. The SearchState is rebuilt at the new
        //   generation by the standard open path. Staging columns
        //   should already be cleared by the swap transaction, but
        //   we defensively clear any leftover (the in-memory
        //   `apply_pending_ops_once` / Layer 5 path is fine here:
        //   any queued ops with embedding_model NOT NULL get
        //   re-encoded under the new embedder via the standard
        //   drain).
        //
        // The decision is durable + idempotent (re-running open()
        // produces the same result). An audit event is written to
        // reembed_events so operators can see "this reembed crashed
        // and was recovered as discarded / completed".
        let reembed_recovery_summary: Option<String> = {
            let in_flight: Option<(u64, String)> = {
                let payload_json: Option<String> = conn
                    .query_row(
                        "SELECT value FROM meta WHERE key = 'reembed_state'",
                        [],
                        |row| row.get::<_, String>(0),
                    )
                    .ok();
                payload_json.and_then(|s| {
                    let v: serde_json::Value = serde_json::from_str(&s).ok()?;
                    let g = v.get("generation")?.as_u64()?;
                    let phase = v
                        .get("phase")
                        .and_then(|p| p.as_str())
                        .unwrap_or("Probing")
                        .to_string();
                    Some((g, phase))
                })
            };

            if let Some((in_flight_gen, in_flight_phase)) = in_flight {
                let recovery_event_ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0);

                if active_generation < in_flight_gen {
                    // SQL swap didn't commit. Discard staging.
                    at(
                        "reembed_recovery",
                        conn.execute(
                            "UPDATE memories SET embedding_new = NULL, \
                             embedding_new_model = NULL WHERE embedding_new IS NOT NULL",
                            [],
                        ),
                    )?;
                    at(
                        "reembed_recovery",
                        conn.execute("DELETE FROM meta WHERE key = 'reembed_state'", []),
                    )?;
                    let evt_payload = serde_json::json!({
                        "recovery": "discarded_staging",
                        "reason": format!(
                            "crash at phase {in_flight_phase}; SQL swap not committed \
                             (active_generation={active_generation} < \
                             in_flight_generation={in_flight_gen})"
                        ),
                        "active_generation_after": active_generation,
                    });
                    at(
                        "reembed_recovery",
                        conn.execute(
                            "INSERT INTO reembed_events (generation, phase, timestamp, payload_json) \
                             VALUES (?1, ?2, ?3, ?4)",
                            params![
                                in_flight_gen as i64,
                                "Aborted",
                                recovery_event_ts,
                                serde_json::to_string(&evt_payload)?,
                            ],
                        ),
                    )?;
                    Some(format!(
                        "discarded_staging (in-flight gen {in_flight_gen} phase {in_flight_phase})"
                    ))
                } else {
                    // SQL swap committed before crash. SearchState will
                    // rebuild at the new generation (active_generation
                    // read above). Defensive: clear any staging
                    // leftover; the swap transaction normally clears
                    // it but we don't trust a crashed transaction.
                    at(
                        "reembed_recovery",
                        conn.execute(
                            "UPDATE memories SET embedding_new = NULL, \
                             embedding_new_model = NULL WHERE embedding_new IS NOT NULL",
                            [],
                        ),
                    )?;
                    at(
                        "reembed_recovery",
                        conn.execute("DELETE FROM meta WHERE key = 'reembed_state'", []),
                    )?;
                    let evt_payload = serde_json::json!({
                        "recovery": "completed_durable",
                        "reason": format!(
                            "crash at phase {in_flight_phase}; SQL swap committed \
                             (active_generation={active_generation} >= \
                             in_flight_generation={in_flight_gen}); SearchState \
                             rebuilt at new generation"
                        ),
                        "active_generation_after": active_generation,
                    });
                    at(
                        "reembed_recovery",
                        conn.execute(
                            "INSERT INTO reembed_events (generation, phase, timestamp, payload_json) \
                             VALUES (?1, ?2, ?3, ?4)",
                            params![
                                in_flight_gen as i64,
                                "Completed",
                                recovery_event_ts,
                                serde_json::to_string(&evt_payload)?,
                            ],
                        ),
                    )?;
                    Some(format!(
                        "completed_durable (gen {in_flight_gen} phase {in_flight_phase})"
                    ))
                }
            } else {
                None
            }
        };
        if let Some(summary) = &reembed_recovery_summary {
            tracing::warn!(
                target: "yantrikdb::reembed::recovery",
                summary = %summary,
                "open(): in-flight reembed detected; applied crash-recovery decision"
            );
        }

        // Resolve node_id: stored in meta > generate random
        let node_id: u32 = match rewrap("node_id", Self::get_meta(&conn, "node_id"))? {
            Some(s) => s.parse().unwrap_or_else(|_| {
                let id: u32 = rand::thread_rng().gen();
                id
            }),
            None => {
                let id: u32 = rand::thread_rng().gen();
                at(
                    "node_id",
                    conn.execute(
                        "INSERT OR REPLACE INTO meta (key, value) VALUES ('node_id', ?1)",
                        params![id.to_string()],
                    ),
                )?;
                id
            }
        };

        // Initialize encryption (envelope pattern: master_key wraps DEK)
        let enc = if let Some(mk) = master_key {
            let provider = match rewrap("encryption_meta", Self::get_meta(&conn, "encrypted_dek"))?
            {
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
                    at(
                        "encryption_meta",
                        conn.execute(
                            "INSERT OR REPLACE INTO meta (key, value) VALUES ('encrypted_dek', ?1)",
                            params![wrapped_b64],
                        ),
                    )?;
                    at(
                        "encryption_meta",
                        conn.execute(
                            "INSERT OR REPLACE INTO meta (key, value) VALUES ('encryption_enabled', '1')",
                            [],
                        ),
                    )?;
                    EncryptionProvider::from_dek(&dek)
                }
            };
            Some(provider)
        } else {
            // Verify we're not opening an encrypted DB without a key
            if rewrap(
                "encryption_meta",
                Self::get_meta(&conn, "encryption_enabled"),
            )?
            .as_deref()
                == Some("1")
            {
                return Err(YantrikDbError::Encryption(
                    "database is encrypted but no master_key provided".into(),
                ));
            }
            None
        };

        let scoring_cache = rewrap("load_scoring_cache", Self::load_scoring_cache(&conn))?;
        let vec_index = rewrap(
            "build_vec_index",
            Self::build_vec_index_with_enc(&conn, embedding_dim, enc.as_ref()),
        )?;
        // C5b: heal possessive-pollution BEFORE the graph index builds,
        // so the very first build folds phantom entities into their
        // canonicals. Idempotent and cheap; best-effort by design (a
        // failed census must never fail an open).
        let _ = graph_ops::migrate_possessive_aliases(&conn);
        let graph_index = rewrap("graph_index", GraphIndex::build_from_db(&conn))?;

        // Load active sessions from DB
        let active_sessions = rewrap("load_sessions", Self::load_active_sessions(&conn))?;

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
            let rc = at("read_pool_open", Connection::open(db_path))?;
            at(
                "read_pool_pragmas",
                rc.execute_batch(
                    "PRAGMA journal_mode=WAL; \
                     PRAGMA synchronous=NORMAL; \
                     PRAGMA foreign_keys=ON; \
                     PRAGMA busy_timeout=5000;",
                ),
            )?;
            read_conns.push(Mutex::new(rc));
        }
        if pool_size > 0 {
            tracing::info!(
                pool_size,
                "yantrikdb-core: read connection pool initialized"
            );
        }

        // **v0.7.1 perf hotfix.** Boot-time SQL count of pending oplog
        // entries; thereafter the counter is maintained in-memory by
        // log_op_pending (increments) and mark_op_applied (decrements).
        // The partial idx_oplog_pending makes this single boot read O(N_pending)
        // — a fixed cost we pay once, in exchange for never paying it again
        // on the foreground hot path.
        //
        // **Fail-CLOSED (v0.10 Item 4a.6a, sol review).** This used to
        // `.unwrap_or(0)`, which is the wrong direction for a queue ceiling: a
        // failed count would seed the counter at zero, so the engine would believe
        // the ingest queue was empty no matter how many pending ops were really in
        // SQL, and `MAX_PENDING_OPS` would never fire — unbounded ingest, silently.
        // A boot read that cannot be trusted must fail the open, not invent a
        // permissive answer. (Same class as the fail-open defaults in 4a.1 and
        // 4a.4.)
        let initial_pending: i64 = at(
            "oplog_pending_count",
            conn.query_row("SELECT COUNT(*) FROM oplog WHERE applied = 0", [], |row| {
                row.get(0)
            }),
        )?;

        // Build the DeltaIndex once, wrap in Arc, and move it into the
        // initial `SearchState`. After issue #41 brainstorm-4 §1, the
        // SearchState is the only owner of the index — there is no
        // standalone field anymore. Reembed Phase-2 can later publish
        // a brand-new `DeltaIndex` atomically with the rest of the
        // SearchState bundle via `search_state.store(new_state)`.
        let vec_index_arc: std::sync::Arc<crate::vector::delta_index::DeltaIndex> = {
            let delta_max = std::env::var("YANTRIKDB_DELTA_MAX")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(crate::vector::delta_index::DEFAULT_DELTA_MAX);
            let max_dirty_age = std::env::var("YANTRIKDB_MAX_DIRTY_AGE_SECS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map(std::time::Duration::from_secs)
                .unwrap_or(crate::vector::delta_index::DEFAULT_MAX_DIRTY_AGE);
            std::sync::Arc::new(crate::vector::delta_index::DeltaIndex::from_cold_with_age(
                vec_index,
                delta_max,
                max_dirty_age,
            ))
        };

        // v0.10 Item 1: hydrate the cached status read policy from meta.
        // Any value other than the exact 'exclude_superseded' opt-in reads
        // as legacy (missing key on migrated DBs, or an operator writing
        // e.g. 'legacy' to switch the policy back off).
        let exclude_superseded_reads = matches!(
            rewrap(
                "final_meta_reads",
                Self::get_meta(&conn, "status_read_policy")
            )?
            .as_deref(),
            Some("exclude_superseded")
        );

        // Item 4a.4: cache the gate mode. The open-time seed above wrote
        // 'enforce' (fresh) or 'warn' (migrated); a missing key defaults to
        // 'warn' (lenient) defensively.
        // Fail-CLOSED: a malformed persisted mode is a typed error (propagated),
        // never a silent `Off` (sol 4a.4).
        let provenance_gate_mode = crate::provenance::GateMode::parse(
            rewrap(
                "final_meta_reads",
                Self::get_meta(&conn, "provenance_gate_mode"),
            )?
            .as_deref()
            .unwrap_or("warn"),
        )?
        .as_u8();

        // Missing is the v42-upgrade-compatible default. A malformed or zero
        // persisted value fails open() loudly: silently disabling a write-
        // amplification bound is the wrong failure direction.
        rewrap(
            "final_meta_reads",
            Self::synthesis_fanout_cap_from_conn(&conn),
        )?;

        // **Packs / issue #117.** Restore durable embedder identity.
        //
        // Before this read existed, `SearchState::initial` reconstructed
        // provenance as `ExternalOrUnknown` on every open, which made
        // `set_embedder`'s same-dim-different-model guard unreachable
        // across a restart — reopen a database, attach a different
        // 64-dim model, and every recall silently searched one vector
        // space with queries encoded in another. Promoting to `Known`
        // here is what arms that guard, and what lets `mount_pack`
        // prove a pack shares this database's embedding space.
        //
        // A recorded dim that disagrees with the index dim is ignored
        // rather than fatal: it means the identity predates a dim
        // change, and `ExternalOrUnknown` is exactly the honest state
        // for "we cannot prove what built these vectors".
        let persisted_embedder =
            Self::read_embedder_identity(&conn)?.filter(|(_, _, dim)| *dim == embedding_dim);
        // Presence, not dim-match: a stored-but-mismatched identity
        // still means the write path has nothing new to stamp, and
        // re-stamping under a different dim is `reembed`'s job.
        let persisted_embedder_present = rewrap(
            "final_meta_reads",
            Self::get_meta(&conn, pack::META_EMBEDDER_DIGEST),
        )?
        .is_some();

        Ok(Self {
            conn: Mutex::new(conn),
            read_conns,
            read_idx: std::sync::atomic::AtomicUsize::new(0),
            embedding_dim,
            db_path: db_path.to_string(),
            hlc: Mutex::new(HLC::new(node_id)),
            actor_id,
            scoring_cache: RwLock::new(scoring_cache),
            vec_seq: std::sync::atomic::AtomicU64::new(0),
            pending_op_count: std::sync::atomic::AtomicI64::new(initial_pending),
            exclude_superseded_reads: std::sync::atomic::AtomicBool::new(exclude_superseded_reads),
            superseded_served_since_boot: std::sync::atomic::AtomicU64::new(0),
            recall_candidate_cap_bound_since_boot: parking_lot::Mutex::new(
                std::collections::HashMap::new(),
            ),
            recall_candidate_cap_namespace_stats_truncated_since_boot:
                std::sync::atomic::AtomicBool::new(false),
            synthesis_fanout_refused_since_boot: std::sync::atomic::AtomicU64::new(0),
            embedder_window_chars: std::sync::atomic::AtomicUsize::new(0),
            embedder_truncated_writes: std::sync::atomic::AtomicU64::new(0),
            embedder_chunked_writes: std::sync::atomic::AtomicU64::new(0),
            provenance_gate_mode: std::sync::atomic::AtomicU8::new(provenance_gate_mode),
            provenance_flagged_since_boot: std::sync::atomic::AtomicU64::new(0),
            correction_epoch: std::sync::atomic::AtomicU64::new(0),
            visible_seq: dashmap::DashMap::new(),
            visible_seq_cv: parking_lot::Condvar::new(),
            visible_seq_wait_mu: parking_lot::Mutex::new(()),
            graph_index: RwLock::new(graph_index),
            enc,
            embedder: None,
            active_sessions: RwLock::new(active_sessions),
            // Issue #41: WriteRouter starts in Normal state. Reembed
            // is the only path that flips it to Queueing; until then,
            // every record/record_text takes the synchronous path
            // unchanged. Adding the field is a no-op for non-reembed
            // code paths until record() is wired to check the gate.
            write_router: std::sync::Arc::new(crate::engine::write_router::WriteRouter::new()),
            // Issue #41 layer 2: initial SearchState mirrors the
            // legacy embedder/embedding_dim fields. Provenance is
            // ExternalOrUnknown(embedding_dim) until set_embedder*
            // populates it with Known(name, digest, dim) or a future
            // reembed publishes a new bundle. HNSW params (M=16,
            // ef_construction=200, ef_search=50) are the engine
            // defaults; the actual DeltaIndex uses those today. The
            // search_state copy here is the source of truth going
            // forward; the future migration sweep retires the legacy
            // embedding_dim + embedder fields and points all readers
            // here.
            //
            // **v28 (issue #41 brainstorm-4 §6).** Override the
            // initial generation (which SearchState::initial defaults
            // to 0) with `meta.active_generation` read above. This is
            // the durable-linearization-point read at open: if the
            // engine crashed between reembed's SQL swap-commit (which
            // updates meta.active_generation) and the in-memory
            // SearchState publish, open() recovers the correct
            // generation here. Pre-v28 DBs and fresh installs both
            // read 0, preserving existing behavior.
            search_state: arc_swap::ArcSwap::from(std::sync::Arc::new({
                let mut s = crate::engine::reembed::SearchState::initial(
                    embedding_dim,
                    16,
                    200,
                    50,
                    vec_index_arc,
                );
                s.generation = active_generation;
                if let Some((name, digest, dim)) = persisted_embedder {
                    s.index_embedding =
                        crate::engine::reembed::EmbeddingProvenance::Known { name, digest, dim };
                }
                s
            })),
            index_write_lock: parking_lot::Mutex::new(()),
            packs: parking_lot::RwLock::new(Vec::new()),
            embedder_identity_stamped: std::sync::atomic::AtomicBool::new(
                persisted_embedder_present,
            ),
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

    /// Run a migration SQL batch with statement-level idempotency.
    ///
    /// **v0.7.3 / v0.7.8 fix for migration-replay class of bugs.**
    /// `conn.execute_batch` aborts on the first error, so a single ALTER TABLE
    /// ADD COLUMN on a column that already exists fails the whole migration —
    /// even though the rest of the batch (CREATE INDEX IF NOT EXISTS, UPDATE,
    /// etc.) is idempotent and safe to re-run. SQLite has no `IF NOT EXISTS`
    /// for ALTER TABLE ADD COLUMN, so we have to detect the harmless cases at
    /// runtime.
    ///
    /// This helper splits the batch on `;`, executes each statement
    /// individually, and swallows specific errors that mean "the change is
    /// already applied or superseded":
    ///   - `duplicate column name: <X>` — re-running ALTER TABLE ADD COLUMN
    ///     on a column already present (v0.7.3 case: V23→V24 embedding column
    ///     on a rewound-meta DB).
    ///   - `<X> already exists` — re-running CREATE TABLE/INDEX without IF
    ///     NOT EXISTS (defensive; our migrations already use IF NOT EXISTS
    ///     for CREATE).
    ///   - `Cannot add a column to a view` — running an ALTER TABLE on a
    ///     name that's been superseded into a backward-compat VIEW by a
    ///     later migration. Hits when meta is rewound to a version BEFORE
    ///     the rename-to-view (V14→V15 case: edges-as-table got renamed to
    ///     claims and replaced with an edges-as-view in V16→V17, but a DB
    ///     with meta rewound to v14 sees on-disk view+claims state and
    ///     can't ADD COLUMN to the view). Safe to skip because: the view
    ///     exists only if a later migration already moved the underlying
    ///     state past where these columns matter. Issue #10 (2026-05-09).
    ///   - `there is already another table or index with this name: <X>` —
    ///     ALTER TABLE ... RENAME TO target where target already exists
    ///     (V16→V17 case on rewound meta: `ALTER TABLE edges RENAME TO claims`
    ///     fails because claims already exists from a prior application of
    ///     this same migration). Safe to skip because: target exists only
    ///     if rename already happened.
    ///   - `no such column: <X>` — ALTER TABLE ... RENAME COLUMN src TO dst
    ///     where src has already been renamed (V16→V17:
    ///     `RENAME COLUMN edge_id TO claim_id` fails on second run because
    ///     edge_id no longer exists). Safe to skip because: column rename
    ///     already happened. False-positive risk (a real "no such column"
    ///     elsewhere) is bounded by the fact that we only swallow per-
    ///     statement; if a later statement legitimately needs that column,
    ///     it still fails.
    ///   - `no such table: <X>` — DROP/ALTER TABLE on a name that's been
    ///     renamed away (V17→V18 mid-cascade: `DROP TABLE claims` after a
    ///     prior partial run already moved it). Safe with the same bounded
    ///     false-positive argument as no-such-column.
    ///
    /// Any other error propagates. This makes every entry in the migration
    /// chain replay-safe retroactively, healing deployments whose
    /// meta.schema_version was rewound (e.g. by an old-binary downgrade)
    /// without manual intervention.
    ///
    /// Splitting on bare `;` is acceptable here because the migration SQL is
    /// authored in this crate — none of the statements contain `;` inside
    /// string literals. If that changes, switch to a sqlite tokenizer pass.
    ///
    /// **Long-term proper fix** (issue #10 suggestion): refactor the
    /// migration runner to introspect schema state via `PRAGMA table_info`
    /// before each ALTER, only run statements whose target column doesn't
    /// already exist. Larger change; tracked separately. The error-swallow
    /// list is the v0.7.x stopgap that heals existing deployments.
    ///
    /// Diagnosed via yantrikdb-server v0.8.13 cluster upgrade incident
    /// (swarm msg 3467c556 → response fa070846, v0.7.3 commit a5de0f2) and
    /// extended for issue #10 view case in v0.7.8.
    /// Split a migration batch into executable statements using SQLite's
    /// own grammar — `sqlite3_complete()` — instead of a hand-rolled lexer.
    ///
    /// **Issue #146, both generations of the bug.** The original
    /// `batch.split(';')` truncated trigger bodies (`BEGIN stmt; stmt; END`),
    /// producing the `incomplete input` the stage instrumentation caught in
    /// CI. The first fix was a depth-tracking scanner — and cold review
    /// reproduced two holes in it within the hour: a quoted identifier
    /// `"begin"` counted as a keyword and jammed the depth counter (grouping
    /// statements so a swallowed already-exists SILENTLY DROPPED the rest of
    /// the group — migration loss), and `"semi;colon"` identifiers split
    /// mid-name. A partial SQL lexer is the original sin repeated: every
    /// unnamed case is a future #146.
    ///
    /// `sqlite3_complete` is the engine's own answer to "is this a complete
    /// statement?" — it understands triggers, CASE/END, every string and
    /// identifier quoting form, and comments, because it IS SQLite. We scan
    /// forward and emit a statement at each `;` where the accumulated prefix
    /// is complete; anything trailing without a terminator is emitted as-is
    /// (execute_batch surfaces its error, which is correct for a malformed
    /// migration).
    fn split_sql_statements(batch: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut start = 0usize;
        let bytes = batch.as_bytes();
        for i in 0..bytes.len() {
            if bytes[i] == b';' {
                let candidate = &batch[start..=i];
                // sqlite3_complete needs a NUL-terminated C string; a
                // candidate containing an interior NUL cannot be a valid
                // migration statement, so treat it as incomplete.
                if let Ok(c) = std::ffi::CString::new(candidate) {
                    let complete = unsafe { rusqlite::ffi::sqlite3_complete(c.as_ptr()) } != 0;
                    if complete {
                        out.push(candidate.to_string());
                        start = i + 1;
                    }
                }
            }
        }
        let tail = batch[start..].trim();
        if !tail.is_empty() {
            out.push(tail.to_string());
        }
        out
    }

    fn run_migration_idempotent(conn: &Connection, batch: &str) -> Result<()> {
        // Strip `-- ... \n` line comments before splitting on `;`. The
        // naive split otherwise breaks on comment text containing
        // semicolons (e.g. MIGRATE_V21_V22 has "ALTER; we add plain-
        // typed columns" inside a comment which would split the next
        // ALTER mid-line). Migration SQL is authored in this crate; no
        // string literals contain `--`, so a simple per-line truncate
        // at the first `--` is safe.
        let stripped: String = batch
            .lines()
            .map(|line| match line.find("--") {
                Some(idx) => &line[..idx],
                None => line,
            })
            .collect::<Vec<_>>()
            .join("\n");

        for raw in Self::split_sql_statements(&stripped) {
            let stmt = raw.trim();
            if stmt.is_empty() {
                continue;
            }
            // Use execute_batch for the per-statement run because some
            // migration statements are not single-row DDL — V17_V18 has
            // INSERT INTO ... SELECT which rusqlite's execute() rejects
            // with ApiMisuse if it returns rows from a sub-select. Per
            // SQLite semantics execute_batch handles the full statement
            // grammar uniformly.
            match conn.execute_batch(stmt) {
                Ok(_) => {}
                Err(e) => {
                    let msg = e.to_string();
                    let is_idempotent_replay = msg.contains("duplicate column name")
                        || msg.contains("already exists")
                        || msg.contains("Cannot add a column to a view")
                        || msg.contains("there is already another table or index with this name")
                        || msg.contains("no such column")
                        || msg.contains("no such table");
                    if is_idempotent_replay {
                        tracing::debug!(
                            statement = %stmt,
                            error = %msg,
                            "migration: skipping already-applied statement (idempotent replay)"
                        );
                        continue;
                    }
                    // This runner is the ONE place open-path SQL is
                    // derived rather than constant (split on `;`, line
                    // comments stripped) — i.e. the one place a
                    // truncated statement could be of our own making.
                    // A parse error like "incomplete input" carries no
                    // SQL of its own (issue #146), so attach the exact
                    // statement we handed to SQLite.
                    let shown: String = stmt.chars().take(200).collect();
                    return Err(YantrikDbError::DatabaseAt {
                        stage: format!("migration statement `{shown}`"),
                        source: e,
                    });
                }
            }
        }
        Ok(())
    }

    fn load_active_sessions(conn: &Connection) -> Result<HashMap<String, String>> {
        let mut map = HashMap::new();
        // Table may not exist yet during initial schema creation
        let mut stmt = match conn
            .prepare("SELECT namespace, session_id FROM sessions WHERE status = 'active'")
        {
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

    /// **v0.10 Item 1.** Whether the status-led read path is active:
    /// `true` = recall excludes superseded records from eligibility
    /// (fresh-install default), `false` = legacy include-everything
    /// (migrated DBs that haven't opted in yet). Mirrors
    /// `meta.status_read_policy`, cached at open.
    pub fn status_read_policy(&self) -> bool {
        self.exclude_superseded_reads
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// **v0.10 Item 1.** Set the status read policy durably (writes
    /// `meta.status_read_policy`) and update the cached flag. This is
    /// the legacy-database opt-in: after migrating a pre-v0.10 DB,
    /// review `stats().superseded_served_since_boot`, then call
    /// `set_status_read_policy(true)` to switch recall to the
    /// status-led read path. `false` returns to legacy behavior.
    pub fn set_status_read_policy(&self, exclude_superseded: bool) -> Result<()> {
        let value = if exclude_superseded {
            "exclude_superseded"
        } else {
            "legacy"
        };
        self.conn().execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('status_read_policy', ?1)",
            params![value],
        )?;
        self.exclude_superseded_reads
            .store(exclude_superseded, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Maximum verified synthesis generations one evidence record may back
    /// through local admission. Replicated durable writes are never discarded;
    /// `stats().synthesis_fanout_sources_over_cap` exposes those exceptions.
    pub fn synthesis_fanout_cap(&self) -> Result<usize> {
        let conn = self.conn();
        Self::synthesis_fanout_cap_from_conn(&conn)
    }

    pub(crate) fn synthesis_fanout_cap_from_conn(conn: &Connection) -> Result<usize> {
        match Self::get_meta(conn, "synthesis_fanout_cap")? {
            None => Ok(DEFAULT_SYNTHESIS_FANOUT_CAP),
            Some(value) => value
                .parse::<usize>()
                .ok()
                .filter(|cap| *cap > 0 && *cap <= i64::MAX as usize)
                .ok_or_else(|| {
                    crate::error::YantrikDbError::InvalidInput(format!(
                        "meta.synthesis_fanout_cap must be in 1..={}, got {value:?}",
                        i64::MAX
                    ))
                }),
        }
    }

    /// Durably configure the local synthesis fan-out admission ceiling.
    /// Lowering below the current high-water is allowed and blocks new local
    /// generations until invalidation/supersession brings pressure under it.
    pub fn set_synthesis_fanout_cap(&self, cap: usize) -> Result<()> {
        if cap == 0 || cap > i64::MAX as usize {
            return Err(crate::error::YantrikDbError::InvalidInput(format!(
                "synthesis fan-out cap must be in 1..={}",
                i64::MAX
            )));
        }
        let conn = self.conn();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('synthesis_fanout_cap', ?1)",
            params![cap.to_string()],
        )?;
        Ok(())
    }

    /// **v0.10 Item 4a.4.** The active anti-laundering gate mode. Fresh installs
    /// default to `Enforce`, migrated/legacy installs to `Warn` (see open()).
    pub fn provenance_gate_mode(&self) -> crate::provenance::GateMode {
        crate::provenance::GateMode::from_u8(
            self.provenance_gate_mode
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Durable opt-in to a gate mode (the migration path for a legacy DB that
    /// has reviewed `stats().provenance_flagged_since_boot` and is ready to
    /// enforce). Updates both the meta key and the cached atomic.
    /// Note: a write already PAST the gate may still commit after this returns —
    /// the transition is linearized at gate-time, not against in-flight writes.
    /// Treat a mode change as a quiescent-ish configuration action.
    pub fn set_provenance_gate_mode(&self, mode: crate::provenance::GateMode) -> Result<()> {
        // Hold the conn guard ACROSS the meta write AND the cached store (sol
        // 4a.4): releasing it between lets two concurrent setters interleave and
        // leave meta and the cache disagreeing (e.g. meta=warn, cache=enforce).
        let conn = self.conn();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('provenance_gate_mode', ?1)",
            params![mode.as_str()],
        )?;
        self.provenance_gate_mode
            .store(mode.as_u8(), std::sync::atomic::Ordering::Relaxed);
        drop(conn);
        Ok(())
    }

    /// **v0.10 Item 4a.4 — the anti-laundering gate.** Parse the record's
    /// DECLARED provenance and enforce internal consistency per the current
    /// mode. `source` is the caller's source string; `metadata` is the FINAL
    /// merged plaintext metadata — `confidence_basis`, `kind`, and
    /// `override_kind` live there (so the engine `record*` signatures are
    /// unchanged). Runs BEFORE any side effect. In `Enforce` a violation is a
    /// typed `ProvenanceInconsistent` refusal; in `Warn` it is counted
    /// (`provenance_flagged_since_boot`) and allowed; `Off` skips entirely.
    pub(crate) fn gate_provenance(
        &self,
        source: &str,
        metadata: &serde_json::Value,
    ) -> Result<GateVerdict> {
        use crate::provenance::{
            check_provenance_consistency_opt, ClaimKind, ConfidenceBasis, GateMode, Source,
        };
        let mode = self.provenance_gate_mode();
        if mode == GateMode::Off {
            return Ok(GateVerdict::Clean);
        }
        let verdict = (|| -> Result<()> {
            // **`source` is a FREE-FORM public dimension — an unrecognized one
            // is NOT refused; the matrix simply does not bind it.**
            //
            // sol 4a.4 asked for strict parsing (reject `source="inference_v2"`
            // + `kind="fact"` as an alias-bypass). We deliberately do not, for
            // two reasons it could not see from the engine source alone:
            //
            // 1. `source` is a documented FREE-FORM dimension of the public API,
            //    not a closed vocabulary: `tests/test_phases.py` records
            //    `source="manager"` and asserts it round-trips verbatim, right
            //    alongside `domain` / `emotional_state`. The four values in the
            //    schema comment are EXAMPLES; only the one-time v26 backfill
            //    ever coerced legacy junk. Rejecting unknown sources is a
            //    BREAKING change to that contract for every existing caller
            //    labelling records `manager` / `slack` / `paper`.
            // 2. It would buy no protection anyway. sol's own r3/4a.4 analysis
            //    concedes that an internally-consistent LIE (`source="user"` +
            //    `kind="fact"`) is undetectable. A caller willing to alias to
            //    `inference_v2` is equally willing to write `user`, so strict
            //    parsing closes only one variant of a hole that stays wide open
            //    — while breaking honest callers. The gate's documented scope is
            //    DECLARED CONTRADICTIONS, never lies.
            //
            // So: the matrix binds the RECOGNIZED `inference` source; anything
            // else is a label the engine takes no position on.
            let Ok(src) = Source::parse(source) else {
                return Ok(());
            };
            let basis = match metadata.get("confidence_basis").and_then(|v| v.as_str()) {
                Some(b) => Some(ConfidenceBasis::parse(b)?),
                None => None,
            };
            let kind = ClaimKind::parse(metadata.get("kind").and_then(|v| v.as_str()));
            let override_kind = metadata
                .get("override_kind")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            check_provenance_consistency_opt(src, basis.as_ref(), &kind, override_kind)
        })();
        match verdict {
            Ok(()) => Ok(GateVerdict::Clean),
            Err(e) => {
                if mode == GateMode::Enforce {
                    Err(e)
                } else {
                    // Warn: allow the write, and REPORT the flag instead of
                    // counting it here (4a.6b). The gate runs before routing, so
                    // ticking `provenance_flagged_since_boot` at this point
                    // counted writes that were subsequently REJECTED —
                    // inflating the very nudge metric an operator reads to
                    // decide when warn can become enforce. The caller ticks via
                    // [`Self::note_flagged_write_committed`] only after the
                    // write is durable.
                    tracing::warn!(reason = %e, "provenance gate (warn): flagged an inconsistent write");
                    Ok(GateVerdict::Flagged)
                }
            }
        }
    }

    /// **4a.6b — the winner-only half of the warn-mode gate.** Call exactly once
    /// AFTER the flagged write's transaction commits. In-memory since-boot
    /// diagnostic: an unwind between commit and this call loses at most one
    /// tick of a counter that re-seeds at boot — acceptable, unlike the
    /// pre-routing overcount this replaces, which inflated the metric with
    /// writes that never landed.
    pub(crate) fn note_flagged_write_committed(&self, verdict: GateVerdict) {
        if verdict == GateVerdict::Flagged {
            self.provenance_flagged_since_boot
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// v0.10 Item 2 — the last learning-loop report (JSON), if any run
    /// has happened. Interim surface: lifts to typed diagnostics()
    /// fields with Item 5 (commitment recorded in nuron's consumer
    /// review of the Item-2 branch).
    pub fn last_learning_report(&self) -> Result<Option<String>> {
        Self::get_meta(&self.conn(), "last_learning_report")
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
        self.search_state.load().vec_index.delta_max()
    }

    /// Current delta-tier length (live entries + tombstone markers).
    /// Pairs with `delta_max()` for pressure-ratio computation.
    pub fn delta_len(&self) -> usize {
        self.search_state.load().vec_index.delta_len()
    }

    /// Current cold-tier length (entries that have been merged into
    /// the HNSW). Useful for ops dashboards that want to see the
    /// hot/cold split — most reads against a healthy engine should
    /// hit cold rather than the linear delta scan.
    pub fn cold_len(&self) -> usize {
        self.search_state.load().vec_index.cold_len()
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

    // ── Oplog payload encryption (0.13.2 security fix) ──
    //
    // The oplog carries the COMPLETE record payload — text, metadata,
    // and embedding — as JSON, and applied rows are retained because
    // the oplog doubles as the replication stream. Encryption was drawn
    // around the `memories` table, so on an encrypted database every
    // record's plaintext sat on disk in `oplog.payload` while
    // `is_encrypted()` reported true. Found by a canary byte-scan of a
    // raw db file (the scan is now a permanent test); the seam rule
    // again — a value crossed the encryption boundary in a
    // write-ahead projection without declaring which side it was on.
    //
    // Ciphertext is marked so a row can be classified without a key:
    // rows written before this fix are bare JSON (`{`), rows written
    // after are `ENCv1:` + base64. Decode tolerates both so a
    // mixed-vintage oplog reads correctly during and after migration.

    /// Prefix marking an encrypted oplog payload. Chosen so it can
    /// never collide with serde_json output, which always starts `{`.
    pub(crate) const OPLOG_ENC_PREFIX: &'static str = "ENCv1:";

    /// Encrypt an oplog payload string when the database is encrypted.
    /// Plaintext databases pass through unchanged (byte-identical
    /// oplog rows, so replication and digests are unaffected).
    pub(crate) fn encode_oplog_payload(&self, payload_json: &str) -> Result<String> {
        match &self.enc {
            Some(e) => Ok(format!(
                "{}{}",
                Self::OPLOG_ENC_PREFIX,
                e.encrypt_string(payload_json)?
            )),
            None => Ok(payload_json.to_string()),
        }
    }

    /// Decode a stored oplog payload. Marked rows are decrypted;
    /// unmarked rows pass through so pre-fix rows still parse (they
    /// are rewritten by the migration, but a reader must never fail on
    /// one it meets first).
    pub(crate) fn decode_oplog_payload(&self, stored: &str) -> Result<String> {
        match stored.strip_prefix(Self::OPLOG_ENC_PREFIX) {
            Some(b64) => match &self.enc {
                Some(e) => e.decrypt_string(b64),
                // An encrypted payload with no key: the caller cannot
                // proceed, and saying so beats handing back a marker
                // string that would parse as garbage downstream.
                None => Err(YantrikDbError::Encryption(
                    "oplog payload is encrypted but this database was opened without a key".into(),
                )),
            },
            None => Ok(stored.to_string()),
        }
    }

    /// How many oplog rows still hold UNSEALED payloads.
    ///
    /// On an encrypted database this must be 0 after open; anything
    /// else means the migration failed and plaintext remains at rest.
    /// Exposed so an operator can verify rather than assume — the
    /// number declares which side of the seal it was read from.
    ///
    /// **This counts ROWS, not BYTES** (0.13.4). Sealing a row with
    /// `UPDATE` does not erase the page that held the old value:
    /// SQLite frees the page and the plaintext survives in the file
    /// until it is reused or the file is rewritten. Reading `0` here
    /// therefore says "no live row holds plaintext" — it did NOT, in
    /// 0.13.2/0.13.3, say "no plaintext is on disk", and that gap was
    /// the same defect as the bug it was reporting on. Since 0.13.4
    /// the migration rewrites the file (VACUUM + WAL truncate) so the
    /// two statements coincide again; the honest check for any
    /// database is still a raw byte scan.
    pub fn oplog_plaintext_rows(&self) -> Result<usize> {
        let conn = self.conn.lock();
        let n: i64 = conn.query_row(
            "SELECT COUNT(*) FROM oplog WHERE payload NOT LIKE 'ENCv1:%'",
            [],
            |r| r.get(0),
        )?;
        Ok(n as usize)
    }

    /// Rewrite pre-fix plaintext oplog payloads as ciphertext.
    ///
    /// Runs on open for encrypted databases only. Idempotent (already
    /// marked rows are skipped by the WHERE clause), bounded by one
    /// pass over unmarked rows, and best-effort in the sense that it
    /// reports how many rows it healed — but a failure propagates,
    /// because silently leaving plaintext behind is the defect this
    /// exists to remove.
    pub(crate) fn migrate_oplog_payload_encryption(&self) -> Result<usize> {
        if self.enc.is_none() {
            return Ok(0);
        }
        let conn = self.conn.lock();
        let rows: Vec<(String, String)> = {
            let mut stmt =
                conn.prepare("SELECT op_id, payload FROM oplog WHERE payload NOT LIKE 'ENCv1:%'")?;
            let mapped = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
            mapped.collect::<std::result::Result<Vec<_>, _>>()?
        };
        if rows.is_empty() {
            return Ok(0);
        }
        let n = rows.len();
        tracing::warn!(
            rows = n,
            "oplog payload encryption migration STARTING — sealing pre-0.13.2 \
             plaintext rows, then rewriting the file to erase freed pages. \
             The database is unavailable until this completes (measured ~8s \
             per 100k rows plus the VACUUM); this message exists so a long \
             pause reads as progress rather than as a hang."
        );
        let started = std::time::Instant::now();
        for (op_id, plain) in rows {
            let sealed = self.encode_oplog_payload(&plain)?;
            conn.execute(
                "UPDATE oplog SET payload = ?1 WHERE op_id = ?2",
                rusqlite::params![sealed, op_id],
            )?;
        }

        // 0.13.4 — THE SEAL IS NOT THE ERASURE.
        //
        // `UPDATE` writes the ciphertext to a new page and frees the old
        // one; the plaintext survives in the file until that page is
        // reused. Measured on a 111,590-row oplog at CT128 scale: after
        // the loop above, `oplog_plaintext_rows()` read 0 while a raw
        // byte scan still found the canary — a verification surface
        // reporting a guarantee the storage did not provide, which is
        // precisely the defect this migration exists to remove. The
        // migration had the shape of the bug.
        //
        // VACUUM rewrites the database without the freed pages;
        // truncating the WAL afterwards drops the copies the rewrite
        // itself journalled. ORDER MATTERS and the reverse does not
        // work — checkpointing first then vacuuming leaves the residue
        // in the main file (measured both ways).
        conn.execute_batch("VACUUM; PRAGMA wal_checkpoint(TRUNCATE);")?;

        tracing::warn!(
            rows = n,
            elapsed_s = started.elapsed().as_secs_f64(),
            "oplog payload encryption migration COMPLETE — rows sealed and \
             freed pages erased"
        );
        Ok(n)
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

    // ── Embedder integration (issue #41 layer 2: SearchState-derived) ──

    /// Set the text-to-embedding converter (mode-aware per brainstorm-3).
    /// Enables `embed()`, `record_text()`, and `recall_text()`.
    ///
    /// **Behavior change vs pre-#41:** the call now returns `Result<()>`
    /// and rejects the silent-corruption shape:
    /// - Different dim → `Err(ChangeEmbedderDimensionRequiresReembed)`
    /// - Different fingerprint on populated `Known`-provenance DB →
    ///   `Err(ChangeEmbedderDigestRequiresReembed)`
    /// - Compatible cases (empty DB, matching digest, or compat-attach
    ///   to `ExternalOrUnknown` provenance) → `Ok(())`
    ///
    /// All publication is under `index_write_lock` so concurrent
    /// set_embedder/reembed calls serialize cleanly.
    pub fn set_embedder(
        &mut self,
        embedder: Box<dyn crate::types::Embedder + Send + Sync>,
    ) -> Result<()> {
        let candidate_dim = embedder.dim();
        let candidate_fp = embedder.fingerprint();
        let candidate_name = embedder.name();
        let arc_embedder: std::sync::Arc<dyn crate::types::Embedder + Send + Sync> =
            std::sync::Arc::from(embedder);

        let _guard = self.index_write_lock.lock();
        let state = self.search_state.load_full();

        if candidate_dim != state.dim() {
            let memory_count = self.count_indexed_memories_for_set_embedder()?;
            return Err(YantrikDbError::ChangeEmbedderDimensionRequiresReembed {
                active_dim: state.dim(),
                candidate_dim,
                memory_count,
            });
        }

        let memory_count = self.count_indexed_memories_for_set_embedder()?;
        let index_empty = memory_count == 0;

        let new_state = if index_empty {
            // Empty index: attach + (if candidate has fingerprint)
            // upgrade provenance to Known. Otherwise stay ExternalOrUnknown.
            let new_provenance = match candidate_fp.as_deref() {
                Some(fp) => crate::engine::reembed::EmbeddingProvenance::Known {
                    name: candidate_name.clone(),
                    digest: fp.to_string(),
                    dim: candidate_dim,
                },
                None => crate::engine::reembed::EmbeddingProvenance::ExternalOrUnknown {
                    dim: candidate_dim,
                },
            };
            crate::engine::reembed::SearchState {
                index_embedding: new_provenance,
                embedder: Some(arc_embedder),
                runtime_embedder_name: candidate_name,
                runtime_embedder_digest: candidate_fp,
                generation: state.generation,
                covers_through_seq: state.covers_through_seq,
                hnsw_m: state.hnsw_m,
                hnsw_ef_construction: state.hnsw_ef_construction,
                hnsw_ef_search: state.hnsw_ef_search,
                // set_embedder never swaps the physical index — only
                // embedder/provenance. Reuse the same Arc<DeltaIndex>
                // so reembed phase 2 stays the only path that
                // republishes a new `vec_index` (brainstorm-4 §1).
                vec_index: std::sync::Arc::clone(&state.vec_index),
            }
        } else {
            match &state.index_embedding {
                crate::engine::reembed::EmbeddingProvenance::Known { digest, dim, .. } => {
                    if candidate_fp.as_deref() != Some(digest.as_str()) {
                        return Err(YantrikDbError::ChangeEmbedderDigestRequiresReembed {
                            active_digest: Some(digest.clone()),
                            candidate_digest: candidate_fp,
                            dim: *dim,
                            memory_count,
                        });
                    }
                    // Same digest: Arc-swap runtime embedder, no
                    // generation/provenance change.
                    crate::engine::reembed::SearchState {
                        index_embedding: state.index_embedding.clone(),
                        embedder: Some(arc_embedder),
                        runtime_embedder_name: candidate_name,
                        runtime_embedder_digest: candidate_fp,
                        generation: state.generation,
                        covers_through_seq: state.covers_through_seq,
                        hnsw_m: state.hnsw_m,
                        hnsw_ef_construction: state.hnsw_ef_construction,
                        hnsw_ef_search: state.hnsw_ef_search,
                        vec_index: std::sync::Arc::clone(&state.vec_index),
                    }
                }
                crate::engine::reembed::EmbeddingProvenance::ExternalOrUnknown { .. } => {
                    // Compat-attach: dim matches, provenance stays
                    // ExternalOrUnknown (we cannot claim the index is
                    // in this embedder's vector space — we don't know
                    // who built the existing vectors).
                    crate::engine::reembed::SearchState {
                        index_embedding: state.index_embedding.clone(),
                        embedder: Some(arc_embedder),
                        runtime_embedder_name: candidate_name,
                        runtime_embedder_digest: candidate_fp,
                        generation: state.generation,
                        covers_through_seq: state.covers_through_seq,
                        hnsw_m: state.hnsw_m,
                        hnsw_ef_construction: state.hnsw_ef_construction,
                        hnsw_ef_search: state.hnsw_ef_search,
                        vec_index: std::sync::Arc::clone(&state.vec_index),
                    }
                }
            }
        };

        // **Issue #41 brainstorm-4 §3.** Route through the
        // monotonic-generation CAS helper so the invariant
        // "SearchState generation never regresses" is enforced
        // uniformly across every publisher. set_embedder publishes
        // with `new_state.generation == state.generation` (it does
        // not advance the vector-space generation; only reembed
        // Phase-2 does), so the >= check inside the helper passes
        // here trivially.
        self.try_publish_search_state(new_state)?;
        // Legacy slot retired post-#41: all reads now route through
        // search_state. Clear it to catch any latent reader.
        self.embedder = None;
        // Chunked embeddings: a window probed under THIS embedder in a
        // previous process survives in `meta` — adopt it so chunking
        // does not silently deactivate across restarts. Digest-guarded
        // inside: a different embedder's window is never adopted.
        self.adopt_persisted_window();
        Ok(())
    }

    /// **Issue #41 brainstorm-4 §3 — monotonic-generation CAS for
    /// SearchState publication.**
    ///
    /// The single chokepoint through which any code path mutates
    /// `self.search_state`. The invariant is strict: a new
    /// SearchState may only be published if its `generation` is
    /// `>= self.search_state.load().generation`. Strictly-lesser
    /// generations are rejected — they represent stale work from a
    /// compactor / writer / reembed step whose snapshot was
    /// invalidated by a concurrent generation advance.
    ///
    /// brainstorm-4 §3 motivation: without this, a future
    /// compactor-style path that runs on the OLD SearchState and
    /// republishes after a reembed swap would ABA-rollback the
    /// active generation. That rollback is durable data omission —
    /// the post-swap materializer reapplies queued ops that were
    /// already covered by the new generation's `covers_through_seq`,
    /// double-applying writes and breaking RYW semantics.
    ///
    /// CAS implementation: uses ArcSwap's `compare_and_swap` so
    /// concurrent publishers race only on pointer identity, not
    /// generation values. The retry loop re-validates the
    /// generation guard each iteration; if a concurrent publisher
    /// races AND has a higher generation, this call returns
    /// `SearchStatePublishStaleGeneration` instead of looping
    /// forever — caller must rebuild their proposed state under the
    /// new active generation.
    ///
    /// Equal-generation publishes (same vector-space, different
    /// runtime metadata — e.g. set_embedder runtime-only Arc swap)
    /// are allowed: the generation tracks the index's vector space,
    /// not arbitrary state changes.
    pub(crate) fn try_publish_search_state(
        &self,
        new_state: crate::engine::reembed::SearchState,
    ) -> Result<()> {
        let new_arc = std::sync::Arc::new(new_state);
        loop {
            let current = self.search_state.load_full();
            if new_arc.generation < current.generation {
                return Err(YantrikDbError::SearchStatePublishStaleGeneration {
                    current_generation: current.generation,
                    attempted_generation: new_arc.generation,
                });
            }
            // ArcSwap::compare_and_swap returns the previous Arc.
            // If it's pointer-equal to `current`, the swap landed.
            // Otherwise a concurrent publisher raced; loop and
            // re-validate against the new current.
            let prev = self
                .search_state
                .compare_and_swap(&current, std::sync::Arc::clone(&new_arc));
            if std::sync::Arc::ptr_eq(&prev, &current) {
                return Ok(());
            }
            // Concurrent publisher swapped between load and CAS.
            // Loop: re-load, re-validate. The retry budget is
            // bounded by the number of concurrent publishers, which
            // is bounded by the index_write_lock (today only
            // set_embedder + reembed contend, and the lock
            // serializes them anyway — the CAS is defense in
            // depth).
        }
    }

    /// Internal helper for `set_embedder*` / future `reembed()`: count
    /// memories that have an embedding (indexed vectors). Uses SQL count
    /// for consistency across delta / cold / tombstoned states. Called
    /// under `index_write_lock`.
    pub(crate) fn count_indexed_memories_for_set_embedder(&self) -> Result<u64> {
        let conn = self.read_conn();
        let n: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL \
             AND consolidation_status = 'active'",
            [],
            |row| row.get(0),
        )?;
        Ok(n.max(0) as u64)
    }

    /// Whether a runtime embedder is configured. Derives from
    /// SearchState — single source of truth after #41 layer 2.
    pub fn has_embedder(&self) -> bool {
        self.search_state.load().embedder.is_some()
    }

    /// Embed text using the configured runtime embedder. Acquires one
    /// SearchState snapshot at the start so the call uses a consistent
    /// embedder even if set_embedder/reembed runs concurrently.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let state = self.search_state.load_full();
        let embedder = state.embedder.as_ref().ok_or(YantrikDbError::NoEmbedder)?;
        let out = embedder
            .embed(text)
            .map_err(|e| YantrikDbError::Inference(e.to_string()))?;
        // v0.9.3 contract gate: validate the EMBEDDER'S output too — an
        // external/BYO embedder with an unguarded 0/0 (the issue #60 org
        // user's ONNX mean-pool bug) can emit NaN; catch it here rather
        // than persist it. Covers every engine-side embedding consumer.
        crate::validate::validate_embedding("embed", &out, state.dim())?;
        // Silent truncation is silent retrieval loss: if this text is
        // longer than the embedder's detected window, its tail is about
        // to be stored intact and never embedded. Counted and warned
        // rather than swallowed. No-op until the window is probed.
        self.note_possible_truncation(text.len());
        // **Issue #117 / packs.** A vector in this database's space was
        // just produced by the attached embedder — record that identity
        // once. This is the hook that covers the binding path, where
        // `record_text` embeds through here and then calls `record()`
        // with the result, so the engine-internal `record_text` stamp
        // never fires.
        self.stamp_embedder_identity_once();
        Ok(out)
    }

    /// Record a memory with automatic embedding generation.
    ///
    /// **Issue #41 brainstorm-4 §2 — writer revalidation loop.**
    /// `record_text` performs the engine-side embed step, which is
    /// SLOW (e.g. tens of ms for a model embedding). Per brainstorm-4
    /// the embed runs OUTSIDE the `WriteRouter` guard so reembed
    /// throughput stays bounded by the index rebuild, not by every
    /// in-flight `record_text`. The price of "embed outside the
    /// barrier" is that the active generation can advance between
    /// the embed and the commit — landing an old-embedder vector in
    /// the new-generation index is durable silent corruption when
    /// dims happen to match (and a noisy `EmbeddingDimensionMismatch`
    /// when they don't).
    ///
    /// The loop closes that window:
    /// 1. Snapshot SearchState (`gen_pre`, embedder, digest_pre).
    /// 2. Embed under that embedder. NO guard held — slow step.
    /// 3. Try to acquire the sync guard. If the router has flipped to
    ///    Queueing, route to `record_queued(text)` — the post-swap
    ///    materializer will re-encode under the new embedder.
    /// 4. With guard held, re-snapshot SearchState (`gen_post`,
    ///    digest_post). Guard prevents reembed from completing its
    ///    swap from this point onward.
    /// 5. If `gen_pre == gen_post && digest_pre == digest_post`: the
    ///    embedding we computed is consistent with the active
    ///    generation. Commit via `record_under_guard_and_state`.
    /// 6. Otherwise: a reembed swap completed between step 1 and
    ///    step 4. Drop the guard and retry from step 1 — the next
    ///    iteration embeds under the NEW embedder.
    ///
    /// The loop is bounded in expectation because reembed completes
    /// at most once per outer call (it advances generation
    /// monotonically and waits-for-no-sync-writers before swapping).
    /// The retry budget is unbounded in the API surface — a
    /// pathological caller can flap embedders forever, but real
    /// reembed runs land once and then stay landed.
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
        self.record_text_with_idempotency(
            text,
            memory_type,
            importance,
            valence,
            half_life,
            metadata,
            namespace,
            certainty,
            domain,
            source,
            emotional_state,
            None,
            None,
        )
    }

    /// `record_text()` plus a durable idempotency key (v0.10 Item 4a.6d).
    ///
    /// Same contract as [`Self::record_with_idempotency`] with ONE deliberate
    /// difference: the digest uses [`PayloadVariant::RecordText`], which
    /// **excludes the engine-generated embedding**. The engine embeds the text
    /// itself here, and an embedder can legitimately be swapped (or drift)
    /// between attempts — digesting the generated vector would turn an honest
    /// retry into a false conflict. Idempotency is decided from the TEXT and
    /// scalars, before any embedding work: the pre-admission probe runs before
    /// the (slow) embed, so a duplicate retry never pays the embed cost at all.
    ///
    /// The variant is also part of the digest's op_kind discriminator, so the
    /// SAME key used across `record()` (embedding-inclusive) and
    /// `record_text()` (embedding-exclusive) is a typed conflict, not a hit —
    /// a cross-surface retry is not the same write.
    ///
    /// `None` is byte-for-byte `record_text()`.
    ///
    /// `created_at`: caller-supplied event time in epoch seconds (historical
    /// import — `RecordInput::created_at` has the full contract). `None`
    /// stamps `now()`. When `Some`, it joins the RecordText digest: a
    /// re-dated write decays and `recall_as_of`s differently, so it is a
    /// different write even under the embedding-excluded variant.
    #[allow(clippy::too_many_arguments)]
    pub fn record_text_with_idempotency(
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
        idempotency_key: Option<&str>,
        created_at: Option<f64>,
    ) -> Result<String> {
        self.record_text_with_idempotency_routed(
            text,
            memory_type,
            importance,
            valence,
            half_life,
            metadata,
            namespace,
            certainty,
            domain,
            source,
            emotional_state,
            idempotency_key,
            created_at,
            true,
            None,
        )
    }

    /// Sync-only variant for callers that must attach durable side effects to
    /// the materialized memory row before returning success.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_text_with_idempotency_sync_only(
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
        idempotency_key: Option<&str>,
        created_at: Option<f64>,
        synthesis: Option<&SynthesisAdmission>,
    ) -> Result<String> {
        self.record_text_with_idempotency_routed(
            text,
            memory_type,
            importance,
            valence,
            half_life,
            metadata,
            namespace,
            certainty,
            domain,
            source,
            emotional_state,
            idempotency_key,
            created_at,
            false,
            synthesis,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_text_with_idempotency_routed(
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
        idempotency_key: Option<&str>,
        created_at: Option<f64>,
        allow_queued_route: bool,
        synthesis: Option<&SynthesisAdmission>,
    ) -> Result<String> {
        // v0.9.3 contract gate: scalars validated BEFORE calibration mutates
        // the namespace's running distribution. (The embedding is engine-
        // generated below and validated inside the embed step.)
        crate::validate::validate_scalars(
            "record_text",
            &[
                ("importance", importance),
                ("valence", valence),
                ("certainty", certainty),
                ("half_life", half_life),
            ],
        )?;
        // Caller-supplied event time: finite or refused, before the digest,
        // the probe, and the (slow) embed — record()'s gate, same rationale.
        if let Some(ts) = created_at {
            crate::validate::validate_scalars("record_text", &[("created_at", ts)])?;
        }
        // v0.10 Item 4a.4 anti-laundering gate — before the (slow) embed and
        // any side effect. `record_text` bypasses `record()`, so it gates here
        // too (T06 coverage). A warn-mode Flagged verdict is carried to the
        // routed path and counted only after the write commits (4a.6b).
        let gate_verdict = self.gate_provenance(source, metadata)?;
        // **Issue #117 / packs.** Public writes preserve the historical
        // early stamp. The sync-only consolidation route delays it until
        // after acquiring and revalidating the sync guard: a queue-mode
        // deferral must not mutate durable metadata while claiming that no
        // durable state changed.
        if allow_queued_route {
            self.stamp_embedder_identity_once();
        }
        // Task 29 (Ingest Integrity): strip any leaked tool-call
        // serialization tail BEFORE embedding, so both the computed vector
        // and the stored text reflect the real memory rather than the
        // artifact. Borrowed (no allocation) on the clean path.
        let sanitized = sanitize::sanitize_tool_call_artifacts(text);
        let text = sanitized.as_ref();

        // EVENT TIME — same merge as the supplied-embedding path. These two
        // write paths are separate implementations rather than delegates, so
        // anything added to one and not the other silently applies to half of
        // all writes; this path is the one the Python binding uses.
        let metadata_owned = crate::base::datetext::merge_event_dates(metadata, text);
        let metadata = &metadata_owned;
        // v0.7.23 normalization, APPLIED HERE for the first time (sol 4a.6d-1
        // finding): record_text never normalized blank namespaces — a
        // pre-existing divergence from record(), which coerces ""/whitespace to
        // "default" at its entry (record.rs). It went unnoticed while the
        // Python wrapper routed embedding=None through record(); routing it
        // through THIS path exposed the gap: rows and idempotency claims would
        // scope under "" while every reader queries "default". Normalize once,
        // before calibration, digest, probe, and routing — the same
        // engine-boundary contract every other write entry keeps.
        let namespace = record::normalize_namespace(namespace);
        // Task 31 (Ingest Integrity): compute the calibrated importance once,
        // before the (retryable) embed loop — READ-ONLY as of 4a.6b, so the
        // "retry must not double-count" property this comment used to defend is
        // now structural: the distribution advances inside the winning path's
        // transaction, and a retry loop commits at most once.
        let raw_importance = importance;
        let importance = self.calibrated_importance(namespace, importance)?;
        // 4a.6d: the RecordText digest — RAW canonical payload with the
        // embedding EXCLUDED (the engine generates it below, and idempotency
        // must be decided before re-embedding; see the method doc). Then the
        // pre-admission probe: a duplicate retry resolves HERE, before the
        // slow embed, before the router, before any admission machinery.
        // "Admission" is precise (sol 4a.6d-2b r1 finding 2): the validation
        // gates above still precede the probe — deterministic payload-shape
        // checks an identical retry passes identically, unlike the
        // saturation-dependent admission this probe exists to bypass.
        let idem: Option<(&str, [u8; 32])> = match idempotency_key {
            None => None,
            Some(key) => {
                if key.trim().is_empty() || key.len() > 512 {
                    return Err(YantrikDbError::InvalidIdempotencyKey {
                        reason: if key.len() > 512 {
                            format!("key is {} bytes; max 512", key.len())
                        } else {
                            "key is empty or whitespace-only".to_string()
                        },
                    });
                }
                let view = crate::payload_digest::PayloadView {
                    variant: crate::payload_digest::PayloadVariant::RecordText,
                    namespace,
                    text,
                    memory_type,
                    importance: raw_importance,
                    valence,
                    half_life,
                    certainty,
                    domain,
                    source,
                    emotional_state,
                    metadata,
                    embedding: None,
                    created_at,
                };
                Some((key, crate::payload_digest::payload_digest(&view)))
            }
        };
        if let Some((key, digest)) = idem.as_ref() {
            if let Some(existing_rid) = idempotency::probe_committed_claim(
                &self.conn(),
                &self.actor_id,
                namespace,
                key,
                digest,
            )? {
                return Ok(existing_rid);
            }
        }
        loop {
            // Step 1: snapshot SearchState for the embed — capture
            // generation + digest so we can revalidate after the embed.
            let state_for_embed = self.search_state.load_full();
            let gen_pre = state_for_embed.generation;
            let digest_pre = state_for_embed.runtime_embedder_digest.clone();
            let embedder = state_for_embed
                .embedder
                .as_ref()
                .ok_or(YantrikDbError::NoEmbedder)?
                .clone();
            // v0.9.3: capture the snapshot's dim for output validation below
            // (authoritative for THIS generation; the step-5 revalidation
            // retries if a swap lands mid-embed).
            let dim_pre = state_for_embed.dim();
            // Release the Arc<SearchState> BEFORE the slow embed —
            // brainstorm-4 §4 invariant ("no holding SearchState
            // across long ops"). The embedder Arc is the small,
            // bounded retention.
            drop(state_for_embed);

            // Step 2: embed OUTSIDE any guard. Slow step.
            let embedding = embedder
                .embed(text)
                .map_err(|e| YantrikDbError::Inference(e.to_string()))?;
            // v0.9.3 contract gate: validate the embedder's output before
            // committing (this loop bypasses `self.embed()`, so it needs
            // its own gate — an external embedder can emit NaN, issue #60).
            crate::validate::validate_embedding("record_text", &embedding, dim_pre)?;

            // Chunked embeddings: when the text overflows the probed
            // window, embed the remaining windows here — same snapshot
            // embedder, same slow step, so the step-5 gen/digest
            // revalidation covers the whole vector SET. (For a
            // truncating embedder the full-text vector above IS the
            // head window's vector — chunk 0 costs nothing extra.)
            let chunks: Vec<(usize, Vec<f32>)> = match self.chunk_plan(text) {
                Some(ranges) => {
                    let mut cv = Vec::with_capacity(ranges.len());
                    for (i, (a, b)) in ranges.iter().enumerate() {
                        let v = embedder
                            .embed(&text[*a..*b])
                            .map_err(|e| YantrikDbError::Inference(e.to_string()))?;
                        crate::validate::validate_embedding("record_text#chunk", &v, dim_pre)?;
                        cv.push((i + 1, v));
                    }
                    cv
                }
                None => Vec::new(),
            };
            // The overflow accounting: a chunked write is HANDLED (its
            // tail is findable), a bare overflow is truncation loss.
            // record_text bypasses `self.embed()`, so it does its own
            // counting — the warning would otherwise miss the engine's
            // primary write path entirely.
            if !chunks.is_empty() {
                self.note_chunked_write();
            } else {
                self.note_possible_truncation(text.len());
            }

            // Step 3: try to enter sync path.
            let sync_guard = match self.write_router.try_enter_sync_writer() {
                Some(g) => g,
                None => {
                    if !allow_queued_route {
                        return Err(YantrikDbError::ConsolidationDeferredDuringReembed);
                    }
                    // Queueing state — reembed cutover is in
                    // flight. Route to the queued path. The
                    // pre-computed embedding is discarded; the
                    // queued path stores TEXT and the post-swap
                    // materializer re-encodes under the new
                    // embedder (brainstorm-3 invariant 8).
                    return self.record_queued(
                        text,
                        memory_type,
                        importance,
                        raw_importance,
                        valence,
                        half_life,
                        metadata,
                        &embedding,
                        namespace,
                        certainty,
                        domain,
                        source,
                        emotional_state,
                        gate_verdict,
                        idem,
                        created_at,
                    );
                }
            };

            // Step 4: re-snapshot SearchState UNDER the guard. From
            // this point, reembed cannot complete its swap until
            // our guard drops, so the loaded state is stable for
            // the rest of the critical section.
            let state_for_commit = self.search_state.load_full();

            // Step 5: revalidate. If a swap completed between step 1
            // and step 4, the embedding is in the wrong vector
            // space and we must retry.
            if state_for_commit.generation != gen_pre
                || state_for_commit.runtime_embedder_digest != digest_pre
            {
                // Generation or digest advanced. Drop guard and
                // retry the whole loop — next iteration embeds
                // under the new active embedder.
                drop(sync_guard);
                tracing::info!(
                    gen_pre,
                    gen_post = state_for_commit.generation,
                    "record_text: SearchState advanced mid-embed, retrying",
                );
                continue;
            }

            if !allow_queued_route {
                self.stamp_embedder_identity_once();
            }

            // Step 6: commit. The shared post-guard helper handles
            // SQL insert + vec_index.append + log_op (which itself
            // stamps applied_generation = state.generation).
            return self.record_under_guard_and_state(
                state_for_commit,
                sync_guard,
                text,
                memory_type,
                importance,
                raw_importance,
                valence,
                half_life,
                metadata,
                &embedding,
                &chunks,
                namespace,
                certainty,
                domain,
                source,
                emotional_state,
                gate_verdict,
                idem,
                created_at,
                synthesis,
            );
        }
    }

    /// Recall memories by text query with automatic embedding.
    ///
    /// Graph expansion is OFF by default as of 2026-08-05: measured on a
    /// 4,297-record production corpus with a paraphrase-labeled query
    /// set, `expand_entities=true` cost 0.24 MRR (0.504 → 0.264) —
    /// entity-linked candidates score `0.3·proximity` into the
    /// relevance core, so proximity-rich noise sharing an entity name
    /// outranks genuinely similar records. On the synthetic *connected*
    /// corpus (built to favor the graph) the lift is NEUTRAL
    /// (+0.000 recall). Callers with curated, dense entity graphs can
    /// opt in per-call via `recall(..., expand_entities: true, ...)`.
    pub fn recall_text(&self, query: &str, top_k: usize) -> Result<Vec<RecallResult>> {
        let embedding = self.embed(query)?;
        self.recall(
            &embedding,
            top_k,
            None,  // time_window
            None,  // memory_type
            false, // include_consolidated
            false, // expand_entities — see doc: measured −0.24 MRR on by default
            Some(query),
            false, // skip_reinforce
            None,  // namespace
            None,  // domain
            None,  // source
            None,  // certainty_min (#46)
            None,  // order (#46) — relevance
            false, // include_superseded (v0.10 Item 1) — policy default
            None,  // event_after (#149)
            None,  // event_before (#149)
        )
    }

    /// v0.13.1 — `recall_text` with the explain surface: same defaults
    /// (`expand_entities` follows the caller so the graph lane's
    /// never-ran provenance is visible rather than hard-coded away),
    /// plus a [`crate::types::RecallExplain`] carrying the candidate
    /// pool, per-row lane-admission sets, per-lane ran/never-ran
    /// status, and the bm25 degeneracy ratio. `skip_reinforce=true` is
    /// the right choice for gates and probes — an explain call should
    /// observe the store, not mutate access_count.
    pub fn recall_text_explained(
        &self,
        query: &str,
        top_k: usize,
        namespace: Option<&str>,
        expand_entities: bool,
        skip_reinforce: bool,
    ) -> Result<(Vec<RecallResult>, crate::types::RecallExplain)> {
        let embedding = self.embed(query)?;
        self.recall_explained(
            &embedding,
            top_k,
            None,  // time_window
            None,  // memory_type
            false, // include_consolidated
            expand_entities,
            Some(query),
            skip_reinforce,
            namespace,
            None,  // domain
            None,  // source
            None,  // certainty_min
            None,  // order — relevance
            false, // include_superseded
            None,  // event_after (#149)
            None,  // event_before (#149)
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
            false, // expand_entities — see recall_text doc (measured −0.24 MRR)
            Some(query),
            false, // skip_reinforce
            None,  // namespace
            domain,
            source,
            None,  // certainty_min (#46)
            None,  // order (#46) — relevance
            false, // include_superseded (v0.10 Item 1) — policy default
            None,  // event_after (#149)
            None,  // event_before (#149)
        )
    }
}
