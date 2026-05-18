//! In-place embedder migration for `YantrikDB` (issue #41).
//!
//! `db.reembed(new_embedder_name, options)` rebuilds every memory's
//! embedding under a new embedder + swaps the active HNSW index, all
//! while concurrent recalls continue serving and (by default) concurrent
//! writes queue rather than block. The operation is atomic from the
//! caller's perspective: at completion the active `SearchState` carries
//! the new embedder + new dim + new index, and every prior memory + every
//! write that arrived during the operation is reflected.
//!
//! ## Why this exists
//!
//! Without it, callers wanting to upgrade between embedder dimensions
//! (e.g. potion-base-2M at dim=64 → potion-base-32M at dim=256) have to
//! dump-and-replay through `record_text`, which loses graph edges,
//! consolidation state, and conflict-detection metadata. Reported by
//! yantrikdb-hermes-agent for downstream user wysie. Engine-side
//! primitive is the right place per the cross-cutting nature of the
//! change (memories table, HNSW lifecycle, embedder slot all need to
//! change in lockstep).
//!
//! ## Correctness invariants (locked by brainstorm 2 redteam)
//!
//! See yantrikos/yantrikdb#41 for the full design rationale. Eight
//! testable invariants the implementation enforces:
//!
//! 1. **Single classification:** every write during reembed is either
//!    sync-before-barrier (in `build_hwm` and rebuilt into new HNSW)
//!    OR queued-after-barrier (replayed into the new generation by the
//!    post-swap materializer). No middle state.
//! 2. **No old application after barrier:** post-cutover, no code path
//!    may mark an oplog row `applied_generation = old_generation` in a
//!    way that suppresses replay into the new generation.
//! 3. **Generation-scoped application:** `oplog.applied_generation`
//!    (v27 column) is the source of truth, not boolean `applied`. The
//!    boolean is kept as derived hint for back-compat only.
//! 4. **High-water coverage:** `SearchState(G).covers_through_seq` is
//!    a durable per-generation coverage statement. Every write with
//!    `seq <= covers_through_seq` is reflected in the active index.
//! 5. **Replay rule:** the post-swap G1 materializer replays
//!    `WHERE applied_generation IS NULL OR applied_generation < G`
//!    ordered by `op_id`.
//! 6. **RYW rule:** `recall_with_seq(N)` waits on the active
//!    generation's `visible_seq >= N` regardless of which generation
//!    accepted the write.
//! 7. **Atomic SearchState:** embedder + index + dim + generation +
//!    covers_through_seq + hnsw_params are swapped together via a
//!    single `ArcSwap<SearchState>::store`. No code path can observe
//!    a mismatched embedder-vs-index pair.
//! 8. **Queued payload correctness:** writes queued during reembed
//!    store logical text only (not pre-encoded embeddings). The G1
//!    materializer computes embeddings under E1 at replay time.
//!
//! ## Module layout
//!
//! This file holds the public type surface only — `ReembedOptions`,
//! `ReembedProgress`, `ReembedReport`, `ReembedPhase`, `ReembedStatus`,
//! `ReembedWritePolicy`, `NamespaceReembedStats`. The actual reembed
//! loop, WriteRouter, and SearchState swap machinery live in sibling
//! modules `reembed_loop` (TODO) and on `YantrikDB` itself once wired.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────
// Public API surface
// ─────────────────────────────────────────────────────────────────────

/// Concurrent-write policy during a reembed operation.
///
/// v1 ships both modes. Default is `Queue` per the brainstorm-2 design
/// lock — the engine already has the oplog `applied=0` + materializer
/// machinery from v0.6.6 / v0.7.x, and shipping pause-only-v1 would
/// force downstream consumers to write workarounds that become
/// obsolete when queue lands. The queue infrastructure is plumbing,
/// not novel design, given the safety invariants (`applied_generation`
/// + `covers_through_seq` + WriteRouter + paused-materializer) the
/// implementation enforces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReembedWritePolicy {
    /// Concurrent `record()` / `record_text()` calls return
    /// `Err(YantrikDbError::WriteRejected { retry_after_ms })`. The
    /// caller is expected to retry after the reembed completes.
    /// Simpler for the engine; harsher for the caller.
    Pause,
    /// Concurrent writes go through `log_op_pending` (applied=0,
    /// `embedding_model = old_embedder_name`). After the SearchState
    /// swap, the post-swap materializer reads the text from each
    /// queued op's payload, encodes it under the new embedder, and
    /// applies to the new HNSW. Read-after-write requires
    /// `recall_with_seq(min_seq)` because the active `visible_seq`
    /// only advances as the materializer drains. Default.
    Queue,
}

impl Default for ReembedWritePolicy {
    /// Defaults to `Queue` per the brainstorm-2 design lock + Pranab's
    /// 2026-05-17 decision. Do NOT change to `Pause` without revisiting
    /// the AskUserQuestion record on issue #41.
    fn default() -> Self {
        ReembedWritePolicy::Queue
    }
}

/// Discrete phases of the reembed state machine. Authoritative source
/// is the `reembed_events` table; the in-memory `on_phase_complete`
/// callback in [`ReembedOptions`] is best-effort notification only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReembedPhase {
    /// Resolve embedder name → digest, download if needed, sanity-encode
    /// a test string, validate dim, write `meta.reembed_state` with the
    /// resolved identity. Failure here leaves no durable state.
    Probing,
    /// Iterate `memories ORDER BY rid` in `batch_size` chunks, encode
    /// each row's text under the new embedder, write to staging columns
    /// `memories.embedding_new` + `memories.embedding_new_model`. The
    /// active `memories.embedding` column is untouched throughout, so
    /// concurrent recall traffic continues to score consistently
    /// against the old SearchState.
    Encoding,
    /// Build new `HnswIndex` from `memories.embedding_new` values via
    /// the existing `DeltaIndex` clone-rebuild path. Disposable on
    /// crash — recovery restarts from Encoding.
    Rebuilding,
    /// Atomic `ArcSwap<SearchState>::store(new_state)` inside the same
    /// SQL transaction as `UPDATE memories SET embedding = embedding_new`.
    /// Past point of no return — crash here completes the swap on
    /// next `open()`.
    Swapping,
    /// Sanity-check that all rows have `embedding_model = new_embedder`,
    /// clear staging columns of any residual non-NULL values, clear
    /// `meta.reembed_state`. Safe to retry.
    Verifying,
    /// Reembed completed successfully. Terminal state.
    Completed,
    /// Reembed aborted before Swap (Probing failed, Encoding crashed
    /// with `resume_from_checkpoint=false`, etc.). Terminal state; the
    /// caller can safely retry from scratch.
    Aborted,
}

impl ReembedPhase {
    /// Stable string form for `reembed_events.phase` rows and
    /// `meta.reembed_state` JSON. Tied to the SQL representation; do
    /// NOT rename without a schema migration.
    pub fn as_str(&self) -> &'static str {
        match self {
            ReembedPhase::Probing => "Probing",
            ReembedPhase::Encoding => "Encoding",
            ReembedPhase::Rebuilding => "Rebuilding",
            ReembedPhase::Swapping => "Swapping",
            ReembedPhase::Verifying => "Verifying",
            ReembedPhase::Completed => "Completed",
            ReembedPhase::Aborted => "Aborted",
        }
    }

    /// Parse from the SQL string form. Returns `None` for unknown
    /// strings — caller must treat that as corruption.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "Probing" => Some(ReembedPhase::Probing),
            "Encoding" => Some(ReembedPhase::Encoding),
            "Rebuilding" => Some(ReembedPhase::Rebuilding),
            "Swapping" => Some(ReembedPhase::Swapping),
            "Verifying" => Some(ReembedPhase::Verifying),
            "Completed" => Some(ReembedPhase::Completed),
            "Aborted" => Some(ReembedPhase::Aborted),
            _ => None,
        }
    }

    /// True if the reembed operation has terminated (success or abort).
    /// `open()` recovery treats terminal states as "no in-flight work"
    /// and clears `meta.reembed_state`.
    pub fn is_terminal(&self) -> bool {
        matches!(self, ReembedPhase::Completed | ReembedPhase::Aborted)
    }
}

impl fmt::Display for ReembedPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Per-batch progress emitted to the optional `progress_cb` callback in
/// [`ReembedOptions`]. The callback is best-effort notification — it
/// runs synchronously on the reembed thread, so a slow callback slows
/// the operation. Authoritative phase-transition history is in the
/// `reembed_events` table; use `db.reembed_status()` for snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReembedProgress {
    pub phase: ReembedPhase,
    /// Memories processed in the current phase so far. Monotonic
    /// within a phase; resets at phase transitions.
    pub processed: u64,
    /// Total memories the current phase will process. May be `None`
    /// during Probing (before count is known) or during Rebuilding /
    /// Swapping / Verifying (phase has no per-row work).
    pub total: Option<u64>,
    /// Wall-clock time since the reembed call started.
    pub elapsed_ms: u64,
    /// If the reembed is namespace-scoped, the namespace being
    /// processed. `None` for cross-namespace reembeds.
    pub namespace: Option<String>,
}

/// Caller-side options for a reembed call.
///
/// Construct via [`ReembedOptions::default()`] (`write_policy = Queue`,
/// `batch_size = 256`, `resume_from_checkpoint = true`, no callbacks)
/// and override per-field as needed.
pub struct ReembedOptions {
    /// If set, only re-embed memories in this namespace. Cross-
    /// namespace reembed is rejected if another reembed is in flight
    /// (single-job invariant via `meta.reembed_state`).
    pub namespace: Option<String>,

    /// Best-effort progress callback fired after each Encoding batch +
    /// at each phase transition. Runs on the reembed thread; keep it
    /// fast. Use `db.reembed_status()` for snapshots instead if you
    /// need durable / pollable observability.
    pub progress_cb: Option<Box<dyn Fn(ReembedProgress) + Send + Sync>>,

    /// Best-effort callback fired at each phase transition. NOT
    /// authoritative — the durable record is in `reembed_events`. If
    /// this callback panics or is slow, the durable record still
    /// reflects the transition.
    pub on_phase_complete: Option<Box<dyn Fn(ReembedPhase, &ReembedStatus) + Send + Sync>>,

    /// Encoding batch size. Higher = fewer SQL transactions but more
    /// memory pressure per batch. Default 256 is balanced for typical
    /// dims (64-768) and memory counts up to ~1M.
    pub batch_size: usize,

    /// Concurrent-write policy. See [`ReembedWritePolicy`]. Default
    /// `Queue`.
    pub write_policy: ReembedWritePolicy,

    /// HNSW M parameter override. `None` preserves the current
    /// SearchState's M. Set explicitly if the new embedder's vector
    /// space has materially different cluster properties.
    pub hnsw_m: Option<u32>,
    /// HNSW efConstruction override. `None` preserves current.
    pub hnsw_ef_construction: Option<u32>,
    /// HNSW efSearch override. `None` preserves current.
    pub hnsw_ef_search: Option<u32>,

    /// If `true` (default) and a previous reembed was interrupted at
    /// Encoding/Rebuilding phase, `open()` resumes from the recorded
    /// checkpoint. If `false`, `open()` rolls back the partial work
    /// and clears `meta.reembed_state`. Swapping/Verifying always
    /// complete forward (past the point of no return).
    pub resume_from_checkpoint: bool,

    /// If `true`, run Probing only and return without writing
    /// `meta.reembed_state` or touching any memories. Useful for "would
    /// this reembed succeed?" checks before committing.
    pub dry_run: bool,
}

impl Default for ReembedOptions {
    fn default() -> Self {
        ReembedOptions {
            namespace: None,
            progress_cb: None,
            on_phase_complete: None,
            batch_size: 256,
            write_policy: ReembedWritePolicy::default(),
            hnsw_m: None,
            hnsw_ef_construction: None,
            hnsw_ef_search: None,
            resume_from_checkpoint: true,
            dry_run: false,
        }
    }
}

// ReembedOptions intentionally does NOT derive Debug because progress_cb
// and on_phase_complete are dyn Fn closures (no Debug). A manual impl
// would just print "<callbacks>" placeholders; not worth the noise.

/// Per-namespace breakdown of a reembed report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NamespaceReembedStats {
    pub encoded_count: u64,
    pub skipped_count: u64,
    pub duration_ms: u64,
}

/// Terminal report from a successful `db.reembed()` call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReembedReport {
    pub generation: u64,
    pub encoded_count: u64,
    /// Memories that had no text content (rare; happens when text was
    /// nulled out via correction or other manipulation). Skipped in
    /// Encoding; their HNSW entries are pruned.
    pub skipped_count: u64,
    pub duration: Duration,
    pub old_embedder: String,
    pub old_embedder_digest: String,
    pub new_embedder: String,
    pub new_embedder_digest: String,
    pub old_dim: usize,
    pub new_dim: usize,
    /// Coverage seq the new generation covers. Every write with
    /// `seq <= build_hwm` was rebuilt into the new HNSW;
    /// `seq > build_hwm` is replayed by the post-swap materializer.
    pub build_hwm: u64,
    pub per_namespace: HashMap<String, NamespaceReembedStats>,
}

/// Read-only snapshot of an in-flight reembed. Returned by
/// `db.reembed_status()`. None when no reembed is active. Authoritative
/// state lives in `reembed_events` + `meta.reembed_state`; this struct
/// is the queryable projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReembedStatus {
    pub generation: u64,
    pub phase: ReembedPhase,
    pub old_embedder: String,
    pub old_embedder_digest: String,
    pub new_embedder: String,
    pub new_embedder_digest: String,
    pub old_dim: usize,
    pub new_dim: usize,
    pub memories_total: u64,
    pub memories_encoded: u64,
    pub queued_writes: u64,
    pub checkpoint_rid: Option<String>,
    pub started_at: SystemTime,
    pub last_event_at: SystemTime,
    pub last_error: Option<String>,
    pub write_policy: ReembedWritePolicy,
}

// ─────────────────────────────────────────────────────────────────────
// EmbeddingProvenance — what model produced the vectors in the index
// ─────────────────────────────────────────────────────────────────────

/// Provenance of vectors currently stored in the active vector index.
///
/// **Independent of the runtime embedder.** After restart with no
/// embedder loaded, the index still has knowable provenance — the
/// embedder name + digest under which its vectors were built. The
/// runtime embedder slot tracks "is there a local embedder available
/// for embed_text() right now?" — a separate question.
///
/// Locked by brainstorm-3 round 2: conflating these two facts loses
/// critical safety information. See yantrikos/yantrikdb#41 comment
/// chain for the bad-state scenarios.
#[derive(Debug, Clone)]
pub enum EmbeddingProvenance {
    /// Index vectors were built with a known embedder identity. Set by
    /// a completed reembed() or by initial population of an empty DB
    /// via record_text() with a configured embedder. The
    /// `set_embedder*` mode logic uses this to reject silent-corruption
    /// scenarios (same-dim-different-digest, dim-mismatch).
    Known {
        /// Optional human-readable name (e.g. "potion-base-2M").
        name: Option<String>,
        /// SHA-256 or equivalent stable fingerprint of the embedder
        /// model. Compared by `set_embedder*` for exact-identity match.
        digest: String,
        /// Dimensionality of vectors in the active index.
        dim: usize,
    },
    /// Index vectors come from external/precomputed input, or from a
    /// legacy DB created before fingerprint tracking. Dimensionality is
    /// known (it's the physical index dim); the originating embedder is
    /// not provable. `set_embedder*` with matching dim can attach a
    /// runtime embedder via the compat path without claiming the index
    /// is in the new embedder's vector space.
    ExternalOrUnknown {
        dim: usize,
    },
}

impl EmbeddingProvenance {
    /// Dimensionality of vectors in the index, regardless of provenance
    /// variant. Single accessor so callers don't pattern-match.
    pub fn dim(&self) -> usize {
        match self {
            EmbeddingProvenance::Known { dim, .. } => *dim,
            EmbeddingProvenance::ExternalOrUnknown { dim } => *dim,
        }
    }

    /// Stable fingerprint of the index-building embedder, if known.
    /// `None` for `ExternalOrUnknown` variant.
    pub fn digest(&self) -> Option<&str> {
        match self {
            EmbeddingProvenance::Known { digest, .. } => Some(digest),
            EmbeddingProvenance::ExternalOrUnknown { .. } => None,
        }
    }

    /// Human-readable name of the index-building embedder, if recorded.
    pub fn name(&self) -> Option<&str> {
        match self {
            EmbeddingProvenance::Known { name, .. } => name.as_deref(),
            EmbeddingProvenance::ExternalOrUnknown { .. } => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// SearchState — the single atomic swap unit
// ─────────────────────────────────────────────────────────────────────

/// The coherent search-stack snapshot that every read path acquires
/// once via `ArcSwap.load()` and uses for the entire request. Locked by
/// brainstorm-2 invariant 7 (atomic SearchState) + brainstorm-3 round 2
/// (provenance and runtime embedder are separate facts).
///
/// All fields are immutable after construction. Mutation is by
/// `ArcSwap<Arc<SearchState>>::store(new_state)` under the engine's
/// `index_write_lock`.
///
/// ## Field design rationale
///
/// `index_embedding` describes what's IN the active index — provenance
/// of stored vectors. `embedder` + `runtime_embedder_*` describes what
/// the engine can produce NOW for embed_text(). These can disagree: a
/// freshly-opened DB has `index_embedding = Known(...)` but
/// `embedder = None` until `set_embedder*` is called.
///
/// **`SearchState.dim` is derived**: use `state.index_embedding.dim()`,
/// not a separate field. The standalone `YantrikDB.embedding_dim` field
/// retires in this refactor; that derivation is the new source of truth.
///
/// ## What's NOT in this struct in v1
///
/// The actual HNSW index handle is intentionally NOT a member. The
/// engine's existing `vec_index: DeltaIndex` continues to own the index
/// lifecycle. Reembed sequences SearchState.store + vec_index.cold
/// ArcSwap inside the same critical section (`index_write_lock`) so
/// they're observed atomically. Moving the index handle into
/// SearchState is a bigger refactor (changes DeltaIndex ownership)
/// deferred to a future release.
///
/// Compensating invariants enforced at runtime:
/// - `set_embedder*` empty-DB path resets vec_index to new dim under
///   the index_write_lock — same critical section as SearchState.store
/// - reembed() cutover holds the same lock across both swaps
/// - recall paths debug_assert `state.index_embedding.dim() == vec_index.dim()`
pub struct SearchState {
    /// Provenance of vectors currently in the active index. Independent
    /// of `embedder` below.
    pub index_embedding: EmbeddingProvenance,

    /// Runtime embedder available now for embed_text(). `None` means
    /// the caller must supply pre-computed vectors. Cloning the Arc is
    /// cheap; cloning the embedder itself is expensive (model weights).
    pub embedder: Option<Arc<dyn crate::types::Embedder + Send + Sync>>,

    /// Human-readable name of the currently-loaded runtime embedder,
    /// if known. May differ from `index_embedding.name()` during the
    /// compat-attach path (e.g. legacy DB with ExternalOrUnknown
    /// provenance + a newly-attached named embedder).
    pub runtime_embedder_name: Option<String>,

    /// Stable fingerprint of the currently-loaded runtime embedder, if
    /// known. May differ from `index_embedding.digest()` during compat
    /// attachment.
    pub runtime_embedder_digest: Option<String>,

    /// Monotonic generation id. Bumps ONLY when a coherent (index +
    /// provenance) bundle is atomically published — i.e. completed
    /// reembed() or empty-DB-with-new-embedder coherent reset. Does NOT
    /// bump on runtime embedder attachment to an existing populated
    /// index. Generation 0 is the pre-reembed sentinel for "the engine
    /// has never published a new index bundle."
    pub generation: u64,

    /// High-water mark of `vec_seq` captured at the cutover barrier for
    /// the generation that built this SearchState. Every write with
    /// `vec_seq <= covers_through_seq` is reflected in the active
    /// index. Writes with `vec_seq > covers_through_seq` are replayed
    /// by the post-swap materializer.
    pub covers_through_seq: u64,

    /// HNSW M parameter. Stored on SearchState (not just on the index)
    /// so reembed can preserve / override it independently.
    pub hnsw_m: u32,
    pub hnsw_ef_construction: u32,
    pub hnsw_ef_search: u32,
}

impl SearchState {
    /// Initial SearchState for a fresh engine. Provenance is
    /// `ExternalOrUnknown { dim }` until set_embedder* or
    /// record_text() with a configured embedder populates the index
    /// with Known-provenance vectors.
    pub fn initial(
        dim: usize,
        hnsw_m: u32,
        hnsw_ef_construction: u32,
        hnsw_ef_search: u32,
    ) -> Self {
        SearchState {
            index_embedding: EmbeddingProvenance::ExternalOrUnknown { dim },
            embedder: None,
            runtime_embedder_name: None,
            runtime_embedder_digest: None,
            generation: 0,
            covers_through_seq: 0,
            hnsw_m,
            hnsw_ef_construction,
            hnsw_ef_search,
        }
    }

    /// Dimensionality of vectors in the active index. Derived from
    /// `index_embedding` — the single source of truth.
    pub fn dim(&self) -> usize {
        self.index_embedding.dim()
    }

    /// True if the runtime embedder is present and can encode text.
    /// Used by `YantrikDB::has_embedder()`.
    pub fn has_runtime_embedder(&self) -> bool {
        self.embedder.is_some()
    }
}

impl fmt::Debug for SearchState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SearchState")
            .field("index_embedding", &self.index_embedding)
            .field("runtime_embedder_name", &self.runtime_embedder_name)
            .field("runtime_embedder_digest", &self.runtime_embedder_digest)
            .field("has_embedder", &self.embedder.is_some())
            .field("generation", &self.generation)
            .field("covers_through_seq", &self.covers_through_seq)
            .field("hnsw_m", &self.hnsw_m)
            .field("hnsw_ef_construction", &self.hnsw_ef_construction)
            .field("hnsw_ef_search", &self.hnsw_ef_search)
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────
// db.reembed() — engine entry point (Layer 4 phase 1)
// ─────────────────────────────────────────────────────────────────────

use crate::error::{Result, YantrikDbError};
use crate::YantrikDB;

impl YantrikDB {
    /// **Issue #41 layer 4 — in-place embedder migration.**
    ///
    /// Re-encodes every memory under a new embedder and atomically
    /// swaps the active SearchState to the new generation. Concurrent
    /// recall traffic continues throughout (against the old generation
    /// until Swap completes); concurrent writes follow the policy in
    /// `ReembedOptions::write_policy` (default `Queue`).
    ///
    /// ## Layer 4 phase 1 scope (this commit)
    ///
    /// - Probing phase (resolve embedder name → same-name no-op check)
    /// - Dry-run path (Probing only, no state mutations beyond audit
    ///   event)
    /// - Same-name no-op detection (returns `Ok(report)` immediately
    ///   without touching anything)
    /// - `reembed_status()` API reading meta.reembed_state
    ///
    /// ## Layer 4 phase 2 (next checkpoint)
    ///
    /// - Encoding phase (batch iterate memories, encode under new
    ///   embedder via embedder-download resolver, write to staging
    ///   columns embedding_new + embedding_new_model)
    /// - Rebuilding phase (build new HNSW from staging columns)
    /// - Swapping phase (atomic ArcSwap + UPDATE memories.embedding =
    ///   embedding_new inside one transaction)
    /// - Verifying phase (sanity check + staging cleanup)
    /// - Crash recovery on open()
    ///
    /// Currently returns `Err` at the Encoding phase boundary if the
    /// call would proceed past Probing for a non-no-op reembed. The
    /// boundary is clean — meta.reembed_state is cleared on the way
    /// out so the DB is not left mid-reembed.
    pub fn reembed(
        &self,
        new_embedder_name: &str,
        options: ReembedOptions,
    ) -> Result<ReembedReport> {
        let started_at = SystemTime::now();
        let progress_cb = options.progress_cb.as_ref();
        let on_phase_complete = options.on_phase_complete.as_ref();

        // Acquire index_write_lock for the full reembed duration.
        // Serializes against concurrent set_embedder + other reembed
        // calls (single-job invariant from brainstorm-3).
        let _index_guard = self.index_write_lock.lock();

        // ── Phase 1: Probing ──
        let probing_state = self.search_state.load_full();
        let active_dim = probing_state.dim();
        let active_digest = probing_state.runtime_embedder_digest.clone();
        let active_name = probing_state
            .runtime_embedder_name
            .clone()
            .unwrap_or_default();

        let same_name = active_name == new_embedder_name;
        if same_name {
            // No-op: caller asked for the embedder already loaded.
            // Brainstorm-3 same-digest no-op rule. No generation bump,
            // no covers_through_seq change, no meta.reembed_state
            // write — fully transparent to crash recovery.
            let duration = started_at.elapsed().unwrap_or_default();
            let _ = progress_cb;
            let _ = on_phase_complete;
            return Ok(ReembedReport {
                generation: probing_state.generation,
                encoded_count: 0,
                skipped_count: 0,
                duration,
                old_embedder: active_name.clone(),
                old_embedder_digest: active_digest.clone().unwrap_or_default(),
                new_embedder: active_name,
                new_embedder_digest: active_digest.unwrap_or_default(),
                old_dim: active_dim,
                new_dim: active_dim,
                build_hwm: probing_state.covers_through_seq,
                per_namespace: HashMap::new(),
            });
        }

        let next_generation = probing_state.generation + 1;
        let probing_event_ts = systime_to_unix_secs(started_at);
        self.write_reembed_event(
            next_generation,
            ReembedPhase::Probing,
            probing_event_ts,
            &serde_json::json!({
                "old_embedder": active_name,
                "new_embedder_name": new_embedder_name,
                "old_dim": active_dim,
                "namespace": options.namespace,
            }),
        )?;

        if let Some(cb) = progress_cb {
            cb(ReembedProgress {
                phase: ReembedPhase::Probing,
                processed: 0,
                total: None,
                elapsed_ms: 0,
                namespace: options.namespace.clone(),
            });
        }

        // Dry-run path: Probing completed, return predicted shape.
        // The Probing audit event is permanent (intentional — dry-runs
        // are observable in the event log).
        if options.dry_run {
            let duration = started_at.elapsed().unwrap_or_default();
            return Ok(ReembedReport {
                generation: probing_state.generation,
                encoded_count: 0,
                skipped_count: 0,
                duration,
                old_embedder: active_name.clone(),
                old_embedder_digest: active_digest.clone().unwrap_or_default(),
                new_embedder: new_embedder_name.to_string(),
                new_embedder_digest: String::new(), // resolved in phase 2
                old_dim: active_dim,
                new_dim: active_dim,
                build_hwm: probing_state.covers_through_seq,
                per_namespace: HashMap::new(),
            });
        }

        // Persist meta.reembed_state so crash recovery on open() can
        // detect the in-flight reembed.
        self.write_reembed_state_meta(&serde_json::json!({
            "generation": next_generation,
            "phase": "Probing",
            "old_embedder": active_name,
            "new_embedder_name": new_embedder_name,
            "old_dim": active_dim,
            "started_at_unix": probing_event_ts,
            "namespace": options.namespace,
            "write_policy": match options.write_policy {
                ReembedWritePolicy::Queue => "Queue",
                ReembedWritePolicy::Pause => "Pause",
            },
        }))?;

        // ── Layer 4 phase 2 boundary ──
        // Clean abort: clear meta + write Aborted event, return Err.
        self.clear_reembed_state_meta()?;
        self.write_reembed_event(
            next_generation,
            ReembedPhase::Aborted,
            systime_to_unix_secs(SystemTime::now()),
            &serde_json::json!({
                "reason": "Encoding+Rebuilding+Swapping+Verifying not yet implemented (Layer 4 phase 2)",
            }),
        )?;

        Err(YantrikDbError::Inference(format!(
            "db.reembed() Layer 4 phase 1: Probing complete, but Encoding/Rebuilding/\
             Swapping/Verifying phases are pending implementation (issue #41 Layer 4 phase 2). \
             Dry-run mode and same-name no-op work; real migration requires the next checkpoint."
        )))
    }

    /// **Issue #41 — read-only status of an in-flight reembed.**
    /// Returns `None` if no reembed is active. Reads from
    /// `meta.reembed_state` (the durable summary row).
    ///
    /// Layer 4 phase 1: returns minimal status (phase + old/new names
    /// + start time + write policy). Phase 2 will populate the
    /// memories_total / memories_encoded counts once Encoding is
    /// implemented.
    pub fn reembed_status(&self) -> Option<ReembedStatus> {
        let conn = self.read_conn();
        let payload_json: Option<String> = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'reembed_state'",
                [],
                |row| row.get(0),
            )
            .ok();
        let payload: serde_json::Value = serde_json::from_str(&payload_json?).ok()?;

        let generation = payload.get("generation")?.as_u64()?;
        let phase = ReembedPhase::parse(payload.get("phase")?.as_str()?)?;
        let old_name = payload
            .get("old_embedder")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let new_name = payload
            .get("new_embedder_name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let old_dim = payload
            .get("old_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let started_at_unix = payload
            .get("started_at_unix")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let started_at = std::time::UNIX_EPOCH
            + std::time::Duration::from_secs_f64(started_at_unix.max(0.0));
        let write_policy = match payload.get("write_policy").and_then(|v| v.as_str()) {
            Some("Pause") => ReembedWritePolicy::Pause,
            _ => ReembedWritePolicy::Queue,
        };

        Some(ReembedStatus {
            generation,
            phase,
            old_embedder: old_name,
            old_embedder_digest: String::new(),
            new_embedder: new_name,
            new_embedder_digest: String::new(),
            old_dim,
            new_dim: old_dim,
            memories_total: 0,
            memories_encoded: 0,
            queued_writes: 0,
            checkpoint_rid: None,
            started_at,
            last_event_at: started_at,
            last_error: None,
            write_policy,
        })
    }

    /// Write a row to `reembed_events` (durable audit log). Called at
    /// each phase transition.
    pub(crate) fn write_reembed_event(
        &self,
        generation: u64,
        phase: ReembedPhase,
        timestamp_unix_secs: f64,
        payload: &serde_json::Value,
    ) -> Result<()> {
        use rusqlite::params;
        let payload_str = serde_json::to_string(payload)?;
        let conn = self.conn();
        conn.execute(
            "INSERT INTO reembed_events (generation, phase, timestamp, payload_json) \
             VALUES (?1, ?2, ?3, ?4)",
            params![
                generation as i64,
                phase.as_str(),
                timestamp_unix_secs,
                payload_str
            ],
        )?;
        Ok(())
    }

    /// Write `meta.reembed_state` with the current job summary JSON.
    pub(crate) fn write_reembed_state_meta(
        &self,
        state_json: &serde_json::Value,
    ) -> Result<()> {
        use rusqlite::params;
        let s = serde_json::to_string(state_json)?;
        let conn = self.conn();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('reembed_state', ?1)",
            params![s],
        )?;
        Ok(())
    }

    /// Clear `meta.reembed_state`, signaling no in-flight reembed.
    pub(crate) fn clear_reembed_state_meta(&self) -> Result<()> {
        let conn = self.conn();
        conn.execute("DELETE FROM meta WHERE key = 'reembed_state'", [])?;
        Ok(())
    }
}

fn systime_to_unix_secs(t: SystemTime) -> f64 {
    t.duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_string_round_trip() {
        // Every phase must survive a string round-trip via as_str + parse.
        // The strings appear in the `reembed_events.phase` column and in
        // `meta.reembed_state` JSON; they are part of the on-disk format.
        for &phase in &[
            ReembedPhase::Probing,
            ReembedPhase::Encoding,
            ReembedPhase::Rebuilding,
            ReembedPhase::Swapping,
            ReembedPhase::Verifying,
            ReembedPhase::Completed,
            ReembedPhase::Aborted,
        ] {
            let s = phase.as_str();
            let parsed = ReembedPhase::parse(s).unwrap_or_else(|| {
                panic!("ReembedPhase::parse({s:?}) returned None; persisted state would be unparseable on restart")
            });
            assert_eq!(parsed, phase, "round-trip mismatch for {phase:?}");
        }
    }

    #[test]
    fn phase_parse_rejects_unknown() {
        assert!(ReembedPhase::parse("Nonsense").is_none());
        assert!(ReembedPhase::parse("").is_none());
        // Lowercase variants are NOT accepted — the on-disk format is
        // case-sensitive. A pre-v27 binary writing lowercase would be a
        // corruption signal, not something to silently accept.
        assert!(ReembedPhase::parse("probing").is_none());
    }

    #[test]
    fn phase_terminal_classification() {
        assert!(ReembedPhase::Completed.is_terminal());
        assert!(ReembedPhase::Aborted.is_terminal());
        assert!(!ReembedPhase::Probing.is_terminal());
        assert!(!ReembedPhase::Encoding.is_terminal());
        assert!(!ReembedPhase::Rebuilding.is_terminal());
        assert!(!ReembedPhase::Swapping.is_terminal());
        assert!(!ReembedPhase::Verifying.is_terminal());
    }

    #[test]
    fn write_policy_default_is_queue() {
        // Locked by brainstorm-2 + Pranab's 2026-05-17 decision. The
        // default is what production handlers + most callers get without
        // opting in; flipping this silently would change downstream
        // behavior. If a future maintainer wants to flip the default,
        // they must read the AskUserQuestion record on issue #41 first.
        assert_eq!(ReembedWritePolicy::default(), ReembedWritePolicy::Queue);
    }

    #[test]
    fn options_default_shape_matches_locked_design() {
        let opts = ReembedOptions::default();
        assert!(opts.namespace.is_none());
        assert!(opts.progress_cb.is_none());
        assert!(opts.on_phase_complete.is_none());
        assert_eq!(opts.batch_size, 256);
        assert_eq!(opts.write_policy, ReembedWritePolicy::Queue);
        assert!(opts.hnsw_m.is_none());
        assert!(opts.hnsw_ef_construction.is_none());
        assert!(opts.hnsw_ef_search.is_none());
        assert!(opts.resume_from_checkpoint);
        assert!(!opts.dry_run);
    }

    #[test]
    fn provenance_known_dim_digest_name_accessors() {
        let p = EmbeddingProvenance::Known {
            name: Some("potion-base-2M".to_string()),
            digest: "sha256:abc123".to_string(),
            dim: 64,
        };
        assert_eq!(p.dim(), 64);
        assert_eq!(p.digest(), Some("sha256:abc123"));
        assert_eq!(p.name(), Some("potion-base-2M"));
    }

    #[test]
    fn provenance_external_or_unknown_has_dim_no_digest_no_name() {
        let p = EmbeddingProvenance::ExternalOrUnknown { dim: 384 };
        assert_eq!(p.dim(), 384);
        assert_eq!(p.digest(), None);
        assert_eq!(p.name(), None);
    }

    #[test]
    fn search_state_initial_is_external_or_unknown_with_no_embedder() {
        // Fresh engine state: index_embedding is ExternalOrUnknown (no
        // embedder has populated the index yet), embedder is None
        // (caller may pass pre-computed vectors), generation=0,
        // covers_through_seq=0.
        let s = SearchState::initial(384, 16, 200, 50);
        assert_eq!(s.dim(), 384);
        assert!(matches!(
            s.index_embedding,
            EmbeddingProvenance::ExternalOrUnknown { dim: 384 }
        ));
        assert!(!s.has_runtime_embedder());
        assert!(s.embedder.is_none());
        assert!(s.runtime_embedder_name.is_none());
        assert!(s.runtime_embedder_digest.is_none());
        assert_eq!(s.generation, 0);
        assert_eq!(s.covers_through_seq, 0);
        assert_eq!(s.hnsw_m, 16);
        assert_eq!(s.hnsw_ef_construction, 200);
        assert_eq!(s.hnsw_ef_search, 50);
    }

    // ─────────────────────────────────────────────────────────────────
    // Layer 4 phase 1: db.reembed() entry-point tests
    // ─────────────────────────────────────────────────────────────────

    /// Helper Embedder for reembed phase-1 tests — has a fingerprint
    /// so the engine's set_embedder can attach it as Known provenance.
    struct PhaseTestEmbedder {
        pub dim: usize,
        pub fp: String,
        pub name: String,
    }

    impl crate::types::Embedder for PhaseTestEmbedder {
        fn embed(
            &self,
            _text: &str,
        ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>>
        {
            Ok(vec![0.1_f32; self.dim])
        }
        fn dim(&self) -> usize {
            self.dim
        }
        fn fingerprint(&self) -> Option<String> {
            Some(self.fp.clone())
        }
        fn name(&self) -> Option<String> {
            Some(self.name.clone())
        }
    }

    #[test]
    fn reembed_same_name_is_no_op() {
        // Caller asks for the embedder already loaded — must short-
        // circuit immediately without writing meta.reembed_state or
        // even a Probing audit event, and return encoded_count=0.
        let mut db = crate::YantrikDB::new(":memory:", 8).unwrap();
        db.set_embedder(Box::new(PhaseTestEmbedder {
            dim: 8,
            fp: "sha256:same".to_string(),
            name: "model-x".to_string(),
        }))
        .unwrap();

        let report = db.reembed("model-x", ReembedOptions::default()).unwrap();
        assert_eq!(report.encoded_count, 0, "same-name no-op encodes 0");
        assert_eq!(report.skipped_count, 0);
        assert_eq!(report.old_embedder, "model-x");
        assert_eq!(report.new_embedder, "model-x");
        // No meta.reembed_state row written — confirms full no-op shape.
        assert!(db.reembed_status().is_none(), "no-op must not leave reembed_status");
    }

    #[test]
    fn reembed_dry_run_returns_predicted_report_and_clears_state() {
        // Dry-run: Probing event is written (audit trail), but
        // meta.reembed_state is NOT (no in-flight signal), and the
        // call returns the predicted ReembedReport.
        let mut db = crate::YantrikDB::new(":memory:", 8).unwrap();
        db.set_embedder(Box::new(PhaseTestEmbedder {
            dim: 8,
            fp: "sha256:original".to_string(),
            name: "original-model".to_string(),
        }))
        .unwrap();

        let opts = ReembedOptions {
            dry_run: true,
            ..Default::default()
        };
        let report = db.reembed("target-model", opts).unwrap();
        assert_eq!(report.old_embedder, "original-model");
        assert_eq!(report.new_embedder, "target-model");
        // Dry-run must NOT leave an in-flight reembed signal.
        assert!(
            db.reembed_status().is_none(),
            "dry-run must NOT write meta.reembed_state"
        );

        // The Probing audit event IS in reembed_events (dry-runs are
        // observable in the event log — intentional design choice).
        let event_count: i64 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM reembed_events WHERE phase = 'Probing'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(event_count >= 1, "Probing event must be in audit log even for dry-run");
    }

    #[test]
    fn reembed_non_no_op_fails_at_layer_4_phase_2_boundary_with_clean_state() {
        // Layer 4 phase 1 boundary check: a real (non-no-op,
        // non-dry-run) reembed call enters Probing, writes
        // meta.reembed_state, then aborts at the Encoding boundary
        // returning Err. CRITICAL: meta.reembed_state must be CLEARED
        // before the Err is returned so the DB isn't left mid-reembed.
        let mut db = crate::YantrikDB::new(":memory:", 8).unwrap();
        db.set_embedder(Box::new(PhaseTestEmbedder {
            dim: 8,
            fp: "sha256:original".to_string(),
            name: "original-model".to_string(),
        }))
        .unwrap();

        let err = db
            .reembed("target-model", ReembedOptions::default())
            .unwrap_err();
        // Specific error class — layer 4 phase 2 boundary signal.
        match err {
            crate::error::YantrikDbError::Inference(msg) => {
                assert!(
                    msg.contains("Layer 4 phase 2"),
                    "expected layer-4-phase-2 boundary message, got: {msg}"
                );
            }
            other => panic!("expected Inference error at phase 2 boundary, got {other:?}"),
        }

        // Critical: meta.reembed_state must have been cleared on the
        // way out. Else the next open() would see an in-flight reembed
        // and try to resume into an unimplemented phase.
        assert!(
            db.reembed_status().is_none(),
            "meta.reembed_state must be cleared on layer-4-phase-2 abort"
        );

        // The Aborted event MUST be in the audit log with the
        // boundary reason.
        let aborted_count: i64 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM reembed_events WHERE phase = 'Aborted'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(aborted_count, 1, "Aborted event must be logged");
    }

    #[test]
    fn search_state_dim_derives_from_provenance_not_a_separate_field() {
        // Locked by brainstorm-3: SearchState.dim() must derive from
        // index_embedding.dim(). A separate `dim` field would risk
        // drift between "what we think the dim is" and "what's
        // actually in the index." Failing this test means someone
        // re-added the standalone field.
        let s_known = SearchState {
            index_embedding: EmbeddingProvenance::Known {
                name: None,
                digest: "x".into(),
                dim: 768,
            },
            embedder: None,
            runtime_embedder_name: None,
            runtime_embedder_digest: None,
            generation: 5,
            covers_through_seq: 1000,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
        };
        assert_eq!(s_known.dim(), 768);
        let s_unknown = SearchState {
            index_embedding: EmbeddingProvenance::ExternalOrUnknown { dim: 128 },
            embedder: None,
            runtime_embedder_name: None,
            runtime_embedder_digest: None,
            generation: 0,
            covers_through_seq: 0,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
        };
        assert_eq!(s_unknown.dim(), 128);
    }
}
