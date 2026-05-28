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
    ExternalOrUnknown { dim: usize },
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

    /// **Brainstorm-4 design pivot.** The active vector index lives
    /// INSIDE SearchState so search_state.store() is the single atomic
    /// publication unit. Holding two separate ArcSwaps (search_state +
    /// vec_index) would have created a split-brain window — readers
    /// could observe new embedder/dim with old index or vice versa.
    /// Moving vec_index in here eliminates the class.
    ///
    /// Arc shared with the engine's prior `vec_index` field (now
    /// retired). DeltaIndex's internal locks handle concurrent
    /// append/search; this Arc is just for ArcSwap-friendly publication.
    pub vec_index: Arc<crate::vector::delta_index::DeltaIndex>,
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
        vec_index: Arc<crate::vector::delta_index::DeltaIndex>,
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
            vec_index,
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

use rusqlite::params;

use crate::error::{Result, YantrikDbError};
use crate::serde_helpers::{deserialize_f32, serialize_f32};
use crate::vector::hnsw::HnswIndex;
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
        // **Issue #41 brainstorm-4 — Phase 2 part A.**
        // Resolve the new embedder by name via the embedder-download
        // registry, then delegate to the internal entrypoint that
        // accepts a pre-resolved Arc<dyn Embedder>. The split lets
        // tests inject synthetic embedders without round-tripping
        // through the registry, while production callers get the
        // implicit auto-load behavior hermes asked for in swarmcode
        // msg 0886504c.
        //
        // The same-name short-circuit lives in the inner entrypoint
        // so registry-resolution is skipped for the no-op case.
        let probing_state_for_name_check = self.search_state.load_full();
        let active_name_for_check = probing_state_for_name_check
            .runtime_embedder_name
            .clone()
            .unwrap_or_default();
        if active_name_for_check == new_embedder_name {
            // Same-name no-op: skip registry resolution entirely.
            return self.reembed_with_embedder(new_embedder_name, None, options);
        }
        drop(probing_state_for_name_check);

        #[cfg(feature = "embedder-download")]
        let resolved: std::sync::Arc<dyn crate::types::Embedder + Send + Sync> = {
            use crate::embedder::DownloadedEmbedder;
            let downloaded = DownloadedEmbedder::fetch(new_embedder_name).map_err(|e| {
                YantrikDbError::Inference(format!(
                    "reembed: failed to resolve embedder {new_embedder_name:?}: {e}"
                ))
            })?;
            std::sync::Arc::new(downloaded)
        };
        #[cfg(not(feature = "embedder-download"))]
        let resolved: std::sync::Arc<dyn crate::types::Embedder + Send + Sync> = {
            return Err(YantrikDbError::Inference(format!(
                "reembed by name {new_embedder_name:?} requires the `embedder-download` \
                 cargo feature (enabled by default). Slim builds must call \
                 reembed_with_embedder() directly with a Box<dyn Embedder>."
            )));
        };
        self.reembed_with_embedder(new_embedder_name, Some(resolved), options)
    }

    /// **Issue #41 brainstorm-4 — Phase 2 internal entrypoint.**
    ///
    /// Same contract as [`Self::reembed`] but skips registry
    /// resolution. When `pre_resolved` is `Some`, the embedder is
    /// used as-is (tests inject synthetic embedders here). When
    /// `None`, this is the same-name no-op path (the caller already
    /// confirmed the active embedder matches the requested name, so
    /// no embedder resolution is needed).
    ///
    /// Public-crate-visible so the test module + future callers
    /// (e.g. an offline test harness that builds embedders without
    /// the download registry) can drive Phase 2 directly.
    pub(crate) fn reembed_with_embedder(
        &self,
        new_embedder_name: &str,
        pre_resolved: Option<std::sync::Arc<dyn crate::types::Embedder + Send + Sync>>,
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

        // **Issue #41 brainstorm-4 — Phase 2 Encoding implementation.**
        //
        // Use the pre-resolved embedder injected by the caller. The
        // public reembed() entry point resolves via the embedder-
        // download registry before delegating here; test paths
        // inject synthetic embedders directly. `pre_resolved` is
        // None only on the same-name path which short-circuited
        // above; if we reach here without one, that's an engine
        // invariant break.
        let new_embedder = pre_resolved.ok_or_else(|| {
            YantrikDbError::Inference(
                "reembed_with_embedder: pre_resolved=None reached past same-name short-circuit \
                 — engine invariant violation"
                    .to_string(),
            )
        })?;

        let new_dim = new_embedder.dim();
        let new_embedder_digest = new_embedder.fingerprint().unwrap_or_default();
        let new_embedder_name_resolved = new_embedder
            .name()
            .unwrap_or_else(|| new_embedder_name.to_string());

        // **Phase 2 v1 limitation — same-dim only.** The engine's
        // standalone `embedding_dim` field is still consulted by
        // ~7 sites (engine/conflict.rs, engine/indices.rs,
        // engine/record.rs, distributed/replication.rs). Until
        // those migrate to SearchState.dim(), reembed cannot
        // safely change dim — record_with_rid's determinism gate
        // would reject post-reembed writes from new-dim leaders.
        // Cross-dim migration is the next structural increment;
        // the documented escape hatch is "open a new DB at the
        // new dim and copy memories over."
        if new_dim != active_dim {
            // Abort cleanly — Probing event was already written; emit Aborted.
            self.write_reembed_event(
                next_generation,
                ReembedPhase::Aborted,
                systime_to_unix_secs(SystemTime::now()),
                &serde_json::json!({
                    "reason": "cross-dim reembed not yet supported",
                    "active_dim": active_dim,
                    "new_dim": new_dim,
                }),
            )?;
            return Err(YantrikDbError::Inference(format!(
                "reembed: new embedder {new_embedder_name_resolved:?} dim={new_dim} \
                 differs from active dim={active_dim}. Cross-dim reembed is not yet \
                 supported (engine's standalone embedding_dim field still gates \
                 record_with_rid/replication paths). Workaround: open a new database \
                 with YantrikDB::new(path, {new_dim}) and copy memories via export/import."
            )));
        }

        // Count rows that need re-encoding. Used to populate the
        // ReembedProgress.total field for hermes's "show 0/N
        // immediately" UX ask (swarmcode msg 0886504c).
        let total_to_encode: u64 = {
            let conn = self.read_conn();
            let n: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM memories \
                     WHERE consolidation_status = 'active' \
                       AND embedding IS NOT NULL \
                       AND COALESCE(embedding_generation, 0) < ?1",
                    params![next_generation as i64],
                    |r| r.get(0),
                )
                .unwrap_or(0);
            n.max(0) as u64
        };

        if let Some(cb) = progress_cb {
            cb(ReembedProgress {
                phase: ReembedPhase::Probing,
                processed: 0,
                total: Some(total_to_encode),
                elapsed_ms: started_at.elapsed().unwrap_or_default().as_millis() as u64,
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
                skipped_count: total_to_encode,
                duration,
                old_embedder: active_name.clone(),
                old_embedder_digest: active_digest.clone().unwrap_or_default(),
                new_embedder: new_embedder_name_resolved.clone(),
                new_embedder_digest,
                old_dim: active_dim,
                new_dim,
                build_hwm: probing_state.covers_through_seq,
                per_namespace: HashMap::new(),
            });
        }

        // Persist meta.reembed_state so crash recovery on open() can
        // detect the in-flight reembed.
        self.write_reembed_state_meta(&serde_json::json!({
            "generation": next_generation,
            "phase": "Encoding",
            "old_embedder": active_name,
            "new_embedder_name": new_embedder_name_resolved,
            "old_dim": active_dim,
            "new_dim": new_dim,
            "started_at_unix": probing_event_ts,
            "total_to_encode": total_to_encode,
            "namespace": options.namespace,
            "write_policy": match options.write_policy {
                ReembedWritePolicy::Queue => "Queue",
                ReembedWritePolicy::Pause => "Pause",
            },
        }))?;

        // ── Phase 2: Encoding ──
        //
        // Scan all active rows whose embedding_generation is strictly
        // less than next_generation, re-encode each row's text under
        // the new embedder, write the result to
        // memories.embedding_new + memories.embedding_new_model.
        // The active `embedding` column is NOT touched here — that
        // happens atomically inside the Swapping SAVEPOINT in the
        // next checkpoint.
        //
        // Defensive idempotency: clear any leftover staging from a
        // prior interrupted attempt at the same target generation.
        // Without this, a retry after a partial Encoding could miss
        // rows that already have stale staging from the prior run.
        {
            let conn = self.conn();
            conn.execute(
                "UPDATE memories SET embedding_new = NULL, embedding_new_model = NULL \
                 WHERE embedding_new IS NOT NULL",
                [],
            )?;
        }

        let encoding_start_ts = systime_to_unix_secs(SystemTime::now());
        self.write_reembed_event(
            next_generation,
            ReembedPhase::Encoding,
            encoding_start_ts,
            &serde_json::json!({
                "total": total_to_encode,
                "batch_size": options.batch_size,
            }),
        )?;

        let batch_size = options.batch_size.max(1);
        let mut processed: u64 = 0;
        let mut offset: usize = 0;

        loop {
            // Read the next batch of (rid, encrypted_text).
            // Holds the read connection briefly; no embedder calls
            // happen under the conn lock.
            let batch: Vec<(String, String)> = {
                let conn = self.read_conn();
                let mut stmt = conn.prepare(
                    "SELECT rid, text FROM memories \
                     WHERE consolidation_status = 'active' \
                       AND embedding IS NOT NULL \
                       AND COALESCE(embedding_generation, 0) < ?1 \
                     ORDER BY rid \
                     LIMIT ?2 OFFSET ?3",
                )?;
                let rows = stmt
                    .query_map(
                        params![next_generation as i64, batch_size as i64, offset as i64],
                        |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
                    )?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                drop(stmt);
                drop(conn);
                rows
            };

            if batch.is_empty() {
                break;
            }

            // Re-encode each row's text under the new embedder.
            // Done OUTSIDE the conn lock — embedder.embed() can be
            // slow (model inference) and must not bottleneck SQL.
            let mut encoded_pairs: Vec<(String, Vec<u8>)> = Vec::with_capacity(batch.len());
            for (rid, stored_text) in &batch {
                let plain = self.decrypt_text(stored_text)?;
                let new_emb = new_embedder.embed(&plain).map_err(|e| {
                    YantrikDbError::Inference(format!(
                        "reembed: embedder failed on rid {rid:?}: {e}"
                    ))
                })?;
                if new_emb.len() != new_dim {
                    return Err(YantrikDbError::Inference(format!(
                        "reembed: embedder returned vector of len {} but reports dim {}; \
                         engine cannot trust this embedder",
                        new_emb.len(),
                        new_dim
                    )));
                }
                let blob = serialize_f32(&new_emb);
                let encrypted = self.encrypt_embedding(&blob)?;
                encoded_pairs.push((rid.clone(), encrypted));
            }

            // Write the batch under one SAVEPOINT so a mid-batch
            // crash leaves the prior batches' staging visible and
            // this batch's atomic. Idempotency on retry: the staging
            // clear at the start of Encoding wipes whatever this
            // batch wrote on the prior attempt.
            {
                let conn = self.conn();
                conn.execute_batch("SAVEPOINT reembed_encoding_batch")?;
                let write_result: Result<()> = (|| {
                    for (rid, encrypted) in &encoded_pairs {
                        conn.execute(
                            "UPDATE memories SET embedding_new = ?1, embedding_new_model = ?2 \
                             WHERE rid = ?3",
                            params![encrypted, new_embedder_name_resolved.as_str(), rid],
                        )?;
                    }
                    Ok(())
                })();
                match write_result {
                    Ok(()) => {
                        conn.execute_batch("RELEASE reembed_encoding_batch")?;
                    }
                    Err(e) => {
                        let _ = conn.execute_batch("ROLLBACK TO reembed_encoding_batch");
                        let _ = conn.execute_batch("RELEASE reembed_encoding_batch");
                        return Err(e);
                    }
                }
            }

            processed += encoded_pairs.len() as u64;
            offset += batch.len();

            if let Some(cb) = progress_cb {
                cb(ReembedProgress {
                    phase: ReembedPhase::Encoding,
                    processed,
                    total: Some(total_to_encode),
                    elapsed_ms: started_at.elapsed().unwrap_or_default().as_millis() as u64,
                    namespace: options.namespace.clone(),
                });
            }
        }

        // Encoding phase complete. Audit event records the final count.
        self.write_reembed_event(
            next_generation,
            ReembedPhase::Encoding,
            systime_to_unix_secs(SystemTime::now()),
            &serde_json::json!({
                "encoded_count": processed,
                "completed": true,
            }),
        )?;

        // ── Phase 2: Rebuilding ──
        //
        // Construct a new HnswIndex of size `new_dim` (== `active_dim`
        // for v1 same-dim reembeds) using the SearchState's HNSW
        // params (with caller overrides if supplied via options).
        // Insert every staged embedding_new bytes into it.
        //
        // The rebuild happens BEFORE the WriteRouter cutover so the
        // hot path stays sync-write-friendly while the (potentially
        // long) rebuild runs. Tail writes that arrive between
        // Encoding-completion and the cutover-barrier are picked up
        // in the post-barrier catch-up step below.
        let new_hnsw_m = options.hnsw_m.unwrap_or(probing_state.hnsw_m);
        let new_hnsw_efc = options
            .hnsw_ef_construction
            .unwrap_or(probing_state.hnsw_ef_construction);
        let new_hnsw_efs = options
            .hnsw_ef_search
            .unwrap_or(probing_state.hnsw_ef_search);

        let rebuilding_start_ts = systime_to_unix_secs(SystemTime::now());
        self.write_reembed_event(
            next_generation,
            ReembedPhase::Rebuilding,
            rebuilding_start_ts,
            &serde_json::json!({
                "expected_count": processed,
                "hnsw_m": new_hnsw_m,
                "hnsw_ef_construction": new_hnsw_efc,
                "hnsw_ef_search": new_hnsw_efs,
            }),
        )?;
        self.update_reembed_state_phase("Rebuilding")?;

        if let Some(cb) = progress_cb {
            cb(ReembedProgress {
                phase: ReembedPhase::Rebuilding,
                processed: 0,
                total: Some(processed),
                elapsed_ms: started_at.elapsed().unwrap_or_default().as_millis() as u64,
                namespace: options.namespace.clone(),
            });
        }

        let mut new_hnsw = HnswIndex::with_params(
            new_dim,
            new_hnsw_m as usize,
            new_hnsw_efc as usize,
            new_hnsw_efs as usize,
        );
        let mut rebuilt: u64 = 0;
        let mut rebuild_offset: usize = 0;

        loop {
            // Read staged embedding_new bytes in batches.
            let batch: Vec<(String, Vec<u8>)> = {
                let conn = self.read_conn();
                let mut stmt = conn.prepare(
                    "SELECT rid, embedding_new FROM memories \
                     WHERE consolidation_status = 'active' \
                       AND embedding_new IS NOT NULL \
                     ORDER BY rid \
                     LIMIT ?1 OFFSET ?2",
                )?;
                let rows = stmt
                    .query_map(params![batch_size as i64, rebuild_offset as i64], |r| {
                        Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                drop(stmt);
                drop(conn);
                rows
            };

            if batch.is_empty() {
                break;
            }

            for (rid, stored_blob) in &batch {
                let decrypted = self.decrypt_embedding(stored_blob)?;
                let vec = deserialize_f32(&decrypted);
                if vec.len() != new_dim {
                    return Err(YantrikDbError::Inference(format!(
                        "reembed Rebuilding: staged embedding_new for rid {rid:?} has len {} \
                         but expected new_dim {new_dim}",
                        vec.len()
                    )));
                }
                new_hnsw.insert(rid, &vec)?;
                rebuilt += 1;
            }
            rebuild_offset += batch.len();

            if let Some(cb) = progress_cb {
                cb(ReembedProgress {
                    phase: ReembedPhase::Rebuilding,
                    processed: rebuilt,
                    total: Some(processed),
                    elapsed_ms: started_at.elapsed().unwrap_or_default().as_millis() as u64,
                    namespace: options.namespace.clone(),
                });
            }
        }

        self.write_reembed_event(
            next_generation,
            ReembedPhase::Rebuilding,
            systime_to_unix_secs(SystemTime::now()),
            &serde_json::json!({
                "rebuilt_count": rebuilt,
                "completed": true,
            }),
        )?;

        // ── Phase 2: Swapping ──
        //
        // Cutover sequence (brainstorm-2 §3-§4):
        // 1. switch_to_queueing — new sync writers route to oplog
        // 2. wait_for_no_sync_writers — drain in-flight sync writers
        // 3. capture vec_seq HWM (covers_through_seq for new gen)
        // 4. tail-catchup: encode any rows that arrived AFTER the
        //    Encoding scan completed (rows with old gen + NULL
        //    embedding_new) — they're between [start_of_Encoding,
        //    cutover-barrier] window
        // 5. atomic SQL transaction: bump meta.active_generation,
        //    promote embedding_new -> embedding + stamp
        //    embedding_generation, clear staging columns
        // 6. try_publish_search_state(new_state) — the in-memory
        //    publish. Single atomic point (brainstorm-4 §1).
        // 7. switch_to_normal — resume sync writes
        //
        // Crash recovery (brainstorm-4 §6): if the engine dies
        // between SQL commit (step 5) and ArcSwap store (step 6),
        // open() reads meta.active_generation and rebuilds the
        // SearchState at the new generation. Layer 7 will lift this
        // out into the open() codepath; for now we rely on a
        // successful uninterrupted run.
        let swapping_start_ts = systime_to_unix_secs(SystemTime::now());
        self.write_reembed_event(
            next_generation,
            ReembedPhase::Swapping,
            swapping_start_ts,
            &serde_json::json!({}),
        )?;
        self.update_reembed_state_phase("Swapping")?;

        if let Some(cb) = progress_cb {
            cb(ReembedProgress {
                phase: ReembedPhase::Swapping,
                processed: 0,
                total: None,
                elapsed_ms: started_at.elapsed().unwrap_or_default().as_millis() as u64,
                namespace: options.namespace.clone(),
            });
        }

        // Cutover step 1: flip the WriteRouter to Queueing. New
        // sync writers will see the gate and route to record_queued.
        self.write_router.switch_to_queueing();

        // Cutover step 2: drain in-flight sync writers. After this
        // returns, every committed write under the OLD generation
        // is durable and visible to our tail-catchup scan below.
        //
        // Held within a closure so any of the remaining cutover
        // steps that fail can flip the router back to Normal —
        // leaving Queueing on error would block all subsequent
        // sync writes until process restart.
        self.write_router.wait_for_no_sync_writers();

        // Cutover step 3: capture vec_seq HWM. This is the
        // covers_through_seq for the new generation — every write
        // with vec_seq <= this is durable in the OLD index and now
        // (after the SQL swap) in the NEW index. Writes past this
        // seq either are queued in oplog (Queueing) or arrived
        // post-cutover and will be applied to the new gen directly.
        let covers_through_seq = self.vec_seq.load(std::sync::atomic::Ordering::Acquire);

        // Cutover step 4: tail-catchup. Encode any rows whose
        // embedding_generation is still under the new generation
        // AND embedding_new is NULL (arrived between Encoding scan
        // and the cutover barrier).
        let tail_rows: Vec<(String, String)> = {
            let conn = self.read_conn();
            let mut stmt = conn.prepare(
                "SELECT rid, text FROM memories \
                 WHERE consolidation_status = 'active' \
                   AND embedding IS NOT NULL \
                   AND embedding_new IS NULL \
                   AND COALESCE(embedding_generation, 0) < ?1",
            )?;
            let rows = stmt
                .query_map(params![next_generation as i64], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            drop(stmt);
            drop(conn);
            rows
        };

        for (rid, stored_text) in &tail_rows {
            let plain = self.decrypt_text(stored_text)?;
            let new_emb = new_embedder.embed(&plain).map_err(|e| {
                YantrikDbError::Inference(format!(
                    "reembed tail-catchup: embedder failed on rid {rid:?}: {e}"
                ))
            })?;
            if new_emb.len() != new_dim {
                self.write_router.switch_to_normal();
                return Err(YantrikDbError::Inference(format!(
                    "reembed tail-catchup: embedder returned vector of len {} but reports dim {}",
                    new_emb.len(),
                    new_dim
                )));
            }
            let blob = serialize_f32(&new_emb);
            let encrypted = self.encrypt_embedding(&blob)?;
            {
                let conn = self.conn();
                conn.execute(
                    "UPDATE memories SET embedding_new = ?1, embedding_new_model = ?2 \
                     WHERE rid = ?3",
                    params![encrypted, new_embedder_name_resolved.as_str(), rid],
                )?;
            }
            new_hnsw.insert(rid, &new_emb)?;
        }
        let tail_caught = tail_rows.len() as u64;

        // Cutover step 5: atomic SQL swap. SAVEPOINT so the entire
        // promotion is one durable unit. If COMMIT succeeds the
        // engine is durably at the new generation; if it fails the
        // staging columns are still intact for the next reembed
        // attempt.
        let total_swapped: i64 = {
            let conn = self.conn();
            conn.execute_batch("SAVEPOINT reembed_swap")?;

            let result: Result<i64> = (|| {
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('active_generation', ?1)",
                    params![next_generation.to_string()],
                )?;
                let n = conn.execute(
                    "UPDATE memories \
                     SET embedding = embedding_new, \
                         embedding_generation = ?1, \
                         embedding_new = NULL, \
                         embedding_new_model = NULL \
                     WHERE embedding_new IS NOT NULL",
                    params![next_generation as i64],
                )?;
                Ok(n as i64)
            })();

            match result {
                Ok(n) => {
                    conn.execute_batch("RELEASE reembed_swap")?;
                    n
                }
                Err(e) => {
                    let _ = conn.execute_batch("ROLLBACK TO reembed_swap");
                    let _ = conn.execute_batch("RELEASE reembed_swap");
                    self.write_router.switch_to_normal();
                    return Err(e);
                }
            }
        };

        // Cutover step 6: in-memory publish. Build a new
        // SearchState carrying:
        //   - new index_embedding (Known under the new embedder)
        //   - the same embedder Arc (we hold one already)
        //   - the new generation
        //   - covers_through_seq captured at step 3
        //   - the new HNSW wrapped in a fresh DeltaIndex
        //   - HNSW params from options or carried over
        let new_delta_index = {
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
                new_hnsw,
                delta_max,
                max_dirty_age,
            ))
        };

        let new_search_state = SearchState {
            index_embedding: EmbeddingProvenance::Known {
                name: Some(new_embedder_name_resolved.clone()),
                digest: new_embedder_digest.clone(),
                dim: new_dim,
            },
            embedder: Some(std::sync::Arc::clone(&new_embedder)),
            runtime_embedder_name: Some(new_embedder_name_resolved.clone()),
            runtime_embedder_digest: Some(new_embedder_digest.clone()),
            generation: next_generation,
            covers_through_seq,
            hnsw_m: new_hnsw_m,
            hnsw_ef_construction: new_hnsw_efc,
            hnsw_ef_search: new_hnsw_efs,
            vec_index: new_delta_index,
        };
        self.try_publish_search_state(new_search_state)?;

        // Cutover step 7: resume sync writes under the new generation.
        // The standalone WriteRouter state flip is observable to any
        // record/record_text call that arrives after this point.
        self.write_router.switch_to_normal();

        self.write_reembed_event(
            next_generation,
            ReembedPhase::Swapping,
            systime_to_unix_secs(SystemTime::now()),
            &serde_json::json!({
                "covers_through_seq": covers_through_seq,
                "tail_caught": tail_caught,
                "rows_swapped": total_swapped,
                "completed": true,
            }),
        )?;

        // ── Phase 2: Verifying ──
        //
        // Sanity check: no rows should remain under the old
        // generation with NULL embedding_new (everything either got
        // promoted or was never staged because it was tombstoned/
        // archived).
        let verifying_start_ts = systime_to_unix_secs(SystemTime::now());
        self.write_reembed_event(
            next_generation,
            ReembedPhase::Verifying,
            verifying_start_ts,
            &serde_json::json!({}),
        )?;
        self.update_reembed_state_phase("Verifying")?;

        let stragglers: i64 = {
            let conn = self.read_conn();
            conn.query_row(
                "SELECT COUNT(*) FROM memories \
                 WHERE consolidation_status = 'active' \
                   AND embedding IS NOT NULL \
                   AND COALESCE(embedding_generation, 0) < ?1",
                params![next_generation as i64],
                |r| r.get(0),
            )
            .unwrap_or(0)
        };

        if let Some(cb) = progress_cb {
            cb(ReembedProgress {
                phase: ReembedPhase::Verifying,
                processed: total_swapped as u64,
                total: Some(total_swapped as u64),
                elapsed_ms: started_at.elapsed().unwrap_or_default().as_millis() as u64,
                namespace: options.namespace.clone(),
            });
        }

        // Stragglers > 0 doesn't necessarily mean failure today —
        // Layer 5 (post-swap materializer) will drain queued writes
        // that came in during cutover. But it DOES mean Verifying
        // is incomplete; surface in the event payload so operators
        // can decide.
        self.write_reembed_event(
            next_generation,
            ReembedPhase::Verifying,
            systime_to_unix_secs(SystemTime::now()),
            &serde_json::json!({
                "stragglers_under_old_gen": stragglers,
                "completed": true,
            }),
        )?;

        // Final Completed event + clear meta.reembed_state.
        self.clear_reembed_state_meta()?;
        self.write_reembed_event(
            next_generation,
            ReembedPhase::Completed,
            systime_to_unix_secs(SystemTime::now()),
            &serde_json::json!({
                "encoded_count": processed,
                "tail_caught": tail_caught,
                "rows_swapped": total_swapped,
                "covers_through_seq": covers_through_seq,
            }),
        )?;

        let _ = on_phase_complete; // future use for per-phase callback dispatch

        let duration = started_at.elapsed().unwrap_or_default();
        Ok(ReembedReport {
            generation: next_generation,
            encoded_count: processed + tail_caught,
            skipped_count: 0,
            duration,
            old_embedder: active_name,
            old_embedder_digest: active_digest.unwrap_or_default(),
            new_embedder: new_embedder_name_resolved,
            new_embedder_digest,
            old_dim: active_dim,
            new_dim,
            build_hwm: covers_through_seq,
            per_namespace: HashMap::new(),
        })
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
        let old_dim = payload.get("old_dim").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let started_at_unix = payload
            .get("started_at_unix")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let started_at =
            std::time::UNIX_EPOCH + std::time::Duration::from_secs_f64(started_at_unix.max(0.0));
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
    pub(crate) fn write_reembed_state_meta(&self, state_json: &serde_json::Value) -> Result<()> {
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

    /// **Issue #41 brainstorm-4 Phase 2 helper.** Mutate the `phase`
    /// field on the durable `meta.reembed_state` row to reflect
    /// progression through Probing → Encoding → Rebuilding →
    /// Swapping → Verifying. Other fields (generation, embedder
    /// names, dims, namespace, write_policy) are preserved.
    ///
    /// Used by reembed_with_embedder at each phase transition so
    /// crash recovery on open() can read the durable phase marker
    /// and apply the right resume strategy (Layer 7).
    pub(crate) fn update_reembed_state_phase(&self, new_phase: &str) -> Result<()> {
        use rusqlite::params;
        let conn = self.conn();
        let raw: Option<String> = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'reembed_state'",
                [],
                |r| r.get::<_, String>(0),
            )
            .ok();
        let Some(s) = raw else {
            // No durable row to update — clear-state path took it.
            return Ok(());
        };
        let mut state: serde_json::Value =
            serde_json::from_str(&s).unwrap_or_else(|_| serde_json::json!({}));
        if let Some(map) = state.as_object_mut() {
            map.insert(
                "phase".to_string(),
                serde_json::Value::String(new_phase.to_string()),
            );
        }
        let s_new = serde_json::to_string(&state)?;
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('reembed_state', ?1)",
            params![s_new],
        )?;
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
        let vec_index = Arc::new(crate::vector::delta_index::DeltaIndex::new(384));
        let s = SearchState::initial(384, 16, 200, 50, vec_index);
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
        // Brainstorm-4: SearchState owns its DeltaIndex; the freshly-
        // constructed initial state has a delta-only index with no
        // entries yet.
        assert_eq!(s.vec_index.len(), 0);
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
        ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
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

    /// Sentinel variant for Layer 5 drain tests. The first element
    /// of the produced vector is set to `sentinel`, so tests can
    /// assert "this embedding came from THIS embedder" by checking
    /// vec[0].
    struct PhaseTestEmbedderSentinel {
        pub dim: usize,
        pub fp: String,
        pub name: String,
        pub sentinel: f32,
    }

    impl crate::types::Embedder for PhaseTestEmbedderSentinel {
        fn embed(
            &self,
            _text: &str,
        ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
            let mut v = vec![0.0_f32; self.dim];
            if !v.is_empty() {
                v[0] = self.sentinel;
            }
            Ok(v)
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
        assert!(
            db.reembed_status().is_none(),
            "no-op must not leave reembed_status"
        );
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

        // Use the internal entrypoint with an injected synthetic
        // embedder — the public reembed() resolves through the
        // embedder-download registry which is feature-gated and
        // off in default test builds.
        let synthetic: std::sync::Arc<dyn crate::types::Embedder + Send + Sync> =
            std::sync::Arc::new(PhaseTestEmbedder {
                dim: 8,
                fp: "sha256:target".to_string(),
                name: "target-model".to_string(),
            });
        let opts = ReembedOptions {
            dry_run: true,
            ..Default::default()
        };
        let report = db
            .reembed_with_embedder("target-model", Some(synthetic), opts)
            .unwrap();
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
        assert!(
            event_count >= 1,
            "Probing event must be in audit log even for dry-run"
        );
    }

    #[test]
    fn reembed_phase_2_full_run_promotes_to_new_generation() {
        // **Checkpoint 17 — Phase 2 full happy path.** Plant rows,
        // run reembed end-to-end (Probing → Encoding → Rebuilding
        // → Swapping → Verifying → Completed). Critical contracts:
        //   - ReembedReport.encoded_count matches planted row count
        //   - SearchState.generation advanced by exactly 1
        //   - SearchState.runtime_embedder is the new one
        //   - meta.active_generation in SQL matches new gen
        //   - Every active row's embedding_generation = new gen
        //   - Staging columns (embedding_new + embedding_new_model)
        //     are cleared post-swap
        //   - meta.reembed_state is cleared
        //   - Completed event is in audit log
        //   - WriteRouter is back in Normal state
        let mut db = crate::YantrikDB::new(":memory:", 8).unwrap();
        db.set_embedder(Box::new(PhaseTestEmbedder {
            dim: 8,
            fp: "sha256:original".to_string(),
            name: "original-model".to_string(),
        }))
        .unwrap();

        // Plant 3 rows so we exercise batching semantics.
        for i in 0..3 {
            db.record(
                &format!("row {i}"),
                "episodic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &vec![0.1_f32; 8],
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        }

        let synthetic: std::sync::Arc<dyn crate::types::Embedder + Send + Sync> =
            std::sync::Arc::new(PhaseTestEmbedder {
                dim: 8,
                fp: "sha256:target".to_string(),
                name: "target-model".to_string(),
            });
        let report = db
            .reembed_with_embedder("target-model", Some(synthetic), ReembedOptions::default())
            .unwrap();

        // ReembedReport contract.
        assert_eq!(report.generation, 1, "new generation = 1 (was 0)");
        assert_eq!(report.encoded_count, 3, "all 3 rows re-encoded");
        assert_eq!(report.old_embedder, "original-model");
        assert_eq!(report.new_embedder, "target-model");
        assert_eq!(report.old_dim, 8);
        assert_eq!(report.new_dim, 8);
        assert!(
            report.build_hwm >= 3,
            "covers_through_seq covers all writes"
        );

        // In-memory SearchState.
        let state = db.search_state.load_full();
        assert_eq!(state.generation, 1, "in-memory generation advanced");
        assert_eq!(state.runtime_embedder_name.as_deref(), Some("target-model"));
        assert!(
            matches!(
                &state.index_embedding,
                EmbeddingProvenance::Known { digest, .. } if digest == "sha256:target"
            ),
            "provenance is Known under the new embedder digest"
        );

        // Durable state.
        let conn = db.conn();
        let active_gen: String = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'active_generation'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(active_gen, "1", "meta.active_generation bumped in SQL");

        let row_gens: Vec<i64> = conn
            .prepare(
                "SELECT embedding_generation FROM memories WHERE consolidation_status = 'active'",
            )
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(row_gens, vec![1, 1, 1], "every row stamped at new gen");

        let staging_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories \
                 WHERE embedding_new IS NOT NULL OR embedding_new_model IS NOT NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            staging_count, 0,
            "staging columns must be cleared post-swap"
        );

        // No mid-reembed signal lingering.
        drop(conn);
        assert!(
            db.reembed_status().is_none(),
            "meta.reembed_state must be cleared on Completed"
        );

        // Completed event in audit log.
        let conn = db.conn();
        let completed_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM reembed_events WHERE phase = 'Completed'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(completed_count, 1, "Completed event must be logged");

        // WriteRouter back in Normal state.
        assert!(
            db.write_router.try_enter_sync_writer().is_some(),
            "WriteRouter must be Normal after reembed completion"
        );
    }

    #[test]
    fn layer_5_materializer_drains_queued_record_under_new_embedder() {
        // **Issue #41 Layer 5 — full Queue-mode integration.**
        // Flow:
        //   1. Engine at gen 0 with embedder E0 ("sentinel=0.42").
        //   2. Flip WriteRouter to Queueing manually (simulating
        //      a reembed in flight at the cutover preamble).
        //   3. record_text → revalidation loop sees Queueing →
        //      routes to record_queued → writes oplog applied=0
        //      with embedding_model="E0-name" and TEXT payload.
        //   4. Manually swap SearchState to gen 1 with embedder E1
        //      ("sentinel=0.99") via try_publish_search_state +
        //      flip router back to Normal (mimics what Phase 2 does).
        //   5. Run apply_pending_ops_once. Layer 5 picks up the
        //      queued op, re-encodes the text under E1, INSERTs
        //      the memory row at embedding_generation=1, appends
        //      to the active vec_index.
        //   6. Assert: memories table now has the row with
        //      embedding_generation=1, the stored bytes encode to
        //      0.99 (E1's signature, not 0.42 of E0), oplog row
        //      is applied=1.
        use std::sync::Arc;
        let mut db = crate::YantrikDB::new(":memory:", 8).unwrap();

        // Configure E0 as active.
        db.set_embedder(Box::new(PhaseTestEmbedderSentinel {
            dim: 8,
            fp: "sha256:E0".to_string(),
            name: "E0-name".to_string(),
            sentinel: 0.42,
        }))
        .unwrap();

        // Simulate reembed cutover preamble: flip router to Queueing.
        db.write_router.switch_to_queueing();

        // record_text routes through the queue path.
        let queued_rid = db
            .record_text(
                "queue-drain-test-text",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({"k": "v"}),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        // memories table is NOT yet written — queue path stores text in oplog.
        let mem_count: i64 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE rid = ?1",
                [&queued_rid],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(mem_count, 0, "queue path does not write memories yet");

        // Now simulate Phase 2 completing the swap: build a new
        // SearchState at gen 1 with embedder E1, publish via CAS,
        // flip router back to Normal.
        let old_state = db.search_state.load_full();
        let e1: Arc<dyn crate::types::Embedder + Send + Sync> =
            Arc::new(PhaseTestEmbedderSentinel {
                dim: 8,
                fp: "sha256:E1".to_string(),
                name: "E1-name".to_string(),
                sentinel: 0.99,
            });
        let new_state = SearchState {
            index_embedding: EmbeddingProvenance::Known {
                name: Some("E1-name".to_string()),
                digest: "sha256:E1".to_string(),
                dim: 8,
            },
            embedder: Some(Arc::clone(&e1)),
            runtime_embedder_name: Some("E1-name".to_string()),
            runtime_embedder_digest: Some("sha256:E1".to_string()),
            generation: 1,
            covers_through_seq: old_state.covers_through_seq,
            hnsw_m: old_state.hnsw_m,
            hnsw_ef_construction: old_state.hnsw_ef_construction,
            hnsw_ef_search: old_state.hnsw_ef_search,
            vec_index: Arc::clone(&old_state.vec_index),
        };
        db.try_publish_search_state(new_state).unwrap();
        db.write_router.switch_to_normal();

        // Layer 5 drain: materializer picks up the queued op and
        // re-encodes under E1.
        let n_applied = db.apply_pending_ops_once(100).unwrap();
        assert!(
            n_applied >= 1,
            "Layer 5 materializer must drain the queued record (applied >= 1, got {n_applied})"
        );

        // Memories row exists, stamped with new generation, encoded
        // under E1 (sentinel 0.99).
        let conn = db.conn();
        let (row_gen, stored_emb_blob): (i64, Vec<u8>) = conn
            .query_row(
                "SELECT embedding_generation, embedding FROM memories WHERE rid = ?1",
                [&queued_rid],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(row_gen, 1, "Layer 5 stamped row at new generation 1");
        let plain = db.decrypt_embedding(&stored_emb_blob).unwrap();
        let vec = crate::serde_helpers::deserialize_f32(&plain);
        assert!(
            (vec[0] - 0.99).abs() < 1e-6,
            "row encoded under E1 (sentinel 0.99), got vec[0]={}",
            vec[0]
        );

        // Oplog op is applied=1.
        let applied: i64 = conn
            .query_row(
                "SELECT applied FROM oplog WHERE target_rid = ?1 AND op_type = 'record'",
                [&queued_rid],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(applied, 1, "queued oplog row marked applied=1 after drain");
    }

    #[test]
    fn layer_5_defers_drain_while_reembed_in_flight() {
        // **Layer 5 invariant — pause-during-reembed.** While
        // meta.reembed_state is set (reembed mid-flight), Layer 5
        // must NOT drain queued reembed writes. It returns
        // applied=0 (the op stays pending) so the next tick after
        // reembed completes picks it up.
        let mut db = crate::YantrikDB::new(":memory:", 8).unwrap();
        db.set_embedder(Box::new(PhaseTestEmbedderSentinel {
            dim: 8,
            fp: "sha256:E0".to_string(),
            name: "E0-name".to_string(),
            sentinel: 0.42,
        }))
        .unwrap();

        // Flip to Queueing + record_text → queued op.
        db.write_router.switch_to_queueing();
        let queued_rid = db
            .record_text(
                "deferred-drain-test",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        // Manually write meta.reembed_state to simulate reembed in flight.
        db.write_reembed_state_meta(&serde_json::json!({
            "generation": 1,
            "phase": "Encoding",
        }))
        .unwrap();

        // Drain should NOT touch the queued op.
        let _ = db.apply_pending_ops_once(100).unwrap();

        // Oplog row still applied=0.
        let applied: i64 = db
            .conn()
            .query_row(
                "SELECT applied FROM oplog WHERE target_rid = ?1 AND op_type = 'record'",
                [&queued_rid],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            applied, 0,
            "Layer 5 must defer drain while reembed in flight; got applied={applied}"
        );
    }

    #[test]
    fn reembed_phase_2_rejects_cross_dim_with_clear_error() {
        // **Phase 2 v1 limitation.** Same-dim reembeds only;
        // cross-dim requires retiring the engine's standalone
        // `embedding_dim` field which is its own structural debt.
        // The error message must point the caller at the
        // documented escape hatch.
        let mut db = crate::YantrikDB::new(":memory:", 8).unwrap();
        db.set_embedder(Box::new(PhaseTestEmbedder {
            dim: 8,
            fp: "sha256:active".to_string(),
            name: "active".to_string(),
        }))
        .unwrap();

        let cross_dim: std::sync::Arc<dyn crate::types::Embedder + Send + Sync> =
            std::sync::Arc::new(PhaseTestEmbedder {
                dim: 16, // different from active dim 8
                fp: "sha256:cross".to_string(),
                name: "cross-dim-target".to_string(),
            });
        let err = db
            .reembed_with_embedder(
                "cross-dim-target",
                Some(cross_dim),
                ReembedOptions::default(),
            )
            .unwrap_err();
        match err {
            crate::error::YantrikDbError::Inference(msg) => {
                assert!(
                    msg.contains("Cross-dim reembed is not yet supported"),
                    "expected cross-dim guard message, got: {msg}"
                );
                assert!(
                    msg.contains("dim=8") && msg.contains("dim=16"),
                    "msg: {msg}"
                );
            }
            other => panic!("expected Inference, got {other:?}"),
        }
        // No mid-reembed state leftover.
        assert!(db.reembed_status().is_none());
    }

    #[test]
    fn reembed_phase_2_part_a_with_progress_callback_emits_total() {
        // **Hermes ask 1 (swarmcode msg 0886504c).** Probing must
        // emit a progress event with `total` populated (not None),
        // so CLIs can show "0/N" immediately before Encoding
        // starts. Locks that the count happens BEFORE the first
        // progress callback fires.
        let mut db = crate::YantrikDB::new(":memory:", 8).unwrap();
        db.set_embedder(Box::new(PhaseTestEmbedder {
            dim: 8,
            fp: "sha256:active".to_string(),
            name: "active".to_string(),
        }))
        .unwrap();

        // Plant 3 rows.
        for i in 0..3 {
            db.record(
                &format!("row {i}"),
                "episodic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &vec![0.1_f32; 8],
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        }

        use std::sync::{Arc, Mutex};
        let events: Arc<Mutex<Vec<ReembedProgress>>> = Arc::new(Mutex::new(Vec::new()));
        let events_clone = Arc::clone(&events);
        let progress_cb: Box<dyn Fn(ReembedProgress) + Send + Sync> =
            Box::new(move |p| events_clone.lock().unwrap().push(p));

        let synthetic: std::sync::Arc<dyn crate::types::Embedder + Send + Sync> =
            std::sync::Arc::new(PhaseTestEmbedder {
                dim: 8,
                fp: "sha256:target".to_string(),
                name: "target".to_string(),
            });
        let opts = ReembedOptions {
            progress_cb: Some(progress_cb),
            ..Default::default()
        };
        // Encoding+abort path; we only care about progress events.
        let _ = db.reembed_with_embedder("target", Some(synthetic), opts);

        let captured = events.lock().unwrap().clone();
        assert!(!captured.is_empty(), "at least one progress event fired");
        // First event must be Probing with total=Some(3).
        let first = &captured[0];
        assert!(
            matches!(first.phase, ReembedPhase::Probing),
            "first event must be Probing, got {:?}",
            first.phase
        );
        assert_eq!(
            first.total,
            Some(3),
            "Probing must populate total so CLIs show 0/N immediately (hermes ask 1)"
        );
        assert_eq!(first.processed, 0, "Probing fires with processed=0");

        // Subsequent Encoding events must have monotonic processed
        // and the same total.
        let encoding_events: Vec<&ReembedProgress> = captured
            .iter()
            .filter(|e| matches!(e.phase, ReembedPhase::Encoding))
            .collect();
        for e in &encoding_events {
            assert_eq!(e.total, Some(3));
            assert!(e.processed <= 3);
        }
        if let Some(last) = encoding_events.last() {
            assert_eq!(last.processed, 3, "final Encoding event reflects all rows");
        }
    }

    #[test]
    fn search_state_dim_derives_from_provenance_not_a_separate_field() {
        // Locked by brainstorm-3: SearchState.dim() must derive from
        // index_embedding.dim(). A separate `dim` field would risk
        // drift between "what we think the dim is" and "what's
        // actually in the index." Failing this test means someone
        // re-added the standalone field.
        let vec_index_known = Arc::new(crate::vector::delta_index::DeltaIndex::new(768));
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
            vec_index: vec_index_known,
        };
        assert_eq!(s_known.dim(), 768);
        let vec_index_unknown = Arc::new(crate::vector::delta_index::DeltaIndex::new(128));
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
            vec_index: vec_index_unknown,
        };
        assert_eq!(s_unknown.dim(), 128);
    }

    // ─────────────────────────────────────────────────────────────
    // Layer 8 — functional matrix. Integration tests covering the
    // happy paths + corner cases that aren't structurally proven
    // by checkpoint-9–19 unit tests.
    // ─────────────────────────────────────────────────────────────

    #[test]
    fn reembed_post_swap_recall_returns_rows_under_new_embedder() {
        // **Layer 8 end-to-end correctness.** After db.reembed(),
        // a recall against the new embedder must return the
        // memory rows that existed before the swap. Locks the
        // "user-facing query still works" contract — the most
        // important property the feature delivers.
        use std::sync::Arc;
        let mut db = crate::YantrikDB::new(":memory:", 8).unwrap();

        // E0: sentinel 0.42 (every vec has [0.42, 0, ...]).
        db.set_embedder(Box::new(PhaseTestEmbedderSentinel {
            dim: 8,
            fp: "sha256:E0".to_string(),
            name: "E0-name".to_string(),
            sentinel: 0.42,
        }))
        .unwrap();

        // Plant 5 rows via record_text so the engine computes
        // embeddings under E0.
        let rids: Vec<String> = (0..5)
            .map(|i| {
                db.record_text(
                    &format!("layer-8 row {i}"),
                    "episodic",
                    0.5,
                    0.0,
                    604800.0,
                    &serde_json::json!({"i": i}),
                    "default",
                    0.8,
                    "general",
                    "user",
                    None,
                )
                .unwrap()
            })
            .collect();

        // Run a full reembed under E1.
        let e1: Arc<dyn crate::types::Embedder + Send + Sync> =
            Arc::new(PhaseTestEmbedderSentinel {
                dim: 8,
                fp: "sha256:E1".to_string(),
                name: "E1-name".to_string(),
                sentinel: 0.99,
            });
        let report = db
            .reembed_with_embedder("E1-name", Some(e1), ReembedOptions::default())
            .unwrap();
        assert_eq!(report.generation, 1);
        assert_eq!(report.encoded_count, 5);

        // Recall under E1's vector space.
        let query = vec![0.99_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = db
            .recall(
                &query, 5, None, None, false, true, None, true, None, None, None, None, None,
            )
            .unwrap();
        assert_eq!(
            results.len(),
            5,
            "recall must return all 5 planted rows after reembed; got {}",
            results.len()
        );
        for r in &results {
            assert!(
                rids.contains(&r.rid),
                "recall returned unexpected rid {:?}",
                r.rid
            );
        }
    }

    #[test]
    fn reembed_on_empty_engine_returns_gracefully() {
        // **Layer 8 corner case.** Fresh engine with no rows: a
        // full reembed should run through every phase and return
        // a clean report with encoded_count=0.
        use std::sync::Arc;
        let mut db = crate::YantrikDB::new(":memory:", 8).unwrap();
        db.set_embedder(Box::new(PhaseTestEmbedderSentinel {
            dim: 8,
            fp: "sha256:E0".to_string(),
            name: "E0".to_string(),
            sentinel: 0.42,
        }))
        .unwrap();
        let e1: Arc<dyn crate::types::Embedder + Send + Sync> =
            Arc::new(PhaseTestEmbedderSentinel {
                dim: 8,
                fp: "sha256:E1".to_string(),
                name: "E1".to_string(),
                sentinel: 0.99,
            });
        let report = db
            .reembed_with_embedder("E1", Some(e1), ReembedOptions::default())
            .unwrap();
        assert_eq!(report.encoded_count, 0);
        assert_eq!(report.generation, 1);
        // All phase audit events present even for the empty case.
        let conn = db.conn();
        for phase in [
            "Probing",
            "Encoding",
            "Rebuilding",
            "Swapping",
            "Verifying",
            "Completed",
        ] {
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM reembed_events WHERE phase = ?1 AND generation = 1",
                    params![phase],
                    |r| r.get(0),
                )
                .unwrap();
            assert!(
                count >= 1,
                "expected {phase} event for empty-engine reembed (got {count})"
            );
        }
    }

    #[test]
    fn reembed_sequential_runs_advance_generation_monotonically() {
        // **Layer 8 corner case.** Two back-to-back reembeds.
        // Generation must advance 0 → 1 → 2; the second reembed
        // sees rows stamped at gen 1 and re-encodes them again.
        use std::sync::Arc;
        let mut db = crate::YantrikDB::new(":memory:", 8).unwrap();
        db.set_embedder(Box::new(PhaseTestEmbedderSentinel {
            dim: 8,
            fp: "sha256:E0".to_string(),
            name: "E0".to_string(),
            sentinel: 0.10,
        }))
        .unwrap();
        let rid = db
            .record_text(
                "seq",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        // First reembed: gen 0 → 1.
        let e1: Arc<dyn crate::types::Embedder + Send + Sync> =
            Arc::new(PhaseTestEmbedderSentinel {
                dim: 8,
                fp: "sha256:E1".to_string(),
                name: "E1".to_string(),
                sentinel: 0.50,
            });
        let r1 = db
            .reembed_with_embedder("E1", Some(e1), ReembedOptions::default())
            .unwrap();
        assert_eq!(r1.generation, 1);

        // Second reembed: gen 1 → 2.
        let e2: Arc<dyn crate::types::Embedder + Send + Sync> =
            Arc::new(PhaseTestEmbedderSentinel {
                dim: 8,
                fp: "sha256:E2".to_string(),
                name: "E2".to_string(),
                sentinel: 0.90,
            });
        let r2 = db
            .reembed_with_embedder("E2", Some(e2), ReembedOptions::default())
            .unwrap();
        assert_eq!(r2.generation, 2);
        assert_eq!(
            r2.encoded_count, 1,
            "second reembed re-encodes the planted row"
        );

        // Row's stamp is the latest generation.
        let row_gen: i64 = db
            .conn()
            .query_row(
                "SELECT embedding_generation FROM memories WHERE rid = ?1",
                [&rid],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(row_gen, 2);

        // Engine SearchState is at gen 2.
        assert_eq!(db.search_state.load().generation, 2);
    }

    #[test]
    fn reembed_audit_event_sequence_is_complete_and_ordered() {
        // **Layer 8 audit-trail lock.** A successful reembed must
        // emit the full state-machine event sequence in order:
        // Probing → Encoding → Rebuilding → Swapping → Verifying
        // → Completed. Per phase there are start + completion
        // events; Probing emits one (start only). The audit log
        // is what operators read post-incident to understand
        // what happened.
        use std::sync::Arc;
        let mut db = crate::YantrikDB::new(":memory:", 8).unwrap();
        db.set_embedder(Box::new(PhaseTestEmbedderSentinel {
            dim: 8,
            fp: "sha256:E0".to_string(),
            name: "E0".to_string(),
            sentinel: 0.10,
        }))
        .unwrap();
        let _ = db.record_text(
            "audit",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({}),
            "default",
            0.8,
            "general",
            "user",
            None,
        );

        let e1: Arc<dyn crate::types::Embedder + Send + Sync> =
            Arc::new(PhaseTestEmbedderSentinel {
                dim: 8,
                fp: "sha256:E1".to_string(),
                name: "E1".to_string(),
                sentinel: 0.50,
            });
        let _ = db
            .reembed_with_embedder("E1", Some(e1), ReembedOptions::default())
            .unwrap();

        // Read all events for gen 1 in timestamp order.
        let phases: Vec<String> = {
            let conn = db.conn();
            let mut stmt = conn
                .prepare("SELECT phase FROM reembed_events WHERE generation = 1 ORDER BY timestamp")
                .unwrap();
            stmt.query_map([], |r| r.get::<_, String>(0))
                .unwrap()
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap()
        };

        // First event is Probing; subsequent events include all phases.
        assert_eq!(phases.first().map(String::as_str), Some("Probing"));
        for required in [
            "Encoding",
            "Rebuilding",
            "Swapping",
            "Verifying",
            "Completed",
        ] {
            assert!(
                phases.iter().any(|p| p == required),
                "audit log missing {required} event; phases recorded: {phases:?}"
            );
        }
        // Completed is the last meaningful event (followups would
        // come from a NEXT reembed at gen 2).
        let last_meaningful = phases.last().map(String::as_str);
        assert_eq!(
            last_meaningful,
            Some("Completed"),
            "Completed must be the terminal event for the reembed run; got {last_meaningful:?}"
        );
    }

    #[test]
    fn reembed_re_encodes_rows_from_multiple_namespaces() {
        // **Layer 8 multi-namespace coverage.** A reembed without
        // a `namespace` option re-encodes EVERY active row
        // regardless of namespace. Locks the cross-namespace
        // uniformity invariant — without this, a reembed could
        // silently skip namespaces the agent forgot to pass
        // (a bug class the user hits when namespaces are
        // dynamically created).
        use std::sync::Arc;
        let mut db = crate::YantrikDB::new(":memory:", 8).unwrap();
        db.set_embedder(Box::new(PhaseTestEmbedderSentinel {
            dim: 8,
            fp: "sha256:E0".to_string(),
            name: "E0".to_string(),
            sentinel: 0.10,
        }))
        .unwrap();

        for ns in ["alpha", "beta", "gamma"] {
            for i in 0..2 {
                db.record_text(
                    &format!("{ns} row {i}"),
                    "episodic",
                    0.5,
                    0.0,
                    604800.0,
                    &serde_json::json!({}),
                    ns,
                    0.8,
                    "general",
                    "user",
                    None,
                )
                .unwrap();
            }
        }

        let e1: Arc<dyn crate::types::Embedder + Send + Sync> =
            Arc::new(PhaseTestEmbedderSentinel {
                dim: 8,
                fp: "sha256:E1".to_string(),
                name: "E1".to_string(),
                sentinel: 0.90,
            });
        let report = db
            .reembed_with_embedder("E1", Some(e1), ReembedOptions::default())
            .unwrap();
        assert_eq!(
            report.encoded_count, 6,
            "6 rows across 3 namespaces must all be re-encoded"
        );

        // Every row in every namespace has embedding_generation = 1.
        let conn = db.conn();
        for ns in ["alpha", "beta", "gamma"] {
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM memories WHERE namespace = ?1 \
                     AND embedding_generation = 1 AND consolidation_status = 'active'",
                    params![ns],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(
                count, 2,
                "namespace {ns} must have 2 rows at gen 1, got {count}"
            );
        }
    }
}
