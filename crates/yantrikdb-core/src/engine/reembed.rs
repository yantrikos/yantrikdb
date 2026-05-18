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
// SearchState — the single atomic swap unit
// ─────────────────────────────────────────────────────────────────────

/// The coherent (embedder + index + dim + generation + coverage + HNSW
/// params) tuple that every read path acquires once via `ArcSwap.load()`
/// and uses for the entire request. Locked by brainstorm 2 invariant 7:
/// no code path may observe a mismatched embedder-vs-index pair.
///
/// All fields are immutable after construction. Mutation is by
/// `ArcSwap<SearchState>::store(new_state)`.
///
/// **NOTE**: this struct is the future of embedder ownership on
/// `YantrikDB`. Today (pre-#41) the engine has a separate `embedder:
/// Option<Box<dyn Embedder + Send + Sync>>` field that recall + record
/// paths read directly. The reembed implementation migrates all those
/// paths to `self.search_state.load().embedder.clone()` and retires the
/// standalone field. Until that refactor lands, `SearchState` is a
/// declaration only; live read paths still use the legacy slot.
pub struct SearchState {
    /// Active embedder. Cloning the Arc is cheap; cloning the embedder
    /// itself is not (some embedders hold large model weights).
    pub embedder: Arc<dyn crate::types::Embedder + Send + Sync>,
    /// User-visible name of the embedder (e.g. "potion-base-2M").
    /// Stored in `memories.embedding_model` and `oplog.embedding_model`
    /// alongside written rows so reembed and replay can discriminate.
    pub embedder_name: String,
    /// Stronger identifier than name — the SHA-256 of the embedder's
    /// model weights. Two embedders with the same name but different
    /// weights have different digests; the no-op detection in
    /// `db.reembed()` uses digest equality, not name equality.
    pub embedder_digest: String,
    /// Embedding dimensionality. Must match the active HNSW's dim.
    pub dim: usize,
    /// Monotonic generation id. Incremented at each successful Swap.
    /// `oplog.applied_generation` references this. Generation 0 is the
    /// pre-reembed sentinel for "the engine has never reembedded."
    pub generation: u64,
    /// High-water mark of `vec_seq` captured at the cutover barrier
    /// for the generation that built this SearchState. Every write
    /// with `vec_seq <= covers_through_seq` is reflected in the
    /// `index` below. Writes with `vec_seq > covers_through_seq` are
    /// replayed by the post-swap materializer.
    pub covers_through_seq: u64,
    /// HNSW M parameter at construction time. Stored here (not just on
    /// the index) so the reembed loop can preserve / override it
    /// independently of the index's internal state.
    pub hnsw_m: u32,
    pub hnsw_ef_construction: u32,
    pub hnsw_ef_search: u32,
    // NOTE: the actual HNSW index handle (Arc<HnswIndex> or
    // Arc<DeltaIndex>) is intentionally NOT a member of this struct in
    // the v1 design. The engine's existing `vec_index: DeltaIndex`
    // field continues to own the index lifecycle. SearchState's swap
    // atomically updates the embedder + metadata; the DeltaIndex's
    // internal ArcSwap<HnswIndex> swap handles the index. Reembed
    // sequences both swaps inside the same critical section so they
    // are observed atomically by readers. This avoids a bigger refactor
    // of DeltaIndex.
    //
    // If a future redesign moves the index into SearchState, the
    // changes are: add `pub index: Arc<HnswIndex>`, retire
    // `vec_index: DeltaIndex` from YantrikDB, and the reembed swap
    // becomes a single `search_state.store(new_state)` call. That
    // change is bigger than v1 reembed scope; deferring.
}

impl fmt::Debug for SearchState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SearchState")
            .field("embedder_name", &self.embedder_name)
            .field("embedder_digest", &self.embedder_digest)
            .field("dim", &self.dim)
            .field("generation", &self.generation)
            .field("covers_through_seq", &self.covers_through_seq)
            .field("hnsw_m", &self.hnsw_m)
            .field("hnsw_ef_construction", &self.hnsw_ef_construction)
            .field("hnsw_ef_search", &self.hnsw_ef_search)
            .finish_non_exhaustive()
    }
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
}
