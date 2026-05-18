use thiserror::Error;

#[derive(Error, Debug)]
pub enum YantrikDbError {
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("No embedder configured. Pass an embedder to YantrikDB() or call set_embedder().")]
    NoEmbedder,

    #[error("Must provide either query or query_embedding")]
    NoQuery,

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("memory not found: {0}")]
    NotFound(String),

    #[error("sync error: {0}")]
    SyncError(String),

    #[error("invalid HLC timestamp: {0}")]
    HlcParseError(String),

    #[error("encryption error: {0}")]
    Encryption(String),

    #[error("model loading error: {0}")]
    ModelLoad(String),

    #[error("inference error: {0}")]
    Inference(String),

    #[error("session conflict: {0}")]
    SessionConflict(String),

    /// **Issue #41 / brainstorm-3.** `set_embedder*` was called with a
    /// candidate embedder whose dimensionality differs from the active
    /// index. On a populated DB this would produce silent corruption
    /// (next insert mismatches HNSW's expected dim) so the engine
    /// rejects. Caller's correct path is `db.reembed(new_embedder_name)`
    /// which rebuilds the index under the new dim.
    #[error(
        "embedder dimensionality change requires db.reembed(): \
         active index dim is {active_dim} ({memory_count} memories already indexed); \
         candidate embedder dim is {candidate_dim}. \
         Call db.reembed(new_embedder_name) to rebuild the index under the new dim."
    )]
    ChangeEmbedderDimensionRequiresReembed {
        active_dim: usize,
        candidate_dim: usize,
        memory_count: u64,
    },

    /// **Issue #41 / brainstorm-3.** `set_embedder*` was called with a
    /// candidate embedder whose fingerprint differs from the active
    /// index's embedder, even though dim matches. Same-dim-different-
    /// embedder is the silent-corruption case: queries encode under E1,
    /// stored vectors are in E0's space, no panic but bad results.
    /// Caller's correct path is `db.reembed(new_embedder_name)` to
    /// re-encode existing memories under the new embedder.
    ///
    /// Returned only when the active index has `Known`-provenance. For
    /// `ExternalOrUnknown` provenance (legacy DBs / external vector
    /// imports) the same dim is accepted as a compat-attach without
    /// claiming new provenance.
    #[error(
        "embedder change requires db.reembed(): \
         active index built with embedder digest {active_digest:?}, \
         candidate digest is {candidate_digest:?} (dim {dim} matches but model differs); \
         {memory_count} memories already indexed in old embedder's vector space. \
         Call db.reembed(new_embedder_name) to re-encode safely; \
         a plain set_embedder() with a different model on a populated index \
         would silently corrupt search results."
    )]
    ChangeEmbedderDigestRequiresReembed {
        active_digest: Option<String>,
        candidate_digest: Option<String>,
        dim: usize,
        memory_count: u64,
    },

    /// **Decoupled write path RFC, Phase 1.**
    ///
    /// The bounded global ingest queue is full. Foreground writers receive
    /// this synchronously when `log_op_pending()` would push a pending op
    /// past `MAX_PENDING_OPS`. Caller policy: retry with backoff after the
    /// hint, surface to user as 503-like, or shed the write.
    ///
    /// `retry_after_ms` is a coarse hint — actual drain rate depends on
    /// background worker throughput.
    #[error(
        "ingest queue full ({pending} pending ops, max={max}); retry after {retry_after_ms}ms"
    )]
    Backpressure {
        pending: i64,
        max: i64,
        retry_after_ms: u64,
    },

    /// **Phase 6 RYW**: caller-requested visible_seq was not reached within
    /// the timeout window. Either the write that should have bumped the seq
    /// has not yet materialized (legitimate timeout — caller should retry or
    /// fall back to non-strict recall) or the seq is from another namespace
    /// / a different engine instance (caller error — retry will not help).
    ///
    /// Field naming aligns with the yantrikdb-server cluster RYW design
    /// (msg 0c2bea4a, 2026-05-07): `requested_seq` is what the caller asked
    /// to wait for (typically the openraft commit-log index of their write);
    /// `observed_seq` is what visible_seq[namespace] held when we gave up;
    /// `waited_ms` is the configured timeout we waited.
    #[error("read-your-writes timeout: visible_seq[{namespace}] = {observed_seq}, requested >= {requested_seq} (waited {waited_ms}ms)")]
    RyWaitTimeout {
        namespace: String,
        requested_seq: u64,
        observed_seq: u64,
        waited_ms: u64,
    },

    /// **Issue #9 — cluster-replication determinism contract.**
    ///
    /// `record_with_rid` (and siblings) require the caller's embedding to
    /// match the engine's configured `embedding_dim`. A dim mismatch is a
    /// fatal contract violation — usually a sign that the leader and
    /// follower are running with different embedder model versions, which
    /// would silently corrupt HNSW state if applied. Reject deterministically.
    #[error("embedding dimension mismatch: expected {expected}, got {got}")]
    EmbeddingDimensionMismatch { expected: usize, got: usize },

    /// **Issue #41 brainstorm-4 §3 — monotonic-generation CAS.**
    ///
    /// `try_publish_search_state` rejects publish attempts whose
    /// `new_state.generation` is strictly less than the
    /// currently-active SearchState generation. This is the
    /// brainstorm-4 §3 defense against compactor / reembed /
    /// long-running write paths publishing stale work that would
    /// ABA-rollback the active generation (durable data omission
    /// when a generation regresses from N back to N-1 — the
    /// post-swap materializer reapplies queued ops that were
    /// already covered by generation N).
    ///
    /// Caller policy: re-load `search_state`, rebuild the proposed
    /// state under the new active generation, retry. Reembed loops
    /// abort the entire reembed (clear meta.reembed_state) and
    /// require a fresh `db.reembed(...)` call from the operator —
    /// silent retry inside reembed would mask a deeper concurrency
    /// bug.
    #[error(
        "search state publish stale generation: current_generation={current_generation}, \
         attempted_generation={attempted_generation}. Caller must re-read \
         search_state and rebuild before retrying."
    )]
    SearchStatePublishStaleGeneration {
        current_generation: u64,
        attempted_generation: u64,
    },

    #[error("invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, YantrikDbError>;
