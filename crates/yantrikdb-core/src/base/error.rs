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


    /// **Decoupled write path RFC, Phase 1.**
    ///
    /// The bounded global ingest queue is full. Foreground writers receive
    /// this synchronously when `log_op_pending()` would push a pending op
    /// past `MAX_PENDING_OPS`. Caller policy: retry with backoff after the
    /// hint, surface to user as 503-like, or shed the write.
    ///
    /// `retry_after_ms` is a coarse hint — actual drain rate depends on
    /// background worker throughput.
    #[error("ingest queue full ({pending} pending ops, max={max}); retry after {retry_after_ms}ms")]
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
    EmbeddingDimensionMismatch {
        expected: usize,
        got: usize,
    },

    #[error("invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, YantrikDbError>;
