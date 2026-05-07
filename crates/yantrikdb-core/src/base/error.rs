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

    #[error("invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, YantrikDbError>;
