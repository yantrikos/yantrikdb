use thiserror::Error;

#[derive(Error, Debug)]
pub enum YantrikDbError {
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// A database error with the open-path stage (and, where the SQL is
    /// derived rather than constant, the statement) that produced it.
    ///
    /// **Why this exists (issue #146).** A truncated-SQL parse error
    /// surfaces from rusqlite as `SqliteFailure(_, "incomplete input")`
    /// with no statement attached: SQLite reports the truncation at the
    /// END of the input, `sqlite3_error_offset()` returns -1 for it, and
    /// rusqlite only builds the SQL-carrying `SqlInputError` when the
    /// offset is >= 0. The blanket `Database` conversion above then
    /// erases which of the open path's many batches was even running —
    /// CI produced exactly `database error: incomplete input` from
    /// inside the constructor, once, on one platform, and the message
    /// named nothing. Every open-path SQL call now tags its stage so a
    /// recurrence localizes itself.
    #[error("database error at {stage}: {source}")]
    DatabaseAt {
        stage: String,
        #[source]
        source: rusqlite::Error,
    },

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

    /// **v0.9.3 numeric/vector contract gate.** An embedding failed
    /// dimension or finiteness validation at an engine entry path. Raised
    /// BEFORE any side effect — a rejected write leaves the engine
    /// unchanged. `index` is the offending element for finiteness
    /// failures, `None` for dimension mismatches.
    #[error("invalid embedding at {path}(): {reason}")]
    InvalidEmbedding {
        path: &'static str,
        index: Option<usize>,
        reason: String,
    },

    /// **v0.9.3 numeric/vector contract gate.** A write-path scoring
    /// scalar (importance / valence / certainty / half_life) was
    /// non-finite. NaN/Inf here silently poisons decay and scoring math,
    /// so it is rejected at the entry path before any side effect.
    #[error("invalid scalar at {path}(): {field} = {value} (must be finite)")]
    InvalidScalar {
        path: &'static str,
        field: &'static str,
        value: f64,
    },

    /// **v0.10 Phase 0 (chain integrity).** A Supersedes link would give the
    /// predecessor (old record) a SECOND active successor. Exactly one
    /// selected inbound Supersedes edge per record is the invariant that
    /// `resolve_current` / `superseded_by` / the continuity packet stand on.
    /// The caller can dispute, or supersede the existing successor instead.
    #[error(
        "supersede conflict: {predecessor_rid} already has an active successor \
         {existing_successor_rid} (edge {existing_edge_id}); dispute it or supersede \
         the successor instead"
    )]
    SupersedeConflict {
        predecessor_rid: String,
        existing_successor_rid: String,
        existing_edge_id: String,
    },

    /// **v0.10 Phase 0.** Inserting this Supersedes edge would create a cycle
    /// in the supersedes graph (the target's successor closure reaches back
    /// to the source).
    #[error("supersede cycle: linking {source_rid} -> {target_rid} would create a cycle")]
    SupersedeCycle {
        source_rid: String,
        target_rid: String,
    },

    /// **v0.10 Phase 0.** A supersedes-chain walk exceeded the traversal cap.
    /// Foreground requests are rejected with this (replication never discards
    /// a durable candidate at the cap — it persists it unselected instead).
    #[error("chain traversal limit ({limit}) exceeded walking from {start_rid}")]
    ChainTraversalLimit { start_rid: String, limit: usize },

    /// **v0.10 Phase 0.** A link endpoint does not exist (or, for a new
    /// active Supersedes edge, is tombstoned) or the endpoints are in
    /// different namespaces.
    #[error("invalid link endpoints: {reason}")]
    InvalidLinkEndpoints { reason: String },

    /// **v0.9.3 (sol converged plan item 3).** `correct(new_text=...)` was
    /// refused because changing a memory's text without re-embedding it
    /// leaves the durable "current truth" and the retrieval vector
    /// permanently disagreeing — the corrected memory keeps being retrieved
    /// under its OLD meaning. Until the vector-coherent correction path
    /// ships (v0.10: embed new text, tombstone old vector, reinsert same
    /// rid), text changes must go through the workaround named in the
    /// message. Metadata / importance / valence corrections are unaffected.
    #[error(
        "correct(new_text=...) on {rid} would leave the memory retrieved under its OLD \
         meaning (text would change but its embedding cannot be updated in place). \
         Until vector-coherent correction ships, use: forget(\"{rid}\") then \
         record_text(new_text, ...) — note this mints a NEW rid. Metadata / importance / \
         valence corrections remain supported without new_text."
    )]
    CorrectionRequiresReembed { rid: String },

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

    /// **Packs.** The pack file is missing, unreadable, or is not a
    /// YantrikDB database.
    #[error("pack at {path} could not be opened: {reason}")]
    PackUnreadable { path: String, reason: String },

    /// **Packs.** The file opened but carries no pack manifest, so it is
    /// a plain database rather than a sealed pack. Seal it first with
    /// `seal_pack()`.
    #[error(
        "no pack manifest in {path} — this is a plain database, not a sealed pack. \
         Produce one with db.seal_pack(dest, manifest, namespace)."
    )]
    PackManifestMissing { path: String },

    /// **Packs.** The manifest is present but malformed.
    #[error("pack manifest in {path} is malformed: {reason}")]
    PackManifestInvalid { path: String, reason: String },

    /// **Packs.** Mounting would place vectors from a different embedding
    /// space alongside the host's. This is the silent-corruption case the
    /// mount-time check exists to prevent: the query is encoded once, by
    /// the host's embedder, and searched against both indexes — so a pack
    /// built by a different model returns confident nonsense.
    #[error(
        "pack '{pack_id}' is not compatible with this database's embedding space: {reason}. \
         The query is encoded once and searched against both indexes, so mounting across \
         embedders returns plausible-looking but meaningless results. \
         Rebuild the pack with the host's embedder, or (only if you are certain the vectors \
         are compatible) mount with MountOptions::allow_unverified_embedder."
    )]
    PackEmbedderMismatch { pack_id: String, reason: String },

    /// **Packs.** A pack with this id is already mounted.
    #[error("pack '{pack_id}' is already mounted (from {path})")]
    PackAlreadyMounted { pack_id: String, path: String },

    /// **Packs.** No mounted pack carries this id.
    #[error("no mounted pack with id '{pack_id}'")]
    PackNotMounted { pack_id: String },

    /// **Packs.** Encrypted packs are not supported: the host would need
    /// the pack's key to build its index, and the key-exchange story is
    /// not designed yet. Rejected rather than half-working.
    #[error("pack at {path} is encrypted; encrypted packs are not supported")]
    PackEncrypted { path: String },

    /// **Packs.** `seal_pack` refuses to overwrite an existing file, so a
    /// mounted pack can never be rewritten underneath its own reader.
    #[error("cannot seal pack to {path}: the file already exists")]
    PackDestinationExists { path: String },

    /// **Packs.** The constitution exceeds its token budget. Enforced at
    /// seal time — where the author can still fix it — because every
    /// constitution rule costs tokens on every turn of every consumer,
    /// and an unbounded constitution degenerates into the
    /// prompt-stuffing the engine exists to replace.
    #[error(
        "pack constitution is ~{approx_tokens} tokens; the budget is {budget}. \
         The constitution is for hard rules the model must always see — \
         move reference facts into the corpus, where retrieval serves them on demand."
    )]
    PackConstitutionTooLarge { approx_tokens: usize, budget: usize },

    /// **Packs.** The pack claims a signature that does not verify.
    ///
    /// This is refused outright — strictly worse than carrying no
    /// signature at all, because a claimed-and-failed signature means
    /// the pack was modified after signing, the signature was forged,
    /// or the key material is corrupt. There is no legitimate state
    /// that produces it, so there is no override.
    #[error(
        "pack '{pack_id}' carries a signature that does not verify: {reason}. \
         The pack was modified after signing or the signature is forged; \
         re-download it from the publisher."
    )]
    PackSignatureInvalid { pack_id: String, reason: String },

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

    /// **v0.10 Item 3.** A text-changing `correct()` needs to re-embed and
    /// tombstone+append the vector, but a `db.reembed()` cutover is in
    /// flight (the write-router is in Queueing state), so the vector-index
    /// mutation cannot be applied against a stable generation. Retryable:
    /// the correction touched NO state and can be reissued once the reembed
    /// completes. Metadata / importance / valence corrections (no `new_text`)
    /// are unaffected and proceed during reembed.
    #[error(
        "correct(new_text=...) on {rid} deferred: a db.reembed() cutover is in progress, \
         so the vector index cannot be updated against a stable generation. No durable \
         state was changed — BUT the OLD text remains ACTIVE and is still served as the \
         record's CURRENT value until a retry succeeds; this is a not-yet-applied \
         correction, not a harmless no-op. Retry after the reembed completes. To stop \
         serving the old text in the meantime, issue a metadata/status correction (NOT \
         blocked by reembed) marking the record correction_pending. (Metadata / \
         importance / valence corrections are never blocked by reembed.)"
    )]
    CorrectionDeferredDuringReembed { rid: String },

    /// **v0.10 Item 4a.** `record_batch` could not acquire a `SyncWriteGuard`
    /// because a `db.reembed()` cutover is in flight (the write-router is in
    /// Queueing state), so its `SearchState` snapshot could be swapped out from
    /// under the batch mid-write. Retryable: the batch touched NO durable state
    /// and can be reissued verbatim once the reembed completes.
    ///
    /// `record()` handles this by falling back to its queued path; there is no
    /// queued-batch primitive yet (v0.10 Item 4a.6c), and routing items
    /// independently would break the batch's all-or-nothing contract. Failing
    /// the whole batch with a typed retryable error is the honest option in the
    /// meantime — the alternative is what this replaced: silently committing
    /// rows stamped with a generation that is being discarded.
    #[error(
        "record_batch({count} inputs) deferred: a db.reembed() cutover is in progress, \
         so the batch cannot be committed against a stable generation. No durable state \
         was changed — retry the batch verbatim once the reembed completes. (Single \
         record() writes are not blocked; they route through the queued path.)"
    )]
    BatchDeferredDuringReembed { count: usize },

    /// A consolidation/synthesis write cannot use `record()`'s queued fallback:
    /// the caller must attach provenance and source bookkeeping to a materialized
    /// memory row before returning success. Queueing only the base record would
    /// expose a rid whose dependent side effects refer to a row that does not yet
    /// exist. Retryable; the sync-only route checks this before writing a claim,
    /// memory row, or oplog entry.
    #[error(
        "consolidation deferred: a db.reembed() cutover is in progress, so the base \
         record cannot be materialized before attaching its provenance. No durable state was \
         changed — retry once the reembed completes"
    )]
    ConsolidationDeferredDuringReembed,

    /// A local synthesis admission would make one evidence record back more
    /// verified synthesis generations than the configured invalidation bound.
    /// The check runs in the synthesis record transaction, after idempotency
    /// resolution, so a refusal leaves no memory, dependency edge, claim, or
    /// oplog row. Replication never drops an already-durable remote operation;
    /// an over-cap follower state is instead exposed through `stats()`.
    #[error(
        "synthesis admission refused: evidence {source_rid} already backs {current} live \
         synthesis generations (cap={limit}); invalidate or supersede an existing \
         generation, or raise it with set_synthesis_fanout_cap()"
    )]
    SynthesisFanoutLimit {
        source_rid: String,
        current: usize,
        limit: usize,
    },

    /// **2026-08-17.** `record_with_rid` — the deterministic replay primitive
    /// used by the cluster applier — could not acquire a `SyncWriteGuard`
    /// because a `db.reembed()` cutover is in flight.
    ///
    /// It previously took its `SearchState` snapshot with NO guard at all
    /// (`self.search_state.load_full()`), which is the exact hazard the
    /// sibling paths were built to avoid: reembed could complete its swap
    /// between the snapshot and the append, landing the vector in a
    /// DISCARDED delta index. The row is then durably in SQL, marked active,
    /// and absent from the live index — stored, alive, unfindable. That is
    /// the HNSW-orphan failure shape, reached through a different door.
    ///
    /// It cannot use `record()`'s queued fallback: this path is deliberately
    /// byte-deterministic (caller-supplied rid, embedding, timestamp and
    /// model, engine embedder never invoked), and the queued materializer
    /// re-encodes under the NEW embedder, which would change the bytes the
    /// caller pinned. Deferring retryably — the `record_batch` precedent —
    /// is the honest option: nothing durable has happened yet.
    #[error(
        "record_with_rid({rid}) deferred: a db.reembed() cutover is in progress, so the \
         caller-supplied embedding cannot be committed against a stable generation. No \
         durable state was changed — retry verbatim once the reembed completes. (This \
         path cannot take record()'s queued fallback without re-encoding the vector the \
         caller pinned, which would break its determinism contract.)"
    )]
    DeterministicWriteDeferredDuringReembed { rid: String },

    /// **2026-08-17 — the F1/F2 forget-resurrection race.** `forget()` /
    /// `tombstone_with_rid()` could not acquire a `SyncWriteGuard` because a
    /// `db.reembed()` cutover is in flight.
    ///
    /// Both funnel through `tombstone_inner`, which tombstones the rid and
    /// its chunks in `search_state.load().vec_index`. Unguarded, a cutover
    /// could publish an index built from a SQL snapshot predating the
    /// tombstone, and the delta tombstone died with the discarded state —
    /// leaving the record tombstoned in SQL and ALIVE in the live index. A
    /// delete that silently un-deletes.
    ///
    /// Retryable: SQL is untouched when this is returned.
    #[error(
        "forget({rid}) deferred: a db.reembed() cutover is in progress, so the tombstone \
         could be applied to an index that is about to be discarded — which would resurrect \
         the record. No durable state was changed; retry once the reembed completes."
    )]
    ForgetDeferredDuringReembed { rid: String },

    /// **2026-08-17.** `rebuild_vec_index` finished building a replacement
    /// cold tier, but a `db.reembed()` cutover started (or completed) while
    /// it was working.
    ///
    /// The rebuild reads `memories.embedding` — vectors in the space that
    /// was active when it started — so installing it after a cutover would
    /// place an index built entirely from OLD-space vectors into the NEW
    /// generation's state. Every cold-tier distance would then be computed
    /// against a query encoded by a different model: not a lost write, but
    /// a whole tier of quietly meaningless scores that no test notices,
    /// because the index is populated and every lookup returns something.
    ///
    /// Retryable: nothing was installed. Re-run the rebuild against the new
    /// generation.
    #[error(
        "rebuild_vec_index deferred: {reason}. Nothing was installed — the rebuilt index was \
         discarded rather than mixed into a different embedding space. Retry the rebuild."
    )]
    IndexRebuildDeferredDuringReembed { reason: String },

    /// **v0.10 Item 3 (correction seqlock, sol r5).** A recall could not
    /// obtain a coherent snapshot within its retry budget because
    /// text-changing corrections kept interleaving with its candidate
    /// generation → hydration span. Retryable: the read touched no state
    /// and never returns a result that pairs a stale ranking vector with
    /// corrected text (coherence is never traded for a result). Practically
    /// unreachable outside a sustained correction storm on the same records.
    #[error(
        "recall could not obtain a coherent snapshot after {attempts} attempts \
         (concurrent corrections kept interleaving); retry"
    )]
    RecallContended { attempts: u32 },

    /// **v0.10 Item 4a — single-origin ingress guard.** A protected write op
    /// (`record` / `record_with_rid` / `correct` / `consolidate`) arrived via
    /// replication from an origin actor that is not this database's configured
    /// `authoritative_origin_actor`. In the single-writer Item 4a model,
    /// provenance is admitted at ONE authority; applying a foreign origin's
    /// record verbatim would let an ungated or malicious peer launder
    /// provenance past the local gate. The ENTIRE incoming batch is rejected
    /// before any HLC merge / oplog / memories / cache / vector change, so a
    /// rejected apply leaves the engine byte-for-byte unchanged. Multi-origin
    /// ingress is an Item 4b (multi-master) capability. The guard is inactive
    /// unless `authoritative_origin_actor` is configured.
    #[error(
        "replicated `{op_type}` op rejected: origin '{origin_actor}' is not the \
         authoritative origin '{authority}' (single-writer Item 4a; multi-origin \
         is v0.11 Item 4b)"
    )]
    ForeignOriginRejected {
        op_type: String,
        origin_actor: String,
        authority: String,
    },

    /// **v0.10 Item 4a — anti-laundering provenance gate.** A write declared an
    /// internally inconsistent provenance — e.g. `source=inference` claiming
    /// `kind=fact` without confirmation/verification basis or an explicit
    /// override, `source=inference` with `confidence_basis=observation`, or an
    /// unparseable protected `source`/`confidence_basis`. Rejected at write time
    /// BEFORE any side effect (mirrors the v0.9.3 validation contract). The gate
    /// prevents DECLARED contradictions only; it does not (and cannot) verify
    /// truthful provenance.
    #[error("provenance rejected at {path}: {reason}")]
    ProvenanceInconsistent { path: &'static str, reason: String },

    /// **v0.10 Item 4a.6c — durable idempotency (T07).** The caller reused an
    /// idempotency key whose committed claim does not match this write —
    /// normally a DIFFERENT payload digest under the same key ("repetition is
    /// not corroboration": the first write's content stands; a same-key rewrite
    /// must be loud, never a silent near-dup merge). `existing_rid` is the
    /// record the key already binds to, so the caller can inspect what it
    /// conflicts with. Also raised, with a distinguishing `reason`, for
    /// anomalous claim states (e.g. a crashed pre-commit claim from a future
    /// engine version) where returning a rid would assert a write that may not
    /// exist.
    #[error(
        "idempotency conflict in namespace '{namespace}' (existing rid {existing_rid}): {reason}"
    )]
    IdempotencyConflict {
        namespace: String,
        existing_rid: String,
        reason: String,
    },

    /// **v0.10 Item 4a.6c.** The caller passed an idempotency key the engine
    /// refuses to accept (empty / whitespace-only / over-long). Rejected
    /// loudly instead of coerced: silently treating `""` as "no key" would
    /// leave a caller believing they have dedup protection they don't have —
    /// the same silent-failure class the rest of Item 4a exists to kill.
    #[error("invalid idempotency key: {reason}")]
    InvalidIdempotencyKey { reason: String },

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
