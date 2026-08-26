//! Typed Python exceptions for the engine's actionable error variants
//! (v0.10, yantrikdb-mcp validation friction 2).
//!
//! Before this module, every engine error crossed the pyo3 boundary as a bare
//! `RuntimeError`, so a host implementing the documented retry semantics had
//! to string-match on message text — the brittle thing that silently breaks
//! on a rewording. These classes let hosts branch on TYPE:
//!
//! - retryable-later: `Backpressure`, `CorrectionDeferredDuringReembed`,
//!   `BatchDeferredDuringReembed`
//! - caller-decision: `IdempotencyConflict` (the message carries the existing
//!   rid — the first write's content stands), `InvalidIdempotencyKey`,
//!   `ProvenanceInconsistent`
//!
//! Every class SUBCLASSES `RuntimeError`, so pre-v0.10 handlers written as
//! `except RuntimeError:` keep working unchanged — this is additive, not a
//! break. Unmatched variants still map to bare `RuntimeError` in `map_err`.

use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;

create_exception!(
    yantrikdb,
    Backpressure,
    PyRuntimeError,
    "The engine is saturated (pending-op queue or delta capacity); nothing \
     was written. Retryable: back off and reissue the identical call. Keyed \
     duplicate retries resolve even under saturation."
);

create_exception!(
    yantrikdb,
    CorrectionDeferredDuringReembed,
    PyRuntimeError,
    "A re-embedding cutover is in flight; the correction was not applied. \
     Retryable: reissue verbatim after the cutover completes."
);

create_exception!(
    yantrikdb,
    BatchDeferredDuringReembed,
    PyRuntimeError,
    "A re-embedding cutover is in flight; the WHOLE batch was deferred and \
     nothing was written. Retryable: reissue the identical batch. Fully \
     keyed duplicate batches resolve without deferring."
);

create_exception!(
    yantrikdb,
    IdempotencyConflict,
    PyRuntimeError,
    "An idempotency claim could not resolve to a hit. USUALLY: the same key \
     was already committed with a DIFFERENT payload — the first write's \
     content stands, the message carries the existing rid, and the fix is to \
     change the key or the payload (not retryable as-is). The variant also \
     covers anomalous claim states (a crashed prior attempt, an \
     engine-invariant violation), some of which the message marks as \
     retryable — read it before deciding, but branch on this TYPE to know \
     you are in claim-resolution territory at all."
);

create_exception!(
    yantrikdb,
    InvalidIdempotencyKey,
    PyRuntimeError,
    "The idempotency key is empty, whitespace-only, or longer than 512 \
     bytes. Not retryable as-is: fix the key."
);

create_exception!(
    yantrikdb,
    RecallContended,
    PyRuntimeError,
    "A recall lost a bounded read-contention race (writer-priority lock \
     tuning). Nothing is wrong with the query. Retryable: back off briefly \
     and reissue the identical recall."
);

create_exception!(
    yantrikdb,
    ProvenanceInconsistent,
    PyRuntimeError,
    "The write's declared provenance is internally inconsistent (e.g. \
     source=inference claiming kind=fact without confirmation/verification) \
     and the anti-laundering gate refused it. Not retryable as-is: fix the \
     declaration or raise the confidence basis."
);

create_exception!(
    yantrikdb,
    PackEmbedderMismatch,
    PyRuntimeError,
    "The pack's vectors are not provably in this database's embedding space, \
     so mounting it would return confident nonsense rather than an error — \
     the query is encoded once, by the host's embedder, and searched against \
     both indexes. Not retryable: rebuild the pack with the host's embedder. \
     If the spaces are merely UNPROVEN (a legacy database with no recorded \
     embedder identity) rather than known to differ, mount with \
     allow_unverified_embedder=True to accept that risk explicitly."
);

create_exception!(
    yantrikdb,
    PackAlreadyMounted,
    PyRuntimeError,
    "A pack with this origin@version is already mounted. Not retryable as-is: \
     unmount it first, or mount a different version."
);

create_exception!(
    yantrikdb,
    PackSignatureInvalid,
    PyRuntimeError,
    "The pack claims a signature that does not verify: it was modified after \
     signing or the signature is forged. There is no legitimate state that \
     produces this and no override — re-download the pack from its publisher."
);

create_exception!(
    yantrikdb,
    PackNotMounted,
    PyRuntimeError,
    "An allowlist named a pack id (origin@version) that is not mounted. \
     Nothing was searched and nothing was returned: a lease that names a \
     pack the host no longer holds is a misconfiguration the caller has to \
     see, not a turn that silently runs without it. Not retryable as-is: \
     mount the pack (or drop the id) and reissue."
);

create_exception!(
    yantrikdb,
    PhraseRouteUnavailableError,
    PyRuntimeError,
    "recall_thread's phrase route is unavailable on this engine: the FTS \
     index on an encrypted store holds ciphertext, so a phrase MATCH would \
     silently match nothing (the engine fails closed instead). Not \
     retryable as-is: drop the phrases (entity and topic routes work under \
     encryption) or query an unencrypted store."
);

create_exception!(
    yantrikdb,
    SourceTurnMaintenanceRequiredError,
    PyRuntimeError,
    "This store's source_turn columns are not known to mirror their \
     metadata — an encrypted store that has not finished its backfill, or \
     ANY store (plaintext included) staled by a raw SQL metadata write — so \
     a strict chronological thread order cannot be guaranteed yet. \
     Retryable after maintenance: call maintain_source_turn_backfill(batch) \
     repeatedly until it reports complete, then reissue the identical query."
);

create_exception!(
    yantrikdb,
    InvalidThreadTopicError,
    PyRuntimeError,
    "A recall_thread topic rid did not resolve to a visible, verified topic \
     synthesis row in the queried namespace. The message is identical \
     regardless of cause (ordinary memory rid, nonexistent, cross-namespace, \
     inactive, unverified) — deliberately leakage-free. Not retryable as-is: \
     pass a topic synthesis rid from this namespace (e.g. from \
     persist_organization / record_synthesis)."
);
