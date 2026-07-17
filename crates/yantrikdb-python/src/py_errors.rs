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
