//! **v0.10 Item 4a.6c — the durable idempotency claim** (design doc §E,
//! "claim-after-routing saga").
//!
//! One owner for the claim's INSERT-or-resolve, used by BOTH write routes
//! (sync: `record_under_guard_and_state`'s transaction; queued: the pending-op
//! transaction in `log_op_pending_for_reembed_queue`). The routes differ in
//! what their authoritative op is; the claim rule is identical — which is
//! exactly why it must not be spelled twice (#83).
//!
//! ## The rule
//!
//! `INSERT ... ON CONFLICT(origin_actor, namespace, idempotency_key) DO
//! NOTHING` **is the serialization point** — never a prior SELECT (that would
//! be the TOCTOU shape). If the INSERT wins, this write owns the key and
//! proceeds; the claim commits atomically with the authoritative op. If it
//! loses, the surviving claim is read back and resolved by payload digest:
//!
//! - same digest  → **idempotent hit**: return the ORIGINAL rid; the caller
//!   aborts its transaction (nothing it wrote survives) and skips every
//!   post-commit effect — no second row, no second op, no stats advance, no
//!   flag tick, no session bump. Repetition is not corroboration (T07).
//! - different digest → typed `IdempotencyConflict` — same key reused for a
//!   semantically different write is a caller bug and must be loud, never a
//!   silent near-dup merge.
//!
//! ## Why the claim is the FIRST statement in the caller's transaction
//!
//! v37 ships a defense-in-depth partial unique index on
//! `memories(origin_actor, namespace, idempotency_key)`. If the row INSERT ran
//! before the claim check, a dup retry would hit that index and surface as a
//! bare constraint error instead of resolving to a hit/conflict. Claim-first
//! also makes the dup path cheap: the losing transaction has written nothing
//! when it aborts.
//!
//! ## Why `state` is written as `'committed'` directly (and `'pending'` is
//! unused here)
//!
//! The design doc's pending→committed lifecycle exists so a claim that is
//! durable BEFORE its write can be reconciled after a crash. In 4a.6c both
//! routes insert the claim INSIDE the same transaction as their authoritative
//! op, so there is no instant where a claim is durable and its op is not — a
//! crash rolls back both, and `'pending'` would be unobservable theater.
//! `'pending'` (and the startup sweep that reconciles it) becomes real the day
//! a claim must be visible across a transaction boundary (4b multi-writer /
//! cross-process claims); the column is schema-reserved for that, not dead.

use rusqlite::{params, OptionalExtension};

use crate::error::{Result, YantrikDbError};

use super::now;

/// Everything the claim row needs. Built by the route that owns the
/// transaction; digested BEFORE routing from the RAW caller payload.
pub(crate) struct ClaimRow<'a> {
    pub origin_actor: &'a str,
    pub namespace: &'a str,
    pub idempotency_key: &'a str,
    /// The rid THIS attempt minted (stored only if the claim wins).
    pub rid: &'a str,
    /// blake3 over the canonical RAW payload (`payload_digest::payload_digest`).
    pub payload_digest: &'a [u8; 32],
    /// The authoritative op this claim binds to — recovery evidence, minted by
    /// the caller BEFORE the transaction so claim and op agree.
    pub op_id: &'a str,
    /// 'sync' | 'queued'.
    pub route: &'static str,
    /// Search generation at claim time.
    pub generation: i64,
}

/// The queued route's claim inputs. The pending-op helper owns the transaction
/// AND mints the op id, so the route hands over everything else and the helper
/// assembles the full [`ClaimRow`] (op_id = the pending op it is about to
/// write, route = "queued") — keeping op and claim agreeing by construction.
pub(crate) struct PendingClaim<'a> {
    pub namespace: &'a str,
    pub idempotency_key: &'a str,
    pub payload_digest: &'a [u8; 32],
    /// The rid the queued write minted (returned to the caller; the
    /// materializer creates the row under it later).
    pub rid: &'a str,
    /// Search generation at claim time.
    pub generation: i64,
}

/// Outcome of [`claim_in_tx`]. A conflict is an `Err`, not a variant — it must
/// propagate.
#[must_use]
pub(crate) enum ClaimAttempt {
    /// This write owns the key. Proceed; the claim commits with the op.
    Won,
    /// The key is already committed with the SAME payload — return this rid to
    /// the caller and abort the surrounding transaction untouched.
    Hit { existing_rid: String },
}

/// INSERT-or-resolve the claim inside the caller's open transaction. See the
/// module doc for the rule. MUST be the first write of the transaction.
pub(crate) fn claim_in_tx(
    conn: &rusqlite::Connection,
    claim: &ClaimRow<'_>,
) -> Result<ClaimAttempt> {
    let inserted = conn.execute(
        "INSERT INTO idempotency_claims \
         (origin_actor, namespace, idempotency_key, rid, payload_digest, \
          op_id, route, generation, state, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'committed', ?9) \
         ON CONFLICT(origin_actor, namespace, idempotency_key) DO NOTHING",
        params![
            claim.origin_actor,
            claim.namespace,
            claim.idempotency_key,
            claim.rid,
            &claim.payload_digest[..],
            claim.op_id,
            claim.route,
            claim.generation,
            now(),
        ],
    )?;
    if inserted == 1 {
        return Ok(ClaimAttempt::Won);
    }

    // Lost: read the surviving claim — under the same conn lock, same tx view,
    // so this cannot race the insert that beat us (#83: re-read the winner,
    // never assume).
    let existing: Option<(String, Vec<u8>, String)> = conn
        .query_row(
            "SELECT rid, payload_digest, state FROM idempotency_claims \
             WHERE origin_actor = ?1 AND namespace = ?2 AND idempotency_key = ?3",
            params![claim.origin_actor, claim.namespace, claim.idempotency_key],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()?;
    let Some((existing_rid, existing_digest, state)) = existing else {
        // DO NOTHING fired yet no row is visible: unreachable under the conn
        // lock (nobody can delete between our INSERT and this SELECT). Loud,
        // not silent, if the invariant ever breaks — and retryable, since the
        // next attempt re-runs the INSERT.
        return Err(YantrikDbError::IdempotencyConflict {
            namespace: claim.namespace.to_string(),
            existing_rid: String::new(),
            reason: "claim row vanished between ON CONFLICT and read-back \
                     (engine invariant violation — retry; if it persists, the \
                     claims table is being mutated outside the engine)"
                .to_string(),
        });
    };

    if existing_digest.as_slice() != &claim.payload_digest[..] {
        return Err(YantrikDbError::IdempotencyConflict {
            namespace: claim.namespace.to_string(),
            existing_rid,
            reason: "same idempotency key with a DIFFERENT payload — the first \
                     write's content stands; change the key or the payload"
                .to_string(),
        });
    }
    if state != "committed" {
        // 4a.6c never durably writes 'pending' (claim + op share one tx). A
        // pending row here means a crash artifact from a future engine version
        // whose claim/commit are split; refuse loudly rather than return a rid
        // whose write may not exist.
        return Err(YantrikDbError::IdempotencyConflict {
            namespace: claim.namespace.to_string(),
            existing_rid,
            reason: format!(
                "claim is in state '{state}', not 'committed' — a crashed \
                 prior attempt; the recovery sweep that reconciles this is \
                 not yet implemented"
            ),
        });
    }
    Ok(ClaimAttempt::Hit { existing_rid })
}
