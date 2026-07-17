use rusqlite::params;
use std::sync::Arc;

use crate::error::Result;
use crate::serde_helpers::serialize_f32;
use crate::types::*;

use super::reembed::SearchState;
use super::reservation::ReservationGuard;
use super::write_router::SyncWriteGuard;
use super::{embedding_hash, now, sanitize, YantrikDB};

/// Coerce a blank namespace to the canonical default (v0.7.23).
///
/// The schema column default and the Python/MCP bindings all use
/// `"default"`; an empty or whitespace-only namespace is virtually always
/// a caller-side defaulting accident (e.g. a server gateway doing
/// `unwrap_or("")`). Normalizing it at the engine boundary keeps a single
/// canonical value so writes, point reads, list filters, and recall all
/// agree across every consumer instead of silently persisting an unscoped
/// `""` partition that no reader queries for.
pub(crate) fn normalize_namespace(ns: &str) -> &str {
    if ns.trim().is_empty() {
        "default"
    } else {
        ns
    }
}

impl YantrikDB {
    /// Store a new memory and return its RID.
    ///
    /// **Issue #41 layer 3 — WriteRouter gating.** At entry, the writer
    /// attempts to acquire a `SyncWriteGuard`. If the engine's
    /// `write_router` is in `Normal` state (no reembed in progress),
    /// the guard is acquired and the synchronous path runs: INSERT
    /// memories + vec_index.append + log_op (applied=1). The guard is
    /// held for the full critical section and drops via RAII when
    /// `record` returns, decrementing the inflight-writer counter.
    /// This is the brainstorm-2 invariant that prevents in-flight
    /// writers from committing `applied=1` against an about-to-be-
    /// discarded old generation during reembed cutover.
    ///
    /// If the router is in `Queueing` state (reembed has flipped the
    /// gate and is waiting for writers to drain before capturing
    /// `build_hwm`), `try_enter_sync_writer()` returns None and this
    /// call routes through the queued path: the op is appended to
    /// `oplog` with `applied=0`, `embedding_model = old_embedder_name`,
    /// the full record payload (text + metadata) — the post-swap
    /// materializer re-encodes under the new embedder + applies to
    /// the new generation. The caller's return value (rid + seq) is
    /// the same shape; read-after-write requires `recall_with_seq` to
    /// wait for the new generation's `visible_seq` to advance.
    #[tracing::instrument(skip(self, metadata, embedding), fields(memory_type, namespace))]
    pub fn record(
        &self,
        text: &str,
        memory_type: &str,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: &serde_json::Value,
        embedding: &[f32],
        namespace: &str,
        certainty: f64,
        domain: &str,
        source: &str,
        emotional_state: Option<&str>,
    ) -> Result<String> {
        self.record_with_idempotency(
            text,
            memory_type,
            importance,
            valence,
            half_life,
            metadata,
            embedding,
            namespace,
            certainty,
            domain,
            source,
            emotional_state,
            None,
        )
    }

    /// `record()` plus a durable idempotency key (v0.10 Item 4a.6c, T07
    /// "repetition is not corroboration").
    ///
    /// With `idempotency_key = Some(k)`, the write is deduplicated on
    /// `(origin_actor, namespace, k)` against the canonical RAW payload digest
    /// (`base/payload_digest`):
    ///
    /// - **same key + same payload** -> the original rid is returned and NOTHING
    ///   is written or moved — no second row, no oplog op, no calibration
    ///   advance, no session bump, no warn-flag tick, certainty untouched. A
    ///   retry is a retry, not corroboration.
    /// - **same key + different payload** -> typed
    ///   [`crate::error::YantrikDbError::IdempotencyConflict`] carrying the
    ///   existing rid. The first write's content stands; a silent near-dup
    ///   merge is exactly what T07 forbids.
    ///
    /// The claim commits atomically with the route's authoritative op (the
    /// memories row + record op on the sync route; the pending oplog op on the
    /// queued route), so a crash leaves either both or neither — recovery never
    /// has to guess from row existence. The digest uses the RAW caller
    /// importance (pre-calibration) deliberately: calibration output depends on
    /// the namespace's running EWMA, which the first attempt itself advances,
    /// so a digest over the calibrated value would make an honest retry into a
    /// false conflict.
    ///
    /// `None` is byte-for-byte `record()`.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip(self, metadata, embedding), fields(memory_type, namespace))]
    pub fn record_with_idempotency(
        &self,
        text: &str,
        memory_type: &str,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: &serde_json::Value,
        embedding: &[f32],
        namespace: &str,
        certainty: f64,
        domain: &str,
        source: &str,
        emotional_state: Option<&str>,
        idempotency_key: Option<&str>,
    ) -> Result<String> {
        // v0.9.3 contract gate: validate before anything else. (Historically
        // "before any side effect" because importance calibration used to
        // autocommit here; as of 4a.6b nothing below this point mutates state
        // until the winner's transaction commits.)
        crate::validate::validate_embedding("record", embedding, self.embedding_dim)?;
        crate::validate::validate_scalars(
            "record",
            &[
                ("importance", importance),
                ("valence", valence),
                ("certainty", certainty),
                ("half_life", half_life),
            ],
        )?;
        // v0.10 Item 4a.4 anti-laundering gate: refuse (enforce) / flag (warn)
        // a record whose declared provenance is internally inconsistent (e.g.
        // source=inference claiming metadata.kind=fact), BEFORE any side effect.
        // For a fresh insert `metadata` IS the final merged metadata. A warn-mode
        // Flagged verdict is counted only after the write COMMITS (4a.6b) — the
        // routed paths below carry it to their post-commit tick.
        let gate_verdict = self.gate_provenance(source, metadata)?;
        // Task 29 (Ingest Integrity): strip any leaked tool-call
        // serialization tail from the stored text. On this entry point the
        // caller supplies the embedding, so the vector may still reflect the
        // pre-clean text; that minor staleness is strictly better than
        // persisting the artifact, and the dominant ingest paths
        // (`record_text`, MCP/HTTP) embed engine-side on the cleaned text.
        let sanitized = sanitize::sanitize_tool_call_artifacts(text);
        let text = sanitized.as_ref();
        // v0.7.23: coerce a blank namespace to the canonical default so no
        // consumer persists an unscoped "" partition. Shadows the param so
        // both the sync and queued paths below see the normalized value.
        let namespace = normalize_namespace(namespace);
        // Task 31 (Ingest Integrity): compute the calibrated importance against
        // this namespace's running distribution — READ-ONLY (4a.6b). The
        // distribution itself advances inside the winning path's transaction
        // (`advance_importance_stats_in_tx`), fed the RAW value, so a rejected
        // write leaves the namespace's calibration untouched. The sync, queued,
        // and oplog paths all store/replicate the calibrated value below.
        let raw_importance = importance;
        let importance = self.calibrated_importance(namespace, importance)?;
        // 4a.6c: the idempotency digest — canonical RAW payload (post-sanitize
        // text, post-normalize namespace, PRE-calibration importance). Raw on
        // purpose: calibration output depends on the namespace EWMA, which the
        // first attempt itself advances, so digesting the calibrated value
        // would turn an honest retry into a false conflict. Computed BEFORE
        // routing so both routes resolve the same key identically.
        //
        // The caller-supplied embedding IS in the digest (PayloadVariant::
        // Record), even though the QUEUED route discards it (the materializer
        // re-encodes). Deliberate, decided at sol's 4a.6c review: record()'s
        // idempotency is API-BYTE identity — on the sync route the embedding is
        // stored, so two calls with different vectors ARE different writes, and
        // the digest must not depend on which route the router happened to pick.
        // A caller whose embedder is non-deterministic across retries belongs on
        // record_text (whose RecordText variant EXCLUDES the generated vector,
        // 4a.6d) — regeneration is legitimate there and only there.
        let idem: Option<(&str, [u8; 32])> = match idempotency_key {
            None => None,
            Some(key) => {
                if key.trim().is_empty() || key.len() > 512 {
                    return Err(crate::error::YantrikDbError::InvalidIdempotencyKey {
                        reason: if key.len() > 512 {
                            format!("key is {} bytes; max 512", key.len())
                        } else {
                            "key is empty or whitespace-only".to_string()
                        },
                    });
                }
                let view = crate::payload_digest::PayloadView {
                    variant: crate::payload_digest::PayloadVariant::Record,
                    namespace,
                    text,
                    memory_type,
                    importance: raw_importance,
                    valence,
                    half_life,
                    certainty,
                    domain,
                    source,
                    emotional_state,
                    metadata,
                    embedding: Some(embedding),
                };
                Some((key, crate::payload_digest::payload_digest(&view)))
            }
        };
        // 4a.6c pre-admission probe (sol finding 1): a duplicate retry writes
        // nothing, so it resolves BEFORE any admission machinery — before the
        // router, the backpressure checks, the delta reservation, and the
        // seq/HLC allocation. Backpressure storms are exactly when clients
        // retry; without this, a keyed dup against a saturated engine could
        // only ever see Backpressure and the retry loop would never converge.
        // "Admission" is the precise word (sol 4a.6d-2b r1 finding 2): the
        // validation gates above still run first, because they are
        // deterministic payload-shape checks an identical retry passes
        // identically — not saturation-dependent rejection.
        // A probe MISS is advisory (the ON CONFLICT INSERT in the write tx
        // stays authoritative); a probe HIT is final — committed claims are
        // immutable in 4a.
        if let Some((key, digest)) = idem.as_ref() {
            if let Some(existing_rid) = super::idempotency::probe_committed_claim(
                &self.conn(),
                &self.actor_id,
                namespace,
                key,
                digest,
            )? {
                return Ok(existing_rid);
            }
        }
        // Issue #41 layer 3: route on write_router state. The guard
        // (if acquired) is held for the full sync path and drops via
        // RAII at function return, panic-safe.
        let sync_guard = self.write_router.try_enter_sync_writer();
        if sync_guard.is_none() {
            // Queueing state — take the queued path. Reembed cutover
            // is in flight; writes go to oplog and the post-swap
            // materializer applies them under the new embedder.
            return self.record_queued(
                text,
                memory_type,
                importance,
                raw_importance,
                valence,
                half_life,
                metadata,
                embedding,
                namespace,
                certainty,
                domain,
                source,
                emotional_state,
                gate_verdict,
                idem,
            );
        }
        // guard is held; RAII Drop at function exit decrements inflight.
        let guard = sync_guard.unwrap();

        // **Issue #41 brainstorm-4 §1.** Load SearchState AFTER the
        // guard is acquired. With the guard held, reembed cannot
        // complete its swap, so the loaded state is the published
        // active generation for the entire critical section. Note:
        // for `record()` (caller-supplied embedding), the engine
        // cannot verify the embedding's generation provenance — the
        // caller is responsible for using the embedder consistent
        // with the active generation. `record_text()` (engine-
        // supplied embedding) has a revalidation loop that ensures
        // the embedding and the active generation match.
        let state = self.search_state.load_full();

        self.record_under_guard_and_state(
            state,
            guard,
            text,
            memory_type,
            importance,
            raw_importance,
            valence,
            half_life,
            metadata,
            embedding,
            namespace,
            certainty,
            domain,
            source,
            emotional_state,
            gate_verdict,
            idem,
        )
    }

    /// **Issue #41 brainstorm-4 §2.** The post-guard, post-load
    /// critical section shared by `record()` and `record_text()`.
    ///
    /// Caller MUST hold the `SyncWriteGuard` — this is the contract
    /// that prevents reembed from completing its SearchState swap
    /// while we are mid-commit, and the contract that makes
    /// `state.generation` the durable answer to "what generation am I
    /// committing under." The guard is moved in by value and drops
    /// via RAII at function exit, decrementing the in-flight counter.
    ///
    /// Caller MUST also pre-load `state` from `self.search_state` and
    /// pass it in — this commit path uses the snapshot rather than
    /// re-loading, so writer revalidation logic in `record_text()`
    /// (which re-loads after embed to detect a generation advance)
    /// is the single source of truth for generation safety on the
    /// text-embed path.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_under_guard_and_state(
        &self,
        state: Arc<SearchState>,
        _guard: SyncWriteGuard<'_>,
        text: &str,
        memory_type: &str,
        importance: f64,
        raw_importance: f64,
        valence: f64,
        half_life: f64,
        metadata: &serde_json::Value,
        embedding: &[f32],
        namespace: &str,
        certainty: f64,
        domain: &str,
        source: &str,
        emotional_state: Option<&str>,
        gate_verdict: crate::provenance::GateVerdict,
        idem: Option<(&str, [u8; 32])>,
    ) -> Result<String> {
        let rid = crate::id::new_id();
        let ts = now();
        let emb_blob = serialize_f32(embedding);
        let meta_str = serde_json::to_string(metadata)?;
        // 4a.6c: the record op's id is minted BEFORE the transaction when a
        // claim rides it — the claim binds to this op as recovery evidence and
        // must be the tx's FIRST statement (the v37 partial unique index on
        // memories would otherwise fire before a dup resolves to a hit).
        let record_op_id = crate::id::new_id();

        // Encrypt fields if encryption is enabled
        let stored_text = self.encrypt_text(text)?;
        let stored_meta = self.encrypt_text(&meta_str)?;
        let stored_emb = self.encrypt_embedding(&emb_blob)?;

        // Read active session for this namespace into a local before acquiring conn
        let session_id = self.active_sessions.read().get(namespace).cloned();

        // **Issue #41 brainstorm-4 §6.** Stamp the v28
        // embedding_generation column with the snapshot's generation
        // so the post-swap materializer can discriminate "this row
        // was indexed under the active generation — skip" from "this
        // row was inserted under an old generation — needs re-encode."
        // Read from `state.generation` (not a fresh load) because we
        // hold the SyncWriteGuard for the entire sync path:
        // search_state cannot advance under us until the guard drops.
        let embedding_generation: i64 = state.generation as i64;

        // **v0.10 Item 4a.6a — durable sync acceptance.**
        //
        // This path used to be four independent autocommit windows: the memories
        // row, the session updates, `log_op("record")`, and the
        // `log_op_pending(materialize_record_post)` enqueue, with a plain
        // `vec_index.append` in the middle. A crash or an error between them left
        // a committed row with NO oplog provenance — the leak the old comment
        // here recorded as "23k rows over 39 days on trader's `default` DB" — and
        // the fix was a best-effort compensating DELETE (plus a second patch to
        // reverse the session `memory_count` the DELETE left behind).
        //
        // It now follows the reserve → commit → publish protocol `correct()` has
        // used since Item 3 (lifecycle.rs): reserve vector capacity BEFORE any
        // durable mutation, commit every durable effect in ONE transaction, then
        // publish (infallible). Backpressure and dim errors now surface having
        // touched nothing, so the orphan-on-Backpressure class is structurally
        // impossible rather than compensated — the DELETE and the memory_count
        // reversal are both deleted below, not relocated.
        //
        // Lock order is CONCURRENCY.md Rule 1 (`conn → … → vec_index`): conn is
        // held across the reservation, the transaction, and the publish. That is
        // load-bearing, exactly as in `correct()` — the conn lock is the only
        // thing serializing append order with commit order (the SyncWriteGuard is
        // a counter, not a mutex). It does not offend Rule 4, whose concern is
        // holding conn across non-O(1) work; a delta reserve/publish is an O(1)
        // Vec push / flag flip.

        // Advisory early reject: cheap, unlocked, and NOT authoritative — the
        // binding check happens under the conn lock below. Doing it here too just
        // avoids the embed/serialize work on an obviously-full queue.
        //
        // This IS "before any side effect" as of 4a.6b: the calibration read
        // upstream is read-only, the stats advance happens inside this write's
        // transaction below, and the warn-gate's flag is counted post-commit —
        // so a rejection here (or anywhere later) leaves no trace anywhere.
        // `record_backpressure_writes_nothing_at_all` is the enforcing test,
        // and its name is the contract.
        //
        // KEYED writes skip the fast check (sol 4a.6c r3): it is a
        // work-avoidance optimization, and for a keyed write the priority
        // inverts — a race-window duplicate must REACH the locked probe below
        // even when the pending queue is full, or saturation can permanently
        // fail a retry that would write nothing. The AUTHORITATIVE locked
        // check still gates every keyed write that wins its claim, so the
        // ceiling holds; the only cost is that a keyed loser does slightly
        // more work before hearing Backpressure.
        if idem.is_none() {
            self.check_pending_backpressure_fast()?;
        }

        let emb_hash = embedding_hash(embedding);
        let record_payload = serde_json::json!({
            "rid": rid,
            "type": memory_type,
            "text": text,
            "importance": importance,
            "valence": valence,
            "half_life": half_life,
            "metadata": metadata,
            "created_at": ts,
            "updated_at": ts,
            "namespace": namespace,
            "certainty": certainty,
            "domain": domain,
            "source": source,
            "emotional_state": emotional_state,
            // 4a.6c: carried so replication's materialize_record writes the
            // same v37 columns the origin row has — a follower's keyed row must
            // mirror its leader's, or the memories partial unique index (the
            // claims table's defense-in-depth) never covers followers. Null for
            // keyless writes; peers on older payloads default to NULL.
            "idempotency_key": idem.as_ref().map(|(k, _)| *k),
            "origin_actor": idem.as_ref().map(|_| self.actor_id.as_str()),
        });
        // **Phase 4.3 Commit B (saga task 3, 2026-05-08).** The unbounded entity
        // / memory_entities / claims loops that used to run inline are enqueued
        // for the materializer thread instead. See docs/phase_4_3_design.md.
        // 4a.6a moves this enqueue INSIDE the transaction: it was previously its
        // own failure boundary, so a crash after the row committed but before the
        // enqueue landed meant that record's entity materialization was skipped
        // FOREVER, with nothing left to indicate it was owed.
        let post_payload = serde_json::json!({
            "rid": rid,
            "text": stored_text,
            "namespace": namespace,
            "ts_secs": ts,
            "domain": domain,
            "source": source,
        });

        let conn = self.conn();

        // 4a.6c sol r2: the LOCKED probe. The unlocked pre-admission probe can
        // race — two same-key writers both MISS, A commits the claim and
        // saturates the engine, and B would then die on the backpressure check
        // below without ever resolving its duplicate. Re-probing here, under
        // the SAME conn guard that stays held through the transaction, closes
        // that window completely: nothing can commit a claim between this read
        // and our tx. A hit resolves BEFORE admission (no backpressure, no seq,
        // no reservation), which is the point — a duplicate writes nothing, so
        // saturation must not be able to fail it. (The in-tx ON CONFLICT stays
        // as the authoritative serialization point; under this locking it is
        // belt-and-suspenders, reachable only by raw-`conn()` writers outside
        // the engine.)
        if let Some((key, digest)) = idem.as_ref() {
            if let Some(existing_rid) = super::idempotency::probe_committed_claim(
                &conn,
                &self.actor_id,
                namespace,
                key,
                digest,
            )? {
                return Ok(existing_rid);
            }
        }

        // THE authoritative admission check: under the lock, before the
        // reservation and before any durable write. The pre-lock check above is a
        // TOCTOU on its own (sol 4a.6a finding 1) — at MAX_PENDING_OPS-1, N
        // writers can all read "under the limit", then serialize here and each
        // commit an enqueue, overshooting the ceiling. Re-reading under the lock
        // serializes the read with the commit that acts on it, so the bound holds.
        self.check_pending_backpressure_locked()?;

        // Mint the seq UNDER the conn lock. Search resolves a rid to its HIGHEST
        // seq, not the most recently appended one, so minting outside the
        // serialized region would let a stalled writer holding seq N append after
        // a writer with seq N+1 committed — serving one writer's vector with
        // another's text. Same reasoning as lifecycle.rs's correction path.
        let seq = self.assign_seq(None);

        // RESERVE: consumes delta capacity and validates dim, but stays invisible
        // to search until published. This is where Backpressure surfaces — before
        // a single durable byte has been written. `AlreadyPresent` is
        // impossible here — the rid is freshly minted — so it is an invariant
        // violation, never a replay.
        if state
            .vec_index
            .append_reserved(rid.clone(), embedding.to_vec(), seq)?
            == crate::vector::delta_index::ReservedAppend::AlreadyPresent
        {
            return Err(crate::error::YantrikDbError::InvalidInput(format!(
                "freshly minted rid {rid} already present in the delta at seq \
                 {seq} — engine invariant violation"
            )));
        }

        // From here until commit, ANY exit — including an unwinding panic —
        // must drop the reservation, or its capacity is held forever
        // (compaction retains unpublished entries by design).
        // with_pending_op, not publish_only: this transaction enqueues a PENDING
        // op (log_op_pending_in_tx, applied=0) that `pending_op_count` caches, so
        // post-commit this writer owes the increment as well as the publish.
        let mut reservation =
            ReservationGuard::with_pending_op(&state, &self.pending_op_count, &rid, seq);

        // ONE transaction: claim (if keyed) + row + session links + the record
        // op + the post-materialization enqueue. Either all of it is durable or
        // none is. Returns Some(existing_rid) on an idempotent hit, in which
        // case the transaction is dropped un-committed (it wrote nothing — the
        // claim lost its ON CONFLICT and everything else comes after).
        let committed = (|| -> Result<Option<String>> {
            let tx = conn.unchecked_transaction()?;
            // 4a.6c: the claim is the FIRST statement — a dup must resolve to a
            // hit/conflict here, not surface later as a bare constraint error
            // from the memories partial unique index.
            if let Some((key, digest)) = idem.as_ref() {
                use super::idempotency::{claim_in_tx, ClaimAttempt, ClaimRow};
                match claim_in_tx(
                    &tx,
                    &ClaimRow {
                        origin_actor: &self.actor_id,
                        namespace,
                        idempotency_key: key,
                        rid: &rid,
                        payload_digest: digest,
                        op_id: &record_op_id,
                        route: "sync",
                        generation: embedding_generation,
                    },
                )? {
                    ClaimAttempt::Won => {}
                    ClaimAttempt::Hit { existing_rid } => return Ok(Some(existing_rid)),
                }
            }
            tx.execute(
                "INSERT INTO memories \
                 (rid, type, text, embedding, created_at, updated_at, importance, \
                  half_life, last_access, valence, metadata, namespace, \
                  certainty, domain, source, emotional_state, embedding_generation, \
                  idempotency_key, origin_actor) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, \
                         ?18, ?19)",
                params![
                    rid,
                    memory_type,
                    stored_text,
                    stored_emb,
                    ts,
                    ts,
                    importance,
                    half_life,
                    ts,
                    valence,
                    stored_meta,
                    namespace,
                    certainty,
                    domain,
                    source,
                    emotional_state,
                    embedding_generation,
                    // v37 idempotency columns: set only for keyed writes (the
                    // partial unique index ignores NULLs, so keyless behavior
                    // is unchanged). origin_actor scopes the key per the
                    // claims-table PK.
                    idem.as_ref().map(|(k, _)| *k),
                    idem.as_ref().map(|_| self.actor_id.as_str()),
                ],
            )?;

            // Auto-link to active session for this namespace.
            if let Some(session_id) = &session_id {
                tx.execute(
                    "UPDATE memories SET session_id = ?1 WHERE rid = ?2",
                    params![session_id, rid],
                )?;
                tx.execute(
                    "UPDATE sessions SET memory_count = memory_count + 1 WHERE session_id = ?1",
                    params![session_id],
                )?;
            }

            // 4a.6b winner-only calibration: the namespace's distribution
            // advances HERE, inside the winning write's transaction, fed the RAW
            // importance (writer intent, not the deflated output). A rollback —
            // or never reaching this transaction at all (backpressure, gate,
            // delta capacity) — leaves the distribution untouched: losers no
            // longer move state.
            self.advance_importance_stats_in_tx(&tx, namespace, raw_importance)?;

            // Kill boundary. Before 4a.6a the row above was already committed by
            // its own autocommit at this point, while the oplog op below had not
            // been written — a process death here left exactly the orphan the
            // "23k rows over 39 days" comment described. Inside the transaction,
            // dying here rolls back BOTH. `kill_record_boundary.rs` proves it.
            crate::testing::fail_point("record.between_row_and_oplog");

            // The user-facing "record" op goes in FIRST so external consumers
            // (replication extract_ops_since, oplog inspectors) see the natural
            // causal order: the record precedes any materialization queued in its
            // wake. `applied_generation` is the guard-pinned snapshot generation,
            // which is the generation the reserved delta entry was written
            // against.
            self.log_op_in_tx(
                &tx,
                "record",
                Some(&rid),
                &record_payload,
                Some(&emb_hash),
                None,
                embedding_generation,
                // The claim (if any) already bound to this id as its recovery
                // evidence — the op and the claim must agree.
                Some(&record_op_id),
            )?;

            // Plain INSERT: if this cannot land, the whole write must fail
            // rather than commit a record whose entity materialization is owed
            // to nobody. See log_op_pending_in_tx.
            self.log_op_pending_in_tx(
                &tx,
                crate::engine::op_types::OP_MATERIALIZE_RECORD_POST,
                Some(&rid),
                &post_payload,
                None,
                None,
            )?;

            tx.commit()?;
            Ok(None)
        })();

        match committed {
            // Durable. The guard's obligation INVERTS here: from "remove the
            // reservation" to "publish it and count the pending op". It is not
            // defused — an unwind between here and complete() must still finish
            // the job, because the row is already committed.
            Ok(None) => reservation.mark_committed(),
            // 4a.6c idempotent hit: the SAME payload already committed under
            // this key. The transaction above was dropped un-committed (it had
            // written nothing), the reservation drops here in Reserved phase and
            // removes the reserved vector entry, and EVERY post-commit effect is
            // skipped — no publish, no pending count, no flag tick, no scoring
            // cache, no visible_seq bump. Repetition is not corroboration: the
            // caller gets the ORIGINAL rid and the store is untouched.
            Ok(Some(existing_rid)) => {
                drop(conn);
                return Ok(existing_rid);
            }
            Err(e) => {
                // Nothing durable exists. `reservation` drops here and removes the
                // entry — a removal, NOT a tombstone (a tombstone would suppress
                // the rid and hide a still-valid older vector).
                drop(conn);
                return Err(e);
            }
        }

        // Durable. PUBLISH makes the vector visible; it is infallible, so there
        // is no failure window between "committed" and "visible". A crash here
        // rebuilds the index from `memories` on next open — the row is
        // authoritative.
        // Discharge both post-commit obligations — publish the vector and count
        // the pending op — in one place the guard also performs on an unwind, so
        // a caught panic cannot strand a durable write with an invisible vector
        // or an uncounted pending row.
        let published = reservation.complete();

        // 4a.6b: a warn-mode Flagged verdict is counted only now — the write is
        // durable, so the nudge metric counts writes that actually landed.
        self.note_flagged_write_committed(gate_verdict);

        // The counter moved inside complete() above — only after the tx
        // committed. Incrementing inside the tx would leak it upward on rollback
        // with no row for `mark_op_applied` to decrement; that drift is monotonic
        // and at MAX_PENDING_OPS wedges every write into Backpressure with an
        // empty queue in SQL. It is unconditional because the enqueue is a plain
        // INSERT in the committed tx, so reaching here means exactly one pending
        // row landed.
        //
        // The assert comes AFTER the counter is discharged: as a post-commit panic
        // point it would otherwise be exactly the hazard the guard exists to
        // close (sol 4a.6a r2 finding 2).
        debug_assert!(
            published,
            "reservation for {rid} seq {seq} vanished before publish"
        );
        if !published {
            tracing::error!(
                rid = %rid,
                seq,
                "reserved vector entry missing at publish — row is durable but \
                 unsearchable until the index is rebuilt from SQL"
            );
        }
        // All obligations are discharged (phase == Done), so this Drop is a no-op.
        // It is explicit only to release the guard's borrow of `rid` before we
        // return it — and it must stay AFTER complete(), which is what makes the
        // drop inert.
        drop(reservation);

        self.cache_insert(
            rid.clone(),
            ScoringRow {
                created_at: ts,
                importance,
                half_life,
                last_access: ts,
                access_count: 0,
                valence,
                consolidation_status: "active".to_string(),
                memory_type: memory_type.to_string(),
                namespace: namespace.to_string(),
                certainty,
                domain: domain.to_string(),
                source: source.to_string(),
                emotional_state: emotional_state.map(|s| s.to_string()),
            },
        );

        // LAST: a read-your-write waiter must not wake against a half-applied
        // record (CONCURRENCY.md: bump visible_seq AFTER the delta append).
        self.bump_visible_seq(namespace, seq);

        drop(conn);
        Ok(rid)
    }

    /// Record multiple memories in a single transaction.
    /// Uses SAVEPOINT for atomicity while keeping `&self` (no `&mut self`).
    #[tracing::instrument(skip(self, inputs), fields(batch_size = inputs.len()))]
    pub fn record_batch(&self, inputs: &[RecordInput]) -> Result<Vec<String>> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }

        // 4a.6d-2a (#98): normalize namespaces ONCE at entry, positionally
        // aligned with `inputs`, and use `namespaces[idx]` for EVERY consumer
        // below — the row, the session lookup, the replicated op payload, the
        // audit event, the scoring cache, the visible_seq bump, and the
        // importance stats. record() and record_text coerce blank namespaces
        // to "default" at entry; record_batch never did, while the calibration
        // helpers it calls normalize INTERNALLY — so one blank-namespace batch
        // item split across two partitions: the row landed under the raw "  "
        // (which no reader queries) and its importance observation advanced
        // "default"'s stats.
        let namespaces: Vec<&str> = inputs
            .iter()
            .map(|i| normalize_namespace(&i.namespace))
            .collect();

        // v0.9.3 contract gate: prevalidate the ENTIRE batch before any side
        // effect (calibration / SQL / oplog / index), so a bad element late
        // in the batch can't leave earlier elements half-committed.
        let mut gate_verdicts: Vec<crate::provenance::GateVerdict> =
            Vec::with_capacity(inputs.len());
        for (i, input) in inputs.iter().enumerate() {
            crate::validate::validate_embedding(
                "record_batch",
                &input.embedding,
                self.embedding_dim,
            )
            .map_err(|e| match e {
                crate::error::YantrikDbError::InvalidEmbedding {
                    path,
                    index,
                    reason,
                } => crate::error::YantrikDbError::InvalidEmbedding {
                    path,
                    index,
                    reason: format!("inputs[{i}]: {reason}"),
                },
                other => other,
            })?;
            crate::validate::validate_scalars(
                "record_batch",
                &[
                    ("importance", input.importance),
                    ("valence", input.valence),
                    ("certainty", input.certainty),
                    ("half_life", input.half_life),
                ],
            )?;
            // v0.10 Item 4a.4b — anti-laundering gate (T06 fan-out). Runs in
            // the batch PREVALIDATION loop, so an inconsistent element late in
            // the batch rejects the whole batch before any side effect rather
            // than half-committing the earlier ones — the same contract the
            // embedding/scalar gates above rely on. Warn-mode Flagged verdicts
            // are only COUNTED after the batch commits (4a.6b).
            let verdict = self
                .gate_provenance(&input.source, &input.metadata)
                .map_err(|e| match e {
                    crate::error::YantrikDbError::ProvenanceInconsistent { path, reason } => {
                        crate::error::YantrikDbError::ProvenanceInconsistent {
                            path,
                            reason: format!("inputs[{i}]: {reason}"),
                        }
                    }
                    other => other,
                })?;
            gate_verdicts.push(verdict);
        }

        // Task 29 (Ingest Integrity): strip any leaked tool-call
        // serialization tail from every input's text once, up front. The
        // same cleaned text feeds entity extraction, the stored row, and the
        // audit features below (indexed positionally — `rids` preserves input
        // order). Borrowed (no allocation) on the clean path; the
        // caller-supplied embedding is left as-is, as in `record`.
        let sanitized_texts: Vec<std::borrow::Cow<'_, str>> = inputs
            .iter()
            .map(|i| sanitize::sanitize_tool_call_artifacts(&i.text))
            .collect();

        // ── 4a.6d-2b: per-item idempotency, prevalidated with everything else ──
        //
        // Digests are the canonical RAW payload exactly as
        // `record_with_idempotency` computes them — SANITIZED text, NORMALIZED
        // namespace, RAW importance, the caller-supplied embedding included
        // (PayloadVariant::Record) — so the same key with a byte-identical
        // payload is the SAME write whether it arrives via record() or a batch
        // item, and a divergent payload conflicts identically. The two
        // overrides below are load-bearing: `from_record_input` views the raw
        // struct, and digesting raw text/namespace would make an honest
        // cross-surface retry a false conflict.
        //
        // In-batch duplicates resolve here, before any probe or side effect:
        // the same (namespace, key) twice with the same digest makes the later
        // item an ALIAS of the first (one write, both positions return its
        // rid); with a different digest the whole batch fails typed — batches
        // are all-or-nothing on failure, and silently dropping one divergent
        // item would leave a retry unable to tell which content won.
        let n = inputs.len();
        let mut digests: Vec<Option<[u8; 32]>> = vec![None; n];
        let mut alias_of: Vec<Option<usize>> = vec![None; n];
        // resolved[i] = the committed rid a keyed item hit — set by the probes.
        let mut resolved: Vec<Option<String>> = vec![None; n];
        {
            let mut first_by_key: std::collections::HashMap<(&str, &str), usize> =
                std::collections::HashMap::new();
            for (i, input) in inputs.iter().enumerate() {
                let Some(key) = input.idempotency_key.as_deref() else {
                    continue;
                };
                if key.trim().is_empty() || key.len() > 512 {
                    return Err(crate::error::YantrikDbError::InvalidIdempotencyKey {
                        reason: if key.len() > 512 {
                            format!("inputs[{i}]: key is {} bytes; max 512", key.len())
                        } else {
                            format!("inputs[{i}]: key is empty or whitespace-only")
                        },
                    });
                }
                let mut view = crate::payload_digest::PayloadView::from_record_input(
                    input,
                    crate::payload_digest::PayloadVariant::Record,
                );
                view.namespace = namespaces[i];
                view.text = sanitized_texts[i].as_ref();
                let digest = crate::payload_digest::payload_digest(&view);
                match first_by_key.entry((namespaces[i], key)) {
                    std::collections::hash_map::Entry::Occupied(e) => {
                        let j = *e.get();
                        if digests[j] != Some(digest) {
                            return Err(crate::error::YantrikDbError::IdempotencyConflict {
                                namespace: namespaces[i].to_string(),
                                existing_rid: String::new(),
                                reason: format!(
                                    "inputs[{i}] reuses inputs[{j}]'s idempotency key \
                                     with a DIFFERENT payload — change the key or make \
                                     the payloads identical"
                                ),
                            });
                        }
                        alias_of[i] = Some(j);
                    }
                    std::collections::hash_map::Entry::Vacant(v) => {
                        v.insert(i);
                    }
                }
                digests[i] = Some(digest);
            }
        }

        // Positional result assembly, shared by the two early-resolution exits
        // and the final return. alias roots are always original (non-alias)
        // items, so one hop suffices.
        fn assemble_rids(
            resolved: &[Option<String>],
            alias_of: &[Option<usize>],
            rid_slots: &[Option<String>],
        ) -> Vec<String> {
            (0..resolved.len())
                .map(|i| {
                    let root = alias_of[i].unwrap_or(i);
                    resolved[root]
                        .clone()
                        .or_else(|| rid_slots[root].clone())
                        .expect("every position resolves to a hit or a written rid")
                })
                .collect()
        }

        // 4a.6d-2b unlocked pre-admission probe (the 4a.6c invariant on the
        // batch surface): no RESOURCE admission check may reject a duplicate
        // that would write nothing. Committed hits leave the write set here —
        // BEFORE the write-router, the delta reservation, and the seq mint —
        // so a fully-duplicate batch resolves to its rids even during a
        // reembed cutover or under full delta saturation. That is exactly
        // when clients retry. The guarantee is deliberately NARROWER than
        // "nothing rejects a duplicate": prevalidation (embedding/scalar/
        // provenance gates, key format, in-batch divergence) runs above and
        // still errors first — those are deterministic payload-shape checks
        // an identical retry passes identically, not saturation-dependent
        // admission that would starve a retry loop (sol 4a.6d-2b r1
        // finding 2). A MISS is advisory (the locked probe under the conn
        // guard below is what closes the race window); a HIT is final —
        // committed claims are immutable in 4a.
        if digests.iter().any(Option::is_some) {
            let conn = self.conn();
            for i in 0..n {
                if alias_of[i].is_some() {
                    continue;
                }
                if let (Some(digest), Some(key)) =
                    (digests[i].as_ref(), inputs[i].idempotency_key.as_deref())
                {
                    if let Some(existing_rid) = super::idempotency::probe_committed_claim(
                        &conn,
                        &self.actor_id,
                        namespaces[i],
                        key,
                        digest,
                    )? {
                        resolved[i] = Some(existing_rid);
                    }
                }
            }
        }
        let all_resolved = (0..n).all(|i| resolved[alias_of[i].unwrap_or(i)].is_some());
        if all_resolved {
            // Every item is an idempotent hit: the batch writes nothing and
            // returns the original rids, saturated engine or not.
            return Ok(assemble_rids(&resolved, &alias_of, &vec![None; n]));
        }

        // Task 31 (Ingest Integrity): each input's importance is calibrated
        // against its namespace distribution INSIDE the savepoint below,
        // positionally aligned with `inputs` via push order. Every item
        // calibrates against the SAME pre-batch snapshot (4a.6b: a batch is
        // one simultaneous act, no within-batch running mean): all the
        // calibration READS happen in the item loop, and the stats ADVANCES
        // run after it — still inside the savepoint, so a rejected batch
        // rolls its advances back with everything else. That in-savepoint
        // advance is safe again BECAUSE of 4a.6d-2a: capacity is reserved
        // before the savepoint opens, so nothing can fail after RELEASE —
        // the deferred best-effort advance (and its silently-skipped-
        // observation gap) existed only to survive the post-RELEASE append
        // failure that no longer exists.
        //
        // 4a.6d-2b: indexed by ORIGINAL input position, `Some` only for items
        // that write — hits and aliases never calibrate (they store nothing).
        let mut calibrated_importances: Vec<Option<f64>> = vec![None; n];

        // **Issue #41 layer 3.** Enter the write-router BEFORE snapshotting
        // SearchState, exactly as `record()` does (record.rs:105/138).
        //
        // record_batch used to skip the router entirely and load the snapshot
        // below unguarded. Nothing then stopped a `db.reembed()` cutover from
        // completing its swap mid-batch, so the batch could commit rows and
        // appends stamped with `embedding_generation` from a generation that was
        // being discarded — the exact corruption the guard exists to prevent.
        // The comment below claimed "every append lands on the same
        // generation-anchored DeltaIndex" while nothing enforced it.
        //
        // Unlike `record()`, there is no queued fallback: no queued-batch
        // primitive exists yet (v0.10 Item 4a.6c), and routing items
        // independently would break the batch's all-or-nothing contract. So the
        // whole batch defers with a typed retryable error, following the Item 3
        // precedent (`CorrectionDeferredDuringReembed`) — nothing durable has
        // happened at this point, so the caller can reissue verbatim.
        let Some(_sync_guard) = self.write_router.try_enter_sync_writer() else {
            return Err(crate::error::YantrikDbError::BatchDeferredDuringReembed {
                count: inputs.len(),
            });
        };

        // **Issue #41 brainstorm-4 §1.** SearchState snapshot for the
        // batch — every append in this batch lands on the same
        // generation-anchored DeltaIndex. Loaded AFTER the guard above, which is
        // what actually makes that true: with the guard held, reembed cannot
        // complete its swap for the rest of this call. `_sync_guard` drops via
        // RAII at function exit (panic-safe), covering the SQL work AND the
        // vector appends.
        let state = self.search_state.load_full();

        // Clone active sessions map before acquiring conn
        let sessions = self.active_sessions.read().clone();

        // Precompute entity candidates per memory before touching conn/graph_index.
        // Two sources:
        //   (a) heuristic extraction from text (capitalized proper-nouns)
        //   (b) match against already-known entities in graph_index
        let known_entities = self.graph_index.read().all_entity_names();
        // 4a.6d-2b: `None` for items leaving the write set as idempotent hits
        // or in-batch aliases — a hit writes nothing, so it must not
        // re-extract entities (mention_count would inflate on every retry,
        // the #80 class), and an alias is the SAME text as its root (one
        // write, one extraction). Locked-probe hits below leave an unused
        // `Some` here; unused is harmless, used-would-be-the-bug.
        let per_memory_linkage: Vec<Option<(Vec<String>, std::collections::HashSet<String>)>> =
            sanitized_texts
                .iter()
                .enumerate()
                .map(|(i, text)| {
                    if resolved[i].is_some() || alias_of[i].is_some() {
                        return None;
                    }
                    let text = text.as_ref();
                    let text_tokens = crate::graph::tokenize(text);
                    let heuristic = crate::graph::extract_heuristic_entities(text);
                    let mut candidates: std::collections::HashSet<String> =
                        heuristic.iter().cloned().collect();
                    for known in &known_entities {
                        if crate::graph::entity_matches_text(known, &text_tokens) {
                            candidates.insert(known.clone());
                        }
                    }
                    Some((heuristic, candidates))
                })
                .collect();

        // Per-ORIGINAL-index slots: `Some` only for items that actually write.
        // Hits and aliases stay `None` and resolve positionally at return.
        let mut rid_slots: Vec<Option<String>> = vec![None; n];
        let mut seq_slots: Vec<Option<u64>> = vec![None; n];
        // 4a.6d-2b (#94): op ids preminted BEFORE the transaction, exactly as
        // record() premints `record_op_id` — a keyed item's claim binds to its
        // op id as recovery evidence, so the id must exist before the claim
        // INSERT, and the op itself now commits INSIDE the savepoint.
        let mut op_ids: Vec<Option<String>> = vec![None; n];

        // 4a.6d-2a (#92): the batch's reserve → commit → publish guard.
        // Declared OUTSIDE the conn scope: it publishes the vectors after the
        // savepoint's fate is decided (and the conn lock is released), and on
        // ANY pre-commit exit its Drop removes every reservation taken so far.
        let mut batch_reservation =
            super::reservation::BatchReservationGuard::new(&state, inputs.len());

        // Lock conn once for the entire batch SQL work
        {
            let conn = self.conn();

            // 4a.6d-2b LOCKED probe (4a.6c sol r2, batch surface): the
            // unlocked probe above can race — another same-key writer may
            // commit between it and this lock. Re-probing under the SAME conn
            // guard that stays held through the transaction closes the window
            // completely: nothing can commit a claim between this read and our
            // tx. Items hitting here leave the write set before any capacity
            // is reserved. (The in-tx ON CONFLICT below stays the
            // authoritative serialization point; under this locking it is
            // belt-and-suspenders, reachable only by raw-`conn()` writers
            // outside the engine.)
            for i in 0..n {
                if resolved[i].is_some() || alias_of[i].is_some() {
                    continue;
                }
                if let (Some(digest), Some(key)) =
                    (digests[i].as_ref(), inputs[i].idempotency_key.as_deref())
                {
                    if let Some(existing_rid) = super::idempotency::probe_committed_claim(
                        &conn,
                        &self.actor_id,
                        namespaces[i],
                        key,
                        digest,
                    )? {
                        resolved[i] = Some(existing_rid);
                    }
                }
            }
            if (0..n).all(|i| resolved[alias_of[i].unwrap_or(i)].is_some()) {
                // The race resolved every remaining item: nothing to write.
                drop(conn);
                return Ok(assemble_rids(&resolved, &alias_of, &rid_slots));
            }

            // RESERVE delta capacity for every WRITING item BEFORE the
            // savepoint opens — the batch analogue of record()'s protocol
            // (4a.6a). This is where Backpressure and dim mismatches surface:
            // before a single durable byte. The old shape appended AFTER the
            // RELEASE and compensated failure with a DELETE that reversed rows
            // and session counts but could never reverse `entities` upserts,
            // `memory_entities` links, or the in-memory graph_index (#92) —
            // now there is nothing to compensate. Idempotent hits and in-batch
            // aliases reserve NOTHING: a duplicate writes nothing, so it must
            // never consume capacity a fresh write is then denied. Seqs are
            // minted under the conn lock for the same reason record() mints
            // there: search resolves a rid to its HIGHEST seq, so minting must
            // serialize with the commits that act on it.
            for (i, input) in inputs.iter().enumerate() {
                if resolved[i].is_some() || alias_of[i].is_some() {
                    continue;
                }
                let rid = crate::id::new_id();
                let seq = self.assign_seq(None);
                batch_reservation.reserve(rid.clone(), input.embedding.clone(), seq)?;
                rid_slots[i] = Some(rid);
                seq_slots[i] = Some(seq);
                op_ids[i] = Some(crate::id::new_id());
            }

            // #91: RAII, not manual unwinding. EVERY fallible statement below
            // — serde, the encrypt wrappers, each INSERT, the stats advances —
            // may `?`-return and the guard's Drop runs `ROLLBACK TO; RELEASE`.
            // The old error arm rolled back WITHOUT releasing (and the paths
            // before the INSERT returned without even the rollback), leaving
            // the savepoint open on the engine's single shared connection so
            // every later write silently nested inside it.
            let savepoint = super::savepoint::SavepointGuard::new(&conn, "batch_record")?;

            // 4a.6c rule, batch surface: the claims are the FIRST statements of
            // the transaction — a dup must resolve to a hit/conflict at the
            // claim, never surface later as a bare constraint error from the
            // v37 partial unique index on memories. All claims precede all row
            // INSERTs; each binds to its item's preminted op id.
            for (i, input) in inputs.iter().enumerate() {
                if resolved[i].is_some() || alias_of[i].is_some() {
                    continue;
                }
                let (Some(digest), Some(key)) =
                    (digests[i].as_ref(), input.idempotency_key.as_deref())
                else {
                    continue;
                };
                use super::idempotency::{claim_in_tx, ClaimAttempt, ClaimRow};
                match claim_in_tx(
                    &conn,
                    &ClaimRow {
                        origin_actor: &self.actor_id,
                        namespace: namespaces[i],
                        idempotency_key: key,
                        rid: rid_slots[i].as_deref().expect("write items have rids"),
                        payload_digest: digest,
                        op_id: op_ids[i].as_deref().expect("write items have op ids"),
                        route: "batch",
                        generation: state.generation as i64,
                    },
                )? {
                    ClaimAttempt::Won => {}
                    // Unreachable in 4a: the LOCKED probe above runs under the
                    // SAME conn guard held continuously through this
                    // transaction, and the conn mutex is the engine's single
                    // write lock — no other claim can commit in between. Loud,
                    // not silent, if that invariant ever breaks; the savepoint
                    // guard rolls the whole batch back.
                    ClaimAttempt::Hit { existing_rid } => {
                        return Err(crate::error::YantrikDbError::IdempotencyConflict {
                            namespace: namespaces[i].to_string(),
                            existing_rid,
                            reason: format!(
                                "inputs[{i}]: claim committed between the locked probe \
                                 and the batch transaction — impossible under the \
                                 engine's single-writer conn unless the claims table \
                                 is written outside the engine"
                            ),
                        });
                    }
                }
            }

            for (idx, input) in inputs.iter().enumerate() {
                if resolved[idx].is_some() || alias_of[idx].is_some() {
                    continue;
                }
                let rid = rid_slots[idx].as_ref().expect("write items have rids");
                let ts = now();
                let emb_blob = serialize_f32(&input.embedding);
                let meta_str = serde_json::to_string(&input.metadata)?;

                // 4a.6b: calibrate under the savepoint. The `_on` variant
                // reads through the HELD guard — calling the locking wrapper
                // here would re-lock `conn` on the same thread, the
                // `learn_category_members` deadlock (#83). Read-only: the
                // matching advances run after this loop (same-snapshot
                // calibration — see the `calibrated_importances` comment).
                let calibrated = super::importance::calibrated_importance_on(
                    &conn,
                    namespaces[idx],
                    input.importance,
                )?;
                calibrated_importances[idx] = Some(calibrated);

                // Encrypt fields if encryption is enabled. Task 29: store the
                // sanitized text (positionally aligned with `inputs`).
                let stored_text = self.encrypt_text(sanitized_texts[idx].as_ref())?;
                let stored_meta = self.encrypt_text(&meta_str)?;
                let stored_emb = self.encrypt_embedding(&emb_blob)?;

                // **Issue #41 brainstorm-4 §6.** v28 embedding_generation
                // stamped from the batch's snapshot.
                let embedding_generation: i64 = state.generation as i64;
                conn.execute(
                    "INSERT INTO memories \
                     (rid, type, text, embedding, created_at, updated_at, importance, \
                      half_life, last_access, valence, metadata, namespace, \
                      certainty, domain, source, emotional_state, embedding_generation, \
                      idempotency_key, origin_actor) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, \
                             ?18, ?19)",
                    params![rid, input.memory_type, stored_text, stored_emb, ts, ts,
                            calibrated, input.half_life, ts, input.valence, stored_meta,
                            namespaces[idx], input.certainty, input.domain, input.source,
                            input.emotional_state, embedding_generation,
                            // v37 idempotency columns, exactly as record()
                            // stamps them: set only for keyed items (the
                            // partial unique index ignores NULLs, so unkeyed
                            // behavior is unchanged).
                            input.idempotency_key.as_deref(),
                            input.idempotency_key.as_ref().map(|_| self.actor_id.as_str())],
                )?;

                // The canonical "record" op, byte-for-byte what record() emits,
                // so peers materialize batch items through the existing, tested
                // "record" arm. Plaintext text/metadata (the encrypted forms
                // above are the at-rest representation, not the replication
                // one); NORMALIZED namespace (#98) so peers land the row in the
                // same partition this node did. 4a.6d-2b (#94): committed HERE,
                // inside the savepoint, under the item's preminted op id — the
                // op the claim binds to either commits with the row or neither
                // exists. The old shape logged all ops AFTER the release, so a
                // crash in between left durable rows that never replicated,
                // and an Err there surfaced as a failure for a batch that had
                // already committed (a retry then duplicated every row).
                let record_payload = serde_json::json!({
                    "rid": rid,
                    "type": input.memory_type,
                    "text": sanitized_texts[idx].as_ref(),
                    "importance": calibrated,
                    "valence": input.valence,
                    "half_life": input.half_life,
                    "metadata": input.metadata,
                    "created_at": ts,
                    "updated_at": ts,
                    "namespace": namespaces[idx],
                    "certainty": input.certainty,
                    "domain": input.domain,
                    "source": input.source,
                    "emotional_state": input.emotional_state,
                    // 4a.6c/4a.6d-2b (sol r2 finding 1): carried so
                    // replication's materialize_record writes the same v37
                    // columns the origin row has — a follower's keyed row must
                    // mirror its leader's, or the memories partial unique
                    // index (the claims table's defense-in-depth) never covers
                    // followers for batch writes. Null for keyless items;
                    // peers on older payloads default to NULL. Same two
                    // fields record() and record_queued emit.
                    "idempotency_key": input.idempotency_key.as_deref(),
                    "origin_actor": input.idempotency_key.as_ref().map(|_| self.actor_id.as_str()),
                });
                let emb_hash = embedding_hash(&input.embedding);
                self.log_op_in_tx(
                    &conn,
                    "record",
                    Some(rid),
                    &record_payload,
                    Some(&emb_hash),
                    None,
                    embedding_generation,
                    Some(op_ids[idx].as_deref().expect("write items have op ids")),
                )?;
            }

            // Auto-link batch to active sessions
            for (idx, rid_slot) in rid_slots.iter().enumerate() {
                let Some(rid) = rid_slot else { continue };
                if let Some(session_id) = sessions.get(namespaces[idx]) {
                    conn.execute(
                        "UPDATE memories SET session_id = ?1 WHERE rid = ?2",
                        params![session_id, rid],
                    )?;
                    conn.execute(
                        "UPDATE sessions SET memory_count = memory_count + 1 WHERE session_id = ?1",
                        params![session_id],
                    )?;
                }
            }

            // Persist entity linkage (SQL side). graph_index in-memory update
            // happens after conn is dropped to avoid holding two write locks.
            // Gated on rid_slots, not the linkage Option: a locked-probe hit
            // has a stale Some(linkage) that must not be persisted.
            let batch_ts = now();
            for (rid, linkage) in rid_slots.iter().zip(per_memory_linkage.iter()) {
                let (Some(rid), Some((heuristic, candidates))) = (rid, linkage) else {
                    continue;
                };
                for entity in heuristic {
                    let entity_type = crate::graph::classify_entity_type(entity);
                    conn.execute(
                        "INSERT INTO entities (name, entity_type, first_seen, last_seen, mention_count) \
                         VALUES (?1, ?2, ?3, ?3, 1) \
                         ON CONFLICT(name) DO UPDATE SET \
                            last_seen = ?3, \
                            mention_count = mention_count + 1, \
                            entity_type = CASE \
                                WHEN entity_type = 'unknown' AND ?2 != 'unknown' THEN ?2 \
                                ELSE entity_type END",
                        params![entity, entity_type, batch_ts],
                    )?;
                }
                for entity in candidates {
                    conn.execute(
                        "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name) VALUES (?1, ?2)",
                        params![rid, entity],
                    )?;
                }
            }

            // 4a.6b winner-only calibration, 4a.6d-2a placement: the advances
            // run INSIDE the savepoint, after every calibration READ above —
            // so items still calibrate against the same pre-batch snapshot,
            // and a rejected batch rolls its advances back with everything
            // else instead of permanently moving a namespace's distribution.
            // This retires the old post-commit best-effort advance: that
            // deferral existed only because the vector append could fail
            // after RELEASE, and with capacity reserved up front it cannot.
            // An Err here is PRE-commit — the whole batch rolls back and a
            // retry writes once — so `?` is correct where post-commit it was
            // the retry-duplicates trap (sol 4a.6b r3 finding 1).
            // Write items only: an idempotent hit stores nothing, so it must
            // not advance the distribution — repetition is not corroboration
            // (T07), and record()'s hit path skips this identically.
            for (idx, input) in inputs.iter().enumerate() {
                if rid_slots[idx].is_none() {
                    continue;
                }
                self.advance_importance_stats_in_tx(&conn, namespaces[idx], input.importance)?;
            }

            savepoint.release()?;
            // The obligation inverts HERE: the rows are durable, so from this
            // point the reservations owe publish, not removal. Nothing
            // fallible may sit between the RELEASE and this call.
            batch_reservation.mark_committed();
        }
        // NOTE: warn-mode flag ticks are DEFERRED to after the publish below
        // (4a.6b finding 1): the nudge metric counts writes that landed and
        // became visible, in the same order record() counts them.
        // conn dropped; now update graph_index in-memory. Write items only —
        // gated on rid_slots (a hit's stale linkage must not re-link).
        {
            let mut gi = self.graph_index.write();
            for (rid, linkage) in rid_slots.iter().zip(per_memory_linkage.iter()) {
                let (Some(rid), Some((_, candidates))) = (rid, linkage) else {
                    continue;
                };
                for entity in candidates {
                    let entity_type = crate::graph::classify_entity_type(entity);
                    gi.add_entity(entity, entity_type);
                    gi.link_memory(rid, entity);
                }
            }
        }

        // RFC 006 Phase 0: emit one audit event per memory WRITTEN in the
        // batch. Hits and aliases extracted nothing and stored nothing, so an
        // audit event for them would attest an extraction that never ran.
        for (idx, input) in inputs.iter().enumerate() {
            let (Some(rid), Some((heuristic_entities, candidates))) =
                (rid_slots[idx].as_ref(), per_memory_linkage[idx].as_ref())
            else {
                continue;
            };
            let heuristic_vec: Vec<String> = heuristic_entities.iter().cloned().collect();
            let features =
                crate::graph::analyze_text_features(sanitized_texts[idx].as_ref(), &heuristic_vec);
            tracing::info!(
                target: "yantrikdb::audit::extraction",
                namespace = %namespaces[idx],
                memory_rid = %rid,
                domain = %input.domain,
                source = %input.source,
                extractor_version = "heuristic_v1",
                batch = true,
                char_length = features.char_length,
                sentence_count = features.sentence_count,
                entity_count = features.entity_count,
                entities_matched_in_graph = candidates.len().saturating_sub(heuristic_entities.len()),
                negation_cue_count = features.negation_cue_count,
                temporal_cue_count = features.temporal_cue_count,
                modality_cue_count = features.modality_cue_count,
                has_compound_markers = features.has_compound_markers,
                likely_assertion = features.likely_assertion,
                "extraction audit"
            );
        }

        // Durable. PUBLISH all N vectors — infallible, so there is no failure
        // window between "committed" and "visible", and therefore nothing to
        // compensate. (This retires the v0.7.19 compensating DELETE and its
        // v0.7.23 session-count reversal: both existed because the append used
        // to happen HERE, after the commit, where it could still fail. The
        // DELETE never reversed `entities`, `memory_entities`, or the
        // in-memory graph_index — #92 — which is unfixable by adding more
        // compensation and gone by construction with the up-front reserve.)
        let all_published = batch_reservation.complete();
        debug_assert!(
            all_published,
            "batch reservation vanished before publish (rids {rid_slots:?})"
        );
        if !all_published {
            tracing::error!(
                batch_size = rid_slots.iter().flatten().count(),
                "reserved batch vector entries missing at publish — rows are \
                 durable but unsearchable until the index is rebuilt from SQL"
            );
        }
        drop(batch_reservation);

        // LAST: a read-your-write waiter must not wake against a half-applied
        // batch (CONCURRENCY.md: bump visible_seq AFTER the delta publish).
        for idx in 0..n {
            if let Some(seq) = seq_slots[idx] {
                self.bump_visible_seq(namespaces[idx], seq);
            }
        }
        // 4a.6b: the warn-mode flags are counted only now — the batch is
        // durable AND visible, so the nudge metric counts writes that landed.
        // Write items only: a hit landed nothing, so its verdict must not
        // tick (record()'s hit path returns before its tick identically).
        for (idx, verdict) in gate_verdicts.into_iter().enumerate() {
            if rid_slots[idx].is_none() {
                continue;
            }
            self.note_flagged_write_committed(verdict);
        }
        // vec_index dropped, now scoring_cache — write items only (a hit's
        // row already has its cache entry from its original write).
        {
            let mut cache = self.scoring_cache.write();
            for (idx, input) in inputs.iter().enumerate() {
                let Some(rid) = rid_slots[idx].as_ref() else {
                    continue;
                };
                let ts = now();
                cache.insert(
                    rid.clone(),
                    ScoringRow {
                        created_at: ts,
                        importance: calibrated_importances[idx]
                            .expect("write items calibrated in the savepoint"),
                        half_life: input.half_life,
                        last_access: ts,
                        access_count: 0,
                        valence: input.valence,
                        consolidation_status: "active".to_string(),
                        memory_type: input.memory_type.clone(),
                        namespace: namespaces[idx].to_string(),
                        certainty: input.certainty,
                        domain: input.domain.clone(),
                        source: input.source.clone(),
                        emotional_state: input.emotional_state.clone(),
                    },
                );
            }
        }

        // The per-item "record" ops committed INSIDE the savepoint above
        // (4a.6d-2b, closing #94 for this path) — there is no post-commit
        // oplog write left to fail. Positional assembly: written items return
        // their fresh rids, hits their original rids, in-batch aliases their
        // root's rid.
        Ok(assemble_rids(&resolved, &alias_of, &rid_slots))
    }

    /// **Issue #9 — deterministic mutation primitive for cluster replication.**
    ///
    /// Sibling of `record()` that takes a caller-assigned rid + caller-supplied
    /// embedding + materialized extracted_entities + caller-supplied
    /// timestamp + embedding_model. Engine does NOT call its own embedder
    /// or NER. Used by yantrikdb-server's cluster-mode applier so
    /// replicated writes are byte-deterministic across leader + followers.
    ///
    /// **`admission` (4a.6b, sol r2 finding 2) — REQUIRED, choose deliberately.**
    /// This method is both a public origin API and the cluster apply primitive.
    /// Pass `WriteAdmission::Admitted` from any consensus/replication APPLY path
    /// (the leader already gated the op at origin ingress; re-gating on the apply
    /// path makes followers reject the leader's committed write and wedge the
    /// cluster). Pass `WriteAdmission::Origin` for a genuinely new write entering
    /// here for the first time — it runs the anti-laundering gate exactly like
    /// `record()`. It is a required argument, not a defaulted flag, so a caller
    /// cannot silently inherit the apply bypass: `record_with_rid` was a public
    /// Enforce bypass before this (a direct caller could persist
    /// `source=inference` + `kind=fact`).
    ///
    /// # Contract
    ///
    /// - **Idempotent on rid**: a second call with the same rid + identical
    ///   other fields succeeds without error and produces identical engine
    ///   state (INSERT OR IGNORE on memories, INSERT OR IGNORE on entities,
    ///   INSERT OR IGNORE on memory_entities, DeltaIndex.append idempotent
    ///   on rid+seq).
    /// - **Caller supplies the embedding.** Engine validates dim and rejects
    ///   `Error::EmbeddingDimensionMismatch` on mismatch — diverged dim is
    ///   undetectable until a query notices, so we fail loudly.
    /// - **Caller supplies created_at_unix_micros.** Materialized into both
    ///   `created_at REAL` (for back-compat scoring) and the v25
    ///   `created_at_unix_micros INTEGER` column. No engine-side `now()`
    ///   call on this path — leader stamps once, followers replay verbatim.
    /// - **Caller supplies extracted_entities.** Engine writes entity_edges
    ///   accordingly. Empty slice = no edges; engine does NOT fall back to
    ///   its own NER. (Heuristic NER lives in `crate::knowledge::graph` and
    ///   is callable directly by the leader if needed — see issue #9 thread.)
    /// - **Caller supplies embedding_model.** Stored on the row as the
    ///   engine-deterministic-surface version pin. RFC 013 may swap the
    ///   field type later behind the same column name.
    /// - **Caller-supplied `seq`** (cluster mode): when `Some(n)`, the
    ///   engine uses `n` as the delta-entry seq and the visible_seq bump
    ///   value, and ratchets `vec_seq` up to at least `n`. Per design
    ///   lock 2026-05-07, the seq IS the openraft commit-log index in
    ///   cluster mode, giving byte-deterministic per-namespace
    ///   visible_seq across leader + followers. Single-node callers pass
    ///   `None` and the engine allocates the seq itself.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success or idempotent re-apply. The rid is the input,
    /// not the output — caller already owns it.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(
        skip(self, metadata, embedding, extracted_entities),
        fields(rid, memory_type, namespace, embedding_model)
    )]
    pub fn record_with_rid(
        &self,
        rid: &str,
        text: &str,
        memory_type: &str,
        importance: f64,
        valence: f64,
        half_life: f64,
        metadata: &serde_json::Value,
        embedding: &[f32],
        namespace: &str,
        certainty: f64,
        domain: &str,
        source: &str,
        emotional_state: Option<&str>,
        created_at_unix_micros: i64,
        extracted_entities: &[&str],
        embedding_model: &str,
        seq: Option<u64>,
        admission: crate::provenance::WriteAdmission,
    ) -> Result<()> {
        // 4a.6b (sol r2 finding 2): the anti-laundering gate. ORIGIN callers are
        // gated exactly like record(); ADMITTED callers (the materializer drain,
        // replication apply) are NOT re-gated — the op was gated at the leader's
        // origin ingress, and re-gating the apply path would make followers
        // reject the leader's consensus-committed write and wedge the cluster
        // (yantrikdb-server). A warn-mode Flagged verdict is counted post-commit.
        let gate_verdict = match admission {
            crate::provenance::WriteAdmission::Origin => self.gate_provenance(source, metadata)?,
            crate::provenance::WriteAdmission::Admitted => crate::provenance::GateVerdict::Clean,
        };
        // v0.7.23: coerce a blank namespace to the canonical default. This is
        // the path the server's commit applier uses (record_with_rid on every
        // node), so it closes the gateway `unwrap_or("")` footgun at the
        // engine boundary for all replicas.
        let namespace = normalize_namespace(namespace);
        // Determinism gate: dim must match. Diverged dim = silent corruption.
        // (Kept as the pre-v0.9.3 EmbeddingDimensionMismatch variant so
        // replication callers matching on it keep working.)
        if embedding.len() != self.embedding_dim {
            return Err(crate::error::YantrikDbError::EmbeddingDimensionMismatch {
                expected: self.embedding_dim,
                got: embedding.len(),
            });
        }
        // v0.9.3 contract gate: finiteness for the vector + scalars (dim
        // already checked above). Replicated writes must not persist NaN.
        crate::validate::validate_embedding("record_with_rid", embedding, self.embedding_dim)?;
        crate::validate::validate_scalars(
            "record_with_rid",
            &[
                ("importance", importance),
                ("valence", valence),
                ("certainty", certainty),
                ("half_life", half_life),
            ],
        )?;

        // **Issue #41 brainstorm-4 §1.** SearchState snapshot for the
        // determinstic-replay path. The replicated write lands on the
        // currently-active generation's DeltaIndex.
        let state = self.search_state.load_full();

        // Caller-supplied timestamp — NEVER call now() on this path.
        let ts_secs = (created_at_unix_micros as f64) / 1_000_000.0;
        let emb_blob = serialize_f32(embedding);
        let meta_str = serde_json::to_string(metadata)?;

        // Encryption is engine-side and deterministic given the same DEK +
        // same plaintext bytes (AES-GCM is non-deterministic across IVs but
        // the encrypt-once-on-leader model means each follower receives the
        // already-encrypted bytes via the WAL replication path — Phase 4
        // wires that. For now we encrypt locally; cluster-mode follower
        // apply will skip this step in a follow-up patch.)
        let stored_text = self.encrypt_text(text)?;
        let stored_meta = self.encrypt_text(&meta_str)?;
        let stored_emb = self.encrypt_embedding(&emb_blob)?;

        let session_id = self.active_sessions.read().get(namespace).cloned();

        // ── 4a.6d-3: reserve → one savepoint → publish ──────────────────
        // The pre-port shape was record()'s pre-4a.6a disease, worse: the row
        // committed alone; the vector appended AFTER (with a partial
        // compensating DELETE on failure — the #92 class); and the op and the
        // materialization enqueue ran as separate post-commit autocommits. A
        // failure or kill between the commit and the op was UNREPAIRABLE:
        // the retry takes the was_new_row=false arm, which skips log_op by
        // design, so the row existed forever with no oplog provenance and the
        // write never replicated (kill_record_with_rid_boundary.rs). Now the
        // row, the session link, the op, and the enqueue commit in ONE
        // savepoint, with the vector reserved before it opens and published
        // after RELEASE.
        let emb_hash = embedding_hash(embedding);
        let conn = self.conn();

        // Seq minted under the conn lock (search resolves a rid to its
        // HIGHEST seq; minting in the serialized region keeps seq order and
        // commit order aligned — record()'s rationale). Cluster callers pass
        // their commit-log index; fetch_max ratchets either way.
        let seq = self.assign_seq(seq);

        // RESERVE before any SQL — Backpressure and dim surface here, before
        // a single durable byte. `inserted == false` is the deterministic-
        // replay case: an identical (rid, seq) is already in the delta
        // (cluster re-delivery; the prior apply published it), and this call
        // then owes the delta NOTHING — publishing is done, and a removal on
        // failure would delete the prior write's PUBLISHED vector — so no
        // guard is constructed at all. `inserted == true` with an EXISTING
        // row (was_new_row=false below) is the repair case the old
        // post-commit append also served: a row whose vector was lost gets
        // it re-published.
        let inserted = state
            .vec_index
            .append_reserved(rid.to_string(), embedding.to_vec(), seq)?
            == crate::vector::delta_index::ReservedAppend::Inserted;
        let mut reservation = if inserted {
            Some(ReservationGuard::publish_only(&state, rid, seq))
        } else {
            None
        };

        // #91's class: RAII — every `?` below and any unwinding panic rolls
        // the whole write back AND releases the frame.
        let savepoint = super::savepoint::SavepointGuard::new(&conn, "record_with_rid")?;

        // **Issue #41 brainstorm-4 §6.** v28 embedding_generation
        // stamp from the SearchState snapshot loaded above.
        let embedding_generation: i64 = state.generation as i64;
        let inserted_row = conn.execute(
            "INSERT OR IGNORE INTO memories \
             (rid, type, text, embedding, created_at, updated_at, importance, \
              half_life, last_access, valence, metadata, namespace, \
              certainty, domain, source, emotional_state, \
              created_at_unix_micros, embedding_model, embedding_generation) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?5, ?6, ?7, ?5, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            params![
                rid, memory_type, stored_text, stored_emb,
                ts_secs,
                importance, half_life, valence, stored_meta, namespace,
                certainty, domain, source, emotional_state,
                created_at_unix_micros, embedding_model,
                embedding_generation,
            ],
        )?;
        let was_new_row = inserted_row == 1;
        debug_assert!(
            inserted || !was_new_row,
            "row {rid} is NEW but its (rid, seq {seq}) vector entry pre-exists — \
             a fresh row cannot have a prior published vector"
        );

        if was_new_row {
            // Auto-link only on first insert. Replay should not
            // re-bump session memory_count.
            if let Some(session_id) = &session_id {
                conn.execute(
                    "UPDATE memories SET session_id = ?1 WHERE rid = ?2",
                    params![session_id, rid],
                )?;
                conn.execute(
                    "UPDATE sessions SET memory_count = memory_count + 1 WHERE session_id = ?1",
                    params![session_id],
                )?;
            }

            // Kill boundary (4a.6d-3): pre-port the process could die between
            // the RELEASEd row and the op's autocommit — the unrepairable
            // orphan. Inside the savepoint, dying here rolls back BOTH.
            crate::testing::fail_point("record_with_rid.between_row_and_oplog");

            // The op commits WITH the row — applied=1, generation pinned to
            // the same snapshot the reserved delta entry was written against.
            // Payload unchanged (peers' record_with_rid materialization
            // contract).
            self.log_op_in_tx(
                &conn,
                "record_with_rid",
                Some(rid),
                &serde_json::json!({
                    "rid": rid,
                    "type": memory_type,
                    "text": text,
                    "importance": importance,
                    "valence": valence,
                    "half_life": half_life,
                    "metadata": metadata,
                    "created_at_unix_micros": created_at_unix_micros,
                    "namespace": namespace,
                    "certainty": certainty,
                    "domain": domain,
                    "source": source,
                    "emotional_state": emotional_state,
                    "embedding_model": embedding_model,
                    "extracted_entities": extracted_entities,
                }),
                Some(&emb_hash),
                None,
                embedding_generation,
                None,
            )?;

            // **Phase 4.3 Commit C**, moved IN-TX (4a.6a's record() fix,
            // ported): the entity-materialization enqueue commits with the
            // row, so a crash can no longer keep the row while losing the
            // enqueue. Gated on was_new_row — the pre-port unconditional
            // re-enqueue on replay was repair for exactly the crash window
            // this savepoint closes; with atomicity it is pure duplicate
            // work inflating the pending queue on every re-delivery.
            if !extracted_entities.is_empty() {
                // Pending-queue admission BEFORE the enqueue, inside the tx:
                // an Err here rolls the whole write back. Pre-port this check
                // lived inside the post-commit log_op_pending — it reported
                // Err for a write that was already durable AND visible, and
                // the retry could never log the skipped op.
                self.check_pending_backpressure_locked()?;
                let post_payload = serde_json::json!({
                    "rid": rid,
                    "namespace": namespace,
                    "ts_secs": ts_secs,
                    "extracted_entities": extracted_entities.to_vec(),
                    "was_new_row": true,
                });
                self.log_op_pending_in_tx(
                    &conn,
                    crate::engine::op_types::OP_MATERIALIZE_RECORD_WITH_RID_POST,
                    Some(rid),
                    &post_payload,
                    None,
                    None,
                )?;
                // The tx now holds a pending row: the guard owes its count
                // post-commit (log_op_pending_in_tx deliberately never
                // touches the counter — counting inside the tx would strand
                // the increment on rollback, the v0.7.1 drift).
                if let Some(r) = reservation.as_mut() {
                    r.count_pending_op_on_completion(&self.pending_op_count);
                }
            }
        }

        savepoint.release()?;
        // The obligation inverts HERE: the write is durable, so the
        // reservation owes publish (+count if it enqueued), not removal.
        // Nothing fallible may sit between the RELEASE and this call.
        if let Some(r) = reservation.as_mut() {
            r.mark_committed();
        }
        let published = match reservation.as_mut() {
            Some(r) => r.complete(),
            // Replay whose vector already exists: nothing was reserved,
            // nothing publishes — the existing entry is the truth.
            None => false,
        };
        debug_assert!(
            published || !inserted,
            "record_with_rid reservation for {rid} seq {seq} vanished before publish"
        );
        if inserted && !published {
            tracing::error!(
                rid = %rid,
                seq,
                "reserved vector entry missing at publish — row is durable but \
                 unsearchable until the index is rebuilt from SQL"
            );
        }
        drop(reservation);

        // LAST: a read-your-write waiter must not wake against a
        // half-applied record (CONCURRENCY.md: bump visible_seq AFTER the
        // delta publish). Idempotent fetch_max, so the no-reservation replay
        // arm bumps harmlessly.
        self.bump_visible_seq(namespace, seq);

        // Scoring cache (engine-internal; replay safe since insert is
        // overwrite-on-rid).
        if was_new_row {
            self.cache_insert(
                rid.to_string(),
                ScoringRow {
                    created_at: ts_secs,
                    importance,
                    half_life,
                    last_access: ts_secs,
                    access_count: 0,
                    valence,
                    consolidation_status: "active".to_string(),
                    memory_type: memory_type.to_string(),
                    namespace: namespace.to_string(),
                    certainty,
                    domain: domain.to_string(),
                    source: source.to_string(),
                    emotional_state: emotional_state.map(|s| s.to_string()),
                },
            );
        }

        // 4a.6b: an ORIGIN write that was warn-flagged and actually WROTE A ROW is
        // durable — count it now. Gated on `was_new_row` (sol r3 finding 2): this
        // path is `INSERT OR IGNORE`, so a replay of an existing rid persists
        // nothing, and ticking there would inflate the warn→enforce nudge metric
        // with no-op replays. ADMITTED writes carry Clean and tick nothing.
        if was_new_row {
            self.note_flagged_write_committed(gate_verdict);
        }
        drop(conn);
        Ok(())
    }

    /// **Issue #41 layer 3 — queued write path.** Called from `record()`
    /// when `write_router.try_enter_sync_writer()` returned None
    /// (router is in `Queueing` state during reembed cutover). The op
    /// is logged to `oplog` with `applied=0` and the v27 columns
    /// (`embedding_model = current_runtime_embedder_name`,
    /// `applied_generation = NULL`). The post-swap materializer drains
    /// these ops, re-encodes the text under the new embedder, and
    /// applies to the new generation's memories table + HNSW.
    ///
    /// Important invariants from brainstorm-2/3 enforced here:
    /// - DO NOT write to `memories` table (would mix old+new dim under
    ///   the rebuild snapshot)
    /// - DO NOT call `vec_index.append` (same reason)
    /// - DO NOT bump `visible_seq` (active generation doesn't yet
    ///   cover this seq; the post-swap materializer bumps it after
    ///   applying)
    /// - DO assign a `vec_seq` for the caller's RYW use
    ///   (`recall_with_seq(min_seq=N)` waits for the new generation to
    ///   advance past N)
    ///
    /// The pre-computed `embedding` argument is intentionally
    /// IGNORED. Per brainstorm-3 invariant 8 (queued payload
    /// correctness), the oplog stores logical text and the materializer
    /// re-encodes under the NEW embedder at replay time. Storing a
    /// pre-encoded old-embedder vector in oplog would race against
    /// post-swap replay and produce dim mismatch when the new HNSW is
    /// at a different dim.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_queued(
        &self,
        text: &str,
        memory_type: &str,
        importance: f64,
        raw_importance: f64,
        valence: f64,
        half_life: f64,
        metadata: &serde_json::Value,
        _embedding: &[f32],
        namespace: &str,
        certainty: f64,
        domain: &str,
        source: &str,
        emotional_state: Option<&str>,
        gate_verdict: crate::provenance::GateVerdict,
        idem: Option<(&str, [u8; 32])>,
    ) -> Result<String> {
        let rid = crate::id::new_id();
        let ts = now();

        // Capture the current runtime embedder name (the one active
        // before reembed flipped the router). The post-swap materializer
        // uses this to discriminate ops queued under the old embedder
        // (need re-encode) from ops produced by the new generation's
        // own writers (apply embedding bytes directly).
        let current_embedder_name = self.search_state.load().runtime_embedder_name.clone();

        // Full record payload — what the materializer needs to
        // reconstruct the row.
        let payload = serde_json::json!({
            "rid": rid,
            "type": memory_type,
            "text": text,
            "importance": importance,
            "valence": valence,
            "half_life": half_life,
            "metadata": metadata,
            "created_at": ts,
            "updated_at": ts,
            "namespace": namespace,
            "certainty": certainty,
            "domain": domain,
            "source": source,
            "emotional_state": emotional_state,
            // 4a.6c: carried so the materializer writes the same v37 columns
            // the sync route writes — the memories partial unique index is the
            // claims table's defense-in-depth mirror, and a queued keyed write
            // must not materialize with a NULL key while its claim exists.
            // null for keyless writes; pre-4a.6c ops lack the fields and the
            // materializer defaults both to NULL, so old rows are unchanged.
            "idempotency_key": idem.as_ref().map(|(k, _)| *k),
            "origin_actor": idem.as_ref().map(|_| self.actor_id.as_str()),
        });

        // Write to oplog with applied=0. The v27 `embedding_model`
        // column carries the OLD embedder name so the post-swap
        // materializer knows this needs re-encoding (vs being a
        // legacy pre-v27 op where embedding_model IS NULL and the
        // materializer trusts the embedding bytes as-is).
        // 4a.6c: the claim rides the pending-op transaction — the op IS the
        // queued write's only durable record, so claim + op commit atomically.
        // The helper mints the op id and assembles the full claim row so the
        // two agree by construction.
        let generation = self.search_state.load().generation as i64;
        let pending_claim = idem
            .as_ref()
            .map(|(key, digest)| super::idempotency::PendingClaim {
                namespace,
                idempotency_key: key,
                payload_digest: digest,
                rid: &rid,
                generation,
            });
        if let Some(existing_rid) = self.log_op_pending_for_reembed_queue(
            "record",
            Some(&rid),
            &payload,
            current_embedder_name.as_deref(),
            // 4a.6b: the stats advance rides in the same transaction as the
            // pending op — the queued write's only durable record. RAW value:
            // the EWMA tracks writer intent, not the deflated output.
            Some((namespace, raw_importance)),
            pending_claim.as_ref(),
        )? {
            // Idempotent hit: the SAME payload is already durably enqueued (or
            // committed) under this key. Nothing was written — the helper
            // resolved before its INSERT — and both the seq mint and the flag
            // tick below are skipped: a retry that landed nothing is not a
            // flagged write and does not advance sequencing (sol 4a.6c r2
            // finding 2).
            return Ok(existing_rid);
        }

        // Assign a seq for caller's RYW use, only now that the write really
        // enqueued. Note we do NOT bump visible_seq — the active generation
        // doesn't yet cover this op; the post-swap materializer is responsible
        // for advancing visible_seq as it drains queued ops.
        let _seq = self
            .vec_seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;

        // 4a.6b: the pending op is durable, so a warn-mode flag counts now.
        self.note_flagged_write_committed(gate_verdict);

        Ok(rid)
    }

    /// **Issue #41 layer 3 — variant of `log_op_pending` that populates
    /// the v27 `oplog.embedding_model` column.** Used by the queued
    /// write path during reembed; lets the post-swap materializer
    /// discriminate queued-during-reembed ops (which need re-encoding
    /// under the new embedder) from legacy pre-v27 ops (which have
    /// NULL `embedding_model` and trust their stored embedding bytes).
    /// `stats_advance`: 4a.6b winner-only calibration for the queued path. The
    /// pending op IS the queued write's only durable record, so the namespace's
    /// importance distribution must advance atomically WITH it — `(namespace,
    /// raw_importance)` here, in the same transaction as the INSERT. `None` for
    /// op types that are not a record write (none today; the parameter exists so
    /// a future non-record caller cannot silently inherit a stats advance that
    /// does not belong to it).
    /// `claim` (4a.6c): a durable idempotency claim to commit atomically WITH
    /// the pending op — the op is the queued write's only durable record, so
    /// this transaction is the claim's only honest home. On a dup, returns
    /// `Ok(Some(existing_rid))` with the transaction aborted before any write.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn log_op_pending_for_reembed_queue(
        &self,
        op_type: &str,
        target_rid: Option<&str>,
        payload: &serde_json::Value,
        embedding_model: Option<&str>,
        stats_advance: Option<(&str, f64)>,
        claim: Option<&super::idempotency::PendingClaim<'_>>,
    ) -> Result<Option<String>> {
        use rusqlite::params;
        use std::sync::atomic::Ordering;

        let payload_str = serde_json::to_string(payload)?;

        // Advisory fast reject (unlocked); the AUTHORITATIVE check is under the
        // lock below. Same TOCTOU, same fix as log_op_pending (sol 4a.6a r2
        // finding 1): two queued writers at MAX_PENDING_OPS-1 could both pass an
        // unlocked load and then serialize their inserts past the ceiling.
        // KEYED writes skip it (sol 4a.6c r3): a race-window duplicate must
        // reach the locked probe below even when the pending queue is full —
        // see the sync route's twin comment.
        if claim.is_none() {
            self.check_pending_backpressure_fast()?;
        }

        let conn = self.conn.lock();
        // 4a.6c sol r2: locked probe BEFORE admission — same rationale as the
        // sync route's (see record_under_guard_and_state): a race-window
        // duplicate must resolve to its hit even when the queue is full, and
        // under this continuously-held guard nothing can commit a claim between
        // this read and our transaction. A hit exits before the backpressure
        // check and before the op id / HLC mint below, burning nothing.
        if let Some(pc) = claim {
            if let Some(existing_rid) = super::idempotency::probe_committed_claim(
                &conn,
                &self.actor_id,
                pc.namespace,
                pc.idempotency_key,
                pc.payload_digest,
            )? {
                return Ok(Some(existing_rid));
            }
        }
        self.check_pending_backpressure_locked()?;
        let op_id = crate::id::new_id();
        let hlc_ts = self.tick_hlc();
        let hlc_bytes = hlc_ts.to_bytes().to_vec();
        // ONE transaction: the pending op + the namespace stats advance. This
        // used to be a bare autocommit INSERT; wrapping it costs nothing on the
        // happy path and makes the stats advance winner-only — an INSERT failure
        // rolls the observation back with it.
        let tx = conn.unchecked_transaction()?;
        // 4a.6c: claim FIRST (cheap dup exit — the losing transaction has
        // written nothing when it aborts). stats_advance's namespace is the
        // claim's namespace: both come from the same normalized caller value.
        if let Some(pc) = claim {
            use super::idempotency::{claim_in_tx, ClaimAttempt, ClaimRow};
            match claim_in_tx(
                &tx,
                &ClaimRow {
                    origin_actor: &self.actor_id,
                    namespace: pc.namespace,
                    idempotency_key: pc.idempotency_key,
                    rid: pc.rid,
                    payload_digest: pc.payload_digest,
                    op_id: &op_id,
                    route: "queued",
                    generation: pc.generation,
                },
            )? {
                ClaimAttempt::Won => {}
                // tx drops un-committed: it has written nothing (the claim
                // lost its ON CONFLICT; the op INSERT comes after).
                ClaimAttempt::Hit { existing_rid } => return Ok(Some(existing_rid)),
            }
        }
        // Plain INSERT, not OR IGNORE. This is the QUEUED write's only durable
        // record — record_queued() writes no memories row (by design), so this op
        // IS the write. `OR IGNORE` here meant any swallowed constraint violation
        // made record_queued() return a rid and Ok while persisting NOTHING: the
        // caller's write vanished silently. The op_id is minted fresh above, so
        // no caller could ever have needed the ignore.
        tx.execute(
            "INSERT INTO oplog \
             (op_id, op_type, timestamp, target_rid, payload, \
              actor_id, hlc, embedding_hash, origin_actor, applied, \
              embedding, embedding_model, applied_generation) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 0, NULL, ?10, NULL)",
            params![
                op_id,
                op_type,
                now(),
                target_rid,
                payload_str,
                self.actor_id,
                hlc_bytes,
                None::<Vec<u8>>,
                self.actor_id,
                embedding_model,
            ],
        )?;
        if let Some((namespace, raw_importance)) = stats_advance {
            self.advance_importance_stats_in_tx(&tx, namespace, raw_importance)?;
        }
        tx.commit()?;
        // Only after commit: a plain INSERT inside a committed tx means exactly
        // one pending row landed. (Moving the increment before the commit would
        // leak it upward on rollback — the v0.7.1 counter-leak class.) Still a
        // claim about this statement, not durability — a caller wrapping its own
        // rolled-back SAVEPOINT around this (via the public `conn()`) leaves the
        // counter high. See the fuller note in `log_op_pending` (sol #83
        // finding 3).
        self.pending_op_count.fetch_add(1, Ordering::Relaxed);
        let _ = op_id; // the op is bound to the claim; callers get the hit signal
        Ok(None)
    }
}
