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
        self.check_pending_backpressure_fast()?;

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
        // a single durable byte has been written.
        state
            .vec_index
            .append_reserved(rid.clone(), embedding.to_vec(), seq)?;

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

        // Task 31 (Ingest Integrity): each input's importance is calibrated
        // against its namespace distribution INSIDE the savepoint below (4a.6b),
        // positionally aligned with `inputs` via push order. Computing +
        // advancing per item under the savepoint's transaction view preserves
        // the documented property that later items in the batch see the
        // running-mean effect of earlier ones in the same namespace — while a
        // ROLLBACK TO takes every advance with it, so a rejected batch leaves
        // every namespace's distribution untouched. (The old pre-routing
        // autocommit advanced ALL of them even when the batch then deferred on
        // the write-router or died mid-savepoint.)
        let mut calibrated_importances: Vec<f64> = Vec::with_capacity(inputs.len());

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
        let per_memory_linkage: Vec<(Vec<String>, std::collections::HashSet<String>)> =
            sanitized_texts
                .iter()
                .map(|text| {
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
                    (heuristic, candidates)
                })
                .collect();

        let mut rids = Vec::with_capacity(inputs.len());
        // One canonical "record" op per item, emitted after the savepoint
        // releases. See `log_record_ops_batch`: the old single "record_batch" op
        // was silently dropped by every peer, so batch writes never replicated.
        let mut record_op_entries: Vec<(String, serde_json::Value, Vec<u8>)> =
            Vec::with_capacity(inputs.len());

        // Lock conn once for the entire batch SQL work
        {
            let conn = self.conn();
            conn.execute_batch("SAVEPOINT batch_record")?;

            for (idx, input) in inputs.iter().enumerate() {
                let rid = crate::id::new_id();
                let ts = now();
                let emb_blob = serialize_f32(&input.embedding);
                let meta_str = serde_json::to_string(&input.metadata)?;

                // 4a.6b: calibrate + advance under the savepoint. The `_on`
                // variant reads through the HELD guard — calling the locking
                // wrapper here would re-lock `conn` on the same thread, the
                // `learn_category_members` deadlock (#83).
                //
                // 4a.6b (sol r2 finding 1): the stats ADVANCE is NOT done here.
                // The vector append happens after the savepoint RELEASE and can
                // still fail, and a committed EWMA blend is irreversible by the
                // compensating DELETE — so advancing inside the savepoint left a
                // rejected batch having permanently moved calibration. The
                // advance is deferred to the post-append-loop block below, run
                // only once the append wins. Consequence: every item calibrates
                // against the SAME pre-batch snapshot (no within-batch running
                // mean). That intra-batch progression was never pinned by a test
                // and is a defensible semantics change — a batch is one
                // simultaneous act — and it buys winner-only correctness. The
                // cross-batch running mean is unchanged (the deferred advances
                // move the table).
                let calibrated = super::importance::calibrated_importance_on(
                    &conn,
                    &input.namespace,
                    input.importance,
                )?;
                calibrated_importances.push(calibrated);

                // Encrypt fields if encryption is enabled. Task 29: store the
                // sanitized text (positionally aligned with `inputs`).
                let stored_text = self.encrypt_text(sanitized_texts[idx].as_ref())?;
                let stored_meta = self.encrypt_text(&meta_str)?;
                let stored_emb = self.encrypt_embedding(&emb_blob)?;

                // **Issue #41 brainstorm-4 §6.** v28 embedding_generation
                // stamped from the batch's snapshot.
                let embedding_generation: i64 = state.generation as i64;
                let result = conn.execute(
                    "INSERT INTO memories \
                     (rid, type, text, embedding, created_at, updated_at, importance, \
                      half_life, last_access, valence, metadata, namespace, \
                      certainty, domain, source, emotional_state, embedding_generation) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
                    params![rid, input.memory_type, stored_text, stored_emb, ts, ts,
                            calibrated_importances[idx], input.half_life, ts, input.valence, stored_meta,
                            input.namespace, input.certainty, input.domain, input.source,
                            input.emotional_state, embedding_generation],
                );

                if let Err(e) = result {
                    conn.execute_batch("ROLLBACK TO batch_record")?;
                    return Err(e.into());
                }

                // Byte-for-byte the payload `record()` emits (record.rs:325-340)
                // so peers materialize batch items through the existing, tested
                // "record" arm. Plaintext text/metadata, matching record() — the
                // encrypted forms above are the at-rest representation, not the
                // replication one.
                record_op_entries.push((
                    rid.clone(),
                    serde_json::json!({
                        "rid": rid,
                        "type": input.memory_type,
                        "text": sanitized_texts[idx].as_ref(),
                        "importance": calibrated_importances[idx],
                        "valence": input.valence,
                        "half_life": input.half_life,
                        "metadata": input.metadata,
                        "created_at": ts,
                        "updated_at": ts,
                        "namespace": input.namespace,
                        "certainty": input.certainty,
                        "domain": input.domain,
                        "source": input.source,
                        "emotional_state": input.emotional_state,
                    }),
                    embedding_hash(&input.embedding),
                ));

                rids.push(rid);
            }

            // Auto-link batch to active sessions
            for (rid, input) in rids.iter().zip(inputs.iter()) {
                if let Some(session_id) = sessions.get(&input.namespace) {
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
            let batch_ts = now();
            for (rid, (heuristic, candidates)) in rids.iter().zip(per_memory_linkage.iter()) {
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

            conn.execute_batch("RELEASE batch_record")?;
        }
        // NOTE: warn-mode flag ticks are DEFERRED to after the vector-append
        // loop below (4a.6b finding 1). The append is a post-RELEASE failure
        // point whose compensation deletes the rows — so ticking here would
        // count writes the caller then sees fail. See the tick loop past the
        // append.
        // conn dropped; now update graph_index in-memory.
        {
            let mut gi = self.graph_index.write();
            for (rid, (_, candidates)) in rids.iter().zip(per_memory_linkage.iter()) {
                for entity in candidates {
                    let entity_type = crate::graph::classify_entity_type(entity);
                    gi.add_entity(entity, entity_type);
                    gi.link_memory(rid, entity);
                }
            }
        }

        // RFC 006 Phase 0: emit one audit event per memory in the batch.
        for (idx, (rid, (input, (heuristic_entities, candidates)))) in rids
            .iter()
            .zip(inputs.iter().zip(per_memory_linkage.iter()))
            .enumerate()
        {
            let heuristic_vec: Vec<String> = heuristic_entities.iter().cloned().collect();
            let features =
                crate::graph::analyze_text_features(sanitized_texts[idx].as_ref(), &heuristic_vec);
            tracing::info!(
                target: "yantrikdb::audit::extraction",
                namespace = %input.namespace,
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

        // Append to vec_index (DeltaIndex) after SQL commit.
        // **v0.7.19 orphan-on-Backpressure fix.** If any append in
        // the batch fails (delta saturation, dim mismatch), the
        // SAVEPOINT above has already committed all N memories
        // rows. Compensating DELETE clears the entire batch so the
        // caller sees an atomic batch-fail outcome rather than
        // partial-commit state. See record() for the rationale on
        // single-row writes.
        for (idx, (rid, input)) in rids.iter().zip(inputs.iter()).enumerate() {
            let seq = self
                .vec_seq
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            if let Err(e) = state
                .vec_index
                .append(rid.clone(), input.embedding.clone(), seq)
            {
                // Roll back all N rows from memories (DELETE is fast
                // under a single conn lock; idempotent via WHERE).
                let conn = self.conn();
                for r in &rids {
                    let _ = conn.execute("DELETE FROM memories WHERE rid = ?1", params![r]);
                }
                // **v0.7.23 residual fix.** Reverse the per-session
                // `memory_count` bumps committed in the SAVEPOINT above,
                // once per memory that was session-linked — mirrors the
                // increment loop exactly so the stat matches the rows the
                // compensating DELETE just removed.
                for input in inputs.iter() {
                    if let Some(session_id) = sessions.get(&input.namespace) {
                        let _ = conn.execute(
                            "UPDATE sessions SET memory_count = memory_count - 1 WHERE session_id = ?1",
                            params![session_id],
                        );
                    }
                }
                let _ = idx; // index of the failing entry, kept for future logging
                return Err(e);
            }
            self.bump_visible_seq(&input.namespace, seq);
        }
        // 4a.6b (sol r2 finding 1): every append landed, so the batch is durable
        // AND visible — the winner is decided. Advance the calibration stats and
        // count the warn-mode flags NOW, both winner-only. Ticking or advancing
        // before this point would have moved state for a batch whose append then
        // failed and whose rows were compensated away — and a committed EWMA
        // blend cannot be un-done by the compensating DELETE.
        //
        // Grouped in one transaction so the N advances are all-or-nothing. Each
        // is a plain upsert (SQL blend against the stored value; composes with
        // concurrent writers), fed the RAW importance.
        //
        // BEST-EFFORT, deliberately NOT `?` (sol 4a.6b r3 finding 1): the rows
        // and vectors are already durable and visible — the batch WON. Calibration
        // is an approximate ranking prior, so failing to advance it is a benign
        // skipped observation. Propagating a SQLITE_FULL/IO error from this tx
        // would report Err for an already-committed batch, and a retrying caller
        // would then write DUPLICATE records under fresh rids — turning a missed
        // stat into data corruption. So it logs and continues.
        //
        // NB (sol r4 / #94): `log_record_ops_batch(...)?` below still `?`-returns
        // after this winner point — a PRE-EXISTING Err-after-commit (#79-era)
        // that is NOT best-effortable (the ops are what replicate the batch). Its
        // correct fix is to move that append inside the savepoint above (needs a
        // held-conn variant to avoid the #83 re-lock); tracked in #94, out of
        // 4a.6b's scope. Calibration is best-effort ONLY because it is
        // approximate — do not copy this pattern to the oplog append.
        {
            let conn = self.conn();
            let advanced = (|| -> Result<()> {
                let tx = conn.unchecked_transaction()?;
                for input in inputs.iter() {
                    self.advance_importance_stats_in_tx(&tx, &input.namespace, input.importance)?;
                }
                tx.commit()?;
                Ok(())
            })();
            if let Err(e) = advanced {
                tracing::warn!(
                    error = %e,
                    "record_batch: post-commit calibration advance failed; \
                     batch is durable, calibration observation skipped"
                );
            }
        }
        for verdict in gate_verdicts {
            self.note_flagged_write_committed(verdict);
        }
        // vec_index dropped, now scoring_cache
        {
            let mut cache = self.scoring_cache.write();
            for (idx, (rid, input)) in rids.iter().zip(inputs.iter()).enumerate() {
                let ts = now();
                cache.insert(
                    rid.clone(),
                    ScoringRow {
                        created_at: ts,
                        importance: calibrated_importances[idx],
                        half_life: input.half_life,
                        last_access: ts,
                        access_count: 0,
                        valence: input.valence,
                        consolidation_status: "active".to_string(),
                        memory_type: input.memory_type.clone(),
                        namespace: input.namespace.clone(),
                        certainty: input.certainty,
                        domain: input.domain.clone(),
                        source: input.source.clone(),
                        emotional_state: input.emotional_state.clone(),
                    },
                );
            }
        }

        // One canonical "record" op per item, under a single conn lock.
        //
        // This REPLACES a single `log_op("record_batch", {count, rids})`. That op
        // was unreplicable twice over: replication has no "record_batch" arm, so
        // peers hit the `_ =>` forward-compat catch-all and silently dropped it;
        // and the payload carried no text/embedding/scalars, so it could not have
        // rebuilt the memories even with an arm. Batch-written memories simply
        // never reached peers. Nothing consumes the old op type locally.
        self.log_record_ops_batch(&record_op_entries)?;

        Ok(rids)
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

        // Single conn block: INSERT OR IGNORE on memories (idempotent on rid),
        // session links, entity persistence. SAVEPOINT for atomicity within
        // the call.
        let was_new_row: bool = {
            let conn = self.conn();
            conn.execute_batch("SAVEPOINT record_with_rid")?;

            let result: Result<bool> = (|| {
                // **Issue #41 brainstorm-4 §6.** v28 embedding_generation
                // stamp from the SearchState snapshot loaded above.
                let embedding_generation: i64 = state.generation as i64;
                let inserted = conn.execute(
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
                let was_new_row = inserted == 1;

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
                }

                // **Phase 4.3 Commit C (saga task 19, 2026-05-08).** The
                // entity / memory_entities INSERT loop was previously here
                // inside the SAVEPOINT, holding `db.conn().lock()` for
                // O(extracted_entities.len()) statements. Now enqueued as
                // OP_MATERIALIZE_RECORD_WITH_RID_POST after the SAVEPOINT
                // releases. See docs/phase_4_3_design.md for the contract.

                Ok(was_new_row)
            })();

            match result {
                Ok(b) => {
                    conn.execute_batch("RELEASE record_with_rid")?;
                    b
                }
                Err(e) => {
                    let _ = conn.execute_batch("ROLLBACK TO record_with_rid");
                    let _ = conn.execute_batch("RELEASE record_with_rid");
                    return Err(e);
                }
            }
        };
        // conn dropped

        // DeltaIndex append. The seq is either caller-supplied (cluster
        // mode: openraft commit-log index for byte-deterministic replay)
        // or engine-allocated (single-node). On idempotent replay the rid
        // is the same and the seq is identical (cluster) or fresh
        // (single-node retry); the compactor's highest-seq-wins rule
        // converges state identically on both paths.
        let seq = self.assign_seq(seq);
        // **v0.7.19 orphan-on-Backpressure fix.** The trader's
        // `trader_ledger` DB shows `record_with_rid` pinned at
        // exactly 256 (the v0.7.17 delta_max wedge ceiling) — every
        // additional call after that left a memories row from the
        // INSERT OR IGNORE above with no oplog provenance because
        // the vec_index.append Err short-circuited the log_op below.
        // Compensating DELETE on failure. Skip the delete when
        // was_new_row=false (replay path: the row pre-existed; we
        // shouldn't yank it).
        if let Err(e) = state
            .vec_index
            .append(rid.to_string(), embedding.to_vec(), seq)
        {
            if was_new_row {
                let conn = self.conn();
                let _ = conn.execute("DELETE FROM memories WHERE rid = ?1", params![rid]);
                // **v0.7.23 residual fix.** The session `memory_count`
                // bumped inside the (already-RELEASEd) SAVEPOINT survives
                // the compensating DELETE. Reverse it so the stat matches
                // the surviving rows. Mirrors the was_new_row guard on the
                // original bump — replay (was_new_row=false) never bumped.
                if let Some(session_id) = &session_id {
                    let _ = conn.execute(
                        "UPDATE sessions SET memory_count = memory_count - 1 WHERE session_id = ?1",
                        params![session_id],
                    );
                }
            }
            return Err(e);
        }
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

        // Op log entry — applied=1 since leader has materialized inline.
        // Followers will receive a separate replicated entry via the
        // cluster sync path; this path never logs applied=0.
        //
        // Logged BEFORE the post-record materialization enqueue so
        // extract_ops_since reports the user-data op in causal order
        // (record_with_rid arrived, then its entity-link materialization
        // was queued).
        let emb_hash = embedding_hash(embedding);
        if was_new_row {
            self.log_op(
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
            )?;
        }

        // **Phase 4.3 Commit C (saga task 19, 2026-05-08).** Enqueue the
        // entity / memory_entities / graph_index materialization for the
        // worker thread. Skip when there are no entities to apply — the
        // dispatch arm short-circuits the same way, but skipping avoids
        // a wasteful oplog row in the common no-entity case.
        //
        // Cluster determinism: the leader and each follower will both
        // enqueue + apply this op against their local state. Convergence
        // on entities + memory_entities is guaranteed by the same
        // INSERT OR IGNORE / ON CONFLICT idempotency the inline path
        // had. The convergence *time* differs by the materializer-lag
        // window (ms-scale), but the converged final state is identical.
        if !extracted_entities.is_empty() {
            let entities_json: Vec<&str> = extracted_entities.to_vec();
            let post_payload = serde_json::json!({
                "rid": rid,
                "namespace": namespace,
                "ts_secs": ts_secs,
                "extracted_entities": entities_json,
                "was_new_row": was_new_row,
            });
            self.log_op_pending(
                crate::engine::op_types::OP_MATERIALIZE_RECORD_WITH_RID_POST,
                Some(rid),
                &post_payload,
                None,
                None,
            )?;
        }

        // 4a.6b: an ORIGIN write that was warn-flagged and actually WROTE A ROW is
        // durable — count it now. Gated on `was_new_row` (sol r3 finding 2): this
        // path is `INSERT OR IGNORE`, so a replay of an existing rid persists
        // nothing, and ticking there would inflate the warn→enforce nudge metric
        // with no-op replays. ADMITTED writes carry Clean and tick nothing.
        if was_new_row {
            self.note_flagged_write_committed(gate_verdict);
        }
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
        self.check_pending_backpressure_fast()?;

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
