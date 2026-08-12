use super::*;

/// 4a.6c (T07 "repetition is not corroboration"), sync route: an exact retry
/// under the same idempotency key returns the ORIGINAL rid and writes NOTHING —
/// no second row, no second op, no pending-op count, no calibration advance.
#[test]
fn idempotent_retry_returns_original_rid_and_writes_nothing() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let write = |db: &YantrikDB| {
        db.record_with_idempotency(
            "the deploy uses zorbium for caching",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &serde_json::json!({"k": "v"}),
            &vec_seed(1.0, 8),
            "idem_ns",
            0.8,
            "work",
            "user",
            None,
            Some("client-req-001"),
            None,
        )
    };
    let rid1 = write(&db).unwrap();

    let rows = |db: &YantrikDB| -> i64 {
        db.conn()
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE namespace = 'idem_ns'",
                [],
                |r| r.get(0),
            )
            .unwrap()
    };
    let ops = |db: &YantrikDB| -> i64 {
        db.conn()
            .query_row("SELECT COUNT(*) FROM oplog", [], |r| r.get(0))
            .unwrap()
    };
    let stats_count = |db: &YantrikDB| -> i64 {
        db.conn()
            .query_row(
                "SELECT count FROM namespace_importance_stats WHERE namespace = 'idem_ns'",
                [],
                |r| r.get(0),
            )
            .unwrap()
    };
    let (r1, o1, s1, p1) = (
        rows(&db),
        ops(&db),
        stats_count(&db),
        db.count_pending_ops().unwrap(),
    );
    assert_eq!(r1, 1, "first keyed write lands one row");
    assert_eq!(s1, 1, "first keyed write advances stats once");

    // The exact retry.
    let rid2 = write(&db).unwrap();
    assert_eq!(rid2, rid1, "retry returns the ORIGINAL rid");
    assert_eq!(rows(&db), r1, "retry wrote no second row");
    assert_eq!(ops(&db), o1, "retry wrote no oplog op of any kind");
    assert_eq!(stats_count(&db), s1, "retry advanced no calibration stats");
    assert_eq!(
        db.count_pending_ops().unwrap(),
        p1,
        "retry enqueued no pending op"
    );
    let claims: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM idempotency_claims", [], |r| r.get(0))
        .unwrap();
    assert_eq!(claims, 1, "one claim row, state committed");

    // Keyless behavior is untouched: the same content twice makes two rows.
    for _ in 0..2 {
        db.record(
            "keyless duplicate content",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(2.0, 8),
            "idem_ns",
            0.8,
            "work",
            "user",
            None,
        )
        .unwrap();
    }
    assert_eq!(rows(&db), r1 + 2, "keyless writes still append");
}

/// 4a.6c: the same key with a DIFFERENT payload is a typed conflict carrying
/// the existing rid — never a silent merge, and it writes nothing.
#[test]
fn idempotency_conflict_on_different_payload() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let write = |db: &YantrikDB, importance: f64| {
        db.record_with_idempotency(
            "conflicting payload probe",
            "semantic",
            importance,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "conf_ns",
            0.8,
            "work",
            "user",
            None,
            Some("key-x"),
            None,
        )
    };
    let rid1 = write(&db, 0.7).unwrap();

    // Same key, different importance — the mutable scalars are IN the digest
    // (design doc: they alter the first write's observable state).
    let err = write(&db, 0.9).expect_err("scalar diff under the same key must conflict");
    match &err {
        crate::error::YantrikDbError::IdempotencyConflict { existing_rid, .. } => {
            assert_eq!(existing_rid, &rid1, "conflict names the existing record");
        }
        other => panic!("expected IdempotencyConflict, got {other:?}"),
    }
    let rows: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE namespace = 'conf_ns'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(rows, 1, "the conflicting write persisted nothing");
    let stats: i64 = db
        .conn()
        .query_row(
            "SELECT count FROM namespace_importance_stats WHERE namespace = 'conf_ns'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(stats, 1, "the conflicting write advanced no stats");
}

/// 4a.6c, queued route: the claim rides the pending-op transaction. A retry
/// during reembed cutover enqueues exactly one op; cross-route retries (sync
/// then queued) also resolve to the same rid — the claim is route-agnostic.
#[test]
fn idempotency_holds_on_and_across_the_queued_route() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let write = |db: &YantrikDB, key: &str| {
        db.record_with_idempotency(
            "queued idempotent probe",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "q_ns",
            0.8,
            "work",
            "user",
            None,
            Some(key),
            None,
        )
    };

    // Queued-only: both attempts under Queueing.
    db.write_router.switch_to_queueing();
    let rid_q = write(&db, "q-key").unwrap();
    let pending_after_first = db.count_pending_ops().unwrap();
    let rid_q2 = write(&db, "q-key").unwrap();
    assert_eq!(rid_q2, rid_q, "queued retry returns the original rid");
    assert_eq!(
        db.count_pending_ops().unwrap(),
        pending_after_first,
        "queued retry enqueued no second op"
    );
    db.write_router.switch_to_normal();

    // Cross-route: first write lands sync, retry arrives during Queueing.
    let rid_s = write(&db, "cross-key").unwrap();
    db.write_router.switch_to_queueing();
    let pending_before = db.count_pending_ops().unwrap();
    let rid_s2 = write(&db, "cross-key").unwrap();
    assert_eq!(rid_s2, rid_s, "cross-route retry resolves to the sync rid");
    assert_eq!(
        db.count_pending_ops().unwrap(),
        pending_before,
        "cross-route retry enqueued nothing"
    );
    db.write_router.switch_to_normal();
}

/// 4a.6c: the digest is computed from the RAW caller importance, never the
/// calibrated value. Discriminating setup: in a saturated namespace the EWMA
/// deepens with every write, so the CALIBRATED value of an identical retry
/// differs from the first attempt's — a digest over calibrated output would
/// turn the honest retry into a false conflict.
#[test]
fn idempotency_digest_uses_raw_importance_not_calibrated() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // Saturate: 8 keyless writes at 1.0 puts the namespace at MIN_COUNT with
    // ewma 1.0, so calibration engages (and keeps deepening) from write 9 on.
    for i in 0..8 {
        db.record(
            &format!("sat-{i}"),
            "semantic",
            1.0,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.0 + i as f32 * 0.01, 8),
            "sat_ns",
            0.8,
            "work",
            "user",
            None,
        )
        .unwrap();
    }
    // raw = 0.9, NOT 1.0, and this is load-bearing for the discrimination:
    // with all-1.0 saturation the EWMA sits pinned at exactly 1.0, so an
    // identical retry calibrates to the SAME deflated value and a digest over
    // calibrated output would accidentally still hit. 0.9 moves the EWMA on
    // the first attempt's own advance (1.0 -> 0.985), so the retry's
    // calibrated value differs (0.7333 -> 0.7458) — a calibrated digest now
    // MUST conflict, and only a raw digest hits. (The first draft used 1.0
    // and survived exactly that mutation.)
    let write = |db: &YantrikDB| {
        db.record_with_idempotency(
            "saturated keyed write",
            "semantic",
            0.9, // raw — deflated by calibration when stored
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(2.0, 8),
            "sat_ns",
            0.8,
            "work",
            "user",
            None,
            Some("sat-key"),
            None,
        )
    };
    let rid1 = write(&db).unwrap();
    let stored: f64 = db
        .conn()
        .query_row(
            "SELECT importance FROM memories WHERE rid = ?1",
            rusqlite::params![rid1],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        stored < 0.9,
        "precondition: calibration deflated the stored importance, got {stored}"
    );
    // The honest retry: same RAW payload. Stats advanced since attempt 1, so
    // the calibrated value now differs — a calibrated-digest would conflict.
    let rid2 = write(&db).expect("identical RAW retry must be a hit, not a conflict");
    assert_eq!(rid2, rid1);
}

/// 4a.6c: empty / oversized keys are refused loudly — silently treating "" as
/// "no key" would leave the caller believing they have dedup they don't have.
#[test]
fn invalid_idempotency_keys_are_rejected() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    for bad in ["", "   ", &"k".repeat(513)] {
        let err = db
            .record_with_idempotency(
                "probe",
                "semantic",
                0.7,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(1.0, 8),
                "default",
                0.8,
                "work",
                "user",
                None,
                Some(bad),
                None,
            )
            .expect_err("bad key must be refused");
        assert!(
            matches!(
                err,
                crate::error::YantrikDbError::InvalidIdempotencyKey { .. }
            ),
            "expected InvalidIdempotencyKey, got {err:?}"
        );
    }
    let rows: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    assert_eq!(rows, 0, "refused keys wrote nothing");
}

/// 4a.6c sol finding 1: a keyed DUPLICATE retry must resolve to its hit even
/// when the engine is saturated — Backpressure storms are exactly when clients
/// retry, and the dup writes nothing, so admission machinery (router, delta
/// reservation, backpressure checks, seq/HLC) must not run before the claim
/// probe. Pre-probe, this test dies with Backpressure instead of the Hit.
#[cfg(feature = "bundled-embedder")]
#[test]
fn keyed_duplicate_resolves_even_under_backpressure() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let dim = db.embedding_dim();

    // The keyed write lands FIRST, while there is capacity.
    let keyed = |db: &YantrikDB| {
        db.record_with_idempotency(
            "the keyed write that must stay resolvable",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(9.0, 8),
            "bp_idem_ns",
            0.8,
            "work",
            "user",
            None,
            Some("bp-key"),
            None,
        )
    };
    let rid = keyed(&db).unwrap();

    // Saturate the delta with keyless writes until Backpressure.
    let mut saturated = false;
    for i in 0..400 {
        let emb: Vec<f32> = (0..dim).map(|j| ((i + j) as f32) * 0.001).collect();
        match db.record(
            &format!("bp-filler-{i}"),
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &emb,
            "default",
            0.8,
            "general",
            "user",
            None,
        ) {
            Ok(_) => {}
            Err(crate::error::YantrikDbError::Backpressure { .. }) => {
                saturated = true;
                break;
            }
            Err(e) => panic!("unexpected: {e:?}"),
        }
    }
    assert!(saturated, "delta never saturated");

    // The DUPLICATE retry resolves to its hit despite saturation.
    let rid2 = keyed(&db).expect(
        "a keyed duplicate writes nothing and must resolve to its Hit, \
         not die on Backpressure",
    );
    assert_eq!(rid2, rid, "the hit returns the original rid");

    // Sanity: a keyed NEW write (different key) is still subject to
    // backpressure — the probe only short-circuits duplicates.
    let err = db
        .record_with_idempotency(
            "a genuinely new keyed write",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(10.0, 8),
            "bp_idem_ns",
            0.8,
            "work",
            "user",
            None,
            Some("bp-key-new"),
            None,
        )
        .expect_err("a NEW keyed write under saturation must still backpressure");
    assert!(
        matches!(err, crate::error::YantrikDbError::Backpressure { .. }),
        "expected Backpressure for the new keyed write, got {err:?}"
    );

    // The QUEUED route under the same saturation: the dup must resolve there
    // too. (No new-key backpressure assert here, and its absence is itself a
    // finding from writing this test: the saturation above is DELTA capacity,
    // and the queued route consumes none — it writes no vector, which is
    // exactly why it is the escape valve during reembed cutover. A new keyed
    // queued write legitimately succeeds under delta saturation; only a full
    // PENDING queue gates it, and the helper's locked probe precedes that
    // check by the same one-owner code path the sync mutation proof covers.)
    db.write_router.switch_to_queueing();
    let rid3 = keyed(&db).expect("queued dup under saturation must resolve");
    assert_eq!(rid3, rid, "queued dup returns the original rid");
    db.write_router.switch_to_normal();
}

/// 4a.6c sol r3: the FAST (pre-lock) backpressure check ran before the locked
/// probe, so a race-window duplicate under PENDING-queue saturation (a
/// different resource from the delta saturation the sibling test exercises)
/// still died with Backpressure before it could resolve. Keyed writes now skip
/// the fast check — the authoritative locked check still gates keyed WINNERS,
/// proven by the second half.
#[test]
fn keyed_duplicate_resolves_under_pending_queue_saturation() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let write = |db: &YantrikDB, key: &str| {
        db.record_with_idempotency(
            "pending-saturation probe",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "pq_ns",
            0.8,
            "work",
            "user",
            None,
            Some(key),
            None,
        )
    };
    // The keyed write lands while there is capacity.
    let rid = write(&db, "pq-key").unwrap();

    // Force the PENDING queue to read as saturated — the exact resource in
    // sol's scenario. The cached counter is what both backpressure checks
    // consult, so storing past the ceiling makes every admission check reject
    // deterministically, with no timing dependence.
    let real = db
        .pending_op_count
        .swap(1_000_000, std::sync::atomic::Ordering::SeqCst);

    // The duplicate must resolve despite full-pending admission rejecting
    // everything: keyed writes skip the fast check and the locked probe runs
    // before the locked check.
    let rid2 = write(&db, "pq-key").expect("keyed dup must resolve under pending-queue saturation");
    assert_eq!(rid2, rid, "dup returns the original rid");

    // A keyed NEW write must still hear Backpressure — from the AUTHORITATIVE
    // locked check, which keyed writes do not skip.
    let err = write(&db, "pq-key-new")
        .expect_err("a NEW keyed write under pending saturation must backpressure");
    assert!(
        matches!(err, crate::error::YantrikDbError::Backpressure { .. }),
        "expected Backpressure from the locked check, got {err:?}"
    );

    // And the queued route, same resource: dup resolves, new key rejects.
    db.write_router.switch_to_queueing();
    let rid3 = write(&db, "pq-key").expect("queued dup must resolve under pending saturation");
    assert_eq!(rid3, rid);
    let err = write(&db, "pq-key-new-q")
        .expect_err("a NEW keyed queued write under pending saturation must backpressure");
    assert!(
        matches!(err, crate::error::YantrikDbError::Backpressure { .. }),
        "expected Backpressure on the queued route, got {err:?}"
    );
    db.write_router.switch_to_normal();

    db.pending_op_count
        .store(real, std::sync::atomic::Ordering::SeqCst);
}

/// 4a.6d: record_text idempotency. The digest EXCLUDES the engine-generated
/// embedding, and the pre-admission probe runs BEFORE the embed — proven with a
/// counting embedder: the duplicate retry returns the original rid without
/// invoking the embedder at all. (Also the reason a drifting embedder can never
/// fake a conflict on this surface.)
#[test]
fn record_text_keyed_retry_hits_without_embedding_again() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct CountingEmbedder(Arc<AtomicUsize>);
    impl crate::types::Embedder for CountingEmbedder {
        fn embed(
            &self,
            t: &str,
        ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            let mut v = vec![0.1; 8];
            v[0] = (t.len() % 7) as f32 * 0.1;
            Ok(v)
        }
        fn dim(&self) -> usize {
            8
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let mut db = YantrikDB::new(":memory:", 8).unwrap();
    db.set_embedder(Box::new(CountingEmbedder(Arc::clone(&calls))))
        .unwrap();

    let write = |db: &YantrikDB| {
        db.record_text_with_idempotency(
            "keyed text write",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            "rt_ns",
            0.8,
            "work",
            "user",
            None,
            Some("rt-key"),
            None,
        )
    };
    let rid = write(&db).unwrap();
    let embeds_after_first = calls.load(Ordering::SeqCst);
    assert!(embeds_after_first >= 1, "first write embeds");

    let rid2 = write(&db).unwrap();
    assert_eq!(rid2, rid, "retry returns the original rid");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        embeds_after_first,
        "the duplicate retry must resolve at the probe, BEFORE the embed — \
         zero additional embedder invocations"
    );

    // And nothing was written by the retry.
    let rows: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE namespace = 'rt_ns'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(rows, 1, "one row total");
}

/// 4a.6d: the SAME key used across record() (embedding-inclusive digest) and
/// record_text() (embedding-exclusive digest) is a typed conflict — a
/// cross-surface retry is not the same write.
#[test]
fn same_key_across_record_and_record_text_is_a_conflict() {
    struct Fake;
    impl crate::types::Embedder for Fake {
        fn embed(
            &self,
            t: &str,
        ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
            let mut v = vec![0.1; 8];
            v[0] = (t.len() % 7) as f32 * 0.1;
            Ok(v)
        }
        fn dim(&self) -> usize {
            8
        }
    }
    let mut db = YantrikDB::new(":memory:", 8).unwrap();
    db.set_embedder(Box::new(Fake)).unwrap();

    // First write via record() with an explicit vector.
    db.record_with_idempotency(
        "cross surface text",
        "semantic",
        0.7,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(1.0, 8),
        "xs_ns",
        0.8,
        "work",
        "user",
        None,
        Some("xs-key"),
        None,
    )
    .unwrap();

    // Same key, same text, via record_text — different surface, different
    // digest variant: typed conflict, never a silent hit.
    let err = db
        .record_text_with_idempotency(
            "cross surface text",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            "xs_ns",
            0.8,
            "work",
            "user",
            None,
            Some("xs-key"),
            None,
        )
        .expect_err("cross-surface same-key must conflict");
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::IdempotencyConflict { .. }
        ),
        "expected IdempotencyConflict, got {err:?}"
    );
}

/// 4a.6d-1 sol finding: record_text never normalized blank namespaces — a
/// pre-existing divergence from record() (which coerces ""/whitespace to
/// "default" at entry) that surfaced when the Python wrapper's
/// embedding=None flow was rerouted through record_text. A blank-namespace
/// record_text write must land in "default" — rows, calibration stats, AND
/// the idempotency claim scope — or every reader querying "default" misses it.
#[test]
fn record_text_normalizes_blank_namespace_like_record() {
    struct Fake;
    impl crate::types::Embedder for Fake {
        fn embed(
            &self,
            t: &str,
        ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
            let mut v = vec![0.1; 8];
            v[0] = (t.len() % 7) as f32 * 0.1;
            Ok(v)
        }
        fn dim(&self) -> usize {
            8
        }
    }
    let mut db = YantrikDB::new(":memory:", 8).unwrap();
    db.set_embedder(Box::new(Fake)).unwrap();

    let rid = db
        .record_text_with_idempotency(
            "blank namespace probe",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            "   ",
            0.8,
            "work",
            "user",
            None,
            Some("ns-key"),
            None,
        )
        .unwrap();

    let ns: String = db
        .conn()
        .query_row(
            "SELECT namespace FROM memories WHERE rid = ?1",
            rusqlite::params![rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(ns, "default", "blank namespace must normalize to default");

    let claim_ns: String = db
        .conn()
        .query_row(
            "SELECT namespace FROM idempotency_claims WHERE idempotency_key = 'ns-key'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        claim_ns, "default",
        "the claim must scope under the NORMALIZED namespace"
    );

    // And the retry with the same blank namespace resolves (same scope).
    let rid2 = db
        .record_text_with_idempotency(
            "blank namespace probe",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            "",
            0.8,
            "work",
            "user",
            None,
            Some("ns-key"),
            None,
        )
        .unwrap();
    assert_eq!(
        rid2, rid,
        "'' and '   ' resolve to the same normalized scope"
    );
}

/// 4a.6d-2a (#92): a batch rejected for delta capacity must write NOTHING AT
/// ALL. The pre-restructure code committed the savepoint (rows + sessions +
/// entities + memory_entities), updated graph_index in-memory, and only THEN
/// appended vectors — so a capacity failure triggered a compensating DELETE
/// that reversed rows and session counts but left `entities`,
/// `memory_entities`, and the in-memory graph_index permanently inflated with
/// linkage for memories that do not exist. The fix reserves delta capacity for
/// ALL N items BEFORE the savepoint opens, so capacity failure surfaces before
/// a single durable byte — the same reserve→commit→publish protocol record()
/// uses (4a.6a), batched.
#[test]
fn batch_append_failure_writes_nothing_at_all() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let dim = db.embedding_dim();

    // CONTROL: prove this text shape actually drives entity extraction, so the
    // absence assertions below are meaningful rather than vacuously true. A
    // regression test whose branch never fires is decoration (#83 lesson).
    db.record_batch(&[RecordInput {
        created_at: None,
        idempotency_key: None,
        text: "Quarterly sync with Klaxonberg about the roadmap".into(),
        memory_type: "episodic".into(),
        importance: 0.6,
        valence: 0.0,
        half_life: 604800.0,
        metadata: serde_json::json!({}),
        embedding: vec_seed(1.5, 8),
        namespace: "ctrl_ns".into(),
        certainty: 0.8,
        domain: "work".into(),
        source: "user".into(),
        emotional_state: None,
    }])
    .unwrap();
    let ctrl_entities: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM entities WHERE name = 'Klaxonberg'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        ctrl_entities, 1,
        "control precondition: batch entity extraction must fire for this \
         text shape, or the absence assertions below prove nothing"
    );

    // Saturate the delta with single-record writes in a different namespace.
    let mut saturated = false;
    for i in 0..400 {
        let emb: Vec<f32> = (0..dim).map(|j| ((i + j) as f32) * 0.001).collect();
        match db.record(
            &format!("filler-{i}"),
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &emb,
            "filler_ns",
            0.8,
            "general",
            "user",
            None,
        ) {
            Ok(_) => {}
            Err(crate::error::YantrikDbError::Backpressure { .. }) => {
                saturated = true;
                break;
            }
            Err(e) => panic!("unexpected: {e:?}"),
        }
    }
    assert!(saturated, "delta never saturated");

    let ops_before: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM oplog", [], |r| r.get(0))
        .unwrap();

    // The doomed batch: same text shape, a DIFFERENT unique entity.
    let err = db
        .record_batch(&[RecordInput {
            created_at: None,
            idempotency_key: None,
            text: "Quarterly sync with Quorvexia about the roadmap".into(),
            memory_type: "episodic".into(),
            importance: 0.9,
            valence: 0.0,
            half_life: 604800.0,
            metadata: serde_json::json!({}),
            embedding: vec_seed(2.0, 8),
            namespace: "batch_ns".into(),
            certainty: 0.8,
            domain: "work".into(),
            source: "user".into(),
            emotional_state: None,
        }])
        .expect_err("batch into the saturated delta must fail");
    assert!(
        matches!(err, crate::error::YantrikDbError::Backpressure { .. }),
        "expected Backpressure, got {err:?}"
    );

    // NOTHING may exist: not rows, not entities, not memory_entities, not the
    // in-memory graph, not oplog entries. Pre-restructure, the first three SQL
    // assertions below held only for `memories` — entities/memory_entities
    // committed in the savepoint and the compensating DELETE never touched
    // them (#92).
    let rows: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE namespace = 'batch_ns'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(rows, 0, "no memory rows for the rejected batch");

    let orphan_entities: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM entities WHERE name = 'Quorvexia'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        orphan_entities, 0,
        "a rejected batch left an orphaned `entities` row (#92)"
    );

    let orphan_links: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memory_entities WHERE entity_name = 'Quorvexia'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        orphan_links, 0,
        "a rejected batch left orphaned `memory_entities` links (#92)"
    );

    assert!(
        !db.graph_index
            .read()
            .all_entity_names()
            .iter()
            .any(|n| n == "Quorvexia"),
        "a rejected batch polluted the in-memory graph_index (#92)"
    );

    let ops_after: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM oplog", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        ops_after, ops_before,
        "no oplog entries for a rejected batch"
    );

    // And the connection is back in autocommit — no savepoint left open (#91's
    // class: an error arm that unwinds the savepoint must RELEASE it too, or
    // every later write on this conn silently nests inside it).
    assert!(
        db.conn().is_autocommit(),
        "record_batch failure left a savepoint open on the shared conn"
    );
}

/// 4a.6d-2a (#98): record_batch never normalized blank namespaces, while the
/// calibration helpers it calls normalize INTERNALLY — so a blank-namespace
/// batch split across two partitions: the ROW landed under the raw "  " (a
/// partition no reader queries), while its importance observation advanced the
/// stats under "default". record() and (since #99) record_text both coerce at
/// entry; the batch must behave identically, and the replicated op payload
/// must carry the NORMALIZED namespace so peers land it in the same partition.
#[test]
fn record_batch_normalizes_blank_namespace_like_record() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    let mk = |text: &str, ns: &str, emb: f32| RecordInput {
        created_at: None,
        idempotency_key: None,
        text: text.into(),
        memory_type: "semantic".into(),
        importance: 0.7,
        valence: 0.0,
        half_life: 604800.0,
        metadata: serde_json::json!({}),
        embedding: vec_seed(emb, 8),
        namespace: ns.into(),
        certainty: 0.8,
        domain: "work".into(),
        source: "user".into(),
        emotional_state: None,
    };

    let rids = db
        .record_batch(&[
            mk("blank ns batch item one", "   ", 1.0),
            mk("blank ns batch item two", "", 2.0),
        ])
        .unwrap();
    assert_eq!(rids.len(), 2);

    let in_default: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE namespace = 'default'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        in_default, 2,
        "blank-namespace batch rows must land under 'default', like record()"
    );

    let in_blank: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE TRIM(namespace) = ''",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        in_blank, 0,
        "no row may persist under a raw blank namespace partition (#98)"
    );

    // The replicated op payload must carry the normalized namespace too —
    // otherwise every PEER materializes the row back into the unreachable
    // blank partition and the fix only holds locally.
    for rid in &rids {
        let payload: String = db
            .conn()
            .query_row(
                "SELECT payload FROM oplog WHERE target_rid = ?1 AND op_type = 'record'",
                rusqlite::params![rid],
                |r| r.get(0),
            )
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(
            v["namespace"], "default",
            "op payload must replicate the NORMALIZED namespace"
        );
    }

    // Stats already normalized internally (that was the divergence); pin the
    // now-consistent whole: one 'default' stats row observing both items.
    let count: i64 = db
        .conn()
        .query_row(
            "SELECT count FROM namespace_importance_stats WHERE namespace = 'default'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 2, "both observations under the normalized namespace");
}

/// 4a.6d-2b helper: a keyed batch input. Explicit embedding (batch items are
/// always caller-supplied vectors -> PayloadVariant::Record digest).
fn keyed_input(text: &str, ns: &str, emb: f32, key: Option<&str>) -> RecordInput {
    RecordInput {
        created_at: None,
        idempotency_key: key.map(|k| k.to_string()),
        text: text.into(),
        memory_type: "semantic".into(),
        importance: 0.7,
        valence: 0.0,
        half_life: 604800.0,
        metadata: serde_json::json!({}),
        embedding: vec_seed(emb, 8),
        namespace: ns.into(),
        certainty: 0.8,
        domain: "work".into(),
        source: "user".into(),
        emotional_state: None,
    }
}

/// 4a.6d-2b: retrying an identical keyed batch is a per-item idempotent HIT —
/// the same rids come back in the same order and the store is untouched:
/// no second rows, no stats advance, no second record ops. Repetition is not
/// corroboration (T07), now for batches.
#[test]
fn keyed_batch_retry_returns_original_rids_and_writes_nothing() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let batch = vec![
        keyed_input("kb item one", "kb_ns", 1.0, Some("kb-1")),
        keyed_input("kb item two", "kb_ns", 2.0, Some("kb-2")),
    ];

    let rids1 = db.record_batch(&batch).unwrap();
    assert_eq!(rids1.len(), 2);

    let count = |sql: &str| -> i64 { db.conn().query_row(sql, [], |r| r.get(0)).unwrap() };
    let rows_before = count("SELECT COUNT(*) FROM memories WHERE namespace = 'kb_ns'");
    let ops_before = count("SELECT COUNT(*) FROM oplog WHERE op_type = 'record'");
    let stats_before: i64 = db
        .conn()
        .query_row(
            "SELECT count FROM namespace_importance_stats WHERE namespace = 'kb_ns'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(rows_before, 2);
    assert_eq!(stats_before, 2);

    let rids2 = db.record_batch(&batch).unwrap();
    assert_eq!(rids2, rids1, "retry returns the ORIGINAL rids, in order");
    assert_eq!(
        count("SELECT COUNT(*) FROM memories WHERE namespace = 'kb_ns'"),
        rows_before,
        "a keyed retry must write no rows"
    );
    assert_eq!(
        count("SELECT COUNT(*) FROM oplog WHERE op_type = 'record'"),
        ops_before,
        "a keyed retry must log no ops"
    );
    let stats_after: i64 = db
        .conn()
        .query_row(
            "SELECT count FROM namespace_importance_stats WHERE namespace = 'kb_ns'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        stats_after, stats_before,
        "a keyed retry must not advance stats"
    );
}

/// 4a.6d-2b: the same key appearing twice WITHIN one batch with an identical
/// payload is one write — both positions return the same rid, one row lands.
#[test]
fn batch_in_batch_same_key_same_payload_aliases_to_one_write() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = db
        .record_batch(&[
            keyed_input("alias item", "ib_ns", 1.0, Some("ib-dup")),
            keyed_input("independent item", "ib_ns", 2.0, None),
            keyed_input("alias item", "ib_ns", 1.0, Some("ib-dup")),
        ])
        .unwrap();
    assert_eq!(rids.len(), 3, "positional result for every input");
    assert_eq!(rids[0], rids[2], "in-batch dup aliases to the first write");
    assert_ne!(rids[0], rids[1]);

    let rows: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE namespace = 'ib_ns'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(rows, 2, "the aliased dup must not write a second row");
}

/// 4a.6d-2b: the same key WITHIN one batch with a DIFFERENT payload is a
/// caller bug — the whole batch fails typed and writes NOTHING (batches stay
/// all-or-nothing on failure; a partial batch would silently drop the item a
/// retry then can never distinguish).
#[test]
fn batch_in_batch_same_key_divergent_payload_is_a_conflict() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let err = db
        .record_batch(&[
            keyed_input("first payload", "dv_ns", 1.0, Some("dv-key")),
            keyed_input("DIFFERENT payload", "dv_ns", 2.0, Some("dv-key")),
        ])
        .expect_err("divergent payloads under one key must fail the batch");
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::IdempotencyConflict { .. }
        ),
        "expected IdempotencyConflict, got {err:?}"
    );
    let rows: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE namespace = 'dv_ns'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(rows, 0, "a conflicted batch must write nothing");
}

/// 4a.6d-2b, the 4a.6c invariant on the batch surface: NOTHING may reject a
/// duplicate that would write nothing. A fully-duplicate keyed batch resolves
/// to its rids even when the delta is saturated — that is exactly when
/// clients retry — because hits are excluded from the write set BEFORE any
/// capacity is reserved.
#[test]
fn keyed_batch_duplicate_resolves_even_under_backpressure() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let dim = db.embedding_dim();
    let batch = vec![
        keyed_input("bp keyed one", "bp_ns", 1.0, Some("bp-1")),
        keyed_input("bp keyed two", "bp_ns", 2.0, Some("bp-2")),
    ];
    let rids1 = db.record_batch(&batch).unwrap();

    // Saturate the delta.
    let mut saturated = false;
    for i in 0..400 {
        let emb: Vec<f32> = (0..dim).map(|j| ((i + j) as f32) * 0.001).collect();
        match db.record(
            &format!("bp-filler-{i}"),
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &emb,
            "bp_filler_ns",
            0.8,
            "general",
            "user",
            None,
        ) {
            Ok(_) => {}
            Err(crate::error::YantrikDbError::Backpressure { .. }) => {
                saturated = true;
                break;
            }
            Err(e) => panic!("unexpected: {e:?}"),
        }
    }
    assert!(saturated, "delta never saturated");

    // A fresh unkeyed batch must still backpressure (the saturation is real)…
    let fresh_err = db
        .record_batch(&[keyed_input("bp fresh", "bp_ns", 3.0, None)])
        .expect_err("fresh batch into saturated delta must fail");
    assert!(matches!(
        fresh_err,
        crate::error::YantrikDbError::Backpressure { .. }
    ));

    // …but the keyed duplicate batch resolves to its original rids.
    let rids2 = db
        .record_batch(&batch)
        .expect("a fully-duplicate keyed batch must resolve, not backpressure");
    assert_eq!(rids2, rids1);
}

/// 4a.6d-2b: record() and record_batch share the Record digest variant, so the
/// SAME key with a byte-identical payload is a HIT across the two surfaces —
/// they are the same write (caller-supplied embedding, identical canonical
/// payload). Divergent payloads under the key conflict as always.
#[test]
fn same_key_across_record_and_record_batch_hits_on_identical_payload() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let input = keyed_input("cross surface batch text", "xb_ns", 1.5, Some("xb-key"));

    let rid = db
        .record_with_idempotency(
            &input.text,
            &input.memory_type,
            input.importance,
            input.valence,
            input.half_life,
            &input.metadata,
            &input.embedding,
            &input.namespace,
            input.certainty,
            &input.domain,
            &input.source,
            input.emotional_state.as_deref(),
            Some("xb-key"),
            None,
        )
        .unwrap();

    let rids = db.record_batch(&[input]).unwrap();
    assert_eq!(
        rids[0], rid,
        "identical payload under the same key is the same write on either surface"
    );
    let rows: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE namespace = 'xb_ns'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(rows, 1, "the batch hit must not write a second row");
}

/// 4a.6d-2b (#94): a keyed batch item's claim binds to a REAL op that
/// committed in the same transaction — op_id in the claim exists in the oplog
/// as an applied record op targeting the claimed rid. This is only possible
/// because 2b moves the batch's op INSERTs INSIDE the savepoint (the 4a.6c
/// rule: recovery never has to guess from row existence).
#[test]
fn keyed_batch_claim_binds_to_a_committed_op() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = db
        .record_batch(&[keyed_input(
            "claim binding probe",
            "cb_ns",
            1.0,
            Some("cb-key"),
        )])
        .unwrap();

    let (claim_rid, op_id, state): (String, String, String) = db
        .conn()
        .query_row(
            "SELECT rid, op_id, state FROM idempotency_claims              WHERE idempotency_key = 'cb-key' AND namespace = 'cb_ns'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .expect("a keyed batch item must write a claim");
    assert_eq!(claim_rid, rids[0]);
    assert_eq!(state, "committed");

    let (op_type, applied, target_rid): (String, i64, String) = db
        .conn()
        .query_row(
            "SELECT op_type, applied, target_rid FROM oplog WHERE op_id = ?1",
            rusqlite::params![op_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .expect("the claim's op_id must exist in the oplog — the claim binds to it");
    assert_eq!(op_type, "record");
    assert_eq!(applied, 1);
    assert_eq!(target_rid, rids[0]);
}

/// 4a.6d-2b: the UNLOCKED pre-admission probe's unique contribution — it runs
/// BEFORE the write-router, so a fully-duplicate keyed batch resolves to its
/// rids even while a reembed cutover has the router in Queueing (a fresh batch
/// correctly defers, since no queued-batch primitive exists). Without the
/// pre-router probe, a dup retry during cutover could only ever see
/// BatchDeferredDuringReembed — the retry loop that never converges, again.
#[test]
fn keyed_batch_duplicate_resolves_during_reembed_cutover() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let batch = vec![
        keyed_input("cutover keyed one", "co_ns", 1.0, Some("co-1")),
        keyed_input("cutover keyed two", "co_ns", 2.0, Some("co-2")),
    ];
    let rids1 = db.record_batch(&batch).unwrap();

    // Simulate a reembed cutover in flight.
    db.write_router.switch_to_queueing();

    // Control: a FRESH batch must defer — the router state is real.
    let err = db
        .record_batch(&[keyed_input("cutover fresh", "co_ns", 3.0, None)])
        .expect_err("fresh batch must defer during cutover");
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::BatchDeferredDuringReembed { .. }
        ),
        "expected BatchDeferredDuringReembed, got {err:?}"
    );

    // The fully-duplicate keyed batch resolves BEFORE the router.
    let rids2 = db
        .record_batch(&batch)
        .expect("a fully-duplicate keyed batch must resolve during cutover");
    assert_eq!(rids2, rids1);
}

/// 4a.6d-2b: a batch mixing an idempotent HIT with fresh items writes ONLY
/// the fresh items — hits are per-item successes (original rid at their
/// position), failures stay all-or-nothing. The hit contributes no row, no
/// second record op, no stats observation, and no entity re-extraction.
#[test]
fn partial_hit_batch_writes_only_the_fresh_items() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let keyed = keyed_input("partial hit anchor", "ph_ns", 1.0, Some("ph-key"));
    let first = db.record_batch(std::slice::from_ref(&keyed)).unwrap();

    let rids = db
        .record_batch(&[
            keyed.clone(),
            keyed_input("partial fresh item", "ph_ns", 2.0, None),
        ])
        .unwrap();
    assert_eq!(rids.len(), 2);
    assert_eq!(rids[0], first[0], "hit position carries the ORIGINAL rid");
    assert_ne!(rids[1], rids[0]);

    let count = |sql: &str| -> i64 { db.conn().query_row(sql, [], |r| r.get(0)).unwrap() };
    assert_eq!(
        count("SELECT COUNT(*) FROM memories WHERE namespace = 'ph_ns'"),
        2,
        "one row from each batch — the hit wrote nothing"
    );
    assert_eq!(
        count("SELECT COUNT(*) FROM oplog WHERE op_type = 'record'"),
        2,
        "one record op per WRITTEN item"
    );
    let stats_count: i64 = db
        .conn()
        .query_row(
            "SELECT count FROM namespace_importance_stats WHERE namespace = 'ph_ns'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        stats_count, 2,
        "two written observations, the hit added none"
    );

    // And the fresh item is durable + visible: point-read it back.
    let ns: String = db
        .conn()
        .query_row(
            "SELECT namespace FROM memories WHERE rid = ?1",
            rusqlite::params![rids[1]],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(ns, "ph_ns");
}

/// 4a.6d-2b sol r2 finding 1: a KEYED batch item's record op must carry the
/// v37 idempotency fields — record()'s payload does (record.rs ~440,
/// "a follower's keyed row must mirror its leader's"), and replication's
/// materialize_record persists them to the follower row. The batch payload
/// omitted both, so a keyed batch row replicated as an UNKEYED row on every
/// peer: the memories partial-unique mirror (the claims table's
/// defense-in-depth) silently never covered follower rows for batch writes.
/// The pre-existing batch replication test missed it because its inputs are
/// unkeyed — same lesson as the python wrapper in r1: bugs live where the
/// keyed variant is untested.
#[test]
fn keyed_batch_record_op_replicates_the_key_to_a_peer() {
    use crate::replication::{apply_ops, extract_ops_since};
    let leader = YantrikDB::new(":memory:", 8).unwrap();
    let follower = YantrikDB::new(":memory:", 8).unwrap();

    let rids = leader
        .record_batch(&[
            keyed_input("replicated keyed item", "rk_ns", 1.0, Some("rk-key")),
            keyed_input("replicated unkeyed item", "rk_ns", 2.0, None),
        ])
        .unwrap();

    // Payload-level: the keyed item's op carries BOTH fields; the unkeyed
    // item's op carries them as null (identical to record()'s shape).
    let payload_for = |rid: &str| -> serde_json::Value {
        let raw: String = leader
            .conn()
            .query_row(
                "SELECT payload FROM oplog WHERE op_type = 'record' AND target_rid = ?1",
                rusqlite::params![rid],
                |r| r.get(0),
            )
            .unwrap();
        serde_json::from_str(&raw).unwrap()
    };
    let keyed_payload = payload_for(&rids[0]);
    assert_eq!(
        keyed_payload["idempotency_key"], "rk-key",
        "keyed batch op payload must carry the key for follower mirroring"
    );
    assert_eq!(
        keyed_payload["origin_actor"],
        leader.actor_id(),
        "keyed batch op payload must carry the origin actor"
    );
    let unkeyed_payload = payload_for(&rids[1]);
    assert!(
        unkeyed_payload["idempotency_key"].is_null(),
        "unkeyed items carry null, matching record()'s keyless shape"
    );

    // End-to-end: the follower row mirrors the leader's v37 columns.
    let ops = extract_ops_since(&leader.conn(), None, None, None, 100).unwrap();
    apply_ops(&follower, &ops).unwrap();
    let (f_key, f_actor): (Option<String>, Option<String>) = follower
        .conn()
        .query_row(
            "SELECT idempotency_key, origin_actor FROM memories WHERE rid = ?1",
            rusqlite::params![rids[0]],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(
        f_key.as_deref(),
        Some("rk-key"),
        "follower keyed row must mirror the leader's idempotency_key"
    );
    assert_eq!(
        f_actor.as_deref(),
        Some(leader.actor_id()),
        "follower keyed row must mirror the leader's origin_actor"
    );
}

/// 4a.6d-3 helper: a record_with_rid call with every knob defaulted.
#[allow(clippy::too_many_arguments)]
fn rwr(
    db: &YantrikDB,
    rid: &str,
    text: &str,
    ns: &str,
    entities: &[&str],
    seq: Option<u64>,
) -> crate::error::Result<()> {
    db.record_with_rid(
        rid,
        text,
        "semantic",
        0.6,
        0.0,
        604800.0,
        &serde_json::json!({}),
        &vec_seed(4.0, 8),
        ns,
        0.8,
        "general",
        "user",
        None,
        1_750_000_000_000_000,
        entities,
        "test-embedder",
        seq,
        crate::provenance::WriteAdmission::Origin,
    )
}

/// 4a.6d-3 (#94's class on this path): the pending-queue admission check ran
/// INSIDE `log_op_pending`, AFTER the row and vector were durable — so a
/// saturated pending queue returned Err for a write that had already
/// committed, and a retrying caller (the cluster applier!) then hit the
/// was_new_row=false arm which never logs the op: the row exists forever
/// without provenance. Post-port the locked check runs BEFORE any durable
/// byte: Err means nothing was written.
#[test]
fn record_with_rid_pending_saturation_rejects_before_writing() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.pending_op_count
        .swap(1_000_000, std::sync::atomic::Ordering::SeqCst);

    let err = rwr(
        &db,
        "0198c1c2-0000-7000-8000-0000000000aa",
        "saturated pending write",
        "sat_ns",
        &["SaturatedEntity"],
        None,
    )
    .expect_err("pending saturation must reject the write");
    assert!(
        matches!(err, crate::error::YantrikDbError::Backpressure { .. }),
        "expected Backpressure, got {err:?}"
    );

    let rows: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE namespace = 'sat_ns'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        rows, 0,
        "a write rejected for pending saturation must write NOTHING —          pre-port the row and vector were already durable when the check ran"
    );
    let ops: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM oplog WHERE op_type = 'record_with_rid'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(ops, 0, "no op for a rejected write");
    assert!(db.conn().is_autocommit(), "no savepoint left open");
}

/// 4a.6d-3: replaying an EXISTING rid must not re-enqueue its entity
/// materialization. Pre-port the enqueue ran post-commit unconditionally
/// (its rationale: repair a crash between the row and the enqueue) — but the
/// port commits row + op + enqueue in ONE savepoint, so the repair case
/// cannot exist and a replay re-enqueue is pure duplicate work inflating the
/// pending queue on every cluster re-delivery.
#[test]
fn record_with_rid_replay_does_not_reenqueue_materialization() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = "0198c1c2-0000-7000-8000-0000000000bb";

    rwr(&db, rid, "replayed write", "rp_ns", &["ReplayEntity"], None).unwrap();
    rwr(&db, rid, "replayed write", "rp_ns", &["ReplayEntity"], None).unwrap();

    let enqueues: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM oplog WHERE op_type = ?1 AND target_rid = ?2",
            rusqlite::params![
                crate::engine::op_types::OP_MATERIALIZE_RECORD_WITH_RID_POST,
                rid
            ],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        enqueues, 1,
        "replay of an existing rid re-enqueued its materialization"
    );

    // And the replay logged no second record_with_rid op (pre-existing
    // was_new_row gate — pinned here so the port cannot regress it).
    let ops: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM oplog WHERE op_type = 'record_with_rid' AND target_rid = ?1",
            rusqlite::params![rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        ops, 1,
        "exactly one record_with_rid op for one logical write"
    );
}

/// 4a.6d-3: the in-tx materialization enqueue owes pending_op_count exactly
/// +1, discharged post-commit by the reservation guard's upgrade
/// (`count_pending_op_on_completion`). Counting is what the backpressure
/// ceiling reads — a skipped increment drifts the counter low and the
/// ceiling stops engaging (the inverse of the v0.7.1 wedge).
#[test]
fn record_with_rid_enqueue_counts_exactly_one_pending_op() {
    use std::sync::atomic::Ordering;
    let db = YantrikDB::new(":memory:", 8).unwrap();

    let before = db.pending_op_count.load(Ordering::SeqCst);
    rwr(
        &db,
        "0198c1c2-0000-7000-8000-0000000000cc",
        "counted enqueue",
        "cnt_ns",
        &["CountedEntity"],
        None,
    )
    .unwrap();
    assert_eq!(
        db.pending_op_count.load(Ordering::SeqCst),
        before + 1,
        "one enqueued materialization op = exactly one count"
    );

    // No entities -> no enqueue -> no count.
    rwr(
        &db,
        "0198c1c2-0000-7000-8000-0000000000cd",
        "uncounted write",
        "cnt_ns",
        &[],
        None,
    )
    .unwrap();
    assert_eq!(
        db.pending_op_count.load(Ordering::SeqCst),
        before + 1,
        "an entity-less write enqueues nothing and must not count"
    );

    // Replay -> no enqueue -> no count.
    rwr(
        &db,
        "0198c1c2-0000-7000-8000-0000000000cc",
        "counted enqueue",
        "cnt_ns",
        &["CountedEntity"],
        None,
    )
    .unwrap();
    assert_eq!(
        db.pending_op_count.load(Ordering::SeqCst),
        before + 1,
        "a replay enqueues nothing and must not count"
    );
}
