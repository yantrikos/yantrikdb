use super::*;

/// A text-changing correction reserves a vector, commits, publishes — the same
/// protocol `record()` runs. But its `correct` op commits via `log_op_in_tx` with
/// `applied = 1`, so it enqueues NO pending op and must never move
/// `pending_op_count`.
///
/// This is the test that catches a correction site built with the WRONG guard
/// constructor (`with_pending_op` instead of `publish_only`). The guard's own unit
/// test cannot: `publish_only` holds `None`, so the type system already stops it
/// there — only a real site can pick the wrong one.
///
/// Getting it wrong inflates the cache against zero pending rows. Nothing brings
/// it back down (`mark_op_applied` only decrements rows it actually transitions,
/// and there is no row), so the drift is monotonic and at `MAX_PENDING_OPS` the
/// admission check wedges EVERY foreground write into `Backpressure` forever —
/// the v0.7.1 counter-leak class, reached by "reusing" record()'s guard.
#[test]
fn text_correction_publishes_its_vector_without_counting_a_pending_op() {
    struct Fake;
    impl crate::types::Embedder for Fake {
        fn embed(
            &self,
            t: &str,
        ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
            // Deterministic, text-dependent: the corrected text must produce a
            // DIFFERENT vector, or there is nothing to reserve and publish.
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
        .record_text(
            "the sky is green today",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            "default",
            0.8,
            "work",
            "user",
            None,
        )
        .unwrap();

    let pending_before = db
        .pending_op_count
        .load(std::sync::atomic::Ordering::Relaxed);

    db.correct(&rid, Some("the sky is blue"), None, None, None, "observed")
        .expect("text correction must succeed");

    assert_eq!(
        db.pending_op_count
            .load(std::sync::atomic::Ordering::Relaxed),
        pending_before,
        "a correction commits an applied op and enqueues nothing pending — \
         counting here is the v0.7.1 counter-leak class"
    );

    // ...and it really did run the reserve→publish protocol: the corrected text
    // is durable and retrievable, i.e. the vector was published, not stranded.
    let text: String = db
        .conn()
        .query_row("SELECT text FROM memories WHERE rid = ?1", [&rid], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(text, "the sky is blue", "correction is durable");
}

/// The REPLICATED correction's twin of the test above (sol's item-2 review: the
/// leader site was covered and this one was not — asymmetric coverage between two
/// sites owning one rule is exactly how the whole 4a series kept getting bitten).
///
/// `apply_replicated_correct` reserves a vector and publishes it post-commit, but
/// its oplog rows all commit `applied = 1`, so like the leader it must never move
/// `pending_op_count`. Mutating its `publish_only` to `with_pending_op` must fail
/// THIS test — the leader's test cannot see that site.
#[test]
fn replicated_correction_publishes_its_vector_without_counting_a_pending_op() {
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
        .record_text(
            "the sky is green today",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            "default",
            0.8,
            "work",
            "user",
            None,
        )
        .unwrap();

    let pending_before = db
        .pending_op_count
        .load(std::sync::atomic::Ordering::Relaxed);

    // reembedded=true with no usable exact bytes → the follower re-embeds
    // locally, which is the branch that reserves a vector.
    db.apply_replicated_correct(
        &serde_json::json!({
            "rid": rid,
            "revision_num": 1,
            "new_text": "the sky is blue",
            "reason": "peer correction",
            "reembedded": true,
        }),
        None,
        "peer-actor",
    )
    .expect("replicated correction must apply");

    assert_eq!(
        db.pending_op_count
            .load(std::sync::atomic::Ordering::Relaxed),
        pending_before,
        "a replicated correction commits applied=1 ops and enqueues nothing pending — \
         counting here is the v0.7.1 counter-leak class"
    );

    let text: String = db
        .conn()
        .query_row("SELECT text FROM memories WHERE rid = ?1", [&rid], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(text, "the sky is blue", "replicated correction is durable");
}

/// The anti-laundering gate had a one-character bypass (found wiring 4a.6b,
/// verified empirically before fixing): 4a.4b gated the scalar-only correction
/// path, but a TEXT-CHANGING correction dispatches to `correct_with_reembed`
/// BEFORE that gate runs — and merges `metadata_merge` all the same. So in
/// Enforce mode, `correct(new_text=<any change>, metadata_merge={"kind":"fact"})`
/// on an inference-sourced record committed the exact laundering the gate
/// exists to refuse. Pre-fix: the scalar flip refused, the text flip stored
/// kind="fact".
#[test]
fn text_changing_correction_cannot_launder_kind() {
    let db = YantrikDB::new(":memory:", 8).unwrap(); // fresh ⇒ Enforce
    assert_eq!(db.stats(None).unwrap().provenance_gate_mode, "enforce");
    let rid = db
        .record(
            "the sky might be green",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "work",
            "inference",
            None,
        )
        .unwrap();
    let gen = db.search_state.load().generation;

    // The laundering attempt, via the path that used to bypass the gate.
    let err = db
        .correct_with_embedding(
            &rid,
            Some("the sky might be green!"),
            &vec_seed(1.1, 8),
            gen,
            Some(&serde_json::json!({"kind": "fact"})),
            None,
            None,
            "launder via text change",
        )
        .expect_err("text-changing kind flip must be refused in Enforce");
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::ProvenanceInconsistent { .. }
        ),
        "expected ProvenanceInconsistent, got {err:?}"
    );

    // Refused BEFORE any side effect: text, kind, and revision chain untouched.
    let (text, kind): (String, Option<String>) = db
        .conn()
        .query_row(
            "SELECT text, json_extract(metadata, '$.kind') FROM memories WHERE rid = ?1",
            rusqlite::params![rid],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(text, "the sky might be green", "text must be unchanged");
    assert_eq!(kind, None, "kind must not have been laundered in");
    assert_eq!(db.history(&rid).unwrap().len(), 0, "no revision recorded");

    // The documented escape is raising the basis — that must still work
    // through the SAME text-changing path (the gate refuses contradictions,
    // not corrections).
    db.correct_with_embedding(
        &rid,
        Some("the sky is verified green"),
        &vec_seed(1.2, 8),
        gen,
        Some(&serde_json::json!({"kind": "fact", "confidence_basis": "verification"})),
        None,
        None,
        "verified independently",
    )
    .expect("basis-raising text correction must pass the gate");
}

/// 4a.6b winner-only, warn-mode half: a FLAGGED write that COMMITS ticks
/// `provenance_flagged_since_boot` exactly once, on every gated path. (The
/// anchor backpressure test proves the rejected side; this proves the
/// accepted side didn't get lost in the verdict refactor.)
#[test]
fn flagged_committed_writes_tick_the_nudge_counter_exactly_once() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.set_provenance_gate_mode(crate::provenance::GateMode::Warn)
        .unwrap();
    let flagged = |db: &YantrikDB| db.stats(None).unwrap().provenance_flagged_since_boot;
    let launder_meta = serde_json::json!({"kind": "fact"});

    // record(): sync path.
    let n0 = flagged(&db);
    let rid = db
        .record(
            "flagged one",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &launder_meta,
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "work",
            "inference",
            None,
        )
        .unwrap();
    assert_eq!(flagged(&db), n0 + 1, "committed flagged record ticks once");

    // record(): queued path (router in Queueing).
    db.write_router.switch_to_queueing();
    db.record(
        "flagged two",
        "semantic",
        0.7,
        0.0,
        604800.0,
        &launder_meta,
        &vec_seed(1.05, 8),
        "default",
        0.8,
        "work",
        "inference",
        None,
    )
    .unwrap();
    assert_eq!(
        flagged(&db),
        n0 + 2,
        "committed flagged queued write ticks once"
    );
    db.write_router.switch_to_normal();

    // record_batch(): two flagged inputs, one clean.
    let mk = |text: &str, meta: serde_json::Value, source: &str| RecordInput {
        idempotency_key: None,
        text: text.into(),
        memory_type: "semantic".into(),
        importance: 0.7,
        valence: 0.0,
        half_life: 604800.0,
        metadata: meta,
        embedding: vec_seed(1.1, 8),
        namespace: "default".into(),
        certainty: 0.8,
        domain: "work".into(),
        source: source.into(),
        emotional_state: None,
    };
    db.record_batch(&[
        mk("flagged three", launder_meta.clone(), "inference"),
        mk("clean one", serde_json::json!({}), "user"),
        mk("flagged four", launder_meta.clone(), "inference"),
    ])
    .unwrap();
    assert_eq!(flagged(&db), n0 + 4, "batch ticks once per flagged input");

    // correct(): scalar path (metadata-only flip on the inference record —
    // warn allows it, counts it).
    db.correct(&rid, None, Some(&launder_meta), None, None, "warn flip")
        .unwrap();
    assert_eq!(
        flagged(&db),
        n0 + 5,
        "committed flagged correction ticks once"
    );

    // correct(): text-changing path (the ex-bypass — now gated AND counted).
    let gen = db.search_state.load().generation;
    db.correct_with_embedding(
        &rid,
        Some("flagged one, changed"),
        &vec_seed(1.15, 8),
        gen,
        Some(&launder_meta),
        None,
        None,
        "warn flip with text",
    )
    .unwrap();
    assert_eq!(
        flagged(&db),
        n0 + 6,
        "committed flagged text-correction ticks once"
    );
}

/// 4a.6b, batch loser path: a batch deferred by the write-router (reembed
/// cutover in flight) must leave every namespace's importance distribution
/// untouched. Pre-fix, `record_batch` calibrated ALL inputs — autocommitting
/// every namespace's EWMA advance — BEFORE it ever consulted the router.
#[test]
fn deferred_batch_leaves_importance_stats_untouched() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.write_router.switch_to_queueing();

    let err = db
        .record_batch(&[RecordInput {
            idempotency_key: None,
            text: "deferred".into(),
            memory_type: "semantic".into(),
            importance: 0.9,
            valence: 0.0,
            half_life: 604800.0,
            metadata: serde_json::json!({}),
            embedding: vec_seed(1.0, 8),
            namespace: "bp_ns".into(),
            certainty: 0.8,
            domain: "work".into(),
            source: "user".into(),
            emotional_state: None,
        }])
        .expect_err("router is Queueing; the batch must defer");
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::BatchDeferredDuringReembed { .. }
        ),
        "expected BatchDeferredDuringReembed, got {err:?}"
    );

    let rows: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM namespace_importance_stats WHERE namespace = 'bp_ns'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        rows, 0,
        "a deferred batch advanced namespace_importance_stats — a loser moved state"
    );
    db.write_router.switch_to_normal();
}

/// 4a.6b sol r2 finding 2: `record_with_rid` is both the public origin API and
/// the cluster apply primitive. As ORIGIN it must gate provenance (it was a
/// public Enforce bypass — a Rust caller could persist source=inference/
/// kind=fact directly). As ADMITTED it must NOT gate — re-gating a
/// consensus-committed op on the apply path makes followers reject the leader
/// and wedge the cluster. The required `WriteAdmission` arg forces the choice.
#[test]
fn record_with_rid_gates_origin_but_not_admitted() {
    let mk = |db: &YantrikDB, rid: &str, adm: crate::provenance::WriteAdmission| {
        db.record_with_rid(
            rid,
            "the sky is green",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &serde_json::json!({"kind": "fact"}), // laundering: inference + fact
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "work",
            "inference",
            None,
            1_000_000,
            &[],
            "test-embedder",
            None,
            adm,
        )
    };

    // ORIGIN, Enforce (fresh DB): the contradiction is refused.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    assert_eq!(db.stats(None).unwrap().provenance_gate_mode, "enforce");
    let err = mk(&db, "rid_origin", crate::provenance::WriteAdmission::Origin)
        .expect_err("origin record_with_rid must gate a declared contradiction");
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::ProvenanceInconsistent { .. }
        ),
        "expected ProvenanceInconsistent, got {err:?}"
    );
    let n: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE rid = 'rid_origin'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 0, "refused origin write persisted nothing");

    // ADMITTED, Enforce: the SAME op applies (it was gated at its origin; the
    // apply path must not re-gate, or the cluster wedges).
    mk(
        &db,
        "rid_admitted",
        crate::provenance::WriteAdmission::Admitted,
    )
    .expect("admitted apply must NOT be re-gated");
    let n: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE rid = 'rid_admitted'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 1, "admitted apply committed");
}

/// 4a.6b sol r3 finding 2: `record_with_rid` is `INSERT OR IGNORE`, so replaying
/// an existing rid persists nothing. A warn-mode ORIGIN replay of a flagged op
/// must NOT tick `provenance_flagged_since_boot` — that counter gates the
/// warn→enforce decision, and counting no-op replays inflates it with writes
/// that never landed.
#[test]
fn record_with_rid_origin_replay_does_not_overcount_flags() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.set_provenance_gate_mode(crate::provenance::GateMode::Warn)
        .unwrap();
    let flagged = |db: &YantrikDB| db.stats(None).unwrap().provenance_flagged_since_boot;

    let write = |db: &YantrikDB| {
        db.record_with_rid(
            "dup_rid",
            "flagged origin write",
            "semantic",
            0.7,
            0.0,
            604800.0,
            &serde_json::json!({"kind": "fact"}),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "work",
            "inference",
            None,
            1_000_000,
            &[],
            "test-embedder",
            None,
            crate::provenance::WriteAdmission::Origin,
        )
    };

    let n0 = flagged(&db);
    write(&db).expect("first origin write lands");
    assert_eq!(
        flagged(&db),
        n0 + 1,
        "first flagged origin write ticks once"
    );

    // Replay the SAME rid: INSERT OR IGNORE persists nothing, so no tick.
    write(&db).expect("replay is a no-op Ok");
    assert_eq!(
        flagged(&db),
        n0 + 1,
        "idempotent replay of an existing rid must not tick the nudge counter"
    );
}

/// 4a.6b sol r2 finding 1: a batch whose VECTOR APPEND fails (after the SQL
/// savepoint has committed) must leave `namespace_importance_stats` untouched.
/// The stats advance is deferred to after the append loop wins; before this fix
/// it ran inside the savepoint, and a committed EWMA blend cannot be reversed by
/// the compensating row DELETE — so a rejected batch permanently moved the
/// namespace's calibration.
#[test]
fn batch_append_failure_leaves_importance_stats_untouched() {
    // Saturate the delta so the NEXT append fails, using single-record writes in
    // a DIFFERENT namespace so the target namespace's stats stay pristine.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let dim = db.embedding_dim();
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

    // The target namespace has no stats row yet.
    let stats = |db: &YantrikDB| -> Option<(f64, i64)> {
        db.conn()
            .query_row(
                "SELECT ewma, count FROM namespace_importance_stats WHERE namespace = 'batch_ns'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .ok()
    };
    assert_eq!(stats(&db), None, "precondition: no stats for batch_ns");

    // A batch into batch_ns: since 4a.6d-2a the capacity reservation fails
    // BEFORE the savepoint even opens (pre-restructure: the savepoint
    // committed, the post-RELEASE append failed, and a compensating DELETE
    // reversed the rows). Either way the caller sees Backpressure and the
    // stats assertions below are what this test pins.
    let err = db
        .record_batch(&[RecordInput {
            idempotency_key: None,
            text: "batch after saturation".into(),
            memory_type: "semantic".into(),
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
        .expect_err("append must fail into the saturated delta");
    assert!(
        matches!(err, crate::error::YantrikDbError::Backpressure { .. }),
        "expected Backpressure from the append, got {err:?}"
    );

    // No rows AND stats untouched (the winner-only guarantee).
    let rows: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE namespace = 'batch_ns'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(rows, 0, "a rejected batch must write no rows");
    assert_eq!(
        stats(&db),
        None,
        "a batch whose append failed advanced namespace_importance_stats — \
         a loser moved calibration state"
    );
}

/// 4a.6b finding 3: the in-tx SQL EWMA advance must reproduce the predecessor's
/// `count == 0 => ewma = raw` seed exactly. A `(ewma=X, count=0)` row is not
/// produced by any engine path, but the schema permits it and `conn()` is
/// public, so a stray zero-count row must SEED to the incoming raw, not blend
/// the stale X in. Without the CASE, this stores 0.32 instead of 1.0.
#[test]
fn stats_advance_seeds_from_raw_when_stored_count_is_zero() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.conn()
        .execute(
            "INSERT INTO namespace_importance_stats (namespace, ewma, count, updated_at) \
             VALUES ('zc', 0.2, 0, 1.0)",
            [],
        )
        .unwrap();

    // Drive one real write through the sync path into namespace 'zc'.
    db.record(
        "seed from raw",
        "semantic",
        1.0,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(1.0, 8),
        "zc",
        0.8,
        "work",
        "user",
        None,
    )
    .unwrap();

    let (ewma, count): (f64, i64) = db
        .conn()
        .query_row(
            "SELECT ewma, count FROM namespace_importance_stats WHERE namespace = 'zc'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert!(
        (ewma - 1.0).abs() < 1e-9,
        "count=0 must seed ewma to raw (1.0), not blend the stale 0.2 in; got {ewma}"
    );
    assert_eq!(count, 1, "count advances to 1");
}
