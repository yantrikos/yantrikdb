use super::*;

// =====================================================================
// v0.8.x — schema v27 reembed-operation foundation
// (issue yantrikos/yantrikdb#41).
//
// v27 introduces:
//   - `memories.embedding_new` BLOB + `memories.embedding_new_model` TEXT
//     staging columns for the `db.reembed()` Encoding phase.
//   - `oplog.embedding_model` TEXT to discriminate pre-reembed pending ops
//     (where oplog.embedding is trustworthy) from post-reembed-queued ops
//     (where the materializer must re-encode from text under the new
//     embedder after the SearchState swap).
//   - `reembed_events` audit-log table with `(generation, phase,
//     timestamp, payload_json)` rows. Authoritative source for crash
//     recovery on open().
//
// Tests cover three paths:
//   1. Fresh install: all v27 surfaces exist (columns + table + index).
//   2. Pre-v27 migration: upgrade cleanly from v26, additive only, no
//      data touched on existing rows.
//   3. Replay-resilience: rewinding meta to 26 on an already-v27 DB
//      doesn't break the second open (per v0.7.3 idempotent runner).
// =====================================================================

// =====================================================================
// Issue #41 layer 3: record() routing through WriteRouter
// =====================================================================

#[test]
fn record_in_normal_state_takes_sync_path() {
    // Sanity: default router state is Normal, record() takes the sync
    // path and the memory is immediately in `memories` + vec_index.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    assert_eq!(
        db.write_router.state(),
        crate::engine::write_router::RouterState::Normal
    );
    let rid = db
        .record(
            "sync path test",
            "episodic",
            0.5,
            0.0,
            86400.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "default",
            0.9,
            "general",
            "user",
            None,
        )
        .unwrap();
    // Row immediately visible in memories table (sync path completed).
    let count: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE rid = ?1",
            params![rid],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        count, 1,
        "sync-path write must land in memories immediately"
    );
}

#[test]
fn record_in_queueing_state_routes_to_oplog_does_not_touch_memories() {
    // Locks the brainstorm-2/3 invariant: when reembed cutover has
    // flipped the router to Queueing, record() must NOT write to
    // memories (would mix old+new dim under the rebuild snapshot)
    // and must NOT call vec_index.append. The op goes to oplog
    // applied=0 with embedding_model populated for the post-swap
    // materializer to re-encode under the new embedder.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // Flip router as reembed cutover would.
    db.write_router.switch_to_queueing();
    assert_eq!(
        db.write_router.state(),
        crate::engine::write_router::RouterState::Queueing
    );
    // Count memories + oplog before the queued record.
    let mem_before: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
        .unwrap();
    let oplog_before: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM oplog WHERE applied = 0 AND op_type = 'record'",
            [],
            |row| row.get(0),
        )
        .unwrap();

    let rid = db
        .record(
            "queued path test",
            "episodic",
            0.5,
            0.0,
            86400.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "default",
            0.9,
            "general",
            "user",
            None,
        )
        .unwrap();

    // memories table count must NOT have grown — the queued path
    // skips the memories INSERT entirely.
    let mem_after: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
        .unwrap();
    assert_eq!(
        mem_after, mem_before,
        "queued path must NOT write to memories table during reembed cutover \
         (brainstorm-2/3 invariant 1: queued-after-barrier writes are replayed \
         by post-swap materializer, not committed to old generation)"
    );

    // oplog count must have grown by 1, with applied=0, op_type='record',
    // target_rid=the new rid, and embedding_model set to whatever the
    // active runtime embedder was (None here since no embedder is set).
    let oplog_after: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM oplog WHERE applied = 0 AND op_type = 'record' \
             AND target_rid = ?1",
            params![rid],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        oplog_after,
        oplog_before + 1,
        "queued path must write the record op to oplog with applied=0"
    );

    // applied_generation must be NULL (this op will be applied to the
    // new generation by the post-swap materializer; until then it's
    // not applied to any generation).
    let applied_gen: Option<i64> = db
        .conn()
        .query_row(
            "SELECT applied_generation FROM oplog WHERE target_rid = ?1",
            params![rid],
            |row| row.get(0),
        )
        .unwrap();
    assert!(
        applied_gen.is_none(),
        "applied_generation must be NULL for queued ops; got Some({applied_gen:?})"
    );

    // Restore Normal for any subsequent test in the same DB.
    db.write_router.switch_to_normal();
}

#[test]
fn record_guard_drops_inflight_counter_panic_safe_via_raii() {
    // Locks brainstorm-2 invariant 2 (no old application after barrier)
    // by exercising the panic-safety of the SyncWriteGuard. Even if
    // record() panics mid-write (simulated here by a record_batch
    // wrapping that panics after the guard is acquired), the inflight
    // counter must return to 0 via Drop.
    let db = std::sync::Arc::new(YantrikDB::new(":memory:", 8).unwrap());
    assert_eq!(db.write_router.inflight(), 0);

    let db_panic = std::sync::Arc::clone(&db);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = db_panic
            .write_router
            .try_enter_sync_writer()
            .expect("Normal state must yield guard");
        // inflight = 1 here
        assert_eq!(db_panic.write_router.inflight(), 1);
        panic!("simulated mid-write panic");
    }));
    assert!(result.is_err(), "panic must propagate up");
    // Guard's Drop ran via panic unwind; inflight back to 0.
    assert_eq!(
        db.write_router.inflight(),
        0,
        "panic-safe inflight counter (RAII Drop) — required for reembed cutover correctness"
    );
}

// =====================================================================
// Issue #41 layer 2: set_embedder mode-aware regression tests
// (brainstorm-3 round 2 §11, 8 cases). These lock the brainstorm-3
// design decisions. Each test names the failure mode it prevents.
// =====================================================================

/// Helper Embedder impls for the mode-table tests. Each provides a
/// distinct fingerprint so the mode logic can discriminate.
mod mode_test_embedders {
    use crate::types::Embedder;

    pub struct FakeEmbedder {
        pub dim: usize,
        pub fp: Option<String>,
        pub name: Option<String>,
        /// Sentinel byte returned in vec[0] so tests can verify
        /// "which embedder produced this vector".
        pub sentinel: f32,
    }

    impl Embedder for FakeEmbedder {
        fn embed(
            &self,
            _text: &str,
        ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
            let mut v = vec![0.0_f32; self.dim];
            if !v.is_empty() {
                v[0] = self.sentinel;
            }
            Ok(v)
        }
        fn dim(&self) -> usize {
            self.dim
        }
        fn fingerprint(&self) -> Option<String> {
            self.fp.clone()
        }
        fn name(&self) -> Option<String> {
            self.name.clone()
        }
    }
}

#[test]
fn set_embedder_test_1_same_dim_different_digest_on_populated_db_rejected() {
    // Locks the silent-corruption-prevention invariant. The pre-#41
    // engine accepted this case silently and produced garbage scores.
    // After #41 it must return ChangeEmbedderDigestRequiresReembed.
    use mode_test_embedders::FakeEmbedder;
    let mut db = YantrikDB::new(":memory:", 64).unwrap();
    db.set_embedder(Box::new(FakeEmbedder {
        dim: 64,
        fp: Some("sha256:embedder_A".to_string()),
        name: Some("embedder_A".to_string()),
        sentinel: 1.0,
    }))
    .unwrap();
    let _ = db
        .record(
            "first memory",
            "semantic",
            0.5,
            0.0,
            86400.0,
            &empty_meta(),
            &vec_seed(1.0, 64),
            "default",
            0.9,
            "general",
            "user",
            None,
        )
        .unwrap();
    let err = db
        .set_embedder(Box::new(FakeEmbedder {
            dim: 64,
            fp: Some("sha256:embedder_B".to_string()),
            name: Some("embedder_B".to_string()),
            sentinel: 2.0,
        }))
        .unwrap_err();
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::ChangeEmbedderDigestRequiresReembed { .. }
        ),
        "same-dim-different-digest on Known-provenance populated DB must \
         return ChangeEmbedderDigestRequiresReembed (silent-corruption \
         prevention invariant from brainstorm-3); got {err:?}"
    );
}

#[test]
fn set_embedder_test_2_different_dim_on_populated_db_rejected() {
    use mode_test_embedders::FakeEmbedder;
    let mut db = YantrikDB::new(":memory:", 64).unwrap();
    db.set_embedder(Box::new(FakeEmbedder {
        dim: 64,
        fp: Some("sha256:fp64".to_string()),
        name: None,
        sentinel: 1.0,
    }))
    .unwrap();
    let _ = db
        .record(
            "m",
            "semantic",
            0.5,
            0.0,
            86400.0,
            &empty_meta(),
            &vec_seed(1.0, 64),
            "default",
            0.9,
            "general",
            "user",
            None,
        )
        .unwrap();
    let err = db
        .set_embedder(Box::new(FakeEmbedder {
            dim: 128,
            fp: Some("sha256:fp128".to_string()),
            name: None,
            sentinel: 2.0,
        }))
        .unwrap_err();
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::ChangeEmbedderDimensionRequiresReembed { .. }
        ),
        "dim change on populated DB must return \
         ChangeEmbedderDimensionRequiresReembed; got {err:?}"
    );
}

#[test]
fn set_embedder_test_3_empty_db_with_fingerprint_upgrades_provenance_to_known() {
    // Empty DB + candidate has fingerprint → provenance upgrades to
    // Known(fp). Locks the initial-attach upgrade path.
    //
    // Dim 128, not 64: at 64 `YantrikDB::new` auto-attaches the bundled
    // embedder, which now carries a real fingerprint, so provenance is
    // already Known before this test's own set_embedder call and the
    // pre-condition below could never hold. A dim with no bundled
    // default keeps the test measuring the upgrade path it names.
    use mode_test_embedders::FakeEmbedder;
    let mut db = YantrikDB::new(":memory:", 128).unwrap();
    assert!(matches!(
        db.search_state.load().index_embedding,
        crate::engine::reembed::EmbeddingProvenance::ExternalOrUnknown { .. }
    ));
    db.set_embedder(Box::new(FakeEmbedder {
        dim: 128,
        fp: Some("sha256:initial".to_string()),
        name: Some("initial".to_string()),
        sentinel: 1.0,
    }))
    .unwrap();
    let s = db.search_state.load_full();
    match &s.index_embedding {
        crate::engine::reembed::EmbeddingProvenance::Known { name, digest, dim } => {
            assert_eq!(name.as_deref(), Some("initial"));
            assert_eq!(digest, "sha256:initial");
            assert_eq!(*dim, 128);
        }
        other => panic!("expected Known provenance after attach on empty DB, got {other:?}"),
    }
}

#[test]
fn set_embedder_test_4_empty_db_no_fingerprint_stays_external_or_unknown() {
    // Empty DB + candidate fingerprint is None → provenance stays
    // ExternalOrUnknown. We cannot claim a digest we don't have.
    use mode_test_embedders::FakeEmbedder;
    let mut db = YantrikDB::new(":memory:", 64).unwrap();
    db.set_embedder(Box::new(FakeEmbedder {
        dim: 64,
        fp: None,
        name: None,
        sentinel: 1.0,
    }))
    .unwrap();
    let s = db.search_state.load_full();
    assert!(
        matches!(
            s.index_embedding,
            crate::engine::reembed::EmbeddingProvenance::ExternalOrUnknown { dim: 64 }
        ),
        "no-fingerprint embedder on empty DB must keep ExternalOrUnknown provenance, \
         got {:?}",
        s.index_embedding
    );
    assert!(s.has_runtime_embedder());
}

#[test]
fn set_embedder_test_5_same_digest_replacement_does_not_bump_generation() {
    // Replacing a runtime embedder with one that has the SAME digest is
    // a same-model swap. Generation must NOT bump (index + provenance
    // unchanged; only the runtime Arc was replaced).
    use mode_test_embedders::FakeEmbedder;
    let mut db = YantrikDB::new(":memory:", 64).unwrap();
    db.set_embedder(Box::new(FakeEmbedder {
        dim: 64,
        fp: Some("sha256:same".to_string()),
        name: Some("same".to_string()),
        sentinel: 1.0,
    }))
    .unwrap();
    let gen_before = db.search_state.load().generation;
    db.set_embedder(Box::new(FakeEmbedder {
        dim: 64,
        fp: Some("sha256:same".to_string()),
        name: Some("same".to_string()),
        sentinel: 2.0,
    }))
    .unwrap();
    let gen_after = db.search_state.load().generation;
    assert_eq!(
        gen_after, gen_before,
        "same-digest replacement must NOT bump generation (no coherent-bundle change)"
    );
    let v = db.embed("anything").unwrap();
    assert!(
        (v[0] - 2.0).abs() < 1e-6,
        "runtime Arc must have been replaced; expected sentinel 2.0, got {}",
        v[0]
    );
}

#[test]
fn set_embedder_test_6_external_or_unknown_compat_attach_does_not_claim_provenance() {
    // ExternalOrUnknown-provenance populated DB + candidate with
    // matching dim → compat attach. Runtime embedder is set, but
    // index_embedding stays ExternalOrUnknown.
    // Dim 128 so `YantrikDB::new` does not auto-attach the bundled
    // embedder — at 64 it would, and its (now real) fingerprint would
    // make provenance Known before this test ever gets to populate the
    // DB from an external source.
    use mode_test_embedders::FakeEmbedder;
    let mut db = YantrikDB::new(":memory:", 128).unwrap();
    // Populate DB without setting an embedder — vectors come from
    // external source. Provenance stays ExternalOrUnknown.
    let _ = db
        .record(
            "external vec",
            "semantic",
            0.5,
            0.0,
            86400.0,
            &empty_meta(),
            &vec_seed(1.0, 128),
            "default",
            0.9,
            "general",
            "user",
            None,
        )
        .unwrap();
    assert!(matches!(
        db.search_state.load().index_embedding,
        crate::engine::reembed::EmbeddingProvenance::ExternalOrUnknown { .. }
    ));
    db.set_embedder(Box::new(FakeEmbedder {
        dim: 128,
        fp: Some("sha256:attached".to_string()),
        name: Some("attached".to_string()),
        sentinel: 1.0,
    }))
    .unwrap();
    let s = db.search_state.load_full();
    assert!(
        matches!(
            s.index_embedding,
            crate::engine::reembed::EmbeddingProvenance::ExternalOrUnknown { .. }
        ),
        "compat-attach must NOT upgrade ExternalOrUnknown provenance to Known; \
         existing vectors weren't built with this embedder. Got {:?}",
        s.index_embedding
    );
    assert!(s.has_runtime_embedder());
    assert_eq!(
        s.runtime_embedder_digest.as_deref(),
        Some("sha256:attached")
    );
}

#[test]
fn set_embedder_test_7_has_embedder_derives_from_search_state() {
    // Locks the brainstorm-3 invariant: has_embedder() reads from
    // search_state, NOT from the (now-retired) legacy embedder slot.
    //
    // Opens at dim=384 so the bundled-embedder auto-attach (which is
    // dim=64 and silently fails on dim mismatch) cannot pollute the
    // "fresh engine, no embedder" precondition. With dim=64 the test
    // would race against the `bundled-embedder` cargo feature.
    let mut db = YantrikDB::new(":memory:", 384).unwrap();
    assert!(!db.has_embedder(), "fresh engine: no embedder");
    use mode_test_embedders::FakeEmbedder;
    db.set_embedder(Box::new(FakeEmbedder {
        dim: 384,
        fp: Some("sha256:x".to_string()),
        name: None,
        sentinel: 0.42,
    }))
    .unwrap();
    assert!(db.has_embedder());
    let v = db.embed("anything").unwrap();
    assert!((v[0] - 0.42).abs() < 1e-6);
}

#[test]
fn record_text_revalidates_generation_and_retries_after_swap() {
    // **Issue #41 brainstorm-4 §2 regression test.** Locks the
    // writer-revalidation invariant: when `record_text`'s engine-side
    // embed step races a SearchState swap, the writer detects the
    // generation mismatch under the WriteRouter guard and retries.
    // Without the loop, the embedding would land in the wrong vector
    // space (durable silent corruption when dims match).
    //
    // Test choreography:
    //   1. Set up db with BlockingEmbedder. SearchState publishes
    //      generation G=0 with digest "sha256:initial".
    //   2. Spawn record_text on a worker thread. It enters embed(),
    //      signals "started_tx", and blocks on release_rx.
    //   3. Test thread receives "started" signal. With the worker
    //      blocked in the embed, the test manually constructs a NEW
    //      SearchState (generation G=1, digest "sha256:rotated") and
    //      stores it via `db.search_state.store(...)` — simulating a
    //      reembed Phase-2 swap completing.
    //   4. Test thread releases the embedder. record_text returns
    //      from embed, acquires the sync guard, revalidates
    //      generation, sees mismatch (0 != 1), drops guard, loops.
    //   5. Second loop iteration loads the NEW SearchState, embeds
    //      (now returns immediately — call_count > 0 takes the
    //      no-block branch), acquires guard, revalidates, MATCH
    //      (state still G=1), commits.
    //   6. Assert call_count >= 2 (retry happened) and the recorded
    //      memory exists in SQL.
    // Test embedder wraps an Arc<AtomicUsize> so the test thread can
    // observe call_count after the worker finishes. (The
    // mode_test_embedders::BlockingEmbedder uses an in-struct
    // AtomicUsize, which is inaccessible once boxed into Arc<dyn
    // Embedder> in set_embedder.)
    use std::sync::mpsc::channel;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    let (started_tx, started_rx) = channel::<()>();
    let (release_tx, release_rx) = channel::<()>();
    let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    struct SharedBlocking {
        dim: usize,
        fp: Option<String>,
        name: Option<String>,
        sentinel: f32,
        started_tx: Mutex<Option<std::sync::mpsc::Sender<()>>>,
        release_rx: Mutex<Option<std::sync::mpsc::Receiver<()>>>,
        call_count: Arc<std::sync::atomic::AtomicUsize>,
    }
    impl crate::types::Embedder for SharedBlocking {
        fn embed(
            &self,
            _text: &str,
        ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
            let n = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                if let Some(tx) = self.started_tx.lock().unwrap().take() {
                    let _ = tx.send(());
                }
                if let Some(rx) = self.release_rx.lock().unwrap().take() {
                    let _ = rx.recv();
                }
            }
            let mut v = vec![0.0_f32; self.dim];
            if !v.is_empty() {
                v[0] = self.sentinel;
            }
            Ok(v)
        }
        fn dim(&self) -> usize {
            self.dim
        }
        fn fingerprint(&self) -> Option<String> {
            self.fp.clone()
        }
        fn name(&self) -> Option<String> {
            self.name.clone()
        }
    }

    let mut db = YantrikDB::new(":memory:", 64).unwrap();
    db.set_embedder(Box::new(SharedBlocking {
        dim: 64,
        fp: Some("sha256:initial".to_string()),
        name: Some("blocking-initial".to_string()),
        sentinel: 0.42,
        started_tx: Mutex::new(Some(started_tx)),
        release_rx: Mutex::new(Some(release_rx)),
        call_count: Arc::clone(&call_count),
    }))
    .unwrap();

    // Snapshot the initial state's generation; capture it for the
    // post-test assertion below.
    let gen_before = db.search_state.load().generation;
    let arc_db = Arc::new(db);

    // Spawn the worker: record_text() runs the revalidation loop.
    let worker_db = Arc::clone(&arc_db);
    let worker = std::thread::spawn(move || {
        worker_db
            .record_text(
                "hello",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap()
    });

    // Wait for the worker to reach the embed step.
    started_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("embed should have started within 5s");

    // Worker is now blocked in embed(). Simulate a reembed Phase-2
    // swap: construct a new SearchState with generation+1 and a
    // different digest, leaving the same vec_index (dim unchanged so
    // we don't trigger downstream dim checks). The new SearchState
    // is what reembed Phase-2 would publish; the writer's
    // revalidation loop must detect the change and re-embed.
    let old_state = arc_db.search_state.load_full();
    let new_state = crate::engine::reembed::SearchState {
        index_embedding: crate::engine::reembed::EmbeddingProvenance::Known {
            name: Some("blocking-rotated".to_string()),
            digest: "sha256:rotated".to_string(),
            dim: 64,
        },
        embedder: old_state.embedder.clone(),
        runtime_embedder_name: Some("blocking-rotated".to_string()),
        runtime_embedder_digest: Some("sha256:rotated".to_string()),
        generation: old_state.generation + 1,
        covers_through_seq: old_state.covers_through_seq,
        hnsw_m: old_state.hnsw_m,
        hnsw_ef_construction: old_state.hnsw_ef_construction,
        hnsw_ef_search: old_state.hnsw_ef_search,
        vec_index: Arc::clone(&old_state.vec_index),
    };
    arc_db.search_state.store(Arc::new(new_state));

    // Release the blocked embed so the worker proceeds: acquire
    // guard, revalidate, see mismatch, retry. The retry iteration
    // doesn't block (call_count > 0).
    release_tx.send(()).unwrap();

    let rid = worker.join().expect("record_text must complete");

    // Locks the revalidation contract:
    //   1. Embed was called at least twice (first under old state,
    //      retry under new state).
    //   2. The recorded memory exists.
    //   3. The active generation advanced (sanity check).
    let n_calls = call_count.load(std::sync::atomic::Ordering::SeqCst);
    assert!(
        n_calls >= 2,
        "record_text must re-embed after SearchState swap; got {n_calls} calls"
    );
    assert!(!rid.is_empty(), "record_text returns a valid rid");
    let gen_after = arc_db.search_state.load().generation;
    assert!(
        gen_after > gen_before,
        "test must observe a generation advance: before={gen_before} after={gen_after}"
    );
    // Verify the row is durably in SQL.
    let conn = arc_db.conn();
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE rid = ?1",
            [&rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 1, "the retried record_text must be durably stored");
}

#[test]
fn record_text_routes_to_queued_when_router_is_queueing() {
    // **Issue #41 brainstorm-4 §2 sibling case.** When reembed has
    // flipped the WriteRouter to Queueing BEFORE record_text reaches
    // the acquire step, the writer must route to the queued path
    // (which stores text and lets the post-swap materializer
    // re-encode), not retry-loop forever. Locks the "Queueing →
    // queued path" branch of the revalidation loop.
    use mode_test_embedders::FakeEmbedder;
    let mut db = YantrikDB::new(":memory:", 64).unwrap();
    db.set_embedder(Box::new(FakeEmbedder {
        dim: 64,
        fp: Some("sha256:initial".to_string()),
        name: Some("initial".to_string()),
        sentinel: 0.5,
    }))
    .unwrap();

    // Flip the router to Queueing — simulates reembed's
    // switch_to_queueing() during the cutover preamble.
    db.write_router.switch_to_queueing();

    // record_text must NOT spin in its revalidation loop — the
    // Queueing branch returns via record_queued.
    let rid = db
        .record_text(
            "hello-queued",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({}),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    assert!(!rid.is_empty());

    // The queued path does NOT write to `memories` (brainstorm-3
    // invariant 7); it writes an applied=0 op to `oplog` with
    // `embedding_model = current_runtime_embedder_name`. Verify both
    // halves of that contract.
    let conn = db.conn();
    let memories_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE rid = ?1",
            [&rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        memories_count, 0,
        "queued path must NOT write to memories table"
    );
    let oplog_row: (i64, Option<String>) = conn
        .query_row(
            "SELECT applied, embedding_model FROM oplog WHERE target_rid = ?1 AND op_type = 'record'",
            [&rid],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(oplog_row.0, 0, "queued op must be applied=0");
    assert_eq!(
        oplog_row.1.as_deref(),
        Some("initial"),
        "queued op carries the current runtime embedder name for post-swap re-encode"
    );
}

#[test]
fn log_op_stamps_applied_generation_from_active_search_state() {
    // **Issue #41 brainstorm-2 §1 / brainstorm-4 §6 regression test.**
    // Sync-write paths must stamp `oplog.applied_generation` with the
    // active SearchState generation. Without this, the post-swap
    // materializer (Layer 5) cannot discriminate "already applied
    // under old gen — skip" from "queued during reembed — need
    // re-encode" (both would show applied=0 NULL or applied=1 NULL
    // and look identical).
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let initial_generation: i64 = db.search_state.load().generation as i64;

    // log_op a synthetic event and assert the column is populated.
    let op_id = db
        .log_op("test_event", None, &serde_json::json!({"x": 1}), None)
        .unwrap();
    // Scope the conn guard tightly — log_op needs the conn lock too,
    // so don't hold it across the next log_op call (deadlock).
    let applied_generation: Option<i64> = {
        let conn = db.conn();
        conn.query_row(
            "SELECT applied_generation FROM oplog WHERE op_id = ?1",
            [&op_id],
            |r| r.get(0),
        )
        .unwrap()
    };
    assert_eq!(
        applied_generation,
        Some(initial_generation),
        "log_op must stamp applied_generation with the current SearchState generation"
    );

    // Bump the generation manually (simulates a reembed swap) and
    // verify a subsequent log_op picks up the new value.
    let old_state = db.search_state.load_full();
    let bumped = crate::engine::reembed::SearchState {
        index_embedding: old_state.index_embedding.clone(),
        embedder: old_state.embedder.clone(),
        runtime_embedder_name: old_state.runtime_embedder_name.clone(),
        runtime_embedder_digest: old_state.runtime_embedder_digest.clone(),
        generation: old_state.generation + 1,
        covers_through_seq: old_state.covers_through_seq,
        hnsw_m: old_state.hnsw_m,
        hnsw_ef_construction: old_state.hnsw_ef_construction,
        hnsw_ef_search: old_state.hnsw_ef_search,
        vec_index: std::sync::Arc::clone(&old_state.vec_index),
    };
    db.search_state.store(std::sync::Arc::new(bumped));

    let op_id2 = db
        .log_op(
            "test_event_after_bump",
            None,
            &serde_json::json!({"x": 2}),
            None,
        )
        .unwrap();
    let applied_generation2: Option<i64> = {
        let conn = db.conn();
        conn.query_row(
            "SELECT applied_generation FROM oplog WHERE op_id = ?1",
            [&op_id2],
            |r| r.get(0),
        )
        .unwrap()
    };
    assert_eq!(
        applied_generation2,
        Some(initial_generation + 1),
        "log_op picks up the new generation after search_state.store"
    );
}

#[test]
fn set_embedder_test_8_atomic_publication_no_partial_state() {
    // Locks the brainstorm-3 atomic-publication invariant: a concurrent
    // load of search_state must see either the OLD state or the NEW
    // state, never a mix. With ArcSwap, this is automatic; the test
    // exercises the property as a regression guard.
    use mode_test_embedders::FakeEmbedder;
    use std::sync::Arc;
    let mut db = YantrikDB::new(":memory:", 64).unwrap();
    db.set_embedder(Box::new(FakeEmbedder {
        dim: 64,
        fp: Some("sha256:initial".to_string()),
        name: None,
        sentinel: 1.0,
    }))
    .unwrap();
    // Self-consistency invariant: if embedder is Some, digest is also Some
    // for FakeEmbedder which always provides a fingerprint.
    let state = db.search_state.load_full();
    assert_eq!(
        state.embedder.is_some(),
        state.runtime_embedder_digest.is_some(),
        "embedder Some <=> digest Some must hold for any consistent snapshot"
    );
    // Multiple same-digest replacements; each load must be consistent.
    for sentinel in [2.0_f32, 3.0, 4.0, 5.0] {
        db.set_embedder(Box::new(FakeEmbedder {
            dim: 64,
            fp: Some("sha256:initial".to_string()),
            name: None,
            sentinel,
        }))
        .unwrap();
        let s = db.search_state.load_full();
        assert_eq!(
            s.embedder.is_some(),
            s.runtime_embedder_digest.is_some(),
            "consistency must hold across replacements (no partial state)"
        );
        let _arc_held: Arc<crate::engine::reembed::SearchState> = s;
    }
}

#[test]
fn search_state_initial_on_fresh_engine() {
    // Issue #41 layer 2: fresh engine must initialize a SearchState with
    // provenance=ExternalOrUnknown(embedding_dim) and no runtime embedder.
    // The standalone embedding_dim field is still source of truth at THIS
    // layer (retired in a later checkpoint); we only verify here that the
    // new search_state field is initialized with the expected initial shape.
    let db = YantrikDB::new(":memory:", 384).unwrap();
    let state = db.search_state.load_full();
    assert_eq!(
        state.dim(),
        384,
        "initial dim must match constructor parameter"
    );
    assert!(matches!(
        state.index_embedding,
        crate::engine::reembed::EmbeddingProvenance::ExternalOrUnknown { dim: 384 }
    ));
    assert!(
        !state.has_runtime_embedder(),
        "fresh engine must have no runtime embedder until set_embedder*"
    );
    assert_eq!(state.generation, 0);
    assert_eq!(state.covers_through_seq, 0);
}

#[test]
fn schema_v27_fresh_install_has_reembed_surfaces() {
    // Fresh DB takes the SCHEMA_SQL path. Locks the invariant that
    // SCHEMA_SQL stays in sync with MIGRATE_V26_TO_V27. If someone adds
    // a column to one but not the other, this test catches the drift
    // before it ships (same shape as the v26 fresh-install test).
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let conn = db.conn();

    let memories_cols = table_columns(&conn, "memories");
    for required in ["embedding_new", "embedding_new_model"] {
        assert!(
            memories_cols.iter().any(|c| c == required),
            "v27: fresh-install memories table missing column {required}, got: {memories_cols:?}"
        );
    }

    let oplog_cols = table_columns(&conn, "oplog");
    assert!(
        oplog_cols.iter().any(|c| c == "embedding_model"),
        "v27: fresh-install oplog table missing column embedding_model, got: {oplog_cols:?}"
    );
    assert!(
        oplog_cols.iter().any(|c| c == "applied_generation"),
        "v27: fresh-install oplog table missing column applied_generation \
         (brainstorm-2 correction \u{2014} per-generation application tracking \
         replaces boolean `applied` as truth), got: {oplog_cols:?}"
    );

    // reembed_events table exists with the right shape
    let events_cols = table_columns(&conn, "reembed_events");
    for required in ["generation", "phase", "timestamp", "payload_json"] {
        assert!(
            events_cols.iter().any(|c| c == required),
            "v27: fresh-install reembed_events missing column {required}, got: {events_cols:?}"
        );
    }

    for required_idx in [
        "idx_reembed_events_generation",
        "idx_oplog_applied_generation",
    ] {
        assert!(
            index_exists(&conn, required_idx),
            "v27: fresh-install missing index {required_idx}"
        );
    }
}

#[test]
fn schema_v27_migration_from_v26_is_additive_only() {
    // Plant a row under v27 schema, then manually rewind meta to 26 and
    // re-open to trigger MIGRATE_V26_TO_V27. Verify:
    //   - the row is untouched (additive migration, no data mutation)
    //   - new columns appear as NULL
    //   - new table exists + is writable
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    let planted_rid = "01900000-0000-7000-8000-00000000c027";
    let planted_embedding = vec![0_u8; 8 * std::mem::size_of::<f32>()];
    {
        let db = YantrikDB::new(path, 8).unwrap();
        let conn = db.conn();
        conn.execute(
            "INSERT INTO memories (rid, type, text, embedding, created_at, updated_at, last_access, source) \
             VALUES (?1, 'episodic', 'planted under v27 schema', ?2, 0.0, 0.0, 0.0, 'user')",
            params![planted_rid, planted_embedding],
        )
        .unwrap();
    }

    // Rewind meta + drop v27 surfaces so the migration recreates them.
    {
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '26')",
            [],
        )
        .unwrap();
        conn.execute("DROP TABLE IF EXISTS reembed_events", [])
            .unwrap();
        conn.execute("DROP INDEX IF EXISTS idx_reembed_events_generation", [])
            .unwrap();
        // Can't easily drop ALTER-added columns in SQLite without table
        // rebuild; the idempotent runner swallows the duplicate-column
        // errors on the ALTER re-run instead.
    }

    let db = YantrikDB::new(path, 8)
        .expect("v27 migration must run cleanly against a rewound-meta v26 DB");
    let conn = db.conn();

    // Row preserved untouched (the migration must not touch existing data)
    let (preserved_text, embedding_new): (String, Option<Vec<u8>>) = conn
        .query_row(
            "SELECT text, embedding_new FROM memories WHERE rid = ?1",
            params![planted_rid],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<Vec<u8>>>(1)?)),
        )
        .unwrap();
    assert_eq!(
        preserved_text, "planted under v27 schema",
        "v27 migration must NOT mutate existing memory data"
    );
    assert!(
        embedding_new.is_none(),
        "v27 migration must leave embedding_new as NULL on pre-existing rows"
    );

    assert!(
        index_exists(&conn, "idx_reembed_events_generation"),
        "v27 migration must recreate idx_reembed_events_generation"
    );

    // Verify the reembed_events table is writable + readable end-to-end
    conn.execute(
        "INSERT INTO reembed_events (generation, phase, timestamp, payload_json) \
         VALUES (?1, ?2, ?3, ?4)",
        params![1_i64, "Probing", 0.0_f64, "{}"],
    )
    .unwrap();
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM reembed_events WHERE generation = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        count, 1,
        "reembed_events table must be writable post-migration"
    );
}

#[test]
fn schema_v27_migration_replay_is_idempotent() {
    // Same shape as the v26 replay test. Rewind meta to 26 on an
    // already-v27 DB and verify the second open heals cleanly. The
    // idempotent runner swallows the duplicate-column errors that the
    // ALTER TABLE statements would raise on the second pass.
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    {
        let _db = YantrikDB::new(path, 8).unwrap();
    }
    {
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '26')",
            [],
        )
        .unwrap();
    }
    let db = YantrikDB::new(path, 8)
        .expect("v27 migration runner must heal rewound-meta deployments on a v27-schema DB");

    db.record(
        "post-v27-heal smoke",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(1.0, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();
}

// ─────────────────────────────────────────────────────────────────
// Issue #41 brainstorm-4 §3 — monotonic-generation CAS regression
// tests. Locks try_publish_search_state's two guarantees: stale
// publishes are rejected, equal-generation publishes are allowed
// (set_embedder runtime-Arc-swap case).
// ─────────────────────────────────────────────────────────────────

#[test]
fn try_publish_search_state_rejects_stale_generation() {
    // Construct a SearchState at generation N, advance the engine
    // to generation N+2 via direct store, then attempt to publish
    // the N-generation state. The helper must reject with
    // SearchStatePublishStaleGeneration — without this, a stale
    // compactor/reembed step could ABA-rollback the active
    // generation.
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let initial = db.search_state.load_full();
    assert_eq!(initial.generation, 0, "fresh engine starts at gen 0");

    // Build a "stale" SearchState replica of gen=0.
    let stale_proposal = crate::engine::reembed::SearchState {
        index_embedding: initial.index_embedding.clone(),
        embedder: initial.embedder.clone(),
        runtime_embedder_name: initial.runtime_embedder_name.clone(),
        runtime_embedder_digest: initial.runtime_embedder_digest.clone(),
        generation: 0,
        covers_through_seq: 0,
        hnsw_m: initial.hnsw_m,
        hnsw_ef_construction: initial.hnsw_ef_construction,
        hnsw_ef_search: initial.hnsw_ef_search,
        vec_index: std::sync::Arc::clone(&initial.vec_index),
    };

    // Manually advance the engine to gen=2 (simulates two
    // back-to-back reembed Phase-2 swaps).
    let advanced = crate::engine::reembed::SearchState {
        index_embedding: initial.index_embedding.clone(),
        embedder: initial.embedder.clone(),
        runtime_embedder_name: initial.runtime_embedder_name.clone(),
        runtime_embedder_digest: initial.runtime_embedder_digest.clone(),
        generation: 2,
        covers_through_seq: 0,
        hnsw_m: initial.hnsw_m,
        hnsw_ef_construction: initial.hnsw_ef_construction,
        hnsw_ef_search: initial.hnsw_ef_search,
        vec_index: std::sync::Arc::clone(&initial.vec_index),
    };
    db.search_state.store(std::sync::Arc::new(advanced));

    // Try to publish the stale gen=0 state. Helper must reject.
    let err = db
        .try_publish_search_state(stale_proposal)
        .expect_err("stale-generation publish must be rejected");
    match err {
        crate::error::YantrikDbError::SearchStatePublishStaleGeneration {
            current_generation,
            attempted_generation,
        } => {
            assert_eq!(current_generation, 2);
            assert_eq!(attempted_generation, 0);
        }
        other => panic!("unexpected error variant: {other:?}"),
    }

    // The engine state must still be at gen=2 — the rejected
    // publish did NOT mutate the ArcSwap.
    assert_eq!(
        db.search_state.load().generation,
        2,
        "rejected publish must leave search_state untouched"
    );
}

#[test]
fn try_publish_search_state_accepts_equal_generation_publish() {
    // set_embedder publishes with `new.generation == current.generation`
    // (runtime-Arc swap, no vector-space change). The CAS helper must
    // accept this — strict less-than is the only rejection condition.
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let initial = db.search_state.load_full();
    let same_gen = crate::engine::reembed::SearchState {
        index_embedding: initial.index_embedding.clone(),
        embedder: initial.embedder.clone(),
        // Differ in some runtime-only field so the test verifies the
        // publish actually landed (vs being a silent no-op).
        runtime_embedder_name: Some("rotated-name".to_string()),
        runtime_embedder_digest: initial.runtime_embedder_digest.clone(),
        generation: initial.generation,
        covers_through_seq: initial.covers_through_seq,
        hnsw_m: initial.hnsw_m,
        hnsw_ef_construction: initial.hnsw_ef_construction,
        hnsw_ef_search: initial.hnsw_ef_search,
        vec_index: std::sync::Arc::clone(&initial.vec_index),
    };
    db.try_publish_search_state(same_gen)
        .expect("equal-generation publish must be accepted");
    assert_eq!(
        db.search_state.load().runtime_embedder_name.as_deref(),
        Some("rotated-name"),
        "the equal-generation publish must have landed"
    );
}

#[test]
fn try_publish_search_state_accepts_strictly_advancing_generation() {
    // Reembed Phase-2 publishes new.generation = current.generation + 1.
    // Lock the success path so the brainstorm-4 §3 CAS doesn't
    // accidentally over-reject and break the reembed swap.
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let initial = db.search_state.load_full();
    let advanced = crate::engine::reembed::SearchState {
        index_embedding: initial.index_embedding.clone(),
        embedder: initial.embedder.clone(),
        runtime_embedder_name: initial.runtime_embedder_name.clone(),
        runtime_embedder_digest: initial.runtime_embedder_digest.clone(),
        generation: initial.generation + 1,
        covers_through_seq: initial.covers_through_seq,
        hnsw_m: initial.hnsw_m,
        hnsw_ef_construction: initial.hnsw_ef_construction,
        hnsw_ef_search: initial.hnsw_ef_search,
        vec_index: std::sync::Arc::clone(&initial.vec_index),
    };
    db.try_publish_search_state(advanced)
        .expect("strictly-advancing-generation publish must be accepted");
    assert_eq!(
        db.search_state.load().generation,
        initial.generation + 1,
        "the advanced publish must have landed"
    );
}

// ─────────────────────────────────────────────────────────────────
// Issue #41 brainstorm-4 §6 — v28 durable-linearization regression
// tests. Locks (a) fresh install + migration both produce the v28
// surfaces, (b) record/record_batch/record_with_rid stamp the new
// column with state.generation, (c) open() reads the durable
// meta.active_generation back into SearchState.
// ─────────────────────────────────────────────────────────────────

#[test]
fn schema_v28_fresh_install_has_embedding_generation_and_active_generation() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let conn = db.conn();

    let cols = table_columns(&conn, "memories");
    assert!(
        cols.iter().any(|c| c == "embedding_generation"),
        "v28: fresh install must add memories.embedding_generation column, got: {cols:?}"
    );

    assert!(
        index_exists(&conn, "idx_memories_embedding_generation"),
        "v28: fresh install must create idx_memories_embedding_generation"
    );

    let active_gen: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'active_generation'",
            [],
            |r| r.get(0),
        )
        .ok();
    assert_eq!(
        active_gen.as_deref(),
        Some("0"),
        "v28: fresh install must seed meta.active_generation = '0'"
    );

    // Schema version stamp is at v28.
    let schema_version: String = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        schema_version,
        crate::base::schema::SCHEMA_VERSION.to_string(),
        "fresh install stamps SCHEMA_VERSION"
    );
}

#[test]
fn schema_v28_migration_from_v27_is_additive_and_idempotent() {
    // Plant a row under v28 schema, then rewind meta.schema_version to 27
    // and re-open to trigger MIGRATE_V27_TO_V28. Verify:
    //   - existing row untouched (additive migration)
    //   - embedding_generation column still present (won't error on duplicate)
    //   - meta.active_generation still '0' (INSERT OR IGNORE preserves)
    //   - re-open succeeds (the run_migration_idempotent runner swallows
    //     "duplicate column name" on the ALTER TABLE ADD COLUMN)
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    let planted_rid = "01900000-0000-7000-8000-00000000c028";
    let planted_embedding = vec![0_u8; 8 * std::mem::size_of::<f32>()];
    {
        let db = YantrikDB::new(path, 8).unwrap();
        let conn = db.conn();
        conn.execute(
            "INSERT INTO memories (rid, type, text, embedding, created_at, updated_at, last_access, source, embedding_generation) \
             VALUES (?1, 'episodic', 'planted under v28 schema', ?2, 0.0, 0.0, 0.0, 'user', 42)",
            params![planted_rid, planted_embedding],
        )
        .unwrap();
    }

    // Rewind meta to 27 to force the v27 -> v28 migration replay path.
    {
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '27')",
            [],
        )
        .unwrap();
    }

    // Re-open — runner must heal idempotently.
    let db =
        YantrikDB::new(path, 8).expect("v28 migration runner must heal rewound-meta deployments");
    let conn = db.conn();

    // Planted row still there with original generation stamp.
    let (text, gen): (String, i64) = conn
        .query_row(
            "SELECT text, embedding_generation FROM memories WHERE rid = ?1",
            [&planted_rid],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(text, "planted under v28 schema");
    assert_eq!(gen, 42, "migration must not mutate existing row data");

    // Schema version stamped back to the current SCHEMA_VERSION.
    let schema_version: String = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        schema_version,
        crate::base::schema::SCHEMA_VERSION.to_string()
    );

    // meta.active_generation preserved (INSERT OR IGNORE didn't clobber).
    let active_gen: String = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'active_generation'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(active_gen, "0");
}

#[test]
fn record_stamps_embedding_generation_from_search_state() {
    // **Brainstorm-4 §6 row-level invariant.** Every sync-path insert
    // stamps memories.embedding_generation = state.generation. Phase-2
    // swap (when it lands) advances state.generation, and the post-swap
    // materializer's scan uses this column to find rows that need
    // re-encode. If the stamp is wrong, the scan returns the wrong
    // population.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "stamped",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    let conn = db.conn();
    let stamped: i64 = conn
        .query_row(
            "SELECT embedding_generation FROM memories WHERE rid = ?1",
            [&rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(stamped, 0, "fresh engine: state.generation = 0, stamp = 0");

    drop(conn);

    // Manually advance the engine SearchState to gen=7 (simulates a
    // future Phase-2 swap publishing). Subsequent record must stamp 7.
    let old_state = db.search_state.load_full();
    let advanced = crate::engine::reembed::SearchState {
        index_embedding: old_state.index_embedding.clone(),
        embedder: old_state.embedder.clone(),
        runtime_embedder_name: old_state.runtime_embedder_name.clone(),
        runtime_embedder_digest: old_state.runtime_embedder_digest.clone(),
        generation: 7,
        covers_through_seq: old_state.covers_through_seq,
        hnsw_m: old_state.hnsw_m,
        hnsw_ef_construction: old_state.hnsw_ef_construction,
        hnsw_ef_search: old_state.hnsw_ef_search,
        vec_index: std::sync::Arc::clone(&old_state.vec_index),
    };
    db.try_publish_search_state(advanced).unwrap();

    let rid2 = db
        .record(
            "stamped at gen 7",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(2.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    let conn = db.conn();
    let stamped2: i64 = conn
        .query_row(
            "SELECT embedding_generation FROM memories WHERE rid = ?1",
            [&rid2],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        stamped2, 7,
        "record after generation advance must stamp the new generation"
    );
}

#[test]
fn open_reads_durable_active_generation_into_search_state() {
    // **Brainstorm-4 §6 durable linearization point.** open() must
    // read meta.active_generation and initialize SearchState.generation
    // from it. Without this, crash recovery between Phase-2's SQL
    // swap-commit and the ArcSwap store would leave the in-memory
    // SearchState at the OLD generation while SQL claims the NEW
    // generation — split-brain on restart.
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    // First open at fresh install — initial gen = 0.
    {
        let db = YantrikDB::new(path, 8).unwrap();
        assert_eq!(db.search_state.load().generation, 0);
    }

    // Simulate Phase-2's SQL commit (without the matching in-memory
    // store) by manually updating meta.active_generation = 3 in SQL.
    {
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('active_generation', '3')",
            [],
        )
        .unwrap();
    }

    // Re-open. SearchState.generation must come back as 3.
    let db = YantrikDB::new(path, 8).unwrap();
    assert_eq!(
        db.search_state.load().generation,
        3,
        "open() must read meta.active_generation into SearchState.generation"
    );
}

#[test]
fn set_embedder_routes_through_try_publish_search_state() {
    // Coverage check: confirm set_embedder calls go through the
    // CAS helper. Done indirectly by verifying that after
    // set_embedder, the search_state generation is preserved (the
    // helper would reject any rogue decrement). This locks the
    // call-graph routing — if a future refactor reintroduces a
    // direct `self.search_state.store(...)` from set_embedder, the
    // brainstorm-4 §3 invariant is no longer load-bearing.
    use mode_test_embedders::FakeEmbedder;
    let mut db = YantrikDB::new(":memory:", 64).unwrap();
    let gen_before = db.search_state.load().generation;
    db.set_embedder(Box::new(FakeEmbedder {
        dim: 64,
        fp: Some("sha256:check".to_string()),
        name: Some("check".to_string()),
        sentinel: 0.1,
    }))
    .unwrap();
    let gen_after = db.search_state.load().generation;
    assert_eq!(
        gen_before, gen_after,
        "set_embedder must preserve generation (only Phase-2 reembed advances it)"
    );
    assert!(db.has_embedder(), "set_embedder must have published");
}

// ─────────────────────────────────────────────────────────────────
// Issue #41 brainstorm-4 §10 — remaining regression tests.
// Items #2, #3, #4, #7, #8 are locked in checkpoints 11-14. The
// 5 tests below close items #1, #5, #6, plus a meta-test for the
// boundary audit logic and a Queue-mode round-trip.
// ─────────────────────────────────────────────────────────────────

#[test]
fn search_state_publish_is_atomic_under_concurrent_reads() {
    // **Brainstorm-4 §10.1 — no read-side dim split-brain.** Spawn
    // many reader threads that capture `search_state.load_full()`
    // and inspect *all* fields of the snapshot. Concurrently, the
    // main thread publishes N alternating SearchStates. Each
    // reader's snapshot must be consistent — every field comes
    // from the same generation. Without SearchState as the single
    // atomic publication unit (brainstorm-4 §1), readers could
    // observe new provenance + old embedder + old vec_index. The
    // ArcSwap guarantees the atomic flip; this test locks the
    // invariant.
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;

    let db = Arc::new(YantrikDB::new(":memory:", 64).unwrap());
    let initial = db.search_state.load_full();
    let stop = Arc::new(AtomicBool::new(false));

    // Reader threads — each captures one snapshot per iteration
    // and asserts the (generation, provenance.dim, hnsw_m) tuple
    // is one of the published combinations, never a mix.
    let mut handles = Vec::new();
    for _ in 0..4 {
        let db_c = Arc::clone(&db);
        let stop_c = Arc::clone(&stop);
        handles.push(thread::spawn(move || {
            let mut observations: Vec<(u64, usize, u32)> = Vec::new();
            while !stop_c.load(Ordering::Relaxed) {
                let s = db_c.search_state.load_full();
                observations.push((s.generation, s.dim(), s.hnsw_m));
            }
            observations
        }));
    }

    // Publish 50 alternating SearchStates, each consistent within
    // itself. Use try_publish_search_state so generation is
    // monotonic-advanced (each call bumps by 1).
    for n in 1..=50u64 {
        let prev = db.search_state.load_full();
        let next = crate::engine::reembed::SearchState {
            index_embedding: prev.index_embedding.clone(),
            embedder: prev.embedder.clone(),
            runtime_embedder_name: prev.runtime_embedder_name.clone(),
            runtime_embedder_digest: prev.runtime_embedder_digest.clone(),
            generation: prev.generation + 1,
            covers_through_seq: prev.covers_through_seq + n,
            // hnsw_m alternates so we can detect a torn read
            // (would manifest as gen even / hnsw_m odd or vice versa).
            hnsw_m: if n % 2 == 0 { 16 } else { 32 },
            hnsw_ef_construction: prev.hnsw_ef_construction,
            hnsw_ef_search: prev.hnsw_ef_search,
            vec_index: std::sync::Arc::clone(&prev.vec_index),
        };
        db.try_publish_search_state(next).unwrap();
    }

    stop.store(true, Ordering::Relaxed);

    let baseline = (initial.generation, initial.dim(), initial.hnsw_m);
    for handle in handles {
        let observations = handle.join().unwrap();
        for (gen, dim, hnsw_m) in &observations {
            // Two valid forms per generation: even-gen → hnsw_m=16,
            // odd-gen → hnsw_m=32. (Or the initial baseline.)
            let consistent = (*gen, *dim, *hnsw_m) == baseline
                || (*gen >= 1
                    && *dim == initial.dim()
                    && ((*gen % 2 == 0 && *hnsw_m == 16) || (*gen % 2 == 1 && *hnsw_m == 32)));
            assert!(
                consistent,
                "torn SearchState observation: gen={gen} dim={dim} hnsw_m={hnsw_m} \
                 (expected even-gen→16 or odd-gen→32, or baseline)"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Issue #41 Layer 7 — crash-recovery regression tests. Two branches
// of the recovery decision on open():
//   (1) active_generation < in_flight_generation
//       → SQL swap did NOT commit; discard staging.
//   (2) active_generation >= in_flight_generation
//       → SQL swap DID commit; SearchState rebuilds at new gen.
// ─────────────────────────────────────────────────────────────────

#[test]
fn open_recovery_discards_staging_when_sql_swap_uncommitted() {
    // **Layer 7 — branch 1.** Simulate a crash during Encoding/
    // Rebuilding/Swapping BEFORE the SQL swap transaction
    // committed. Plant: meta.reembed_state with gen=5,
    // phase='Encoding', AND populated embedding_new columns,
    // AND meta.active_generation still at '0' (the swap commit
    // never happened).
    //
    // Expected on open():
    //   - meta.reembed_state cleared
    //   - embedding_new + embedding_new_model NULLed
    //   - active_generation unchanged at 0
    //   - SearchState.generation = 0
    //   - reembed_events has an Aborted event for gen 5 with
    //     recovery="discarded_staging"
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    let planted_rid = "01900000-0000-7000-8000-00000000d017";
    {
        let db = YantrikDB::new(path, 8).unwrap();
        db.record(
            "pre-reembed row",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(0.5, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
        // Plant the staging columns + the in-flight reembed_state
        // directly so we don't need the full reembed machinery.
        let conn = db.conn();
        conn.execute(
            "UPDATE memories SET embedding_new = X'AABBCCDD', \
             embedding_new_model = 'simulated-target' WHERE rowid = 1",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('reembed_state', ?1)",
            params![serde_json::json!({
                "generation": 5,
                "phase": "Encoding",
                "old_embedder": "old",
                "new_embedder_name": "simulated-target",
            })
            .to_string()],
        )
        .unwrap();
        // active_generation still '0' (the swap commit never ran).
        let _ = planted_rid;
    }

    // Re-open: Layer 7 recovery decides "discard staging".
    let db = YantrikDB::new(path, 8).unwrap();
    let conn = db.conn();

    // meta.reembed_state cleared.
    let still_in_flight: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'reembed_state'",
            [],
            |r| r.get(0),
        )
        .ok();
    assert!(
        still_in_flight.is_none(),
        "Layer 7 must clear meta.reembed_state on uncommitted-swap recovery; got: {still_in_flight:?}"
    );

    // Staging NULLed.
    let staged: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE embedding_new IS NOT NULL OR \
             embedding_new_model IS NOT NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        staged, 0,
        "staging columns must be NULL after discard recovery"
    );

    // active_generation unchanged.
    let active: String = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'active_generation'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(active, "0");

    // SearchState.generation = 0.
    assert_eq!(db.search_state.load().generation, 0);

    // Aborted recovery event present.
    let aborted_recovery: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM reembed_events WHERE phase = 'Aborted' AND generation = 5 \
             AND payload_json LIKE '%discarded_staging%'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(aborted_recovery, 1, "Aborted recovery event must be logged");
}

#[test]
fn open_recovery_durable_swap_resumes_at_new_generation() {
    // **Layer 7 — branch 2.** Simulate a crash AFTER the SQL swap
    // committed but before the in-memory ArcSwap store landed
    // (the §10.4 case). Plant: meta.active_generation = '3'
    // (durable swap done) AND meta.reembed_state with gen=3
    // (the in-flight marker that never got cleared).
    //
    // Expected on open():
    //   - meta.reembed_state cleared
    //   - SearchState.generation = 3 (durable + rebuilt)
    //   - active_generation = '3' (unchanged)
    //   - Staging defensively cleared (should already be empty)
    //   - reembed_events has a Completed event for gen 3 with
    //     recovery="completed_durable"
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    {
        let db = YantrikDB::new(path, 8).unwrap();
        let _ = db.record(
            "pre-reembed row",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(0.5, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        );
        // Simulate: swap COMMITTED (active_generation bumped to 3,
        // row's embedding_generation stamped 3) but the matching
        // in-memory publish never landed AND meta.reembed_state
        // marker still says "we're in flight at gen 3".
        let conn = db.conn();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('active_generation', '3')",
            [],
        )
        .unwrap();
        conn.execute(
            "UPDATE memories SET embedding_generation = 3 WHERE embedding IS NOT NULL",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('reembed_state', ?1)",
            params![serde_json::json!({
                "generation": 3,
                "phase": "Swapping",
                "old_embedder": "old",
                "new_embedder_name": "new",
            })
            .to_string()],
        )
        .unwrap();
    }

    let db = YantrikDB::new(path, 8).unwrap();
    let conn = db.conn();

    // meta.reembed_state cleared.
    let still_in_flight: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'reembed_state'",
            [],
            |r| r.get(0),
        )
        .ok();
    assert!(
        still_in_flight.is_none(),
        "Layer 7 must clear meta.reembed_state on durable-swap recovery"
    );

    // SearchState.generation = 3.
    assert_eq!(
        db.search_state.load().generation,
        3,
        "SearchState rebuilds at durable active generation"
    );

    // active_generation preserved.
    let active: String = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'active_generation'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(active, "3");

    // Completed recovery event present.
    let completed_recovery: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM reembed_events WHERE phase = 'Completed' AND generation = 3 \
             AND payload_json LIKE '%completed_durable%'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        completed_recovery, 1,
        "Completed recovery event must be logged"
    );
}

#[test]
fn open_with_uncommitted_staging_columns_stays_at_old_generation() {
    // **Brainstorm-4 §10.5 — crash before SQL promotion commit.**
    // Simulate a Phase-2 Encoding run that wrote to
    // memories.embedding_new BUT crashed before the swap
    // SAVEPOINT committed (meta.active_generation still records
    // the old generation). open() must read the OLD generation
    // and ignore the staged columns — promoting would mix old and
    // new vector spaces.
    //
    // The staged rows themselves are not corrupted because
    // embedding_new is in its own column; the active `embedding`
    // column still carries old-generation bytes. A subsequent
    // reembed call will overwrite embedding_new and run to
    // completion.
    use tempfile::NamedTempFile;
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    // Establish a baseline row + active_generation=0.
    let planted_rid = {
        let db = YantrikDB::new(path, 8).unwrap();
        db.record(
            "pre-reembed row",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(0.5, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap()
    };

    // Simulate a partial Phase-2 Encoding: write to embedding_new
    // and embedding_new_model on the planted row, but do NOT
    // bump meta.active_generation (the commit never happened).
    {
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.execute(
            "UPDATE memories SET embedding_new = X'AABBCCDD', \
             embedding_new_model = 'simulated-new-embedder' WHERE rid = ?1",
            params![planted_rid],
        )
        .unwrap();
    }

    // Re-open. SearchState.generation must STILL be 0 (no
    // promotion happened). The staged columns are present but
    // not active.
    let db = YantrikDB::new(path, 8).unwrap();
    assert_eq!(
        db.search_state.load().generation,
        0,
        "open() must not promote partial staging into the active generation"
    );

    let conn = db.conn();
    let staged_present: bool = conn
        .query_row(
            "SELECT embedding_new IS NOT NULL FROM memories WHERE rid = ?1",
            [&planted_rid],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        staged_present,
        "staged column survives the open (Phase-2 resume logic decides what to do with it)"
    );

    // The active embedding column is unchanged — readers see the
    // pre-reembed bytes (gen 0).
    let active_present: bool = conn
        .query_row(
            "SELECT embedding IS NOT NULL FROM memories WHERE rid = ?1",
            [&planted_rid],
            |r| r.get(0),
        )
        .unwrap();
    assert!(
        active_present,
        "pre-reembed active embedding bytes preserved"
    );

    // Row's embedding_generation is the pre-reembed value (0).
    let row_gen: i64 = conn
        .query_row(
            "SELECT embedding_generation FROM memories WHERE rid = ?1",
            [&planted_rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        row_gen, 0,
        "row's stamped generation unchanged by partial staging"
    );
}

#[test]
fn covers_through_seq_is_durably_carried_on_published_search_state() {
    // **Brainstorm-4 §10.6 — covers_through_seq invariant.** Phase 2's
    // cutover captures `vec_seq.load(Acquire)` at the barrier and
    // stamps it into `SearchState.covers_through_seq` for the new
    // generation. The post-swap materializer uses this to decide
    // which oplog ops still need replay (those with seq >
    // covers_through_seq). Locks the struct-level invariant that
    // try_publish_search_state preserves the value verbatim.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let initial = db.search_state.load_full();
    assert_eq!(initial.covers_through_seq, 0, "fresh engine: covers 0");

    // Simulate a Phase-2 cutover that captured vec_seq high-water
    // mark = 12345 and published a new generation with that
    // coverage.
    let next = crate::engine::reembed::SearchState {
        index_embedding: initial.index_embedding.clone(),
        embedder: initial.embedder.clone(),
        runtime_embedder_name: initial.runtime_embedder_name.clone(),
        runtime_embedder_digest: initial.runtime_embedder_digest.clone(),
        generation: initial.generation + 1,
        covers_through_seq: 12345,
        hnsw_m: initial.hnsw_m,
        hnsw_ef_construction: initial.hnsw_ef_construction,
        hnsw_ef_search: initial.hnsw_ef_search,
        vec_index: std::sync::Arc::clone(&initial.vec_index),
    };
    db.try_publish_search_state(next).unwrap();
    assert_eq!(
        db.search_state.load().covers_through_seq,
        12345,
        "published covers_through_seq must be readable from the active SearchState"
    );

    // covers_through_seq is independent of generation — verify by
    // publishing again with a DIFFERENT covers_through_seq at the
    // SAME generation increment shape (different runtime metadata
    // but same gen advance pattern).
    let bumped = crate::engine::reembed::SearchState {
        index_embedding: initial.index_embedding.clone(),
        embedder: initial.embedder.clone(),
        runtime_embedder_name: initial.runtime_embedder_name.clone(),
        runtime_embedder_digest: initial.runtime_embedder_digest.clone(),
        generation: initial.generation + 2,
        covers_through_seq: 98765,
        hnsw_m: initial.hnsw_m,
        hnsw_ef_construction: initial.hnsw_ef_construction,
        hnsw_ef_search: initial.hnsw_ef_search,
        vec_index: std::sync::Arc::clone(&initial.vec_index),
    };
    db.try_publish_search_state(bumped).unwrap();
    assert_eq!(
        db.search_state.load().covers_through_seq,
        98765,
        "covers_through_seq advances per swap"
    );
}

#[test]
fn record_text_round_trip_through_queue_path_under_reembed() {
    // **Brainstorm-2 invariant 8 + brainstorm-4 §2 sibling test.**
    // Full round-trip of the queued write path: record_text with
    // the WriteRouter in Queueing state stores TEXT in oplog
    // (applied=0) with embedding_model set to the active runtime
    // embedder. The pre-computed embedding is intentionally
    // discarded — when Phase-2 / Layer-5 materializer drains the
    // op, it re-encodes the text under the NEW embedder.
    //
    // This test locks the integration shape end-to-end:
    //   - record_text → embed → router check → queue path
    //   - oplog row carries applied=0, applied_generation=NULL,
    //     embedding_model=<current name>, payload text intact
    //   - memories table is NOT written (post-swap materializer
    //     is responsible for that under the new generation)
    use mode_test_embedders::FakeEmbedder;

    let mut db = YantrikDB::new(":memory:", 64).unwrap();
    db.set_embedder(Box::new(FakeEmbedder {
        dim: 64,
        fp: Some("sha256:queued-test".to_string()),
        name: Some("queued-test-embedder".to_string()),
        sentinel: 0.7,
    }))
    .unwrap();

    // Reembed flips router to Queueing during the cutover preamble.
    db.write_router.switch_to_queueing();

    let rid = db
        .record_text(
            "queued-round-trip-text",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({"k": "v"}),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    let conn = db.conn();
    // memories table NOT written.
    let mem_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE rid = ?1",
            [&rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(mem_count, 0, "queued write does not touch memories table");

    // Oplog has the queued record op.
    let (op_type, applied, applied_generation, embedding_model, payload): (
        String,
        i64,
        Option<i64>,
        Option<String>,
        String,
    ) = conn
        .query_row(
            "SELECT op_type, applied, applied_generation, embedding_model, payload \
             FROM oplog WHERE target_rid = ?1 AND op_type = 'record'",
            [&rid],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )
        .unwrap();
    assert_eq!(op_type, "record");
    assert_eq!(applied, 0, "queued op is applied=0");
    assert_eq!(
        applied_generation, None,
        "queued op has applied_generation=NULL (post-swap materializer fills under new gen)"
    );
    assert_eq!(
        embedding_model.as_deref(),
        Some("queued-test-embedder"),
        "queued op carries the active runtime embedder name"
    );
    let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
    assert_eq!(
        v["text"].as_str(),
        Some("queued-round-trip-text"),
        "payload preserves the original text for re-encode"
    );
}

#[test]
fn boundary_audit_pattern_detects_synthetic_violation() {
    // **Meta-test for the brainstorm-4 §5 boundary audit logic.**
    // The audit in `engine::durable_embeddings::tests::
    // recall_rs_has_no_raw_sql_embedding_reads` greps recall.rs at
    // test-build time. If the pattern logic is broken, the audit
    // could silently pass even on a violation. This test
    // exercises the SAME pattern logic against a synthetic
    // string that DOES contain a forbidden pattern, asserting
    // the detector catches it.
    let synthetic_violation =
        "    let sql = \"SELECT rid, embedding FROM memories WHERE rid = ?1\";";
    let lower = synthetic_violation.to_ascii_lowercase();
    let patterns = [
        "select embedding ",
        "select embedding,",
        "select embedding\"",
        "select embedding\\",
        ", embedding ",
        ", embedding,",
        ", embedding\"",
        ", embedding\\",
        ", embedding\n",
    ];
    let any_match = patterns.iter().any(|p| lower.contains(p));
    assert!(
        any_match,
        "boundary audit pattern must catch the synthetic raw-SQL-embedding pattern; \
         if this asserts, the audit in durable_embeddings.rs is letting violations slip"
    );

    // And the inverse: an allowlist-safe pattern (suffixed name)
    // does NOT match.
    let allowlist_safe = "    let sql = \"SELECT rid, embedding_hash FROM memories\";";
    let lower_safe = allowlist_safe.to_ascii_lowercase();
    let safe_match = patterns.iter().any(|p| lower_safe.contains(p));
    assert!(
        !safe_match,
        "audit must NOT flag the allowlist-safe `embedding_hash` pattern; \
         the audit over-rejects which would prevent legitimate refactors"
    );
}

// =====================================================================
// 2026-08-17: the three ReembedOptions knobs that are ACCEPTED but NOT
// IMPLEMENTED must fail loudly.
//
// The 2026-08-15 knob audit found `write_policy: Pause` and
// `resume_from_checkpoint` in that state and made them error — but pinned
// neither, and MISSED `namespace`, which was the worst of the three: it was
// echoed into every progress event and the durable status while applying to
// no query, so a caller asking to re-embed one namespace silently re-embedded
// the whole store and got told it had done what it asked.
//
// One test per knob, because "the audit fixed the instances it happened to
// look at" is exactly how the third one survived.
// =====================================================================

fn reembed_err(db: &YantrikDB, opts: crate::engine::reembed::ReembedOptions) -> String {
    match db.reembed("test-embedder", opts) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("an unimplemented ReembedOptions knob must error, not succeed"),
    }
}

#[test]
fn unimplemented_reembed_namespace_is_rejected_not_silently_global() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let opts = crate::engine::reembed::ReembedOptions {
        namespace: Some("only-this-one".to_string()),
        ..Default::default()
    };
    let msg = reembed_err(&db, opts);
    assert!(
        msg.contains("embedding space belongs to the ENGINE"),
        "namespace must be refused as an engine-scoped-embedding request; got: {msg}"
    );
}

#[test]
fn unimplemented_reembed_pause_policy_is_rejected() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let opts = crate::engine::reembed::ReembedOptions {
        write_policy: crate::engine::reembed::ReembedWritePolicy::Pause,
        ..Default::default()
    };
    let msg = reembed_err(&db, opts);
    assert!(
        msg.contains("Pause is not implemented"),
        "Pause must be refused rather than silently granting Queue; got: {msg}"
    );
}

#[test]
fn unimplemented_reembed_resume_from_checkpoint_is_rejected() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let opts = crate::engine::reembed::ReembedOptions {
        resume_from_checkpoint: true,
        ..Default::default()
    };
    let msg = reembed_err(&db, opts);
    assert!(
        msg.contains("resume_from_checkpoint is not implemented"),
        "resume must be refused before an interrupted run discards its work; got: {msg}"
    );
}

// =====================================================================
// THE GATE for namespace-scoped reembed, written BEFORE the feature.
//
// A scoped reembed touches ONE namespace; everything about every other
// namespace must be bit-identical afterwards. Four parts of this pipeline
// are global by construction and each would violate that silently:
//
//   1. Rebuilding sources only rows with `embedding_new` (reembed.rs
//      ~:1127), so a scoped run rebuilds an index containing ONLY the
//      scoped namespace — every other namespace silently disappears from
//      vector search while still being present and "active" in SQL. That
//      is the HNSW-orphan failure shape again.
//   2. The swap does `DELETE FROM memory_chunks` with no predicate
//      (~:1343), destroying chunk vectors for untouched namespaces.
//   3. `meta.active_generation` is bumped globally (~:1319), so untouched
//      rows are left behind the generation watermark.
//   4. Verifying asserts EVERY row carries the new embedder.
//
// While scoping is unimplemented this asserts the guard is SIDE-EFFECT
// FREE — a rejected call must not leave the store half-migrated. When
// scoping lands, the same assertions become the real invariant and this
// test starts doing its full job without being rewritten.
// =====================================================================
#[test]
fn scoped_reembed_must_not_disturb_other_namespaces() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    for i in 0..8 {
        db.record(
            &format!("keep record {i} about ledgers"),
            "semantic",
            0.5,
            0.0,
            86400.0,
            &empty_meta(),
            &vec_seed(i as f32 + 1.0, 8),
            "keep",
            0.9,
            "general",
            "user",
            None,
        )
        .unwrap();
        db.record(
            &format!("touch record {i} about deployments"),
            "semantic",
            0.5,
            0.0,
            86400.0,
            &empty_meta(),
            &vec_seed(i as f32 + 100.0, 8),
            "touch",
            0.9,
            "general",
            "user",
            None,
        )
        .unwrap();
    }

    let snapshot = |db: &YantrikDB| -> Vec<(String, i64, usize)> {
        let conn = db.conn();
        let mut stmt = conn
            .prepare(
                "SELECT rid, COALESCE(embedding_generation,0), length(embedding) \
                 FROM memories WHERE namespace = 'keep' ORDER BY rid",
            )
            .unwrap();
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i64>(2)? as usize,
                ))
            })
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        rows
    };
    let chunk_count = |db: &YantrikDB| -> i64 {
        db.conn()
            .query_row("SELECT COUNT(*) FROM memory_chunks", [], |r| r.get(0))
            .unwrap_or(0)
    };

    let keep_before = snapshot(&db);
    let chunks_before = chunk_count(&db);
    assert!(
        !keep_before.is_empty(),
        "fixture must have 'keep' rows to protect"
    );

    let opts = crate::engine::reembed::ReembedOptions {
        namespace: Some("touch".to_string()),
        ..Default::default()
    };
    // Unimplemented today -> Err. Implemented later -> Ok. EITHER WAY the
    // assertions below must hold; that is the whole point of the gate.
    let _ = db.reembed("test-embedder", opts);

    assert_eq!(
        keep_before,
        snapshot(&db),
        "a namespace-scoped reembed (or its rejection) changed rows in ANOTHER \
         namespace — rid/generation/embedding-length must be bit-identical"
    );
    assert_eq!(
        chunks_before,
        chunk_count(&db),
        "memory_chunks was purged globally; untouched namespaces lost their \
         chunk vectors (reembed.rs DELETE FROM memory_chunks has no predicate)"
    );
}

// =====================================================================
// 2026-08-17: record_with_rid must not cross a reembed cutover.
//
// It is the deterministic replay primitive (caller-supplied rid, vector,
// timestamp and model; the engine's own embedder is never invoked), used by
// the cluster applier and the materializer drain. It took its SearchState
// snapshot with NO sync-writer guard, so a reembed cutover could publish a
// new state between the snapshot and the index append — landing the vector
// in a DISCARDED delta index while the row committed to SQL as active.
// Stored, alive, unfindable: the HNSW-orphan shape through another door.
//
// It cannot use record()'s queued fallback, because the queued materializer
// re-encodes under the NEW embedder and this path exists to be
// byte-identical across leader and followers. So it defers retryably,
// following the record_batch precedent.
// =====================================================================
#[test]
fn record_with_rid_defers_instead_of_crossing_a_reembed_cutover() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // Put the router where a reembed cutover puts it.
    db.write_router.switch_to_queueing();

    let err = db
        .record_with_rid(
            "01900000-0000-7000-8000-0000000000ab",
            "deterministic replicated fact",
            "semantic",
            0.6,
            0.0,
            1000.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "work",
            0.9,
            "general",
            "inference",
            None,
            1_700_000_000_000_000,
            &[],
            "test-model",
            None,
            crate::provenance::WriteAdmission::Admitted,
        )
        .expect_err("a cutover in flight must defer, not commit against a doomed generation");

    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::DeterministicWriteDeferredDuringReembed { .. }
        ),
        "must be the typed retryable deferral, got: {err}"
    );

    // NOTHING durable may have happened — the whole point of deferring.
    let n: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE rid = '01900000-0000-7000-8000-0000000000ab'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        n, 0,
        "a deferred deterministic write must leave no row behind"
    );

    // And it works again once the cutover finishes.
    db.write_router.switch_to_normal();
    db.record_with_rid(
        "01900000-0000-7000-8000-0000000000ab",
        "deterministic replicated fact",
        "semantic",
        0.6,
        0.0,
        1000.0,
        &empty_meta(),
        &vec_seed(1.0, 8),
        "work",
        0.9,
        "general",
        "inference",
        None,
        1_700_000_000_000_000,
        &[],
        "test-model",
        None,
        crate::provenance::WriteAdmission::Admitted,
    )
    .expect("must succeed once the router is Normal again");
}

// =====================================================================
// 2026-08-17 — the F1/F2 forget-resurrection race, as a gate.
//
// forget() and tombstone_with_rid() both funnel into tombstone_inner,
// which tombstones the rid AND its chunks in
// search_state.load().vec_index. Unguarded, a reembed cutover could
// publish an index built from a SQL snapshot predating the tombstone,
// while the delta tombstone died with the discarded state: the record
// ends up tombstoned in SQL and ALIVE in the live index. A delete that
// silently un-deletes.
// =====================================================================
#[test]
fn forget_defers_rather_than_tombstoning_into_a_doomed_index() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "a fact the user will ask to forget",
            "semantic",
            0.5,
            0.0,
            86400.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "default",
            0.9,
            "general",
            "user",
            None,
        )
        .unwrap();

    // A reembed cutover is in flight.
    db.write_router.switch_to_queueing();

    let err = db
        .forget(&rid)
        .expect_err("forget during a cutover must defer, not tombstone into a doomed index");
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::ForgetDeferredDuringReembed { .. }
        ),
        "must be the typed retryable deferral, got: {err}"
    );

    // The row must be untouched — a deferred delete is not a partial delete.
    let status: String = db
        .conn()
        .query_row(
            "SELECT consolidation_status FROM memories WHERE rid = ?1",
            [&rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        status, "active",
        "a deferred forget must leave the row exactly as it was"
    );

    // And it works once the cutover completes.
    db.write_router.switch_to_normal();
    assert!(db.forget(&rid).unwrap(), "forget must succeed post-cutover");
    let status: String = db
        .conn()
        .query_row(
            "SELECT consolidation_status FROM memories WHERE rid = ?1",
            [&rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(status, "tombstoned");
}

// =====================================================================
// 2026-08-17 — the last two index mutators that could cross a cutover.
// Both are deletes-or-installs that read SearchState and then mutate it.
// =====================================================================

#[test]
fn rebuild_vec_index_discards_rather_than_mixing_embedding_spaces() {
    // The rebuild reads memories.embedding — the space active when it
    // STARTED — and installs the result as the cold tier of whatever state
    // is live when it FINISHES. Across a cutover that means old-space
    // vectors inside the new generation's index: every cold-tier distance
    // measured against a query encoded by a different model. Nothing is
    // lost and nothing errors, which is exactly why no test caught it —
    // the index is populated and every lookup returns something.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    for i in 0..6 {
        db.record(
            &format!("rebuildable record {i}"),
            "semantic",
            0.5,
            0.0,
            86400.0,
            &empty_meta(),
            &vec_seed(i as f32 + 1.0, 8),
            "default",
            0.9,
            "general",
            "user",
            None,
        )
        .unwrap();
    }
    // Healthy case: no cutover, the rebuild installs.
    let n = db.rebuild_vec_index().expect("rebuild must work normally");
    assert!(n > 0, "rebuild should install a populated cold tier");

    // Cutover in flight: the rebuilt index describes a space that may no
    // longer be current, so it must be thrown away, not installed.
    db.write_router.switch_to_queueing();
    let err = db
        .rebuild_vec_index()
        .expect_err("a rebuild finishing during a cutover must not install");
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::IndexRebuildDeferredDuringReembed { .. }
        ),
        "must be the typed retryable deferral, got: {err}"
    );

    db.write_router.switch_to_normal();
    assert!(
        db.rebuild_vec_index().unwrap() > 0,
        "rebuild must work again once the cutover completes"
    );
}
