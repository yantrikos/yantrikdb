use super::*;

// ── Phase 6 RYW — caller-supplied seq + visible_seq bump from all 4 primitives ──

#[test]
fn record_with_rid_uses_caller_supplied_seq_and_bumps_visible() {
    // Cluster determinism: when caller passes Some(n), the visible_seq for
    // the namespace must reach exactly n (not n+something) and vec_seq
    // must ratchet up to at least n.
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let emb = vec_seed(11.0, 64);
    db.record_with_rid(
        "rid_seq_supplied",
        "x",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &emb,
        "alpha",
        0.8,
        "general",
        "user",
        None,
        1_700_000_100_000_000,
        &[],
        "test-model.v1",
        Some(1_000_000),
        crate::provenance::WriteAdmission::Admitted,
    )
    .unwrap();
    assert_eq!(
        db.visible_seq_for("alpha"),
        1_000_000,
        "visible_seq[alpha] equals caller-supplied seq"
    );

    // Subsequent engine-allocated seq for a fresh write must be > 1_000_000
    // because vec_seq was ratcheted.
    db.record_with_rid(
        "rid_after_ratchet",
        "y",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(12.0, 64),
        "alpha",
        0.8,
        "general",
        "user",
        None,
        1_700_000_101_000_000,
        &[],
        "test-model.v1",
        None,
        crate::provenance::WriteAdmission::Admitted,
    )
    .unwrap();
    assert!(
        db.visible_seq_for("alpha") > 1_000_000,
        "engine-allocated seq is > ratcheted high-water"
    );
}

#[test]
fn tombstone_with_rid_bumps_visible_seq_even_when_rid_missing() {
    // Cluster determinism: a follower replaying a tombstone for a rid it
    // does not have locally (snapshot lag) must still bump visible_seq for
    // the supplied namespace, because the caller waiting on RYW for that
    // namespace expects the watermark to advance regardless of whether
    // the local SQL state knows the rid.
    let db = YantrikDB::new(":memory:", 64).unwrap();
    assert_eq!(db.visible_seq_for("beta"), 0);
    db.tombstone_with_rid(
        "rid_unknown_locally",
        "beta",
        None,
        1_700_000_200_000_000,
        Some(2_000_000),
    )
    .unwrap();
    assert_eq!(
        db.visible_seq_for("beta"),
        2_000_000,
        "tombstone_with_rid bumps visible_seq[beta] even on missing rid"
    );
}

#[test]
fn upsert_entity_edge_with_id_bumps_visible_seq() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    db.upsert_entity_edge_with_id(
        "edge_seq",
        "X",
        "Y",
        "knows",
        0.5,
        "gamma",
        1_700_000_300_000_000,
        Some(3_000_000),
    )
    .unwrap();
    assert_eq!(db.visible_seq_for("gamma"), 3_000_000);

    // Idempotent re-apply with the SAME seq is a no-op (fetch_max keeps it).
    db.upsert_entity_edge_with_id(
        "edge_seq",
        "X",
        "Y",
        "knows",
        0.5,
        "gamma",
        1_700_000_300_000_000,
        Some(3_000_000),
    )
    .unwrap();
    assert_eq!(
        db.visible_seq_for("gamma"),
        3_000_000,
        "same-seq replay does not regress watermark"
    );

    // A larger supplied seq advances. (Edge-replay should never happen with
    // a smaller seq in cluster mode, but fetch_max protects us regardless.)
    db.upsert_entity_edge_with_id(
        "edge_seq2",
        "P",
        "Q",
        "knows",
        0.5,
        "gamma",
        1_700_000_301_000_000,
        Some(3_500_000),
    )
    .unwrap();
    assert_eq!(db.visible_seq_for("gamma"), 3_500_000);
}

#[test]
fn delete_entity_edge_with_id_bumps_visible_seq_even_when_edge_missing() {
    // Snapshot-lag follower scenario: edge_id unknown locally, but the
    // commit-log entry must still advance visible_seq for the namespace.
    let db = YantrikDB::new(":memory:", 64).unwrap();
    db.delete_entity_edge_with_id(
        "edge_never",
        "delta",
        1_700_000_400_000_000,
        Some(4_000_000),
    )
    .unwrap();
    assert_eq!(db.visible_seq_for("delta"), 4_000_000);
}

// ── Issue #8 reproduction: tombstoned memories must not appear in recall ──

#[test]
fn issue_8_tombstoned_memories_excluded_from_recall() {
    // Repro from yantrikos/yantrikdb#8 (filed 2026-04-30):
    // 1. record memory, capture rid
    // 2. forget(rid) → consolidation_status='tombstoned'
    // 3. recall with semantically-related query → MUST NOT return the rid
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let emb = vec_seed(42.0, 64);
    let rid = db
        .record(
            "memory to forget for issue 8 repro",
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
        )
        .unwrap();

    // Sanity: visible before forget.
    let before = db
        .recall(
            &emb, 5, None, None, false, false, None, true, None, None, None, None, None, false,
            None, // event_after (#149)
            None, // event_before (#149)
        )
        .unwrap();
    assert!(
        before.iter().any(|r| r.rid == rid),
        "memory must be findable before forget"
    );

    db.forget(&rid).unwrap();

    // After forget, should NOT appear in recall.
    let after = db
        .recall(
            &emb, 5, None, None, false, false, None, true, None, None, None, None, None, false,
            None, // event_after (#149)
            None, // event_before (#149)
        )
        .unwrap();
    assert!(
        !after.iter().any(|r| r.rid == rid),
        "issue #8: tombstoned memory must NOT appear in recall results"
    );
}

#[test]
fn issue_8_tombstoned_persists_across_engine_reopen() {
    // The original bug also manifested across engine restart: rebuild_vec_index
    // would re-load tombstoned memories from the SQL table. Verify that
    // build_vec_index_with_enc filters consolidation_status correctly.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("issue8.db");
    let path_str = path.to_str().unwrap();
    let emb = vec_seed(44.0, 64);
    let rid;
    {
        let db = YantrikDB::new(path_str, 64).unwrap();
        rid = db
            .record(
                "memory survives reopen test",
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
            )
            .unwrap();
        db.forget(&rid).unwrap();
    }
    // Reopen — engine rebuilds vec_index from disk.
    {
        let db2 = YantrikDB::new(path_str, 64).unwrap();
        let after = db2
            .recall(
                &emb, 5, None, None, false, false, None, true, None, None, None, None, None, false,
                None, // event_after (#149)
                None, // event_before (#149)
            )
            .unwrap();
        assert!(
            !after.iter().any(|r| r.rid == rid),
            "tombstoned memory must stay hidden across engine reopen"
        );
    }
}

// ── Phase 6 RYW — visible_seq + wait_for_visible_seq + recall_with_seq ──

#[test]
fn visible_seq_starts_at_zero_for_new_namespace() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    assert_eq!(db.visible_seq_for("never_used"), 0);
}

#[test]
fn record_bumps_visible_seq_for_namespace() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let before = db.visible_seq_for("default");
    let _ = db
        .record(
            "test",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.0, 64),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    let after = db.visible_seq_for("default");
    assert!(after > before, "record() must bump visible_seq[default]");
}

#[test]
fn visible_seq_isolated_per_namespace() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    db.record(
        "ns_a memory",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(1.0, 64),
        "ns_a",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();
    let seq_a = db.visible_seq_for("ns_a");
    let seq_b = db.visible_seq_for("ns_b");
    assert!(seq_a > 0);
    assert_eq!(seq_b, 0, "ns_b unaffected by writes to ns_a");
}

#[test]
fn wait_for_visible_seq_succeeds_when_already_reached() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    db.record(
        "set watermark",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(1.0, 64),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();
    let current = db.visible_seq_for("default");
    // Wait for a seq we've already passed — should return immediately.
    db.wait_for_visible_seq("default", current, std::time::Duration::from_millis(100))
        .expect("already-reached watermark");
}

#[test]
fn wait_for_visible_seq_times_out_on_unreachable() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let err = db
        .wait_for_visible_seq("never", 9999, std::time::Duration::from_millis(50))
        .expect_err("must timeout");
    match err {
        crate::error::YantrikDbError::RyWaitTimeout {
            namespace,
            requested_seq,
            observed_seq,
            waited_ms,
        } => {
            assert_eq!(namespace, "never");
            assert_eq!(requested_seq, 9999);
            assert_eq!(observed_seq, 0);
            assert_eq!(waited_ms, 50);
        }
        other => panic!("expected RyWaitTimeout, got {other:?}"),
    }
}

#[test]
fn wait_for_visible_seq_wakes_on_concurrent_write() {
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    let db = Arc::new(YantrikDB::new(":memory:", 64).unwrap());
    // Start a waiter that wants seq=1.
    let db_w = Arc::clone(&db);
    let waiter =
        thread::spawn(move || db_w.wait_for_visible_seq("default", 1, Duration::from_secs(2)));

    // Spawn a writer after a brief delay.
    let db_writer = Arc::clone(&db);
    let writer = thread::spawn(move || {
        thread::sleep(Duration::from_millis(50));
        db_writer
            .record(
                "wake the waiter",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(1.0, 64),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
    });

    writer.join().unwrap();
    let result = waiter.join().unwrap();
    assert!(result.is_ok(), "waiter should be notified by the write");
}

#[test]
fn recall_with_seq_returns_results_when_seq_reached() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let emb = vec_seed(1.0, 64);
    let _ = db
        .record(
            "ryw test",
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
        )
        .unwrap();
    let current = db.visible_seq_for("default");
    let r = db
        .recall_with_seq(
            &emb,
            5,
            None,
            None,
            false,
            false,
            None,
            true,
            Some("default"),
            None,
            None,
            current,
            std::time::Duration::from_millis(100),
        )
        .unwrap();
    assert!(
        !r.is_empty(),
        "recall_with_seq returns results once seq reached"
    );
}

#[test]
fn recall_with_seq_times_out_on_unreachable() {
    let db = YantrikDB::new(":memory:", 64).unwrap();
    let err = db
        .recall_with_seq(
            &vec_seed(1.0, 64),
            5,
            None,
            None,
            false,
            false,
            None,
            true,
            Some("default"),
            None,
            None,
            9999,
            std::time::Duration::from_millis(50),
        )
        .expect_err("must timeout");
    assert!(matches!(
        err,
        crate::error::YantrikDbError::RyWaitTimeout { .. }
    ));
}
