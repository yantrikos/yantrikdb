use super::*;

// ─────────────────────────────────────────────────────────────────
// v0.7.19 regression tests — orphan-on-Backpressure compensating
// DELETE + replication_apply_log audit table.
// ─────────────────────────────────────────────────────────────────

#[cfg(feature = "bundled-embedder")]
#[test]
fn record_backpressure_writes_nothing_at_all() {
    // **v0.10 Item 4a.6a + 4a.6b.** The v0.7.19 sibling below asserts the OLD
    // contract: the row was written, then a compensating DELETE reclaimed it.
    // record() no longer needs compensating — it RESERVES delta capacity before
    // touching SQL, so Backpressure surfaces without a row, an op, or session
    // state.
    //
    // This test spent 4a.6a named `..._writes_no_row_op_or_session_state`,
    // because "writes nothing at all" was then a LIE (sol 4a.6a finding 2):
    // `calibrate_importance` autocommitted a `namespace_importance_stats` update
    // before routing, and the warn-mode gate ticked its nudge counter before
    // routing — so a rejected write HAD moved state. 4a.6b made both
    // winner-only, and the name is finally the contract. Asserted:
    //
    //   1. no memories row              (the old design also achieved this, by
    //                                    writing one and deleting it)
    //   2. no oplog op of ANY kind      (row + ops are now one transaction)
    //   3. session memory_count UNCHANGED — the sharp one. The old path bumped
    //      the count and needed a SECOND patch (v0.7.23) to reverse it. The bump
    //      now never happens, so there is nothing to reverse.
    //   4. pending_op_count UNCHANGED — the materialize enqueue is in the same
    //      transaction, and the counter only moves after commit.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let dim = db.embedding_dim();

    let session_id = db.session_start("default", "bp-client", &empty_meta()).ok();
    let count_for = |sid: &str| -> i64 {
        db.conn()
            .query_row(
                "SELECT memory_count FROM sessions WHERE session_id = ?1",
                rusqlite::params![sid],
                |r| r.get(0),
            )
            .unwrap_or(-1)
    };

    // Pump until the delta tier saturates (no compactor is running, so it never
    // drains). Record the state at the moment Backpressure first appears.
    let mut hit: Option<(i64, i64, i64, String)> = None;
    for i in 0..400 {
        let embedding: Vec<f32> = (0..dim).map(|j| ((i + j) as f32) * 0.001).collect();
        let rows_before: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        let ops_before: i64 = db
            .conn()
            .query_row("SELECT COUNT(*) FROM oplog", [], |r| r.get(0))
            .unwrap();
        let sess_before = session_id.as_deref().map(count_for).unwrap_or(-1);
        let pend_before = db.count_pending_ops().unwrap();

        let text = format!("bp-probe-{i}");
        match db.record(
            &text,
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &embedding,
            "default",
            0.8,
            "general",
            "user",
            None,
        ) {
            Ok(_) => continue,
            Err(crate::error::YantrikDbError::Backpressure { .. }) => {
                let rows_after: i64 = db
                    .conn()
                    .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
                    .unwrap();
                let ops_after: i64 = db
                    .conn()
                    .query_row("SELECT COUNT(*) FROM oplog", [], |r| r.get(0))
                    .unwrap();
                let sess_after = session_id.as_deref().map(count_for).unwrap_or(-1);
                let pend_after = db.count_pending_ops().unwrap();

                assert_eq!(rows_after, rows_before, "Backpressure wrote a memories row");
                assert_eq!(ops_after, ops_before, "Backpressure wrote an oplog op");
                assert_eq!(
                    sess_after, sess_before,
                    "Backpressure bumped sessions.memory_count (the v0.7.23 residual)"
                );
                assert_eq!(
                    pend_after, pend_before,
                    "Backpressure moved pending_op_count"
                );
                // The rejected text must not be findable at all.
                let found: i64 = db
                    .conn()
                    .query_row(
                        "SELECT COUNT(*) FROM memories WHERE text = ?1",
                        rusqlite::params![text],
                        |r| r.get(0),
                    )
                    .unwrap();
                assert_eq!(found, 0, "rejected record's text is present in memories");
                hit = Some((rows_after, ops_after, sess_after, text));
                break;
            }
            Err(e) => panic!("unexpected error while pumping the delta: {e:?}"),
        }
    }
    assert!(
        hit.is_some(),
        "delta never saturated — test did not exercise Backpressure"
    );

    // ── 4a.6b: the LOSER-side effects. The delta is saturated, so this probe is
    // guaranteed to be rejected — and a rejected write must leave NO trace:
    //
    //   5. namespace_importance_stats UNCHANGED — calibrate_importance used to
    //      autocommit the EWMA advance BEFORE routing, so every rejected write
    //      still permanently moved this namespace's calibration distribution.
    //   6. provenance_flagged_since_boot UNCHANGED — the warn-mode gate used to
    //      tick its nudge counter BEFORE routing, so flagged-but-rejected writes
    //      inflated the very metric that decides when warn can become enforce.
    //
    // The probe is deliberately BOTH flagged and rejected: source="inference" +
    // kind="fact" + no confidence_basis is the anti-laundering violation, and in
    // Warn mode that is counted-and-allowed — so only the Backpressure rejection
    // downstream separates "accepted flagged write" (must count) from "rejected
    // flagged write" (must not).
    db.set_provenance_gate_mode(crate::provenance::GateMode::Warn)
        .unwrap();
    let stats_row = |db: &YantrikDB| -> Option<(f64, i64)> {
        db.conn()
            .query_row(
                "SELECT ewma, count FROM namespace_importance_stats WHERE namespace = 'default'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .ok()
    };
    let stats_before = stats_row(&db);
    let flagged_before = db.stats(None).unwrap().provenance_flagged_since_boot;

    let embedding: Vec<f32> = (0..dim).map(|j| ((999 + j) as f32) * 0.001).collect();
    let err = db
        .record(
            "bp-probe-flagged-and-rejected",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({"kind": "fact"}),
            &embedding,
            "default",
            0.8,
            "general",
            "inference",
            None,
        )
        .expect_err("delta is saturated; the probe must be rejected");
    assert!(
        matches!(err, crate::error::YantrikDbError::Backpressure { .. }),
        "probe must fail with Backpressure, got {err:?}"
    );

    assert_eq!(
        stats_row(&db),
        stats_before,
        "rejected write advanced namespace_importance_stats — a loser moved the \
         namespace's calibration distribution permanently"
    );
    assert_eq!(
        db.stats(None).unwrap().provenance_flagged_since_boot,
        flagged_before,
        "flagged-but-REJECTED write ticked provenance_flagged_since_boot — \
         inflating the warn→enforce nudge metric with writes that never landed"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn consolidate_entity_overlap_guard_depends_on_materializer_progress() {
    // Pins a genuine, load-bearing behaviour of consolidate(): its result depends
    // on whether the materializer has populated `memory_entities` yet.
    //
    // The entity-overlap guard (require_entity_overlap, default true since v0.6.0)
    // refuses to merge two memories whose entity sets are non-empty and DISJOINT —
    // it exists to stop cosine-only false merges across distinct named subjects
    // ("Alice is CEO" vs "Sarah is CTO"). Entities are extracted ASYNCHRONOUSLY by
    // the materializer, so the SAME pair of memories clusters before extraction
    // and refuses to cluster after it.
    //
    // This was found the hard way: v0.10 Item 4a.6a shifted record()'s commit
    // timing and two python consolidation tests began flaking ~50%. They recorded
    // "A1"/"A2" — distinct entities — and had only ever passed by outrunning the
    // materializer. The engine was right both times; the tests encoded pre-guard
    // behaviour. They now use entity-free text so they test consolidation
    // mechanics rather than a race.
    //
    // Keep this test: it is the executable statement of that asymmetry, and it
    // fails loudly if the guard's semantics ever drift.
    let mk = |db: &YantrikDB, text: &str, seed: f32| {
        db.record(
            text,
            "episodic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(seed, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap()
    };

    // --- WITHOUT draining: entities not yet extracted ---
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let r1 = mk(&db, "A1", 1.0);
    let r2 = mk(&db, "A2", 1.02);
    let ents_before: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM memory_entities", [], |r| r.get(0))
        .unwrap();
    let clusters_nodrain =
        crate::cognition::consolidate::find_consolidation_candidates(&db, 0.9, 30.0, 2, 100, true)
            .unwrap()
            .len();

    // --- WITH draining: materializer has populated memory_entities ---
    let db2 = YantrikDB::new(":memory:", 8).unwrap();
    let _r3 = mk(&db2, "A1", 1.0);
    let _r4 = mk(&db2, "A2", 1.02);
    db2.apply_pending_ops_once(100).unwrap();
    let ents_after: i64 = db2
        .conn()
        .query_row("SELECT COUNT(*) FROM memory_entities", [], |r| r.get(0))
        .unwrap();
    let clusters_drained =
        crate::cognition::consolidate::find_consolidation_candidates(&db2, 0.9, 30.0, 2, 100, true)
            .unwrap()
            .len();

    // Report the mechanism plainly; the assertion is on the DIFFERENCE.
    println!(
        "no-drain: memory_entities={ents_before} clusters={clusters_nodrain}  |  \
         drained: memory_entities={ents_after} clusters={clusters_drained}  (rids {r1} {r2})"
    );

    assert_ne!(
        clusters_nodrain, clusters_drained,
        "if these match, the entity-overlap guard is NOT the mechanism — hypothesis refuted"
    );
    assert_eq!(
        clusters_drained, 0,
        "once entities exist, the disjoint-entity guard must block the merge"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn record_commits_row_and_both_oplog_ops_atomically() {
    // **v0.10 Item 4a.6a.** The row, the user-facing "record" op, and the
    // materialize_record_post enqueue were three independent autocommit windows;
    // a crash between them left a row with no provenance (the 23k-row leak) or a
    // record whose entity materialization was owed to nobody. They now commit in
    // ONE transaction, so all three exist together or not at all.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let pend_before = db.count_pending_ops().unwrap();

    let rid = db
        .record(
            "atomic commit probe",
            "semantic",
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
    let row: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE rid = ?1",
            rusqlite::params![rid],
            |r| r.get(0),
        )
        .unwrap();
    let rec_op: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM oplog WHERE op_type = 'record' AND target_rid = ?1 AND applied = 1",
            rusqlite::params![rid],
            |r| r.get(0),
        )
        .unwrap();
    let post_op: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM oplog WHERE op_type = ?1 AND target_rid = ?2 AND applied = 0",
            rusqlite::params![crate::engine::op_types::OP_MATERIALIZE_RECORD_POST, rid],
            |r| r.get(0),
        )
        .unwrap();
    drop(conn);

    assert_eq!(row, 1, "memories row missing");
    assert_eq!(rec_op, 1, "the record op did not commit with the row");
    assert_eq!(
        post_op, 1,
        "the materialize enqueue did not commit with the row"
    );

    // The counter moves only after commit, and exactly once.
    assert_eq!(
        db.count_pending_ops().unwrap(),
        pend_before + 1,
        "pending_op_count did not track the committed enqueue"
    );

    // Published, therefore visible to search.
    let hits = db
        .recall(
            &vec_seed(1.0, 8),
            5,
            None,
            None,
            false,
            false,
            None,
            true,
            None,
            None,
            None,
            None,
            None,
            false,
        )
        .unwrap();
    assert!(
        hits.iter().any(|h| h.rid == rid),
        "committed record was not published to the index"
    );
}

#[test]
fn record_with_rid_backpressure_does_not_leak_orphan_memories_row() {
    // **v0.7.19 fix verification.** Reproduces the trader's
    // 23k-orphan pattern in miniature: fill the DeltaIndex to its
    // delta_max cap without a compactor, then the next record_with_rid
    // call must (a) return Backpressure (the bug never had a fix for
    // this — that's intentional, surfaces the limit), AND
    // (b) NOT leave a memories row behind.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    // No compactor spawned — delta fills to cap and never drains.
    // Drive enough writes via insert_vector to saturate the delta
    // tier (default delta_max = 256). We use insert_vector for the
    // pump because record_with_rid already includes the fix; the
    // test simulates the failure mode before the fix.
    let _ = db; // suppress unused: we'll exercise via record_with_rid

    let db = YantrikDB::new(":memory:", 8).unwrap();
    // Pump until delta_max reached. The exact count doesn't matter
    // — we just need vec_index.append to return Backpressure on the
    // next call.
    let dim = db.embedding_dim();
    let mut last_backpressure_rid: Option<String> = None;
    for i in 0..400 {
        let embedding: Vec<f32> = (0..dim).map(|j| ((i + j) as f32) * 0.001).collect();
        let attempted_rid = format!("orphan-test-rid-{i}");
        let res = db.record_with_rid(
            &attempted_rid,
            &format!("orphan-test-text-{i}"),
            "episodic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({}),
            &embedding,
            "default",
            0.8,
            "general",
            "user",
            None,
            (i as i64) * 1_000_000,
            &[],
            "test-embedder",
            None,
            crate::provenance::WriteAdmission::Admitted,
        );
        if let Err(crate::error::YantrikDbError::Backpressure { .. }) = res {
            last_backpressure_rid = Some(attempted_rid);
            break;
        }
    }
    let rid = last_backpressure_rid
        .expect("test infrastructure: pump must produce a Backpressure within 400 writes");

    // The compensating DELETE must have run. memories table should
    // have NO row for the rid that hit Backpressure.
    let conn = db.conn();
    let exists: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE rid = ?1",
            params![&rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        exists, 0,
        "v0.7.19: compensating DELETE must remove memories row when Backpressure fires; \
         leaving the row is the trader's 23k-orphan pattern"
    );

    // Sanity: oplog also has NO record_with_rid op for this rid
    // (the log_op was correctly short-circuited by the error path).
    let oplog_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM oplog WHERE target_rid = ?1 AND op_type = 'record_with_rid'",
            params![&rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        oplog_count, 0,
        "Backpressure path correctly leaves no oplog entry"
    );
}

#[test]
fn record_with_rid_backpressure_does_not_overcount_session_memory_count() {
    // **v0.7.23 residual fix verification.** The v0.7.19 compensating
    // DELETE reclaims the orphaned memories row, but before this fix it
    // left the session `memory_count` bumped for a row that no longer
    // exists — over-counting by 1 per backpressure-rejected record.
    // After the fix, the session's stored memory_count must equal the
    // number of memories rows actually linked to that session.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let sid = db
        .session_start("default", "claude", &serde_json::json!({}))
        .unwrap();
    let dim = db.embedding_dim();

    // Pump until at least one Backpressure fires (delta saturates with
    // no compactor draining it). Successful records bump the count and
    // insert a row; rejected ones must do neither.
    let mut saw_backpressure = false;
    for i in 0..400 {
        let embedding: Vec<f32> = (0..dim).map(|j| ((i + j) as f32) * 0.001).collect();
        let res = db.record_with_rid(
            &format!("sess-count-rid-{i}"),
            &format!("sess-count-text-{i}"),
            "episodic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({}),
            &embedding,
            "default",
            0.8,
            "general",
            "user",
            None,
            (i as i64) * 1_000_000,
            &[],
            "test-embedder",
            None,
            crate::provenance::WriteAdmission::Admitted,
        );
        if let Err(crate::error::YantrikDbError::Backpressure { .. }) = res {
            saw_backpressure = true;
            break;
        }
    }
    assert!(
        saw_backpressure,
        "test infrastructure: pump must produce a Backpressure within 400 writes"
    );

    let conn = db.conn();
    let actual_rows: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE session_id = ?1",
            params![&sid],
            |r| r.get(0),
        )
        .unwrap();
    let stored_count: i64 = conn
        .query_row(
            "SELECT memory_count FROM sessions WHERE session_id = ?1",
            params![&sid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        stored_count, actual_rows,
        "v0.7.23: session memory_count ({stored_count}) must match the number of \
         memories rows actually linked to the session ({actual_rows}); a higher \
         count means the backpressure compensation failed to reverse the bump"
    );
}

#[test]
fn record_coerces_blank_namespace_to_default() {
    // **v0.7.23.** A blank namespace is a caller-side defaulting accident
    // (e.g. a server gateway `unwrap_or("")`). The engine coerces it to the
    // canonical "default" so writes/reads/recall all agree on one value.
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // record() with empty namespace → coerced to "default".
    let rid = db
        .record(
            "alpha",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "",
            0.8,
            "d",
            "user",
            None,
        )
        .unwrap();
    assert_eq!(db.get(&rid).unwrap().unwrap().namespace, "default");

    // whitespace-only namespace → also coerced.
    let rid_ws = db
        .record(
            "beta",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(2.0, 8),
            "   ",
            0.8,
            "d",
            "user",
            None,
        )
        .unwrap();
    assert_eq!(db.get(&rid_ws).unwrap().unwrap().namespace, "default");

    // explicit namespace is preserved untouched.
    let rid_ns = db
        .record(
            "gamma",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(3.0, 8),
            "acme",
            0.8,
            "d",
            "user",
            None,
        )
        .unwrap();
    assert_eq!(db.get(&rid_ns).unwrap().unwrap().namespace, "acme");
}

#[test]
fn record_with_rid_coerces_blank_namespace_to_default() {
    // The server's commit applier writes via record_with_rid; this is the
    // path that turned the gateway's `unwrap_or("")` into stored `""` rows.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record_with_rid(
        "blank-ns-rid",
        "alpha",
        "semantic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(1.0, 8),
        "",
        0.8,
        "d",
        "user",
        None,
        1_000_000,
        &[],
        "test-embedder",
        None,
        crate::provenance::WriteAdmission::Admitted,
    )
    .unwrap();
    assert_eq!(
        db.get("blank-ns-rid").unwrap().unwrap().namespace,
        "default",
        "record_with_rid must coerce blank namespace to default (server write path)"
    );
}

#[test]
fn list_records_enumerates_by_kind_with_keyset_cursor() {
    // **v0.7.24 structural query.** The relational counterpart to recall:
    // reliably enumerate ALL records of metadata.kind=X, ordered, paginated by
    // rid keyset — the thing similarity recall structurally cannot do.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let meta = |k: &str, d: &str| serde_json::json!({ "kind": k, "drive_id": d });

    // Interleave two kinds so a kind filter must actually discriminate.
    let mut reply_rids = Vec::new();
    for i in 0..5 {
        reply_rids.push(
            db.record(
                &format!("reply {i}"),
                "semantic",
                0.5,
                0.0,
                604800.0,
                &meta("operator_reply_v1", "D1"),
                &vec_seed(i as f32, 8),
                "ns",
                0.8,
                "d",
                "user",
                None,
            )
            .unwrap(),
        );
        db.record(
            &format!("thought {i}"),
            "semantic",
            0.5,
            0.0,
            604800.0,
            &meta("focal_thought", "D2"),
            &vec_seed((i + 100) as f32, 8),
            "ns",
            0.8,
            "d",
            "user",
            None,
        )
        .unwrap();
    }

    // Page 1 (asc, limit 3): exactly 3 replies, rid-ascending, full-page cursor.
    let (p1, c1) = db
        .list_records(
            Some("ns"),
            Some("operator_reply_v1"),
            None,
            None,
            None,
            None,
            3,
            "asc",
        )
        .unwrap();
    assert_eq!(p1.len(), 3);
    assert!(p1.iter().all(|m| m.metadata["kind"] == "operator_reply_v1"));
    assert!(
        p1[0].rid < p1[1].rid && p1[1].rid < p1[2].rid,
        "rid ascending"
    );
    let c1 = c1.expect("full page yields a cursor");

    // Page 2 via keyset cursor: remaining 2, then no cursor (end reached).
    let (p2, c2) = db
        .list_records(
            Some("ns"),
            Some("operator_reply_v1"),
            None,
            None,
            None,
            Some(&c1),
            3,
            "asc",
        )
        .unwrap();
    assert_eq!(p2.len(), 2);
    assert!(c2.is_none(), "partial page yields no cursor");

    // Completeness: the two pages together == ALL 5 replies (recall can't promise this).
    let mut got: Vec<String> = p1.iter().chain(p2.iter()).map(|m| m.rid.clone()).collect();
    got.sort();
    let mut want = reply_rids.clone();
    want.sort();
    assert_eq!(
        got, want,
        "keyset pages cover the full kind=X set exactly once"
    );

    // Descending order = newest first.
    let (desc, _) = db
        .list_records(
            Some("ns"),
            Some("operator_reply_v1"),
            None,
            None,
            None,
            None,
            10,
            "desc",
        )
        .unwrap();
    assert_eq!(desc.len(), 5);
    assert!(desc[0].rid > desc[4].rid, "desc = newest first");

    // drive_id FK filter (referential integrity): all D1 rows are the replies.
    let (d1, _) = db
        .list_records(Some("ns"), None, Some("D1"), None, None, None, 100, "asc")
        .unwrap();
    assert_eq!(d1.len(), 5);
    assert!(d1.iter().all(|m| m.metadata["drive_id"] == "D1"));

    // Unknown kind → empty page, no cursor.
    let (none, cn) = db
        .list_records(Some("ns"), Some("nope"), None, None, None, None, 10, "asc")
        .unwrap();
    assert!(none.is_empty() && cn.is_none());
}

#[test]
fn list_records_tolerates_non_json_metadata() {
    // **v0.7.24.** The generated kind/drive_id columns guard json_extract with
    // json_valid, so a row whose metadata is NOT valid JSON (encrypted
    // ciphertext, or corruption) resolves to NULL instead of erroring the
    // index build / query. Enumeration of the valid rows must still work.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let good = db
        .record(
            "good",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({ "kind": "k1" }),
            &vec_seed(1.0, 8),
            "ns",
            0.8,
            "d",
            "user",
            None,
        )
        .unwrap();
    let bad = db
        .record(
            "bad",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(2.0, 8),
            "ns",
            0.8,
            "d",
            "user",
            None,
        )
        .unwrap();
    // Corrupt one row's metadata to non-JSON (simulates ciphertext / malformed).
    db.conn()
        .execute(
            "UPDATE memories SET metadata = 'not::json::{' WHERE rid = ?1",
            params![bad],
        )
        .unwrap();

    // kind enumeration returns only the valid row — and does NOT error.
    let (page, _) = db
        .list_records(Some("ns"), Some("k1"), None, None, None, None, 10, "asc")
        .unwrap();
    assert_eq!(page.len(), 1);
    assert_eq!(page[0].rid, good);

    // A broad listing also must not blow up on the bad row.
    let (all, _) = db
        .list_records(Some("ns"), None, None, None, None, None, 10, "asc")
        .unwrap();
    assert_eq!(all.len(), 2);
}

#[test]
fn migrate_v31_to_v32_indexes_existing_rows_and_tolerates_bad_json() {
    // **v0.7.24.** Fresh-install is covered above; this exercises the UPGRADE
    // path: ALTER TABLE ADD a VIRTUAL generated column + build its index over
    // PRE-EXISTING rows, including one whose metadata is not valid JSON.
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE memories (rid TEXT PRIMARY KEY, metadata TEXT, consolidation_status TEXT);\
         INSERT INTO memories VALUES ('r1', '{\"kind\":\"operator_reply_v1\",\"drive_id\":\"D1\"}', 'active');\
         INSERT INTO memories VALUES ('r2', 'not-valid-json-{', 'active');",
    )
    .unwrap();

    // Apply the v32 migration to the pre-existing (v31-shaped) table.
    conn.execute_batch(crate::schema::MIGRATE_V31_TO_V32)
        .unwrap();

    // Existing valid row is now queryable via the generated + indexed columns.
    let kind: Option<String> = conn
        .query_row("SELECT kind FROM memories WHERE rid='r1'", [], |r| r.get(0))
        .unwrap();
    assert_eq!(kind.as_deref(), Some("operator_reply_v1"));
    let drive: Option<String> = conn
        .query_row("SELECT drive_id FROM memories WHERE rid='r1'", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(drive.as_deref(), Some("D1"));

    // The non-JSON row resolved to NULL via the json_valid guard — no error.
    let bad: Option<String> = conn
        .query_row("SELECT kind FROM memories WHERE rid='r2'", [], |r| r.get(0))
        .unwrap();
    assert_eq!(bad, None);

    // The index covers the pre-existing row.
    let n: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE kind='operator_reply_v1'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 1);
}

#[test]
fn think_extract_attribute_claims_detects_free_text_value_update() {
    // **v0.7.23 prototype verification.** Reproduces the reported gap
    // (yantrikdb_conflict_gap.md): a free-text "<subject> is <value>" update
    // ("brand color is blue" → "brand color is now green") is not detected as
    // a conflict by default — then shows the opt-in extractor bridging it into
    // the claim layer so the EXISTING scan_claim_conflicts detector fires.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record(
        "Brand color is blue #1F4E79.",
        "semantic",
        0.9,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(1.0, 8),
        "u",
        0.8,
        "brand_color",
        "user",
        None,
    )
    .unwrap();
    db.record(
        "Brand color is now green #2E7D32.",
        "semantic",
        0.95,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(2.0, 8),
        "u",
        0.8,
        "brand_color",
        "user",
        None,
    )
    .unwrap();

    // Baseline: flag OFF reproduces the gap — no conflict detected.
    let off = ThinkConfig {
        run_consolidation: false,
        run_personality: false,
        ..Default::default()
    };
    assert_eq!(
        db.think(&off).unwrap().conflicts_found,
        0,
        "baseline: free-text attribute-value update is not detected (the reported gap)"
    );

    // Opt-in: flag ON — the existing claim-conflict detector now fires.
    let on = ThinkConfig {
        run_consolidation: false,
        run_personality: false,
        extract_attribute_claims: true,
        ..Default::default()
    };
    let res = db.think(&on).unwrap();
    assert!(
        res.conflicts_found >= 1,
        "v0.7.23: attribute-value conflict should be detected, got {}",
        res.conflicts_found
    );

    // Extraction sanity: exactly two distinct values claimed for the same
    // (subject, relation) — "blue" and "green" — visible via the edges view.
    let conn = db.conn();
    let n: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM edges WHERE src = 'brand color' AND rel_type = 'is'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        n, 2,
        "two attribute-value claims (blue, green) should exist for (brand color, is)"
    );
}

#[test]
fn schema_v29_fresh_install_has_replication_apply_log_table() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let conn = db.conn();

    let cols = table_columns(&conn, "replication_apply_log");
    for required in ["rid", "op_type", "source_actor", "applied_at"] {
        assert!(
            cols.iter().any(|c| c == required),
            "v29: replication_apply_log missing column {required}, got: {cols:?}"
        );
    }
    assert!(
        index_exists(&conn, "idx_replication_apply_log_source_actor"),
        "v29: idx_replication_apply_log_source_actor must exist"
    );

    let schema_version: String = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    // v29 was bumped to v30 by issue #47 (record_revisions table). The
    // v29 test predates that — keep it as a fresh-install smoke check on
    // the v29 invariants (replication_apply_log table + index) but
    // accept any schema_version >= 29.
    let v: i32 = schema_version.parse().unwrap();
    assert!(
        v >= 29,
        "schema_version should be at least 29 (when replication_apply_log landed), got {v}",
    );
}

// Note: replication_apply_log audit-row verification lives in
// distributed/replication.rs's tests module where materialize_record
// is in scope. See test_materialize_record_writes_audit_log there.
