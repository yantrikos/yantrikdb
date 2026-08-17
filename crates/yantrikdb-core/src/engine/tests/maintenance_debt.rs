//! The cognitive compactor's ledger (v0.15.x): `maintenance_debt()` and the
//! `writes_since_think` counter.
//!
//! The core is a passive library — it cannot schedule maintenance, but it
//! must always be able to ANSWER "how overdue is maintenance?". These tests
//! pin the ledger's three load-bearing properties:
//!
//! 1. **Content writes count, exactly** — record / batch (per item) /
//!    correct move the counter by the number of memories deposited; the
//!    access-pattern ops (forget, relate, archive, get-reinforce) and
//!    idempotent retries move it by nothing.
//! 2. **Only completed cognition clears** — think's conflict scan and a
//!    non-dry `run_maintenance_cycle` reset the counter and stamp
//!    `last_think_at`; a dry run clears NOTHING (the 0.15.0 dry-run
//!    contract: a preview must not masquerade as hygiene).
//! 3. **Only origin writes count** — `record_with_rid` with
//!    `WriteAdmission::Admitted` (the replication/materializer apply path)
//!    and idempotent replays deposit no debt: a follower's imports get
//!    thought about on the leader.

use super::*;

fn debt_db() -> YantrikDB {
    YantrikDB::new(":memory:", 8).unwrap()
}

fn write(db: &YantrikDB, text: &str, seed: f32) -> String {
    db.record(
        text,
        "semantic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(seed, 8),
        "default",
        0.8,
        "work",
        "user",
        None,
    )
    .unwrap()
}

fn batch_input(text: &str, seed: f32) -> RecordInput {
    RecordInput {
        text: text.to_string(),
        memory_type: "episodic".to_string(),
        importance: 0.5,
        valence: 0.0,
        half_life: 604800.0,
        metadata: empty_meta(),
        embedding: vec_seed(seed, 8),
        namespace: "default".to_string(),
        certainty: 0.8,
        domain: "work".to_string(),
        source: "user".to_string(),
        emotional_state: None,
        idempotency_key: None,
        created_at: None,
    }
}

/// A think that runs ONLY the conflict scan — the pass that defines "the
/// accumulated writes were looked at" — so the tests stay slim-build-safe
/// and assert the reset comes from the scan, not from a side pass.
fn scan_only_think() -> ThinkConfig {
    ThinkConfig {
        run_conflict_scan: true,
        run_consolidation: false,
        run_pattern_mining: false,
        run_personality: false,
        extract_attribute_claims: false,
        ..Default::default()
    }
}

#[test]
fn virgin_store_reports_zeros_and_none() {
    let db = debt_db();
    let debt = db.maintenance_debt();
    assert_eq!(debt.writes_since_think, 0, "no writes yet");
    assert_eq!(debt.last_think_at, None, "cognition has never run");
    assert_eq!(debt.open_conflicts, 0);
    assert_eq!(debt.pending_triggers, 0);
}

#[test]
fn content_writes_move_the_counter_exactly() {
    let db = debt_db();

    // 2 single records + a 3-item batch + 1 metadata correction + 1
    // text-changing correction (caller-supplied vector — the slim-build
    // route through correct_with_reembed) = 7 exactly.
    let rid = write(&db, "single one", 1.0);
    write(&db, "single two", 2.0);

    db.record_batch(&[
        batch_input("batch item one", 3.0),
        batch_input("batch item two", 4.0),
        batch_input("batch item three", 5.0),
    ])
    .unwrap();
    assert_eq!(
        db.maintenance_debt().writes_since_think,
        5,
        "per ITEM, not per call"
    );

    db.correct(
        &rid,
        None,
        Some(&serde_json::json!({"kind": "note"})),
        Some(0.9),
        None,
        "raise importance",
    )
    .unwrap();
    assert_eq!(
        db.maintenance_debt().writes_since_think,
        6,
        "a metadata/scalar correction rewrites content state — it counts"
    );

    let generation = db.search_generation();
    db.correct_with_embedding(
        &rid,
        Some("single one, corrected"),
        &vec_seed(9.0, 8),
        generation,
        None,
        None,
        None,
        "fix the text",
    )
    .unwrap();
    assert_eq!(
        db.maintenance_debt().writes_since_think,
        7,
        "a text-changing correction is new material for cognition"
    );
}

#[test]
fn access_pattern_ops_do_not_move_the_counter() {
    let db = debt_db();
    let a = write(&db, "keep me", 1.0);
    let b = write(&db, "forget me", 2.0);
    assert_eq!(db.maintenance_debt().writes_since_think, 2);

    // Access/lifecycle/graph ops are not new material: reinforcement (get),
    // relate, archive/hydrate, forget.
    db.get(&a).unwrap();
    db.relate("Alice", "Acme", "works_at", 1.0).unwrap();
    db.archive(&a).unwrap();
    db.hydrate(&a).unwrap();
    db.forget(&b).unwrap();

    assert_eq!(
        db.maintenance_debt().writes_since_think,
        2,
        "reinforce/relate/archive/hydrate/forget must not move the write debt"
    );
}

#[test]
fn idempotent_retry_deposits_no_debt() {
    let db = debt_db();
    let first = db
        .record_with_idempotency(
            "keyed write",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "work",
            "user",
            None,
            Some("key-1"),
            None,
        )
        .unwrap();
    assert_eq!(db.maintenance_debt().writes_since_think, 1);

    // Byte-identical retry: resolves to the original rid, writes nothing —
    // and therefore counts nothing (the counter lives INSIDE the winning
    // transaction; a hit never reaches it).
    let retry = db
        .record_with_idempotency(
            "keyed write",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "work",
            "user",
            None,
            Some("key-1"),
            None,
        )
        .unwrap();
    assert_eq!(retry, first);
    assert_eq!(
        db.maintenance_debt().writes_since_think,
        1,
        "repetition is not corroboration, and it is not debt either"
    );
}

#[test]
fn think_with_conflict_scan_resets_and_stamps() {
    let db = debt_db();
    write(&db, "before think one", 1.0);
    write(&db, "before think two", 2.0);
    write(&db, "before think three", 3.0);
    assert_eq!(db.maintenance_debt().writes_since_think, 3);
    assert_eq!(db.maintenance_debt().last_think_at, None);

    db.think(&scan_only_think()).unwrap();

    let debt = db.maintenance_debt();
    assert_eq!(
        debt.writes_since_think, 0,
        "the conflict scan looked at the accumulated writes — debt settles"
    );
    let stamped = debt.last_think_at.expect("last_think_at stamped by think");
    assert!(stamped > 0.0);

    // New writes after the pass accumulate fresh debt.
    write(&db, "after think", 4.0);
    assert_eq!(db.maintenance_debt().writes_since_think, 1);
}

#[test]
fn think_without_conflict_scan_stamps_but_does_not_clear() {
    let db = debt_db();
    write(&db, "unscanned material", 1.0);

    db.think(&ThinkConfig {
        run_conflict_scan: false,
        ..scan_only_think()
    })
    .unwrap();

    let debt = db.maintenance_debt();
    assert!(
        debt.last_think_at.is_some(),
        "last_think_at stamps on every think (pre-existing behavior)"
    );
    assert_eq!(
        debt.writes_since_think, 1,
        "nothing scanned the new writes, so the write debt must stand"
    );
}

/// THE sacred one: a dry run is a preview and clears nothing.
#[test]
fn dry_run_cycle_leaves_debt_untouched() {
    let db = debt_db();
    write(&db, "material a", 1.0);
    write(&db, "material b", 2.0);

    let cfg = crate::engine::maintenance::MaintenanceCycleConfig {
        dry_run: true,
        ..Default::default()
    };
    db.run_maintenance_cycle(&cfg).unwrap();

    let debt = db.maintenance_debt();
    assert_eq!(
        debt.writes_since_think, 2,
        "dry_run must NOT clear the write debt — a preview is not hygiene"
    );
    assert_eq!(
        debt.last_think_at, None,
        "dry_run must NOT stamp last_think_at"
    );
}

#[test]
fn wet_cycle_clears_debt_and_stamps() {
    let db = debt_db();
    write(&db, "material a", 1.0);
    write(&db, "material b", 2.0);

    let report = db
        .run_maintenance_cycle(&crate::engine::maintenance::MaintenanceCycleConfig::default())
        .unwrap();
    assert!(
        report.errors.is_empty(),
        "cycle passes should not error on a small healthy store: {:?}",
        report.errors
    );

    let debt = db.maintenance_debt();
    assert_eq!(
        debt.writes_since_think, 0,
        "a real cycle settles the ledger"
    );
    assert!(debt.last_think_at.is_some());

    // A cycle configured WITHOUT the think pass still clears at completion:
    // the cycle's own hygiene passes are the completed cognition pass.
    write(&db, "material c", 3.0);
    assert_eq!(db.maintenance_debt().writes_since_think, 1);
    let cfg = crate::engine::maintenance::MaintenanceCycleConfig {
        run_think: false,
        ..Default::default()
    };
    db.run_maintenance_cycle(&cfg).unwrap();
    assert_eq!(db.maintenance_debt().writes_since_think, 0);
}

#[test]
fn record_with_rid_counts_origin_once_and_admitted_never() {
    let db = debt_db();
    let call = |rid: &str, admission: crate::provenance::WriteAdmission| {
        db.record_with_rid(
            rid,
            "replicated or origin content",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "work",
            "user",
            None,
            1_700_000_000_000_000,
            &[],
            "test-model",
            None,
            admission,
        )
        .unwrap();
    };

    call("rid-origin", crate::provenance::WriteAdmission::Origin);
    assert_eq!(
        db.maintenance_debt().writes_since_think,
        1,
        "an ORIGIN record_with_rid is a fresh write entering here first"
    );

    // Idempotent replay of the same rid: no new row, no new debt.
    call("rid-origin", crate::provenance::WriteAdmission::Origin);
    assert_eq!(
        db.maintenance_debt().writes_since_think,
        1,
        "a replay deposits no new material"
    );

    // The replication/materializer apply path: thought about on the leader.
    call("rid-admitted", crate::provenance::WriteAdmission::Admitted);
    assert_eq!(
        db.maintenance_debt().writes_since_think,
        1,
        "an ADMITTED apply must not count — followers do not owe cognition \
         for the leader's writes"
    );
}

#[test]
fn replication_apply_deposits_no_debt_on_the_follower() {
    // The full-path version of the Admitted exemption: leader records,
    // follower applies the extracted ops via apply_ops (materialize_record),
    // and the follower's ledger stays at zero.
    let leader = debt_db();
    let follower = debt_db();
    write(&leader, "leader writes this", 1.0);
    write(&leader, "and this", 2.0);
    assert_eq!(leader.maintenance_debt().writes_since_think, 2);

    let ops = crate::replication::extract_ops_since(&leader.conn(), None, None, None, 100).unwrap();
    assert!(!ops.is_empty());
    let stats = crate::replication::apply_ops(&follower, &ops).unwrap();
    assert!(stats.ops_applied >= 2);

    assert_eq!(
        follower.maintenance_debt().writes_since_think,
        0,
        "replication apply is exempt — the material was (or will be) \
         thought about at its origin"
    );
    // ...and the rows really landed, so the zero is an exemption, not a
    // missing apply.
    let stats = follower.stats(None).unwrap();
    assert_eq!(stats.active_memories, 2);
}

#[test]
fn debt_counts_open_conflicts_and_pending_triggers() {
    let db = debt_db();
    let ts = crate::time::now_secs();
    db.conn()
        .execute(
            "INSERT INTO conflicts (conflict_id, conflict_type, status, memory_a, memory_b, \
             detected_at, detected_by, detection_reason, hlc, origin_actor) \
             VALUES ('c1', 'identity_fact', 'open', 'ra', 'rb', ?1, 'test', 'seeded', X'00', 'test')",
            params![ts],
        )
        .unwrap();
    db.conn()
        .execute(
            "INSERT INTO conflicts (conflict_id, conflict_type, status, memory_a, memory_b, \
             detected_at, detected_by, detection_reason, hlc, origin_actor) \
             VALUES ('c2', 'identity_fact', 'resolved', 'ra', 'rb', ?1, 'test', 'seeded', X'00', 'test')",
            params![ts],
        )
        .unwrap();
    db.conn()
        .execute(
            "INSERT INTO trigger_log (trigger_id, trigger_type, urgency, status, reason, \
             suggested_action, created_at, hlc, origin_actor) \
             VALUES ('t1', 'decay', 0.5, 'pending', 'seeded', 'ack', ?1, X'00', 'test')",
            params![ts],
        )
        .unwrap();
    db.conn()
        .execute(
            "INSERT INTO trigger_log (trigger_id, trigger_type, urgency, status, reason, \
             suggested_action, created_at, hlc, origin_actor) \
             VALUES ('t2', 'decay', 0.5, 'dismissed', 'seeded', 'ack', ?1, X'00', 'test')",
            params![ts],
        )
        .unwrap();

    let debt = db.maintenance_debt();
    assert_eq!(debt.open_conflicts, 1, "only status='open' counts");
    assert_eq!(debt.pending_triggers, 1, "only status='pending' counts");
}
