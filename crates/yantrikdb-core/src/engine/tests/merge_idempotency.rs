//! A merge resolution must create its record exactly once — across an
//! interrupted run and across two concurrent writers.
//!
//! Both cases were measured on 0.23.6. The interrupted case is reachable by
//! building the state an abort leaves (record + links + one tombstone, ledger
//! still `open`); the concurrent case by two threads calling the same
//! resolution. Both used to produce two merged records with the same text.

use super::*;

fn seed(db: &YantrikDB, ta: &str, tb: &str, ent: &str) -> (String, String, String) {
    let a = db
        .record(ta, "semantic", 0.8, 0.0, 604800.0, &empty_meta(), &vec_seed(1.0, 8),
                "default", 0.8, "general", "user", None)
        .unwrap();
    let b = db
        .record(tb, "semantic", 0.6, 0.0, 604800.0, &empty_meta(), &vec_seed(2.0, 8),
                "default", 0.8, "general", "user", None)
        .unwrap();
    let c = crate::conflict::create_conflict(
        db, &crate::types::ConflictType::IdentityFact, &a, &b,
        Some(ent), Some("value"), "conflicting values",
    )
    .unwrap();
    (a, b, c.conflict_id)
}

fn merged_count(db: &YantrikDB, text: &str) -> i64 {
    db.conn()
        .query_row("SELECT COUNT(*) FROM memories WHERE text = ?1", [text], |r| r.get(0))
        .unwrap()
}

/// The state an abort between `record()` and the second `forget()` leaves.
/// Built through `record_with_idempotency` on purpose: the claim is committed
/// with the record, so a state "record without claim" is unreachable and must
/// not be tested as if it existed.
#[test]
fn resumed_merge_does_not_create_a_second_record() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let (a, b, cid) = seed(&db, "deploy on tuesday", "deploy on wednesday", "deploy");
    let text = "deploy on tuesday, confirmed";

    let meta = serde_json::json!({ "merged_from": [a, b], "conflict_id": cid });
    let key = format!("merge:{cid}");
    // Exactly what the merge arm writes: same metadata, same (zero) embedding,
    // same importance. A different embedding would change the payload digest
    // and turn the replay into an IdempotencyConflict — a state the engine
    // cannot reach, because the digest is taken over the real payload.
    let zero = vec![0.0f32; 8];
    let merged = db
        .record_with_idempotency(text, "semantic", 0.8, 0.0, 604800.0, &meta, &zero,
                                 "default", 0.8, "general", "user", None, Some(&key), None)
        .unwrap();
    db.link(&merged, &crate::types::RecordLink {
        target_rid: a.clone(), link_type: crate::types::LinkType::Supersedes }).unwrap();
    db.link(&merged, &crate::types::RecordLink {
        target_rid: b.clone(), link_type: crate::types::LinkType::Supersedes }).unwrap();
    db.forget(&a).unwrap();
    assert_eq!(merged_count(&db, text), 1, "half-applied merge has one record");

    db.resolve_conflict(&cid, "merge", None, Some(text), Some("resumed")).unwrap();
    assert_eq!(
        merged_count(&db, text), 1,
        "the resume must reuse the record, not create a second one"
    );
}

#[test]
fn concurrent_merge_does_not_create_two_records() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.db");
    let db = std::sync::Arc::new(YantrikDB::new(path.to_str().unwrap(), 8).unwrap());
    let (_a, _b, cid) = seed(&db, "deploy on tuesday", "deploy on wednesday", "deploy");
    let text = "deploy on tuesday, confirmed";

    let mut handles = Vec::new();
    for _ in 0..2 {
        let d = std::sync::Arc::clone(&db);
        let cid = cid.clone();
        handles.push(std::thread::spawn(move || {
            d.resolve_conflict(&cid, "merge", None, Some(text), Some("concurrent")).is_ok()
        }));
    }
    let _ = handles.into_iter().map(|h| h.join().unwrap_or(false)).filter(|x| *x).count();

    let records = merged_count(&db, text);
    assert_eq!(records, 1, "a concurrent merge must create exactly one record");
}
