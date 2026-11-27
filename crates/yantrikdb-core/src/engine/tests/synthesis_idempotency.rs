//! #330 — an honest retry must survive a recall.
//!
//! `record_synthesis` and keyed `summarize_cluster` derive the consolidated
//! half-life from the sources' *current* values (`max(sources) * 1.5`), and a
//! plain `recall` reinforces those sources by 1.2. With `half_life` in the
//! payload digest, the retry after a recall looked like a different payload and
//! was refused — so the documented replay guarantee was unreachable in the
//! order an agent actually works: read, synthesise, retry when unsure.
//!
//! Both paths go through `commit_consolidation`, which now digests with
//! `PayloadVariant::Consolidation`: everything the caller supplied, without the
//! engine-derived half-life. Plain `record()`/`record_text()` are untouched —
//! there the half-life is a caller argument and stays in the digest.

use super::*;

fn seed(db: &YantrikDB, texts: &[&str]) -> Vec<String> {
    texts
        .iter()
        .enumerate()
        .map(|(i, t)| {
            db.record(
                t,
                "episodic",
                0.5,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(1.0 + i as f32 * 0.01, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap()
        })
        .collect()
}

/// A recall that reinforces the hits — the step that moved the half-life.
fn recall_once(db: &YantrikDB, skip_reinforce: bool) {
    db.recall(
        &vec_seed(1.0, 8),
        5,
        None,
        None,
        false,
        false,
        Some("alpha one"),
        skip_reinforce,
        Some("default"),
        None,
        None,
        None,
        None,
        false,
        None,
        None,
    )
    .unwrap();
}

/// The half-life actually stored for a record — the value the bug moved.
fn stored_half_life(db: &YantrikDB, rid: &str) -> f64 {
    db.conn()
        .query_row(
            "SELECT half_life FROM memories WHERE rid = ?1",
            rusqlite::params![rid],
            |r| r.get(0),
        )
        .unwrap()
}

const SYNTH_TEXT: &str = "Alpha shipped the release and closed the milestone.";

fn synthesis(
    db: &YantrikDB,
    rids: &[String],
    text: &str,
    key: &str,
) -> std::result::Result<serde_json::Value, crate::error::YantrikDbError> {
    // An explicit vector: `:memory:` carries no embedder, and this is the
    // path whose digest DOES include the caller's vector. The engine-embedded
    // path goes through the same `commit_consolidation` variant.
    let emb = vec_seed(9.0, 8);
    crate::consolidate::record_synthesis(
        db,
        rids,
        text,
        Some(&emb),
        "topic",
        "rollup",
        &empty_meta(),
        key,
    )
}

#[test]
fn synthesis_replay_survives_a_recall_between_the_writes() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["alpha one", "alpha two"]);

    let first = synthesis(&db, &rids, SYNTH_TEXT, "k-recall").unwrap();
    assert_eq!(first["idempotent_replay"], false);

    let before = stored_half_life(&db, &rids[0]);
    recall_once(&db, false);
    let after = stored_half_life(&db, &rids[0]);
    assert!(
        after > before,
        "the control needs the recall to have reinforced: {before} -> {after}"
    );

    let second = synthesis(&db, &rids, SYNTH_TEXT, "k-recall")
        .expect("a retry over the same evidence must replay, not conflict");
    assert_eq!(second["idempotent_replay"], true);
    assert_eq!(
        second["rid"], first["rid"],
        "the retry returns the first write"
    );
}

#[test]
fn synthesis_replay_without_a_recall_still_replays() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["alpha one", "alpha two"]);

    let first = synthesis(&db, &rids, SYNTH_TEXT, "k-plain").unwrap();
    let second = synthesis(&db, &rids, SYNTH_TEXT, "k-plain").unwrap();
    assert_eq!(second["idempotent_replay"], true);
    assert_eq!(second["rid"], first["rid"]);
}

#[test]
fn synthesis_replay_survives_a_recall_that_does_not_reinforce() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["alpha one", "alpha two"]);

    let first = synthesis(&db, &rids, SYNTH_TEXT, "k-skip").unwrap();
    recall_once(&db, true);
    let second = synthesis(&db, &rids, SYNTH_TEXT, "k-skip").unwrap();
    assert_eq!(second["idempotent_replay"], true);
    assert_eq!(second["rid"], first["rid"]);
}

#[test]
fn synthesis_refuses_a_changed_text_after_a_recall() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["alpha one", "alpha two"]);

    synthesis(&db, &rids, SYNTH_TEXT, "k-text").unwrap();
    recall_once(&db, false);
    let err = synthesis(&db, &rids, "A different conclusion entirely.", "k-text")
        .expect_err("a changed text is a different payload and must be refused");
    match err {
        crate::error::YantrikDbError::IdempotencyConflict { .. } => {}
        other => panic!("expected IdempotencyConflict, got {other:?}"),
    }
}

#[test]
fn summary_replay_survives_a_recall_between_the_writes() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["beta one", "beta two"]);

    let first = crate::consolidate::summarize_cluster_with_key(
        &db,
        &rids,
        "Beta shipped too.",
        Some(&vec_seed(9.0, 8)),
        "k-sum-recall",
    )
    .unwrap();

    let before = stored_half_life(&db, &rids[0]);
    recall_once(&db, false);
    let after = stored_half_life(&db, &rids[0]);
    assert!(after > before, "control: {before} -> {after}");

    let second = crate::consolidate::summarize_cluster_with_key(
        &db,
        &rids,
        "Beta shipped too.",
        Some(&vec_seed(9.0, 8)),
        "k-sum-recall",
    )
    .expect("a keyed summary retry must replay after a recall");
    assert_eq!(second["idempotent_replay"], true);
    assert_eq!(second["rid"], first["rid"]);
}

#[test]
fn summary_replay_without_a_recall_still_replays() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["beta one", "beta two"]);

    let first = crate::consolidate::summarize_cluster_with_key(
        &db,
        &rids,
        "Beta shipped too.",
        Some(&vec_seed(9.0, 8)),
        "k-sum-plain",
    )
    .unwrap();
    let second = crate::consolidate::summarize_cluster_with_key(
        &db,
        &rids,
        "Beta shipped too.",
        Some(&vec_seed(9.0, 8)),
        "k-sum-plain",
    )
    .unwrap();
    assert_eq!(second["idempotent_replay"], true);
    assert_eq!(second["rid"], first["rid"]);
}

/// A stored digest from before the fix keeps conflicting — for every retry.
///
/// The old rule fed `half_life`, so a key written by an earlier version holds a
/// digest this one cannot reproduce. `resolve_existing` returns the conflict
/// without touching the stored row, so the mismatch is permanent for that key
/// and the caller must mint a new one. This is the upgrade cost of the fix,
/// measured rather than assumed.
#[test]
fn a_stored_digest_from_the_old_rule_keeps_conflicting() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["gamma one", "gamma two"]);
    synthesis(&db, &rids, SYNTH_TEXT, "k-old").unwrap();

    // Simulate a digest written under the old rule: any byte-different digest
    // in the claim row stands for it. The engine never rewrites it.
    let mut old: Vec<u8> = db
        .conn()
        .query_row(
            "SELECT payload_digest FROM idempotency_claims WHERE idempotency_key LIKE 'k-old%'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    old[0] ^= 0xff;
    db.conn()
        .execute(
            "UPDATE idempotency_claims SET payload_digest = ?1 WHERE idempotency_key LIKE 'k-old%'",
            rusqlite::params![old],
        )
        .unwrap();

    for attempt in 1..=2 {
        let err =
            synthesis(&db, &rids, SYNTH_TEXT, "k-old").expect_err("an old-rule digest conflicts");
        assert!(
            matches!(
                err,
                crate::error::YantrikDbError::IdempotencyConflict { .. }
            ),
            "attempt {attempt}: {err:?}"
        );
    }

    // The stored bytes are untouched, so the key is unusable until the caller
    // changes it — the reason this needs the maintainer's decision.
    let still: Vec<u8> = db
        .conn()
        .query_row(
            "SELECT payload_digest FROM idempotency_claims WHERE idempotency_key LIKE 'k-old%'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(still, old, "a conflict must not rewrite the claim");
}

#[test]
fn summary_refuses_a_changed_text_after_a_recall() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["beta one", "beta two"]);

    crate::consolidate::summarize_cluster_with_key(
        &db,
        &rids,
        "Beta shipped too.",
        Some(&vec_seed(9.0, 8)),
        "k-sum-text",
    )
    .unwrap();
    recall_once(&db, false);
    let err = crate::consolidate::summarize_cluster_with_key(
        &db,
        &rids,
        "A different summary.",
        Some(&vec_seed(9.0, 8)),
        "k-sum-text",
    )
    .expect_err("a changed summary text must be refused");
    match err {
        crate::error::YantrikDbError::IdempotencyConflict { .. } => {}
        other => panic!("expected IdempotencyConflict, got {other:?}"),
    }
}
