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
        second["consolidated_rid"], first["consolidated_rid"],
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
    assert_eq!(second["consolidated_rid"], first["consolidated_rid"]);
}

#[test]
fn synthesis_replay_survives_a_recall_that_does_not_reinforce() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["alpha one", "alpha two"]);

    let first = synthesis(&db, &rids, SYNTH_TEXT, "k-skip").unwrap();
    let before = stored_half_life(&db, &rids[0]);
    recall_once(&db, true);
    let after = stored_half_life(&db, &rids[0]);
    assert_eq!(
        after, before,
        "skip_reinforce must not move the half-life — otherwise this arm is not the control"
    );
    let second = synthesis(&db, &rids, SYNTH_TEXT, "k-skip").unwrap();
    assert_eq!(second["idempotent_replay"], true);
    assert_eq!(second["consolidated_rid"], first["consolidated_rid"]);
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
    assert_eq!(second["consolidated_rid"], first["consolidated_rid"]);
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
    assert_eq!(second["consolidated_rid"], first["consolidated_rid"]);
}

/// A stored digest that differs keeps conflicting — for every retry.
///
/// Scope, stated honestly: this test plants a byte-different digest, it does not
/// construct a pre-fix one. What it proves is the *mechanism* — `resolve_existing`
/// returns the conflict without touching the stored row, so the key stays
/// unusable no matter how often the caller retries — and that is the half that
/// decides whether the upgrade cost is "once" or "forever".
///
/// The other half (that the pre-fix digest really is byte-different) was
/// reproduced in two stages — write under the pre-fix rule, then retry against
/// the fixed code — with both attempts conflicting and the stored digest left
/// byte-identical. That reproduction and its output are in the discussion of
/// the issue that asked for this fix, not in this file.
#[test]
fn a_stored_digest_that_differs_is_permanently_unusable() {
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
fn a_changed_source_set_is_a_new_generation_not_a_conflict() {
    // Spranab's spec asks for a case showing a real change to the source set is
    // still refused. Measured first: it is NOT refused — the engine derives the
    // key from `logical_key + evidence_version`, so different evidence is a
    // different key, and the documented semantics are "changed evidence creates
    // a new generation and atomically supersedes the older logical generation".
    // A conflict here would mean the two are conflated, so the test asserts the
    // documented behaviour instead of the expected one.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["alpha one", "alpha two"]);
    let first = synthesis(&db, &rids, SYNTH_TEXT, "k-set").unwrap();

    let mut wider = rids.clone();
    wider.push(
        db.record(
            "alpha three",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(3.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap(),
    );

    let second = synthesis(&db, &wider, SYNTH_TEXT, "k-set")
        .expect("different evidence is a new generation, not a conflict");
    assert_ne!(
        second["consolidated_rid"], first["consolidated_rid"],
        "a new generation is a new record"
    );
    assert_eq!(second["idempotent_replay"], false);

    // The older generation is retired through `synthesis_state`, not through
    // `consolidation_status` — the refold picks the winner by generation HLC
    // and marks the rest 'superseded'.
    let state: Option<String> = db
        .conn()
        .query_row(
            "SELECT synthesis_state FROM memories WHERE rid = ?1",
            rusqlite::params![first["consolidated_rid"].as_str().unwrap()],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        state.as_deref(),
        Some("superseded"),
        "the older generation must be retired, not left current"
    );

    let verified: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE synthesis_logical_key = ?1 \
             AND synthesis_state = 'verified'",
            // `admission.logical_key` is the key the caller passed.
            rusqlite::params!["k-set"],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(verified, 1, "exactly one generation may be current");
}

#[test]
fn summary_replay_survives_a_recall_that_does_not_reinforce() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["beta one", "beta two"]);

    let first = crate::consolidate::summarize_cluster_with_key(
        &db,
        &rids,
        "Beta shipped too.",
        Some(&vec_seed(9.0, 8)),
        "k-sum-skip",
    )
    .unwrap();
    let before = stored_half_life(&db, &rids[0]);
    recall_once(&db, true);
    let after = stored_half_life(&db, &rids[0]);
    assert_eq!(after, before, "skip_reinforce must not move the half-life");

    let second = crate::consolidate::summarize_cluster_with_key(
        &db,
        &rids,
        "Beta shipped too.",
        Some(&vec_seed(9.0, 8)),
        "k-sum-skip",
    )
    .unwrap();
    assert_eq!(second["idempotent_replay"], true);
    assert_eq!(second["consolidated_rid"], first["consolidated_rid"]);
}

/// #330, test 1 of the three the maintainer asked for: a claim written before
/// this change must conflict **with its original rid**, write nothing, and leave
/// the stored digest alone. The caller then uses that rid — minting a fresh key
/// for the same evidence would create a second logical item.
#[test]
fn a_legacy_claim_conflicts_with_its_original_rid_and_writes_nothing() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["delta one", "delta two"]);
    let first = synthesis(&db, &rids, SYNTH_TEXT, "k-legacy").unwrap();
    let first_rid = first["consolidated_rid"].as_str().unwrap().to_string();

    // A legacy digest is one this code cannot reproduce: plant a byte-different
    // value in the claim row, which is what a pre-fix write leaves behind.
    let mut planted: Vec<u8> = db
        .conn()
        .query_row(
            "SELECT payload_digest FROM idempotency_claims WHERE idempotency_key LIKE 'k-legacy%'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    planted[7] ^= 0xff;
    db.conn()
        .execute(
            "UPDATE idempotency_claims SET payload_digest = ?1 WHERE idempotency_key LIKE 'k-legacy%'",
            rusqlite::params![planted],
        )
        .unwrap();

    let before: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    let err =
        synthesis(&db, &rids, SYNTH_TEXT, "k-legacy").expect_err("a legacy digest must conflict");
    match err {
        crate::error::YantrikDbError::IdempotencyConflict { existing_rid, .. } => {
            assert_eq!(
                existing_rid, first_rid,
                "the conflict names the original rid"
            );
        }
        other => panic!("expected IdempotencyConflict, got {other:?}"),
    }

    let after: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    assert_eq!(after, before, "a conflict must not write a record");
    let still: Vec<u8> = db
        .conn()
        .query_row(
            "SELECT payload_digest FROM idempotency_claims WHERE idempotency_key LIKE 'k-legacy%'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(still, planted, "a conflict must not rewrite the claim");
    let generations: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE synthesis_logical_key = 'k-legacy'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(generations, 1, "no new generation from a conflict");
}

/// #330, test 2: after such a conflict, **changed evidence** under the same
/// logical key is a different derived key, so it writes a new generation and
/// retires the old one. The conflict does not lock the logical key.
#[test]
fn after_a_legacy_conflict_changed_evidence_still_supersedes_the_old_generation() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["epsilon one", "epsilon two"]);
    let first = synthesis(&db, &rids, SYNTH_TEXT, "k-legacy-gen").unwrap();
    let first_rid = first["consolidated_rid"].as_str().unwrap().to_string();

    let mut planted: Vec<u8> = db
        .conn()
        .query_row(
            "SELECT payload_digest FROM idempotency_claims WHERE idempotency_key LIKE 'k-legacy-gen%'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    planted[3] ^= 0xff;
    db.conn()
        .execute(
            "UPDATE idempotency_claims SET payload_digest = ?1 WHERE idempotency_key LIKE 'k-legacy-gen%'",
            rusqlite::params![planted],
        )
        .unwrap();
    let _ = synthesis(&db, &rids, SYNTH_TEXT, "k-legacy-gen").expect_err("legacy conflict first");

    // Changed evidence: a third source joins the cluster.
    let mut wider = rids.clone();
    wider.push(
        db.record(
            "epsilon three",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(4.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap(),
    );
    let second = synthesis(&db, &wider, SYNTH_TEXT, "k-legacy-gen")
        .expect("changed evidence is a new generation, not a conflict");
    assert_ne!(
        second["consolidated_rid"].as_str().unwrap(),
        first_rid,
        "a new generation is a new record"
    );

    let old_state: Option<String> = db
        .conn()
        .query_row(
            "SELECT synthesis_state FROM memories WHERE rid = ?1",
            rusqlite::params![first_rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(old_state.as_deref(), Some("superseded"));
    let verified: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE synthesis_logical_key = 'k-legacy-gen' \
             AND synthesis_state = 'verified'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(verified, 1, "exactly one generation stays current");
}

/// #330, test 3: importance is derived like the others — recalibration rewrites
/// a source's importance in place, and that must not break the replay either.
#[test]
fn a_source_importance_change_does_not_prevent_the_replay() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["zeta one", "zeta two"]);
    let first = synthesis(&db, &rids, SYNTH_TEXT, "k-importance").unwrap();

    // Importance recalibration rewrites the stored value in place; this is the
    // input the digest used to consume.
    let before: f64 = db
        .conn()
        .query_row(
            "SELECT importance FROM memories WHERE rid = ?1",
            rusqlite::params![rids[0]],
            |r| r.get(0),
        )
        .unwrap();
    db.conn()
        .execute(
            "UPDATE memories SET importance = ?1 WHERE rid = ?2",
            rusqlite::params![(before + 0.25).min(1.0), rids[0]],
        )
        .unwrap();

    let second = synthesis(&db, &rids, SYNTH_TEXT, "k-importance")
        .expect("a source-importance change must not break the replay");
    assert_eq!(second["idempotent_replay"], true);
    assert_eq!(second["consolidated_rid"], first["consolidated_rid"]);
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
