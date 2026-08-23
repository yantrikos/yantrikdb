use super::*;

#[test]
fn recall_fetch_plan_reports_only_effective_cap_binding() {
    let ordinary = crate::engine::recall::recall_fetch_plan(20, 5_000, false);
    assert_eq!(ordinary.fetch_k, 400);
    assert_eq!(ordinary.requested_candidates, 400);
    assert_eq!(ordinary.candidate_cap, 10_000);
    assert!(!ordinary.cap_bound);

    let filtered = crate::engine::recall::recall_fetch_plan(20, 20_000, true);
    assert_eq!(filtered.fetch_k, 10_000);
    assert_eq!(filtered.requested_candidates, 20_000);
    assert!(filtered.cap_bound);

    let small_index = crate::engine::recall::recall_fetch_plan(1_000, 5_000, false);
    assert_eq!(small_index.fetch_k, 5_000);
    assert!(!small_index.cap_bound);

    let above_static_cap = crate::engine::recall::recall_fetch_plan(10_001, 20_000, false);
    assert_eq!(above_static_cap.fetch_k, 10_001);
    assert_eq!(above_static_cap.candidate_cap, 10_001);
    assert!(above_static_cap.cap_bound);
}

// ── Issue #46: confidence first-class on recall ───────────────────────

/// Seed a small DB with N records, each carrying a different `certainty`.
/// The i-th record has `certainty = (i + 1) as f64 / N as f64` (so 1..=N
/// maps to evenly-spaced certainty in (0, 1.0]). Embeddings are nearly
/// identical (same `vec_seed(1.0, ...)` shape) so that all candidates
/// survive the HNSW pass and downstream filtering / re-sorting is the
/// only thing distinguishing them.
fn seed_for_certainty_test(db: &YantrikDB, n: usize) -> Vec<String> {
    let mut rids = Vec::with_capacity(n);
    for i in 0..n {
        let certainty = (i as f64 + 1.0) / (n as f64);
        let rid = db
            .record(
                &format!("certainty test memory {i}"),
                "semantic",
                0.5,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(1.0 + (i as f32) * 0.001, 8),
                "default",
                certainty,
                "general",
                "user",
                None,
            )
            .unwrap();
        rids.push(rid);
        // Space out created_at so order=recency has a meaningful ranking.
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    rids
}

#[test]
fn recall_certainty_min_filters_low_certainty() {
    // Issue #46: candidates whose certainty falls below the requested
    // floor must NOT appear in results. Seed 5 memories with certainty
    // 0.2, 0.4, 0.6, 0.8, 1.0; recall with certainty_min=0.5 must
    // return at most 3 (the 0.6/0.8/1.0 ones).
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let _rids = seed_for_certainty_test(&db, 5);

    let results = db
        .recall(
            &vec_seed(1.0, 8),
            10,
            None,
            None,
            false,
            false,
            None,
            true,
            None,
            None,
            None,
            Some(0.5), // certainty_min
            None,
            false,
            None, // event_after (#149)
            None, // event_before (#149)
        )
        .unwrap();

    assert!(
        !results.is_empty(),
        "expected at least one high-certainty result"
    );
    for r in &results {
        assert!(
            r.certainty >= 0.5,
            "certainty_min=0.5 must filter out cert={}: rid={}",
            r.certainty,
            r.rid
        );
    }
}

#[test]
fn recall_order_certainty_returns_results_in_certainty_desc() {
    // Issue #46: order="certainty" must re-sort the final top_k by
    // certainty descending. Seed 5 memories with varying certainty,
    // recall all 5 with order="certainty", assert the sequence is
    // monotonically non-increasing.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let _rids = seed_for_certainty_test(&db, 5);

    let results = db
        .recall(
            &vec_seed(1.0, 8),
            10,
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
            Some("certainty"),
            false,
            None, // event_after (#149)
            None, // event_before (#149)
        )
        .unwrap();

    assert!(
        results.len() >= 2,
        "need at least 2 results to verify ordering, got {}",
        results.len()
    );
    for w in results.windows(2) {
        assert!(
            w[0].certainty >= w[1].certainty,
            "order=certainty must be non-increasing; got [{}, {}]",
            w[0].certainty,
            w[1].certainty,
        );
    }
}

#[test]
fn recall_order_recency_returns_results_in_created_at_desc() {
    // Issue #46: order="recency" must re-sort the final top_k by
    // created_at descending (newest first). The seed helper spaces out
    // writes by 2ms each so created_at has a strict ordering.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let _rids = seed_for_certainty_test(&db, 5);

    let results = db
        .recall(
            &vec_seed(1.0, 8),
            10,
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
            Some("recency"),
            false,
            None, // event_after (#149)
            None, // event_before (#149)
        )
        .unwrap();

    assert!(
        results.len() >= 2,
        "need at least 2 results to verify ordering, got {}",
        results.len()
    );
    for w in results.windows(2) {
        assert!(
            w[0].created_at >= w[1].created_at,
            "order=recency must be non-increasing on created_at; got [{}, {}]",
            w[0].created_at,
            w[1].created_at,
        );
    }
}

#[test]
fn recall_order_first_mention_uses_metadata_then_created_at() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let inputs = [
        ("first concern", 300.0, Some(100.0)),
        ("second concern", 200.0, None),
        ("third concern", 100.0, Some(250.0)),
    ];
    for (text, created_at, first_mention_at) in inputs {
        let metadata = first_mention_at
            .map(|value| serde_json::json!({"first_mention_at": value}))
            .unwrap_or_else(empty_meta);
        db.record_with_idempotency(
            text,
            "semantic",
            0.5,
            0.0,
            604800.0,
            &metadata,
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "general",
            "inference",
            None,
            None,
            Some(created_at),
        )
        .unwrap();
    }

    for order in ["first_mention", "chronological"] {
        let results = db
            .recall(
                &vec_seed(1.0, 8),
                3,
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
                Some(order),
                false,
                None, // event_after (#149)
                None, // event_before (#149)
            )
            .unwrap();
        assert_eq!(
            results
                .iter()
                .map(|result| result.text.as_str())
                .collect::<Vec<_>>(),
            vec!["first concern", "second concern", "third concern"],
            "order={order} must use first_mention_at and fall back to created_at"
        );
    }
}

#[test]
fn recall_order_invalid_string_returns_invalid_input_error() {
    // Issue #46: unknown `order` values must surface as a typed
    // InvalidInput error rather than silently falling back to relevance.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let _rids = seed_for_certainty_test(&db, 3);

    let err = db
        .recall(
            &vec_seed(1.0, 8),
            10,
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
            Some("most_relevant"), // typo / not a valid order
            false,
            None, // event_after (#149)
            None, // event_before (#149)
        )
        .expect_err("invalid order string must return error");

    match err {
        crate::error::YantrikDbError::InvalidInput(msg) => {
            assert!(
                msg.contains("order") && msg.contains("most_relevant"),
                "error message should name the bad order value, got: {msg}"
            );
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn recall_default_order_is_relevance_unchanged() {
    // Issue #46: passing None for `order` must NOT change the existing
    // relevance-based behaviour. The first result's score must be >= the
    // last result's score (monotone non-increasing on score).
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let _rids = seed_for_certainty_test(&db, 5);

    let results = db
        .recall(
            &vec_seed(1.0, 8),
            10,
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
            None, // default order = relevance
            false,
            None, // event_after (#149)
            None, // event_before (#149)
        )
        .unwrap();

    assert!(
        results.len() >= 2,
        "need at least 2 results to verify default order, got {}",
        results.len()
    );
    for w in results.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "default order must be score-desc (relevance); got [{}, {}]",
            w[0].score,
            w[1].score,
        );
    }
}

// ── recall_as_of: bitemporal recall over the revision + link ledgers ──

#[test]
fn recall_as_of_returns_pre_correction_state() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "The deadline is March 1st",
            "semantic",
            0.9,
            0.0,
            604800.0,
            &serde_json::json!({"rev": "v1"}),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "work",
            "user",
            None,
        )
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(25));
    let t_mid = super::super::now();
    std::thread::sleep(std::time::Duration::from_millis(25));
    let generation = db.search_generation();
    db.correct_with_embedding(
        &rid,
        Some("The deadline is March 15th"),
        &vec_seed(1.0, 8),
        generation,
        None,
        None,
        None,
        "slip",
    )
    .unwrap();

    // Present-day recall sees the corrected text.
    let today = db
        .recall_as_of(&vec_seed(1.0, 8), 5, super::super::now(), None, None)
        .unwrap();
    assert_eq!(today.len(), 1);
    assert_eq!(today[0].text, "The deadline is March 15th");
    assert!(
        !today[0]
            .why_retrieved
            .iter()
            .any(|w| w.starts_with("as_of:")),
        "no rollback tag when nothing was rolled back"
    );

    // As of t_mid, the belief was the ORIGINAL text — the correction
    // archived exactly that state in record_revisions.
    let then = db
        .recall_as_of(&vec_seed(1.0, 8), 5, t_mid, None, None)
        .unwrap();
    assert_eq!(then.len(), 1);
    assert_eq!(then[0].text, "The deadline is March 1st");
    assert!((then[0].importance - 0.9).abs() < 1e-9);
    assert_eq!(
        then[0].metadata.get("rev").and_then(|v| v.as_str()),
        Some("v1")
    );
    assert!(
        then[0]
            .why_retrieved
            .iter()
            .any(|w| w.starts_with("as_of:")),
        "rolled-back hit is labeled"
    );

    // Before the record existed there was nothing to believe.
    let before = db
        .recall_as_of(&vec_seed(1.0, 8), 5, then[0].created_at - 10.0, None, None)
        .unwrap();
    assert!(before.is_empty());
}

#[test]
fn recall_as_of_respects_supersede_edge_time() {
    use crate::types::{LinkType, RecordLink};

    let db = YantrikDB::new(":memory:", 8).unwrap();
    let old = db
        .record(
            "Use the v1 endpoint",
            "semantic",
            0.8,
            0.0,
            604800.0,
            &serde_json::json!({}),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "work",
            "user",
            None,
        )
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(25));
    let t_mid = super::super::now();
    std::thread::sleep(std::time::Duration::from_millis(25));
    let new = db
        .record(
            "Use the v2 endpoint",
            "semantic",
            0.8,
            0.0,
            604800.0,
            &serde_json::json!({}),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "work",
            "user",
            None,
        )
        .unwrap();
    db.link(
        &new,
        &RecordLink {
            target_rid: old.clone(),
            link_type: LinkType::Supersedes,
        },
    )
    .unwrap();

    // As of t_mid: the old record was still current (the supersede edge
    // did not exist yet) and the successor did not exist at all.
    let then = db
        .recall_as_of(&vec_seed(1.0, 8), 5, t_mid, None, None)
        .unwrap();
    let then_rids: Vec<&str> = then.iter().map(|r| r.rid.as_str()).collect();
    assert!(
        then_rids.contains(&old.as_str()),
        "old was current at t_mid"
    );
    assert!(
        !then_rids.contains(&new.as_str()),
        "successor did not exist at t_mid"
    );

    // As of now: the edge exists, so the old record is out and the
    // successor is in.
    let today = db
        .recall_as_of(&vec_seed(1.0, 8), 5, super::super::now(), None, None)
        .unwrap();
    let today_rids: Vec<&str> = today.iter().map(|r| r.rid.as_str()).collect();
    assert!(today_rids.contains(&new.as_str()));
    assert!(!today_rids.contains(&old.as_str()));
}
