//! v0.13.1 explain surface (co-iteration wheel 2, spec locked with
//! hermes 2026-08-06). What these tests pin, in the spec's words:
//! the pool is snapshotted where admission can be judged (not just
//! survivors), every row carries the stable rid, lane admission is a
//! SET distinct from numeric contributions, and a lane that never ran
//! says so with its precondition — never an ambiguous zero.

use super::*;
use helpers::empty_meta;

fn seeded_db() -> YantrikDB {
    let db = YantrikDB::with_default(":memory:").unwrap();
    for i in 0..30 {
        db.record_text(
            &format!("Taylor reports to Grace on the platform team, memo {i}"),
            "semantic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    }
    for i in 0..10 {
        db.record_text(
            &format!("unrelated note about the quarterly budget, revision {i}"),
            "semantic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    }
    db
}

#[test]
fn explain_pool_is_a_superset_of_results_and_marks_selection() {
    let db = seeded_db();
    let (results, explain) = db
        .recall_text_explained("Who does Taylor report to?", 5, None, false, true)
        .unwrap();

    assert_eq!(explain.retrieval_limits.requested_top_k, 5);
    assert_eq!(explain.retrieval_limits.requested_candidates, 100);
    assert_eq!(explain.retrieval_limits.index_len, 40);
    assert_eq!(explain.retrieval_limits.fetch_k, 40);
    assert!(!explain.retrieval_limits.has_post_filters);
    assert!(!explain.retrieval_limits.cap_bound);

    assert!(!results.is_empty(), "seeded corpus must return results");
    assert!(
        explain.pool.len() >= results.len(),
        "the pool is the set that ENTERS selection — it cannot be \
         smaller than what survived it"
    );
    let pool_rids: std::collections::HashSet<&str> =
        explain.pool.iter().map(|p| p.rid.as_str()).collect();
    for r in &results {
        assert!(
            pool_rids.contains(r.rid.as_str()),
            "every returned rid must appear in the pool"
        );
    }
    let selected: Vec<&str> = explain
        .pool
        .iter()
        .filter(|p| p.selected)
        .map(|p| p.rid.as_str())
        .collect();
    let result_rids: std::collections::HashSet<&str> =
        results.iter().map(|r| r.rid.as_str()).collect();
    assert_eq!(
        selected.len(),
        result_rids.len(),
        "selected marks exactly the survivors"
    );
    for rid in selected {
        assert!(result_rids.contains(rid));
    }
}

#[test]
fn explain_rows_carry_rid_lanes_and_ranks() {
    let db = seeded_db();
    let (_, explain) = db
        .recall_text_explained("Who does Taylor report to?", 5, None, false, true)
        .unwrap();

    for (i, row) in explain.pool.iter().enumerate() {
        assert!(!row.rid.is_empty(), "stable rid on every explain row");
        assert!(
            !row.lanes_admitted.is_empty(),
            "every pooled candidate was admitted by SOME lane: {row:?}"
        );
        assert_eq!(
            row.rank_post_fusion, i,
            "post-fusion rank is the pool's comparator order"
        );
    }
    // The pre-fusion ranks are a permutation of the pool positions.
    let mut pre: Vec<usize> = explain.pool.iter().map(|p| p.rank_pre_fusion).collect();
    pre.sort_unstable();
    assert_eq!(pre, (0..explain.pool.len()).collect::<Vec<_>>());
    assert!(!explain.comparator.is_empty());
    assert!(
        explain.score_algebra.contains("do NOT sum"),
        "the algebra must refuse the invalid derivation by name"
    );
}

#[test]
fn never_ran_lanes_name_their_precondition() {
    let db = seeded_db();
    // expand_entities=false → the graph lane must report never_ran with
    // the flag named, NOT an ambiguous zero.
    let (_, explain) = db
        .recall_text_explained("Who does Taylor report to?", 5, None, false, true)
        .unwrap();
    let graph = &explain.lanes["graph"];
    assert_eq!(graph.status, "never_ran");
    assert_eq!(graph.reason.as_deref(), Some("expand_entities=false"));

    // Neutral query → valence scan names the neutral band.
    let valence = &explain.lanes["valence_scan"];
    assert_eq!(valence.status, "never_ran");
    assert!(valence.reason.as_deref().unwrap().contains("neutral band"));

    // The FTS lane ran (query text present, keywords extractable).
    let fts = &explain.lanes["fts"];
    assert_eq!(fts.status, "ran", "keyword query over seeded text: {fts:?}");
    assert!(explain.bm25_near_best_fraction.is_some());

    // With expand_entities=true the graph lane must not report never_ran.
    let (_, explain2) = db
        .recall_text_explained("Who does Taylor report to?", 5, None, true, true)
        .unwrap();
    assert_ne!(explain2.lanes["graph"].status, "never_ran");
}

#[test]
fn explain_skip_reinforce_leaves_access_counts_alone() {
    let db = seeded_db();
    let before: Vec<(String, u32)> = {
        let conn = db.conn();
        let mut stmt = conn
            .prepare("SELECT rid, access_count FROM memories ORDER BY rid")
            .unwrap();
        stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    };
    let _ = db
        .recall_text_explained("Who does Taylor report to?", 5, None, false, true)
        .unwrap();
    let after: Vec<(String, u32)> = {
        let conn = db.conn();
        let mut stmt = conn
            .prepare("SELECT rid, access_count FROM memories ORDER BY rid")
            .unwrap();
        stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    };
    assert_eq!(
        before, after,
        "an explain probe observes the store; it must not mutate it"
    );
}
