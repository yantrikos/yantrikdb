use super::*;

// ── #149 phase 2: valid-time filter on recall ────────────────────────
//
// `event_after` / `event_before` (epoch seconds) define the ELIGIBLE
// UNIVERSE before relevance ranking and top_k — filter-first, per the
// reviewer-confirmed contract:
//
//   - NULL `event_time_min` rows are EXCLUDED whenever either bound is
//     set (unknown-when is not in-window; a future opt-in may re-admit
//     them, never a default);
//   - inclusive interval overlap: `after` alone ⇒ max >= after,
//     `before` alone ⇒ min <= before, both ⇒ the intervals overlap,
//     boundary equality included;
//   - `event_after > event_before` is a typed InvalidInput error, not
//     an empty result;
//   - transaction-time filtering (`time_window`, `recall_as_of`) is a
//     SEPARATE axis and is untouched.
//
// The tests below are the six adversarial cases from the contract. The
// pin test is the load-bearing one: it is the case that post-filtering
// a bounded similarity pool cannot pass.
// =====================================================================

/// Caller-supplied event-time metadata. The caller owns all three keys
/// as one unit (merge_event_dates ownership rule), so all three are
/// supplied together.
fn event_meta(iso: &str, min: f64, max: f64) -> serde_json::Value {
    serde_json::json!({
        "event_dates": [iso],
        "event_time_min": min,
        "event_time_max": max,
    })
}

/// A unit vector nearly parallel to axis 0 — decoy similarity ~1.0
/// against an axis-0 query, each decoy distinct.
fn decoy_vec(i: usize, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[0] = 1.0;
    v[1 + (i % (dim - 1))] = 0.001 * (1.0 + (i / (dim - 1)) as f32);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

/// A unit vector with LOW (0.3) similarity to the axis-0 query — far
/// below every decoy, so it can never enter a bounded similarity pool
/// on merit.
fn low_sim_vec(dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[0] = 0.3;
    v[1] = (1.0f32 - 0.09).sqrt();
    v
}

fn axis0(dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[0] = 1.0;
    v
}

/// The full recall call with only the valid-time bounds varying.
/// `skip_reinforce` so repeated recalls in one test observe, not mutate.
fn recall_window(
    db: &YantrikDB,
    query: &[f32],
    top_k: usize,
    namespace: Option<&str>,
    event_after: Option<f64>,
    event_before: Option<f64>,
) -> crate::error::Result<Vec<RecallResult>> {
    db.recall(
        query,
        top_k,
        None,  // time_window (transaction time — separate axis)
        None,  // memory_type
        false, // include_consolidated
        false, // expand_entities
        None,  // query_text
        true,  // skip_reinforce
        namespace,
        None,  // domain
        None,  // source
        None,  // certainty_min
        None,  // order
        false, // include_superseded
        event_after,
        event_before,
    )
}

/// Record with a fixed low importance (0.4 — below every importance-
/// fallback threshold, so ONLY the vector pool or the #149 universe
/// lane can admit these rows; nothing else can mask a false negative).
fn put(db: &YantrikDB, text: &str, meta: &serde_json::Value, emb: &[f32]) -> String {
    db.record(
        text, "episodic", 0.4, 0.0, 604800.0, meta, emb, "default", 0.8, "general", "user", None,
    )
    .unwrap()
}

fn rid_set(results: &[RecallResult]) -> std::collections::HashSet<String> {
    results.iter().map(|r| r.rid.clone()).collect()
}

/// (a) THE PIN TEST — the case this issue exists to fix.
///
/// 80 in-cache decoys sit at similarity ~1.0 to the query and carry NO
/// event time; ONE relevant record carries an in-window event time but
/// similarity 0.3. With top_k=3 the unfiltered candidate pool is
/// `recall_fetch_plan`'s top_k*20 = 60 nearest — all decoys — so the
/// relevant record's similarity rank (81st) is BELOW the pool size.
/// An implementation that post-filters that bounded pool returns
/// nothing (the bounded-by-today's-top-k false negative); filter-first
/// must return the record because it is the only member of the
/// eligible universe.
#[test]
fn pin_in_window_record_below_the_similarity_pool_is_still_returned() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let query = axis0(8);

    for i in 0..80 {
        put(
            &db,
            &format!("decoy note {i}"),
            &empty_meta(),
            &decoy_vec(i, 8),
        );
    }
    let relevant = put(
        &db,
        "the outage happened back then",
        &event_meta("2001-09-09", 1_000_000.0, 1_000_000.0),
        &low_sim_vec(8),
    );

    // Precondition making the pin adversarial: unbounded recall at the
    // same top_k never surfaces the relevant record on similarity.
    let unbounded = recall_window(&db, &query, 3, Some("default"), None, None).unwrap();
    assert!(
        !rid_set(&unbounded).contains(&relevant),
        "precondition: the in-window record must NOT be reachable through \
         the unfiltered top-k similarity pool"
    );

    let bounded = recall_window(
        &db,
        &query,
        3,
        Some("default"),
        Some(999_000.0),
        Some(1_001_000.0),
    )
    .unwrap();
    assert_eq!(
        rid_set(&bounded),
        std::collections::HashSet::from([relevant.clone()]),
        "filter-first: the ONLY member of the eligible universe must be \
         returned even though its similarity rank is below the candidate \
         pool size — post-filtering a bounded pool fails exactly here"
    );
    // The universe lane stamps its admission.
    assert!(
        bounded[0]
            .why_retrieved
            .iter()
            .any(|w| w == "event_time_window"),
        "the direct-scored admission should carry the event_time_window why marker"
    );
}

/// (b) Better similarity does not buy eligibility: an out-of-window row
/// scoring far above every in-window row is excluded.
#[test]
fn higher_scoring_out_of_window_rows_are_excluded() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let query = axis0(8);

    // Near-perfect similarity, but the event sits OUTSIDE the window.
    let outside = put(
        &db,
        "perfect match, wrong era",
        &event_meta("1970-01-24", 2_000_000.0, 2_000_000.0),
        &decoy_vec(0, 8),
    );
    // Weak similarity, event inside the window.
    let inside = put(
        &db,
        "weak match, right era",
        &event_meta("1970-01-12", 1_000_000.0, 1_000_000.0),
        &low_sim_vec(8),
    );

    let results = recall_window(&db, &query, 10, None, Some(900_000.0), Some(1_100_000.0)).unwrap();
    let rids = rid_set(&results);
    assert!(rids.contains(&inside), "in-window row must be returned");
    assert!(
        !rids.contains(&outside),
        "out-of-window row must be excluded despite better similarity"
    );
}

/// (c) Boundary equality is INCLUDED on both edges: `max == after` and
/// `min == before` both qualify (inclusive interval overlap).
#[test]
fn boundary_equality_is_included() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let query = axis0(8);

    // Interval [1_000.0, 2_000.0].
    let interval = put(
        &db,
        "spanning event",
        &event_meta("1970-01-01", 1_000.0, 2_000.0),
        &decoy_vec(0, 8),
    );

    // after == event_time_max: overlap at exactly one point — included.
    let at_max = recall_window(&db, &query, 10, None, Some(2_000.0), None).unwrap();
    assert!(
        rid_set(&at_max).contains(&interval),
        "event_time_max == event_after must be included (inclusive)"
    );

    // before == event_time_min: overlap at exactly one point — included.
    let at_min = recall_window(&db, &query, 10, None, None, Some(1_000.0)).unwrap();
    assert!(
        rid_set(&at_min).contains(&interval),
        "event_time_min == event_before must be included (inclusive)"
    );

    // And one epsilon past each edge is OUT.
    let past_max = recall_window(&db, &query, 10, None, Some(2_000.5), None).unwrap();
    assert!(
        !rid_set(&past_max).contains(&interval),
        "event_after just past event_time_max must exclude"
    );
    let past_min = recall_window(&db, &query, 10, None, None, Some(999.5)).unwrap();
    assert!(
        !rid_set(&past_min).contains(&interval),
        "event_before just before event_time_min must exclude"
    );
}

/// (d) NULL event-time rows: excluded whenever EITHER bound is set,
/// still returned when no bound is set.
#[test]
fn null_event_time_rows_are_excluded_only_when_a_bound_is_set() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let query = axis0(8);

    let dated = put(
        &db,
        "dated row",
        &event_meta("1970-01-12", 1_000_000.0, 1_000_000.0),
        &decoy_vec(0, 8),
    );
    let undated = put(&db, "undated row", &empty_meta(), &decoy_vec(1, 8));

    // No bounds: both reachable.
    let open = recall_window(&db, &query, 10, None, None, None).unwrap();
    assert!(rid_set(&open).contains(&undated));
    assert!(rid_set(&open).contains(&dated));

    // after alone — a window the undated row would trivially "satisfy"
    // if NULL were treated as always-in: excluded instead.
    let after_only = recall_window(&db, &query, 10, None, Some(0.0), None).unwrap();
    assert!(
        !rid_set(&after_only).contains(&undated),
        "NULL event_time_min must be excluded when event_after is set"
    );
    assert!(rid_set(&after_only).contains(&dated));

    // before alone: same exclusion.
    let before_only = recall_window(&db, &query, 10, None, None, Some(2_000_000.0)).unwrap();
    assert!(
        !rid_set(&before_only).contains(&undated),
        "NULL event_time_min must be excluded when event_before is set"
    );
    assert!(rid_set(&before_only).contains(&dated));
}

/// (e) Half-open windows follow the overlap semantics: `after` alone
/// keeps everything ending at/after it, `before` alone keeps
/// everything starting at/before it.
#[test]
fn only_after_and_only_before_each_follow_overlap_semantics() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let query = axis0(8);

    let early = put(
        &db,
        "early event",
        &event_meta("1970-01-01", 1_000.0, 1_000.0),
        &decoy_vec(0, 8),
    );
    let middle = put(
        &db,
        "middle event",
        &event_meta("1970-01-01", 2_000.0, 2_000.0),
        &decoy_vec(1, 8),
    );
    let late = put(
        &db,
        "late event",
        &event_meta("1970-01-01", 3_000.0, 3_000.0),
        &decoy_vec(2, 8),
    );

    let after = recall_window(&db, &query, 10, None, Some(2_000.0), None).unwrap();
    assert_eq!(
        rid_set(&after),
        std::collections::HashSet::from([middle.clone(), late.clone()]),
        "after alone: event_time_max >= after (boundary in, earlier out)"
    );

    let before = recall_window(&db, &query, 10, None, None, Some(2_000.0)).unwrap();
    assert_eq!(
        rid_set(&before),
        std::collections::HashSet::from([early, middle]),
        "before alone: event_time_min <= before (boundary in, later out)"
    );
}

/// An inverted window is a caller error (typed InvalidInput), never an
/// empty result — empty would falsely claim "nothing happened then".
#[test]
fn inverted_window_is_invalid_input_not_empty() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let query = axis0(8);
    put(
        &db,
        "some event",
        &event_meta("1970-01-12", 1_000_000.0, 1_000_000.0),
        &decoy_vec(0, 8),
    );

    let err = recall_window(&db, &query, 10, None, Some(2_000.0), Some(1_000.0)).unwrap_err();
    assert!(
        matches!(err, crate::error::YantrikDbError::InvalidInput(_)),
        "event_after > event_before must be InvalidInput, got {err:?}"
    );
}

/// (f) Leader/follower ELIGIBLE-UNIVERSE parity: after replication
/// apply, the same bounded recall draws from the same eligible rid set
/// on both engines. One row is stamped by the datetext extractor from
/// prose (the natural path), one by caller-supplied keys, one is
/// undated, one is out-of-window.
///
/// DELIBERATELY NARROWED CONTRACT (review decision, #173): this proves
/// universe MEMBERSHIP parity, not returned-set parity under a tight
/// top_k. A follower's record ops carry only the embedding hash, so its
/// vectorless rows score 0.0 and RANKING can diverge from the leader
/// until embeddings materialize — see
/// `bounded_recall_follower_topk_ranking_divergence_is_deterministic`,
/// which pins that degradation explicitly. Full replica query parity
/// (embedding replication/materialization) is tracked separately.
#[test]
fn bounded_recall_is_replication_parity_safe() {
    use crate::replication::{apply_ops, extract_ops_since};

    let leader = YantrikDB::new(":memory:", 8).unwrap();
    let query = axis0(8);

    // Natural path: the datetext extractor stamps 2024-03-15 from prose.
    let rid_prose = put(
        &leader,
        "met Alice on 2024-03-15",
        &empty_meta(),
        &decoy_vec(0, 8),
    );
    // Caller-supplied keys, same window.
    let rid_caller = put(
        &leader,
        "spring planning session",
        &event_meta("2024-03-20", 1_710_892_800.0, 1_710_892_800.0),
        &decoy_vec(1, 8),
    );
    // Out-of-window (2023) and undated rows must be absent on BOTH.
    let rid_2023 = put(
        &leader,
        "an older event",
        &event_meta("2023-03-15", 1_678_838_400.0, 1_678_838_400.0),
        &decoy_vec(2, 8),
    );
    let rid_undated = put(&leader, "no date at all", &empty_meta(), &decoy_vec(3, 8));

    let follower = YantrikDB::new(":memory:", 8).unwrap();
    let ops = extract_ops_since(&leader.conn(), None, None, None, 100).unwrap();
    apply_ops(&follower, &ops).unwrap();

    // March 2024 window.
    let (after, before) = (1_709_251_200.0, 1_711_929_600.0);
    let on_leader = recall_window(&leader, &query, 10, None, Some(after), Some(before)).unwrap();
    let on_follower =
        recall_window(&follower, &query, 10, None, Some(after), Some(before)).unwrap();

    let expected: std::collections::HashSet<String> =
        [rid_prose.clone(), rid_caller.clone()].into();
    assert_eq!(
        rid_set(&on_leader),
        expected,
        "leader: exactly the two in-window rows (prose-stamped + caller-stamped)"
    );
    assert_eq!(
        rid_set(&on_follower),
        rid_set(&on_leader),
        "follower after apply_ops must return the same rid set for the same bounds"
    );
    // Guard the negative half explicitly on both engines.
    for (name, results) in [("leader", &on_leader), ("follower", &on_follower)] {
        let rids = rid_set(results);
        assert!(!rids.contains(&rid_2023), "{name}: 2023 row must be out");
        assert!(
            !rids.contains(&rid_undated),
            "{name}: undated row must be out"
        );
    }
}

/// The documented follower degradation the narrowed contract accepts
/// (review finding, #173): a follower's vectorless eligible rows all
/// score 0.0, so under a tight top_k its RANKING diverges from the
/// leader deterministically (rid tie-break) even though the eligible
/// universe is identical. Reproduces the reviewer's exact scenario:
/// orthogonal distractor recorded first (earlier rid), exact-vector
/// relevant row second, top_k=1 — leader returns the relevant row,
/// follower returns the distractor. If embedding materialization ever
/// lands and this test FAILS on the follower half, that is the signal
/// to restore full returned-set parity in the contract.
#[test]
fn bounded_recall_follower_topk_ranking_divergence_is_deterministic() {
    use crate::replication::{apply_ops, extract_ops_since};

    let leader = YantrikDB::new(":memory:", 8).unwrap();
    let query = axis0(8);

    // Distractor FIRST: earlier rid wins a 0.0-score tie-break. Low
    // similarity (0.3) so the leader unambiguously ranks it below the
    // exact-vector row. (NOT decoy_vec — that helper is deliberately
    // ~parallel to the query.)
    let rid_distractor = put(
        &leader,
        "in-window distractor",
        &event_meta("2024-03-10", 1_710_028_800.0, 1_710_028_800.0),
        &low_sim_vec(8),
    );
    let rid_relevant = put(
        &leader,
        "in-window relevant",
        &event_meta("2024-03-20", 1_710_892_800.0, 1_710_892_800.0),
        &query, // exact query vector
    );

    let follower = YantrikDB::new(":memory:", 8).unwrap();
    let ops = extract_ops_since(&leader.conn(), None, None, None, 100).unwrap();
    apply_ops(&follower, &ops).unwrap();

    let (after, before) = (1_709_251_200.0, 1_711_929_600.0);
    let on_leader = recall_window(&leader, &query, 1, None, Some(after), Some(before)).unwrap();
    let on_follower = recall_window(&follower, &query, 1, None, Some(after), Some(before)).unwrap();

    assert_eq!(
        on_leader[0].rid, rid_relevant,
        "leader ranks by real similarity"
    );
    assert_eq!(
        on_follower[0].rid, rid_distractor,
        "follower (vectorless, all 0.0) tie-breaks by rid — the documented \
         deterministic degradation; if this ever returns the relevant row, \
         embedding materialization landed and the contract can widen"
    );
}

/// Non-finite bounds are rejected up front (repo-wide caller-scalar
/// rule): NaN makes every comparison false, so it would silently pass
/// the inversion check and match nothing — a lie shaped like an empty
/// result.
#[test]
fn non_finite_bounds_are_invalid_scalars() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let query = axis0(8);
    for (after, before) in [
        (Some(f64::NAN), None),
        (None, Some(f64::NAN)),
        (Some(f64::INFINITY), None),
        (None, Some(f64::NEG_INFINITY)),
        (Some(f64::NAN), Some(f64::NAN)),
    ] {
        let err = recall_window(&db, &query, 5, None, after, before).unwrap_err();
        assert!(
            matches!(err, crate::error::YantrikDbError::InvalidScalar { .. }),
            "non-finite bound (after={after:?}, before={before:?}) must be \
             InvalidScalar, got {err:?}"
        );
    }
}
