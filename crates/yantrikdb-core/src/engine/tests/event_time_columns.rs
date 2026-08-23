use super::*;

// ── v48 (#149): valid time as first-class columns ────────────────────
//
// `memories.event_time_min` / `memories.event_time_max` mirror the
// metadata JSON keys of the same names, stamped by EVERY writer that
// persists the metadata column (base::datetext::event_time_bounds is the
// single source). The census test below is the category enforcer: it
// does not test any one writer — it asserts that NO row anywhere in the
// store carries event-time metadata without the columns, after driving
// memories through several distinct write paths. A new writer that
// forgets to stamp fails the census, not a code review.
// =====================================================================

/// Rows whose metadata JSON exposes an event time the columns do not carry.
/// Zero is the invariant.
fn census_violations(conn: &rusqlite::Connection) -> i64 {
    conn.query_row(
        "SELECT COUNT(*) FROM memories \
         WHERE json_valid(metadata) \
           AND json_extract(metadata, '$.event_time_min') IS NOT NULL \
           AND event_time_min IS NULL",
        [],
        |r| r.get(0),
    )
    .unwrap()
}

/// (column, json) pairs for one rid — the positive half of the invariant:
/// where both exist they must be EQUAL, not merely both present.
fn column_vs_json(
    conn: &rusqlite::Connection,
    rid: &str,
) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
    conn.query_row(
        "SELECT event_time_min, \
                CASE WHEN json_valid(metadata) \
                     THEN json_extract(metadata, '$.event_time_min') END, \
                event_time_max, \
                CASE WHEN json_valid(metadata) \
                     THEN json_extract(metadata, '$.event_time_max') END \
         FROM memories WHERE rid = ?1",
        [rid],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
    )
    .unwrap()
}

#[test]
fn census_every_memories_writer_stamps_the_event_time_columns() {
    use crate::replication::{apply_ops, extract_ops_since};

    let leader = YantrikDB::new(":memory:", 8).unwrap();

    // Path 1: plain record with a date-bearing text — merge_event_dates
    // stamps the JSON keys engine-side; the columns must follow.
    let rid_date = leader
        .record(
            "met Alice on 2024-03-15",
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

    // Path 2: caller-supplied event_dates metadata — the caller owns all
    // three keys (merge_event_dates ownership rule) and the columns must
    // mirror the caller's values.
    let caller_meta = serde_json::json!({
        "event_dates": ["2023-12-01"],
        "event_time_min": 1_701_388_800.0,
        "event_time_max": 1_701_388_800.0,
    });
    let rid_caller = leader
        .record(
            "quarterly planning kicked off",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &caller_meta,
            &vec_seed(2.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    // Path 3: a correct() that changes the text to a DIFFERENT date — the
    // event keys are re-derived from the corrected prose and the columns
    // must be re-stamped in the same transaction.
    let generation = leader.search_generation();
    leader
        .correct_with_embedding(
            &rid_date,
            Some("actually we met Alice on 2025-01-02"),
            &vec_seed(3.0, 8),
            generation,
            None,
            None,
            None,
            "the date was wrong",
        )
        .unwrap();

    // Path 4: a metadata-only correction whose patch carries NEW explicit
    // event keys (caller-owned) — the scalar correction path must re-stamp.
    leader
        .correct(
            &rid_caller,
            None,
            Some(&serde_json::json!({
                "event_dates": ["2024-01-01"],
                "event_time_min": 1_704_067_200.0,
                "event_time_max": 1_704_067_200.0,
            })),
            None,
            None,
            "the kickoff actually happened in January",
        )
        .unwrap();

    // Path 5: replication apply into a second engine — materialize_record
    // for both rows plus the replicated metadata-only correct arm. The
    // text-changing correct op is excluded: it replays exact embedding
    // bytes only when leader and follower share an embedder model, and
    // these embedder-less test engines cannot re-embed (the
    // bundled-embedder replication suite covers that arm's transport; the
    // stamping code under test is shared with the metadata-only arm).
    let follower = YantrikDB::new(":memory:", 8).unwrap();
    let ops: Vec<_> = extract_ops_since(&leader.conn(), None, None, None, 100)
        .unwrap()
        .into_iter()
        .filter(|o| o.op_type != "correct" || o.embedding.is_none())
        .collect();
    apply_ops(&follower, &ops).unwrap();

    // THE CENSUS. Not per-writer: any row on either engine whose JSON
    // carries event time without the columns is a failure, whoever wrote it.
    for (name, db) in [("leader", &leader), ("follower", &follower)] {
        let conn = db.conn();
        assert_eq!(
            census_violations(&conn),
            0,
            "{name}: a writer persisted event-time metadata without stamping the v48 columns; \
             every memories writer must bind event_time_bounds()"
        );
    }

    // Positive half — the columns EQUAL the JSON, on both engines, for both
    // the engine-derived and the caller-supplied rows.
    let expected_corrected =
        crate::base::datetext::extract_event_dates("actually we met Alice on 2025-01-02");
    assert_eq!(
        expected_corrected.len(),
        1,
        "precondition: the corrected text carries exactly one extractable date"
    );
    // The re-derived value itself is asserted on the LEADER only: the
    // follower's correct-apply path replays `metadata_merge` over its own
    // prior JSON and does not re-run merge_event_dates on a text change, so
    // its JSON keeps the pre-correction event keys (a pre-existing JSON
    // divergence, out of #149 phase 1's scope). What v48 guarantees on BOTH
    // engines is column == JSON, whatever each engine's JSON says.
    {
        let conn = leader.conn();
        let (_, json_min, _, _) = column_vs_json(&conn, &rid_date);
        assert_eq!(
            json_min,
            Some(expected_corrected[0].epoch),
            "leader: the correction must have re-derived the JSON event time from the new prose"
        );
    }
    for (name, db) in [("leader", &leader), ("follower", &follower)] {
        let conn = db.conn();

        let (col_min, json_min, col_max, json_max) = column_vs_json(&conn, &rid_date);
        assert!(
            json_min.is_some(),
            "{name}: precondition — the date-bearing row's JSON carries event time"
        );
        assert_eq!(
            col_min, json_min,
            "{name}: event_time_min column must equal the metadata JSON value"
        );
        assert_eq!(
            col_max, json_max,
            "{name}: event_time_max column must equal the metadata JSON value"
        );

        // rid_caller went through record (caller keys) THEN the metadata-only
        // correction (new caller keys) on both engines — the columns must
        // carry the corrected value, not the recorded one.
        let (col_min, json_min, col_max, json_max) = column_vs_json(&conn, &rid_caller);
        assert_eq!(
            col_min,
            Some(1_704_067_200.0),
            "{name}: the metadata-only correction must re-stamp event_time_min"
        );
        assert_eq!(
            json_min,
            Some(1_704_067_200.0),
            "{name}: corrected JSON min"
        );
        assert_eq!(
            col_max, json_max,
            "{name}: corrected max must match its JSON"
        );
    }
}
