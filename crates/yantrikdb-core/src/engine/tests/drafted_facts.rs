//! Session auto-capture's two halves: `draft_facts_from_summary` (the split)
//! and `record_drafted_facts` (the write, with caller-made vectors). They
//! exist so a Python-side embedder, which the engine cannot reach, drafts
//! the same memories `draft_memories_from_summary` does.

use super::*;

const SUMMARY: &str = "We decided to use keyset cursors for list_records. \
                       Alice will own the database migration next sprint. \
                       The production launch slipped to March 30 because of it.";

/// Every column a drafted memory's write path chooses, for one rid.
#[derive(Debug, PartialEq)]
struct DraftRow {
    text: String,
    memory_type: String,
    importance: f64,
    half_life: f64,
    certainty: f64,
    source: String,
    domain: String,
    namespace: String,
}

fn draft_row(db: &YantrikDB, rid: &str) -> DraftRow {
    let conn = db.conn();
    conn.query_row(
        "SELECT text, type, importance, half_life, certainty, source, domain, namespace \
         FROM memories WHERE rid = ?1",
        params![rid],
        |r| {
            Ok(DraftRow {
                text: r.get(0)?,
                memory_type: r.get(1)?,
                importance: r.get(2)?,
                half_life: r.get(3)?,
                certainty: r.get(4)?,
                source: r.get(5)?,
                domain: r.get(6)?,
                namespace: r.get(7)?,
            })
        },
    )
    .unwrap()
}

fn draft_meta(db: &YantrikDB, rid: &str) -> serde_json::Value {
    let raw: String = db
        .conn()
        .query_row(
            "SELECT metadata FROM memories WHERE rid = ?1",
            params![rid],
            |r| r.get(0),
        )
        .unwrap();
    serde_json::from_str(&raw).unwrap()
}

#[test]
fn draft_facts_from_summary_is_the_auto_capture_split() {
    let facts = YantrikDB::draft_facts_from_summary(SUMMARY);
    assert!(facts.len() >= 2, "summary split into facts: {facts:?}");
    assert_eq!(
        facts,
        crate::engine::split::segment_into_atomic_facts(SUMMARY, 120, 60),
        "same segmenter and targets as before the split was exposed"
    );
    assert!(facts.iter().any(|f| f.contains("migration")));
    assert!(YantrikDB::draft_facts_from_summary("").is_empty());
}

#[test]
fn record_drafted_facts_writes_provisional_auto_capture_memories() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let facts = YantrikDB::draft_facts_from_summary(SUMMARY);
    let drafts: Vec<(String, Vec<f32>)> = facts
        .iter()
        .enumerate()
        .map(|(i, f)| {
            // One axis per fact: unambiguous nearest neighbours below.
            let mut v = vec![0.0f32; 8];
            v[i % 8] = 1.0;
            (f.clone(), v)
        })
        .collect();

    let rids = db.record_drafted_facts(&drafts, "session", "work").unwrap();
    assert_eq!(rids.len(), facts.len(), "one memory per fact, in order");

    for (rid, fact) in rids.iter().zip(&facts) {
        let row = draft_row(&db, rid);
        assert_eq!(&row.text, fact);
        assert_eq!(row.memory_type, "semantic");
        assert_eq!(row.importance, 0.5);
        assert_eq!(row.half_life, 604_800.0);
        assert_eq!(row.certainty, 0.7);
        assert_eq!(row.source, "session_auto_capture");
        assert_eq!(row.domain, "work");
        assert_eq!(row.namespace, "session");
        let meta = draft_meta(&db, rid);
        assert_eq!(meta["provisional"], serde_json::json!(true));
        assert_eq!(meta["kind"], serde_json::json!("session_auto_capture"));
    }

    // The caller's vectors are the ones stored: the nearest record to the
    // second vector is the second fact.
    let hits = db
        .recall(
            &drafts[1].1,
            1,
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
            None,
            None,
        )
        .unwrap();
    assert_eq!(hits[0].rid, rids[1]);
}

#[test]
fn record_drafted_facts_with_nothing_writes_nothing() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    assert!(db
        .record_drafted_facts(&[], "session", "work")
        .unwrap()
        .is_empty());
    assert_eq!(db.stats(None).unwrap().active_memories, 0);
}

/// The two write paths store the same memory: everything but the rid,
/// timestamps and vector matches `draft_memories_from_summary`.
#[cfg(feature = "bundled-embedder")]
#[test]
fn record_drafted_facts_matches_draft_memories_from_summary() {
    let engine = YantrikDB::with_default(":memory:").unwrap();
    let engine_rids = engine
        .draft_memories_from_summary(SUMMARY, "session", "work")
        .unwrap();

    let caller = YantrikDB::with_default(":memory:").unwrap();
    let drafts: Vec<(String, Vec<f32>)> = YantrikDB::draft_facts_from_summary(SUMMARY)
        .into_iter()
        .map(|f| {
            let v = caller.embed(&f).unwrap();
            (f, v)
        })
        .collect();
    let caller_rids = caller
        .record_drafted_facts(&drafts, "session", "work")
        .unwrap();

    assert_eq!(engine_rids.len(), caller_rids.len());
    for (a, b) in engine_rids.iter().zip(&caller_rids) {
        assert_eq!(draft_row(&engine, a), draft_row(&caller, b));
        assert_eq!(draft_meta(&engine, a), draft_meta(&caller, b));
    }
}

/// A drafted fact longer than the known embedder window is counted as a
/// truncated write, as the engine path counts it, so
/// stats().embedder_truncated_writes sees Python-embedded drafts too.
#[test]
fn record_drafted_facts_counts_a_fact_past_the_window_as_truncated() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.embedder_window_chars
        .store(40, std::sync::atomic::Ordering::Relaxed);
    let before = db.stats(None).unwrap().embedder_truncated_writes;
    let long = "the migration plan has many steps and each one needs an owner".to_string();
    let short = "short fact".to_string();
    db.record_drafted_facts(
        &[(long, vec_seed(1.0, 8)), (short, vec_seed(2.0, 8))],
        "session",
        "work",
    )
    .unwrap();
    let after = db.stats(None).unwrap().embedder_truncated_writes;
    assert_eq!(after - before, 1, "only the fact past the window counts");
}
