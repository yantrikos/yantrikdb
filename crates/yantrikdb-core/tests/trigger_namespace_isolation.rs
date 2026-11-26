//! Namespace isolation for `think()` trigger artifacts.
//!
//! `memories.namespace` is the tenant boundary in this engine: the
//! consolidation path refuses a cluster whose members disagree on
//! namespace, and the claim-conflict scans join on `namespace` equality.
//! The trigger phase did not hold that line — `check_redundancy` loaded the
//! most recent active memories store-wide and paired all of them, so two
//! people recorded in one store produced a `redundancy` trigger whose
//! `context` carried BOTH texts.
//!
//! These tests pin the invariant at the artifact: no trigger that think()
//! returns may have source rids from more than one namespace, and the
//! same-namespace case must keep firing so the check cannot be "fixed" by
//! disabling it.

use std::collections::HashSet;

use yantrikdb::types::ThinkConfig;
use yantrikdb::YantrikDB;

const DIM: usize = 8;

/// One-hot-ish unit vector; equal vectors give cosine 1.0, which is well
/// above the redundancy threshold and keeps the test independent of any
/// downloaded embedder.
fn vec_on(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[axis % DIM] = 1.0;
    v
}

fn open(dir: &tempfile::TempDir) -> YantrikDB {
    YantrikDB::new(dir.path().join("ns.db").to_str().unwrap(), DIM).unwrap()
}

fn record(db: &YantrikDB, text: &str, namespace: &str, embedding: &[f32]) -> String {
    db.record(
        text,
        "semantic",
        0.7,
        0.0,
        604800.0,
        &serde_json::json!({}),
        embedding,
        namespace,
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap()
}

/// Trigger checks only: consolidation and the conflict scans already scope
/// by namespace, and they would otherwise add artifacts to the assertion.
fn trigger_only_config() -> ThinkConfig {
    ThinkConfig {
        run_consolidation: false,
        run_conflict_scan: false,
        run_pattern_mining: false,
        run_personality: false,
        // Keep consolidation triggers quiet: they count active memories.
        min_active_memories: i64::MAX,
        ..ThinkConfig::default()
    }
}

fn assert_single_namespace(
    triggers: &[yantrikdb::types::Trigger],
    owner: &[(String, &str)],
    case: &str,
) {
    let by_rid: std::collections::HashMap<&str, &str> =
        owner.iter().map(|(rid, ns)| (rid.as_str(), *ns)).collect();

    for trigger in triggers {
        let namespaces: HashSet<&str> = trigger
            .source_rids
            .iter()
            .filter_map(|rid| by_rid.get(rid.as_str()).copied())
            .collect();
        assert!(
            namespaces.len() <= 1,
            "{case}: trigger '{}' spans namespaces {:?} (rids {:?})",
            trigger.trigger_type,
            namespaces,
            trigger.source_rids
        );
    }
}

#[test]
fn identical_text_in_two_namespaces_yields_no_cross_namespace_trigger() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);

    let same = "Der Mailserver laeuft auf Debian 12 mit Postfix.";
    let a = record(&db, same, "alice", &vec_on(1));
    let b = record(&db, same, "bob", &vec_on(1));

    let result = db.think(&trigger_only_config()).unwrap();

    // Precondition: the two rows really are candidates for the redundancy
    // check (same text, identical embedding). Without this the test could
    // pass because nothing was comparable at all.
    assert!(!a.is_empty() && !b.is_empty() && a != b);

    assert_single_namespace(
        &result.triggers,
        &[(a, "alice"), (b, "bob")],
        "identical text in alice/bob",
    );
}

#[test]
fn near_identical_text_in_one_namespace_still_raises_redundancy() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);

    let a = record(
        &db,
        "Der Mailserver laeuft auf Debian 12 mit Postfix.",
        "alice",
        &vec_on(1),
    );
    let b = record(
        &db,
        "Der Mailserver laeuft unter Debian 12 mit Postfix.",
        "alice",
        &vec_on(1),
    );

    let result = db.think(&trigger_only_config()).unwrap();

    let redundancy: Vec<_> = result
        .triggers
        .iter()
        .filter(|t| t.trigger_type == "redundancy")
        .collect();
    assert!(
        !redundancy.is_empty(),
        "same-namespace near-duplicates must still raise redundancy; got {:?}",
        result
            .triggers
            .iter()
            .map(|t| t.trigger_type.as_str())
            .collect::<Vec<_>>()
    );
    assert_single_namespace(
        &result.triggers,
        &[(a, "alice"), (b, "alice")],
        "both in alice",
    );
}

#[test]
fn no_trigger_payload_carries_two_peoples_text() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir);

    // Distinctive strings, identical embeddings: the pair is exactly what
    // check_redundancy used to join store-wide. If the join is scoped, no
    // single payload can contain both.
    const ALICE: &str = "ALICE-SECRET der Server steht in Berlin.";
    const BOB: &str = "BOB-SECRET der Server steht in Wien.";
    let _ = record(&db, ALICE, "alice", &vec_on(3));
    let _ = record(&db, BOB, "bob", &vec_on(3));

    let result = db.think(&trigger_only_config()).unwrap();

    for trigger in &result.triggers {
        let payload = serde_json::to_string(&trigger.context).unwrap();
        let both = payload.contains("ALICE-SECRET") && payload.contains("BOB-SECRET");
        assert!(
            !both,
            "trigger '{}' carries both namespaces' text: {payload}",
            trigger.trigger_type
        );
    }
}
