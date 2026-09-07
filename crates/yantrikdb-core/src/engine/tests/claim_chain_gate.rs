//! The claim-chain eligibility gate, end to end: the v53 grounding
//! column, the durable mode, the shadow counters and enforcement on a
//! real recall.

use crate::{ChainGateMode, StatedClaim, YantrikDB};
use rusqlite::params;

fn rec(db: &YantrikDB, text: &str) -> String {
    db.record_text(
        text,
        "semantic",
        0.5,
        0.0,
        604800.0,
        &serde_json::json!({}),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap()
}

fn whys_for(db: &YantrikDB, query: &str, rid: &str) -> Vec<String> {
    let resp = db
        .recall_with_response(
            &db.embed(query).unwrap(),
            10,
            None,
            None,
            false,
            true,
            Some(query),
            true,
            None,
            None,
            None,
        )
        .unwrap();
    resp.results
        .iter()
        .find(|r| r.rid == rid)
        .map(|r| r.why_retrieved.clone())
        .unwrap_or_default()
}

fn claim_whys(whys: &[String]) -> Vec<&str> {
    whys.iter()
        .filter(|w| w.starts_with("claims_match"))
        .map(String::as_str)
        .collect()
}

#[test]
fn grounding_marks_cooperative_claims_and_nothing_else() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    let rid = rec(&db, "Sarah works at Google. Sarah lives in Berlin.");
    db.attach_claims(
        &rid,
        &[StatedClaim {
            src: "Sarah".into(),
            rel_type: "works_at".into(),
            dst: "Google".into(),
            polarity: 1,
            valid_from: None,
            valid_to: None,
        }],
    )
    .unwrap();
    // The heuristic extractor mints the same facts under its own label.
    db.reextract_claims(None, false).unwrap();
    let claims = db.get_claims("Sarah", None).unwrap();
    let mut by_extractor: Vec<(String, String, i64)> = claims
        .iter()
        .map(|c| {
            (
                c["extractor"].as_str().unwrap().to_string(),
                c["rel_type"].as_str().unwrap().to_string(),
                c["grounding"].as_i64().unwrap(),
            )
        })
        .collect();
    by_extractor.sort();
    assert!(
        by_extractor.contains(&("agent_stated".into(), "works_at".into(), 1)),
        "{by_extractor:?}"
    );
    assert!(
        by_extractor
            .iter()
            .filter(|(e, _, _)| e == "heuristic_v1")
            .all(|(_, _, g)| *g == 0),
        "heuristic rows are ungrounded until an extractor validates its bindings: {by_extractor:?}"
    );
    assert!(
        by_extractor.iter().any(|(e, _, _)| e == "heuristic_v1"),
        "fixture must mint at least one heuristic claim: {by_extractor:?}"
    );
}

#[test]
fn mode_defaults_to_shadow_and_persists_across_opens() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gate.db");
    let path = path.to_str().unwrap();
    {
        let db = YantrikDB::with_default(path).unwrap();
        assert_eq!(db.claim_chain_gate_mode(), ChainGateMode::Shadow);
        assert_eq!(db.stats(None).unwrap().claim_chain_gate_mode, "shadow");
        db.set_claim_chain_gate_mode(ChainGateMode::Enforce)
            .unwrap();
    }
    let db = YantrikDB::with_default(path).unwrap();
    assert_eq!(db.claim_chain_gate_mode(), ChainGateMode::Enforce);
    assert_eq!(db.stats(None).unwrap().claim_chain_gate_mode, "enforce");
}

/// The production repro on a live engine: two ungrounded claims chain
/// junk onto a query about CT128. Shadow admits and counts; enforce
/// refuses; a cooperative claim about the same entity still comes through.
#[test]
fn enforce_stops_the_pypi_chain_and_keeps_cooperative_evidence() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    let run_rid = rec(
        &db,
        "Deploy note: core runs CT128 dogfood after the PyPI publish.",
    );
    let work_rid = rec(
        &db,
        "PyPI and latest release both 0.15.6. Re-verified: 'Sarah works at Google'.",
    );
    // A named object on purpose: the lane's phantom filter still refuses a
    // value object (`0.21.2`) as an endpoint at read time — a pre-existing
    // limit, noted for the next step, not this gate's concern.
    let coop_rid = rec(&db, "CT128 runs Hermes now, verified by content.");
    {
        let conn = db.conn.lock();
        for (name, etype, mc) in [
            ("CT128", "tech", 9),
            ("PyPI", "org", 4),
            ("Google", "org", 2),
        ] {
            conn.execute(
                "INSERT OR REPLACE INTO entities (name, entity_type, mention_count, first_seen, last_seen) \
                 VALUES (?1, ?2, ?3, 1.0, 1.0)",
                params![name, etype, mc],
            )
            .unwrap();
        }
        for (cid, src, dst, rel, rid) in [
            ("cRun", "PyPI", "CT128", "runs", run_rid.as_str()),
            ("cWork", "PyPI", "Google", "works_at", work_rid.as_str()),
        ] {
            conn.execute(
                "INSERT INTO claims (claim_id, src, dst, rel_type, weight, created_at, \
                 extractor, source_memory_rid) VALUES (?1, ?2, ?3, ?4, 1.0, 1.0, 'heuristic_v1', ?5)",
                params![cid, src, dst, rel, rid],
            )
            .unwrap();
        }
    }
    db.rebuild_graph_index().unwrap();
    let query = "What does CT128 run?";

    // Shadow (the default): the junk path is still built, and counted.
    let run_whys = whys_for(&db, query, &run_rid);
    assert!(
        claim_whys(&run_whys)
            .iter()
            .any(|w| w.contains("PyPI -runs-> CT128")),
        "shadow admits the hop-1 claim: {run_whys:?}"
    );
    let work_whys = whys_for(&db, query, &work_rid);
    assert!(
        claim_whys(&work_whys)
            .iter()
            .any(|w| w.contains("PyPI -works_at-> Google") && w.contains("(path via PyPI")),
        "shadow builds the junk path: {work_whys:?}"
    );
    let stats = db.stats(None).unwrap();
    for key in ["hop1:ungrounded", "seed:ungrounded", "hop2:ungrounded"] {
        assert!(
            stats
                .claim_chain_gate_suppressed_since_boot
                .get(key)
                .copied()
                .unwrap_or(0)
                > 0,
            "shadow counts {key}: {:?}",
            stats.claim_chain_gate_suppressed_since_boot
        );
    }

    // Enforce: neither PyPI edge reaches the reader.
    db.set_claim_chain_gate_mode(ChainGateMode::Enforce)
        .unwrap();
    assert!(claim_whys(&whys_for(&db, query, &run_rid)).is_empty());
    assert!(claim_whys(&whys_for(&db, query, &work_rid)).is_empty());

    // A cooperative claim about CT128 is grounded and still admitted.
    let report = db
        .attach_claims(
            &coop_rid,
            &[StatedClaim {
                src: "CT128".into(),
                rel_type: "runs".into(),
                dst: "Hermes".into(),
                polarity: 1,
                valid_from: None,
                valid_to: None,
            }],
        )
        .unwrap();
    assert_eq!(report.accepted.len(), 1, "{:?}", report.rejected);
    let coop_whys = whys_for(&db, query, &coop_rid);
    assert!(
        claim_whys(&coop_whys)
            .iter()
            .any(|w| w.contains("CT128 -runs-> Hermes")),
        "grounded claim survives enforce: {coop_whys:?}"
    );
}
