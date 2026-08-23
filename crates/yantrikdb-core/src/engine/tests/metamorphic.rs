//! Metamorphic retrieval invariants.
//!
//! WHY THIS FILE EXISTS. On 2026-08-13 a phantom entity `AT` was found in a
//! production store, matching EVERY query containing the ordinary word "at" —
//! it returned a real-estate tax memo for "encryption at rest and key
//! rotation". Every unit test passed. The 400-query flagship benchmark could
//! not see it. It was found by an agent noticing its own memory was wrong.
//!
//! No example-based test would have caught it, because nobody writes a test
//! asserting "the word 'at' does not matter". That is the point of a
//! metamorphic test: it asserts a RELATION BETWEEN TWO RETRIEVALS rather than
//! a specific output, so it needs no golden answers and no labelled set, and
//! it holds for any corpus.
//!
//! These are deliberately LOOSE. A static embedder genuinely shifts when a
//! token is appended — mean pooling gives the extra token 1/(n+1) of the
//! weight — so demanding identical output would make the tests flaky and they
//! would be deleted.
//!
//! HONEST LIMITATION, MEASURED, NOT ASSUMED. The two gates below DO NOT catch
//! the stoplist bug. Restoring the case-sensitive comparison — with graph
//! expansion on and a phantom `AT` carrying six mentions — leaves both of them
//! passing. The only probe that fired was the length-matched counterfactual,
//! and at n=8 it measures a one-query difference, which is noise; it is
//! `#[ignore]`d below rather than loosened into a false pass.
//!
//! The reason they do not fire is itself informative: the bounded graph
//! multiplier shipped alongside the entity fix caps any phantom's influence at
//! +12.5%, so a phantom can exist and still fail to move the ranking. The two
//! fixes are complementary, and these invariants hold FOR THE RIGHT REASON on
//! current code.
//!
//! So treat what follows as EXECUTABLE SPECIFICATION of properties retrieval
//! must have, and as a gate against a severe regression — not as proof the
//! historical defect is covered. Claiming otherwise would repeat the mistake
//! this file exists to document: believing an instrument sees something
//! without checking that it can.

use super::*;

/// Distinct topics with no shared vocabulary, so a ranking change means a
/// retrieval change rather than two near-ties swapping.
#[cfg(feature = "bundled-embedder")]
fn seed_corpus(db: &YantrikDB) {
    // MUST be record_batch, not record_text — but not for the reason it first
    // appeared. record_batch extracts entities INLINE; single writes ENQUEUE
    // that work for the materializer thread (Phase 4.3 Commit B), so a store
    // written by record_text has zero entities until something drains the
    // queue. Verified: after record_text the oplog holds one unapplied op, and
    // apply_pending_ops_once turns 0 entities into 4.
    //
    // A test that never spawns a materializer therefore sees an empty graph.
    // An earlier version of this file seeded with record_text, found no
    // entities, and passed with the bug deliberately restored — there was no
    // phantom to find. Seeding through the inline path removes the dependency
    // on background work rather than papering over it.
    let texts = [
        "The postgres migration finished on the platform team's staging cluster",
        "Alice Chen prefers dark mode in the editor and a 14 inch laptop",
        "Encryption at rest uses AES-256-GCM with per-database keys",
        "The quarterly revenue forecast was revised upward by the finance group",
        "Sourdough starter needs feeding every twelve hours at room temperature",
        "The flight to Lisbon leaves from terminal two on Sunday morning",
        "Rust borrow checker errors usually mean a lifetime is too short",
        "The dentist appointment moved to the second Tuesday of next month",
        "Kubernetes ingress routes traffic by hostname and path prefix",
        "My grandmother's piano was tuned by a shop on Fillmore Street",
        // THE PRECONDITION. Agent-written memory is full of shouty headings,
        // and that is what mints phantoms: an ALL-CAPS function word survives
        // a case-sensitive stoplist and becomes an entity that then matches
        // every query containing the ordinary word.
        // The bare phantom needs an all-caps function word surrounded by
        // LOWERCASE. "AT THE STANDUP" yields the multi-word entity
        // "AT THE STANDUP"; only "AT the standup" yields the bare "AT" that
        // then matches every query containing the ordinary word "at".
        // THE PRECONDITION, AND ITS DOSE. A phantom is only dangerous in
        // proportion to how many records it links: in production `AT` carried
        // TEN mentions and bridged unrelated records. With one mention it is
        // harmless, and an earlier version of this corpus produced exactly one
        // and therefore passed with the bug restored. These records each carry
        // a bare all-caps function word surrounded by lowercase, on subjects
        // unrelated to any test query, so the phantom becomes a real bridge.
        "We agreed AT the standup that the analytics dashboard needs a rewrite.",
        "The vendor renewed AT the last minute after finance objected twice.",
        "She left AT dawn to catch the ferry across the estuary.",
        "The kettle whistles AT precisely the wrong moment every morning.",
        "He proposed AT the summit of a hill above the reservoir.",
        "The bells ring AT noon in the village below the monastery.",
        "The nightly build did NOT pass because a certificate expired quietly.",
        "The archive was NOT indexed before the storage tier was rotated.",
    ];
    let inputs: Vec<RecordInput> = texts
        .iter()
        .map(|t| RecordInput {
            text: (*t).into(),
            memory_type: "semantic".into(),
            importance: 0.5,
            valence: 0.0,
            half_life: 604800.0,
            metadata: empty_meta(),
            embedding: db.embed(t).expect("embed"),
            namespace: "default".into(),
            certainty: 0.8,
            domain: "general".into(),
            source: "user".into(),
            emotional_state: None,
            idempotency_key: None,
            created_at: None,
        })
        .collect();
    db.record_batch(&inputs).expect("record_batch");
}

/// Retrieve WITH GRAPH EXPANSION ON.
///
/// This is load-bearing and was got wrong once. The first version of these
/// tests called `recall_text`, which takes the DEFAULT path where
/// `expand_entities = false` — so phantom entities cannot influence
/// retrieval, and every test passed with the bug deliberately restored.
///
/// That is the same reason the 400-query benchmark never saw the defect: the
/// instrument was pointed at a configuration the failure does not live in.
/// A metamorphic test aimed at the wrong path is not a weaker test, it is a
/// test of something else entirely.
#[cfg(feature = "bundled-embedder")]
fn top_rids(db: &YantrikDB, query: &str, k: usize) -> Vec<String> {
    let embedding = db.embed(query).expect("embed");
    db.recall(
        &embedding,
        k,
        None,        // time_window
        None,        // memory_type
        false,       // include_consolidated
        true,        // expand_entities — THE POINT
        Some(query), // query_text, so the lexical + entity lanes run
        true,        // skip_reinforce: determinism across the paired calls
        None,
        None,
        None,
        None,
        None,
        false,
        None, // event_after (#149)
        None, // event_before (#149)
    )
    .expect("recall")
    .into_iter()
    .map(|r| r.rid)
    .collect()
}

#[cfg(feature = "bundled-embedder")]
fn overlap(a: &[String], b: &[String]) -> usize {
    a.iter().filter(|r| b.contains(r)).count()
}

/// THE INVARIANT THE STOPLIST BUG BROKE.
///
/// Appending a function word to a query must not change which memories come
/// back. When `AT` was an entity, adding " at" to a query injected every
/// record linked to that phantom — which was most of them.
#[cfg(feature = "bundled-embedder")]
#[test]
fn appending_a_stopword_does_not_change_what_is_retrieved() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    seed_corpus(&db);

    for query in [
        "postgres migration staging",
        "what does Alice prefer",
        "encryption keys",
        "revenue forecast",
    ] {
        let base = top_rids(&db, query, 5);
        for stopword in ["at", "the", "did", "not"] {
            let perturbed = top_rids(&db, &format!("{query} {stopword}"), 5);
            let shared = overlap(&base, &perturbed);
            assert!(
                shared >= 3,
                "appending {stopword:?} to {query:?} changed retrieval: \
                 only {shared}/5 shared.\n  base      = {base:?}\n  perturbed = {perturbed:?}\n\
                 A function word became a join key — check entity extraction \
                 and the graph lane."
            );
            assert!(
                base.first().map_or(false, |t| perturbed.contains(t)),
                "appending {stopword:?} to {query:?} evicted the top hit entirely.\n  \
                 base = {base:?}\n  perturbed = {perturbed:?}"
            );
        }
    }
}

/// Query casing must not change retrieval. The stoplist bug was a CASING bug —
/// "At" was stripped and "AT" was not — so casing invariance is the property
/// that most directly encodes what went wrong.
#[cfg(feature = "bundled-embedder")]
#[test]
fn query_casing_does_not_change_what_is_retrieved() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    seed_corpus(&db);

    for query in [
        "Alice Chen dark mode",
        "encryption at rest",
        "postgres migration",
    ] {
        let base = top_rids(&db, query, 5);
        for variant in [query.to_lowercase(), query.to_uppercase()] {
            let perturbed = top_rids(&db, &variant, 5);
            let shared = overlap(&base, &perturbed);
            assert!(
                shared >= 4,
                "casing changed retrieval for {query:?} -> {variant:?}: \
                 {shared}/5 shared.\n  base = {base:?}\n  perturbed = {perturbed:?}"
            );
        }
    }
}

/// THE AUTHORING-SIDE HALF, as a CONTROLLED counterfactual.
///
/// An agent writing "USER MUST UPDATE MCP CONFIG" created a multi-word entity
/// that joined unrelated records. But "does a shouty record appear in the top
/// 3" is not a test — with a small corpus the top 3 is a large fraction of it,
/// and any record appears sometimes. That version of this test failed for
/// exactly that reason and proved nothing.
///
/// So vary ONE thing. Two records with the SAME body and the SAME length,
/// differing only in whether the heading is ALL-CAPS. If capitalisation alone
/// makes a record more retrievable by unrelated queries, that is the defect;
/// if both surface equally, the corpus is just small.
/// NOT A GATE — a diagnostic, deliberately `#[ignore]`d. Run with
/// `cargo test -- --ignored an_all_caps`.
///
/// With every fix in place this measures shouty 3/8 versus calm 2/8: a
/// ONE-QUERY difference at n=8, which is noise. The assertion below fails on
/// any difference at all, so as a gate it would be flaky, and the two
/// available cures are both wrong — loosening the threshold until it passes is
/// fitting the test to the code, and deleting it discards a real question.
///
/// The question it asks is real and still open: does ALL-CAPS text embed more
/// generically (toward an OOV/centroid region) and therefore behave like a
/// hub? Answering it needs the treatment both external reviewers prescribed —
/// k-occurrence skewness against a null model, length-matched counterfactuals,
/// bootstrap by query — not a boolean in a unit test. Kept here, runnable, so
/// the question is not lost.
#[cfg(feature = "bundled-embedder")]
#[test]
#[ignore]
fn an_all_caps_heading_does_not_outrank_its_sentence_case_twin() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    seed_corpus(&db);

    let body = "The staging cluster reboots on Fridays and the runbook lives in the wiki.";
    let shouty = db
        .record_text(
            &format!("IMPORTANT NOTE ABOUT THE DEPLOYMENT PROCESS AND WHAT USERS MUST DO. {body}"),
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
    let calm = db
        .record_text(
            &format!(
                "Important note about the release process and what operators should do. {body}"
            ),
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

    // Queries sharing no subject with either record.
    // More queries, not a looser threshold. At three queries a single
    // incidental appearance decides the outcome, which is noise rather than
    // evidence — and loosening the assertion instead would be fitting the
    // test to the code.
    let unrelated = [
        "sourdough starter feeding",
        "piano tuning Fillmore",
        "dentist appointment",
        "flight to Lisbon terminal",
        "borrow checker lifetime",
        "quarterly revenue forecast",
        "dark mode editor preference",
        "ingress hostname routing",
    ];
    let mut shouty_hits = 0;
    let mut calm_hits = 0;
    for query in unrelated {
        let hits = top_rids(&db, query, 5);
        if hits.contains(&shouty) {
            shouty_hits += 1;
        }
        if hits.contains(&calm) {
            calm_hits += 1;
        }
    }

    assert!(
        shouty_hits <= calm_hits,
        "CAPITALISATION ALONE made a record more retrievable by unrelated queries:          the ALL-CAPS variant surfaced for {shouty_hits}/{} unrelated queries against          {calm_hits}/{} for its sentence-case twin with the same body and length.          Check entity extraction — a heading is becoming a join key.",
        unrelated.len(),
        unrelated.len()
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
#[ignore]
fn diag_what_entities_exist() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    seed_corpus(&db);
    let conn = db.conn();
    let mut stmt = conn
        .prepare("SELECT name, mention_count FROM entities ORDER BY mention_count DESC")
        .unwrap();
    let rows: Vec<(String, i64)> = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();
    drop(stmt);
    drop(conn);
    println!("ENTITIES ({}):", rows.len());
    for (n, c) in &rows {
        println!("  {n:?} x{c}");
    }
    let a = top_rids(&db, "postgres migration staging", 5);
    let b = top_rids(&db, "postgres migration staging at", 5);
    println!("base      = {a:?}");
    println!("perturbed = {b:?}");
    println!("shared    = {}", overlap(&a, &b));
}
#[cfg(feature = "bundled-embedder")]
#[test]
#[ignore]
fn diag_which_write_paths_populate_the_graph() {
    let count = |db: &YantrikDB| -> i64 {
        let conn = db.conn();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM entities", [], |r| r.get(0))
            .unwrap();
        let l: i64 = conn
            .query_row("SELECT COUNT(*) FROM memory_entities", [], |r| r.get(0))
            .unwrap();
        drop(conn);
        println!("    entities={n} links={l}");
        n
    };
    let text = "Alice Chen met Bob Smith at Yantrik Systems in San Francisco";

    println!("record_text (single):");
    let db1 = YantrikDB::with_default(":memory:").unwrap();
    db1.record_text(
        text,
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
    let a = count(&db1);

    println!("record (single, explicit embedding):");
    let db2 = YantrikDB::with_default(":memory:").unwrap();
    let e = db2.embed(text).unwrap();
    db2.record(
        text,
        "semantic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &e,
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();
    let b = count(&db2);

    println!("record_batch (one input):");
    let db3 = YantrikDB::with_default(":memory:").unwrap();
    let e3 = db3.embed(text).unwrap();
    db3.record_batch(&[RecordInput {
        text: text.into(),
        memory_type: "semantic".into(),
        importance: 0.5,
        valence: 0.0,
        half_life: 604800.0,
        metadata: empty_meta(),
        embedding: e3,
        namespace: "default".into(),
        certainty: 0.8,
        domain: "general".into(),
        source: "user".into(),
        emotional_state: None,
        idempotency_key: None,
        created_at: None,
    }])
    .unwrap();
    let c = count(&db3);

    println!("\nSUMMARY: record_text={a}  record={b}  record_batch={c}");
}

#[cfg(feature = "bundled-embedder")]
#[test]
#[ignore]
fn diag_does_think_backfill_the_graph() {
    let text = "Alice Chen met Bob Smith at Yantrik Systems in San Francisco";
    let db = YantrikDB::with_default(":memory:").unwrap();
    db.record_text(
        text,
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
    let n0: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM entities", [], |r| r.get(0))
        .unwrap();
    println!("after record_text:        entities={n0}");

    let cfg = ThinkConfig::default();
    let _ = db.think(&cfg);
    let n1: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM entities", [], |r| r.get(0))
        .unwrap();
    println!("after think(default):     entities={n1}");
}

#[cfg(feature = "bundled-embedder")]
#[test]
#[ignore]
fn diag_materializer_drains_single_write_entities() {
    let text = "Alice Chen met Bob Smith at Yantrik Systems in San Francisco";
    let db = YantrikDB::with_default(":memory:").unwrap();
    db.record_text(
        text,
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
    let n0: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM entities", [], |r| r.get(0))
        .unwrap();
    let pend: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM oplog WHERE applied = 0", [], |r| {
            r.get(0)
        })
        .unwrap();
    println!("after record_text: entities={n0}  pending_ops={pend}");
    let applied = db.apply_pending_ops_once(256);
    println!("apply_pending_ops_once -> {applied:?}");
    let n1: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM entities", [], |r| r.get(0))
        .unwrap();
    println!("after drain:       entities={n1}");
}

// ── 2026-08-13: cross-lane agreement (the ranking half of why_retrieved) ──

#[cfg(feature = "bundled-embedder")]
#[test]
fn lane_agreement_breaks_ties_between_near_equals() {
    // Two records nearly indistinguishable to the vector lane; the query
    // shares literal vocabulary with only one of them, so the lexical
    // lane surfaces that one too. Agreement must break the tie in its
    // favor — that is the entire job of the multiplier.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let rec = |t: &str| {
        db.record_text(
            t,
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
        .unwrap()
    };
    let both_lanes = rec("the deploy pipeline pushes to the staging cluster nightly");
    let _vector_only =
        rec("the release automation ships builds to the test environment each evening");
    let hits = db.recall_text("deploy pipeline staging", 2).unwrap();
    assert_eq!(
        hits[0].rid,
        both_lanes,
        "the record surfaced by BOTH vector and lexical lanes must outrank a \
         vector-only near-equal; got {:?}",
        hits.iter().map(|h| (&h.rid, h.score)).collect::<Vec<_>>()
    );
    assert!(
        hits[0]
            .why_retrieved
            .iter()
            .any(|w| w.contains("multi-lane agreement")),
        "the boost must be explainable in why_retrieved; got {:?}",
        hits[0].why_retrieved
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn lane_agreement_cannot_promote_an_irrelevant_record() {
    // The inversion budget, asserted: a record whose only virtue is
    // matching query KEYWORDS must not overtake a semantically on-topic
    // record with a similarity lead beyond 12.5%. This is the exact
    // failure the old additive graph form had (similarity 0.0026 beating
    // 0.309), rebuilt as a guard for the agreement multiplier.
    // Rewritten 2026-08-13: agreement is no longer an independent
    // multiplier with its own +12.5% ceiling — it is a SHARE of the one
    // policy budget, so the property to pin is that it stays inside that
    // budget and still saturates at two extra lanes.
    use crate::base::scoring::{agreement_mult, POLICY_BUDGET_LN, PW_AGREEMENT};
    assert!((agreement_mult(0) - 1.0).abs() < 1e-12);
    assert!(
        (agreement_mult(2) - (POLICY_BUDGET_LN * PW_AGREEMENT).exp()).abs() < 1e-12,
        "agreement must spend exactly its budget share, no more"
    );
    assert_eq!(
        agreement_mult(2),
        agreement_mult(9),
        "cap at two extra lanes"
    );
    assert!(
        agreement_mult(9) < POLICY_BUDGET_LN.exp(),
        "one prior alone must never consume the whole budget"
    );

    let db = YantrikDB::with_default(":memory:").unwrap();
    let rec = |t: &str| {
        db.record_text(
            t,
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
        .unwrap()
    };
    let on_topic = rec("the quarterly budget review moved to the first Monday of the month");
    // Shares the literal words "deploy" and "staging" with nothing —
    // build a keyword-bait record for a DIFFERENT query's vocabulary:
    let _bait = rec("deploy staging deploy staging unrelated grocery list apples");
    let hits = db
        .recall_text("when is the quarterly budget review", 2)
        .unwrap();
    assert_eq!(
        hits[0].rid,
        on_topic,
        "keyword bait with no semantic relevance must not overtake the on-topic \
         record, whatever lanes it matched; got {:?}",
        hits.iter().map(|h| (&h.rid, h.score)).collect::<Vec<_>>()
    );
}

// ── 2026-08-13: filter integrity across ALL retrieval lanes ──
//
// Found by external code review, then reproduced on a fresh store: a
// recall with domain="work" returned a domain="health" record, and one
// with certainty_min=0.8 returned a certainty=0.2 record. The vector lane
// applied the caller's filters; the FTS, claims and graph lanes
// re-admitted candidates checking only a subset. Filtering is a property
// of the REQUEST, not of the lane that happened to find the row.
//
// These use text that MATCHES LEXICALLY so the FTS lane genuinely fires —
// a test whose excluded record is semantically distant would pass even
// with the bug present.

#[cfg(feature = "bundled-embedder")]
#[test]
fn domain_filter_holds_across_every_lane() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    let rec = |text: &str, domain: &str| {
        db.record_text(
            text,
            "semantic",
            0.6,
            0.0,
            604800.0,
            &empty_meta(),
            "default",
            0.9,
            domain,
            "user",
            None,
        )
        .unwrap()
    };
    let _work = rec("the postgres migration runs on the staging cluster", "work");
    let health = rec(
        "the postgres migration runs on the staging cluster nightly",
        "health",
    );

    let hits = db
        .recall_text_filtered("postgres migration staging cluster", 20, Some("work"), None)
        .unwrap();
    assert!(
        !hits.iter().any(|h| h.rid == health),
        "a domain='work' recall returned a domain='health' record — the caller's \
         filter was bypassed by a secondary lane; got {:?}",
        hits.iter().map(|h| (&h.rid, &h.domain)).collect::<Vec<_>>()
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn certainty_floor_holds_across_every_lane() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    // importance 0.9 is ABOVE the high-importance fallback threshold (0.7 for
    // small databases), so this row is eligible for the fallback lane as well as
    // the vector/FTS lanes. The first version of this test used importance 0.6 and
    // could never reach that path — it passed while the bypass was still live.
    let low = db
        .record_text(
            "the postgres migration runs on the staging cluster",
            "semantic",
            0.9,
            0.0,
            604800.0,
            &empty_meta(),
            "default",
            0.2,
            "general",
            "user",
            None,
        )
        .unwrap();
    db.record_text(
        "unrelated grocery list apples bananas",
        "semantic",
        0.6,
        0.0,
        604800.0,
        &empty_meta(),
        "default",
        0.95,
        "general",
        "user",
        None,
    )
    .unwrap();

    let emb = db.embed("postgres migration staging cluster").unwrap();
    let hits = db
        .recall(
            &emb,
            20,
            None,
            None,
            false,
            false,
            Some("postgres migration staging cluster"),
            true,
            None,
            None,
            None,
            Some(0.8),
            None,
            false,
            None, // event_after (#149)
            None, // event_before (#149)
        )
        .unwrap();
    assert!(
        !hits.iter().any(|h| h.rid == low),
        "a certainty_min=0.8 recall returned a certainty=0.2 record; got {:?}",
        hits.iter()
            .map(|h| (&h.rid, h.certainty, &h.why_retrieved))
            .collect::<Vec<_>>()
    );
}

// ── 2026-08-14: lane slot quotas ──
//
// A quota is a COUNT, which is why it exists. Every score-space bound in the
// engine is a raw-cosine ratio, and on a real 5,035-record store the whole
// top 100 spanned 1.286x — so a "1.30x budget" could reorder 120-252 records.
// "At most 2 of 8 slots" means the same thing in any embedding space.

#[cfg(feature = "bundled-embedder")]
#[test]
fn quotas_are_unlimited_by_default() {
    // Enabling quotas must be an explicit act. If this ever fails, some
    // default changed and every existing deployment's ranking moved with it.
    let t = crate::base::tuning::Tuning::default();
    assert_eq!(t.quota_vector, 1.0);
    assert_eq!(t.quota_lexical, 1.0);
    assert_eq!(t.quota_claims, 1.0);
    assert_eq!(t.quota_graph, 1.0);
    assert_eq!(t.quota_exploration, 1.0);
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn a_quota_actually_removes_over_quota_candidates() {
    // The semantics that make a quota DO anything. The first version moved
    // over-quota candidates to the tail of the pool so that top_k could
    // always be filled — which was a no-op, because MMR selects from the
    // whole pool by its own criterion and could pick them anyway. A ceiling
    // only changes what is selected if the over-quota candidates are gone.
    use crate::engine::recall::apply_lane_quotas;
    let db = YantrikDB::with_default(":memory:").unwrap();
    for i in 0..30 {
        db.record_text(
            &format!("deploy pipeline staging cluster note number {i}"),
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
    let hits = db.recall_text("deploy pipeline staging", 20).unwrap();
    assert!(hits.len() >= 10, "sanity: need a pool, got {}", hits.len());

    // Defaults are unlimited: the pool must be untouched.
    let empty = std::collections::HashMap::new();
    let mut pool = hits.clone();
    apply_lane_quotas(&mut pool, &empty, 20);
    assert_eq!(
        pool.len(),
        hits.len(),
        "unlimited quotas must not change the pool"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn a_quota_preserves_relative_order_within_a_lane() {
    // A quota withholds slots; it must never PROMOTE anything. If it
    // reordered within a lane it would be a scoring change wearing a
    // quota's clothes, and would inherit exactly the calibration problem
    // quotas exist to avoid.
    use crate::engine::recall::apply_lane_quotas;
    let db = YantrikDB::with_default(":memory:").unwrap();
    for i in 0..20 {
        db.record_text(
            &format!("release checklist item {i} for the deployment runbook"),
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
    let hits = db.recall_text("release checklist deployment", 15).unwrap();
    let before: Vec<String> = hits.iter().map(|h| h.rid.clone()).collect();
    let mut pool = hits.clone();
    let empty = std::collections::HashMap::new();
    apply_lane_quotas(&mut pool, &empty, 15);
    let after: Vec<String> = pool.iter().map(|h| h.rid.clone()).collect();
    // With one lane owning everything and quotas at their defaults, order is
    // untouched; the assertion pins that the function is order-preserving.
    assert_eq!(before, after, "quotas must not reorder within a lane");
}
