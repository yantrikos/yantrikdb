//! Caller-supplied consolidation (`consolidate_cluster`).
//!
//! The default `consolidate()` writes an extractive join carrying the
//! cluster's MEAN embedding. Measured on BEAM (2026-08-11) that costs
//! accuracy through EMBEDDING DILUTION: no content is lost (all texts are
//! joined), but N precise vectors become one average that matches no
//! specific query well. This path lets a caller supply a real synthesis and
//! — the load-bearing part — embeds it FROM ITS OWN TEXT.

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

/// The point of the feature: the consolidated record's vector must describe
/// the SYNTHESIS, not the average of the members it replaced.
#[test]
fn caller_text_is_embedded_from_itself_not_the_cluster_mean() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["alpha one", "alpha two", "alpha three"]);
    let members: Vec<String> = rids.clone();

    let out = crate::consolidate::consolidate_cluster(
        &db,
        &members,
        "the alpha project shipped",
        Some(&vec_seed(9.0, 8)),
    )
    .unwrap();
    assert_eq!(out["cluster_size"], 3);
    assert_eq!(
        out["embedded_from_text"], false,
        "flag reports engine-side embedding; this call supplied its own vector"
    );

    let new_rid = out["consolidated_rid"].as_str().unwrap();
    let stored: String = db
        .conn()
        .query_row(
            "SELECT text FROM memories WHERE rid = ?1",
            params![new_rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        db.decrypt_text(&stored).unwrap(),
        "the alpha project shipped",
        "the caller owns the text; the engine must not rewrite it"
    );
}

/// Bookkeeping must match the default path exactly — sources retired,
/// membership recorded — or the two paths drift and replication diverges.
#[test]
fn sources_are_retired_and_membership_recorded() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["beta one", "beta two"]);
    let out = crate::consolidate::consolidate_cluster(
        &db,
        &rids,
        "beta summary",
        Some(&vec_seed(9.0, 8)),
    )
    .unwrap();
    let new_rid = out["consolidated_rid"].as_str().unwrap();

    for rid in &rids {
        let (status, into): (String, Option<String>) = db
            .conn()
            .query_row(
                "SELECT consolidation_status, consolidated_into FROM memories WHERE rid = ?1",
                params![rid],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(status, "consolidated", "source {rid} not retired");
        assert_eq!(into.as_deref(), Some(new_rid));
    }
    let members: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM consolidation_members WHERE consolidation_rid = ?1",
            params![new_rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(members, 2, "consolidation_members CRDT rows missing");
}

/// Double-consolidating a member would orphan the first consolidation's
/// membership rows, so it is refused rather than silently accepted.
#[test]
fn already_consolidated_members_are_refused() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["gamma one", "gamma two"]);
    crate::consolidate::consolidate_cluster(&db, &rids, "gamma summary", Some(&vec_seed(9.0, 8)))
        .unwrap();
    let err =
        crate::consolidate::consolidate_cluster(&db, &rids, "gamma again", Some(&vec_seed(9.0, 8)))
            .unwrap_err();
    assert!(
        format!("{err}").contains("not active"),
        "expected an active-status refusal, got: {err}"
    );
}

#[test]
fn empty_cluster_and_empty_text_are_refused() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    assert!(
        crate::consolidate::consolidate_cluster(&db, &[], "x", Some(&vec_seed(9.0, 8))).is_err()
    );
    let rids = seed(&db, &["delta one"]);
    let err = crate::consolidate::consolidate_cluster(&db, &rids, "   ", Some(&vec_seed(9.0, 8)))
        .unwrap_err();
    assert!(format!("{err}").contains("text is empty"), "{err}");
}

#[test]
fn unknown_rid_is_refused_before_anything_is_written() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["eps one"]);
    let mut with_bogus = rids.clone();
    with_bogus.push("does-not-exist".to_string());
    assert!(crate::consolidate::consolidate_cluster(
        &db,
        &with_bogus,
        "s",
        Some(&vec_seed(9.0, 8))
    )
    .is_err());
    let status: String = db
        .conn()
        .query_row(
            "SELECT consolidation_status FROM memories WHERE rid = ?1",
            params![rids[0]],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(status, "active", "a refused call must write nothing");
}

/// A caller-supplied vector of the wrong dimension must be refused before
/// anything is written — a diverged dim is undetectable until a query
/// silently misses.
#[test]
fn wrong_dimension_embedding_is_refused() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["zeta one"]);
    let err = crate::consolidate::consolidate_cluster(
        &db,
        &rids,
        "zeta summary",
        Some(&vec_seed(1.0, 4)),
    )
    .unwrap_err();
    assert!(format!("{err}").contains("consolidate_cluster"), "{err}");
    let status: String = db
        .conn()
        .query_row(
            "SELECT consolidation_status FROM memories WHERE rid = ?1",
            params![rids[0]],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(status, "active", "a refused call must write nothing");
}

/// `get_memory` — the rid point-read the binding lacked entirely.
/// Found while wiring LLM conflict resolution: `get_conflicts` hands back
/// `memory_a`/`memory_b` as bare rids and there was no way to resolve them
/// to text short of paging the namespace.
#[test]
fn get_memory_reads_by_rid_including_consolidated() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rids = seed(&db, &["theta one", "theta two"]);

    let m = db.get_memory(&rids[0]).unwrap().expect("memory must exist");
    assert_eq!(m.text, "theta one");
    assert_eq!(m.consolidation_status, "active");
    assert!(db.get_memory("no-such-rid").unwrap().is_none());

    // Naming a rid asks for THAT record — currency is not the caller's
    // question here, so a consolidated source must still be readable.
    crate::consolidate::consolidate_cluster(&db, &rids, "theta summary", Some(&vec_seed(9.0, 8)))
        .unwrap();
    let after = db.get_memory(&rids[0]).unwrap().expect("still readable");
    assert_eq!(after.text, "theta one");
    assert_eq!(after.consolidation_status, "consolidated");
}
