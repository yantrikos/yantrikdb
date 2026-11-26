//! Regression: `what_changed_since` must not report a superseded record as
//! Active.
//!
//! A resuming session asks "what moved while I was away". A record that was
//! created **and** superseded after `since` shows up in both lists of the
//! answer. Before the fix the entry in `new_records` carried
//! `current_status: Active` and `superseded_by: None` while
//! `status_transitions` reported the very same rid as `Active → Superseded`
//! with a successor — the packet contradicted itself, and a host that reads
//! `new_records` (the shorter view) learns the opposite of what happened.
//!
//! The field doc on `DigestEntry::current_status` already promises a
//! chain-derived status; these tests pin that promise.

use super::helpers::*;
use super::YantrikDB;
use crate::types::{LinkDirection, LinkType, RecordLink, RecordStatus};

fn record(db: &YantrikDB, text: &str) -> String {
    db.record(
        text,
        "semantic",
        0.9,
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
    .unwrap()
}

/// The defect: created after `since`, superseded after `since`.
#[test]
fn new_record_superseded_after_since_reports_its_chain_status() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let since = super::super::now();
    std::thread::sleep(std::time::Duration::from_millis(25));

    let old = record(&db, "The endpoint is v1");
    std::thread::sleep(std::time::Duration::from_millis(25));
    let new = record(&db, "The endpoint is v2");
    db.link(
        &new,
        &RecordLink {
            target_rid: old.clone(),
            link_type: LinkType::Supersedes,
        },
    )
    .unwrap();

    let changes = db.what_changed_since(since, None, 240).unwrap();

    let entry = changes
        .new_records
        .iter()
        .find(|e| e.rid == old)
        .expect("the superseded record was created after `since`");
    assert_eq!(
        entry.current_status,
        RecordStatus::Superseded,
        "a record with a selected inbound supersedes edge is not Active"
    );
    assert_eq!(
        entry.superseded_by.as_deref(),
        Some(new.as_str()),
        "the entry must name its successor, not leave the caller to guess"
    );

    // The two lists of one answer must agree about the same record.
    let transition = changes
        .status_transitions
        .iter()
        .find(|t| t.rid == old)
        .expect("the supersession was committed after `since`");
    assert_eq!(transition.to, entry.current_status);
    assert_eq!(transition.by_rid, entry.superseded_by);
}

/// The legitimate case must keep passing: an undisturbed record is Active and
/// names no successor.
#[test]
fn new_record_without_successor_stays_active() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let since = super::super::now();
    std::thread::sleep(std::time::Duration::from_millis(25));

    let live = record(&db, "The endpoint is v1");
    let changes = db.what_changed_since(since, None, 240).unwrap();

    let entry = changes
        .new_records
        .iter()
        .find(|e| e.rid == live)
        .expect("the record was created after `since`");
    assert_eq!(entry.current_status, RecordStatus::Active);
    assert_eq!(entry.superseded_by, None);
    assert!(
        changes.status_transitions.is_empty(),
        "no link was written, so there is no transition to report"
    );
}

/// The successor itself is not superseded by its own outbound edge. Guards the
/// obvious wrong fix: mapping both endpoints of every supersedes edge.
#[test]
fn successor_is_not_marked_superseded_by_its_own_edge() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let since = super::super::now();
    std::thread::sleep(std::time::Duration::from_millis(25));

    let old = record(&db, "The endpoint is v1");
    std::thread::sleep(std::time::Duration::from_millis(25));
    let new = record(&db, "The endpoint is v2");
    db.link(
        &new,
        &RecordLink {
            target_rid: old.clone(),
            link_type: LinkType::Supersedes,
        },
    )
    .unwrap();

    let changes = db.what_changed_since(since, None, 240).unwrap();
    let successor = changes
        .new_records
        .iter()
        .find(|e| e.rid == new)
        .expect("the successor was created after `since`");
    assert_eq!(successor.current_status, RecordStatus::Active);
    assert_eq!(successor.superseded_by, None);

    // And a non-supersedes link never changes a status.
    let note = record(&db, "A note that only relates");
    db.link(
        &note,
        &RecordLink {
            target_rid: old.clone(),
            link_type: LinkType::DerivedFrom,
        },
    )
    .unwrap();
    let after = db.what_changed_since(since, None, 240).unwrap();
    let old_entry = after.new_records.iter().find(|e| e.rid == old).unwrap();
    assert_eq!(
        old_entry.current_status,
        RecordStatus::Superseded,
        "an extra RelatedTo edge must not undo the supersession"
    );
}
// ── Closing checks, 26.11.: multiple successors, determinism, imports ──

/// The record layer keeps a single selected inbound successor per target
/// (`gate_supersedes`). A second one is refused or stored as a losing
/// candidate — either way the *read* path must never see two active successors.
#[test]
fn a_second_successor_never_becomes_a_second_active_edge() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let since = super::super::now();
    std::thread::sleep(std::time::Duration::from_millis(25));

    let old = record(&db, "The endpoint is v1");
    let n1 = record(&db, "The endpoint is v2");
    let n2 = record(&db, "The endpoint is v2 (second attempt)");
    db.link(
        &n1,
        &RecordLink {
            target_rid: old.clone(),
            link_type: LinkType::Supersedes,
        },
    )
    .unwrap();
    // The second attempt may be refused outright; that is the point.
    let _ = db.link(
        &n2,
        &RecordLink {
            target_rid: old.clone(),
            link_type: LinkType::Supersedes,
        },
    );

    let inbound = db
        .linked_records(&old, LinkDirection::Inbound, Some(&LinkType::Supersedes))
        .unwrap();
    assert_eq!(
        inbound.len(),
        1,
        "one selected successor, not two: {inbound:?}"
    );
}

/// Legacy data (pre Phase-0) can violate that invariant. The change feed must
/// still answer deterministically: the newest committed edge wins, and two
/// calls return the same successor.
#[test]
fn legacy_multiple_successors_resolve_deterministically() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let since = super::super::now();
    std::thread::sleep(std::time::Duration::from_millis(25));

    let old = record(&db, "The endpoint is v1");
    let n1 = record(&db, "The endpoint is v2");
    std::thread::sleep(std::time::Duration::from_millis(25));
    let n2 = record(&db, "The endpoint is v2 (later)");
    db.link(
        &n1,
        &RecordLink {
            target_rid: old.clone(),
            link_type: LinkType::Supersedes,
        },
    )
    .unwrap();
    let _ = db.link(
        &n2,
        &RecordLink {
            target_rid: old.clone(),
            link_type: LinkType::Supersedes,
        },
    );

    // The gate refuses the second successor, so there is no row to re-select.
    // Recreate the legacy shape explicitly: a second selected inbound edge,
    // committed later than the first.
    {
        let conn = db.conn();
        conn.execute(
            "INSERT INTO record_links
                 (link_id, source_rid, target_rid, link_type, status, selection_state,
                  created_at, hlc, origin_actor)
             VALUES (?1, ?2, ?3, 'supersedes', 'active', 'selected', ?4, x'00', 'legacy')",
            rusqlite::params!["legacy-edge-0002", n2, old, super::super::now() + 1.0],
        )
        .unwrap();
    }

    let first = db.what_changed_since(since, None, 240).unwrap();
    let second = db.what_changed_since(since, None, 240).unwrap();
    let pick = |c: &crate::ChangesSince| {
        c.new_records
            .iter()
            .find(|e| e.rid == old)
            .and_then(|e| e.superseded_by.clone())
    };
    assert!(
        pick(&first).is_some(),
        "legacy row still reports a successor"
    );
    assert_eq!(pick(&first), pick(&second), "two calls, same answer");
    assert_eq!(
        pick(&first).as_deref(),
        Some(n2.as_str()),
        "the newest committed edge wins — documented, not incidental"
    );
}

/// Imported records carry a caller-supplied `created_at` (event time). They must
/// not appear as "new since T", but a supersession committed now is a real
/// change and must be reported — for a record the caller never saw arrive.
#[test]
fn imported_records_keep_event_time_and_still_report_later_supersession() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let now = super::super::now();
    let since = now - 3600.0;

    let imported = db
        .record_with_idempotency(
            "Imported measurement from last month",
            "semantic",
            0.8,
            0.0,
            604800.0,
            &serde_json::json!({}),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "work",
            "import",
            None,
            None,
            Some(now - 86400.0 * 30.0),
        )
        .unwrap();
    let successor = record(&db, "Corrected measurement, imported today");

    let before = db.what_changed_since(since, None, 240).unwrap();
    assert!(
        !before.new_records.iter().any(|e| e.rid == imported),
        "a backdated import was not created in the last hour"
    );

    db.link(
        &successor,
        &RecordLink {
            target_rid: imported.clone(),
            link_type: LinkType::Supersedes,
        },
    )
    .unwrap();
    let after = db.what_changed_since(since, None, 240).unwrap();
    let tr = after
        .status_transitions
        .iter()
        .find(|t| t.rid == imported)
        .expect("the supersession happened now, whatever the record's event time");
    assert_eq!(tr.by_rid.as_deref(), Some(successor.as_str()));

    // The imported record is not in new_records, yet the transition list marks
    // it superseded. The asymmetry is the documented behaviour: the two lists
    // answer different questions (what arrived / what moved).
    assert!(!after.new_records.iter().any(|e| e.rid == imported));
}
