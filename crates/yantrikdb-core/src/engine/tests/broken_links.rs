//! `forget` keeps the supersedes edge as `broken_target_forgotten` instead of
//! deleting it, so the reason a record disappeared survives. Every read path
//! filtered `status = 'active'`, which made that provenance write-only: present
//! in `record_links`, unreachable through any API. `broken_links()` is the read
//! side, deliberately separate from `linked_records()` so the active view — what
//! ranking and expansion consume — does not change.

use super::*;

#[test]
fn broken_link_is_readable_without_touching_the_active_view() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let a = db
        .record("value is nine", "semantic", 0.8, 0.0, 604800.0, &empty_meta(),
                &vec_seed(1.0, 8), "default", 0.8, "general", "user", None).unwrap();
    let b = db
        .record("value is ten", "semantic", 0.6, 0.0, 604800.0, &empty_meta(),
                &vec_seed(2.0, 8), "default", 0.8, "general", "user", None).unwrap();
    db.link(&a, &crate::types::RecordLink {
        target_rid: b.clone(), link_type: crate::types::LinkType::Supersedes }).unwrap();
    db.forget(&b).unwrap();

    let active = db.linked_records(&a, crate::types::LinkDirection::Outbound, None).unwrap();
    assert_eq!(active.len(), 0, "the active view must keep hiding a broken edge");

    let broken = db.broken_links(&a, crate::types::LinkDirection::Outbound).unwrap();
    assert_eq!(broken.len(), 1, "the provenance edge must be reachable");
    assert_eq!(broken[0].rid, b);
    assert_eq!(broken[0].link_type, "supersedes");
}
