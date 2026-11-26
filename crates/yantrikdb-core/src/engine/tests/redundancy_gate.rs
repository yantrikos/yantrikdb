use super::*;

/// Seed one exclusive substitution category with two members, so the pairwise
/// redundancy scan has something it would persist a conflict for.
fn seed_category(db: &YantrikDB) {
    let conn = db.conn();
    let hlc = vec![0u8; 16];
    conn.execute(
        "INSERT OR IGNORE INTO substitution_categories \
         (id, name, conflict_mode, status, created_at, updated_at, hlc, origin_actor) \
         VALUES ('cg', 'gate_test_slot', 'exclusive', 'active', 0.0, 0.0, ?1, 'test')",
        rusqlite::params![hlc],
    )
    .unwrap();
    for (i, m) in ["openai", "google"].iter().enumerate() {
        conn.execute(
            "INSERT OR IGNORE INTO substitution_members \
             (id, category_id, token_normalized, token_display, confidence, source, status, \
              created_at, updated_at, hlc, origin_actor) \
             VALUES (?1, 'cg', ?2, ?2, 1.0, 'test', 'active', 0.0, 0.0, ?3, 'test')",
            rusqlite::params![format!("cg{i}"), m, vec![0u8; 16]],
        )
        .unwrap();
    }
    drop(conn);
}

/// Two records with the SAME embedding and a substitution-category pair of
/// tokens: exactly what `check_redundancy` turns into a persisted conflict.
fn seed_pair(db: &YantrikDB) {
    let emb = vec_seed(0.42, 8);
    for text in [
        "The provider for the summary model is openai and it runs locally.",
        "The provider for the summary model is google and it runs locally.",
    ] {
        db.record(
            text,
            "episodic",
            0.9,
            0.0,
            604800.0,
            &empty_meta(),
            &emb,
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    }
}

fn conflict_count(db: &YantrikDB) -> i64 {
    db.conn()
        .query_row("SELECT COUNT(*) FROM conflicts", [], |r| r.get(0))
        .unwrap()
}

/// The gate: with `run_conflict_scan: false` the cycle must neither run the
/// pairwise scan nor persist what it would have found.
#[test]
fn conflict_scan_off_persists_no_redundancy_conflict() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    seed_category(&db);
    seed_pair(&db);

    let before = conflict_count(&db);
    let report = db
        .think(&ThinkConfig {
            run_consolidation: false,
            run_conflict_scan: false,
            run_pattern_mining: false,
            run_personality: false,
            ..Default::default()
        })
        .unwrap();

    assert_eq!(
        conflict_count(&db),
        before,
        "conflicts were persisted with the scan off"
    );
    assert_eq!(report.conflicts_found, 0);
}

/// Control: with the scan on, the same input does produce the conflict. Without
/// this the test above would also pass on a store where the detector never fires.
#[test]
fn conflict_scan_on_persists_the_same_conflict() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    seed_category(&db);
    seed_pair(&db);

    let before = conflict_count(&db);
    db.think(&ThinkConfig {
        run_consolidation: false,
        run_conflict_scan: true,
        run_pattern_mining: false,
        run_personality: false,
        ..Default::default()
    })
    .unwrap();

    assert!(
        conflict_count(&db) > before,
        "scan on must persist the substitution conflict this pair describes"
    );
}
