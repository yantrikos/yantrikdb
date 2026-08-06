use super::*;

// ════════════════════════════════════════════════════════════════════════════════
// Phase 2: Adaptive Learning (feedback, weights, learning)
// ════════════════════════════════════════════════════════════════════════════════

#[test]
fn test_recall_feedback_stores() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);
    let rid = db
        .record(
            "feedback target",
            "episodic",
            0.5,
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

    // Submit feedback
    db.recall_feedback(
        Some("test query"),
        Some(&emb),
        &rid,
        "relevant",
        Some(0.85),
        Some(1),
    )
    .unwrap();

    // Verify the row exists in recall_feedback table
    let count: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM recall_feedback WHERE rid = ?1 AND feedback = 'relevant'",
            params![rid],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 1, "Expected 1 feedback row, got {}", count);
}

#[test]
fn test_learned_weights_default() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let weights = db.load_learned_weights().unwrap();

    assert!(
        (weights.w_sim - 0.50).abs() < 1e-6,
        "w_sim default should be 0.50, got {}",
        weights.w_sim
    );
    assert!(
        (weights.w_decay - 0.20).abs() < 1e-6,
        "w_decay default should be 0.20, got {}",
        weights.w_decay
    );
    assert!(
        (weights.w_recency - 0.30).abs() < 1e-6,
        "w_recency default should be 0.30, got {}",
        weights.w_recency
    );
    assert!(
        (weights.gate_tau - 0.25).abs() < 1e-6,
        "gate_tau default should be 0.25, got {}",
        weights.gate_tau
    );
    assert!(
        (weights.alpha_imp - 0.80).abs() < 1e-6,
        "alpha_imp default should be 0.80, got {}",
        weights.alpha_imp
    );
    assert_eq!(weights.generation, 0, "generation should start at 0");
}

#[test]
fn test_feedback_count_increments() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);
    let rid = db
        .record(
            "counting feedback",
            "episodic",
            0.5,
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

    // Submit 5 feedback items
    for i in 0..5 {
        let feedback_type = if i % 2 == 0 { "relevant" } else { "irrelevant" };
        db.recall_feedback(
            Some("query"),
            Some(&emb),
            &rid,
            feedback_type,
            Some(0.5),
            Some(i + 1),
        )
        .unwrap();
    }

    let count = db.feedback_count().unwrap();
    assert_eq!(count, 5, "Expected feedback_count=5, got {}", count);
}

#[test]
fn test_learning_skipped_under_threshold() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);
    let rid = db
        .record(
            "learning test",
            "episodic",
            0.5,
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

    // Submit fewer than the episode gate's worth of evidence.
    for i in 0..10 {
        db.recall_feedback(Some("q"), Some(&emb), &rid, "relevant", Some(0.5), Some(i))
            .unwrap();
    }

    // v0.10 Item 2: the loop must ABSTAIN with a typed reason — not
    // learn — when preference evidence is insufficient.
    let report = db.run_learning().unwrap();
    assert_eq!(
        report.outcome, "insufficient_evidence",
        "loop abstains below the episode gate: {report:?}"
    );
    assert_eq!(report.generation, 0, "no generation minted");
    assert_eq!(report.engine_resurface_positive_count, 0);
}

#[test]
fn test_learning_runs_with_enough_feedback() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);
    let rid = db
        .record(
            "learning convergence",
            "episodic",
            0.7,
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

    // Submit 25 feedback items (above the MIN_FEEDBACK=20 threshold)
    for i in 0..25 {
        let feedback_type = if i % 3 == 0 { "irrelevant" } else { "relevant" };
        let score = 0.3 + (i as f64 * 0.02);
        db.recall_feedback(
            Some("learning query"),
            Some(&emb),
            &rid,
            feedback_type,
            Some(score),
            Some(i + 1),
        )
        .unwrap();
    }

    // run_learning should complete without error (may return true or false
    // depending on whether the optimizer found an improvement)
    let result = db.run_learning();
    assert!(
        result.is_ok(),
        "run_learning should not error with 25 feedback items: {:?}",
        result.err()
    );
}

#[test]
fn test_think_includes_learning() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);
    let rid = db
        .record(
            "think learning integration",
            "episodic",
            0.7,
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

    // Submit 25+ feedback items so learning has enough data
    for i in 0..26 {
        let feedback_type = if i % 4 == 0 { "irrelevant" } else { "relevant" };
        db.recall_feedback(
            Some("think query"),
            Some(&emb),
            &rid,
            feedback_type,
            Some(0.5),
            Some(i + 1),
        )
        .unwrap();
    }

    // think() internally calls run_learning() — it should not panic or error
    let config = ThinkConfig::default();
    let result = db.think(&config);
    assert!(
        result.is_ok(),
        "think() should not error when learning has enough feedback: {:?}",
        result.err()
    );
}

// ── Contradiction Classifier Tests ──

#[test]
fn test_conflict_entity_substitution_org() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb1 = vec_seed(1.0, 8);
    let emb2 = vec_seed(1.1, 8); // Very similar embedding

    // Create entities of the same type (organization)
    db.relate("User", "Google", "works_at", 1.0).unwrap();
    db.relate("User", "Meta", "works_at", 1.0).unwrap();

    // Record memories mentioning these entities
    db.record(
        "User works at Google as a senior engineer",
        "episodic",
        0.7,
        0.0,
        604800.0,
        &empty_meta(),
        &emb1,
        "default",
        0.8,
        "work",
        "user",
        None,
    )
    .unwrap();
    db.record(
        "User works at Meta as a senior engineer",
        "episodic",
        0.7,
        0.0,
        604800.0,
        &empty_meta(),
        &emb2,
        "default",
        0.8,
        "work",
        "user",
        None,
    )
    .unwrap();

    // Scan for conflicts — the entity substitution classifier should detect
    // that Google and Meta are both organizations, making this an identity_fact conflict
    let conflicts = crate::conflict::scan_conflicts(&db).unwrap();
    // Edge-based conflicts should be found (works_at is an identity rel type)
    assert!(!conflicts.is_empty(), "should detect works_at conflict");
    assert_eq!(conflicts[0].conflict_type, "identity_fact");
}

#[test]
fn test_conflict_entity_substitution_tech() {
    let db = YantrikDB::new(":memory:", 384).unwrap();

    // Create tech entities
    db.relate("API", "PostgreSQL", "uses", 1.0).unwrap();
    db.relate("API", "MySQL", "uses", 1.0).unwrap();

    // Record memories with similar embeddings but different tech choices
    let emb1 = vec_seed(2.0, 384);
    let emb2 = vec_seed(2.05, 384);
    db.record(
        "The API service uses PostgreSQL for the database layer",
        "semantic",
        0.8,
        0.0,
        604800.0,
        &empty_meta(),
        &emb1,
        "default",
        0.8,
        "architecture",
        "user",
        None,
    )
    .unwrap();
    db.record(
        "The API service uses MySQL for the database layer",
        "semantic",
        0.8,
        0.0,
        604800.0,
        &empty_meta(),
        &emb2,
        "default",
        0.8,
        "architecture",
        "user",
        None,
    )
    .unwrap();

    let conflicts = crate::conflict::scan_conflicts(&db).unwrap();
    // Should detect entity-based semantic conflict with tech substitution
    let _entity_based = conflicts
        .iter()
        .filter(|c| c.detection_reason.contains("contradict"))
        .collect::<Vec<_>>();
    // May or may not detect depending on similarity threshold — the scan
    // not panicking is the assertion.
}

/// sol #83 finding 2: reclassify's provisional-category branch minted a FRESH
/// `cat_id`, `INSERT OR IGNORE`d it, then attached members with that id — so on a
/// `UNIQUE(name)` collision the id named NO row and the members hit the
/// `substitution_members.category_id` FK (`IGNORE` resolves as `ABORT` for FK
/// errors), failing a legitimate reclassify.
///
/// The collision is reachable because `find_member_category` — which decides
/// "unknown token", routing us into this branch — matches only
/// `m.status = 'active'`, while `learn_category_members(source="llm_suggested")`
/// (public, via the Python binding) creates members as `'pending'`. So the
/// category name is taken while both its tokens still look unknown.
///
/// Fails on the unfixed code with a FOREIGN KEY constraint error.
#[test]
fn reclassify_reuses_existing_category_when_name_already_taken() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // A category whose name is exactly what strategy 3 will mint, with 'pending'
    // members so both tokens still read as unknown to find_member_category.
    //
    // Set up via raw SQL rather than learn_category_members ON PURPOSE: that API
    // deadlocks while creating a new category (see
    // learn_category_members_creates_new_category_without_deadlocking), and
    // routing this test through it would make it hang before it ever reached the
    // behavior under test. One test, one claim.
    // Invented tokens ON PURPOSE: the schema seeds real categories (redis, for
    // instance, ships as an active member of "databases"), and any seeded token
    // makes find_member_category return Some → known_a non-empty → strategy 2
    // handles it and strategy 3 never runs. An earlier draft of this test used
    // redis/memcached and passed against the UNFIXED code for exactly that
    // reason — it never reached the branch it was written to cover.
    let pre_id = "cat_pre_existing";
    {
        let conn = db.conn();
        conn.execute(
            "INSERT INTO substitution_categories \
             (id, name, conflict_mode, status, created_at, updated_at, hlc, origin_actor) \
             VALUES (?1, 'learned_zorblat_quibnix', 'exclusive', 'active', 1.0, 1.0, X'00', 'test')",
            rusqlite::params![pre_id],
        )
        .unwrap();
        for (i, tok) in ["zorblat", "quibnix"].iter().enumerate() {
            conn.execute(
                "INSERT INTO substitution_members \
                 (id, category_id, token_normalized, token_display, confidence, source, \
                  status, created_at, updated_at, hlc, origin_actor) \
                 VALUES (?1, ?2, ?3, ?3, 1.0, 'llm_suggested', 'pending', 1.0, 1.0, X'00', 'test')",
                rusqlite::params![format!("m{i}"), pre_id, tok],
            )
            .unwrap();
        }
    }

    // Two memories differing by exactly one meaningful token per side.
    let mk = |text: &str, emb: &[f32]| {
        db.record(
            text,
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            emb,
            "default",
            0.8,
            "work",
            "user",
            None,
        )
        .unwrap()
    };
    let a = mk("the config sets zorblat for caching", &vec_seed(1.0, 8));
    let b = mk("the config sets quibnix for caching", &vec_seed(1.1, 8));

    let mk_conflict = |id: &str| {
        db.conn()
            .execute(
                "INSERT INTO conflicts \
                 (conflict_id, conflict_type, priority, status, memory_a, memory_b, \
                  detected_at, detected_by, detection_reason, hlc, origin_actor) \
                 VALUES (?1, 'redundancy', 'medium', 'open', ?2, ?3, 2000.0, 'test', \
                         'same attribute, different value', X'00', 'test')",
                rusqlite::params![id, a, b],
            )
            .unwrap();
    };
    mk_conflict("cf1");
    mk_conflict("cf2");

    // First pass seeds the recurrence (strategy 3 requires a PRIOR
    // conflict_reclassify carrying this token pair).
    db.reclassify_conflict("cf1", "semantic").unwrap();

    // Second pass: recurrence >= 1, both tokens unknown → strategy 3 mints
    // "learned_zorblat_quibnix", which already exists. This is the call that
    // failed before the fix (FOREIGN KEY constraint).
    let res = db
        .reclassify_conflict("cf2", "semantic")
        .expect("reclassify must reuse the existing category, not die on its FK");

    // Guard against a vacuous pass: prove strategy 3 ACTUALLY RAN. Without this,
    // the assertions below hold trivially on any code path that never reaches
    // the branch under test — which is exactly how the first draft of this test
    // passed against the unfixed code.
    assert!(
        res.learned_members
            .iter()
            .any(|m| m.category_name == "learned_zorblat_quibnix"),
        "strategy 3 must have run and targeted the colliding name, got {:?}",
        res.learned_members
    );

    // The name still resolves to ONE category — the pre-existing one.
    let n: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM substitution_categories WHERE name = 'learned_zorblat_quibnix'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 1, "no duplicate category row");

    // No member may reference a category id that doesn't exist. This is the
    // invariant the fresh-id + OR IGNORE shape broke; asserting it directly
    // catches the bug whether or not the FK happens to be enforced.
    let orphans: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM substitution_members m \
             WHERE NOT EXISTS (SELECT 1 FROM substitution_categories c WHERE c.id = m.category_id)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        orphans, 0,
        "no members stranded under a non-existent category"
    );

    // The user_confirmed reclassify must PROMOTE the pre-existing pending/
    // llm_suggested rows, not silently skip them (sol #83 r2). Before this, the
    // UNIQUE(category_id, token_normalized) ignore left both members 'pending' —
    // and find_member_category reads only 'active', so the pair stayed invisible
    // to every later classification. The user's confirmation was thrown away with
    // no error: the same silent-loss class as the rest of this item.
    let promoted: Vec<(String, String, String)> = db
        .conn()
        .prepare(
            "SELECT token_normalized, source, status FROM substitution_members \
             WHERE category_id = ?1 ORDER BY token_normalized",
        )
        .unwrap()
        .query_map(rusqlite::params![pre_id], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .unwrap()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    for (tok, source, status) in &promoted {
        assert_eq!(
            (source.as_str(), status.as_str()),
            ("user_confirmed", "active"),
            "{tok} must be promoted to user_confirmed/active, got {source}/{status}"
        );
    }

    // The POINT of promoting: the learned pair is now visible to later
    // classification, which is what reclassify claimed to have achieved.
    let visible: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM substitution_members \
             WHERE token_normalized IN ('zorblat', 'quibnix') AND status = 'active'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        visible, 2,
        "both learned tokens are findable as active members"
    );

    // ...and both tokens still hang off the SURVIVING category.
    let members: Vec<String> = db
        .conn()
        .prepare(
            "SELECT token_normalized FROM substitution_members \
             WHERE category_id = ?1 ORDER BY token_normalized",
        )
        .unwrap()
        .query_map(rusqlite::params![pre_id], |r| r.get(0))
        .unwrap()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    assert!(
        members.contains(&"zorblat".to_string()) && members.contains(&"quibnix".to_string()),
        "members belong to the surviving category, got {members:?}"
    );
}

/// `learn_category_members` deadlocked outright on the branch that CREATES a
/// category — i.e. the half its own doc comment advertises ("Creates the category
/// if it doesn't exist"). Found while proving the #83 finding-2 regression test:
/// that test hung here during setup instead of reaching the bug it targeted.
///
/// Cause: the `MutexGuard` from `self.conn()` sat in a `match` SCRUTINEE. Rust
/// keeps scrutinee temporaries alive until the end of the whole `match`, so the
/// guard was still held inside the `QueryReturnedNoRows` arm when that arm called
/// `self.conn()` again — a re-lock of a non-reentrant `parking_lot::Mutex` on the
/// same thread. It wedges the entire DB, not just the caller: every other writer
/// then blocks on `conn` forever.
///
/// Reachable from the public Python binding (`learn_category_members(...,
/// source="llm_suggested")`). The fix (`ensure_substitution_category`) removes the
/// RE-LOCK: it acquires `conn` ONCE and runs both the INSERT and the read-back on
/// that single guard, so there is no second `self.conn()` to deadlock against.
///
/// Note it deliberately does NOT drop the guard between the two statements —
/// holding it across the pair is what makes "who won the name" a question with one
/// answer (see the helper's own doc). Splitting them into separate acquisitions
/// would fix the deadlock too, and quietly reopen that gap.
///
/// Runs on a worker with a deadline so the deadlock FAILS this test instead of
/// hanging the whole suite.
#[test]
fn learn_category_members_creates_new_category_without_deadlocking() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let r = db.learn_category_members(
            "brand_new_category",
            &[("alpha".to_string(), 1.0), ("beta".to_string(), 1.0)],
            "user_confirmed",
        );
        let n: i64 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM substitution_categories WHERE name = 'brand_new_category'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(-1);
        let _ = tx.send((r.map_err(|e| e.to_string()), n));
    });

    match rx.recv_timeout(std::time::Duration::from_secs(30)) {
        Ok((res, n)) => {
            let added = res.expect("learn_category_members must create the category");
            assert_eq!(added, 2, "both members ingested");
            assert_eq!(n, 1, "the new category row exists");
        }
        Err(_) => panic!(
            "learn_category_members deadlocked creating a new category: \
             self.conn() re-locked inside a match arm whose scrutinee still holds the guard"
        ),
    }
}

/// The other half of "promote, never demote" (sol #83 r2). Promoting is the
/// obvious direction; the silent loss runs both ways. A later `llm_suggested`
/// gossip must NOT knock a `user_confirmed` member back to `'pending'`, which
/// would make an already-learned substitution vanish from
/// `find_member_category` — the same invisibility, arrived at from the other
/// side. `seed` outranks everything and is never overwritten.
#[test]
fn learn_category_members_promotes_but_never_demotes() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // Weak evidence first: lands 'pending'.
    db.learn_category_members("promo_cat", &[("tok".to_string(), 0.4)], "llm_suggested")
        .unwrap();
    let row = |db: &YantrikDB| -> (String, String, f64) {
        db.conn()
            .query_row(
                "SELECT source, status, confidence FROM substitution_members \
                 WHERE token_normalized = 'tok'",
                [],
                |r| Ok((r.get(0).unwrap(), r.get(1).unwrap(), r.get(2).unwrap())),
            )
            .unwrap()
    };
    assert_eq!(
        (row(&db).0.as_str(), row(&db).1.as_str()),
        ("llm_suggested", "pending"),
        "llm_suggested starts pending"
    );

    // Stronger evidence: PROMOTES.
    db.learn_category_members("promo_cat", &[("tok".to_string(), 0.9)], "user_confirmed")
        .unwrap();
    let (source, status, conf) = row(&db);
    assert_eq!(
        (source.as_str(), status.as_str()),
        ("user_confirmed", "active"),
        "user_confirmed promotes the pending row"
    );
    assert!(
        (conf - 0.9).abs() < 1e-9,
        "confidence promoted too, got {conf}"
    );

    // Weak evidence again: must NOT demote.
    db.learn_category_members("promo_cat", &[("tok".to_string(), 0.1)], "llm_suggested")
        .unwrap();
    let (source, status, conf) = row(&db);
    assert_eq!(
        (source.as_str(), status.as_str()),
        ("user_confirmed", "active"),
        "a later llm_suggested must not demote a user_confirmed member"
    );
    assert!(
        (conf - 0.9).abs() < 1e-9,
        "confidence must not be clobbered by weaker evidence, got {conf}"
    );
}

/// Strategy 1 UPDATEs members directly instead of going through
/// `add_member_to_category`, so #83's rank guard did not cover it and it kept
/// rewriting `source` to 'user_confirmed' unconditionally — including
/// `source='seed'` rows (sol #83 r3).
///
/// That is not cosmetic. `reset_category_to_seed` deletes `source != 'seed'`, so
/// a reclassify of two SEEDED tokens made them deletable by a later reset; and
/// the gossip trigger keys on `source != 'seed'`, so an untouched seed category
/// started reading as user-expanded.
///
/// `postgresql` and `mysql` both ship seeded into "databases", so this is the
/// real, default-configuration path — no fixture needed.
#[test]
fn reclassify_reinforcement_does_not_rebrand_seed_members() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    let seed_sources = |db: &YantrikDB| -> Vec<(String, String)> {
        db.conn()
            .prepare(
                "SELECT token_normalized, source FROM substitution_members \
                 WHERE token_normalized IN ('postgresql', 'mysql') ORDER BY token_normalized",
            )
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap()
    };
    assert_eq!(
        seed_sources(&db),
        vec![
            ("mysql".to_string(), "seed".to_string()),
            ("postgresql".to_string(), "seed".to_string()),
        ],
        "precondition: both ship as seed members"
    );

    let mk = |text: &str, emb: &[f32]| {
        db.record(
            text,
            "semantic",
            0.7,
            0.0,
            604800.0,
            &empty_meta(),
            emb,
            "default",
            0.8,
            "work",
            "user",
            None,
        )
        .unwrap()
    };
    let a = mk("the service stores rows in postgresql", &vec_seed(1.0, 8));
    let b = mk("the service stores rows in mysql", &vec_seed(1.1, 8));
    db.conn()
        .execute(
            "INSERT INTO conflicts \
             (conflict_id, conflict_type, priority, status, memory_a, memory_b, \
              detected_at, detected_by, detection_reason, hlc, origin_actor) \
             VALUES ('cf_seed', 'redundancy', 'medium', 'open', ?1, ?2, 2000.0, 'test', \
                     'same attribute, different value', X'00', 'test')",
            rusqlite::params![a, b],
        )
        .unwrap();

    // Both tokens are known active members of the SAME category → strategy 1.
    db.reclassify_conflict("cf_seed", "semantic").unwrap();

    assert_eq!(
        seed_sources(&db),
        vec![
            ("mysql".to_string(), "seed".to_string()),
            ("postgresql".to_string(), "seed".to_string()),
        ],
        "seed provenance must survive a user_confirmed reinforcement"
    );

    // The consequence that made it data loss: a later reset must not delete them.
    let removed = db.reset_category_to_seed("databases").unwrap();
    let survivors: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM substitution_members \
             WHERE token_normalized IN ('postgresql', 'mysql')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        survivors, 2,
        "reset_category_to_seed deleted seed members that reclassify had rebranded \
         (removed {removed})"
    );
}

/// The Rust ladder (`member_source_rank`, which computes the INCOMING rank) and
/// the SQL ladder (`MEMBER_SOURCE_RANK_SQL`, which computes the STORED rank) are
/// two spellings of one policy that must agree — every rank guard compares one
/// against the other, so a silent disagreement re-opens exactly the demotion bug
/// this pins shut.
///
/// This evaluates the REAL constant, not a copy of it. (The first draft inlined
/// its own third copy of the CASE, which would have pinned the Rust fn against
/// the test's private duplicate and gone on passing while the two real
/// definitions drifted — the same mistake this whole PR is about, committed
/// inside the test written to prevent it.)
#[test]
fn member_source_rank_agrees_with_sql() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let conn = db.conn();
    conn.execute(
        "INSERT INTO substitution_categories \
         (id, name, conflict_mode, status, created_at, updated_at, hlc, origin_actor) \
         VALUES ('rc', 'rank_probe_cat', 'exclusive', 'active', 1.0, 1.0, X'00', 't')",
        [],
    )
    .unwrap();

    for (i, source) in ["seed", "user_confirmed", "llm_suggested", "something_else"]
        .iter()
        .enumerate()
    {
        let tok = format!("tok{i}");
        conn.execute(
            "INSERT INTO substitution_members \
             (id, category_id, token_normalized, token_display, confidence, source, \
              status, context_hint, created_at, updated_at, hlc, origin_actor) \
             VALUES (?1, 'rc', ?2, ?2, 1.0, ?3, 'active', NULL, 1.0, 1.0, X'00', 't')",
            rusqlite::params![format!("rp{i}"), tok, source],
        )
        .unwrap();

        let sql_rank: i64 = conn
            .query_row(
                &format!(
                    "SELECT {} FROM substitution_members WHERE token_normalized = ?1",
                    crate::engine::conflict::MEMBER_SOURCE_RANK_SQL
                ),
                rusqlite::params![tok],
                |r| r.get(0),
            )
            .unwrap();
        let rust_rank = YantrikDB::member_source_rank(source);
        assert_eq!(
            sql_rank, rust_rank as i64,
            "rank ladders disagree for source '{source}'"
        );
    }
}
