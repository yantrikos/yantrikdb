use super::*;

// ── Issue #47: correct() semantics tightened (v0.7.20) ───────────────

#[test]
fn correct_preserves_rid_and_created_at() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "original text",
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
    let original = db.get(&rid).unwrap().unwrap();
    let original_created_at = original.created_at;
    std::thread::sleep(std::time::Duration::from_millis(10));

    // v0.9.3: text corrections are refused (CorrectionRequiresReembed);
    // the rid/created_at preservation contract is exercised via an
    // importance correction instead.
    let result = db
        .correct(&rid, None, None, Some(0.9), None, "test correction")
        .unwrap();
    assert_eq!(result.corrected_rid, rid, "rid must be preserved");
    assert_eq!(result.original_rid, rid);
    assert!(!result.original_tombstoned);
    assert_eq!(result.revision_num, 1);

    let updated = db.get(&rid).unwrap().unwrap();
    assert_eq!(updated.text, "original text", "text untouched");
    assert!((updated.importance - 0.9).abs() < 1e-9);
    assert!(
        (updated.created_at - original_created_at).abs() < 1e-9,
        "created_at must be preserved (was {}, became {})",
        original_created_at,
        updated.created_at,
    );
}

#[test]
fn correct_text_change_without_embedder_returns_no_embedder_and_no_side_effects() {
    // v0.10 Item 3: a text-changing correction must re-embed to keep the
    // retrieval vector coherent. On a raw-embedding DB with no embedder
    // attached there is nothing to embed with, so it returns NoEmbedder
    // BEFORE touching any state — no revision row, no text mutation.
    // (Metadata/scalar corrections don't re-embed and are unaffected.)
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "alice owns service A",
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

    let err = db
        .correct(
            &rid,
            Some("bob owns service B"),
            None,
            None,
            None,
            "handover",
        )
        .unwrap_err();
    assert!(
        matches!(err, crate::error::YantrikDbError::NoEmbedder),
        "wrong error: {err}"
    );

    // Zero side effects.
    let unchanged = db.get(&rid).unwrap().unwrap();
    assert_eq!(unchanged.text, "alice owns service A");
    assert!(db.history(&rid).unwrap().is_empty(), "no revision row");

    // A no-op text change (same bytes) is NOT a re-embed — it takes the
    // metadata path and succeeds even without an embedder.
    db.correct(
        &rid,
        Some("alice owns service A"),
        None,
        Some(0.9),
        None,
        "touch",
    )
    .unwrap();
    assert_eq!(db.history(&rid).unwrap().len(), 1);

    // Metadata / importance / valence corrections remain available.
    db.correct(
        &rid,
        None,
        Some(&serde_json::json!({"owner": "bob"})),
        Some(0.9),
        Some(0.1),
        "ownership metadata updated",
    )
    .unwrap();
    assert_eq!(db.history(&rid).unwrap().len(), 2);
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn correct_reembed_updates_retrieval_vector_delta_and_cold() {
    // v0.10 Item 3 / trace T4: a text-changing correction re-embeds so the
    // record is retrieved under its NEW meaning, not its old one. rid,
    // created_at, and the revision chain are preserved; the revision row
    // records the prior embedding's provenance. Verified on BOTH a
    // delta-resident record and one compacted to the cold tier.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let record = |text: &str| {
        db.record_text(
            text,
            "semantic",
            0.6,
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

    // Similarity of `rid` to a free-text query, via the real recall path.
    let sim_to = |rid: &str, q: &str| -> f64 {
        let emb = db.embed(q).unwrap();
        db.recall(
            &emb, 10, None, None, false, false, None, true, None, None, None, None, None, false,
        )
        .unwrap()
        .into_iter()
        .find(|r| r.rid == rid)
        .map(|r| r.scores.similarity)
        .unwrap_or(0.0)
    };

    // A distractor on the NEW topic so ranking is contested, not trivial.
    let _distractor = record("annual financial revenue and profit projections");

    // ── Case A: delta-resident record ──
    let rid = record("I love hiking in the mountains at dawn");
    let created_before = db.get(&rid).unwrap().unwrap().created_at;
    let old_topic_before = sim_to(&rid, "hiking trails and mountain trekking");
    assert!(
        old_topic_before > 0.3,
        "record embeds as its old topic: {old_topic_before}"
    );

    db.correct(
        &rid,
        Some("the quarterly financial revenue grew twenty percent"),
        None,
        None,
        None,
        "topic corrected",
    )
    .unwrap();

    // rid + created_at preserved; text updated.
    let after = db.get(&rid).unwrap().unwrap();
    assert_eq!(after.created_at, created_before, "created_at preserved");
    assert!(after.text.contains("financial revenue"));

    // New-topic query retrieves it more strongly than the old topic now,
    // and its old-topic similarity fell — the vector followed the text.
    let new_topic_after = sim_to(&rid, "financial revenue report");
    let old_topic_after = sim_to(&rid, "hiking trails and mountain trekking");
    assert!(
        new_topic_after > old_topic_after,
        "re-embedded to new topic: new={new_topic_after} old={old_topic_after}"
    );
    assert!(
        old_topic_after < old_topic_before,
        "moved away from old topic: {old_topic_after} < {old_topic_before}"
    );

    // Revision row captured the prior embedding provenance.
    let revs = db.history(&rid).unwrap();
    assert_eq!(revs.len(), 1);
    // The prior VECTOR always existed, so its hash is captured. The model
    // name is only present when the original write stamped one (reembed
    // does; plain record_text leaves it NULL) — so we assert the hash,
    // which is the load-bearing "the vector changed, here's its fingerprint"
    // provenance a replica verifies against.
    let has_hash: bool = db
        .conn()
        .query_row(
            "SELECT prior_embedding_hash IS NOT NULL \
             FROM record_revisions WHERE rid = ?1",
            rusqlite::params![rid],
            |r| r.get(0),
        )
        .unwrap();
    assert!(has_hash, "prior embedding hash captured");

    // ── Case B: cold-tier record (correction tombstones a cold vector) ──
    let cold_rid = record("the weather today is sunny and warm");
    db.search_state.load().vec_index.compact().unwrap();
    assert!(sim_to(&cold_rid, "sunny warm weather forecast") > 0.3);

    db.correct(
        &cold_rid,
        Some("the stock market closed sharply higher today"),
        None,
        None,
        None,
        "cold topic corrected",
    )
    .unwrap();

    let cold_new = sim_to(&cold_rid, "stock market closing prices");
    let cold_old = sim_to(&cold_rid, "sunny warm weather forecast");
    assert!(
        cold_new > cold_old,
        "cold-tier re-embed works: new={cold_new} old={cold_old}"
    );
}

#[test]
fn correction_epoch_toggles_and_validates() {
    // v0.10 Item 3 seqlock (sol r4/r5): the guard bumps ODD on enter, EVEN
    // on drop; correction_epoch_even() returns a stable even snapshot;
    // correction_epoch_validate() (Acquire-fenced) passes only if unchanged
    // and even. The guard borrows conn, so drop order is compiler-enforced.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let e_start = db.correction_epoch_even().unwrap();
    assert_eq!(e_start % 2, 0, "boot epoch is even");
    assert!(
        db.correction_epoch_validate(e_start),
        "even + unchanged validates"
    );
    {
        let conn = db.conn();
        let _g = db.enter_correction_epoch(&conn);
        assert!(
            !db.correction_epoch_validate(e_start),
            "odd (correction in flight) fails the reader's validation"
        );
    }
    let e_end = db.correction_epoch_even().unwrap();
    assert_eq!(e_end, e_start + 2, "one correction advances the epoch by 2");
    assert!(
        db.correction_epoch_validate(e_end),
        "post-correction even validates"
    );
    assert!(
        !db.correction_epoch_validate(e_start),
        "a stale pre-correction snapshot never re-validates"
    );
}

#[test]
fn correct_with_embedding_rejects_stale_generation() {
    // sol r8: a caller-supplied vector is pinned to the generation it was
    // embedded against. If that generation no longer matches the index's
    // (a reembed cutover raced the correction), the vector is stale-space
    // and MUST be rejected retryably — never committed — else the corrected
    // text would rank under the old vector space. Here we simulate the race
    // by passing a generation that does not match the current one.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "alice owns service A",
            "semantic",
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
    let gen = db.search_generation();
    let new_emb = vec_seed(9.0, 8);

    // Stale generation → retryable rejection, NO mutation.
    let err = db
        .correct_with_embedding(
            &rid,
            Some("bob owns service B"),
            &new_emb,
            gen + 1,
            None,
            None,
            None,
            "handover",
        )
        .unwrap_err();
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::CorrectionDeferredDuringReembed { .. }
        ),
        "stale-generation caller embedding must be deferred, got {err:?}"
    );
    assert_eq!(
        db.get_untracked(&rid).unwrap().unwrap().text,
        "alice owns service A",
        "rejected correction must leave the text unchanged"
    );
    assert_eq!(
        db.history(&rid).unwrap().len(),
        0,
        "rejected correction must record no revision"
    );

    // Current generation → succeeds and updates the text in place.
    db.correct_with_embedding(
        &rid,
        Some("bob owns service B"),
        &new_emb,
        gen,
        None,
        None,
        None,
        "handover",
    )
    .unwrap();
    assert_eq!(
        db.get_untracked(&rid).unwrap().unwrap().text,
        "bob owns service B",
        "generation-matched caller embedding must apply in place at the same rid"
    );
    assert_eq!(
        db.history(&rid).unwrap().len(),
        1,
        "applied correction records exactly one revision"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn correct_new_text_drops_stale_entity_links() {
    // nuron finding (v0.10 Item-3 follow-up): correct(new_text) must drop
    // memory->entity links whose entity no longer appears (WORD-BOUNDARY) in
    // the corrected text — else graph expansion keeps serving the record under
    // its OLD association (why_retrieved=["graph-connected via <old-entity>"]).
    // Reproduces the Volkan/Aurelian case AND the substring-collision false-keep
    // guard: "Volkan" must drop even when the new text contains "Volkanic".
    let db = YantrikDB::with_default(":memory:").unwrap();
    let rid = db
        .record_text(
            "The Volkan salt marshes are cold",
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
    // Seed links deterministically (extraction is async + heuristic; we test the
    // DROP, not extraction): an entity present in the NEW text ("Aurelian") must
    // be kept; one absent must drop; and "Volkan" is a substring of the new
    // text's "Volkanic" — naive substring would false-KEEP it, word-boundary
    // must drop it.
    {
        let conn = db.conn();
        for e in ["Volkan", "Aurelian"] {
            conn.execute(
                "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name) VALUES (?1, ?2)",
                params![rid, e],
            )
            .unwrap();
        }
    }
    // Also seed the IN-MEMORY graph_index — this is what recall's
    // expand_entities actually reads. nuron's live verification showed a
    // durable-only drop leaves this index stale, so the corrected record keeps
    // being served under its old association. The durable-table assertion alone
    // gave a false green; this test now covers the live path.
    {
        let mut gi = db.graph_index.write();
        gi.link_memory(&rid, "Volkan");
        gi.link_memory(&rid, "Aurelian");
    }
    db.correct(
        &rid,
        Some("The Aurelian highland lakes near Volkanic rock"),
        None,
        None,
        None,
        "winter grounds moved",
    )
    .unwrap();
    let links: Vec<String> = {
        let conn = db.conn();
        let mut stmt = conn
            .prepare("SELECT entity_name FROM memory_entities WHERE memory_rid = ?1")
            .unwrap();
        stmt.query_map(params![rid], |r| r.get::<_, String>(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap()
    };
    assert!(
        links.contains(&"Aurelian".to_string()),
        "durable entity present in corrected text must be kept: {links:?}"
    );
    assert!(
        !links.contains(&"Volkan".to_string()),
        "durable stale entity absent from corrected text must be dropped (and NOT false-kept by 'Volkanic'): {links:?}"
    );
    // The in-memory graph_index (what recall's expand_entities reads) must ALSO
    // reflect the drop — the assertion that would have caught nuron's live
    // finding.
    let gi_entities: Vec<String> = db
        .graph_index
        .read()
        .entities_for_memory(&rid)
        .iter()
        .map(|s| s.to_lowercase())
        .collect();
    assert!(
        gi_entities.iter().any(|e| e == "aurelian"),
        "in-memory graph_index must keep the surviving entity: {gi_entities:?}"
    );
    assert!(
        !gi_entities.iter().any(|e| e == "volkan"),
        "in-memory graph_index must EVICT the stale entity (recall's expand_entities reads it): {gi_entities:?}"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn correct_correct_revision_chain_records_true_prior_state() {
    // sol finding 7: two successive text corrections must chain — the
    // second revision records the FIRST correction's state as its prior,
    // not the original (which would happen if both snapshotted the same
    // pre-loop `original`). Sequential here; the fix (re-read prior INSIDE
    // the serialized tx) is what makes the concurrent case correct too.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let rid = db
        .record_text(
            "the capital is Paris",
            "semantic",
            0.6,
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

    db.correct(
        &rid,
        Some("the capital is Lyon"),
        None,
        Some(0.7),
        None,
        "first fix",
    )
    .unwrap();
    db.correct(
        &rid,
        Some("the capital is Marseille"),
        None,
        Some(0.8),
        None,
        "second fix",
    )
    .unwrap();

    let hist = db.history(&rid).unwrap();
    assert_eq!(hist.len(), 2);
    // Revision 1's prior is the ORIGINAL.
    assert_eq!(hist[0].prior_text, "the capital is Paris");
    assert!((hist[0].prior_importance - 0.6).abs() < 1e-9);
    // Revision 2's prior is the FIRST correction's state — the chain fix.
    assert_eq!(hist[1].prior_text, "the capital is Lyon");
    assert!((hist[1].prior_importance - 0.7).abs() < 1e-9);
    // Both text corrections captured prior embedding provenance.
    assert!(hist[0].prior_embedding_hash.is_some());
    assert!(hist[1].prior_embedding_hash.is_some());
    // The two prior vectors differ (Paris vs Lyon embed differently).
    assert_ne!(hist[0].prior_embedding_hash, hist[1].prior_embedding_hash);

    // Current state is the last correction.
    assert_eq!(
        db.get(&rid).unwrap().unwrap().text,
        "the capital is Marseille"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn correct_then_forget_leaves_no_resurrected_vector() {
    // sol finding 3 (correct/forget shape): after a re-embedding correction
    // then a forget, the record must be GONE from recall — the correction's
    // superseding delta append must not resurface a tombstoned SQL row.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let rid = db
        .record_text(
            "hiking in the alps",
            "semantic",
            0.6,
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
    db.correct(
        &rid,
        Some("quarterly revenue growth"),
        None,
        None,
        None,
        "topic",
    )
    .unwrap();
    db.forget(&rid).unwrap();

    let q = db.embed("quarterly revenue report").unwrap();
    let hits = db
        .recall(
            &q, 10, None, None, false, false, None, true, None, None, None, None, None, false,
        )
        .unwrap();
    assert!(
        !hits.iter().any(|h| h.rid == rid),
        "forgotten record must not resurface via the correction's append"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn correction_clears_reembed_staging_columns() {
    // sol finding 4: a text correction must clear embedding_new so a
    // reembed mid-Encoding re-encodes this row (from the new text) instead
    // of promoting the stale staged vector at swap. We simulate staging
    // directly, then correct, and assert the staging is cleared.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let rid = db
        .record_text(
            "old topic text",
            "semantic",
            0.6,
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
    // Simulate a reembed having staged an embedding for the OLD text.
    db.conn()
        .execute(
            "UPDATE memories SET embedding_new = X'0011', embedding_new_model = 'stale-model' \
             WHERE rid = ?1",
            rusqlite::params![rid],
        )
        .unwrap();

    db.correct(
        &rid,
        Some("completely different new topic"),
        None,
        None,
        None,
        "fix",
    )
    .unwrap();

    let (en, enm): (Option<Vec<u8>>, Option<String>) = db
        .conn()
        .query_row(
            "SELECT embedding_new, embedding_new_model FROM memories WHERE rid = ?1",
            rusqlite::params![rid],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert!(en.is_none(), "embedding_new cleared by correction");
    assert!(enm.is_none(), "embedding_new_model cleared by correction");
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn record_batch_defers_during_reembed_instead_of_writing_a_doomed_generation() {
    // REGRESSION: record_batch never consulted the write_router. It loaded its
    // SearchState snapshot unguarded, so a reembed cutover could complete its
    // swap mid-batch and the batch would commit rows + appends stamped with an
    // `embedding_generation` that was being discarded. record() has always
    // guarded this (record.rs:105 before :138); batch silently did not, while
    // its own comment claimed every append lands on one anchored generation.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let inputs = vec![RecordInput {
        created_at: None,
        idempotency_key: None,
        text: "written during a cutover".to_string(),
        memory_type: "semantic".to_string(),
        importance: 0.5,
        valence: 0.0,
        half_life: 604800.0,
        metadata: empty_meta(),
        embedding: vec_seed(1.0, 8),
        namespace: "default".to_string(),
        certainty: 0.8,
        domain: "general".to_string(),
        source: "user".to_string(),
        emotional_state: None,
    }];

    // Simulate a reembed cutover in flight.
    db.write_router.switch_to_queueing();

    let err = db.record_batch(&inputs).unwrap_err();
    assert!(
        matches!(
            err,
            crate::error::YantrikDbError::BatchDeferredDuringReembed { count: 1 }
        ),
        "batch must defer rather than write against a doomed generation, got {err:?}"
    );
    // Retryable means exactly that: NO durable state was touched.
    let count: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0, "a deferred batch must write nothing");

    // Once the cutover finishes the same batch succeeds verbatim.
    db.write_router.switch_to_normal();
    let rids = db.record_batch(&inputs).unwrap();
    assert_eq!(rids.len(), 1);
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn replaying_materialize_record_post_does_not_inflate_mention_count() {
    // REGRESSION: apply_materialize_record_post bumped entities.mention_count
    // unconditionally, under a comment calling the statement "idempotent". The
    // INSERT does not FAIL on conflict, but `mention_count = mention_count + 1`
    // is not idempotent — and this op really is executed more than once:
    // mark_op_applied is a completion ACK, not a pre-work claim, so N workers
    // draining the same op all do the work and only then race on
    // `WHERE applied = 0`; and an op that errors stays pending and is retried.
    // Each duplicate inflated the count, which feeds the IDF term in patterns.rs
    // and silently skewed salience.
    //
    // Resetting applied=0 and draining again is exactly the retry/duplicate-worker
    // shape, and is the only way to reach the private handler from here.
    let db = YantrikDB::with_default(":memory:").unwrap();
    db.record_text(
        "Alice and Bob shipped the Acme project",
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

    db.apply_pending_ops_once(100).unwrap();
    let counts_after_first: Vec<(String, i64)> = {
        let conn = db.conn();
        let mut stmt = conn
            .prepare("SELECT name, mention_count FROM entities ORDER BY name")
            .unwrap();
        let rows = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        rows
    };
    assert!(
        !counts_after_first.is_empty(),
        "test needs at least one extracted entity to be meaningful"
    );

    // Simulate the duplicate execution: put the op back to pending and drain again.
    db.conn()
        .execute(
            "UPDATE oplog SET applied = 0 WHERE op_type = ?1",
            rusqlite::params![crate::engine::op_types::OP_MATERIALIZE_RECORD_POST],
        )
        .unwrap();
    db.apply_pending_ops_once(100).unwrap();

    let counts_after_replay: Vec<(String, i64)> = {
        let conn = db.conn();
        let mut stmt = conn
            .prepare("SELECT name, mention_count FROM entities ORDER BY name")
            .unwrap();
        let rows = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        rows
    };
    assert_eq!(
        counts_after_first, counts_after_replay,
        "replaying materialize_record_post must not re-count mentions"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn record_batch_replicates_to_a_peer() {
    // REGRESSION: record_batch logged ONE opaque op — log_op("record_batch",
    // {count, rids}) — and replication has no "record_batch" arm, so peers hit
    // the `_ =>` forward-compat catch-all and silently dropped it. The payload
    // carried no text/embedding/scalars, so it could not have rebuilt the rows
    // even with an arm. Every memory written via the public record_batch() API
    // therefore never reached any peer, with no error anywhere.
    use crate::replication::{apply_ops, extract_ops_since};
    let leader = YantrikDB::new(":memory:", 8).unwrap();
    let follower = YantrikDB::new(":memory:", 8).unwrap();

    let mk = |text: &str, seed: f32| RecordInput {
        created_at: None,
        idempotency_key: None,
        text: text.to_string(),
        memory_type: "semantic".to_string(),
        importance: 0.6,
        valence: 0.25,
        half_life: 604800.0,
        metadata: serde_json::json!({"tag": text}),
        embedding: vec_seed(seed, 8),
        namespace: "default".to_string(),
        certainty: 0.7,
        domain: "general".to_string(),
        source: "user".to_string(),
        emotional_state: None,
    };
    let inputs = vec![mk("first batch item", 1.0), mk("second batch item", 2.0)];
    let rids = leader.record_batch(&inputs).unwrap();
    assert_eq!(rids.len(), 2);

    let ops = extract_ops_since(&leader.conn(), None, None, None, 100).unwrap();
    // One canonical "record" op per item — not one opaque batch op.
    let record_ops: Vec<_> = ops.iter().filter(|o| o.op_type == "record").collect();
    assert_eq!(
        record_ops.len(),
        2,
        "expected one canonical record op per batch item, got {:?}",
        ops.iter().map(|o| &o.op_type).collect::<Vec<_>>()
    );
    assert!(
        !ops.iter().any(|o| o.op_type == "record_batch"),
        "the unreplicable record_batch op must be gone"
    );

    apply_ops(&follower, &ops).unwrap();

    // The follower actually has the memories, with content intact.
    for (rid, input) in rids.iter().zip(inputs.iter()) {
        let got = follower
            .get(rid)
            .unwrap()
            .unwrap_or_else(|| panic!("batch item {rid} did not replicate"));
        assert_eq!(got.text, input.text);
        assert_eq!(got.source, "user");
        assert_eq!(got.namespace, "default");
        assert_eq!(got.certainty, input.certainty);
        assert_eq!(got.valence, input.valence);
        assert_eq!(got.metadata["tag"], input.text.as_str());
    }
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn follower_replay_applies_corrected_embedding() {
    // sol finding 6: a re-embedding correction replicates its EXACT bytes,
    // so a same-DEK follower's stored vector + recall track the new meaning
    // (not a re-embed, not the stale old vector).
    use crate::replication::{apply_ops, extract_ops_since};
    let leader = YantrikDB::with_default(":memory:").unwrap();
    let follower = YantrikDB::with_default(":memory:").unwrap();

    let rid = leader
        .record_text(
            "sunny beach vacation",
            "semantic",
            0.6,
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
    // Replicate the record first.
    let ops = extract_ops_since(&leader.conn(), None, None, None, 100).unwrap();
    apply_ops(&follower, &ops).unwrap();

    // Correct on the leader (re-embed), then replicate the correct op.
    leader
        .correct(
            &rid,
            Some("corporate tax filing deadline"),
            None,
            None,
            None,
            "topic",
        )
        .unwrap();
    let correct_ops: Vec<_> = extract_ops_since(&leader.conn(), None, None, None, 100)
        .unwrap()
        .into_iter()
        .filter(|o| o.op_type == "correct")
        .collect();
    assert_eq!(correct_ops.len(), 1);
    assert!(
        correct_ops[0].embedding.is_some(),
        "correct op carries exact embedding bytes"
    );
    apply_ops(&follower, &correct_ops).unwrap();

    // Follower's SQL text updated.
    let ftext: String = follower
        .conn()
        .query_row(
            "SELECT text FROM memories WHERE rid = ?1",
            rusqlite::params![rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(ftext, "corporate tax filing deadline");

    // Follower recall reflects the NEW meaning (vector applied, not stale).
    let q = follower.embed("corporate tax deadline").unwrap();
    let hits = follower
        .recall(
            &q, 5, None, None, false, false, None, true, None, None, None, None, None, false,
        )
        .unwrap();
    assert!(
        hits.iter().any(|h| h.rid == rid),
        "follower retrieves the record under its corrected meaning"
    );
}

#[test]
fn correct_writes_revision_row_with_prior_state() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "v0",
            "episodic",
            0.5,
            -0.2,
            604800.0,
            &serde_json::json!({"key": "before"}),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    // v0.9.3: metadata/importance/valence correction (text refused).
    let _ = db
        .correct(
            &rid,
            None,
            Some(&serde_json::json!({"key": "after"})),
            Some(0.7),
            Some(0.3),
            "first correction",
        )
        .unwrap();
    let history = db.history(&rid).unwrap();
    assert_eq!(history.len(), 1, "one revision expected");
    let rev = &history[0];
    assert_eq!(rev.revision_num, 1);
    assert_eq!(rev.prior_text, "v0");
    assert!((rev.prior_importance - 0.5).abs() < 1e-9);
    assert!((rev.prior_valence + 0.2).abs() < 1e-9);
    assert_eq!(rev.reason, "first correction");
    assert_eq!(
        rev.prior_metadata.get("key").and_then(|v| v.as_str()),
        Some("before")
    );
}

#[test]
fn correct_multiple_revisions_increment_revision_num() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "v0",
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
    // v0.9.3: revision-number chain exercised via importance corrections
    // (text corrections are refused pending vector-coherent correct).
    let r1 = db
        .correct(&rid, None, None, Some(0.6), None, "first")
        .unwrap();
    let r2 = db
        .correct(&rid, None, None, Some(0.7), None, "second")
        .unwrap();
    let r3 = db
        .correct(&rid, None, None, Some(0.9), None, "third")
        .unwrap();
    assert_eq!(r1.revision_num, 1);
    assert_eq!(r2.revision_num, 2);
    assert_eq!(r3.revision_num, 3);

    let history = db.history(&rid).unwrap();
    assert_eq!(history.len(), 3, "three revisions expected");
    // history returns oldest-first; prior importances chain through.
    assert!((history[0].prior_importance - 0.5).abs() < 1e-9);
    assert!((history[1].prior_importance - 0.6).abs() < 1e-9);
    assert!((history[2].prior_importance - 0.7).abs() < 1e-9);

    let final_state = db.get(&rid).unwrap().unwrap();
    assert_eq!(final_state.text, "v0", "text untouched throughout");
    assert!((final_state.importance - 0.9).abs() < 1e-9);
}

#[test]
fn correct_rejects_empty_reason() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "text",
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

    // Empty string.
    let err = db
        .correct(&rid, Some("new"), None, None, None, "")
        .expect_err("empty reason must be rejected");
    match err {
        crate::error::YantrikDbError::InvalidInput(msg) => {
            assert!(msg.contains("reason"), "got: {msg}");
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }

    // Whitespace-only.
    let err2 = db
        .correct(&rid, Some("new"), None, None, None, "   \t\n  ")
        .expect_err("whitespace-only reason must be rejected");
    assert!(matches!(
        err2,
        crate::error::YantrikDbError::InvalidInput(_)
    ));
}

#[test]
fn correct_rejects_no_mutation_fields() {
    // All None mutation fields == no-op correction; must be rejected.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "text",
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
    let err = db
        .correct(&rid, None, None, None, None, "no fields supplied")
        .expect_err("no-op correction must be rejected");
    assert!(matches!(err, crate::error::YantrikDbError::InvalidInput(_)));
}

#[test]
fn correct_preserves_inbound_graph_edges() {
    // Inbound link integrity: a graph edge pointing TO the corrected
    // rid must continue to resolve after correct(). This is the central
    // win of the v0.7.20 semantics over the v0.7.19 "tombstone + new rid"
    // approach where inbound references dangled.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid_subject = db
        .record(
            "subject memory",
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
    // Create an entity-graph edge that mentions rid_subject as an
    // endpoint. Using the entity-relate path (rid acts as one of two
    // entity names — the test only cares that an edge persists).
    db.relate("anchor_entity", &rid_subject, "tags", 1.0)
        .unwrap();
    let edges_before = db.get_edges(&rid_subject).unwrap();
    assert!(!edges_before.is_empty(), "edge should exist before correct");

    // Correct the memory.
    // v0.9.3: importance correction (text refused); the link-integrity
    // property under test is about rid stability, not the text.
    db.correct(
        &rid_subject,
        None,
        None,
        Some(0.9),
        None,
        "test link integrity",
    )
    .unwrap();

    // Inbound edges must still resolve (rid_subject still exists).
    let edges_after = db.get_edges(&rid_subject).unwrap();
    assert_eq!(
        edges_before.len(),
        edges_after.len(),
        "inbound edges must be preserved across correct(); \
         this is the central v0.7.20 win over the v0.7.19 tombstone semantics"
    );
}

#[test]
fn correct_history_empty_for_never_corrected_record() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "never corrected",
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
    let history = db.history(&rid).unwrap();
    assert!(
        history.is_empty(),
        "never-corrected record has no revisions"
    );
}

#[test]
fn schema_v30_fresh_install_has_record_revisions_table() {
    // Fresh install must apply v30 schema and create the
    // record_revisions table + its index. Regression guard against
    // forgetting to add the table to SCHEMA_SQL alongside the migration.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let conn = db.conn();
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master \
             WHERE type = 'table' AND name = 'record_revisions'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        count, 1,
        "record_revisions table must exist on fresh install"
    );

    // Schema version meta should be at least 30.
    let schema_version: String = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let v: i32 = schema_version.parse().unwrap();
    assert!(
        v >= crate::base::schema::SCHEMA_VERSION,
        "schema_version must be at least {}, got {v}",
        crate::base::schema::SCHEMA_VERSION,
    );
}

#[test]
fn schema_v31_fresh_install_has_record_links_table() {
    // Issue #48: fresh install must create record_links + both covering
    // indexes. Regression guard against forgetting the table in SCHEMA_SQL
    // alongside MIGRATE_V30_TO_V31.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let conn = db.conn();
    let table: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master \
             WHERE type = 'table' AND name = 'record_links'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(table, 1, "record_links table must exist on fresh install");

    let idx: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' \
             AND name IN ('idx_record_links_source', 'idx_record_links_target')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(idx, 2, "both record_links covering indexes must exist");

    assert!(crate::base::schema::SCHEMA_VERSION >= 31);
}

/// Event time follows the corrected TEXT. Before 2026-08-15, correcting
/// "March 15, 2024" to "April 20, 2024" left event_dates/event_time_min/max
/// saying March — metadata contradicting its own text, on the one surface
/// that rewrites text. The three keys re-derive as ONE UNIT unless this
/// correction's own metadata_merge supplies any of them (caller ownership).
#[test]
fn correct_rederives_event_time_from_new_text() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    let rid = db
        .record_text(
            "the launch deadline is March 15, 2024",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({}),
            "default",
            0.8,
            "work",
            "user",
            None,
        )
        .unwrap();
    let before = db.get_memory(&rid).unwrap().unwrap();
    assert_eq!(
        before
            .metadata
            .get("event_dates")
            .and_then(|v| v.as_array())
            .map(|a| a.len()),
        Some(1),
        "precondition: write-path extraction produced March"
    );

    db.correct(
        &rid,
        Some("the launch deadline is April 20, 2024"),
        None,
        None,
        None,
        "deadline moved",
    )
    .unwrap();

    let after = db.get_memory(&rid).unwrap().unwrap();
    let dates: Vec<String> = after
        .metadata
        .get("event_dates")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|d| d.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(
        dates,
        vec!["2024-04-20".to_string()],
        "event keys must follow the corrected text, got {dates:?}"
    );
}

/// forget() must delete the DURABLE memory_entities rows, not just the
/// in-memory index — the rebuild loads durable rows with no tombstone
/// filter, so leftovers resurrected forgotten records' links on restart.
#[test]
fn forget_deletes_durable_entity_links() {
    let db = YantrikDB::with_default(":memory:").unwrap();
    // Batch path extracts entities INLINE (the single-record path is
    // async), so the durable rows exist deterministically before forget.
    let rids = db
        .record_batch(&[crate::types::RecordInput {
            created_at: None,
            idempotency_key: None,
            text: "Alice Chen is the CEO of Acme Corp".to_string(),
            memory_type: "semantic".to_string(),
            importance: 0.5,
            valence: 0.0,
            half_life: 604800.0,
            metadata: serde_json::json!({}),
            embedding: {
                let raw: Vec<f32> = (0..64).map(|i| (i as f32 + 1.0) * 0.1).collect();
                let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
                raw.iter().map(|x| x / norm).collect()
            },
            namespace: "default".to_string(),
            certainty: 0.8,
            domain: "people".to_string(),
            source: "user".to_string(),
            emotional_state: None,
        }])
        .unwrap();
    let rid = rids[0].clone();
    let conn = db.conn();
    let before: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memory_entities WHERE memory_rid = ?1",
            rusqlite::params![rid],
            |r| r.get(0),
        )
        .unwrap();
    drop(conn);
    assert!(before > 0, "precondition: entity links exist");

    db.forget(&rid).unwrap();

    let conn = db.conn();
    let after: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memory_entities WHERE memory_rid = ?1",
            rusqlite::params![rid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(after, 0, "durable entity links must die with the record");
}
