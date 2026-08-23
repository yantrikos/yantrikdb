use super::*;

#[test]
fn test_new_and_stats() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let s = db.stats(None).unwrap();
    assert_eq!(s.active_memories, 0);
    assert_eq!(s.edges, 0);
}

#[test]
fn test_actor_id_auto_generated() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    assert_eq!(db.actor_id().len(), 36); // UUIDv7
}

#[test]
fn test_actor_id_explicit() {
    let db = YantrikDB::new_with_actor(":memory:", 8, "device-A").unwrap();
    assert_eq!(db.actor_id(), "device-A");
}

#[test]
fn test_record_auto_extracts_entities() {
    // Regression: /v1/remember should populate memory_entities from heuristic
    // extraction so conflict detection can fire on raw-text inputs without
    // requiring the user to call /v1/relate first. Fixes issue #2.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "Alice Chen is the CEO of Acme Corp",
            "semantic",
            0.8,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "people",
            "user",
            None,
        )
        .unwrap();

    // Phase 4.3: entity persistence is enqueued by record() and applied by
    // the materializer thread. In tests we drain the queue inline before
    // asserting on entity-graph state.
    db.apply_pending_ops_once(100).unwrap();

    let entities: Vec<String> = {
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
        entities.contains(&"Alice Chen".to_string()),
        "got: {:?}",
        entities
    );
    assert!(
        entities.contains(&"Acme Corp".to_string()),
        "got: {:?}",
        entities
    );

    // Also verify the entities table was populated.
    let entity_count: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM entities", [], |r| r.get(0))
        .unwrap();
    assert!(
        entity_count >= 2,
        "expected >= 2 entities, got {}",
        entity_count
    );
}

#[test]
fn test_record_batch_extracts_event_time_like_record() {
    // Third instance of the batch-path-skips-what-record-does family (the
    // test above is the entity-linking instance). merge_event_dates was
    // wired into record() and record_text() as "the fix for the category";
    // the batch surface was left out, so a batch-ingested "deadline
    // March 15, 2024" got no event keys while the identical text through
    // record() did — silently.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let inputs = vec![RecordInput {
        created_at: None,
        idempotency_key: None,
        text: "the launch deadline is March 15, 2024".to_string(),
        memory_type: "episodic".to_string(),
        importance: 0.5,
        valence: 0.0,
        half_life: 604800.0,
        metadata: empty_meta(),
        embedding: vec_seed(1.0, 8),
        namespace: "default".to_string(),
        certainty: 0.8,
        domain: "work".to_string(),
        source: "user".to_string(),
        emotional_state: None,
    }];
    let rids = db.record_batch(&inputs).unwrap();
    let m = db.get_memory(&rids[0]).unwrap().unwrap();
    let dates = m
        .metadata
        .get("event_dates")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        dates,
        vec![serde_json::json!("2024-03-15")],
        "batch write must extract event time exactly as record() does"
    );
    assert!(m.metadata.get("event_time_min").is_some());
    assert!(m.metadata.get("event_time_max").is_some());
}

#[test]
fn test_record_batch_auto_extracts_entities() {
    // Same regression as above but for the batch path, which previously
    // skipped entity linking entirely.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let inputs = vec![
        RecordInput {
            created_at: None,
            idempotency_key: None,
            text: "Alice Chen is the CEO of Acme Corp".to_string(),
            memory_type: "semantic".to_string(),
            importance: 0.8,
            valence: 0.0,
            half_life: 604800.0,
            metadata: empty_meta(),
            embedding: vec_seed(1.0, 8),
            namespace: "default".to_string(),
            certainty: 0.8,
            domain: "people".to_string(),
            source: "user".to_string(),
            emotional_state: None,
        },
        RecordInput {
            created_at: None,
            idempotency_key: None,
            text: "Sarah Kim is the CTO of Acme Corp".to_string(),
            memory_type: "semantic".to_string(),
            importance: 0.8,
            valence: 0.0,
            half_life: 604800.0,
            metadata: empty_meta(),
            embedding: vec_seed(1.05, 8),
            namespace: "default".to_string(),
            certainty: 0.8,
            domain: "people".to_string(),
            source: "user".to_string(),
            emotional_state: None,
        },
    ];
    let rids = db.record_batch(&inputs).unwrap();
    assert_eq!(rids.len(), 2);

    let total_links: i64 = db
        .conn()
        .query_row("SELECT COUNT(*) FROM memory_entities", [], |r| r.get(0))
        .unwrap();
    assert!(
        total_links >= 3,
        "expected batch to link both memories to entities, got {} links",
        total_links
    );

    // The two memories refer to different people — verify extraction
    // distinguished them rather than lumping both into one entity.
    let load_entities = |rid: &str| -> Vec<String> {
        let conn = db.conn();
        let mut stmt = conn
            .prepare("SELECT entity_name FROM memory_entities WHERE memory_rid = ?1")
            .unwrap();
        stmt.query_map(params![rid], |r| r.get::<_, String>(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap()
    };
    let m1_entities = load_entities(&rids[0]);
    let m2_entities = load_entities(&rids[1]);
    assert!(m1_entities.contains(&"Alice Chen".to_string()));
    assert!(m2_entities.contains(&"Sarah Kim".to_string()));
    assert!(!m1_entities.contains(&"Sarah Kim".to_string()));
    assert!(!m2_entities.contains(&"Alice Chen".to_string()));
}

#[test]
fn test_record_and_get() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(1.0, 8);
    let rid = db
        .record(
            "hello world",
            "episodic",
            0.8,
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
    assert_eq!(rid.len(), 36);

    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.text, "hello world");
    assert_eq!(mem.memory_type, "episodic");
    assert_eq!(mem.importance, 0.8);
    assert_eq!(mem.consolidation_status, "active");
}

#[test]
fn test_record_updates_stats() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record(
        "one",
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
    db.record(
        "two",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(2.0, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();
    assert_eq!(db.stats(None).unwrap().active_memories, 2);
}

#[test]
fn test_recall_basic() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record(
        "the cat sat on the mat",
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
    db.record(
        "dogs are loyal friends",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(5.0, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();
    db.record(
        "cats love warm places",
        "episodic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(1.1, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();

    let results = db
        .recall(
            &vec_seed(1.0, 8),
            2,
            None,
            None,
            false,
            false,
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            false,
            None, // event_after (#149)
            None, // event_before (#149)
        )
        .unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn recall_survives_nan_embedding_in_candidate_pool() {
    // Issue #60 end-to-end repro. A stored embedding with a NaN component used
    // to yield a NaN similarity score — the old hnsw zero-norm guard missed NaN
    // (`NaN == 0.0` is false) — which, mixed with the finite scores of normal
    // candidates in the SAME recall sort, tripped Rust >= 1.81 driftsort's
    // total-order panic ("comparison function does not implement a total
    // order"). Now cosine_distance guards the NaN to a finite value AND every
    // recall sort uses f64::total_cmp, so recall returns cleanly. This guards
    // both fixes: a regressed hnsw guard trips the is_finite assertion below
    // even if driftsort happens not to panic on this input size.
    //
    // v0.9.3 update: the write-path contract gate now REJECTS NaN embeddings
    // at record() (see `contract_gate_rejects_invalid_writes`), so this test
    // simulates a LEGACY database — one whose NaN embedding was persisted
    // before the gate existed — by corrupting the stored blob under the gate
    // via SQL, then rebuilding the index the way open() would. Consumption-
    // side hardening must keep protecting those databases forever.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let add = |text: &str, emb: Vec<f32>| -> String {
        db.record(
            text,
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
        .unwrap()
    };
    add("finite one", vec_seed(1.0, 8));
    add("finite two", vec_seed(5.0, 8));
    add("finite three", vec_seed(1.1, 8));
    let legacy_rid = add("nan memory", vec_seed(2.0, 8));

    // The poison pill, injected BELOW the gate (legacy-data simulation):
    // corrupt the stored blob via SQL, then rebuild the in-memory index
    // from SQL rows exactly like open() does.
    let mut nan_emb = vec_seed(2.0, 8);
    nan_emb[0] = f32::NAN;
    {
        let conn = db.conn();
        conn.execute(
            "UPDATE memories SET embedding = ?1 WHERE rid = ?2",
            rusqlite::params![crate::serde_helpers::serialize_f32(&nan_emb), legacy_rid],
        )
        .unwrap();
    }
    db.rebuild_vec_index().unwrap();

    let results = db
        .recall(
            &vec_seed(1.0, 8),
            10,
            None,
            None,
            false,
            false,
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            false,
            None, // event_after (#149)
            None, // event_before (#149)
        )
        .expect("recall must not panic with a NaN embedding in the candidate pool");
    assert!(
        !results.is_empty(),
        "recall returns the finite-scored candidates"
    );
    assert!(
        results.iter().all(|r| r.score.is_finite()),
        "every returned score is finite — the NaN embedding was guarded, not \
         propagated into the sort comparator"
    );
}

#[test]
fn cold_tier_embeddings_decompress_on_read() {
    // Issue #62 defect A. archive() rewrites the embedding column with a
    // zstd-COMPRESSED blob; before the fix, every recall scoring path that
    // fetched a cold record's embedding reinterpreted those compressed
    // bytes as raw f32 — garbage similarities always, and (empirically
    // ~50-70% per record) a NaN that poisoned the response confidence.
    // The durable read path must hand back the ORIGINAL vector,
    // bit-identical (zstd is lossless), exactly as hydrate() would.
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let original = vec_seed(3.0, 8);
    let rid = db
        .record(
            "a fact destined for cold storage",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &original,
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    assert!(db.archive(&rid).unwrap(), "archived to cold");

    // Sanity: the stored blob really is compressed at rest.
    {
        let conn = db.conn();
        let stored: Vec<u8> = conn
            .query_row(
                "SELECT embedding FROM memories WHERE rid = ?1",
                rusqlite::params![rid],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            crate::compression::is_compressed(&stored),
            "archive() stores zstd bytes in the embedding column"
        );
    }

    // The sanctioned durable reader must decompress.
    let store = super::super::durable_embeddings::DurableEmbeddingStore::new(&db);
    let map = store.read_embeddings_for_rids(&[rid.as_str()]).unwrap();
    let entry = map.get(&rid).expect("cold rid present in read map");
    let decoded = crate::serde_helpers::deserialize_f32(&entry.bytes);
    assert_eq!(
        decoded, original,
        "cold read must return the original vector, not reinterpreted zstd bytes"
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn recall_scores_stay_finite_and_sane_with_cold_candidates() {
    // Issue #62 end-to-end (the reporter's minimal repro): a cold record
    // surfaced through the keyword lane must score from its REAL vector —
    // finite score, finite response-level confidence.
    let db = YantrikDB::with_default(":memory:").unwrap();
    let cold = db
        .record_text(
            "the zanzibar deployment uses the falcon cache",
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
    db.record_text(
        "an unrelated hot memory about lunch",
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

    assert!(db.archive(&cold).unwrap());

    // Keyword-lane query pulls the cold record into the candidate pool
    // (it is tombstoned in the ANN index, so FTS is its route in).
    let resp = db
        .recall_with_response(
            &db.embed("zanzibar falcon cache deployment").unwrap(),
            5,
            None,
            None,
            false,
            true,
            Some("zanzibar falcon cache deployment"),
            true,
            None,
            None,
            None,
        )
        .unwrap();
    assert!(
        resp.confidence.is_finite(),
        "response confidence must be finite, got {}",
        resp.confidence
    );
    for r in &resp.results {
        assert!(
            r.score.is_finite(),
            "every score finite; {} scored {}",
            r.rid,
            r.score
        );
    }
    let cold_hit = resp.results.iter().find(|r| r.rid == cold);
    let hit = cold_hit.expect("cold record surfaces via the keyword lane");
    assert!(
        hit.score.is_finite() && hit.score > 0.0,
        "cold record scores from its real (decompressed) vector, got {}",
        hit.score
    );
}

#[cfg(feature = "bundled-embedder")]
#[test]
fn exact_phrase_in_a_long_record_survives_frame_noise() {
    // The exact-phrase starvation defect (production report, repro at
    // tests/repros/exact_phrase_starvation.py) as a permanent gate: a
    // LONG record containing the query phrase verbatim, buried among
    // frame-shaped distractors that share the query's common terms.
    // Mean-pooled embeddings dilute the long record to near-orthogonal,
    // so bm25 fusion is its only route into top_k: rare-term match →
    // lexical strength ~1.0 → boost + reserve slot; the distractors
    // match only the common terms and pay the lexical discount.
    let db = YantrikDB::with_default(":memory:").unwrap();
    for i in 0..120 {
        db.record_text(
            &format!(
                "Agent run {i}: the pipeline reported a failure in subsystem {} \
                 during the nightly build; retries resolved it and the class of \
                 error was logged for later triage.",
                i % 9
            ),
            "episodic",
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
    let long_text = format!(
        "{}NAMED CLASS ADOPTED by core under my name: MISATTRIBUTING FAILURE \
         SURFACES — 4 instances found this cycle, each one a case where the \
         reported subsystem was not the causal subsystem. {}",
        "Session retrospective, planning notes and follow-ups. ".repeat(20),
        "Additional trailing context about unrelated matters follows. ".repeat(10),
    );
    let target = db
        .record_text(
            &long_text,
            "semantic",
            0.7,
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

    let query = "misattributing failure surfaces named class";
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
    let rank = resp.results.iter().position(|r| r.rid == target);
    assert!(
        rank.is_some(),
        "verbatim-phrase record must rank in top-10; got {:?}",
        resp.results
            .iter()
            .map(|r| (&r.rid, r.score, &r.why_retrieved))
            .collect::<Vec<_>>()
    );
    let hit = &resp.results[rank.unwrap()];
    assert!(
        hit.why_retrieved
            .iter()
            .any(|w| w == "keyword_match" || w == "keyword_reserved"),
        "the route in is the lexical lane, why={:?}",
        hit.why_retrieved
    );
}

#[test]
fn contract_gate_rejects_invalid_writes() {
    // v0.9.3 central numeric/vector contract gate (issue #60 follow-up, sol
    // converged plan item 1). Table-driven entry-path matrix: every invalid
    // input is rejected with the typed error BEFORE any side effect — the
    // engine is left byte-for-byte unchanged (memory count + index entries).
    use crate::error::YantrikDbError;

    let db = YantrikDB::new(":memory:", 8).unwrap();
    // One valid memory so the engine has real state to leave untouched.
    db.record(
        "baseline memory",
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
    let baseline = db.stats(None).unwrap();

    let nan_emb = {
        let mut v = vec_seed(2.0, 8);
        v[3] = f32::NAN;
        v
    };
    let wrong_dim = vec_seed(2.0, 5);

    // record: non-finite element (carries the element index) + wrong dim.
    match db
        .record(
            "x",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &nan_emb,
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap_err()
    {
        YantrikDbError::InvalidEmbedding { path, index, .. } => {
            assert_eq!(path, "record");
            assert_eq!(index, Some(3));
        }
        other => panic!("wrong error: {other}"),
    }
    assert!(matches!(
        db.record(
            "x",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &wrong_dim,
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap_err(),
        YantrikDbError::InvalidEmbedding { index: None, .. }
    ));

    // record: each non-finite scalar is rejected by field name.
    for (field, imp, val, cert, hl) in [
        ("importance", f64::NAN, 0.0, 0.8, 604800.0),
        ("valence", 0.5, f64::INFINITY, 0.8, 604800.0),
        ("certainty", 0.5, 0.0, f64::NAN, 604800.0),
        ("half_life", 0.5, 0.0, 0.8, f64::NEG_INFINITY),
    ] {
        match db
            .record(
                "x",
                "episodic",
                imp,
                val,
                hl,
                &empty_meta(),
                &vec_seed(2.0, 8),
                "default",
                cert,
                "general",
                "user",
                None,
            )
            .unwrap_err()
        {
            YantrikDbError::InvalidScalar { field: f, .. } => assert_eq!(f, field),
            other => panic!("wrong error for {field}: {other}"),
        }
    }

    // record_batch: a bad element LATE in the batch rejects the WHOLE batch
    // (no earlier element half-committed), and the error names the position.
    let batch = vec![
        crate::types::RecordInput {
            created_at: None,
            idempotency_key: None,
            text: "good".into(),
            memory_type: "episodic".into(),
            importance: 0.5,
            valence: 0.0,
            half_life: 604800.0,
            metadata: empty_meta(),
            embedding: vec_seed(3.0, 8),
            namespace: "default".into(),
            certainty: 0.8,
            domain: "general".into(),
            source: "user".into(),
            emotional_state: None,
        },
        crate::types::RecordInput {
            created_at: None,
            idempotency_key: None,
            text: "bad".into(),
            memory_type: "episodic".into(),
            importance: 0.5,
            valence: 0.0,
            half_life: 604800.0,
            metadata: empty_meta(),
            embedding: nan_emb.clone(),
            namespace: "default".into(),
            certainty: 0.8,
            domain: "general".into(),
            source: "user".into(),
            emotional_state: None,
        },
    ];
    match db.record_batch(&batch).unwrap_err() {
        YantrikDbError::InvalidEmbedding { path, reason, .. } => {
            assert_eq!(path, "record_batch");
            assert!(reason.contains("inputs[1]"), "{reason}");
        }
        other => panic!("wrong error: {other}"),
    }

    // insert_vector: typed error instead of the historical dim-assert panic.
    assert!(matches!(
        db.insert_vector("some-rid", &nan_emb).unwrap_err(),
        YantrikDbError::InvalidEmbedding {
            path: "insert_vector",
            ..
        }
    ));

    // recall: NaN / wrong-dim QUERY vectors are rejected, not searched.
    for bad_query in [nan_emb.clone(), wrong_dim.clone()] {
        assert!(matches!(
            db.recall(
                &bad_query, 5, None, None, false, false, None, true, None, None, None, None, None,
                false, None, // event_after (#149)
                None, // event_before (#149)
            )
            .unwrap_err(),
            YantrikDbError::InvalidEmbedding { path: "recall", .. }
        ));
    }

    // No side effects: rejected calls left the engine exactly as it was.
    let after = db.stats(None).unwrap();
    assert_eq!(after.active_memories, baseline.active_memories);
    assert_eq!(after.vec_index_entries, baseline.vec_index_entries);
    assert_eq!(
        db.recall(
            &vec_seed(1.0, 8),
            5,
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
            None, // event_after (#149)
            None, // event_before (#149)
        )
        .unwrap()
        .len(),
        1,
        "the one valid memory is still the only memory"
    );
}

#[test]
fn test_recall_empty() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let results = db
        .recall(
            &vec_seed(1.0, 8),
            5,
            None,
            None,
            false,
            false,
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            false,
            None, // event_after (#149)
            None, // event_before (#149)
        )
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_relate_and_get_edges() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let eid = db.relate("Alice", "Bob", "knows", 1.0).unwrap();
    assert_eq!(eid.len(), 36);

    let edges = db.get_edges("Alice").unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].src, "Alice");
    assert_eq!(edges[0].dst, "Bob");
}

#[test]
fn test_forget() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "forget me",
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
    assert!(db.forget(&rid).unwrap());
    let mem = db.get(&rid).unwrap().unwrap();
    assert_eq!(mem.consolidation_status, "tombstoned");
}

#[test]
fn test_forget_nonexistent() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    assert!(!db.forget("nonexistent").unwrap());
}

#[test]
fn test_decay_fresh() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record(
        "fresh",
        "episodic",
        0.9,
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
    let decayed = db.decay(0.01).unwrap();
    assert!(decayed.is_empty());
}

#[test]
fn test_oplog_has_hlc() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record(
        "test",
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

    let hlc_bytes: Vec<u8> = db
        .conn()
        .query_row(
            "SELECT hlc FROM oplog ORDER BY rowid DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(hlc_bytes.len(), 16);

    let ts = HLCTimestamp::from_bytes(&hlc_bytes).unwrap();
    assert!(ts.millis > 0);
}

#[test]
fn test_oplog_has_embedding_hash() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record(
        "test",
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

    // The record op should have an embedding_hash
    let hash: Vec<u8> = db
        .conn()
        .query_row(
            "SELECT embedding_hash FROM oplog WHERE op_type = 'record' LIMIT 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(hash.len(), 32); // BLAKE3 output is 32 bytes
}

#[test]
fn test_oplog_enriched_payload() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record(
        "test payload",
        "semantic",
        0.7,
        0.3,
        1000.0,
        &serde_json::json!({"key": "val"}),
        &vec_seed(1.0, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();

    let payload_str: String = db
        .conn()
        .query_row(
            "SELECT payload FROM oplog WHERE op_type = 'record' LIMIT 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let payload: serde_json::Value = serde_json::from_str(&payload_str).unwrap();

    assert_eq!(payload["type"], "semantic");
    assert_eq!(payload["text"], "test payload");
    assert_eq!(payload["importance"], 0.7);
    assert_eq!(payload["valence"], 0.3);
    assert_eq!(payload["half_life"], 1000.0);
    assert!(payload["rid"].is_string());
    assert!(payload["created_at"].is_number());
    assert!(payload["metadata"]["key"] == "val");
}

#[test]
fn test_schema_v3_has_conflicts_table() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let count: i64 = db
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='conflicts'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_resolve_keep_a() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid_a = db
        .record(
            "birthday March 5",
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
    let rid_b = db
        .record(
            "birthday March 15",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(2.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    let conflict = crate::conflict::create_conflict(
        &db,
        &crate::types::ConflictType::IdentityFact,
        &rid_a,
        &rid_b,
        Some("User"),
        Some("birthday"),
        "conflicting birthdays",
    )
    .unwrap();

    let result = db
        .resolve_conflict(
            &conflict.conflict_id,
            "keep_a",
            Some(&rid_a),
            None,
            Some("User confirmed March 5"),
        )
        .unwrap();
    assert!(result.loser_tombstoned);

    let mem_b = db.get(&rid_b).unwrap().unwrap();
    assert_eq!(mem_b.consolidation_status, "tombstoned");

    let resolved = db.get_conflict(&conflict.conflict_id).unwrap().unwrap();
    assert_eq!(resolved.status, "resolved");
    assert_eq!(resolved.strategy.as_deref(), Some("keep_a"));
}

#[test]
fn test_resolve_keep_both() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid_a = db
        .record(
            "a",
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
    let rid_b = db
        .record(
            "b",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(2.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    let conflict = crate::conflict::create_conflict(
        &db,
        &crate::types::ConflictType::Minor,
        &rid_a,
        &rid_b,
        None,
        None,
        "test",
    )
    .unwrap();
    let result = db
        .resolve_conflict(&conflict.conflict_id, "keep_both", None, None, None)
        .unwrap();
    assert!(!result.loser_tombstoned);

    let mem_a = db.get(&rid_a).unwrap().unwrap();
    let mem_b = db.get(&rid_b).unwrap().unwrap();
    assert_eq!(mem_a.consolidation_status, "active");
    assert_eq!(mem_b.consolidation_status, "active");
}

#[test]
fn test_correct_memory() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid = db
        .record(
            "favorite color is green",
            "episodic",
            0.7,
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

    // v0.9.3: importance correction (text corrections refused pending
    // vector-coherent correct in v0.10).
    let result = db
        .correct(
            &rid,
            None,
            None,                                  // metadata_merge
            Some(0.9),                             // new_importance
            None,                                  // new_valence
            "User corrected their favorite color", // reason (required, #47)
        )
        .unwrap();

    // Issue #47 (v0.7.20): correct() now mutates in place. rid is
    // preserved; original is not tombstoned; revision_num is 1.
    assert_eq!(result.corrected_rid, rid);
    assert_eq!(result.original_rid, rid);
    assert!(!result.original_tombstoned);
    assert_eq!(result.revision_num, 1);

    let updated = db.get(&rid).unwrap().unwrap();
    assert_ne!(updated.consolidation_status, "tombstoned");
    assert!((updated.importance - 0.9).abs() < 1e-9);
}

#[test]
fn test_get_conflicts_filtered() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid_a = db
        .record(
            "a",
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
    let rid_b = db
        .record(
            "b",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(2.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    let rid_c = db
        .record(
            "c",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(3.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    crate::conflict::create_conflict(
        &db,
        &crate::types::ConflictType::IdentityFact,
        &rid_a,
        &rid_b,
        Some("User"),
        Some("birthday"),
        "test 1",
    )
    .unwrap();
    crate::conflict::create_conflict(
        &db,
        &crate::types::ConflictType::Preference,
        &rid_b,
        &rid_c,
        Some("User"),
        Some("prefers"),
        "test 2",
    )
    .unwrap();

    let all = db.get_conflicts(None, None, None, None, None, 50).unwrap();
    assert_eq!(all.len(), 2);

    let identity_only = db
        .get_conflicts(None, Some("identity_fact"), None, None, None, 50)
        .unwrap();
    assert_eq!(identity_only.len(), 1);

    let critical = db
        .get_conflicts(None, None, None, Some("critical"), None, 50)
        .unwrap();
    assert_eq!(critical.len(), 1);
}

#[test]
fn test_dismiss_conflict() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let rid_a = db
        .record(
            "a",
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
    let rid_b = db
        .record(
            "b",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(2.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

    let conflict = crate::conflict::create_conflict(
        &db,
        &crate::types::ConflictType::Minor,
        &rid_a,
        &rid_b,
        None,
        None,
        "test",
    )
    .unwrap();

    db.dismiss_conflict(&conflict.conflict_id, Some("Not really a conflict"))
        .unwrap();

    let c = db.get_conflict(&conflict.conflict_id).unwrap().unwrap();
    assert_eq!(c.status, "dismissed");
}

#[test]
fn test_stats_include_conflicts() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let s = db.stats(None).unwrap();
    assert_eq!(s.open_conflicts, 0);
    assert_eq!(s.resolved_conflicts, 0);

    let rid_a = db
        .record(
            "a",
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
    let rid_b = db
        .record(
            "b",
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(2.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    crate::conflict::create_conflict(
        &db,
        &crate::types::ConflictType::Minor,
        &rid_a,
        &rid_b,
        None,
        None,
        "test",
    )
    .unwrap();

    let s = db.stats(None).unwrap();
    assert_eq!(s.open_conflicts, 1);
    assert_eq!(s.resolved_conflicts, 0);
}
