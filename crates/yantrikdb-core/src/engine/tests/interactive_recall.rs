use super::*;

// ════════════════════════════════════════════════════════════════════════════════
// Phase 1: Interactive Recall (confidence, hints, recall_refine)
// ════════════════════════════════════════════════════════════════════════════════

#[test]
fn test_recall_with_response_structure() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    db.record(
        "memory for recall response",
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

    let response = db
        .recall_with_response(
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
        )
        .unwrap();

    // RecallResponse must have all four fields
    assert!(!response.results.is_empty(), "results should not be empty");
    assert!(
        response.confidence >= 0.0 && response.confidence <= 1.0,
        "confidence should be in [0, 1], got {}",
        response.confidence
    );
    // retrieval_summary should have sources_used and candidate_count
    assert!(
        !response.retrieval_summary.sources_used.is_empty(),
        "sources_used should not be empty"
    );
    assert!(
        response.retrieval_summary.candidate_count > 0,
        "candidate_count should be > 0"
    );
    // hints is a Vec, may be empty or not
    let _ = response.hints; // just ensure the field exists and is accessible
}

#[test]
fn recall_with_response_reports_typed_search_coverage() {
    // v0.10 Item 1b / trace T08 "absence-with-coverage": a consumer must
    // be able to distinguish, from TYPED fields, (a) "this scope has no
    // records" from (b) "candidates exist but none clears the relevance
    // gate" from (c) a real match — nuron's false-retry loop came from
    // reading an empty result as a transient failure.
    use crate::types::CoverageOutcome;
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // One record along basis e0, in the default namespace.
    let mut e0 = vec![0.0f32; 8];
    e0[0] = 1.0;
    let mut e1 = vec![0.0f32; 8];
    e1[1] = 1.0;
    db.record(
        "the only fact in the store",
        "semantic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &e0,
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap();

    let respond = |query: &[f32], ns: Option<&str>| {
        db.recall_with_response(
            query, 5, None, None, false, false, None, true, ns, None, None,
        )
        .unwrap()
    };

    // (a) Scope with zero records: typed NoMatchingRecord, scope echoed.
    let resp = respond(&e0, Some("empty_ns"));
    assert_eq!(resp.coverage.outcome, CoverageOutcome::NoMatchingRecord);
    assert_eq!(resp.coverage.candidate_count, 0);
    assert_eq!(resp.coverage.namespace.as_deref(), Some("empty_ns"));

    // (b) Candidates exist but the best hit is orthogonal to the query —
    // similarity 0 sits under the gate (default gate_tau = 0.25).
    let resp = respond(&e1, None);
    assert_eq!(resp.coverage.outcome, CoverageOutcome::BelowThreshold);
    assert!(resp.coverage.candidate_count > 0);
    assert!(resp.coverage.top_similarity < resp.coverage.threshold_tau);

    // (c) Query on the record's own axis: matched, gate cleared.
    let resp = respond(&e0, None);
    assert_eq!(resp.coverage.outcome, CoverageOutcome::Matched);
    assert!(resp.coverage.top_similarity >= resp.coverage.threshold_tau);
    assert!(resp.coverage.threshold_tau > 0.0, "tau reported");
}

#[test]
fn test_high_confidence_no_hints() {
    let db = YantrikDB::new(":memory:", 8).unwrap();
    let emb = vec_seed(5.0, 8);

    // Record several memories with similar embeddings so density is high
    for i in 0..5 {
        db.record(
            &format!("exact match memory {}", i),
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

    // Recall with exact same embedding and top_k matching the number of stored memories.
    // This maximises the density signal (results.len / top_k = 1.0).
    let response = db
        .recall_with_response(
            &emb, 5, None, None, false, false, None, true, None, None, None,
        )
        .unwrap();

    assert!(!response.results.is_empty());
    // With 5 exact matches and top_k=5: signal_density=1.0, signal_sim~1.0
    // confidence = 0.35*sim + 0.25*gap + 0.20*(1/4) + 0.20*1.0
    // should be >= 0.60
    // Tolerance of 1e-6 — the engine's own ranking resolution (fix h):
    // records within the float-summation jitter band now tie under the
    // quantized comparator and rid picks the order, so the top result
    // can differ from the raw-float winner by <1e-6 in score. 0.60 was
    // brushing that band (observed 0.5999999993 vs 0.6000000001).
    assert!(
        response.confidence >= 0.60 - 1e-6,
        "Confidence should be ~>= 0.60 for exact match with full density, got {}",
        response.confidence,
    );
    assert!(
        response.hints.is_empty(),
        "Hints should be empty for high-confidence recall, got {} hints",
        response.hints.len(),
    );
}

#[test]
fn test_low_confidence_has_hints() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // Record a memory with one embedding
    db.record(
        "something about cats",
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

    // Recall with a very different embedding and a short query_text.
    // The short query_text (<=3 words) triggers the "specificity" hint, and
    // low density (1 result / 10 top_k) keeps confidence below 0.60.
    let far_emb = vec_seed(100.0, 8);
    let response = db
        .recall_with_response(
            &far_emb,
            10,
            None,
            None,
            false,
            false,
            Some("cats"), // short query_text triggers specificity hint
            true,
            None,
            None,
            None,
        )
        .unwrap();

    // With only 1 memory, top_k=10, no entities, no gap — confidence should be low
    assert!(
        response.confidence < 0.60,
        "Confidence should be < 0.60 for distant query, got {}",
        response.confidence,
    );
    assert!(
        !response.hints.is_empty(),
        "Hints should be non-empty for low-confidence recall with short query_text",
    );
}

#[test]
fn test_recall_refine_excludes_originals() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // Record 5 memories with distinct embeddings
    let mut rids = Vec::new();
    for i in 1..=5 {
        let rid = db
            .record(
                &format!("memory number {}", i),
                "episodic",
                0.5,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(i as f32, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        rids.push(rid);
    }

    // First recall: get top 2
    let first_results = db
        .recall(
            &vec_seed(1.0, 8),
            2,
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
        )
        .unwrap();
    assert_eq!(first_results.len(), 2);
    let original_rids: Vec<String> = first_results.iter().map(|r| r.rid.clone()).collect();

    // Refine: exclude the first 2 RIDs
    let refined = db
        .recall_refine(
            &vec_seed(1.0, 8), // original query
            &vec_seed(2.0, 8), // refinement embedding
            &original_rids
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<&str>>()
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<String>>(),
            3, // top_k
            None,
            None,
            None,
        )
        .unwrap();

    // Refined results should not contain any of the original RIDs
    for result in &refined.results {
        assert!(
            !original_rids.contains(&result.rid),
            "Refined result should not contain original RID {}, but it does",
            result.rid,
        );
    }
}

#[test]
fn test_recall_refine_returns_response() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // Record a few memories
    for i in 1..=4 {
        db.record(
            &format!("refine test {}", i),
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(i as f32, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    }

    let exclude: Vec<String> = vec![];
    let response = db
        .recall_refine(
            &vec_seed(1.0, 8),
            &vec_seed(2.0, 8),
            &exclude,
            3,
            None,
            None,
            None,
        )
        .unwrap();

    // Verify RecallResponse structure
    assert!(response.confidence >= 0.0 && response.confidence <= 1.0);
    assert!(!response.retrieval_summary.sources_used.is_empty());
    assert!(response.retrieval_summary.candidate_count > 0);
    // hints may or may not be present
    let _ = &response.hints;
}

#[test]
fn test_retrieval_summary_fields() {
    let db = YantrikDB::new(":memory:", 8).unwrap();

    // Record some memories so recall has candidates
    for i in 1..=3 {
        db.record(
            &format!("summary test {}", i),
            "episodic",
            0.5,
            0.0,
            604800.0,
            &empty_meta(),
            &vec_seed(i as f32, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    }

    let response = db
        .recall_with_response(
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
        )
        .unwrap();

    let summary = &response.retrieval_summary;
    assert!(
        summary.top_similarity > 0.0,
        "top_similarity should be > 0, got {}",
        summary.top_similarity
    );
    assert!(
        summary.sources_used.contains(&"hnsw".to_string()),
        "sources_used should contain 'hnsw', got {:?}",
        summary.sources_used,
    );
    assert!(
        summary.candidate_count > 0,
        "candidate_count should be > 0, got {}",
        summary.candidate_count
    );
}
