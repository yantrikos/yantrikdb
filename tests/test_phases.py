"""Tests for YantrikDB Phase 1 (Interactive Recall), Phase 2 (Adaptive Learning),
and Phase 3 (Richer Dimensions) features.

Each test class maps to a phase; an Integration class covers the full lifecycle.
"""

import math

import pytest

from yantrikdb import YantrikDB


# -- Helpers ---------------------------------------------------------------

DIM = 8  # tiny embeddings for fast tests


def _vec(seed: float) -> list[float]:
    """Generate a deterministic unit vector with good angular diversity.

    Uses a hash-like scramble so that _vec(0), _vec(1), ... are spread
    across the unit sphere rather than nearly parallel.
    """
    raw = [
        math.sin(seed * 1.7 + i * 2.3) + math.cos(seed * 0.3 + i * 3.1)
        for i in range(DIM)
    ]
    norm = math.sqrt(sum(x * x for x in raw))
    if norm < 1e-9:
        raw[0] = 1.0
        norm = 1.0
    return [x / norm for x in raw]


@pytest.fixture
def db():
    """In-memory YantrikDB with no embedder (pre-computed vectors only)."""
    engine = YantrikDB(db_path=":memory:", embedding_dim=DIM)
    yield engine
    engine.close()


# ==========================================================================
# Phase 3 -- Richer Dimensions (certainty, domain, source, emotional_state)
# ==========================================================================


class TestPhase3RicherDimensions:

    # 1
    def test_record_with_new_dimensions(self, db):
        """Record with certainty, domain, source, emotional_state. Verify all fields."""
        rid = db.record(
            "project deadline moved to Friday",
            embedding=_vec(1.0),
            certainty=0.95,
            domain="work",
            source="manager",
            emotional_state="stressed",
        )
        mem = db.get(rid)
        assert mem is not None
        assert mem["certainty"] == 0.95
        assert mem["domain"] == "work"
        assert mem["source"] == "manager"
        assert mem["emotional_state"] == "stressed"

    # 2
    def test_default_dimension_values(self, db):
        """Record without new dims; verify defaults."""
        rid = db.record("just a plain memory", embedding=_vec(2.0))
        mem = db.get(rid)
        assert mem["certainty"] == 0.8
        assert mem["domain"] == "general"
        assert mem["source"] == "user"
        assert mem["emotional_state"] is None

    # 3
    def test_domain_filter_in_recall(self, db):
        """Record 3 memories: 2 domain='work', 1 domain='health'. Filter domain='work'."""
        db.record("quarterly review prep", embedding=_vec(1.0), domain="work")
        db.record("sprint planning notes", embedding=_vec(1.1), domain="work")
        db.record("morning run schedule", embedding=_vec(1.2), domain="health")

        results = db.recall(
            query_embedding=_vec(1.0),
            top_k=10,
            domain="work",
            skip_reinforce=True,
        )
        assert len(results) == 2
        assert all(r["domain"] == "work" for r in results)

    # 4
    def test_source_filter_in_recall(self, db):
        """Record 3 memories: 2 source='user', 1 source='system'. Filter source='user'."""
        db.record("I like coffee", embedding=_vec(3.0), source="user")
        db.record("Meeting at 3pm", embedding=_vec(3.1), source="user")
        db.record("Auto-generated summary", embedding=_vec(3.2), source="system")

        results = db.recall(
            query_embedding=_vec(3.0),
            top_k=10,
            source="user",
            skip_reinforce=True,
        )
        assert len(results) == 2
        assert all(r["source"] == "user" for r in results)

    # 5
    def test_combined_domain_source_filter(self, db):
        """Filter by both domain and source simultaneously."""
        db.record("team standup", embedding=_vec(4.0), domain="work", source="user")
        db.record("auto deploy log", embedding=_vec(4.1), domain="work", source="system")
        db.record("gym workout", embedding=_vec(4.2), domain="health", source="user")
        db.record("health alert", embedding=_vec(4.3), domain="health", source="system")

        results = db.recall(
            query_embedding=_vec(4.0),
            top_k=10,
            domain="work",
            source="user",
            skip_reinforce=True,
        )
        assert len(results) == 1
        assert results[0]["domain"] == "work"
        assert results[0]["source"] == "user"

    # 6
    def test_dimensions_in_recall_results(self, db):
        """Each recall result dict should have the new dimension keys."""
        db.record(
            "recall dim check",
            embedding=_vec(5.0),
            certainty=0.7,
            domain="science",
            source="paper",
            emotional_state="curious",
        )
        results = db.recall(
            query_embedding=_vec(5.0),
            top_k=1,
            skip_reinforce=True,
        )
        assert len(results) == 1
        r = results[0]
        assert "certainty" in r
        assert "domain" in r
        assert "source" in r
        assert "emotional_state" in r
        assert r["certainty"] == 0.7
        assert r["domain"] == "science"
        assert r["source"] == "paper"
        assert r["emotional_state"] == "curious"

    # 7
    def test_batch_record_with_dimensions(self, db):
        """record_batch should accept and store certainty, domain, source."""
        inputs = [
            {
                "text": "batch work 1",
                "embedding": _vec(6.0),
                "certainty": 0.9,
                "domain": "work",
                "source": "colleague",
            },
            {
                "text": "batch health 1",
                "embedding": _vec(6.1),
                "certainty": 0.6,
                "domain": "health",
                "source": "app",
            },
            {
                "text": "batch general",
                "embedding": _vec(6.2),
                # defaults
            },
        ]
        rids = db.record_batch(inputs)
        assert len(rids) == 3

        m0 = db.get(rids[0])
        assert m0["certainty"] == 0.9
        assert m0["domain"] == "work"
        assert m0["source"] == "colleague"

        m1 = db.get(rids[1])
        assert m1["certainty"] == 0.6
        assert m1["domain"] == "health"
        assert m1["source"] == "app"

        m2 = db.get(rids[2])
        assert m2["certainty"] == 0.8  # default
        assert m2["domain"] == "general"  # default
        assert m2["source"] == "user"  # default

    # 8
    def test_correct_preserves_dimensions(self, db):
        """Correcting a memory should preserve the original dimensions.

        Issue #47 (v0.7.20): correct() is now in-place. `reason` is
        required, `embedding` removed (HNSW limitation).
        """
        rid = db.record(
            "original text",
            embedding=_vec(7.0),
            domain="work",
            source="manager",
            certainty=0.85,
        )
        # v0.9.3: importance correction (text corrections are refused).
        result = db.correct(rid, reason="test", new_importance=0.9)
        corrected = db.get(result["corrected_rid"])
        assert corrected["domain"] == "work"
        assert corrected["source"] == "manager"
        assert corrected["certainty"] == 0.85


# ==========================================================================
# Phase 1 -- Interactive Recall (recall_with_response, recall_refine)
# ==========================================================================


class TestPhase1InteractiveRecall:

    # 9
    def test_recall_with_response_returns_dict(self, db):
        """recall_with_response should return a dict with the expected top-level keys."""
        db.record("test memory for response", embedding=_vec(10.0))
        resp = db.recall_with_response(query_embedding=_vec(10.0))
        assert isinstance(resp, dict)
        assert "results" in resp
        assert "confidence" in resp
        assert "retrieval_summary" in resp
        assert "hints" in resp

    # 10
    def test_confidence_is_float(self, db):
        """confidence should be a float in [0.0, 1.0]."""
        db.record("confidence check", embedding=_vec(11.0))
        resp = db.recall_with_response(query_embedding=_vec(11.0))
        conf = resp["confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    # 11
    def test_retrieval_summary_structure(self, db):
        """retrieval_summary should contain top_similarity, score_spread, sources_used, candidate_count."""
        db.record("summary structure test", embedding=_vec(12.0))
        resp = db.recall_with_response(query_embedding=_vec(12.0))
        summary = resp["retrieval_summary"]
        assert isinstance(summary, dict)
        assert "top_similarity" in summary
        assert "score_spread" in summary
        assert "sources_used" in summary
        assert "candidate_count" in summary
        # Types
        assert isinstance(summary["top_similarity"], float)
        assert isinstance(summary["score_spread"], float)
        assert isinstance(summary["sources_used"], list)
        assert isinstance(summary["candidate_count"], int)

    # 12
    def test_high_confidence_no_hints(self, db):
        """Recall with exact same embedding should yield moderate+ confidence and few/no hints."""
        emb = _vec(13.0)
        db.record("exact match memory", embedding=emb)

        resp = db.recall_with_response(query_embedding=emb, skip_reinforce=True)
        assert resp["confidence"] >= 0.35
        # Hints should be empty or minimal when the match is strong
        assert isinstance(resp["hints"], list)
        assert len(resp["hints"]) == 0

    # 13
    def test_low_similarity_generates_hints(self, db):
        """Recall with low confidence should produce hints."""
        # Store a single memory at a specific location on the unit sphere
        basis_a = [0.0] * DIM
        basis_a[0] = 1.0
        db.record("stored at basis-0", embedding=basis_a)

        # Query with an orthogonal direction so similarity is ~0
        basis_b = [0.0] * DIM
        basis_b[1] = 1.0

        # Use a short query (<=3 words) and top_k=10 with only 1 candidate
        # to keep confidence well below 0.60 (low similarity, low density).
        resp = db.recall_with_response(
            query="hi",
            query_embedding=basis_b,
            top_k=10,
            skip_reinforce=True,
        )
        # Verify confidence is low enough to trigger hints
        assert resp["confidence"] < 0.60, f"Expected low confidence, got {resp['confidence']}"
        assert isinstance(resp["hints"], list)
        assert len(resp["hints"]) > 0
        # Each hint should have required fields
        for hint in resp["hints"]:
            assert "hint_type" in hint
            assert "suggestion" in hint

    # 14
    def test_recall_refine_basic(self, db):
        """recall_refine should exclude previously seen RIDs."""
        # Record 5 memories with spread-out vectors
        rids = []
        for i in range(5):
            rid = db.record(f"refine memory {i}", embedding=_vec(float(i) * 3.0))
            rids.append(rid)

        # First recall: get top 2
        first_results = db.recall(
            query_embedding=_vec(0.0), top_k=2, skip_reinforce=True
        )
        first_rids = [r["rid"] for r in first_results]
        assert len(first_rids) == 2

        # Refine: exclude those 2, ask for more with a slightly different query
        refined = db.recall_refine(
            original_query_embedding=_vec(0.0),
            refinement_embedding=_vec(3.0),
            original_rids=first_rids,
            top_k=3,
        )
        assert isinstance(refined, dict)
        assert "results" in refined
        # None of the excluded RIDs should appear in the refined results
        refined_rids = [r["rid"] for r in refined["results"]]
        for excluded in first_rids:
            assert excluded not in refined_rids

    # 15
    def test_recall_refine_returns_response(self, db):
        """recall_refine should return a dict with results, confidence, hints."""
        db.record("refine response mem", embedding=_vec(20.0))
        refined = db.recall_refine(
            original_query_embedding=_vec(20.0),
            refinement_embedding=_vec(20.1),
            original_rids=[],
            top_k=5,
        )
        assert isinstance(refined, dict)
        assert "results" in refined
        assert "confidence" in refined
        assert "hints" in refined
        assert isinstance(refined["confidence"], float)
        assert isinstance(refined["hints"], list)


# ==========================================================================
# Phase 2 -- Adaptive Learning (recall_feedback, learned_weights)
# ==========================================================================


class TestPhase2AdaptiveLearning:

    # 16
    def test_recall_feedback_basic(self, db):
        """recall_feedback with minimal args should not raise."""
        rid = db.record("feedback target", embedding=_vec(30.0))
        # This should succeed without error
        db.recall_feedback(rid, "relevant")

    # 17
    def test_recall_feedback_with_all_params(self, db):
        """recall_feedback with all optional params should not raise."""
        rid = db.record("full feedback target", embedding=_vec(31.0))
        db.recall_feedback(
            rid,
            "relevant",
            query_text="test query",
            query_embedding=_vec(31.0),
            score_at_retrieval=0.85,
            rank_at_retrieval=1,
        )

    # 18
    def test_learned_weights_defaults(self, db):
        """learned_weights() should return default weights on a fresh DB."""
        w = db.learned_weights()
        assert isinstance(w, dict)
        assert w["w_sim"] == pytest.approx(0.50)
        assert w["w_decay"] == pytest.approx(0.20)
        assert w["w_recency"] == pytest.approx(0.30)
        # Additional fields from the struct
        assert "gate_tau" in w
        assert "alpha_imp" in w
        assert "keyword_boost" in w
        assert "generation" in w

    # 19
    def test_think_after_feedback(self, db):
        """Providing feedback then running think() should not error."""
        rid = db.record("think after feedback", embedding=_vec(32.0))
        db.recall_feedback(rid, "relevant", query_text="test")
        result = db.think()
        assert isinstance(result, dict)
        assert "triggers" in result
        assert "consolidation_count" in result
        assert "duration_ms" in result


# ==========================================================================
# Integration -- Full Interactive Recall Flow
# ==========================================================================


class TestPhaseIntegration:

    # 20
    def test_full_interactive_recall_flow(self, db):
        """Full lifecycle: record -> recall_with_response -> inspect -> refine -> feedback."""
        # Step 1: Record memories across different domains
        rid_work1 = db.record(
            "Sprint planning for Q1",
            embedding=_vec(50.0),
            domain="work",
            source="user",
            certainty=0.9,
        )
        rid_work2 = db.record(
            "Code review guidelines updated",
            embedding=_vec(50.5),
            domain="work",
            source="system",
            certainty=0.7,
        )
        rid_health = db.record(
            "Morning yoga routine",
            embedding=_vec(55.0),
            domain="health",
            source="user",
            certainty=0.8,
        )
        rid_personal = db.record(
            "Birthday party planning",
            embedding=_vec(60.0),
            domain="personal",
            source="user",
            certainty=0.95,
            emotional_state="excited",
        )

        # Step 2: Recall with response (work domain)
        resp = db.recall_with_response(
            query_embedding=_vec(50.0),
            top_k=3,
            domain="work",
            skip_reinforce=True,
        )

        # Verify structure
        assert isinstance(resp, dict)
        assert "results" in resp
        assert "confidence" in resp
        assert "retrieval_summary" in resp
        assert "hints" in resp

        # Should find the 2 work memories
        assert len(resp["results"]) == 2
        result_rids = [r["rid"] for r in resp["results"]]
        assert rid_work1 in result_rids
        assert rid_work2 in result_rids

        # Confidence should be a valid float
        assert isinstance(resp["confidence"], float)
        assert 0.0 <= resp["confidence"] <= 1.0

        # Retrieval summary
        summary = resp["retrieval_summary"]
        assert summary["candidate_count"] >= 2

        # Step 3: Dimensions present in results
        for r in resp["results"]:
            assert r["domain"] == "work"
            assert "certainty" in r
            assert "source" in r

        # Step 4: Refine the search (exclude first results, shift query slightly)
        first_rids = [r["rid"] for r in resp["results"]]
        refined = db.recall_refine(
            original_query_embedding=_vec(50.0),
            refinement_embedding=_vec(55.0),
            original_rids=first_rids,
            top_k=5,
        )
        assert isinstance(refined, dict)
        assert "results" in refined
        # Excluded RIDs should not appear
        for r in refined["results"]:
            assert r["rid"] not in first_rids

        # Step 5: Provide feedback on the best result
        db.recall_feedback(
            rid_work1,
            "relevant",
            query_text="sprint planning",
            query_embedding=_vec(50.0),
            score_at_retrieval=resp["results"][0]["score"],
            rank_at_retrieval=1,
        )

        # Step 6: Verify learned weights still valid
        w = db.learned_weights()
        assert w["w_sim"] > 0
        assert w["w_decay"] >= 0
        assert w["w_recency"] >= 0

        # Step 7: Run think to process feedback
        think_result = db.think()
        assert isinstance(think_result, dict)
        assert "duration_ms" in think_result

    def test_phase3_dimensions_flow_through_all_apis(self, db):
        """Verify dimensions survive record -> get -> recall -> batch -> correct pipeline."""
        # Record
        rid = db.record(
            "dimensions flow test",
            embedding=_vec(70.0),
            certainty=0.65,
            domain="science",
            source="experiment",
            emotional_state="focused",
        )

        # Get
        mem = db.get(rid)
        assert mem["certainty"] == 0.65
        assert mem["domain"] == "science"
        assert mem["source"] == "experiment"
        assert mem["emotional_state"] == "focused"

        # Recall
        results = db.recall(
            query_embedding=_vec(70.0), top_k=1, skip_reinforce=True
        )
        assert len(results) == 1
        assert results[0]["certainty"] == 0.65
        assert results[0]["domain"] == "science"
        assert results[0]["source"] == "experiment"
        assert results[0]["emotional_state"] == "focused"

        # recall_with_response
        resp = db.recall_with_response(
            query_embedding=_vec(70.0), top_k=1, skip_reinforce=True
        )
        assert resp["results"][0]["domain"] == "science"

        # Correct — Issue #47 (v0.7.20): in-place, `reason` required.
        # v0.9.3: importance correction (text corrections are refused).
        correction = db.correct(rid, reason="test", new_importance=0.9)
        corrected = db.get(correction["corrected_rid"])
        assert corrected["domain"] == "science"
        assert corrected["source"] == "experiment"
        assert corrected["certainty"] == 0.65

    def test_recall_with_response_empty_db(self, db):
        """recall_with_response on an empty DB should return valid structure with empty results."""
        resp = db.recall_with_response(query_embedding=_vec(1.0))
        assert isinstance(resp, dict)
        assert resp["results"] == []
        assert isinstance(resp["confidence"], float)
        assert isinstance(resp["hints"], list)

    def test_recall_refine_empty_exclusions(self, db):
        """recall_refine with no exclusions should return all matches."""
        for i in range(3):
            db.record(f"refine no excl {i}", embedding=_vec(float(i) * 5.0))

        refined = db.recall_refine(
            original_query_embedding=_vec(0.0),
            refinement_embedding=_vec(0.0),
            original_rids=[],
            top_k=10,
        )
        assert len(refined["results"]) == 3

    def test_multiple_feedback_rounds(self, db):
        """Multiple feedback calls should accumulate without error."""
        rids = []
        for i in range(5):
            rid = db.record(f"multi feedback {i}", embedding=_vec(float(i) * 2.0))
            rids.append(rid)

        # Provide feedback on multiple memories
        for rid in rids:
            db.recall_feedback(rid, "relevant", query_text="multi feedback test")

        # Also provide negative feedback
        db.recall_feedback(rids[0], "irrelevant", query_text="negative test")

        # Think should still work after all that feedback
        result = db.think()
        assert isinstance(result, dict)
