"""Tests for the consolidation engine."""

import math
import time

import pytest

from yantrikdb import YantrikDB, record_synthesis
from yantrikdb.consolidate import (
    _cosine_similarity,
    _extractive_summary,
    _find_clusters,
    consolidate,
    find_consolidation_candidates,
)

DIM = 8


def _vec(seed: float) -> list[float]:
    """Generate a unit vector. Uses sin/cos to ensure different seeds produce
    genuinely different directions."""
    raw = [math.sin(seed * (i + 1) * 1.7) + math.cos(seed * (i + 2) * 0.3) for i in range(DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


@pytest.fixture
def db():
    engine = YantrikDB(db_path=":memory:", embedding_dim=DIM)
    yield engine
    engine.close()


class TestCosineSimlarity:
    def test_identical_vectors(self):
        v = _vec(1.0)
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_similar_vectors(self):
        a = _vec(1.0)
        b = _vec(1.1)
        sim = _cosine_similarity(a, b)
        assert sim > 0.8  # close seeds -> similar vectors


class TestFindClusters:
    def test_clusters_similar_memories(self):
        now = time.time()
        memories = [
            {"rid": "a", "text": "t1", "embedding": _vec(1.0), "created_at": now, "importance": 0.5, "valence": 0.0, "half_life": 604800, "last_access": now, "metadata": {}},
            {"rid": "b", "text": "t2", "embedding": _vec(1.05), "created_at": now + 100, "importance": 0.5, "valence": 0.0, "half_life": 604800, "last_access": now, "metadata": {}},
            {"rid": "c", "text": "t3", "embedding": _vec(10.0), "created_at": now + 200, "importance": 0.5, "valence": 0.0, "half_life": 604800, "last_access": now, "metadata": {}},
        ]
        clusters = _find_clusters(memories, sim_threshold=0.9, min_cluster_size=2)
        # a and b should cluster, c should not
        assert len(clusters) == 1
        rids = {m["rid"] for m in clusters[0]}
        assert "a" in rids
        assert "b" in rids
        assert "c" not in rids

    def test_no_clusters_below_threshold(self):
        now = time.time()
        memories = [
            {"rid": "a", "text": "t1", "embedding": _vec(1.0), "created_at": now, "importance": 0.5, "valence": 0.0, "half_life": 604800, "last_access": now, "metadata": {}},
            {"rid": "b", "text": "t2", "embedding": _vec(100.0), "created_at": now + 100, "importance": 0.5, "valence": 0.0, "half_life": 604800, "last_access": now, "metadata": {}},
        ]
        clusters = _find_clusters(memories, sim_threshold=0.9, min_cluster_size=2)
        assert len(clusters) == 0

    def test_time_window_respected(self):
        now = time.time()
        memories = [
            {"rid": "a", "text": "t1", "embedding": _vec(1.0), "created_at": now, "importance": 0.5, "valence": 0.0, "half_life": 604800, "last_access": now, "metadata": {}},
            {"rid": "b", "text": "t2", "embedding": _vec(1.05), "created_at": now + 30 * 86400, "importance": 0.5, "valence": 0.0, "half_life": 604800, "last_access": now, "metadata": {}},
        ]
        # Similar embeddings but 30 days apart (> 7 day window)
        clusters = _find_clusters(memories, sim_threshold=0.9, time_window_days=7.0, min_cluster_size=2)
        assert len(clusters) == 0


class TestExtractiveSummary:
    def test_single_memory(self):
        memories = [{"text": "The cat sat on the mat", "importance": 0.5}]
        assert _extractive_summary(memories) == "The cat sat on the mat"

    def test_multiple_memories(self):
        memories = [
            {"text": "We chose PyTorch", "importance": 0.9},
            {"text": "The team discussed frameworks", "importance": 0.5},
        ]
        summary = _extractive_summary(memories)
        assert "We chose PyTorch" in summary
        assert "The team discussed frameworks" in summary

    def test_highest_importance_leads(self):
        memories = [
            {"text": "Low importance", "importance": 0.1},
            {"text": "High importance lead", "importance": 0.9},
        ]
        summary = _extractive_summary(memories)
        assert summary.startswith("High importance lead")


class TestConsolidate:
    def test_record_synthesis_public_api_preserves_sources_and_clocks(self, db):
        rid1 = db.record(
            "Bryan shared storytelling advice",
            embedding=_vec(1.0),
            created_at=1_000.0,
        )
        rid2 = db.record(
            "Shawn added storytelling-impact detail",
            embedding=_vec(2.0),
            created_at=2_000.0,
        )

        result = record_synthesis(
            db,
            [rid2, rid1],
            "The user incorporated storytelling input from Bryan and Shawn",
            "contributed",
            "test:contributed:storytelling",
            embedding=_vec(3.0),
        )
        synthesis = db.get(result["consolidated_rid"])

        assert synthesis["source"] == "inference"
        assert synthesis["created_at"] == 2_000.0
        assert synthesis["metadata"]["first_mention_at"] == 1_000.0
        assert set(synthesis["metadata"]["evidence_ids"]) == {rid1, rid2}
        assert db.get(rid1)["consolidation_status"] == "active"
        assert db.get(rid2)["consolidation_status"] == "active"

    def test_dry_run(self, db):
        # Record similar memories
        db.record("Python is a great language", importance=0.7, embedding=_vec(1.0))
        db.record("Python is widely used", importance=0.6, embedding=_vec(1.02))

        results = consolidate(db, sim_threshold=0.9, dry_run=True)
        assert len(results) >= 1
        assert "preview_summary" in results[0]
        assert results[0]["cluster_size"] >= 2
        # Nothing should be consolidated in dry run
        assert db.stats()["consolidated_memories"] == 0

    def test_full_consolidation(self, db):
        rid1 = db.record("We decided to use PyTorch", importance=0.8, embedding=_vec(1.0))
        rid2 = db.record("PyTorch was chosen for the project", importance=0.6, embedding=_vec(1.02))
        rid3 = db.record("Something completely different", importance=0.5, embedding=_vec(10.0))

        results = consolidate(db, sim_threshold=0.9)

        assert len(results) == 1
        r = results[0]
        assert r["cluster_size"] == 2
        assert rid1 in r["source_rids"]
        assert rid2 in r["source_rids"]

        # Source memories should be marked as consolidated
        mem1 = db.get(rid1)
        mem2 = db.get(rid2)
        assert mem1["consolidation_status"] == "consolidated"
        assert mem2["consolidation_status"] == "consolidated"
        assert mem1["consolidated_into"] == r["consolidated_rid"]

        # New consolidated memory should exist
        consolidated = db.get(r["consolidated_rid"])
        assert consolidated is not None
        assert consolidated["type"] == "semantic"
        assert consolidated["consolidation_status"] == "active"

        # Unrelated memory should be untouched
        mem3 = db.get(rid3)
        assert mem3["consolidation_status"] == "active"

    def test_consolidation_preserves_recall(self, db):
        """Consolidated memories should still be findable via recall."""
        db.record("PyTorch is great for deep learning", importance=0.8, embedding=_vec(1.0))
        db.record("PyTorch supports GPU acceleration", importance=0.6, embedding=_vec(1.02))

        # Consolidate
        results = consolidate(db, sim_threshold=0.9)
        assert len(results) == 1

        # Recall should find the consolidated memory
        recall_results = db.recall(query_embedding=_vec(1.0), top_k=5)
        assert len(recall_results) >= 1
        # The consolidated memory should appear
        consolidated_rids = {r["consolidated_rid"] for r in results}
        found_consolidated = any(r["rid"] in consolidated_rids for r in recall_results)
        assert found_consolidated

    def test_consolidation_transfers_edges(self, db):
        rid1 = db.record("Sarah built the pipeline", importance=0.7, embedding=_vec(1.0))
        rid2 = db.record("Sarah completed the data work", importance=0.6, embedding=_vec(1.02))

        db.relate(rid1, "Sarah", rel_type="involves")
        db.relate(rid1, "pipeline", rel_type="about")

        results = consolidate(db, sim_threshold=0.9)
        assert len(results) == 1

        # Consolidated memory should have edges
        edges = db.get_edges(results[0]["consolidated_rid"])
        assert len(edges) > 0

    def test_importance_boosted(self, db):
        # Entity-free text ON PURPOSE. consolidate()'s entity-overlap guard
        # (require_entity_overlap, default True since v0.6.0) refuses to merge two
        # memories whose entity sets are non-empty and DISJOINT. Capitalized text
        # like "Important fact A" / "Related fact B" extracts distinct entities, so
        # this test used to pass only when it outran the materializer that
        # populates memory_entities — a race, proven by
        # `diagnostic_consolidate_entity_overlap_guard_is_materializer_race`
        # (no-drain: 0 entities -> 1 cluster; drained: 2 entities -> 0 clusters).
        # Similarity here comes from the EXPLICIT embeddings below, so the text
        # affects nothing except entity extraction: lowercasing it makes the test
        # deterministic without weakening what it checks.
        db.record("the fact that was written", importance=0.8, embedding=_vec(1.0))
        db.record("the related fact nearby", importance=0.6, embedding=_vec(1.02))

        results = consolidate(db, sim_threshold=0.9)
        consolidated = db.get(results[0]["consolidated_rid"])

        # Importance should be boosted (max * 1.1)
        assert consolidated["importance"] >= 0.8

    def test_source_importance_reduced(self, db):
        rid1 = db.record("Fact 1", importance=0.8, embedding=_vec(1.0))
        rid2 = db.record("Fact 2", importance=0.6, embedding=_vec(1.02))

        consolidate(db, sim_threshold=0.9)

        mem1 = db.get(rid1)
        mem2 = db.get(rid2)
        # Source importance reduced to 30%
        assert mem1["importance"] < 0.3
        assert mem2["importance"] < 0.2

    def test_stats_after_consolidation(self, db):
        # Entity-free text — see test_importance_boosted. "A1"/"A2" extracted as
        # DISTINCT entities, so the entity-overlap guard blocked the merge
        # whenever the materializer won the race, flaking this test ~50%.
        db.record("the first note", importance=0.7, embedding=_vec(1.0))
        db.record("the second note", importance=0.6, embedding=_vec(1.02))
        db.record("an unrelated note", importance=0.5, embedding=_vec(10.0))

        assert db.stats()["active_memories"] == 3
        assert db.stats()["consolidated_memories"] == 0

        consolidate(db, sim_threshold=0.9)

        stats = db.stats()
        assert stats["consolidated_memories"] == 2  # two sources consolidated
        assert stats["active_memories"] == 2  # 1 consolidated + 1 untouched

    def test_no_clusters_no_consolidation(self, db):
        db.record("Totally different A", importance=0.7, embedding=_vec(1.0))
        db.record("Totally different B", importance=0.6, embedding=_vec(100.0))

        results = consolidate(db, sim_threshold=0.99)
        assert len(results) == 0
        assert db.stats()["consolidated_memories"] == 0
