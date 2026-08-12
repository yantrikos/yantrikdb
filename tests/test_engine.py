"""Tests for the YantrikDB engine — record, recall, relate, decay, forget."""

import math
import time

import pytest

from yantrikdb import YantrikDB


# ── Helpers ──────────────────────────────────────────────

DIM = 8  # tiny embeddings for fast tests


def _vec(seed: float) -> list[float]:
    """Generate a deterministic unit vector with good angular diversity.

    Uses a hash-like scramble so that _vec(0), _vec(1), ... are spread
    across the unit sphere rather than nearly parallel.
    """
    raw = [math.sin(seed * 1.7 + i * 2.3) + math.cos(seed * 0.3 + i * 3.1) for i in range(DIM)]
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


# ── record() ────────────────────────────────────────────

class TestRecord:
    def test_record_returns_rid(self, db):
        rid = db.record("hello world", embedding=_vec(1.0))
        assert isinstance(rid, str)
        assert len(rid) == 36  # UUIDv7 format

    def test_record_stores_memory(self, db):
        rid = db.record(
            "test memory",
            memory_type="semantic",
            importance=0.8,
            valence=-0.3,
            embedding=_vec(2.0),
        )
        mem = db.get(rid)
        assert mem is not None
        assert mem["text"] == "test memory"
        assert mem["type"] == "semantic"
        assert mem["importance"] == 0.8
        assert mem["valence"] == -0.3
        assert mem["consolidation_status"] == "active"

    def test_record_with_metadata(self, db):
        rid = db.record(
            "with meta",
            metadata={"source": "test", "tags": ["a", "b"]},
            embedding=_vec(3.0),
        )
        mem = db.get(rid)
        assert mem["metadata"]["source"] == "test"
        assert mem["metadata"]["tags"] == ["a", "b"]

    def test_record_without_embedding_raises(self, db):
        with pytest.raises(RuntimeError, match="No embedder configured"):
            db.record("no embedding")

    def test_record_updates_stats(self, db):
        assert db.stats()["active_memories"] == 0
        db.record("one", embedding=_vec(1.0))
        db.record("two", embedding=_vec(2.0))
        assert db.stats()["active_memories"] == 2


# ── recall() ────────────────────────────────────────────

class TestRecall:
    def test_recall_basic(self, db):
        db.record("the cat sat on the mat", embedding=_vec(1.0))
        db.record("dogs are loyal friends", embedding=_vec(5.0))
        db.record("cats love warm places", embedding=_vec(1.1))

        results = db.recall(query_embedding=_vec(1.0), top_k=2)
        assert len(results) == 2
        # Most similar to _vec(1.0) should be first
        assert "cat" in results[0]["text"]

    def test_recall_returns_scores(self, db):
        db.record("memory one", embedding=_vec(1.0))
        results = db.recall(query_embedding=_vec(1.0), top_k=1)
        assert len(results) == 1
        r = results[0]
        assert "score" in r
        assert "scores" in r
        assert "similarity" in r["scores"]
        assert "decay" in r["scores"]
        assert "recency" in r["scores"]
        assert "why_retrieved" in r
        # Explainability: contributions and valence_multiplier
        assert "valence_multiplier" in r["scores"]
        assert r["scores"]["valence_multiplier"] >= 1.0
        assert "contributions" in r["scores"]
        c = r["scores"]["contributions"]
        assert "similarity" in c and "decay" in c and "recency" in c
        assert "importance" in c and "graph_proximity" in c
        # Contributions should be non-negative weighted signals
        assert all(v >= 0.0 for v in c.values())

    def test_recall_respects_top_k(self, db):
        for i in range(20):
            db.record(f"memory {i}", embedding=_vec(float(i)))
        results = db.recall(query_embedding=_vec(0.0), top_k=5)
        assert len(results) == 5

    def test_recall_filters_by_type(self, db):
        db.record("episodic mem", memory_type="episodic", embedding=_vec(1.0))
        db.record("semantic mem", memory_type="semantic", embedding=_vec(1.1))

        results = db.recall(
            query_embedding=_vec(1.0), top_k=10, memory_type="semantic"
        )
        assert all(r["type"] == "semantic" for r in results)

    def test_recall_reinforces_memories(self, db):
        rid = db.record("reinforce me", embedding=_vec(1.0), half_life=1000.0)
        original = db.get(rid)

        db.recall(query_embedding=_vec(1.0), top_k=1)
        after = db.get(rid)

        # half_life should increase by 20%
        assert after["half_life"] > original["half_life"]
        assert after["last_access"] >= original["last_access"]

    def test_recall_empty_db(self, db):
        results = db.recall(query_embedding=_vec(1.0), top_k=5)
        assert results == []

    def test_recall_requires_query(self, db):
        with pytest.raises(ValueError, match="Must provide"):
            db.recall(top_k=5)


# ── relate() ────────────────────────────────────────────

class TestRelate:
    def test_relate_creates_edge(self, db):
        edge_id = db.relate("Alice", "Bob", rel_type="knows")
        assert isinstance(edge_id, str)

        edges = db.get_edges("Alice")
        assert len(edges) == 1
        assert edges[0]["src"] == "Alice"
        assert edges[0]["dst"] == "Bob"
        assert edges[0]["rel_type"] == "knows"

    def test_relate_bidirectional_lookup(self, db):
        db.relate("Alice", "Bob", rel_type="knows")
        assert len(db.get_edges("Alice")) == 1
        assert len(db.get_edges("Bob")) == 1

    def test_relate_upserts_weight(self, db):
        db.relate("A", "B", rel_type="x", weight=0.5)
        db.relate("A", "B", rel_type="x", weight=0.9)

        edges = db.get_edges("A")
        assert len(edges) == 1
        assert edges[0]["weight"] == 0.9

    def test_relate_updates_stats(self, db):
        assert db.stats()["edges"] == 0
        db.relate("X", "Y")
        assert db.stats()["edges"] == 1
        assert db.stats()["entities"] == 2


# ── decay() ─────────────────────────────────────────────

class TestDecay:
    def test_decay_finds_old_memories(self, db):
        rid = db.record("old memory", importance=0.1, half_life=1.0, embedding=_vec(1.0))
        # Manually backdate last_access to simulate time passing
        db._conn.execute(
            "UPDATE memories SET last_access = ? WHERE rid = ?",
            (time.time() - 100, rid),
        )
        db._conn.commit()

        decayed = db.decay(threshold=0.01)
        assert len(decayed) >= 1
        assert any(d["rid"] == rid for d in decayed)

    def test_decay_skips_fresh_memories(self, db):
        db.record("fresh", importance=0.9, half_life=604800.0, embedding=_vec(1.0))
        decayed = db.decay(threshold=0.01)
        assert len(decayed) == 0

    def test_decay_returns_score_info(self, db):
        rid = db.record("decaying", importance=0.5, half_life=1.0, embedding=_vec(1.0))
        db._conn.execute(
            "UPDATE memories SET last_access = ? WHERE rid = ?",
            (time.time() - 50, rid),
        )
        db._conn.commit()

        decayed = db.decay(threshold=1.0)  # high threshold catches everything
        assert len(decayed) >= 1
        d = decayed[0]
        assert "current_score" in d
        assert "days_since_access" in d
        assert "original_importance" in d


# ── forget() ────────────────────────────────────────────

class TestForget:
    def test_forget_tombstones_memory(self, db):
        rid = db.record("forget me", embedding=_vec(1.0))
        assert db.forget(rid) is True

        mem = db.get(rid)
        assert mem["consolidation_status"] == "tombstoned"

    def test_forget_removes_from_vector_index(self, db):
        rid = db.record("forget vec", embedding=_vec(1.0))
        db.forget(rid)

        # Should not appear in recall results
        results = db.recall(query_embedding=_vec(1.0), top_k=10)
        assert all(r["rid"] != rid for r in results)

    def test_forget_nonexistent_returns_false(self, db):
        assert db.forget("nonexistent-rid") is False

    def test_forget_updates_stats(self, db):
        rid = db.record("bye", embedding=_vec(1.0))
        assert db.stats()["active_memories"] == 1
        db.forget(rid)
        assert db.stats()["active_memories"] == 0
        assert db.stats()["tombstoned_memories"] == 1


# ── stats() ──────────────────────────────────────────────

class TestStats:
    def test_stats_all_fields(self, db):
        s = db.stats()
        expected_keys = {
            "active_memories", "consolidated_memories", "tombstoned_memories",
            "archived_memories", "edges", "entities", "operations",
            "open_conflicts", "resolved_conflicts",
            "pending_triggers", "active_patterns",
            "scoring_cache_entries", "vec_index_entries",
            "graph_index_entities", "graph_index_edges",
            # v0.10 Item 1: status-led read path adoption surface.
            "status_read_policy", "superseded_records",
            "superseded_served_since_boot",
            # v0.10 Item 4a.4: anti-laundering gate adoption surface.
            "provenance_gate_mode", "provenance_flagged_since_boot",
            # v0.12.1: embedder-window truncation surface + chunked
            # embeddings (window in chars once probed, overflows lost vs
            # chunked, durable window-vector count).
            "embedder_window_chars", "embedder_truncated_writes",
            "embedder_chunked_writes", "chunk_vectors",
            # 0.13.1 C5b: possessive-pollution census — apostrophe
            # entities remaining vs migration aliases written.
            "apostrophe_entities", "possessive_aliases",
        }
        assert set(s.keys()) == expected_keys
        # Fresh databases default to the status-led read path.
        assert s["status_read_policy"] == "exclude_superseded"
        assert s["superseded_records"] == 0
        assert s["superseded_served_since_boot"] == 0
        # Fresh databases default to enforcing the provenance gate; migrated
        # ones default to "warn" (count + nudge, never refuse).
        assert s["provenance_gate_mode"] == "enforce"
        assert s["provenance_flagged_since_boot"] == 0
        # Chunking surface: fresh DB, nothing probed, nothing chunked.
        assert s["embedder_window_chars"] is None
        assert s["embedder_truncated_writes"] == 0
        assert s["embedder_chunked_writes"] == 0
        assert s["chunk_vectors"] == 0

    def test_stats_tracks_operations(self, db):
        db.record("op1", embedding=_vec(1.0))
        db.relate("A", "B")
        s = db.stats()
        # At minimum: 1 record + 1 relate = 2 ops. Engine may track
        # additional derived ops (entity-link, materialize-record-post)
        # depending on feature config — assert lower-bound rather than
        # exact equality so the contract is "ops tracked" not "exactly
        # this many."
        assert s["operations"] >= 2, f"expected >= 2 ops, got {s['operations']}"


# ── v0.10 Item 1: status-led read path ───────────────────

class TestStatusReadPath:
    def _seed_chain(self, db):
        """Old fact superseded by a near-identical new fact."""
        old = db.record("standup is at 9am", embedding=_vec(1.0))
        new = db.record("standup is at 10am", embedding=_vec(1.0))
        db.link(new, old, "supersedes")
        return old, new

    def test_fresh_db_excludes_superseded_by_default(self, db):
        old, new = self._seed_chain(db)
        assert db.status_read_policy() is True
        rids = [r["rid"] for r in db.recall(query_embedding=_vec(1.0), top_k=10)]
        assert new in rids
        assert old not in rids, "superseded record must be excluded (T01 hard zero)"

    def test_include_superseded_readmits_with_typed_fields(self, db):
        old, new = self._seed_chain(db)
        hits = db.recall(query_embedding=_vec(1.0), top_k=10, include_superseded=True)
        by_rid = {r["rid"]: r for r in hits}
        assert by_rid[old]["current_status"] == "superseded"
        assert by_rid[old]["superseded_by"] == new
        assert by_rid[new]["current_status"] == "active"
        assert by_rid[new]["superseded_by"] is None

    def test_legacy_policy_serves_stamped_and_counts_nudge(self, db):
        old, new = self._seed_chain(db)
        db.set_status_read_policy(False)  # simulate migrated pre-v0.10 DB
        rids = [r["rid"] for r in db.recall(query_embedding=_vec(1.0), top_k=10)]
        assert old in rids and new in rids
        s = db.stats()
        assert s["status_read_policy"] == "legacy"
        assert s["superseded_records"] == 1
        assert s["superseded_served_since_boot"] >= 1
        db.set_status_read_policy(True)
        rids = [r["rid"] for r in db.recall(query_embedding=_vec(1.0), top_k=10)]
        assert old not in rids

    def test_recall_with_response_typed_coverage(self, db):
        # T08: empty scope vs below-threshold vs matched, typed.
        resp = db.recall_with_response(
            query_embedding=_vec(1.0), top_k=5, namespace="empty_ns"
        )
        cov = resp["coverage"]
        assert cov["outcome"] == "no_matching_record"
        assert cov["candidate_count"] == 0
        assert cov["namespace"] == "empty_ns"

        db.record("a stored fact", embedding=_vec(1.0))
        resp = db.recall_with_response(query_embedding=_vec(1.0), top_k=5)
        cov = resp["coverage"]
        assert cov["outcome"] == "matched"
        assert cov["top_similarity"] >= cov["threshold_tau"] > 0.0

    def test_what_changed_since(self, db):
        import json
        old, new = self._seed_chain(db)
        changes = json.loads(db.what_changed_since(0.0))
        new_rids = [r["rid"] for r in changes["new_records"]]
        assert old in new_rids and new in new_rids
        transitions = changes["status_transitions"]
        assert len(transitions) == 1
        assert transitions[0]["rid"] == old
        assert transitions[0]["to"] == "superseded"
        assert transitions[0]["by_rid"] == new
        # Nothing after the far future.
        quiet = json.loads(db.what_changed_since(time.time() + 60.0))
        assert quiet["new_records"] == []
        assert quiet["status_transitions"] == []


# ── Integration ──────────────────────────────────────────

class TestIntegration:
    def test_full_lifecycle(self, db):
        """Record -> recall -> relate -> decay -> forget lifecycle."""
        # Record
        rid1 = db.record("Python is great", importance=0.9, embedding=_vec(1.0))
        rid2 = db.record("Rust is fast", importance=0.7, embedding=_vec(5.0))
        rid3 = db.record("Python typing is improving", importance=0.5, embedding=_vec(1.2))

        # Recall — should find Python memories
        results = db.recall(query_embedding=_vec(1.0), top_k=2)
        assert len(results) == 2

        # Relate
        db.relate("Python", "typing", rel_type="has_feature")
        db.relate("Python", "Rust", rel_type="compared_with")

        assert db.stats()["edges"] == 2
        assert db.stats()["entities"] == 3

        # Decay — nothing should be decayed yet
        decayed = db.decay(threshold=0.01)
        assert len(decayed) == 0

        # Forget
        db.forget(rid2)
        assert db.stats()["active_memories"] == 2
        assert db.stats()["tombstoned_memories"] == 1

        # Verify forgotten memory not in recall
        results = db.recall(query_embedding=_vec(5.0), top_k=10)
        assert all(r["rid"] != rid2 for r in results)

    def test_valence_affects_ranking(self, db):
        """High-valence memories should rank higher when similarity is close."""
        db.record("neutral memory", valence=0.0, importance=0.5, embedding=_vec(1.0))
        db.record("emotional memory", valence=0.9, importance=0.5, embedding=_vec(1.01))

        results = db.recall(query_embedding=_vec(1.005), top_k=2)
        # The emotional memory should get a valence boost
        emotional = [r for r in results if "emotional" in r["text"]]
        assert len(emotional) == 1
        assert emotional[0]["score"] > results[-1]["score"]


# ── Graph-augmented recall integration tests ──────────

class TestGraphRecall:
    def test_recall_deterministic_with_skip_reinforce(self, db):
        """Same query with skip_reinforce=True returns identical results every time."""
        for i in range(10):
            db.record(f"memory {i}", embedding=_vec(float(i)))
        query = _vec(3.0)

        r1 = db.recall(query_embedding=query, top_k=5, skip_reinforce=True)
        r2 = db.recall(query_embedding=query, top_k=5, skip_reinforce=True)
        r3 = db.recall(query_embedding=query, top_k=5, skip_reinforce=True)

        rids1 = [r["rid"] for r in r1]
        rids2 = [r["rid"] for r in r2]
        rids3 = [r["rid"] for r in r3]
        assert rids1 == rids2 == rids3

    def test_skip_reinforce_prevents_mutation(self, db):
        """skip_reinforce=True should not modify half_life."""
        rid = db.record("test", embedding=_vec(1.0), half_life=1000.0)
        original = db.get(rid)

        db.recall(query_embedding=_vec(1.0), top_k=1, skip_reinforce=True)
        after = db.get(rid)
        assert after["half_life"] == original["half_life"]

    def test_graph_expansion_toggle(self, db):
        """expand_entities=True should set graph_proximity on connected memories."""
        r1 = db.record("Alice discussed the plan", embedding=_vec(1.0))
        r2 = db.record("Bob reviewed the code", embedding=_vec(5.0))
        db.relate("Alice", "Bob", rel_type="knows")
        db.link_memory_entity(r1, "Alice")
        db.link_memory_entity(r2, "Bob")

        # With expansion off
        results_off = db.recall(
            query="What is Alice working on?",
            query_embedding=_vec(1.0), top_k=10,
            expand_entities=False, skip_reinforce=True,
        )
        for r in results_off:
            assert r["scores"]["graph_proximity"] == 0.0

        # With expansion on
        results_on = db.recall(
            query="What is Alice working on?",
            query_embedding=_vec(1.0), top_k=10,
            expand_entities=True, skip_reinforce=True,
        )
        alice_result = [r for r in results_on if r["rid"] == r1]
        assert len(alice_result) == 1
        assert alice_result[0]["scores"]["graph_proximity"] > 0.0

    def test_entity_type_stored_after_relate(self, db):
        """relate() should classify and store entity_type correctly."""
        db.relate("Sarah", "data pipeline", rel_type="leads")
        db.relate("FAISS", "recommendation engine", rel_type="used_in")

        # Check entity types via internal DB access (rows are dicts)
        sarah_type = db._conn.execute(
            "SELECT entity_type FROM entities WHERE name = 'Sarah'"
        ).fetchone()["entity_type"]
        assert sarah_type == "person"

        faiss_type = db._conn.execute(
            "SELECT entity_type FROM entities WHERE name = 'FAISS'"
        ).fetchone()["entity_type"]
        assert faiss_type == "tech"

        # "data pipeline" was originally expected to fall through to
        # "unknown" because the classifier didn't recognize it. The
        # entity-type classifier has since been extended to recognize
        # workflow/system phrases like this; it now returns "project".
        # Assert membership in the set of known classifications rather
        # than exact match so future classifier upgrades don't break
        # this test (the contract is "type is stored," not "type is
        # exactly this string").
        pipeline_type = db._conn.execute(
            "SELECT entity_type FROM entities WHERE name = 'data pipeline'"
        ).fetchone()["entity_type"]
        assert pipeline_type in {"unknown", "project", "tech", "system"}, (
            f"unexpected pipeline_type: {pipeline_type!r}"
        )

    def test_link_memory_entity_idempotent(self, db):
        """Linking same entity twice should not error or create duplicates."""
        rid = db.record("test", embedding=_vec(1.0))
        db.relate("Alice", "Bob", rel_type="knows")
        db.link_memory_entity(rid, "Alice")
        db.link_memory_entity(rid, "Alice")  # duplicate

        count = db._conn.execute(
            "SELECT COUNT(*) FROM memory_entities WHERE memory_rid = ? AND entity_name = 'Alice'",
            (rid,),
        ).fetchone()["COUNT(*)"]
        assert count == 1

    def test_recall_scores_non_negative(self, db):
        """All scores should be non-negative."""
        for i in range(10):
            db.record(
                f"memory {i}",
                importance=i * 0.1,
                valence=(i - 5) * 0.2,
                embedding=_vec(float(i)),
            )

        results = db.recall(query_embedding=_vec(5.0), top_k=10, skip_reinforce=True)
        for r in results:
            assert r["score"] >= 0.0, f"score should be non-negative, got {r['score']}"
            assert r["scores"]["similarity"] >= -1.0
            assert r["scores"]["decay"] >= 0.0
            assert r["scores"]["recency"] >= 0.0

    def test_recall_top_k_respected_with_graph(self, db):
        """top_k must be respected even when graph expansion adds candidates."""
        for i in range(15):
            rid = db.record(f"memory about topic {i}", embedding=_vec(float(i)))
            entity = f"Entity{i}"
            db.relate(entity, f"Entity{(i + 1) % 15}", rel_type="related_to")
            db.link_memory_entity(rid, entity)

        results = db.recall(
            query="Entity0 topic",
            query_embedding=_vec(0.0),
            top_k=5,
            expand_entities=True,
            skip_reinforce=True,
        )
        assert len(results) <= 5

    def test_backfill_memory_entities(self, db):
        """backfill_memory_entities should link memories to entities."""
        db.relate("Alice", "Bob", rel_type="knows")
        r1 = db.record("Alice discussed the plan", embedding=_vec(1.0))
        r2 = db.record("Bob reviewed the code", embedding=_vec(2.0))

        count = db.backfill_memory_entities()
        assert count > 0

        # Check links were created (rows are dicts)
        linked = db._conn.execute(
            "SELECT entity_name FROM memory_entities WHERE memory_rid = ?", (r1,)
        ).fetchall()
        entity_names = [row["entity_name"] for row in linked]
        assert "Alice" in entity_names


class TestStorageTier:
    def test_archive_hydrate_cycle(self, db):
        """Archive a memory to cold, verify invisible to recall, hydrate back."""
        emb = _vec(1.0)
        rid = db.record("archivable", embedding=emb)
        assert db.get(rid)["storage_tier"] == "hot"

        # Archive
        assert db.archive(rid) is True
        assert db.get(rid)["storage_tier"] == "cold"
        assert db.stats()["archived_memories"] == 1

        # Should not appear in recall
        results = db.recall(query_embedding=emb, top_k=10, skip_reinforce=True)
        assert all(r["rid"] != rid for r in results)

        # Hydrate back
        assert db.hydrate(rid) is True
        assert db.get(rid)["storage_tier"] == "hot"
        assert db.stats()["archived_memories"] == 0

        # Should appear in recall again
        results = db.recall(query_embedding=emb, top_k=10, skip_reinforce=True)
        assert any(r["rid"] == rid for r in results)

    def test_evict(self, db):
        """Evict memories to keep max_active, verify stats and recall."""
        for i in range(15):
            db.record(f"evict mem {i}", embedding=_vec(float(i)))

        assert db.stats()["active_memories"] == 15
        archived = db.evict(max_active=10)
        assert len(archived) == 5
        assert db.stats()["archived_memories"] == 5

        # Archived should not appear in recall
        results = db.recall(query_embedding=_vec(0.0), top_k=20, skip_reinforce=True)
        for r in results:
            assert r["rid"] not in archived

    def test_record_batch(self, db):
        """Batch record multiple memories at once."""
        inputs = [
            {"text": f"batch {i}", "embedding": _vec(float(i))}
            for i in range(10)
        ]
        rids = db.record_batch(inputs)
        assert len(rids) == 10
        assert db.stats()["active_memories"] == 10

        # All retrievable
        for rid in rids:
            mem = db.get(rid)
            assert mem is not None
            assert mem["storage_tier"] == "hot"


# ── Namespace Tests ──────────────────────────────────────

class TestNamespace:
    def test_default_namespace(self, db):
        """Records without explicit namespace go to 'default'."""
        rid = db.record("default ns", embedding=_vec(1.0))
        mem = db.get(rid)
        assert mem["namespace"] == "default"

    def test_explicit_namespace(self, db):
        """Records with explicit namespace are stored correctly."""
        rid = db.record("agent-1 mem", embedding=_vec(1.0), namespace="agent-1")
        mem = db.get(rid)
        assert mem["namespace"] == "agent-1"

    def test_recall_isolation(self, db):
        """Recall with namespace filter only returns memories from that namespace."""
        db.record("shared fact", embedding=_vec(1.0), namespace="ns-a")
        db.record("private fact", embedding=_vec(1.1), namespace="ns-b")
        db.record("another a", embedding=_vec(1.2), namespace="ns-a")

        results_a = db.recall(query_embedding=_vec(1.0), top_k=10, namespace="ns-a", skip_reinforce=True)
        assert len(results_a) == 2
        assert all(r["namespace"] == "ns-a" for r in results_a)

        results_b = db.recall(query_embedding=_vec(1.0), top_k=10, namespace="ns-b", skip_reinforce=True)
        assert len(results_b) == 1
        assert results_b[0]["namespace"] == "ns-b"

    def test_recall_no_namespace_returns_all(self, db):
        """Recall without namespace filter returns memories from all namespaces."""
        db.record("ns-a mem", embedding=_vec(1.0), namespace="ns-a")
        db.record("ns-b mem", embedding=_vec(1.1), namespace="ns-b")
        db.record("default mem", embedding=_vec(1.2))

        results = db.recall(query_embedding=_vec(1.0), top_k=10, skip_reinforce=True)
        assert len(results) == 3
        namespaces = {r["namespace"] for r in results}
        assert namespaces == {"ns-a", "ns-b", "default"}

    def test_stats_filtered_by_namespace(self, db):
        """Stats with namespace filter only count memories in that namespace."""
        db.record("a1", embedding=_vec(1.0), namespace="ns-a")
        db.record("a2", embedding=_vec(2.0), namespace="ns-a")
        db.record("b1", embedding=_vec(3.0), namespace="ns-b")

        all_stats = db.stats()
        assert all_stats["active_memories"] == 3

        a_stats = db.stats(namespace="ns-a")
        assert a_stats["active_memories"] == 2

        b_stats = db.stats(namespace="ns-b")
        assert b_stats["active_memories"] == 1

    def test_batch_with_mixed_namespaces(self, db):
        """Batch record respects per-entry namespace."""
        inputs = [
            {"text": "batch-a", "embedding": _vec(1.0), "namespace": "ns-a"},
            {"text": "batch-b", "embedding": _vec(2.0), "namespace": "ns-b"},
            {"text": "batch-default", "embedding": _vec(3.0)},
        ]
        rids = db.record_batch(inputs)
        assert len(rids) == 3

        assert db.get(rids[0])["namespace"] == "ns-a"
        assert db.get(rids[1])["namespace"] == "ns-b"
        assert db.get(rids[2])["namespace"] == "default"

    def test_namespace_preserved_on_correct(self, db):
        """Correcting a memory preserves its namespace.

        Issue #47 (v0.7.20): correct() is now in-place. `reason` is
        required, `embedding` removed (HNSW limitation). The
        namespace-preservation contract is exercised via importance.
        """
        rid = db.record("original", embedding=_vec(1.0), namespace="my-ns")
        result = db.correct(rid, reason="test", new_importance=0.9)
        corrected = db.get(result["corrected_rid"])
        assert corrected["namespace"] == "my-ns"

    def test_correct_new_text_requires_embedder(self, db):
        """v0.10 Item 3: a text-changing correction re-embeds. With NO
        embedder configured (this fixture records explicit vectors), the
        re-embed cannot run, so it is rejected cleanly — no side effects,
        and metadata/importance/valence corrections still work. (The
        message names the embedder, not the old forget+record_text
        workaround, which v0.10 removed.)"""
        rid = db.record("alice owns service A", embedding=_vec(1.0))
        with pytest.raises(Exception, match="[Ee]mbedder"):
            db.correct(rid, reason="handover", new_text="bob owns service B")
        # No side effects: text unchanged, and non-text corrections still work.
        assert db.get(rid)["text"] == "alice owns service A"
        result = db.correct(rid, reason="handover", metadata_merge={"owner": "bob"})
        assert result["revision_num"] == 1

    def test_correct_new_text_reembeds_with_embedder(self, tmp_path):
        """v0.10 Item 3: with an embedder attached, correct(new_text=...)
        re-embeds and updates the record in place at the SAME rid — the
        durable text and the retrieval vector stay coherent (the whole
        point of Item 3). No new rid is minted."""

        class _MockEmbedder:
            def encode(self, text: str) -> list[float]:
                return _vec(float(hash(text) % 1000) / 100.0)

        db = YantrikDB(str(tmp_path / "reembed.db"), embedding_dim=DIM)
        db.set_embedder(_MockEmbedder())
        rid = db.record("alice owns service A", embedding=_vec(1.0))
        result = db.correct(rid, reason="handover", new_text="bob owns service B")
        # In-place: same rid, revision advances, text is the corrected text.
        assert result["corrected_rid"] == rid
        assert result["original_tombstoned"] is False
        assert result["revision_num"] == 1
        assert db.get(rid)["text"] == "bob owns service B"


class TestQueryBuilder:
    """Tests for the composable query() API."""

    @pytest.fixture
    def db(self, tmp_path):
        return YantrikDB(str(tmp_path / "query_test.db"), embedding_dim=DIM)

    def test_query_basic(self, db):
        for i in range(10):
            db.record(f"memory {i}", embedding=_vec(float(i)))
        results = db.query(embedding=_vec(0.0), top_k=3, skip_reinforce=True)
        assert len(results) == 3
        assert all("score" in r for r in results)

    def test_query_with_type_filter(self, db):
        db.record("ep", memory_type="episodic", embedding=_vec(1.0))
        db.record("sem", memory_type="semantic", embedding=_vec(1.1))
        results = db.query(
            embedding=_vec(1.0), top_k=10,
            memory_type="episodic", skip_reinforce=True,
        )
        assert len(results) == 1
        assert results[0]["type"] == "episodic"

    def test_query_with_namespace_filter(self, db):
        db.record("work mem", embedding=_vec(1.0), namespace="work")
        db.record("personal mem", embedding=_vec(1.1), namespace="personal")
        results = db.query(
            embedding=_vec(1.0), top_k=10,
            namespace="work", skip_reinforce=True,
        )
        assert len(results) == 1
        assert results[0]["namespace"] == "work"

    def test_query_contributions_present(self, db):
        db.record("test", embedding=_vec(1.0), importance=0.8, valence=0.5)
        results = db.query(embedding=_vec(1.0), top_k=1, skip_reinforce=True)
        assert len(results) == 1
        scores = results[0]["scores"]
        assert "contributions" in scores
        assert "valence_multiplier" in scores
        c = scores["contributions"]
        assert all(k in c for k in ["similarity", "decay", "recency", "importance", "graph_proximity"])


class TestSetEmbedderNamedWorkers:
    """Regression for issue #58.

    v0.9.0 made the pyo3 constructors spawn a background worker pool
    (materializer + compactor). Those threads hold ``Weak<YantrikDB>`` refs,
    and ``set_embedder_named`` reaches the engine via ``Arc::get_mut``, which
    requires weak count 0 — so the call ALWAYS failed with
    "set_embedder_named requires exclusive access to the engine", regardless
    of the model name. The binding now stops the workers (joining the threads,
    releasing the weak refs), swaps, then respawns the pool.

    These probes use an UNKNOWN model name so they stay hermetic (no network /
    no model download): before the fix the call could never get past the
    exclusive-access guard; after it, the engine itself rejects the unknown
    name with a *different* error. So the invariant is simply: the failure is
    never the exclusive-access one.
    """

    def _skip_if_no_download(self, db):
        """Skip on slim wheels (built --no-default-features) where the named-
        download path compiles out to a feature-absent stub."""
        try:
            db.set_embedder_named("issue-58-no-such-model")
        except Exception as exc:  # noqa: BLE001 - inspecting the message
            msg = str(exc).lower()
            if "embedder-download" in msg and "feature" in msg:
                pytest.skip("wheel built --no-default-features; named-download path absent")
            return msg
        return ""

    def test_set_embedder_named_not_blocked_by_workers(self):
        """A freshly constructed engine (workers running) can reach the swap."""
        db = YantrikDB.with_default(":memory:")
        try:
            msg = self._skip_if_no_download(db)
            assert "exclusive access" not in msg, (
                "set_embedder_named must get past the Arc::get_mut guard once the "
                f"worker pool is stopped; got: {msg!r}"
            )
        finally:
            db.close()

    def test_stop_swap_respawn_is_repeatable(self):
        """A second call behaves identically — proves the pool was respawned
        and the stop/swap/respawn cycle is repeatable, not a one-shot."""
        db = YantrikDB.with_default(":memory:")
        try:
            self._skip_if_no_download(db)  # first cycle (also handles skip)
            with pytest.raises(Exception) as exc:  # noqa: PT011 - message asserted below
                db.set_embedder_named("issue-58-still-no-such-model")
            assert "exclusive access" not in str(exc.value).lower()
            # Engine is still live after the swap+respawn cycle.
            assert db.stats()["active_memories"] == 0
        finally:
            db.close()


# ── Encryption tests ──


class TestEncryption:
    """Tests for encryption at rest."""

    def test_encrypted_record_and_get(self, tmp_path):
        key = bytes(range(32))
        db = YantrikDB(str(tmp_path / "enc.db"), embedding_dim=DIM, encryption_key=key)
        assert db.is_encrypted
        rid = db.record("secret memory", embedding=_vec(1.0), importance=0.8)
        mem = db.get(rid)
        assert mem["text"] == "secret memory"
        assert mem["importance"] == 0.8
        db.close()

    def test_encrypted_data_not_plaintext(self, tmp_path):
        key = bytes(range(32))
        db = YantrikDB(str(tmp_path / "enc2.db"), embedding_dim=DIM, encryption_key=key)
        rid = db.record("top secret", embedding=_vec(1.0))

        # Read raw from DB — text should be encrypted (not plaintext)
        row = db._conn.execute(
            "SELECT text FROM memories WHERE rid = ?", (rid,)
        ).fetchone()
        assert row["text"] != "top secret"
        db.close()

    def test_encrypted_recall(self, tmp_path):
        key = bytes(range(32))
        db = YantrikDB(str(tmp_path / "enc3.db"), embedding_dim=DIM, encryption_key=key)
        db.record("cat on mat", embedding=_vec(1.0))
        db.record("dog in park", embedding=_vec(5.0))
        results = db.recall(query_embedding=_vec(1.0), top_k=1, skip_reinforce=True)
        assert len(results) == 1
        assert "cat" in results[0]["text"]
        db.close()

    def test_encrypted_reopen_same_key(self, tmp_path):
        key = bytes(range(32))
        path = str(tmp_path / "enc4.db")
        db = YantrikDB(path, embedding_dim=DIM, encryption_key=key)
        rid = db.record("persistent", embedding=_vec(1.0))
        db.close()

        # Reopen with same key
        db2 = YantrikDB(path, embedding_dim=DIM, encryption_key=key)
        mem = db2.get(rid)
        assert mem["text"] == "persistent"
        db2.close()

    def test_encrypted_wrong_key_fails(self, tmp_path):
        key_a = bytes(range(32))
        path = str(tmp_path / "enc5.db")
        db = YantrikDB(path, embedding_dim=DIM, encryption_key=key_a)
        db.record("data", embedding=_vec(1.0))
        db.close()

        key_b = bytes([99] + list(range(1, 32)))
        with pytest.raises(RuntimeError):
            YantrikDB(path, embedding_dim=DIM, encryption_key=key_b)

    def test_open_encrypted_without_key_fails(self, tmp_path):
        key = bytes(range(32))
        path = str(tmp_path / "enc6.db")
        db = YantrikDB(path, embedding_dim=DIM, encryption_key=key)
        db.record("data", embedding=_vec(1.0))
        db.close()

        with pytest.raises(RuntimeError):
            YantrikDB(path, embedding_dim=DIM)

    def test_invalid_key_length_rejected(self):
        with pytest.raises((ValueError, RuntimeError)):
            YantrikDB(":memory:", embedding_dim=DIM, encryption_key=b"short")

    def test_unencrypted_unchanged(self, tmp_path):
        db = YantrikDB(str(tmp_path / "plain.db"), embedding_dim=DIM)
        assert not db.is_encrypted
        rid = db.record("plaintext", embedding=_vec(1.0))
        row = db._conn.execute(
            "SELECT text FROM memories WHERE rid = ?", (rid,)
        ).fetchone()
        assert row["text"] == "plaintext"
        db.close()


# ── Tenant isolation tests ──


class TestTenantIsolation:
    """Tests for multi-tenant manager."""

    def test_tenant_isolation(self, tmp_path):
        from yantrikdb import TenantManager

        mgr = TenantManager(str(tmp_path / "tenants"), embedding_dim=DIM)
        db_a = mgr.get("tenant-a")
        db_b = mgr.get("tenant-b")

        db_a.record("a-memory", embedding=_vec(1.0))
        assert db_a.stats()["active_memories"] == 1
        assert db_b.stats()["active_memories"] == 0

        db_a.close()
        db_b.close()

    def test_tenant_with_encryption(self, tmp_path):
        from yantrikdb import TenantManager

        mgr = TenantManager(str(tmp_path / "tenants"), embedding_dim=DIM)
        mgr.register_tenant("secure", encryption_key=bytes(range(32)))

        db = mgr.get("secure")
        assert db.is_encrypted
        rid = db.record("tenant secret", embedding=_vec(1.0))
        mem = db.get(rid)
        assert mem["text"] == "tenant secret"
        db.close()

    def test_discovered_tenants(self, tmp_path):
        from yantrikdb import TenantManager

        mgr = TenantManager(str(tmp_path / "tenants"), embedding_dim=DIM)
        mgr.get("alpha").close()
        mgr.get("beta").close()

        discovered = mgr.discovered_tenants()
        assert "alpha" in discovered
        assert "beta" in discovered

    def test_cross_tenant_data_isolation(self, tmp_path):
        from yantrikdb import TenantManager

        mgr = TenantManager(str(tmp_path / "tenants"), embedding_dim=DIM)

        # Tenant A stores a memory
        db_a = mgr.get("a")
        db_a.record("only for A", embedding=_vec(1.0))
        db_a.close()

        # Tenant B should not see it
        db_b = mgr.get("b")
        results = db_b.recall(query_embedding=_vec(1.0), top_k=10, skip_reinforce=True)
        assert len(results) == 0
        db_b.close()

    def test_tenant_different_encryption_keys(self, tmp_path):
        from yantrikdb import TenantManager

        mgr = TenantManager(str(tmp_path / "tenants"), embedding_dim=DIM)

        key_a = bytes([1] * 32)
        key_b = bytes([2] * 32)
        mgr.register_tenant("a", encryption_key=key_a)
        mgr.register_tenant("b", encryption_key=key_b)

        db_a = mgr.get("a")
        db_b = mgr.get("b")

        db_a.record("A's secret", embedding=_vec(1.0))
        db_b.record("B's secret", embedding=_vec(2.0))

        assert db_a.get(db_a.recall(query_embedding=_vec(1.0), top_k=1, skip_reinforce=True)[0]["rid"])["text"] == "A's secret"
        assert db_b.get(db_b.recall(query_embedding=_vec(2.0), top_k=1, skip_reinforce=True)[0]["rid"])["text"] == "B's secret"

        db_a.close()
        db_b.close()


def test_recall_as_of_rolls_back_corrections():
    """Bitemporal recall: a correction after t must not rewrite what was
    believed at t.

    Uses the BUNDLED embedder, deliberately. This previously called
    ``set_embedder_named("potion-base-8M")``, which downloads a 28 MB
    artifact from GitHub Releases — so a test of ``recall_as_of``
    semantics failed whenever CI could not complete that download. On
    GitHub runners that is often: it failed FOUR CONSECUTIVE retries on
    two different runner architectures, while succeeding at 5 MB/s from
    a developer machine. Retrying is not the remedy when the environment
    cannot complete the transfer at all.

    The download path has its own retry and its own coverage. What this
    test needs is *an* embedder, not a *particular* one — so it takes the
    one compiled into the binary and touches no network.
    """
    import time

    db = YantrikDB.with_default(":memory:")
    rid = db.record_text("The deadline is March 1st", importance=0.9)
    time.sleep(0.03)
    t_mid = time.time()
    time.sleep(0.03)
    db.correct(rid, new_text="The deadline is March 15th", reason="slip")

    now_hits = db.recall_as_of(time.time(), query="deadline", top_k=5)
    assert any(h["text"] == "The deadline is March 15th" for h in now_hits)

    then_hits = db.recall_as_of(t_mid, query="deadline", top_k=5)
    assert any(h["text"] == "The deadline is March 1st" for h in then_hits)
    rolled = [h for h in then_hits if h["text"] == "The deadline is March 1st"]
    assert any(w.startswith("as_of:") for w in rolled[0]["why_retrieved"])

    before = db.recall_as_of(t_mid - 3600, query="deadline", top_k=5)
    assert before == []


# ── token diet: best_span / snippets / min_score_ratio (0.13) ──────


class TestTokenDiet:
    LONG = (
        "filler sentence padding out the head of the record. " * 25
        + "the ZEPHYR-KEY rotates through the northern vault every solstice. "
        + "trailing filler about entirely unrelated matters. " * 15
    )

    def test_best_span_reported_for_long_records(self, db):
        db.record(self.LONG, embedding=_vec(1.0))
        db.record("short record", embedding=_vec(2.0))

        hits = db.recall(
            query="ZEPHYR-KEY northern vault solstice",
            query_embedding=_vec(1.0),
            top_k=2,
        )
        long_hit = next(h for h in hits if "ZEPHYR-KEY" in h["text"])
        span = long_hit["best_span"]
        assert span is not None, "long record must carry a span"
        a, b = span
        assert "ZEPHYR-KEY" in self.LONG[a:b], "span must cover the matched phrase"
        assert a > 0, "the phrase is not in the head window"

        short_hit = next(h for h in hits if h["text"] == "short record")
        assert short_hit["best_span"] is None

    def test_snippets_replace_text_with_matched_window(self, db):
        db.record(self.LONG, embedding=_vec(1.0))
        hits = db.recall(
            query="ZEPHYR-KEY northern vault solstice",
            query_embedding=_vec(1.0),
            top_k=1,
            snippets=True,
        )
        h = hits[0]
        assert "ZEPHYR-KEY" in h["text"], "snippet must contain the match"
        assert len(h["text"]) < len(self.LONG) // 2, "snippet must actually trim"
        assert h["text"].startswith("…"), "mid-text slice is marked"
        assert any(w.startswith("snippet ") for w in h["why_retrieved"])
        # best_span still carries ORIGINAL text coordinates.
        a, b = h["best_span"]
        assert "ZEPHYR-KEY" in self.LONG[a:b]

    def test_min_score_ratio_trims_the_tail(self, db):
        # Exactly orthogonal vectors so the score gap is deterministic
        # (a hash-spread _vec pair can land at any cosine).
        e0 = [1.0] + [0.0] * (DIM - 1)
        e1 = [0.0, 1.0] + [0.0] * (DIM - 2)
        e2 = [0.0, 0.0, 1.0] + [0.0] * (DIM - 3)
        db.record("the exact thing asked about", embedding=e0)
        db.record("unrelated one", embedding=e1)
        db.record("unrelated two", embedding=e2)

        full = db.recall(query_embedding=e0, top_k=10)
        assert len(full) == 3

        trimmed = db.recall(query_embedding=e0, top_k=10, min_score_ratio=0.8)
        assert 1 <= len(trimmed) < 3, f"cliff must trim noise, kept {len(trimmed)}"
        assert trimmed[0]["text"] == "the exact thing asked about"

        # The best result always survives its own floor, even at 1.0.
        keeps_best = db.recall(query_embedding=e0, top_k=10, min_score_ratio=1.0)
        assert keeps_best[0]["text"] == "the exact thing asked about"

    def test_min_score_ratio_validates_range(self, db):
        db.record("anything", embedding=_vec(1.0))
        with pytest.raises(ValueError):
            db.recall(query_embedding=_vec(1.0), top_k=5, min_score_ratio=1.5)
