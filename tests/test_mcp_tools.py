"""Tests for YantrikDB MCP tool functions.

Tests the tool logic directly with an in-memory YantrikDB, bypassing MCP transport.

Requires the optional `mcp` extra: `pip install yantrikdb[mcp]`. Without it,
the whole file is skipped (rather than ImportError-failing) so users running
the test suite without the MCP integration installed get a clean skip.
"""

import json
import math
import threading

import pytest

# Skip the entire module if mcp isn't installed (optional extra).
pytest.importorskip("mcp")

from yantrikdb import YantrikDB

DIM = 8


def _vec(seed: float) -> list[float]:
    """Generate a deterministic unit-norm vector from a seed."""
    raw = [(seed + i) * 0.1 for i in range(DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class _MockEmbedder:
    """Mock embedder with .encode() method matching SentenceTransformer API."""

    def encode(self, text: str) -> list[float]:
        seed = float(hash(text) % 1000) / 100.0
        return _vec(seed)


class MockLifespanContext:
    """Mock for ctx.request_context.lifespan_context."""

    def __init__(self, db):
        self.lifespan_context = {"db": db, "lock": threading.Lock()}


class MockRequestContext:
    """Wraps the lifespan context to match ctx.request_context pattern."""

    def __init__(self, db):
        self._lc = MockLifespanContext(db)

    @property
    def lifespan_context(self):
        return self._lc.lifespan_context


class MockContext:
    """Minimal mock of mcp.server.fastmcp.Context for direct tool testing."""

    def __init__(self, db):
        self._request_context = MockRequestContext(db)

    @property
    def request_context(self):
        return self._request_context


@pytest.fixture
def db():
    """In-memory YantrikDB with mock embedder for tool tests."""
    engine = YantrikDB(db_path=":memory:", embedding_dim=DIM)
    engine.set_embedder(_MockEmbedder())
    yield engine
    engine.close()


@pytest.fixture
def ctx(db):
    """Mock MCP context backed by the in-memory YantrikDB."""
    return MockContext(db)


# ── memory_record ──


class TestMemoryRecord:
    def test_record_returns_rid(self, db, ctx):
        from yantrikdb.mcp.tools import memory_record

        result = json.loads(memory_record(text="hello world", ctx=ctx))
        assert "rid" in result
        assert result["status"] == "recorded"
        assert len(result["rid"]) == 36

    def test_record_with_all_fields(self, db, ctx):
        from yantrikdb.mcp.tools import memory_record

        result = json.loads(memory_record(
            text="important fact",
            memory_type="semantic",
            importance=0.9,
            valence=0.5,
            metadata={"source": "test"},
            ctx=ctx,
        ))
        assert result["status"] == "recorded"

        # Verify stored correctly
        mem = db.get(result["rid"])
        assert mem["type"] == "semantic"
        assert mem["importance"] == 0.9


# ── memory_recall ──


class TestMemoryRecall:
    def test_recall_returns_results(self, db, ctx):
        from yantrikdb.mcp.tools import memory_recall, memory_record

        # Record some memories with pre-computed embeddings
        db.record("cats are fluffy", embedding=_vec(1.0))
        db.record("dogs are loyal", embedding=_vec(2.0))
        db.record("fish swim in water", embedding=_vec(3.0))

        result = json.loads(memory_recall(query="fluffy animals", ctx=ctx))
        assert result["count"] > 0
        assert len(result["results"]) > 0
        assert "rid" in result["results"][0]
        assert "score" in result["results"][0]
        assert "why_retrieved" in result["results"][0]

    def test_recall_empty_db(self, db, ctx):
        from yantrikdb.mcp.tools import memory_recall

        result = json.loads(memory_recall(query="anything", ctx=ctx))
        assert result["count"] == 0

    def test_recall_with_type_filter(self, db, ctx):
        from yantrikdb.mcp.tools import memory_recall

        db.record("event happened", memory_type="episodic", embedding=_vec(1.0))
        db.record("fact about world", memory_type="semantic", embedding=_vec(2.0))

        result = json.loads(memory_recall(
            query="anything",
            memory_type="semantic",
            ctx=ctx,
        ))
        for r in result["results"]:
            assert r["type"] == "semantic"


# ── memory_get ──


class TestMemoryGet:
    def test_get_existing(self, db, ctx):
        from yantrikdb.mcp.tools import memory_get

        rid = db.record("test", embedding=_vec(1.0))
        result = json.loads(memory_get(rid=rid, ctx=ctx))
        assert result["rid"] == rid
        assert result["text"] == "test"

    def test_get_nonexistent(self, db, ctx):
        from yantrikdb.mcp.tools import memory_get

        result = json.loads(memory_get(rid="nonexistent", ctx=ctx))
        assert "error" in result


# ── memory_forget ──


class TestMemoryForget:
    def test_forget_existing(self, db, ctx):
        from yantrikdb.mcp.tools import memory_forget

        rid = db.record("to forget", embedding=_vec(1.0))
        result = json.loads(memory_forget(rid=rid, ctx=ctx))
        assert result["forgotten"] is True

        # Verify actually forgotten
        mem = db.get(rid)
        assert mem["consolidation_status"] == "tombstoned"

    def test_forget_nonexistent(self, db, ctx):
        from yantrikdb.mcp.tools import memory_forget

        result = json.loads(memory_forget(rid="nonexistent", ctx=ctx))
        assert result["forgotten"] is False


# ── memory_correct ──


class TestMemoryCorrect:
    def test_correct_mutates_in_place(self, db, ctx):
        """Issue #47 (v0.7.20): correct() is now in-place. `reason` is
        required, `correction_note` removed. rid is preserved, original
        is NOT tombstoned, and the revision is recorded with revision_num=1.
        """
        from yantrikdb.mcp.tools import memory_correct

        rid = db.record("wrong fact", embedding=_vec(1.0))
        result = json.loads(memory_correct(
            rid=rid,
            reason="fixed typo",
            new_text="correct fact",
            ctx=ctx,
        ))
        # In-place: rid is preserved, original is NOT tombstoned.
        assert result["original_rid"] == rid
        assert result["corrected_rid"] == rid
        assert result["original_tombstoned"] is False
        assert result["revision_num"] == 1

        # Memory still exists at same rid, with updated text.
        updated = db.get(rid)
        assert updated["consolidation_status"] != "tombstoned"
        assert updated["text"] == "correct fact"


# ── entity_relate ──


class TestEntityRelate:
    def test_relate_returns_edge_id(self, db, ctx):
        from yantrikdb.mcp.tools import entity_relate

        result = json.loads(entity_relate(
            source="Alice",
            target="Bob",
            relationship="knows",
            ctx=ctx,
        ))
        assert "edge_id" in result
        assert result["source"] == "Alice"
        assert result["target"] == "Bob"


# ── entity_edges ──


class TestEntityEdges:
    def test_get_edges(self, db, ctx):
        from yantrikdb.mcp.tools import entity_edges, entity_relate

        entity_relate(source="Alice", target="Bob", relationship="knows", ctx=ctx)
        entity_relate(source="Alice", target="Carol", relationship="works_with", ctx=ctx)

        result = json.loads(entity_edges(entity="Alice", ctx=ctx))
        assert result["count"] == 2
        assert len(result["edges"]) == 2

    def test_get_edges_empty(self, db, ctx):
        from yantrikdb.mcp.tools import entity_edges

        result = json.loads(entity_edges(entity="Nobody", ctx=ctx))
        assert result["count"] == 0


# ── memory_think ──


class TestMemoryThink:
    def test_think_runs_without_error(self, db, ctx):
        from yantrikdb.mcp.tools import memory_think

        result = json.loads(memory_think(ctx=ctx))
        assert "triggers" in result
        assert "consolidation_count" in result
        assert "duration_ms" in result


# ── conflict_list ──


class TestConflictList:
    def test_conflict_list_empty(self, db, ctx):
        from yantrikdb.mcp.tools import conflict_list

        result = json.loads(conflict_list(ctx=ctx))
        assert result["count"] == 0


# ── trigger_list ──


class TestTriggerList:
    def test_trigger_list_empty(self, db, ctx):
        from yantrikdb.mcp.tools import trigger_list

        result = json.loads(trigger_list(ctx=ctx))
        assert result["count"] == 0


# ── memory_stats ──


class TestMemoryStats:
    def test_stats_returns_expected_fields(self, db, ctx):
        from yantrikdb.mcp.tools import memory_stats

        db.record("test", embedding=_vec(1.0))
        result = json.loads(memory_stats(ctx=ctx))
        assert result["active_memories"] == 1
        assert "edges" in result
        assert "entities" in result
        assert "scoring_cache_entries" in result
        assert "vec_index_entries" in result
