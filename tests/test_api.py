"""Tests for the YantrikDB REST API."""

import math
import threading

import pytest
from fastapi.testclient import TestClient

from yantrikdb import YantrikDB

DIM = 8


def _vec(seed: float) -> list[float]:
    raw = [(seed + i) * 0.1 for i in range(DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


class _MockEmbedder:
    def encode(self, text: str) -> list[float]:
        seed = float(hash(text) % 1000) / 100.0
        return _vec(seed)


@pytest.fixture
def client():
    """Create a test client with in-memory YantrikDB, bypassing lifespan."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI as _FastAPI

    import yantrikdb.api as api_mod

    # Create a test lifespan that creates the DB on the event loop thread
    @asynccontextmanager
    async def test_lifespan(app):
        db = YantrikDB(db_path=":memory:", embedding_dim=DIM)
        db.set_embedder(_MockEmbedder())
        lock = threading.Lock()
        app.state.db = db
        app.state.lock = lock
        original = api_mod._db
        api_mod._db = lambda: (db, lock)
        try:
            yield
        finally:
            api_mod._db = original
            db.close()

    test_app = _FastAPI(title="YantrikDB", version="0.1.0", lifespan=test_lifespan)
    for route in api_mod.app.routes:
        test_app.routes.append(route)

    with TestClient(test_app, raise_server_exceptions=True) as c:
        yield c


class TestMemoryEndpoints:
    def test_record_memory(self, client):
        resp = client.post("/memories", json={"text": "hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert "rid" in data

    def test_get_memory(self, client):
        rid = client.post("/memories", json={"text": "test"}).json()["rid"]
        resp = client.get(f"/memories/{rid}")
        assert resp.status_code == 200
        assert resp.json()["rid"] == rid
        assert resp.json()["text"] == "test"
        assert "embedding" not in resp.json()

    def test_get_memory_not_found(self, client):
        resp = client.get("/memories/nonexistent")
        assert resp.status_code == 404

    def test_recall_memories(self, client):
        client.post("/memories", json={"text": "the sky is blue"})
        client.post("/memories", json={"text": "grass is green"})

        resp = client.post("/memories/recall", json={"query": "sky color"})
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert "results" in data

    def test_recall_empty(self, client):
        resp = client.post("/memories/recall", json={"query": "anything"})
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_forget_memory(self, client):
        rid = client.post("/memories", json={"text": "forget me"}).json()["rid"]
        resp = client.delete(f"/memories/{rid}")
        assert resp.status_code == 200
        assert resp.json()["forgotten"] is True

        # Memory is tombstoned — get() still returns it but it won't appear in recall
        resp = client.get(f"/memories/{rid}")
        assert resp.status_code == 200
        assert resp.json()["consolidation_status"] == "tombstoned"

    def test_correct_memory(self, client):
        # Issue #47 (v0.7.20): correct() is now in-place; `reason` is
        # required, `correction_note` removed, `embedding` removed.
        rid = client.post("/memories", json={"text": "wrong"}).json()["rid"]
        resp = client.post(f"/memories/{rid}/correct", json={
            "new_text": "correct",
            "reason": "fixed",
        })
        assert resp.status_code == 200
        data = resp.json()
        # In-place mutation: corrected_rid == original rid.
        assert data["corrected_rid"] == rid
        assert data["original_rid"] == rid
        assert data["original_tombstoned"] is False
        assert data["revision_num"] == 1

    def test_record_with_fields(self, client):
        resp = client.post("/memories", json={
            "text": "important fact",
            "memory_type": "semantic",
            "importance": 0.9,
            "valence": 0.5,
            "metadata": {"source": "test"},
        })
        assert resp.status_code == 200


class TestEntityEndpoints:
    def test_relate(self, client):
        resp = client.post("/entities/relate", json={
            "src": "Alice", "dst": "Bob", "rel_type": "knows",
        })
        assert resp.status_code == 200
        assert "edge_id" in resp.json()

    def test_get_edges(self, client):
        client.post("/entities/relate", json={
            "src": "Alice", "dst": "Bob",
        })
        resp = client.get("/entities/Alice/edges")
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_get_edges_empty(self, client):
        resp = client.get("/entities/Unknown/edges")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


class TestCognitionEndpoints:
    def test_think(self, client):
        resp = client.post("/think")
        assert resp.status_code == 200
        data = resp.json()
        assert "consolidation_count" in data

    def test_conflicts_empty(self, client):
        resp = client.get("/conflicts")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_triggers_empty(self, client):
        resp = client.get("/triggers")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


class TestSystemEndpoints:
    def test_stats(self, client):
        client.post("/memories", json={"text": "test"})
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_memories"] == 1

    def test_export(self, client):
        client.post("/memories", json={"text": "exportable"})
        resp = client.get("/export")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "yantrikdb-export-v1"
        assert len(data["memories"]) == 1

    def test_openapi_schema(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "YantrikDB"
        assert "/memories" in schema["paths"]
        assert "/memories/recall" in schema["paths"]
        assert "/stats" in schema["paths"]
