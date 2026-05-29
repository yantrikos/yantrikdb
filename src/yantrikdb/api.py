"""YantrikDB REST API — FastAPI server with OpenAPI docs."""

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("yantrikdb.api")


# ── Pydantic Models ──


class RecordRequest(BaseModel):
    text: str
    memory_type: str = "episodic"
    importance: float = 0.5
    valence: float = 0.0
    metadata: dict | None = None


class RecordResponse(BaseModel):
    rid: str


class RecallRequest(BaseModel):
    query: str
    top_k: int = 10
    memory_type: str | None = None
    include_consolidated: bool = False
    expand_entities: bool = True


class CorrectRequest(BaseModel):
    # Issue #47 (v0.7.20): correct() is now in-place mutation with an
    # audit trail. `reason` is REQUIRED. `new_text` is optional (pass None
    # to keep). `metadata_merge` is a new optional patch dict. `embedding`
    # / `correction_note` were removed — embedding changes still go
    # through forget+record; correction_note is replaced by `reason`.
    reason: str
    new_text: str | None = None
    metadata_merge: dict | None = None
    new_importance: float | None = None
    new_valence: float | None = None


class RelateRequest(BaseModel):
    src: str
    dst: str
    rel_type: str = "related_to"
    weight: float = 1.0


class RelateResponse(BaseModel):
    edge_id: str


class ThinkRequest(BaseModel):
    run_consolidation: bool = True
    run_conflict_scan: bool = True
    run_pattern_mining: bool = True


class ResolveRequest(BaseModel):
    strategy: str
    winner_rid: str | None = None
    new_text: str | None = None
    resolution_note: str | None = None


# ── App Setup ──


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize YantrikDB + embedder on startup."""
    from sentence_transformers import SentenceTransformer

    from yantrikdb import YantrikDB

    db_path = os.environ.get("YANTRIKDB_DB_PATH", str(Path.home() / ".yantrikdb" / "memory.db"))
    model_name = os.environ.get("YANTRIKDB_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dim = int(os.environ.get("YANTRIKDB_EMBEDDING_DIM", "384"))

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading embedding model: %s", model_name)
    embedder = SentenceTransformer(model_name)

    log.info("Opening YantrikDB at: %s (dim=%d)", db_path, embedding_dim)
    db = YantrikDB(db_path=db_path, embedding_dim=embedding_dim, embedder=embedder)
    lock = threading.Lock()

    app.state.db = db
    app.state.lock = lock

    try:
        yield
    finally:
        log.info("Shutting down YantrikDB")
        db.close()


app = FastAPI(
    title="YantrikDB",
    description="A Cognitive Memory Engine for Persistent AI Systems",
    version="0.1.0",
    lifespan=lifespan,
)


def _db():
    return app.state.db, app.state.lock


# ── Memory Endpoints ──


@app.post("/memories", response_model=RecordResponse, tags=["memories"])
async def record_memory(req: RecordRequest):
    """Store a new memory."""
    db, lock = _db()
    with lock:
        rid = db.record(
            req.text,
            memory_type=req.memory_type,
            importance=req.importance,
            valence=req.valence,
            metadata=req.metadata or {},
        )
    return RecordResponse(rid=rid)


@app.post("/memories/recall", tags=["memories"])
async def recall_memories(req: RecallRequest):
    """Search memories by semantic similarity."""
    db, lock = _db()
    with lock:
        results = db.recall(
            query=req.query,
            top_k=req.top_k,
            memory_type=req.memory_type,
            include_consolidated=req.include_consolidated,
            expand_entities=req.expand_entities,
            skip_reinforce=True,
        )
    # Strip embeddings from response
    for r in results:
        r.pop("embedding", None)
    return {"count": len(results), "results": results}


@app.get("/memories/{rid}", tags=["memories"])
async def get_memory(rid: str):
    """Get a single memory by RID."""
    db, lock = _db()
    with lock:
        mem = db.get(rid)
    if mem is None:
        raise HTTPException(status_code=404, detail=f"Memory {rid} not found")
    mem.pop("embedding", None)
    return mem


@app.delete("/memories/{rid}", tags=["memories"])
async def forget_memory(rid: str):
    """Tombstone a memory (soft delete)."""
    db, lock = _db()
    with lock:
        found = db.forget(rid)
    if not found:
        raise HTTPException(status_code=404, detail=f"Memory {rid} not found")
    return {"rid": rid, "forgotten": True}


@app.post("/memories/{rid}/correct", tags=["memories"])
async def correct_memory(rid: str, req: CorrectRequest):
    """Correct a memory in place — preserves rid + created_at, appends
    revision history. Issue #47 (v0.7.20)."""
    db, lock = _db()
    with lock:
        result = db.correct(
            rid,
            req.reason,
            new_text=req.new_text,
            metadata_merge=req.metadata_merge,
            new_importance=req.new_importance,
            new_valence=req.new_valence,
        )
    return result


# ── Entity / Graph Endpoints ──


@app.post("/entities/relate", response_model=RelateResponse, tags=["entities"])
async def relate_entities(req: RelateRequest):
    """Create or update an entity relationship."""
    db, lock = _db()
    with lock:
        edge_id = db.relate(req.src, req.dst, rel_type=req.rel_type, weight=req.weight)
    return RelateResponse(edge_id=edge_id)


@app.get("/entities/{entity}/edges", tags=["entities"])
async def get_edges(entity: str):
    """Get all relationships for an entity."""
    db, lock = _db()
    with lock:
        edges = db.get_edges(entity)
    return {"entity": entity, "count": len(edges), "edges": edges}


# ── Cognition Endpoints ──


@app.post("/think", tags=["cognition"])
async def think(req: ThinkRequest = ThinkRequest()):
    """Run the cognition loop."""
    db, lock = _db()
    config = {
        "run_consolidation": req.run_consolidation,
        "run_conflict_scan": req.run_conflict_scan,
        "run_pattern_mining": req.run_pattern_mining,
    }
    with lock:
        result = db.think(config=config)
    return result


@app.get("/conflicts", tags=["cognition"])
async def list_conflicts(status: str | None = None, limit: int = 50):
    """List memory conflicts."""
    db, lock = _db()
    with lock:
        conflicts = db.get_conflicts(status=status, limit=limit)
    return {"count": len(conflicts), "conflicts": conflicts}


@app.post("/conflicts/{conflict_id}/resolve", tags=["cognition"])
async def resolve_conflict(conflict_id: str, req: ResolveRequest):
    """Resolve a memory conflict."""
    db, lock = _db()
    with lock:
        result = db.resolve_conflict(
            conflict_id, req.strategy,
            winner_rid=req.winner_rid,
            new_text=req.new_text,
            resolution_note=req.resolution_note,
        )
    return result


@app.get("/triggers", tags=["cognition"])
async def list_triggers(limit: int = 10):
    """Get pending triggers."""
    db, lock = _db()
    with lock:
        triggers = db.get_pending_triggers(limit=limit)
    return {"count": len(triggers), "triggers": triggers}


# ── Stats ──


@app.get("/stats", tags=["system"])
async def get_stats():
    """Get engine statistics."""
    db, lock = _db()
    with lock:
        return db.stats()


# ── Export/Import ──


@app.get("/export", tags=["system"])
async def export_db(include_embeddings: bool = False):
    """Export the entire database as JSON."""
    from yantrikdb.cli import _export_db
    db, lock = _db()
    with lock:
        return _export_db(db, include_embeddings=include_embeddings)


# ── Entry Point ──


def main():
    """Entry point for the yantrikdb-server console script."""
    host = os.environ.get("YANTRIKDB_HOST", "127.0.0.1")
    port = int(os.environ.get("YANTRIKDB_PORT", "8321"))
    uvicorn.run(app, host=host, port=port, log_level="info")
