"""MCP tool implementations for YantrikDB cognitive memory engine."""

import json

from mcp.server.fastmcp import Context

from .server import mcp


def _get_db(ctx: Context):
    """Get the YantrikDB instance and lock from the lifespan context."""
    lc = ctx.request_context.lifespan_context
    return lc["db"], lc["lock"]


# ── Core Memory Tools ──


@mcp.tool()
def memory_record(
    text: str,
    memory_type: str = "episodic",
    importance: float = 0.5,
    valence: float = 0.0,
    metadata: dict | None = None,
    namespace: str = "default",
    certainty: float = 0.8,
    domain: str = "general",
    source: str = "user",
    emotional_state: str | None = None,
    ctx: Context = None,
) -> str:
    """Store a new memory in the cognitive memory engine.

    Args:
        text: The memory content to store.
        memory_type: One of 'episodic' (events), 'semantic' (facts), 'procedural' (how-to).
        importance: How important this memory is (0.0 to 1.0). Higher = decays slower.
        valence: Emotional tone (-1.0 negative to 1.0 positive). 0.0 is neutral.
        metadata: Optional key-value metadata (e.g. {"source": "conversation", "topic": "work"}).
        namespace: Memory namespace for isolation (default: "default").
        certainty: How confident the agent was when storing (0.0 to 1.0).
        domain: Topic domain (work, health, family, finance, hobby, travel, general, etc.).
        source: Who/what provided this memory (user, system, document, inference).
        emotional_state: Richer emotion label: joy, sadness, anger, fear, surprise, neutral, etc.

    Returns the memory ID (rid) of the stored memory.
    """
    db, lock = _get_db(ctx)
    with lock:
        rid = db.record(
            text,
            memory_type=memory_type,
            importance=importance,
            valence=valence,
            metadata=metadata or {},
            namespace=namespace,
            certainty=certainty,
            domain=domain,
            source=source,
            emotional_state=emotional_state,
        )
    return json.dumps({"rid": rid, "status": "recorded"})


@mcp.tool()
def memory_recall(
    query: str,
    top_k: int = 10,
    memory_type: str | None = None,
    include_consolidated: bool = False,
    expand_entities: bool = True,
    namespace: str | None = None,
    domain: str | None = None,
    source: str | None = None,
    ctx: Context = None,
) -> str:
    """Search memories by semantic similarity to a natural language query.

    Uses multi-signal scoring: vector similarity, temporal decay, recency,
    importance, and optional knowledge graph expansion.

    Args:
        query: Natural language search query.
        top_k: Maximum number of results (default 10).
        memory_type: Filter by type ('episodic', 'semantic', 'procedural'). None for all.
        include_consolidated: Whether to include consolidated (merged) memories.
        expand_entities: Whether to use knowledge graph to find related memories.
        namespace: Filter by namespace. None returns all namespaces.
        domain: Filter by topic domain (work, health, family, etc.). None for all.
        source: Filter by memory source (user, system, document, inference). None for all.

    Returns matching memories ranked by relevance with score breakdowns.
    """
    db, lock = _get_db(ctx)
    with lock:
        response = db.recall_with_response(
            query=query,
            top_k=top_k,
            memory_type=memory_type,
            include_consolidated=include_consolidated,
            expand_entities=expand_entities,
            namespace=namespace,
            domain=domain,
            source=source,
        )
    # Convert PyO3 dicts to plain dicts for JSON serialization
    items = []
    for r in response["results"]:
        items.append({
            "rid": r["rid"],
            "text": r["text"],
            "type": r["type"],
            "score": round(r["score"], 4),
            "importance": r["importance"],
            "created_at": r["created_at"],
            "scores": {
                "similarity": round(r["scores"]["similarity"], 4),
                "decay": round(r["scores"]["decay"], 4),
                "recency": round(r["scores"]["recency"], 4),
                "importance": round(r["scores"]["importance"], 4),
                "graph_proximity": round(r["scores"]["graph_proximity"], 4),
            },
            "why_retrieved": r["why_retrieved"],
        })
    hints = [
        {
            "hint_type": h["hint_type"],
            "suggestion": h["suggestion"],
            "related_entities": h["related_entities"],
        }
        for h in response["hints"]
    ]
    return json.dumps({
        "count": len(items),
        "results": items,
        "confidence": round(response["confidence"], 4),
        "retrieval_summary": {
            "top_similarity": round(response["retrieval_summary"]["top_similarity"], 4),
            "score_spread": round(response["retrieval_summary"]["score_spread"], 4),
            "sources_used": response["retrieval_summary"]["sources_used"],
            "candidate_count": response["retrieval_summary"]["candidate_count"],
        },
        "hints": hints,
    })


@mcp.tool()
def memory_recall_refine(
    original_query: str,
    refinement_text: str,
    original_rids: list[str] | None = None,
    top_k: int = 10,
    namespace: str | None = None,
    domain: str | None = None,
    source: str | None = None,
    ctx: Context = None,
) -> str:
    """Refine a previous recall with a follow-up query.

    Use this when recall returned low-confidence results and hints suggested
    rephrasing. Combines the original query embedding with the refinement
    embedding (weighted: 0.4 original + 0.6 refinement) and excludes
    already-seen memory IDs.

    Args:
        original_query: The original search query text.
        refinement_text: The follow-up/clarifying query text.
        original_rids: Memory IDs from the first recall to exclude from results.
        top_k: Maximum number of results (default 10).
        namespace: Filter by namespace.
        domain: Filter by domain.
        source: Filter by source.

    Returns new results with confidence and hints.
    """
    db, lock = _get_db(ctx)
    with lock:
        # Get embeddings for both queries
        original_emb = db.embed(original_query)
        response = db.recall_refine(
            original_query_embedding=original_emb,
            refinement_text=refinement_text,
            original_rids=original_rids or [],
            top_k=top_k,
            namespace=namespace,
            domain=domain,
            source=source,
        )
    items = []
    for r in response["results"]:
        items.append({
            "rid": r["rid"],
            "text": r["text"],
            "type": r["type"],
            "score": round(r["score"], 4),
            "importance": r["importance"],
            "created_at": r["created_at"],
        })
    hints = [
        {
            "hint_type": h["hint_type"],
            "suggestion": h["suggestion"],
            "related_entities": h["related_entities"],
        }
        for h in response["hints"]
    ]
    return json.dumps({
        "count": len(items),
        "results": items,
        "confidence": round(response["confidence"], 4),
        "hints": hints,
    })


@mcp.tool()
def memory_get(rid: str, ctx: Context = None) -> str:
    """Get a specific memory by its ID.

    Args:
        rid: The memory ID to retrieve.

    Returns the full memory record including text, type, importance, timestamps, and metadata.
    """
    db, lock = _get_db(ctx)
    with lock:
        mem = db.get(rid)
    if mem is None:
        return json.dumps({"error": "Memory not found", "rid": rid})
    return json.dumps({
        "rid": mem["rid"],
        "text": mem["text"],
        "type": mem["type"],
        "importance": mem["importance"],
        "valence": mem["valence"],
        "created_at": mem["created_at"],
        "last_access": mem["last_access"],
        "consolidation_status": mem["consolidation_status"],
        "storage_tier": mem["storage_tier"],
        "metadata": mem["metadata"],
        "certainty": mem["certainty"],
        "domain": mem["domain"],
        "source": mem["source"],
        "emotional_state": mem["emotional_state"],
    })


@mcp.tool()
def memory_forget(rid: str, ctx: Context = None) -> str:
    """Permanently forget (tombstone) a memory.

    Args:
        rid: The memory ID to forget.

    Returns whether the memory was found and forgotten.
    """
    db, lock = _get_db(ctx)
    with lock:
        forgotten = db.forget(rid)
    return json.dumps({"rid": rid, "forgotten": forgotten})


@mcp.tool()
def memory_correct(
    rid: str,
    reason: str,
    new_text: str | None = None,
    metadata_merge: dict | None = None,
    new_importance: float | None = None,
    new_valence: float | None = None,
    ctx: Context = None,
) -> str:
    """Correct an existing memory in place (Issue #47, v0.7.20).

    Preserves the original `rid` + `created_at` and appends a revision row
    to the queryable history. Inbound graph edges and any other
    references to this rid continue to resolve. `reason` is REQUIRED and
    must be non-empty — the audit trail is load-bearing.

    Embedding changes are NOT supported by `correct()` (HNSW limitation).
    To change the embedding, use forget+record.

    Args:
        rid: The memory ID to correct.
        reason: Required non-empty string explaining why the correction
                was made.
        new_text: Optional new text. Pass None to keep the existing text.
        metadata_merge: Optional dict patch-merged into existing metadata.
        new_importance: Optional new importance score (0.0 to 1.0).
        new_valence: Optional new emotional valence (-1.0 to 1.0).

    Returns: dict with `original_rid` (== rid), `corrected_rid` (== rid),
    `original_tombstoned` (always False under the new semantics), and
    `revision_num` (1-indexed, monotonically increasing per-rid).
    """
    db, lock = _get_db(ctx)
    with lock:
        result = db.correct(
            rid,
            reason,
            new_text=new_text,
            metadata_merge=metadata_merge,
            new_importance=new_importance,
            new_valence=new_valence,
        )
    return json.dumps({
        "original_rid": result["original_rid"],
        "corrected_rid": result["corrected_rid"],
        "original_tombstoned": result["original_tombstoned"],
        "revision_num": result["revision_num"],
    })


# ── Entity / Graph Tools ──


@mcp.tool()
def entity_relate(
    source: str,
    target: str,
    relationship: str = "related_to",
    weight: float = 1.0,
    ctx: Context = None,
) -> str:
    """Create a relationship between two entities in the knowledge graph.

    Args:
        source: Source entity name (e.g. "Alice", "Python", "project_x").
        target: Target entity name.
        relationship: Relationship type (e.g. "works_at", "knows", "likes", "related_to").
        weight: Relationship strength (0.0 to 1.0).

    Returns the edge ID of the created relationship.
    """
    db, lock = _get_db(ctx)
    with lock:
        edge_id = db.relate(source, target, relationship, weight)
    return json.dumps({"edge_id": edge_id, "source": source, "target": target, "relationship": relationship})


@mcp.tool()
def entity_edges(entity: str, ctx: Context = None) -> str:
    """Get all relationships for an entity in the knowledge graph.

    Args:
        entity: The entity name to look up.

    Returns all edges (relationships) connected to this entity.
    """
    db, lock = _get_db(ctx)
    with lock:
        edges = db.get_edges(entity)
    items = [
        {
            "edge_id": e["edge_id"],
            "src": e["src"],
            "dst": e["dst"],
            "rel_type": e["rel_type"],
            "weight": e["weight"],
        }
        for e in edges
    ]
    return json.dumps({"entity": entity, "count": len(items), "edges": items})


# ── Cognition Tools ──


@mcp.tool()
def memory_think(
    run_consolidation: bool = True,
    run_conflict_scan: bool = True,
    run_pattern_mining: bool = True,
    ctx: Context = None,
) -> str:
    """Run the cognitive maintenance loop on the memory store.

    This performs background processing that keeps the memory system healthy:
    - Checks for decaying memories that need review
    - Consolidates similar memories into summaries
    - Scans for contradictions between memories
    - Mines for recurring patterns across memories

    Call this periodically or when you want the memory system to 'reflect'.

    Args:
        run_consolidation: Whether to merge similar memories (default True).
        run_conflict_scan: Whether to scan for contradictions (default True).
        run_pattern_mining: Whether to detect patterns (default True).

    Returns a summary of what the cognition loop found and did.
    """
    db, lock = _get_db(ctx)
    config = {
        "run_consolidation": run_consolidation,
        "run_conflict_scan": run_conflict_scan,
        "run_pattern_mining": run_pattern_mining,
    }
    with lock:
        result = db.think(config)
    triggers = []
    for t in result["triggers"]:
        triggers.append({
            "trigger_type": t["trigger_type"],
            "reason": t["reason"],
            "urgency": t["urgency"],
            "suggested_action": t["suggested_action"],
        })
    return json.dumps({
        "triggers": triggers,
        "consolidation_count": result["consolidation_count"],
        "conflicts_found": result["conflicts_found"],
        "patterns_new": result["patterns_new"],
        "patterns_updated": result["patterns_updated"],
        "expired_triggers": result["expired_triggers"],
        "duration_ms": round(result["duration_ms"], 2),
    })


# ── Conflict Tools ──


@mcp.tool()
def conflict_list(
    status: str | None = None,
    limit: int = 10,
    ctx: Context = None,
) -> str:
    """List memory conflicts (contradictions) that need resolution.

    Conflicts are detected when the memory system finds contradictory information,
    such as two memories stating different facts about the same entity.

    Args:
        status: Filter by status ('open', 'resolved', 'dismissed'). Default shows all.
        limit: Maximum number of conflicts to return (default 10).

    Returns conflicts with their IDs, types, priorities, and the conflicting memories.
    """
    db, lock = _get_db(ctx)
    with lock:
        conflicts = db.get_conflicts(status=status, limit=limit)
    items = [
        {
            "conflict_id": c["conflict_id"],
            "conflict_type": c["conflict_type"],
            "priority": c["priority"],
            "status": c["status"],
            "memory_a": c["memory_a"],
            "memory_b": c["memory_b"],
            "entity": c["entity"],
            "detection_reason": c["detection_reason"],
        }
        for c in conflicts
    ]
    return json.dumps({"count": len(items), "conflicts": items})


@mcp.tool()
def conflict_resolve(
    conflict_id: str,
    strategy: str,
    winner_rid: str | None = None,
    new_text: str | None = None,
    resolution_note: str | None = None,
    ctx: Context = None,
) -> str:
    """Resolve a memory conflict using a strategy.

    Args:
        conflict_id: The conflict ID to resolve.
        strategy: One of 'keep_a' (keep memory A), 'keep_b' (keep memory B),
                  'keep_both' (keep both, mark as non-conflicting), 'merge' (combine into one).
        winner_rid: For keep_a/keep_b, which memory wins (optional, inferred from strategy).
        new_text: For 'merge' strategy, the merged text content.
        resolution_note: Optional note explaining the resolution.

    Returns the resolution result.
    """
    db, lock = _get_db(ctx)
    with lock:
        result = db.resolve_conflict(
            conflict_id,
            strategy,
            winner_rid=winner_rid,
            new_text=new_text,
            resolution_note=resolution_note,
        )
    return json.dumps({
        "conflict_id": result["conflict_id"],
        "strategy": result["strategy"],
        "winner_rid": result.get("winner_rid"),
        "loser_tombstoned": result.get("loser_tombstoned", False),
        "new_memory_rid": result.get("new_memory_rid"),
    })


# ── Feedback / Learning Tools ──


@mcp.tool()
def memory_recall_feedback(
    rid: str,
    feedback: str,
    query_text: str | None = None,
    score_at_retrieval: float | None = None,
    rank_at_retrieval: int | None = None,
    ctx: Context = None,
) -> str:
    """Provide feedback on a recall result for adaptive learning.

    Call this when you know a returned memory was relevant or irrelevant.
    After enough feedback (20+), the engine learns optimal scoring weights.

    Args:
        rid: The memory ID to provide feedback on.
        feedback: 'relevant' or 'irrelevant'.
        query_text: The original query text (optional, helps learning).
        score_at_retrieval: The score the memory received (optional).
        rank_at_retrieval: The rank position (optional).

    Returns confirmation of recorded feedback.
    """
    db, lock = _get_db(ctx)
    with lock:
        db.recall_feedback(
            rid=rid,
            feedback=feedback,
            query_text=query_text,
            score_at_retrieval=score_at_retrieval,
            rank_at_retrieval=rank_at_retrieval,
        )
    return json.dumps({"rid": rid, "feedback": feedback, "status": "recorded"})


# ── Trigger Tools ──


@mcp.tool()
def trigger_list(limit: int = 10, ctx: Context = None) -> str:
    """Get pending proactive triggers from the memory system.

    Triggers are generated by memory_think() and represent insights, warnings,
    or suggestions such as: decaying important memories, consolidation opportunities,
    detected patterns, or relationship insights.

    Args:
        limit: Maximum number of triggers to return (default 10).

    Returns pending triggers sorted by urgency.
    """
    db, lock = _get_db(ctx)
    with lock:
        triggers = db.get_pending_triggers(limit=limit)
    items = [
        {
            "trigger_id": t["trigger_id"],
            "trigger_type": t["trigger_type"],
            "urgency": t["urgency"],
            "reason": t["reason"],
            "suggested_action": t["suggested_action"],
            "source_rids": t["source_rids"],
        }
        for t in triggers
    ]
    return json.dumps({"count": len(items), "triggers": items})


# ── Stats ──


@mcp.tool()
def memory_stats(namespace: str | None = None, ctx: Context = None) -> str:
    """Get current memory engine statistics.

    Args:
        namespace: Filter stats to a specific namespace. None for global stats.

    Returns counts of active, consolidated, tombstoned, and archived memories,
    entity and edge counts, open conflicts, pending triggers, active patterns,
    and internal index sizes.
    """
    db, lock = _get_db(ctx)
    with lock:
        stats = db.stats(namespace=namespace)
    return json.dumps(stats)
