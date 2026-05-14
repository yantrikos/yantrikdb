"""YantrikDB CLI — inspect, query, and manage the cognitive memory engine."""

import json
import sys
from pathlib import Path

import click

# Shared options
_db_option = click.option(
    "--db", "db_path",
    envvar="YANTRIKDB_DB_PATH",
    default=str(Path.home() / ".yantrikdb" / "memory.db"),
    show_envvar=True,
    help="Path to the YantrikDB database file.",
)
_dim_option = click.option(
    "--dim", "embedding_dim",
    envvar="YANTRIKDB_EMBEDDING_DIM",
    default=384, type=int, show_envvar=True,
    help="Embedding dimension.",
)


def _open_db(db_path: str, embedding_dim: int):
    """Open YantrikDB without an embedder (read-only operations)."""
    from yantrikdb import YantrikDB
    return YantrikDB(db_path=db_path, embedding_dim=embedding_dim)


def _open_db_with_embedder(db_path: str, embedding_dim: int, model_name: str):
    """Open YantrikDB with a SentenceTransformer embedder."""
    from sentence_transformers import SentenceTransformer

    from yantrikdb import YantrikDB
    embedder = SentenceTransformer(model_name)
    return YantrikDB(db_path=db_path, embedding_dim=embedding_dim, embedder=embedder)


@click.group()
@click.version_option(version="0.1.0", prog_name="yantrikdb")
def cli():
    """YantrikDB — A Cognitive Memory Engine for Persistent AI Systems."""


# ── Stats ──


@cli.command()
@_db_option
@_dim_option
def stats(db_path, embedding_dim):
    """Show engine statistics."""
    db = _open_db(db_path, embedding_dim)
    try:
        s = db.stats()
        click.echo(f"Active memories:       {s['active_memories']}")
        click.echo(f"Consolidated memories: {s['consolidated_memories']}")
        click.echo(f"Tombstoned memories:   {s['tombstoned_memories']}")
        click.echo(f"Archived memories:     {s['archived_memories']}")
        click.echo(f"Entities:              {s['entities']}")
        click.echo(f"Edges:                 {s['edges']}")
        click.echo(f"Operations:            {s['operations']}")
        click.echo(f"Open conflicts:        {s['open_conflicts']}")
        click.echo(f"Active patterns:       {s['active_patterns']}")
        click.echo(f"Pending triggers:      {s['pending_triggers']}")
        click.echo(f"Scoring cache:         {s['scoring_cache_entries']}")
        click.echo(f"Vec index entries:     {s['vec_index_entries']}")
    finally:
        db.close()


# ── Inspect ──


@cli.command()
@_db_option
@_dim_option
@click.argument("rid")
def inspect(db_path, embedding_dim, rid):
    """Inspect a single memory by its RID."""
    db = _open_db(db_path, embedding_dim)
    try:
        mem = db.get(rid)
        if mem is None:
            click.echo(f"Memory {rid} not found.", err=True)
            sys.exit(1)
        # Remove embedding from display (too noisy)
        mem.pop("embedding", None)
        click.echo(json.dumps(mem, indent=2, default=str))
    finally:
        db.close()


# ── Recall ──


@cli.command()
@_db_option
@_dim_option
@click.option("--model", "model_name", envvar="YANTRIKDB_EMBEDDING_MODEL",
              default="all-MiniLM-L6-v2", help="Embedding model for query.")
@click.option("-k", "--top-k", default=10, type=int, help="Number of results.")
@click.option("--type", "memory_type", default=None, help="Filter by memory type.")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON.")
@click.argument("query")
def recall(db_path, embedding_dim, model_name, top_k, memory_type, as_json, query):
    """Search memories by semantic similarity."""
    db = _open_db_with_embedder(db_path, embedding_dim, model_name)
    try:
        results = db.recall(
            query=query, top_k=top_k, memory_type=memory_type,
            skip_reinforce=True,
        )
        if as_json:
            for r in results:
                r.pop("embedding", None)
            click.echo(json.dumps(results, indent=2, default=str))
        else:
            if not results:
                click.echo("No results.")
                return
            for i, r in enumerate(results, 1):
                score = r.get("scores", {}).get("final", 0)
                click.echo(f"\n[{i}] {r['rid']}  score={score:.4f}")
                click.echo(f"    type={r.get('memory_type', '?')}  importance={r.get('importance', 0):.2f}")
                text = r.get("text", "")
                if len(text) > 200:
                    text = text[:200] + "..."
                click.echo(f"    {text}")
    finally:
        db.close()


# ── Think ──


@cli.command()
@_db_option
@_dim_option
def think(db_path, embedding_dim):
    """Run the cognition loop (consolidation, conflict scan, pattern mining)."""
    db = _open_db(db_path, embedding_dim)
    try:
        result = db.think()
        click.echo(f"Consolidations:    {result['consolidation_count']}")
        click.echo(f"Conflicts found:   {result['conflicts_found']}")
        click.echo(f"New patterns:      {result['patterns_new']}")
        click.echo(f"Updated patterns:  {result['patterns_updated']}")
        click.echo(f"Expired triggers:  {result['expired_triggers']}")
        click.echo(f"Duration:          {result['duration_ms']:.1f}ms")
        triggers = result.get("triggers", [])
        if triggers:
            click.echo(f"\nTriggers ({len(triggers)}):")
            for t in triggers:
                click.echo(f"  [{t['urgency']:.2f}] {t['trigger_type']}: {t['reason']}")
    finally:
        db.close()


# ── Export ──


@cli.command()
@_db_option
@_dim_option
@click.option("-o", "--output", "output_path", default="-",
              help="Output file path (default: stdout).")
@click.option("--no-embeddings", is_flag=True,
              help="Exclude embedding vectors to reduce size.")
def export(db_path, embedding_dim, output_path, no_embeddings):
    """Export the entire database as JSON."""
    db = _open_db(db_path, embedding_dim)
    try:
        data = _export_db(db, include_embeddings=not no_embeddings)
        payload = json.dumps(data, indent=2, default=str)
        if output_path == "-":
            click.echo(payload)
        else:
            Path(output_path).write_text(payload)
            click.echo(f"Exported to {output_path} ({len(payload)} bytes)", err=True)
    finally:
        db.close()


def _export_db(db, include_embeddings: bool = True) -> dict:
    """Export all data from YantrikDB as a dict."""
    conn = db._conn

    # Memories (column names match schema.rs)
    rows = conn.execute(
        "SELECT rid, type, text, importance, valence, half_life, "
        "consolidation_status, storage_tier, metadata, "
        "created_at, updated_at, last_access "
        "FROM memories WHERE consolidation_status != 'tombstoned'"
    ).fetchall()
    memories = []
    for r in rows:
        m = {
            "rid": r["rid"], "type": r["type"], "text": r["text"],
            "importance": r["importance"], "valence": r["valence"],
            "half_life": r["half_life"],
            "consolidation_status": r["consolidation_status"],
            "storage_tier": r["storage_tier"], "metadata": r["metadata"],
            "created_at": r["created_at"], "updated_at": r["updated_at"],
            "last_access": r["last_access"],
        }
        memories.append(m)

    # If embeddings requested, fetch them via YantrikDB.get() for proper deserialization
    if include_embeddings:
        for m in memories:
            full = db.get(m["rid"])
            if full and "embedding" in full:
                m["embedding"] = full["embedding"]

    # Entities
    entities = [
        {"name": r["name"], "entity_type": r["entity_type"],
         "first_seen": r["first_seen"], "last_seen": r["last_seen"],
         "mention_count": r["mention_count"]}
        for r in conn.execute(
            "SELECT name, entity_type, first_seen, last_seen, mention_count "
            "FROM entities"
        ).fetchall()
    ]

    # Edges (exclude tombstoned)
    edges = [
        {"edge_id": r["edge_id"], "src": r["src"], "dst": r["dst"],
         "rel_type": r["rel_type"], "weight": r["weight"],
         "created_at": r["created_at"]}
        for r in conn.execute(
            "SELECT edge_id, src, dst, rel_type, weight, created_at "
            "FROM edges WHERE tombstoned = 0"
        ).fetchall()
    ]

    # Memory-entity links
    memory_entities = [
        {"memory_rid": r["memory_rid"], "entity_name": r["entity_name"]}
        for r in conn.execute(
            "SELECT memory_rid, entity_name FROM memory_entities"
        ).fetchall()
    ]

    # Conflicts
    conflicts = [
        {
            "conflict_id": r["conflict_id"], "conflict_type": r["conflict_type"],
            "priority": r["priority"], "status": r["status"],
            "memory_a": r["memory_a"], "memory_b": r["memory_b"],
            "entity": r["entity"], "detection_reason": r["detection_reason"],
            "strategy": r["strategy"], "resolution_note": r["resolution_note"],
            "winner_rid": r["winner_rid"],
            "detected_at": r["detected_at"], "resolved_at": r["resolved_at"],
        }
        for r in conn.execute(
            "SELECT conflict_id, conflict_type, priority, status, "
            "memory_a, memory_b, entity, detection_reason, "
            "strategy, resolution_note, winner_rid, detected_at, resolved_at "
            "FROM conflicts"
        ).fetchall()
    ]

    # Patterns
    patterns = db.get_patterns()

    # Triggers
    triggers = db.get_pending_triggers(limit=1000)

    s = db.stats()
    return {
        "version": "yantrikdb-export-v1",
        "stats": s,
        "memories": memories,
        "entities": entities,
        "edges": edges,
        "memory_entities": memory_entities,
        "conflicts": conflicts,
        "patterns": patterns,
        "triggers": triggers,
    }


# ── Import ──


@cli.command("import")
@_db_option
@_dim_option
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--merge", is_flag=True,
              help="Merge into existing DB (skip duplicates). Default: fail if DB has data.")
def import_cmd(db_path, embedding_dim, input_path, merge):
    """Import a JSON export into the database."""
    data = json.loads(Path(input_path).read_text())
    if data.get("version") not in ("yantrikdb-export-v1", "aidb-export-v1"):
        click.echo(f"Unknown export version: {data.get('version')}", err=True)
        sys.exit(1)

    # Ensure parent directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    db = _open_db(db_path, embedding_dim)
    try:
        existing = db.stats()
        if existing["active_memories"] > 0 and not merge:
            click.echo(
                "Target database is not empty. Use --merge to import anyway.",
                err=True,
            )
            sys.exit(1)

        imported = _import_db(db, data, merge=merge)
        click.echo(f"Imported: {imported['memories']} memories, "
                    f"{imported['entities']} entities, "
                    f"{imported['edges']} edges, "
                    f"{imported['memory_entities']} memory_entities")
    finally:
        db.close()


def _import_db(db, data: dict, merge: bool = False) -> dict:
    """Import data into YantrikDB. Returns counts of imported items."""
    conn = db._conn
    counts = {"memories": 0, "entities": 0, "edges": 0, "memory_entities": 0}

    import struct

    def _encode_embedding(emb):
        """Encode a float list as a blob (little-endian f32 array)."""
        if emb is None:
            return None
        return struct.pack(f"<{len(emb)}f", *emb)

    # Import memories
    for m in data.get("memories", []):
        if merge:
            existing = db.get(m["rid"])
            if existing is not None:
                continue

        embedding = m.get("embedding")
        metadata = m.get("metadata", "{}")
        if not isinstance(metadata, str):
            metadata = json.dumps(metadata)

        now = m.get("created_at", 0)
        conn.execute(
            "INSERT OR IGNORE INTO memories "
            "(rid, type, text, importance, valence, half_life, "
            "consolidation_status, storage_tier, metadata, "
            "embedding, created_at, updated_at, last_access) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                m["rid"], m.get("type", "episodic"), m["text"],
                m["importance"], m["valence"], m.get("half_life", 604800.0),
                m.get("consolidation_status", "active"),
                m.get("storage_tier", "hot"), metadata,
                _encode_embedding(embedding),
                m["created_at"], m.get("updated_at", now),
                m.get("last_access", now),
            ),
        )
        counts["memories"] += 1

    # Import entities
    for e in data.get("entities", []):
        now = e.get("first_seen", 0)
        conn.execute(
            "INSERT OR IGNORE INTO entities "
            "(name, entity_type, first_seen, last_seen, mention_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (e["name"], e.get("entity_type", "unknown"),
             e["first_seen"], e.get("last_seen", now),
             e.get("mention_count", 1)),
        )
        counts["entities"] += 1

    # Import edges. The `edges` name is a backward-compat VIEW since
    # V16→V17 (RFC 006 Phase 5) — real backing table is `claims` with
    # `claim_id` instead of `edge_id`. Write to `claims` so the INSERT
    # actually lands; the existing export side reads through the view,
    # which still works fine because the view aliases claim_id → edge_id.
    for e in data.get("edges", []):
        conn.execute(
            "INSERT OR IGNORE INTO claims "
            "(claim_id, src, dst, rel_type, weight, created_at, tombstoned) "
            "VALUES (?, ?, ?, ?, ?, ?, 0)",
            (e["edge_id"], e["src"], e["dst"], e["rel_type"],
             e["weight"], e["created_at"]),
        )
        counts["edges"] += 1

    # Import memory_entities
    for me in data.get("memory_entities", []):
        conn.execute(
            "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name) "
            "VALUES (?, ?)",
            (me["memory_rid"], me["entity_name"]),
        )
        counts["memory_entities"] += 1

    conn.commit()
    return counts


# ── Conflicts ──


@cli.command()
@_db_option
@_dim_option
@click.option("--status", default=None, help="Filter: open, resolved, dismissed.")
def conflicts(db_path, embedding_dim, status):
    """List memory conflicts."""
    db = _open_db(db_path, embedding_dim)
    try:
        items = db.get_conflicts(status=status)
        if not items:
            click.echo("No conflicts.")
            return
        for c in items:
            click.echo(f"\n[{c['priority']:.2f}] {c['conflict_id'][:12]}...  "
                        f"status={c['status']}  type={c['conflict_type']}")
            click.echo(f"    entity={c.get('entity', '?')}")
            click.echo(f"    {c['detection_reason']}")
    finally:
        db.close()


# ── Triggers ──


@cli.command()
@_db_option
@_dim_option
def triggers(db_path, embedding_dim):
    """List pending triggers."""
    db = _open_db(db_path, embedding_dim)
    try:
        items = db.get_pending_triggers(limit=50)
        if not items:
            click.echo("No pending triggers.")
            return
        for t in items:
            click.echo(f"  [{t['urgency']:.2f}] {t['trigger_type']}: {t['reason']}")
            click.echo(f"         action: {t['suggested_action']}")
    finally:
        db.close()


def main():
    """Entry point for the yantrikdb CLI console script."""
    cli()
