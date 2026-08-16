"""Source readers — each yields ImportRecord from one known memory system.

File-based readers take a path and need no third-party package. API-based
readers (mem0 live, Zep, Letta) lazy-import their client the way
eval/competitors.py does, so none of them is a hard dependency.

Timestamp handling is the load-bearing part: every reader maps the source's
own creation time onto `created_at` so imported memories keep their history.
A reader that cannot find a timestamp leaves it None (= now) rather than
guessing — a wrong backdate is worse than none.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from . import ImportRecord


def _epoch(value) -> float | None:
    """ISO string / epoch number / None -> epoch seconds or None."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        # Heuristic: mem0 and friends ship seconds; anything absurdly large
        # is milliseconds.
        return float(value) / 1000.0 if value > 1e12 else float(value)
    try:
        s = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


# ── generic ────────────────────────────────────────────────────────────

def iter_jsonl(path: str, *, namespace: str = "default",
               text_field: str = "text", id_field: str = "id",
               time_field: str = "created_at",
               user_field: str = "user_id", **_) -> Iterator[ImportRecord]:
    """JSONL or CSV with one memory per row. The lingua franca exporter:
    any system that can dump rows can reach this."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        with open(p, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        with open(p, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    for row in rows:
        user = row.get(user_field)
        yield ImportRecord(
            text=str(row.get(text_field) or ""),
            source_system="jsonl",
            source_id=str(row[id_field]) if row.get(id_field) else None,
            created_at=_epoch(row.get(time_field)),
            namespace=str(user) if user else namespace,
            metadata={k: v for k, v in row.items()
                      if k not in (text_field,) and v not in (None, "")},
        )


# ── mem0 ───────────────────────────────────────────────────────────────

def iter_mem0(path: str | None = None, *, api_key: str | None = None,
              user_id: str | None = None, namespace: str = "default",
              **_) -> Iterator[ImportRecord]:
    """mem0 memories, from an export file or the live API.

    File form: the JSON shape of `Memory.get_all()` — either a bare list or
    `{"results": [...]}`, items like
    `{"id", "memory", "user_id", "created_at", "metadata", "categories"}`.
    Live form: `--api-key` (or MEM0_API_KEY) pulls via the mem0 client.
    """
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        items = data.get("results", data) if isinstance(data, dict) else data
    else:
        # Optional dep, lint-gate convention: the disable rides the import line.
        from mem0 import MemoryClient  # pylint: disable=import-error
        client = MemoryClient(api_key=api_key or os.environ.get("MEM0_API_KEY"))
        got = client.get_all(user_id=user_id) if user_id else client.get_all()
        items = got.get("results", got) if isinstance(got, dict) else got

    for it in items or []:
        meta = dict(it.get("metadata") or {})
        if it.get("categories"):
            meta["mem0_categories"] = it["categories"]
        user = it.get("user_id")
        yield ImportRecord(
            text=str(it.get("memory") or it.get("text") or ""),
            source_system="mem0",
            source_id=str(it["id"]) if it.get("id") else None,
            created_at=_epoch(it.get("created_at")),
            namespace=str(user) if user else namespace,
            metadata=meta,
        )


# ── zep ────────────────────────────────────────────────────────────────

def iter_zep(path: str | None = None, *, api_key: str | None = None,
             user_id: str | None = None, namespace: str = "default",
             **_) -> Iterator[ImportRecord]:
    """Zep facts. File form: JSON list of fact rows (`{"fact"|"content",
    "uuid", "created_at"}`). Live form: the zep client's user graph/facts."""
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        items = data.get("facts", data) if isinstance(data, dict) else data
    else:
        from zep_cloud.client import Zep  # pylint: disable=import-error
        client = Zep(api_key=api_key or os.environ.get("ZEP_API_KEY"))
        items = [f.dict() for f in
                 (client.user.get_facts(user_id=user_id).facts or [])]
    for it in items or []:
        yield ImportRecord(
            text=str(it.get("fact") or it.get("content") or ""),
            source_system="zep",
            source_id=str(it.get("uuid") or it.get("id") or "") or None,
            created_at=_epoch(it.get("created_at")),
            namespace=namespace if not user_id else str(user_id),
            metadata={k: v for k, v in it.items()
                      if k in ("rating", "source_node_name", "target_node_name")
                      and v is not None},
        )


# ── letta (memgpt) ─────────────────────────────────────────────────────

def iter_letta(path: str | None = None, *, agent_id: str | None = None,
               namespace: str = "default", **_) -> Iterator[ImportRecord]:
    """Letta archival memory. File form: JSON list of passages
    (`{"text", "id", "created_at"}`). Live form: the letta client's
    archival passages for an agent."""
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("passages", [])
    else:
        from letta_client import Letta  # pylint: disable=import-error
        client = Letta(base_url=os.environ.get("LETTA_BASE_URL",
                                               "http://localhost:8283"))
        items = [p.dict() for p in
                 client.agents.passages.list(agent_id=agent_id)]
    for it in items or []:
        yield ImportRecord(
            text=str(it.get("text") or ""),
            source_system="letta",
            source_id=str(it.get("id") or "") or None,
            created_at=_epoch(it.get("created_at")),
            memory_type="episodic",
            namespace=namespace,
        )


# ── markdown notes ─────────────────────────────────────────────────────

def iter_markdown(path: str, *, namespace: str = "default",
                  **_) -> Iterator[ImportRecord]:
    """A folder of .md notes (Obsidian-style): one memory per file, title
    leading the text, file mtime as the backdate. Deliberately per-file —
    splitting someone's notes without asking is a judgment call an importer
    has no right to make."""
    root = Path(path)
    for p in sorted(root.rglob("*.md")):
        body = p.read_text(encoding="utf-8", errors="replace").strip()
        if not body:
            continue
        title = p.stem.replace("-", " ").replace("_", " ")
        text = body if body.lower().startswith(title.lower()) \
            else f"{title}: {body}"
        yield ImportRecord(
            text=text[:8000],
            source_system="markdown",
            source_id=str(p.relative_to(root)),
            created_at=p.stat().st_mtime,
            namespace=namespace,
            metadata={"file": str(p.relative_to(root))},
        )


# ── mnemosyne (hermes-agent) ───────────────────────────────────────────

def iter_mnemosyne(path: str, *, namespace: str | None = None,
                   **_) -> Iterator[ImportRecord]:
    """Mnemosyne's SQLite store (AxDSan/mnemosyne, the Hermes Agent memory).

    Schema verified against their source (mnemosyne/core/{memory,beam}.py):
    a legacy flat `memories` table plus BEAM tiers `working_memory` and
    `episodic_memory` — identical columns (id, content, source, timestamp,
    session_id, importance, metadata_json, created_at) with the tiers adding
    `veracity` and episodic adding `summary_of`. Read directly with stdlib
    sqlite3; importing a competitor's store must not depend on the
    competitor's package.

    Tier handling: the same id can exist in several tiers as a memory is
    promoted, so rows dedupe by id with episodic (the consolidated tier)
    winning. `scratchpad` is deliberately NOT imported — it is transient
    working notes, and importing someone's scratch as durable memory is the
    kind of favour nobody asked for. Their `session_id` maps to namespace
    unless one is forced.
    """
    import sqlite3 as _sq

    con = _sq.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = _sq.Row
    have = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    # episodic first so it wins the id-dedupe.
    tiers = [t for t in ("episodic_memory", "memories", "working_memory")
             if t in have]
    if not tiers:
        con.close()
        raise ValueError(
            f"{path}: no mnemosyne tables found (looked for episodic_memory/"
            f"memories/working_memory; present: {sorted(have)[:8]})")

    seen: set[str] = set()
    for table in tiers:
        cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        for row in con.execute(f"SELECT * FROM {table}"):
            rid = str(row["id"])
            if rid in seen:
                continue
            seen.add(rid)
            meta: dict = {"mnemosyne_tier": table}
            if row["metadata_json"]:
                try:
                    inner = json.loads(row["metadata_json"])
                    if isinstance(inner, dict):
                        meta.update(inner)
                except (ValueError, TypeError):
                    meta["metadata_json_raw"] = str(row["metadata_json"])[:500]
            if "veracity" in cols and row["veracity"]:
                meta["veracity"] = row["veracity"]
            if "summary_of" in cols and row["summary_of"]:
                meta["derived_summary_of"] = row["summary_of"]
            if row["source"]:
                meta["mnemosyne_source"] = row["source"]
            veracity = str(row["veracity"]) if "veracity" in cols and row["veracity"] else "unknown"
            certainty = {"verified": 0.95, "unknown": 0.8}.get(veracity, 0.5)
            sess = str(row["session_id"] or "default")
            yield ImportRecord(
                text=str(row["content"] or ""),
                source_system="mnemosyne",
                source_id=rid,
                created_at=_epoch(row["timestamp"]) or _epoch(row["created_at"]),
                memory_type="episodic" if table == "episodic_memory" else "semantic",
                importance=float(row["importance"] or 0.5),
                namespace=namespace or sess,
                certainty=certainty,
                metadata=meta,
            )
    con.close()
