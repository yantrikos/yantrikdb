"""CrewAI memory adapter for YantrikDB.

Provides short-term (episodic), long-term (semantic), and entity (graph)
memory backends for CrewAI agents.

Usage:
    from yantrikdb.adapters.crewai import YantrikDBShortTermMemory, YantrikDBLongTermMemory, YantrikDBEntityMemory

    crew = Crew(
        short_term_memory=YantrikDBShortTermMemory(db),
        long_term_memory=YantrikDBLongTermMemory(db),
        entity_memory=YantrikDBEntityMemory(db),
    )
"""

from __future__ import annotations

from typing import Any


class YantrikDBShortTermMemory:
    """CrewAI short-term memory backed by YantrikDB episodic memories."""

    def __init__(
        self,
        db: Any,
        top_k: int = 5,
        namespace: str = "default",
        source: str = "assistant",
    ):
        self.db = db
        self.top_k = top_k
        self.namespace = namespace
        self.source = source

    def save(self, value: str, metadata: dict | None = None, agent: str | None = None) -> None:
        """Store a short-term memory (episodic)."""
        meta = dict(metadata or {})
        meta.setdefault("speaker_role", self.source)
        meta.setdefault("provenance_verified", True)
        meta.setdefault("provenance_method", "adapter_declared_source_v1")
        if agent:
            meta["agent"] = agent
        self.db.record(
            text=value,
            memory_type="episodic",
            importance=0.4,
            metadata=meta,
            namespace=self.namespace,
            source=self.source,
        )

    def search(self, query: str, limit: int | None = None) -> list[dict]:
        """Search short-term memories."""
        k = limit or self.top_k
        results = self.db.recall(
            query=query,
            top_k=k,
            memory_type="episodic",
            namespace=self.namespace,
        )
        return [{"context": r["text"], "score": r["score"]} for r in results]

    def reset(self) -> None:
        """Reset is a no-op — YantrikDB uses decay, not deletion."""
        pass


class YantrikDBLongTermMemory:
    """CrewAI long-term memory backed by YantrikDB semantic memories."""

    def __init__(
        self,
        db: Any,
        top_k: int = 5,
        namespace: str = "default",
        source: str = "assistant",
    ):
        self.db = db
        self.top_k = top_k
        self.namespace = namespace
        self.source = source

    def save(self, value: str, metadata: dict | None = None, agent: str | None = None) -> None:
        """Store a long-term memory (semantic)."""
        meta = dict(metadata or {})
        meta.setdefault("speaker_role", self.source)
        meta.setdefault("provenance_verified", True)
        meta.setdefault("provenance_method", "adapter_declared_source_v1")
        if agent:
            meta["agent"] = agent
        self.db.record(
            text=value,
            memory_type="semantic",
            importance=0.7,
            metadata=meta,
            namespace=self.namespace,
            source=self.source,
        )

    def search(self, query: str, limit: int | None = None) -> list[dict]:
        """Search long-term memories."""
        k = limit or self.top_k
        results = self.db.recall(
            query=query,
            top_k=k,
            memory_type="semantic",
            namespace=self.namespace,
        )
        return [{"context": r["text"], "score": r["score"]} for r in results]

    def reset(self) -> None:
        """Reset is a no-op — YantrikDB uses decay, not deletion."""
        pass


class YantrikDBEntityMemory:
    """CrewAI entity memory backed by YantrikDB knowledge graph."""

    def __init__(
        self,
        db: Any,
        top_k: int = 5,
        namespace: str = "default",
        source: str = "assistant",
    ):
        self.db = db
        self.top_k = top_k
        self.namespace = namespace
        self.source = source

    def save(self, value: str, metadata: dict | None = None, agent: str | None = None) -> None:
        """Store an entity observation.

        If metadata contains 'entity' and 'relationship' keys,
        also creates a graph edge.
        """
        meta = dict(metadata or {})
        meta.setdefault("speaker_role", self.source)
        meta.setdefault("provenance_verified", True)
        meta.setdefault("provenance_method", "adapter_declared_source_v1")
        if agent:
            meta["agent"] = agent

        rid = self.db.record(
            text=value,
            memory_type="semantic",
            importance=0.6,
            metadata=meta,
            namespace=self.namespace,
            source=self.source,
        )

        # Auto-link entities if provided
        entity = meta.get("entity")
        if entity:
            self.db.link_memory_entity(rid, entity)
            target = meta.get("target_entity")
            rel = meta.get("relationship", "related_to")
            if target:
                self.db.relate(src=entity, dst=target, rel_type=rel)
                self.db.link_memory_entity(rid, target)

    def search(self, query: str, limit: int | None = None) -> list[dict]:
        """Search entity memories with graph expansion."""
        k = limit or self.top_k
        results = self.db.recall(
            query=query,
            top_k=k,
            expand_entities=True,
            namespace=self.namespace,
        )
        return [{"context": r["text"], "score": r["score"]} for r in results]

    def reset(self) -> None:
        """Reset is a no-op — YantrikDB uses decay, not deletion."""
        pass
