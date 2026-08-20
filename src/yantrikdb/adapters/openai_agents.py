"""OpenAI Agents SDK adapter for YantrikDB.

Generates function-calling tool definitions and dispatches tool calls
to the YantrikDB engine.

Usage:
    from yantrikdb.adapters.openai_agents import get_tools, handle_tool_call

    tools = get_tools()
    # Add tools to your agent definition
    # When a tool call comes in:
    result = handle_tool_call(db, tool_name, arguments)
"""

from __future__ import annotations

from typing import Any


def get_tools() -> list[dict]:
    """Return OpenAI function-calling tool definitions for YantrikDB."""
    return [
        {
            "type": "function",
            "function": {
                "name": "memory_record",
                "description": "Store a new memory in the cognitive memory engine.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The memory content to store.",
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["episodic", "semantic", "procedural"],
                            "description": "Type of memory. Default: episodic.",
                        },
                        "importance": {
                            "type": "number",
                            "description": "Importance score 0.0-1.0. Default: 0.5.",
                        },
                        "valence": {
                            "type": "number",
                            "description": "Emotional tone -1.0 to 1.0. Default: 0.0.",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional key-value metadata.",
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Memory namespace for isolation. Default: default.",
                        },
                        "source": {
                            "type": "string",
                            "enum": [
                                "user",
                                "assistant",
                                "system",
                                "document",
                                "inference",
                            ],
                            "description": (
                                "Who originally supplied the claim. Use user only for facts "
                                "explicitly stated by the user. Default: assistant."
                            ),
                        },
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_recall",
                "description": "Search memories by semantic similarity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max results. Default: 10.",
                        },
                        "memory_type": {
                            "type": "string",
                            "description": "Filter by type.",
                        },
                        "namespace": {
                            "type": "string",
                            "description": "Filter by namespace. Omit for all.",
                        },
                        "source": {
                            "type": "string",
                            "description": "Filter by claim origin. Omit to search all sources.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_forget",
                "description": "Tombstone a memory by its ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rid": {
                            "type": "string",
                            "description": "The memory ID to forget.",
                        },
                    },
                    "required": ["rid"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "entity_relate",
                "description": "Create a relationship between two entities.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Source entity name.",
                        },
                        "target": {
                            "type": "string",
                            "description": "Target entity name.",
                        },
                        "relationship": {
                            "type": "string",
                            "description": "Relationship type. Default: related_to.",
                        },
                        "weight": {
                            "type": "number",
                            "description": "Strength 0.0-1.0. Default: 1.0.",
                        },
                    },
                    "required": ["source", "target"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "entity_edges",
                "description": "Get all relationships for an entity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {
                            "type": "string",
                            "description": "Entity name to look up.",
                        },
                    },
                    "required": ["entity"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_stats",
                "description": "Get memory engine statistics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Filter stats to a namespace. Omit for global.",
                        },
                    },
                },
            },
        },
    ]


def handle_tool_call(db: Any, name: str, arguments: dict) -> Any:
    """Dispatch a tool call to the YantrikDB engine.

    Args:
        db: A YantrikDB instance (with embedder configured).
        name: Tool function name.
        arguments: Tool call arguments.

    Returns:
        The result of the tool call.
    """
    if name == "memory_record":
        metadata = dict(arguments.get("metadata") or {})
        metadata.setdefault("provenance_verified", False)
        metadata.setdefault("provenance_method", "agent_declared_source_v1")
        rid = db.record(
            text=arguments["text"],
            memory_type=arguments.get("memory_type", "episodic"),
            importance=arguments.get("importance", 0.5),
            valence=arguments.get("valence", 0.0),
            metadata=metadata,
            namespace=arguments.get("namespace", "default"),
            source=arguments.get("source", "assistant"),
        )
        return {"rid": rid}

    elif name == "memory_recall":
        results = db.recall(
            query=arguments["query"],
            top_k=arguments.get("top_k", 10),
            memory_type=arguments.get("memory_type"),
            namespace=arguments.get("namespace"),
            source=arguments.get("source"),
        )
        return {"memories": results}

    elif name == "memory_forget":
        success = db.forget(arguments["rid"])
        return {"forgotten": success}

    elif name == "entity_relate":
        edge_id = db.relate(
            src=arguments["source"],
            dst=arguments["target"],
            rel_type=arguments.get("relationship", "related_to"),
            weight=arguments.get("weight", 1.0),
        )
        return {"edge_id": edge_id}

    elif name == "entity_edges":
        edges = db.get_edges(arguments["entity"])
        return {"edges": edges}

    elif name == "memory_stats":
        return db.stats(namespace=arguments.get("namespace"))

    else:
        raise ValueError(f"Unknown tool: {name}")
