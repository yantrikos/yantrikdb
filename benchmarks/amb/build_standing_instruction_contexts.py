#!/usr/bin/env python3
"""Build equal-budget contexts with authoritative standing instructions first."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path

try:
    from .audit_knowledge_update_gold import iter_turns, load_conversations
    from .reorder_speaker_first_contexts import split_memory_blocks
except ImportError:  # pragma: no cover - direct script execution
    from audit_knowledge_update_gold import iter_turns, load_conversations
    from reorder_speaker_first_contexts import split_memory_blocks


STANDING_INSTRUCTION_RE = re.compile(r"^Always\b", re.IGNORECASE)
TRAILING_LINK_RE = re.compile(r"\s*->->.*$")


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, list) else payload.get("results") or []


def extract_standing_instructions(conversation: dict) -> list[dict]:
    """Return explicit user-authored `Always ...` instructions in turn order."""
    records = []
    for turn in iter_turns(conversation.get("chat") or []):
        if str(turn.get("role") or "").casefold() != "user":
            continue
        text = TRAILING_LINK_RE.sub("", str(turn.get("content") or "")).strip()
        if STANDING_INSTRUCTION_RE.match(text):
            records.append({"turn": turn.get("id"), "text": text})
    return records


def render_instruction_panel(records: list[dict]) -> str:
    lines = [
        f"- [Turn {record['turn']}] User: {record['text']}" for record in records
    ]
    return (
        "## Memory 0\n"
        "Standing user instructions (authoritative user statements):\n"
        + "\n".join(lines)
        + "\n\n"
    )


def cap_whole_blocks(
    context: str,
    token_budget: int,
    token_counter: Callable[[str], int],
) -> tuple[str, int, int]:
    blocks = split_memory_blocks(context)
    low, high = 1, len(blocks)
    selected_count = 0
    selected_tokens = 0
    while low <= high:
        middle = (low + high) // 2
        candidate = "".join(blocks[:middle])
        candidate_tokens = token_counter(candidate)
        if candidate_tokens <= token_budget:
            selected_count = middle
            selected_tokens = candidate_tokens
            low = middle + 1
        else:
            high = middle - 1
    if selected_count == 0:
        raise ValueError("standing-instruction panel exceeds the reference budget")
    return "".join(blocks[:selected_count]), selected_count, selected_tokens


def build_contexts(
    rows: list[dict],
    conversations: dict[str, dict],
    token_counter: Callable[[str], int],
    category: str | None = "instruction_following",
) -> dict:
    output = []
    total_reference_tokens = 0
    total_treatment_tokens = 0
    for row in rows:
        metadata = row.get("meta") or {}
        if category is not None and metadata.get("question_category") != category:
            continue
        query_id = str(row.get("query_id") or "")
        conversation_id = str(metadata.get("conversation_id") or "")
        conversation = conversations.get(conversation_id)
        if conversation is None:
            raise ValueError(f"missing conversation {conversation_id!r}")
        records = extract_standing_instructions(conversation)
        if not records:
            raise ValueError(f"no standing instructions for query {query_id!r}")
        reference_context = str(row.get("context") or "")
        reference_tokens = token_counter(reference_context)
        treatment_before = render_instruction_panel(records) + reference_context
        treatment_context, blocks_after, treatment_tokens = cap_whole_blocks(
            treatment_before, reference_tokens, token_counter
        )
        if not treatment_context.startswith("## Memory 0\nStanding user instructions"):
            raise AssertionError("standing-instruction panel was not preserved")
        transformed = dict(row)
        transformed["context"] = treatment_context
        transformed["standing_instruction_audit"] = {
            "conversation_id": conversation_id,
            "instruction_count": len(records),
            "instruction_turns": [record["turn"] for record in records],
            "reference_tokens": reference_tokens,
            "treatment_tokens": treatment_tokens,
            "blocks_before_cap": len(split_memory_blocks(treatment_before)),
            "blocks_after_cap": blocks_after,
            "whole_blocks_only": True,
        }
        output.append(transformed)
        total_reference_tokens += reference_tokens
        total_treatment_tokens += treatment_tokens
    if not output:
        raise ValueError(f"no rows found for category {category!r}")
    return {
        "protocol": "standing-user-instruction-context-v1",
        "artifact_transform": {
            "category": category or "all",
            "rows": len(output),
            "selection_changed": True,
            "query_or_gold_used_for_instruction_extraction": False,
            "authoritative_user_turns_only": True,
            "reference_context_tokens": total_reference_tokens,
            "treatment_context_tokens": total_treatment_tokens,
            "treatment_within_reference_budget": (
                total_treatment_tokens <= total_reference_tokens
            ),
            "external_calls": 0,
            "synthetic_benchmark_data_only": True,
            "real_companion_memories_included": False,
        },
        "results": output,
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    from memory_bench.utils import count_tokens

    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("documents", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--category",
        default="instruction_following",
        help="question category to transform, or 'all' for the full cohort",
    )
    args = parser.parse_args()
    payload = build_contexts(
        load_rows(args.results),
        load_conversations(args.documents),
        count_tokens,
        None if args.category == "all" else args.category,
    )
    payload["artifact_transform"].update(
        {
            "results_sha256": _sha256_file(args.results),
            "documents_sha256": _sha256_file(args.documents),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["artifact_transform"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
