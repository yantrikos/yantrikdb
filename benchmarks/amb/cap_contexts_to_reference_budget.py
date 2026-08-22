#!/usr/bin/env python3
"""Cap treatment contexts to each reference row's token budget."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

try:
    from .reorder_speaker_first_contexts import split_memory_blocks
except ImportError:  # pragma: no cover - direct script execution
    from reorder_speaker_first_contexts import split_memory_blocks


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, list) else payload.get("results") or []


def cap_context(
    context: str,
    token_budget: int,
    token_counter: Callable[[str], int],
) -> tuple[str, dict]:
    if token_budget < 1:
        raise ValueError("token budget must be positive")
    blocks = split_memory_blocks(context)
    treatment_tokens_before = token_counter(context)
    low = 1
    high = len(blocks)
    selected_count = 0
    selected_tokens = 0
    while low <= high:
        middle = (low + high) // 2
        candidate_tokens = token_counter("".join(blocks[:middle]))
        if candidate_tokens <= token_budget:
            selected_count = middle
            selected_tokens = candidate_tokens
            low = middle + 1
        else:
            high = middle - 1
    if selected_count == 0:
        raise ValueError("first treatment memory block exceeds the reference budget")
    capped = "".join(blocks[:selected_count])
    return capped, {
        "reference_token_budget": token_budget,
        "treatment_tokens_before": treatment_tokens_before,
        "treatment_tokens_after": selected_tokens,
        "blocks_before": len(blocks),
        "blocks_after": selected_count,
        "prefix_preserved": True,
    }


def transform(
    reference_rows: list[dict],
    treatment_rows: list[dict],
    token_counter: Callable[[str], int],
) -> dict:
    treatment_by_id = {
        str(row.get("query_id") or ""): row
        for row in treatment_rows
        if row.get("query_id")
    }
    output_rows = []
    for reference in reference_rows:
        query_id = str(reference.get("query_id") or "")
        treatment = treatment_by_id.get(query_id)
        if treatment is None:
            raise ValueError(f"treatment is missing query {query_id!r}")
        reference_context = str(reference.get("context") or "")
        treatment_context = str(treatment.get("context") or "")
        if not reference_context.strip() or not treatment_context.strip():
            raise ValueError(f"empty context for query {query_id!r}")
        capped, audit = cap_context(
            treatment_context,
            token_counter(reference_context),
            token_counter,
        )
        output_rows.append(
            {
                "query_id": query_id,
                "query": treatment.get("query") or reference.get("query"),
                "gold_answers": treatment.get("gold_answers")
                or reference.get("gold_answers"),
                "context": capped,
                "budget_audit": audit,
            }
        )
    total_reference = sum(
        token_counter(str(row.get("context") or "")) for row in reference_rows
    )
    total_treatment = sum(
        row["budget_audit"]["treatment_tokens_after"] for row in output_rows
    )
    return {
        "artifact_transform": {
            "name": "reference-token-budget-prefix-cap-v1",
            "selection_order_changed": False,
            "whole_memory_blocks_only": True,
            "external_calls": 0,
            "rows": len(output_rows),
            "reference_context_tokens": total_reference,
            "treatment_context_tokens": total_treatment,
            "treatment_within_reference_budget": total_treatment <= total_reference,
        },
        "results": output_rows,
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    from memory_bench.utils import count_tokens

    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("treatment", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    output = transform(
        load_rows(args.reference),
        load_rows(args.treatment),
        count_tokens,
    )
    output["artifact_transform"].update(
        {
            "reference_sha256": _sha256_file(args.reference),
            "treatment_sha256": _sha256_file(args.treatment),
            "synthetic_benchmark_data_only": True,
            "real_companion_memories_included": False,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["artifact_transform"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
