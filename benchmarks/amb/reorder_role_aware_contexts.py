"""Freeze a chronological arm from role-aware AMB contexts, with no LLM calls."""

import argparse
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

try:
    from .chronological_presentation import chronological_document_key
except ImportError:  # pragma: no cover - direct script execution
    from chronological_presentation import chronological_document_key


def reorder_row(row: dict) -> dict:
    """Return a pure presentation permutation of one frozen-context row."""
    documents = list(row.get("documents") or [])
    selection = deepcopy(row.get("selection") or {})
    selected_results = list(selection.get("results") or [])
    if selected_results and len(selected_results) != len(documents):
        raise ValueError(
            f"{row.get('query_id')}: {len(documents)} documents but "
            f"{len(selected_results)} selection results"
        )

    order = sorted(
        range(len(documents)),
        key=lambda index: chronological_document_key(documents[index], index),
    )
    ordered_documents = [documents[index] for index in order]
    if Counter(ordered_documents) != Counter(documents):
        raise AssertionError(f"{row.get('query_id')}: evidence set changed")

    selection["presentation_order"] = "chronological"
    selection["presentation_reordered"] = order != list(range(len(order)))
    if selected_results:
        selection["presented_results"] = [selected_results[index] for index in order]

    output = deepcopy(row)
    output["selection"] = selection
    output["documents"] = ordered_documents
    output["context"] = "\n\n".join(
        f"## Memory {index}\n{content}"
        for index, content in enumerate(ordered_documents, 1)
    )
    return output


def transform(payload: dict) -> dict:
    output = deepcopy(payload)
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("artifact must contain a results list")
    output["results"] = [reorder_row(row) for row in rows]
    output["artifact_transform"] = {
        "name": "role-aware-chronological-presentation-v1",
        "selection_changed": False,
        "llm_calls": 0,
    }
    return output


def audit_transform(source: dict, output: dict) -> dict:
    """Prove the transformed arm changes presentation and nothing else."""
    source_rows = source.get("results") or []
    output_rows = output.get("results") or []
    if len(source_rows) != len(output_rows):
        raise AssertionError("row count changed")

    reordered_rows = 0
    document_count = 0
    unknown_prefixes = 0
    for original, transformed in zip(source_rows, output_rows, strict=True):
        query_id = original.get("query_id")
        if transformed.get("query_id") != query_id:
            raise AssertionError(f"{query_id}: query order changed")
        before = list(original.get("documents") or [])
        after = list(transformed.get("documents") or [])
        if Counter(before) != Counter(after):
            raise AssertionError(f"{query_id}: document multiset changed")
        if (original.get("selection") or {}).get("results") != (
            transformed.get("selection") or {}
        ).get("results"):
            raise AssertionError(f"{query_id}: relevance selection trace changed")
        if len(original.get("context") or "") != len(
            transformed.get("context") or ""
        ):
            raise AssertionError(f"{query_id}: context length changed")

        keys = [
            chronological_document_key(content, index)
            for index, content in enumerate(after)
        ]
        if keys != sorted(keys):
            raise AssertionError(f"{query_id}: context is not chronological")
        unknown_prefixes += sum(key[0] == float("inf") for key in keys)
        document_count += len(after)
        reordered_rows += before != after

    return {
        "rows": len(output_rows),
        "documents": document_count,
        "reordered_rows": reordered_rows,
        "unknown_prefixes": unknown_prefixes,
        "selection_changed": False,
        "context_lengths_changed": False,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    output = transform(payload)
    audit = audit_transform(payload, output)
    output["artifact_transform"]["audit"] = audit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        f"wrote {args.output} audit={json.dumps(audit, sort_keys=True)} "
        f"sha256={_sha256(args.output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
