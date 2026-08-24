"""Verify the sealed BEAM-500k holdout without displaying its contents."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL = "amb-event-ordering-thread-v3-holdout-seal-v1"


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else path.open
    with opener(path, "rt", encoding="utf-8") if path.suffix == ".gz" else opener(
        "r", encoding="utf-8"
    ) as handle:
        return json.load(handle)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ordered_ids_sha256(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(str(row["id"]) for row in rows).encode("utf-8")
    ).hexdigest()


def _index_unique(rows: list[dict[str, Any]], label: str) -> None:
    ids = [str(row.get("id") or "") for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains missing or duplicate IDs")


def verify(
    manifest: dict[str, Any],
    queries_path: Path,
    documents_path: Path,
    burned_queries_path: Path | None = None,
    categories_path: Path | None = None,
    stats_path: Path | None = None,
) -> dict[str, Any]:
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("unexpected holdout manifest protocol")
    if manifest.get("status") != "active":
        raise ValueError("holdout seal is not active")
    source_hashes = manifest["source_sha256"]
    source_paths = {
        "queries.json.gz": queries_path,
        "documents.json.gz": documents_path,
        "categories.json.gz": categories_path,
        "stats.json.gz": stats_path,
    }
    for name, expected_hash in source_hashes.items():
        path = source_paths.get(name)
        if path is None:
            raise ValueError(f"sealed source path was not supplied: {name}")
        if sha256_path(path) != expected_hash:
            raise ValueError(f"sealed source hash mismatch: {name}")

    queries = read_json(queries_path)
    documents = read_json(documents_path)
    if not isinstance(queries, list) or not isinstance(documents, list):
        raise ValueError("BEAM query and document sources must be arrays")
    units = set(manifest["holdout_units"])
    selected_queries = [
        row for row in queries if str(row.get("user_id")) in units
    ]
    selected_documents = [
        row for row in documents if str(row.get("user_id")) in units
    ]
    event_category = manifest["event_category"]
    event_queries = [
        row
        for row in selected_queries
        if (row.get("meta") or {}).get("question_category") == event_category
    ]
    _index_unique(selected_queries, "holdout queries")
    _index_unique(event_queries, "holdout event queries")
    _index_unique(selected_documents, "holdout documents")

    expected = manifest["expected"]
    actual_counts = {
        "query_rows": len(selected_queries),
        "event_rows": len(event_queries),
        "document_rows": len(selected_documents),
        "unit_count": len({str(row.get("user_id")) for row in selected_queries}),
    }
    for key, value in actual_counts.items():
        if value != expected[key]:
            raise ValueError(f"holdout {key} mismatch: expected {expected[key]}, got {value}")

    actual_hashes = {
        "all_queries": canonical_sha256(selected_queries),
        "event_queries": canonical_sha256(event_queries),
        "all_ordered_query_ids": ordered_ids_sha256(selected_queries),
        "event_ordered_query_ids": ordered_ids_sha256(event_queries),
        "documents": canonical_sha256(selected_documents),
    }
    for key, value in actual_hashes.items():
        if value != manifest["subset_sha256"][key]:
            raise ValueError(f"holdout subset hash mismatch: {key}")

    overlap = None
    if burned_queries_path is not None:
        burned = read_json(burned_queries_path)
        if not isinstance(burned, list):
            raise ValueError("burned query source must be an array")
        burned_event_text = {
            str(row.get("query") or "").casefold().strip()
            for row in burned
            if (row.get("meta") or {}).get("question_category") == event_category
        }
        overlap = sum(
            str(row.get("query") or "").casefold().strip() in burned_event_text
            for row in event_queries
        )
        if overlap != expected["exact_event_query_overlap_with_beam_100k"]:
            raise ValueError("holdout event-query overlap mismatch")

    return {
        "protocol": PROTOCOL,
        "verified": True,
        "counts": actual_counts,
        "subset_sha256": actual_hashes,
        "exact_event_query_overlap_with_beam_100k": overlap,
        "content_emitted": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--queries", type=Path, required=True)
    result.add_argument("--documents", type=Path, required=True)
    result.add_argument("--categories", type=Path, required=True)
    result.add_argument("--stats", type=Path, required=True)
    result.add_argument("--burned-queries", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    manifest = read_json(args.manifest)
    report = verify(
        manifest,
        args.queries,
        args.documents,
        args.burned_queries,
        args.categories,
        args.stats,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
