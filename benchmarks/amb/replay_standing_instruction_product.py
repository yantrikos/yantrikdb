#!/usr/bin/env python3
"""Replay the frozen standing-instruction arm through persisted facets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

try:
    from .audit_knowledge_update_gold import iter_turns, load_conversations
    from .build_standing_instruction_contexts import (
        TRAILING_LINK_RE,
        cap_whole_blocks,
        load_rows,
        render_instruction_panel,
    )
    from .reorder_speaker_first_contexts import split_memory_blocks
except ImportError:  # pragma: no cover - direct script execution
    from audit_knowledge_update_gold import iter_turns, load_conversations
    from build_standing_instruction_contexts import (
        TRAILING_LINK_RE,
        cap_whole_blocks,
        load_rows,
        render_instruction_panel,
    )
    from reorder_speaker_first_contexts import split_memory_blocks


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(value: object, newline: str = "\n") -> bytes:
    text = json.dumps(value, indent=2)
    if newline != "\n":
        text = text.replace("\n", newline)
    return text.encode("utf-8")


def _first_block(context: str) -> str:
    blocks = split_memory_blocks(context)
    return blocks[0] if blocks else ""


def _remove_store(path: Path) -> None:
    for suffix in ("", "-shm", "-wal"):
        candidate = Path(f"{path}{suffix}")
        if candidate.exists():
            candidate.unlink()


def _selected_rows(rows: list[dict], category: str) -> list[dict]:
    if category == "all":
        return rows
    return [
        row
        for row in rows
        if (row.get("meta") or {}).get("question_category") == category
    ]


def _ingest_and_extract(
    db: object,
    conversations: dict[str, dict],
    conversation_ids: list[str],
) -> tuple[dict[str, dict[str, object]], dict[str, dict], int]:
    """Use raw turns only; queries, gold answers, and reference panels stay out."""
    source_by_rid: dict[str, dict[str, object]] = {}
    audits: dict[str, dict] = {}
    ingested = 0
    for conversation_id in conversation_ids:
        conversation = conversations.get(conversation_id)
        if conversation is None:
            raise ValueError(f"missing conversation {conversation_id!r}")
        for position, turn in enumerate(
            iter_turns(conversation.get("chat") or []), start=1
        ):
            text = TRAILING_LINK_RE.sub("", str(turn.get("content") or "")).strip()
            if not text:
                continue
            role = str(turn.get("role") or "unknown").casefold()
            # The benchmark provider supplies embeddings at ingestion time.
            # A fixed valid vector is sufficient here because facet detection
            # reads text/provenance, while newly synthesized facets still use
            # the engine's configured embedder through record_synthesis.
            rid = db.record(
                text,
                embedding=[1.0] + [0.0] * 255,
                metadata={
                    "conversation_id": conversation_id,
                    "source_turn": turn.get("id"),
                },
                namespace=conversation_id,
                source=role,
                created_at=1_700_000_000.0 + position,
            )
            source_by_rid[rid] = {
                "conversation_id": conversation_id,
                "turn": turn.get("id"),
                "role": role,
            }
            ingested += 1
        audits[conversation_id] = db.extract_standing_instructions(
            namespace=conversation_id,
            dry_run=False,
        )
    return source_by_rid, audits, ingested


def _source_map_from_store(path: Path) -> tuple[dict[str, dict[str, object]], int, int]:
    """Recover the ingestion manifest for a recall-only verification pass."""
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT rid, namespace, source, metadata FROM memories "
            "WHERE synthesis_axis IS NULL"
        ).fetchall()
        accepted_facets = int(
            connection.execute(
                "SELECT COUNT(*) FROM memories "
                "WHERE synthesis_axis = 'standing_instruction'"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    source_by_rid = {}
    for rid, namespace, source, raw_metadata in rows:
        metadata = json.loads(raw_metadata or "{}")
        source_by_rid[rid] = {
            "conversation_id": str(metadata.get("conversation_id") or namespace),
            "turn": metadata.get("source_turn"),
            "role": source,
        }
    return source_by_rid, len(rows), accepted_facets


def _records_from_lane(
    lane: dict,
    source_by_rid: dict[str, dict[str, object]],
) -> list[dict]:
    if lane.get("omitted") != 0:
        raise AssertionError(f"facet lane unexpectedly omitted rows: {lane!r}")
    records = []
    for facet in lane.get("facets") or []:
        sources = [source_by_rid[rid] for rid in facet.get("source_rids") or []]
        if not sources:
            raise AssertionError(f"facet has no resolvable source: {facet!r}")
        if any(source["role"] != "user" for source in sources):
            raise AssertionError(f"facet has non-user evidence: {facet!r}")
        records.append(
            {
                "turn": min(source["turn"] for source in sources),
                "text": facet["text"],
            }
        )
    return records


def replay(args: argparse.Namespace) -> tuple[dict, bytes]:
    sys.path.insert(0, str(args.yantrikdb_python.resolve()))
    from memory_bench.utils import count_tokens
    from yantrikdb import YantrikDB

    control_rows = _selected_rows(load_rows(args.results), args.category)
    reference_rows = load_rows(args.reference_treatment)
    reference_by_id = {str(row["query_id"]): row for row in reference_rows}
    query_ids = [str(row["query_id"]) for row in control_rows]
    if query_ids != [str(row["query_id"]) for row in reference_rows]:
        raise ValueError("control and frozen treatment query order differs")

    conversation_ids = list(
        dict.fromkeys(
            str((row.get("meta") or {}).get("conversation_id") or "")
            for row in control_rows
        )
    )
    if any(not conversation_id for conversation_id in conversation_ids):
        raise ValueError("control row lacks conversation_id")

    if args.reuse_store and not args.db.exists():
        raise FileNotFoundError(f"store does not exist: {args.db}")
    if not args.reuse_store and args.db.exists() and not args.overwrite:
        raise FileExistsError(f"store already exists: {args.db}")
    if not args.reuse_store and args.overwrite:
        _remove_store(args.db)
    args.db.parent.mkdir(parents=True, exist_ok=True)

    if args.reuse_store:
        source_by_rid, ingested, accepted_facets = _source_map_from_store(args.db)
    else:
        conversations = load_conversations(args.documents)
        db = YantrikDB.with_default(str(args.db))
        try:
            source_by_rid, audits, ingested = _ingest_and_extract(
                db, conversations, conversation_ids
            )
        finally:
            db.close()
        accepted_facets = sum(int(audit["accepted"]) for audit in audits.values())

    # This reopen is deliberate: the acceptance claim is about persisted
    # product facets, not objects still resident in the extraction process.
    db = YantrikDB.with_default(str(args.db))
    product_rows = []
    mismatches = []
    try:
        for row in control_rows:
            query_id = str(row["query_id"])
            conversation_id = str((row.get("meta") or {})["conversation_id"])
            lane = db.recall_facets(namespace=conversation_id, limit=10_000)
            records = _records_from_lane(lane, source_by_rid)
            reference_context = str(row.get("context") or "")
            reference_tokens = count_tokens(reference_context)
            candidate = render_instruction_panel(records) + reference_context
            product_context, _, _ = cap_whole_blocks(
                candidate, reference_tokens, count_tokens
            )
            product_rows.append({"query_id": query_id, "context": product_context})

            expected_context = str(reference_by_id[query_id].get("context") or "")
            expected_panel = _first_block(expected_context)
            actual_panel = _first_block(product_context)
            if expected_context != product_context:
                mismatches.append(
                    {
                        "query_id": query_id,
                        "conversation_id": conversation_id,
                        "expected_panel_sha256": _sha256_bytes(
                            expected_panel.encode("utf-8")
                        ),
                        "actual_panel_sha256": _sha256_bytes(
                            actual_panel.encode("utf-8")
                        ),
                        "panel_exact": expected_panel == actual_panel,
                        "context_exact": False,
                        "facet_count": len(records),
                    }
                )
    finally:
        db.close()

    paired_payload = {"results": product_rows}
    reference_paired_bytes = args.reference_paired.read_bytes()
    newline = "\r\n" if b"\r\n" in reference_paired_bytes else "\n"
    paired_bytes = _json_bytes(paired_payload, newline)
    reference_paired_sha = _sha256_bytes(reference_paired_bytes)
    product_paired_sha = _sha256_bytes(paired_bytes)
    report = {
        "protocol": "standing-user-instruction-product-replay-v1",
        "product_path": {
            "raw_turn_ingestion": True,
            "persisted_facet_extraction": True,
            "store_reopened_before_recall": True,
            "recall_facets_used": True,
            "query_or_gold_used_during_extraction": False,
            "external_calls": 0,
        },
        "category": args.category,
        "rows": len(product_rows),
        "conversations": len(conversation_ids),
        "ingested_turns": ingested,
        "accepted_facets": accepted_facets,
        "ordered_query_ids_exact": query_ids
        == [row["query_id"] for row in product_rows],
        "exact_contexts": len(product_rows) - len(mismatches),
        "mismatch_count": len(mismatches),
        "paired_artifact_sha256": product_paired_sha,
        "reference_paired_sha256": reference_paired_sha,
        "paired_artifact_exact": product_paired_sha == reference_paired_sha,
        "source_sha256": {
            "results": _sha256_file(args.results),
            "documents": _sha256_file(args.documents),
            "reference_treatment": _sha256_file(args.reference_treatment),
        },
        "mismatches": mismatches,
    }
    return report, paired_bytes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--reference-treatment", type=Path, required=True)
    parser.add_argument("--reference-paired", type=Path, required=True)
    parser.add_argument("--yantrikdb-python", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired-output", type=Path, required=True)
    parser.add_argument("--category", default="instruction_following")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-store", action="store_true")
    args = parser.parse_args()

    report, paired_bytes = replay(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.paired_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.paired_output.write_bytes(paired_bytes)
    print(json.dumps(report, indent=2))
    return 0 if report["paired_artifact_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
