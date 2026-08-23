#!/usr/bin/env python3
"""Freeze additive, complete standing-instruction contexts from product facets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path

try:
    from .audit_knowledge_update_gold import iter_turns, load_conversations
    from .reorder_speaker_first_contexts import split_memory_blocks
except ImportError:  # pragma: no cover - direct script execution
    from audit_knowledge_update_gold import iter_turns, load_conversations
    from reorder_speaker_first_contexts import split_memory_blocks


PROTOCOL = "complete-standing-facet-composition-v2"
MAX_FACET_TOKENS = 256
EXPECTED_ROWS = 400
EXPECTED_INSTRUCTION_TARGETS = 40
TRAILING_LINK_RE = re.compile(r"\s*->->.*$")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2).encode("utf-8")


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, list) else payload.get("results") or []


def overlay_control_contexts(
    rows: list[dict], control_rows: list[dict]
) -> list[dict]:
    """Join frozen control contexts onto metadata-rich rows in exact order."""
    row_ids = [str(row.get("query_id") or "") for row in rows]
    control_ids = [str(row.get("query_id") or "") for row in control_rows]
    if not row_ids or any(not query_id for query_id in row_ids):
        raise ValueError("metadata results contain a missing query_id")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("metadata results contain duplicate query_ids")
    if row_ids != control_ids:
        raise ValueError("metadata results and control contexts differ in query order")
    return [
        {**row, "context": str(control.get("context") or "")}
        for row, control in zip(rows, control_rows, strict=True)
    ]


def normalize_directive(text: str) -> str:
    return " ".join(text.split()).casefold()


def select_facets(facets: list[dict]) -> list[dict]:
    """Return the complete verified lane in deterministic first-mention order."""
    return sorted(
        facets,
        key=lambda row: (
            float(row.get("first_mention_at") or 0.0),
            str(row.get("rid") or ""),
        ),
    )


def render_facet_panel(facets: list[dict]) -> str:
    lines = [
        f"- [Turn {facet['turn']}] User: {facet['text']}" for facet in facets
    ]
    return (
        "## Memory 0\n"
        "Standing user instructions (authoritative user statements):\n"
        + "\n".join(lines)
        + "\n\n"
    )


def build_context_rows(
    rows: list[dict],
    facets_by_namespace: dict[str, dict],
    token_counter: Callable[[str], int],
) -> tuple[dict, dict]:
    """Compose the treatment and return its pre-call invariant audit."""
    treatment_rows = []
    row_audits = []
    target_rows = 0
    targets_retained = 0
    max_extra_tokens = 0
    facet_counts = []

    for row in rows:
        query_id = str(row.get("query_id") or "")
        if not query_id:
            raise ValueError("row lacks query_id")
        metadata = row.get("meta") or {}
        namespace = str(metadata.get("conversation_id") or "")
        if not namespace:
            raise ValueError(f"row {query_id!r} lacks conversation_id")
        lane = facets_by_namespace.get(namespace)
        if lane is None:
            raise ValueError(f"row {query_id!r} has no facet lane snapshot")
        if int(lane.get("omitted") or 0) != 0:
            raise ValueError(
                f"row {query_id!r} facet inventory is incomplete: {lane!r}"
            )
        facets = list(lane.get("facets") or [])
        selected = select_facets(facets)
        if not selected:
            raise ValueError(f"row {query_id!r} has an empty verified facet lane")
        facet_counts.append(len(selected))

        panel = render_facet_panel(selected)
        panel_tokens = token_counter(panel)
        if panel_tokens > MAX_FACET_TOKENS:
            raise ValueError(
                f"row {query_id!r} facet panel uses {panel_tokens} tokens; "
                f"limit is {MAX_FACET_TOKENS}"
            )
        reference_context = str(row.get("context") or "")
        treatment_context = panel + reference_context
        ordinary_suffix = treatment_context[len(panel) :]
        if ordinary_suffix != reference_context:
            raise AssertionError(f"row {query_id!r} changed ordinary context bytes")
        if split_memory_blocks(ordinary_suffix) != split_memory_blocks(reference_context):
            raise AssertionError(f"row {query_id!r} changed ordinary memory blocks")

        target = str(metadata.get("instruction_being_tested") or "").strip()
        target_retained = None
        if target:
            target_rows += 1
            target_retained = normalize_directive(target) in {
                normalize_directive(str(facet.get("text") or ""))
                for facet in selected
            }
            targets_retained += int(target_retained)

        max_extra_tokens = max(max_extra_tokens, panel_tokens)
        treatment_rows.append({"query_id": query_id, "context": treatment_context})
        row_audits.append(
            {
                "query_id": query_id,
                "namespace": namespace,
                "available_facets": len(facets),
                "selected_facets": len(selected),
                "selected_rids": [str(facet.get("rid") or "") for facet in selected],
                "selected_turns": [facet.get("turn") for facet in selected],
                "facet_tokens": panel_tokens,
                "ordinary_context_sha256": _sha256_bytes(
                    reference_context.encode("utf-8")
                ),
                "ordinary_context_exact": ordinary_suffix == reference_context,
                "target_retained": target_retained,
            }
        )

    treatment = {"results": treatment_rows}
    audit = {
        "protocol": PROTOCOL,
        "rows": len(rows),
        "max_facet_tokens_allowed": MAX_FACET_TOKENS,
        "max_facet_tokens_observed": max_extra_tokens,
        "complete_lane_rows": sum(
            row["selected_facets"] == row["available_facets"]
            for row in row_audits
        ),
        "min_facets_per_row": min(facet_counts, default=0),
        "max_facets_per_row": max(facet_counts, default=0),
        "ordinary_contexts_exact": sum(
            row["ordinary_context_exact"] for row in row_audits
        ),
        "instruction_target_rows": target_rows,
        "instruction_targets_retained": targets_retained,
        "selection_fields": ["facet.first_mention_at", "facet.rid"],
        "selection_uses_query": False,
        "selection_uses_category": False,
        "selection_uses_gold_rubric_answer_or_score": False,
        "target_metadata_used_for_abort_audit_only": True,
        "row_audits": row_audits,
    }
    return treatment, audit


def validate_full400_preflight(audit: dict) -> None:
    errors = []
    if audit["rows"] != EXPECTED_ROWS:
        errors.append(f"expected {EXPECTED_ROWS} rows, got {audit['rows']}")
    if audit["instruction_target_rows"] != EXPECTED_INSTRUCTION_TARGETS:
        errors.append(
            "expected "
            f"{EXPECTED_INSTRUCTION_TARGETS} instruction targets, got "
            f"{audit['instruction_target_rows']}"
        )
    if audit["instruction_targets_retained"] != EXPECTED_INSTRUCTION_TARGETS:
        errors.append(
            "canonical target retention is "
            f"{audit['instruction_targets_retained']}/{EXPECTED_INSTRUCTION_TARGETS}"
        )
    if audit["complete_lane_rows"] != audit["rows"]:
        errors.append("one or more rows omitted a verified standing facet")
    if audit["ordinary_contexts_exact"] != audit["rows"]:
        errors.append("one or more ordinary contexts changed")
    if audit["max_facet_tokens_observed"] > MAX_FACET_TOKENS:
        errors.append("facet token ceiling exceeded")
    if errors:
        raise RuntimeError("pre-call facet gate failed:\n- " + "\n- ".join(errors))


def _remove_store(path: Path) -> None:
    for suffix in ("", "-shm", "-wal"):
        candidate = Path(f"{path}{suffix}")
        if candidate.exists():
            candidate.unlink()


def _ingest_and_extract(db: object, conversations: dict, namespaces: list[str]) -> dict:
    audits = {}
    placeholder_dim = len(db.embed("__facet_source_placeholder__"))
    placeholder_embedding = [1.0] + [0.0] * (placeholder_dim - 1)
    for namespace in namespaces:
        conversation = conversations.get(namespace)
        if conversation is None:
            raise ValueError(f"missing conversation {namespace!r}")
        for position, turn in enumerate(
            iter_turns(conversation.get("chat") or []), start=1
        ):
            text = TRAILING_LINK_RE.sub("", str(turn.get("content") or "")).strip()
            if not text:
                continue
            db.record(
                text,
                embedding=placeholder_embedding,
                metadata={
                    "conversation_id": namespace,
                    "source_turn": turn.get("id"),
                },
                namespace=namespace,
                source=str(turn.get("role") or "unknown").casefold(),
                created_at=1_700_000_000.0 + position,
            )
        audits[namespace] = db.extract_standing_instructions(
            namespace=namespace, dry_run=False
        )
    return audits


def _source_map(path: Path) -> dict[str, dict]:
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT rid, namespace, source, metadata FROM memories "
            "WHERE synthesis_axis IS NULL"
        ).fetchall()
    finally:
        connection.close()
    out = {}
    for rid, namespace, source, raw_metadata in rows:
        metadata = json.loads(raw_metadata or "{}")
        out[rid] = {
            "namespace": namespace,
            "role": source,
            "turn": metadata.get("source_turn"),
        }
    return out


def _facet_snapshots(db: object, namespaces: list[str], source_map: dict) -> dict:
    snapshots = {}
    for namespace in namespaces:
        lane = db.recall_facets(namespace=namespace, limit=10_000)
        facets = []
        for facet in lane.get("facets") or []:
            sources = [source_map[rid] for rid in facet.get("source_rids") or []]
            if not sources or any(source["role"] != "user" for source in sources):
                raise AssertionError(f"facet lacks verified user evidence: {facet!r}")
            facets.append(
                {
                    **facet,
                    "turn": min(source["turn"] for source in sources),
                }
            )
        snapshots[namespace] = {
            "facets": facets,
            "omitted": int(lane.get("omitted") or 0),
        }
    return snapshots


def _control_payload(rows: list[dict]) -> dict:
    return {
        "results": [
            {"query_id": str(row["query_id"]), "context": str(row.get("context") or "")}
            for row in rows
        ]
    }


def main() -> int:
    from memory_bench.utils import count_tokens

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--control-contexts", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--yantrikdb-python", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", default="deepseek-v4-flash:0731-cloud")
    args = parser.parse_args()

    rows = overlay_control_contexts(
        load_rows(args.results), load_rows(args.control_contexts)
    )
    namespaces = list(
        dict.fromkeys(
            str((row.get("meta") or {}).get("conversation_id") or "")
            for row in rows
        )
    )
    if any(not namespace for namespace in namespaces):
        raise ValueError("one or more rows lack conversation_id")
    if args.db.exists() and not args.overwrite:
        raise FileExistsError(f"store already exists: {args.db}")
    if args.overwrite:
        _remove_store(args.db)
    args.db.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.yantrikdb_python.resolve()))
    from yantrikdb import YantrikDB

    conversations = load_conversations(args.documents)
    db = YantrikDB.with_default(str(args.db))
    try:
        extraction_audits = _ingest_and_extract(db, conversations, namespaces)
    finally:
        db.close()

    source_map = _source_map(args.db)
    db = YantrikDB.with_default(str(args.db))
    try:
        snapshots = _facet_snapshots(db, namespaces, source_map)
        treatment, audit = build_context_rows(
            rows, snapshots, count_tokens
        )
    finally:
        db.close()
    validate_full400_preflight(audit)

    # A second fresh open must reconstruct the same artifact exactly.
    db = YantrikDB.with_default(str(args.db))
    try:
        replay_snapshots = _facet_snapshots(db, namespaces, source_map)
        replay_treatment, replay_audit = build_context_rows(
            rows, replay_snapshots, count_tokens
        )
    finally:
        db.close()
    replay_exact = _json_bytes(replay_treatment) == _json_bytes(treatment)
    replay_selection_exact = [
        row["selected_rids"] for row in replay_audit["row_audits"]
    ] == [row["selected_rids"] for row in audit["row_audits"]]
    if not replay_exact or not replay_selection_exact:
        raise RuntimeError("fresh-open product replay did not reconstruct the artifact")

    control = _control_payload(rows)
    control_path = args.out_dir / "control.json"
    treatment_path = args.out_dir / "selective-facets.json"
    preflight_path = args.out_dir / "preflight.json"
    control_path.write_bytes(_json_bytes(control))
    treatment_path.write_bytes(_json_bytes(treatment))

    query_ids = [str(row["query_id"]) for row in rows]
    preflight = {
        **audit,
        "status": "passed",
        "product_path": {
            "raw_turn_ingestion": True,
            "persisted_facet_extraction": True,
            "store_reopened_before_selection": True,
            "query_independent_complete_lane": True,
            "second_fresh_open_replay_exact": replay_exact,
            "second_fresh_open_selection_exact": replay_selection_exact,
            "query_or_gold_used_during_extraction": False,
        },
        "extraction_audits": extraction_audits,
        "source_sha256": {
            "results": _sha256_file(args.results),
            "control_contexts": _sha256_file(args.control_contexts),
            "documents": _sha256_file(args.documents),
        },
        "ordered_query_ids_sha256": _sha256_bytes(
            json.dumps(query_ids, separators=(",", ":")).encode("utf-8")
        ),
        "arms": {
            "control": {
                "file": control_path.name,
                "sha256": _sha256_file(control_path),
            },
            "treatment": {
                "file": treatment_path.name,
                "sha256": _sha256_file(treatment_path),
            },
        },
        "external_evaluation": {
            "model": args.model,
            "answer_repeats": 1,
            "judge_repeats": 1,
            "answer_calls": len(rows) * 2,
            "judge_calls": len(rows) * 2,
            "synthetic_benchmark_data_only": True,
            "real_companion_memories_included": False,
        },
    }
    preflight_path.write_bytes(_json_bytes(preflight))
    print(
        json.dumps(
            {
                key: value
                for key, value in preflight.items()
                if key not in {"row_audits", "extraction_audits"}
            },
            indent=2,
        )
    )
    print(f"preflight_sha256={_sha256_file(preflight_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
