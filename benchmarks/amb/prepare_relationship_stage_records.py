"""Freeze answer-sized relationship stages from one globally selected thread."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from .global_organizer_probe import _chat_json
    from .prepare_contextual_synthesis_pair import (
        render_relationship_thread,
        select_relationship_thread,
    )
    from .replay_contextual_synthesis import _ollama_model_identity
except ImportError:  # pragma: no cover - direct script execution
    from global_organizer_probe import _chat_json
    from prepare_contextual_synthesis_pair import (
        render_relationship_thread,
        select_relationship_thread,
    )
    from replay_contextual_synthesis import _ollama_model_identity


PROTOCOL = "relationship-stage-records-global-v1"
SCHEMA = {
    "type": "object",
    "properties": {
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_group": {"type": "string"},
                    "item": {"type": "string"},
                },
                "required": ["source_group", "item"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["records"],
    "additionalProperties": False,
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )


def _normalize_host(host: str) -> str:
    if not host.startswith("http"):
        host = f"http://{host}"
    for wildcard in ("//0.0.0.0", "//[::]", "//::"):
        host = host.replace(wildcard, "//127.0.0.1")
    return host.rstrip("/")


def _load_query(path: Path, query_id: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("query_id") == query_id:
        return payload
    matches = [
        row for row in payload.get("queries") or [] if row.get("query_id") == query_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {query_id!r} row, found {len(matches)}")
    return matches[0]


def _ordered_groups(artifact: dict) -> tuple[str, list[dict]]:
    thread = select_relationship_thread(artifact)
    ordered = sorted(
        thread["groups"].items(),
        key=lambda entry: (
            min(
                row.get("created_at")
                if row.get("created_at") is not None
                else float("inf")
                for row in entry[1]
            ),
            min(
                row.get("turn") if row.get("turn") is not None else float("inf")
                for row in entry[1]
            ),
            str(entry[0]),
        ),
    )
    groups = []
    for group_key, rows in ordered:
        rows = sorted(
            rows,
            key=lambda row: (
                row.get("turn") if row.get("turn") is not None else float("inf"),
                row.get("identity") or "",
            ),
        )
        source_group = str(group_key[1])
        groups.append(
            {
                "source_group": source_group,
                "source_kind": group_key[0],
                "first_mention_at": min(
                    (
                        row["created_at"]
                        for row in rows
                        if row.get("created_at") is not None
                    ),
                    default=None,
                ),
                "first_mention_turn": min(
                    (row["turn"] for row in rows if row.get("turn") is not None),
                    default=None,
                ),
                "evidence_ids": [row.get("identity") for row in rows],
                "evidence_turns": [
                    row.get("turn") for row in rows if row.get("turn") is not None
                ],
                "evidence": [
                    {
                        "evidence_id": row.get("identity"),
                        "turn": row.get("turn"),
                        "created_at": row.get("created_at"),
                        "text": row.get("text") or "",
                    }
                    for row in rows
                ],
            }
        )
    return thread["anchor"], groups


def build_prompt(anchor: str, groups: list[dict]) -> str:
    input_groups = [
        {
            "source_group": group["source_group"],
            "evidence": group["evidence"],
        }
        for group in groups
    ]
    return (
        "You organize durable memory before any future question is known. "
        "The input is synthetic benchmark evidence for one relationship thread. "
        f"Create exactly one concise stage record for {anchor.title()} per source "
        "conversation. Merge every evidence row within that source conversation, "
        "but never merge across source_group values. Preserve the concrete details "
        "that distinguish the stage: meeting medium or location, named work, "
        "recommendations or feedback, decisions, outcomes, follow-up plans, and "
        "explicit dates. Do not infer unsupported facts. Each item must be a single "
        "sentence of at most 55 words. Copy each source_group exactly once. Return "
        "only one JSON object with exactly this shape and no other keys: "
        '{"records":[{"source_group":"COPY_FROM_INPUT","item":"ONE '
        'SENTENCE"}]}. Do not answer or refer to any question.\n\nSOURCE GROUPS:\n'
        + json.dumps(input_groups, indent=2, ensure_ascii=False)
    )


def materialize_records(response: dict, groups: list[dict]) -> list[dict]:
    expected = {group["source_group"]: group for group in groups}
    raw_records = response.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("response records must be a list")
    if any(not isinstance(record, dict) for record in raw_records):
        raise ValueError("every response record must be an object")
    actual_groups = [record.get("source_group") for record in raw_records]
    if len(actual_groups) != len(set(actual_groups)):
        raise ValueError("response contains duplicate source groups")
    if set(actual_groups) != set(expected):
        raise ValueError(
            f"source group mismatch: expected {sorted(expected)}, "
            f"got {sorted(str(value) for value in actual_groups)}"
        )

    materialized = []
    by_group = {record["source_group"]: record for record in raw_records}
    for group in groups:
        raw_item = by_group[group["source_group"]].get("item")
        item = " ".join(str(raw_item or "").split())
        if not item:
            raise ValueError(f"empty item for {group['source_group']}")
        if len(item.split()) > 55:
            raise ValueError(f"item exceeds 55 words for {group['source_group']}")
        materialized.append(
            {
                "source_group": group["source_group"],
                "source_kind": group["source_kind"],
                "item": item,
                "first_mention_at": group["first_mention_at"],
                "first_mention_turn": group["first_mention_turn"],
                "evidence_ids": group["evidence_ids"],
                "evidence_turns": group["evidence_turns"],
            }
        )
    return materialized


def _date_text(timestamp: float | None) -> str:
    if timestamp is None:
        return "date unknown"
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()


def render_records(anchor: str, records: list[dict]) -> str:
    memories = []
    for index, record in enumerate(records, 1):
        stamp = _date_text(record["first_mention_at"])
        turn = record["first_mention_turn"]
        if turn is not None:
            stamp += f" | Turn {turn}"
        memories.append(
            f"## Memory {index}\n"
            f"[{stamp}] Relationship stage with {anchor.title()}: "
            f"{record['item']}\n"
            f"Source conversation: {record['source_group']} | "
            f"Evidence turns: {', '.join(map(str, record['evidence_turns']))}"
        )
    return "\n\n".join(memories)


def build_preflight(artifact: dict, query_id: str, model: str) -> dict:
    _, selector_audit = render_relationship_thread(artifact)
    anchor, groups = _ordered_groups(artifact)
    prompt = build_prompt(anchor, groups)
    return {
        "protocol": PROTOCOL,
        "status": "dry_run",
        "query_id": query_id,
        "model": model,
        "model_calls": 0,
        "query_exposed_to_synthesis": False,
        "anchor": anchor,
        "source_group_count": len(groups),
        "evidence_row_count": sum(len(group["evidence"]) for group in groups),
        "selection_sha256": selector_audit["selection_sha256"],
        "request_sha256": _sha256_json({"prompt": prompt, "schema": SCHEMA}),
        "prompt": prompt,
        "schema": SCHEMA,
        "groups": groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-selection-sha256")
    parser.add_argument("--expect-request-sha256")
    parser.add_argument("--model", default="deepseek-v4-flash:0731-cloud")
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--replay-response",
        type=Path,
        help="reuse a prior raw artifact's parsed_response without a model call",
    )
    args = parser.parse_args()

    artifact = _load_query(args.source, args.query_id)
    preflight = build_preflight(artifact, args.query_id, args.model)
    if (
        args.expect_selection_sha256
        and preflight["selection_sha256"] != args.expect_selection_sha256
    ):
        raise ValueError(
            "selection hash mismatch: expected "
            f"{args.expect_selection_sha256}, got {preflight['selection_sha256']}"
        )
    if (
        args.expect_request_sha256
        and preflight["request_sha256"] != args.expect_request_sha256
    ):
        raise ValueError(
            "request hash mismatch: expected "
            f"{args.expect_request_sha256}, got {preflight['request_sha256']}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        args.output.write_text(
            json.dumps(preflight, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return 0

    ollama_host = _normalize_host(args.ollama_host)
    if args.replay_response:
        replay = json.loads(args.replay_response.read_text(encoding="utf-8"))
        if replay.get("request_sha256") != preflight["request_sha256"]:
            raise ValueError("replayed response does not match the frozen request")
        response = replay.get("parsed_response")
        if not isinstance(response, dict):
            raise ValueError("replayed artifact has no parsed_response object")
        raw_path = args.replay_response
        execution_model_calls = 0
    else:
        response, raw_response = _chat_json(
            host=ollama_host,
            model=args.model,
            prompt=preflight["prompt"],
            schema=SCHEMA,
            num_predict=1600,
            timeout=args.timeout,
        )
        raw_path = args.output.with_suffix(".raw.json")
        raw_path.write_text(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "request_sha256": preflight["request_sha256"],
                    "parsed_response": response,
                    "raw_response": raw_response,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        execution_model_calls = 1
    records = materialize_records(response, preflight["groups"])
    context = render_records(preflight["anchor"], records)
    model_identity = _ollama_model_identity(args.model, ollama_host)
    result = {
        "results": [
            {
                "query_id": args.query_id,
                "context": context,
                "audit": {
                    "protocol": PROTOCOL,
                    "model": args.model,
                    "model_identity": model_identity,
                    "generation_model_calls": 1,
                    "execution_model_calls": execution_model_calls,
                    "query_exposed_to_synthesis": False,
                    "anchor": preflight["anchor"],
                    "source_group_count": len(records),
                    "evidence_row_count": preflight["evidence_row_count"],
                    "selection_sha256": preflight["selection_sha256"],
                    "request_sha256": preflight["request_sha256"],
                    "response_sha256": _sha256_json(response),
                    "records": records,
                },
            }
        ],
        "raw_response_file": str(raw_path),
    }
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
