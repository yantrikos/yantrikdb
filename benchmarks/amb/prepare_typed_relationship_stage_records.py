"""Freeze typed relationship stages while preserving speaker perspective."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

try:
    from .global_organizer_probe import _chat_json
    from .prepare_relationship_stage_records import (
        _date_text,
        _load_query,
        _normalize_host,
        _ordered_groups,
        _sha256_json,
    )
    from .prepare_contextual_synthesis_pair import render_relationship_thread
    from .replay_contextual_synthesis import _ollama_model_identity
except ImportError:  # pragma: no cover - direct script execution
    from global_organizer_probe import _chat_json
    from prepare_relationship_stage_records import (
        _date_text,
        _load_query,
        _normalize_host,
        _ordered_groups,
        _sha256_json,
    )
    from prepare_contextual_synthesis_pair import render_relationship_thread
    from replay_contextual_synthesis import _ollama_model_identity


PROTOCOL = "typed-relationship-stage-records-global-v1"
FACETS = ("goal_or_concern", "event", "decision", "outcome", "follow_up")
_FIRST_PERSON_RE = re.compile(r"\b(?:i(?:['’](?:m|ve|d|ll))?|my|me)\b", re.IGNORECASE)
SCHEMA = {
    "type": "object",
    "properties": {
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_group": {"type": "string"},
                    "speaker_perspective": {
                        "type": "string",
                        "enum": ["first_person"],
                    },
                    **{facet: {"type": "string"} for facet in FACETS},
                },
                "required": [
                    "source_group",
                    "speaker_perspective",
                    *FACETS,
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["records"],
    "additionalProperties": False,
}


def build_typed_prompt(anchor: str, groups: list[dict]) -> str:
    input_groups = [
        {
            "source_group": group["source_group"],
            "evidence": group["evidence"],
        }
        for group in groups
    ]
    shape = {
        "records": [
            {
                "source_group": "COPY_FROM_INPUT",
                "speaker_perspective": "first_person",
                "goal_or_concern": "I/my clause",
                "event": "event facts",
                "decision": "decision or empty string",
                "outcome": "outcome or empty string",
                "follow_up": "follow-up or empty string",
            }
        ]
    }
    return (
        "You organize durable memory before any future question is known. "
        "The input is synthetic benchmark evidence for one relationship thread. "
        f"Create exactly one typed stage record for {anchor.title()} per source "
        "conversation. Merge every evidence row within that source conversation, "
        "but never merge across source_group values. Preserve distinct information "
        "instead of compressing it into one generic summary. Put the memory owner's "
        "expressed goal, worry, request, or concern in goal_or_concern using I/my "
        "language; never call the owner 'the user'. Put concrete occurrences, dates, "
        "locations, media, named work, recommendations, and feedback in event. Put "
        "choices or prioritization in decision, results or feelings in outcome, and "
        "planned next actions or meetings in follow_up. Use an empty string only when "
        "the evidence has no information for that facet. Each non-empty facet must "
        "be at most 30 words. Do not infer unsupported facts. Copy every source_group "
        "exactly once. Return only one JSON object with exactly this shape and no "
        f"other keys: {json.dumps(shape, ensure_ascii=False)}. Do not answer or refer "
        "to any question.\n\nSOURCE GROUPS:\n"
        + json.dumps(input_groups, indent=2, ensure_ascii=False)
    )


def materialize_typed_records(response: dict, groups: list[dict]) -> list[dict]:
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

    by_group = {record["source_group"]: record for record in raw_records}
    materialized = []
    for group in groups:
        raw = by_group[group["source_group"]]
        if raw.get("speaker_perspective") != "first_person":
            raise ValueError(f"invalid speaker perspective for {group['source_group']}")
        facets = {}
        for facet in FACETS:
            if not isinstance(raw.get(facet), str):
                raise ValueError(
                    f"{facet} must be a string for {group['source_group']}"
                )
            value = " ".join(raw[facet].split())
            if len(value.split()) > 30:
                raise ValueError(
                    f"{facet} exceeds 30 words for {group['source_group']}"
                )
            if "the user" in value.casefold() or "user's" in value.casefold():
                raise ValueError(
                    f"{facet} loses first-person perspective for "
                    f"{group['source_group']}"
                )
            facets[facet] = value
        goal = facets["goal_or_concern"]
        if not goal or not _FIRST_PERSON_RE.search(goal):
            raise ValueError(
                f"goal_or_concern must preserve first person for "
                f"{group['source_group']}"
            )
        if sum(bool(value) for value in facets.values()) < 2:
            raise ValueError(f"typed record is too sparse for {group['source_group']}")
        materialized.append(
            {
                "source_group": group["source_group"],
                "source_kind": group["source_kind"],
                "speaker_perspective": "first_person",
                **facets,
                "first_mention_at": group["first_mention_at"],
                "first_mention_turn": group["first_mention_turn"],
                "evidence_ids": group["evidence_ids"],
                "evidence_turns": group["evidence_turns"],
            }
        )
    return materialized


def render_typed_records(anchor: str, records: list[dict]) -> str:
    labels = {
        "goal_or_concern": "My goal or concern",
        "event": "What happened",
        "decision": "My decision",
        "outcome": "Outcome",
        "follow_up": "Next step",
    }
    memories = []
    for index, record in enumerate(records, 1):
        stamp = _date_text(record["first_mention_at"])
        turn = record["first_mention_turn"]
        if turn is not None:
            stamp += f" | Turn {turn}"
        lines = [
            f"## Memory {index}",
            f"[{stamp}] Relationship stage with {anchor.title()}",
        ]
        lines.extend(
            f"{labels[facet]}: {record[facet]}" for facet in FACETS if record[facet]
        )
        lines.append(
            f"Source conversation: {record['source_group']} | "
            f"Evidence turns: {', '.join(map(str, record['evidence_turns']))}"
        )
        memories.append("\n".join(lines))
    return "\n\n".join(memories)


def build_typed_preflight(artifact: dict, query_id: str, model: str) -> dict:
    _, selector_audit = render_relationship_thread(artifact)
    anchor, groups = _ordered_groups(artifact)
    prompt = build_typed_prompt(anchor, groups)
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
    parser.add_argument("--replay-response", type=Path)
    args = parser.parse_args()

    artifact = _load_query(args.source, args.query_id)
    preflight = build_typed_preflight(artifact, args.query_id, args.model)
    for label, expected, actual in (
        (
            "selection",
            args.expect_selection_sha256,
            preflight["selection_sha256"],
        ),
        ("request", args.expect_request_sha256, preflight["request_sha256"]),
    ):
        if expected and expected != actual:
            raise ValueError(
                f"{label} hash mismatch: expected {expected}, got {actual}"
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
            num_predict=2400,
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

    records = materialize_typed_records(response, preflight["groups"])
    context = render_typed_records(preflight["anchor"], records)
    result = {
        "results": [
            {
                "query_id": args.query_id,
                "context": context,
                "audit": {
                    "protocol": PROTOCOL,
                    "model": args.model,
                    "model_identity": _ollama_model_identity(args.model, ollama_host),
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
