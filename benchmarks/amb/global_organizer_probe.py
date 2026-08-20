"""Probe query-free global organization over persisted AMB atomic items.

The benchmark query and gold answer are deliberately withheld from the model.
Returned evidence IDs are resolved locally so a probe can be scored without
sending the expected turns to the organizer.
"""

import argparse
import hashlib
import json
import math
import re
import sqlite3
import urllib.request
from pathlib import Path

try:
    from .write_synthesis_selection import deduplicate_thread_items
except ImportError:  # Direct script execution.
    from write_synthesis_selection import deduplicate_thread_items


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        value = json.loads(fenced.group(1))
        if isinstance(value, dict):
            return value

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("organizer returned no JSON object")


def _load_atomics(db_path: Path) -> list[dict]:
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            "SELECT rid, text, metadata FROM memories "
            "WHERE source = 'inference' ORDER BY created_at, rid"
        ).fetchall()

    atomics = []
    for rid, text, raw_metadata in rows:
        metadata = json.loads(raw_metadata or "{}")
        if (
            metadata.get("synthesis_kind") != "multi_axis_item"
            or metadata.get("granularity") != "atomic"
        ):
            continue
        atomics.append(
            {
                "rid": rid,
                "turn": metadata.get("first_mention_turn"),
                "date": metadata.get("first_mention_at"),
                "axis": metadata.get("synthesis_axis"),
                "item": text,
                "text": text,
                "evidence_ids": metadata.get("benchmark_evidence_ids") or [],
            }
        )
    deduplicated = deduplicate_thread_items(atomics)
    for index, item in enumerate(deduplicated, 1):
        item["id"] = f"A{index:04d}"
    return deduplicated


def _normalize_handles(
    handles: list[dict], known: dict[str, dict]
) -> tuple[list[dict], list[str]]:
    """Canonicalize model output and keep evidence IDs out of entity metadata."""
    normalized = []
    misplaced_anchor_ids = []
    for handle in handles:
        if not isinstance(handle, dict):
            continue
        evidence_ids = list(
            dict.fromkeys(
                value
                for value in handle.get("evidence_ids", [])
                if isinstance(value, str)
            )
        )
        anchor_entities = []
        for value in handle.get("anchor_entities", []):
            if not isinstance(value, str) or not value.strip():
                continue
            value = value.strip()
            if value in known:
                misplaced_anchor_ids.append(value)
                continue
            if value not in anchor_entities:
                anchor_entities.append(value)
        normalized.append(
            {
                "label": str(handle.get("label") or "").strip(),
                "anchor_entities": anchor_entities,
                "summary": str(handle.get("summary") or "").strip(),
                "evidence_ids": evidence_ids,
                "selection_rationale": str(
                    handle.get("selection_rationale") or ""
                ).strip(),
            }
        )
    return normalized, sorted(set(misplaced_anchor_ids))


def _apply_assignments(
    handles: list[dict],
    assignments: list[dict],
    expected_ids: set[str],
    max_evidence_per_handle: int = 12,
) -> tuple[set[str], list[str]]:
    """Apply valid repair assignments and report malformed references."""
    assigned = set()
    invalid = []
    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue
        evidence_id = assignment.get("id")
        if evidence_id not in expected_ids:
            invalid.append(str(evidence_id))
            continue
        handle_numbers = assignment.get("handle_numbers")
        if handle_numbers is None and "handle_number" in assignment:
            handle_numbers = [assignment["handle_number"]]
        handle_numbers = list(dict.fromkeys(handle_numbers or []))
        valid_numbers = [
            number
            for number in handle_numbers
            if isinstance(number, int) and 1 <= number <= len(handles)
        ]
        if len(valid_numbers) != len(handle_numbers) or not valid_numbers:
            invalid.append(f"{evidence_id}:handles={handle_numbers!r}")
            continue
        for number in valid_numbers:
            evidence_ids = handles[number - 1]["evidence_ids"]
            if (
                evidence_id not in evidence_ids
                and len(evidence_ids) >= max_evidence_per_handle
            ):
                invalid.append(f"{evidence_id}:handle={number}:full")
                continue
            if evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
            assigned.add(evidence_id)
    return assigned, sorted(set(invalid))


def _chat_json(
    *,
    host: str,
    model: str,
    prompt: str,
    schema: dict,
    num_predict: int,
    timeout: int,
) -> tuple[dict, dict]:
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {
                "temperature": 0.0,
                "seed": 0,
                "num_ctx": 65536,
                "num_predict": num_predict,
            },
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw_response = json.load(response)
    return _extract_json(raw_response["message"]["content"]), raw_response


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _capacity_constrained_assignments(
    handles: list[dict],
    evidence_ids: list[str],
    handle_embeddings: list[list[float]],
    item_embeddings: list[list[float]],
    max_evidence_per_handle: int = 12,
) -> tuple[list[dict], list[float]]:
    """Assign each item to its best available handle, confident items first."""
    capacities = [
        max(0, max_evidence_per_handle - len(handle.get("evidence_ids") or []))
        for handle in handles
    ]
    rankings = []
    for item_index, item_embedding in enumerate(item_embeddings):
        scored = sorted(
            (
                (_cosine_similarity(item_embedding, handle_embedding), handle_index)
                for handle_index, handle_embedding in enumerate(handle_embeddings)
                if capacities[handle_index] > 0
            ),
            reverse=True,
        )
        best = scored[0][0] if scored else float("-inf")
        second = scored[1][0] if len(scored) > 1 else float("-inf")
        margin = best - second if math.isfinite(second) else float("inf")
        rankings.append((margin, best, item_index, scored))

    assignments = []
    similarities = []
    for _, _, item_index, scored in sorted(rankings, reverse=True):
        selected = next(
            (
                (score, handle_index)
                for score, handle_index in scored
                if capacities[handle_index] > 0
            ),
            None,
        )
        if selected is None:
            continue
        score, handle_index = selected
        capacities[handle_index] -= 1
        assignments.append(
            {
                "id": evidence_ids[item_index],
                "handle_number": handle_index + 1,
            }
        )
        similarities.append(score)
    return assignments, similarities


def _embed_texts(
    *, host: str, model: str, texts: list[str], timeout: int
) -> list[list[float]]:
    body = json.dumps({"model": model, "input": texts}).encode("utf-8")
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/embed",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    embeddings = payload.get("embeddings") or []
    if len(embeddings) != len(texts):
        raise ValueError(
            f"embedder returned {len(embeddings)} vectors for {len(texts)} texts"
        )
    return embeddings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--resume-artifact",
        type=Path,
        help="reuse discovered handles and run only validation/assignment repair",
    )
    parser.add_argument("--min-handles", type=int, default=8)
    parser.add_argument("--max-handles", type=int, default=24)
    parser.add_argument("--num-predict", type=int, default=6000)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--repair-passes", type=int, default=2)
    parser.add_argument("--overflow-passes", type=int, default=3)
    parser.add_argument(
        "--assignment-mode",
        choices=("model", "embedding"),
        default="model",
    )
    parser.add_argument("--assignment-embed-model", default="mxbai-embed-large")
    parser.add_argument("--allow-unassigned", action="store_true")
    parser.add_argument("--anchor", help="locally inspect handles for this entity")
    parser.add_argument("--expect-turns", help="local-only comma-separated turns")
    args = parser.parse_args()
    if args.min_handles < 1 or args.max_handles < args.min_handles:
        parser.error("handle bounds must satisfy 1 <= min-handles <= max-handles")

    atomics = _load_atomics(args.db)
    model_rows = [
        {
            "id": item["id"],
            "turn": item["turn"],
            "axis": item["axis"],
            "text": item["text"],
        }
        for item in atomics
    ]
    serialized_rows = json.dumps(model_rows, ensure_ascii=True, separators=(",", ":"))
    input_sha256 = hashlib.sha256(serialized_rows.encode()).hexdigest()
    prompt = (
        "You are organizing a person's chronological memory into durable retrieval "
        "handles before any future query is known. Discover the distinct recurring "
        "concerns, projects, relationships, and event arcs represented below. Create "
        f"between {args.min_handles} and {args.max_handles} handles. A handle "
        "should be narrow enough that its evidence tells "
        "one coherent story, but should combine related evidence across distant turns. "
        "Preserve rare but concrete contributions, decisions, artifacts, advice acted "
        "on, and joint work; do not rank by raw mention frequency. For a recurring "
        "named relationship, create a person-specific timeline when useful. Use "
        "anchor_entities only for real named people, organizations, projects, or "
        "durable concepts, never atomic IDs. Select 3-12 chronological evidence IDs "
        "per handle, never invent an ID, and do not duplicate "
        "the same event merely because it appears under two extraction axes. Never "
        "create a catch-all handle for the whole history. Assign every supplied "
        "atomic ID to at least one coherent handle; do not silently drop rare events. Do not "
        "answer a question; no query is available. Return JSON only as "
        '{"handles":[{"label":"...","anchor_entities":["..."],'
        '"summary":"...","evidence_ids":["A0001"],'
        '"selection_rationale":"..."}]}.\n\nATOMIC ITEMS:\n'
        + serialized_rows
    )
    schema = {
        "type": "object",
        "properties": {
            "handles": {
                "type": "array",
                "minItems": args.min_handles,
                "maxItems": args.max_handles,
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "anchor_entities": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "summary": {"type": "string"},
                        "evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 12,
                            "uniqueItems": True,
                        },
                        "selection_rationale": {"type": "string"},
                    },
                    "required": [
                        "label",
                        "anchor_entities",
                        "summary",
                        "evidence_ids",
                        "selection_rationale",
                    ],
                },
            }
        },
        "required": ["handles"],
    }
    raw_response = None
    if args.resume_artifact:
        prior = json.loads(args.resume_artifact.read_text(encoding="utf-8"))
        if prior.get("input_sha256") != input_sha256:
            raise ValueError("resume artifact does not match the current atomic input")
        result = {"handles": prior.get("handles") or []}
    else:
        try:
            result, raw_response = _chat_json(
                host=args.host,
                model=args.model,
                prompt=prompt,
                schema=schema,
                num_predict=args.num_predict,
                timeout=args.timeout,
            )
        except (ValueError, json.JSONDecodeError):
            raw_path = args.output.with_suffix(args.output.suffix + ".raw.json")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(
                json.dumps(raw_response, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            print(f"unparseable organizer response={raw_path}")
            raise

    known = {item["id"]: item for item in atomics}
    handles, misplaced_anchor_ids = _normalize_handles(
        result.get("handles", []), known
    )
    invalid_ids = sorted(
        {
            evidence_id
            for handle in handles
            for evidence_id in handle.get("evidence_ids", [])
            if evidence_id not in known
        }
    )
    assigned_ids = {
        evidence_id
        for handle in handles
        for evidence_id in handle.get("evidence_ids", [])
        if evidence_id in known
    }
    unassigned_ids = sorted(set(known) - assigned_ids)
    repair_attempts = []
    overflow_attempts = []

    def write_checkpoint() -> dict:
        artifact = {
            "model": args.model,
            "input_count": len(atomics),
            "input_sha256": input_sha256,
            "handle_count": len(handles),
            "invalid_evidence_ids": invalid_ids,
            "misplaced_anchor_ids": misplaced_anchor_ids,
            "assigned_count": len(assigned_ids),
            "unassigned_evidence_ids": unassigned_ids,
            "repair_attempts": repair_attempts,
            "overflow_attempts": overflow_attempts,
            "input_items": model_rows,
            "handles": handles,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        return artifact

    write_checkpoint()
    if not (args.min_handles <= len(handles) <= args.max_handles):
        if raw_response is not None:
            raw_path = args.output.with_suffix(args.output.suffix + ".raw.json")
            raw_path.write_text(
                json.dumps(raw_response, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            print(f"invalid organizer response={raw_path}")
        raise ValueError(
            "organizer handle count violates discovery bounds: "
            f"expected {args.min_handles}-{args.max_handles}, got {len(handles)}"
        )
    if args.assignment_mode == "embedding" and unassigned_ids:
        handle_texts = [
            f"{handle['label']}. {handle['summary']}" for handle in handles
        ]
        item_texts = [known[evidence_id]["text"] for evidence_id in unassigned_ids]
        embeddings = _embed_texts(
            host=args.host,
            model=args.assignment_embed_model,
            texts=[*handle_texts, *item_texts],
            timeout=args.timeout,
        )
        assignments, similarities = _capacity_constrained_assignments(
            handles,
            unassigned_ids,
            embeddings[: len(handles)],
            embeddings[len(handles) :],
        )
        expected = set(unassigned_ids)
        repaired, invalid_assignments = _apply_assignments(
            handles, assignments, expected
        )
        repair_attempts.append(
            {
                "pass": 0,
                "mode": "embedding",
                "model": args.assignment_embed_model,
                "requested_count": len(expected),
                "assigned_count": len(repaired),
                "mean_similarity": (
                    sum(similarities) / len(similarities) if similarities else None
                ),
                "minimum_similarity": min(similarities) if similarities else None,
                "invalid_assignments": invalid_assignments,
            }
        )
        assigned_ids.update(repaired)
        unassigned_ids = sorted(set(known) - assigned_ids)
        write_checkpoint()

    model_repair_passes = (
        args.repair_passes if args.assignment_mode == "model" else 0
    )
    for pass_number in range(1, model_repair_passes + 1):
        if not unassigned_ids:
            break
        handle_rows = [
            {
                "number": index,
                "label": handle["label"],
                "summary": handle["summary"],
                "evidence_count": len(handle["evidence_ids"]),
                "remaining_capacity": max(0, 12 - len(handle["evidence_ids"])),
            }
            for index, handle in enumerate(handles, 1)
        ]
        omitted_rows = [
            {
                "id": evidence_id,
                "turn": known[evidence_id]["turn"],
                "axis": known[evidence_id]["axis"],
                "text": known[evidence_id]["text"],
            }
            for evidence_id in unassigned_ids
        ]
        repair_prompt = (
            "Complete an existing query-independent organization of a person's "
            "memory. Every omitted atomic item below must be assigned to exactly one "
            "coherent existing handle with remaining capacity. A handle may contain "
            "at most 12 evidence items after assignment. Use only the numbered handles; "
            "do not answer a question and do not omit an item. Return exactly one "
            "assignment for every omitted ID as JSON.\n\nHANDLES:\n"
            + json.dumps(handle_rows, ensure_ascii=True, separators=(",", ":"))
            + "\n\nOMITTED ATOMIC ITEMS:\n"
            + json.dumps(omitted_rows, ensure_ascii=True, separators=(",", ":"))
        )
        repair_schema = {
            "type": "object",
            "properties": {
                "assignments": {
                    "type": "array",
                    "minItems": len(unassigned_ids),
                    "maxItems": len(unassigned_ids),
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "handle_number": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": len(handles),
                            },
                        },
                        "required": ["id", "handle_number"],
                    },
                }
            },
            "required": ["assignments"],
        }
        repair_result, _ = _chat_json(
            host=args.host,
            model=args.model,
            prompt=repair_prompt,
            schema=repair_schema,
            num_predict=args.num_predict,
            timeout=args.timeout,
        )
        expected = set(unassigned_ids)
        repaired, invalid_assignments = _apply_assignments(
            handles, repair_result.get("assignments", []), expected
        )
        repair_attempts.append(
            {
                "pass": pass_number,
                "requested_count": len(expected),
                "assigned_count": len(repaired),
                "invalid_assignments": invalid_assignments,
            }
        )
        assigned_ids.update(repaired)
        unassigned_ids = sorted(set(known) - assigned_ids)
        write_checkpoint()

    for pass_number in range(1, args.overflow_passes + 1):
        if not unassigned_ids:
            break
        existing_handles = [
            {"label": handle["label"], "summary": handle["summary"]}
            for handle in handles
        ]
        omitted_rows = [
            {
                "id": evidence_id,
                "turn": known[evidence_id]["turn"],
                "axis": known[evidence_id]["axis"],
                "text": known[evidence_id]["text"],
            }
            for evidence_id in unassigned_ids
        ]
        min_new_handles = max(1, math.ceil(len(unassigned_ids) / 12))
        max_new_handles = min(
            8,
            max(min_new_handles, math.ceil(len(unassigned_ids) / 3)),
        )
        min_evidence = 1 if len(unassigned_ids) == 1 else 2
        overflow_prompt = (
            "Extend a query-independent memory organization. The residual atomic "
            "items could not fit their most coherent existing handles without "
            "exceeding 12 evidence items. Create new, distinct handles that complement "
            "the existing taxonomy. Assign every residual ID to at least one new "
            "handle, use 2-12 chronological IDs per handle, never invent an ID, and "
            "use anchor_entities only for real names or durable concepts. Do not "
            "answer a question. Return JSON only.\n\nEXISTING HANDLES:\n"
            + json.dumps(existing_handles, ensure_ascii=True, separators=(",", ":"))
            + "\n\nRESIDUAL ATOMIC ITEMS:\n"
            + json.dumps(omitted_rows, ensure_ascii=True, separators=(",", ":"))
        )
        overflow_schema = {
            "type": "object",
            "properties": {
                "handles": {
                    "type": "array",
                    "minItems": min_new_handles,
                    "maxItems": max_new_handles,
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "anchor_entities": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "summary": {"type": "string"},
                            "evidence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": min_evidence,
                                "maxItems": 12,
                                "uniqueItems": True,
                            },
                            "selection_rationale": {"type": "string"},
                        },
                        "required": [
                            "label",
                            "anchor_entities",
                            "summary",
                            "evidence_ids",
                            "selection_rationale",
                        ],
                    },
                }
            },
            "required": ["handles"],
        }
        overflow_result, _ = _chat_json(
            host=args.host,
            model=args.model,
            prompt=overflow_prompt,
            schema=overflow_schema,
            num_predict=args.num_predict,
            timeout=args.timeout,
        )
        new_handles, new_misplaced = _normalize_handles(
            overflow_result.get("handles", []), known
        )
        expected = set(unassigned_ids)
        out_of_scope_overflow_ids = sorted(
            {
                evidence_id
                for handle in new_handles
                for evidence_id in handle["evidence_ids"]
                if evidence_id not in expected
            }
        )
        valid_new_handles = []
        newly_assigned = set()
        for handle in new_handles:
            handle["evidence_ids"] = [
                evidence_id
                for evidence_id in handle["evidence_ids"]
                if evidence_id in expected
            ]
            if len(handle["evidence_ids"]) < min_evidence:
                continue
            valid_new_handles.append(handle)
            newly_assigned.update(handle["evidence_ids"])
        handles.extend(valid_new_handles)
        misplaced_anchor_ids = sorted(
            set(misplaced_anchor_ids).union(new_misplaced)
        )
        assigned_ids.update(newly_assigned)
        unassigned_ids = sorted(set(known) - assigned_ids)
        overflow_attempts.append(
            {
                "pass": pass_number,
                "requested_count": len(expected),
                "new_handle_count": len(valid_new_handles),
                "assigned_count": len(newly_assigned),
                "out_of_scope_evidence_ids": out_of_scope_overflow_ids,
            }
        )
        write_checkpoint()

    write_checkpoint()

    print(
        f"atomics={len(atomics)} handles={len(handles)} "
        f"invalid_ids={len(invalid_ids)} unassigned={len(unassigned_ids)}"
    )
    print(f"artifact={args.output}")
    if invalid_ids:
        raise ValueError(f"organizer returned invalid evidence IDs: {invalid_ids}")
    if unassigned_ids and not args.allow_unassigned:
        raise ValueError(
            f"organizer left {len(unassigned_ids)} atomic items unassigned; "
            "use --allow-unassigned only for diagnostic probes"
        )
    if args.anchor:
        needle = args.anchor.casefold()
        matched = [
            handle
            for handle in handles
            if needle
            in " ".join(
                [
                    handle.get("label", ""),
                    handle.get("summary", ""),
                    *handle.get("anchor_entities", []),
                ]
            ).casefold()
        ]
        selected_turns = sorted(
            {
                known[evidence_id]["turn"]
                for handle in matched
                for evidence_id in handle.get("evidence_ids", [])
                if evidence_id in known and known[evidence_id]["turn"] is not None
            }
        )
        print(f"anchor={args.anchor} matched_handles={len(matched)} turns={selected_turns}")
        for handle in matched:
            print(json.dumps(handle, ensure_ascii=True))
        if args.expect_turns:
            expected = sorted(int(value) for value in args.expect_turns.split(","))
            print(f"expected={expected} exact={selected_turns == expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
