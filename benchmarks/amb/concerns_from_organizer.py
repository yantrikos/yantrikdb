"""Build answer-sized concerns inside globally discovered topic handles."""

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from .global_organizer_probe import _chat_json, _load_atomics
except ImportError:  # Direct script execution.
    from global_organizer_probe import _chat_json, _load_atomics


def _input_digest(atomics: list[dict]) -> str:
    rows = [
        {
            "id": item["id"],
            "turn": item["turn"],
            "axis": item["axis"],
            "text": item["text"],
        }
        for item in atomics
    ]
    serialized = json.dumps(rows, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _artifact_atomics(artifact: dict) -> list[dict]:
    """Load the organizer's frozen model input without requiring its source DB."""
    atomics = []
    seen = set()
    for index, value in enumerate(artifact.get("input_items") or [], 1):
        if not isinstance(value, dict):
            raise ValueError(f"input item {index} is not an object")
        item_id = str(value.get("id") or "").strip()
        text = str(value.get("text") or "").strip()
        if not item_id or not text:
            raise ValueError(f"input item {index} has an empty id or text")
        if item_id in seen:
            raise ValueError(f"duplicate input item id: {item_id}")
        seen.add(item_id)
        atomics.append(
            {
                "id": item_id,
                "turn": value.get("turn"),
                "axis": str(value.get("axis") or ""),
                "text": text,
            }
        )
    if not atomics:
        raise ValueError("organizer artifact has no input_items")
    return atomics


def _handle_prompt(handle: dict, evidence: list[dict], max_items: int = 6) -> str:
    payload = [
        {
            "id": item["id"],
            "turn": item["turn"],
            "text": item["text"],
        }
        for item in evidence
    ]
    return (
        f"Before any future query is known, assemble the evidence below into 1-{max_items} "
        "narrow chronological concern threads within one globally discovered topic. "
        "Each output item is a durable record for one coherent concern, relationship, "
        "artifact, or goal. When several events update the same concern over time, keep "
        "their distinct milestones in chronological order inside one self-contained "
        "item; do not flatten every event into a separate item. Merge question, answer, "
        "and follow-up fragments about the same milestone. Keep unrelated concerns "
        "separate even when they share the broad topic. Preserve names, dates, "
        "quantities, artifacts, corrections, and specific actions. Cover every supplied "
        "evidence ID in at least one output item. Use 1-6 supplied evidence IDs per item, "
        "allow an ID in at most two items, and never invent an ID. Do not emit profile "
        "facts, catch-all summaries, or filler. No benchmark question or expected answer "
        "is available. Return ONLY one JSON object with exactly this shape and no "
        "markdown: "
        '{"items":[{"text":"chronological self-contained thread",'
        '"anchor_entities":["name or artifact"],'
        '"evidence_ids":["A0001","A0002"]}]}.'
        f"\n\nTOPIC: {handle.get('label')}\n"
        f"TOPIC SUMMARY: {handle.get('summary')}\n"
        "EVIDENCE:\n"
        + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    )


def _schema(max_items: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_items,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "anchor_entities": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 6,
                            "uniqueItems": True,
                        },
                    },
                    "required": ["text", "anchor_entities", "evidence_ids"],
                },
            }
        },
        "required": ["items"],
    }


def _normalize_handle_items(
    handle_index: int,
    handle: dict,
    values: list[dict],
    known_ids: set[str],
) -> tuple[list[dict], list[str]]:
    items = []
    invalid = []
    for local_index, value in enumerate(values, 1):
        if not isinstance(value, dict):
            continue
        text = str(value.get("text") or "").strip()
        evidence_ids = list(
            dict.fromkeys(
                rid
                for rid in value.get("evidence_ids") or []
                if isinstance(rid, str)
            )
        )
        invalid.extend(rid for rid in evidence_ids if rid not in known_ids)
        evidence_ids = [rid for rid in evidence_ids if rid in known_ids]
        if not text or not evidence_ids:
            continue
        items.append(
            {
                "text": text,
                "anchor_entities": list(
                    dict.fromkeys(
                        entity.strip()
                        for entity in value.get("anchor_entities") or []
                        if isinstance(entity, str) and entity.strip()
                    )
                ),
                "evidence_ids": evidence_ids,
                "topic_handle": str(handle.get("label") or ""),
                "topic_index": handle_index,
                "local_index": local_index,
            }
        )
    return items, invalid


_DEDUP_STOPWORDS = {
    "about",
    "concern",
    "personal",
    "statement",
    "their",
    "user",
    "with",
}


def _content_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[^\W\d_]+", text)
        if len(token) >= 4 and token.casefold() not in _DEDUP_STOPWORDS
    }


def _same_concern(left: dict, right: dict) -> bool:
    if left["topic_index"] == right["topic_index"]:
        return False
    left_ids = set(left["evidence_ids"])
    right_ids = set(right["evidence_ids"])
    overlap = len(left_ids & right_ids)
    if not overlap or overlap / min(len(left_ids), len(right_ids)) < 0.5:
        return False
    return bool(_content_tokens(left["text"]) & _content_tokens(right["text"]))


def merge_cross_handle_duplicates(
    items: list[dict], max_memberships: int = 2
) -> list[dict]:
    """Collapse overlapping views and enforce bounded evidence memberships."""
    merged = []
    for candidate in sorted(
        items,
        key=lambda item: (
            -len(item["evidence_ids"]),
            item["topic_index"],
            item["local_index"],
        ),
    ):
        duplicate = next(
            (item for item in merged if _same_concern(item, candidate)), None
        )
        if duplicate is not None:
            duplicate["evidence_ids"] = list(
                dict.fromkeys([*duplicate["evidence_ids"], *candidate["evidence_ids"]])
            )[:6]
            duplicate["anchor_entities"] = list(
                dict.fromkeys(
                    [
                        *duplicate["anchor_entities"],
                        *candidate["anchor_entities"],
                    ]
                )
            )
            if "topic_handles" not in duplicate:
                duplicate["topic_handles"] = [duplicate.pop("topic_handle")]
            duplicate["topic_handles"].append(candidate["topic_handle"])
            continue
        candidate = dict(candidate)
        candidate["topic_handles"] = [candidate.pop("topic_handle")]
        merged.append(candidate)

    memberships: dict[str, int] = {}
    bounded = []
    for item in merged:
        admissible = [
            rid
            for rid in item["evidence_ids"]
            if memberships.get(rid, 0) < max_memberships
        ]
        if not admissible:
            continue
        item["evidence_ids"] = admissible
        for rid in admissible:
            memberships[rid] = memberships.get(rid, 0) + 1
        identity = hashlib.sha256(
            "\0".join([item["text"], *sorted(admissible)]).encode()
        ).hexdigest()[:24]
        item["id"] = f"concern-{identity}"
        item.pop("topic_index", None)
        item.pop("local_index", None)
        bounded.append(item)
    return bounded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path)
    parser.add_argument("--organizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-items-per-handle", type=int, default=6)
    parser.add_argument("--handle-index", type=int, action="append", default=[])
    parser.add_argument("--require-exhaustive-handles", action="store_true")
    parser.add_argument("--num-predict", type=int, default=1800)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.max_items_per_handle <= 12:
        parser.error("--max-items-per-handle must be between 1 and 12")

    organizer = json.loads(args.organizer.read_text(encoding="utf-8"))
    atomics = _load_atomics(args.db) if args.db else _artifact_atomics(organizer)
    known = {item["id"]: item for item in atomics}
    digest = _input_digest(atomics)
    if digest != organizer.get("input_sha256"):
        raise ValueError("organizer artifact does not match atomic input")

    jobs = []
    for index, handle in enumerate(organizer.get("handles") or [], 1):
        if args.handle_index and index not in args.handle_index:
            continue
        evidence = [
            known[rid]
            for rid in dict.fromkeys(handle.get("evidence_ids") or [])
            if rid in known
        ]
        if evidence:
            jobs.append((index, handle, evidence))

    raw_items = []
    invalid = []
    responses = []

    def generate(job):
        index, handle, evidence = job
        result, response = _chat_json(
            host=args.host,
            model=args.model,
            prompt=_handle_prompt(handle, evidence, args.max_items_per_handle),
            schema=_schema(args.max_items_per_handle),
            num_predict=args.num_predict,
            timeout=args.timeout,
        )
        items, bad = _normalize_handle_items(
            index,
            handle,
            result.get("items") or [],
            {item["id"] for item in evidence},
        )
        assigned = {rid for item in items for rid in item["evidence_ids"]}
        missing = sorted({item["id"] for item in evidence} - assigned)
        return (
            index,
            items,
            bad,
            missing,
            response,
            str((response.get("message") or {}).get("content") or ""),
        )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(generate, job): job[0] for job in jobs}
        for future in as_completed(futures):
            index, items, bad, missing, response, response_content = future.result()
            raw_items.extend(items)
            invalid.extend(f"handle={index}:{rid}" for rid in bad)
            responses.append(
                {
                    "handle_index": index,
                    "item_count": len(items),
                    "unassigned_evidence_ids": missing,
                    "eval_count": response.get("eval_count"),
                    "prompt_eval_count": response.get("prompt_eval_count"),
                    "response_content": response_content,
                }
            )
            print(
                f"handle={index}/{len(jobs)} concerns={len(items)} "
                f"unassigned={len(missing)}"
            )

    items = merge_cross_handle_duplicates(raw_items)
    assigned = {rid for item in items for rid in item["evidence_ids"]}
    selected_evidence = {
        item["id"] for _, _, evidence in jobs for item in evidence
    }
    artifact = {
        "model": args.model,
        "organization_level": "concern",
        "generator": "global_topics_then_local_concern_threads_v3",
        "input_count": len(atomics),
        "input_sha256": digest,
        "topic_handle_count": len(jobs),
        "selected_handle_indexes": [job[0] for job in jobs],
        "selected_evidence_count": len(selected_evidence),
        "max_items_per_handle": args.max_items_per_handle,
        "raw_item_count": len(raw_items),
        "item_count": len(items),
        "assigned_count": len(assigned),
        "unassigned_evidence_ids": sorted(selected_evidence - assigned),
        "invalid_evidence_ids": sorted(set(invalid)),
        "responses": sorted(responses, key=lambda row: row["handle_index"]),
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(
        f"raw={len(raw_items)} merged={len(items)} assigned={len(assigned)} "
        f"unassigned={len(artifact['unassigned_evidence_ids'])} invalid={len(invalid)}"
    )
    print(f"wrote {args.output}")
    if invalid:
        raise ValueError(f"concern generation used invalid evidence: {invalid}")
    incomplete_handles = [
        response
        for response in responses
        if response["unassigned_evidence_ids"]
    ]
    if args.require_exhaustive_handles and incomplete_handles:
        details = ", ".join(
            f"{response['handle_index']}="
            f"{len(response['unassigned_evidence_ids'])}"
            for response in sorted(
                incomplete_handles, key=lambda row: row["handle_index"]
            )
        )
        raise ValueError(f"concern generation left handles incomplete: {details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
