"""Generate query-independent answer-sized concerns from AMB atomic memories."""

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .global_organizer_probe import _chat_json, _load_atomics
except ImportError:  # Direct script execution.
    from global_organizer_probe import _chat_json, _load_atomics


def _normalize_items(values: list[dict], known_ids: set[str]) -> tuple[list[dict], list[str]]:
    items = []
    invalid = []
    item_ids = set()
    memberships: dict[str, int] = {}
    for index, value in enumerate(values, 1):
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
        identity = hashlib.sha256(
            "\0".join([text, *sorted(evidence_ids)]).encode()
        ).hexdigest()[:24]
        item_id = f"concern-{identity}"
        if item_id in item_ids:
            continue
        item_ids.add(item_id)
        for rid in evidence_ids:
            memberships[rid] = memberships.get(rid, 0) + 1
        items.append(
            {
                "id": item_id,
                "text": text,
                "anchor_entities": list(
                    dict.fromkeys(
                        entity.strip()
                        for entity in value.get("anchor_entities") or []
                        if isinstance(entity, str) and entity.strip()
                    )
                ),
                "evidence_ids": evidence_ids,
                "rationale": str(value.get("rationale") or "").strip(),
                "ordinal": index,
            }
        )
    overused = sorted(rid for rid, count in memberships.items() if count > 2)
    if overused:
        invalid.extend(f"{rid}:memberships={memberships[rid]}" for rid in overused)
    return items, sorted(set(invalid))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--min-items", type=int, default=24)
    parser.add_argument("--max-items", type=int, default=60)
    parser.add_argument("--num-predict", type=int, default=12000)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.min_items < 1 or args.max_items < args.min_items:
        parser.error("item bounds must satisfy 1 <= min-items <= max-items")

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
    serialized = json.dumps(model_rows, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    prompt = (
        "You are consolidating a person's chronological memory before any future "
        "query is known. Produce durable answer-sized concern items, not broad topic "
        "summaries and not one item per conversational turn. Each concern must express "
        "one concrete event, decision, contribution, recurring issue, or milestone in "
        "one self-contained sentence. Merge adjacent question/answer/follow-up fragments "
        "about the same concern. Combine distant evidence only when it clearly updates "
        "the same concern. Keep separate concerns separate even when they share a person "
        "or project. Preserve named people, organizations, artifacts, quantities, and "
        "specific actions. Do not infer facts not stated by evidence. Create between "
        f"{args.min_items} and {args.max_items} items using 1-6 evidence IDs each. An "
        "evidence ID may support at most two items. Rare evidence may remain unassigned; "
        "never create catch-all or filler items merely for coverage. No question or gold "
        "answer is available. Return JSON only as "
        '{"items":[{"text":"...","anchor_entities":["..."],'
        '"evidence_ids":["A0001"],"rationale":"..."}]}.'
        "\n\nATOMIC MEMORIES:\n"
        + serialized
    )
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": args.min_items,
                "maxItems": args.max_items,
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
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "text",
                        "anchor_entities",
                        "evidence_ids",
                        "rationale",
                    ],
                },
            }
        },
        "required": ["items"],
    }
    result, raw_response = _chat_json(
        host=args.host,
        model=args.model,
        prompt=prompt,
        schema=schema,
        num_predict=args.num_predict,
        timeout=args.timeout,
    )
    items, invalid = _normalize_items(
        result.get("items") or [], {item["id"] for item in atomics}
    )
    assigned = {rid for item in items for rid in item["evidence_ids"]}
    artifact = {
        "model": args.model,
        "organization_level": "concern",
        "input_count": len(atomics),
        "input_sha256": digest,
        "item_count": len(items),
        "assigned_count": len(assigned),
        "unassigned_evidence_ids": sorted(
            {item["id"] for item in atomics} - assigned
        ),
        "invalid_evidence_ids": invalid,
        "items": items,
        "response_eval_count": raw_response.get("eval_count"),
        "response_prompt_eval_count": raw_response.get("prompt_eval_count"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(
        f"atomics={len(atomics)} concerns={len(items)} assigned={len(assigned)} "
        f"unassigned={len(artifact['unassigned_evidence_ids'])} invalid={len(invalid)}"
    )
    print(f"wrote {args.output}")
    if invalid:
        raise ValueError(f"concern generator returned invalid evidence: {invalid}")
    if not args.min_items <= len(items) <= args.max_items:
        raise ValueError(
            f"concern count must be {args.min_items}-{args.max_items}, got {len(items)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
