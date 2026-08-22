"""Build query-blind role collections above organizer leaf handles."""

import argparse
import hashlib
import json
from pathlib import Path

def normalize_collections(
    values: list[dict], leaf_handles: dict[str, dict]
) -> tuple[list[dict], list[str], list[str]]:
    """Resolve member handles and report invalid or multiply assigned IDs."""
    collections = []
    invalid = []
    memberships: dict[str, int] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        member_ids = list(
            dict.fromkeys(
                str(member_id).strip().upper()
                for member_id in value.get("member_handle_ids") or []
            )
        )
        invalid.extend(
            member_id for member_id in member_ids if member_id not in leaf_handles
        )
        member_ids = [member_id for member_id in member_ids if member_id in leaf_handles]
        label = str(value.get("label") or "").strip()
        summary = str(value.get("summary") or "").strip()
        if not label or not summary or not member_ids:
            continue
        for member_id in member_ids:
            memberships[member_id] = memberships.get(member_id, 0) + 1
        evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for member_id in member_ids
                for evidence_id in leaf_handles[member_id]["evidence_ids"]
            )
        )
        collections.append(
            {
                "label": label,
                "anchor_entities": list(
                    dict.fromkeys(
                        str(entity).strip()
                        for entity in value.get("anchor_entities") or []
                        if str(entity).strip()
                    )
                ),
                "summary": summary,
                "member_handle_ids": member_ids,
                "evidence_ids": evidence_ids,
            }
        )
    duplicated = sorted(
        member_id for member_id, count in memberships.items() if count > 1
    )
    return collections, sorted(set(invalid)), duplicated


def append_unassigned_singletons(
    collections: list[dict], leaf_handles: dict[str, dict]
) -> list[str]:
    """Preserve model-omitted leaves without inventing a semantic merge."""
    assigned = {
        member_id
        for collection in collections
        for member_id in collection["member_handle_ids"]
    }
    unassigned = sorted(set(leaf_handles) - assigned)
    for member_id in unassigned:
        leaf = leaf_handles[member_id]
        collections.append(
            {
                "label": str(leaf.get("label") or member_id),
                "anchor_entities": list(leaf.get("anchor_entities") or []),
                "summary": str(leaf.get("summary") or ""),
                "member_handle_ids": [member_id],
                "evidence_ids": list(leaf["evidence_ids"]),
                "fallback_singleton": True,
            }
        )
    return unassigned


def main() -> int:
    try:
        from .global_organizer_probe import _chat_json
    except ImportError:  # Direct script execution.
        from global_organizer_probe import _chat_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--organizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--min-collections", type=int, default=8)
    parser.add_argument("--max-collections", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.min_collections < 1 or args.max_collections < args.min_collections:
        parser.error("collection bounds must satisfy 1 <= min <= max")

    organizer = json.loads(args.organizer.read_text(encoding="utf-8"))
    leaf_handles = {
        f"H{index:03d}": {
            **handle,
            "evidence_ids": list(dict.fromkeys(handle.get("evidence_ids") or [])),
        }
        for index, handle in enumerate(organizer.get("handles") or [], 1)
        if handle.get("evidence_ids")
    }
    rows = [
        {
            "id": handle_id,
            "label": handle.get("label"),
            "anchor_entities": handle.get("anchor_entities") or [],
            "summary": handle.get("summary"),
        }
        for handle_id, handle in leaf_handles.items()
    ]
    serialized = json.dumps(rows, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    prompt = (
        "Before any future query is known, partition the leaf memory handles below "
        "into stable higher-level collections. Each leaf handle must appear in "
        "exactly one collection. Organize primarily by relationship role and type of "
        "contribution or concern, not merely by a shared project. Keep industry mentors "
        "and professional peers separate from academic advisors, family/partners, "
        "friends, collaborators, and administrative or service contacts. A multi-person "
        "collection is appropriate when people share a role and contribute to the same "
        "kind of arc. Put broad drafting/process handles into process collections rather "
        "than using them as catch-alls for person-specific handles. Preserve correction "
        "or denial semantics in collection summaries when a leaf contains them. Do not "
        "invent events, people, or leaf IDs. No user query or expected answer is "
        "available. Create between "
        f"{args.min_collections} and {args.max_collections} collections. Return JSON "
        "only as "
        '{"collections":[{"label":"...","anchor_entities":["..."],'
        '"summary":"...","member_handle_ids":["H001"]}]}.'
        "\n\nLEAF HANDLES:\n"
        + serialized
    )
    schema = {
        "type": "object",
        "properties": {
            "collections": {
                "type": "array",
                "minItems": args.min_collections,
                "maxItems": args.max_collections,
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "anchor_entities": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "summary": {"type": "string"},
                        "member_handle_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "uniqueItems": True,
                        },
                    },
                    "required": [
                        "label",
                        "anchor_entities",
                        "summary",
                        "member_handle_ids",
                    ],
                },
            }
        },
        "required": ["collections"],
    }
    result, raw_response = _chat_json(
        host=args.host,
        model=args.model,
        prompt=prompt,
        schema=schema,
        num_predict=5000,
        timeout=args.timeout,
    )
    raw_collections = result.get("collections") or []
    collections, invalid, duplicated = normalize_collections(
        raw_collections[: args.max_collections], leaf_handles
    )
    fallback_singletons = append_unassigned_singletons(collections, leaf_handles)
    assigned = {
        member_id
        for collection in collections
        for member_id in collection["member_handle_ids"]
    }
    unassigned = sorted(set(leaf_handles) - assigned)
    artifact = {
        "model": args.model,
        "organization_level": "role_collection",
        "source_model": organizer.get("model"),
        "source_handle_count": len(leaf_handles),
        "source_handle_sha256": digest,
        "model_collection_count": len(raw_collections),
        "model_collection_overflow": max(
            0, len(raw_collections) - args.max_collections
        ),
        "input_sha256": organizer.get("input_sha256"),
        "input_items": organizer.get("input_items") or [],
        "handle_count": len(collections),
        "invalid_member_handle_ids": invalid,
        "duplicated_member_handle_ids": duplicated,
        "unassigned_member_handle_ids": unassigned,
        "fallback_singleton_member_handle_ids": fallback_singletons,
        "handles": collections,
        "response_eval_count": raw_response.get("eval_count"),
        "response_prompt_eval_count": raw_response.get("prompt_eval_count"),
    }
    rendered = json.dumps(artifact, indent=2, ensure_ascii=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        f"leaves={len(leaf_handles)} collections={len(collections)} "
        f"unassigned={len(unassigned)} duplicated={len(duplicated)} "
        f"invalid={len(invalid)}"
    )
    print(f"wrote {args.output}")
    if invalid or duplicated or unassigned:
        raise ValueError(
            "role collections must be an exhaustive partition: "
            f"invalid={invalid}, duplicated={duplicated}, unassigned={unassigned}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
