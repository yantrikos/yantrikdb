"""Build frozen AMB contexts from query-independent organizer topic cards."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, list) else payload["results"]


def _record_span(handle: dict, source_items: dict[str, dict]) -> str:
    evidence = [
        source_items[evidence_id]
        for evidence_id in handle.get("evidence_ids") or []
        if evidence_id in source_items
    ]
    dates = sorted(item["date"] for item in evidence if item.get("date") is not None)
    turns = sorted(item["turn"] for item in evidence if item.get("turn") is not None)
    parts = []
    if dates:
        first = datetime.fromtimestamp(dates[0], tz=UTC).date().isoformat()
        last = datetime.fromtimestamp(dates[-1], tz=UTC).date().isoformat()
        parts.append(f"recorded {first}" if first == last else f"recorded {first} to {last}")
    if turns:
        parts.append(
            f"turn {turns[0]}" if turns[0] == turns[-1] else f"turns {turns[0]}-{turns[-1]}"
        )
    return "; ".join(parts)


def topic_card_documents(
    artifact: dict, include_singletons: bool = True
) -> list[str]:
    fallback_ids = set(artifact.get("singleton_fallback_evidence_ids") or [])
    source_items = {
        item["id"]: item
        for item in artifact.get("input_items") or []
        if item.get("id")
    }
    cards = []
    for handle in artifact.get("handles") or []:
        evidence_ids = handle.get("evidence_ids") or []
        is_singleton = len(evidence_ids) == 1 and evidence_ids[0] in fallback_ids
        if is_singleton and not include_singletons:
            continue
        label = str(handle.get("label") or "Memory topic").strip()
        summary = str(handle.get("summary") or "").strip()
        if summary:
            span = _record_span(handle, source_items)
            heading = f"Topic: {label}" + (f" ({span})" if span else "")
            cards.append(f"{heading}\n{summary}")
    return cards


def render_topic_cards(artifact: dict, include_singletons: bool = True) -> str:
    cards = topic_card_documents(artifact, include_singletons)
    return "\n\n".join(
        f"## Memory {index}\n{card}" for index, card in enumerate(cards, 1)
    )


def build_context_rows(
    query_rows: list[dict],
    artifact: dict,
    unit: str,
    include_singletons: bool = True,
    query_ids: set[str] | None = None,
) -> list[dict]:
    documents = topic_card_documents(artifact, include_singletons)
    context = "\n\n".join(
        f"## Memory {index}\n{card}"
        for index, card in enumerate(documents, 1)
    )
    if not context:
        raise ValueError("organizer artifact contains no presentable topic cards")
    prefix = f"{unit}_"
    selected = []
    for row in query_rows:
        if not str(row.get("query_id") or "").startswith(prefix):
            continue
        if query_ids is not None and str(row.get("query_id")) not in query_ids:
            continue
        result = dict(row)
        result["context"] = context
        result["documents"] = documents
        result["selection"] = {
            "mode": "query_independent_topic_cards_v1",
            "handle_count": len(artifact.get("handles") or []),
            "singleton_fallback_count": (
                int(artifact.get("singleton_fallback_count") or 0)
                if include_singletons
                else 0
            ),
            "organizer_input_sha256": artifact.get("input_sha256"),
        }
        selected.append(result)
    if not selected:
        raise ValueError(f"query source contains no rows for unit {unit!r}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-singletons", action="store_true")
    parser.add_argument("--query-ids", help="optional comma-separated query IDs")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    organizer = json.loads(args.artifact.read_text(encoding="utf-8"))
    rows = build_context_rows(
        load_rows(args.queries),
        organizer,
        args.unit,
        include_singletons=not args.exclude_singletons,
        query_ids=(
            {value.strip() for value in args.query_ids.split(",") if value.strip()}
            if args.query_ids
            else None
        ),
    )
    existing = load_rows(args.out) if args.append and args.out.exists() else []
    existing_ids = {row.get("query_id") for row in existing}
    duplicates = sorted(
        row["query_id"] for row in rows if row.get("query_id") in existing_ids
    )
    if duplicates:
        raise ValueError(f"output already contains query IDs: {duplicates}")
    rows = [*existing, *rows]
    payload = {
        "protocol": "query-independent-topic-cards-v1",
        "units": sorted(
            {str(row["query_id"]).split("_", 1)[0] for row in rows}
        ),
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote rows={len(rows)} {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
