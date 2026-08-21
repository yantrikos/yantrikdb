"""Present every persisted organizer handle after exact raw AMB evidence."""

from __future__ import annotations

from datetime import UTC, datetime


def _date(value: object) -> str | None:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(timestamp, tz=UTC).date().isoformat()


def _turns(metadata: dict) -> list[int]:
    values = [
        occurrence.get("first_mention_turn")
        for occurrence in metadata.get("organizer_evidence_timeline") or []
        if isinstance(occurrence, dict)
    ]
    values.extend(
        metadata.get(key)
        for key in ("organizer_first_turn", "organizer_last_turn")
    )
    return sorted({int(value) for value in values if value is not None})


def _record_span(metadata: dict) -> str:
    first = _date(metadata.get("first_mention_at"))
    last = _date(metadata.get("evidence_span_end_at"))
    parts = []
    if first:
        parts.append(
            f"recorded {first}"
            if not last or first == last
            else f"recorded {first} to {last}"
        )
    turns = _turns(metadata)
    if turns:
        parts.append(
            f"turn {turns[0]}"
            if turns[0] == turns[-1]
            else f"turns {turns[0]}-{turns[-1]}"
        )
    return "; ".join(parts)


def topic_card_document(record: dict) -> str:
    metadata = record.get("metadata") or {}
    label = str(metadata.get("organizer_label") or "Topic trajectory").strip()
    text = str(record.get("text") or "").strip()
    summary = str(metadata.get("organizer_summary") or "").strip()
    if not summary:
        prefix = f"Topic trajectory: {label}."
        summary = text[len(prefix):].strip() if text.startswith(prefix) else text
    if not summary:
        return ""
    span = _record_span(metadata)
    heading = f"Topic: {label}" + (f" ({span})" if span else "")
    return f"{heading}\n{summary}"


def _record_key(record: dict) -> tuple:
    # UUIDv7 RIDs are lexically chronological. persist_organization writes
    # handles synchronously in plan order, without putting presentation-only
    # fields into the idempotent synthesis payload.
    return (record.get("rid", ""),)


def load_persisted_topic_cards(
    db, namespace: str | None, page_size: int = 500
) -> tuple[list[dict], dict]:
    """Enumerate complete topic handles; similarity recall must not top-k them."""
    if page_size < 1:
        raise ValueError("page_size must be positive")
    records = []
    cursor = None
    pages = 0
    scanned = 0
    while True:
        page = db.list_records(
            namespace=namespace,
            since_rid=cursor,
            limit=page_size,
            order="asc",
        )
        pages += 1
        page_records = page.get("records") or []
        scanned += len(page_records)
        for record in page_records:
            metadata = record.get("metadata") or {}
            if (
                metadata.get("organizer_kind") == "query_independent_topic"
                or metadata.get("thread_builder") == "llm_topic_organizer_v1"
            ):
                records.append(record)
        next_cursor = page.get("next_cursor")
        if not next_cursor:
            break
        if next_cursor == cursor:
            raise RuntimeError("list_records returned a non-advancing cursor")
        cursor = next_cursor

    records.sort(key=_record_key)
    cards = []
    seen_documents = set()
    duplicate_count = 0
    for record in records:
        document = topic_card_document(record)
        if not document:
            continue
        if document in seen_documents:
            duplicate_count += 1
            continue
        seen_documents.add(document)
        cards.append(
            {
                "rid": str(record.get("rid") or ""),
                "content": document,
            }
        )
    return cards, {
        "pages": pages,
        "records_scanned": scanned,
        "organizer_records": len(records),
        "duplicate_cards_removed": duplicate_count,
        "cards_returned": len(cards),
    }
