"""Deterministic chronology helpers shared by AMB providers and artifacts."""

import re
from datetime import datetime, timezone
from typing import Any


_ROLE_AWARE_PREFIX_RE = re.compile(
    r"^\[Speaker: [^|\]]+ \| "
    r"(?P<date>[A-Z][a-z]+ \d{2}, \d{4})"
    r"(?: \| Turn (?P<turn>\d+))?\]"
)


def _number_or_last(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float("inf")


def chronological_hit_key(hit: dict) -> tuple[float, float, float, str]:
    """Order a selected hit by event time without changing selection."""
    metadata = hit.get("metadata") or {}
    return (
        _number_or_last(hit.get("created_at")),
        _number_or_last(metadata.get("turn_id")),
        _number_or_last(metadata.get("chunk_idx")),
        str(hit.get("rid") or ""),
    )


def chronological_document_key(
    content: str,
    original_index: int,
) -> tuple[float, float, int]:
    """Parse the role-aware display prefix; unknown dates sort last."""
    match = _ROLE_AWARE_PREFIX_RE.match(content)
    if match is None:
        return (float("inf"), float("inf"), original_index)
    occurred_at = datetime.strptime(
        match.group("date"), "%B %d, %Y"
    ).replace(tzinfo=timezone.utc).timestamp()
    turn = match.group("turn")
    return (
        occurred_at,
        float(turn) if turn is not None else float("inf"),
        original_index,
    )
