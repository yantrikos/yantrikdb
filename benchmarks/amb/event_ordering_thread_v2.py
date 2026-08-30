"""Query-only routing primitives for the event-ordering thread-v2 arm."""

from __future__ import annotations

import re


_BROUGHT_UP_RE = re.compile(r"\bbrought\s+up\b", re.IGNORECASE)
_ORDER_RE = re.compile(r"\b(?:in\s+order|order\s+in\s+which)\b", re.IGNORECASE)
_CONVERSATION_RE = re.compile(r"\b(?:conversations?|chats?)\b", re.IGNORECASE)
_SCOPE_SUFFIX_RE = re.compile(
    r"\b(?:throughout|across|during|in)\s+"
    r"(?:our\s+)?(?:conversations?|chats?)\b",
    re.IGNORECASE,
)
_GENERIC_FOCUS_PREFIX_RE = re.compile(
    r"^(?:different\s+)?(?:aspects?\s+of\s+|ways\s+)?",
    re.IGNORECASE,
)


def is_event_ordering_chronology_query(query: str) -> bool:
    """Return the preregistered narrow chronology-route decision."""
    return bool(
        _BROUGHT_UP_RE.search(query)
        and _ORDER_RE.search(query)
        and _CONVERSATION_RE.search(query)
    )


def event_ordering_focus(query: str) -> str | None:
    """Extract the semantic focus without using labels, answers, or evidence."""
    if not is_event_ordering_chronology_query(query):
        return None

    after_brought_up = _BROUGHT_UP_RE.split(query, maxsplit=1)[1]
    focus = _SCOPE_SUFFIX_RE.split(after_brought_up, maxsplit=1)[0]
    focus = focus.strip(" \t\r\n,.;:?!")
    focus = _GENERIC_FOCUS_PREFIX_RE.sub("", focus, count=1)
    focus = focus.strip(" \t\r\n,.;:?!")
    return focus or None
