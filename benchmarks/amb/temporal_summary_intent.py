"""Query-only routing for bounded calendar summary requests."""

import re


MONTH_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_MONTH_RE = re.compile(rf"\b({'|'.join(MONTH_NAMES)})\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_ISO_DATE_RE = re.compile(r"\b(?:19|20)\d{2}-\d{2}(?:-\d{2})?\b")
_SUMMARY_RE = re.compile(r"\b(?:summarize|summary)\b", re.IGNORECASE)


def bounded_calendar_summary_intent(query: str) -> tuple[bool, dict]:
    """Identify summaries bounded by explicit calendar points using query only."""
    query = str(query or "")
    months = [match.group(1).casefold() for match in _MONTH_RE.finditer(query)]
    iso_dates = _ISO_DATE_RE.findall(query)
    years = _YEAR_RE.findall(query)
    calendar_points = [*months, *iso_dates]
    has_summary_intent = bool(_SUMMARY_RE.search(query))
    has_bounded_calendar = bool(
        len(set(calendar_points)) >= 2 or (calendar_points and years)
    )
    matched = has_summary_intent and has_bounded_calendar
    return matched, {
        "classifier": "bounded_calendar_summary_v1",
        "matched": matched,
        "has_summary_intent": has_summary_intent,
        "months": months,
        "iso_dates": iso_dates,
        "years": years,
    }
