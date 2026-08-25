"""Build frozen AMB contexts from query-independent organizer topic cards."""

from __future__ import annotations

import argparse
import calendar
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# `datetime.UTC` is 3.11+; the repository's floor is 3.10 (requires-python >=3.10).
UTC = timezone.utc


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
YEAR_RE = re.compile(r"\b(20\d{2})\b")
MONTHS = {
    name.casefold(): number
    for number, name in enumerate(calendar.month_name)
    if name
}
QUERY_STOPWORDS = {
    "and", "around", "based", "been", "between", "can", "complete",
    "comprehensive", "conversations", "developed", "development", "different",
    "discussing", "for", "from", "give", "have", "how", "including", "into",
    "key", "main", "major", "over", "our", "past", "quick", "so", "summary",
    "summarize", "the", "through", "time", "upcoming", "various", "we", "with",
}


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


def _topic_cards(
    artifact: dict, include_singletons: bool = True
) -> list[dict]:
    fallback_ids = set(artifact.get("singleton_fallback_evidence_ids") or [])
    source_items = {
        item["id"]: item
        for item in artifact.get("input_items") or []
        if item.get("id")
    }
    cards = []
    cards_by_document = {}
    for index, handle in enumerate(artifact.get("handles") or []):
        evidence_ids = handle.get("evidence_ids") or []
        is_singleton = len(evidence_ids) == 1 and evidence_ids[0] in fallback_ids
        if is_singleton and not include_singletons:
            continue
        label = str(handle.get("label") or "Memory topic").strip()
        summary = str(handle.get("summary") or "").strip()
        if summary:
            span = _record_span(handle, source_items)
            heading = f"Topic: {label}" + (f" ({span})" if span else "")
            evidence_dates = [
                source_items[evidence_id]["date"]
                for evidence_id in evidence_ids
                if evidence_id in source_items
                and source_items[evidence_id].get("date") is not None
            ]
            document = f"{heading}\n{summary}"
            existing = cards_by_document.get(document)
            if existing is not None:
                existing["evidence_dates"] = sorted(
                    set(existing["evidence_dates"]).union(evidence_dates)
                )
                continue
            card = {
                "index": index,
                "label": label,
                "span": span,
                "document": document,
                "evidence_dates": evidence_dates,
                "is_singleton": is_singleton,
            }
            cards.append(card)
            cards_by_document[document] = card
    return cards


def topic_card_documents(
    artifact: dict, include_singletons: bool = True
) -> list[str]:
    return [
        card["document"] for card in _topic_cards(artifact, include_singletons)
    ]


def topic_index_document(
    artifact: dict, include_spans: bool = True
) -> tuple[str, int]:
    cards = [card for card in _topic_cards(artifact) if not card["is_singleton"]]
    lines = [
        f"- {card['label']}"
        + (f" ({card['span']})" if include_spans and card["span"] else "")
        for card in cards
    ]
    if not lines:
        return "", 0
    return "Memory topic index:\n" + "\n".join(lines), len(lines)


def _stem_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    for suffix in ("ing", "ed"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def _tokens(text: str) -> list[str]:
    return [
        _stem_token(token.casefold())
        for token in TOKEN_RE.findall(text or "")
        if token.casefold() not in QUERY_STOPWORDS
    ]


def _query_date_window(query: str) -> tuple[float, float] | None:
    years = [int(value) for value in YEAR_RE.findall(query)]
    months = [
        MONTHS[match.group(1).casefold()]
        for match in re.finditer(
            rf"\b({'|'.join(MONTHS)})\b", query, re.IGNORECASE
        )
    ]
    if not years or not months:
        return None
    year = years[0]
    start_month = months[0]
    end_month = months[-1]
    if end_month < start_month:
        return None
    start = datetime(year, start_month, 1, tzinfo=UTC).timestamp()
    end_day = calendar.monthrange(year, end_month)[1]
    end = datetime(year, end_month, end_day, 23, 59, 59, tzinfo=UTC).timestamp()
    return start, end


def rank_topic_cards(
    artifact: dict,
    query: str,
    limit: int,
    include_singletons: bool = True,
) -> tuple[list[str], list[dict]]:
    cards = _topic_cards(artifact, include_singletons)
    if not cards:
        return [], []
    query_terms = Counter(_tokens(query))
    document_terms = [
        _tokens(f"{card['label']} {card['document']}") for card in cards
    ]
    document_frequency = Counter(
        term for terms in document_terms for term in set(terms)
    )
    average_length = sum(map(len, document_terms)) / len(document_terms)
    date_window = _query_date_window(query)
    ranked = []
    for card, terms in zip(cards, document_terms):
        frequencies = Counter(terms)
        lexical_score = 0.0
        for term, query_frequency in query_terms.items():
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            inverse_frequency = math.log(
                1.0
                + (len(cards) - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            normalization = frequency + 1.2 * (
                0.25 + 0.75 * len(terms) / max(average_length, 1.0)
            )
            lexical_score += (
                query_frequency * inverse_frequency * frequency * 2.2 / normalization
            )
        temporal_score = 0.0
        if date_window and card["evidence_dates"]:
            in_window = sum(
                date_window[0] <= value <= date_window[1]
                for value in card["evidence_dates"]
            )
            temporal_score = 2.0 * in_window / len(card["evidence_dates"])
        score = lexical_score + temporal_score
        ranked.append(
            {
                **card,
                "score": score,
                "lexical_score": lexical_score,
                "temporal_score": temporal_score,
            }
        )
    ranked.sort(key=lambda card: (-card["score"], card["index"]))
    selected = ranked[: min(limit, len(ranked))]
    return [card["document"] for card in selected], [
        {
            "organizer_index": card["index"],
            "label": card["label"],
            "score": card["score"],
            "lexical_score": card["lexical_score"],
            "temporal_score": card["temporal_score"],
        }
        for card in selected
    ]


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
    card_limit: int | None = None,
    include_index: bool = False,
) -> list[dict]:
    all_documents = topic_card_documents(artifact, include_singletons)
    if not all_documents:
        raise ValueError("organizer artifact contains no presentable topic cards")
    prefix = f"{unit}_"
    selected = []
    for row in query_rows:
        if not str(row.get("query_id") or "").startswith(prefix):
            continue
        if query_ids is not None and str(row.get("query_id")) not in query_ids:
            continue
        if card_limit is None:
            documents = all_documents
            ranking = []
        else:
            documents, ranking = rank_topic_cards(
                artifact,
                str(row.get("query") or ""),
                card_limit,
                include_singletons,
            )
        index_includes_spans = bool(_query_date_window(str(row.get("query") or "")))
        index_document, index_count = (
            topic_index_document(artifact, index_includes_spans)
            if include_index
            else ("", 0)
        )
        if index_document:
            documents = [index_document, *documents]
        context = "\n\n".join(
            f"## Memory {index}\n{card}"
            for index, card in enumerate(documents, 1)
        )
        result = dict(row)
        result["context"] = context
        result["documents"] = documents
        result["selection"] = {
            "mode": (
                "query_ranked_topic_cards_v1"
                if card_limit is not None
                else "query_independent_topic_cards_v1"
            ),
            "handle_count": len(artifact.get("handles") or []),
            "selected_handle_count": len(documents) - int(bool(index_document)),
            "card_limit": card_limit,
            "topic_index_handle_count": index_count,
            "topic_index_includes_spans": (
                index_includes_spans if include_index else False
            ),
            "ranking": ranking,
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
    parser.add_argument("--card-limit", type=int)
    parser.add_argument("--include-index", action="store_true")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    if args.card_limit is not None and args.card_limit < 0:
        parser.error("--card-limit must be non-negative")
    if args.card_limit == 0 and not args.include_index:
        parser.error("--card-limit 0 requires --include-index")

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
        card_limit=args.card_limit,
        include_index=args.include_index,
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
