"""Query-independent organization of synthesized memory items.

Discovery is caller-supplied because model policy belongs at the application
boundary. YantrikDB owns the deterministic completion and durable synthesis
lifecycle that follow discovery.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Mapping, Sequence

from yantrikdb.consolidate import record_synthesis


@dataclass(frozen=True)
class TopicHandle:
    """A stable logical concern and the evidence currently assigned to it."""

    id: str
    label: str
    summary: str
    evidence_ids: tuple[str, ...] = ()
    anchor_entities: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConcernItem:
    """A durable answer-sized item assembled from one coherent concern."""

    id: str
    text: str
    evidence_ids: tuple[str, ...]
    anchor_entities: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceAssignment:
    evidence_id: str
    handle_id: str
    similarity: float


@dataclass(frozen=True)
class OrganizationPlan:
    handles: tuple[TopicHandle, ...]
    assignments: tuple[EvidenceAssignment, ...] = ()
    unassigned_evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConcernPlan:
    items: tuple[ConcernItem, ...]
    unassigned_evidence_ids: tuple[str, ...] = ()


DiscoveryCallback = Callable[[Mapping[str, str]], Sequence[TopicHandle | Mapping]]
ConcernDiscoveryCallback = Callable[
    [Mapping[str, str]], Sequence[ConcernItem | Mapping]
]

_ITEM_QUERY_RE = re.compile(
    r"\b(list|order|ordered|sequence|timeline|stages|items|aspects)\b|"
    r"\bwalk\s+me\s+through\b",
    re.IGNORECASE,
)
_ROLLUP_QUERY_RE = re.compile(
    r"\b(summarize|summarise|summary|overview|recap|themes?|patterns?|overall|"
    r"broadly)\b",
    re.IGNORECASE,
)
_CONVERSATION_ORDER_RE = re.compile(
    r"\b(brought\s+up|mentioned|asked|discussed|conversations?)\b",
    re.IGNORECASE,
)
_ITEM_COUNT_RE = re.compile(
    r"\b(?:only(?:\s+and\s+only)?|exactly)\s+"
    r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty)\b",
    re.IGNORECASE,
)
_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_ENTITY_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_QUERY_ENTITY_STOPWORDS = {
    "across",
    "ai",
    "can",
    "conversations",
    "different",
    "i",
    "list",
    "mention",
    "only",
    "order",
    "throughout",
    "walk",
}
_QUERY_FOCUS_STOPWORDS = _QUERY_ENTITY_STOPWORDS | {
    "brought",
    "could",
    "five",
    "four",
    "from",
    "have",
    "into",
    "items",
    "nine",
    "one",
    "about",
    "aspects",
    "collaboration",
    "journey",
    "personal",
    "refining",
    "statement",
    "seven",
    "six",
    "ten",
    "three",
    "through",
    "their",
    "these",
    "this",
    "what",
    "when",
    "which",
    "with",
    "ways",
    "writing",
    "two",
}


def _as_handle(value: TopicHandle | Mapping) -> TopicHandle:
    if isinstance(value, TopicHandle):
        handle = value
    elif isinstance(value, Mapping):
        handle = TopicHandle(
            id=str(value.get("id", "")),
            label=str(value.get("label", "")),
            summary=str(value.get("summary", "")),
            evidence_ids=tuple(str(item) for item in value.get("evidence_ids", ())),
            anchor_entities=tuple(
                str(item) for item in value.get("anchor_entities", ())
            ),
        )
    else:
        raise TypeError("topic handles must be TopicHandle instances or mappings")

    if not handle.id.strip():
        raise ValueError("topic handle id must be non-empty and stable across runs")
    if not handle.label.strip():
        raise ValueError(f"topic handle {handle.id!r} has an empty label")
    if not handle.summary.strip():
        raise ValueError(f"topic handle {handle.id!r} has an empty summary")
    evidence_ids = tuple(dict.fromkeys(handle.evidence_ids))
    anchor_entities = tuple(
        dict.fromkeys(entity.strip() for entity in handle.anchor_entities if entity.strip())
    )
    return replace(
        handle,
        id=handle.id.strip(),
        label=handle.label.strip(),
        summary=handle.summary.strip(),
        evidence_ids=evidence_ids,
        anchor_entities=anchor_entities,
    )


def _as_concern(value: ConcernItem | Mapping) -> ConcernItem:
    if isinstance(value, ConcernItem):
        item = value
    elif isinstance(value, Mapping):
        item = ConcernItem(
            id=str(value.get("id", "")),
            text=str(value.get("text", "")),
            evidence_ids=tuple(str(rid) for rid in value.get("evidence_ids", ())),
            anchor_entities=tuple(
                str(entity) for entity in value.get("anchor_entities", ())
            ),
        )
    else:
        raise TypeError("concern items must be ConcernItem instances or mappings")

    if not item.id.strip():
        raise ValueError("concern item id must be non-empty and stable across runs")
    if not item.text.strip():
        raise ValueError(f"concern item {item.id!r} has empty text")
    evidence_ids = tuple(dict.fromkeys(item.evidence_ids))
    if not evidence_ids:
        raise ValueError(f"concern item {item.id!r} has no evidence")
    anchor_entities = tuple(
        dict.fromkeys(
            entity.strip() for entity in item.anchor_entities if entity.strip()
        )
    )
    return replace(
        item,
        id=item.id.strip(),
        text=item.text.strip(),
        evidence_ids=evidence_ids,
        anchor_entities=anchor_entities,
    )


def validate_topic_handles(
    handles: Iterable[TopicHandle | Mapping],
    *,
    max_evidence_per_handle: int = 12,
    max_handle_memberships: int = 3,
) -> tuple[TopicHandle, ...]:
    """Normalize discovery output and enforce bounded many-to-many evidence."""
    if max_evidence_per_handle < 1:
        raise ValueError("max_evidence_per_handle must be positive")
    if max_handle_memberships < 1:
        raise ValueError("max_handle_memberships must be positive")
    normalized = tuple(_as_handle(handle) for handle in handles)
    if not normalized:
        raise ValueError("topic discovery returned no handles")

    handle_ids: set[str] = set()
    evidence_owners: dict[str, list[str]] = {}
    for handle in normalized:
        if handle.id in handle_ids:
            raise ValueError(f"duplicate topic handle id: {handle.id}")
        handle_ids.add(handle.id)
        if len(handle.evidence_ids) > max_evidence_per_handle:
            raise ValueError(
                f"topic handle {handle.id!r} has {len(handle.evidence_ids)} evidence "
                f"items; maximum is {max_evidence_per_handle}"
            )
        for evidence_id in handle.evidence_ids:
            owners = evidence_owners.setdefault(evidence_id, [])
            owners.append(handle.id)
            if len(owners) > max_handle_memberships:
                raise ValueError(
                    f"evidence {evidence_id!r} is assigned to {len(owners)} handles; "
                    f"maximum is {max_handle_memberships}: {owners!r}"
                )
    return normalized


def validate_concern_items(
    items: Iterable[ConcernItem | Mapping],
    *,
    max_evidence_per_item: int = 6,
    max_item_memberships: int = 2,
) -> tuple[ConcernItem, ...]:
    """Normalize answer-sized concerns and bound evidence reuse."""
    if max_evidence_per_item < 1:
        raise ValueError("max_evidence_per_item must be positive")
    if max_item_memberships < 1:
        raise ValueError("max_item_memberships must be positive")
    normalized = tuple(_as_concern(item) for item in items)
    if not normalized:
        raise ValueError("concern discovery returned no items")

    item_ids = set()
    evidence_owners: dict[str, list[str]] = {}
    for item in normalized:
        if item.id in item_ids:
            raise ValueError(f"duplicate concern item id: {item.id}")
        item_ids.add(item.id)
        if len(item.evidence_ids) > max_evidence_per_item:
            raise ValueError(
                f"concern item {item.id!r} has {len(item.evidence_ids)} evidence "
                f"records; maximum is {max_evidence_per_item}"
            )
        for evidence_id in item.evidence_ids:
            owners = evidence_owners.setdefault(evidence_id, [])
            owners.append(item.id)
            if len(owners) > max_item_memberships:
                raise ValueError(
                    f"evidence {evidence_id!r} supports {len(owners)} concern "
                    f"items; maximum is {max_item_memberships}: {owners!r}"
                )
    return normalized


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _finite_float(value) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _get_for_organization(db, rid: str):
    """Read organizer evidence without emitting a downstream-use label."""
    get_memory = getattr(db, "get_memory", None)
    return get_memory(rid) if get_memory is not None else db.get(rid)


def _evidence_occurrence(memory: Mapping, evidence_id: str) -> dict:
    metadata = memory.get("metadata") or {}
    created_at = _finite_float(memory.get("created_at"))
    first_mention_at = _finite_float(metadata.get("first_mention_at"))
    span_end_at = _finite_float(metadata.get("evidence_span_end_at"))
    first_mention_turn = _finite_float(metadata.get("first_mention_turn"))
    occurrence_at = first_mention_at if first_mention_at is not None else created_at
    span_end_at = span_end_at if span_end_at is not None else created_at
    return {
        "rid": evidence_id,
        "occurrence_at": occurrence_at,
        "evidence_span_end_at": span_end_at,
        "created_at": created_at,
        "first_mention_turn": first_mention_turn,
        "date_source": (
            "first_mention_at" if first_mention_at is not None else "created_at"
        ),
    }


def _occurrence_key(occurrence: Mapping) -> tuple:
    return (
        occurrence.get("occurrence_at")
        if occurrence.get("occurrence_at") is not None
        else float("inf"),
        occurrence.get("evidence_span_end_at")
        if occurrence.get("evidence_span_end_at") is not None
        else float("inf"),
        occurrence["rid"],
    )


def _organization_query_mode(query: str) -> str | None:
    asks_for_items = bool(_ITEM_QUERY_RE.search(query))
    asks_for_rollup = bool(_ROLLUP_QUERY_RE.search(query))
    if asks_for_items == asks_for_rollup:
        return None
    return "items" if asks_for_items else "handles"


def _query_handle_entity_tokens(query: str) -> set[str]:
    return {
        token.casefold()
        for token in _ENTITY_TOKEN_RE.findall(query)
        if token[:1].isupper()
        and token.casefold() not in _QUERY_ENTITY_STOPWORDS
    }


def _query_entity_handles(query: str, handles: Sequence[Mapping]) -> list[Mapping]:
    query_tokens = _query_handle_entity_tokens(query)
    if not query_tokens:
        return []
    matched = []
    for handle in handles:
        metadata = handle.get("metadata") or {}
        entities = [
            *(metadata.get("anchor_entities") or []),
            *(metadata.get("thread_entities") or []),
        ]
        entity_tokens = {
            token.casefold()
            for value in entities
            for token in _ENTITY_TOKEN_RE.findall(str(value))
        }
        label_tokens = {
            token.casefold()
            for token in _ENTITY_TOKEN_RE.findall(
                str(metadata.get("organizer_label") or "")
            )
        }
        if query_tokens & (entity_tokens | label_tokens):
            matched.append(handle)
    return matched


def _anchor_tokens(result: Mapping) -> set[str]:
    metadata = result.get("metadata") or {}
    return {
        token.casefold()
        for value in (
            *(metadata.get("anchor_entities") or []),
            *(metadata.get("thread_entities") or []),
        )
        for token in _ENTITY_TOKEN_RE.findall(str(value))
    }


def _tokens_match(left: str, right: str) -> bool:
    return left == right or (
        min(len(left), len(right)) >= 5
        and (left.startswith(right) or right.startswith(left))
    )


def _query_focus_tokens(query: str) -> set[str]:
    return {
        token.casefold()
        for token in _ENTITY_TOKEN_RE.findall(query)
        if len(token) >= 4 and token.casefold() not in _QUERY_FOCUS_STOPWORDS
    }


def _focus_match_count(query_tokens: set[str], text: str) -> int:
    text_tokens = {
        token.casefold() for token in _ENTITY_TOKEN_RE.findall(text)
    }
    return sum(
        any(_tokens_match(query_token, text_token) for text_token in text_tokens)
        for query_token in query_tokens
    )


def _entity_focus_source(
    result: Mapping,
    query_tokens: set[str],
    hits_by_rid: Mapping[str, Mapping],
    db,
) -> str | None:
    metadata = result.get("metadata") or {}
    anchor_text = " ".join(
        str(value) for value in metadata.get("anchor_entities") or []
    )
    if _focus_match_count(query_tokens, anchor_text) >= 1:
        return "anchor"
    if _focus_match_count(query_tokens, str(result.get("text") or "")) >= 1:
        return "text"
    for child_rid in metadata.get("child_rids") or []:
        child = hits_by_rid.get(str(child_rid))
        if child is None:
            child = _get_for_organization(db, str(child_rid))
        if child is not None and _focus_match_count(
            query_tokens, str(child.get("text") or "")
        ) >= 1:
            return "provenance"
    return None


def _requested_item_count(query: str) -> int | None:
    match = _ITEM_COUNT_RE.search(query)
    if match is None:
        return None
    value = match.group(1).casefold()
    return int(value) if value.isdigit() else _COUNT_WORDS[value]


def _rollup_query_shape(query: str) -> str:
    if _ROLLUP_QUERY_RE.search(query):
        return "summary"
    if _ITEM_QUERY_RE.search(query):
        if _CONVERSATION_ORDER_RE.search(query) or re.search(
            r"\b(order|ordered|sequence|timeline|stages)\b|"
            r"\bwalk\s+me\s+through\b",
            query,
            re.IGNORECASE,
        ):
            return "ordered_list"
        return "list"
    return "point"


def _query_focus_handles(query: str, handles: Sequence[Mapping]) -> list[Mapping]:
    query_tokens = _query_focus_tokens(query)
    if len(query_tokens) < 2:
        return []
    focused = []
    for handle in handles:
        metadata = handle.get("metadata") or {}
        handle_text = " ".join(
            [
                str(metadata.get("organizer_label") or ""),
                *(str(value) for value in metadata.get("anchor_entities") or []),
                *(str(value) for value in metadata.get("thread_entities") or []),
            ]
        )
        matches = _focus_match_count(query_tokens, handle_text)
        if matches >= 2:
            focused.append(handle)
    return focused


def _result_occurrence_key(result: Mapping) -> tuple:
    metadata = result.get("metadata") or {}
    occurrence = {
        "rid": str(result.get("rid") or ""),
        "occurrence_at": _finite_float(metadata.get("first_mention_at")),
        "evidence_span_end_at": _finite_float(
            metadata.get("evidence_span_end_at")
        ),
    }
    if occurrence["occurrence_at"] is None:
        occurrence["occurrence_at"] = _finite_float(result.get("created_at"))
    if occurrence["evidence_span_end_at"] is None:
        occurrence["evidence_span_end_at"] = _finite_float(
            result.get("created_at")
        )
    return _occurrence_key(occurrence)


def _result_conversation_key(result: Mapping) -> tuple:
    metadata = result.get("metadata") or {}
    turn = _finite_float(metadata.get("first_mention_turn"))
    return (
        turn if turn is not None else float("inf"),
        *_result_occurrence_key(result),
    )


def _concern_content_tokens(
    result: Mapping, excluded_tokens: set[str] | None = None
) -> set[str]:
    stopwords = _QUERY_FOCUS_STOPWORDS | {
        "asking",
        "considering",
        "could",
        "first",
        "help",
        "improve",
        "might",
        "sought",
        "that",
        "user",
        "worried",
    }
    tokens = {
        token.casefold()
        for token in _ENTITY_TOKEN_RE.findall(str(result.get("text") or ""))
        if len(token) >= 4 and token.casefold() not in stopwords
    }
    return tokens - (excluded_tokens or set())


def _deduplicate_concern_hits(
    results: Sequence[Mapping], excluded_tokens: set[str] | None = None
) -> list[Mapping]:
    selected_indexes = set()
    signatures = []
    ordered = sorted(
        enumerate(results),
        key=lambda pair: (*_result_conversation_key(pair[1]), pair[0]),
    )
    for index, result in ordered:
        metadata = result.get("metadata") or {}
        if metadata.get("organizer_kind") != "query_independent_concern":
            selected_indexes.add(index)
            signatures.append(set())
            continue
        tokens = _concern_content_tokens(result, excluded_tokens)
        duplicate = any(
            len(tokens & existing) >= 2
            and len(tokens & existing) / min(len(tokens), len(existing)) >= 0.12
            for existing in signatures
            if existing and tokens
        )
        if not duplicate:
            selected_indexes.add(index)
            signatures.append(tokens)
    return [
        result for index, result in enumerate(results) if index in selected_indexes
    ]


def _note_rollup_impressions(
    db,
    query: str,
    handles: Sequence[Mapping],
    results: Sequence[Mapping] | None = None,
):
    note_impression = getattr(db, "note_rollup_impression", None)
    note_impression_features = getattr(db, "note_rollup_impression_features", None)
    note_expansion = getattr(db, "note_rollup_expansion", None)
    note_expansion_features = getattr(db, "note_rollup_expansion_features", None)
    if note_impression is None and note_impression_features is None:
        return
    if results is not None and note_expansion is None and note_expansion_features is None:
        return
    requested_count = _requested_item_count(query)
    query_shape = _rollup_query_shape(query)
    for rank, handle in enumerate(handles):
        metadata = handle.get("metadata") or {}
        impression_args = (
            str(handle.get("rid") or ""),
            query,
            handle.get("namespace"),
            rank,
            float(handle.get("score") or 0.0),
        )
        if note_impression_features is not None:
            impression_id = note_impression_features(
                *impression_args,
                requested_count,
                query_shape,
            )
        else:
            impression_id = note_impression(*impression_args)
        if results is None:
            handle_metadata = dict(metadata)
            handle_metadata["organization_rollup_impression_id"] = impression_id
            handle["metadata"] = handle_metadata
            continue
        child_rids = {str(rid) for rid in metadata.get("child_rids") or []}
        returned_children = [
            result
            for result in results
            if str(result.get("rid") or "") in child_rids
        ]
        returned_rids = [str(result.get("rid") or "") for result in returned_children]
        if note_expansion_features is not None:
            returned_scores = [
                _finite_float(result.get("score")) for result in returned_children
            ]
            note_expansion_features(impression_id, returned_rids, returned_scores)
        else:
            note_expansion(impression_id, returned_rids)
        for result in results:
            if str(result.get("rid") or "") not in child_rids:
                continue
            result_metadata = dict(result.get("metadata") or {})
            ids = list(result_metadata.get("organization_rollup_impression_ids") or [])
            if impression_id not in ids:
                ids.append(impression_id)
            result_metadata["organization_rollup_impression_ids"] = ids
            result["metadata"] = result_metadata


def recall_organized(
    db,
    query: str,
    *,
    top_k: int = 10,
    candidate_pool: int = 1000,
    max_handles: int = 16,
    handle_weight: float = 0.0,
    mode: str = "auto",
    order: str = "auto",
    **recall_kwargs,
) -> list[dict]:
    """Recall organizer handles or their items according to query shape.

    Selection remains relevance-first. Chronological order is applied only to
    the selected item set, so early low-relevance history cannot consume the
    result budget merely because it is old.
    """
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if candidate_pool < top_k:
        raise ValueError("candidate_pool must be at least top_k")
    if max_handles < 1:
        raise ValueError("max_handles must be positive")
    if not 0.0 <= handle_weight <= 1.0:
        raise ValueError("handle_weight must be between 0 and 1")
    if mode not in {"auto", "items", "handles", "raw"}:
        raise ValueError("mode must be one of: auto, items, handles, raw")
    if order not in {
        "auto",
        "relevance",
        "first_mention",
        "chronological",
        "conversation",
    }:
        raise ValueError(
            "order must be one of: auto, relevance, first_mention, "
            "chronological, conversation"
        )
    if top_k == 0:
        return []

    recall_kwargs = dict(recall_kwargs)
    recall_kwargs.setdefault("skip_reinforce", True)
    hits = db.recall(query=query, top_k=candidate_pool, **recall_kwargs)
    selected_mode = _organization_query_mode(query) if mode == "auto" else mode
    if selected_mode in {None, "raw"}:
        return hits[:top_k]

    handles = [
        hit
        for hit in hits
        if (hit.get("metadata") or {}).get("organizer_kind")
        == "query_independent_topic"
    ]
    concerns = [
        hit
        for hit in hits
        if (hit.get("metadata") or {}).get("organizer_kind")
        == "query_independent_concern"
    ]
    if selected_mode == "items" and concerns and not handles:
        concerns = concerns[:top_k]
        selected_order = (
            "conversation"
            if order == "auto" and _CONVERSATION_ORDER_RE.search(query)
            else "first_mention" if order == "auto" else order
        )
        if selected_order == "conversation":
            concerns.sort(key=_result_conversation_key)
        elif selected_order in {"first_mention", "chronological"}:
            concerns.sort(key=_result_occurrence_key)
        return concerns
    if not handles:
        return hits[:top_k]
    if selected_mode == "handles":
        selected_handles = handles[:top_k]
        _note_rollup_impressions(db, query, selected_handles)
        return selected_handles

    focused_handles = _query_focus_handles(query, handles)
    entity_handles = _query_entity_handles(query, handles)
    if focused_handles:
        selected_handles = focused_handles[:max_handles]
    elif entity_handles:
        selected_handles = entity_handles[:max_handles]
    else:
        selected_handles = handles[: min(max_handles, 8)]
    hits_by_rid = {
        str(hit.get("rid")): hit for hit in hits if hit.get("rid") is not None
    }
    child_parents: dict[str, list[dict]] = {}
    for handle in selected_handles:
        child_ids = (handle.get("metadata") or {}).get("child_rids") or []
        for child_rid in child_ids:
            child_parents.setdefault(str(child_rid), []).append(handle)

    children = []
    for child_rid, parents in child_parents.items():
        recalled = hits_by_rid.get(child_rid)
        child = (
            dict(recalled)
            if recalled is not None
            else _get_for_organization(db, child_rid)
        )
        if child is None:
            continue
        child = dict(child)
        parent_anchor_tokens = {
            token for parent in parents for token in _anchor_tokens(parent)
        }
        if (
            focused_handles
            and parent_anchor_tokens
            and (child.get("metadata") or {}).get("organizer_kind")
            == "query_independent_concern"
            and _entity_focus_source(
                child, parent_anchor_tokens, hits_by_rid, db
            )
            is None
        ):
            continue
        parent_score = max(float(parent.get("score") or 0.0) for parent in parents)
        direct_score = child.get("score")
        child["score"] = (
            parent_score
            if direct_score is None
            else (1.0 - handle_weight) * float(direct_score)
            + handle_weight * parent_score
        )
        metadata = dict(child.get("metadata") or {})
        metadata["organization_handle_ids"] = sorted(
            {
                str((parent.get("metadata") or {}).get("organizer_handle_id"))
                for parent in parents
                if (parent.get("metadata") or {}).get("organizer_handle_id")
            }
        )
        metadata["organization_direct_score"] = direct_score
        metadata["organization_parent_score"] = parent_score
        child["metadata"] = metadata
        why = list(child.get("why_retrieved") or [])
        why.append("organized_handle_expansion")
        child["why_retrieved"] = why
        children.append(child)

    if focused_handles:
        focus_tokens = _query_focus_tokens(query)
        direct_concerns = [
            concern
            for concern in concerns
            if _focus_match_count(
                focus_tokens,
                " ".join(
                    [
                        str(concern.get("text") or ""),
                        *(
                            str(value)
                            for value in (concern.get("metadata") or {}).get(
                                "anchor_entities", []
                            )
                        ),
                    ]
                ),
            )
            >= 2
        ]
    elif entity_handles:
        entity_tokens = _query_handle_entity_tokens(query)
        direct_concerns = []
        for concern in concerns:
            source = _entity_focus_source(
                concern, entity_tokens, hits_by_rid, db
            )
            if source is None:
                continue
            concern = dict(concern)
            metadata = dict(concern.get("metadata") or {})
            metadata["organization_entity_match_source"] = source
            concern["metadata"] = metadata
            direct_concerns.append(concern)
    else:
        direct_concerns = concerns

    child_rids = {str(child.get("rid") or "") for child in children}
    direct_limit = top_k if entity_handles else min(8, top_k)
    for concern in direct_concerns[:direct_limit]:
        rid = str(concern.get("rid") or "")
        if rid in child_rids:
            continue
        child = dict(concern)
        metadata = dict(child.get("metadata") or {})
        metadata.setdefault("organization_handle_ids", [])
        metadata["organization_direct_score"] = child.get("score")
        metadata["organization_parent_score"] = None
        child["metadata"] = metadata
        why = list(child.get("why_retrieved") or [])
        why.append("organized_direct_concern")
        child["why_retrieved"] = why
        children.append(child)
        child_rids.add(rid)

    children.sort(
        key=lambda result: (
            -float(result.get("score") or 0.0),
            _result_conversation_key(result),
            str(result.get("rid") or ""),
        )
    )
    if entity_handles:
        children = _deduplicate_concern_hits(
            children, _query_handle_entity_tokens(query)
        )
        requested_count = _requested_item_count(query)
        if requested_count is not None and len(children) > requested_count:
            source_priority = {
                "anchor": 1,
                "provenance": 1,
                "text": 2,
            }
            children = sorted(
                children,
                key=lambda result: (
                    0
                    if (result.get("metadata") or {}).get(
                        "organization_handle_ids"
                    )
                    else source_priority.get(
                        (result.get("metadata") or {}).get(
                            "organization_entity_match_source"
                        ),
                        3,
                    ),
                    -float(result.get("score") or 0.0),
                    _result_conversation_key(result),
                ),
            )[:requested_count]
    children = children[:top_k]
    selected_order = (
        "conversation"
        if order == "auto" and _CONVERSATION_ORDER_RE.search(query)
        else "first_mention" if order == "auto" else order
    )
    if selected_order == "conversation":
        children.sort(key=_result_conversation_key)
    elif selected_order in {"first_mention", "chronological"}:
        children.sort(key=_result_occurrence_key)
    _note_rollup_impressions(db, query, selected_handles, children)
    return children


def assign_evidence_to_handles(
    db,
    evidence: Mapping[str, str],
    handles: Iterable[TopicHandle | Mapping],
    *,
    max_evidence_per_handle: int = 12,
    max_handle_memberships: int = 3,
    minimum_similarity: float | None = None,
    require_exhaustive: bool = True,
) -> OrganizationPlan:
    """Assign omitted evidence to the best available handles, confidently first."""
    normalized = list(
        validate_topic_handles(
            handles,
            max_evidence_per_handle=max_evidence_per_handle,
            max_handle_memberships=max_handle_memberships,
        )
    )
    clean_evidence = {str(key): str(text).strip() for key, text in evidence.items()}
    empty = sorted(key for key, text in clean_evidence.items() if not key or not text)
    if empty:
        raise ValueError(f"evidence ids and texts must be non-empty: {empty}")

    already_assigned = {
        evidence_id for handle in normalized for evidence_id in handle.evidence_ids
    }
    pending_ids = sorted(set(clean_evidence) - already_assigned)
    capacity = sum(
        max_evidence_per_handle - len(handle.evidence_ids) for handle in normalized
    )
    if require_exhaustive and len(pending_ids) > capacity:
        raise ValueError(
            f"organization has capacity for {capacity} new items but received "
            f"{len(pending_ids)}; discover overflow handles first"
        )
    if not pending_ids:
        return OrganizationPlan(tuple(normalized))

    handle_embeddings = [
        db.embed(f"{handle.label}. {handle.summary}") for handle in normalized
    ]
    item_embeddings = [db.embed(clean_evidence[evidence_id]) for evidence_id in pending_ids]
    capacities = [
        max_evidence_per_handle - len(handle.evidence_ids) for handle in normalized
    ]

    rankings = []
    for evidence_id, item_embedding in zip(pending_ids, item_embeddings):
        scored = sorted(
            (
                (_cosine_similarity(item_embedding, embedding), index)
                for index, embedding in enumerate(handle_embeddings)
                if capacities[index] > 0
            ),
            key=lambda item: (-item[0], normalized[item[1]].id),
        )
        best = scored[0][0] if scored else float("-inf")
        second = scored[1][0] if len(scored) > 1 else float("-inf")
        margin = best - second if math.isfinite(second) else float("inf")
        rankings.append((evidence_id, margin, best, scored))

    assignments: list[EvidenceAssignment] = []
    for evidence_id, _, _, scored in sorted(
        rankings, key=lambda item: (-item[1], -item[2], item[0])
    ):
        selected = next(
            (
                (score, index)
                for score, index in scored
                if capacities[index] > 0
                and (minimum_similarity is None or score >= minimum_similarity)
            ),
            None,
        )
        if selected is None:
            continue
        score, index = selected
        capacities[index] -= 1
        handle = normalized[index]
        normalized[index] = replace(
            handle, evidence_ids=(*handle.evidence_ids, evidence_id)
        )
        assignments.append(EvidenceAssignment(evidence_id, handle.id, score))

    assigned_ids = {assignment.evidence_id for assignment in assignments}
    unassigned = sorted(set(pending_ids) - assigned_ids)
    if require_exhaustive and unassigned:
        raise ValueError(
            "organization could not assign every evidence item: " + ", ".join(unassigned)
        )
    return OrganizationPlan(tuple(normalized), tuple(assignments), tuple(unassigned))


def persist_organization(
    db,
    plan: OrganizationPlan,
    *,
    axis: str = "topic",
    idempotency_prefix: str = "organizer:v1",
    max_handle_memberships: int = 3,
) -> list[dict]:
    """Persist each handle as an evidence-versioned rollup synthesis."""
    if plan.unassigned_evidence_ids:
        raise ValueError(
            "cannot persist an incomplete organization: "
            + ", ".join(plan.unassigned_evidence_ids)
        )
    handles = validate_topic_handles(
        plan.handles, max_handle_memberships=max_handle_memberships
    )
    results = []
    evidence_cache: dict[str, Mapping] = {}
    for handle in handles:
        if not handle.evidence_ids:
            raise ValueError(f"cannot persist empty topic handle {handle.id!r}")
        timeline = []
        for evidence_id in handle.evidence_ids:
            if evidence_id not in evidence_cache:
                memory = _get_for_organization(db, evidence_id)
                if memory is None:
                    raise ValueError(
                        f"topic handle {handle.id!r} references missing evidence "
                        f"{evidence_id!r}"
                    )
                evidence_cache[evidence_id] = memory
            timeline.append(
                _evidence_occurrence(evidence_cache[evidence_id], evidence_id)
            )
        timeline.sort(key=_occurrence_key)
        child_rids = [occurrence["rid"] for occurrence in timeline]
        text = f"Topic trajectory: {handle.label}. {handle.summary}"
        embedding = db.embed(text)
        results.append(
            record_synthesis(
                db,
                child_rids,
                text,
                axis,
                f"{idempotency_prefix}:{handle.id}",
                granularity="rollup",
                embedding=embedding,
                metadata={
                    "organizer_kind": "query_independent_topic",
                    "organizer_handle_id": handle.id,
                    "organizer_label": handle.label,
                    "anchor_entities": list(handle.anchor_entities),
                    "thread_entities": list(handle.anchor_entities),
                    "child_rids": child_rids,
                    "organizer_evidence_timeline": timeline,
                },
            )
        )
    return results


def _concern_turn_bounds(timeline: Sequence[Mapping]) -> tuple[int | None, int | None]:
    turns = [
        int(turn)
        for occurrence in timeline
        if (turn := occurrence.get("first_mention_turn")) is not None
    ]
    return (min(turns), max(turns)) if turns else (None, None)


def persist_concerns(
    db,
    plan: ConcernPlan,
    *,
    axis: str = "concern",
    idempotency_prefix: str = "concerns:v1",
    max_evidence_per_item: int = 6,
    max_item_memberships: int = 2,
) -> list[dict]:
    """Persist answer-sized concerns through the synthesis lifecycle."""
    items = validate_concern_items(
        plan.items,
        max_evidence_per_item=max_evidence_per_item,
        max_item_memberships=max_item_memberships,
    )
    results = []
    evidence_cache: dict[str, Mapping] = {}
    for item in items:
        timeline = []
        for evidence_id in item.evidence_ids:
            if evidence_id not in evidence_cache:
                memory = _get_for_organization(db, evidence_id)
                if memory is None:
                    raise ValueError(
                        f"concern item {item.id!r} references missing evidence "
                        f"{evidence_id!r}"
                    )
                evidence_cache[evidence_id] = memory
            timeline.append(
                _evidence_occurrence(evidence_cache[evidence_id], evidence_id)
            )
        timeline.sort(key=_occurrence_key)
        child_rids = [occurrence["rid"] for occurrence in timeline]
        first_turn, last_turn = _concern_turn_bounds(timeline)
        metadata = {
            "organizer_kind": "query_independent_concern",
            "concern_item_id": item.id,
            "anchor_entities": list(item.anchor_entities),
            "thread_entities": list(item.anchor_entities),
            "child_rids": child_rids,
            "concern_evidence_timeline": timeline,
        }
        occurrence_times = [
            occurrence["occurrence_at"]
            for occurrence in timeline
            if occurrence.get("occurrence_at") is not None
        ]
        span_end_times = [
            occurrence["evidence_span_end_at"]
            for occurrence in timeline
            if occurrence.get("evidence_span_end_at") is not None
        ]
        if occurrence_times:
            metadata["first_mention_at"] = min(occurrence_times)
        if span_end_times:
            metadata["evidence_span_end_at"] = max(span_end_times)
        if first_turn is not None:
            metadata["first_mention_turn"] = first_turn
            metadata["evidence_span_end_turn"] = last_turn
        embedding = db.embed(item.text)
        results.append(
            record_synthesis(
                db,
                child_rids,
                item.text,
                axis,
                f"{idempotency_prefix}:{item.id}",
                granularity="atomic",
                embedding=embedding,
                metadata=metadata,
            )
        )
    return results


def organize_concerns(
    db,
    evidence: Mapping[str, str],
    discover: ConcernDiscoveryCallback,
    *,
    max_evidence_per_item: int = 6,
    max_item_memberships: int = 2,
    require_exhaustive: bool = False,
    persist: bool = True,
    axis: str = "concern",
    idempotency_prefix: str = "concerns:v1",
) -> tuple[ConcernPlan, list[dict]]:
    """Discover and optionally persist query-free answer-sized concerns."""
    clean_evidence = {str(rid): str(text).strip() for rid, text in evidence.items()}
    invalid_input = sorted(
        rid for rid, text in clean_evidence.items() if not rid or not text
    )
    if invalid_input:
        raise ValueError(
            f"evidence ids and texts must be non-empty: {invalid_input}"
        )
    items = validate_concern_items(
        discover(dict(clean_evidence)),
        max_evidence_per_item=max_evidence_per_item,
        max_item_memberships=max_item_memberships,
    )
    referenced = {rid for item in items for rid in item.evidence_ids}
    invented = sorted(referenced - set(clean_evidence))
    if invented:
        raise ValueError("concern discovery invented evidence ids: " + ", ".join(invented))
    unassigned = tuple(sorted(set(clean_evidence) - referenced))
    if require_exhaustive and unassigned:
        raise ValueError(
            "concern discovery left evidence unassigned: " + ", ".join(unassigned)
        )
    plan = ConcernPlan(items, unassigned)
    writes = (
        persist_concerns(
            db,
            plan,
            axis=axis,
            idempotency_prefix=idempotency_prefix,
            max_evidence_per_item=max_evidence_per_item,
            max_item_memberships=max_item_memberships,
        )
        if persist
        else []
    )
    return plan, writes


def organize_evidence(
    db,
    evidence: Mapping[str, str],
    discover: DiscoveryCallback,
    *,
    max_evidence_per_handle: int = 12,
    max_handle_memberships: int = 3,
    minimum_similarity: float | None = None,
    persist: bool = True,
    axis: str = "topic",
    idempotency_prefix: str = "organizer:v1",
) -> tuple[OrganizationPlan, list[dict]]:
    """Run query-free discovery, deterministic completion, and optional persistence."""
    discovered = discover(dict(evidence))
    plan = assign_evidence_to_handles(
        db,
        evidence,
        discovered,
        max_evidence_per_handle=max_evidence_per_handle,
        max_handle_memberships=max_handle_memberships,
        minimum_similarity=minimum_similarity,
        require_exhaustive=True,
    )
    writes = (
        persist_organization(
            db,
            plan,
            axis=axis,
            idempotency_prefix=idempotency_prefix,
            max_handle_memberships=max_handle_memberships,
        )
        if persist
        else []
    )
    return plan, writes


__all__ = [
    "ConcernItem",
    "ConcernPlan",
    "EvidenceAssignment",
    "OrganizationPlan",
    "TopicHandle",
    "assign_evidence_to_handles",
    "organize_evidence",
    "organize_concerns",
    "persist_concerns",
    "persist_organization",
    "recall_organized",
    "validate_topic_handles",
    "validate_concern_items",
]
