"""Query-intent selectors for persisted AMB synthesis timelines."""

import re
from collections import defaultdict
from collections.abc import Callable


_RELATIONSHIP_QUERY_RE = re.compile(
    r"\b(famil(?:y|ies)|relationship|relationships|partner|spouse|"
    r"parent|parents|mother|mom|father|dad|sister|brother)\b",
    re.IGNORECASE,
)
_SUPPORT_QUERY_RE = re.compile(
    r"\b(support\w*|help\w*|care|cared|encourag\w*)\b",
    re.IGNORECASE,
)
_RELATIONSHIP_ROLE_RE = re.compile(
    r"\b(famil(?:y|ies)|mom|mother|father|dad|parent|sister|brother|"
    r"wife|husband|spouse|partner|dating|boyfriend|girlfriend)\b",
    re.IGNORECASE,
)
_ACTIVE_SUPPORT_RE = re.compile(
    r"\b(help\w*|suggest\w*|advis\w*|encourag\w*|wrote|remind\w*|"
    r"told|gave)\b",
    re.IGNORECASE,
)
_PASSIVE_SUPPORT_RE = re.compile(
    r"\b(support\w*|concern\w*|check-ins?)\b",
    re.IGNORECASE,
)
_THREAD_DEDUP_STOPWORDS = {
    "a", "an", "and", "after", "asked", "by", "for", "from", "in",
    "of", "on", "the", "to", "user", "with",
}


def cap_temporal_span_items(
    extracted: dict,
    span_keys: list[str],
    per_span_target: int,
) -> dict[str, list]:
    """Apply each temporal quota before spans are flattened.

    Structured model output can exceed the requested array size. Capping only
    after concatenation lets early arrays consume the global item budget and
    silently removes late-session evidence.
    """
    return {
        key: (
            value[:per_span_target]
            if isinstance((value := extracted.get(key)), list)
            else []
        )
        for key in span_keys
    }


def is_relationship_support_query(query: str) -> bool:
    """Return true only when both relationship and support intent are explicit."""
    return bool(
        _RELATIONSHIP_QUERY_RE.search(query)
        and _SUPPORT_QUERY_RE.search(query)
    )


def is_relationship_role_timeline(text: str) -> bool:
    """Identify named-person timelines that contain a relationship role."""
    return bool(_RELATIONSHIP_ROLE_RE.search(text))


def _thread_item_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]+", text.casefold())
        if term not in _THREAD_DEDUP_STOPWORDS
    }


def deduplicate_thread_items(items: list[dict]) -> list[dict]:
    """Drop cross-axis paraphrases without collapsing distinct source facts."""
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for item in items:
        evidence_key = tuple(sorted(item.get("evidence_ids") or []))
        if evidence_key:
            groups[evidence_key].append(item)

    kept = []
    for evidence_key in sorted(groups):
        group = sorted(
            groups[evidence_key],
            key=lambda item: (
                item.get("axis") != "contributed",
                item.get("axis", ""),
                item.get("item", "").casefold(),
            ),
        )
        contributed_terms = [
            _thread_item_terms(item.get("item", ""))
            for item in group
            if item.get("axis") == "contributed"
        ]
        for item in group:
            if item.get("axis") == "contributed":
                kept.append(item)
                continue
            terms = _thread_item_terms(item.get("item", ""))
            is_paraphrase = any(
                terms
                and contributed
                and len(terms & contributed) / min(len(terms), len(contributed))
                >= 0.5
                for contributed in contributed_terms
            )
            if not is_paraphrase:
                kept.append(item)
    return kept


def select_relationship_support_children(
    hits: list[dict],
    target_count: int | None,
    extract_people: Callable[[str], set[str]],
) -> list[dict]:
    """Prefer concrete help across family members, then restore event order."""
    if target_count is None or len(hits) <= target_count:
        return hits

    people_by_rid: dict[str, set[str]] = {}
    frequency: dict[str, int] = defaultdict(int)
    for hit in hits:
        structural_people = {
            str(person).casefold()
            for person in (hit.get("metadata") or {}).get(
                "selection_entities", []
            )
            if str(person).strip()
        }
        people = structural_people or extract_people(hit.get("text", ""))
        people_by_rid[hit.get("rid", "")] = people
        for person in people:
            frequency[person] += 1

    def first_mention_key(hit: dict) -> tuple:
        metadata = hit.get("metadata") or {}
        return (
            metadata.get("first_mention_at") or float("inf"),
            metadata.get("first_mention_turn")
            if metadata.get("first_mention_turn") is not None
            else float("inf"),
            hit.get("rid", ""),
        )

    def selection_key(hit: dict) -> tuple:
        text = hit.get("text", "")
        support = 2 if _ACTIVE_SUPPORT_RE.search(text) else (
            1 if _PASSIVE_SUPPORT_RE.search(text) else 0
        )
        people = people_by_rid.get(hit.get("rid", ""), set())
        coverage = sum(1.0 / frequency[person] for person in people)
        return (
            -support,
            -coverage,
            -(hit.get("score") or 0.0),
            first_mention_key(hit),
        )

    selected = sorted(hits, key=selection_key)[:target_count]
    selected.sort(key=first_mention_key)
    return selected


def select_entity_timeline_children(
    hits: list[dict],
    anchor: str,
    target_count: int | None,
    extract_people: Callable[[str], set[str]],
) -> list[dict]:
    """Prefer direct, anchor-led relations over incidental later mentions."""
    if target_count is None or len(hits) <= target_count:
        return hits

    escaped = re.escape(anchor)
    central_relation = re.compile(
        rf"(?:\b(?:with|from|following|through|by|after|when)\s+{escaped}\b|"
        rf"\b{escaped}\s*,\s*(?:who|a|an)\b|"
        rf"\b{escaped}\s+and\s+I\b)",
        re.IGNORECASE,
    )
    possessive_relation = re.compile(
        rf"\b{escaped}(?:'s|’s)\b", re.IGNORECASE
    )

    def first_mention_key(hit: dict) -> tuple:
        metadata = hit.get("metadata") or {}
        return (
            metadata.get("first_mention_at") or float("inf"),
            metadata.get("first_mention_turn")
            if metadata.get("first_mention_turn") is not None
            else float("inf"),
            hit.get("rid", ""),
        )

    def relation_key(hit: dict) -> tuple:
        text = hit.get("text", "")
        folded = text.casefold()
        anchor_position = folded.find(anchor.casefold())
        other_positions = [
            folded.find(person.casefold())
            for person in extract_people(text)
            if person.casefold() != anchor.casefold()
            and folded.find(person.casefold()) >= 0
        ]
        anchor_led = anchor_position >= 0 and not any(
            position < anchor_position for position in other_positions
        )
        score = 1.0 if anchor_position >= 0 else 0.0
        score += 3.0 if central_relation.search(text) else 0.0
        score += 1.0 if possessive_relation.search(text) else 0.0
        score += 2.0 if anchor_led else 0.0
        return (-score, first_mention_key(hit))

    selected = sorted(hits, key=relation_key)[:target_count]
    selected.sort(key=first_mention_key)
    return selected


def merge_organizer_rollup_shards(rollups: list[dict]) -> list[dict]:
    """Expose repeated bounded shards as one logical topic at recall time."""
    grouped: dict[str, dict] = {}
    order = []
    for rollup in rollups:
        metadata = rollup.get("metadata") or {}
        label = " ".join(
            str(metadata.get("organizer_label") or "").casefold().split()
        )
        key = label or f"rid:{rollup.get('rid', '')}"
        if key not in grouped:
            merged = dict(rollup)
            merged_metadata = dict(metadata)
            merged_metadata["child_rids"] = list(
                dict.fromkeys(metadata.get("child_rids") or [])
            )
            merged_metadata["organizer_shard_rids"] = [rollup.get("rid")]
            merged["metadata"] = merged_metadata
            grouped[key] = merged
            order.append(key)
            continue

        merged = grouped[key]
        merged_metadata = merged["metadata"]
        merged_metadata["child_rids"] = list(
            dict.fromkeys(
                [
                    *merged_metadata.get("child_rids", []),
                    *metadata.get("child_rids", []),
                ]
            )
        )
        merged_metadata["organizer_shard_rids"].append(rollup.get("rid"))
        starts = [
            value
            for value in (
                merged_metadata.get("first_mention_at"),
                metadata.get("first_mention_at"),
            )
            if isinstance(value, (int, float))
        ]
        ends = [
            value
            for value in (
                merged_metadata.get("evidence_span_end_at"),
                metadata.get("evidence_span_end_at"),
            )
            if isinstance(value, (int, float))
        ]
        if starts:
            merged_metadata["first_mention_at"] = min(starts)
        if ends:
            merged_metadata["evidence_span_end_at"] = max(ends)
        if (rollup.get("score") or 0.0) > (merged.get("score") or 0.0):
            merged["score"] = rollup.get("score")
            merged["rid"] = rollup.get("rid")
            merged["text"] = rollup.get("text")
    return [grouped[key] for key in order]
