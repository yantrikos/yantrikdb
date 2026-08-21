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
_BEAM_TURN_HEADER_RE = re.compile(
    r"\[(?:[A-Z][a-z]+-\d+-\d+ \| )?Turn (\d+)\]",
    re.IGNORECASE,
)


def first_beam_turn(text: str) -> int | None:
    """Return the earliest turn from an exact BEAM header, not body prose."""
    turns = [int(match) for match in _BEAM_TURN_HEADER_RE.findall(text)]
    return min(turns) if turns else None


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


def ground_synthesized_item_provenance(
    items: list[dict],
    block_temporal_keys: dict[str, tuple],
    block_dates: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    """Derive first-mention provenance from each item's cited evidence.

    Generators are allowed to extract item text, but block IDs are the
    auditable source of chronology. This prevents a plausible-looking model
    turn number or date from being accepted when it disagrees with the cited
    evidence.
    """
    grounded_items = []
    events = []
    for item in items:
        cited_evidence_ids = list(dict.fromkeys(
            str(evidence_id).strip()
            for evidence_id in item.get("evidence_ids", [])
            if str(evidence_id).strip()
        ))
        evidence_ids = [
            evidence_id
            for evidence_id in cited_evidence_ids
            if evidence_id in block_temporal_keys
        ]
        invalid_evidence_ids = [
            evidence_id
            for evidence_id in cited_evidence_ids
            if evidence_id not in block_temporal_keys
        ]
        if not evidence_ids:
            events.append({
                "status": "rejected_invalid_evidence",
                "item_id": item.get("id", ""),
                "invalid_evidence_ids": invalid_evidence_ids,
            })
            continue
        if invalid_evidence_ids:
            events.append({
                "status": "dropped_invalid_evidence",
                "item_id": item.get("id", ""),
                "invalid_evidence_ids": invalid_evidence_ids,
            })

        first_block = min(
            evidence_ids,
            key=lambda evidence_id: block_temporal_keys[evidence_id],
        )
        temporal_key = block_temporal_keys[first_block]
        grounded_turn = temporal_key[1]
        if grounded_turn == 999999:
            grounded_turn = None
        grounded_date = block_dates.get(first_block, "unknown")
        before = {
            "first_mention_block_id": item.get("first_mention_block_id", ""),
            "first_mention_turn": item.get("first_mention_turn"),
            "first_mention_date": item.get("first_mention_date", "unknown"),
        }
        after = {
            "first_mention_block_id": first_block,
            "first_mention_turn": grounded_turn,
            "first_mention_date": grounded_date,
        }
        item["evidence_ids"] = evidence_ids
        item.update(after)
        if before != after:
            events.append({
                "status": "corrected",
                "item_id": item.get("id", ""),
                "before": before,
                "after": after,
            })
        grounded_items.append(item)
    return grounded_items, events


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
