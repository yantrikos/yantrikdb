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
    r"^\[(?:[A-Z][a-z]+-\d+-\d+ \| )?Turn (\d+)\]"
    r"(?: \(cont\.\))?\s+(?:User|Assistant):",
    re.IGNORECASE | re.MULTILINE,
)


def beam_header_turns(text: str) -> list[int]:
    """Return turns from exact BEAM headers, excluding body references."""
    return [int(match) for match in _BEAM_TURN_HEADER_RE.findall(text)]


def synthesized_item_evidence_sets(item: dict) -> tuple[list[str], list[str]]:
    """Normalize semantic-support and chronology evidence without conflating them."""
    evidence = item.get("evidence_ids") or item.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        evidence = []
    evidence_ids = list(dict.fromkeys(
        str(evidence_id).strip()
        for evidence_id in evidence
        if str(evidence_id).strip()
    ))

    chronology = item.get("chronology_evidence_ids")
    if isinstance(chronology, str):
        chronology = [chronology]
    if not isinstance(chronology, list):
        chronology = []
    chronology_ids = list(dict.fromkeys(
        str(evidence_id).strip()
        for evidence_id in chronology
        if str(evidence_id).strip()
    )) or evidence_ids
    return evidence_ids, chronology_ids


def merge_synthesized_evidence_sets(
    items: list[dict],
) -> tuple[list[str], list[str]]:
    """Union both evidence channels across selected source items."""
    evidence_ids: list[str] = []
    chronology_ids: list[str] = []
    for item in items:
        item_evidence, item_chronology = synthesized_item_evidence_sets(item)
        evidence_ids.extend(item_evidence)
        chronology_ids.extend(item_chronology)
    return list(dict.fromkeys(evidence_ids)), list(dict.fromkeys(chronology_ids))


def ground_ordered_items_to_candidates(
    items: list[dict],
    source_items: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Replace model-emitted provenance with evidence from linked candidates."""
    by_id = {
        str(item.get("id") or "").strip(): item
        for item in source_items
        if str(item.get("id") or "").strip()
    }
    grounded = []
    events = []
    for item in items:
        item_id = str(item.get("id") or "").strip()
        source_ids = item.get("source_item_ids") or []
        if isinstance(source_ids, str):
            source_ids = [source_ids]
        if not isinstance(source_ids, list):
            source_ids = []
        source_ids = list(dict.fromkeys(
            str(source_id).strip()
            for source_id in source_ids
            if str(source_id).strip()
        ))
        invalid_ids = [source_id for source_id in source_ids if source_id not in by_id]
        if invalid_ids:
            events.append({
                "status": "dropped_invalid_source_candidates",
                "item_id": item_id,
                "source_item_ids": invalid_ids,
            })
        source_ids = [source_id for source_id in source_ids if source_id in by_id]
        if not source_ids:
            events.append({
                "status": "rejected_missing_source_candidate",
                "item_id": item_id,
            })
            continue
        evidence_ids, chronology_ids = merge_synthesized_evidence_sets([
            by_id[source_id] for source_id in source_ids
        ])
        item["source_item_ids"] = source_ids
        item["evidence_ids"] = evidence_ids
        item["chronology_evidence_ids"] = chronology_ids
        grounded.append(item)
    return grounded, events


def first_beam_turn(text: str) -> int | None:
    """Return the earliest turn from an exact BEAM header, not body prose."""
    turns = beam_header_turns(text)
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

        chronology_ids = list(dict.fromkeys(
            str(evidence_id).strip()
            for evidence_id in item.get(
                "chronology_evidence_ids", evidence_ids
            )
            if str(evidence_id).strip() in block_temporal_keys
        )) or evidence_ids
        first_block = min(
            chronology_ids,
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
        if "chronology_evidence_ids" in item:
            item["chronology_evidence_ids"] = chronology_ids
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


def validate_synthesized_item_support_quotes(
    items: list[dict],
    block_texts: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    """Require a substantive verbatim quote from a cited evidence block.

    Literal membership proves citation integrity, not semantic entailment. It
    detects invalid quote-to-block assignments but cannot prove that the quote
    supports the generated item text.
    """
    validated = []
    events = []
    normalized_blocks = {
        block_id: " ".join(text.split()).casefold()
        for block_id, text in block_texts.items()
    }
    for item in items:
        item_id = item.get("id", "")
        quote = str(item.get("support_quote") or "").strip()
        normalized_quote = " ".join(quote.split()).casefold()
        quote_body = _BEAM_TURN_HEADER_RE.sub("", quote)
        quote_body = re.sub(
            r"^\s*(?:User|Assistant)\s*:\s*", "", quote_body,
            flags=re.IGNORECASE,
        )
        quote_words = re.findall(r"[a-z0-9]+", quote_body.casefold())
        if (
            len(normalized_quote) < 24
            or len(quote_words) < 8
            or len(quote_words) > 25
        ):
            events.append({
                "status": "rejected_missing_substantive_quote",
                "item_id": item_id,
            })
            continue

        cited_evidence_ids = list(dict.fromkeys(
            str(evidence_id).strip()
            for evidence_id in item.get("evidence_ids", [])
            if str(evidence_id).strip()
        ))
        invalid_evidence_ids = [
            evidence_id
            for evidence_id in cited_evidence_ids
            if evidence_id not in normalized_blocks
        ]
        if invalid_evidence_ids:
            events.append({
                "status": "dropped_invalid_evidence",
                "item_id": item_id,
                "evidence_ids": invalid_evidence_ids,
            })
        evidence_ids = [
            evidence_id
            for evidence_id in cited_evidence_ids
            if evidence_id in normalized_blocks
        ]
        supporting_ids = [
            evidence_id
            for evidence_id in evidence_ids
            if normalized_quote in normalized_blocks[evidence_id]
        ]
        if not supporting_ids:
            events.append({
                "status": "rejected_quote_not_in_evidence",
                "item_id": item_id,
                "support_block_id": item.get("support_block_id", ""),
            })
            continue

        unsupported_ids = [
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id not in supporting_ids
        ]
        if unsupported_ids:
            events.append({
                "status": "unverified_chronology_evidence",
                "item_id": item_id,
                "evidence_ids": unsupported_ids,
            })

        claimed_block = str(item.get("support_block_id") or "").strip()
        support_block = (
            claimed_block if claimed_block in supporting_ids else supporting_ids[0]
        )
        if claimed_block != support_block:
            events.append({
                "status": "corrected_support_block",
                "item_id": item_id,
                "before": claimed_block,
                "after": support_block,
            })
        item["support_quote"] = quote
        item["support_block_id"] = support_block
        item["chronology_evidence_ids"] = evidence_ids
        item["evidence_ids"] = supporting_ids
        validated.append(item)
    return validated, events


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
