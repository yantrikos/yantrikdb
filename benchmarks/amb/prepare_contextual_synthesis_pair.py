"""Freeze two contextual-synthesis artifacts for paired BEAM scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .replay_contextual_synthesis import _named_entities
except ImportError:  # pragma: no cover - direct script execution
    from replay_contextual_synthesis import _named_entities


_RELATIONSHIP_TIMELINE_QUERY_RE = re.compile(
    r"\b(mentor(?:ship)?|advis(?:or|er))\b",
    re.IGNORECASE,
)
_MENTOR_ROLE_RE = re.compile(
    r"\b(academic|mentor(?:ship)?|professor|advis\w*|feedback|"
    r"recommend\w*|review\w*)\b",
    re.IGNORECASE,
)
_PROFESSIONAL_ROLE_RE = re.compile(
    r"\b(professional|career|colleague|collaborat\w*|connect\w*|"
    r"network\w*|mentor\w*|advis\w*|recommend\w*|meeting)\b",
    re.IGNORECASE,
)
_PERSONAL_ROLE_RE = re.compile(
    r"\b(famil\w*|partner|spouse|parents?|mother|mom|father|dad|"
    r"sister|brother|support\w*|help\w*|encourag\w*)\b",
    re.IGNORECASE,
)
_FAMILY_SUPPORT_QUERY_RE = re.compile(
    r"\bfamil(?:y|ies)\b.*\b(support\w*|help\w*|encourag\w*)\b|"
    r"\b(support\w*|help\w*|encourag\w*)\b.*\bfamil(?:y|ies)\b",
    re.IGNORECASE,
)
_FAMILY_ANCHOR_RE = re.compile(
    r"\b(?:my\s+)?(?:mom|mother|father|dad|partner|spouse|sister|brother)"
    r",?\s+(?P<role_name>[A-Z][a-z]+)|"
    r"\b(?:dating|married\s+to)\s+(?P<partner_name>[A-Z][a-z]+)",
    re.IGNORECASE,
)
_ACTIVE_RELATIONSHIP_SUPPORT_RE = re.compile(
    r"\b(support\w*|help\w*|encourag\w*|remind\w*|told|gave|"
    r"care package|letters?|rehears\w*|concern\w*|check-ins?)\b",
    re.IGNORECASE,
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _query_ids_sha256(query_ids: list[str]) -> str:
    payload = json.dumps(
        query_ids, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(payload)


def render_context(items: list[dict]) -> str:
    memories = []
    for index, item in enumerate(items, 1):
        evidence = ", ".join(item.get("evidence_ids") or []) or "none"
        turn = item.get("first_mention_turn")
        position = item.get("first_mention_position")
        stamp = item.get("first_mention_date") or "unknown"
        if turn is not None:
            stamp += f" | Turn {turn}"
        if position is not None:
            stamp += f" | Mention {position}"
        confidence = float(item.get("date_confidence") or 0.0)
        memories.append(
            f"## Memory {index}\n"
            f"[{stamp}] {item.get('item') or ''}\n"
            f"Evidence: {evidence} | Date source: "
            f"{item.get('date_source') or 'unknown'} "
            f"(confidence={confidence:.1f})"
        )
    return "\n\n".join(memories)


def render_selected_evidence(preflight: dict) -> str:
    kept = sorted(
        (
            row
            for row in preflight.get("evidence_ledger") or []
            if row.get("status") == "kept"
        ),
        key=lambda row: row.get("contextual_rank", 999999),
    )
    return "\n\n".join(
        f"## Memory {index}\n{row.get('text') or ''}"
        for index, row in enumerate(kept, 1)
    )


def _thread_role_pattern(query: str) -> re.Pattern[str]:
    if re.search(
        r"\b(mentor(?:ship)?|academic|advis(?:or|er|ing))\b", query, re.I
    ):
        return _MENTOR_ROLE_RE
    if re.search(
        r"\b(professional|career|collaborat\w*|connections?|network\w*)\b",
        query,
        re.I,
    ):
        return _PROFESSIONAL_ROLE_RE
    return _PERSONAL_ROLE_RE


def select_relationship_thread(artifact: dict) -> dict:
    """Select one bounded person thread using query intent and source links."""
    query = artifact.get("query") or ""
    if not _RELATIONSHIP_TIMELINE_QUERY_RE.search(query):
        raise ValueError("query has no explicit relationship-timeline intent")

    preflight = artifact.get("preflight") or artifact
    rows = [
        row
        for row in preflight.get("evidence_ledger") or []
        if row.get("status") == "kept"
    ]
    seed_counts = Counter(
        entity
        for row in rows
        for entity in _named_entities(row.get("text") or "")
    )
    anchors = sorted(entity for entity, count in seed_counts.items() if count >= 2)
    if not anchors:
        raise ValueError("no recurring high-confidence person thread")

    role_pattern = _thread_role_pattern(query)
    threads = []
    for anchor in anchors:
        mention = re.compile(rf"\b{re.escape(anchor)}\b", re.IGNORECASE)
        direct_rows = [row for row in rows if mention.search(row.get("text") or "")]
        direct_turns = {row.get("turn") for row in direct_rows}
        members = [
            row
            for row in rows
            if row in direct_rows or row.get("parent_turn") in direct_turns
        ]
        groups: dict[tuple[str, object], list[dict]] = defaultdict(list)
        for row in members:
            if row.get("source_doc_id"):
                key = ("source_doc_id", row["source_doc_id"])
            elif row.get("created_at") is not None:
                key = ("created_at", row["created_at"])
            else:
                key = ("turn", row.get("turn"))
            groups[key].append(row)
        role_matches = sum(
            bool(role_pattern.search(row.get("text") or "")) for row in members
        )
        threads.append({
            "anchor": anchor,
            "members": members,
            "groups": groups,
            "role_matches": role_matches,
            "max_contextual_score": max(
                (row.get("contextual_score") or 0.0 for row in members),
                default=0.0,
            ),
        })

    selected = max(
        threads,
        key=lambda thread: (
            thread["role_matches"],
            len(thread["groups"]),
            len(thread["members"]),
            thread["max_contextual_score"],
            thread["anchor"],
        ),
    )
    selected["members"].sort(key=lambda row: (
        row.get("created_at")
        if row.get("created_at") is not None else float("inf"),
        row.get("turn") if row.get("turn") is not None else float("inf"),
        row.get("identity") or "",
    ))
    return selected


def render_relationship_thread(artifact: dict) -> tuple[str, dict]:
    """Render one selected person timeline as source-conversation buckets."""
    thread = select_relationship_thread(artifact)
    ordered_groups = sorted(
        thread["groups"].items(),
        key=lambda item: (
            min(
                row.get("created_at")
                if row.get("created_at") is not None else float("inf")
                for row in item[1]
            ),
            min(
                row.get("turn")
                if row.get("turn") is not None else float("inf")
                for row in item[1]
            ),
            str(item[0]),
        ),
    )
    memories = []
    source_groups = []
    for index, (group_key, rows) in enumerate(ordered_groups, 1):
        rows = sorted(rows, key=lambda row: (
            row.get("turn") if row.get("turn") is not None else float("inf"),
            row.get("identity") or "",
        ))
        source_groups.append({
            "kind": group_key[0],
            "value": group_key[1],
            "turns": [row.get("turn") for row in rows],
        })
        evidence = "\n".join(row.get("text") or "" for row in rows)
        memories.append(
            f"## Memory {index}\n"
            f"Relationship thread: {thread['anchor'].title()} | "
            f"Source conversation: {group_key[1]}\n{evidence}"
        )

    preflight = artifact.get("preflight") or artifact
    ceiling = preflight.get("source_evidence_ceiling") or {}
    gold_turns = set(ceiling.get("available_source_turns") or []) | set(
        ceiling.get("missing_source_turns") or []
    )
    selected_turns = {
        row.get("turn") for row in thread["members"] if row.get("turn") is not None
    }
    selection_payload = {
        "anchor": thread["anchor"],
        "groups": source_groups,
    }
    audit = {
        "selected_anchor": thread["anchor"],
        "role_matches": thread["role_matches"],
        "selected_row_count": len(thread["members"]),
        "source_group_count": len(source_groups),
        "selected_turns": sorted(selected_turns),
        "selected_gold_turns": sorted(gold_turns & selected_turns),
        "missing_gold_turns": sorted(gold_turns - selected_turns),
        "selection_sha256": _sha256_bytes(json.dumps(
            selection_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")),
        "source_groups": source_groups,
    }
    return "\n\n".join(memories), audit


def render_relationship_support_stages(artifact: dict) -> tuple[str, dict]:
    """Select one active family-support event per source conversation."""
    query = artifact.get("query") or ""
    if not _FAMILY_SUPPORT_QUERY_RE.search(query):
        raise ValueError("query has no explicit family-support intent")

    preflight = artifact.get("preflight") or artifact
    rows = [
        row
        for row in preflight.get("evidence_ledger") or []
        if row.get("status") == "kept"
    ]
    anchors = {
        (match.group("role_name") or match.group("partner_name")).casefold()
        for row in rows
        for match in _FAMILY_ANCHOR_RE.finditer(row.get("text") or "")
    }
    if not anchors:
        raise ValueError("no explicit family-role anchors")

    groups: dict[tuple[str, object], list[dict]] = defaultdict(list)
    for row in rows:
        text = row.get("text") or ""
        if not _ACTIVE_RELATIONSHIP_SUPPORT_RE.search(text) or not any(
            re.search(rf"\b{re.escape(anchor)}\b", text, re.IGNORECASE)
            for anchor in anchors
        ):
            continue
        if row.get("source_doc_id"):
            key = ("source_doc_id", row["source_doc_id"])
        elif row.get("created_at") is not None:
            key = ("created_at", row["created_at"])
        else:
            key = ("turn", row.get("turn"))
        groups[key].append(row)
    if not groups:
        raise ValueError("no active family-support stages")

    representatives = [
        max(
            members,
            key=lambda row: (
                row.get("contextual_score") or 0.0,
                -(row.get("turn") or 0),
                row.get("identity") or "",
            ),
        )
        for members in groups.values()
    ]
    representatives.sort(key=lambda row: (
        row.get("created_at")
        if row.get("created_at") is not None else float("inf"),
        row.get("turn") if row.get("turn") is not None else float("inf"),
        row.get("identity") or "",
    ))
    memories = []
    source_groups = []
    for index, row in enumerate(representatives, 1):
        group_value = row.get("source_doc_id") or row.get("created_at")
        source_groups.append({
            "value": group_value,
            "turn": row.get("turn"),
        })
        memories.append(
            f"## Memory {index}\n"
            f"Relationship support stage | Source conversation: {group_value}\n"
            f"{row.get('text') or ''}"
        )

    ceiling = preflight.get("source_evidence_ceiling") or {}
    gold_turns = set(ceiling.get("available_source_turns") or []) | set(
        ceiling.get("missing_source_turns") or []
    )
    selected_turns = {
        row.get("turn") for row in representatives if row.get("turn") is not None
    }
    selection_payload = {
        "anchors": sorted(anchors),
        "source_groups": source_groups,
    }
    audit = {
        "selected_anchors": sorted(anchors),
        "selected_row_count": len(representatives),
        "source_group_count": len(source_groups),
        "selected_turns": sorted(selected_turns),
        "selected_gold_turns": sorted(gold_turns & selected_turns),
        "missing_gold_turns": sorted(gold_turns - selected_turns),
        "selection_sha256": _sha256_bytes(json.dumps(
            selection_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")),
        "source_groups": source_groups,
    }
    return "\n\n".join(memories), audit


def context_row(
    artifact: dict,
    query_id: str,
    label: str,
    context_mode: str = "synthesis",
) -> dict:
    preflight = artifact.get("preflight") or artifact
    ceiling = (artifact.get("preflight") or {}).get(
        "source_evidence_ceiling"
    ) or preflight.get("source_evidence_ceiling") or {}
    thread_audit = {}
    if context_mode == "selected-evidence":
        context = render_selected_evidence(preflight)
    elif context_mode == "relationship-thread":
        context, thread_audit = render_relationship_thread(artifact)
    elif context_mode == "relationship-support-stages":
        context, thread_audit = render_relationship_support_stages(artifact)
    else:
        context = render_context(artifact.get("items") or [])
    return {
        "query_id": query_id,
        "context": context,
        "audit": {
            "label": label,
            "context_mode": context_mode,
            "input_sha256": preflight.get("input_sha256"),
            "rerank_sha256": preflight.get("rerank_sha256"),
            "candidate_bank_sha256": preflight.get("candidate_bank_sha256"),
            "gold_in_input": ceiling.get(
                "all_gold_source_turns_available"
            ),
            "available_source_turns": ceiling.get(
                "available_source_turns", []
            ),
            "missing_source_turns": ceiling.get("missing_source_turns", []),
            "candidate_gold_turns": artifact.get("candidate_gold_turns", []),
            "result_gold_turns": artifact.get("result_gold_turns", []),
            **thread_audit,
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def main() -> int:
    from memory_bench.utils import count_tokens

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-artifact", "--baseline-synthesis",
        dest="baseline_artifact", type=Path, required=True,
    )
    parser.add_argument(
        "--treatment-artifact", "--treatment-synthesis",
        dest="treatment_artifact", type=Path, required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-id", default="9_event_ordering_0")
    parser.add_argument("--label-a", default="contextual-top40")
    parser.add_argument("--label-b", default="candidate-bank50")
    parser.add_argument(
        "--model", default="deepseek-v4-flash:0731-cloud"
    )
    parser.add_argument(
        "--line-comparable",
        action="store_true",
        help="mark the pinned answer/judge model as comparable to the target line",
    )
    parser.add_argument("--judge-repeats", type=int, default=3)
    parser.add_argument(
        "--context-mode",
        choices=(
            "synthesis", "selected-evidence", "relationship-thread",
            "relationship-support-stages",
        ),
        default="synthesis",
    )
    parser.add_argument(
        "--context-mode-a",
        choices=(
            "synthesis", "selected-evidence", "relationship-thread",
            "relationship-support-stages",
        ),
    )
    parser.add_argument(
        "--context-mode-b",
        choices=(
            "synthesis", "selected-evidence", "relationship-thread",
            "relationship-support-stages",
        ),
    )
    args = parser.parse_args()
    if args.judge_repeats < 1 or args.judge_repeats % 2 == 0:
        parser.error("--judge-repeats must be a positive odd number")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = (
        args.context_mode_a or args.context_mode,
        args.context_mode_b or args.context_mode,
    )
    sources = (
        (args.baseline_artifact, args.label_a, modes[0]),
        (args.treatment_artifact, args.label_b, modes[1]),
    )
    arms = []
    audit = {}
    for source, label, context_mode in sources:
        filename = f"{_safe_name(args.query_id)}-{_safe_name(label)}.json"
        artifact = json.loads(source.read_text(encoding="utf-8"))
        row = context_row(
            artifact, args.query_id, label, context_mode
        )
        output = args.output_dir / filename
        _write_json(output, {"results": [row]})
        arms.append({
            "file": filename,
            "sha256": _sha256_file(output),
            "rows": 1,
            "context_tokens": count_tokens(row["context"]),
            "query_ids_sha256": _query_ids_sha256([args.query_id]),
        })
        audit[label] = row["audit"]

    manifest = {
        "model": args.model,
        "judge_model": args.model,
        "is_line_comparable": args.line_comparable,
        "judge_repeats": args.judge_repeats,
        "answer_calls": 2,
        "judge_calls": 2 * args.judge_repeats,
        "total_context_tokens": sum(
            arm["context_tokens"] for arm in arms
        ),
        "query_ids_encoding": "utf8-json-compact-ordered-v1",
        "synthetic_benchmark_data_only": True,
        "real_companion_memories_included": False,
        "context_mode": (
            modes[0] if modes[0] == modes[1]
            else {"arm_a": modes[0], "arm_b": modes[1]}
        ),
        "arms": arms,
        "audit": audit,
    }
    manifest_path = args.output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "arms": arms,
        "audit": audit,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
