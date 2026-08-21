"""Measure BEAM source membership inside query-independent organizer handles.

This is an oracle diagnostic, not a retrieval policy. BEAM source turn IDs are
used only after organization to measure whether one or a small union of stored
handles contains the authoritative query-level evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from statistics import fmean

try:
    from .analyze_membership_funnel import load_beam_event_sources
except ImportError:  # Direct script execution.
    from analyze_membership_funnel import load_beam_event_sources


def organizer_turn_sets(artifact: dict) -> list[dict]:
    """Resolve each organizer handle to its source-turn set."""
    atomics = {
        str(item["id"]): item
        for item in artifact.get("input_items") or []
        if isinstance(item, dict) and item.get("id")
    }
    collections = []
    for index, handle in enumerate(artifact.get("handles") or [], 1):
        evidence_ids = list(dict.fromkeys(handle.get("evidence_ids") or []))
        invalid = sorted(set(evidence_ids) - set(atomics))
        if invalid:
            raise ValueError(f"handle {index} has invalid evidence IDs: {invalid}")
        turns = {
            atomics[evidence_id].get("turn")
            for evidence_id in evidence_ids
            if atomics[evidence_id].get("turn") is not None
        }
        if turns:
            collections.append(
                {
                    "kind": "topic_handle",
                    "label": str(handle.get("label") or f"Handle {index}"),
                    "turns": turns,
                }
            )
    return collections


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anchor_turn_sets(artifact: dict, *, min_handles: int = 2) -> list[dict]:
    """Build virtual anchor unions for an upper-bound hierarchy diagnostic."""
    if min_handles < 2:
        raise ValueError("min_handles must be at least 2")
    atomics = {
        str(item["id"]): item
        for item in artifact.get("input_items") or []
        if isinstance(item, dict) and item.get("id")
    }
    anchors: dict[str, dict] = {}
    for index, handle in enumerate(artifact.get("handles") or [], 1):
        evidence_ids = {
            str(evidence_id)
            for evidence_id in handle.get("evidence_ids") or []
            if str(evidence_id) in atomics
        }
        seen = set()
        for raw_anchor in handle.get("anchor_entities") or []:
            anchor = str(raw_anchor).strip()
            key = anchor.casefold()
            if not anchor or key in seen:
                continue
            seen.add(key)
            group = anchors.setdefault(
                key,
                {"anchor": anchor, "handle_indexes": set(), "evidence_ids": set()},
            )
            group["handle_indexes"].add(index)
            group["evidence_ids"].update(evidence_ids)

    collections = []
    for group in anchors.values():
        if len(group["handle_indexes"]) < min_handles:
            continue
        turns = {
            atomics[evidence_id].get("turn")
            for evidence_id in group["evidence_ids"]
            if atomics[evidence_id].get("turn") is not None
        }
        if turns:
            collections.append(
                {
                    "kind": "virtual_anchor_union",
                    "label": str(group["anchor"]),
                    "turns": turns,
                    "source_handle_count": len(group["handle_indexes"]),
                }
            )
    return collections


def best_cover(
    gold_turns: set[int], collections: list[dict], count: int
) -> dict:
    """Return the best source-turn cover available from exactly count groups."""
    if count < 1:
        raise ValueError("count must be positive")
    if not collections:
        return {"covered": 0, "recall": 0.0, "labels": [], "turns": []}
    count = min(count, len(collections))
    best = None
    for selected in itertools.combinations(collections, count):
        turns = set().union(*(collection["turns"] for collection in selected))
        covered = turns & gold_turns
        key = (len(covered), -len(turns - gold_turns), -len(turns))
        if best is None or key > best[0]:
            best = (key, selected, covered)
    assert best is not None
    _, selected, covered = best
    return {
        "covered": len(covered),
        "recall": len(covered) / len(gold_turns) if gold_turns else 0.0,
        "labels": [collection["label"] for collection in selected],
        "kinds": [collection["kind"] for collection in selected],
        "turns": sorted(covered),
    }


def summarize(rows: list[dict], field: str, counts: tuple[int, ...]) -> dict:
    summary = {}
    for count in counts:
        values = [row[field][str(count)]["recall"] for row in rows]
        summary[str(count)] = {
            "mean_source_recall": fmean(values),
            "exact_queries": sum(value == 1.0 for value in values),
            "queries_at_least_80pct": sum(value >= 0.8 for value in values),
        }
    return summary


def analyze(
    questions: list[dict],
    artifacts: dict[str, dict],
    *,
    counts: tuple[int, ...] = (1, 2, 3),
) -> dict:
    rows = []
    for question in questions:
        unit = question["query_id"].split("_", 1)[0]
        artifact = artifacts.get(unit)
        if artifact is None:
            raise ValueError(f"missing organizer artifact for unit {unit}")
        gold = set(question["source_turn_ids"])
        topics = organizer_turn_sets(artifact)
        anchors = anchor_turn_sets(artifact)
        rows.append(
            {
                "query_id": question["query_id"],
                "source_turns": sorted(gold),
                "topic_handle_count": len(topics),
                "virtual_anchor_union_count": len(anchors),
                "topic_handles": {
                    str(count): best_cover(gold, topics, count) for count in counts
                },
                "topics_plus_virtual_anchors": {
                    str(count): best_cover(gold, [*topics, *anchors], count)
                    for count in counts
                },
            }
        )
    return {
        "protocol": "query-independent-organizer-membership-oracle-v1",
        "queries": len(rows),
        "source_turn_references": sum(len(row["source_turns"]) for row in rows),
        "gold_used_for_generation_or_retrieval": False,
        "topic_handles": summarize(rows, "topic_handles", counts),
        "topics_plus_virtual_anchors": summarize(
            rows, "topics_plus_virtual_anchors", counts
        ),
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam-source", type=Path, required=True)
    parser.add_argument("--context-artifact", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    context = json.loads(args.context_artifact.read_text(encoding="utf-8"))
    versions = context.get("organizer_versions") or {}
    if not versions:
        parser.error("context artifact has no organizer_versions mapping")
    artifact_paths = {
        str(unit): args.artifacts_dir
        / f"summarization-unit{unit}-topic-cards-deepseek-{version}.json"
        for unit, version in versions.items()
    }
    artifacts = {
        unit: json.loads(path.read_text(encoding="utf-8"))
        for unit, path in artifact_paths.items()
    }
    questions = load_beam_event_sources(args.beam_source)
    report = analyze(questions, artifacts)
    report.update(
        {
            "beam_source": str(args.beam_source),
            "beam_source_sha256": _sha256(args.beam_source),
            "context_artifact": str(args.context_artifact),
            "context_artifact_sha256": _sha256(args.context_artifact),
            "organizer_artifacts": {
                unit: {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "input_sha256": artifacts[unit].get("input_sha256"),
                }
                for unit, path in artifact_paths.items()
            },
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    print(f"wrote={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
