"""Measure BEAM source membership inside query-independent concern records.

This is an oracle diagnostic, not a retrieval policy. BEAM queries and source
turn IDs are joined only after concern generation to measure whether the
stored answer-sized records could represent the requested event set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import fmean

try:
    from .analyze_membership_funnel import load_beam_event_sources
except ImportError:  # Direct script execution.
    from analyze_membership_funnel import load_beam_event_sources


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
_COUNT_RE = re.compile(
    r"\b(?:only(?:\s+and\s+only)?|exactly)\s+"
    r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty)\b",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def requested_item_count(query: str) -> int | None:
    match = _COUNT_RE.search(query or "")
    if match is None:
        return None
    value = match.group(1).casefold()
    return int(value) if value.isdigit() else _COUNT_WORDS[value]


def concern_turn_sets(concern_artifact: dict, organizer_artifact: dict) -> list[dict]:
    """Resolve each generated concern to exact source turns."""
    concern_digest = concern_artifact.get("input_sha256")
    organizer_digest = organizer_artifact.get("input_sha256")
    if not concern_digest or concern_digest != organizer_digest:
        raise ValueError("concern and organizer artifacts do not share an input digest")

    atomics = {
        str(item["id"]): item
        for item in organizer_artifact.get("input_items") or []
        if isinstance(item, dict) and item.get("id")
    }
    collections = []
    for index, item in enumerate(concern_artifact.get("items") or [], 1):
        evidence_ids = list(dict.fromkeys(item.get("evidence_ids") or []))
        invalid = sorted(set(evidence_ids) - set(atomics))
        if invalid:
            raise ValueError(f"concern {index} has invalid evidence IDs: {invalid}")
        turns = {
            atomics[evidence_id].get("turn")
            for evidence_id in evidence_ids
            if atomics[evidence_id].get("turn") is not None
        }
        if turns:
            collections.append(
                {
                    "kind": "concern",
                    "label": str(item.get("id") or f"Concern {index}"),
                    "text": str(item.get("text") or ""),
                    "turns": turns,
                }
            )
    return collections


def best_bounded_cover(
    gold_turns: set[int], collections: list[dict], count: int
) -> dict:
    """Find exact maximum gold coverage using at most ``count`` records."""
    if count < 1:
        raise ValueError("count must be positive")
    if not gold_turns or not collections:
        return {"covered": 0, "recall": 0.0, "labels": [], "turns": []}

    ordered_gold = sorted(gold_turns)
    turn_bits = {turn: 1 << index for index, turn in enumerate(ordered_gold)}
    candidates = []
    for index, collection in enumerate(collections):
        mask = 0
        for turn in collection["turns"] & gold_turns:
            mask |= turn_bits[turn]
        if mask:
            candidates.append(
                (
                    index,
                    mask,
                    len(collection["turns"] - gold_turns),
                )
            )

    # (used, gold mask) -> (summed contamination, selected indexes). Summed
    # contamination is only a deterministic tie-break; coverage is exact.
    states: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {(0, 0): (0, ())}
    for index, mask, contamination in candidates:
        updated = dict(states)
        for (used, current_mask), (cost, selected) in states.items():
            if used >= count:
                continue
            key = (used + 1, current_mask | mask)
            value = (cost + contamination, (*selected, index))
            if key not in updated or value < updated[key]:
                updated[key] = value
        states = updated

    (_, best_mask), (_, selected) = max(
        states.items(),
        key=lambda item: (
            item[0][1].bit_count(),
            -item[1][0],
            -item[0][0],
            tuple(-index for index in item[1][1]),
        ),
    )
    covered_turns = [
        turn for turn in ordered_gold if best_mask & turn_bits[turn]
    ]
    return {
        "covered": len(covered_turns),
        "recall": len(covered_turns) / len(gold_turns),
        "labels": [collections[index]["label"] for index in selected],
        "turns": covered_turns,
    }


def analyze(
    questions: list[dict],
    concern_artifacts: dict[str, dict],
    organizer_artifacts: dict[str, dict],
    *,
    counts: tuple[int, ...] = (1, 3, 5),
) -> dict:
    rows = []
    for question in questions:
        unit = question["query_id"].split("_", 1)[0]
        if unit not in concern_artifacts or unit not in organizer_artifacts:
            continue
        gold = set(question["source_turn_ids"])
        concerns = concern_turn_sets(
            concern_artifacts[unit], organizer_artifacts[unit]
        )
        requested = requested_item_count(question.get("query") or "")
        rows.append(
            {
                "query_id": question["query_id"],
                "source_turns": sorted(gold),
                "concern_count": len(concerns),
                "requested_item_count": requested,
                "fixed_counts": {
                    str(count): best_bounded_cover(gold, concerns, count)
                    for count in counts
                },
                "requested_count": (
                    best_bounded_cover(gold, concerns, requested)
                    if requested is not None
                    else None
                ),
            }
        )

    summary = {}
    for count in counts:
        values = [row["fixed_counts"][str(count)]["recall"] for row in rows]
        summary[str(count)] = {
            "mean_source_recall": fmean(values) if values else 0.0,
            "exact_queries": sum(value == 1.0 for value in values),
        }
    requested_values = [
        row["requested_count"]["recall"]
        for row in rows
        if row["requested_count"] is not None
    ]
    return {
        "protocol": "query-independent-concern-membership-oracle-v1",
        "queries": len(rows),
        "source_turn_references": sum(len(row["source_turns"]) for row in rows),
        "gold_used_for_generation_or_retrieval": False,
        "fixed_counts": summary,
        "requested_count": {
            "queries": len(requested_values),
            "mean_source_recall": fmean(requested_values) if requested_values else 0.0,
            "exact_queries": sum(value == 1.0 for value in requested_values),
        },
        "results": rows,
    }


def _path_map(values: list[list[str]], label: str) -> dict[str, Path]:
    paths = {}
    for unit, raw_path in values:
        if unit in paths:
            raise ValueError(f"duplicate {label} unit {unit}")
        paths[unit] = Path(raw_path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam-source", type=Path, required=True)
    parser.add_argument("--concern", nargs=2, action="append", default=[])
    parser.add_argument("--organizer", nargs=2, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    concern_paths = _path_map(args.concern, "concern")
    organizer_paths = _path_map(args.organizer, "organizer")
    if not concern_paths or set(concern_paths) != set(organizer_paths):
        parser.error("--concern and --organizer must provide the same non-empty units")

    concerns = {
        unit: json.loads(path.read_text(encoding="utf-8"))
        for unit, path in concern_paths.items()
    }
    organizers = {
        unit: json.loads(path.read_text(encoding="utf-8"))
        for unit, path in organizer_paths.items()
    }
    report = analyze(
        load_beam_event_sources(args.beam_source), concerns, organizers
    )
    report.update(
        {
            "beam_source": str(args.beam_source),
            "beam_source_sha256": _sha256(args.beam_source),
            "artifacts": {
                unit: {
                    "concern": str(concern_paths[unit]),
                    "concern_sha256": _sha256(concern_paths[unit]),
                    "organizer": str(organizer_paths[unit]),
                    "organizer_sha256": _sha256(organizer_paths[unit]),
                    "input_sha256": concerns[unit].get("input_sha256"),
                }
                for unit in sorted(concern_paths)
            },
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "results"},
            indent=2,
        )
    )
    print(f"wrote={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
