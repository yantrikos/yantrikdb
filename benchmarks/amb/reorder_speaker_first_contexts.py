"""Freeze a user-first presentation arm without changing retrieved evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path


_MEMORY_BOUNDARY = re.compile(r"(?m)(?=^## Memory \d+\b)")
_EXPLICIT_USER = re.compile(
    r"(?:said by the USER|\[Speaker:\s*User\b)", re.IGNORECASE
)
_EXPLICIT_ASSISTANT = re.compile(
    r"(?:said by the ASSISTANT|\[Speaker:\s*Assistant\b)", re.IGNORECASE
)
_TURN_ROLE = re.compile(r"\]\s*(User|Assistant):", re.IGNORECASE)


def split_memory_blocks(context: str) -> list[str]:
    """Split a rendered AMB context while retaining every source character."""
    if not context:
        return []
    blocks = [block for block in _MEMORY_BOUNDARY.split(context) if block]
    return blocks if len(blocks) > 1 or blocks[0].startswith("## Memory ") else [context]


def speaker_bucket(block: str) -> str:
    """Classify a memory block from explicit provenance or its first turn marker."""
    header = block[:500]
    if _EXPLICIT_USER.search(header):
        return "user"
    if _EXPLICIT_ASSISTANT.search(header):
        return "assistant"
    match = _TURN_ROLE.search(block)
    if match:
        return match.group(1).lower()
    return "unknown"


def _body_and_trailing_space(block: str) -> tuple[str, str]:
    match = re.search(r"\s*\Z", block)
    assert match is not None
    return block[: match.start()], block[match.start() :]


def reorder_context(context: str) -> tuple[str, dict]:
    """Stable-partition blocks as user, unknown, assistant."""
    blocks = split_memory_blocks(context)
    body_and_space = [_body_and_trailing_space(block) for block in blocks]
    bodies = [body for body, _space in body_and_space]
    separators = [space for _body, space in body_and_space]
    order = sorted(
        range(len(blocks)),
        key=lambda index: (
            {"user": 0, "unknown": 1, "assistant": 2}[
                speaker_bucket(bodies[index])
            ],
            index,
        ),
    )
    reordered = "".join(
        bodies[index] + separators[position]
        for position, index in enumerate(order)
    )
    reordered_bodies = [
        _body_and_trailing_space(block)[0]
        for block in split_memory_blocks(reordered)
    ]
    if Counter(bodies) != Counter(reordered_bodies):
        raise AssertionError("memory-block multiset changed")
    if len(reordered) != len(context):
        raise AssertionError("context length changed")
    counts = Counter(speaker_bucket(body) for body in bodies)
    return reordered, {
        "blocks": len(blocks),
        "user_blocks": counts["user"],
        "unknown_blocks": counts["unknown"],
        "assistant_blocks": counts["assistant"],
        "presentation_reordered": order != list(range(len(order))),
    }


def transform(payload: dict) -> dict:
    """Return an artifact whose only per-row behavioral change is presentation."""
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("artifact must contain a results list")

    output = deepcopy(payload)
    transformed_rows = []
    reordered_rows = 0
    total_blocks = 0
    speaker_counts: Counter[str] = Counter()
    for row in rows:
        transformed = deepcopy(row)
        transformed["context"], audit = reorder_context(str(row.get("context") or ""))
        transformed_rows.append(transformed)
        reordered_rows += int(audit["presentation_reordered"])
        total_blocks += audit["blocks"]
        for speaker in ("user", "unknown", "assistant"):
            speaker_counts[speaker] += audit[f"{speaker}_blocks"]

    output["results"] = transformed_rows
    output["artifact_transform"] = {
        "name": "speaker-user-first-presentation-v1",
        "selection_changed": False,
        "llm_calls": 0,
        "audit": {
            "rows": len(rows),
            "blocks": total_blocks,
            "reordered_rows": reordered_rows,
            "user_blocks": speaker_counts["user"],
            "unknown_blocks": speaker_counts["unknown"],
            "assistant_blocks": speaker_counts["assistant"],
            "context_lengths_changed": False,
        },
    }
    return output


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    output = transform(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        f"wrote {args.output} "
        f"audit={json.dumps(output['artifact_transform']['audit'], sort_keys=True)} "
        f"sha256={_sha256(args.output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
