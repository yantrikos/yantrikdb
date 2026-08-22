"""Combine bounded derived and raw-evidence lanes into frozen AMB contexts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MEMORY_BOUNDARY = re.compile(r"(?m)(?=^## Memory \d+\b)")
MEMORY_HEADER = re.compile(r"^## Memory \d+[^\n]*\n")


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, list) else payload.get("results") or []


def split_memory_bodies(context: str) -> list[str]:
    blocks = [
        block.strip()
        for block in MEMORY_BOUNDARY.split(context or "")
        if block.strip()
    ]
    return [MEMORY_HEADER.sub("", block, count=1).strip() for block in blocks]


def build_hybrid_rows(
    raw_rows: list[dict],
    derived_rows: list[dict],
    raw_limit: int,
    derived_limit: int,
    ordering: str = "derived_then_raw",
) -> list[dict]:
    if ordering not in {"derived_then_raw", "raw_then_derived"}:
        raise ValueError(f"unsupported hybrid ordering: {ordering!r}")
    raw_by_id = {
        str(row.get("query_id") or ""): row
        for row in raw_rows
        if row.get("query_id")
    }
    output = []
    for derived in derived_rows:
        query_id = str(derived.get("query_id") or "")
        raw = raw_by_id.get(query_id)
        if raw is None:
            raise ValueError(f"raw context missing for {query_id!r}")
        derived_bodies = [
            str(document).strip()
            for document in derived.get("documents") or []
            if str(document).strip()
        ]
        if not derived_bodies:
            derived_bodies = split_memory_bodies(
                str(derived.get("context") or "")
            )
        derived_bodies = derived_bodies[:derived_limit]
        raw_bodies = split_memory_bodies(str(raw.get("context") or ""))[:raw_limit]
        bodies = (
            raw_bodies + derived_bodies
            if ordering == "raw_then_derived"
            else derived_bodies + raw_bodies
        )
        if not bodies:
            raise ValueError(f"hybrid context is empty for {query_id!r}")
        row = dict(derived)
        row["context"] = "\n\n".join(
            f"## Memory {index}\n{body}"
            for index, body in enumerate(bodies, 1)
        )
        row["documents"] = bodies
        row["hybrid"] = {
            "derived_documents": len(derived_bodies),
            "raw_documents": len(raw_bodies),
            "ordering": ordering,
        }
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--raw-limit", type=int, default=20)
    parser.add_argument("--derived-limit", type=int, default=20)
    parser.add_argument(
        "--ordering",
        choices=("derived_then_raw", "raw_then_derived"),
        default="derived_then_raw",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.raw_limit < 1 or args.derived_limit < 1:
        parser.error("lane limits must be positive")

    results = build_hybrid_rows(
        load_rows(args.raw),
        load_rows(args.derived),
        args.raw_limit,
        args.derived_limit,
        args.ordering,
    )
    payload = {
        "config": {
            "raw_source": str(args.raw),
            "derived_source": str(args.derived),
            "raw_limit": args.raw_limit,
            "derived_limit": args.derived_limit,
            "ordering": args.ordering,
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"rows={len(results)} wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
