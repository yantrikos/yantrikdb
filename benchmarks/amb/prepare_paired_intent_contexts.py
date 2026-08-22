"""Freeze a pre-registered intent cohort from two context artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


INTENT_NAME = "count-set-prefix-v1"
COUNT_SET_QUERY = re.compile(r"^(?:How many|How much|What two)\b")
EXCLUDED_CATEGORIES = frozenset(
    {"multi_session_reasoning", "temporal_reasoning"}
)


def _rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, list) else payload.get("results") or []


def is_independent_count_set_row(row: dict) -> bool:
    """Apply the locked holdout rule without consulting gold or score fields."""
    category = str((row.get("meta") or {}).get("question_category") or "")
    query = str(row.get("query") or "")
    return bool(
        category
        and category not in EXCLUDED_CATEGORIES
        and COUNT_SET_QUERY.match(query)
    )


def select_common_rows(
    rows_a: list[dict], rows_b: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Select aligned contexts in arm-A order using only query and category."""
    by_id_b = {row.get("query_id"): row for row in rows_b if row.get("query_id")}
    selected_a = []
    selected_b = []
    for row_a in rows_a:
        if not is_independent_count_set_row(row_a):
            continue
        query_id = str(row_a.get("query_id") or "")
        row_b = by_id_b.get(query_id)
        if row_b is None:
            continue
        context_a = str(row_a.get("context") or "")
        context_b = str(row_b.get("context") or "")
        if not context_a.strip() or not context_b.strip():
            continue
        selected_a.append({"query_id": query_id, "context": context_a})
        selected_b.append({"query_id": query_id, "context": context_b})
    return selected_a, selected_b


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _query_ids_sha256(rows: list[dict]) -> str:
    payload = json.dumps(
        [row["query_id"] for row in rows],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def main() -> int:
    from memory_bench.utils import count_tokens

    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts-a", type=Path, required=True)
    parser.add_argument("--contexts-b", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--model", default="deepseek-v4-flash:0731-cloud")
    parser.add_argument("--judge-repeats", type=int, default=3)
    args = parser.parse_args()
    if args.judge_repeats < 1 or args.judge_repeats % 2 == 0:
        parser.error("--judge-repeats must be a positive odd number")

    arm_a, arm_b = select_common_rows(
        _rows(args.contexts_a), _rows(args.contexts_b)
    )
    if not arm_a or len(arm_a) != len(arm_b):
        raise ValueError("paired intent contexts are empty or misaligned")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path_a = args.out_dir / f"{args.label_a}.json"
    path_b = args.out_dir / f"{args.label_b}.json"
    path_a.write_text(json.dumps({"results": arm_a}, indent=2), encoding="utf-8")
    path_b.write_text(json.dumps({"results": arm_b}, indent=2), encoding="utf-8")

    arms = []
    for path, rows in ((path_a, arm_a), (path_b, arm_b)):
        arms.append(
            {
                "file": path.name,
                "sha256": _sha256_bytes(path.read_bytes()),
                "rows": len(rows),
                "context_tokens": sum(
                    count_tokens(row["context"]) for row in rows
                ),
                "query_ids_sha256": _query_ids_sha256(rows),
            }
        )
    row_count = len(arm_a)
    manifest = {
        "query_ids_encoding": "utf8-json-compact-ordered-v1",
        "synthetic_benchmark_data_only": True,
        "real_companion_memories_included": False,
        "model": args.model,
        "judge_repeats": args.judge_repeats,
        "total_context_tokens": sum(arm["context_tokens"] for arm in arms),
        "answer_calls": row_count * 2,
        "judge_calls": row_count * 2 * args.judge_repeats,
        "intent": INTENT_NAME,
        "selection_fields": ["query", "meta.question_category"],
        "query_pattern": COUNT_SET_QUERY.pattern,
        "excluded_categories": sorted(EXCLUDED_CATEGORIES),
        "labels": [args.label_a, args.label_b],
        "source_files": [str(args.contexts_a), str(args.contexts_b)],
        "arms": arms,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"intent={INTENT_NAME} rows={row_count} "
        f"tokens_a={arms[0]['context_tokens']} "
        f"tokens_b={arms[1]['context_tokens']} "
        f"answer_calls={manifest['answer_calls']} "
        f"judge_calls={manifest['judge_calls']}"
    )
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
