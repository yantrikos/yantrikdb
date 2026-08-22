"""Freeze baseline and query-decomposed temporal contexts for paired scoring."""

import argparse
import hashlib
import json
from pathlib import Path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _query_ids_sha256(query_ids: list[str]) -> str:
    encoded = json.dumps(
        query_ids, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def render_endpoint_context(row: dict, hits_per_endpoint: int) -> str:
    sections = []
    seen_rids = set()
    for index, (endpoint, lane) in enumerate(
        zip(row.get("endpoints") or [], row.get("lane_hits") or []), 1
    ):
        candidates = []
        for hit in lane:
            rid = str(hit.get("rid") or "")
            if rid and rid in seen_rids:
                continue
            if rid:
                seen_rids.add(rid)
            text = str(hit.get("text") or "").strip()
            if not text:
                continue
            candidates.append(text)
            if len(candidates) >= hits_per_endpoint:
                break
        if candidates:
            rendered = "\n\n".join(
                f"### Candidate {candidate_index}\n{text}"
                for candidate_index, text in enumerate(candidates, 1)
            )
            sections.append(
                f"## Temporal endpoint {index}\n"
                f"Query fragment: {endpoint}\n\n{rendered}"
            )
    return "\n\n".join(sections)


def main() -> int:
    from memory_bench.utils import count_tokens

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--decomposition", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hits-per-endpoint", type=int, default=2)
    parser.add_argument(
        "--endpoint-mode",
        choices=("only", "prepend"),
        default="only",
        help="Use endpoint lanes alone or prepend them to unchanged baseline context.",
    )
    parser.add_argument("--model", default="deepseek-v4-flash:0731-cloud")
    parser.add_argument("--judge-repeats", type=int, default=1)
    args = parser.parse_args()
    if args.hits_per_endpoint < 1:
        parser.error("--hits-per-endpoint must be positive")
    if args.judge_repeats < 1 or args.judge_repeats % 2 == 0:
        parser.error("--judge-repeats must be a positive odd number")

    baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8-sig"))
    baseline_rows = {
        row["query_id"]: row
        for row in baseline_payload.get("results") or []
        if row.get("query_id")
    }
    decomposition = json.loads(
        args.decomposition.read_text(encoding="utf-8")
    )
    endpoint_rows = [
        row
        for row in decomposition.get("results") or []
        if len(row.get("endpoints") or []) == 2
        and len(row.get("lane_hits") or []) == 2
        and row.get("query_id") in baseline_rows
    ]
    query_ids = [row["query_id"] for row in endpoint_rows]
    arm_a = [
        {
            "query_id": query_id,
            "context": baseline_rows[query_id].get("context") or "",
        }
        for query_id in query_ids
    ]
    arm_b = []
    for row in endpoint_rows:
        query_id = row["query_id"]
        endpoint_context = render_endpoint_context(row, args.hits_per_endpoint)
        if args.endpoint_mode == "prepend":
            endpoint_context += (
                "\n\n## Additional retrieved conversation context\n"
                + (baseline_rows[query_id].get("context") or "")
            )
        arm_b.append({"query_id": query_id, "context": endpoint_context})
    if any(not row["context"].strip() for row in [*arm_a, *arm_b]):
        raise ValueError("every paired row must have non-empty context")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path_a = args.out_dir / "temporal-baseline-contexts.json"
    path_b = args.out_dir / "temporal-endpoint-contexts.json"
    path_a.write_text(json.dumps({"results": arm_a}, indent=2), encoding="utf-8")
    path_b.write_text(json.dumps({"results": arm_b}, indent=2), encoding="utf-8")

    arms = []
    for path, rows in ((path_a, arm_a), (path_b, arm_b)):
        arms.append(
            {
                "file": path.name,
                "sha256": _sha256_file(path),
                "rows": len(rows),
                "context_tokens": sum(
                    count_tokens(row["context"]) for row in rows
                ),
                "query_ids_sha256": _query_ids_sha256(
                    [row["query_id"] for row in rows]
                ),
            }
        )
    row_count = len(query_ids)
    manifest = {
        "query_ids_encoding": "utf8-json-compact-ordered-v1",
        "synthetic_benchmark_data_only": True,
        "real_companion_memories_included": False,
        "model": args.model,
        "judge_repeats": args.judge_repeats,
        "total_context_tokens": sum(arm["context_tokens"] for arm in arms),
        "answer_calls": row_count * 2,
        "judge_calls": row_count * 2 * args.judge_repeats,
        "arms": arms,
        "decomposition_file": str(args.decomposition),
        "hits_per_endpoint": args.hits_per_endpoint,
        "endpoint_mode": args.endpoint_mode,
        "speaker_constraint": (
            endpoint_rows[0].get("speaker_constraint") if endpoint_rows else None
        ),
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"rows={row_count} baseline_tokens={arms[0]['context_tokens']} "
        f"endpoint_tokens={arms[1]['context_tokens']} "
        f"answer_calls={manifest['answer_calls']} "
        f"judge_calls={manifest['judge_calls']}"
    )
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
