"""Quote-grounded coverage probe for low-scoring AMB summarization rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != HERE]
PROTOCOL_VERSION = 5


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _query_ids_sha256(rows: list[dict]) -> str:
    payload = json.dumps(
        [row["query_id"] for row in rows],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def select_rows(payload: dict, max_score: float) -> list[dict]:
    """Select the frozen low-score cohort without inspecting rubric content."""
    return [
        row
        for row in payload.get("results") or []
        if (row.get("meta") or {}).get("question_category") == "summarization"
        and float(row.get("score") or 0.0) <= max_score
        and str(row.get("context") or "").strip()
    ]


def rubric_items(row: dict) -> list[str]:
    prefix = re.compile(r"^LLM response should contain:\s*", re.IGNORECASE)
    return [prefix.sub("", str(item)).strip() for item in (row.get("meta") or {}).get("rubric") or []]


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    return " ".join(re.findall(r"[^\W_]+|[$%]+", normalized))


def validate_verdicts(raw_items: object, expected: int, context: str) -> list[dict]:
    """Require complete indices and verify every claimed quote locally."""
    if not isinstance(raw_items, list) or len(raw_items) != expected:
        actual = len(raw_items) if isinstance(raw_items, list) else type(raw_items).__name__
        raise ValueError(f"expected {expected} verdicts, got {actual}")
    context_normalized = normalize_text(context)
    validated = []
    for position, item in enumerate(raw_items, 1):
        if not isinstance(item, dict) or item.get("index") != position:
            raise ValueError(f"invalid verdict index at position {position}")
        supported = item.get("supported")
        raw_quotes = item.get("quotes")
        if not isinstance(raw_quotes, list) or not all(
            isinstance(quote, str) for quote in raw_quotes
        ):
            raise ValueError(f"verdict {position} has no string quotes list")
        quotes = [quote.strip() for quote in raw_quotes if quote.strip()]
        if not isinstance(supported, bool):
            raise ValueError(f"verdict {position} has no boolean supported field")
        quotes_verified = [
            normalize_text(quote) in context_normalized for quote in quotes
        ]
        if supported and (not quotes or not all(quotes_verified)):
            raise ValueError(f"verdict {position} claimed unverifiable quotes")
        if not supported and quotes:
            raise ValueError(f"verdict {position} supplied quotes while unsupported")
        validated.append(
            {
                "index": position,
                "supported": supported,
                "quotes": quotes,
                "quotes_verified": quotes_verified,
            }
        )
    return validated


def build_prompt(row: dict, items: list[str]) -> str:
    numbered = "\n".join(f"{index}. {item}" for index, item in enumerate(items, 1))
    return f"""You are auditing retrieval coverage for a synthetic memory benchmark.

For each RUBRIC ITEM, decide whether the MEMORY CONTEXT explicitly supports the
whole factual claim. Semantic paraphrases count, but related facts that omit a
required entity, action, number, or relationship do not.

For supported=true, copy one to four exact 5-25 word quotes from MEMORY CONTEXT
that collectively prove every part of the claim. A claim may require evidence
from several memories. For supported=false, use an empty quotes list. Do not
infer facts or mark partial support as complete support.
Return every item exactly once and in numeric order as:
{{"items":[{{"index":1,"supported":true,"quotes":["exact context words"]}}]}}

RUBRIC ITEMS
{numbered}

MEMORY CONTEXT
{row['context']}
"""


def _write_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    from memory_bench.llm.base import Schema
    from memory_bench.llm.ollama import OllamaLLM
    from memory_bench.utils import count_tokens

    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--max-score", type=float, default=0.4)
    parser.add_argument("--model", default="deepseek-v4-flash:0731-cloud")
    parser.add_argument("--num-ctx", type=int)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.preflight_only and args.out is None:
        parser.error("--out is required unless --preflight-only is set")

    payload = json.loads(args.results.read_text(encoding="utf-8-sig"))
    rows = select_rows(payload, args.max_score)
    if not rows:
        parser.error("selected summarization cohort is empty")
    run_config = {
        "results_sha256": _sha256_bytes(args.results.read_bytes()),
        "protocol_version": PROTOCOL_VERSION,
        "query_ids_sha256": _query_ids_sha256(rows),
        "query_ids": [row["query_id"] for row in rows],
        "max_score": args.max_score,
        "model": args.model,
        "num_ctx": args.num_ctx,
        "think": False if args.no_think else None,
        "rows": len(rows),
        "rubric_items": sum(len(rubric_items(row)) for row in rows),
        "context_tokens": sum(count_tokens(str(row["context"])) for row in rows),
        "external_calls": len(rows),
        "synthetic_benchmark_data_only": True,
        "real_companion_memories_included": False,
    }
    run_fingerprint = _sha256_bytes(
        json.dumps(run_config, sort_keys=True, separators=(",", ":")).encode()
    )
    print(json.dumps({**run_config, "run_fingerprint": run_fingerprint}, indent=2))
    if args.preflight_only:
        return 0

    completed = {}
    if args.resume and args.out.exists():
        prior = json.loads(args.out.read_text(encoding="utf-8"))
        if prior.get("run_fingerprint") != run_fingerprint:
            parser.error("resume artifact does not match the frozen run fingerprint")
        completed = {row["query_id"]: row for row in prior.get("results") or []}

    llm = OllamaLLM(
        args.model,
        num_ctx=args.num_ctx,
        think=False if args.no_think else None,
    )
    items_schema = Schema(
        properties={
            "items": {
                "type": "array",
                "description": "One support verdict per numbered rubric item.",
            }
        },
        required=["items"],
    )
    for row in rows:
        query_id = row["query_id"]
        if query_id in completed:
            continue
        items = rubric_items(row)
        response = llm.generate(build_prompt(row, items), items_schema)
        try:
            verdicts = validate_verdicts(
                response.get("items"), len(items), row["context"]
            )
        except ValueError:
            print(
                f"invalid_response query={query_id}: "
                f"{llm.last_response_content[:2000]!r}",
                file=sys.stderr,
            )
            raise
        completed[query_id] = {
            "query_id": query_id,
            "score": row.get("score"),
            "rubric_items": items,
            "verdicts": verdicts,
            "supported": sum(item["supported"] for item in verdicts),
            "total": len(verdicts),
        }
        ordered = [completed[r["query_id"]] for r in rows if r["query_id"] in completed]
        _write_checkpoint(
            args.out,
            {
                "run_config": run_config,
                "run_fingerprint": run_fingerprint,
                "results": ordered,
            },
        )
        print(f"completed={len(ordered)}/{len(rows)} query={query_id}", flush=True)

    ordered = [completed[row["query_id"]] for row in rows]
    supported = sum(row["supported"] for row in ordered)
    total = sum(row["total"] for row in ordered)
    output = {
        "run_config": run_config,
        "run_fingerprint": run_fingerprint,
        "summary": {
            "rows": len(ordered),
            "supported_items": supported,
            "total_items": total,
            "support_rate": supported / total if total else 0.0,
            "fully_supported_rows": sum(row["supported"] == row["total"] for row in ordered),
        },
        "results": ordered,
    }
    _write_checkpoint(args.out, output)
    print(json.dumps(output["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
