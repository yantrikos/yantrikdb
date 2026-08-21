"""Measure summarization rubric-token loss from source to context to answer."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import json
import re
from pathlib import Path
from statistics import fmean


STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "around",
    "as",
    "at",
    "be",
    "became",
    "been",
    "before",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "including",
    "into",
    "is",
    "it",
    "later",
    "more",
    "my",
    "of",
    "on",
    "or",
    "our",
    "over",
    "should",
    "so",
    "such",
    "than",
    "that",
    "the",
    "their",
    "then",
    "through",
    "to",
    "up",
    "was",
    "we",
    "were",
    "what",
    "when",
    "while",
    "with",
    "you",
    "your",
}
RUBRIC_PREFIX = re.compile(r"^LLM response should contain:\s*", re.IGNORECASE)
TOKEN_RE = re.compile(r"\$?\d+(?:\.\d+)?%?|[^\W\d_]{3,}", re.UNICODE)


def tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in TOKEN_RE.findall(text or "")
        if token.casefold() not in STOPWORDS
    }


def token_coverage(item_tokens: set[str], corpus_tokens: set[str]) -> float:
    return len(item_tokens & corpus_tokens) / len(item_tokens) if item_tokens else 0.0


def load_source_documents(path: Path) -> dict[str, str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        rows = json.load(handle)
    grouped: dict[str, list[str]] = {}
    for row in rows:
        user_id = str(row.get("user_id") or "")
        if not user_id:
            raise ValueError("source document is missing user_id")
        grouped.setdefault(user_id, []).append(str(row.get("content") or ""))
    return {user_id: "\n".join(parts) for user_id, parts in grouped.items()}


def analyze_row(row: dict, source: str) -> dict:
    source_tokens = tokens(source)
    context_tokens = tokens(str(row.get("context") or ""))
    answer_tokens = tokens(str(row.get("answer") or ""))
    analyzed = []
    for index, raw_item in enumerate((row.get("meta") or {}).get("rubric") or [], 1):
        item = RUBRIC_PREFIX.sub("", str(raw_item)).strip()
        item_tokens = tokens(item)
        source_coverage = token_coverage(item_tokens, source_tokens)
        context_coverage = token_coverage(item_tokens, context_tokens)
        answer_coverage = token_coverage(item_tokens, answer_tokens)
        source_supported_tokens = item_tokens & source_tokens
        retrieval_ratio = (
            len(source_supported_tokens & context_tokens) / len(source_supported_tokens)
            if source_supported_tokens
            else 0.0
        )
        answer_ratio = (
            len(source_supported_tokens & answer_tokens) / len(source_supported_tokens)
            if source_supported_tokens
            else 0.0
        )
        analyzed.append(
            {
                "index": index,
                "item": item,
                "item_tokens": sorted(item_tokens),
                "source_coverage": source_coverage,
                "context_coverage": context_coverage,
                "answer_coverage": answer_coverage,
                "source_normalized_retrieval": retrieval_ratio,
                "source_normalized_answer": answer_ratio,
                "source_tokens_missing_from_context": sorted(
                    source_supported_tokens - context_tokens
                ),
                "context_tokens_missing_from_answer": sorted(
                    (source_supported_tokens & context_tokens) - answer_tokens
                ),
            }
        )
    return {
        "query_id": row.get("query_id"),
        "score": row.get("score"),
        "items": analyzed,
        "mean_source_coverage": fmean(
            item["source_coverage"] for item in analyzed
        ),
        "mean_source_normalized_retrieval": fmean(
            item["source_normalized_retrieval"] for item in analyzed
        ),
        "mean_source_normalized_answer": fmean(
            item["source_normalized_answer"] for item in analyzed
        ),
    }


def replace_contexts(rows: list[dict], context_payload: dict) -> list[dict]:
    """Replace only context text while retaining baseline rubric and score."""
    by_id = {
        str(row.get("query_id") or ""): str(row.get("context") or "")
        for row in context_payload.get("results") or []
        if row.get("query_id")
    }
    output = []
    for row in rows:
        query_id = str(row.get("query_id") or "")
        context = by_id.get(query_id)
        if context is None or not context.strip():
            raise ValueError(f"alternate context missing for {query_id!r}")
        replaced = deepcopy(row)
        replaced["context"] = context
        output.append(replaced)
    return output


def select_rows(
    payload: dict,
    max_score: float,
    query_ids: set[str] | None = None,
) -> list[dict]:
    """Select the frozen summarization cohort without reading rubric text."""
    return [
        row
        for row in payload.get("results") or []
        if (row.get("meta") or {}).get("question_category") == "summarization"
        and float(row.get("score") or 0.0) <= max_score
        and (query_ids is None or str(row.get("query_id") or "") in query_ids)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("documents", type=Path)
    parser.add_argument("--contexts", type=Path)
    parser.add_argument("--max-score", type=float, default=0.4)
    parser.add_argument(
        "--query-ids",
        help="Optional comma-separated frozen query IDs for a preflight cohort.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8-sig"))
    sources = load_source_documents(args.documents)
    query_ids = (
        {value.strip() for value in args.query_ids.split(",") if value.strip()}
        if args.query_ids
        else None
    )
    rows = select_rows(payload, args.max_score, query_ids)
    if args.contexts:
        context_payload = json.loads(args.contexts.read_text(encoding="utf-8-sig"))
        rows = replace_contexts(rows, context_payload)
    analyzed = []
    for row in rows:
        conversation_id = str((row.get("meta") or {}).get("conversation_id") or "")
        if conversation_id not in sources:
            raise ValueError(f"missing source conversation {conversation_id!r}")
        analyzed.append(analyze_row(row, sources[conversation_id]))

    items = [item for row in analyzed for item in row["items"]]
    summary = {
        "rows": len(analyzed),
        "items": len(items),
        "mean_source_coverage": fmean(item["source_coverage"] for item in items),
        "mean_source_normalized_retrieval": fmean(
            item["source_normalized_retrieval"] for item in items
        ),
        "mean_source_normalized_answer": fmean(
            item["source_normalized_answer"] for item in items
        ),
        "items_below_0_75_retrieval": sum(
            item["source_normalized_retrieval"] < 0.75 for item in items
        ),
        "items_below_0_75_answer": sum(
            item["source_normalized_answer"] < 0.75 for item in items
        ),
    }
    output = {
        "protocol": "distinctive-token-source-normalized-v1",
        "context_artifact": str(args.contexts) if args.contexts else None,
        "max_score": args.max_score,
        "summary": summary,
        "results": analyzed,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
