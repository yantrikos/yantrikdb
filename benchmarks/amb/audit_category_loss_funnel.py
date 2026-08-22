#!/usr/bin/env python3
"""Build a judge-free loss funnel for high-impact BEAM categories.

The audit separates lexical source support, source-to-context retention,
context-to-answer retention, knowledge-update label chronology, and explicit
speaker-provenance leakage. These are diagnostic proxies, not benchmark scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import fmean

try:
    from .audit_knowledge_update_gold import (
        iter_turns,
        load_conversations,
        value_tokens,
    )
    from .audit_summarization_lexical_funnel import (
        load_source_documents,
        token_coverage,
        tokens,
    )
    from .reorder_speaker_first_contexts import speaker_bucket, split_memory_blocks
except ImportError:  # pragma: no cover - direct script execution
    from audit_knowledge_update_gold import (
        iter_turns,
        load_conversations,
        value_tokens,
    )
    from audit_summarization_lexical_funnel import (
        load_source_documents,
        token_coverage,
        tokens,
    )
    from reorder_speaker_first_contexts import speaker_bucket, split_memory_blocks


DEFAULT_CATEGORIES = (
    "summarization",
    "knowledge_update",
    "multi_session_reasoning",
    "abstention",
)
RUBRIC_PREFIX = re.compile(r"^LLM response should contain:\s*", re.IGNORECASE)
SOURCE_SUPPORT_THRESHOLD = 0.75
RETENTION_THRESHOLD = 0.75
BEAM_CATEGORY_WEIGHT = 0.1


def reference_items(row: dict) -> list[str]:
    """Return benchmark reference claims, excluding abstention boilerplate."""
    category = str((row.get("meta") or {}).get("question_category") or "")
    if category == "abstention":
        return []
    if category == "summarization":
        return [
            RUBRIC_PREFIX.sub("", str(item)).strip()
            for item in (row.get("meta") or {}).get("rubric") or []
        ]
    return [str(item).strip() for item in row.get("gold_answers") or [] if str(item).strip()]


def lexical_funnel(row: dict, source: str) -> list[dict]:
    source_tokens = tokens(source)
    context_tokens = tokens(str(row.get("context") or ""))
    answer_tokens = tokens(str(row.get("answer") or ""))
    category = str((row.get("meta") or {}).get("question_category") or "")
    output = []
    for index, item in enumerate(reference_items(row), 1):
        item_tokens = tokens(item)
        source_supported = item_tokens & source_tokens
        source_coverage = token_coverage(item_tokens, source_tokens)
        context_retention = token_coverage(source_supported, context_tokens)
        answer_retention = token_coverage(source_supported, answer_tokens)
        if source_coverage < SOURCE_SUPPORT_THRESHOLD:
            stage = (
                "synthesis_required"
                if category == "multi_session_reasoning"
                else "source_or_label_mismatch"
            )
        elif context_retention < RETENTION_THRESHOLD:
            stage = "retrieval_loss"
        elif answer_retention < RETENTION_THRESHOLD:
            stage = "answer_loss"
        else:
            stage = "covered"
        output.append(
            {
                "index": index,
                "item": item,
                "source_coverage": source_coverage,
                "source_normalized_context_retention": context_retention,
                "source_normalized_answer_retention": answer_retention,
                "stage": stage,
            }
        )
    return output


def speaker_provenance(row: dict) -> dict:
    """Measure answer tokens supported only by explicitly assistant-authored blocks."""
    block_tokens = {"user": set(), "assistant": set(), "unknown": set()}
    counts: Counter[str] = Counter()
    for block in split_memory_blocks(str(row.get("context") or "")):
        bucket = speaker_bucket(block)
        counts[bucket] += 1
        block_tokens[bucket].update(tokens(block))

    factual_answer_tokens = tokens(str(row.get("answer") or "")) - tokens(
        str(row.get("query") or "")
    )
    explicitly_supported = factual_answer_tokens & (
        block_tokens["user"] | block_tokens["assistant"]
    )
    assistant_only = (
        factual_answer_tokens
        & block_tokens["assistant"]
        - block_tokens["user"]
    )
    user_only = factual_answer_tokens & block_tokens["user"] - block_tokens["assistant"]
    denominator = len(explicitly_supported)
    assistant_only_ratio = len(assistant_only) / denominator if denominator else 0.0
    return {
        "blocks": sum(counts.values()),
        "user_blocks": counts["user"],
        "assistant_blocks": counts["assistant"],
        "unknown_blocks": counts["unknown"],
        "factual_answer_tokens": len(factual_answer_tokens),
        "explicitly_supported_answer_tokens": denominator,
        "assistant_only_answer_tokens": sorted(assistant_only),
        "user_only_answer_tokens": sorted(user_only),
        "assistant_only_supported_ratio": assistant_only_ratio,
        "assistant_only_dominant": (
            denominator > 0
            and assistant_only_ratio >= 0.25
            and len(assistant_only) > len(user_only)
        ),
    }


def knowledge_update_verdict(row: dict, conversation: dict) -> dict:
    """Locate exact gold and distinct predicted values in user-authored turns."""
    gold_values = value_tokens(" ".join(row.get("gold_answers") or []))
    predicted_values = value_tokens(str(row.get("answer") or "")) - gold_values
    gold_turns = []
    predicted_turns = []
    for turn in iter_turns(conversation.get("chat") or []):
        if str(turn.get("role") or "").casefold() != "user":
            continue
        turn_values = value_tokens(str(turn.get("content") or ""))
        turn_id = int(turn.get("id") or -1)
        if turn_values & gold_values:
            gold_turns.append(turn_id)
        if turn_values & predicted_values:
            predicted_turns.append(turn_id)
    if not gold_turns:
        verdict = "gold_value_not_exact_in_user"
    elif predicted_turns and max(predicted_turns) > max(gold_turns):
        verdict = "gold_precedes_later_prediction"
    else:
        verdict = "review"
    return {
        "verdict": verdict,
        "gold_values": sorted(gold_values),
        "predicted_values": sorted(predicted_values),
        "gold_turns": gold_turns,
        "predicted_turns": predicted_turns,
    }


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _mean_or_none(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def category_attribution(
    category: str,
    rows: list[dict],
    items: list[dict],
    stages: Counter,
    knowledge_verdicts: Counter,
) -> dict:
    zero_rows = [row for row in rows if row["score"] == 0.0]
    assistant_dominant = sum(
        row["speaker_provenance"]["assistant_only_dominant"] for row in zero_rows
    )
    label_defects = (
        knowledge_verdicts["gold_precedes_later_prediction"]
        + knowledge_verdicts["gold_value_not_exact_in_user"]
    )
    if category == "summarization":
        primary = "reader_compression"
    elif category == "knowledge_update":
        primary = "benchmark_label"
    elif category == "multi_session_reasoning":
        primary = "reader_set_assembly"
    elif category == "abstention":
        primary = "provenance_rendering"
    else:
        primary = "unclassified"
    return {
        "primary": primary,
        "ours": {
            "count": (
                assistant_dominant
                if category == "abstention"
                else stages["retrieval_loss"]
            ),
            "denominator": len(zero_rows) if category == "abstention" else len(items),
            "unit": "zero_score_rows" if category == "abstention" else "reference_items",
        },
        "label": {
            "count": label_defects,
            "denominator": len(zero_rows),
            "unit": "zero_score_rows",
        },
        "reader": {
            "count": (
                max(len(zero_rows) - assistant_dominant, 0)
                if category == "abstention"
                else stages["answer_loss"]
            ),
            "denominator": len(zero_rows) if category == "abstention" else len(items),
            "unit": "zero_score_rows" if category == "abstention" else "reference_items",
        },
        "synthesis_required": {
            "count": stages["synthesis_required"],
            "denominator": len(items),
            "unit": "reference_items",
        },
    }


def analyze(
    result_payload: dict,
    sources: dict[str, str],
    conversations: dict[str, dict],
    categories: set[str],
) -> dict:
    per_query = []
    for row in result_payload.get("results") or []:
        meta = row.get("meta") or {}
        category = str(meta.get("question_category") or "")
        if category not in categories:
            continue
        conversation_id = str(meta.get("conversation_id") or "")
        if conversation_id not in sources or conversation_id not in conversations:
            raise ValueError(f"missing source conversation {conversation_id!r}")
        score = float(row.get("score") or 0.0)
        item_funnel = lexical_funnel(row, sources[conversation_id])
        analyzed = {
            "query_id": row.get("query_id"),
            "category": category,
            "score": score,
            "items": item_funnel,
            "speaker_provenance": speaker_provenance(row),
        }
        if category == "knowledge_update" and score == 0.0:
            analyzed["knowledge_update"] = knowledge_update_verdict(
                row, conversations[conversation_id]
            )
        per_query.append(analyzed)

    summaries = {}
    for category in sorted(categories):
        rows = [row for row in per_query if row["category"] == category]
        items = [item for row in rows for item in row["items"]]
        stages = Counter(item["stage"] for item in items)
        zero_rows = [row for row in rows if row["score"] == 0.0]
        knowledge_verdicts = Counter(
            row["knowledge_update"]["verdict"]
            for row in zero_rows
            if "knowledge_update" in row
        )
        mean_score = _mean([row["score"] for row in rows])
        summaries[category] = {
            "rows": len(rows),
            "mean_score": mean_score,
            "equal_weight_overall_points_lost": round(
                100 * BEAM_CATEGORY_WEIGHT * (1.0 - mean_score), 6
            ),
            "zero_score_rows": len(zero_rows),
            "reference_items": len(items),
            "mean_source_coverage": _mean_or_none(
                [item["source_coverage"] for item in items]
            ),
            "mean_source_normalized_context_retention": _mean_or_none(
                [item["source_normalized_context_retention"] for item in items]
            ),
            "mean_source_normalized_answer_retention": _mean_or_none(
                [item["source_normalized_answer_retention"] for item in items]
            ),
            "item_stages": dict(sorted(stages.items())),
            "mean_assistant_only_supported_ratio": _mean(
                [
                    row["speaker_provenance"]["assistant_only_supported_ratio"]
                    for row in rows
                ]
            ),
            "zero_rows_assistant_only_dominant": sum(
                row["speaker_provenance"]["assistant_only_dominant"]
                for row in zero_rows
            ),
            "knowledge_update_zero_verdicts": dict(sorted(knowledge_verdicts.items())),
            "attribution": category_attribution(
                category, rows, items, stages, knowledge_verdicts
            ),
        }

    return {
        "protocol": "beam-category-loss-funnel-v1",
        "thresholds": {
            "source_support": SOURCE_SUPPORT_THRESHOLD,
            "context_and_answer_retention": RETENTION_THRESHOLD,
        },
        "categories": summaries,
        "results": per_query,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("documents", type=Path)
    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help="Comma-separated BEAM categories.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    categories = {value.strip() for value in args.categories.split(",") if value.strip()}
    if not categories:
        parser.error("--categories must contain at least one category")
    result_payload = json.loads(args.results.read_text(encoding="utf-8-sig"))
    output = analyze(
        result_payload,
        load_source_documents(args.documents),
        load_conversations(args.documents),
        categories,
    )
    output["run_config"] = {
        "results": str(args.results),
        "results_sha256": _sha256_file(args.results),
        "documents": str(args.documents),
        "documents_sha256": _sha256_file(args.documents),
        "categories": sorted(categories),
        "external_calls": 0,
        "synthetic_benchmark_data_only": True,
        "real_companion_memories_included": False,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["categories"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
