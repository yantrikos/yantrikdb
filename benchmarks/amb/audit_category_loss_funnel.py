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
    "contradiction_resolution",
    "information_extraction",
    "instruction_following",
    "preference_following",
)
RUBRIC_PREFIX = re.compile(
    r"^LLM response should (?:contain|state|mention):\s*", re.IGNORECASE
)
SOURCE_SUPPORT_THRESHOLD = 0.75
RETENTION_THRESHOLD = 0.75
BEAM_CATEGORY_WEIGHT = 0.1
BEAM_CATEGORY_COUNT = 10
TAIL_CATEGORIES = (
    "contradiction_resolution",
    "information_extraction",
    "instruction_following",
    "preference_following",
)
RUBRIC_REFERENCE_CATEGORIES = {
    "contradiction_resolution",
    "information_extraction",
    "instruction_following",
    "preference_following",
    "summarization",
}


def reference_items(row: dict) -> list[str]:
    """Return benchmark reference claims, excluding abstention boilerplate."""
    category = str((row.get("meta") or {}).get("question_category") or "")
    if category == "abstention":
        return []
    if category in RUBRIC_REFERENCE_CATEGORIES:
        items = [
            RUBRIC_PREFIX.sub("", str(item)).strip()
            for item in (row.get("meta") or {}).get("rubric") or []
            if str(item).strip()
        ]
        # The first and fourth contradiction rubrics are response directives;
        # only the middle pair are claims that should be present in memory.
        return items[1:3] if category == "contradiction_resolution" else items
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


def behavior_target_funnel(row: dict, source: str) -> dict | None:
    """Measure retrieval of the canonical standing instruction or preference."""
    metadata = row.get("meta") or {}
    category = str(metadata.get("question_category") or "")
    target_key = {
        "instruction_following": "instruction_being_tested",
        "preference_following": "preference_being_tested",
    }.get(category)
    if target_key is None:
        return None
    target = str(metadata.get(target_key) or "").strip()
    if not target:
        return None
    target_tokens = tokens(target)
    source_tokens = tokens(source)
    source_supported = target_tokens & source_tokens
    source_coverage = token_coverage(target_tokens, source_tokens)
    context_retention = token_coverage(
        source_supported, tokens(str(row.get("context") or ""))
    )
    if source_coverage < SOURCE_SUPPORT_THRESHOLD:
        stage = "source_or_label_mismatch"
    elif context_retention < RETENTION_THRESHOLD:
        stage = "retrieval_loss"
    else:
        stage = "target_retrieved"
    return {
        "target_key": target_key,
        "target": target,
        "source_coverage": source_coverage,
        "source_normalized_context_retention": context_retention,
        "stage": stage,
    }


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
    elif category == "contradiction_resolution":
        primary = "reader_conflict_resolution"
    elif category == "information_extraction":
        primary = "reader_fact_selection"
    elif category == "instruction_following":
        primary = "standing_instruction_salience"
    elif category == "preference_following":
        primary = "preference_salience"
    else:
        primary = "unclassified"
    if category in {"instruction_following", "preference_following"}:
        deficits = Counter()
        for row in rows:
            deficit = 1.0 - float(row["score"])
            target = row.get("behavior_target") or {}
            owner = {
                "retrieval_loss": "ours",
                "source_or_label_mismatch": "label",
                "target_retrieved": "reader",
            }.get(target.get("stage"), "reader")
            deficits[owner] += deficit
        denominator = sum(deficits.values())
        return {
            "primary": primary,
            **{
                owner: {
                    "count": round(deficits[owner], 6),
                    "denominator": round(denominator, 6),
                    "unit": "row_score_deficit",
                }
                for owner in ("ours", "label", "reader")
            },
            "synthesis_required": {
                "count": 0,
                "denominator": round(denominator, 6),
                "unit": "row_score_deficit",
            },
        }
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


def _attribution_share(summary: dict, owner: str) -> float:
    attribution = (summary.get("attribution") or {}).get(owner) or {}
    denominator = float(attribution.get("denominator") or 0)
    return float(attribution.get("count") or 0) / denominator if denominator else 0.0


def ceiling_estimate(result_payload: dict, summaries: dict[str, dict]) -> dict:
    """Build a conservative, loss-conserving recovery budget for the full line."""
    scores_by_category: dict[str, list[float]] = {}
    for row in result_payload.get("results") or []:
        category = str((row.get("meta") or {}).get("question_category") or "")
        if category:
            scores_by_category.setdefault(category, []).append(
                float(row.get("score") or 0.0)
            )
    category_losses = {
        category: 100 * BEAM_CATEGORY_WEIGHT * (1.0 - _mean(scores))
        for category, scores in scores_by_category.items()
    }

    def loss(category: str) -> float:
        return category_losses.get(category, 0.0)

    summarization = summaries.get("summarization") or {}
    knowledge_update = summaries.get("knowledge_update") or {}
    multi_session = summaries.get("multi_session_reasoning") or {}
    abstention = summaries.get("abstention") or {}
    tail_summaries = {
        category: summaries.get(category) or {} for category in TAIL_CATEGORIES
    }

    dead = (
        loss("knowledge_update") * _attribution_share(knowledge_update, "label")
        + loss("multi_session_reasoning")
        * _attribution_share(multi_session, "synthesis_required")
    )
    reader_shaping = (
        loss("summarization") * _attribution_share(summarization, "reader")
        + loss("multi_session_reasoning")
        * _attribution_share(multi_session, "reader")
        + loss("abstention") * _attribution_share(abstention, "reader")
        + sum(
            loss(category) * _attribution_share(summary, "reader")
            for category, summary in tail_summaries.items()
        )
    )
    ours_direct = (
        loss("event_ordering")
        + loss("temporal_reasoning")
        + loss("summarization") * _attribution_share(summarization, "ours")
        + loss("abstention") * _attribution_share(abstention, "ours")
        + sum(
            loss(category) * _attribution_share(summary, "ours")
            for category, summary in tail_summaries.items()
        )
    )
    undiagnosed_tail = sum(
        loss(category)
        for category, summary in tail_summaries.items()
        if not summary
    )
    total_loss = sum(category_losses.values())
    audited_residual = total_loss - (
        dead + reader_shaping + ours_direct + undiagnosed_tail
    )
    if audited_residual < -1e-9:
        raise ValueError("ceiling attribution double-counts benchmark loss")
    audited_residual = max(audited_residual, 0.0)
    buckets = {
        "dead_or_benchmark_integrity": dead,
        "reader_via_context_shaping": reader_shaping,
        "ours_direct_engine": ours_direct,
        "undiagnosed_tail": undiagnosed_tail,
        "audited_residual": audited_residual,
    }
    bucket_total = sum(buckets.values())
    complete_line = (
        len(scores_by_category) == BEAM_CATEGORY_COUNT
        and len({len(scores) for scores in scores_by_category.values()}) == 1
    )
    baseline_percent = 100.0 - total_loss
    recoverable = total_loss - dead
    recovery_to_90 = max(90.0 - baseline_percent, 0.0)
    shaping_sensitivity = {
        f"{rate:.1f}": 100.0 - dead - reader_shaping * (1.0 - rate)
        for rate in (0.0, 0.5, 0.7, 1.0)
    }
    return {
        "complete_equal_weight_ten_category_line": complete_line,
        "baseline_score_percent": round(baseline_percent, 6),
        "total_points_lost": round(total_loss, 6),
        "category_points_lost": {
            category: round(value, 6)
            for category, value in sorted(category_losses.items())
        },
        "buckets": {name: round(value, 6) for name, value in buckets.items()},
        "bucket_conservation_delta": round(total_loss - bucket_total, 12),
        "optimistic_ceiling_percent": round(100.0 - dead, 6),
        "points_required_to_reach_90": round(recovery_to_90, 6),
        "recoverable_points": round(recoverable, 6),
        "recoverable_share_required_to_reach_90": (
            round(recovery_to_90 / recoverable, 6) if recoverable else None
        ),
        "reader_shaping_recovery_sensitivity": {
            rate: round(projected, 6)
            for rate, projected in shaping_sensitivity.items()
        },
        "assumptions": {
            "dead": (
                "knowledge-update label-defect share plus multi-session "
                "unstated-derived-answer share"
            ),
            "reader_via_context_shaping": (
                "reader-attributed shares in every audited category"
            ),
            "ours_direct_engine": (
                "all event-ordering and temporal loss plus retrieval-attributed "
                "shares in audited categories"
            ),
            "optimistic_ceiling": "full recovery of every point not classified dead",
            "reader_shaping_sensitivity": (
                "projected score at each reader-loss recovery rate, assuming full "
                "recovery of every other non-dead bucket"
            ),
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
        behavior_target = behavior_target_funnel(row, sources[conversation_id])
        if behavior_target is not None:
            analyzed["behavior_target"] = behavior_target
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
        behavior_targets = [
            row["behavior_target"] for row in rows if "behavior_target" in row
        ]
        behavior_stages = Counter(target["stage"] for target in behavior_targets)
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
            "behavior_target_rows": len(behavior_targets),
            "mean_behavior_target_source_coverage": _mean_or_none(
                [target["source_coverage"] for target in behavior_targets]
            ),
            "mean_behavior_target_context_retention": _mean_or_none(
                [
                    target["source_normalized_context_retention"]
                    for target in behavior_targets
                ]
            ),
            "behavior_target_stages": dict(sorted(behavior_stages.items())),
            "attribution": category_attribution(
                category, rows, items, stages, knowledge_verdicts
            ),
        }

    return {
        "protocol": "beam-category-loss-funnel-v2",
        "thresholds": {
            "source_support": SOURCE_SUPPORT_THRESHOLD,
            "context_and_answer_retention": RETENTION_THRESHOLD,
        },
        "categories": summaries,
        "ceiling_estimate": ceiling_estimate(result_payload, summaries),
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
