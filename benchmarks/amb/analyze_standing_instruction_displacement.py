#!/usr/bin/env python3
"""Diagnose whole-block displacement in the standing-instruction full-400 arm."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections.abc import Callable
from pathlib import Path

try:
    from .reorder_speaker_first_contexts import split_memory_blocks
except ImportError:  # pragma: no cover - direct script execution
    from reorder_speaker_first_contexts import split_memory_blocks


INSTRUCTION_CATEGORY = "instruction_following"
WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "again",
    "along",
    "already",
    "always",
    "another",
    "about",
    "after",
    "against",
    "around",
    "also",
    "because",
    "based",
    "before",
    "being",
    "between",
    "both",
    "came",
    "comes",
    "could",
    "during",
    "each",
    "first",
    "from",
    "have",
    "having",
    "however",
    "into",
    "later",
    "more",
    "most",
    "next",
    "only",
    "other",
    "overall",
    "same",
    "should",
    "some",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "through",
    "under",
    "user",
    "using",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "your",
}


def _tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


def _significant_terms(text: str) -> set[str]:
    return {word for word in _tokens(text) if len(word) >= 4 and word not in STOPWORDS}


def _significant_bigrams(text: str) -> set[str]:
    words = _tokens(text)
    return {
        f"{left} {right}"
        for left, right in zip(words, words[1:])
        if left not in STOPWORDS
        and right not in STOPWORDS
        and len(left) >= 4
        and len(right) >= 4
    }


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _summary(rows: list[dict]) -> dict:
    deltas = [float(row["score_delta_b_minus_a"]) for row in rows]
    return {
        "n": len(rows),
        "mean_score_delta_b_minus_a": _mean(deltas),
        "wins_b": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "wins_a": sum(delta < 0 for delta in deltas),
        "mean_displaced_blocks": _mean(
            [float(row["displaced_blocks"]) for row in rows]
        ),
        "mean_displaced_tokens": _mean(
            [float(row["displaced_tokens"]) for row in rows]
        ),
        "rows_with_removed_only_gold_terms": sum(
            bool(row["removed_only_gold_terms"]) for row in rows
        ),
        "rows_with_removed_only_gold_bigrams": sum(
            bool(row["removed_only_gold_bigrams"]) for row in rows
        ),
    }


def analyze(
    source_rows: list[dict],
    treatment_rows: list[dict],
    result: dict,
    token_counter: Callable[[str], int],
) -> dict:
    """Join frozen artifacts and quantify the source context evicted per row."""
    source_by_id = {str(row["query_id"]): row for row in source_rows}
    treatment_by_id = {str(row["query_id"]): row for row in treatment_rows}
    pairs_by_id = {str(pair["query_id"]): pair for pair in result.get("pairs") or []}
    query_ids = set(source_by_id)
    if not query_ids or query_ids != set(treatment_by_id) or query_ids != set(pairs_by_id):
        raise ValueError("source, treatment, and paired-result query IDs must match")

    audits = []
    for query_id in sorted(query_ids):
        source = source_by_id[query_id]
        treatment = treatment_by_id[query_id]
        pair = pairs_by_id[query_id]
        source_blocks = split_memory_blocks(str(source.get("context") or ""))
        treatment_blocks = split_memory_blocks(str(treatment.get("context") or ""))
        if not treatment_blocks or not treatment_blocks[0].startswith("## Memory 0\n"):
            raise ValueError(f"{query_id}: treatment is missing the instruction panel")

        retained_blocks = treatment_blocks[1:]
        if retained_blocks != source_blocks[: len(retained_blocks)]:
            raise ValueError(f"{query_id}: treatment is not panel + source prefix")
        removed_blocks = source_blocks[len(retained_blocks) :]
        if not removed_blocks:
            raise ValueError(f"{query_id}: expected at least one displaced source block")

        retained_text = "".join(retained_blocks).casefold()
        removed_text = "".join(removed_blocks).casefold()
        gold_parts = [str(value) for value in source.get("gold_answers") or []]
        gold_parts.extend(str(value) for value in (source.get("meta") or {}).get("rubric") or [])
        gold_text = " ".join(gold_parts)
        gold_terms = _significant_terms(gold_text)
        gold_bigrams = _significant_bigrams(gold_text)
        retained_terms = _significant_terms(retained_text)
        removed_terms = _significant_terms(removed_text)
        retained_bigrams = _significant_bigrams(retained_text)
        removed_bigrams = _significant_bigrams(removed_text)
        audit = treatment.get("standing_instruction_audit") or {}

        audits.append(
            {
                "query_id": query_id,
                "category": str(
                    (source.get("meta") or {}).get("question_category") or "unknown"
                ),
                "score_a": float(pair["score_a"]),
                "score_b": float(pair["score_b"]),
                "score_delta_b_minus_a": float(pair["score_b"])
                - float(pair["score_a"]),
                "source_blocks": len(source_blocks),
                "retained_source_blocks": len(retained_blocks),
                "displaced_blocks": len(removed_blocks),
                "panel_tokens": token_counter(treatment_blocks[0]),
                "displaced_tokens": token_counter(removed_text),
                "budget_slack_tokens": int(audit.get("reference_tokens", 0))
                - int(audit.get("treatment_tokens", 0)),
                "removed_only_gold_terms": sorted(
                    (gold_terms & removed_terms) - retained_terms
                ),
                "removed_only_gold_bigrams": sorted(
                    (gold_bigrams & removed_bigrams) - retained_bigrams
                ),
            }
        )

    by_category: dict[str, list[dict]] = {}
    for audit in audits:
        by_category.setdefault(audit["category"], []).append(audit)

    lexical_displacement = [row for row in audits if row["removed_only_gold_bigrams"]]
    no_lexical_displacement = [
        row for row in audits if not row["removed_only_gold_bigrams"]
    ]
    selective_scores = [
        row["score_b"]
        if row["category"] == INSTRUCTION_CATEGORY
        else row["score_a"]
        for row in audits
    ]
    control_scores = [row["score_a"] for row in audits]

    return {
        "protocol": "standing-instruction-displacement-diagnostic-v1",
        "interpretation": {
            "causal_claim": False,
            "promotion_evidence": False,
            "gold_overlap_is_lexical_proxy_only": True,
            "selective_composition_is_posthoc_category_oracle": True,
        },
        "overall": _summary(audits),
        "categories": {
            category: _summary(category_rows)
            for category, category_rows in sorted(by_category.items())
        },
        "gold_bigram_displacement_cohorts": {
            "removed_only_gold_bigram": _summary(lexical_displacement),
            "no_removed_only_gold_bigram": _summary(no_lexical_displacement),
        },
        "category_oracle_selective_composition": {
            "policy": "treatment for instruction_following; control otherwise",
            "n": len(audits),
            "mean_control": _mean(control_scores),
            "mean_selective": _mean(selective_scores),
            "mean_delta_selective_minus_control": _mean(selective_scores)
            - _mean(control_scores),
        },
        "rows": audits,
    }


def _load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, list) else payload.get("results") or []


def main() -> int:
    from memory_bench.utils import count_tokens

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = analyze(
        _load_rows(args.source),
        _load_rows(args.treatment),
        json.loads(args.result.read_text(encoding="utf-8")),
        count_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
