"""Measure where BEAM event-ordering rubric items disappear.

The adaptive-rollup artifact preserves fine candidates and the selected subset,
but not its raw retrieved block text. This evaluator therefore uses a separately
frozen YantrikDB baseline context as an explicitly labelled retrieval comparator,
then measures the exact adaptive candidate -> selection -> answer chain.

Gold audit text supplies expanded semantic targets, while the scored BEAM run
supplies the authoritative terse rubric labels and final rubric score. The raw
BEAM cache supplies query-level source turn IDs; these are intentionally kept
separate from the approximate per-rubric semantic alignments because BEAM may
group several rubric items into one source turn (or one item across turns).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import fmean


_TURN_HEADER_RE = re.compile(
    r"(?m)^\[(?:[A-Z][a-z]+-\d+-\d+ \| Turn (?P<dated>\d+)"
    r"|Turn (?P<plain>\d+))\](?: \(cont\.\))?\s+(?:User|Assistant):"
)

try:
    from .calibrate_rollup_membership import (
        _embed_batch,
        cosine,
        load_jsonl,
        ollama_model_metadata,
    )
except ImportError:  # Direct script execution.
    from calibrate_rollup_membership import (
        _embed_batch,
        cosine,
        load_jsonl,
        ollama_model_metadata,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index_unique(rows: list[dict], key: str, label: str) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            raise ValueError(f"{label} row is missing {key!r}")
        if value in indexed:
            raise ValueError(f"duplicate {label} {key} {value!r}")
        indexed[value] = row
    return indexed


def flatten_source_chat_ids(values: list) -> list[int]:
    turns = []
    for value in values:
        nested = value if isinstance(value, list) else [value]
        for turn in nested:
            if isinstance(turn, bool) or not isinstance(turn, int):
                raise ValueError(f"invalid BEAM source turn ID: {turn!r}")
            turns.append(turn)
    return sorted(set(turns))


def load_beam_event_sources(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for conversation in raw:
        conversation_id = str(conversation.get("conversation_id") or "")
        probing = conversation.get("probing_questions") or {}
        if isinstance(probing, str):
            try:
                probing = json.loads(probing)
            except json.JSONDecodeError:
                probing = ast.literal_eval(probing)
        for index, question in enumerate(probing.get("event_ordering") or []):
            rows.append(
                {
                    "query_id": f"{conversation_id}_event_ordering_{index}",
                    "query": question.get("question"),
                    "rubric": question.get("rubric") or [],
                    "source_chat_ids": question.get("source_chat_ids") or [],
                    "source_turn_ids": flatten_source_chat_ids(
                        question.get("source_chat_ids") or []
                    ),
                    "conversation_references": question.get(
                        "conversation_references"
                    )
                    or [],
                }
            )
    return rows


def split_context_memories(context: str) -> list[dict]:
    parts = re.split(r"(?m)^## Memory \d+\s*$", context or "")
    memories = []
    for part in parts:
        text = part.strip()
        if not text:
            continue
        turns = sorted(
            {
                int(match.group("dated") or match.group("plain"))
                for match in _TURN_HEADER_RE.finditer(text)
            }
        )
        memories.append({"text": text, "turns": turns})
    return memories


def split_answer_items(answer: str, target_count: int | None = None) -> list[str]:
    text = (answer or "").strip()
    if not text:
        return []
    numbered = re.findall(
        r"(?:^|\s)(?:\d{1,2}[.)])\s*(.+?)(?=(?:\s+\d{1,2}[.)]\s)|$)",
        text,
        flags=re.DOTALL,
    )
    if len(numbered) >= 2:
        return [item.strip(" \t\r\n,;") for item in numbered if item.strip()]
    bullets = [
        re.sub(r"^\s*[-*]\s+", "", line).strip()
        for line in text.splitlines()
        if re.match(r"^\s*[-*]\s+", line)
    ]
    if len(bullets) >= 2:
        return bullets
    semicolon = [item.strip() for item in text.split(";") if item.strip()]
    if target_count and len(semicolon) == target_count:
        return semicolon
    comma = [item.strip() for item in text.split(",") if item.strip()]
    if target_count and len(comma) == target_count:
        return comma
    return [text]


def _one_to_one_pairs(
    similarities: list[list[float]], threshold: float | None = None
) -> list[tuple[int, int, float]]:
    target_count = len(similarities)
    element_count = len(similarities[0]) if similarities else 0
    states: dict[int, tuple[int, float, tuple[tuple[int, int, float], ...]]] = {
        0: (0, 0.0, ())
    }
    for element_index in range(element_count):
        updated = dict(states)
        for mask, (hits, total, pairs) in states.items():
            for target_index in range(target_count):
                bit = 1 << target_index
                if mask & bit:
                    continue
                similarity = similarities[target_index][element_index]
                candidate = (
                    hits + int(threshold is not None and similarity >= threshold),
                    total + similarity,
                    pairs + ((target_index, element_index, similarity),),
                )
                current = updated.get(mask | bit)
                candidate_key = (
                    (candidate[0], candidate[1])
                    if threshold is not None
                    else (candidate[1],)
                )
                current_key = None
                if current is not None:
                    current_key = (
                        (current[0], current[1])
                        if threshold is not None
                        else (current[1],)
                    )
                if current_key is None or candidate_key > current_key:
                    updated[mask | bit] = candidate
        states = updated
    _, _, pairs = max(
        states.values(),
        key=lambda value: (
            (len(value[2]), value[0], value[1])
            if threshold is not None
            else (len(value[2]), value[1])
        ),
    )
    return sorted(pairs)


def stage_metrics(
    targets: list[dict],
    elements: list[dict],
    vectors: dict[str, list[float]],
    threshold: float,
) -> dict:
    if not targets:
        raise ValueError("stage metrics require at least one target")
    similarities = [
        [cosine(vectors[target["text"]], vectors[element["text"]]) for element in elements]
        for target in targets
    ]
    semantic_pairs = _one_to_one_pairs(similarities)
    threshold_pairs = _one_to_one_pairs(similarities, threshold)
    semantic_by_target = {
        target_index: score for target_index, _, score in semantic_pairs
    }
    threshold_by_target = {
        target_index: (element_index, score)
        for target_index, element_index, score in threshold_pairs
    }
    target_rows = []

    def element_summary(index: int | None) -> dict | None:
        if index is None:
            return None
        element = elements[index]
        return {
            key: value
            for key, value in element.items()
            if key != "text"
        } | {"text": element["text"][:500]}

    for target_index, target in enumerate(targets):
        best_index = None
        best_similarity = 0.0
        if elements:
            best_index = max(
                range(len(elements)),
                key=lambda index: similarities[target_index][index],
            )
            best_similarity = similarities[target_index][best_index]
        matched_index, matched_similarity = threshold_by_target.get(
            target_index, (None, 0.0)
        )
        gold_turn = target.get("turn")
        turn_present = bool(
            gold_turn is not None
            and any(gold_turn in element.get("turns", []) for element in elements)
        )
        target_rows.append(
            {
                "rubric": target["rubric"],
                "gold_text": target["text"],
                "gold_turn": gold_turn,
                "semantic_similarity": semantic_by_target.get(target_index, 0.0),
                "matched_similarity": matched_similarity,
                "present": matched_similarity >= threshold,
                "turn_present": turn_present,
                "best_element_index": best_index,
                "best_element_similarity": best_similarity,
                "best_element": (
                    elements[best_index]["text"][:500]
                    if best_index is not None
                    else None
                ),
                "matched_element_index": matched_index,
                "best_element_record": element_summary(best_index),
                "matched_element_record": element_summary(matched_index),
            }
        )
    semantic = [row["semantic_similarity"] for row in target_rows]
    present = [row["present"] for row in target_rows]
    turns = [row["turn_present"] for row in target_rows]
    return {
        "element_count": len(elements),
        "semantic_coverage": fmean(semantic),
        "matched_recall": fmean(present),
        "source_turn_recall": fmean(turns),
        "targets": target_rows,
    }


def source_turn_metrics(source_turns: list[int], elements: list[dict]) -> dict:
    element_turns = {
        int(turn)
        for element in elements
        for turn in element.get("turns", [])
    }
    expected = sorted(set(source_turns))
    present = [turn for turn in expected if turn in element_turns]
    missing = [turn for turn in expected if turn not in element_turns]
    return {
        "expected": expected,
        "present": present,
        "missing": missing,
        "recall": len(present) / len(expected) if expected else None,
    }


def candidate_provenance_turns(
    item: dict, evidence_block_turns: dict[str, list[int]]
) -> list[int]:
    evidence_turns = {
        int(turn)
        for evidence_id in item.get("evidence_ids") or []
        for turn in evidence_block_turns.get(str(evidence_id), [])
    }
    if evidence_turns:
        return sorted(evidence_turns)
    turn = item.get("first_mention_turn")
    return [int(turn)] if turn is not None else []


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else None


def analyze(
    candidate_rows: list[dict],
    gold_rows: list[dict],
    synthesis_rows: list[dict],
    baseline_rows: list[dict],
    beam_source_rows: list[dict],
    vectors: dict[str, list[float]],
    threshold: float = 0.55,
) -> dict:
    gold_by_query = _index_unique(gold_rows, "query", "gold")
    synthesis_by_id = _index_unique(synthesis_rows, "query_id", "synthesis")
    baseline_by_id = _index_unique(baseline_rows, "query_id", "baseline")
    source_by_id = _index_unique(beam_source_rows, "query_id", "BEAM source")
    candidate_by_query = _index_unique(candidate_rows, "query", "candidate")
    per_query = []
    for query, candidate in candidate_by_query.items():
        gold = gold_by_query.get(query)
        if gold is None:
            raise ValueError(f"candidate query has no gold row: {query!r}")
        query_id = gold["query_id"]
        synthesis = synthesis_by_id.get(query_id)
        baseline = baseline_by_id.get(query_id)
        source = source_by_id.get(query_id)
        if synthesis is None or baseline is None or source is None:
            raise ValueError(f"missing run join for {query_id}")
        rubrics = synthesis.get("meta", {}).get("rubric") or []
        source_rubrics = source.get("rubric") or []
        source_turns = source.get("source_turn_ids") or []
        references = source.get("conversation_references") or []
        alignments = gold.get("alignments") or []
        if rubrics != source_rubrics:
            raise ValueError(f"scored/source rubrics differ for {query_id}")
        if str(source.get("query") or "").strip() != str(query).strip():
            raise ValueError(f"candidate/source queries differ for {query_id}")
        if len(rubrics) != len(alignments):
            raise ValueError(
                f"rubric/alignment length mismatch for {query_id}: "
                f"rubric={len(rubrics)}, alignments={len(alignments)}, "
                f"source_turns={len(source_turns)}, references={len(references)}"
            )
        targets = [
            {
                "rubric": str(rubric),
                "text": str(alignment["item"]),
                "turn": (
                    int(alignment["matches"][0]["turn"])
                    if alignment.get("matches")
                    else None
                ),
                "conversation_reference": (
                    str(references[index])
                    if len(references) == len(rubrics)
                    else None
                ),
            }
            for index, (rubric, alignment) in enumerate(
                zip(rubrics, alignments)
            )
        ]
        evidence_block_turns = candidate.get("evidence_block_turns") or {}
        source_identity_mode = (
            "evidence_block_turns"
            if evidence_block_turns
            else "first_mention_fallback"
        )
        retrieval = split_context_memories(baseline.get("context") or "")
        candidates = [
            {
                "id": str(item.get("id") or ""),
                "text": str(item.get("item") or ""),
                "turns": candidate_provenance_turns(
                    item, evidence_block_turns
                ),
                "date": item.get("first_mention_date"),
                "best_retrieval_rank": item.get("best_retrieval_rank"),
            }
            for item in candidate.get("candidate_items") or []
        ]
        candidate_by_id = {item["id"]: item for item in candidates}
        selected = [
            {
                "id": str(item.get("id") or ""),
                "text": str(item.get("item") or ""),
                "turns": sorted(
                    {
                        turn
                        for source_id in item.get("source_item_ids") or []
                        for turn in candidate_by_id.get(
                            str(source_id), {}
                        ).get("turns", [])
                    }
                    or (
                        {item["first_mention_turn"]}
                        if item.get("first_mention_turn") is not None
                        else set()
                    )
                ),
                "date": item.get("first_mention_date"),
                "best_retrieval_rank": item.get("best_retrieval_rank"),
                "source_item_ids": [
                    str(value) for value in item.get("source_item_ids") or []
                ],
            }
            for item in candidate.get("results") or []
        ]
        answer = [
            {"text": text, "turns": []}
            for text in split_answer_items(
                synthesis.get("answer") or "", len(targets)
            )
        ]
        stages = {
            "baseline_retrieval_comparator": stage_metrics(
                targets, retrieval, vectors, threshold
            ),
            "candidate_pool": stage_metrics(
                targets, candidates, vectors, threshold
            ),
            "selected_items": stage_metrics(
                targets, selected, vectors, threshold
            ),
            "final_answer": stage_metrics(targets, answer, vectors, threshold),
        }
        for stage_name, elements in (
            ("baseline_retrieval_comparator", retrieval),
            ("candidate_pool", candidates),
            ("selected_items", selected),
        ):
            stages[stage_name]["source_turn_identity"] = source_turn_metrics(
                source_turns, elements
            )
        stages["final_answer"]["source_turn_identity"] = None
        target_funnel = []
        selected_source_ids = {
            source_id
            for item in selected
            for source_id in item.get("source_item_ids", [])
        }
        for index, target in enumerate(targets):
            retrieval_target = stages["baseline_retrieval_comparator"]["targets"][index]
            candidate_target = stages["candidate_pool"]["targets"][index]
            selected_target = stages["selected_items"]["targets"][index]
            answer_target = stages["final_answer"]["targets"][index]
            candidate_match = candidate_target.get("matched_element_record")
            candidate_source_selected = bool(
                candidate_match
                and candidate_match.get("id") in selected_source_ids
            )
            selection_loss = candidate_target["present"] and not selected_target["present"]
            target_funnel.append(
                {
                    **target,
                    "retrieval_present": retrieval_target["present"],
                    "retrieval_turn_present": retrieval_target["turn_present"],
                    "candidate_present": candidate_target["present"],
                    "candidate_turn_present": candidate_target["turn_present"],
                    "candidate_match": candidate_match,
                    "candidate_source_selected": candidate_source_selected,
                    "selected_present": selected_target["present"],
                    "selected_turn_present": selected_target["turn_present"],
                    "answer_present": answer_target["present"],
                    "candidate_missing": not candidate_target["present"],
                    "selection_loss": selection_loss,
                    "candidate_discard_loss": selection_loss
                    and not candidate_source_selected,
                    "rollup_rewrite_loss": selection_loss
                    and candidate_source_selected,
                    "answer_loss": selected_target["present"]
                    and not answer_target["present"],
                    "answer_rescue": not selected_target["present"]
                    and answer_target["present"],
                }
            )
        stage_source_turns = {
            stage_name: set(stage["source_turn_identity"]["present"])
            for stage_name, stage in stages.items()
            if stage["source_turn_identity"] is not None
        }
        source_funnel = [
            {
                "turn": turn,
                "retrieval_comparator_present": turn
                in stage_source_turns["baseline_retrieval_comparator"],
                "candidate_present": turn
                in stage_source_turns["candidate_pool"],
                "selected_present": turn
                in stage_source_turns["selected_items"],
            }
            for turn in source_turns
        ]
        for row in source_funnel:
            row["candidate_missing"] = not row["candidate_present"]
            row["comparator_to_candidate_loss"] = (
                row["retrieval_comparator_present"]
                and not row["candidate_present"]
            )
            row["selection_loss"] = (
                row["candidate_present"] and not row["selected_present"]
            )
        per_query.append(
            {
                "query_id": query_id,
                "query": query,
                "rubric_count": len(targets),
                "requested_item_count": candidate.get("requested_item_count"),
                "actual_score": float(synthesis.get("score") or 0.0),
                "answer": synthesis.get("answer"),
                "stages": stages,
                "targets": target_funnel,
                "source_turn_ids": source_turns,
                "source_identity_mode": source_identity_mode,
                "conversation_references": references,
                "source_funnel": source_funnel,
            }
        )

    stage_names = list(per_query[0]["stages"])
    aggregate_stages = {}
    for stage_name in stage_names:
        target_rows = [
            target
            for row in per_query
            for target in row["stages"][stage_name]["targets"]
        ]
        aggregate_stages[stage_name] = {
            "semantic_coverage": fmean(
                target["semantic_similarity"] for target in target_rows
            ),
            "matched_recall": fmean(target["present"] for target in target_rows),
            "alignment_turn_recall": fmean(
                target["turn_present"] for target in target_rows
            ),
        }
        exact_rows = [
            row["stages"][stage_name]["source_turn_identity"]
            for row in per_query
            if row["stages"][stage_name]["source_turn_identity"] is not None
        ]
        aggregate_stages[stage_name]["source_turn_identity_recall"] = (
            sum(len(row["present"]) for row in exact_rows)
            / sum(len(row["expected"]) for row in exact_rows)
            if exact_rows
            else None
        )
    all_targets = [target for row in per_query for target in row["targets"]]
    losses = {
        key: sum(target[key] for target in all_targets)
        for key in (
            "candidate_missing",
            "selection_loss",
            "candidate_discard_loss",
            "rollup_rewrite_loss",
            "answer_loss",
            "answer_rescue",
        )
    }
    source_rows = [row for query in per_query for row in query["source_funnel"]]
    losses.update(
        {
            "source_identity_candidate_missing": sum(
                row["candidate_missing"] for row in source_rows
            ),
            "source_identity_comparator_to_candidate_loss": sum(
                row["comparator_to_candidate_loss"] for row in source_rows
            ),
            "source_identity_selection_loss": sum(
                row["selection_loss"] for row in source_rows
            ),
        }
    )
    actual_scores = [row["actual_score"] for row in per_query]
    answer_recalls = [
        row["stages"]["final_answer"]["matched_recall"] for row in per_query
    ]
    return {
        "queries": len(per_query),
        "rubric_items": len(all_targets),
        "source_turn_identities": len(source_rows),
        "threshold": threshold,
        "retrieval_stage_is_comparator": True,
        "candidate_source_identity_modes": sorted(
            {row["source_identity_mode"] for row in per_query}
        ),
        "stages": aggregate_stages,
        "loss_counts": losses,
        "actual_score_mean": fmean(actual_scores),
        "answer_proxy_mean": fmean(answer_recalls),
        "answer_proxy_actual_score_pearson": _pearson(
            answer_recalls, actual_scores
        ),
        "per_query": sorted(per_query, key=lambda row: row["query_id"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--synthesis-run", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--beam-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()

    candidate_rows = load_jsonl(args.candidates)
    gold_payload = json.loads(args.gold.read_text(encoding="utf-8"))
    synthesis_payload = json.loads(
        args.synthesis_run.read_text(encoding="utf-8")
    )
    baseline_payload = json.loads(args.baseline_run.read_text(encoding="utf-8"))
    beam_source_rows = load_beam_event_sources(args.beam_source)
    gold_rows = gold_payload.get("results") or []
    synthesis_rows = synthesis_payload.get("results") or []
    baseline_rows = baseline_payload.get("results") or []

    gold_by_query = _index_unique(gold_rows, "query", "gold")
    synthesis_by_id = _index_unique(synthesis_rows, "query_id", "synthesis")
    baseline_by_id = _index_unique(baseline_rows, "query_id", "baseline")
    texts = []
    for candidate in candidate_rows:
        gold = gold_by_query.get(candidate.get("query"))
        if gold is None:
            raise ValueError(f"candidate query has no gold row: {candidate.get('query')!r}")
        query_id = gold["query_id"]
        synthesis = synthesis_by_id.get(query_id)
        baseline = baseline_by_id.get(query_id)
        if synthesis is None or baseline is None:
            raise ValueError(f"missing run join for {query_id}")
        texts.extend(str(item["item"]) for item in gold.get("alignments") or [])
        texts.extend(
            memory["text"]
            for memory in split_context_memories(baseline.get("context") or "")
        )
        texts.extend(
            str(item.get("item") or "")
            for item in candidate.get("candidate_items") or []
        )
        texts.extend(
            str(item.get("item") or "")
            for item in candidate.get("results") or []
        )
        texts.extend(
            split_answer_items(
                synthesis.get("answer") or "",
                len(gold.get("alignments") or []),
            )
        )
    vectors = _embed_batch(args.host, args.model, texts)
    report = analyze(
        candidate_rows,
        gold_rows,
        synthesis_rows,
        baseline_rows,
        beam_source_rows,
        vectors,
        threshold=args.threshold,
    )
    report.update(
        {
            "candidate_artifact": str(args.candidates),
            "candidate_sha256": _sha256(args.candidates),
            "gold_artifact": str(args.gold),
            "gold_sha256": _sha256(args.gold),
            "synthesis_run": str(args.synthesis_run),
            "synthesis_run_sha256": _sha256(args.synthesis_run),
            "baseline_run": str(args.baseline_run),
            "baseline_run_sha256": _sha256(args.baseline_run),
            "beam_source": str(args.beam_source),
            "beam_source_sha256": _sha256(args.beam_source),
            "embedding_model": args.model,
            "embedding_model_metadata": ollama_model_metadata(
                args.host, args.model
            ),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "per_query"},
            indent=2,
        )
    )
    print(f"wrote={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
