"""Offline calibration for AMB rollup membership rescue.

The adaptive-rollup debug artifact freezes both the fine-grained candidate pool
and the exact subset returned to the answerer. The gold audit independently
maps each requested answer item to user-authored source turns. This script uses
those two artifacts to test conservative candidate swaps without another LLM or
judge call.

Configuration is selected out of fold. Both queries from one source dialogue
share a group and are always held out together, preventing near-duplicate query
leakage. Gold text is used only to score a held-out selection or choose a policy
on the remaining groups; the policy itself sees query/candidate embeddings,
retrieval rank, item detail, redundancy, and temporal coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable


_DIALOGUE_RE = re.compile(r"^(\d+)_event_ordering_")
SCORER_PROTOCOL_VERSION = 2


@dataclass(frozen=True)
class RescueConfig:
    rank_weight: float = 0.0
    detail_weight: float = 0.0
    centrality_weight: float = 0.0
    anchor_weight: float = 0.0
    judge_weight: float = 0.0
    redundancy_weight: float = 0.0
    temporal_weight: float = 0.0
    session_weight: float = 0.0
    margin: float = 0.0
    max_replacements: int = 0


BASELINE_CONFIG = RescueConfig()


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm or 1.0)


def dialogue_group(query_id: str) -> str:
    match = _DIALOGUE_RE.match(query_id)
    return match.group(1) if match else query_id


def _group_sort_key(group: str) -> tuple[bool, int | str]:
    return (not group.isdigit(), int(group) if group.isdigit() else group)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def candidate_payload_sha256(row: dict) -> str:
    payload = {
        "query": row.get("query"),
        "requested_item_count": row.get("requested_item_count"),
        "candidate_items": [
            {"id": candidate.get("id"), "item": candidate.get("item")}
            for candidate in row.get("candidate_items") or []
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def validate_score_artifact(
    score_rows: list[dict],
    rows_by_query: dict[str, dict],
    candidate_sha256: str,
) -> dict[str, dict[str, dict]]:
    score_rows_by_query = _index_unique(
        score_rows, "query", "candidate score"
    )
    if set(score_rows_by_query) != set(rows_by_query):
        missing = sorted(set(rows_by_query) - set(score_rows_by_query))
        extra = sorted(set(score_rows_by_query) - set(rows_by_query))
        raise ValueError(
            f"candidate score query coverage differs: missing={missing}, extra={extra}"
        )
    signatures = set()
    scores_by_query = {}
    for query, scored in score_rows_by_query.items():
        row = rows_by_query[query]
        expected = {
            "candidate_artifact_sha256": candidate_sha256,
            "candidate_payload_sha256": candidate_payload_sha256(row),
            "scorer_protocol_version": SCORER_PROTOCOL_VERSION,
        }
        mismatched = {
            key: (scored.get(key), value)
            for key, value in expected.items()
            if scored.get(key) != value
        }
        prompt_sha256 = str(scored.get("prompt_sha256") or "")
        model_metadata = scored.get("model_metadata") or {}
        if not re.fullmatch(r"[0-9a-f]{64}", prompt_sha256):
            mismatched["prompt_sha256"] = (prompt_sha256, "64 lowercase hex")
        if not model_metadata.get("digest"):
            mismatched["model_metadata.digest"] = (
                model_metadata.get("digest"),
                "non-empty digest",
            )
        if mismatched:
            raise ValueError(
                f"candidate score provenance does not match {query!r}: {mismatched}"
            )
        signature = json.dumps(
            {
                "model": scored.get("model"),
                "model_metadata": model_metadata,
                "num_ctx": scored.get("num_ctx"),
                "temperature": scored.get("temperature"),
                "seed": scored.get("seed"),
                "think": scored.get("think"),
                "scorer_protocol_version": scored.get(
                    "scorer_protocol_version"
                ),
            },
            sort_keys=True,
        )
        signatures.add(signature)
        scores = scored.get("scores") or []
        score_ids = [str(score.get("id") or "") for score in scores]
        if len(score_ids) != len(set(score_ids)):
            raise ValueError(f"duplicate candidate score IDs for {query!r}")
        scores_by_query[query] = {
            candidate_id: score
            for candidate_id, score in zip(score_ids, scores)
        }
    if len(signatures) != 1:
        raise ValueError("candidate score artifact mixes scorer configurations")
    return scores_by_query


def _embed_batch(
    host: str, model: str, texts: list[str], batch_size: int = 128
) -> dict[str, list[float]]:
    vectors: dict[str, list[float]] = {}
    unique = list(dict.fromkeys(texts))
    for offset in range(0, len(unique), batch_size):
        batch = unique[offset : offset + batch_size]
        body = json.dumps({"model": model, "input": batch}).encode()
        request = urllib.request.Request(
            f"{host.rstrip('/')}/api/embed",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            embedded = json.load(response)["embeddings"]
        if len(embedded) != len(batch):
            raise ValueError("embedding endpoint returned an incomplete batch")
        vectors.update(zip(batch, embedded))
    return vectors


def ollama_model_metadata(host: str, model: str) -> dict:
    request = urllib.request.Request(f"{host.rstrip('/')}/api/tags")
    with urllib.request.urlopen(request, timeout=30) as response:
        models = json.load(response).get("models") or []
    requested_base = model.removesuffix(":latest")
    match = next(
        (
            item
            for item in models
            if str(item.get("name") or "").removesuffix(":latest")
            == requested_base
            or str(item.get("model") or "").removesuffix(":latest")
            == requested_base
        ),
        None,
    )
    if match is None:
        return {"requested": model, "resolved": False}
    return {
        "requested": model,
        "resolved": True,
        "name": match.get("name"),
        "model": match.get("model"),
        "digest": match.get("digest"),
        "modified_at": match.get("modified_at"),
        "size": match.get("size"),
        "details": match.get("details"),
    }


def _gold_items(gold: dict) -> list[dict]:
    turns = gold.get("matched_turns") or []
    alignments = gold.get("alignments") or []
    if len(turns) != len(alignments):
        raise ValueError(
            f"gold row {gold.get('query_id')!r} has {len(alignments)} alignments "
            f"but {len(turns)} matched turns"
        )
    return [
        {
            "item": alignment["item"],
            "turn": turns[index],
        }
        for index, alignment in enumerate(alignments)
    ]


def _baseline_ids(row: dict) -> list[str]:
    selected = []
    for result in row.get("results") or []:
        source_ids = result.get("source_item_ids") or []
        if len(source_ids) == 1:
            selected.append(str(source_ids[0]))
            continue
        item_text = result.get("item")
        match = next(
            (
                candidate["id"]
                for candidate in row.get("candidate_items") or []
                if candidate.get("item") == item_text
            ),
            None,
        )
        if match is not None:
            selected.append(str(match))
    return list(dict.fromkeys(selected))


def _rank_score(rank: int | None) -> float:
    if rank is None or rank < 0:
        return 0.0
    return 1.0 / math.log2(rank + 2.0)


def _detail_score(text: str) -> float:
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return min(len(words), 30) / 30.0


def build_example(
    row: dict,
    gold: dict,
    vectors: dict[str, list[float]],
    judged_scores: dict[str, dict] | None = None,
) -> dict:
    candidates = row.get("candidate_items") or []
    gold_items = _gold_items(gold)
    query = row["query"]
    candidate_ids = [str(candidate["id"]) for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"candidate IDs are not unique for {gold['query_id']}")
    if judged_scores is not None and set(judged_scores) != set(candidate_ids):
        missing = sorted(set(candidate_ids) - set(judged_scores))
        extra = sorted(set(judged_scores) - set(candidate_ids))
        raise ValueError(
            f"candidate scores do not match {gold['query_id']}: "
            f"missing={missing}, extra={extra}"
        )
    candidate_text = {
        str(candidate["id"]): str(candidate.get("item") or "")
        for candidate in candidates
    }
    features = {}
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        text = candidate_text[candidate_id]
        features[candidate_id] = {
            "query": cosine(vectors[query], vectors[text]),
            "rank": _rank_score(candidate.get("best_retrieval_rank")),
            "detail": _detail_score(text),
            "judge": 0.0,
            "turn": candidate.get("first_mention_turn"),
            "session": candidate.get("first_mention_date"),
        }
        if judged_scores and candidate_id in judged_scores:
            judged = judged_scores[candidate_id]
            features[candidate_id]["judge"] = (
                0.8 * float(judged["relevance"])
                + 0.2 * float(judged["atomicity"])
            ) / 100.0

    pair_similarity = {}
    for left in candidate_ids:
        for right in candidate_ids:
            pair_similarity[left, right] = cosine(
                vectors[candidate_text[left]], vectors[candidate_text[right]]
            )
    seed_ids = sorted(
        candidate_ids,
        key=lambda candidate_id: (
            -(
                features[candidate_id]["query"]
                + 0.5 * features[candidate_id]["rank"]
            ),
            candidate_id,
        ),
    )[:5]
    for candidate_id in candidate_ids:
        neighbors = sorted(
            (
                pair_similarity[candidate_id, other]
                for other in candidate_ids
                if other != candidate_id
            ),
            reverse=True,
        )[:5]
        features[candidate_id]["centrality"] = (
            fmean(neighbors) if neighbors else 0.0
        )
        seed_support = [
            pair_similarity[candidate_id, seed]
            for seed in seed_ids
            if seed != candidate_id
        ]
        features[candidate_id]["anchor"] = (
            fmean(seed_support) if seed_support else 0.0
        )
    gold_similarity = {
        (gold_index, candidate_id): cosine(
            vectors[gold_item["item"]], vectors[candidate_text[candidate_id]]
        )
        for gold_index, gold_item in enumerate(gold_items)
        for candidate_id in candidate_ids
    }
    return {
        "query_id": gold["query_id"],
        "group": dialogue_group(gold["query_id"]),
        "query": query,
        "target_count": int(row["requested_item_count"]),
        "candidate_ids": candidate_ids,
        "baseline_ids": _baseline_ids(row),
        "candidate_text": candidate_text,
        "features": features,
        "pair_similarity": pair_similarity,
        "gold_items": gold_items,
        "gold_similarity": gold_similarity,
    }


def _base_score(example: dict, candidate_id: str, config: RescueConfig) -> float:
    feature = example["features"][candidate_id]
    return (
        feature["query"]
        + config.rank_weight * feature["rank"]
        + config.detail_weight * feature["detail"]
        + config.centrality_weight * feature["centrality"]
        + config.anchor_weight * feature["anchor"]
        + config.judge_weight * feature["judge"]
    )


def _temporal_novelty(example: dict, candidate_id: str, others: list[str]) -> float:
    turn = example["features"][candidate_id]["turn"]
    other_turns = [
        example["features"][other]["turn"]
        for other in others
        if example["features"][other]["turn"] is not None
    ]
    all_turns = [
        feature["turn"]
        for feature in example["features"].values()
        if feature["turn"] is not None
    ]
    if turn is None or not other_turns or len(all_turns) < 2:
        return 0.0
    span = max(all_turns) - min(all_turns)
    return min(abs(turn - other) for other in other_turns) / max(span, 1)


def _session_novelty(example: dict, candidate_id: str, others: list[str]) -> float:
    session = example["features"][candidate_id].get("session")
    if session is None:
        return 0.0
    return float(
        all(
            example["features"][other].get("session") != session
            for other in others
        )
    )


def _conditional_score(
    example: dict,
    candidate_id: str,
    others: list[str],
    config: RescueConfig,
) -> float:
    redundancy = max(
        (
            example["pair_similarity"][candidate_id, other]
            for other in others
        ),
        default=0.0,
    )
    return (
        _base_score(example, candidate_id, config)
        - config.redundancy_weight * redundancy
        + config.temporal_weight * _temporal_novelty(
            example, candidate_id, others
        )
        + config.session_weight * _session_novelty(
            example, candidate_id, others
        )
    )


def rescue_selection(example: dict, config: RescueConfig) -> list[str]:
    target = example["target_count"]
    selected = [
        candidate_id
        for candidate_id in example["baseline_ids"]
        if candidate_id in example["features"]
    ][:target]
    if len(selected) < target:
        remaining = sorted(
            set(example["candidate_ids"]) - set(selected),
            key=lambda candidate_id: (
                -_base_score(example, candidate_id, config), candidate_id
            ),
        )
        selected.extend(remaining[: target - len(selected)])
    if config.max_replacements <= 0:
        return selected

    for _ in range(config.max_replacements):
        omitted = [
            candidate_id
            for candidate_id in example["candidate_ids"]
            if candidate_id not in selected
        ]
        best_swap = None
        for incoming in omitted:
            for outgoing in selected:
                rest = [item for item in selected if item != outgoing]
                delta = _conditional_score(
                    example, incoming, rest, config
                ) - _conditional_score(example, outgoing, rest, config)
                candidate = (delta, incoming, outgoing)
                if best_swap is None or candidate > best_swap:
                    best_swap = candidate
        if best_swap is None or best_swap[0] <= config.margin:
            break
        _, incoming, outgoing = best_swap
        selected[selected.index(outgoing)] = incoming
    return selected


def greedy_gold_selection(example: dict) -> list[str]:
    selected: list[str] = []
    for _ in range(example["target_count"]):
        best = None
        for candidate_id in example["candidate_ids"]:
            if candidate_id in selected:
                continue
            trial = selected + [candidate_id]
            semantic = selection_metrics(example, trial)["semantic_coverage"]
            candidate = (
                semantic,
                _base_score(example, candidate_id, BASELINE_CONFIG),
                candidate_id,
            )
            if best is None or candidate > best:
                best = candidate
        if best is None:
            break
        selected.append(best[2])
    return selected


def _one_to_one_pairs(
    example: dict,
    selected: list[str],
    threshold: float | None = None,
) -> list[tuple[int, str, float]]:
    """Return a maximum-cardinality one-to-one assignment.

    Semantic coverage maximizes total similarity. Threshold metrics first
    maximize the number of qualifying pairs, then total similarity.
    """
    gold_count = len(example["gold_items"])
    states: dict[
        int, tuple[int, float, tuple[tuple[int, str, float], ...]]
    ] = {
        0: (0, 0.0, ())
    }
    for candidate_id in selected:
        updated = dict(states)
        for mask, (hits, total, pairs) in states.items():
            for gold_index in range(gold_count):
                bit = 1 << gold_index
                if mask & bit:
                    continue
                similarity = example["gold_similarity"][gold_index, candidate_id]
                candidate = (
                    hits + int(threshold is not None and similarity >= threshold),
                    total + similarity,
                    pairs + ((gold_index, candidate_id, similarity),),
                )
                current = updated.get(mask | bit)
                candidate_key = (
                    (candidate[0], candidate[1])
                    if threshold is not None
                    else (candidate[1],)
                )
                current_key = (
                    (current[0], current[1])
                    if threshold is not None and current is not None
                    else ((current[1],) if current is not None else None)
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


def selection_metrics(
    example: dict, selected: list[str], threshold: float = 0.55
) -> dict[str, float]:
    gold_count = len(example["gold_items"])
    if not gold_count:
        return {
            "semantic_coverage": 0.0,
            "matched_recall": 0.0,
            "selection_precision": 0.0,
            "turn_recall": 0.0,
            "chronological_similarity": 0.0,
        }
    semantic_pairs = _one_to_one_pairs(example, selected)
    threshold_pairs = _one_to_one_pairs(example, selected, threshold)
    semantic_by_gold = {
        gold_index: score for gold_index, _, score in semantic_pairs
    }
    pair_by_gold = {
        gold_index: (candidate_id, score)
        for gold_index, candidate_id, score in threshold_pairs
    }
    pair_by_candidate = {
        candidate_id: (gold_index, score)
        for gold_index, candidate_id, score in threshold_pairs
    }
    semantic_by_gold = [
        semantic_by_gold.get(index, 0.0) for index in range(gold_count)
    ]
    matched_by_gold = [
        pair_by_gold.get(index, (None, 0.0))[1]
        for index in range(gold_count)
    ]
    matched_by_candidate = [
        pair_by_candidate.get(candidate_id, (None, 0.0))[1]
        for candidate_id in selected
    ]
    turn_hits = [
        (
            gold_index in pair_by_gold
            and example["features"][pair_by_gold[gold_index][0]]["turn"]
            == gold_item["turn"]
            and pair_by_gold[gold_index][1] >= threshold - 0.05
        )
        for gold_index, gold_item in enumerate(example["gold_items"])
    ]
    chronological_candidates = sorted(
        selected,
        key=lambda candidate_id: (
            example["features"][candidate_id]["turn"]
            if example["features"][candidate_id]["turn"] is not None
            else math.inf,
            candidate_id,
        ),
    )
    chronological_gold = sorted(
        range(gold_count),
        key=lambda index: (
            example["gold_items"][index]["turn"]
            if example["gold_items"][index]["turn"] is not None
            else math.inf,
            index,
        ),
    )
    chronological_similarity = [
        example["gold_similarity"][gold_index, candidate_id]
        for gold_index, candidate_id in zip(
            chronological_gold, chronological_candidates
        )
    ]
    return {
        "semantic_coverage": fmean(semantic_by_gold),
        "matched_recall": fmean(score >= threshold for score in matched_by_gold),
        "selection_precision": (
            fmean(score >= threshold for score in matched_by_candidate)
            if matched_by_candidate
            else 0.0
        ),
        "turn_recall": fmean(turn_hits),
        "chronological_similarity": (
            fmean(chronological_similarity) if chronological_similarity else 0.0
        ),
    }


def _objective(metrics: dict[str, float]) -> float:
    return metrics["semantic_coverage"] + 0.2 * metrics["matched_recall"]


def _mean_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    return {
        key: fmean(row[key] for row in rows)
        for key in rows[0]
    }


def config_grid(
    include_judge: bool = False, include_session: bool = False
) -> list[RescueConfig]:
    configs = [BASELINE_CONFIG]
    if include_judge:
        for rank_weight in (0.0, 0.25):
            for centrality_weight in (0.0, 0.25):
                for judge_weight in (0.5, 1.0, 2.0, 3.0):
                    for redundancy_weight in (0.0, 0.2):
                        for temporal_weight in (0.0, 0.1):
                            for margin in (0.05, 0.1, 0.15, 0.2):
                                for max_replacements in (1, 2):
                                    configs.append(
                                        RescueConfig(
                                            rank_weight=rank_weight,
                                            centrality_weight=centrality_weight,
                                            judge_weight=judge_weight,
                                            redundancy_weight=redundancy_weight,
                                            temporal_weight=temporal_weight,
                                            margin=margin,
                                            max_replacements=max_replacements,
                                        )
                                    )
        return configs
    for rank_weight in (0.0, 0.25, 0.5):
        for detail_weight in (0.0, 0.2):
            for centrality_weight in (0.0, 0.25, 0.5):
                for anchor_weight in (0.0, 0.25, 0.5):
                    if centrality_weight == anchor_weight == 0.0:
                        continue
                    for redundancy_weight in (0.0, 0.2):
                        for temporal_weight in (0.0, 0.1):
                            for margin in (0.05, 0.1, 0.15):
                                for max_replacements in (1, 2):
                                    configs.append(
                                        RescueConfig(
                                            rank_weight=rank_weight,
                                            detail_weight=detail_weight,
                                            centrality_weight=centrality_weight,
                                            anchor_weight=anchor_weight,
                                            redundancy_weight=redundancy_weight,
                                            temporal_weight=temporal_weight,
                                            margin=margin,
                                            max_replacements=max_replacements,
                                        )
                                    )
    if include_session:
        for session_weight in (0.1, 0.25, 0.5):
            for rank_weight in (0.0, 0.25):
                for centrality_weight in (0.0, 0.25):
                    for anchor_weight in (0.0, 0.25):
                        for margin in (0.05, 0.1, 0.15):
                            for max_replacements in (1, 2):
                                configs.append(
                                    RescueConfig(
                                        rank_weight=rank_weight,
                                        centrality_weight=centrality_weight,
                                        anchor_weight=anchor_weight,
                                        session_weight=session_weight,
                                        margin=margin,
                                        max_replacements=max_replacements,
                                    )
                                )
    return configs


def choose_config(
    examples: list[dict],
    threshold: float,
    minimum_train_delta: float,
    configurations: list[RescueConfig] | None = None,
    metrics_cache: dict[tuple[str, RescueConfig], dict[str, float]] | None = None,
) -> RescueConfig:
    def metrics_for(example: dict, config: RescueConfig) -> dict[str, float]:
        if metrics_cache is not None:
            return metrics_cache[example["query_id"], config]
        return selection_metrics(
            example, rescue_selection(example, config), threshold
        )

    baseline_objective = fmean(
        _objective(metrics_for(example, BASELINE_CONFIG))
        for example in examples
    )
    best_config = BASELINE_CONFIG
    best_key = (baseline_objective, 0, 0.0, 0.0)
    for config in (configurations or config_grid())[1:]:
        objective = fmean(
            _objective(metrics_for(example, config))
            for example in examples
        )
        key = (
            objective,
            -config.max_replacements,
            config.margin,
            -(
                config.rank_weight
                + config.detail_weight
                + config.centrality_weight
                + config.anchor_weight
                + config.judge_weight
                + config.redundancy_weight
                + config.temporal_weight
                + config.session_weight
            ),
        )
        if key > best_key:
            best_key = key
            best_config = config
    if best_key[0] < baseline_objective + minimum_train_delta:
        return BASELINE_CONFIG
    return best_config


def _cluster_bootstrap(
    per_query: list[dict], metric: str, seed: int, samples: int = 20_000
) -> tuple[float, float]:
    by_group: dict[str, list[float]] = {}
    for row in per_query:
        by_group.setdefault(row["group"], []).append(
            row["policy"][metric] - row["baseline"][metric]
        )
    group_deltas = [fmean(values) for values in by_group.values()]
    rng = random.Random(seed)
    means = sorted(
        fmean(group_deltas[rng.randrange(len(group_deltas))] for _ in group_deltas)
        for _ in range(samples)
    )
    return means[int(0.025 * samples)], means[int(0.975 * samples)]


def _accept_for_product_probe(
    delta: dict[str, float], intervals: dict[str, tuple[float, float]]
) -> bool:
    return (
        intervals["semantic_coverage"][0] > 0.0
        and delta["matched_recall"] >= 0.0
        and delta["turn_recall"] >= 0.0
        and delta["chronological_similarity"] >= 0.0
    )


def calibrate(
    examples: list[dict],
    threshold: float = 0.55,
    minimum_train_delta: float = 0.002,
    seed: int = 20260820,
    include_session: bool = False,
) -> dict:
    if len({example["group"] for example in examples}) < 2:
        raise ValueError("grouped calibration requires at least two dialogue groups")
    groups = sorted(
        {example["group"] for example in examples}, key=_group_sort_key
    )
    include_judge = any(
        feature.get("judge", 0.0) > 0.0
        for example in examples
        for feature in example["features"].values()
    )
    configurations = config_grid(
        include_judge=include_judge, include_session=include_session
    )
    selection_cache: dict[tuple[str, RescueConfig], list[str]] = {}
    metrics_cache: dict[tuple[str, RescueConfig], dict[str, float]] = {}
    for example in examples:
        for config in configurations:
            selected = rescue_selection(example, config)
            key = (example["query_id"], config)
            selection_cache[key] = selected
            metrics_cache[key] = selection_metrics(example, selected, threshold)
    per_query = []
    fold_configs = {}
    for group in groups:
        train = [example for example in examples if example["group"] != group]
        held_out = [example for example in examples if example["group"] == group]
        config = choose_config(
            train,
            threshold,
            minimum_train_delta,
            configurations,
            metrics_cache,
        )
        fold_configs[group] = asdict(config)
        for example in held_out:
            baseline_ids = example["baseline_ids"]
            policy_ids = selection_cache[example["query_id"], config]
            gold_informed_ids = greedy_gold_selection(example)
            per_query.append(
                {
                    "query_id": example["query_id"],
                    "group": group,
                    "target_count": example["target_count"],
                    "candidate_count": len(example["candidate_ids"]),
                    "config": asdict(config),
                    "baseline_ids": baseline_ids,
                    "policy_ids": policy_ids,
                    "gold_informed_ids": gold_informed_ids,
                    "replacements": len(set(policy_ids) - set(baseline_ids)),
                    "baseline": selection_metrics(example, baseline_ids, threshold),
                    "policy": selection_metrics(example, policy_ids, threshold),
                    "gold_informed": selection_metrics(
                        example, gold_informed_ids, threshold
                    ),
                }
            )

    baseline = _mean_metrics(row["baseline"] for row in per_query)
    policy = _mean_metrics(row["policy"] for row in per_query)
    gold_informed = _mean_metrics(row["gold_informed"] for row in per_query)
    delta = {key: policy[key] - baseline[key] for key in baseline}
    intervals = {
        key: _cluster_bootstrap(per_query, key, seed)
        for key in baseline
    }
    accepted = _accept_for_product_probe(delta, intervals)
    return {
        "queries": len(per_query),
        "groups": len(groups),
        "threshold": threshold,
        "minimum_train_delta": minimum_train_delta,
        "selection_protocol": "leave-one-dialogue-group-out",
        "pointwise_judge_features": include_judge,
        "session_features": include_session,
        "temporal_holdout_available": False,
        "baseline": baseline,
        "policy": policy,
        "greedy_gold_informed": gold_informed,
        "delta": delta,
        "cluster_bootstrap_95": intervals,
        "accepted_for_product_probe": accepted,
        "fold_configs": fold_configs,
        "per_query": sorted(per_query, key=lambda row: row["query_id"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--minimum-train-delta", type=float, default=0.002)
    parser.add_argument(
        "--include-session-feature",
        action="store_true",
        help="Include conversation-date diversity configurations.",
    )
    parser.add_argument(
        "--candidate-scores",
        type=Path,
        help="Optional JSONL from score_rollup_membership.py.",
    )
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()

    rows = load_jsonl(args.candidates)
    candidate_sha256 = hashlib.sha256(args.candidates.read_bytes()).hexdigest()
    gold_payload = json.loads(args.gold.read_text(encoding="utf-8"))
    gold_by_query = _index_unique(
        gold_payload.get("results") or [], "query", "gold"
    )
    rows_by_query = _index_unique(rows, "query", "candidate")
    missing_gold = sorted(set(rows_by_query) - set(gold_by_query))
    if missing_gold:
        raise ValueError(
            f"candidate artifact has {len(missing_gold)} queries without gold rows"
        )
    scores_by_query = {}
    score_rows = []
    if args.candidate_scores:
        score_rows = load_jsonl(args.candidate_scores)
        scores_by_query = validate_score_artifact(
            score_rows, rows_by_query, candidate_sha256
        )
    matched = [(row, gold_by_query[row["query"]]) for row in rows]
    texts = []
    for row, gold in matched:
        texts.append(row["query"])
        texts.extend(candidate["item"] for candidate in row.get("candidate_items") or [])
        texts.extend(item["item"] for item in _gold_items(gold))
    vectors = _embed_batch(args.host, args.model, texts)
    missing_scores = [
        row["query"]
        for row, _ in matched
        if args.candidate_scores and row["query"] not in scores_by_query
    ]
    if missing_scores:
        raise ValueError(
            f"candidate score artifact is incomplete for {len(missing_scores)} queries"
        )
    examples = [
        build_example(row, gold, vectors, scores_by_query.get(row["query"]))
        for row, gold in matched
    ]
    report = calibrate(
        examples,
        threshold=args.threshold,
        minimum_train_delta=args.minimum_train_delta,
        seed=args.seed,
        include_session=args.include_session_feature,
    )
    report.update(
        {
            "candidate_artifact": str(args.candidates),
            "candidate_sha256": candidate_sha256,
            "gold_artifact": str(args.gold),
            "gold_sha256": hashlib.sha256(args.gold.read_bytes()).hexdigest(),
            "embedding_model": args.model,
            "embedding_model_metadata": ollama_model_metadata(
                args.host, args.model
            ),
            "candidate_score_artifact": (
                str(args.candidate_scores) if args.candidate_scores else None
            ),
            "candidate_score_sha256": (
                hashlib.sha256(args.candidate_scores.read_bytes()).hexdigest()
                if args.candidate_scores
                else None
            ),
            "candidate_score_models": sorted(
                {
                    str(row.get("model"))
                    for row in score_rows
                    if row.get("model")
                }
            ),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_query"}, indent=2))
    print(f"wrote={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
