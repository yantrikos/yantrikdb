"""Paired frozen-context evaluation for two memory retrieval arms.

Each query is answered and judged for both arms in the same worker. Arm order
alternates deterministically by query id and answer draw, reducing time/order
drift while the answerer, judge, query, and rubric remain fixed. Repeated answer
draws retain their full score distributions and select the median-scored result
for the pair-level comparison. Checkpoints contain complete pairs and can be
resumed after an interrupted cloud run.

This fingerprint is sufficient only because contexts are frozen artifacts. A
live paired harness must also bind the provider code/version that creates its
contexts; copying this artifact-only contract into a live run would permit
provider drift under an unchanged manifest.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import statistics
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else payload["results"]


def load_contexts(path: Path) -> dict[str, str]:
    return {
        row["query_id"]: row.get("context") or ""
        for row in load_rows(path)
        if row.get("query_id")
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _ordered_query_ids_sha256(query_ids: list[str]) -> str:
    payload = json.dumps(
        query_ids,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def validate_manifest(
    manifest_path: Path,
    context_paths: tuple[Path, Path],
    model: str,
    judge_repeats: int,
    token_counter: Callable[[str], int],
    answer_repeats: int = 1,
) -> dict:
    """Validate the exact frozen payload and projected external call budget."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_arms = {arm["file"]: arm for arm in manifest.get("arms", [])}
    errors = []
    actual_arms = []
    ordered_ids_by_arm = []

    for path in context_paths:
        rows = load_rows(path)
        query_ids = [row.get("query_id") for row in rows]
        contexts = [row.get("context") or "" for row in rows]
        if any(not query_id for query_id in query_ids):
            errors.append(f"{path.name}: every row must have a query_id")
        if len(set(query_ids)) != len(query_ids):
            errors.append(f"{path.name}: duplicate query_ids")

        actual = {
            "file": path.name,
            "sha256": _sha256_file(path),
            "rows": len(rows),
            "context_tokens": sum(token_counter(context) for context in contexts),
            "query_ids_sha256": _ordered_query_ids_sha256(query_ids),
        }
        actual_arms.append(actual)
        ordered_ids_by_arm.append(query_ids)
        expected = expected_arms.get(path.name)
        if expected is None:
            errors.append(f"{path.name}: absent from manifest arms")
            continue
        for field in ("sha256", "rows", "context_tokens", "query_ids_sha256"):
            if actual[field] != expected.get(field):
                errors.append(
                    f"{path.name}: {field} expected={expected.get(field)!r} "
                    f"actual={actual[field]!r}"
                )

    if len(expected_arms) != 2:
        errors.append(f"manifest: expected exactly 2 arms, found {len(expected_arms)}")
    if len(ordered_ids_by_arm) == 2 and ordered_ids_by_arm[0] != ordered_ids_by_arm[1]:
        errors.append(
            "context arms do not contain the same query_ids in the same order"
        )
    total_context_tokens = sum(arm["context_tokens"] for arm in actual_arms)
    row_count = len(ordered_ids_by_arm[0]) if ordered_ids_by_arm else 0
    answer_calls = row_count * 2 * answer_repeats
    judge_calls = answer_calls * judge_repeats
    checks = {
        "model": model,
        "answer_repeats": answer_repeats,
        "judge_repeats": judge_repeats,
        "total_context_tokens": total_context_tokens,
        "answer_calls": answer_calls,
        "judge_calls": judge_calls,
    }
    for field, actual in checks.items():
        expected = (
            manifest.get(field, 1) if field == "answer_repeats" else manifest.get(field)
        )
        if expected != actual:
            errors.append(f"manifest: {field} expected={expected!r} actual={actual!r}")
    if manifest.get("query_ids_encoding") != "utf8-json-compact-ordered-v1":
        errors.append("manifest: unsupported or missing query_ids_encoding")
    if not manifest.get("synthetic_benchmark_data_only"):
        errors.append("manifest: synthetic_benchmark_data_only must be true")
    if manifest.get("real_companion_memories_included") is not False:
        errors.append("manifest: real_companion_memories_included must be false")
    if errors:
        raise ValueError("manifest preflight failed:\n- " + "\n- ".join(errors))

    return {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "arms": actual_arms,
        "query_ids": ordered_ids_by_arm[0],
        **checks,
    }


def _run_fingerprint(config: dict) -> str:
    encoded = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _resolve_bootstrap_seed(run_seed: int, bootstrap_seed: int | None) -> int:
    return run_seed if bootstrap_seed is None else bootstrap_seed


def _resolve_model_seed(model_seed: int | None) -> int:
    """Preserve the provider's historical environment/default seed behavior."""
    if model_seed is not None:
        return model_seed
    return int(os.environ.get("OMB_OLLAMA_SEED", "0"))


def _load_resume_checkpoint(path: Path, run_fingerprint: str) -> dict[str, dict]:
    prior = json.loads(path.read_text(encoding="utf-8"))
    if prior.get("run_fingerprint") != run_fingerprint:
        raise ValueError(
            "resume checkpoint does not match the current manifest, contexts, "
            "model, labels, split, run/model/bootstrap seeds, workers, and judge "
            "settings"
        )
    pairs = prior.get("pairs", [])
    query_ids = [pair.get("query_id") for pair in pairs]
    if any(not query_id for query_id in query_ids) or len(set(query_ids)) != len(
        query_ids
    ):
        raise ValueError("resume checkpoint contains missing or duplicate query_ids")
    return {pair["query_id"]: pair for pair in pairs}


def _median_judgement(dataset, result, judge, repeats: int) -> float:
    votes = [dataset.score_result(result, judge) for _ in range(repeats)]
    result.meta["judge_votes"] = votes
    return sorted(votes)[len(votes) // 2]


def _stable_arm_order(query_id: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{query_id}".encode()).digest()
    return ("a", "b") if digest[0] % 2 == 0 else ("b", "a")


def _answer_arm_orders(query_id: str, seed: int, repeats: int) -> list[tuple[str, str]]:
    """Interleave repeated answers so neither arm always runs first."""
    first = _stable_arm_order(query_id, seed)
    return [first if repeat % 2 == 0 else first[::-1] for repeat in range(repeats)]


def _median_scored_result(results: list):
    if not results:
        raise ValueError("cannot select a median from no answer results")
    ranked = sorted(
        enumerate(results),
        key=lambda entry: (float(entry[1].score or 0.0), entry[0]),
    )
    return ranked[len(ranked) // 2][1]


def _answer_repeat_comparison(samples: dict[str, list]) -> dict:
    scores_a = [float(result.score or 0.0) for result in samples["a"]]
    scores_b = [float(result.score or 0.0) for result in samples["b"]]
    if len(scores_a) != len(scores_b):
        raise ValueError("answer repeat arms have different sample counts")
    deltas = [score_b - score_a for score_a, score_b in zip(scores_a, scores_b)]
    return {
        "scores_a": scores_a,
        "scores_b": scores_b,
        "mean_a": statistics.fmean(scores_a),
        "mean_b": statistics.fmean(scores_b),
        "median_a": statistics.median(scores_a),
        "median_b": statistics.median(scores_b),
        "range_a": [min(scores_a), max(scores_a)],
        "range_b": [min(scores_b), max(scores_b)],
        "deltas_b_minus_a": deltas,
        "mean_delta_b_minus_a": statistics.fmean(deltas),
        "wins_b": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "wins_a": sum(delta < 0 for delta in deltas),
    }


def _paired_bootstrap_interval(
    deltas: list[float], seed: int, samples: int = 20_000
) -> tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(
        statistics.fmean(deltas[rng.randrange(n)] for _ in range(n))
        for _ in range(samples)
    )
    return means[int(0.025 * samples)], means[int(0.975 * samples)]


def _write_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, default=str), encoding="utf-8")
    temporary.replace(path)


def _publish_complete_output(
    output_path: Path,
    checkpoint_path: Path,
    payload: dict,
    cohort_complete: bool,
) -> bool:
    """Publish only complete cohorts; incomplete checkpoints stay resumable."""
    if not cohort_complete:
        return False
    _write_checkpoint(output_path, payload)
    checkpoint_path.unlink(missing_ok=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts-a", type=Path, required=True)
    parser.add_argument("--contexts-b", type=Path, required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--model", default="deepseek-v4-flash:0731-cloud")
    parser.add_argument("--split", default="100k")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--answer-repeats", type=int, default=1)
    parser.add_argument("--judge-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.judge_repeats < 1 or args.judge_repeats % 2 == 0:
        parser.error("--judge-repeats must be a positive odd number")
    if args.answer_repeats < 1 or args.answer_repeats % 2 == 0:
        parser.error("--answer-repeats must be a positive odd number")
    bootstrap_seed = _resolve_bootstrap_seed(args.seed, args.bootstrap_seed)
    try:
        model_seed = _resolve_model_seed(args.model_seed)
    except ValueError as error:
        parser.error(f"invalid OMB_OLLAMA_SEED: {error}")
    if args.limit is not None:
        parser.error("--limit is incompatible with a frozen manifest run")
    if not args.preflight_only and args.out is None:
        parser.error("--out is required unless --preflight-only is set")

    # memory_bench.dataset imports the llm package, whose Ollama seed is captured
    # at module import time. Pin the environment before any memory_bench import.
    os.environ["OMB_OLLAMA_SEED"] = str(model_seed)

    # These imports are deliberately after argument validation. Preflight never
    # constructs an LLM client or makes an external call.
    from memory_bench.dataset import get_dataset
    from memory_bench.utils import count_tokens

    try:
        preflight = validate_manifest(
            args.manifest,
            (args.contexts_a, args.contexts_b),
            args.model,
            args.judge_repeats,
            count_tokens,
            args.answer_repeats,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    dataset = get_dataset("beam")
    queries = {query.id: query for query in dataset.load_queries(args.split)}
    contexts = {
        "a": load_contexts(args.contexts_a),
        "b": load_contexts(args.contexts_b),
    }
    ids = [
        query_id
        for query_id in contexts["a"]
        if query_id in contexts["b"]
        and query_id in queries
        and contexts["a"][query_id].strip()
        and contexts["b"][query_id].strip()
    ]
    only_a = set(contexts["a"]) - set(contexts["b"])
    only_b = set(contexts["b"]) - set(contexts["a"])
    if only_a or only_b:
        print(
            f"warning: unmatched ids a_only={len(only_a)} b_only={len(only_b)}",
            file=sys.stderr,
        )

    if ids != preflight["query_ids"]:
        parser.error(
            "dataset/context eligibility changed after manifest validation; "
            "refusing to score a partial or reordered cohort"
        )
    print(
        json.dumps(
            {
                **{
                    key: value for key, value in preflight.items() if key != "query_ids"
                },
                "execution": {
                    "seed": args.seed,
                    "model_seed": model_seed,
                    "bootstrap_seed": bootstrap_seed,
                    "workers": args.workers,
                },
            },
            indent=2,
        )
    )
    if args.preflight_only:
        return 0

    from memory_bench.llm import ollama as ollama_module
    from memory_bench.models import QueryResult
    from memory_bench.modes.rag import RAGMode

    imported_model_seed = getattr(ollama_module, "_SEED", None)
    if imported_model_seed != model_seed:
        parser.error(
            "Ollama provider was imported before --model-seed was bound; refusing "
            f"requested={model_seed} imported={imported_model_seed}"
        )
    OllamaLLM = ollama_module.OllamaLLM

    run_config = {
        "manifest_sha256": preflight["manifest_sha256"],
        "contexts_a_sha256": preflight["arms"][0]["sha256"],
        "contexts_b_sha256": preflight["arms"][1]["sha256"],
        "query_ids_sha256": preflight["arms"][0]["query_ids_sha256"],
        "label_a": args.label_a,
        "label_b": args.label_b,
        "model": args.model,
        "split": args.split,
        "answer_repeats": args.answer_repeats,
        "judge_repeats": args.judge_repeats,
        "seed": args.seed,
        "model_seed": model_seed,
        "bootstrap_seed": bootstrap_seed,
        "workers": args.workers,
    }
    run_fingerprint = _run_fingerprint(run_config)

    checkpoint = args.out.with_suffix(args.out.suffix + ".partial")
    completed: dict[str, dict] = {}
    if args.resume and checkpoint.exists():
        try:
            completed = _load_resume_checkpoint(checkpoint, run_fingerprint)
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            parser.error(str(error))
        unknown = set(completed) - set(ids)
        if unknown:
            parser.error(
                f"resume checkpoint contains {len(unknown)} query_ids outside the cohort"
            )
    remaining = [query_id for query_id in ids if query_id not in completed]
    print(
        f"[{args.label_a} vs {args.label_b}] pairs={len(ids)} "
        f"resume={len(completed)} model={args.model} model_seed={model_seed}",
        file=sys.stderr,
        flush=True,
    )

    answer_llm = OllamaLLM(args.model)
    answer_mode = RAGMode(llm=answer_llm)
    judge_llm = OllamaLLM(args.model)

    def evaluate_answer(query_id: str, arm: str) -> QueryResult:
        query = queries[query_id]
        meta = dict(query.meta)
        meta["_prompt_fn"] = lambda question, context, meta=meta: (
            dataset.build_rag_prompt(
                question, context, dataset.task_type, args.split, meta=meta
            )
        )
        answer = answer_mode.answer_from_context(
            query.query,
            contexts[arm][query_id],
            dataset.task_type,
            meta=meta,
        )
        result = QueryResult(
            query_id=query_id,
            query=query.query,
            answer=answer.answer,
            reasoning=answer.reasoning,
            context=contexts[arm][query_id],
            context_tokens=count_tokens(contexts[arm][query_id]),
            retrieve_time_ms=0.0,
            gold_answers=query.gold_answers,
            correct=False,
            judge_reason="",
            meta=meta,
        )
        result.score = _median_judgement(dataset, result, judge_llm, args.judge_repeats)
        result.correct = (result.score or 0) >= 0.5
        return result

    def select_arm_result(results: list[QueryResult]) -> QueryResult:
        selected = _median_scored_result(results)
        selected.meta["answer_repeats"] = args.answer_repeats
        selected.meta["answer_repeat_scores"] = [
            float(result.score or 0.0) for result in results
        ]
        selected.meta["answer_candidates"] = [
            {
                "answer": result.answer,
                "reasoning": result.reasoning,
                "score": float(result.score or 0.0),
                "judge_votes": list(result.meta.get("judge_votes") or []),
            }
            for result in results
        ]
        return selected

    def evaluate_pair(query_id: str) -> dict | None:
        samples = {"a": [], "b": []}
        arm_orders = _answer_arm_orders(query_id, args.seed, args.answer_repeats)
        try:
            for arm_order in arm_orders:
                for arm in arm_order:
                    samples[arm].append(evaluate_answer(query_id, arm))
        except Exception as error:
            print(
                f"  {query_id}: pair failed {type(error).__name__}: {str(error)[:120]}",
                file=sys.stderr,
                flush=True,
            )
            return None
        results = {
            arm: select_arm_result(arm_results) for arm, arm_results in samples.items()
        }
        score_a = float(results["a"].score or 0)
        score_b = float(results["b"].score or 0)
        repeat_comparison = _answer_repeat_comparison(samples)
        return {
            "query_id": query_id,
            "arm_order": list(_stable_arm_order(query_id, args.seed)),
            "answer_arm_orders": [list(order) for order in arm_orders],
            "score_a": score_a,
            "score_b": score_b,
            "delta_b_minus_a": score_b - score_a,
            "answer_repeat_comparison": repeat_comparison,
            "result_a": results["a"].__dict__,
            "result_b": results["b"].__dict__,
        }

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, pair in enumerate(executor.map(evaluate_pair, remaining), 1):
            if pair is not None:
                completed[pair["query_id"]] = pair
            if index % 5 == 0 or index == len(remaining):
                ordered_pairs = [completed[qid] for qid in ids if qid in completed]
                _write_checkpoint(
                    checkpoint,
                    {
                        "run_config": run_config,
                        "run_fingerprint": run_fingerprint,
                        "label_a": args.label_a,
                        "label_b": args.label_b,
                        "model": args.model,
                        "pairs": ordered_pairs,
                    },
                )
                rate = index / max(time.perf_counter() - started, 1e-9) * 60
                print(
                    f"  completed={len(ordered_pairs)}/{len(ids)} rate={rate:.1f} pairs/min",
                    file=sys.stderr,
                    flush=True,
                )

    pairs = [completed[query_id] for query_id in ids if query_id in completed]
    deltas = [pair["delta_b_minus_a"] for pair in pairs]
    scores_a = [pair["score_a"] for pair in pairs]
    scores_b = [pair["score_b"] for pair in pairs]
    answer_repeat_scores_a = [
        score
        for pair in pairs
        for score in pair["answer_repeat_comparison"]["scores_a"]
    ]
    answer_repeat_scores_b = [
        score
        for pair in pairs
        for score in pair["answer_repeat_comparison"]["scores_b"]
    ]
    answer_repeat_deltas = [
        delta
        for pair in pairs
        for delta in pair["answer_repeat_comparison"]["deltas_b_minus_a"]
    ]
    lower, upper = _paired_bootstrap_interval(deltas, bootstrap_seed)
    summary = {
        "n": len(pairs),
        "mean_a": statistics.fmean(scores_a) if scores_a else 0.0,
        "mean_b": statistics.fmean(scores_b) if scores_b else 0.0,
        "mean_delta_b_minus_a": statistics.fmean(deltas) if deltas else 0.0,
        "paired_bootstrap_95_ci": [lower, upper],
        "paired_bootstrap_seed": bootstrap_seed,
        "wins_b": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "wins_a": sum(delta < 0 for delta in deltas),
        "answer_draws_per_arm": len(pairs) * args.answer_repeats,
        "mean_answer_repeat_score_a": (
            statistics.fmean(answer_repeat_scores_a) if answer_repeat_scores_a else 0.0
        ),
        "mean_answer_repeat_score_b": (
            statistics.fmean(answer_repeat_scores_b) if answer_repeat_scores_b else 0.0
        ),
        "answer_repeat_score_range_a": (
            [min(answer_repeat_scores_a), max(answer_repeat_scores_a)]
            if answer_repeat_scores_a
            else [0.0, 0.0]
        ),
        "answer_repeat_score_range_b": (
            [min(answer_repeat_scores_b), max(answer_repeat_scores_b)]
            if answer_repeat_scores_b
            else [0.0, 0.0]
        ),
        "mean_answer_repeat_delta_b_minus_a": (
            statistics.fmean(answer_repeat_deltas) if answer_repeat_deltas else 0.0
        ),
        "answer_repeat_wins_b": sum(delta > 0 for delta in answer_repeat_deltas),
        "answer_repeat_ties": sum(delta == 0 for delta in answer_repeat_deltas),
        "answer_repeat_wins_a": sum(delta < 0 for delta in answer_repeat_deltas),
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = {
        "run_config": run_config,
        "run_fingerprint": run_fingerprint,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "model": args.model,
        "answer_repeats": args.answer_repeats,
        "judge_repeats": args.judge_repeats,
        "seed": args.seed,
        "model_seed": model_seed,
        "bootstrap_seed": bootstrap_seed,
        "summary": summary,
        "pairs": pairs,
    }
    cohort_complete = len(pairs) == len(ids)
    _publish_complete_output(args.out, checkpoint, output, cohort_complete)
    if not cohort_complete:
        print(
            f"incomplete cohort: retained checkpoint at {checkpoint}",
            file=sys.stderr,
        )
    print(json.dumps(summary, indent=2))
    return 0 if cohort_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
