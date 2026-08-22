"""Measure contextual candidate-bank source reachability over the AMB cohort."""

import argparse
import ast
import json
import os
from pathlib import Path
from statistics import fmean, median

try:
    from benchmarks.amb import replay_contextual_synthesis as replay
except ModuleNotFoundError:  # Direct script execution.
    import replay_contextual_synthesis as replay


def load_event_queries(path: Path) -> list[dict]:
    rows = []
    for conversation in json.loads(path.read_text(encoding="utf-8")):
        conversation_id = str(conversation.get("conversation_id") or "")
        probing = conversation.get("probing_questions") or {}
        if isinstance(probing, str):
            try:
                probing = json.loads(probing)
            except json.JSONDecodeError:
                probing = ast.literal_eval(probing)
        for index, question in enumerate(probing.get("event_ordering") or []):
            source_turns = {
                int(turn)
                for value in question.get("source_chat_ids") or []
                for turn in (value if isinstance(value, list) else [value])
                if not isinstance(turn, bool) and isinstance(turn, int)
            }
            rows.append({
                "query_id": f"{conversation_id}_event_ordering_{index}",
                "user_id": conversation_id,
                "query": question.get("question") or "",
                "source_turns": source_turns,
            })
    return rows


def aggregate(rows: list[dict]) -> dict:
    source_total = sum(row["gold_source_turn_count"] for row in rows)
    base_total = sum(row["base_available_source_turn_count"] for row in rows)
    bank_total = sum(row["bank_available_source_turn_count"] for row in rows)
    bank_sizes = [row["candidate_bank_count"] for row in rows]
    return {
        "query_count": len(rows),
        "gold_source_turn_count": source_total,
        "base_available_source_turn_count": base_total,
        "base_source_turn_recall": base_total / source_total if source_total else None,
        "bank_available_source_turn_count": bank_total,
        "bank_source_turn_recall": bank_total / source_total if source_total else None,
        "source_turn_gain": bank_total - base_total,
        "all_source_queries_base": sum(row["base_all_available"] for row in rows),
        "all_source_queries_bank": sum(row["bank_all_available"] for row in rows),
        "candidate_bank_mean": fmean(bank_sizes) if bank_sizes else 0.0,
        "candidate_bank_median": median(bank_sizes) if bank_sizes else 0.0,
        "candidate_bank_max": max(bank_sizes, default=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--beam-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--rerank-pool", type=int, default=1000)
    parser.add_argument("--entity-seed-k", type=int, default=10)
    parser.add_argument("--entity-closure-slots", type=int, default=10)
    parser.add_argument("--continuation-slots", type=int, default=2)
    parser.add_argument("--context-bridge-slots", type=int, default=0)
    parser.add_argument(
        "--user-ids",
        default=",".join(str(value) for value in range(1, 10)),
        help="comma-separated frozen cohort IDs (default: AMB groups 1-9)",
    )
    parser.add_argument("--expect-model-digest")
    parser.add_argument("--compare-preflight", type=Path)
    args = parser.parse_args()

    if (
        args.top_k < 1
        or args.rerank_pool < args.top_k
        or args.entity_seed_k < 1
        or args.entity_closure_slots < 0
        or args.continuation_slots < 0
        or args.context_bridge_slots < 0
    ):
        parser.error("require rerank-pool >= top-k >= 1")

    os.environ["YDB_BENCH_SYNTH_BLOCKS"] = str(args.rerank_pool)
    provider_cls = replay._load_provider(args.repo.resolve())
    requested_user_ids = {
        value.strip() for value in args.user_ids.split(",") if value.strip()
    }
    query_rows = [
        row
        for row in load_event_queries(args.beam_source)
        if row["user_id"] in requested_user_ids
    ]
    user_ids = {row["user_id"] for row in query_rows}
    missing_banks = sorted(
        user_id
        for user_id in user_ids
        if not (args.bank / "yantrikdb" / f"{user_id}.db").is_file()
    )
    if missing_banks:
        raise FileNotFoundError(
            "missing frozen YantrikDB banks for user IDs: "
            + ", ".join(missing_banks)
        )
    provider = provider_cls()
    model_identity = replay._ollama_model_identity(
        args.embed_model, args.ollama_host
    )
    if args.expect_model_digest and (
        model_identity.get("digest") != args.expect_model_digest
    ):
        raise RuntimeError(
            "Ollama model digest mismatch: expected "
            f"{args.expect_model_digest}, got {model_identity.get('digest')}"
        )

    results = []
    try:
        provider.prepare(args.bank.resolve(), user_ids, False)
        for query_row in query_rows:
            query = query_row["query"]
            recalled = provider._recall(
                query, args.rerank_pool, query_row["user_id"]
            )
            user_hits = provider._select_evidence_hits(recalled)
            embeddings = replay._ollama_embeddings(
                [query, *(hit.get("text") or "" for hit in user_hits)],
                args.embed_model,
                args.ollama_host,
            )
            query_embedding, document_embeddings = embeddings[0], embeddings[1:]
            reranked = sorted(
                (
                    dict(
                        hit,
                        _contextual_score=replay._cosine(
                            query_embedding, embedding
                        ),
                    )
                    for hit, embedding in zip(
                        user_hits, document_embeddings, strict=True
                    )
                ),
                key=lambda hit: (
                    -hit["_contextual_score"], hit["_retrieval_rank"]
                ),
            )
            selected, selection_reasons = replay._candidate_bank(
                reranked,
                top_k=args.top_k,
                entity_seed_k=args.entity_seed_k,
                entity_closure_slots=args.entity_closure_slots,
                continuation_slots=args.continuation_slots,
                context_bridge_slots=args.context_bridge_slots,
            )
            rank_by_turn = replay._rank_by_turn(reranked)
            base_turns = {
                turn
                for hit in reranked[:args.top_k]
                if (turn := replay._turn(hit.get("text") or "")) is not None
            }
            bank_turns = {
                turn
                for hit in selected
                if (turn := replay._turn(hit.get("text") or "")) is not None
            }
            gold_turns = query_row["source_turns"]
            base_ceiling = replay._source_evidence_ceiling(
                gold_turns, rank_by_turn, base_turns
            )
            bank_ceiling = replay._source_evidence_ceiling(
                gold_turns, rank_by_turn, bank_turns
            )
            preflight = replay._preflight_record(
                query=query,
                model_identity=model_identity,
                embed_model=args.embed_model,
                rerank_pool=args.rerank_pool,
                top_k=args.top_k,
                user_hits=user_hits,
                reranked=reranked,
                selected=selected,
                selection_reasons=selection_reasons,
                gold_turns=gold_turns,
                entity_seed_k=args.entity_seed_k,
                entity_closure_slots=args.entity_closure_slots,
                continuation_slots=args.continuation_slots,
                context_bridge_slots=args.context_bridge_slots,
            )
            results.append({
                **query_row,
                "source_turns": sorted(gold_turns),
                "raw_recall_count": len(recalled),
                "user_turn_count": len(user_hits),
                "candidate_bank_count": len(selected),
                "gold_source_turn_count": len(gold_turns),
                "base_available_source_turn_count": base_ceiling[
                    "available_source_turn_count"
                ],
                "bank_available_source_turn_count": bank_ceiling[
                    "available_source_turn_count"
                ],
                "base_all_available": base_ceiling[
                    "all_gold_source_turns_available"
                ],
                "bank_all_available": bank_ceiling[
                    "all_gold_source_turns_available"
                ],
                "base_ceiling": base_ceiling,
                "bank_ceiling": bank_ceiling,
                "preflight": preflight,
            })
            print(
                f"{query_row['query_id']}: "
                f"{base_ceiling['available_source_turn_count']}/"
                f"{len(gold_turns)} -> "
                f"{bank_ceiling['available_source_turn_count']}/"
                f"{len(gold_turns)}; bank={len(selected)}",
                flush=True,
            )
    finally:
        provider.cleanup()

    artifact = {
        "config": {
            "embed_model": args.embed_model,
            "model_identity": model_identity,
            "top_k": args.top_k,
            "rerank_pool": args.rerank_pool,
            "entity_seed_k": args.entity_seed_k,
            "entity_closure_slots": args.entity_closure_slots,
            "continuation_slots": args.continuation_slots,
            "context_bridge_slots": args.context_bridge_slots,
        },
        "summary": aggregate(results),
        "cohort_sha256": replay._sha256_json([
            {
                "query_id": row["query_id"],
                "input_sha256": row["preflight"]["input_sha256"],
                "rerank_sha256": row["preflight"]["rerank_sha256"],
                "candidate_bank_sha256": row["preflight"][
                    "candidate_bank_sha256"
                ],
            }
            for row in results
        ]),
        "queries": results,
    }
    if args.compare_preflight:
        prior = json.loads(args.compare_preflight.read_text(encoding="utf-8"))
        artifact["comparison"] = {
            "artifact": str(args.compare_preflight),
            "expected": prior.get("cohort_sha256"),
            "actual": artifact["cohort_sha256"],
            "matched": prior.get("cohort_sha256") == artifact["cohort_sha256"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "cohort_sha256": artifact["cohort_sha256"],
        "summary": artifact["summary"],
        "comparison": artifact.get("comparison"),
    }, indent=2))
    if artifact.get("comparison", {}).get("matched") is False:
        raise RuntimeError("cohort preflight mismatch")


if __name__ == "__main__":
    main()
