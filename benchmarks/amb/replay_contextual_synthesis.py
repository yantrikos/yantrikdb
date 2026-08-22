"""Replay one AMB synthesis query with contextual reranking over a frozen bank."""

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import urllib.request
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_provider(repo: Path):
    script_directory = Path(__file__).resolve().parent
    sys.path = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != script_directory
    ]
    _load_module(
        "memory_bench.memory.chronological_presentation",
        repo / "benchmarks" / "amb" / "chronological_presentation.py",
    )
    _load_module(
        "memory_bench.memory.write_synthesis_selection",
        repo / "benchmarks" / "amb" / "write_synthesis_selection.py",
    )
    return _load_module(
        "memory_bench.memory.yantrikdb_contextual_probe",
        repo / "benchmarks" / "amb" / "yantrikdb.py",
    ).YantrikDBGlobalSynthesisMemoryProvider


def _ollama_embeddings(
    texts: list[str], model: str, host: str
) -> list[list[float]]:
    if not host.startswith("http"):
        host = f"http://{host}"
    for wildcard in ("//0.0.0.0", "//[::]", "//::"):
        host = host.replace(wildcard, "//127.0.0.1")
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/embed",
        data=json.dumps({"model": model, "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    embeddings = payload.get("embeddings") or []
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Ollama returned {len(embeddings)} embeddings for {len(texts)} texts"
        )
    return embeddings


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _turn(text: str) -> int | None:
    match = re.search(r"\[(?:[A-Z][a-z]+-\d+-\d+ \| )?Turn (\d+)\]", text)
    return int(match.group(1)) if match else None


def _cited_turns(items: list[dict], telemetry: dict) -> set[int]:
    block_turns = telemetry.get("evidence_block_turns") or {}
    return {
        int(turn)
        for item in items
        for evidence_id in item.get("evidence_ids", [])
        for turn in block_turns.get(evidence_id, [])
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--user-id", default="9")
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--rerank-pool", type=int, default=1000)
    parser.add_argument(
        "--support-quotes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--gold-turns", default="")
    args = parser.parse_args()

    os.environ["YDB_BENCH_SYNTH_BLOCKS"] = str(args.rerank_pool)
    os.environ["YDB_BENCH_SYNTH_SUPPORT_QUOTES"] = (
        "1" if args.support_quotes else "0"
    )
    if os.environ.get("YDB_BENCH_SYNTH_EVIDENCE_CANDIDATES") == "1":
        os.environ["YDB_BENCH_SYNTH_MAX_ITEMS"] = str(max(
            args.top_k,
            int(os.environ.get("YDB_BENCH_SYNTH_MAX_ITEMS", "24")),
        ))
    provider_cls = _load_provider(args.repo.resolve())
    provider = provider_cls()
    provider.prepare(args.bank.resolve(), {args.user_id}, False)
    recalled = provider._recall(args.query, args.rerank_pool, args.user_id)
    user_hits = provider._select_evidence_hits(recalled)

    embeddings = _ollama_embeddings(
        [args.query, *(hit.get("text") or "" for hit in user_hits)],
        args.embed_model,
        args.ollama_host,
    )
    query_embedding, document_embeddings = embeddings[0], embeddings[1:]
    reranked = sorted(
        (
            dict(hit, _contextual_score=_cosine(query_embedding, embedding))
            for hit, embedding in zip(user_hits, document_embeddings, strict=True)
        ),
        key=lambda hit: (-hit["_contextual_score"], hit["_retrieval_rank"]),
    )
    selected = [
        dict(
            hit,
            _base_retrieval_rank=hit["_retrieval_rank"],
            _retrieval_rank=rank,
        )
        for rank, hit in enumerate(reranked[:args.top_k], 1)
    ]
    items, telemetry = provider._synthesize(args.query, selected)

    gold_turns = {
        int(value) for value in args.gold_turns.split(",") if value.strip()
    }
    rerank_by_turn = {
        turn: rank
        for rank, hit in enumerate(reranked, 1)
        if (turn := _turn(hit.get("text") or "")) is not None
    }
    candidate_turns = _cited_turns(
        telemetry.get("candidate_items", []), telemetry
    )
    result_turns = _cited_turns(telemetry.get("results", []), telemetry)
    artifact = {
        "query": args.query,
        "embed_model": args.embed_model,
        "generator_model": os.environ.get("YDB_BENCH_SYNTH_MODEL", ""),
        "raw_recall_count": len(recalled),
        "user_turn_count": len(user_hits),
        "selected_count": len(selected),
        "gold_turn_rerank": {
            str(turn): rerank_by_turn.get(turn) for turn in sorted(gold_turns)
        },
        "gold_turns_available": sorted(
            turn
            for turn in gold_turns
            if (rerank_by_turn.get(turn) or args.top_k + 1) <= args.top_k
        ),
        "candidate_gold_turns": sorted(gold_turns.intersection(candidate_turns)),
        "result_gold_turns": sorted(gold_turns.intersection(result_turns)),
        "selected_turns": [_turn(hit.get("text") or "") for hit in selected],
        "selected_evidence": [
            {
                "turn": _turn(hit.get("text") or ""),
                "contextual_rank": hit["_retrieval_rank"],
                "base_rank": hit["_base_retrieval_rank"],
                "contextual_score": hit["_contextual_score"],
            }
            for hit in selected
        ],
        "items": items,
        "telemetry": telemetry,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "gold_turn_rerank": artifact["gold_turn_rerank"],
        "gold_turns_available": artifact["gold_turns_available"],
        "candidate_gold_turns": artifact["candidate_gold_turns"],
        "result_gold_turns": artifact["result_gold_turns"],
        "candidate_count": telemetry.get("synthesis_items"),
        "result_count": len(items),
        "extraction_support_quote_events": telemetry.get(
            "extraction_support_quote_events", []
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
