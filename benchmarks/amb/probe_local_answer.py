"""Run an answer-only local BEAM probe over a replay artifact."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from memory_bench.dataset import get_dataset
from memory_bench.modes.rag import RAGMode
from memory_bench.models import QueryResult
from memory_bench.utils import count_tokens


def _load_workspace_ollama(path: Path):
    name = "memory_bench.llm._workspace_ollama_probe"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Ollama provider from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.OllamaLLM


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--judge-model")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-gold",
        action="store_true",
        help="Include gold text in the saved audit artifact after answering.",
    )
    parser.add_argument(
        "--handle-id",
        help="Restrict replay hits to one persisted organization handle.",
    )
    parser.add_argument("--split", default="100k")
    args = parser.parse_args()

    rows = json.loads(args.replay.read_text(encoding="utf-8"))
    row = next(
        (candidate for candidate in rows if candidate["query_id"] == args.query_id),
        None,
    )
    if row is None:
        parser.error(f"query {args.query_id!r} is absent from {args.replay}")

    dataset = get_dataset("beam")
    query = next(
        candidate
        for candidate in dataset.load_queries(args.split)
        if candidate.id == args.query_id
    )
    documents = row["documents"]
    if args.handle_id:
        handle_hits = [
            hit
            for hit in row.get("hits", [])
            if args.handle_id
            in ((hit.get("metadata") or {}).get("organization_handle_ids") or [])
        ]
        handle_hits.sort(key=lambda hit: (
            (hit.get("metadata") or {}).get("first_mention_at", 0),
            (hit.get("metadata") or {}).get("first_mention_turn", 999999),
            str(hit.get("rid") or ""),
        ))
        documents = [
            "[Turn "
            f"{int((hit.get('metadata') or {}).get('first_mention_turn', 0))}] "
            f"User: {hit.get('text') or ''}"
            for hit in handle_hits
        ]
        if not documents:
            parser.error(
                f"handle {args.handle_id!r} is absent from replay hits"
            )
    context = "\n\n".join(
        f"## Memory {index}\n{document}"
        for index, document in enumerate(documents, 1)
    )
    meta = dict(query.meta)
    meta["_prompt_fn"] = lambda question, supplied, meta=meta: dataset.build_rag_prompt(
        question,
        supplied,
        dataset.task_type,
        args.split,
        meta=meta,
    )
    OllamaLLM = _load_workspace_ollama(Path(__file__).with_name("ollama.py"))
    answer = RAGMode(
        llm=OllamaLLM(
            args.model,
            think=False,
            num_predict=1200,
            num_ctx=65536,
        )
    ).answer_from_context(
        query.query,
        context,
        dataset.task_type,
        meta=meta,
    )
    payload = {
        "query_id": query.id,
        "query": query.query,
        "answer_model": args.model,
        "answer": answer.answer,
        "reasoning": answer.reasoning,
        "context_documents": len(documents),
        "context_tokens": count_tokens(context),
    }
    if args.handle_id:
        payload["organization_handle_id"] = args.handle_id
    if args.include_gold:
        payload["gold_answers"] = query.gold_answers
    if args.judge_model:
        judged = QueryResult(
            query_id=query.id,
            query=query.query,
            answer=answer.answer,
            reasoning=answer.reasoning,
            context=context,
            context_tokens=payload["context_tokens"],
            retrieve_time_ms=0.0,
            gold_answers=query.gold_answers,
            correct=False,
            judge_reason="",
            meta=meta,
        )
        judge = OllamaLLM(
            args.judge_model,
            think=False,
            num_predict=1200,
            num_ctx=65536,
        )
        payload["judge_model"] = args.judge_model
        payload["score"] = dataset.score_result(judged, judge)
        payload["all_rubric_nuggets_matched"] = payload["score"] == 1.0
        payload["score_scope"] = (
            "mean rubric-nugget coverage; ordering and exact-N are not judged"
        )
    rendered = json.dumps(payload, indent=2, ensure_ascii=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
