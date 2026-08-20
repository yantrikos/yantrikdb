"""Run an answer-only local BEAM probe over a replay artifact."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from memory_bench.dataset import get_dataset
from memory_bench.modes.rag import RAGMode


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
    context = "\n\n".join(
        f"## Memory {index}\n{document}"
        for index, document in enumerate(row["documents"], 1)
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
    print(
        json.dumps(
            {
                "query_id": query.id,
                "query": query.query,
                "gold_answers": query.gold_answers,
                "answer": answer.answer,
                "reasoning": answer.reasoning,
                "context_documents": len(row["documents"]),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
