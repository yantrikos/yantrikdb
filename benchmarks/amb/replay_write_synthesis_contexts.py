"""Build frozen AMB contexts from query-independent source-turn rollups.

This arm deliberately disables model-generated write-time extraction. It
persists verbatim user turns, builds semantic/global handles locally, and then
replays selected queries through the normal write-synthesis retrieval path.
Any attempt to invoke an extraction model fails closed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != HERE]

def _load_workspace_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_source_turn_rollups(provider_module, rollup_mode: str) -> type:
    """Configure a zero-model-ingest provider and guard the model boundary."""
    provider_module._WRITE_SYNTH_AXES = ()
    provider_module._WRITE_SYNTH_SOURCE_TURNS = True
    provider_module._WRITE_SYNTH_THREADS = True
    provider = provider_module.YantrikDBWriteTimeSynthesisMemoryProvider
    if rollup_mode == "global":
        provider._build_semantic_threads = staticmethod(lambda _items: [])

    def reject_axis(*_args, **_kwargs):
        raise RuntimeError("source-turn-only replay attempted model extraction")

    provider._extract_axis = reject_axis
    return provider


def select_queries(dataset, split: str, units: set[str], categories: set[str]):
    return [
        query
        for query in dataset.load_queries(split)
        if query.user_id in units
        and (not categories or query.meta.get("question_category") in categories)
    ]


def main() -> int:
    from memory_bench.dataset import get_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--units", required=True)
    parser.add_argument("--split", default="100k")
    parser.add_argument("--categories", default="summarization")
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument(
        "--rollup-mode",
        choices=("semantic", "global"),
        default="semantic",
        help="Use semantic handles or force the single global source timeline.",
    )
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    units = {value.strip() for value in args.units.split(",") if value.strip()}
    categories = {
        value.strip() for value in args.categories.split(",") if value.strip()
    }
    if not units:
        parser.error("--units must contain at least one unit")

    _load_workspace_module(
        "memory_bench.memory.chronological_presentation",
        HERE / "chronological_presentation.py",
    )
    _load_workspace_module(
        "memory_bench.memory.write_synthesis_selection",
        HERE / "write_synthesis_selection.py",
    )
    provider_module = _load_workspace_module(
        "memory_bench.memory._yantrikdb_source_turn_replay",
        HERE / "yantrikdb.py",
    )
    Provider = configure_source_turn_rollups(provider_module, args.rollup_mode)

    dataset = get_dataset("beam")
    documents = dataset.load_documents(args.split, user_ids=units)
    queries = select_queries(dataset, args.split, units, categories)
    provider = Provider()
    provider.prepare(args.store, units, reset=not args.reuse)
    results = []
    try:
        if not args.reuse:
            provider.ingest(documents)
        for query in queries:
            docs, trace = provider.retrieve(
                query.query,
                args.k,
                query.user_id,
            )
            context = "\n\n".join(
                f"## Memory {index}\n{doc.content}"
                for index, doc in enumerate(docs, 1)
            )
            results.append(
                {
                    "query_id": query.id,
                    "query": query.query,
                    "gold_answers": query.gold_answers,
                    "selection": trace,
                    "context": context,
                    "documents": [doc.content for doc in docs],
                }
            )
            print(
                f"{query.id}: mode={(trace or {}).get('selection_mode')} "
                f"children={(trace or {}).get('child_selection')} "
                f"returned={len(docs)}"
            )
    finally:
        provider.cleanup()

    payload = {
        "config": {
            "provider": "yantrikdb-write-synthesis",
            "ingest_mode": "verbatim_user_turn_rollups_v1",
            "rollup_mode": args.rollup_mode,
            "write_synthesis_axes": [],
            "source_turns": True,
            "threads": True,
            "external_ingest_calls": 0,
            "units": sorted(units),
            "categories": sorted(categories),
            "k": args.k,
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
