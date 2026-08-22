"""Replay one paid write-time extraction across several queries without LLM calls."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path


_HERE = Path(__file__).resolve().parent
sys.path = [
    entry for entry in sys.path if Path(entry or ".").resolve() != _HERE
]

from memory_bench.dataset import get_dataset  # noqa: E402


def _load_workspace_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load_workspace_module(
    "memory_bench.memory.chronological_presentation",
    _HERE / "chronological_presentation.py",
)
_load_workspace_module(
    "memory_bench.memory.write_synthesis_selection",
    _HERE / "write_synthesis_selection.py",
)
_PROVIDER_MODULE = _load_workspace_module(
    "memory_bench.memory._yantrikdb_workspace",
    _HERE / "yantrikdb.py",
)
Provider = _PROVIDER_MODULE.YantrikDBWriteTimeSynthesisMemoryProvider


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--split", default="100k")
    parser.add_argument("--categories", default="knowledge_update,temporal_reasoning")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--keep-store",
        action="store_true",
        help="leave the rebuilt store on disk for local recall diagnostics",
    )
    args = parser.parse_args()

    recorded = {
        row["axis"]: json.loads(row["model_response"])["items"]
        for line in args.debug.read_text(encoding="utf-8").splitlines()
        for row in [json.loads(line)]
        if row.get("model_response")
    }

    def replay_axis(self, axis, rows):
        items = self._normalize_write_items(
            recorded[axis], {row["evidence_id"] for row in rows}
        )
        return items, {
            "axis": axis,
            "evidence_rows": len(rows),
            "items": len(items),
            "model_response": "replayed",
        }

    Provider._extract_axis = replay_axis
    categories = {value.strip() for value in args.categories.split(",") if value.strip()}
    dataset = get_dataset("beam")
    documents = dataset.load_documents(args.split, user_ids={args.unit})
    queries = [
        query
        for query in dataset.load_queries(args.split)
        if query.user_id == args.unit
        and query.meta.get("question_category") in categories
    ]

    provider = Provider()
    provider.prepare(args.store, {args.unit}, True)
    output = []
    try:
        provider.ingest(documents)
        for query in queries:
            docs, raw = provider.retrieve(query.query, 40, args.unit)
            context = "\n\n".join(
                f"## Memory {index}\n{doc.content}"
                for index, doc in enumerate(docs, 1)
            )
            row = {
                "query_id": query.id,
                "query": query.query,
                "gold_answers": query.gold_answers,
                "selection": raw,
                "context": context,
                "documents": [doc.content for doc in docs],
            }
            output.append(row)
            print(
                f"{query.id}: mode={(raw or {}).get('selection_mode', 'atomic')} "
                f"returned={len(docs)}"
            )
            for index, doc in enumerate(docs, 1):
                print(f"  {index:02d} {doc.content.replace(chr(10), ' ')[:240]}")
    finally:
        if not args.keep_store:
            provider.cleanup()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
