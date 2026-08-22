"""Build and replay the query-free, turn-level role-aware AMB provider."""

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != HERE]

from memory_bench.dataset import get_dataset
from memory_bench.memory.yantrikdb import YantrikDBRoleAwareMemoryProvider


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--units", required=True)
    parser.add_argument("--split", default="100k")
    parser.add_argument("--categories", default="")
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    units = {value.strip() for value in args.units.split(",") if value.strip()}
    categories = {
        value.strip() for value in args.categories.split(",") if value.strip()
    }
    dataset = get_dataset("beam")
    documents = dataset.load_documents(args.split, user_ids=units)
    queries = [
        query
        for query in dataset.load_queries(args.split)
        if query.user_id in units
        and (not categories or query.meta.get("question_category") in categories)
    ]

    provider = YantrikDBRoleAwareMemoryProvider()
    provider.prepare(args.store, units, reset=not args.reuse)
    output = []
    try:
        if not args.reuse:
            provider.ingest(documents)
        for query in queries:
            docs, raw = provider.retrieve(query.query, args.k, query.user_id)
            context = "\n\n".join(
                f"## Memory {index}\n{doc.content}"
                for index, doc in enumerate(docs, 1)
            )
            output.append({
                "query_id": query.id,
                "query": query.query,
                "gold_answers": query.gold_answers,
                "selection": raw,
                "context": context,
                "documents": [doc.content for doc in docs],
            })
            print(
                f"{query.id}: speaker={(raw or {}).get('requested_speaker')} "
                f"returned={len(docs)}"
            )
    finally:
        provider.cleanup()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"results": output}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
