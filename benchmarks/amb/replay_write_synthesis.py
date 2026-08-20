"""Replay a paid write-synthesis extraction without another model call.

This rebuilds one AMB unit from the source documents, feeds the recorded raw
axis responses through the current normalizer/persistence path, and reports
engine recall ranks. It is intended for candidate-pool and retrieval-policy
experiments where changing extraction would confound the result.
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

# Executing this file directly puts benchmarks/amb first on sys.path, where
# yantrikdb.py would shadow the installed native yantrikdb package.
_HERE = Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != _HERE
]

from memory_bench.dataset import get_dataset


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


GOLD_PATTERNS = [
    r"Bryan.*storytell|storytell.*Bryan",
    r"Shawn.*storytell|storytell.*Shawn",
    r"Bryan.*recommendation letter|recommendation letter.*Bryan",
    r"Matthew.*grant|grant.*Matthew",
    r"Matthew.*(introduction|career gap)|(introduction|career gap).*Matthew",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--unit", default="9")
    parser.add_argument("--split", default="100k")
    parser.add_argument("--query-id", default="9_event_ordering_0")
    parser.add_argument(
        "--expect-turns",
        help="comma-separated selected turns; exit nonzero on a mismatch",
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
    dataset = get_dataset("beam")
    documents = dataset.load_documents(args.split, user_ids={args.unit})
    query = next(q for q in dataset.load_queries(args.split) if q.id == args.query_id)

    provider = Provider()
    provider.prepare(args.store, {args.unit}, True)
    try:
        provider.ingest(documents)
        hits = provider._recall(query.query, 1000, args.unit)
        synthesized = [
            hit
            for hit in hits
            if (hit.get("metadata") or {}).get("synthesis_kind")
            == "multi_axis_item"
        ]
        print(f"documents={len(documents)} hits={len(hits)} synth={len(synthesized)}")
        for index, pattern in enumerate(GOLD_PATTERNS, 1):
            matches = [
                (rank, hit)
                for rank, hit in enumerate(synthesized, 1)
                if re.search(pattern, hit.get("text", ""), re.IGNORECASE)
                and "never met Bryan" not in hit.get("text", "")
            ]
            if not matches:
                print(f"gold{index}=MISS")
                continue
            rank, hit = matches[0]
            print(
                f"gold{index}=rank:{rank} score:{hit.get('score', 0):.4f} "
                f"text:{hit.get('text', '')}"
            )
        docs, raw = provider.retrieve(query.query, 40, args.unit)
        selected_turns = [
            int(match.group(1))
            for doc in docs
            for match in [re.search(r"\bTurn (\d+)\b", doc.content)]
            if match
        ]
        print(
            f"selection_mode={(raw or {}).get('selection_mode')} "
            f"child_selection={(raw or {}).get('child_selection')} "
            f"turns={','.join(map(str, selected_turns))}"
        )
        if args.expect_turns:
            expected_turns = [
                int(value.strip())
                for value in args.expect_turns.split(",")
                if value.strip()
            ]
            if selected_turns != expected_turns:
                print(
                    f"expected_turns={expected_turns} "
                    f"actual_turns={selected_turns}"
                )
                return 1
    finally:
        provider.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
