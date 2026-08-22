"""Persist a query-free organizer artifact and replay one AMB unit."""

import argparse
import hashlib
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
_ORGANIZER_MODULE = _load_workspace_module(
    "memory_bench.memory._global_organizer_probe",
    _HERE / "global_organizer_probe.py",
)
_PROVIDER_MODULE = _load_workspace_module(
    "memory_bench.memory._yantrikdb_workspace",
    _HERE / "yantrikdb.py",
)
Provider = _PROVIDER_MODULE.YantrikDBWriteTimeSynthesisMemoryProvider


def _validated_handle_plans(
    atomics: list[dict], artifact: dict, require_exhaustive: bool
) -> list[dict]:
    known = {item["id"]: item for item in atomics}
    plans = []
    assigned = set()
    invalid = set()
    for index, handle in enumerate(artifact.get("handles") or [], 1):
        evidence_ids = list(dict.fromkeys(handle.get("evidence_ids") or []))
        invalid.update(evidence_id for evidence_id in evidence_ids if evidence_id not in known)
        valid_ids = [evidence_id for evidence_id in evidence_ids if evidence_id in known]
        assigned.update(valid_ids)
        child_rids = sorted({known[evidence_id]["rid"] for evidence_id in valid_ids})
        if not child_rids:
            continue
        anchor_entities = [
            entity
            for entity in dict.fromkeys(handle.get("anchor_entities") or [])
            if isinstance(entity, str) and entity.strip() and entity not in known
        ]
        plans.append(
            {
                "index": index,
                "handle": handle,
                "child_rids": child_rids,
                "anchor_entities": anchor_entities,
            }
        )
    if invalid:
        raise ValueError(f"organizer artifact has invalid evidence IDs: {sorted(invalid)}")
    missing = sorted(set(known) - assigned)
    if require_exhaustive and missing:
        raise ValueError(
            f"organizer artifact leaves {len(missing)} atomic items unassigned"
        )
    return plans


def _persist_handles(
    provider,
    unit: str,
    artifact: dict,
    db_path: Path,
    *,
    require_exhaustive: bool = True,
) -> int:
    import yantrikdb

    atomics = _ORGANIZER_MODULE._load_atomics(db_path)
    model_rows = [
        {
            "id": item["id"],
            "turn": item["turn"],
            "axis": item["axis"],
            "text": item["text"],
        }
        for item in atomics
    ]
    serialized = json.dumps(
        model_rows, ensure_ascii=True, separators=(",", ":")
    )
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    if digest != artifact.get("input_sha256"):
        raise ValueError(
            "organizer artifact input does not match the replay store: "
            f"expected {artifact.get('input_sha256')}, got {digest}"
        )

    plans = _validated_handle_plans(atomics, artifact, require_exhaustive)
    db = provider._db_for(unit)
    persisted = 0
    for plan in plans:
        index = plan["index"]
        handle = plan["handle"]
        child_rids = plan["child_rids"]
        label = str(handle.get("label") or f"Topic trajectory {index}").strip()
        summary = str(handle.get("summary") or "").strip()
        text = f"Topic trajectory: {label}. {summary}".strip()
        identity = hashlib.sha256(
            "\0".join([digest, label, *child_rids]).encode()
        ).hexdigest()[:24]
        result = yantrikdb.record_synthesis(
            db,
            child_rids,
            text,
            "topic_trajectory",
            f"amb-topic-organizer-v1:{identity}",
            granularity="rollup",
            embedding=db.embed(text),
            metadata={
                "child_rids": child_rids,
                "thread_entities": plan["anchor_entities"],
                "thread_builder": "llm_topic_organizer_v1",
                "organizer_model": artifact.get("model"),
                "organizer_input_sha256": digest,
                "organizer_label": label,
            },
        )
        if result.get("consolidated_rid"):
            persisted += 1
    return persisted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--split", default="100k")
    parser.add_argument("--categories", default="event_ordering")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="persist a diagnostic artifact that does not cover every atomic item",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    db_path = args.store / "yantrikdb" / f"{args.unit}.db"
    provider = Provider()
    provider.prepare(args.store, {args.unit}, False)
    persisted = _persist_handles(
        provider,
        args.unit,
        artifact,
        db_path,
        require_exhaustive=not args.allow_incomplete,
    )
    print(f"persisted organizer handles={persisted}")

    categories = {
        value.strip() for value in args.categories.split(",") if value.strip()
    }
    dataset = get_dataset("beam")
    queries = [
        query
        for query in dataset.load_queries(args.split)
        if query.user_id == args.unit
        and query.meta.get("question_category") in categories
    ]
    output = []
    try:
        for query in queries:
            docs, raw = provider.retrieve(query.query, 40, args.unit)
            row = {
                "query_id": query.id,
                "query": query.query,
                "gold_answers": query.gold_answers,
                "selection": raw,
                "documents": [doc.content for doc in docs],
            }
            output.append(row)
            print(
                f"{query.id}: mode={(raw or {}).get('selection_mode')} "
                f"children={(raw or {}).get('child_selection')} "
                f"returned={len(docs)}"
            )
            for index, doc in enumerate(docs, 1):
                print(f"  {index:02d} {doc.content.replace(chr(10), ' ')[:220]}")
    finally:
        provider.cleanup()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
