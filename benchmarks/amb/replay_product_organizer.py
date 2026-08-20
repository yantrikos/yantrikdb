"""Replay an AMB organizer artifact through YantrikDB's public organizer API."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


_HERE = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != _HERE]

from memory_bench.dataset import get_dataset  # noqa: E402
from yantrikdb import (  # noqa: E402
    OrganizationPlan,
    TopicHandle,
    persist_organization,
    recall_organized,
)


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


def _artifact_plan(atomics: list[dict], artifact: dict) -> tuple[OrganizationPlan, str]:
    model_rows = [
        {
            "id": item["id"],
            "turn": item["turn"],
            "axis": item["axis"],
            "text": item["text"],
        }
        for item in atomics
    ]
    serialized = json.dumps(model_rows, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    if digest != artifact.get("input_sha256"):
        raise ValueError(
            "organizer artifact input does not match the replay store: "
            f"expected {artifact.get('input_sha256')}, got {digest}"
        )

    known = {item["id"]: item for item in atomics}
    handles = []
    assigned = set()
    invalid = set()
    for index, value in enumerate(artifact.get("handles") or [], 1):
        evidence_ids = list(dict.fromkeys(value.get("evidence_ids") or []))
        invalid.update(evidence_id for evidence_id in evidence_ids if evidence_id not in known)
        valid_ids = [evidence_id for evidence_id in evidence_ids if evidence_id in known]
        assigned.update(valid_ids)
        child_rids = tuple(known[evidence_id]["rid"] for evidence_id in valid_ids)
        if not child_rids:
            continue
        label = str(value.get("label") or f"Topic trajectory {index}").strip()
        summary = str(value.get("summary") or "").strip()
        identity = hashlib.sha256(
            "\0".join([digest, label, *sorted(child_rids)]).encode()
        ).hexdigest()[:24]
        handles.append(
            TopicHandle(
                id=f"amb-{identity}",
                label=label,
                summary=summary,
                evidence_ids=child_rids,
                anchor_entities=tuple(value.get("anchor_entities") or ()),
            )
        )

    if invalid:
        raise ValueError(f"organizer artifact has invalid evidence IDs: {sorted(invalid)}")
    missing = sorted(set(known) - assigned)
    if missing:
        raise ValueError(f"organizer artifact leaves {len(missing)} atomic items unassigned")
    return OrganizationPlan(tuple(handles)), digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--split", default="100k")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-handles", type=int, default=16)
    parser.add_argument("--handle-weight", type=float, default=0.0)
    parser.add_argument("--skip-persist", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    db_path = args.store / "yantrikdb" / f"{args.unit}.db"
    atomics = _ORGANIZER_MODULE._load_atomics(db_path)
    plan, digest = _artifact_plan(atomics, artifact)

    provider = Provider()
    provider.prepare(args.store, {args.unit}, False)
    output = []
    try:
        db = provider._db_for(args.unit)
        writes = (
            []
            if args.skip_persist
            else persist_organization(
                db,
                plan,
                idempotency_prefix=f"amb:product-organizer:{digest}",
            )
        )
        print(f"persisted organizer handles={len(writes)}")
        queries = [
            query
            for query in get_dataset("beam").load_queries(args.split)
            if query.user_id == args.unit
            and query.meta.get("question_category") == "event_ordering"
        ]
        for query in queries:
            hits = recall_organized(
                db,
                query.query,
                top_k=args.top_k,
                candidate_pool=1000,
                max_handles=args.max_handles,
                handle_weight=args.handle_weight,
                mode="auto",
                order="auto",
            )
            row = {
                "query_id": query.id,
                "query": query.query,
                "gold_answers": query.gold_answers,
                "hits": hits,
                "documents": [
                    "[Turn "
                    f"{(hit.get('metadata') or {}).get('first_mention_turn')}] "
                    f"User: {hit.get('text') or ''}"
                    for hit in hits
                ],
            }
            output.append(row)
            expanded = sum(
                "organized_handle_expansion" in (hit.get("why_retrieved") or [])
                for hit in hits
            )
            print(f"{query.id}: returned={len(hits)} expanded={expanded}")
            for index, hit in enumerate(hits, 1):
                text = str(hit.get("text") or "").replace("\n", " ")
                metadata = hit.get("metadata") or {}
                print(
                    f"  {index:02d} turn={metadata.get('first_mention_turn')} "
                    f"score={float(hit.get('score') or 0.0):.4f} {text[:180]}"
                )
    finally:
        provider.cleanup()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
