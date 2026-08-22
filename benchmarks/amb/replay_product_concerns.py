"""Persist a query-free concern artifact and replay AMB event-ordering queries."""

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
    ConcernItem,
    ConcernPlan,
    OrganizationPlan,
    TopicHandle,
    persist_concerns,
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


def _artifact_plan(
    atomics: list[dict], artifact: dict, *, complete_singletons: bool = False
) -> tuple[ConcernPlan, str, dict[str, tuple[str, ...]]]:
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
            "concern artifact input does not match replay store: "
            f"expected {artifact.get('input_sha256')}, got {digest}"
        )
    known = {item["id"]: item for item in atomics}
    concerns = []
    source_ids_by_concern = {}
    invalid = set()
    assigned_ids = set()
    for value in artifact.get("items") or []:
        evidence_ids = list(dict.fromkeys(value.get("evidence_ids") or []))
        invalid.update(rid for rid in evidence_ids if rid not in known)
        child_rids = tuple(
            known[rid]["rid"] for rid in evidence_ids if rid in known
        )
        if not child_rids:
            continue
        assigned_ids.update(evidence_ids)
        concern_id = str(value.get("id") or "")
        concerns.append(
            ConcernItem(
                id=concern_id,
                text=str(value.get("text") or ""),
                evidence_ids=child_rids,
                anchor_entities=tuple(value.get("anchor_entities") or ()),
            )
        )
        source_ids_by_concern[concern_id] = tuple(
            rid for rid in evidence_ids if rid in known
        )
    if invalid:
        raise ValueError(f"concern artifact has invalid evidence IDs: {sorted(invalid)}")
    if complete_singletons:
        for atomic_id, atomic in known.items():
            if atomic_id in assigned_ids:
                continue
            identity = hashlib.sha256(
                f"{digest}\0{atomic_id}\0{atomic['text']}".encode()
            ).hexdigest()[:24]
            concern_id = f"singleton-{identity}"
            concerns.append(
                ConcernItem(
                    id=concern_id,
                    text=atomic["text"],
                    evidence_ids=(atomic["rid"],),
                )
            )
            source_ids_by_concern[concern_id] = (atomic_id,)
    return ConcernPlan(tuple(concerns)), digest, source_ids_by_concern


def _topic_plan(
    artifact: dict,
    concern_plan: ConcernPlan,
    concern_writes: list[dict],
    source_ids_by_concern: dict[str, tuple[str, ...]],
    digest: str,
) -> OrganizationPlan:
    handles = artifact.get("handles") or []
    handle_sources = [set(handle.get("evidence_ids") or []) for handle in handles]
    concern_rids = {
        item.id: write["consolidated_rid"]
        for item, write in zip(concern_plan.items, concern_writes)
    }
    assignments = [[] for _ in handles]
    memberships = {item.id: 0 for item in concern_plan.items}
    candidates = []
    for item in concern_plan.items:
        source_ids = set(source_ids_by_concern[item.id])
        for handle_index, sources in enumerate(handle_sources):
            overlap = len(source_ids & sources)
            if overlap:
                candidates.append(
                    (-overlap, -len(source_ids), item.id, handle_index)
                )
    for _, _, concern_id, handle_index in sorted(candidates):
        if memberships[concern_id] >= 3 or len(assignments[handle_index]) >= 12:
            continue
        assignments[handle_index].append(concern_rids[concern_id])
        memberships[concern_id] += 1

    topics = []
    for index, (handle, child_rids) in enumerate(zip(handles, assignments), 1):
        if not child_rids:
            continue
        label = str(handle.get("label") or f"Topic trajectory {index}").strip()
        identity = hashlib.sha256(f"{digest}\0{label}".encode()).hexdigest()[:24]
        topics.append(
            TopicHandle(
                id=f"concern-topic-{identity}",
                label=label,
                summary=str(handle.get("summary") or "").strip(),
                evidence_ids=tuple(child_rids),
                anchor_entities=tuple(handle.get("anchor_entities") or ()),
            )
        )
    return OrganizationPlan(tuple(topics))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--organizer", type=Path)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--split", default="100k")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-handles", type=int, default=16)
    parser.add_argument("--handle-weight", type=float, default=0.0)
    parser.add_argument("--skip-persist", action="store_true")
    parser.add_argument("--complete-singletons", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    db_path = args.store / "yantrikdb" / f"{args.unit}.db"
    atomics = _ORGANIZER_MODULE._load_atomics(db_path)
    plan, digest, source_ids_by_concern = _artifact_plan(
        atomics, artifact, complete_singletons=args.complete_singletons
    )
    organizer = (
        json.loads(args.organizer.read_text(encoding="utf-8"))
        if args.organizer
        else None
    )
    if organizer and organizer.get("input_sha256") != digest:
        raise ValueError("topic organizer artifact does not match concern input")

    provider = Provider()
    provider.prepare(args.store, {args.unit}, False)
    output = []
    try:
        db = provider._db_for(args.unit)
        writes = [] if args.skip_persist else persist_concerns(
                db,
                plan,
                idempotency_prefix=f"amb:product-concerns:{digest}",
            )
        print(f"persisted concerns={len(writes)}")
        topic_writes = []
        if organizer is not None and not args.skip_persist:
            topic_plan = _topic_plan(
                organizer,
                plan,
                writes,
                source_ids_by_concern,
                digest,
            )
            topic_writes = persist_organization(
                db,
                topic_plan,
                idempotency_prefix=f"amb:concern-topics:{digest}",
            )
        print(f"persisted concern topics={len(topic_writes)}")
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
            concern_count = sum(
                (hit.get("metadata") or {}).get("organizer_kind")
                == "query_independent_concern"
                for hit in hits
            )
            print(
                f"{query.id}: returned={len(hits)} concerns={concern_count}"
            )
            for index, hit in enumerate(hits, 1):
                metadata = hit.get("metadata") or {}
                text = str(hit.get("text") or "").replace("\n", " ")
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
