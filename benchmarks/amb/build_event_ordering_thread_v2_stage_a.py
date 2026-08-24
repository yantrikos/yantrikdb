#!/usr/bin/env python3
"""Build and audit the zero-model-call event-ordering thread-v2 artifact.

The command is deliberately split in two. ``build`` cannot accept benchmark
source-turn labels; it freezes a treatment artifact from query text and
query-independent organizer records. ``audit`` accepts those labels only
after verifying the frozen artifact hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

try:
    from .analyze_event_ordering_v5_autopsy import (
        KNOWN_LABEL_DEFECTS,
        QUERY_ROUTE_COHORTS,
    )
    from .event_ordering_thread_v2 import (
        event_ordering_focus,
        is_event_ordering_chronology_query,
    )
except ImportError:  # pragma: no cover - direct script execution
    from analyze_event_ordering_v5_autopsy import (
        KNOWN_LABEL_DEFECTS,
        QUERY_ROUTE_COHORTS,
    )
    from event_ordering_thread_v2 import (
        event_ordering_focus,
        is_event_ordering_chronology_query,
    )

# Direct script execution puts benchmarks/amb first on sys.path, where the
# benchmark's yantrikdb.py provider shim would shadow the installed product.
_HERE = Path(__file__).resolve().parent
sys.path = [
    entry for entry in sys.path if Path(entry or ".").resolve() != _HERE
]


PROTOCOL = "amb-event-ordering-thread-v2-stage-a-v1"
FREEZE_PROTOCOL = "amb-event-ordering-thread-v2-freeze-v1"
AUDIT_PROTOCOL = "amb-event-ordering-thread-v2-membership-audit-v1"
EXPECTED_ROWS = 400
EXPECTED_TREATMENT_ROWS = 40
THREAD_LIMIT = 100
MAX_TOPICS = 3
MAX_HANDLE_MEMBERSHIPS = 3
MAX_EVIDENCE_PER_HANDLE = 12
_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_GENERIC_NAMES = {
    "Across",
    "Can",
    "Chats",
    "Conversations",
    "Different",
    "During",
    "Mention",
    "Only",
    "Order",
    "Throughout",
    "Walk",
}


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def index_unique(rows: list[dict], label: str) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        query_id = str(row.get("query_id") or "")
        if not query_id:
            raise ValueError(f"{label} row is missing query_id")
        if query_id in indexed:
            raise ValueError(f"duplicate {label} query_id {query_id!r}")
        indexed[query_id] = row
    return indexed


def query_entities(focus: str) -> list[str]:
    """Extract conservative title-case person anchors from query-only focus."""
    return list(
        dict.fromkeys(
            name
            for name in _NAME_RE.findall(focus)
            if name not in _GENERIC_NAMES and not name.isupper()
        )
    )


def _clean_item_text(text: str) -> str:
    return re.sub(r"^User said:\s*", "", text.strip(), flags=re.IGNORECASE)


def render_thread(items: list[dict]) -> str:
    rendered = []
    for item in items:
        created_at = float(item["created_at"])
        stamp = datetime.fromtimestamp(created_at, tz=UTC).strftime("%B-%d-%Y")
        turn = item.get("source_turn")
        marker = f"{stamp} | Turn {turn}" if turn is not None else stamp
        rendered.append(
            f"## Memory {item['position']}\n"
            f"[{marker}] User: {_clean_item_text(str(item.get('text') or ''))}"
        )
    return "\n\n".join(rendered)


def chronological_key(item: dict) -> tuple:
    turn = item.get("source_turn")
    return (
        float(item["created_at"]),
        turn is None,
        0 if turn is None else int(turn),
        str(item["rid"]),
    )


def validate_thread(result: dict) -> None:
    items = result.get("items") or []
    total = int(result.get("total", -1))
    returned = int(result.get("returned", -1))
    omitted = int(result.get("omitted", -1))
    if returned != len(items) or total != returned + omitted:
        raise ValueError("thread accounting is inconsistent")
    if omitted != 0 or total != returned:
        raise ValueError(
            f"thread truncation is prohibited: total={total}, "
            f"returned={returned}, omitted={omitted}"
        )
    positions = [int(item["position"]) for item in items]
    if positions != list(range(1, total + 1)):
        raise ValueError("thread positions are not continuous and complete")
    if items != sorted(items, key=chronological_key):
        raise ValueError("thread items are not in canonical chronological order")


def _resolve_artifact(root: Path, relative: str) -> Path:
    return root / Path(relative.replace("\\", "/"))


def _load_organizer_artifact(
    root: Path, entry: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    path = _resolve_artifact(root, str(entry["path"]))
    if sha256_path(path) != entry["sha256"]:
        raise ValueError(f"organizer artifact hash mismatch: {path}")
    artifact = read_json(path)
    if artifact.get("input_sha256") != entry.get("input_sha256"):
        raise ValueError(f"organizer input hash mismatch: {path}")
    return path, artifact


def _reset_db_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for candidate in path.parent.glob(path.name + "*"):
        if candidate.is_file():
            candidate.unlink()


def bounded_handle_evidence(
    raw_handles: list[dict],
    max_memberships: int = MAX_HANDLE_MEMBERSHIPS,
    max_evidence_per_handle: int = MAX_EVIDENCE_PER_HANDLE,
) -> list[tuple[int, dict, list[str]]]:
    """Apply the product's deterministic evidence-membership bound."""
    if max_memberships < 1 or max_evidence_per_handle < 1:
        raise ValueError("organizer bounds must be positive")
    evidence_by_handle = [
        list(dict.fromkeys(str(value) for value in raw.get("evidence_ids") or []))
        for raw in raw_handles
    ]
    owners: dict[str, list[int]] = {}
    for index, evidence_ids in enumerate(evidence_by_handle):
        for evidence_id in evidence_ids:
            owners.setdefault(evidence_id, []).append(index)
    permitted = {
        evidence_id: set(
            sorted(
                indexes,
                key=lambda index: (len(evidence_by_handle[index]), index),
            )[:max_memberships]
        )
        for evidence_id, indexes in owners.items()
    }
    bounded = []
    for index, (raw, evidence_ids) in enumerate(
        zip(raw_handles, evidence_by_handle, strict=True),
        1,
    ):
        kept = [
            evidence_id
            for evidence_id in evidence_ids
            if index - 1 in permitted[evidence_id]
        ][:max_evidence_per_handle]
        if kept:
            bounded.append((index, raw, kept))
    return bounded


def _persist_unit(db_path: Path, unit: str, artifact: dict) -> tuple[Any, list[dict]]:
    import yantrikdb
    from yantrikdb import OrganizationPlan, TopicHandle, persist_organization

    _reset_db_path(db_path)
    db = yantrikdb.YantrikDB.with_default(str(db_path))
    evidence_rids = {}
    for item in artifact.get("input_items") or []:
        evidence_id = str(item.get("id") or "")
        turn = item.get("turn")
        created_at = item.get("date")
        text = str(item.get("text") or "").strip()
        if (
            not evidence_id
            or evidence_id in evidence_rids
            or isinstance(turn, bool)
            or not isinstance(turn, int)
            or turn < 0
            or not isinstance(created_at, (int, float))
            or isinstance(created_at, bool)
            or not math.isfinite(float(created_at))
            or not text
        ):
            raise ValueError(f"unit {unit} has an invalid organizer input item")
        evidence_rids[evidence_id] = db.record(
            text,
            memory_type="episodic",
            metadata={
                "source_turn": turn,
                "first_mention_turn": turn,
                "speaker": "user",
                "organizer_evidence_id": evidence_id,
            },
            namespace="default",
            source="user",
            idempotency_key=f"amb:thread-v2:{unit}:evidence:{evidence_id}",
            created_at=float(created_at),
        )

    handles = []
    raw_handles = artifact.get("handles") or []
    for index, raw, evidence_ids in bounded_handle_evidence(raw_handles):
        missing = sorted(set(evidence_ids) - set(evidence_rids))
        if missing:
            raise ValueError(f"unit {unit} handle {index} has missing evidence: {missing}")
        if not evidence_ids:
            raise ValueError(f"unit {unit} handle {index} is empty")
        handles.append(
            TopicHandle(
                id=f"amb-unit-{unit}-topic-{index:03d}",
                label=str(raw.get("label") or f"Topic {index}").strip(),
                summary=str(raw.get("summary") or "").strip(),
                evidence_ids=tuple(evidence_rids[value] for value in evidence_ids),
                anchor_entities=tuple(raw.get("anchor_entities") or ()),
            )
        )
    writes = persist_organization(
        db,
        OrganizationPlan(tuple(handles)),
        idempotency_prefix=f"amb:thread-v2:{unit}:organizer",
        max_evidence_per_handle=MAX_EVIDENCE_PER_HANDLE,
        max_handle_memberships=MAX_HANDLE_MEMBERSHIPS,
    )
    progress = db.maintain_source_turn_backfill(10_000)
    if not progress.get("complete"):
        raise ValueError(f"unit {unit} source_turn maintenance did not complete")
    return db, writes


def select_topics(db: Any, focus: str, max_topics: int = MAX_TOPICS) -> tuple[list[str], list[dict]]:
    """Use bounded semantic recall over persisted inference records only."""
    hits = db.recall(
        query=focus,
        top_k=max_topics,
        namespace="default",
        source="inference",
        include_consolidated=True,
        skip_reinforce=True,
    )
    selected = []
    trace = []
    for hit in hits:
        metadata = hit.get("metadata") or {}
        if metadata.get("organizer_kind") != "query_independent_topic":
            continue
        rid = str(hit.get("rid") or "")
        if not rid or rid in selected:
            continue
        selected.append(rid)
        trace.append(
            {
                "rid": rid,
                "label": metadata.get("organizer_label"),
                "score": hit.get("score"),
            }
        )
        if len(selected) == max_topics:
            break
    if not selected:
        raise ValueError("bounded organizer lookup returned no topic handles")
    return selected, trace


def _control_rows(control: dict) -> list[dict]:
    rows = control.get("results") or []
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} frozen control rows, found {len(rows)}")
    index_unique(rows, "control")
    return rows


def build(args: argparse.Namespace) -> dict:
    control = read_json(args.control)
    rows = _control_rows(control)
    organizer_manifest = read_json(args.organizer_manifest)
    if organizer_manifest.get("gold_fields_present") is not False:
        raise ValueError("organizer manifest must explicitly contain no gold fields")

    # Freeze predicate results before looking at category labels. The category
    # join below is an audit only and is never passed to a selector.
    routing = [
        {
            "query_id": str(row["query_id"]),
            "selected": is_event_ordering_chronology_query(str(row["query"])),
            "focus_text": event_ordering_focus(str(row["query"])),
        }
        for row in rows
    ]
    route_by_id = index_unique(routing, "routing")
    selected_ids = [row["query_id"] for row in routing if row["selected"]]
    if len(selected_ids) != EXPECTED_TREATMENT_ROWS:
        raise ValueError(
            f"expected {EXPECTED_TREATMENT_ROWS} predicate hits, found {len(selected_ids)}"
        )
    for row in rows:
        category = (row.get("meta") or {}).get("question_category")
        selected = route_by_id[str(row["query_id"])]["selected"]
        if selected != (category == "event_ordering"):
            raise ValueError(f"predicate/category audit mismatch for {row['query_id']}")

    unit_queries: dict[str, list[dict]] = {}
    for row in rows:
        route = route_by_id[str(row["query_id"])]
        if route["selected"]:
            unit = str(row["query_id"]).split("_", 1)[0]
            unit_queries.setdefault(unit, []).append(row)

    treatment_by_id = {}
    organizer_inputs = {}
    for unit in sorted(unit_queries, key=int):
        entry = (organizer_manifest.get("units") or {}).get(unit)
        if entry is None:
            raise ValueError(f"organizer manifest is missing unit {unit}")
        path, organizer = _load_organizer_artifact(args.organizer_root, entry)
        organizer_inputs[unit] = {
            "path": str(entry["path"]).replace("\\", "/"),
            "sha256": entry["sha256"],
            "input_sha256": entry["input_sha256"],
        }
        db_path = args.work_dir / "yantrikdb" / f"{unit}.db"
        db, writes = _persist_unit(db_path, unit, organizer)
        try:
            expected_writes = len(
                bounded_handle_evidence(organizer.get("handles") or [])
            )
            if len(writes) != expected_writes:
                raise ValueError(f"unit {unit} did not persist every bounded handle")
            for control_row in unit_queries[unit]:
                query_id = str(control_row["query_id"])
                focus = str(route_by_id[query_id]["focus_text"] or "")
                if not focus:
                    raise ValueError(f"empty focus for selected row {query_id}")
                entities = query_entities(focus)
                phrases = [focus]
                topic_rids, topic_trace = select_topics(db, focus)
                thread = db.recall_thread_v2(
                    "default",
                    entities=entities,
                    limit=THREAD_LIMIT,
                    phrases=phrases,
                    topic_rids=topic_rids,
                )
                validate_thread(thread)
                treatment_by_id[query_id] = {
                    "focus_text": focus,
                    "entities": entities,
                    "phrases": phrases,
                    "topic_rids": topic_rids,
                    "topic_selection": topic_trace,
                    "thread": thread,
                    "context": render_thread(thread["items"]),
                }
        finally:
            db.close()

    artifact_rows = []
    for control_row in rows:
        query_id = str(control_row["query_id"])
        control_context = str(control_row.get("context") or "")
        treatment = treatment_by_id.get(query_id)
        treatment_context = treatment["context"] if treatment else control_context
        artifact_rows.append(
            {
                "query_id": query_id,
                "query": str(control_row.get("query") or ""),
                "predicate_selected": route_by_id[query_id]["selected"],
                "focus_text": route_by_id[query_id]["focus_text"],
                "control_context": control_context,
                "control_context_sha256": hashlib.sha256(
                    control_context.encode("utf-8")
                ).hexdigest(),
                "treatment_context": treatment_context,
                "non_event_byte_identical": (
                    treatment_context.encode("utf-8") == control_context.encode("utf-8")
                ),
                "selection": (
                    None
                    if treatment is None
                    else {
                        key: treatment[key]
                        for key in (
                            "entities",
                            "phrases",
                            "topic_rids",
                            "topic_selection",
                        )
                    }
                ),
                "thread": None if treatment is None else treatment["thread"],
            }
        )

    artifact = {
        "protocol": PROTOCOL,
        "external_model_calls": 0,
        "gold_source_turns_available_during_build": False,
        "predicate_frozen_before_category_join": True,
        "product_commit": args.product_commit,
        "harness_commit": args.harness_commit,
        "wheel_path": str(args.wheel),
        "wheel_sha256": sha256_path(args.wheel),
        "python_version": platform.python_version(),
        "control_path": str(args.control),
        "control_sha256": sha256_path(args.control),
        "organizer_manifest_sha256": sha256_path(args.organizer_manifest),
        "organizer_inputs": organizer_inputs,
        "encryption_mode": "plaintext",
        "phrase_route_available": True,
        "source_turn_backfill_complete": True,
        "thread_api": "recall_thread_v2",
        "thread_limit": THREAD_LIMIT,
        "max_topic_rids_per_query": MAX_TOPICS,
        "max_evidence_per_handle": MAX_EVIDENCE_PER_HANDLE,
        "max_handle_memberships": MAX_HANDLE_MEMBERSHIPS,
        "row_count": len(artifact_rows),
        "treatment_row_count": len(selected_ids),
        "rows": artifact_rows,
    }
    write_json(args.output, artifact)
    freeze = {
        "protocol": FREEZE_PROTOCOL,
        "artifact_path": str(args.output),
        "artifact_sha256": sha256_path(args.output),
        "control_sha256": artifact["control_sha256"],
        "organizer_manifest_sha256": artifact["organizer_manifest_sha256"],
        "product_commit": args.product_commit,
        "harness_commit": args.harness_commit,
        "wheel_sha256": artifact["wheel_sha256"],
        "external_model_calls_before_freeze": 0,
    }
    write_json(args.freeze, freeze)
    return freeze


def _coverage(rows: list[dict]) -> dict:
    references = sum(len(row["source_turns"]) for row in rows)
    covered = sum(len(row["present_source_turns"]) for row in rows)
    recalls = [
        len(row["present_source_turns"]) / len(row["source_turns"]) for row in rows
    ]
    return {
        "queries": len(rows),
        "source_turn_references": references,
        "covered_source_turns": covered,
        "macro_recall": fmean(recalls),
        "micro_recall": covered / references,
        "nonzero_queries": sum(value > 0 for value in recalls),
        "exact_queries": sum(value == 1 for value in recalls),
    }


def audit(args: argparse.Namespace) -> dict:
    freeze = read_json(args.freeze)
    if freeze.get("protocol") != FREEZE_PROTOCOL:
        raise ValueError("unexpected freeze protocol")
    actual_hash = sha256_path(args.artifact)
    if actual_hash != freeze.get("artifact_sha256"):
        raise ValueError("artifact changed after freeze; refusing the gold join")
    artifact = read_json(args.artifact)
    if artifact.get("protocol") != PROTOCOL:
        raise ValueError("unexpected Stage A artifact protocol")
    membership = read_json(args.membership)
    artifact_rows = index_unique(artifact.get("rows") or [], "artifact")
    membership_rows = index_unique(membership.get("results") or [], "membership")
    selected = [row for row in artifact_rows.values() if row["predicate_selected"]]
    if len(artifact_rows) != EXPECTED_ROWS or len(selected) != EXPECTED_TREATMENT_ROWS:
        raise ValueError("Stage A artifact has the wrong 400/40 row shape")

    joined = []
    for row in selected:
        query_id = row["query_id"]
        gold = membership_rows.get(query_id)
        if gold is None:
            raise ValueError(f"membership is missing {query_id}")
        source_turns = sorted(set(gold.get("source_turns") or []))
        if not source_turns:
            raise ValueError(f"membership has no source turns for {query_id}")
        returned_turns = {
            item.get("source_turn")
            for item in (row.get("thread") or {}).get("items") or []
            if item.get("source_turn") is not None
        }
        joined.append(
            {
                "query_id": query_id,
                "source_turns": source_turns,
                "present_source_turns": sorted(set(source_turns) & returned_turns),
                "missing_source_turns": sorted(set(source_turns) - returned_turns),
            }
        )
    joined_by_id = {row["query_id"]: row for row in joined}
    clean = [row for row in joined if row["query_id"] not in KNOWN_LABEL_DEFECTS]
    phrase = [
        joined_by_id[query_id]
        for query_id in sorted(QUERY_ROUTE_COHORTS["bounded_focus_phrase"])
    ]
    broad = [
        joined_by_id[query_id]
        for query_id in sorted(QUERY_ROUTE_COHORTS["broad_compound_topic_union"])
    ]
    clean_summary = _coverage(clean)
    phrase_summary = _coverage(phrase)
    broad_summary = _coverage(broad)
    carla = joined_by_id["10_event_ordering_1"]
    douglas = joined_by_id["13_event_ordering_1"]

    non_event_identical = all(
        row["non_event_byte_identical"]
        for row in artifact_rows.values()
        if not row["predicate_selected"]
    )
    omitted_zero = all((row.get("thread") or {}).get("omitted") == 0 for row in selected)
    gates = {
        "row_shape_400_40": len(artifact_rows) == 400 and len(selected) == 40,
        "non_event_contexts_byte_identical": non_event_identical,
        "all_event_threads_untruncated": omitted_zero,
        "clean_macro_recall_at_least_0_55": clean_summary["macro_recall"] >= 0.55,
        "clean_micro_recall_at_least_0_55": clean_summary["micro_recall"] >= 0.55,
        "clean_nonzero_at_least_30": clean_summary["nonzero_queries"] >= 30,
        "clean_exact_at_least_8": clean_summary["exact_queries"] >= 8,
        "phrase_cohort_micro_at_least_0_55": phrase_summary["micro_recall"] >= 0.55,
        "broad_cohort_micro_at_least_0_55": broad_summary["micro_recall"] >= 0.55,
        "carla_6_of_6": len(carla["present_source_turns"]) == 6,
        "douglas_at_least_4_of_9": len(douglas["present_source_turns"]) >= 4,
    }
    report = {
        "protocol": AUDIT_PROTOCOL,
        "artifact_sha256": actual_hash,
        "freeze_sha256": sha256_path(args.freeze),
        "membership_sha256": sha256_path(args.membership),
        "clean_37": clean_summary,
        "all_40": _coverage(joined),
        "bounded_focus_phrase_22": phrase_summary,
        "broad_compound_topic_union_13": broad_summary,
        "carla": carla,
        "douglas": douglas,
        "gates": gates,
        "stage_a_pass": all(gates.values()),
        "rows": joined,
    }
    write_json(args.output, report)
    return report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build")
    build_parser.add_argument("--control", type=Path, required=True)
    build_parser.add_argument("--organizer-manifest", type=Path, required=True)
    build_parser.add_argument("--organizer-root", type=Path, required=True)
    build_parser.add_argument("--work-dir", type=Path, required=True)
    build_parser.add_argument("--product-commit", required=True)
    build_parser.add_argument("--harness-commit", required=True)
    build_parser.add_argument("--wheel", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--freeze", type=Path, required=True)

    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--artifact", type=Path, required=True)
    audit_parser.add_argument("--freeze", type=Path, required=True)
    audit_parser.add_argument("--membership", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    result = build(args) if args.command == "build" else audit(args)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if args.command == "audit" and not result["stage_a_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
