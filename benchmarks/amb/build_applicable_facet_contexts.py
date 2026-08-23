#!/usr/bin/env python3
"""Build standing-facet contexts with auditable form-conflict suppression."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

try:
    from .build_complete_facet_contexts import (
        MAX_FACET_TOKENS,
        _control_payload,
        _facet_snapshots,
        _ingest_and_extract,
        _json_bytes,
        _remove_store,
        _sha256_bytes,
        _sha256_file,
        _source_map,
        load_rows,
        normalize_directive,
        overlay_control_contexts,
        render_facet_panel,
        select_facets,
    )
    from .audit_knowledge_update_gold import load_conversations
    from .reorder_speaker_first_contexts import split_memory_blocks
except ImportError:  # pragma: no cover - direct script execution
    from build_complete_facet_contexts import (
        MAX_FACET_TOKENS,
        _control_payload,
        _facet_snapshots,
        _ingest_and_extract,
        _json_bytes,
        _remove_store,
        _sha256_bytes,
        _sha256_file,
        _source_map,
        load_rows,
        normalize_directive,
        overlay_control_contexts,
        render_facet_panel,
        select_facets,
    )
    from audit_knowledge_update_gold import load_conversations
    from reorder_speaker_first_contexts import split_memory_blocks


PROTOCOL = "applicable-standing-facet-composition-v4"
PREDICATE_VERSION = "facet-form-conflict-v1"
EXPECTED_ROWS = 400
EXPECTED_INSTRUCTION_TARGETS = 40
EXPECTED_FACETS = 90

CONDITION_RE = re.compile(
    r"\bwhen\s+I\s+ask\s+about\s+(.+?)(?:[.!?]\s*)?$", re.IGNORECASE
)
CHRONOLOGY_RE = re.compile(
    r"\b(?:in\s+order|order\s+in\s+which|chronological(?:ly)?|sequence)\b",
    re.IGNORECASE,
)
HISTORY_RE = re.compile(
    r"\b(?:brought\s+up|mentioned|discussed|throughout|across)\b",
    re.IGNORECASE,
)
DATE_REQUEST_RE = re.compile(
    r"^\s*when\b|\b(?:what|which)\s+(?:date|time)\b|\bdate\s+of\b|"
    r"\b(?:due|deadline|deadlines|scheduled|scheduling|meeting\s+time|"
    r"appointment\s+time)\b",
    re.IGNORECASE,
)
PROCESS_RE = re.compile(r"\b(?:steps?|stages?|process(?:es)?|procedure)\b", re.I)
PROCESS_REQUEST_RE = re.compile(
    r"\b(?:what|which|how|walk\s+me\s+through|order)\b", re.IGNORECASE
)
FORMAT_RE = re.compile(
    r"\b(?:format|formatting|organize|organise|organization|design|layout|"
    r"structure|style|citation|references?|markup|html|bullets?|headings?)\b",
    re.IGNORECASE,
)
DATE_RULE_RE = re.compile(
    r"\b(?:date|dates|time|times|time\s*zone|timezone|deadline|deadlines|"
    r"timeline|timelines|schedule|scheduling|meeting|appointment)\b",
    re.IGNORECASE,
)
FORMAT_RULE_RE = re.compile(
    r"\b(?:format|formatting|bullet|bullets|heading|headings|citation\s+style|"
    r"syntax\s+highlighting|tree\s+diagram|visual\s+aid|markup|html|layout|"
    r"minimalist)\b",
    re.IGNORECASE,
)


def parse_condition(directive: str) -> tuple[str, str] | None:
    """Return exact action and condition for the narrow v1 conditional form."""
    match = CONDITION_RE.search(directive.strip())
    if match is None:
        return None
    action = directive[: match.start()].strip(" ,;:")
    condition = match.group(1).strip(" ,;:.!?")
    if not action or not condition:
        return None
    return action, condition


def directive_type(action: str) -> str:
    """Classify only answer-form transforms; everything else is non-transforming."""
    if DATE_RULE_RE.search(action):
        return "date_time"
    if FORMAT_RULE_RE.search(action):
        return "formatting_structure"
    return "non_transforming"


def scope_type(condition: str) -> str:
    """Classify the request shape named by the directive's own condition."""
    if PROCESS_RE.search(condition):
        return "process_timeline"
    if DATE_RULE_RE.search(condition):
        return "date_time"
    if FORMAT_RE.search(condition):
        return "formatting_structure"
    return "other"


def query_shape(query: str) -> dict[str, bool]:
    """Return deterministic, non-exclusive answer-shape features."""
    return {
        "chronology_of_mentions": bool(
            CHRONOLOGY_RE.search(query) and HISTORY_RE.search(query)
        ),
        "date_time_request": bool(DATE_REQUEST_RE.search(query)),
        "process_timeline_request": bool(
            PROCESS_RE.search(query) and PROCESS_REQUEST_RE.search(query)
        ),
        "formatting_request": bool(FORMAT_RE.search(query)),
    }


def facet_decision(query: str, facet: dict) -> dict:
    """Default-include, suppressing only a positive form conflict."""
    text = str(facet.get("text") or "")
    parsed = parse_condition(text)
    shapes = query_shape(query)
    base = {
        "rid": str(facet.get("rid") or ""),
        "predicate_version": PREDICATE_VERSION,
        "query_shape": shapes,
        "condition_parsed": parsed is not None,
    }
    if parsed is None:
        return {
            **base,
            "action": None,
            "condition": None,
            "directive_type": "unparsed",
            "scope_type": "unparsed",
            "include": True,
            "reason": "default_include_unparsed",
        }

    action, condition = parsed
    rule_type = directive_type(action)
    rule_scope = scope_type(condition)
    typed = {
        **base,
        "action": action,
        "condition": condition,
        "directive_type": rule_type,
        "scope_type": rule_scope,
    }
    if not shapes["chronology_of_mentions"] or rule_type == "non_transforming":
        return {**typed, "include": True, "reason": "default_include_no_conflict"}

    if rule_type == "date_time":
        compatible = (shapes["date_time_request"] and rule_scope == "date_time") or (
            shapes["process_timeline_request"] and rule_scope == "process_timeline"
        )
        return {
            **typed,
            "include": compatible,
            "reason": (
                "include_compatible_date_time"
                if compatible
                else "suppress_chronology_date_time_conflict"
            ),
        }

    compatible = shapes["formatting_request"] and rule_scope == "formatting_structure"
    return {
        **typed,
        "include": compatible,
        "reason": (
            "include_compatible_formatting"
            if compatible
            else "suppress_chronology_formatting_conflict"
        ),
    }


def select_applicable_facets(
    query: str, facets: list[dict]
) -> tuple[list[dict], list[dict]]:
    ordered = select_facets(facets)
    decisions = [facet_decision(query, facet) for facet in ordered]
    selected = [
        facet
        for facet, decision in zip(ordered, decisions, strict=True)
        if decision["include"]
    ]
    return selected, decisions


def build_context_rows(
    rows: list[dict],
    facets_by_namespace: dict[str, dict],
    token_counter: Callable[[str], int],
) -> tuple[dict, dict]:
    """Compose v4 treatment rows and a complete predicate trace."""
    treatment_rows = []
    row_audits = []
    target_rows = 0
    targets_retained = 0
    max_extra_tokens = 0
    reasons: Counter[str] = Counter()
    date_query_rows = 0
    date_facets_available = 0
    date_facets_retained = 0
    compatible_process_timeline_inclusions = 0

    inventory = [
        (namespace, facet)
        for namespace, lane in facets_by_namespace.items()
        for facet in lane.get("facets") or []
    ]
    parsed_inventory = sum(
        parse_condition(str(facet.get("text") or "")) is not None
        for _, facet in inventory
    )

    for row in rows:
        query_id = str(row.get("query_id") or "")
        query = str(row.get("query") or "")
        metadata = row.get("meta") or {}
        namespace = str(metadata.get("conversation_id") or "")
        if not query_id or not query or not namespace:
            raise ValueError(f"row lacks query identity fields: {query_id!r}")
        lane = facets_by_namespace.get(namespace)
        if lane is None or int(lane.get("omitted") or 0) != 0:
            raise ValueError(f"row {query_id!r} lacks a complete facet inventory")
        facets = list(lane.get("facets") or [])
        selected, decisions = select_applicable_facets(query, facets)
        reasons.update(decision["reason"] for decision in decisions)
        shapes = query_shape(query)
        if shapes["date_time_request"]:
            date_query_rows += 1
            date_facets_available += sum(
                decision["directive_type"] == "date_time" for decision in decisions
            )
            date_facets_retained += sum(
                decision["directive_type"] == "date_time" and decision["include"]
                for decision in decisions
            )
        if shapes["chronology_of_mentions"] and shapes["process_timeline_request"]:
            compatible_process_timeline_inclusions += sum(
                decision["reason"] == "include_compatible_date_time"
                and decision["scope_type"] == "process_timeline"
                for decision in decisions
            )

        panel = render_facet_panel(selected) if selected else ""
        panel_tokens = token_counter(panel)
        if panel_tokens > MAX_FACET_TOKENS:
            raise ValueError(
                f"row {query_id!r} facet panel uses {panel_tokens} tokens; "
                f"limit is {MAX_FACET_TOKENS}"
            )
        reference_context = str(row.get("context") or "")
        treatment_context = panel + reference_context
        ordinary_suffix = treatment_context[len(panel) :]
        if ordinary_suffix != reference_context:
            raise AssertionError(f"row {query_id!r} changed ordinary context bytes")
        if split_memory_blocks(ordinary_suffix) != split_memory_blocks(
            reference_context
        ):
            raise AssertionError(f"row {query_id!r} changed ordinary memory blocks")

        target = str(metadata.get("instruction_being_tested") or "").strip()
        target_retained = None
        if target:
            target_rows += 1
            target_retained = normalize_directive(target) in {
                normalize_directive(str(facet.get("text") or "")) for facet in selected
            }
            targets_retained += int(target_retained)

        max_extra_tokens = max(max_extra_tokens, panel_tokens)
        treatment_rows.append({"query_id": query_id, "context": treatment_context})
        row_audits.append(
            {
                "query_id": query_id,
                "namespace": namespace,
                "predicate_version": PREDICATE_VERSION,
                "query_shape": shapes,
                "available_facets": len(facets),
                "selected_facets": len(selected),
                "suppressed_facets": len(facets) - len(selected),
                "selected_rids": [str(facet.get("rid") or "") for facet in selected],
                "facet_tokens": panel_tokens,
                "ordinary_context_exact": ordinary_suffix == reference_context,
                "target_retained": target_retained,
                "decisions": decisions,
            }
        )

    audit = {
        "protocol": PROTOCOL,
        "predicate_version": PREDICATE_VERSION,
        "rows": len(rows),
        "facet_inventory": len(inventory),
        "parsed_facets": parsed_inventory,
        "parse_rate": parsed_inventory / len(inventory) if inventory else 0.0,
        "max_facet_tokens_allowed": MAX_FACET_TOKENS,
        "max_facet_tokens_observed": max_extra_tokens,
        "ordinary_contexts_exact": sum(
            row["ordinary_context_exact"] for row in row_audits
        ),
        "instruction_target_rows": target_rows,
        "instruction_targets_retained": targets_retained,
        "rows_with_suppression": sum(
            row["suppressed_facets"] > 0 for row in row_audits
        ),
        "suppressed_facets": sum(row["suppressed_facets"] for row in row_audits),
        "decision_reasons": dict(sorted(reasons.items())),
        "date_query_rows": date_query_rows,
        "date_facets_available_on_date_queries": date_facets_available,
        "date_facets_retained_on_date_queries": date_facets_retained,
        "compatible_process_timeline_inclusions": (
            compatible_process_timeline_inclusions
        ),
        "selection_uses_query_text": True,
        "selection_uses_category": False,
        "selection_uses_gold_rubric_answer_or_score": False,
        "unparsed_policy": "default_include",
        "row_audits": row_audits,
    }
    return {"results": treatment_rows}, audit


def validate_full400_preflight(audit: dict) -> None:
    errors = []
    expected = {
        "rows": EXPECTED_ROWS,
        "facet_inventory": EXPECTED_FACETS,
        "parsed_facets": EXPECTED_FACETS,
        "ordinary_contexts_exact": EXPECTED_ROWS,
        "instruction_target_rows": EXPECTED_INSTRUCTION_TARGETS,
        "instruction_targets_retained": EXPECTED_INSTRUCTION_TARGETS,
        "rows_with_suppression": 28,
        "suppressed_facets": 53,
        "date_query_rows": 41,
        "date_facets_available_on_date_queries": 68,
        "date_facets_retained_on_date_queries": 68,
        "compatible_process_timeline_inclusions": 1,
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            errors.append(f"expected {key}={value}, got {audit.get(key)}")
    if audit["max_facet_tokens_observed"] > MAX_FACET_TOKENS:
        errors.append("facet token ceiling exceeded")
    expected_reasons = {
        "default_include_no_conflict": 1746,
        "include_compatible_date_time": 1,
        "suppress_chronology_date_time_conflict": 41,
        "suppress_chronology_formatting_conflict": 12,
    }
    if audit["decision_reasons"] != expected_reasons:
        errors.append(
            f"expected decision_reasons={expected_reasons}, "
            f"got {audit['decision_reasons']}"
        )
    for row in audit["row_audits"]:
        for decision in row["decisions"]:
            if decision["include"]:
                continue
            if not row["query_shape"]["chronology_of_mentions"]:
                errors.append(f"non-chronology suppression in {row['query_id']}")
            if decision["directive_type"] not in {"date_time", "formatting_structure"}:
                errors.append(f"unsupported suppression type in {row['query_id']}")
            if not decision["condition_parsed"]:
                errors.append(f"unparsed directive suppressed in {row['query_id']}")
    if errors:
        raise RuntimeError(
            "pre-call applicability gate failed:\n- " + "\n- ".join(errors)
        )


def main() -> int:
    from memory_bench.utils import count_tokens

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--control-contexts", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--yantrikdb-python", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", default="deepseek-v4-flash:0731-cloud")
    args = parser.parse_args()

    rows = overlay_control_contexts(
        load_rows(args.results), load_rows(args.control_contexts)
    )
    namespaces = list(
        dict.fromkeys(
            str((row.get("meta") or {}).get("conversation_id") or "") for row in rows
        )
    )
    if any(not namespace for namespace in namespaces):
        raise ValueError("one or more rows lack conversation_id")
    if args.db.exists() and not args.overwrite:
        raise FileExistsError(f"store already exists: {args.db}")
    if args.overwrite:
        _remove_store(args.db)
    args.db.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.yantrikdb_python.resolve()))
    from yantrikdb import YantrikDB

    conversations = load_conversations(args.documents)
    db = YantrikDB.with_default(str(args.db))
    try:
        extraction_audits = _ingest_and_extract(db, conversations, namespaces)
    finally:
        db.close()

    source_map = _source_map(args.db)
    db = YantrikDB.with_default(str(args.db))
    try:
        snapshots = _facet_snapshots(db, namespaces, source_map)
        treatment, audit = build_context_rows(rows, snapshots, count_tokens)
    finally:
        db.close()
    validate_full400_preflight(audit)

    db = YantrikDB.with_default(str(args.db))
    try:
        replay_snapshots = _facet_snapshots(db, namespaces, source_map)
        replay_treatment, replay_audit = build_context_rows(
            rows, replay_snapshots, count_tokens
        )
    finally:
        db.close()
    replay_exact = _json_bytes(replay_treatment) == _json_bytes(treatment)
    replay_selection_exact = [
        row["selected_rids"] for row in replay_audit["row_audits"]
    ] == [row["selected_rids"] for row in audit["row_audits"]]
    replay_trace_exact = [row["decisions"] for row in replay_audit["row_audits"]] == [
        row["decisions"] for row in audit["row_audits"]
    ]
    if not replay_exact or not replay_selection_exact or not replay_trace_exact:
        raise RuntimeError("fresh-open applicability replay was not exact")

    control = _control_payload(rows)
    control_path = args.out_dir / "control.json"
    treatment_path = args.out_dir / "applicable-facets.json"
    preflight_path = args.out_dir / "preflight.json"
    control_path.write_bytes(_json_bytes(control))
    treatment_path.write_bytes(_json_bytes(treatment))

    query_ids = [str(row["query_id"]) for row in rows]
    preflight = {
        **audit,
        "status": "passed",
        "product_path": {
            "raw_turn_ingestion": True,
            "persisted_facet_extraction": True,
            "store_reopened_before_selection": True,
            "query_shape_filtering": True,
            "second_fresh_open_replay_exact": replay_exact,
            "second_fresh_open_selection_exact": replay_selection_exact,
            "second_fresh_open_predicate_trace_exact": replay_trace_exact,
            "category_or_gold_used_during_selection": False,
        },
        "extraction_audits": extraction_audits,
        "source_sha256": {
            "results": _sha256_file(args.results),
            "control_contexts": _sha256_file(args.control_contexts),
            "documents": _sha256_file(args.documents),
        },
        "ordered_query_ids_sha256": _sha256_bytes(
            json.dumps(query_ids, separators=(",", ":")).encode("utf-8")
        ),
        "arms": {
            "control": {
                "file": control_path.name,
                "sha256": _sha256_file(control_path),
            },
            "treatment": {
                "file": treatment_path.name,
                "sha256": _sha256_file(treatment_path),
            },
        },
        "external_evaluation": {
            "model": args.model,
            "answer_repeats": 1,
            "judge_repeats": 1,
            "answer_calls": len(rows) * 2,
            "judge_calls": len(rows) * 2,
            "synthetic_benchmark_data_only": True,
            "real_companion_memories_included": False,
        },
    }
    preflight_path.write_bytes(_json_bytes(preflight))
    print(
        json.dumps(
            {
                key: value
                for key, value in preflight.items()
                if key not in {"row_audits", "extraction_audits"}
            },
            indent=2,
        )
    )
    print(f"preflight_sha256={_sha256_file(preflight_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
