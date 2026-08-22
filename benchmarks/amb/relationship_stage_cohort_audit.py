"""Audit bounded relationship-stage routing without answer or judge calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean

try:
    from .prepare_contextual_synthesis_pair import (
        _FAMILY_SUPPORT_QUERY_RE,
        _RELATIONSHIP_TIMELINE_QUERY_RE,
        render_relationship_support_stages,
        render_relationship_thread,
    )
except ImportError:  # pragma: no cover - direct script execution
    from prepare_contextual_synthesis_pair import (
        _FAMILY_SUPPORT_QUERY_RE,
        _RELATIONSHIP_TIMELINE_QUERY_RE,
        render_relationship_support_stages,
        render_relationship_thread,
    )


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_query(row: dict) -> dict:
    """Route one query, falling back to the unchanged bank on abstention."""
    query = row.get("query") or ""
    bank_ceiling = row.get("bank_ceiling") or (
        (row.get("preflight") or {}).get("source_evidence_ceiling") or {}
    )
    gold_turns = set(row.get("source_turns") or [])
    bank_turns = set(bank_ceiling.get("available_source_turns") or [])
    route = "abstain"
    selector_audit = {}
    error = None
    try:
        if _FAMILY_SUPPORT_QUERY_RE.search(query):
            route = "relationship_support_stages"
            _, selector_audit = render_relationship_support_stages(row)
        elif _RELATIONSHIP_TIMELINE_QUERY_RE.search(query):
            route = "relationship_thread"
            _, selector_audit = render_relationship_thread(row)
    except ValueError as exc:
        route = "abstain"
        error = str(exc)

    if route == "abstain":
        selected_turns = set()
        effective_turns = bank_turns
        selected_count = row.get("candidate_bank_count") or len(bank_turns)
    else:
        selected_turns = set(selector_audit.get("selected_turns") or [])
        effective_turns = gold_turns & selected_turns
        selected_count = selector_audit.get("selected_row_count") or 0

    return {
        "query_id": row.get("query_id"),
        "query": query,
        "route": route,
        "error": error,
        "bank_row_count": row.get("candidate_bank_count") or 0,
        "selected_row_count": selected_count,
        "gold_source_turn_count": len(gold_turns),
        "bank_available_source_turns": sorted(gold_turns & bank_turns),
        "selected_source_turns": sorted(gold_turns & selected_turns),
        "effective_available_source_turns": sorted(gold_turns & effective_turns),
        "source_turn_delta": len(gold_turns & effective_turns) - len(
            gold_turns & bank_turns
        ),
        "selector": selector_audit,
    }


def aggregate(rows: list[dict]) -> dict:
    fired = [row for row in rows if row["route"] != "abstain"]
    bank_available = sum(len(row["bank_available_source_turns"]) for row in rows)
    effective_available = sum(
        len(row["effective_available_source_turns"]) for row in rows
    )
    fired_bank_rows = sum(row["bank_row_count"] for row in fired)
    fired_selected_rows = sum(row["selected_row_count"] for row in fired)
    fired_bank_gold = sum(
        len(row["bank_available_source_turns"]) for row in fired
    )
    fired_selected_gold = sum(
        len(row["selected_source_turns"]) for row in fired
    )
    return {
        "query_count": len(rows),
        "fired_query_count": len(fired),
        "fire_rate": len(fired) / len(rows) if rows else 0.0,
        "abstained_query_count": len(rows) - len(fired),
        "abstention_rate": (len(rows) - len(fired)) / len(rows) if rows else 0.0,
        "route_counts": {
            route: sum(row["route"] == route for row in rows)
            for route in sorted({row["route"] for row in rows})
        },
        "bank_available_source_turn_count": bank_available,
        "effective_available_source_turn_count": effective_available,
        "source_turn_delta": effective_available - bank_available,
        "fired_bank_gold_turn_count": fired_bank_gold,
        "fired_selected_gold_turn_count": fired_selected_gold,
        "fired_gold_retention": (
            fired_selected_gold / fired_bank_gold if fired_bank_gold else None
        ),
        "fired_bank_row_count": fired_bank_rows,
        "fired_selected_row_count": fired_selected_rows,
        "fired_row_reduction": (
            1.0 - fired_selected_rows / fired_bank_rows
            if fired_bank_rows else None
        ),
        "mean_selected_rows_when_fired": (
            fmean(row["selected_row_count"] for row in fired) if fired else 0.0
        ),
        "negative_query_count": sum(row["source_turn_delta"] < 0 for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-cohort-sha256")
    args = parser.parse_args()

    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    cohort_sha256 = cohort.get("cohort_sha256")
    if args.expect_cohort_sha256 and cohort_sha256 != args.expect_cohort_sha256:
        raise ValueError(
            f"cohort hash mismatch: expected {args.expect_cohort_sha256}, "
            f"got {cohort_sha256}"
        )
    rows = [audit_query(row) for row in cohort.get("queries") or []]
    artifact = {
        "protocol": "query-dependent-relationship-stage-cohort-audit-v1",
        "answer_calls": 0,
        "judge_calls": 0,
        "cohort_sha256": cohort_sha256,
        "summary": aggregate(rows),
        "queries": rows,
    }
    artifact["audit_sha256"] = _sha256_json(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "audit_sha256": artifact["audit_sha256"],
        "summary": artifact["summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
