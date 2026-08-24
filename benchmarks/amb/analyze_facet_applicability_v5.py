#!/usr/bin/env python3
"""Apply the final v5 gates to a validated mean-of-three paired result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .analyze_facet_applicability_v4 import analyze
    from .paired_frozen_context_eval import _run_fingerprint, load_rows
except ImportError:  # pragma: no cover - direct script execution
    from analyze_facet_applicability_v4 import analyze
    from paired_frozen_context_eval import _run_fingerprint, load_rows


EXPECTED_PROTOCOL = "paired-independent-mean-of-three-v1"
EXPECTED_REPLICATE_SEEDS = [20260828, 20260829, 20260830]
EXPECTED_BOOTSTRAP_SEED = 20260831


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze_v5(result: dict, source_rows: list[dict]) -> dict:
    config = result.get("run_config") or {}
    if result.get("run_fingerprint") != _run_fingerprint(config):
        raise ValueError("combined result fingerprint does not match its run config")
    if config.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("result is not the frozen independent mean-of-three protocol")
    if result.get("replicate_seeds") != EXPECTED_REPLICATE_SEEDS:
        raise ValueError("result run seeds do not match the frozen v5 seeds")
    if result.get("model_seeds") != EXPECTED_REPLICATE_SEEDS:
        raise ValueError("result model seeds do not match the frozen v5 seeds")
    if config.get("replicate_seeds") != EXPECTED_REPLICATE_SEEDS:
        raise ValueError("run config seeds do not match the frozen v5 seeds")
    if config.get("model_seeds") != EXPECTED_REPLICATE_SEEDS:
        raise ValueError("run config model seeds do not match the frozen v5 seeds")
    if config.get("replicate_count") != 3:
        raise ValueError("result does not contain exactly three replicates")
    category_rows = [
        {**row, "query_id": row.get("query_id") or row.get("id")} for row in source_rows
    ]
    report = analyze(result, category_rows, EXPECTED_BOOTSTRAP_SEED)
    report["protocol"] = "facet-applicability-v5-final-power-analysis-v1"
    report["finality"] = (
        "default-on-promotion" if report["promotion_passed"] else "terminal-opt-in"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    report = analyze_v5(result, load_rows(args.source))
    report["result_sha256"] = _sha256_file(args.result)
    report["source_sha256"] = _sha256_file(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
