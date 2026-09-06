#!/usr/bin/env python3
"""Deploy simulation: open a store built by an OLDER engine with the current
build, run the re-extraction heal, and census the claims table before and
after — judge-free, on the operator's own machine.

This is the CT128 scenario end to end: schema migration on open, legacy
extractor claims present, current extractor applied by `reextract_claims`.

    python benchmarks/prod_reextract_sim.py --store legacy_store/memory.db --out sim.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prod_extraction_census import is_junk_endpoint  # noqa: E402


def census(path: Path) -> dict:
    conn = sqlite3.connect(str(path))
    claims = conn.execute(
        "SELECT src, rel_type, dst, extractor FROM claims WHERE tombstoned=0"
    ).fetchall()
    ver = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    conn.close()
    ext = [c for c in claims if c[3] in ("heuristic_v1", "learned_v1")]
    junk = [c for c in ext if is_junk_endpoint(c[0]) or is_junk_endpoint(c[2])]
    by_rel = Counter(c[1] for c in ext)
    return {
        "schema_version": ver[0] if ver else None,
        "claims_total": len(claims),
        "extractor_claims": len(ext),
        "other_claims": len(claims) - len(ext),
        "junk_extractor_claims": len(junk),
        "junk_share": round(len(junk) / len(ext), 4) if ext else None,
        "by_rel": dict(by_rel.most_common(10)),
        "sample": [f"{c[0]} -{c[1]}-> {c[2]}" for c in ext[:8]],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep-copy", action="store_true", help="work on a copy, leave the input untouched")
    args = ap.parse_args()

    import yantrikdb

    work = args.store
    if args.keep_copy:
        work = args.store.with_name("memory-healed.db")
        for suffix in ("", "-wal", "-shm"):
            src = args.store.with_name(args.store.name + suffix)
            if src.exists():
                shutil.copy(src, work.with_name(work.name + suffix))
    before = census(work)

    t0 = time.perf_counter()
    db = yantrikdb.YantrikDB.with_default(str(work))  # migrates on open
    open_s = time.perf_counter() - t0
    dry = db.reextract_claims(dry_run=True)
    t0 = time.perf_counter()
    report = db.reextract_claims()
    heal_s = time.perf_counter() - t0
    db.close()
    after = census(work)

    out = {
        "engine_version": yantrikdb.__version__,
        "engine_file": yantrikdb.__file__,
        "open_seconds": round(open_s, 1),
        "heal_seconds": round(heal_s, 1),
        "dry_run": dry,
        "report": {k: v for k, v in report.items() if k not in ("before_by_rel", "after_by_rel")},
        "before": before,
        "after": after,
    }
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: out[k] for k in ("engine_version", "open_seconds", "heal_seconds")}))
    print("report:", out["report"])
    for tag in ("before", "after"):
        c = out[tag]
        print(f"{tag}: schema v{c['schema_version']} claims={c['claims_total']} extractor={c['extractor_claims']} "
              f"junk={c['junk_extractor_claims']} ({c['junk_share']}) by_rel={list(c['by_rel'].items())[:6]}")
        print("   sample:", c["sample"][:5])
    return 0


if __name__ == "__main__":
    sys.exit(main())
