#!/usr/bin/env python3
"""Re-ingest a corpus's texts into a fresh store and census what the CURRENT
engine build extracts: claims by relation, junk share, entity-table quality,
and claim-chain exposure. Judge-free; run once per build and diff.

Why this exists (2026-09-05): on the production memory store the claims
table was 57% `leads`, 877 of them minted from the word "runs" ("CT128 runs
0.15.2"), and 63% of the claims that survive read-side phantom suppression
were junk edges such as `Python -leads-> R8g`. Claim-chain traversal walks
exactly those. BEAM cannot see any of this. This census is the instrument
that can.

Input: a JSON array of {rid, namespace, created_at, memory_type, text}
(exported from the store with sqlite3 -json). The corpus never needs to
leave the operator's machines; the census reports aggregates and a small,
optional sample.

    python benchmarks/prod_extraction_census.py --texts prod_active_texts.json \
        --label main --out census-main.json [--limit N] [--sample 12]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path


def is_junk_endpoint(name: str) -> bool:
    """Mechanical junk rule, mirrors the engine's read-side rejection plus
    the concrete shapes seen on production (None, bare numbers, versions)."""
    n = name.strip()
    if not n or not any(c.isalpha() for c in n):
        return True
    if n in {"None", "null", "NULL"}:
        return True
    toks = n.split()
    if len(toks) > 6:
        return True
    allcaps = [t for t in toks if t.isupper() and any(c.isalpha() for c in t)]
    return len(allcaps) > 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sample", type=int, default=12)
    ap.add_argument("--chain-deny", default="co_occurs_with,related_to",
                    help="relations the chain-hop deny-list excludes (for the exposure count)")
    args = ap.parse_args()

    import yantrikdb

    rows = json.load(open(args.texts, encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    store_dir = Path(tempfile.mkdtemp(prefix="census-"))
    path = store_dir / "census.db"
    db = yantrikdb.YantrikDB.with_default(str(path))
    t0 = time.perf_counter()
    for r in rows:
        db.record(
            r["text"], memory_type=r.get("memory_type") or "episodic",
            namespace=r.get("namespace") or "default",
            created_at=float(r["created_at"]) if r.get("created_at") else None,
        )
    ingest_s = time.perf_counter() - t0
    # Drain the materializer deterministically (think drains <= 4096 ops per call).
    t0 = time.perf_counter()
    drained = 0
    for _ in range(1 + len(rows) // 2000):
        db.think({"run_consolidation": False, "run_pattern_mining": False})
        drained += 1
    drain_s = time.perf_counter() - t0
    db.close()

    conn = sqlite3.connect(str(path))
    claims = conn.execute(
        "SELECT src, rel_type, dst, extractor, source_memory_rid FROM claims WHERE tombstoned=0"
    ).fetchall()
    entities = conn.execute("SELECT name, entity_type, mention_count FROM entities").fetchall()
    n_mem = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    conn.close()

    by_rel = Counter(c[1] for c in claims)
    by_ext = Counter(c[3] for c in claims)
    # A value object (no letters, has a digit) is a legitimate claim OBJECT
    # since #213 (`CT128 -runs-> 0.19.0`); only a junk SUBJECT, or a junk
    # object that is not a value, counts against the build.
    def _is_value(x):
        return bool(x) and not any(ch.isalpha() for ch in x) and any(ch.isdigit() for ch in x)
    junk = [c for c in claims if is_junk_endpoint(c[0]) or (is_junk_endpoint(c[2]) and not _is_value(c[2]))]
    clean = [c for c in claims if c not in junk]
    clean_by_rel = Counter(c[1] for c in clean)
    deny = set(x.strip() for x in args.chain_deny.split(",") if x.strip())
    # Chain exposure: edges a hop-2 traversal could follow = clean (both endpoints
    # pass the read-side predicate), provenance-bearing, not deny-listed.
    exposure = [c for c in clean if c[4] and c[1] not in deny]
    exposure_by_rel = Counter(c[1] for c in exposure)
    ent_junk = sum(1 for e in entities if is_junk_endpoint(e[0]))
    # Issue #213 admission classes (counts only; names never leave the box).
    names = [e[0] for e in entities]
    ent_classes = {
        "all_caps": sum(1 for n in names if len(n) >= 2 and n == n.upper() and n != n.lower()),
        "has_digit": sum(1 for n in names if any(ch.isdigit() for ch in n)),
        "no_letters": sum(1 for n in names if not any(ch.isalpha() for ch in n)),
        "four_plus_words": sum(1 for n in names if len(n.split()) >= 4),
        "long_40": sum(1 for n in names if len(n) >= 40),
        "possessive": sum(1 for n in names if n.endswith("'s") or n.endswith("’s")),
    }
    claims_allcaps_endpoint = sum(
        1 for c in claims
        if any(len(x) >= 2 and x == x.upper() and x != x.lower() for x in (c[0], c[2]))
    )
    ent_top = sorted(entities, key=lambda e: -(e[2] or 0))[:12]

    # "runs"-as-leads on this corpus: leads claims whose source text says " runs ".
    text_by_rid = {}
    # We cannot map fresh rids back to the input rids without the memory text; use text match.
    conn = sqlite3.connect(str(path))
    leads_from_runs = conn.execute(
        "SELECT COUNT(*) FROM claims c JOIN memories m ON m.rid=c.source_memory_rid "
        "WHERE c.tombstoned=0 AND c.rel_type='leads' AND lower(m.text) LIKE '% runs %'"
    ).fetchone()[0]
    runs_rel = conn.execute(
        "SELECT COUNT(*) FROM claims WHERE tombstoned=0 AND rel_type='runs'"
    ).fetchone()[0]
    place = conn.execute(
        "SELECT COUNT(*) FROM claims WHERE tombstoned=0 AND rel_type IN ('lives_in','hometown')"
    ).fetchone()[0]
    conn.close()

    out = {
        "label": args.label,
        "engine_version": yantrikdb.__version__,
        "engine_file": yantrikdb.__file__,
        "memories": n_mem,
        "ingest_seconds": round(ingest_s, 1),
        "drain_calls": drained,
        "drain_seconds": round(drain_s, 1),
        "claims_total": len(claims),
        "claims_by_extractor": dict(by_ext),
        "claims_by_rel": dict(by_rel.most_common()),
        "junk_claims": len(junk),
        "junk_share": round(len(junk) / len(claims), 4) if claims else None,
        "clean_claims": len(clean),
        "clean_by_rel": dict(clean_by_rel.most_common()),
        "leads_total": by_rel.get("leads", 0),
        "leads_from_runs_sentences": leads_from_runs,
        "runs_relation_claims": runs_rel,
        "place_claims_lives_in_hometown": place,
        "chain_exposure_edges": len(exposure),
        "chain_exposure_by_rel": dict(exposure_by_rel.most_common()),
        "entities_total": len(entities),
        "entities_junk": ent_junk,
        "entities_classes": ent_classes,
        "claims_with_allcaps_endpoint": claims_allcaps_endpoint,
        "entities_top12": [{"name": e[0], "type": e[1], "mentions": e[2]} for e in ent_top],
        "sample_clean_leads": [f"{c[0]} -leads-> {c[2]}" for c in clean if c[1] == "leads"][: args.sample],
        "sample_junk": [f"{c[0]} -{c[1]}-> {c[2]}" for c in junk][: args.sample],
        # Every extractor claim, so two runs can be diffed claim by claim.
        "all_claims": sorted(f"{c[0]} -{c[1]}-> {c[2]}" for c in claims),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("sample") and k not in ("claims_by_rel", "clean_by_rel", "chain_exposure_by_rel", "entities_top12")}, indent=1))
    print("top rels:", by_rel.most_common(8))
    print("chain exposure by rel:", exposure_by_rel.most_common(8))
    print("top entities:", [(e[0], e[2]) for e in ent_top[:8]])
    print("clean leads sample:", out["sample_clean_leads"][:6])
    return 0


if __name__ == "__main__":
    sys.exit(main())
