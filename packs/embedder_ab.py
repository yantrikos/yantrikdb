#!/usr/bin/env python3
"""Does a better embedder lift a pack? Same corpus, same questions, same model.

mcp-spec reached 49/53 with MISS_CONTENT at zero — every remaining
failure is retrieval or application, not missing knowledge. Two of them
are provably unreachable: the record titled "MCP deprecated Roots,
Sampling and Logging" scores 0.458 and ranks 39th of 213 for the query
"which three MCP features were deprecated together", which is close to
verbatim. No floor or top_k reaches rank 39 without dragging in 38
irrelevant records.

The engine's bundled default is potion-2M — a static lookup-table
embedding distilled from bge-base-en-v1.5, 64 dimensions, no
transformer inference. Its own docstring puts it at R@10 0.90 against
MiniLM's 1.00, and a 10-point recall gap is exactly the shape of a
correct record landing at rank 17 or 39.

This measures the swap end to end rather than arguing from that table.
The corpus is recorded into a fresh database under each embedder, the
same 53 questions are asked of the same model, and the answers are
graded by the same deterministic matcher. Only the embedder changes.

    python packs/embedder_ab.py --pack mcp-spec
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate  # noqa: E402
from yantrikdb import YantrikDB  # noqa: E402

# name -> dim. None means the engine's bundled default.
EMBEDDERS = [(None, 64), ("potion-base-8M", 256), ("potion-base-32M", 512)]


def build(pack: str, name: str | None, dim: int, td: Path) -> YantrikDB:
    db = YantrikDB(str(td / f"{name or 'bundled'}.db"), dim)
    if name:
        db.set_embedder_named(name)
    md = (HERE / pack / "corpus.md").read_text(encoding="utf-8")
    recs = [c.strip() for c in re.split(r"^## ", md, flags=re.M) if c.strip()]
    for r in recs:
        db.record_text(r)
    return db, len(recs)


def score(db, qs, model, k, floor):
    ok, ranks = 0, []
    for q in qs:
        hits = [h for h in db.recall_text(q["q"], top_k=k)
                if h.get("scores", {}).get("similarity", 0.0) >= floor]
        ans = evaluate.ask(model, q["q"], [h["text"] for h in hits])
        ok += evaluate.grade(ans, q["expect"], q.get("reject"))
        ranks.append(len(hits))
    return ok, sum(ranks) / len(ranks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="mcp-spec")
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--floors", default="0.55,0.60,0.65")
    a = ap.parse_args()

    evaluate.OLLAMA = evaluate.resolve_host(None)
    qs = evaluate.load_jsonl(HERE / a.pack / "eval.jsonl")
    ctl = evaluate.load_jsonl(HERE / "control.jsonl")
    floors = [float(f) for f in a.floors.split(",")]

    print(f"\n{a.pack}: {len(qs)} questions, {a.model}, top_k={a.top_k}")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        for name, dim in EMBEDDERS:
            label = name or "bundled potion-2M"
            try:
                db, n = build(a.pack, name, dim, Path(td))
            except Exception as exc:                       # noqa: BLE001
                print(f"\n  {label:<20} unavailable: {str(exc)[:90]}")
                continue
            print(f"\n  {label} ({dim}d, {n} records)")
            for f in floors:
                s, avg = score(db, qs, a.model, a.top_k, f)
                c, _ = score(db, ctl, a.model, a.top_k, f)
                print(f"    floor {f:<5} {s:>3}/{len(qs)}   controls {c}/{len(ctl)}"
                      f"   avg {avg:.1f} records injected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
