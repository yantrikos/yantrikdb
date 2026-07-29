#!/usr/bin/env python3
"""Does the constitution tier earn its tokens?

`evaluate.py` measures whether a pack supplies KNOWLEDGE — questions the
model cannot otherwise answer. This measures whether it installs
BEHAVIOUR: tasks where a rule must fire even though the task is about
something else entirely, so nothing in the phrasing invites the model to
go looking for the rule.

Three conditions, same model, same prompts:

  baseline      no pack
  corpus        pack mounted, similarity-gated retrieval injected
                (exactly what evaluate.py calls "mounted")
  constitution  the same, plus db.pack_context() in the system prompt

The claim under test is narrow and falsifiable: **corpus ≈ baseline and
constitution > both.** If corpus alone already carries these tasks, the
constitution tier is redundant and should be deleted rather than shipped
— retrieval is strictly cheaper, since constitution rules cost tokens on
every single turn.

Usage:
    python packs/evaluate_tiers.py --model qwen3.5:4b
    python packs/evaluate_tiers.py --model qwen3.6:27b --model granite4:3b
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from yantrikdb import YantrikDB

from evaluate import (  # reuse one implementation of transport + scoring
    DIST,
    MIN_SIMILARITY,
    grade,
    load_jsonl,
    resolve_host,
)
import evaluate

HERE = Path(__file__).resolve().parent

BASE_SYSTEM = (
    "You are an assistant that manages an agent's persistent memory store. "
    "Carry out the request in at most three sentences. Be concrete: if you "
    "would store text, give the exact text; if you would refuse or change "
    "something, say so plainly."
)


def ask(model: str, system: str, user: str) -> str:
    """Delegates to evaluate.ask's transport by temporarily swapping the
    system prompt — the two harnesses must not drift in how they call the
    model, or their numbers stop being comparable."""
    saved = evaluate.SYSTEM
    evaluate.SYSTEM = system
    try:
        return evaluate.ask(model, user, None)
    finally:
        evaluate.SYSTEM = saved


def with_context(question: str, context: list[str]) -> str:
    if not context:
        return question
    joined = "\n".join(f"- {c}" for c in context)
    return (
        f"Reference material retrieved from an attached knowledge pack:\n{joined}\n\n"
        f"Using that material where relevant, carry out this request:\n{question}"
    )


def run(model: str, pack_dir: Path, top_k: int, min_similarity: float) -> dict:
    tasks = load_jsonl(pack_dir / "eval_apply.jsonl")
    candidates = sorted(DIST.glob(f"{pack_dir.name}-*.ydbpack"))
    if not candidates:
        raise SystemExit(f"no built pack for {pack_dir.name} — run: python packs/build.py --all")
    pack_file = candidates[-1]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = YantrikDB(str(Path(td) / "host.db"), 64)
        pack_id = db.mount_pack(str(pack_file))
        ctx = db.pack_context()
        if not ctx:
            raise SystemExit(
                f"{pack_dir.name} declares no constitution or coverage — "
                "nothing to measure. Add constitution.md and rebuild."
            )
        const_system = f"{BASE_SYSTEM}\n\n{ctx}"

        rows = []
        for t in tasks:
            hits = db.recall_text(t["q"], top_k=top_k)
            retrieved = [
                h["text"]
                for h in hits
                if h.get("scores", {}).get("similarity", 0.0) >= min_similarity
            ]
            answers = {
                "baseline": ask(model, BASE_SYSTEM, t["q"]),
                "corpus": ask(model, BASE_SYSTEM, with_context(t["q"], retrieved)),
                "constitution": ask(model, const_system, with_context(t["q"], retrieved)),
            }
            scored = {k: grade(v, t["expect"]) for k, v in answers.items()}
            rows.append({"id": t["id"], "retrieved": len(retrieved), **scored, "answers": answers})
            flags = "".join(
                "Y" if scored[c] else "." for c in ("baseline", "corpus", "constitution")
            )
            print(f"  {t['id']:<26} base/corpus/const  {flags}   (retrieved {len(retrieved)})")

        db.unmount_pack(pack_id)
        db.close()

    errors = sum(
        1 for r in rows for a in r["answers"].values() if a.startswith("<<error")
    )
    if errors:
        raise SystemExit(f"{errors} model calls failed — refusing to report a score.")

    n = len(rows)
    return {
        "model": model,
        "pack": pack_dir.name,
        "n": n,
        "baseline": sum(r["baseline"] for r in rows),
        "corpus": sum(r["corpus"] for r in rows),
        "constitution": sum(r["constitution"] for r in rows),
        "constitution_tokens": len(ctx) // 4,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--pack", default="agent-memory-discipline")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-similarity", type=float, default=MIN_SIMILARITY)
    ap.add_argument("--host", default=None)
    ap.add_argument("--out", type=Path, default=HERE / "efficacy-tiers.json")
    args = ap.parse_args()

    evaluate.OLLAMA = resolve_host(args.host)
    print(f"ollama: {evaluate.OLLAMA}")

    results = []
    for model in args.model or ["qwen3.5:4b"]:
        print(f"\n=== {model}  x  {args.pack} (rule application) ===")
        t0 = time.time()
        res = run(model, HERE / args.pack, args.top_k, args.min_similarity)
        res["seconds"] = round(time.time() - t0, 1)
        results.append(res)

    print("\n" + "=" * 76)
    print(f"{'model':<16}{'tasks':>7}{'baseline':>10}{'corpus':>9}{'constitution':>14}{'tok':>7}")
    print("-" * 76)
    for r in results:
        print(
            f"{r['model']:<16}{r['n']:>7}{r['baseline']:>10}{r['corpus']:>9}"
            f"{r['constitution']:>14}{r['constitution_tokens']:>7}"
        )
    print("=" * 76)

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
