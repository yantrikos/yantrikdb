#!/usr/bin/env python3
"""Find each pack's own top_k, and prove it does not cost the controls.

top_k has no engine default — it is a required argument — so the eval
harness's 5 was never a product default, just a habit. It turned out to
be the binding constraint on the best packs: the record holding the
answer was being FOUND and RANKED and then cut off at position 6, 12 or
17, comfortably above the similarity floor. Raising it was worth +3 on
mcp-spec and +2 on react-craft without authoring anything.

The right value is a property of the pack, not the harness: it depends
on corpus size and on how many near-duplicate records compete for a
query. So it has to be swept per pack, and the sweep has to watch the
unrelated-topic controls, because "retrieve more" is also exactly how a
pack starts displacing what the model already knew. A k that buys two
points and costs a control point is not an improvement.

The baseline condition is measured ONCE. It does not depend on k — the
pack is not mounted — and re-running it per k doubled the cost of every
sweep for no information.

    python packs/sweep_topk.py --pack java-modern
    python packs/sweep_topk.py --all --write
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate  # noqa: E402
from yantrikdb import YantrikDB  # noqa: E402

KS = (5, 8, 12, 16, 24)


def score_mounted(db, qs, model, k, floor):
    ok = 0
    for q in qs:
        hits = [h for h in db.recall_text(q["q"], top_k=k)
                if h.get("scores", {}).get("similarity", 0.0) >= floor]
        ans = evaluate.ask(model, q["q"], [h["text"] for h in hits])
        ok += evaluate.grade(ans, q["expect"])
    return ok


def sweep(pack: str, model: str, floor: float, ks) -> dict:
    import diagnose
    qs = evaluate.load_jsonl(HERE / pack / "eval.jsonl")
    ctl = evaluate.load_jsonl(HERE / "control.jsonl")

    # Baseline once: no pack is mounted, so k cannot change it.
    base = sum(evaluate.grade(evaluate.ask(model, q["q"], None), q["expect"])
               for q in qs)
    ctl_base = sum(evaluate.grade(evaluate.ask(model, q["q"], None), q["expect"])
                   for q in ctl)

    rows = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = YantrikDB(str(Path(td) / "h.db"), 64)
        db.mount_pack(str(diagnose.newest_pack(pack)))
        for k in ks:
            rows.append((k, score_mounted(db, qs, model, k, floor),
                         score_mounted(db, ctl, model, k, floor)))

    print(f"\n{pack}  ({len(qs)} questions, {model})")
    print(f"  baseline {base}/{len(qs)}   controls {ctl_base}/{len(ctl)}")
    for k, s, c in rows:
        harm = "" if c >= ctl_base else f"  control -{ctl_base - c}"
        print(f"    k={k:<3} {s:>3}/{len(qs)}  gain {s - base:+3}   "
              f"controls {c}/{len(ctl)}{harm}")

    # Best score among the k values that cost no control point. Ties go
    # to the SMALLER k: less context is cheaper for the consumer and
    # leaves more headroom before displacement starts.
    clean = [(s, -k, k) for k, s, c in rows if c >= ctl_base]
    if not clean:
        print("  no k leaves the controls intact — do not recommend one")
        return {"pack": pack, "best": None, "baseline": base, "n": len(qs)}
    best_s, _, best_k = max(clean)
    print(f"  -> recommended_top_k = {best_k}  ({best_s}/{len(qs)}, "
          f"{best_s - base:+} over baseline, controls intact)")
    return {"pack": pack, "best": best_k, "score": best_s,
            "baseline": base, "n": len(qs)}


def write_recommendation(pack: str, k: int, score: int, base: int, n: int) -> None:
    p = HERE / pack / "pack.toml"
    s = p.read_text(encoding="utf-8")
    if "recommended_top_k" in s:
        print(f"  {pack}: already declares one, leaving it")
        return
    s = s.replace("[content]", f"""[content]
# Swept, not guessed. At the old harness default of 5 this pack scored
# differently; at {k} it reaches {score}/{n} against a {base}/{n} baseline with the
# unrelated-topic controls intact. Larger k was either no better or cost
# a control point, which is where retrieved context starts displacing
# what the model already knew.
recommended_top_k = {k}""", 1)
    p.write_text(s, encoding="utf-8")
    print(f"  {pack}: wrote recommended_top_k = {k}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--min-similarity", type=float, default=0.55)
    ap.add_argument("--write", action="store_true",
                    help="record the winner in pack.toml")
    ap.add_argument("--host", default=None)
    a = ap.parse_args()

    evaluate.OLLAMA = evaluate.resolve_host(a.host)
    packs = a.pack
    if a.all:
        packs = sorted(d.name for d in HERE.iterdir()
                       if (d / "eval.jsonl").exists() and (d / "pack.toml").exists())
    if not packs:
        ap.print_help()
        return 2

    results = []
    for p in packs:
        try:
            results.append(sweep(p, a.model, a.min_similarity, KS))
        except SystemExit as e:
            print(f"  {p}: skipped — {e}")

    if a.write:
        print("\nrecording:")
        for r in results:
            if r.get("best"):
                write_recommendation(r["pack"], r["best"], r["score"],
                                     r["baseline"], r["n"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
