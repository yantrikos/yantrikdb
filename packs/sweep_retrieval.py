#!/usr/bin/env python3
"""Joint sweep of the two retrieval knobs a pack declares: top_k AND floor.

`sweep_topk.py` sweeps k at whatever floor it was handed, which is only
valid if the two are independent. They are not. Diagnosing the last
failures in yantrikdb-engine showed why:

  member-ladder    correct record ranks 3rd of 45, similarity 0.543,
                   pack floor 0.62  -> thrown away
  write-admission  correct record rank 12, similarity 0.550, while
                   "the schema version is 38" was injected at 0.662

Retrieval had already done its job — rank 3 is not a retrieval failure.
An absolute floor tuned to keep noise out of the CONTROL questions was
also cutting the correct record out of the pack's own questions, and no
value of k can rescue a record the floor has already removed. Sweeping
either knob alone finds a local optimum on the wrong axis.

There is a second reason to re-run this now. Every pack's declared
settings were swept BEFORE the grader was corrected, so each sweep
maximised a score that mis-graded snake_case answers and any expectation
written as a stem. Those settings are conclusions drawn from a broken
instrument and have to be re-derived, exactly like the pack verdicts
were.

Cost control: the model's answer is a pure function of the question and
the injected context, at temperature 0. Many (floor, k) pairs produce an
IDENTICAL injected set, so answers are memoised on that signature. This
is exact, not an approximation — it does not change a single grade, it
only skips asking the same thing twice. In practice it cuts a 16-cell
grid to roughly the cost of three.

    python packs/sweep_retrieval.py --pack yantrikdb-engine
    python packs/sweep_retrieval.py --all --write
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

FLOORS = [0.45, 0.50, 0.55, 0.60, 0.65]
TOP_KS = [4, 6, 8, 12, 16]


def ask_cached(cache, model, question, texts):
    """One answer per (question, exact injected context). Temperature is
    0, so the same inputs give the same output and re-asking is pure
    waste."""
    key = (question, tuple(texts))
    if key not in cache:
        cache[key] = evaluate.ask(model, question, texts)
    return cache[key]


def score(db, questions, model, k, floor, cache):
    ok = 0
    for q in questions:
        hits = [h for h in db.recall_text(q["q"], top_k=k)
                if h.get("scores", {}).get("similarity", 0.0) >= floor]
        texts = [h["text"] for h in hits]
        ok += evaluate.grade(ask_cached(cache, model, q["q"], texts),
                             q["expect"], q.get("reject"))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--write", action="store_true",
                    help="record the winner in the pack's pack.toml")
    ap.add_argument("--host", default=None)
    args = ap.parse_args()

    evaluate.OLLAMA = evaluate.resolve_host(args.host)
    # A pack.toml alone is not enough — a directory can be a pack source
    # with no eval set yet (einstein-method), and crashing on it throws
    # away every result already computed in the same run.
    packs = ([p.name for p in sorted(HERE.iterdir())
              if (p / "pack.toml").exists() and (p / "eval.jsonl").exists()]
             if args.all else [args.pack])
    ctl = evaluate.load_jsonl(HERE / "control.jsonl")

    for pack in packs:
        qs = evaluate.load_jsonl(HERE / pack / "eval.jsonl")
        try:
            pack_file = sorted((HERE / "dist").glob(f"{pack}-*.ydbpack"))[-1]
        except IndexError:
            print(f"\n{pack}: not built — skipping")
            continue

        cur_k = evaluate.recommended_top_k(pack, 16)
        cur_f = evaluate.pack_setting(pack, "recommended_min_similarity", 0.55)
        print(f"\n{'='*74}\n{pack}  —  {len(qs)} questions, {len(ctl)} controls, "
              f"{args.model}\n  currently declared: top_k={cur_k} floor={cur_f}")

        cache: dict = {}
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db = YantrikDB(str(Path(td) / "host.db"), 64)
            db.mount_pack(str(pack_file))

            # Controls with NOTHING injected. This is the number the
            # control set has to be compared against: a control question
            # the model gets wrong on its own is not attach-harm, and
            # scoring it as such would blame the pack for the model.
            ctl_base = sum(evaluate.grade(evaluate.ask(args.model, c["q"], None),
                                          c["expect"], c.get("reject")) for c in ctl)

            print(f"  control baseline (nothing mounted): {ctl_base}/{len(ctl)}\n")
            print(f"  {'floor':<7}" + "".join(f"k={k:<7}" for k in TOP_KS))

            results = []
            for f in FLOORS:
                row = f"  {f:<7}"
                for k in TOP_KS:
                    s = score(db, qs, args.model, k, f, cache)
                    c = score(db, ctl, args.model, k, f, cache)
                    harmed = c < ctl_base
                    row += f"{s:>2}/{len(qs)}{'!' if harmed else ' ':<1}   "
                    results.append((s, c, f, k, harmed))
                print(row)

            print(f"\n  '!' marks a config that costs a control question "
                  f"(control fell below {ctl_base}/{len(ctl)})")

            # Winner: highest score among configs that harm no control.
            # Ties break toward the SMALLER k and the HIGHER floor —
            # less injected context for the same answer is strictly
            # better, and it is the direction that resists attach-harm
            # on questions the control set does not happen to cover.
            clean = [r for r in results if not r[4]]
            if not clean:
                print("  every config costs a control — not writing anything")
                continue
            best = max(clean, key=lambda r: (r[0], -r[3], r[2]))
            s, c, f, k, _ = best
            cur = next((r for r in results if r[2] == cur_f and r[3] == cur_k), None)
            print(f"\n  winner: top_k={k} floor={f}  ->  {s}/{len(qs)}, controls {c}/{len(ctl)}")
            if cur:
                delta = s - cur[0]
                print(f"  currently declared config scores {cur[0]}/{len(qs)}"
                      f"  ({delta:+d} available)")

            if args.write and (k != cur_k or f != cur_f):
                toml_path = HERE / pack / "pack.toml"
                text = toml_path.read_text(encoding="utf-8")
                import re
                text = re.sub(r"^recommended_top_k\s*=.*$",
                              f"recommended_top_k = {k}", text, flags=re.M)
                text = re.sub(r"^recommended_min_similarity\s*=.*$",
                              f"recommended_min_similarity = {f}", text, flags=re.M)
                toml_path.write_text(text, encoding="utf-8")
                print(f"  wrote top_k={k} floor={f} to {pack}/pack.toml")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
