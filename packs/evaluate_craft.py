#!/usr/bin/env python3
"""The craft measurement: does compiled craft beat the rulebook-in-context?

Three arms, all on the same frozen weights through serve_compiled.py,
all graded by the same deterministic checker that gated the training
data. The briefs are the SEALED holdout from wp_theme_checks — site
types the training set never contains, crossed with mood/type/palette
combinations it never used.

  bare       the brief alone. What the model does by default.
  rulebook   the brief plus the full constitution in context. This is
             the arm the YDS result predicts leaks: a model handed 23
             rules must re-apply them on every token and mostly doesn't.
  compiled   the adapter, brief alone. The claim under test.

The result is a per-check compliance table, because "which rules leak
in context but hold in weights" is worth more than any single number.

    .venv-compile/Scripts/python packs/serve_compiled.py \
        --adapter wordpress-theme-craft --tag v1 &
    python packs/evaluate_craft.py --host http://127.0.0.1:11555 \
        --adapter wordpress-theme-craft
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from compile import craft_module, craft_system  # noqa: E402


def ask(host: str, model: str, user: str, num_predict: int = 4000,
        system: str | None = None) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system or ""},
                     {"role": "user", "content": user}],
        "options": {"num_predict": num_predict, "temperature": 0.0},
    }).encode()
    req = urllib.request.Request(f"{host}/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r).get("message", {}).get("content", "") or ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="http://127.0.0.1:11555")
    ap.add_argument("--adapter", default="wordpress-theme-craft",
                    help="model name serve_compiled serves the adapter under")
    ap.add_argument("--pack", default="wordpress-theme")
    ap.add_argument("--out", type=Path, default=HERE / "efficacy-craft.json")
    ap.add_argument("--corpus", action="store_true",
                    help="add the two-tier arm: compiled adapter + the "
                         "pack's corpus retrieved through the engine")
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--min-similarity", type=float, default=0.6)
    a = ap.parse_args()

    W = craft_module(a.pack)
    constitution = (HERE / a.pack / "constitution.md").read_text(encoding="utf-8")
    briefs = W.holdout_briefs()
    arms = [
        ("bare", "base", lambda b: b),
        ("rulebook", "base",
         lambda b: f"{constitution}\n\n---\n\nApply every rule above.\n\nBRIEF: {b}\n\n"
                   f"Reply with the complete {W.ARTIFACT} only — no prose."),
        ("compiled", a.adapter, lambda b: b),
    ]

    # The two-tier arm: rules in weights, FACTS retrieved from the pack's
    # own corpus through the engine. This is the configuration the
    # carrier split predicts is the only one that scales — compiling the
    # constitution frees the whole context window for the corpus, and
    # neither ceiling (completion tax, blast radius) binds. It is also
    # the attach-harm question at the craft layer: does putting text
    # back in front of a compiled model disturb the compiled behaviour?
    if a.corpus:
        import tempfile

        from yantrikdb import YantrikDB
        dist = sorted((HERE / "dist").glob(f"{a.pack}-*.ydbpack"))
        if not dist:
            print(f"no built pack for {a.pack} — run build.py", file=sys.stderr)
            return 2
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        db = YantrikDB(str(Path(tmp.name) / "host.db"), 64)
        db.mount_pack(str(dist[-1]))

        def with_corpus(b):
            hits = [h["text"] for h in db.recall_text(b, top_k=a.top_k)
                    if h.get("scores", {}).get("similarity", 0.0) >= a.min_similarity]
            if not hits:
                return b
            joined = "\n".join(f"- {h}" for h in hits)
            return (f"Reference material from the attached knowledge pack:\n{joined}\n\n"
                    f"BRIEF: {b}\n\nReply with the complete {W.ARTIFACT} only — no prose.")

        arms.append(("compiled+corpus", a.adapter, with_corpus))

    results = {arm: [] for arm, _, _ in arms}
    for i, brief in enumerate(briefs, 1):
        for arm, model, wrap in arms:
            raw = ask(a.host, model, wrap(brief), system=craft_system(W))
            p, n, res = W.grade_text(raw)
            results[arm].append({
                "brief": brief, "passed": p, "total": n,
                "parsed": res is not None,
                "checks": {cid: ok for cid, (ok, _) in res.items()} if res else {},
                "answer_head": raw[:200],
            })
            print(f"  [{i:>2}/{len(briefs)}] {arm:<9} {p:>2}/{n}", flush=True)

    ncheck = W.grade_text('{}')[1]
    print("\n" + "=" * 72)
    print(f"{'arm':<10}{'mean':>8}{'full-pass':>11}{'parse-fail':>12}   n={len(briefs)} sealed briefs")
    print("-" * 72)
    for arm, rows in results.items():
        mean = sum(r["passed"] for r in rows) / len(rows)
        full = sum(1 for r in rows if r["passed"] == ncheck)
        nop = sum(1 for r in rows if not r["parsed"])
        print(f"{arm:<10}{mean:>7.1f}/{ncheck}{full:>8}/{len(rows)}{nop:>10}")
    print("=" * 72)

    print(f"\nper-check pass rate ({len(briefs)} briefs):")
    print("  " + f"{'check':<26}" + "".join(f"{nm:>10}" for nm in results))
    for cid in ({} if not any(r["checks"] for rows in results.values() for r in rows)
                else next(r["checks"] for r in results["compiled"] if r["checks"])):
        row = [sum(1 for r in rows if r["checks"].get(cid))
               for rows in results.values()]
        cells = "".join(f"{v:>10}" for v in row)
        b, rb, cp = row[0], row[1], row[2]
        mark = "  <- holds in weights, leaks in context" if cp > rb and cp > b else ""
        print(f"  {cid:<26}{cells}{mark}")

    a.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
