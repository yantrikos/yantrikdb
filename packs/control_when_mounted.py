#!/usr/bin/env python3
"""Attach-harm for the weights tier: what does a mounted capability cost?

The knowledge tier has had this gate since the beginning — a pack that
wins its own category by capturing attention and wrecking everything
else is a bad pack, and a control regression fails certification
outright. A compiled capability needs the same gate for a stronger
reason: retrieval is filtered per query by a similarity floor, so a
mounted pack contributes nothing to an unrelated question. An adapter
has no floor. While it is mounted it is present on every token of every
answer.

Same 31-question control set the pack line already uses — twelve
unrelated, nineteen NEAR-DOMAIN, which is the distinction that matters:
a capability does not degrade a question about Mars, it degrades a
question about its own neighbourhood.

    python packs/control_when_mounted.py --host http://127.0.0.1:11556 \
        --cap motion-craft --cap wordpress-theme
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate  # noqa: E402


def post(host: str, path: str, payload: dict, timeout: int = 900):
    req = urllib.request.Request(host + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ask(host: str, question: str) -> str:
    r = post(host, "/api/chat", {
        "model": "current",
        "messages": [{"role": "system", "content": evaluate.SYSTEM},
                     {"role": "user", "content": question}],
        "options": {"num_predict": 200, "temperature": 0.0}})
    return r.get("message", {}).get("content", "") or ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="http://127.0.0.1:11556")
    ap.add_argument("--cap", action="append", default=[])
    ap.add_argument("--out", type=Path, default=HERE / "control-when-mounted.json")
    a = ap.parse_args()

    control = [json.loads(l) for l in
               (HERE / "control.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip()]

    post(a.host, "/api/caps/unmount", {})
    base = {}
    for item in control:
        ans = ask(a.host, item["q"])
        base[item["id"]] = evaluate.grade(ans, item["expect"], item.get("reject"))
        print(f"  base      {item['id']:<26} {'ok' if base[item['id']] else 'MISS'}",
              flush=True)
    base_score = sum(base.values())

    results = {"base": {"score": base_score, "n": len(control)}}
    for cap in a.cap:
        r = post(a.host, "/api/caps/mount", {"pack": cap})
        if "error" in r:
            print(f"  cannot mount {cap}: {r['error']}", file=sys.stderr)
            continue
        lost, gained = [], []
        score = 0
        for item in control:
            ans = ask(a.host, item["q"])
            ok = evaluate.grade(ans, item["expect"], item.get("reject"))
            score += ok
            if base[item["id"]] and not ok:
                lost.append((item["id"], ans.strip()[:160]))
            elif ok and not base[item["id"]]:
                gained.append(item["id"])
            print(f"  {cap:<9} {item['id']:<26} {'ok' if ok else 'MISS'}", flush=True)
        results[cap] = {"score": score, "n": len(control),
                        "lost": [i for i, _ in lost], "gained": gained}
        print(f"\n  {cap}: {base_score} -> {score} / {len(control)}"
              f"   {'PASS' if score >= base_score else 'REGRESSION'}")
        for i, ans in lost:
            print(f"    lost [{i}]: {ans}")
        post(a.host, "/api/caps/unmount", {})

    print("\n" + "=" * 64)
    print(f"{'arm':<20}{'control':>10}   verdict")
    print("-" * 64)
    print(f"{'base (unmounted)':<20}{base_score:>7}/{len(control)}")
    for cap in a.cap:
        if cap not in results:
            continue
        v = results[cap]
        verdict = "PASS" if v["score"] >= v["n"] * 0 + base_score else "REGRESSION"
        print(f"{cap:<20}{v['score']:>7}/{v['n']}   {verdict}")
    print("=" * 64)
    a.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
