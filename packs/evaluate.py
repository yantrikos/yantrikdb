#!/usr/bin/env python3
"""Measure what a knowledge pack is worth: the efficacy score on a listing.

For each question the model is asked twice against the same prompt:

  baseline  no context
  mounted   the pack is mounted and the top-k recall hits are injected

Both conditions run in the same process against the same model, so the
only difference is the mounted pack.

Scoring is **deterministic string matching**, never an LLM judge. A judge
would be a second unvalidated model sitting between the pack and its
own efficacy number, and the number is the entire product claim.
`expect` is a list of groups; the answer passes when at least one
alternative from every group appears (case-insensitive).

It also runs an unrelated **control set** in both conditions. A pack that
wins its own category by capturing attention and displacing everything
else is a bad pack, and this is the measurement that says so.

Usage:
    python packs/evaluate.py --model qwen3.5:4b
    python packs/evaluate.py --model qwen3.6:27b --pack yantrikdb-engine
    python packs/evaluate.py --model qwen3.6:27b --model granite4:3b
    python packs/evaluate.py --min-similarity 0   # reproduce the attach-harm result

`--model` and `--pack` repeat; models are the outer loop so a large model
stays resident in Ollama instead of being evicted on every question.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from yantrikdb import YantrikDB

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"
DEFAULT_OLLAMA = "http://192.168.4.35:11434"


def resolve_host(explicit: str | None) -> str:
    """Deliberately not plain `OLLAMA_HOST`: that variable is commonly set
    to a *bind* address like `0.0.0.0` for serving, which is not a URL a
    client can dial. Silently inheriting it made every call fail while
    the harness still reported clean zeros."""
    host = explicit or os.environ.get("PACK_EVAL_OLLAMA") or DEFAULT_OLLAMA
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")


OLLAMA = DEFAULT_OLLAMA

SYSTEM = (
    "Answer the question in at most two sentences. State the concrete name, "
    "number or identifier asked for. If reference material is supplied it is "
    "supplementary, not a boundary: when it does not address the question, "
    "ignore it completely and answer from your own knowledge as usual. Only "
    "say you do not know if you genuinely do not know the answer."
)

# Gate injection on SIMILARITY, not on the composite recall score.
#
# Measured against the yantrikdb-engine pack: on-topic queries return a
# top similarity of 0.65-0.79, off-topic queries 0.09-0.45 — a clean gap.
# The composite score overlaps far more (0.83-0.91 vs 0.37-0.69) because
# it folds in importance and recency, which are near-uniform across a
# freshly built pack and so carry no signal about relevance.
MIN_SIMILARITY = 0.55


def ask(
    model: str,
    question: str,
    context: list[str] | None,
    timeout: int = 180,
    num_predict: int = 400,
) -> str:
    """One chat turn. `think:false` matters: these models otherwise spend
    the whole token budget on reasoning and return empty content."""
    if context:
        joined = "\n".join(f"- {c}" for c in context)
        user = (
            f"Reference material retrieved from an attached knowledge pack:\n"
            f"{joined}\n\nUsing that material where relevant, answer:\n{question}"
        )
    else:
        user = question

    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "keep_alive": "20m",
            "options": {"num_predict": num_predict, "temperature": 0.0},
        }
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r).get("message", {}).get("content", "") or ""
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return f"<<error: {e}>>"


def grade(answer: str, expect: list[list[str]]) -> bool:
    low = answer.lower()
    return all(any(alt.lower() in low for alt in group) for group in expect)


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_pack(
    model: str,
    pack_dir: Path,
    top_k: int,
    control: list[dict],
    min_similarity: float = MIN_SIMILARITY,
) -> dict:
    cfg_name = pack_dir.name
    candidates = sorted(DIST.glob(f"{cfg_name}-*.ydbpack"))
    if not candidates:
        raise SystemExit(f"no built pack for {cfg_name} — run: python packs/build.py --all")
    pack_file = candidates[-1]
    questions = load_jsonl(pack_dir / "eval.jsonl")

    # A fresh empty host: this is the qwen-27B scenario — a capable model
    # with no memories of its own, gaining a domain by mounting it.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = YantrikDB(str(Path(td) / "host.db"), 64)
        pack_id = db.mount_pack(str(pack_file))

        rows = []
        for item in questions + control:
            is_control = item in control
            hits = db.recall_text(item["q"], top_k=top_k)
            # Unconditional top-k injection is measurably harmful: with no
            # floor, an unrelated question still receives five pack facts,
            # and the model concludes it must answer only from them. That
            # cost 7 of 12 control questions on qwen3.5:4b.
            context = [
                h["text"]
                for h in hits
                if h.get("scores", {}).get("similarity", 0.0) >= min_similarity
            ]

            base = ask(model, item["q"], None)
            mnt = ask(model, item["q"], context)
            rows.append(
                {
                    "id": item["id"],
                    "control": is_control,
                    "baseline": grade(base, item["expect"]),
                    "mounted": grade(mnt, item["expect"]),
                    "baseline_answer": base.strip()[:400],
                    "mounted_answer": mnt.strip()[:400],
                    "retrieved": len(context),
                    "hits": len(hits),
                }
            )
            tag = "ctl" if is_control else "pack"
            mark = {
                (False, True): "GAIN",
                (True, False): "LOSS",
                (True, True): "both",
                (False, False): "none",
            }[(rows[-1]["baseline"], rows[-1]["mounted"])]
            print(f"  [{tag}] {item['id']:<24} {mark}")

        db.unmount_pack(pack_id)
        db.close()

    # A benchmark whose transport is broken must not report clean zeros.
    # Before this check, every call failing produced a tidy 0/20 baseline
    # and 0/20 mounted — which reads exactly like "the pack did nothing".
    errors = sum(
        1 for r in rows for k in ("baseline_answer", "mounted_answer") if r[k].startswith("<<error")
    )
    if errors:
        sample = next(
            r[k]
            for r in rows
            for k in ("baseline_answer", "mounted_answer")
            if r[k].startswith("<<error")
        )
        raise SystemExit(
            f"{errors}/{len(rows) * 2} model calls failed against {OLLAMA} — "
            f"refusing to report a score.\nfirst error: {sample}"
        )

    pack_rows = [r for r in rows if not r["control"]]
    ctl_rows = [r for r in rows if r["control"]]
    return {
        "model": model,
        "pack": cfg_name,
        "pack_file": pack_file.name,
        "n": len(pack_rows),
        "baseline": sum(r["baseline"] for r in pack_rows),
        "mounted": sum(r["mounted"] for r in pack_rows),
        "control_n": len(ctl_rows),
        "control_baseline": sum(r["baseline"] for r in ctl_rows),
        "control_mounted": sum(r["mounted"] for r in ctl_rows),
        "regressions": [r["id"] for r in pack_rows + ctl_rows if r["baseline"] and not r["mounted"]],
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--pack", action="append", default=[])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument(
        "--min-similarity",
        type=float,
        default=MIN_SIMILARITY,
        help="similarity floor for injecting a retrieved fact; 0 disables gating",
    )
    ap.add_argument("--host", default=None, help=f"Ollama base URL (default {DEFAULT_OLLAMA})")
    ap.add_argument("--out", type=Path, default=HERE / "efficacy.json")
    args = ap.parse_args()

    global OLLAMA
    OLLAMA = resolve_host(args.host)
    print(f"ollama: {OLLAMA}")

    models = args.model or ["qwen3.5:4b"]
    pack_dirs = (
        [HERE / p for p in args.pack]
        if args.pack
        else sorted(p.parent for p in HERE.glob("*/pack.toml"))
    )
    control = load_jsonl(HERE / "control.jsonl")

    results = []
    for model in models:
        for pd in pack_dirs:
            print(f"\n=== {model}  x  {pd.name} ===")
            t0 = time.time()
            res = run_pack(model, pd, args.top_k, control, args.min_similarity)
            res["seconds"] = round(time.time() - t0, 1)
            results.append(res)

    print("\n" + "=" * 78)
    print(
        f"{'model':<22}{'pack':<26}{'pack Q':>9}{'baseline':>10}{'mounted':>9}{'control':>10}"
    )
    print("-" * 78)
    for r in results:
        b = f"{r['baseline']}/{r['n']}"
        m = f"{r['mounted']}/{r['n']}"
        c = f"{r['control_baseline']}->{r['control_mounted']}/{r['control_n']}"
        print(f"{r['model']:<22}{r['pack']:<26}{r['n']:>9}{b:>10}{m:>9}{c:>10}")
    print("=" * 78)
    for r in results:
        if r["regressions"]:
            print(f"REGRESSIONS {r['model']} x {r['pack']}: {', '.join(r['regressions'])}")

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
