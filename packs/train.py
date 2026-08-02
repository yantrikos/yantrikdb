#!/usr/bin/env python3
"""Author a pack the way a model is trained: split, step, early-stop, test once.

Pack authoring has been a single forward pass — read the source, write
records, measure, ship — and every discipline that makes model training
trustworthy was missing. This adds them, and enforces them rather than
relying on the author to remember:

  SPLIT     questions are divided train / validation / test by a seeded
            shuffle. You may look at TRAIN as much as you like. You see
            validation only as a score and a failure CLASS, never its
            contents. Test is sealed.

  STEP      one epoch: measure validation, attribute every failure, and
            print the prioritised work list. The failure class dictates
            the fix — retrieval failures are not repaired by writing
            more records, and doing so is the commonest wasted day.

  CURVE     every epoch is appended to a log, so improvement is visible
            as a trajectory rather than a single number, and a plateau
            is a stop signal rather than a mystery.

  OVERFIT   the train/validation gap is computed every epoch. letterpress
            scored 17/40 on questions its author wrote and 3/24 on an
            independent set. That gap was the whole story and nothing in
            the process surfaced it until an outside reviewer did.

  TEST      the sealed set runs ONCE. The commitment hash is written at
            init and checked at test time, so a test set quietly edited
            to be kinder is detectable. Every run is appended to a
            ledger; a second run is refused without --i-know.

USAGE
    python packs/train.py --pack mcp-spec --init
    python packs/train.py --pack mcp-spec --step
    ... author against the work list, rebuild, step again ...
    python packs/train.py --pack mcp-spec --curve
    python packs/train.py --pack mcp-spec --test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import diagnose  # noqa: E402
import evaluate  # noqa: E402
from yantrikdb import YantrikDB  # noqa: E402

RUNS = HERE / "training"


def commit(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


def paths(pack: str) -> tuple[Path, Path, Path]:
    d = RUNS / pack
    return d, d / "split.json", d / "log.jsonl"


def do_init(pack: str, seed: int, ratios: tuple[float, float]) -> int:
    d, split_f, _ = paths(pack)
    if split_f.exists():
        print(f"{split_f} exists. Delete it to re-split — but note that "
              f"re-splitting after authoring INVALIDATES the test set, "
              f"because records were written while its questions were "
              f"in train.", file=sys.stderr)
        return 2
    qs = evaluate.load_jsonl(HERE / pack / "eval.jsonl")
    ids = [q["id"] for q in qs]
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    split = {
        "pack": pack,
        "seed": seed,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train": ids[:n_tr],
        "val": ids[n_tr:n_tr + n_va],
        "test": ids[n_tr + n_va:],
    }
    # The commitment is what makes a later edit detectable. It covers
    # the test IDs only; the questions themselves stay in eval.jsonl
    # where they are already version-controlled.
    split["test_commitment"] = commit(split["test"])
    d.mkdir(parents=True, exist_ok=True)
    split_f.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"split {n} questions -> train {len(split['train'])}, "
          f"val {len(split['val'])}, test {len(split['test'])} (sealed "
          f"{split['test_commitment']})\n  {split_f}")
    print("\n  You may read TRAIN questions. Do not read val or test — "
          "the loop reports their score and failure classes, which is "
          "all you need to act on.")
    return 0


def measure(pack: str, model: str, ids: set[str], top_k: int,
            floor: float) -> tuple[int, int, Counter, list[tuple[str, str]]]:
    """Score a subset and attribute every failure. Returns (pass, n, classes, rows)."""
    qs = [q for q in evaluate.load_jsonl(HERE / pack / "eval.jsonl")
          if q["id"] in ids]
    corpus_md = (HERE / pack / "corpus.md").read_text(encoding="utf-8")
    import re
    corpus = [c for c in re.split(r"^## ", corpus_md, flags=re.M) if c.strip()]

    passed, classes, rows = 0, Counter(), []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = YantrikDB(str(Path(td) / "h.db"), 64)
        db.mount_pack(str(diagnose.newest_pack(pack)))
        for q in qs:
            hits = [h for h in db.recall(q["q"], top_k=top_k)
                    if h.get("score", 1.0) >= floor]
            texts = [h["text"] for h in hits]
            ans = evaluate.ask(model, q["q"], texts)
            if evaluate.grade(ans, q["expect"]):
                passed += 1
                continue
            joined = "\n\n".join(texts)
            if joined and diagnose.answer_present(q["expect"], joined):
                cls = "MISS_APPLICATION"
            elif any(diagnose.answer_present(q["expect"], r) for r in corpus):
                cls = "MISS_RETRIEVAL"
            else:
                cls = "MISS_CONTENT"
            classes[cls] += 1
            rows.append((cls, q["id"]))
    return passed, len(qs), classes, rows


def do_step(pack: str, model: str, top_k: int, floor: float) -> int:
    d, split_f, log_f = paths(pack)
    if not split_f.exists():
        print("no split — run --init first", file=sys.stderr)
        return 2
    split = json.loads(split_f.read_text(encoding="utf-8"))

    tr_pass, tr_n, _, _ = measure(pack, model, set(split["train"]), top_k, floor)
    va_pass, va_n, classes, rows = measure(
        pack, model, set(split["val"]), top_k, floor)

    epoch = sum(1 for _ in log_f.open(encoding="utf-8")) + 1 if log_f.exists() else 1
    tr_r = tr_pass / tr_n if tr_n else 0
    va_r = va_pass / va_n if va_n else 0
    entry = {
        "epoch": epoch,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "train": [tr_pass, tr_n], "val": [va_pass, va_n],
        "gap": round(tr_r - va_r, 3),
        "classes": dict(classes),
        "corpus_bytes": (HERE / pack / "corpus.md").stat().st_size,
    }
    with log_f.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"\nepoch {epoch}  {pack}  {model}")
    print(f"  train {tr_pass}/{tr_n} ({tr_r:.0%})   "
          f"validation {va_pass}/{va_n} ({va_r:.0%})   gap {tr_r - va_r:+.0%}")

    if tr_r - va_r > 0.25:
        print("\n  OVERFIT WARNING. The pack does markedly better on the "
              "questions its author has read. letterpress showed exactly "
              "this shape — 17/40 self-authored against 3/24 independent — "
              "and it meant the records had been written toward the "
              "questions rather than the domain.")

    prev = None
    if log_f.exists():
        entries = [json.loads(l) for l in log_f.read_text(encoding="utf-8").splitlines()]
        if len(entries) >= 3:
            recent = [e["val"][0] / e["val"][1] for e in entries[-3:]]
            if max(recent) - min(recent) < 0.02:
                print("\n  PLATEAU. Validation has not moved in three "
                      "epochs. More of the same authoring will not help; "
                      "either the remaining failures need a different "
                      "class of fix, or this pack is finished.")
            prev = entries[-2]["val"][0] / entries[-2]["val"][1] if len(entries) > 1 else None

    if prev is not None:
        print(f"  previous validation {prev:.0%}  ->  {va_r:.0%}")

    if not classes:
        print("\n  no validation failures left")
        return 0

    print("\n  work list, highest leverage first:")
    order = [("MISS_RETRIEVAL", "rewrite headings and opening lines, split or "
                                "de-duplicate. Do NOT author records — the "
                                "content already exists and is not surfacing."),
             ("MISS_CONTENT", "author records. The only class where writing "
                              "more is the correct response."),
             ("MISS_APPLICATION", "rewrite the record that was retrieved. It "
                                  "reached the model and did not land, so "
                                  "adding beside it makes the page longer "
                                  "and no better.")]
    for cls, advice in order:
        n = classes.get(cls, 0)
        if not n:
            continue
        ids = [qid for c, qid in rows if c == cls]
        print(f"    {cls} x{n}: {advice}")
        print(f"      {', '.join(ids)}")
    return 0


def do_curve(pack: str) -> int:
    _, _, log_f = paths(pack)
    if not log_f.exists():
        print("no epochs logged yet", file=sys.stderr)
        return 2
    entries = [json.loads(l) for l in log_f.read_text(encoding="utf-8").splitlines()]
    print(f"\n{pack} — {len(entries)} epochs\n")
    print(f"  {'ep':>3} {'train':>9} {'val':>9} {'gap':>6} {'corpus':>8}   classes")
    for e in entries:
        tr = f"{e['train'][0]}/{e['train'][1]}"
        va = f"{e['val'][0]}/{e['val'][1]}"
        cls = " ".join(f"{k.split('_')[1][:4].lower()}={v}"
                       for k, v in sorted(e["classes"].items()))
        print(f"  {e['epoch']:>3} {tr:>9} {va:>9} {e['gap']:>+6.0%} "
              f"{e['corpus_bytes'] // 1024:>6}KB   {cls}")
    return 0


def do_test(pack: str, model: str, top_k: int, floor: float, force: bool) -> int:
    d, split_f, _ = paths(pack)
    split = json.loads(split_f.read_text(encoding="utf-8"))
    ledger = d / "test-runs.jsonl"

    if commit(split["test"]) != split["test_commitment"]:
        print("TEST SET HAS CHANGED since init. The commitment does not "
              "match. A test set edited after authoring measures nothing.",
              file=sys.stderr)
        return 2
    if ledger.exists() and not force:
        runs = ledger.read_text(encoding="utf-8").strip().splitlines()
        print(f"the sealed test has already been run {len(runs)} time(s):",
              file=sys.stderr)
        for r in runs:
            print(f"  {r}", file=sys.stderr)
        print("\nRunning it again turns it into a validation set — every "
              "look leaks information into the next authoring decision. "
              "Pass --i-know if you genuinely need another number and will "
              "report all of them.", file=sys.stderr)
        return 2

    p, n, classes, _ = measure(pack, model, set(split["test"]), top_k, floor)
    entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "model": model, "test": [p, n], "classes": dict(classes)}
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\n  SEALED TEST  {pack}  {model}:  {p}/{n} ({p / n:.0%})")
    print(f"  recorded in {ledger.name}. This is the number to report.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--step", action="store_true")
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--i-know", action="store_true",
                    help="re-run an already-used sealed test")
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-similarity", type=float, default=0.55)
    a = ap.parse_args()

    evaluate.OLLAMA = evaluate.resolve_host(None)
    if a.init:
        return do_init(a.pack, a.seed, (0.5, 0.25))
    if a.step:
        return do_step(a.pack, a.model, a.top_k, a.min_similarity)
    if a.curve:
        return do_curve(a.pack)
    if a.test:
        return do_test(a.pack, a.model, a.top_k, a.min_similarity, a.i_know)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
