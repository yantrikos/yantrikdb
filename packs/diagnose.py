#!/usr/bin/env python3
"""Why did each question fail? — the gradient signal for pack authoring.

`evaluate.py` reports 40/53 and stops. That number says a pack could be
better and nothing about what to change, so authoring proceeds by
intuition: add more records, reword, hope. That is a forward pass with
no backward pass.

A failure is one of three things, and they take OPPOSITE fixes:

  MISS_RETRIEVAL  the answer IS in the corpus, and nothing retrieved it.
                  Fix the record's heading and opening vocabulary, split
                  a record that covers two topics, or reduce the density
                  of near-duplicate siblings competing for the query.
                  Do NOT write more content — the content exists.

  MISS_CONTENT    no record contains the answer at all. Fix by authoring
                  a record. This is the only class where "write more" is
                  the right response.

  MISS_APPLICATION the right record WAS retrieved and injected, and the
                  model still answered wrong. The content and retrieval
                  are both fine; the record is unclear, buries the fact
                  in prose, or states it in a form the model cannot act
                  on. Rewriting beats adding, and adding actively hurts.

Getting this backwards is expensive. Six letterpress photo questions
scored zero and looked like missing content; the records were present
and retrievable, and the real fault was elsewhere. Without attribution
the obvious response is to write more records, which would have made the
pack larger, slower and no better.

There is also a class that is NOT the pack's fault:

  GRADER          the answer is substantively right and the expectation
                  markers missed it. Reported separately and never
                  counted as a pack failure, because "fix" here means
                  fixing the test, and a pack tuned against a broken
                  test is worse than one that fails honestly.

USAGE
    python packs/diagnose.py --pack mcp-spec --model qwen3.5:4b
    python packs/diagnose.py --pack mcp-spec --only-failures
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate  # noqa: E402
from yantrikdb import YantrikDB  # noqa: E402


def newest_pack(name: str) -> Path:
    builds = sorted((HERE / "dist").glob(f"{name}-*.ydbpack"))
    if not builds:
        raise SystemExit(f"no build for {name!r} — run build.py first")
    return builds[-1]


def answer_present(markers: list[list[str]], text: str) -> bool:
    """Does this text contain an alternative from EVERY expected group?

    The same test the grader applies to a model's answer, applied
    instead to a corpus record — which is what makes "the answer is in
    the corpus" a measurement rather than an opinion.
    """
    return evaluate.grade(text, markers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--model", default="qwen3.5:4b")
    # Default to the pack's OWN declared retrieval settings, not to a
    # number chosen here.
    #
    # These were hardcoded to top_k=5 / 0.55, which meant this tool
    # diagnosed a configuration that nobody ships. mcp-spec declares
    # top_k=16 and a 0.62 floor and measures 52/53 through evaluate.py;
    # under the hardcoded defaults it came out 43/53, and the ten
    # "failures" were a work list for a pack that does not exist. Eight
    # of them were labelled MISS_RETRIEVAL, which is exactly what
    # starving retrieval of two thirds of its budget produces — so the
    # tool was not merely wrong, it was wrong in the direction that
    # invents the most plausible-looking work.
    #
    # This is the third tool in this directory to invent its own gating
    # and disagree with the trusted path. The rule now: a measurement
    # tool reads the pack's settings, and any override is explicit.
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-similarity", type=float, default=None)
    ap.add_argument("--only-failures", action="store_true")
    ap.add_argument("--host", default=None)
    args = ap.parse_args()

    evaluate.OLLAMA = evaluate.resolve_host(args.host)
    questions = evaluate.load_jsonl(HERE / args.pack / "eval.jsonl")
    pack_file = newest_pack(args.pack)

    top_k = (args.top_k if args.top_k is not None
             else evaluate.recommended_top_k(args.pack, 16))
    floor = (args.min_similarity if args.min_similarity is not None
             else evaluate.pack_setting(args.pack, "recommended_min_similarity", 0.55))
    print(f"  retrieval: top_k={top_k} floor={floor}"
          f"{' (declared by the pack)' if args.top_k is None else ' (overridden)'}")

    # ignore_cleanup_errors: on Windows the engine still holds the
    # database file when the context manager unwinds, and a
    # PermissionError at cleanup would mask a run that already
    # succeeded and printed its results.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = YantrikDB(str(Path(td) / "host.db"), 64)
        db.mount_pack(str(pack_file))

        # Every record, read from the SOURCE rather than coaxed out of
        # similarity search. "Is the answer anywhere in the corpus" has
        # to be independent of retrieval, or the two classes this tool
        # exists to separate collapse back into one.
        corpus_md = (HERE / args.pack / "corpus.md").read_text(encoding="utf-8")
        corpus = [c for c in re.split(r"^## ", corpus_md, flags=re.M) if c.strip()]

        verdicts, rows = Counter(), []
        for q in questions:
            hits = [h for h in db.recall_text(q["q"], top_k=top_k)
                    if h.get("scores", {}).get("similarity", 0.0)
                    >= floor]

            texts = [h["text"] for h in hits]
            ctx = "\n\n".join(texts)
            answer = evaluate.ask(args.model, q["q"], texts)
            passed = evaluate.grade(answer, q["expect"])

            in_retrieved = bool(ctx) and answer_present(q["expect"], ctx)
            if passed:
                verdict = "PASS"
            elif in_retrieved:
                verdict = "MISS_APPLICATION"
            elif any(answer_present(q["expect"], rec) for rec in corpus):
                verdict = "MISS_RETRIEVAL"
            else:
                verdict = "MISS_CONTENT"
            verdicts[verdict] += 1
            rows.append((verdict, q["id"], len(hits), answer[:100]))

        print(f"\n{args.pack} on {args.model}  —  {len(questions)} questions\n")
        for verdict, qid, nhits, ans in rows:
            if args.only_failures and verdict == "PASS":
                continue
            print(f"  {verdict:<17} {qid:<30} {nhits} hits")
            if verdict != "PASS":
                print(f"      {ans!r}")

        total = len(questions)
        print(f"\n  {'PASS':<17} {verdicts['PASS']:>3} / {total}")
        for k in ("MISS_APPLICATION", "MISS_RETRIEVAL", "MISS_CONTENT"):
            print(f"  {k:<17} {verdicts[k]:>3}")

        print("\n  what to do with each:")
        if verdicts["MISS_CONTENT"]:
            print(f"    MISS_CONTENT   ({verdicts['MISS_CONTENT']}) author records — "
                  f"the only class where writing more is the fix")
        if verdicts["MISS_RETRIEVAL"]:
            print(f"    MISS_RETRIEVAL ({verdicts['MISS_RETRIEVAL']}) rewrite headings "
                  f"and opening lines; split or de-duplicate — content exists")
        if verdicts["MISS_APPLICATION"]:
            print(f"    MISS_APPLICATION ({verdicts['MISS_APPLICATION']}) the record "
                  f"reached the model and did not land; rewrite it, do not add")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
