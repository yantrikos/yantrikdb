#!/usr/bin/env python3
"""Decide whether a domain is worth building a pack for — BEFORE building it.

Every pack in this repo was chosen by argument. Some of those arguments
were right (mcp-spec: 7/53 -> 44/53) and some were expensively wrong
(java-stdlib went NET NEGATIVE; letterpress needed a compiler and still
showed no measurable teaching). Argument is not a selection method, and
the cost of being wrong is a day of authoring plus a listing that has to
be withdrawn.

This is the gate gpt-5.6-sol specified and nobody built:

    12 probes x 2 local models x 3 conditions

    cold        the model is asked plainly
    challenged  the same question, paraphrased and pushed back on
    rescued     the same question with the AUTHORITATIVE SPAN injected

From those, three numbers:

    C  the share of probes where the model is CONSISTENTLY and
       CONFIDENTLY WRONG — same wrong answer cold and challenged. A
       model that waffles is not carrying a false belief a pack can
       correct; it is carrying no belief at all.
    S  of those confident errors, the share REPAIRED by injecting the
       exact source span. This is the one that matters most: it is a
       pack-free rescue test. If handing the model the truth does not
       fix it, a pack containing that truth will not fix it either.
    A  how many probes test a genuinely ARBITRARY normative constraint —
       a choice with plausible alternatives that could not be derived
       from first principles. Arbitrariness is the strongest predictor
       we have measured: mcp-spec (+37) is almost entirely arbitrary
       protocol decisions, python-stdlib (+2) and java-stdlib (-1) are
       mostly derivable.

    BUILD only if  C >= 25%  and  S >= 50%  and  A >= 3.

A fourth test was added after letterpress, which passed none of the
above because it was never run: every constraint in the domain must be
one a model can JUDGE, not one it must COMPUTE. A 4B handed "body text
must reach 4.5:1 contrast" produces text at 1.00:1 — not because it
lacks the rule but because it cannot evaluate it. Computable constraints
need a compiler, and a domain that needs a compiler is a tool, not a
pack. That question cannot be probed, so the worksheet asks the author
to answer it in writing before anything else.

USAGE

    python packs/probe_gate.py --new candidates/npm-v12.json
    ...fill in the worksheet...
    python packs/probe_gate.py --run candidates/npm-v12.json \\
        --model qwen3.5:4b --model qwen3.5:9b

The worksheet demands a VERBATIM span per probe, not a URL. That is
deliberate: a scout once returned a fabricated CLI flag with a correct
URL attached, and a URL is trivially attachable to an invented fact.
Requiring the quote makes the fabrication visible while the worksheet is
being filled in, which is the cheapest place to catch it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

OLLAMA = "http://localhost:11434"

TEMPLATE = {
    "candidate": "name-of-domain",
    "source": "https://example.com/spec — the primary, authoritative source",
    "source_snapshot": "sha256 of the fetched bytes, or a commit/date",
    "judge_or_compute": (
        "REQUIRED, written before probing. Are this domain's constraints "
        "ones a model can JUDGE from a rule, or ones it must COMPUTE? "
        "If any load-bearing constraint is arithmetic the model cannot "
        "perform (a contrast ratio, a checksum, a layout measurement), "
        "say so — that part needs a compiler and is not packable."
    ),
    "probes": [
        {
            "id": "short-kebab-id",
            "q": "The question, as a practitioner would ask it.",
            "wrong": ["markers of the STALE or WRONG answer you expect"],
            "right": ["markers of the CORRECT answer"],
            "span": (
                "The VERBATIM sentence(s) from the source that establish "
                "the correct answer. Not a summary. Not a URL. If you "
                "cannot paste the quote, you have not verified the fact."
            ),
            "arbitrary": True,
        }
    ],
}


def ask(model: str, prompt: str, system: str = "") -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}]
    payload = json.dumps({
        "model": model, "messages": msgs, "stream": False, "think": False,
        "keep_alive": "20m", "options": {"num_predict": 320, "temperature": 0},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r).get("message", {}).get("content", "") or ""


MARKUP = str.maketrans("", "", "`*_~")

# An abstention is not a false belief. A model that declines is already
# behaving correctly about its own ignorance, and a pack that "fixes" it
# is only supplying information, not correcting an error — which is a
# weaker case for building, because plain retrieval does that too.
HEDGES = ["i don't know", "i do not know", "not sure", "cannot determine",
          "unable to determine", "no information", "not specified",
          "i'm not certain", "i am not certain", "depends on",
          "cannot answer", "insufficient information", "not aware"]


def hit(markers: list[str], answer: str) -> bool:
    """Whole-word for bare alphanumerics, substring otherwise.

    Same rule the efficacy grader uses, including the markdown strip —
    models answer in markdown and "note section" never matched
    "`note` section", which deflated three questions in a held-out run
    before anyone noticed.
    """
    low = answer.lower().translate(MARKUP)
    for m in markers:
        m = m.lower()
        if m.isalnum():
            if re.search(rf"\b{re.escape(m)}\b", low):
                return True
        elif m in low:
            return True
    return False


def run(spec: dict, models: list[str]) -> int:
    probes = spec["probes"]
    print(f"\n{spec['candidate']}  —  {len(probes)} probes x {len(models)} models")
    print(f"source: {spec['source']}")
    print(f"\njudge-or-compute (author's answer, not measured):\n"
          f"  {spec['judge_or_compute']}\n")

    per_model = {}
    for model in models:
        confident_wrong, rescued, rows = [], [], []
        for p in probes:
            cold = ask(model, p["q"])
            challenged = ask(
                model,
                f"{p['q']}\n\nAre you certain? Many sources disagree with "
                f"the common answer. Give your final answer plainly.")
            # WRONG means "did not produce the right answer" — NOT
            # "produced the specific wrong answer I predicted".
            #
            # The first version required hit(wrong) in both conditions,
            # and it scored the mcp-spec control at C=25% when that
            # pack's real baseline is 7/53, i.e. the model gets 87% of
            # them wrong. The gap is every answer that was vague,
            # off-topic or invented in a way I had not anticipated:
            # those matched neither my `wrong` markers nor `right`, and
            # were silently counted as correct. An instrument that
            # cannot see an unanticipated wrong answer will clear every
            # domain, which is the direction that wastes a week.
            #
            # CONFIDENT means it committed to an answer instead of
            # abstaining. A model that says "I don't know" is not
            # carrying a false belief for a pack to correct.
            hedged = hit(HEDGES, challenged)
            is_wrong = (not hit(p["right"], cold)
                        and not hit(p["right"], challenged)
                        and not hedged)
            fixed = None
            if is_wrong:
                confident_wrong.append(p["id"])
                rescued_ans = ask(
                    model,
                    f"Authoritative source says:\n\n\"{p['span']}\"\n\n"
                    f"Given only that, answer: {p['q']}")
                fixed = hit(p["right"], rescued_ans) and not hit(
                    p["wrong"], rescued_ans)
                if fixed:
                    rescued.append(p["id"])
            rows.append((p["id"], is_wrong, fixed, p.get("arbitrary", False)))

        n = len(probes)
        C = len(confident_wrong) / n
        S = (len(rescued) / len(confident_wrong)) if confident_wrong else 0.0
        per_model[model] = (C, S, confident_wrong, rescued)
        print(f"  {model}")
        for pid, wrong, fixed, arb in rows:
            mark = ("confidently wrong" if wrong else "ok")
            if wrong:
                mark += ", rescued by span" if fixed else ", NOT rescued"
            print(f"      {'A' if arb else ' '} {pid:<34} {mark}")
        print(f"      C = {C:.0%} confidently wrong   "
              f"S = {S:.0%} of those repaired by the span")

    A = sum(1 for p in probes if p.get("arbitrary"))
    # Both models must clear the bar. One model failing is a model
    # quirk; the claim is about a class of model, not a lucky one.
    C_min = min(v[0] for v in per_model.values())
    S_min = min(v[1] for v in per_model.values())
    ok_c, ok_s, ok_a = C_min >= 0.25, S_min >= 0.50, A >= 3

    print(f"\n  VERDICT for {spec['candidate']}")
    print(f"    C >= 25%   {C_min:>5.0%}   {'pass' if ok_c else 'FAIL — models are not confidently wrong; there is no false belief to correct'}")
    print(f"    S >= 50%   {S_min:>5.0%}   {'pass' if ok_s else 'FAIL — the source text does not repair the error, so a pack carrying it will not either'}")
    print(f"    A >= 3     {A:>5}   {'pass' if ok_a else 'FAIL — too few arbitrary constraints; the model can derive this'}")
    verdict = ok_c and ok_s and ok_a
    print(f"\n    {'BUILD' if verdict else 'DO NOT BUILD'}")
    if verdict:
        print("    (and re-read the judge-or-compute answer above before "
              "starting — a computable domain still needs a compiler)")
    return 0 if verdict else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=Path, help="write a blank worksheet")
    ap.add_argument("--run", type=Path, help="run a filled worksheet")
    ap.add_argument("--model", action="append", default=[])
    args = ap.parse_args()

    if args.new:
        args.new.parent.mkdir(parents=True, exist_ok=True)
        args.new.write_text(json.dumps(TEMPLATE, indent=2), encoding="utf-8")
        print(f"worksheet -> {args.new}\n"
              f"Fill in 12 probes. Each needs a VERBATIM span from the "
              f"source; a URL is not enough.")
        return 0

    if not args.run:
        ap.print_help()
        return 2

    spec = json.loads(args.run.read_text(encoding="utf-8"))
    missing = [p["id"] for p in spec["probes"]
               if not p.get("span") or p["span"].startswith("The VERBATIM")]
    if missing:
        print(f"probes without a verbatim span: {missing}", file=sys.stderr)
        return 2
    return run(spec, args.model or ["qwen3.5:4b", "qwen3.5:9b"])


if __name__ == "__main__":
    raise SystemExit(main())
