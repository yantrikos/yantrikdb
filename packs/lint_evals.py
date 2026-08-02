#!/usr/bin/env python3
"""Lint eval sets for graders that cannot fail.

The efficacy number on a listing is the entire product claim, and it is
produced by deterministic string matching. That makes the grader itself
the most load-bearing code in the pack pipeline — and a grader bug is
invisible in exactly the way that matters: it does not crash, it just
scores everything correct, and the pack looks better than it is.

Two real bugs found by hand before this file existed:

  - c-safety/strtol-errors listed "s" as an alternative. Nearly every
    English sentence contains an "s", so that group always matched and
    a three-part question was really a two-part question.
  - react-craft/react19-removed asked for two APIs and gave both groups
    the same five alternatives, so one match satisfied both. A question
    that asks for two things must have disjoint groups, or it is asking
    for one.

Both scored the pack *higher* than the truth. That is the direction
grader bugs usually fail in, because a grader that is too strict gets
noticed the first time a correct answer is marked wrong, and a grader
that is too lenient never gets noticed at all.

The too-strict direction did eventually appear, and it is worth its own
check: `evaluate.py` matches on WORD BOUNDARIES (the fix for the "s"
bug above), so a truncated stem can never match its own inflection.
An expectation of "cach" scores a model that answered "cache the tool
list" as wrong. That is a false negative — cheap to notice on one
question, and invisible when it silently deflates a whole listing.
`stem_suspects` below catches it before the number is published.

Usage:
    python packs/lint_evals.py            # every pack
    python packs/lint_evals.py c-safety
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# An alternative shorter than this is a substring of ordinary prose more
# often than it is evidence. Judged by length in characters, not words:
# "npe" and "vla" are legitimate three-character answers.
MIN_ALT_LEN = 3

# Words that appear in almost any technical answer. Matching one of them
# is not evidence the model knew anything.
TOO_COMMON = {
    "the", "and", "for", "not", "you", "use", "used", "using", "with",
    "that", "this", "was", "are", "can", "will", "value", "values",
    "function", "code", "data", "type", "return", "returns", "set",
    "call", "calls", "called", "when", "what", "how", "yes",
}


# Suffixes an author silently assumes a stem will cover. It will not:
# the grader matches whole words.
INFLECTIONS = ("e", "s", "es", "ed", "ing", "y", "ies", "ion", "ions",
               "al", "ity", "ies", "er", "ers", "ly",
               # "le" and "ility": "visib" -> visible / visibility. Missed
               # the first time and it cost a near-domain control, which
               # was scored as a model failure on a correct answer.
               "le", "les", "ility", "ilities")


def stem_suspects(alt: str, context: str, alts: set[str]) -> str | None:
    """Return the inflected form an alternative was probably meant to
    cover, if the alternative looks like a truncated stem.

    Evidence, not guesswork: the inflected form has to actually appear
    in the question or note — the author wrote the real word there while
    typing a stem into `expect`. A stem whose inflection is also listed
    as its own alternative is fine, since the group can still match.
    """
    # Six characters and up now WORK as stems: the grader matches a long
    # alphabetic alternative as a left-bounded prefix, so "optimiz"
    # covers optimize/optimized/optimisation deliberately. Flagging those
    # would be flagging the feature.
    #
    # Four and five characters are the trap. They fall under the stem
    # threshold, so they match whole-word only, and an author writing
    # "visib", "modif", "cach" or "decay" gets an expectation that can
    # never match visible / modifying / caching / decays. Three of the
    # near-domain controls were scored as model failures on answers that
    # were completely correct, and the threshold itself is not the bug —
    # lowering it to five would let "state" match "statement".
    if not alt.isalpha() or not (4 <= len(alt) <= 5):
        return None

    # If the alternative appears as a WHOLE WORD in the evidence, it is a
    # real word and not a truncated stem — leave it alone.
    #
    # Without this the check drowns itself. Scanning a whole corpus finds
    # an inflection of nearly every common word, so "throw" got flagged
    # because "throws" exists somewhere, and likewise user/users,
    # path/paths, hook/hooks, link/links, tool/tools. Eighty-two hits,
    # almost all of them correct expectations, which is how a linter
    # trains people to ignore it.
    #
    # A genuine stem is not a word: "optimiz", "visib", "cach", "modif"
    # and "eliminat" never appear standalone anywhere, which is exactly
    # what distinguishes them from "throw".
    if re.search(rf"\b{re.escape(alt)}\b", context, re.I):
        return None

    for suf in INFLECTIONS:
        cand = alt + suf
        if cand in alts:
            return None
        # Word-boundary search so "task" does not "find" itself in "tasks"
        # only because the substring is there.
        if re.search(rf"\b{re.escape(cand)}\b", context, re.I):
            return cand
    return None


def lint_file(path: Path) -> list[str]:
    problems: list[str] = []
    seen_ids: set[str] = set()

    # The pack's own corpus is evidence too, and it is the evidence that
    # matters most. A model's answer vocabulary comes from the records
    # injected into it, so if the corpus says "compilers optimize away"
    # and the expectation says "optimiz", the corpus is where the proof
    # lives — not in the question, which is what this check used to read.
    #
    # That gap is exactly how 35 unmatchable stems reached publication:
    # every one appeared in answers and corpus records, and none of them
    # appeared in the question that was supposed to reveal it.
    corpus_path = path.parent / "corpus.md"
    corpus = corpus_path.read_text(encoding="utf-8") if corpus_path.exists() else ""

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        where = f"{path.parent.name}/{path.name}:{lineno}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"{where}: not valid JSON — {e}")
            continue

        qid = row.get("id", f"<line {lineno}>")
        if not row.get("q"):
            problems.append(f"{where} [{qid}]: no question")
        if qid in seen_ids:
            problems.append(f"{where} [{qid}]: duplicate id")
        seen_ids.add(qid)

        groups = row.get("expect")
        if not groups:
            problems.append(f"{where} [{qid}]: no expect groups — cannot be graded")
            continue

        # Some short alternatives really are the answer: "20" for the post
        # type length limit, "0" for what React renders, "when" for the
        # guard keyword. `short_ok` is how a question declares that it
        # meant it. It is a list, not a boolean, so each waiver names the
        # exact string it excuses and a later reader can see the judgment
        # instead of inheriting a blanket exemption.
        waived = {a.lower().strip() for a in row.get("short_ok", [])}

        normalised: list[frozenset[str]] = []
        for gi, group in enumerate(groups):
            if not isinstance(group, list) or not group:
                problems.append(f"{where} [{qid}]: group {gi} is empty")
                continue
            alts = frozenset(a.lower().strip() for a in group)
            normalised.append(alts)

            for alt in sorted(alts):
                if alt in waived:
                    continue
                if len(alt) < MIN_ALT_LEN:
                    problems.append(
                        f"{where} [{qid}]: group {gi} alternative {alt!r} is "
                        f"{len(alt)} chars — matches incidental prose, so the "
                        f"group can never fail (add to short_ok if deliberate)"
                    )
                if alt in TOO_COMMON:
                    problems.append(
                        f"{where} [{qid}]: group {gi} alternative {alt!r} is a "
                        f"word almost any answer contains (add to short_ok if "
                        f"it is genuinely the answer)"
                    )
                # The too-strict direction: a stem the word-boundary
                # matcher can never match, marking correct answers wrong.
                own = f"{row.get('q', '')} {row.get('note', '')}"
                inflected = stem_suspects(alt, own, set(alts))
                where_seen = "this question's own text"
                if not inflected and corpus:
                    inflected = stem_suspects(alt, corpus, set(alts))
                    where_seen = "the pack's corpus"
                if inflected:
                    problems.append(
                        f"{where} [{qid}]: group {gi} alternative {alt!r} is "
                        f"{len(alt)} chars, below the stem threshold, so the "
                        f"grader matches it whole-word only and it will NOT "
                        f"match {inflected!r} — which {where_seen} uses. List "
                        f"the full form(s), or lengthen the stem to 6+ chars."
                    )

        for alt in sorted(waived):
            if not any(alt in g for g in normalised):
                problems.append(
                    f"{where} [{qid}]: short_ok lists {alt!r}, which is not in "
                    f"any expect group — a stale waiver hides the next problem"
                )

        # A question asking for N things needs N groups that can be
        # satisfied INDEPENDENTLY. Identical or nested groups collapse
        # into one, silently reducing what the question demands.
        for i in range(len(normalised)):
            for j in range(i + 1, len(normalised)):
                a, b = normalised[i], normalised[j]
                if a == b:
                    problems.append(
                        f"{where} [{qid}]: groups {i} and {j} are identical — "
                        f"one match satisfies both, so they count as one group"
                    )
                elif a <= b or b <= a:
                    smaller, larger = (i, j) if a <= b else (j, i)
                    problems.append(
                        f"{where} [{qid}]: group {smaller} is a subset of group "
                        f"{larger} — any match for the smaller also satisfies "
                        f"the larger"
                    )

        # A note explaining WHY the model gets this wrong is what keeps
        # the set honest over time; without it, a future edit cannot tell
        # a load-bearing alternative from a decorative one.
        if not row.get("note"):
            problems.append(f"{where} [{qid}]: no note explaining the failure mode")

    return problems


def main() -> int:
    wanted = sys.argv[1:]
    files = sorted(HERE.glob("*/eval.jsonl"))
    if wanted:
        files = [f for f in files if f.parent.name in wanted]
    if not files:
        print("no eval.jsonl found")
        return 1

    total = 0
    for f in files:
        problems = lint_file(f)
        rows = sum(1 for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
        status = "OK" if not problems else f"{len(problems)} problem(s)"
        print(f"{f.parent.name:<26} {rows:>3} questions   {status}")
        for p in problems:
            print(f"    {p}")
        total += len(problems)

    print(f"\n{total} problem(s) across {len(files)} eval set(s)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
