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
grader bugs always fail in, because a grader that is too strict gets
noticed the first time a correct answer is marked wrong, and a grader
that is too lenient never gets noticed at all.

Usage:
    python packs/lint_evals.py            # every pack
    python packs/lint_evals.py c-safety
"""

from __future__ import annotations

import json
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


def lint_file(path: Path) -> list[str]:
    problems: list[str] = []
    seen_ids: set[str] = set()

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
