#!/usr/bin/env python3
"""Compare what different models do with the same kit, on the properties
that were actually in question.

Written after reporting "more variety" on the strength of a conformance
rate, while shipping four pages with identical structure. A pass rate
cannot see sameness: every one of those pages passed. So this measures
the adjectives directly — how many DISTINCT page structures a model
produced, how much it wrote, and how often it tried to state something
the brief never said.

    python packs/webkit/compare.py n4 n9
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def structure(ops: Path) -> list[str]:
    return re.findall(r"kind=(\w+)", ops.read_text(encoding="utf-8"))


NUM = re.compile(r"\d[\d,.]*[A-Za-z]*")
# Quantities written as words. The enforced check looks for digits, so
# "under ten seconds" and "over ninety percent" walk straight past it.
WORD_NUM = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
    r"thousand|million|billion)\b", re.I)


def ungrounded(ops: Path, brief: str) -> tuple[int, list[str]]:
    """Every quantity in the copy that the brief never stated.

    Reported, not enforced — the enforced check covers only prices and
    clock times, the two a reader acts on. This exists because "0
    fabrications caught" was about to get repeated as "the 9B does not
    fabricate", when its own page claims a scan "finishes in under ten
    seconds" on "a 50TB cluster". The metric measured the two classes it
    was built for and I nearly read it as measuring honesty.
    """
    ground = brief.lower().replace(",", "")
    bad = []
    for m in re.finditer(r'\b(?:text|title|body|label)="([^"]*)"',
                         ops.read_text(encoding="utf-8")):
        for tok in NUM.findall(m.group(1)) + WORD_NUM.findall(m.group(1)):
            t = tok.lower().replace(",", "")
            if t not in ground:
                bad.append(tok)
    return len(bad), bad


def words(ops: Path) -> int:
    """Words of visible copy: text=, title=, body=, label=."""
    n = 0
    for m in re.finditer(r'\b(?:text|title|body|label)="([^"]*)"',
                         ops.read_text(encoding="utf-8")):
        n += len(m.group(1).split())
    return n


def main() -> int:
    import json
    briefs = json.loads((HERE / "briefs.json").read_text())["briefs"]
    prefixes = sys.argv[1:] or ["n4", "n9"]
    rows = []
    for pre in prefixes:
        files = sorted((HERE / "generated").glob(f"{pre}-*.ops"))
        if not files:
            print(f"no ops for prefix {pre!r}")
            continue
        structs = [structure(f) for f in files]
        rows.append((pre, files, structs))

    for pre, files, structs in rows:
        uniq = {tuple(s) for s in structs}
        wc = [words(f) for f in files]
        print(f"\n{pre}")
        print(f"  {len(files)} briefs, {len(uniq)} distinct structures, "
              f"{sum(len(s) for s in structs) / len(structs):.1f} sections avg")
        print(f"  copy: {min(wc)}-{max(wc)} words "
              f"(mean {sum(wc) // len(wc)})")
        total = 0
        for f, s in zip(files, structs):
            brief = briefs.get(f.stem.split("-", 1)[1], "")
            n, toks = ungrounded(f, brief)
            total += n
            print(f"    {f.stem:<14} {' '.join(s)}")
            if toks:
                print(f"      {n} unstated quantities: "
                      f"{', '.join(dict.fromkeys(toks))[:88]}")
        print(f"  unstated quantities in copy: {total} across {len(files)} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
