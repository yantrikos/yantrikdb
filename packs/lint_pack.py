#!/usr/bin/env python3
"""Check that a pack's two tiers are in sync: every rule has evidence.

A pack has two delivery mechanisms and they are supposed to work as one
system:

  constitution  injected unconditionally — the rules the model applies
  corpus        retrieved by similarity  — the facts and worked examples

They fail together in a way neither shows on its own. The constitution
says "declare presets and then apply them under styles"; the corpus holds
the worked theme.json that demonstrates it; and the model never sees the
example, because a record that is mostly code embeds as code and does not
retrieve on its topic. The rule arrives with no evidence, the evidence is
never delivered, and both tiers look fine in isolation.

This checks the join. For every rule in the constitution it queries the
mounted pack with the rule's own vocabulary and asks whether any corpus
record clears the similarity floor the consumer will use. A rule with no
retrievable support is an assertion the pack cannot back up.

It also reports the inverse — corpus records that nothing retrieves,
whatever the query — because an unreachable memory is dead weight that
costs pack size and buys nothing.

Usage:
    python packs/lint_pack.py wordpress-theme
    python packs/lint_pack.py --all
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"

# The floor a consumer actually gates on. Measured, not chosen: on-topic
# queries land 0.65-0.79 and off-topic 0.09-0.45, and ungated injection
# took an unrelated control set from 12/12 to 5/12.
MIN_SIMILARITY = 0.55


def headings(md: Path) -> list[str]:
    if not md.exists():
        return []
    return [m.group(1).strip()
            for m in re.finditer(r"^##\s+(.+?)\s*$", md.read_text(encoding="utf-8"), re.M)]


def rule_queries(md: Path) -> list[tuple[str, str]]:
    """(heading, query) for each rule, where the query is heading + body.

    Querying on the heading alone was the wrong proxy. Constitution
    headings are imperatives — "Sections carry rhythm", "Colour is
    chosen" — which make good rules and terrible queries: the bundled
    64-dim embedder matches concrete technical vocabulary, not abstract
    instruction. "Layout type is chosen deliberately" scores 0.386
    against a corpus that answers "constrained flex grid layout type" at
    0.679. A consumer asks in task language, so the check has to as well.
    """
    if not md.exists():
        return []
    out = []
    for block in re.split(r"(?m)^##\s+", md.read_text(encoding="utf-8"))[1:]:
        lines = block.splitlines()
        head = lines[0].strip()
        body = " ".join(l.strip() for l in lines[1:] if l.strip())
        out.append((head, f"{head}. {body}"[:400]))
    return out


def code_ratio(text: str) -> float:
    """Fraction of the record inside fenced code blocks.

    The number that predicts whether a record is retrievable. With the
    bundled 64-dim embedder a record's embedding is dominated by whatever
    there is most of, so a mostly-code record embeds as generic code and
    stops matching its own topic. Measured: a 2.5KB theme.json exemplar
    was unreachable by every query tried, while the functions.php
    exemplar — least code, most prose — won even for theme.json queries.
    """
    fenced = sum(len(b) for b in re.findall(r"```.*?```", text, re.S))
    return fenced / max(len(text), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("packs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--floor", type=float, default=MIN_SIMILARITY)
    args = ap.parse_args()

    names = args.packs
    if args.all or not names:
        names = sorted(p.name for p in HERE.iterdir()
                       if p.is_dir() and (p / "corpus.md").exists())

    sys.path.insert(0, str(HERE.parent / "src"))
    from yantrikdb import YantrikDB  # noqa: PLC0415

    problems = 0
    for name in names:
        src = HERE / name
        built = sorted(DIST.glob(f"{name}-*.ydbpack"))
        if not built:
            print(f"{name}: not built — run build.py first")
            problems += 1
            continue

        rules = headings(src / "constitution.md")
        facts = headings(src / "corpus.md")
        print(f"\n{name}  —  {len(rules)} rules, {len(facts)} corpus records")

        td = tempfile.mkdtemp()
        try:
            db = YantrikDB(os.path.join(td, "host.ydb"), 64)
            db.mount_pack(str(built[-1]))

            # Which rules have retrievable evidence?
            orphan_rules = []
            for rule, query in rule_queries(src / "constitution.md"):
                best = 0.0
                for hit in db.recall(query, top_k=5):
                    best = max(best, hit.get("scores", {}).get("similarity", 0.0))
                if best < args.floor:
                    orphan_rules.append((rule, best))

            # Which records does nothing reach? Query each by its own
            # heading — the most favourable query it will ever get. A
            # record that cannot be retrieved by its own title cannot be
            # retrieved at all.
            unreachable = []
            corpus_text = (src / "corpus.md").read_text(encoding="utf-8")
            blocks = re.split(r"^##\s+", corpus_text, flags=re.M)[1:]
            for block in blocks:
                head = block.splitlines()[0].strip()
                best, top = 0.0, ""
                for hit in db.recall(head, top_k=3):
                    sim = hit.get("scores", {}).get("similarity", 0.0)
                    if sim > best:
                        best, top = sim, hit.get("text", "").split(" — ")[0]
                if top.strip() != head or best < args.floor:
                    unreachable.append((head, best, top[:46], code_ratio(block)))

            db.close()
        finally:
            shutil.rmtree(td, ignore_errors=True)

        if orphan_rules:
            print(f"  {len(orphan_rules)} rule(s) with no corpus record above "
                  f"{args.floor} — asserted without evidence:")
            for rule, best in orphan_rules:
                print(f"      {best:.3f}  {rule[:66]}")
            problems += len(orphan_rules)

        heavy = [u for u in unreachable if u[3] > 0.5]
        if heavy:
            print(f"  {len(heavy)} record(s) not retrievable by their own heading, "
                  f"code-dominated:")
            for head, best, top, ratio in heavy:
                print(f"      {best:.3f}  {int(ratio * 100):>3}% code  {head[:44]}")
                print(f"               instead got: {top}")
            problems += len(heavy)

        light = [u for u in unreachable if u[3] <= 0.5]
        if light:
            print(f"  {len(light)} record(s) outranked by a sibling on their own "
                  f"heading (near-duplicates compete):")
            for head, best, top, _ in light[:6]:
                print(f"      {best:.3f}  {head[:44]}  ->  {top}")

        if not orphan_rules and not heavy:
            print("  rules and records are in sync")

    print(f"\n{problems} sync problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
