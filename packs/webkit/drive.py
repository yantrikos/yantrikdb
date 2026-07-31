#!/usr/bin/env python3
"""Have a local model drive the op protocol, and measure whether it did.

Everything rendered so far came from ops I wrote by hand, which proves
only that the compiler works. The claim is about what a 4B can produce,
so this is the experiment that decides whether there is a kit at all.

The design point that makes the number honest is the CONFORMANCE GATE.
A generous compiler will happily turn broken output into a handsome
page — substituting a default family here, dropping an illegal slot
there — and we would be measuring the compiler while reporting a model
result. So a run counts only if the model's own lines parse, every
enum and slot is legal, and NO fallback fired. Rendered quality is
scored separately, and only for runs that already cleared conformance.

Five calls per site, never the whole page in one response: the measured
failure mode is that small models close short structured units cleanly
and fall apart on long documents.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLLAMA = "http://localhost:11434"

# This is the pack: the grammar and the taste tables, as the model sees
# them. Nothing here is HTML or CSS — the model cannot express markup
# even if it wants to.
GRAMMAR = """You write ops for a website compiler. You NEVER write HTML, CSS or
JavaScript — the compiler does all of that. You only choose content and
make choices from the tables below.

FORMAT: one op per line. `KEY=value`, or `KEY="value with spaces"`.
Emit ONLY op lines. No prose, no markdown, no code fences, no comments.

EVERY OP, WITH A LITERAL EXAMPLE. Copy these shapes exactly, changing
only the values. Every argument is written as KEY=value.

  SITE     name="Tide Mill"  nav="Bread,Visit,Contact"  location="Bristol"  contact="hello@tidemill.co"
  THEME    family=editorial  accent=#b8442f  mode=light  density=roomy
  SECTION  id=s1  kind=hero  layout=split  tone=quiet
  TEXT     sec=s1  slot=title  text="Bread [worth the walk]"
  TEXT     sec=s1  slot=lede  text="Two ovens and no cafe."
  ACTION   sec=s1  slot=primary  label="Collection times"  href="#times"
  MEDIA    sec=s1  slot=figure  motif=elevation
  ITEM     sec=s2  slot=item  title="Tuesday"  body="Rye and caraway, forty loaves."

In a `proof` section, ITEM title= is the FIGURE and body= says what it
counts. Keep the figure short enough to read at a glance. These two are
about a DIFFERENT business, a bicycle workshop — the figures in them are
not facts about your site, they only show the shape:
  ITEM     sec=s3  slot=item  title="1968"  body="The year the workshop opened."
  ITEM     sec=s3  slot=item  title="11"    body="Frames built last year."

Never write a person's name as a customer, a review or an endorsement.
You have no way to know what anyone said.

Note the shape of TEXT: the slot is the VALUE of slot=, and the words
go in text=. Never write `eyebrow=...` or `title=...` as the key.

To emphasise words, put [square brackets] around them INSIDE text=.
The bracketed words are set in the accent colour. Use this on ONE
title, around two or three of its own words.
  RIGHT: text="Bread [worth the walk]"
  WRONG: text="Bread worth the walk"  mark="worth the walk"
  WRONG: text="Wren & Slip"  (marking words that are not in the title)
The only colour anywhere in this language is THEME accent=.

CLOSED VOCABULARIES — any other value is invalid
  family   editorial | studio | technical
  mode     light | dark
  density  tight | normal | roomy
  kind     hero | features | cta | proof | detail
  layout   split | centred | stack | grid | list
  tone     quiet | bold | inverted
  motif    elevation | topography | schematic | dial | specimen
  slot     eyebrow | title | lede | primary | secondary | figure | item

WHICH FAMILY TO CHOOSE
  editorial  serif, magazine feel. Architecture, food, writing, craft.
  studio     heavy sans, poster feel. Design, portfolios, campaigns.
  technical  monospace, schematic feel. Developer tools, hardware, data.

WHICH MOTIF TO CHOOSE
The compiler draws an abstract line DIAGRAM — never a photograph and
never the object itself. Choose the closest one. You do not write a
caption; the compiler captions the drawing it made.
  elevation   a building drawn face-on with a dimension line.
              For premises, property, places.
  topography  nested contour rings, like a map.
              For land, outdoors, environment, coverage, spread.
  schematic   boxes joined by routed connectors.
              For software, infrastructure, pipelines, systems.
  dial        a graduated circular gauge with a needle.
              For instruments, precision, time, measurement.
  specimen    a large letterform over baseline rules.
              For typography, publishing, language, archives of text.

RULES
  THEME accent must be a 6-digit hex like #b8442f. Choose a hue that
    suits the subject; the compiler derives every other colour from it.
  There is no mark= argument. Emphasis is [brackets] inside text=.
  Slots legal per kind:
    hero      eyebrow, title, lede, primary, secondary, figure
    features  eyebrow, title, lede, item
    cta       eyebrow, title, lede, primary, secondary
    proof     eyebrow, title, item
    detail    eyebrow, title, lede, primary, figure
  The closing title must say something the hero did not. Repeating the
    headline at the bottom of the page is the commonest way a site
    reads as filled-in rather than written.
  Write specific copy. Real names, real numbers, real detail. Never
    "Lorem ipsum", never "Your Company", never "Feature One"."""

# The harness emits SECTION lines itself. They are pure structure —
# every value was already dictated word-for-word by the prompt, so
# asking for them back added a failure mode and no expressive power:
# the features call silently omitted its SECTION and orphaned five
# following ops. Ask the model only for what it actually decides.
# Six sections rather than three. The three-kind skeleton (hero,
# features, cta) meant every page had the same shape whatever the brief
# said, and a fixed shape is what reads as generated no matter how well
# each part is set. Each call stays a short structured unit — the
# measured thing a small model can close — so the page gets longer
# without any single response getting harder.
#
# Fourth field is the section this call may write to. It used to be
# derived from the scaffold with a special case for `items`, which was
# fine while exactly one call emitted ITEMs and wrong the moment two did.
CALLS = [
    ("frame", None, None, """Site: {brief}

Emit EXACTLY two lines: one SITE and one THEME. Nothing else."""),
    ("hero", "SECTION  id=s1  kind=hero  layout=split  tone=quiet", "s1",
     """Site: {brief}

The hero section already exists. Emit ONLY its contents:
TEXT for slot=eyebrow, slot=title and slot=lede; two ACTION lines
(slot=primary, slot=secondary); one MEDIA line with slot=figure and a
motif. Do NOT emit SECTION. Do NOT emit THEME or SITE again.
In the title's text, wrap two or three of its own words in [brackets]."""),
    ("proof", "SECTION  id=s2  kind=proof  layout=stack  tone=quiet", "s2",
     """Site: {brief}

These are every figure in the brief, extracted for you:
{figures}

The proof section already exists. Emit TEXT for slot=eyebrow, TEXT for
slot=title, then ONE ITEM line with sec=s2 and slot=item FOR EACH
figure in that list — and no others.

In each ITEM, title= is the figure exactly as the brief gives it, and
body= is one short line saying what it counts.

Write each figure exactly as it appears there. Do not estimate,
benchmark, price or round, and do not add a figure that is not listed.
If the list is empty, emit NOTHING AT ALL — not the eyebrow, not the
title. An empty response is the correct answer to a brief with nothing
to count. Do NOT emit SECTION."""),
    ("detail", "SECTION  id=s3  kind=detail  layout=split  tone=quiet", "s3",
     """Site: {brief}

The detail section already exists. Pick the ONE thing about this
business that a competitor could not copy, and explain it. Emit TEXT
for slot=eyebrow, slot=title and slot=lede, one ACTION with
slot=primary, and one MEDIA with slot=figure and a motif. Wrap two or
three words of the title in [brackets]. Do NOT emit SECTION."""),
    ("features", "SECTION  id=s4  kind=features  layout=list  tone=quiet", "s4",
     """Site: {brief}

The features section already exists. Emit ONLY two lines: TEXT for
slot=eyebrow and TEXT for slot=title, both with sec=s4.
Do NOT emit SECTION. Do NOT emit ITEM in this response."""),
    ("items", None, "s4", """Site: {brief}

Emit EXACTLY three ITEM lines, each `sec=s4  slot=item`, with a short
title and a body of one or two specific sentences. Nothing else."""),
    # There is deliberately NO quote call.
    #
    # The compiler supports kind=quote and renders it well, but a model
    # cannot source a testimonial. Asked for one, this 4B produced
    # "Sarah Chen, DevOps Lead at RiverScale" — a named person at a
    # named company, neither of which exists, attesting to a product
    # they have never used. That is not a copy defect to tune away; it
    # is a fabricated endorsement, and it would go on a real site under
    # someone's real business name.
    #
    # Unlike the invented figures below, there is no computable ground
    # truth to check a quote against: the brief has no customers in it,
    # so every possible answer is a fabrication. When a slot cannot be
    # verified even in principle, the honest move is to keep it out of
    # the model's reach and let a human supply it. kind=quote stays
    # available for hand-authored ops, where the words came from a
    # person who actually said them.
    ("cta", "SECTION  id=s6  kind=cta  layout=stack  tone=quiet", "s6",
     """Site: {brief}

The closing section already exists. Emit ONLY its contents: TEXT for
slot=title, TEXT for slot=lede, and one ACTION with slot=primary, all
with sec=s6. Do NOT emit SECTION.
The title must NOT repeat the hero headline — say the next thing:
when to come, what happens first, or what it costs."""),
]

OP_LINE = re.compile(r"^\s*(SITE|THEME|SECTION|TEXT|ACTION|MEDIA|ITEM)\b")

NUMERIC = re.compile(r"\d[\d,.]*")

# A figure plus whatever unit is fused to it ("4K", "12,400") and the
# word that follows ("12,400 reels"), so the model is labelling a thing
# rather than a bare integer.
FIGURE_IN_CONTEXT = re.compile(r"\d[\d,.]*[A-Za-z]*(?:\s+[a-z]+)?")


def figures_in(brief: str) -> str:
    """The numbers the brief actually contains, as a list for the prompt.

    Asked to write one item per figure in the brief, the 4B found one of
    the six in the archive brief and the section was dropped for being
    too thin — a correct outcome from an avoidable cause. Locating
    digits in a paragraph is something a regex does perfectly and a 4B
    does unreliably, so it should not have been the model's job. Extract
    them here and leave the model the part that genuinely needs
    judgement: saying what each one counts.
    """
    seen, out = set(), []
    for m in FIGURE_IN_CONTEXT.finditer(brief):
        text = m.group(0).strip().rstrip(".,")
        key = NUMERIC.search(text).group(0)
        if key in seen:
            continue
        seen.add(key)
        out.append(f"  - {text}")
    return "\n".join(out) if out else "  (none — the brief states no figures)"


def ungrounded_figures(line: str, brief: str) -> list[str]:
    """Numbers in a stats item that are nowhere in the brief.

    Told plainly not to invent figures, the 4B answered a brief with no
    numbers in it with "140ms — time to scan 5GB of schema", "$3.8k/yr —
    cost savings on a medium cluster" and "99% — accuracy in index
    detection". Three fabricated benchmarks with the authority of a
    measurement, on a page a real project would publish.

    This is the same shape as the contrast result: the instruction was
    stated clearly and cost nothing, because the model has no way to
    check itself against it. Whether a digit appears in the brief IS
    checkable, so check it here instead of asking.

    Compares digits only. "Two ovens" does not license "2 ovens" — the
    brief writes numbers the way it writes them, and a model rewriting
    prose into a figure is still reporting something the brief did not
    say.
    """
    m = re.search(r'title="([^"]*)"', line)
    if not m:
        return []
    # A stats band entry with no digits in it is not a figure. Given a
    # brief with nothing to count, the 4B filled the slot anyway with
    # "Terminal output showing index names" — which passes a check that
    # only looks for ungrounded NUMBERS, because it contains none. Empty
    # is the answer here, so treat a figureless figure as ungrounded.
    if not NUMERIC.search(m.group(1)):
        return ["<no figure>"]
    ground = set(NUMERIC.findall(brief.replace(",", "")))
    bad = []
    for tok in NUMERIC.findall(m.group(1).replace(",", "")):
        if tok in ground:
            continue
        # A figure the brief states as part of a longer number is still
        # grounded ("2019" licenses "2019", not "19").
        if any(tok in g for g in ground):
            continue
        bad.append(tok)
    return bad


def ask(model: str, system: str, user: str) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False, "think": False, "keep_alive": "20m",
        "options": {"num_predict": 900, "temperature": 0.2},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r).get("message", {}).get("content", "") or ""


def clean(raw: str) -> tuple[list[str], int]:
    """Keep op lines; count everything else as noise the model emitted."""
    kept, noise = [], 0
    for line in raw.splitlines():
        line = line.strip().strip("`")
        if not line:
            continue
        if OP_LINE.match(line):
            kept.append(line)
        else:
            noise += 1
    return kept, noise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--brief", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--debug", action="store_true",
                    help="print every non-op line the model emitted")
    args = ap.parse_args()

    lines: list[str] = []
    noise_total = 0
    raw_log: list[str] = []
    dropped: list[str] = []
    for label, scaffold, want_sec, template in CALLS:
        # The scaffold is held back until the call has actually produced
        # content for it. Emitting it up front is what would leave an
        # empty `proof` heading on a brief with nothing to count — a
        # section promising figures and delivering none.
        scaffold_pending = scaffold
        # Each call is a fresh context, so the model reconstructs the
        # document by restating it — replaying SITE, THEME and the hero
        # SECTION before answering, and sometimes writing ahead into
        # sections it was not asked for. Showing it what already exists
        # removes the reason to invent it. Cheaper and more reliable
        # than telling it "do not repeat" more loudly.
        prior = ("\n".join(lines + ([scaffold_pending] if scaffold_pending else []))
                 if lines or scaffold_pending else "(nothing yet)")
        user = (f"ALREADY WRITTEN — this is the document so far. Do not "
                f"repeat or modify any of it:\n{prior}\n\n"
                + template.format(brief=args.brief,
                                  figures=figures_in(args.brief)))
        raw = ask(args.model, GRAMMAR, user)
        raw_log.append(f"### {label}\n{raw}")
        kept, noise = clean(raw)
        if noise and args.debug:
            for line in raw.splitlines():
                s = line.strip().strip("`")
                if s and not OP_LINE.match(s):
                    print(f"      NOISE: {s[:96]!r}")
        # A repeated SITE/THEME/SECTION is the model overstepping its
        # brief; count it as noise rather than letting a later THEME
        # silently overwrite the first.
        # Stray = structure the harness owns, or content aimed at a
        # section this call was not about. The second kind is what
        # produced ops referencing s3 before s3 existed.
        def _stray(line: str) -> bool:
            head = line.split()[0].upper()
            if head == "SECTION":
                return True
            if label != "frame" and head in {"SITE", "THEME"}:
                return True
            m = re.search(r"\bsec=\"?(s\d)", line)
            return bool(want_sec and m and m.group(1) != want_sec)

        stray = [l for l in kept if _stray(l)]
        kept = [l for l in kept if l not in stray]
        # An alt= the model volunteered anyway would outrank the derived
        # caption in the compiler, which trusts alt= as human-authored.
        # Strip it rather than let one leak through and reinstate the
        # very claim the derivation exists to prevent.
        kept = [re.sub(r'\s*alt="[^"]*"', "", l) if l.split()[0].upper() == "MEDIA"
                else l for l in kept]
        if stray and args.debug:
            for s in stray:
                print(f"      STRAY: {s[:96]!r}")

        # Figures the brief never stated are removed, not reported and
        # kept. A fabricated benchmark that survives into the page with
        # a warning printed beside it has still shipped.
        fabricated = 0
        if label == "proof":
            surviving = []
            for line in kept:
                if line.split()[0].upper() != "ITEM":
                    surviving.append(line)
                    continue
                bad = ungrounded_figures(line, args.brief)
                if bad:
                    fabricated += 1
                    if args.debug:
                        print(f"      UNGROUNDED {','.join(bad)}: {line[:80]!r}")
                else:
                    surviving.append(line)
            kept = surviving
            # Two is the floor for a band of figures. An eyebrow and a
            # title over nothing is a promise the section cannot keep;
            # over a single stat it is worse, because the layout reads
            # as a row with its other cells missing. Dropping the
            # section is the honest rendering of "the brief does not
            # support this".
            n_items = sum(1 for l in kept if l.split()[0].upper() == "ITEM")
            if n_items < 2:
                dropped.append(
                    f"{label} ({n_items} grounded figure(s) in the brief, needs 2)")
                kept = []

        noise_total += noise + len(stray)
        if kept and scaffold_pending:
            lines.append(scaffold_pending)
            scaffold_pending = None
        print(f"  {label:<9} {len(kept):>2} ops, {noise:>2} non-op lines"
              + (f", {len(stray)} stray structural" if stray else "")
              + (f", {fabricated} UNGROUNDED figures dropped" if fabricated else ""))
        lines.extend(kept)

    ops_path = HERE / "generated" / f"{args.name}.ops"
    ops_path.parent.mkdir(parents=True, exist_ok=True)
    ops_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = HERE / "out" / f"{args.name}.html"
    py = Path(sys.executable)
    proc = subprocess.run(
        [str(py), str(HERE / "compiler.py"), str(ops_path),
         "--out", str(out), "--strict"],
        capture_output=True, text=True)
    print(proc.stdout.strip())

    conformant = proc.returncode == 0 and noise_total == 0
    print(f"\n  CONFORMANCE: {'PASS' if conformant else 'FAIL'}"
          f"  (compiler clean={proc.returncode == 0}, "
          f"non-op lines={noise_total})")
    for d in dropped:
        print(f"  DROPPED: {d}")
    print(f"  ops -> {ops_path}")
    return 0 if conformant else 1


if __name__ == "__main__":
    raise SystemExit(main())
