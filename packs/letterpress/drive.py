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
  MEDIA    sec=s1  slot=figure  photo="sourdough bread bakery"
  MEDIA    sec=s3  slot=figure  motif=lattice
  ITEM     sec=s2  slot=item  title="Tuesday"  body="Rye and caraway, forty loaves."

In a `proof` section, ITEM title= is the FIGURE and body= says what it
counts. Keep the figure short enough to read at a glance. These two are
about a DIFFERENT business, a bicycle workshop — the figures in them are
not facts about your site, they only show the shape:
  ITEM     sec=s3  slot=item  title="1968"  body="The year the workshop opened."
  ITEM     sec=s3  slot=item  title="11"    body="Frames built last year."

Never write a person's name as a customer, a review or an endorsement.
You have no way to know what anyone said.

Never write a price or a clock time that is not in the brief. If the
brief does not say what something costs or when it opens, write around
it — "priced by the loaf", "opening hours on the door" — because a
reader acts on those two and turns up with the wrong money at the wrong
hour. Any line containing one is deleted before it reaches the page.

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
  kind     hero | features | cta | proof | detail | faq | roster | note
  layout   split | centred | stack | grid | list
  tone     quiet | bold | inverted
  motif    elevation | topography | schematic | dial | specimen
           | strata | lattice | orbit
  slot     eyebrow | title | lede | primary | secondary | figure | item

WHICH FAMILY TO CHOOSE
  editorial  serif, magazine feel. Architecture, food, writing, craft.
  studio     heavy sans, poster feel. Design, portfolios, campaigns.
  technical  monospace, schematic feel. Developer tools, hardware, data.

A PHOTOGRAPH OR A DRAWING
MEDIA takes EITHER `photo="..."` or `motif=...`, never both.

  photo=  a real photograph, searched for and licensed by the compiler.
          Use it whenever the subject is a THING SOMEONE CAN SEE — food,
          a room, a landscape, a made object, a place. A restaurant, a
          trip, a recipe, a property or a product wants photographs.

  motif=  an abstract line diagram the compiler draws. Use it when the
          subject has nothing to photograph — software, a service, a
          span of years, an area covered.

HOW TO WRITE photo=
Two to four plain nouns, and nothing else. The search matches ALL your
words at once, so a sentence finds nothing at all:
  RIGHT: photo="sourdough bread bakery"
  RIGHT: photo="mountain hiking trail"
  WRONG: photo="sourdough loaves cooling on a wooden counter at dawn"
         (returns zero results — every word must match)
  WRONG: photo="https://images.example.com/bread.jpg"
         (you cannot know that any URL exists; never write one)
Name the SUBJECT, not the mood. "bakery oven bread" finds bread;
"warm inviting artisanal atmosphere" finds nothing.

WHICH MOTIF TO CHOOSE
The compiler draws an abstract line DIAGRAM — never a photograph and
never the object itself. Choose the closest one. You do not write a
caption; the compiler captions the drawing it made.
  elevation   a building drawn face-on with a dimension line.
              ONLY for premises and property — a place someone visits.
              Not for a business that merely has an address.
  topography  nested contour rings, like a map.
              For land, outdoors, environment, coverage, spread.
  schematic   nodes on a grid joined by routed connectors.
              For software, infrastructure, pipelines, systems.
  dial        a graduated circular gauge with a needle.
              For instruments, precision, measurement.
  specimen    a large letterform over baseline rules.
              For typography, publishing, language.
  strata      stacked horizontal bands cut by a section line.
              For a span of time — a record, a back catalogue, a
              history, anything defined by the years it covers.
  lattice     an over-under woven grid.
              For craft and material — food, textile, joinery, print,
              anything made by hand in batches.
  orbit       concentric rings with nodes placed around them.
              For reach and membership — services, a body of people, a
              community, coverage around a centre.

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
    faq       eyebrow, title, lede, item
    roster    eyebrow, title, lede, item
    note      title, lede
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
# The model chooses which sections the page has.
#
# Adding section KINDS did not fix the sameness, because the harness
# still emitted one hardcoded sequence for every brief: three of four
# generated pages came out hero/detail/features/cta, identical in
# structure, differing only in words. Six kinds inside a fixed skeleton
# is still one skeleton.
#
# So the shape becomes a decision, made once, from the brief. hero and
# cta are the fixed ends — every page opens and closes — and everything
# between them is planned.
# What each KIND of site is shaped like.
#
# The menu below is generic and a business is not, so a model choosing
# freely from it converges on the same middle every time: features,
# detail, faq, for a bakery and a database tool alike. Naming the genre
# first gives it a reason to prefer one shape over another, and carries
# the content each genre cannot omit — a restaurant page without hours
# and an address has failed at the only job it had.
GENRES = """restaurant, cafe, bar   hero, note, features, detail, faq  PHOTOGRAPHS
    The note holds hours and street address, high on the page.
    Features is a few dishes, never the whole menu.
travel, tours           hero, proof, features, detail, faq  PHOTOGRAPHS
    Proof is days, group size and what is included. Features is the
    itinerary, one item per day.
portfolio               hero, features, detail  PHOTOGRAPHS
    The shortest page here. Never an FAQ, never a stats band.
blog, publication       hero, features, detail
    Features is recent pieces, one sentence making the case for each.
recipe                  hero, proof, roster, features, faq  PHOTOGRAPHS
    Proof is serves and times. Roster is INGREDIENTS with quantities.
    Features is the METHOD, one action per step.
trade, plumber, builder hero, features, detail, proof, faq
    Features is services named plainly. Detail is the area covered.
clinic, practitioner    hero, features, detail, roster, faq
    Roster is services or disciplines, never named people.
developer tool          hero, features, detail, faq
    Hero says the verb and the object, not the category.
charity, nonprofit      hero, proof, detail, features, note
    Proof is where the money goes. Note is the reassurance.
event, conference       hero, proof, features, roster, faq
    Proof is date, place, venue, ticket. Never invent a date.
shop, single product    hero, detail, proof, features, faq  PHOTOGRAPHS
    Proof is the specification. FAQ is shipping and returns.
studio, agency          hero, features, detail
property, letting       hero, proof, features, detail  PHOTOGRAPHS
    Proof is beds, floor area, price. Features is the rooms.
school, course          hero, features, proof, roster, faq
    Features is the curriculum: what a student can do afterwards."""

PLAN_MENU = """proof     A band of figures. ONLY if the brief states numbers.
detail    One thing done differently, explained, with a drawing.
features  Three to five things offered, each a title and two sentences.
faq       Questions a customer actually asks, answered in full sentences.
          The only section that carries real paragraphs. Use it when
          people need to know how something works before they commit.
roster    The people or the services, named, with a line each.
note      One wide sentence and nothing else. Use it to break two dense
          sections apart. Never two of these."""

# Layout and tone are the harness's, not the model's. They are pure
# presentation with no information in them that the brief supplies, and
# asking for them back only adds ways to be wrong — the same reason
# SECTION lines stopped being asked for.
KIND_LAYOUT = {
    "proof": "stack", "detail": "split", "features": "list",
    "faq": "stack", "gallery": "stack", "roster": "stack",
    "note": "stack", "hero": "split", "cta": "stack",
}

# What a section MEANS in a given genre.
#
# The genre was chosen at plan time and then forgotten: every section
# call used the generic prompt, so a recipe's `roster` — which the genre
# table says is the ingredients — came back as "Monday morning / Fresh
# loaves ready at eight". The plan was right and the content ignored it,
# which is worse than not planning by genre at all, because the shape
# promises something the words do not deliver.
#
# Only the cells that genuinely differ are listed. Anything absent keeps
# the generic prompt, which is correct far more often than not.
GENRE_NOTES = {
    ("recipe", "roster"):
        "In a recipe this section is the INGREDIENTS. One ITEM per "
        "ingredient: title= is the quantity and the ingredient exactly "
        "as the brief gives it (\"400g rye flour\"), body= is any "
        "preparation (\"sifted\", \"at room temperature\"). Do not "
        "invent an ingredient or a quantity that is not in the brief.",
    ("recipe", "features"):
        "In a recipe this section is the METHOD. One ITEM per step, in "
        "order, each step a SINGLE action. title= names the step "
        "(\"Autolyse\", \"First fold\"), body= says exactly what to do "
        "and for how long. Never put two actions in one step.",
    ("recipe", "proof"):
        "In a recipe these figures are the ones a cook checks first: "
        "how many it serves, working time, proving time, baking time "
        "and oven temperature.",
    ("travel", "features"):
        "On a trip page this is the ITINERARY. One ITEM per day or "
        "stage, in order, each with something concrete: a distance, a "
        "crossing, where the night is spent.",
    ("travel", "proof"):
        "On a trip page these figures are what qualifies a booking: "
        "days, group size, nights under canvas, months of departure.",
    ("restaurant", "features"):
        "On a restaurant page this is a HANDFUL OF DISHES that show the "
        "kitchen's range — never the whole menu.",
    ("restaurant", "note"):
        "On a restaurant page this carries the two facts a reader came "
        "for: when it is open and where it is.",
    ("event", "proof"):
        "For an event these figures are the date, the venue, the city "
        "and the number of seats. Never write a date the brief does not "
        "state.",
    ("blog", "features"):
        "For a publication this is the RECENT PIECES. title= is the "
        "piece's title, body= is one sentence making the case for "
        "reading it — never a truncated opening paragraph.",
    ("property", "features"):
        "For a property this is the ROOMS, one ITEM each, with the "
        "dimensions the brief gives.",
    ("school", "features"):
        "For a course this is the CURRICULUM. Each ITEM says what a "
        "student can DO afterwards, not what is covered.",
    ("shop", "proof"):
        "For a product these figures are the specification: "
        "dimensions, weight, materials, warranty.",
    ("charity", "proof"):
        "For a charity these figures are WHERE THE MONEY GOES, which is "
        "the question a donor actually has.",
    ("trade", "detail"):
        "For a trade this section is the AREA COVERED, with the towns "
        "named.",
    ("portfolio", "features"):
        "In a portfolio this is SELECTED WORK. title= is the piece, "
        "body= is the material, the size and who it was for. No "
        "services, no process, no selling.",
}

KIND_PROMPT = {
    "proof": """These are every figure in the brief, extracted for you:
{figures}

Emit TEXT for slot=eyebrow, TEXT for slot=title, then ONE ITEM line
with sec={sec} and slot=item for each figure in that list THAT BELONGS
IN THIS SECTION. Leave out any that belong somewhere else — a recipe's
flour weights belong with the ingredients, not in a band of headline
figures — and never add a figure that is not listed.

In each ITEM, title= is the figure exactly as the brief gives it, and
body= is one short line saying what it counts.

Do not estimate, benchmark, price or round, and do not add a figure
that is not listed. If the list is empty, emit NOTHING AT ALL.""",

    "detail": """Pick the ONE thing about this business that a competitor could not
copy, and explain it. Emit TEXT for slot=eyebrow, slot=title and
slot=lede, one ACTION with slot=primary, and one MEDIA with
slot=figure carrying either photo="two to four nouns" or motif=... —
and not a repeat of the picture the hero already used.
The lede is two or three full sentences.
Wrap two or three words of the title in [brackets].""",

    "features": """Emit TEXT for slot=eyebrow, TEXT for slot=title, then FOUR ITEM lines
with sec={sec} and slot=item.

Each ITEM body= is TWO full sentences: what it is, then one concrete
detail — a time, a material, a step, a limit. One-line bodies are what
make a page look unfinished.""",

    "faq": """Emit TEXT for slot=eyebrow, TEXT for slot=title, then FIVE ITEM lines
with sec={sec} and slot=item.

title= is a question a real customer would type, written the way they
would type it. body= answers it in TWO OR THREE full sentences, with
the specifics — when, how long, what it costs them, what happens if
they cannot. This is the section people actually read; do not write
one-line answers.""",

    # `gallery` is off the menu, for the reason `quote` is.
    #
    # It renders three motifs at small scale, and the model captions
    # each tile. Given a bakery it produced "TUESDAY MORNING — freshly
    # stamped loaves stacked on the bench" over a line drawing of a
    # BUILDING, then "crusts cooling under a single window light" over
    # contour rings, then a clock face. Five abstract parametric
    # diagrams are not a picture library, and a gallery is a promise of
    # photographs. I had just removed exactly this claim from the hero
    # caption and reintroduced it one section down.
    #
    # kind=gallery stays in the compiler for hand-authored ops, where
    # someone choosing `schematic` for three stages of a pipeline is
    # captioning a diagram that really is one.

    "roster": """Emit TEXT for slot=eyebrow, TEXT for slot=title, then THREE ITEM lines
with sec={sec} and slot=item.

title= is the name of a service or a role — NOT the name of a person,
which you have no way to know. body= is two sentences saying what it
covers and who it is for.""",

    "note": """Emit EXACTLY two lines: TEXT with sec={sec} slot=title, and TEXT with
sec={sec} slot=lede.

The title is one short sentence stating the single most useful fact in
the brief — the thing you would tell someone in a doorway. The lede is
one sentence more. No list, no button, no eyebrow.""",
}

HERO_PROMPT = """Emit ONLY the hero's contents: TEXT for slot=eyebrow, slot=title and
slot=lede; two ACTION lines (slot=primary, slot=secondary); one MEDIA
line with slot=figure carrying EITHER photo="two to four nouns" if the
subject is something a camera can see, OR motif=... if it is not.
The lede is two full sentences, not a fragment.
In the title's text, wrap two or three of its own words in [brackets]."""

CTA_PROMPT = """Emit ONLY the closing section's contents: TEXT for slot=title, TEXT for
slot=lede, and one ACTION with slot=primary, all with sec={sec}.
The title must NOT repeat the hero headline — say the next thing: when
to come, what happens first, or what it costs."""

# There is deliberately no quote call, and `quote` is not on the menu.
#
# The compiler supports kind=quote and renders it well, but a model
# cannot source a testimonial. Asked for one, this 4B produced "Sarah
# Chen, DevOps Lead at RiverScale" — a named person at a named company,
# neither of which exists, attesting to a product they have never used.
# That is not a copy defect to tune away; it is a fabricated
# endorsement, and it would go on a real site under someone's real
# business name.
#
# Unlike the invented figures above, there is no computable ground truth
# to check a quote against: the brief has no customers in it, so every
# possible answer is a fabrication. When a slot cannot be verified even
# in principle, the honest move is to keep it out of the model's reach
# and let a human supply it. kind=quote stays available for
# hand-authored ops, where the words came from a person who said them.

OP_LINE = re.compile(r"^\s*(SITE|THEME|SECTION|TEXT|ACTION|MEDIA|ITEM)\b")

NUMERIC = re.compile(r"\d[\d,.]*")

# A figure plus whatever unit is fused to it ("4K", "12,400") and the
# word that follows ("12,400 reels"), so the model is labelling a thing
# rather than a bare integer.
FIGURE_IN_CONTEXT = re.compile(r"\d[\d,.]*[A-Za-z]*(?:\s+[a-z]+)?")

# Money and clock times, anywhere in any section.
#
# The grounding check was written for `proof`, because that is where the
# fabrication was found — and the FAQ then quietly wrote "£4.20 each",
# "£3.80 per piece", "a £2.50 charge", "call us before 10:45 AM" and
# "between 12:30 PM and 4:00 PM" for a bakery whose brief contains no
# price and no clock time at all. Checking the section where the bug
# turned up, rather than the class of claim, is the same mistake the
# palette function made four times.
#
# These two classes get enforced rather than reported because they are
# the ones a reader ACTS on: someone turns up at 10:45 with £4.20. Other
# invented numbers are surfaced in the run output instead of being
# stripped, since gutting every sentence with a digit in it would take
# the specificity out of the copy along with the errors.
# The second alternative needs the bare-hour form too. Written as
# `\d{1,2}:\d{2}` it required minutes, so "we open at 10am" and "until
# 4pm" went onto a clinic page untouched while "10:45 AM" was stripped —
# the same claim, the same harm, one colon apart.
HARD_CLAIM = re.compile(
    r"[£$€]\s?\d[\d,.]*"
    r"|\b\d{1,2}:\d{2}\s*(?:[ap]\.?m\.?)?"
    r"|\b\d{1,2}\s?[ap]\.?m\.?(?=\W|$)", re.I)


def hard_claims(line: str, brief: str) -> list[str]:
    """Prices and clock times in a line that the brief never stated."""
    ground = brief.lower().replace(" ", "")
    out = []
    for m in HARD_CLAIM.finditer(line):
        tok = m.group(0).strip()
        if tok.lower().replace(" ", "") not in ground:
            out.append(tok)
    return out


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


def read_plan(raw: str) -> list[str]:
    """The middle sections the model asked for, cleaned and made legal.

    Validated rather than trusted, because an invalid plan is not a
    conformance failure worth throwing the run away for — it is one
    line of output the model got loose about. Unknown words are
    dropped, an immediate repeat is dropped (two galleries in a row is
    the same sameness one level up), `note` is capped at one, and the
    length is bounded so the page cannot degenerate into either a
    business card or a scroll.
    """
    words = re.findall(r"[a-z]+", raw.lower())
    out: list[str] = []
    for w in words:
        if w not in KIND_PROMPT or w == "hero" or w == "cta":
            continue
        if out and out[-1] == w:
            continue
        if w == "note" and "note" in out:
            continue
        if w in out and w != "note":
            continue
        out.append(w)
        if len(out) == 5:
            break
    # A page needs a middle. If the model returned nothing usable, fall
    # back to the shape the fixed harness used to hardcode — reported as
    # a fallback, not passed off as a choice.
    if not out:
        return ["detail", "features"]
    # At least one section that carries a drawing. Left to itself the
    # model plans features/faq/roster — all of them text — and the
    # richer copy turned the pages into unbroken columns of prose with a
    # single picture at the top. Which section is *appropriate* is the
    # model's call; that a page needs some artwork below the fold is a
    # property of pages, so the harness holds it.
    if not ({"detail", "gallery"} & set(out)):
        out.insert(1 if len(out) > 1 else 0, "detail")
        out = out[:5]
    return out


def tone_for(kind: str, index: int, planned: list[str]) -> str:
    """Alternate the grounds so consecutive bands do not merge.

    Every section came out `quiet` before, which is why a page with six
    sections still read as one continuous column. Tone is derived rather
    than asked for: it depends only on what is next to a section, which
    is something the harness knows and the model, writing one section
    per call, cannot.
    """
    if kind == "note":
        return "inverted"          # the break should read as a break
    if kind == "proof":
        return "bold"
    # Otherwise lift every other long section onto the panel ground,
    # unless its neighbour is already lifted.
    if kind in {"faq", "gallery", "roster"} and index % 2 == 1:
        return "bold"
    return "quiet"


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
    ap.add_argument("--photo-python", default=r"C:\Python313\python.exe",
                    help="interpreter with certifi, for resolving photos")
    args = ap.parse_args()

    lines: list[str] = []
    noise_total = 0
    raw_log: list[str] = []
    dropped: list[str] = []

    figures = figures_in(args.brief)
    # A plain system prompt, NOT the op grammar. GRAMMAR opens with
    # "emit ONLY op lines, no prose", so asking it for a list of section
    # names underneath that produced `SECTION=features` — the model
    # obeying the system prompt over the question, correctly. The plan
    # is not ops and should not be asked for inside the op language.
    plan_raw = ask(args.model,
                   "You plan the structure of a small marketing website. "
                   "You answer with a list of section names and nothing "
                   "else — no code, no markup, no explanation.",
                   f"""Site: {args.brief}

Figures stated in the brief:
{figures}

First decide what KIND of site this is, then use the shape that kind of
site has. These are the shapes that work:

{GENRES}

Then choose which sections this page needs, in order. It opens with a
hero and closes with a call to action; you are choosing what goes
BETWEEN them. Take THREE OR FOUR, following the shape for this genre
unless the brief clearly wants something else.

{PLAN_MENU}

Answer with the genre on the FIRST line as `genre: <name>`, then three
or four section names, one per line. Nothing else.""")
    plan = read_plan(plan_raw)
    # The genre the model named, matched against the ones we have notes
    # for. An unrecognised genre is not an error — the section prompts
    # simply stay generic, which is the behaviour before this existed.
    known = {g for g, _ in GENRE_NOTES}
    gm = re.search(r"genre\s*[:\-]\s*([a-z ,]+)", plan_raw, re.I)
    # Asked for `genre: portfolio` the model often writes just
    # `portfolio` on the first line, which is a reasonable reading of
    # the instruction and was scoring as "generic". Accept the bare form.
    said = (gm.group(1) if gm else plan_raw.strip().splitlines()[0]
            if plan_raw.strip() else "").lower()
    genre = next((g for g in known if g in said), "")
    if args.debug:
        print(f"      PLAN RAW: {plan_raw.strip()[:300]!r}")
    print(f"  plan       [{genre or 'generic'}] "
          f"hero -> {' -> '.join(plan)} -> cta")

    # (label, kind, section id) for every call after the frame.
    schedule = [("hero", "hero", "s1")]
    schedule += [(k, k, f"s{i + 2}") for i, k in enumerate(plan)]
    schedule.append(("cta", "cta", f"s{len(plan) + 2}"))

    CALLS = [("frame", None, None,
              "Site: {brief}\n\nEmit EXACTLY two lines: one SITE and one "
              "THEME. Nothing else.")]
    for i, (label, kind, sec) in enumerate(schedule):
        body = (HERO_PROMPT if kind == "hero"
                else CTA_PROMPT if kind == "cta"
                else KIND_PROMPT[kind])
        CALLS.append((
            label,
            f"SECTION  id={sec}  kind={kind}  "
            f"layout={KIND_LAYOUT[kind]}  tone={tone_for(kind, i, plan)}",
            sec,
            "Site: {brief}\n\nThe " + kind + " section already exists. "
            + body.replace("{sec}", sec)
            # What this section means in THIS genre, if it differs.
            + ("\n\n" + GENRE_NOTES[(genre, kind)]
               if (genre, kind) in GENRE_NOTES else "")
            + "\nDo NOT emit SECTION. Do NOT emit THEME or SITE again."))
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
        # Naming ONE motif per family fixed a Postgres CLI getting a
        # drawing of a house and immediately created a worse problem:
        # the mapping said editorial -> elevation, and a bakery, a
        # clinic and a film archive are all editorial, so every one of
        # them got the house instead. Half of all figures across twelve
        # pages came out `elevation`. A default is not a suggestion to a
        # small model; it is an instruction.
        #
        # So: a shortlist per family rather than a single answer, and
        # the drawings already on the page are excluded, because the
        # other half of the problem was that ten of twelve pages used
        # the SAME motif for both their figures — the hero drawing and
        # the detail drawing were literally the same picture.
        free: list[str] = []
        if "motif" in user:
            fam = re.search(r"family=(\w+)", "\n".join(lines))
            fam = fam.group(1) if fam else ""
            shortlist = {
                "technical": ["schematic", "orbit", "dial", "strata"],
                "studio": ["specimen", "lattice", "orbit", "topography"],
                "editorial": ["strata", "lattice", "elevation", "topography"],
            }.get(fam, [])
            used = set(re.findall(r"motif=(\w+)", "\n".join(lines)))
            free = [m for m in shortlist if m not in used]
            if free:
                user += (f"\n\nThis site is family={fam}. Choose the motif "
                         f"from these, whichever suits the subject: "
                         f"{', '.join(free)}.")
                if used:
                    user += (f" The page already has a "
                             f"{'/'.join(sorted(used))} drawing — do not "
                             f"repeat it.")
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

        # Told which motifs are free and which is already on the page,
        # the model still reaches for the one it just used. Whether a
        # drawing is a repeat is something the harness can simply look
        # up, so it does, and substitutes the first free alternative.
        if free:
            already = set(re.findall(r"motif=(\w+)", "\n".join(lines)))
            fixed = []
            for line in kept:
                m = re.search(r"motif=(\w+)", line)
                if m and m.group(1) in already:
                    line = line.replace(f"motif={m.group(1)}",
                                        f"motif={free[0]}")
                    print(f"      REPEAT motif={m.group(1)} -> {free[0]}")
                if (m2 := re.search(r"motif=(\w+)", line)):
                    already.add(m2.group(1))
                fixed.append(line)
            kept = fixed
        if stray and args.debug:
            for s in stray:
                print(f"      STRAY: {s[:96]!r}")

        # Prices and times the brief never stated, in ANY section.
        invented = []
        for line in kept:
            bad = hard_claims(line, args.brief)
            if bad:
                invented.append((line, bad))
        if invented:
            kept = [l for l in kept if l not in {i[0] for i in invented}]
            for line, bad in invented:
                print(f"      INVENTED {', '.join(bad)}: {line[:74]!r}")

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
                # A figure set at display size shows its punctuation:
                # the model wrote title="6," and the stats band read
                # "6," in 40px type. Trailing punctuation is never part
                # of a figure.
                line = re.sub(r'title="([^"]*?)[\s,;:.]+"', r'title="\1"', line)
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

    # Resolve any photo= queries BEFORE compiling. Without this the
    # compiler finds no sidecar, every photo falls back to a drawing,
    # and the run looks like a success — the pipeline existed for two
    # commits while nothing generated actually used it.
    #
    # Run under args.photo_python: resolving needs a current CA bundle
    # for hosts whose chain the system store rejects, and certifi lives
    # in the system interpreter rather than this venv.
    if 'photo="' in "\n".join(lines):
        pr = subprocess.run([args.photo_python, str(HERE / "photos.py"),
                             str(ops_path)], capture_output=True, text=True)
        for line in (pr.stdout or "").splitlines():
            print(f"  {line.strip()}")
        if pr.returncode != 0:
            print(f"  photo resolver failed: {(pr.stderr or '')[:160]}")

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
