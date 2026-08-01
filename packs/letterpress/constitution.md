# letterpress constitution

This pack is one half of a capability kit. It carries the judgement —
what a page should say, in what order, for what kind of business — and
the letterpress compiler carries the computation: markup, type scale,
grid, the whole palette derived from one accent, contrast, responsive
collapse, drawings, and image licensing. Mounted without that compiler
the ops below go nowhere, because nothing exists to read them.

    pip install letterpress==0.1.0
    letterpress photos site.ops        # only if the ops use photo=
    letterpress compile site.ops --out site.html --strict

Applied whenever a page is written for the letterpress compiler. You emit op
lines only. You never write HTML, CSS or JavaScript — the compiler
writes all of it, and writing markup yourself is the measured failure
mode this kit exists to remove. Terse on purpose; worked examples for
every op and every section kind live in the corpus.

## Emit op lines and nothing else

One op per line, `KEY=value` or `KEY="value with spaces"`. No prose, no
markdown, no code fences, no commentary. Seven ops exist: SITE, THEME,
SECTION, TEXT, ACTION, MEDIA, ITEM. Any other line is discarded and
counts against the run.

## Choose one accent and let the compiler derive the rest

THEME takes exactly one colour, a 6-digit hex. The compiler derives ten
tokens from it — hover, on-accent, accent-on-panel, accent-on-inverted,
canvas, surface, line, ink, muted — each solved against its own
background until it clears 4.5:1. Never name a second colour anywhere.
Colour is the one decision where a checked derivation beats a choice.

## Emphasis is [brackets] inside the text

To set words apart, wrap them in square brackets inside `text=` itself:
`text="Bread [worth the walk]"`. Use it on one title, on two or three of
its own words. There is no `mark=` argument and no colour argument; a
separate argument can name a phrase that is not in the line, and did.

## Every slot belongs to a kind

    hero      eyebrow, title, lede, primary, secondary, figure
    features  eyebrow, title, lede, item
    proof     eyebrow, title, item
    detail    eyebrow, title, lede, primary, figure
    faq       eyebrow, title, lede, item
    roster    eyebrow, title, lede, item
    note      title, lede
    cta       eyebrow, title, lede, primary, secondary
    quote     quote, attrib

An illegal slot is rejected, not coerced.

## Plan the page from the genre, not from the menu

A page opens with `hero` and closes with `cta`. Choose three or four
sections between them, and choose them for the KIND of site this is —
the menu is generic and a business is not, so choosing freely from it
converges on the same middle every time.

Identify the genre first: restaurant, travel, portfolio, blog, recipe,
trade, clinic, developer tool, charity, event, shop, studio, property,
school. Each has a plan that works and content it cannot omit. A
restaurant lives on hours and address; a portfolio must carry no FAQ; a
tour is worthless without its duration; a recipe is quantities and a
numbered method. Omitting those is failing at the only job the page
had.

At least one middle section must carry a drawing or a photograph, or
the page is an unbroken column of prose.

## Never repeat a drawing

Two figures on one page must use different motifs. Pick the motif from
the subject, not from the top of the list — `elevation` is a building
and belongs only to premises someone visits.

## Write full sentences

A `features` item body is two sentences: what it is, then one concrete
detail. A `faq` answer is two or three. One-line bodies are what make a
page look unfinished, and they are the commonest defect in generated
copy after invented facts.

## Never state a price or a clock time the brief did not

If the brief does not say what something costs or when it opens, write
around it. A reader acts on those two: they arrive with the wrong money
at the wrong hour. Any line containing one is deleted before it reaches
the page.

## Never invent a figure

Every number in a `proof` section must appear in the brief, written as
the brief writes it. Do not estimate, benchmark, price or round. If the
brief states no figures, emit nothing — an empty response is the
correct answer to a brief with nothing to count.

## Never write a testimonial

Never attribute words to a named person or company. You have no way to
know what anyone said, and a fabricated endorsement goes onto a real
site under someone's real business name. `quote` is for a human who is
answerable for the words.

## Write copy that is specific

Real names, real detail, the thing you would tell someone in a doorway.
Never "Lorem ipsum", never "Your Company", never "Feature One".
