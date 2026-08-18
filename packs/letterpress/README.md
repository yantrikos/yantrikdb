# letterpress

A site compiler a small model can drive. The model never writes HTML or
CSS — it emits lines in a seven-operation language, and this compiler
turns them into one self-contained page.

```
pip install yantrik-letterpress
letterpress compile site.ops --out site.html --strict
```

## Why the split is where it is

Measured, not assumed. The same local model, the same briefs, the same
number of calls and the same design guidance in prose, writing HTML
directly, **failed every render gate on every brief**. Writing the whole
page in one call instead of five failed every one again. Both arms were
handed *"body text must reach at least 4.5:1 contrast"* verbatim, and
both produced text at 1.00:1 — the same colour as its own background.

A model can assert a contrast ratio. It cannot evaluate one. So the
split follows the constraint: judgement to the model, computation to the
compiler.

The compiler owns the type scale, the twelve-column grid, the entire
palette derived from a single accent hex, contrast solving, responsive
collapse, eight parametric SVG drawings, and licensed photograph
resolution. The model owns what the page says.

## The language

```
SITE     name="Tide Mill"  nav="Bread,Visit"  location="Bristol"  contact="hello@tidemill.co"
THEME    family=editorial  accent=#b8442f  mode=light  density=roomy
SECTION  id=s1  kind=hero  layout=split  tone=quiet
TEXT     sec=s1  slot=title  text="Bread [worth the walk]"
TEXT     sec=s1  slot=lede   text="Two ovens and no cafe. We sell out by noon."
ACTION   sec=s1  slot=primary  label="Collection times"  href="#times"
MEDIA    sec=s1  slot=figure  photo="sourdough bread bakery"
ITEM     sec=s2  slot=item  title="Tuesday"  body="Rye and caraway, forty loaves."
```

Section kinds: `hero`, `features`, `proof`, `detail`, `faq`, `roster`,
`note`, `quote`, `gallery`, `cta`. Families: `editorial` (magazine),
`studio` (poster), `technical` (specification) — and they differ in
*composition*, not only in typeface.

## Photographs

`MEDIA` takes `photo="two to four nouns"` and the compiler resolves it
against openly licensed images, accepting only licences that permit
commercial use and modification, and rendering the credit for those that
require attribution.

The model never writes a URL. It cannot know that one exists, is
licensed, or shows what it claims.

```
letterpress photos site.ops     # resolve first
letterpress compile site.ops --out site.html --strict
```

## Verifying

```
pip install 'yantrik-letterpress[verify]'
python -m playwright install chromium
letterpress shoot site.html
```

Renders at 1440 and 390 and gates on content, horizontal overflow,
contrast (composited through every ancestor opacity, on every element
that owns text), readable size (scaled by the real screen transform, so
SVG labels are measured as a reader sees them), exactly one `h1`, and
text that actually paints.

## The knowledge pack

This compiler is one half of a kit. The other half is the `letterpress`
knowledge pack, which teaches a model the language, the genre
conventions, and the refusals — never invent a price, a clock time, a
figure, or a testimonial. The compiler works without it; a model driving
the compiler is much better with it.

The pack is distributed separately, versioned and signed, and names the
compiler version it was written against.

## Licence

Apache-2.0.
