# asset-craft corpus

The distance between a competent generated page and a frontier one is not
model size — it is a few dozen craft decisions that strong models make by
default and weaker ones miss the same way every time. Two biases run through
this selection: the defaults that make output *read as generated* (named
explicitly, so they can be refused), and the small numeric rules — scale
ratios, contrast floors, spacing units — that survive being applied
mechanically.

## Typography starts from a scale, not from taste

Pick a base size (16px body) and a ratio (1.2 minor third for dense UI, 1.25
major third for marketing), and derive every heading from it: 16 → 20 → 25 →
31 → 39. Sizes chosen one-off drift into seven near-identical values that
read as noise. Two font families maximum — one for headings, one for body,
or just one with weights. The system stack (`-apple-system, "Segoe UI",
system-ui, sans-serif`) beats a webfont the design didn't specifically need:
zero load cost, native rendering, never a flash of fallback.

## Line-height is inversely proportional to size

Body text wants 1.5–1.6; headings want 1.1–1.25. The single most common
generated-page tell is headline line-height inherited from body: a two-line
headline with daylight between the lines. Set `line-height: 1.15` on
headings explicitly.

## Measure: 45–75 characters, enforced by max-width

Body text wider than ~75 characters per line is measurably harder to read;
narrower than ~45 looks like a sidebar. `max-width: 65ch` on prose
containers, always. A full-bleed paragraph on a 1440px display is the
second most common generated-page tell.

## Optical hierarchy comes from size AND weight AND color, spent sparingly

A hierarchy where everything is bold is flat. Give each level ONE cue
beyond size: headings get weight (600–700), secondary text gets muted color
(not smaller size), labels get letter-spacing and uppercase at 11–12px.
Never all three cues on one element — that is shouting.

## Spacing is a system: one unit, multiplied

Pick 4px or 8px and make every margin, padding and gap a multiple. Related
things sit close (8–12px); separate concerns sit far (32–64px). Proximity
IS the grouping mechanism — if two things are related, closeness should say
so before any border or box does. Uniform 16px-everywhere spacing is how
generated layouts flatten into porridge.

## Whitespace is structure, not absence

Section padding of 64–96px vertical on marketing pages, 24–32px in dense
app UI. When a layout feels cluttered the first move is MORE space between
groups and LESS between members, never smaller text.

## Borders are the last resort for separation

Order of preference: space, then background shift (a card two lightness
steps off the page), then a border. Pages ruled into boxes with 1px lines
on every element read as engineered, not designed. When a border is right,
it is low-contrast: `1px solid` at ~10–15% of the ink color, never pure
black or a saturated hue.

## Neutral-first palette: one accent, earned

Frontier pages are mostly grays: background, panel, ink, muted ink, line —
five neutrals — plus ONE accent used for interactive elements and moments
of emphasis, and semantic red/green/amber used only for meaning. The
generated-page tell is the inverse: saturated color as decoration,
gradients as filler, purple because nothing chose it. If an element is not
interactive and not semantic, it is neutral.

## Dark mode is a palette, not a filter

Never pure black (#000 backgrounds make halation on OLED and crush
shadows); use 10–15% lightness (#131316-class). Elevation in dark mode is
LIGHTER surfaces, not shadows — a raised card is a step up in lightness.
Desaturate accents slightly (a saturated blue on dark vibrates), lift muted
text (what was 45% gray on white needs ~60% light on dark), and keep pure
white text off dark backgrounds — #ececf0-class ink reads better. Wire it
as `@media (prefers-color-scheme: dark)` on CSS custom properties, with an
explicit `[data-theme]` override that wins in both directions.

## Contrast has floors, not vibes

Body text 4.5:1 against its background minimum; large text (18px+ bold or
24px+) 3:1; non-text UI (borders of inputs, icons) 3:1. Muted-gray-on-gray
below these floors is the most common accessibility failure in generated
UI. When in doubt: darker ink, not bigger font.

## Layout: a container, a grid, and fluidity before breakpoints

A centered `max-width: 1100–1200px` container for content pages; CSS grid
with `repeat(auto-fill, minmax(280px, 1fr))` for card collections —
responsive with zero media queries. Fluid type via
`clamp(1.75rem, 4vw, 2.5rem)` on the hero, not five breakpoint rewrites.
Media queries are for LAYOUT CHANGES (sidebar collapses, nav becomes
drawer), not for size tweaking.

## Every interactive element has four visible states

Rest, hover, focus, disabled — and focus is NOT `outline: none`. Hover
shifts one property subtly (background one step, or translateY(-1px));
focus gets a visible ring (`outline: 2px solid accent; outline-offset:
2px`); disabled drops opacity to ~0.5 AND removes the pointer cursor. A
button with only a rest state is the fastest way to make a page feel like
a mockup.

## Border-radius is one decision, applied consistently

Pick a family: sharp (0–2px), soft (6–10px), or round (14px+ / pills), and
use ONE value for like elements. Inner elements never have a larger radius
than their container (a 12px-radius button inside a 4px-radius card looks
wrong in a way viewers feel but cannot name — radius should decrease with
nesting).

## Shadows have one light source and earn their elevation

`0 1px 3px rgba(0,0,0,.08)` for resting cards, `0 8px 30px rgba(0,0,0,.12)`
for overlays — always y-positive (light from above), always low-alpha,
never spread-heavy gray halos. Elements at the same conceptual elevation
share the same shadow. Shadow-on-everything is a generated-page tell;
so is `box-shadow: 0 0 10px` glow.

## Cards are for collections, not for wrapping everything

A card earns its box by being one of several peers (products, posts,
metrics). A single card alone on a page is a div wearing a costume. And
the metric-tile grid — four cards, each an icon + big number + label, all
identical radius and shadow — is THE signature of generated dashboards;
vary structure by information priority instead: the primary number large
and unboxed, secondary metrics in a quiet row.

## SVG-first imagery, consistent stroke

Interface icons are inline SVG (currentColor fill/stroke so they theme for
free), one stroke width family (1.5 or 2), one corner style, sized 16/20/24
on the same grid as text. Mixing outline and filled icon sets in one view
reads instantly as assembled-from-parts. Favicons: an emoji or a single
glyph on a solid rounded square beats a shrunken logo.

## Charts: the comparison chooses the form

Change over time → line. Parts of a whole at one moment → stacked bar (pie
only if ≤5 slices and precision doesn't matter). Distribution → histogram.
Ranked categories → horizontal bars, sorted by value, labeled directly at
the bar end (a legend forces eye travel a label doesn't). Never dual
y-axes — split into two charts. Categorical palettes cap at ~6
distinguishable hues; magnitude gets a sequential ramp of one hue, not a
rainbow. Gridlines at ~10% ink, axis lines optional, chartjunk (3D,
gradients-in-bars, heavy borders) never.

## Numbers in UI are typeset, not just printed

Tabular figures (`font-variant-numeric: tabular-nums`) wherever numbers
stack or update in place, thousands separators, unit suffixes chosen once
(1.2k or 1,200 — not both), and deltas signed with color AND symbol
(▲ +12%, green) because color alone fails a tenth of male viewers.

## Microcopy: sentence case, verbs on buttons, no lorem

Buttons say what they do ("Save changes", not "Submit"; "Delete 3 items",
not "OK"). Sentence case everywhere except acronyms — Title Case On
Everything is a generated tell. Empty states teach the next action, never
just announce emptiness. And lorem ipsum in a deliverable means the work
is not done: write plausible real copy, marked as placeholder in a
comment, not in the visible text.

## Motion is functional and interruptible

150–250ms ease-out for state changes; anything over 400ms is theater.
Animate transform and opacity (compositor-cheap), never width/height/top.
Honor `prefers-reduced-motion: reduce` by collapsing to instant states.
Scroll-triggered entrance animations on content pages are a generated
tell — content should exist when the reader arrives.

## Accessibility is craft, not compliance decoration

Hit targets 44×44px minimum on touch surfaces. Semantic elements first
(`button`, `nav`, `main`, `label for`) — ARIA only where semantics cannot
say it, because wrong ARIA is worse than none. Every image an `alt` that
says the CONTENT ("Q3 revenue up 14%"), decorative images `alt=""`. Focus
order follows visual order. One `h1`, levels never skipped.

## The generated-look checklist, named to be refused

The tells, collected: purple-to-blue gradients as decoration; emoji as
bullet points in interface copy; identical card grids for unequal
information; Title Case Headings Everywhere; full-width text; headline
line-height too open; `outline: none`; uniform 16px spacing; shadow and
radius on every element; centered body text beyond two lines; a rainbow
categorical palette; scroll-entrance animations. A reviewer who scans for
exactly these tells catches most of the gap between generated and crafted
in one pass.

## The critique pass is part of creation

A frontier asset is drafted, then reviewed against the brief with fresh
eyes, then corrected — in that order, at least once. The review asks, in
order: does it satisfy the actual brief; does it hold at 360px and at
1440px; does it hold in both themes; do the tells above appear; is the
hierarchy scannable in three seconds. Shipping the first draft is the
choice to skip the half of the craft that is judgment.
