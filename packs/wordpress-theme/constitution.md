# wordpress-theme constitution

Applied to every WordPress theme written while this pack is mounted.
Each rule exists because the default output violates it — producing a
theme that either will not activate, or activates and looks unstyled.

Terse on purpose: this is injected on every request, and the reasoning
behind each rule lives in the corpus where retrieval serves it on demand.

## Block theme, not classic

`style.css` (header comment), `theme.json`, `templates/index.html`. No
`index.php`, `header.php`, `footer.php`, `wp_head()`, or hand-written
loop. Missing `templates/index.html` makes it a classic theme silently,
with no site editor.

## style.css opens with a valid header

`Theme Name` minimum, plus Version, Requires at least, Requires PHP,
License, Text Domain — inside the first 8 KB.

## Declare presets, then APPLY them under styles

The defect that makes a correct theme look unstyled. `settings` says what
is *available*; `styles` says what is *used*. `fontFamilies` without
`styles.typography.fontFamily` renders in Times New Roman and reports no
error. Minimum `styles`: typography fontFamily, fontSize, lineHeight;
color background and text; `elements.link` colour; `spacing.blockGap`.

## line-height is set explicitly

Unset computes to `normal` (~1.2), too tight for body copy. Root 1.5–1.7;
display 1.05–1.2 via `styles.elements.h1`/`h2`.

## One type ratio, whole scale fluid

Five to seven steps from a single ratio (1.2, 1.25, 1.333), never
arbitrary values. `settings.typography.fluid: true` so every step gets a
clamp. Display travels between viewports; body copy barely moves.

## contentSize is a measure, and content is actually constrained

Target 45–75 characters per line — about 34–42rem, not 1200px. It only
applies inside a group with `{"layout":{"type":"constrained"}}`, so
templates wrap their content in one. `wideSize` for media and section
furniture.

## Colour is chosen, not defaulted to the extremes

Near-black on off-white, never `#000` on `#fff` — 21:1 is glare. Four
roles named by role, not hue: base, contrast, primary, secondary, so a
style variation can swap the theme by redefining four values. AA (4.5:1
body, 3:1 large) is the floor, not the target.

## CSS reads presets through their custom properties

`var(--wp--preset--color--primary)`, `var(--wp--preset--spacing--50)`,
`var(--wp--preset--font-size--large)` — never a repeated hex or rem.
`settings.custom` tokens read as `var(--wp--custom--…)` with camelCase
keys kebab-cased.

## Spacing comes from the scale; blockGap is set once

Section padding uses `var(--wp--preset--spacing--N)`, never a fixed rem.
`styles.spacing.blockGap` set at the root is worth more than any
per-block margin.

## Sections carry rhythm

Alternate full-bleed bands (`{"align":"full"}` with a background), a
constrained reading column, and wide media. Identical spacing throughout
reads as a list, not a design.

## Interactive elements get hover and focus

Every link and button declares `:hover` and `:focus-visible`. Hover
colours come from `color-mix()` against a preset so they track the
palette. Never `outline: none` without a visible replacement.

## Optical corrections are applied

`letter-spacing: -0.02em` and `text-wrap: balance` on display headings,
`text-wrap: pretty` on body copy, about `0.08em` positive tracking on
all-caps labels.

## Header and footer contain real content

Header: site title or logo plus navigation in a `flex` group with
`justifyContent: space-between`. Footer: site title, tagline or
copyright, usually navigation. An empty part reads as broken, not simple.

## Never put PHP in a template or part

`templates/*.html` and `parts/*.html` are not executed. PHP in them is
emitted as literal text, which leaves an attribute unterminated and
swallows the rest of the document — a catastrophically broken page with
no PHP error and a 200 response. Dynamic values come from blocks
(`wp:site-title`, `wp:site-tagline`, `wp:navigation`). Only
`patterns/*.php` are PHP.

## Block markup uses correct delimiters

`<!-- wp:group -->`…`<!-- /wp:group -->` with content, `<!-- wp:post-title /-->`
self-closing, valid JSON attributes. Template parts referenced as
`<!-- wp:template-part {"slug":"header","tagName":"header"} /-->` with
the file present in `parts/` and an entry in `templateParts`.

## Layout type is chosen deliberately

`constrained` for content, `default` for full-bleed wrappers, `flex` with
`justifyContent` for rows, `grid` with `minimumColumnWidth` for cards.
Never a hand-rolled `max-width` container.

## Root padding uses root-padding-aware alignments

When the root has horizontal padding, `useRootPaddingAwareAlignments` is
true and the padding is declared in `styles.spacing.padding`, so
`.alignfull` reaches the edges.

## functions.php stays minimal, and never duplicates theme.json

At most: enqueue `get_stylesheet_uri()` versioned by
`wp_get_theme()->get('Version')`, `wp-block-styles`, `custom-logo`,
`add_editor_style()`. No `editor-color-palette`, no `editor-font-sizes`,
no `$content_width`, no `register_nav_menus`.

## Modern CSS, and no !important against core

`clamp()` preferred terms include a `rem` component — never viewport units
alone, which breaks zoom. `repeat(auto-fit, minmax(min(Xrem, 100%), 1fr))`
and container queries over viewport breakpoints. Logical properties
(`margin-inline`, `padding-block`) so RTL needs no second stylesheet.
Global styles are wrapped in `:root :where(...)` and carry near-zero
specificity, so a plain class already wins.

## Accessibility is built in, not added later

Visible focus rings, a `prefers-reduced-motion` block, meaningful `alt`,
a skip-link anchor on the main region, tap targets ≥ 24px.

## Strings are translated; output is escaped

`__()` / `esc_html__()` with a literal text domain matching the header.
`esc_html()`, `esc_attr()`, `esc_url()` on anything dynamic a PHP pattern
or template prints.

## Patterns and child themes follow their conventions

Patterns: files in `patterns/` with `Title` and a namespaced `Slug`, no
`register_block_pattern()`. Child themes: `Template:` is the parent's
directory name exactly, and theme.json merges rather than replaces.

## Ship screenshot.png and readme.txt

1200×900 screenshot, or the Appearance screen shows a grey placeholder.
Readme carries licence attributions. Fonts are bundled locally and
GPL-compatible, never fetched from a CDN.
