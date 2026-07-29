# wordpress-theme corpus

Building a WordPress **block theme** — the kind WordPress has shipped since
6.0 — and the CSS that has to coexist with the CSS WordPress generates on
your behalf.

The dominant failure this targets is not a subtle one: asked for a modern
WordPress theme, a model writes a *classic* theme. `index.php`,
`header.php`, `functions.php`, a `wp_head()` call, a hand-rolled loop.
That is a 2014 answer, it is what most training data contains, and on a
current WordPress it produces a theme that works but cannot be edited in
the site editor and ignores every design tool the platform now has.

## A block theme's minimum viable file set

Three files, and only three:

```
my-theme/
  style.css            ← header comment only; can contain no CSS at all
  theme.json           ← design tokens and settings
  templates/index.html ← required; block markup, not PHP
```

`index.php` is **not** required and `functions.php` is optional. If
`templates/index.html` is absent WordPress treats the directory as a
classic theme, silently, and the site editor disappears.

## style.css is a manifest before it is a stylesheet

WordPress reads the theme's identity from the header comment, and parses
only the first 8 KB of the file:

```css
/*
Theme Name: My Theme
Theme URI: https://example.com/my-theme
Author: You
Description: A block theme.
Version: 1.0.0
Requires at least: 6.5
Tested up to: 6.8
Requires PHP: 7.4
License: GNU General Public License v2 or later
License URI: http://www.gnu.org/licenses/gpl-2.0.html
Text Domain: my-theme
Tags: block-patterns, full-site-editing
*/
```
`Theme Name` is the only strictly required field. A missing or malformed
header is why a theme does not appear on the Appearance screen at all.

## theme.json version 3, and what the version means

```json
{
  "$schema": "https://schemas.wp.org/trunk/theme.json",
  "version": 3,
  "settings": { },
  "styles": { }
}
```
The `version` is the **schema** version, not the WordPress version. 3 is
current (WordPress 6.6+); 2 is still read correctly. The most visible v2 →
v3 change is that default font sizes and spacing are no longer inherited
unless opted into, so a v3 file that omits `settings.typography.fontSizes`
gets the theme's own list only.

## Every preset becomes a CSS custom property

This is the single highest-value fact about theme.json. A colour declared
as:

```json
{ "settings": { "color": { "palette": [
  { "slug": "primary", "color": "#2f5d50", "name": "Primary" } ] } } }
```
is emitted by WordPress as `--wp--preset--color--primary`, and is usable
anywhere in your CSS. The naming is mechanical:

```
--wp--preset--color--{slug}
--wp--preset--font-size--{slug}
--wp--preset--font-family--{slug}
--wp--preset--spacing--{slug}
--wp--custom--{path}--{to}--{value}
```
Hard-coding `#2f5d50` in `style.css` after declaring it in theme.json is
the most common way a theme's colours drift out of sync with its editor.

## settings.custom is a free-form token namespace

```json
{ "settings": { "custom": {
  "lineHeight": { "tight": 1.1, "body": 1.6 },
  "shadow": { "card": "0 1px 3px rgb(0 0 0 / 0.12)" } } } }
```
becomes `--wp--custom--line-height--tight` and
`--wp--custom--shadow--card`. Note the transformation: camelCase keys are
**kebab-cased** in the generated property name. `lineHeight` → `line-height`.

## appearanceTools turns on a whole panel of settings at once

`"settings": { "appearanceTools": true }` opts into border, colour link,
spacing (margin, padding, blockGap), and typography (lineHeight)
controls in one flag, instead of setting eight booleans individually. It
is the sane default for a new theme.

## Layout: contentSize and wideSize are what make alignment work

```json
{ "settings": { "layout": {
  "contentSize": "680px", "wideSize": "1200px" } } }
```
These two values are the entire basis of `.alignwide` and `.alignfull`.
WordPress generates the constrained-layout CSS from them; a theme that
writes its own `max-width` container instead will fight that CSS and
alignment will not work in the editor.

## useRootPaddingAwareAlignments, for full-width inside a padded page

```json
{
  "settings": { "useRootPaddingAwareAlignments": true },
  "styles": { "spacing": { "padding": {
      "left": "var(--wp--preset--spacing--50)",
      "right": "var(--wp--preset--spacing--50)" } } }
}
```
Without it, root padding indents full-width blocks too, and edge-to-edge
sections become impossible. With it, WordPress applies the padding to
content while letting `.alignfull` escape it. It only works when the root
padding is declared in `styles.spacing.padding`.

## Fluid typography is a setting, not a clamp() you write

```json
{ "settings": { "typography": {
  "fluid": true,
  "fontSizes": [
    { "slug": "large", "size": "1.75rem", "name": "Large",
      "fluid": { "min": "1.5rem", "max": "2.5rem" } } ] } } }
```
WordPress generates the `clamp()` itself. Setting `"fluid": true` alone
derives min and max from the size; supplying the object controls them.
Hand-writing `clamp()` in a font-size preset works but is invisible to
the editor's size controls.

## Spacing presets, and the scale WordPress generates

`settings.spacing.spacingScale` produces a numbered set —
`--wp--preset--spacing--30`, `--40`, `--50` and so on — or
`spacingSizes` declares them explicitly. Block gap and padding controls
offer exactly these values. A theme that uses raw `rem` values for
section spacing gives the editor nothing to offer the user.

## styles is where the theme actually looks like something

```json
{ "styles": {
    "color": { "background": "var(--wp--preset--color--base)",
               "text": "var(--wp--preset--color--contrast)" },
    "typography": { "fontFamily": "var(--wp--preset--font-family--body)",
                    "lineHeight": "1.6" },
    "elements": { "link": { "color": { "text": "var(--wp--preset--color--primary)" } } },
    "blocks": { "core/heading": { "typography": { "fontWeight": "600" } } } } }
```
`elements` covers link, heading, h1–h6, button, caption, cite.
`blocks` targets any registered block by name. Both accept the same
property shapes as the top level.

## WordPress wraps global styles in :root :where(...) to keep specificity low

Since WordPress 6.1 the generated global-styles CSS uses
`:root :where(.wp-block-quote)` rather than a bare class. The
`:where()` contributes **zero** specificity, so a plain
`.wp-block-quote { … }` in your `style.css` overrides it without
`!important`. Reaching for `!important` against a block style is almost
always a sign the selector was written against the old assumption.

## Templates are block markup in .html files

`templates/index.html` is not HTML in the ordinary sense — it is
serialised block markup:

```html
<!-- wp:template-part {"slug":"header","tagName":"header"} /-->

<!-- wp:group {"tagName":"main","layout":{"type":"constrained"}} -->
<main class="wp-block-group">
  <!-- wp:query {"queryId":1,"query":{"perPage":10,"postType":"post"}} -->
  <div class="wp-block-query">
    <!-- wp:post-template -->
      <!-- wp:post-title {"isLink":true} /-->
      <!-- wp:post-excerpt /-->
    <!-- /wp:post-template -->
  </div>
  <!-- /wp:query -->
</main>
<!-- /wp:group -->

<!-- wp:template-part {"slug":"footer","tagName":"footer"} /-->
```
The comment delimiters are the markup; the HTML between them is a cached
rendering. Self-closing blocks use `/-->`. Malformed delimiters produce
a block-recovery error in the editor rather than a PHP failure, which is
why a theme can look fine on the front end and be broken in the editor.

## The block template hierarchy

In `templates/`: `index.html` (required), then `front-page.html`,
`home.html`, `single.html`, `page.html`, `archive.html`, `category.html`,
`tag.html`, `author.html`, `date.html`, `search.html`, `404.html`,
`singular.html`, `attachment.html`. Specific wins over general and the
resolution order mirrors the classic hierarchy — `single-{post-type}.html`
before `single.html`.

## Template parts live in parts/ and are declared in theme.json

Files go in `parts/header.html` and `parts/footer.html`. Their areas are
declared so the editor groups them correctly:

```json
{ "templateParts": [
  { "name": "header", "title": "Header", "area": "header" },
  { "name": "footer", "title": "Footer", "area": "footer" } ] }
```
`name` matches the filename without extension. Valid areas are `header`,
`footer` and `uncategorized`. A part with no declaration still works but
appears uncategorised in the editor.

## Patterns are PHP files with a header comment

Since WordPress 6.0, anything in `patterns/` is auto-registered:

```php
<?php
/**
 * Title: Hero with heading and button
 * Slug: my-theme/hero
 * Categories: featured, banner
 * Block Types: core/post-content
 * Viewport Width: 1400
 */
?>
<!-- wp:cover {"minHeight":60,"minHeightUnit":"vh"} -->
…
<!-- /wp:cover -->
```
`Title` and `Slug` are required and the slug must be namespaced. No
`register_block_pattern()` call is needed. A pattern referenced by a
template must exist or the template renders empty.

## Style variations are JSON files in styles/

`styles/dark.json` with the same shape as theme.json — `version`,
`settings`, `styles`, plus a `title` — appears in the site editor as an
alternate style for the whole theme. This is how a theme ships light and
dark, or three colourways, without a settings page. Section-level
variations add `blockTypes` and a `slug` to apply to individual blocks.

## functions.php in a block theme should be nearly empty

Everything a classic theme did in `functions.php` — registering menus,
sidebars, editor styles, colour palettes, content width — theme.json now
does. What remains:

```php
<?php
add_action( 'wp_enqueue_scripts', function () {
    wp_enqueue_style(
        'my-theme',
        get_stylesheet_uri(),
        [],
        wp_get_theme()->get( 'Version' )
    );
} );
```
Themes that declare a colour palette in both `add_theme_support(
'editor-color-palette', … )` and theme.json end up with two competing
palettes; theme.json wins and the PHP is dead code.

## Theme supports that still matter for a block theme

Block themes get `title-tag`, `post-thumbnails`, `responsive-embeds`,
`html5` and editor styles implicitly. The ones still worth declaring:

```php
add_theme_support( 'wp-block-styles' );   // core block default styles
add_theme_support( 'custom-logo', [ 'height' => 60, 'flex-width' => true ] );
```
`add_theme_support('align-wide')` is implied by declaring `wideSize` in
theme.json and does not need repeating.

## Enqueue with get_stylesheet_uri, and version by the theme version

`get_stylesheet_uri()` returns the active theme's `style.css` — the
*child* theme's when one is active, which is what you want.
`get_template_directory_uri()` always points at the parent. Passing the
theme version as `$ver` means the browser cache breaks on every release
without a manual cache-buster.

## Child themes need Template, and inherit theme.json

```css
/*
Theme Name: My Theme Child
Template: my-theme
Version: 1.0.0
*/
```
`Template` is the parent's **directory name**, and a wrong value makes
the child fail to activate. A child theme's `theme.json` is merged over
the parent's rather than replacing it, so a child can override one colour
without restating the palette. Since WordPress 5.7 the parent stylesheet
is not auto-enqueued for block themes — check whether you need it at all
before adding an enqueue.

## Loading a stylesheet only for the editor

Block themes get theme.json styles in the editor automatically, but a
`style.css` rule is front-end only unless added:

```php
add_action( 'after_setup_theme', function () {
    add_editor_style( 'style.css' );
} );
```
This is why a theme can look right on the front end and wrong in the
editor — the editor never saw the stylesheet.

## The block class naming convention

Every core block outputs `wp-block-{name}`: `wp-block-group`,
`wp-block-post-title`, `wp-block-columns`. Preset choices become
utility classes — `has-primary-color`, `has-large-font-size`,
`has-background`. Targeting these is stable; targeting the generated
`wp-container-*` layout classes is not, because they are hashed per
layout and change between renders.

## Layout types: constrained, flex, grid

`{"layout":{"type":"constrained"}}` centres children and honours
contentSize/wideSize — the right choice for a page's main column.
`{"type":"default"}` (flow) applies no width constraint.
`{"type":"flex","orientation":"vertical"}` for a row or stack, with
`justifyContent` and `flexWrap`. `{"type":"grid","columnCount":3}` or
`minimumColumnWidth` for auto-fitting grids. Choosing `constrained` for
a header that should be full-bleed is the usual reason a header refuses
to span the viewport.

## Fluid type without theme.json, when you need it in CSS

```css
h1 { font-size: clamp(2rem, 1.5rem + 2.5vw, 3.5rem); }
```
The middle term must include a `rem` component, not `vw` alone —
otherwise the text does not scale when the user changes their browser's
default size, which is a WCAG 1.4.4 failure. `clamp()` with a
viewport-only preferred value is the most common accessibility defect in
modern CSS.

## Layout without media queries

```css
.cards {
  display: grid;
  gap: var(--wp--preset--spacing--40);
  grid-template-columns: repeat(auto-fit, minmax(min(18rem, 100%), 1fr));
}
```
`auto-fit` plus `minmax` gives a responsive grid with no breakpoints, and
the inner `min(18rem, 100%)` is what stops it overflowing on a narrow
screen — without it the track floors at 18rem and the layout breaks below
that width.

## Container queries size a component by its container, not the viewport

```css
.card-area { container-type: inline-size; }
@container (min-width: 30rem) {
  .card { display: grid; grid-template-columns: 12rem 1fr; }
}
```
This is what a media query cannot do: the same card in a narrow sidebar
and a wide main column lays out differently without knowing where it is.
Baseline across all major browsers since early 2023.

## Cascade layers keep theme CSS out of a specificity war

```css
@layer base, components, utilities;
@layer components { .card { padding: 1.5rem; } }
```
Any rule inside a layer loses to any rule outside one, regardless of
specificity — so unlayered WordPress core CSS will beat layered theme
CSS. That makes layers excellent for organising a theme's *own* CSS and
a poor tool for overriding core's.

## Logical properties, because themes get translated

`margin-inline`, `padding-block`, `inset-inline-start`, `border-inline-end`
follow the writing direction, so a theme built with them works in RTL
with no `rtl.css` and no `body.rtl` overrides. WordPress ships RTL
support and a theme using `margin-left` opts out of it.

## :has() replaces the classes themes used to add in PHP

```css
.wp-block-group:has(> .wp-block-cover) { padding-block: 0; }
.wp-block-post-title:has(+ .wp-block-post-featured-image) { margin-block-end: 0; }
```
Styling a parent by its children removes a whole category of
`body_class` filters and conditional wrapper classes. Baseline since
December 2023.

## color-mix() for hover states derived from a preset

```css
.wp-block-button__link:hover {
  background: color-mix(in oklab, var(--wp--preset--color--primary) 85%, black);
}
```
The hover colour tracks the palette automatically, so a user changing the
primary colour in the site editor gets a matching hover without the theme
declaring a second preset. `in oklab` keeps the mix perceptually even —
`in srgb` darkens unevenly across hues.

## Respect prefers-reduced-motion, and do it as an opt-out

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
```
One of the few legitimate uses of `!important` — it must beat whatever
any block or plugin declares.

## Focus visibility is a theme responsibility

Removing an outline without replacing it is the most common accessibility
failure in themes:

```css
:focus-visible { outline: 2px solid var(--wp--preset--color--primary);
                 outline-offset: 2px; }
```
`:focus-visible` shows the ring for keyboard users and not on mouse
click, which is what makes designers accept keeping it. Never
`outline: none` without a visible replacement.

## Screen-reader text and skip links are expected of a theme

WordPress ships the `.screen-reader-text` convention, and the
`wp-block-navigation` skip link relies on the template exposing an id
the link can target — typically `<main id="wp--skip-link--target">` via
the group block's `anchor`. A block theme that never sets an anchor on
its main region silently drops skip-link support.

## Images: aspect-ratio instead of padding hacks

```css
.wp-block-post-featured-image img {
  aspect-ratio: 16 / 9;
  object-fit: cover;
  width: 100%;
  height: auto;
}
```
The featured-image block also accepts `aspectRatio` and `scale`
attributes directly in block markup, which is preferable because the
editor then shows the same crop.

## What WordPress already does, so the theme should not

Core emits: layout CSS from theme.json, block default styles (with
`wp-block-styles`), `wp-image-{id}` sizing, responsive embeds, gap
handling for flex and grid layouts, and `:root :where()` global styles.
A theme reimplementing a container width, a grid gap or a button reset is
usually fighting one of these — and the symptom is a rule that "does not
apply" until it is given `!important`.

## Debugging a theme that will not activate

In order: is `style.css` present with a `Theme Name` header; is the
header inside the first 8 KB; does `templates/index.html` exist for a
block theme; is `theme.json` valid JSON (a trailing comma is the usual
culprit and it fails silently, falling back to defaults); for a child
theme, does `Template` exactly match the parent directory name. Enable
`WP_DEBUG` with `WP_DEBUG_LOG` and read `wp-content/debug.log` — a fatal
in `functions.php` produces a white screen with nothing in the browser.

## Theme check before shipping

The Theme Check plugin runs the directory's automated requirements:
escaping on output, no removed functions, text domain present and
literal, no hard-coded `wp-content` paths (`content_url()` instead), a
`readme.txt` for the directory, and GPL-compatible licensing for every
bundled asset. Bundled fonts in particular must be GPL-compatible and
served locally — linking Google Fonts from a CDN is a GDPR problem and a
directory rejection.

## Declaring a preset is not applying it — this is the defect

The most consequential mistake in a block theme, because everything looks
unstyled and nothing errors. `settings` declares what is *available*;
`styles` declares what is *used*. A theme.json with a beautiful
`fontFamilies` list and no `styles.typography.fontFamily` renders in the
browser default — Times New Roman — and WordPress reports no problem at
all.

```json
{
  "settings": { "typography": { "fontFamilies": [
      { "slug": "body", "name": "Body", "fontFamily": "Source Serif 4, Georgia, serif" } ] } },
  "styles": {
    "typography": {
      "fontFamily": "var(--wp--preset--font-family--body)",
      "fontSize":   "var(--wp--preset--font-size--medium)",
      "lineHeight": "1.6"
    },
    "color": { "background": "var(--wp--preset--color--base)",
               "text":       "var(--wp--preset--color--contrast)" }
  }
}
```

Every preset a theme intends to use as a default needs a matching entry
under `styles`. The checklist is short: font family, font size, line
height, background, text, link colour, and `spacing.blockGap`.

## line-height must be set, because the default is not a number

An unset `line-height` computes to `normal`, roughly 1.2 for most faces —
far too tight for body copy, and the fastest way for a page to look
amateur. Body copy wants 1.5 to 1.7; display sizes want 1.05 to 1.2,
because leading that suits 16px looks like a gap at 48px. Set both:
`styles.typography.lineHeight` for the root, and
`styles.elements.h1.typography.lineHeight` for display.

## A type scale is a ratio, not a list of sizes someone liked

Pick a ratio and derive every step from it: 1.2 (minor third) for dense
interfaces, 1.25 (major third) for general use, 1.333 (perfect fourth)
for editorial contrast. From a 1rem base at 1.25: 0.8, 1, 1.25, 1.563,
1.953, 2.441. Five to seven steps is a scale; fifteen arbitrary values is
a pile, and it shows as a page where nothing feels related to anything
else.

## Make the whole scale fluid, not one size in it

A single `fluid` entry on `large` while everything else is fixed produces
a page that reflows unevenly. Set `settings.typography.fluid: true` so
WordPress derives clamps for every size, then override `fluid.min` and
`fluid.max` per step where you want control. Display sizes should travel
the most between small and large viewports; body copy should barely move
at all — 1rem to 1.125rem is plenty.

## contentSize is a measure, not a layout width

The number that matters is characters per line: 45 to 75 for comfortable
reading, and past about 85 the eye loses its place returning to the left
edge. At a 1.125rem body size that is roughly 34rem to 42rem, not the
1200px a theme reaches for by reflex. Set `contentSize` for reading and
`wideSize` for media and section furniture. A page whose paragraphs run
the full width of a desktop monitor is the most common single reason a
theme looks unconsidered.

## Declaring contentSize does nothing unless the content is constrained

The layout only applies to blocks inside a group whose layout type is
`constrained`. A template that drops post content directly into the
document body ignores `contentSize` entirely — the setting is present,
correct, and irrelevant:

```html
<!-- wp:group {"tagName":"main","layout":{"type":"constrained"}} -->
<main class="wp-block-group">
  <!-- wp:post-content /-->
</main>
<!-- /wp:group -->
```

## Pure black on pure white is a decision not to decide

21:1 is the maximum contrast the medium allows and it reads as glare;
long-form reading is more comfortable a little inside the extremes. Pick
a near-black carrying a hint of the theme's hue and an off-white ground —
`#141210` on `#faf8f5` is about 16:1, comfortably past AA, and looks
composed rather than default. WCAG sets the floor at 4.5:1 for body text
and 3:1 for large text; the floor is not the target.

## A palette needs roles, not favourite colours

Four named roles carry a whole theme: `base` (page ground), `contrast`
(text), `primary` (interactive and emphasis), `secondary` (support). Add
accent steps only when something needs them. Naming by role rather than
by hue — `primary`, not `teal` — is what lets a style variation swap the
entire look by redefining four values, and it is why WordPress's own
themes use exactly this vocabulary.

## Derive states from the palette rather than adding presets

```css
.wp-block-button__link:hover {
  background: color-mix(in oklab, var(--wp--preset--color--primary) 88%, black);
}
a:hover { text-decoration-thickness: 2px; text-underline-offset: 0.15em; }
```

Every interactive element needs a hover and a `:focus-visible`. A link
that changes only its colour on hover fails for anyone who cannot
distinguish those two hues. Underline thickness and offset are the
cheapest way to make a text link feel deliberate.

## Optical corrections are what separate typeset from typed

```css
h1, h2, h3 { letter-spacing: -0.02em; text-wrap: balance; }
p { text-wrap: pretty; }
```

Large text needs *negative* tracking — spacing that suits 16px looks
loose at 48px. `text-wrap: balance` stops a heading leaving one word
alone on its own line; `text-wrap: pretty` prevents orphans in body copy.
All-caps labels need *positive* tracking, around 0.08em, or the letters
collide.

## Section rhythm is what makes a page feel designed

A page of identically-spaced blocks reads as a list. Alternate: a
full-bleed section with a different background, then a constrained
reading column, then a wide media band. In block markup that is
`{"align":"full"}` with a background colour on the group, and vertical
padding from the spacing scale rather than a fixed value:

```html
<!-- wp:group {"align":"full","backgroundColor":"primary","style":{"spacing":{"padding":{"top":"var(--wp--preset--spacing--60)","bottom":"var(--wp--preset--spacing--60)"}}},"layout":{"type":"constrained"}} -->
```

## blockGap is the vertical rhythm of the whole site

`styles.spacing.blockGap` sets the default space between sibling blocks,
and setting it once is worth more than any amount of per-block margin.
Leave it unset and WordPress uses a flat default that makes every
relationship on the page look equally important.

## A header and footer with nothing in them are not a header and footer

`parts/header.html` needs the site title or logo and navigation, laid out
with a `flex` group and `justifyContent: space-between`. The footer needs
at least the site title, a tagline or copyright line, and usually
navigation. A template part containing a bare group renders as empty
space, which is worse than not having one — it reads as a broken page
rather than a simple one.

## Screenshot and readme are what make a theme look finished

`screenshot.png` at 1200x900 is what the Appearance screen shows; without
it a theme appears as a grey placeholder no matter how good it looks in
use. `readme.txt` carries the licence attributions the directory
requires. Neither affects rendering, and both are why a theme feels
unfinished when they are missing.

## A block template .html file is not PHP and is never executed

The reflex a classic theme trains, and the damage is spectacular rather
than subtle. Files under `templates/` and `parts/` are parsed as block
markup and served as HTML. PHP in them is not executed and not stripped —
it is emitted as literal text.

```html
<!-- WRONG: .html files are not PHP -->
<a href="<?php echo esc_url( home_url( '/' ) ); ?>">Home</a>
```

The `<?php` opens no valid tag, so the `href="` attribute never
terminates and the HTML parser swallows everything that follows into it —
including the inline scripts WordPress injects at `wp_footer`, which then
render as visible body text. A page can look catastrophically broken with
no PHP error, no block-recovery warning, and a 200 response.

Use blocks for dynamic values instead. The site title and its home link
are `<!-- wp:site-title /-->`; the tagline is
`<!-- wp:site-tagline /-->`; a home link is `<!-- wp:navigation-link -->`
or a `wp:site-logo`. There is no block for a dynamic copyright year —
write the year as static text, or if it must be dynamic, register a
pattern (patterns *are* PHP) and reference it from the template.

## Patterns are PHP; templates and parts are not

This is the whole distinction. `patterns/*.php` are executed, so
`esc_url( home_url() )`, `__()` and a dynamic year all work there.
`templates/*.html` and `parts/*.html` are not. When a template needs
something only PHP can produce, the answer is a pattern referenced from
the template — never PHP inlined into the HTML.
## Colour is chosen: the palette and ground of a composed theme

Colour choice is what separates a designed theme from a default one, and
the decision is about *restraint* rather than range. Four roles carry a
whole site. The ground is an off-white with a hint of warmth rather than
`#ffffff`; the text is a near-black carrying the same warmth rather than
`#000000`. That pairing lands near 16:1 — far past the 4.5:1 floor, and
without the glare of the 21:1 extreme, which reads as a decision not to
decide.

```json
"palette": [
  { "slug": "base",      "name": "Base",      "color": "#faf7f2" },
  { "slug": "contrast",  "name": "Contrast",  "color": "#171310" },
  { "slug": "primary",   "name": "Primary",   "color": "#8c3a1e" },
  { "slug": "secondary", "name": "Secondary", "color": "#55635c" },
  { "slug": "tint",      "name": "Tint",      "color": "#eee7dc" }
]
```

`tint` exists to make section bands possible without introducing a new
hue. Naming by role rather than by hue is what lets a style variation
restyle the entire theme by redefining five values.

## Applying the palette and type: the styles block that makes design appear

The most consequential few lines in a block theme. Presets declared under
`settings` are only *available*; nothing renders differently until they
are applied under `styles`. A theme that skips this looks unstyled and
reports no error at all.

```json
"styles": {
  "color": { "background": "var(--wp--preset--color--base)",
             "text": "var(--wp--preset--color--contrast)" },
  "typography": { "fontFamily": "var(--wp--preset--font-family--text)",
                  "fontSize": "var(--wp--preset--font-size--medium)",
                  "lineHeight": "1.65" },
  "spacing": { "blockGap": "var(--wp--preset--spacing--40)" },
  "elements": { "link": { "color": { "text": "var(--wp--preset--color--primary)" } } }
}
```

Font family, font size, line height, background, text, link colour, block
gap. Six entries, and their absence is the single most common reason a
structurally correct theme renders in Times New Roman at line-height
normal.

## A type scale from one ratio, applied to headings through elements

Sizes derive from a single ratio so that everything on the page feels
related. Six steps from a 1.0625rem base at 1.25, with `fluid` true so
WordPress generates a clamp for every step rather than for one favourite
size. Display sizes then get their own line height, because leading that
suits body copy looks like a gap at 42px.

```json
"elements": {
  "heading": { "typography": { "fontFamily": "var(--wp--preset--font-family--display)",
                               "fontWeight": "600", "lineHeight": "1.12",
                               "letterSpacing": "-0.021em" } },
  "h1": { "typography": { "fontSize": "var(--wp--preset--font-size--huge)",
                          "lineHeight": "1.05" } },
  "h2": { "typography": { "fontSize": "var(--wp--preset--font-size--xx-large)" } }
}
```

## Sections carry rhythm: alternating full-bleed bands and reading columns

A page of identically spaced blocks reads as a list. Rhythm comes from
alternating the *kind* of section: a full-bleed tinted band, then a
constrained reading column, then a closing band. The band is a group with
`align: full` and a background colour; the vertical padding comes from
the spacing scale rather than a fixed rem.

```html
<!-- wp:group {"align":"full","backgroundColor":"tint","style":{"spacing":{"padding":{"top":"var:preset|spacing|60","bottom":"var:preset|spacing|60"}}},"layout":{"type":"constrained"}} -->
<div class="wp-block-group alignfull has-tint-background-color has-background">
  <!-- section content -->
</div>
<!-- /wp:group -->
```

Note that the band is `align: full` while its *layout* is `constrained` —
the background reaches the viewport edges and the text inside still obeys
`contentSize`. Getting those two the wrong way round produces either a
band that will not span or text that runs the full width of a monitor.

## Layout type is chosen deliberately, one per purpose

Four layout types and each has one job. `constrained` centres children and
applies `contentSize` — the reading column. `default` applies no width
constraint, for a wrapper that must span. `flex` with `justifyContent`
for a header row or a vertical stack. `grid` with `minimumColumnWidth`
for a card set that should reflow without breakpoints.

```html
<!-- wp:group {"layout":{"type":"flex","justifyContent":"space-between","flexWrap":"wrap"}} -->
<!-- wp:group {"layout":{"type":"grid","minimumColumnWidth":"16rem"}} -->
```

Choosing `constrained` for a header that should be full-bleed is the usual
reason a header refuses to span the viewport.

## Optical corrections are applied: tracking and text-wrap

Typeset rather than typed. Large text needs negative tracking, because
spacing that suits 16px reads loose at 42px. `text-wrap: balance` stops a
heading stranding one word on its own line, and `text-wrap: pretty`
prevents orphans in body copy. All-caps labels need positive tracking or
the letters collide.

```css
h1, h2, h3 { letter-spacing: -0.02em; text-wrap: balance; }
p          { text-wrap: pretty; hanging-punctuation: first last; }
.eyebrow   { letter-spacing: 0.1em; text-transform: uppercase; }
```

These belong in `style.css` rather than theme.json, because theme.json has
no expression for `text-wrap` or `hanging-punctuation`.

## Accessibility is built in: focus, motion and tap targets

Three floors, all in the theme's own stylesheet. A visible focus ring with
an offset, never `outline: none` alone. A reduced-motion block, which is
one of the few legitimate uses of `!important` because it must beat
whatever any block or plugin declares. And interactive targets reaching
24px in their smaller dimension, which inline text links do not manage on
their own.

```css
:focus-visible { outline: 2px solid var(--wp--preset--color--primary);
                 outline-offset: 3px; }

.wp-block-navigation .wp-block-navigation-item__content,
.wp-block-post-date a { display: inline-block; min-height: 24px;
                        padding-block: 0.3rem; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition-duration: 0.01ms !important;
                           animation-duration: 0.01ms !important; }
}
```

Screen-reader-only elements are deliberately 1px and are the one exception
to the tap-target floor.

## States derived from the palette, so they track a restyle

Every link and button needs a hover and a focus state, and the colour for
it should be computed from the preset rather than added as a second
hard-coded value. `color-mix` in `oklab` darkens perceptually evenly,
where `srgb` shifts unevenly across hues.

```css
a:hover { color: color-mix(in oklab, var(--wp--preset--color--primary) 78%, black);
          text-decoration-thickness: 2px; text-underline-offset: 0.18em; }

.wp-block-button__link:hover {
  background: color-mix(in oklab, var(--wp--preset--color--primary) 86%, black);
}
```

A style variation that redefines `primary` then gets matching hover states
for free, which is the whole point of deriving rather than declaring.

## A header with real content: title, tagline and navigation

An empty template part renders as blank space and reads as a broken page.
A header needs the site title, usually the tagline, and navigation — laid
out as a flex group pushed apart. No PHP: the title is `wp:site-title` and
the tagline is `wp:site-tagline`.

```html
<!-- wp:group {"layout":{"type":"flex","justifyContent":"space-between","flexWrap":"wrap"}} -->
<div class="wp-block-group">
  <!-- wp:group {"layout":{"type":"flex","orientation":"vertical"}} -->
  <div class="wp-block-group">
    <!-- wp:site-title {"level":0} /-->
    <!-- wp:site-tagline {"textColor":"secondary"} /-->
  </div>
  <!-- /wp:group -->
  <!-- wp:navigation {"overlayMenu":"mobile"} --><!-- wp:page-list /--><!-- /wp:navigation -->
</div>
<!-- /wp:group -->
```

## A footer on a dark band needs its text colour set explicitly

Where dark footers go wrong. Setting `backgroundColor` to the contrast
colour without also setting `textColor` leaves near-black text on a
near-black ground: invisible, HTTP 200, no error anywhere. The group needs
a `textColor`, and links need an `elements.link.color` override, or they
stay the primary colour and disappear against the dark.

```html
<!-- wp:group {"tagName":"footer","align":"full","backgroundColor":"contrast","textColor":"base","style":{"elements":{"link":{"color":{"text":"var:preset|color|tint"}}}},"layout":{"type":"constrained"}} -->
<footer class="wp-block-group alignfull has-base-color has-contrast-background-color has-text-color has-background has-link-color">
  <!-- wp:site-title {"level":0,"textColor":"base"} /-->
</footer>
<!-- /wp:group -->
```

## Strings are translated and output is escaped, in patterns

Templates are HTML and hold no PHP, so this applies where PHP legitimately
lives: `patterns/*.php`, and `functions.php`. Every user-facing string
goes through a translation function with a literal text domain matching
the `Text Domain` header, and anything dynamic is escaped at the point of
output.

```php
<p><?php echo esc_html__( 'The Journal', 'field-notes' ); ?></p>
<a href="<?php echo esc_url( home_url( '/' ) ); ?>"><?php
    echo esc_html( get_bloginfo( 'name' ) ); ?></a>
```

The text domain must be a literal — the extraction tooling parses source
statically and silently skips a variable, so nothing translates and
nothing warns.

## The one thing functions.php must do

A block theme's `style.css` is not enqueued automatically. Without this,
every optical correction, focus ring and hover state in it is inert — and
the page still looks designed, because theme.json is carrying the rest,
which is exactly what makes the omission hard to notice.

```php
add_action( 'wp_enqueue_scripts', static function () {
    wp_enqueue_style( 'my-theme', get_stylesheet_uri(), array(),
                      wp_get_theme()->get( 'Version' ) );
} );
```

Everything a classic theme did here — colour palette, font sizes, content
width, editor styles — is theme.json's job now, and duplicating any of it
creates a second source of truth that theme.json silently wins.

## Ship screenshot.png and readme.txt, or the theme looks unfinished

Neither affects rendering, and their absence is what makes a finished
theme feel like a work in progress. `screenshot.png` at 1200x900 is what
the Appearance screen displays; without it the theme appears as a grey
placeholder however good it looks in use. `readme.txt` carries the licence
attributions the directory requires for every bundled asset, and fonts in
particular must be GPL-compatible and served from the theme rather than a
CDN — which is both a directory rejection and a GDPR exposure.
