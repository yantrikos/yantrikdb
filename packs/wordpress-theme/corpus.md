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


## A complete theme.json for a block theme — worked example

Every design decision in one file, and the shape to imitate. Note what makes it work rather than merely validate: every preset declared under `settings` is *applied* under `styles`; `fluid` is true so the whole scale gets clamps; `contentSize` is 38rem because that is a reading measure rather than a layout width; the palette is four roles named by role; and `elements` plus `blocks` carry the typographic detail. This file scores full marks on a rendered-craft benchmark that measures contrast, measure, type scale, spacing scale and fluid type in a real browser.

```json
{
  "$schema": "https://schemas.wp.org/trunk/theme.json",
  "version": 3,
  "settings": {
    "appearanceTools": true,
    "useRootPaddingAwareAlignments": true,
    "layout": {
      "contentSize": "38rem",
      "wideSize": "72rem"
    },
    "color": {
      "defaultPalette": false,
      "defaultGradients": false,
      "palette": [
        { "slug": "base",      "name": "Base",      "color": "#faf7f2" },
        { "slug": "contrast",  "name": "Contrast",  "color": "#171310" },
        { "slug": "primary",   "name": "Primary",   "color": "#8c3a1e" },
        { "slug": "secondary", "name": "Secondary", "color": "#55635c" },
        { "slug": "tint",      "name": "Tint",      "color": "#eee7dc" }
      ]
    },
    "typography": {
      "fluid": true,
      "defaultFontSizes": false,
      "fontFamilies": [
        {
          "slug": "display",
          "name": "Display",
          "fontFamily": "\"Iowan Old Style\", \"Palatino Linotype\", Palatino, \"Book Antiqua\", Georgia, serif"
        },
        {
          "slug": "text",
          "name": "Text",
          "fontFamily": "\"Charter\", \"Bitstream Charter\", \"Sitka Text\", Cambria, Georgia, serif"
        },
        {
          "slug": "ui",
          "name": "Interface",
          "fontFamily": "\"Inter\", \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif"
        }
      ],
      "fontSizes": [
        { "slug": "small",    "name": "Small",    "size": "0.85rem",   "fluid": { "min": "0.8rem",   "max": "0.9rem" } },
        { "slug": "medium",   "name": "Medium",   "size": "1.0625rem", "fluid": { "min": "1rem",     "max": "1.125rem" } },
        { "slug": "large",    "name": "Large",    "size": "1.33rem",   "fluid": { "min": "1.2rem",   "max": "1.45rem" } },
        { "slug": "x-large",  "name": "XL",       "size": "1.66rem",   "fluid": { "min": "1.45rem",  "max": "1.9rem" } },
        { "slug": "xx-large", "name": "2XL",      "size": "2.08rem",   "fluid": { "min": "1.75rem",  "max": "2.6rem" } },
        { "slug": "huge",     "name": "3XL",      "size": "2.6rem",    "fluid": { "min": "2.1rem",   "max": "3.6rem" } }
      ]
    },
    "spacing": {
      "units": ["rem", "em", "%", "px"],
      "spacingScale": { "steps": 0 },
      "spacingSizes": [
        { "slug": "20", "name": "1", "size": "0.5rem" },
        { "slug": "30", "name": "2", "size": "1rem" },
        { "slug": "40", "name": "3", "size": "1.75rem" },
        { "slug": "50", "name": "4", "size": "3rem" },
        { "slug": "60", "name": "5", "size": "5rem" },
        { "slug": "70", "name": "6", "size": "8rem" }
      ]
    }
  },
  "styles": {
    "color": {
      "background": "var(--wp--preset--color--base)",
      "text": "var(--wp--preset--color--contrast)"
    },
    "typography": {
      "fontFamily": "var(--wp--preset--font-family--text)",
      "fontSize": "var(--wp--preset--font-size--medium)",
      "lineHeight": "1.65",
      "letterSpacing": "0.002em"
    },
    "spacing": {
      "blockGap": "var(--wp--preset--spacing--40)",
      "padding": {
        "top": "0",
        "bottom": "0",
        "left": "var(--wp--preset--spacing--40)",
        "right": "var(--wp--preset--spacing--40)"
      }
    },
    "elements": {
      "link": {
        "color": { "text": "var(--wp--preset--color--primary)" },
        "typography": { "textDecoration": "underline" },
        ":hover": { "typography": { "textDecoration": "underline" } }
      },
      "heading": {
        "typography": {
          "fontFamily": "var(--wp--preset--font-family--display)",
          "fontWeight": "600",
          "lineHeight": "1.12",
          "letterSpacing": "-0.021em"
        }
      },
      "h1": { "typography": { "fontSize": "var(--wp--preset--font-size--huge)", "lineHeight": "1.05" } },
      "h2": { "typography": { "fontSize": "var(--wp--preset--font-size--xx-large)" } },
      "h3": { "typography": { "fontSize": "var(--wp--preset--font-size--x-large)", "lineHeight": "1.2" } },
      "button": {
        "color": {
          "background": "var(--wp--preset--color--primary)",
          "text": "var(--wp--preset--color--base)"
        },
        "typography": {
          "fontFamily": "var(--wp--preset--font-family--ui)",
          "fontSize": "var(--wp--preset--font-size--small)",
          "fontWeight": "600",
          "letterSpacing": "0.04em"
        },
        "spacing": { "padding": { "top": "0.85rem", "bottom": "0.85rem", "left": "1.5rem", "right": "1.5rem" } },
        "border": { "radius": "2px" }
      },
      "caption": {
        "typography": {
          "fontFamily": "var(--wp--preset--font-family--ui)",
          "fontSize": "var(--wp--preset--font-size--small)"
        },
        "color": { "text": "var(--wp--preset--color--secondary)" }
      }
    },
    "blocks": {
      "core/site-title": {
        "typography": {
          "fontFamily": "var(--wp--preset--font-family--display)",
          "fontSize": "var(--wp--preset--font-size--large)",
          "fontWeight": "600",
          "letterSpacing": "-0.015em"
        }
      },
      "core/post-title": {
        "typography": { "fontFamily": "var(--wp--preset--font-family--display)" }
      },
      "core/post-date": {
        "typography": {
          "fontFamily": "var(--wp--preset--font-family--ui)",
          "fontSize": "var(--wp--preset--font-size--small)",
          "letterSpacing": "0.06em",
          "textTransform": "uppercase"
        },
        "color": { "text": "var(--wp--preset--color--secondary)" }
      },
      "core/pullquote": {
        "typography": { "fontFamily": "var(--wp--preset--font-family--display)" }
      }
    }
  },
  "templateParts": [
    { "name": "header", "title": "Header", "area": "header" },
    { "name": "footer", "title": "Footer", "area": "footer" }
  ]
}
```

## A complete templates/index.html — worked example

Block markup, not PHP. The structure to imitate is the section rhythm: a full-bleed tinted band, then a constrained reading column, then a closing band — rather than one undifferentiated stack. Note that the reading column is a group with `{"layout":{"type":"constrained"}}`, which is what makes `contentSize` take effect at all.

```html
<!-- wp:template-part {"slug":"header","tagName":"header"} /-->

<!-- wp:group {"tagName":"main","align":"full","layout":{"type":"default"}} -->
<main class="wp-block-group alignfull">

	<!-- A full-bleed tinted band. Section rhythm: band, then reading
	     column, then band — so the page reads as composed rather than as
	     one long stack. -->
	<!-- wp:group {"align":"full","backgroundColor":"tint","style":{"spacing":{"padding":{"top":"var:preset|spacing|60","bottom":"var:preset|spacing|60"}}},"layout":{"type":"constrained"}} -->
	<div class="wp-block-group alignfull has-tint-background-color has-background" style="padding-top:var(--wp--preset--spacing--60);padding-bottom:var(--wp--preset--spacing--60)">
		<!-- wp:paragraph {"className":"eyebrow"} -->
		<p class="eyebrow">The Journal</p>
		<!-- /wp:paragraph -->

		<!-- wp:heading {"level":1,"style":{"spacing":{"margin":{"top":"var:preset|spacing|20"}}}} -->
		<h1 class="wp-block-heading" style="margin-top:var(--wp--preset--spacing--20)">Notes from the measurement floor</h1>
		<!-- /wp:heading -->

		<!-- wp:paragraph {"style":{"typography":{"fontSize":"var:preset|font-size|large","lineHeight":"1.5"}},"textColor":"secondary"} -->
		<p class="has-secondary-color has-text-color" style="font-size:var(--wp--preset--font-size--large);line-height:1.5">Short pieces on packs, small models, and the difference between a number and a measurement.</p>
		<!-- /wp:paragraph -->
	</div>
	<!-- /wp:group -->

	<!-- The reading column. contentSize does the width; nothing here
	     hand-rolls a max-width. -->
	<!-- wp:group {"style":{"spacing":{"padding":{"top":"var:preset|spacing|60","bottom":"var:preset|spacing|60"}}},"layout":{"type":"constrained"}} -->
	<div class="wp-block-group" style="padding-top:var(--wp--preset--spacing--60);padding-bottom:var(--wp--preset--spacing--60)">

		<!-- wp:query {"queryId":1,"query":{"perPage":6,"pages":0,"offset":0,"postType":"post","order":"desc","orderBy":"date","inherit":true},"layout":{"type":"default"}} -->
		<div class="wp-block-query">
			<!-- wp:post-template {"style":{"spacing":{"blockGap":"var:preset|spacing|50"}}} -->
				<!-- wp:post-date {"format":"j M Y"} /-->

				<!-- wp:post-title {"level":2,"isLink":true,"style":{"spacing":{"margin":{"top":"var:preset|spacing|20"}},"elements":{"link":{"color":{"text":"var:preset|color|contrast"}}}},"fontSize":"x-large"} /-->

				<!-- wp:post-excerpt {"moreText":"Keep reading","excerptLength":42} /-->
			<!-- /wp:post-template -->

			<!-- wp:query-pagination {"style":{"spacing":{"margin":{"top":"var:preset|spacing|60"}},"typography":{"fontFamily":"var:preset|font-family|ui","fontSize":"var:preset|font-size|small"}},"layout":{"type":"flex","justifyContent":"space-between"}} -->
				<!-- wp:query-pagination-previous /-->
				<!-- wp:query-pagination-next /-->
			<!-- /wp:query-pagination -->

			<!-- wp:query-no-results -->
				<!-- wp:paragraph -->
				<p>Nothing published yet.</p>
				<!-- /wp:paragraph -->
			<!-- /wp:query-no-results -->
		</div>
		<!-- /wp:query -->

	</div>
	<!-- /wp:group -->

	<!-- Closing band, with the one button on the page. -->
	<!-- wp:group {"align":"full","backgroundColor":"tint","style":{"spacing":{"padding":{"top":"var:preset|spacing|60","bottom":"var:preset|spacing|60"}}},"layout":{"type":"constrained"}} -->
	<div class="wp-block-group alignfull has-tint-background-color has-background" style="padding-top:var(--wp--preset--spacing--60);padding-bottom:var(--wp--preset--spacing--60)">
		<!-- wp:heading {"level":2,"fontSize":"xx-large"} -->
		<h2 class="wp-block-heading has-xx-large-font-size">Measure it, or it did not happen</h2>
		<!-- /wp:heading -->

		<!-- wp:paragraph {"textColor":"secondary"} -->
		<p class="has-secondary-color has-text-color">Every claim in the journal carries the run that produced it.</p>
		<!-- /wp:paragraph -->

		<!-- wp:buttons {"style":{"spacing":{"margin":{"top":"var:preset|spacing|40"}}}} -->
		<div class="wp-block-buttons" style="margin-top:var(--wp--preset--spacing--40)">
			<!-- wp:button -->
			<div class="wp-block-button"><a class="wp-block-button__link wp-element-button" href="#top">Read the archive</a></div>
			<!-- /wp:button -->
		</div>
		<!-- /wp:buttons -->
	</div>
	<!-- /wp:group -->

</main>
<!-- /wp:group -->

<!-- wp:template-part {"slug":"footer","tagName":"footer"} /-->
```

## A complete parts/header.html — worked example

A header with real content: site title and tagline stacked on the left, navigation on the right, in a `flex` group with `justifyContent: space-between`. No PHP — the site title is `wp:site-title` and the tagline is `wp:site-tagline`.

```html
<!-- wp:group {"tagName":"header","align":"full","style":{"spacing":{"padding":{"top":"var:preset|spacing|40","bottom":"var:preset|spacing|40"}},"border":{"bottom":{"color":"var:preset|color|tint","width":"1px"}}},"layout":{"type":"constrained","wideSize":"72rem"}} -->
<header class="wp-block-group alignfull has-border-color" style="border-bottom-color:var(--wp--preset--color--tint);border-bottom-width:1px;padding-top:var(--wp--preset--spacing--40);padding-bottom:var(--wp--preset--spacing--40)">
	<!-- wp:group {"layout":{"type":"flex","justifyContent":"space-between","flexWrap":"wrap"}} -->
	<div class="wp-block-group">
		<!-- wp:group {"style":{"spacing":{"blockGap":"0.1rem"}},"layout":{"type":"flex","orientation":"vertical"}} -->
		<div class="wp-block-group">
			<!-- wp:site-title {"level":0} /-->
			<!-- wp:site-tagline {"style":{"typography":{"fontSize":"var:preset|font-size|small","fontStyle":"italic","fontWeight":"400"}},"textColor":"secondary"} /-->
		</div>
		<!-- /wp:group -->

		<!-- wp:navigation {"overlayMenu":"mobile","style":{"typography":{"fontFamily":"var:preset|font-family|ui","fontSize":"var:preset|font-size|small","letterSpacing":"0.04em"},"spacing":{"blockGap":"var:preset|spacing|40"}},"textColor":"contrast"} -->
			<!-- wp:page-list /-->
		<!-- /wp:navigation -->
	</div>
	<!-- /wp:group -->
</header>
<!-- /wp:group -->
```

## A complete parts/footer.html — worked example

The footer is where a dark band goes wrong. Setting `backgroundColor: contrast` without also setting `textColor` leaves near-black text on a near-black ground — invisible, and no error anywhere. Note the explicit `textColor` on the group AND on the blocks inside it, plus the `elements.link.color` override so links stay legible against the dark ground.

```html
<!-- wp:group {"tagName":"footer","align":"full","backgroundColor":"contrast","textColor":"base","style":{"spacing":{"padding":{"top":"var:preset|spacing|60","bottom":"var:preset|spacing|60"}},"elements":{"link":{"color":{"text":"var:preset|color|tint"}}}},"layout":{"type":"constrained","wideSize":"72rem"}} -->
<footer class="wp-block-group alignfull has-base-color has-contrast-background-color has-text-color has-background has-link-color" style="padding-top:var(--wp--preset--spacing--60);padding-bottom:var(--wp--preset--spacing--60)">

	<!-- wp:group {"layout":{"type":"flex","justifyContent":"space-between","flexWrap":"wrap","verticalAlignment":"top"}} -->
	<div class="wp-block-group">

		<!-- wp:group {"style":{"spacing":{"blockGap":"var:preset|spacing|20"}},"layout":{"type":"flex","orientation":"vertical"}} -->
		<div class="wp-block-group">
			<!-- wp:site-title {"level":0,"textColor":"base","style":{"elements":{"link":{"color":{"text":"var:preset|color|base"}}}}} /-->
			<!-- wp:site-tagline {"style":{"typography":{"fontSize":"var:preset|font-size|small","fontStyle":"italic"}},"textColor":"tint"} /-->
		</div>
		<!-- /wp:group -->

		<!-- wp:group {"style":{"spacing":{"blockGap":"var:preset|spacing|20"}},"layout":{"type":"flex","orientation":"vertical"}} -->
		<div class="wp-block-group">
			<!-- wp:paragraph {"className":"eyebrow","style":{"typography":{"fontSize":"var:preset|font-size|small","letterSpacing":"0.1em","textTransform":"uppercase"}},"textColor":"tint"} -->
			<p class="eyebrow has-tint-color has-text-color" style="font-size:var(--wp--preset--font-size--small);letter-spacing:0.1em;text-transform:uppercase">Elsewhere</p>
			<!-- /wp:paragraph -->
			<!-- wp:navigation {"overlayMenu":"never","style":{"typography":{"fontFamily":"var:preset|font-family|ui","fontSize":"var:preset|font-size|small"},"spacing":{"blockGap":"var:preset|spacing|30"}},"textColor":"base"} -->
				<!-- wp:page-list /-->
			<!-- /wp:navigation -->
		</div>
		<!-- /wp:group -->

	</div>
	<!-- /wp:group -->

	<!-- wp:paragraph {"align":"left","style":{"typography":{"fontSize":"var:preset|font-size|small"},"spacing":{"margin":{"top":"var:preset|spacing|50"}}},"textColor":"tint"} -->
	<p class="has-text-align-left has-tint-color has-text-color" style="font-size:var(--wp--preset--font-size--small);margin-top:var(--wp--preset--spacing--50)">Built and measured with YantrikDB packs. Licensed GPL-2.0-or-later.</p>
	<!-- /wp:paragraph -->

</footer>
<!-- /wp:group -->
```

## A complete style.css for a block theme — worked example

Only what theme.json cannot express: optical corrections, state styling, and the accessibility floors. Note the negative tracking and `text-wrap` on display type, hover colours derived with `color-mix` so they track the palette, `:focus-visible` with an offset, the `prefers-reduced-motion` block, and explicit `min-height` on inline links so tap targets reach 24px.

```css
/*
Theme Name: Field Notes
Theme URI: https://packs.yantrikdb.com/p/yantrik-wordpress-theme
Author: YantrikDB
Description: The reference block theme for the wordpress-theme pack — an editorial journal built to score full marks on the pack's rendered-craft benchmark, and to be imitated rather than merely obeyed.
Version: 1.0.0
Requires at least: 6.5
Tested up to: 6.8
Requires PHP: 7.4
License: GNU General Public License v2 or later
License URI: http://www.gnu.org/licenses/gpl-2.0.html
Text Domain: field-notes
Tags: block-patterns, full-site-editing, editorial, one-column
*/

/* Everything that can live in theme.json lives there. What remains is
   what theme.json has no expression for: optical corrections, state
   styling, and the accessibility floors. */

/* ── Optical corrections ─────────────────────────────────────────────
   Tracking that suits 16px reads loose at 42px, so display sizes get
   negative tracking; all-caps labels get positive tracking or the
   letters collide. text-wrap does the work that manual line breaks
   used to. */

h1, h2, h3, .wp-block-site-title {
  text-wrap: balance;
}

p, .wp-block-post-excerpt__excerpt {
  text-wrap: pretty;
  hanging-punctuation: first last;
}

.wp-block-post-date,
.eyebrow {
  font-variant-numeric: tabular-nums;
}

/* Optical alignment: a quote's opening mark should hang into the margin
   rather than indenting the first line. */
.wp-block-quote {
  border-inline-start: 2px solid var(--wp--preset--color--primary);
  padding-inline-start: var(--wp--preset--spacing--40);
  font-style: normal;
}

/* ── State styling ───────────────────────────────────────────────────
   Derived from the palette with color-mix, so a style variation that
   redefines `primary` gets matching hovers for free. */

a {
  text-decoration-thickness: 1px;
  text-underline-offset: 0.18em;
  transition: color 120ms ease, text-decoration-thickness 120ms ease;
}

a:hover {
  color: color-mix(in oklab, var(--wp--preset--color--primary) 78%, black);
  text-decoration-thickness: 2px;
}

.wp-block-button__link:hover {
  background: color-mix(in oklab, var(--wp--preset--color--primary) 86%, black);
}

/* Focus must be visible and must not be the only signal. */
:focus-visible {
  outline: 2px solid var(--wp--preset--color--primary);
  outline-offset: 3px;
  border-radius: 1px;
}

/* ── Accessibility floors ────────────────────────────────────────────
   Interactive targets reach 24px in their smaller dimension (WCAG 2.2
   AA, 2.5.8). Nav links are the usual offender: inline text with no
   padding measures the cap height and nothing else. */

.wp-block-navigation .wp-block-navigation-item__content,
.wp-block-post-date a,
.wp-block-read-more,
.wp-block-post-title a {
  display: inline-block;
  min-height: 24px;
  padding-block: 0.3rem;
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}

/* ── Editorial furniture ─────────────────────────────────────────────
   A rule above each entry in the feed, so the list reads as a sequence
   of articles rather than a stack of paragraphs. */

.entry-list > li + li,
.wp-block-post-template > li + li {
  border-block-start: 1px solid color-mix(in oklab, var(--wp--preset--color--contrast) 12%, transparent);
  padding-block-start: var(--wp--preset--spacing--40);
}

.eyebrow {
  font-family: var(--wp--preset--font-family--ui);
  font-size: var(--wp--preset--font-size--small);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--wp--preset--color--secondary);
}

/* Responsive without breakpoints: the track floors at 16rem but the
   inner min() stops it overflowing a narrow screen. */
.card-grid {
  display: grid;
  gap: var(--wp--preset--spacing--40);
  grid-template-columns: repeat(auto-fit, minmax(min(16rem, 100%), 1fr));
}

/* Component-relative layout — the same card lays out differently in a
   sidebar and in the main column without knowing where it is. */
.card-area {
  container-type: inline-size;
}

@container (min-width: 34rem) {
  .card {
    display: grid;
    grid-template-columns: 8rem 1fr;
    gap: var(--wp--preset--spacing--40);
    align-items: start;
  }
}
```

## A complete functions.php for a block theme — worked example

Nearly empty, and the one thing it must do is the thing that is easy to omit: a block theme's `style.css` is NOT enqueued automatically. Without this file every rule in style.css is inert — and the page still looks designed, because theme.json is carrying it, which is what makes the omission so hard to notice.

```php
<?php
/**
 * Field Notes — theme setup.
 *
 * Deliberately almost empty: the palette, type scale, spacing scale and
 * every default style live in theme.json, where the site editor can see
 * them. Duplicating any of that here would create a second source of
 * truth that theme.json silently wins.
 *
 * What cannot live in theme.json is this enqueue. A block theme's
 * `style.css` is NOT loaded automatically — it is a manifest that
 * WordPress reads for the theme header, and its CSS reaches the page only
 * if something enqueues it. The first version of this theme had no
 * functions.php at all, so every optical correction, focus ring, hover
 * state and tap-target rule in style.css was inert. The page still looked
 * designed, because theme.json was carrying it, which is exactly what
 * made the omission hard to notice.
 *
 * @package field-notes
 */

defined( 'ABSPATH' ) || exit;

add_action(
	'wp_enqueue_scripts',
	static function () {
		wp_enqueue_style(
			'field-notes',
			get_stylesheet_uri(),
			array(),
			wp_get_theme()->get( 'Version' )
		);
	}
);

add_action(
	'after_setup_theme',
	static function () {
		// Core block default styles. Everything else a classic theme
		// declared here — colour palette, font sizes, content width,
		// editor styles — is theme.json's job now.
		add_theme_support( 'wp-block-styles' );
		add_theme_support( 'custom-logo', array( 'height' => 48, 'flex-width' => true ) );

		// theme.json reaches the editor on its own; style.css does not.
		add_editor_style( 'style.css' );
	}
);
```
