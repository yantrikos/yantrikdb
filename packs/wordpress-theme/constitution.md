# wordpress-theme constitution

Applied to every WordPress theme written while this pack is mounted.
Each rule targets a default that produces a theme which either fails to
activate, or activates and cannot be edited.

## Build a block theme unless a classic theme is explicitly requested

`style.css` with a header, `theme.json`, and `templates/index.html`. No
`index.php`, no `header.php`, no `footer.php`, no hand-written loop, no
`wp_head()` / `wp_footer()` calls — templates are block markup and
WordPress renders the document. A theme missing `templates/index.html`
is a classic theme, silently, with no site editor.

## style.css opens with a valid header comment

`Theme Name` at minimum, plus Version, Requires at least, Requires PHP,
License, Text Domain. It sits at the very top, because only the first
8 KB is parsed for it.

## theme.json declares the design, and declares it once

Schema `"version": 3` with `$schema` set. Colours, font sizes, font
families and spacing are **presets**, not literals. `appearanceTools` is
on. `settings.layout.contentSize` and `wideSize` are set, because they
are what makes alignment work at all.

## CSS references presets through their generated custom properties

`var(--wp--preset--color--primary)`, never the hex value again.
`var(--wp--preset--spacing--50)`, `var(--wp--preset--font-size--large)`.
Custom tokens go in `settings.custom` and are read as
`var(--wp--custom--…)`, remembering that camelCase keys become
kebab-case in the property name.

## Never duplicate in PHP what theme.json already declares

No `add_theme_support('editor-color-palette')`, no
`add_theme_support('editor-font-sizes')`, no `$content_width`, no
`register_nav_menus` for a block theme. theme.json wins, and the PHP is
dead code that will drift.

## functions.php stays minimal or absent

At most: enqueue `get_stylesheet_uri()` versioned by
`wp_get_theme()->get('Version')`, `add_theme_support('wp-block-styles')`,
`add_theme_support('custom-logo')`, and `add_editor_style()` if the
stylesheet must reach the editor. Everything else belongs in theme.json
or a plugin.

## Templates use serialised block markup with correct delimiters

`<!-- wp:group -->` … `<!-- /wp:group -->` for blocks with content,
`<!-- wp:post-title /-->` self-closing. Attributes are valid JSON inside
the opening delimiter. Template parts are referenced as
`<!-- wp:template-part {"slug":"header","tagName":"header"} /-->` and
the corresponding file exists in `parts/`.

## Template parts are declared in theme.json

Every file in `parts/` has a matching entry in `templateParts` with
`name`, `title` and an `area` of `header`, `footer` or `uncategorized`.

## Layout type is chosen deliberately

`constrained` for the content column, `default` for a full-bleed
wrapper, `flex` for rows and stacks with `justifyContent`, `grid` with
`minimumColumnWidth` for card grids. A theme does not hand-roll a
`max-width` container — that is what `contentSize` is.

## Full-width sections use root-padding-aware alignments

When the root has horizontal padding, `settings.useRootPaddingAware
Alignments` is true and the padding is declared in
`styles.spacing.padding`, so `.alignfull` can still reach the edges.

## Patterns are files in patterns/ with a namespaced slug

A header comment carrying at least `Title` and `Slug: theme-name/pattern`.
No `register_block_pattern()` call. Any pattern referenced by a template
exists.

## Modern CSS, and no `!important` against core

Fluid type via theme.json `typography.fluid`, or `clamp()` whose
preferred term includes a `rem` component — never viewport units alone,
which breaks browser zoom. Responsive layout via
`repeat(auto-fit, minmax(min(Xrem, 100%), 1fr))` and container queries
rather than viewport breakpoints where the component's own width is what
matters. Logical properties (`margin-inline`, `padding-block`) so RTL
works without a second stylesheet. Global styles are wrapped in
`:root :where(...)` and carry near-zero specificity, so a plain class
selector already wins — `!important` against a block style means the
selector was wrong.

## Accessibility is part of the theme, not a later pass

Visible `:focus-visible` styling with an outline and offset, never
`outline: none` alone. A `prefers-reduced-motion` block. Meaningful
`alt` on decorative-versus-content images. A skip-link target anchor on
the main region. Colour choices that meet 4.5:1 for body text.

## Every user-facing string is translated with a literal text domain

`__()`, `esc_html__()`, `_x()` with the theme's own text domain written
as a literal string, matching the `Text Domain` header.

## Escape on output, in templates and in patterns alike

`esc_html()`, `esc_attr()`, `esc_url()` on anything dynamic that a PHP
pattern or template file prints. A theme is reviewed to the same standard
as a plugin.

## Child themes declare Template and merge rather than restate

`Template:` is the parent's directory name exactly. The child's
theme.json overrides only what changes; it is merged over the parent's,
not substituted for it.

## Assets are local and GPL-compatible

Fonts are bundled and served from the theme, never fetched from a CDN —
that is both a directory rejection and a GDPR exposure. Paths use
`get_theme_file_uri()` or `content_url()`, never a hard-coded
`/wp-content/`.
