# asset-craft constitution

Applied to every visual asset (page, component, chart, icon, document)
produced while this pack is mounted. Each rule closes a default that makes
output read as generated rather than crafted.

## Neutrals carry the page; the accent is earned

Five neutrals (background, panel, ink, muted, line) do the work. One accent,
only on interactive elements and deliberate emphasis. Color that is neither
interactive nor semantic is not used. No decorative gradients.

## Both themes, or say which and why

Every asset is built light-and-dark via custom properties with
`prefers-color-scheme` plus an explicit `data-theme` override — or commits
to one theme as a named design decision. Dark mode is its own palette:
no pure black, elevation by lightness, desaturated accents.

## The scale is derived, the spacing is a system

Type sizes come from one base and one ratio. Spacing is multiples of one
unit (4 or 8px), with proximity encoding relationship. One border-radius
family, decreasing with nesting. One shadow style per elevation, one light
source.

## Text is set to be read

`max-width: 65ch` on prose, line-height 1.5+ body and ≤1.25 headings,
4.5:1 contrast floor for body text, sentence case, verbs on buttons,
no lorem in anything shown.

## Interactive means four states

Rest, hover, visible focus (never `outline: none`), disabled. Hit targets
44px on touch. Semantic HTML before ARIA.

## Draft, then critique against the brief, then correct

At least one review pass before anything ships, scanning: brief satisfied;
holds at 360px and 1440px; holds in both themes; none of the corpus's named
generated-look tells present; hierarchy scannable in three seconds.
