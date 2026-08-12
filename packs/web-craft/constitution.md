## Page ground and surfaces

Every dark page sits on a deep charcoal ground near oklch(18% .01 260), and every light page on a warm off-white near oklch(97% .005 90); pure #000000 and pure #ffffff are never the page background. Every card or panel stands on a visible surface: a background at least 3% lighter than the page ground, or a 1px hairline border of about oklch(100% 0 0 / 10%). Give every card a border-radius of 6px or more; only full-bleed sections get square corners.

## Colour system

Declare every colour in oklch in one token block at the top of the CSS. Body text sits at least 7:1 contrast against the ground and muted text at least 4.5:1. Define exactly one accent hue, for example oklch(72% .15 250), and use it on links, buttons, focus rings, and small highlights; keep semantic success, warning, and danger colours on separate hues at least 30 degrees away from the accent, near oklch(70% .15 150), oklch(80% .13 85), and oklch(65% .2 25).

## Type scale

Set at least five font sizes on a modular scale stepping by 1.2 or more, anchored by a hero headline that is the largest, boldest element on the page, sized at least clamp(2.5rem, 5vw, 4rem). Display settings differ visibly from body settings: display runs at weight 650–800 with line-height 1.1–1.2, while body runs at 1rem–1.125rem, weight 400, line-height 1.5–1.7. All weights come from the system font stack; do not fake weight with letter-spacing or opacity.

## Measure

Body copy holds to 45–75 characters per line: cap every prose block at max-width 65ch. Anything wider becomes columns, a grid, or a list instead of one long line. Verify by counting the longest paragraph line at a typical desktop width.

## Spacing rhythm

Space elements on a scale of 8, 16, 24, 32, 48, 64, 96, 128 pixels, and take every margin and gap from that list. The gap between a heading and the text that follows it is one step smaller than the gap above the heading. No two adjacent text blocks sit closer than 16px.

## Layout and section separation

Wrap page content in a container 1000px–1200px wide, centred, with at least 20px side padding on mobile. Separate sections by at least 64px of vertical space, a 1px divider, or a background change, so the start of each section is visible at a glance. Open the page with a real hero section containing the headline, one supporting sentence, and one primary action.

## Elevation

Show depth first with background steps and hairline borders, then add shadow only to elements that genuinely float, such as menus and modals. When a shadow appears, keep its blur at least 3x its y-offset and its alpha at 0.25 or less, as in 0 8px 24px oklch(0% 0 0 / .2). Never use pure-black hard shadows with zero blur on resting cards.

## Dark mode as a second design

If the page offers both schemes, design dark mode as its own design with its own token values, not a filter on the light one: drop surface lightness so backgrounds sit near 15%–20%, pull accent and semantic colour lightness up into the 70%–80% range, and swap pure-white text for about oklch(93% .01 260). Switch the entire token set under prefers-color-scheme so no value from the other scheme survives. A single-scheme page is fine; half-adapted colour is not.

## Motion

Animate only state changes — hovers, reveals, toggles — with durations of 120ms–300ms and a named easing like ease-out or cubic-bezier(.2,.7,.3,1); nothing loops or bounces on a content page. Ship a prefers-reduced-motion block that sets animation and transition durations to 0.01ms. Verify the media query exists and names both properties.

## Focus and hover states

Every interactive element carries a hover state that changes background, border, or lightness by a visible step, applied within 150ms. Every focusable element shows a focus-visible outline at least 2px wide, offset 2px from the element, in the accent colour; never remove it without a replacement of equal visibility. The focus ring is always at least as visible as the hover state.

## Real copy

Write real, specific copy: a page title, a headline that names the page's subject, body text that says true things about it, and one real primary action. Use placeholder text or invented facts — fake names, prices, statistics, dates — only when the user supplies them, and then keep them exactly as given. Read the finished page as a stranger: if any sentence could sit unchanged on a different page, rewrite it.

## Self-containment

The page is one HTML file that works offline: link no CDN, stylesheet, script, font, or image URL, and load nothing over the network. Use the system font stack — system-ui, -apple-system, "Segoe UI", Roboto, sans-serif — with no @font-face and no webfonts. Icons come from inline SVG or plain text glyphs, and every colour references the oklch tokens from the top block.

## A finished document

Emit one complete document that closes with the html tag and nothing after it. The document opens with a doctype, an html tag carrying a lang attribute, meta charset, meta viewport, and a title. Before finishing, check the opening and closing tags for html, head, body, and every section.

## Looks to avoid

Do not open with a centred purple-to-pink gradient behind white text, a uniform grid of six identical icon cards, or emoji as bullet points. If a background glow appears, keep it to one, anchored in a corner, at 20% alpha or less. Judge the screenshot in a browser: if it reads as generic AI output, rework the hero and the palette.
