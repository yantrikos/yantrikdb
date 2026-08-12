## Pick the one idea the page argues for

A page is about one claim, so state that claim as a single sentence before writing any markup and judge every section by whether it argues for the claim. The minimum viable structure is a hero that names the idea, one section of substance or evidence, and one action that follows from it. Sections that merely fill space — a features trio that references nothing, a testimonials band with no names — read as padding and weaken the honest sections around them. The test is deletion: if removing a section would not change the page's one sentence, remove it.

## What makes a hero worth the screen it takes

A hero earns its viewport only if it does three jobs at once: name the subject, say what is different about it, and offer one next step. If the headline could be swapped onto a different product's page and still fit, the hero is decoration, and the rest of the page will read as filler too. The minimum honest hero is a headline of 8–14 words, one supporting sentence under 25 words, and one primary action — that can live in 40–60vh and does not need a full 100vh. A full-screen gradient with centered text saying nothing is the single loudest machine-made tell.

## Write copy that fails the swap test

Specificity is what separates a real page from a template: name the actual subject, use the real numbers the brief provides, and prefer working verbs — "Deploys in 90 seconds" beats "Blazing fast". Every sentence should fail the swap test: if it could sit unchanged on a different page, it is filler and gets rewritten. The floor for a finished page is a title that names the subject, a headline containing the subject, at least one paragraph of 40–80 words about the actual thing, and one action whose label says what it does. The failure looks like "Elevate your workflow with seamless solutions" over a gradient — copy that could be about anything, which means it is about nothing.

## The default looks that read as machine-made

A handful of patterns read as generated on sight: centered hero over a purple-to-pink gradient, six identical icon cards in a row, emoji as bullet points, glow blobs behind everything, headline words like elevate, unleash, or seamless, and dark pages with one neon accent and no mid-tones. Each is the average choice, so the fix is never more decoration but one specific decision: a single accent hue used sparingly, one asymmetric layout move, real nouns in the headline. The minimum repair that works is a left-aligned hero, content of differing sizes instead of an identical icon grid, and a single tinted background instead of a gradient. If three or more tells are present, assume the whole page reads as machine-made and rework the hero and palette first.

## Build oklch ramps with even perceived steps

Build an accent ramp by stepping lightness in equal increments at a fixed hue — say 95, 82, 69, 56, 43 — because oklch lightness tracks perception, so equal steps look even. Taper chroma at the ends: a chroma that looks right at 70% lightness goes neon at 90% and muddy at 30%, so drop it roughly 20–30% on the lightest and darkest steps. Five steps covers most single-accent pages: a tint for backgrounds, the mid step as the accent itself, and shades for hover states and text. Verify by checking adjacent pairs in greyscale; if two steps merge or one jumps out, the ramp is uneven.

## Choose neutrals that belong to the accent

Pure zero-chroma greys sit dead next to any accent, so give every neutral — page ground, borders, muted text — the accent's hue at chroma 0.005–0.02 and the whole page reads as one palette. Keep the tint below the level where it becomes a visible color cast: above about 0.02 chroma on large surfaces starts to look like a deliberate beige or blue wash. The check is to swap the accent hue for its complement: the neutrals should visibly shift with it, and if they do not, they are disconnected. This is the difference between "dark grey page with a blue button" and "a blue page".

## Hierarchy that survives greyscale

Desaturate the page and the reading order must still be obvious: headline first, primary action second, body third. If the order only holds in color, the hierarchy is carried by hue alone, and hue-carried hierarchy is weak even when color is on. Adjacent levels of the hierarchy need at least about 10–15% lightness difference, or a weight step of 150 or more, or a size step of 1.2x or more — and vary at least two of those axes, because one alone is fragile. The failure looks like the eye landing on a colored badge or an accent link before it lands on the headline.

## Proximity versus borders, and when a border is right

Whitespace groups and borders separate, so default to grouping with space: 8–16px inside a group, 24–48px between groups, and no border anywhere proximity alone can hold. A border is actually right where proximity cannot hold: three or more repeated items in a row whose edges matter, tables, form fields, and any card sitting on a same-lightness background. It is also right where content scrolls under a sticky header, because the line marks a boundary that motion would otherwise blur. If a border is removed, the gap across that boundary must grow to at least 1.5x the within-group gap, or the grouping is lost. Both failure modes are common: bordering every block until the page reads as a wireframe, or removing all borders until distinct items blur together.

## Elevation is spent, not applied

Shadows mean "this floats above the page", so spend them only on elements that actually overlap other content: popovers, menus, dialogs, sticky bars. Resting cards read better with a one-step background lift or a hairline border, and a page where every card casts a shadow reads as flat because nothing stands out anymore. When a shadow is earned, the minimum is a y-offset of at least 4px, blur at least 3x the offset, and alpha at or below 0.25, so the light reads as coming from above and the shadow stays quiet. The failure looks like six resting cards each on a 0 10px 30px shadow — the page reads muddy, not deep.

## Dark mode is a second design, not an inversion

Inverting a light design inverts its logic: borders that were darker than surfaces become lighter than them, tinted backgrounds become glowing panels, and large mid-tone areas become voids. Design dark as its own scheme: backgrounds at 15–20% lightness, text around 93% rather than 100%, accent lightness pulled up into the 70–80% range so it does not vibrate against dark grounds, and borders at 8–12% white alpha. light-dark() is the cheapest way to hold both token sets, but it only pays off when every token genuinely has two values — a half-filled list with shared fallbacks produces a scheme that is wrong in both modes. Check both schemes side by side with every text token clearing 4.5:1 in each; one honest scheme beats two half-adapted ones.

## Motion choreography and stagger timing

Motion order communicates structure: the element acted on moves first, its direct children second, and everything else stays still. Stagger repeated items by 40–80ms each with a cap — beyond about 5 items the last one should still begin within roughly 300ms, or the page feels slow rather than choreographed. Durations stay in 120–300ms with ease-out for entrances, and never combine bounce with stagger; pick one gesture. The floor: a single restrained entrance — one 200–300ms fade-and-rise on the hero — beats none at all, because total stillness reads as unfinished rather than disciplined. The failure looks like everything animating on load with 600ms bounces while the user waits a second to read.

## Focus and selection detail

The focus ring is furniture, not ornament: a 2px accent outline offset 2px, shown on :focus-visible only, so keyboard users get it and mouse users do not. Go beyond the minimum on dark grounds, where an accent-colored ring on a similar-lightness button disappears; give the ring a light outer edge or a double offset so it survives. Tint ::selection too — the accent at 70–80% alpha, or a lightness that keeps selected text at 4.5:1 or better — because default selection blue on a tinted page looks like an accident. Verify by tabbing through every interactive element; if the ring vanishes anywhere, that element fails.

## Fluid sizing with clamp

clamp() earns its keep when the fluid middle term is viewport-linked and both ends are real limits: the floor is the smallest size that still reads at a 320px screen, the ceiling is where the size stops adding emphasis. The fluid range should span viewports that actually exist — roughly 360px to 1200px — so a heading like clamp(2.5rem, 2rem + 3vw, 4rem) does its work in the middle and rests at its ends. Use clamp for the hero headline, section headings, and section padding; keep body text fixed at 1rem–1.125rem, because fluid body copy that dips under 16px on phones is a readability regression. The failure looks like clamp(1rem, 4vw, 3rem), where the floor never engages and the ceiling is arbitrary.

## Container queries for components

Reach for container queries when one component renders at genuinely different widths — the same card in a 1200px grid and in a 300px sidebar — so it can switch layout on its own width instead of the viewport's. The tradeoff is debuggability: layout now depends on an ancestor's size, so the component can legitimately look different in two places on one page, which is correct but harder to reason about; reserve them for repeated components, not the page shell. Give the component its breakpoint where it actually breaks — typically 400–480px for a card switching from stacked to side-by-side — not a copy of the page breakpoints. The failure looks like a two-column card interior squeezed into a narrow rail because the rule keyed off the viewport.

## Subgrid for aligning card interiors

Use subgrid when repeated cards must align their insides across columns — titles on one row, bodies on the next, buttons on a shared baseline — because equal-height cards alone do not align internal seams. Reach for it in any grid of three or more cards with varying content lengths, which is exactly where per-card alignment falls apart. The tradeoff: rows lock to the tallest content, so one 200-character description stretches every card's body row — usually the right call, but check your longest realistic content first. The failure looks like card buttons floating at different heights mid-card because each card aligned itself independently.

## Tabular figures wherever numbers stack

Anywhere numbers stack or change — tables, prices, timers, stats that count up — turn on font-variant-numeric: tabular-nums so digits take fixed widths and columns do not wiggle as values update. The tradeoff is cosmetic only: tabular digits are slightly less even in running prose, so scope the rule to the elements showing numbers rather than body text. The minimum coverage is every table with a numeric column, every animated counter, and any price or percentage list. The failure looks like a counting-up stat whose trailing text crawls sideways each frame, or decimal points drifting out of alignment down a table.

## Empty and edge states are designed states

Every list, card grid, or results area needs a designed empty state, because the first load is the empty state. The minimum empty state is one sentence saying what will appear here plus one action that starts it — "Add your first project" beats "No items found". Also design the long-content edge: an 80-character title and a 40-character name must not break the layout, so give card titles a 2–3 line clamp and let body text wrap. The failure looks like a blank box where content should be, and a layout that survives lorem ipsum but breaks on the first real entry.

## text-wrap: balance for headings, pretty for prose

text-wrap: balance belongs on headings and short display text of two to six lines — it evens line lengths and kills the one-word last line that makes headings look unedited, and beyond about six lines it silently does nothing. text-wrap: pretty belongs on prose paragraphs, where it prevents single-word last lines; do not balance body copy, where it works against readability. The minimum is balance on every h1 and h2 and pretty on every paragraph — both are free and neither can break layout. The failure looks like a 30-character heading balanced into three equal stubs that read as a centered poem.

## starting-style for entrance transitions

starting-style is the right tool for entrance transitions — an element fading or rising as it appears — because it covers the first-render case plain transitions miss, including dialogs and popovers that previously could not transition in at all. Reach for it when an element appears through a state change — a dialog opening, an item joining a list, a details block expanding — and give each entrance 150–250ms of opacity plus a small translate with ease-out. The tradeoff: it only fires when an element becomes rendered, so elements already in the DOM at load need a class toggle to animate; do not build a whole page-load choreography on it. Cap entrances at two or three distinct moments; one well-timed entrance is the floor and enough, while the failure looks like every section popping in on load.

## popover for transient UI

Use popover for anything transient that overlays content — menus, tooltips, filter panels — because light dismiss, top-layer stacking, and Escape handling come free, and those three behaviors are exactly what hand-rolled versions get wrong. The tradeoff is placement: top-layer elements escape normal flow, so positioning near the trigger and any pointer arrow are your problem. Keep the surface small — 320px wide at most, 8–16px padding — and pair the open with a 120–150ms entrance so it does not blink into existence. The failure looks like using popover for content that belongs in the page, or a menu that opens covering its own trigger.

## :has for parent-aware styling

:has is for styling a container based on its contents: highlight a form row containing an invalid field, dim a card whose checkbox is checked, condense a label once its input has a value. Reach for it when the state lives in the content but the styling belongs to the container, because it replaces a JavaScript observer for exactly these cases. The tradeoff is readability: the dependency hides inside the selector, so a page full of nested .card:has(...) rules becomes impossible to predict — limit yourself to one to three :has rules and comment each. The failure looks like re-implementing what a class toggle already did, or chaining :has inside :has until nobody can say what matches.

## Scroll-driven animation, sparingly

Scroll-driven animation is right for progress-tied effects — a reading progress bar, a section reveal, a 10–20px parallax — because it replaces scroll listeners and stays smooth under load. Zero such effects is an acceptable answer, but on a page with long scrolling content one subtle reveal is the minimum that keeps it from feeling static, and two is the ceiling. Keep amplitudes small: translate at most 10% of the element's height, opacity no lower than about 0.6 while readable, and never scroll-control anything the user must read — text at full opacity by center screen. Ship a static fallback for non-supporting browsers and kill every scroll effect inside prefers-reduced-motion. The failure looks like headings sliding sideways as the user scrolls, or a hero shrinking 30% and dragging the layout with it.

## field-sizing for textareas that grow

field-sizing: content on textareas removes both the scrollbar-in-a-box look and the fixed-rows guess, because the field grows with what the user types. Use it anywhere a user writes more than one line — comment boxes, description inputs — with a min-height of 2.5–3em so an empty field still reads as a field, and a max-height so it cannot become a page within the page. The tradeoff is layout shift while typing, so place growing fields at the end of a form group or leave clear space beneath. The failure looks like a mid-form textarea pushing the submit button down one line per keystroke.

## content-visibility for long pages

content-visibility: auto pays off on long pages — five or more sections, or any page with heavy below-the-fold content — because it skips render work until the user approaches. Scope it to below-the-fold sections only, and always pair it with contain-intrinsic-size set to the section's approximate height so the scrollbar does not jump as content materializes. The tradeoff: a size estimate wrong by more than about 20% produces visible pop-in and scrollbar jumps, and find-in-page or anchor links into skipped content can misbehave, so never apply it to the hero, nav, or anything interactive. The failure looks like a scrollbar that grows as you scroll, or a section flashing into place a moment too late.

## Cascade layers to keep overrides honest

Cascade layers pay off the moment a page has both defaults and overrides: put resets and element defaults in a base layer and component rules above it, so overrides win without !important or specificity wars. In a single-file page the benefit is modest but real: element selectors stay cheap in the base layer and any class rule above beats them regardless of specificity. The minimum sensible use is two layers — base and components — and more than three in a one-page site is ceremony. The failure looks like defining layers but still writing high-specificity selectors inside them, so the layers change nothing.
