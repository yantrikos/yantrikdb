# a11y-craft constitution

Applied to every piece of HTML written while this pack is mounted. Each
rule exists because the default output violates it — producing markup
that looks correct in a browser and is unusable with a screen reader or
a keyboard.

Terse on purpose. The reasoning lives in the corpus; these are the
constraints the markup must satisfy.

## Every control is a real control

A thing that is clicked is a `<button>`; a thing that navigates is an
`<a href>`. Never a `<div onclick>` or a `<span role="button">` with a
hand-rolled key handler. A real control is focusable, announces its
role, fires on Enter and Space, and appears in the tab order — four
behaviours you otherwise reimplement and get wrong.

## Every input has a programmatic label

A `<label for>` pointing at the input's `id`, or the input wrapped in
its label. Placeholder text is not a label: it disappears on focus,
fails contrast in most themes, and is not announced as a name by every
screen reader. `aria-label` is the fallback for controls with no visible
text, not the default.

## Inputs declare their purpose

`type` chosen honestly (`email`, `tel`, `url`, `number`, `search`),
`autocomplete` set for anything about the user (`name`, `email`, `tel`,
`street-address`, `postal-code`, `current-password`, `one-time-code`),
and `inputmode` where the keyboard should change. This is the difference
between a form a person can fill on a phone and one they abandon.

## Errors are associated, not just coloured

An invalid field carries `aria-invalid="true"` and
`aria-describedby` pointing at the message element's `id`. The message
says what to do, not merely that something is wrong. Colour alone
carries no information to a screen reader and none to the eight percent
of men with a colour vision deficiency.

## One h1, and headings do not skip

Exactly one `<h1>` per document, then `h2`, `h3` in order with no gaps.
Headings are the document's outline and the primary way non-visual users
navigate — they are structure, never a way to make text big.

## Landmarks wrap the page

`<header>`, `<nav>`, `<main>`, `<footer>`, with exactly one `<main>`.
Multiple `<nav>` elements are distinguished by `aria-label`. A
screen-reader user jumps between landmarks the way a sighted user
scans; a page of `<div>`s offers nothing to jump to.

## Images say what they mean, or say nothing

Informative images get `alt` describing their purpose. Decorative images
get `alt=""` — empty, present, never omitted, because a missing `alt`
makes a screen reader read the filename. Never begin with "image of".

## Keyboard reachable, and visibly so

No positive `tabindex`. Interactive elements are reachable in DOM order,
and `:focus-visible` gets a visible indicator with at least a 2px
outline or ring. Removing focus outlines with `outline: none` and no
replacement makes a page unusable without a mouse.

## Nothing important is conveyed by colour alone

Status, validity, required-ness and selection carry a second signal: an
icon, text, a shape, an underline. Links inside body text are
underlined or otherwise distinguishable without colour.

## Text contrast clears AA

4.5:1 for body text, 3:1 for large text (18.66px bold or 24px), and 3:1
for the visual boundary of controls and focus indicators. This is the
floor, not the target.

## Dynamic changes are announced

Content that appears without a page load — validation summaries, toasts,
search results counts — sits in a container with `role="status"` or
`aria-live="polite"`, present in the DOM before it fills. A live region
added at the same moment as its content announces nothing.

## Dialogs trap and return focus

A modal has `role="dialog"` with `aria-modal="true"`, is labelled by its
own heading via `aria-labelledby`, moves focus in on open, keeps Tab
inside while open, closes on Escape, and returns focus to whatever
opened it. A dialog that leaves focus behind it strands the user in the
page underneath.

## The page declares its language, the document its title

`<html lang="…">` and a `<title>` that names the page before the site.
Language drives screen-reader pronunciation; the title is the first
thing announced and the label in the tab list.

## Tables are data, not layout

A data table has `<th>` with `scope`, and a `<caption>` describing it.
Layout is done with CSS. A layout table read cell by cell is noise.
