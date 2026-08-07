# motion-craft constitution

Applied to every piece of web motion written while this pack is mounted —
CSS transitions, keyframe animations, and Web Animations API calls.
Each rule exists because the default output violates it: animating
layout properties, easing everything linearly, ignoring users who asked
for less motion. Terse on purpose; the reasoning lives in the corpus.

## Animate compositor properties only

`transform` and `opacity` carry every entrance, exit, and emphasis.
Never animate `top`, `left`, `width`, `height`, `margin`, or `padding` —
each forces layout on every frame and stutters on mid-range phones.
Movement is `translate`, growth is `scale`, never a changing offset.

## Reduced motion is honored, always

Every stylesheet that animates carries
`@media (prefers-reduced-motion: reduce)` and inside it either disables
the animation or collapses it to a fade. A vestibular-disorder user who
set that flag gets no sliding, scaling, or parallax. This is not
optional polish; it ships in v1 or the motion does not ship.

## Nothing UI-visible is linear

`linear` is for spinners, marquees, and progress — motion with no start
or end. Entrances decelerate (`ease-out` or a cubic-bezier with fast
start), exits accelerate (`ease-in` shape), moves between states use
ease-in-out shapes. A card that arrives at constant velocity reads as
mechanical; nothing in the physical world moves that way.

## Durations live in bands

Micro-feedback (hover, press, toggle) 100–200 ms. Entrances and exits
150–400 ms. Larger transitions (modal, page, accordion) 250–600 ms.
Ambient loops 2 s or slower. Anything UI-blocking over 700 ms is the
interface making the user wait for a performance.

## Exits are faster than entrances

An element leaving carries information the user already processed;
get it out in roughly two-thirds of its entrance time. Symmetric
in/out durations read as sluggish dismissal.

## Stagger, don't dogpile

When a list or grid enters, siblings are offset by 30–80 ms each
(`animation-delay` or `transitionDelay` stepped per index), capped
around 8 items — after that, the remainder enters together. Twelve
cards arriving at once is a wall; twelve arriving over a second is
a queue the eye can follow.

## Transforms declare their origin

Anything that scales or rotates sets `transform-origin` to the edge or
point the motion grows from — a dropdown scales from `top`, a
context-menu from its corner, a pressed button from `center`. The
default center origin on a dropdown makes it inflate like a balloon
instead of unfolding from its trigger.

## will-change is a scalpel

Applied to the one or two properties about to animate, on the elements
that animate, ideally just before the animation and removed after.
Never on `*`, never left on permanently, never as a blanket "make it
fast" — each use pins a compositor layer and eats memory.

## Transitions name their properties

`transition: transform 200ms ease-out, opacity 200ms ease-out` — never
`transition: all`. `all` animates properties you did not consider,
including ones a later refactor adds, and turns every style change into
an unplanned animation.

## Entrances settle, exits complete

A keyframe entrance ends at the element's resting state (`opacity: 1`,
`transform: none` or identity) and applies `animation-fill-mode` so the
element does not snap when the animation ends. An exit animation is
paired with actually removing or hiding the element when it finishes,
not left at opacity 0 still swallowing clicks.

## Loops are ambient, never anxious

Infinite animations are reserved for genuine ambience — a slow float, a
breathing glow, a spinner. They run 2 s per cycle or slower, move
subtly (a few px or a few % of scale), and never touch content the user
is reading. Text that pulses while being read is hostile.

## Motion means something

Every animation encodes one of: origin (where a thing came from),
hierarchy (what caused what), feedback (the press registered), or
continuity (this is the same object, moved). Decoration that encodes
none of these is removed. If two elements animate simultaneously with
equal weight, the motion says nothing; pick the one the user should
watch.
