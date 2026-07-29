# react-craft constitution

Applied to every component written while this pack is mounted. Each rule
targets a default that produces code which renders correctly on the
first try and then fails under reordering, StrictMode, concurrency or a
slow network.

## Function components, hooks at the top level

Every component is a function. Hooks are called unconditionally at the
top — never inside a condition, a loop, a nested function, or after an
early return. Conditional logic lives inside the hook, not around it.
The one construct that still requires a class is an error boundary, and
that is either a dedicated class or the `react-error-boundary` package.

## Components are declared at module scope

No component is defined inside another component's body — the inner
function is a new type every render and remounts its entire subtree. JSX
is not stored in state; data is stored and elements are described during
render.

## State is replaced, never mutated

Updates produce new objects and arrays — spread, `map`, `filter` — at
every level touched. Deep nesting that makes this awkward is flattened
or moved to `useReducer` rather than mutated in place.

## Updates derived from current state use the functional form

`setX(prev => …)` whenever the next value depends on the previous one,
and always inside async callbacks, timers and event handlers that may
run more than once per render.

## An effect must name the external system it synchronises with

If it cannot, it is not an effect. Values derivable from props or state
are computed during render. Work that happens because the user acted
lives in the event handler. No `useState` plus `useEffect` pair mirroring
derived data.

## Dependency arrays are exhaustive and the lint rule is not silenced

Everything reactive that the effect reads is declared. A loop is fixed by
addressing the value being recreated each render — moving it inside the
effect, wrapping it in `useCallback`, or hoisting it out of the
component — never by deleting a dependency.

## Every effect that starts something returns a cleanup

Subscriptions, timers, listeners and in-flight requests are all torn
down. Fetches abort via `AbortController` or guard with an ignore flag,
so a stale response can never land in state. Code is written to survive
StrictMode's double-invoke rather than to detect and skip it.

## Keys come from the data's own identity

Every element produced by `.map()` carries a `key` taken from a stable
id. The array index is used only for a list that is append-only and
never reordered, filtered or deleted from. A `key` on a component is
also the correct way to reset its state when an identity prop changes,
in place of a resetting effect.

## Inputs are controlled with both value and onChange

Controlled fields initialise to `''` rather than `undefined` or `null`,
so they never switch modes mid-life. `defaultValue` marks a deliberately
uncontrolled field.

## React 19 APIs where they replace older ones

`ref` is an ordinary prop — no `forwardRef` in new code. Context renders
as `<Context value={…}>`. Form submission uses an `action` with
`useActionState`, pending UI uses `useFormStatus` and `useOptimistic`,
and a promise read during render uses `use()` with a cached promise.
`propTypes` and `defaultProps` do not appear; types come from TypeScript
and defaults from default parameters.

## Context values are memoised, and contexts are split by change rate

A context value that is an object literal is wrapped in `useMemo`.
Frequently-changing state does not share a context with stable
configuration, because every consumer re-renders regardless of the field
it reads.

## Memoisation follows a measurement

`React.memo`, `useMemo` and `useCallback` are applied to a profiled
problem, not by default. The first structural remedies are moving state
down and passing `children` as a prop.

## Untrusted values are never markup, and never a URL scheme

`dangerouslySetInnerHTML` appears only with sanitised content and a
comment saying which sanitiser ran. Any `href` or `src` built from input
has its scheme validated against an allowlist before it is rendered.

## Accessible by construction

Actions are `<button type="button">`, navigation is `<a href>`. Every
input pairs with `<label htmlFor>` or carries an `aria-label`. Every
`<img>` has a meaningful `alt` — empty only when genuinely decorative.
Dynamic status messages render into an `aria-live` region, and focus is
managed when content appears or disappears.

## Conditional rendering does not leak a falsy value

`{count > 0 && …}`, never `{count && …}` — a zero-length array renders a
literal `0`. Nested ternaries beyond one level are extracted to a
variable or an early return.
