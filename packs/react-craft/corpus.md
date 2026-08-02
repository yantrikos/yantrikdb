# react-craft corpus

React as of version 19. Two biases run through the selection: the APIs
that replaced older ones recently enough that a model still reaches for
the old shape, and the bugs that only appear under concurrency,
StrictMode, or a slow network — which is to say, not on the developer's
machine.

## Effects are for synchronising with something outside React

If an effect's only job is to compute a value from props or state,
delete it and compute during render. An effect that sets state from
other state causes a second render pass, can loop, and leaves the
derived value one frame behind. The question to ask is "what external
system am I synchronising with?" — a subscription, a DOM measurement, a
network connection. "Nothing" means it should not be an effect.

## Derived data is computed, and memoised only when measured

```jsx
const visible = useMemo(
  () => items.filter(i => i.status === filter),
  [items, filter]
);
```
Never a `useState` plus `useEffect` pair mirroring something already
derivable. The `useMemo` here is a performance tool, not a correctness
one: it costs a comparison and a slot on every render, so it earns its
place from a profile, not from a habit.

## Event handlers, not effects, respond to interaction

Code that runs *because the user did something* belongs in the handler.
Putting it in an effect keyed on the resulting state makes it run on
mount, on hydration, and again whenever anything else touches that
state. Analytics on click, a POST on submit, a toast on save: all
handlers.

## Never mutate state

`items.push(x)` followed by `setItems(items)` renders nothing — the
reference is unchanged, so React bails out. Replace instead:
`setItems([...items, x])`,
`setItems(items.map(i => i.id === id ? { ...i, done: true } : i))`,
`setItems(items.filter(i => i.id !== id))`. Nested updates need a copy
at every level touched, which is the signal to flatten the shape or move
to `useReducer`.

## Functional updates when the next state depends on the previous

`setCount(count + 1)` called twice in one handler increments by one —
both reads see the same captured `count`. `setCount(c => c + 1)` queues
both. Any update derived from current state uses the function form, and
any update inside an async callback or a timer must, because the
captured value is stale by the time it runs.

## The stale closure

A callback created in one render captures that render's props and state
permanently. Registered once in an effect with `[]` dependencies, it
keeps reading the first render's values while the UI shows the tenth.
The fixes, in order of preference: a correct dependency array, the
functional updater form, or a ref holding the latest value when the
callback's identity genuinely must stay stable.

## Dependency arrays are exhaustive, or the code is wrong

Every reactive value an effect reads — props, state, and anything
derived from them — belongs in the array. Removing a dependency to stop
a loop treats the symptom; the cause is usually a value recreated each
render (an object or function literal) that belongs inside the effect,
inside `useCallback`, or hoisted out of the component entirely.
Silencing the lint rule is how stale-closure bugs reach production.

## Every effect that starts something returns a cleanup

Subscriptions unsubscribe, timers clear, listeners detach, requests
abort. Without cleanup, an effect that re-runs on a dependency change
leaves the previous one alive: two intervals, two sockets, and two
responses racing to set the same state.

## StrictMode double-invokes deliberately

In development, React 18+ mounts, unmounts and remounts each component,
running every effect twice. That is a test, not a bug — an effect that
misbehaves under it has a missing or incorrect cleanup, and it would
misbehave the same way in production when React interrupts and resumes a
render. Do not defeat it with a `hasRun` ref.

## Two fetches race in an effect: the first resolves last and shows stale results

**The bug**: as a user types, request 1 and request 2 both fire, request 1 resolves LAST, and the UI shows its stale results over the newer ones. **The fix**: an `AbortController` aborted in the effect's cleanup, so a superseded request is cancelled rather than allowed to land late. An ignore-flag in cleanup works too; aborting is better because it also stops the network work.

Fast typing fires request 1, then request 2; if 1 resolves last, the UI
shows stale results. Guard with an abort in the cleanup:
```jsx
useEffect(() => {
  const ac = new AbortController();
  fetch(url, { signal: ac.signal })
    .then(r => r.json())
    .then(setData)
    .catch(e => { if (e.name !== 'AbortError') setError(e); });
  return () => ac.abort();
}, [url]);
```
Beyond the simplest case this is precisely what a data library or a
framework loader exists to handle — caching, deduplication and
revalidation are not worth reimplementing per component.

## use() reads a promise or a context during render

React 19 added `use(promise)`, which suspends until the promise settles,
and `use(Context)`, which — unlike `useContext` — may be called
conditionally and inside loops. The promise must come from a cache or a
framework loader: one created during render is a new promise every
render, and the component suspends forever.

## Actions, useActionState and useFormStatus

React 19 lets `<form action={fn}>` take an async function. React manages
the pending state, resets the form on success, and surfaces errors.
```jsx
const [state, formAction, isPending] = useActionState(submit, initialState);
```
`useFormStatus()` reads the enclosing form's pending state from a child,
which is what lets a shared submit button know it is submitting without
prop drilling — it must be called from a component *inside* the form,
not from the one that renders the form.

## useOptimistic for the pending UI

`const [optimistic, addOptimistic] = useOptimistic(actual, reducer)`
shows the intended result immediately and reverts automatically when the
action fails or completes. It replaces the hand-rolled "set local state,
fire the request, roll back in the catch" pattern, and it reverts
correctly because React owns the transition rather than the component.

## ref is an ordinary prop in React 19

Function components receive `ref` in props directly. `forwardRef` is no
longer required and is deprecated; existing code keeps working, but new
components declare `ref` alongside their other props.

## Ref callbacks may return a cleanup function

React 19 allows
`<div ref={node => { const o = observe(node); return () => o.disconnect(); }} />`.
Previously the callback was invoked again with `null` on unmount and
cleanup had to be inferred from that argument. Returning a function is
now the documented shape, and returning anything else is an error.

## Context renders as a provider directly

React 19 allows `<ThemeContext value={theme}>` where
`<ThemeContext.Provider value={theme}>` was required. `.Provider` still
works and is deprecated.

## Context re-renders every consumer, unconditionally

A context whose value is an object literal produces a new reference each
render and re-renders every consumer regardless of which field that
consumer reads. Memoise the value, or split one context into several so
a frequently-changing field does not drag the rest along. Context is
dependency injection, not a state manager — there is no per-field
subscription.

## Keys are stable, unique among siblings, and not the array index

Index keys are correct only for a list that is append-only and never
reordered, filtered or deleted from. Otherwise React reuses the wrong
DOM node and component state attaches to the wrong item — the visible
symptom is checkbox state or typed input jumping rows after a deletion.
Keys need only be unique among siblings, never globally.

## Changing a key remounts a component on purpose

`key={userId}` resets all of a component's internal state when the id
changes. This is the idiomatic replacement for an effect that "resets
state when the prop changes", and it is both cheaper and more correct:
the reset happens in the same render rather than one frame later.

## useState's initialiser: pass the function, do not call it

`useState(expensiveInit())` runs on every render and discards the
result; `useState(expensiveInit)` runs it once. For a value that is not
really state — a mutable box, a stable id, a DOM handle — use `useRef`,
which does not trigger a render when written and is not read during
rendering.

## Controlled inputs need both value and onChange

`value={x}` without `onChange` yields a read-only field and a console
warning. Switching `value` from `undefined` to a string mid-life flips
the input from uncontrolled to controlled and warns — initialise to
`''`, never `undefined` or `null`. `defaultValue` is for a genuinely
uncontrolled field.

## Lift state to the closest common parent, then push it back down

Two siblings needing the same value share it through their common
parent, with the editing child receiving a callback. The inverse matters
just as much: state that has drifted upward and is read by only one
subtree should move back down, because that is the cheapest re-render
optimisation available and it needs no memoisation.

## useTransition keeps the app responsive during an expensive update

`const [isPending, startTransition] = useTransition()` marks an update
as interruptible, so typing stays responsive while a heavy list
re-renders. `useDeferredValue(value)` expresses the same idea as a value
that lags. Both exist for the pattern where a controlled input drives an
expensive derived view.

## Error boundaries are still class components

There is no hook equivalent. Write a class with
`getDerivedStateFromError` and `componentDidCatch`, or use the
`react-error-boundary` package. Boundaries catch errors thrown during
render, in lifecycle methods, and in constructors beneath them — **not**
in event handlers, async callbacks or `setTimeout`, which need their own
try/catch and an explicit error state.

## Suspense handles pending; an error boundary handles failed

`<Suspense fallback={…}>` covers lazy components and `use()`d promises
below it. Placed too high, one slow item blanks a large region; too low
and the page flickers with many spinners. The two boundary types are
complementary and a real subtree usually needs both.

## Server Components run once, on the server, and never hydrate

They may be `async`, may read a database or the filesystem directly, and
ship no JavaScript to the browser. They cannot use hooks, state, effects
or event handlers. `'use client'` marks where interactivity begins, and
everything imported below that boundary enters the bundle — so the
directive belongs on the leaf that needs it, not on the page.

## Props crossing to a client component must be serialisable

Functions, class instances and symbols cannot cross the server-client
boundary. Server functions are the deliberate exception: they cross as
references, not as code. The failure is raised at build or request time
rather than silently, but the message points at the boundary rather than
at the offending prop.

## Document metadata hoists in React 19

`<title>`, `<meta>` and `<link>` rendered anywhere in the tree are
hoisted into `<head>`, so a component can own its own metadata without a
helper library. React 19 also exposes `preload`, `preinit` and
`preconnect` from `react-dom` for resource hinting.

## Memoisation is measured, not sprinkled

`React.memo`, `useMemo` and `useCallback` each cost a comparison and
retained memory. `React.memo` does nothing at all when the parent passes
a fresh object or inline function on every render, which is the usual
case. The common real win is not memoisation but moving state down, or
passing `children` as a prop so the subtree is not recreated. The React
Compiler automates much of this and is still maturing — never assume a
given project has it.

## The two remaining ways a React component can introduce XSS: dangerouslySetInnerHTML and href

JSX escapes interpolated text, so the two ways XSS still gets in are **dangerouslySetInnerHTML** and a **`href`/`src` URL whose scheme is not validated**.

JSX escapes interpolated text, so `{userInput}` is safe.
`dangerouslySetInnerHTML` is not — sanitise with DOMPurify at the point
of render. Separately, `<a href={userInput}>` will happily accept
`javascript:alert(1)`, so the scheme is validated against an allowlist
first. The same applies to `src` and to any `style` value assembled from
input.

## Components are declared at module scope

Never define a component inside another component's body: the inner
function is a new type on every render, so React unmounts and remounts
the entire subtree, destroying its state and its DOM nodes. Hoist it, or
pass it as a prop. Likewise, JSX elements do not belong in state — store
the data and describe the element during render.

## Hooks run in order, so they run unconditionally

No hook inside a condition, a loop, an early return or a nested
function. React identifies hooks by call order, so an early
`return null` above a `useEffect` shifts every later hook by one and
produces errors far from the cause. Conditional logic goes *inside* the
hook, never around it.

## Custom hooks extract logic, not markup

Repeated stateful logic moves into a `useXxx` function that calls the
primitive hooks and returns values — `useDebounced(value, ms)`,
`useMediaQuery(query)`. Each call site gets its own independent state,
which is the difference between a custom hook and a module-level
singleton. A hook that returns JSX should have been a component.

## Conditional rendering, without the falsy-zero trap

Boolean gate: `{isOpen && <Modal />}`. But `{items.length && <List />}`
renders a literal `0` when the array is empty, because `0` is falsy and
React renders numbers. Use `{items.length > 0 && …}` or a ternary.
Nested ternaries beyond one level become an extracted variable or an
early return.

## Accessible by construction

A clickable thing is a `<button type="button">`; something that
navigates is an `<a href>`. A `<div onClick>` is invisible to keyboards
and screen readers, and reconstructing it needs `role`, `tabIndex` and a
key handler — at which point the button was simpler. Inputs pair with
`<label htmlFor>`, icon-only controls carry `aria-label`, and status
messages that appear dynamically live in an `aria-live` region.

## What React 19 removed

`propTypes` and `defaultProps` on function components are gone — use
TypeScript and default parameter values. Legacy context, string refs and
`ReactDOM.render` are removed: the entry point is `createRoot` (or
`hydrateRoot`) from `react-dom/client`. Code still calling
`ReactDOM.render` is written against React 17 semantics and will not
run.
