# java-modern constitution

Applied to every piece of Java written while this pack is mounted. Each
rule targets a default the model would otherwise produce: Java 8 idioms
where a current LTS has a better construct, or a silent contract
violation.

## Data carriers are records

A class whose purpose is to hold values is a `record`. Validation and
defensive copying (`List.copyOf`) go in the compact constructor. Manual
`equals`/`hashCode` pairs are not written by hand for value types.

## Closed hierarchies are sealed and switched exhaustively

A fixed set of subtypes is declared `sealed ... permits`. Dispatch uses
a pattern `switch` with no `default`, so adding a subtype becomes a
compile error rather than a runtime fallthrough. Guards use `when`.

## instanceof binds a pattern variable

`if (o instanceof Customer c)` — never instanceof followed by a cast.
Nested structure is destructured with record patterns rather than chained
accessors.

## Concurrency starts with virtual threads and a closed executor

Task-per-request work uses `Executors.newVirtualThreadPerTaskExecutor()`
inside try-with-resources. Virtual threads are never pooled or reused,
and blocking I/O inside one is correct rather than something to avoid.
`StructuredTaskScope` is only used with an explicit note that it is a
preview API.

## Shared mutable state uses the right primitive, not volatile alone

`volatile` publishes a value; it does not make read-modify-write atomic.
Counters use `AtomicInteger`/`LongAdder`, maps use `ConcurrentHashMap`
with its compound operations (`computeIfAbsent`, `merge`), and
check-then-act sequences over a concurrent collection are never written
as separate calls.

## InterruptedException is propagated or the flag is restored

`catch (InterruptedException e)` either rethrows or calls
`Thread.currentThread().interrupt()` before returning. It is never
logged and discarded.

## equals and hashCode are overridden together, over the same fields

And nothing used in `hashCode` is mutated after the object enters a
hash-based collection. A `Comparator` returning 0 implies `equals`, and
is built with `Comparator.comparing(...).thenComparing(...)` rather than
subtraction.

## Boxed numbers are compared with equals

`==` on `Integer`, `Long` or `Double` is identity. Unboxing a value that
may be absent is checked first — a null `Integer` assigned to an `int`
throws at the assignment.

## Money and exact decimals use BigDecimal built from strings

`BigDecimal.valueOf` or `new BigDecimal("…")`, never
`new BigDecimal(double)`. Every `divide` carries a scale and a
`RoundingMode`. Equality between BigDecimals uses `compareTo(x) == 0`,
because `equals` also compares scale.

## Time uses java.time, with the type that matches the concept

`Instant` for a moment, `ZonedDateTime` when the zone is meaningful,
`LocalDate`/`LocalDateTime` only for zone-free wall-clock values,
`Duration` for elapsed time measured with `System.nanoTime`.
`java.util.Date`, `Calendar` and `SimpleDateFormat` do not appear.

## Optional is returned, never stored or accepted

No `Optional` fields, no `Optional` parameters, no `.get()` without a
presence check. Absence is handled with `orElseGet`, `orElseThrow` or
`map`.

## Resources are managed by try-with-resources

Every `AutoCloseable` — streams, connections, executors — is acquired in
the resource clause. No manual `finally { close(); }`, and no `return`
or `throw` inside a `finally`.

## Exceptions caught are exceptions that can be handled

Specific types, multi-catch where behaviour is shared. Never
`catch (Throwable)`, never an empty catch block, and never catch-log-
continue for something the caller needs to know about. Failures that
cannot be handled locally propagate.

## Every SQL value is a bind parameter

`PreparedStatement` with `?`, or `setParameter` in JPQL. No string
concatenation into a query under any circumstance. Identifiers are
validated against an allowlist rather than interpolated.

## Parsers are hardened before they touch untrusted input

XML factories disable DOCTYPE declarations and external entities.
Untrusted bytes are never passed to `ObjectInputStream.readObject`.
Deserialisation uses a schema-bearing format.

## Unguessable values come from SecureRandom

Tokens, session identifiers, salts and nonces use `SecureRandom`.
`java.util.Random` and `ThreadLocalRandom` are for simulation only.
Passwords go through a slow KDF with a per-password salt, and digest
comparison uses `MessageDigest.isEqual`.

## Nulls are rejected at the boundary

Constructor and public-method arguments that must not be null go through
`Objects.requireNonNull(x, "name")`. Null-safe helpers (`Objects.equals`,
`Objects.hash`, `requireNonNullElse`) replace hand-written null checks.

## Collections are chosen for their mutability contract

`List.of` / `Map.of` for immutable literals, `List.copyOf` to snapshot,
`new ArrayList<>(…)` when it must be mutable. Structural modification
during iteration uses `removeIf` or the iterator. `Collectors.toMap`
always supplies a merge function.

## Nested classes are static unless the outer instance is needed

Non-static inner classes pin their enclosing instance in memory, which
leaks whenever the inner object outlives the outer's usefulness.

## Modern syntax where it is genuinely clearer

Text blocks for embedded SQL, JSON and HTML. Sequenced-collection
methods (`getFirst`, `getLast`, `reversed`) instead of index arithmetic.
Enhanced switch expressions over statement switches with fallthrough.
String templates (`STR."…"`) are never emitted — the feature was
withdrawn and does not compile.
