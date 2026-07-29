# java-modern corpus

Java on a current LTS. The bias here is toward two things a model gets
wrong by default: writing Java 8 when Java 21+ has a better construct,
and violating a contract (equality, visibility, interruption) in code
that compiles and passes a single-threaded test.

Version claims name the JDK that finalised the feature. Where something
is still a preview it says so, because shipping a preview API without
`--enable-preview` is a build failure, and preview APIs change.

## Records are the default for data carriers

`public record Point(int x, int y) {}` generates the constructor,
accessors (`x()`, not `getX()`), `equals`, `hashCode` and `toString`,
all consistent with each other. Records are implicitly final, cannot
extend, and their fields are final. Finalised in Java 16 — hand-writing
a value class with a manual `equals`/`hashCode` pair is now a
correctness risk with no upside.

## Compact constructors validate and normalise

```java
public record Range(int lo, int hi) {
    public Range {
        if (lo > hi) throw new IllegalArgumentException("lo > hi");
        hi = Math.min(hi, MAX);   // assigns the parameter, not this.hi
    }
}
```
The compact form takes no parameter list and no field assignments — the
canonical constructor assigns the fields from the parameters after the
block runs. Assigning `this.hi` inside it is a compile error.

## Records are shallowly immutable

A record holding a `List` still exposes the caller's mutable list. Defensive
copying belongs in the compact constructor —
`items = List.copyOf(items)` — which both copies and rejects nulls. Without
it, a record is a value type with a hole in it.

## Sealed types make a hierarchy exhaustive

`public sealed interface Shape permits Circle, Square, Triangle {}`
restricts implementations to the named types, and every permitted
subclass must itself be `final`, `sealed` or `non-sealed`. Finalised in
Java 17. The payoff is that a `switch` over a sealed type needs no
`default` and stops compiling when someone adds a case.

## Pattern matching for switch, with exhaustiveness

Finalised in Java 21:
```java
String describe(Shape s) {
    return switch (s) {
        case Circle c when c.radius() > 10 -> "big circle";
        case Circle c    -> "circle r=" + c.radius();
        case Square sq   -> "square " + sq.side();
        case Triangle t  -> "triangle";
    };
}
```
The guard keyword is `when`, not `if`. Case order matters — a dominating
pattern before a more specific one is a compile error, which is the
compiler catching a bug that an if-else chain would have hidden.

## Switch over a reference type must handle null

A traditional `switch` on a reference throws `NullPointerException`
before entering any branch. A pattern switch can take `case null ->`
explicitly, and `case null, default ->` folds it into the default. If
neither appears, the NPE behaviour is preserved — so null handling is a
decision you now make in the open.

## Record patterns destructure

```java
if (obj instanceof Point(int x, int y)) { ... }
case Line(Point(var x1, var y1), Point(var x2, var y2)) -> ...
```
Finalised in Java 21, nestable, and `var` is allowed for the components.
This replaces the instanceof-cast-accessor triple that dominated older
code.

## Text blocks handle embedded quotes and indentation

`"""` opens a block; the closing delimiter's indentation sets the
baseline that is stripped from every line. Trailing spaces are removed
unless escaped with `\s`, and `\` at end of line suppresses the newline.
Finalised in Java 15. Building JSON or SQL with `+` and `\n` is now
purely a legibility loss.

## String templates were withdrawn — do not emit them

The `STR."..."` template syntax previewed in Java 21 and 22 and was
**withdrawn** in Java 23. It is not in any current release and will not
compile. Use `formatted()`, `String.format`, or concatenation. This is
worth knowing precisely because the syntax appears in training data from
its preview window.

## Virtual threads: millions of them, blocking is fine

Finalised in Java 21 (JEP 444). `Thread.ofVirtual().start(r)` or
`Executors.newVirtualThreadPerTaskExecutor()`. They are cheap enough to
create one per task, so the whole pool-sizing discipline goes away — and
blocking on I/O inside one is *correct*, because it parks the virtual
thread and frees the carrier.

## Never pool virtual threads, and never pool ThreadLocals in them

Pooling exists to amortise the cost of creating a platform thread.
Virtual threads have no such cost, so a fixed pool of them reimposes the
limit it was meant to remove. `ThreadLocal` still works but loses its
point: with a thread per task there is no reuse, and heavy thread-locals
now multiply by the number of tasks.

## What pins a virtual thread

A pinned virtual thread cannot unmount, so its carrier blocks and
throughput collapses. Native frames (JNI) pin. `synchronized` blocks
pinned in Java 21 — the standard advice then was to use
`ReentrantLock` — and JEP 491 in Java 24 removed that limitation for
monitors. On Java 21 the advice stands; on 24+ `synchronized` no longer
pins. Diagnose with `-Djdk.tracePinnedThreads=full`.

## Sequenced collections gave order a vocabulary

Java 21 added `SequencedCollection` with `getFirst()`, `getLast()`,
`addFirst()`, `addLast()`, `removeFirst()`, `removeLast()` and
`reversed()`, retrofitted onto `List`, `Deque`, `LinkedHashMap` and
`SortedSet`. `list.get(list.size() - 1)` and the
`new ArrayList<>(...)`-then-`Collections.reverse` dance are both obsolete.

## ExecutorService is closeable, and its threads keep the JVM alive

Since Java 19, `ExecutorService` implements `AutoCloseable`, so
`try (var ex = Executors.newVirtualThreadPerTaskExecutor()) { ... }`
shuts it down and awaits termination at the closing brace. Without that,
a non-daemon pool that is never `shutdown()` keeps the JVM running after
`main` returns — a hang that looks like a deadlock and is not one.

## Structured concurrency is still preview

`StructuredTaskScope` ties a set of concurrent subtasks to a lexical
scope so they cannot outlive it and errors propagate to the parent. It
has been in preview across several releases and its API has changed
between them, so it requires `--enable-preview` and pinning to a
specific JDK. Do not present it as generally available.

## Double-checked locking requires volatile

Without `volatile` on the field, another thread can observe a non-null
reference to a partially constructed object, because the constructor's
writes are not ordered against the reference publication. The field must
be `volatile` and read into a local first. For a static singleton the
holder idiom — a private static nested class initialised on first
access — is simpler and correct by class-initialisation semantics.

## final fields have a publication guarantee that non-final ones do not

The JMM guarantees that a thread which sees a reference to a correctly
constructed object sees its `final` fields fully initialised — provided
`this` did not escape the constructor. Non-final fields carry no such
guarantee. Starting a thread, registering a listener, or calling an
overridable method from a constructor breaks it.

## volatile gives visibility, not atomicity

`volatile int count; count++` is three operations and races. `volatile`
is correct for a flag written by one thread and read by others; for a
counter use `AtomicInteger` or `LongAdder`, which is markedly faster
under high contention because it shards the count across cells.

## get-then-put on a ConcurrentHashMap is not atomic

`if (!map.containsKey(k)) map.put(k, v)` is a race even though each call
is individually thread-safe. Use the compound operations the class
provides: `putIfAbsent`, `computeIfAbsent`, `compute`, `merge`. The
mapping function passed to `computeIfAbsent` must not modify the same
map — that can deadlock or corrupt the table.

## HashMap under concurrent write loses data

`HashMap` is not thread-safe. Concurrent resize used to produce an
infinite loop on lookup in Java 7; on Java 8+ the failure is quieter —
lost entries and wrong sizes. `Collections.synchronizedMap` locks each
call but still needs external synchronisation around iteration.
`ConcurrentHashMap` is the answer for anything genuinely shared.

## Iterating a synchronized wrapper needs a manual lock

`Collections.synchronizedList` synchronises individual methods; the
iterator does not hold the lock. Iteration must be wrapped:
`synchronized (list) { for (var x : list) ... }`. Forgetting this yields
`ConcurrentModificationException` intermittently under load.

## Removing during iteration: removeIf, not remove

Calling `collection.remove(x)` inside an enhanced for loop throws
`ConcurrentModificationException` — the fail-fast iterator detects the
structural modification. Use `collection.removeIf(predicate)`, or the
iterator's own `it.remove()`. The exception is best-effort and may not
fire, which makes the bug worse, not better.

## InterruptedException: restore the flag or propagate

Catching it and logging discards the interruption, and the code above
never learns it was asked to stop. Either declare and propagate it, or
`Thread.currentThread().interrupt()` in the catch to restore the flag
before returning. Swallowing it is why shutdown hangs.

## equals and hashCode move together, over the same fields

Two objects that are `equals` must have the same `hashCode`. Override
one without the other and the object silently misbehaves in every
`HashMap` and `HashSet` — `contains` returns false for an object that is
in the set. Mutating a field used by `hashCode` after insertion strands
the entry the same way. `record` gives you both, correctly.

## compareTo should agree with equals

`TreeMap` and `TreeSet` use `compareTo`, not `equals`, to decide
identity. A comparator that returns 0 for objects that are not `equals`
makes those collections quietly deduplicate. A comparator must also be
transitive and antisymmetric — `Comparator.comparing(...)`, chained with
`thenComparing`, is the way to stay inside the contract; a hand-rolled
`a.value - b.value` also overflows for large magnitudes.

## == on boxed Integers is identity, with a cache that hides it

`Integer.valueOf` caches −128 to 127, so `==` on boxed values *appears*
to work in tests and fails on real data. Compare with `equals`, or
unbox to `int` deliberately. Autoboxing a null `Integer` into an `int`
throws NPE at the point of use, often far from where the null entered —
`map.get(missing)` assigned to an `int` is the usual shape.

## new BigDecimal(double) captures the binary error

`new BigDecimal(0.1)` is `0.1000000000000000055511151231257827…` because
the `double` was never 0.1. `BigDecimal.valueOf(0.1)` (which routes via
`Double.toString`) and `new BigDecimal("0.1")` both give exactly 0.1.
For money, use `BigDecimal` from strings or long minor units, with an
explicit `RoundingMode` on every `divide` — the two-argument `divide`
throws `ArithmeticException` on a non-terminating result.

## BigDecimal equals compares scale; compareTo does not

`new BigDecimal("1.0").equals(new BigDecimal("1.00"))` is **false** —
`equals` includes the scale. `compareTo` returns 0. So `BigDecimal` in a
`HashSet` deduplicates by scale as well as value, and comparisons should
use `compareTo(x) == 0`.

## java.time, and which type actually models the concept

`Instant` for a machine timestamp, `LocalDate`/`LocalDateTime` for a
wall-clock value with no zone, `ZonedDateTime` when the zone matters,
`Duration` for elapsed time, `Period` for calendar amounts. The common
defect is `LocalDateTime` for something that is a moment in time — it
has no zone, so it is ambiguous across DST and cannot be ordered against
another zone's value. `java.util.Date` and `Calendar` are legacy.

## DateTimeFormatter is thread-safe; SimpleDateFormat is not

A `static SimpleDateFormat` shared across threads produces silently
wrong dates, not an exception — it holds mutable parsing state. Every
`java.time` formatter is immutable and safe to share. `System.nanoTime`,
not `currentTimeMillis`, measures elapsed time; the wall clock can jump.

## Optional is a return type, not a field or a parameter

It exists to make "no value" explicit in an API's return. As a field it
adds an allocation and breaks serialisation; as a parameter it forces
callers to wrap. Never call `get()` without `isPresent()` — use
`orElse`, `orElseGet` (lazy), `orElseThrow`, `map`, `ifPresent`. And
`Optional` itself must never be null.

## Streams are single-use and must not have side effects

Consuming a stream twice throws `IllegalStateException`. A lambda inside
`map` or `filter` that mutates external state is unsafe under
`parallel()` and unspecified even sequentially — collect instead. Reach
for `parallel()` only for large, CPU-bound, side-effect-free work over a
splittable source; on a small list it is slower, and it shares the
common ForkJoinPool with everything else in the process.

## Collectors.toMap throws on duplicate keys and rejects null values

The two-argument form throws `IllegalStateException` on a duplicate key;
supply a merge function to decide. It also throws NPE if a value maps to
null, where `HashMap.put(k, null)` would have been fine. `toList()` on
the stream (Java 16+) returns an unmodifiable list, while
`Collectors.toList()` returns a mutable `ArrayList` — a difference that
surfaces as `UnsupportedOperationException` after a refactor.

## List.of and Arrays.asList are different kinds of not-a-list

`List.of(...)` is immutable and rejects null elements with NPE.
`Arrays.asList(...)` is a fixed-size view over the array: `set` works,
`add`/`remove` throw, and writes pass through to the backing array.
`List.copyOf` snapshots. `new ArrayList<>(List.of(...))` is the way to
get something genuinely mutable.

## try-with-resources closes in reverse and suppresses correctly

Resources close in reverse declaration order, and if both the body and a
`close()` throw, the body's exception propagates with the close
exception attached via `getSuppressed()`. A hand-written
`finally { close(); }` inverts that — the close exception replaces the
real one and the original cause is lost.

## return or throw inside finally discards the pending exception

A `return` in `finally` silently swallows an exception that was
propagating, and so does a `throw`. Compile with the warning enabled and
treat any `finally` that can complete abruptly as a bug.

## Never catch Throwable, and never catch and log-and-continue

`catch (Throwable t)` catches `OutOfMemoryError` and `StackOverflowError`,
which the code cannot meaningfully handle, and `ThreadDeath`. Catch the
specific exceptions you can act on. Multi-catch — `catch (IOException |
SQLException e)` — avoids the copy-paste that leads to catching
`Exception`.

## Every SQL parameter is a bind parameter

`PreparedStatement` with `?` placeholders, values set via `setString` /
`setLong`. String concatenation into SQL is injection regardless of what
sanitising happens first. Table and column names cannot be parameterised
— validate them against an allowlist. In JPA/Hibernate the equivalent is
`setParameter`, never string-built JPQL.

## XML parsers are insecure by default

`DocumentBuilderFactory`, `SAXParserFactory` and `XMLInputFactory`
resolve external entities out of the box, which is XXE — local file
disclosure and SSRF. Set
`factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true)`
or, at minimum, disable external general and parameter entities and
`setXIncludeAware(false)`. The same caution applies to YAML: SnakeYAML's
plain `new Yaml()` constructor can instantiate arbitrary types.

## Java deserialization of untrusted bytes is remote code execution

`ObjectInputStream.readObject` on attacker-controlled data is
exploitable through gadget chains in libraries you merely have on the
classpath. There is no way to make it safe by validating afterwards —
the damage happens during construction. Use a data format (JSON, protobuf)
with an explicit schema. If it is unavoidable, apply a strict
`ObjectInputFilter` allowlist.

## SecureRandom for anything a person must not guess

`java.util.Random` is a 48-bit linear congruential generator whose
entire future output is recoverable from two consecutive values —
tokens, session ids, password resets and nonces need `SecureRandom`.
`ThreadLocalRandom` is the right choice for simulation and load
generation, where the requirement is speed and non-contention rather
than unpredictability.

## Passwords are hashed with a slow KDF, and verified in constant time

Argon2id, bcrypt or PBKDF2 with a per-password salt and a real work
factor. SHA-256 is a fast hash and therefore the wrong tool, salted or
not. Compare MACs and tokens with `MessageDigest.isEqual`, which is
constant-time, rather than `Arrays.equals` or `String.equals`.

## Objects.requireNonNull at the boundary, with a message

`this.name = Objects.requireNonNull(name, "name")` fails at the
constructor with a useful message instead of at some later dereference
with none. `Objects.requireNonNullElse` supplies a default, and
`Objects.equals` / `Objects.hash` are null-safe.

## Helpful NullPointerExceptions are on by default since Java 15

The message now names the expression that was null —
"Cannot invoke String.length() because the return value of Map.get(k) is
null". Debugging advice that assumes a bare NPE with only a line number
is out of date.

## Static nested unless the instance link is used

A non-static inner class holds a reference to its enclosing instance,
which keeps the outer object alive for as long as the inner one lives —
a classic leak when the inner class is a listener or a `Runnable`
handed to a long-lived executor. Declare nested classes `static` unless
they genuinely need the outer instance.

## finalize is gone; Cleaner is the replacement

`Object.finalize` was deprecated for removal and is disabled or removed
on current releases. For native resource cleanup use `Cleaner` or a
`PhantomReference`, and treat both as a backstop — the real mechanism is
`AutoCloseable` plus try-with-resources.

## var infers, it does not defer

`var` takes the type of the initialiser, so `var list = new ArrayList<String>()`
gives `ArrayList<String>`, not `List<String>` — and `var x = null` does
not compile. It is not allowed for fields, method parameters or return
types. Use it where the right-hand side already names the type; avoid it
where it hides a factory method's result.
