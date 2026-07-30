# java-stdlib constitution

Applied to every piece of Java written while this pack is mounted.
Each rule targets the measured failure mode of local models writing
Java: invented methods, wrong signatures, and pre-Java-8 idioms.
Terse on purpose — signatures live in the corpus.

## Verify the method before writing the call

Before calling a standard-library method, recall its signature from
this pack. If the pack has no record of the method, say so and use a
documented alternative — never write a call from memory alone.

## No invented APIs

Call only methods the recalled signatures show. If a convenience method
seems like it should exist (String.capitalize, List.first before 21,
Files.readLines), check — it usually does not.

## Prefer the modern spelling

`Path.of` over `Paths.get`; `Files.readString`/`writeString` over
manual readers; `java.net.http.HttpClient` over `URLConnection`;
`java.time` over `java.util.Date` and `Calendar`; records over
getter-setter beans; `List.of`/`Map.of` for fixed data;
`Executors.newVirtualThreadPerTaskExecutor` for I/O-bound concurrency.

## Checked exceptions are part of the signature

The fences show `throws` clauses. Handle or declare exactly those —
IOException on Files calls, InterruptedException on waits — and never
swallow InterruptedException without re-interrupting the thread.

## Immutability is the default

`List.of` and `Map.of` return immutable collections that throw
UnsupportedOperationException on mutation; wrap in `new ArrayList<>()`
when mutation is needed. Strings are immutable — repeated `+=` in a
loop is quadratic; use StringBuilder.

## Version-gate anything newer than 17

Virtual threads and `SequencedCollection` (`getFirst`/`getLast`) are
21+; `Math.clamp` and unnamed variables are 21+/22+. If the project
targets 17 or 11, use the compatible spelling.

## State the source of an uncertain claim

If asked about an API this pack does not cover, prefix the answer with
the fact that it is from model memory, unverified against the local
JDK.
