#!/usr/bin/env python3
"""Generate a Java API ground-truth pack by disassembling the local JDK.

Same architecture as gen_api_pack.py (Python), same laws: signatures come
from the artifact — `javap -public` on the running JDK — never from
memory. One difference forced by the platform: class files carry no doc
comments, so the prose lead for each record is AUTHORED below, in task
vocabulary, while every signature in the fence is javap's own output.
A wrong signature cannot survive this pipeline; a wrong prose line can,
so the prose stays one sentence of orientation and never asserts API.

Usage:
    python packs/gen_java_pack.py --out packs/java-stdlib
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

# The classes agent code actually touches, each with an authored
# one-line description in the vocabulary a consumer queries with.
CLASSES: list[tuple[str, str]] = [
    ("java.lang.String", "Immutable text: split, join, strip, replace, format, and text blocks."),
    ("java.lang.StringBuilder", "Mutable string building without quadratic concatenation."),
    ("java.util.List", "Ordered collection: positional access, of-factories. List.of AND List.copyOf both return immutable lists that throw UnsupportedOperationException on add or remove; a mutable copy is new ArrayList<>(list)."),
    ("java.util.Map", "Key-value mapping: getOrDefault, computeIfAbsent, merge. Map.of and Map.copyOf return immutable maps and reject null keys and values."),
    ("java.util.Set", "Unique elements: membership tests. Set.of returns an immutable set and throws IllegalArgumentException on duplicate elements."),
    ("java.util.Optional", "Container for a possibly-absent value instead of null returns. get() throws NoSuchElementException when empty; prefer orElse, orElseGet or orElseThrow."),
    ("java.util.ArrayList", "Resizable array list, the default List implementation."),
    ("java.util.HashMap", "Hash table Map implementation, the default for unordered mappings."),
    ("java.util.Arrays", "Static helpers for arrays: sort, binarySearch, stream, fill. asList returns a fixed-size view backed by the array — add throws UnsupportedOperationException."),
    ("java.util.Collections", "Static helpers for collections: unmodifiable views, sort, reverse, shuffle."),
    ("java.util.Objects", "Null-safe helpers: requireNonNull, equals, hash, toString with default."),
    ("java.util.stream.Stream", "Lazy pipeline over elements: filter, map, flatMap, collect, reduce."),
    ("java.util.stream.Collectors", "Terminal collectors: toList, joining, groupingBy, partitioningBy, toMap."),
    ("java.util.stream.IntStream", "Primitive int pipeline: range, rangeClosed, sum, average, boxed."),
    ("java.nio.file.Files", "File operations on Path: readString, writeString, lines, walk, copy, createDirectories."),
    ("java.nio.file.Path", "Immutable file path: of-factory, resolve, getParent, getFileName, toAbsolutePath."),
    ("java.nio.file.Paths", "Legacy Path factory; new code uses Path.of instead."),
    ("java.io.BufferedReader", "Buffered character reading: readLine and lines stream."),
    ("java.io.IOException", "Checked exception thrown when file or stream I/O fails."),
    ("java.net.URI", "Parsed uniform resource identifier; the argument HttpRequest builders take."),
    ("java.net.http.HttpClient", "HTTP client for synchronous send and asynchronous sendAsync requests."),
    ("java.net.http.HttpRequest", "Immutable HTTP request built with newBuilder: uri, header, GET, POST, timeout."),
    ("java.net.http.HttpResponse", "HTTP response: statusCode, body, headers; body handlers pick the body type."),
    ("java.time.LocalDate", "Calendar date without time zone: now, of, parse, plusDays, format."),
    ("java.time.LocalDateTime", "Date and time without zone: now, of, parse, formatting with DateTimeFormatter."),
    ("java.time.Instant", "Machine timestamp on the UTC timeline: now, ofEpochMilli, plusSeconds."),
    ("java.time.Duration", "Time-based amount: ofSeconds, ofMillis, between, toMillis."),
    ("java.time.ZonedDateTime", "Date-time with time zone: now, of, withZoneSameInstant conversions."),
    ("java.time.format.DateTimeFormatter", "Formatting and parsing patterns: ofPattern, ISO constants."),
    ("java.util.concurrent.CompletableFuture", "Asynchronous result composition: supplyAsync, thenApply, thenCompose, exceptionally, join."),
    ("java.util.concurrent.ExecutorService", "Thread pool interface: submit, invokeAll, shutdown, close."),
    ("java.util.concurrent.Executors", "Thread pool factories: newFixedThreadPool, newVirtualThreadPerTaskExecutor, newScheduledThreadPool."),
    ("java.util.concurrent.ConcurrentHashMap", "Thread-safe map: computeIfAbsent for caches. Unlike HashMap it rejects null keys AND null values, throwing NullPointerException immediately on put."),
    ("java.util.concurrent.TimeUnit", "Time granularity for timeouts: SECONDS, MILLISECONDS, sleep and conversions."),
    ("java.util.concurrent.CountDownLatch", "One-shot synchronization barrier: countDown and await."),
    ("java.util.concurrent.atomic.AtomicInteger", "Lock-free integer counter: incrementAndGet, compareAndSet."),
    ("java.util.regex.Pattern", "Compiled regular expression: compile, matcher, and the matches test."),
    ("java.util.regex.Matcher", "Regex match state: find, group, replaceAll, start and end offsets."),
    ("java.util.Scanner", "Text tokenizing over input streams and strings: nextLine, nextInt, hasNext."),
    ("java.util.Random", "Pseudorandom numbers: nextInt with bound, ints stream; not for security."),
    ("java.security.SecureRandom", "Cryptographically strong random numbers for tokens and salts."),
    ("java.security.MessageDigest", "Cryptographic hashing: getInstance SHA-256, update, digest."),
    ("java.util.Base64", "Base64 encoding and decoding: getEncoder, getDecoder, getUrlEncoder."),
    ("java.util.UUID", "Universally unique identifiers: randomUUID, fromString, toString."),
    ("java.util.Iterator", "Sequential element access: hasNext, next, remove."),
    ("java.util.Comparator", "Ordering strategy: comparing, thenComparing, reversed, naturalOrder."),
    ("java.util.StringJoiner", "Joining strings with delimiter, prefix and suffix."),
    ("java.lang.Integer", "Boxed int: parseInt, valueOf, MAX_VALUE, toString with radix."),
    ("java.lang.Long", "Boxed long: parseLong, valueOf, and unsigned helpers."),
    ("java.lang.Double", "Boxed double: parseDouble, isNaN, compare."),
    ("java.lang.Character", "Char tests and conversions: isDigit, isLetter, toLowerCase."),
    ("java.lang.Math", "Numeric functions: abs, max, min, pow, sqrt, floorDiv, clamp."),
    ("java.lang.Thread", "Threads: ofVirtual and ofPlatform builders, sleep, currentThread, interrupt."),
    ("java.lang.Runtime", "JVM runtime: availableProcessors, addShutdownHook, exec is legacy."),
    ("java.lang.ProcessBuilder", "Launching external processes: command list, redirectErrorStream, start."),
    ("java.lang.Process", "A running external process: waitFor, exitValue, getInputStream, onExit."),
    ("java.lang.AutoCloseable", "The try-with-resources contract: close."),
    ("java.lang.Iterable", "The for-each contract: iterator, forEach, spliterator."),
    ("java.lang.Record", "Base class of record types: compact immutable data carriers."),
    ("java.lang.Exception", "Base checked exception: message, cause, suppressed exceptions."),
]

MODIFIERS = re.compile(
    r"^\s*(?:public|protected|private|static|final|native|synchronized|"
    r"abstract|default|strictfp|transient|volatile)\s+")


def javap(fqcn: str) -> list[str]:
    """Member signature lines from `javap -public`, modifiers stripped."""
    try:
        out = subprocess.run(["javap", "-public", fqcn],
                             capture_output=True, text=True, timeout=60,
                             check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired) as e:
        print(f"  skip {fqcn}: {e}")
        return []
    sigs = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.endswith("{") or line == "}":
            continue
        prev = None
        while prev != line:
            prev, line = line, MODIFIERS.sub("", line)
        line = line.rstrip(";")
        # Drop generic-method type prefixes for readability: `<T> T foo()`
        if line and "(" in line:
            sigs.append(line)
    return sigs


def dedupe_overloads(sigs: list[str], per_name: int = 2) -> list[str]:
    """Keep at most N overloads per method name — the third overload of
    println buys no retrieval and crowds out other methods."""
    seen: dict[str, int] = {}
    out = []
    for s in sigs:
        m = re.search(r"([A-Za-z_$][\w$]*)\s*\(", s)
        name = m.group(1) if m else s
        seen[name] = seen.get(name, 0) + 1
        if seen[name] <= per_name:
            out.append(s)
    return out


def record_for(fqcn: str, desc: str) -> str | None:
    sigs = dedupe_overloads(javap(fqcn))
    if not sigs:
        return None
    simple = fqcn.rsplit(".", 1)[-1]
    # javap emits declaration order, which for String means charAt wins
    # the fence budget over split/join/strip — the very members the
    # description promises. The authored description names what matters;
    # let it order the fence.
    desc_words = {w.strip(",.").lower() for w in desc.split()}
    def prio(s: str) -> int:
        m = re.search(r"([A-Za-z_$][\w$]*)\s*\(", s)
        return 0 if m and m.group(1).lower() in desc_words else 1
    sigs = sorted(sigs, key=prio)
    # Prose-balance the fence, same law as the Python generator: the
    # 64-dim embedder sees prose, so method names join the prose and
    # signatures stop once code would outweigh it (floor 6, cap 24).
    names = []
    for s in sigs:
        m = re.search(r"([A-Za-z_$][\w$]*)\s*\(", s)
        if m and m.group(1) not in names and m.group(1) != simple:
            names.append(m.group(1))
    # Heading is the FIRST clause only — the full description as heading
    # buried "List.of" under 200 chars of lowercased noise and the List
    # record fell out of top-5 for its own core question. The hazard
    # sentences stay in the body where the embedder still sees them.
    head = f"{fqcn}: {desc.split('. ')[0].rstrip('.').strip().lower()}"
    name_prose = f"Key members are {', '.join(names[:18])}." if names else ""
    prose_len = len(head) + len(desc) + len(name_prose)
    kept, code_len = [], 0
    for s in sigs:
        if len(kept) >= 6 and code_len + len(s) > prose_len:
            break
        kept.append(s)
        code_len += len(s)
        if len(kept) >= 24:
            break
    parts = [f"## {head}\n", desc]
    if name_prose:
        parts.append(name_prose)
    parts.append("\n```java\n" + "\n".join(kept) + "\n```")
    return "\n".join(parts) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ver = subprocess.run(["java", "-version"], capture_output=True,
                         text=True).stderr.splitlines()[0]

    records = []
    for fqcn, desc in CLASSES:
        rec = record_for(fqcn, desc)
        if rec:
            records.append(rec)
    print(f"{len(records)}/{len(CLASSES)} classes disassembled")

    header = (
        f"# java-stdlib corpus — signatures from `javap` on {ver}\n\n"
        f"Every fenced signature below is javap's own output from the\n"
        f"local JDK — correct by construction, regenerated rather than\n"
        f"edited. Prose leads are authored orientation; the fence is the\n"
        f"authority.\n")
    # Two authored records that back the constitution's process rules —
    # a rule with no retrievable corpus evidence is an assertion the
    # pack cannot support (lint_pack checks the join).
    process = """
## Hallucinated Java APIs: String.capitalize and Files.readLines do not exist

Frequently invented APIs that do NOT exist in the JDK:
String.capitalize, Files.readLines, List.first before Java 21,
Optional.getOrNull, Stream.toSet. The fences in this pack are javap
output from a real JDK, so a method absent from a class's record
usually does not exist — verify against the recalled signatures and
use the documented alternative the record does show.

## Calling add or remove on List.of and Map.of throws UnsupportedOperationException

The of-factories (List.of, Map.of, Set.of) and List.copyOf all return
IMMUTABLE collections: calling add, remove, put or set on them throws
UnsupportedOperationException. Arrays.asList is a fixed-size view with
the same behaviour on add. The mutable copy is a constructor, not a
factory: new ArrayList<>(list) or new HashMap<>(map). List.copyOf is
NOT the mutable-copy route — it is immutable too.

## String concatenation with += in a loop is quadratic

Each += on a String copies the whole accumulated text because Strings
are immutable, so a loop of appends is O(n squared). Use StringBuilder
with append in the loop and toString at the end; String.join or a
Collectors.joining stream handles the delimiter case.

## Catching InterruptedException must restore the interrupt flag

Catching InterruptedException clears the thread's interrupt status. A
catch block that neither rethrows nor calls
Thread.currentThread().interrupt() swallows the interrupt and the
thread can no longer be stopped cooperatively. Restore the flag or
propagate the exception; never catch and ignore it.

## Answering a Java API question this pack does not cover

When a question concerns a class or method with no record in this
pack, state that the answer comes from model memory, unverified
against the local JDK, and may not match its version. The pack was
generated from a specific JDK; an uncertain claim about another
version deserves the same caveat. Verification command for any class:
`javap -public java.util.List` prints its true public signatures.
"""
    body = header + "\n" + "\n".join(records) + process
    recipes = out / "recipes.md"
    if recipes.exists():
        body += "\n" + recipes.read_text(encoding="utf-8")
    (out / "corpus.md").write_text(body, encoding="utf-8")
    n = body.count("\n## ")
    print(f"wrote {out / 'corpus.md'} — {n} records, {len(body.split())} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
