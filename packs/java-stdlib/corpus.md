# java-stdlib corpus — signatures from `javap` on java version "24.0.1" 2025-04-15

Every fenced signature below is javap's own output from the
local JDK — correct by construction, regenerated rather than
edited. Prose leads are authored orientation; the fence is the
authority.

## java.lang.String: immutable text: split, join, strip, replace, format, and text blocks

Immutable text: split, join, strip, replace, format, and text blocks.
Key members are replace, split, join, strip, format, length, isEmpty, charAt, codePointAt, codePointBefore, codePointCount, offsetByCodePoints, getChars, getBytes, equals, contentEquals, equalsIgnoreCase, compareTo.

```java
java.lang.String replace(char, char)
java.lang.String replace(java.lang.CharSequence, java.lang.CharSequence)
java.lang.String[] split(java.lang.String, int)
java.lang.String[] split(java.lang.String)
java.lang.String join(java.lang.CharSequence, java.lang.CharSequence...)
java.lang.String join(java.lang.CharSequence, java.lang.Iterable<? extends java.lang.CharSequence>)
```

## java.lang.StringBuilder: mutable string building without quadratic concatenation

Mutable string building without quadratic concatenation.
Key members are compareTo, append, appendCodePoint, delete, deleteCharAt, replace, insert, indexOf, lastIndexOf, reverse, repeat, toString, codePoints, chars, substring, subSequence, setCharAt, getChars.

```java
java.lang.StringBuilder()
java.lang.StringBuilder(int)
int compareTo(java.lang.StringBuilder)
java.lang.StringBuilder append(java.lang.Object)
java.lang.StringBuilder append(java.lang.String)
java.lang.StringBuilder appendCodePoint(int)
java.lang.StringBuilder delete(int, int)
java.lang.StringBuilder deleteCharAt(int)
```

## java.util.List: ordered collection: positional access, of-factories

Ordered collection: positional access, of-factories. List.of AND List.copyOf both return immutable lists that throw UnsupportedOperationException on add or remove; a mutable copy is new ArrayList<>(list).
Key members are add, size, isEmpty, contains, iterator, toArray, remove, containsAll, addAll, removeAll, retainAll, replaceAll, sort, clear, equals, hashCode, get, set.

```java
boolean add(E)
void add(int, E)
int size()
boolean isEmpty()
boolean contains(java.lang.Object)
java.util.Iterator<E> iterator()
java.lang.Object[] toArray()
<T> T[] toArray(T[])
boolean remove(java.lang.Object)
boolean containsAll(java.util.Collection<?>)
boolean addAll(java.util.Collection<? extends E>)
boolean addAll(int, java.util.Collection<? extends E>)
boolean removeAll(java.util.Collection<?>)
boolean retainAll(java.util.Collection<?>)
```

## java.util.Map: key-value mapping: getordefault, computeifabsent, merge

Key-value mapping: getOrDefault, computeIfAbsent, merge. Map.of and Map.copyOf return immutable maps and reject null keys and values.
Key members are values, getOrDefault, computeIfAbsent, merge, size, isEmpty, containsKey, containsValue, get, put, remove, putAll, clear, keySet, entrySet, equals, hashCode, forEach.

```java
java.util.Collection<V> values()
V getOrDefault(java.lang.Object, V)
V computeIfAbsent(K, java.util.function.Function<? super K, ? extends V>)
V merge(K, V, java.util.function.BiFunction<? super V, ? super V, ? extends V>)
int size()
boolean isEmpty()
boolean containsKey(java.lang.Object)
boolean containsValue(java.lang.Object)
V get(java.lang.Object)
V put(K, V)
V remove(java.lang.Object)
```

## java.util.Set: unique elements: membership tests

Unique elements: membership tests. Set.of returns an immutable set and throws IllegalArgumentException on duplicate elements.
Key members are size, isEmpty, contains, iterator, toArray, add, remove, containsAll, addAll, retainAll, removeAll, clear, equals, hashCode, spliterator, of, copyOf.

```java
int size()
boolean isEmpty()
boolean contains(java.lang.Object)
java.util.Iterator<E> iterator()
java.lang.Object[] toArray()
<T> T[] toArray(T[])
boolean add(E)
boolean remove(java.lang.Object)
boolean containsAll(java.util.Collection<?>)
boolean addAll(java.util.Collection<? extends E>)
boolean retainAll(java.util.Collection<?>)
```

## java.util.Optional: container for a possibly-absent value instead of null returns

Container for a possibly-absent value instead of null returns. get() throws NoSuchElementException when empty; prefer orElse, orElseGet or orElseThrow.
Key members are of, or, orElse, orElseGet, orElseThrow, empty, ofNullable, get, isPresent, isEmpty, ifPresent, ifPresentOrElse, filter, map, flatMap, stream, equals, hashCode.

```java
<T> java.util.Optional<T> of(T)
java.util.Optional<T> or(java.util.function.Supplier<? extends java.util.Optional<? extends T>>)
T orElse(T)
T orElseGet(java.util.function.Supplier<? extends T>)
T orElseThrow()
<X extends java.lang.Throwable> T orElseThrow(java.util.function.Supplier<? extends X>) throws X
<T> java.util.Optional<T> empty()
<T> java.util.Optional<T> ofNullable(T)
T get()
boolean isPresent()
```

## java.util.ArrayList: resizable array list, the default list implementation

Resizable array list, the default List implementation.
Key members are trimToSize, ensureCapacity, size, isEmpty, contains, indexOf, lastIndexOf, clone, toArray, get, getFirst, getLast, set, add, addFirst, addLast, remove, removeFirst.

```java
java.util.ArrayList(int)
java.util.ArrayList()
void trimToSize()
void ensureCapacity(int)
int size()
boolean isEmpty()
boolean contains(java.lang.Object)
int indexOf(java.lang.Object)
int lastIndexOf(java.lang.Object)
java.lang.Object clone()
java.lang.Object[] toArray()
<T> T[] toArray(T[])
E get(int)
E getFirst()
```

## java.util.HashMap: hash table map implementation, the default for unordered mappings

Hash table Map implementation, the default for unordered mappings.
Key members are size, isEmpty, get, containsKey, put, putAll, remove, clear, containsValue, keySet, values, entrySet, getOrDefault, putIfAbsent, replace, computeIfAbsent, computeIfPresent, compute.

```java
java.util.HashMap(int, float)
java.util.HashMap(int)
int size()
boolean isEmpty()
V get(java.lang.Object)
boolean containsKey(java.lang.Object)
V put(K, V)
void putAll(java.util.Map<? extends K, ? extends V>)
V remove(java.lang.Object)
void clear()
boolean containsValue(java.lang.Object)
java.util.Set<K> keySet()
java.util.Collection<V> values()
```

## java.util.Arrays: static helpers for arrays: sort, binarysearch, stream, fill

Static helpers for arrays: sort, binarySearch, stream, fill. asList returns a fixed-size view backed by the array — add throws UnsupportedOperationException.
Key members are sort, binarySearch, fill, asList, stream, parallelSort, parallelPrefix, equals, copyOf, copyOfRange, hashCode, deepHashCode, deepEquals, toString, deepToString, setAll, parallelSetAll, spliterator.

```java
void sort(int[])
void sort(int[], int, int)
int binarySearch(long[], long)
int binarySearch(long[], int, int, long)
void fill(long[], long)
void fill(long[], int, int, long)
<T> java.util.List<T> asList(T...)
<T> java.util.stream.Stream<T> stream(T[])
<T> java.util.stream.Stream<T> stream(T[], int, int)
void parallelSort(byte[])
void parallelSort(byte[], int, int)
<T> void parallelPrefix(T[], java.util.function.BinaryOperator<T>)
```

## java.util.Collections: static helpers for collections: unmodifiable views, sort, reverse, shuffle

Static helpers for collections: unmodifiable views, sort, reverse, shuffle.
Key members are sort, reverse, shuffle, binarySearch, swap, fill, copy, min, max, rotate, replaceAll, indexOfSubList, lastIndexOfSubList, unmodifiableCollection, unmodifiableSequencedCollection, unmodifiableSet, unmodifiableSequencedSet, unmodifiableSortedSet.

```java
<T extends java.lang.Comparable<? super T>> void sort(java.util.List<T>)
<T> void sort(java.util.List<T>, java.util.Comparator<? super T>)
void reverse(java.util.List<?>)
void shuffle(java.util.List<?>)
void shuffle(java.util.List<?>, java.util.Random)
<T> int binarySearch(java.util.List<? extends java.lang.Comparable<? super T>>, T)
<T> int binarySearch(java.util.List<? extends T>, T, java.util.Comparator<? super T>)
```

## java.util.Objects: null-safe helpers: requirenonnull, equals, hash, tostring with default

Null-safe helpers: requireNonNull, equals, hash, toString with default.
Key members are equals, hash, toString, requireNonNull, deepEquals, hashCode, toIdentityString, compare, isNull, nonNull, requireNonNullElse, requireNonNullElseGet, checkIndex, checkFromToIndex, checkFromIndexSize.

```java
boolean equals(java.lang.Object, java.lang.Object)
int hash(java.lang.Object...)
java.lang.String toString(java.lang.Object)
java.lang.String toString(java.lang.Object, java.lang.String)
<T> T requireNonNull(T)
<T> T requireNonNull(T, java.lang.String)
boolean deepEquals(java.lang.Object, java.lang.Object)
int hashCode(java.lang.Object)
```

## java.util.stream.Stream: lazy pipeline over elements: filter, map, flatmap, collect, reduce

Lazy pipeline over elements: filter, map, flatMap, collect, reduce.
Key members are filter, map, flatMap, reduce, collect, mapToInt, mapToLong, mapToDouble, flatMapToInt, flatMapToLong, flatMapToDouble, mapMulti, mapMultiToInt, mapMultiToLong, mapMultiToDouble, distinct, sorted, peek.

```java
java.util.stream.Stream<T> filter(java.util.function.Predicate<? super T>)
<R> java.util.stream.Stream<R> map(java.util.function.Function<? super T, ? extends R>)
<R> java.util.stream.Stream<R> flatMap(java.util.function.Function<? super T, ? extends java.util.stream.Stream<? extends R>>)
T reduce(T, java.util.function.BinaryOperator<T>)
java.util.Optional<T> reduce(java.util.function.BinaryOperator<T>)
<R> R collect(java.util.function.Supplier<R>, java.util.function.BiConsumer<R, ? super T>, java.util.function.BiConsumer<R, R>)
```

## java.util.stream.Collectors: terminal collectors: tolist, joining, groupingby, partitioningby, tomap

Terminal collectors: toList, joining, groupingBy, partitioningBy, toMap.
Key members are toList, joining, groupingBy, partitioningBy, toMap, toCollection, toUnmodifiableList, toSet, toUnmodifiableSet, mapping, flatMapping, filtering, collectingAndThen, counting, minBy, maxBy, summingInt, summingLong.

```java
<T> java.util.stream.Collector<T, ?, java.util.List<T>> toList()
java.util.stream.Collector<java.lang.CharSequence, ?, java.lang.String> joining()
java.util.stream.Collector<java.lang.CharSequence, ?, java.lang.String> joining(java.lang.CharSequence)
<T, K> java.util.stream.Collector<T, ?, java.util.Map<K, java.util.List<T>>> groupingBy(java.util.function.Function<? super T, ? extends K>)
<T, K, A, D> java.util.stream.Collector<T, ?, java.util.Map<K, D>> groupingBy(java.util.function.Function<? super T, ? extends K>, java.util.stream.Collector<? super T, A, D>)
<T> java.util.stream.Collector<T, ?, java.util.Map<java.lang.Boolean, java.util.List<T>>> partitioningBy(java.util.function.Predicate<? super T>)
```

## java.util.stream.IntStream: primitive int pipeline: range, rangeclosed, sum, average, boxed

Primitive int pipeline: range, rangeClosed, sum, average, boxed.
Key members are sum, average, boxed, range, rangeClosed, filter, map, mapToObj, mapToLong, mapToDouble, flatMap, mapMulti, distinct, sorted, peek, limit, skip, takeWhile.

```java
int sum()
java.util.OptionalDouble average()
java.util.stream.Stream<java.lang.Integer> boxed()
java.util.stream.IntStream range(int, int)
java.util.stream.IntStream rangeClosed(int, int)
java.util.stream.IntStream filter(java.util.function.IntPredicate)
java.util.stream.IntStream map(java.util.function.IntUnaryOperator)
```

## java.nio.file.Files: file operations on path: readstring, writestring, lines, walk, copy, createdirectories

File operations on Path: readString, writeString, lines, walk, copy, createDirectories.
Key members are createDirectories, copy, readString, writeString, walk, lines, newInputStream, newOutputStream, newByteChannel, newDirectoryStream, createFile, createDirectory, createTempFile, createTempDirectory, createSymbolicLink, createLink, delete, deleteIfExists.

```java
java.nio.file.Path createDirectories(java.nio.file.Path, java.nio.file.attribute.FileAttribute<?>...) throws java.io.IOException
java.nio.file.Path copy(java.nio.file.Path, java.nio.file.Path, java.nio.file.CopyOption...) throws java.io.IOException
long copy(java.io.InputStream, java.nio.file.Path, java.nio.file.CopyOption...) throws java.io.IOException
java.lang.String readString(java.nio.file.Path) throws java.io.IOException
java.lang.String readString(java.nio.file.Path, java.nio.charset.Charset) throws java.io.IOException
java.nio.file.Path writeString(java.nio.file.Path, java.lang.CharSequence, java.nio.file.OpenOption...) throws java.io.IOException
```

## java.nio.file.Path: immutable file path: of-factory, resolve, getparent, getfilename, toabsolutepath

Immutable file path: of-factory, resolve, getParent, getFileName, toAbsolutePath.
Key members are getFileName, getParent, resolve, toAbsolutePath, of, getFileSystem, isAbsolute, getRoot, getNameCount, getName, subpath, startsWith, endsWith, normalize, resolveSibling, relativize, toUri, toRealPath.

```java
java.nio.file.Path getFileName()
java.nio.file.Path getParent()
java.nio.file.Path resolve(java.nio.file.Path)
java.nio.file.Path resolve(java.lang.String)
java.nio.file.Path toAbsolutePath()
java.nio.file.Path of(java.lang.String, java.lang.String...)
java.nio.file.Path of(java.net.URI)
java.nio.file.FileSystem getFileSystem()
boolean isAbsolute()
java.nio.file.Path getRoot()
int getNameCount()
```

## java.nio.file.Paths: legacy path factory; new code uses path.of instead

Legacy Path factory; new code uses Path.of instead.
Key members are get.

```java
java.nio.file.Path get(java.lang.String, java.lang.String...)
java.nio.file.Path get(java.net.URI)
```

## java.io.BufferedReader: buffered character reading: readline and lines stream

Buffered character reading: readLine and lines stream.
Key members are readLine, lines, read, skip, ready, markSupported, mark, reset, close.

```java
java.lang.String readLine() throws java.io.IOException
java.util.stream.Stream<java.lang.String> lines()
java.io.BufferedReader(java.io.Reader, int)
java.io.BufferedReader(java.io.Reader)
int read() throws java.io.IOException
int read(char[], int, int) throws java.io.IOException
```

## java.io.IOException: checked exception thrown when file or stream i/o fails

Checked exception thrown when file or stream I/O fails.

```java
java.io.IOException()
java.io.IOException(java.lang.String)
```

## java.net.URI: parsed uniform resource identifier; the argument httprequest builders take

Parsed uniform resource identifier; the argument HttpRequest builders take.
Key members are create, parseServerAuthority, normalize, resolve, relativize, toURL, getScheme, isAbsolute, isOpaque, getRawSchemeSpecificPart, getSchemeSpecificPart, getRawAuthority, getAuthority, getRawUserInfo, getUserInfo, getHost, getPort, getRawPath.

```java
java.net.URI(java.lang.String) throws java.net.URISyntaxException
java.net.URI(java.lang.String, java.lang.String, java.lang.String, int, java.lang.String, java.lang.String, java.lang.String) throws java.net.URISyntaxException
java.net.URI create(java.lang.String)
java.net.URI parseServerAuthority() throws java.net.URISyntaxException
java.net.URI normalize()
java.net.URI resolve(java.net.URI)
```

## java.net.http.HttpClient: http client for synchronous send and asynchronous sendasync requests

HTTP client for synchronous send and asynchronous sendAsync requests.
Key members are send, sendAsync, newHttpClient, newBuilder, cookieHandler, connectTimeout, followRedirects, proxy, sslContext, sslParameters, authenticator, version, executor, newWebSocketBuilder, shutdown, awaitTermination, isTerminated, shutdownNow.

```java
<T> java.net.http.HttpResponse<T> send(java.net.http.HttpRequest, java.net.http.HttpResponse$BodyHandler<T>) throws java.io.IOException, java.lang.InterruptedException
<T> java.util.concurrent.CompletableFuture<java.net.http.HttpResponse<T>> sendAsync(java.net.http.HttpRequest, java.net.http.HttpResponse$BodyHandler<T>)
<T> java.util.concurrent.CompletableFuture<java.net.http.HttpResponse<T>> sendAsync(java.net.http.HttpRequest, java.net.http.HttpResponse$BodyHandler<T>, java.net.http.HttpResponse$PushPromiseHandler<T>)
java.net.http.HttpClient newHttpClient()
java.net.http.HttpClient$Builder newBuilder()
java.util.Optional<java.net.CookieHandler> cookieHandler()
```

## java.net.http.HttpRequest: immutable http request built with newbuilder: uri, header, get, post, timeout

Immutable HTTP request built with newBuilder: uri, header, GET, POST, timeout.
Key members are timeout, uri, newBuilder, bodyPublisher, method, expectContinue, version, headers, equals, hashCode.

```java
java.util.Optional<java.time.Duration> timeout()
java.net.URI uri()
java.net.http.HttpRequest$Builder newBuilder(java.net.URI)
java.net.http.HttpRequest$Builder newBuilder(java.net.http.HttpRequest, java.util.function.BiPredicate<java.lang.String, java.lang.String>)
java.util.Optional<java.net.http.HttpRequest$BodyPublisher> bodyPublisher()
java.lang.String method()
```

## java.net.http.HttpResponse: http response: statuscode, body, headers; body handlers pick the body type

HTTP response: statusCode, body, headers; body handlers pick the body type.
Key members are statusCode, body, request, previousResponse, headers, sslSession, uri, version.

```java
int statusCode()
T body()
java.net.http.HttpRequest request()
java.util.Optional<java.net.http.HttpResponse<T>> previousResponse()
java.net.http.HttpHeaders headers()
java.util.Optional<javax.net.ssl.SSLSession> sslSession()
java.net.URI uri()
```

## java.time.LocalDate: calendar date without time zone: now, of, parse, plusdays, format

Calendar date without time zone: now, of, parse, plusDays, format.
Key members are now, of, parse, plusDays, format, ofYearDay, ofInstant, ofEpochDay, from, isSupported, range, get, getLong, getChronology, getEra, getYear, getMonthValue, getMonth.

```java
java.time.LocalDate now()
java.time.LocalDate now(java.time.ZoneId)
java.time.LocalDate of(int, java.time.Month, int)
java.time.LocalDate of(int, int, int)
java.time.LocalDate parse(java.lang.CharSequence)
java.time.LocalDate parse(java.lang.CharSequence, java.time.format.DateTimeFormatter)
java.time.LocalDate plusDays(long)
```

## java.time.LocalDateTime: date and time without zone: now, of, parse, formatting with datetimeformatter

Date and time without zone: now, of, parse, formatting with DateTimeFormatter.
Key members are now, of, parse, with, ofInstant, ofEpochSecond, from, isSupported, range, get, getLong, toLocalDate, getYear, getMonthValue, getMonth, getDayOfMonth, getDayOfYear, getDayOfWeek.

```java
java.time.LocalDateTime now()
java.time.LocalDateTime now(java.time.ZoneId)
java.time.LocalDateTime of(int, java.time.Month, int, int, int)
java.time.LocalDateTime of(int, java.time.Month, int, int, int, int)
java.time.LocalDateTime parse(java.lang.CharSequence)
java.time.LocalDateTime parse(java.lang.CharSequence, java.time.format.DateTimeFormatter)
```

## java.time.Instant: machine timestamp on the utc timeline: now, ofepochmilli, plusseconds

Machine timestamp on the UTC timeline: now, ofEpochMilli, plusSeconds.
Key members are now, ofEpochMilli, plusSeconds, ofEpochSecond, from, parse, isSupported, range, get, getLong, getEpochSecond, getNano, with, truncatedTo, plus, plusMillis, plusNanos, minus.

```java
java.time.Instant now()
java.time.Instant now(java.time.Clock)
java.time.Instant ofEpochMilli(long)
java.time.Instant plusSeconds(long)
java.time.Instant ofEpochSecond(long)
java.time.Instant ofEpochSecond(long, long)
java.time.Instant from(java.time.temporal.TemporalAccessor)
java.time.Instant parse(java.lang.CharSequence)
```

## java.time.Duration: time-based amount: ofseconds, ofmillis, between, tomillis

Time-based amount: ofSeconds, ofMillis, between, toMillis.
Key members are ofSeconds, ofMillis, between, toMillis, ofDays, ofHours, ofMinutes, ofNanos, of, from, parse, get, getUnits, isPositive, isZero, isNegative, getSeconds, getNano.

```java
java.time.Duration ofSeconds(long)
java.time.Duration ofSeconds(long, long)
java.time.Duration ofMillis(long)
java.time.Duration between(java.time.temporal.Temporal, java.time.temporal.Temporal)
long toMillis()
java.time.Duration ofDays(long)
java.time.Duration ofHours(long)
java.time.Duration ofMinutes(long)
```

## java.time.ZonedDateTime: date-time with time zone: now, of, withzonesameinstant conversions

Date-time with time zone: now, of, withZoneSameInstant conversions.
Key members are now, of, withZoneSameInstant, with, ofLocal, ofInstant, ofStrict, from, parse, isSupported, range, get, getLong, getOffset, withEarlierOffsetAtOverlap, withLaterOffsetAtOverlap, getZone, withZoneSameLocal.

```java
java.time.ZonedDateTime now()
java.time.ZonedDateTime now(java.time.ZoneId)
java.time.ZonedDateTime of(java.time.LocalDate, java.time.LocalTime, java.time.ZoneId)
java.time.ZonedDateTime of(java.time.LocalDateTime, java.time.ZoneId)
java.time.ZonedDateTime withZoneSameInstant(java.time.ZoneId)
java.time.ZonedDateTime with(java.time.temporal.TemporalAdjuster)
```

## java.time.format.DateTimeFormatter: formatting and parsing patterns: ofpattern, iso constants

Formatting and parsing patterns: ofPattern, ISO constants.
Key members are ofPattern, ofLocalizedDate, ofLocalizedTime, ofLocalizedDateTime, ofLocalizedPattern, parsedExcessDays, parsedLeapSecond, getLocale, withLocale, localizedBy, getDecimalStyle, withDecimalStyle, getChronology, withChronology, getZone, withZone, getResolverStyle, withResolverStyle.

```java
java.time.format.DateTimeFormatter ofPattern(java.lang.String)
java.time.format.DateTimeFormatter ofPattern(java.lang.String, java.util.Locale)
java.time.format.DateTimeFormatter ofLocalizedDate(java.time.format.FormatStyle)
java.time.format.DateTimeFormatter ofLocalizedTime(java.time.format.FormatStyle)
java.time.format.DateTimeFormatter ofLocalizedDateTime(java.time.format.FormatStyle)
java.time.format.DateTimeFormatter ofLocalizedDateTime(java.time.format.FormatStyle, java.time.format.FormatStyle)
```

## java.util.concurrent.CompletableFuture: asynchronous result composition: supplyasync, thenapply, thencompose, exceptionally, join

Asynchronous result composition: supplyAsync, thenApply, thenCompose, exceptionally, join.
Key members are supplyAsync, join, thenApply, thenCompose, exceptionally, runAsync, completedFuture, isDone, get, getNow, resultNow, exceptionNow, complete, completeExceptionally, thenApplyAsync, thenAccept, thenAcceptAsync, thenRun.

```java
<U> java.util.concurrent.CompletableFuture<U> supplyAsync(java.util.function.Supplier<U>)
<U> java.util.concurrent.CompletableFuture<U> supplyAsync(java.util.function.Supplier<U>, java.util.concurrent.Executor)
T join()
<U> java.util.concurrent.CompletableFuture<U> thenApply(java.util.function.Function<? super T, ? extends U>)
<U> java.util.concurrent.CompletableFuture<U> thenCompose(java.util.function.Function<? super T, ? extends java.util.concurrent.CompletionStage<U>>)
java.util.concurrent.CompletableFuture<T> exceptionally(java.util.function.Function<java.lang.Throwable, ? extends T>)
```

## java.util.concurrent.ExecutorService: thread pool interface: submit, invokeall, shutdown, close

Thread pool interface: submit, invokeAll, shutdown, close.
Key members are shutdown, submit, invokeAll, close, shutdownNow, isShutdown, isTerminated, awaitTermination, invokeAny.

```java
void shutdown()
<T> java.util.concurrent.Future<T> submit(java.util.concurrent.Callable<T>)
<T> java.util.concurrent.Future<T> submit(java.lang.Runnable, T)
<T> java.util.List<java.util.concurrent.Future<T>> invokeAll(java.util.Collection<? extends java.util.concurrent.Callable<T>>) throws java.lang.InterruptedException
<T> java.util.List<java.util.concurrent.Future<T>> invokeAll(java.util.Collection<? extends java.util.concurrent.Callable<T>>, long, java.util.concurrent.TimeUnit) throws java.lang.InterruptedException
void close()
```

## java.util.concurrent.Executors: thread pool factories: newfixedthreadpool, newvirtualthreadpertaskexecutor, newscheduledthreadpool

Thread pool factories: newFixedThreadPool, newVirtualThreadPerTaskExecutor, newScheduledThreadPool.
Key members are newFixedThreadPool, newVirtualThreadPerTaskExecutor, newScheduledThreadPool, newWorkStealingPool, newSingleThreadExecutor, newCachedThreadPool, newThreadPerTaskExecutor, newSingleThreadScheduledExecutor, unconfigurableExecutorService, unconfigurableScheduledExecutorService, defaultThreadFactory, privilegedThreadFactory, callable, privilegedCallable, privilegedCallableUsingCurrentClassLoader.

```java
java.util.concurrent.ExecutorService newFixedThreadPool(int)
java.util.concurrent.ExecutorService newFixedThreadPool(int, java.util.concurrent.ThreadFactory)
java.util.concurrent.ExecutorService newVirtualThreadPerTaskExecutor()
java.util.concurrent.ScheduledExecutorService newScheduledThreadPool(int)
java.util.concurrent.ScheduledExecutorService newScheduledThreadPool(int, java.util.concurrent.ThreadFactory)
java.util.concurrent.ExecutorService newWorkStealingPool(int)
java.util.concurrent.ExecutorService newWorkStealingPool()
java.util.concurrent.ExecutorService newSingleThreadExecutor()
```

## java.util.concurrent.ConcurrentHashMap: thread-safe map: computeifabsent for caches

Thread-safe map: computeIfAbsent for caches. Unlike HashMap it rejects null keys AND null values, throwing NullPointerException immediately on put.
Key members are put, values, computeIfAbsent, keys, size, isEmpty, get, containsKey, containsValue, putAll, remove, clear, keySet, entrySet, hashCode, toString, equals, putIfAbsent.

```java
V put(K, V)
java.util.Collection<V> values()
V computeIfAbsent(K, java.util.function.Function<? super K, ? extends V>)
java.util.Enumeration<K> keys()
java.util.concurrent.ConcurrentHashMap()
java.util.concurrent.ConcurrentHashMap(int)
int size()
boolean isEmpty()
V get(java.lang.Object)
boolean containsKey(java.lang.Object)
boolean containsValue(java.lang.Object)
void putAll(java.util.Map<? extends K, ? extends V>)
```

## java.util.concurrent.TimeUnit: time granularity for timeouts: seconds, milliseconds, sleep and conversions

Time granularity for timeouts: SECONDS, MILLISECONDS, sleep and conversions.
Key members are sleep, values, valueOf, convert, toNanos, toMicros, toMillis, toSeconds, toMinutes, toHours, toDays, timedWait, timedJoin, toChronoUnit, of.

```java
void sleep(long) throws java.lang.InterruptedException
java.util.concurrent.TimeUnit[] values()
java.util.concurrent.TimeUnit valueOf(java.lang.String)
long convert(long, java.util.concurrent.TimeUnit)
long convert(java.time.Duration)
long toNanos(long)
long toMicros(long)
long toMillis(long)
long toSeconds(long)
long toMinutes(long)
```

## java.util.concurrent.CountDownLatch: one-shot synchronization barrier: countdown and await

One-shot synchronization barrier: countDown and await.
Key members are await, countDown, getCount, toString.

```java
void await() throws java.lang.InterruptedException
boolean await(long, java.util.concurrent.TimeUnit) throws java.lang.InterruptedException
void countDown()
java.util.concurrent.CountDownLatch(int)
long getCount()
java.lang.String toString()
```

## java.util.concurrent.atomic.AtomicInteger: lock-free integer counter: incrementandget, compareandset

Lock-free integer counter: incrementAndGet, compareAndSet.
Key members are compareAndSet, incrementAndGet, get, set, lazySet, getAndSet, weakCompareAndSet, weakCompareAndSetPlain, getAndIncrement, getAndDecrement, getAndAdd, decrementAndGet, addAndGet, getAndUpdate, updateAndGet, getAndAccumulate, accumulateAndGet, toString.

```java
boolean compareAndSet(int, int)
int incrementAndGet()
java.util.concurrent.atomic.AtomicInteger(int)
java.util.concurrent.atomic.AtomicInteger()
int get()
void set(int)
void lazySet(int)
int getAndSet(int)
boolean weakCompareAndSet(int, int)
boolean weakCompareAndSetPlain(int, int)
int getAndIncrement()
int getAndDecrement()
int getAndAdd(int)
int decrementAndGet()
int addAndGet(int)
int getAndUpdate(java.util.function.IntUnaryOperator)
```

## java.util.regex.Pattern: compiled regular expression: compile, matcher, and the matches test

Compiled regular expression: compile, matcher, and the matches test.
Key members are compile, matcher, matches, pattern, toString, flags, split, splitWithDelimiters, quote, namedGroups, asPredicate, asMatchPredicate, splitAsStream.

```java
java.util.regex.Pattern compile(java.lang.String)
java.util.regex.Pattern compile(java.lang.String, int)
java.util.regex.Matcher matcher(java.lang.CharSequence)
boolean matches(java.lang.String, java.lang.CharSequence)
java.lang.String pattern()
java.lang.String toString()
int flags()
```

## java.util.regex.Matcher: regex match state: find, group, replaceall, start and end offsets

Regex match state: find, group, replaceAll, start and end offsets.
Key members are start, end, group, find, replaceAll, pattern, toMatchResult, usePattern, reset, groupCount, matches, lookingAt, quoteReplacement, appendReplacement, appendTail, results, replaceFirst, region.

```java
int start()
int start(int)
int end()
int end(int)
java.lang.String group()
java.lang.String group(int)
boolean find()
boolean find(int)
java.lang.String replaceAll(java.lang.String)
java.lang.String replaceAll(java.util.function.Function<java.util.regex.MatchResult, java.lang.String>)
java.util.regex.Pattern pattern()
java.util.regex.MatchResult toMatchResult()
```

## java.util.Scanner: text tokenizing over input streams and strings: nextline, nextint, hasnext

Text tokenizing over input streams and strings: nextLine, nextInt, hasNext.
Key members are hasNext, nextLine, nextInt, close, ioException, delimiter, useDelimiter, locale, useLocale, radix, useRadix, match, toString, next, remove, hasNextLine, findInLine, findWithinHorizon.

```java
boolean hasNext()
boolean hasNext(java.lang.String)
java.lang.String nextLine()
int nextInt()
int nextInt(int)
java.util.Scanner(java.lang.Readable)
java.util.Scanner(java.io.InputStream)
void close()
java.io.IOException ioException()
java.util.regex.Pattern delimiter()
java.util.Scanner useDelimiter(java.util.regex.Pattern)
java.util.Scanner useDelimiter(java.lang.String)
```

## java.util.Random: pseudorandom numbers: nextint with bound, ints stream; not for security

Pseudorandom numbers: nextInt with bound, ints stream; not for security.
Key members are nextInt, ints, from, setSeed, nextBytes, nextLong, nextBoolean, nextFloat, nextDouble, nextGaussian, longs, doubles.

```java
int nextInt()
int nextInt(int)
java.util.stream.IntStream ints(long)
java.util.stream.IntStream ints()
java.util.Random()
java.util.Random(long)
java.util.Random from(java.util.random.RandomGenerator)
void setSeed(long)
void nextBytes(byte[])
long nextLong()
boolean nextBoolean()
float nextFloat()
```

## java.security.SecureRandom: cryptographically strong random numbers for tokens and salts

Cryptographically strong random numbers for tokens and salts.
Key members are getInstance, getProvider, getAlgorithm, toString, getParameters, setSeed, nextBytes, getSeed, generateSeed, getInstanceStrong, reseed.

```java
java.security.SecureRandom()
java.security.SecureRandom(byte[])
java.security.SecureRandom getInstance(java.lang.String) throws java.security.NoSuchAlgorithmException
java.security.SecureRandom getInstance(java.lang.String, java.lang.String) throws java.security.NoSuchAlgorithmException, java.security.NoSuchProviderException
java.security.Provider getProvider()
java.lang.String getAlgorithm()
```

## java.security.MessageDigest: cryptographic hashing: getinstance sha-256, update, digest

Cryptographic hashing: getInstance SHA-256, update, digest.
Key members are getInstance, update, digest, getProvider, toString, isEqual, reset, getAlgorithm, getDigestLength, clone.

```java
java.security.MessageDigest getInstance(java.lang.String) throws java.security.NoSuchAlgorithmException
java.security.MessageDigest getInstance(java.lang.String, java.lang.String) throws java.security.NoSuchAlgorithmException, java.security.NoSuchProviderException
void update(byte)
void update(byte[], int, int)
byte[] digest()
int digest(byte[], int, int) throws java.security.DigestException
```

## java.util.Base64: base64 encoding and decoding: getencoder, getdecoder, geturlencoder

Base64 encoding and decoding: getEncoder, getDecoder, getUrlEncoder.
Key members are getEncoder, getUrlEncoder, getDecoder, getMimeEncoder, getUrlDecoder, getMimeDecoder.

```java
java.util.Base64$Encoder getEncoder()
java.util.Base64$Encoder getUrlEncoder()
java.util.Base64$Decoder getDecoder()
java.util.Base64$Encoder getMimeEncoder()
java.util.Base64$Encoder getMimeEncoder(int, byte[])
java.util.Base64$Decoder getUrlDecoder()
```

## java.util.UUID: universally unique identifiers: randomuuid, fromstring, tostring

Universally unique identifiers: randomUUID, fromString, toString.
Key members are randomUUID, fromString, toString, nameUUIDFromBytes, getLeastSignificantBits, getMostSignificantBits, version, variant, timestamp, clockSequence, node, hashCode, equals, compareTo.

```java
java.util.UUID randomUUID()
java.util.UUID fromString(java.lang.String)
java.lang.String toString()
java.util.UUID(long, long)
java.util.UUID nameUUIDFromBytes(byte[])
long getLeastSignificantBits()
long getMostSignificantBits()
int version()
int variant()
long timestamp()
int clockSequence()
long node()
int hashCode()
boolean equals(java.lang.Object)
```

## java.util.Iterator: sequential element access: hasnext, next, remove

Sequential element access: hasNext, next, remove.
Key members are hasNext, next, remove, forEachRemaining.

```java
boolean hasNext()
E next()
void remove()
void forEachRemaining(java.util.function.Consumer<? super E>)
```

## java.util.Comparator: ordering strategy: comparing, thencomparing, reversed, naturalorder

Ordering strategy: comparing, thenComparing, reversed, naturalOrder.
Key members are reversed, thenComparing, naturalOrder, comparing, compare, equals, thenComparingInt, thenComparingLong, thenComparingDouble, reverseOrder, nullsFirst, nullsLast, comparingInt, comparingLong, comparingDouble.

```java
java.util.Comparator<T> reversed()
java.util.Comparator<T> thenComparing(java.util.Comparator<? super T>)
<U> java.util.Comparator<T> thenComparing(java.util.function.Function<? super T, ? extends U>, java.util.Comparator<? super U>)
<T extends java.lang.Comparable<? super T>> java.util.Comparator<T> naturalOrder()
<T, U> java.util.Comparator<T> comparing(java.util.function.Function<? super T, ? extends U>, java.util.Comparator<? super U>)
<T, U extends java.lang.Comparable<? super U>> java.util.Comparator<T> comparing(java.util.function.Function<? super T, ? extends U>)
```

## java.util.StringJoiner: joining strings with delimiter, prefix and suffix

Joining strings with delimiter, prefix and suffix.
Key members are setEmptyValue, toString, add, merge, length.

```java
java.util.StringJoiner(java.lang.CharSequence)
java.util.StringJoiner(java.lang.CharSequence, java.lang.CharSequence, java.lang.CharSequence)
java.util.StringJoiner setEmptyValue(java.lang.CharSequence)
java.lang.String toString()
java.util.StringJoiner add(java.lang.CharSequence)
java.util.StringJoiner merge(java.util.StringJoiner)
```

## java.lang.Integer: boxed int: parseint, valueof, max_value, tostring with radix

Boxed int: parseInt, valueOf, MAX_VALUE, toString with radix.
Key members are toString, parseInt, valueOf, toUnsignedString, toHexString, toOctalString, toBinaryString, parseUnsignedInt, byteValue, shortValue, intValue, longValue, floatValue, doubleValue, hashCode, equals, getInteger, decode.

```java
java.lang.String toString(int, int)
java.lang.String toString(int)
int parseInt(java.lang.String, int) throws java.lang.NumberFormatException
int parseInt(java.lang.CharSequence, int, int, int) throws java.lang.NumberFormatException
java.lang.Integer valueOf(java.lang.String, int) throws java.lang.NumberFormatException
java.lang.Integer valueOf(java.lang.String) throws java.lang.NumberFormatException
```

## java.lang.Long: boxed long: parselong, valueof, and unsigned helpers

Boxed long: parseLong, valueOf, and unsigned helpers.
Key members are parseLong, valueOf, toString, toUnsignedString, toHexString, toOctalString, toBinaryString, parseUnsignedLong, decode, byteValue, shortValue, intValue, longValue, floatValue, doubleValue, hashCode, equals, getLong.

```java
long parseLong(java.lang.String, int) throws java.lang.NumberFormatException
long parseLong(java.lang.CharSequence, int, int, int) throws java.lang.NumberFormatException
java.lang.Long valueOf(java.lang.String, int) throws java.lang.NumberFormatException
java.lang.Long valueOf(java.lang.String) throws java.lang.NumberFormatException
java.lang.String toString(long, int)
java.lang.String toUnsignedString(long, int)
```

## java.lang.Double: boxed double: parsedouble, isnan, compare

Boxed double: parseDouble, isNaN, compare.
Key members are parseDouble, isNaN, compare, toString, toHexString, valueOf, isInfinite, isFinite, byteValue, shortValue, intValue, longValue, floatValue, doubleValue, hashCode, equals, doubleToLongBits, doubleToRawLongBits.

```java
double parseDouble(java.lang.String) throws java.lang.NumberFormatException
boolean isNaN(double)
boolean isNaN()
int compare(double, double)
java.lang.String toString(double)
java.lang.String toHexString(double)
java.lang.Double valueOf(java.lang.String) throws java.lang.NumberFormatException
java.lang.Double valueOf(double)
```

## java.lang.Character: char tests and conversions: isdigit, isletter, tolowercase

Char tests and conversions: isDigit, isLetter, toLowerCase.
Key members are isDigit, isLetter, toLowerCase, describeConstable, valueOf, charValue, hashCode, equals, toString, isValidCodePoint, isBmpCodePoint, isSupplementaryCodePoint, isHighSurrogate, isLowSurrogate, isSurrogate, isSurrogatePair, charCount, toCodePoint.

```java
boolean isDigit(char)
boolean isDigit(int)
boolean isLetter(char)
boolean isLetter(int)
char toLowerCase(char)
int toLowerCase(int)
java.util.Optional<java.lang.constant.DynamicConstantDesc<java.lang.Character>> describeConstable()
java.lang.Character(char)
java.lang.Character valueOf(char)
char charValue()
int hashCode()
int hashCode(char)
boolean equals(java.lang.Object)
java.lang.String toString()
```

## java.lang.Math: numeric functions: abs, max, min, pow, sqrt, floordiv, clamp

Numeric functions: abs, max, min, pow, sqrt, floorDiv, clamp.
Key members are sqrt, pow, floorDiv, abs, max, min, clamp, sin, cos, tan, asin, acos, atan, toRadians, toDegrees, exp, log, log10.

```java
double sqrt(double)
double pow(double, double)
int floorDiv(int, int)
long floorDiv(long, int)
int abs(int)
long abs(long)
int max(int, int)
long max(long, long)
int min(int, int)
long min(long, long)
int clamp(long, int, int)
long clamp(long, long, long)
double sin(double)
```

## java.lang.Thread: threads: ofvirtual and ofplatform builders, sleep, currentthread, interrupt

Threads: ofVirtual and ofPlatform builders, sleep, currentThread, interrupt.
Key members are currentThread, sleep, ofPlatform, ofVirtual, interrupt, yield, onSpinWait, startVirtualThread, isVirtual, start, run, stop, interrupted, isInterrupted, isAlive, suspend, resume, setPriority.

```java
java.lang.Thread currentThread()
void sleep(long) throws java.lang.InterruptedException
void sleep(long, int) throws java.lang.InterruptedException
java.lang.Thread$Builder$OfPlatform ofPlatform()
java.lang.Thread$Builder$OfVirtual ofVirtual()
void interrupt()
void yield()
void onSpinWait()
java.lang.Thread()
java.lang.Thread(java.lang.Runnable)
```

## java.lang.Runtime: jvm runtime: availableprocessors, addshutdownhook, exec is legacy

JVM runtime: availableProcessors, addShutdownHook, exec is legacy.
Key members are addShutdownHook, exec, availableProcessors, getRuntime, exit, removeShutdownHook, halt, freeMemory, totalMemory, maxMemory, gc, runFinalization, load, loadLibrary, version.

```java
void addShutdownHook(java.lang.Thread)
java.lang.Process exec(java.lang.String) throws java.io.IOException
java.lang.Process exec(java.lang.String, java.lang.String[]) throws java.io.IOException
int availableProcessors()
java.lang.Runtime getRuntime()
void exit(int)
boolean removeShutdownHook(java.lang.Thread)
void halt(int)
long freeMemory()
```

## java.lang.ProcessBuilder: launching external processes: command list, redirecterrorstream, start

Launching external processes: command list, redirectErrorStream, start.
Key members are command, redirectErrorStream, start, environment, directory, redirectInput, redirectOutput, redirectError, inheritIO, startPipeline.

```java
java.lang.ProcessBuilder command(java.util.List<java.lang.String>)
java.lang.ProcessBuilder command(java.lang.String...)
boolean redirectErrorStream()
java.lang.ProcessBuilder redirectErrorStream(boolean)
java.lang.Process start() throws java.io.IOException
java.lang.ProcessBuilder(java.util.List<java.lang.String>)
```

## java.lang.Process: a running external process: waitfor, exitvalue, getinputstream, onexit

A running external process: waitFor, exitValue, getInputStream, onExit.
Key members are getInputStream, waitFor, exitValue, onExit, getOutputStream, getErrorStream, inputReader, errorReader, outputWriter, destroy, destroyForcibly, supportsNormalTermination, isAlive, pid, toHandle, info, children, descendants.

```java
java.io.InputStream getInputStream()
int waitFor() throws java.lang.InterruptedException
boolean waitFor(long, java.util.concurrent.TimeUnit) throws java.lang.InterruptedException
int exitValue()
java.util.concurrent.CompletableFuture<java.lang.Process> onExit()
java.lang.Process()
java.io.OutputStream getOutputStream()
java.io.InputStream getErrorStream()
java.io.BufferedReader inputReader()
```

## java.lang.AutoCloseable: the try-with-resources contract: close

The try-with-resources contract: close.
Key members are close.

```java
void close() throws java.lang.Exception
```

## java.lang.Iterable: the for-each contract: iterator, foreach, spliterator

The for-each contract: iterator, forEach, spliterator.
Key members are iterator, forEach, spliterator.

```java
java.util.Iterator<T> iterator()
void forEach(java.util.function.Consumer<? super T>)
java.util.Spliterator<T> spliterator()
```

## java.lang.Record: base class of record types: compact immutable data carriers

Base class of record types: compact immutable data carriers.
Key members are equals, hashCode, toString.

```java
boolean equals(java.lang.Object)
int hashCode()
java.lang.String toString()
```

## java.lang.Exception: base checked exception: message, cause, suppressed exceptions

Base checked exception: message, cause, suppressed exceptions.

```java
java.lang.Exception()
java.lang.Exception(java.lang.String)
```

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
