# c-safety corpus

C is the language where the compiler assumes you were right. Every entry
here is a rule whose violation compiles cleanly, passes a smoke test,
and fails in production or in an exploit.

The selection bias is deliberate: not "how C works" — a competent model
already knows that — but the specific places where the obvious code is
wrong and looks right.

## strncpy does not guarantee a terminator

`strncpy(dst, src, n)` copies at most `n` bytes and writes **no**
terminator if `strlen(src) >= n`. The result is a non-string that every
later `strlen` or `printf("%s")` runs off the end of. If it must be
used, terminate explicitly: `dst[n - 1] = '\0';`. It also zero-pads to
the full `n` on short sources, which is a performance trap on large
buffers.

## snprintf tells you about truncation, if you check it

`snprintf` always terminates, but it returns **the length it would have
written**, not the length it wrote. So `int r = snprintf(buf, sizeof buf,
...); if (r < 0 || (size_t)r >= sizeof buf) { /* truncated */ }` is the
only correct use. Ignoring the return silently produces truncated paths,
truncated commands, and truncated security decisions.

## strlcpy and strlcat are the ones that behave

`strlcpy(dst, src, size)` always terminates and returns the length of
the source, so `>= size` means truncation. Available on BSD and macOS
natively and in glibc since 2.38; on older Linux, either vendor them or
use the `snprintf` idiom. `strcat` in a loop is O(n²) as well as unsafe —
track the offset instead.

## The functions with no bound at all

`gets` (removed from C11), `strcpy`, `strcat`, `sprintf`, `scanf("%s")`
and `vsprintf` take no destination size. There is no safe way to call
them on data whose length you do not already control. For `scanf`, a
width is mandatory: `scanf("%63s", buf)` where `buf` is 64 bytes — the
width excludes the terminator.

## printf with a non-literal format is a vulnerability

`printf(user_input)` lets the input specify conversions: `%n` writes to
memory, `%s` dereferences a stack value. The rule is that the format
string is always a literal. `printf("%s", user_input)` is the fix, and
`-Wformat-security` (or `-Wformat=2`) makes the compiler enforce it.

## Signed integer overflow is undefined; unsigned wraps

`INT_MAX + 1` is undefined behaviour, and the optimiser uses that: it
will delete `if (x + 1 < x)` entirely because signed overflow "cannot"
happen. Unsigned arithmetic wraps modulo 2ⁿ and is fully defined. Check
*before* the operation — `if (a > INT_MAX - b)` — or use
`__builtin_add_overflow(a, b, &r)`, which is supported by both GCC and
Clang and returns true on overflow.

## Unsigned subtraction underflows into a huge number

`size_t` is unsigned, so `if (len - 1 >= 0)` is always true and
`buf[len - 1]` on `len == 0` indexes `SIZE_MAX`. Any expression of the
form `a - b` where both are unsigned needs `if (a >= b)` first. This is
the single most common C bug that looks like a bounds check and is not
one.

## Integer promotion changes the type before the comparison

Anything narrower than `int` is promoted to `int` in an expression, and
comparing signed with unsigned converts the signed operand to unsigned.
So `int i = -1; unsigned u = 1; i < u` is **false**. Compile with
`-Wsign-compare` (included in `-Wextra`) and do not silence it with a
cast until you have worked out which way the conversion goes.

## Shifting is undefined at or past the width, and into the sign bit

`x << n` is undefined when `n >= width of x` or `n < 0`, and for signed
`x`, when the result overflows. `1 << 31` on 32-bit `int` is undefined;
`1u << 31` is fine. For a 64-bit result from 32-bit operands the cast
goes on the *operand*: `(uint64_t)1 << 40`, not `(uint64_t)(1 << 40)`,
which has already overflowed.

## Check every allocation, and do not assign realloc to its own pointer

`p = realloc(p, n)` leaks the original block when realloc returns NULL.
The correct form keeps the old pointer alive:
```c
void *tmp = realloc(p, n);
if (!tmp) { /* p is still valid; clean up */ return -1; }
p = tmp;
```
`malloc(0)` may legitimately return NULL or a unique pointer — do not
treat NULL from a zero-size request as failure.

## Multiplication in a size argument overflows before malloc sees it

`malloc(count * sizeof(struct item))` wraps if `count` is attacker-
controlled, allocating a small buffer for a large loop — a textbook
heap overflow. Either check `count > SIZE_MAX / sizeof(struct item)`
first, or use `calloc(count, sizeof(struct item))`, which is required to
detect the overflow itself.

## Free the pointer, then forget it

After `free(p)`, `p` is indeterminate — even *reading* its value is
undefined, never mind dereferencing it. Set `p = NULL` immediately;
`free(NULL)` is defined as a no-op, so this also makes double-free
impossible on that path. Freeing a pointer that was not returned by
malloc — an interior pointer, a stack address — is undefined.

## Never return a pointer to a local

The lifetime of an automatic object ends at the closing brace. Returning
`&local` or a pointer to a local array yields a dangling pointer that
usually still "works" in a debug build and corrupts memory under
optimisation. Return by value, take a caller-provided buffer, or
allocate.

## sizeof an array parameter is the size of a pointer

`void f(char buf[64]) { sizeof buf; }` yields 8, not 64 — array
parameters decay to pointers regardless of the declared bound. The
length must travel as a separate argument. `sizeof` is only trustworthy
where the array is actually declared, and there `sizeof arr / sizeof
arr[0]` gives the element count.

## memcpy on overlapping regions is undefined; memmove is not

`memcpy` is allowed to assume the regions are disjoint and vectorises on
that assumption, so a self-shifting buffer silently corrupts. Use
`memmove` whenever the regions might overlap. Passing NULL to `memcpy`
is undefined even when `n` is zero.

## Type punning through a cast breaks strict aliasing

`float f = *(float *)&i;` is undefined: the optimiser may reorder or
elide accesses because it assumes an `int*` and a `float*` never refer
to the same object. Use `memcpy(&f, &i, sizeof f)` — every mainstream
compiler recognises the idiom and emits the same single instruction — or
a `union`, which C (unlike C++) explicitly permits for this.

## ctype functions take an int that must be an unsigned char

`isalpha(c)` where `c` is a plain `char` is undefined for negative
values, and `char` is signed on x86 Linux. Any byte above 127 becomes a
negative index into the ctype table. The correct call is
`isalpha((unsigned char)c)`. The same applies to `toupper`, `isspace`,
and the rest of the family.

## getchar returns int, not char

It returns `EOF` (a negative value, typically -1) which does not fit in
`char`. Storing the result in a `char` makes the EOF test either never
fire or fire on a legitimate 0xFF byte. `int c; while ((c = getchar()) !=
EOF)` is the only correct shape.

## atoi cannot report failure; strtol can

`atoi("abc")` returns 0, indistinguishable from `atoi("0")`, and its
behaviour on overflow is undefined. Use `strtol`:
```c
errno = 0;
char *end;
long v = strtol(s, &end, 10);
if (end == s || *end || errno == ERANGE) { /* invalid */ }
```
Checking `end` catches trailing garbage; checking `errno` catches range.

## errno is only meaningful after a failure is indicated

Library functions do not clear `errno` on success, so reading it without
first seeing a failure return gives a stale value from some earlier call.
Set `errno = 0` before calls (like `strtol`) that signal errors only
through it, and read it *immediately* after the failing call, before any
other library call — including `printf`, which can overwrite it.

## Variable-length arrays put attacker-controlled sizes on the stack

`char buf[n];` with an unvalidated `n` is a stack-clash primitive, and
there is no way to detect the failure — there is no NULL to check. VLAs
are optional in C11 and later. Use a fixed bound with an explicit check,
or heap-allocate.

## alloca has the same problem and no error path

`alloca` cannot fail gracefully; exceeding the stack is a crash at best.
It also interacts badly with inlining, since the "function scope" it
frees at may not be the one you wrote. Treat it as unavailable.

## The goto-cleanup pattern is the idiomatic way to not leak

C has no destructors, so multi-resource functions use a single exit
path:
```c
int f(void) {
    int rc = -1;
    FILE *fp = NULL; char *buf = NULL;
    fp = fopen(path, "r");   if (!fp)  goto out;
    buf = malloc(n);         if (!buf) goto out;
    /* … */
    rc = 0;
out:
    free(buf);
    if (fp) fclose(fp);
    return rc;
}
```
Every resource is initialised to a safe value first so the cleanup block
is unconditionally correct no matter where the jump came from.

## Comparing secrets with memcmp leaks them through timing

`memcmp` returns at the first differing byte, so the time it takes
reveals the length of the matching prefix — enough to recover a token
byte by byte. Compare in constant time by accumulating:
`for (i = 0, d = 0; i < n; i++) d |= a[i] ^ b[i]; return d == 0;`

## memset to erase a secret gets optimised away

If the buffer is not read after the `memset`, the compiler is free to
delete the call as dead — and does, at `-O2`. Use
`explicit_bzero` (BSD, glibc 2.25+), `memset_s` (C11 Annex K, optional),
or `SecureZeroMemory` on Windows. `volatile` on the buffer is the
portable fallback.

## volatile is not atomic and is not a memory barrier

`volatile` prevents the compiler from caching a value in a register. It
does not make read-modify-write atomic, does not order accesses to other
objects, and does not synchronise across cores. For concurrency use
`_Atomic` / `<stdatomic.h>`; `volatile` is for memory-mapped I/O and
`sig_atomic_t` in signal handlers.

## Signal handlers may only call async-signal-safe functions

Inside a handler, `printf`, `malloc` and most of the standard library
are undefined — they can deadlock on a lock the interrupted code holds.
The safe set is small (`write`, `_exit`, `signal`). The standard pattern
is to set a `volatile sig_atomic_t` flag and do the real work in the
main loop.

## Pointer arithmetic outside an object is undefined, even without dereferencing

Forming a pointer more than one past the end of an array is undefined by
itself. So `if (p + len > end)` can be optimised away when the compiler
proves `p + len` would be out of bounds. The correct bounds check
subtracts instead: `if (len > (size_t)(end - p))`.

## Comparing or subtracting pointers into different objects is undefined

`ptr_a < ptr_b` is only defined when both point into the same array
object (or one past its end). Ordering pointers from two separate
allocations, and `ptrdiff_t` from subtracting them, are both undefined —
which matters for hand-rolled overlap checks.

## Struct layout carries padding you cannot ignore

The compiler inserts padding for alignment, so `sizeof(struct)` is not
the sum of its members, `memcmp` on two structs compares uninitialised
padding bytes, and writing a struct to disk or a socket bakes in your
compiler's ABI. Serialise field by field. Zero with `= {0}` or `memset`
before use if the padding will ever be copied.

## const on a pointer: read it right to left

`const char *p` — pointer to const char, the data cannot be modified
through `p`, but `p` can be reassigned. `char *const p` — const pointer,
`p` cannot be reassigned. `const char *const p` — neither. Casting away
const and then writing is undefined if the object was actually declared
const.

## static gives a function or global internal linkage

At file scope, `static` restricts a symbol to that translation unit —
it is the only encapsulation C offers, and it lets the optimiser inline
and prove things it otherwise cannot. Every function and global not in
the public header should be `static`. Inside a function, `static` means
a single instance shared across all calls, which makes the function
non-reentrant and thread-unsafe.

## restrict is a promise you make and the compiler believes

`void copy(char *restrict dst, const char *restrict src, size_t n)`
asserts the pointers do not alias, permitting vectorisation. If they do
alias, the behaviour is undefined and the corruption is silent. Only
apply it where the contract is documented and enforced at the call site.

## Flexible array members, not the [1] hack

C99 allows `struct msg { size_t len; char data[]; };` allocated as
`malloc(sizeof(struct msg) + len)`. The old `char data[1]` trick makes
every index past zero technically out of bounds, which the sanitizers
correctly flag. `sizeof` a struct with a flexible member excludes it,
and the allocation arithmetic still needs its own overflow check.

## Reading an uninitialised variable is undefined, not "random"

It is not merely an unpredictable value: the compiler may assume the
read never happens and delete the code around it, and the same variable
may read as two different values in two places. Initialise at
declaration. `-Wmaybe-uninitialized` catches some cases; MemorySanitizer
catches the rest.

## Build with the flags that turn bugs into errors

`-Wall -Wextra -Wpedantic -Werror` is the baseline, plus
`-Wshadow -Wconversion -Wsign-conversion -Wcast-qual -Wvla
-Wformat=2 -Wnull-dereference`. Add
`-D_FORTIFY_SOURCE=3 -fstack-protector-strong` and, for hardening,
`-fPIE -Wl,-z,relro,-z,now`. These cost nothing at runtime and catch a
whole class of the entries above at compile time.

## Sanitizers find at runtime what the compiler cannot prove

`-fsanitize=address,undefined -fno-omit-frame-pointer` catches
use-after-free, buffer overflow, and most UB in this document, at
roughly 2× slowdown — fast enough for the whole test suite.
`-fsanitize=thread` finds data races; it cannot be combined with ASan.
`-fsanitize=memory` (Clang) finds uninitialised reads but needs the
whole program, libc included, instrumented. Run them in CI, not just
locally.

## assert disappears under NDEBUG

`assert(p = malloc(n))` — note the single `=` — is a real pattern in
real code, and the entire allocation vanishes in a release build.
Assertions are for programmer invariants only; never put a side effect
or a runtime-input validation inside one. Input validation is `if` and
an error return.

## Check the file operation, not just the open

`fclose` can fail — buffered data is flushed there, so a full disk
surfaces at close, not at write. `fwrite` returns the number of *items*
written, and a short write is not an error return. `fread` returning
less than requested means EOF or error, distinguished by `feof` /
`ferror`, not by the count.

## Time-of-check to time-of-use

`access(path, W_OK)` followed by `fopen(path, "w")` is a race: the path
can be replaced with a symlink between the two calls. Operate on the
handle, not the name — `open` with `O_NOFOLLOW` / `O_EXCL`, then
`fstat` the descriptor. The same applies to `stat`-then-`open` and to
any check-then-act on a filesystem path.
