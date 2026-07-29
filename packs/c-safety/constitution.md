# c-safety constitution

Rules applied to every line of C written while this pack is mounted.
Each one exists because the default output violates it in code that
compiles without a warning.

## Every allocation is checked before it is used

`malloc`, `calloc`, `realloc` and `strdup` return NULL on failure. The
check is on the line after the call, before any dereference. `realloc`
goes into a temporary — `p = realloc(p, n)` leaks the block it was
holding when the call fails.

## Element counts are checked for overflow before multiplication

`malloc(n * sizeof t)` wraps for large `n`. Use `calloc(n, sizeof t)`,
which must detect it, or check `n > SIZE_MAX / sizeof t` first. Any
size arithmetic on an externally supplied count gets the same treatment.

## Freed pointers are set to NULL immediately

`free(p); p = NULL;` in one motion. This makes double-free impossible on
that path and turns a use-after-free into a NULL dereference, which
crashes loudly instead of corrupting quietly.

## No unbounded string function, ever

`gets`, `strcpy`, `strcat`, `sprintf`, `vsprintf` and bare `scanf("%s")`
do not appear. Bounded equivalents are used with the destination size
taken from `sizeof` at the point of declaration, and `scanf` conversions
carry an explicit field width.

## snprintf's return value is checked for truncation

`snprintf` returns what it *would* have written. Any call whose output
feeds a path, a command, a query or a security decision compares the
return against the buffer size and treats `>=` as an error. Silent
truncation is a correctness bug, not a cosmetic one.

## strncpy is either avoided or terminated by hand

If `strncpy` is used, `dst[n - 1] = '\0'` follows it unconditionally.
The function writes no terminator when the source fills the buffer.

## Unsigned subtraction is guarded before it is performed

Any `a - b` on unsigned types (`size_t` above all) is preceded by
`if (a >= b)`. `len - 1` where `len` may be zero is a wrap to `SIZE_MAX`,
not a negative number, and no `>= 0` test will catch it.

## Signed overflow is prevented, not detected afterwards

Checks happen before the arithmetic — `if (a > INT_MAX - b)` — or via
`__builtin_add_overflow` / `__builtin_mul_overflow`. Post-hoc tests like
`if (a + b < a)` are undefined behaviour on signed types and the
optimiser deletes them.

## Shifts stay inside the type's width and out of the sign bit

Shift counts are checked against the operand width, and any shift that
could reach the top bit uses an unsigned type. Widening casts go on the
operand before the shift, never on the result.

## Type punning goes through memcpy, never a pointer cast

Reinterpreting bytes as another type uses `memcpy` into a properly typed
object. `*(T *)&x` breaks strict aliasing and the compiler is entitled
to reorder around it.

## Character classification casts to unsigned char

`isalpha`, `isdigit`, `toupper` and the rest receive
`(unsigned char)c`. `getchar` and `fgetc` results are stored in an `int`
so `EOF` is distinguishable from a valid byte.

## Numeric parsing uses strtol/strtod with full error checking

`atoi` and `atof` do not appear — they cannot report failure. `strtol`
is used with `errno = 0` before, and the end pointer, trailing
characters and `ERANGE` all checked after.

## Multi-resource functions have one exit path

Resources are declared and initialised to safe values up front, failures
`goto` a single cleanup label, and cleanup runs unconditionally in
reverse order. No early `return` leaves an allocation or a file handle
behind.

## Buffer lengths travel with buffers

Any function taking a pointer to memory it will write takes the size as
a parameter. `sizeof` on an array parameter yields the pointer size, so
it is never used for a bound inside the callee.

## No variable-length arrays or alloca on unvalidated sizes

Stack allocation uses compile-time constant bounds. Sizes derived from
input go to the heap, where failure is a NULL that can be checked.

## Format strings are literals

The first argument of a `printf`-family call is always a string literal.
User data is passed as an argument to `%s`.

## Secrets are compared in constant time and erased explicitly

Token and MAC comparison accumulates differences over the full length
rather than returning early. Erasure uses `explicit_bzero` or an
equivalent the compiler may not elide — a plain `memset` before a free
is dead code and gets removed.

## Overlapping copies use memmove

`memcpy` is only used where the regions are provably disjoint. Anything
shifting within one buffer uses `memmove`.

## Builds run with warnings as errors and sanitizers in test

`-Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wvla
-Wformat=2 -D_FORTIFY_SOURCE=3 -fstack-protector-strong` for the build;
`-fsanitize=address,undefined` for the test run. Code is written to pass
these, not adjusted afterwards to silence them.

## assert never carries a side effect or validates input

Assertions vanish under `NDEBUG`. Runtime validation of anything
external is an `if` and an error return.
