# php-modern constitution

Applied to every line of PHP written while this pack is mounted. Each
rule replaces a default that is either a decade out of date or a known
vulnerability class.

## Every file declares strict types

`declare(strict_types=1);` is the first statement in every PHP file.
Parameters, return values and properties carry declared types. Coercion
is not a feature being relied on.

## Data objects use promotion, readonly and enums

Constructors use promoted properties. Values that must not change after
construction are `readonly`. A fixed set of alternatives is an `enum`,
never a set of class constants or bare strings — and dispatch over it
uses `match`, which throws on an unhandled case rather than falling
through silently.

## Comparison is strict

`===` and `!==` everywhere. `in_array($x, $a, true)` with the strict
flag. `==` does not appear, and no logic depends on PHP's
string-to-number juggling in either direction. Secrets are compared with
`hash_equals`.

## Null is handled with ?? and ?->, deliberately

`??` for a possibly-absent value, `??=` for a default assignment, `?->`
for a chain that may break on null. `?:` is not used where null rather
than falsy is the real question — `0` and `''` are values.

## Every database value is a bound parameter

PDO with `prepare` and placeholders. The connection sets
`ATTR_ERRMODE => ERRMODE_EXCEPTION` and `ATTR_EMULATE_PREPARES => false`.
Identifiers and sort directions are checked against an explicit
allowlist, never interpolated. `LIKE` values have their wildcards
escaped before binding.

## Escape at output, matched to the context

`htmlspecialchars($v, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8')` for HTML,
`json_encode` with the hex flags for data embedded in JavaScript,
`rawurlencode` for URL components. Escaping happens where the value is
printed, never where it is stored, and HTML escaping is never assumed to
cover a script or URL context.

## Untrusted input is never deserialised, included, or extracted

No `unserialize` on request data, no `extract`, no `eval`, and no
`include`/`require` of a path built from input — routes map through a
fixed array. File paths are resolved with `realpath` and confirmed to
sit under an intended prefix.

## Uploads are validated server-side

`$_FILES` `type` and `name` are treated as attacker-controlled. The type
comes from `finfo_file`, the stored filename is generated, the error
code is checked against `UPLOAD_ERR_OK`, and the file is moved with
`move_uploaded_file` into a location that cannot execute.

## Passwords and tokens use the dedicated APIs

`password_hash` / `password_verify`, with `password_needs_rehash` on
successful login. `random_int` and `random_bytes` for anything that must
be unguessable — never `rand`, `mt_rand` or `uniqid`. No hand-rolled
salting, and no general-purpose hash function used as a password hash.

## Sessions regenerate on privilege change

`session_regenerate_id(true)` after login and after any change in
privilege. Cookies are `httponly`, `secure` and `SameSite`, with
`use_strict_mode` enabled.

## Arrays are handled with their key semantics in mind

`array_filter` results are passed through `array_values` when a list is
required. `array_is_list` is used where list-ness matters. A `foreach`
taken by reference is always followed by `unset` of the reference
variable.

## Text handling is multibyte-aware

`mb_strlen`, `mb_substr`, `mb_strtolower` and `mb_str_split` with an
explicit encoding for anything user-supplied. `str_contains`,
`str_starts_with` and `str_ends_with` replace `strpos` comparisons.

## Dates are immutable and zone-explicit

`DateTimeImmutable` with an explicit `DateTimeZone`, never `DateTime`
and never relying on the ini default. Elapsed time and formatting go
through the object API rather than string arithmetic.

## Money is integers or bcmath

Monetary amounts are stored and computed as integer minor units, or with
`bcmath` at an explicit scale. Floats do not hold currency, and `round()`
is not a correctness strategy.

## Failures are exceptions, and they are not printed

`JSON_THROW_ON_ERROR` on encode and decode. Specific exception types are
caught, with `Throwable` only at the top-level boundary — `Exception`
alone does not catch a `TypeError`. `display_errors` is off in
production, `log_errors` on, and password or token parameters carry
`#[\SensitiveParameter]`. The `@` operator does not appear.

## Modern syntax where it is clearer

First-class callable syntax (`strlen(...)`) over callable strings and
arrays. `new` in initialisers over null-sentinel defaults. `never` for
functions that always throw. `#[\Override]` on every method that
overrides. Feature use is matched to the stated minimum PHP version, and
8.4-only constructs such as property hooks are not emitted for an
earlier target.

## Classes are autoloaded, one per file, PSR-4

Namespace mirrors the directory path and the filename matches the class
name exactly, including case. `require` appears only in entry points.
