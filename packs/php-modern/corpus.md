# php-modern corpus

PHP 8. The two failure modes this targets are writing PHP 5 in a PHP 8
file, and reproducing a security default that has been wrong since 2005
and is still wrong.

Version numbers name the release that introduced a feature, because
"PHP 8" spans five years and using an 8.4 construct on 8.1 is a parse
error, not a graceful degradation.

## declare(strict_types=1) on every file

Without it, PHP coerces: passing `"5 apples"` to an `int` parameter
succeeds with a warning. With it, a type mismatch is a `TypeError` at
the call. The declaration is per-file and must be the very first
statement, and it governs the *calling* file, not the file where the
function is declared — so it has to be everywhere to be worth anything.

## Constructor property promotion

```php
final class Money {
    public function __construct(
        public readonly int $amount,
        public readonly Currency $currency,
    ) {}
}
```
PHP 8.0. The property declaration, the constructor parameter and the
assignment collapse into one. `readonly` (8.1) makes the property
writable exactly once, from inside the declaring class's scope —
attempting a second write throws `Error`, which is how immutability is
enforced rather than merely documented.

## Enums are real types, not class constants

```php
enum Status: string {
    case Draft = 'draft';
    case Published = 'published';
    public function label(): string {
        return match ($this) {
            Status::Draft => 'Draft',
            Status::Published => 'Live',
        };
    }
}
```
PHP 8.1. Backed enums give `::from()` (throws `ValueError` on an unknown
value) and `::tryFrom()` (returns null). `::cases()` enumerates. An enum
can implement interfaces and hold methods and constants, but not state —
there are no properties, because each case is a singleton.

## match is not switch

`match` compares with `===`, returns a value, requires no `break`, does
not fall through, and throws `UnhandledMatchError` when nothing matches
and there is no `default`. That last property is the point: a `switch`
silently does nothing on an unhandled value, and `match` refuses to.
PHP 8.0.

## Readonly classes and asymmetric visibility

PHP 8.2 allows `readonly class Point {}`, which marks every property
readonly. PHP 8.4 adds asymmetric visibility — `public private(set)
string $name;` — a property readable everywhere and writable only inside
the class, which covers the common case that `readonly` is too strict
for.

## Property hooks replace most getters and setters

PHP 8.4:
```php
class User {
    public string $fullName {
        get => $this->first . ' ' . $this->last;
        set (string $v) { [$this->first, $this->last] = explode(' ', $v, 2); }
    }
}
```
The property stays a property to every caller, so a plain public field
can gain validation later without changing its call sites. Only available
on 8.4 and later — on 8.3 this is a parse error.

## Nullsafe operator short-circuits the whole chain

`$country = $session?->user?->address?->country;` — if any link is null
the entire expression is null and the remaining calls are never
evaluated. PHP 8.0. It applies to property and method access, not to
array offsets, and it does not make a *missing* property safe, only a
null one.

## Named arguments and what they lock in

`htmlspecialchars($s, double_encode: false)` skips intermediate optional
parameters. The consequence is that parameter *names* become part of a
public API — renaming one is a breaking change. Named arguments can
follow positional ones but never precede them.

## The string-to-number comparison rules changed in PHP 8

`0 == "foo"` was **true** before PHP 8 and is **false** from PHP 8.0 on:
a number compared with a non-numeric string now compares as strings.
`"abc" == 0`, `null == "0"` and the login-bypass patterns that grew out
of them behave differently across the boundary. Use `===` and stop
depending on any of it.

## == still surprises where both sides look numeric

`"1e3" == "1000"` is true — both are numeric strings, so they compare
numerically. `"0x1A" == 26` is false, because hex strings stopped being
numeric in PHP 7. Comparing hashes with `==` is the classic
type-juggling vulnerability: two hashes beginning `0e` followed by
digits both cast to 0 and compare equal. Use `===`, or `hash_equals`.

## ?? and ?: are different

`??` tests for null-or-undefined and does not emit a notice for a
missing key: `$name = $_GET['name'] ?? 'anon';`. `?:` tests for falsy,
so `0`, `''` and `'0'` all fall through to the default — usually not
what was meant. `??=` assigns only when the left side is null.

## Arrays: filter preserves keys, map does not always

`array_filter` keeps the original keys, so a filtered list is no longer
a list and `json_encode` renders it as an object instead of an array.
Wrap it in `array_values`. `array_map` preserves keys for a single
array but reindexes when given several. `array_is_list()` (8.1) is the
check. `array_merge` renumbers integer keys while the `+` operator
keeps the left-hand side's.

## foreach by reference leaves the reference behind

```php
foreach ($rows as &$row) { $row['x'] = 1; }
unset($row);           // ← without this, the next foreach corrupts the last element
```
After the loop, `$row` still references the final element, so a
subsequent `foreach ($rows as $row)` overwrites it on each iteration.
This is the most-reported PHP bug that is not a bug. Always `unset` the
reference variable.

## String functions are byte-oriented

`strlen`, `substr`, `strtoupper` and `str_pad` operate on bytes, so they
corrupt UTF-8 mid-character. Use the `mb_` family with an explicit
encoding for anything user-supplied, and `mb_str_split` rather than
`str_split`. PHP 8 added `str_contains`, `str_starts_with` and
`str_ends_with`, which finally retire the `strpos(...) !== false` idiom
and its `0`-is-falsy trap.

## Every query is a prepared statement, and emulation is off

```php
$pdo = new PDO($dsn, $user, $pass, [
    PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
    PDO::ATTR_EMULATE_PREPARES   => false,
    PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
]);
$stmt = $pdo->prepare('SELECT * FROM users WHERE email = ? AND active = ?');
$stmt->execute([$email, 1]);
```
With emulation **on** — which is the default for MySQL — PDO
interpolates the values itself and sends one string, so a charset
mismatch can still produce injection, and `LIMIT ?` binds as a quoted
string and breaks. Turning it off makes the database do the binding.

## LIKE patterns need their own escaping

Placeholders protect against injection but not against wildcards: a user
searching for `100%` matches everything. Escape `%`, `_` and the escape
character itself in the value before binding, and declare it:
`LIKE ? ESCAPE '\\'`.

## Identifiers cannot be bound

Table names, column names and `ORDER BY` directions are not values and
cannot be parameterised. Validate them against an explicit allowlist —
`in_array($col, ['name','created_at'], true)` — and never interpolate
even a "checked" string that came from input.

## Escape on output, with the right function for the context

`htmlspecialchars($s, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8')` for HTML
text and attributes — PHP 8.1 made `ENT_QUOTES | ENT_SUBSTITUTE` the
default flags, but passing them explicitly is still correct and works on
older versions. HTML escaping is *not* sufficient inside a `<script>`
block, inside a URL, or in a CSS context: use `json_encode` with
`JSON_HEX_TAG | JSON_HEX_AMP | JSON_HEX_APOS | JSON_HEX_QUOT` for JS,
and `rawurlencode` for URL components.

## unserialize on user input is object injection

Deserialising attacker-controlled data invokes `__wakeup` and
`__destruct` on types the attacker chooses, which is remote code
execution through gadget chains. `unserialize($data, ['allowed_classes'
=> false])` limits it, but the correct answer is `json_decode`. The same
applies to `extract()` on request data, which lets a caller overwrite
any local variable.

## json_decode returns null on failure unless you ask it not to

`json_decode($s)` yields `null` both for the input `"null"` and for
malformed JSON. Pass `JSON_THROW_ON_ERROR` (PHP 7.3+) so it raises
`JsonException` instead, and `json_encode` likewise — a silent `false`
from `json_encode` on invalid UTF-8 becomes an empty response body.
`json_validate()` (8.3) checks without building the structure.

## Passwords: password_hash, and rehash on login

`password_hash($pw, PASSWORD_DEFAULT)` produces a salted bcrypt or
Argon2 hash containing its own parameters; `password_verify($pw, $hash)`
checks it. Never `md5`, never `sha1`, never a hand-rolled salt. On each
successful login call `password_needs_rehash($hash, PASSWORD_DEFAULT)`
and re-hash if true, so the cost factor rises with the platform default.

## Randomness: random_int and random_bytes, never rand or mt_rand

`rand`, `mt_rand`, `uniqid` and `shuffle` are predictable and must not
produce tokens, password resets, session identifiers or salts.
`random_int` and `random_bytes` are cryptographically secure and throw
on failure rather than returning weak output. PHP 8.2 added the
`Random\Randomizer` object API over the same engines.

## Compare secrets with hash_equals

`==` and `===` on strings return at the first differing byte, leaking the
matching prefix through timing, and `==` additionally type-juggles.
`hash_equals($known, $given)` is constant time. It is the correct
comparison for HMAC signatures, CSRF tokens and API keys.

## Sessions: regenerate the id on privilege change

`session_regenerate_id(true)` immediately after login and after any
privilege change, otherwise a session fixed by the attacker before login
remains valid after it. Cookie parameters belong in configuration:
`httponly` on, `secure` on, `samesite` `Lax` or `Strict`, and
`session.use_strict_mode = 1` so PHP refuses to adopt a session id it
did not issue.

## Never include or require a path built from input

`include $_GET['page'] . '.php'` is local file inclusion, and with
`allow_url_include` it is remote code execution. Map the input through a
fixed array of permitted routes. The same rule covers `file_get_contents`,
`fopen` and `unlink` — validate with `realpath()` and confirm the result
is inside the intended directory prefix.

## Uploaded files: trust nothing the browser said

`$_FILES['f']['type']` and `['name']` are attacker-controlled. Determine
the type server-side with `finfo_file`, generate your own filename, and
store outside the web root or in a directory with execution disabled.
Check `$_FILES['f']['error'] === UPLOAD_ERR_OK`, verify with
`is_uploaded_file`, and move with `move_uploaded_file` rather than
`rename`.

## filter_var validates; it does not sanitise into safety

`filter_var($e, FILTER_VALIDATE_EMAIL)` and `FILTER_VALIDATE_INT` return
the value or `false` — note that `false` is also what an integer `0`
looks like under a loose check, so compare with `!== false`. The
`FILTER_SANITIZE_*` filters are deprecated as of 8.1 for string
sanitising: escape at output instead of mangling at input.

## DateTimeImmutable, not DateTime

`DateTime::modify()` mutates the object in place, so a date passed into
a function can come back changed. `DateTimeImmutable` returns a new
instance from every operation. Always construct with an explicit
`DateTimeZone`; the default comes from `date.timezone` in php.ini and
differs between the developer's machine and the server. `DateTime` and
`DateTimeImmutable` compare correctly with `<`, `>` and `==`.

## Money is not a float

`0.1 + 0.2 === 0.3` is false in PHP as everywhere else. Store money as
an integer number of minor units, or use `bcmath` (`bcadd`, `bcmul`
with an explicit scale). Never `round()` a float and hope. `intdiv()`
avoids the float round-trip of `/` for integer division.

## The @ suppression operator hides fatal conditions

`@$foo['bar']` suppresses the diagnostic without preventing the
condition, and before PHP 8 it could suppress errors that abort the
request, making the failure invisible. Use `??` for missing keys,
`isset`/`array_key_exists` for presence, and let real errors surface to
the error handler.

## Errors and exceptions are the same hierarchy now

PHP 7 introduced `Throwable`, with `Error` (type errors, division by
zero, calling a method on null) alongside `Exception`. `catch
(Exception $e)` does **not** catch a `TypeError`. Catch `Throwable` at
the top-level boundary only, and specific types everywhere else.
Converting warnings to exceptions with `set_error_handler` makes the
whole surface uniform.

## Never echo an exception to the browser

`display_errors` must be off in production, with `log_errors` on. A
stack trace exposes paths, database names and, when the trace includes
arguments, credentials. PHP 8.2's `#[\SensitiveParameter]` attribute
redacts a parameter from stack traces — put it on password and token
arguments.

## First-class callables and new in initialisers

PHP 8.1: `$fn = strlen(...)` produces a `Closure` from any callable,
replacing `'strlen'` strings and `[$obj, 'method']` arrays that no
static analyser could follow. Also 8.1: `new` expressions are allowed in
parameter defaults, attribute arguments and static variable
initialisers, so a default dependency no longer needs a null sentinel
plus a body check.

## never and void are different return types

`void` means the function returns no value; `never` (8.1) means it does
not return at all — it always throws or exits. `never` lets static
analysis know the code after a call is unreachable, which is what makes
a `fail()` helper usable as the last arm of a `match`.

## #[\Override] catches the rename

PHP 8.3's `#[\Override]` attribute makes it a compile-time error if the
method does not in fact override a parent or interface method. Without
it, renaming a parent method silently turns the child's override into a
new, never-called method — the same class of bug `@Override` was
introduced to catch in Java.

## Typed constants and the readonly clone problem

PHP 8.3 allows a type on class constants (`public const string NAME =
'x'`). It also relaxed readonly re-initialisation inside `__clone`, so a
`with…()` style copy can modify a readonly property during cloning — a
`readonly` object was otherwise impossible to derive from.

## Composer autoloading is PSR-4, and require is for entry points only

Class files are found by autoloader, not by `require`. One class per
file, namespace matching the directory path under the PSR-4 prefix, file
name matching the class name including case — which works on a
developer's case-insensitive filesystem and fails on the Linux server.
`composer.lock` is committed for applications and not for libraries.

## Static analysis is where PHP's type system actually lives

PHPStan or Psalm at a high level catches what the runtime does not:
nullable returns dereferenced without a check, array shapes, unreachable
branches, and unhandled exception paths. Docblock generics
(`@return list<User>`) carry information the language has no syntax for.
Add `declare(strict_types=1)` first; the analysers get far more precise
once coercion is off.
