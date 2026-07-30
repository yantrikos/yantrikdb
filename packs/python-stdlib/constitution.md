# python-stdlib constitution

Applied to every piece of Python written while this pack is mounted.
Each rule targets the measured failure mode of local models writing
Python: invented imports, wrong signatures, and deprecated idioms.
Terse on purpose — reasoning and signatures live in the corpus.

## Verify the API before writing the call

Before calling a standard-library function, recall its signature from
this pack. If the pack has no record of the function for this Python
version, say so and use a documented alternative — never write a call
from memory alone.

## No invented imports

Import only modules that exist: the standard library as recorded here,
or third-party packages the project's dependency file already declares.
Never import `retrying`, `dateutils`, or any package whose existence you
have not confirmed.

## Keyword names come from the signature

`subprocess.run` captures with `capture_output=True`, not `output=True`.
`json.dumps` indents with `indent=2`, not `pretty=True`. When a keyword
argument is not in the recalled signature, it does not exist.

## Prefer the modern spelling

`pathlib.Path` over `os.path` string juggling; `subprocess.run` over
`os.system` and `subprocess.call`; `datetime.now(timezone.utc)` over the
deprecated `utcnow()`; f-strings over `%` and `.format()`; explicit
`encoding="utf-8"` on every text file open.

## Commands are lists, queries are parameters

`subprocess.run` takes a list, never a shell string built from
variables. `sqlite3` takes `?` placeholders, never an f-string. Both
rules exist because the interpolated version runs fine until the first
hostile input.

## Errors are handled at the narrowest useful scope

Catch the specific exception the call raises (`FileNotFoundError`,
`CalledProcessError`, `json.JSONDecodeError`) — a bare `except:` or
`except Exception` around a block hides the defect this pack exists to
prevent.

## Version-gate anything newer than 3.8

`itertools.batched` (3.12), `hashlib.file_digest` (3.11), `dataclass
slots=True` (3.10), `dirs_exist_ok` (3.8). If the project targets an
older Python, use the compatible spelling instead of the newest one.

## State the source of an uncertain claim

If asked about an API this pack does not cover, prefix the answer with
the fact that it is from model memory, unverified for the local Python
version.
