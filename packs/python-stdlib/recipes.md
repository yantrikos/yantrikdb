# Recipes — authored, not generated

Each recipe names a task in the words someone doing it would use, then
shows the shortest stdlib-only way that is correct on Python 3.13.

## Reading and writing a text file with pathlib

The whole-file read and write are one call each on `Path` — no `open()`
needed, and encoding should always be explicit.

```python
from pathlib import Path
text = Path("notes.txt").read_text(encoding="utf-8")
Path("out.txt").write_text(text, encoding="utf-8")
```

`read_bytes()` / `write_bytes()` are the binary pair. For appending or
line-by-line streaming, fall back to `Path.open("a", encoding="utf-8")`.

## Running a shell command and capturing its output

`subprocess.run` with `capture_output=True, text=True` returns stdout and
stderr as strings on a `CompletedProcess`. Pass the command as a list —
never build a shell string from variables.

```python
import subprocess
r = subprocess.run(["git", "status", "--short"],
                   capture_output=True, text=True, check=True)
print(r.stdout)
```

`check=True` raises `CalledProcessError` on nonzero exit; without it you
must test `r.returncode` yourself. There is no `subprocess.run(...).output`
attribute — it is `stdout`.

## Walking a directory tree and matching file names

`Path.rglob` is the recursive glob; it yields `Path` objects, not strings.

```python
from pathlib import Path
for p in Path("src").rglob("*.py"):
    if p.is_file():
        print(p, p.stat().st_size)
```

For a flat, non-recursive match use `Path.glob("*.py")`. `os.walk` still
exists but returns string triples; prefer `rglob` in new code.

## Reading and writing JSON with a file

`json.load` / `json.dump` take file objects; `json.loads` / `json.dumps`
take strings — the trailing `s` means *string*.

```python
import json
from pathlib import Path
data = json.loads(Path("cfg.json").read_text(encoding="utf-8"))
Path("cfg.json").write_text(
    json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
```

`json.dumps` has no `pretty=True` parameter — indentation is `indent=2`.
Keys are always strings after a round-trip; integer dict keys do not
survive.

## Parsing and formatting dates and times

`datetime.strptime` parses, `strftime` formats, and the format codes are
the same in both directions. Timezone-aware "now" is
`datetime.now(timezone.utc)` — `utcnow()` is deprecated and naive.

```python
from datetime import datetime, timezone
d = datetime.strptime("2026-07-30 14:00", "%Y-%m-%d %H:%M")
stamp = datetime.now(timezone.utc).isoformat()
back = datetime.fromisoformat(stamp)
```

`fromisoformat` accepts the `Z` suffix since 3.11. There is no
`datetime.parse` and no `strptime` on `date` objects' instances.

## Making an HTTP GET request without third-party packages

`urllib.request.urlopen` is the stdlib route; the response is bytes and
must be decoded. There is no `requests` in the standard library.

```python
from urllib.request import urlopen, Request
req = Request("https://api.example.com/items",
              headers={"Accept": "application/json"})
with urlopen(req, timeout=10) as resp:
    body = resp.read().decode("utf-8")
```

Always pass `timeout` — the default is no timeout at all. For anything
beyond a simple GET/POST, recommend installing `httpx` or `requests`
rather than fighting `urllib`.

## Building a URL query string safely

`urllib.parse.urlencode` handles quoting and multi-value keys;
never concatenate user input into a URL by hand.

```python
from urllib.parse import urlencode, urlparse, parse_qs
qs = urlencode({"q": "black & white", "page": 2})   # q=black+%26+white&page=2
parts = urlparse("https://x.dev/search?" + qs)
params = parse_qs(parts.query)                       # values are LISTS
```

`parse_qs` returns `{"q": ["black & white"]}` — every value is a list.

## Extracting groups from text with a regular expression

`re.search` finds the first match anywhere; `re.match` only matches at the
start of the string — using `match` where `search` is meant is the classic
silent bug. A no-match returns `None`, so guard before `.group()`.

```python
import re
m = re.search(r"v(\d+)\.(\d+)\.(\d+)", "release v0.11.2 shipped")
if m:
    major, minor, patch = m.groups()
```

`re.findall` returns strings (or tuples when the pattern has groups);
`re.finditer` returns match objects. Compile with `re.compile` only when
the pattern is reused in a loop.

## Copying, moving and deleting files and directories

`shutil.copy2` preserves metadata, plain `copy` does not; `shutil.move`
works across filesystems; directory trees need `copytree`/`rmtree` —
`Path.unlink` and `os.remove` only delete single files.

```python
import shutil
from pathlib import Path
shutil.copy2("a.db", "a.db.bak")
shutil.copytree("site", "site_backup", dirs_exist_ok=True)
shutil.rmtree("build")            # directory tree, no confirmation
Path("stale.log").unlink(missing_ok=True)
```

`copytree` raises if the target exists unless `dirs_exist_ok=True` (3.8+).

## Creating a temporary file or directory that cleans itself up

The context-manager forms of `tempfile` remove what they created.
On Windows a `NamedTemporaryFile` cannot be reopened while open — use a
`TemporaryDirectory` and create files inside it instead.

```python
import tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as td:
    scratch = Path(td) / "work.json"
    scratch.write_text("{}", encoding="utf-8")
# gone here
```

## Counting, grouping and deduplicating with collections

`Counter` counts hashables, `defaultdict(list)` groups, and `dict` itself
deduplicates while preserving order.

```python
from collections import Counter, defaultdict
words = ["red", "blue", "red", "green", "red"]
Counter(words).most_common(2)        # [("red", 3), ("blue", 1)]
by_len = defaultdict(list)
for w in words:
    by_len[len(w)].append(w)
unique = list(dict.fromkeys(words))  # order-preserving dedupe
```

`most_common()` with no argument returns everything, sorted.

## Batching, pairing and flattening iterables with itertools

`itertools.batched` (3.12+) chunks an iterable; `pairwise` gives sliding
pairs; `chain.from_iterable` flattens one level.

```python
from itertools import batched, pairwise, chain
list(batched("ABCDEFG", 3))          # [('A','B','C'), ('D','E','F'), ('G',)]
list(pairwise([1, 4, 9]))            # [(1, 4), (4, 9)]
list(chain.from_iterable([[1, 2], [3]]))   # [1, 2, 3]
```

Before 3.12 there is no `batched` — write the two-line grouper instead of
inventing an import for it.

## Caching a pure function's results

`functools.lru_cache` memoises by argument; `functools.cache` (3.9+) is
the unbounded spelling. Arguments must be hashable — passing a list or
dict raises `TypeError` at call time.

```python
from functools import lru_cache
@lru_cache(maxsize=256)
def price(sku: str) -> float:
    ...
price.cache_clear()      # reset between tests
```

## Defining a small data-carrying class

`dataclasses.dataclass` writes `__init__`, `__repr__` and `__eq__`.
Mutable defaults must use `field(default_factory=...)` — a bare `[]`
default is rejected at class-creation time.

```python
from dataclasses import dataclass, field
@dataclass
class Job:
    name: str
    retries: int = 3
    tags: list[str] = field(default_factory=list)
```

`frozen=True` makes instances hashable and immutable; `slots=True` (3.10+)
saves memory on many instances.

## Hashing a file or string with hashlib

`hashlib.sha256` wants bytes, not str; `file_digest` (3.11+) streams a
file without loading it.

```python
import hashlib
digest = hashlib.sha256("hello".encode("utf-8")).hexdigest()
with open("wheel.whl", "rb") as f:
    fd = hashlib.file_digest(f, "sha256").hexdigest()
```

Use `secrets.token_hex(16)` for tokens — never a hash of `random()`.

## Reading and writing CSV with headers

`csv.DictReader` maps rows to dicts using the header row; always open CSV
files with `newline=""` or Windows doubles every line ending.

```python
import csv
with open("rows.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
with open("out.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["name", "qty"])
    w.writeheader()
    w.writerows(rows)
```

## Setting up logging instead of print

One `basicConfig` call at the entry point; modules take
`logging.getLogger(__name__)` and never configure anything themselves.

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)
log.info("mounted %s in %.1fms", pack, ms)
```

Pass lazy `%s` arguments, not f-strings, so skipped levels cost nothing.
`log.exception("...")` inside an `except` block records the traceback.

## Running functions in parallel threads with a pool

`ThreadPoolExecutor.map` keeps input order; `as_completed` yields in
finish order. Threads suit I/O-bound work; CPU-bound work needs
`ProcessPoolExecutor`.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(fetch, u): u for u in urls}
    for fut in as_completed(futs):
        url, body = futs[fut], fut.result()   # .result() re-raises errors
```

A future swallows its exception until `.result()` is called — iterate the
results or the errors vanish silently.

## Using sqlite3 with parameters and rows as dicts

Placeholders are `?`, never string formatting; `row_factory = sqlite3.Row`
gives name access; `with conn` commits or rolls back a transaction.

```python
import sqlite3
conn = sqlite3.connect("app.db")
conn.row_factory = sqlite3.Row
with conn:
    conn.execute("INSERT INTO jobs(name, qty) VALUES(?, ?)", ("build", 3))
row = conn.execute("SELECT * FROM jobs WHERE name=?", ("build",)).fetchone()
print(row["qty"])
```

`executemany` takes a sequence of parameter tuples for bulk inserts.

## Command-line arguments with argparse

Flags with `--` are optional; bare names are positional; `action=
"store_true"` makes a boolean switch. `parse_args()` exits with a usage
message on bad input — that is intended behaviour, not a crash.

```python
import argparse
ap = argparse.ArgumentParser(description="Sync packs")
ap.add_argument("target")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--limit", type=int, default=10)
args = ap.parse_args()
```

Attribute names swap dashes for underscores: `--dry-run` → `args.dry_run`.

## Replacing a function with a mock in a test

Patch where the name is *looked up*, not where it is defined: if
`app.sync` does `from client import fetch`, the patch target is
`app.sync.fetch`.

```python
from unittest.mock import patch
with patch("app.sync.fetch", return_value={"ok": True}) as m:
    run_sync()
    m.assert_called_once_with("packs")
```

`side_effect=Exception("boom")` makes the mock raise;
`side_effect=[a, b]` returns successive values across calls.

## Retrying an operation with backoff, stdlib only

There is no `retrying` or `tenacity` in the standard library. The honest
stdlib version is a loop; sleep grows exponentially and re-raises on the
final attempt.

```python
import time
for attempt in range(4):
    try:
        result = flaky()
        break
    except TimeoutError:
        if attempt == 3:
            raise
        time.sleep(2 ** attempt)      # 1s, 2s, 4s
```

## Wrapping and dedenting multi-line text

`textwrap.dedent` strips the common leading whitespace from triple-quoted
strings; `shorten` truncates on word boundaries with a placeholder.

```python
import textwrap
sql = textwrap.dedent("""\
    SELECT name, qty
    FROM jobs
    WHERE qty > ?""")
title = textwrap.shorten(long_title, width=60, placeholder="…")
```

## Base64 for binary data in JSON or URLs

`b64encode` takes and returns bytes — decode to str for JSON. URL-safe
variants swap `+/` for `-_`.

```python
import base64
payload = base64.b64encode(b"\x00\x01binary").decode("ascii")
raw = base64.b64decode(payload)
token = base64.urlsafe_b64encode(raw).rstrip(b"=")
```

Padding matters on decode: add `==` back if you stripped it, or pass
`validate=False` and accept the risk.
