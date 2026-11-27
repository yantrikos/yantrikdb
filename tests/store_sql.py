"""Raw SQL against an engine store — from a CHILD process, never in-process.

The engine links its own SQLite; Python's stdlib ``sqlite3`` is a second copy.
Both rely on POSIX advisory locks, which the kernel scopes per **process**, so
an in-process writer never serialises against the engine's writers and a WAL
commit from one can land on top of the other's (sqlite.org/howtocorrupt.html,
"multiple copies of SQLite linked into the same application"). Measured on the
0.18/0.19 wheels under CPU contention: 2-8 % of runs corrupted the store; the
py3.10 CI job hit it twice in one day. On Linux the engine also notices the
second library via ``/proc/self/maps`` and may refuse writes.

The engine also has a materializer thread, so "it is idle right now" is not a
property a test can rely on.

A child process holds its own locks and the kernel serialises it against the
engine like any other client, which is why every raw touch in a test must go
through here. Background: ``tests/test_foreign_sqlite.py`` and issue #225.

Reads that have an engine API should use it instead — ``get_patterns()`` and
friends. These helpers exist for the rows no API can write (synthetic pattern
fixtures) and for checks the API does not expose.
"""
from __future__ import annotations

import json
import subprocess
import sys


def raw_sql_in_subprocess(path, sql, params=()):
    """Run one raw statement against `path` in a separate process, then commit."""
    payload = json.dumps([str(path), sql, list(params)])
    prog = (
        "import json, sqlite3, sys\n"
        "path, sql, params = json.loads(sys.argv[1])\n"
        "c = sqlite3.connect(path)\n"
        "c.execute(sql, params)\n"
        "c.commit()\n"
        "c.close()\n"
    )
    subprocess.run(
        [sys.executable, "-c", prog, payload],
        check=True,
        capture_output=True,
    )


def query_in_subprocess(path, sql, params=()):
    """Run one SELECT in a separate process and return the rows as lists."""
    payload = json.dumps([str(path), sql, list(params)])
    prog = (
        "import json, sqlite3, sys\n"
        "path, sql, params = json.loads(sys.argv[1])\n"
        "c = sqlite3.connect(path)\n"
        "rows = [list(r) for r in c.execute(sql, params).fetchall()]\n"
        "c.close()\n"
        "sys.stdout.write(json.dumps(rows))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", prog, payload],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout or "[]")
