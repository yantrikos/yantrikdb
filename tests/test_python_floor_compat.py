"""The repository's Python floor is 3.10 (``requires-python = ">=3.10"``).

Names that only exist on 3.11+ have shipped twice: 0.16.0 and 0.17.0 carried
``from datetime import UTC`` in a module the package imports eagerly, so
``import yantrikdb`` raised ImportError on 3.10. CI now runs a 3.10 leg for the
package, but benchmark and pack tooling outside ``src/`` is not imported by that
leg — this scan covers every git-tracked Python file instead.

A line that deliberately uses a 3.11+ name behind a guard must carry the
``# py310-ok`` marker on (or within three lines above) the use.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATTERNS: dict[str, re.Pattern[str]] = {
    "datetime.UTC (3.11+; use timezone.utc)": re.compile(
        r"from datetime import[^#\n]*\bUTC\b|datetime\.UTC\b"
    ),
    "enum.StrEnum (3.11+)": re.compile(
        r"from enum import[^#\n]*\bStrEnum\b|enum\.StrEnum\b"
    ),
    "tomllib (3.11+ stdlib; guard with a tomli fallback)": re.compile(
        r"^\s*(import tomllib\b|from tomllib\b)"
    ),
    "typing.Self (3.11+)": re.compile(
        r"from typing import[^#\n]*\bSelf\b|typing\.Self\b"
    ),
    "ExceptionGroup / except* (3.11+)": re.compile(
        r"\bExceptionGroup\b|except\s*\*"
    ),
}


def _tracked_python_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return [ROOT / line.strip() for line in out.stdout.splitlines() if line.strip()]


def _guarded(lines: list[str], idx: int) -> bool:
    """A use whose own line or one of the three lines above carries
    ``# py310-ok`` is a deliberate, guarded use."""
    return any("# py310-ok" in lines[j] for j in range(max(0, idx - 3), idx + 1))


def test_no_python_311_only_names_anywhere_in_the_repository() -> None:
    offenders: list[str] = []
    for path in _tracked_python_files():
        if path.name == Path(__file__).name:
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            code = line.split("#", 1)[0]
            for label, rx in PATTERNS.items():
                if rx.search(code) and not _guarded(lines, i):
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{i + 1}: {label}: {line.strip()}"
                    )
    assert not offenders, (
        "Python 3.11+-only names in a repository whose floor is 3.10:\n  "
        + "\n  ".join(offenders)
    )
