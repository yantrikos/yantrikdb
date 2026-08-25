"""The repository's Python floor is 3.10 (``requires-python = ">=3.10"``).

Names that only exist on 3.11+ have shipped twice: 0.16.0 and 0.17.0 carried
``from datetime import UTC`` in a module the package imports eagerly, so
``import yantrikdb`` raised ImportError on 3.10. CI now runs a 3.10 leg for the
package, but benchmark and pack tooling outside ``src/`` is not imported by that
leg — this scan covers every git-tracked Python file instead.

A line that deliberately uses a 3.11+ name behind a guard must carry the
``# py310-ok`` marker on (or within three lines above) the use.

The scanner itself is tested: every pattern must detect its canonical
offender, a guarded use must be accepted, and clean code must produce nothing —
otherwise a broken regex would make the repository-wide check vacuously green.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARD_MARKER = "# py310-ok"

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

# One canonical offender per pattern. Each MUST be detected.
FORBIDDEN_SAMPLES: dict[str, str] = {
    "datetime.UTC (3.11+; use timezone.utc)": "from datetime import UTC, datetime",
    "enum.StrEnum (3.11+)": "from enum import StrEnum",
    "tomllib (3.11+ stdlib; guard with a tomli fallback)": "import tomllib",
    "typing.Self (3.11+)": "from typing import Self",
    "ExceptionGroup / except* (3.11+)": "except* ValueError as eg:",
}


def _find_offenders(lines: list[str]) -> list[tuple[int, str, str]]:
    """Return ``(1-based line number, pattern label, stripped line)`` for every
    unguarded use of a 3.11+-only name in ``lines``."""
    found: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        code = line.split("#", 1)[0]
        if not code.strip():
            continue
        guarded = any(
            GUARD_MARKER in lines[j] for j in range(max(0, i - 3), i + 1)
        )
        if guarded:
            continue
        for label, rx in PATTERNS.items():
            if rx.search(code):
                found.append((i + 1, label, line.strip()))
    return found


def _tracked_python_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return [ROOT / line.strip() for line in out.stdout.splitlines() if line.strip()]


@pytest.mark.parametrize("label", sorted(PATTERNS))
def test_scanner_detects_each_forbidden_sample(label: str) -> None:
    sample = FORBIDDEN_SAMPLES[label]
    found = _find_offenders(["import os", sample, "x = 1"])
    assert [(n, lab) for n, lab, _ in found] == [(2, label)], found


def test_scanner_accepts_a_guarded_use() -> None:
    lines = [
        "try:  # py310-ok: tomllib is 3.11+; tomli is the backport",
        "    import tomllib",
        "except ModuleNotFoundError:",
        "    import tomli as tomllib",
    ]
    assert _find_offenders(lines) == []


def test_scanner_is_silent_on_clean_code() -> None:
    lines = [
        "from datetime import datetime, timezone",
        "UTC = timezone.utc",
        "from enum import Enum",
        "try:",
        "    x = 1",
        "except ValueError as e:",
        "    raise",
    ]
    assert _find_offenders(lines) == []


def test_no_python_311_only_names_anywhere_in_the_repository() -> None:
    offenders: list[str] = []
    for path in _tracked_python_files():
        if path.name == Path(__file__).name:
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for lineno, label, text in _find_offenders(lines):
            offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {label}: {text}")
    assert not offenders, (
        "Python 3.11+-only names in a repository whose floor is 3.10:\n  "
        + "\n  ".join(offenders)
    )
