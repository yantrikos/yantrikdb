"""The slim build's stubs take the same keyword arguments as the real methods.

A wheel built with --no-default-features compiles the embedder-download paths
out and keeps stubs that raise RuntimeError with an actionable message. The
stubs once named their parameters `_src`, `_name` and so on, so a caller who
passed arguments by keyword got TypeError instead. On the default build the
same calls reach the real methods and fail for their own reasons (no such
file, no such model); never with TypeError.
"""

import os

import pytest

from yantrikdb import YantrikDB


def _keyword_calls(db, tmp_path):
    missing = os.path.join(str(tmp_path), "no-such.ydbpack")
    return {
        "convert_pack": lambda: YantrikDB.convert_pack(
            src=missing, dest=missing + ".out", embedder_name="no-such-model"
        ),
        "reembed": lambda: db.reembed(new_embedder_name="no-such-model"),
        "set_embedder_named": lambda: db.set_embedder_named(name="no-such-model"),
    }


@pytest.mark.parametrize("name", ["convert_pack", "reembed", "set_embedder_named"])
def test_keyword_arguments_reach_the_method(tmp_path, name):
    db = YantrikDB(os.path.join(str(tmp_path), "slim.db"), 32)
    try:
        call = _keyword_calls(db, tmp_path)[name]
        with pytest.raises(RuntimeError) as err:
            call()
        if "embedder-download" in str(err.value):
            assert "--no-default-features" in str(err.value)
    finally:
        db.close()


# ── the same guarantee, checked on the source in every CI run ────────────
#
# CI does not build a slim wheel to run the test above against, so this also
# compares the Rust source: every method compiled out by
# `#[cfg(not(feature = "embedder-download"))]` must take the same
# Python-visible parameters as its real twin.

import pathlib  # noqa: E402
import re  # noqa: E402

_SRC = pathlib.Path(__file__).resolve().parent.parent / "crates" / "yantrikdb-python" / "src"
_RECEIVERS = {"self", "&self", "&mut self"}


def _split_params(text):
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _python_visible(params):
    names = []
    for p in params:
        if p in _RECEIVERS:
            continue
        name, _, ty = p.partition(":")
        name, ty = name.strip(), ty.strip()
        if ty.startswith("Python<") or ty.startswith("&Bound<'_, Self>"):
            continue  # injected by PyO3, not passed by the caller
        names.append(name)
    return names


def cfg_twins(source):
    """{name: {"on": [params], "off": [params]}} for embedder-download twins."""
    found = {}
    pending = None
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == '#[cfg(feature = "embedder-download")]':
            pending = "on"
        elif line == '#[cfg(not(feature = "embedder-download"))]':
            pending = "off"
        m = re.match(r"(?:pub(?:\(crate\))? )?fn (\w+)\(", line)
        if m and pending:
            sig = line[m.end():]
            j = i
            while ")" not in sig.split("->")[0] or sig.count("(") >= sig.count(")") + 1:
                j += 1
                sig += " " + lines[j].strip()
                if ") ->" in sig or sig.rstrip().endswith(") {"):
                    break
            params = sig.split(") ->")[0].split(") {")[0]
            found.setdefault(m.group(1), {})[pending] = _python_visible(_split_params(params))
            pending = None
            i = j
        elif line and not line.startswith(("#[", "///", "//")):
            pending = None if not m else pending
        i += 1
    return {k: v for k, v in found.items() if "on" in v and "off" in v}


def test_every_slim_stub_takes_its_real_methods_parameters():
    if not _SRC.is_dir():
        pytest.skip("the Rust sources are not beside this test (installed tests)")
    twins = {}
    for path in sorted(_SRC.rglob("*.rs")):
        twins.update(cfg_twins(path.read_text(encoding="utf-8")))
    assert {"convert_pack", "reembed", "set_embedder_named"} <= set(twins), sorted(twins)
    for name, pair in twins.items():
        assert pair["off"] == pair["on"], (
            f"slim stub {name}{pair['off']} differs from {name}{pair['on']}: a keyword "
            f"call would raise TypeError instead of the stub's RuntimeError"
        )
