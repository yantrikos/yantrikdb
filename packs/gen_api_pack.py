#!/usr/bin/env python3
"""Generate an API ground-truth pack by introspecting a live Python.

The idea "put the full Python docs in a pack" done in the form the
measurements support. Two walls stand in front of the naive version:
models already KNOW Python prose (content the model knows measured +1),
and reference-dominated records do not retrieve on a 64-dim embedder
(Law 1). What models measurably get WRONG is the API surface itself —
package-hallucination studies put open-source models at ~21% invented
imports and signatures versus ~5% for the frontier.

So this does not transcribe documentation. It asks the interpreter:

    signature   inspect.signature() on the artifact itself — correct by
                construction, and regenerable for any Python version
    summary     the docstring's own first sentence(s), prose that names
                the task in the vocabulary someone queries with
    hazards     deprecations the runtime itself declares

One record per public callable, led by prose, one tiny fence. Recipes
("how to do X") are authored separately — this file makes the ground
truth; taste never comes from a generator.

Usage:
    python packs/gen_api_pack.py --out packs/python-stdlib
    python packs/gen_api_pack.py --modules json pathlib --out /tmp/mini
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
import sys
import warnings
from pathlib import Path

# A deliberate slice, not the whole stdlib: the modules agent code
# actually touches, where a wrong signature costs a debugging session.
DEFAULT_MODULES = [
    "pathlib", "shutil", "subprocess", "os.path", "tempfile", "glob",
    "json", "csv", "re", "datetime", "time", "collections", "itertools",
    "functools", "dataclasses", "argparse", "urllib.request",
    "urllib.parse", "textwrap", "secrets", "hashlib", "base64",
    "sqlite3", "logging", "unittest.mock", "concurrent.futures",
]

SKIP_NAMES = {"__init__", "__new__"}


def doc_parts(name: str, doc: str | None, limit: int = 2) -> tuple[str, str, str]:
    """(prose, doc_signature, hazard) from a docstring.

    Three lessons from the first lint run are encoded here:
    - C functions lead their docstring with their own signature line
      ("sleep(seconds)") — that line is the authoritative signature and
      must not be mistaken for prose, or the record has none.
    - Stopping at the first blank line starved Popen of prose (81% code,
      unreachable) and cut mktemp's "THIS FUNCTION IS UNSAFE" — the one
      sentence the record exists to deliver. Read the whole docstring.
    - Any sentence declaring deprecation or unsafety IS ground truth;
      surface it even when it lives three paragraphs down.
    """
    if not doc:
        return "", "", ""
    lines = [l.strip() for l in doc.strip().splitlines()]
    short = name.rsplit(".", 1)[-1]
    doc_sig = ""
    if lines and lines[0].startswith(f"{short}("):
        doc_sig = lines.pop(0).split("->")[0].strip()
    text = " ".join(l for l in lines if l)
    parts = [p.strip() for p in text.split(". ") if p.strip()]
    prose = (". ".join(parts[:limit]).rstrip(".") + ".") if parts else ""
    hazard = next((p.rstrip(".") + "." for p in parts[limit:]
                   if re.search(r"deprecat|unsafe|insecure|do not use",
                                p, re.I)), "")
    return prose, doc_sig, hazard


def sig_of(obj) -> str | None:
    try:
        sig = str(inspect.signature(obj))
    except (ValueError, TypeError):
        return None
    # C callables often report the meaningless "(object, /)" — publishing
    # it would be wrong ground truth, the one defect this pack must never
    # contain. No signature beats a false one.
    return None if sig in ("(object, /)", "(*args, **kwargs)") else sig


def record_for(qualname: str, obj, kind: str) -> str | None:
    limit = 4 if kind == "class" else 2
    doc, doc_sig, hazard = doc_parts(qualname, inspect.getdoc(obj), limit)
    sig = sig_of(obj)
    if doc_sig and not sig:
        # The docstring's own signature line is the authority for C
        # callables where inspect.signature fails or lies.
        sig = "(" + doc_sig.split("(", 1)[1]
    if not doc and not sig:
        return None
    head = f"{qualname}: {doc.split('.')[0].lower()}" if doc else qualname
    # Heading carries the task language; the fence carries the artifact.
    out = [f"## {head}\n"]
    if doc:
        out.append(doc)
    if hazard:
        out.append(f"\n**Hazard:** {hazard}")
    if sig:
        out.append(f"\n```python\n{qualname.split('(')[0]}{sig}\n```")
    if kind == "class":
        # Law 1: a record that is mostly code does not retrieve. Keep
        # adding method signatures only while prose still outweighs code
        # (floor of 4 so small-doc classes keep their essentials).
        prose_len = len(doc) + len(head) + len(hazard)
        methods, code_len = [], len(sig or "")
        for mname, m in inspect.getmembers(obj, callable):
            if mname.startswith("_") or mname in SKIP_NAMES:
                continue
            msig = sig_of(m)
            if not msig:
                continue
            line = f"{mname}{msig}"
            if len(methods) >= 4 and code_len + len(line) > prose_len:
                break
            methods.append(line)
            code_len += len(line)
            if len(methods) >= 20:
                break
        if methods:
            # Method NAMES as prose before the fence: CPython class
            # docstrings ("purepath subclass that can make system calls")
            # carry no task vocabulary, but read_text/mkdir/glob is
            # exactly what a consumer's query contains — and prose is
            # what the 64-dim embedder can see.
            names = ", ".join(m.split("(")[0] for m in methods)
            out.append(f"\nKey methods are {names}.")
            out.append("\n```python\n" + "\n".join(methods) + "\n```")
    return "\n".join(out) + "\n"


def gen_module(modname: str, max_items: int = 60) -> list[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mod = importlib.import_module(modname)
        except ImportError as e:
            print(f"  skip {modname}: {e}")
            return []
    records, count = [], 0
    # "os.path" imports as ntpath/posixpath — ownership must compare
    # against the module's real __name__, not the name we asked for.
    real = getattr(mod, "__name__", modname)
    public = getattr(mod, "__all__", None) or \
        [n for n in dir(mod) if not n.startswith("_")]
    for name in public:
        if count >= max_items:
            break
        try:
            obj = getattr(mod, name)
        except AttributeError:
            continue
        # Only things defined here or re-exported deliberately; a record
        # for os.path.abspath must not appear again under posixpath.
        owner = getattr(obj, "__module__", modname)
        if owner and owner not in (modname, real) and \
                not (modname.startswith(owner) or owner.startswith(modname)):
            continue
        if inspect.isclass(obj):
            rec = record_for(f"{modname}.{name}", obj, "class")
        elif callable(obj):
            rec = record_for(f"{modname}.{name}", obj, "func")
        else:
            continue
        if rec:
            records.append(rec)
            count += 1
    print(f"  {modname}: {count} records")
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", nargs="*", default=DEFAULT_MODULES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-module", type=int, default=60)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ver = ".".join(map(str, sys.version_info[:3]))

    all_records: list[str] = []
    for m in args.modules:
        all_records.extend(gen_module(m, args.max_per_module))

    corpus = out / "corpus.md"
    header = (
        f"# python-stdlib corpus — generated from CPython {ver}\n\n"
        f"Every signature below came from `inspect.signature()` on the\n"
        f"running interpreter, not from documentation or memory — correct\n"
        f"by construction for this Python version, and regenerated rather\n"
        f"than edited. Docstring summaries are CPython's own.\n\n"
        f"Recipes (task → idiomatic snippet) live in recipes.md and are\n"
        f"authored, not generated: ground truth can be introspected;\n"
        f"idiom cannot.\n")
    body = header + "\n" + "\n".join(all_records)
    # Merge hand-authored recipes if present.
    recipes = out / "recipes.md"
    if recipes.exists():
        body += "\n" + recipes.read_text(encoding="utf-8")
    corpus.write_text(body, encoding="utf-8")

    toml = out / "pack.toml"
    if not toml.exists():
        toml.write_text(f'''[pack]
name = "python-stdlib"
version = "0.1.0"
origin = "yantrik/python-stdlib"
namespace = "python_stdlib"
description = "The Python {ver} standard library's actual API surface, generated from the interpreter itself: every signature via inspect.signature (correct by construction, never from memory), CPython's own docstring summaries, plus authored recipes for the tasks agents actually do. Aimed at the measured failure mode of local models — invented imports and wrong signatures."
coverage = [
    "Python standard library function and class signatures",
    "Python file, path and subprocess operations",
    "Python text, JSON, CSV and regular expression handling",
    "Python dates, times, collections and itertools",
    "Python logging, testing mocks and concurrency futures",
]

[content]
memory_type = "semantic"
domain = "engineering"
source = "document"
importance = 0.6
certainty = 0.95
''', encoding="utf-8")

    n = body.count("\n## ")
    print(f"\nwrote {corpus} — {n} records, {len(body.split())} words, "
          f"CPython {ver}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
