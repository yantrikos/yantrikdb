#!/usr/bin/env python3
"""Build a sealed .ydbpack from a pack source directory.

A pack source is a directory containing `pack.toml` and `corpus.md`.
Facts in `corpus.md` are separated by `## ` headings; the heading is the
topic and the following prose is the fact. The stored text is
`"<topic> — <body>"` so retrieval matches on both.

Usage:
    python packs/build.py packs/yantrikdb-engine
    python packs/build.py --all

Design notes worth keeping:

- **One fact per record.** Retrieval returns records, not documents, so a
  record that only makes sense next to its neighbours is a record that
  will be served without them. Each entry must stand alone.
- **Importance 0.6, not 1.0.** Write-time calibration deflates a
  namespace once it passes MIN_COUNT=8 writes at a high mean
  (`engine/importance.rs`), so a pack that stamps everything 1.0 pushes
  its own later facts *below* its earlier ones. 0.6 sits under the
  saturation threshold and keeps every fact comparable.
- **source="document".** The provenance gate refuses `source=inference`
  claiming `kind=fact`; a pack asserting authored knowledge declares
  itself a document, which is what it is.
"""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import tomllib
from pathlib import Path

from yantrikdb import YantrikDB

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"


def parse_corpus(path: Path) -> list[tuple[str, str]]:
    """Split corpus.md into (topic, body) pairs on `## ` headings."""
    text = path.read_text(encoding="utf-8")
    # Drop an optional preamble before the first heading.
    parts = re.split(r"^## +", text, flags=re.MULTILINE)[1:]
    out: list[tuple[str, str]] = []
    for part in parts:
        lines = part.strip().split("\n")
        topic = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        # Citations are for the human reader, not the embedding.
        body = re.sub(r"^_cite:.*$", "", body, flags=re.MULTILINE).strip()
        body = re.sub(r"\n{2,}", " ", body).replace("\n", " ").strip()
        if topic and body:
            out.append((topic, body))
    return out


def build(src: Path, out_dir: Path = DIST) -> Path:
    cfg = tomllib.loads((src / "pack.toml").read_text(encoding="utf-8"))
    pack = cfg["pack"]
    content = cfg.get("content", {})

    namespace = pack["namespace"]
    facts = parse_corpus(src / "corpus.md")
    if not facts:
        raise SystemExit(f"{src}: corpus.md produced no facts")

    # Tier 1: optional constitution.md, one rule per `## ` heading. These
    # inject on EVERY turn while mounted, so the file should hold only
    # rules that fail if they are ever absent — reference facts belong in
    # corpus.md where retrieval serves them on demand. seal_pack enforces
    # a ~1500-token budget and refuses an oversized one.
    constitution: list[str] = []
    con_path = src / "constitution.md"
    if con_path.exists():
        constitution = [f"{topic}: {body}" for topic, body in parse_corpus(con_path)]

    # Tier 3: coverage index — author-curated short phrases in pack.toml,
    # NOT auto-derived from the 45 corpus headings. The index is read on
    # every turn, so it must be a handful of phrases a model can hold,
    # not a table of contents.
    coverage: list[str] = list(pack.get("coverage", []))

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{pack['name']}-{pack['version']}.ydbpack"
    if dest.exists():
        dest.unlink()  # seal_pack refuses to overwrite, by design

    # Staging lives in a fresh temp dir per run. Reusing a staging file
    # would re-record the corpus on top of the previous build's rows and
    # silently ship a pack with every fact duplicated. Windows also keeps
    # the file locked briefly after close, so cleanup errors are ignored.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = YantrikDB(str(Path(td) / "staging.db"), 64)
        for topic, body in facts:
            db.record_text(
                f"{topic} — {body}",
                memory_type=content.get("memory_type", "semantic"),
                importance=float(content.get("importance", 0.6)),
                namespace=namespace,
                domain=content.get("domain", "general"),
                source=content.get("source", "document"),
                certainty=float(content.get("certainty", 0.9)),
            )
        manifest = db.seal_pack(
            str(dest),
            name=pack["name"],
            version=pack["version"],
            origin=pack["origin"],
            namespace=namespace,
            description=pack.get("description"),
            constitution=constitution or None,
            coverage=coverage or None,
        )
        db.close()

    size_kb = dest.stat().st_size / 1024
    tiers = f"{manifest['corpus_rows']:>3} facts"
    if constitution:
        approx = sum(len(r) + 1 for r in constitution) // 4
        tiers += f"  {len(constitution):>2} rules (~{approx} tok)"
    if coverage:
        tiers += f"  {len(coverage)} topics"
    print(f"built {manifest['pack_id']:<38} {tiers:<38} {size_kb:>7.1f} KB")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", nargs="*", help="pack source directory")
    ap.add_argument("--all", action="store_true", help="build every pack here")
    ap.add_argument("--clean", action="store_true", help="wipe dist/ first")
    args = ap.parse_args()

    if args.clean and DIST.exists():
        shutil.rmtree(DIST)

    sources = [Path(s) for s in args.source]
    if args.all or not sources:
        sources = sorted(p.parent for p in HERE.glob("*/pack.toml"))
    for src in sources:
        build(src)


if __name__ == "__main__":
    main()
