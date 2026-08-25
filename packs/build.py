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
from pathlib import Path

try:  # py310-ok: tomllib is 3.11+; tomli is the backport
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as e:
        raise SystemExit("packs tooling needs Python 3.11+ or `pip install tomli` on 3.10") from e

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



# An external embedder, attached through the engine's public
# set_embedder(). The bundled default is a static lookup-table
# distillation: measured against the same corpus and questions it
# reaches 46/53, where a contextual embedder reaches 50/53, and the
# potion family does not close that gap at any width (256d 47/53,
# 512d 45/53). A pack is only as findable as the vectors it ships.
class OllamaEmbedder:
    def __init__(self, model: str, host: str):
        self.model, self.host = model, host

    def encode(self, text):
        import json as _json, urllib.request as _u
        payload = _json.dumps({"model": self.model,
                               "prompt": str(text)[:2000]}).encode()
        req = _u.Request(f"{self.host}/api/embeddings", data=payload,
                         headers={"Content-Type": "application/json"})
        with _u.urlopen(req, timeout=120) as r:
            return _json.load(r)["embedding"]


def build(src: Path, out_dir: Path = DIST, dim: int = 64,
          embedder_name: str | None = None,
          ollama_model: str | None = None,
          ollama_host: str = "http://192.168.4.35:11434") -> Path:
    cfg = tomllib.loads((src / "pack.toml").read_text(encoding="utf-8"))
    pack = cfg["pack"]
    content = cfg.get("content", {})

    namespace = pack["namespace"]
    facts = parse_corpus(src / "corpus.md")
    if not facts:
        raise SystemExit(f"{src}: corpus.md produced no facts")

    # A record that is mostly code does not retrieve on its own topic.
    # The bundled embedder is 64-dimensional, so an embedding is dominated
    # by whatever the record has most of: a 2.5KB theme.json exemplar was
    # unreachable by every query tried, while the shortest exemplar — most
    # prose, least code — won even for the other's queries. A fact nobody
    # can retrieve is dead weight that costs pack size and delivers
    # nothing, and the failure is silent because the record is present and
    # correct. Warn at build time; `lint_pack.py` proves it per record.
    heavy = []
    for topic, body in facts:
        fenced = sum(len(b) for b in re.findall(r"```.*?```", body, re.S))
        if len(body) > 400 and fenced / max(len(body), 1) > 0.5:
            heavy.append((topic, int(100 * fenced / len(body))))
    if heavy:
        print(f"  warning: {len(heavy)} record(s) are code-dominated and may not "
              f"be retrievable on their own topic — lead with prose, keep the "
              f"snippet short:")
        for topic, pct in heavy[:6]:
            print(f"    {pct:>3}% code  {topic[:60]}")

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
        db = YantrikDB(str(Path(td) / "staging.db"), dim)
        if embedder_name:
            db.set_embedder_named(embedder_name)
        elif ollama_model:
            db.set_embedder(OllamaEmbedder(ollama_model, ollama_host))
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
            # The swept retrieval settings travel INSIDE the sealed pack.
            # They used to live only here in pack.toml, which was fine
            # while the author and the consumer were the same party and
            # useless the moment they are not: a host holding only the
            # file had to guess a floor for a corpus it had never seen,
            # and a guessed floor is what injects near-domain records
            # into questions the pack should have declined.
            recommended_top_k=content.get("recommended_top_k"),
            recommended_min_similarity=content.get("recommended_min_similarity"),
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
    ap.add_argument("--dim", type=int, default=64,
                    help="vector width; must match the embedder")
    ap.add_argument("--embedder",
                    help="downloadable model name, e.g. potion-base-8M")
    ap.add_argument("--ollama-embedder",
                    help="ollama model name, e.g. nomic-embed-text")
    ap.add_argument("--ollama-host", default="http://192.168.4.35:11434")
    args = ap.parse_args()

    if args.clean and DIST.exists():
        shutil.rmtree(DIST)

    sources = [Path(s) for s in args.source]
    if args.all or not sources:
        sources = sorted(p.parent for p in HERE.glob("*/pack.toml"))
    for src in sources:
        build(src, dim=args.dim, embedder_name=args.embedder,
              ollama_model=args.ollama_embedder, ollama_host=args.ollama_host)


if __name__ == "__main__":
    main()
