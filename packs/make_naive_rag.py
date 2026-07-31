#!/usr/bin/env python3
"""Build a naive chunk-and-embed pack, as the control a curated pack must beat.

This is the experiment the whole product rests on and that we had never
run. Every efficacy number we publish compares a pack against NO pack,
which answers "does retrieval help" — a question nobody was asking. The
question buyers and reviewers actually ask is "does your *curation* beat
chunking the same document", and if the answer is no, the content
product largely disappears and what remains is distribution and
provenance machinery.

So this arm gets every advantage that is fair: the publisher's own full
text, the same 64-dimensional embedder, the same pack pipeline, the same
retrieval gate, the same top-k. The only variable removed is authorship
— no prose-led records, no task-vocabulary headings, no constitution.

Fixed-size overlapping chunks are the honest naive baseline: it is what
a default LangChain/LlamaIndex splitter does, and what anyone would get
in an afternoon without reading the retrieval laws.

Usage:
    python packs/make_naive_rag.py --source llms-full.txt \\
        --filter 2026-07-28 --out packs/mcp-naive
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

CHUNK = 1200      # characters — a common default splitter size
OVERLAP = 200


def select(text: str, needle: str | None) -> str:
    """Keep only the sections whose page path matches the revision.

    The docs dump carries every protocol revision. Indexing all of them
    would hand the naive arm a harder problem than the curated pack
    faced, and the question here is curation, not corpus hygiene — so
    both arms get the same revision's pages.
    """
    if not needle:
        return text
    # Pages in llms-full.txt are delimited by markdown H1s, each followed
    # by a `Source: <url>` line. Split on the H1 and match against the
    # page header. Multiple needles may be given comma-separated, since
    # the curated pack was built from two path prefixes.
    parts = re.split(r"(?m)^(?=# )", text)
    needles = [n.strip() for n in needle.split(",") if n.strip()]
    keep = [p for p in parts if any(n in p[:400] for n in needles)]
    if not keep:
        raise SystemExit(f"filter {needle!r} matched no pages of {len(parts)}")
    print(f"filter kept {len(keep)} of {len(parts)} pages")
    return "\n".join(keep)


def chunks(text: str) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text)
    out, i = [], 0
    while i < len(text):
        piece = text[i : i + CHUNK].strip()
        if len(piece) > 120:
            out.append(piece)
        i += CHUNK - OVERLAP
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--filter", default=None,
                    help="keep only pages whose header mentions this")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="mcp-naive")
    args = ap.parse_args()

    raw = Path(args.source).read_text(encoding="utf-8", errors="replace")
    body = select(raw, args.filter)
    pieces = chunks(body)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {args.name} — naive fixed-size chunks of the publisher's own text\n",
        "Not a curated pack. Chunked at "
        f"{CHUNK} characters with {OVERLAP} overlap, no authored prose, no "
        "task-vocabulary headings, no constitution. This exists to answer "
        "whether curation beats chunking at a matched retrieval budget.\n",
    ]
    for n, piece in enumerate(pieces, 1):
        # A naive store embeds the chunk itself. build.py records
        # "topic — body", so the topic is the chunk's own opening words
        # rather than an authored heading; anything else would either
        # flatter this arm or handicap it.
        head = " ".join(piece.split()[:8]).replace("#", "").strip() or f"chunk {n}"
        lines.append(f"## {head}\n\n{piece}\n")

    (out / "corpus.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "pack.toml").write_text(f'''[pack]
name = "{args.name}"
version = "0.1.0"
origin = "yantrik/{args.name}"
namespace = "{args.name.replace('-', '_')}"
description = "Control arm: naive fixed-size chunks of the source document, no curation. Built to measure whether authored records beat chunking at a matched retrieval budget."
coverage = ["control arm, not for publication"]

[content]
memory_type = "semantic"
domain = "engineering"
source = "document"
importance = 0.6
certainty = 0.95
''', encoding="utf-8")

    words = sum(len(p.split()) for p in pieces)
    print(f"{len(pieces)} chunks, {words} words -> {out / 'corpus.md'}")
    print(f"(source {len(raw)} chars, after filter {len(body)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
