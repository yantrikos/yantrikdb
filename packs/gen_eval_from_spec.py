#!/usr/bin/env python3
"""Derive an eval set mechanically from a specification's own normative text.

Hand-written questions leak. Measured on 2026-07-30: Claude Fable 5
scored 29/53 cold on the hand-written `mcp-spec` set, against 7/53 for
a 4B — not because it knew the July protocol, but because questions like
"what must a client do with the `requestState` it receives in an
`InputRequiredResult`" hand a capable model both concepts and let it
reconstruct the answer. The pack's real gain was hiding under the
leak, and so was the local models' real ignorance.

A cloze over the spec's own MUST/SHOULD sentences removes the leak by
construction. The question is the requirement with one arbitrary token
blanked; the answer is that token. Three properties follow for free:

  - **No leak.** The blanked token cannot appear in its own question,
    and sentences that mention it twice are skipped.
  - **Citable ground truth.** The answer is the specification's wording,
    not my judgment, which retires the "self-authored evals" caveat.
  - **Targets what models actually miss.** Only *arbitrary* tokens are
    blanked — identifiers, error codes, header names, enum values. The
    error analysis showed a frontier model gets the derivable facts
    right and the arbitrary ones confidently wrong, so this is exactly
    the axis worth measuring.

Usage:
    python packs/gen_eval_from_spec.py --source spec.md --name "MCP 2026-07-28" \\
        --out packs/mcp-spec/eval_cloze.jsonl --limit 40
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

NORMATIVE = re.compile(r"\b(MUST NOT|MUST|SHOULD NOT|SHOULD|SHALL|REQUIRED)\b")

# What counts as arbitrary — a fact reasoning cannot reach.
ARBITRARY = [
    re.compile(r"`(-3\d{4})`"),                       # JSON-RPC error codes
    re.compile(r"`([a-z][a-z0-9]*/[a-z][A-Za-z0-9/_-]+)`"),   # method names
    re.compile(r"`((?:[A-Za-z][\w.-]*\.)+[A-Za-z][\w-]*/[\w.-]+)`"),  # reverse-DNS ids
    re.compile(r"`([A-Z][A-Za-z-]*-[A-Za-z-]+)`"),    # header names
    re.compile(r"`([a-z][a-zA-Z0-9]{3,})`"),          # camelCase field names
    re.compile(r'`"([a-z_]{3,})"`'),                  # enum string values
]

# Two kinds of sentence produce unanswerable questions, both found by
# reading the first generated batch rather than by theory:
#
#   - Illustrations. "two servers each exposing a `search` tool" blanks
#     to a question whose answer is an arbitrary example name nobody
#     could supply.
#   - Meta-sentences. "The tools page defines `annotations` only at this
#     level" describes the documentation, not a requirement, so the
#     blank tests whether you have read this paragraph.
ILLUSTRATIVE = re.compile(
    r"\b(for example|e\.g\.|such as|for instance|say|imagine)\b", re.I)
META = re.compile(
    r"\b(this page|the .{0,20}page|this section|the spec(ification)?'s example"
    r"|documented|described (above|below)|see )\b", re.I)

# Tokens common enough that blanking them tests English, not the spec.
TOO_GENERIC = {
    "true", "false", "null", "string", "number", "object", "array",
    "boolean", "integer", "value", "params", "result", "method",
    "request", "response", "error", "data", "content", "type", "name",
}


def sentences(text: str) -> list[str]:
    # Strip fenced code: a requirement stated only inside an example is
    # not a sentence, and fences are where the answer usually also sits.
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\s+", " ", text)
    out = []
    for chunk in re.split(r"(?<=[.;])\s+(?=[A-Z`])", text):
        chunk = chunk.strip()
        if not (60 <= len(chunk) <= 400) or not NORMATIVE.search(chunk):
            continue
        if ILLUSTRATIVE.search(chunk) or META.search(chunk):
            continue
        out.append(chunk)
    return out


def arbitrary_tokens(sentence: str) -> list[str]:
    found: list[str] = []
    for pat in ARBITRARY:
        for m in pat.finditer(sentence):
            tok = m.group(1)
            if tok.lower() in TOO_GENERIC or len(tok) < 3:
                continue
            if tok not in found:
                found.append(tok)
    return found


def make_item(spec: str, sentence: str, token: str, idx: int) -> dict | None:
    # A token appearing twice would survive the blank and leak.
    if sentence.count(token) != 1:
        return None
    blanked = sentence.replace(f"`{token}`", "____", 1)
    if "____" not in blanked:
        return None
    # Anything that still spells the answer defeats the exercise —
    # including a case variant or an underscore/hyphen respelling.
    stem = re.sub(r"[^a-z0-9]", "", token.lower())
    if stem and stem in re.sub(r"[^a-z0-9]", "", blanked.lower()):
        return None
    return {
        "id": f"cloze-{idx:03d}",
        "q": (
            f"This is a normative requirement from {spec}, with one "
            f"identifier removed. Give exactly the term that belongs in "
            f'the blank, and nothing else: "{blanked}"'
        ),
        "expect": [[token]],
        "note": f"Cloze over the spec's own wording. Source sentence: {sentence[:160]}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="spec text/markdown file")
    ap.add_argument("--name", required=True, help='e.g. "MCP revision 2026-07-28"')
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    text = Path(args.source).read_text(encoding="utf-8", errors="replace")
    items, seen_tokens = [], set()
    for sentence in sentences(text):
        for token in arbitrary_tokens(sentence):
            # One question per distinct fact: ten questions whose answer
            # is `tools/call` measure one thing ten times.
            if token in seen_tokens:
                continue
            item = make_item(args.name, sentence, token, len(items) + 1)
            if item:
                items.append(item)
                seen_tokens.add(token)
                break
        if len(items) >= args.limit:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out} — {len(items)} cloze questions over {len(seen_tokens)} distinct terms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
