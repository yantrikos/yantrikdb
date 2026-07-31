#!/usr/bin/env python3
"""The control arm: the same model writing HTML and CSS itself.

Without this, "6/6 conformance" only says a 4B can fill in a form we
designed. The question that decides whether the compiler earns its
complexity is whether the same model, given the same brief, the same
number of calls and the same design guidance, does better or worse
writing the markup directly.

Matched deliberately, because an unmatched comparison proves nothing:
  - same model, same temperature, same token budget
  - FIVE calls, one section fragment each, never the whole document
  - the same taste guidance the ops grammar carries, in prose
  - fragments assembled by the harness into the same shell
  - identical render gates

The one thing that cannot be matched is conformance, since free-form
HTML has no grammar to violate. So this arm is scored on the render
gates alone, and the ops arm is compared on those same gates.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLLAMA = "http://localhost:11434"

GUIDANCE = """You write small fragments of HTML for a marketing page, plus the CSS
they need. You are given the same design guidance a component library
would give you.

DESIGN GUIDANCE
  Pick ONE accent colour and use it consistently. Body text must reach
  at least 4.5:1 contrast against its background; large headings 3:1.
  Set a type scale from one ratio and stay on it. Body copy no smaller
  than 16px; labels no smaller than 12px unless uppercase and tracked.
  Keep line length near 65 characters. Use system font stacks only, no
  webfonts and no external requests of any kind. Every page must work
  at 390px wide with no horizontal scrolling. Exactly one <h1>.
  Write specific copy: real names, real numbers. Never "Lorem ipsum".

OUTPUT FORMAT
  Emit ONLY HTML. Put any CSS this fragment needs in a single <style>
  block inside the fragment. No markdown, no code fences, no commentary."""

CALLS = [
    ("hero", """Site: {brief}

Write the hero section: a <section> containing an eyebrow line, one
<h1>, a short lede paragraph, and two links styled as buttons. Include
its CSS in a <style> block. Emit only HTML."""),
    ("art", """Site: {brief}

Write a decorative visual for the hero as INLINE SVG inside a
<figure>, with a caption. No external images. Include its CSS.
Emit only HTML."""),
    ("features", """Site: {brief}

Write a features section: a <section> with a small eyebrow label, an
<h2>, and three items each with a title and one or two specific
sentences. Include its CSS. Emit only HTML."""),
    ("cta", """Site: {brief}

Write a closing call-to-action <section> with an <h2>, one short
paragraph and one link styled as a button. Include its CSS.
Emit only HTML."""),
    ("chrome", """Site: {brief}

Write a masthead <header> with a wordmark and three nav links, and a
site <footer> with the name, location and a contact. Include their
CSS. Emit only HTML."""),
]

FENCE = re.compile(r"```[a-z]*\n?|```")


def ask(model: str, system: str, user: str) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False, "think": False, "keep_alive": "20m",
        "options": {"num_predict": 900, "temperature": 0.2},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r).get("message", {}).get("content", "") or ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--brief", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    parts: dict[str, str] = {}
    for label, template in CALLS:
        raw = ask(args.model, GUIDANCE, template.format(brief=args.brief))
        frag = FENCE.sub("", raw).strip()
        parts[label] = frag
        has_style = "<style" in frag.lower()
        print(f"  {label:<9} {len(frag):>5} chars, style={'yes' if has_style else 'NO'}")

    chrome = parts.get("chrome", "")
    header = chrome[:chrome.lower().find("</header>") + 9] if "</header>" in chrome.lower() else ""
    footer = chrome[chrome.lower().find("<footer"):] if "<footer" in chrome.lower() else ""

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{args.name}</title></head>
<body>
{header}
<main>
{parts.get('hero','')}
{parts.get('art','')}
{parts.get('features','')}
{parts.get('cta','')}
</main>
{footer}
</body></html>
"""
    out = HERE / "out" / f"{args.name}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"  assembled -> {out}  ({len(page)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
