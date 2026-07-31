#!/usr/bin/env python3
"""The second control arm: one call, the whole page.

The five-call direct arm failed every gate, and the mechanism was style
collision — five independent <style> blocks each inventing their own
custom-property names, so `color: var(--text-primary)` in one fragment
resolved against a background another fragment set. The obvious attack
on that result is that fragment assembly, not free-form authoring, did
the damage. So this arm removes the confound entirely: one call, one
document, one stylesheet, nothing assembled by the harness.

It is deliberately the STRONGEST form of the control. It gets a bigger
token budget than either five-call arm, because a whole page needs one,
and giving it less would be rigging the comparison.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLLAMA = "http://localhost:11434"

SYSTEM = """You are a web designer who writes hand-crafted HTML and CSS.

DESIGN GUIDANCE
  Pick ONE accent colour and use it consistently. Body text must reach
  at least 4.5:1 contrast against its background; large headings 3:1.
  Set a type scale from one ratio and stay on it. Body copy no smaller
  than 16px; labels no smaller than 12px unless uppercase and tracked.
  Keep line length near 65 characters. Use system font stacks only, no
  webfonts and no external requests of any kind. Every page must work
  at 390px wide with no horizontal scrolling. Exactly one <h1>.
  Write specific copy: real names, real numbers. Never "Lorem ipsum".
  Include at least one decorative INLINE SVG. No external images.

OUTPUT FORMAT
  Emit ONE complete HTML document, starting <!doctype html> and ending
  </html>, with all CSS in a single <style> block in the head.
  No markdown, no code fences, no commentary."""

USER = """Site: {brief}

Write the complete single-page site: a masthead with a wordmark and nav,
a hero with an eyebrow, one h1, a lede and two buttons, a decorative
inline SVG, a features section with three items, a closing call to
action, and a footer."""

FENCE = re.compile(r"```[a-z]*\n?|```")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--brief", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    payload = json.dumps({
        "model": args.model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": USER.format(brief=args.brief)}],
        "stream": False, "think": False, "keep_alive": "20m",
        # 4500, not 900: five calls x 900 was the fragment arm's budget,
        # and a whole document needs at least as much in one response.
        "options": {"num_predict": 4500, "temperature": 0.2},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        raw = json.load(r).get("message", {}).get("content", "") or ""

    page = FENCE.sub("", raw).strip()
    # Record truncation honestly: an unterminated document is a result,
    # not something to patch up before measuring.
    closed = page.lower().rstrip().endswith("</html>")
    out = HERE / "out" / f"{args.name}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"  {args.name:<14} {len(page):>6} chars, "
          f"closed={'yes' if closed else 'NO (truncated)'}, "
          f"h1={page.lower().count('<h1')}, svg={page.lower().count('<svg')}"
          f" -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
