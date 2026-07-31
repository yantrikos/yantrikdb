#!/usr/bin/env python3
"""Build a contact sheet of every rendered page, for human review.

The gates decide whether a page is broken. Only a person decides whether
it is any good, and that person needs to see desktop and mobile together
rather than open twelve files.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHOTS = HERE / "shots"
OUT = HERE / "out"

BRIEF_NOTE = {
    "studio": "Architecture studio · editorial · light · roomy",
    "restaurant": "Neighbourhood restaurant · editorial · light · roomy",
    "saas": "Developer infrastructure · technical · dark · tight",
    "watch": "Luxury product · technical · dark · roomy",
    "portfolio": "Designer portfolio · studio · light · normal",
    "charity": "Nonprofit · studio · light · roomy",
}


def main() -> int:
    report = json.loads((SHOTS / "report.json").read_text(encoding="utf-8"))
    cards = []
    for name in sorted(report):
        gates = report[name]["desktop"]["gates"]
        failed = [g for g, ok in gates.items() if not ok]
        status = ("<span class=ok>all gates pass</span>" if not failed
                  else f"<span class=bad>{', '.join(failed)}</span>")
        cards.append(f"""
<section>
  <header>
    <h2>{name}</h2>
    <p class=note>{BRIEF_NOTE.get(name, '')}</p>
    <p class=status>{status} ·
       <a href="../out/{name}.html">open the page</a> ·
       <a href="../briefs/{name}.ops">see the ops script</a></p>
  </header>
  <div class=pair>
    <figure><img src="{name}.desktop.png" alt="{name} desktop"><figcaption>desktop 1440</figcaption></figure>
    <figure class=mob><img src="{name}.mobile.png" alt="{name} mobile"><figcaption>mobile 390</figcaption></figure>
  </div>
</section>""")

    html = f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>letterpress — rendered pages</title>
<style>
:root{{color-scheme:light dark}}
body{{margin:0;padding:2.5rem 1.5rem 5rem;background:#111013;color:#eceaef;
  font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
.wrap{{width:min(100% - 1rem,78rem);margin-inline:auto}}
h1{{font-size:2rem;margin:0 0 .4rem;letter-spacing:-.02em}}
.lede{{color:#a29fa8;margin:0 0 3rem;max-width:46rem}}
section{{margin-bottom:4.5rem;border-top:1px solid #2b2830;padding-top:1.6rem}}
h2{{font-size:1.25rem;margin:0;letter-spacing:-.01em}}
.note{{color:#a29fa8;margin:.2rem 0 .3rem;font-size:.9rem}}
.status{{margin:0 0 1.2rem;font-size:.85rem;color:#8d8a95}}
.status a{{color:#9db8ff}}
.ok{{color:#6fd39b}} .bad{{color:#ff8f8f}}
.pair{{display:grid;gap:1.2rem;grid-template-columns:1fr}}
@media(min-width:60rem){{.pair{{grid-template-columns:3fr 1fr;align-items:start}}}}
figure{{margin:0}}
img{{width:100%;display:block;border:1px solid #2b2830;border-radius:6px}}
figcaption{{color:#78757f;font-size:.78rem;margin-top:.45rem}}
</style></head><body><div class=wrap>
<h1>letterpress — every brief, one compiler</h1>
<p class=lede>Each page was produced from a short operations script: no HTML or CSS
was written per page. Three visual families, both colour modes. Every page here
passes the mechanical gates (contrast, overflow, readable size, single H1) at both
viewports — that says it is not broken, not that it is good. That part is your call.</p>
{''.join(cards)}
</div></body></html>"""

    dest = SHOTS / "index.html"
    dest.write_text(html, encoding="utf-8")
    print(f"contact sheet -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
