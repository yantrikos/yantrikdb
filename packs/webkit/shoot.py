#!/usr/bin/env python3
"""Render compiled pages and report the mechanically-decidable defects.

Two jobs, deliberately kept together: take the screenshot a human will
look at, and run the checks a human should not have to. The screenshot
is the point — every craft fix in the WordPress arc came from looking
at a render and writing a record, never from reading the markup.

The checks are only the ones a browser can actually decide. Whether the
page is beautiful is not among them and must never be smuggled in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VIEWPORTS = [("desktop", 1440, 900), ("mobile", 390, 844)]

PROBE = """() => {
  const out = {overflow: 0, tiny: [], lowcontrast: [], empty: false,
               h1: document.querySelectorAll('h1').length, longest: 0};
  out.overflow = Math.max(0, document.documentElement.scrollWidth
                             - document.documentElement.clientWidth);
  const text = (document.body.innerText || '').trim();
  out.empty = text.length < 120;
  out.words = text.split(/\\s+/).length;
  const lum = (c) => {
    const m = c.match(/\\d+(\\.\\d+)?/g); if (!m) return 1;
    const [r,g,b] = m.slice(0,3).map(v => {
      v = v/255; return v <= 0.03928 ? v/12.92 : Math.pow((v+0.055)/1.055, 2.4);
    });
    return 0.2126*r + 0.7152*g + 0.0722*b;
  };
  const bgOf = (el) => {
    let n = el;
    while (n && n !== document.documentElement) {
      const c = getComputedStyle(n).backgroundColor;
      if (c && !c.includes('rgba(0, 0, 0, 0)')) return c;
      n = n.parentElement;
    }
    return 'rgb(255,255,255)';
  };
  const rgb = (c) => {
    const m = (c.match(/[\\d.]+/g) || ['0','0','0']).map(Number);
    return [m[0]||0, m[1]||0, m[2]||0, m.length > 3 ? m[3] : 1];
  };
  // What the pixel actually shows. The design leans on `opacity:.78` for
  // secondary text on inverted bands, and reading `color` alone treats
  // that as fully opaque — so a token could be faded to unreadable and
  // still be reported at its nominal ratio. Composite the declared
  // colour over its ground using every opacity between the element and
  // the body, plus any alpha in the colour itself.
  const painted = (el, bg) => {
    let a = rgb(getComputedStyle(el).color)[3];
    for (let n = el; n && n !== document.documentElement; n = n.parentElement)
      a *= parseFloat(getComputedStyle(n).opacity);
    const f = rgb(getComputedStyle(el).color), b = rgb(bg);
    return 'rgb(' + [0,1,2].map(i => f[i]*a + b[i]*(1-a)).join(',') + ')';
  };
  // Content hidden behind opacity still counts as innerText, so
  // HAS_CONTENT passed on a page whose last two sections were invisible.
  // Check paint, not markup.
  out.invisible = [];
  document.querySelectorAll('section,article,div,p,h1,h2').forEach(el => {
    const t = (el.innerText||'').trim();
    if (t.length > 20 && parseFloat(getComputedStyle(el).opacity) < 0.05)
      out.invisible.push(t.slice(0,40));
  });
  // Every element that OWNS text, not a hand-listed set of tags. The
  // list this replaces was `p,h1,h2,h3,a,span,li`, which silently
  // exempted <cite>, <blockquote>, <figcaption>, <td>, <button> and
  // <label> — and a <cite> attribution duly shipped at 2.51:1 on an
  // inverted band with the gate reporting PASS. A gate that misses is
  // worse than no gate, because it is also a claim.
  //
  // Direct text nodes only: measuring containers too would attribute a
  // child's colour to its parent and report the same defect twice, and
  // it is the element that SETS the colour we want named in the output.
  document.querySelectorAll('body *').forEach(el => {
    let t = '';
    for (const n of el.childNodes)
      if (n.nodeType === 3) t += n.nodeValue;
    t = t.trim();
    if (!t) return;
    const cs = getComputedStyle(el);
    // Text that never paints cannot have a contrast defect; blanking is
    // NO_INVISIBLE_TEXT's job, and double-reporting it here would only
    // bury the real failures.
    if (cs.display === 'none' || cs.visibility === 'hidden') return;
    if (!el.getClientRects().length) return;
    // Inside an <svg>, computed fontSize is in USER units and ignores the
    // viewBox transform, so an 11-unit dimension label reported 11px at
    // every viewport while actually rendering near 14px on desktop and
    // under 10px on mobile. Neither number was the truth. Scale by the
    // element's own screen CTM to get the size a reader actually sees —
    // the widened selector only surfaced this because <text> had never
    // been measured at all.
    let scale = 1;
    if (el.ownerSVGElement && el.getScreenCTM) {
      const m = el.getScreenCTM();
      if (m) scale = Math.sqrt(Math.abs(m.a * m.d - m.b * m.c)) || 1;
    }
    const size = parseFloat(cs.fontSize) * scale;
    // Calibrated against the hand-written reference, which failed this
    // check on 11.5px tracked uppercase nav links. A tracked micro-label
    // is a deliberate editorial convention and reads fine; accidentally
    // tiny text is a different animal, and that is what this catches —
    // the real bug it found was a 10.7px BUTTON label with no tracking,
    // produced by deriving the small step from a 1.5 display ratio.
    // So: 12px floor generally, 11px floor for uppercase text tracked
    // at 0.08em or more. Loosening a check because our own artifact
    // failed is exactly how graders rot, so the exemption is narrow and
    // still rejects the defect that motivated the check.
    const tracked = parseFloat(cs.letterSpacing) >= size * 0.08;
    const caps = cs.textTransform === 'uppercase' || t === t.toUpperCase();
    const floor = (tracked && caps) ? 11 : 12;
    if (size < floor) out.tiny.push(t.slice(0,40) + ' @' + size.toFixed(1) + 'px');
    const ground = bgOf(el);
    const l1 = lum(painted(el, ground)), l2 = lum(ground);
    const ratio = (Math.max(l1,l2)+0.05)/(Math.min(l1,l2)+0.05);
    const big = size >= 24 || (size >= 18.66 && parseInt(cs.fontWeight) >= 700);
    if (ratio < (big ? 3 : 4.5))
      out.lowcontrast.push(t.slice(0,40) + ' ' + ratio.toFixed(2) + ':1');
    // Longest body line in characters — the readability tell.
    if (el.tagName === 'P' && t.length > out.longest) {
      const w = el.getBoundingClientRect().width;
      const ch = w / (size * 0.5);
      out.longest = Math.max(out.longest, Math.round(ch));
    }
  });
  return out;
}"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+")
    ap.add_argument("--shots", default="packs/webkit/shots")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    shots = Path(args.shots)
    shots.mkdir(parents=True, exist_ok=True)
    report = {}

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for page_path in args.pages:
            src = Path(page_path).resolve()
            name = src.stem
            report[name] = {}
            for label, w, h in VIEWPORTS:
                # Verify the SETTLED page. With motion reduced, the
                # compiler's own media query puts every revealed block
                # in its final state, so the screenshot shows what a
                # reader ends up with rather than a frame of animation.
                pg = browser.new_page(viewport={"width": w, "height": h},
                                      reduced_motion="reduce")
                pg.goto(src.as_uri())
                pg.wait_for_timeout(250)
                res = pg.evaluate(PROBE)
                pg.screenshot(path=str(shots / f"{name}.{label}.png"),
                              full_page=(label == "desktop"))
                pg.close()

                gates = {
                    "HAS_CONTENT": not res["empty"],
                    "NO_OVERFLOW": res["overflow"] <= 1,
                    "CONTRAST_AA": not res["lowcontrast"],
                    "READABLE_SIZE": not res["tiny"],
                    "ONE_H1": res["h1"] == 1,
                    "NO_INVISIBLE_TEXT": not res.get("invisible"),
                }
                report[name][label] = {"gates": gates, "detail": res}
                bad = [g for g, ok in gates.items() if not ok]
                flag = "PASS" if not bad else "FAIL " + ",".join(bad)
                extra = ""
                if res["longest"] > 95:
                    extra = f"  (measure {res['longest']}ch — long)"
                print(f"{name:<12} {label:<8} {flag}  {res['words']}w{extra}")
                for c in res["lowcontrast"][:3]:
                    print(f"    contrast: {c}")
                for t in res["tiny"][:3]:
                    print(f"    tiny: {t}")
        browser.close()

    Path(args.shots, "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nshots in {shots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
