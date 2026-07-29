#!/usr/bin/env python3
"""Measure how a rendered theme LOOKS, not merely that it renders.

`run.py` grades plumbing: does the theme activate, is it a block theme,
does the front page return 200. A theme can pass every one of those
twelve checks and look like unstyled HTML — which means the benchmark
could not tell "basic working" from "exquisite", and that distinction is
the entire product.

This measures the rendered document instead. Not with an LLM judge — the
whole discipline here is that the number cannot come from a second
unvalidated model — but from things a browser can compute exactly:

    overflow-390      does anything escape a 390px viewport
    contrast-body     WCAG ratio of body text against its real background
    measure           line length in characters (45-85 is readable)
    type-scale        count of distinct font sizes (a scale, not a pile)
    font-chosen       is the body face a deliberate choice or Times
    focus-visible     does keyboard focus produce a visible ring
    tap-targets       are interactive targets >= 24px (WCAG 2.2 AA)
    spacing-scale     count of distinct margin/padding values
    fluid-type        is any font-size fluid rather than fixed
    states            hover/focus declared for links

Screenshots are written alongside, because some of "exquisite" is not
reducible to a number and a person should look at it.

Usage:
    python packs/theme-harness/look.py                 # all candidates
    python packs/theme-harness/look.py qwen3-6-27b-mounted
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
THEMES = HERE / "themes"
SHOTS = HERE / "shots"
SITE = "http://localhost:8099"
WIDTHS = [(390, "mobile"), (768, "tablet"), (1440, "desktop")]

# Measured in the page. Kept in one string so the browser does the work
# and Python only reports it.
PROBE = r"""
() => {
  const px = v => parseFloat(v) || 0;

  // Effective background: walk up until something is not transparent.
  const bgOf = el => {
    for (let n = el; n && n !== document.documentElement.parentNode; n = n.parentElement) {
      const c = getComputedStyle(n).backgroundColor;
      const m = c.match(/rgba?\(([^)]+)\)/);
      if (!m) continue;
      const p = m[1].split(',').map(s => parseFloat(s));
      if (p.length < 4 || p[3] > 0.05) return [p[0], p[1], p[2]];
    }
    return [255, 255, 255];
  };
  const rgb = s => {
    const m = s.match(/rgba?\(([^)]+)\)/);
    if (!m) return [0, 0, 0];
    const p = m[1].split(',').map(x => parseFloat(x));
    return [p[0], p[1], p[2]];
  };
  const lum = ([r, g, b]) => {
    const f = c => { c /= 255; return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
  };
  const ratio = (a, b) => {
    const [l1, l2] = [lum(a), lum(b)].sort((x, y) => y - x);
    return (l1 + 0.05) / (l2 + 0.05);
  };

  // The longest run of body copy is the thing to judge type on.
  const paras = [...document.querySelectorAll('p, li')]
    .filter(e => (e.textContent || '').trim().length > 40 && e.offsetParent !== null);
  const body = paras.sort((a, b) =>
    (b.textContent.length) - (a.textContent.length))[0] || document.body;
  const bs = getComputedStyle(body);

  // Line length in characters: width / average glyph advance, estimated
  // by measuring a known string in the same font.
  const probe = document.createElement('span');
  probe.style.cssText = 'position:absolute;visibility:hidden;white-space:pre;';
  probe.style.font = bs.font || `${bs.fontSize} ${bs.fontFamily}`;
  probe.textContent = 'abcdefghijklmnopqrstuvwxyz';
  document.body.appendChild(probe);
  const glyph = probe.getBoundingClientRect().width / 26;
  probe.remove();
  const contentW = body.getBoundingClientRect().width
    - px(bs.paddingLeft) - px(bs.paddingRight);

  const visible = [...document.querySelectorAll('body *')]
    .filter(e => e.offsetParent !== null && e.getBoundingClientRect().height > 0);

  const sizes = new Set();
  const spaces = new Set();
  let fluid = false;
  for (const e of visible.slice(0, 800)) {
    const s = getComputedStyle(e);
    if ((e.textContent || '').trim()) sizes.add(Math.round(px(s.fontSize) * 10) / 10);
    for (const p of [s.marginTop, s.marginBottom, s.paddingTop, s.paddingBottom]) {
      const v = Math.round(px(p) * 10) / 10;
      if (v > 0) spaces.add(v);
    }
  }
  // Fluid type is visible in the cascade, not the computed value.
  for (const sheet of document.styleSheets) {
    try {
      for (const rule of sheet.cssRules || []) {
        const t = rule.cssText || '';
        // WordPress emits fluid sizes as clamp() inside the PRESET
        // CUSTOM PROPERTIES (--wp--preset--font-size--large: clamp(...)),
        // not as a font-size declaration. Looking only for
        // "font-size: clamp(" reported fluid_type false on a theme whose
        // every preset was fluid.
        if (/(font-size[^;]*|--wp--preset--font-size--[a-z0-9-]+\s*):[^;]*clamp\(/.test(t)) {
          fluid = true; break;
        }
      }
    } catch (e) { /* cross-origin sheet */ }
    if (fluid) break;
  }

  // Visually-hidden controls are 1px BY DESIGN — skip links and
  // screen-reader text use the clip technique. Counting them as
  // undersized tap targets failed a theme for implementing
  // accessibility correctly, which is the opposite of the check's point.
  const hidden = e => {
    const s = getComputedStyle(e);
    if (/(^|\s)screen-reader-text(\s|$)|skip-link/.test(e.className || '')) return true;
    if (s.clip !== 'auto' && s.clip !== '' ) return true;
    if (s.clipPath && s.clipPath !== 'none') return true;
    const r = e.getBoundingClientRect();
    return r.width <= 2 && r.height <= 2;
  };
  const targets = [...document.querySelectorAll('a, button, input, select')]
    .filter(e => e.offsetParent !== null && !hidden(e))
    .map(e => { const r = e.getBoundingClientRect(); return Math.min(r.width, r.height); })
    .filter(v => v > 0);

  let stateRules = 0;
  for (const sheet of document.styleSheets) {
    try {
      for (const rule of sheet.cssRules || []) {
        if (/:hover|:focus-visible|:focus\b/.test(rule.selectorText || '')) stateRules++;
      }
    } catch (e) { /* ignore */ }
  }

  return {
    overflow: Math.max(0, document.documentElement.scrollWidth - window.innerWidth),
    contrast: Math.round(ratio(rgb(bs.color), bgOf(body)) * 100) / 100,
    measure: glyph > 0 ? Math.round(contentW / glyph) : 0,
    font_family: bs.fontFamily,
    font_size: Math.round(px(bs.fontSize) * 10) / 10,
    line_height: bs.lineHeight,
    distinct_sizes: sizes.size,
    distinct_spaces: spaces.size,
    fluid_type: fluid,
    tap_min: targets.length ? Math.round(Math.min(...targets)) : 0,
    tap_under_24: targets.filter(v => v < 24).length,
    state_rules: stateRules,
    // WORST-CASE contrast, not the contrast of one paragraph. Measuring
    // only the longest run of body copy passed a theme whose footer put
    // near-black text on a near-black background — invisible, and the
    // page scored contrast-aa because the post excerpt elsewhere was
    // fine. A page is as readable as its least readable text.
    contrast_min: (() => {
      let worst = 99, where = '';
      for (const e of visible.slice(0, 600)) {
        const own = [...e.childNodes].some(n =>
          n.nodeType === 3 && (n.textContent || '').trim().length > 1);
        if (!own) continue;
        const st = getComputedStyle(e);
        if (parseFloat(st.opacity) < 0.5) continue;
        const r = ratio(rgb(st.color), bgOf(e));
        if (r < worst) { worst = r; where = e.tagName.toLowerCase() +
          '.' + (e.className || '').toString().split(' ')[0]; }
      }
      return { ratio: Math.round(worst * 100) / 100, el: where };
    })(),
    text_len: (document.body.innerText || '').trim().length,
    // Source code rendering as body text. A theme put PHP into a
    // template .html file, which is never executed: the literal `<?php`
    // left an href="" unterminated, the parser swallowed the rest of the
    // document into that attribute, and WordPress's injected skip-link
    // script rendered as visible prose. Zero PHP errors, HTTP 200, and
    // catastrophic to look at — invisible to all twelve plumbing checks
    // and to the first ten craft checks.
    leaked: (() => {
      const t = document.body.innerText || '';
      const tells = ['<?php', 'querySelector(', 'document.createElement',
                     'function()', '=>', 'addEventListener'];
      return tells.filter(s => t.includes(s));
    })(),
  };
}
"""

# Substring match, not equality. The first version compared the whole
# first family name against this list, so a page rendering in literal
# "Times New Roman" PASSED font-chosen — a lenient check in the very
# rubric written to catch leniency. The theme had declared a font preset
# and never applied it, which is exactly the defect the check exists for.
FALLBACK_FACES = ("times", "serif", "georgia", "system-ui", "-apple-system",
                  "blinkmacsystemfont", "sans-serif", "monospace")


def wp(*args: str) -> tuple[int, str]:
    cmd = ["docker", "compose", "exec", "-T", "cli", "wp", *args, "--path=/var/www/html"]
    env = {**os.environ, "MSYS_NO_PATHCONV": "1"}
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=180, env=env)
    return r.returncode, (r.stdout + r.stderr).strip()


def judge(m: dict) -> dict[str, bool]:
    """Deterministic proxies for craft. None of these needs an opinion."""
    first = (m.get("font_family") or "").lower().split(",")[0].strip(" \"'")
    return {
        "no-overflow": m["overflow"] == 0,
        "contrast-aa": m["contrast"] >= 4.5,
        # The page is as readable as its least readable text.
        "contrast-everywhere": (m.get("contrast_min") or {}).get("ratio", 0) >= 4.5,
        # Pure #000 on #fff is 21:1 and reads as unconsidered glare. The
        # first ceiling here was 17, which failed a warm near-black on a
        # warm off-white at 17.28 — good typography, rejected by an
        # arbitrary bound. Only the true extreme is crude.
        "contrast-not-crude": 4.5 <= m["contrast"] <= 19.5,
        "measure-readable": 45 <= m["measure"] <= 85,
        "type-scale": 4 <= m["distinct_sizes"] <= 9,
        "font-chosen": bool(first) and not any(f in first for f in FALLBACK_FACES),
        # line-height: normal means nobody set it. Body copy needs ~1.5-1.7.
        "line-height-set": m["line_height"] not in ("normal", "", None),
        "spacing-scale": 2 <= m["distinct_spaces"] <= 12,
        "fluid-type": bool(m["fluid_type"]),
        "tap-targets": m["tap_under_24"] == 0,
        "has-states": m["state_rules"] >= 2,
        "has-content": m["text_len"] > 120,
        # No source code rendering as prose.
        "no-leaked-source": not m.get("leaked"),
    }


def main() -> int:
    wanted = sys.argv[1:]
    cands = sorted(p.name for p in THEMES.iterdir() if p.is_dir()) if THEMES.exists() else []
    if wanted:
        cands = [c for c in cands if c in wanted]
    if not cands:
        print("no candidate themes in themes/ — run run.py first")
        return 1

    SHOTS.mkdir(exist_ok=True)
    out = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for slug in cands:
            code, msg = wp("--skip-themes", "theme", "activate", f"harness/{slug}")
            # An already-active theme makes WP-CLI print a warning, not
            # "Success" — which rejected a theme that was working fine.
            ok = code == 0 and ("Success" in msg or "already active" in msg)
            if not ok:
                print(f"{slug:<26} cannot activate — skipped")
                continue
            per_width = {}
            for width, label in WIDTHS:
                page = browser.new_page(viewport={"width": width, "height": 900},
                                        device_scale_factor=2 if width == 390 else 1)
                page.goto(SITE + "/", wait_until="networkidle", timeout=60000)
                page.wait_for_timeout(400)
                m = page.evaluate(PROBE)
                page.screenshot(path=str(SHOTS / f"{slug}.{label}.png"), full_page=(width != 390))
                per_width[label] = m
                page.close()
            d = per_width["desktop"]
            verdict = judge(d)
            verdict["no-overflow"] = per_width["mobile"]["overflow"] == 0  # judged where it breaks
            score = sum(verdict.values())
            out.append({"theme": slug, "score": score, "checks": verdict,
                        "desktop": d, "mobile": per_width["mobile"]})
            print(f"{slug:<26} {score:>2}/{len(verdict)}   "
                  f"contrast {d['contrast']} (min {d.get('contrast_min',{}).get('ratio','?')} "f"on {d.get('contrast_min',{}).get('el','?')})  measure {d['measure']}ch  "
                  f"sizes {d['distinct_sizes']}  overflow(390) {per_width['mobile']['overflow']}px")
            print(f"{'':<26} failed: "
                  f"{', '.join(k for k, v in verdict.items() if not v) or '—'}")
        browser.close()

    wp("--skip-themes", "theme", "activate", "twentytwentyfive")
    (HERE / "look.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nshots in {SHOTS}\nwrote {HERE / 'look.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
