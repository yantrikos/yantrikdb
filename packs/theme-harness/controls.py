#!/usr/bin/env python3
"""The three-control experiment: how much of the score is the model?

A capability kit bundles pack-authored taste, a schema-assembling harness,
and a model making bounded choices. The rendered-craft score measures the
SYSTEM — so before crediting any of it to the model, replace the model's
choices with policies that contain no model and measure what remains.

    model     the 4B's actual field choices (palette, sizes, layout)
    random    uniform-random schema-satisfying values
    trivial   the naive defaults a lazy implementation would pick

Everything else — template, parts, style.css, functions.php, the
assemble_theme_json() schema, WordPress, the rubric — is held constant.
The template comes from the model's own iterated build, so this isolates
exactly one contribution: the design tokens, which is where we have been
claiming the model showed taste.

Reading the result:
  model >> trivial, random   the model's choices earn their score
  trivial ≈ model            the kit is a design system; the model is
                             decoration, and the listing must say so
  random ≈ model             the schema itself carries the score; the
                             word "generative" would be dishonest

Random has variance by construction, so it runs with several seeds and
reports each. This is the generation-kit analogue of the attach-harm
control: the number nobody else will publish, which is why we do.

Usage:
    python packs/theme-harness/controls.py --source qwen3-5-4b-iterated
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_iter import assemble_theme_json  # noqa: E402
from run import THEMES, reset_to_core, wp  # noqa: E402

# ── the two no-model policies ────────────────────────────────────────

# "Trivial" is ill-defined over continuous domains, so it is pinned to
# what a lazy implementation would actually emit: pure black on pure
# white, a web-safe blue, one font size everywhere, generic families,
# pixel widths. Every value satisfies the schema; none was chosen.
TRIVIAL = {
    "palette": [
        {"slug": "base",      "name": "Base",      "color": "#ffffff"},
        {"slug": "contrast",  "name": "Contrast",  "color": "#000000"},
        {"slug": "primary",   "name": "Primary",   "color": "#0000ff"},
        {"slug": "secondary", "name": "Secondary", "color": "#808080"},
        {"slug": "tint",      "name": "Tint",      "color": "#eeeeee"},
    ],
    "fontSizes": [
        {"slug": s, "name": s, "size": "1rem"}
        for s in ("small", "medium", "large", "x-large", "xx-large", "huge")
    ],
    "fontFamilies": [
        {"slug": "display", "name": "Display", "fontFamily": "serif"},
        {"slug": "text",    "name": "Text",    "fontFamily": "sans-serif"},
    ],
    "layout": {"contentSize": "600px", "wideSize": "1000px"},
}

# Random stays schema-VALID — the control measures the value of choosing,
# not the value of validity, which the schema already guarantees.
FAMILY_POOL = [
    "Georgia, serif", "Charter, Georgia, serif", "Verdana, sans-serif",
    "\"Courier New\", monospace", "\"Comic Sans MS\", cursive, sans-serif",
    "\"Times New Roman\", serif", "Arial, sans-serif",
    "\"Trebuchet MS\", sans-serif", "Palatino, serif", "Garamond, serif",
]


def random_fields(rng: random.Random) -> dict:
    hexc = lambda: f"#{rng.randrange(0x1000000):06x}"  # noqa: E731
    return {
        "palette": [
            {"slug": s, "name": s.title(), "color": hexc()}
            for s in ("base", "contrast", "primary", "secondary", "tint")
        ],
        "fontSizes": [
            {"slug": s, "name": s, "size": f"{rng.uniform(0.5, 4.0):.3f}rem"}
            for s in ("small", "medium", "large", "x-large", "xx-large", "huge")
        ],
        "fontFamilies": [
            {"slug": "display", "name": "Display",
             "fontFamily": rng.choice(FAMILY_POOL)},
            {"slug": "text", "name": "Text",
             "fontFamily": rng.choice(FAMILY_POOL)},
        ],
        "layout": {
            "contentSize": f"{rng.uniform(10, 100):.0f}rem",
            "wideSize": f"{rng.uniform(10, 100):.0f}rem",
        },
    }


def make_variant(source: str, name: str, theme_json: str) -> str:
    src, dst = THEMES / source, THEMES / name
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    (dst / "theme.json").write_text(theme_json, encoding="utf-8")
    # A distinct Theme Name, or WordPress refuses the duplicate.
    style = dst / "style.css"
    if style.exists():
        css = style.read_text(encoding="utf-8")
        css = css.replace("Theme Name:", f"Theme Name: ctl-{name} —", 1) \
            if "Theme Name:" in css else f"/*\nTheme Name: ctl-{name}\n*/\n" + css
        style.write_text(css, encoding="utf-8")
    return name


def measure(slugs: list[str]) -> dict[str, dict]:
    """One look.py invocation for all variants; parse its JSON output."""
    subprocess.run(
        [sys.executable, str(HERE / "look.py"), *slugs],
        cwd=HERE.parent.parent, capture_output=True, text=True, timeout=1800,
        env={**__import__("os").environ, "MSYS_NO_PATHCONV": "1"})
    out = json.loads((HERE / "look.json").read_text(encoding="utf-8"))
    return {e["theme"]: e for e in out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="qwen3-5-4b-iterated",
                    help="model-built theme to hold constant")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    src = THEMES / args.source
    if not (src / "theme.json").exists():
        print(f"{args.source} has no theme.json — run build_iter.py first")
        return 1

    # The model variant is the source itself, untouched.
    variants = [args.source]

    variants.append(make_variant(
        args.source, "ctl-trivial", assemble_theme_json(TRIVIAL)))

    for seed in range(args.seeds):
        rng = random.Random(seed)
        variants.append(make_variant(
            args.source, f"ctl-random-{seed}",
            assemble_theme_json(random_fields(rng))))

    print(f"measuring {len(variants)} variants "
          f"(template and harness held constant)…\n")
    results = measure(variants)

    def row(slug: str) -> tuple[int, int, str]:
        e = results.get(slug)
        if not e:
            return 0, 0, "no result"
        d = e["desktop"]
        gate = f"  HARD-FAIL: {','.join(e['hard_fail'])}" if e.get("hard_fail") else ""
        return (e["score"], len(e["checks"]),
                f"contrast {d['contrast']:>5}  measure {d['measure']:>3}ch  "
                f"sizes {d['distinct_sizes']}{gate}")

    print(f"{'policy':<16} {'craft':>7}   detail")
    print("-" * 64)
    ms, mt, md = row(args.source)
    print(f"{'model':<16} {ms:>3}/{mt:<3}   {md}")
    ts, tt, td = row("ctl-trivial")
    print(f"{'trivial':<16} {ts:>3}/{tt:<3}   {td}")
    rand_scores = []
    for seed in range(args.seeds):
        rs, rt, rd = row(f"ctl-random-{seed}")
        rand_scores.append(rs)
        print(f"{'random-' + str(seed):<16} {rs:>3}/{rt:<3}   {rd}")
    print("-" * 64)

    rand_scores.sort()
    med = rand_scores[len(rand_scores) // 2] if rand_scores else 0
    print(f"\nmodel {ms}  |  trivial {ts}  |  random median {med} "
          f"(range {rand_scores[0]}-{rand_scores[-1]})")
    print(f"model - trivial = {ms - ts:+d}   model - random(med) = {ms - med:+d}")
    print("\nNote: our measured single-run spread on the FULL pipeline was "
          "3 points; here the template is held constant, so differences are "
          "attributable to the tokens — but treat |diff| <= 1 as noise.")

    (HERE / "controls.json").write_text(json.dumps({
        "source": args.source,
        "model": ms, "trivial": ts,
        "random": rand_scores, "random_median": med,
        "results": {k: {"score": v["score"], "checks": v["checks"]}
                    for k, v in results.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {HERE / 'controls.json'}")

    reset_to_core()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
