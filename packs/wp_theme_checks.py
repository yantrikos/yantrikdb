#!/usr/bin/env python3
"""Deterministic checks over a generated theme.json — the craft grader.

This is the piece that makes compiling craft measurable at all. Every
check below is a structural predicate over the PARSED artifact, taken
from one named rule of the wordpress-theme constitution. No string
matching against prose, no LLM judge: the same functions gate a
teacher's artifact into training and grade a student's artifact at eval,
so "compliant" means the same thing on both sides of the experiment.

The checks encode the constitution's reasons, not just its keywords —
`styles_apply_presets` exists because `settings` without `styles` is the
defect that renders in Times New Roman and reports no error, and a
checker that only looked for the word "fontFamily" would pass it.

Every check returns (ok, detail). run_checks returns an ordered dict so
a report can show WHICH rule fails, per artifact, per arm — the
compliance table is the experiment's entire result.
"""

from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict | None:
    """The artifact out of a model answer: fenced, bare, or embedded."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fence.group(1)] if fence else []
    brace = text.find("{")
    if brace != -1:
        depth = 0
        for i, ch in enumerate(text[brace:], brace):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[brace:i + 1])
                    break
    for c in candidates:
        try:
            d = json.loads(c)
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            continue
    return None


def _get(d: dict, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _num(v) -> float | None:
    """A number out of 1.6, "1.6", "1.6rem" — unitless where possible."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.match(r"^([0-9.]+)", v.strip())
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


_EXTREME = {"#000", "#000000", "#fff", "#ffffff", "black", "white"}


def _is_extreme(v) -> bool:
    return isinstance(v, str) and v.strip().lower() in _EXTREME


# Each entry: (check id, constitution rule it enforces, function).
# The rule names match constitution.md headings so a failure points at
# the exact record a human would reread.

def chk_valid_version(t):
    v = t.get("version")
    return (v in (2, 3), f"version={v!r}")


def chk_fluid_typography(t):
    ok = _get(t, "settings", "typography", "fluid") is True
    return (ok, "settings.typography.fluid is not true" if not ok else "")


def chk_type_scale(t):
    sizes = _get(t, "settings", "typography", "fontSizes", default=[])
    n = len(sizes) if isinstance(sizes, list) else 0
    return (n >= 5, f"{n} fontSizes (scale wants 5-7 steps)")


def chk_styles_apply_presets(t):
    """The Times New Roman defect: declared but never applied."""
    missing = [k for k, v in {
        "styles.typography.fontFamily": _get(t, "styles", "typography", "fontFamily"),
        "styles.typography.fontSize": _get(t, "styles", "typography", "fontSize"),
        "styles.color.background": _get(t, "styles", "color", "background"),
        "styles.color.text": _get(t, "styles", "color", "text"),
    }.items() if not v]
    return (not missing, "unapplied: " + ", ".join(missing) if missing else "")


def chk_root_line_height(t):
    lh = _num(_get(t, "styles", "typography", "lineHeight"))
    return (lh is not None and 1.4 <= lh <= 1.8,
            f"root lineHeight={lh} (want 1.4-1.8, unset computes to ~1.2)")


def chk_display_line_height(t):
    for el in ("h1", "h2"):
        lh = _num(_get(t, "styles", "elements", el, "typography", "lineHeight"))
        if lh is not None and 1.0 <= lh < 1.35:
            return (True, "")
    return (False, "no h1/h2 lineHeight in 1.0-1.35 (display runs tight)")


def chk_link_color(t):
    ok = bool(_get(t, "styles", "elements", "link", "color", "text"))
    return (ok, "styles.elements.link.color.text missing" if not ok else "")


def chk_block_gap(t):
    ok = _get(t, "styles", "spacing", "blockGap") is not None
    return (ok, "styles.spacing.blockGap unset" if not ok else "")


def chk_content_size_is_measure(t):
    cs = _get(t, "settings", "layout", "contentSize")
    if not isinstance(cs, str):
        return (False, "settings.layout.contentSize missing")
    v = _num(cs)
    if cs.strip().endswith(("rem", "ch", "em")):
        ok = v is not None and 30 <= v <= 50 if cs.strip().endswith(("rem", "em")) \
            else v is not None and 45 <= v <= 80
        return (ok, f"contentSize={cs} (a measure: 45-75 characters per line)")
    if cs.strip().endswith("px"):
        return (v is not None and 500 <= v <= 780,
                f"contentSize={cs} (px accepted only 500-780; 1200px is not a measure)")
    return (False, f"contentSize={cs} (unit?)")


def chk_wide_size(t):
    ok = bool(_get(t, "settings", "layout", "wideSize"))
    return (ok, "settings.layout.wideSize missing" if not ok else "")


def chk_palette_roles(t):
    pal = _get(t, "settings", "color", "palette", default=[])
    slugs = {p.get("slug", "").lower() for p in pal if isinstance(p, dict)}
    roles = {"base", "contrast", "primary", "secondary", "accent"}
    named = len(slugs & roles)
    return (len(pal) >= 4 and named >= 3,
            f"palette={len(pal)} colours, {named} role-named slugs "
            f"(want >=4 with base/contrast/primary...)")


def chk_no_extreme_colours(t):
    bad = []
    for path in (("styles", "color", "background"), ("styles", "color", "text")):
        v = _get(t, *path)
        if _is_extreme(v):
            bad.append(f"{'.'.join(path)}={v}")
    for p in _get(t, "settings", "color", "palette", default=[]) or []:
        if isinstance(p, dict) and p.get("slug", "").lower() in ("base", "contrast") \
                and _is_extreme(p.get("color")):
            bad.append(f"palette.{p.get('slug')}={p.get('color')}")
    return (not bad, "pure extremes (21:1 is glare): " + ", ".join(bad) if bad else "")


def chk_root_padding_aware(t):
    pad = _get(t, "styles", "spacing", "padding")
    aware = _get(t, "settings", "useRootPaddingAwareAlignments")
    if pad:
        return (aware is True,
                "root padding declared but useRootPaddingAwareAlignments is not true "
                "(.alignfull will not reach the edges)")
    return (True, "")


def chk_spacing_scale(t):
    sc = _get(t, "settings", "spacing", "spacingScale")
    sizes = _get(t, "settings", "spacing", "spacingSizes")
    ok = bool(sc) or (isinstance(sizes, list) and len(sizes) >= 4)
    return (ok, "no spacingScale/spacingSizes (spacing comes from the scale)" if not ok else "")


CHECKS = [
    ("valid-version", "style.css opens with a valid header", chk_valid_version),
    ("fluid-typography", "One type ratio, whole scale fluid", chk_fluid_typography),
    ("type-scale", "One type ratio, whole scale fluid", chk_type_scale),
    ("styles-apply-presets", "Declare presets, then APPLY them under styles", chk_styles_apply_presets),
    ("root-line-height", "line-height is set explicitly", chk_root_line_height),
    ("display-line-height", "line-height is set explicitly", chk_display_line_height),
    ("link-color", "Declare presets, then APPLY them under styles", chk_link_color),
    ("block-gap", "Spacing comes from the scale; blockGap is set once", chk_block_gap),
    ("content-size-measure", "contentSize is a measure", chk_content_size_is_measure),
    ("wide-size", "contentSize is a measure", chk_wide_size),
    ("palette-roles", "Colour is chosen, not defaulted to the extremes", chk_palette_roles),
    ("no-extreme-colours", "Colour is chosen, not defaulted to the extremes", chk_no_extreme_colours),
    ("root-padding-aware", "Root padding uses root-padding-aware alignments", chk_root_padding_aware),
    ("spacing-scale", "Spacing comes from the scale; blockGap is set once", chk_spacing_scale),
]


def run_checks(theme: dict) -> dict[str, tuple[bool, str]]:
    out = {}
    for cid, _rule, fn in CHECKS:
        try:
            out[cid] = fn(theme)
        except Exception as e:                                 # noqa: BLE001
            out[cid] = (False, f"checker error: {e}")
    return out


def grade_text(text: str) -> tuple[int, int, dict[str, tuple[bool, str]] | None]:
    """(passed, total, per-check) for a raw model answer. A theme that
    does not parse scores zero — an invalid theme.json does not activate,
    so unparseable IS the score, not a measurement failure."""
    theme = extract_json(text)
    if theme is None:
        return 0, len(CHECKS), None
    res = run_checks(theme)
    return sum(1 for ok, _ in res.values() if ok), len(CHECKS), res


# ------------------------------------------------------------------ briefs

SITE_TYPES = [
    "an independent bakery", "a documentary photographer's portfolio",
    "a small law firm", "an indie game studio", "a travel journal",
    "a SaaS product landing site", "a literary magazine",
    "a neighbourhood restaurant", "a ceramics studio shop",
    "a podcast about city history", "a climbing gym",
    "a children's bookshop", "an architecture practice",
    "a coffee roastery", "a personal engineering blog",
    "a yoga studio", "a vinyl record store", "a florist",
]
MOODS = [
    "warm and editorial", "minimal and airy", "bold and high-energy",
    "quiet and bookish", "playful but disciplined", "dark and cinematic",
]
TYPE_VIBES = [
    "a serif display over a humanist sans body",
    "a single geometric sans throughout",
    "a slab display with a neutral grotesque body",
    "an old-style serif for everything",
]
PALETTES = [
    "earthy neutrals with one deep accent", "cool blues on warm paper",
    "monochrome with a single vivid accent", "sun-bleached pastels",
    "forest greens and cream", "ink and terracotta",
]


def briefs(seed_combos: list[tuple[int, int, int, int]]) -> list[str]:
    return [
        f"Design theme.json for a WordPress block theme for {SITE_TYPES[a]}. "
        f"The mood is {MOODS[b]}. Typography: {TYPE_VIBES[c]}. "
        f"Palette: {PALETTES[d]}."
        for a, b, c, d in seed_combos
    ]


def train_briefs() -> list[str]:
    """72 distinct combos over the first 12 site types.

    Walked as j = 25k over the full 12x6x4x6 product. 25 is coprime with
    1728, so the walk visits 72 distinct cells spread across every
    dimension. The first version used linear maps of i whose periods all
    divide 12 — 72 intended combos collapsed to 12 unique briefs, which
    a dataset line count would have reported as a small dataset rather
    than a broken generator.
    """
    combos = []
    for k in range(72):
        j = (25 * k) % (12 * 6 * 4 * 6)
        a, rest = divmod(j, 6 * 4 * 6)
        b, rest = divmod(rest, 4 * 6)
        c, d = divmod(rest, 6)
        combos.append((a, b, c, d))
    assert len(set(combos)) == 72
    return briefs(combos)


def holdout_briefs() -> list[str]:
    """Sealed: the LAST 6 site types never appear in training, crossed
    with mood/type/palette combos training never used together."""
    combos = [(12 + i % 6, (i * 3 + 1) % 6, (i + 2) % 4, (i * 2 + 3) % 6)
              for i in range(12)]
    return briefs(combos)


if __name__ == "__main__":
    import sys
    text = sys.stdin.read()
    p, n, res = grade_text(text)
    print(f"{p}/{n}")
    if res:
        for cid, (ok, detail) in res.items():
            print(f"  {'PASS' if ok else 'FAIL'} {cid:<24} {detail}")
