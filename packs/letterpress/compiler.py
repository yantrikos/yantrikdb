#!/usr/bin/env python3
"""Compile a six-operation script into one self-contained HTML page.

The model never writes HTML or CSS. It emits lines like

    THEME  family=editorial  accent=#b8442f  mode=light  density=roomy
    SECTION  id=s1  kind=hero  layout=split  tone=quiet
    TEXT     sec=s1  slot=eyebrow  text=Independent studio
    TEXT     sec=s1  slot=title    text=We build quiet software
    ACTION   sec=s1  slot=primary  label=See the work  href=#work

and everything that makes a page look designed — type scale, rhythm,
grid, colour derivation, contrast, responsive collapse — happens here,
deterministically. That split is the whole thesis: a 4B is good at
choosing *what* a section says and bad at closing a 2000-line document,
so it should never be asked to write one.

Two rules this file exists to enforce:

- **No fallback is silent.** Every substitution is recorded and
  reported. A compiler that quietly rescues a malformed script produces
  a beautiful page while the model failed completely, and we would be
  measuring the compiler and calling it a model result.
- **No external requests.** Fonts are system stacks, ornament is drawn
  as inline SVG. A page that needs a CDN is not self-contained and
  cannot be verified offline.
"""

from __future__ import annotations

import argparse
import colorsys
import html
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

OPS = {"SITE", "THEME", "SECTION", "TEXT", "ACTION", "MEDIA", "ITEM"}

# ── Design system ────────────────────────────────────────────────────
# Deliberately NOT the house style of generated pages: no cream-and-
# terracotta, no purple gradient, no Inter. Each family commits to a
# different typographic world, and all three use system stacks because
# the artifact must render offline with no network.

FAMILIES = {
    "editorial": {
        "display": '"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif',
        "body": 'Georgia,"Times New Roman",serif',
        "mono": 'ui-monospace,"SF Mono",Menlo,monospace',
        "scale": 1.333,      # perfect fourth — wide steps, magazine feel
        "measure": "34rem",
        "radius": "2px",
        "tracking_display": "-0.011em",
        "tracking_eyebrow": "0.16em",
        "weight_display": "600",
        "rule": "1px solid var(--line)",
    },
    "studio": {
        "display": 'system-ui,-apple-system,"Segoe UI",Roboto,sans-serif',
        "body": 'system-ui,-apple-system,"Segoe UI",Roboto,sans-serif',
        "mono": 'ui-monospace,"SF Mono",Menlo,monospace',
        "scale": 1.5,        # perfect fifth — poster-like jumps
        "measure": "32rem",
        "radius": "14px",
        "tracking_display": "-0.032em",
        "tracking_eyebrow": "0.1em",
        "weight_display": "800",
        "rule": "1px solid var(--line)",
    },
    "technical": {
        "display": 'ui-monospace,"SF Mono",Menlo,Consolas,monospace',
        "body": 'system-ui,-apple-system,"Segoe UI",Roboto,sans-serif',
        "mono": 'ui-monospace,"SF Mono",Menlo,Consolas,monospace',
        "scale": 1.25,       # major third — dense, technical
        "measure": "36rem",
        "radius": "0px",
        "tracking_display": "-0.005em",
        "tracking_eyebrow": "0.18em",
        "weight_display": "600",
        "rule": "1px dashed var(--line)",
    },
}

DENSITY = {"tight": 0.72, "normal": 1.0, "roomy": 1.42}

# Annotation size inside a motif, in viewBox user units.
#
# Text in a scaled viewBox has no fixed pixel size: these drawings render
# at roughly 1.26x on desktop and 0.74x at 390px wide, so the labels must
# be sized against the SMALLEST scale they will ever appear at, not the
# one you happen to be looking at. At 11 units they rendered 8.1px on
# mobile — unreadable — and the size gate could not see it, because
# getComputedStyle reports user units and dutifully said "11px" at every
# viewport. 18 units clears the 12px floor at 0.74x while staying
# proportionate to artwork 420 units wide.
ANNOT = 18


def motif_elevation(pal: str, ink: str, line: str, surface: str = "#ffffff") -> str:
    """A building in elevation — hatched mass, roof, lit windows, and a
    dimension line, which is the detail that makes it read as a drawing
    rather than an icon."""
    return f"""
<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse"
 patternTransform="rotate(38)"><line x1="0" y1="0" x2="0" y2="6"
 stroke="{pal}" stroke-width="1" opacity=".3"/></pattern></defs>
<g fill="none" stroke="{ink}" stroke-width="1.4" class="draw">
<line x1="20" y1="452" x2="400" y2="452"/>
<rect x="74" y="214" width="188" height="238" fill="url(#hatch)"/>
<path d="M60 214 L168 138 L276 214"/>
<rect x="212" y="150" width="18" height="42"/>
<rect x="262" y="300" width="104" height="152"/>
<path d="M254 300 L314 262 L374 300"/>
<rect x="160" y="252" width="38" height="52"/>
<rect x="104" y="336" width="38" height="52"/>
<rect x="292" y="340" width="44" height="60"/>
<rect x="212" y="368" width="30" height="84"/>
</g>
<rect x="104" y="252" width="38" height="52" fill="{pal}" opacity=".85" class="lit"/>
<rect x="160" y="336" width="38" height="52" fill="{pal}" opacity=".85" class="lit"/>
<g stroke="{line}" stroke-width="1" class="draw">
<line x1="74" y1="486" x2="262" y2="486"/><line x1="74" y1="480" x2="74" y2="492"/>
<line x1="262" y1="480" x2="262" y2="492"/></g>
<text x="168" y="478" text-anchor="middle" fill="{line}"
 font-family="ui-monospace,monospace" font-size="{ANNOT}">9 400</text>"""


def motif_topography(pal: str, ink: str, line: str, surface: str = "#ffffff") -> str:
    """Nested contour rings — reads as land, maps, terrain."""
    rings = []
    for i in range(9):
        k = i * 22
        rings.append(
            f'<path d="M{40 + k * .5} {430 - k * .8} '
            f'C{120 + k} {300 - k},{200 - k * .4} {250 + k * .3},'
            f'{380 - k * .6} {380 - k * .9}" />')
    return (f'<g fill="none" stroke="{ink}" stroke-width="1.1" opacity=".55" '
            f'class="draw">{"".join(rings)}</g>'
            f'<circle cx="232" cy="236" r="7" fill="{pal}" class="lit"/>')


def motif_schematic(pal: str, ink: str, line: str, surface: str = "#ffffff") -> str:
    """A wiring/flow schematic — nodes on a grid joined by orthogonal runs."""
    parts = [f'<g stroke="{line}" stroke-width=".6" opacity=".5">']
    for x in range(60, 401, 40):
        parts.append(f'<line x1="{x}" y1="60" x2="{x}" y2="460"/>')
    for y in range(60, 461, 40):
        parts.append(f'<line x1="60" y1="{y}" x2="400" y2="{y}"/>')
    parts.append("</g>")
    parts.append(f'<g fill="none" stroke="{ink}" stroke-width="1.6" class="draw">'
                 f'<path d="M100 140 H260 V260 H340"/>'
                 f'<path d="M100 380 H180 V260"/>'
                 f'<path d="M340 140 V200 H260"/></g>')
    for cx, cy in ((100, 140), (260, 260), (340, 140), (100, 380), (340, 260)):
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="{pal}" class="lit"/>')
    return "".join(parts)


def motif_dial(pal: str, ink: str, line: str, surface: str = "#ffffff") -> str:
    """Concentric dial with index marks — instruments, precision, product."""
    marks = []
    for i in range(60):
        import math
        a = math.radians(i * 6 - 90)
        long_ = i % 5 == 0
        r1, r2 = (150, 176) if long_ else (163, 176)
        marks.append(
            f'<line x1="{210 + r1 * math.cos(a):.1f}" y1="{258 + r1 * math.sin(a):.1f}"'
            f' x2="{210 + r2 * math.cos(a):.1f}" y2="{258 + r2 * math.sin(a):.1f}"'
            f' stroke="{ink}" stroke-width="{1.6 if long_ else .8}"/>')
    return (f'<g class="draw" fill="none" stroke="{ink}" stroke-width="1.2">'
            f'<circle cx="210" cy="258" r="188"/><circle cx="210" cy="258" r="140"/>'
            f'</g>{"".join(marks)}'
            f'<line x1="210" y1="258" x2="210" y2="136" stroke="{pal}"'
            f' stroke-width="3" class="lit"/>'
            f'<line x1="210" y1="258" x2="288" y2="300" stroke="{ink}"'
            f' stroke-width="2" class="lit"/>'
            f'<circle cx="210" cy="258" r="6" fill="{pal}" class="lit"/>')


def motif_specimen(pal: str, ink: str, line: str, surface: str = "#ffffff") -> str:
    """A type specimen — the letterform itself as the artwork."""
    return (f'<text x="210" y="368" text-anchor="middle" fill="{ink}"'
            f' font-family="Georgia,serif" font-size="330"'
            f' font-weight="600" class="lit">Aa</text>'
            f'<g stroke="{line}" stroke-width="1" class="draw">'
            f'<line x1="40" y1="368" x2="380" y2="368"/>'
            f'<line x1="40" y1="176" x2="380" y2="176"/>'
            f'<line x1="40" y1="252" x2="380" y2="252"/></g>'
            # "CAP HEIGHT" and "X-HEIGHT" at 18 units run about 100 units
            # wide and collided with the letterform's left stem. A
            # specimen marks these short anyway.
            f'<text x="44" y="168" fill="{pal}" font-family="ui-monospace,monospace"'
            f' font-size="{ANNOT}">CAP</text>'
            f'<text x="44" y="244" fill="{pal}" font-family="ui-monospace,monospace"'
            f' font-size="{ANNOT}">X-HT</text>')


def motif_strata(pal: str, ink: str, line: str, surface: str = "#ffffff") -> str:
    """Layered horizontal bands, cut through — time, record, sequence.

    For anything with a span: an archive running 1957 to 1994, a back
    catalogue, a history. The five original motifs had nothing for a
    subject whose defining property is DURATION, so an archive got a
    house.
    """
    bands, y = [], 96
    for i, h in enumerate((34, 18, 46, 12, 28, 58, 22, 38, 16, 44)):
        fill = pal if i in (2, 5) else "none"
        # class="lit" QUOTED. Written unquoted as `class=lit/>`, the
        # HTML parser reads the value as "lit/", the rect never
        # self-closes, and every later band becomes its child — seven of
        # ten layers silently vanished. Only visible by rendering all
        # eight motifs side by side; on a real page it just looked like
        # a slightly plain drawing.
        cls = ' class="lit"' if fill != "none" else ""
        bands.append(
            f'<rect x="58" y="{y}" width="304" height="{h}" fill="{fill}"'
            f' stroke="{ink}" stroke-width="1"{cls}/>')
        y += h + 6
    # A core sample line down the middle, the way a section is marked.
    return ("".join(bands)
            + f'<g class="draw" stroke="{line}" stroke-width="1.2" fill="none">'
              f'<line x1="210" y1="82" x2="210" y2="{y + 8}"/>'
              f'<line x1="196" y1="82" x2="224" y2="82"/>'
              f'<line x1="196" y1="{y + 8}" x2="224" y2="{y + 8}"/></g>')


def motif_lattice(pal: str, ink: str, line: str, surface: str = "#ffffff") -> str:
    """An over-under weave — craft, material, things made by hand.

    Food, textile, joinery, print. A bakery is not a building, a gauge
    or a letterform, and it was getting a building.
    """
    # Wide bands with a narrow gap, not thin outlines on a wide pitch —
    # the first version used 26-wide bars on a 44 pitch and read as
    # graph paper. A weave has to be mostly band and barely gap, and it
    # has to actually go over and under: the verticals are drawn, then
    # the horizontals, then the verticals are RE-drawn where they pass
    # over, which is what makes the interlacing legible.
    n, band, gap = 5, 58, 10
    step = band + gap
    x0 = y0 = 210 - (n * step - gap) // 2 + 24
    # The two directions must be told apart or the occlusion is
    # invisible: filled with the same colour as the ground, a perfect
    # weave renders as plain graph paper, which is what the first
    # version did. Warp carries a faint accent wash, weft stays on the
    # surface colour, so the over-under alternation reads as a checker.
    verts = "".join(
        f'<rect x="{x0 + i * step}" y="{y0}" width="{band}"'
        f' height="{n * step - gap}" fill="{pal}" fill-opacity=".16"'
        f' stroke="{ink}" stroke-width="1.2"/>' for i in range(n))
    horzs = "".join(
        f'<rect x="{x0}" y="{y0 + j * step}" width="{n * step - gap}"'
        f' height="{band}" fill="{surface}" stroke="{ink}"'
        f' stroke-width="1.2"/>' for j in range(n))
    # Redraw the warp wherever it should cross OVER, which is every
    # other intersection — the checker pattern of a plain weave.
    over = "".join(
        f'<rect x="{x0 + i * step}" y="{y0 + j * step}" width="{band}"'
        f' height="{band}" fill="{pal}" fill-opacity=".16" stroke="{ink}"'
        f' stroke-width="1.2"/>'
        for i in range(n) for j in range(n) if (i + j) % 2 == 0)
    lit = "".join(
        f'<rect x="{x0 + i * step}" y="{y0 + j * step}" width="{band}"'
        f' height="{band}" fill="{pal}" class="lit"/>'
        for i, j in ((1, 2), (3, 1)))
    return f'<g class="draw">{verts}{horzs}{over}</g>{lit}'


def motif_orbit(pal: str, ink: str, line: str, surface: str = "#ffffff") -> str:
    """Concentric arcs with nodes — reach, network, a body of people.

    For services, membership, coverage, anything organised around a
    centre. Distinct from `dial`, which reads as a single instrument.
    """
    import math
    parts = [f'<g class="draw" fill="none" stroke="{ink}" stroke-width="1.1">']
    for r in (58, 104, 150, 196):
        parts.append(f'<circle cx="210" cy="258" r="{r}"/>')
    parts.append("</g>")
    for r, deg in ((58, 200), (104, 42), (104, 300), (150, 128), (196, 8),
                   (196, 232)):
        a = math.radians(deg)
        parts.append(
            f'<circle cx="{210 + r * math.cos(a):.1f}"'
            f' cy="{258 + r * math.sin(a):.1f}" r="7" fill="{pal}"'
            f' class="lit"/>')
    parts.append(f'<circle cx="210" cy="258" r="9" fill="{ink}"/>')
    return "".join(parts)


MOTIFS = {
    "elevation": motif_elevation, "topography": motif_topography,
    "schematic": motif_schematic, "dial": motif_dial, "specimen": motif_specimen,
    "strata": motif_strata, "lattice": motif_lattice, "orbit": motif_orbit,
}

# What each drawing actually contains, for the caption and the aria
# label.
#
# The model used to write this line, and it wrote it as though the
# figure were a photograph: "A reel of film on a turntable" captioning
# a line drawing of a building, "Terminal output" over a gauge. Told
# explicitly that the artwork is an abstract diagram, told which five
# diagrams exist and given a worked alt= for each, it carried on
# describing imagined photographs — the same way it carried on failing
# contrast after being handed the 4.5:1 rule.
#
# So the caption stops being something the model asserts. The compiler
# drew the picture and is the only party that knows what is in it. The
# cost is a less specific line than a person would write, which is why
# a hand-authored `alt=` still wins: a human writing ops knows what the
# building is called, and is answerable for saying so.
MOTIF_CAPTION = {
    "elevation": "Elevation",
    "topography": "Contour survey",
    "schematic": "System diagram",
    "dial": "Instrument face",
    "specimen": "Type specimen",
    "strata": "Section through the record",
    "lattice": "Weave detail",
    "orbit": "Reach diagram",
}

# Three kinds made every page the same page. Palette and family varied,
# the skeleton never did — hero, features, cta — and a fixed skeleton is
# what reads as generated no matter how well each part is set. These
# three additions were chosen because each has a DIFFERENT visual
# rhythm, not because they add more of the same: numerals at display
# size, a single sentence with no competition, and the artwork returning
# mirrored at a second scale.
SECTION_KINDS = {"hero", "features", "cta", "proof", "quote", "detail",
                 "faq", "gallery", "note", "roster"}
LAYOUTS = {"split", "centred", "stack", "grid", "list"}
TONES = {"quiet", "bold", "inverted"}
SLOTS = {
    "hero": {"eyebrow", "title", "lede", "primary", "secondary", "figure"},
    "features": {"eyebrow", "title", "lede", "item"},
    "cta": {"eyebrow", "title", "lede", "primary", "secondary"},
    # A stats band. ITEM title= carries the figure, body= what it counts.
    "proof": {"eyebrow", "title", "item"},
    # One sentence, one attribution, nothing else competing with it.
    "quote": {"quote", "attrib"},
    # The motif returns mirrored, so the drawing is a running element of
    # the page rather than a one-off decoration in the hero.
    "detail": {"eyebrow", "title", "lede", "primary", "figure"},
    # Questions people actually ask. Long-form, no artwork, and the one
    # kind that carries real paragraphs rather than captions.
    "faq": {"eyebrow", "title", "lede", "item"},
    # Several drawings at small scale. ITEM motif= picks each one.
    "gallery": {"eyebrow", "title", "lede", "item"},
    # A single wide statement, no art, no buttons. Punctuation between
    # two dense sections.
    "note": {"title", "lede"},
    # People or services: a name, what they do, and one line of detail.
    "roster": {"eyebrow", "title", "lede", "item"},
}


# ── Colour ───────────────────────────────────────────────────────────

def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(c * 255))) for c in (r, g, b))


def _lum(c: str) -> float:
    def chan(v: float) -> float:
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (chan(x) for x in _hex_to_rgb(c))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = _lum(a), _lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def derive_palette(accent: str, mode: str) -> dict[str, str]:
    """One accent in, a whole coherent palette out.

    The model picks a hue it likes and nothing else. Everything derived
    here is checked for contrast, because a 4B choosing seven related
    colours is how pages end up with grey-on-grey body text.
    """
    r, g, b = _hex_to_rgb(accent)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = max(0.28, min(0.82, s))          # keep it a colour, not a mud

    if mode == "dark":
        l_acc = max(0.58, min(0.74, l))
        canvas = colorsys.hls_to_rgb(h, 0.075, min(s, 0.16))
        surface = colorsys.hls_to_rgb(h, 0.125, min(s, 0.14))
        line = colorsys.hls_to_rgb(h, 0.24, min(s, 0.16))
        ink, muted = "#f4f2f0", "#a8a29c"
    else:
        l_acc = max(0.34, min(0.47, l))   # dark enough to carry white text
        canvas = colorsys.hls_to_rgb(h, 0.985, min(s, 0.22))
        surface = colorsys.hls_to_rgb(h, 0.955, min(s, 0.20))
        line = colorsys.hls_to_rgb(h, 0.86, min(s, 0.20))
        ink, muted = "#15120f", "#5d5751"

    acc = _rgb_to_hex(*colorsys.hls_to_rgb(h, l_acc, s))

    # A filled button is a background that has to carry a LABEL, so the
    # accent is itself a contrast constraint and not merely a hue.
    # Choosing the better of white and ink is NOT the same as clearing
    # 4.5:1 against either: a mid-green (#1f6f5c) came out at 4.30:1
    # against both and the old expression shipped the loser of two
    # failures without noticing. Walk the accent's lightness until one
    # of the two label colours genuinely clears, then take that one.
    #
    # This is the fourth token in this function that needed SOLVING
    # rather than picking — after accent-text against surface and
    # accent-inv against ink. The rule the other three should have
    # taught: any colour pair that carries text is solved in a loop with
    # a measured exit condition. A conditional expression that selects
    # between two candidates has no way to report that both were bad.
    for _ in range(24):
        if max(contrast(acc, "#ffffff"), contrast(acc, "#15120f")) >= 4.5:
            break
        l_acc = (max(0.10, l_acc - 0.02) if mode == "light"
                 else min(0.90, l_acc + 0.02))
        acc = _rgb_to_hex(*colorsys.hls_to_rgb(h, l_acc, s))

    hover = _rgb_to_hex(*colorsys.hls_to_rgb(h, max(0.2, l_acc - 0.09), s))
    on_acc = ("#ffffff" if contrast(acc, "#ffffff") >= contrast(acc, "#15120f")
              else "#15120f")

    pal = {
        "accent": acc, "accent-hover": hover, "on-accent": on_acc,
        "canvas": _rgb_to_hex(*canvas), "surface": _rgb_to_hex(*surface),
        "line": _rgb_to_hex(*line), "ink": ink, "muted": muted,
    }

    def solve_on(bg: str, target: float = 4.5) -> str:
        """The accent hue, lightened or darkened until it reads on `bg`.

        The direction is decided by the BACKGROUND's luminance, not by
        the page mode. Tying it to mode is the bug this replaces: the
        old inverted-band loop only ever lightened, which is right in
        light mode (where `ink` is near-black) and exactly backwards in
        dark mode, where `ink` is the near-WHITE band colour. A dark
        accent in dark mode therefore walked toward white and finished
        at 1.05:1 — an emphasised word invisible on its own band. The
        hue-space sweep found it; no brief we had ever hit it.
        """
        step = -0.03 if _lum(bg) > 0.4 else 0.03
        li, out = l_acc, acc
        for _ in range(40):
            if contrast(out, bg) >= target:
                return out
            nxt = li + step
            if not 0.02 <= nxt <= 0.98:
                break
            li = nxt
            out = _rgb_to_hex(*colorsys.hls_to_rgb(h, li, s))
        # The hue cannot reach AA on this ground at any lightness (very
        # low-saturation greys against a mid ground). Fall back to the
        # ground's own opposite rather than shipping the closest miss.
        return "#0d0b09" if _lum(bg) > 0.4 else "#ffffff"

    # An accent that carries white text is not automatically legible AS
    # text: a mid-amber passed on-accent and then failed at 3.60:1 as an
    # outline-button label. And the same problem again on the inverted
    # band, where the accent sits on `ink` rather than on `surface` — a
    # pale clay accent passed everywhere else and hit 2.49:1 there.
    # Two grounds means two solved tokens.
    pal["accent-text"] = solve_on(pal["surface"])
    pal["accent-inv"] = solve_on(pal["ink"])
    # Last-resort contrast repair: never ship unreadable body text.
    if contrast(pal["ink"], pal["canvas"]) < 7:
        pal["ink"] = "#0d0b09" if mode == "light" else "#ffffff"
    if contrast(pal["muted"], pal["canvas"]) < 4.5:
        pal["muted"] = "#4a453f" if mode == "light" else "#c3bdb6"

    # Four separate contrast bugs shipped from this one function, each
    # found by rendering a page and looking, each fixed only where it was
    # found. So state the invariant the four fixes were groping toward
    # and check it here: every pair in this palette that will carry text
    # clears AA, for EVERY accent, not just the ones a brief happened to
    # use. Failing here is loud and immediate; the alternative is a
    # screenshot gate catching it three hues later, or not at all.
    for fg, bg, what in (
        ("on-accent", "accent", "button label on its fill"),
        ("accent-text", "surface", "accent as text on a panel"),
        ("accent-text", "canvas", "accent as text on the page"),
        ("accent-inv", "ink", "accent inside an inverted band"),
        ("ink", "canvas", "body text"),
        ("ink", "surface", "body text on a panel"),
        ("muted", "canvas", "secondary text"),
        ("muted", "surface", "secondary text on a panel"),
        ("canvas", "ink", "inverted body text"),
    ):
        got = contrast(pal[fg], pal[bg])
        if got < 4.5:
            raise AssertionError(
                f"palette({accent}, {mode}): {what} is {got:.2f}:1 "
                f"({fg}={pal[fg]} on {bg}={pal[bg]}), needs 4.5:1")
    return pal


# ── Parsing ──────────────────────────────────────────────────────────

@dataclass
class Issue:
    line: int
    code: str
    detail: str


@dataclass
class Doc:
    theme: dict = field(default_factory=dict)
    site: dict = field(default_factory=dict)
    sections: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    fallbacks: list = field(default_factory=list)
    # Resolved photographs, keyed by the query the model wrote. Loaded
    # from <ops>.photos.json; empty when nothing was resolved, which is
    # the normal state for a drawings-only page.
    photos: dict = field(default_factory=dict)
    # Where the images sit relative to the OUTPUT file, not to the ops
    # file. Written as "assets/" the first time, the pages 404'd every
    # image: assets/ is a sibling of out/, so from a page inside out/
    # the correct prefix is "../assets/". Computed per build rather than
    # assumed, so a page written anywhere still finds its pictures.
    asset_base: str = "assets/"


KV = re.compile(r"(\w+)=((?:\"[^\"]*\")|(?:\S+))")


def parse(text: str) -> Doc:
    doc = Doc()
    by_id: dict[str, dict] = {}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        head, _, rest = line.partition(" ")
        op = head.strip().upper()
        if op not in OPS:
            doc.issues.append(Issue(n, "UNKNOWN_OP", head))
            continue
        args = {k: v.strip('"') for k, v in KV.findall(rest)}

        if op == "THEME":
            doc.theme = args
        elif op == "SITE":
            doc.site = args
        elif op == "SECTION":
            sid = args.get("id")
            kind = args.get("kind", "")
            if not sid:
                doc.issues.append(Issue(n, "MISSING_ID", line[:60]))
                continue
            if kind not in SECTION_KINDS:
                doc.issues.append(Issue(n, "BAD_KIND", kind))
                continue
            sec = {"id": sid, "kind": kind,
                   "layout": args.get("layout", "stack"),
                   "tone": args.get("tone", "quiet"),
                   "texts": [], "actions": [], "items": [], "media": []}
            if sec["layout"] not in LAYOUTS:
                doc.issues.append(Issue(n, "BAD_LAYOUT", sec["layout"]))
                continue
            if sec["tone"] not in TONES:
                doc.issues.append(Issue(n, "BAD_TONE", sec["tone"]))
                continue
            by_id[sid] = sec
            doc.sections.append(sec)
        else:
            sid = args.get("sec")
            sec = by_id.get(sid)
            if sec is None:
                doc.issues.append(Issue(n, "UNKNOWN_SECTION", str(sid)))
                continue
            slot = args.get("slot", "")
            legal = SLOTS.get(sec["kind"], set())
            if op in {"TEXT", "ACTION"} and slot not in legal:
                doc.issues.append(Issue(n, "ILLEGAL_SLOT", f"{slot} in {sec['kind']}"))
                continue
            if op == "TEXT":
                # `accent` is one of the few bounded freedoms the model
                # keeps: a substring of its own text to set apart. One
                # word in a headline carrying the brand colour is the
                # cheapest thing that separates typeset from typed.
                # Emphasis is marked INLINE, with [brackets] inside the
                # text itself, rather than by a separate argument that
                # names a substring. Two earlier spellings failed for
                # the same reason: a separate argument can name
                # something that is not there. `accent=` collided with
                # THEME's colour and got a hex; `mark=` got phrases
                # lifted from the brief instead of from the title
                # ("Wren & Slip" marked "hand-thrown"). Brackets cannot
                # miss, because the mark IS part of the string.
                sec["texts"].append((slot, args.get("text", ""), ""))
            elif op == "ACTION":
                sec["actions"].append((slot, args.get("label", ""), args.get("href", "#")))
            elif op == "ITEM":
                sec["items"].append((args.get("title", ""), args.get("body", ""),
                                     args.get("motif", "")))
            elif op == "MEDIA":
                sec["media"].append((args.get("alt", ""),
                                     args.get("motif", "elevation"),
                                     args.get("photo", "")))
    return doc


# ── Rendering ────────────────────────────────────────────────────────

def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def _steps(scale: float) -> dict[str, str]:
    """A fluid type scale from one ratio. Display travels between
    viewports; body copy barely moves, which is what keeps long text
    readable on a phone.

    A mono-specific reduction lived here briefly: I added it to fix a
    mobile overflow without measuring first, and the actual culprit was
    the masthead nav. Removed rather than kept as a lucky guess — an
    unverified change is a future mystery for whoever reads this next.
    """
    def clamp(step: int, travel: float) -> str:
        base = 1.0 * (scale ** step)
        small = base / (1 + travel * 0.5)
        return (f"clamp({small:.3f}rem, {small:.3f}rem + "
                f"{(base - small) * 1.6:.2f}vw, {base:.3f}rem)")
    # The down-step must NOT inherit the display ratio. Dividing by it
    # gave the studio family (ratio 1.5) a 0.667rem = 10.7px eyebrow and
    # button label — caught by the readable-size gate on two of six
    # pages. A dramatic ratio is wanted going UP, where it creates
    # contrast between headline and body; going down it just produces
    # text nobody can read. Half the ratio's excess, floored at 0.8rem.
    small = max(0.8, 1 / (1 + (scale - 1) / 2))
    return {
        "d1": clamp(4, 0.55), "d2": clamp(3, 0.45), "d3": clamp(2, 0.3),
        "lede": clamp(1, 0.14), "body": clamp(0, 0.04),
        "small": f"{small:.3f}rem",
    }


def render(doc: Doc) -> tuple[str, list[str]]:
    fam_name = doc.theme.get("family", "editorial")
    if fam_name not in FAMILIES:
        doc.fallbacks.append(f"family={fam_name} unknown -> editorial")
        fam_name = "editorial"
    fam = FAMILIES[fam_name]

    accent = doc.theme.get("accent", "")
    if not re.fullmatch(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})", accent):
        doc.fallbacks.append(f"accent={accent!r} invalid -> #b8442f")
        accent = "#b8442f"
    mode = doc.theme.get("mode", "light")
    if mode not in {"light", "dark"}:
        doc.fallbacks.append(f"mode={mode} -> light")
        mode = "light"
    dens = doc.theme.get("density", "normal")
    if dens not in DENSITY:
        doc.fallbacks.append(f"density={dens} -> normal")
        dens = "normal"

    pal = derive_palette(accent, mode)
    st = _steps(fam["scale"])
    k = DENSITY[dens]

    css = f"""
*,*::before,*::after{{box-sizing:border-box}}
html{{-webkit-text-size-adjust:100%}}
body{{margin:0;background:{pal['canvas']};color:{pal['ink']};
  font-family:{fam['body']};font-size:{st['body']};line-height:1.65;
  text-rendering:optimizeLegibility}}
h1,h2,h3{{font-family:{fam['display']};font-weight:{fam['weight_display']};
  letter-spacing:{fam['tracking_display']};line-height:1.06;margin:0;
  text-wrap:balance;
  /* Guard: a single long word must break rather than push the page
     sideways. Belt to the clamp's braces — a horizontal scrollbar is
     the most visible defect a layout can have. */
  overflow-wrap:break-word}}
p{{margin:0;max-width:{fam['measure']}}}
a{{color:inherit}}
img{{max-width:100%;display:block}}
:root{{--accent:{pal['accent']};--line:{pal['line']};--surface:{pal['surface']};
  --muted:{pal['muted']};--gap:{2.1 * k:.2f}rem;--bleed:{5.5 * k:.2f}rem}}
.wrap{{width:min(100% - 2.5rem,76rem);margin-inline:auto}}
section{{padding-block:var(--bleed);border-top:{fam['rule']}}}
section:first-of-type{{border-top:0}}

/* Masthead and footer. Without them a page opens on a section and
   stops mid-thought — it reads as a fragment rather than a site, which
   was the most-felt difference against the hand-written reference. */
/* The masthead must wrap. Three tracked uppercase nav items measured
   313px against a 390px viewport and pushed the page to a 429px
   scrollWidth — the only mobile overflow across six pages, and it came
   from the chrome rather than the content. */
.masthead{{display:flex;justify-content:space-between;align-items:baseline;
  gap:.6rem 2rem;flex-wrap:wrap;padding:1.5rem 0 1.3rem;
  border-bottom:{fam['rule']}}}
.wordmark{{font-family:{fam['mono']};font-size:{st['small']};
  letter-spacing:.2em;text-transform:uppercase;font-weight:600}}
.masthead nav{{display:flex;gap:.5rem 1.4rem;flex-wrap:wrap;
  font-family:{fam['mono']};font-size:{st['small']};letter-spacing:.14em;
  text-transform:uppercase;color:{pal['muted']}}}
.masthead nav a{{text-decoration:none;padding-bottom:2px;
  border-bottom:1px solid transparent}}
.masthead nav a:hover{{border-bottom-color:{pal['accent']}}}
.sitefoot{{padding:2.2rem 0 2.8rem;border-top:{fam['rule']};
  font-family:{fam['mono']};font-size:{st['small']};letter-spacing:.1em;
  text-transform:uppercase;color:{pal['muted']}}}
.sitefoot .wrap{{display:flex;justify-content:space-between;flex-wrap:wrap;
  gap:.9rem}}

/* A short rule before the label — one of those details that costs
   nothing and reads as considered. */
.eyebrow{{font-family:{fam['mono']};font-size:{st['small']};
  letter-spacing:{fam['tracking_eyebrow']};text-transform:uppercase;
  color:{pal['muted']};margin:0 0 1.4rem}}
.eyebrow::before{{content:"";display:inline-block;width:1.8rem;height:1px;
  background:{pal['accent']};vertical-align:.3em;margin-right:.7rem}}
/* Scoped to h1/h2 originally, which meant the emphasis brackets did
   NOTHING inside a pullquote — the span rendered, inherited the body
   colour and looked like ordinary text. Bind emphasis to the mark, not
   to the two tags that happened to use it first. */
.em{{color:{pal['accent-text']};font-style:italic}}
.inverted .em{{color:{pal['accent-inv']}}}
h1{{font-size:{st['d1']}}} h2{{font-size:{st['d2']}}} h3{{font-size:{st['d3']}}}
.lede{{font-size:{st['lede']};color:{pal['muted']};margin-top:1.3rem;
  line-height:1.5}}
.actions{{display:flex;flex-wrap:wrap;gap:.85rem;margin-top:2.2rem}}
.btn{{display:inline-block;padding:.85em 1.6em;border-radius:{fam['radius']};
  text-decoration:none;font-family:{fam['body']};font-size:{st['small']};
  letter-spacing:.02em;border:1px solid var(--accent);transition:background .15s}}
.btn-primary{{background:{pal['accent']};color:{pal['on-accent']}}}
.btn-primary:hover{{background:{pal['accent-hover']}}}
.btn-secondary{{background:transparent;color:{pal['accent-text']}}}
.btn-secondary:hover{{background:var(--surface)}}
.btn:focus-visible,a:focus-visible{{outline:2px solid {pal['accent']};
  outline-offset:3px}}
/* Asymmetric hero on a 12-column grid: art occupies 7..13 and the text
   sits at 1..8, so the headline crosses the gutter and overlaps the
   art's column. A symmetric 50/50 split with everything vertically
   centred is what made the first version read competent-but-inert. */
.split{{display:grid;gap:var(--gap);grid-template-columns:1fr}}
@media(min-width:56rem){{
  .split{{grid-template-columns:repeat(12,1fr);column-gap:1.5rem;
    align-items:end}}
  .split > div:first-child{{grid-column:1 / 8;grid-row:1;z-index:2;
    position:relative;padding-bottom:.5rem}}
  .split > .figure{{grid-column:7 / 13;grid-row:1;align-self:stretch}}
}}
.steps{{list-style:none;margin:2.4rem 0 0;padding:0}}
.steps li{{display:grid;grid-template-columns:3.6rem minmax(0,13rem) 1fr;
  gap:1.4rem;padding:1.7rem 0;border-top:{fam['rule']};align-items:baseline}}
.steps li:last-child{{border-bottom:{fam['rule']}}}
.steps .num{{font-family:{fam['mono']};font-size:{st['small']};
  color:{pal['accent-text']};letter-spacing:.08em}}
.steps h3{{font-size:{st['lede']};margin:0}}
.steps p{{color:{pal['muted']};max-width:34rem}}
@media(max-width:56rem){{
  .steps li{{grid-template-columns:2.6rem 1fr;gap:.35rem 1rem}}
  .steps p{{grid-column:2}}
}}
/* Closing band: heading left, action pushed right. Centring everything
   is the safe choice and reads as a template. */
.closing{{display:grid;grid-template-columns:1fr;gap:1.6rem}}
@media(min-width:56rem){{
  .closing{{grid-template-columns:repeat(12,1fr);column-gap:1.5rem;
    align-items:end}}
  .closing > div:first-child{{grid-column:1 / 8}}
  .closing > .actions{{grid-column:9 / 13;margin-top:0;justify-content:flex-end}}
}}
.centred{{text-align:center}}
.centred p,.centred .actions{{margin-inline:auto}}
.centred .actions{{justify-content:center}}
.grid{{display:grid;gap:var(--gap);grid-template-columns:1fr;margin-top:3rem}}
@media(min-width:44rem){{.grid{{grid-template-columns:repeat(2,1fr)}}}}
@media(min-width:64rem){{.grid{{grid-template-columns:repeat(3,1fr)}}}}
.card{{border:{fam['rule']};border-radius:{fam['radius']};padding:1.8rem;
  background:var(--surface)}}
.card h3{{font-size:{st['lede']};margin-bottom:.7rem}}
.card p{{color:{pal['muted']}}}
/* Drawn artwork, not a colour block. The flat accent panel this
   replaces was the single most obviously generated element on the
   page — a rectangle standing in for a picture. The model chooses a
   motif from a table; the compiler draws it, so the model never emits
   SVG (it cannot close long markup) and the drawing is always valid. */
.figure{{margin:0;position:relative;background:{pal['surface']};
  border:{fam['rule']};border-radius:{fam['radius']};overflow:hidden;
  aspect-ratio:4/5;padding:1.2rem}}
.figure svg{{width:100%;height:100%;display:block}}
.figure figcaption{{position:absolute;left:1.3rem;bottom:1.1rem;
  font-family:{fam['mono']};font-size:{st['small']};letter-spacing:.1em;
  text-transform:uppercase;color:{pal['muted']}}}
@media(min-width:56rem){{.figure{{height:100%}}}}
/* A photograph is not a drawing and must not wear the drawing's frame.
   The motif figure is a padded, bordered panel because a line diagram
   needs a field around it; the same treatment around a photo reads as a
   slide deck. Photos go edge to edge in their box, keep their own
   aspect ratio, and carry the credit line underneath rather than
   floating over the image where it would need its own contrast fix. */
.figure.photo{{padding:0;border:0;background:transparent;aspect-ratio:auto;
  display:flex;flex-direction:column;gap:.85rem}}
.figure.photo img{{display:block;width:100%;height:auto;object-fit:cover;
  border:{fam['rule']};border-radius:{fam['radius']};background:{pal['surface']};
  aspect-ratio:4/3}}
.figure.photo figcaption{{position:static;left:auto;bottom:auto;
  text-transform:none;letter-spacing:.01em;line-height:1.45;
  font-family:{fam['body']};max-width:44ch}}
@media(min-width:56rem){{
  .figure.photo{{height:100%;justify-content:flex-end}}
  .figure.photo img{{aspect-ratio:3/4;flex:1;min-height:0}}
}}
/* The detail band mirrors the hero rather than repeating it: same
   12-column grid, columns swapped, so art leads on the left and the
   text block closes on the right. */
@media(min-width:56rem){{
  .mirror > .figure{{grid-column:1 / 7;grid-row:1}}
  .mirror > div:last-child{{grid-column:7 / 13;grid-row:1;z-index:2;
    position:relative}}
}}
/* A pullquote is a pause. One sentence, generous leading, an accent
   rule instead of quotation marks — decorative glyphs at this size
   read as clip art. */
/* The measure belongs on the element that carries the LARGE type. A ch
   limit on the blockquote is computed against the blockquote's own
   16px, so `30ch` came out 295px wide and wrapped a 28px quote into
   four stubby lines in the corner of a full-bleed band. ch units are
   relative to the font-size of the element they are written on. */
.pull{{margin:0;border-left:2px solid {pal['accent']};padding-left:2rem}}
.pull p{{font-family:{fam['display']};font-weight:{fam['weight_display']};
  letter-spacing:{fam['tracking_display']};font-size:{st['d2']};
  line-height:1.28;margin:0;max-width:19ch;text-wrap:balance}}
.pull cite{{display:block;margin-top:1.5rem;font-family:{fam['mono']};
  font-size:{st['small']};font-style:normal;letter-spacing:.08em;
  text-transform:uppercase;color:{pal['muted']}}}
.inverted .pull p{{color:{pal['canvas']}}}
/* `muted` is solved against `canvas`; on an inverted band its ground is
   `ink`, where it measured 2.51:1. Every other inverted rule already
   restated its colour — this one was missed because <cite> was outside
   the gate's tag list, so nothing objected. */
.inverted .pull cite{{color:{pal['canvas']};opacity:.72}}
/* Numerals as the artwork. tabular-nums so the figures line up down
   the row, and the caption stays small so the figure carries it. */
.stats{{display:grid;gap:var(--gap) 1.5rem;grid-template-columns:1fr;
  margin-top:2.8rem}}
@media(min-width:40rem){{
  .stats{{grid-template-columns:repeat(min(var(--cols,4),2),1fr)}}}}
@media(min-width:60rem){{
  .stats{{grid-template-columns:repeat(var(--cols,4),1fr)}}}}
.stat{{border-top:{fam['rule']};padding-top:1.2rem}}
.stat .fig{{display:block;font-family:{fam['display']};font-size:{st['d2']};
  font-weight:{fam['weight_display']};line-height:1;
  font-variant-numeric:tabular-nums;color:{pal['accent-text']};
  letter-spacing:-.01em}}
.inverted .stat .fig{{color:{pal['accent-inv']}}}
.stat .cap{{display:block;margin-top:.7rem;font-size:{st['small']};
  color:{pal['muted']};max-width:22ch;line-height:1.45}}
.inverted .stat .cap{{color:{pal['canvas']};opacity:.78}}
/* Questions and answers. Two columns on wide screens so a long answer
   does not run the full page width, and the question sits in the
   display face to separate it from its own answer without a rule. */
.qas{{display:grid;gap:2.2rem 3rem;grid-template-columns:1fr;
  margin-top:2.8rem}}
@media(min-width:56rem){{.qas{{grid-template-columns:repeat(2,1fr)}}}}
.qa h3{{font-size:{st['lede']};margin:0 0 .6rem}}
.qa p{{color:{pal['muted']};line-height:1.62;margin:0;max-width:46ch}}
.inverted .qa p{{color:{pal['canvas']};opacity:.78}}
/* Several drawings at small scale. 3:4 rather than the hero's 4:5 so a
   row of them is not taller than the text that introduces it. */
.tiles{{display:grid;gap:var(--gap);grid-template-columns:1fr;
  margin-top:2.8rem}}
@media(min-width:44rem){{.tiles{{grid-template-columns:repeat(2,1fr)}}}}
@media(min-width:64rem){{.tiles{{grid-template-columns:repeat(3,1fr)}}}}
.tile{{margin:0}}
.tile svg{{display:block;width:100%;aspect-ratio:3/4;background:{pal['surface']};
  border:{fam['rule']};border-radius:{fam['radius']};padding:1rem}}
.tile figcaption{{margin-top:.9rem}}
.tile b{{display:block;font-family:{fam['mono']};font-size:{st['small']};
  letter-spacing:.1em;text-transform:uppercase;font-weight:400;
  color:{pal['accent-text']}}}
.tile span{{display:block;margin-top:.4rem;font-size:{st['small']};
  color:{pal['muted']};line-height:1.5}}
.inverted .tile b{{color:{pal['accent-inv']}}}
.inverted .tile span{{color:{pal['canvas']};opacity:.78}}
/* A single wide statement. No max-width on the heading: this is the one
   place the type is allowed to run long, which is what makes it read as
   a break rather than another section. */
.note h2{{font-size:{st['d2']};max-width:20ch}}
.note .lede{{max-width:52ch;font-size:{st['lede']}}}
/* People or services. Names in the display face, roles beneath. */
.whos{{display:grid;gap:2rem 2.4rem;grid-template-columns:1fr;
  margin-top:2.6rem}}
@media(min-width:40rem){{.whos{{grid-template-columns:repeat(2,1fr)}}}}
@media(min-width:64rem){{.whos{{grid-template-columns:repeat(3,1fr)}}}}
.who{{border-top:{fam['rule']};padding-top:1.1rem}}
.who h3{{font-size:{st['lede']};margin:0 0 .45rem}}
.who p{{color:{pal['muted']};line-height:1.55;margin:0}}
.inverted .who p{{color:{pal['canvas']};opacity:.78}}

/* Motion. Two effects only, both cheap and both reversible: strokes
   draw themselves once, and blocks rise as they enter. Anything more
   is the thing that makes a generated page feel like a template demo.
   The reduced-motion query below is not decoration — it disables the
   dash offset and the transform so the page is complete without JS
   and without movement. */
.draw [class],.draw{{}}
.draw *{{stroke-dasharray:var(--len,900);stroke-dashoffset:var(--len,900);
  animation:draw 1.5s cubic-bezier(.22,.61,.36,1) forwards}}
@keyframes draw{{to{{stroke-dashoffset:0}}}}
.lit{{opacity:0;animation:lit .7s ease 1.1s forwards}}
@keyframes lit{{to{{opacity:.85}}}}
/* CRITICAL: the hidden state is applied ONLY once script has set
   html.js. The first version started every section at opacity:0 and
   waited for an observer, which meant a JS failure — or a full-page
   screenshot taken before the observer fired — showed a blank page
   with two of three sections missing. Content must never depend on
   script to become visible; the animation is an enhancement layered
   on top of a page that already works. */
html.js .rise{{opacity:0;transform:translateY(14px);
  transition:opacity .6s ease,transform .6s cubic-bezier(.22,.61,.36,1)}}
html.js .rise.in{{opacity:1;transform:none}}
.inverted{{background:{pal['ink']};color:{pal['canvas']}}}
.inverted .lede,.inverted .eyebrow,.inverted .card p{{color:{pal['canvas']};
  opacity:.78}}
.inverted .card{{background:transparent;border-color:{pal['muted']}}}
.bold-tone{{background:var(--surface)}}
@media(prefers-reduced-motion:reduce){{
  *{{transition:none!important;animation:none!important}}
  .draw *{{stroke-dashoffset:0!important}}
  .lit{{opacity:.85!important}}
  .rise{{opacity:1!important;transform:none!important}}
}}
"""

    def _accented(text: str, _unused: str = "") -> str:
        """Turn `Buildings that [age] well` into an emphasised span.

        Escape first, then substitute: escaping afterwards would eat the
        span, and wrapping before escaping would let model-authored text
        inject markup.
        """
        safe = _esc(text)
        if "[" not in safe:
            return safe
        return re.sub(r"\[([^\[\]]{1,60})\]",
                      r'<span class="em">\1</span>', safe, count=1).replace(
            "[", "").replace("]", "")

    body: list[str] = []
    for sec in doc.sections:
        texts = {t[0]: (t[1], t[2] if len(t) > 2 else "") for t in sec["texts"]}
        # Tone paints the FULL-BLEED section; layout arranges content
        # INSIDE the centred wrap. Putting the layout class on <section>
        # made the section itself a two-column grid, so .wrap landed in
        # column one and the hero sat at x=20 while every other section
        # started at x=144 — visible instantly in the render, invisible
        # in the markup.
        cls = []
        if sec["tone"] == "inverted":
            cls.append("inverted")
        elif sec["tone"] == "bold":
            cls.append("bold-tone")
        inner: list[str] = []
        if texts.get("eyebrow"):
            inner.append(f'<p class="eyebrow">{_esc(texts["eyebrow"][0])}</p>')
        tag = "h1" if sec["kind"] == "hero" else "h2"
        if texts.get("title"):
            t, acc = texts["title"]
            inner.append(f"<{tag}>{_accented(t, acc)}</{tag}>")
        if texts.get("lede"):
            inner.append(f'<p class="lede">{_esc(texts["lede"][0])}</p>')
        actions_html = ""
        if sec["actions"]:
            btns = "".join(
                f'<a class="btn btn-{"primary" if slot == "primary" else "secondary"}"'
                f' href="{_esc(href)}">{_esc(label)}</a>'
                for slot, label, href in sec["actions"])
            actions_html = f'<div class="actions">{btns}</div>'
        if sec["kind"] != "cta":
            inner.append(actions_html)

        col = "\n".join(inner)
        centred = ' class="centred"' if sec["layout"] == "centred" else ""

        def _figure() -> str:
            media = sec["media"][0] if sec["media"] else ("", "elevation", "")
            alt, motif, photo = (list(media) + ["", "", ""])[:3]

            # A resolved photograph wins over a drawing. The query is the
            # model's; the file, its dimensions and its licence come from
            # the resolver, because none of those are things a model can
            # know. An unresolved query falls back to the drawing rather
            # than to a broken <img>: a missing picture must degrade to a
            # real picture, never to an alt-text box.
            got = doc.photos.get(photo) if photo else None
            if photo and not got:
                doc.fallbacks.append(f'photo="{photo}" unresolved -> motif drawing')
            if got:
                w, h = got.get("width") or 1600, got.get("height") or 1000
                # width/height attributes are not decoration: without the
                # intrinsic ratio the browser cannot reserve the box and
                # every photo shoves the page down as it loads.
                cap = got["credit"] if got.get("needs_credit") else (
                    alt.strip() or got["credit"])
                # The first picture on the page is almost always the
                # largest thing in the viewport, and lazy-loading the
                # element that decides your LCP is a well-known way to
                # make a page feel slow. Later figures stay lazy.
                doc.seen_photo = getattr(doc, "seen_photo", False)
                first = not doc.seen_photo
                doc.seen_photo = True
                load = ('fetchpriority="high" decoding="async"' if first
                        else 'loading="lazy" decoding="async"')
                return (
                    f'<figure class="figure photo">'
                    f'<img src="{_esc(doc.asset_base)}{_esc(got["file"])}" '
                    f'alt="{_esc(alt.strip() or photo)}" '
                    f'width="{w}" height="{h}" {load}>'
                    f'<figcaption>{_esc(cap)}</figcaption></figure>')

            if motif not in MOTIFS:
                doc.fallbacks.append(f"motif={motif} unknown -> elevation")
                motif = "elevation"
            # Derived unless a human wrote one. See MOTIF_CAPTION.
            label = alt.strip() or " — ".join(
                x for x in (MOTIF_CAPTION[motif], doc.site.get("name", "")) if x)
            art = MOTIFS[motif](pal["accent"], pal["ink"], pal["muted"], pal["surface"])
            return (
                f'<figure class="figure">'
                f'<svg viewBox="0 0 420 520" role="img" aria-label="{_esc(label)}">'
                f'{art}</svg>'
                f'<figcaption>{_esc(label)}</figcaption></figure>')

        if sec["kind"] == "quote":
            # No heading, no eyebrow, no button. The whole effect is one
            # sentence set large with room around it; adding anything
            # else to this band is what turns a pause into another slab.
            said = texts.get("quote", ("", ""))[0]
            who = texts.get("attrib", ("", ""))[0]
            cite = f'<cite>{_esc(who)}</cite>' if who else ""
            content = (f'<blockquote class="pull">'
                       f'<p>{_accented(said)}</p>{cite}</blockquote>')
        elif sec["kind"] == "proof":
            # Figures at display size, tabular so the digits align down
            # the row. This is the one place numerals are the artwork.
            stats = "".join(
                f'<div class="stat"><span class="fig">{_esc(v)}</span>'
                f'<span class="cap">{_esc(lab)}</span></div>'
                for v, lab, _ in sec["items"])
            # Track count follows the item count. A fixed four-column
            # grid is right for four figures and leaves half the row
            # visibly empty for two — and two is now the common case,
            # because ungrounded figures get dropped rather than
            # invented. The layout has to survive its own honesty.
            n = max(1, min(len(sec["items"]), 4))
            content = (f'<div{centred}>{col}</div>'
                       f'<div class="stats" style="--cols:{n}">{stats}</div>')
        elif sec["kind"] == "faq":
            # The one kind that carries paragraphs. Questions are set in
            # the display face at body size so the page has somewhere to
            # put real prose — the rest of the vocabulary is captions and
            # one-liners, which is a large part of why every page read
            # thin however many sections it had.
            rows = "".join(
                f'<div class="qa"><h3>{_esc(q)}</h3><p>{_esc(a)}</p></div>'
                for q, a, _ in sec["items"])
            content = f'<div{centred}>{col}</div><div class="qas">{rows}</div>'
        elif sec["kind"] == "gallery":
            tiles = []
            # `cap, line, motif` — NOT `body`, which is the page-level
            # accumulator built a few lines below. Unpacking into it
            # replaced the list of rendered sections with a string and
            # the compile died on .append two sections later, nowhere
            # near the cause.
            for cap, line, motif in sec["items"]:
                if motif not in MOTIFS:
                    if motif:
                        doc.fallbacks.append(f"motif={motif} unknown -> schematic")
                    motif = "schematic"
                art = MOTIFS[motif](pal["accent"], pal["ink"], pal["muted"], pal["surface"])
                label = cap or MOTIF_CAPTION[motif]
                tiles.append(
                    f'<figure class="tile">'
                    f'<svg viewBox="0 0 420 520" role="img"'
                    f' aria-label="{_esc(label)}">{art}</svg>'
                    f'<figcaption><b>{_esc(label)}</b>'
                    + (f'<span>{_esc(line)}</span>' if line else "")
                    + '</figcaption></figure>')
            content = (f'<div{centred}>{col}</div>'
                       f'<div class="tiles">{"".join(tiles)}</div>')
        elif sec["kind"] == "note":
            # Deliberately the least furnished thing on the page: no
            # eyebrow, no art, no button. It exists to break two dense
            # sections apart, and gains nothing from being given more.
            content = f'<div class="note">{col}</div>'
        elif sec["kind"] == "roster":
            rows = "".join(
                f'<div class="who"><h3>{_esc(n)}</h3><p>{_esc(d)}</p></div>'
                for n, d, _ in sec["items"])
            content = f'<div{centred}>{col}</div><div class="whos">{rows}</div>'
        elif sec["kind"] == "detail":
            # Mirrored against the hero: art left, text right. Same grid,
            # reversed columns, so the eye travels the other way and the
            # page has a rhythm instead of a stack of identical rows.
            content = (f'<div class="split mirror">{_figure()}'
                       f'<div>{col}</div></div>')
        elif sec["kind"] == "hero" and sec["layout"] == "split":
            content = f'<div class="split"><div>{col}</div>{_figure()}</div>'
        elif sec["items"] and sec["layout"] == "list":
            # A rule-separated numbered list instead of bordered cards.
            # Cards are the safe default and read Bootstrap-generic;
            # numerals on hairlines read editorial.
            rows = "".join(
                f'<li><span class="num">{i:02d}</span>'
                f'<h3>{_esc(t)}</h3><p>{_esc(b)}</p></li>'
                for i, (t, b, _) in enumerate(sec["items"], 1))
            content = f'<div{centred}>{col}</div><ol class="steps">{rows}</ol>'
        elif sec["items"]:
            cards = "".join(
                f'<div class="card"><h3>{_esc(t)}</h3><p>{_esc(b)}</p></div>'
                for t, b, _ in sec["items"])
            content = f'<div{centred}>{col}</div><div class="grid">{cards}</div>'
        elif sec["kind"] == "cta" and sec["layout"] != "centred":
            content = f'<div class="closing"><div>{col}</div>{actions_html}</div>'
        else:
            content = f'<div{centred}>{col}{actions_html}</div>'

        cls.append("rise")
        section_cls = " ".join(cls)
        body.append(
            f'<section class="{section_cls}"><div class="wrap">{content}</div></section>')

    site = doc.site
    masthead = ""
    if site.get("name"):
        nav = "".join(
            f'<a href="#{_esc(n.strip().lower())}">{_esc(n.strip())}</a>'
            for n in site.get("nav", "").split(",") if n.strip())
        masthead = (f'<header class="masthead wrap">'
                    f'<span class="wordmark">{_esc(site["name"])}</span>'
                    f'<nav>{nav}</nav></header>')
    foot = ""
    bits = [site.get("name", ""), site.get("location", ""), site.get("contact", "")]
    if any(bits):
        cells = "".join(f"<span>{_esc(b)}</span>" for b in bits if b)
        foot = f'<footer class="sitefoot"><div class="wrap">{cells}</div></footer>'

    title = ""
    for sec in doc.sections:
        for t in sec["texts"]:
            if t[0] == "title" and t[1]:
                title = t[1]
                break
        if title:
            break
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title or "Untitled")}</title>
<style>{css}</style>
</head>
<body>
{masthead}
<main>
{chr(10).join(body)}
</main>
{foot}
<script>
/* Reveal-on-enter, and the exact stroke length for each drawn path so
   the dash animation is proportional instead of guessed. Fifteen lines,
   no dependency, and the page is already complete without it: every
   .rise element is made visible immediately if motion is reduced or if
   IntersectionObserver is missing. Progressive enhancement, not a
   requirement. */
(function(){{
  var reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  // Opt into the hidden state only when we can guarantee we will undo
  // it: script is running AND motion is wanted.
  if (!reduce && 'IntersectionObserver' in window) {{
    document.documentElement.classList.add('js');
  }}
  document.querySelectorAll('.draw *').forEach(function(el){{
    if (el.getTotalLength) {{
      try {{ el.style.setProperty('--len', Math.ceil(el.getTotalLength())); }}
      catch (e) {{}}
    }}
  }});
  var items = document.querySelectorAll('.rise');
  if (reduce || !('IntersectionObserver' in window)) {{
    items.forEach(function(el){{ el.classList.add('in'); }});
    return;
  }}
  var io = new IntersectionObserver(function(entries){{
    entries.forEach(function(e){{
      if (e.isIntersecting) {{ e.target.classList.add('in'); io.unobserve(e.target); }}
    }});
  }}, {{rootMargin: '0px 0px -12% 0px'}});
  items.forEach(function(el){{ io.observe(el); }});
}})();
</script>
</body>
</html>
"""
    return page, doc.fallbacks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ops")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any issue or fallback occurred")
    args = ap.parse_args()

    ops_path = Path(args.ops)
    doc = parse(ops_path.read_text(encoding="utf-8"))
    # Photographs resolved by photos.py, if this brief has any. Absent
    # sidecar is not an error — a drawings-only page is the default, and
    # any query that failed to resolve is reported as a fallback when it
    # falls back to a motif.
    side = ops_path.with_suffix(".photos.json")
    if side.exists():
        doc.photos = json.loads(side.read_text(encoding="utf-8"))
        assets = (ops_path.parent.parent / "assets").resolve()
        out_dir = Path(args.out).resolve().parent
        rel = os.path.relpath(assets, out_dir).replace("\\", "/")
        doc.asset_base = rel.rstrip("/") + "/"
    page, fallbacks = render(doc)
    Path(args.out).write_text(page, encoding="utf-8")

    for i in doc.issues:
        print(f"  line {i.line}: {i.code} {i.detail}")
    for f in fallbacks:
        print(f"  FALLBACK {f}")
    clean = not doc.issues and not fallbacks
    print(f"{len(doc.sections)} sections, {len(doc.issues)} issues, "
          f"{len(fallbacks)} fallbacks -> {args.out}"
          f"{'  [CLEAN]' if clean else ''}")
    return 1 if args.strict and not clean else 0


if __name__ == "__main__":
    raise SystemExit(main())
