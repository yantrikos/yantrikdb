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

def motif_elevation(pal: str, ink: str, line: str) -> str:
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
 font-family="ui-monospace,monospace" font-size="11">9 400</text>"""


def motif_topography(pal: str, ink: str, line: str) -> str:
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


def motif_schematic(pal: str, ink: str, line: str) -> str:
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


def motif_dial(pal: str, ink: str, line: str) -> str:
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


def motif_specimen(pal: str, ink: str, line: str) -> str:
    """A type specimen — the letterform itself as the artwork."""
    return (f'<text x="210" y="368" text-anchor="middle" fill="{ink}"'
            f' font-family="Georgia,serif" font-size="330"'
            f' font-weight="600" class="lit">Aa</text>'
            f'<g stroke="{line}" stroke-width="1" class="draw">'
            f'<line x1="40" y1="368" x2="380" y2="368"/>'
            f'<line x1="40" y1="176" x2="380" y2="176"/>'
            f'<line x1="40" y1="252" x2="380" y2="252"/></g>'
            f'<text x="44" y="166" fill="{pal}" font-family="ui-monospace,monospace"'
            f' font-size="11">CAP HEIGHT</text>'
            f'<text x="44" y="242" fill="{pal}" font-family="ui-monospace,monospace"'
            f' font-size="11">X-HEIGHT</text>')


MOTIFS = {
    "elevation": motif_elevation, "topography": motif_topography,
    "schematic": motif_schematic, "dial": motif_dial, "specimen": motif_specimen,
}

SECTION_KINDS = {"hero", "features", "cta"}
LAYOUTS = {"split", "centred", "stack", "grid", "list"}
TONES = {"quiet", "bold", "inverted"}
SLOTS = {
    "hero": {"eyebrow", "title", "lede", "primary", "secondary", "figure"},
    "features": {"eyebrow", "title", "lede", "item"},
    "cta": {"eyebrow", "title", "lede", "primary", "secondary"},
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
    hover = _rgb_to_hex(*colorsys.hls_to_rgb(h, max(0.2, l_acc - 0.09), s))
    on_acc = "#ffffff" if contrast(acc, "#ffffff") >= 4.5 else "#15120f"

    pal = {
        "accent": acc, "accent-hover": hover, "on-accent": on_acc,
        "canvas": _rgb_to_hex(*canvas), "surface": _rgb_to_hex(*surface),
        "line": _rgb_to_hex(*line), "ink": ink, "muted": muted,
    }

    # An accent that carries white text is not automatically legible AS
    # text. A mid-amber passed on-accent and then failed at 3.60:1 as an
    # outline-button label — one hue, one page, and only because the
    # gate looked. Walk lightness until the accent clears 4.5:1 against
    # the lightest thing it will sit on, and keep it as its own token.
    lo, hi = (0.05, l_acc) if mode == "light" else (l_acc, 0.97)
    accent_text = acc
    for _ in range(24):
        if contrast(accent_text, pal["surface"]) >= 4.5:
            break
        if mode == "light":
            hi = max(0.04, hi - 0.03)
            accent_text = _rgb_to_hex(*colorsys.hls_to_rgb(h, hi, s))
        else:
            lo = min(0.98, lo + 0.03)
            accent_text = _rgb_to_hex(*colorsys.hls_to_rgb(h, lo, s))
    pal["accent-text"] = accent_text
    # Last-resort contrast repair: never ship unreadable body text.
    if contrast(pal["ink"], pal["canvas"]) < 7:
        pal["ink"] = "#0d0b09" if mode == "light" else "#ffffff"
    if contrast(pal["muted"], pal["canvas"]) < 4.5:
        pal["muted"] = "#4a453f" if mode == "light" else "#c3bdb6"
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
                sec["texts"].append((slot, args.get("text", ""),
                                     args.get("accent", "")))
            elif op == "ACTION":
                sec["actions"].append((slot, args.get("label", ""), args.get("href", "#")))
            elif op == "ITEM":
                sec["items"].append((args.get("title", ""), args.get("body", "")))
            elif op == "MEDIA":
                sec["media"].append((args.get("alt", ""),
                                     args.get("motif", "elevation")))
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
h1 .em,h2 .em{{color:{pal['accent-text']};font-style:italic}}
.inverted h1 .em,.inverted h2 .em{{color:{pal['accent']}}}
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

    def _accented(text: str, sub: str) -> str:
        """Escape first, then wrap the chosen substring. Escaping after
        insertion would eat the span; wrapping before escaping would let
        model text inject markup."""
        safe = _esc(text)
        if sub:
            safe_sub = _esc(sub)
            if safe_sub in safe:
                return safe.replace(safe_sub, f'<span class="em">{safe_sub}</span>', 1)
            doc.fallbacks.append(f"accent={sub!r} not found in title -> plain")
        return safe

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
        if sec["kind"] == "hero" and sec["layout"] == "split":
            alt, motif = (sec["media"][0] if sec["media"]
                          else ("Illustrative drawing", "elevation"))
            if motif not in MOTIFS:
                doc.fallbacks.append(f"motif={motif} unknown -> elevation")
                motif = "elevation"
            art = MOTIFS[motif](pal["accent"], pal["ink"], pal["muted"])
            figure = (
                f'<figure class="figure">'
                f'<svg viewBox="0 0 420 520" role="img" aria-label="{_esc(alt)}">'
                f'{art}</svg>'
                f'<figcaption>{_esc(alt)}</figcaption></figure>')
            content = f'<div class="split"><div>{col}</div>{figure}</div>'
        elif sec["items"] and sec["layout"] == "list":
            # A rule-separated numbered list instead of bordered cards.
            # Cards are the safe default and read Bootstrap-generic;
            # numerals on hairlines read editorial.
            rows = "".join(
                f'<li><span class="num">{i:02d}</span>'
                f'<h3>{_esc(t)}</h3><p>{_esc(b)}</p></li>'
                for i, (t, b) in enumerate(sec["items"], 1))
            content = f'<div{centred}>{col}</div><ol class="steps">{rows}</ol>'
        elif sec["items"]:
            cards = "".join(
                f'<div class="card"><h3>{_esc(t)}</h3><p>{_esc(b)}</p></div>'
                for t, b in sec["items"])
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

    doc = parse(Path(args.ops).read_text(encoding="utf-8"))
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
