"""motion-craft's craft definition: grader + briefs for web motion.

The artifact is one self-contained HTML file (markup + <style>). Every
check is a deterministic predicate over the stylesheet TEXT — no
browser, no screenshot, no judge. That is what makes this the second
grader family proven: wordpress-theme parsed a JSON document; this
parses CSS declarations. Same contract: the functions that gate
training data grade the eval.

Every brief asks for the same three motion features (a staggered
collection, a dismissible overlay, an interactive control) so the
checker can require them unconditionally instead of guessing what the
brief implied. The product context and mood vary; the mechanical
skeleton does not.
"""
from __future__ import annotations

import re

ARTIFACT = "a single self-contained HTML file (markup plus one <style> block)"

CHECKLIST = """MECHANICAL CHECKLIST — the validator tests these exactly:
- at least one @keyframes and at least one transition
- keyframes and transitions touch ONLY transform / opacity (plus paint-only
  properties like color, background-color, box-shadow, filter on transitions) —
  never top, left, right, bottom, width, height, margin, padding, font-size
- never `transition: all`
- @media (prefers-reduced-motion: reduce) present with real overrides inside
- `linear` easing only on infinite animations (spinners); everything
  UI-visible uses ease-out / ease-in / ease-in-out / cubic-bezier
- transition durations 70-700ms; at least one micro-feedback transition <=200ms
- non-infinite keyframe animations 150-700ms; infinite ones >= 2s
- the collection enters STAGGERED: >= 3 stepped delays, 30-120ms apart
- the overlay has BOTH an enter and an exit animation (names like
  *-in/*-enter and *-out/*-exit), and the exit is SHORTER than the enter
- anything using scale() declares transform-origin somewhere
- entrance keyframes end at rest (final frame has opacity: 1 and/or
  transform: none / scale(1) / translate...(0)) or animation-fill-mode is set
- will-change, if used, only lists transform and/or opacity, never on *
- a :hover or :focus-visible rule exists with a transition behind it"""

LAYOUT_PROPS = r"(?:top|left|right|bottom|width|height|margin[a-z-]*|padding[a-z-]*|font-size)"


def _style(text: str) -> str | None:
    m = re.findall(r"<style[^>]*>(.*?)</style>", text, re.S | re.I)
    if not m:
        # fenced css or bare css fallback
        f = re.search(r"```css\s*(.*?)```", text, re.S)
        if f:
            return f.group(1)
        return None
    return "\n".join(m)


def _ms(v: str) -> float:
    v = v.strip().lower()
    if v.endswith("ms"):
        return float(v[:-2])
    if v.endswith("s"):
        return float(v[:-1]) * 1000
    return float(v)


def _strip_reduced_motion(css: str) -> str:
    """The reduced-motion override intentionally uses near-zero durations
    (0.01ms !important) to kill motion; scanning it produced 27 false
    duration failures in round 1. Duration bands apply to the motion
    users normally see, so the override block is excised before any
    duration check runs."""
    m = re.search(r"@media[^{]*prefers-reduced-motion[^{]*\{", css, re.I)
    if not m:
        return css
    depth, i = 1, m.end()
    while i < len(css) and depth:
        depth += css[i] == "{"
        depth -= css[i] == "}"
        i += 1
    return css[:m.start()] + css[i:]


def _durations(css: str, prop: str) -> list[float]:
    css = _strip_reduced_motion(css)
    out = []
    for m in re.finditer(rf"{prop}[^;{{}}]*", css, re.I):
        decl = m.group(0)
        if "linear-gradient" in decl:
            continue
        out += [_ms(d) for d in re.findall(r"(\d+\.?\d*m?s)\b", decl)]
    return out


def _keyframe_blocks(css: str) -> dict[str, str]:
    blocks = {}
    for m in re.finditer(r"@keyframes\s+([\w-]+)\s*\{", css):
        depth, i = 1, m.end()
        while i < len(css) and depth:
            depth += css[i] == "{"
            depth -= css[i] == "}"
            i += 1
        blocks[m.group(1)] = css[m.end():i - 1]
    return blocks


def _anim_decls(css: str) -> list[str]:
    return [m.group(0) for m in
            re.finditer(r"animation(?:-name)?\s*:[^;{}]*", css, re.I)]


def chk_has_motion(css):
    ok = bool(re.search(r"@keyframes", css)) and bool(re.search(r"transition", css))
    return ok, "needs at least one @keyframes and one transition"


def chk_no_layout_animation(css):
    bad = []
    for name, body in _keyframe_blocks(css).items():
        for p in re.findall(rf"\b({LAYOUT_PROPS})\s*:", body):
            bad.append(f"@keyframes {name} animates {p}")
    for m in re.finditer(r"transition(?:-property)?\s*:([^;{}]*)", css, re.I):
        for p in re.findall(rf"\b({LAYOUT_PROPS})\b", m.group(1)):
            bad.append(f"transition on {p}")
    return not bad, "; ".join(bad[:4])


def chk_no_transition_all(css):
    ok = not re.search(r"transition(?:-property)?\s*:\s*all\b", css, re.I)
    return ok, "transition: all animates properties never considered"


def chk_reduced_motion(css):
    m = re.search(r"@media[^{]*prefers-reduced-motion\s*:\s*reduce[^{]*\{(.*)", css, re.S | re.I)
    if not m:
        return False, "@media (prefers-reduced-motion: reduce) missing"
    depth, i, start = 1, 0, 0
    body = m.group(1)
    for i, ch in enumerate(body):
        depth += ch == "{"
        depth -= ch == "}"
        if not depth:
            break
    inner = body[:i]
    ok = bool(re.search(r"animation|transition", inner, re.I))
    return ok, "reduced-motion block is empty of overrides"


def chk_no_ui_linear(css):
    bad = []
    for decl in _anim_decls(css) + [m.group(0) for m in re.finditer(r"transition\s*:[^;{}]*", css, re.I)]:
        if "linear-gradient" in decl:
            continue
        if re.search(r"\blinear\b", decl) and "infinite" not in decl:
            bad.append(decl.strip()[:60])
    return not bad, "linear on UI-visible motion: " + "; ".join(bad[:3])


def chk_transition_durations(css):
    ds = _durations(css, "transition")
    bad = [d for d in ds if not 70 <= d <= 700]
    return (bool(ds) and not bad,
            f"transition durations outside 100-700ms: {sorted(set(bad))[:5]}" if bad
            else ("no transition durations found" if not ds else ""))


def chk_micro_feedback(css):
    ds = _durations(css, "transition")
    ok = any(d <= 200 for d in ds)
    return ok, "no micro-feedback transition <= 200ms"


def chk_animation_durations(css):
    bad = []
    for decl in _anim_decls(css):
        ds = [_ms(d) for d in re.findall(r"(\d+\.?\d*m?s)\b", decl)]
        if not ds:
            continue
        dur = ds[0]
        if "infinite" in decl:
            if dur < 2000:
                bad.append(f"infinite loop at {dur:.0f}ms (< 2s is anxious)")
        elif not 150 <= dur <= 700:
            bad.append(f"one-shot animation at {dur:.0f}ms")
    return not bad, "; ".join(bad[:4])


def chk_stagger(css):
    # calc(var(--i) * 60ms) is the elegant one-declaration stagger; the
    # step value is right there in the expression. Round 1's checker
    # only counted literal delays and failed two artifacts for being
    # better-engineered than the check.
    calc = re.search(
        r"(?:animation|transition)-delay\s*:\s*calc\([^)]*var\(--[\w-]+\)\s*\*\s*(\d+\.?\d*m?s)",
        css, re.I)
    if calc and 20 <= _ms(calc.group(1)) <= 150:
        return True, ""
    delays = [_ms(d) for m in re.finditer(
        r"(?:animation|transition)-delay\s*:\s*([^;{}]+)", css, re.I)
        for d in re.findall(r"(\d+\.?\d*m?s)\b", m.group(1))]
    steps = sorted({d for d in delays if d > 0})
    if len(steps) < 3:
        return False, f"stagger needs >=3 stepped delays, found {len(steps)}"
    gaps = [b - a for a, b in zip(steps, steps[1:])]
    ok = all(20 <= g <= 150 for g in gaps[:6])
    return ok, f"stagger gaps {['%.0f' % g for g in gaps[:5]]}ms (want ~30-120ms steps)"


_ENTER = r"(?:in|enter|appear|show|open|up|rise)"
_EXIT = r"(?:out|exit|leave|dismiss|hide|close|down)"


def _paired_durations(css):
    frames = _keyframe_blocks(css)
    enter_names = [n for n in frames if re.search(rf"(?:^|[-_]){_ENTER}(?:$|[-_])", n, re.I)]
    exit_names = [n for n in frames if re.search(rf"(?:^|[-_]){_EXIT}(?:$|[-_])", n, re.I)]
    def dur_of(names):
        ds = []
        for decl in _anim_decls(css):
            if any(re.search(rf"\b{re.escape(n)}\b", decl) for n in names):
                t = re.findall(r"(\d+\.?\d*m?s)\b", decl)
                if t:
                    ds.append(_ms(t[0]))
        return ds
    return enter_names, exit_names, dur_of(enter_names), dur_of(exit_names)


def chk_enter_exit_pair(css):
    en, ex, _, _ = _paired_durations(css)
    ok = bool(en) and bool(ex)
    return ok, f"needs named enter AND exit keyframes (found enter={en[:2]}, exit={ex[:2]})"


def chk_exit_faster(css):
    _, _, ed, xd = _paired_durations(css)
    if not ed or not xd:
        return False, "cannot compare: missing enter or exit animation use"
    ok = min(xd) < min(ed)
    return ok, f"exit {min(xd):.0f}ms not faster than enter {min(ed):.0f}ms"


def chk_transform_origin(css):
    if not re.search(r"scale\(", css):
        return True, ""
    ok = bool(re.search(r"transform-origin\s*:", css))
    return ok, "scale() used but no transform-origin declared"


def chk_settle(css):
    frames = _keyframe_blocks(css)
    if not frames:
        return False, "no keyframes"
    if re.search(r"animation-fill-mode|\bforwards\b|\bboth\b", css):
        return True, ""
    for name, body in frames.items():
        if re.search(rf"(?:^|[-_]){_ENTER}(?:$|[-_])", name, re.I):
            last = re.split(r"(?:100%|to)\s*\{", body)[-1]
            if not re.search(r"opacity\s*:\s*1|transform\s*:\s*(?:none|scale\(1\)|translate[XY]?\(0)", last):
                return False, f"@keyframes {name} does not end at rest and no fill-mode set"
    return True, ""


def chk_will_change(css):
    if re.search(r"\*\s*\{[^}]*will-change", css):
        return False, "will-change on the universal selector"
    for m in re.finditer(r"will-change\s*:\s*([^;{}]+)", css):
        vals = {v.strip() for v in m.group(1).split(",")}
        if not vals <= {"transform", "opacity"}:
            return False, f"will-change lists {sorted(vals)} (only transform/opacity)"
    return True, ""


def chk_hover_feedback(css):
    ok = bool(re.search(r":(?:hover|focus-visible)\b", css))
    return ok, "no :hover / :focus-visible state"


CHECKS = [
    ("has-motion", chk_has_motion),
    ("no-layout-animation", chk_no_layout_animation),
    ("no-transition-all", chk_no_transition_all),
    ("reduced-motion", chk_reduced_motion),
    ("no-ui-linear", chk_no_ui_linear),
    ("transition-durations", chk_transition_durations),
    ("micro-feedback", chk_micro_feedback),
    ("animation-durations", chk_animation_durations),
    ("stagger", chk_stagger),
    ("enter-exit-pair", chk_enter_exit_pair),
    ("exit-faster", chk_exit_faster),
    ("transform-origin", chk_transform_origin),
    ("settle-at-rest", chk_settle),
    ("will-change-discipline", chk_will_change),
    ("hover-feedback", chk_hover_feedback),
]


def grade_text(text: str):
    css = _style(text)
    if css is None:
        return 0, len(CHECKS), None
    res = {}
    for cid, fn in CHECKS:
        try:
            res[cid] = fn(css)
        except Exception as e:                                # noqa: BLE001
            res[cid] = (False, f"checker error: {e}")
    return sum(1 for ok, _ in res.values() if ok), len(CHECKS), res


def canonical(raw: str) -> str | None:
    """Keep the whole HTML file; strip surrounding prose/fences."""
    m = re.search(r"```(?:html)?\s*(<!doctype.*?|<html.*?|<[a-z].*?)```", raw, re.S | re.I)
    body = m.group(1).strip() if m else raw.strip()
    return body if _style(body) is not None else None


# ------------------------------------------------------------------ briefs

CONTEXTS = [
    "a recipe box app", "a personal finance dashboard", "a museum audio guide",
    "a code review tool", "a plant care tracker", "a flight status board",
    "a podcast player", "a language flashcards app", "a fitness log",
    "a wedding RSVP site", "a support inbox", "a photo portfolio",
    # sealed holdout contexts — never in training
    "a train timetable kiosk", "a book club platform", "a weather almanac",
    "a volunteer signup portal", "a vinyl marketplace", "a night-sky observation log",
]
COLLECTIONS = ["a card grid of items", "a vertical activity feed",
               "a search-results list", "a row of summary stat tiles"]
OVERLAYS = ["a confirmation modal", "a slide-in side panel",
            "a toast notification", "a dropdown menu from a toolbar button"]
CONTROLS = ["a primary action button", "a favorite/save toggle",
            "a segmented filter control"]
MOODS = ["calm and unhurried", "crisp and efficient", "warm and friendly",
         "precise and quietly technical"]


def _brief(ctx, col, ov, ctl, mood):
    return (f"Build the motion for {ctx}: {col} whose items enter staggered, "
            f"{ov} that enters and can be dismissed, and {ctl} with hover/press "
            f"feedback. Mood: {mood}. One self-contained HTML file with a "
            f"single <style> block; class names as if JS toggles them; no "
            f"JavaScript required.")


def train_briefs() -> list[str]:
    out = []
    for k in range(72):
        j = (29 * k) % (12 * 4 * 4 * 3 * 4)     # 29 coprime with 2304
        a, r = divmod(j, 4 * 4 * 3 * 4)
        b, r = divmod(r, 4 * 3 * 4)
        c, r = divmod(r, 3 * 4)
        d, e = divmod(r, 4)
        out.append(_brief(CONTEXTS[a], COLLECTIONS[b], OVERLAYS[c],
                          CONTROLS[d], MOODS[e]))
    assert len(set(out)) == 72
    return out


def holdout_briefs() -> list[str]:
    out = []
    for i in range(12):
        out.append(_brief(CONTEXTS[12 + i % 6], COLLECTIONS[(i * 3 + 1) % 4],
                          OVERLAYS[(i + 2) % 4], CONTROLS[(i * 2 + 1) % 3],
                          MOODS[(i + 1) % 4]))
    assert len(set(out)) == 12
    return out
