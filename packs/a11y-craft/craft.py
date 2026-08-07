"""a11y-craft's grader: structural accessibility checks over HTML.

Third grader family in the line — wordpress-theme parses a JSON
document, motion-craft parses CSS declarations, this parses a DOM. It
uses html.parser from the standard library rather than a dependency,
because a grader that needs an install is a grader a buyer will not run.

Every check is a property of the markup, decidable without a browser and
without a model. Deliberately NOT checked here: computed colour
contrast, which needs layout and cascade resolution to do honestly — a
regex over hex pairs would produce a confident number that is wrong
whenever a colour comes from a variable, and a check that is wrong in
the flattering direction is worse than no check.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser

ARTIFACT = "a single self-contained HTML file (markup plus one <style> block)"

CHECKLIST = """MECHANICAL CHECKLIST — the validator tests these exactly:
- <html lang="..."> present, and a non-empty <title>
- exactly one <h1>; heading levels never skip (h2 -> h4 is a failure)
- landmarks present: <header>, <nav>, <main>, <footer>, with exactly one <main>
- every <input>/<select>/<textarea> that is not hidden/submit/button has a
  programmatic name: <label for=ID>, a wrapping <label>, or aria-label /
  aria-labelledby. A placeholder is NOT a name
- personal-data inputs carry autocomplete (name, email, tel, street-address,
  postal-code, current-password, one-time-code ...)
- at least one input declares inputmode or a specific type (email/tel/url/number/search)
- an invalid field uses aria-invalid AND aria-describedby pointing at an existing id
- no <div>/<span> carrying onclick or role="button"; use <button> / <a href>
- every <img> has an alt attribute (alt="" is correct for decorative), and no
  alt text begins with "image of" / "picture of"
- no positive tabindex anywhere
- a :focus-visible (or :focus) rule exists with a visible outline/box-shadow, and
  outline:none is never left without a replacement in the same rule
- a live region exists: role="status", role="alert" or aria-live
- any dialog has role="dialog", aria-modal="true" and aria-labelledby
- data tables use <th scope> and a <caption>"""


class Dom(HTMLParser):
    """Enough of a DOM for structural questions: tags, attributes, ids,
    text under each element, and parent chains for the label-wrapping
    case."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict, int]] = []   # (tag, attrs, depth)
        self.stack: list[tuple[str, dict]] = []
        self.ids: set[str] = set()
        self.text_of: dict[int, str] = {}
        self.labels_wrapping: list[tuple[dict, list[str]]] = []
        self._label_depth: int | None = None
        self._label_children: list[str] = []
        self._label_attrs: dict = {}
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        self.tags.append((tag, a, len(self.stack)))
        if a.get("id"):
            self.ids.add(a["id"])
        if tag == "title":
            self._in_title = True
        if tag == "label" and self._label_depth is None:
            self._label_depth, self._label_children, self._label_attrs = len(self.stack), [], a
        elif self._label_depth is not None and tag in ("input", "select", "textarea"):
            self._label_children.append(tag)
        if tag not in ("img", "input", "br", "hr", "meta", "link", "source"):
            self.stack.append((tag, a))

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if self._label_depth is not None and tag == "label" and len(self.stack) - 1 <= self._label_depth:
            self.labels_wrapping.append((self._label_attrs, self._label_children))
            self._label_depth = None
        while self.stack and self.stack[-1][0] != tag:
            self.stack.pop()
        if self.stack:
            self.stack.pop()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self.stack:
            i = len(self.tags) - 1
            self.text_of[i] = self.text_of.get(i, "") + data

    def find(self, tag):
        """Always (tag, attrs) pairs. An earlier version returned bare
        attrs from one call site and pairs from another, which the
        planted-good document caught immediately — a checker error
        scores as a FAIL, so an inconsistent helper reads as the
        artifact's fault rather than the grader's."""
        return [(t, a) for t, a, _ in self.tags if t == tag]

    def attrs_anywhere(self, name):
        return [a for _, a, _ in self.tags if name in a]


def _style(text: str) -> str:
    return "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", text, re.S | re.I))


def _named(a: dict, dom: Dom) -> bool:
    if a.get("aria-label", "").strip():
        return True
    if a.get("aria-labelledby", "").strip():
        return True
    if a.get("id") and any(la.get("for") == a["id"] for _, la in dom.find("label")):
        return True
    return False


NEEDS_NAME = {"input", "select", "textarea"}
SKIP_TYPES = {"hidden", "submit", "button", "reset", "image"}
PERSONAL = re.compile(r"name|email|tel|phone|address|postal|zip|password|card|otp|code|city|country",
                      re.I)


def chk_lang_title(dom, css, raw):
    html_tags = dom.find("html")
    lang = bool(html_tags and html_tags[0][1].get("lang", "").strip())
    title = bool(dom.title.strip())
    if lang and title:
        return True, ""
    miss = [n for n, ok in (("<html lang>", lang), ("<title>", title)) if not ok]
    return False, "missing " + ", ".join(miss)


def chk_one_h1(dom, css, raw):
    n = len(dom.find("h1"))
    return n == 1, f"{n} <h1> elements (want exactly 1)"


def chk_heading_order(dom, css, raw):
    levels = [int(t[1]) for t, _, _ in dom.tags if re.fullmatch(r"h[1-6]", t)]
    for prev, cur in zip(levels, levels[1:]):
        if cur > prev + 1:
            return False, f"heading jumps h{prev} -> h{cur}"
    return bool(levels), "no headings at all" if not levels else ""


def chk_landmarks(dom, css, raw):
    have = {t for t, _, _ in dom.tags}
    missing = [t for t in ("header", "nav", "main", "footer") if t not in have]
    mains = len(dom.find("main"))
    if missing:
        return False, "missing landmark(s): " + ", ".join(missing)
    return mains == 1, f"{mains} <main> elements (want exactly 1)"


def chk_inputs_named(dom, css, raw):
    wrapped = sum(len(kids) for _, kids in dom.labels_wrapping)
    unnamed = []
    for tag, a, _ in dom.tags:
        if tag not in NEEDS_NAME:
            continue
        if a.get("type", "text").lower() in SKIP_TYPES:
            continue
        if not _named(a, dom):
            unnamed.append(a.get("name") or a.get("id") or a.get("type") or tag)
    # a wrapping label names its own child, so forgive that many
    unnamed = unnamed[wrapped:]
    return not unnamed, "inputs with no programmatic label: " + ", ".join(unnamed[:4])


def chk_autocomplete(dom, css, raw):
    want, have = [], []
    for tag, a, _ in dom.tags:
        if tag != "input" or a.get("type", "text").lower() in SKIP_TYPES:
            continue
        key = f"{a.get('name','')} {a.get('id','')} {a.get('type','')}"
        if PERSONAL.search(key):
            want.append(key.strip())
            if a.get("autocomplete", "").strip():
                have.append(key.strip())
    if not want:
        return True, ""
    return len(have) == len(want), \
        f"{len(want) - len(have)} personal-data input(s) without autocomplete"


def chk_input_purpose(dom, css, raw):
    ok = any(a.get("inputmode") or a.get("type", "").lower() in
             ("email", "tel", "url", "number", "search", "date")
             for t, a, _ in dom.tags if t == "input")
    return ok, "no input declares a specific type or inputmode"


def chk_error_association(dom, css, raw):
    invalid = [a for a in dom.attrs_anywhere("aria-invalid")
               if a.get("aria-invalid", "").lower() == "true"]
    described = [a for a in dom.attrs_anywhere("aria-describedby")]
    if not invalid and not described:
        return False, "no aria-invalid / aria-describedby error association anywhere"
    dangling = [a["aria-describedby"] for a in described
                if not all(i in dom.ids for i in a["aria-describedby"].split())]
    if dangling:
        return False, f"aria-describedby points at missing id: {dangling[0]}"
    return bool(invalid), "aria-invalid never used on an errored field"


def chk_real_controls(dom, css, raw):
    bad = []
    for tag, a, _ in dom.tags:
        if tag in ("div", "span", "li", "p"):
            if "onclick" in a:
                bad.append(f"<{tag} onclick>")
            elif a.get("role", "").lower() in ("button", "link"):
                bad.append(f'<{tag} role="{a["role"]}">')
    return not bad, "fake controls: " + ", ".join(bad[:4])


def chk_img_alt(dom, css, raw):
    imgs = dom.find("img")
    missing = [a.get("src", "?")[:30] for _, a in imgs if "alt" not in a]
    if missing:
        return False, f"{len(missing)} <img> without alt: {missing[0]}"
    bad = [a["alt"] for _, a in imgs
           if re.match(r"\s*(image|picture|photo|graphic) of", a.get("alt", ""), re.I)]
    return not bad, f'alt starts with a redundant phrase: "{bad[0][:40]}"' if bad else ""


def chk_no_positive_tabindex(dom, css, raw):
    bad = [a["tabindex"] for a in dom.attrs_anywhere("tabindex")
           if a["tabindex"].strip().lstrip("+").isdigit() and int(a["tabindex"]) > 0]
    return not bad, f"positive tabindex: {bad[:3]}"


def chk_focus_visible(dom, css, raw):
    rules = re.findall(r"([^{}]*:focus(?:-visible)?[^{}]*)\{([^}]*)\}", css, re.I)
    if not rules:
        return False, "no :focus or :focus-visible rule"
    for sel, body in rules:
        if re.search(r"outline\s*:\s*(?!none|0)", body) or "box-shadow" in body:
            return True, ""
    return False, "focus rules exist but declare no visible indicator"


def chk_no_bare_outline_none(dom, css, raw):
    for sel, body in re.findall(r"([^{}]+)\{([^}]*)\}", css):
        if re.search(r"outline\s*:\s*(none|0)\b", body):
            if not (re.search(r"box-shadow|outline-offset|border", body)
                    or ":focus" not in sel):
                return False, f"outline removed with no replacement in `{sel.strip()[:40]}`"
    return True, ""


def chk_live_region(dom, css, raw):
    ok = any(a.get("role", "").lower() in ("status", "alert") or "aria-live" in a
             for _, a, _ in dom.tags)
    return ok, "no role=status / role=alert / aria-live container"


def chk_dialog(dom, css, raw):
    dialogs = [a for t, a, _ in dom.tags
               if t == "dialog" or a.get("role", "").lower() == "dialog"]
    if not dialogs:
        return True, ""
    for a in dialogs:
        if a.get("aria-modal", "").lower() != "true":
            return False, "dialog without aria-modal=true"
        if not (a.get("aria-labelledby") or a.get("aria-label")):
            return False, "dialog with no accessible name"
    return True, ""


def chk_table_semantics(dom, css, raw):
    if not dom.find("table"):
        return True, ""
    if not dom.find("th"):
        return False, "table with no <th>"
    if not any("scope" in a for _, a in dom.find("th")):
        return False, "<th> without scope"
    return bool(dom.find("caption")), "table without <caption>"


CHECKS = [
    ("lang-and-title", chk_lang_title),
    ("one-h1", chk_one_h1),
    ("heading-order", chk_heading_order),
    ("landmarks", chk_landmarks),
    ("inputs-named", chk_inputs_named),
    ("autocomplete", chk_autocomplete),
    ("input-purpose", chk_input_purpose),
    ("error-association", chk_error_association),
    ("real-controls", chk_real_controls),
    ("img-alt", chk_img_alt),
    ("no-positive-tabindex", chk_no_positive_tabindex),
    ("focus-visible", chk_focus_visible),
    ("outline-replacement", chk_no_bare_outline_none),
    ("live-region", chk_live_region),
    ("dialog-semantics", chk_dialog),
    ("table-semantics", chk_table_semantics),
]


def grade_text(text: str):
    body = canonical(text) or text
    if "<" not in body:
        return 0, len(CHECKS), None
    dom = Dom()
    try:
        dom.feed(body)
    except Exception:                                          # noqa: BLE001
        return 0, len(CHECKS), None
    css = _style(body)
    res = {}
    for cid, fn in CHECKS:
        try:
            res[cid] = fn(dom, css, body)
        except Exception as e:                                 # noqa: BLE001
            res[cid] = (False, f"checker error: {e}")
    return sum(1 for ok, _ in res.values() if ok), len(CHECKS), res


def canonical(raw: str) -> str | None:
    m = re.search(r"```(?:html)?\s*(<!doctype.*?|<html.*?)```", raw, re.S | re.I)
    body = m.group(1).strip() if m else raw.strip()
    return body if "<" in body else None


# ------------------------------------------------------------------ briefs

SUBJECTS = [
    "a clinic appointment booking form", "a bank transfer screen",
    "a university course enrolment page", "a job application form",
    "a charity donation page", "a rental application",
    "a government benefits claim", "a conference registration",
    "an insurance quote form", "a pharmacy prescription refill",
    "a library card signup", "a utility account setup",
    # sealed holdout subjects — never in training
    "a voter registration check", "a hospital pre-admission form",
    "a legal aid intake", "a school enrolment form",
    "a disability grant application", "a food bank referral",
]
SHAPES = [
    "a multi-step form with a progress indicator",
    "a single long form grouped into fieldsets",
    "a form beside a summary panel that updates",
    "a form with a confirmation dialog before submit",
]
STATES = [
    "show two fields in an error state with messages",
    "show a success confirmation after submit",
    "show a required field left empty and flagged",
]
TONES = ["plain and official", "warm and reassuring",
         "compact and efficient", "careful and unhurried"]


def _brief(s, sh, st, t):
    return (f"Build {s} as one self-contained HTML page. Structure: {sh}. "
            f"State to render: {st}. Tone: {t}. Include a data table of "
            f"submitted entries, an image, and a status message region. "
            f"One <style> block, no JavaScript required, real copy.")


def train_briefs() -> list[str]:
    out = []
    for k in range(72):
        j = (37 * k) % (12 * 4 * 3 * 4)          # 37 coprime with 576
        a, r = divmod(j, 4 * 3 * 4)
        b, r = divmod(r, 3 * 4)
        c, d = divmod(r, 4)
        out.append(_brief(SUBJECTS[a], SHAPES[b], STATES[c], TONES[d]))
    assert len(set(out)) == 72
    return out


def holdout_briefs() -> list[str]:
    out = [_brief(SUBJECTS[12 + i % 6], SHAPES[(i * 3 + 1) % 4],
                  STATES[(i + 2) % 3], TONES[(i * 2 + 1) % 4]) for i in range(12)]
    assert len(set(out)) == 12
    return out
