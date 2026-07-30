#!/usr/bin/env python3
"""Build a theme by iteration: write a piece, verify it, repair it, continue.

`run.py` already splits generation one file per call, but every piece is
written blind of the others and nothing is checked until the whole theme is
assembled. That is where most of the remaining damage comes from, and the
failures are coordination failures rather than knowledge gaps:

    style.css written, functions.php never written  -> the stylesheet is
                                                       dead; nothing loads it
    "contentSize": 700                             -> no unit, so the
                                                       constrained layout
                                                       never applies and body
                                                       text runs 196 chars

Both pieces look fine alone. Neither is something the model does not know.
It is also why the single-shot harness looked so noisy — 77ch on one run
and 196ch on the next is not the model knowing more on Tuesday, it is
whether independently-written files happened to agree.

So this builds in dependency order and checks between steps:

    1  manifest        + required-file and manifest-gap repair
    2  theme.json      + parse, CSS-length and presets-applied repair
    3  templates/parts + written against the REAL theme.json, no-PHP repair
    4  style.css       + functions.php must enqueue it
    5  activate        + WP-CLI's own error fed back as the repair prompt
    6  render          + measured craft failures fed back once

Reported alongside the single-shot number, never instead of it. A repair
loop can carry a pack that is not teaching anything, so the unrepaired
score has to stay visible next to it.

Usage:
    python packs/theme-harness/build_iter.py --model qwen3.5:4b
    python packs/theme-harness/build_iter.py --model qwen3.5:4b --rounds 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run import (  # noqa: E402
    CHECKS, MAX_FILES, PackHost, SAFE_PATH, SPEC, SYSTEM, THEMES, ask,
    file_queries, grade, parse_manifest, reset_to_core, resolve_host,
    size_hint, strip_fences, theme_slug, wp, write_theme,
)

REQUIRED = ("style.css", "theme.json", "templates/index.html")
CSS_LEN = re.compile(r"^\s*(clamp\(.+\)|[\d.]+\s*(px|rem|em|%|vw|ch))\s*$", re.I)

# The six styles entries whose absence makes a structurally correct theme
# render as unstyled Times New Roman at line-height normal.
APPLIED = (
    ("typography", "fontFamily"), ("typography", "fontSize"),
    ("typography", "lineHeight"), ("color", "background"),
    ("color", "text"), ("spacing", "blockGap"),
)


# The 4B could not produce a valid large JSON document: the first attempt
# broke at line 2036 and the repair at line 103. That is not a knowledge
# gap, it is a consistency limit over long structured output — the same
# limit that produced a 56 KB file in the single-shot harness.
#
# So the file is not asked for as a blob. The model supplies each FIELD as
# a small array it can close correctly, and the harness assembles the
# document. The split is deliberate and worth stating plainly: the model
# owns every design decision — which colours, which ratio, which sizes,
# which measure — and the harness owns only the schema, the part that is
# mechanical and that a size-limited model gets wrong for reasons that
# have nothing to do with taste.
FIELD_ASKS = {
    "palette": (
        'A JSON array of 5 colour presets for a WordPress theme.json palette. '
        'Roles, not hues: base (page ground, an off-white), contrast (text, a '
        'near-black — never #000 on #fff, that is glare), primary '
        '(interactive), secondary (support), tint (section bands). Each entry '
        '{"slug","name","color"}. Output only the array.',
        '[{"slug":"base","name":"Base","color":"#faf7f2"}]'),
    "fontSizes": (
        'A JSON array of 6 font-size presets derived from ONE ratio (1.2, '
        '1.25 or 1.333) off a ~1.0625rem base. Slugs small, medium, large, '
        'x-large, xx-large, huge. Each entry {"slug","name","size"} with a '
        'rem size. Output only the array.',
        '[{"slug":"medium","name":"Medium","size":"1.0625rem"}]'),
    "fontFamilies": (
        'A JSON array of 2 font-family presets for a WordPress theme.json: '
        'slug "display" for headings and slug "text" for body. Use real font '
        'stacks ending in a generic family. Each entry '
        '{"slug","name","fontFamily"}. Output only the array.',
        '[{"slug":"text","name":"Text","fontFamily":"Charter, Georgia, serif"}]'),
    "layout": (
        'A JSON object with contentSize and wideSize for a WordPress '
        'theme.json. contentSize is a READING MEASURE — 45 to 75 characters '
        'per line, about 34rem to 42rem, NOT 1200px. Both need units. Output '
        'only the object.',
        '{"contentSize":"38rem","wideSize":"72rem"}'),
}


def assemble_theme_json(fields: dict) -> str:
    """Put the model's design decisions into a schema that is correct by
    construction — including the styles block, whose absence is what makes a
    theme render as unstyled Times New Roman."""
    pal = fields.get("palette") or []
    slugs = {p.get("slug") for p in pal if isinstance(p, dict)}
    base = "base" if "base" in slugs else (next(iter(slugs), "base"))
    text = "contrast" if "contrast" in slugs else base
    doc = {
        "$schema": "https://schemas.wp.org/trunk/theme.json",
        "version": 3,
        "settings": {
            "appearanceTools": True,
            "useRootPaddingAwareAlignments": True,
            "layout": fields.get("layout") or {"contentSize": "38rem",
                                               "wideSize": "72rem"},
            "color": {"defaultPalette": False, "palette": pal},
            "typography": {"fluid": True, "defaultFontSizes": False,
                           "fontSizes": fields.get("fontSizes") or [],
                           "fontFamilies": fields.get("fontFamilies") or []},
            "spacing": {"units": ["rem", "em", "%", "px"], "spacingSizes": [
                {"slug": "30", "name": "2", "size": "1rem"},
                {"slug": "40", "name": "3", "size": "1.75rem"},
                {"slug": "50", "name": "4", "size": "3rem"},
                {"slug": "60", "name": "5", "size": "5rem"}]},
        },
        "styles": {
            "color": {"background": f"var(--wp--preset--color--{base})",
                      "text": f"var(--wp--preset--color--{text})"},
            "typography": {
                "fontFamily": "var(--wp--preset--font-family--text)",
                "fontSize": "var(--wp--preset--font-size--medium)",
                "lineHeight": "1.65"},
            "spacing": {"blockGap": "var(--wp--preset--spacing--40)",
                        "padding": {"left": "var(--wp--preset--spacing--40)",
                                    "right": "var(--wp--preset--spacing--40)"}},
            "elements": {
                "link": {"color": {"text": "var(--wp--preset--color--primary)"}},
                "heading": {"typography": {
                    "fontFamily": "var(--wp--preset--font-family--display)",
                    "lineHeight": "1.12", "letterSpacing": "-0.02em"}},
            },
            # Post titles are links and would inherit the primary colour —
            # six terracotta headlines where near-black belongs. The pack's
            # pairing rule, applied mechanically: titles in contrast,
            # interactivity on hover.
            "blocks": {"core/post-title": {"elements": {"link": {
                "color": {"text": f"var(--wp--preset--color--{text})"},
                ":hover": {"color": {"text": "var(--wp--preset--color--primary)"}},
            }}}},
        },
        "templateParts": [
            {"name": "header", "title": "Header", "area": "header"},
            {"name": "footer", "title": "Footer", "area": "footer"}],
    }
    return json.dumps(doc, indent=2)


def defects_theme_json(raw: str) -> list[str]:
    try:
        tj = json.loads(raw)
    except json.JSONDecodeError as e:
        return [f"it is not valid JSON: {e.msg} at line {e.lineno}. "
                f"Every brace and bracket must close; no trailing commas."]
    out = []
    settings, styles = tj.get("settings") or {}, tj.get("styles") or {}
    layout = settings.get("layout") or {}
    for key in ("contentSize", "wideSize"):
        v = layout.get(key)
        if v is None:
            out.append(f'settings.layout.{key} is missing.')
        elif not CSS_LEN.match(str(v)):
            out.append(f'settings.layout.{key} is {v!r} — that is not a CSS '
                       f'length. It needs a unit, e.g. "38rem" or "700px". '
                       f'Without one the constrained layout never applies and '
                       f'body text runs the full width of the screen.')
    for group, key in APPLIED:
        if not (styles.get(group) or {}).get(key):
            out.append(f'styles.{group}.{key} is not set. Declaring presets '
                       f'under settings only makes them available; nothing '
                       f'renders differently until they are applied here.')
    if not (settings.get("color") or {}).get("palette"):
        out.append("settings.color.palette is missing.")
    return out


OUTLINE_BRIEF = """List the sections of the front page, top to bottom,
between the header and the footer. One line per section: a short name, a
colon, then the blocks it contains. 2 to 6 sections. Example:

intro band: heading, tagline paragraph
recent posts: post date, post title, post excerpt, pagination
closing band: heading, one button

Output only the lines, nothing else."""


def parse_outline(text: str) -> list[tuple[str, str]]:
    out = []
    for raw in strip_fences(text).splitlines():
        line = raw.strip().strip("`").lstrip("-*0123456789. ")
        if ":" in line and 3 <= len(line) <= 120:
            name, blocks = line.split(":", 1)
            out.append((name.strip().lower(), blocks.strip()))
        if len(out) >= 6:
            break
    return out


def defects_outline(outline: list[tuple[str, str]]) -> list[str]:
    """The two composition mistakes measured on real runs, caught before
    any markup exists: no post loop at all, and site identity inside the
    repeated section (which renders the site name once per post)."""
    defects = []
    joined = " ".join(n + " " + b for n, b in outline)
    if not any(w in joined for w in ("post", "article", "quer", "blog", "entries")):
        defects.append("no section lists the recent posts — the front page "
                       "must show the post query.")
    for name, blocks in outline:
        if any(w in name + blocks for w in ("post", "article", "quer"))                 and "site title" in (name + " " + blocks):
            defects.append(f'section "{name}" puts the site title inside the '
                           f'post loop — site identity belongs in the header, '
                           f'or it renders once per post.')
    return defects


def section_queries(name: str, blocks: str, index: int, total: int) -> tuple[str, ...]:
    """Retrieval vocabulary per section kind — the model's own outline
    words plus the concrete terms the composition records are named by."""
    text = f"{name} {blocks}"
    extra = (f"{name} {blocks} section block markup",)
    if any(w in text for w in ("hero", "cover", "banner", "welcome")):
        return extra + ("premium hero cover overlay display heading paired buttons",)
    if any(w in text for w in ("featured", "magazine", "grid", "card")):
        return extra + ("magazine layout featured post large card grid offset",)
    if any(w in text for w in ("quote", "pull", "testimonial", "voice")):
        return extra + ("pullquote band one voice huge tint",)
    if any(w in text for w in ("stat", "number", "metric", "count")):
        return extra + ("stats band three numbers display face dark",)
    if any(w in text for w in ("photo", "gallery", "image", "picture")):
        return extra + ("photo gallery section core gallery caption wide",)
    if any(w in text for w in ("contact", "email", "write", "reach")):
        return extra + ("contact section core no form block mailto button tint",)
    if any(w in text for w in ("feature", "column", "showcase", "media")):
        return extra + ("feature rows media-text columns grid alternate",)
    if any(w in text for w in ("post", "article", "quer", "blog", "entries")):
        return extra + ("post query section date title excerpt each exactly once",
                        "block ordering inside a post entry reading order",)
    if index == total - 1 and total > 1:
        return extra + ("closing band section heading one button full-bleed tint",)
    return extra + ("intro band section full-bleed tint eyebrow heading constrained",
                    "closing band section heading one button full-bleed tint",)


def defects_parsed(slug: str, path: str) -> list[str]:
    """Ask WordPress's own parser what is wrong with the markup.

    Not a Python reimplementation of the block grammar — that would be a
    second parser to keep in step with the first, and the point of this
    harness is that WordPress decides. It is also the only thing that
    knows the defect that produced an empty page:

        <!-- wp:post-template;{"layout":...}-->

    a semicolon where a space belongs. parse_blocks() drops the block
    entirely, so the query loop has no body and renders nothing. HTTP 200,
    no PHP error, blank page.

    Calibrated before use: 0 defects on all three reference-theme files,
    3 real defects on the 4B's. A validator that has never passed a
    known-good artifact is just an opinion.
    """
    code, out = wp("eval-file",
                   "/var/www/html/wp-content/themes/harness/validate_blocks.php",
                   f"{slug}/{path}")
    # A validation that could not RUN must never read as a validation that
    # PASSED. The first version returned [] here, so when WP-CLI failed to
    # bootstrap — a broken theme left active by an earlier run — the loop
    # printed "verified" for a template whose query loop had no body. That
    # is the sixth check in this project to fail by being silently
    # permissive, and the shape is always the same: the absence of a
    # finding treated as the absence of a problem.
    if code != 0 or "{" not in out:
        first = (out.strip().splitlines() or ["no output"])[0][:160]
        return [f"__unvalidated__ WordPress could not parse this file "
                f"({first}) — treat as unverified, not as clean."]
    try:
        payload = json.loads(out[out.index("{"):out.rindex("}") + 1])
    except (json.JSONDecodeError, ValueError):
        return ["__unvalidated__ validator output was not JSON — treat as "
                "unverified, not as clean."]
    return list(payload.get("defects") or [])


def defects_markup(path: str, body: str, manifest: list[str],
                   theme_json: str) -> list[str]:
    out = []
    if "<?php" in body:
        out.append("it contains PHP. Files under templates/ and parts/ are "
                   "never executed — the literal <?php leaves an attribute "
                   "unterminated and swallows the rest of the document. Use "
                   "blocks instead: wp:site-title, wp:site-tagline, "
                   "wp:navigation.")
    if path.startswith("templates/") and '"type":"constrained"' not in body.replace(" ", ""):
        out.append('no group uses {"layout":{"type":"constrained"}}, so '
                   'settings.layout.contentSize has no effect at all.')
    for ref in re.findall(r'wp:template-part\s*\{[^}]*"slug"\s*:\s*"([^"]+)"', body):
        if f"parts/{ref}.html" not in manifest:
            out.append(f'it references template part "{ref}" but '
                       f'parts/{ref}.html is not one of the theme files.')
    if "backgroundColor" in body and "textColor" not in body:
        out.append("it sets a backgroundColor without a textColor. On a dark "
                   "ground the text inherits the dark colour and becomes "
                   "invisible.")
    # Slugs must exist. An invented preset silently renders as nothing.
    try:
        tj = json.loads(theme_json)
    except json.JSONDecodeError:
        return out
    known = {p["slug"] for p in
             ((tj.get("settings") or {}).get("color") or {}).get("palette", [])}
    if known:
        for slug in set(re.findall(r'"(?:backgroundColor|textColor)"\s*:\s*"([^"]+)"', body)):
            if slug not in known:
                out.append(f'it uses the colour slug "{slug}", which is not in '
                           f'the theme.json palette ({", ".join(sorted(known))}).')
    return out


def repair(host, ollama, model, path: str, body: str, defects: list[str],
           extra: str = "") -> str:
    listed = "\n".join(f"- {d}" for d in defects)
    prompt = (
        f"{extra}This is the current {path}:\n\n{body}\n\n"
        f"It has these specific problems:\n{listed}\n\n"
        f"Rewrite {path} completely, fixing every problem listed and changing "
        f"nothing else. Output only the file contents — no explanation, no "
        f"code fences. {size_hint(path)}")
    out, _ = ask(ollama, model, host.reference(*file_queries(path), floor=0.42) + prompt, SYSTEM)
    return strip_fences(out)


def main() -> int:
    global SPEC
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", default=str(HERE.parent / "dist" / "wordpress-theme-0.1.0.ydbpack"))
    ap.add_argument("--rounds", type=int, default=2, help="repair attempts per file")
    ap.add_argument("--extended", action="store_true",
                    help="add gallery and contact sections to the brief")
    ap.add_argument("--premium", action="store_true",
                    help="the premium brief: cover hero, magazine layout, "
                         "pullquote and stats bands")
    ap.add_argument("--ollama")
    args = ap.parse_args()

    ollama = resolve_host(args.ollama)
    host = PackHost(Path(args.pack))
    model = args.model
    if args.premium:
        SPEC = SPEC + """
- a full-width cover hero with an image overlay, display heading and two buttons
- a magazine post layout: newest post large with its featured image, the rest as an image card grid
- one pullquote band and one stats band
- a three-column footer"""
    elif args.extended:
        # A versioned change to the task, not a silent one.
        SPEC = SPEC + """
- a photo gallery section
- a contact section (core blocks only)"""
    kind = "premium" if args.premium else ("extended" if args.extended else "iterated")
    slug = theme_slug(model, kind)
    print(f"model: {model}\nslug : {slug}\n")

    def gen(path: str, prompt_extra: str = "") -> str:
        body, _ = ask(
            ollama, model,
            host.reference(*file_queries(path), floor=0.42)
            + f"You are writing {path} for {SPEC}\n\n{prompt_extra}"
            f"Write the full contents of {path} and nothing else. No "
            f"explanation, no code fences. {size_hint(path)}",
            SYSTEM)
        return strip_fences(body)

    repairs = 0

    # ── 1. manifest ─────────────────────────────────────────────────
    raw, _ = ask(ollama, model,
                 host.reference("block theme file structure and required files", floor=0.42)
                 + f"List every file needed for {SPEC}\n\nOutput ONLY the file "
                 f"paths, one per line, relative to the theme directory.",
                 SYSTEM)
    manifest = parse_manifest(raw)
    print(f"1 manifest: {', '.join(manifest)}")

    added = [f for f in REQUIRED if f not in manifest]
    # The manifest gap that killed the last run: a stylesheet nothing loads.
    if "style.css" in manifest + added and "functions.php" not in manifest:
        added.append("functions.php")
    if added:
        # Required files go FIRST: this run the model wrote "styles.css"
        # (plural), the fix-up appended style.css — and [:MAX_FILES]
        # sliced it off the end. WordPress identifies a theme BY
        # style.css, so the activation error was "theme could not be
        # found". Required additions must survive the cap.
        manifest = (added + manifest)[:MAX_FILES]
        repairs += 1
        print(f"  repaired: added {', '.join(added)} "
              f"(required, or nothing would load style.css)")

    files: dict[str, str] = {}

    # ── 2. theme.json, field by field, assembled by the harness ─────
    fields = {}
    for name, (askfor, example) in FIELD_ASKS.items():
        got = None
        for attempt in range(args.rounds + 1):
            out, _ = ask(ollama, model,
                         host.reference(*file_queries("theme.json"))
                         + askfor
                         + "\n\nShape:\n" + example
                         + "\n\nOutput only valid JSON, nothing else.",
                         SYSTEM)
            txt = strip_fences(out)
            m = re.search(r"(\[.*\]|\{.*\})", txt, re.S)
            try:
                got = json.loads(m.group(1) if m else txt)
                break
            except (json.JSONDecodeError, AttributeError):
                repairs += 1
        if got is None:
            print(f"2 {name}: unusable after {args.rounds + 1} tries")
            continue
        fields[name] = got
        n = len(got) if isinstance(got, list) else len(got.keys())
        print(f"2 {name}: {n} entries")
    body = assemble_theme_json(fields)
    for attempt in range(args.rounds + 1):
        defects = defects_theme_json(body)
        if not defects:
            break
        if attempt == args.rounds:
            print(f"  theme.json still has {len(defects)} defect(s) after "
                  f"{args.rounds} repair(s)")
            break
        print(f"2 theme.json: {len(defects)} defect(s) -> repairing")
        for d in defects:
            print(f"    - {d[:96]}")
        body = repair(host, ollama, model, "theme.json", body, defects)
        repairs += 1
    files["theme.json"] = body + "\n"
    if not defects_theme_json(body):
        print("2 theme.json: verified")

    # ── 3. markup, written against the theme.json that actually exists ──
    ctx = (f"The theme.json for this theme is already written and is FINAL. "
           f"Use only the preset slugs it defines:\n\n{files['theme.json']}\n\n")
    # ── 3a. the front template is COMPOSED, not written whole ────────
    # An outline first (short lines — the unit a small model closes
    # correctly), validated for the two measured composition mistakes,
    # then ONE SECTION of markup at a time with retrieval vocabulary
    # matched to that section. Assembly is mechanical concatenation in
    # the model's own declared order: the model owns the composition,
    # the harness owns the stapler.
    main_tpl = next((p for p in manifest if p.startswith("templates/")), None)
    if main_tpl:
        raw, _ = ask(ollama, model,
                     host.reference("composing templates sections in order "
                                    "band reading column", floor=0.42)
                     + "You are planning the front page for " + SPEC
                     + "\n\n" + OUTLINE_BRIEF, SYSTEM)
        outline = parse_outline(raw)
        for d in defects_outline(outline):
            print(f"3a outline defect: {d}")
            raw, _ = ask(ollama, model,
                         "Your section plan:\n" + raw
                         + "\n\nProblem: " + d + "\n\n"
                         + OUTLINE_BRIEF, SYSTEM)
            outline = parse_outline(raw)
        print("3a outline: " + "; ".join(n for n, _ in outline))

        HEADER_REF = '<!-- wp:template-part {"slug":"header","tagName":"header"} /-->'
        FOOTER_REF = '<!-- wp:template-part {"slug":"footer","tagName":"footer"} /-->'

        def gen_section(name: str, blocks: str, i: int, total: int,
                        defect: str = "") -> str:
            """One section, regenerated the same way whether it is being
            written for the first time or repaired — the defect is just
            more context. Sections never contain site identity or
            navigation; those live in the header and footer parts."""
            extra = (f"\nThe previous attempt had this problem, fix it: "
                     f"{defect}\n") if defect else ""
            frag, _ = ask(
                ollama, model,
                host.reference(*section_queries(name, blocks, i, total), floor=0.42)
                + ctx
                + f'Write ONLY the block markup for one section of the front '
                f'page: "{name}" containing {blocks}. One complete wp:group '
                f'with its own background and spacing presets, 8-20 lines. '
                f'Do NOT include wp:site-title, wp:navigation, header or '
                f'footer — those exist elsewhere.{extra} No explanation, no '
                f'code fences — just this section.',
                SYSTEM)
            out = strip_fences(frag)
            if len(out.splitlines()) < 3 or "wp:group" not in out:
                return gen_section(name, blocks, i, total,
                                   "it was empty or not a wp:group") \
                    if not defect else out
            return out

        sections: list[str] = []
        for i, (name, blocks) in enumerate(outline):
            sections.append(gen_section(name, blocks, i, len(outline)))
            print(f"3b section '{name}': {len(sections[-1].splitlines())} lines")

        def assemble() -> str:
            return "\n\n".join([HEADER_REF, *sections, FOOTER_REF]) + "\n"

        def localise(defect: str) -> int:
            """Which section does a defect belong to? Match quoted block
            names or the dropped-delimiter snippet against each section's
            text; fall back to topic routing. Whole-file rewriting is NOT
            a fallback — it is how the extended run died: each full-file
            repair fixed one defect and collapsed the query loop to a
            self-closing tag, and three rounds never converged. The repair
            unit must be the same size as the generation unit."""
            for m in re.findall(r'"([a-z0-9/_-]+)"', defect):
                for idx, sec in enumerate(sections):
                    if m in sec:
                        return idx
            snip = re.search(r'dropped: "(.{10,40})', defect)
            if snip:
                for idx, sec in enumerate(sections):
                    if snip.group(1)[:20] in sec:
                        return idx
            low = defect.lower()
            if any(w in low for w in ("query", "post", "loop", "excerpt", "date")):
                for idx, sec in enumerate(sections):
                    if "wp:query" in sec or "wp:post-template" in sec:
                        return idx
            return 0

        files[main_tpl] = assemble()
        write_theme(slug, files)
        seen: dict[int, list[str]] = {}
        for round_ in range(args.rounds):
            defects = defects_parsed(slug, main_tpl)
            # Site identity belongs to the parts, never a section.
            for idx, sec in enumerate(sections):
                if "wp:site-title" in sec or "wp:navigation" in sec:
                    defects.append(f'section {idx} duplicates the header — it '
                                   f'contains site-title or navigation, which '
                                   f'live in the header part. "{outline[idx][0]}"')
            if not defects:
                print(f"3c {main_tpl}: verified section-by-section")
                break
            print(f"3c {main_tpl}: {len(defects)} defect(s) -> section-scoped repair")
            snapshot = list(sections)
            prev_count = len(defects)
            hit = set()
            for d in defects:
                idx = localise(d)
                seen.setdefault(idx, [])
                if d not in seen[idx]:
                    seen[idx].append(d)
                if idx in hit:      # one repair per section per round
                    continue
                hit.add(idx)
                name, blocks = outline[idx] if idx < len(outline) else outline[0]
                print(f"    section '{name}': {d[:96]}")
                # The FULL defect history for this section, not the latest:
                # single-defect feedback oscillated — the hero fixed
                # core/simple-button and invented core/navigation-item,
                # fixed that and reinvented the first.
                history = "; ".join(seen[idx][-4:])
                sections[idx] = gen_section(name, blocks, idx, len(outline), history)
                repairs += 1
            files[main_tpl] = assemble()
            write_theme(slug, files)
            new_count = len(defects_parsed(slug, main_tpl))
            if new_count > prev_count + 3:
                # A regeneration exploded (348 "defects" from one mangled
                # fragment). Hill-climb: revert this round's changes and
                # try again from the better state.
                print(f"3c repair made it worse ({prev_count} -> {new_count}) "
                      f"— reverting this round")
                sections[:] = snapshot
                files[main_tpl] = assemble()
                write_theme(slug, files)
        else:
            left = defects_parsed(slug, main_tpl)
            if left:
                print(f"3c {main_tpl}: {len(left)} defect(s) remain")

    for path in [p for p in manifest if p.endswith(".html") and p != main_tpl]:
        body = gen(path, ctx)
        for attempt in range(args.rounds + 1):
            defects = defects_markup(path, body, manifest, files["theme.json"])
            if not defects or attempt == args.rounds:
                break
            print(f"3 {path}: {len(defects)} defect(s) -> repairing")
            for d in defects:
                print(f"    - {d[:96]}")
            body = repair(host, ollama, model, path, body, defects, ctx)
            repairs += 1
        files[path] = body + "\n"
        if not defects_markup(path, body, manifest, files["theme.json"]):
            print(f"3 {path}: verified")

    # ── 4. the rest, with the enqueue coupling enforced ─────────────
    for path in [p for p in manifest if p not in files and not p.endswith(".png")]:
        files[path] = gen(path, ctx if path.endswith(".php") else "") + "\n"
    if "style.css" in files and "functions.php" in files:
        fn = files["functions.php"]
        if "get_stylesheet_uri" not in fn:
            print("4 functions.php: does not enqueue style.css -> repairing")
            files["functions.php"] = repair(
                host, ollama, model, "functions.php", fn,
                ["it does not enqueue style.css. A block theme's style.css is "
                 "NOT loaded automatically, so every rule in it is inert. It "
                 "must call wp_enqueue_style with get_stylesheet_uri() on the "
                 "wp_enqueue_scripts action."]) + "\n"
            repairs += 1
        else:
            print("4 functions.php: enqueues style.css")

    # style.css is how WordPress identifies a theme, and its header is
    # boilerplate, not design. Guarantee both mechanically: adopt a
    # wrongly-named stylesheet, and prepend the header if absent.
    for wrong in ("styles.css", "css/style.css", "assets/css/style.css"):
        if "style.css" not in files and wrong in files:
            files["style.css"] = files.pop(wrong)
    css = files.get("style.css", "")
    if "Theme Name" not in css[:1000]:
        header = ("/*\n"
                  "Theme Name: Harness Demo\n"
                  "Version: 1.0.0\n"
                  "License: GNU General Public License v2 or later\n"
                  "Text Domain: harness-demo\n"
                  "*/\n\n")
        files["style.css"] = header + css

    # ── 5. activate, feeding WP-CLI's own error back ────────────────
    root = write_theme(slug, files)
    reset_to_core()
    code, out = wp("theme", "activate", f"harness/{slug}")
    if code != 0 or "Success" not in out:
        first = out.strip().splitlines()[0][:200] if out.strip() else "unknown"
        print(f"5 activate FAILED: {first}")
        target = "functions.php" if "functions.php" in files else "style.css"
        files[target] = repair(
            host, ollama, model, target, files[target],
            [f"WordPress refused to activate the theme: {first}"]) + "\n"
        repairs += 1
        root = write_theme(slug, files)
        reset_to_core()
        code, out = wp("theme", "activate", f"harness/{slug}")
    print(f"5 activate: {'ok' if 'Success' in out else 'still failing'}")

    # ── 6. verify the ASSEMBLED theme, then repair ──────────────────
    # The in-loop check validated each file the instant it was written,
    # and on a Windows bind mount the container had not seen it yet — so
    # WordPress parsed a stale copy and the loop printed "verified" for a
    # template whose query loop had no body. Same lesson as the release
    # that hid behind PYTHONPATH: verify the artifact that exists, not the
    # step that produced it.
    # main_tpl is repaired section-by-section in 3c; whole-file rewriting
    # it here is the exact mechanism that collapsed the extended run.
    for path in [p for p in files if p.endswith(".html") and p != main_tpl]:
        for attempt in range(args.rounds):
            defects = defects_parsed(slug, path)
            if not defects:
                print(f"6 {path}: verified against the assembled theme")
                break
            print(f"6 {path}: {len(defects)} defect(s) -> repairing")
            for d in defects:
                print(f"    - {d[:110]}")
            files[path] = repair(host, ollama, model, path, files[path],
                                 defects, ctx) + "\n"
            repairs += 1
            write_theme(slug, files)
            reset_to_core()
            wp("theme", "activate", f"harness/{slug}")
        else:
            left = defects_parsed(slug, path)
            if left:
                print(f"6 {path}: {len(left)} defect(s) remain after "
                      f"{args.rounds} repair(s)")

    root = write_theme(slug, files)
    reset_to_core()
    wp("theme", "activate", f"harness/{slug}")

    res = grade(slug, root, files)
    print(f"\n{sum(res.values())}/{len(CHECKS)} plumbing   repairs: {repairs}")
    print(f"failed: {', '.join(k for k, v in res.items() if not v) or '-'}")

    (HERE / f"iterated-{slug}.json").write_text(json.dumps(
        {"model": model, "slug": slug, "repairs": repairs,
         "score": sum(res.values()), "checks": res,
         "files": sorted(files)}, indent=2), encoding="utf-8")
    host.close()
    print(f"\nnow run: python packs/theme-harness/look.py {slug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
