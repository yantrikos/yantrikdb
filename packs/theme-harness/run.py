#!/usr/bin/env python3
"""Execution-graded benchmark: can a model build a WordPress theme that works?

Every other harness in `packs/` grades a string. This one does not. A
model is asked to write a complete theme, the files are dropped into a
real WordPress 6.8 container, and WP-CLI is asked to activate it. The
checks are things WordPress itself answers:

    does it appear on the theme list at all
    does `wp theme activate` return success
    does `wp_is_block_theme()` say yes
    does the front page return 200
    is the debug log free of fatals
    did the template actually render (site title present in the output)

A model cannot talk its way past any of those. That is the entire point:
the wordpress-theme pack's claim is "a small model can build a working
block theme", and the only honest evidence is a working block theme.

Usage:
    python packs/theme-harness/run.py --model qwen3.5:4b
    python packs/theme-harness/run.py --model granite4:3b --model qwen3.6:27b

Requires the stack to be up:
    cd packs/theme-harness && docker compose up -d
    docker compose exec -T cli wp core install --url=http://localhost:8099 \
        --title="Pack Harness" --admin_user=admin --admin_password=admin \
        --admin_email=a@b.test --skip-email --path=/var/www/html
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKS = HERE.parent
THEMES = HERE / "themes"
RAW = HERE / "raw"
SITE = "http://localhost:8099"
DEFAULT_OLLAMA = "http://192.168.4.35:11434"

# Same gate as evaluate.py: measured, not guessed. Off-topic recall sits
# below this, on-topic above it.
MIN_SIMILARITY = 0.55

SPEC = """a WordPress theme called "Harness Demo", targeting the current
WordPress release, with:

- a home page showing the site title, a tagline, and a list of recent posts
- a header and a footer
- a colour palette and typography defined once and reused
- responsive layout without fixed-width breakpoints where avoidable
- accessible focus styles"""

# Two phases, because one response cannot hold a whole theme on a small
# model — the 4B emitted style.css, stopped cleanly, and scored 1/12 for
# a theme it had not been given room to finish.
#
# Splitting also makes the manifest measurable in its own right, and it
# turns out to be the single most diagnostic answer in the whole
# benchmark: a model that lists index.php + functions.php + header.php is
# building a 2014 classic theme, and a model that lists theme.json +
# templates/index.html is building a block theme. That distinction is
# visible before a single line of content is written.
MANIFEST_BRIEF = f"""List every file needed for {SPEC}

Output ONLY the file paths, one per line, relative to the theme directory.
No explanation, no numbering, no code fences, no blank lines."""

FILE_BRIEF = """You are writing {path} for {spec}

The complete theme consists of exactly these files:
{manifest}

Write the full contents of {path} and nothing else. No explanation, no
code fences, no file-path header — output starts with the first character
of the file. {size}"""

# A size expectation, because without one a small model does not stop. The
# 4B produced a 56,314-character theme.json in a repetition loop inventing
# [data-atomic-type] selectors until it hit the token ceiling, so the JSON
# was truncated and invalid. Knowledge was never the constraint on that
# file. Numbers are taken from the reference theme, which scores full
# marks — so they describe a sufficient file rather than a small one.
SIZE_HINT = {
    ".json": "Aim for roughly 120-160 lines. Be complete but do not invent "
             "selectors or settings beyond what the theme needs — stop when "
             "the design is expressed.",
    ".css":  "Aim for roughly 60-120 lines. Only what theme.json cannot "
             "express: optical corrections, states, accessibility.",
    ".html": "Aim for roughly 20-60 lines of block markup.",
    ".php":  "Aim for roughly 20-50 lines.",
    ".txt":  "A few short sections.",
}


# Law 2 applied to the QUERY side. A bare path is a terrible query for a
# 64-dimensional embedder: "theme.json" retrieves "Style variations are
# JSON files in styles/" at 0.653 and misses every record about colour,
# type and applying presets. The embedder matches concrete technical
# vocabulary, so the consumer has to supply it — a real agent forming a
# retrieval query would do the same rather than sending a filename.
FILE_QUERIES = {
    "theme.json": ("theme.json palette colour base contrast primary presets "
                   "applied under styles background text link",
                   "theme.json typography font sizes one ratio fluid clamp "
                   "line height elements headings",
                   "theme.json layout contentSize wideSize spacing scale "
                   "blockGap root padding aware alignments"),
    "style.css":  ("style.css optical corrections letter-spacing tracking "
                   "text-wrap balance pretty hanging punctuation",
                   "style.css focus-visible outline prefers-reduced-motion "
                   "tap target min-height accessibility",
                   "style.css hover focus states color-mix oklab derived "
                   "from the palette"),
    "templates/index.html": (
                   "block markup template full-bleed band align full "
                   "background constrained reading column section rhythm",
                   "wp:query post-template post-title post-excerpt "
                   "template-part header footer block markup"),
    "parts/header.html": (
                   "stunning header identity stacked left navigation right "
                   "hairline site title tagline",
                   "which goes with what pairing table fonts colours",),
    "parts/footer.html": (
                   "stunning footer dark band light text two columns licence "
                   "textColor base",
                   "which goes with what pairing table fonts colours",),
    "functions.php": (
                   "functions.php enqueue get_stylesheet_uri wp_get_theme "
                   "version block theme style.css not loaded automatically",),
}


def file_queries(path: str) -> tuple[str, ...]:
    for key, qs in FILE_QUERIES.items():
        if path.endswith(key):
            return qs
    return (path, f"a complete {path} worked example")


def size_hint(path: str) -> str:
    for ext, hint in SIZE_HINT.items():
        if path.endswith(ext):
            return hint
    return ""

SYSTEM = (
    "You write files that will be installed into a real WordPress "
    "installation and activated. Output only what is asked for, with no "
    "commentary. If reference material is supplied, follow it exactly — it "
    "describes the WordPress version this theme must run on."
)

# Bounded so a model that lists forty files does not cost forty calls.
MAX_FILES = 9
SAFE_PATH = re.compile(r"^[A-Za-z0-9._/-]{1,60}$")

# ── model plumbing ───────────────────────────────────────────────────

def resolve_host(explicit: str | None) -> str:
    host = explicit or os.environ.get("PACK_EVAL_OLLAMA") or DEFAULT_OLLAMA
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")


def ask(host: str, model: str, prompt: str, system: str,
        timeout: int = 1800) -> tuple[str, str]:
    """Return (content, done_reason).

    `num_predict` is large on purpose. These are thinking models: the
    first run of this harness capped generation at 6144 tokens, the
    reasoning consumed most of it, and the response was cut off after the
    first file — which the grader then scored as a model that could not
    write a theme. It was a model that was not allowed to finish.

    That is the same failure family as a grader that cannot fail: the
    number looked plausible and was measuring the harness. `done_reason`
    now travels with the answer so truncation is reported rather than
    quietly folded into the score.
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "stream": False,
        # Thinking off, in BOTH conditions equally. On a long structured
        # generation the reasoning trace competes directly with the files
        # being asked for: the 4B spent its entire 16k budget thinking and
        # emitted zero bytes of content. Measured, not assumed — the raw/
        # capture is what showed it.
        #
        # It also happens to be the honest configuration for what this
        # pack claims. A constitution puts the rules in front of the model
        # so it does not have to derive them; paying thousands of tokens
        # to re-derive them would be measuring the opposite thing. And on
        # Pi-class hardware those tokens are pure latency.
        "think": False,
        "options": {"temperature": 0.2, "num_ctx": 32768, "num_predict": 16384},
    }).encode()
    req = urllib.request.Request(f"{host}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        return (d.get("message", {}).get("content", "") or "",
                d.get("done_reason", "?"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        # Never silently score a failed call as a bad theme — an earlier
        # harness reported clean zeros while every call was erroring.
        raise SystemExit(f"model call failed for {model}: {e}")


# ── writing the candidate ────────────────────────────────────────────

def strip_fences(text: str) -> str:
    """Models add markdown fences even when told not to.

    Docking a theme for a formatting habit would measure
    instruction-following rather than WordPress knowledge, so the fences
    come off instead of costing a point.
    """
    body = text.strip()
    body = re.sub(r"^```[a-zA-Z0-9.+-]*\s*\n", "", body)
    body = re.sub(r"\n```\s*$", "", body)
    return body.strip()


def parse_manifest(text: str) -> list[str]:
    """One path per line, defensively.

    Strips a shared leading directory before validating. The 9B answered
    with a PERFECT block-theme manifest — theme.json, templates/index.html,
    parts/ — but prefixed every line with "Harness Demo/", and the
    SAFE_PATH check rejected the space, dropped every line, and scored the
    model 0/12 for a correct answer. Prefixing files with the theme
    directory is a reasonable reading of "relative to the theme directory",
    and normalising it away is mechanical, not semantic.
    """
    lines = []
    for raw in strip_fences(text).splitlines():
        line = raw.strip().strip("`").lstrip("-*0123456789. ").lstrip("./")
        line = line.split("#")[0].split("//")[0].strip()
        if line and not line.endswith("/") and ".." not in line:
            lines.append(line)

    # A first segment shared by every line is a wrapper directory, not a
    # file-layout decision. Drop it once.
    if len(lines) > 1:
        firsts = {l.split("/", 1)[0] for l in lines if "/" in l}
        if len(firsts) == 1 and all("/" in l for l in lines):
            lines = [l.split("/", 1)[1] for l in lines]

    out: list[str] = []
    for line in lines:
        if not SAFE_PATH.match(line) or line in out:
            continue
        out.append(line)
        if len(out) >= MAX_FILES:
            break
    return out


def write_theme(slug: str, files: dict[str, str]) -> Path:
    root = THEMES / slug
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    for name, body in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8", newline="\n")
    return root


# ── WordPress does the grading ───────────────────────────────────────

def wp(*args: str, timeout: int = 120) -> tuple[int, str]:
    cmd = ["docker", "compose", "exec", "-T", "cli", "wp",
           *args, "--path=/var/www/html"]
    env = {**os.environ, "MSYS_NO_PATHCONV": "1"}
    try:
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                           timeout=timeout, env=env)
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return 1, "<timed out>"


def fetch(url: str, timeout: int = 30) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 0, str(e)


def reset_to_core() -> None:
    """Return the site to a known-good theme before grading anything.

    Without this the benchmark contaminates itself. A candidate whose
    `functions.php` has a parse error stays *active* after its own run,
    and WordPress loads the active theme during WP-CLI bootstrap — so
    every later candidate failed `activates` and `appears-in-wp` with the
    previous candidate's fatal. Good themes scored 4/12 because a
    different model had written bad PHP an hour earlier.

    `--skip-themes` is the whole fix: it stops WP-CLI loading the broken
    theme, which is otherwise a deadlock — the only way to deactivate the
    theme runs code from the theme.

    The tell was a scorecard that contradicted itself: `activates` false
    while an earlier identical run had passed it. A contradiction is
    worth more attention than a low score, because only one of them can
    be a real measurement.
    """
    wp("--skip-themes", "theme", "activate", "twentytwentyfive")


CHECKS = [
    "files-emitted", "style-header", "theme-json-valid", "templates-index",
    "appears-in-wp", "activates", "is-block-theme", "front-page-200",
    "no-php-error", "renders-title", "layout-sizes", "uses-presets",
]


def grade(slug: str, root: Path, files: dict[str, str]) -> dict[str, bool]:
    r = dict.fromkeys(CHECKS, False)
    r["files-emitted"] = len(files) >= 3

    style = files.get("style.css", "")
    r["style-header"] = bool(re.search(r"^\s*Theme Name\s*:", style[:8192], re.M | re.I))

    raw = files.get("theme.json", "")
    tj: dict = {}
    try:
        tj = json.loads(raw) if raw.strip() else {}
        r["theme-json-valid"] = isinstance(tj, dict) and bool(tj)
    except json.JSONDecodeError:
        r["theme-json-valid"] = False

    r["templates-index"] = (root / "templates" / "index.html").is_file()

    # Present is not the same as valid. A 4B wrote "contentSize": 700 with
    # no unit; WordPress needs a CSS length, so the constrained layout never
    # applied and body text ran 196 characters — while this check passed,
    # because it only asked whether the key existed. Fifth lenient check
    # found in my own rubric, and the same shape as all the others.
    layout = ((tj.get("settings") or {}).get("layout") or {})
    def _len(v: object) -> bool:
        return bool(re.fullmatch(r"\s*(clamp\(.+\)|[\d.]+\s*(px|rem|em|%|vw|ch))\s*",
                                 str(v or ""), re.I))
    r["layout-sizes"] = _len(layout.get("contentSize")) and _len(layout.get("wideSize"))

    all_css = "\n".join(v for k, v in files.items()
                        if k.endswith((".css", ".html", ".json", ".php")))
    r["uses-presets"] = "--wp--preset--" in all_css

    # From here WordPress answers, not us.
    # WordPress supports one level of nesting under themes/, and reports
    # the stylesheet as "harness/<slug>" rather than the bare directory
    # name. Matching on the bare slug failed a theme that had just
    # activated successfully — a contradiction in the scorecard is worth
    # more attention than a low score, because only one of them can be a
    # real result.
    theme_dir = f"harness/{slug}"
    reset_to_core()
    code, out = wp("theme", "list", "--field=name")
    r["appears-in-wp"] = code == 0 and theme_dir in out.split()

    reset_to_core()
    code, out = wp("theme", "activate", theme_dir)
    r["activates"] = code == 0 and "Success" in out

    if r["activates"]:
        code, out = wp("eval", "echo wp_is_block_theme() ? 'BLOCK' : 'CLASSIC';")
        r["is-block-theme"] = "BLOCK" in out

        status, html = fetch(SITE + "/")
        r["front-page-200"] = status == 200
        bad = ("Fatal error", "Parse error", "Warning:", "Notice:",
               "There has been a critical error")
        r["no-php-error"] = status == 200 and not any(b in html for b in bad)
        r["renders-title"] = "Pack Harness" in html

    return r


def theme_slug(model: str, condition: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", f"{model}-{condition}".lower()).strip("-")


# ── pack context ─────────────────────────────────────────────────────

class PackHost:
    """Keeps the pack mounted so retrieval can be done PER FILE.

    The first version built one reference blob from the brief and reused it
    for every file. That is the wrong shape for a generation task: the
    harness asks for one file at a time, so the query that should drive
    retrieval is the FILE PATH, not the brief. Recalling on "theme.json"
    lands on the theme.json exemplar; recalling on the brief lands on
    whatever is nearest the brief and nothing in particular.

    The constitution is injected every time — it is the always-on tier.
    Retrieval is gated at the same 0.55 similarity floor as evaluate.py,
    because ungated injection measured 12/12 -> 5/12 on an unrelated
    control set.
    """

    def __init__(self, pack_file: Path):
        sys.path.insert(0, str(PACKS.parent / "src"))
        from yantrikdb import YantrikDB  # noqa: PLC0415
        self._td = tempfile.mkdtemp()
        self.db = YantrikDB(os.path.join(self._td, "host.ydb"), 64)
        self.db.mount_pack(str(pack_file))
        self.constitution = self.db.pack_context()

    def reference(self, *queries: str, top_k: int = 5,
                  floor: float = MIN_SIMILARITY) -> str:
        """`floor` defaults to the open-Q&A gate (0.55, measured against
        attach-harm). Generation stages may pass a lower floor: their
        queries are deliberately scoped task vocabulary, and the bands
        measured for this pack's section routes are on-topic 0.435-0.544
        vs off-topic <= 0.354 — a 0.55 floor was silently rejecting
        rank-1 on-topic exemplars, which is why small models never saw
        them."""
        parts = [self.constitution]
        seen: set[str] = set()
        for q in queries:
            for h in self.db.recall(q, top_k=top_k):
                # scores.similarity, NOT a top-level "similarity" key —
                # there is no such key, so `h.get("similarity") or 0` is
                # always 0 and the gate rejected EVERY retrieved fact.
                # Every "mounted" run before this fix used the
                # constitution alone. evaluate.py had it right; this file
                # re-implemented the read from memory instead of copying
                # the working one, and got it subtly wrong.
                if h.get("scores", {}).get("similarity", 0.0) < floor:
                    continue
                text = h.get("text", "")
                if text and text not in seen:
                    seen.add(text)
                    parts.append(text)
        return "\n\n".join(p for p in parts if p)

    def close(self) -> None:
        try:
            self.db.close()
        finally:
            shutil.rmtree(self._td, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--pack", default=str(PACKS / "dist" / "wordpress-theme-0.1.0.ydbpack"))
    ap.add_argument("--ollama")
    ap.add_argument("--keep", action="store_true", help="leave candidate themes on disk")
    args = ap.parse_args()

    ollama = resolve_host(args.ollama)
    print(f"ollama: {ollama}\nsite  : {SITE}\n")

    RAW.mkdir(exist_ok=True)
    host = PackHost(Path(args.pack))
    # Start from an empty themes dir: a candidate left over from a prior
    # run is another way for one model's bad PHP to reach another's score.
    reset_to_core()
    # Clear only the slugs THIS run regenerates. Wiping the whole
    # directory destroyed the iterated 9B theme, the hand-written
    # reference and every control variant as a side effect of re-running
    # the single-shot — an hour of evidence gone to tidy-up. Isolation
    # means owning your own artifacts, not everyone else's.
    for model in args.model:
        for condition in ("baseline", "mounted"):
            stale = THEMES / theme_slug(model, condition)
            if stale.exists():
                shutil.rmtree(stale, ignore_errors=True)
    print(f"constitution: {len(host.constitution.split())} words\n")

    rows = []
    for model in args.model:
        for condition in ("baseline", "mounted"):
            def ref(*queries: str, _c=condition) -> str:
                """Reference material for THIS file, not for the run.

                Retrieval keyed on the file path is what reaches the
                worked exemplar for that file. Keying it on the brief
                instead returns whatever is nearest the brief, which is
                nothing in particular.
                """
                if _c == "baseline":
                    return ""
                return ("Reference material for the WordPress version you "
                        "are targeting. Follow it closely, including the "
                        "worked examples:\n\n"
                        + host.reference(*queries) + "\n\n---\n\n")
            slug = theme_slug(model, condition)
            print(f"=== {model} x {condition} ===", flush=True)

            manifest_raw, done = ask(
                  ollama, model,
                  ref("block theme file structure and required files") + MANIFEST_BRIEF,
                  SYSTEM)
            (RAW / f"{slug}.manifest.txt").write_text(manifest_raw, encoding="utf-8")
            paths = parse_manifest(manifest_raw)
            print(f"    manifest ({done}): {', '.join(paths) or '(none)'}", flush=True)

            files: dict[str, str] = {}
            for path in paths:
                # Query BY FILE PATH — this is what reaches the exemplar
                # record for the file about to be written.
                body, d2 = ask(
                    ollama, model,
                    ref(path, f"a complete {path} worked example")
                    + FILE_BRIEF.format(path=path, spec=SPEC,
                                        manifest="\n".join(paths),
                                        size=size_hint(path)),
                    SYSTEM)
                content = strip_fences(body)

                # Validate and retry once. A truncated theme.json fails two
                # checks for one reason, and one extra call is cheaper than
                # reporting a model as unable to write JSON it can write.
                if path.endswith(".json") and content.strip():
                    try:
                        json.loads(content)
                    except json.JSONDecodeError as e:
                        print(f"      ! {path} invalid JSON ({e.msg}) — "
                              f"retrying once, shorter", flush=True)
                        body, d2 = ask(
                            ollama, model,
                            ref(*file_queries(path))
                            + FILE_BRIEF.format(
                                path=path, spec=SPEC,
                                manifest="\n".join(paths),
                                size="Keep it under 120 lines. It MUST be "
                                     "valid JSON: every brace and bracket "
                                     "closed, no trailing commas, no "
                                     "comments. Prefer fewer settings over "
                                     "an unterminated file."),
                            SYSTEM)
                        content = strip_fences(body)

                files[path] = content + "\n"
                if d2 not in ("stop", "?"):
                    print(f"      ! {path} {d2}", flush=True)
            (RAW / f"{slug}.files.json").write_text(
                json.dumps(files, indent=1), encoding="utf-8")

            root = write_theme(slug, files)
            res = grade(slug, root, files)
            score = sum(res.values())
            rows.append((model, condition, score, res, sorted(files), done))
            print(f"    {score}/{len(CHECKS)}  " +
                  " ".join(k for k, v in res.items() if v), flush=True)
            failed = ", ".join(k for k, v in res.items() if not v) or "—"
            print(f"    failed: {failed}\n", flush=True)

    host.close()
    reset_to_core()

    print("=" * 78)
    print(f"{'model':<18} {'condition':<10} {'score':>6}   failed checks")
    print("-" * 78)
    for model, condition, score, res, _, done in rows:
        failed = ", ".join(k for k, v in res.items() if not v) or "—"
        flag = "" if done in ("stop", "?") else f" [{done}]"
        print(f"{model:<18} {condition:<10} {score:>3}/{len(CHECKS):<2}{flag}   {failed}")
    print("=" * 78)

    (HERE / "results.json").write_text(json.dumps([
        {"model": m, "condition": c, "score": s, "checks": r, "files": f,
         "done_reason": d}
        for m, c, s, r, f, d in rows], indent=2), encoding="utf-8")
    print(f"\nwrote {HERE / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
