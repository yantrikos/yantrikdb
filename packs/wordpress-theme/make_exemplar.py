"""Append the reference theme to the corpus as worked exemplars.

Generated from the actual reference/ files rather than transcribed, so the
exemplar the pack teaches IS the artifact that scores 14/14 on the
rendered-craft benchmark. If the reference changes, this regenerates.

One record per file, and the heading names the file — because the harness
asks for one file per call, so a recall on "theme.json" lands on the
theme.json exemplar and nothing else competes with it.
"""
import pathlib

REF = pathlib.Path(r"c:\Users\sync\codes\yantrikdb\packs\wordpress-theme\reference")
CORPUS = pathlib.Path(r"c:\Users\sync\codes\yantrikdb\packs\wordpress-theme\corpus.md")

FENCE = {".json": "json", ".css": "css", ".html": "html", ".php": "php"}

# (path, heading, lead paragraph)
ITEMS = [
    ("theme.json", "A complete theme.json for a block theme — worked example",
     "Every design decision in one file, and the shape to imitate. Note what "
     "makes it work rather than merely validate: every preset declared under "
     "`settings` is *applied* under `styles`; `fluid` is true so the whole "
     "scale gets clamps; `contentSize` is 38rem because that is a reading "
     "measure rather than a layout width; the palette is four roles named by "
     "role; and `elements` plus `blocks` carry the typographic detail. This "
     "file scores full marks on a rendered-craft benchmark that measures "
     "contrast, measure, type scale, spacing scale and fluid type in a real "
     "browser."),
    ("templates/index.html", "A complete templates/index.html — worked example",
     "Block markup, not PHP. The structure to imitate is the section rhythm: "
     "a full-bleed tinted band, then a constrained reading column, then a "
     "closing band — rather than one undifferentiated stack. Note that the "
     "reading column is a group with `{\"layout\":{\"type\":\"constrained\"}}`, "
     "which is what makes `contentSize` take effect at all."),
    ("parts/header.html", "A complete parts/header.html — worked example",
     "A header with real content: site title and tagline stacked on the left, "
     "navigation on the right, in a `flex` group with `justifyContent: "
     "space-between`. No PHP — the site title is `wp:site-title` and the "
     "tagline is `wp:site-tagline`."),
    ("parts/footer.html", "A complete parts/footer.html — worked example",
     "The footer is where a dark band goes wrong. Setting `backgroundColor: "
     "contrast` without also setting `textColor` leaves near-black text on a "
     "near-black ground — invisible, and no error anywhere. Note the explicit "
     "`textColor` on the group AND on the blocks inside it, plus the "
     "`elements.link.color` override so links stay legible against the dark "
     "ground."),
    ("style.css", "A complete style.css for a block theme — worked example",
     "Only what theme.json cannot express: optical corrections, state "
     "styling, and the accessibility floors. Note the negative tracking and "
     "`text-wrap` on display type, hover colours derived with `color-mix` so "
     "they track the palette, `:focus-visible` with an offset, the "
     "`prefers-reduced-motion` block, and explicit `min-height` on inline "
     "links so tap targets reach 24px."),
    ("functions.php", "A complete functions.php for a block theme — worked example",
     "Nearly empty, and the one thing it must do is the thing that is easy to "
     "omit: a block theme's `style.css` is NOT enqueued automatically. Without "
     "this file every rule in style.css is inert — and the page still looks "
     "designed, because theme.json is carrying it, which is what makes the "
     "omission so hard to notice."),
]

out = ["\n"]
for rel, heading, lead in ITEMS:
    src = REF / rel
    body = src.read_text(encoding="utf-8").rstrip()
    lang = FENCE.get(src.suffix, "")
    out.append(f"## {heading}\n\n{lead}\n\n```{lang}\n{body}\n```\n")

CORPUS.write_text(
    CORPUS.read_text(encoding="utf-8").rstrip() + "\n" + "\n".join(out),
    encoding="utf-8")

n = CORPUS.read_text(encoding="utf-8").count("\n## ") + 1
print(f"corpus now {n} records, {len(CORPUS.read_text(encoding='utf-8').split())} words")
