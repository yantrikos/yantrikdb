#!/usr/bin/env python3
"""Resolve `photo="..."` queries in an ops file into real, licensed images.

The drawings are the reason these pages read as generated. Eight
parametric SVG motifs are honest and they are not photography, and no
amount of layout work makes a line diagram look like a bakery.

The division of labour is the same one the rest of the kit uses. Saying
what a picture should SHOW is judgement, and the model is good at it.
Knowing that a particular image exists, is licensed for use, and depicts
what its filename claims is not something a model can do at all — an
invented Flickr id is a 404 or, worse, an unrelated photograph
presented as your premises. So the model writes a query and never a
URL, and this resolves it.

    python packs/letterpress/photos.py briefs/charity.ops

Writes assets/<hash>.jpg next to the ops file and a sidecar
<name>.photos.json the compiler reads. Re-running is cheap: anything
already downloaded is left alone, so a build is not a fresh scrape.

LICENSING. Openverse indexes work under several licences and they are
not interchangeable. This accepts only the four that permit commercial
use and modification — CC0 and Public Domain Mark, which need no
attribution, and CC BY and CC BY-SA, which require it and get it
rendered on the page. NonCommercial and NoDerivatives are excluded
outright rather than left to a flag someone might flip: a site is a
commercial context often enough that the safe set is the only
defensible default, and an unattributed CC BY image is a licence breach
on the user's own domain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.openverse.org/v1/images/"
UA = "letterpress-site-compiler/0.1 (https://github.com/yantrikdb; contact via repo)"

# Some image hosts — upload.wikimedia.org among them — do not validate
# against this machine's system trust store, while others do. Use a
# current CA bundle when one is importable. Verification is NEVER
# disabled: an unverified download of a file you are about to publish on
# someone's site is a worse failure than no image, and "it worked when I
# turned the checks off" is how a supply chain gets poisoned.
try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:                                   # pragma: no cover
    SSL_CTX = ssl.create_default_context()

# Attribution is required for BY and BY-SA and the page renders it.
# CC0 and PDM do not require it; they still get a credit line, because a
# page that credits everything is easier to defend than one where the
# absence of a credit is itself a claim about the licence.
ALLOWED = {"cc0", "pdm", "by", "by-sa"}
NEEDS_CREDIT = {"by", "by-sa"}

PHOTO_ARG = re.compile(r'photo="([^"]+)"')


# Openverse ANDs every term, so a descriptive phrase matches nothing at
# all: "sourdough bread loaves on a wooden bakery counter" returns 0
# while "sourdough bread bakery" returns 240. The pack teaches short
# keyword queries for this reason, and this strips the connective tissue
# in case the model writes a sentence anyway.
STOP = {"a", "an", "the", "on", "in", "at", "of", "with", "and", "for",
        "to", "from", "by", "into", "over", "under", "near", "some",
        "its", "their", "his", "her", "it", "is", "are", "being"}


def _terms(query: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Z][a-zA-Z'-]+", query.lower())
            if w not in STOP]


def _call(q: str, want: int, strict: bool) -> list[dict]:
    params = {
        "q": q,
        "page_size": want,
        "license_type": "commercial",   # excludes every NC licence
        "mature": "false",
    }
    if strict:
        # Better hero images, at the cost of most of the index — these
        # two cut a 240-result query to 11, so they are the first thing
        # relaxed rather than the last.
        params |= {"aspect_ratio": "wide", "size": "large"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=25, context=SSL_CTX) as r:
        return json.load(r).get("results", [])


def search(query: str, want: int = 12) -> tuple[list[dict], str]:
    """Widen the search until something comes back, and say what worked.

    Returns the results and the query that actually produced them, so a
    picture on the page can always be traced to the words that found it.
    Narrowing steps, in order: the words as written, then the three most
    specific, then the two, each tried strict before relaxed.
    """
    terms = _terms(query)
    attempts = [" ".join(terms)]
    if len(terms) > 3:
        attempts.append(" ".join(terms[:3]))
    if len(terms) > 2:
        attempts.append(" ".join(terms[-2:]))
    for strict in (True, False):
        for q in attempts:
            if not q:
                continue
            hits = _call(q, want, strict)
            if pick(hits, q.split()):
                return hits, q + ("" if strict else "  (relaxed)")
    return [], query


TAGSTRIP = re.compile(r"<[^>]+>")


def clean_title(raw: str) -> str:
    """Openverse titles can carry raw markup from the upstream source.

    One Wikimedia record came through as `<div class='fn'> Medieval
    harness pendant</div>` and went straight into a caption. Titles are
    third-party text and get treated as such.
    """
    return " ".join(TAGSTRIP.sub(" ", raw or "").split())[:120]


def relevant(hit: dict, terms: list[str]) -> bool:
    """Does this image plausibly show what was asked for?

    Openverse ranks by relevance but will still return its best effort
    when nothing matches well: the query "baker hands shaping" returned
    a photograph of a medieval harness pendant, which compiled cleanly,
    passed every render gate and would have gone onto a bakery's website
    as its own kitchen. No gate can catch that, because the page is
    perfectly well-formed — it is simply a picture of the wrong thing.

    So require lexical overlap between the query and what the source
    says the picture is. This is weak — it cannot tell a good bread
    photograph from a poor one — but it is the difference between "a
    picture of bread" and "a picture of something else entirely", which
    is the failure that actually matters.
    """
    hay = " ".join([
        clean_title(hit.get("title", "")),
        " ".join(t.get("name", "") for t in (hit.get("tags") or [])),
    ]).lower()
    return any(t in hay for t in terms if len(t) > 3)


def pick(results: list[dict], terms: list[str] | None = None) -> dict | None:
    """The best usable result: licensed, big enough, and on-topic.

    Licence preference is light — a public domain image beats an
    attributed one at equal relevance, because it puts no obligation on
    the person whose site this becomes.
    """
    rank = {"cc0": 0, "pdm": 0, "by": 1, "by-sa": 2}
    usable = [
        r for r in results
        if r.get("license") in ALLOWED
        and (r.get("width") or 0) >= 1000        # enough for a 2x hero
        and r.get("url")
        and (not terms or relevant(r, terms))
    ]
    if not usable:
        return None
    usable.sort(key=lambda r: (rank.get(r["license"], 9), -(r.get("width") or 0)))
    return usable[0]


def sniff(data: bytes) -> str | None:
    """The real format, from the bytes rather than from the URL.

    A Rawpixel result served a WebP from a path ending .jpg. Saved under
    that extension it downloaded fine, compiled fine and passed every
    render gate — and showed nothing at all, because a page opened from
    disk gets its MIME type from the file extension, so the browser
    tried to decode WebP as JPEG and gave up silently. The gates cannot
    see this: an image that fails to decode still occupies its reserved
    box, so there is no layout shift and no missing element to find.
    """
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:3] == b"GIF":
        return ".gif"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return None          # TIFF: no browser renders it, reject
    return None


def fetch(url: str, dest_stem: Path) -> Path | None:
    """Download, verify it is a web image, and name it by what it is."""
    for ext in (".jpg", ".png", ".webp", ".gif"):
        existing = dest_stem.with_suffix(ext)
        if existing.exists() and existing.stat().st_size > 4096:
            return existing
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=45, context=SSL_CTX) as r:
            data = r.read()
    except Exception as exc:                      # noqa: BLE001
        print(f"    download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    if len(data) < 4096:
        return None
    ext = sniff(data)
    if not ext:
        print(f"    not a web image format: {data[:4]!r}", file=sys.stderr)
        return None
    dest = dest_stem.with_suffix(ext)
    dest.write_bytes(data)
    return dest


def credit_of(hit: dict) -> str:
    """One line naming the work, the maker and the licence."""
    lic = hit.get("license", "")
    name = clean_title(hit.get("title")) or "Untitled"
    who = (hit.get("creator") or "unknown").strip()
    if lic in ("cc0", "pdm"):
        return f"{name} — {who}, public domain"
    ver = hit.get("license_version") or ""
    return f"{name} — {who}, CC {lic.upper()} {ver}".strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ops")
    ap.add_argument("--refresh", action="store_true",
                    help="re-resolve queries already in the sidecar")
    args = ap.parse_args()

    ops = Path(args.ops)
    text = ops.read_text(encoding="utf-8")
    queries = sorted(set(PHOTO_ARG.findall(text)))
    if not queries:
        print(f"  {ops.name}: no photo= queries")
        return 0

    assets = ops.parent.parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    side = ops.with_suffix(".photos.json")
    have = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}

    for q in queries:
        if q in have and not args.refresh and (assets / have[q]["file"]).exists():
            print(f"  cached  {q!r}")
            continue
        hits, used = search(q)
        hit = pick(hits, used.replace("  (relaxed)", "").split())
        if not hit:
            print(f"  NONE    {q!r} — nothing usable under a commercial licence")
            continue
        key = hashlib.sha1(q.encode()).hexdigest()[:12]
        dest = fetch(hit["url"], assets / key)
        if dest is None:
            print(f"  FAIL    {q!r}")
            continue
        have[q] = {
            "file": dest.name,
            "width": hit.get("width"),
            "height": hit.get("height"),
            "license": hit.get("license"),
            "credit": credit_of(hit),
            "needs_credit": hit.get("license") in NEEDS_CREDIT,
            "source": hit.get("foreign_landing_url"),
            "matched": used,
        }
        note = "" if used.startswith(q.lower()[:12]) else f"  <- {used!r}"
        print(f"  got     {q!r}  {hit.get('width')}x{hit.get('height')} "
              f"CC {hit.get('license','').upper()}{note}")

    side.write_text(json.dumps(have, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {side.name}  ({len(have)} image(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
