#!/usr/bin/env python3
"""Sweep every accent a model could plausibly choose, in both modes.

Four contrast bugs shipped from derive_palette, each found by rendering
one page with one hue and looking at it. That method finds the hue you
happened to try. This one finds the hue you didn't: 4320 palettes across
the wheel, every text-carrying pair checked against AA.

It has already earned its keep. The fourth bug (accent-inv walking
toward white on a near-white inverted band in dark mode) was invisible
to every brief in the repo and would have shipped as an emphasised word
that simply is not there.

    python packs/webkit/test_palette.py
"""

from __future__ import annotations

import colorsys
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compiler import contrast, derive_palette  # noqa: E402

# Coarse enough to run in under a second, fine enough that no hue family
# is unrepresented. The lightness and saturation rows matter as much as
# the hue: the bugs clustered at the extremes, where a "sensible" accent
# never lands but a 4B's guess sometimes does.
HUES = range(0, 360, 5)
LIGHTNESS = (0.18, 0.30, 0.42, 0.55, 0.68, 0.82)
SATURATION = (0.15, 0.35, 0.60, 0.85, 1.0)


def main() -> int:
    failures: list[str] = []
    checked = 0
    for hd in HUES:
        for l in LIGHTNESS:
            for s in SATURATION:
                rgb = colorsys.hls_to_rgb(hd / 360.0, l, s)
                accent = "#%02x%02x%02x" % tuple(round(c * 255) for c in rgb)
                for mode in ("light", "dark"):
                    checked += 1
                    try:
                        derive_palette(accent, mode)
                    except AssertionError as exc:
                        failures.append(str(exc))

    print(f"{checked} palettes swept across {len(HUES)} hues x "
          f"{len(LIGHTNESS)} lightness x {len(SATURATION)} saturation x 2 modes")
    if failures:
        for f in failures[:12]:
            print(f"  {f}")
        if len(failures) > 12:
            print(f"  ... and {len(failures) - 12} more")
        print(f"FAIL: {len(failures)} palettes ship text below 4.5:1")
        return 1

    # A palette that passes by collapsing every accent to black is
    # "accessible" and useless, so check the accent stayed a colour.
    # Without this the suite would happily accept a fix that fell back
    # to the ground's opposite for every hue.
    greyed = 0
    for hd in HUES:
        pal = derive_palette(
            "#%02x%02x%02x" % tuple(
                round(c * 255) for c in colorsys.hls_to_rgb(hd / 360.0, 0.45, 0.7)),
            "light")
        r, g, b = (int(pal["accent"][i:i + 2], 16) for i in (1, 3, 5))
        if max(r, g, b) - min(r, g, b) < 24:
            greyed += 1
    if greyed:
        print(f"FAIL: {greyed} saturated accents were flattened to grey")
        return 1

    print(f"OK: every text-carrying pair clears 4.5:1, and all "
          f"{len(HUES)} saturated accents stayed chromatic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
