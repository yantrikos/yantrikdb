"""One entry point, so the kit is installable rather than a folder of scripts.

Each subcommand delegates to the module that already owns the work,
by rewriting argv and calling its main(). That is deliberately dumb:
those modules are also run directly during development, and giving the
CLI its own argument parsing would create a second definition of every
flag that could drift from the first.

`shoot` is the exception that needs a third-party package, so it is an
optional extra and says so plainly rather than dying on an ImportError
three frames deep.
"""

from __future__ import annotations

import sys

from . import PACK_CONTRACT, __version__

USAGE = f"""letterpress {__version__} — a site compiler a small model can drive
paired with pack {PACK_CONTRACT}

  letterpress compile <ops> --out <html> [--strict]
      Compile an ops script into one self-contained page.

  letterpress photos <ops>
      Resolve every photo="..." query in an ops script against openly
      licensed images, download them, and write the sidecar the
      compiler reads. Run this BEFORE compile, or photo queries fall
      back to drawings.

  letterpress shoot <html>...
      Render each page at desktop and mobile and run the gates:
      content, overflow, contrast, readable size, one h1, visible text.
      Needs the `verify` extra:  pip install letterpress[verify]

  letterpress version
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0

    cmd, rest = argv[0], argv[1:]

    if cmd == "version":
        print(f"letterpress {__version__}  (pack {PACK_CONTRACT})")
        return 0

    if cmd == "compile":
        from . import compiler
        sys.argv = ["letterpress compile", *rest]
        return compiler.main()

    if cmd == "photos":
        from . import photos
        sys.argv = ["letterpress photos", *rest]
        return photos.main()

    if cmd == "shoot":
        try:
            from . import shoot
        except ImportError as exc:
            print(f"letterpress shoot needs the verify extra: {exc}\n"
                  f"    pip install 'letterpress[verify]'\n"
                  f"    python -m playwright install chromium",
                  file=sys.stderr)
            return 2
        sys.argv = ["letterpress shoot", *rest]
        return shoot.main() or 0

    print(f"unknown command {cmd!r}\n\n{USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
