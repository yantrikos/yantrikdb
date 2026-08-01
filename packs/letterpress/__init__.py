"""letterpress — a site compiler a small model can drive.

The model writes lines in a seven-operation language; everything that
has to be *computed* rather than judged happens here: type scale, grid,
the whole palette derived from a single accent hex, contrast solving,
responsive collapse, drawings, and licensed photograph resolution.

This package exists because the knowledge pack that teaches the language
is inert without it. An independent review put it plainly: a pack
teaching a private op language for a compiler nobody has is packaging,
and a dependency note in the description does not fix that. So the
compiler ships as an installable, version-pinned artifact, and the pack
declares which version it was written against.

    pip install yantrik-letterpress
    letterpress compile site.ops --out site.html --strict
"""

__version__ = "0.1.0"

# The pack build this compiler is contractually paired with. The pack
# names the same string, so a mismatch is detectable rather than
# something a user discovers through pages that quietly come out wrong.
PACK_CONTRACT = "yantrik/letterpress@0.1.0"
