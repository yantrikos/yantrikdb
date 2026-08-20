"""Every manifest that can name a release artifact must name the same version.

`crates/yantrikdb-python/pyproject.toml` sat at 0.15.3 while the root
pyproject and both Cargo.toml files said 0.16.0. Nothing failed: maturin
invoked as `-m crates/yantrikdb-python/Cargo.toml` reads the pyproject
ADJACENT to that manifest, so it built, tested clean, and produced
`yantrikdb-0.15.3-...whl` — a filename already published to PyPI. Building
from the repo root instead yields 0.16.0 from the same source tree.

The failure mode is the dangerous kind: no error, a valid wheel, and a
version string that is a plausible lie about which code is inside. That is
the same class as trusting `__version__` after a stale install — the reason
the deploy runbook verifies artifacts by content.

The wasm crate and the packs are versioned independently and are excluded.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Every file whose version can end up in a published artifact's name.
RELEASE_MANIFESTS = [
    "pyproject.toml",
    "crates/yantrikdb-core/Cargo.toml",
    "crates/yantrikdb-python/Cargo.toml",
    "crates/yantrikdb-python/pyproject.toml",
]

_VERSION = re.compile(r"^version\s*=\s*[\"']([^\"']+)[\"']", re.M)


def _declared(rel):
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not present in this checkout")
    m = _VERSION.search(path.read_text(encoding="utf-8"))
    assert m, f"{rel} declares no version"
    return m.group(1)


def test_all_release_manifests_declare_the_same_version():
    found = {rel: _declared(rel) for rel in RELEASE_MANIFESTS}
    distinct = set(found.values())
    assert len(distinct) == 1, (
        "release manifests disagree on the version, so which wheel you get "
        "depends on which directory you build from:\n"
        + "\n".join(f"  {v:>10}  {rel}" for rel, v in sorted(found.items()))
    )


def test_the_installed_package_matches_the_manifests():
    """The release-verification direction: an INSTALLED wheel must say what it is.

    Skipped when running against the source tree, which is not a weaker
    check but a necessary one. `__version__` resolves through
    `importlib.metadata`, i.e. installed DIST metadata, while
    `PYTHONPATH=src` supplies the CODE. Under that combination the two come
    from different places by construction and comparing them says nothing
    about the build.

    That gap is the whole reason the deploy runbook installs the published
    wheel into a clean venv before believing a feature exists — a source-tree
    run has already hidden two release-blocking bugs by answering for code
    that was never in the artifact.
    """
    yantrikdb = pytest.importorskip("yantrikdb")
    module = Path(yantrikdb.__file__).resolve()
    if ROOT in module.parents:
        pytest.skip(
            "running against the source tree: __version__ comes from installed "
            "dist metadata, not this code. Verify version against a wheel "
            "installed in a clean venv instead."
        )
    installed = getattr(yantrikdb, "__version__", None)
    if installed is None:
        pytest.skip("no __version__ on the installed package")
    declared = _declared("pyproject.toml")
    assert installed == declared, (
        f"installed yantrikdb is {installed} but the source tree declares "
        f"{declared} — the artifact under test is not the one about to ship"
    )
