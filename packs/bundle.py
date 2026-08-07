#!/usr/bin/env python3
"""Package, sign, install and mount a compiled capability (.ycap).

The knowledge tier already has this: seal_pack -> sign_pack ->
install_pack, packs living beside the database, remounted on every open.
This is the same lifecycle for the weights tier, deliberately mirroring
those decisions rather than inventing new ones:

  BESIDE THE DATABASE, NOT A GLOBAL CACHE. Caps install to
        <db_stem>.caps/ exactly as packs install to <db_stem>.packs/, so
        a database plus its knowledge plus its capabilities copy, move
        and back up as one unit.

  DIGEST-BOUND, AND A MISMATCH IS A REFUSAL. A LoRA is a delta against
        one specific base; applied to a different one it is noise
        wearing a capability's name. The manifest records the base repo
        AND its snapshot revision, and mount refuses on mismatch — the
        same shape as PackEmbedderMismatch, which refuses rather than
        guessing when it cannot prove two things share a space.

  A BROKEN CAP NEVER BLOCKS. list/mount skip what they cannot verify and
        say why, keeping the record so it can be reinstalled. An engine
        held hostage by a third-party file is worse than one that loses
        a capability.

  THE CERTIFICATE TRAVELS WITH THE ARTIFACT. Efficacy numbers are
        recorded against the exact sealed brief set and grader digest
        that produced them, signed over the content digest. Re-train the
        adapter and the certificate no longer applies, by construction.

USAGE
    python packs/bundle.py build motion-craft --tag v1 [--key <hex>]
    python packs/bundle.py verify dist/motion-craft-0.1.0.ycap
    python packs/bundle.py install dist/motion-craft-0.1.0.ycap --db mem.db
    python packs/bundle.py list --db mem.db
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RUNS = HERE / "training"
DIST = HERE / "dist"

MANIFEST = "capability.json"
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")


def digest_of(paths: list[Path]) -> str:
    """Content digest over the adapter's bytes, in a fixed order.

    Names are hashed alongside contents so a renamed file cannot pass as
    the same artifact, and the order is sorted rather than filesystem
    order so the digest is reproducible on any machine.
    """
    h = hashlib.blake2b(digest_size=32)
    for p in sorted(paths, key=lambda x: x.name):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def base_identity(base: str) -> dict:
    """Repo id plus the resolved snapshot revision.

    The revision is what actually pins the weights: "Qwen/Qwen3.5-4B" is
    a moving target, `851bf6e8…` is not. Read from the local HF cache so
    it records what was really trained against, not what was asked for.
    """
    ident = {"repo": base, "revision": None}
    cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{base.replace('/', '--')}"
    ref = cache / "refs" / "main"
    if ref.exists():
        ident["revision"] = ref.read_text(encoding="utf-8").strip()
    return ident


def canonical_bytes(manifest: dict) -> bytes:
    unsigned = {k: v for k, v in manifest.items() if k != "signature"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()


def read_efficacy(pack: str) -> dict | None:
    """The five-row table, if this pack has been measured.

    A listing number that is not tied to a sealed brief set and a named
    grader is marketing, so the certificate carries the arm scores, the
    brief count and the grader's own digest — re-authoring a check
    invalidates the claim it produced.
    """
    for name in (f"efficacy-craft-{pack.replace('-craft', '')}.json",
                 "efficacy-craft.json"):
        f = HERE / name
        if not f.exists():
            continue
        rows = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(rows, dict):
            continue
        out = {}
        for arm, items in rows.items():
            if not items:
                continue
            total = items[0]["total"]
            out[arm] = {
                "mean": round(sum(i["passed"] for i in items) / len(items), 2),
                "full_pass": sum(1 for i in items if i["passed"] == total),
                "n": len(items), "checks": total,
            }
        if out:
            return {"source": name, "arms": out}
    return None


def grader_digest(pack: str) -> str | None:
    src = HERE / pack / "craft.py"
    if not src.exists():
        return None
    return hashlib.blake2b(src.read_bytes(), digest_size=16).hexdigest()


# ------------------------------------------------------------------- build

def do_build(pack: str, tag: str, key: str | None, version: str) -> int:
    src_pack = pack if (HERE / pack).exists() else pack.replace("-craft", "")
    run_dir = RUNS / (pack if (RUNS / pack).exists() else f"{pack}-craft") / tag
    if not run_dir.exists():
        print(f"no adapter at {run_dir}", file=sys.stderr)
        return 2
    files = [run_dir / f for f in ADAPTER_FILES]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        print(f"adapter incomplete, missing {missing}", file=sys.stderr)
        return 2

    compile_meta = {}
    cj = run_dir / "compile.json"
    if cj.exists():
        compile_meta = json.loads(cj.read_text(encoding="utf-8"))

    manifest = {
        "format": "ycap/1",
        "name": src_pack,
        "version": version,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base": base_identity(compile_meta.get("base", "Qwen/Qwen3.5-4B")),
        "adapter": {
            "kind": "lora",
            "rank": compile_meta.get("rank"),
            "alpha": compile_meta.get("alpha"),
            "bytes": sum(f.stat().st_size for f in files),
            "content_digest": digest_of(files),
        },
        "training": {
            "examples": compile_meta.get("examples"),
            "steps": compile_meta.get("steps"),
            "train_loss": compile_meta.get("train_loss"),
            "grader_digest": grader_digest(src_pack),
        },
        "efficacy": read_efficacy(pack),
        "requires": {
            "constitution": (HERE / src_pack / "constitution.md").exists(),
            "corpus_pack": bool(sorted(DIST.glob(f"{src_pack}-*.ydbpack"))),
        },
    }

    if key:
        from yantrikdb import YantrikDB
        manifest["signature"] = {
            "publisher_pubkey": YantrikDB.pubkey_of(key),
            "value": YantrikDB.sign_bytes(key, canonical_bytes(manifest)),
        }

    DIST.mkdir(parents=True, exist_ok=True)
    out = DIST / f"{src_pack}-{version}.ycap"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(MANIFEST, json.dumps(manifest, indent=2))
        for f in files:
            z.write(f, f.name)
        con = HERE / src_pack / "constitution.md"
        if con.exists():
            # The source of the compiled behaviour ships with it. A
            # buyer can read what the weights were taught, and a future
            # recompile for another base has the input it needs.
            z.write(con, "constitution.md")

    print(f"built {out.name}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  base      {manifest['base']['repo']} @ {(manifest['base']['revision'] or '?')[:12]}")
    print(f"  adapter   rank {manifest['adapter']['rank']}, "
          f"digest {manifest['adapter']['content_digest'][:16]}…")
    if manifest["efficacy"]:
        for arm, v in manifest["efficacy"]["arms"].items():
            print(f"  {arm:<16} {v['mean']}/{v['checks']}  full-pass {v['full_pass']}/{v['n']}")
    print(f"  signed    {'yes' if key else 'NO — unsigned, installs at lowest trust'}")
    return 0


# ------------------------------------------------------------ verify/install

def load_manifest(cap: Path) -> dict:
    with zipfile.ZipFile(cap) as z:
        return json.loads(z.read(MANIFEST))


def do_verify(cap: Path, expect_pubkey: str | None) -> int:
    man = load_manifest(cap)
    with zipfile.ZipFile(cap) as z, __import__("tempfile").TemporaryDirectory() as td:
        paths = []
        for f in ADAPTER_FILES:
            z.extract(f, td)
            paths.append(Path(td) / f)
        actual = digest_of(paths)
    if actual != man["adapter"]["content_digest"]:
        print("FAIL: adapter bytes do not match the manifest digest", file=sys.stderr)
        return 1
    sig = man.get("signature")
    if not sig:
        print(f"UNSIGNED: {man['name']}@{man['version']} — content digest OK, "
              f"no publisher claim to verify")
        return 0
    from yantrikdb import YantrikDB
    if not YantrikDB.verify_bytes(sig["publisher_pubkey"], canonical_bytes(man), sig["value"]):
        print("FAIL: signature does not verify — manifest tampered or forged", file=sys.stderr)
        return 1
    if expect_pubkey and sig["publisher_pubkey"] != expect_pubkey:
        print("FAIL: validly signed, but by a DIFFERENT publisher than expected",
              file=sys.stderr)
        return 1
    print(f"OK: {man['name']}@{man['version']} signed by {sig['publisher_pubkey'][:16]}…")
    print(f"    base {man['base']['repo']} @ {(man['base']['revision'] or '?')[:12]}")
    if man.get("efficacy"):
        for arm, v in man["efficacy"]["arms"].items():
            print(f"    {arm:<16} {v['mean']}/{v['checks']}")
    return 0


def cap_dir(db_path: str) -> Path:
    """Beside the database, matching pack_dir()'s convention."""
    p = Path(db_path)
    return p.with_name(p.stem + ".caps")


def do_install(cap: Path, db_path: str) -> int:
    if do_verify(cap, None) != 0:
        print("refusing to install an artifact that does not verify", file=sys.stderr)
        return 1
    man = load_manifest(cap)
    dest = cap_dir(db_path) / man["name"]
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(cap) as z:
        z.extractall(dest)
    print(f"installed {man['name']}@{man['version']} -> {dest}")
    print(f"  mount with: python packs/cap.py mount {man['name']}")
    return 0


def do_list(db_path: str) -> int:
    d = cap_dir(db_path)
    if not d.exists():
        print(f"no capabilities installed beside {db_path}")
        return 0
    print(f"capabilities beside {db_path}:")
    for sub in sorted(d.iterdir()):
        mf = sub / MANIFEST
        if not mf.exists():
            print(f"  {sub.name:<24} (no manifest — reinstall)")
            continue
        man = json.loads(mf.read_text(encoding="utf-8"))
        eff = man.get("efficacy") or {}
        comp = (eff.get("arms") or {}).get("compiled", {})
        score = f"{comp.get('mean')}/{comp.get('checks')}" if comp else "unmeasured"
        trust = "signed" if man.get("signature") else "unsigned"
        print(f"  {man['name']:<24} v{man['version']:<8} {trust:<9} "
              f"{man['base']['repo'].split('/')[-1]:<14} {score}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("pack")
    b.add_argument("--tag", default="v1")
    b.add_argument("--version", default="0.1.0")
    b.add_argument("--key", help="publisher secret key hex (from generate_pack_keypair)")

    v = sub.add_parser("verify")
    v.add_argument("cap", type=Path)
    v.add_argument("--pubkey", help="expected publisher key")

    i = sub.add_parser("install")
    i.add_argument("cap", type=Path)
    i.add_argument("--db", required=True)

    l = sub.add_parser("list")
    l.add_argument("--db", required=True)

    a = ap.parse_args()
    if a.cmd == "build":
        return do_build(a.pack, a.tag, a.key, a.version)
    if a.cmd == "verify":
        return do_verify(a.cap, a.pubkey)
    if a.cmd == "install":
        return do_install(a.cap, a.db)
    if a.cmd == "list":
        return do_list(a.db)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
