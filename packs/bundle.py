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
import os
import re
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
GGUF_NAME = "adapter.gguf"
def _ollama_url() -> str:
    """Where to DIAL Ollama.

    Not plain OLLAMA_HOST. That variable is conventionally a *bind*
    address — `0.0.0.0` on a machine that serves — and dialling it fails
    while looking like Ollama is simply absent. evaluate.py carries the
    same warning after the same bug; I inherited OLLAMA_HOST anyway and
    got "not reachable at http://0.0.0.0:11434" on a machine where
    Ollama was running the whole time.

    So: an explicit YANTRIK_OLLAMA wins, OLLAMA_HOST is used only when
    it names a real host, and the wildcard falls back to loopback.
    """
    raw = (os.environ.get("YANTRIK_OLLAMA")
           or os.environ.get("OLLAMA_HOST") or "").strip()
    if not raw:
        raw = "127.0.0.1:11434"
    host = raw.split("://")[-1]
    if host.split(":")[0] in ("0.0.0.0", "::", "[::]", ""):
        host = "127.0.0.1" + (":" + host.split(":", 1)[1] if ":" in host else ":11434")
    if ":" not in host:
        host += ":11434"
    return f"http://{host}".rstrip("/")


OLLAMA = _ollama_url()


def ollama_up(timeout: int = 3) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=timeout).read()
        return True
    except Exception:                                          # noqa: BLE001
        return False


def ollama_models() -> list[str]:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=10) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:                                          # noqa: BLE001
        return []


def to_gguf(adapter_dir: Path, base_repo: str, out: Path) -> str | None:
    """Convert a peft adapter to a GGUF LoRA, publisher-side.

    Done at BUILD time on purpose. The conversion needs llama.cpp's
    `conversion` package — a sparse clone, not a pip install — and
    asking every consumer to set that up would put a toolchain between
    someone and a working local model. The publisher pays it once and
    ships the result inside the .ycap.

    Returns a reason string on failure rather than raising: a capability
    without a GGUF is still perfectly usable through a Python serving
    process, so this is a missing convenience, not a broken build.
    """
    import subprocess

    conv = os.environ.get("LLAMA_CPP_DIR")
    cands = [Path(conv)] if conv else []
    cands += [HERE.parent / "vendor" / "llama.cpp", Path.home() / "llama.cpp"]
    repo = next((c for c in cands if (c / "convert_lora_to_gguf.py").exists()), None)
    if repo is None:
        return ("no llama.cpp checkout found — set LLAMA_CPP_DIR to one with "
                "convert_lora_to_gguf.py and its conversion/ package")

    # The converter reads config.json off the base, so it needs the
    # resolved snapshot on disk. A bare repo id is treated as a path and
    # fails with a confusing FileNotFoundError.
    cache = (Path.home() / ".cache" / "huggingface" / "hub" /
             f"models--{base_repo.replace('/', '--')}")
    ref = cache / "refs" / "main"
    if not ref.exists():
        return f"base {base_repo} is not in the local HF cache; cannot resolve its config"
    snap = cache / "snapshots" / ref.read_text(encoding="utf-8").strip()

    # The converter needs transformers and gguf. Prefer the training
    # venv, which by construction has both — bundle.py itself is run
    # from the light pack venv and must not require them.
    py = os.environ.get("YANTRIK_CONVERT_PYTHON")
    if not py:
        cand = HERE.parent / ".venv-compile" / "Scripts" / "python.exe"
        cand2 = HERE.parent / ".venv-compile" / "bin" / "python"
        py = str(cand if cand.exists() else cand2 if cand2.exists() else sys.executable)

    try:
        r = subprocess.run(
            [py, str(repo / "convert_lora_to_gguf.py"),
             "--base", str(snap), "--outfile", str(out), "--outtype", "f16",
             str(adapter_dir)],
            capture_output=True, text=True, timeout=900)
    except Exception as e:                                     # noqa: BLE001
        return f"converter failed to run: {e}"
    if r.returncode != 0 or not out.exists():
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or ["unknown"]
        return f"conversion failed: {tail[0][:160]}"
    return None


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

    DIST.mkdir(parents=True, exist_ok=True)
    gguf = run_dir / GGUF_NAME
    gguf_err = None
    if not gguf.exists():
        gguf_err = to_gguf(run_dir, manifest["base"]["repo"], gguf)
    if gguf.exists():
        manifest["adapter"]["gguf_bytes"] = gguf.stat().st_size

    if key:
        from yantrikdb import YantrikDB
        manifest["signature"] = {
            "publisher_pubkey": YantrikDB.pubkey_of(key),
            "value": YantrikDB.sign_bytes(key, canonical_bytes(manifest)),
        }
    # NOTHING may mutate `manifest` past this point. Adding a field after
    # signing is how the marketplace validator came to report every
    # genuine capability as forged, and how this build did the same.

    out = DIST / f"{src_pack}-{version}.ycap"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(MANIFEST, json.dumps(manifest, indent=2))
        for f in files:
            z.write(f, f.name)
        if gguf.exists():
            # The whole reason install can be one command on a machine
            # with nothing but Ollama.
            z.write(gguf, GGUF_NAME)
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
    if gguf.exists():
        print(f"  gguf      {gguf.stat().st_size / 1e6:.1f} MB — installs straight into Ollama")
    else:
        print(f"  gguf      not included: {gguf_err}")
        print("            (still usable through a Python serving process)")
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


def base_tag(base_repo: str) -> str:
    """The Ollama tag most likely to hold this base.

    A guess, and it is reported as one. Ollama names models by family
    and size, HuggingFace by org and repo; there is no registry mapping
    the two, so anything here is a heuristic and the user is shown what
    was tried when it misses.
    """
    name = base_repo.split("/")[-1].lower()          # qwen3.5-4b
    m = re.match(r"([a-z]+[\d.]*)-([\d.]+b)", name)
    return f"{m.group(1)}:{m.group(2)}" if m else name


def install_into_ollama(dest: Path, man: dict) -> int:
    """Make the capability an ordinary local model.

    This is the shape that matters: afterwards there is no daemon to
    run, no mount call, and no plugin. `ollama list` shows it, Ollama's
    OpenAI-compatible /v1 serves it, and any harness that speaks that
    API selects it by name. Ollama also stores the base once and layers
    the adapter on top, so a second capability on the same base costs
    only its adapter, not another copy of the model.
    """
    import subprocess

    gguf = dest / GGUF_NAME
    if not gguf.exists():
        print("  ollama: this build ships no GGUF adapter, so it cannot become "
              "an Ollama model.\n          Serve it with packs/serve_compiled.py instead.")
        return 0
    if not ollama_up():
        print(f"  ollama: not reachable at {OLLAMA} — skipped.\n"
              f"          Start it and re-run, or serve with packs/serve_compiled.py.")
        return 0

    have = ollama_models()
    want = base_tag(man["base"]["repo"])
    match = next((m for m in have if m.split(":")[0] == want.split(":")[0]
                  and want.split(":")[1] in m), None)
    if not match:
        near = [m for m in have if m.split(":")[0] == want.split(":")[0]]
        print(f"  ollama: base not found. This capability was compiled against "
              f"{man['base']['repo']}.")
        print(f"          Tried to match `{want}`."
              + (f" You have: {', '.join(near[:4])}" if near else ""))
        print(f"          Pull a matching base, then re-run install.")
        return 0

    name = man["name"]
    (dest / "Modelfile").write_text(f"FROM {match}\nADAPTER ./{GGUF_NAME}\n",
                                    encoding="utf-8")
    r = subprocess.run(["ollama", "create", name, "-f", str(dest / "Modelfile")],
                       capture_output=True, text=True, cwd=str(dest), timeout=900)
    if r.returncode != 0:
        print(f"  ollama: create failed — {(r.stderr or '').strip().splitlines()[-1:] or ['?']}")
        return 0

    print(f"\n  Ready. `{name}` is now a local model on top of {match}.")
    print(f"    ollama run {name}")
    print(f"    curl {OLLAMA}/v1/chat/completions -d '{{\"model\":\"{name}\", ...}}'")
    print(f"\n  Point any agent at {OLLAMA}/v1 and pick `{name}` as the model.")
    print(f"  Nothing else to run — no daemon, no mount call.")
    # Measured, not assumed. This base reasons by default and the
    # reasoning spends the whole token budget: on /v1 a request came
    # back with 13,347 characters of reasoning and 809 of content, and
    # an earlier one was empty with done_reason=stop. Sending
    # chat_template_kwargs {"enable_thinking": false} on /v1 did NOT
    # suppress it; "think": false on /api/chat did, and the same brief
    # then scored 13/15 against the base model's 8/15.
    print(f"\n  Thinking mode matters on this base. Send \"think\": false on")
    print(f"  {OLLAMA}/api/chat. On /v1, chat_template_kwargs did not")
    print(f"  suppress it here — reasoning ate the budget and answers came back")
    print(f"  short or empty. Hermes sets think=false itself.")
    return 0


def do_install(cap: Path, db_path: str, ollama: bool = True) -> int:
    if do_verify(cap, None) != 0:
        print("refusing to install an artifact that does not verify", file=sys.stderr)
        return 1
    man = load_manifest(cap)
    dest = cap_dir(db_path) / man["name"]
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(cap) as z:
        z.extractall(dest)
    print(f"installed {man['name']}@{man['version']} -> {dest}")
    if ollama:
        return install_into_ollama(dest, man)
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
    i.add_argument("--no-ollama", action="store_true",
                   help="install the files only; do not create a local model")

    l = sub.add_parser("list")
    l.add_argument("--db", required=True)

    a = ap.parse_args()
    if a.cmd == "build":
        return do_build(a.pack, a.tag, a.key, a.version)
    if a.cmd == "verify":
        return do_verify(a.cap, a.pubkey)
    if a.cmd == "install":
        return do_install(a.cap, a.db, ollama=not a.no_ollama)
    if a.cmd == "list":
        return do_list(a.db)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
