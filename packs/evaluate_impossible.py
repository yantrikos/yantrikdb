#!/usr/bin/env python3
"""The impossible-by-construction benchmark.

Task: write working Python against the YantrikDB *pack API* —
`seal_pack`, `sign_pack`, `mount_pack`, `install_pack`,
`trust_publisher`. This API was invented on 2026-07-28. No model has it
in training data, and that is provable by construction rather than
assumed: the API is younger than every model tested.

So the baseline is not "worse" — it is structurally zero. A model
without the pack must hallucinate the API, and the interpreter rejects
hallucinations. With the pack mounted, the same model reads the API it
has never seen and writes code that runs.

Grading is **execution**: the generated script runs in a scratch
directory against the real installed wheel, with a timeout. Pass =
exit 0 AND the task's postcondition holds (a file exists, stdout says
what it should). No string matching against the code, no judge — the
interpreter is the grader.

Usage:
    python packs/evaluate_impossible.py --model qwen3.6:27b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from yantrikdb import YantrikDB

import evaluate
from evaluate import DIST, MIN_SIMILARITY, resolve_host

HERE = Path(__file__).resolve().parent

BASE_SYSTEM = (
    "You are an expert Python developer. Output a single complete Python "
    "script and nothing else — no explanation before or after. Use a "
    "```python code fence. The script must be self-contained and runnable."
)

# Each task states its contract; the postcondition checks it after the
# run. SRC_PACK is a real sealed pack the harness provides for tasks
# that consume one.
TASKS = [
    {
        "id": "imp-seal",
        "q": (
            "Using the yantrikdb Python library, write a script that: creates a "
            "database at the path in the environment variable DB_PATH (embedding "
            "dimension 64), records the fact 'Gluons mediate the strong force.' "
            "in namespace 'physics' at importance 0.6, seals that namespace into "
            "a pack file at the path in the environment variable PACK_OUT with "
            "name 'physics', version '1.0.0', origin 'demo/physics', then prints "
            "the pack_id from the returned manifest and closes the database."
        ),
        "post": lambda td, out: (Path(os.environ["PACK_OUT"]).exists() if False else True),
    },
    {
        "id": "imp-mount",
        "q": (
            "Using the yantrikdb Python library, write a script that: creates a "
            "database at the path in the environment variable DB_PATH (embedding "
            "dimension 64), mounts the existing pack file at the path in the "
            "environment variable SRC_PACK, prints the number of currently "
            "mounted packs, then closes the database."
        ),
        "post": None,
    },
    {
        "id": "imp-sign-trust",
        "q": (
            "Using the yantrikdb Python library, write a script that: generates "
            "a publisher keypair; signs the existing pack file at the path in "
            "the environment variable SRC_PACK with the secret key; creates a "
            "database at DB_PATH (embedding dimension 64); makes the database "
            "trust the public key with label 'demo publisher'; mounts the pack; "
            "and prints the 'trust' field of the mounted pack. Close the "
            "database at the end."
        ),
        "post": None,
    },
    {
        "id": "imp-install",
        "q": (
            "Using the yantrikdb Python library, write a script that: creates a "
            "database at DB_PATH (embedding dimension 64), durably installs the "
            "pack file at SRC_PACK so it survives restarts, closes the database, "
            "reopens it from the same path, and prints the number of mounted "
            "packs in the reopened database."
        ),
        "post": None,
    },
]

EXPECT_STDOUT = {
    "imp-seal": lambda s: "demo/physics@1.0.0" in s,
    "imp-mount": lambda s: re.search(r"\b1\b", s) is not None,
    "imp-sign-trust": lambda s: "signed" in s.lower(),
    "imp-install": lambda s: re.search(r"\b1\b", s) is not None,
}


def extract_code(answer: str) -> str | None:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", answer, re.S)
    if m:
        return m.group(1)
    # A bare script without fences is acceptable if it imports yantrikdb.
    if "import" in answer and "yantrikdb" in answer:
        return answer
    return None


def make_src_pack(workdir: Path) -> Path:
    """A real sealed pack for the consuming tasks, built with the real API."""
    src = workdir / "srcpack-builder.db"
    pack = workdir / "source.ydbpack"
    db = YantrikDB(str(src), 64)
    db.record_text("The speed of light is invariant.", namespace="physics", importance=0.6)
    db.seal_pack(str(pack), name="source", version="1.0.0", origin="harness/source",
                 namespace="physics")
    db.close()
    return pack


def run_generated(code: str, task_id: str, workdir: Path, src_pack: Path) -> tuple[bool, str]:
    script = workdir / f"{task_id}.py"
    script.write_text(code, encoding="utf-8")
    env = dict(os.environ)
    scratch = workdir / f"{task_id}-scratch"
    scratch.mkdir(exist_ok=True)
    env["DB_PATH"] = str(scratch / "gen.db")
    env["PACK_OUT"] = str(scratch / "out.ydbpack")
    env["SRC_PACK"] = str(src_pack)
    try:
        r = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=120, env=env, cwd=str(scratch),
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    if r.returncode != 0:
        return False, (r.stderr.strip().splitlines() or ["nonzero exit"])[-1][:200]
    if task_id == "imp-seal" and not Path(env["PACK_OUT"]).exists():
        return False, "exit 0 but PACK_OUT was not created"
    if not EXPECT_STDOUT[task_id](r.stdout):
        return False, f"exit 0 but stdout wrong: {r.stdout.strip()[:120]!r}"
    return True, "ok"


def ask(model: str, system: str, user: str) -> str:
    saved = evaluate.SYSTEM
    evaluate.SYSTEM = system
    try:
        return evaluate.ask(model, user, None, num_predict=1200)
    finally:
        evaluate.SYSTEM = saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--host", default=None)
    ap.add_argument("--out", type=Path, default=HERE / "efficacy-impossible.json")
    args = ap.parse_args()
    evaluate.OLLAMA = resolve_host(args.host)
    print(f"ollama: {evaluate.OLLAMA}")

    candidates = sorted(DIST.glob("yantrikdb-pack-api-*.ydbpack"))
    if not candidates:
        raise SystemExit("build the pack first: python packs/build.py packs/yantrikdb-pack-api")

    results = []
    for model in args.model or ["qwen3.6:27b"]:
        print(f"\n=== {model}  x  impossible-by-construction (execution-graded) ===")
        t0 = time.time()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            workdir = Path(td)
            src_pack = make_src_pack(workdir)
            db = YantrikDB(str(workdir / "advisor.db"), 64)
            pack_id = db.mount_pack(str(candidates[-1]))
            ctx = db.pack_context() or ""
            mounted_system = f"{BASE_SYSTEM}\n\n{ctx}"

            rows = []
            for t in TASKS:
                hits = db.recall_text(t["q"], top_k=args.top_k)
                retrieved = [
                    h["text"] for h in hits
                    if h.get("scores", {}).get("similarity", 0.0) >= MIN_SIMILARITY
                ]
                user_mounted = t["q"]
                if retrieved:
                    joined = "\n".join(f"- {c}" for c in retrieved)
                    user_mounted = (
                        f"API reference from an attached knowledge pack:\n{joined}\n\n"
                        f"Task:\n{t['q']}"
                    )

                outcomes = {}
                for cond, (system, user) in {
                    "baseline": (BASE_SYSTEM, t["q"]),
                    "mounted": (mounted_system, user_mounted),
                }.items():
                    answer = ask(model, system, user)
                    if answer.startswith("<<error"):
                        raise SystemExit(f"model call failed: {answer}")
                    code = extract_code(answer)
                    if code is None:
                        outcomes[cond] = (False, "no code block in answer")
                        continue
                    outcomes[cond] = run_generated(code, t["id"], workdir, src_pack)

                rows.append({
                    "id": t["id"],
                    "baseline": outcomes["baseline"][0],
                    "mounted": outcomes["mounted"][0],
                    "baseline_why": outcomes["baseline"][1],
                    "mounted_why": outcomes["mounted"][1],
                    "retrieved": len(retrieved),
                })
                b, m = outcomes["baseline"], outcomes["mounted"]
                print(f"  {t['id']:<16} baseline: {'PASS' if b[0] else 'fail — ' + b[1][:80]}")
                print(f"  {'':<16} mounted:  {'PASS' if m[0] else 'fail — ' + m[1][:80]}")

            db.unmount_pack(pack_id)
            db.close()

        results.append({
            "model": model,
            "n": len(rows),
            "baseline": sum(r["baseline"] for r in rows),
            "mounted": sum(r["mounted"] for r in rows),
            "seconds": round(time.time() - t0, 1),
            "rows": rows,
        })

    print("\n" + "=" * 60)
    print(f"{'model':<16}{'tasks':>7}{'baseline':>10}{'mounted':>9}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<16}{r['n']:>7}{r['baseline']:>10}{r['mounted']:>9}")
    print("=" * 60)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
