#!/usr/bin/env python3
"""Independent pack certification: signed efficacy certificates.

The problem this solves: `evaluate.py` is seller-runnable. A seller who
writes their own eval questions scores 100% trivially — ask questions
whose answers are verbatim in the corpus — so a self-reported listing
number is worthless the day it starts selling packs. The fix is
structural, not procedural:

  1. The EVALUATOR holds the question set, and it is **held out** —
     never published, never shown to sellers. A seller cannot tune a
     pack against questions they have never seen.
  2. The evaluator SIGNS the result over the pack's content digest, so
     the certificate binds to one exact pack build. Re-seal the pack —
     even one changed row — and the certificate no longer applies.
  3. Buyers verify offline with the evaluator's public key, exactly the
     TOFU trust model used for publisher signing. No portal, no API.

The certificate also carries the attach-harm control result, because a
pack that wins its category by wrecking everything else must fail
certification, not pass it with an asterisk.

Certify:
    python packs/certify.py --pack dist/yantrikdb-engine-0.1.0.ydbpack \
        --holdout holdout/yantrikdb-engine.jsonl \
        --model qwen3.5:4b --key <evaluator_secret_hex>

Verify (buyer side, offline):
    python packs/certify.py --verify cert.json --pack <file> \
        --evaluator-pubkey <hex>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import tempfile
from pathlib import Path

from yantrikdb import YantrikDB

import evaluate
from evaluate import MIN_SIMILARITY, ask, grade, load_jsonl, resolve_host

HERE = Path(__file__).resolve().parent

CERT_SCHEMA = "yantrikdb.pack.cert.v1"


def canonical_bytes(cert: dict) -> bytes:
    """Sorted-keys, no-whitespace JSON over everything except the
    signature itself. One canonical form, so signer and verifier can
    never disagree about serialization."""
    unsigned = {k: v for k, v in cert.items() if k != "signature"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()


def run_certification(
    pack_file: Path, holdout: list[dict], control: list[dict], model: str, top_k: int
) -> dict:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = YantrikDB(str(Path(td) / "host.db"), 64)
        manifest = YantrikDB.read_pack_manifest(str(pack_file))
        pack_id = db.mount_pack(str(pack_file))
        ctx = db.pack_context()

        def condition(item: dict, mounted: bool) -> bool:
            context = None
            if mounted:
                hits = db.recall_text(item["q"], top_k=top_k)
                context = [
                    h["text"]
                    for h in hits
                    if h.get("scores", {}).get("similarity", 0.0) >= MIN_SIMILARITY
                ] or None
            saved = evaluate.SYSTEM
            if mounted and ctx:
                evaluate.SYSTEM = f"{saved}\n\n{ctx}"
            try:
                answer = ask(model, item["q"], context)
            finally:
                evaluate.SYSTEM = saved
            if answer.startswith("<<error"):
                raise SystemExit(f"model call failed: {answer} — refusing to certify")
            return grade(answer, item["expect"])

        holdout_base = sum(condition(i, False) for i in holdout)
        holdout_mnt = sum(condition(i, True) for i in holdout)
        ctl_base = sum(condition(i, False) for i in control)
        ctl_mnt = sum(condition(i, True) for i in control)
        db.unmount_pack(pack_id)
        db.close()

    return {
        "schema": CERT_SCHEMA,
        "pack_id": manifest["pack_id"],
        "content_digest": manifest["content_digest"],
        "model": model,
        "harness": "certify.py/holdout-v1",
        "issued": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "holdout": {"n": len(holdout), "baseline": holdout_base, "mounted": holdout_mnt},
        "attach_harm": {
            "n": len(control),
            "control_baseline": ctl_base,
            "control_mounted": ctl_mnt,
        },
        # The gate, not a footnote: certification FAILS on regression.
        "attach_harm_pass": ctl_mnt >= ctl_base,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--holdout", type=Path)
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--key", help="evaluator secret key hex (certify mode)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--host", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verify", type=Path, help="certificate to verify instead of certifying")
    ap.add_argument("--evaluator-pubkey", help="expected evaluator key (verify mode)")
    args = ap.parse_args()

    if args.verify:
        cert = json.loads(args.verify.read_text(encoding="utf-8"))
        sig = cert.get("signature", {})
        ok = YantrikDB.verify_bytes(
            sig.get("evaluator_pubkey", ""), canonical_bytes(cert), sig.get("value", "")
        )
        if not ok:
            raise SystemExit("FAIL: signature does not verify — certificate tampered or forged")
        if args.evaluator_pubkey and sig["evaluator_pubkey"] != args.evaluator_pubkey:
            raise SystemExit(
                "FAIL: certificate is validly signed, but by a DIFFERENT evaluator "
                f"({sig['evaluator_pubkey'][:16]}…) — trust is per-key, not per-format"
            )
        local = YantrikDB.read_pack_manifest(str(args.pack))
        if local["content_digest"] != cert["content_digest"]:
            raise SystemExit(
                "FAIL: certificate is genuine but for a DIFFERENT BUILD of this pack — "
                "the file was re-sealed since certification"
            )
        h, a = cert["holdout"], cert["attach_harm"]
        print(f"OK: {cert['pack_id']} certified by {sig['evaluator_pubkey'][:16]}…")
        print(f"    held-out: {h['baseline']}/{h['n']} -> {h['mounted']}/{h['n']}  ({cert['model']})")
        print(f"    attach-harm: {a['control_baseline']} -> {a['control_mounted']} / {a['n']}"
              f"  {'PASS' if cert['attach_harm_pass'] else 'FAIL'}")
        return

    if not args.key or not args.holdout:
        raise SystemExit("certify mode needs --key and --holdout")
    evaluate.OLLAMA = resolve_host(args.host)
    holdout = load_jsonl(args.holdout)
    control = load_jsonl(HERE / "control.jsonl")

    cert = run_certification(args.pack, holdout, control, args.model, args.top_k)
    cert["signature"] = {
        "evaluator_pubkey": YantrikDB.pubkey_of(args.key),
        "value": YantrikDB.sign_bytes(args.key, canonical_bytes(cert)),
    }

    out = args.out or Path(f"{args.pack}.cert.json")
    out.write_text(json.dumps(cert, indent=2), encoding="utf-8")
    h, a = cert["holdout"], cert["attach_harm"]
    print(f"certified {cert['pack_id']}")
    print(f"  held-out:    {h['baseline']}/{h['n']} -> {h['mounted']}/{h['n']}")
    print(f"  attach-harm: {a['control_baseline']} -> {a['control_mounted']} / {a['n']}"
          f"  {'PASS' if cert['attach_harm_pass'] else 'FAIL'}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
