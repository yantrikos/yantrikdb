#!/usr/bin/env python3
"""Measure what a knowledge pack is worth: the efficacy score on a listing.

For each question the model is asked twice against the same prompt:

  baseline  no context
  mounted   the pack is mounted and the top-k recall hits are injected

Both conditions run in the same process against the same model, so the
only difference is the mounted pack.

Scoring is **deterministic string matching**, never an LLM judge. A judge
would be a second unvalidated model sitting between the pack and its
own efficacy number, and the number is the entire product claim.
`expect` is a list of groups; the answer passes when at least one
alternative from every group appears (case-insensitive).

It also runs an unrelated **control set** in both conditions. A pack that
wins its own category by capturing attention and displacing everything
else is a bad pack, and this is the measurement that says so.

Usage:
    python packs/evaluate.py --model qwen3.5:4b
    python packs/evaluate.py --model qwen3.6:27b --pack yantrikdb-engine
    python packs/evaluate.py --model qwen3.6:27b --model granite4:3b
    python packs/evaluate.py --min-similarity 0   # reproduce the attach-harm result

`--model` and `--pack` repeat; models are the outer loop so a large model
stays resident in Ollama instead of being evicted on every question.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from yantrikdb import YantrikDB

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"
DEFAULT_OLLAMA = "http://192.168.4.35:11434"


def resolve_host(explicit: str | None) -> str:
    """Deliberately not plain `OLLAMA_HOST`: that variable is commonly set
    to a *bind* address like `0.0.0.0` for serving, which is not a URL a
    client can dial. Silently inheriting it made every call fail while
    the harness still reported clean zeros."""
    host = explicit or os.environ.get("PACK_EVAL_OLLAMA") or DEFAULT_OLLAMA
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")


OLLAMA = DEFAULT_OLLAMA
_ALLOW_UNVERIFIED = [False]
HOST_DIM = 64
HOST_OLLAMA_EMBEDDER = None

SYSTEM = (
    "Answer the question in at most two sentences. State the concrete name, "
    "number or identifier asked for. If reference material is supplied it is "
    "supplementary, not a boundary: when it does not address the question, "
    "ignore it completely and answer from your own knowledge as usual. Only "
    "say you do not know if you genuinely do not know the answer."
)

# Gate injection on SIMILARITY, not on the composite recall score.
#
# Measured against the yantrikdb-engine pack: on-topic queries return a
# top similarity of 0.65-0.79, off-topic queries 0.09-0.45 — a clean gap.
# The composite score overlaps far more (0.83-0.91 vs 0.37-0.69) because
# it folds in importance and recency, which are near-uniform across a
# freshly built pack and so carry no signal about relevance.
MIN_SIMILARITY = 0.55


def ask(
    model: str,
    question: str,
    context: list[str] | None,
    timeout: int = 180,
    num_predict: int = 400,
) -> str:
    """One chat turn. `think:false` matters: these models otherwise spend
    the whole token budget on reasoning and return empty content."""
    if context:
        joined = "\n".join(f"- {c}" for c in context)
        user = (
            f"Reference material retrieved from an attached knowledge pack:\n"
            f"{joined}\n\nUsing that material where relevant, answer:\n{question}"
        )
    else:
        user = question

    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "keep_alive": "20m",
            "options": {"num_predict": num_predict, "temperature": 0.0},
        }
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r).get("message", {}).get("content", "") or ""
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return f"<<error: {e}>>"


# Word boundaries that treat "_" as a SEPARATOR rather than a letter.
#
# `\b` uses the regex word class, which includes underscore, so
# `\bfinfo\b` cannot match "finfo_file" — the boundary after "finfo"
# fails against "_". php-modern's upload record says exactly the right
# thing ("Determine the type server-side with `finfo_file`") and the
# expectation `finfo` scored it as absent, so diagnose.py classified the
# question MISS_CONTENT: no record contains the answer. I very nearly
# authored a duplicate record for a fact that was already there, and
# lint_pack's near-duplicate check is the only reason I did not.
#
# In every language this repo has packs for, "_" separates words inside
# an identifier — it is punctuation that looks like a letter. So the
# boundary excludes it deliberately, while keeping digits and letters,
# which is what the anti-leniency guard actually depends on: "no" still
# cannot match k*no*w, "not" cannot match can*not*, "20" cannot match
# "2024". Only underscore adjacency changes.
#
# Third distinct bug from the same root, after the markup strip and the
# unmatchable stems. Underscore is not decoration and it is not a
# letter, and every layer that assumed one or the other was wrong.
_LEFT = r"(?<![0-9a-z])"
_RIGHT = r"(?![0-9a-z])"


def _alt_matches(alt: str, low: str) -> bool:
    """Match one alternative against a lowercased answer.

    Plain `alt in low` was wrong in the direction that inflates scores:
    "no" matched *k*no*w*, "not" matched can*not* and *not*hing, "20"
    matched "2024", "19" matched "1999". Every one of those turns a
    question the model failed into a pass, and nothing about the run
    looks wrong afterwards.

    So a bare alphanumeric alternative is matched on word boundaries.
    Anything containing punctuation or spaces — `wp_unslash`,
    `#[\\Override]`, `strlen(...)`, `no found rows` — stays a substring
    test, because \\b does not behave usefully around symbols and those
    alternatives are long enough not to collide by accident.
    """
    if alt.isalnum():
        # A LONG alphabetic alternative is treated as a stem: bounded on
        # the left only, so "optimiz" covers optimize / optimized /
        # optimisation and "eliminat" covers eliminated / elimination.
        #
        # Authors write these deliberately and whole-word matching made
        # every one unmatchable. c-safety lost points on answers that
        # said exactly the right thing ("compilers optimize away the
        # operation") because the expectation read "optimiz". lint_evals
        # only flags a stem when its inflection appears in the QUESTION,
        # and these appear only in the answer, so nothing caught them.
        #
        # The six-character floor is what keeps this safe. The
        # anti-leniency guard exists because "no" once matched k*no*w
        # and "not" matched can*not* — two and three characters, which
        # stay whole-word and cannot creep back in.
        if alt.isalpha() and len(alt) >= 6:
            return re.search(rf"{_LEFT}{re.escape(alt)}", low) is not None
        return re.search(rf"{_LEFT}{re.escape(alt)}{_RIGHT}", low) is not None
    return alt in low


# Markdown emphasis, removed before matching.
#
# Models answer in markdown. They write "the `note` section" and
# "**detail** section", and a multi-word alternative is a SUBSTRING
# test, so "note section" did not match "`note` section" and the answer
# was scored wrong. Three questions in one held-out run failed this way
# on answers that were substantively correct.
#
# This deflates rather than inflates, which is the direction that hides:
# a pack looks worse than it is and nothing about the run looks broken.
# Stripping the characters cannot invent a match, because it removes
# only punctuation and never joins two words — "`note` section" becomes
# "note section", not "notesection".
# ONLY backtick and asterisk. NOT underscore, and not tilde.
#
# The first version stripped "`*_~", and underscore is not decoration in
# any technical domain — it is half the identifiers. Stripping it turned
# the answer's `_meta` into "meta" while the expected marker stayed
# `_meta`, so three questions failed on answers that said the exact
# thing being asked for: "carries its own protocolVersion in `_meta`",
# "resides in the `_meta` object", "error code insufficient_scope".
#
# A grader that mangles snake_case cannot measure a protocol pack, and
# it fails silently in the direction that hides — the pack looks worse
# and every run looks clean. This is the second time today that a fix
# aimed at markdown quietly broke matching; the lesson is that the
# answer text must be normalised as little as possible, because every
# character removed is a character an identifier might have needed.
#
# Re-measuring the whole line afterwards showed "three questions" was a
# large understatement, and understated it in a specific way. The cost
# was not spread evenly — it fell on each pack in proportion to how much
# snake_case its domain uses. At a fixed top_k, correcting this moved
# wordpress-expert 6/20 -> 17/20 and yantrikdb-engine 12/20 -> 18/20,
# while camelCase and prose packs moved by one question or none.
#
# So the bug did not merely depress the scores, it REORDERED them, and
# ordering is what pack decisions are made on: java-stdlib was shelved
# as net-negative and wordpress-expert written off as content-bound,
# both on readings of this instrument. A measurement error correlated
# with the property being measured is more dangerous than a large
# uncorrelated one, because the averages and the control set both stay
# clean while the ranking silently goes wrong.
MARKUP = str.maketrans("", "", "`*")


def grade(answer: str, expect: list[list[str]], reject: list[str] | None = None) -> bool:
    """Did the answer say the right thing, and not the wrong thing?

    `reject` exists because numeric domains cannot be graded by presence
    alone. The UK minimum wage for a worker under 18 is £8, and the
    superseded 18-to-20 rate is £8.60 — "£8" is a substring of "£8.60",
    and word boundaries do not help, because in "£8.60" the 8 sits
    between "£" and ".", which are both non-word characters, so `\\b8\\b`
    matches it. A model reciting last year's rate would have scored as
    correct on the exact question the pack exists to fix.

    That is the lenient direction, which is the one that hides: the pack
    reports a win for reproducing the error it was built to prevent.
    Every rate, version and date pack has this shape — the wrong answers
    are not random, they are the previous edition, and the previous
    edition is usually a numeric prefix or sibling of the right one.

    So a question may name the specific wrong answers it must not
    contain. This only ever makes the grader stricter: a question
    without `reject` grades exactly as before.
    """
    low = answer.lower().translate(MARKUP)
    if reject and any(_alt_matches(alt.lower(), low) for alt in reject):
        return False
    return all(any(_alt_matches(alt.lower(), low) for alt in group) for group in expect)


def pack_setting(pack: str, key: str, fallback):
    """A retrieval parameter the pack declares for itself."""
    try:
        import tomllib
        cfg = tomllib.loads(
            (Path(__file__).resolve().parent / pack / "pack.toml")
            .read_text(encoding="utf-8"))
        v = cfg.get("content", {}).get(key)
        return type(fallback)(v) if v is not None else fallback
    except Exception:                                  # noqa: BLE001
        return fallback


def recommended_top_k(pack: str, fallback: int) -> int:
    """The pack's own swept top_k, if it declares one.

    top_k has no engine default — it is a required argument — so the
    harness's 5 was never a product default, just a habit. It turned out
    to be the binding constraint on the best packs: on mcp-spec the
    answer to four questions sat at rank 6 to 17, above the similarity
    floor and simply cut off. Raising it to 16 was worth +3 there and
    +2 on react-craft with no authoring at all.

    The right value depends on corpus size and density, so it belongs to
    the pack rather than to the harness. It is reported alongside every
    score, and the unrelated-topic controls are measured at the same k —
    which is what stops this becoming a dial for inflating a listing,
    because attach-harm climbs with k and the controls show it.
    """
    return pack_setting(pack, "recommended_top_k", fallback)


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_pack(
    model: str,
    pack_dir: Path,
    top_k: int,
    control: list[dict],
    min_similarity: float = MIN_SIMILARITY,
) -> dict:
    cfg_name = pack_dir.name
    candidates = sorted(DIST.glob(f"{cfg_name}-*.ydbpack"))
    if not candidates:
        raise SystemExit(f"no built pack for {cfg_name} — run: python packs/build.py --all")
    pack_file = candidates[-1]
    questions = load_jsonl(pack_dir / "eval.jsonl")

    # A fresh empty host: this is the qwen-27B scenario — a capable model
    # with no memories of its own, gaining a domain by mounting it.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        db = YantrikDB(str(Path(td) / "host.db"), HOST_DIM)
        if HOST_OLLAMA_EMBEDDER:
            # The host must embed queries the same way the pack embedded
            # its records. PackEmbedderMismatch enforces the pairing, so
            # a mismatch fails loudly rather than returning
            # plausible-looking nonsense.
            import json as _j, urllib.request as _u

            class _E:
                def encode(self, text):
                    pl = _j.dumps({"model": HOST_OLLAMA_EMBEDDER,
                                   "prompt": str(text)[:2000]}).encode()
                    rq = _u.Request(f"{OLLAMA}/api/embeddings", data=pl,
                                    headers={"Content-Type": "application/json"})
                    with _u.urlopen(rq, timeout=120) as r:
                        return _j.load(r)["embedding"]

            db.set_embedder(_E())
            # A freshly-created host has no recorded embedder identity,
            # so mount_pack refuses it — correctly, since it cannot
            # prove the pack's vectors and this database's queries share
            # an embedding space. Here the claim is true by
            # construction: the pack was built with the same ollama
            # model this host is about to query with. adopt is an
            # ASSERTION, not a measurement, and it is only honest to
            # call it when you built both sides yourself.
            # A raw Python callable is not fingerprinted, so there is
            # no identity to adopt and mount_pack cannot prove
            # compatibility. Compatibility here is unproven rather than
            # false: the same ollama model embedded both sides. That is
            # precisely the case this flag documents, and it is only
            # honest to use it when you built both artifacts yourself.
            _ALLOW_UNVERIFIED[0] = True
        pack_id = db.mount_pack(str(pack_file),
                                  allow_unverified_embedder=_ALLOW_UNVERIFIED[0])

        rows = []
        for item in questions + control:
            is_control = item in control
            hits = db.recall_text(item["q"], top_k=top_k)
            # Unconditional top-k injection is measurably harmful: with no
            # floor, an unrelated question still receives five pack facts,
            # and the model concludes it must answer only from them. That
            # cost 7 of 12 control questions on qwen3.5:4b.
            context = [
                h["text"]
                for h in hits
                if h.get("scores", {}).get("similarity", 0.0) >= min_similarity
            ]

            base = ask(model, item["q"], None)
            mnt = ask(model, item["q"], context)
            rows.append(
                {
                    "id": item["id"],
                    "control": is_control,
                    "baseline": grade(base, item["expect"], item.get("reject")),
                    "mounted": grade(mnt, item["expect"], item.get("reject")),
                    "baseline_answer": base.strip()[:400],
                    "mounted_answer": mnt.strip()[:400],
                    "retrieved": len(context),
                    "hits": len(hits),
                }
            )
            tag = "ctl" if is_control else "pack"
            mark = {
                (False, True): "GAIN",
                (True, False): "LOSS",
                (True, True): "both",
                (False, False): "none",
            }[(rows[-1]["baseline"], rows[-1]["mounted"])]
            print(f"  [{tag}] {item['id']:<24} {mark}")

        db.unmount_pack(pack_id)
        db.close()

    # A benchmark whose transport is broken must not report clean zeros.
    # Before this check, every call failing produced a tidy 0/20 baseline
    # and 0/20 mounted — which reads exactly like "the pack did nothing".
    errors = sum(
        1 for r in rows for k in ("baseline_answer", "mounted_answer") if r[k].startswith("<<error")
    )
    if errors:
        sample = next(
            r[k]
            for r in rows
            for k in ("baseline_answer", "mounted_answer")
            if r[k].startswith("<<error")
        )
        raise SystemExit(
            f"{errors}/{len(rows) * 2} model calls failed against {OLLAMA} — "
            f"refusing to report a score.\nfirst error: {sample}"
        )

    pack_rows = [r for r in rows if not r["control"]]
    ctl_rows = [r for r in rows if r["control"]]
    return {
        "model": model,
        "pack": cfg_name,
        "pack_file": pack_file.name,
        "n": len(pack_rows),
        "baseline": sum(r["baseline"] for r in pack_rows),
        "mounted": sum(r["mounted"] for r in pack_rows),
        "control_n": len(ctl_rows),
        "control_baseline": sum(r["baseline"] for r in ctl_rows),
        "control_mounted": sum(r["mounted"] for r in ctl_rows),
        "regressions": [r["id"] for r in pack_rows + ctl_rows if r["baseline"] and not r["mounted"]],
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--pack", action="append", default=[])
    ap.add_argument("--top-k", type=int, default=None,
                    help="override the pack's recommended_top_k")
    ap.add_argument(
        "--min-similarity",
        type=float,
        default=MIN_SIMILARITY,
        help="similarity floor for injecting a retrieved fact; 0 disables gating",
    )
    ap.add_argument("--host", default=None, help=f"Ollama base URL (default {DEFAULT_OLLAMA})")
    ap.add_argument("--dim", type=int, default=64,
                    help="host vector width; must match the pack's embedder")
    ap.add_argument("--ollama-embedder",
                    help="embed queries with this ollama model instead of "
                         "the engine's bundled static embedder")
    ap.add_argument("--out", type=Path, default=HERE / "efficacy.json")
    args = ap.parse_args()

    global OLLAMA, HOST_DIM, HOST_OLLAMA_EMBEDDER
    OLLAMA = resolve_host(args.host)
    HOST_DIM = args.dim
    HOST_OLLAMA_EMBEDDER = args.ollama_embedder
    print(f"ollama: {OLLAMA}")

    models = args.model or ["qwen3.5:4b"]
    pack_dirs = (
        [HERE / p for p in args.pack]
        if args.pack
        else sorted(p.parent for p in HERE.glob("*/pack.toml"))
    )
    control = load_jsonl(HERE / "control.jsonl")

    results = []
    for model in models:
        for pd in pack_dirs:
            print(f"\n=== {model}  x  {pd.name} ===")
            t0 = time.time()
            k = args.top_k or recommended_top_k(pd.name, 5)
            # The floor is per-pack too. yantrikdb-engine is the reason:
            # its own deletion records clear the default 0.55 on the word
            # "delete" and derail an unrelated SQL question at every k.
            floor = pack_setting(pd.name, "recommended_min_similarity",
                                 args.min_similarity)
            res = run_pack(model, pd, k, control, floor)
            res["top_k"], res["min_similarity"] = k, floor
            res["seconds"] = round(time.time() - t0, 1)
            results.append(res)

    print("\n" + "=" * 78)
    print(
        f"{'model':<22}{'pack':<26}{'pack Q':>9}{'baseline':>10}{'mounted':>9}{'control':>10}"
    )
    print("-" * 78)
    for r in results:
        b = f"{r['baseline']}/{r['n']}"
        m = f"{r['mounted']}/{r['n']}"
        c = f"{r['control_baseline']}->{r['control_mounted']}/{r['control_n']}"
        print(f"{r['model']:<22}{r['pack']:<26}{r['n']:>9}{b:>10}{m:>9}{c:>10}")
    print("=" * 78)
    for r in results:
        if r["regressions"]:
            print(f"REGRESSIONS {r['model']} x {r['pack']}: {', '.join(r['regressions'])}")

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
