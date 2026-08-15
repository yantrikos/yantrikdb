#!/usr/bin/env python3
"""Compile a pack into weights: corpus.md -> verified examples -> LoRA adapter.

A pack is knowledge a model mounts. Today it mounts as *context*: the
records are retrieved and injected, and they cost tokens on every single
request. This compiles the same pack into a LoRA adapter instead, so the
knowledge is carried in weights at zero context cost and the adapter is
mounted and unmounted per task.

WHAT MAKES THE NUMBER MEAN ANYTHING

  HELD OUT BY CONSTRUCTION   Training data is synthesised from
        corpus.md and constitution.md and NEVER from eval.jsonl. All 53
        questions are therefore unseen, and the compiled arm is compared
        against the retrieval arm reading the same corpus. This is
        stronger than a train/test split over the questions, because
        there is no split to get wrong.

  VERIFIED, NOT GENERATED    A teacher proposes question/answer pairs
        from one record. A pair enters training only if every key it
        claims is present in THAT SOURCE RECORD, the answer actually
        contains those keys, and the question does not give them away.
        Grounding is checked with the pack harness's own matcher, so a
        fluent invention is rejected by the same code that grades the
        eval. The prior in-house result was explicit that the binding
        constraint on cheap training is verification quality, not data
        volume — so the verifier is the product here, not the teacher.

  ONE STACK, FOUR ARMS       Measurement runs through the existing
        evaluate.py against serve_compiled.py, which serves base and
        adapter from the SAME process and the same bf16 weights. Nothing
        differs between arms except whether the adapter is mounted, so a
        difference cannot be a quantisation or tokeniser artifact.

  A NEGATIVE CONTROL ADAPTER Compile a DIFFERENT pack and evaluate it on
        this pack's questions. If that also scores, the harness is
        leaking and the result is void.

USAGE
    python packs/compile.py --pack mcp-spec --synthesize
    python packs/compile.py --pack mcp-spec --train
    python packs/serve_compiled.py --adapter mcp-spec &
    python packs/evaluate.py --host http://127.0.0.1:11555 \
        --model base --model mcp-spec --pack mcp-spec

--synthesize needs only the pack venv (it talks to a teacher over HTTP).
--train needs torch/peft, which live in .venv-compile:
    .venv-compile/Scripts/python packs/compile.py --pack mcp-spec --train
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RUNS = HERE / "training"
DEFAULT_TEACHER = "qwen3.6:27b"
DEFAULT_BASE = "Qwen/Qwen3.5-4B"

# The prompt the model will be graded under. Training examples are built
# in exactly this shape minus the reference material, because that is
# the deployment shape for a compiled pack: no context, just the
# question. Any other framing trains for a prompt nobody will send.
SYSTEM = (
    "Answer the question in at most two sentences. State the concrete name, "
    "number or identifier asked for. If reference material is supplied it is "
    "supplementary, not a boundary: when it does not address the question, "
    "ignore it completely and answer from your own knowledge as usual. Only "
    "say you do not know if you genuinely do not know the answer."
)


# Teacher transports.
#
# The local 27B was the first teacher and it does not scale here: one
# request at a time, and it competes for the same two GPUs that training
# and serving need. Synthesis stalled outright once a training run and
# the eval server were both resident — the teacher could not even load.
# An OpenAI-compatible endpoint costs nothing in VRAM and runs the
# records concurrently, which is what makes eight forms per fact
# affordable to regenerate rather than a thing you do once.
PROVIDERS = {
    "qwen": ("QWEN_OPENAI_COMPATIBLE_API", "QWEN_API_KEY", "qwen3.8-max"),
    "nim": ("NIM_BASE_URL", "NVIDIA_NIM_API_KEY", "deepseek-ai/deepseek-v4-pro"),
    "openai": ("OPENAI_BASE_URL", "OPENAI_API_KEY", "gpt-5.5"),
}
DEFAULT_BASE_URLS = {
    "nim": "https://integrate.api.nvidia.com/v1",
    "openai": "https://api.openai.com/v1",
}


def load_env_file(path: str | None) -> None:
    """KEY=VALUE lines into the environment, for a keyfile kept outside
    the repository. Nothing here is ever written back or logged."""
    if not path:
        return
    p = Path(path)
    if not p.exists():
        print(f"  --env-file {p} not found", file=sys.stderr)
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def api_asker(provider: str):
    """An `ask` with evaluate.ask's signature, backed by a chat endpoint."""
    import urllib.error
    import urllib.request

    url_var, key_var, _ = PROVIDERS[provider]
    base = os.environ.get(url_var) or DEFAULT_BASE_URLS.get(provider)
    key = os.environ.get(key_var)
    if not base or not key:
        raise SystemExit(
            f"{provider}: need {url_var} and {key_var} in the environment "
            f"(pass --env-file to load them from a keyfile)")

    def ask(model, question, context, timeout=300, num_predict=2400):
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0.0,
            "max_tokens": num_predict,
        }).encode()
        req = urllib.request.Request(
            base.rstrip("/") + "/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            return d["choices"][0]["message"].get("content") or ""
        except (urllib.error.URLError, TimeoutError, KeyError,
                IndexError, json.JSONDecodeError) as e:
            return f"<<error: {e}>>"

    return ask


def chat_prefix(tok, messages: list[dict]) -> str:
    """The exact prompt text both training and serving must build.

    This function exists because the two drifted and it cost a whole
    training run. Qwen's template ends the prompt at `<think>\\n` by
    default and at `<think>\\n\\n</think>\\n\\n` when thinking is
    suppressed. Training on the first and serving the second teaches the
    model to answer INSIDE the reasoning block, and what comes back at
    eval time is a repetition loop — a result that looks like a failed
    experiment rather than a mismatched prompt.

    So there is one function, imported by both sides. A format that
    differs between training and inference is not a hyperparameter, it
    is a silent invalidation of the number.
    """
    try:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)


def records(pack_dir: Path) -> list[tuple[str, str]]:
    """Every `## ` section of the corpus and the constitution.

    Both files are authored one-fact-per-heading — the first authoring
    law — so a section is the natural training unit, and the unit the
    retrieval arm serves. Keeping them identical is what makes arm B and
    arm C comparable at all.
    """
    out = []
    for name in ("corpus.md", "constitution.md"):
        f = pack_dir / name
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        for chunk in re.split(r"^## ", text, flags=re.M)[1:]:
            head, _, body = chunk.partition("\n")
            body = body.strip()
            if body:
                out.append((head.strip(), body))
    return out


# ---------------------------------------------------------------- synthesis

# Craft compilation: brief -> artifact, gated by structural checks.
#
# The QA pipeline above compiles KNOWLEDGE — can the model state a fact.
# This one compiles CRAFT — does the model's artifact comply with the
# constitution without being shown it. The teacher is given the rulebook
# (context distillation: the teacher may lean on context; the student
# must learn the behaviour bare), and its artifact is admitted only if
# every deterministic check in wp_theme_checks passes. A failed artifact
# gets ONE repair pass with the named failures — repair-in-loop tripled
# effective teacher throughput in the YDS run — and is dropped if it
# still fails. Near-misses are logged, not trained: they are future
# preference pairs, not positives.
CRAFT_SYSTEM = (
    "You are an expert WordPress block theme designer. Reply with the "
    "complete theme.json content only — valid JSON, no prose, no code fence."
)


def craft_system(W) -> str:
    """The system prompt for THIS pack — never another pack's.

    One hardcoded CRAFT_SYSTEM was sent by all three call sites (trainer,
    evaluator, repair loop), so motion-craft was trained and graded while
    being told to reply with theme.json for an HTML artifact: the adapter
    learned to ignore the instruction and the control arms met it cold,
    which inflated the compiled-vs-context gap by an unknown amount.

    A pack declares SYSTEM. If it does not, one is built from its own
    ARTIFACT, so a new pack cannot inherit this failure by omission.
    """
    own = getattr(W, "SYSTEM", None)
    if own:
        return own
    artifact = getattr(W, "ARTIFACT", "the requested artifact")
    return (f"You are an expert in this domain. Reply with {artifact} only "
            f"— no prose, no code fence.")

# The constitution states rules as prose; the checklist restates the
# mechanically-checked ones as JSON paths. The distinction is the webkit
# experiment's finding — ops arm 12/12, both prose arms 0/6, and every
# control failure was a numeric constraint the prose stated and the
# model did not compute. Measured here before the checklist existed: 42
# of 63 near-misses omitted useRootPaddingAwareAlignments, a flag the
# constitution names in exactly one prose sentence.
CRAFT_CHECKLIST = """MECHANICAL CHECKLIST — the validator tests these paths exactly:
- version: 2 or 3
- settings.typography.fluid: true
- settings.typography.fontSizes: 5-7 entries
- settings.layout.contentSize: 34-42rem (a measure, never 1200px); wideSize set
- settings.color.palette: >=4 entries, slugs named by role (base, contrast, primary, secondary)
- settings.spacing.spacingScale or spacingSizes (>=4 steps)
- settings.useRootPaddingAwareAlignments: true whenever styles.spacing.padding is set
- styles.typography.fontFamily, fontSize AND lineHeight (1.4-1.8) — declared presets must be APPLIED
- styles.elements.h1 (or h2) typography.lineHeight: 1.0-1.3
- styles.color.background and .text — never #000, #fff, black or white anywhere in base/contrast/background/text
- styles.elements.link.color.text set
- styles.spacing.blockGap set"""

CRAFT_PROMPT = """{constitution}

---

{checklist}

---

You are writing {artifact} for the following brief. Apply every rule
above; the output will be validated mechanically.

BRIEF: {brief}

Reply with the complete {artifact} only — no prose, no explanation."""

CRAFT_REPAIR = """Your artifact failed these mechanical checks:

{failures}

Fix every failure and reply with the complete corrected artifact only.
No prose."""


def craft_module(pack: str):
    """The pack's own craft definition: grader, briefs, prompt shape.

    A craft pack ships `craft.py` beside its constitution, exposing:
        ARTIFACT      one-line name of what gets generated ("theme.json")
        grade_text(text) -> (passed, total, per_check|None)
        train_briefs() / holdout_briefs() -> list[str]
        CHECKLIST     the mechanical constraints as exact paths/values
    The compiler and the evaluator both load THIS module, so "compliant"
    keeps meaning the same thing on both sides of the experiment — the
    property that made the first result credible. wordpress-theme's
    craft.py wraps wp_theme_checks; a new craft pack brings its own.
    """
    import importlib.util
    d = HERE / pack
    if not (d / "craft.py").exists():
        raise SystemExit(f"{pack} has no craft.py — not a craft pack")
    spec = importlib.util.spec_from_file_location(f"{pack}_craft", d / "craft.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for req in ("ARTIFACT", "CHECKLIST", "grade_text", "train_briefs", "holdout_briefs"):
        if not hasattr(mod, req):
            raise SystemExit(f"{pack}/craft.py lacks {req}")
    return mod


def do_synthesize_craft(pack: str, teacher: str, host: str | None,
                        api: str | None, workers: int) -> int:
    from concurrent.futures import ThreadPoolExecutor

    import evaluate

    W = craft_module(pack)

    if api:
        ask = api_asker(api)
        where = f"{api}"
    else:
        evaluate.OLLAMA = evaluate.resolve_host(host)
        ask, where = evaluate.ask, evaluate.OLLAMA

    constitution = (HERE / pack / "constitution.md").read_text(encoding="utf-8")
    briefs = W.train_briefs()
    out_dir = RUNS / f"{pack}-craft"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"teacher {teacher} at {where}, {workers} workers")
    print(f"{len(briefs)} training briefs, {len(W.CHECKS)} checks, "
          f"one repair pass\n")

    def one(idx_brief):
        idx, brief = idx_brief
        prompt = CRAFT_PROMPT.format(constitution=constitution,
                                     checklist=W.CHECKLIST,
                                     artifact=W.ARTIFACT, brief=brief)
        raw = ask(teacher, prompt, None, timeout=600, num_predict=4000)
        p, n, res = W.grade_text(raw)
        attempts = 1
        # Up to two repair rounds. One round admitted 9/72: flash fixes
        # the named failure and disturbs something else often enough
        # that convergence needs a second look.
        while res is not None and p < n and attempts <= 2:
            failures = "\n".join(f"- {cid}: {detail}"
                                 for cid, (ok, detail) in res.items() if not ok)
            raw2 = ask(teacher, prompt + "\n\n" + CRAFT_REPAIR.format(failures=failures),
                       None, timeout=600, num_predict=4000)
            p2, _, res2 = W.grade_text(raw2)
            attempts += 1
            if p2 > p:
                raw, p, res = raw2, p2, res2
            else:
                break
        artifact = W.canonical(raw) if hasattr(W, "canonical") else raw.strip()
        if artifact is not None and p < n:
            pass  # near-miss keeps its artifact for the preference log
        return idx, brief, p, n, attempts, artifact

    kept, near = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, brief, p, n, attempts, artifact in pool.map(one, enumerate(briefs, 1)):
            status = "ADMIT" if p == n and artifact else f"{p}/{n}"
            # The row carries the pack's OWN system prompt, so training
            # and serving cannot disagree about what was asked for.
            row = {"q": brief, "a": artifact, "craft": True,
                   "system": craft_system(W),
                   "checks": f"{p}/{n}", "attempts": attempts}
            if p == n and artifact:
                kept.append(row)
            else:
                near.append(row)
            print(f"  [{idx:>2}/{len(briefs)}] {status:>6}  {brief[:64]}",
                  flush=True)

    with (out_dir / "dataset.jsonl").open("w", encoding="utf-8") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    with (out_dir / "nearmiss.jsonl").open("w", encoding="utf-8") as f:
        for r in near:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  {len(kept)}/{len(briefs)} briefs produced a fully compliant "
          f"artifact -> {out_dir / 'dataset.jsonl'}")
    print(f"  {len(near)} near-misses logged (future preference pairs, "
          f"never positives)")
    return 0 if kept else 2


# The frame taxonomy, and why it is a fixed list rather than "write N
# questions".
#
# The first run asked for free-form pairs and got what a teacher
# naturally writes: 84% of 411 questions opened with "what" or "which",
# 14 were counterfactual and 2 were yes/no. Lexically they were diverse
# — median Jaccard overlap between a record's two questions was 0.09 —
# so the teacher was not rephrasing itself. It was asking the SAME KIND
# of question 411 times.
#
# That matters because a fact is only retrievable from weights along the
# directions it was trained in. Two same-shaped questions per fact is
# roughly six gradient exposures at three epochs, all pointing one way,
# and the measured result was +2 out of 53 against a retrieval arm
# scoring 52. The fix is not more source material — retrieval proves the
# corpus already contains every answer — it is more DIRECTIONS through
# the same fact.
#
# `inverse` earns its place separately: a model trained only on
# "X does Y" does not thereby learn "what does Y" — the direction has to
# be trained too, or the fact is only reachable from one side.
FRAMES = {
    "direct": "Ask plainly for the fact. 'What ... ?'",
    "inverse": "Give the behaviour, effect or value and ask WHICH thing "
               "has it — the answer is the name, the question must not "
               "contain it. This trains the fact backwards, which is a "
               "separate direction from asking it forwards.",
    "scenario": "A developer situation, two clauses of setup, then what "
                "they must do or what they will see.",
    "counterfactual": "What breaks, and how it presents, if someone does "
                      "the older or the obvious-but-wrong thing instead.",
    "contrast": "Distinguish this from the adjacent or superseded thing "
                "it is most often confused with.",
    "yesno": "A yes/no or must/may question whose answer states the rule "
             "and the identifier it turns on.",
    "enumerate": "Ask for the complete set, list or ordering the record "
                 "specifies. Skip this frame if the record has no set.",
    "debug": "Symptom first — an error, a rejected request, a wrong "
             "value — and the answer names the cause from the record.",
}

PROPOSE = """You are given ONE record from a technical reference. Write \
{n} distinct question/answer pairs that a user might ask, each answerable \
from this record ALONE.

RULES, and a pair that breaks any of them is discarded:
- The answer must be one or two sentences and must state the concrete \
name, number or identifier.
- "keys" lists the exact literal strings (identifiers, numbers, header \
names, verbs) that any correct answer MUST contain. Copy them character \
for character out of the record. Do not invent, expand or normalise them.
- Every key must appear verbatim in the record.
- The question must NOT contain any of its own keys. A question that \
gives away the answer trains nothing.
- Ask about the substance of the record, not about the document \
("what does this section say" is worthless).

Write EXACTLY ONE pair for each of these frames, in this order. The \
frame is the shape of the question, and two questions of the same shape \
teach the fact only one way:
{frames}

RECORD
## {head}

{body}

Reply with ONLY a JSON array, no prose, no code fence. Include the frame \
name on each entry:
[{{"frame": "...", "q": "...", "a": "...", "keys": ["...", "..."]}}]"""


def propose(ask, teacher: str, head: str, body: str, n: int,
            frames: list[str] | None = None) -> list[dict]:
    frames = frames or list(FRAMES)
    spec = "\n".join(f"- {f}: {FRAMES[f]}" for f in frames)
    raw = ask(teacher, PROPOSE.format(n=len(frames), frames=spec, head=head,
                                      body=body),
              None, timeout=420, num_predict=2400)
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [i for i in items if isinstance(i, dict)]


def verify(item: dict, head: str, body: str, matches) -> tuple[bool, str]:
    """Three checks, each of which kills a distinct way to poison training.

    GROUNDED    every key is present in this record. A teacher asked to
                write questions about a spec will happily answer from its
                own training data instead, which for a post-cutoff pack
                means the OLD protocol — precisely the error the pack
                exists to correct. Checking the key against the record is
                the only thing standing between that and the weights.

    ANSWERED    the answer contains the keys it claims. Teachers write a
                fluent paragraph and a key list that does not match it.

    NOT LEAKED  the question contains none of its keys, or the model
                learns to copy from the question and scores nothing at
                eval time where the question no longer carries the answer.

    Matching uses the harness's own `_alt_matches`, so a pair is admitted
    under the same rule the eval is graded under, not a looser one.
    """
    q, a, keys = item.get("q", ""), item.get("a", ""), item.get("keys") or []
    if not (isinstance(q, str) and isinstance(a, str) and isinstance(keys, list)):
        return False, "malformed"
    if not q.strip() or not a.strip() or not keys:
        return False, "empty"
    if len(a) > 600:
        return False, "answer too long"
    if len(keys) > 6:
        return False, "too many keys"

    source = f"{head}\n{body}".lower()
    low_a, low_q = a.lower(), q.lower()
    for k in keys:
        if not isinstance(k, str) or not k.strip():
            return False, "bad key"
        kl = k.lower()
        if not matches(kl, source):
            return False, f"ungrounded key: {k}"
        if not matches(kl, low_a):
            return False, f"answer omits key: {k}"
        if matches(kl, low_q):
            return False, f"question leaks key: {k}"
    return True, ""


def tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", s.lower()))


def too_similar(q: str, accepted: list[dict], ceiling: float) -> bool:
    """Reject a form that is a rephrasing of one already accepted.

    Asking for more forms per fact only helps if they are actually
    different questions. Without this the amplifier's failure mode is
    twenty ways of saying the same sentence, which multiplies the
    exposure count without adding a single new direction through the
    fact — and the count is the number that would get reported.
    """
    t = tokens(q)
    return any(
        len(t & tokens(a["q"])) / max(1, len(t | tokens(a["q"]))) > ceiling
        for a in accepted)


def do_synthesize(pack: str, teacher: str, per_record: int, keep: int,
                  host: str | None, batch_frames: int, ceiling: float,
                  include_records: bool, api: str | None, workers: int) -> int:
    from concurrent.futures import ThreadPoolExecutor

    import evaluate

    if api:
        ask = api_asker(api)
        where = f"{api} ({os.environ.get(PROVIDERS[api][0]) or DEFAULT_BASE_URLS.get(api)})"
    else:
        evaluate.OLLAMA = evaluate.resolve_host(host)
        ask, where = evaluate.ask, evaluate.OLLAMA
    pack_dir = HERE / pack
    recs = records(pack_dir)
    if not recs:
        print(f"no records in {pack_dir}", file=sys.stderr)
        return 2

    out_dir = RUNS / pack
    out_dir.mkdir(parents=True, exist_ok=True)
    data_f = out_dir / "dataset.jsonl"
    rej_f = out_dir / "rejected.jsonl"

    names = list(FRAMES)
    groups = [names[i:i + batch_frames]
              for i in range(0, len(names), batch_frames)]

    print(f"teacher {teacher} at {where}, {workers} workers")
    print(f"{len(recs)} records x {len(names)} frames in {len(groups)} calls, "
          f"keep up to {keep}, near-duplicate ceiling {ceiling}\n")

    def one(idx_rec):
        idx, (head, body) = idx_rec
        good: list[dict] = []
        bad: list[dict] = []
        for grp in groups:
            if len(good) >= keep:
                break
            for it in propose(ask, teacher, head, body, per_record, grp):
                ok, why = verify(it, head, body, evaluate._alt_matches)
                if ok and too_similar(it.get("q", ""), good, ceiling):
                    ok, why = False, "near-duplicate of an accepted form"
                if ok and len(good) < keep:
                    it["record"] = head
                    good.append(it)
                elif not ok:
                    bad.append({"record": head, "why": why, **it})
        if include_records:
            # The record itself, trained as plain text. Question/answer
            # pairs teach retrieval of the fact; the passage teaches the
            # fact's own phrasing, which is what the eval expectations
            # were written against.
            good.append({"record": head, "frame": "passage", "lm": True,
                         "q": "", "a": f"## {head}\n\n{body}", "keys": []})
        return idx, head, good, bad

    kept, rejected, reasons = [], [], {}
    frame_counts: dict[str, int] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, head, good, bad in pool.map(one, enumerate(recs, 1)):
            done += 1
            for it in good:
                frame_counts[it.get("frame", "unlabelled")] = \
                    frame_counts.get(it.get("frame", "unlabelled"), 0) + 1
            for b in bad:
                reasons[b["why"].split(":")[0]] = \
                    reasons.get(b["why"].split(":")[0], 0) + 1
            kept.extend(good)
            rejected.extend(bad)
            print(f"  [{done:>3}/{len(recs)}] {len(good):>2} forms   {head[:56]}",
                  flush=True)

    with data_f.open("w", encoding="utf-8") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    with rej_f.open("w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(kept) + len(rejected)
    print(f"\n  {len(kept)} verified / {total} proposed "
          f"({len(kept) / total:.0%} admitted) -> {data_f}")
    print("  rejections by class:")
    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {why:<24} {n}")
    print("\n  A high rejection rate is the filter working. The teacher does "
          "not know this pack's domain;\n  it answers from its own training "
          "data unless the record forces it not to.")

    print("\n  frames admitted:")
    for f, n in sorted(frame_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {f:<18} {n}")

    covered = len({k["record"] for k in kept})
    print(f"\n  coverage: {covered}/{len(recs)} records produced at least one "
          f"verified pair")
    if covered < len(recs):
        missing = [h for h, _ in recs if h not in {k["record"] for k in kept}]
        print("  records with NO training signal (their facts cannot reach "
              "the weights):")
        for h in missing[:20]:
            print(f"    {h[:70]}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")

    # The number the first run got wrong, printed where it cannot be
    # missed. 411 examples over 215 records at 3 epochs is 5.7 gradient
    # exposures per fact, and 5.7 moved a 53-question eval by +2.
    thin = sorted((h for h, _ in recs
                   if sum(1 for k in kept if k["record"] == h) < 4))
    per_fact = len(kept) / max(1, covered)
    print(f"\n  {per_fact:.1f} forms per fact. At E epochs that is "
          f"{per_fact:.1f}xE gradient exposures.")
    if thin:
        print(f"  {len(thin)} records have fewer than 4 forms and are "
              f"unlikely to become answerable.")
    return 0


# ----------------------------------------------------------------- training

def do_train(pack: str, base: str, rank: int, alpha: int, epochs: int,
             lr: float, seed: int, max_len: int, tag: str,
             batch: int, accum: int, checkpointing: bool,
             vram_fraction: float, quant4: bool = False,
             multi_gpu: bool = False, early_stop_loss: float = 0.0) -> int:
    # Set BEFORE torch initialises CUDA, which is why the import is
    # below it rather than at module scope.
    #
    # This dataset mixes 126-token question/answer pairs with 306-token
    # passages, so block sizes vary batch to batch and the default
    # allocator fragments. The symptom is not a crash and not a constant
    # slowdown — it is a run that DEGRADES: 8.2 s/step at step 38, 63
    # s/step by step 53, GPU still reporting ~92% because kernels are
    # running, just not usefully. Expandable segments let the allocator
    # grow a block instead of hunting for a new one of the right size.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)

    out_dir = RUNS / pack
    data_f = out_dir / "dataset.jsonl"
    if not data_f.exists():
        print(f"no dataset — run --synthesize first ({data_f})", file=sys.stderr)
        return 2
    rows = [json.loads(l) for l in data_f.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        print("dataset is empty", file=sys.stderr)
        return 2

    adapter_dir = out_dir / (tag or "adapter")
    torch.manual_seed(seed)

    # Fail loudly instead of spilling into host RAM.
    #
    # On Windows the driver answers an over-large allocation by backing
    # it with shared system memory rather than raising OutOfMemory. The
    # job does not crash — it runs at PCIe speed. Measured on this
    # machine: 3.5 s/step with checkpointing inside VRAM, 35 s/step at
    # batch 32, 538 s/step at batch 16 without checkpointing, GPU
    # utilisation pinned at 0% the whole time because it was waiting on
    # transfers. It reads as a hung job rather than a bad batch size,
    # and the third of those attempts took the machine down with it by
    # pinning host RAM.
    #
    # Capping the fraction makes the allocator raise before the driver
    # can fall back, which turns a two-hour mystery into an immediate,
    # correct error message naming the batch size.
    if torch.cuda.is_available() and not multi_gpu:
        torch.cuda.set_per_process_memory_fraction(vram_fraction)

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def build(row):
        """Loss on the ANSWER only.

        Training on the question as well teaches the model to generate
        MCP-flavoured questions, which is not the capability being
        compiled and spends the budget on the wrong tokens.
        """
        if row.get("lm"):
            # A passage carries no question, so there is nothing to mask
            # and the loss runs over the whole text.
            ids = tok(row["a"] + tok.eos_token,
                      add_special_tokens=False)["input_ids"][:max_len]
            return {"input_ids": ids, "labels": list(ids),
                    "attention_mask": [1] * len(ids)}
        # A craft row trains under the craft system prompt — the one the
        # artifact will be generated under at eval. Mixing the two
        # system prompts across train/serve is the chat_prefix drift bug
        # in another costume.
        # Prefer the prompt the row was SYNTHESISED under. Older craft
        # datasets carry no `system` field, so they keep the historic
        # constant and stay reproducible.
        system = row.get("system") or (
            CRAFT_SYSTEM if row.get("craft") else SYSTEM)
        prompt = chat_prefix(tok, [{"role": "system", "content": system},
                                   {"role": "user", "content": row["q"]}])
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(row["a"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p_ids + a_ids)[:max_len]
        labels = ([-100] * len(p_ids) + a_ids)[:max_len]
        return {"input_ids": ids, "labels": labels,
                "attention_mask": [1] * len(ids)}

    ds = Dataset.from_list([build(r) for r in rows])

    if quant4:
        # QLoRA: the base in nf4 halves its footprint (8.5 -> ~4.5 GB),
        # which is what makes 4k-token artifacts trainable on 24 GB at
        # all — this architecture's linear-attention fast path is not
        # installed on Windows, and the torch fallback's activations at
        # 4096 tokens exceeded the card twice with a bf16 base. The
        # adapter's own weights stay bf16 and serve unchanged on the
        # bf16 base, the standard QLoRA deployment shape.
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            base, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        # device_map="auto" splits layers across every visible GPU.
        # A 4096-token artifact OOM'd one 24 GB card even at nf4 —
        # this architecture's linear-attention fast path is missing on
        # Windows and the fallback's activation peak lands on whichever
        # card holds the layer, so splitting the layers splits the peak.
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16,
            device_map="auto" if multi_gpu else {"": 0})
    model.config.use_cache = False
    model.enable_input_require_grads()

    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  base {base}")
    print(f"  {len(rows)} verified examples, rank {rank}, alpha {alpha}")
    print(f"  trainable {trainable / 1e6:.1f}M / {total / 1e9:.2f}B "
          f"({trainable / total:.2%})")

    args = TrainingArguments(
        output_dir=str(out_dir / "hf"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=checkpointing,
        report_to=[],
        seed=seed,
    )
    from transformers import TrainerCallback

    class StopWhenLearned(TrainerCallback):
        """Stop once the loss says the material is learned.

        Measured on motion-craft: 0.243 -> 0.160 -> 0.109 -> 0.074 by
        epoch 4 of 20, and the remaining sixteen epochs bought nothing
        but wall-clock. A fixed epoch count is a guess about a curve we
        can simply read, and on a fleet that guess is paid for on every
        pack.

        Deliberately conservative — it needs the threshold met twice in
        a row, because a single low step can be one easy batch rather
        than a converged model.
        """

        def __init__(self, floor: float):
            self.floor, self.hits = floor, 0

        def on_log(self, args, state, control, logs=None, **kw):
            loss = (logs or {}).get("loss")
            if loss is None:
                return
            self.hits = self.hits + 1 if loss <= self.floor else 0
            if self.hits >= 2:
                print(f"\n  early stop: loss {loss:.4f} <= {self.floor} twice "
                      f"at step {state.global_step} of {state.max_steps}",
                      flush=True)
                control.should_training_stop = True

    trainer = Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
        callbacks=[StopWhenLearned(early_stop_loss)] if early_stop_loss else [],
    )
    result = trainer.train()

    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))

    size = sum(f.stat().st_size for f in adapter_dir.rglob("*.safetensors"))
    meta = {
        "pack": pack, "base": base, "rank": rank, "alpha": alpha,
        "examples": len(rows), "epochs": epochs, "lr": lr, "seed": seed,
        "steps": int(result.global_step),
        "train_loss": round(float(result.training_loss), 4),
        "adapter_bytes": size,
        "trainable_params": trainable,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (adapter_dir / "compile.json").write_text(json.dumps(meta, indent=2),
                                              encoding="utf-8")
    # Every run is appended, never overwritten. Reporting the best of N
    # seeds without saying how many were run is the commonest way a
    # training result stops meaning anything.
    with (out_dir / "compiles.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")

    print(f"\n  {result.global_step} steps, final loss "
          f"{result.training_loss:.4f}")
    print(f"  adapter {size / 1e6:.1f} MB -> {adapter_dir}")
    print(f"  logged to {out_dir / 'compiles.jsonl'}")
    return 0


def do_forgetting(pack: str, base: str, tag: str, n: int) -> int:
    """What the 31-question control set structurally cannot see.

    The control set grades substrings. A model can keep answering
    "Tokyo" while its distribution over ordinary English has shifted
    badly — the answer stays gradeable and the damage stays invisible.
    So: take the BASE model's own greedy continuation of each control
    question, then score that exact text under both arms. The base
    prefers its own output by construction, so any rise in negative
    log-likelihood under the adapter is drift, measured on text the pack
    has nothing to do with, with no external corpus to download.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = RUNS / pack / tag
    if not adapter_dir.exists():
        print(f"no adapter at {adapter_dir}", file=sys.stderr)
        return 2

    control = [json.loads(l) for l in
               (HERE / "control.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip()][:n]
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    lora = model.base_model

    def nll(prompt_ids, answer_ids) -> float:
        ids = torch.cat([prompt_ids, answer_ids], dim=1)
        labels = ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100
        with torch.no_grad():
            return float(model(input_ids=ids, labels=labels).loss)

    deltas = []
    for item in control:
        text = chat_prefix(tok, [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": item["q"]}])
        p = tok(text, return_tensors="pt").to(model.device)

        lora.disable_adapter_layers()
        with torch.no_grad():
            out = model.generate(**p, max_new_tokens=64, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        a = out[:, p["input_ids"].shape[1]:]
        if a.shape[1] == 0:
            continue
        base_nll = nll(p["input_ids"], a)
        lora.enable_adapter_layers()
        ad_nll = nll(p["input_ids"], a)
        deltas.append((item["id"], base_nll, ad_nll, ad_nll - base_nll))
        print(f"  {item['id']:<28} base {base_nll:5.3f}  adapter {ad_nll:5.3f}"
              f"  {ad_nll - base_nll:+6.3f}")

    mean = sum(d[3] for d in deltas) / len(deltas)
    worst = max(deltas, key=lambda d: d[3])
    print(f"\n  mean NLL delta on {len(deltas)} control continuations: {mean:+.4f}")
    print(f"  worst: {worst[0]} {worst[3]:+.3f}")
    print("\n  This is drift on text the pack has nothing to do with. A near-"
          "zero mean with\n  the control set still at full marks is the pair "
          "of readings that supports a\n  no-forgetting claim; either one "
          "alone does not.")
    (RUNS / pack / "forgetting.json").write_text(
        json.dumps({"mean_nll_delta": mean,
                    "rows": [{"id": i, "base": b, "adapter": ad, "delta": d}
                             for i, b, ad, d in deltas]}, indent=2),
        encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--synthesize", action="store_true")
    ap.add_argument("--synthesize-craft", action="store_true",
                    help="brief -> artifact pairs gated by wp_theme_checks; "
                         "writes to training/<pack>-craft/")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--forgetting", action="store_true",
                    help="base-vs-adapter NLL drift on control continuations")
    ap.add_argument("--n-control", type=int, default=31)
    ap.add_argument("--teacher", default=DEFAULT_TEACHER)
    ap.add_argument("--host", default=None, help="ollama base URL for the teacher")
    ap.add_argument("--per-record", type=int, default=3,
                    help="unused when frames drive the count; kept for the "
                         "old free-form call")
    ap.add_argument("--keep", type=int, default=8,
                    help="verified forms admitted per record. 2 was "
                         "measured at +2/53; the floor for a fact to "
                         "become answerable is higher than that")
    ap.add_argument("--batch-frames", type=int, default=4,
                    help="frames requested per teacher call")
    ap.add_argument("--dup-ceiling", type=float, default=0.5,
                    help="reject a form whose question overlaps an accepted "
                         "one above this Jaccard")
    ap.add_argument("--include-records", action="store_true",
                    help="also train on each record's own text")
    ap.add_argument("--teacher-api", choices=sorted(PROVIDERS),
                    help="use a chat endpoint instead of local ollama; "
                         "frees both GPUs and runs records concurrently")
    ap.add_argument("--env-file",
                    help="KEY=VALUE file holding the endpoint and key")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--tag", default="adapter",
                    help="subdirectory for this adapter, e.g. seed2")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2,
                    help="effective batch is --batch x --accum; the step "
                         "count that matters is examples x epochs / that")
    ap.add_argument("--no-checkpointing", dest="checkpointing",
                    action="store_false",
                    help="disable gradient checkpointing. Leave it ON: the "
                         "adapter is small but the ACTIVATIONS are not — a "
                         "12288-wide MLP at batch 16 exceeds 24 GB, and "
                         "Windows answers by spilling to shared system "
                         "memory rather than raising OutOfMemory. Measured "
                         "cost of that spill: 538 seconds per step against "
                         "3.5 with checkpointing on. It reads as a hung job, "
                         "not as an allocation failure.")
    ap.set_defaults(checkpointing=True)
    ap.add_argument("--quant4", action="store_true",
                    help="QLoRA: nf4 base for long-sequence training on 24 GB")
    ap.add_argument("--early-stop-loss", type=float, default=0.08,
                    help="stop once training loss holds at/below this for two "
                         "logs; 0 disables. The loss curve is readable, so a "
                         "fixed epoch count is a guess paid for on every pack")
    ap.add_argument("--multi-gpu", action="store_true",
                    help="split base layers across all visible GPUs")
    ap.add_argument("--vram-fraction", type=float, default=0.92,
                    help="cap the allocator so an over-large batch "
                         "raises OutOfMemory instead of silently "
                         "spilling to host RAM at 1/100th the speed")
    a = ap.parse_args()

    if a.synthesize_craft:
        load_env_file(a.env_file)
        teacher = a.teacher
        if a.teacher_api and teacher == DEFAULT_TEACHER:
            teacher = PROVIDERS[a.teacher_api][2]
        return do_synthesize_craft(a.pack, teacher, a.host, a.teacher_api,
                                   a.workers)
    if a.synthesize:
        load_env_file(a.env_file)
        teacher = a.teacher
        if a.teacher_api and teacher == DEFAULT_TEACHER:
            teacher = PROVIDERS[a.teacher_api][2]
        return do_synthesize(a.pack, teacher, a.per_record, a.keep, a.host,
                             a.batch_frames, a.dup_ceiling, a.include_records,
                             a.teacher_api, a.workers)
    if a.train:
        return do_train(a.pack, a.base, a.rank, a.alpha, a.epochs, a.lr,
                        a.seed, a.max_len, a.tag, a.batch, a.accum,
                        a.checkpointing, a.vram_fraction, a.quant4,
                        a.multi_gpu, a.early_stop_loss)
    if a.forgetting:
        return do_forgetting(a.pack, a.base, a.tag, a.n_control)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
