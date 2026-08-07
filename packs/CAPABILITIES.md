# Authoring a capability pack

A pack ships knowledge in **two carriers**, and which carrier a thing
belongs in is the first and most consequential decision:

| | corpus → **context** | constitution → **weights** |
|---|---|---|
| holds | facts, versions, API surfaces | rules applied on every token |
| gated | per query, by similarity | per mount, by task |
| updates | re-seal the pack | recompile the adapter |
| fails by | nothing — retrieval is gated | domain intrusion, if you compile facts |

**Ask of every rule: is it consulted, or is it exercised?** A fact a
model looks up belongs in the corpus. A rule that must hold on every
line of output belongs in the constitution and gets compiled.

Measured basis for the split, on frozen Qwen3.5-4B:

- Compiling 213 arbitrary MCP facts: 11/53 → 37/53, and cost 7 of 31
  control questions to domain intrusion (`stdio transport provides
  guaranteed ordered delivery` for a TCP question). Retrieval scores
  52/53 on the same facts with no control damage. **Facts stay in
  context.**
- Compiling 23 WordPress craft rules: 3.4/14 → **14.0/14**, 12 of 12
  briefs fully compliant. The same rules pasted into context reached
  4.9/14. **Rules go in weights.**
- The rulebook-in-context arm scored *below* its own bare arm for a
  frontier model on both domains (8.4 → 6.8; 5.9 → 5.2): carrying rules
  costs output budget and completion collapses. Weights carry them free.

---

## 1. Write the constitution

One rule per `## ` heading, terse, each stating a constraint that a
checker can test. Rules that cannot be mechanically checked can still be
written — they simply will not be measured, and unmeasured rules do not
belong in a certificate.

Every rule should exist because *the default output violates it*. A rule
the model already follows buys nothing and dilutes training.

## 2. Write the grader — this is the product

`<pack>/craft.py` exposes:

```python
ARTIFACT      # one line: what gets generated
CHECKLIST     # the mechanical constraints as exact paths/values
grade_text(text) -> (passed, total, per_check | None)
train_briefs() / holdout_briefs() -> list[str]
canonical(raw) -> str | None      # optional: normalise before training
```

Four grader families cover most domains:

1. **structured artifact** — parse JSON/YAML/TOML, assert paths
   (`wp_theme_checks.py`)
2. **execution** — run it, assert exit code and postconditions
   (the pack-api benchmark)
3. **source text** — parse declarations, assert values and relations
   (`motion-craft/craft.py`)
4. **regex with teeth** — required and forbidden markers
   (`evaluate._alt_matches`)

Rules that pay for themselves:

- **Test the grader in both directions before using it.** Plant a
  compliant artifact and a deliberately broken one. A grader that cannot
  fail inflates silently; a grader that cannot pass deflates silently,
  and deflation lands on your *best* artifacts. Round one of
  motion-craft admitted 27/72 — three of the failures were checker bugs,
  one of which punished the correct accessibility pattern
  (`transition-duration: 0.01ms` inside the reduced-motion override) and
  one of which failed artifacts for using `calc(var(--i) * 60ms)`, a
  *better* stagger than the literal delays the check expected.
- **Attribute before fixing.** Roughly half of all admission failures
  are the checker, not the teacher.
- **Hold out by construction.** Sealed briefs must use subjects the
  training briefs never contain, and the brief generator must be checked
  for collisions — a linear index map silently collapsed 72 intended
  combinations into 12.

## 3. Compose the briefs

Every brief should require the same mechanical skeleton so the checks
apply unconditionally, while the *subject* varies. motion-craft asks
every brief for a staggered collection, a dismissible overlay and an
interactive control; only the product and mood change.

## 4. Synthesise, with a mechanical checklist

```bash
python packs/compile.py --pack <pack> --synthesize-craft \
    --teacher-api qwen --teacher qwen3.6-flash \
    --env-file <keyfile> --workers 24
```

**Give the teacher the checklist, not just the prose.** With the
constitution alone, admission was 9/72 and the misses were almost
entirely *computable* constraints — 42 of 63 omitted one flag the
constitution names in a single sentence. Restating the checked
constraints as exact paths took the same teacher to 71/72.

The verifier is what makes a cheap teacher safe: `qwen3.6-flash` admits
about three of four where the top tier admits four of four, at a quarter
the latency, and the fourth is discarded rather than trained. Near
misses are logged, never admitted — they are preference pairs for a
later on-policy stage, not positives.

## 5. Train

```bash
.venv-compile/Scripts/python packs/compile.py --pack <pack>-craft --train \
    --base Qwen/Qwen3.5-4B --rank 16 --alpha 32 --epochs 20 \
    --batch 1 --accum 4 --max-len 3600 --quant4 --tag v1
```

- Around 30–70 fully compliant artifacts is enough; verification quality
  binds long before volume does.
- Gradient checkpointing stays **on**. The adapter is small; the
  activations are not.
- `--vram-fraction` makes an over-large batch raise `OutOfMemory`
  instead of spilling into host RAM at a hundredth the speed — on
  Windows the driver will not refuse, it will simply run 150× slower and
  look like a hung job.
- Windows tops out near 3k tokens per sequence for this architecture
  (no `flash-linear-attention` build); Linux is the correct home for
  anything longer.

## 6. Measure against sealed briefs and a pinned bar

```bash
python packs/evaluate_craft.py --pack <pack> --adapter <pack>-craft --corpus
```

Report five rows: bare, rulebook-in-context, compiled, compiled+corpus,
and a **pinned frontier model taking the same exam**. The frontier is an
examinee, never a judge — the deterministic checker stays the only
grader, or the rating is circular. Pin the model id and date; a claim of
"92% of frontier" must name which frontier and when.

Then check what pass rates cannot see: generate several sealed artifacts
and confirm they are *structurally different* from each other and are
not verbatim training copies.

## 7. Publish

```bash
python packs/bundle.py build <pack>-craft --tag v1 --version 0.1.0 --key <sk>
python packs/bundle.py verify packs/dist/<pack>-0.1.0.ycap
python packs/bundle.py install packs/dist/<pack>-0.1.0.ycap --db mem.db
python packs/bundle.py list --db mem.db
```

A `.ycap` carries the adapter, the constitution it was compiled from,
the training provenance, the grader's digest, and the efficacy table —
signed over the manifest's canonical bytes. Editing any number breaks
the signature. Re-authoring a check changes the grader digest, so a
certificate cannot outlive the instrument that produced it.

Capabilities install **beside the database**, mirroring packs: a
database plus its knowledge plus its capabilities copy and back up as
one unit.

## 8. Serve

**Name the adapter per request.** This is the only shape that is safe
for measurement or for more than one caller:

```json
{"model": "motion-craft-craft", "messages": [...]}   // adapter applied
{"model": "base",               "messages": [...]}   // adapter disabled
```

Every measured number in this repo uses it, and it is what vLLM exposes
natively — an adapter is served as its own model id, so "use this
capability for this agent" is a field in the request rather than a
state change on the server.

For a single operator at a terminal there is also a stateful pair:

```bash
python packs/cap.py status
python packs/cap.py mount <pack>
python packs/cap.py ask "..."
python packs/cap.py unmount
```

Convenient, and **wrong for anything else**. Mount state lives on the
server. Two clients sharing a daemon each read a flag the other is
setting — that produced byte-identical "compiled" and "bare" artifacts
here once, with nothing in either output looking wrong. llama.cpp's
`POST /lora-adapters` sets scale globally too, so with N agents against
one serving process, one agent mounting changes every other agent's
weights mid-session. A request that names its adapter cannot be
answered by whatever the last request left mounted.

Mounting is otherwise a flag flip on resident tensors, and unmounting
restores the base exactly — the adapter sat beside the weights and
never modified them, the weights-tier form of the guarantee that
unmounting a pack leaves the host byte-identical.

A capability compiled against a different base **revision** is refused,
not mounted. A LoRA is a delta against specific weights; applied to
others it is noise wearing a capability's name.
