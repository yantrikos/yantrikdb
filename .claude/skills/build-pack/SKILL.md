---
name: build-pack
description: Author, measure and publish a YantrikDB knowledge pack. Use when asked to create a pack, turn documentation or a domain into a pack, package knowledge or skills for a local model, or publish to packs.yantrikdb.com. Covers the three tiers (constitution / corpus / coverage), the measurement that decides whether a pack is worth shipping, and the authoring rules derived from how the engine actually behaves.
---

# Building a pack

A pack is a sealed, signed YantrikDB file a model mounts to gain
knowledge and behaviour it lacks, and unmounts without a trace. This
skill covers authoring one that is *worth shipping* — which is decided
by measurement, not by how good the corpus looks.

## The one thing to get right first

**A pack shifts knowledge and procedure. It does not raise raw
capability.** Every claim must be phrased so it can be checked:

- ✗ "makes your model write like Shakespeare"
- ✓ "raises compliance with Shakespearean craft rules from 7/12 to 10/12
  on qwen3.5:4b, on checks written down before the run"

If you cannot state the claim as a measured delta on named checks, you
do not yet know whether the pack works.

## Which of the four shapes is this?

Measured across three model sizes; the shape predicts the payoff and the
buyer.

| Shape | Example | Typical lift | Who buys it |
|---|---|---|---|
| **Knowledge** the model cannot have | private codebase, post-cutoff API | huge (1/20 → 18/20) | anyone running a local model |
| **Method** it knows but doesn't apply | reasoning discipline, review procedure | large at *every* size (+5 to +8) | everyone — the most underserved shape |
| **Framework** house rules | React, WordPress conventions | moderate; ceiling is high already | small-model operators |
| **Style** craft rules | period voice, brand tone | real but smallest | small-model operators |

Two findings worth carrying: **method packs lift even a 27B**, because
models know what a procedure is and reach for it only when it is in
front of them. **Craft and domain packs are inverse to model size** — a
27B often scores near ceiling unaided, so say who the pack is for
instead of promising uniform lift.

## The three tiers

```
pack/
  pack.toml         identity, namespace, ingest defaults, coverage
  corpus.md         the knowledge — retrieved on similarity
  constitution.md   the rules — injected on EVERY turn (optional)
  eval.jsonl        questions with deterministic expectations
```

**Corpus (tier 2) — what it knows.** Retrieved by similarity, unbounded.

**Constitution (tier 1) — what it does.** Injected unconditionally,
~1500 token budget enforced at seal time. This tier exists because
similarity retrieval cannot carry a hard rule: measured on
rule-application tasks, retrieval bought +1/+1/+0 while the constitution
bought +5/+4/+5. **7 of 8 tasks retrieved zero facts** — the rule was in
the pack, stored correctly, and never surfaced. Put a rule here only if
it fails when absent; everything else belongs in the corpus, because
every constitution line costs tokens on every turn.

**Coverage (tier 3) — what it covers.** Three to five short phrases. A
model does not consult knowledge it does not know exists.

## The tiers must be in sync, and that is checkable

The two tiers fail together in a way neither shows on its own. The
constitution says *"declare presets and then apply them under styles"*;
the corpus holds the worked example that demonstrates it; and the model
never sees the example, because the record is mostly code and does not
retrieve on its topic. **The rule arrives with no evidence, the evidence
is never delivered, and both files look correct in isolation.**

```bash
python packs/lint_pack.py <pack>      # rules without evidence, records nobody can reach
```

Measured on `wordpress-theme`: **7 of 23 rules had no corpus record above
the 0.55 floor** — the pack instructed the model to do things it could not
show — and **every worked example was unreachable**. Two laws came out of
fixing it, and both are enforced now: `build.py` warns, `lint_pack.py`
proves it per record.

### Law 1 — a record that is mostly code does not retrieve

The bundled embedder is 64-dimensional, so an embedding is dominated by
whatever the record has most of. A 2.5 KB `theme.json` example was
unreachable by **every** query tried, while the shortest example — most
prose, least code — won even for the other's queries. Retrievability is
inversely proportional to code volume.

So: **lead with prose that names the subject, and keep the snippet
short.** Ten to twenty lines of code inside sixty percent prose retrieves;
a whole file does not. A large worked example belongs split into
rule-aligned fragments, not stored whole. If a complete file must ship,
ship it as a `reference/` directory in the pack source and teach from
fragments of it.

### Law 2 — write in the vocabulary a consumer queries with

Constitution headings are imperatives, and imperatives make terrible
queries. `"Layout type is chosen deliberately"` retrieves nothing at
0.386; `"constrained flex grid layout type"` finds the right record at
0.679. The embedder matches concrete technical vocabulary — identifiers,
API names, error strings — not abstract instruction.

Corpus headings therefore carry the concrete terms someone would actually
ask about. The rule can stay imperative; the record it depends on must
not.

### Pair every rule with a record

Each constitution rule should have at least one corpus record that shows
it, reachable by the task language around that rule. A rule with no
retrievable support is an assertion the pack cannot back up — and on a
generation task the model will follow the rule shape without the detail
that makes it work, which is how a theme ends up structurally correct and
visually unstyled.

Run the linter before publishing. Zero orphan rules is the bar.

## Authoring rules that come from engine behaviour

- **One fact per record.** Retrieval serves records, not documents. A
  record holding five facts gets served whole when one is relevant, and
  ranks poorly because its embedding is the average of five directions.
- **Each record stands alone.** It will be retrieved without its
  neighbours.
- **Ingest at importance 0.6, never 1.0.** Write-time calibration
  compresses new high marks once a namespace passes 8 writes at a high
  mean, so a pack stamping everything 1.0 ranks its own later facts
  *below* its earlier ones.
- **`source = "document"`.** The provenance gate refuses
  `source=inference` claiming `kind=fact`.
- **Keep procedural rules far apart in meaning.** Small models blend
  near-neighbour rules; both 4B regressions measured were "answered with
  the adjacent rule".
- **Ground every claim.** Cite the source in the corpus (`_cite:` lines
  are stripped before embedding). Never write a fact you have not
  checked — a pack that ships recalled-but-unverified content is exactly
  the laundering the provenance gate exists to prevent.

## Workflow

```bash
# 1. author, then build
python packs/build.py packs/<name>

# 2. are the tiers in sync? (rules with no evidence, unreachable records)
python packs/lint_pack.py <pack>
python packs/lint_evals.py <pack>

# 3. does it teach? (knowledge questions + attach-harm control)
python packs/evaluate.py --model qwen3.5:4b --pack <name>

# 4. if it has a constitution, does that tier earn its tokens?
python packs/evaluate_tiers.py --model qwen3.5:4b --pack <name>

# 5. sign and publish
yantrikdb pack keygen                       # once; keep the secret offline
yantrikdb pack sign packs/dist/<name>-<v>.ydbpack --key <secret>
# upload at https://packs.yantrikdb.com/dashboard
```

## The gates a pack must pass

1. **It teaches.** A real delta on questions the model fails cold.
2. **It does no harm.** The unrelated control set must not regress.
   A pack that wins its category by capturing attention is a bad pack —
   measured: ungated top-k injection took a control set from 12/12 to
   **5/12**. Consumers must gate injection on *similarity* (floor ~0.55),
   not on the composite recall score.
3. **Its constitution is justified.** If corpus ≈ constitution, delete
   the constitution — retrieval is strictly cheaper.
4. **It generalizes.** A held-out score far below the public score means
   the pack was tuned to its own eval.
5. **It survives the scanners.** No credentials, no injection phrasing,
   no bidi characters. Rules and skills are graded stricter than plain
   memories because they are what a consumer's model acts on.

## Writing evaluation questions honestly

Deterministic string matching only — never an LLM judge, which would put
a second unvalidated model between the pack and its own score.

- Prefer **identifiers, names and numbers** over common words; "3" as an
  expectation is guessable, `YANTRIKDB_READ_POOL` is not.
- Ask questions the model **fails cold**. If the baseline already passes,
  the question measures the model, not the pack.
- For rule-application, phrase tasks in an **unrelated domain** so
  nothing invites the model to look for the rule.
- **Write the checks before the run** and report them as written. When a
  check turns out to be wrong, say so and fix it *for the next run* —
  never retune after seeing results.
- Grade style by **density and consistency**, not vocabulary breadth: a
  distinct-count threshold punishes disciplined output. This is a real
  bug we shipped — a 27B wrote flawless period verse and scored zero
  because it used four archaic markers consistently where the check
  demanded five distinct ones.

## Reference packs

`packs/` in the engine repo holds seven worked examples spanning all
four shapes — read `yantrikdb-engine` (knowledge), `einstein-method`
(method), `react-craft` (framework), `shakespeare-voice` (style) before
authoring a new one.
