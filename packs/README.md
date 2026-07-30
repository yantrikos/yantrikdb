# Sample packs

Reference packs for the mountable-knowledge marketplace, plus the tooling
that builds and scores them. See [`docs/PACKS.md`](../docs/PACKS.md) for
the design and [`crates/yantrikdb-core/src/engine/pack.rs`](../crates/yantrikdb-core/src/engine/pack.rs)
for the engine.

```bash
python packs/build.py --all                      # source dirs → sealed .ydbpack
python packs/evaluate.py --model qwen3.6:27b     # measure the lift
```

Building requires a `yantrikdb` wheel with pack support installed.

## What a pack is for

A pack is a sealed, single-file database a model mounts to gain knowledge
it does not have and unmounts to give back. The number that belongs on a
marketplace listing is not the row count — it is **how much better a
model gets when the pack is mounted, and what it costs on everything
else.**

## The packs

| Pack | Facts | Shape of value |
|---|---|---|
| `yantrikdb-engine` | 45 | Knowledge the model cannot have — private, post-cutoff engine internals |
| `agent-memory-discipline` | 19 + 5 rules | Procedure the model half-knows — operating rules with non-obvious rationale |
| `shakespeare-voice` | 14 + 6 rules | Style as craft: Early Modern grammar, verse shape, rhetorical devices |
| `einstein-method` | 8 + 5 rules | Method as procedure: thought experiments, invariance, limit cases |
| `c-safety` | 40 + 20 rules | The C that compiles clean and has a CVE in it: lifetime, UB, integer rules |
| `java-modern` | 47 + 20 rules | Java on a current LTS: records, virtual threads, and the contracts that fail silently |
| `php-modern` | 38 + 17 rules | PHP 8 language and the security defaults that decide if the app is exploitable |
| `react-craft` | 36 + 15 rules | React 19: Actions, `use()`, RSC boundary, and the bugs that only appear under load |
| `wordpress-expert` | 46 + 15 rules | Plugin and theme engineering: load order, REST, WooCommerce, and review-grade security |

The four language and framework packs share a selection rule, and it is
the opposite of what a "learn X" pack would do. A competent model
already knows how C, Java, PHP and React work, so explaining them buys
nothing — the first `react-craft` (10 general facts) moved a 27B model
by **+1** for exactly that reason. What is left worth packing is the
delta between what the model knows and what is true, plus the defects it
produces by default. So every entry is one of:

- **a version fact it cannot have** — React 19 `ref`-as-a-prop, PHP 8.4
  property hooks, Java 21 sequenced collections;
- **a rule whose violation compiles clean and passes a smoke test** —
  `strncpy` not terminating, `foreach` by reference leaving its
  reference bound, check-then-act on a `ConcurrentHashMap`, PDO's
  emulated prepares, a `fetch` in an effect racing itself;
- **a correction to training data** — Java string templates
  (`STR."…"`) previewed in JDK 21 and 22 and were withdrawn in 23, so
  the syntax is in every model's training set and compiles nowhere.
  Same shape for `forwardRef`, deprecated in React 19.

## Use-case validation: can a 4B model do the thing?

The honest claim first: a pack shifts **knowledge and procedure, not raw
capability**. The testable form of "write like Shakespeare" is "comply
with Shakespeare's craft rules on checks written down before the run" —
and every check is a deterministic function of the output (regexes,
syllable estimates, code patterns), pre-registered in
`evaluate_usecases.py`. No LLM judges anything.

Per-check totals across 4 tasks each, three model sizes:

| Use case | qwen3.6:27b | qwen3.5:4b | granite4:3b |
|---|---|---|---|
| shakespeare-voice (style) | 8 → 8 † | 7 → **10** | 8 → 8 † |
| einstein-method (method) | 4 → **10** | 4 → **10** | 3 → **11** |
| react-craft (framework) | 11 → **12** | 9 → **12** | 11 → **12** |
| wordpress-expert (domain) | 11 → 11 | 9 → **12** | 6 → **9** |

Three findings, one per row-pattern:

**Method packs are the universal win.** einstein-method lifts +6, +6 and
+8 across every model size — even the 27B, whose baselines elsewhere sit
near ceiling, does not *apply* a reasoning procedure unprompted. Models
at every size know what a thought experiment is and reach for one only
when it is in front of them. This is a marketplace category nobody is
selling: reasoning-discipline packs, not knowledge packs.

**Craft and domain packs are inverse to model size.** wordpress-expert
buys granite4:3b +3 and buys the 27B nothing — it already scores 11/13
unaided. Same for react-craft, where big-model baselines start at 11/12.
The buyer of a craft pack is the small-model operator; a listing should
say so rather than promising uniform lift.

**† The flat Shakespeare rows are a grader artifact, reported rather
than retuned.** The 27B's mounted verse is genuinely period — "Thy
steadfast heart doth stand against the tide... Nor Time's sharp scythe
thy truth unfold" — but uses 4 distinct archaic markers consistently,
and the pre-registered check demands ≥5 *distinct* markers: it measured
marker variety where consistency was the craft goal. Pre-registered
means the number stands as printed; the check gets revised before the
next run, never after this one. (The 4B's +3 is real under the same
check.)

What actually changes, verbatim: baseline praise-poem opens *"Your faith
remains as steady as the morning sun"* — modern register throughout.
Mounted: *"O thou whose heart doth never yield to fear or change, / Thy
faith is stronger than the winter's biting air; / Though Time with
scythe shall carve thy beauty from this sphere, / Yet here in verse thy
virtue lives"* — consistent thou/thy/doth, and the Time-with-scythe
motif lifted straight from the corpus. The Einstein pack turns flat
physics answers into ones that open with a thought experiment and argue
from invariants; the React pack closes the gaps a 4B reliably leaves
(functional updates, labelled inputs); the WordPress pack adds the
capability check and output escaping that plugin reviews reject code for
missing.

Each case also shows honest misses: verse meter still fails on one task,
`limit_case` never fires for the mountain-clock task, the shortcode task
loses `wp_hooks` in both conditions. The per-task marks in the harness
output name every one.

Both are grounded content. Every fact in `yantrikdb-engine` is checked
against the source tree at schema v38; the corpus carries `_cite:` lines
pointing at file and line. Nothing in either pack is recalled from
memory and asserted as fact.

## Measured efficacy

Deterministic string-match scoring against a fresh empty host — the
qwen-27B scenario, a capable model with no memories of its own. No LLM
judge: a judge would be a second unvalidated model standing between the
pack and its own efficacy number, and that number is the whole product
claim.

`control` is 12 unrelated general-knowledge questions run in both
conditions. It is there to catch a pack that wins its category by
capturing attention and wrecking everything else.

> **Revised twice, in both directions, and both times the grader moved
> rather than the pack.**
>
> *2026-07-29, down.* The figures previously read 18/20 across the board.
> The grader matched substrings, so `"no"` matched *k**no**w* and `"20"`
> matched "20**24**"; every model lost exactly 3 points once matching
> became word-boundary aware. A uniform drop across six models is the
> signature of a grader fix, not a regression.
>
> *2026-07-30, up.* Word-boundary matching then exposed the opposite bug:
> an expectation written as a truncated stem can never match its own
> inflection. `forget-semantics` expected `"tombston"`, which does not
> match "tombstoned" — so every model that answered correctly was scored
> wrong. An audit found 35 such stems across 11 packs, each one silently
> *deflating* a published number. That direction hides better than the
> first, because a lower score reads as an honest one.
>
> Both figures are withdrawn; the table below is the corrected run.
> `lint_evals.py` now checks the matcher in both directions — that it can
> fail, and that it can pass.

| Model | Pack | Baseline | Mounted | Control |
|---|---|---|---|---|
| gemma3:270m | yantrikdb-engine | 1/20 | 10/20 | 7/12 → 8/12 |
| qwen3.5:0.8b | yantrikdb-engine | 0/20 | **16/20** | 8/12 → 8/12 |
| qwen3.5:2b | yantrikdb-engine | 2/20 | 14/20 | 10/12 → **8/12** |
| granite4:3b | yantrikdb-engine | 2/20 | **16/20** | 12/12 → 12/12 |
| qwen3.5:4b | yantrikdb-engine | 2/20 | **16/20** | 12/12 → 12/12 |
| qwen3.6:27b | yantrikdb-engine | 1/20 | **16/20** | 11/12 → 11/12 |
| gemma3:270m | agent-memory-discipline | 1/12 | 4/12 | 7/12 → 8/12 |
| qwen3.5:0.8b | agent-memory-discipline | 1/12 | 8/12 | 8/12 → 8/12 |
| qwen3.5:2b | agent-memory-discipline | 3/12 | 8/12 | 10/12 → 10/12 |
| granite4:3b | agent-memory-discipline | 5/12 | 9/12 | 12/12 → 12/12 |
| qwen3.5:4b | agent-memory-discipline | 5/12 | **10/12** | 12/12 → 12/12 |
| qwen3.6:27b | agent-memory-discipline | 6/12 | **11/12** | 11/12 → 11/12 |

### The pack, not the model, supplies the knowledge

Four models spanning **0.8B to 27B** — a 34× parameter range — start
between 0 and 2 out of 20 and all land on exactly **16/20** with the same
pack mounted. Model size predicts almost nothing about the baseline
(being bigger does not help you know a private codebase) and nothing at
all about the mounted score. `qwen3.5:2b` is the one model between the
floor and the plateau, at 14/20.

That is the marketplace thesis in one column: an 0.8B model with the
right pack answers domain questions as well as a 27B model with the same
pack. The pack is the capability.

### Where the floor is

Convergence does not continue forever. `gemma3:270m` reaches only 10/20,
and its control baseline is 7/12 against 12/12 for the 3B and 4B — it is
not merely worse at the pack's subject, it is worse at everything, and
there is not enough model left for retrieved facts to be assembled into
an answer. **The floor sits between 270M and 800M.**

The other signal at the small end is attach-harm. `qwen3.5:2b` is the
only model whose control set *dropped* when the pack was mounted, 10/12 →
8/12. One run at n=12 is not proof, but it points the same way as an
earlier finding — smaller models blend near-neighbour facts more — and it
is the reason a small-model listing should carry its control column
rather than only its headline.

### Two packs, two sizes of claim

A pack of facts a model cannot possibly know moves it from ~1/20 to
16/20. A pack of operating rules it already half-knows moves it from
~5/12 to ~10/12 — real, but a different and much smaller claim, and one
that keeps some dependence on model size rather than flattening the way
the facts pack does. Both belong on a listing; publishing only the first
would be marketing, not measurement.

One caveat on the control column: `qwen3.6:27b` scores 11/12 rather than
12/12 in *both* conditions, because it answered the git question with
`git checkout` where the question explicitly asks for the one-step form
and the expectation accepts either `checkout -b` or `switch -c`. Earlier
revisions of this document called that a strictness artifact; re-reading
the question, it is a genuine incomplete answer. Either way it is
identical in both conditions, so it does not affect the attach-harm
measurement.

`agent-memory-discipline` also produces a regression on every model
below 27B — `searchable-phrasing` on qwen3.5:4b, `hard-constraint` and
`granularity` on granite4:3b, `granularity` on qwen3.5:2b,
`relative-dates` on gemma3:270m. In each case the model answered with
the wrong neighbouring rule: asked about unresolvable pronouns, it
explained the relative-dates rule instead. Retrieving several
near-adjacent procedural rules can blend them, and smaller models blend
more. `qwen3.6:27b` is the only model with none, which is the same
pattern as the mounted column: this pack keeps a dependence on model
size that the facts pack does not. The harness names every such case
rather than netting it out of the total.

## The finding that matters most: unconditional injection is harmful

The first run of this harness injected the top 5 retrieved facts on
**every** question, including unrelated ones. Result:

| | Pack questions | Control |
|---|---|---|
| Unconditional top-k | 1/20 → 18/20 † | 12/12 → **5/12** |
| Similarity-gated | 2/20 → 18/20 † | 12/12 → **12/12** |

† Both rows are from the pre-2026-07-29 grader and are shown only for
the *comparison between them*, which the grader change affects equally.
The absolute figures are superseded by the table above.

Mounting the pack destroyed the model's ability to answer questions the
pack had nothing to do with. Not because the content was wrong, but
because injected irrelevant context convinced the model it was only
allowed to answer from that context:

> **Q:** What is the capital city of Japan?
> **A (pack mounted, ungated):** *I do not know; I cannot answer questions
> about general knowledge using only the provided software system
> documentation.*

That is the attach-harm failure mode in its purest form, and it is
invisible to any evaluation that only measures the pack's own category.

**Gate on similarity, not on the composite recall score.** Measured
against `yantrikdb-engine`:

| | top-1 similarity | top-1 composite score |
|---|---|---|
| On-topic queries | 0.65 – 0.79 | 0.83 – 0.91 |
| Off-topic queries | 0.09 – 0.45 | 0.37 – 0.69 |

Similarity separates cleanly; the composite score overlaps, because it
folds in importance and recency, which are near-uniform across a freshly
built pack and therefore carry no relevance signal. The harness uses a
floor of `0.55` and injects nothing when no hit clears it.

This is a consumer-side rule, not an engine bug — but it is the
difference between a pack that helps and a pack that quietly makes a
model worse, so it belongs in the pack documentation rather than in a
footnote.

## The impossible-by-construction benchmark

Every other number here measures a pack making a model *better*. This
one measures a pack making something possible that was structurally
zero: **writing working code against an API younger than the model.**

The `yantrikdb-pack-api` pack documents the pack API itself —
`seal_pack`, `sign_pack`, `mount_pack`, `install_pack`,
`trust_publisher` — which was introduced on 2026-07-28. No model has it
in training data, and unlike every public benchmark that is *provable by
construction*, not assumed: the API is younger than every model tested.

Grading is execution: the generated script runs against the real
installed wheel; pass = exit 0 plus the postcondition (the pack file
exists, stdout reports the right count or trust tier). No string
matching, no judge — the interpreter is the grader.

Four tasks, both models:

| Task | 27B baseline | 4B baseline | Mounted (both) |
|---|---|---|---|
| seal a namespace into a pack | `no attribute 'Database'` | `cannot import 'Database'` | **PASS / PASS** |
| mount a pack, count mounts | `cannot import 'Database'` | `SyntaxError` | **PASS / PASS** |
| keygen → sign → trust → mount signed | `no attribute 'generate_keypair'` | `cannot import 'Publisher'` | **PASS / PASS** |
| durable install surviving reopen | `no attribute 'Database'` | `cannot import 'YantriDB'` | fail / fail |

Both models: **0/4 → 3/4**, identical pattern. The 4B's hallucinations
are the richer exhibit — it invented a `Publisher` class and a typo'd
`YantriDB` — and with the pack, a model 7× smaller than the 27B
completes the same sign-and-trust flow against an API neither could
touch unaided. Both mounted failures are on the same task (the durable
install), reported as-is.

The baseline column is the point: the model doesn't do *worse* without
the pack — it hallucinates a plausible API (`yantrikdb.Database`,
`generate_keypair`) and the interpreter refuses it. 0/4 is structural,
not a low score. Mounted, the same model completes the full
sign-and-trust flow it could not have seen. The one mounted failure is
reported as-is: it invented a `db.closed` attribute the API does not
have — a pack narrows hallucination, it does not abolish it.

## Evaluators: signed certificates over held-out sets

`evaluate.py` is seller-runnable, which makes self-reported numbers
worthless the day money is involved — a seller scores 100% by asking
questions whose answers are verbatim in their corpus. `certify.py` is
the structural fix:

1. **The evaluator holds the questions.** The held-out set is never
   published; a seller cannot tune a pack against questions they have
   never seen. (`holdout/yantrikdb-engine.jsonl` is a demonstration set —
   real held-out sets stay private to the evaluator.)
2. **The result is signed over the pack's content digest**, so a
   certificate binds to one exact build. Re-seal the pack — one changed
   row — and the certificate no longer applies.
3. **Buyers verify offline** with the evaluator's public key: the same
   trust-on-first-use model as publisher signing. No portal, no API.
4. **Attach-harm is a gate, not a footnote**: a control regression fails
   certification outright.

First real certificate, issued and verified in this repo:

```
certified yantrik/yantrikdb-engine@0.1.0
  held-out:    0/8 -> 7/8        (questions the eval author never saw)
  attach-harm: 12 -> 12 / 12  PASS
```

The held-out rate (0→7/8) matching the public-set rate (2→18/20, old
grader) is
itself a finding: the pack generalizes to unseen questions rather than
being tuned to its own eval.

Verified refusals: an inflated score breaks the signature; a valid
certificate from a *different* evaluator is rejected as such (trust is
per-key, not per-format); a certificate for a previous build fails the
digest match.

## Authoring conventions

These come from the engine's own behaviour, and ignoring them degrades a
pack silently:

- **One fact per record.** Retrieval serves records, not documents. A
  record holding five loosely related facts gets served whole when one is
  relevant and ranks poorly, because its embedding is the average of five
  directions.
- **Ingest at importance 0.6, never 1.0.** Write-time calibration
  compresses new high marks once a namespace passes 8 writes at a high
  mean, so a pack stamping everything 1.0 ranks its own later facts
  *below* its earlier ones.
- **`source = "document"`.** The provenance gate refuses
  `source=inference` claiming `kind=fact`. A pack asserting authored
  knowledge declares itself a document, which is what it is.
- **One namespace per pack**, matching `pack.toml`. `seal_pack` scopes
  the export to it, so anything outside it never ships.
- **Write records to stand alone.** A record that only makes sense next
  to its neighbours will be retrieved without them.

## Layout

```
packs/
  build.py                    corpus.md + pack.toml → sealed .ydbpack
  evaluate.py                 baseline vs mounted, with attach-harm control
  control.jsonl               unrelated questions, shared by every pack
  <pack>/pack.toml            identity, namespace, ingest defaults
  <pack>/corpus.md            facts, one per `## ` heading
  <pack>/eval.jsonl           questions + deterministic expectations
  dist/                       built packs (gitignored)
```

A sealed pack is about 1 MB before content, because a YantrikDB file
carries the full schema. Content is cheap by comparison — the 19-fact
pack is 1048 KB and the 45-fact pack is 1104 KB, so roughly 2 KB per
fact. Pack size is dominated by the schema floor until a pack runs to
thousands of records.

## Not yet included

The longevity/ITP domain pack — the flagship demo, where a 27B model
confidently predicts a large effect for 17α-estradiol in female mice and
the truth is zero — is **not** here yet. Its ground-truth values are
currently entered from recall and flagged unverified in the source
project. Shipping them as vendor-asserted fact is exactly the laundering
the provenance gate exists to prevent. Verifying those values against
their publications is the prerequisite, and it is a research task, not a
packaging one.
