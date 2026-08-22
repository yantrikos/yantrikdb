# BEAM: isolating memory quality from answerer quality

**Date:** 2026-08-12 · **Engine:** yantrikdb 0.13.4 · **Harness:** [vectorize-io/agent-memory-benchmark](https://github.com/vectorize-io/agent-memory-benchmark) · **Dataset:** BEAM-100K, all 400 queries

---

## The problem with published memory benchmarks

An end-to-end agent-memory score confounds at least three things:

1. the memory system's retrieval quality
2. whatever processing happens at ingest
3. the model that reads the retrieved context and writes the answer

AMB's own README notes that generation setup matters and that small changes in
prompts or models move accuracy substantially. We measured that directly:
**swapping only the answerer, holding retrieval identical, moved our score by
~9 points.**

So when two systems report different end-to-end numbers under different
answerers, the difference is not attributable to either system's memory. This
page separates the two.

---

## Method: frozen-context evaluation

Hindsight publishes per-query result files containing the **exact context
string** they injected for each query, for two configurations. Every one of the
400 BEAM-100K `query_id`s matches ours.

That makes the following possible:

```
A   yantrikdb context   (rag)          ─┐
R   hindsight context   (rag)          ─┼─→ same answerer → same judge → score
E   hindsight context   (single-query) ─┘
```

Answerer and judge are held fixed (`deepseek-v4-flash:0731`, temperature 0)
across all three conditions. The only thing that varies is which memory system
produced the context, and under which configuration.

**A and R are mode-matched** — both are `rag`-mode runs — so A-vs-R is the
primary comparison. E is their weaker published configuration, included
because it was the original comparison and because it calibrates how much
their own configuration choice matters.

We verified R is a faithful condition before relying on it: their rag result
file carries one `context` string per query with `trajectory: null` and no
multi-round structure, and their recorded `avg_context_tokens` (23,662)
matches our independent `cl100k_base` count (23,689). The stored context is
what their answerer received, not one round of several.

---

## Results

### End-to-end (production harness, our system)

| | |
|---|---|
| binary accuracy | **71.5%** (286/400) |
| rubric score | **0.611** |
| ingest | **241 s** for 170 documents, **no LLM, no network** |
| retrieval | **80 ms** mean, 44 ms p50, 258 ms p95 |
| context | 13,673 tokens/query |

> **Engine version, corrected 2026-08-13.** This page briefly claimed 0.14.0.
> It is wrong: the benchmark's virtualenv holds `yantrikdb-0.13.4.dist-info`,
> installed 2026-08-11 21:05, before the run. The number was never verified —
> it was changed from `0.14.0-dev` to `0.14.0` on the reasoning that 0.14.0 was
> "the release the code corresponds to", which is an assumption, not a
> measurement. Nothing else on this page is affected: every result was produced
> by that same 0.13.4 build, so the comparisons remain internally consistent.
> The pending newer-engine run happened on 0.15.0 — see "The engine re-run"
> section below; these 0.13.4 numbers stand as the anchor it is measured
> against.
>
> The run files record no engine version at all, which is why this was
> undetectable from the outputs. That is now a gap worth closing in the
> provider.

Configuration: turn-aware chunking, `k=40`, bundled 7 MB static embedder
(potion-base-2M, 64-dim, in-process), HNSW + BM25 fusion. Answerer and judge
`deepseek-v4-flash:0731`.

### Controlled comparison (frozen context, same reader)

| condition | mode | binary | rubric | context tokens |
|---|---|---|---|---|
| **A — yantrikdb** | rag | **289/400 = 72.2%** | **0.607** | **13,673** |
| **R — hindsight** | rag | 286/400 = 71.5% | 0.592 | 23,689 |
| E — hindsight | single-query | 260/400 = 65.0% | 0.563 | 17,655 |

### Significance

Queries are nested inside 20 conversations, so they are not independent. All
tests are computed at the **conversation** level.

| comparison | binary | rubric | bootstrap CI (rubric) | verdict |
|---|---|---|---|---|
| **A vs R** | +3, p = 0.845 | +0.015, p = 0.438 | **[−0.021, +0.052]** | **equivalent** |
| A vs E | +29, **p = 0.016** | +0.044, **p = 0.025** | [+0.009, +0.077] | A better |
| R vs E | +26, **p = 0.006** | +0.029, p = 0.089 | [−0.001, +0.061] | R better (binary) |

Sign-flip tests are **exact** — all 2²⁰ = 1,048,576 conversation sign
assignments enumerated, no sampling or asymptotics. Bootstrap is 4,000
conversation-clustered resamples.

For **A vs R**, every robustness check agrees with equivalence:

- per-conversation direction: **7 favour A · 5 tie · 8 favour R** — a coin flip
- leave-one-conversation-out: binary delta ranges −2 to +7; **the direction is
  not stable** across removals
- the bootstrap CI comfortably contains zero

A naive McNemar over the 400 query pairs would report A-vs-E at p = 0.00077,
but that assumes independence the design does not have. The conversation-level
numbers are the ones to quote.

### Per-category (point estimates only)

Each category holds 40 queries drawn from the same 20 conversations, so these
are neither independent nor corrected for multiple comparisons. They describe
this run; they are not per-category findings.

| category | A (ours) | R (theirs, rag) | Δ | E (theirs, single-query) | Δ |
|---|---|---|---|---|---|
| knowledge_update | 0.619 | 0.494 | **+0.125** | 0.594 | +0.025 |
| information_extraction | 0.790 | 0.693 | **+0.096** | 0.658 | +0.132 |
| multi_session_reasoning | 0.542 | 0.470 | **+0.072** | 0.457 | +0.085 |
| temporal_reasoning | 0.425 | 0.381 | +0.044 | 0.319 | +0.106 |
| contradiction_resolution | 0.656 | 0.628 | +0.028 | 0.628 | +0.028 |
| abstention | 0.625 | 0.625 | 0.000 | 0.650 | −0.025 |
| instruction_following | 0.719 | 0.762 | −0.044 | 0.731 | −0.012 |
| summarization | 0.583 | 0.637 | **−0.054** | 0.484 | +0.099 |
| preference_following | 0.808 | 0.863 | −0.054 | 0.800 | +0.008 |
| event_ordering | 0.298 | 0.362 | **−0.064** | 0.306 | −0.008 |

Our two weakest categories in absolute terms are `event_ordering` (0.298) and
`temporal_reasoning` (0.425). Both involve reconstructing *sequence*, and both
are where their rag configuration matches or beats us. That is the clearest
improvement target this benchmark surfaces.

### `contradiction_resolution`: the paper's intent and the judge's behaviour differ

Worth pinning down, because acting on the paper's wording alone costs points
here.

The BEAM paper (arXiv 2510.27246, Appendix D item 2) defines four nuggets:

1. states there is contradictory information;
2. mentions claim **A**;
3. mentions claim **B**;
4. **which statement is correct?**

Its error analysis (Appendix G) names the dominant failure mode: *"the context
contains only one side of the contradiction, leading the LLM to answer based
solely on that information."*

**Measured on our own run** (`ydb-final`, category n = 40):

| | |
|---|---|
| binary accuracy | 1.000 |
| rubric score | 0.688 |
| answers that literally ask "which one is correct?" | 28 / 40 |
| score of nugget 4 across the category | **0.000 on all 40** |

Appending one sentence naming which statement is current moves that nugget to
**0.92 (6/6, offline scoring)**, and a live run reproduced **0.0 -> 1.0** on
both queries tested. Category score moved 0.688 -> 0.850 (`ydb-confirm-ctrl`,
n = 40), worth roughly **+1.6 pp overall**.

So nugget 4 reads as a request for clarification, but the judge scores it as
*"did the response identify which statement is correct"*. Asking scores zero.
Naming scores one. An earlier draft of this section asserted the reverse — that
asking "is what the rubric rewards" — reasoning from the paper's intent without
checking the judge; the table above contradicts it.

**Two operations that must not be conflated:**

* **Destructive resolution** — `auto_resolve_conflicts` (think loop) tombstones
  one side, destroying nuggets 1, 2 and 3 at once: no conflict left to state,
  one claim gone. **Do not run destructive conflict resolution before this
  benchmark.** That warning stands and is unaffected by the above.
* **Naming the current statement in the answer** — quote BOTH claims, say they
  conflict, then say which is current. Nuggets 1-3 untouched, nugget 4
  satisfied. This is the change that moved the category.

**Honest caveat on the gain.** The paper's intent is that a memory system
surface a contradiction and defer to the user. Our change optimises the judge's
implementation rather than that intent, so part of it is a rubric artefact
rather than better memory behaviour. It is defensible on its own terms — the
harness's global answer rule 3 already says prefer the more recent, and
reporting which value is *current* is more useful than handing the conflict
back — but it should be reported as "we satisfy the nugget as scored", not as
"we solved contradiction resolution".

The `yantrikdb-cognitive` arm (conflict detection + counterpart injection, no
resolution) measured flat on this category. The remaining headroom is naming
both claims crisply, which is an answer-shape concern rather than a
memory-resolution one.

---

## Interpretation

### The main finding

Hindsight's published BEAM-100K rag result is **0.862**, read by
`gemini-3.1-pro-preview`. Ours is **0.611**, read by `deepseek-v4-flash`. That
is a 25-point gap.

Under a **common reader**, the same two memory systems' retrieved contexts
score **0.607 and 0.592** — a difference of 0.015 that is not statistically
significant by any test applied.

The published gap is therefore overwhelmingly a property of the **reading
model**, not of the memory systems. This is the confound the method was built
to isolate, and it turns out to account for almost all of a headline
difference that reads, in a leaderboard, as a decisive result.

### What this establishes

- **Parity with their strongest published configuration**, on **42% fewer
  context tokens** (13,673 vs 23,689), with **no LLM at ingest**.
- **Their configuration choice matters more than the gap between us.** Their
  rag mode beats their own single-query mode by +26 binary (p = 0.006) —
  larger and more significant than any difference between their system and
  ours.
- Against their single-query configuration specifically, our context produced
  better answers (p = 0.016). That comparison stands, but it is a comparison
  against their weaker setup and should not be quoted as the headline.

### What it does not establish

- **It is not a claim that our memory is better than theirs.** On the
  mode-matched comparison the honest reading is equivalence, and we cannot
  rule out a small advantage in either direction — the CI spans −0.021 to
  +0.052.
- **It is not a refutation of their published 0.862.** We show only that their
  retrieved context, read by a mid-tier model, does not outperform ours. We
  did not run our context through their answerer, so the 2×2 is incomplete.
  Prompts, decoding and context/reader interaction may all differ.
- **Length is not controlled.** Longer context can hurt through dilution, so
  we cannot rule out that their larger context cost them something a
  budget-matched test would recover. The result favours us on
  quality-per-token; it does not isolate content selection from context length.
- **The contexts are not information-matched, and part of our token advantage
  is a capability we are not exercising.** Measured after publication: their
  rag contexts carry a **median of 117 dates each**; ours carry **5**. Ours
  have any only because BEAM's own `[March-15-2024 | Turn 0]` turn headers
  survive in the minority of chunks that span a turn boundary — the engine
  stores `created_at` for every chunk and the provider discards it. So the
  42% token saving is partly the cost of *not carrying event time*, which is
  not a free efficiency: it is a difference in what each memory layer emits.
  On any question requiring relative sequence, their answerer has the
  timestamps and ours largely does not.

  This does not obviously explain the category results — we lose
  `event_ordering` but win `temporal_reasoning` and `knowledge_update`, all
  date-dependent — which suggests ordering needs relative sequence across
  many units while the others need a single anchor. A dated,
  chronologically-ordered variant exists (`yantrikdb-temporal`) and measured
  no effect at n=80, but that null sits inside a ~0.039 noise floor and is
  underpowered. It is being re-run at full scale; this page will be updated
  with the result either way.
- **Their context was produced for a frontier reader** and may be
  disadvantaged by a mid-tier one in ways not characterised here. This cuts
  against us, and is the most likely way the equivalence finding is wrong.

### A hypothesis the stronger comparison does not support

Hindsight performs LLM-based fact/entity/relationship extraction at ingest;
YantrikDB performs none. Against their **single-query** configuration, our
advantages clustered neatly in categories where losing a qualifier or a
transition proves fatal later — information extraction, temporal reasoning,
summarization, multi-session — which looked like evidence for an
information-bottleneck mechanism.

**Against their rag configuration, that pattern largely dissolves.**
Summarization flips from +0.099 to −0.054. Temporal reasoning falls from
+0.106 to +0.044. They lead on event ordering, preference following and
instruction following. Only knowledge update, information extraction and
multi-session reasoning survive as advantages.

A category pattern observed against one competitor configuration is a property
of that configuration, not evidence for a general mechanism. **We are
withdrawing the information-bottleneck claim as unsupported by this
experiment.**

Three internal results on the same corpus still point that direction, but they
are convergent, not independent (same substrate, overlapping fixtures), and
they are about *our own* pipeline rather than a comparison:

| intervention | effect |
|---|---|
| engine-native `think()` consolidation | −7/80 |
| LLM-written consolidation (0.8B, grounding-gated) | −7/80 |
| retrieving *more* raw context (k=10 → k=40) | +11.1 pp binary at full scale |

The two consolidation arms share 6 of 9 regressions against an independent
expectation of 1.0 — one structural cause, not two coincidences. That is
evidence that *our* consolidation hurts, not that anyone else's does.

---

## Cost profile

Same split (100K), same statistic, from each system's own published result
files:

| | ingest (whole corpus) | retrieval (mean) | context tokens | LLM at ingest |
|---|---|---|---|---|
| **YantrikDB** (rag) | **241 s** | **80 ms** | **13,673** | **none** |
| Hindsight (rag) | 404 s | 2,565 ms | 23,689 | fact extraction |
| Hindsight (single-query) | 404 s | 6,379 ms | 17,655 | fact extraction |

Their rag run records `ingestion_time_ms: 0` / `0 docs` because it reuses the
store built by the single-query run; 404 s is the ingest cost behind both.
Document counts are not comparable (170 vs 6) because they reflect different
chunking granularity, so only total wall-clock is quoted.

Our ingest is chunk → embed → index with a bundled 7 MB static model running
in-process: no model server, no network, no API key. The only LLM anywhere in
our pipeline is the benchmark's own answerer, which is identical for every
system AMB evaluates.

**Retrieval is ~32× faster than their rag configuration for statistically
indistinguishable answer quality.**

---

## Validation of the evaluator

Condition A is our own context replayed through the frozen-context harness. It
should reproduce the production run, and does:

| | rubric |
|---|---|
| production harness | 0.611 |
| frozen-context evaluator | 0.607 |

A 0.004 difference across two full independent 400-query runs of identical
input, which also serves as an estimate of answerer/judge stochasticity — and
is an order of magnitude smaller than the A-vs-E effect, though of the same
order as the A-vs-R difference. That is a further reason to read A-vs-R as
equivalence.

---

## Caveats on token counts

Two distinct quantities, easy to conflate:

- **Benchmark-normalized size** — AMB's `count_tokens` (tiktoken
  `cl100k_base`). Canonical for comparison; this is what the tables report,
  and it matches the harness's own `avg_context_tokens` field.
- **Actual answerer input tokens** — requires the answerer's own tokenizer and
  chat template. **Not measured here.** Any billing or context-pressure claim
  needs this instead.

An earlier draft of this analysis estimated tokens as `chars // 4`. That proxy
is biased by text style — our conversation prose runs 4.63 chars/token while
their marked-up extracted statements run 3.82 — so it inflated our count
relative to theirs and made a **23% size difference read as 6%**. Result files
written before the fix still carry the proxy in their `context_tokens` field;
[`frozen_stats.py`](#reproducing) therefore recomputes token counts from the
context text rather than trusting the stored field.

---

## Reproducing

```bash
git clone https://github.com/vectorize-io/agent-memory-benchmark
cd agent-memory-benchmark
uv sync

python /path/to/yantrikdb/benchmarks/amb/install.py .

# Treat limits as part of the measured configuration. Provider selection
# traces record requested_k, provider_default_k, effective_recall_k,
# recall_candidates, and returned; token-budgeted arms also record whether
# that budget bound. Preserve those fields in frozen artifacts so sync/async
# default drift is detectable before judging.

# 1. End-to-end run (writes its own contexts)
YDB_BENCH_TURN_AWARE=1 YDB_BENCH_TOPK=40 \
  uv run amb run --dataset beam --split 100k --memory yantrikdb -n ydb-final

# 1b. Query-time global synthesis ceiling probe.
# This intentionally uses two LLM calls per retrieval in the memory layer:
# a full-bank recall, one extraction pass over up to 160 user-authored
# evidence blocks, then one ordering pass.
YDB_BENCH_TURN_AWARE=1 YDB_BENCH_TOPK=40 \
  YDB_BENCH_SYNTH_RECALL_POOL=1000 YDB_BENCH_SYNTH_BLOCKS=160 \
  YDB_BENCH_SYNTH_USER_ONLY=1 \
  uv run amb run --dataset beam --split 100k \
    --memory yantrikdb-global-synthesis -n ydb-global-synthesis

# Optional evidence-ID selector before extraction. This adds one bounded LLM
# call that chooses 20-30 relevant user-turn blocks, then runs extraction and
# ordering only over that concern-focused evidence set.
YDB_BENCH_SYNTH_PREFILTER=1

# 1c. Evidence-preserving synthesis arm. This uses speaker-level ingestion,
# keeps the role-aware raw evidence in the answer context, and appends the
# synthesized candidate timeline as a derived navigation aid. If synthesis
# fails, retrieval returns the raw evidence instead of an empty context.
YDB_BENCH_TURN_AWARE=1 YDB_BENCH_TOPK=40 \
  YDB_BENCH_SYNTH_RECALL_POOL=1000 YDB_BENCH_SYNTH_BLOCKS=160 \
  YDB_BENCH_SYNTH_USER_ONLY=1 YDB_BENCH_SYNTH_PREFILTER=1 \
  uv run amb run --dataset beam --split 100k \
    --memory yantrikdb-role-aware-synthesis \
    -n ydb-role-aware-synthesis

# Synthesized-item date precedence is explicit event date, then the first
# evidence block's historical created_at, then synthesis-record created_at.
# Returned items expose date_source and date_confidence so fallback dates are
# sortable without being presented as equally precise event timestamps.
# The synthesis-only Ollama client disables hidden reasoning, requests a
# 65,536-token context, and caps output at 4,096 tokens. This reduced a
# measured structured cloud call from a 10-minute stall to seconds without
# changing answerer or judge settings.
# Full-bank recall is flattened and deduplicated into atomic user turns before
# applying the 160-turn / 48k-token evidence budget. Extraction sees turns in
# retrieval-relevance order; source date and turn metadata are retained for a
# separate ordering pass. Chronological evidence order caused the extractor to
# harvest the start of the history even when the query's named thread was
# present later in the prompt.
# `YDB_BENCH_SYNTH_ORACLE_TURNS=4,60,...` is a diagnostic-only filter over
# recalled turn ids. Synthesis failures return empty context with an explicit
# status/error instead of silently evaluating raw evidence blocks.

# Query-9 oracle result (2026-08-19): five exact gold turns scored 0.9, while
# a hand-identified mentor/advice concern cluster scored 0.4. Adding a
# cross-session coverage instruction scored 0.2. Extraction, event dates, and
# ordering therefore have a high ceiling once the exact subset is supplied;
# the unresolved failure is selecting the rubric's five milestones from many
# semantically valid milestones inside the same concern.

# Query-9 concern-transfer result (2026-08-21): a fresh DeepSeek answer over
# the frozen ydb-0151 context of 40 retrieved excerpts (18,680 tokens) again
# chose five plausible generic refinement milestones and received three judge
# votes of 0.0. Replaying the same query through the persisted concern-thread
# selector returned five source-backed items (484 tokens); the identical
# answerer named the exact Bryan -> Shawn -> Bryan -> Matthew -> Matthew
# sequence and received three judge votes of 1.0. The companion family-support
# query also scored 1.0 on all three votes from five selected concern items.
# This is a two-query mechanism result, not a full-category score claim. It
# isolates query-dependent item selection and representation as the Q9 gap:
# adding more raw evidence or another ordering pass does not choose the
# canonical five-item concern thread.

# Full event-ordering results (40 queries, 2026-08-19):
#   baseline                              0.2817
#   global synthesis, chronological input 0.2535
#   global synthesis, relevance-first     0.3003  (+0.0186 vs baseline)
#   evidence selector + relevance-first   0.3397  (+0.0580 vs baseline)
# The relevance-first arm completed without synthesis/JSON failures, with
# median 0.20, 7 zeros, 10 queries >= 0.5, 6.8s average retrieval, and 312
# answer-context tokens. It proves query-time global synthesis can lift some
# named/technical threads, but the aggregate gain is too small and variable
# to justify write-time cross-session synthesis yet. The remaining bottleneck
# is canonical subset selection among several valid milestones, especially for
# diffuse narrative concerns.

# Evidence-selector arm (40 queries, 2026-08-19): median 0.345, 7 zeros,
# 12 queries >= 0.5, 6.0s average retrieval, and 317 answer-context tokens.
# Against relevance-first it improved 16 queries, tied 11, and regressed 13.
# This validates concern-focused evidence selection as the next lever, but the
# arm remains opt-in: model variance and broad narrative concern boundaries
# still make write-time permanent synthesis premature.

# Evidence-preserving hybrid frozen artifact (unjudged, 2026-08-20): appending
# those query-focused candidate timelines to the 40 role-aware user contexts
# retained all 236/242 date/number gold values and moved word coverage from
# 0.818 to 0.827. Synthesis-only retained 222/242. This passes the deterministic
# no-loss gate, but it is not a score claim until the paired frozen-context arm
# is judged with the same answerer and judge.

# Role-aware chronological artifact (unjudged, 2026-08-20): the V3 arm is a
# pure presentation permutation of V2. Across 40 event-ordering queries it
# retains the same 5,070 documents and relevance-selection traces, reorders all
# 40 rows, parses every date/turn prefix, and leaves word coverage at 0.818 and
# date/number coverage at 236/242. The frozen artifact SHA-256 is
# bfe5c3d522da4b934ade67d61d3ec8f96af670ca95dc6b62e3c43f796b5e534f.
# Its separate V2-vs-V3 manifest preflight reports 1,050,939 synthetic context
# tokens, 80 answer calls, and 80 judge calls. This is not a score claim until
# that paired arm is run.

# Pre-registered paired evaluation order and interpretation:
#   1. V1 mixed-speaker vs V2 user-only asks whether provenance routing helps.
#   2. V2 user-only vs V2+candidate hybrid asks whether item assembly adds value.
#   3. Separately, V2 user-only vs chronological V3 asks whether ordering the
#      unchanged selected evidence helps the reader reconstruct sequence.
# Do not select the best of three after seeing scores. Each comparison stands
# alone. A benchmark lift requires the paired bootstrap 95% interval to exclude
# zero. A mean gain >= 0.03 with more wins than losses is only a reason for a
# broader follow-up, not a score claim. V2 may still ship for attribution
# correctness when the score comparison is non-inferior or inconclusive.
# Preflight validates both artifact hashes, ordered query IDs, row/token counts,
# model, and projected call budget. It exits before importing the LLM client.
uv run python paired_frozen_context_eval.py \
  --contexts-a /path/to/role-aware-event40-mixed-v1.json \
  --contexts-b /path/to/role-aware-event40-user-only.json \
  --manifest /path/to/external-eval-manifest-v1-v2.json \
  --label-a role-aware-v1 --label-b role-aware-v2 \
  --workers 2 --judge-repeats 1 \
  --preflight-only

# Remove --preflight-only only after reviewing its exact payload/call report.
# Checkpoints are bound to the manifest, artifacts, and run configuration;
# --resume rejects mismatched or legacy partial files.
uv run python paired_frozen_context_eval.py \
  --contexts-a /path/to/role-aware-event40-mixed-v1.json \
  --contexts-b /path/to/role-aware-event40-user-only.json \
  --manifest /path/to/external-eval-manifest-v1-v2.json \
  --label-a role-aware-v1 --label-b role-aware-v2 \
  --workers 2 --judge-repeats 1 \
  --out outputs/paired-role-aware-v1-v2.json

### Paired provenance result (2026-08-20)

The authorized V1 mixed-speaker versus V2 user-only run completed all 40
event-ordering pairs with `deepseek-v4-flash:0731-cloud` as both answerer and
judge. The pre-registered prediction that V2 would be at least as accurate was
not supported:

| context arm | mean rubric |
|---|---:|
| V1 mixed-speaker | 0.29347 |
| V2 user-only | 0.28684 |

Paired delta (V2 - V1): **-0.00664**, bootstrap 95% CI
**[-0.05948, +0.04763]**; V2 wins / ties / V1 wins = **14 / 11 / 15**.
This is a null accuracy result, not a lift and not evidence that assistant
turns are generally harmful. V2 does preserve essentially the same score with
41% less context (14,529 to 8,580 mean words), so its supported benefits are
speaker-attribution correctness and context efficiency.

The paired instrument is itself a useful result: its roughly +/-0.05 interval
is about three times sharper than the earlier unpaired synthesis comparison.
The V2-versus-hybrid item-assembly arm remains unrun and retains its own
pre-registered decision rule; this result does not authorize or predict it.
The immutable result is
`benchmarks/amb/artifacts/paired-v1-v2-result.json` under manifest SHA-256
`396164f5880e6ac6e90153534c7a52a16f40be51a0bd5aa86ab060e58b672e7d`.

# 2. Frozen-context conditions — same answerer + judge for all three
uv run python frozen_context_eval.py \
  --contexts outputs/beam/ydb-final/rag/100k.json            --label A-yantrikdb-ctx
uv run python frozen_context_eval.py \
  --contexts outputs/beam/hindsight/rag/100k.json.gz         --label R-hindsight-rag-ctx
uv run python frozen_context_eval.py \
  --contexts outputs/beam/hindsight/single-query/100k.json.gz --label E-hindsight-ctx

# 3. Conversation-level statistics for any pair
uv run python frozen_stats.py A-yantrikdb-ctx R-hindsight-rag-ctx
```

Hindsight's per-query contexts for both configurations ship inside the AMB
repository, so conditions R and E need no access to their system.

---

## The rag-mode 2×2 under an equalized frontier-class reader (2026-08-15)

Open item 2 asked for the missing cells of the 2×2. Their rag-mode contexts —
the retrieval behind the published 0.862 — turned out to ship in the result
files (`hindsight/rag/100k.json.gz`, all 400 rows), so both memory systems'
cached contexts were replayed through ONE frontier-class reader
(`moonshotai/kimi-k2.6`) under the judge Hindsight published with
(`meta-llama/llama-4-maverick`), scored with `dataset.score_result`.

| contexts | rubric (393 paired queries) |
|---|---|
| **YantrikDB** (zero-LLM ingest, ~28% fewer tokens) | **0.6331** |
| **Hindsight rag-mode** (their 0.862 configuration) | **0.5982** |

Delta **+3.5pp**, conversation-clustered bootstrap 95% CI **[+0.7, +6.2]pp**,
excludes zero. Query-level: 98 wins / 71 losses / 224 ties.

Two findings, in order of importance:

1. **The reader collapse.** Their own recorded 0.8577 on these *identical*
   context rows drops to 0.5982 when the reader stops being
   gemini-3.1-pro-preview (their `temporal_reasoning` alone: 0.900 → 0.331).
   The published headline is overwhelmingly the answerer.
2. **The lead concentrates where evidence fidelity matters.** Largest gap:
   `information_extraction` +17.7pp — their ingest-time LLM extraction
   compresses away exactly the specifics that category grades. Also +5.9
   preference, +5.8 temporal, +5.0 knowledge_update, +4.4 contradiction,
   +4.1 instruction. Their remaining leads are small: summarization −4.0,
   event_ordering −2.7, abstention −2.5.

Honesty notes. kimi-k2.6 is a frontier-*class* open model, not their exact
reader, so the claim is "under an equalized frontier-class reader", never "in
their published setup". Both arms initially lost all 40 summarization queries
to the same evaluator defect — the reader's 1024-token cap truncated JSON
mid-summary, and the errors silently shrank the denominator; both arms were
re-run for that category at 4096 (`NANOGPT_MAX_TOKENS`) before the numbers
above. ~7 queries per arm remain excluded on residual request errors.

## The engine re-run: 0.13.4 → 0.15.0, everything else frozen (2026-08-16)

Engine 0.15.0 deliberately changed retrieval — per-lane filter integrity,
lane slot quotas, a shared prior-boost budget, and four tuning knobs that
previous releases parsed and silently never read. Every number above predates
those changes, so this run answers one question: did the correctness campaign
cost retrieval quality? Config, answerer, judge, and dataset all held
identical to `ydb-final` (turn-aware chunking, k=40, rag mode,
`deepseek-v4-flash` for both roles); the only variable is the engine — and,
one honesty note, the published 0.15.0 wheel replaced what turns out to have
been a locally-built 0.13.4 in the harness venv.

| | binary | rubric |
|---|---|---|
| **ydb-0150 — engine 0.15.0** | **290/400 = 72.5%** | **0.6375** |
| ydb-final — engine 0.13.4 (anchor) | 286/400 = 71.5% | 0.6107 |

Rubric delta **+0.027**, conversation-clustered bootstrap 95% CI
**[+0.007, +0.045]** — excludes zero. Direction is unusually consistent for
this benchmark: **17 of 20 conversations favour 0.15.0, 0 tie, 3 favour the
anchor** (compare A-vs-R above, which was a 7/5/8 coin flip). The delta is
~7× the 0.004 answerer-stochasticity estimate from the validation section.

Where it moved (point estimates, same non-independence caveats as above):
`contradiction_resolution` +0.141 — the naming-the-current-statement change
plus the engine no longer letting reserve lanes smuggle filtered records;
`temporal_reasoning` +0.062; `abstention` +0.050; `information_extraction`
+0.034. The largest loss is `multi_session_reasoning` at −0.026, inside
noise. `event_ordering` stays this system's worst category (0.298) — the
sequence-reconstruction signal from the original analysis is unchanged and
still the clearest improvement target.

What this does NOT claim: it is one run per side, the interval is against a
single anchor draw, and 0.13.4→0.15.0 is a cumulative jump (0.14.x rode
along) — it attributes the gain to the release span, not to any single
mechanism inside it. The regression question it was run to answer is
answered: **the silent-defect campaign did not cost retrieval quality; it
measurably improved it.**

## Open work

1. **Token-budget curve.** Ask each engine for a fixed budget
   (2K/4K/8K/12K/16K) and plot rubric against context tokens. Removes the
   length objection entirely and yields quality-per-token rather than a single
   point. This is now the most valuable remaining experiment, because parity
   on 42% fewer tokens is a claim about the *curve*, and we have measured one
   point on it.
2. ~~Matched-answerer end-to-end~~ — **done 2026-08-15** (section above):
   under an equalized frontier-class reader the equivalence strengthens to a
   significant lead. The exact-gemini variant remains open only as a
   robustness check.
3. **Sequence reconstruction.** `event_ordering` (0.298) and
   `temporal_reasoning` (0.425) are our weakest categories and the ones where
   their rag configuration leads. This benchmark's clearest actionable signal.
4. **A third arm**: raw → answer vs LLM-compressed → answer on the same
   substrate. Tests the information-bottleneck idea directly, which — per
   above — this experiment does *not* establish.
