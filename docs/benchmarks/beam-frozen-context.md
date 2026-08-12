# BEAM: isolating memory quality from answerer quality

**Date:** 2026-08-12 · **Engine:** yantrikdb 0.14.0-dev · **Harness:** [vectorize-io/agent-memory-benchmark](https://github.com/vectorize-io/agent-memory-benchmark) · **Dataset:** BEAM-100K, all 400 queries

---

## The problem with published memory benchmarks

An end-to-end agent-memory score confounds at least three things:

1. the memory system's retrieval quality
2. whatever processing happens at ingest
3. the model that reads the retrieved context and writes the answer

AMB's own README notes that generation setup matters and that small changes in
prompts or models move accuracy substantially. So when two systems report
different end-to-end numbers under different answerers, the difference is not
attributable to either system's memory.

This page reports two things: our end-to-end number under a stated
configuration, and a **controlled comparison that removes the answerer as a
variable**.

---

## Method: frozen-context evaluation

Hindsight publishes per-query result files containing the **exact context
string** they injected for each query. Every one of the 400 BEAM-100K
`query_id`s matches ours.

That makes the following possible:

```
A   yantrikdb context  ─┐
                        ├─→  same answerer  →  same judge  →  score
E   hindsight context  ─┘
```

Answerer and judge are held fixed (`deepseek-v4-flash:0731`, temperature 0)
across both conditions. The only thing that varies is which memory system
produced the context. Whatever separates A from E is the memory system.

The evaluator is [`frozen_context_eval.py`](#reproducing). It reuses the
dataset's own rubric judge and per-category prompts, so its numbers are
comparable to a normal harness run — verified below.

---

## Results

### End-to-end (production harness, our system)

| | |
|---|---|
| binary accuracy | **71.5%** (286/400) |
| rubric score | **0.611** |
| ingest | **241s** for 170 documents, **no LLM, no network** |
| retrieval | **44 ms** p50, 258 ms p95 |
| context | 13,673 tokens/query |

Configuration: turn-aware chunking, `k=40`, bundled 7 MB static embedder
(potion-base-2M, 64-dim, in-process), HNSW + BM25 fusion. Answerer and judge
`deepseek-v4-flash:0731`.

### Controlled comparison (frozen context, same reader)

| condition | binary | rubric | context (benchmark tokenizer) |
|---|---|---|---|
| **A — yantrikdb context** | **289/400 = 72.2%** | **0.607** | **12,897** |
| E — hindsight context | 260/400 = 65.0% | 0.563 | 16,533 |

**Δ = +0.044 rubric in favour of YantrikDB, on ~22% fewer context tokens.**

### Significance

Queries are nested inside 20 conversations, so they are not independent. Tests
are therefore computed at the **conversation** level.

| test | result |
|---|---|
| exact cluster sign-flip, binary (2²⁰ permutations, all enumerated) | **p = 0.016** |
| exact cluster sign-flip, rubric | **p = 0.025** |
| conversation-clustered bootstrap, rubric (4000 resamples) | Δ = +0.044, **95% CI [+0.010, +0.077]** |
| leave-one-conversation-out | advantage survives **all 20**; worst case +22 binary, +0.038 rubric |
| per-conversation direction | 13 favour A · 3 tie · 4 favour E |

A naive McNemar over the 400 query pairs gives p = 0.00077, but that assumes
independence the design does not have. **0.016 is the number to quote.**

### Per-category

| category | ours | hindsight | Δ |
|---|---|---|---|
| information_extraction | 0.790 | 0.658 | **+0.132** |
| temporal_reasoning | 0.425 | 0.319 | **+0.106** |
| summarization | 0.583 | 0.484 | **+0.099** |
| multi_session_reasoning | 0.542 | 0.457 | **+0.085** |
| contradiction_resolution | 0.656 | 0.628 | +0.028 |
| knowledge_update | 0.619 | 0.594 | +0.025 |
| preference_following | 0.808 | 0.800 | +0.008 |
| event_ordering | 0.298 | 0.306 | −0.008 |
| instruction_following | 0.719 | 0.731 | −0.013 |
| abstention | 0.625 | 0.650 | −0.025 |

Hindsight's observed advantages are small; our largest advantages cluster in
categories where **preservation of source evidence** matters. Category-level
significance was **not** tested separately, so these are point estimates, not
per-category findings.

---

## Interpretation

### What this establishes

Under a controlled reader, YantrikDB's retrieved context produced better
downstream answers than Hindsight's on the same 400 queries, while supplying
fewer context tokens. The effect is statistically significant at the
conversation level and robust to removing any single conversation.

### What it does not establish

- **It is not a claim about Hindsight's published configuration.** Their
  BEAM-100K result of 0.734 rubric uses `gemini-3.1-pro-preview` as answerer.
  Our experiment shows only that *their retrieved context, read by a
  mid-tier model, predicts a lower score than their end-to-end number*. The
  gap between 0.734 and 0.563 cannot be attributed to the answerer alone —
  prompts, decoding, formatting and context/reader interaction may all differ.
- **It uses their single-query configuration**, the only one whose per-query
  contexts are published. Their `rag`-mode result (0.862 rubric) is untested
  here.
- **Length is not controlled.** Longer context can hurt through dilution, so
  one cannot rule out that Hindsight scored lower partly *because* it supplied
  more tokens. The result favours YantrikDB on the **combined
  quality-per-token objective**; a token-budget-matched ablation is needed to
  separate content selection from context length.
- **Their context was produced for a frontier reader.** Reading it through a
  mid-tier model may disadvantage it in ways not characterised here.

### A hypothesis, not a conclusion

Hindsight performs LLM-based fact/entity/relationship extraction at ingest;
YantrikDB performs none. Our largest advantages fall in exactly the categories
where losing a qualifier, a transition, or an apparently unimportant detail is
fatal later.

That is **consistent with** — not proof of — the idea that ingest-time
semantic normalization must decide what a memory *means* and what will matter
before the question exists, and that this decision is underdetermined.

Three further observations from the same corpus point the same way, though
they are **convergent, not independent** (same substrate, overlapping
fixtures):

| intervention | effect |
|---|---|
| engine-native `think()` consolidation | −7/80 |
| LLM-written consolidation (0.8B, grounding-gated) | −7/80 |
| retrieving *more* raw context (k=10 → k=40) | +11.1pp binary at full scale |

The two consolidation arms share 6 of 9 regressions against an independent
expectation of 1.0 — one structural cause, not two coincidences.

---

## Validation of the evaluator

Condition A is our own context replayed through the frozen-context harness.
It should reproduce the production run, and does:

| | rubric |
|---|---|
| production harness | 0.611 |
| frozen-context evaluator | 0.607 |

A 0.004 difference across two full independent 400-query runs of identical
input — which also serves as an estimate of answerer/judge stochasticity.

---

## Cost profile

| | ingest | retrieval p50 | LLM at ingest |
|---|---|---|---|
| **YantrikDB** | 241s / 170 docs | **44 ms** | **none** |
| Hindsight (500K split) | 816s / 364 docs | 1,956 ms | fact extraction |
| Hindsight (1M split) | 2,715s / 1,228 docs | 2,968 ms | fact extraction |

Ingest figures are from each system's own published result files and are not
size-matched across splits; treat them as order-of-magnitude.

---

## Caveats on token counts

Two distinct quantities, easy to conflate:

- **Benchmark-normalized size** — AMB's `count_tokens` (tiktoken
  `cl100k_base`). Canonical for comparison. This is what the tables report.
- **Actual answerer input tokens** — requires the answerer's own tokenizer and
  chat template. **Not measured here.** Any cost or context-pressure claim
  needs this instead.

An earlier draft of this analysis estimated tokens as `chars // 4`, which is
biased by text style — conversation prose runs ~4.9 chars/token while
Hindsight's marked-up extracted statements run ~4.1, making a 28% size
difference read as 7%.

---

## Reproducing

```bash
git clone https://github.com/vectorize-io/agent-memory-benchmark
cd agent-memory-benchmark
uv sync

# 1. End-to-end run (writes its own contexts)
YDB_BENCH_TURN_AWARE=1 YDB_BENCH_TOPK=40 \
  uv run amb run --dataset beam --split 100k --memory yantrikdb -n ydb-final

# 2. Frozen-context conditions — same answerer + judge for both
uv run python frozen_context_eval.py \
  --contexts outputs/beam/ydb-final/rag/100k.json --label A-yantrikdb-ctx
uv run python frozen_context_eval.py \
  --contexts outputs/beam/hindsight/single-query/100k.json.gz --label E-hindsight-ctx
```

Hindsight's per-query contexts ship inside the AMB repository, so condition E
needs no access to their system.

---

## Open work

1. **Token-budget curve.** Ask each engine for a fixed budget (2K/4K/8K/12K/16K)
   and plot rubric against context tokens. Removes the length objection
   entirely and yields quality-per-token rather than a single point.
2. **Their `rag`-mode contexts**, to address the 0.862 configuration.
3. **Matched-answerer end-to-end**, i.e. our context through
   `gemini-3.1-pro-preview`, completing the 2×2.
4. **A third arm**: raw → answer, vs LLM-compressed → answer, on the same
   substrate. Tests the information-bottleneck hypothesis directly instead of
   inferring it from a competitor comparison.
