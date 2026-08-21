# AMB Event-Ordering Membership Funnel

## Scope

This analysis measures where event-ordering evidence disappears in the
adaptive-rollup pipeline. It does not treat semantic similarity as ground
truth: BEAM scores independent rubric nuggets, and adjacent real milestones
can be semantically close while belonging to a different generator partition.

The comparable cohort contains 18 queries (conversation groups 1-9), 96 rubric
items, and 95 unique query-level source turns.

Evaluator provenance:

- AMB commit: `d5c81960aaebf695f2b8bada9ce1486f8b684a51`
- `src/memory_bench/dataset/beam.py` SHA-256:
  `15a6ff268ced7909d8644dfa37635b164a518594902b43eb00378230b92c27cd`
- Raw BEAM cache SHA-256:
  `939c907ce64bc8ade1d5fce926f5c500004a25703a36ca5931dad4d4f1d3cbf3`
- Candidate artifact SHA-256:
  `4d9f9c00699a71609b1978d1d042fd01353511b22847226d7e9b860d0aab0d56`
- Scored synthesis run SHA-256:
  `a4bbf6922fd800371b9aeaef29f7a7d08f96e3b4304ec53e51eb99d9e3d0d953`
- Raw retrieval comparator SHA-256:
  `048b38bd9a285ab788677d60815ec6533268a3fddf0a4d31e60fbf53a5b8b601`
- Expanded gold audit SHA-256:
  `7f53d98bcf4348cc105c283ec0d807c1a43fcc2f5539a84ce9eac4b3bff4ba63`
- Embedding model: `nomic-embed-text:latest`, digest
  `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`

## Source Schema

`source_chat_ids` is authoritative query-level provenance, but it is not a
one-to-one rubric mapping. The field can contain scalar IDs, nested ID groups,
or a deduplicated set shared by several rubric items. Likewise,
`conversation_references` can describe one turn, several turns, or a logical
session label whose number is not a turn ID.

For example, `2_event_ordering_1` has five rubrics but only source turns 28 and
162; several rubrics are facts contained in the same source turn. Conversely,
some references combine multiple source turns. The evaluator therefore:

1. Flattens and deduplicates `source_chat_ids` only at query scope.
2. Measures source-turn identity independently of rubric semantics.
3. Uses the expanded gold audit only for approximate per-rubric semantic
   coverage, never as authoritative source provenance.

## Results

| Stage | Semantic rubric recall | Source-turn identity recall |
| --- | ---: | ---: |
| Raw YantrikDB retrieval comparator | 97.9% | 17/95 (17.9%) |
| Global-synthesis candidate pool | 93.8% | 38/95 (40.0%) |
| Exact-N selected items | 71.9% | 22/95 (23.2%) |
| Final answer | 60.4% | unavailable |

The synthesis run's actual judged mean on these 18 queries is `0.32037`. The
semantic final-answer proxy is `0.55556` with Pearson `r=0.7255`, confirming
that semantic coverage is directionally useful but materially optimistic.

Source-identity losses:

- 57/95 source turns are absent from the candidate pool.
- 16 of the 38 available source turns are removed by exact-N selection.
- Only two source turns present in the separate raw comparator disappear before
  the candidate pool. This is not a true sequential transition because the
  comparator came from a different frozen run.

For this historical run, candidate and selected identity is measured from
`first_mention_turn`, because the artifact does not retain a block-to-turn map
for every `evidence_id`. A multi-evidence candidate may therefore cite a later
source turn that this fallback cannot credit. New synthesis debug artifacts
emit `evidence_block_turns`; the evaluator uses the full cited-turn union when
that map is present and records its provenance mode in the report.

Candidate source survival falls sharply with top-level conversation depth:

| Session | Source turns retained |
| --- | ---: |
| 1 | 23/39 (59.0%) |
| 2 | 7/24 (29.2%) |
| 3 | 5/15 (33.3%) |
| 4 | 2/10 (20.0%) |
| 5 | 1/7 (14.3%) |

All 95 source records are user turns. Their question types are 84
`main_question`, six `answer_ai_question`, and five `followup_question`; no
single metadata marker identifies the partition.

## Head-Bias Bug

The historical `adaptive-rollup-v4-full40` artifact predates the current
per-span output cap. DeepSeek over-emitted roughly 9-10 candidates in each of
the Q1-Q4 temporal arrays. The arrays were flattened in Q1, Q2, Q3, Q4 order
and then globally capped at 24 items, so early arrays silently consumed the
budget and late-session candidates were removed.

Current code slices every span to `per_span_target` before flattening. A pure
regression test now pins that ordering and rejects malformed span values. This
fixes the measured array truncation, but a current-provider Q9 replay proved it
does not fix partition recall by itself. The replay returned exactly four
candidates from each Q1-Q4 span and reached turn 246, yet retained zero of Q9's
six authoritative source turns.

## Q9 Retrieval Probe

All six Q9 source turns exist in the frozen bank, but the bundled 64-dimensional
retriever places them at raw ranks 274-388. After user-turn extraction their
evidence ranks are 54, 73, 84, 94, 123, and 124, so none enters the 40-block
synthesis input. Synthesis cannot recover evidence it never receives.

A local, non-oracle `nomic-embed-text` rerank over the same 135 user turns moves
five source turns into the top 40 at ranks 7, 22, 26, 29, and 35. Turn 108, a
recommendation-letter concern with little lexical or semantic overlap with the
broad personal-statement query, remains at rank 106. This makes contextual
reranking the next cheap intervention; expanding the cloud context to all 135
turns is unnecessary unless the reranked 40-block probe fails.

A local-generator replay over those reranked blocks also exposed an evaluator
hazard: model-supplied source-turn metadata cannot establish that candidate text
is supported. Candidate chronology is now deterministically grounded to the
earliest valid cited evidence block, invalid citations are rejected, and every
correction or rejection is emitted in synthesis telemetry. This still does not
prove that item text is entailed by its citation. The next extraction probe must
require exact source support (for example, a validated evidence quote) before a
candidate can enter selection.

## Decision Gate

Run one frozen-bank Q9 replay using the same DeepSeek generator but select its
40 evidence blocks with the contextual reranker. Before judging, compare
candidate provenance against Q9's exact source turns. If source survival
improves, run a two-query paired judged replay. Only a positive paired result
justifies implementing the reranker in the provider or starting a new full-40
cycle.

The reproducible funnel command writes the untracked detailed artifact
`benchmarks/amb/artifacts/membership-funnel-v2-source-ids.json`:

```powershell
.venv\Scripts\python.exe benchmarks\amb\analyze_membership_funnel.py `
  --candidates benchmarks\amb\artifacts\adaptive-rollup-v4-full40.jsonl `
  --gold benchmarks\amb\artifacts\event40-gold-alignment-nomic.json `
  --synthesis-run C:\Users\sync\codes\agent-memory-benchmark\outputs\beam\synth-adaptive-rollup-v4-full40\rag\100k.json `
  --baseline-run C:\Users\sync\codes\agent-memory-benchmark\outputs\beam\ydb-final\rag\100k.json `
  --beam-source C:\Users\sync\codes\agent-memory-benchmark\.datasets\beam\100k.json `
  --model nomic-embed-text `
  --out benchmarks\amb\artifacts\membership-funnel-v2-source-ids.json
```
