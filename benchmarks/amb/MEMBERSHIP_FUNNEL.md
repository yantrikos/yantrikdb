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
first enforce literal citation integrity, then evaluate semantic support before
a candidate can be treated as fully grounded.

## Q9 Local Selection Controls

The quote-gated local controls used `qwen3.5:9b` only; they are mechanism tests,
not official AMB scores and not evidence about DeepSeek's selector quality.
Every generated candidate had to provide a substantive literal quote found in a
cited block. Contextual rank replaced the stale bundled-retriever rank before
synthesis and the original rank remained available as telemetry.

| Q9 arm | Gold turns in top 40 | Gold turns cited by candidates | Gold turns cited after selection |
| --- | ---: | ---: | ---: |
| Broad personal-statement query, generated candidates | 4/6 | 2/6 | 1/6 |
| Broad query, evidence-preserving 40-candidate bank + entity labels | 4/6 | 4/6 | 1/6 |
| Explicit family-support query, same evidence bank | 5/5 | 5/5 | 1/5 |
| Family query with hard facet gate + visible source quotes | 5/5 | 5/5 | 1/5 |
| Broad query, same candidate bank, blinded Codex selector | 4/6 | 4/6 | 4/6 |
| Family query, same candidate bank, blinded Codex selector | 5/5 | 5/5 | 3/5 |

The evidence bank removes extraction omission for every source turn that reaches
the input, yet Qwen still chooses an early generic prefix. The explicit family
control rules out Q9's under-specified professional-advice partition as the only
cause: selection itself is a binding failure for this local model. More temporal
coverage, entity closure, or candidate expansion is therefore inert until the
selector changes.

A fresh `gpt-5.6-sol` selection-only control was restricted to the query and
`telemetry.candidate_items`; its prompt explicitly excluded gold metadata,
prior selections, results, and the source database. It recovered every broad
Q9 gold turn available in the bank (4/4 available, 4/6 total) and three of five
family-support turns. This materially lifts both Qwen controls without changing
retrieval or extraction, confirming selector capacity as a causal bottleneck.
The remaining family miss shows that a stronger selector alone is not a complete
solution.

The last two columns are source-membership diagnostics derived from exact BEAM
headers in cited blocks. They do not claim that generated candidate or rollup
text semantically represents every cited source turn.

Literal quote membership also remains weaker than entailment. In the family
control, one candidate described a care package while quoting the earlier
handwritten resilience letter from the same valid block. The validator correctly
proves citation membership, but a semantic support judge or extractive item
representation is still required before generated text can be treated as fully
grounded.

## Decision Gate

The selection-only gate passed, but the paid DeepSeek controls split the result
by query specificity:

| DeepSeek arm | Broad Q9 source turns | Explicit family source turns | Official score |
| --- | ---: | ---: | ---: |
| Chronological 40-item evidence bank | 0/4 available | 1/5 | diagnostic only |
| Relevance-first 40-item evidence bank | 1/4 available | 4/5 | diagnostic only |
| Flat concern context | not measured | not measured | broad `0.2` |
| One coarse mentorship handle | 1/5 rubric events | n/a | broad `0.2` |
| Two-stage person handles | 0/5 rubric events | 4/5 rubric events | broad `0.0`, family `0.8` |
| Two-stage role collections | 0/5 rubric events | 4/5 rubric events | broad `0.0`, family `0.8` |

The first two rows remain source-membership diagnostics, not official rubric
scores. Broad top-40 retrieval contains only four of the six historical source
turns, while the official answer has five rubric events; the two counts are not
interchangeable.

Relevance-first presentation is a real, general improvement for an explicit
facet: it raises family membership from 1/5 to 4/5 without changing the bank.
Grouped handle hydration then turns that evidence into an official `0.8` answer.
Here `0.8` is mean rubric-nugget coverage, not exact correctness: BEAM's judge
does not independently score ordering or the exact-N constraint.
The missing family item is also ambiguous: the selector chooses an earlier,
literal statement that partner Tanya supports the user's career instead of the
gold's later pitch-rehearsal event. Both satisfy the wording, but only one is in
the hidden rubric.

The broad Q9 query has a stronger ambiguity ceiling. Query-blind organization
successfully discovers separate drafting, academic-advisor, industry-mentor,
professional-peer, friend-feedback, and family-support arcs. DeepSeek retrieves
the relevant mentor/peer collections, then reasonably selects direct refinement
events such as outlining, reducing word count, adapting versions, and condensing
paragraphs. Those are literal answers to "aspects of refining my personal
statement," but the hidden rubric instead requires Bryan, Shawn, and Matthew.
Neither the query nor the stored evidence identifies that latent partition.

Two follow-up controls show that this is not unique to Q9. For the nine-item
triangle-geometry query, contextual retrieval exposed five of ten historical
source turns and the evidence bank preserved all five, but exact-N selection
chose nine earlier, valid triangle concepts from turns 4-32 instead of the
hidden late-session partition beginning at turn 116. For the narrower city-
autocomplete query, contextual retrieval exposed four of five source turns and
the bank again preserved all four; selection retained two while choosing three
other literal autocomplete implementation concerns. The historical city answer
still scores `0.7`, demonstrating that source-turn identity and rubric coverage
can disagree in either direction.

## Full-Category Organizer Membership Ceiling

A later query-independent organizer audit expands the structural measurement to
all 40 event-ordering queries and all 20 dialogue units. It uses the selected
DeepSeek organizer artifacts that were frozen for the full summarization run.
The query and `source_chat_ids` are withheld from organization; gold turn IDs
are joined only after generation to measure the best source membership available
from one, two, or three stored handles.

| Available collections | Mean source-turn recall | Exact query sets | At least 80% |
| --- | ---: | ---: | ---: |
| One topic handle | 56.7% | 1/40 | 7/40 |
| Two topic handles | 80.8% | 14/40 | 27/40 |
| Three topic handles | 91.8% | 27/40 | 34/40 |
| One topic or virtual anchor union | 68.3% | 11/40 | 16/40 |
| Two topics or virtual anchor unions | 87.9% | 21/40 | 32/40 |
| Three topics or virtual anchor unions | 94.9% | 31/40 | 35/40 |

The report contains 230 query-level source-turn references. Virtual anchor
unions group topic handles by their organizer-declared `anchor_entities`; they
are an oracle hierarchy diagnostic, not a persisted or query-selected product
arm. Report SHA-256:
`41a31216b4a8211b8bbf13db39a1919e3c0b0cfce98ae3404a8a5c208904d077`.

This moves the structural diagnosis. A single topic handle usually cannot
represent the hidden answer partition, while a small union usually contains it.
The remaining problem is selecting and compressing that union without admitting
the much larger set of equally literal adjacent milestones.

### Anchor-Hierarchy Rejection

An opt-in two-level product prototype persisted bounded anchor collections over
the existing Q9 and Q10 topic handles and recursively hydrated them. The local
four-query source-identity preflight regressed from `12/22` with flat public
organization to `9/22` with the anchor hierarchy:

| Query | Flat organizer | Anchor hierarchy |
| --- | ---: | ---: |
| Q9 broad personal statement | 1/6 | 1/6 |
| Q9 family support | 5/5 | 4/5 |
| Q10 broad writing journey | 3/5 | 0/5 |
| Q10 Carla collaboration | 3/6 | 4/6 |

The prototype was removed rather than shipped. Its oracle gain did not survive
actual handle ranking and bounded child selection. As a separate control, Q9's
complete 69-item synthesized child pool contained every gold turn in only 2,919
tokens, yet DeepSeek still chose the earliest generic drafting milestones and
scored `0.2`. More evidence, recursive hydration, and anchor grouping are inert
for the broad query unless storage first emits answer-sized cross-session
concerns that encode the latent event thread.

### Concern-Record Gate

A query-blind source-membership audit over the existing hierarchical concern
artifacts for Q9 and Q10 measured whether a bounded set of answer-sized records
could contain each authoritative source set. Gold was joined only after concern
generation:

| Maximum concern records | Mean source-turn recall | Exact query sets |
| --- | ---: | ---: |
| One | 22.5% | 0/4 |
| Three | 59.2% | 0/4 |
| Five | 87.5% | 3/4 |

The Q9 family query and both Q10 queries reached exact source coverage within
their requested item budget. Broad Q9 reached only `3/6`: turns 108, 164, and
166 were all assigned to the same 12-evidence mentorship topic, but the local
concern generator was capped at six one-event outputs and silently omitted
those three IDs. Report SHA-256:
`11cf08e888aa06e4c8ffc997b5ec849cbcb9b23a61faf6b3fe455e52211ba816`.

The generator probe now supports a single-handle preflight, records raw model
output, states the exact JSON contract for cloud models that ignore Ollama's
schema, and can fail closed when any selected-handle evidence is unassigned. A
bounded DeepSeek rerun over only Q9's mentorship handle produced six coherent
chronological concern records and covered all `12/12` supplied IDs. Artifact
SHA-256:
`b9f01462fac13db58d71f0c44a38fdfb014c7814b11ff90b6596e5860465f3b3`.

That representation did not lift the answer score. With all six records in a
clean public replay, DeepSeek again scored `0.2`. An additional isolated
connected-trajectory control also scored `0.2`. The reason is correction
semantics: the source history later says the user never met Bryan at the film
festival and asks to correct the earlier claim, while Q9's gold still requires
the earlier Bryan-advice event. The historical synthetic thread that scored
`1.0` had filtered the correction out. Optimizing Q9 further would therefore
reward stale-fact omission, which is unacceptable for the product. Judged
artifact SHA-256s are
`5ab46314b6605e13e22240531536cf68062bbe2dee7f8a3163156706b1d2b9d4`
for the complete narrow-thread context and
`da769a0aacbb23e67040f0249ed6e37306693823f106e16c90f7ccebc86fbf4f`
for the isolated generated trajectory.

Concern synthesis remains viable for the three non-conflicting local queries,
but broad Q9 is now a hard stopping case: preserve corrections and move the
score experiment to a query whose rubric agrees with the current memory state.

The next bounded gate used Q10, where no correction conflict was found. Flat
concern retrieval returned 40 records and scored `0.0` on the broad writing
journey and `0.6` on the explicit Carla collaboration query. The existing
grouped concern-topic path reduced the Carla context to exactly five records at
turns 52, 78, 176, 228, and 230. Three independent DeepSeek answer-and-judge
runs scored `0.8`, `0.8`, and `1.0` (median `0.8`). This is a real but narrow
lift: entity/facet routing plus grouped chronological concerns helps when the
query identifies its partition; it does not identify the hidden partition in a
broad timeline query.

Q10 judged artifact SHA-256s:

- Flat broad concerns (`0.0`):
  `ecff4d560db8cd66b93d2a0798bc42fdd0f4d0df01e83da3a24bea9505f04c70`.
- Flat Carla concerns (`0.6`):
  `9c41f85be82f3a59e9fa42f82264c0ca89bdb038955e9fca04b0495ad1c45da1`.
- Grouped Carla concerns (`0.8`, `0.8`, `1.0`):
  `50ca23d36a42ee66fa7c39e53cf88bdc9e90e5a32742d4536b93bd5db324e6ac`,
  `65763420c028f28fc1d86d34ac172264456be43cfb04446b4df0754a278e1b5e`,
  and `435f8d39041806e01c3953e598a71b5403e26b0ce79cc98954702bb09833de15`.

### Query-Route Cohort

The organizer audit now also runs the product's query-only focus and entity
metadata matchers over all 40 frozen event-ordering queries. Source IDs remain
evaluation-only. After excluding the generic acronym `AI` from named-entity
routing, 22/40 queries match at least one handle without semantic ranking:

| Route | Queries | Mean source-turn recall |
| --- | ---: | ---: |
| Multi-token focus | 19 | 41.1% |
| Explicit named entity | 3 | 100.0% |
| Combined | 22 | 49.1% |

Before the exclusion, `AI` was treated like a person's name and expanded 19
different hiring handles. This was a real classifier bug: uppercase domain
acronyms are not reliable named-entity anchors.

An entity-first counterfactual raised combined source recall only from 49.1% to
50.2%. It improved the Douglas entertainment query from `1/9` to `3/9` source
turns by expanding three Douglas handles instead of one focus handle, but both
frozen DeepSeek contexts scored `0.0`. The product routing-order change was
therefore reverted. Focus-first and entity-first judged artifact SHA-256s:
`810766a2e43d0d0749bbec924c137d026d5d713cac2c77bf87da925a5d794f0c`
and `4b9c11e68c865dfdff89b3436ff5fa778f9eb57871cf9f936e16d1c0dcac7e28`.

The other exact named-person routes show why source membership is necessary
but not sufficient:

- Patrick: the entity route contains all `6/6` source turns, but 29 raw items
  score `0.5`. DeepSeek then generated 13 query-blind concerns with all `29/29`
  selected evidence IDs assigned; the six Patrick-grounded concerns also score
  `0.5`. The hidden answer requires one merge for the PMR discussion but splits
  the interview/stress and leadership/implementation updates, a granularity
  policy not stated by the query.
- Douglas estate plans: the entity route contains the full source set, but 44
  literal Douglas memories lead the answerer to the earliest five valid plans
  and score `0.3`, not the benchmark's later five-item partition.

The corresponding SHA-256s are
`b1a3eb80c108449e068ff80ddd76dc4d70c1ba460a04ac26cc07e9573206627e`
for Patrick raw,
`a048bcd68abb1420d674cc5d930ed7c1b88cfdbf8ac89554536caddb73cc9a92`
for Patrick concern generation,
`9297ec305c137aaec8c904c5bb835702a3a85177d03baa1d816e0b0fd515fe25`
for Patrick filtered concerns, and
`ec518d71ab680263aa94590ef71fdd5ecc6456a1796648e014f0ef93817d2e75`
for Douglas estate plans. The complete route report SHA-256 is
`b42e30d9f6652d38423cd225e6acc2358271790129c41c9ea551aee4692a01da`.

The keep/reject boundary is now sharper: keep exact entity routing, correction
preservation, grouped concern chronology, and exhaustive generation checks.
Reject broader entity expansion and query-time merge/split heuristics whose
only justification is recovering an unstated benchmark partition.

Reproduce the structural report with:

```powershell
python benchmarks/amb/analyze_organizer_membership.py `
  --beam-source C:\path\to\agent-memory-benchmark\.datasets\beam\100k.json `
  --context-artifact benchmarks/amb/artifacts/summarization-full40-topic-card-contexts-dated-v9.json `
  --artifacts-dir benchmarks/amb/artifacts `
  --out benchmarks/amb/artifacts/event40-organizer-membership-v1.json
```

Therefore Q9 is no longer a sound target for more retrieval expansion or prompt
tuning, and the same stopping rule applies to similarly underidentified
event-ordering prompts. The evidence-backed product changes to retain are
relevance-first candidate presentation, role-sensitive query-blind organization,
grouped child hydration, and explicit correction filtering. Cohort work should
move to source losses where the requested facet is identifiable; otherwise
optimization would teach the engine benchmark-specific preferences that make
real answers worse.

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
