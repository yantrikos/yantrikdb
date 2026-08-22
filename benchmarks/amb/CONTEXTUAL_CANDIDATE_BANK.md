# Contextual Candidate-Bank Preflight

## Question

Can a bounded, evidence-preserving expansion improve the source-evidence ceiling
for AMB event-ordering queries before any extraction, selection, answer, or judge
call is made?

This is a reachability test, not an answer-score claim. It uses the frozen AMB
groups 1-9 cohort: 18 queries and 95 authoritative query-level source turns.

## Mechanism

The base lane reranks every user turn with query-to-turn cosine similarity from
`nomic-embed-text`, then keeps the top 40. The candidate-bank lane preserves all
40 and may add three bounded evidence classes:

1. Up to ten turns from recurring named-person threads seeded by the contextual
   top 10. Person detection requires relationship language such as feedback,
   advice, meeting, or recommendation; generic capitalized words are excluded.
   Candidates are allocated across people before a second turn from one person
   is added.
2. Up to two direct user continuations. A continuation must immediately follow
   a selected user turn and begin with a continuation marker such as `Sure` or
   `Yes`.
3. Up to two context bridges. A bridge must immediately follow a selected user
   turn and carry specific anaphoric relationship language such as `our Zoom
   meeting` or `that feedback`. Generic phrases such as `such a short time` do
   not qualify. The parent can itself enter through named-person closure, which
   permits a bounded two-step chain without global graph expansion.

Every input row receives a keep/drop reason. The artifact pins the Ollama model
digest and hashes the source input, complete rerank, candidate bank, and cohort.
A comparison run fails closed if any hash changes.

## Result

Both complete runs produced cohort SHA-256
`3f442beb0cf5b91c72fe1a1685f7092866e0e8ea538ad37444f26a7e9e48521f`.
The model was `nomic-embed-text:latest`, digest
`0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`.

| Measure | Contextual top 40 | Bounded candidate bank |
| --- | ---: | ---: |
| Source turns available | 55/95 (57.9%) | 60/95 (63.2%) |
| Queries with every source turn | 3/18 | 5/18 |
| Candidate count, mean | 40 | 42.6 |
| Candidate count, median | 40 | 41 |
| Candidate count, maximum | 40 | 50 |

The five newly reachable turns occur in three broad queries:

| Query | Added evidence |
| --- | --- |
| `7_event_ordering_0` | Robert recommendation, turn 124; relationship-linked follow-up, turn 214 |
| `8_event_ordering_0` | Greg portfolio advice, turn 8 |
| `9_event_ordering_0` | Bryan recommendation, turn 108; direct continuation, turn 168 |

No base evidence is removed, so this lane cannot lower source reachability. It
can still reduce answer quality by increasing selector load; that must be tested
separately.

## Decision

Keep the candidate bank as a measured optional experiment. Do not promote it to
product behavior or claim an AMB score lift yet. The gain is real but small, and
the five added turns belong to only three broad prompts. Q7 now has a coherent
Robert relationship chain; Q8 and Q9 still have underidentified partitions.

The context bridge is a stronger product candidate than unconstrained synthesis:
it preserves source evidence, exposes its parent turn and keep reason, has a
separate two-row budget, and raised clean Q7 from 3/5 to 5/5. It still requires a
score-bearing cohort evaluation before promotion.

In particular, broad Q9 is not a valid optimization target: its authoritative
source set requires an earlier Bryan event that the history later corrects. A
product memory system must preserve the correction even when doing so disagrees
with the benchmark rubric.

The next score-bearing run must report the frozen source-evidence ceiling beside
the answer score and must use a query whose requested facet is identifiable and
whose gold does not conflict with current memory state.

## Query-Dependent Thread Result

Q7 supplied that clean score-bearing case. The bridge raised its source ceiling
from `3/5` to `5/5`, but both synthesized and raw 44-row contexts still scored
`0.00`. DeepSeek selected generic academic-writing events instead of the latent
Robert mentorship trajectory. This ruled out extraction rendering as the
remaining explanation.

A deterministic relationship-thread reducer then used only the query's explicit
`mentorship` intent, recurring high-confidence person mentions, context-bridge
parent links, and stored source provenance. It selected Robert, retained eight
source rows across five `source_doc_id` groups, and preserved all `5/5` source
turns. It did not generate, merge, or rewrite any memory.

| Frozen arm | Rows / groups | Context tokens | Median score | Judge votes |
| --- | ---: | ---: | ---: | --- |
| Raw bridge bank | 44 rows | 3007 | `0.00` | `0.00, 0.00, 0.00` |
| Relationship thread | 8 rows / 5 groups | 624 | `0.70` | `0.70, 0.70, 0.70` |

The raw arm reused SHA-256
`89989e9f27b9f50f2d11c27e4d8adf2342623535b2657e95fbe3ee49ecd2a5f3`
from the prior null run. The grouped arm SHA-256 was
`6dc0a4610ac2b9ca2bbc3fab4224ec4675bc4c80f41f72f670951b6cd3d29df7`;
the paired manifest SHA-256 was
`290e576b007665e32ac23258db52cb4477dc51a6171e904e6554dba8e1958d70`.
The model snapshot was `deepseek-v4-flash:0731-cloud` for both answering and
three repeated judgments. The paired delta is internally comparable, but these
scores are not line-comparable to runs judged by the rolling `:cloud` tag.

This is evidence for query-dependent thread selection, not for write-time
cross-session synthesis. The treatment still chose two early distractors from
otherwise correct source groups, so within-session representative selection is
the residual gap. Keep this route experimental until an untouched relationship
timeline query confirms the lift without a category regression.

### Within-source representative preregistration

The residual arm was frozen before scoring. Inside each selected source
conversation, it ranks direct relationship mentions by a small decision-cue
signal (`decide`, `whether`, `prioritize`, `focus`, or `approach`), then by the
existing contextual score and chronology. It retains only the winner plus any
`context_bridge` row whose stored `parent_turn` points to that winner. This cue
lexicon is benchmark harness logic only; a product implementation must consume
structured organizer intent rather than these words.

The rule was derived from Q7, so Q7 alone is not validation. Before freezing the
judged pair, the identical rank was applied to Q9_1's five family-support source
groups. All cue counts were zero and the contextual-score fallback reproduced
the exact prior identities and turns (`24, 76, 118, 208, 260`). The full frozen
18-query audit retained `10/10` fired gold turns, preserved cohort reachability
at `60/95`, had zero negative queries, and reduced fired rows from `92` to `11`.
Its audit SHA-256 is
`757e9a4226e156701146a89a0c705ca171123abf3495b22bcaaae87d90b7a0d3`.

The Q7 control is the accepted eight-row relationship thread and exactly
reproduces SHA-256
`6dc0a4610ac2b9ca2bbc3fab4224ec4675bc4c80f41f72f670951b6cd3d29df7`.
The six-row treatment keeps turns `14, 64, 124, 170, 212, 214`, drops only
`156, 168`, and has SHA-256
`2c8bd6368ca1bb4cd7a68de0edbe8679101c427efd59f12656d97595d6609494`.
The paired manifest SHA-256 is
`3fdd24417b933d8c5938026ceb6ffff6e68d6e4b05269fb5a76368d5c77fb2f8`.

The preregistered hypothesis is that removing the two same-source distractors
raises Q7 above the control's unanimous `0.70`. The null is a median paired
delta of `0.00`; any negative delta is a failed arm. Answering and three repeated
judgments use the pinned `deepseek-v4-flash:0731-cloud` snapshot. The result is
internally paired but not line-comparable to the rolling `:cloud` benchmark.
No other query will be judged for this residual.

The frozen run rejected the hypothesis:

| Frozen arm | Rows | Context tokens | Median score | Judge votes |
| --- | ---: | ---: | ---: | --- |
| Relationship thread control | 8 | 624 | `0.70` | `0.70, 0.70, 0.70` |
| Within-source representatives | 6 | 489 | `0.50` | `0.40, 0.50, 0.50` |

The paired delta was `-0.20`, a preregistered failed arm. No rerun was made.
The treatment answer identified the five broad stages, but compressed away exact
rubric details such as the Zoom-call context, improving the essay before the
conference paper, and follow-up Zoom/conference planning. This result rejects
raw-row deletion as the residual fix. It points back to representation: a source
stage needs an answer-sized synthesized item that preserves its distinguishing
detail. The representative selector remains opt-in experimental harness code
and must not be promoted to a product default.

### Global stage-record result

The next arm tested representation directly. It made one query-blind global
synthesis call over the accepted eight-row Robert thread and required exactly
one record for each of the five stored source conversations. The model could
write only the stage sentence. The harness attached first-mention time, source
conversation, turns, and all eight evidence IDs deterministically; full IDs
remained in audit metadata instead of consuming answer context.

The synthesis selection SHA-256 was
`6b0ee0e74184740785c3decb4309524408d6fb271af4b024e54452882dfcd892`,
the request SHA-256 was
`e813693d86a49aed04e65890116bd74bd14c56e0b1a400032e4747290f1a1916`,
and the response SHA-256 was
`4edc534772bb19b440b7f619e1e323a08f5d77e325329b579c54e0af65c5e874`.
The frozen control and treatment context SHA-256 values were respectively
`0b09412744f477c6e095949f9a938271762978849d736e1805eb52f82efd9f17`
and
`f3229add21c6da073cdc766073c3c76129e467838dd9be40475ca489251abeb0`.
The paired manifest SHA-256 was
`a7aab531232e2be20bf60d05b8e32ba838b15458c8c06d167e851a6e2a8d0c49`.

The frozen run rejected compact prose stage records:

| Frozen arm | Rows | Context tokens | Median score | Judge votes |
| --- | ---: | ---: | ---: | --- |
| Relationship thread control | 8 | 624 | `0.60` | `0.60, 0.60, 0.70` |
| Global stage records | 5 | 331 | `0.20` | `0.20, 0.20, 0.20` |

The paired delta was `-0.40`; no rerun was made. The model converted the
first-person concern about meeting Robert and making a good impression into a
third-person fact. The answerer then explicitly excluded that record as not an
academic-work aspect and split the June record into separate journal and
conference items. The synthesis also omitted the central June decision to
strengthen the essay before the conference paper and compressed the July
confidence and conference-planning details into generic next steps.

This rejects untyped compact prose as the item representation. A subsequent
arm must preserve speaker perspective and separately encode the user's
goal/concern/decision, event facts, and outcome or follow-up. Q7 is now a
development case for that schema, not untouched validation.

### Untouched holdout

The untouched Q9 family-support query provided that confirmation. Its raw bank
already contained all `5/5` source turns, but 48 flat rows still scored `0.00`
with unanimous judgments. A separate multi-person policy extracted only names
grounded by explicit family roles, required active support language, grouped
matches by `source_doc_id`, and selected the highest query-scoring row in each
source conversation. It selected Wendy and Tanya without receiving the gold
answer or source turns.

| Frozen arm | Rows / groups | Context tokens | Median score | Judge votes |
| --- | ---: | ---: | ---: | --- |
| Raw family-support bank | 48 rows | 3374 | `0.00` | `0.00, 0.00, 0.00` |
| Relationship-support stages | 5 rows / 5 groups | 366 | `1.00` | `1.00, 1.00, 1.00` |

The manifest SHA-256 was
`799ec1595e41f5fa94ad12fc3159272ece8f857f9fbcf96a0351a9b7cf4b2530`;
the treatment context SHA-256 was
`9c6c1ee6980ab55f5218a036ea81101b7fc53a8c1e0695745280497d0f47b8a2`;
and the deterministic selection SHA-256 was
`201d555440e6a1dd4811a8e1554046d35d41cb22b1386e53cbff445c58690695`.

The two paired results isolate the architecture: broad retrieval remains the
reachability layer, while relationship handles and source-conversation groups
form a bounded query-time selection layer. They do not justify general regex
rules as a product default; production integration should consume organizer
entities, relationship roles, provenance, and query intent from structured
metadata, with the benchmark regexes retained only in this experimental harness.

## Full-Cohort Routing Gate

The first judge-free routing audit exposed one false fire: treating
`professional connections` as a single-person timeline selected Greg for
`8_event_ordering_1` and retained `0/5` source turns. That broad rule was
rejected before scoring. The final classifier fires only for explicit
mentorship/advisor intent or explicit family-support intent; all other prompts
fall back to the unchanged candidate bank.

The final audit ran over the same frozen 18-query cohort, SHA-256
`3f442beb0cf5b91c72fe1a1685f7092866e0e8ea538ad37444f26a7e9e48521f`:

| Measure | Result |
| --- | ---: |
| Selector fires | 2/18 (11.1%) |
| Selector abstains | 16/18 (88.9%) |
| Gold retained in fired queries | 10/10 (100%) |
| Cohort source turns before / after routing | 60/95 / 60/95 |
| Fired payload rows before / after | 92 / 13 |
| Fired payload reduction | 85.9% |
| Queries with a negative source-turn delta | 0 |

The deterministic audit SHA-256 was
`f8c3f9e5a5d58ee3af0b1818a00c098e3420badca7af97ce57c3cf017c279613`.
This clears the judge-free fire/abstention and gold-retention gate for the two
narrow intents. It does not clear the full score cohort or no-category-regression
operator gate required before default promotion.

## Reproduction

```powershell
$env:PYTHONPATH='C:\path\to\agent-memory-benchmark\src'
python benchmarks/amb/contextual_funnel_preflight.py `
  --repo . `
  --bank C:\path\to\frozen-bank `
  --beam-source C:\path\to\agent-memory-benchmark\.datasets\beam\100k.json `
  --top-k 40 `
  --rerank-pool 1000 `
  --entity-seed-k 10 `
  --entity-closure-slots 10 `
  --continuation-slots 2 `
  --context-bridge-slots 2 `
  --expect-model-digest 0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f `
  --output cohort-preflight.json
```

Repeat with `--compare-preflight cohort-preflight.json` and a different output
path. The command must report `"matched": true` before any score-bearing arm is
run.
