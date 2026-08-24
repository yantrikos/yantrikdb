# AMB Event-Ordering v5 Autopsy

## Decision

The terminal v5 composition result leaves a retrieval problem, not another
composition-tuning opportunity. Across the 40 event-ordering queries, the
ordinary control contexts contain only `18.29%` of authoritative source turns.
Only one query has complete source coverage and 18 have none.

The next product mechanism should be a coverage-first chronological thread API
with multiple query-anchor routes. Entity-only retrieval is too narrow: after
quarantining three benchmark defects, only one row is a clean exact-entity
thread, one needs entity plus focus, 22 have bounded topic phrases, and 13 need
a compound topic union.

This is an autopsy and route-design input, not a preregistered score arm.

## Frozen Inputs

- Combined v5 result SHA-256:
  `16c18af8fda47e0de8a36f434b1ae129b585cfba0fbceacb9819216aa5031ee7`
- Control-context replicate SHA-256:
  `e139ef95d1e0b08c687a4d4159c502e26c0234677bbdab5b011c66a4460bd388`
- Organizer membership SHA-256:
  `8229c9ffe8d779295e4e161b003cd81afc45f0d4d5b53c07c82925eb54c2b86e`
- Generated autopsy SHA-256:
  `627b13de031d0ffa8970ae70aff2039027f88f29e3beb15f6ee927894978805c`

The analyzer verifies both run fingerprints, arm labels, replicate seeds,
per-row scores, and that the byte hash of the supplied replicate occurs exactly
once in the combined result's frozen source list. It parses source coverage only
from `result_a.context`. Treatment contexts prepend standing-facet evidence and
would falsely credit retrieval coverage.

## Failure Split

| Diagnostic | Result |
|---|---:|
| Mean authoritative source-turn recall | 18.29% |
| Exact source coverage | 1/40 |
| Zero source coverage | 18/40 |
| Correlation of source recall with treatment score | 0.4370 |
| Negative treatment deltas with incomplete sources | 17/17 |
| Treatment mean-zero rows | 7/40 |

Every treatment row has a score deficit from `1.0`. With known benchmark
defects taking precedence, the mutually exclusive classification is:

| Failure class | Queries | Mean treatment score | Mean deficit |
|---|---:|---:|---:|
| Retrieval miss | 36 | 0.2455 | 0.7545 |
| Source-complete answer residual | 1 | 0.7333 | 0.2667 |
| Benchmark-label defect | 3 | 0.3000 | 0.7000 |

The sole source-complete row is `2_event_ordering_1`. Its `0.7333` treatment
score is the useful warning against equating coverage with correctness. BEAM
scores rubric-item coverage, not ordering independently, so the residual bucket
includes selection, granularity, presentation, and reader behavior.

The three quarantined rows are:

- `9_event_ordering_0`: the gold requires a stale Bryan event after a user
  correction.
- `18_event_ordering_0`: the Patrick gold requires an unstated merge/split
  granularity despite a complete entity route in the separate product probe.
- `19_event_ordering_0`: the Douglas gold selects an unstated later partition
  over earlier valid plans.

## Query Routes

The taxonomy is query-only and mutually exclusive. The top-three topic column
is an oracle ceiling: topics were generated without the query or gold, but gold
source IDs select the best three only after generation. It proves structural
availability, not that the product can choose those handles.

| Recommended route | Queries | Control source turns | Oracle top-three topics |
|---|---:|---:|---:|
| Benchmark-defect quarantine | 3 | 1/16 (6.3%) | 15/16 (93.8%) |
| Exact entity thread | 1 | 0/6 (0.0%) | 6/6 (100.0%) |
| Entity plus focus | 1 | 3/9 (33.3%) | 5/9 (55.6%) |
| Bounded focus phrase | 22 | 25/124 (20.2%) | 109/124 (87.9%) |
| Broad compound topic union | 13 | 8/75 (10.7%) | 72/75 (96.0%) |

The row assignments are frozen in
`analyze_event_ordering_v5_autopsy.py`. Notable decisions:

- `10_event_ordering_1` is the only clean exact-entity route: Carla.
- `13_event_ordering_1` needs Douglas plus the focus "shared entertainment
  interests"; the whole Douglas thread is too broad.
- `11_event_ordering_0` treats `AI` as a bounded hiring-domain phrase, never as
  a named entity.
- Broad prompts such as "writing journey", "professional progress", and
  "career development and relocation" need two or three complementary topic
  handles rather than one global semantic top-K.

## Product Boundary

`recall_thread` v2 should preserve the exact-entity fast path while adding
query-selected phrase and topic-union anchors. For the benchmark arm it must:

1. Select anchors from the query without source IDs or rubric text.
2. Union the selected threads before chronological composition.
3. Record `total`, `returned`, and `omitted`, and abort the arm when
   `omitted > 0`.
4. Keep correction and supersession semantics intact.
5. Hydrate and decrypt only the bounded result page in the product path.

Topic membership is already product-persisted. Organizer handles are additive
synthesis rows in `memories`; their direct evidence is relationally available
through `synthesis_dependencies` (`is_direct=1`, namespace-scoped) and
`consolidation_members`. A bounded v2 API should accept already-resolved topic
synthesis RIDs and join those tables. Looking up topic labels by enumerating and
decrypting organizer metadata would recreate the full-materialization problem.

The next gate is a local source-membership preflight for product-selected
anchors, not an official score run. A score arm is justified only if query-only
selection materially closes the gap toward the topic-union ceiling while
keeping the three quarantined rows out of optimization decisions.

## Reproduce

```powershell
uv run python benchmarks/amb/analyze_event_ordering_v5_autopsy.py `
  --combined C:\path\to\combined-v5.json `
  --replicate C:\path\to\replicate-20260828.json `
  --membership benchmarks/amb/artifacts/event40-organizer-membership-v2.json `
  --output C:\path\to\event-ordering-v5-autopsy.json
```

The command is deterministic and makes no model or network calls.
