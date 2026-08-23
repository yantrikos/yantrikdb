# AMB Complete Facet Composition Preregistration

Status: v2 protocol frozen after the v1 pre-call abort; product preflight
passed and external scoring was authorized without changing the mechanism.
The completed result and no-promotion decision are recorded in
`COMPLETE_FACET_COMPOSITION_RESULT.md`.

## Question

Can additive composition of the complete verified standing-instruction lane
preserve the instruction-following lift without the unrelated-category
context displacement observed in the rejected full-400 default-on arm?

## Why V2

The preregistered v1 scope-similarity selector aborted before external scoring:
it retained 36 of 40 canonical instruction targets. All four misses were
answer-form rules, such as date formatting, that apply to a topical question
without being what the question is about. Changing `k` after this audit is
forbidden.

V2 removes query selection. Each BEAM namespace's complete verified lane fits
below the additive 256-token ceiling. At this scale, selecting a subset would
add semantic failure modes without solving a budget problem. Selection under
genuine budget pressure is a separate follow-up for larger facet stores.

## Frozen Mechanism

- Cohort: all 400 rows from the same frozen `ydb-0151` control.
- Control: every original context byte remains unchanged.
- Treatment: a dedicated standing-instruction panel is prepended. No ordinary
  retrieved block is removed, truncated, reordered, or rewritten.
- Extraction uses persisted, verified `standing_instruction` facets and their
  user evidence. The store is closed and reopened before composition.
- Every query receives its namespace's complete verified facet lane in
  first-mention order, with RID as the deterministic tie-break.
- Any lane truncation aborts. There is no query route, similarity threshold,
  category route, or score-tuned parameter.
- The facet panel has a separate ceiling of 256 tokens. It is additive rather
  than charged against the ordinary retrieval budget.
- Composition may read facet text, first-mention time, and RID. It may not
  read query text, category, gold answers, rubrics, prior answers, or scores.

## Pre-Call Abort Gate

`build_complete_facet_contexts.py` must construct the arm through the public
product path and abort before any model call unless all conditions hold:

1. Exactly 400 rows are aligned and all 40 canonical instruction targets are
   present in their treatment panels. Target metadata is used only for this
   abort audit, never for composition.
2. Every row receives every verified standing facet in its namespace.
3. Every original context is an exact byte suffix of treatment and its parsed
   ordinary memory-block sequence is unchanged.
4. No facet panel exceeds 256 tokens.
5. A second fresh product open reconstructs identical selected RIDs and an
   identical treatment artifact.

Failure abandons or redesigns the arm before scoring. The mechanism and budget
must not be tuned against external full-400 scores.

## External Run

- Model: `deepseek-v4-flash:0731-cloud`.
- One answer and one judge per arm per row.
- Projected calls: 800 answers and 800 judges.
- Synthetic benchmark data only; no real companion memories.
- The paired bootstrap uses 20,000 resamples and seed `20260820`.

## Promotion Gates

All gates must pass:

1. `instruction_following` delta is at least `+0.05`, with more wins than
   losses.
2. Overall delta is non-negative and its paired 95% interval lower bound is at
   least `-0.01`.
3. The pooled other-nine-category delta is at least `-0.005`.
4. `summarization` delta is at least `-0.01`.
5. No other category delta is below `-0.025`.

Every category mean, paired win/tie/loss count, and bootstrap interval is
reported regardless of outcome. Failure keeps complete-lane composition
opt-in and does not authorize post-hoc routing against benchmark categories.
