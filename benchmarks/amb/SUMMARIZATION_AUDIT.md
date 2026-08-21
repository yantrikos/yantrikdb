# AMB Summarization Audit

## Scope

The frozen `ydb-0151` BEAM-100K run scores `0.5930` on 40 summarization
questions. There are no zeroes; each query is scored against three to six
rubric facts. This audit examines the ten rows scoring at most `0.40` (47
rubric items) and separates source, retrieval, and answer loss.

## Deterministic Funnel

`audit_summarization_lexical_funnel.py` extracts distinctive words, names,
numbers, and percentages from each rubric item. It measures their set coverage
in the complete source conversation, retrieved context, and final answer, then
normalizes context and answer coverage to tokens actually supported by source.
This is a lexical diagnostic, not a semantic correctness judge.

For the low-ten cohort:

- Mean rubric-token support in source: `93.72%`
- Source-normalized retrieval coverage: `83.47%`
- Source-normalized answer coverage: `46.36%`
- Items below 75% retrieval coverage: `7/47`
- Items below 75% answer coverage: `42/47`

The dominant loss is answer compression/selection after retrieval. Examples of
the seven weaker retrieval items include a Bootstrap navbar DOM-safety fix, a
single-column cover-letter format, Toronto clothing budget details, hiking
shoe model comparisons, and therapy/workplace-conflict milestones.

## Wider Mixed-Speaker Context

The frozen role-aware mixed-speaker artifact raises source-normalized retrieval
coverage from `83.47%` to `86.92%` and reduces weak items from seven to six.
However, its context budget grows from `156,157` to `207,442` tokens across the
same ten queries, a `32.8%` increase. This is poor leverage and does not address
the much larger answer-stage loss.

A quote-grounded model probe was kept fail-closed: every positive requires one
or more locally verified context spans. DeepSeek transport exhausted its retry
budget, the 0.8B model could not follow the complete structured contract, and
the 4B model was too slow for the full cohort. One completed 4B row confirmed
two of five rubric items in context, consistent with partial rather than empty
retrieval. Incomplete or invalid model runs are not used as evidence.

## Source-Turn Rollup Controls

A proposed intent expansion would have routed all `40/40` summarization
questions to persisted rollups instead of the `17/40` carrying explicit
coverage phrases such as "over time", "across", or "chronological". The
proposal was tested before being retained. A fail-closed replay disabled all
model-generated write-time axes, persisted verbatim user turns, and built
query-independent semantic and global handles locally.

On three low-scoring preflight rows (13 rubric items), replacing raw context
with semantic-handle children reduced source-normalized retrieval coverage from
`81.52%` to `72.63%`; a global-handle replacement reached only `74.27%`.
Bounded `20 derived + 20 raw` and `10 derived + 30 raw` hybrids also remained
below baseline. A `5 derived + 35 raw` arm reached `82.67%` on the small cohort
and scored `+0.025` over four DeepSeek pairs (`1` win, `3` ties), but failed to
generalize: the full low-ten cohort fell from `83.47%` to `82.67%` and gained
one additional weak item.

Preserving the complete raw lane and prepending five derived turns raised the
full low-ten lexical funnel to `84.39%` for `4.54%` more context tokens. Its
four-pair DeepSeek control nevertheless regressed by `-0.05625` (`0` wins, `2`
ties, `2` losses). The identical frozen baseline context also moved from
`0.43125` in the preceding run to `0.4875`, confirming material answer/judge
variance at this sample size. The broad intent expansion and augmentation arm
are rejected.

## Decision

Do not increase raw top-k globally: a 32.8% context increase recovered only
3.45 percentage points of source-normalized rubric tokens. Do not route generic
"summary" wording to a rollup-only lane, and do not prepend derived turns merely
because they improve lexical coverage. The remaining gap is mostly downstream
answer selection; future work needs compact source-grounded items whose utility
is demonstrated on a broad frozen cohort, with repeated judging to separate a
real lift from model variance.
