# AMB Count/Set Intent Preregistration

## Hypothesis

For count/set questions, stable user-first presentation of the same retrieved
evidence improves answer quality by making user claims more salient without
discarding assistant support. This is an independent confirmation attempt for
the post-hoc signal reported in `MULTI_SESSION_REASONING_AUDIT.md`.

## Frozen Cohort

The cohort is selected without consulting scores, gold answers, or answer
text. `prepare_paired_intent_contexts.py` includes questions matching the
case-sensitive pattern `^(?:How many|How much|What two)\b` and excludes the
previously analyzed `multi_session_reasoning` and `temporal_reasoning`
categories.

- Queries: 18
- Categories: knowledge update, information extraction, instruction following
- Arm A: frozen `ydb-0151` context
- Arm B: the identical memory-block multiset in stable user, unknown, assistant
  order
- Model: `deepseek-v4-flash:0731-cloud`
- Judge repeats: 3, aggregated by median
- Answer calls: 36
- Judge calls: 108
- Manifest SHA-256:
  `48ed0d959568e995d6b31a2aeff929a748cd5150f9f7d1e3ac45dc7d6b25adb4`
- Query-ID SHA-256:
  `01b9abf8673fd37582ce8eb073eb8c66c63e745e60562fa26e0598133021ac74`
- Arm A SHA-256:
  `5d46b4ac23f5a6dce6792f4c23d75b2eae775ea68e726ca2cc05157782579a66`
- Arm B SHA-256:
  `370621a4b80a2230f79aa241e60dda465c666c58779dc466182bfd1545ce598b`

## Analysis And Gate

The primary statistic is the mean paired score delta, arm B minus arm A. Its
95% interval is computed by the existing 20,000-sample paired bootstrap with
seed `20260820`. Wins, ties, and losses are secondary diagnostics.

The experiment passes only when the mean delta is positive and the bootstrap
95% lower bound is greater than zero. A point estimate above zero with an
interval crossing zero is directional evidence only and does not authorize a
product change. Category-specific and wording-specific splits are exploratory
because this holdout is too small to power them independently.

## Result

The frozen run completed all 18 pairs using the committed manifest:

- Mean arm A: `0.7083`
- Mean arm B: `0.7083`
- Mean paired delta: `0.0000`
- Paired bootstrap 95% interval: `[0.0000, 0.0000]`
- Outcomes: 0 arm-B wins, 18 ties, 0 arm-A wins
- Exact answer text: identical for 13 of 18 pairs
- Median-of-three score: identical for all 18 pairs

The five text differences were semantically immaterial formatting or wording
changes, such as "52" versus "52 sources". Two pairs had different individual
judge-vote vectors, but their medians were unchanged.

## Decision

The preregistered gate failed. Do not implement user-first presentation for
count/set intent. Together with the uncertain multi-session exploratory result,
this shows that speaker-block ordering is not a reliable score lever when the
evidence set is held constant. Further work should target evidence identity,
revision provenance, and item construction rather than presentation order.
