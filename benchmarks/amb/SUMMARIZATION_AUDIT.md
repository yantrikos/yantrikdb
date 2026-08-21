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

## Coverage Routing Bug

The persisted write-synthesis provider already has topic/thread rollups for
broad coverage queries, but its inline classifier recognized only phrases such
as "over time", "across", or "chronological". It classified just `17/40`
summarization questions as coverage intent. Queries explicitly saying
"summarize", "summary", "overview", "evolved", or "developed" could fall back
to ordinary source top-k retrieval.

The classifier is now centralized as `is_coverage_query` and recognizes all
`40/40` summarization questions while excluding narrow current-value questions
such as "What is my current monthly budget?" This routes summaries to persisted
topic/thread expansion without changing the answer prompt or discarding source
evidence.

## Decision

Do not increase raw top-k globally: a 32.8% context increase recovered only
3.45 percentage points of source-normalized rubric tokens. Route explicit
summary intent to existing persisted rollups, then pair-score that arm before
claiming a benchmark lift. The remaining gap is mostly downstream answer
selection; memory-layer work should focus on compact, source-grounded topic
items rather than more raw blocks.
