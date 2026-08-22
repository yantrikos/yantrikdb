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

## Dated Topic-Card Experiment

A query-independent DeepSeek organizer was then tested over verbatim user-turn
atomics. Every card retained source evidence IDs. Omitted items were preserved
as deterministic verbatim singleton cards; invented IDs were rejected and
audited. The card presentation added locally derived first/last recorded dates
and turn spans. These are storage chronology, not model-inferred event dates.

Organizer generation was extended to all 20 units with a fail-closed recovery
protocol for long responses. Recovery is allowed only when the transport marks
the response as length-truncated, and only fully decoded handle objects with the
complete schema are retained; a partial tail is discarded. Complete responses
that exceed the requested handle bound use the same deterministic cap, greedily
maximizing distinct valid evidence coverage and then restoring model order.
Raw responses and pre/post-cap telemetry remain in the audit artifacts. All 20
selected artifacts pass their input hashes, contain no surviving invented IDs,
assign every atomic item, and have no unassigned evidence.

On four preflight queries, dated cards alone used 27.6% of the raw token budget
and scored `0.6792` versus `0.4458` raw (`+0.2333`; three wins, one tie).
Cards alone did not pass the broader lexical gate, however: low-ten retrieval
coverage fell to `66.93%`. Prepending every dated card to the unchanged raw
context preserved evidence, raised lexical coverage from `83.47%` to `90.58%`,
and reduced weak rubric items from `7/47` to `2/47`.

The repeated-judge low-ten decision run with every card before the raw context
scored `0.5358` versus `0.3892` raw, mean paired delta `+0.1467`, with six wins,
two ties, two losses and bootstrap 95% CI `[+0.0350, +0.2617]`. Context tokens
increased 25.4%. A six-query companion regression cohort from the same units
moved in the other direction: `-0.0972` mean delta, one win, one tie, four
losses, CI `[-0.2347, +0.0181]`. Across all 16 tested rows the mean delta was
only `+0.0552`, and its CI crossed zero.

Two cheaper gates did not solve that regression. Query-ranked top-eight cards
used only 4.4% more tokens and slightly raised low-ten lexical coverage from
`83.47%` to `85.02%`, but judged delta was a null `+0.0008` (two wins, six ties,
two losses; CI `[-0.0583, +0.0600]`). A labels-only topic index used 3.6% more
tokens and raised lexical coverage to `85.64%`, but regressed by `-0.0458` (one
win, four ties, five losses; CI `[-0.1025, +0.0150]`). A strict
missing-query-term gate was also rejected: it selected 11 of 40 rows largely
because generic verbs and pronouns were absent, including already-strong
answers. These controls show that broad synthesis coverage matters and that
compact labels can distract the answer model instead of guiding it.

## Raw-First Ordering

The same complete dated cards were then moved after the unchanged raw timeline.
This is the only substantive difference from the prepended-card arm. On the
low-ten decision cohort, raw-first ordering scored `0.5550` versus `0.3933` raw:
mean paired delta `+0.1617`, five wins, four ties, one loss, bootstrap 95% CI
`[+0.0442, +0.2900]`. Context tokens increased 24.5%. Lexical retrieval
coverage was `90.58%`, with only `2/47` rubric items below 75%.

The companion cohort no longer regressed: it scored `+0.0306`, with three wins,
two ties, one loss and CI `[-0.0625, +0.1000]`. Across all 16 frozen rows,
raw-first complete synthesis produced mean delta `+0.1125`, eight wins, six
ties, two losses and bootstrap 95% CI `[+0.0292, +0.2063]`. This is the first
broad statistically positive result in the summarization audit.

The evidence supports a concrete retrieval contract: preserve exact source
chronology first, then append complete query-independent, evidence-grounded
dated topic cards as a consolidation checklist. Do not replace raw evidence,
prepend the cards, top-k the cards, or substitute a labels-only index. Exact
duplicate cards are removed while their evidence dates are unioned. The result
is validated on 16 queries from eight units, not yet the full 40-query category;
the next step is a production/replay provider arm followed by full-category
validation before changing the default runtime path.

An opt-in provider replay then persisted the unit-3 organizer through the
public organization API and structurally enumerated all 39 handles. Its dated
card documents matched the winning frozen presentation exactly, including
order. With 40 live raw records, the two-query transfer preflight scored
`0.6250` versus `0.5250` frozen raw (`+0.1000`, one win, one tie), but context
cost grew 62.0%. Reducing only the live raw lane to 30 records retained the
`0.6250` candidate score and no-loss result (`+0.0500` versus a `0.5750`
repeated baseline) while limiting context growth to 27.3%. The opt-in provider
therefore defaults to 30 raw records plus every card. This replay is a narrow
transfer check, not additional full-cohort evidence; the 16-query frozen result
remains the decision basis until organizer artifacts cover all 20 units.

## Full-Category Validation And Temporal Gate

Organizer artifacts were completed for all 20 units and passed a strict local
audit: their input hashes match, every atomic item is assigned, no invalid
evidence reference survives, and no evidence is unassigned. The frozen
full-category arm kept all 40 raw blocks first and appended every dated card.
Across all 40 summarization questions it scored `0.5966` versus `0.5634` raw,
for mean paired delta `+0.0332`, 13 wins, 19 ties, and 8 losses. The bootstrap
95% CI `[-0.0205, +0.0851]` crosses zero, while context grew 22.6%. Complete
cards therefore remain opt-in and must not become the generic default.

The earlier 16-query cohort replicated inside this run at `+0.1281` (nine
wins, five ties, two losses), but the 24 newly covered rows regressed by
`-0.0301`. A query-only split exposed a narrower mechanism: the four requests
that ask for a summary over explicit named calendar points scored `0.4750`
raw versus `0.7542` with raw-first cards, delta `+0.2792`, with four wins and
no losses. The classifier requires `summary` or `summarize` plus either a
named month and year, at least two named months, or explicit ISO calendar
points. It does not inspect query IDs, gold answers, scores, or retrieved text.

This four-query lane was positive without a loss in two additional repeated-
judge runs: `+0.2313` (three wins, one tie) in the prior frozen evaluation and
`+0.1313` (two wins, two ties) in a fresh confirmation. Across the three runs
the same mechanism produced nine wins, three ties, and zero losses. Applying
the gate leaves 36/40 contexts byte-identical to raw and adds only 3.0% context
tokens across the category, compared with 22.6% for generic complete cards.
The new `yantrikdb-temporal-summary-organizer` provider is an opt-in candidate:
raw-only on a gate miss, raw followed by every persisted dated card on a match,
and a trace containing the classifier evidence. This is a defensible narrow
win, but the four-query intent cohort is small and should be expanded on an
independent dataset before making it a default runtime policy.

## Public-API Transfer And Storage Bounds

The four gated queries were then replayed from disposable copies of the live
unit-18 and unit-20 stores through public `persist_organization` and the actual
`yantrikdb-temporal-summary-organizer` provider. The treatment's raw prefix was
byte-for-byte identical to a same-store current-version raw control, and its
card suffix matched the bounded organizer artifact presentation exactly.

The paired repeated-judge run scored `0.8042` for the product provider versus
`0.6479` for its raw control: mean delta `+0.1563`, three wins, one tie, no
losses, and bootstrap 95% CI `[+0.0500, +0.2563]`. The per-query deltas were
`+0.2000`, `0.0000`, `+0.1250`, and `+0.3000`. This transfers the temporal
summary mechanism through real persistence and retrieval rather than relying
on frozen context concatenation.

That replay also exposed a contract defect in the probe artifacts. The public
API admits at most 12 evidence items per handle and three handle memberships
per evidence item, but the 20 generated artifacts contained 25 overfull
handles, 33 overmembered evidence IDs, one 102-evidence handle, and 16 exact
duplicate handles in unit 20. The probe now applies deterministic final bounds:
duplicate stable identities are removed, narrower handles win membership
contention, overfull handles retain a temporally stratified sample, and any
dropped sole assignments become source-preserving singleton handles. The
bounded unit-18 and unit-20 artifacts persisted without invalid references or
unassigned evidence. These bounds are a storage-compatibility repair, not a new
score mechanism.
