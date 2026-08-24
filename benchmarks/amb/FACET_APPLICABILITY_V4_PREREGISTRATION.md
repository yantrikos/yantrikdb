# AMB Standing-Facet Applicability V4 Preregistration

Status: mechanism and gates frozen before product preflight or external calls.

The completed zero-call product preflight and paired artifact freeze are
recorded in [FACET_APPLICABILITY_V4_PREFLIGHT.md](FACET_APPLICABILITY_V4_PREFLIGHT.md).
The frozen evaluation and no-promotion decision are recorded in
[FACET_APPLICABILITY_V4_RESULT.md](FACET_APPLICABILITY_V4_RESULT.md).

## Question

Can a narrow, auditable rule-type versus answer-shape predicate preserve the
standing-instruction lift while removing the form-transform collisions caused
by complete query-independent composition?

V2 improved the full cohort by `+0.028727` and instruction following by
`+0.093750`, but event ordering crossed the frozen harm floor. The subsequent
zero-call audit found plausible collisions in `4/10` event-ordering losses and
prohibited an identical replication. V4 changes the mechanism; it does not
reuse or pool v2 answers or scores.

## Design Law

V4 is **default-include and suppresses only on a positive form conflict**.
Topic similarity is not used. Absence of a match never suppresses a facet.
Unparsed or ambiguous directives remain included.

The predicate version is `facet-form-conflict-v1`. It is recorded in every
row audit and decision trace.

### Directive Parsing

The parser recognizes only the exact conditional form:

```text
<action> when I ask about <condition>
```

It preserves the exact action and condition text. The action is typed as:

| Type | Positive lexical evidence in the action |
|---|---|
| `date_time` | date, time, time zone, deadline, timeline, schedule, meeting, or appointment |
| `formatting_structure` | format, bullets, headings, citation style, syntax highlighting, tree diagram, visual aid, markup, HTML, layout, or minimalist style |
| `non_transforming` | every other parsed action |

Date/time typing runs before formatting typing, so `format dates` is a
`date_time` transform.

The condition is independently typed as `process_timeline`, `date_time`,
`formatting_structure`, or `other`. Process terms are checked first, then
date/time terms, then formatting terms.

### Query Answer Shape

The query-only classifier emits non-exclusive Boolean features:

- `chronology_of_mentions`: requires both an ordering cue (`in order`,
  `order in which`, `chronological`, or `sequence`) and a conversation-history
  cue (`brought up`, `mentioned`, `discussed`, `throughout`, or `across`).
- `date_time_request`: explicit `when`, date/time, due, deadline, scheduled,
  scheduling, meeting-time, or appointment-time language.
- `process_timeline_request`: a steps/stages/process/procedure term plus a
  request term such as what, which, how, walk me through, or order.
- `formatting_request`: explicit format, organize, design, layout, structure,
  style, citation, references, markup, HTML, bullets, or headings language.

The classifier reads query text only. It never reads benchmark category, gold
answer, rubric, prior answer, judge output, or score.

### Compatibility Table

Outside `chronology_of_mentions`, every facet is included. Within a chronology
of mentions:

| Directive type | Include when | Otherwise |
|---|---|---|
| `non_transforming` | always | n/a |
| `date_time` | query requests date/time and condition is date/time; or query requests a process timeline and condition is a process timeline | suppress |
| `formatting_structure` | query explicitly requests formatting and condition is formatting/structure | suppress |
| unparsed | always | n/a |

This distinguishes `When was X?`, where date rules remain active, from `In
what order did I mention X?`, where an unrelated date-format rule can transform
the requested answer. It also retains a patent-process timeline rule for an
ordered patent-stage request and retains date formatting for an ordered list of
deadlines. Bare `list` language is not treated as a formatting request.

## Frozen Cohort And Composition

- Cohort: the same 400 BEAM 100k queries as the frozen `ydb-0151` control.
- Control: exact original context bytes, with SHA-256
  `918f572927b75ab1bb2ae3edf5656eada132cf9a644953f31bf693c695d46863`.
- Treatment: product-extracted, active, verified, user-backed standing facets
  filtered by `facet-form-conflict-v1` and prepended in first-mention order.
- No ordinary block is removed, truncated, reordered, or rewritten.
- The facet panel remains additive and capped at 256 tokens.
- Extraction, persistence, close/reopen, provenance validation, and dependency
  resolution are identical to v2.

The v4 treatment hash and paired manifest hash must be committed before any
external call.

## Query-Only Design Census

Before this protocol was frozen, a score-blind census over query text and the
v2 facet panels produced the following expected product-preflight values:

- rows: `400`
- unique persisted directive texts: `90`
- conditional parses: `90/90`
- canonical instruction targets retained: `40/40`
- rows with suppression: `28`
- facets suppressed: `53`
- suppression reasons: `41` date/time conflicts and `12` formatting conflicts
- explicit date-request rows: `41`
- date/time facet exposures on those rows retained: `68/68`
- compatible process-timeline inclusions inside chronology rows: `1`

No score, category, gold answer, rubric, answer text, or judge output was read
by the census. General product behavior still defaults unparsed directives to
inclusion; the frozen corpus's expected parse coverage is 100%.

A second query-only check covered the four already-frozen collision rows. All
`4/4` fall inside the 28-row suppression cohort. The exact named collision
directive is suppressed on rows 6, 9, and 10. On row 20, the patent-process
timeline rule remains included because the query explicitly requests stages of
the patent process; two unrelated deadline/meeting rules are suppressed. This
is the intended near-domain control, not an ID-specific exception: the same
predicate retains any process-timeline rule when the query asks for a process
timeline.

## Pre-Call Abort Gate

`build_applicable_facet_contexts.py` must rebuild the treatment through the
product path and abort before model calls unless:

1. All query-only census counts above reproduce exactly.
2. All 40 canonical instruction targets remain selected.
3. Every suppressed decision is a parsed `date_time` or
   `formatting_structure` directive on a `chronology_of_mentions` query.
4. All original contexts remain exact byte suffixes with identical parsed
   ordinary memory blocks.
5. No facet panel exceeds 256 tokens.
6. A second fresh product open reproduces identical selected RIDs, predicate
   traces, and treatment bytes.
7. Unit controls prove both directions: unrelated date/format transforms are
   suppressed on mention chronologies, while direct date requests, ordered
   deadlines, and matching process timelines retain their rules.

Any mismatch abandons or amends the arm before external calls and requires a
new preregistration commit.

## External Run

- Model: `deepseek-v4-flash:0731-cloud`
- One answer and one judge per arm per query
- Seed: `20260826`
- Projected calls: 800 answers and 800 judges
- Paired bootstrap: 20,000 resamples, seed `20260827`
- Synthetic benchmark data only; no real companion memories
- New output path and initially absent checkpoint; resume only after an
  interruption within this run

## Promotion Gates

All gates must pass:

1. Instruction-following delta is at least `+0.05`, with more wins than
   losses.
2. Overall delta is non-negative and its paired 95% interval lower bound is at
   least `-0.01`.
3. Pooled other-nine-category delta is at least `-0.005`.
4. Summarization delta is at least `-0.01`.
5. No category other than instruction following is below `-0.025`.
6. Event-ordering delta is non-negative.

Gate 6 is intentionally stricter than v2's harm floor. V4 exists specifically
to remove a named event-ordering interaction; a negative point estimate does
not demonstrate that fix well enough for a default-on database feature.

Every category mean, paired interval, and win/tie/loss count is reported. Any
failure leaves standing-facet composition opt-in. There is no post-hoc
threshold change, category router, query allowlist, or score-tuned exception.
