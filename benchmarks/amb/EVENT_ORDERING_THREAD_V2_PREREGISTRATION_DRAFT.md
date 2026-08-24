# AMB Event-Ordering Thread v2 Preregistration Draft

Status: **draft only**. The mechanism, product artifact, hashes, commands, and
gates are not frozen until `recall_thread` v2 and its query-only selector exist,
the zero-call preflight passes, and this document is committed before any
external answer or judge call.

## Decision Question

Can a query-only, multi-anchor chronological thread retrieve the cross-session
evidence needed for BEAM event ordering and convert that evidence into a
material score lift without optimizing toward known benchmark defects?

This is a fundamentally different retrieval mechanism from the terminal
standing-facet composition line. The control is the frozen ordinary
`ydb-0151` context. The treatment changes evidence selection and hydration; it
does not add standing instructions or use gold source IDs, rubrics, answers, or
scores during retrieval.

## Evidence Base

The frozen v5 autopsy supplies the design and stopping evidence:

- 40 event-ordering queries; 37 clean primary rows and three quarantined rows.
- Clean-row control source coverage: `19.09%` macro and `36/214` (`16.82%`)
  micro, with one exact query and 21 nonzero queries.
- Clean-row oracle top-three-topic ceiling: `91.64%` macro and `192/214`
  (`89.72%`) micro, with 25 exact queries and all 37 nonzero.
- All 17 negative v5 deltas are source-incomplete.
- The sole source-complete row scored `0.7333`, so full coverage is not a
  perfect-answer assumption.
- Frozen control score: `0.279247` on the clean 37 and `0.277192` on all 40.

A planning ceiling multiplies clean-row oracle macro coverage (`0.9164`) by
the observed full-source reader result (`0.7333`), yielding `0.6720`. Holding
the three quarantined rows at their observed `0.3000` treatment mean yields an
all-40 ceiling of about `0.6441`, or roughly `+0.367` over the frozen control.
This is an optimistic mechanism ceiling, not the expected effect or a gate:
the reader residual is `n=1`, from one query, and query-only topic selection
will not attain oracle handle selection. Neither `0.6720` nor `0.6441` is a
measured score ceiling.

## Cohort And Estimands

The scorer runs all 40 event-ordering rows from the frozen BEAM 100k cohort.
No query is removed from execution or reporting.

The product artifact still contains all 400 frozen rows. The treatment's
deterministic chronology-query predicate may change only the 40 event-ordering
rows; all 360 non-event rows must remain byte-identical to control. Event-only
scoring is acceptable because cross-category safety is proved by construction,
not because the other categories go unmeasured. The same predicate and code
path used to build the artifact are the only behavior eligible for promotion.

The primary estimand is the paired treatment-minus-control delta over the 37
rows not known before this arm to have defective or underidentified labels.
The three quarantined rows remain a separately reported diagnostic stratum:

- `9_event_ordering_0`: correction conflict; the gold requires stale Bryan
  history.
- `18_event_ordering_0`: unstated Patrick merge/split granularity.
- `19_event_ordering_0`: unstated later Douglas partition.

Secondary estimands are the all-40 event-ordering delta and the clean bounded-
phrase and broad-compound route deltas. Exact-entity and entity-plus-focus
strata contain one query each and are descriptive only.

## Treatment

`recall_thread` v2 receives a query-only `ThreadQuery`:

- `entities`: exact names resolved through the persisted normalized entity
  key.
- `phrases`: bounded phrases matched through `memories_fts`.
- `topic_rids`: already-resolved organizer synthesis RIDs. Direct members come
  from `synthesis_dependencies` with `is_direct=1` and the requested namespace.

The engine unions and deduplicates all route RIDs, applies active/synthesis/
supersession visibility in SQL, assigns full-thread positions by
`(created_at, source_turn, rid)`, applies the SQL limit, and decrypts only the
returned page. Every item reports route provenance. The response reports
`total`, `returned`, and `omitted`.

Exact bounded ordering requires `source_turn` to be a persisted nullable
`memories` column, stamped from plaintext metadata on every write and
replication path. The v2 artifact must use that column; decrypting metadata for
every eligible row before ordering would violate the bounded-work contract.
Unencrypted migrations backfill the column before serving v2. Encrypted
migrations set `source_turn_backfill_complete=false`; strict v2 returns a typed
`MaintenanceRequired` error until the resumable, keyset-paged decrypt-and-stamp
maintenance operation completes and sets the marker. Opportunistic write-time
repair does not waive that marker.

The caller selects no more than three organizer topics from the query and
persisted query-independent handle representations. Gold source turns, rubric
text, answers, and prior scores are unavailable to selection. `AI` is a domain
phrase, not a named entity. Topic labels or metadata must not be found by a
global decrypting scan; the caller passes resolved synthesis RIDs.

The treatment context renders the complete returned union chronologically with
stable position markers. The benchmark request's exact item count and answer-
only form remain ordinary query instructions; no gold count is introduced.

The fixed benchmark limit is `100`. This is not a product claim that all user
threads fit in 100 rows. Any query with `omitted > 0` aborts this arm before
external calls.

The FTS phrase route is plaintext-store-only. An encrypted engine receiving a
nonempty phrase route must return a typed capability error naming that route,
never an empty result that looks like no evidence. Entity and resolved-topic
routes remain available under encryption. The benchmark artifact must record
its actual encryption mode and route capabilities; this arm is expected to use
the same plaintext mode as its frozen control and makes no searchable-
encryption claim.

## Stage A: Zero-Call Product Gate

The complete 400-row artifact is built through the public product path. The
preflight makes no answer, judge, or other external model call. All conditions
must pass:

The treatment and its hash are finalized before authoritative source turns are
joined for membership scoring. Frozen route labels are evaluator strata only;
neither they nor their thresholds are inputs to the selector.

1. The exact 400 frozen query IDs are present once in control and treatment.
2. Control contexts are byte-identical to frozen `ydb-0151`; all 360 non-event
   treatment contexts are byte-identical to their controls, and exactly the 40
   event-ordering rows enter the deterministic chronology-query path.
3. Every event-ordering treatment row records its selected entities, phrases, topic RIDs,
   route provenance, `total`, `returned`, and `omitted`.
4. Every treatment row has `omitted=0`, `returned=total`, continuous unique
   positions, and chronological `(created_at, source_turn, rid)` order.
5. Namespace and visibility behavior matches default host recall; correction
   and supersession semantics are unchanged.
6. Selection uses only the query and persisted product records. No evaluation
   source ID, rubric, gold answer, v5 score, or frozen route label enters the
   selector.
7. At most three topic RIDs are selected per query; every selected topic RID
   resolves to an active, verified organizer synthesis in the same namespace.
8. On the 37 clean rows, authoritative source-turn recall is at least `0.55`
   macro and `0.55` micro, at least `30/37` rows have nonzero coverage, and at
   least `8/37` have exact source coverage.
9. Bounded-focus-phrase micro recall is at least `0.55` over its frozen 22
   rows; broad-compound-topic micro recall is at least `0.55` over its frozen
   13 rows.
10. The exact-entity Carla row contains `6/6` source turns. The Douglas
    entity-plus-focus row contains at least `4/9` source turns.
11. The artifact records plaintext mode with phrase capability available. A
    separate product test proves that encrypted phrase queries return the typed
    capability error while encrypted entity-only and topic-only queries remain
    functional.
12. The artifact records `source_turn_backfill_complete=true`. Product tests
    prove an encrypted migrated store returns `MaintenanceRequired` before the
    named maintenance operation and exact positions after it; an unencrypted
    migration completes the marker before v2 can be queried.

The Stage A thresholds require more than half of the structurally available
coverage while leaving room for the unavoidable query-only selection gap. They
raise both clean macro and micro coverage by more than three times their frozen
baselines. If any condition fails, the arm aborts with zero external calls. No
threshold is relaxed and no failed row is removed.

## Stage B: Paired Score Gate

Stage B is filled and frozen only after Stage A passes and the following are
recorded in this document:

- control, treatment, manifest, ordered-query-ID, membership-report, and source
  dataset SHA-256 values;
- exact product and benchmark commits;
- exact commands, answer model, judge model, worker count, seeds, output paths,
  call budget, resume policy, and analyzer version;
- a no-client preflight proving those bindings without an external call.

The intended estimator is three independent paired replicates with one answer
and one judge result per arm and query in each replicate. Per-query arm scores
are arithmetic means across replicates. A separate frozen seed drives 20,000
paired query-level bootstrap resamples. The existing median-answer selection
path is prohibited.

Provisional run/model seeds are `20260901`, `20260902`, and `20260903`; the
provisional bootstrap seed is `20260904`. They remain placeholders until the
artifact hashes and commands are frozen.

The projected event-only budget is 240 answer invocations and 240
`score_result` invocations: `40 queries * 2 arms * 3 replicates` for each. This
is not an exact provider HTTP-request count because BEAM may issue one judge
request per rubric item inside a `score_result` invocation.

All promotion gates must pass:

1. The clean 37-row mean delta is at least `+0.08`, its paired 95% interval
   lower bound is non-negative, and wins exceed losses.
2. The all-40 event-ordering mean delta is at least `+0.05`.
3. The 22-row bounded-focus-phrase mean delta is non-negative.
4. The 13-row broad-compound-topic mean delta is non-negative.
5. The three quarantined rows are reported but do not enter gates 1, 3, or 4.
   Their answers and contexts are inspected only for correction or stale-fact
   regressions, never to tune selector thresholds.
6. The score-bearing treatment artifact has the same `omitted=0`, provenance,
   ordering, visibility, and hash properties as the Stage A artifact.

The `+0.08` clean-row floor is below the planning ceiling by a wide margin but
large enough to justify a new default event-ordering route. At the Stage A
minimum coverage of `0.55`, applying the observed `0.7333` reader discount
implies about `0.403`, approximately `+0.124` above the frozen clean control;
the score gate retains roughly two thirds of that conservative planning lift.

## Decision And Finality

Pass Stage A and all Stage B gates: enable the multi-anchor chronological route
by default only for queries accepted by the exact deterministic chronology-
query predicate used in the frozen artifact, with the same selector, bounded-
work, visibility, and omission behavior. No broader recall path is promoted.

Fail Stage A: no external run; revise query-only topic selection only with new
row-level evidence. Fail Stage B: keep the route opt-in and stop power
escalation. No post-hoc route allowlist, query removal, threshold change,
replicate exclusion, seed replacement, or score-driven topic mapping is
allowed. A Stage B failure is terminal for this exact mechanism: no identical
rerun or power escalation is allowed. A future attempt requires a materially
changed mechanism, new evidence, and a fresh preregistration.
