# AMB Event-Time Filter Opportunity

## Question

How much of the frozen 400-row `ydb-0151` result can first-class event-time
filtering reach directly, before adding endpoint resolution or another model?

The cohort is selected from query text only. Gold and scores are consulted
after selection solely to compute a perfect-arm upper bound. That bound is not
an expected lift and does not justify an external run by itself.

## Frozen Query-Only Cohorts

| scope | rows | imperfect rows | perfect-arm full-400 ceiling |
|---|---:|---:|---:|
| any explicit calendar date | 19 | 5 | +0.0125 |
| unambiguous filter semantics | 18 | 4 | +0.0100 |
| closed exact-day or bounded window | 13 | 2 | +0.0050 |

The one excluded ambiguous query asks for advice "beyond the April 15
deadline." The date modifies the deadline, but does not safely imply that the
evidence itself occurred after April 15.

Only one of the 40 `temporal_reasoning` queries contains a calendar date in
the query. It supplies a one-sided starting date, so even a perfect direct
filter can add at most `+0.025` to that category. No `event_ordering` query
contains an explicit calendar date. Direct event-time filtering is therefore
not the missing high-headroom AMB mechanism.

## Product Interpretation

Issue #149 remains valuable database work. It makes valid time independently
queryable, supports precise application workflows, and provides the indexed
primitive a future two-stage endpoint resolver needs. Its benchmark role must
be stated narrowly:

1. A precision-first explicit-date arm can test exact-day and closed-window
   filtering without gold-derived bounds.
2. One-sided `before` and `after` filters should be a separate cohort because
   they retain much larger candidate universes.
3. Queries that name events but no dates require endpoint resolution first;
   counting them as directly reachable would overstate #149's gain.
4. Any model-scored arm needs a fresh preregistration and repeated judging.

The reproducible query parser and ceiling calculation are in
`benchmarks/amb/audit_event_time_filter_opportunity.py`.
