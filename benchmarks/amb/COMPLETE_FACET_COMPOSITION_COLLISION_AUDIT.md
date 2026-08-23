# AMB Complete Facet Composition Collision Audit

## Decision

**Do not run the byte-identical v3 replication.**

Four of the ten v2 event-ordering loss rows have a plausible facet collision.
The frozen funding gate prohibited another identical run when three or more
rows qualified. Complete-lane composition therefore remains opt-in, and the
v2 overall lift of `+0.028727` is not banked for default promotion.

This result names a plausible mechanism for the v2 event-ordering decline:
conditionally applicable standing instructions can alter an unrelated answer
when every verified facet is composed without query-scope compatibility.

## Frozen Inputs

- Evaluation SHA-256:
  `27ab6cacb1370b631f086ab70c371e4d6b49ac9c2f859e15cadce0c22c63bea5`
- Event-ordering losses inspected: 10
- External calls made for this audit: 0
- Protocol: `COMPLETE_FACET_COMPOSITION_REPLICATION_PREREGISTRATION.md`

The loss rows were not inspected until the funding protocol was committed,
independently countersigned, and merged in PR #177.

## Frozen Criterion

A row has a plausible collision when a composed directive governs dates,
timelines, scheduling, deadlines, event chronology, or ordered output and the
control-to-treatment answer change could reasonably be affected by that rule.

## Row Audit

| Query ID | Collision | Basis |
|---|---|---|
| `3_event_ordering_1` | No | The panel governs markup, debugging, and performance metrics. It has no qualifying temporal or ordered-output rule. |
| `6_event_ordering_0` | **Yes** | Same-subject resume rules require structured, quantified bullets. Treatment promoted that formatting rule into an answer item. |
| `9_event_ordering_1` | **Yes** | The panel is entirely date, time, deadline, and time-zone rules. A dated control answer became an abstention. |
| `10_event_ordering_1` | **Yes** | Timeline and date rules directly changed treatment into a fully date-stamped list. |
| `11_event_ordering_0` | No | A detailed-timeline rule applies to project schedules, not to an ordered list of hiring concepts; the other rules are unrelated. |
| `12_event_ordering_1` | No | The date rule applies to scheduled events, not to undated philosophical reflection. Query-side chronology alone does not satisfy the directive-side criterion. |
| `16_event_ordering_0` | No | The panel contains budgeting and investment-content rules, with no qualifying temporal or ordered-output rule. |
| `16_event_ordering_1` | No | The same financial-content panel contains no qualifying temporal or ordered-output rule. |
| `18_event_ordering_1` | No | Scheduling-date rules do not govern a chronology of personal and work challenges. The topical mental-health rule is outside the frozen criterion. |
| `20_event_ordering_1` | **Yes** | A detailed-timeline rule for patent application processes directly matches the patent-process ordering request. |

Total: **4 plausible collisions, 6 non-collisions**.

## Consequence

The preregistered `4/10 >= 3/10` stop condition is met. No identical v3 run is
authorized, and there will be no third attempt at complete-lane replication.
A future scored arm must change the mechanism and receive a fresh
preregistration.

The next mechanism must use a general, auditable rule-type versus answer-type
compatibility predicate. The ten loss rows may motivate the design, but no
threshold or query list may be tuned to them. Near-domain controls must also
prove that date and timeline rules remain available when the request actually
calls for dates, deadlines, schedules, meetings, or process timelines.
