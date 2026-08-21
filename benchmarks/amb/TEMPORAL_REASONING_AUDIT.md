# AMB Temporal-Reasoning Audit

## Scope

This audit asks whether YantrikDB's `0.41` temporal-reasoning mean in the
`ydb-0151` BEAM-100K run is primarily caused by missing endpoint evidence,
context dilution, answer arithmetic, or evaluator defects. All model-scored
controls use frozen synthetic benchmark contexts and
`deepseek-v4-flash:0731-cloud` as both answerer and judge.

## Endpoint Retrieval

`temporal_decomposition_probe.py` splits explicit "between A and B" and
"after A did B" questions into two independent retrieval lanes. The original
base-provider probe mixed assistant prose with user events. Constraining both
lanes to trustworthy turn-level `source=user` provenance changed:

- correct ordered endpoint-date pairs: `22/33` to `24/33`;
- gold-value coverage: `106/135` to `107/135`;
- rows gaining any missing gold value: `0` to `1`.

The baseline contexts already contain most decisive values. More endpoint
recall is therefore not the primary category bottleneck.

## Gold Arithmetic

`audit_temporal_gold.py` checks every gold response containing an explicit
day/week quantity and at least two stated calendar dates. Of 40 temporal rows,
31 are mechanically auditable and three gold calculations are inconsistent:

| Query | Gold claim | Calendar interval | Baseline answer |
| --- | ---: | ---: | ---: |
| `1_temporal_reasoning_0` | 4 weeks / 28 days | Jan 15 to Mar 15 = 60 days | 13 weeks |
| `14_temporal_reasoning_0` | 11 days | Mar 20 to Apr 6 = 17 days | 18 days |
| `17_temporal_reasoning_1` | 46 days | Apr 20 to Jul 5 = 76 days | 76 days |

The last row is a mathematically exact answer scored `0`. The audit treats
date ranges such as April 20-21 as ending on the second day and permits gold
prose to mention the later endpoint before the earlier one.

## Frozen Paired Controls

Both arms contain the same 33 splittable query IDs. Endpoint contexts contain
at most two user-provenance hits per lane and no gold text.

### Endpoint Only

- Manifest SHA-256:
  `94a68214c7d4ea46d2607a8c3faed5175cf4d26f6b8592581b48f5c380717d8f`
- Baseline context: `423,120` tokens
- Endpoint context: `13,586` tokens, a 96.8% reduction
- Mean: `0.364` baseline vs `0.303` endpoint-only
- Delta: `-0.061`, paired bootstrap 95% CI `[-0.167, 0.045]`
- Outcomes: 3 endpoint wins, 22 ties, 8 baseline wins

Endpoint-only context is rejected. It fixes several operand-selection failures
but removes supporting evidence and causes new abstentions.

### Endpoint Lanes Prepended

- Manifest SHA-256:
  `c13a8942eba9369527ffe8c58a4b68b030798ffc2321965f4ac9cb5928883ffe`
- Baseline context: `423,120` tokens
- Hybrid context: `436,932` tokens
- Mean: `0.341` baseline vs `0.379` hybrid
- Delta: `+0.038`, paired bootstrap 95% CI `[-0.061, 0.136]`
- Outcomes: 5 hybrid wins, 24 ties, 4 baseline wins

The hybrid is also rejected for production. Its interval includes zero, and
four regressions arise when a semantically close but wrong revised deadline is
made salient. For example, the patent history contains both an initial June 1
filing target and a later May 15 target. Other queries prefer an earlier target,
so neither "latest wins" nor "earliest wins" is a valid general rule.

## Decision

Do not add unconditional endpoint-only retrieval, endpoint prepending, or a
date-revision heuristic to YantrikDB. The tested category gap combines noisy
long-context operand selection, ambiguous revised plans, judge variance, and
at least three invalid gold calculations. A future temporal relation feature
must carry explicit event identity and revision provenance at write time, then
be tested on a corrected evaluator with repeated judging. It must not infer a
single canonical endpoint from similarity rank alone.
