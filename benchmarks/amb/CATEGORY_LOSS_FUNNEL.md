# BEAM Category Loss Funnel

## Scope

This judge-free audit examines all eight non-temporal categories in the frozen
`ydb-0151` BEAM-100K run. It measures distinctive reference-token
support in the source conversation, retention in retrieved context, retention
in the final answer, exact knowledge-update value chronology, and answer tokens
supported only by explicitly assistant-authored memory blocks.

The input hashes are:

- Results: `43b8eb888adeb524caba70b013a8ca510c639bbf5312216e9b453d60185bcb8e`
- Source documents: `fc0e64bac38fcde26eece776e818f70374338d4591ecc75346cb27b613d4c128`

No model or judge calls are made. Lexical coverage is a diagnostic proxy, not
a replacement benchmark score.

## Results

| Category | Mean score | Overall points lost | Context retention | Answer retention | Primary attribution |
|---|---:|---:|---:|---:|---|
| summarization | 0.593 | 4.07 | 0.869 | 0.555 | reader compression |
| knowledge update | 0.631 | 3.69 | 0.988 | 0.608 | benchmark labels |
| multi-session reasoning | 0.611 | 3.89 | 0.908 | 0.427 | reader set assembly |
| abstention | 0.675 | 3.25 | n/a | n/a | provenance rendering |
| contradiction resolution | 0.828 | 1.72 | 0.978 | 0.733 | reader conflict resolution |
| information extraction | 0.773 | 2.27 | 0.948 | 0.697 | reader fact selection |
| instruction following | 0.788 | 2.13 | 0.794 | 0.259 | standing-instruction salience |
| preference following | 0.913 | 0.88 | 0.824 | 0.378 | preference salience |

`Overall points lost` assumes BEAM's ten equally weighted 40-query categories:
`10 * (1 - category mean)`. It makes the rows additive with the full-line loss
ledger without pretending that every lost point is product-recoverable.

Summarization has 22 retrieval-loss items, 131 answer-loss items, 41 covered
items, and one source/label mismatch. Multi-session reasoning has 30
answer-loss items, eight covered items, and two derived-count items whose gold
form is not stated directly in source. These categories are primarily reader
selection and synthesis problems at the current context budget, not evidence
reachability failures.

For the 14 zero-score knowledge-update rows, six gold values precede a later
distinct value returned by the answerer, five gold values have no exact
user-turn match, and only three remain ordinary review cases. Product
supersession policy must not be tuned to the eleven benchmark-integrity cases.

Across all abstention rows, 58.2% of explicitly speaker-supported factual answer
tokens occur only in assistant-authored blocks. Seven of the thirteen zeroes
meet the stricter assistant-only-dominance rule. The retrieved contexts already
label these blocks as assistant suggestions, but the reader often promotes them
to user facts.

For contradiction resolution, only the two conflicting evidence claims are
treated as retrieval targets; the rubric's conflict-acknowledgement and verdict
directives are reader requirements. Of 80 evidence claims, 46 are covered, 32
are lost in the answer, one is lost in retrieval, and one is a source mismatch.
Information extraction is similarly reader-heavy: only 4/92 claims are lost in
retrieval, versus 41 answer losses and 45 covered claims.

Instruction and preference rows expose their canonical standing behavior in
`instruction_being_tested` and `preference_being_tested`. The exact instruction
is source-backed in all 40 rows and retrieved in 36; the exact preference is
retrieved in 36/40. Partitioning actual row-score deficits assigns `3.5/8.5`
instruction deficit units and `1.0/3.5` preference deficit units to retrieval;
the remaining `5.0` and `2.5` units occur after the target is already present.

## Ceiling And Recovery Budget

The audit now emits a loss-conserving `ceiling_estimate` over all ten frozen
categories. The `ydb-0151` baseline is `65.1485`, so the line loses `34.8515`
points and needs `24.8515` points, not 28.9, to reach 90.

`ydb-0151` is the canonical anchor because every row-level attribution in this
audit comes from that exact result (SHA-256 `43b8eb888adeb524caba70b013a8ca510c639bbf5312216e9b453d60185bcb8e`).
The separate `ydb-final` run scored `61.0689` (SHA-256
`048b38bd9a285ab788677d60815ec6533268a3fddf0a4d31e60fbf53a5b8b601`).
That `4.0796`-point cross-run gap is replication evidence, not part of the
anchor arithmetic. A future claim of reaching 90 means 90 on the declared
anchor configuration and should use a repeated full-line median.

| Mutually exclusive bucket | Points |
|---|---:|
| Dead or benchmark integrity | 3.092 |
| Reader, potentially reachable through context shaping | 10.725 |
| Direct engine mechanisms | 16.422 |
| Undiagnosed category tail | 0.000 |
| Residual inside audited categories | 4.612 |
| **Total frozen loss** | **34.851** |

The conservation delta is exactly zero. The optimistic ceiling is `96.9081`:
it removes only the 11/14 knowledge-update zero-row label share and the 2/40
multi-session answers whose derived gold form is unstated. Reaching 90 requires
recovering `78.25%` of the remaining `31.7596` potentially recoverable points.
It therefore requires broad success, but not literally every non-dead point.

The conversion rules are explicit in the JSON. Reader points use audited shares
in all eight categories. Instruction and preference shares use actual row-score
deficits; fact-selection and conflict shares use reference-item counts. Direct
engine points include all event-ordering and temporal loss plus every audited
retrieval or provenance share. Treating all temporal loss as directly
product-addressable is an optimistic recovery assumption, not a causal finding;
the temporal audit found reader arithmetic, revision ambiguity, and invalid
gold calculations as well. The residual bucket prevents uncertain and
overlapping cases from being silently assigned twice.

The machine-readable sensitivity table varies only reader-shaping recovery and
optimistically recovers every other non-dead bucket:

| Reader-attributed loss recovered | Projected score |
|---:|---:|
| 0% | 86.183 |
| 50% | 91.545 |
| 70% | 93.691 |
| 100% | 96.908 |

This is not a forecast: 50% and 70% are scenario parameters, while the evidence
currently includes one context-shaping win and one abstention arm that missed
its gate. The Q7 A/A calibration also observed about `0.02` answer-generation
standard error within a 40-query category. Threshold claims therefore require
replication even when a single full run crosses 90.

### Direct-engine bucket decomposition

The `16.422` direct-engine label is substantially more optimistic than its
name suggests. The machine-readable audit now decomposes it:

| Component | Points | Share of bucket | Basis |
|---|---:|---:|---|
| event ordering | 7.093 | 43.2% | whole category, optimistic |
| temporal reasoning | 5.875 | 35.8% | whole category, optimistic |
| abstention provenance | 1.750 | 10.7% | audited zero-row proxy |
| instruction salience | 0.875 | 5.3% | audited score-deficit share |
| summarization retrieval | 0.459 | 2.8% | audited item-retention proxy |
| preference salience | 0.250 | 1.5% | audited score-deficit share |
| information-extraction retrieval | 0.099 | 0.6% | audited item-retention proxy |
| contradiction retrieval | 0.021 | 0.1% | audited item-retention proxy |

Thus `12.968` points, or `79.0%` of the bucket, are not row-level engine
attributions at all. They are the complete loss of two weak categories placed
in the engine bucket as an optimistic ceiling assumption. This distinction is
important when choosing work: a category being temporal does not prove an
event-time index can recover its loss.

The existing experiments narrow those two wholesale assignments further:

- Event ordering's best complete concern-selected synthesis result is
  `0.33969` versus the frozen `0.29071`, a measured recovery of `0.490` full
  benchmark points, or only `6.9%` of the category's assigned loss. It remains
  opt-in because 13/40 rows regressed and required query-dependent item
  granularity; fixed write-time coarse synthesis is not valid.
- Temporal reasoning exposes an explicit calendar date in only 1/40 queries,
  for at most `0.250` full benchmark points under a perfect direct filter.
  Three mechanically invalid gold calculations account for `0.750` points,
  while endpoint-only retrieval regressed and endpoint-prepending was
  inconclusive. The remainder is primarily operand selection, arithmetic,
  and revision ambiguity, not yet an engine attribution.
- The standing-instruction facet recovered `0.750` of its `0.875` attributed
  points on the target slice. Its global-default failure came from composition
  cost, now measured separately as whole-block displacement.
- Equal-budget role-aware abstention recovered `0.500` points in its discovery
  run but missed the preregistered lift gate and showed answer variance.

Knowledge update and multi-session reasoning contribute exactly zero to the
current direct-engine bucket. Their large losses are assigned to benchmark
integrity, reader assembly, and residual shares. They may motivate product
features, but they must not be cited as explaining the `16.422` direct-engine
number without a new row-level attribution.

## Decision

Do not spend the next product cycle increasing generic top-k for summarization,
multi-session reasoning, or knowledge update. Their context reachability is
already high, and the knowledge-update zero cohort is mostly unsuitable for
product tuning.

The next product-shaped experiment is an equal-budget abstention comparison:
current mixed-speaker context versus role-aware user-authoritative context on
all 40 abstention queries. The treatment must not receive more context tokens.
Because a complete 40-query cohort averages answer variance, use one answer per
arm for the discovery pass; require at least `+0.075` mean lift and more wins
than losses before a repeated-answer confirmation. A failed or merely neutral
arm leaves mixed-speaker retrieval unchanged.

## Equal-Budget Abstention Result

The discovery arm completed all 40 pairs under manifest SHA-256
`394d074ee3b6316409c90196d14d224fd404fc8b81346f75123b2190a65844f1`.
The role-aware treatment used `553,969` context tokens versus `583,816` for the
baseline. Each arm used one answer and one judge call per query with
`deepseek-v4-flash:0731-cloud`.

| Context arm | Mean rubric |
|---|---:|
| frozen `ydb-0151` | 0.650 |
| equal-budget role-aware | 0.700 |

The paired delta was `+0.050`, with four treatment wins, 34 ties, and two
baseline wins. The paired bootstrap 95% interval was `[-0.075, +0.175]`.
This misses the preregistered `+0.075` lift gate, so no repeated-answer
confirmation or product-default change is authorized.

The six discordant rows also reinforce the answer-variance finding. Two rows
that scored `1.0` in the original frozen line scored `0.0` when the identical
baseline context was re-answered, while one original zero scored `1.0`.
Role-aware retrieval fixed three abstention failures in units 4 and 5 but
introduced failures in units 1 and 9. The split is not identifiable from query
wording alone and must not become a post-hoc routing rule.

## Reproduction

```powershell
python benchmarks/amb/audit_category_loss_funnel.py `
  C:/path/to/ydb-0151/rag/100k.json `
  C:/path/to/beam/100k/documents.json.gz `
  --out C:/path/to/category-loss-funnel.json
```
