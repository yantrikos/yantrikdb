# AMB Standing-Facet Applicability V4 Result

## Decision

**Do not promote applicability-filtered standing-facet composition to the
default recall path.**

The arm passed four of the six frozen promotion gates. It missed the
summarization floor and the deliberately strict event-ordering gate:

- summarization moved `-0.014152`, below the frozen `-0.01` floor;
- event ordering moved `-0.015694`, below the required non-negative delta.

The thresholds are not waived after observing the result. Complete standing
facets and the v4 applicability filter remain opt-in.

## Frozen Run

- Model: `deepseek-v4-flash:0731-cloud`
- Cohort: 400 BEAM 100k queries
- Repeats: one answer and one judge per arm per query
- Answer and arm-order seed: `20260826`
- Paired-bootstrap seed: `20260827`
- Paired manifest SHA-256:
  `cbf3471d3fbaf656097899fe4426ed70f74c931c1f5a50ed56bc86a30c9f9fc4`
- Control SHA-256:
  `918f572927b75ab1bb2ae3edf5656eada132cf9a644953f31bf693c695d46863`
- Treatment SHA-256:
  `688c17fa3d2b50d4484b16b6906a2a2ae3285a7c9d1b6901cd330f364ac8e4f2`
- Evaluation SHA-256:
  `5c98f64e5f913fb297d086b3e592e481b89ffb0cb7845dc99711c75dad8cdee6`
- Analysis SHA-256:
  `a54d27afeaed4530772a437d2caf5728137cb14b931468432d8666c4d2faa7d7`
- Elapsed time: 9,697 seconds

The preregistration, product preflight, artifact lineage, and separate-seed
harness fix were committed, independently countersigned, and merged before
external scoring. The run completed all 400 pairs without a resume or failed
pair.

## Overall Result

| Arm | Mean rubric score |
|---|---:|
| Frozen `ydb-0151` control | 0.630107 |
| Applicable standing-facet lane | 0.644362 |
| Paired delta | **+0.014255** |

The paired bootstrap 95% interval was `[-0.008603, +0.037275]`. Treatment had
63 wins, 283 ties, and 54 losses. This passes the frozen overall gate because
the point estimate is non-negative and the lower interval bound is at least
`-0.01`.

## Category Results

| Category | Control | Treatment | Delta | 95% interval | W/T/L |
|---|---:|---:|---:|---:|---:|
| abstention | 0.625000 | 0.650000 | +0.025000 | [-0.100000, +0.150000] | 4/33/3 |
| contradiction resolution | 0.837500 | 0.890625 | +0.053125 | [+0.006250, +0.103125] | 11/25/4 |
| event ordering | 0.304375 | 0.288681 | **-0.015694** | [-0.079147, +0.043958] | 13/16/11 |
| information extraction | 0.805208 | 0.792188 | -0.013021 | [-0.044271, +0.019792] | 3/29/8 |
| instruction following | 0.787500 | 0.862500 | **+0.075000** | [0.000000, +0.162500] | 6/33/1 |
| knowledge update | 0.556250 | 0.593750 | +0.037500 | [-0.025000, +0.125000] | 2/37/1 |
| multi-session reasoning | 0.528750 | 0.542292 | +0.013542 | [-0.061458, +0.083333] | 6/29/5 |
| preference following | 0.883333 | 0.870833 | -0.012500 | [-0.062500, +0.037500] | 2/35/3 |
| summarization | 0.573155 | 0.559003 | **-0.014152** | [-0.051652, +0.021875] | 12/14/14 |
| temporal reasoning | 0.400000 | 0.393750 | -0.006250 | [-0.093750, +0.081250] | 4/32/4 |

The pooled other-nine-category delta was `+0.007506`, passing its `-0.005`
floor.

## Promotion Gates

| Frozen gate | Result |
|---|---|
| Instruction delta at least +0.05 and wins exceed losses | PASS (+0.075000; 6 > 1) |
| Overall non-negative with CI lower bound at least -0.01 | PASS (+0.014255; -0.008603) |
| Pooled other-nine delta at least -0.005 | PASS (+0.007506) |
| Summarization delta at least -0.01 | **FAIL (-0.014152)** |
| No non-instruction category below -0.025 | PASS (minimum -0.015694) |
| Event-ordering delta non-negative | **FAIL (-0.015694)** |

## Interpretation

The product-backed facet lane continues to show a meaningful instruction lift,
and the query-only form-conflict predicate preserves all canonical instruction
targets. That is useful evidence for an explicit, opt-in preference context.
It is not evidence that broad standing-facet composition is safe as a database
default.

Event ordering again had more wins than losses, but the losses were larger.
The v4 predicate removed the specifically identified date/format collisions
without eliminating the category-level harm. Summarization independently
crossed its frozen floor. A category router, query allowlist, relaxed threshold,
or score-tuned predicate would be post-hoc and is not authorized.

Any next default-on attempt needs a genuinely different product mechanism and
a new preregistration. The current complete and applicability-filtered facet
lanes remain opt-in.
