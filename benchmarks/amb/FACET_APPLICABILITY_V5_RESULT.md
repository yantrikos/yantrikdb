# AMB Standing-Facet Applicability V5 Final Power Result

## Decision

**Standing-facet composition is terminally opt-in. The composition line is
closed.**

The final mean-of-three arm passed only the instruction-following gate. It
failed the overall confidence floor, pooled other-nine floor, summarization
floor, no-category-harm gate, and event-ordering gate. Per the frozen finality
clause, there is no further power escalation or predicate variant without a
fundamentally new evidence base.

## Frozen Run

- Model: `deepseek-v4-flash:0731-cloud`
- Cohort: 400 BEAM 100k queries
- Estimator: three independent one-answer/one-judge paired runs, followed by
  per-query arithmetic means
- Run and provider-model seeds: `20260828`, `20260829`, `20260830`
- Paired-bootstrap seed: `20260831`
- Control SHA-256:
  `918f572927b75ab1bb2ae3edf5656eada132cf9a644953f31bf693c695d46863`
- Treatment SHA-256:
  `688c17fa3d2b50d4484b16b6906a2a2ae3285a7c9d1b6901cd330f364ac8e4f2`
- Manifest SHA-256:
  `cbf3471d3fbaf656097899fe4426ed70f74c931c1f5a50ed56bc86a30c9f9fc4`
- Replicate SHA-256 values:
  - `20260828`: `e139ef95d1e0b08c687a4d4159c502e26c0234677bbdab5b011c66a4460bd388`
  - `20260829`: `63c0f6831f05eb39aa4270347bac5c3dc8e025114c3d64d818d093be6b580562`
  - `20260830`: `95fc03ae6670b22532439dcadc07e91be679d7c57a6fa5188bd2deb19b6a2ad0`
- Combined result SHA-256:
  `16c18af8fda47e0de8a36f434b1ae129b585cfba0fbceacb9819216aa5031ee7`
- Analysis SHA-256:
  `58e0dbb6674488474dc99946464cdab6d6305545752a790e2d68c743e8401012`

The preregistered budget counted 2,400 answer invocations and 2,400
`score_result` invocations. That is not an exact raw HTTP-call count: BEAM's
scorer may issue one provider judge request per rubric item inside a single
`score_result` invocation. This distinction does not change scores or gates.

## Interruption Record

Replicate `20260829` was interrupted at `283/400` when the provider returned
HTTP 429 for every remaining pair after four retries. No scores were inspected,
and replicate `20260830` had not started. The incomplete output was preserved
as an audit copy and restored byte-for-byte to `.partial`; all three copies had
SHA-256
`b8594bbec8f2170ff4256a6476b9100812a04c5d6fc26deb8bfbb809a78e2668`.
After a 30-minute cooldown, the exact command resumed with unchanged run/model
seed, bootstrap seed, workers, contexts, manifest, and fingerprint
`c4679aec4d46737f88ba1d762249f5ba08f4bc7efa4d4583792ec7ad22136f20`.
It completed the missing 117 pairs without another failure. Replicate
`20260830` launched only after `20260829` reached `400/400` and another
30-minute cooldown completed.

The interruption exposed a harness bug: an incomplete cohort was written to
the final output path and its checkpoint was removed. The result change fixes
that behavior for future runs: incomplete cohorts retain `.partial` and are
never published as final outputs.

## Overall Result

| Arm | Mean rubric score |
|---|---:|
| Frozen `ydb-0151` control | 0.627788 |
| Applicable standing-facet lane | 0.630788 |
| Paired delta | **+0.003000** |

The paired bootstrap 95% interval was `[-0.011497, +0.017480]`. Treatment had
87 wins, 223 ties, and 90 losses. The point estimate was positive, but the
lower interval bound missed the frozen `-0.01` floor.

## Category Results

| Category | Control | Treatment | Delta | 95% interval | W/T/L |
|---|---:|---:|---:|---:|---:|
| abstention | 0.658333 | 0.650000 | -0.008333 | [-0.050000, +0.033333] | 3/33/4 |
| contradiction resolution | 0.850000 | 0.841667 | -0.008333 | [-0.038542, +0.022917] | 11/16/13 |
| event ordering | 0.277192 | 0.261756 | **-0.015437** | [-0.048826, +0.019676] | 13/10/17 |
| information extraction | 0.781424 | 0.764583 | -0.016840 | [-0.062674, +0.019965] | 7/24/9 |
| instruction following | 0.758333 | 0.850000 | **+0.091667** | [+0.016667, +0.170833] | 13/25/2 |
| knowledge update | 0.572917 | 0.606250 | +0.033333 | [-0.012500, +0.083333] | 5/32/3 |
| multi-session reasoning | 0.572778 | 0.531458 | **-0.041319** | [-0.080000, -0.002500] | 4/20/16 |
| preference following | 0.852083 | 0.859028 | +0.006944 | [-0.036806, +0.052083] | 7/26/7 |
| summarization | 0.577738 | 0.551473 | **-0.026265** | [-0.062014, +0.005263] | 17/6/17 |
| temporal reasoning | 0.377083 | 0.391667 | +0.014583 | [-0.022917, +0.047917] | 7/31/2 |

The pooled other-nine-category delta was `-0.006852`, below its frozen
`-0.005` floor.

## Promotion Gates

| Frozen gate | Result |
|---|---|
| Instruction delta at least +0.05 and wins exceed losses | PASS (+0.091667; 13 > 2) |
| Overall non-negative with CI lower bound at least -0.01 | **FAIL (-0.011497 floor)** |
| Pooled other-nine delta at least -0.005 | **FAIL (-0.006852)** |
| Summarization delta at least -0.01 | **FAIL (-0.026265)** |
| No non-instruction category below -0.025 | **FAIL (multi-session -0.041319)** |
| Event-ordering delta non-negative | **FAIL (-0.015437)** |

The machine-readable analyzer verdict is `promotion_passed=false` and
`finality="terminal-opt-in"`.

## Interpretation

Standing facets have a repeatable and now well-powered instruction-following
benefit. They also have cross-category costs that query-only form filtering did
not remove. The larger sample preserved the event-ordering harm, deepened the
summarization loss, and exposed statistically negative multi-session behavior.
This is not residual answer noise that warrants another repeat.

Complete and applicability-filtered facet composition remain available only as
explicit opt-in context. They must not enter the default recall path. The next
benchmark work is a row-level event-ordering autopsy and a fundamentally
different coverage-first retrieval mechanism, not another composition arm. The
completed diagnosis and query-route taxonomy are recorded in
`EVENT_ORDERING_V5_AUTOPSY.md`.
