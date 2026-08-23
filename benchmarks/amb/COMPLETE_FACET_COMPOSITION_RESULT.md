# AMB Complete Facet Composition V2 Result

## Decision

**Do not promote complete-lane facet composition to the default recall path.**

The arm produced a statistically positive overall paired lift and passed four
of the five frozen promotion gates. It missed the final per-category harm
floor: event ordering moved `-0.030486`, below the preregistered `-0.025`
minimum. The floor is not waived after observing the result. Complete-lane
composition remains opt-in.

The miss is `0.005486` beyond the floor. It is not rounded away or waived:
the preregistered gate is binary by design.

The preregistered zero-call follow-up found plausible facet collisions in four
of the ten event-ordering loss rows. That exceeded the frozen stop threshold,
so the byte-identical v3 replication was not run. See
`COMPLETE_FACET_COMPOSITION_COLLISION_AUDIT.md` for the row-level evidence.

## Frozen Run

- Model: `deepseek-v4-flash:0731-cloud`
- Cohort: 400 BEAM 100k queries
- Repeats: one answer and one judge per arm per query
- Paired manifest SHA-256:
  `b1b3b009b4f5d531d58edb80f157d46f02d80169535f190e1d603806cb673989`
- Control SHA-256:
  `918f572927b75ab1bb2ae3edf5656eada132cf9a644953f31bf693c695d46863`
- Treatment SHA-256:
  `43a222af04195cc2d18d3a2ad15124a73208afeb0df352c956e69e86c5c5d9fb`
- Evaluation SHA-256:
  `27ab6cacb1370b631f086ab70c371e4d6b49ac9c2f859e15cadce0c22c63bea5`
- Elapsed time: 8,535 seconds

The preregistration and product preflight were committed and merged before
scored results were inspected.

## Overall Result

| Arm | Mean rubric score |
|---|---:|
| Frozen `ydb-0151` control | 0.619626 |
| Complete standing-facet lane | 0.648353 |
| Paired delta | **+0.028727** |

The paired bootstrap 95% interval was `[+0.006320, +0.051793]`. Treatment had
68 wins, 295 ties, and 37 losses.

The control mean is the fresh paired re-answer, not the historical frozen
answer score. The paired delta is the valid causal comparison because both
arms were answered and judged together under the same model and run.

## Category Results

| Category | Control | Treatment | Delta | 95% interval | W/T/L |
|---|---:|---:|---:|---:|---:|
| abstention | 0.625000 | 0.700000 | +0.075000 | [-0.050000, +0.200000] | 5/33/2 |
| contradiction resolution | 0.856250 | 0.856250 | +0.000000 | [-0.037500, +0.040625] | 7/26/7 |
| event ordering | 0.278006 | 0.247520 | **-0.030486** | [-0.094375, +0.030764] | 11/19/10 |
| information extraction | 0.753646 | 0.805729 | +0.052083 | [+0.010417, +0.114583] | 8/31/1 |
| instruction following | 0.768750 | 0.862500 | **+0.093750** | [+0.012500, +0.193750] | 6/33/1 |
| knowledge update | 0.568750 | 0.593750 | +0.025000 | [-0.050000, +0.100000] | 2/37/1 |
| multi-session reasoning | 0.542292 | 0.572917 | +0.030625 | [-0.031875, +0.098750] | 7/30/3 |
| preference following | 0.875000 | 0.889583 | +0.014583 | [-0.006250, +0.037500] | 3/36/1 |
| summarization | 0.541071 | 0.567783 | +0.026711 | [-0.003333, +0.056592] | 16/16/8 |
| temporal reasoning | 0.387500 | 0.387500 | +0.000000 | [-0.062500, +0.062500] | 3/34/3 |

## Promotion Gates

| Frozen gate | Result |
|---|---|
| Instruction delta at least +0.05 and wins exceed losses | PASS (+0.093750; 6 > 1) |
| Overall delta non-negative and CI lower bound at least -0.01 | PASS (+0.028727; +0.006320) |
| Pooled other-nine delta at least -0.005 | PASS (+0.021502) |
| Summarization delta at least -0.01 | PASS (+0.026711) |
| No other category below -0.025 | **FAIL (event ordering -0.030486)** |

## Interpretation

The product mechanism works: a persisted, complete, user-verified standing
facet lane materially improves instruction following, and the full cohort
shows a positive paired lift. The experiment also shows why composition cannot
be made default from this run alone. Event ordering had slightly more wins than
losses, but its losses were larger, and the frozen magnitude guard catches that
tail even though the category interval includes zero.

No category-based router, larger repeat, or threshold adjustment is authorized
from this result. A future mechanism may use product-valid intent or budget
semantics, but it requires a new preregistration and must not select on these
observed category outcomes.

The collision audit narrows that future work: conditionally applicable facets
need a principled rule-type versus answer-type compatibility check before
composition. Complete-lane composition without that check remains permanently
opt-in under the replication protocol.
