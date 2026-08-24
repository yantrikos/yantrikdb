# AMB Standing-Facet Applicability V5 Final Power Preregistration

Status: estimator, seeds, tooling, gates, and finality frozen before any v5
external call.

The completed evaluation and terminal opt-in decision are recorded in
[FACET_APPLICABILITY_V5_RESULT.md](FACET_APPLICABILITY_V5_RESULT.md).

## Decision Question

The unchanged `facet-form-conflict-v1` treatment produced a positive v4
overall point estimate (`+0.014255`) and instruction-following lift (`+0.075`),
but failed the summarization and event-ordering gates on one answer draw. Does
the same mechanism pass all six gates when answer variance is reduced by three
independent draws per query and arm?

This is a power check, not a new treatment. No predicate, context, category,
prompt, judge, threshold, or cohort is changed after v4. No v4 answer, score,
or judge output is pooled into v5.

## Frozen Inputs

- Cohort: the same 400 BEAM 100k queries.
- Model: `deepseek-v4-flash:0731-cloud` for answers and judges.
- Control SHA-256:
  `918f572927b75ab1bb2ae3edf5656eada132cf9a644953f31bf693c695d46863`.
- Treatment SHA-256:
  `688c17fa3d2b50d4484b16b6906a2a2ae3285a7c9d1b6901cd330f364ac8e4f2`.
- Paired manifest SHA-256:
  `cbf3471d3fbaf656097899fe4426ed70f74c931c1f5a50ed56bc86a30c9f9fc4`.
- Ordered query-ID SHA-256:
  `08b325da5c5a9830bdc94cb25fc09a74c5e4dc06b70e433911b27c5352901b52`.
- Score-blind category source: BEAM `data/beam/100k/queries.json.gz`,
  SHA-256
  `f58e001b3aeb0f13c894aa0ca6896a8181addbec4c140cc982cd0de9c0f0c465`.

The v4 product preflight remains the lineage proof: 400/400 ordinary contexts
were exact, all 40 instruction targets were retained, all close/reopen replays
were exact, and the treatment was built by the product path. V5 consumes the
already-frozen scorer artifacts byte for byte and does not rebuild them.

Before this document was committed, all three exact `--preflight-only`
commands reproduced the frozen manifest, arm, and ordered query-ID hashes. Each
reported two 400-row arms, 800 projected answers, 800 projected judges,
`workers=2`, bootstrap seed `20260831`, and its matching run/model seed from the
table below. All three output paths and all three `.partial` checkpoint paths
were absent. The focused evaluator, combiner, v4-gate, and v5-gate test set
passed `21/21`; Ruff check and format passed.

## Estimator

Three independent paired runs each produce one answer and one judge per arm
per query. They use these run and provider-model seed pairs:

| Replicate | Run/arm-order seed | Provider model seed |
|---|---:|---:|
| 1 | `20260828` | `20260828` |
| 2 | `20260829` | `20260829` |
| 3 | `20260830` | `20260830` |

For each query, the final control score is the arithmetic mean of its three
control scores and the final treatment score is the arithmetic mean of its
three treatment scores. The paired delta is treatment mean minus control mean.
Every overall/category mean, win/tie/loss count, interval, and gate is computed
over those 400 paired mean-of-three deltas.

The existing evaluator's `--answer-repeats 3` path is prohibited because it
selects median-scored answers for pair-level comparison. V5 instead runs three
separate `--answer-repeats 1 --judge-repeats 1` evaluations and combines them
with `combine_paired_replicates.py`.

`--model-seed` is bound before the benchmark package imports its Ollama
provider, recorded in each run fingerprint, and verified against the imported
provider value before calls. A cloud provider may honor this seed loosely;
independence rests on three separate runs, while seed binding makes their
configuration auditable rather than promising provider determinism.

The final paired bootstrap uses 20,000 query-level resamples and the separate
seed `20260831`. Individual replicate intervals are not decision evidence.

## Call Budget

- Answers: `400 queries * 2 arms * 3 replicates = 2,400`.
- Judges: `2,400 answers * 1 judge = 2,400`.
- Total external calls: `4,800`.

No output or checkpoint path may exist at first launch. A `.partial` file may
be resumed only for the same interrupted replicate and only when its complete
run fingerprint matches. Replicates run serially. Their logs are not inspected
for scores until all three runs complete, so a partial result cannot influence
whether later replicates run.

## Frozen Commands

Each replicate first runs `--preflight-only` with the exact shared paths,
model, manifest, worker count, repeat counts, run seed, model seed, and
bootstrap seed. Preflight must report two 400-row arms with the frozen ordered
query-ID hash, 800 answer calls, 800 judge calls, and the expected execution
seeds without constructing a client or making an external call.

The scoring command template is:

```powershell
python benchmarks/amb/paired_frozen_context_eval.py `
  --contexts-a .tmp-facet-applicability-v4-product/paired/ydb0151-all400.json `
  --contexts-b .tmp-facet-applicability-v4-product/paired/applicable-facets-all400-v4.json `
  --label-a ydb0151-all400 `
  --label-b applicable-facets-all400-v4 `
  --model deepseek-v4-flash:0731-cloud `
  --split 100k `
  --workers 2 `
  --answer-repeats 1 `
  --judge-repeats 1 `
  --seed <20260828|20260829|20260830> `
  --model-seed <same-seed> `
  --bootstrap-seed 20260831 `
  --manifest .tmp-facet-applicability-v4-product/paired/manifest.json `
  --out .tmp-facet-applicability-v5-power/replicate-<same-seed>.json
```

After all three complete, the frozen combiner validates every input hash,
fingerprint, configuration, 400-row cohort, and ordered query ID before writing
`combined-v5.json`. `analyze_facet_applicability_v5.py` then reads only IDs and
frozen category labels from the score-blind BEAM query source and applies the
six gates below with bootstrap seed `20260831`.

```powershell
python benchmarks/amb/combine_paired_replicates.py `
  --replicate .tmp-facet-applicability-v5-power/replicate-20260828.json `
  --replicate .tmp-facet-applicability-v5-power/replicate-20260829.json `
  --replicate .tmp-facet-applicability-v5-power/replicate-20260830.json `
  --expected-seed 20260828 --expected-seed 20260829 --expected-seed 20260830 `
  --expected-model-seed 20260828 `
  --expected-model-seed 20260829 `
  --expected-model-seed 20260830 `
  --bootstrap-seed 20260831 `
  --output .tmp-facet-applicability-v5-power/combined-v5.json

python benchmarks/amb/analyze_facet_applicability_v5.py `
  --result .tmp-facet-applicability-v5-power/combined-v5.json `
  --source $env:AMB_ROOT/data/beam/100k/queries.json.gz `
  --output .tmp-facet-applicability-v5-power/analysis-v5.json
```

## Promotion Gates

All six v4 gates are unchanged and all must pass:

1. Instruction-following delta is at least `+0.05`, with more wins than
   losses.
2. Overall delta is non-negative and its paired 95% interval lower bound is at
   least `-0.01`.
3. Pooled other-nine-category delta is at least `-0.005`.
4. Summarization delta is at least `-0.01`.
5. No category other than instruction following is below `-0.025`.
6. Event-ordering delta is non-negative.

There is no post-hoc threshold change, category router, query allowlist,
replicate exclusion, score-tuned exception, or v4 pooling.

## Finality

this is the LAST composition arm on this line regardless of outcome. Pass all
six → default-on promotion. Any fail → opt-in is TERMINAL, arc closes, no
further power escalation, no mechanism variants without a fundamentally new
evidence base.
