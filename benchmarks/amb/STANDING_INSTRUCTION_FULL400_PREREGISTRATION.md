# AMB Standing-Instruction Full-400 Preregistration

Status: completed. Product replay passed, but the external acceptance run
failed two of four preregistered gates. Global default-on promotion is rejected;
the standing-instruction lane remains opt-in.

## Question

Does exposing the complete, authoritative standing-instruction set improve its
target category without degrading unrelated BEAM behavior when applied across
the entire benchmark?

## Frozen Arm

- Cohort: all 400 rows from frozen `ydb-0151`.
- Control: unchanged `ydb-0151` contexts.
- Treatment: every user-authored turn beginning with `Always` from the same
  conversation unit, in source-turn order, before ordinary memories.
- Extraction is query-independent and does not inspect gold, rubrics, answers,
  or scores.
- Each treatment row is capped independently to its control token budget using
  complete memory blocks. The instruction panel is never truncated.
- Model: `deepseek-v4-flash:0731-cloud`.
- Evaluation: one answer and one judge per arm per row.
- Synthetic benchmark data only; no real companion memories.

## Frozen Hashes And Budget

- Source result SHA-256: `43b8eb888adeb524caba70b013a8ca510c639bbf5312216e9b453d60185bcb8e`.
- Source documents SHA-256: `fc0e64bac38fcde26eece776e818f70374338d4591ecc75346cb27b613d4c128`.
- Treatment artifact SHA-256: `5ae03eade3a39c1fbaa3cf84eb9e0a2b64d50f179d6270eb8d9b5a645f051236`.
- Manifest SHA-256: `a08be2ed0c8f07c5224350717af8bbceaa3ea1e20da815059532f6ccf97cdbce`.
- Ordered query-ID SHA-256: `08b325da5c5a9830bdc94cb25fc09a74c5e4dc06b70e433911b27c5352901b52`.
- Control arm: 400 rows, 5,913,445 tokens, SHA-256
  `918f572927b75ab1bb2ae3edf5656eada132cf9a644953f31bf693c695d46863`.
- Treatment arm: 400 rows, 5,782,046 tokens, SHA-256
  `551662988effa63397928146b73775eac653fc0c37cf29ea23b018aded6838e3`.
- Projected calls: 800 answers and 800 judges.

## Gates

All gates must pass for default-on promotion:

1. `instruction_following` treatment minus control is at least `+0.05`, with
   more wins than losses.
2. Overall treatment minus control is non-negative, and its paired bootstrap
   95% lower bound is at least `-0.01`.
3. The pooled mean delta across the other nine categories is at least `-0.01`.
4. No non-instruction category mean delta is below `-0.025`.

Every category mean, paired win/tie/loss count, and bootstrap interval is
reported regardless of outcome. A failure keeps the feature opt-in and does not
authorize tuning the detector or thresholds against these results.

## Product Replay Requirement

Before model calls, an implementation-backed replay must prove that persisted
`standing_instruction` facets reconstruct the same ordered instruction panels
for all 400 query IDs, use only verified user evidence, stay within each frozen
token budget, and preserve the manifest hashes above. If product behavior
intentionally differs, it requires a new artifact and preregistration rather
than silently reusing this arm.

## Product Replay Result

The requirement passed before any external calls. A database containing all
5,732 raw turns from the 20 benchmark conversations persisted 90 source-keyed
standing-instruction facets. The store was closed and reopened, then the public
`recall_facets` lane reconstructed and whole-block-capped all 400 treatment
contexts without query, gold, rubric, answer, or score access during
extraction.

All 400 contexts matched the frozen treatment byte for byte, with zero panel or
context mismatches and identical ordered query IDs. The reconstructed artifact
SHA-256 was
`551662988effa63397928146b73775eac653fc0c37cf29ea23b018aded6838e3`,
exactly matching the preregistered treatment arm. The replay made no model or
network calls, so it did not consume or alter the external acceptance run.

## Full-400 Acceptance Result

The frozen run completed all 400 pairs in 7,020.5 seconds with no failed pairs.
Its result SHA-256 is
`0b30ebe5fbf045785a8c3dcbd33fbb4b1d117b456954a983eb208024a4948b64`.

| cohort | control | treatment | delta | paired 95% CI | W/T/L |
|---|---:|---:|---:|---:|---:|
| all 400 | 0.62967 | 0.63182 | +0.00215 | [-0.02316, +0.02731] | 60/275/65 |
| instruction following | 0.78750 | 0.86250 | +0.07500 | [+0.01250, +0.15000] | 5/34/1 |
| other nine pooled | 0.61213 | 0.60619 | -0.00595 | [-0.03340, +0.02140] | 55/241/64 |

The complete category result was:

| category | control | treatment | delta | paired 95% CI | W/T/L |
|---|---:|---:|---:|---:|---:|
| abstention | 0.67500 | 0.65000 | -0.02500 | [-0.15000, +0.10000] | 3/33/4 |
| contradiction resolution | 0.85313 | 0.85313 | +0.00000 | [-0.04063, +0.04375] | 5/29/6 |
| event ordering | 0.27069 | 0.24997 | -0.02072 | [-0.06572, +0.02110] | 12/13/15 |
| information extraction | 0.75677 | 0.74271 | -0.01406 | [-0.11198, +0.09063] | 7/25/8 |
| instruction following | 0.78750 | 0.86250 | +0.07500 | [+0.01250, +0.15000] | 5/34/1 |
| knowledge update | 0.51875 | 0.58125 | +0.06250 | [-0.03750, +0.17500] | 5/33/2 |
| multi-session reasoning | 0.56458 | 0.55062 | -0.01396 | [-0.07792, +0.05208] | 5/26/9 |
| preference following | 0.88750 | 0.86250 | -0.02500 | [-0.07500, +0.02500] | 2/34/4 |
| summarization | 0.57025 | 0.54049 | -0.02976 | [-0.09223, +0.02545] | 11/16/13 |
| temporal reasoning | 0.41250 | 0.42500 | +0.01250 | [-0.07500, +0.08750] | 5/32/3 |

Gate 1 passed: instruction following improved by `+0.075`, with five wins
against one loss. Gate 2 failed: the overall mean was non-negative, but the
confidence-interval lower bound (`-0.02316`) was below the `-0.01` floor. Gate
3 passed: the other-nine pooled delta was `-0.00595`. Gate 4 failed:
summarization fell by `-0.02976`, below its `-0.025` floor.

Per the frozen decision rule, the authoritative standing-instruction facet is
validated for its target behavior but must not be globally injected by
default. Product rollout remains explicit or query-scoped until a separately
preregistered mechanism can preserve the instruction lift without paying the
unrelated-category context cost.

## Post-Run Displacement Diagnosis

This section is diagnostic only. It was written after the frozen acceptance
result, makes no causal or promotion claim, and does not authorize threshold
tuning against the full-400 scores.

The equal-budget transform inserted one instruction panel and retained a
prefix of the original memory blocks. A deterministic audit verified that
relationship for every row and found that all 400 treatments displaced at
least one original block: `1.155` blocks and `434.54` tokens per row on
average. The target instruction cohort displaced `453.75` tokens per row but
had no rows where a conservative gold-answer bigram appeared only in removed
context. Summarization displaced `433.43` tokens per row and contained four of
the seven such lexical source-loss cases across the full cohort.

The lexical overlap is only a proxy: it neither proves that displacement
caused a score change nor captures paraphrased evidence. Its value is that it
confirms the proposed failure mechanism is reachable, including in the one
category that crossed the preregistered harm floor. A post-hoc category oracle
that uses treatment only for `instruction_following` and control everywhere
else would score `0.63717`, or `+0.00750` over control, while preserving the
observed `+0.075` target lift. That oracle is not deployable or fresh evidence;
it identifies query-gated composition as the next mechanism to preregister.

The reproducible audit is
`benchmarks/amb/analyze_standing_instruction_displacement.py`; its full local
row-level artifact is excluded from version control with the other model-run
outputs.
