# AMB Standing-Instruction Full-400 Preregistration

Status: frozen preflight only. No external calls have been made for this arm.
Run only after the product persistence and recall replay reproduces the frozen
treatment contract.

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
