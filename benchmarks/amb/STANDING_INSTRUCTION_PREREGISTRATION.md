# AMB Standing-Instruction Salience Preregistration

## Hypothesis

Explicit user turns beginning with `Always` are authoritative standing
instructions. Persisting them as typed records and presenting the complete
unit-level set before ordinary memories will improve instruction-following
without inferring preferences from assistant prose.

## Frozen Arm

- Cohort: all 40 `instruction_following` rows from frozen `ydb-0151`.
- Control: frozen `ydb-0151` context.
- Treatment: every user-authored `Always ...` turn from the same conversation
  unit, in source-turn order, prepended as a standing-instruction panel.
- Extraction is query-independent and does not inspect gold, rubrics, answers,
  or scores.
- Treatment is capped to each control row's exact token budget using whole
  memory blocks. The standing-instruction panel is never truncated.
- Model: `deepseek-v4-flash:0731-cloud`.
- Discovery: one answer and one judge per arm per row.

## Frozen Preflight

- Source result SHA-256: `43b8eb888adeb524caba70b013a8ca510c639bbf5312216e9b453d60185bcb8e`.
- Source documents SHA-256: `fc0e64bac38fcde26eece776e818f70374338d4591ecc75346cb27b613d4c128`.
- Treatment artifact SHA-256: `96801afa6279cf93ddfa3cb14f81e2a2b9cb790cbf2a46fb2f9c90d35a00ea02`.
- Frozen manifest SHA-256: `a538f141c8e38b97293eac7b2b0e9e36973d39449ea793126c04d902ac04520c`.
- Ordered query-ID SHA-256: `5f97b2b76b8cff240e628724c74135179ecae343114da80ef94661e9a9fa7f15`.
- Control arm: 40 rows, 605,929 context tokens, SHA-256
  `2565f428ce52090e935cd7c7f73d4370ea97850c38fe83274be41f42245f840b`.
- Treatment arm: 40 rows, 592,006 context tokens, SHA-256
  `c3d7f8799cefcdd25b0b499de31b0ef4531b92e5881333b1153e868f6196aa87`.
- Each unit contains either three instructions (10 rows) or five instructions
  (30 rows). Whole-block capping removes one trailing block in 37 rows and two
  trailing blocks in three rows.
- Canonical instruction-target retention rises from a mean of `0.9284`
  (`36/40` rows at or above `0.75`) to `1.0000` (`40/40`).
- Projected external calls: 80 answers and 80 judges over synthetic benchmark
  data only. No real companion memories are included.

## Gate

The arm proceeds to repeated-answer confirmation only if treatment improves the
40-row category mean by at least `+0.05` and treatment wins outnumber losses.
The point floor is below the prior `+0.075` discovery gate because the audited
retrieval-owned category headroom is `0.0875`. Failure leaves product behavior
unchanged. This arm tests standing-instruction storage and salience; it does not
authorize heuristic preference extraction.

## Discovery Result

The frozen discovery run passed both gates:

| arm | mean rubric score |
|---|---:|
| frozen `ydb-0151` control | 0.800 |
| standing-instruction treatment | 0.900 |

Treatment minus control was `+0.100`, with 7 wins, 32 ties, and 1 loss. The
paired bootstrap 95% interval was `[+0.01875, +0.20000]`. The run completed all
40 pairs in 533.7 seconds. Its result SHA-256 is
`431497ec889e8ff5e33c7a601072323537b527d8ea518d87df227d1064b9b178`.

Five of the seven winning rows lacked the exact canonical instruction in the
control context. The other two already contained the user instruction, so
their improvements are consistent with a salience benefit rather than
retrieval repair. The sole losing row also lacked its canonical instruction in
control, but treatment still failed to follow it; repeated answers are needed
to distinguish generation variance from a stable regression.

This is a successful discovery result, not yet a production score claim. The
next stage must freeze its confirmation cohort and interpretation before making
more model calls. In particular, an outcome-selected discordant subset may test
whether row-level direction persists, but must not be used to re-estimate the
40-row category mean.
