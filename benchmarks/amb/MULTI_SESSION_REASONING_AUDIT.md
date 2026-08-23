# AMB Multi-Session Reasoning Audit

## Scope

This audit asks whether YantrikDB's `0.61` multi-session-reasoning mean in the
`ydb-0151` BEAM-100K run can be improved by using speaker provenance to reduce
assistant-suggestion overcount. All model-scored controls use frozen synthetic
benchmark contexts and `deepseek-v4-flash:0731-cloud` as both answerer and
judge.

## User-Only Retrieval

The first paired arm compared the frozen `ydb-0151` contexts with the newer
role-aware artifact, which retrieves turn-level user evidence and excludes
assistant turns.

- Manifest SHA-256:
  `694f22643931f1593801e15f84a19c9332c935534aebf1babd7b36a32747fb03`
- Baseline context: `604,906` tokens
- User-only context: `534,674` tokens
- Mean: `0.5625` baseline vs `0.5069` user-only
- Delta: `-0.0556`, paired bootstrap 95% CI `[-0.1244, 0.0094]`
- Outcomes: 4 user-only wins, 26 ties, 10 baseline wins

This arm is rejected. It changed both evidence selection and presentation, and
its losses show that assistant confirmations and elaborations sometimes carry
facts needed for cross-session synthesis. Speaker provenance should constrain
claims, not erase potentially supporting evidence.

## Evidence-Preserving User-First Presentation

`reorder_speaker_first_contexts.py` performs a stable user, unknown, assistant
partition over each frozen context. It makes no model calls and proves that the
memory-block multiset and character count are unchanged. Across the complete
400-query artifact it reordered all 400 rows while preserving 16,150 blocks:
4,434 user, 5,521 unknown, and 6,195 assistant blocks.

- Manifest SHA-256:
  `16b9802d98ad6e9f81715f5c449becd0b8e445275e2fe175c54260857241eb4c`
- Baseline context: `604,906` tokens
- User-first context: `604,903` tokens; the three-token difference is caused
  only by tokenizer boundary changes after permutation
- Mean: `0.5233` baseline vs `0.5579` user-first
- Delta: `+0.0346`, paired bootstrap 95% CI `[-0.0331, 0.1146]`
- Outcomes: 6 user-first wins, 27 ties, 7 baseline wins

This global arm is also rejected for production because the interval includes
zero and losses slightly outnumber wins.

## Query-Dependent Signal

A post-hoc split provides a useful next hypothesis, not a confirmed result.
For 22 questions beginning with "How many", "How much", or "What two",
user-first presentation averaged `+0.0739` with a bootstrap 95% interval of
`[-0.0284, 0.2045]` (4 wins, 16 ties, 2 losses). The other 18 questions
averaged `-0.0134` with interval `[-0.0852, 0.0722]` (2 wins, 11 ties, 5
losses).

The direction matches the mechanism: count and set queries benefit when user
claims are made salient, while advisory synthesis queries use assistant
support. Because the split was selected after seeing results and remains
uncertain, it must be pre-registered and tested on an independent cohort before
it controls production retrieval.

## Query-Independent Topic-Card Transfer

A judge-free preflight tested whether existing query-independent dated topic
cards could provide the missing set-assembly structure. For each of the 20
conversation units, it transplanted the exact same card set from both
summarization questions onto both multi-session questions. Context generation
did not inspect multi-session queries, answers, scores, or gold text.

- Topic-card artifact SHA-256:
  `5070ff22bfff81141f8329c0dad867384b42855b1fe61b1ef575a8600ac881c1`
- Frozen baseline SHA-256:
  `43b8eb888adeb524caba70b013a8ca510c639bbf5312216e9b453d60185bcb8e`
- Source-document SHA-256:
  `fc0e64bac38fcde26eece776e818f70374338d4591ecc75346cb27b613d4c128`
- Context tokens: `604,906` baseline versus `613,482` topic cards
- Source-normalized reference retention: `0.9078` versus `0.6068`
- Per-query retention outcomes: 3 topic-card wins, 5 ties, 32 losses
- Funnel stages: 8 covered, 30 answer-loss, and 2 synthesis-required became
  4 covered, 9 answer-loss, 2 synthesis-required, and 25 retrieval-loss items

The 22 query-only count/set rows show the same failure: retention falls from
`0.8844` to `0.6099`, with one win, four ties, and 17 losses. This is a decisive
pre-model stop. Concern-level topic summaries are too coarse to preserve the
distinct values, examples, and events needed for counting and set assembly.
Future organizer work must expose grounded answer-sized concern items with
evidence IDs; rendering broader topic labels and summaries is not a substitute.

## Decision

Do not replace mixed-speaker retrieval with user-only evidence, and do not
apply user-first ordering globally. Preserve explicit source-role provenance
and all retrieved evidence. The next speaker-aware experiment should classify
count/set intent before retrieval, use stable user-first presentation only for
that intent, and evaluate on an independent frozen cohort with repeated
judging. Until that lifts, the current relevance-first presentation remains
the evidence-backed default.

The independent count/set confirmation later returned an exact null, and the
topic-card transfer failed its deterministic retention preflight. The remaining
product-shaped path is atomic concern-item construction and routing, not speaker
ordering or concern-level summary cards.
