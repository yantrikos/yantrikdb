# AMB Knowledge-Update Audit

## Scope

This audit examines the 14 zero-score knowledge-update rows in the frozen
`ydb-0151` BEAM-100K result. It compares gold and predicted values against
chronological user turns from the published `documents.json.gz` cache. The
goal is to avoid converting stale or absent benchmark labels into an unsafe
automatic supersession policy.

`audit_knowledge_update_gold.py` now reads both the raw-generator schema and
the published flattened gzip cache, merges continuation fragments by turn ID,
uses only user turns for source truth, and ignores bare calendar years when
matching values.

## Label Integrity

Six zero-score rows have a distinct conflicting user value after the gold
value:

| Query | Gold | Answer / later user value |
| --- | --- | --- |
| `10_knowledge_update_0` | 1,350 words | 1,800 words |
| `11_knowledge_update_1` | 90% | 92% |
| `13_knowledge_update_1` | $50 | later $35 and $75 budgets |
| `14_knowledge_update_0` | $75 | later $70 budget |
| `16_knowledge_update_1` | $450 | later $400 cap |
| `18_knowledge_update_0` | 4 overtime hours | later 5 hours |

Five more gold values have no exact user-turn support at all:

- `6_knowledge_update_1`: gold 7 women; no user turn states 7 mentees.
- `9_knowledge_update_1`: gold May 22; the user later states August 10.
- `11_knowledge_update_0`: gold March 27; the user states March 20.
- `12_knowledge_update_0`: gold April 22; the user confirms April 25.
- `14_knowledge_update_1`: gold 30 cupcakes; the user states 24.

Therefore 11 of 14 zeroes cannot safely train or gate a "latest value"
policy. Later mentions may be real revisions, temporary plans, scope changes,
or reversions; the evaluator does not expose that distinction.

## Residual Cases

The remaining three gold values all occur in the retrieved baseline context:

- `10_knowledge_update_1`: April 25 appears in an assistant continuation at
  memory rank 7; the answer selected the older April 20 user deadline at rank
  1.
- `12_knowledge_update_1`: March 30 appears directly in a user memory at rank
  3; the answer abstained.
- `19_knowledge_update_0`: 5-7 months appears at rank 8, but its source says
  6-9 months is usual and 5-7 months is only achievable after shortening the
  process. The question asks what the process "usually" takes, so the gold is
  semantically inconsistent with its own evidence.

Only the first two rows are plausible answer-selection failures. Neither is a
retrieval miss, and one depends on assistant evidence, matching the earlier
finding that user-only retrieval removes necessary support.

## Decision

Do not infer automatic supersedes links from mention time alone, and do not
change current-value retrieval to chase these labels. `created_at` provides
chronology but cannot determine whether a later value is a correction,
temporary scenario, different scope, or reversion. YantrikDB's explicit
correction and supersession provenance remains the safer production contract.

The official `0.6313` category mean materially understates source-grounded
quality. A useful next evaluator must relabel current values from the source
history and annotate revision identity before it can support a product gate.
