# BEAM Category Loss Funnel

## Scope

This judge-free audit examines the four largest non-temporal loss buckets in
the frozen `ydb-0151` BEAM-100K run. It measures distinctive reference-token
support in the source conversation, retention in retrieved context, retention
in the final answer, exact knowledge-update value chronology, and answer tokens
supported only by explicitly assistant-authored memory blocks.

The input hashes are:

- Results: `43b8eb888adeb524caba70b013a8ca510c639bbf5312216e9b453d60185bcb8e`
- Source documents: `fc0e64bac38fcde26eece776e818f70374338d4591ecc75346cb27b613d4c128`

No model or judge calls are made. Lexical coverage is a diagnostic proxy, not
a replacement benchmark score.

## Results

| Category | Mean score | Source support | Context retention | Answer retention | Dominant finding |
|---|---:|---:|---:|---:|---|
| summarization | 0.593 | 0.944 | 0.869 | 0.555 | 131/195 reference items fail mainly after retrieval |
| knowledge update | 0.631 | 0.988 | 0.988 | 0.608 | 11/14 zeroes are stale-label or absent-gold cases |
| multi-session reasoning | 0.611 | 0.921 | 0.908 | 0.427 | 30/40 answers lose source-supported content |
| abstention | 0.675 | n/a | n/a | n/a | 7/13 zeroes are dominated by assistant-only support |

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

## Reproduction

```powershell
python benchmarks/amb/audit_category_loss_funnel.py `
  C:/path/to/ydb-0151/rag/100k.json `
  C:/path/to/beam/100k/documents.json.gz `
  --out C:/path/to/category-loss-funnel.json
```
