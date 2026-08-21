# AMB Rollup Membership Calibration

## Question

Can a cheap, query-dependent rescoring pass recover relevant items that are
already present in the adaptive-rollup candidate pool but omitted from the
served subset?

## Benchmark Semantics

BEAM currently scores `event_ordering` by independently judging whether the
answer mentions each rubric nugget, then averaging those scores. The Kendall
tau ordering implementation in `beam.py` is not called by `score_result`.
Therefore the measured `0.29347` YantrikDB result is primarily a hidden
partition-membership score. Ordering matters to answer quality, but it is not
the binding scored metric in this benchmark implementation.

This interpretation was verified at agent-memory-benchmark commit
`d5c81960aaebf695f2b8bada9ce1486f8b684a51`; `beam.py` SHA-256 was
`15a6ff268ced7909d8644dfa37635b164a518594902b43eb00378230b92c27cd`.

## Protocol

- Frozen input: `adaptive-rollup-v4-full40.jsonl`, SHA-256
  `4d9f9c00699a71609b1978d1d042fd01353511b22847226d7e9b860d0aab0d56`.
- Gold audit: `event40-gold-alignment-nomic.json`, SHA-256
  `7f53d98bcf4348cc105c283ec0d807c1a43fcc2f5539a84ce9eac4b3bff4ba63`.
- Available cohort: 18 queries from 9 source dialogues, with 405 frozen
  candidates and 96 requested answer items.
- Selection: leave one source dialogue out. Both query variants from a dialogue
  always remain in the same fold.
- Confidence intervals: 20,000 cluster-bootstrap samples over source dialogues.
- Membership metrics: maximum-weight one-to-one assignment between selected
  candidates and gold items, preventing one umbrella candidate from satisfying
  multiple requested items.
- Product gate: semantic-coverage CI lower bound above zero, with no regression
  in matched recall, source-turn recall, or chronological similarity.
- Chronology diagnostic: compare candidates and gold items after independently
  sorting both by source turn. The raw gold alignment order is nonmonotonic in
  most of the audited corpus and is not used as a chronology target.
- Temporal holdout: unavailable because the frozen artifact contains only the
  first 9 dialogue groups and no collection timestamp split.

## Results

The existing served subset scored `0.61280` semantic coverage, `0.68333`
matched recall, and `0.54544` chronological similarity under the corrected
offline proxy.

The best grouped deterministic density experiment produced a mean semantic
lift of `+0.01339`, matched-recall lift of `+0.07407`, and chronological lift
of `+0.03028`. Its semantic 95% CI was `[-0.00976, +0.03693]`, and source-turn
recall fell `-0.01728`. It failed the product gate. Raising the required
training-fold objective lift from `+0.002` to `+0.01` selected the same held-out
policies and did not change the result.

A follow-up feature rewarded coverage of a previously unrepresented
conversation date, testing the generator-partition hypothesis without gold
labels. It was selected in only 1 of 9 held-out dialogue folds. Semantic lift
was `+0.01416` with 95% CI `[-0.00790, +0.03707]`, while source-turn recall fell
`-0.01728`. It also failed the gate. Report SHA-256:
`064cc38b3c7501dfd387e5f42028ff48ad348d87d14f3ede3324e2b2afb71eb4`.

The two generated reports are:

- `membership-calibration-nomic-one-to-one-v4.json`, SHA-256
  `44c26d2f3a373fa220db8dec9c026f4d1f84258ff13c09fb98cfd5ec0f41f0c2`.
- `membership-calibration-nomic-session-v5.json`, SHA-256
  `064cc38b3c7501dfd387e5f42028ff48ad348d87d14f3ede3324e2b2afb71eb4`.

Both used `nomic-embed-text:latest`, digest
`0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`.
Run the first report with the CLI defaults; add `--include-session-feature` to
reproduce the second. Use `--minimum-train-delta 0.01` for the conservative
activation check.

Pointwise LLM scoring was probed with `qwen3.5:9b`, `qwen3.5:4b`,
`qwen3.8:27b`, `qwen2.5:14b`, and `deepseek-v4-flash:0731-cloud`. Small models
collapsed to almost uniform scores; the larger local model was slow and still
overgeneralized adjacent work; the cloud model swung from all-zero to
overinclusive after referent-resolution prompting. No pointwise scorer was
stable enough for full-cohort calibration or product use.

All completed pointwise probes used candidate row indices 4 and 14 except the
final DeepSeek prompt, which used row 4 only. These are legacy diagnostic
artifacts from before scorer protocol v2 and are deliberately rejected by the
calibrator's provenance gate. Probe artifact hashes:

- `membership-scores-qwen9b-v1.jsonl`: `c73cdaefc76b324c8399af5df8147526b2aac22872c256132f4caf2f7f8a4d00`.
- `membership-scores-qwen38-27b-probe.jsonl`: `a1f01d7b561e83961263978ffb74570aa932584f9a4d5054abd3d7f555aaea18`.
- `membership-scores-qwen35-4b-probe.jsonl`: `5617bb0be714598331a9c4836d2d14e18d9204d006e6e0001a46e0ee86911d6f`.
- `membership-scores-deepseek-v4-probe.jsonl`: `5fbf818e0b53f93bf836f6e856c14847dcac5306b193a88893216e3f01a4ac17`.
- `membership-scores-deepseek-v4-resolved-probe.jsonl`: `6ff31cb10cd43c33bcf26403e80e3cb901da16b69e87ddb071233ffbf7376cf1`.
- `membership-scores-deepseek-v4-resolved-v2-probe.jsonl`: `709590d1e7d3cee87802c85cc7657e585ed17530d9ff95a5adb10ca0ffee7e23`.

The greedy gold-informed candidate-pool selector reached `0.69765` semantic
coverage, `0.92593` matched recall, and `0.61759` chronological similarity.
Relative to the served subset, that is `+0.08485`, `+0.24259`, and `+0.07215`,
respectively. This is a diagnostic selector, not a true combinatorial oracle.

## Decision

Do not ship a deterministic rescue gate or pointwise LLM judge from this
cohort. The candidate pool contains meaningful headroom, but the tested
selectors do not identify the missing members reliably out of dialogue.

The next membership experiment should first expand the frozen cohort to all 20
dialogue groups (or collect explicit omission labels in product telemetry),
then test a selector that resolves the query's concrete referent before scoring
candidates. Sorting and date ordering should remain downstream of membership
selection.
