# Standing-Facet Applicability V4: Product Preflight

The frozen v4 mechanism passed its full 400-row product preflight before any
external answer or judge call was made.

## Result

| Gate | Result |
|---|---:|
| Rows aligned | 400/400 |
| Persisted facets parsed | 90/90 |
| Canonical instruction targets retained | 40/40 |
| Ordinary contexts byte-exact | 400/400 |
| Rows with positive form-conflict suppression | 28 |
| Facets suppressed | 53 |
| Date facets retained on direct date queries | 68/68 |
| Compatible process-timeline inclusions | 1 |
| Maximum additive facet tokens | 131/256 |
| Second fresh-open context replay | exact |
| Second fresh-open selection replay | exact |
| Second fresh-open predicate trace replay | exact |

The exact frozen decision census reproduced: 1,746 default inclusions, one
compatible process-timeline inclusion, 41 chronology/date-time suppressions,
and 12 chronology/formatting suppressions. Selection read query text but did
not read benchmark category, gold answer, rubric, prior answer, judge output,
or score. Unparsed directives default to inclusion, although all 90 directives
in this cohort parsed successfully.

## Frozen Inputs

| Input | SHA-256 |
|---|---|
| Metadata-rich 400-row result | `5ae03eade3a39c1fbaa3cf84eb9e0a2b64d50f179d6270eb8d9b5a645f051236` |
| Frozen control contexts | `918f572927b75ab1bb2ae3edf5656eada132cf9a644953f31bf693c695d46863` |
| BEAM 100k documents | `fc0e64bac38fcde26eece776e818f70374338d4591ecc75346cb27b613d4c128` |
| Ordered query IDs | `08b325da5c5a9830bdc94cb25fc09a74c5e4dc06b70e433911b27c5352901b52` |

## Paired Evaluation Freeze

The product preflight emitted treatment artifact
`44668a6f55289009aa6a7bc3ee011c3867c1de108d7444d71551746f8778f256`.
`prepare_paired_category_contexts.py --category all` deterministically converts
that artifact to the scorer format by retaining the original row order and
exact context bytes while serializing only `query_id` and `context`. A second
fresh output-directory replay reproduced the control, treatment, and manifest
hashes below byte-for-byte.

| Artifact | SHA-256 |
|---|---|
| Control arm | `918f572927b75ab1bb2ae3edf5656eada132cf9a644953f31bf693c695d46863` |
| Applicable-facet arm | `688c17fa3d2b50d4484b16b6906a2a2ae3285a7c9d1b6901cd330f364ac8e4f2` |
| Paired manifest | `cbf3471d3fbaf656097899fe4426ed70f74c931c1f5a50ed56bc86a30c9f9fc4` |

The paired evaluator independently validated 400 rows per arm, identical query
order, 5,913,445 control context tokens, 5,955,579 treatment context tokens,
and a fixed budget of 800 answer plus 800 judge calls using
`deepseek-v4-flash:0731-cloud`. The answer/run seed is `20260826`; the separate
20,000-resample paired-bootstrap seed is `20260827`.

The full product preflight report SHA-256 was
`d5935a575621df01b9fb8d6f6ee0a1bdb3a8bb2483bdbc3d68fdf6b6fb33b035`.
Large context and database artifacts remain untracked; the product builder
reconstructs them from the frozen inputs above.
