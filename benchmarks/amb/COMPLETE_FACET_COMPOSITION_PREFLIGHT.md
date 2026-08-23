# Complete Facet Composition V2: Product Preflight

The frozen v2 mechanism passed its full 400-row product preflight before any
external answer or judge call was made.

## Result

| Gate | Result |
|---|---:|
| Rows aligned | 400/400 |
| Canonical instruction targets retained | 40/40 |
| Complete verified lanes composed | 400/400 |
| Ordinary contexts byte-exact | 400/400 |
| Maximum additive facet tokens | 131/256 |
| Facets per namespace | 3–5 |
| Fresh-open artifact replay | exact |
| Fresh-open RID selection replay | exact |

Composition used no query text, category, gold answer, rubric, prior answer,
or score. Extraction used raw turns only, persisted product facets, verified
user evidence, and a fresh store open before composition.

## Frozen Inputs

| Input | SHA-256 |
|---|---|
| Metadata-rich 400-row result | `5ae03eade3a39c1fbaa3cf84eb9e0a2b64d50f179d6270eb8d9b5a645f051236` |
| Frozen control contexts | `918f572927b75ab1bb2ae3edf5656eada132cf9a644953f31bf693c695d46863` |
| BEAM 100k documents | `fc0e64bac38fcde26eece776e818f70374338d4591ecc75346cb27b613d4c128` |
| Ordered query IDs | `08b325da5c5a9830bdc94cb25fc09a74c5e4dc06b70e433911b27c5352901b52` |

## Paired Evaluation Freeze

| Artifact | SHA-256 |
|---|---|
| Control arm | `918f572927b75ab1bb2ae3edf5656eada132cf9a644953f31bf693c695d46863` |
| Complete-facet arm | `43a222af04195cc2d18d3a2ad15124a73208afeb0df352c956e69e86c5c5d9fb` |
| Paired manifest | `b1b3b009b4f5d531d58edb80f157d46f02d80169535f190e1d603806cb673989` |

The paired evaluator independently validated 400 rows per arm, identical query
order, 5,913,445 control context tokens, 5,956,705 treatment context tokens,
and a fixed budget of 800 answer plus 800 judge calls using
`deepseek-v4-flash:0731-cloud`.

The full local preflight report SHA-256 was
`9129ae7ef2939eea031e675d8f8fad6eb390356cfc9b59cec0e784a860b101c5`.
Large context and database artifacts remain untracked; the builder reconstructs
them from the frozen inputs above.
