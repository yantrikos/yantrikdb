# BEAM Gate for the Extractor / Claim-Chain Branch (2026-09-05)

## Decision

**No harm, no measurable line movement.** The branch
`feat/extractor-word-boundaries-and-capability-probes` (head `74cd9a8`)
changes the retrieved set on 24 of 400 BEAM-100K queries. On those 24, a
same-day paired judged run scored +0.038 for the branch with a bootstrap
interval that includes zero (6 wins, 15 ties, 3 losses). Projected onto the
full line that is about +0.2 rubric points, inside the noise band, so a
full 400-query judged run is not spent. The gate the branch needed before
release is passed on the "does not hurt" side; it makes no lift claim.

## Method — judge-free gate first, judge only where retrieval changed

1. `context_diff_arm.py` (AMB fork) re-ran the `ydb-0151` retrieval
   environment — provider `yantrikdb`, `YDB_BENCH_TURN_AWARE=1`,
   `YDB_BENCH_TOPK=40`, `YDB_BENCH_EMBEDDER=potion-base-8M` — with the fork's
   CURRENT provider code (relevance-ordered raw chunks; `ydb-0151` itself
   also carried the env-gated conditional timeline lane, absent here), for
   all 400 queries, once per build, same day, same machine. Both arms share
   the provider byte for byte, so the A/B is exact; the scores are not
   line-comparable to `ydb-0151`'s per-category numbers:
   - A: published `yantrikdb==0.18.0` wheel;
   - B: the branch built into the same harness venv.
2. `compare_contexts.py` diffed the 400 contexts query by query.
3. `build_ctxdiff_arms.py` rendered only the changed queries into two
   frozen context arms (rag-mode rendering) with the evaluator's manifest.
4. `paired_frozen_context_eval.py` answered and judged both arms with
   `deepseek-v4-flash:0731-cloud`, one draw each, order alternated.

Stated claims and learned templates are NOT exercised by BEAM (the harness
writes no claims), so this gate measures only the extractor precision fix,
the two anchored place templates, and claim-chain traversal on the default
recall path.

## Retrieval diff (judge-free)

| Category | n | set changed | order changed | mean Jaccard |
|---|---:|---:|---:|---:|
| abstention | 40 | 4 | 5 | 0.967 |
| contradiction_resolution | 40 | 3 | 3 | 0.986 |
| event_ordering | 40 | 2 | 4 | 0.996 |
| information_extraction | 40 | 0 | 0 | 1.000 |
| instruction_following | 40 | 0 | 1 | 1.000 |
| knowledge_update | 40 | 1 | 1 | 0.999 |
| multi_session_reasoning | 40 | 3 | 4 | 0.977 |
| preference_following | 40 | 7 | 9 | 0.978 |
| summarization | 40 | 3 | 3 | 0.992 |
| temporal_reasoning | 40 | 1 | 1 | 0.994 |
| **all** | 400 | **24** | 31 | 0.989 |

Mean context size changed by +4 tokens; lexical gold coverage on the
changed rows moved +0.003. The changes are chunk swaps inside a fixed
k=40 budget, not additions.

## Paired judged result on the 24 changed queries

| Arm | Mean rubric |
|---|---:|
| A published 0.18.0 | 0.7125 |
| B branch 74cd9a8 | 0.7503 |
| Delta (B − A) | **+0.0378**, bootstrap 95% CI [−0.0156, +0.0941] |

Wins B / ties / wins A: 6 / 15 / 3. Per category on the changed rows:
contradiction +0.083 (3), event_ordering +0.117 (2), multi_session +0.167
(3), preference +0.036 (7), summarization −0.108 (3), abstention,
knowledge_update and temporal unchanged. One draw per query; individual
rows carry the ±0.13 answer-generation variance measured on 2026-08-20, so
none of the per-category numbers is a claim.


## Full-line run under the ydb-0151 recipe (2026-09-05, later the same day)

To answer "where are the gaps NOW", the branch engine ran all 400 queries
under the exact `ydb-0151` recipe: the d5c8196 harness (git worktree) with
`YDB_BENCH_RELATIONS=query`, `REL_ORDER=chrono`, `REL_TAGS=minimal`,
`REL_SCAFFOLD=1`, `TURN_AWARE=1`, `TOPK=40`, `EMBEDDER=potion-base-8M`, and
`deepseek-v4-flash:0731-cloud` pinned for answer and judge (`ydb-0151` used
the rolling `:cloud` tag on 2026-08-16). Run `ydb-branch-74cd9a8`, 1 h 20 m,
result SHA-256 prefix `50976b9fb75b9bca`.

| Category | branch | ydb-0151 | delta | W/T/L | identical turn sets |
|---|---:|---:|---:|---:|---:|
| abstention | 0.725 | 0.675 | +0.050 | 5/32/3 | 30/40 |
| contradiction_resolution | 0.847 | 0.828 | +0.019 | 9/23/8 | 31/40 |
| event_ordering | 0.289 | 0.291 | −0.002 | 10/21/9 | 35/40 |
| information_extraction | 0.797 | 0.773 | +0.023 | 6/28/6 | 37/40 |
| instruction_following | 0.800 | 0.787 | +0.013 | 2/36/2 | 39/40 |
| knowledge_update | 0.519 | 0.631 | −0.113 | 0/35/5 | 38/40 |
| multi_session_reasoning | 0.573 | 0.611 | −0.038 | 5/26/9 | 38/40 |
| preference_following | 0.844 | 0.912 | −0.069 | 1/34/5 | 34/40 |
| summarization | 0.587 | 0.593 | −0.006 | 11/19/10 | 38/40 |
| temporal_reasoning | 0.375 | 0.412 | −0.037 | 0/37/3 | 38/40 |
| **all** | **0.6355** | **0.6515** | **−0.016** | | **358/400** |

Paired bootstrap 95% CI on the line delta: [−0.040, +0.009].

**Attribution.** The retrieved turn sets are identical on 358 of 400
queries (mean Jaccard 0.978) between engine 0.15.1 and this branch. Every
query that lost score in knowledge_update, preference_following and
temporal_reasoning had an identical context (Jaccard 1.00, no turns added
or removed); the answer changed, not the evidence — e.g. `8_knowledge_update_0`
went from "April 22" to "May 12" and `3_knowledge_update_0` from "April 5"
to "April 1" on the same forty chunks, and `7_preference_following_0`
flipped to an abstention. The −1.6 points is answerer/judge drift between
the August rolling tag and today's pinned snapshot, not an engine
regression. Consequence: the `ydb-0151` per-category gap map still
describes the current engine; the knowledge_update losses are the reader
choosing between a stale and a current value on identical evidence.

## Hashes

- Capture A `ctx-0180-published.json` and B `ctx-patched.json`: regenerate
  with the commands below; the diff and the manifest pin them.
- Manifest SHA-256 `81ca8dc3e0c7e22bf2ac515f989879d4fda55c877b022d79a66ea3f5836310d4`
- Contexts A SHA-256 `b911bdf793a0faa2f7e8cdb0cd13432a5f665a3edb21edfe309b55f6561f1073`
- Contexts B SHA-256 `88b96083370214e74a798a495820c8bec69e59416075178c7bff8802d9ff6ad8`
- Ordered query ids SHA-256 `54d802b64fc914198b268baa0fad14f4164a63c6cd5b30b983553834e3b5ac66`
- Run fingerprint `b03fb56932c2a1da3ec7150521f5162abacb2331cc13b880d9de92f25c1fc96f`
- Result SHA-256 `2b1be515738d4afa814fd13464231fec03fd10973dda35fca94fb6b82a8fd8b4`
- Seeds: run 20260820, model 0, bootstrap 20260820; workers 2; elapsed 736 s.

## Harness note

Engine 0.18 refuses `set_embedder_named` on a store opened by
`with_default` (64d). The AMB provider's registry-embedder path now opens
at the registry model's dimension first (`_REGISTRY_DIMS`), which is how
the `potion-base-8M` configuration of `ydb-0150/0151` is reproduced on
0.18.

## Reproduce

```powershell
# in codes/agent-memory-benchmark, harness venv
$env:YDB_BENCH_TURN_AWARE=1; $env:YDB_BENCH_TOPK=40; $env:YDB_BENCH_EMBEDDER="potion-base-8M"
python context_diff_arm.py --name <arm> --out outputs/ctxdiff/<arm>.json   # once per build
python compare_contexts.py A.json B.json --changed-out changed.json
python build_ctxdiff_arms.py A.json B.json --changed changed.json --out-dir outputs/ctxdiff/paired
# in codes/yantrikdb/benchmarks/amb
python paired_frozen_context_eval.py --contexts-a ... --contexts-b ... --manifest ... `
  --label-a published-0.18.0 --label-b patched-74cd9a8 --workers 2 --out result.json
```
