# AMB Event-Ordering Selector Preflights (judge-free)

## Decision

**Event ordering is not an engine lever for the LLM-free core.** With the
entire user side of a conversation reachable, no query-only, model-free
selector — relevance, embedding novelty, clustering, chronology, or session
structure — picks the gold source turns above two to three times chance.
The gold "aspects" partition is drawn by an LLM and needs an LLM to redraw
it. This closes the preflight that `EVENT_ORDERING_V5_AUTOPSY.md` asked
for, negatively, before any answer or judge call was spent.

## Why these preflights existed

The frozen ledger already contained the two halves of a contradiction:

| Frozen fact | Source |
|---|---|
| Control `ydb-0151` contexts hold 18.3% of gold source turns at ~18.5K tokens | autopsy |
| Role-aware user-only contexts hold 95.3% at 20.5K tokens | `role-aware-event40-user-only.json`, re-measured here |
| User-only scored the same as the control (0.287 vs 0.293, paired null) | paired provenance run, 2026-08-20 |
| Five exact gold turns score 0.9; an LLM concern selector picking ~5 items scored 0.7–1.0 on Q9 and +0.058 category-wide | Q9 oracle, evidence-selector arm |

Reachability was therefore already solved; the reader drowns when handed
~84 turns and thrives on ~5 precise ones. The open question was whether a
small, precise set can be selected **without** an LLM.

## Inputs

- Engine: published `yantrikdb==0.18.0` wheel, isolated venv, one fresh
  store per conversation, every turn recorded with its speaker as
  `source` and its BEAM turn id as metadata (the role-aware ingest).
- Documents: `data/beam/100k/documents.json.gz`, SHA-256
  `fc0e64bac38fcde26eece776e818f70374338d4591ecc75346cb27b613d4c128`.
- Gold: BEAM `probing_questions[event_ordering].source_chat_ids` from
  `.datasets/beam/100k.json`, SHA-256
  `939c907ce64bc8ade1d5fce926f5c500004a25703a36ca5931dad4d4f1d3cbf3` —
  verified identical to `event40-organizer-membership-v2.json` (40/40).
- Gold is joined only at scoring. Every selector sees the query text, the
  conversation's user turns, the engine relevance score, turn order, and a
  `nomic-embed-text` embedding.
- The three rows quarantined by the autopsy (`9_event_ordering_0`,
  `18_event_ordering_0`, `19_event_ordering_0`) are excluded from the clean
  means (n = 37).

## Preflight 1 — first-mention selectors

`first_mention_preflight.py`, artifact
`artifacts/event40-first-mention-selector-preflight-v1.json`, SHA-256
`268fadce7c6b041b5dac8e4f541a075247a57b32c260e0c3a75a727a32c092ee`.

Novelty diagnostic over all 230 gold references and 5,502 non-gold user
turns: maximum cosine to any **earlier** user turn is `0.799` for gold and
`0.799` for non-gold. Gold first mentions are not novel under embeddings.
Gold turns sit at relevance rank percentile `0.36` with score ratio `0.56`
to the best hit — mid-pack, not top.

Clean means (37 queries), best row per family, rows ≈ 3N where N is the
requested item count; chance precision ≈ 5.75 / 143 ≈ 0.04:

| Selector | Recall | Precision | Rows | Tokens |
|---|---:|---:|---:|---:|
| relevance top-k (role-aware control) | 0.980 | 0.042 | 139 | 11,182 |
| chrono-stratified null (floor 0.5) | 0.195 | 0.072 | 15.5 | 1,472 |
| first-mention greedy novelty (θ 0.9) | 0.236 | 0.079 | 15.9 | 1,478 |
| cluster-first (top-40, θ 0.8) | 0.169 | 0.091 | 10.6 | 1,016 |

Every combination is in the artifact; none reaches 0.25 recall or 0.12
precision, and none puts a single query at ≥ 0.8 recall.

## Preflight 2 — session structure

`session_stratified_preflight.py`, artifact
`artifacts/event40-session-stratified-preflight-v1.json`, SHA-256
`1cacd5c3ef8e83ff495e9aec07cd63247fe3aad7f9d47717d75c7c076e760c37`.

The hypothesis came from conversation 1, whose gold turns 4/60/116 sit at
session starts 0/60/116. It does not generalise: sessions (split by header
date) average 9.0 per conversation with 16 user turns each; the gold offset
from session start has median 7 user turns; only 8% of gold turns open a
session and 23% fall within the first three; sessions equal the requested
count in 1/40 queries.

| Selector (per session, query relevance) | Recall | Precision | Rows |
|---|---:|---:|---:|
| first user turn of every session (no query) | 0.108 | 0.060 | 8.9 |
| top-1 of the whole session | 0.159 | 0.085 | 8.9 |
| top-2 of the first five user turns | 0.248 | 0.095 | 13.4 |

## What this permits and forbids

- Do not build an embedding-novelty or session-sampling "first mention"
  recall mode against this benchmark; the instrument says it cannot work.
- The remaining event-ordering levers are the opt-in read-time LLM selector
  (+0.058, 2026-08-19) and write-time organizer routes, both rejected as
  defaults for cross-category harm. Neither is core-engine work.
- Together with the honest headroom ledger of 2026-08-25, the LLM-free
  core has no measured BEAM lever left; further engine work is judged by
  the mechanical capability suite (`tests/test_capability_probes.py`), not
  by this benchmark.

## Reproduce

```powershell
python -m benchmarks.amb.first_mention_preflight `
  --documents C:/path/to/data/beam/100k/documents.json.gz `
  --beam-source C:/path/to/.datasets/beam/100k.json `
  --membership benchmarks/amb/artifacts/event40-organizer-membership-v2.json `
  --out fm-preflight.json --embed-cache nomic-cache.json --store-dir stores
python -m benchmarks.amb.session_stratified_preflight `
  --documents ... --beam-source ... --store-dir stores --out session-preflight.json
```

Run them as modules from the repository root: executed as plain scripts,
`benchmarks/amb/yantrikdb.py` (a provider copy) shadows the installed engine.
Ingest takes about six minutes for the 20 conversations; embeddings need a
local Ollama with `nomic-embed-text`. No answer or judge call is made.
