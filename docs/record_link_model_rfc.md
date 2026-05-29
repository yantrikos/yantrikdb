# RFC: First-class record-to-record link model on the `remember` write path (0.7.x series, schema v31)

**Status:** draft (v2 — post-redteam)
**Author:** yantrikdb-core
**Date:** 2026-05-28
**Tracking:** engine main; closes [#48](https://github.com/yantrikos/yantrikdb/issues/48); responds to Phase 1 XC1 / Phase 2 Proposal 2 from the [yantrikdb-agi gap analysis](https://github.com/yantrikos/yantrikdb-agi) (2026-05-26 / 2026-05-27). Coordinated with yantrikdb-server v0.9.x MCP roadmap.
**Redteam:** v2 incorporates adversarial review (gpt-5.5, deepseek-chat) + three code-grounding investigations of `recall.rs` / `graph_ops.rs` / `hlc.rs`. Changes from v1 summarised in the changelog at the end.

## Problem

Algo (Phase 1 gap XC1, line 533, after ~1000 iters of lived experience):

> "I have no way to express 'this is related to that.' My entire knowledge base is flat records. No graph, no links, no traversal. This is the deepest gap — it's not about a missing parameter, it's about a missing data model."

This is **partially overstated** — the engine DOES have a graph layer (`db.relate(src, dst, rel_type, weight)`, the `claims` table, the `entities` table, the recall-time `expand_entities` BFS) — but algo's complaint accurately identifies a real gap: **none of these are first-class on the `remember` / `record` write path.** To link record A to record B today, the caller must:

1. `db.record(...)` → returns `rid_A`
2. `db.record(...)` → returns `rid_B`
3. `db.relate(rid_A, rid_B, ...)` — separately, non-atomically

If step 3 fails or is skipped, the records are flat. Algo never reliably reaches step 3 in their flow because nothing in the API surface requires it. Their knowledge base is, in practice, flat.

### The hidden second bug: linking records today actively corrupts the entity graph

Verified against `crates/yantrikdb-core/src/engine/graph_ops.rs:72-89`: `relate(src, dst, ...)` **unconditionally** upserts both endpoints into the `entities` table after inserting the claim, classifying each via `classify_with_relationship`. For a UUID rid, `classify_entity_type` (`knowledge/graph.rs:539-587`) has no rid-shaped branch and falls through to `entity_type = "unknown"`.

This is not hypothetical. `crates/yantrikdb-core/src/cognition/consolidate.rs` already calls `relate()` with `consolidated_rid` as an endpoint during consolidation edge-transfer. **Every consolidation today writes phantom `unknown`-typed entity rows for memory rids.** Those phantoms then pollute `db.list_entities()`, the entity-overlap potential_conflict detector, and — critically — the `expand_entities` BFS seed set. So the "just use `relate()` for record links" path is already in the codebase and already degrading the entity graph.

So the question is **not** "should we add a graph layer" — we have one, and it's already being misused for rids. The questions are: (1) what's the right data model for record-to-record links, and (2) how do those links participate in recall without re-creating the existing graph machinery or polluting the entity graph.

## Considered alternatives

Three designs were weighed. The redteam's central finding shaped the verdict: **record-to-record links that are invisible to recall are not "links" — they are decorative metadata, and worse, they hide a correctness bug** (see "Mandatory recall integration" below). That reframes the trade-off: the storage question (which table) matters less than the retrieval question (does the link affect what gets recalled).

### (B) Minimal fix — reuse `claims` with a sentinel `extractor='__rid_link'`. **Rejected.**

Tempting (cheapest; reuses `expand_entities`), but rejected for four verified reasons:

1. **Endpoint ambiguity is structural.** `claims.src` / `claims.dst` are `TEXT` with no kind discriminator. A query can't tell an entity name from a rid from an external URI without out-of-band convention. Sentinel `extractor` values are, per the redteam, "a disguised edge class — schema abuse."
2. **Entity-creation is tightly coupled to the claims insert** (`graph_ops.rs:72-89`) — same lock scope, unconditional, no flag to suppress. Reusing `claims` keeps minting phantom entities unless `relate()` itself is refactored (which we do anyway as a prerequisite — see below — but that doesn't make `claims` the right *home* for record links).
3. **Schema mismatch.** `claims` carries 20+ columns specific to NLP-extracted entity assertions: `polarity`, `modality`, `valid_from`/`valid_to`, `extractor_version`, `confidence_band`, `proposition_id`, `regime_tag`, `self_generated`, `source_lineage`, `modality_signal`. Record links need almost none of these and need things `claims` lacks (link lifecycle status tied to `forget()`). Forcing record links into this shape inherits accidental semantics.
4. **`expand_entities` budget contention.** Its BFS has a 30-node cap and 8-seed cap (`recall.rs:1476-1485`). If rids enter as pseudo-entities, record chains consume the budget meant for semantic neighbourhoods.

### (C) Full unification — one typed graph; records and entities are both first-class node kinds (`node_kind ∈ {entity, record}`). **Deferred — this is the eventual endgame, not the first link cut.**

The redteam (gpt-5.5) makes the strongest case for C as the *long-term* shape, and it's right that it's the cleanest model: one `graph_nodes` / `graph_edges` substrate, typed endpoints, edge classes, relation-aware traversal, no UUIDs masquerading as entities. We agree C is where this lands eventually.

But C-now is rejected as **over-architecture for the current cut**, on the same principle the [decoupled-write-path RFC](decoupled_write_path_rfc.md) used to reject per-tenant-thread designs: don't build for a shape the deployment doesn't have yet. Concretely, C requires subsuming the 20-column `claims` table + `entities` + the entire `knowledge/` and `cognition/` layer that reads them into a typed-node/edge model, with a compatibility view to avoid breaking every existing reader. That's a v0.9–v1.0 migration touching the whole graph subsystem. Doing it under the banner of "algo wants record links" would be scope capture.

**The discipline that makes deferral safe: design (A) to be a forward-compatible stepping stone to C** — see "Forward compatibility" §8. `record_links` is shaped so a future migration can fold it into `graph_edges WHERE src_kind='record'` without a semantic rewrite.

### (A) Dedicated `record_links` table, behind a unified recall-traversal abstraction. **Chosen.**

Physically separate storage (clean rid-specific schema, no entity pollution, audit-friendly lifecycle) — but **not** a second invisible graph. The recall layer gets a unified traversal abstraction so record links and entity claims both feed the candidate pool through the *same* `graph_proximity` scoring path. This is the redteam's explicit "A done right": *"A is not wrong because it is separate. A is wrong if it becomes invisible to the graph/retrieval layer."*

## Mandatory recall integration (the correctness argument)

This is the part v1 of this RFC got wrong by deferring. The redteam surfaced a concrete **correctness** failure — not a ranking nicety:

> A `supersedes` B. A user query is semantically close to B. HNSW retrieves B by embedding match. `expand_entities` expands the *entity* graph around B but knows nothing about `record_links`. MMR runs. B survives, A is never surfaced. **The engine returns a stale memory it explicitly knows was superseded.**

Therefore record-link expansion at recall time is **in scope for the first link cut, not a follow-up.** A separate table whose links don't reach recall would ship the correctness bug. The good news from the code-grounding pass: we do **not** invent new scoring — the machinery already exists and is reused.

### Reusing the existing `graph_proximity` machinery

Verified in `recall.rs` + `knowledge/graph_index.rs`:
- `expand_bfs(seeds, max_hops=2, max_nodes=30)` does bidirectional BFS returning `(node, hops, cumulative_weight)`.
- `graph_proximity(rid, expanded) = max over linked nodes of (weight / 4^hops)` — sharp decay so 1-hop ≈ 0.25, 2-hop ≈ 0.0625.
- Fused via `adaptive_graph_composite_score` at `GW_GRAPH = 0.30`, applied **after** the HNSW pool is scored and **before** MMR diversity selection. Graph-only candidates are pre-selected `top_k × 5` then MMR-filtered.

`expand_links` plugs into the **same** post-HNSW / pre-MMR stage with the **same** proximity-decay shape and the **same** fusion weight, operating over `record_links` instead of `claims`. The only addition is relation-aware polarity (below). `RecallResult.scores.graph_proximity` is reused; `why_retrieved` gains a `"linked via <link_type> from <rid>"` line mirroring the existing `"graph-connected via <entity>"`.

### Relation-aware scoring (new — v1 missed this entirely)

The redteam's sharpest technical catch: **not every link is a positive proximity boost.** A single scalar weight is wrong. Proximity *polarity* is derived from `link_type`:

| link_type | recall effect |
|---|---|
| `supports` | positive boost to the supported record (evidence pulls its target in) |
| `advances` | positive, directional — boost the advanced (newer) record when the older surfaces |
| `derived_from` | positive, provenance — weak boost |
| `supersedes` | **boost successor, demote predecessor** — surfacing B should pull in A (the superseder) AND down-weight B itself. This is what fixes the stale-recall bug. |
| `contradicts` | relevant-but-not-endorsing — surface the contradictor so the caller sees the conflict, but do not let it inherit a positive relevance halo |
| `questions` | weak relevance / uncertainty signal |
| `custom:<name>` | neutral positive (engine doesn't interpret) — treated like a weak `supports` |

Polarity is a pure function of `link_type`, computed engine-side. This **replaces** the caller-supplied `weight` column from v1 (which had no defined semantics — see changelog). A later revision may reintroduce a caller confidence-in-the-link scalar if a real use case appears, but v1 ships type-derived polarity only.

## Components

### 1. Prerequisite (independent of A/B/C): fix `relate()` entity pollution

Both redteam models flagged this as must-fix regardless of the link-model choice. `relate()` (and `upsert_entity_edge_with_id`, `ingest_claim`) must stop unconditionally creating `entities` rows for rid-shaped endpoints. Minimal change: a `rid`-shaped guard (the engine already mints UUIDv7 rids; `classify_entity_type` can detect the shape) that skips entity upsert for endpoints that are rids. Plus a one-shot cleanup migration for the existing phantom `unknown` entities that `consolidate.rs` has already written. This lands as its own small PR ahead of the link model so the cleanup is auditable in isolation.

### 2. Schema migration v31 — `record_links` table

```sql
CREATE TABLE IF NOT EXISTS record_links (
    link_id        TEXT PRIMARY KEY,         -- UUIDv7
    source_rid     TEXT NOT NULL,
    target_rid     TEXT NOT NULL,
    link_type      TEXT NOT NULL,            -- closed set + 'custom:<name>'
    status         TEXT NOT NULL DEFAULT 'active',
                                             -- active | broken_source_forgotten
                                             -- | broken_target_forgotten
    created_at     REAL NOT NULL,
    hlc            BLOB NOT NULL,            -- real HLC (see §6)
    origin_actor   TEXT NOT NULL,
    UNIQUE(source_rid, target_rid, link_type)
);
CREATE INDEX IF NOT EXISTS idx_record_links_source
    ON record_links(source_rid, link_type, status);
CREATE INDEX IF NOT EXISTS idx_record_links_target
    ON record_links(target_rid, link_type, status);
```

`UNIQUE(source_rid, target_rid, link_type)` + `INSERT OR IGNORE` makes link inserts idempotent across replication re-syncs. Indexes cover both traversal directions; `status` is in the index so the active-only filter is covered. **No `weight` column** (v1 had one; cut — see relation-aware scoring + changelog). Column order + naming deliberately mirror a future `graph_edges` row (§8).

### 2. Engine API surface

```rust
pub enum LinkType {
    Advances, Supersedes, Contradicts, Supports, Questions, DerivedFrom,
    Custom(String),   // stored as "custom:<name>"; engine does not interpret
}

pub struct RecordLink {
    pub target_rid: String,
    pub link_type: LinkType,
}

pub enum LinkDirection { Outbound, Inbound, Both }

pub struct LinkedRecord {
    pub rid: String,
    pub link_type: LinkType,
    pub created_at: f64,
    pub direction: String,   // "outbound" | "inbound"
}

impl YantrikDB {
    // record() gains a trailing param; callers not using links pass &[].
    pub fn record(&self, /* ...existing... */, links: &[RecordLink]) -> Result<String>;

    pub fn link(&self, source_rid: &str, link: &RecordLink) -> Result<String>;
    pub fn unlink(&self, source_rid: &str, target_rid: &str, link_type: &LinkType) -> Result<bool>;
    pub fn linked_records(
        &self, rid: &str, direction: LinkDirection, link_type: Option<&LinkType>,
    ) -> Result<Vec<LinkedRecord>>;
}
```

`record(... links)` is atomic: the memories row + all `record_links` rows commit in one transaction, or none do. Same coordinated-breaking-change pattern as v0.7.20's `correct()` rewrite — the Python binding wraps the new positional with a kwarg default (`links=None`).

### 3. Recall integration — `expand_links: Option<usize>`

```rust
pub fn recall(&self, /* ...existing... */, expand_links: Option<usize>) -> Result<Vec<RecallResult>>;
```

- `None` / `Some(0)`: no link expansion (default; existing behavior bit-for-bit).
- `Some(N)`: after the HNSW pool is scored, BFS the **active** `record_links` from each top-K rid up to N hops, apply relation-aware polarity (§ "Relation-aware scoring") through the existing `graph_proximity` fusion, then proceed to MMR. Default hop cap mirrors `expand_entities` (2); shares the `top_k × 5` candidate pre-select pool.
- Separate from `expand_entities`; both may be set; results dedupe by rid. **Shared** candidate-budget cap (total expansion ≤ 50) so the two expanders can't compound into pathological fan-out — answers v1 open-question #4 by decision rather than deferral.
- `supersedes` predecessor-demotion is applied here: when a record surfaces that is the `target` of an active `supersedes` link, its score is multiplied by a <1 factor so the superseder ranks above it.

### 4. Directionality + symmetry

Links are stored directionally (`source → target`). Traversal honours `LinkDirection`. One relation needs symmetric treatment: **`contradicts` is queried bidirectionally** — surfacing either endpoint should reveal the other, since "A contradicts B" and "B is contradicted by A" are the same fact. `linked_records(rid, Both, Some(Contradicts))` and the recall expander both treat `contradicts` as undirected. All others are strictly directional.

### 5. Replication — new `link` / `unlink` op kinds

| `op_type` | payload |
|---|---|
| `link` | `{source_rid, target_rid, link_type, created_at}` |
| `unlink` | `{source_rid, target_rid, link_type}` |

Atomic `record(links=...)` writes append a `links: [...]` array to the existing `record` op payload; the follower's `materialize_record` drains it after inserting the memories row. Standalone `db.link()` / `db.unlink()` emit dedicated ops. Follower apply (`materialize_link` / `materialize_unlink`): `INSERT OR IGNORE` (idempotent via UNIQUE) + extend `replication_apply_log` to cover the new op types.

### 6. HLC on links + migration backfill (v1 was broken here)

Code-grounding (`hlc.rs`, `replication.rs`) confirmed v1's `zeroblob(8)` HLC was a real bug: an all-zero HLC is the minimum BLOB value, sorts before everything in the HLC-ordered `extract_ops_since` scan, and once any peer watermark advances past it the row is **permanently locked out of replication**. Corrected:

- **Live links** (`db.link()` / `record(links=)`): stamped with `tick_hlc()` like any other op, `origin_actor = self`. Replicate normally.
- **Migration-backfilled links** (§7): stamped with `tick_hlc()` **at migration time**, `origin_actor = 'migration_v31'`. They become real, ordered, replicable ops with honest provenance — distinguishable in audit by the actor string. (Alternative considered: exclude backfill from replication entirely and let each node reify locally from its own `metadata.supersedes`. Rejected — nodes may have diverged metadata; one node reifying + replicating is the single source of truth.)

### 7. Migration v31 reifies `metadata.supersedes`

Algo's pre-link-model workflow encodes supersession as a `metadata.supersedes = "<rid>"` string. The migration reifies these into `Supersedes` links:

```sql
INSERT OR IGNORE INTO record_links
    (link_id, source_rid, target_rid, link_type, status, created_at, hlc, origin_actor)
SELECT
    <uuidv7 minted per row>,                          -- deterministic per (src,target) in the impl
    m.rid,
    json_extract(m.metadata, '$.supersedes'),
    'supersedes',
    'active',
    m.created_at,
    <tick_hlc() at migration time>,                   -- NOT zeroblob (see §6)
    'migration_v31'
FROM memories m
WHERE json_extract(m.metadata, '$.supersedes') IS NOT NULL
  AND json_extract(m.metadata, '$.supersedes') != '';
```

(SQL shown illustratively; the HLC + uuid stamping happen in Rust at migration time, not in pure SQL, precisely because HLC must come from the engine clock.) The original `metadata.supersedes` field is **left in place** for back-compat; removal is opt-in via a later `compact_metadata()` (not this cut). Open question for algo: are there other metadata-as-link conventions (`metadata.advances`, `metadata.derives_from`) worth reifying in the same pass? (§ Open questions.)

### 8. Forward compatibility to (C)

The deferral of full unification is only safe if A is a stepping stone, not a dead end. `record_links` is shaped so a future v0.9+ migration can fold it into a typed `graph_edges` table without semantic rewrite:
- `source_rid` / `target_rid` → `src_id` / `dst_id` with an added `src_kind='record'` / `dst_kind='record'`.
- `link_type` → `rel_type`; the closed set survives unchanged.
- `status` lifecycle survives.
- The recall-traversal abstraction (§3) is *already* the unified interface the redteam wanted — at the C migration, it simply points at `graph_edges` instead of `record_links ∪ claims`. Building it now is what lets C be a storage migration later rather than a retrieval rewrite.

### 9. Interaction with v0.7.20 `correct()` — a synergy, not a conflict

The redteam raised "if A is corrected, what happens to A's outgoing links?" The just-shipped v0.7.20 `correct()` answers this cleanly: `correct()` mutates **in place** and **preserves the rid**. Links are keyed on rid, so a correction does not touch them — A's links survive a correction automatically, no special handling. This is a concrete payoff of the v0.7.20 in-place semantics (had `correct()` still minted a new rid + tombstoned, every correction would have orphaned the corrected record's links).

`Supersedes` links and `correct()` remain at distinct layers and are NOT auto-inferred from each other: `correct()` = "this claim was wrong, fix in place, keep revision history"; `Supersedes` link = "this new record replaces that old one, both retained, relationship recorded."

### 10. `forget()` marks links broken, never deletes

Consistent with the engine's audit-trail bias (`forget()` tombstones; `correct()` keeps history). On `forget(rid)`: outbound links from rid → `status='broken_source_forgotten'`; inbound links to rid → `status='broken_target_forgotten'`. Traversal + recall filter `WHERE status='active'`. The same handling extends to the replication apply path. No cascade delete — the fact that a link existed before an endpoint was forgotten is retained.

## Open questions — RESOLVED for the link-model implementation

**Implementation decision (2026-05-28):** algo + server run autonomously on a lower-capability model and cannot meaningfully redteam the API in the abstract — they validate by *using* it. So the three remaining open questions are decided here by the authoring model, with reversible-by-default choices, and algo validates empirically against the rc build:

1. **`Supersedes` auto-tombstone? → NO.** Recall predecessor-demotion already de-emphasises the superseded record; the audit-trail bias says keep it; and the choice is reversible (adding auto-tombstone later is cheap, un-deleting is not). Algo can `forget()` explicitly to retire the predecessor.
2. **Closed set → keep all 6 + `custom:`, do NOT merge `derived_from` into `advances`.** Provenance ("came from", no correctness claim) and improvement ("is better than") are distinct; merging is lossy and hard to reverse, keeping them separate costs nothing (algo simply doesn't use one if unneeded). No speculative 7th — `custom:<name>` is the escape hatch.
3. **Reify only `metadata.supersedes` in v31.** Reifying conventions algo may not use consistently risks garbage links. Expandable in a later migration once algo confirms what else they encode as string fields.

Resolved-by-decision (were open in v1): recall integration is mandatory not deferred; `weight` cut in favour of type-derived polarity; `expand_links`/`expand_entities` share one budget cap; migration HLC uses `tick_hlc()` not zero.

### Implementation deviation from §2 API (2026-05-28)

§2 specs `record(... links: &[RecordLink])` and §3 specs `recall(... expand_links)`. **Implementation instead adds separate `record_with_links(...)` and `recall_with_links(...)` methods, leaving `record()` and `recall()` signatures untouched** (they delegate / are called internally). Rationale: `record()` and `recall()` have 100+ call sites across the engine, benches, examples, tests, and the pyo3 binding; the v0.7.20 `correct()` and `recall()` signature changes both caused call-site-cascade CI failures that the Windows dev environment could not catch locally (no pyo3 build). New methods preserve identical semantics with zero cascade. The batch path gains `RecordInput.links: Vec<RecordLink>` for the same reason.

**Consequence — no feature flag needed.** Because the link model is delivered entirely through *new* methods, every existing path (`record()`, `recall()`, etc.) is byte-identical with the feature present-but-unused. There is no existing-path behavior change to gate, so the RFC's "feature flag for soak" is unnecessary: a node can carry the link model dormant and behave exactly as before until something calls a `*_with_links` method. The one existing-path touch is `forget()` updating `record_links` rows, which is a no-op on an empty table. (The `relate()` prerequisite is a separate bugfix PR and does change `relate()` behavior, but that's a phantom-entity fix, not the link feature.)

**Recall integration — isolated post-pass, not a core weave.** `recall_with_links()` runs `recall()` for a larger base pool then applies supersedes-demotion + budget-capped neighbor surfacing as a bounded transform on the result set. Tradeoff: surfaced neighbors skip MMR diversity (the post-pass runs after `recall()`'s MMR). Chosen over weaving into the 3700-line recall hot path because an isolated transform is testable and low-blast-radius; the MMR tradeoff is acceptable for v1 since link sets are small and intentional. A future rev can move pre-MMR if empirically warranted.

## Test plan

- **Atomicity** — fail-inject the link INSERT after the memories INSERT; assert neither lands (transaction rollback).
- **Stale-recall correctness** — the motivating case: A `supersedes` B; query matches B's embedding; assert with `expand_links=Some(1)` that A surfaces AND B is demoted below A; with `expand_links=None` behaviour is unchanged (B alone).
- **Relation-aware polarity** — `contradicts` surfaces the contradictor without a positive relevance halo; `supports` boosts target; assert per-type effects.
- **`contradicts` bidirectionality** — link A→B as contradicts; assert `linked_records(B, Both, contradicts)` finds A.
- **forget() marks broken** — A linked to B; forget A; assert B's inbound link is `broken_source_forgotten`, not deleted; assert recall no longer traverses it.
- **correct() preserves links** — A linked to B; `correct(A, ...)`; assert A's links survive unchanged (rid preserved).
- **Replication idempotency** — apply the same `link` op twice; assert UNIQUE + INSERT OR IGNORE → no duplicate.
- **Replication HLC ordering** — assert migration-backfilled links have non-zero HLC and replicate to a fresh peer (regression guard for the v1 zeroblob bug).
- **Migration reifies metadata.supersedes** — seed a v0.7.x DB with 100 `metadata.supersedes` records; migrate; assert 100 active `supersedes` links with `origin_actor='migration_v31'` and non-zero HLC.
- **relate() pollution prereq** — call `relate(rid_a, rid_b, ...)`; assert NO `entities` rows are created for the rids (post-fix).
- **Custom round-trip** — insert `Custom("my_link")`; query via `linked_records(... Custom("my_link"))`; assert match.
- **expand_links + expand_entities shared cap** — both set on a densely linked corpus; assert total expansion candidates ≤ 50.

## Stats and footprint

- Schema: 1 new table + 2 covering indexes (v31). Plus the prerequisite phantom-entity cleanup migration.
- Engine API: 1 trailing param on `record()`; 3 new methods (`link`/`unlink`/`linked_records`); 1 trailing param on `recall()`.
- Recall: reuses `expand_bfs` + `graph_proximity` + `adaptive_graph_composite_score`; adds a relation→polarity map (~30 LOC) + predecessor-demotion in the fusion step.
- Replication: 2 new op types; `record` payload extended with `links`.
- Migration time on trader's ~24k records: <5s (single `json_extract` scan + bulk insert; HLC stamping is in-process).

## Coordination

- **yantrikdb-agi**: review the link-type closed set + `Supersedes` semantics (open questions #1, #2) + other-metadata-to-reify (#3). Their lived experience is ground truth.
- **yantrikdb-server**: MCP tool surface for `link`/`unlink`/`linked_records` — recommend mirroring the engine `LinkType` strings (closed set + `custom:<name>`) rather than a server-defined vocabulary, for one source of truth. Also: schema-timing — at least one clean production week on v0.7.20+v30 before cutting the schema-v31 release; want an rc build on trader (CT 168) behind the feature flag for soak + bench re-run.
- **trader**: the pre-v0.7.19 23k-orphan postmortem may want an rc1 build — link traversal could help characterise the orphan source.

## Decision summary

1. **`record_links` table** (v31), physically separate from `claims`/entities — chosen over reusing `claims` (B, schema abuse + pollution) and over full graph unification (C, over-architecture for current scale; deferred as the eventual endgame with A as a forward-compatible stepping stone).
2. **`db.record(... links)`** — atomic with the write.
3. **`db.link` / `unlink` / `linked_records`** — explicit traversal; `contradicts` bidirectional.
4. **`recall(... expand_links)` is mandatory in the first link cut, not deferred** — record links must reach recall or they ship a stale-recall correctness bug. Reuses the existing `graph_proximity` machinery with relation-aware polarity (`supersedes` boosts successor + demotes predecessor, `contradicts` relevant-not-endorsing, etc.).
5. **`forget()` marks links broken**, never deletes.
6. **Migration v31 reifies `metadata.supersedes`** with real `tick_hlc()` HLC + `origin_actor='migration_v31'` (NOT zeroblob).
7. **Prerequisite PR**: fix `relate()` so rid endpoints don't create phantom entities; clean up existing `consolidate.rs`-origin phantoms.
8. **0.7.x release vehicle (schema v31)** — likely v0.7.21+, consistent with v0.7.20 shipping a breaking `correct()` change as a patch. No feature flag needed (additive-method design = no existing-path behavior change to gate). Still gated on a clean v0.7.20 soak week + algo's empirical validation on an rc build before GA.

The engine HAS a graph layer; the schema-v31 link model makes record-to-record relations a first-class concern of the **write path** *and* the **recall path**, with the atomicity + link-integrity + recall-visibility guarantees mechanical rather than caller-discipline.

---

## Changelog v1 → v2 (post-redteam)

- **Added Considered-Alternatives section** weighing A/B/C explicitly. v1 asserted "separate table" without earning it against the minimal-fix baseline or the unification endgame.
- **Recall integration promoted from a deferred hand-wave to a mandatory, fully-specified component**, motivated by the stale-recall *correctness* failure the redteam surfaced. v1 said "graph-proximity boost (mirrors expand_entities)" with no detail; v2 specifies reuse of `expand_bfs`/`graph_proximity`/`adaptive_graph_composite_score` with the real constants.
- **Added relation-aware scoring** — the single biggest technical gap in v1. `supersedes` demotes predecessor; `contradicts` is relevant-not-endorsing; etc.
- **Cut the `weight` column** — v1 carried it with no defined semantics. Replaced by `link_type`-derived proximity polarity.
- **Fixed the migration HLC** — v1's `zeroblob(8)` was a verified replication bug (locks rows out behind the watermark). v2 uses `tick_hlc()` at migration time + `origin_actor='migration_v31'`.
- **Added the `relate()` entity-pollution prerequisite** — verified that record-rid linking already corrupts the entity graph via `consolidate.rs`; this must be fixed regardless of the link-model choice.
- **Added `contradicts` bidirectionality + directionality rules.**
- **Added the v0.7.20 `correct()`-rid-preservation synergy** — in-place correction preserves links for free.
- **Added forward-compatibility-to-C** section so the deferral of unification is a stepping stone, not a dead end.
- **Narrowed open questions from 6 to 3** by deciding the ones that were decidable (weight, budget cap, HLC, recall-mandatory).

*End of v2 draft. Open to further redteam from yantrikdb-agi, yantrikdb-server, and architect. No code changes until this lands.*
