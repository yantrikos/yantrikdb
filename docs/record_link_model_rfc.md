# RFC: First-class record-to-record link model on the `remember` write path (engine v0.8.0)

**Status:** draft
**Author:** yantrikdb-core
**Date:** 2026-05-28
**Tracking:** engine main; closes [#48](https://github.com/yantrikos/yantrikdb/issues/48); responds to Phase 1 XC1 / Phase 2 Proposal 2 from the [yantrikdb-agi gap analysis](https://github.com/yantrikos/yantrikdb-agi) (2026-05-26 / 2026-05-27). Coordinated with yantrikdb-server v0.9.x MCP roadmap.

## Problem

Algo (Phase 1 gap XC1, line 533, after ~1000 iters of lived experience):

> "I have no way to express 'this is related to that.' My entire knowledge base is flat records. No graph, no links, no traversal. This is the deepest gap — it's not about a missing parameter, it's about a missing data model."

This is partially overstated — the engine DOES have a graph layer (`db.relate(src, dst, rel_type, weight)`, `claims` table, `entities` table, `memory_entities` join, `CognitiveEdge` / knowledge graph) — but algo's complaint accurately identifies a real gap: **none of these are first-class on the `remember` / `record` write path**. To link record A to record B today, the caller must:

1. Call `db.record(...)` → returns `rid_A`
2. Call `db.record(...)` → returns `rid_B`
3. Call `db.relate(rid_A, rid_B, ...)` — separately, non-atomically

If step 3 fails or is skipped, the records are flat. Algo never reliably reaches step 3 in their flow because nothing in the API surface requires it. Their entire knowledge base is, in practice, flat.

Beyond the atomicity gap, the existing `claims` machinery has semantic mismatches when used for record-to-record relations:

- `claims.src` / `claims.dst` are text columns intended to hold **entity names** (auto-classified via `crate::graph::classify_with_relationship`). Treating them as rids works but pollutes the entity classifier with UUIDs and confuses graph traversal queries.
- `claims` rows go into the `entities` table on insert (auto-creating one entity row per rid). Algo gets a phantom entity per memory, breaking `db.list_entities()` and entity-overlap potential_conflict detection.
- The `claims.weight` field's semantics (relation strength) don't map cleanly to record-to-record link types (`advances`, `supersedes`, `contradicts`, `supports`, `questions`, `derived_from`).

So the question isn't "should we add a graph layer" — we have one. The question is: **what's the right data model for record-to-record links, given the entity-graph already in place?**

## Design — `record_links` table + atomic `remember()` parameter

```
                 ┌─────────────────────────────────────────────┐
   remember ───► │ db.record(... links: Vec<RecordLink>)        │ ── atomic
                 │ inserts memories row + N record_links rows   │    transaction
                 └─────────────────────────────────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────────────────┐
                 │ record_links table                            │
                 │ source_rid, target_rid, link_type, weight,    │
                 │ created_at, hlc, origin_actor                 │
                 └─────────────────────────────────────────────┘
                                  ▲
                                  │ queries
                                  │
                 ┌─────────────────────────────────────────────┐
   recall  ───► │ db.recall(... expand_links: Option<usize>)    │
                 │ optionally chases N hops of links to surface  │
                 │ rids the relevance pool wouldn't have found   │
                 └─────────────────────────────────────────────┘

                 ┌─────────────────────────────────────────────┐
                 │ db.linked_records(rid, direction, link_type)  │
                 │ explicit traversal — inbound, outbound, both  │
                 └─────────────────────────────────────────────┘
```

**Core principle:** record-to-record links are a **distinct concern** from the entity graph. They share the SQL substrate but live in their own table with rid-specific semantics. Atomic write contract is the central win — if the caller's `remember` includes `links`, the engine guarantees the links land iff the record lands.

### Why not reuse `claims`?

Considered and rejected. Three reasons:

1. **Semantic clarity.** `claims.src` / `claims.dst` mean "entity names." Treating them as rids works mechanically but breaks the abstraction. Future engine work that assumes `claims` rows mean entity-entity (graph reasoning, entity-overlap conflict detection, knowledge-graph traversal) would have to special-case rid values.

2. **Auto-entity-creation side effect.** Every `claims` INSERT runs the entity classifier and writes to `entities`. For record-to-record links, that's wrong — we don't want a phantom entity per memory.

3. **Different link-type set.** Entity relations are open-set (`works_at`, `knows`, `tagged_with`, ...); record-to-record links are a small closed set (`advances`, `supersedes`, `contradicts`, `supports`, `questions`, `derived_from`) with optional `custom` for extensibility. Mixing them in one table makes both harder to query.

The two graphs share substrate (SQLite) but are logically separate. Same precedent as `record_revisions` (v30) sitting next to `memories` rather than encoding revisions as a special kind of edge.

## Components

### 1. Schema migration v31 — `record_links` table

```sql
CREATE TABLE IF NOT EXISTS record_links (
    link_id        TEXT PRIMARY KEY,
    source_rid     TEXT NOT NULL,
    target_rid     TEXT NOT NULL,
    link_type      TEXT NOT NULL,    -- see closed-set + Custom() below
    weight         REAL NOT NULL DEFAULT 1.0,
    created_at     REAL NOT NULL,
    hlc            BLOB NOT NULL,
    origin_actor   TEXT NOT NULL,
    UNIQUE(source_rid, target_rid, link_type)
);
CREATE INDEX IF NOT EXISTS idx_record_links_source
    ON record_links(source_rid, link_type);
CREATE INDEX IF NOT EXISTS idx_record_links_target
    ON record_links(target_rid, link_type);
```

`UNIQUE(source_rid, target_rid, link_type)` makes link inserts idempotent — re-running the same `remember(... links: ...)` (e.g. on replication apply) inserts once, no duplicates.

Indexes on both `source_rid` and `target_rid` make traversal queries cheap in either direction.

### 2. Engine API surface

```rust
/// Issue #48 (v0.8.0): record-to-record link types. Closed set + Custom
/// escape hatch. Engine validates the str value when inserting Custom.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkType {
    /// This record builds on / refines the target.
    Advances,
    /// This record replaces the target. The target is NOT automatically
    /// archived or tombstoned — replacement is a semantic claim, not a
    /// lifecycle action. Callers wanting deletion still use forget().
    Supersedes,
    /// This record disagrees with the target.
    Contradicts,
    /// This record provides evidence for the target.
    Supports,
    /// This record raises a question about the target.
    Questions,
    /// This record was inspired by / derived from the target. (Distinct
    /// from Advances in that no claim is made about correctness.)
    DerivedFrom,
    /// Extensibility hatch. The string is stored verbatim; engine does
    /// not interpret it. Server / MCP layer can document its own set.
    Custom(String),
}

pub struct RecordLink {
    pub target_rid: String,
    pub link_type: LinkType,
    pub weight: f64,    // default 1.0 per UPSERT contract
}

impl YantrikDB {
    /// db.record() gains an optional final parameter:
    pub fn record(
        &self,
        text: &str,
        memory_type: &str,
        // ... existing args ...
        links: &[RecordLink],   // NEW
    ) -> Result<String>;

    /// Add links to an existing record after the fact. Useful when
    /// the caller learns about a relationship later.
    pub fn link(&self, source_rid: &str, link: &RecordLink) -> Result<String>;

    /// Drop a single link.
    pub fn unlink(
        &self,
        source_rid: &str,
        target_rid: &str,
        link_type: &LinkType,
    ) -> Result<bool>;

    /// Traverse outbound links from rid.
    pub fn linked_records(
        &self,
        rid: &str,
        direction: LinkDirection,
        link_type: Option<&LinkType>,
    ) -> Result<Vec<LinkedRecord>>;
}

pub enum LinkDirection { Outbound, Inbound, Both }

pub struct LinkedRecord {
    pub rid: String,
    pub link_type: LinkType,
    pub weight: f64,
    pub created_at: f64,
    /// "outbound" or "inbound" — surfaces direction when caller queries Both.
    pub direction: String,
}
```

The engine `record()` gains one new positional parameter at the end (`links: &[RecordLink]`). Callers that don't use links pass `&[]`. This is a breaking change but follows the same pattern as v0.7.20's `correct()` rewrite — a coordinated bump where the Python binding wraps the new positional with a Pythonic kwarg default.

### 3. Recall integration — `expand_links: Option<usize>`

Today, `recall()` accepts `expand_entities: bool` which chases `claims` edges out from query-text-mentioned entities to pull in graph-adjacent memories. The link-model proposal adds a parallel knob:

```rust
pub fn recall(
    &self,
    // ... existing args ...
    expand_links: Option<usize>,    // NEW — None or Some(0) = no expansion;
                                    //       Some(N) = expand N hops
) -> Result<Vec<RecallResult>>;
```

Semantics:
- `None` / `Some(0)`: existing behavior; no link expansion.
- `Some(N)`: after the standard relevance pool is computed, the engine BFS's outbound links from each top-K rid up to N hops; each rid found is added to the candidate pool with a configurable graph-proximity boost (mirrors how `expand_entities` works today). Final scoring + MMR diversity selection proceeds normally.
- `RecallResult` gains a new field `via_link: Option<LinkPath>` so the caller can tell which results came from direct relevance vs which came from link traversal.

`expand_links` is **separate from** `expand_entities`. Both can be set; results dedupe by rid.

Performance note: traversal cost is `O(K * out_degree^N)` worst case. For practical N=1-2 and typical link-out-degree<10, this is bounded. The engine should still cap total candidates added via expansion at a tunable limit (default 50) to avoid pathological fan-out.

### 4. Replication — new "link" / "unlink" op kinds

The replication oplog gains two new op_type values:

| `op_type` | Payload |
|---|---|
| `link` | `{source_rid, target_rid, link_type, weight, created_at}` |
| `unlink` | `{source_rid, target_rid, link_type}` |

Atomic `record(links=...)` writes are recorded as the existing `record` op with `links: [...]` appended to the payload. The follower's `materialize_record` is extended to drain the links array after inserting the memories row, mirroring the leader's behaviour.

For the standalone `db.link()` / `db.unlink()` paths, dedicated `link` / `unlink` ops are emitted.

`record_links` has its own apply path on the follower (`materialize_link` / `materialize_unlink`):
- `INSERT OR IGNORE INTO record_links` — UNIQUE constraint makes apply idempotent.
- `INSERT OR IGNORE INTO replication_apply_log` — the v0.7.19 audit trail extended to cover the new ops.

### 5. Migration path from existing `supersedes`-as-string-metadata

Algo's pre-v0.8.0 workflow uses `metadata.supersedes = "<rid>"` (a string field) to express the supersedes relationship. The v0.8.0 migration includes a one-shot reifier:

```sql
-- v31 migration tail: reify metadata.supersedes into record_links
INSERT OR IGNORE INTO record_links
    (link_id, source_rid, target_rid, link_type, weight,
     created_at, hlc, origin_actor)
SELECT
    -- deterministic link_id derived from source+target so re-runs
    -- of the migration don't accumulate duplicates
    hex(randomblob(8)) || '-' || hex(randomblob(8)) as link_id,
    m.rid              as source_rid,
    json_extract(m.metadata, '$.supersedes') as target_rid,
    'supersedes'       as link_type,
    1.0                as weight,
    m.created_at       as created_at,
    zeroblob(8)        as hlc,    -- pre-existing rows have no recorded HLC
    'migration_v31'    as origin_actor
FROM memories m
WHERE json_extract(m.metadata, '$.supersedes') IS NOT NULL
  AND json_extract(m.metadata, '$.supersedes') != '';
```

The original `metadata.supersedes` field is **left in place** — removal is opt-in via a follow-up `db.compact_metadata()` call (not part of v0.8.0). This preserves back-compat for any reader still consuming the string field.

Algo's compression-wonder-cascade story (Phase 1 line 531) — 20 near-paraphrases written in 2 days — would post-migration produce 19 `supersedes` links (one per advance) forming a chain. `db.linked_records(rid_latest, Inbound, Some(Supersedes))` walks the chain backwards; `db.linked_records(rid_root, Outbound, Some(Supersedes))` walks it forwards.

### 6. Interaction with v0.7.20 `correct()` revision history

`correct()` mutates in place; `Supersedes` links express "this NEW record replaces that OLD record." They're orthogonal:
- Use `correct()` when the **same** semantic claim was wrong and is being fixed (preserves rid, appends revision).
- Use `Supersedes` link when a **new** claim replaces an old one (distinct rids, both retained, semantic relationship recorded).

The two patterns are at different semantic layers. The RFC does NOT propose automatically inferring `Supersedes` links from `correct()` calls; that would conflate the layers.

### 7. Interaction with `forget()`

When a record is forgotten (tombstoned), its outbound + inbound links MUST be handled. Two options considered:

**Option A — cascade delete** (DELETE FROM record_links WHERE source_rid = rid OR target_rid = rid). Simple, but loses audit trail.

**Option B — mark broken** (UPDATE record_links SET status = 'broken_target_forgotten'). Preserves history; requires adding a `status` column.

**RFC recommendation: Option B.** The audit trail value of "this link existed before the target was forgotten" is non-zero, and the storage cost is bounded by total link count. Implementation adds a `status TEXT NOT NULL DEFAULT 'active'` column to `record_links` and a `WHERE status = 'active'` clause to traversal queries.

`forget()` is updated to set `status = 'broken_target_forgotten'` on inbound links to the forgotten rid, and `status = 'broken_source_forgotten'` on outbound. The same handling extends to the replication apply path.

### 8. Backout strategy

v0.8.0 is the introduction. If the model needs revision:
- v0.8.1 can add columns to `record_links` without breaking back-compat.
- v0.8.x can deprecate / rename `LinkType` variants by extending the closed set and the engine accepting both old + new strings during a transition window.
- A full removal (rolling back to no record_links) is unsupported once any record_links rows exist — same constraint as v0.7.20's `record_revisions` rollback.

Feature flag: `--feature record_links` gates the new code path. v0.8.0 ships with the feature default-on; v0.8.0-rc.X can ship with it default-off for soak testing on the trader / yantrikdb-server side before the gate flips.

## Open questions

1. **`Supersedes` automatic tombstone?** Should adding a `Supersedes` link automatically tombstone the target? RFC says NO (separation of semantic claim from lifecycle action), but algo's mental model may treat them as one. Worth confirming with yantrikdb-agi during review.

2. **`weight` semantics on record_links.** What does `weight` mean for an `Advances` link? For `Custom`? RFC currently defaults to 1.0 with no enforced semantics. Could be left to the caller. Could be removed entirely.

3. **Link types — closed set vs open set.** RFC proposes 6 closed-set + `Custom(String)` escape hatch. Algo's Phase 2 (line 73-79) lists 6 link types. Are these the right 6? Specifically: should `derived_from` be merged into `advances` (the line between them is fuzzy)?

4. **`expand_links` interaction with `expand_entities`.** Both can be set. Should they share the candidate-budget cap, or have separate caps? RFC currently proposes shared cap (total expansion <= 50).

5. **MCP tool surface.** yantrikdb-server will need to add `link` / `unlink` / `linked_records` MCP tools. Their layer's call. Should the engine expose `link_type` strings (closed set + custom) or pre-encode for MCP? RFC recommends strings; server can document the closed set in tool docs.

6. **Schema migration v31 timing.** v0.8.0 is the natural cut. But the `correct()` rewrite (v0.7.20) just landed and the schema-v30 migration is still settling on trader + server clusters. Should we wait one more release cycle before bumping again? Probably yes — RFC review timeline naturally accomplishes this.

## Test plan

- **Atomicity** — fail-inject the link INSERT after the memories INSERT; assert neither lands (transaction rollback).
- **Link integrity across forget()** — create A linked to B; forget A; assert B's inbound link is `status = 'broken_source_forgotten'` and not deleted.
- **Replication apply idempotency** — apply the same `link` op twice; assert UNIQUE constraint kicks in and no duplicate row.
- **`expand_links` semantics** — create A→B→C chain; query for A's relevance pool; assert with `expand_links=Some(2)` that B and C surface; with `expand_links=None` they don't.
- **Migration v31 reifies metadata.supersedes** — seed a v0.7.x DB with 100 records using the old `metadata.supersedes` pattern; apply migration; assert 100 record_links rows with `link_type = 'supersedes'`.
- **`Custom(String)` round-trip** — insert a `Custom("my_special_link")` link; query via `linked_records(... Some(&LinkType::Custom("my_special_link".into())))`; assert match.
- **Encryption pass-through** — N/A; record_links doesn't store user text, just rids + link_type + numeric fields. No encryption pass needed.

## Stats and footprint

- Schema migration: 1 new table + 2 indexes + (in v0.8.x) 1 column (`status`).
- Engine API: 1 new positional param on `record()`, 3 new methods (`link`, `unlink`, `linked_records`).
- Python binding: corresponding kwargs + new methods.
- Replication: 2 new op_types (`link`, `unlink`); `record` op payload extended with `links: [...]` array.
- Expected migration time on trader's ~24k records: <5s (single JSON_EXTRACT scan + bulk INSERT OR IGNORE).

## Coordination

- **yantrikdb-agi**: review of the link-type closed set (open question #3) + `Supersedes` semantics (open question #1). Algo's lived experience is the ground truth for what these should mean.
- **yantrikdb-server**: MCP tool surface design (open question #5). Once the engine ships, server adds the user-facing `link` / `unlink` / `linked_records` tools.
- **trader**: pre-v0.7.19 23k-orphan postmortem (still pending) may want to run on a v0.8.0-rc1 build to see whether the link traversal would surface the orphan source. Possibly relevant.

## Decision summary

This RFC proposes:

1. **New `record_links` table** (schema v31), distinct from `claims` / entity graph.
2. **`db.record(... links: &[RecordLink])`** — atomic with the write.
3. **`db.link()` / `db.unlink()` / `db.linked_records()`** — explicit traversal API.
4. **`db.recall(... expand_links: Option<usize>)`** — N-hop expansion at recall time.
5. **`forget()` marks links broken** rather than deleting (audit trail preserved).
6. **Migration v31 reifies `metadata.supersedes`** into `record_links` rows; original metadata left in place for back-compat.
7. **v0.8.0 release vehicle.** Feature flag for soak testing before the gate flips.

The engine HAS a graph layer; the v0.8.0 work is about making record-to-record relations a first-class concern of the **write path** so the atomicity / link-integrity guarantee is mechanical, not caller-discipline.

---

*End of draft. Open to redteam review from yantrikdb-agi, yantrikdb-server, and architect. No code changes until this lands.*
