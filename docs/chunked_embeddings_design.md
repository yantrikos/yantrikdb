# Chunked embeddings — multi-vector-per-record

Status: IMPLEMENTED (2026-08-04, schema v40). The fix half of the silent-
embedder-truncation defect; the detection half shipped as
`detect_embedder_window()` (commit 043e580). Pinned by the
`chunked_write_makes_the_tail_findable…` test family in engine/tests.rs
and the unit tests in vector/chunk.rs.

Deployment for an existing corpus (e.g. a production install at 73%
records-over-window): upgrade, `detect_embedder_window()` once, then
`rechunk_long_records()` once — both exposed in the Python binding.

## Why

A truncating embedder (all-MiniLM-L6-v2: 256 tokens) silently drops every
token past its window. Measured on a production install: 73% of records
exceeded the window; a verbatim fragment from a record's END retrieved its
parent 8% of the time (start: 28%). On a paraphrase-labeled query set over
the same corpus, production scored MRR 0.057. A chunked twin (200-token
windows, 40 overlap, dedupe to parent) scored MRR 0.395 — 7×, and beat
swapping to a cap-free static embedder on every metric. Chunking keeps the
stronger paraphrase model AND embeds the whole record.

## Shape

**One record, N vectors.** When a record's text exceeds the embedder's
known window, the engine splits it into overlapping character windows,
embeds each, and indexes every window under a synthetic key. Retrieval
maps chunk keys back to the parent rid and keeps the best-scoring window
per record — the record is findable from any part of its text, and the
API surface (recall results, get, stats counts of records) is unchanged.

- **Chunk 0 is the existing embedding.** `memories.embedding` stays the
  head vector under the plain rid key — byte-compatible with every
  existing DB, pack, and replica. Chunks 1..N are additive.
- **Chunk key encoding:** `"{rid}#c{idx}"` (idx ≥ 1). Rids are UUIDv7
  strings; `#` cannot appear in one. Keys are opaque strings at every
  index layer (verified: no UUID validation anywhere in the crate).
- **Durable storage:** new table

  ```sql
  CREATE TABLE memory_chunks (
      rid        TEXT NOT NULL,
      chunk_idx  INTEGER NOT NULL,      -- 1-based; 0 lives in memories
      embedding  BLOB NOT NULL,          -- same encryption as memories.embedding
      PRIMARY KEY (rid, chunk_idx)
  );
  ```

  Text is NOT duplicated into the table — chunks are derivable from
  `memories.text` + the recorded window/overlap parameters; only the
  vectors are stored (they are what a rebuild needs).

## When chunking activates

Chunking requires a known window. The window is known when:
1. `detect_embedder_window()` has run (binary-search probe, ~24 embeds), or
2. a persisted window exists in `meta` for the CURRENT embedder digest.

On successful detection the window is persisted to `meta`
(`embedder_window_chars`, keyed alongside `META_EMBEDDER_DIGEST`), so a
restart does not silently disable chunking. A digest change invalidates
the persisted window (a new embedder has a new window).

No window known → behavior identical to today (head-only vector +
truncation warning). Bundled potion/model2vec embedders are static and
cap-free — the probe reports no truncation and chunking never activates.

Writes that ARE chunked no longer fire the truncation warning; they
increment a separate `embedder_chunked_writes` counter surfaced in
`stats()` (the overflow is handled, not lost).

## Chunk geometry

- window = detected window in chars (e.g. ~1000 chars for MiniLM-256tok)
- stride = window − overlap, overlap = window / 5 (20%, mirrors the
  measured 200/40-token prototype)
- chunk count cap: 16 (a 1215-token max record needs ~5; the cap bounds
  delta-index capacity consumption and pathological inputs)
- chunks 1..N cover [stride, stride+window), [2·stride, …), … until end
  of text; a final short tail window is kept if ≥ overlap chars long.

## Write path (record_text)

The only entry that can chunk is the engine-embeds path (`record_text`):
`record()` receives a caller-supplied single vector and cannot know the
text/vector relationship. In record_text step 2 (embed outside the
guard), when the window is known and text exceeds it, embed all windows
under the SAME embedder snapshot; the revalidation loop (gen/digest
check) covers the whole set. Under the guard:

1. Reserve the parent (rid, seq) as today, PLUS each chunk key at the
   SAME seq — `(rid, seq)` pairs are the uniqueness unit in the delta
   index, so `rid@N` and `rid#c1@N` coexist and are individually
   addressable. Same-seq inheritance keeps the highest-seq-wins rule
   correct across corrections for free.
2. Guard: extend the reservation guard to carry the chunk keys
   (`BatchReservationGuard` already implements the N-key pattern:
   reserve-each / publish-each / remove-each-on-unwind).
3. In the one transaction: INSERT the `memory_chunks` rows next to the
   memories row.
4. Publish parent + chunks on commit. (Publish is per-key and not
   atomic across keys; the parent is published FIRST so the record is
   never findable by chunk while its row-lookup path would miss.
   A momentarily-unpublished chunk is only a recall miss, never a
   phantom.)

Capacity: chunks consume delta slots (delta_max 256). With the cap of
16 and median ~2–3 chunks for long records this is acceptable;
`record_batch` of long records is the pressure case and backpressure is
the existing, correct response.

## Recall path — collapse at the choke point

`DeltaIndex::search` is the single funnel for every engine consumer of
the main index (recall_inner, recall_profiled_inner, cognition
patterns). The chunk→parent collapse happens THERE, after the
delta+cold merge and before the final sort/truncate:

```
parent_of(key) = key up to '#c' suffix, else key itself
fold to min-distance per parent; sort; truncate(k)
```

Consequences:
- The `k` contract strengthens: callers receive k distinct PARENT rids
  (a record can never crowd the result list with its own windows).
- No caller changes: scoring-cache lookups (which silently drop unknown
  keys), RecallResult.rid, MMR's fetch_embeddings_by_rids, and
  patterns.rs's self-match all see only real memories rids.
- The cold-tier inner fetch keeps headroom for collapse (chunked
  records can eat candidate slots); recall's fetch_k (top_k × 20, cap
  500) absorbs the multiplicity — the same conservative shape the 7×
  measurement used.

The pack candidate path (`collect_pack_candidates`) calls its
`HnswIndex` directly, bypassing DeltaIndex — it gets the same collapse
inline before the pack scoring-table lookup. These are the only two
places.

Counters: `vec_index_entries` (stats) counts index ENTRIES and will
exceed record count when chunking is active — honest under its name; a
new `chunk_vectors` stat makes the difference legible. Delta capacity:
a chunked write consumes 1+N delta slots, so backpressure fires
proportionally sooner for long-record bursts — bounded by the 16-chunk
cap and the existing Backpressure/compaction machinery.

## Lifecycle

Every removal path that tombstones/removes the parent key must fan out
over its chunk keys. The chunk table is the enumeration source
(`SELECT chunk_idx FROM memory_chunks WHERE rid = ?`):

- forget / archive / replicated forget / conflict-loser suppression →
  tombstone each chunk key at the same seq, delete chunk rows.
- **correct/supersede (the subtle one):** new text re-chunks at a NEW
  seq under the SAME keys — highest-seq-wins shadows the old windows.
  If the new text yields FEWER chunks, the surplus old keys
  (`#c{new_n+1}..#c{old_n}`) must be explicitly tombstoned or they keep
  serving stale text forever. Chunk rows are replaced in the same tx.
- **correct with a CALLER-SUPPLIED vector** (`correct_with_embedding`,
  the Python-embedder path): the engine cannot chunk text it did not
  embed, so the old windows are purged (rows deleted, all old keys
  tombstoned) and the record becomes head-only until a
  `rechunk_long_records()` run repays it. (The scalar/metadata-only
  correct path never changes text — it re-routes any real text change
  to the re-embedding path — so it needs no chunk handling.)
- reembed (generation bump): chunks are regenerated from text under
  the new embedder — `DELETE FROM memory_chunks` inside the cutover
  savepoint, re-derive during the rebuild loop. No chunk staging
  column; chunks are a pure function of (text, window) and the window
  is re-detected per new embedder (digest change invalidates the
  persisted one).
- queued-write drain (`apply_materialize_record_post`) re-embeds text
  post-swap — it produces chunks too, same as record_text.
- repair (`repair_tool_call_artifacts`) re-embeds cleaned text — same.
- tier moves (hot→cold): chunk vectors follow the parent's tier —
  archive compresses chunk rows + tombstones chunk keys; hydrate
  decompresses + re-appends; the cold rebuild scan joins on
  memories.storage_tier so cold chunks never reappear at reopen.
- pack sealing: delete chunk rows orphaned by the namespace deletes,
  in the same transaction.

## Rebuild / cold index

`build_vec_index_with_enc` gains a second loop over `memory_chunks`
JOIN memories (same active/hot filter, ORDER BY rid, chunk_idx for
determinism), inserting under chunk keys. Same decrypt + decompress
defenses. `ensure_all_reachable` runs after, unchanged. Without this,
every reopen/rebuild/reembed silently drops chunk vectors — this is the
single biggest integration point.

## Replication

Chunk vectors are derived state — never a new op type, never in an op
payload. This mirrors existing behavior exactly:

- The `record` op does NOT carry the parent embedding (the replica
  materializes the row with `embedding IS NULL` and an out-of-band
  backfill re-embeds locally). Chunks follow: whatever supplies the
  parent vector supplies the chunks, from text + the replica's own
  embedder + its own detected window.
- The `correct` op carries exact bytes only when the follower runs the
  SAME embedder model; otherwise the follower re-embeds `new_text`
  locally. `apply_replicated_correct` re-chunks under whichever branch
  it takes, and tombstones surplus chunk keys — same rule as the
  leader.
- `materialize_forget` and conflict-loser suppression fan out chunk
  tombstones like the leader's forget.
- Older replicas skip unknown behavior safely: no new op types exist
  to skip.

## Packs

Pack mount builds its index with the SAME `build_vec_index_with_enc`
the host uses — the chunk loop there makes packs chunk-aware for free.
Specifics:

- The chunk loop must TOLERATE A MISSING `memory_chunks` TABLE: packs
  sealed by pre-chunk engines don't have it, structural vetting doesn't
  enumerate tables, and mounts are read-only (`query_only`) so
  derive-on-mount is impossible. Old packs mount head-only, unchanged.
- The pack content digest is blake3 over (rid, text) only — chunk rows
  cannot change any pack's digest or break verify-on-copy.
- Sealing is `VACUUM INTO` (carries the table automatically) followed
  by namespace deletes; `seal_pack` deletes chunk rows orphaned by
  those deletes in the same transaction. `memory_chunks` stays OUT of
  SCRUB_TABLES — its vectors are the pack's value.
- `collect_pack_candidates` searches the pack HnswIndex directly
  (bypassing DeltaIndex), so it applies the same parent collapse inline
  before its scoring-table lookup.

## Out of scope

- Chunking in `record()` (caller-supplied vectors — the caller owns
  the text/vector contract).
- Per-chunk text storage or per-chunk metadata.
- Query-side chunking (queries are short).
- Cross-encoder rerank / learned adapter (next items on the measured
  leverage list, independent of this).
