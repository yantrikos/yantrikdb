# yantrikdb-engine corpus

Every fact below is checked against the source tree at schema v38
(v0.10.1). Each `## ` block becomes one record. Blocks are written to
stand alone, because retrieval serves one record at a time.

## YantrikDB deletion is tombstoning, never a hard DELETE

`forget(rid)` in YantrikDB does not remove the row. It sets
`consolidation_status = 'tombstoned'`, removes the rid from the vector
index, unlinks it from the graph index, drops it from the scoring cache,
and marks any record_links as `broken_source_forgotten` or
`broken_target_forgotten`. There is no hard `DELETE` of a memory row
anywhere in the engine. A forgotten memory still occupies space in the
database file.

_cite: crates/yantrikdb-core/src/engine/lifecycle.rs:555-645_

## YantrikDB has no bulk delete and no delete-by-namespace

Deletion is per-rid only: `forget(rid)` and the cluster-deterministic
`tombstone_with_rid`. There is no `forget_by_namespace`, no
`forget_by_origin`, and no way to purge a whole namespace in one call.

## A YantrikDB database is a single SQLite file

The whole store — memories, embeddings, FTS index, graph claims, oplog,
sessions, meta — lives in one SQLite file opened in WAL mode. The `-wal`
and `-shm` sidecars are checkpoint-transient and vanish on clean close.
There is no separate vector-index file.

## The YantrikDB vector index is in RAM and rebuilt on every open

The HNSW index is never persisted. It is rebuilt from
`SELECT rid, embedding FROM memories` at every `open()`, ordered by rid
so the graph is deterministic across reopens. Opening a large database
therefore costs an O(rows) index build.

_cite: crates/yantrikdb-core/src/engine/indices.rs:10-53_

## YantrikDB namespaces are a plain column, not a registry

`namespace` is a `TEXT NOT NULL DEFAULT 'default'` column on `memories`
and about fifteen other tables. There is no namespaces table, no
`create_namespace`, no `drop_namespace`, and no `list_namespaces`. A
namespace exists exactly when some row carries the string. Blank or
whitespace namespaces are coerced to `'default'`.

## YantrikDB recall filters one namespace or all namespaces, never a set

Every recall entry point takes `namespace: Option<&str>`. `Some(ns)` is
exact string equality; `None` means all namespaces. There is no list, no
`IN (...)`, and no prefix or glob matching. Querying several namespaces
means several calls plus a caller-side merge.

## The YantrikDB vector index is global, so namespace filtering happens after search

There is one HNSW index per database covering every namespace. Namespace
filtering is applied to the approximate-nearest-neighbour results after
the search, not as an index restriction. A namespace holding one percent
of the rows still competes for the same candidate slots as everything
else.

## YantrikDB recall fetches twenty times top_k as candidates, capped at 500

The candidate pool is `fetch_k = (top_k * 20).min(500)`. That wide pool
exists so diverse, high-quality candidates survive the later MMR
diversity stage.

_cite: crates/yantrikdb-core/src/engine/recall.rs:298_

## The YantrikDB relevance score is 0.50 similarity, 0.20 decay, 0.30 recency

The base relevance is `W_SIM * similarity + W_DECAY * decay +
W_RECENCY * recency` with `W_SIM = 0.50`, `W_DECAY = 0.20`, and
`W_RECENCY = 0.30`. These are defaults; each database can learn its own
weights.

_cite: crates/yantrikdb-core/src/base/scoring.rs:208_

## YantrikDB gates the importance boost behind a similarity sigmoid

The final score is `base_rel * (1 + gate * ALPHA_IMP * importance) *
valence_boost`, where `gate = sigmoid(GATE_K * (similarity - GATE_TAU))`
with `GATE_K = 12.0` and `GATE_TAU = 0.25`. The gate stops a very
important but irrelevant memory from outranking a relevant one.

## YantrikDB decay is exponential in half-life, recency in a seven-day constant

`decay = importance * 2^(-elapsed / half_life)` and
`recency = exp(-age / 7 days)`. Decay uses the record's own half_life;
recency uses a fixed one-week constant.

## YantrikDB uses MMR with lambda 0.7 for result diversity

Maximal Marginal Relevance runs with `LAMBDA = 0.7`, skipping
near-duplicates above cosine 0.98. It only engages when the candidate
pool is at least `max(top_k * 3, 20)` — below that there is nothing to
diversify.

_cite: crates/yantrikdb-core/src/engine/recall.rs:1943_

## YantrikDB reserves three recall slots for keyword matches

Up to `KEYWORD_RESERVE_SLOTS = 3` result slots are reserved for FTS
keyword matches with similarity at least 0.25 that would otherwise rank
below the cutoff. They are boosted just past the cutoff and are exempt
from the MMR diversity penalty, so topic-relevant but low-importance
matches survive.

## YantrikDB write-time importance calibration deflates saturated namespaces

Per-namespace calibration tracks an EWMA of importance. Once a namespace
passes `MIN_COUNT = 8` writes with a mean above
`SATURATION_THRESHOLD = 0.80`, further high marks are compressed into
`[HIGH_FLOOR = 0.70, ceiling]` where the ceiling falls toward
`MIN_CEILING = 0.75`. Writing everything at importance 1.0 therefore
ranks your ninth critical fact *below* your first eight.

_cite: crates/yantrikdb-core/src/engine/importance.rs:47-56_

## YantrikDB namespace write counters never decrement on forget

`namespace_importance_stats.count` is a cumulative write counter used for
importance calibration. It never decreases when a memory is forgotten, so
bulk-loading and then deleting a large body of records permanently shifts
that namespace's calibration.

## YantrikDB's bundled embedder is potion-base-2M at 64 dimensions

The default embedder is `potion-base-2M`, compiled into the binary via
`include_bytes!`, producing 64-dimensional vectors. `BUNDLED_EMBEDDER_DIM`
is 64. No network access is needed at install or at runtime.

## YantrikDB records embedder identity in the meta table

A database stores `embedder_name`, `embedder_digest`, and `embedder_dim`
in its `meta` table. The identity is stamped when the engine itself
produces a vector, not when an embedder is attached — attaching says what
the engine can encode, not what built the vectors already stored.
`adopt_embedder_identity()` is the explicit operator assertion for
databases holding externally-computed vectors.

## Changing a YantrikDB embedder at the same dimension is refused, not accepted

`set_embedder` returns `ChangeEmbedderDigestRequiresReembed` when the
candidate's fingerprint differs from the one that built the index, even
though the dimensions match. Same-dimension-different-model is the
silent-corruption case: queries encode in one space, stored vectors live
in another, nothing panics and results are quietly wrong. The correct
path is `reembed()`.

## YantrikDB HNSW defaults are M=16, ef_construction=200, ef_search=50

These are the parameters the engine constructs its initial SearchState
with. They are carried on SearchState rather than only on the index so a
re-embed can preserve or override them independently.

## YantrikDB read connections come from a pool of four by default

The engine opens a pool of read connections against the same file, sized
by the `YANTRIKDB_READ_POOL` environment variable, defaulting to 4.
Setting it to 0 routes reads through the write connection. In-memory
databases skip the pool entirely, because each `:memory:` open would be a
different database.

## The YantrikDB provenance gate has three modes and fails closed

`GateMode` is `Off`, `Warn`, or `Enforce`. Fresh installs default to
`enforce`; databases migrated from older versions default to `warn`. A
malformed persisted mode is a typed error rather than a silent `Off`.

## The YantrikDB provenance gate blocks inference laundering itself as fact

The gate refuses a write whose declared provenance is internally
inconsistent — for example `source=inference` claiming `kind=fact`
without a confirmation or verification basis. It catches *declared*
contradictions only: a caller that lies about its source is undetectable.

## YantrikDB idempotency claims are keyed by origin, namespace and key

The `idempotency_claims` table has primary key
`(origin_actor, namespace, idempotency_key)`, and an
`INSERT ... ON CONFLICT DO NOTHING` on that key is the serialization
point. Keys must be non-empty after trimming and at most 512 bytes.

## Reusing a YantrikDB idempotency key with different content is a typed error

Same key plus identical payload returns the original rid with nothing
re-written. Same key plus a *different* payload raises
`IdempotencyConflict` — the first write's content stands, and the fix is
to change the key or the payload. In a batch, one conflict fails the
whole batch; batches stay all-or-nothing.

## YantrikDB idempotency claims are not reaped by forget

Nothing updates or deletes a claim row. Retrying a key whose record was
since forgotten returns the tombstoned rid, which is the honest answer:
the write happened exactly once and forgetting it was a separate later
act. Re-recording forgotten content requires a new key.

## memories.origin_actor in YantrikDB is an idempotency scope, not a provenance field

Despite the name, `memories.origin_actor` is only written for writes that
carry an idempotency key, and it is hardcoded to the local actor — a
caller cannot supply a foreign origin. `record_with_rid` does not write it
at all. It shares a name with the oplog's genuine provenance column but
not its meaning.

## The YantrikDB WriteAdmission enum has exactly two variants: Origin and Admitted

Name both variants of the `WriteAdmission` enum and they are `Origin`
and `Admitted`. There is no third variant and no default.

`WriteAdmission` is `Origin` or `Admitted`. `Origin` means a fresh write
that must run the provenance gate; `Admitted` means a consensus-committed
operation being re-applied, which must *not* be re-gated — re-gating a
committed entry would wedge a cluster. It is a required argument, so an
origin caller who forgets gets a compile error rather than a silent
bypass.

## The YantrikDB substitution-category member evidence ladder, ranked strongest to weakest

The three source values, ranked from strongest to weakest, are `seed`,
then `user_confirmed`, then `llm_suggested`. That is the whole ranking:
strongest is `seed`, weakest is `llm_suggested`.

The member evidence ladder is `seed` (3) > `user_confirmed` (2) >
`llm_suggested` (1), with unknown runtime sources treated as 2. Only
`llm_suggested` lands a member as `pending`, which makes it invisible to
lookups. Promotion never demotes.

## YantrikDB recall ranking ignores certainty and source

The scoring function uses similarity, decay, recency, importance and
valence. It does not use certainty, provenance source, correction
lineage, or confirmation history. "Trusted facts outrank untrusted ones"
is true of YantrikDB's write path and false of its retrieval path.

## YantrikDB excludes superseded records from recall by default

When the status read policy is active, records superseded by a newer
revision leave the candidate pool before slot reservation and MMR, so a
stale fact can never outrank or crowd out its own successor.
`include_superseded = true` re-admits them for history queries.

## Use chain_head, not recall, for the current value of a changing fact

Similarity search returns the most *similar* record, which for a value
that changed over time is often a stale revision. `memory(action=
"chain_head", namespace=...)` returns the actual current value of a
chain-shaped namespace.

## YantrikDB full-text search is disabled on encrypted databases

Lexical FTS5 matching over `memories_fts` is skipped entirely when the
database is encrypted, because the stored text is ciphertext. Encrypted
databases fall back to vector similarity alone.

## YantrikDB cold-tier embeddings are zstd-compressed in the same column

Archiving moves a memory from hot to cold storage by compressing its
embedding in place in the `embedding` column and removing it from the
vector index. Compressed blobs are detected by a four-byte magic check
rather than by the tier column, so tier drift cannot corrupt an index
rebuild.

## YantrikDB procedural memory is ordinary rows with type='procedural'

There is no separate skills table. `record_procedural` writes a normal
`memories` row with `memory_type = "procedural"`, importance set from the
effectiveness score, and a four-week half-life.
`surface_procedural` is literally `recall()` filtered to
`memory_type = "procedural"` — same ranking, same miss risk.

## YantrikDB has no pinned or always-inject recall tier

Every retrieval path is top_k similarity. There is no engine primitive
that guarantees a specific memory is surfaced, which means a
hard-constraint rule can be stored correctly and still never reach the
model. Callers needing guaranteed constraints must load them
unconditionally themselves.

## A YantrikDB pack is a sealed single file mounted read-only

`seal_pack` produces a `.ydbpack`: a `VACUUM INTO` copy scoped to one
namespace, with tombstones dropped, host-private tables scrubbed, a
manifest and blake3 content digest written into its own `meta` table, and
`journal_mode=DELETE` so it has no WAL sidecar and opens read-only from
anywhere.

## Mounting a YantrikDB pack does not modify the host database

`mount_pack` opens the pack read-only and builds its own HNSW index and
scoring cache. `unmount_pack` drops that handle. The host file is
byte-identical before and after, which is why mounting is reversible in a
way that importing rows and later deleting them is not.

## YantrikDB refuses to mount a pack from a different embedding space

The query is encoded once, by the host's embedder, and searched against
both indexes — so a pack built by a different model returns confident
nonsense rather than an error. `mount_pack` raises
`PackEmbedderMismatch` unless it can prove the spaces match. Dimension
mismatch is always fatal.

## YantrikDB's allow_unverified_embedder covers unproven, never proven-wrong

The mount override applies when compatibility is *unknown* — a legacy
host with no recorded embedder identity, or a manifest with no declared
digest. When both sides declare identities and they disagree, the mount
stays fatal: a flag that buys a known-bad mount is not a safety valve.

## YantrikDB pack results are ranked below host results by a trust tier

Pack candidates are scored with the host's learned weights and then
multiplied by a trust tier — 0.85 signed, 0.75 unsigned, 0.60 unverified
— while host rows keep 1.0. A pack fact must be meaningfully more similar
than one of yours to outrank it.

## A YantrikDB host record can supersede a pack record

Linking a host record to a pack rid with `supersedes` removes that pack
row from the candidate pool. The edge outlives the mount: it dangles
harmlessly while the pack is unmounted and re-applies on remount, so user
corrections survive detach and pack upgrades.

## YantrikDB pack rows reach recall through vector similarity only

A mounted pack contributes candidates from its own HNSW index. Pack rows
do not currently participate in FTS keyword matching, and graph expansion
does not cross the mount boundary.

## YantrikDB pack manifests live in the pack's own meta table

The manifest is stored as JSON under the `meta` key `pack_manifest`,
which makes the file self-describing. The content digest is blake3 over
each `(rid, text)` pair in rid order, length-prefixed, and is
re-verified at mount — a pack edited after sealing is refused.

## YantrikDB seal_pack refuses to overwrite an existing file

Sealing to a path that already exists raises `PackDestinationExists`, so
a mounted pack can never be rewritten underneath its own reader.

## The current YantrikDB schema version is 38

`SCHEMA_VERSION` is 38 as of v0.10.1. Version 37 added the
`origin_actor`, `idempotency_key` and `confidence_basis` columns to
`memories` plus the `idempotency_claims` table; version 38 was an oplog
index fix.
