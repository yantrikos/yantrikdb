# Evidence-Versioned Synthesis Lifecycle

Status: v43 lifecycle substrate and the Python query-free topic and concern
contracts are implemented. The paired provenance-routing evaluation is
complete and null on accuracy; automatic model discovery and item assembly
remain gated on their separate frozen paired evaluation.

## Problem

`record_synthesis` already stores a useful additive representation:

- source records remain active;
- `created_at` is the newest evidence time (availability);
- `metadata.first_mention_at` is the earliest evidence time (ordering);
- `evidence_ids`, axis, granularity, and namespace are preserved;
- retries are idempotent for identical model output.

Before v42, the missing contract was lifecycle. `consolidation_members` recorded
membership, but recall did not consult evidence state. Correcting or forgetting
a source could therefore leave an old synthesized interpretation eligible. v42
adds typed state, revision-bound dependencies, atomic invalidation, and
evidence-versioned generation keys; the remaining gate is measured generation
quality and bounded unattended scheduling.

The prototype formerly had a second admission bug: `record_*` could route
through the re-embedding queue and return a reserved rid before a `memories`
row existed, after which `commit_consolidation` continued with dependent side
effects around that absent row. Consolidation now uses sync-only record routes
and fails with `ConsolidationDeferredDuringReembed` before durable mutation.
That closes the queued-row corruption path, but the synchronous record and its
dependent side effects still need to become one combined production operation.

## Invariants

1. A synthesis is eligible only while every direct evidence dependency exists,
   is active, and still has the revision observed when synthesis ran.
2. Forgetting evidence must immediately make all transitive dependent
   syntheses ineligible. No derived text may survive as a back door to forgotten
   content.
3. Correcting evidence must immediately make prior dependent syntheses
   ineligible. Regeneration may later restore coverage, but recall must never
   serve a stale interpretation while waiting.
4. Invalidation must add no SQLite writes or dependency joins to recall. A
   correction/forget may perform one indexed, set-based invalidation of bounded
   dependents; it must never recurse through the synthesis graph.
5. A retry over the same evidence revision and payload returns the same rid. A
   changed payload over the same evidence revision conflicts. A new evidence
   revision can create a new synthesis generation without deleting an immutable
   idempotency claim.
6. Namespace is a hard boundary. Every dependency and generated record belongs
   to one namespace.
7. Atomic and rollup items remain simultaneously addressable. Query-time
   granularity selects among representations; it does not rewrite storage.

## Durable Substrate

Axis, granularity, logical identity, and evidence version must be typed nullable
columns on `memories`, not metadata-only fields:

```sql
ALTER TABLE memories ADD COLUMN synthesis_axis TEXT;
ALTER TABLE memories ADD COLUMN synthesis_granularity TEXT;
ALTER TABLE memories ADD COLUMN synthesis_logical_key TEXT;
ALTER TABLE memories ADD COLUMN synthesis_evidence_version TEXT;
ALTER TABLE memories ADD COLUMN synthesis_state TEXT;
```

Recall ranks before final text/metadata hydration, and encrypted stores do not
expose JSON metadata to SQL. Typed columns make the representation available to
every candidate lane without decrypting the corpus or trusting caller metadata.
NULL means an ordinary record.

These columns deliberately reveal limited structure beside encrypted payloads:
an observer with database-file access can distinguish synthesized records,
their axis/granularity, lifecycle state, and logical-generation distribution.
They still cannot read the synthesized text, metadata, or evidence content.
This leakage is accepted so ranking and lifecycle correctness do not silently
disappear when metadata encryption is enabled, and it must be documented on
the encryption surface.

Add a dedicated dependency table rather than overloading the set-union CRDT
semantics of `consolidation_members`:

```sql
CREATE TABLE synthesis_dependencies (
    synthesis_rid TEXT NOT NULL,
    source_rid TEXT NOT NULL,
    source_revision_num INTEGER NOT NULL,
    namespace TEXT NOT NULL,
    is_direct INTEGER NOT NULL,
    PRIMARY KEY (synthesis_rid, source_rid)
);
CREATE INDEX idx_synthesis_dependencies_source
    ON synthesis_dependencies(namespace, source_rid);
```

`record_synthesis` inserts the engine-resolved dependency closure: every direct
source plus every transitive leaf evidence record. `is_direct` distinguishes
the caller-named inputs from inherited leaves. Closure is resolved through the
engine-owned `synthesis_dependencies` graph, never by trusting a caller's
`metadata.evidence_ids`. This closes a provenance hole in the current
prototype: an ordinary record can carry arbitrary metadata and must not make a
synthesis claim evidence that was never linked by the engine. Tracking both
direct sources and flattened leaves means a corrected raw leaf invalidates all
descendants, while forgetting or correcting an intermediate synthesis also
invalidates a rollup that depended on its interpretation. No recursive recall
query is required.

`source_revision_num` is the maximum authoritative `record_revisions` number
observed at admission (zero for an uncorrected record). `updated_at` is not a
valid substitute in the current engine: metadata/scalar corrections append a
revision but do not advance `memories.updated_at`, while text corrections do.
The revision number is already carried through correction replication and
captures every correction shape. First-mention and availability clocks are
derived from this engine-resolved closure, not inherited from arbitrary source
metadata.

Admission is one operation: idempotency claim, memory row, typed synthesis
columns, dependencies, and authoritative record oplog payload either all commit
or none do. `consolidation_members`, entity transfer, and the separate
consolidate oplog remain post-admission bookkeeping; a failure there cannot
create an eligible synthesis without its descriptor or evidence closure, but
that bookkeeping still needs a resumable combined operation before destructive
consolidation can be treated as fully atomic.

Writing dependencies first was considered and rejected for the current engine.
The rid is minted inside `record_under_guard_and_state`, idempotent races can
resolve to a different pre-existing rid, and the queued re-embedding route does
not create a memory row at admission. Reversed autocommit ordering would
therefore strand dependencies on losing or queued rids. The production path
needs a transaction-aware record extension on the synchronous route and a
single combined queued/materialized operation on the re-embedding route.

The oplog payload carries the typed synthesis descriptor and complete
dependency closure. A follower materializes the row, descriptor, and
dependencies in one transaction. A synthesis descriptor with zero dependency
rows, malformed closure, namespace mismatch, or an already-mismatched source
revision lands with lifecycle state `unverified` and is excluded from recall
even though its ordinary consolidation status remains `active`. Out-of-order
rows are promoted to `verified` only after all exact source revisions arrive;
rollup promotion cascades through newly verified children. This makes
replication lag and out-of-order delivery fail closed without sacrificing
eventual convergence.

The engine derives an `evidence_version` from the canonical ordered sequence of
`(source_rid, source_revision_num)` pairs. The effective idempotency key is:

```text
<caller logical key>:evidence-v1:<evidence_version>
```

The caller key identifies the logical item; the suffix identifies one immutable
generation. The engine returns both values and stores them as engine-owned
metadata. Same-generation stochastic drift remains an idempotency conflict.

## Eligibility And Invalidation

An always-on anti-join was rejected for the hot path. It would silently make
every recall a post-filtered recall, forcing the current fetch planner to scan
small indexes exhaustively. At 10,000 candidates and 5-20 dependencies per
rollup it also creates 50,000-200,000 dependency comparisons per call.

Instead, `synthesis_state` is one of `verified`, `invalidated`, `unverified`, or
`superseded` (NULL for ordinary records). Only `verified` syntheses are recall
eligible. The state is loaded into the scoring cache, checked by the shared lane
predicate, and checked again on the merged pool before keyword reservation/MMR
and final `top_k` in both ordinary and profiled recall. The final boundary
prevents an older or future lane with a partial filter copy from leaking an
invalidated synthesis. There is no
per-recall dependency query and synthesis does not automatically set
`has_post_filters` for every database.

Correction and forget perform one indexed statement in the same transaction as
their source lifecycle change:

```sql
UPDATE memories
SET synthesis_state = 'invalidated'
WHERE rid IN (
    SELECT synthesis_rid
    FROM synthesis_dependencies
    WHERE namespace = ?1 AND source_rid = ?2
)
AND synthesis_state = 'verified';
```

Because the table stores the dependency closure, this one statement invalidates
direct and transitive dependents without graph recursion. Post-commit cache
updates mirror the affected rid set before the lifecycle epoch is published.
Follower correction/forget materializers execute the same statement.

The `forget` prerequisite is now repaired: the memory tombstone, durable chunk
purge, entity unlink, link-status updates, and applied oplog row share one
transaction, with vector/cache publication remaining post-commit under the
existing write-router guard. An adversarial trigger test forces the oplog insert
to fail and proves all earlier durable projections roll back. Dependent
invalidation can therefore join this transaction without reopening the old
crash window.

Local write amplification is now admission-bounded: one evidence record may
back at most `synthesis_fanout_cap` verified synthesis generations (default
64, durably configurable with `set_synthesis_fanout_cap`). The authoritative
count, persisted cap read, and refusal run inside the synthesis record
transaction after idempotency resolution, so already-open handles observe a
cap changed through another connection. Exactly-at-cap is legal; the next
distinct generation
returns typed `SynthesisFanoutLimit`, and the memory row, dependency edges,
idempotency claim, oplog rows, and vector reservation all roll back. An
identical retry still resolves to the prior rid even when the evidence is at
capacity.

Every stored dependency consumes fan-out because every
`synthesis_dependencies` edge creates the same correction/forget invalidation
obligation. `is_direct` records provenance shape; it does not exempt an edge
from invalidation work.

`stats()` exposes the configured cap, local refusals since boot, current
high-water, sources exactly at cap, and sources over cap. Replication remains
convergence-first: a follower does not discard an already-durable origin write
merely because its local cap is lower, and reports that state through the
over-cap counter instead. The available BEAM atomic-synthesis artifacts contain
148 deduplicated source observations with median/p95/max fan-out all equal to
2, so 64 is a generous safety default rather than a tuned quality threshold.
Broader corpus distribution and correction/forget p50/p95/p99 stress
measurement remain required before automatic background synthesis is enabled.

Every synthesized row must have at least one direct and one leaf dependency.
Missing dependencies are invalid, not legacy-compatible.
`audit_synthesis_evidence(namespace, max_issues)` now rechecks every active
`verified` row against provenance shape, source existence/status/namespace,
source synthesis state, current revision numbers, dependency cycles, and
duplicate active logical generations. The report also counts orphan dependency
rows and sources above the local fan-out cap after convergence-first
replication. Its candidate count is exact, its diagnostic sample is bounded,
and it is report-only so detecting legacy or direct-SQL damage cannot create an
unreplicated repair. Because exact counting and cycle detection are a full-store
scan, it remains an explicit operator call rather than adding unbounded work to
every routine maintenance cycle. The synchronous correction/forget
transactions and replication materializers remain the foreground correctness
boundary.

## Query-Dependent Granularity

Recall classifies only the requested representation shape:

| Query shape | Preferred records |
|---|---|
| exact fact, date, quote, who-said | raw evidence or atomic synthesis |
| ordered list of concerns/milestones | concern items with occurrence provenance |
| summary/theme/progression | rollup synthesis, with atomic evidence available |
| ambiguous | mixed candidates; diversity prevents one axis from monopolizing k |

The public `organize_evidence` and `organize_concerns` paths keep model policy
at the application boundary. A caller discovers stable `TopicHandle` or
answer-sized `ConcernItem` values; YantrikDB validates bounded many-to-many
evidence membership and rejects invented evidence IDs. Topic organization can
assign omitted evidence by embedding similarity. Both representations persist
through `record_synthesis`. Persisted metadata includes ordered child rids,
anchor entities, and an evidence timeline with occurrence, span-end,
availability, first-mention turn, and date-source values.

`recall_organized` chooses raw, topic-handle, or expanded-item presentation from
the requested query shape. Selection remains relevance-first and presentation
ordering happens afterward. Conversation-history questions are a distinct
temporal axis: `metadata.first_mention_turn` orders when the user disclosed an
item, while `metadata.first_mention_at` orders the real-world event. If event
time is absent, presentation falls back to `created_at`.

Schema v44 adds an explicit rollup-outcome protocol for learning the grouping
policy from real use. `rollup_impressions` freezes the surfaced rollup's hashed
query, namespace, rank, and score. Expansion and child selection/correction are
separate, ID-bound events, so a later recall of the same rollup cannot steal the
outcome and a generic `get(child_rid)` cannot silently label a parent. Retries
are idempotent; reusing an impression ID with a different payload fails closed.
Schema v45 adds `finalize_rollup_outcome`, an exact complete-set boundary:
corrected children imply selection, every child must belong to the recorded
expansion, exact retries are idempotent, and changed payloads fail closed. Until
finalization, the absence of a child selection is unknown telemetry rather than
a negative label.

`rollup_outcome_report(namespace, since)` reads this ledger without mutation.
Its per-rank rates and readiness gate use only finalized outcomes. Offline
evaluation readiness requires 200 finalized impressions spanning 30 queries
and 20 rollups, at least 50 selected and 50 explicitly unselected children,
80% completion among expanded impressions, and no query or rollup contributing
more than 25% of the finalized cohort. This gate authorizes an offline
evaluation only. Promotion into ranking labels or learned features still
requires evidence that the outcome predicts usefulness without introducing
exposure bias.

Local frozen-store probes show why the middle representation is required.
Atomic records can split one concern into adjacent follow-up actions, while a
topic handle can combine a dozen concerns. Hierarchical query-free discovery on
q9 produced 73 durable concerns from 138 atomic records; deterministic singleton
completion preserved full evidence coverage. Q10 produced 102 merged concerns
from 122 raw candidates, covering 131 of 171 atomic records before singleton
completion, with no invented evidence IDs.

Concern-to-topic expansion recovered exact chronological gold support for both
q9 prompts. Multi-token focus plus handle-anchor filtering reduced the q9 family
query to exactly turns `[22, 76, 118, 208, 260]`; the local answer model matched
all five expected support items. Explicit-entity focus, child-provenance rescue,
and entity-scoped concern deduplication reduced the q10 Carla query to exactly
`[52, 78, 176, 228, 230]`; the same answer model matched all five expected
collaboration items. Generic q9 personal-statement and q10 writing-journey
prompts remain the honest boundary: several stored concerns are equally valid
under their wording, and a small answer model can choose plausible non-gold
items even when all gold evidence is present. Automatic discovery and broad
selector policy therefore remain evaluation gates, while the validated concern
persistence and recall substrate is public.

Axis and granularity are ranking features, never hard filters unless the caller
explicitly requests one. A wrong classifier must degrade ranking rather than
erase evidence. Both the ordinary and profiled recall implementations must use
one shared predicate so the diagnostic path cannot drift from production.

The first implementation loads `synthesis_axis` and
`synthesis_granularity` into the scoring cache but computes preference only on
the per-query result pool; no query-dependent score is cached. Token-boundary
signals prefer `atomic` for list/order/item requests and `rollup` for
summary/overview/theme requests. A query carrying both signal families is an
explicit conflict and receives no granularity boost. Phrase-led `asked`,
`contributed`, and `who_said` axes are supported; axes without a measured query
class remain neutral.

Only an exact label match on a lifecycle-`verified` synthesis changes its
score: granularity uses a 1.06 multiplier and axis uses 1.04. There is no
mismatch penalty, raw records are unchanged, and neutral queries are exact
no-ops. Matches are reported as
`representation_match:granularity=<label>` and
`representation_match:axis=<label>` in `why_retrieved`. A 400-question AMB
classifier preflight selected atomic+contributed for all 40 event-ordering
queries, rollup for 38 of 40 summarization queries, and stayed neutral on
granularity for 38-40 of 40 queries in every other category. The two mixed
summarization queries are the conflict-neutral cases.

## Regeneration

A maintenance scan uses `idx_synthesis_dependencies_source` to enumerate
invalid generations and schedule their logical item keys. Generation is
write-new, never in-place mutation:

1. load current active evidence;
2. generate atomic items globally across the concern cluster;
3. validate every evidence id and namespace;
4. write the new evidence-versioned generation;
5. mark prior verified generations for that logical key `superseded` in the
   same transaction;
6. retain only a bounded number of invalidated/superseded generations in the
   live tables, with full history remaining in the oplog.

The first production version should expose detection and explicit regeneration
before enabling unattended background generation.

## Required Proofs

- correction removes direct and rollup descendants from recall immediately;
- forget removes every transitive synthesis and no derived text is retrievable;
- reopening and replication preserve the same eligibility result;
- same evidence + same output is idempotent;
- same evidence + changed output conflicts;
- corrected evidence + regenerated output creates one new generation;
- cross-namespace dependencies are rejected;
- failure between any two admission statements leaves neither a live synthesis
  nor an orphan idempotency claim;
- queued re-embedding admission never exposes a rid whose synthesis descriptor
  or dependencies are absent;
- follower materialization is atomic and missing dependencies fail closed;
- encrypted stores rank axis/granularity identically to plaintext stores;
- ordinary recall does no dependency SQL and does not become exhaustive merely
  because the database contains syntheses;
- correction latency and write amplification stay within the admitted fan-out
  bound at the measured p50/p95/p99 corpus distribution;
- superseded-generation retention is bounded and preserves oplog auditability;
- event-ordering recall selects atomic items and orders by
  `first_mention_at`, while summary queries can prefer rollups;
- the completed frozen V1/V2 result is treated as null on accuracy and used
  only to justify provenance correctness and context efficiency;
- the V2/hybrid paired evaluation passes its separately pre-registered gate
  before automatic synthesis is enabled.
