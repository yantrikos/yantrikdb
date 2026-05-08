# Phase 4.3 — SQL writes off foreground (design memo)

**Status:** in_progress (saga task 3, epic 1)
**Scope:** narrow — only the unbounded SQL loops in `record()` / `record_with_rid()` move to the materializer. Fixed-cost SQL (memories INSERT, session UPDATE) stays on foreground.
**Target:** v0.7.0
**Author:** 2026-05-08

---

## Why narrow scope

The wedge audit (`docs/wedge_lock_scope_audit_2026-05-06.md`) named two
primitives. v0.6.6 fixed primitive #1 (vec_index lock). Primitive #2 is
foreground SQL holding `Mutex<Connection>` for unbounded time as the
extracted-entity / extracted-relation loops grow.

The fixed-cost SQL on foreground is ~3 statements (memories INSERT,
optional session UPDATE × 2). That is bounded. The wedge bites at the
*loops* — N entities × 2 INSERTs each + K relations × (1 SELECT + 1
ingest_claim transaction). Real-world `record(text="...")` calls with
50-token sentences extract 5-15 entities + 0-3 relations regularly,
which means 10-30+ SQL statements held under `conn().lock()` per
record.

Moving ONLY those loops off foreground preserves the synchronous
"`record()` returns rid" contract while bounding the conn-hold time
to fixed-cost work.

---

## Contract changes

### Visible to callers

- `record()` / `record_with_rid()` still return `Result<String>` /
  `Result<()>` (rid is the input or the generated id). No new error
  variants.
- **The synchronous read-after-write window changes.** Previously, after
  `record()` returned, `db.entity_profile("Alice")` saw the new memory
  immediately if "Alice" appeared in the text. After Phase 4.3, that
  query may not see it for a few ms (until materializer drains the post-record op).
- Recall path is unaffected — the delta append happens synchronously
  on foreground, so `recall(text=...)` finds the new memory immediately.
  Only the entity-graph and relation-graph paths have a brief
  materialization lag.

### Strict read-after-write callers

Callers needing strict read-after-write semantics for entity/graph
queries already have the Phase 6 RYW primitive — `record()` returns
`(rid, seq)` (TODO if not already), and `wait_for_visible_seq(ns, seq)`
gates on the watermark. Phase 4.3 must bump `visible_seq[ns]` ONLY
after the materializer applies the post-record op, not on foreground
return. That is the change that makes the RYW gate cover entity-graph
visibility.

**Open question 1.** Today `record()` bumps `visible_seq` on foreground
return (after delta append). After Phase 4.3, do we move the bump to
the materializer (correct for entity-graph RYW, but breaks
recall_with_seq's contract — recall reads delta which IS visible
post-foreground)?

**Resolution candidate.** Two seqs: `visible_seq_recall[ns]` (bumped
on foreground delta append, gates `recall_with_seq`) and
`visible_seq_graph[ns]` (bumped on materializer apply, gates a future
`entity_profile_with_seq`). For now ship Phase 4.3 with single
`visible_seq` bumped on foreground (preserve current contract),
document the entity-graph lag, defer the dual-seq design until a
caller needs it. **(This is the v0.7.0 ship plan.)**

---

## Op shape

New oplog op_type: `"materialize_record_post"`.

Payload (JSON):

```json
{
  "rid": "01HX...",
  "text": "<plaintext for entity extraction; encrypted only if engine encrypted>",
  "namespace": "default",
  "ts_secs": 1715184000.0,
  "domain": "general",
  "source": "user"
}
```

Worker dispatch arm in `apply_pending_ops_once` does, for each pending
`materialize_record_post` op:

1. Decrypt `text` if engine encrypted.
2. `extract_heuristic_entities(text)` → seed entities table (loop A).
3. Compose candidate set: heuristic + entity-name match.
4. INSERT OR IGNORE into `memory_entities` for each candidate (loop B).
5. Take `graph_index.write()` lock briefly, add candidates + link memory.
6. `extract_heuristic_relations(text, candidates)` → for each, dedup
   SELECT + ingest_claim if missing (loop C+D).
7. Emit extraction audit telemetry (already a `tracing::info!` event,
   moves verbatim).
8. mark_op_applied.

All work happens on the materializer thread. Foreground does NONE of
this synchronously.

---

## Implementation plan (2 commits)

### Commit A — dispatch arm, no foreground change

- Add a const `OP_MATERIALIZE_RECORD_POST: &str = "materialize_record_post";`
  in `engine/stats.rs` or a new `engine/op_types.rs`.
- Extend `apply_pending_ops_once` match arm to recognize the new
  op_type and call a new private fn `apply_materialize_record_post(payload: &str) -> Result<()>`
  that runs the 8 steps above against `&self`.
- Add unit tests:
  - `materialize_record_post_inserts_entities`: enqueue an op with
    text="Alice met Acme", drain via apply_pending_ops_once, assert
    entities table contains Alice and Acme.
  - `materialize_record_post_inserts_memory_entities`: same shape,
    assert memory_entities row count == 2.
  - `materialize_record_post_idempotent_on_replay`: enqueue twice with
    same op_id, assert no double-insert.
  - `materialize_record_post_emits_relation_claims`: text with
    relation pattern, assert claims row appears.
  - `materialize_record_post_concurrent_workers_no_double_apply`: 4
    workers + 50 ops, assert exactly-once.
- Foreground is **untouched**. The new dispatch is dead code on the
  production path until Commit B flips it. This is intentional —
  validates correctness in isolation.

### Commit B — flip foreground to enqueue (follow-on session)

- `record()` body: keep memories INSERT + session UPDATE +
  vec_index.append + scoring_cache.insert; replace inline
  entity/relation loops with `log_op_pending("materialize_record_post", Some(&rid), &payload, None, None)`.
- Same surgery for `record_with_rid()`.
- `record_batch()` accumulates one materialize op per record (or one
  batched op if we add a batch op_type — defer for now).
- Empirical validation: rerun `wedge_repro --writers 32 --readers 8`,
  confirm write throughput rises and read latency stays bounded under
  load.
- Update `CONCURRENCY.md` Rule 4 ("never hold conn across…") to reflect
  the new shape.

### Out of scope for this phase

- `forget()` / `correct()` / `archive()` / `hydrate()` — these are
  bounded SQL (1-2 statements), don't need materializer routing.
- `record_batch()` batched op_type optimization — works under the
  per-op routing; batching is a follow-on perf delta, not a
  correctness fix.
- Dual-seq for entity-graph RYW — Open question 1 above.

---

## Test strategy

- Unit tests on the dispatch arm (Commit A).
- Integration test: 1000-record run with extracted_entities, verify
  graph_index has expected nodes after `count_pending_ops() == 0`.
- Stress test: extend wedge_repro with a `--with-extraction` flag
  that injects realistic text (5-15 entities each), measure conn-hold
  time before and after.
- Regression: full lib suite (currently 1399) must stay green at
  every commit.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Caller relies on synchronous entity-graph visibility post-record | Document the lag in CHANGELOG + add `wait_for_visible_seq` example to docs |
| Materializer falls behind under sustained ingest, entity-graph stale for seconds | Bound oplog `applied=0` count via existing `MAX_PENDING_OPS=10_000`; backpressure surfaces as `Error::Backpressure` to caller |
| Encrypted DB: text plaintext lives in oplog payload until materialized | Encrypt the text field same as `memories.text` does; payload stays sealed at rest |
| Materializer crashes mid-dispatch | Each op is idempotent (INSERT OR IGNORE on every loop); restart resumes via SELECT applied=0 ORDER BY hlc; race-safe across N workers via UPDATE WHERE applied=0 filter |

---

## References

- [docs/decoupled_write_path_rfc.md](decoupled_write_path_rfc.md) — Phase 4.3 was named here.
- [docs/wedge_lock_scope_audit_2026-05-06.md](wedge_lock_scope_audit_2026-05-06.md) — primitive #2 identification.
- [CONCURRENCY.md](../CONCURRENCY.md) Rule 4 — the invariant being enforced.
- Saga task 3 (epic 1).
