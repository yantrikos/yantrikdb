//! Canonical oplog `op_type` string constants.
//!
//! Names are stored verbatim in the `oplog.op_type` column. Treat each
//! constant as a wire/format value: changing a string here is a
//! schema-shaped change and needs a migration story for any deployment
//! that has pending (`applied=0`) oplog entries on disk.
//!
//! Materialization-routed op_types (Phase 4.3+) live here; legacy op
//! types that the engine still applies inline are recognized by literal
//! string in `apply_pending_ops_once` for now and will migrate to this
//! module as Phase 4.3 expands.

/// **Phase 4.3 — saga task 3.** Foreground `record()` enqueues this op
/// after committing the memories row + delta append. The materializer
/// drains it and runs the unbounded heuristic-entity / heuristic-
/// relation extraction loops that previously held `db.conn().lock()`
/// on the foreground request path.
///
/// See [`docs/phase_4_3_design.md`](../../../docs/phase_4_3_design.md) for
/// the contract and payload shape.
pub const OP_MATERIALIZE_RECORD_POST: &str = "materialize_record_post";
