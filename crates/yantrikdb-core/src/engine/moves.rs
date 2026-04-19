//! RFC 008 M5b: Cognitive moves — the spine of reasoning.
//!
//! Per M5a locked spec (Saga note 19): this module owns the move substrate.
//! Moves are append-only events recording reasoning transformations.
//! Inputs/outputs/side-effects flow through normalized edge tables.
//! Corrections are first-class events; originals are never mutated for
//! semantic correction (posthoc outcome enrichment is a narrow exception).
//! Adversarial instances are staged (candidate/confirmed/rejected) with
//! governance enforced at the API layer.
//!
//! M5b is observational-first: moves record what happened; they do NOT
//! actively rewrite mobility_state. Active propagation (if ever needed)
//! is a separate feature.
//!
//! Start here when reading:
//! - `record_move_event` — create a move event + its edges atomically
//! - `record_move_outcome` — narrow posthoc mutation of outcome/yield
//! - `submit_move_correction` — correct a move's structural fields via event
//! - `list_moves_consuming_claim` / `list_moves_producing_claim` — reverse edge lookups
//! - Adversarial instance APIs: `create_adversarial_candidate`, `promote_adversarial_candidate`

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::error::{Result, YantrikDbError};

/// Seed vocabulary for move_type_registry. Loaded at bootstrap. Extending
/// this list is a normal operation; the registry is a soft reference, not
/// a schema CHECK.
pub const SEED_MOVE_TYPES: &[(&str, &str, Option<i64>)] = &[
    ("analogy", "Cross-domain pattern transfer from known claims to a target", Some(60_000)),
    ("decomposition", "Split a claim into case-wise sub-claims", Some(30_000)),
    ("aggregate_back", "Recombine case-wise sub-claims into an aggregate (dual of decomposition)", Some(30_000)),
    ("negate_and_test", "Generate the negation of a claim and actively seek disconfirming evidence", Some(300_000)),
    ("source_audit", "Inspect and reweight claims from a shared source (requires ψ_ancestral < 1.0)", Some(900_000)),
    ("ladder_up", "Abstract a specific claim to a more general proposition", Some(60_000)),
    ("contradiction_triage", "Structured evaluation of a Γ(c) contest signature", Some(120_000)),
    ("source_downgrade", "Reduce the weight of claims from a specific source", None),
    ("source_upgrade", "Increase the weight of claims from a specific source", None),
    ("regime_transfer", "Transport a claim/mobility state across regimes", Some(300_000)),
    ("compression", "Consolidate a source span into a compressed artifact", Some(604_800_000)),
    ("hypothesis_generation", "Escrow a candidate explanation (stress-residual minimization)", Some(2_592_000_000)),
    ("quarantine", "Flag a claim-neighborhood for isolation from downstream reasoning", None),
];

/// Seed vocabulary for inference_basis_registry. Only relevant when
/// observability = 'inferred'.
pub const SEED_INFERENCE_BASES: &[(&str, &str)] = &[
    ("structural_pattern_match", "Output claims match the structural signature of a known move type"),
    ("temporal_correlation", "Time-proximity between a trigger and observed effects suggests a move"),
    ("source_lineage_inference", "Source lineage patterns imply a move was applied"),
    ("operator_signature_match", "Operator-specific side effects (e.g. compression artifact shape) were observed"),
    ("human_annotation", "A curator declared the move retrospectively"),
];

/// Lifecycle status values for adversarial instances.
pub mod adversarial_status {
    pub const CANDIDATE: &str = "candidate";
    pub const CONFIRMED: &str = "confirmed";
    pub const REJECTED: &str = "rejected";
}

/// Observability modes for move_events.
pub mod observability {
    pub const OBSERVED: &str = "observed";
    pub const SELF_REPORTED: &str = "self_reported";
    pub const INFERRED: &str = "inferred";
}

/// Posthoc outcome labels.
pub mod posthoc_outcome {
    pub const CORROBORATED: &str = "corroborated";
    pub const RETRACTED: &str = "retracted";
    pub const HARMFUL_SIDE_EFFECT: &str = "harmful_side_effect";
}

/// A recorded cognitive move event. Append-only after insertion; only
/// posthoc_outcome / posthoc_recorded_at / yield_json may be updated
/// after the fact (narrow posthoc enrichment — everything else is
/// corrected via `submit_move_correction`, not mutation).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MoveEvent {
    pub move_id: String,
    pub move_type: String,
    pub operator_version: String,
    pub actor_id: String,
    pub context_regime: String,
    pub observability: String,
    pub inference_confidence: Option<f64>,
    pub inference_basis_json: Option<String>,
    pub dependencies_json: String,
    pub cost_tokens: Option<i64>,
    pub cost_latency_ms: Option<i64>,
    pub cost_memory_reads: Option<i64>,
    pub yield_json: String,
    pub posthoc_outcome: Option<String>,
    pub posthoc_recorded_at: Option<f64>,
    pub expected_evaluation_horizon_ms: Option<i64>,
    pub mobility_state_hash_at_move: Option<String>,
    pub contest_state_hash_at_move: Option<String>,
    pub created_at: f64,
    /// HLC bytes (16 bytes, hex-encoded in this struct for convenience).
    pub hlc_hex: String,
    pub origin_actor: String,
}

/// Input describing a claim_id referenced by a move in a specific role.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimRef {
    pub claim_id: String,
    pub role: String,
    pub ordinal: i64,
}

/// Input describing a side-effect target with a labeled effect kind.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SideEffectRef {
    pub claim_id: String,
    pub effect_kind: String,
}

/// All fields needed to record a new move_event plus its edges atomically.
#[derive(Debug, Clone, Default)]
pub struct RecordMoveEventInput {
    pub move_type: String,
    pub operator_version: String,
    pub context_regime: Option<String>, // defaults to 'default'
    pub observability: String,
    pub inference_confidence: Option<f64>,
    pub inference_basis: Option<Vec<String>>,
    pub dependencies: Vec<String>,
    pub cost_tokens: Option<i64>,
    pub cost_latency_ms: Option<i64>,
    pub cost_memory_reads: Option<i64>,
    pub yield_json: Option<String>,
    pub expected_evaluation_horizon_ms: Option<i64>,
    pub mobility_state_hash_at_move: Option<String>,
    pub contest_state_hash_at_move: Option<String>,
    pub inputs: Vec<ClaimRef>,
    pub outputs: Vec<ClaimRef>,
    pub side_effects: Vec<SideEffectRef>,
}

/// A correction that supersedes one or more structural fields of an
/// existing move. Append-only record; downstream consumers reconstruct
/// the canonical view by joining move_events with the latest correction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveCorrection {
    pub correction_id: String,
    pub original_move_id: String,
    pub corrected_move_type: Option<String>,
    pub corrected_operator_version: Option<String>,
    pub corrected_context_regime: Option<String>,
    pub correction_reason: String,
    pub corrected_by_actor_id: String,
    pub corrected_at: f64,
}

/// Adversarial instance record. Candidate → confirmed promotion is the
/// governed path; only confirmed instances may carry generalized_lesson /
/// lesson_scope_json. Rejection is terminal.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdversarialInstance {
    pub instance_id: String,
    pub move_id: String,
    pub status: String,
    pub discovered_via: String,
    pub traced_root_cause: Option<String>,
    pub generalized_lesson: Option<String>,
    pub lesson_scope_json: Option<String>,
    pub curation_actor_id: Option<String>,
    pub discovered_at: f64,
    pub created_at: f64,
}

impl crate::engine::YantrikDB {
    // ── Bootstrap ─────────────────────────────────────────────────

    /// Seed the move_type_registry and inference_basis_registry with the
    /// canonical vocabulary. Idempotent — INSERT OR IGNORE. Safe to call
    /// multiple times at bootstrap or after migrations.
    pub fn seed_move_registries(&self) -> Result<()> {
        let conn = self.conn.lock();
        seed_registries_inner(&conn)
    }

    // ── Move event lifecycle ──────────────────────────────────────

    /// Record a new move_event and its input/output/side-effect edges
    /// atomically under a single connection lock.
    ///
    /// Soft validation:
    /// - unknown move_type logs a tracing::warn but does not reject
    /// - unknown inference_basis values log warnings but do not reject
    /// - observability rules enforced: inference_confidence and
    ///   inference_basis are valid only when observability='inferred'
    /// - expected_evaluation_horizon_ms falls back to the registry
    ///   default if not supplied
    ///
    /// Returns the new move_id (UUIDv7).
    pub fn record_move_event(&self, input: RecordMoveEventInput) -> Result<String> {
        // Observability semantic rules (enforced before anything touches SQL).
        let is_inferred = input.observability == observability::INFERRED;
        if !is_inferred {
            if input.inference_confidence.is_some() {
                return Err(YantrikDbError::InvalidInput(
                    "inference_confidence may only be set when observability='inferred'".into(),
                ));
            }
            if input
                .inference_basis
                .as_ref()
                .map(|b| !b.is_empty())
                .unwrap_or(false)
            {
                return Err(YantrikDbError::InvalidInput(
                    "inference_basis may only be non-empty when observability='inferred'".into(),
                ));
            }
        }
        if ![observability::OBSERVED, observability::SELF_REPORTED, observability::INFERRED]
            .contains(&input.observability.as_str())
        {
            return Err(YantrikDbError::InvalidInput(format!(
                "observability must be one of observed|self_reported|inferred, got '{}'",
                input.observability
            )));
        }

        let move_id = crate::id::new_id();
        let now_ts = super::now();
        let hlc = self.tick_hlc();
        let origin_actor = self.actor_id().to_string();
        let regime = input.context_regime.clone().unwrap_or_else(|| "default".to_string());

        let conn = self.conn.lock();

        // Soft-registry warnings — never reject.
        let known_type: bool = conn
            .query_row(
                "SELECT 1 FROM move_type_registry WHERE move_type = ?1",
                params![input.move_type],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if !known_type {
            tracing::warn!(
                move_type = %input.move_type,
                "recording move with unregistered move_type; consider adding to move_type_registry"
            );
        }

        if is_inferred {
            if let Some(ref bases) = input.inference_basis {
                for b in bases {
                    let known: bool = conn
                        .query_row(
                            "SELECT 1 FROM inference_basis_registry WHERE basis_type = ?1",
                            params![b],
                            |_| Ok(true),
                        )
                        .unwrap_or(false);
                    if !known {
                        tracing::warn!(
                            basis_type = %b,
                            "unregistered inference_basis; consider adding to inference_basis_registry"
                        );
                    }
                }
            }
        }

        // Fall back to the registry's default horizon when not supplied
        // and only when the move_type is registered (otherwise nothing to
        // fall back to).
        let horizon = match input.expected_evaluation_horizon_ms {
            Some(h) => Some(h),
            None => conn
                .query_row(
                    "SELECT default_expected_evaluation_horizon_ms \
                     FROM move_type_registry WHERE move_type = ?1",
                    params![input.move_type],
                    |row| row.get::<_, Option<i64>>(0),
                )
                .unwrap_or(None),
        };

        let inference_basis_json = input
            .inference_basis
            .as_ref()
            .map(|b| serde_json::to_string(b).unwrap_or_else(|_| "[]".into()));
        let dependencies_json = serde_json::to_string(&input.dependencies)
            .unwrap_or_else(|_| "[]".into());
        let yield_json = input.yield_json.clone().unwrap_or_else(|| "{}".into());
        let hlc_bytes = hlc.to_bytes();

        // Perform INSERT into move_events and all edge tables inside a
        // transaction — atomic together.
        let tx = conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO move_events (\
             move_id, move_type, operator_version, actor_id, context_regime, \
             observability, inference_confidence, inference_basis_json, dependencies_json, \
             cost_tokens, cost_latency_ms, cost_memory_reads, \
             yield_json, posthoc_outcome, posthoc_recorded_at, \
             expected_evaluation_horizon_ms, mobility_state_hash_at_move, contest_state_hash_at_move, \
             created_at, hlc, origin_actor) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, NULL, NULL, \
                     ?14, ?15, ?16, ?17, ?18, ?19)",
            params![
                move_id, input.move_type, input.operator_version, origin_actor, regime,
                input.observability, input.inference_confidence, inference_basis_json, dependencies_json,
                input.cost_tokens, input.cost_latency_ms, input.cost_memory_reads,
                yield_json,
                horizon, input.mobility_state_hash_at_move, input.contest_state_hash_at_move,
                now_ts, hlc_bytes.as_slice(), self.actor_id(),
            ],
        )?;
        for ClaimRef { claim_id, role, ordinal } in &input.inputs {
            tx.execute(
                "INSERT INTO move_input_edge (move_id, claim_id, input_role, ordinal) \
                 VALUES (?1, ?2, ?3, ?4)",
                params![move_id, claim_id, role, ordinal],
            )?;
        }
        for ClaimRef { claim_id, role, ordinal } in &input.outputs {
            tx.execute(
                "INSERT INTO move_output_edge (move_id, claim_id, output_role, ordinal) \
                 VALUES (?1, ?2, ?3, ?4)",
                params![move_id, claim_id, role, ordinal],
            )?;
        }
        for SideEffectRef { claim_id, effect_kind } in &input.side_effects {
            tx.execute(
                "INSERT INTO move_side_effect_edge (move_id, claim_id, effect_kind) \
                 VALUES (?1, ?2, ?3)",
                params![move_id, claim_id, effect_kind],
            )?;
        }
        tx.commit()?;

        Ok(move_id)
    }

    /// Narrow posthoc enrichment — the only allowed mutation on a
    /// move_events row after INSERT. Sets posthoc_outcome, records the
    /// time, and optionally attaches a yield_json payload.
    ///
    /// Rejects attempts to overwrite an already-set posthoc_outcome
    /// (use `submit_move_correction` to re-assess a finalized outcome).
    pub fn record_move_outcome(
        &self,
        move_id: &str,
        outcome: &str,
        yield_json: Option<String>,
    ) -> Result<()> {
        if ![
            posthoc_outcome::CORROBORATED,
            posthoc_outcome::RETRACTED,
            posthoc_outcome::HARMFUL_SIDE_EFFECT,
        ]
        .contains(&outcome)
        {
            return Err(YantrikDbError::InvalidInput(format!(
                "posthoc_outcome must be corroborated|retracted|harmful_side_effect, got '{}'",
                outcome
            )));
        }
        let conn = self.conn.lock();
        let existing: Option<String> = conn
            .query_row(
                "SELECT posthoc_outcome FROM move_events WHERE move_id = ?1",
                params![move_id],
                |row| row.get(0),
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    YantrikDbError::NotFound(format!("move_event: {}", move_id))
                }
                other => other.into(),
            })?;
        if existing.is_some() {
            return Err(YantrikDbError::InvalidInput(format!(
                "move {} already has a posthoc_outcome; use submit_move_correction to re-assess",
                move_id
            )));
        }
        let now_ts = super::now();
        let rows = if let Some(y) = yield_json {
            conn.execute(
                "UPDATE move_events SET posthoc_outcome = ?1, posthoc_recorded_at = ?2, yield_json = ?3 \
                 WHERE move_id = ?4",
                params![outcome, now_ts, y, move_id],
            )?
        } else {
            conn.execute(
                "UPDATE move_events SET posthoc_outcome = ?1, posthoc_recorded_at = ?2 \
                 WHERE move_id = ?3",
                params![outcome, now_ts, move_id],
            )?
        };
        if rows == 0 {
            return Err(YantrikDbError::NotFound(format!("move_event: {}", move_id)));
        }

        // RFC 008 M10: auto-file adversarial candidate on negative outcomes.
        // Retractions and harmful-side-effects are signals the move was
        // epistemically or operationally wrong; record a candidate so the
        // curator or induction pipeline has a queue to work from.
        //
        // Governance rule (M5a spec): automatic systems create `candidate`
        // status with traced_root_cause only; generalized_lesson and
        // lesson_scope_json are left NULL until a curator promotes.
        if outcome == posthoc_outcome::RETRACTED || outcome == posthoc_outcome::HARMFUL_SIDE_EFFECT {
            let instance_id = crate::id::new_id();
            let discovered_via = if outcome == posthoc_outcome::RETRACTED {
                "retraction"
            } else {
                "calibration_signal"
            };
            let root_cause = format!(
                "auto-generated from posthoc_outcome='{}' on move {}",
                outcome, move_id
            );
            conn.execute(
                "INSERT INTO move_adversarial_instance (\
                 instance_id, move_id, status, discovered_via, traced_root_cause, \
                 discovered_at, created_at) \
                 VALUES (?1, ?2, 'candidate', ?3, ?4, ?5, ?5)",
                params![instance_id, move_id, discovered_via, root_cause, now_ts],
            )?;
        }
        Ok(())
    }

    /// Record a correction to a move's structural fields. Supply only
    /// the fields that changed (None = unchanged). Always succeeds as a
    /// new append-only event — never mutates the original move_events row.
    pub fn submit_move_correction(
        &self,
        original_move_id: &str,
        corrected_move_type: Option<String>,
        corrected_operator_version: Option<String>,
        corrected_context_regime: Option<String>,
        correction_reason: String,
        corrected_by_actor_id: String,
    ) -> Result<String> {
        if corrected_move_type.is_none()
            && corrected_operator_version.is_none()
            && corrected_context_regime.is_none()
        {
            return Err(YantrikDbError::InvalidInput(
                "submit_move_correction must specify at least one corrected_* field".into(),
            ));
        }
        if correction_reason.trim().is_empty() {
            return Err(YantrikDbError::InvalidInput(
                "correction_reason must not be empty".into(),
            ));
        }
        let conn = self.conn.lock();
        // Verify the original move exists (FK is also enforced at DB layer).
        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM move_events WHERE move_id = ?1",
                params![original_move_id],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if !exists {
            return Err(YantrikDbError::NotFound(format!(
                "original move_event: {}",
                original_move_id
            )));
        }
        let correction_id = crate::id::new_id();
        let now_ts = super::now();
        conn.execute(
            "INSERT INTO move_correction_event (\
             correction_id, original_move_id, corrected_move_type, corrected_operator_version, \
             corrected_context_regime, correction_reason, corrected_by_actor_id, corrected_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                correction_id, original_move_id,
                corrected_move_type, corrected_operator_version, corrected_context_regime,
                correction_reason, corrected_by_actor_id, now_ts,
            ],
        )?;
        Ok(correction_id)
    }

    // ── Read APIs ─────────────────────────────────────────────────

    /// Fetch a move_event by its move_id. Returns None if not found.
    pub fn get_move_event(&self, move_id: &str) -> Result<Option<MoveEvent>> {
        let conn = self.conn.lock();
        read_move_event(&conn, move_id)
    }

    /// All moves that consumed a given claim as an input. Ordered by
    /// created_at ASC (earliest first).
    pub fn list_moves_consuming_claim(&self, claim_id: &str, limit: usize) -> Result<Vec<MoveEvent>> {
        self.list_moves_by_edge("move_input_edge", claim_id, limit)
    }

    /// All moves that produced a given claim as an output.
    pub fn list_moves_producing_claim(&self, claim_id: &str, limit: usize) -> Result<Vec<MoveEvent>> {
        self.list_moves_by_edge("move_output_edge", claim_id, limit)
    }

    /// All moves whose side-effects touched a given claim.
    pub fn list_moves_side_effecting_claim(
        &self,
        claim_id: &str,
        limit: usize,
    ) -> Result<Vec<MoveEvent>> {
        self.list_moves_by_edge("move_side_effect_edge", claim_id, limit)
    }

    fn list_moves_by_edge(
        &self,
        edge_table: &str,
        claim_id: &str,
        limit: usize,
    ) -> Result<Vec<MoveEvent>> {
        let conn = self.conn.lock();
        let sql = format!(
            "SELECT m.* FROM move_events m \
             INNER JOIN {} e ON e.move_id = m.move_id \
             WHERE e.claim_id = ?1 \
             ORDER BY m.created_at ASC LIMIT ?2",
            edge_table
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params![claim_id, limit as i64], row_to_move_event)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// All correction events recorded against a given original move.
    /// Ordered newest-first (latest correction dominates in canonical
    /// reconstruction).
    pub fn list_move_corrections(&self, original_move_id: &str) -> Result<Vec<MoveCorrection>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT correction_id, original_move_id, corrected_move_type, \
             corrected_operator_version, corrected_context_regime, \
             correction_reason, corrected_by_actor_id, corrected_at \
             FROM move_correction_event WHERE original_move_id = ?1 \
             ORDER BY corrected_at DESC",
        )?;
        let rows = stmt
            .query_map(params![original_move_id], |row| {
                Ok(MoveCorrection {
                    correction_id: row.get(0)?,
                    original_move_id: row.get(1)?,
                    corrected_move_type: row.get(2)?,
                    corrected_operator_version: row.get(3)?,
                    corrected_context_regime: row.get(4)?,
                    correction_reason: row.get(5)?,
                    corrected_by_actor_id: row.get(6)?,
                    corrected_at: row.get(7)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Reconstruct the canonical view of a move: base event overlaid by
    /// the latest correction (if any). Mutates the returned MoveEvent's
    /// structural fields according to the latest correction, but leaves
    /// move_id, created_at, hlc, origin_actor, and all edges untouched.
    pub fn get_move_event_canonical(&self, move_id: &str) -> Result<Option<MoveEvent>> {
        let Some(mut event) = self.get_move_event(move_id)? else {
            return Ok(None);
        };
        let corrections = self.list_move_corrections(move_id)?;
        // Latest correction wins for each field (corrections ordered newest
        // first). The first non-None value for each corrected_* field is
        // the canonical override.
        for c in &corrections {
            if let Some(ref mt) = c.corrected_move_type {
                event.move_type = mt.clone();
                break;
            }
        }
        for c in &corrections {
            if let Some(ref v) = c.corrected_operator_version {
                event.operator_version = v.clone();
                break;
            }
        }
        for c in &corrections {
            if let Some(ref r) = c.corrected_context_regime {
                event.context_regime = r.clone();
                break;
            }
        }
        Ok(Some(event))
    }

    /// Inputs of a move, in ordinal order.
    pub fn get_move_inputs(&self, move_id: &str) -> Result<Vec<ClaimRef>> {
        self.get_move_edges_generic("move_input_edge", "input_role", move_id)
    }

    /// Outputs of a move, in ordinal order.
    pub fn get_move_outputs(&self, move_id: &str) -> Result<Vec<ClaimRef>> {
        self.get_move_edges_generic("move_output_edge", "output_role", move_id)
    }

    /// Side-effects of a move.
    pub fn get_move_side_effects(&self, move_id: &str) -> Result<Vec<SideEffectRef>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT claim_id, effect_kind FROM move_side_effect_edge \
             WHERE move_id = ?1 ORDER BY effect_kind, claim_id",
        )?;
        let rows = stmt
            .query_map(params![move_id], |row| {
                Ok(SideEffectRef {
                    claim_id: row.get(0)?,
                    effect_kind: row.get(1)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    fn get_move_edges_generic(
        &self,
        edge_table: &str,
        role_col: &str,
        move_id: &str,
    ) -> Result<Vec<ClaimRef>> {
        let conn = self.conn.lock();
        let sql = format!(
            "SELECT claim_id, {}, ordinal FROM {} WHERE move_id = ?1 ORDER BY ordinal, claim_id",
            role_col, edge_table
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params![move_id], |row| {
                Ok(ClaimRef {
                    claim_id: row.get(0)?,
                    role: row.get(1)?,
                    ordinal: row.get(2)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    // ── Adversarial instance lifecycle ────────────────────────────

    /// Create an adversarial instance in `candidate` status. Automatic
    /// systems (calibration signals, retraction detectors) use this
    /// entry point. Enforces the governance rule: no generalized_lesson
    /// or lesson_scope_json may be populated on candidates.
    pub fn create_adversarial_candidate(
        &self,
        move_id: &str,
        discovered_via: &str,
        traced_root_cause: Option<String>,
    ) -> Result<String> {
        if ![
            "contradiction",
            "retraction",
            "calibration_signal",
            "human_audit",
        ]
        .contains(&discovered_via)
        {
            return Err(YantrikDbError::InvalidInput(format!(
                "discovered_via must be one of contradiction|retraction|calibration_signal|human_audit, got '{}'",
                discovered_via
            )));
        }
        let conn = self.conn.lock();
        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM move_events WHERE move_id = ?1",
                params![move_id],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if !exists {
            return Err(YantrikDbError::NotFound(format!(
                "move_event: {}",
                move_id
            )));
        }
        let instance_id = crate::id::new_id();
        let now_ts = super::now();
        conn.execute(
            "INSERT INTO move_adversarial_instance (\
             instance_id, move_id, status, discovered_via, traced_root_cause, \
             discovered_at, created_at) \
             VALUES (?1, ?2, 'candidate', ?3, ?4, ?5, ?5)",
            params![instance_id, move_id, discovered_via, traced_root_cause, now_ts],
        )?;
        Ok(instance_id)
    }

    /// Promote a candidate to `confirmed`, attaching the curator-validated
    /// generalized lesson and its scope. Rejects promotion of non-candidate
    /// rows and requires both lesson fields to be non-empty on confirmation.
    pub fn promote_adversarial_candidate(
        &self,
        instance_id: &str,
        generalized_lesson: String,
        lesson_scope_json: String,
        curation_actor_id: String,
    ) -> Result<()> {
        if generalized_lesson.trim().is_empty() {
            return Err(YantrikDbError::InvalidInput(
                "generalized_lesson must be non-empty when promoting to confirmed".into(),
            ));
        }
        if lesson_scope_json.trim().is_empty() {
            return Err(YantrikDbError::InvalidInput(
                "lesson_scope_json must be non-empty when promoting to confirmed".into(),
            ));
        }
        let conn = self.conn.lock();
        let status: String = conn
            .query_row(
                "SELECT status FROM move_adversarial_instance WHERE instance_id = ?1",
                params![instance_id],
                |row| row.get(0),
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    YantrikDbError::NotFound(format!("adversarial_instance: {}", instance_id))
                }
                other => other.into(),
            })?;
        if status != adversarial_status::CANDIDATE {
            return Err(YantrikDbError::InvalidInput(format!(
                "can only promote from candidate, current status is '{}'",
                status
            )));
        }
        conn.execute(
            "UPDATE move_adversarial_instance SET \
             status = 'confirmed', generalized_lesson = ?1, lesson_scope_json = ?2, \
             curation_actor_id = ?3 WHERE instance_id = ?4",
            params![generalized_lesson, lesson_scope_json, curation_actor_id, instance_id],
        )?;
        Ok(())
    }

    /// Reject an adversarial candidate — terminal state, no further
    /// transitions. Rejected instances stay in the table for audit.
    pub fn reject_adversarial_candidate(
        &self,
        instance_id: &str,
        curation_actor_id: String,
    ) -> Result<()> {
        let conn = self.conn.lock();
        let rows = conn.execute(
            "UPDATE move_adversarial_instance SET \
             status = 'rejected', curation_actor_id = ?1 \
             WHERE instance_id = ?2 AND status = 'candidate'",
            params![curation_actor_id, instance_id],
        )?;
        if rows == 0 {
            return Err(YantrikDbError::InvalidInput(format!(
                "cannot reject {}: either not found or already promoted/rejected",
                instance_id
            )));
        }
        Ok(())
    }

    /// Fetch an adversarial instance by id.
    pub fn get_adversarial_instance(
        &self,
        instance_id: &str,
    ) -> Result<Option<AdversarialInstance>> {
        let conn = self.conn.lock();
        let res = conn.query_row(
            "SELECT instance_id, move_id, status, discovered_via, traced_root_cause, \
             generalized_lesson, lesson_scope_json, curation_actor_id, discovered_at, created_at \
             FROM move_adversarial_instance WHERE instance_id = ?1",
            params![instance_id],
            row_to_adversarial,
        );
        match res {
            Ok(i) => Ok(Some(i)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// List adversarial instances for a given move.
    pub fn list_adversarial_for_move(
        &self,
        move_id: &str,
    ) -> Result<Vec<AdversarialInstance>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT instance_id, move_id, status, discovered_via, traced_root_cause, \
             generalized_lesson, lesson_scope_json, curation_actor_id, discovered_at, created_at \
             FROM move_adversarial_instance WHERE move_id = ?1 ORDER BY discovered_at DESC",
        )?;
        let rows = stmt
            .query_map(params![move_id], row_to_adversarial)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }
}

// ──────────────────────────────────────────────────────────────────
// RFC 008 M8: Axiomatic composition rules.
//
// Per M5a locked spec: axioms are definitional — they live in code, not
// in the move_composition_rule table. The table stores only empirical
// and user-declared rules. Axioms are compile-checked and versioned
// with the system source; mixing them with noisy empirical observations
// would let evidence override definitional truths.
// ──────────────────────────────────────────────────────────────────

/// Axiomatic composition kinds. Each axiom declares one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxiomKind {
    /// The sequence `left ∘ right` approximates the identity at warrant
    /// level. Engine may elide such pairs from processing.
    ApproxIdentity,
    /// `left ∘ right` and `right ∘ left` are NOT equivalent; move order
    /// matters for this pair.
    NonCommutative,
    /// The `left_move_type` has a definitional precondition; applying it
    /// without the precondition holding is incoherent.
    PreconditionRequirement,
}

/// Precondition checks evaluated against mobility_state / contest_state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Precondition {
    /// Require self_gen_ancestral strictly less than the given threshold.
    /// e.g. `SelfGenAncestralBelow(1.0)` means "there must be at least
    /// some external ancestry (not fully self-generated)."
    SelfGenAncestralBelow(f64),
    /// Require support_mass (σ) strictly above the given threshold.
    SupportMassAbove(f64),
    /// Require contest_state.heuristic_flags bitmask intersection equals zero
    /// (i.e. none of the listed flag bits are set).
    NoContestFlags(u64),
}

#[derive(Debug, Clone, Copy)]
pub struct Axiom {
    pub name: &'static str,
    pub kind: AxiomKind,
    pub left_move_type: Option<&'static str>,
    pub right_move_type: Option<&'static str>,
    pub precondition: Option<Precondition>,
    pub description: &'static str,
}

/// The axiom registry. Populated from RFC 008 § 2 and refined as new
/// definitional laws are discovered. Bumping this registry is a code
/// change, reviewed with the usual rigor; it is NOT schema migration.
pub const AXIOMS: &[Axiom] = &[
    Axiom {
        name: "decompose_aggregate_identity",
        kind: AxiomKind::ApproxIdentity,
        left_move_type: Some("decomposition"),
        right_move_type: Some("aggregate_back"),
        precondition: None,
        description: "decomposition ∘ aggregate_back ≈ identity at warrant level \
                      (case splitting followed by reaggregation preserves support structure)",
    },
    Axiom {
        name: "negate_analogize_non_commutative",
        kind: AxiomKind::NonCommutative,
        left_move_type: Some("negate_and_test"),
        right_move_type: Some("analogy"),
        precondition: None,
        description: "negating then analogizing explores different structure than \
                      analogizing then negating; the two orderings are not equivalent",
    },
    Axiom {
        name: "source_audit_requires_external_ancestry",
        kind: AxiomKind::PreconditionRequirement,
        left_move_type: Some("source_audit"),
        right_move_type: None,
        precondition: Some(Precondition::SelfGenAncestralBelow(1.0)),
        description: "source_audit is incoherent on fully self-generated claims \
                      (there is no external source to audit when ψ_ancestral = 1.0)",
    },
    Axiom {
        name: "compression_requires_support",
        kind: AxiomKind::PreconditionRequirement,
        left_move_type: Some("compression"),
        right_move_type: None,
        precondition: Some(Precondition::SupportMassAbove(0.0)),
        description: "compression over a proposition with no support mass has \
                      nothing to consolidate; apply only once claims exist",
    },
    Axiom {
        name: "hypothesis_generation_skips_present_tense_conflict",
        kind: AxiomKind::PreconditionRequirement,
        left_move_type: Some("hypothesis_generation"),
        right_move_type: None,
        precondition: Some(Precondition::NoContestFlags(0b10000)), // PRESENT_TENSE_CONFLICT bit
        description: "generating new hypotheses over an actively-contested proposition \
                      compounds noise; resolve the present-tense conflict first",
    },
];

/// A violated precondition, returned by `check_move_preconditions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreconditionViolation {
    pub axiom_name: &'static str,
    pub move_type: String,
    pub description: &'static str,
    /// Human-readable observation of what was seen vs required.
    pub observation: String,
}

impl crate::engine::YantrikDB {
    /// Return the static axiom registry. Useful for introspection tests
    /// and documentation tools.
    pub fn composition_axioms(&self) -> &'static [Axiom] {
        AXIOMS
    }

    /// Check whether a pair of moves matches any ApproxIdentity or
    /// NonCommutative axiom. Given two move_ids, this looks up their
    /// canonical move_types (applying the latest correction) and then
    /// scans the axiom registry for a match. Returns all matching
    /// axioms — usually zero or one.
    pub fn check_composition_axioms(
        &self,
        left_move_id: &str,
        right_move_id: &str,
    ) -> Result<Vec<&'static Axiom>> {
        let left = self.get_move_event_canonical(left_move_id)?;
        let right = self.get_move_event_canonical(right_move_id)?;
        let (Some(l), Some(r)) = (left, right) else {
            return Ok(Vec::new());
        };
        let matches: Vec<&'static Axiom> = AXIOMS
            .iter()
            .filter(|ax| matches!(ax.kind, AxiomKind::ApproxIdentity | AxiomKind::NonCommutative))
            .filter(|ax| {
                ax.left_move_type == Some(l.move_type.as_str())
                    && ax.right_move_type == Some(r.move_type.as_str())
            })
            .collect();
        Ok(matches)
    }

    /// Check whether a move of the given type would violate any of its
    /// PreconditionRequirement axioms, given the current mobility and
    /// contest state for (proposition_id, regime). Returns a list of
    /// violated preconditions — empty list means the move is coherent.
    ///
    /// Callers typically check this BEFORE recording a move, to either
    /// reject the operation or flag it for review. M8 provides the
    /// mechanism; policy (reject vs warn) is the caller's decision.
    pub fn check_move_preconditions(
        &self,
        move_type: &str,
        proposition_id: &str,
        regime: &str,
    ) -> Result<Vec<PreconditionViolation>> {
        let mobility = self.get_mobility_state(proposition_id, regime)?;
        let contest = self.get_contest_state(proposition_id, regime)?;
        let mut violations = Vec::new();
        for ax in AXIOMS {
            if ax.kind != AxiomKind::PreconditionRequirement {
                continue;
            }
            if ax.left_move_type != Some(move_type) {
                continue;
            }
            let Some(pre) = ax.precondition else { continue };
            let violated = match pre {
                Precondition::SelfGenAncestralBelow(threshold) => {
                    let psi_a = mobility.as_ref().and_then(|m| m.self_gen_ancestral);
                    match psi_a {
                        Some(v) if v < threshold => None,
                        Some(v) => Some(format!(
                            "self_gen_ancestral = {:.3} (required < {})",
                            v, threshold
                        )),
                        None => Some(format!(
                            "self_gen_ancestral not yet computed (required < {}); \
                             run compute_background_mobility first",
                            threshold
                        )),
                    }
                }
                Precondition::SupportMassAbove(threshold) => {
                    let sigma = mobility.as_ref().and_then(|m| m.support_mass);
                    match sigma {
                        Some(v) if v > threshold => None,
                        Some(v) => Some(format!(
                            "support_mass = {:.3} (required > {})",
                            v, threshold
                        )),
                        None => Some(format!(
                            "support_mass not yet computed (required > {})",
                            threshold
                        )),
                    }
                }
                Precondition::NoContestFlags(bitmask) => {
                    let flags = contest.as_ref().map(|c| c.heuristic_flags).unwrap_or(0);
                    if flags & bitmask == 0 {
                        None
                    } else {
                        Some(format!(
                            "contest heuristic_flags = {:#x} intersects required-zero mask {:#x}",
                            flags, bitmask
                        ))
                    }
                }
            };
            if let Some(obs) = violated {
                violations.push(PreconditionViolation {
                    axiom_name: ax.name,
                    move_type: move_type.to_string(),
                    description: ax.description,
                    observation: obs,
                });
            }
        }
        Ok(violations)
    }
}

// ──────────────────────────────────────────────────────────────────
// RFC 008 M9: move_type_profile derivation.
//
// Background loop that folds resolved-or-past-horizon move_events into
// per-(move_type, operator_version, context_regime) distortion profiles.
// Called by app-layer scan or think() integration. Designed to be safe
// under concurrent ingestion: each profile is recomputed from scratch
// on demand, no incremental drift.
// ──────────────────────────────────────────────────────────────────

/// A single distortion profile row. Mirrors the move_type_profile schema.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MoveTypeProfile {
    pub move_type: String,
    pub operator_version: String,
    pub context_regime: String,
    pub uses_count: i64,
    pub resolved_count: i64,
    pub corroborated_count: i64,
    pub retracted_count: i64,
    pub harmful_side_effect_count: i64,
    pub contradiction_introduction_rate: Option<f64>,
    pub avg_mobility_shift: Option<f64>,
    pub predictive_gain_avg: Option<f64>,
    pub calibration_gain_avg: Option<f64>,
    pub last_updated: f64,
}

impl crate::engine::YantrikDB {
    /// Recompute the move_type_profile row for a specific key. Scans all
    /// move_events matching (move_type, operator_version, context_regime)
    /// and aggregates outcomes into counters + rates. Returns the written
    /// profile.
    ///
    /// A move counts as "resolved" when `posthoc_outcome IS NOT NULL` OR
    /// when `created_at + expected_evaluation_horizon_ms` has passed.
    /// The latter is measured against the current wall clock at compute
    /// time — consumers that want strict determinism should pin the
    /// horizon-check logic separately.
    pub fn recompute_move_type_profile(
        &self,
        move_type: &str,
        operator_version: &str,
        context_regime: &str,
    ) -> Result<MoveTypeProfile> {
        let conn = self.conn.lock();
        let now_ms = (crate::engine::now() * 1000.0) as i64;

        // Count uses (all events).
        let uses_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM move_events \
             WHERE move_type = ?1 AND operator_version = ?2 AND context_regime = ?3",
            params![move_type, operator_version, context_regime],
            |row| row.get(0),
        )?;

        // Count resolved (outcome set OR past horizon).
        let resolved_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM move_events \
             WHERE move_type = ?1 AND operator_version = ?2 AND context_regime = ?3 \
               AND (posthoc_outcome IS NOT NULL \
                    OR (expected_evaluation_horizon_ms IS NOT NULL \
                        AND (created_at * 1000 + expected_evaluation_horizon_ms) < ?4))",
            params![move_type, operator_version, context_regime, now_ms],
            |row| row.get(0),
        )?;

        // Per-outcome counters.
        let corroborated_count = self.count_by_outcome(&conn, move_type, operator_version, context_regime, "corroborated")?;
        let retracted_count = self.count_by_outcome(&conn, move_type, operator_version, context_regime, "retracted")?;
        let harmful_side_effect_count = self.count_by_outcome(&conn, move_type, operator_version, context_regime, "harmful_side_effect")?;

        // Rates — only well-defined when there are resolved moves.
        let contradiction_introduction_rate = if resolved_count > 0 {
            // Approximation: treat retracted + harmful as "introduced contradiction."
            Some((retracted_count + harmful_side_effect_count) as f64 / resolved_count as f64)
        } else {
            None
        };

        // predictive_gain_avg and calibration_gain_avg come from yield_json;
        // parsing those requires more structure than M9 ships. Leave NULL
        // until a consumer defines the yield_json schema.
        let predictive_gain_avg = None;
        let calibration_gain_avg = None;

        // avg_mobility_shift would come from comparing mobility snapshots
        // before/after moves via the content_hash pointers. Leave NULL
        // until M6+ provides the snapshot timeline.
        let avg_mobility_shift = None;

        let last_updated = crate::engine::now();

        conn.execute(
            "INSERT INTO move_type_profile (\
             move_type, operator_version, context_regime, \
             uses_count, resolved_count, \
             corroborated_count, retracted_count, harmful_side_effect_count, \
             contradiction_introduction_rate, avg_mobility_shift, \
             predictive_gain_avg, calibration_gain_avg, last_updated) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13) \
             ON CONFLICT(move_type, operator_version, context_regime) DO UPDATE SET \
             uses_count = excluded.uses_count, \
             resolved_count = excluded.resolved_count, \
             corroborated_count = excluded.corroborated_count, \
             retracted_count = excluded.retracted_count, \
             harmful_side_effect_count = excluded.harmful_side_effect_count, \
             contradiction_introduction_rate = excluded.contradiction_introduction_rate, \
             avg_mobility_shift = excluded.avg_mobility_shift, \
             predictive_gain_avg = excluded.predictive_gain_avg, \
             calibration_gain_avg = excluded.calibration_gain_avg, \
             last_updated = excluded.last_updated",
            params![
                move_type, operator_version, context_regime,
                uses_count, resolved_count,
                corroborated_count, retracted_count, harmful_side_effect_count,
                contradiction_introduction_rate, avg_mobility_shift,
                predictive_gain_avg, calibration_gain_avg, last_updated,
            ],
        )?;

        Ok(MoveTypeProfile {
            move_type: move_type.to_string(),
            operator_version: operator_version.to_string(),
            context_regime: context_regime.to_string(),
            uses_count,
            resolved_count,
            corroborated_count,
            retracted_count,
            harmful_side_effect_count,
            contradiction_introduction_rate,
            avg_mobility_shift,
            predictive_gain_avg,
            calibration_gain_avg,
            last_updated,
        })
    }

    fn count_by_outcome(
        &self,
        conn: &Connection,
        move_type: &str,
        operator_version: &str,
        context_regime: &str,
        outcome: &str,
    ) -> Result<i64> {
        let n: i64 = conn.query_row(
            "SELECT COUNT(*) FROM move_events \
             WHERE move_type = ?1 AND operator_version = ?2 \
               AND context_regime = ?3 AND posthoc_outcome = ?4",
            params![move_type, operator_version, context_regime, outcome],
            |row| row.get(0),
        )?;
        Ok(n)
    }

    /// Scan all distinct (move_type, operator_version, context_regime)
    /// triples present in move_events and recompute each profile.
    /// Returns the number of profiles updated.
    pub fn recompute_all_move_type_profiles(&self) -> Result<usize> {
        let keys = self.list_profile_keys()?;
        let mut count = 0;
        for (mt, op, regime) in keys {
            self.recompute_move_type_profile(&mt, &op, &regime)?;
            count += 1;
        }
        Ok(count)
    }

    fn list_profile_keys(&self) -> Result<Vec<(String, String, String)>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT DISTINCT move_type, operator_version, context_regime \
             FROM move_events",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Read a move_type_profile row. None if not yet computed.
    pub fn get_move_type_profile(
        &self,
        move_type: &str,
        operator_version: &str,
        context_regime: &str,
    ) -> Result<Option<MoveTypeProfile>> {
        let conn = self.conn.lock();
        let res = conn.query_row(
            "SELECT move_type, operator_version, context_regime, \
             uses_count, resolved_count, corroborated_count, retracted_count, \
             harmful_side_effect_count, contradiction_introduction_rate, \
             avg_mobility_shift, predictive_gain_avg, calibration_gain_avg, \
             last_updated \
             FROM move_type_profile \
             WHERE move_type = ?1 AND operator_version = ?2 AND context_regime = ?3",
            params![move_type, operator_version, context_regime],
            |row| {
                Ok(MoveTypeProfile {
                    move_type: row.get(0)?,
                    operator_version: row.get(1)?,
                    context_regime: row.get(2)?,
                    uses_count: row.get(3)?,
                    resolved_count: row.get(4)?,
                    corroborated_count: row.get(5)?,
                    retracted_count: row.get(6)?,
                    harmful_side_effect_count: row.get(7)?,
                    contradiction_introduction_rate: row.get(8)?,
                    avg_mobility_shift: row.get(9)?,
                    predictive_gain_avg: row.get(10)?,
                    calibration_gain_avg: row.get(11)?,
                    last_updated: row.get(12)?,
                })
            },
        );
        match res {
            Ok(p) => Ok(Some(p)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// Connection-level bootstrap helper (callable from YantrikDB::new).
// ──────────────────────────────────────────────────────────────────

pub(super) fn seed_registries_inner(conn: &Connection) -> Result<()> {
    let now = super::now();
    for (mt, desc, horizon) in SEED_MOVE_TYPES {
        conn.execute(
            "INSERT OR IGNORE INTO move_type_registry \
             (move_type, status, description, introduced_at, default_expected_evaluation_horizon_ms) \
             VALUES (?1, 'active', ?2, ?3, ?4)",
            params![mt, desc, now, horizon],
        )?;
    }
    for (basis, desc) in SEED_INFERENCE_BASES {
        conn.execute(
            "INSERT OR IGNORE INTO inference_basis_registry \
             (basis_type, description, status) VALUES (?1, ?2, 'active')",
            params![basis, desc],
        )?;
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────
// Row decoders.
// ──────────────────────────────────────────────────────────────────

fn read_move_event(conn: &Connection, move_id: &str) -> Result<Option<MoveEvent>> {
    let mut stmt = conn.prepare(
        "SELECT move_id, move_type, operator_version, actor_id, context_regime, \
         observability, inference_confidence, inference_basis_json, dependencies_json, \
         cost_tokens, cost_latency_ms, cost_memory_reads, yield_json, \
         posthoc_outcome, posthoc_recorded_at, expected_evaluation_horizon_ms, \
         mobility_state_hash_at_move, contest_state_hash_at_move, \
         created_at, hlc, origin_actor \
         FROM move_events WHERE move_id = ?1",
    )?;
    let result = stmt.query_row(params![move_id], row_to_move_event);
    match result {
        Ok(e) => Ok(Some(e)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn row_to_move_event(row: &rusqlite::Row) -> rusqlite::Result<MoveEvent> {
    let hlc_bytes: Vec<u8> = row.get(19)?;
    Ok(MoveEvent {
        move_id: row.get(0)?,
        move_type: row.get(1)?,
        operator_version: row.get(2)?,
        actor_id: row.get(3)?,
        context_regime: row.get(4)?,
        observability: row.get(5)?,
        inference_confidence: row.get(6)?,
        inference_basis_json: row.get(7)?,
        dependencies_json: row.get(8)?,
        cost_tokens: row.get(9)?,
        cost_latency_ms: row.get(10)?,
        cost_memory_reads: row.get(11)?,
        yield_json: row.get(12)?,
        posthoc_outcome: row.get(13)?,
        posthoc_recorded_at: row.get(14)?,
        expected_evaluation_horizon_ms: row.get(15)?,
        mobility_state_hash_at_move: row.get(16)?,
        contest_state_hash_at_move: row.get(17)?,
        created_at: row.get(18)?,
        hlc_hex: hex::encode(&hlc_bytes),
        origin_actor: row.get(20)?,
    })
}

fn row_to_adversarial(row: &rusqlite::Row) -> rusqlite::Result<AdversarialInstance> {
    Ok(AdversarialInstance {
        instance_id: row.get(0)?,
        move_id: row.get(1)?,
        status: row.get(2)?,
        discovered_via: row.get(3)?,
        traced_root_cause: row.get(4)?,
        generalized_lesson: row.get(5)?,
        lesson_scope_json: row.get(6)?,
        curation_actor_id: row.get(7)?,
        discovered_at: row.get(8)?,
        created_at: row.get(9)?,
    })
}
