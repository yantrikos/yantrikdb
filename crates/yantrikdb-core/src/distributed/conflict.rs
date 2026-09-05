//! Conflict detection and resolution.
//!
//! Rule-based detection engine for semantic contradictions across synced memories.
//! Conflicts are first-class data: stored in their own table, queryable, auditable,
//! and replicated via the oplog.
//!
//! ## Claim-keyed detection (2026-08 rewrite)
//!
//! Detection keys on **claims** — (subject, relation) → value — never on
//! shared-entity word bags. The previous detector also carried three lexical
//! heuristics (temporal-keyword substitution, date-like-token substitution,
//! same-type entity substitution over word-set diffs). Hand-labeled precision
//! on the production store's own open set: **0/16** (2026-08-17). Six of the
//! sixteen were the word-bag shape — two memories about completely different
//! projects, paired solely because they shared one entity and carried
//! different dates. Different topics on different days are not
//! contradictions, and no tuning makes a word-set diff a contradiction test,
//! so that path was REMOVED, not patched. The remaining ten were multi-valued
//! relations walked pairwise, phantom subjects, and self-conflicts — each now
//! blocked structurally (see [`FUNCTIONAL_REL_TYPES`], [`subject_admissible`],
//! [`conflict_pair_eligible`]).

use rusqlite::{params, OptionalExtension};

use crate::engine::YantrikDB;
use crate::error::{Result, YantrikDbError};
use crate::types::{Conflict, ConflictType};

/// Rel types that indicate unique-value identity facts (should not have multiple values).
const IDENTITY_REL_TYPES: &[&str] = &[
    "birthday",
    "age",
    "lives_in",
    "works_at",
    "email",
    "phone",
    "full_name",
    "spouse",
    // The heuristic extractor mints `married_to` ("is married to"), never
    // `spouse`; without this entry a second spouse edge was classified
    // Minor and never surfaced (population-of-zero whitelist entry).
    "married_to",
    "hometown",
];

/// Rel types that indicate preferences (concurrent differences are suspicious).
const PREFERENCE_REL_TYPES: &[&str] = &["prefers", "favorite", "likes", "dislikes"];

/// Classify a conflict type from the rel_type.
fn classify_conflict(rel_type: &str) -> ConflictType {
    if IDENTITY_REL_TYPES.contains(&rel_type) {
        ConflictType::IdentityFact
    } else if PREFERENCE_REL_TYPES.contains(&rel_type) {
        ConflictType::Preference
    } else {
        ConflictType::Minor
    }
}

/// Relations treated as FUNCTIONAL — single-valued at any point in time, so
/// two active claims with distinct objects genuinely contradict (or succeed
/// one another). Every relation NOT in this set defaults to MULTI-valued and
/// never conflicts on distinct objects alone: nine of the sixteen labeled
/// false positives (2026-08-17 set) were multi-valued relations ("created",
/// "leads") walked pairwise, O(n²) — one "Pranab created {four projects}"
/// fact alone generated five conflicts.
///
/// Deliberately conservative: a missed conflict costs one review; a false one
/// erodes trust in every flag. The set covers the four families the labeled
/// set names as genuinely functional — version-of, located-at,
/// current-state, is-the-current — and grows only with evidence. Notably
/// ABSENT: "created", "founded", "leads", "acquired", "works_at", "likes" —
/// all legitimately many-valued (or, for "leads", known extraction junk).
const FUNCTIONAL_REL_TYPES: &[&str] = &[
    // Copular attribute claims — attribute_claims.rs mints rel = "is" for
    // "<subject> is <value>" ("brand color is blue" → "is now green").
    "is",
    // version-of
    "runs",
    "runs_version",
    "is_at_version",
    "version_of",
    "schema_version",
    "pinned_to",
    "pin_range",
    // located-at
    "located_at",
    "located_in",
    "based_in",
    "lives_in",
    "hometown",
    "born_in",
    "headquartered_in",
    // current-state
    "current_state",
    "deployed_state",
    "released_state",
    // is-the-current (role identity: one CEO, one parent company, one spouse)
    "ceo_of",
    "cto_of",
    "cfo_of",
    "married_to",
    "spouse",
    "subsidiary_of",
    // single-valued personal identity facts
    "birthday",
    "age",
    "email",
    "phone",
    "full_name",
];

/// Is this relation single-valued (and therefore able to conflict)?
fn is_functional_relation(rel_type: &str) -> bool {
    FUNCTIONAL_REL_TYPES.contains(&rel_type)
}

/// Can this name anchor a conflict as a claim subject?
///
/// A subject today's entity extractor would not mint cannot anchor a
/// conflict: the labeled set contains conflicts anchored on `546`, `12`, and
/// bare heading words — extraction pollution feeding the detector. Reuses the
/// 0.14.1 graph-heal predicate ([`crate::graph::is_rejected_entity_name`]:
/// stopword-only names, ALL-CAPS headings, prose runs) plus a bare-number
/// guard the entity path never needed (its capitalized-chunk heuristic cannot
/// emit digits, but the claims lane can).
///
/// Callers exempt subjects a user deliberately created via `relate()`
/// (`claims.extractor = 'manual'`) — same provenance rule as the graph-index
/// heal: an explicit relation protects its endpoints even if they look like
/// prose.
fn subject_admissible(name: &str) -> bool {
    name.chars().any(|c| c.is_alphabetic()) && !crate::graph::is_rejected_entity_name(name)
}

/// Structural gate every detected pair must pass before a conflict row is
/// created. Encodes three of the labeled-set rules:
///
/// 1. **Never self-conflict.** Several of the sixteen labeled false positives
///    had `memory_a == memory_b` — one record mentioning a progression
///    ("13 → 15") flagged against itself.
/// 2. **A supersession link resolves the pair.** A selected active
///    `supersedes` edge between the two records IS the resolution a
///    succession conflict exists to bring about — flagging on top of it is
///    noise.
/// 3. **Both records must still be active.** A contradiction with a
///    consolidated/superseded record is history, not a live conflict. Rids
///    that are not memory rows (edge ids from `relate()`, synthetic
///    `claim:` fallbacks) skip this sub-check — there is no lifecycle row to
///    consult.
fn conflict_pair_eligible(db: &YantrikDB, rid_a: &str, rid_b: &str) -> Result<bool> {
    if rid_a == rid_b {
        return Ok(false);
    }
    let conn = db.conn();
    let superseded: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM record_links \
         WHERE link_type = 'supersedes' AND status = 'active' \
           AND selection_state = 'selected' \
           AND ((source_rid = ?1 AND target_rid = ?2) \
             OR (source_rid = ?2 AND target_rid = ?1))",
        params![rid_a, rid_b],
        |row| row.get(0),
    )?;
    if superseded {
        return Ok(false);
    }
    let mut status_stmt =
        conn.prepare_cached("SELECT consolidation_status FROM memories WHERE rid = ?1")?;
    for rid in [rid_a, rid_b] {
        let status: Option<String> = status_stmt
            .query_row(params![rid], |row| row.get(0))
            .optional()?;
        if let Some(s) = status {
            if s != "active" {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Map substitution category names to ConflictType.
/// Identity-like categories produce IdentityFact; everything else produces Preference.
/// Used by the redundancy-trigger path (learned substitution categories), which
/// pairs memories the embedding space already put next to each other — a much
/// tighter gate than the removed word-bag scan ever had.
const IDENTITY_CATEGORIES: &[&str] = &["cloud_providers"];

/// Map a substitution category name to the appropriate ConflictType.
pub(crate) fn category_to_conflict_type(cat_name: &str) -> ConflictType {
    if IDENTITY_CATEGORIES.contains(&cat_name) {
        ConflictType::IdentityFact
    } else {
        ConflictType::Preference
    }
}

/// Check if a conflict already exists for this (memory_a, memory_b) pair.
/// Checks both orderings.
pub(crate) fn conflict_exists(db: &YantrikDB, rid_a: &str, rid_b: &str) -> Result<bool> {
    let conn = db.conn();
    let exists: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM conflicts
         WHERE (memory_a = ?1 AND memory_b = ?2)
            OR (memory_a = ?2 AND memory_b = ?1)",
        params![rid_a, rid_b],
        |row| row.get(0),
    )?;
    Ok(exists)
}

/// Find the oplog target_rid for a relate op with given (src, dst, rel_type).
fn find_memory_for_edge(
    conn: &rusqlite::Connection,
    src: &str,
    dst: &str,
    rel_type: &str,
) -> Result<Option<String>> {
    let result = conn.query_row(
        "SELECT target_rid FROM oplog
         WHERE op_type = 'relate'
           AND json_extract(payload, '$.src') = ?1
           AND json_extract(payload, '$.dst') = ?2
           AND json_extract(payload, '$.rel_type') = ?3
         ORDER BY hlc DESC LIMIT 1",
        params![src, dst, rel_type],
        |row| row.get::<_, Option<String>>(0),
    );

    match result {
        Ok(rid) => Ok(rid),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Create a conflict record and log it to the oplog for replication.
///
/// Rejects `memory_a == memory_b` at the engine boundary: a record cannot
/// contradict itself, yet the labeled production set contained exactly that
/// shape (one record mentioning a progression "13 → 15" flagged against
/// itself). Guarding here — not just at detection sites — makes the whole
/// category unrepresentable for future detectors too.
pub fn create_conflict(
    db: &YantrikDB,
    conflict_type: &ConflictType,
    memory_a: &str,
    memory_b: &str,
    entity: Option<&str>,
    rel_type: Option<&str>,
    detection_reason: &str,
) -> Result<Conflict> {
    if memory_a == memory_b {
        return Err(YantrikDbError::InvalidInput(format!(
            "conflict: memory_a == memory_b ({memory_a}) — a record cannot conflict with itself"
        )));
    }
    let conflict_id = crate::id::new_id();
    let ts = crate::time::now_secs();
    let priority = conflict_type.default_priority();
    let hlc_ts = db.tick_hlc();
    let hlc_bytes = hlc_ts.to_bytes().to_vec();
    let actor_id = db.actor_id().to_string();

    db.conn().execute(
        "INSERT OR IGNORE INTO conflicts
         (conflict_id, conflict_type, priority, status, memory_a, memory_b,
          entity, rel_type, detected_at, detected_by, detection_reason,
          hlc, origin_actor)
         VALUES (?1, ?2, ?3, 'open', ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            conflict_id,
            conflict_type.as_str(),
            priority,
            memory_a,
            memory_b,
            entity,
            rel_type,
            ts,
            actor_id,
            detection_reason,
            hlc_bytes,
            actor_id,
        ],
    )?;

    // Log to oplog for replication
    db.log_op(
        "conflict_detect",
        Some(&conflict_id),
        &serde_json::json!({
            "conflict_id": conflict_id,
            "conflict_type": conflict_type.as_str(),
            "priority": priority,
            "memory_a": memory_a,
            "memory_b": memory_b,
            "entity": entity,
            "rel_type": rel_type,
            "detected_at": ts,
            "detected_by": actor_id,
            "detection_reason": detection_reason,
        }),
        None,
    )?;

    Ok(Conflict {
        conflict_id,
        conflict_type: conflict_type.as_str().to_string(),
        priority: priority.to_string(),
        status: "open".to_string(),
        memory_a: memory_a.to_string(),
        memory_b: memory_b.to_string(),
        entity: entity.map(String::from),
        rel_type: rel_type.map(String::from),
        detected_at: ts,
        detected_by: actor_id,
        detection_reason: detection_reason.to_string(),
        resolved_at: None,
        resolved_by: None,
        strategy: None,
        winner_rid: None,
        resolution_note: None,
    })
}

/// Detect edge-based contradictions for a newly materialized edge.
/// Called from materialize_relate in replication.rs during sync.
pub fn detect_edge_conflicts(
    db: &YantrikDB,
    src: &str,
    dst: &str,
    rel_type: &str,
    incoming_target_rid: Option<&str>,
) -> Result<Vec<Conflict>> {
    let mut conflicts = Vec::new();

    // Only check identity and preference rel_types
    let is_identity = IDENTITY_REL_TYPES.contains(&rel_type);
    let is_preference = PREFERENCE_REL_TYPES.contains(&rel_type);
    if !is_identity && !is_preference {
        return Ok(conflicts);
    }

    // Collect data while holding the conn lock, then release before calling
    // conflict_exists/create_conflict (which also acquire the lock).
    let edge_data: Vec<(String, Option<String>, Option<String>)> = {
        let conn = db.conn();
        let mut stmt = conn.prepare(
            "SELECT edge_id, dst FROM edges
             WHERE src = ?1 AND rel_type = ?2 AND dst != ?3 AND tombstoned = 0",
        )?;

        let existing: Vec<(String, String)> = stmt
            .query_map(params![src, rel_type, dst], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        existing
            .into_iter()
            .map(|(_edge_id, existing_dst)| {
                let memory_a = find_memory_for_edge(&conn, src, &existing_dst, rel_type)
                    .ok()
                    .flatten();
                let memory_b = incoming_target_rid.map(String::from).or_else(|| {
                    find_memory_for_edge(&conn, src, dst, rel_type)
                        .ok()
                        .flatten()
                });
                (existing_dst, memory_a, memory_b)
            })
            .collect()
    }; // conn lock released here

    for (existing_dst, memory_a, memory_b) in edge_data {
        let conflict_type = classify_conflict(rel_type);

        if let (Some(ref mem_a), Some(ref mem_b)) = (&memory_a, &memory_b) {
            if !conflict_pair_eligible(db, mem_a, mem_b)? {
                continue;
            }
            if !conflict_exists(db, mem_a, mem_b)? {
                let conflict = create_conflict(
                    db,
                    &conflict_type,
                    mem_a,
                    mem_b,
                    Some(src),
                    Some(rel_type),
                    &format!(
                        "Entity '{}' has conflicting {} values: '{}' vs '{}'",
                        src, rel_type, existing_dst, dst
                    ),
                )?;
                conflicts.push(conflict);
            }
        }
    }

    Ok(conflicts)
}

/// Full-database conflict scan. Finds all edge-based contradictions
/// and concurrent consolidation conflicts.
pub fn scan_conflicts(db: &YantrikDB) -> Result<Vec<Conflict>> {
    scan_conflicts_limited(db, 50)
}

/// Scan for conflicts with a limit on max conflicts to detect per scan.
///
/// Two passes remain: contradicting edges on the small identity/preference
/// relation whitelist, and concurrent-consolidation conflicts. The third pass
/// this function used to run — pairing memories that shared an entity and
/// diffing their word sets ("date substitution", "temporal substitution") —
/// is GONE. On the hand-labeled production set (2026-08-17) it produced six
/// conflicts, all false: completely unrelated memories flagged because they
/// shared one entity and were written on different days. That is not a
/// contradiction test and could not be tuned into one. Claim-level
/// contradictions are [`scan_claim_conflicts`]'s job.
pub fn scan_conflicts_limited(db: &YantrikDB, max_conflicts: usize) -> Result<Vec<Conflict>> {
    let mut conflicts = Vec::new();

    // Phase 1: Collect edge-based conflict candidates while holding conn lock.
    // Each candidate: (src, rel_type, dst_i, dst_j, mem_a, mem_b)
    let edge_candidates: Vec<(
        String,
        String,
        String,
        String,
        Option<String>,
        Option<String>,
    )>;
    let cm_rows: Vec<(String, String, String)>;

    {
        let conn = db.conn();

        // Scan for contradicting edges: same (src, rel_type) with different dst values
        let mut stmt = conn.prepare(
            "SELECT src, rel_type, GROUP_CONCAT(DISTINCT dst) as dsts, COUNT(DISTINCT dst) as cnt
             FROM edges
             WHERE tombstoned = 0
             GROUP BY src, rel_type
             HAVING cnt > 1",
        )?;

        let rows: Vec<(String, String, String)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let mut manual_stmt = conn.prepare(
            "SELECT COUNT(*) > 0 FROM claims \
             WHERE src = ?1 AND rel_type = ?2 AND extractor = 'manual'",
        )?;

        let mut candidates = Vec::new();
        for (src, rel_type, dsts_csv) in rows {
            let is_identity = IDENTITY_REL_TYPES.contains(&rel_type.as_str());
            let is_preference = PREFERENCE_REL_TYPES.contains(&rel_type.as_str());
            if !is_identity && !is_preference {
                continue;
            }

            // Phantom-subject guard: a subject today's extractor would not
            // mint cannot anchor a conflict — unless the caller deliberately
            // created the relation via relate() (extractor = 'manual'), which
            // protects its endpoints exactly as in the graph-index heal.
            if !subject_admissible(&src) {
                let manual: bool = manual_stmt.query_row(params![src, rel_type], |r| r.get(0))?;
                if !manual {
                    continue;
                }
            }

            let dsts: Vec<String> = dsts_csv.split(',').map(|s| s.trim().to_string()).collect();
            if dsts.len() < 2 {
                continue;
            }

            for i in 0..dsts.len() {
                for j in (i + 1)..dsts.len() {
                    let mem_a = find_memory_for_edge(&conn, &src, &dsts[i], &rel_type)
                        .ok()
                        .flatten();
                    let mem_b = find_memory_for_edge(&conn, &src, &dsts[j], &rel_type)
                        .ok()
                        .flatten();
                    candidates.push((
                        src.clone(),
                        rel_type.clone(),
                        dsts[i].clone(),
                        dsts[j].clone(),
                        mem_a,
                        mem_b,
                    ));
                }
            }
        }
        edge_candidates = candidates;

        // Scan for concurrent consolidation conflicts
        let mut cm_stmt = conn.prepare(
            "SELECT cm1.consolidation_rid, cm2.consolidation_rid, cm1.source_rid
             FROM consolidation_members cm1
             JOIN consolidation_members cm2
               ON cm1.source_rid = cm2.source_rid
              AND cm1.consolidation_rid < cm2.consolidation_rid",
        )?;

        cm_rows = cm_stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
    } // conn lock released here

    // Phase 2: Create conflicts (these functions acquire conn lock internally).

    // Edge-based conflicts
    for (src, rel_type, dst_i, dst_j, mem_a, mem_b) in &edge_candidates {
        if conflicts.len() >= max_conflicts {
            break;
        }
        if let (Some(a), Some(b)) = (mem_a, mem_b) {
            if !conflict_pair_eligible(db, a, b)? {
                continue;
            }
            if !conflict_exists(db, a, b)? {
                let conflict_type = classify_conflict(rel_type);
                let conflict = create_conflict(
                    db,
                    &conflict_type,
                    a,
                    b,
                    Some(src),
                    Some(rel_type),
                    &format!(
                        "Entity '{}' has conflicting {} values: '{}' vs '{}'",
                        src, rel_type, dst_i, dst_j
                    ),
                )?;
                conflicts.push(conflict);
            }
        }
    }

    // Consolidation conflicts
    let mut seen_pairs = std::collections::HashSet::new();
    for (rid_a, rid_b, shared_source) in cm_rows {
        let pair = if rid_a < rid_b {
            (rid_a.clone(), rid_b.clone())
        } else {
            (rid_b.clone(), rid_a.clone())
        };
        if seen_pairs.contains(&pair) {
            continue;
        }
        seen_pairs.insert(pair);

        if !conflict_exists(db, &rid_a, &rid_b)? {
            let conflict = create_conflict(
                db,
                &ConflictType::Consolidation,
                &rid_a,
                &rid_b,
                None,
                None,
                &format!(
                    "Concurrent consolidation: both '{}' and '{}' consumed source '{}'",
                    rid_a, rid_b, shared_source
                ),
            )?;
            conflicts.push(conflict);
        }
    }

    Ok(conflicts)
}

// ── RFC 006 Phase 1: Claim-Aware Conflict Scanner ──

/// Reason codes for claim-based conflicts (RFC 006 Phase 1).
pub mod reason_codes {
    pub const SAME_SUBJECT_SAME_REL_DISTINCT_OBJECT: &str =
        "same_subject_same_relation_distinct_object";
    pub const OVERLAPPING_VALIDITY: &str = "overlapping_validity_windows";
    pub const MISSING_TEMPORAL: &str = "missing_temporal_qualifier";
    pub const POSSIBLE_SUCCESSION: &str = "possible_temporal_succession";
    /// A positive claim and a negative claim about the same (src, rel_type, dst)
    /// exist — someone asserted X and someone else denied X.
    pub const POLARITY_CONTRADICTION: &str = "polarity_contradiction";
    /// A claim has modality other than 'asserted' — reported, hypothetical, denied.
    /// Lower confidence but still relevant for conflict tracking.
    pub const MODALITY_MISMATCH: &str = "modality_mismatch";
}

/// Check if two time intervals overlap.
fn intervals_overlap(
    from_a: Option<f64>,
    to_a: Option<f64>,
    from_b: Option<f64>,
    to_b: Option<f64>,
) -> Option<bool> {
    // If both have at least from or to, we can check
    let start_a = from_a?;
    let start_b = from_b?;
    let end_a = to_a.unwrap_or(f64::MAX);
    let end_b = to_b.unwrap_or(f64::MAX);
    Some(start_a < end_b && start_b < end_a)
}

/// One candidate claim pair from the same-subject-same-relation join.
struct ClaimPairCandidate {
    src: String,
    rel_type: String,
    dst1: String,
    dst2: String,
    vf1: Option<f64>,
    vt1: Option<f64>,
    vf2: Option<f64>,
    vt2: Option<f64>,
    namespace: String,
    extractor1: String,
    extractor2: String,
    created1: f64,
    created2: f64,
}

/// Scan claims for scoped conflicts — the claim-keyed detector.
///
/// Operates on structured claims with polarity, modality, and temporal
/// qualifiers; conflicts key on (subject, relation) pairs, never on shared
/// entities or word bags. Gates, in order:
///
/// - **Functional relations only.** A pair fires only when the relation is in
///   [`FUNCTIONAL_REL_TYPES`] (single-valued); everything else is multi-valued
///   by default and distinct objects are just distinct facts. A relation
///   policy can additionally SUPPRESS (`overlap_allowed = 1`) but no longer
///   ADMITS a relation the functional model excludes — the seeded `leads`
///   policy (`overlap_allowed = 0`) was firing pairwise on extraction junk,
///   so policies are subordinated to the model rather than trusted over it.
/// - **Subject admission** ([`subject_admissible`]): phantom subjects (`546`,
///   ALL-CAPS headings) cannot anchor a conflict; `extractor = 'manual'`
///   claims are exempt (deliberate `relate()` calls).
/// - **Pair eligibility** ([`conflict_pair_eligible`]): never self-conflict;
///   a `supersedes` link between the records suppresses; both records must
///   still be active.
/// - **Temporal awareness**: when validity windows or creation times order
///   the two values, the conflict is classified as a SUCCESSION candidate
///   (`possible_temporal_succession`, priority medium) with a newer-supersedes
///   resolution hint in the reason — never resolved destructively here.
///   Genuinely overlapping closed validity windows stay `high`.
pub fn scan_claim_conflicts(db: &YantrikDB, max_conflicts: usize) -> Result<Vec<Conflict>> {
    let mut conflicts = Vec::new();

    // Phase 1: Query claim groups (same src + rel_type, different dst).
    // Only positive, asserted/reported claims on FUNCTIONAL relations
    // participate — filtering in SQL keeps multi-valued junk from crowding
    // real candidates out of the LIMIT window.
    let functional_in = FUNCTIONAL_REL_TYPES
        .iter()
        .map(|r| format!("'{r}'"))
        .collect::<Vec<_>>()
        .join(", ");

    let candidates: Vec<ClaimPairCandidate> = {
        let conn = db.conn();
        let sql = format!(
            "SELECT e1.src, e1.rel_type, e1.dst, e2.dst,
                    e1.valid_from, e1.valid_to, e2.valid_from, e2.valid_to,
                    e1.namespace, e1.extractor, e2.extractor,
                    e1.created_at, e2.created_at
             FROM edges e1
             JOIN edges e2
               ON e1.src = e2.src
              AND e1.rel_type = e2.rel_type
              AND e1.dst < e2.dst
              AND e1.namespace = e2.namespace
             WHERE e1.tombstoned = 0 AND e2.tombstoned = 0
               AND e1.polarity = 1 AND e2.polarity = 1
               AND e1.modality IN ('asserted', 'reported')
               AND e2.modality IN ('asserted', 'reported')
               AND e1.rel_type IN ({functional_in})
             ORDER BY e1.created_at DESC
             LIMIT ?1"
        );
        let mut stmt = conn.prepare(&sql)?;

        let rows = stmt
            .query_map(params![max_conflicts * 5], |row| {
                Ok(ClaimPairCandidate {
                    src: row.get(0)?,
                    rel_type: row.get(1)?,
                    dst1: row.get(2)?,
                    dst2: row.get(3)?,
                    vf1: row.get(4)?,
                    vt1: row.get(5)?,
                    vf2: row.get(6)?,
                    vt2: row.get(7)?,
                    namespace: row.get(8)?,
                    extractor1: row.get(9)?,
                    extractor2: row.get(10)?,
                    created1: row.get(11)?,
                    created2: row.get(12)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        rows
    }; // conn released

    // Phase 2: Evaluate each candidate pair.
    for c in &candidates {
        if conflicts.len() >= max_conflicts {
            break;
        }

        // Defense in depth: the SQL already filters to functional relations.
        if !is_functional_relation(&c.rel_type) {
            continue;
        }

        // RFC 006 Phase 3 policy, SUPPRESS-only. Look up namespace-specific
        // policy first, then global '*'.
        let policy: Option<(bool, bool, String)> = {
            let conn = db.conn();
            conn.query_row(
                "SELECT overlap_allowed, temporal_required, missing_time_severity \
                 FROM relation_policies \
                 WHERE relation_type = ?1 AND (namespace = ?2 OR namespace = '*') \
                 ORDER BY CASE WHEN namespace = ?2 THEN 0 ELSE 1 END \
                 LIMIT 1",
                params![c.rel_type, c.namespace],
                |row| {
                    Ok((
                        row.get::<_, bool>(0)?,
                        row.get::<_, bool>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                },
            )
            .ok()
        };

        // If policy says overlap is allowed for this relation, skip
        if let Some((overlap_allowed, _, _)) = &policy {
            if *overlap_allowed {
                continue; // Multiple values are normal for this relation
            }
        }

        // Subject admission: phantom subjects cannot anchor a conflict.
        // Deliberate relate() claims (extractor = 'manual') are exempt.
        let deliberate = c.extractor1 == "manual" || c.extractor2 == "manual";
        if !deliberate && !subject_admissible(&c.src) {
            continue;
        }

        // Resolve aliases
        let src_canonical = db.resolve_alias(&c.src, &c.namespace);
        let dst1_canonical = db.resolve_alias(&c.dst1, &c.namespace);
        let dst2_canonical = db.resolve_alias(&c.dst2, &c.namespace);

        if dst1_canonical == dst2_canonical {
            continue;
        }

        // Determine severity — use policy's missing_time_severity if available
        let policy_missing_severity = policy
            .as_ref()
            .map(|(_, _, s)| s.as_str())
            .unwrap_or("medium");
        let policy_temporal_required = policy.as_ref().map(|(_, t, _)| *t).unwrap_or(false);

        let mut reason_codes =
            vec![reason_codes::SAME_SUBJECT_SAME_REL_DISTINCT_OBJECT.to_string()];
        // (older value, newer value) when time orders the pair — carried into
        // the reason as a newer-supersedes resolution hint. Never acted on
        // destructively here.
        let mut succession: Option<(&str, &str)> = None;
        let priority;

        match intervals_overlap(c.vf1, c.vt1, c.vf2, c.vt2) {
            Some(true) if c.vt1.is_none() && c.vt2.is_none() && c.vf1 != c.vf2 => {
                // Both open-ended ("valid until further notice") with ordered
                // starts: the newer assertion opened after the older one.
                // That is the succession shape — "CT128 runs 0.14.1" then
                // "runs 0.15.0", both active, nothing superseded yet — not a
                // simultaneous contradiction.
                let (older, newer) = if c.vf1 < c.vf2 {
                    (c.dst1.as_str(), c.dst2.as_str())
                } else {
                    (c.dst2.as_str(), c.dst1.as_str())
                };
                reason_codes.push(reason_codes::POSSIBLE_SUCCESSION.to_string());
                succession = Some((older, newer));
                priority = "medium";
            }
            Some(true) => {
                // Genuinely overlapping closed windows → real simultaneous
                // contradiction.
                reason_codes.push(reason_codes::OVERLAPPING_VALIDITY.to_string());
                priority = "high";
            }
            Some(false) => {
                // Disjoint windows → succession. Mark the OLDER claim as
                // historical (claim-layer metadata only, non-destructive).
                // A failed UPDATE here previously fell through to `continue`
                // — the older claim stayed current AND the pair was skipped
                // as handled, so a DETECTED contradiction became permanently
                // invisible: not superseded, not flagged, never revisited.
                // Propagate instead; the caller already returns Result.
                let (older_dst, newer_dst, newer_vf) =
                    if c.vf1.unwrap_or(0.0) < c.vf2.unwrap_or(0.0) {
                        (c.dst1.as_str(), c.dst2.as_str(), c.vf2)
                    } else {
                        (c.dst2.as_str(), c.dst1.as_str(), c.vf1)
                    };
                {
                    let conn = db.conn();
                    conn.execute(
                        "UPDATE claims SET valid_to = ?1 \
                         WHERE src = ?2 AND rel_type = ?3 AND dst = ?4 AND valid_to IS NULL AND tombstoned = 0",
                        params![newer_vf.unwrap_or(0.0), c.src, c.rel_type, older_dst],
                    )?;
                }
                // Still FIRE (labeled-set acceptance gate 2): while both
                // RECORDS are active with no supersedes link, the stale value
                // keeps being served — the conflict is the prompt to link it.
                reason_codes.push(reason_codes::POSSIBLE_SUCCESSION.to_string());
                succession = Some((older_dst, newer_dst));
                priority = "medium";
            }
            None => {
                // Validity windows missing → fall back to claim creation
                // order for the succession hint.
                reason_codes.push(reason_codes::MISSING_TEMPORAL.to_string());
                if c.created1 != c.created2 {
                    let (older, newer) = if c.created1 < c.created2 {
                        (c.dst1.as_str(), c.dst2.as_str())
                    } else {
                        (c.dst2.as_str(), c.dst1.as_str())
                    };
                    reason_codes.push(reason_codes::POSSIBLE_SUCCESSION.to_string());
                    succession = Some((older, newer));
                    priority = "medium";
                } else if policy_temporal_required {
                    priority = policy_missing_severity;
                } else {
                    priority = "medium";
                }
            }
        }

        let mut reason = format!(
            "Claim conflict: {} has different {} values: '{}' vs '{}'. Reasons: [{}]",
            src_canonical,
            c.rel_type,
            dst1_canonical,
            dst2_canonical,
            reason_codes.join(", ")
        );
        if let Some((older, newer)) = succession {
            reason.push_str(&format!(
                ". Resolution hint: succession — '{newer}' is the newer assertion and \
                 likely supersedes '{older}'; link the newer record `supersedes` the \
                 older to resolve (non-destructive)"
            ));
        }

        // Find source memory rids for the conflicting claims
        let (mem_a, mem_b) = {
            let conn = db.conn();
            let a = conn.query_row(
                "SELECT source_memory_rid FROM edges WHERE src = ?1 AND rel_type = ?2 AND dst = ?3 AND tombstoned = 0",
                params![c.src, c.rel_type, c.dst1],
                |row| row.get::<_, Option<String>>(0),
            ).ok().flatten();
            let b = conn.query_row(
                "SELECT source_memory_rid FROM edges WHERE src = ?1 AND rel_type = ?2 AND dst = ?3 AND tombstoned = 0",
                params![c.src, c.rel_type, c.dst2],
                |row| row.get::<_, Option<String>>(0),
            ).ok().flatten();
            (a, b)
        };

        // Use source memory rids if available, otherwise use src entity as fallback
        let rid_a = mem_a.unwrap_or_else(|| format!("claim:{}:{}:{}", c.src, c.rel_type, c.dst1));
        let rid_b = mem_b.unwrap_or_else(|| format!("claim:{}:{}:{}", c.src, c.rel_type, c.dst2));

        // Never self-conflict; a supersedes link or an inactive record
        // suppresses the pair.
        if !conflict_pair_eligible(db, &rid_a, &rid_b)? {
            continue;
        }

        if !conflict_exists(db, &rid_a, &rid_b)? {
            let mut conflict = create_conflict(
                db,
                &ConflictType::IdentityFact,
                &rid_a,
                &rid_b,
                Some(&src_canonical),
                Some(&c.rel_type),
                &reason,
            )?;

            // Override priority based on our temporal analysis
            {
                let conn = db.conn();
                conn.execute(
                    "UPDATE conflicts SET priority = ?1 WHERE conflict_id = ?2",
                    params![priority, conflict.conflict_id],
                )?;
            }
            conflict.priority = priority.to_string();

            conflicts.push(conflict);
        }
    }

    // RFC 006 Phase 4: polarity contradiction scan.
    // Find cases where the SAME (src, rel_type, dst) has both positive and
    // negative polarity claims — someone asserted X and someone denied X.
    if conflicts.len() < max_conflicts {
        let polarity_candidates: Vec<(
            String,
            String,
            String,
            String,
            Option<String>,
            Option<String>,
        )>;
        {
            let conn = db.conn();
            let mut stmt = conn.prepare(
                "SELECT e1.src, e1.rel_type, e1.dst, e1.namespace,
                        e1.source_memory_rid, e2.source_memory_rid
                 FROM edges e1
                 JOIN edges e2
                   ON e1.src = e2.src AND e1.rel_type = e2.rel_type AND e1.dst = e2.dst
                   AND e1.namespace = e2.namespace
                 WHERE e1.polarity = 1 AND e2.polarity = -1
                   AND e1.tombstoned = 0 AND e2.tombstoned = 0
                 LIMIT ?1",
            )?;
            let rows: Vec<_> = stmt
                .query_map(params![max_conflicts - conflicts.len()], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, Option<String>>(4)?,
                        row.get::<_, Option<String>>(5)?,
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            polarity_candidates = rows;
        }

        for (src, rel_type, dst, _ns, mem_a, mem_b) in &polarity_candidates {
            let rid_a = mem_a.as_deref().unwrap_or("unknown");
            let rid_b = mem_b.as_deref().unwrap_or("unknown");
            if rid_a == rid_b {
                // One record both asserting and denying is an intra-record
                // ambiguity, not a conflict pair — never self-conflict.
                continue;
            }
            if rid_a != "unknown" && rid_b != "unknown" && !conflict_exists(db, rid_a, rid_b)? {
                let reason = format!(
                    "Polarity contradiction: '{}' has both positive and negative claims for {} → {}. Reasons: [{}]",
                    src, rel_type, dst, reason_codes::POLARITY_CONTRADICTION
                );
                let conflict = create_conflict(
                    db,
                    &ConflictType::IdentityFact,
                    rid_a,
                    rid_b,
                    Some(src),
                    Some(rel_type),
                    &reason,
                )?;
                // Polarity contradictions are always high priority
                {
                    let conn = db.conn();
                    conn.execute(
                        "UPDATE conflicts SET priority = 'high' WHERE conflict_id = ?1",
                        params![conflict.conflict_id],
                    )?;
                }
                conflicts.push(conflict);
            }
        }
    }

    Ok(conflicts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim).map(|i| (seed + i as f32) * 0.1).collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    }

    fn empty_meta() -> serde_json::Value {
        serde_json::json!({})
    }

    #[test]
    fn test_create_conflict() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let rid_a = db
            .record(
                "User likes coffee",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(1.0, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        let rid_b = db
            .record(
                "User likes tea",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(2.0, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        let conflict = create_conflict(
            &db,
            &ConflictType::Preference,
            &rid_a,
            &rid_b,
            Some("User"),
            Some("prefers"),
            "User has conflicting preference: coffee vs tea",
        )
        .unwrap();

        assert_eq!(conflict.status, "open");
        assert_eq!(conflict.conflict_type, "preference");
        assert_eq!(conflict.priority, "high");
        assert_eq!(conflict.memory_a, rid_a);
        assert_eq!(conflict.memory_b, rid_b);
    }

    #[test]
    fn test_conflict_dedup() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let rid_a = db
            .record(
                "a",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(1.0, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        let rid_b = db
            .record(
                "b",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(2.0, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        assert!(!conflict_exists(&db, &rid_a, &rid_b).unwrap());
        create_conflict(
            &db,
            &ConflictType::Minor,
            &rid_a,
            &rid_b,
            None,
            None,
            "test",
        )
        .unwrap();
        assert!(conflict_exists(&db, &rid_a, &rid_b).unwrap());
        assert!(conflict_exists(&db, &rid_b, &rid_a).unwrap()); // reversed order
    }

    #[test]
    fn test_classify_conflict() {
        assert_eq!(classify_conflict("birthday"), ConflictType::IdentityFact);
        assert_eq!(classify_conflict("works_at"), ConflictType::IdentityFact);
        assert_eq!(classify_conflict("married_to"), ConflictType::IdentityFact);
        assert_eq!(classify_conflict("lives_in"), ConflictType::IdentityFact);
        assert_eq!(classify_conflict("favorite"), ConflictType::Preference);
        assert_eq!(classify_conflict("prefers"), ConflictType::Preference);
        assert_eq!(classify_conflict("random_rel"), ConflictType::Minor);
    }

    #[test]
    fn test_scan_contradicting_edges() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        db.relate("User", "Google", "works_at", 1.0).unwrap();
        db.relate("User", "Meta", "works_at", 1.0).unwrap();

        let conflicts = scan_conflicts(&db).unwrap();
        assert!(!conflicts.is_empty());
        assert_eq!(conflicts[0].conflict_type, "identity_fact");
        assert_eq!(conflicts[0].entity.as_deref(), Some("User"));
    }

    #[test]
    fn test_scan_no_conflict_for_non_identity_edges() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        db.relate("User", "Alice", "friends_with", 1.0).unwrap();
        db.relate("User", "Bob", "friends_with", 1.0).unwrap();

        let conflicts = scan_conflicts(&db).unwrap();
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_conflict_type_default_priorities() {
        assert_eq!(ConflictType::IdentityFact.default_priority(), "critical");
        assert_eq!(ConflictType::Preference.default_priority(), "high");
        assert_eq!(ConflictType::Temporal.default_priority(), "high");
        assert_eq!(ConflictType::Consolidation.default_priority(), "medium");
        assert_eq!(ConflictType::Minor.default_priority(), "low");
    }

    // ── The labeled-set gate (2026-08-17): every shape below reconstructs one
    // of the sixteen hand-labeled production false positives (classes A/B) or
    // the true-positive shape the old detector never found (class C). The
    // class A/B tests all FAIL on the pre-rewrite detector.

    /// Class A — word-bag "date substitution". Two memories about COMPLETELY
    /// different projects, paired solely because they share the entity
    /// "Pranab" and carry different date tokens. Six of the sixteen labeled
    /// false positives had this shape; the word-bag scan that produced them
    /// is removed, not tuned.
    #[test]
    fn class_a_unrelated_dated_memories_sharing_entity_do_not_conflict() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        // cosine ≈ 0.8 (same-topic band for the removed heuristic), low word
        // overlap, shared entity, differing date-like tokens (5 vs 11) — the
        // exact firing condition of the removed classifier.
        let mut e1 = vec![0.0f32; 8];
        e1[0] = 1.0;
        let mut e2 = vec![0.0f32; 8];
        e2[0] = 0.8;
        e2[1] = 0.6;
        let m1 = db
            .record(
                "Pranab shipped the chitta-canon constitutional work on August 5",
                "episodic",
                0.7,
                0.0,
                604800.0,
                &empty_meta(),
                &e1,
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        let m2 = db
            .record(
                "Pranab preregistered the yantrikdb-agi study design on August 11",
                "episodic",
                0.7,
                0.0,
                604800.0,
                &empty_meta(),
                &e2,
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        // Link both memories to the shared entity, as the maintenance-cycle
        // entity backfill does on the production store (where all six labeled
        // class A conflicts were anchored on exactly this join).
        {
            let conn = db.conn();
            for rid in [&m1, &m2] {
                conn.execute(
                    "INSERT OR IGNORE INTO memory_entities (memory_rid, entity_name)                      VALUES (?1, 'Pranab')",
                    params![rid],
                )
                .unwrap();
            }
        }

        let conflicts = scan_conflicts(&db).unwrap();
        assert!(
            conflicts.is_empty(),
            "different topics on different days are not contradictions, got: {:?}",
            conflicts
                .iter()
                .map(|c| c.detection_reason.as_str())
                .collect::<Vec<_>>()
        );
        let claim_conflicts = scan_claim_conflicts(&db, 50).unwrap();
        assert!(claim_conflicts.is_empty());
    }

    /// Class B — a multi-valued relation ("created") must not conflict
    /// pairwise. One "Pranab created {four projects}" fact generated FIVE
    /// pairwise conflicts on the labeled set; the default is now
    /// multi-valued (no conflict) and "created" is not in the functional set.
    #[test]
    fn class_b_multi_valued_relation_no_pairwise_conflicts() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        for dst in ["SDF Protocol", "YantrikDB", "Yantrik Memory", "ClawColab"] {
            db.relate("Pranab", dst, "created", 1.0).unwrap();
        }
        assert!(
            scan_claim_conflicts(&db, 50).unwrap().is_empty(),
            "'created' is many-valued; distinct objects are distinct facts"
        );
        assert!(scan_conflicts(&db).unwrap().is_empty());
    }

    /// Class B — phantom subjects (bare numbers, ALL-CAPS headings) cannot
    /// anchor a conflict, even on a FUNCTIONAL relation. Reuses the 0.14.1
    /// graph-heal predicate; the claims here carry a non-manual extractor,
    /// so no deliberate-relate exemption applies.
    #[test]
    fn class_b_phantom_subject_cannot_anchor_conflict() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let m1 = db
            .record(
                "progress note one",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(1.0, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        let m2 = db
            .record(
                "progress note two",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(2.0, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        // Bare-number subject ("546 leads 13 vs 15" in the labeled set).
        for (dst, mem) in [("13", &m1), ("15", &m2)] {
            db.ingest_claim(
                "546",
                "is_at_version",
                dst,
                "default",
                1,
                "asserted",
                None,
                None,
                "heuristic_v1",
                None,
                "medium",
                Some(mem),
                None,
                None,
                1.0,
            )
            .unwrap();
        }
        // ALL-CAPS heading subject — today's extractor would never mint it.
        for (dst, mem) in [("done", &m1), ("pending", &m2)] {
            db.ingest_claim(
                "USER MUST UPDATE MCP CONFIG",
                "is",
                dst,
                "default",
                1,
                "asserted",
                None,
                None,
                "heuristic_v1",
                None,
                "medium",
                Some(mem),
                None,
                None,
                1.0,
            )
            .unwrap();
        }

        let conflicts = scan_claim_conflicts(&db, 50).unwrap();
        assert!(
            conflicts.is_empty(),
            "phantom subjects must not anchor conflicts, got: {:?}",
            conflicts
                .iter()
                .map(|c| c.detection_reason.as_str())
                .collect::<Vec<_>>()
        );
    }

    /// Class B — a single record mentioning a progression ("13 → 15") must
    /// never be flagged against itself. Guarded at the detection site AND at
    /// the engine boundary (create_conflict rejects memory_a == memory_b).
    #[test]
    fn class_b_self_conflict_never_fires() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let m1 = db
            .record(
                "CT128 went from engine 0.14.1 to 0.15.0 today",
                "episodic",
                0.7,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(1.0, 8),
                "default",
                0.8,
                "infra",
                "user",
                None,
            )
            .unwrap();
        // Both values extracted from the SAME record — a progression, not a
        // contradiction. "runs" IS functional and "CT128" IS admissible, so
        // only the self-guard stands between this and a false positive.
        for dst in ["0.14.1", "0.15.0"] {
            db.ingest_claim(
                "CT128",
                "runs",
                dst,
                "default",
                1,
                "asserted",
                None,
                None,
                "heuristic_v1",
                None,
                "medium",
                Some(&m1),
                None,
                None,
                1.0,
            )
            .unwrap();
        }

        let conflicts = scan_claim_conflicts(&db, 50).unwrap();
        assert!(
            conflicts.is_empty(),
            "a record cannot conflict with itself, got: {:?}",
            conflicts
                .iter()
                .map(|c| (c.memory_a.as_str(), c.memory_b.as_str()))
                .collect::<Vec<_>>()
        );

        // Engine boundary: the whole category is unrepresentable.
        assert!(
            create_conflict(&db, &ConflictType::IdentityFact, &m1, &m1, None, None, "x").is_err(),
            "create_conflict must reject memory_a == memory_b"
        );
    }

    /// Class C — the true-positive shape the old detector never found:
    /// same subject + FUNCTIONAL attribute + different value + both records
    /// ACTIVE + no supersession link. Fires, classified as a succession
    /// candidate with a newer-supersedes resolution hint — never resolved
    /// destructively by the scanner itself.
    #[test]
    fn true_positive_functional_succession_fires_with_temporal_hint() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let m1 = db
            .record(
                "CT128 runs engine 0.14.1",
                "semantic",
                0.8,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(1.0, 8),
                "default",
                0.8,
                "infra",
                "user",
                None,
            )
            .unwrap();
        let m2 = db
            .record(
                "CT128 runs engine 0.15.0",
                "semantic",
                0.8,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(2.0, 8),
                "default",
                0.8,
                "infra",
                "user",
                None,
            )
            .unwrap();
        // Open-ended validity windows with ordered starts — the succession
        // shape ("valid until further notice", newer opened later).
        db.ingest_claim(
            "CT128",
            "runs",
            "0.14.1",
            "default",
            1,
            "asserted",
            Some(1000.0),
            None,
            "heuristic_v1",
            None,
            "medium",
            Some(&m1),
            None,
            None,
            1.0,
        )
        .unwrap();
        db.ingest_claim(
            "CT128",
            "runs",
            "0.15.0",
            "default",
            1,
            "asserted",
            Some(2000.0),
            None,
            "heuristic_v1",
            None,
            "medium",
            Some(&m2),
            None,
            None,
            1.0,
        )
        .unwrap();

        let conflicts = scan_claim_conflicts(&db, 50).unwrap();
        assert_eq!(conflicts.len(), 1, "the succession case must fire");
        let c = &conflicts[0];
        let pair = [c.memory_a.as_str(), c.memory_b.as_str()];
        assert!(pair.contains(&m1.as_str()) && pair.contains(&m2.as_str()));
        assert!(
            c.detection_reason
                .contains(reason_codes::POSSIBLE_SUCCESSION),
            "must carry the temporal qualifier, got: {}",
            c.detection_reason
        );
        assert!(
            c.detection_reason
                .contains("'0.15.0' is the newer assertion"),
            "must hint newer-supersedes with the newer value named, got: {}",
            c.detection_reason
        );
        assert_eq!(c.priority, "medium", "succession is review, not critical");
        // Never destructive: both records still active after the scan.
        for rid in [&m1, &m2] {
            let status: String = db
                .conn()
                .query_row(
                    "SELECT consolidation_status FROM memories WHERE rid = ?1",
                    params![rid],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(status, "active");
        }
    }

    /// Class C, resolved arm — the same succession pair with a `supersedes`
    /// link between the records must NOT fire: the link IS the resolution
    /// the conflict exists to bring about.
    #[test]
    fn true_positive_suppressed_by_supersession_link() {
        use crate::types::{LinkType, RecordLink};

        let db = YantrikDB::new(":memory:", 8).unwrap();
        let m1 = db
            .record(
                "CT128 runs engine 0.14.1",
                "semantic",
                0.8,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(1.0, 8),
                "default",
                0.8,
                "infra",
                "user",
                None,
            )
            .unwrap();
        let m2 = db
            .record(
                "CT128 runs engine 0.15.0",
                "semantic",
                0.8,
                0.0,
                604800.0,
                &empty_meta(),
                &vec_seed(2.0, 8),
                "default",
                0.8,
                "infra",
                "user",
                None,
            )
            .unwrap();
        db.ingest_claim(
            "CT128",
            "runs",
            "0.14.1",
            "default",
            1,
            "asserted",
            Some(1000.0),
            None,
            "heuristic_v1",
            None,
            "medium",
            Some(&m1),
            None,
            None,
            1.0,
        )
        .unwrap();
        db.ingest_claim(
            "CT128",
            "runs",
            "0.15.0",
            "default",
            1,
            "asserted",
            Some(2000.0),
            None,
            "heuristic_v1",
            None,
            "medium",
            Some(&m2),
            None,
            None,
            1.0,
        )
        .unwrap();

        // The newer record supersedes the older (edge direction new → old).
        db.link(
            &m2,
            &RecordLink {
                target_rid: m1.clone(),
                link_type: LinkType::Supersedes,
            },
        )
        .unwrap();

        let conflicts = scan_claim_conflicts(&db, 50).unwrap();
        assert!(
            conflicts.is_empty(),
            "a supersession link resolves the succession — no conflict, got: {:?}",
            conflicts
                .iter()
                .map(|c| c.detection_reason.as_str())
                .collect::<Vec<_>>()
        );
    }
}
