//! RFC 008 Phase 1: Warrant Flow — the control stack that replaces scalar
//! confidence. This module implements the three operators (⊕, ⋈, ↝_m) and
//! the mobility state M(c|ρ) that they operate on.
//!
//! Start here when reading: the mobility state is a 13-dim vector per
//! (proposition_id, regime). It is NOT a confidence score. It represents
//! how the claim's warrant is moving through its epistemic neighborhood:
//!
//!   σ — support mass (weighted, dependence-discounted)
//!   α — attack mass
//!   δ — source diversity
//!   ι — effective independence (ratio of support to raw sum)
//!   τ — temporal coherence
//!   γ — transportability across regimes
//!   μ — mutability
//!   λ — load-bearingness in downstream dependencies
//!   χ — modality consilience (cross-modal independent corroboration)
//!   ψ_l — self-generation ratio (immediate)
//!   ψ_a — self-generation ratio (ancestral)
//!   κ — contamination risk (shared pipelines)
//!   ν — novelty isolation
//!
//! Components are materialized at three tiers:
//!   write (<10ms per claim insert): σ, α, χ, ψ_l
//!   read (50-200ms, query-cached): Γ, R_c(u)
//!   background (async snapshot-tagged): ψ_a, κ, ν, τ, γ, μ, λ, δ, ι refinements
//!
//! M3 locked spec (Saga note 14, GPT-5.4 red-team session bab6d0b7):
//!   ⊕ is a *deterministic functional over the live claim set* — not a
//!   stream-fold, not order-sensitive. For each live claim k, overlap is
//!   measured against the union of *all other* live claims' lineages
//!   (leave-one-out symmetric). Adding or removing a claim is not a local
//!   delta; it is a full recompute. The recompute is idempotent via
//!   content_hash — if the hash of the current live claim set matches the
//!   stored hash, we skip the UPSERT.

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::error::Result;

/// Current formula version. Bump when any part of the ⊕ definition, the
/// per-dimension overlap semantics, or the content_hash input changes —
/// doing so causes existing `mobility_state` rows to be recomputed the
/// next time they are accessed (state_status='stale_formula'). Background
/// reconciler upgrades stale rows proactively.
pub const FORMULA_VERSION: i32 = 1;

/// Dependence-discount penalty weights for the accumulation operator ⊕.
/// Fixed in v1; RFC 008 Phase 2 will make them learned.
///
///   ω_k = 1 / (1 + 0.5·D_k + 0.3·P_k + 0.7·S_k)
///
/// where D_k, P_k, S_k are the leave-one-out overlaps described on
/// `accumulate_mass`. Rationale for the ratios:
///   0.5 — source overlap is the primary anti-echo-chamber lever
///   0.3 — pipeline overlap is secondary (same extractor ≠ full duplication)
///   0.7 — self-generation is the strongest discount (prevents self-loops)
pub const DEPENDENCE_WEIGHT_SOURCE: f64 = 0.5;
pub const DEPENDENCE_WEIGHT_PIPELINE: f64 = 0.3;
pub const DEPENDENCE_WEIGHT_SELF_GEN: f64 = 0.7;

/// Max modality count used to normalize χ into [0, 1]: text, image,
/// numeric, audio, code, telemetry.
const MAX_MODALITIES: f64 = 6.0;

/// State lifecycle values for `mobility_state.state_status`.
pub mod state_status {
    pub const FRESH: &str = "fresh";
    pub const RECOMPUTING: &str = "recomputing";
    pub const FAILED: &str = "failed";
    pub const STALE_FORMULA: &str = "stale_formula";
}

/// A single row of the `mobility_state` table. All 13 mobility components
/// are Option because they are populated at different tiers — a freshly
/// recomputed write-tier row only fills in σ, α, χ, ψ_l; the rest are
/// NULL until the background job computes them.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MobilityState {
    pub proposition_id: String,
    pub regime: String,
    pub snapshot_ts: f64,
    pub support_mass: Option<f64>,            // σ
    pub attack_mass: Option<f64>,             // α
    pub source_diversity: Option<f64>,        // δ
    pub effective_independence: Option<f64>,  // ι
    pub temporal_coherence: Option<f64>,      // τ
    pub transportability: Option<f64>,        // γ
    pub mutability: Option<f64>,              // μ
    pub load_bearingness: Option<f64>,        // λ
    pub modality_consilience: Option<f64>,    // χ
    pub self_gen_local: Option<f64>,          // ψ_l
    pub self_gen_ancestral: Option<f64>,      // ψ_a
    pub contamination_risk: Option<f64>,      // κ
    pub novelty_isolation: Option<f64>,       // ν
    /// JSON array of component names populated at each tier.
    pub tier_write_components: Vec<String>,
    pub tier_read_components: Vec<String>,
    pub tier_bg_components: Vec<String>,
    /// M3 reproducibility fields.
    pub formula_version: i32,
    pub content_hash: String,
    pub live_claim_count: i64,
    pub state_status: String,
    pub computed_at: i64,
}

/// A minimal projection of a claim row containing only the fields needed
/// to compute write-tier mobility contributions. Fetched with a narrow
/// SELECT to keep the hot path fast.
#[derive(Debug, Clone)]
struct ClaimRow {
    claim_id: String,
    polarity: i32,
    weight: f64,
    extractor: String,
    source_lineage: Vec<String>, // normalized to deduped, sorted
    self_generated: bool,
    modality_signal: String,
}

impl crate::engine::YantrikDB {
    /// Compute the write-tier mobility state for (proposition_id, regime).
    ///
    /// Reads all live (non-tombstoned) claims, computes a content_hash
    /// over the normalized input. If a mobility_state row with matching
    /// hash already exists, returns it unchanged (idempotent — repeated
    /// calls on the same live set are free). Otherwise, runs the full
    /// ⊕ accumulation and upserts a fresh row with state_status='fresh'.
    ///
    /// This is the hot-path entry point from the ingestion hook and the
    /// tombstone path. Transaction scope is controlled by the caller —
    /// typically `ingest_claim` wraps both the INSERT and this call in
    /// one BEGIN IMMEDIATE → COMMIT.
    pub fn compute_write_tier_mobility(
        &self,
        proposition_id: &str,
        regime: &str,
    ) -> Result<MobilityState> {
        let conn = self.conn.lock();
        compute_write_tier_mobility_conn(&conn, proposition_id, regime)
    }

    /// Read mobility state for (proposition_id, regime). Returns the most
    /// recent snapshot by snapshot_ts. None if none recorded.
    pub fn get_mobility_state(
        &self,
        proposition_id: &str,
        regime: &str,
    ) -> Result<Option<MobilityState>> {
        let conn = self.conn.lock();
        read_latest_state(&conn, proposition_id, regime)
    }

    /// Upsert mobility state, replacing any row with the same
    /// (proposition_id, regime, snapshot_ts) key. Usually called by
    /// `compute_write_tier_mobility`; background tiers will also use it
    /// to append snapshot-tagged rows with later snapshot_ts values.
    pub fn upsert_mobility_state(&self, state: &MobilityState) -> Result<()> {
        let conn = self.conn.lock();
        upsert_mobility_state_inner(&conn, state)
    }
}

// ──────────────────────────────────────────────────────────────────
// Connection-level helpers. These accept a `&Connection` so callers
// can use them inside their own transaction scope.
// ──────────────────────────────────────────────────────────────────

fn read_latest_state(
    conn: &Connection,
    proposition_id: &str,
    regime: &str,
) -> Result<Option<MobilityState>> {
    let mut stmt = conn.prepare(
        "SELECT proposition_id, regime, snapshot_ts, \
         support_mass, attack_mass, source_diversity, effective_independence, \
         temporal_coherence, transportability, mutability, load_bearingness, \
         modality_consilience, self_gen_local, self_gen_ancestral, \
         contamination_risk, novelty_isolation, \
         tier_write_components, tier_read_components, tier_bg_components, \
         formula_version, content_hash, live_claim_count, state_status, computed_at \
         FROM mobility_state \
         WHERE proposition_id = ?1 AND regime = ?2 \
         ORDER BY snapshot_ts DESC LIMIT 1",
    )?;
    let result = stmt.query_row(params![proposition_id, regime], row_to_mobility_state);
    match result {
        Ok(state) => Ok(Some(state)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn upsert_mobility_state_inner(conn: &Connection, state: &MobilityState) -> Result<()> {
    let tier_write = serde_json::to_string(&state.tier_write_components)?;
    let tier_read = serde_json::to_string(&state.tier_read_components)?;
    let tier_bg = serde_json::to_string(&state.tier_bg_components)?;
    conn.execute(
        "INSERT INTO mobility_state (\
         proposition_id, regime, snapshot_ts, \
         support_mass, attack_mass, source_diversity, effective_independence, \
         temporal_coherence, transportability, mutability, load_bearingness, \
         modality_consilience, self_gen_local, self_gen_ancestral, \
         contamination_risk, novelty_isolation, \
         tier_write_components, tier_read_components, tier_bg_components, \
         formula_version, content_hash, live_claim_count, state_status, computed_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, \
                 ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24) \
         ON CONFLICT(proposition_id, regime, snapshot_ts) DO UPDATE SET \
         support_mass = excluded.support_mass, \
         attack_mass = excluded.attack_mass, \
         source_diversity = excluded.source_diversity, \
         effective_independence = excluded.effective_independence, \
         temporal_coherence = excluded.temporal_coherence, \
         transportability = excluded.transportability, \
         mutability = excluded.mutability, \
         load_bearingness = excluded.load_bearingness, \
         modality_consilience = excluded.modality_consilience, \
         self_gen_local = excluded.self_gen_local, \
         self_gen_ancestral = excluded.self_gen_ancestral, \
         contamination_risk = excluded.contamination_risk, \
         novelty_isolation = excluded.novelty_isolation, \
         tier_write_components = excluded.tier_write_components, \
         tier_read_components = excluded.tier_read_components, \
         tier_bg_components = excluded.tier_bg_components, \
         formula_version = excluded.formula_version, \
         content_hash = excluded.content_hash, \
         live_claim_count = excluded.live_claim_count, \
         state_status = excluded.state_status, \
         computed_at = excluded.computed_at",
        params![
            state.proposition_id, state.regime, state.snapshot_ts,
            state.support_mass, state.attack_mass, state.source_diversity,
            state.effective_independence, state.temporal_coherence,
            state.transportability, state.mutability, state.load_bearingness,
            state.modality_consilience, state.self_gen_local,
            state.self_gen_ancestral, state.contamination_risk,
            state.novelty_isolation,
            tier_write, tier_read, tier_bg,
            state.formula_version, state.content_hash, state.live_claim_count,
            state.state_status, state.computed_at,
        ],
    )?;
    Ok(())
}

/// Connection-level variant of `compute_write_tier_mobility` so the
/// ingestion hook can call it from within an already-locked connection
/// scope (same transaction as the claim INSERT).
pub(super) fn compute_write_tier_mobility_conn(
    conn: &Connection,
    proposition_id: &str,
    regime: &str,
) -> Result<MobilityState> {
    let claims = fetch_claims_for_mobility(conn, proposition_id, regime)?;
    let hash = content_hash(&claims);

    // Idempotence: if the latest stored row was computed under the current
    // FORMULA_VERSION with an identical content_hash and is still fresh,
    // no recompute is needed. Repeated inserts of the same (src, dst, rel,
    // extractor, polarity, namespace) resolve to the same set of live
    // claims, so this path is the common case on UPSERT.
    if let Some(existing) = read_latest_state(conn, proposition_id, regime)? {
        if existing.formula_version == FORMULA_VERSION
            && existing.content_hash == hash
            && existing.state_status == state_status::FRESH
        {
            return Ok(existing);
        }
    }

    let support: Vec<&ClaimRow> = claims.iter().filter(|c| c.polarity == 1).collect();
    let attack: Vec<&ClaimRow> = claims.iter().filter(|c| c.polarity == -1).collect();

    let support_mass = accumulate_mass(&support);
    let attack_mass = accumulate_mass(&attack);
    let chi = compute_modality_consilience(&support);
    let psi_l = compute_self_gen_local(&support);

    let state = MobilityState {
        proposition_id: proposition_id.to_string(),
        regime: regime.to_string(),
        snapshot_ts: crate::engine::now(),
        support_mass: Some(support_mass),
        attack_mass: Some(attack_mass),
        modality_consilience: Some(chi),
        self_gen_local: Some(psi_l),
        tier_write_components: vec![
            "support_mass".into(),
            "attack_mass".into(),
            "modality_consilience".into(),
            "self_gen_local".into(),
        ],
        formula_version: FORMULA_VERSION,
        content_hash: hash,
        live_claim_count: claims.len() as i64,
        state_status: state_status::FRESH.to_string(),
        computed_at: unix_seconds(),
        ..Default::default()
    };

    upsert_mobility_state_inner(conn, &state)?;
    Ok(state)
}

// ──────────────────────────────────────────────────────────────────
// Free functions — the actual math, testable without the engine.
// ──────────────────────────────────────────────────────────────────

/// Fetch a narrow projection of claim rows for (proposition_id, regime),
/// deduping and sorting each lineage inline so the downstream hash and
/// Jaccard math is deterministic. ORDER BY claim_id for stable hashes.
fn fetch_claims_for_mobility(
    conn: &Connection,
    proposition_id: &str,
    regime: &str,
) -> Result<Vec<ClaimRow>> {
    let mut stmt = conn.prepare(
        "SELECT claim_id, polarity, weight, extractor, source_lineage, self_generated, modality_signal \
         FROM claims \
         WHERE proposition_id = ?1 AND regime_tag = ?2 AND tombstoned = 0 \
         ORDER BY claim_id ASC",
    )?;
    let rows = stmt
        .query_map(params![proposition_id, regime], |row| {
            let lineage_json: String = row.get(4)?;
            let lineage = normalize_lineage(&lineage_json);
            let self_gen_int: i64 = row.get(5)?;
            Ok(ClaimRow {
                claim_id: row.get(0)?,
                polarity: row.get(1)?,
                weight: row.get(2)?,
                extractor: row.get(3)?,
                source_lineage: lineage,
                self_generated: self_gen_int != 0,
                modality_signal: row.get(6)?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Normalize a source_lineage JSON string into a deduped, sorted vector.
/// NULL and unparseable input become empty vectors. This normalization is
/// also what makes content_hash reproducible across equivalent inputs.
fn normalize_lineage(lineage_json: &str) -> Vec<String> {
    let parsed: Vec<String> = serde_json::from_str(lineage_json).unwrap_or_default();
    let mut set: Vec<String> = parsed.into_iter().collect::<HashSet<_>>().into_iter().collect();
    set.sort();
    set
}

/// Compute the M3 content_hash. Inputs are normalized and sorted so
/// equivalent claim sets produce identical hashes regardless of row
/// order, extractor casing insignificance, or JSON serialization noise.
///
/// The hash covers:
///   1. FORMULA_VERSION (so a version bump invalidates every row)
///   2. sorted claim_ids
///   3. per-claim: polarity, self_generated flag, extractor, modality_signal
///   4. per-claim: sorted source_lineage
///   5. claim weight (3 decimal places to avoid float drift)
///
/// NOT in the hash: created_at, any mutable metadata. We are hashing the
/// *semantic input to the accumulator*, not the row metadata.
fn content_hash(claims: &[ClaimRow]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"yantrikdb.warrant.v");
    hasher.update(&FORMULA_VERSION.to_le_bytes());
    hasher.update(b"\x00claims\x00");
    // claims are already ORDER BY claim_id from fetch_claims_for_mobility,
    // but we do not rely on that — hash is stable under any input order
    // because we re-sort defensively. This also lets the pure-function
    // accumulator tests call content_hash directly.
    let mut indexed: Vec<(&String, &ClaimRow)> = claims.iter().map(|c| (&c.claim_id, c)).collect();
    indexed.sort_by(|a, b| a.0.cmp(b.0));
    for (cid, c) in indexed {
        hasher.update(cid.as_bytes());
        hasher.update(b"|");
        hasher.update(&c.polarity.to_le_bytes());
        hasher.update(b"|");
        hasher.update(&[c.self_generated as u8]);
        hasher.update(b"|");
        hasher.update(c.extractor.as_bytes());
        hasher.update(b"|");
        hasher.update(c.modality_signal.as_bytes());
        hasher.update(b"|");
        // weight rounded to millesimals for hash stability across f64
        // representation variations; the arithmetic still uses full precision.
        let w_scaled = (c.weight * 1000.0).round() as i64;
        hasher.update(&w_scaled.to_le_bytes());
        hasher.update(b"|");
        for src in &c.source_lineage {
            hasher.update(src.as_bytes());
            hasher.update(b",");
        }
        hasher.update(b"\x00");
    }
    hex::encode(hasher.finalize().as_bytes())
}

fn unix_seconds() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// ⊕ accumulation: sum of weighted supports with leave-one-out dependence
/// discount. For each claim k:
///
///   ω_k = 1 / (1 + λ·D_k + μ·P_k + τ·S_k)
///
/// where the three overlaps are measured against the rest of the live set:
///
///   D_k — Jaccard of source_lineage(k) vs. union of all others' lineages
///   P_k — fraction of other claims sharing the same extractor
///   S_k — 1 if claim k is self-generated AND any other claim is, else 0
///
/// The result is a **deterministic functional over the live claim set** —
/// order-invariant, tombstone-correct. P_k and S_k are degenerate overlap
/// proxies in v1: P_k could become Jaccard over pipeline_lineage once that
/// column exists on claims; S_k could become Jaccard over a self-gen
/// ancestry lineage. Both preserve the set-symmetry property regardless.
pub(super) fn accumulate_mass(claims: &[&ClaimRow]) -> f64 {
    if claims.is_empty() {
        return 0.0;
    }
    // Per-dimension element frequency map for source_lineage. Each entry
    // counts how many claims in the live set contain that element.
    // Using this we can derive each claim's leave-one-out Jaccard in
    // O(|lineage_k|) rather than scanning all other claims.
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for c in claims {
        // claim's lineage is already normalized (deduped + sorted), so
        // each element contributes exactly once per claim.
        for e in &c.source_lineage {
            *freq.entry(e.as_str()).or_insert(0) += 1;
        }
    }
    let total_distinct = freq.len();

    let mut total = 0.0;
    for (k, claim) in claims.iter().enumerate() {
        let d_k = leave_one_out_jaccard(&claim.source_lineage, &freq, total_distinct);
        let p_k = pipeline_overlap_ratio(claim, claims, k);
        let s_k = self_gen_overlap_binary(claim, claims, k);
        let discount = 1.0
            + DEPENDENCE_WEIGHT_SOURCE * d_k
            + DEPENDENCE_WEIGHT_PIPELINE * p_k
            + DEPENDENCE_WEIGHT_SELF_GEN * s_k;
        total += (1.0 / discount) * claim.weight;
    }
    total
}

/// Leave-one-out Jaccard of a claim's source_lineage set X against the
/// union U of every *other* claim's source_lineage.
///
///   intersection(X, U) = elements of X that appear in >= 1 other claim
///   union(X, U) = |total_distinct| − |elements only in this claim|
///                 + 0 (since X is a subset of {e : freq[e] ≥ 1})
///                 ... equivalently |total_distinct|.
///
/// Convention for empty inputs (M3 locked spec, revised):
///   Jaccard(∅, _) = Jaccard(_, ∅) = 0  — empty lineage carries no signal.
///
/// Rationale: an empty/missing lineage is informationally inert — absent
/// positive evidence of shared provenance, we don't penalize. This is also
/// what preserves the "single claim has no rest" intuition: with one claim,
/// the rest union is empty, overlap is 0, discount is neutral, ω = 1.
fn leave_one_out_jaccard(
    claim_set: &[String],
    freq: &HashMap<&str, usize>,
    total_distinct: usize,
) -> f64 {
    if claim_set.is_empty() || total_distinct == 0 {
        return 0.0;
    }
    let rest_union_is_empty = {
        // "rest" is empty iff every element in freq has count == 1 AND is
        // in claim_set — i.e., claim_set is the only contributor.
        let claim_set_lookup: HashSet<&str> = claim_set.iter().map(|s| s.as_str()).collect();
        freq.iter().all(|(e, c)| *c == 1 && claim_set_lookup.contains(e))
    };
    if rest_union_is_empty {
        return 0.0;
    }
    // |X ∩ rest_union| = # of claim_set elements e with freq[e] > 1
    let intersection = claim_set
        .iter()
        .filter(|e| freq.get(e.as_str()).copied().unwrap_or(0) > 1)
        .count();
    // |X ∪ rest_union| = |total_distinct| — every element with freq ≥ 1
    // lives somewhere, and claim_set ⊆ freq-keys.
    intersection as f64 / total_distinct as f64
}

/// P_k — fraction of other claims sharing the same extractor. Symmetric
/// under permutation of `claims`, so order-invariant.
fn pipeline_overlap_ratio(claim: &ClaimRow, claims: &[&ClaimRow], own_index: usize) -> f64 {
    if claims.len() <= 1 {
        return 0.0;
    }
    let matching = claims
        .iter()
        .enumerate()
        .filter(|(i, c)| *i != own_index && c.extractor == claim.extractor)
        .count();
    matching as f64 / (claims.len() - 1) as f64
}

/// S_k — 1.0 if this claim is self-generated AND at least one other claim
/// in the live set is also self-generated. 0.0 otherwise. Binary v1;
/// background tier will refine to consider ancestry depth.
fn self_gen_overlap_binary(claim: &ClaimRow, claims: &[&ClaimRow], own_index: usize) -> f64 {
    if !claim.self_generated {
        return 0.0;
    }
    let has_other = claims
        .iter()
        .enumerate()
        .any(|(i, c)| i != own_index && c.self_generated);
    if has_other {
        1.0
    } else {
        0.0
    }
}

/// χ — modality consilience: distinct modalities in the live set, normalized
/// by MAX_MODALITIES. Higher when independent modalities corroborate. [0, 1].
fn compute_modality_consilience(claims: &[&ClaimRow]) -> f64 {
    if claims.is_empty() {
        return 0.0;
    }
    let distinct: HashSet<&String> = claims.iter().map(|c| &c.modality_signal).collect();
    (distinct.len() as f64 / MAX_MODALITIES).min(1.0)
}

/// ψ_l — local self-generation ratio: fraction of supporting claims whose
/// `self_generated` flag is set.
fn compute_self_gen_local(claims: &[&ClaimRow]) -> f64 {
    if claims.is_empty() {
        return 0.0;
    }
    let self_gen_count = claims.iter().filter(|c| c.self_generated).count();
    self_gen_count as f64 / claims.len() as f64
}

/// Row-to-struct decoder for mobility_state SELECTs.
fn row_to_mobility_state(row: &rusqlite::Row) -> rusqlite::Result<MobilityState> {
    let tier_write_json: String = row.get(16)?;
    let tier_read_json: String = row.get(17)?;
    let tier_bg_json: String = row.get(18)?;
    Ok(MobilityState {
        proposition_id: row.get(0)?,
        regime: row.get(1)?,
        snapshot_ts: row.get(2)?,
        support_mass: row.get(3)?,
        attack_mass: row.get(4)?,
        source_diversity: row.get(5)?,
        effective_independence: row.get(6)?,
        temporal_coherence: row.get(7)?,
        transportability: row.get(8)?,
        mutability: row.get(9)?,
        load_bearingness: row.get(10)?,
        modality_consilience: row.get(11)?,
        self_gen_local: row.get(12)?,
        self_gen_ancestral: row.get(13)?,
        contamination_risk: row.get(14)?,
        novelty_isolation: row.get(15)?,
        tier_write_components: serde_json::from_str(&tier_write_json).unwrap_or_default(),
        tier_read_components: serde_json::from_str(&tier_read_json).unwrap_or_default(),
        tier_bg_components: serde_json::from_str(&tier_bg_json).unwrap_or_default(),
        formula_version: row.get(19)?,
        content_hash: row.get(20)?,
        live_claim_count: row.get(21)?,
        state_status: row.get(22)?,
        computed_at: row.get(23)?,
    })
}

// ──────────────────────────────────────────────────────────────────
// Unit tests for the pure functions. Integration tests (round-trip
// through YantrikDB) live in engine/tests.rs.
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_claim(
        claim_id: &str,
        extractor: &str,
        lineage: &[&str],
        self_gen: bool,
        modality: &str,
        weight: f64,
    ) -> ClaimRow {
        let mut normalized: Vec<String> = lineage.iter().map(|s| s.to_string()).collect();
        normalized.sort();
        normalized.dedup();
        ClaimRow {
            claim_id: claim_id.to_string(),
            polarity: 1,
            weight,
            extractor: extractor.to_string(),
            source_lineage: normalized,
            self_generated: self_gen,
            modality_signal: modality.to_string(),
        }
    }

    #[test]
    fn accumulate_single_claim_returns_weight() {
        let c = mk_claim("c1", "ext_a", &["src_1"], false, "text", 1.0);
        let refs = vec![&c];
        let total = accumulate_mass(&refs);
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn accumulate_independent_claims_scales_linearly() {
        let c1 = mk_claim("c1", "ext_a", &["src_a1", "src_a2"], false, "text", 1.0);
        let c2 = mk_claim("c2", "ext_b", &["src_b1"], false, "image", 1.0);
        let c3 = mk_claim("c3", "ext_c", &["src_c1"], false, "numeric", 1.0);
        let total = accumulate_mass(&[&c1, &c2, &c3]);
        // Disjoint lineages, different extractors, no self-gen → ω = 1 each.
        assert!(
            (total - 3.0).abs() < 1e-9,
            "independent claims should sum to raw weights, got {}",
            total
        );
    }

    #[test]
    fn accumulate_duplicate_lineage_discounts() {
        let c1 = mk_claim("c1", "ext_a", &["src_shared"], false, "text", 1.0);
        let c2 = mk_claim("c2", "ext_a", &["src_shared"], false, "text", 1.0);
        let c3 = mk_claim("c3", "ext_a", &["src_shared"], false, "text", 1.0);
        let total = accumulate_mass(&[&c1, &c2, &c3]);
        // D_k = 1, P_k = 1, S_k = 0 → discount = 1.8; ω = 1/1.8; total ≈ 1.67
        assert!(total < 2.0, "duplicate lineage should discount, got {}", total);
        assert!(total > 1.5, "discount shouldn't be excessive, got {}", total);
    }

    #[test]
    fn accumulate_self_generated_is_strongly_discounted() {
        let c1 = mk_claim("c1", "self_reasoning", &["self_1"], true, "text", 1.0);
        let c2 = mk_claim("c2", "self_reasoning", &["self_1"], true, "text", 1.0);
        let total = accumulate_mass(&[&c1, &c2]);
        // discount = 1 + 0.5·1 + 0.3·1 + 0.7·1 = 2.5; ω = 0.4; total ≈ 0.8
        assert!(total < 1.0, "self-generated duplicates should collapse; got {}", total);
    }

    #[test]
    fn accumulate_is_order_invariant() {
        let c1 = mk_claim("c1", "ext_a", &["src_a"], false, "text", 1.0);
        let c2 = mk_claim("c2", "ext_b", &["src_b"], false, "image", 1.0);
        let c3 = mk_claim("c3", "ext_c", &["src_c"], false, "numeric", 1.0);
        let forward = accumulate_mass(&[&c1, &c2, &c3]);
        let reverse = accumulate_mass(&[&c3, &c2, &c1]);
        let shuffled = accumulate_mass(&[&c2, &c3, &c1]);
        assert!((forward - reverse).abs() < 1e-9);
        assert!((forward - shuffled).abs() < 1e-9);
    }

    #[test]
    fn modality_consilience_tracks_distinct_modalities() {
        let c1 = mk_claim("c1", "ext_a", &["src_1"], false, "text", 1.0);
        let c2 = mk_claim("c2", "ext_a", &["src_2"], false, "text", 1.0);
        let mono = compute_modality_consilience(&[&c1, &c2]);
        let c3 = mk_claim("c3", "ext_a", &["src_3"], false, "image", 1.0);
        let c4 = mk_claim("c4", "ext_a", &["src_4"], false, "numeric", 1.0);
        let multi = compute_modality_consilience(&[&c1, &c3, &c4]);
        assert!(multi > mono);
    }

    #[test]
    fn self_gen_local_is_correct_ratio() {
        let c1 = mk_claim("c1", "ext_a", &["src_1"], true, "text", 1.0);
        let c2 = mk_claim("c2", "ext_a", &["src_2"], false, "text", 1.0);
        let c3 = mk_claim("c3", "ext_a", &["src_3"], true, "text", 1.0);
        let r = compute_self_gen_local(&[&c1, &c2, &c3]);
        assert!((r - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn content_hash_is_order_invariant() {
        let c1 = mk_claim("c1", "ext_a", &["src_a"], false, "text", 1.0);
        let c2 = mk_claim("c2", "ext_b", &["src_b"], false, "image", 1.0);
        let c3 = mk_claim("c3", "ext_c", &["src_c"], false, "numeric", 1.0);
        let h_forward = content_hash(&[c1.clone(), c2.clone(), c3.clone()]);
        let h_reverse = content_hash(&[c3.clone(), c2.clone(), c1.clone()]);
        assert_eq!(h_forward, h_reverse);
    }

    #[test]
    fn content_hash_discriminates_on_lineage_change() {
        let c1 = mk_claim("c1", "ext_a", &["src_a"], false, "text", 1.0);
        let c2 = mk_claim("c1", "ext_a", &["src_b"], false, "text", 1.0);
        assert_ne!(content_hash(&[c1]), content_hash(&[c2]));
    }

    #[test]
    fn jaccard_empty_inputs_are_zero() {
        // Empty claim set or empty rest union → 0 (no signal, no discount).
        // This is the revised M3 convention; it means a single claim with
        // empty lineage gets ω = 1, which matches the "no one to be
        // dependent on" intuition.
        let freq: HashMap<&str, usize> = HashMap::new();
        assert_eq!(leave_one_out_jaccard(&[], &freq, 0), 0.0);
        let claim = vec!["a".to_string()];
        assert_eq!(leave_one_out_jaccard(&claim, &freq, 0), 0.0);
    }

    #[test]
    fn jaccard_disjoint_is_zero() {
        // claim = {a}, others contribute {b}. freq = {a:1, b:1}, total=2.
        // rest_union for claim = {b} (only b has freq outside claim).
        // intersection = {} (a has freq 1, not > 1).
        // union = 2. Jaccard = 0.
        let mut freq: HashMap<&str, usize> = HashMap::new();
        freq.insert("a", 1);
        freq.insert("b", 1);
        let j = leave_one_out_jaccard(&["a".to_string()], &freq, 2);
        assert_eq!(j, 0.0);
    }

    #[test]
    fn jaccard_fully_shared_is_one() {
        // claim = {a}, another claim also has {a}. freq = {a:2}, total=1.
        // intersection = 1 (a is in both); union = 1. Jaccard = 1.
        let mut freq: HashMap<&str, usize> = HashMap::new();
        freq.insert("a", 2);
        let j = leave_one_out_jaccard(&["a".to_string()], &freq, 1);
        assert_eq!(j, 1.0);
    }
}
