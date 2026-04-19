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
//! Milestone 2 of Phase 1: this commit ships the write-tier accumulation
//! operator ⊕ with dependence discount — the anti-echo-chamber math. Raw
//! function + round-trip storage. No ingestion hook yet (next milestone).

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// A single row of the `mobility_state` table. All 13 components are Option
/// because they are populated at different tiers — a freshly-inserted row
/// from the write path only fills in σ, α, χ, ψ_l; the rest are NULL until
/// the background job computes them.
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
    /// JSON array of component names populated at write tier.
    pub tier_write_components: Vec<String>,
    pub tier_read_components: Vec<String>,
    pub tier_bg_components: Vec<String>,
}

/// Dependence-discount penalty weights for the accumulation operator ⊕.
/// These are fixed in v1; RFC 008 Phase 2 will make them learned.
///
/// Formula: ω_k = 1 / (1 + λ·D_k + μ·P_k + τ·S_k)
///   D_k — source overlap with prior supports (Jaccard over source_lineage)
///   P_k — extractor/pipeline overlap (binary for now)
///   S_k — self-generation overlap (1.0 if this + any prior are self-gen)
///
/// Rationale for defaults:
///   λ=0.5 — source overlap is the primary anti-echo-chamber lever
///   μ=0.3 — pipeline overlap is secondary (same extractor ≠ duplication)
///   τ=0.7 — self-generation is the strongest discount (prevents loops)
pub const DEPENDENCE_WEIGHT_SOURCE: f64 = 0.5;
pub const DEPENDENCE_WEIGHT_PIPELINE: f64 = 0.3;
pub const DEPENDENCE_WEIGHT_SELF_GEN: f64 = 0.7;

/// A minimal projection of a claim row containing only the fields needed
/// to compute write-tier mobility contributions. Fetched with a narrow
/// SELECT to keep the hot path fast.
#[derive(Debug, Clone)]
struct ClaimRow {
    polarity: i32,
    weight: f64,
    extractor: String,
    source_lineage: Vec<String>, // parsed from JSON
    self_generated: bool,
    modality_signal: String,
}

impl crate::engine::YantrikDB {
    /// Compute the write-tier mobility state for a proposition in a given
    /// regime. Reads all live (non-tombstoned) claim rows for the
    /// proposition, aggregates support and attack masses with dependence
    /// discount, and returns a fresh `MobilityState` with σ, α, χ, ψ_l
    /// populated.
    ///
    /// This is the implementation of ⊕ (accumulation). Given supporting
    /// contributions f_1..f_n and a dependence structure D, it computes:
    ///
    ///   s_eff = Σ_k ω_k · w_k
    ///   where ω_k = 1 / (1 + λ·D_k + μ·P_k + τ·S_k)
    ///
    /// D_k is estimated as Jaccard similarity of claim k's source_lineage
    /// with the union of prior claims' lineages. P_k is the fraction of
    /// prior claims sharing the same extractor. S_k is 1 if this claim
    /// and any prior are self-generated, else 0.
    ///
    /// The order of claims affects the ω values (each claim's overlap is
    /// measured against *prior* claims in insertion order), but the final
    /// s_eff converges regardless of order because the discount is applied
    /// symmetrically via union of lineages. Property-tested.
    pub fn compute_write_tier_mobility(
        &self,
        proposition_id: &str,
        regime: &str,
    ) -> Result<MobilityState> {
        let conn = self.conn.lock();
        let claims = fetch_claims_for_mobility(&conn, proposition_id, regime)?;
        drop(conn);

        let now = super::now();
        let mut state = MobilityState {
            proposition_id: proposition_id.to_string(),
            regime: regime.to_string(),
            snapshot_ts: now,
            tier_write_components: vec![
                "support_mass".into(),
                "attack_mass".into(),
                "modality_consilience".into(),
                "self_gen_local".into(),
            ],
            ..Default::default()
        };

        let support_claims: Vec<&ClaimRow> = claims.iter().filter(|c| c.polarity == 1).collect();
        let attack_claims: Vec<&ClaimRow> = claims.iter().filter(|c| c.polarity == -1).collect();

        state.support_mass = Some(accumulate_mass(&support_claims));
        state.attack_mass = Some(accumulate_mass(&attack_claims));
        state.modality_consilience = Some(compute_modality_consilience(&support_claims));
        state.self_gen_local = Some(compute_self_gen_local(&support_claims));

        Ok(state)
    }

    /// Read mobility state for a proposition+regime. Returns the most
    /// recent snapshot if multiple exist. None if none recorded.
    pub fn get_mobility_state(
        &self,
        proposition_id: &str,
        regime: &str,
    ) -> Result<Option<MobilityState>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT proposition_id, regime, snapshot_ts, \
             support_mass, attack_mass, source_diversity, effective_independence, \
             temporal_coherence, transportability, mutability, load_bearingness, \
             modality_consilience, self_gen_local, self_gen_ancestral, \
             contamination_risk, novelty_isolation, \
             tier_write_components, tier_read_components, tier_bg_components \
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

    /// Upsert mobility state, replacing any row with the same
    /// (proposition_id, regime, snapshot_ts) key. Snapshot_ts makes the
    /// primary key unique-by-snapshot, so background jobs can insert
    /// derived facts tagged with their snapshot without overwriting
    /// write-tier rows.
    pub fn upsert_mobility_state(&self, state: &MobilityState) -> Result<()> {
        let conn = self.conn.lock();
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
             tier_write_components, tier_read_components, tier_bg_components) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19) \
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
             tier_bg_components = excluded.tier_bg_components",
            params![
                state.proposition_id, state.regime, state.snapshot_ts,
                state.support_mass, state.attack_mass, state.source_diversity,
                state.effective_independence, state.temporal_coherence,
                state.transportability, state.mutability, state.load_bearingness,
                state.modality_consilience, state.self_gen_local,
                state.self_gen_ancestral, state.contamination_risk,
                state.novelty_isolation,
                tier_write, tier_read, tier_bg,
            ],
        )?;
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────
// Free functions — the actual math, testable without the engine.
// ──────────────────────────────────────────────────────────────────

/// Fetch a narrow projection of claim rows for a proposition+regime, used
/// as input to ⊕. Only the fields needed for mobility computation.
fn fetch_claims_for_mobility(
    conn: &Connection,
    proposition_id: &str,
    regime: &str,
) -> Result<Vec<ClaimRow>> {
    let mut stmt = conn.prepare(
        "SELECT polarity, weight, extractor, source_lineage, self_generated, modality_signal \
         FROM claims \
         WHERE proposition_id = ?1 AND regime_tag = ?2 AND tombstoned = 0 \
         ORDER BY created_at ASC",
    )?;
    let rows = stmt
        .query_map(params![proposition_id, regime], |row| {
            let lineage_json: String = row.get(3)?;
            let lineage: Vec<String> = serde_json::from_str(&lineage_json).unwrap_or_default();
            let self_gen_int: i64 = row.get(4)?;
            Ok(ClaimRow {
                polarity: row.get(0)?,
                weight: row.get(1)?,
                extractor: row.get(2)?,
                source_lineage: lineage,
                self_generated: self_gen_int != 0,
                modality_signal: row.get(5)?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Accumulate mass with dependence discount. The core of ⊕.
///
/// Applies ω_k = 1 / (1 + λ·D_k + μ·P_k + τ·S_k) to each claim's weight
/// and returns the sum. Order-dependent in the sense that each claim's
/// overlap is measured against *all other* claims (symmetric), not prior
/// claims only — this makes the result order-independent (tested).
pub fn accumulate_mass(claims: &[&ClaimRow]) -> f64 {
    if claims.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for (k, claim) in claims.iter().enumerate() {
        let others: Vec<&&ClaimRow> = claims
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != k)
            .map(|(_, c)| c)
            .collect();
        let d_k = compute_source_overlap(claim, &others);
        let p_k = compute_pipeline_overlap(claim, &others);
        let s_k = compute_self_gen_overlap(claim, &others);
        let discount = 1.0
            + DEPENDENCE_WEIGHT_SOURCE * d_k
            + DEPENDENCE_WEIGHT_PIPELINE * p_k
            + DEPENDENCE_WEIGHT_SELF_GEN * s_k;
        let omega = 1.0 / discount;
        total += omega * claim.weight;
    }
    total
}

/// D_k: Jaccard similarity of this claim's source_lineage with the union
/// of other claims' source_lineages. Range [0, 1]. Zero when lineages are
/// disjoint; one when identical.
fn compute_source_overlap(claim: &ClaimRow, others: &[&&ClaimRow]) -> f64 {
    if others.is_empty() {
        return 0.0;
    }
    let mut union: std::collections::HashSet<&String> = std::collections::HashSet::new();
    for o in others {
        for src in &o.source_lineage {
            union.insert(src);
        }
    }
    if union.is_empty() {
        return 0.0;
    }
    let claim_set: std::collections::HashSet<&String> = claim.source_lineage.iter().collect();
    if claim_set.is_empty() {
        return 0.0;
    }
    let intersection = claim_set.intersection(&union).count();
    let total_union = union.union(&claim_set).count();
    if total_union == 0 {
        0.0
    } else {
        intersection as f64 / total_union as f64
    }
}

/// P_k: fraction of other claims sharing the same extractor.
fn compute_pipeline_overlap(claim: &ClaimRow, others: &[&&ClaimRow]) -> f64 {
    if others.is_empty() {
        return 0.0;
    }
    let matching = others.iter().filter(|o| o.extractor == claim.extractor).count();
    matching as f64 / others.len() as f64
}

/// S_k: 1.0 if this claim is self-generated AND at least one other claim
/// is self-generated. 0.0 otherwise. This is a crude binary for v1; the
/// background tier will refine it to consider ancestry depth.
fn compute_self_gen_overlap(claim: &ClaimRow, others: &[&&ClaimRow]) -> f64 {
    if !claim.self_generated {
        return 0.0;
    }
    if others.iter().any(|o| o.self_generated) {
        1.0
    } else {
        0.0
    }
}

/// χ — modality consilience: number of distinct modalities among supporting
/// claims, normalized by the max modality count. Higher when independent
/// modalities corroborate the same proposition. Range [0, 1].
fn compute_modality_consilience(claims: &[&ClaimRow]) -> f64 {
    if claims.is_empty() {
        return 0.0;
    }
    let distinct: std::collections::HashSet<&String> = claims.iter().map(|c| &c.modality_signal).collect();
    // Max plausible modality count for normalization — text, image, numeric,
    // audio, code, telemetry = 6. We normalize by this cap so χ is bounded [0,1].
    const MAX_MODALITIES: f64 = 6.0;
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
    })
}

// ──────────────────────────────────────────────────────────────────
// Unit tests for the pure functions. Integration tests (round-trip
// through YantrikDB) live in engine/tests.rs.
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_claim(extractor: &str, lineage: &[&str], self_gen: bool, modality: &str, weight: f64) -> ClaimRow {
        ClaimRow {
            polarity: 1,
            weight,
            extractor: extractor.to_string(),
            source_lineage: lineage.iter().map(|s| s.to_string()).collect(),
            self_generated: self_gen,
            modality_signal: modality.to_string(),
        }
    }

    #[test]
    fn accumulate_single_claim_returns_weight() {
        let c = mk_claim("ext_a", &["src_1"], false, "text", 1.0);
        let refs = vec![&c];
        let total = accumulate_mass(&refs);
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn accumulate_independent_claims_scales_linearly() {
        // Three claims with completely disjoint source_lineage, different
        // extractors, none self-gen. Dependence should be ~0; total ~= 3.0.
        let c1 = mk_claim("ext_a", &["src_a1", "src_a2"], false, "text", 1.0);
        let c2 = mk_claim("ext_b", &["src_b1"], false, "image", 1.0);
        let c3 = mk_claim("ext_c", &["src_c1"], false, "numeric", 1.0);
        let all = vec![&c1, &c2, &c3];
        let total = accumulate_mass(&all);
        // Pipeline overlap is 0 (all different extractors). Source overlap is 0
        // (disjoint lineages). Self-gen overlap is 0. So ω = 1 for each.
        assert!((total - 3.0).abs() < 1e-9, "independent claims should sum to raw weights, got {}", total);
    }

    #[test]
    fn accumulate_duplicate_lineage_discounts() {
        // Three claims all from the same source lineage. Should be heavily
        // discounted — NOT 3.0.
        let c1 = mk_claim("ext_a", &["src_shared"], false, "text", 1.0);
        let c2 = mk_claim("ext_a", &["src_shared"], false, "text", 1.0);
        let c3 = mk_claim("ext_a", &["src_shared"], false, "text", 1.0);
        let all = vec![&c1, &c2, &c3];
        let total = accumulate_mass(&all);
        // For each claim: D_k = 1.0 (identical lineage to others' union),
        //                 P_k = 1.0 (all share extractor),
        //                 S_k = 0.0 (none self-gen).
        // discount = 1 + 0.5·1 + 0.3·1 + 0 = 1.8; ω = 1/1.8 ≈ 0.555
        // total ≈ 3 · 0.555 = 1.666
        assert!(total < 2.0, "duplicate-lineage claims should be heavily discounted, got {}", total);
        assert!(total > 1.5, "discount shouldn't be excessive, got {}", total);
    }

    #[test]
    fn accumulate_self_generated_is_strongly_discounted() {
        // Two self-generated claims from shared lineage — both discount
        // terms active. Total should be much less than raw 2.0.
        let c1 = mk_claim("self_reasoning", &["self_1"], true, "text", 1.0);
        let c2 = mk_claim("self_reasoning", &["self_1"], true, "text", 1.0);
        let all = vec![&c1, &c2];
        let total = accumulate_mass(&all);
        // discount = 1 + 0.5 + 0.3 + 0.7 = 2.5; ω = 0.4; total ≈ 0.8
        assert!(total < 1.0, "self-generated duplicates should collapse; got {}", total);
    }

    #[test]
    fn accumulate_is_order_independent() {
        let c1 = mk_claim("ext_a", &["src_a"], false, "text", 1.0);
        let c2 = mk_claim("ext_b", &["src_b"], false, "image", 1.0);
        let c3 = mk_claim("ext_c", &["src_c"], false, "numeric", 1.0);
        let forward = accumulate_mass(&[&c1, &c2, &c3]);
        let reverse = accumulate_mass(&[&c3, &c2, &c1]);
        let shuffled = accumulate_mass(&[&c2, &c3, &c1]);
        assert!((forward - reverse).abs() < 1e-9);
        assert!((forward - shuffled).abs() < 1e-9);
    }

    #[test]
    fn modality_consilience_tracks_distinct_modalities() {
        let c1 = mk_claim("ext_a", &["src_1"], false, "text", 1.0);
        let c2 = mk_claim("ext_a", &["src_2"], false, "text", 1.0);
        let mono = compute_modality_consilience(&[&c1, &c2]);
        let c3 = mk_claim("ext_a", &["src_3"], false, "image", 1.0);
        let c4 = mk_claim("ext_a", &["src_4"], false, "numeric", 1.0);
        let multi = compute_modality_consilience(&[&c1, &c3, &c4]);
        assert!(multi > mono, "multi-modal should score higher (multi={}, mono={})", multi, mono);
    }

    #[test]
    fn self_gen_local_is_correct_ratio() {
        let c1 = mk_claim("ext_a", &["src_1"], true, "text", 1.0);
        let c2 = mk_claim("ext_a", &["src_2"], false, "text", 1.0);
        let c3 = mk_claim("ext_a", &["src_3"], true, "text", 1.0);
        let r = compute_self_gen_local(&[&c1, &c2, &c3]);
        assert!((r - 2.0 / 3.0).abs() < 1e-9);
    }
}
