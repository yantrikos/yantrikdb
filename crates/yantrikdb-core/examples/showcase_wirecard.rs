//! Wirecard — evidence-first test of the RFC 008 substrate on real data.
//!
//! The €1.9B that both existed and didn't. For nearly a decade, Wirecard AG
//! reported €1.9 billion held in escrow at two Philippine banks. The auditor
//! (EY) signed off. In 2019 the Financial Times raised doubts. In June 2020
//! both Philippine banks formally denied ever holding the accounts. Wirecard
//! collapsed into insolvency 12 days later.
//!
//! This example seeds the public record across five sources, computes the
//! RFC 008 substrate state (mobility + contest + moves), and prints what
//! the substrate sees that naive polarity-counting doesn't.
//!
//! Run with: `cargo run --example showcase_wirecard -- --nocapture`
//! Or the standard: `cargo run --example showcase_wirecard`

use yantrikdb::engine::moves::{ClaimRef, RecordMoveEventInput, observability};
use yantrikdb::YantrikDB;

fn main() {
    let db = YantrikDB::new(":memory:", 8).expect("open db");

    banner("WIRECARD — the €1.9B that existed and didn't");
    println!("  Sources: Wirecard filings, EY audit, FT investigation, BPI, BDO, BSP");
    println!("  Period:  2014 (first reported) → 2020-06-25 (insolvency)\n");

    // ──────────────────────────────────────────────────────────────
    // Phase 1: Seed claims with source_lineage that encodes real
    // epistemic dependency. This is where naive vs substrate diverges.
    // ──────────────────────────────────────────────────────────────

    sub("Phase 1 — seeding the public record");

    let prop_key = ("Philippine_trustee_accounts", "balance_equals", "EUR_1.9_billion");

    // (1) Wirecard's own filings — self-reported. source_lineage = ["wirecard"].
    ingest_claim_with_lineage(
        &db, prop_key, 1, "asserted", "wirecard.filing",
        Some(d(2018, 12, 31)), Some(d(2020, 6, 18)),
        &["wirecard"],
    );

    // (2) EY audit — relied on Wirecard's own trustee confirmation letters
    //     (which later turned out to be forged). Shared lineage with
    //     Wirecard. source_lineage = ["wirecard", "ey"].
    //     This is the epistemic reality: EY's "confirmation" did NOT come
    //     from an independent channel — it came from Wirecard-provided
    //     documents that EY failed to verify directly with the banks.
    ingest_claim_with_lineage(
        &db, prop_key, 1, "asserted", "ey.audit",
        Some(d(2014, 1, 1)), Some(d(2020, 6, 18)),
        &["wirecard", "ey"],
    );

    // (3) FT — independent investigative reporting. source_lineage = ["ft"].
    //     Polarity -1 (denies the balance exists in the form claimed).
    ingest_claim_with_lineage(
        &db, prop_key, -1, "reported", "ft.investigation",
        Some(d(2019, 1, 30)), None,
        &["ft"],
    );

    // (4) BPI — Philippine bank's own formal denial. Independent source.
    ingest_claim_with_lineage(
        &db, prop_key, -1, "asserted", "bpi.statement",
        Some(d(2020, 6, 18)), None,
        &["bpi"],
    );

    // (5) BDO — the other Philippine bank. Independent.
    ingest_claim_with_lineage(
        &db, prop_key, -1, "asserted", "bdo.statement",
        Some(d(2020, 6, 19)), None,
        &["bdo"],
    );

    // (6) Bangko Sentral ng Pilipinas — central bank. Relies on the two
    //     commercial banks' statements. source_lineage = ["bsp", "bpi", "bdo"].
    //     ⊕ should discount this substantially because it shares lineage
    //     with BPI + BDO.
    ingest_claim_with_lineage(
        &db, prop_key, -1, "asserted", "bsp.circular",
        Some(d(2020, 6, 21)), None,
        &["bsp", "bpi", "bdo"],
    );

    println!("  6 claims seeded across 5 source lineages.");
    println!("  Supports: wirecard.filing, ey.audit   (2 claims)");
    println!("  Attacks:  ft, bpi, bdo, bsp           (4 claims)\n");

    // ──────────────────────────────────────────────────────────────
    // Phase 2: Trigger write-tier + contest + background recompute.
    // The ingest_claim hook already fires write-tier mobility + contest;
    // we explicitly trigger background for τ/λ/ψ_a.
    // ──────────────────────────────────────────────────────────────

    let prop_id = find_proposition(&db, prop_key);
    db.compute_background_mobility(&prop_id, "default").unwrap();

    // ──────────────────────────────────────────────────────────────
    // Phase 3: Naive baseline vs substrate.
    // ──────────────────────────────────────────────────────────────

    sub("Phase 2 — naive polarity counting (what a vector DB would see)");
    let mobility = db.get_mobility_state(&prop_id, "default").unwrap().unwrap();
    let contest = db.get_contest_state(&prop_id, "default").unwrap().unwrap();

    let naive_pro = 2.0_f64;   // raw count of positive-polarity claims
    let naive_con = 4.0_f64;   // raw count of negative-polarity claims
    println!("  naive σ (support count) = {:.3}", naive_pro);
    println!("  naive α (attack count)  = {:.3}", naive_con);
    println!("  naive net = {:.3}  (attack-dominant, straightforward tally)",
             naive_pro - naive_con);

    sub("Phase 3 — substrate leave-one-out aggregation (RFC 008 ⊕)");
    println!("  σ (dependence-discounted support) = {:.3}",
             mobility.support_mass.unwrap_or(0.0));
    println!("  α (dependence-discounted attack)  = {:.3}",
             mobility.attack_mass.unwrap_or(0.0));
    println!("  support_distinct_source_count = {}", contest.support_distinct_source_count);
    println!("  attack_distinct_source_count  = {}", contest.attack_distinct_source_count);
    println!("  support_effective_independence = {:.3}", contest.support_effective_independence);
    println!("  attack_effective_independence  = {:.3}", contest.attack_effective_independence);

    sub("Phase 4 — what the substrate discounts");
    println!("  ey.audit shares source_lineage ['wirecard','ey'] with wirecard.filing.");
    println!("  ⊕'s leave-one-out Jaccard penalizes that overlap — EY's claim");
    println!("  carries ω_k < 1 instead of the naive ω_k = 1. Correctly:");
    println!("  EY was NOT an independent confirmation of Wirecard's cash position;");
    println!("  EY audited the documents Wirecard provided, which were forged.");
    println!();
    println!("  bsp.circular shares lineage with bpi + bdo; it gets discounted too.");
    println!("  The central bank's statement is correct but not independent evidence.");

    // ──────────────────────────────────────────────────────────────
    // Phase 4: Contest diagnostics.
    // ──────────────────────────────────────────────────────────────

    sub("Phase 5 — contest Γ(c) diagnostics");
    println!("  same_source_opposite_polarity_count            = {}",
             contest.same_source_opposite_polarity_count);
    println!("  same_artifact_extractor_polarity_conflict_count = {}",
             contest.same_artifact_extractor_polarity_conflict_count);
    println!("  temporal_overlap_conflict_count                 = {}",
             contest.temporal_overlap_conflict_count);
    println!("  temporal_separable_opposition_count             = {}",
             contest.temporal_separable_opposition_count);
    println!("  referent_schema_heterogeneity_count             = {}",
             contest.referent_schema_heterogeneity_count);

    let flags = contest.heuristic_flags;
    println!("\n  heuristic_flags bitmap = 0x{:X}", flags);
    for (name, bit) in [
        ("DUPLICATION_RISK", 0b00001),
        ("SAME_SOURCE_CONFLICT", 0b00010),
        ("REFERENT_HETEROGENEITY_PRESENT", 0b00100),
        ("SAME_ARTIFACT_EXTRACTOR_CONFLICT", 0b01000),
        ("PRESENT_TENSE_CONFLICT", 0b10000),
    ] {
        let set = if flags & bit != 0 { "SET" } else { "-" };
        println!("    [{}] {}", set, name);
    }

    // ──────────────────────────────────────────────────────────────
    // Phase 5: Background-tier mobility (τ, λ, ψ_a).
    // ──────────────────────────────────────────────────────────────

    sub("Phase 6 — background tier (τ, λ, ψ_a)");
    println!("  τ (temporal_coherence)    = {:?}", mobility.temporal_coherence);
    println!("     polarity flipped from +1 (2014-2019) to -1 (2020); coherence < 1");
    println!("  λ (load_bearingness)      = {:?}", mobility.load_bearingness);
    println!("     no downstream moves yet consume these claims");
    println!("  ψ_a (self_gen_ancestral)  = {:?}", mobility.self_gen_ancestral);
    println!("     no cognitive moves yet in the chain");

    // ──────────────────────────────────────────────────────────────
    // Phase 6: Record cognitive moves for the reasoning chain.
    // ──────────────────────────────────────────────────────────────

    sub("Phase 7 — move events: FT triage → banks formal deny → insolvency");

    let claim_ids = fetch_claim_ids_for_prop(&db, &prop_id);
    // 0 wirecard, 1 ey, 2 ft, 3 bpi, 4 bdo, 5 bsp — insertion order preserved
    let (c_wirecard, c_ey, c_ft, c_bpi, c_bdo, c_bsp) =
        (claim_ids[0].clone(), claim_ids[1].clone(), claim_ids[2].clone(),
         claim_ids[3].clone(), claim_ids[4].clone(), claim_ids[5].clone());

    // Move 1: FT performs contradiction_triage against wirecard+ey claims.
    let m_ft = db.record_move_event(RecordMoveEventInput {
        move_type: "contradiction_triage".into(),
        operator_version: "v1".into(),
        observability: observability::OBSERVED.into(),
        inputs: vec![
            ClaimRef { claim_id: c_wirecard.clone(), role: "subject".into(), ordinal: 0 },
            ClaimRef { claim_id: c_ey.clone(),      role: "subject".into(), ordinal: 1 },
        ],
        outputs: vec![
            ClaimRef { claim_id: c_ft.clone(), role: "attack_claim".into(), ordinal: 0 },
        ],
        ..Default::default()
    }).unwrap();
    println!("  [{}] FT contradiction_triage", m_ft);

    // Move 2: BPI source_audit (banks audit their own records, find nothing).
    let m_bpi = db.record_move_event(RecordMoveEventInput {
        move_type: "source_audit".into(),
        operator_version: "v1".into(),
        observability: observability::OBSERVED.into(),
        inputs: vec![
            ClaimRef { claim_id: c_wirecard.clone(), role: "subject".into(), ordinal: 0 },
        ],
        outputs: vec![
            ClaimRef { claim_id: c_bpi.clone(), role: "audit_finding".into(), ordinal: 0 },
        ],
        ..Default::default()
    }).unwrap();
    println!("  [{}] BPI source_audit → no such accounts", m_bpi);

    // Move 3: BDO source_audit (same deal).
    db.record_move_event(RecordMoveEventInput {
        move_type: "source_audit".into(),
        operator_version: "v1".into(),
        observability: observability::OBSERVED.into(),
        inputs: vec![
            ClaimRef { claim_id: c_wirecard.clone(), role: "subject".into(), ordinal: 0 },
        ],
        outputs: vec![
            ClaimRef { claim_id: c_bdo.clone(), role: "audit_finding".into(), ordinal: 0 },
        ],
        ..Default::default()
    }).unwrap();

    // Move 4: BSP contradiction_triage — central bank's determination
    // consumes both bank findings.
    db.record_move_event(RecordMoveEventInput {
        move_type: "contradiction_triage".into(),
        operator_version: "v1".into(),
        observability: observability::OBSERVED.into(),
        inputs: vec![
            ClaimRef { claim_id: c_bpi.clone(), role: "evidence".into(), ordinal: 0 },
            ClaimRef { claim_id: c_bdo.clone(), role: "evidence".into(), ordinal: 1 },
        ],
        outputs: vec![
            ClaimRef { claim_id: c_bsp.clone(), role: "official_determination".into(), ordinal: 0 },
        ],
        ..Default::default()
    }).unwrap();

    // Recompute background so λ and ψ_a reflect the new moves.
    db.compute_background_mobility(&prop_id, "default").unwrap();
    let mobility_post = db.get_mobility_state(&prop_id, "default").unwrap().unwrap();
    println!("\n  After recording moves, background mobility updates:");
    println!("    λ (load_bearingness) = {:?} — moves now cite these claims",
             mobility_post.load_bearingness);
    println!("    ψ_a (self_gen_ancestral) = {:?} — ancestry traced through the DAG",
             mobility_post.self_gen_ancestral);

    // ──────────────────────────────────────────────────────────────
    // Phase 7: Verdict.
    // ──────────────────────────────────────────────────────────────

    banner("Verdict — what naive vs substrate say");
    println!();
    println!("  NAIVE TALLY:");
    println!("    2 sources say yes, 4 sources say no — attack-dominant by 2.");
    println!("    Decision: 'no' wins. But three of the four 'no' sources are");
    println!("    BPI, BDO, BSP — and BSP just repeats BPI+BDO. A naive approach");
    println!("    would give BSP the same weight as BPI and BDO.");
    println!();
    println!("  SUBSTRATE:");
    let mobility = db.get_mobility_state(&prop_id, "default").unwrap().unwrap();
    println!("    σ = {:.3}  (dependence-discounted support)", mobility.support_mass.unwrap());
    println!("    α = {:.3}  (dependence-discounted attack)",  mobility.attack_mass.unwrap());
    println!("    support_effective_independence = {:.3}", contest.support_effective_independence);
    println!("    attack_effective_independence  = {:.3}", contest.attack_effective_independence);
    println!();
    println!("    ey.audit shares lineage with wirecard.filing → support is");
    println!("    closer to 1 independent source, not 2. Correctly reflects");
    println!("    that EY's 'confirmation' came from Wirecard-provided documents.");
    println!();
    println!("    bsp.circular shares lineage with bpi+bdo → attack effective");
    println!("    independence is less than 4. BSP is an authoritative");
    println!("    determination but not a fourth independent witness.");
    println!();
    println!("  That's what RFC 008 buys — grounded epistemic accounting,");
    println!("  not a counting argument.");

    banner("done");
}

// ──────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────

fn ingest_claim_with_lineage(
    db: &YantrikDB,
    prop: (&str, &str, &str),
    polarity: i32,
    modality: &str,
    extractor: &str,
    valid_from: Option<f64>,
    valid_to: Option<f64>,
    lineage: &[&str],
) -> String {
    let (src, rel, dst) = prop;
    let claim_id = db.ingest_claim(
        src, rel, dst, "default",
        polarity, modality,
        valid_from, valid_to,
        extractor, None, "medium",
        None, None, None,
        1.0,
    ).expect("ingest_claim");
    // Post-insert: patch source_lineage. The default-schema-constrained
    // ingest_claim doesn't take lineage directly, but the substrate math
    // keys on the claims.source_lineage column. Re-ingestion would hit
    // the UNIQUE constraint, so we UPDATE in place before the hook
    // would have captured the final row (the hook already ran; re-
    // triggering recompute picks up the updated lineage).
    db.conn().execute(
        "UPDATE claims SET source_lineage = ?1 WHERE claim_id = ?2",
        rusqlite::params![
            serde_json::to_string(lineage).unwrap(),
            claim_id
        ],
    ).unwrap();
    // Re-trigger mobility + contest recompute so the new lineage is reflected.
    let prop_id = find_proposition(db, prop);
    db.compute_write_tier_mobility(&prop_id, "default").unwrap();
    db.compute_contest_state(&prop_id, "default").unwrap();
    claim_id
}

fn find_proposition(db: &YantrikDB, prop: (&str, &str, &str)) -> String {
    let (src, rel, dst) = prop;
    db.conn().query_row(
        "SELECT proposition_id FROM propositions \
         WHERE src = ?1 AND rel_type = ?2 AND dst = ?3 AND namespace = 'default'",
        rusqlite::params![src, rel, dst],
        |r| r.get(0),
    ).expect("proposition exists")
}

fn fetch_claim_ids_for_prop(db: &YantrikDB, prop_id: &str) -> Vec<String> {
    let conn = db.conn();
    let mut stmt = conn.prepare(
        "SELECT claim_id FROM claims \
         WHERE proposition_id = ?1 AND tombstoned = 0 \
         ORDER BY created_at ASC",
    ).unwrap();
    stmt.query_map(rusqlite::params![prop_id], |row| row.get::<_, String>(0))
        .unwrap()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap()
}

/// Compute unix timestamp (seconds) for a date. Hand-rolled to avoid a
/// chrono dependency for this example.
fn d(year: i32, month: u32, day: u32) -> f64 {
    // Days since epoch 1970-01-01, computed via the "Zeller-ish" approach.
    // Only valid for Gregorian dates 1970-2100.
    let mut days: i64 = 0;
    for y in 1970..year {
        days += if is_leap(y) { 366 } else { 365 };
    }
    let months_in_year = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    for m in 1..month {
        days += months_in_year[(m - 1) as usize] as i64;
        if m == 2 && is_leap(year) {
            days += 1;
        }
    }
    days += (day - 1) as i64;
    (days * 86_400) as f64
}

fn is_leap(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn banner(title: &str) {
    println!("\n{}", "=".repeat(74));
    println!(" {}", title);
    println!("{}", "=".repeat(74));
}

fn sub(title: &str) {
    println!("\n-- {} {}", title, "-".repeat(70usize.saturating_sub(title.len())));
}
