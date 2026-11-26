use super::*;
use crate::belief::{
    assert_evidence, propagate_evidence, revise_beliefs, BeliefRevisionConfig, Evidence,
};
use crate::state::{BeliefPayload, CognitiveEdge, CognitiveEdgeKind, EvidenceEntry, Provenance};
use crate::{CognitiveNode, NodeId, NodeKind, NodePayload};
use std::collections::HashMap;

fn belief(seq: u32, trail: Vec<EvidenceEntry>, log_odds: f64) -> CognitiveNode {
    let id = NodeId::new(NodeKind::Belief, seq);
    CognitiveNode::new(
        id,
        format!("b{seq}"),
        NodePayload::Belief(BeliefPayload {
            proposition: "Probe.".into(),
            log_odds,
            domain: "general".into(),
            evidence_trail: trail,
            user_confirmed: false,
            undecayed_log_odds: None,
            decayed_at: 0.0,
            propagated: Vec::new(),
        }),
    )
}

fn lo(n: &CognitiveNode) -> f64 {
    match &n.payload {
        NodePayload::Belief(b) => b.log_odds,
        _ => panic!(),
    }
}

/// A second run at the SAME instant must not decay the same span again.
///
/// KNOWN DEFECT (25.11.2026, before the fix): measured 0.1000 -> 0.0050.
/// `apply_staleness_decay` recomputed "days since the last evidence" on EVERY
/// call and applied the factor to the already decayed value. The fix computes
/// from the undecayed base (`undecayed_log_odds`).
#[test]
fn repeated_revision_at_same_time_is_idempotent() {
    let cfg = BeliefRevisionConfig::default();
    let now = 1_800_000_000.0_f64;
    let trail = vec![EvidenceEntry {
        source: "alt".into(),
        weight: 2.0,
        timestamp: now - 60.0 * 86400.0,
        observation_id: None,
        history: Vec::new(),
    }];
    let mut a = belief(21, trail.clone(), 2.0);
    let mut b = belief(22, trail, 2.0);

    revise_beliefs(&mut [&mut a], now, &cfg);
    let once = lo(&a);
    // same instant, second call
    revise_beliefs(&mut [&mut a], now, &cfg);
    let twice = lo(&a);

    revise_beliefs(&mut [&mut b], now, &cfg);
    println!(
        "SELBE ZEIT: einmal={once:.4} zweimal={twice:.4} (Referenz {:.4})",
        lo(&b)
    );
    assert!(
        (twice - once).abs() < 1e-9,
        "second revision at the same instant decayed again: {once} -> {twice}"
    );
}

/// Decay must depend on elapsed time, not on the number of calls.
///
/// KNOWN DEFECT (25.11.2026, before the fix): one call after 30 days = 0.1000,
/// thirty calls = 0.0000 — a maintenance run that calls `revise_beliefs` once
/// per turn erased every belief. The fix makes the result a function of
/// (evidence, elapsed time) instead of the call count.
#[test]
fn decay_follows_elapsed_time_not_call_count() {
    let cfg = BeliefRevisionConfig::default();
    let t0 = 1_800_000_000.0_f64;
    let mk = |seq| {
        belief(
            seq,
            vec![EvidenceEntry {
                source: "alt".into(),
                weight: 2.0,
                timestamp: t0 - 60.0 * 86400.0,
                observation_id: None,
                history: Vec::new(),
            }],
            2.0,
        )
    };

    // A: one call, 30 days after t0
    let mut a = mk(23);
    revise_beliefs(&mut [&mut a], t0 + 30.0 * 86400.0, &cfg);
    let one_call = lo(&a);

    // B: thirty calls, one day apart
    let mut b = mk(24);
    for d in 1..=30 {
        revise_beliefs(&mut [&mut b], t0 + d as f64 * 86400.0, &cfg);
    }
    let thirty_calls = lo(&b);

    println!(
        "AUFRUFABHÄNGIGKEIT: 1 Aufruf nach 30 d = {one_call:.4} · 30 Aufrufe = {thirty_calls:.4}"
    );
    assert!(
        (one_call - thirty_calls).abs() < 1e-6,
        "decay depends on how often revise_beliefs runs: {one_call} vs {thirty_calls}"
    );
}

fn edge(src: NodeId, dst: NodeId) -> CognitiveEdge {
    CognitiveEdge {
        src,
        dst,
        kind: CognitiveEdgeKind::Supports,
        weight: 1.0,
        created_at_ms: 1_800_000_000_000,
        last_confirmed_ms: 1_800_000_000_000,
        observation_count: 1,
        confidence: 1.0,
    }
}

/// The same edge propagated twice: does it count twice?
#[test]
fn repeated_propagation_of_one_edge() {
    let cfg = BeliefRevisionConfig::default();
    let a = belief(25, vec![], 0.0);
    let b = belief(26, vec![], 0.0);
    let (ai, bi) = (a.id, b.id);

    let mut once: HashMap<NodeId, CognitiveNode> = HashMap::new();
    once.insert(bi, b.clone());
    let _ = propagate_evidence(ai, 1.0, &[edge(ai, bi)], &mut once, &cfg, 0);

    let mut twice: HashMap<NodeId, CognitiveNode> = HashMap::new();
    twice.insert(bi, b);
    let _ = propagate_evidence(ai, 1.0, &[edge(ai, bi)], &mut twice, &cfg, 0);
    let _ = propagate_evidence(ai, 1.0, &[edge(ai, bi)], &mut twice, &cfg, 0);

    let o = lo(&once[&bi]);
    let t = lo(&twice[&bi]);
    println!(
        "PROPAGATION: einmal={o:.4} zweimal={t:.4} doppelt={}",
        (t - 2.0 * o).abs() < 1e-9
    );
    assert!(t >= o);
}

fn old_trail(now: f64) -> Vec<EvidenceEntry> {
    vec![EvidenceEntry {
        source: "alt".into(),
        weight: 2.0,
        timestamp: now - 60.0 * 86400.0,
        observation_id: None,
        history: Vec::new(),
    }]
}

/// A re-assertion of the same observation must not take the already decayed
/// value as an undecayed base — otherwise the same span is charged twice on the
/// next run. Found in a two-process acceptance run, not in a unit test.
#[test]
fn reassertion_after_decay_does_not_charge_the_same_time_twice() {
    let cfg = BeliefRevisionConfig::default();
    let now = 1_800_000_000.0_f64;
    let oid = "b31|rid|1";
    // weight as an assertion of 2.0 under Observed (0.9) leaves it
    let trail = vec![EvidenceEntry {
        source: "alt".into(),
        weight: 1.8,
        timestamp: now - 60.0 * 86400.0,
        observation_id: Some(oid.into()),
        history: Vec::new(),
    }];
    let mut node = belief(31, trail, 1.8);
    let id = node.id;

    revise_beliefs(&mut [&mut node], now, &cfg);
    let decayed = lo(&node);

    // same observation, same effective weight: no re-assessment
    let again = Evidence {
        target_belief: id,
        weight: 2.0,
        source: "alt".into(),
        provenance: Provenance::Observed,
        propagate: false,
        timestamp: now - 60.0 * 86400.0,
        observation_id: Some(oid.into()),
    };
    let r = assert_evidence(&mut node, &again, &cfg).unwrap();
    let after_assert = r.posterior_log_odds;

    revise_beliefs(&mut [&mut node], now, &cfg);
    let after_revision = lo(&node);

    println!(
        "RE-ASSERTION: decayed {decayed:.4} · after assertion {after_assert:.4} · after revision {after_revision:.4}"
    );
    assert!(
        (after_assert - decayed).abs() < 1e-12,
        "an unchanged re-assertion must not move the value"
    );
    assert!(
        (after_revision - decayed).abs() < 1e-12,
        "the elapsed time must not be charged twice after a re-assertion"
    );
}

/// The state the fix rests on must be PERSISTED: `BeliefPayload.undecayed_log_odds`
/// and `decayed_at` (both in the JSON payload). A fix that only holds in memory
/// would be the old defect again after a restart.
#[test]
fn decay_state_survives_persistence() {
    let cfg = BeliefRevisionConfig::default();
    let now = 1_800_000_000.0_f64;
    let db = YantrikDB::new(":memory:", 4).unwrap();

    let mut node = belief(30, old_trail(now), 2.0);
    node.attrs.persistence = 0.9;
    db.persist_cognitive_node(&node).unwrap();

    let mut loaded = db.load_cognitive_node(node.id).unwrap().unwrap();
    println!(
        "PERSISTENZ-SPALTE: gespeichert 0,9 · geladen {:.4}",
        loaded.attrs.persistence
    );
    assert!(
        (loaded.attrs.persistence - 0.9).abs() < 1e-12,
        "persistence must round-trip"
    );

    revise_beliefs(&mut [&mut loaded], now, &cfg);
    let first = lo(&loaded);
    db.persist_cognitive_node(&loaded).unwrap();

    let mut reloaded = db.load_cognitive_node(node.id).unwrap().unwrap();
    match &reloaded.payload {
        NodePayload::Belief(b) => assert!(
            b.undecayed_log_odds.is_some(),
            "the decay anchor must survive the payload round-trip"
        ),
        _ => panic!("expected belief"),
    }
    revise_beliefs(&mut [&mut reloaded], now, &cfg);
    println!(
        "PERSISTENZ-RUNDLAUF: erster Lauf {first:.4} · nach Neuladen {:.4}",
        lo(&reloaded)
    );
    assert!(
        (lo(&reloaded) - first).abs() < 1e-12,
        "a reloaded belief must not decay again for the same elapsed time"
    );
}

// ── The three sequences from the maintainer review (26.11.2026) ────────────

/// A capped belief must not come back after the next decay.
///
/// Review: two Told +5.0, then Observed -3.0 (the cap fires), then decay after
/// 8 days. The anchor stayed at 6.8 and decay rebuilt 6.0 from it instead of
/// 3.234 — the evidence the cap had removed came back.
#[test]
fn a_capped_belief_does_not_come_back_after_decay() {
    let cfg = BeliefRevisionConfig::default();
    let now = 1_800_000_000.0_f64;
    let mut node = belief(40, Vec::new(), 0.0);
    let id = node.id;

    for _ in 0..2 {
        assert_evidence(
            &mut node,
            &Evidence {
                target_belief: id,
                weight: 5.0,
                source: "told".into(),
                provenance: Provenance::Told,
                propagate: false,
                timestamp: now,
                observation_id: None,
            },
            &cfg,
        )
        .unwrap();
    }
    let after_support = lo(&node);

    assert_evidence(
        &mut node,
        &Evidence {
            target_belief: id,
            weight: -3.0,
            source: "seen".into(),
            provenance: Provenance::Observed,
            propagate: false,
            timestamp: now,
            observation_id: None,
        },
        &cfg,
    )
    .unwrap();
    let after_contradiction = lo(&node);
    assert!(
        after_contradiction < after_support,
        "the contradiction must lower the belief: {after_support} -> {after_contradiction}"
    );

    revise_beliefs(&mut [&mut node], now + 8.0 * 86400.0, &cfg);
    let after_decay = lo(&node);

    assert!(
        after_decay <= after_contradiction + 1e-9,
        "decay must not raise the belief above its pre-decay value \
         (was {after_contradiction}, became {after_decay}) — that is the cap \
         stored in the anchor, not evidence"
    );
}

/// A propagated contribution without an origin must sit in the anchor, or the
/// next decay erases it (review: 2.9 -> 0.882).
#[test]
fn a_propagated_contribution_is_in_the_anchor() {
    let now = 1_800_000_000.0_f64;
    let mut node = belief(41, Vec::new(), 0.9);
    if let NodePayload::Belief(b) = &mut node.payload {
        b.undecayed_log_odds = Some(0.9);
        b.decayed_at = now - 30.0 * 86400.0;
    }

    if let NodePayload::Belief(b) = &mut node.payload {
        b.shift_log_odds(2.0);
    }

    match &node.payload {
        NodePayload::Belief(b) => assert!(
            (b.undecayed_log_odds.unwrap_or(f64::NAN) - 2.9).abs() < 1e-9,
            "the anchor must carry the propagated weight: {:?}",
            b.undecayed_log_odds
        ),
        _ => panic!("expected belief"),
    }
}

/// Waiting must not make a belief more confident.
///
/// Review: 2.7 decays over a year to 0.135. Then +0.45 (-> 0.585). Eight days
/// later, with no new evidence, the patch rebuilt 3.087 — the old, long-decayed
/// evidence came back through the anchor.
#[test]
fn waiting_does_not_make_a_decayed_belief_more_confident() {
    let cfg = BeliefRevisionConfig::default();
    let now = 1_800_000_000.0_f64;
    let mut node = belief(42, Vec::new(), 0.135);
    let id = node.id;
    if let NodePayload::Belief(b) = &mut node.payload {
        b.undecayed_log_odds = Some(2.7);
        b.decayed_at = now - 365.0 * 86400.0;
        b.evidence_trail.push(EvidenceEntry {
            source: "alt".into(),
            weight: 2.7,
            timestamp: now - 365.0 * 86400.0,
            observation_id: None,
            history: Vec::new(),
        });
    }

    assert_evidence(
        &mut node,
        &Evidence {
            target_belief: id,
            weight: 0.45,
            source: "neu".into(),
            provenance: Provenance::Observed,
            propagate: false,
            timestamp: now,
            observation_id: None,
        },
        &cfg,
    )
    .unwrap();
    let after_assert = lo(&node);

    revise_beliefs(&mut [&mut node], now + 8.0 * 86400.0, &cfg);
    let after_wait = lo(&node);

    assert!(
        after_wait <= after_assert + 1e-9,
        "eight days without new evidence must not raise the belief \
         ({after_assert} -> {after_wait})"
    );
}

/// Main's rate, unscaled: a default belief at 1.9 decays to 0.76 at +37 days,
/// and `persistence` no longer changes that (0, 0.8 and 1.0 agree).
///
/// The persistence-scaled rate was a fivefold slowdown of every default belief
/// and made persistence 1.0 immortal. It is out of this PR; if it comes back,
/// it is an opt-in config field, not a default.
#[test]
fn default_decay_rate_is_unscaled_by_persistence() {
    let cfg = BeliefRevisionConfig::default();
    let now = 1_800_000_000.0_f64;
    let mut results = Vec::new();

    for (i, persistence) in [0.0, 0.8, 1.0].into_iter().enumerate() {
        // Evidence at `now`, decay pass at +37 days: 30 effective days after
        // the 7-day grace, at the configured rate.
        let trail = vec![EvidenceEntry {
            source: "alt".into(),
            weight: 1.9,
            timestamp: now,
            observation_id: None,
            history: Vec::new(),
        }];
        let mut node = belief(50 + i as u32, trail, 1.9);
        node.attrs.persistence = persistence;
        revise_beliefs(&mut [&mut node], now + 37.0 * 86400.0, &cfg);
        results.push((persistence, lo(&node)));
    }

    for (persistence, value) in &results {
        assert!(
            (value - 0.76).abs() < 1e-9,
            "persistence {persistence}: expected 0.76, got {value}"
        );
    }
    assert!(
        results
            .iter()
            .all(|(_, v)| (v - results[0].1).abs() < 1e-12),
        "the decay rate must not depend on persistence: {results:?}"
    );
}

/// Direct and propagated evidence about the same origin must not depend on
/// arrival order.
///
/// Review (26.11.): direct-then-propagated gave 0.9, propagated-then-direct
/// gave 1.9, because the "direct outranks propagation" rule was only consulted
/// in one direction. With propagation in its own channel the rule is gone and
/// both orders give main's 1.9.
#[test]
fn direct_and_propagated_evidence_are_order_independent() {
    let now = 1_800_000_000.0_f64;
    let origin = "rid-x";
    let via = NodeId::new(NodeKind::Belief, 62);

    // Order A: direct first, then the propagation of the same origin.
    let mut a = belief(60, Vec::new(), 0.0);
    if let NodePayload::Belief(b) = &mut a.payload {
        b.update_with_id(1.0, 0.9, "obs", now, Some(origin));
        b.update_propagated(1.0, now, origin, via);
    }
    let a_value = lo(&a);

    // Order B: the propagation arrives first, then the direct observation.
    let mut b = belief(61, Vec::new(), 0.0);
    if let NodePayload::Belief(payload) = &mut b.payload {
        payload.update_propagated(1.0, now, origin, via);
        payload.update_with_id(1.0, 0.9, "obs", now, Some(origin));
    }
    let b_value = lo(&b);

    println!("ORDER: direct-first {a_value:.4} · propagated-first {b_value:.4}");
    assert!(
        (a_value - b_value).abs() < 1e-9,
        "the result must not depend on arrival order: direct-first {a_value}, propagated-first {b_value}"
    );
    assert!(
        (a_value - 1.9).abs() < 1e-9,
        "direct 0.9 + propagated 1.0 must be main's 1.9, got {a_value}"
    );
}

/// Re-assessing an aged observation must not flip its sign.
///
/// Review (26.11.): +2.0 with an id decays to 0.1; re-asserted at 1.0 it gave
/// -0.9 and retracted -1.9, because the undecayed delta was applied to the
/// decayed value. A revision moves toward the new weight and never crosses the
/// prior because of decay.
#[test]
fn re_assessing_an_aged_observation_does_not_flip_the_sign() {
    let cfg = BeliefRevisionConfig::default();
    let now = 1_800_000_000.0_f64;
    let oid = "obs-aged";
    // 2.0 asserted under Observed (0.9) leaves 1.8 in the trail and in the value.
    let trail = vec![EvidenceEntry {
        source: "alt".into(),
        weight: 1.8,
        timestamp: now - 365.0 * 86400.0,
        observation_id: Some(oid.into()),
        history: Vec::new(),
    }];
    let mut node = belief(63, trail, 1.8);
    let id = node.id;

    revise_beliefs(&mut [&mut node], now, &cfg);
    let decayed = lo(&node);
    assert!(decayed > 0.0 && decayed < 0.2, "fixture: {decayed}");

    // Re-assess the SAME observation at 1.0 (Observed 0.9).
    assert_evidence(
        &mut node,
        &Evidence::new(id, 1.0, "alt", Provenance::Observed, now).with_observation_id(oid),
        &cfg,
    )
    .unwrap();
    let after = lo(&node);
    println!("RE-ASSESS: decayed {decayed:.4} -> re-asserted 1.0 -> {after:.4}");
    assert!(
        after >= 0.0,
        "re-assessing must not cross into disbelief, got {after}"
    );
    assert!(
        after <= decayed + 1e-12,
        "a lower re-assessment must not raise the value, got {after}"
    );

    // Retract the same observation: the belief must lose what it supported,
    // not flip to the opposite.
    assert_evidence(
        &mut node,
        &Evidence::new(id, 0.0, "alt", Provenance::Observed, now).with_observation_id(oid),
        &cfg,
    )
    .unwrap();
    let retracted = lo(&node);
    println!("RETRACT: {after:.4} -> {retracted:.4}");
    assert!(
        (retracted - 0.0).abs() < 1e-12,
        "retracting must reach zero, not -1.9 (got {retracted})"
    );
    match &node.payload {
        NodePayload::Belief(b) => assert!(
            (b.undecayed_log_odds.unwrap_or(f64::NAN) - 0.0).abs() < 1e-12,
            "the anchor must follow the retraction, got {:?}",
            b.undecayed_log_odds
        ),
        _ => panic!("expected belief"),
    }
}

/// A volatile belief's ceiling must go through the anchor.
///
/// Review (26.11.): with volatility 0.9 and two Told +5.0 the value was 4.0
/// while the anchor stayed at 8.75; after 37 days that rebuilt 3.5 from
/// evidence the ceiling had removed. Main's formula gives 1.6.
#[test]
fn the_volatility_ceiling_goes_through_the_anchor() {
    let cfg = BeliefRevisionConfig::default();
    let now = 1_800_000_000.0_f64;
    let mut node = belief(64, Vec::new(), 0.0);
    node.attrs.volatility = 0.9;
    let id = node.id;

    for _ in 0..2 {
        assert_evidence(
            &mut node,
            &Evidence::new(id, 5.0, "told", Provenance::Told, now - 37.0 * 86400.0),
            &cfg,
        )
        .unwrap();
    }

    let capped = lo(&node);
    let anchor = match &node.payload {
        NodePayload::Belief(b) => b.undecayed_log_odds.unwrap_or(f64::NAN),
        _ => panic!("expected belief"),
    };
    println!("CEILING: value {capped:.4} anchor {anchor:.4}");
    assert!(
        (capped - cfg.volatile_ceiling_log_odds).abs() < 1e-12,
        "value must sit at the ceiling, got {capped}"
    );
    assert!(
        (anchor - cfg.volatile_ceiling_log_odds).abs() < 1e-9,
        "the anchor must be the clamped value, not the raw evidence sum ({anchor})"
    );

    revise_beliefs(&mut [&mut node], now, &cfg);
    let decayed = lo(&node);
    println!("CEILING + 37 days: {decayed:.4}");
    assert!(
        (decayed - 1.6).abs() < 1e-6,
        "main's formula gives 1.6, got {decayed}"
    );
}
