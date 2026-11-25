use super::*;
use crate::belief::{propagate_evidence, revise_beliefs, BeliefRevisionConfig};
use crate::state::{BeliefPayload, CognitiveEdge, CognitiveEdgeKind, EvidenceEntry};
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
        }),
    )
}

fn lo(n: &CognitiveNode) -> f64 {
    match &n.payload { NodePayload::Belief(b) => b.log_odds, _ => panic!() }
}

/// Ein zweiter Lauf zur SELBEN Zeit darf nicht erneut denselben Zeitraum verfallen lassen.
///
/// BEKANNTER DEFEKT (25.11.2026): gemessen 0,1000 → 0,0050. `apply_staleness_decay`
/// rechnet „Tage seit der letzten Evidenz" bei JEDEM Aufruf neu und schiebt den
/// Evidenz-Zeitstempel nicht fort — deshalb fällt derselbe Zeitraum mehrfach an.
#[ignore = "known defect: second revision at the same instant decays again (0.1000 -> 0.0050)"]
#[test]
fn repeated_revision_at_same_time_is_idempotent() {
    let cfg = BeliefRevisionConfig::default();
    let now = 1_800_000_000.0_f64;
    let trail = vec![EvidenceEntry { source: "alt".into(), weight: 2.0, timestamp: now - 60.0 * 86400.0, observation_id: None, history: Vec::new(), paths: Vec::new(),}];
    let mut a = belief(21, trail.clone(), 2.0);
    let mut b = belief(22, trail, 2.0);

    revise_beliefs(&mut [&mut a], now, &cfg);
    let once = lo(&a);
    // derselbe Zeitpunkt, zweiter Aufruf
    revise_beliefs(&mut [&mut a], now, &cfg);
    let twice = lo(&a);

    revise_beliefs(&mut [&mut b], now, &cfg);
    println!("SELBE ZEIT: einmal={once:.4} zweimal={twice:.4} (Referenz {:.4})", lo(&b));
    assert!((twice - once).abs() < 1e-9,
            "second revision at the same instant decayed again: {once} -> {twice}");
}

/// Der Verfall muss von der verstrichenen Zeit abhängen, nicht von der Zahl der Aufrufe.
///
/// BEKANNTER DEFEKT (25.11.2026): 1 Aufruf nach 30 Tagen = 0,1000, 30 Aufrufe = 0,0000.
/// Ein Wartungslauf, der je Turn `revise_beliefs` ruft, löscht jede Überzeugung.
#[ignore = "known defect: decay scales with call count (30 calls -> 0.0000 vs 1 call -> 0.1000)"]
#[test]
fn decay_follows_elapsed_time_not_call_count() {
    let cfg = BeliefRevisionConfig::default();
    let t0 = 1_800_000_000.0_f64;
    let mk = |seq| belief(seq, vec![EvidenceEntry { source: "alt".into(), weight: 2.0, timestamp: t0 - 60.0 * 86400.0, observation_id: None, history: Vec::new(), paths: Vec::new(),}], 2.0);

    // A: ein Aufruf, 30 Tage nach t0
    let mut a = mk(23);
    revise_beliefs(&mut [&mut a], t0 + 30.0 * 86400.0, &cfg);
    let one_call = lo(&a);

    // B: dreißig Aufrufe, je einen Tag später
    let mut b = mk(24);
    for d in 1..=30 {
        revise_beliefs(&mut [&mut b], t0 + d as f64 * 86400.0, &cfg);
    }
    let thirty_calls = lo(&b);

    println!("AUFRUFABHÄNGIGKEIT: 1 Aufruf nach 30 d = {one_call:.4} · 30 Aufrufe = {thirty_calls:.4}");
    assert!((one_call - thirty_calls).abs() < 1e-6,
            "decay depends on how often revise_beliefs runs: {one_call} vs {thirty_calls}");
}

fn edge(src: NodeId, dst: NodeId) -> CognitiveEdge {
    CognitiveEdge {
        src, dst, kind: CognitiveEdgeKind::Supports, weight: 1.0,
        created_at_ms: 1_800_000_000_000, last_confirmed_ms: 1_800_000_000_000,
        observation_count: 1, confidence: 1.0,
    }
}

/// Dieselbe Kante zweimal propagiert: zählt sie doppelt?
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

    let o = lo(&once[&bi]); let t = lo(&twice[&bi]);
    println!("PROPAGATION: einmal={o:.4} zweimal={t:.4} doppelt={}", (t - 2.0 * o).abs() < 1e-9);
    assert!(t >= o);
}
