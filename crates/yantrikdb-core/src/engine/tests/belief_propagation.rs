use super::*;
use crate::belief::{propagate_evidence, revise_beliefs, BeliefRevisionConfig};
use crate::state::{BeliefPayload, CognitiveEdge, CognitiveEdgeKind, EvidenceEntry};
use crate::{CognitiveNode, NodeId, NodeKind, NodePayload};
use std::collections::HashMap;

fn belief(seq: u32, prop: &str, trail: Vec<EvidenceEntry>) -> CognitiveNode {
    let id = NodeId::new(NodeKind::Belief, seq);
    CognitiveNode::new(
        id,
        format!("belief-{seq}"),
        NodePayload::Belief(BeliefPayload {
            proposition: prop.to_string(),
            log_odds: 0.0,
            domain: "general".to_string(),
            evidence_trail: trail,
            user_confirmed: false,
            undecayed_log_odds: None,
        }),
    )
}

fn lo(n: &CognitiveNode) -> f64 {
    match &n.payload {
        NodePayload::Belief(b) => b.log_odds,
        _ => panic!("not a belief"),
    }
}

fn edge(src: NodeId, dst: NodeId, kind: CognitiveEdgeKind, weight: f64) -> CognitiveEdge {
    CognitiveEdge {
        src,
        dst,
        kind,
        weight,
        created_at_ms: 1_800_000_000_000,
        last_confirmed_ms: 1_800_000_000_000,
        observation_count: 1,
        confidence: 1.0,
    }
}

/// Eine Supports-Kante trägt Evidenz auf den Zielknoten.
#[test]
fn supports_edge_raises_target() {
    let a = belief(11, "Quelle.", vec![]);
    let b = belief(12, "Ziel.", vec![]);
    let e = edge(a.id, b.id, CognitiveEdgeKind::Supports, 1.0);
    let mut down: HashMap<NodeId, CognitiveNode> = HashMap::new();
    down.insert(b.id, b);

    let effects = propagate_evidence(
        a.id,
        1.0,
        &[e],
        &mut down,
        &BeliefRevisionConfig::default(),
        0,
    );
    assert!(!effects.is_empty(), "supports edge must produce an effect");
    assert!(lo(&down[&NodeId::new(NodeKind::Belief, 12)]) > 0.0);
}

/// Eine Contradicts-Kante senkt den Zielknoten.
#[test]
fn contradicts_edge_lowers_target() {
    let a = belief(13, "Quelle.", vec![]);
    let b = belief(14, "Ziel.", vec![]);
    let e = edge(a.id, b.id, CognitiveEdgeKind::Contradicts, 1.0);
    let mut down: HashMap<NodeId, CognitiveNode> = HashMap::new();
    down.insert(b.id, b);

    let effects = propagate_evidence(
        a.id,
        1.0,
        &[e],
        &mut down,
        &BeliefRevisionConfig::default(),
        0,
    );
    assert!(!effects.is_empty());
    assert!(lo(&down[&NodeId::new(NodeKind::Belief, 14)]) < 0.0);
}

/// Ein Zyklus (A stützt B, B stützt A) darf die Konfidenz nicht aufschaukeln:
/// nach mehreren Hops bleibt der Betrag unter dem Deckel der Konfiguration.
#[test]
fn support_cycle_stays_bounded() {
    let cfg = BeliefRevisionConfig::default();
    let a = belief(15, "A.", vec![]);
    let b = belief(16, "B.", vec![]);
    let (ai, bi) = (a.id, b.id);
    let edges = vec![
        edge(ai, bi, CognitiveEdgeKind::Supports, 1.0),
        edge(bi, ai, CognitiveEdgeKind::Supports, 1.0),
    ];
    let mut down: HashMap<NodeId, CognitiveNode> = HashMap::new();
    down.insert(ai, a);
    down.insert(bi, b);

    for hop in 0..6u32 {
        let src = if hop % 2 == 0 { ai } else { bi };
        let _ = propagate_evidence(src, 1.0, &edges, &mut down, &cfg, hop);
    }
    let max_seen = lo(&down[&ai]).abs().max(lo(&down[&bi]).abs());
    println!(
        "ZYKLUS max_abs_log_odds nach 6 Hops = {max_seen:.4} (Deckel {:.1})",
        cfg.max_abs_log_odds
    );
    assert!(
        max_seen <= cfg.max_abs_log_odds + 1e-9,
        "cycle amplification exceeded the configured ceiling: {max_seen}"
    );
}

/// Veraltung: ein Belief ohne neue Evidenz über die Karenzzeit hinaus verliert Konfidenz,
/// und `revise_beliefs` meldet die Revision.
#[test]
fn revise_beliefs_decays_stale_evidence() {
    let cfg = BeliefRevisionConfig::default();
    let now = 1_800_000_000.0_f64;
    let old = now - 60.0 * 86400.0; // 60 Tage alt, Karenz ist 7
    let mut b = belief(
        17,
        "Nur alte Belege.",
        vec![EvidenceEntry {
            source: "alt".into(),
            weight: 2.0,
            timestamp: old,
            observation_id: None,
            history: Vec::new(),
            paths: Vec::new(),
        }],
    );
    if let NodePayload::Belief(p) = &mut b.payload {
        p.log_odds = 2.0;
    }
    let before = lo(&b);
    let summary = revise_beliefs(&mut [&mut b], now, &cfg);
    let after = lo(&b);
    println!(
        "REVISION vorher={before:.4} nachher={after:.4} decays={} revised={}",
        summary.staleness_decays, summary.beliefs_revised
    );
    assert!(after < before, "stale belief must lose confidence");
    assert!(summary.staleness_decays >= 1);
}
