use super::*;
use crate::belief::{assert_evidence, observation_key, BeliefRevisionConfig, Evidence};
use crate::state::{BeliefPayload, Provenance};
use crate::{CognitiveNode, NodeId, NodeKind, NodePayload};

fn belief(seq: u32, prior: f64) -> CognitiveNode {
    let id = NodeId::new(NodeKind::Belief, seq);
    CognitiveNode::new(
        id,
        format!("b{seq}"),
        NodePayload::Belief(BeliefPayload {
            proposition: "Probe.".into(),
            log_odds: prior,
            domain: "general".into(),
            evidence_trail: Vec::new(),
            user_confirmed: false,
            undecayed_log_odds: None,
            decayed_at: 0.0,
        }),
    )
}

fn lo(n: &CognitiveNode) -> f64 {
    match &n.payload {
        NodePayload::Belief(b) => b.log_odds,
        _ => panic!(),
    }
}

fn trail(n: &CognitiveNode) -> Vec<crate::state::EvidenceEntry> {
    match &n.payload {
        NodePayload::Belief(b) => b.evidence_trail.clone(),
        _ => panic!(),
    }
}

fn ev(id: NodeId, weight: f64, provenance: Provenance, oid: Option<String>) -> Evidence {
    Evidence {
        target_belief: id,
        weight,
        source: "quelle".into(),
        provenance,
        propagate: false,
        timestamp: 1_800_000_000.0,
        observation_id: oid,
    }
}

/// Regel 1a: EINE Quelle kann mehrere unabhängige Beobachtungen tragen.
#[test]
fn one_source_can_carry_several_observations() {
    let cfg = BeliefRevisionConfig::default();
    let mut node = belief(31, 0.0);
    let id = node.id;
    for local in ["satz-1", "satz-2"] {
        let e = ev(
            id,
            0.5,
            Provenance::Observed,
            Some(observation_key(id, "rid-a", Some(local))),
        );
        let _ = assert_evidence(&mut node, &e, &cfg);
    }
    println!(
        "EINE QUELLE, ZWEI BEOBACHTUNGEN: trail={} log_odds={:.4}",
        trail(&node).len(),
        lo(&node)
    );
    assert_eq!(
        trail(&node).len(),
        2,
        "two local observations must both count"
    );
}

/// Regel 1b: dieselbe Beobachtung erhält durch eine andere Provenienzart KEINE neue Identität.
#[test]
fn provenance_kind_does_not_change_the_identity() {
    let cfg = BeliefRevisionConfig::default();
    let mut node = belief(32, 0.0);
    let id = node.id;
    let key = observation_key(id, "rid-a", Some("satz-1"));
    let _ = assert_evidence(
        &mut node,
        &ev(id, 0.5, Provenance::Observed, Some(key.clone())),
        &cfg,
    );
    let _ = assert_evidence(
        &mut node,
        &ev(id, 0.5, Provenance::Extracted, Some(key)),
        &cfg,
    );
    println!(
        "PROVENIENZART: trail={} log_odds={:.4}",
        trail(&node).len(),
        lo(&node)
    );
    assert_eq!(
        trail(&node).len(),
        1,
        "re-classifying must not create a second observation"
    );
    // Die Identität bleibt, der WERT darf sich ändern: eine andere Provenienzart
    // skaliert die Zuverlässigkeit (Observed 0,9 · Extracted 0,75), die
    // Neubewertung ersetzt also das Gewicht, statt zu addieren.
    let sum: f64 = trail(&node).iter().map(|e| e.weight).sum();
    assert!((lo(&node) - sum).abs() < 1e-12);
    assert!(
        lo(&node) < 0.5,
        "a weaker provenance must lower the weight, not add"
    );
}

/// Regel 2: die vorherige Bewertung bleibt rekonstruierbar.
#[test]
fn re_evaluation_keeps_the_previous_assessment() {
    let cfg = BeliefRevisionConfig::default();
    let mut node = belief(33, 0.0);
    let id = node.id;
    let key = observation_key(id, "rid-a", None);
    let _ = assert_evidence(
        &mut node,
        &ev(id, 1.0, Provenance::Observed, Some(key.clone())),
        &cfg,
    );
    let first = trail(&node)[0].weight;
    let _ = assert_evidence(
        &mut node,
        &ev(id, -0.4, Provenance::Observed, Some(key)),
        &cfg,
    );
    let t = trail(&node);
    println!(
        "NEUBEWERTUNG MIT HISTORIE: aktuell {:.4} · Historie {:?}",
        t[0].weight, t[0].history
    );
    assert_eq!(t.len(), 1, "one observation, not two");
    assert_eq!(
        t[0].history.len(),
        1,
        "the previous assessment must be kept"
    );
    assert!(
        (t[0].history[0].0 - first).abs() < 1e-12,
        "history must hold the old weight"
    );
    assert!((t[0].weight + 0.4 * 0.9).abs() < 1e-9 || t[0].weight < 0.0);
}

/// Regel 3: die Invariante gilt OHNE Prior, Deckel und Verfall.
#[test]
fn invariant_holds_without_prior_ceiling_or_decay() {
    let cfg = BeliefRevisionConfig::default();
    let mut node = belief(34, 0.0);
    let id = node.id;
    for (i, w) in [0.4, -0.2, 0.7].iter().enumerate() {
        let e = ev(
            id,
            *w,
            Provenance::Observed,
            Some(observation_key(id, "rid-a", Some(&i.to_string()))),
        );
        let _ = assert_evidence(&mut node, &e, &cfg);
    }
    let sum: f64 = trail(&node).iter().map(|e| e.weight).sum();
    println!(
        "INVARIANTE ohne Störgrößen: log_odds={:.4} Summe={:.4}",
        lo(&node),
        sum
    );
    assert!((lo(&node) - sum).abs() < 1e-12);
}

/// Regel 3 (Gegenprobe): mit PRIOR gilt `log_odds == prior + Summe`.
#[test]
fn invariant_becomes_prior_plus_sum_with_a_prior() {
    let cfg = BeliefRevisionConfig::default();
    let mut node = belief(35, 0.8);
    let id = node.id;
    for i in 0..2 {
        let e = ev(
            id,
            0.3,
            Provenance::Observed,
            Some(observation_key(id, "rid-a", Some(&i.to_string()))),
        );
        let _ = assert_evidence(&mut node, &e, &cfg);
    }
    let sum: f64 = trail(&node).iter().map(|e| e.weight).sum();
    println!(
        "INVARIANTE mit Prior: log_odds={:.4} prior+Summe={:.4}",
        lo(&node),
        0.8 + sum
    );
    assert!((lo(&node) - (0.8 + sum)).abs() < 1e-12);
}

/// Regel 3 (Gegenprobe): der Konfidenzdeckel bricht die Summeninvariante — bewusst.
#[test]
fn confidence_ceiling_breaks_the_sum_invariant() {
    let cfg = BeliefRevisionConfig::default();
    let mut node = belief(36, 0.0);
    let id = node.id;
    // deutlich über max_abs_log_odds (Default 6.0) aufsummieren
    for i in 0..20 {
        let e = ev(
            id,
            1.0,
            Provenance::Observed,
            Some(observation_key(id, "rid-a", Some(&i.to_string()))),
        );
        let _ = assert_evidence(&mut node, &e, &cfg);
    }
    let sum: f64 = trail(&node).iter().map(|e| e.weight).sum();
    println!(
        "DECKEL: log_odds={:.4} Summe={:.4} Deckel={:.1}",
        lo(&node),
        sum,
        cfg.max_abs_log_odds
    );
    assert!(
        (lo(&node).abs() - cfg.max_abs_log_odds).abs() < 1e-9,
        "ceiling must clamp"
    );
    assert!(
        (lo(&node) - sum).abs() > 1e-6,
        "and the sum invariant must visibly break"
    );
}
