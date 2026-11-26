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
        }),
    )
}

fn lo(n: &CognitiveNode) -> f64 {
    match &n.payload {
        NodePayload::Belief(b) => b.log_odds,
        _ => panic!(),
    }
}

/// Ein zweiter Lauf zur SELBEN Zeit darf nicht erneut denselben Zeitraum verfallen lassen.
///
/// BEKANNTER DEFEKT (25.11.2026, vor dem Fix): gemessen 0,1000 → 0,0050.
/// `apply_staleness_decay` rechnete „Tage seit der letzten Evidenz" bei JEDEM
/// Aufruf neu und wandte den Faktor auf den bereits verfallenen Wert an.
/// Der Fix rechnet vom unverfallenen Grundwert (`undecayed_log_odds`).
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
        paths: Vec::new(),
    }];
    let mut a = belief(21, trail.clone(), 2.0);
    let mut b = belief(22, trail, 2.0);

    revise_beliefs(&mut [&mut a], now, &cfg);
    let once = lo(&a);
    // derselbe Zeitpunkt, zweiter Aufruf
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

/// Der Verfall muss von der verstrichenen Zeit abhängen, nicht von der Zahl der Aufrufe.
///
/// BEKANNTER DEFEKT (25.11.2026, vor dem Fix): 1 Aufruf nach 30 Tagen = 0,1000,
/// 30 Aufrufe = 0,0000 — ein Wartungslauf, der je Turn `revise_beliefs` ruft,
/// löschte jede Überzeugung. Der Fix macht das Ergebnis zu einer Funktion von
/// (Evidenz, verstrichene Zeit, Persistenz) statt von der Aufrufzahl.
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
                paths: Vec::new(),
            }],
            2.0,
        )
    };

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
        paths: Vec::new(),
    }]
}

/// Der Verfall wiegt die Persistenz des Knotens: derselbe Beleg, dieselbe Zeit,
/// aber `attrs.persistence` 0,2 gegen 0,9. Geprüft wird gegen die dokumentierte
/// Formel, nicht gegen „kleiner als vorher".
#[test]
fn persistence_weights_the_time_decay() {
    let cfg = BeliefRevisionConfig::default();
    let now = 1_800_000_000.0_f64;

    let mut low = belief(27, old_trail(now), 2.0);
    low.attrs.persistence = 0.2;
    let mut high = belief(28, old_trail(now), 2.0);
    high.attrs.persistence = 0.9;

    revise_beliefs(&mut [&mut low], now, &cfg);
    revise_beliefs(&mut [&mut high], now, &cfg);

    let days = 60.0 - cfg.staleness_grace_period_days;
    let expect_low = 2.0 * (1.0 - (cfg.staleness_rate_per_day * (1.0 - 0.2) * days).min(0.95));
    let expect_high = 2.0 * (1.0 - (cfg.staleness_rate_per_day * (1.0 - 0.9) * days).min(0.95));
    println!(
        "PERSISTENZ-GEWICHT: p=0,2 -> {:.4} (erwartet {expect_low:.4}) · p=0,9 -> {:.4} (erwartet {expect_high:.4})",
        lo(&low), lo(&high)
    );
    assert!(
        (lo(&low) - expect_low).abs() < 1e-9,
        "decay must use rate * (1 - persistence)"
    );
    assert!((lo(&high) - expect_high).abs() < 1e-9);
    assert!(lo(&high) > lo(&low), "higher persistence must decay less");
}

/// Persistenz 1,0 schützt vor ZEITVERFALL — nicht vor Widerspruch.
#[test]
fn persistence_one_survives_time_but_not_contradiction() {
    let cfg = BeliefRevisionConfig::default();
    let now = 1_800_000_000.0_f64;
    let mut node = belief(29, old_trail(now), 2.0);
    node.attrs.persistence = 1.0;
    let id = node.id;

    revise_beliefs(&mut [&mut node], now + 3650.0 * 86400.0, &cfg);
    assert!(
        (lo(&node) - 2.0).abs() < 1e-12,
        "persistence 1.0 must not decay with time"
    );

    let counter = Evidence {
        target_belief: id,
        weight: -3.0,
        source: "gegenprobe".into(),
        provenance: Provenance::Observed,
        propagate: false,
        timestamp: now,
        observation_id: None,
    };
    let r = assert_evidence(&mut node, &counter, &cfg).unwrap();
    println!(
        "PERSISTENZ 1,0: nach 10 Jahren {:.4} · nach Widerspruch {:.4}",
        2.0, r.posterior_log_odds
    );
    assert!(
        r.posterior_log_odds < 2.0 - 1.0,
        "contradicting evidence must still revise it"
    );
}

/// Eine erneute Assertion derselben Beobachtung darf den bereits verfallenen
/// Wert nicht als unverfallenen Grundwert übernehmen — sonst wird derselbe
/// Zeitraum beim nächsten Lauf ein zweites Mal abgezogen. Gefunden im
/// Zwei-Prozess-Abnahmelauf (examples/belief_acceptance.rs), nicht im Unit-Test.
#[test]
fn reassertion_after_decay_does_not_charge_the_same_time_twice() {
    let cfg = BeliefRevisionConfig::default();
    let now = 1_800_000_000.0_f64;
    let oid = "b31|rid|1";
    // Gewicht so, wie es eine Assertion mit 2,0 unter Observed (0,9) hinterlässt
    let trail = vec![EvidenceEntry {
        source: "alt".into(),
        weight: 1.8,
        timestamp: now - 60.0 * 86400.0,
        observation_id: Some(oid.into()),
        history: Vec::new(),
        paths: Vec::new(),
    }];
    let mut node = belief(31, trail, 1.8);
    let id = node.id;

    revise_beliefs(&mut [&mut node], now, &cfg);
    let decayed = lo(&node);

    // dieselbe Beobachtung, gleiches effektives Gewicht: keine Neubewertung
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
        "ERNEUTE ASSERTION: verfällt {decayed:.4} · nach Assertion {after_assert:.4} · nach Revision {after_revision:.4}"
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

/// Die beiden Zustände, auf denen der Fix beruht, müssen PERSISTIERT sein:
/// `attrs.persistence` (eigene Spalte) und `BeliefPayload.undecayed_log_odds`
/// (JSON-Payload). Ein Fix, der nur im Speicher gilt, wäre nach einem Neustart
/// wieder der alte Defekt.
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
