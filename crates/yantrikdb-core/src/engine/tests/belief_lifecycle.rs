use super::*;
use crate::belief::{assert_evidence, BeliefRevisionConfig, Evidence};
use crate::state::{BeliefPayload, Provenance};
use crate::{CognitiveNode, NodeId, NodeKind, NodePayload};

fn sigmoid(l: f64) -> f64 {
    1.0 / (1.0 + (-l).exp())
}

fn belief_node(seq: u32, proposition: &str) -> CognitiveNode {
    let id = NodeId::new(NodeKind::Belief, seq);
    CognitiveNode::new(
        id,
        format!("belief-{seq}"),
        NodePayload::Belief(BeliefPayload {
            proposition: proposition.to_string(),
            log_odds: 0.0,
            domain: "general".to_string(),
            evidence_trail: Vec::new(),
            user_confirmed: false,
            undecayed_log_odds: None,
        }),
    )
}

fn log_odds_of(node: &CognitiveNode) -> f64 {
    match &node.payload {
        NodePayload::Belief(b) => b.log_odds,
        _ => panic!("not a belief"),
    }
}

fn evidence(id: NodeId, weight: f64, source: &str) -> Evidence {
    Evidence {
        target_belief: id,
        weight,
        source: source.to_string(),
        provenance: Provenance::Observed,
        propagate: false,
        timestamp: 1_800_000_000.0,
        observation_id: None,
    }
}

/// Der Knoten überlebt den Neustart: persistieren, neu laden, gleicher Zustand.
#[test]
fn belief_node_persists_and_reloads() {
    let dir = std::env::temp_dir().join(format!("ydb-belief-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("store.db");

    let id = {
        let db = YantrikDB::new(path.to_str().unwrap(), 8).unwrap();
        let mut node = belief_node(1, "A2A-Koordination folgt einem abgestimmten Verfahren.");
        let cfg = BeliefRevisionConfig::default();
        let nid = id_of(&node);
        let _ = assert_evidence(&mut node, &evidence(nid, 1.4, "quelle-a"), &cfg);
        db.persist_cognitive_node(&node).unwrap();
        log_odds_of(&node)
    };

    let db = YantrikDB::new(path.to_str().unwrap(), 8).unwrap();
    let loaded = db
        .load_cognitive_node(NodeId::new(NodeKind::Belief, 1))
        .unwrap()
        .expect("node");
    assert!(
        (log_odds_of(&loaded) - id).abs() < 1e-9,
        "log_odds must survive the restart"
    );
    match &loaded.payload {
        NodePayload::Belief(b) => assert!(!b.evidence_trail.is_empty()),
        _ => panic!("payload kind changed across reload"),
    }
    assert_eq!(db.count_cognitive_nodes(Some(NodeKind::Belief)).unwrap(), 1);
    assert!(db
        .explain_belief(NodeId::new(NodeKind::Belief, 1))
        .unwrap()
        .is_some());
    let _ = std::fs::remove_dir_all(&dir);
}

fn id_of(node: &CognitiveNode) -> NodeId {
    node.id
}

/// Bestätigung hebt, Widerspruch senkt — auf der Belief-Ebene, nicht beim Pattern.
#[test]
fn evidence_raises_and_lowers_log_odds() {
    let mut node = belief_node(2, "Testaussage.");
    let cfg = BeliefRevisionConfig::default();
    let id = id_of(&node);

    let _ = assert_evidence(&mut node, &evidence(id, 1.4, "stuetzt"), &cfg);
    let after_support = log_odds_of(&node);
    assert!(
        after_support > 0.0,
        "supporting evidence must raise log_odds"
    );

    let _ = assert_evidence(&mut node, &evidence(id, -1.3, "widerspricht"), &cfg);
    let after_contra = log_odds_of(&node);
    assert!(
        after_contra < after_support,
        "contradicting evidence must lower log_odds"
    );
}

/// Zwei stützende Belege ergeben eine andere Aussage als zwanzig — die Zahl der
/// unabhängigen Stützungen ist die Evidenz, nicht die Zahl der Cluster-Mitglieder.
#[test]
fn support_scales_with_independent_evidence_not_with_cluster_size() {
    let cfg = BeliefRevisionConfig::default();
    let mut few = belief_node(3, "Zwei unabhaengige Belege.");
    let mut many = belief_node(4, "Zwanzig Cluster-Mitglieder.");
    let (fid, mid) = (id_of(&few), id_of(&many));
    for i in 0..2 {
        let _ = assert_evidence(&mut few, &evidence(fid, 0.35, &format!("beleg-{i}")), &cfg);
    }
    for i in 0..20 {
        let _ = assert_evidence(
            &mut many,
            &evidence(mid, 0.35, &format!("mitglied-{i}")),
            &cfg,
        );
    }
    let p_few = sigmoid(log_odds_of(&few));
    let p_many = sigmoid(log_odds_of(&many));
    assert!(
        p_few < 0.70,
        "two independent supports must stay moderate, got {p_few}"
    );
    assert!(
        p_many > 0.95,
        "twenty supports reach near-certainty, got {p_many}"
    );
    // Dieselbe widersprechende Beobachtung trifft die beiden sehr verschieden.
    let _ = assert_evidence(&mut few, &evidence(fid, -1.5, "zustand geaendert"), &cfg);
    let _ = assert_evidence(&mut many, &evidence(mid, -1.5, "zustand geaendert"), &cfg);
    assert!(
        sigmoid(log_odds_of(&few)) < 0.5,
        "correctly counted belief is refuted"
    );
    assert!(
        sigmoid(log_odds_of(&many)) > 0.9,
        "over-counted belief shrugs it off"
    );
}

/// Dieselbe Beobachtung zweimal: mit stabiler Identität ist sie idempotent,
/// ohne Identität zählt sie doppelt (Ist-Zustand, festgehalten).
#[test]
fn same_observation_is_idempotent_with_an_identity() {
    let cfg = BeliefRevisionConfig::default();

    // (a) mit observation_id: zweimal dieselbe Beobachtung ändert nichts
    let mut node = belief_node(5, "Idempotenz mit Identitaet.");
    let id = id_of(&node);
    let mut ev = evidence(id, 1.0, "dieselbe quelle, derselbe zeitpunkt");
    ev.observation_id = Some("belief:5|src:rid-a|observed".into());
    let _ = assert_evidence(&mut node, &ev, &cfg);
    let once = log_odds_of(&node);
    let _ = assert_evidence(&mut node, &ev, &cfg);
    let twice = log_odds_of(&node);
    let trail = match &node.payload {
        NodePayload::Belief(b) => b.evidence_trail.len(),
        _ => 0,
    };
    println!("MIT ID: einmal={once:.4} zweimal={twice:.4} trail={trail}");
    assert!(
        (twice - once).abs() < 1e-12,
        "same observation counted twice: {once} -> {twice}"
    );
    assert_eq!(trail, 1, "the trail must hold one entry per observation");

    // (b) Neubewertung derselben Beobachtung: Wirkung wird ersetzt, nicht addiert
    let mut ev2 = evidence(id, -0.6, "neu bewertet");
    ev2.observation_id = Some("belief:5|src:rid-a|observed".into());
    let _ = assert_evidence(&mut node, &ev2, &cfg);
    let revised = log_odds_of(&node);
    // Invariante: log_odds ist die Summe der Trail-Gewichte. Eine Neubewertung
    // ersetzt ein Gewicht, sie addiert keines.
    let (sum, trail_weights) = match &node.payload {
        NodePayload::Belief(b) => (
            b.evidence_trail.iter().map(|e| e.weight).sum::<f64>(),
            b.evidence_trail
                .iter()
                .map(|e| e.weight)
                .collect::<Vec<_>>(),
        ),
        _ => (0.0, vec![]),
    };
    println!("NEUBEWERTUNG: {once:.4} -> {revised:.4} · Trail-Gewichte {trail_weights:?} · Summe {sum:.4}");
    assert!(
        (revised - sum).abs() < 1e-12,
        "log_odds must equal the sum of the trail"
    );
    assert_eq!(trail_weights.len(), 1);
    let trail2 = match &node.payload {
        NodePayload::Belief(b) => b.evidence_trail.len(),
        _ => 0,
    };
    assert_eq!(trail2, 1, "re-evaluation must not add a second observation");
}

/// Ohne Identität bleibt das alte Verhalten: jede Äußerung zählt.
#[test]
fn without_an_identity_every_assertion_counts() {
    let cfg = BeliefRevisionConfig::default();
    let mut node = belief_node(6, "Ohne Identitaet.");
    let id = id_of(&node);
    let ev = evidence(id, 1.0, "keine identitaet");
    let _ = assert_evidence(&mut node, &ev, &cfg);
    let once = log_odds_of(&node);
    let _ = assert_evidence(&mut node, &ev, &cfg);
    let twice = log_odds_of(&node);
    println!("OHNE ID: einmal={once:.4} zweimal={twice:.4}");
    assert!((twice - 2.0 * once).abs() < 1e-12);
}
