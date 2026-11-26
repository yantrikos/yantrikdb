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
            decayed_at: 0.0,
            propagated: Vec::new(),
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
    Evidence::new(id, weight, source, Provenance::Observed, 1_800_000_000.0)
}

/// The node survives the restart: persist, reload, same state.
#[test]
fn belief_node_persists_and_reloads() {
    let dir = std::env::temp_dir().join(format!("ydb-belief-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("store.db");

    let id = {
        let db = YantrikDB::new(path.to_str().unwrap(), 8).unwrap();
        let mut node = belief_node(1, "A2A coordination follows an agreed procedure.");
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

/// Confirmation raises, contradiction lowers — on the belief level, not at the
/// pattern.
#[test]
fn evidence_raises_and_lowers_log_odds() {
    let mut node = belief_node(2, "Test proposition.");
    let cfg = BeliefRevisionConfig::default();
    let id = id_of(&node);

    let _ = assert_evidence(&mut node, &evidence(id, 1.4, "supports"), &cfg);
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

/// Two supporting pieces of evidence make a different statement than twenty —
/// the number of independent supports is the evidence, not the number of cluster
/// members.
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
    // The same contradicting observation hits the two very differently.
    let _ = assert_evidence(&mut few, &evidence(fid, -1.5, "state changed"), &cfg);
    let _ = assert_evidence(&mut many, &evidence(mid, -1.5, "state changed"), &cfg);
    assert!(
        sigmoid(log_odds_of(&few)) < 0.5,
        "correctly counted belief is refuted"
    );
    assert!(
        sigmoid(log_odds_of(&many)) > 0.9,
        "over-counted belief shrugs it off"
    );
}

/// The same observation twice: with a stable identity it is idempotent, without
/// one it counts twice (current behaviour, pinned).
#[test]
fn same_observation_is_idempotent_with_an_identity() {
    let cfg = BeliefRevisionConfig::default();

    // (a) with observation_id: the same observation twice changes nothing
    let mut node = belief_node(5, "Idempotence with an identity.");
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

    // (b) re-assessment of the same observation: the effect is replaced, not added
    let mut ev2 = evidence(id, -0.6, "re-assessed");
    ev2.observation_id = Some("belief:5|src:rid-a|observed".into());
    let _ = assert_evidence(&mut node, &ev2, &cfg);
    let revised = log_odds_of(&node);
    // Invariant: log_odds is the sum of the trail weights. A re-assessment
    // replaces a weight, it does not add one.
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
    println!(
        "RE-ASSESSMENT: {once:.4} -> {revised:.4} · trail weights {trail_weights:?} · sum {sum:.4}"
    );
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

/// Without an identity the old behaviour stays: every assertion counts.
#[test]
fn without_an_identity_every_assertion_counts() {
    let cfg = BeliefRevisionConfig::default();
    let mut node = belief_node(6, "Without identity.");
    let id = id_of(&node);
    let ev = evidence(id, 1.0, "no identity");
    let _ = assert_evidence(&mut node, &ev, &cfg);
    let once = log_odds_of(&node);
    let _ = assert_evidence(&mut node, &ev, &cfg);
    let twice = log_odds_of(&node);
    println!("NO ID: once={once:.4} twice={twice:.4}");
    assert!((twice - 2.0 * once).abs() < 1e-12);
    // Main's values, pinned: Observed scales 1.0 to 0.9, so two assertions give 1.8.
    assert!((once - 0.9).abs() < 1e-12, "first assertion, got {once}");
    assert!((twice - 1.8).abs() < 1e-12, "second assertion, got {twice}");
}

/// End-to-end through `assert_belief_evidence` with an observation id, against
/// a real database, including a reopen.
///
/// The identity has to survive persistence: the second assertion — before and
/// after the reopen — must revise the entry instead of adding a second vote,
/// and the propagated channel must not appear in the direct trail.
#[test]
fn observation_identity_survives_a_reopen() {
    let dir = std::env::temp_dir().join(format!(
        "ydb-belief-e2e-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("store.db");
    let path = path.to_str().unwrap().to_string();

    let id = NodeId::new(NodeKind::Belief, 7);
    let cfg = BeliefRevisionConfig::default();
    let evidence_with_id = || {
        Evidence::new(
            id,
            2.0,
            "same-source",
            Provenance::Observed,
            1_800_000_000.0,
        )
        .with_observation_id("b7|rid-1|span-3")
    };

    let first_value = {
        let db = YantrikDB::new(&path, 8).unwrap();
        let node = belief_node(7, "Identity is the same observation, not a new one.");
        db.persist_cognitive_node(&node).unwrap();

        let first = db
            .assert_belief_evidence(&evidence_with_id(), &cfg)
            .unwrap()
            .expect("belief present");
        let second = db
            .assert_belief_evidence(&evidence_with_id(), &cfg)
            .unwrap()
            .expect("belief present");
        assert!(
            (first.posterior_log_odds - second.posterior_log_odds).abs() < 1e-12,
            "the same id must revise, not vote twice: {} -> {}",
            first.posterior_log_odds,
            second.posterior_log_odds
        );
        first.posterior_log_odds
    };

    let db = YantrikDB::new(&path, 8).unwrap();
    let loaded = db
        .load_cognitive_node(id)
        .unwrap()
        .expect("node survives the reopen");
    let (trail, ids, propagated_len, stored_value) = match &loaded.payload {
        NodePayload::Belief(b) => (
            b.evidence_trail.len(),
            b.evidence_trail
                .iter()
                .map(|e| e.observation_id.clone())
                .collect::<Vec<_>>(),
            b.propagated.len(),
            b.log_odds,
        ),
        _ => panic!("payload kind changed across reload"),
    };
    println!(
        "REOPEN: value={stored_value:.4} trail={trail} ids={ids:?} propagated={propagated_len}"
    );
    assert_eq!(trail, 1, "one observation, one entry, after the reopen");
    assert_eq!(ids, vec![Some("b7|rid-1|span-3".to_string())]);
    assert_eq!(propagated_len, 0, "direct-only trail");
    assert!((stored_value - first_value).abs() < 1e-12);

    // A re-assertion in the reopened process still revises the same entry.
    let third = db
        .assert_belief_evidence(&evidence_with_id(), &cfg)
        .unwrap()
        .expect("belief present");
    assert!((third.posterior_log_odds - stored_value).abs() < 1e-12);
    let loaded = db.load_cognitive_node(id).unwrap().expect("node");
    match &loaded.payload {
        NodePayload::Belief(b) => assert_eq!(b.evidence_trail.len(), 1),
        _ => panic!("payload kind changed"),
    }

    let _ = std::fs::remove_dir_all(&dir);
}
