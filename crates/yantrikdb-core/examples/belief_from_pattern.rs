//! Erkenntniskreislauf: Cluster-Mitglieder -> Belief -> Bestätigung -> Widerspruch.
//!
//! cargo run --release --example belief_from_pattern -- <db-path> <spec.json>
//!
//! spec.json: {"proposition": "...", "rids": ["...", ...], "contradiction": "..."}
//!
//! Die Mitglieder sind die Evidenz (ein Eintrag je Record), die Proposition ist
//! aus dem Cluster abgeleitet. Danach eine widersprechende Beobachtung.

use std::time::{SystemTime, UNIX_EPOCH};

use yantrikdb::belief::{assert_evidence, BeliefRevisionConfig, Evidence};
use yantrikdb::state::{BeliefPayload, Provenance};
use yantrikdb::{CognitiveNode, NodeId, NodeKind, NodePayload, YantrikDB};

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}
fn sigmoid(l: f64) -> f64 {
    1.0 / (1.0 + (-l).exp())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: <db> <spec.json>");
    let spec_path = args.next().expect("usage: <db> <spec.json>");

    let raw = std::fs::read_to_string(&spec_path).expect("read spec");
    let spec: serde_json::Value = serde_json::from_str(&raw).expect("parse spec");
    let proposition = spec["proposition"]
        .as_str()
        .expect("proposition")
        .to_string();
    let rids: Vec<String> = spec["rids"]
        .as_array()
        .expect("rids")
        .iter()
        .map(|v| v.as_str().unwrap_or_default().to_string())
        .collect();
    let contradiction = spec["contradiction"].as_str().unwrap_or("").to_string();

    let db = YantrikDB::new(&path, 1024).expect("open copy");
    let id = NodeId::new(NodeKind::Belief, 1);

    let mut node = CognitiveNode::new(
        id,
        "pattern-derived".to_string(),
        NodePayload::Belief(BeliefPayload {
            proposition: proposition.clone(),
            log_odds: 0.0,
            domain: "general".to_string(),
            evidence_trail: Vec::new(),
            user_confirmed: false,
            undecayed_log_odds: None,
        }),
    );
    db.persist_cognitive_node(&node).expect("persist");
    println!(
        "BELIEF proposition={:?}",
        &proposition.chars().take(90).collect::<String>()
    );
    println!("BELIEF start log_odds=0.0000 p_true=0.5000");

    let cfg = BeliefRevisionConfig::default();

    // Ein Evidenzeintrag je Cluster-Mitglied: die Beobachtung ist die Quelle, nicht eine Zahl.
    let per_member = 0.35; // festes Gewicht JE unabhaengigem Beleg
    for r in &rids {
        let ev = Evidence {
            target_belief: id,
            weight: per_member,
            source: format!("cluster member {r}"),
            provenance: Provenance::Observed,
            propagate: false,
            timestamp: now(),
            observation_id: None,
        };
        let _ = assert_evidence(&mut node, &ev, &cfg);
    }
    let after_support = match &node.payload {
        NodePayload::Belief(b) => b.log_odds,
        _ => 0.0,
    };
    db.persist_cognitive_node(&node).expect("persist support");
    println!(
        "SUPPORT n={} je {:+.3} -> log_odds {:.4} p_true={:.4}",
        rids.len(),
        per_member,
        after_support,
        sigmoid(after_support)
    );

    if !contradiction.is_empty() {
        let ev = Evidence {
            target_belief: id,
            weight: -1.5,
            source: contradiction.clone(),
            provenance: Provenance::Observed,
            propagate: false,
            timestamp: now(),
            observation_id: None,
        };
        let _ = assert_evidence(&mut node, &ev, &cfg);
        let after_contra = match &node.payload {
            NodePayload::Belief(b) => b.log_odds,
            _ => 0.0,
        };
        db.persist_cognitive_node(&node)
            .expect("persist contradict");
        println!(
            "CONTRA weight=-1.5 ({:?}) -> log_odds {:.4} p_true={:.4}",
            &contradiction.chars().take(50).collect::<String>(),
            after_contra,
            sigmoid(after_contra)
        );
    }

    let node2 = db.load_cognitive_node(id).unwrap().expect("reload");
    let p = match &node2.payload {
        NodePayload::Belief(b) => b,
        _ => panic!(),
    };
    println!(
        "FINAL log_odds={:.4} p_true={:.4} trail={} beliefs={:?}",
        p.log_odds,
        sigmoid(p.log_odds),
        p.evidence_trail.len(),
        db.count_cognitive_nodes(Some(NodeKind::Belief)).unwrap()
    );
    if let Ok(Some(exp)) = db.explain_belief(id) {
        println!(
            "EXPLAIN {:?}",
            format!("{exp:?}").chars().take(200).collect::<String>()
        );
    }
}
