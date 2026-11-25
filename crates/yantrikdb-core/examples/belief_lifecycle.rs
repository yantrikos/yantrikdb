//! Native lifecycle of a `CognitiveNode(Belief)` — on a scratch copy only.
//!
//! cargo run --release --example belief_lifecycle -- <db-path> create|reopen
//!
//! create: erzeugt den Knoten, ordnet bestätigende und widersprechende Evidenz
//!         zu, persistiert nach jedem Schritt und gibt die Log-Odds aus.
//! reopen: liest den Knoten in einem NEUEN Prozess aus derselben Datei,
//!         fragt ihn ab und lässt ihn erklären — der Persistenzbeweis.

use std::time::{SystemTime, UNIX_EPOCH};

use yantrikdb::belief::{assert_evidence, BeliefRevisionConfig, Evidence};
use yantrikdb::state::BeliefPayload;
use yantrikdb::state::Provenance;
use yantrikdb::{CognitiveNode, NodeId, NodeKind, NodePayload, YantrikDB};

const PROPOSITION: &str =
    "A2A-Zustimmung ist ein wiederkehrendes Thema: Agenten stimmen Ablauf und Zustimmung ab, \
     bevor Nachrichten fliessen.";

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

fn sigmoid(l: f64) -> f64 {
    1.0 / (1.0 + (-l).exp())
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: belief_lifecycle <db> create|reopen");
    let mode = std::env::args().nth(2).unwrap_or_else(|| "create".into());

    let db = YantrikDB::new(&path, 1024).expect("open scratch copy");
    let id = NodeId::new(NodeKind::Belief, 1);

    match mode.as_str() {
        "reopen" => {
            let node = db.load_cognitive_node(id).expect("load").expect("node present");
            let p = match &node.payload {
                NodePayload::Belief(b) => b,
                _ => panic!("not a belief"),
            };
            println!("REOPEN present=true proposition={:?}", &p.proposition[..40.min(p.proposition.len())]);
            println!("REOPEN log_odds={:.4} p_true={:.4} evidence_trail={}", p.log_odds, sigmoid(p.log_odds), p.evidence_trail.len());
            println!("REOPEN count_beliefs={:?}", db.count_cognitive_nodes(Some(NodeKind::Belief)).unwrap());
            if let Ok(Some(exp)) = db.explain_belief(id) {
                println!("REOPEN explain={:?}", format!("{exp:?}").chars().take(220).collect::<String>());
            } else {
                println!("REOPEN explain=<none>");
            }
        }
        _ => {
            let mut node = CognitiveNode::new(
                id,
                "a2a-zustimmung".to_string(),
                NodePayload::Belief(BeliefPayload {
                    proposition: PROPOSITION.to_string(),
                    log_odds: 0.0,
                    domain: "general".to_string(),
                    evidence_trail: Vec::new(),
                    user_confirmed: false,
                }),
            );
            db.persist_cognitive_node(&node).expect("persist creation");
            println!("CREATE persisted log_odds=0.0000 p_true=0.5000");

            let cfg = BeliefRevisionConfig::default();
            for (weight, source) in [(1.4_f64, "20 Topic-Cluster-Mitglieder, 25 Tage"),
                                     (1.1, "zweiter Lauf, unveraenderte Evidenz"),
                                     (-1.3, "widersprechender Record: A2A abgeschaltet")] {
                let ev = Evidence {
                    target_belief: id,
                    weight,
                    source: source.to_string(),
                    provenance: Provenance::Inferred,
                    propagate: false,
                    timestamp: now(),
                    observation_id: None,
                };
                let before = match &node.payload {
                    NodePayload::Belief(b) => b.log_odds,
                    _ => 0.0,
                };
                match assert_evidence(&mut node, &ev, &cfg) {
                    Some(_) => {
                        let after = match &node.payload {
                            NodePayload::Belief(b) => b.log_odds,
                            _ => 0.0,
                        };
                        db.persist_cognitive_node(&node).expect("persist evidence");
                        println!("EVIDENCE weight={weight:+.1} log_odds {before:.4} -> {after:.4} p_true={:.4}", sigmoid(after));
                    }
                    None => println!("EVIDENCE weight={weight:+.1} REJECTED"),
                }
            }
        }
    }
}
