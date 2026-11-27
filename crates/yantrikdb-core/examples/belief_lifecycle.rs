//! Native lifecycle of a `CognitiveNode(Belief)` — on a scratch copy only.
//!
//! cargo run --release --example belief_lifecycle -- <db-path> create|reopen
//!
//! create: builds the node, adds confirming and contradicting evidence,
//!         persists after every step and prints the log-odds.
//! reopen: reads the node in a NEW process from the same file, queries it and
//!         asks it to explain itself — the persistence proof.

use std::time::{SystemTime, UNIX_EPOCH};

use yantrikdb::belief::{assert_evidence, BeliefRevisionConfig, Evidence};
use yantrikdb::state::BeliefPayload;
use yantrikdb::state::Provenance;
use yantrikdb::{CognitiveNode, NodeId, NodeKind, NodePayload, YantrikDB};

const PROPOSITION: &str =
    "A2A agreement is a recurring theme: agents coordinate procedure and consent \
     before messages flow.";

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
    let path = std::env::args()
        .nth(1)
        .expect("usage: belief_lifecycle <db> create|reopen");
    let mode = std::env::args().nth(2).unwrap_or_else(|| "create".into());

    let db = YantrikDB::new(&path, 1024).expect("open scratch copy");
    let id = NodeId::new(NodeKind::Belief, 1);

    match mode.as_str() {
        "reopen" => {
            let node = db
                .load_cognitive_node(id)
                .expect("load")
                .expect("node present");
            let p = match &node.payload {
                NodePayload::Belief(b) => b,
                _ => panic!("not a belief"),
            };
            println!(
                "REOPEN present=true proposition={:?}",
                &p.proposition[..40.min(p.proposition.len())]
            );
            println!(
                "REOPEN log_odds={:.4} p_true={:.4} evidence_trail={} propagated={}",
                p.log_odds,
                sigmoid(p.log_odds),
                p.evidence_trail.len(),
                p.propagated.len()
            );
            println!(
                "REOPEN count_beliefs={:?}",
                db.count_cognitive_nodes(Some(NodeKind::Belief)).unwrap()
            );
            if let Ok(Some(exp)) = db.explain_belief(id) {
                println!(
                    "REOPEN explain={:?}",
                    format!("{exp:?}").chars().take(220).collect::<String>()
                );
            } else {
                println!("REOPEN explain=<none>");
            }
        }
        _ => {
            let mut node = CognitiveNode::new(
                id,
                "a2a-agreement".to_string(),
                NodePayload::Belief(BeliefPayload::new(PROPOSITION, "general", 0.0)),
            );
            db.persist_cognitive_node(&node).expect("persist creation");
            println!("CREATE persisted log_odds=0.0000 p_true=0.5000");

            let cfg = BeliefRevisionConfig::default();
            for (weight, source) in [
                (1.4_f64, "20 topic-cluster members, 25 days"),
                (1.1, "second run, unchanged evidence"),
                (-1.3, "contradicting record: A2A switched off"),
            ] {
                let ev = Evidence::new(id, weight, source, Provenance::Inferred, now());
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
