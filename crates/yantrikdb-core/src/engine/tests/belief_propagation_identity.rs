use super::*;
use crate::belief::{propagate_evidence_from, BeliefRevisionConfig};
use crate::state::{BeliefPayload, CognitiveEdge, CognitiveEdgeKind};
use crate::{CognitiveNode, NodeId, NodeKind, NodePayload};
use std::collections::HashMap;

fn belief(seq: u32) -> CognitiveNode {
    let id = NodeId::new(NodeKind::Belief, seq);
    CognitiveNode::new(
        id,
        format!("b{seq}"),
        NodePayload::Belief(BeliefPayload {
            proposition: "Probe.".into(),
            log_odds: 0.0,
            domain: "general".into(),
            evidence_trail: Vec::new(),
            user_confirmed: false,
            undecayed_log_odds: None,
        }),
    )
}

fn lo(n: &CognitiveNode) -> f64 {
    match &n.payload { NodePayload::Belief(b) => b.log_odds, _ => panic!() }
}

fn entries(n: &CognitiveNode) -> usize {
    match &n.payload { NodePayload::Belief(b) => b.evidence_trail.len(), _ => 0 }
}

fn edge(src: NodeId, dst: NodeId) -> CognitiveEdge {
    CognitiveEdge {
        src, dst, kind: CognitiveEdgeKind::Supports, weight: 1.0,
        created_at_ms: 1_800_000_000_000, last_confirmed_ms: 1_800_000_000_000,
        observation_count: 1, confidence: 1.0,
    }
}

/// Wiederholte Propagation derselben Ursprungsbeobachtung zählt einmal.
#[test]
fn repeated_propagation_with_origin_is_idempotent() {
    let cfg = BeliefRevisionConfig::default();
    let a = belief(41);
    let d = belief(42);
    let (ai, di) = (a.id, d.id);
    let mut down: HashMap<NodeId, CognitiveNode> = HashMap::new();
    down.insert(di, d);

    let _ = propagate_evidence_from(Some("obs-1"), ai, 1.0, &[edge(ai, di)], &mut down, &cfg, 0);
    let once = lo(&down[&di]);
    let _ = propagate_evidence_from(Some("obs-1"), ai, 1.0, &[edge(ai, di)], &mut down, &cfg, 0);
    let twice = lo(&down[&di]);
    println!("WIEDERHOLT: einmal={once:.4} zweimal={twice:.4} trail={}", entries(&down[&di]));
    assert!((once - twice).abs() < 1e-12, "same origin must not vote twice");
    assert_eq!(entries(&down[&di]), 1);
}

/// Diamond: dieselbe Ursprungsbeobachtung erreicht das Ziel über zwei Beliefs.
/// Sie darf dort EINE Erfahrung bleiben, und der stärkere Pfad gewinnt.
#[test]
fn diamond_counts_the_origin_once_and_keeps_the_stronger_path() {
    let cfg = BeliefRevisionConfig::default();
    let a = belief(43); let b = belief(44); let c = belief(45); let d = belief(46);
    let (ai, bi, ci, di) = (a.id, b.id, c.id, d.id);

    // Hop 1: A -> B (stark), A -> C (schwach)
    let mut hop1: HashMap<NodeId, CognitiveNode> = HashMap::new();
    hop1.insert(bi, b.clone());
    hop1.insert(ci, c);
    let _ = propagate_evidence_from(Some("obs-diamond"), ai, 1.0, &[edge(ai, bi), edge(ai, ci)], &mut hop1, &cfg, 0);
    let w_b = lo(&hop1[&bi]);
    let w_c = lo(&hop1[&ci]);

    // Hop 2: B -> D, C -> D, beide mit demselben Ursprung
    let mut hop2: HashMap<NodeId, CognitiveNode> = HashMap::new();
    hop2.insert(di, d);
    let _ = propagate_evidence_from(Some("obs-diamond"), bi, w_b, &[edge(bi, di)], &mut hop2, &cfg, 1);
    let after_first_path = lo(&hop2[&di]);
    // zweiter Pfad bewusst schwächer: halbes Kantengewicht
    let mut weak = edge(ci, di); weak.weight = 0.5;
    let _ = propagate_evidence_from(Some("obs-diamond"), ci, w_c, &[weak], &mut hop2, &cfg, 1);
    let after_second_path = lo(&hop2[&di]);

    println!("DIAMOND: Pfad B={w_b:.4} Pfad C={w_c:.4} · nach 1. Pfad {after_first_path:.4} · nach 2. Pfad {after_second_path:.4} · trail={}",
             entries(&hop2[&di]));
    assert_eq!(entries(&hop2[&di]), 1, "the diamond must hold one observation, not two");
    assert!((after_second_path - after_first_path).abs() < 1e-12,
            "a weaker second path must not add to the first");
}

/// Rücknahme: die Ursprungsbeobachtung auf 0 zurücknehmen korrigiert die propagierte Wirkung.
#[test]
fn retracting_the_origin_corrects_the_propagated_effect() {
    let cfg = BeliefRevisionConfig::default();
    let a = belief(47); let d = belief(48);
    let (ai, di) = (a.id, d.id);
    let mut down: HashMap<NodeId, CognitiveNode> = HashMap::new();
    down.insert(di, d);

    let _ = propagate_evidence_from(Some("obs-retract"), ai, 1.0, &[edge(ai, di)], &mut down, &cfg, 0);
    let with_evidence = lo(&down[&di]);
    let _ = propagate_evidence_from(Some("obs-retract"), ai, 0.0, &[edge(ai, di)], &mut down, &cfg, 0);
    let after_retraction = lo(&down[&di]);
    println!("RÜCKNAHME: {with_evidence:.4} -> {after_retraction:.4} trail={}", entries(&down[&di]));
    assert!(after_retraction.abs() < 1e-12, "retraction must remove the propagated contribution");
    assert_eq!(entries(&down[&di]), 1, "the entry stays, its weight becomes 0");
}

/// Ohne Ursprung bleibt das alte Verhalten (additiv) — Rückwärtskompatibilität.
#[test]
fn without_origin_propagation_stays_additive() {
    let cfg = BeliefRevisionConfig::default();
    let a = belief(49); let d = belief(50);
    let (ai, di) = (a.id, d.id);
    let mut down: HashMap<NodeId, CognitiveNode> = HashMap::new();
    down.insert(di, d);
    let _ = propagate_evidence_from(None, ai, 1.0, &[edge(ai, di)], &mut down, &cfg, 0);
    let once = lo(&down[&di]);
    let _ = propagate_evidence_from(None, ai, 1.0, &[edge(ai, di)], &mut down, &cfg, 0);
    let twice = lo(&down[&di]);
    println!("OHNE URSPRUNG: einmal={once:.4} zweimal={twice:.4}");
    assert!((twice - 2.0 * once).abs() < 1e-12);
}

fn with_weight(src: NodeId, dst: NodeId, weight: f64) -> CognitiveEdge {
    let mut e = edge(src, dst);
    e.weight = weight;
    e
}

fn contradicting(src: NodeId, dst: NodeId, weight: f64) -> CognitiveEdge {
    let mut e = edge(src, dst);
    e.kind = CognitiveEdgeKind::Contradicts;
    e.weight = weight;
    e
}

/// Der stärkste Pfad wird zurückgenommen — der schwächere gültige bleibt stehen.
///
/// Vor dem Fix lag pro Ursprung EIN Beitrag im Trail: die Rücknahme des starken
/// Pfades löschte auch den schwachen, obwohl der weiter gültig war.
#[test]
fn withdrawing_the_strongest_path_promotes_the_weaker_one() {
    let cfg = BeliefRevisionConfig::default();
    let b = belief(51);
    let c = belief(52);
    let d = belief(53);
    let (bi, ci, di) = (b.id, c.id, d.id);
    let mut down: HashMap<NodeId, CognitiveNode> = HashMap::new();
    down.insert(di, d);

    // zwei Pfade, eine Ursprungsbeobachtung: B trägt 0,5 · C trägt 0,2
    let _ = propagate_evidence_from(Some("obs-zwei-pfade"), bi, 1.0, &[edge(bi, di)], &mut down, &cfg, 0);
    let _ = propagate_evidence_from(Some("obs-zwei-pfade"), ci, 1.0, &[with_weight(ci, di, 0.4)], &mut down, &cfg, 0);
    let both = lo(&down[&di]);

    // stärksten Pfad zurücknehmen
    let _ = propagate_evidence_from(Some("obs-zwei-pfade"), bi, 0.0, &[edge(bi, di)], &mut down, &cfg, 0);
    let after = lo(&down[&di]);

    println!("RÜCKNAHME STARKER PFAD: beide={both:.4} · nach Rücknahme={after:.4} trail={}", entries(&down[&di]));
    assert!((both - 0.5).abs() < 1e-12, "the stronger path must carry the aggregate");
    assert!((after - 0.2).abs() < 1e-12, "the weaker valid path must remain");
    assert_eq!(entries(&down[&di]), 1, "still one observation, not two");
}

/// Die Ursprungsbeobachtung wird schwächer bewertet, ohne zurückgenommen zu werden.
#[test]
fn weakening_the_origin_lowers_the_propagated_effect() {
    let cfg = BeliefRevisionConfig::default();
    let b = belief(54);
    let d = belief(55);
    let (bi, di) = (b.id, d.id);
    let mut down: HashMap<NodeId, CognitiveNode> = HashMap::new();
    down.insert(di, d);

    let _ = propagate_evidence_from(Some("obs-neubewertung"), bi, 1.0, &[edge(bi, di)], &mut down, &cfg, 0);
    let strong = lo(&down[&di]);
    let _ = propagate_evidence_from(Some("obs-neubewertung"), bi, 0.4, &[edge(bi, di)], &mut down, &cfg, 0);
    let weakened = lo(&down[&di]);

    println!("ABSCHWÄCHUNG: {strong:.4} -> {weakened:.4} trail={}", entries(&down[&di]));
    assert!((strong - 0.5).abs() < 1e-12);
    assert!((weakened - 0.2).abs() < 1e-12, "a re-assessed origin must lower the contribution, not keep or erase it");
    assert_eq!(entries(&down[&di]), 1);
}

/// Dieselbe Beobachtung erreicht einen Belief direkt UND über Propagation.
#[test]
fn direct_evidence_outranks_its_own_propagation() {
    let cfg = BeliefRevisionConfig::default();
    let a = belief(56);
    let mut d = belief(57);
    let (ai, di) = (a.id, d.id);

    // direkt: dieselbe Beobachtungsidentität liegt schon am Ziel
    if let NodePayload::Belief(p) = &mut d.payload {
        p.update_with_id(0.3, 1.0, "direkt", 1_800_000_000.0, Some("obs-direkt-und-pfad"));
    }
    let direct = lo(&d);

    let mut down: HashMap<NodeId, CognitiveNode> = HashMap::new();
    down.insert(di, d);
    let _ = propagate_evidence_from(Some("obs-direkt-und-pfad"), ai, 1.0, &[edge(ai, di)], &mut down, &cfg, 0);
    let after = lo(&down[&di]);

    println!("DIREKT + PROPAGIERT: direkt={direct:.4} danach={after:.4} trail={}", entries(&down[&di]));
    assert!((after - direct).abs() < 1e-12, "the same observation must not vote twice");
    assert_eq!(entries(&down[&di]), 1, "only the direct entry speaks");
}

/// Zwei Pfade tragen dieselbe Beobachtung mit ENTGEGENGESETZTER Polarität.
/// Eine Beobachtung hat eine Stimme — die stärkere Aussage — und die Reihenfolge
/// der Pfade darf das Ergebnis nicht ändern.
#[test]
fn opposite_polarity_paths_speak_with_one_voice() {
    let cfg = BeliefRevisionConfig::default();

    let run = |support_first: bool| -> f64 {
        let b = belief(58);
        let c = belief(59);
        let d = belief(60);
        let (bi, ci, di) = (b.id, c.id, d.id);
        let mut down: HashMap<NodeId, CognitiveNode> = HashMap::new();
        down.insert(di, d);
        let supports = edge(bi, di);            // +0,5
        let contradicts = contradicting(ci, di, 1.6); // -0,8
        let order: Vec<(NodeId, CognitiveEdge)> = if support_first {
            vec![(bi, supports), (ci, contradicts)]
        } else {
            vec![(ci, contradicts), (bi, supports)]
        };
        for (src, e) in order {
            let _ = propagate_evidence_from(Some("obs-polaritaet"), src, 1.0, &[e], &mut down, &cfg, 0);
        }
        println!("POLARITÄT (support_first={support_first}): {:.4}", lo(&down[&di]));
        entries(&down[&di]);
        lo(&down[&di])
    };

    let a = run(true);
    let b = run(false);
    println!("POLARITÄT: erst stützend={a:.4} · erst widersprechend={b:.4}");
    assert!((a - b).abs() < 1e-12, "the order of the paths must not matter");
    assert!((a + 0.8).abs() < 1e-12, "the stronger statement carries the aggregate");
}
