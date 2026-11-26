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
            decayed_at: 0.0,
            propagated: Vec::new(),
        }),
    )
}

fn lo(n: &CognitiveNode) -> f64 {
    match &n.payload {
        NodePayload::Belief(b) => b.log_odds,
        _ => panic!(),
    }
}

fn propagated(n: &CognitiveNode) -> usize {
    match &n.payload {
        NodePayload::Belief(b) => b.propagated.len(),
        _ => 0,
    }
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

/// Repeated propagation of the same originating observation counts once.
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
    println!(
        "REPEATED: once={once:.4} twice={twice:.4} propagated={}",
        propagated(&down[&di])
    );
    assert!(
        (once - twice).abs() < 1e-12,
        "same origin must not vote twice"
    );
    assert_eq!(propagated(&down[&di]), 1);
}

/// Diamond: the same originating observation reaches the target over two
/// beliefs. It stays ONE experience there, and the stronger path wins.
#[test]
fn diamond_counts_the_origin_once_and_keeps_the_stronger_path() {
    let cfg = BeliefRevisionConfig::default();
    let a = belief(43);
    let b = belief(44);
    let c = belief(45);
    let d = belief(46);
    let (ai, bi, ci, di) = (a.id, b.id, c.id, d.id);

    // Hop 1: A -> B (strong), A -> C (weak)
    let mut hop1: HashMap<NodeId, CognitiveNode> = HashMap::new();
    hop1.insert(bi, b.clone());
    hop1.insert(ci, c);
    let _ = propagate_evidence_from(
        Some("obs-diamond"),
        ai,
        1.0,
        &[edge(ai, bi), edge(ai, ci)],
        &mut hop1,
        &cfg,
        0,
    );
    let w_b = lo(&hop1[&bi]);
    let w_c = lo(&hop1[&ci]);

    // Hop 2: B -> D, C -> D, both with the same origin
    let mut hop2: HashMap<NodeId, CognitiveNode> = HashMap::new();
    hop2.insert(di, d);
    let _ = propagate_evidence_from(
        Some("obs-diamond"),
        bi,
        w_b,
        &[edge(bi, di)],
        &mut hop2,
        &cfg,
        1,
    );
    let after_first_path = lo(&hop2[&di]);
    // deliberately weaker second path: half the edge weight
    let mut weak = edge(ci, di);
    weak.weight = 0.5;
    let _ = propagate_evidence_from(Some("obs-diamond"), ci, w_c, &[weak], &mut hop2, &cfg, 1);
    let after_second_path = lo(&hop2[&di]);

    println!("DIAMOND: Pfad B={w_b:.4} Pfad C={w_c:.4} · nach 1. Pfad {after_first_path:.4} · nach 2. Pfad {after_second_path:.4} · trail={}",
             propagated(&hop2[&di]));
    assert_eq!(
        propagated(&hop2[&di]),
        1,
        "the diamond must hold one observation, not two"
    );
    assert!(
        (after_second_path - after_first_path).abs() < 1e-12,
        "a weaker second path must not add to the first"
    );
}

/// Retraction: taking the originating observation back to 0 corrects the
/// propagated effect.
#[test]
fn retracting_the_origin_corrects_the_propagated_effect() {
    let cfg = BeliefRevisionConfig::default();
    let a = belief(47);
    let d = belief(48);
    let (ai, di) = (a.id, d.id);
    let mut down: HashMap<NodeId, CognitiveNode> = HashMap::new();
    down.insert(di, d);

    let _ = propagate_evidence_from(
        Some("obs-retract"),
        ai,
        1.0,
        &[edge(ai, di)],
        &mut down,
        &cfg,
        0,
    );
    let with_evidence = lo(&down[&di]);
    let _ = propagate_evidence_from(
        Some("obs-retract"),
        ai,
        0.0,
        &[edge(ai, di)],
        &mut down,
        &cfg,
        0,
    );
    let after_retraction = lo(&down[&di]);
    println!(
        "RETRACTION: {with_evidence:.4} -> {after_retraction:.4} propagated={}",
        propagated(&down[&di])
    );
    assert!(
        after_retraction.abs() < 1e-12,
        "retraction must remove the propagated contribution"
    );
    assert_eq!(
        propagated(&down[&di]),
        1,
        "the entry stays, its weight becomes 0"
    );
}

/// Without an origin the old additive behaviour is kept, for compatibility.
#[test]
fn without_origin_propagation_stays_additive() {
    let cfg = BeliefRevisionConfig::default();
    let a = belief(49);
    let d = belief(50);
    let (ai, di) = (a.id, d.id);
    let mut down: HashMap<NodeId, CognitiveNode> = HashMap::new();
    down.insert(di, d);
    let _ = propagate_evidence_from(None, ai, 1.0, &[edge(ai, di)], &mut down, &cfg, 0);
    let once = lo(&down[&di]);
    let _ = propagate_evidence_from(None, ai, 1.0, &[edge(ai, di)], &mut down, &cfg, 0);
    let twice = lo(&down[&di]);
    println!("WITHOUT ORIGIN: once={once:.4} twice={twice:.4}");
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

/// The strongest path is withdrawn — the weaker valid one stays standing.
///
/// Before the fix one contribution per origin sat in the trail: withdrawing the
/// strong path also erased the weak one, although it was still valid.
#[test]
fn withdrawing_the_strongest_path_promotes_the_weaker_one() {
    let cfg = BeliefRevisionConfig::default();
    let b = belief(51);
    let c = belief(52);
    let d = belief(53);
    let (bi, ci, di) = (b.id, c.id, d.id);
    let mut down: HashMap<NodeId, CognitiveNode> = HashMap::new();
    down.insert(di, d);

    // two paths, one originating observation: B carries 0.5, C carries 0.2
    let _ = propagate_evidence_from(
        Some("obs-zwei-pfade"),
        bi,
        1.0,
        &[edge(bi, di)],
        &mut down,
        &cfg,
        0,
    );
    let _ = propagate_evidence_from(
        Some("obs-zwei-pfade"),
        ci,
        1.0,
        &[with_weight(ci, di, 0.4)],
        &mut down,
        &cfg,
        0,
    );
    let both = lo(&down[&di]);

    // withdraw the strongest path
    let _ = propagate_evidence_from(
        Some("obs-zwei-pfade"),
        bi,
        0.0,
        &[edge(bi, di)],
        &mut down,
        &cfg,
        0,
    );
    let after = lo(&down[&di]);

    println!(
        "WITHDRAW STRONG PATH: both={both:.4} · after withdrawal={after:.4} trail={}",
        propagated(&down[&di])
    );
    assert!(
        (both - 0.5).abs() < 1e-12,
        "the stronger path must carry the aggregate"
    );
    assert!(
        (after - 0.2).abs() < 1e-12,
        "the weaker valid path must remain"
    );
    assert_eq!(propagated(&down[&di]), 1, "still one observation, not two");
}

/// The originating observation is re-assessed weaker without being retracted.
#[test]
fn weakening_the_origin_lowers_the_propagated_effect() {
    let cfg = BeliefRevisionConfig::default();
    let b = belief(54);
    let d = belief(55);
    let (bi, di) = (b.id, d.id);
    let mut down: HashMap<NodeId, CognitiveNode> = HashMap::new();
    down.insert(di, d);

    let _ = propagate_evidence_from(
        Some("obs-neubewertung"),
        bi,
        1.0,
        &[edge(bi, di)],
        &mut down,
        &cfg,
        0,
    );
    let strong = lo(&down[&di]);
    let _ = propagate_evidence_from(
        Some("obs-neubewertung"),
        bi,
        0.4,
        &[edge(bi, di)],
        &mut down,
        &cfg,
        0,
    );
    let weakened = lo(&down[&di]);

    println!(
        "ABSCHWÄCHUNG: {strong:.4} -> {weakened:.4} trail={}",
        propagated(&down[&di])
    );
    assert!((strong - 0.5).abs() < 1e-12);
    assert!(
        (weakened - 0.2).abs() < 1e-12,
        "a re-assessed origin must lower the contribution, not keep or erase it"
    );
    assert_eq!(propagated(&down[&di]), 1);
}

/// Direct and propagated evidence from the same origin are two channels.
///
/// `observation_key` scopes ids per belief, so an id used as documented can
/// never match across beliefs — the old "direct outranks its own propagation"
/// rule could therefore not fire. With propagation in its own channel, each
/// side is deduplicated within itself and the arrival order does not matter:
/// both orders give main's 1.9 (0.9 direct + 1.0 propagated).
#[test]
fn direct_and_propagated_from_one_origin_are_separate_channels() {
    let cfg = BeliefRevisionConfig::default();

    let run = |direct_first: bool| -> f64 {
        let a = belief(56);
        let mut d = belief(57);
        let (ai, di) = (a.id, d.id);

        let apply_direct = |node: &mut CognitiveNode| {
            if let NodePayload::Belief(p) = &mut node.payload {
                p.update_with_id(1.0, 0.9, "source", 1_800_000_000.0, Some("obs-same"), 1.0);
            }
        };
        let apply_propagation = |node: &mut CognitiveNode| {
            let mut down: HashMap<NodeId, CognitiveNode> = HashMap::new();
            down.insert(di, node.clone());
            let _ = propagate_evidence_from(
                Some("obs-same"),
                ai,
                2.0,
                &[edge(ai, di)],
                &mut down,
                &cfg,
                0,
            );
            *node = down.remove(&di).expect("downstream belief");
        };

        if direct_first {
            apply_direct(&mut d);
            apply_propagation(&mut d);
        } else {
            apply_propagation(&mut d);
            apply_direct(&mut d);
        }
        lo(&d)
    };

    let direct_first = run(true);
    let propagation_first = run(false);
    println!("ORDER: direct-first={direct_first:.4} propagation-first={propagation_first:.4}");
    assert!(
        (direct_first - 1.9).abs() < 1e-12,
        "direct 0.9 + propagated 1.0 must be 1.9, got {direct_first}"
    );
    assert!(
        (propagation_first - 1.9).abs() < 1e-12,
        "the arrival order must not change the result, got {propagation_first}"
    );
}

/// A caller id shaped like the internal propagation key must not collide.
///
/// Review (26.11.): a direct id `prop|o1` was overwritten by a propagation from
/// origin `o1` (0.9 -> 0.2). Propagation no longer shares the trail's key
/// space, so both channels count and neither overwrites the other.
#[test]
fn a_caller_id_shaped_like_a_propagation_key_does_not_collide() {
    let cfg = BeliefRevisionConfig::default();
    let a = belief(70);
    let mut d = belief(71);
    let (ai, di) = (a.id, d.id);

    if let NodePayload::Belief(p) = &mut d.payload {
        p.update_with_id(1.0, 0.9, "caller", 1_800_000_000.0, Some("prop|o1"), 1.0);
    }
    let direct_only = lo(&d);

    let mut down: HashMap<NodeId, CognitiveNode> = HashMap::new();
    down.insert(di, d);
    let effects = propagate_evidence_from(Some("o1"), ai, 2.0, &[edge(ai, di)], &mut down, &cfg, 0);
    let both = lo(&down[&di]);

    println!("COLLISION PROBE: direct-only {direct_only:.4} · both {both:.4}");
    assert_eq!(effects.len(), 1, "the propagation must be applied");
    assert!((direct_only - 0.9).abs() < 1e-12);
    assert!(
        (both - 1.9).abs() < 1e-12,
        "both channels count, neither overwrites the other: {both}"
    );
    match &down[&di].payload {
        NodePayload::Belief(b) => {
            assert_eq!(b.evidence_trail.len(), 1, "the caller's entry survives");
            assert_eq!(b.evidence_trail[0].weight, 0.9);
            assert_eq!(b.propagated.len(), 1);
            assert_eq!(b.propagated[0].origin, "o1");
        }
        _ => panic!("expected belief"),
    }
}

/// Two paths carry the same observation with OPPOSITE polarity. One observation
/// has one voice — the stronger statement — and the order of the paths must not
/// change the result.
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
        let supports = edge(bi, di); // +0,5
        let contradicts = contradicting(ci, di, 1.6); // -0,8
        let order: Vec<(NodeId, CognitiveEdge)> = if support_first {
            vec![(bi, supports), (ci, contradicts)]
        } else {
            vec![(ci, contradicts), (bi, supports)]
        };
        for (src, e) in order {
            let _ =
                propagate_evidence_from(Some("obs-polaritaet"), src, 1.0, &[e], &mut down, &cfg, 0);
        }
        println!(
            "POLARITÄT (support_first={support_first}): {:.4}",
            lo(&down[&di])
        );
        propagated(&down[&di]);
        lo(&down[&di])
    };

    let a = run(true);
    let b = run(false);
    println!("POLARITY: support first={a:.4} · contradict first={b:.4}");
    assert!(
        (a - b).abs() < 1e-12,
        "the order of the paths must not matter"
    );
    assert!(
        (a + 0.8).abs() < 1e-12,
        "the stronger statement carries the aggregate"
    );
}
