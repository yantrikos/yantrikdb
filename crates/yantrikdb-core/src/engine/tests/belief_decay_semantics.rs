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

// ── Die drei Sequenzen aus dem Maintainer-Review (26.11.2026) ──────────────

/// Ein geklemmter Glaube darf nach dem nächsten Verfall nicht zurückkommen.
///
/// Review: zwei Told +5,0, dann Observed −3,0 (Klemme greift), dann Verfall
/// nach 8 Tagen. Der Anker blieb bei 6,8 stehen, der Verfall baute daraus 6,0
/// statt 3,234 — die weggedeckelte Evidenz kam zurück.
#[test]
fn a_capped_belief_does_not_come_back_after_decay() {
    let cfg = BeliefRevisionConfig::default();
    let now = 1_800_000_000.0_f64;
    let mut node = belief(40, Vec::new(), 0.0);
    let id = node.id;

    for _ in 0..2 {
        assert_evidence(
            &mut node,
            &Evidence {
                target_belief: id,
                weight: 5.0,
                source: "told".into(),
                provenance: Provenance::Told,
                propagate: false,
                timestamp: now,
                observation_id: None,
            },
            &cfg,
        )
        .unwrap();
    }
    let after_support = lo(&node);

    assert_evidence(
        &mut node,
        &Evidence {
            target_belief: id,
            weight: -3.0,
            source: "seen".into(),
            provenance: Provenance::Observed,
            propagate: false,
            timestamp: now,
            observation_id: None,
        },
        &cfg,
    )
    .unwrap();
    let after_contradiction = lo(&node);
    assert!(
        after_contradiction < after_support,
        "the contradiction must lower the belief: {after_support} -> {after_contradiction}"
    );

    revise_beliefs(&mut [&mut node], now + 8.0 * 86400.0, &cfg);
    let after_decay = lo(&node);

    assert!(
        after_decay <= after_contradiction + 1e-9,
        "decay must not raise the belief above its pre-decay value \
         (was {after_contradiction}, became {after_decay}) — that is the cap \
         stored in the anchor, not evidence"
    );
}

/// Ein propagierter Beitrag ohne Origin muss im Anker stehen, sonst löscht der
/// nächste Verfall ihn (Review: 2,9 → 0,882).
#[test]
fn a_propagated_contribution_is_in_the_anchor() {
    let now = 1_800_000_000.0_f64;
    let mut node = belief(41, Vec::new(), 0.9);
    if let NodePayload::Belief(b) = &mut node.payload {
        b.undecayed_log_odds = Some(0.9);
        b.decayed_at = now - 30.0 * 86400.0;
    }

    if let NodePayload::Belief(b) = &mut node.payload {
        b.shift_log_odds(2.0);
    }

    match &node.payload {
        NodePayload::Belief(b) => assert!(
            (b.undecayed_log_odds.unwrap_or(f64::NAN) - 2.9).abs() < 1e-9,
            "the anchor must carry the propagated weight: {:?}",
            b.undecayed_log_odds
        ),
        _ => panic!("expected belief"),
    }
}

/// Warten darf einen Glauben nicht sicherer machen.
///
/// Review: 2,7 verfällt über ein Jahr auf 0,135. Dann +0,45 (→ 0,585). Acht
/// Tage später, ohne neue Evidenz, baute der Patch 3,087 — die alte, längst
/// verfallene Evidenz kam über den Anker zurück.
#[test]
fn waiting_does_not_make_a_decayed_belief_more_confident() {
    let cfg = BeliefRevisionConfig::default();
    let now = 1_800_000_000.0_f64;
    let mut node = belief(42, Vec::new(), 0.135);
    let id = node.id;
    if let NodePayload::Belief(b) = &mut node.payload {
        b.undecayed_log_odds = Some(2.7);
        b.decayed_at = now - 365.0 * 86400.0;
        b.evidence_trail.push(EvidenceEntry {
            source: "alt".into(),
            weight: 2.7,
            timestamp: now - 365.0 * 86400.0,
            observation_id: None,
            history: Vec::new(),
            paths: Vec::new(),
        });
    }

    assert_evidence(
        &mut node,
        &Evidence {
            target_belief: id,
            weight: 0.45,
            source: "neu".into(),
            provenance: Provenance::Observed,
            propagate: false,
            timestamp: now,
            observation_id: None,
        },
        &cfg,
    )
    .unwrap();
    let after_assert = lo(&node);

    revise_beliefs(&mut [&mut node], now + 8.0 * 86400.0, &cfg);
    let after_wait = lo(&node);

    assert!(
        after_wait <= after_assert + 1e-9,
        "eight days without new evidence must not raise the belief \
         ({after_assert} -> {after_wait})"
    );
}

/// Main's rate, unscaled: a default belief at 1.9 decays to 0.76 at +37 days,
/// and `persistence` no longer changes that (0, 0.8 and 1.0 agree).
///
/// The persistence-scaled rate was a fivefold slowdown of every default belief
/// and made persistence 1.0 immortal. It is out of this PR; if it comes back,
/// it is an opt-in config field, not a default.
#[test]
fn default_decay_rate_is_unscaled_by_persistence() {
    let cfg = BeliefRevisionConfig::default();
    let now = 1_800_000_000.0_f64;
    let mut results = Vec::new();

    for (i, persistence) in [0.0, 0.8, 1.0].into_iter().enumerate() {
        // Evidence at `now`, decay pass at +37 days: 30 effective days after
        // the 7-day grace, at the configured rate.
        let trail = vec![EvidenceEntry {
            source: "alt".into(),
            weight: 1.9,
            timestamp: now,
            observation_id: None,
            history: Vec::new(),
            paths: Vec::new(),
        }];
        let mut node = belief(50 + i as u32, trail, 1.9);
        node.attrs.persistence = persistence;
        revise_beliefs(&mut [&mut node], now + 37.0 * 86400.0, &cfg);
        results.push((persistence, lo(&node)));
    }

    for (persistence, value) in &results {
        assert!(
            (value - 0.76).abs() < 1e-9,
            "persistence {persistence}: expected 0.76, got {value}"
        );
    }
    assert!(
        results
            .iter()
            .all(|(_, v)| (v - results[0].1).abs() < 1e-12),
        "the decay rate must not depend on persistence: {results:?}"
    );
}

/// Direct and propagated evidence about the SAME origin must not depend on
/// arrival order.
///
/// Review (26.11.): direct-then-propagated gave 0.9, propagated-then-direct
/// gave 1.9 — `already_direct` was only consulted in the propagation
/// direction, so the second arrival added a second vote.
#[test]
fn direct_and_propagated_evidence_are_order_independent() {
    let cfg = BeliefRevisionConfig::default();
    let now = 1_800_000_000.0_f64;
    let origin = "rid-x";
    let id_direct = NodeId::new(NodeKind::Belief, 60);
    let id_prop = NodeId::new(NodeKind::Belief, 61);
    let via = NodeId::new(NodeKind::Belief, 62);

    // Order A: direct first, then the propagation of the same origin.
    let mut a = belief(60, Vec::new(), 0.0);
    a.payload = NodePayload::Belief(match a.payload {
        NodePayload::Belief(mut b) => {
            b.update_with_id(1.0, 0.9, "obs", now, Some(&format!("{origin}")));
            b
        }
        _ => unreachable!(),
    });
    if let NodePayload::Belief(b) = &mut a.payload {
        b.update_propagated(1.0, "prop", now, &format!("{origin}"), via);
    }
    let a_value = lo(&a);

    // Order B: the propagation arrives first, then the direct observation.
    let mut b = belief(61, Vec::new(), 0.0);
    if let NodePayload::Belief(payload) = &mut b.payload {
        payload.update_propagated(1.0, "prop", now, &format!("{origin}"), via);
        payload.update_with_id(1.0, 0.9, "obs", now, Some(&format!("{origin}")));
    }
    let b_value = lo(&b);

    let _ = (id_direct, id_prop);
    assert!(
        (a_value - b_value).abs() < 1e-9,
        "the result must not depend on arrival order: direct-first {a_value}, propagated-first {b_value}"
    );
}
