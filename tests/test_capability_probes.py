"""Capability probes — mechanical, judge-free contracts for the memory engine.

Companion to ``tests/capability_audit.py`` (the five original probes), grown
into a standing suite as agreed on 2026-08-17. Every probe builds a fresh
store where every fact is controlled, asks a question with exactly ONE
defensible answer, and checks retrieval mechanically. A probe either
documents a capability or hands us a defect; there is no LLM judge anywhere.

Two engine facts every probe respects (both documented in the engine):

* ``record()`` enqueues entity/claim extraction to the background
  materializer (#95). ``think()`` drains that backlog deterministically, so
  any assertion about entities, threads or conflicts calls ``think()`` first
  instead of sleeping.
* ``recall(skip_reinforce=True)`` keeps a probe from mutating what it
  measures (the top_k-monotonicity "bug" of 2026-08-17 was a probe artifact).
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

import pytest

from yantrikdb import YantrikDB


def ts(day: str) -> float:
    return datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()


@pytest.fixture
def db():
    store = YantrikDB.with_default(os.path.join(tempfile.mkdtemp(), "probe.db"))
    yield store
    store.close()


def texts(hits):
    return [h["text"] for h in hits]


def rids(hits):
    return [h["rid"] for h in hits]


# ── currency / succession ───────────────────────────────────────────────


def test_currency_survives_mmr_engagement(db):
    """Three revisions of one value plus enough same-topic distractors that
    the pool crosses ``min_pool_for_mmr = max(3*top_k, 20)``. MMR treats a
    succession chain as near-duplicates; the CURRENT value must still be
    the first hit (08-17 follow-up B, measured closed on 0.18.0)."""
    db.record("The memory server runs yantrikdb 0.15.1.", created_at=ts("2026-08-10"))
    db.record("The memory server runs yantrikdb 0.15.2.", created_at=ts("2026-08-17"))
    current = db.record("The memory server runs yantrikdb 0.18.0.", created_at=ts("2026-09-01"))
    for i in range(40):
        db.record(
            f"Ops note {i}: the backup job for the memory server ran and rotated logs.",
            created_at=ts("2026-08-20"),
        )
    hits = db.recall(
        query="which yantrikdb version does the memory server run", top_k=5, skip_reinforce=True
    )
    assert rids(hits)[0] == current, texts(hits)


def test_paraphrased_successor_outranks_older_phrasing(db):
    """The newer fact is worded differently from the old one. Currency must
    not depend on the successor copying the predecessor's phrasing."""
    db.record("The memory server is hosted on CT128 on node4.", created_at=ts("2026-07-15"))
    current = db.record(
        "We migrated the memory server; it now lives on container CT130 (node5).",
        created_at=ts("2026-08-30"),
    )
    for i in range(25):
        db.record(f"Ops note {i}: node4 backups completed, disk at 40 percent.", created_at=ts("2026-08-01"))
    hits = db.recall(query="which container hosts the memory server", top_k=5, skip_reinforce=True)
    assert rids(hits)[0] == current, texts(hits)


# ── forgetting ──────────────────────────────────────────────────────────


def test_forget_is_complete_across_every_read_surface(db):
    gone = db.record("Patrick reviewed the grant proposal draft on Monday.", created_at=ts("2026-03-02"))
    kept = db.record("Patrick sent feedback on the grant budget.", created_at=ts("2026-03-09"))
    db.think()  # drain extraction so the entity thread exists before the forget
    assert db.recall_thread("default", ["Patrick"], 10)["total"] == 2

    assert db.forget(gone) is True

    assert gone not in rids(db.recall(query="grant proposal draft review", top_k=5, skip_reinforce=True))
    assert gone not in rids(
        db.recall(
            query="grant proposal",
            top_k=5,
            skip_reinforce=True,
            event_after=ts("2026-03-01"),
            event_before=ts("2026-03-05"),
        )
    )
    assert gone not in rids(db.recall_as_of(ts("2026-03-05"), query="grant proposal", top_k=5))
    thread = db.recall_thread("default", ["Patrick"], 10)
    assert [i["rid"] for i in thread["items"]] == [kept]
    # get() keeps the tombstone visible by design — it must say so.
    assert db.get(gone)["consolidation_status"] == "tombstoned"


# ── namespaces ──────────────────────────────────────────────────────────


def test_namespace_filter_isolates_tenants(db):
    a = db.record("The API key rotation happens every 30 days.", namespace="tenant_a")
    b = db.record("The API key rotation happens every 30 days.", namespace="tenant_b")
    got = rids(db.recall(query="how often are API keys rotated", namespace="tenant_a", top_k=5, skip_reinforce=True))
    assert got == [a], got
    assert b not in got


def test_namespace_none_is_a_wildcard_not_the_default_namespace(db):
    """Contract worth pinning: ``namespace=None`` reads across ALL
    namespaces; ``namespace="default"`` reads only the default one. A
    caller that omits the namespace on a multi-tenant store sees every
    tenant — that is the documented meaning, not a leak, but it must stay
    explicit."""
    a = db.record("The API key rotation happens every 30 days.", namespace="tenant_a")
    b = db.record("The API key rotation happens every 30 days.", namespace="tenant_b")
    everything = rids(db.recall(query="how often are API keys rotated", namespace=None, top_k=5, skip_reinforce=True))
    assert set(everything) == {a, b}
    assert db.recall(query="how often are API keys rotated", namespace="default", top_k=5, skip_reinforce=True) == []


# ── entity threads / extraction ─────────────────────────────────────────


def test_entity_thread_is_deterministic_after_think(db):
    first = db.record("Patrick reviewed the grant proposal draft on Monday.", created_at=ts("2026-03-02"))
    second = db.record("Patrick sent feedback on the grant budget.", created_at=ts("2026-03-09"))
    db.think()
    assert [e["name"] for e in db.search_entities("Patrick")] == ["Patrick"]
    thread = db.recall_thread("default", ["Patrick"], 10)
    assert thread["total"] == 2 and thread["omitted"] == 0
    assert [i["rid"] for i in thread["items"]] == [first, second], "threads read in event order"


def test_correct_revises_in_place_and_chain_head_follows(db):
    rid = db.record("The memory server runs yantrikdb 0.15.2.", created_at=ts("2026-08-17"))
    out = db.correct(rid, reason="upgraded", new_text="The memory server runs yantrikdb 0.18.0.")
    assert out["original_rid"] == rid
    assert texts(db.recall(query="which yantrikdb version does the memory server run", top_k=5, skip_reinforce=True)) == [
        "The memory server runs yantrikdb 0.18.0."
    ]
    assert db.chain_head("default")["text"] == "The memory server runs yantrikdb 0.18.0."
    assert db.history(rid)[0]["prior_text"] == "The memory server runs yantrikdb 0.15.2."


# ── contradiction surfacing ────────────────────────────────────────────


def _conflict_between(db, a, b):
    return [
        c
        for c in db.get_conflicts()
        if {c["memory_a"], c["memory_b"]} == {a, b}
    ]


def test_immutable_identity_contradiction_surfaces_after_think(db):
    a = db.record("Pranab was born in 1985.")
    db.think()
    b = db.record("Pranab was born in 1990.")
    db.think()
    found = _conflict_between(db, a, b)
    assert found and found[0]["conflict_type"] == "identity_fact", db.get_conflicts()
    flagged = db.recall(query="when was Pranab born", top_k=2, skip_reinforce=True)
    assert all(any("conflict" in w for w in h["why_retrieved"]) for h in flagged), flagged


def test_place_contradiction_surfaces_after_think(db):
    """`lives_in` is on the identity whitelist and the functional list, but
    no extractor template produced it until the anchored place patterns
    landed — so a changed city was silently two active facts."""
    a = db.record("Pranab lives in Berlin.")
    db.think()
    b = db.record("Pranab lives in Munich.")
    db.think()
    edges = {(e["rel_type"], e["dst"]) for e in db.get_edges("Pranab")}
    assert {("lives_in", "Berlin"), ("lives_in", "Munich")} <= edges, edges
    assert _conflict_between(db, a, b), db.get_conflicts()


def test_spouse_contradiction_surfaces_after_think(db):
    a = db.record("Pranab is married to Maria.")
    db.think()
    b = db.record("Pranab is married to Sofia.")
    db.think()
    found = _conflict_between(db, a, b)
    assert found and found[0]["conflict_type"] == "identity_fact", db.get_conflicts()


def test_same_fact_reworded_is_not_a_contradiction(db):
    a = db.record("Pranab lives in Berlin.")
    db.think()
    b = db.record("Pranab is living in Berlin.")
    db.think()
    assert not _conflict_between(db, a, b), db.get_conflicts()


def test_headquartered_does_not_mint_a_reverse_leads_edge(db):
    db.record("Fennwick Labs is headquartered in Berlin.")
    db.think()
    rels = {(e["src"], e["rel_type"], e["dst"]) for e in db.get_edges("Fennwick Labs")}
    assert rels == {("Fennwick Labs", "headquartered_in", "Berlin")}, rels


# ── graph expansion ────────────────────────────────────────────────────


@pytest.mark.parametrize("expand", [False, True])
def test_claim_chain_reaches_the_second_hop(db, expand):
    """Measured on 0.18.0: expand_entities boosted records similarity had
    already found (graph_proximity 0.25) but the one-hop record ranked 16th
    of 20 behind unrelated notes. The claims lane now follows the chain one
    hop and admits the provenance record with the path spelled out."""
    db.record("Alice Moreau works at Fennwick Labs as a data engineer.")
    hop = db.record("Fennwick Labs is headquartered in Berlin.")
    for i in range(15):
        db.record(f"Random note {i}: the weather in Lisbon was mild and the cafe served pastel de nata.")
    db.think({"run_consolidation": False})
    hits = db.recall(query="Which city does Alice Moreau work in?", top_k=5, expand_entities=expand, skip_reinforce=True)
    assert hop in rids(hits), texts(hits)
    why = next(h for h in hits if h["rid"] == hop)["why_retrieved"]
    assert any(
        "Alice Moreau -works_at-> Fennwick Labs ; Fennwick Labs -headquartered_in-> Berlin (path via Fennwick Labs, anchor Alice Moreau)" in w
        for w in why
    ), why
    assert "keyword_reserved" in why, why


def test_claim_chain_does_not_displace_direct_hits(db):
    """A path is admitted at the bottom of the band, never over a direct hit."""
    direct = db.record("Alice Moreau works at Fennwick Labs as a data engineer.")
    hop = db.record("Fennwick Labs is headquartered in Berlin.")
    db.record("Alice Moreau leads the data platform team at Fennwick Labs.")
    for i in range(15):
        db.record(f"Random note {i}: the weather in Lisbon was mild and the cafe served pastel de nata.")
    db.think({"run_consolidation": False})
    hits = rids(db.recall(query="What does Alice Moreau do at work?", top_k=5, skip_reinforce=True))
    assert hits.index(direct) < hits.index(hop)


# ── cooperative claims: the writer states, the engine grounds ───────────


def test_stated_claims_are_grounded_and_rejections_carry_reasons(db):
    rid = db.record("Pranab prefers Vim for editing Rust.")
    report = db.attach_claims(
        rid,
        [
            {"subject": "Pranab", "relation": "prefers", "object": "Vim"},
            {"subject": "Pranab", "relation": "prefers", "object": "Emacs"},
        ],
    )
    assert [(a["src"], a["rel_type"], a["dst"]) for a in report["accepted"]] == [("Pranab", "prefers", "Vim")]
    assert len(report["rejected"]) == 1 and "Emacs" in report["rejected"][0]["reason"]
    # No drain needed: the stated claim links its entities synchronously.
    assert db.recall_thread("default", ["Pranab"], 10)["total"] == 1


def test_stated_claims_feed_the_claims_lane(db):
    rid = db.record("Pranab prefers Vim for editing Rust.")
    db.attach_claims(rid, [{"src": "Pranab", "rel_type": "prefers", "dst": "Vim"}])
    for i in range(15):
        db.record(f"Random note {i}: the weather in Lisbon was mild and the cafe served pastel de nata.")
    hits = db.recall(query="Which editor does Pranab prefer?", top_k=3, skip_reinforce=True)
    why = next(h for h in hits if h["rid"] == rid)["why_retrieved"]
    assert any(w.startswith("claims_match: Pranab -prefers-> Vim") for w in why), why


def test_stated_preference_contradiction_surfaces(db):
    """`prefers` is on the preference whitelist but no extractor template
    mints it (and must not: it is multi-valued across domains). A writer
    that STATES both preferences gets the contradiction surfaced."""
    a = db.record("Pranab prefers Vim as his editor.")
    db.attach_claims(a, [{"subject": "Pranab", "relation": "prefers", "object": "Vim"}])
    db.think()
    b = db.record("Pranab prefers Emacs as his editor now.")
    db.attach_claims(b, [{"subject": "Pranab", "relation": "prefers", "object": "Emacs"}])
    db.think()
    found = _conflict_between(db, a, b)
    assert found and found[0]["conflict_type"] == "preference", db.get_conflicts()


# ── self-mined templates: stated claims teach the extractor ──────────────


def test_two_stated_pairs_teach_a_template_that_extracts_from_plain_writes(db):
    a = db.record("Dana mentors Priya on the data platform.")
    db.attach_claims(a, [{"subject": "Dana", "relation": "mentors", "object": "Priya"}])
    assert db.learned_relation_patterns()[0]["active"] is False
    b = db.record("Kim mentors Alex during onboarding.")
    db.attach_claims(b, [{"subject": "Kim", "relation": "mentors", "object": "Alex"}])
    patterns = db.learned_relation_patterns()
    assert patterns[0]["phrase"] == "mentors" and patterns[0]["active"] is True, patterns

    db.record("Sam mentors Jordan on release engineering.")  # plain write, no claim
    db.think()
    assert ("Sam", "mentors", "Jordan") in {(e["src"], e["rel_type"], e["dst"]) for e in db.get_edges("Sam")}
    assert db.forget_learned_relation_patterns() == 1
    assert db.learned_relation_patterns() == []


# ── re-extraction heal ───────────────────────────────────────────────────


def test_reextract_claims_replaces_legacy_junk_and_keeps_assertions(db):
    a = db.record("Alice Moreau works at Fennwick Labs as a data engineer.")
    b = db.record("Pranab confirmed the Materializer runs the loop every tick at UTC midnight.")
    db.think()
    db.relate("Pranab", "Acme", "works_at")
    db.attach_claims(a, [{"subject": "Alice Moreau", "relation": "mentors", "object": "Fennwick Labs"}])
    before = db.reextract_claims(dry_run=True)
    assert before["claims_removed"] == 0 and before["memories_scanned"] == 2
    report = db.reextract_claims()
    assert report["memories_scanned"] == 2 and report["claims_written"] >= 1
    edges = {(e["src"], e["rel_type"], e["dst"]) for e in db.get_edges("Alice Moreau")}
    assert ("Alice Moreau", "works_at", "Fennwick Labs") in edges and ("Alice Moreau", "mentors", "Fennwick Labs") in edges
    assert ("Pranab", "works_at", "Acme") in {(e["src"], e["rel_type"], e["dst"]) for e in db.get_edges("Pranab")}


# ── entity admission (issue #213) ────────────────────────────────────────


def test_values_and_headings_are_never_entities_but_values_still_serve_claims(db):
    """The measured junk classes at write time: a bare version, a year and a
    shouted heading must not become graph nodes, while `runs 0.19.0` still
    mints its claim with the value as the object."""
    rid = db.record("STRATEGIC POINT: CT128 runs 0.19.0 since 2026. Alice Moreau was born in 1985.")
    db.think()
    names = {e["name"] for e in db.list_entities(limit=100)} if hasattr(db, "list_entities") else None
    if names is not None:
        for bad in ("STRATEGIC POINT", "0.19.0", "2026", "1985"):
            assert bad not in names, names
        assert "CT128" in names, names
    edges = {(e["src"], e["rel_type"], e["dst"]) for e in db.get_edges("CT128")}
    assert ("CT128", "runs", "0.19.0") in edges, edges
    edges = {(e["src"], e["rel_type"], e["dst"]) for e in db.get_edges("Alice Moreau")}
    assert ("Alice Moreau", "born_in", "1985") in edges, edges
    rep = db.attach_claims(rid, [{"subject": "CT128", "relation": "runs", "object": "0.19.0"}])
    assert len(rep["accepted"]) == 1, rep


def test_reextract_entities_drops_legacy_junk_nodes_and_keeps_asserted_ones(db):
    """A store written by an older extractor keeps its headings and numbers as
    nodes; the heal removes them (links and heuristic claims included) and
    keeps any node a writer asserted a claim on."""
    a = db.record("Alice Moreau works at Fennwick Labs in Berlin.")
    db.think()
    db.relate("Alice Moreau", "ACME HOLDINGS INTERNATIONAL", "works_at")
    dry = db.reextract_entities(dry_run=True)
    assert dry["dry_run"] is True and dry["entities_removed"] == 0
    assert dry["kept_by_claims"] >= 1, dry
    report = db.reextract_entities()
    assert report["entities_removed"] == report["inadmissible"] - report["kept_by_claims"], report
    edges = {(e["src"], e["rel_type"], e["dst"]) for e in db.get_edges("Alice Moreau")}
    assert ("Alice Moreau", "works_at", "Fennwick Labs") in edges, edges
    assert ("Alice Moreau", "works_at", "ACME HOLDINGS INTERNATIONAL") in edges, edges
    again = db.reextract_entities()
    assert again["entities_removed"] == 0 and again["claims_removed"] == 0
