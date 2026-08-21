from __future__ import annotations

import pytest

from yantrikdb import YantrikDB
from yantrikdb.organize import (
    _requested_item_count,
    ConcernItem,
    ConcernPlan,
    OrganizationPlan,
    TopicHandle,
    assign_evidence_to_handles,
    organize_evidence,
    organize_concerns,
    persist_concerns,
    persist_organization,
    recall_organized,
    validate_topic_handles,
    validate_concern_items,
)


class FixedEmbedDB:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return self.vectors[text]


class PersistDB:
    def __init__(self):
        self.memories = {
            "r1": {"created_at": 20.0, "metadata": {}},
            "r2": {
                "created_at": 30.0,
                "metadata": {
                    "first_mention_at": 10.0,
                    "evidence_span_end_at": 15.0,
                    "first_mention_turn": 10,
                },
            },
        }

    def get(self, rid):
        return self.memories.get(rid)

    def embed(self, text):
        if text.startswith("Topic trajectory:"):
            return [1.0, 0.0]
        if text == "The assembled concern":
            return [0.0, 1.0]
        raise AssertionError(f"unexpected embedding text: {text}")


class RecallDB:
    def __init__(self, hits, memories=None):
        self.hits = hits
        self.memories = memories or {}
        self.calls = []

    def recall(self, **kwargs):
        self.calls.append(kwargs)
        return self.hits[: kwargs["top_k"]]

    def get(self, rid):
        return self.memories.get(rid)


def _recall_hit(rid, score, *, metadata=None, created_at=0.0):
    return {
        "rid": rid,
        "score": score,
        "text": rid,
        "created_at": created_at,
        "metadata": metadata or {},
        "why_retrieved": [],
    }


def test_assignment_is_exhaustive_capacity_bounded_and_reroutes_overflow():
    db = FixedEmbedDB(
        {
            "Alpha. alpha concern": [1.0, 0.0],
            "Beta. beta concern": [0.0, 1.0],
            "strong alpha": [1.0, 0.0],
            "weaker alpha": [0.8, 0.2],
        }
    )
    handles = [
        TopicHandle("alpha", "Alpha", "alpha concern", ("existing",)),
        TopicHandle("beta", "Beta", "beta concern"),
    ]

    plan = assign_evidence_to_handles(
        db,
        {"existing": "already assigned", "a": "strong alpha", "b": "weaker alpha"},
        handles,
        max_evidence_per_handle=2,
    )

    by_id = {handle.id: handle for handle in plan.handles}
    assert by_id["alpha"].evidence_ids == ("existing", "a")
    assert by_id["beta"].evidence_ids == ("b",)
    assert {assignment.evidence_id for assignment in plan.assignments} == {"a", "b"}
    assert all(len(handle.evidence_ids) <= 2 for handle in plan.handles)


def test_assignment_refuses_insufficient_capacity_before_embedding():
    db = FixedEmbedDB({})
    with pytest.raises(ValueError, match="discover overflow handles"):
        assign_evidence_to_handles(
            db,
            {"a": "one", "b": "two"},
            [TopicHandle("only", "Only", "one handle")],
            max_evidence_per_handle=1,
        )
    assert db.calls == []


def test_validation_allows_bounded_cross_handle_evidence_views():
    handles = validate_topic_handles(
        [
            TopicHandle("topic", "Topic", "thematic view", ("e1",)),
            TopicHandle("person", "Person", "relationship view", ("e1",)),
        ]
    )

    assert [handle.id for handle in handles] == ["topic", "person"]


def test_validation_rejects_too_many_cross_handle_evidence_views():
    with pytest.raises(ValueError, match="assigned to 3 handles; maximum is 2"):
        validate_topic_handles(
            [
                TopicHandle("one", "One", "first", ("e1",)),
                TopicHandle("two", "Two", "second", ("e1",)),
                TopicHandle("three", "Three", "third", ("e1",)),
            ],
            max_handle_memberships=2,
        )


def test_concern_validation_bounds_evidence_reuse():
    items = validate_concern_items(
        [
            ConcernItem("one", "First concern", ("e1", "e1")),
            ConcernItem("two", "Second concern", ("e1", "e2")),
        ]
    )

    assert items[0].evidence_ids == ("e1",)
    with pytest.raises(ValueError, match="supports 2 concern items; maximum is 1"):
        validate_concern_items(items, max_item_memberships=1)


def test_organize_concerns_rejects_invented_ids_and_tracks_unassigned():
    db = FixedEmbedDB({})

    def discover(evidence):
        assert evidence == {"e1": "one", "e2": "two"}
        return [ConcernItem("one", "First concern", ("e1",))]

    plan, writes = organize_concerns(
        db,
        {"e1": "one", "e2": "two"},
        discover,
        persist=False,
    )

    assert plan.unassigned_evidence_ids == ("e2",)
    assert writes == []
    with pytest.raises(ValueError, match="invented evidence ids: absent"):
        organize_concerns(
            db,
            {"e1": "one"},
            lambda _: [ConcernItem("bad", "Bad concern", ("absent",))],
            persist=False,
        )


def test_persist_organization_uses_stable_logical_keys(monkeypatch):
    calls = []

    def fake_record_synthesis(*args, **kwargs):
        calls.append((args, kwargs))
        return {"consolidated_rid": f"s{len(calls)}"}

    monkeypatch.setattr("yantrikdb.organize.record_synthesis", fake_record_synthesis)
    plan = OrganizationPlan(
        (
            TopicHandle(
                "writing",
                "Writing",
                "The user's writing journey",
                ("r1", "r2"),
                ("personal statement",),
            ),
        )
    )

    writes = persist_organization(
        PersistDB(), plan, idempotency_prefix="topics:v2"
    )

    assert writes == [{"consolidated_rid": "s1"}]
    args, kwargs = calls[0]
    assert args[1] == ["r2", "r1"]
    assert args[2] == (
        "Topic trajectory: Writing. The user's writing journey"
    )
    assert args[4] == "topics:v2:writing"
    assert kwargs["granularity"] == "rollup"
    assert kwargs["embedding"] == [1.0, 0.0]
    assert kwargs["metadata"]["organizer_handle_id"] == "writing"
    assert kwargs["metadata"]["thread_entities"] == ["personal statement"]
    assert kwargs["metadata"]["child_rids"] == ["r2", "r1"]
    assert kwargs["metadata"]["organizer_evidence_timeline"] == [
        {
            "rid": "r2",
            "occurrence_at": 10.0,
            "evidence_span_end_at": 15.0,
            "created_at": 30.0,
            "first_mention_turn": 10.0,
            "date_source": "first_mention_at",
        },
        {
            "rid": "r1",
            "occurrence_at": 20.0,
            "evidence_span_end_at": 20.0,
            "created_at": 20.0,
            "first_mention_turn": None,
            "date_source": "created_at",
        },
    ]


def test_persist_concerns_uses_atomic_synthesis_and_turn_bounds(monkeypatch):
    calls = []

    def fake_record_synthesis(*args, **kwargs):
        calls.append((args, kwargs))
        return {"consolidated_rid": "concern-rid"}

    monkeypatch.setattr("yantrikdb.organize.record_synthesis", fake_record_synthesis)
    plan = ConcernPlan(
        (
            ConcernItem(
                "assembled",
                "The assembled concern",
                ("r1", "r2"),
                ("Carla",),
            ),
        )
    )

    writes = persist_concerns(PersistDB(), plan, idempotency_prefix="items:v2")

    assert writes == [{"consolidated_rid": "concern-rid"}]
    args, kwargs = calls[0]
    assert args[1] == ["r2", "r1"]
    assert args[2] == "The assembled concern"
    assert args[3] == "concern"
    assert args[4] == "items:v2:assembled"
    assert kwargs["granularity"] == "atomic"
    assert kwargs["embedding"] == [0.0, 1.0]
    assert kwargs["metadata"]["organizer_kind"] == "query_independent_concern"
    assert kwargs["metadata"]["first_mention_turn"] == 10
    assert kwargs["metadata"]["evidence_span_end_turn"] == 10
    assert kwargs["metadata"]["thread_entities"] == ["Carla"]


def test_persist_organization_rejects_missing_evidence(monkeypatch):
    monkeypatch.setattr(
        "yantrikdb.organize.record_synthesis", lambda *args, **kwargs: None
    )
    plan = OrganizationPlan(
        (TopicHandle("missing", "Missing", "broken handle", ("absent",)),)
    )

    with pytest.raises(ValueError, match="references missing evidence 'absent'"):
        persist_organization(PersistDB(), plan)


def test_organized_item_recall_selects_by_relevance_then_orders_occurrences():
    handle = _recall_hit(
        "handle",
        0.9,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "writing",
            "child_rids": ["early", "late"],
        },
    )
    late = _recall_hit(
        "late", 0.8, metadata={"first_mention_at": 20.0}, created_at=20.0
    )
    early = _recall_hit(
        "early", 0.7, metadata={"first_mention_at": 10.0}, created_at=10.0
    )
    unrelated = _recall_hit("unrelated", 0.95)
    db = RecallDB([unrelated, handle, late, early])

    results = recall_organized(
        db,
        "List the writing journey in order",
        top_k=2,
        candidate_pool=10,
    )

    assert [result["rid"] for result in results] == ["early", "late"]
    assert all(
        result["metadata"]["organization_handle_ids"] == ["writing"]
        for result in results
    )
    assert all(
        "organized_handle_expansion" in result["why_retrieved"]
        for result in results
    )


def test_organized_item_recall_prefers_concerns_over_fragments():
    raw = _recall_hit("raw", 0.99)
    late = _recall_hit(
        "late-concern",
        0.8,
        metadata={
            "organizer_kind": "query_independent_concern",
            "first_mention_turn": 20,
        },
    )
    early = _recall_hit(
        "early-concern",
        0.7,
        metadata={
            "organizer_kind": "query_independent_concern",
            "first_mention_turn": 10,
        },
    )
    db = RecallDB([raw, late, early])

    results = recall_organized(
        db,
        "List when I brought these up in our conversations",
        top_k=2,
        candidate_pool=10,
    )

    assert [result["rid"] for result in results] == [
        "early-concern",
        "late-concern",
    ]


def test_organized_item_recall_keeps_direct_concerns_in_mixed_store():
    selected_handle = _recall_hit(
        "selected-handle",
        0.99,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "selected",
            "child_rids": ["selected-child"],
        },
    )
    other_handle = _recall_hit(
        "other-handle",
        0.7,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "other",
            "child_rids": ["direct-concern"],
        },
    )
    selected_child = _recall_hit("selected-child", 0.2)
    direct_concern = _recall_hit(
        "direct-concern",
        0.9,
        metadata={
            "organizer_kind": "query_independent_concern",
            "first_mention_turn": 10,
        },
    )
    db = RecallDB(
        [selected_handle, direct_concern, other_handle, selected_child]
    )

    results = recall_organized(
        db,
        "List the relevant items in order",
        top_k=1,
        candidate_pool=10,
        max_handles=1,
        order="relevance",
    )

    assert [result["rid"] for result in results] == ["direct-concern"]
    assert "organized_direct_concern" in results[0]["why_retrieved"]


def test_organized_expansion_uses_non_labeling_point_reads():
    class TrackingDB(RecallDB):
        def __init__(self, results, memories=None):
            super().__init__(results, memories)
            self.get_calls = []
            self.get_memory_calls = []

        def get(self, rid):
            self.get_calls.append(rid)
            return super().get(rid)

        def get_memory(self, rid):
            self.get_memory_calls.append(rid)
            return super().get(rid)

    handle = _recall_hit(
        "handle",
        0.9,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "topic",
            "child_rids": ["not-in-recall-pool"],
        },
    )
    db = TrackingDB(
        [handle],
        {"not-in-recall-pool": _recall_hit("not-in-recall-pool", 0.7)},
    )

    results = recall_organized(
        db,
        "List the topic items in order",
        top_k=1,
        candidate_pool=10,
    )

    assert [result["rid"] for result in results] == ["not-in-recall-pool"]
    assert db.get_memory_calls == ["not-in-recall-pool"]
    assert db.get_calls == []


def test_organized_recall_records_only_returned_handle_children():
    handle = _recall_hit(
        "handle",
        0.9,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "topic",
            "child_rids": ["best", "not-returned"],
        },
    )
    best = _recall_hit("best", 0.8)
    not_returned = _recall_hit("not-returned", 0.2)
    db = RecallDB([handle, best, not_returned])
    db.impressions = []
    db.expansions = []
    db.note_rollup_impression = lambda rid, query, namespace, rank, score: (
        db.impressions.append((rid, query, namespace, rank, score)) or "imp-1"
    )
    db.note_rollup_expansion = lambda impression_id, children: db.expansions.append(
        (impression_id, children)
    )

    results = recall_organized(
        db,
        "List the topic items in order",
        top_k=1,
        candidate_pool=10,
        order="relevance",
    )

    assert [result["rid"] for result in results] == ["best"]
    assert db.impressions == [("handle", "List the topic items in order", None, 0, 0.9)]
    assert db.expansions == [("imp-1", ["best"])]
    assert results[0]["metadata"]["organization_rollup_impression_ids"] == [
        "imp-1"
    ]


def test_organized_recall_records_query_shape_count_and_child_scores():
    handle = _recall_hit(
        "handle",
        0.9,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "topic",
            "child_rids": ["best", "not-returned"],
        },
    )
    best = _recall_hit("best", 0.81)
    not_returned = _recall_hit("not-returned", 0.2)
    db = RecallDB([handle, best, not_returned])
    db.impressions = []
    db.expansions = []
    db.note_rollup_impression_features = (
        lambda rid,
        query,
        namespace,
        rank,
        score,
        requested_count,
        query_shape: db.impressions.append(
            (
                rid,
                query,
                namespace,
                rank,
                score,
                requested_count,
                query_shape,
            )
        )
        or "imp-features"
    )
    db.note_rollup_expansion_features = (
        lambda impression_id, children, scores: db.expansions.append(
            (impression_id, children, scores)
        )
    )

    results = recall_organized(
        db,
        "List exactly five topic items in order across conversations",
        top_k=1,
        candidate_pool=10,
        order="relevance",
    )

    assert [result["rid"] for result in results] == ["best"]
    assert db.impressions == [
        (
            "handle",
            "List exactly five topic items in order across conversations",
            None,
            0,
            0.9,
            5,
            "ordered_list",
        )
    ]
    assert db.expansions == [("imp-features", ["best"], [0.81])]


def test_organized_recall_uses_turn_order_for_conversation_queries():
    handle = _recall_hit(
        "handle",
        0.9,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "support",
            "child_rids": ["later-event", "earlier-event"],
        },
    )
    later_event = _recall_hit(
        "later-event",
        0.8,
        metadata={"first_mention_at": 10.0, "first_mention_turn": 20},
    )
    earlier_event = _recall_hit(
        "earlier-event",
        0.7,
        metadata={"first_mention_at": 20.0, "first_mention_turn": 10},
    )
    db = RecallDB([handle, later_event, earlier_event])

    results = recall_organized(
        db,
        "List when I brought these up across our conversations",
        top_k=2,
        candidate_pool=10,
    )

    assert [result["rid"] for result in results] == [
        "earlier-event",
        "later-event",
    ]


def test_organized_recall_hydrates_children_outside_candidate_pool():
    handle = _recall_hit(
        "handle",
        0.6,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "project",
            "child_rids": ["hidden"],
        },
    )
    hidden = _recall_hit(
        "hidden", 0.0, metadata={"first_mention_at": 5.0}, created_at=5.0
    )
    hidden.pop("score")
    db = RecallDB([handle], {"hidden": hidden})

    results = recall_organized(
        db, "Walk me through the project timeline", top_k=1
    )

    assert results[0]["rid"] == "hidden"
    assert results[0]["score"] == 0.6


def test_organized_recall_fuses_parent_and_child_relevance():
    strong_handle = _recall_hit(
        "strong-handle",
        0.9,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "strong",
            "child_rids": ["indirect-child"],
        },
    )
    weak_handle = _recall_hit(
        "weak-handle",
        0.3,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "weak",
            "child_rids": ["direct-child"],
        },
    )
    indirect = _recall_hit("indirect-child", 0.2)
    direct = _recall_hit("direct-child", 0.6)
    db = RecallDB([strong_handle, weak_handle, direct, indirect])

    results = recall_organized(
        db,
        "List the project timeline",
        top_k=1,
        candidate_pool=10,
        handle_weight=0.5,
        order="relevance",
    )

    assert [result["rid"] for result in results] == ["indirect-child"]
    assert results[0]["score"] == pytest.approx(0.55)
    assert results[0]["metadata"]["organization_direct_score"] == 0.2
    assert results[0]["metadata"]["organization_parent_score"] == 0.9


def test_organized_recall_expands_budget_for_named_handle_entity():
    handles = []
    children = []
    for index in range(9):
        handle_id = f"handle-{index}"
        label = "Carla collaboration" if index == 8 else f"Topic {index}"
        handles.append(
            _recall_hit(
                handle_id,
                1.0 - index * 0.01,
                metadata={
                    "organizer_kind": "query_independent_topic",
                    "organizer_handle_id": handle_id,
                    "organizer_label": label,
                    "child_rids": [f"child-{index}"],
                },
            )
        )
        children.append(_recall_hit(f"child-{index}", 0.5 - index * 0.01))
    db = RecallDB([*handles, *children])

    results = recall_organized(
        db,
        "List my collaboration with Carla in order",
        top_k=9,
        candidate_pool=20,
        order="relevance",
    )

    assert "child-8" in {result["rid"] for result in results}
    assert {result["rid"] for result in results} == {"child-8"}


def test_query_scaffolding_is_not_treated_as_entity_or_focus():
    from yantrikdb.organize import (
        _query_focus_tokens,
        _query_handle_entity_tokens,
    )

    query = (
        "Can I list different aspects of my project in order? "
        "Mention ONLY and ONLY three items."
    )

    assert _query_handle_entity_tokens(query) == set()
    assert "three" not in _query_focus_tokens(query)


def test_named_entity_focus_follows_concern_provenance():
    handle = _recall_hit(
        "carla-handle",
        0.9,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "carla",
            "organizer_label": "Carla collaboration",
            "child_rids": ["direct-carla"],
        },
    )
    lossy_concern = _recall_hit(
        "webinar-concern",
        0.8,
        metadata={
            "organizer_kind": "query_independent_concern",
            "child_rids": ["webinar-evidence"],
        },
    )
    lossy_concern["text"] = "The user promoted a webinar through guild leaders."
    evidence = _recall_hit("webinar-evidence", 0.4)
    evidence["text"] = "The user planned a webinar Q&A session with Carla."
    direct = _recall_hit("direct-carla", 0.6)
    db = RecallDB([handle, lossy_concern, evidence, direct])

    results = recall_organized(
        db,
        "List my collaboration with Carla in order",
        top_k=3,
        candidate_pool=10,
        order="relevance",
    )

    assert {result["rid"] for result in results} == {
        "direct-carla",
        "webinar-concern",
    }


def test_named_entity_focus_keeps_unanchored_text_mentions_without_a_count():
    handle = _recall_hit(
        "carla-handle",
        0.9,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "carla",
            "organizer_label": "Carla collaboration",
            "child_rids": ["direct-carla"],
        },
    )
    incidental = _recall_hit(
        "incidental",
        0.8,
        metadata={
            "organizer_kind": "query_independent_concern",
            "child_rids": ["incidental-evidence"],
        },
    )
    incidental["text"] = "The user considered asking Amy and Carla for referrals."
    evidence = _recall_hit("incidental-evidence", 0.4)
    evidence["text"] = incidental["text"]
    direct = _recall_hit("direct-carla", 0.6)
    db = RecallDB([handle, incidental, evidence, direct])

    results = recall_organized(
        db,
        "List my collaboration with Carla in order",
        top_k=3,
        candidate_pool=10,
        order="relevance",
    )

    assert {result["rid"] for result in results} == {"direct-carla", "incidental"}


def test_named_entity_focus_fills_an_explicit_count_from_grounded_matches_first():
    handle = _recall_hit(
        "carla-handle",
        0.9,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "carla",
            "organizer_label": "Carla collaboration",
            "child_rids": ["handled"],
        },
    )
    handled = _recall_hit(
        "handled",
        0.6,
        metadata={"organizer_kind": "query_independent_concern"},
    )
    anchored = _recall_hit(
        "anchored",
        0.4,
        metadata={
            "organizer_kind": "query_independent_concern",
            "anchor_entities": ["Carla"],
        },
    )
    incidental = _recall_hit(
        "incidental",
        0.8,
        metadata={"organizer_kind": "query_independent_concern"},
    )
    incidental["text"] = "The user considered asking Amy and Carla for referrals."
    db = RecallDB([handle, incidental, handled, anchored])

    results = recall_organized(
        db,
        "List exactly two parts of my collaboration with Carla in order",
        top_k=4,
        candidate_pool=10,
        order="relevance",
    )

    assert {result["rid"] for result in results} == {"handled", "anchored"}


def test_named_entity_focus_deduplicates_repeated_concern_wording():
    child_ids = ["checklist-1", "checklist-2", "webinar-plan", "webinar-promo"]
    handle = _recall_hit(
        "carla-handle",
        0.9,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "carla",
            "organizer_label": "Carla collaboration",
            "child_rids": child_ids,
        },
    )
    texts = {
        "checklist-1": "Carla shared a checklist for reducing passive voice.",
        "checklist-2": "The user reduced passive voice using Carla's checklist.",
        "webinar-plan": "The user planned a joint editing webinar with Carla.",
        "webinar-promo": "The webinar used guild newsletters and prize incentives.",
    }
    children = []
    for turn, rid in enumerate(child_ids, 1):
        child = _recall_hit(
            rid,
            0.7,
            metadata={
                "organizer_kind": "query_independent_concern",
                "first_mention_turn": turn,
            },
        )
        child["text"] = texts[rid]
        children.append(child)
    db = RecallDB([handle, *children])

    results = recall_organized(
        db,
        "List my collaboration with Carla in order",
        top_k=4,
        candidate_pool=10,
        order="relevance",
    )

    assert [result["rid"] for result in results] == [
        "checklist-1",
        "webinar-plan",
        "webinar-promo",
    ]


def test_named_entity_focus_parses_exact_count_through_twenty():
    assert _requested_item_count("List exactly twelve items about Carla") == 12
    assert _requested_item_count("Mention only twenty items about Carla") == 20


def test_organized_recall_focuses_a_multi_token_concern():
    generic = _recall_hit(
        "generic-handle",
        0.9,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "generic",
            "organizer_label": "Application logistics",
            "child_rids": ["generic-child"],
        },
    )
    family = _recall_hit(
        "family-handle",
        0.7,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "family",
            "organizer_label": "Family Support System",
            "child_rids": ["family-child"],
        },
    )
    db = RecallDB(
        [
            generic,
            family,
            _recall_hit("generic-child", 0.8),
            _recall_hit("family-child", 0.6),
        ]
    )

    results = recall_organized(
        db,
        "List the ways my family supported me in our conversations",
        top_k=2,
        candidate_pool=10,
        order="relevance",
    )

    assert [result["rid"] for result in results] == ["family-child"]


def test_organized_recall_direct_concerns_must_match_the_full_focus():
    family = _recall_hit(
        "family-handle",
        0.7,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "family",
            "organizer_label": "Family Support System",
            "child_rids": ["wendy-child"],
        },
    )
    wendy = _recall_hit("wendy-child", 0.6)
    hardship = _recall_hit(
        "hardship",
        0.9,
        metadata={"organizer_kind": "query_independent_concern"},
    )
    hardship["text"] = "A career gap caused by family illness"
    tanya = _recall_hit(
        "tanya",
        0.5,
        metadata={"organizer_kind": "query_independent_concern"},
    )
    tanya["text"] = "Tanya provided family support by rehearsing the pitch"
    db = RecallDB([family, hardship, tanya, wendy])

    results = recall_organized(
        db,
        "List the ways my family supported me in our conversations",
        top_k=4,
        candidate_pool=10,
        order="relevance",
    )

    assert {result["rid"] for result in results} == {"wendy-child", "tanya"}
    assert "hardship" not in {result["rid"] for result in results}


def test_multi_token_focus_rejects_children_outside_handle_anchors():
    family = _recall_hit(
        "family-handle",
        0.7,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "family",
            "organizer_label": "Family Support System",
            "anchor_entities": ["Wendy"],
            "child_rids": ["wendy", "danielle"],
        },
    )
    wendy = _recall_hit(
        "wendy",
        0.6,
        metadata={"organizer_kind": "query_independent_concern"},
    )
    wendy["text"] = "Wendy sent a letter encouraging the user."
    danielle = _recall_hit(
        "danielle",
        0.8,
        metadata={
            "organizer_kind": "query_independent_concern",
            "anchor_entities": ["Danielle"],
        },
    )
    tanya = _recall_hit(
        "tanya",
        0.5,
        metadata={"organizer_kind": "query_independent_concern"},
    )
    tanya["text"] = "Tanya provided family support by rehearsing the pitch"
    db = RecallDB([family, danielle, wendy, tanya])

    results = recall_organized(
        db,
        "List the ways my family supported me in our conversations",
        top_k=4,
        candidate_pool=10,
        order="relevance",
    )

    assert {result["rid"] for result in results} == {"wendy", "tanya"}


def test_organized_recall_keeps_concerns_orphaned_from_topic_handles():
    handle = _recall_hit(
        "handle",
        0.9,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "project",
            "organizer_label": "Project",
            "child_rids": ["attached"],
        },
    )
    orphan = _recall_hit(
        "orphan",
        0.8,
        metadata={"organizer_kind": "query_independent_concern"},
    )
    attached = _recall_hit(
        "attached",
        0.7,
        metadata={"organizer_kind": "query_independent_concern"},
    )
    db = RecallDB([handle, orphan, attached])

    results = recall_organized(
        db,
        "List the project items",
        top_k=2,
        candidate_pool=10,
        order="relevance",
    )

    assert {result["rid"] for result in results} == {"attached", "orphan"}


def test_organized_recall_returns_handles_for_summary_queries():
    handle = _recall_hit(
        "handle",
        0.9,
        metadata={
            "organizer_kind": "query_independent_topic",
            "organizer_handle_id": "writing",
            "child_rids": ["child"],
        },
    )
    db = RecallDB([handle, _recall_hit("child", 0.8)])

    results = recall_organized(db, "Give me an overview of my writing")

    assert [result["rid"] for result in results] == ["handle"]


def test_organized_recall_keeps_neutral_queries_on_raw_path():
    hits = [_recall_hit("raw", 0.9), _recall_hit("other", 0.8)]
    db = RecallDB(hits)

    results = recall_organized(db, "What is my current project?", top_k=1)

    assert [result["rid"] for result in results] == ["raw"]


def test_organized_recall_round_trips_through_real_database(tmp_path):
    class SemanticEmbedder:
        def encode(self, text):
            vector = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if "early" in text.casefold():
                vector[2] = 0.3
            elif "late" in text.casefold():
                vector[3] = 0.3
            elif "topic trajectory" in text.casefold():
                vector[1] = 0.3
            return vector

    db = YantrikDB(str(tmp_path / "organized.db"), embedding_dim=8)
    db.set_embedder(SemanticEmbedder())
    try:
        late = db.record("late writing milestone", created_at=20.0)
        early = db.record("early writing milestone", created_at=10.0)
        plan = OrganizationPlan(
            (
                TopicHandle(
                    "writing",
                    "Writing journey",
                    "The user's writing milestones",
                    (late, early),
                ),
            )
        )

        writes = persist_organization(db, plan)
        handle = db.get(writes[0]["consolidated_rid"])
        results = recall_organized(
            db,
            "List the writing journey in order",
            top_k=2,
            candidate_pool=10,
        )

        assert handle["metadata"]["child_rids"] == [early, late]
        assert [result["rid"] for result in results] == [early, late]
    finally:
        db.close()


def test_concern_items_round_trip_through_real_database(tmp_path):
    class SemanticEmbedder:
        def encode(self, text):
            vector = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if "checklist" in text.casefold():
                vector[1] = 0.5
            return vector

    db = YantrikDB(str(tmp_path / "concerns.db"), embedding_dim=8)
    db.set_embedder(SemanticEmbedder())
    try:
        first = db.record(
            "Carla shared an editing checklist",
            created_at=20.0,
            metadata={"first_mention_turn": 12},
        )
        second = db.record(
            "The checklist reduced passive voice",
            created_at=30.0,
            metadata={"first_mention_turn": 18},
        )
        plan = ConcernPlan(
            (
                ConcernItem(
                    "carla-checklist",
                    "Carla's checklist helped reduce passive voice.",
                    (first, second),
                    ("Carla",),
                ),
            )
        )

        writes = persist_concerns(db, plan)
        concern = db.get(writes[0]["consolidated_rid"])
        results = recall_organized(
            db,
            "List what I discussed about Carla's checklist in our conversations",
            top_k=1,
            candidate_pool=10,
        )

        assert concern["metadata"]["child_rids"] == [first, second]
        assert concern["metadata"]["first_mention_turn"] == 12
        assert concern["metadata"]["evidence_span_end_turn"] == 18
        assert concern["metadata"]["first_mention_at"] == 20.0
        assert concern["metadata"]["evidence_span_end_at"] == 30.0
        assert [result["rid"] for result in results] == [concern["rid"]]
    finally:
        db.close()


def test_concern_dates_fall_back_to_evidence_created_at(tmp_path):
    db = YantrikDB(str(tmp_path / "concern-created-at.db"), embedding_dim=8)
    db.set_embedder(type("Embedder", (), {"encode": lambda self, text: [1.0] * 8})())
    try:
        late = db.record("late evidence", created_at=30.0)
        early = db.record("early evidence", created_at=10.0)

        writes = persist_concerns(
            db,
            ConcernPlan((ConcernItem("dated", "assembled concern", (late, early)),)),
        )

        stored = db.get(writes[0]["consolidated_rid"])
        assert stored["metadata"]["child_rids"] == [early, late]
        assert stored["metadata"]["first_mention_at"] == 10.0
        assert stored["metadata"]["evidence_span_end_at"] == 30.0
    finally:
        db.close()


def test_concern_persistence_honors_a_custom_evidence_bound(tmp_path):
    db = YantrikDB(str(tmp_path / "large-concern.db"), embedding_dim=8)
    db.set_embedder(type("Embedder", (), {"encode": lambda self, text: [1.0] * 8})())
    try:
        evidence_ids = tuple(
            db.record(f"evidence {index}", created_at=float(index))
            for index in range(1, 8)
        )
        plan = ConcernPlan((ConcernItem("large", "large concern", evidence_ids),))

        writes = persist_concerns(db, plan, max_evidence_per_item=7)

        assert len(writes) == 1
        stored = db.get(writes[0]["consolidated_rid"])
        assert stored["metadata"]["child_rids"] == list(evidence_ids)
    finally:
        db.close()


def test_organize_evidence_keeps_discovery_model_outside_the_engine():
    db = FixedEmbedDB(
        {
            "Writing. writing concern": [1.0, 0.0],
            "draft detail": [1.0, 0.0],
        }
    )
    seen = []

    def discover(evidence):
        seen.append(evidence)
        return [TopicHandle("writing", "Writing", "writing concern")]

    plan, writes = organize_evidence(
        db,
        {"r1": "draft detail"},
        discover,
        persist=False,
    )

    assert seen == [{"r1": "draft detail"}]
    assert plan.handles[0].evidence_ids == ("r1",)
    assert writes == []
