import pytest

from benchmarks.amb.build_event_ordering_thread_v2_stage_a import (
    FREEZE_PROTOCOL,
    audit,
    bounded_handle_evidence,
    parser,
    query_entities,
    render_thread,
    select_topics,
    sha256_path,
    validate_thread,
    write_json,
)
from benchmarks.amb.event_ordering_thread_v2 import (
    event_ordering_focus,
    is_event_ordering_chronology_query,
)


@pytest.mark.parametrize(
    "query, focus",
    [
        (
            "Can you list the order in which I brought up different aspects "
            "of refining my personal statement throughout our conversations "
            "in order? Mention only five items.",
            "refining my personal statement",
        ),
        (
            "Can you walk me through the order in which I brought up "
            "different aspects of my writing journey across our conversations?",
            "my writing journey",
        ),
        (
            "Can you list in order how I brought up different ways my family "
            "supported me in my personal statement throughout our chats?",
            "my family supported me in my personal statement",
        ),
        (
            "CAN YOU WALK ME THROUGH THE ORDER IN WHICH I BROUGHT UP SHARED "
            "ENTERTAINMENT INTERESTS WITH DOUGLAS DURING OUR CHATS?",
            "SHARED ENTERTAINMENT INTERESTS WITH DOUGLAS",
        ),
        (
            "Can you walk me through the order in which I brought up concerns "
            "related to my family's care in our conversations, in order?",
            "concerns related to my family's care",
        ),
    ],
)
def test_preregistered_predicate_and_focus(query, focus):
    assert is_event_ordering_chronology_query(query)
    assert event_ordering_focus(query) == focus


@pytest.mark.parametrize(
    "query",
    [
        "Can you list the order in which these events happened?",
        "What topics did I bring up throughout our conversations?",
        "Can you summarize what I brought up in order?",
        "Can you list our conversations in chronological order?",
        "",
    ],
)
def test_preregistered_predicate_rejects_near_misses(query):
    assert not is_event_ordering_chronology_query(query)
    assert event_ordering_focus(query) is None


def test_query_entities_is_conservative_and_excludes_ai():
    assert query_entities("my collaboration with Carla") == ["Carla"]
    assert query_entities("shared interests with Douglas and Patrick") == [
        "Douglas",
        "Patrick",
    ]
    assert query_entities("using AI in our hiring process") == []


class _TopicDB:
    def __init__(self):
        self.call = None

    def recall(self, **kwargs):
        self.call = kwargs
        return [
            {
                "rid": "topic-1",
                "score": 0.8,
                "metadata": {
                    "organizer_kind": "query_independent_topic",
                    "organizer_label": "One",
                },
            },
            {
                "rid": "not-a-topic",
                "score": 0.7,
                "metadata": {"organizer_kind": "query_independent_concern"},
            },
            {
                "rid": "topic-2",
                "score": 0.6,
                "metadata": {
                    "organizer_kind": "query_independent_topic",
                    "organizer_label": "Two",
                },
            },
        ]


def test_topic_selection_is_bounded_to_persisted_inference_records():
    db = _TopicDB()
    rids, trace = select_topics(db, "specific focus", max_topics=2)
    assert rids == ["topic-1", "topic-2"]
    assert [row["label"] for row in trace] == ["One", "Two"]
    assert db.call == {
        "query": "specific focus",
        "top_k": 2,
        "namespace": "default",
        "source": "inference",
        "include_consolidated": True,
        "skip_reinforce": True,
    }


def test_handle_membership_bound_is_query_free_and_deterministic():
    handles = [
        {"label": "large", "evidence_ids": ["shared", "a", "b"]},
        {"label": "small-1", "evidence_ids": ["shared"]},
        {"label": "small-2", "evidence_ids": ["shared"]},
        {"label": "small-3", "evidence_ids": ["shared"]},
    ]
    bounded = bounded_handle_evidence(handles, max_memberships=3)
    by_label = {raw["label"]: evidence for _, raw, evidence in bounded}
    assert "shared" not in by_label["large"]
    assert [by_label[f"small-{index}"] for index in (1, 2, 3)] == [
        ["shared"],
        ["shared"],
        ["shared"],
    ]


def test_thread_validation_and_rendering():
    items = [
        {
            "rid": "a",
            "text": "User said: first point",
            "created_at": 1_710_460_800.0,
            "source_turn": 4,
            "position": 1,
        },
        {
            "rid": "b",
            "text": "second point",
            "created_at": 1_710_460_800.0,
            "source_turn": 8,
            "position": 2,
        },
    ]
    result = {"items": items, "total": 2, "returned": 2, "omitted": 0}
    validate_thread(result)
    rendered = render_thread(items)
    assert "[March-15-2024 | Turn 4] User: first point" in rendered
    assert "## Memory 2" in rendered

    result["omitted"] = 1
    with pytest.raises(ValueError, match="accounting|truncation"):
        validate_thread(result)


def test_build_parser_has_no_gold_or_membership_input():
    build_parser = next(
        action
        for action in parser()._actions
        if action.dest == "command"
    ).choices["build"]
    destinations = {action.dest for action in build_parser._actions}
    assert "membership" not in destinations
    assert "gold" not in destinations


def test_audit_refuses_artifact_changed_after_freeze(tmp_path):
    artifact = tmp_path / "artifact.json"
    freeze = tmp_path / "freeze.json"
    membership = tmp_path / "membership.json"
    output = tmp_path / "audit.json"
    write_json(artifact, {"protocol": "placeholder"})
    write_json(
        freeze,
        {
            "protocol": FREEZE_PROTOCOL,
            "artifact_sha256": sha256_path(artifact),
        },
    )
    write_json(membership, {})
    artifact.write_text("{}\n", encoding="utf-8")

    args = type(
        "Args",
        (),
        {
            "artifact": artifact,
            "freeze": freeze,
            "membership": membership,
            "output": output,
        },
    )()
    with pytest.raises(ValueError, match="changed after freeze"):
        audit(args)
