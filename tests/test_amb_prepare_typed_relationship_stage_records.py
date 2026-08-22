import pytest

from benchmarks.amb.prepare_relationship_stage_records import _ordered_groups
from benchmarks.amb.prepare_typed_relationship_stage_records import (
    build_typed_preflight,
    materialize_typed_records,
    render_typed_records,
)
from tests.test_amb_prepare_relationship_stage_records import _artifact


def _response():
    return {
        "records": [
            {
                "source_group": "s2",
                "speaker_perspective": "first_person",
                "goal_or_concern": "I wanted stronger feedback on my draft.",
                "event": "Robert reviewed the draft.",
                "decision": "",
                "outcome": "",
                "follow_up": "I planned to revise it.",
            },
            {
                "source_group": "s1",
                "speaker_perspective": "first_person",
                "goal_or_concern": "I wanted to make a good impression.",
                "event": "I met Robert at the library.",
                "decision": "",
                "outcome": "",
                "follow_up": "I planned a follow-up call.",
            },
        ]
    }


def test_typed_preflight_is_query_blind_and_preserves_all_rows():
    preflight = build_typed_preflight(_artifact(), "q", "model")

    assert preflight["query_exposed_to_synthesis"] is False
    assert preflight["source_group_count"] == 2
    assert preflight["evidence_row_count"] == 3
    assert "List my academic mentorship stages" not in preflight["prompt"]
    assert "goal_or_concern" in preflight["prompt"]
    assert "never call the owner 'the user'" in preflight["prompt"]


def test_materialization_restores_order_and_attaches_provenance():
    _, groups = _ordered_groups(_artifact())

    records = materialize_typed_records(_response(), groups)

    assert [record["source_group"] for record in records] == ["s1", "s2"]
    assert records[0]["speaker_perspective"] == "first_person"
    assert records[0]["evidence_ids"] == ["e1", "e2"]
    assert records[0]["evidence_turns"] == [10, 12]


def test_materialization_accepts_first_person_contractions():
    _, groups = _ordered_groups(_artifact())
    response = _response()
    response["records"][0]["goal_or_concern"] = "I'm seeking draft feedback."

    records = materialize_typed_records(response, groups)

    assert records[1]["goal_or_concern"] == "I'm seeking draft feedback."


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("speaker_perspective", "third_person", "speaker perspective"),
        ("goal_or_concern", "The user wanted help.", "loses first-person"),
        ("goal_or_concern", "Wanted help.", "must preserve first person"),
    ],
)
def test_materialization_rejects_perspective_loss(field, value, message):
    _, groups = _ordered_groups(_artifact())
    response = _response()
    response["records"][0][field] = value

    with pytest.raises(ValueError, match=message):
        materialize_typed_records(response, groups)


def test_render_keeps_facets_distinct_and_ids_out_of_answer_context():
    preflight = build_typed_preflight(_artifact(), "q", "model")
    records = materialize_typed_records(_response(), preflight["groups"])

    context = render_typed_records(preflight["anchor"], records)

    assert "My goal or concern: I wanted to make a good impression." in context
    assert "What happened: I met Robert at the library." in context
    assert "Next step: I planned a follow-up call." in context
    assert "Evidence turns: 10, 12" in context
    assert "Evidence IDs:" not in context
