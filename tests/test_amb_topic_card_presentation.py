import pytest

from benchmarks.amb.topic_card_presentation import (
    load_persisted_topic_cards,
    topic_card_document,
)
from yantrikdb import (
    load_persisted_topic_cards as public_load_persisted_topic_cards,
)
from yantrikdb import topic_card_document as public_topic_card_document


def _topic(rid, label="Writing", summary="Revised the draft."):
    return {
        "rid": rid,
        "text": f"Topic trajectory: {label}. {summary}",
        "metadata": {
            "organizer_kind": "query_independent_topic",
            "organizer_label": label,
            "organizer_summary": summary,
            "first_mention_at": 1709251200.0,
            "evidence_span_end_at": 1711929600.0,
            "organizer_evidence_timeline": [
                {"first_mention_turn": 3},
                {"first_mention_turn": 9},
            ],
        },
    }


class _PagedDB:
    def __init__(self):
        self.calls = []

    def list_records(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["since_rid"] is None:
            return {
                "records": [
                    _topic("002", "Career", "Changed direction."),
                    {"rid": "raw", "metadata": {}},
                ],
                "next_cursor": "002",
            }
        return {
            "records": [_topic("001"), _topic("003")],
            "next_cursor": None,
        }


def test_topic_card_document_carries_storage_dates_and_turn_bounds():
    document = topic_card_document(_topic("topic"))

    assert document == (
        "Topic: Writing (recorded 2024-03-01 to 2024-04-01; turns 3-9)\n"
        "Revised the draft."
    )


def test_benchmark_path_reexports_public_topic_card_api():
    assert load_persisted_topic_cards is public_load_persisted_topic_cards
    assert topic_card_document is public_topic_card_document


def test_topic_card_document_ignores_invalid_chronology_metadata():
    record = _topic("topic")
    record["metadata"].update(
        {
            "first_mention_at": float("nan"),
            "evidence_span_end_at": float("inf"),
            "organizer_evidence_timeline": [
                {"first_mention_turn": "unknown"},
                {"first_mention_turn": -1},
            ],
        }
    )

    assert topic_card_document(record) == "Topic: Writing\nRevised the draft."


def test_load_persisted_topic_cards_pages_orders_and_deduplicates():
    db = _PagedDB()

    cards, trace = load_persisted_topic_cards(db, "person-a", page_size=2)

    assert [card["rid"] for card in cards] == ["001", "002"]
    assert db.calls[0]["namespace"] == "person-a"
    assert db.calls[1]["since_rid"] == "002"
    assert trace == {
        "pages": 2,
        "records_scanned": 4,
        "organizer_records": 3,
        "duplicate_cards_removed": 1,
        "cards_returned": 2,
    }


def test_load_persisted_topic_cards_rejects_non_advancing_cursor():
    class StalledDB:
        def list_records(self, **kwargs):
            return {"records": [], "next_cursor": "same"}

    with pytest.raises(RuntimeError, match="non-advancing cursor"):
        load_persisted_topic_cards(StalledDB(), page_size=1)
