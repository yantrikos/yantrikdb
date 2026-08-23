import pytest

from benchmarks.amb.replay_standing_instruction_product import (
    _json_bytes,
    _records_from_lane,
    _selected_rows,
)


def test_json_bytes_preserves_requested_frozen_newlines():
    payload = {"results": [{"query_id": "q", "context": "line\nline"}]}

    encoded = _json_bytes(payload, "\r\n")

    assert b"\r\n" in encoded
    assert b"\n" not in encoded.replace(b"\r\n", b"")


def test_selected_rows_accepts_the_preregistered_all_category():
    rows = [
        {"meta": {"question_category": "instruction_following"}},
        {"meta": {"question_category": "temporal_reasoning"}},
    ]

    assert _selected_rows(rows, "all") == rows


def test_records_from_lane_uses_earliest_user_evidence_turn():
    lane = {
        "facets": [
            {
                "text": "Always cite sources.",
                "source_rids": ["later", "earlier"],
            }
        ],
        "omitted": 0,
    }
    sources = {
        "later": {"turn": 12, "role": "user"},
        "earlier": {"turn": 4, "role": "user"},
    }

    assert _records_from_lane(lane, sources) == [
        {"turn": 4, "text": "Always cite sources."}
    ]


def test_records_from_lane_rejects_non_user_evidence():
    lane = {
        "facets": [{"text": "Always cite sources.", "source_rids": ["assistant"]}],
        "omitted": 0,
    }

    with pytest.raises(AssertionError, match="non-user evidence"):
        _records_from_lane(
            lane,
            {"assistant": {"turn": 4, "role": "assistant"}},
        )
