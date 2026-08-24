import hashlib

import pytest

from benchmarks.amb.analyze_event_ordering_v5_autopsy import (
    analyze,
    extract_context_turns,
)
from benchmarks.amb.paired_frozen_context_eval import _run_fingerprint


def _fixture():
    combined_config = {"protocol": "paired-independent-mean-of-three-v1"}
    replicate_config = {"protocol": "paired-v1", "seed": 7}
    replicate_bytes = b"frozen replicate"
    replicate_sha = hashlib.sha256(replicate_bytes).hexdigest()
    combined = {
        "label_a": "control",
        "label_b": "treatment",
        "run_config": combined_config,
        "run_fingerprint": _run_fingerprint(combined_config),
        "source_replicates": [
            {
                "sha256": replicate_sha,
                "run_fingerprint": _run_fingerprint(replicate_config),
                "seed": 7,
                "model_seed": 7,
            }
        ],
        "pairs": [
            {
                "query_id": "1_event_ordering_0",
                "score_a": 0.5,
                "score_b": 0.25,
                "delta_b_minus_a": -0.25,
                "replicate_scores_a": [0.5],
                "replicate_scores_b": [0.25],
            },
            {
                "query_id": "2_event_ordering_0",
                "score_a": 0.5,
                "score_b": 0.75,
                "delta_b_minus_a": 0.25,
                "replicate_scores_a": [0.5],
                "replicate_scores_b": [0.75],
            },
            {
                "query_id": "3_event_ordering_0",
                "score_a": 0.5,
                "score_b": 0.0,
                "delta_b_minus_a": -0.5,
                "replicate_scores_a": [0.5],
                "replicate_scores_b": [0.0],
            },
        ],
    }

    def result(query_id, context):
        return {
            "query_id": query_id,
            "score_a": next(
                row["score_a"]
                for row in combined["pairs"]
                if row["query_id"] == query_id
            ),
            "score_b": next(
                row["score_b"]
                for row in combined["pairs"]
                if row["query_id"] == query_id
            ),
            "result_a": {
                "query_id": query_id,
                "query": f"Question {query_id}",
                "context": context,
                "meta": {"question_category": "event_ordering"},
            },
        }

    replicate = {
        "label_a": "control",
        "label_b": "treatment",
        "seed": 7,
        "model_seed": 7,
        "run_config": replicate_config,
        "run_fingerprint": _run_fingerprint(replicate_config),
        "pairs": [
            result("1_event_ordering_0", "[Turn 1] User: one"),
            result(
                "2_event_ordering_0",
                "[March-1-2024 | Turn 3] User: three\n[Turn 4] Assistant: four",
            ),
            result("3_event_ordering_0", "[Turn 5] User: five"),
        ],
    }
    membership = {
        "results": [
            {
                "query_id": "1_event_ordering_0",
                "source_turns": [1, 2],
                "query_matched_handles": {"route": "focus"},
                "topic_handles": {"3": {"turns": [1, 2]}},
            },
            {
                "query_id": "2_event_ordering_0",
                "source_turns": [3, 4],
                "query_matched_handles": {"route": "entity"},
                "topic_handles": {"3": {"turns": [3]}},
            },
            {
                "query_id": "3_event_ordering_0",
                "source_turns": [5, 6],
                "query_matched_handles": {},
                "topic_handles": {"3": {"turns": [5, 6]}},
            },
        ]
    }
    return combined, replicate, membership, replicate_sha


def test_extract_context_turns_accepts_dated_plain_and_continuation_headers():
    context = (
        "[Turn 4] User: plain\n"
        "[March-14-2024 | Turn 8] (cont.) Assistant: dated\n"
        "[Turn 4] User: duplicate\n"
        "not [Turn 99] User: inline"
    )
    assert extract_context_turns(context) == [4, 8]


def test_analyze_uses_control_coverage_and_label_defect_precedence():
    combined, replicate, membership, replicate_sha = _fixture()
    report = analyze(
        combined,
        replicate,
        membership,
        replicate_sha256=replicate_sha,
        expected_queries=3,
        label_defects={"3_event_ordering_0": "known bad partition"},
    )

    assert report["mean_source_turn_recall"] == pytest.approx(2 / 3)
    assert report["exact_source_coverage"] == 1
    assert report["zero_source_coverage"] == 0
    assert report["query_routes"]["bounded_focus_phrase"] == {
        "queries": 2,
        "query_ids": ["1_event_ordering_0", "2_event_ordering_0"],
        "source_turn_references": 4,
        "control_context_source_turns": 3,
        "control_context_micro_recall": 0.75,
        "oracle_top3_topic_source_turns": 3,
        "oracle_top3_topic_micro_recall": 0.75,
    }
    assert report["query_routes"]["broad_compound_topic_union"]["queries"] == 1
    assert report["score_deficits"]["by_class"] == {
        "label_defect": 1,
        "retrieval_miss": 1,
        "source_complete_answer_residual": 1,
    }
    assert report["negative_deltas"] == {
        "queries": 2,
        "source_incomplete": 2,
        "by_class": {"label_defect": 1, "retrieval_miss": 1},
        "query_ids": ["1_event_ordering_0", "3_event_ordering_0"],
    }
    rows = {row["query_id"]: row for row in report["rows"]}
    assert rows["1_event_ordering_0"]["missing_source_turns"] == [2]
    assert rows["2_event_ordering_0"]["failure_class"] == (
        "source_complete_answer_residual"
    )
    assert rows["3_event_ordering_0"]["failure_class"] == "label_defect"


def test_analyze_rejects_replicate_outside_frozen_combined_sources():
    combined, replicate, membership, _ = _fixture()
    with pytest.raises(ValueError, match="not one unique combined source"):
        analyze(
            combined,
            replicate,
            membership,
            replicate_sha256="0" * 64,
            expected_queries=3,
            label_defects={},
        )


def test_analyze_rejects_treatment_context_category_or_score_drift():
    combined, replicate, membership, replicate_sha = _fixture()
    replicate["pairs"][0]["score_b"] = 0.5
    with pytest.raises(ValueError, match="treatment score mismatch"):
        analyze(
            combined,
            replicate,
            membership,
            replicate_sha256=replicate_sha,
            expected_queries=3,
            label_defects={},
        )
