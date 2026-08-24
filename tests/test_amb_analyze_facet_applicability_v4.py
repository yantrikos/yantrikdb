import pytest

from benchmarks.amb.analyze_facet_applicability_v4 import analyze


CATEGORIES = [
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
]


def _fixture(category_deltas=None, bootstrap_seed=11):
    category_deltas = category_deltas or {"instruction_following": 0.1}
    pairs = []
    rows = []
    for category in CATEGORIES:
        for index in range(40):
            query_id = f"{category}-{index}"
            delta = category_deltas.get(category, 0.0)
            pairs.append({"query_id": query_id, "score_a": 0.5, "score_b": 0.5 + delta})
            rows.append({"query_id": query_id, "meta": {"question_category": category}})
    result = {
        "bootstrap_seed": bootstrap_seed,
        "run_config": {"bootstrap_seed": bootstrap_seed},
        "summary": {"paired_bootstrap_seed": bootstrap_seed},
        "pairs": pairs,
    }
    return result, rows


def test_v4_analyzer_passes_all_six_frozen_gates():
    result, rows = _fixture()

    report = analyze(result, rows, bootstrap_seed=11)

    assert report["promotion_passed"] is True
    assert len(report["gates"]) == 6
    assert all(report["gates"].values())


@pytest.mark.parametrize(
    ("deltas", "gate"),
    [
        (
            {"instruction_following": 0.04},
            "instruction_delta_at_least_0_05_and_wins_exceed_losses",
        ),
        (
            {"instruction_following": 0.1, "summarization": -0.02},
            "summarization_delta_at_least_minus_0_01",
        ),
        (
            {"instruction_following": 0.1, "event_ordering": -0.001},
            "event_ordering_delta_nonnegative",
        ),
    ],
)
def test_v4_analyzer_rejects_named_gate_failures(deltas, gate):
    result, rows = _fixture(deltas)

    report = analyze(result, rows, bootstrap_seed=11)

    assert report["gates"][gate] is False
    assert report["promotion_passed"] is False


def test_v4_analyzer_rejects_bootstrap_seed_mismatch():
    result, rows = _fixture(bootstrap_seed=11)

    with pytest.raises(ValueError, match="bootstrap seeds"):
        analyze(result, rows, bootstrap_seed=12)
