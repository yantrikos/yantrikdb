import pytest

from benchmarks.amb.analyze_facet_applicability_v5 import analyze_v5
from benchmarks.amb.paired_frozen_context_eval import _run_fingerprint


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
SEEDS = [20260828, 20260829, 20260830]


def _fixture(category_deltas=None):
    category_deltas = category_deltas or {"instruction_following": 0.1}
    pairs = []
    rows = []
    for category in CATEGORIES:
        for index in range(40):
            query_id = f"{category}-{index}"
            delta = category_deltas.get(category, 0.0)
            pairs.append({"query_id": query_id, "score_a": 0.5, "score_b": 0.5 + delta})
            rows.append({"query_id": query_id, "meta": {"question_category": category}})
    run_config = {
        "protocol": "paired-independent-mean-of-three-v1",
        "replicate_count": 3,
        "replicate_seeds": SEEDS,
        "model_seeds": SEEDS,
        "bootstrap_seed": 20260831,
    }
    result = {
        "bootstrap_seed": 20260831,
        "replicate_seeds": SEEDS,
        "model_seeds": SEEDS,
        "run_config": run_config,
        "run_fingerprint": _run_fingerprint(run_config),
        "summary": {"paired_bootstrap_seed": 20260831},
        "pairs": pairs,
    }
    return result, rows


def test_v5_analyzer_passes_and_records_default_on_finality():
    result, rows = _fixture()

    report = analyze_v5(result, rows)

    assert report["promotion_passed"] is True
    assert report["finality"] == "default-on-promotion"
    assert report["protocol"] == "facet-applicability-v5-final-power-analysis-v1"


def test_v5_analyzer_failure_is_terminal_opt_in():
    result, rows = _fixture({"instruction_following": 0.04})

    report = analyze_v5(result, rows)

    assert report["promotion_passed"] is False
    assert report["finality"] == "terminal-opt-in"


def test_v5_analyzer_rejects_mutated_combined_config():
    result, rows = _fixture()
    result["run_config"]["bootstrap_seed"] = 7

    with pytest.raises(ValueError, match="fingerprint"):
        analyze_v5(result, rows)
