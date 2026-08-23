from benchmarks.amb.analyze_standing_instruction_full400 import analyze


CATEGORIES = [
    "instruction_following",
    "event_ordering",
    "knowledge_update",
    "multi_session",
    "preference",
    "summarization",
    "temporal_reasoning",
    "entity_recall",
    "speaker_identification",
    "other",
]


def _fixture(category_deltas=None):
    category_deltas = category_deltas or {"instruction_following": 0.1}
    pairs = []
    rows = []
    for category in CATEGORIES:
        for index in range(40):
            query_id = f"{category}-{index}"
            delta = category_deltas.get(category, 0.0)
            pairs.append({"query_id": query_id, "score_a": 0.5, "score_b": 0.5 + delta})
            rows.append(
                {
                    "query_id": query_id,
                    "meta": {"question_category": category},
                }
            )
    return {"pairs": pairs}, rows


def test_analyze_passes_when_all_preregistered_gates_pass():
    result, rows = _fixture()

    report = analyze(result, rows, seed=7)

    assert report["promotion_passed"] is True
    assert all(report["gates"].values())
    assert report["instruction_following"]["mean_delta_b_minus_a"] >= 0.05


def test_analyze_rejects_a_harmful_non_instruction_category():
    result, rows = _fixture({"instruction_following": 0.1, "event_ordering": -0.05})

    report = analyze(result, rows, seed=7)

    assert report["gates"]["no_other_category_below_floor"] is False
    assert report["promotion_passed"] is False
