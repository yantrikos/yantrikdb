from benchmarks.amb.analyze_standing_instruction_displacement import analyze


def _row(query_id, category, context, gold):
    return {
        "query_id": query_id,
        "context": context,
        "gold_answers": [gold],
        "meta": {"question_category": category, "rubric": [gold]},
    }


def test_analyze_finds_evicted_gold_evidence_and_labels_oracle():
    kept = "## Memory 1\nThe project uses SQLite.\n\n"
    removed = "## Memory 2\nThe launch date is Friday morning.\n\n"
    source = [
        _row("instruction", "instruction_following", kept + removed, "launch date Friday"),
        _row("summary", "summarization", kept + removed, "launch date Friday"),
    ]
    treatment = []
    for row in source:
        transformed = dict(row)
        transformed["context"] = "## Memory 0\nAlways be concise.\n\n" + kept
        transformed["standing_instruction_audit"] = {
            "reference_tokens": 20,
            "treatment_tokens": 18,
        }
        treatment.append(transformed)
    result = {
        "pairs": [
            {"query_id": "instruction", "score_a": 0.5, "score_b": 1.0},
            {"query_id": "summary", "score_a": 1.0, "score_b": 0.0},
        ]
    }

    report = analyze(source, treatment, result, lambda text: len(text.split()))

    assert report["overall"]["mean_displaced_blocks"] == 1
    assert report["overall"]["rows_with_removed_only_gold_bigrams"] == 2
    assert report["rows"][0]["removed_only_gold_bigrams"] == ["launch date"]
    assert report["category_oracle_selective_composition"][
        "mean_delta_selective_minus_control"
    ] == 0.25
    assert report["interpretation"]["promotion_evidence"] is False


def test_analyze_rejects_a_treatment_that_is_not_a_source_prefix():
    source = [_row("q", "summarization", "## Memory 1\nA\n\n## Memory 2\nB\n", "B")]
    treatment = [
        {
            **source[0],
            "context": "## Memory 0\nAlways concise.\n\n## Memory 2\nB\n",
        }
    ]
    result = {"pairs": [{"query_id": "q", "score_a": 0.0, "score_b": 0.0}]}

    try:
        analyze(source, treatment, result, len)
    except ValueError as error:
        assert "panel + source prefix" in str(error)
    else:
        raise AssertionError("expected source-prefix validation to fail")
