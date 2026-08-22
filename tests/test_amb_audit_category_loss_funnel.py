from benchmarks.amb.audit_category_loss_funnel import (
    analyze,
    behavior_target_funnel,
    ceiling_estimate,
    reference_items,
)


def _row(query_id, category, query, answer, gold, context, rubric=None):
    return {
        "query_id": query_id,
        "query": query,
        "answer": answer,
        "gold_answers": gold,
        "context": context,
        "score": 0.0,
        "meta": {
            "question_category": category,
            "conversation_id": "1",
            "rubric": rubric or [],
        },
    }


def test_reference_items_use_rubric_claims_for_tail_categories():
    for category in (
        "information_extraction",
        "instruction_following",
        "preference_following",
    ):
        row = _row(
            f"1_{category}_0",
            category,
            "Question",
            "Answer",
            ["coarse gold"],
            "context",
            [
                "LLM response should contain: first claim",
                "LLM response should state: second claim",
                "LLM response should mention: third claim",
            ],
        )
        assert reference_items(row) == [
            "first claim",
            "second claim",
            "third claim",
        ]

    contradiction = _row(
        "1_contradiction_resolution_0",
        "contradiction_resolution",
        "Question",
        "Answer",
        [],
        "context",
        [
            "LLM response should state: conflicting information",
            "LLM response should mention: first evidence claim",
            "LLM response should mention: second evidence claim",
            "LLM response should mention: which statement is correct?",
        ],
    )
    assert reference_items(contradiction) == [
        "first evidence claim",
        "second evidence claim",
    ]


def test_behavior_target_funnel_uses_canonical_instruction():
    row = _row(
        "1_instruction_following_0",
        "instruction_following",
        "Question",
        "Answer",
        [],
        "## Memory 1\nAlways use APA 7th edition citations when formatting references.",
    )
    row["meta"]["instruction_being_tested"] = (
        "Always use APA 7th edition citations when formatting references."
    )

    target = behavior_target_funnel(
        row, "User: Always use APA 7th edition citations when formatting references."
    )

    assert target is not None
    assert target["stage"] == "target_retrieved"
    assert target["source_coverage"] == 1.0


def test_analyze_separates_retrieval_answer_label_and_provenance_losses():
    rows = [
        _row(
            "1_summarization_0",
            "summarization",
            "Summarize the project",
            "alpha",
            [],
            "## Memory 1\n[Speaker: User] alpha beta",
            ["LLM response should contain: alpha beta gamma"],
        ),
        _row(
            "1_knowledge_update_0",
            "knowledge_update",
            "How long now?",
            "6-9 months",
            ["5-7 months"],
            "## Memory 1\n[Speaker: User] 5-7 months",
        ),
        _row(
            "1_multi_session_reasoning_0",
            "multi_session_reasoning",
            "How many areas?",
            "28",
            ["Four"],
            "## Memory 1\n[Speaker: User] alpha beta",
        ),
        _row(
            "1_abstention_0",
            "abstention",
            "What techniques did Shawn recommend?",
            "Authenticity and weaving threads",
            ["There is no information."],
            "## Memory 1\n[Speaker: User] Shawn discussed storytelling impact.\n\n"
            "## Memory 2\n[Speaker: Assistant] Try authenticity and weaving threads.",
        ),
    ]
    sources = {
        "1": "[Turn 2] User: alpha beta gamma and 5-7 months\n"
        "[Turn 4] User: the revised estimate is 6-9 months"
    }
    conversations = {
        "1": {
            "chat": [
                {"id": 2, "role": "user", "content": "alpha beta gamma and 5-7 months"},
                {"id": 4, "role": "user", "content": "the revised estimate is 6-9 months"},
            ]
        }
    }

    report = analyze(
        {"results": rows},
        sources,
        conversations,
        {
            "summarization",
            "knowledge_update",
            "multi_session_reasoning",
            "abstention",
        },
    )

    by_id = {row["query_id"]: row for row in report["results"]}
    assert by_id["1_summarization_0"]["items"][0]["stage"] == "retrieval_loss"
    assert (
        by_id["1_knowledge_update_0"]["knowledge_update"]["verdict"]
        == "gold_precedes_later_prediction"
    )
    assert (
        by_id["1_multi_session_reasoning_0"]["items"][0]["stage"]
        == "synthesis_required"
    )
    provenance = by_id["1_abstention_0"]["speaker_provenance"]
    assert provenance["assistant_only_dominant"] is True
    assert {"authenticity", "weaving"} <= set(provenance["assistant_only_answer_tokens"])

    summaries = report["categories"]
    assert summaries["summarization"]["item_stages"] == {"retrieval_loss": 1}
    assert summaries["abstention"]["reference_items"] == 0
    assert summaries["abstention"]["mean_source_coverage"] is None
    assert summaries["abstention"]["zero_rows_assistant_only_dominant"] == 1
    assert summaries["knowledge_update"]["knowledge_update_zero_verdicts"] == {
        "gold_precedes_later_prediction": 1
    }
    assert summaries["knowledge_update"]["attribution"]["primary"] == "benchmark_label"
    assert summaries["knowledge_update"]["attribution"]["label"]["count"] == 1
    assert summaries["summarization"]["attribution"]["reader"]["count"] == 0
    assert summaries["abstention"]["attribution"]["ours"] == {
        "count": 1,
        "denominator": 1,
        "unit": "zero_score_rows",
    }
    assert summaries["abstention"]["equal_weight_overall_points_lost"] == 10.0


def test_ceiling_estimate_conserves_full_line_loss():
    means = {
        "abstention": 0.675,
        "contradiction_resolution": 0.828125,
        "event_ordering": 0.2907143,
        "information_extraction": 0.7734375,
        "instruction_following": 0.7875,
        "knowledge_update": 0.63125,
        "multi_session_reasoning": 0.6108333,
        "preference_following": 0.9125,
        "summarization": 0.5929911,
        "temporal_reasoning": 0.4125,
    }
    payload = {
        "results": [
            {"score": score, "meta": {"question_category": category}}
            for category, score in means.items()
        ]
    }
    summaries = {
        "summarization": {
            "attribution": {
                "ours": {"count": 22, "denominator": 195},
                "reader": {"count": 131, "denominator": 195},
            }
        },
        "knowledge_update": {
            "attribution": {"label": {"count": 11, "denominator": 14}}
        },
        "multi_session_reasoning": {
            "attribution": {
                "reader": {"count": 30, "denominator": 40},
                "synthesis_required": {"count": 2, "denominator": 40},
            }
        },
        "abstention": {
            "attribution": {
                "ours": {"count": 7, "denominator": 13},
                "reader": {"count": 6, "denominator": 13},
            }
        },
    }

    estimate = ceiling_estimate(payload, summaries)

    assert estimate["complete_equal_weight_ten_category_line"] is True
    assert estimate["baseline_score_percent"] == 65.148512
    assert estimate["points_required_to_reach_90"] == 24.851488
    assert estimate["optimistic_ceiling_percent"] == 96.908095
    assert estimate["reader_shaping_recovery_sensitivity"] == {
        "0.0": 89.75508,
        "0.5": 93.331588,
        "0.7": 94.762191,
        "1.0": 96.908095,
    }
    assert estimate["bucket_conservation_delta"] == 0.0
    assert abs(
        sum(estimate["buckets"].values()) - estimate["total_points_lost"]
    ) <= 2e-6

    full_summaries = {
        **summaries,
        "contradiction_resolution": {
            "attribution": {
                "ours": {"count": 1, "denominator": 80},
                "reader": {"count": 32, "denominator": 80},
            }
        },
        "information_extraction": {
            "attribution": {
                "ours": {"count": 4, "denominator": 92},
                "reader": {"count": 41, "denominator": 92},
            }
        },
        "instruction_following": {
            "attribution": {
                "ours": {"count": 3.5, "denominator": 8.5},
                "reader": {"count": 5.0, "denominator": 8.5},
            }
        },
        "preference_following": {
            "attribution": {
                "ours": {"count": 1.0, "denominator": 3.5},
                "reader": {"count": 2.5, "denominator": 3.5},
            }
        },
    }
    full_estimate = ceiling_estimate(payload, full_summaries)
    assert full_estimate["buckets"] == {
        "dead_or_benchmark_integrity": 3.091905,
        "reader_via_context_shaping": 10.725196,
        "ours_direct_engine": 16.422036,
        "undiagnosed_tail": 0,
        "audited_residual": 4.612351,
    }
    assert full_estimate["reader_shaping_recovery_sensitivity"]["0.5"] == 91.545497
