from benchmarks.amb.audit_category_loss_funnel import analyze


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
