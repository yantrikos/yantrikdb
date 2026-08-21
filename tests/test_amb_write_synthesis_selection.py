from benchmarks.amb.write_synthesis_selection import (
    beam_header_turns,
    cap_temporal_span_items,
    deduplicate_thread_items,
    first_beam_turn,
    ground_ordered_items_to_candidates,
    ground_synthesized_item_provenance,
    is_relationship_role_timeline,
    is_relationship_support_query,
    merge_organizer_rollup_shards,
    merge_synthesized_evidence_sets,
    order_synthesis_candidates_for_selection,
    select_entity_timeline_children,
    select_relationship_support_children,
    synthesized_item_evidence_sets,
    validate_synthesized_item_support_quotes,
)


def test_support_quote_must_be_literal_and_substantive():
    items = [
        {
            "id": "I001",
            "evidence_ids": ["B001"],
            "support_block_id": "B001",
            "support_quote": (
                "Matthew offered tips on adapting my statements for grant "
                "applications"
            ),
        },
        {
            "id": "I002",
            "evidence_ids": ["B001"],
            "support_block_id": "B001",
            "support_quote": (
                "Wendy sent a care package with local spices and notes"
            ),
        },
        {
            "id": "I003",
            "evidence_ids": ["B001"],
            "support_block_id": "B001",
            "support_quote": "[Turn 164] User:",
        },
        {
            "id": "I004",
            "evidence_ids": ["B001"],
            "support_block_id": "B001",
            "support_quote": " ".join(["word"] * 26),
        },
    ]

    validated, events = validate_synthesized_item_support_quotes(
        items,
        {
            "B001": (
                "[Turn 164] User: Matthew offered tips on adapting my "
                "statements for grant applications and global opportunities."
            )
        },
    )

    assert [item["id"] for item in validated] == ["I001"]
    assert [event["status"] for event in events] == [
        "rejected_quote_not_in_evidence",
        "rejected_missing_substantive_quote",
        "rejected_missing_substantive_quote",
    ]


def test_support_quote_can_correct_claimed_block_within_evidence_set():
    item = {
        "id": "I001",
        "evidence_ids": ["B001", "missing", "B002"],
        "support_block_id": "B001",
        "support_quote": "career gap from 2022 could use some fine-tuning",
    }

    validated, events = validate_synthesized_item_support_quotes(
        [item],
        {
            "B001": "A different source block with unrelated details.",
            "B002": "The career gap from 2022 could use some fine-tuning.",
        },
    )

    assert validated[0]["support_block_id"] == "B002"
    assert validated[0]["evidence_ids"] == ["B002"]
    assert validated[0]["chronology_evidence_ids"] == ["B001", "B002"]
    assert events == [
        {
            "status": "dropped_invalid_evidence",
            "item_id": "I001",
            "evidence_ids": ["missing"],
        },
        {
            "status": "unverified_chronology_evidence",
            "item_id": "I001",
            "evidence_ids": ["B001"],
        },
        {
            "status": "corrected_support_block",
            "item_id": "I001",
            "before": "B001",
            "after": "B002",
        },
    ]


def test_quote_support_keeps_separate_earlier_chronology():
    item = {
        "id": "I001",
        "evidence_ids": ["B002"],
        "chronology_evidence_ids": ["B001", "B002"],
        "first_mention_block_id": "B002",
        "first_mention_turn": 20,
        "first_mention_date": "2024-02-01",
    }

    grounded, _ = ground_synthesized_item_provenance(
        [item],
        {
            "B001": ("2024-01-01", 10, 2),
            "B002": ("2024-02-01", 20, 1),
        },
        {"B001": "2024-01-01", "B002": "2024-02-01"},
    )

    assert grounded[0]["evidence_ids"] == ["B002"]
    assert grounded[0]["chronology_evidence_ids"] == ["B001", "B002"]
    assert grounded[0]["first_mention_block_id"] == "B001"
    assert grounded[0]["first_mention_turn"] == 10


def test_synthesized_evidence_sets_preserve_separate_chronology():
    evidence_ids, chronology_ids = synthesized_item_evidence_sets({
        "evidence_ids": ["B002", "B002"],
        "chronology_evidence_ids": ["B001", "B002", "B001"],
    })

    assert evidence_ids == ["B002"]
    assert chronology_ids == ["B001", "B002"]


def test_rollup_evidence_union_preserves_both_channels():
    evidence_ids, chronology_ids = merge_synthesized_evidence_sets([
        {
            "evidence_ids": ["B002"],
            "chronology_evidence_ids": ["B001", "B002"],
        },
        {"evidence_ids": ["B004"]},
    ])

    assert evidence_ids == ["B002", "B004"]
    assert chronology_ids == ["B001", "B002", "B004"]


def test_ordered_items_inherit_provenance_from_linked_candidates():
    ordered, events = ground_ordered_items_to_candidates(
        [{
            "id": "F001",
            "item": "Selected milestone",
            "source_item_ids": ["I002", "missing"],
            "evidence_ids": ["invented-valid-block"],
            "chronology_evidence_ids": ["invented-valid-block"],
        }],
        [
            {"id": "I001", "evidence_ids": ["B001"]},
            {
                "id": "I002",
                "evidence_ids": ["B003"],
                "chronology_evidence_ids": ["B002", "B003"],
            },
        ],
    )

    assert ordered[0]["source_item_ids"] == ["I002"]
    assert ordered[0]["evidence_ids"] == ["B003"]
    assert ordered[0]["chronology_evidence_ids"] == ["B002", "B003"]
    assert events == [{
        "status": "dropped_invalid_source_candidates",
        "item_id": "F001",
        "source_item_ids": ["missing"],
    }]


def test_ordered_items_without_valid_candidate_link_are_rejected():
    ordered, events = ground_ordered_items_to_candidates(
        [{"id": "F001", "item": "Ungrounded", "source_item_ids": ["missing"]}],
        [{"id": "I001", "evidence_ids": ["B001"]}],
    )

    assert ordered == []
    assert [event["status"] for event in events] == [
        "dropped_invalid_source_candidates",
        "rejected_missing_source_candidate",
    ]


def test_selection_candidates_can_be_presented_by_relevance_without_losing_ties():
    items = [
        {"id": "I001", "best_retrieval_rank": 20, "first_mention_turn": 2},
        {"id": "I002", "best_retrieval_rank": 3, "first_mention_turn": 8},
        {"id": "I003", "best_retrieval_rank": 3, "first_mention_turn": 4},
        {"id": "I004", "best_retrieval_rank": "bad", "first_mention_turn": 1},
    ]

    assert order_synthesis_candidates_for_selection(items, "chronological") == items
    assert [item["id"] for item in order_synthesis_candidates_for_selection(
        items, "relevance"
    )] == ["I003", "I002", "I001", "I004"]


def test_synthesis_provenance_is_grounded_in_earliest_cited_block():
    items = [{
        "id": "I001",
        "item": "A grounded event with invented model provenance.",
        "first_mention_date": "2024-09-09",
        "first_mention_turn": 164,
        "first_mention_block_id": "B099",
        "evidence_ids": ["B023", "B004", "missing"],
    }]

    grounded, events = ground_synthesized_item_provenance(
        items,
        {
            "B004": ("2024-02-01", 60, 7),
            "B023": ("2024-06-01", 166, 22),
        },
        {"B004": "2024-02-01", "B023": "2024-06-01"},
    )

    assert grounded[0]["first_mention_block_id"] == "B004"
    assert grounded[0]["first_mention_turn"] == 60
    assert grounded[0]["first_mention_date"] == "2024-02-01"
    assert grounded[0]["evidence_ids"] == ["B023", "B004"]
    assert events == [
        {
            "status": "dropped_invalid_evidence",
            "item_id": "I001",
            "invalid_evidence_ids": ["missing"],
        },
        {
            "status": "corrected",
            "item_id": "I001",
            "before": {
                "first_mention_block_id": "B099",
                "first_mention_turn": 164,
                "first_mention_date": "2024-09-09",
            },
            "after": {
                "first_mention_block_id": "B004",
                "first_mention_turn": 60,
                "first_mention_date": "2024-02-01",
            },
        },
    ]


def test_synthesis_provenance_rejects_items_without_valid_evidence():
    item = {
        "first_mention_date": "2024-01-02",
        "first_mention_turn": 12,
        "first_mention_block_id": "B012",
        "evidence_ids": ["missing"],
    }

    grounded, events = ground_synthesized_item_provenance(
        [item], {"B001": ("2024-01-01", 1, 1)}, {"B001": "2024-01-01"}
    )

    assert grounded == []
    assert events == [{
        "status": "rejected_invalid_evidence",
        "item_id": "",
        "invalid_evidence_ids": ["missing"],
    }]


def test_first_turn_uses_beam_headers_not_body_mentions():
    text = (
        "[Turn 60] User: I compared this with the advice from Turn 4.\n\n"
        "[June-14-2024 | Turn 62] Assistant: Follow-up."
    )

    assert first_beam_turn(text) == 60
    assert first_beam_turn(
        "This body mentions Turn 2 without a BEAM header."
    ) is None
    assert first_beam_turn(
        "This body cites [Turn 2] User: as prose, not a line header."
    ) is None
    assert beam_header_turns(text) == [60, 62]


def test_temporal_span_caps_apply_before_flattening():
    extracted = {
        "q1_items": ["early-1", "early-2", "early-overflow"],
        "q2_items": ["middle-1", "middle-2", "middle-overflow"],
        "q3_items": ["late-1", "late-2", "late-overflow"],
        "q4_items": "malformed",
    }

    capped = cap_temporal_span_items(
        extracted,
        ["q1_items", "q2_items", "q3_items", "q4_items"],
        per_span_target=2,
    )

    assert capped == {
        "q1_items": ["early-1", "early-2"],
        "q2_items": ["middle-1", "middle-2"],
        "q3_items": ["late-1", "late-2"],
        "q4_items": [],
    }


def _hit(
    rid: str,
    turn: int,
    text: str,
    score: float = 0.0,
    selection_entity: str | None = None,
) -> dict:
    metadata = {
        "first_mention_at": float(turn),
        "first_mention_turn": turn,
    }
    if selection_entity:
        metadata["selection_entities"] = [selection_entity]
    return {
        "rid": rid,
        "text": text,
        "score": score,
        "metadata": metadata,
    }


def _people(text: str) -> set[str]:
    return {name for name in ("alex", "blair") if name in text.casefold()}


def test_organizer_shards_merge_by_logical_label_without_losing_children():
    rollups = [
        {
            "rid": "shard-a",
            "score": 0.2,
            "text": "Topic trajectory: Writing workflow. First shard.",
            "metadata": {
                "organizer_label": "Writing Workflow",
                "child_rids": ["a", "shared"],
                "first_mention_at": 20.0,
                "evidence_span_end_at": 30.0,
            },
        },
        {
            "rid": "shard-b",
            "score": 0.4,
            "text": "Topic trajectory: Writing workflow. Better shard.",
            "metadata": {
                "organizer_label": " writing   workflow ",
                "child_rids": ["shared", "b"],
                "first_mention_at": 10.0,
                "evidence_span_end_at": 40.0,
            },
        },
    ]

    merged = merge_organizer_rollup_shards(rollups)

    assert len(merged) == 1
    assert merged[0]["rid"] == "shard-b"
    assert merged[0]["score"] == 0.4
    assert merged[0]["metadata"]["child_rids"] == ["a", "shared", "b"]
    assert merged[0]["metadata"]["first_mention_at"] == 10.0
    assert merged[0]["metadata"]["evidence_span_end_at"] == 40.0
    assert merged[0]["metadata"]["organizer_shard_rids"] == [
        "shard-a",
        "shard-b",
    ]


def test_organizer_shards_without_labels_remain_distinct():
    rollups = [
        {"rid": "a", "metadata": {"child_rids": ["x"]}},
        {"rid": "b", "metadata": {"child_rids": ["y"]}},
    ]

    assert len(merge_organizer_rollup_shards(rollups)) == 2


def test_relationship_support_intent_requires_both_concepts():
    assert is_relationship_support_query(
        "How did my family support me across our conversations?"
    )
    assert not is_relationship_support_query(
        "How did my family appear across our conversations?"
    )
    assert not is_relationship_support_query(
        "Who helped with my professional feedback?"
    )


def test_relationship_role_timeline_filter_is_narrow():
    assert is_relationship_role_timeline("Alex, my mom, offered perspective")
    assert is_relationship_role_timeline("I was dating Blair at the time")
    assert not is_relationship_role_timeline(
        "Casey, Devon, and Ellis gave professional feedback"
    )


def test_relationship_support_selector_prefers_actions_and_restores_order():
    hits = [
        _hit("a10", 10, "They told me to foreground my motivation", 0.20, "alex"),
        _hit("a30", 30, "They wrote a note encouraging persistence", 0.19, "alex"),
        _hit("a50", 50, "They offered practical support during revisions", 0.28, "alex"),
        _hit("a70", 70, "They reminded me to maintain balance", 0.22, "alex"),
        _hit("b5", 5, "They were supportive during the process", 0.17, "blair"),
        _hit("b20", 20, "They helped me prepare a short presentation", 0.16, "blair"),
        _hit("b25", 25, "They discussed limiting social outings", 0.16, "blair"),
        _hit("b35", 35, "They expressed concern in check-ins", 0.15, "blair"),
        _hit("b40", 40, "We practiced difficult answers", 0.10, "blair"),
        _hit("b45", 45, "They marked our anniversary", 0.12, "blair"),
        _hit("b55", 55, "They discussed voice consistency", 0.18, "blair"),
        _hit("b60", 60, "They planned a future trip", 0.30, "blair"),
        _hit("b65", 65, "They joined weekly video calls", 0.17, "blair"),
    ]

    selected = select_relationship_support_children(hits, 5, _people)

    assert [hit["rid"] for hit in selected] == [
        "a10",
        "b20",
        "a30",
        "a50",
        "a70",
    ]


def test_relationship_support_selector_is_noop_without_a_target_count():
    hits = [_hit("later", 2, "Alex helped"), _hit("earlier", 1, "Blair helped")]
    assert select_relationship_support_children(hits, None, _people) is hits


def test_rare_contributor_coverage_beats_raw_relevance_for_equal_actions():
    hits = [
        _hit("a1", 1, "Alex helped with one revision", 0.99),
        _hit("a2", 2, "Alex helped with another revision", 0.98),
        _hit("a3", 3, "Alex helped with a third revision", 0.97),
        _hit("b4", 4, "Blair helped with a separate concern", 0.01),
    ]

    selected = select_relationship_support_children(hits, 1, _people)

    assert [hit["rid"] for hit in selected] == ["b4"]


def test_selected_children_are_presented_by_first_mention():
    hits = [
        _hit("late", 30, "Alex helped later", 0.90),
        _hit("early", 10, "Blair helped first", 0.80),
        _hit("middle", 20, "Blair helped next", 0.70),
    ]

    selected = select_relationship_support_children(hits, 2, _people)

    assert [hit["rid"] for hit in selected] == ["early", "late"]


def test_thread_dedup_keeps_distinct_facts_from_one_evidence_chunk():
    items = [
        {
            "axis": "contributed",
            "item": "Alex chose a lightweight editor because of the budget.",
            "evidence_ids": ["E0004"],
        },
        {
            "axis": "contributed",
            "item": "Alex reduced passive voice after Blair shared a checklist.",
            "evidence_ids": ["E0004"],
        },
    ]

    assert deduplicate_thread_items(items) == items


def test_thread_dedup_removes_only_cross_axis_paraphrases():
    contributed = {
        "axis": "contributed",
        "item": "Alex reduced passive voice after Blair shared an editing checklist.",
        "evidence_ids": ["E0004"],
    }
    paraphrase = {
        "axis": "asked",
        "item": "Alex asked how to reduce passive voice further using Blair's editing checklist.",
        "evidence_ids": ["E0004"],
    }
    distinct = {
        "axis": "asked",
        "item": "Alex asked whether the current software still fit the project budget.",
        "evidence_ids": ["E0004"],
    }

    selected = deduplicate_thread_items([distinct, paraphrase, contributed])

    assert selected == [contributed, distinct]


def test_entity_timeline_prefers_anchor_led_relation_over_late_compound():
    hits = [
        _hit("intro", 10, "Alex, who reviewed the opening, shared feedback."),
        _hit("checklist", 20, "The draft improved after Alex revealed a checklist."),
        _hit("collab", 30, "Work with Alex resolved three structural issues."),
        _hit("plan", 40, "Alex and I planned the next review."),
        _hit(
            "incidental",
            50,
            "Blair finished a separate review and later hosted a webinar with Alex.",
        ),
    ]

    selected = select_entity_timeline_children(hits, "Alex", 4, _people)

    assert [item["rid"] for item in selected] == [
        "intro", "checklist", "collab", "plan"
    ]
