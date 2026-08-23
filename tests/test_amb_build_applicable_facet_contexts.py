from benchmarks.amb.build_applicable_facet_contexts import (
    PREDICATE_VERSION,
    facet_decision,
    parse_condition,
    query_shape,
    select_applicable_facets,
)


def _facet(rid, text, first=1.0):
    return {
        "rid": rid,
        "text": text,
        "turn": int(first),
        "first_mention_at": first,
    }


DATE_RULE = _facet(
    "date",
    'Always format dates as "Month Day, Year" when I ask about timeline details.',
)
PATENT_RULE = _facet(
    "patent",
    "Always provide detailed timelines when I ask about patent application processes.",
    2.0,
)
RESUME_RULE = _facet(
    "resume",
    "Always use structured bullet points with quantified achievements when I ask "
    "about resume formatting preferences.",
    3.0,
)
CONTENT_RULE = _facet(
    "content",
    "Always include cultural context when I ask about social norms.",
    4.0,
)


def test_parse_condition_preserves_action_and_scope():
    assert parse_condition(DATE_RULE["text"]) == (
        'Always format dates as "Month Day, Year"',
        "timeline details",
    )
    assert parse_condition("Always reply in French.") is None


def test_query_shape_disambiguates_date_request_from_mention_chronology():
    date = query_shape("When was the Montserrat Writers' Festival?")
    assert date["date_time_request"]
    assert not date["chronology_of_mentions"]

    chronology = query_shape(
        "List the order in which I brought up project topics throughout our "
        "conversations, in order."
    )
    assert chronology["chronology_of_mentions"]
    assert not chronology["date_time_request"]


def test_default_include_applies_outside_positive_form_conflict():
    decision = facet_decision("When was the festival?", DATE_RULE)
    assert decision["include"]
    assert decision["predicate_version"] == PREDICATE_VERSION

    unparsed = facet_decision(
        "List events in chronological sequence throughout our chats.",
        _facet("u", "Always reply in French."),
    )
    assert unparsed["include"]
    assert unparsed["reason"] == "default_include_unparsed"


def test_chronology_suppresses_unrequested_date_and_format_transforms():
    query = (
        "Can you list the order in which I brought up different aspects of my "
        "resume throughout our conversations in order?"
    )
    selected, decisions = select_applicable_facets(
        query, [DATE_RULE, PATENT_RULE, RESUME_RULE, CONTENT_RULE]
    )
    assert [facet["rid"] for facet in selected] == ["content"]
    assert {
        decision["reason"] for decision in decisions if not decision["include"]
    } == {
        "suppress_chronology_date_time_conflict",
        "suppress_chronology_formatting_conflict",
    }


def test_ordered_deadline_inverse_control_retains_date_rule():
    query = (
        "List the order in which I mentioned each deadline throughout our "
        "conversations, in order."
    )
    decision = facet_decision(query, DATE_RULE)
    assert decision["query_shape"]["chronology_of_mentions"]
    assert decision["query_shape"]["date_time_request"]
    assert decision["include"]
    assert decision["reason"] == "include_compatible_date_time"


def test_process_timeline_inverse_control_retains_matching_rule():
    query = (
        "Walk me through the order in which I brought up the different stages "
        "of my patent process throughout our conversations."
    )
    decision = facet_decision(query, PATENT_RULE)
    assert decision["query_shape"]["process_timeline_request"]
    assert decision["include"]
    assert decision["reason"] == "include_compatible_date_time"


def test_same_topic_does_not_override_answer_shape_conflict():
    query = (
        "List the order in which I brought up improvements to my resume "
        "throughout our conversations."
    )
    decision = facet_decision(query, RESUME_RULE)
    assert decision["scope_type"] == "formatting_structure"
    assert not decision["query_shape"]["formatting_request"]
    assert not decision["include"]
