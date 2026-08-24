import pytest

from benchmarks.amb.event_ordering_thread_v2 import (
    event_ordering_focus,
    is_event_ordering_chronology_query,
)


@pytest.mark.parametrize(
    "query, focus",
    [
        (
            "Can you list the order in which I brought up different aspects "
            "of refining my personal statement throughout our conversations "
            "in order? Mention only five items.",
            "refining my personal statement",
        ),
        (
            "Can you walk me through the order in which I brought up "
            "different aspects of my writing journey across our conversations?",
            "my writing journey",
        ),
        (
            "Can you list in order how I brought up different ways my family "
            "supported me in my personal statement throughout our chats?",
            "my family supported me in my personal statement",
        ),
        (
            "CAN YOU WALK ME THROUGH THE ORDER IN WHICH I BROUGHT UP SHARED "
            "ENTERTAINMENT INTERESTS WITH DOUGLAS DURING OUR CHATS?",
            "SHARED ENTERTAINMENT INTERESTS WITH DOUGLAS",
        ),
        (
            "Can you walk me through the order in which I brought up concerns "
            "related to my family's care in our conversations, in order?",
            "concerns related to my family's care",
        ),
    ],
)
def test_preregistered_predicate_and_focus(query, focus):
    assert is_event_ordering_chronology_query(query)
    assert event_ordering_focus(query) == focus


@pytest.mark.parametrize(
    "query",
    [
        "Can you list the order in which these events happened?",
        "What topics did I bring up throughout our conversations?",
        "Can you summarize what I brought up in order?",
        "Can you list our conversations in chronological order?",
        "",
    ],
)
def test_preregistered_predicate_rejects_near_misses(query):
    assert not is_event_ordering_chronology_query(query)
    assert event_ordering_focus(query) is None
