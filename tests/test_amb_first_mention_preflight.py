"""Unit tests for the judge-free event-ordering selector preflights.

These cover the pure, model-free parts: BEAM turn splitting, query parsing,
the selectors over synthetic candidates, and session splitting. The scored
preflight itself needs the BEAM corpus and an engine store and is run by
hand; its results are banked as artifacts and summarised in
``benchmarks/amb/FIRST_MENTION_SELECTOR_PREFLIGHT.md``.
"""

from __future__ import annotations

import pytest

from benchmarks.amb.first_mention_preflight import (
    cosine,
    focus_of,
    requested_count,
    sel_chrono_stratified,
    sel_cluster_first,
    sel_first_mention,
    sel_relevance_topk,
    split_turns,
)
from benchmarks.amb.session_stratified_preflight import sessions_of


SESSION = (
    "[March-15-2024 | Turn 0] User: I'm planning a budget tracker.\n\n"
    "[March-15-2024 | Turn 1] Assistant: Great, let's start.\n\n"
    "[March-15-2024 | Turn 2] User: First I need authentication."
)


def test_split_turns_keeps_role_turn_and_date():
    turns = split_turns(SESSION)
    assert [(t["role"], t["turn"]) for t in turns] == [("user", 0), ("assistant", 1), ("user", 2)]
    assert all(t["created_at"] is not None for t in turns)
    assert turns[0]["created_at"] == turns[2]["created_at"]


def test_split_turns_ignores_bracket_lookalikes_in_bodies():
    body = "[March-15-2024 | Turn 0] User: see [docs](x) and arr[3] then [Turn 9 was fun]"
    turns = split_turns(body)
    assert [t["turn"] for t in turns] == [0]


@pytest.mark.parametrize(
    "query, n",
    [
        ("... in order? Mention ONLY and ONLY three items.", 3),
        ("... Mention ONLY and ONLY 5 items.", 5),
        ("Walk me through the order with no count stated.", None),
    ],
)
def test_requested_count(query, n):
    assert requested_count(query) == n


def test_focus_of_strips_the_ordering_template():
    q = (
        "Can you list the order in which I brought up different aspects of developing my "
        "personal budget tracker throughout our conversations, in order? Mention ONLY and ONLY three items."
    )
    assert focus_of(q) == "aspects of developing my personal budget tracker"


def _cands():
    # turn, score, embedding: turns 1/2 near-duplicates, 3 novel, 4 duplicate of 3, 5 novel low-score
    return [
        {"turn": 1, "score": 1.0, "tokens": 10, "emb": [1.0, 0.0, 0.0]},
        {"turn": 2, "score": 0.9, "tokens": 10, "emb": [0.99, 0.1, 0.0]},
        {"turn": 3, "score": 0.8, "tokens": 10, "emb": [0.0, 1.0, 0.0]},
        {"turn": 4, "score": 0.7, "tokens": 10, "emb": [0.0, 0.99, 0.1]},
        {"turn": 5, "score": 0.1, "tokens": 10, "emb": [0.0, 0.0, 1.0]},
    ]


def test_first_mention_keeps_the_earliest_of_each_near_duplicate_group():
    kept = sel_first_mention(_cands(), 3, {"floor": 0.0, "theta": 0.9, "cap_mult": 1})
    assert [c["turn"] for c in kept] == [1, 3, 5]


def test_first_mention_floor_drops_weak_turns():
    kept = sel_first_mention(_cands(), 3, {"floor": 0.5, "theta": 0.9, "cap_mult": 1})
    assert [c["turn"] for c in kept] == [1, 3]


def test_cluster_first_returns_earliest_per_cluster_chronologically():
    kept = sel_cluster_first(_cands(), 2, {"top_m": 4, "theta": 0.9, "cap_mult": 1})
    assert [c["turn"] for c in kept] == [1, 3]


def test_relevance_topk_is_budget_bound():
    kept = sel_relevance_topk(_cands(), 3, {"budget_tokens": 25})
    assert [c["turn"] for c in kept] == [1, 2, 3]


def test_chrono_stratified_spreads_evenly():
    kept = sel_chrono_stratified(_cands(), 3, {"floor": 0.0, "cap_mult": 1})
    assert [c["turn"] for c in kept] == [1, 3, 5]


def test_cosine_basics():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)


def test_sessions_split_on_header_date():
    docs = [
        {"id": "1_s0_0", "content": SESSION},
        {"id": "1_s1_0", "content": "[April-02-2024 | Turn 3] User: Now deployment.\n\n[April-02-2024 | Turn 4] Assistant: ok"},
    ]
    sessions = sessions_of(docs)
    assert [[t["turn"] for t in s] for s in sessions] == [[0, 1, 2], [3, 4]]


# ── preflight 3: lexical openers / TextTiling ──────────────────────────

from benchmarks.amb.lexical_opener_preflight import (  # noqa: E402
    content_terms,
    opener_scores,
    tiling_depths,
    user_turns_of,
)


def test_content_terms_drop_header_and_stopwords():
    terms = content_terms("[March-15-2024 | Turn 2] User: Sure, let's build the authentication module first.")
    assert "authentication" in terms and "module" in terms and "build" in terms
    assert "turn" not in terms and "user" not in terms and "first" not in terms


def _turns():
    docs = [{
        "id": "1_s0_0",
        "content": (
            "[March-15-2024 | Turn 0] User: I want authentication with sessions.\n\n"
            "[March-15-2024 | Turn 1] Assistant: ok\n\n"
            "[March-15-2024 | Turn 2] User: authentication sessions again please.\n\n"
            "[March-15-2024 | Turn 3] Assistant: ok\n\n"
            "[March-15-2024 | Turn 4] User: authentication sessions once more.\n\n"
            "[March-15-2024 | Turn 5] Assistant: ok\n\n"
            "[March-15-2024 | Turn 6] User: Now deployment pipelines and docker.\n\n"
            "[March-15-2024 | Turn 7] Assistant: ok\n\n"
            "[March-15-2024 | Turn 8] User: deployment pipelines docker again."
        ),
    }]
    return user_turns_of(docs)


def test_opener_scores_credit_the_first_turn_of_a_recurring_term():
    turns = _turns()
    scores = opener_scores(turns, min_recur=2)
    # "authentication"/"sessions" recur in two later user turns → turn 0 opens two threads
    assert scores[0] == 2.0
    # deployment/pipelines/docker recur only once more → below min_recur=2
    assert scores.get(6, 0.0) == 0.0
    assert opener_scores(turns, min_recur=1)[6] == 3.0


def test_tiling_depth_is_highest_at_the_topic_shift():
    turns = _turns()
    depths = tiling_depths(turns, window=1)
    shift = depths[6]
    assert shift > depths[2] and shift > depths[4] and shift > depths[8]
    assert depths[0] > shift, "the first turn always opens a tile"
