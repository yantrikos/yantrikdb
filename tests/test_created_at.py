"""Wrapper-level `created_at` tests (historical import, 0.14).

Engine semantics (row triple, digest participation, scoring clamp) are
pinned by the Rust suite; these pin the PYTHON WRAPPER's plumbing — that
the kwarg reaches the engine on every write surface it is exposed on, and
that the value is visible THROUGH RECALL rather than only in SQL.

That last distinction is the point: recall scores from the in-memory
scoring cache, not from the `memories` row, so the two can silently
disagree. `record_batch` shipped the correct row while its cache insert
still stamped `now()` — SQL-reading assertions all passed and every
through-recall consumer (decay, recency, recall_as_of) saw today.
"""

import math
import time

import pytest

from yantrikdb import YantrikDB

DIM = 8
JAN = 1_735_700_000.0  # 2025-01-01
MAR = 1_740_800_000.0  # 2025-03-01
APR = 1_743_500_000.0  # 2025-04-01


def _vec(seed: float) -> list[float]:
    raw = [math.sin(seed * 1.7 + i * 2.3) + math.cos(seed * 0.3 + i * 3.1) for i in range(DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    if norm < 1e-9:
        raw[0] = 1.0
        norm = 1.0
    return [x / norm for x in raw]


@pytest.fixture
def db():
    engine = YantrikDB(db_path=":memory:", embedding_dim=DIM)
    yield engine
    engine.close()


def _created_at_by_rid(db, query_seed: float, top_k: int = 20) -> dict[str, float]:
    hits = db.recall(query_embedding=_vec(query_seed), top_k=top_k, skip_reinforce=True)
    return {h["rid"]: h["created_at"] for h in hits}


def test_record_carries_created_at_through_recall(db):
    historical = db.record("historical", embedding=_vec(1.0), created_at=JAN)
    present = db.record("present", embedding=_vec(1.05))

    seen = _created_at_by_rid(db, 1.0)
    assert seen[historical] == JAN
    assert abs(seen[present] - time.time()) < 60


def test_record_batch_carries_per_item_created_at_through_recall(db):
    rids = db.record_batch([
        {"text": "batch historical", "embedding": _vec(1.0), "created_at": JAN},
        {"text": "batch present", "embedding": _vec(1.05)},
    ])

    seen = _created_at_by_rid(db, 1.0)
    assert seen[rids[0]] == JAN, "batch event time did not reach the recall path"
    assert abs(seen[rids[1]] - time.time()) < 60


def test_recall_as_of_filters_on_imported_event_times(db):
    old = db.record("works at the observatory", embedding=_vec(1.0), created_at=JAN)
    new = db.record("moved to the planetarium", embedding=_vec(1.02), created_at=APR)

    as_of_march = [h["rid"] for h in db.recall_as_of(MAR, query_embedding=_vec(1.0), top_k=10)]
    assert old in as_of_march
    assert new not in as_of_march, "a later record was visible in the past"

    today = [h["rid"] for h in db.recall_as_of(time.time(), query_embedding=_vec(1.0), top_k=10)]
    assert old in today and new in today


def test_omitting_created_at_stamps_now(db):
    before = time.time()
    rid = db.record("no explicit event time", embedding=_vec(2.0))
    after = time.time()
    assert before - 1 <= _created_at_by_rid(db, 2.0)[rid] <= after + 1


def test_future_created_at_does_not_amplify_ranking(db):
    """A future-dated decoy must score as 'new', never as an amplifier.

    decay is `importance * 2^(-elapsed/half_life)` and recency `e^(-age/7d)`;
    both grow without bound for a negative age, which caller-supplied
    timestamps made reachable. Clamped at the scoring boundary.
    """
    target = db.record("the target memory about the launch", embedding=_vec(1.0))
    db.record(
        "unrelated future decoy",
        embedding=_vec(9.0),
        created_at=time.time() + 365 * 86400,
    )
    top = db.recall(query_embedding=_vec(1.0), top_k=1, skip_reinforce=True)
    assert top[0]["rid"] == target


def test_non_finite_created_at_is_refused(db):
    for bad in (float("nan"), float("inf")):
        with pytest.raises(Exception):
            db.record("bad", embedding=_vec(1.0), created_at=bad)
    assert db.stats()["active_memories"] == 0


def test_redated_keyed_write_is_a_different_write(db):
    def write(ts):
        return db.record(
            "the launch happened",
            embedding=_vec(1.0),
            idempotency_key="launch-evt-1",
            created_at=ts,
        )

    rid = write(JAN)
    assert write(JAN) == rid, "an identical retry must return the original rid"
    with pytest.raises(Exception):
        write(JAN + 1.0)
