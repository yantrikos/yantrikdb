"""Wrapper-level idempotency tests (v0.10 4a.6c / 4a.6d).

The engine-side semantics (claims, digests, probes) are pinned by the Rust
suite; these tests pin the PYTHON WRAPPER's routing and refusals — the layer
pytest alone can exercise.
"""

import math

import pytest

from yantrikdb import YantrikDB

DIM = 8


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


class TestRecordIdempotency:
    def test_keyed_retry_returns_original_rid(self, db):
        kwargs = dict(
            text="keyed write",
            embedding=_vec(1.0),
            idempotency_key="k-1",
        )
        rid1 = db.record(**kwargs)
        rid2 = db.record(**kwargs)
        assert rid2 == rid1

    def test_same_key_different_payload_conflicts(self, db):
        db.record(text="first payload", embedding=_vec(1.0), idempotency_key="k-2")
        with pytest.raises(Exception, match="[Ii]dempoten"):
            db.record(text="DIFFERENT payload", embedding=_vec(2.0), idempotency_key="k-2")


class TestBatchIdempotency:
    def test_keyed_batch_retry_returns_original_rids(self, db):
        batch = [
            {"text": "batch one", "embedding": _vec(1.0), "idempotency_key": "b-1"},
            {"text": "batch two", "embedding": _vec(2.0), "idempotency_key": "b-2"},
        ]
        rids1 = db.record_batch(batch)
        rids2 = db.record_batch(batch)
        assert rids2 == rids1

    def test_unkeyed_items_need_no_embedding_refusal(self, db):
        # Unkeyed items without an embedding are refused only because this
        # fixture has NO embedder at all — the error must be about embedding
        # generation, not about idempotency.
        with pytest.raises(Exception) as exc_info:
            db.record_batch([{"text": "no vector, no key"}])
        assert "idempotency" not in str(exc_info.value).lower()

    def test_keyed_batch_item_without_embedding_is_refused(self, db):
        # sol 4a.6d-2b r1 finding 1: the batch digest is API-byte identity
        # (PayloadVariant::Record, caller vector INCLUDED), so a
        # wrapper-synthesized vector would make honest retries false
        # conflicts. The wrapper must refuse loudly, pointing at the two
        # honest options.
        with pytest.raises(ValueError, match="idempotency_key.*embedding"):
            db.record_batch([{"text": "keyed but no vector", "idempotency_key": "b-3"}])

    def test_keyed_batch_item_with_none_embedding_is_refused(self, db):
        # An explicit None embedding is the same hazard as an absent one.
        with pytest.raises(ValueError, match="idempotency_key.*embedding"):
            db.record_batch(
                [{"text": "keyed, None vector", "embedding": None, "idempotency_key": "b-4"}]
            )

    def test_cross_surface_hit_record_then_batch(self, db):
        # Identical payload + key via record() and then a batch item is ONE
        # write: the batch position returns record()'s rid.
        common = dict(
            text="cross surface",
            memory_type="episodic",
            importance=0.5,
            valence=0.0,
            half_life=604800.0,
            namespace="default",
            certainty=0.8,
            domain="general",
            source="user",
        )
        rid = db.record(embedding=_vec(3.0), idempotency_key="x-1", **common)
        rids = db.record_batch(
            [{**common, "embedding": _vec(3.0), "idempotency_key": "x-1"}]
        )
        assert rids == [rid]
