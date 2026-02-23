"""Benchmarks for AIDB Rust engine core operations.

Run with: pytest benchmarks/ -v --benchmark-columns=mean,stddev,median,rounds
"""

import math
import random

import pytest

from aidb import AIDB

DIM = 64


def random_embedding(seed=None):
    """Generate a random unit-norm embedding."""
    rng = random.Random(seed)
    raw = [rng.gauss(0, 1) for _ in range(DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


@pytest.fixture
def db():
    d = AIDB(":memory:", embedding_dim=DIM)
    yield d
    d.close()


def _seed_db(db, n):
    for i in range(n):
        emb = random_embedding(seed=i)
        db.record(
            text=f"Memory number {i} about topic {i % 10}",
            memory_type="episodic" if i % 2 == 0 else "semantic",
            importance=0.3 + (i % 7) * 0.1,
            valence=(i % 5 - 2) * 0.2,
            half_life=604800.0,
            embedding=emb,
        )


@pytest.fixture
def db_100(db):
    _seed_db(db, 100)
    return db


@pytest.fixture
def db_1000(db):
    _seed_db(db, 1000)
    return db


class TestRecord:
    def test_record(self, benchmark, db):
        i = [0]

        def do_record():
            emb = random_embedding(seed=10000 + i[0])
            db.record(
                text=f"bench record {i[0]}",
                memory_type="episodic",
                importance=0.5,
                valence=0.0,
                half_life=604800.0,
                embedding=emb,
            )
            i[0] += 1

        benchmark(do_record)


class TestRecall:
    def test_recall_100(self, benchmark, db_100):
        emb = random_embedding(seed=9999)
        benchmark(lambda: db_100.recall(query_embedding=emb, top_k=10))

    def test_recall_1000(self, benchmark, db_1000):
        emb = random_embedding(seed=9999)
        benchmark(lambda: db_1000.recall(query_embedding=emb, top_k=10))


class TestRelate:
    def test_relate(self, benchmark, db_100):
        i = [0]

        def do_relate():
            db_100.relate(f"entity_{i[0]}", f"entity_{i[0]+1}", "related_to", 1.0)
            i[0] += 1

        benchmark(do_relate)


class TestDecay:
    def test_decay_100(self, benchmark, db_100):
        benchmark(lambda: db_100.decay(threshold=0.01))


class TestStats:
    def test_stats(self, benchmark, db_100):
        benchmark(db_100.stats)


class TestGet:
    def test_get(self, benchmark, db_100):
        rid = db_100.record(
            text="lookup target",
            memory_type="episodic",
            importance=0.5,
            valence=0.0,
            half_life=604800.0,
            embedding=random_embedding(seed=77777),
        )
        benchmark(lambda: db_100.get(rid))


class TestBulkInsert:
    def test_bulk_insert_500(self, benchmark, db):
        def bulk():
            for i in range(500):
                db.record(
                    text=f"bulk {i}",
                    memory_type="episodic",
                    importance=0.5,
                    valence=0.0,
                    half_life=604800.0,
                    embedding=random_embedding(seed=i + 50000),
                )

        benchmark.pedantic(bulk, iterations=1, rounds=3)


class TestEndToEnd:
    def test_record_then_recall(self, benchmark, db):
        def cycle():
            for i in range(50):
                emb = random_embedding(seed=i + 80000)
                db.record(
                    text=f"e2e {i}",
                    memory_type="episodic",
                    importance=0.5,
                    valence=0.0,
                    half_life=604800.0,
                    embedding=emb,
                )
            q = random_embedding(seed=99999)
            return db.recall(query_embedding=q, top_k=10)

        benchmark.pedantic(cycle, iterations=1, rounds=3)
