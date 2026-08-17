"""Tests for the maintenance-debt ledger binding: maintenance_debt().

The core is a passive library — it cannot schedule maintenance, but it must
always be able to answer "how overdue is maintenance?". The binding returns a
dict with exactly four keys; the reactive host (MCP) hands it to the calling
LLM, which acts as the scheduler.
"""

import math

import pytest

from yantrikdb import YantrikDB

DIM = 8

EXPECTED_KEYS = {
    "writes_since_think",
    "last_think_at",
    "open_conflicts",
    "pending_triggers",
}


def _vec(seed: float) -> list[float]:
    raw = [math.sin(seed * (i + 1) * 1.7) + math.cos(seed * (i + 2) * 0.3) for i in range(DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


@pytest.fixture
def db():
    engine = YantrikDB(db_path=":memory:", embedding_dim=DIM)
    yield engine
    engine.close()


class TestMaintenanceDebt:
    def test_virgin_store_zeros_and_none(self, db):
        debt = db.maintenance_debt()
        assert isinstance(debt, dict)
        assert set(debt.keys()) == EXPECTED_KEYS
        assert debt["writes_since_think"] == 0
        assert debt["last_think_at"] is None
        assert debt["open_conflicts"] == 0
        assert debt["pending_triggers"] == 0

    def test_writes_accumulate_and_think_settles(self, db):
        db.record("debt item one", embedding=_vec(1.0))
        db.record("debt item two", embedding=_vec(2.0))
        assert db.maintenance_debt()["writes_since_think"] == 2

        db.think()

        debt = db.maintenance_debt()
        assert debt["writes_since_think"] == 0
        assert debt["last_think_at"] is not None
        assert debt["last_think_at"] > 0.0

    def test_dry_run_cycle_preserves_debt(self, db):
        db.record("unexamined material", embedding=_vec(1.0))

        db.run_maintenance_cycle(dry_run=True)

        debt = db.maintenance_debt()
        assert debt["writes_since_think"] == 1, "a preview must not clear debt"
        assert debt["last_think_at"] is None

        db.run_maintenance_cycle(dry_run=False)
        debt = db.maintenance_debt()
        assert debt["writes_since_think"] == 0
        assert debt["last_think_at"] is not None
