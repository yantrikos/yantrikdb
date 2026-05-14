"""Tests for the V3 cognition loop: think(), trigger lifecycle, and patterns."""

import math
import time

import pytest

from yantrikdb import YantrikDB

DIM = 8


def _vec(seed: float) -> list[float]:
    raw = [math.sin(seed * (i + 1) * 1.7) + math.cos(seed * (i + 2) * 0.3) for i in range(DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


@pytest.fixture
def db():
    engine = YantrikDB(db_path=":memory:", embedding_dim=DIM)
    yield engine
    engine.close()


class TestThink:
    def test_think_returns_dict(self, db):
        result = db.think()
        assert isinstance(result, dict)
        # Engine ThinkResult additions are backward-compat additions, so we
        # assert subset rather than equality. The test fails honestly only
        # if any expected key DISAPPEARS, not when a new one shows up.
        expected_keys = {
            "triggers", "consolidation_count", "conflicts_found",
            "patterns_new", "patterns_updated", "expired_triggers", "duration_ms",
        }
        assert expected_keys.issubset(result.keys()), (
            f"missing keys: {expected_keys - set(result.keys())}"
        )

    def test_think_empty_db(self, db):
        result = db.think()
        assert result["triggers"] == []
        assert result["consolidation_count"] == 0
        assert result["conflicts_found"] == 0

    def test_think_with_config(self, db):
        config = {
            "importance_threshold": 0.3,
            "decay_threshold": 0.05,
            "max_triggers": 5,
            "run_consolidation": False,
            "run_conflict_scan": False,
            "run_pattern_mining": False,
        }
        result = db.think(config)
        assert isinstance(result, dict)
        assert result["consolidation_count"] == 0

    def test_think_finds_decayed_triggers(self, db):
        rid = db.record(
            "important deadline March 15",
            importance=0.9,
            half_life=100.0,
            embedding=_vec(1.0),
        )
        # Backdate to simulate time passing
        db._conn.execute(
            "UPDATE memories SET last_access = ?  WHERE rid = ?",
            (time.time() - 10000, rid),
        )
        db._conn.commit()

        result = db.think({"importance_threshold": 0.5, "decay_threshold": 0.1})
        assert len(result["triggers"]) >= 1
        types = [t["trigger_type"] for t in result["triggers"]]
        assert "decay_review" in types

    def test_think_duration_ms(self, db):
        result = db.think()
        assert result["duration_ms"] >= 0


class TestTriggerLifecycle:
    def _create_trigger(self, db):
        """Helper: create a decayed memory so think() produces a trigger."""
        rid = db.record("critical task", importance=0.95, half_life=50.0, embedding=_vec(1.0))
        db._conn.execute(
            "UPDATE memories SET last_access = ? WHERE rid = ?",
            (time.time() - 20000, rid),
        )
        db._conn.commit()
        result = db.think({"importance_threshold": 0.5, "decay_threshold": 0.1})
        assert len(result["triggers"]) >= 1
        # Get the persisted trigger id
        pending = db.get_pending_triggers(limit=10)
        assert len(pending) >= 1
        return pending[0]["trigger_id"]

    def test_get_pending_triggers(self, db):
        trigger_id = self._create_trigger(db)
        pending = db.get_pending_triggers()
        assert any(t["trigger_id"] == trigger_id for t in pending)
        t = [t for t in pending if t["trigger_id"] == trigger_id][0]
        assert t["status"] == "pending"

    def test_deliver_trigger(self, db):
        trigger_id = self._create_trigger(db)
        assert db.deliver_trigger(trigger_id) is True
        # Should no longer be "pending"
        history = db.get_trigger_history(limit=50)
        t = [h for h in history if h["trigger_id"] == trigger_id][0]
        assert t["status"] == "delivered"
        assert t["delivered_at"] is not None

    def test_acknowledge_trigger(self, db):
        trigger_id = self._create_trigger(db)
        db.deliver_trigger(trigger_id)
        assert db.acknowledge_trigger(trigger_id) is True
        history = db.get_trigger_history(limit=50)
        t = [h for h in history if h["trigger_id"] == trigger_id][0]
        assert t["status"] == "acknowledged"

    def test_act_on_trigger(self, db):
        trigger_id = self._create_trigger(db)
        db.deliver_trigger(trigger_id)
        assert db.act_on_trigger(trigger_id) is True
        history = db.get_trigger_history(limit=50)
        t = [h for h in history if h["trigger_id"] == trigger_id][0]
        assert t["status"] == "acted"

    def test_dismiss_trigger(self, db):
        trigger_id = self._create_trigger(db)
        assert db.dismiss_trigger(trigger_id) is True
        history = db.get_trigger_history(limit=50)
        t = [h for h in history if h["trigger_id"] == trigger_id][0]
        assert t["status"] == "dismissed"

    def test_trigger_history_filter_by_type(self, db):
        self._create_trigger(db)
        history = db.get_trigger_history(trigger_type="decay_review", limit=50)
        assert all(h["trigger_type"] == "decay_review" for h in history)


class TestPatterns:
    def test_get_patterns_empty(self, db):
        patterns = db.get_patterns()
        assert patterns == []

    def test_entity_hub_pattern(self, db):
        """Create a high-degree entity hub and verify think() finds it."""
        # Create entity "Python" with 6 edges (threshold is 5)
        for dst in ["typing", "asyncio", "pydantic", "fastapi", "django", "flask"]:
            db.relate("Python", dst, rel_type="has_library")

        result = db.think({"run_pattern_mining": True, "run_consolidation": False})
        patterns = db.get_patterns(pattern_type="entity_hub")
        # Should detect Python as a hub
        assert len(patterns) >= 1
        p = patterns[0]
        assert p["pattern_type"] == "entity_hub"
        assert p["confidence"] > 0
        assert "Python" in p["entity_names"]

    def test_patterns_filter_by_status(self, db):
        # With no patterns, filtering should return empty
        active = db.get_patterns(status="active")
        assert active == []
        stale = db.get_patterns(status="stale")
        assert stale == []


class TestStatsV3:
    def test_stats_include_trigger_and_pattern_counts(self, db):
        s = db.stats()
        assert "pending_triggers" in s
        assert "active_patterns" in s
        assert s["pending_triggers"] == 0
        assert s["active_patterns"] == 0
