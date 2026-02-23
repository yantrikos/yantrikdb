"""AIDB Proactive Triggers — re-exported from Rust engine."""

from aidb._aidb_rust import (
    Trigger,
    check_all_triggers,
    check_consolidation_triggers,
    check_decay_triggers,
)

__all__ = [
    "Trigger",
    "check_all_triggers",
    "check_consolidation_triggers",
    "check_decay_triggers",
]
