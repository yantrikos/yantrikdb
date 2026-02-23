"""AIDB — A Cognitive Memory Engine for Persistent AI Systems."""

from aidb._aidb_rust import AIDB
from aidb.consolidate import consolidate
from aidb.triggers import check_all_triggers

__version__ = "0.1.0"
__all__ = ["AIDB", "consolidate", "check_all_triggers"]
