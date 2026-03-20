"""YantrikDB — A Cognitive Memory Engine for Persistent AI Systems."""

from yantrikdb._yantrikdb_rust import YantrikDB, TenantManager
from yantrikdb.consolidate import consolidate
from yantrikdb.triggers import check_all_triggers

# Backward-compat alias
AIDB = YantrikDB

__version__ = "0.2.3"
__all__ = ["YantrikDB", "AIDB", "TenantManager", "consolidate", "check_all_triggers"]
