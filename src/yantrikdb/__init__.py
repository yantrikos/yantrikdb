"""YantrikDB — A Cognitive Memory Engine for Persistent AI Systems."""

from yantrikdb._yantrikdb_rust import (
    BatchDeferredDuringReembed,
    Backpressure,
    CorrectionDeferredDuringReembed,
    IdempotencyConflict,
    InvalidIdempotencyKey,
    PackAlreadyMounted,
    PackEmbedderMismatch,
    PackSignatureInvalid,
    ProvenanceInconsistent,
    RecallContended,
    TenantManager,
    YantrikDB,
)
from yantrikdb.consolidate import consolidate, record_synthesis
from yantrikdb.organize import (
    ConcernItem,
    ConcernPlan,
    EvidenceAssignment,
    OrganizationPlan,
    TopicHandle,
    assign_evidence_to_handles,
    organize_evidence,
    organize_concerns,
    persist_concerns,
    persist_organization,
    recall_organized,
    validate_topic_handles,
    validate_concern_items,
)
from yantrikdb.rerank import recall_reranked, rerank_hits
from yantrikdb.triggers import check_all_triggers

# Backward-compat alias
AIDB = YantrikDB

# The single source of the version is the installed distribution metadata
# (pyproject.toml). A hardcoded string here drifted to "0.2.4" while the
# wheel shipped 0.9.4 — the exact version-string lie the v0.10 release
# coordination flagged (yantrikdb-mcp friction 1).
try:
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("yantrikdb")
except Exception:  # pragma: no cover - source tree without installed dist
    __version__ = "0.0.0+unknown"

__all__ = [
    "YantrikDB",
    "AIDB",
    "TenantManager",
    "consolidate",
    "record_synthesis",
    "ConcernItem",
    "ConcernPlan",
    "TopicHandle",
    "EvidenceAssignment",
    "OrganizationPlan",
    "validate_topic_handles",
    "assign_evidence_to_handles",
    "persist_organization",
    "organize_evidence",
    "organize_concerns",
    "persist_concerns",
    "recall_organized",
    "validate_concern_items",
    "check_all_triggers",
    "recall_reranked",
    "rerank_hits",
    "Backpressure",
    "BatchDeferredDuringReembed",
    "CorrectionDeferredDuringReembed",
    "IdempotencyConflict",
    "InvalidIdempotencyKey",
    "PackAlreadyMounted",
    "PackEmbedderMismatch",
    "PackSignatureInvalid",
    "ProvenanceInconsistent",
    "RecallContended",
]
