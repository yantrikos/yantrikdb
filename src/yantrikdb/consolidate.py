"""YantrikDB Consolidation — re-exported from Rust engine."""

from yantrikdb._yantrikdb_rust import (
    _cosine_similarity,
    _extractive_summary,
    _find_clusters,
    consolidate,
    consolidate_cluster,
    find_consolidation_candidates,
    record_synthesis,
    restore_from_consolidation,
    summarize_cluster,
)

__all__ = [
    "consolidate",
    "consolidate_cluster",
    "find_consolidation_candidates",
    "record_synthesis",
    "restore_from_consolidation",
    "summarize_cluster",
    "_cosine_similarity",
    "_extractive_summary",
    "_find_clusters",
]
