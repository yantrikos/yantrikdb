"""Importers — bring memories from other systems and start immediately.

The switching-cost killer: anyone with an existing memory store (mem0, Zep,
Letta, a JSONL export, a folder of markdown notes) imports it in one command
and their history is live — backdated, deduplicated, retry-safe.

Design decisions, each carrying a measured scar:

- **Keyed single-record writes, not batch.** Every record goes through
  `record(..., idempotency_key="import:<system>:<id>")`: re-running an
  interrupted import never duplicates (same key + same payload = same rid).
  The batch path is deliberately avoided twice over: keyed batch items
  require caller-supplied embeddings (the digest includes the vector), and
  the shipped engine's batch surface does not run event-time extraction
  (fixed in source 2026-08-15; single-record writes always had it).

- **Backdating is the point.** `created_at` carries the memory's ORIGINAL
  timestamp. Without it every imported memory looks written today and
  recency scoring treats a two-year-old preference as this morning's news.
  Event dates mentioned in the text are extracted by the engine on top.

- **Provenance survives.** `metadata.imported_from`, `original_id`, and the
  source system's own metadata ride along; provenance `source` defaults to
  "document" (an import is testimony, not the user speaking now).

- **The census is not optional.** Import ends by verifying every written
  rid is readable back (the HNSW-orphan lesson: a self-retrieval census is
  the only proof of delivery), plus a recall spot-check on a sample. The
  report states read/written/aliased/verified counts — never "done".
"""
from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

__all__ = ["ImportRecord", "ImportReport", "import_records", "SOURCES"]


@dataclass
class ImportRecord:
    text: str
    source_system: str
    source_id: str | None = None
    created_at: float | None = None
    memory_type: str = "semantic"
    importance: float = 0.5
    namespace: str = "default"
    certainty: float = 0.8
    domain: str = "general"
    metadata: dict = field(default_factory=dict)

    def idempotency_key(self) -> str:
        # A stable id from the source when it has one; a content hash when it
        # does not. Either way a re-run maps to the same key, so an
        # interrupted import is resumable by simply running it again.
        sid = self.source_id or hashlib.sha256(
            f"{self.namespace}\x00{self.text}".encode()).hexdigest()[:24]
        return f"import:{self.source_system}:{sid}"


@dataclass
class ImportReport:
    source_system: str
    read: int = 0
    written: int = 0
    aliased: int = 0
    skipped_empty: int = 0
    errors: int = 0
    error_samples: list = field(default_factory=list)
    census_checked: int = 0
    census_missing: int = 0
    recall_spot_checks: int = 0
    recall_spot_hits: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            f"imported from {self.source_system}: read {self.read}, "
            f"new {self.written - self.aliased}, "
            f"already-present {self.aliased}, "
            f"empty-skipped {self.skipped_empty}, "
            f"errors {self.errors} ({self.seconds:.1f}s)",
            f"census: {self.census_checked - self.census_missing}/"
            f"{self.census_checked} written rids readable back"
            + (" — MISSING RECORDS, DO NOT TRUST THIS IMPORT"
               if self.census_missing else ""),
        ]
        if self.recall_spot_checks:
            lines.append(
                f"recall spot-check: {self.recall_spot_hits}/"
                f"{self.recall_spot_checks} sampled memories retrievable by "
                f"their own opening words")
        if self.error_samples:
            lines.append("first errors: " + "; ".join(self.error_samples[:3]))
        return "\n".join(lines)


def import_records(db, records: Iterable[ImportRecord], *,
                   dry_run: bool = False, limit: int = 0,
                   progress: Callable[[int], None] | None = None,
                   spot_checks: int = 10) -> ImportReport:
    """Write `records` into `db` exactly-once and prove they landed."""
    t0 = time.perf_counter()
    report = ImportReport(source_system="?")
    written_rids: list[tuple[str, str, str]] = []  # (rid, text, namespace)

    # A keyed retry returns the EXISTING rid, so "written" alone would count
    # a resumed import as if it wrote everything again — the misleading-count
    # sin. The store's own active count before/after tells new from aliased.
    def _active() -> int:
        try:
            return int((db.stats() or {}).get("active_memories") or 0)
        except Exception:  # noqa: BLE001
            return -1

    active_before = _active()

    for rec in records:
        report.source_system = rec.source_system
        report.read += 1
        if limit and report.read > limit:
            report.read -= 1
            break
        text = (rec.text or "").strip()
        if not text:
            report.skipped_empty += 1
            continue
        if dry_run:
            continue
        meta = dict(rec.metadata or {})
        meta.setdefault("imported_from", rec.source_system)
        if rec.source_id:
            meta.setdefault("original_id", rec.source_id)
        try:
            rid = db.record(
                text,
                memory_type=rec.memory_type,
                importance=rec.importance,
                namespace=rec.namespace,
                certainty=rec.certainty,
                domain=rec.domain,
                source="document",
                metadata=meta,
                created_at=rec.created_at,
                idempotency_key=rec.idempotency_key(),
            )
            report.written += 1
            written_rids.append((rid, text, rec.namespace))
        except Exception as e:  # noqa: BLE001
            report.errors += 1
            if len(report.error_samples) < 5:
                report.error_samples.append(
                    f"{type(e).__name__}: {str(e)[:90]}")
        if progress and report.read % 100 == 0:
            progress(report.read)

    # Census: every written rid must be readable back. Existence is the hard
    # gate; retrievability is a sampled report line (short or duplicated
    # texts legitimately lose recall races, so it informs rather than fails).
    for rid, _, _ in written_rids:
        report.census_checked += 1
        if db.get_memory(rid) is None:
            report.census_missing += 1
    if written_rids and spot_checks:
        rng = random.Random(0)
        for rid, text, ns in rng.sample(
                written_rids, min(spot_checks, len(written_rids))):
            report.recall_spot_checks += 1
            probe = " ".join(text.split()[:8])
            try:
                hits = db.recall(probe, top_k=10, namespace=ns,
                                 skip_reinforce=True)
                if any(h.get("rid") == rid for h in hits):
                    report.recall_spot_hits += 1
            except Exception:  # noqa: BLE001
                pass

    active_after = _active()
    if active_before >= 0 and active_after >= 0:
        new = max(0, active_after - active_before)
        report.aliased = max(0, report.written - new)

    report.seconds = round(time.perf_counter() - t0, 2)
    return report


def _sources() -> dict[str, Callable[..., Iterator[ImportRecord]]]:
    from . import sources
    return {
        "jsonl": sources.iter_jsonl,
        "mem0": sources.iter_mem0,
        "zep": sources.iter_zep,
        "letta": sources.iter_letta,
        "markdown": sources.iter_markdown,
        "mnemosyne": sources.iter_mnemosyne,
    }


SOURCES = _sources
