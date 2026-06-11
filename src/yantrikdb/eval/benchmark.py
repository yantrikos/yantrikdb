"""Scaling self-benchmark (tasks 46 + 47): precision and latency vs corpus size.

Validates the core product claim — recall precision stays stable as the
corpus grows (more memories don't degrade retrieval) — and measures recall
latency, so a regression in either can fail CI.

The signal corpus (golden queries with known answers) is held fixed while
increasing volumes of *distractor* memories are added around it; a healthy
system keeps finding the signal as the noise grows, with bounded latency.

Run:
    python -m yantrikdb.eval.benchmark

Runs dependency-free on the bundled embedder; CI gates on
`BenchmarkReport.regression_check()`.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from yantrikdb import YantrikDB
from yantrikdb.eval.graph_lift import _build_embedder
from yantrikdb.eval.synthetic import GOLDEN_QUERIES, load_sessions_into_db

_TOPICS = [
    "weather", "sports", "cooking", "travel", "music",
    "gardening", "finance", "history", "biology", "cinema",
]


def _distractors(n: int):
    """Generate `n` plausible-but-irrelevant memories (noise around the signal)."""
    return [
        f"A passing note about {_TOPICS[i % len(_TOPICS)]}, entry {i}, "
        f"unrelated to any tracked project or person."
        for i in range(n)
    ]


def _record_with_retry(db, **kwargs):
    """Record, retrying on ingest-queue backpressure so bulk loads complete."""
    for _ in range(200):
        try:
            return db.record(**kwargs)
        except RuntimeError as e:
            if "queue full" in str(e):
                time.sleep(0.02)
                continue
            raise
    raise RuntimeError("ingest queue stayed full")


def _percentile(values, pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


@dataclass
class ScalePoint:
    distractors: int
    total_memories: int
    precision_at_k: float
    recall_at_k: float
    latency_p50_ms: float
    latency_p95_ms: float


@dataclass
class BenchmarkReport:
    points: list = field(default_factory=list)
    embedder: str = ""

    def regression_check(self, min_recall: float = 0.5, max_p95_ms: float = 400.0) -> list:
        """Return a list of regression issues (empty = healthy). CI fails on
        any. Guards the two claims: recall doesn't collapse as the corpus
        grows, and latency stays bounded."""
        issues = []
        for p in self.points:
            if p.recall_at_k < min_recall:
                issues.append(
                    f"recall {p.recall_at_k:.3f} < {min_recall} at {p.total_memories} memories"
                )
            if p.latency_p95_ms > max_p95_ms:
                issues.append(
                    f"p95 {p.latency_p95_ms:.1f}ms > {max_p95_ms}ms at {p.total_memories} memories"
                )
        # Recall must not degrade materially from the smallest to the largest corpus.
        if len(self.points) >= 2:
            drop = self.points[0].recall_at_k - self.points[-1].recall_at_k
            if drop > 0.15:
                issues.append(
                    f"recall degraded {drop:.3f} as corpus grew "
                    f"({self.points[0].total_memories}->{self.points[-1].total_memories})"
                )
        return issues

    def summary(self) -> str:
        lines = [
            f"=== Scaling Self-Benchmark ({self.embedder}) ===",
            f"{'memories':>10}{'recall@k':>10}{'prec@k':>9}{'p50 ms':>9}{'p95 ms':>9}",
        ]
        for p in self.points:
            lines.append(
                f"{p.total_memories:>10}{p.recall_at_k:>10.3f}{p.precision_at_k:>9.3f}"
                f"{p.latency_p50_ms:>9.1f}{p.latency_p95_ms:>9.1f}"
            )
        issues = self.regression_check()
        lines.append("")
        lines.append("REGRESSION: " + ("none — healthy" if not issues else "; ".join(issues)))
        return "\n".join(lines)


def _measure(db, text_to_rid, embedder, top_k: int) -> tuple[float, float, list]:
    precs, recs, lats = [], [], []
    for gq in GOLDEN_QUERIES:
        qemb = None
        if embedder is not None:
            vec = embedder.encode(gq["query"])
            qemb = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        t0 = time.perf_counter()
        results = db.recall(
            query=gq["query"], query_embedding=qemb, top_k=top_k, skip_reinforce=True
        )
        lats.append((time.perf_counter() - t0) * 1000.0)
        retrieved = {r["rid"] for r in results}
        expected = {text_to_rid[t] for t in gq["expected_texts"] if t in text_to_rid}
        hits = sum(1 for t in gq["expected_texts"] if text_to_rid.get(t) in retrieved)
        recs.append(hits / len(gq["expected_texts"]) if gq["expected_texts"] else 0.0)
        rel = sum(1 for rid in retrieved if rid in expected)
        precs.append(rel / len(results) if results else 0.0)
    return statistics.mean(precs), statistics.mean(recs), lats


def run_benchmark(scales=(0, 100, 200), top_k: int = 10, embedder=None) -> BenchmarkReport:
    name = "provided"
    if embedder is None:
        embedder, name = _build_embedder()

    report = BenchmarkReport(embedder=name)
    for n in scales:
        if embedder is not None and hasattr(embedder, "get_sentence_embedding_dimension"):
            db = YantrikDB(
                db_path=":memory:",
                embedding_dim=embedder.get_sentence_embedding_dimension(),
                embedder=embedder,
            )
        elif embedder is not None:
            dim = len(embedder.encode("probe"))
            db = YantrikDB(db_path=":memory:", embedding_dim=dim, embedder=embedder)
        else:
            db = YantrikDB.with_default(":memory:")

        text_to_rid = load_sessions_into_db(db, embedder=embedder)
        signal = len(text_to_rid)
        for d in _distractors(n):
            emb = None
            if embedder is not None:
                vec = embedder.encode(d)
                emb = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            _record_with_retry(
                db, text=d, memory_type="semantic", importance=0.3, valence=0.0, embedding=emb
            )

        prec, rec, lats = _measure(db, text_to_rid, embedder, top_k)
        report.points.append(
            ScalePoint(
                distractors=n,
                total_memories=signal + n,
                precision_at_k=prec,
                recall_at_k=rec,
                latency_p50_ms=_percentile(lats, 50),
                latency_p95_ms=_percentile(lats, 95),
            )
        )
        db.close()
    return report


if __name__ == "__main__":
    import sys

    report = run_benchmark()
    print(report.summary())
    # Non-zero exit on regression so CI gates on it.
    sys.exit(1 if report.regression_check() else 0)
