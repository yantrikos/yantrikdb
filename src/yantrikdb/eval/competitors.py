"""Competitive benchmark harness (task 45): YantrikDB vs mem0 / Zep / Letta.

A common interface over each memory system and a runner that scores them all
on the SAME golden-query corpus with the SAME metrics. Competitor adapters
lazily import their library and are skipped — clearly, never silently — when
it isn't installed. So this runs YantrikDB today and produces the full
comparison the moment you `pip install mem0ai zep-python letta`.

Pre-registered methodology (frozen so results can't be tuned after the fact):
- Same corpus (the synthetic connected sessions), same 40 golden queries,
  same top_k, same recall@k / precision@k / MRR for every system.
- No per-system tuning: each adapter uses its library's default add/search.
- Comparison is text-based (expected_texts vs retrieved texts) since systems
  don't share record ids.

Run:
    python -m yantrikdb.eval.competitors
"""

from __future__ import annotations

from dataclasses import dataclass

from yantrikdb.eval.synthetic import GOLDEN_QUERIES, SESSIONS


class MemorySystem:
    """Common interface every adapter implements."""

    name = "base"

    def add(self, text: str) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int) -> list[str]:
        """Return the texts of the top_k retrieved memories."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class YantrikDBSystem(MemorySystem):
    name = "yantrikdb"

    def __init__(self):
        from yantrikdb import YantrikDB

        self.db = YantrikDB.with_default(":memory:")

    def add(self, text: str) -> None:
        self.db.record_text(text, "semantic", 0.5, 0.0, 604800.0)

    def search(self, query: str, top_k: int) -> list[str]:
        results = self.db.recall(query=query, top_k=top_k, skip_reinforce=True)
        return [r["text"] for r in results]

    def close(self) -> None:
        self.db.close()


class Mem0System(MemorySystem):
    name = "mem0"

    def __init__(self):
        # Optional competitor dep, not installed in CI — the ImportError is the
        # intended signal that this adapter is unavailable.
        from mem0 import Memory  # pylint: disable=import-error

        self.m = Memory()
        self.user = "bench"

    def add(self, text: str) -> None:
        self.m.add(text, user_id=self.user)

    def search(self, query: str, top_k: int) -> list[str]:
        hits = self.m.search(query, user_id=self.user, limit=top_k)
        items = hits.get("results", hits) if isinstance(hits, dict) else hits
        return [h.get("memory", h.get("text", "")) for h in items]


class ZepSystem(MemorySystem):
    name = "zep"

    def __init__(self):
        # Optional competitor dep, not installed in CI (see Mem0System).
        from zep_python.client import Zep  # pylint: disable=import-error

        self.client = Zep(api_key="bench")
        self.session = "bench"

    def add(self, text: str) -> None:
        self.client.memory.add(session_id=self.session, messages=[{"role": "user", "content": text}])

    def search(self, query: str, top_k: int) -> list[str]:
        res = self.client.memory.search(session_id=self.session, text=query, limit=top_k)
        return [r.message.content for r in res]


class LettaSystem(MemorySystem):
    name = "letta"

    def __init__(self):
        # Optional competitor dep, not installed in CI (see Mem0System).
        from letta_client import Letta  # pylint: disable=import-error

        self.client = Letta(base_url="http://localhost:8283")
        self.agent = self.client.agents.create(name="bench")

    def add(self, text: str) -> None:
        self.client.agents.passages.create(agent_id=self.agent.id, text=text)

    def search(self, query: str, top_k: int) -> list[str]:
        res = self.client.agents.passages.search(agent_id=self.agent.id, query=query, limit=top_k)
        return [p.text for p in res]


ADAPTERS = [YantrikDBSystem, Mem0System, ZepSystem, LettaSystem]


@dataclass
class SystemResult:
    name: str
    available: bool
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    note: str = ""


def _score(system: MemorySystem, top_k: int) -> tuple[float, float, float]:
    for session in SESSIONS:
        for mem in session["memories"]:
            system.add(mem["text"])
    rec = prec = mrr = 0.0
    n = len(GOLDEN_QUERIES)
    for gq in GOLDEN_QUERIES:
        retrieved = system.search(gq["query"], top_k)
        expected = set(gq["expected_texts"])
        hits = sum(1 for t in gq["expected_texts"] if t in retrieved)
        rec += hits / len(gq["expected_texts"]) if gq["expected_texts"] else 0.0
        relevant = sum(1 for t in retrieved if t in expected)
        prec += relevant / len(retrieved) if retrieved else 0.0
        for rank, t in enumerate(retrieved, 1):
            if t in expected:
                mrr += 1.0 / rank
                break
    return rec / n, prec / n, mrr / n


def run_competitive_benchmark(top_k: int = 10) -> list[SystemResult]:
    """Score every adapter that's available. Missing competitors are reported
    as unavailable (never silently dropped)."""
    results: list[SystemResult] = []
    for cls in ADAPTERS:
        try:
            system = cls()
        except Exception as e:  # noqa: BLE001 — import or connect failure
            results.append(
                SystemResult(cls.name, False, note=f"unavailable ({type(e).__name__}: {e})")
            )
            continue
        try:
            rec, prec, mrr = _score(system, top_k)
            results.append(SystemResult(cls.name, True, rec, prec, mrr))
        finally:
            system.close()
    return results


def summary(results: list[SystemResult]) -> str:
    lines = [
        "=== Competitive Benchmark (same corpus, same queries, no tuning) ===",
        f"{'system':<12}{'avail':>7}{'recall@k':>10}{'prec@k':>9}{'mrr':>8}",
    ]
    for r in results:
        if r.available:
            lines.append(
                f"{r.name:<12}{'yes':>7}{r.recall_at_k:>10.3f}{r.precision_at_k:>9.3f}{r.mrr:>8.3f}"
            )
        else:
            lines.append(f"{r.name:<12}{'no':>7}   {r.note}")
    lines.append("")
    lines.append(
        "Competitors marked 'no' are not installed — `pip install mem0ai zep-python "
        "letta-client` to run the full comparison."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary(run_competitive_benchmark()))
