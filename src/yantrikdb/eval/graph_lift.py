"""Graph-lift evaluation (task 43): does the knowledge graph earn its keep?

The 2026-06-10 audit found the graph at noise-level density and asked, with
data, whether `expand_entities=True` actually improves recall or just costs
cycles. This measures recall quality with the graph signal ON vs OFF on the
synthetic *connected* corpus (which has real entity relations) and prints an
invest-vs-drop verdict.

Run:
    python -m yantrikdb.eval.graph_lift

It uses SentenceTransformer if available (best signal), else the bundled
embedder, so it runs with or without the heavy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

from yantrikdb import YantrikDB
from yantrikdb.eval.synthetic import GOLDEN_QUERIES, load_sessions_into_db


@dataclass
class GraphLiftReport:
    """On/off recall metrics and the resulting decision."""

    recall_on: float
    recall_off: float
    precision_on: float
    precision_off: float
    mrr_on: float
    mrr_off: float
    queries: int
    embedder: str

    @property
    def recall_delta(self) -> float:
        return self.recall_on - self.recall_off

    @property
    def precision_delta(self) -> float:
        return self.precision_on - self.precision_off

    @property
    def mrr_delta(self) -> float:
        return self.mrr_on - self.mrr_off

    def verdict(self, threshold: float = 0.01) -> str:
        deltas = [self.recall_delta, self.precision_delta, self.mrr_delta]
        positives = sum(1 for d in deltas if d > threshold)
        if max(deltas) > threshold and positives >= 2:
            return (
                "INVEST: graph expansion measurably improves recall on connected "
                "data — fund continuous relation extraction (task 44) to raise density."
            )
        if min(deltas) < -threshold:
            return (
                "DROP: graph expansion degrades recall — turn expand_entities off by "
                "default and stop paying its cost."
            )
        return (
            "NEUTRAL: no measurable lift at this density — keep expand_entities cheap "
            "or off by default until extraction raises edge density (then re-run)."
        )

    def summary(self) -> str:
        return "\n".join(
            [
                f"=== Graph-Lift Eval ({self.embedder}, {self.queries} queries) ===",
                f"{'metric':<14}{'graph ON':>10}{'graph OFF':>11}{'delta':>9}",
                f"{'recall@k':<14}{self.recall_on:>10.3f}{self.recall_off:>11.3f}{self.recall_delta:>+9.3f}",
                f"{'precision@k':<14}{self.precision_on:>10.3f}{self.precision_off:>11.3f}{self.precision_delta:>+9.3f}",
                f"{'mrr':<14}{self.mrr_on:>10.3f}{self.mrr_off:>11.3f}{self.mrr_delta:>+9.3f}",
                "",
                f"VERDICT: {self.verdict()}",
            ]
        )


def _build_embedder():
    """Return (embedder, name). Prefer SentenceTransformer; fall back to None
    (bundled embedder via with_default)."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("all-MiniLM-L6-v2"), "all-MiniLM-L6-v2"
    except Exception:
        return None, "bundled"


def _metrics(db, text_to_rid, embedder, expand: bool, top_k: int):
    rec = prec = mrr = 0.0
    n = len(GOLDEN_QUERIES)
    for gq in GOLDEN_QUERIES:
        qemb = None
        if embedder is not None:
            vec = embedder.encode(gq["query"])
            qemb = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        results = db.recall(
            query=gq["query"],
            query_embedding=qemb,
            top_k=top_k,
            skip_reinforce=True,
            expand_entities=expand,
        )
        retrieved = [r["rid"] for r in results]
        retrieved_set = set(retrieved)
        expected = {text_to_rid[t] for t in gq["expected_texts"] if t in text_to_rid}

        hits = sum(1 for t in gq["expected_texts"] if text_to_rid.get(t) in retrieved_set)
        rec += hits / len(gq["expected_texts"]) if gq["expected_texts"] else 0.0
        relevant = sum(1 for rid in retrieved if rid in expected)
        prec += relevant / len(retrieved) if retrieved else 0.0
        for rank, rid in enumerate(retrieved, 1):
            if rid in expected:
                mrr += 1.0 / rank
                break
    return rec / n, prec / n, mrr / n


def run_graph_lift_eval(top_k: int = 10, embedder=None) -> GraphLiftReport:
    """Load the connected synthetic corpus and measure recall with the graph
    signal on vs off. Same embeddings both ways, so the delta isolates the
    graph's contribution.

    Pass an `embedder` (anything with `.encode`) to inject one — e.g. a fast
    deterministic mock in tests. When omitted, SentenceTransformer is used if
    available, else the bundled embedder.
    """
    name = "provided"
    if embedder is None:
        embedder, name = _build_embedder()
    if embedder is not None:
        if hasattr(embedder, "get_sentence_embedding_dimension"):
            dim = embedder.get_sentence_embedding_dimension()
        else:
            dim = len(embedder.encode("dimension probe"))
        db = YantrikDB(db_path=":memory:", embedding_dim=dim, embedder=embedder)
    else:
        db = YantrikDB.with_default(":memory:")
    text_to_rid = load_sessions_into_db(db, embedder=embedder)

    ron, pon, mon = _metrics(db, text_to_rid, embedder, True, top_k)
    roff, poff, moff = _metrics(db, text_to_rid, embedder, False, top_k)
    db.close()
    return GraphLiftReport(
        recall_on=ron,
        recall_off=roff,
        precision_on=pon,
        precision_off=poff,
        mrr_on=mon,
        mrr_off=moff,
        queries=len(GOLDEN_QUERIES),
        embedder=name,
    )


if __name__ == "__main__":
    print(run_graph_lift_eval().summary())
