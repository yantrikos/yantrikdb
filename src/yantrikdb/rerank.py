"""Cross-encoder reranking over recall results — the measured ceiling-breaker.

Measured on a 4,297-record production clone with a 40-query
paraphrase-labeled set (2026-08-06, engine ac6022d):

    engine alone (bm25 fusion, emb+text)   rank-1 48%  top-3 65%  MRR 0.600
    + rerank(pool=20, clip=1500)           rank-1 70%  top-3 82%  MRR 0.763

at ~700ms/query on server CPU. The exact-cosine ceiling of the
underlying embedder is MRR 0.566 — the cross-encoder reads query and
candidate TOGETHER, which is signal no bi-encoder score can carry.

Two findings that shape this API, both measured (do not "fix" them
without re-measuring):
- The reranker must read the record's TEXT HEAD (clip ~1500 chars),
  NOT the matched `best_span` window: snippet-fed reranking scored
  WORSE than no reranking (0.545 vs 0.600) — the cross-encoder needs
  the record's framing context.
- pool=20 beats pool=50 on MRR (0.763 vs 0.750): past ~20 candidates
  the reranker's own ordering noise outweighs the extra recall.

Requires the optional `sentence-transformers` dependency (already
present wherever a Python-side embedder is configured). The model is
loaded once per process and cached.
"""

from __future__ import annotations

from typing import Any

_CE_CACHE: dict[str, Any] = {}

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_POOL = 20
DEFAULT_CLIP = 1500


def _cross_encoder(model_name: str):
    ce = _CE_CACHE.get(model_name)
    if ce is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover - env without extra
            raise ImportError(
                "rerank requires the optional sentence-transformers "
                "dependency: pip install sentence-transformers"
            ) from e
        ce = CrossEncoder(model_name)
        _CE_CACHE[model_name] = ce
    return ce


def rerank_hits(
    query: str,
    hits: list[dict],
    top_k: int | None = None,
    model: str = DEFAULT_MODEL,
    clip: int = DEFAULT_CLIP,
) -> list[dict]:
    """Reorder recall `hits` by cross-encoder relevance to `query`.

    Each hit dict gains a `rerank_score` field; the input order is not
    mutated. `top_k` truncates AFTER reranking (pass your display k;
    feed the function a larger pool — DEFAULT_POOL — for headroom).
    """
    if not hits:
        return []
    ce = _cross_encoder(model)
    scores = ce.predict([(query, h["text"][:clip]) for h in hits])
    out = [dict(h, rerank_score=float(s)) for h, s in zip(hits, scores)]
    out.sort(key=lambda h: -h["rerank_score"])
    return out[:top_k] if top_k else out


def recall_reranked(
    db,
    query: str,
    top_k: int = 10,
    pool_k: int = DEFAULT_POOL,
    model: str = DEFAULT_MODEL,
    clip: int = DEFAULT_CLIP,
    **recall_kwargs,
) -> list[dict]:
    """Recall a `pool_k` candidate pool from `db`, cross-encoder rerank
    it, and return the top `top_k`.

    Any extra keyword arguments pass through to `db.recall` (namespace,
    memory_type, query_embedding, skip_reinforce, ...). `snippets` and
    `min_score_ratio` are applied only AFTER reranking would make no
    sense here, so they are rejected — trim tokens on the caller's side
    of the final list instead.
    """
    for bad in ("snippets", "min_score_ratio"):
        if bad in recall_kwargs:
            raise ValueError(
                f"recall_reranked drives the pool itself; apply {bad!r} "
                "and other presentation trims to its output"
            )
    hits = db.recall(query=query, top_k=pool_k, **recall_kwargs)
    return rerank_hits(query, hits, top_k=top_k, model=model, clip=clip)
