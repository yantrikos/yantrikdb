"""
Empirical quality eval: potion-base-2M vs all-MiniLM-L6-v2 vs Slice A
hash-trick baseline, on yantrikdb-shaped memory texts.

Outputs Recall@5, MRR, plus sample top-k rankings so we can eyeball
whether 2M's quality is good enough for "ships with the engine."

Usage:
  python scratch/eval_potion_2m.py
"""
from __future__ import annotations

import os
import re
import time
from typing import List, Tuple

import numpy as np

# ── 1. Test corpus: yantrikdb-shaped memories ──
#
# Short factual sentences a typical user would record into yantrikdb.
# Spread across domains so we exercise both lexical and semantic recall.
MEMORIES: List[str] = [
    # people / org
    "Alice Chen is the engineering lead at Acme Corp",
    "Bob runs the platform team and reports to Alice",
    "Carol is the head of design, formerly at Pinterest",
    "David from finance closed the Series C last quarter",
    "Eve is our security engineer; she previously worked at Cloudflare",
    "Frank is a contractor on the mobile team",

    # decisions
    "We decided to use PostgreSQL instead of MongoDB for the new service",
    "Switched the build system to Bazel after evaluating Buck and Pants",
    "Chose Rust for the embedding engine over Go",
    "Picked Stripe for payments — Adyen was the runner-up",
    "Adopted gRPC for internal service-to-service calls",

    # project context
    "Project Atlas launches in March and depends on the new auth system",
    "The migration to Kubernetes is scheduled for Q3",
    "API rate limits will be cut from 1000 rpm to 500 rpm next month",
    "The website redesign is being led by Carol's team",
    "Mobile app version 4.0 ships in June with offline mode",

    # preferences
    "User prefers dark mode in VS Code",
    "User uses tabs over spaces for indentation",
    "User likes async-first communication; standups are written, not verbal",
    "User's preferred deploy window is Tuesday morning, never Friday",

    # technical facts
    "Our HNSW vector index uses cosine distance, not L2",
    "The cluster runs on three nodes with openraft consensus",
    "Embedding dimension is 384 for the default model",
    "WAL mode is enabled with autocheckpoint at 1000 pages",
    "The compactor wakes every 250ms in v0.6.7+",

    # episodic / events
    "Met with the architecture review board on Tuesday about the wedge fix",
    "Had a rough day — production p99 spiked to 1.2s",
    "Lunch with David, discussed the upcoming offsite",
    "Onboarded two new engineers this week, both on the platform team",
    "Outage on March 15 caused by a misconfigured load balancer",
]

# ── 2. Queries with ground-truth top-k ──
#
# Each query has a list of memory indices that are "relevant" (could
# reasonably be the answer). Recall@5 = fraction of these that show
# up in the model's top-5.
QUERIES: List[Tuple[str, List[int]]] = [
    ("who leads engineering?", [0, 1]),                       # Alice + Bob
    ("which database did we pick?", [6]),                     # PostgreSQL
    ("when does Project Atlas launch?", [11]),                # Atlas
    ("user's editor preferences", [16, 17]),                  # dark mode + tabs
    ("payment processor decision", [9]),                      # Stripe
    ("what's our build system?", [7]),                        # Bazel
    ("vector index distance metric", [20]),                   # HNSW cosine
    ("recent production incident", [26, 29]),                 # rough day p99 + outage
    ("who works on the design team?", [2, 14]),               # Carol + website redesign
    ("default embedding model dimension", [22]),              # dim=384
]


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return a @ b.T


def recall_at_k(rankings: List[List[int]], gt: List[List[int]], k: int) -> float:
    hits = 0
    total = 0
    for r, g in zip(rankings, gt):
        if not g:
            continue
        topk = set(r[:k])
        hits += sum(1 for gi in g if gi in topk) / len(g)
        total += 1
    return hits / total if total else 0.0


def mrr(rankings: List[List[int]], gt: List[List[int]]) -> float:
    score = 0.0
    n = 0
    for r, g in zip(rankings, gt):
        if not g:
            continue
        for rank, idx in enumerate(r, 1):
            if idx in g:
                score += 1.0 / rank
                break
        n += 1
    return score / n if n else 0.0


def eval_embedder(name: str, encode_many) -> Tuple[float, float, float, List[List[int]]]:
    t0 = time.time()
    mem_emb = encode_many(MEMORIES)
    q_emb = encode_many([q for q, _ in QUERIES])
    sims = cos(np.asarray(q_emb), np.asarray(mem_emb))
    rankings = sims.argsort(axis=1)[:, ::-1].tolist()  # descending
    elapsed = time.time() - t0
    gt = [g for _, g in QUERIES]
    r5 = recall_at_k(rankings, gt, 5)
    r10 = recall_at_k(rankings, gt, 10)
    return r5, r10, mrr(rankings, gt), elapsed, rankings


# ── 3. Slice A baseline: hash-trick TF-IDF (port of crates/yantrikdb-core/src/embedder/default.rs) ──
def hash_trick_embed(text: str, dim: int = 384) -> np.ndarray:
    tokens = [t.lower() for t in re.split(r"[^A-Za-z0-9]+", text) if t]
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    v = np.zeros(dim, dtype=np.float32)
    for tok, count in tf.items():
        # FNV-1a-shaped 64-bit hash, matching the Rust impl
        h = 0xcbf29ce484222325
        for b in tok.encode("utf-8"):
            h ^= b
            h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
        bucket = h % dim
        sign = 1.0 if (((h * 0x9e3779b97f4a7c15) >> 33) & 1) == 0 else -1.0
        weight = float(np.log1p(count))
        v[bucket] += sign * weight
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def show_top(name: str, rankings, k=5):
    print(f"\n--- {name} top-{k} per query ---")
    for (q, gt), r in zip(QUERIES, rankings):
        topk = r[:k]
        marks = ["+" if i in gt else " " for i in topk]
        snippets = [f"{m} [{i:2}] {MEMORIES[i][:60]}" for m, i in zip(marks, topk)]
        gt_str = "/".join(str(i) for i in gt) if gt else "-"
        print(f"  Q: {q!r:50}  gt={gt_str}")
        for s in snippets:
            print(f"      {s}")


def main():
    # Ensure deterministic numpy printing
    np.set_printoptions(precision=3, suppress=True)

    print(f"corpus: {len(MEMORIES)} memories, {len(QUERIES)} queries")
    print(f"ground-truth coverage: {sum(len(g) for _, g in QUERIES)} relevant hits across queries\n")

    results = {}

    # 1. Slice A baseline: hash-trick (in-memory)
    print("=== Slice A: hash-trick TF-IDF (current shipped Rust default) ===")
    def hash_encode(texts):
        return np.stack([hash_trick_embed(t, 384) for t in texts])
    results["hash-trick"] = eval_embedder("hash-trick", hash_encode)

    # 2. potion-base-2M (Slice B candidate)
    print("\n=== Slice B candidate: minishlab/potion-base-2M ===")
    from model2vec import StaticModel
    m2 = StaticModel.from_pretrained("minishlab/potion-base-2M")
    print(f"  potion-base-2M dim: {len(m2.encode(['x'])[0])}")
    def potion_encode(texts):
        return np.asarray(m2.encode(texts))
    results["potion-base-2M"] = eval_embedder("potion-base-2M", potion_encode)

    # 3. all-MiniLM-L6-v2 (reference quality, ~80MB)
    print("\n=== Reference: all-MiniLM-L6-v2 (sentence-transformers) ===")
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    def st_encode(texts):
        return np.asarray(st.encode(texts, show_progress_bar=False))
    results["all-MiniLM-L6-v2"] = eval_embedder("all-MiniLM-L6-v2", st_encode)

    # ── Summary table ──
    print("\n\n=== summary ===")
    print(f"{'embedder':<24} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'time':>10}")
    print("-" * 64)
    for name, (r5, r10, mrr_, t, _) in results.items():
        print(f"{name:<24} {r5:>8.3f} {r10:>8.3f} {mrr_:>8.3f} {t:>9.2f}s")

    # ── Top-k rankings for visual inspection ──
    for name, (_, _, _, _, rankings) in results.items():
        show_top(name, rankings, k=5)


if __name__ == "__main__":
    main()
