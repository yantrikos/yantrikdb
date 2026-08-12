"""Cluster-level statistics for any pair of frozen-context conditions.

BEAM's 400 queries are nested inside 20 conversations, so they are not
independent observations. A paired test over the 400 pairs (McNemar) treats
them as if they were and overstates significance — on the A-vs-E pair it
reported p=0.00077 where the conversation-level test gives p=0.016, a ~20x
difference. Every test here therefore operates on CONVERSATIONS.

With exactly 20 conversations the sign-flip permutation is EXACT: 2^20 =
1,048,576 assignments, enumerable in seconds. No sampling, no asymptotics.

    python frozen_stats.py A-yantrikdb-ctx R-hindsight-rag-ctx
"""
import argparse
import itertools
import json
import random
import statistics
import sys
from collections import defaultdict

from memory_bench.utils import count_tokens


def load(label: str) -> dict:
    """Load a condition, RECOMPUTING context_tokens from the context text.

    The stored `context_tokens` field is NOT trustworthy across files: runs
    made before the tokenizer fix wrote chars//4 into it, later runs write
    real cl100k_base counts. Both are in outputs/ right now, and the error is
    not a wash — Hindsight's markup-heavy statements run 3.82 chars/token
    against our prose at 4.63, so the proxy inflates our count relative to
    theirs and turned a 23% size advantage into 6%. Recomputing from `context`
    makes every condition comparable regardless of when it was produced.
    """
    with open(f"outputs/frozen-{label}.json", encoding="utf-8") as f:
        rows = json.load(f)["results"]
    for r in rows:
        r["context_tokens"] = count_tokens(r["context"])
    return {r["query_id"]: r for r in rows}


def exact_sign_flip(diffs: list[float]) -> tuple[float, float]:
    """Exact two-sided p over all 2^n conversation sign assignments."""
    obs = sum(diffs)
    extreme = sum(
        1
        for signs in itertools.product((-1, 1), repeat=len(diffs))
        if abs(sum(s * d for s, d in zip(signs, diffs))) >= abs(obs)
    )
    return obs, extreme / (2 ** len(diffs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("left", help="label of condition A (the reference)")
    ap.add_argument("right", help="label of condition B")
    ap.add_argument("--boots", type=int, default=4000)
    a = ap.parse_args()

    L, R = load(a.left), load(a.right)
    common = sorted(set(L) & set(R))
    convs = defaultdict(list)
    for q in common:
        convs[q.split("_")[0]].append(q)
    keys = sorted(convs)
    print(f"{a.left}  vs  {a.right}")
    print(f"{len(common)} paired queries across {len(keys)} conversations\n")

    lb = sum(1 for q in common if L[q]["correct"])
    rb = sum(1 for q in common if R[q]["correct"])
    lr = sum(L[q]["score"] or 0 for q in common) / len(common)
    rr = sum(R[q]["score"] or 0 for q in common) / len(common)
    lt = sum(L[q]["context_tokens"] for q in common) / len(common)
    rt = sum(R[q]["context_tokens"] for q in common) / len(common)
    print(f"  {a.left:24} {lb}/{len(common)} = {lb/len(common):.1%} binary | {lr:.3f} rubric | {lt:,.0f} ctx")
    print(f"  {a.right:24} {rb}/{len(common)} = {rb/len(common):.1%} binary | {rr:.3f} rubric | {rt:,.0f} ctx")
    print(f"  delta (left - right): {lb-rb:+d} binary, {lr-rr:+.4f} rubric, {lt-rt:+,.0f} ctx\n")

    dbin = [sum(int(L[q]["correct"]) - int(R[q]["correct"]) for q in convs[k]) for k in keys]
    drub = [
        sum((L[q]["score"] or 0) - (R[q]["score"] or 0) for q in convs[k]) / len(convs[k])
        for k in keys
    ]

    print("EXACT CLUSTER SIGN-FLIP (all 2^%d permutations)" % len(keys))
    for name, d in (("binary", dbin), ("rubric", drub)):
        obs, p = exact_sign_flip(d)
        verdict = "significant" if p < 0.05 else "NOT significant at 0.05"
        print(f"  {name:7} observed {obs:+.4g}  p = {p:.5f}  -> {verdict}")

    print("\nCONVERSATION-CLUSTERED BOOTSTRAP (rubric)")
    random.seed(11)
    deltas = []
    for _ in range(a.boots):
        samp = [random.choice(keys) for _ in keys]
        qs = [q for k in samp for q in convs[k]]
        deltas.append(
            sum(L[q]["score"] or 0 for q in qs) / len(qs)
            - sum(R[q]["score"] or 0 for q in qs) / len(qs)
        )
    deltas.sort()
    lo, hi = deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas))]
    straddles = lo <= 0 <= hi
    print(f"  delta {statistics.mean(deltas):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  {'CI STRADDLES ZERO — consistent with equivalence' if straddles else 'CI excludes zero'}")

    print("\nLEAVE-ONE-CONVERSATION-OUT")
    worst_b = min(sum(d for j, d in enumerate(dbin) if j != i) for i in range(len(keys)))
    worst_r = min(
        statistics.mean([d for j, d in enumerate(drub) if j != i]) for i in range(len(keys))
    )
    best_b = max(sum(d for j, d in enumerate(dbin) if j != i) for i in range(len(keys)))
    print(f"  binary delta range {worst_b:+d} .. {best_b:+d}")
    print(f"  rubric worst case  {worst_r:+.4f}")
    sign_stable = (worst_b > 0) == (best_b > 0)
    print(f"  direction stable across all removals: {sign_stable}")

    print("\nPER-CONVERSATION (binary, left - right)")
    print("  " + " ".join(f"{d:+d}" for d in dbin))
    print(f"  favouring left: {sum(1 for d in dbin if d>0)}  tied: {sum(1 for d in dbin if d==0)}"
          f"  favouring right: {sum(1 for d in dbin if d<0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
