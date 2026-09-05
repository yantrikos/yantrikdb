#!/usr/bin/env python3
"""Judge-free preflight 3: can LEXICAL structure (not embeddings) find the
gold first-mention turns for BEAM event-ordering queries?

Follow-up to first_mention_preflight.py (embedding novelty, clustering) and
session_stratified_preflight.py (session structure), both at chance. Those
measured similarity-shaped signals; within one conversation every turn is
about the same project, so similarity is flat. This preflight measures two
discourse-shaped, model-free signals instead:

  thread_opener   the first user turn that introduces a content term which
                  later recurs in >= r other user turns ("opens a thread");
                  a turn's opener score is the number of such terms it
                  introduces, optionally gated by engine relevance.
  text_tiling     TextTiling (Hearst 1997): lexical-cohesion valleys between
                  adjacent windows of user turns mark topic boundaries; the
                  first user turn after each boundary is a candidate, ranked
                  by boundary depth.

Same gold, same quarantine, same scoring as the earlier preflights; gold is
joined only at scoring. Stores written by first_mention_preflight.py are
reopened, not rebuilt, and no embedding is used anywhere.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

import yantrikdb

try:
    from .first_mention_preflight import (
        QUARANTINE, load_docs, load_gold, requested_count, sha256_file, split_turns,
    )
except ImportError:  # pragma: no cover - direct script execution
    from first_mention_preflight import (
        QUARANTINE, load_docs, load_gold, requested_count, sha256_file, split_turns,
    )

_TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]{3,}")
STOPWORDS = set(
    """
    that this with from have will would could should about there their they them
    then than what when where which while your yours ours into onto over under
    also just like make made making need needs want wants really very much more
    most some such only other another each every both been being were does done
    doing here able sure okay thanks thank please help helps helping great good
    using used uses know knows think thought still already since after before
    between through during without within again back next last first second
    time times week weeks month months year years today tomorrow yesterday
    project projects work working works things thing something anything
    everything nothing right well feel feels felt maybe might must shall
    """.split()
)


def content_terms(text: str) -> list[str]:
    body = text.split(":", 1)[1] if "] User:" in text or "] Assistant:" in text else text
    return [t for t in _TOKEN_RE.findall(body.lower()) if t not in STOPWORDS]


def user_turns_of(docs: list[dict]) -> list[dict]:
    seen, turns = set(), []
    for d in docs:
        for t in split_turns(d["content"]):
            if t["role"] != "user" or t["turn"] is None or t["turn"] in seen:
                continue
            seen.add(t["turn"])
            t["terms"] = content_terms(t["text"])
            t["tf"] = Counter(t["terms"])
            turns.append(t)
    turns.sort(key=lambda t: t["turn"])
    return turns


# ---------------------------------------------------------------- signals
def opener_scores(turns: list[dict], min_recur: int) -> dict[int, float]:
    """Number of terms a turn introduces that recur in >= min_recur LATER turns."""
    df: Counter = Counter()
    for t in turns:
        df.update(set(t["terms"]))
    first_seen: dict[str, int] = {}
    scores: dict[int, float] = defaultdict(float)
    for t in turns:
        for term in set(t["terms"]):
            if term not in first_seen:
                first_seen[term] = t["turn"]
                if df[term] - 1 >= min_recur:
                    scores[t["turn"]] += 1.0
    return scores


def tiling_depths(turns: list[dict], window: int) -> dict[int, float]:
    """TextTiling depth score for the gap BEFORE each turn (turn i starts a
    new tile when the cohesion valley at i is deep relative to its peaks)."""
    n = len(turns)
    if n < 3:
        return {}

    def block_vec(lo: int, hi: int) -> Counter:
        c: Counter = Counter()
        for t in turns[max(0, lo):min(n, hi)]:
            c.update(t["tf"])
        return c

    def cos(a: Counter, b: Counter) -> float:
        num = sum(a[k] * b[k] for k in a if k in b)
        da = math.sqrt(sum(v * v for v in a.values())) or 1.0
        dbb = math.sqrt(sum(v * v for v in b.values())) or 1.0
        return num / (da * dbb)

    gaps = [cos(block_vec(i - window, i), block_vec(i, i + window)) for i in range(1, n)]
    depths: dict[int, float] = {}
    for gi, g in enumerate(gaps):
        left = g
        for j in range(gi - 1, -1, -1):
            if gaps[j] >= left:
                left = gaps[j]
            else:
                break
        right = g
        for j in range(gi + 1, len(gaps)):
            if gaps[j] >= right:
                right = gaps[j]
            else:
                break
        depths[turns[gi + 1]["turn"]] = (left - g) + (right - g)
    depths[turns[0]["turn"]] = max(depths.values(), default=0.0) + 1.0  # first turn opens a tile
    return depths


# ---------------------------------------------------------------- selectors
def select(turns, signal: dict[int, float], relevance: dict[int, float], n: int, params):
    floor, cap = params["floor"], params["cap_mult"] * n
    best = max(relevance.values(), default=0.0)
    eligible = [t for t in turns if signal.get(t["turn"], 0.0) > 0 and relevance.get(t["turn"], 0.0) >= floor * best]
    if params.get("mix"):  # rank by signal * relevance ratio
        key = lambda t: (-(signal[t["turn"]] * (relevance.get(t["turn"], 0.0) / best if best else 1.0)), t["turn"])
    else:
        key = lambda t: (-signal[t["turn"]], t["turn"])
    picked = sorted(eligible, key=key)[:cap]
    return sorted(picked, key=lambda t: t["turn"])


def grid():
    for min_recur in (2, 3, 5):
        for floor in (0.0, 0.5):
            for mix in (False, True):
                for cap in (1, 2, 3):
                    yield "thread_opener", {"min_recur": min_recur, "floor": floor, "mix": mix, "cap_mult": cap}
    for window in (1, 2, 3):
        for floor in (0.0, 0.5):
            for cap in (1, 2, 3):
                yield "text_tiling", {"window": window, "floor": floor, "mix": False, "cap_mult": cap}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--documents", type=Path, required=True)
    ap.add_argument("--beam-source", type=Path, required=True)
    ap.add_argument("--store-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--recall-top-k", type=int, default=400)
    ap.add_argument("--default-n", type=int, default=5)
    args = ap.parse_args()

    by_user = load_docs(args.documents)
    gold = load_gold(args.beam_source)
    turns_by_user = {uid: user_turns_of(docs) for uid, docs in by_user.items()}
    stores = {uid: yantrikdb.YantrikDB.with_default(str(args.store_dir / f"{uid}.db")) for uid in by_user}

    relevance: dict[str, dict[int, float]] = {}
    for qid, g in gold.items():
        hits = stores[g["user_id"]].recall(query=g["query"], top_k=args.recall_top_k, source="user", skip_reinforce=True)
        rel: dict[int, float] = {}
        for h in hits:
            tid = (h.get("metadata") or {}).get("turn_id")
            if tid is not None:
                rel[int(tid)] = max(rel.get(int(tid), 0.0), float(h.get("score") or 0.0))
        relevance[qid] = rel

    # Signal diagnostic: do gold turns carry more opener/tiling signal than non-gold?
    diag = {}
    for name, fn, arg in (("opener_r3", opener_scores, 3), ("tiling_w2", tiling_depths, 2)):
        gv, nv, gold_pct = [], [], []
        for qid, g in gold.items():
            turns = turns_by_user[g["user_id"]]
            sig = fn(turns, arg)
            gset = set(g["source_turns"])
            ranked = sorted(turns, key=lambda t: -sig.get(t["turn"], 0.0))
            rank = {t["turn"]: i for i, t in enumerate(ranked)}
            for t in turns:
                (gv if t["turn"] in gset else nv).append(sig.get(t["turn"], 0.0))
                if t["turn"] in gset:
                    gold_pct.append(rank[t["turn"]] / max(1, len(turns) - 1))
        diag[name] = {"gold_mean": fmean(gv), "nongold_mean": fmean(nv), "gold_rank_pct": fmean(gold_pct)}
    print("signal diagnostic:", json.dumps(diag))

    arms = []
    for name, params in grid():
        rows = []
        for qid, g in gold.items():
            turns = turns_by_user[g["user_id"]]
            sig = opener_scores(turns, params["min_recur"]) if name == "thread_opener" else tiling_depths(turns, params["window"])
            n = requested_count(g["query"]) or args.default_n
            sel = select(turns, sig, relevance[qid], n, params)
            gset, picked = set(g["source_turns"]), {t["turn"] for t in sel}
            hit = len(gset & picked)
            rows.append({"query_id": qid, "quarantined": qid in QUARANTINE, "rows": len(sel), "hit": hit,
                         "recall": hit / len(gset), "precision": hit / len(picked) if picked else 0.0,
                         "selected_turns": sorted(picked)})
        clean = [r for r in rows if not r["quarantined"]]
        agg = {"n": len(clean), "recall": fmean(r["recall"] for r in clean), "precision": fmean(r["precision"] for r in clean),
               "rows": fmean(r["rows"] for r in clean), "queries_recall_ge_0_8": sum(r["recall"] >= 0.8 for r in clean),
               "queries_recall_0": sum(r["recall"] == 0 for r in clean)}
        arms.append({"selector": name, "params": params, "clean": agg, "rows": rows})

    print(f"\n{'selector':14s} {'params':58s} {'recall':>7s} {'prec':>6s} {'rows':>5s} {'>=.8':>5s} {'=0':>3s}")
    for a in arms:
        c = a["clean"]
        print(f"{a['selector']:14s} {json.dumps(a['params']):58s} {c['recall']:7.3f} {c['precision']:6.3f} {c['rows']:5.1f} {c['queries_recall_ge_0_8']:5d} {c['queries_recall_0']:3d}")

    json.dump({"protocol": "lexical-opener-preflight-v1", "engine_version": yantrikdb.__version__,
               "documents_sha256": sha256_file(args.documents), "beam_source_sha256": sha256_file(args.beam_source),
               "gold_is_selector_input": False, "quarantine": sorted(QUARANTINE), "signal_diagnostic": diag, "arms": arms},
              open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {args.out}")
    for db in stores.values():
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
