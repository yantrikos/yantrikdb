#!/usr/bin/env python3
"""Judge-free preflight: can an LLM-free, query-only selector pick the
first-mention user turns that BEAM event-ordering gold is built from?

Why this exists (frozen facts, 2026-08-19..26):
- The frozen control (ydb-0151, k=40 turn-aware chunks, ~18.5K tokens)
  reaches 18.3% of the authoritative gold source turns.
- The role-aware user-only arm (84 user turns, 20.5K tokens) reaches 95.3%
  of them and scored the SAME as the control (0.287 vs 0.293, paired null).
- Five exact gold turns handed to the same reader score 0.9; an LLM
  concern selector picking ~5 items scored 0.7-1.0 on Q9.
So reachability is solved and the reader drowns in selection. The open
question is whether a SMALL, PRECISE set can be selected WITHOUT an LLM.

This script measures exactly that, with no model or judge call:
for each of the 40 event-ordering queries, every selector sees only the
query text and the conversation's user turns (engine relevance score +
turn order + an embedding for novelty), and is scored against the gold
source turn ids by recall, precision and size. Gold is never an input to
any selector; it is joined only at scoring time.

Selectors:
  relevance_topk      the role-aware arm (control reproduction)
  chrono_stratified   N turns evenly spaced over the relevant set (null
                      control: chronology without novelty)
  first_mention       chronological greedy novelty: keep a relevant turn
                      only if it is not near-duplicate of an already-kept
                      earlier turn (the "first time I brought X up" rule)
  cluster_first       cluster the top-M relevant turns, take the earliest
                      turn per cluster, keep the N strongest clusters

Every parameter combination is reported; nothing is selected post hoc.
The three benchmark-defect rows named in EVENT_ORDERING_V5_AUTOPSY.md are
reported separately and excluded from the clean means.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

# Exact BEAM header forms, copied from the AMB provider (never loosened —
# a permissive bracket pattern matches markdown links and code indices).
_HEADER = r"\[(?:[A-Z][a-z]+-\d+-\d+(?: \| Turn \d+)?|Turn \d+)\]"
_TURN_SPLIT_RE = re.compile(rf"(?=\n*{_HEADER})")
_ROLE_RE = re.compile(rf"^(?P<header>{_HEADER})\s+(?P<role>User|Assistant):\s*", re.I)
_TURN_RE = re.compile(r"\bTurn\s+(\d+)\b", re.I)
_DATE_RE = re.compile(r"\[([A-Z][a-z]+)-(\d+)-(\d+)")

QUARANTINE = {"9_event_ordering_0", "18_event_ordering_0", "19_event_ordering_0"}
COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_ONLY_RE = re.compile(r"(?i)only and only (\w+)")
_TEMPLATE_RE = re.compile(
    r"(?i)^can you (?:list|walk me through)(?: the order in which)?(?: in order how)? i brought up "
    r"(?:the )?different |throughout our (?:conversations?|chats?|discussions?).*$|"
    r"across our (?:conversations?|chats?|discussions?).*$|during our (?:conversations?|chats?).*$|"
    r", in order\??.*$"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_date(header: str) -> float | None:
    m = _DATE_RE.search(header)
    if not m:
        return None
    try:
        return datetime.strptime(" ".join(m.groups()), "%B %d %Y").replace(
            tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def split_turns(content: str) -> list[dict]:
    out = []
    for part in (p.strip() for p in _TURN_SPLIT_RE.split(content)):
        if not part:
            continue
        rm = _ROLE_RE.match(part)
        role = rm.group("role").lower() if rm else "unknown"
        tm = _TURN_RE.search(part)
        turn = int(tm.group(1)) if tm else None
        out.append({"text": part, "role": role, "turn": turn,
                    "created_at": parse_date(rm.group("header")) if rm else None})
    return out


def load_docs(path: Path) -> dict[str, list[dict]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        docs = json.load(f)
    by_user: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        by_user[str(d["user_id"])].append(d)
    return by_user


def load_gold(beam_source: Path) -> dict[str, dict]:
    """Authoritative source turns straight from BEAM's probing questions."""
    import ast
    convs = json.load(open(beam_source, encoding="utf-8"))
    gold = {}
    for c in convs:
        pq = c["probing_questions"]
        pq = ast.literal_eval(pq) if isinstance(pq, str) else pq
        for i, q in enumerate(pq["event_ordering"]):
            ids = []
            stack = list(q["source_chat_ids"])
            while stack:
                x = stack.pop(0)
                if isinstance(x, list):
                    stack = list(x) + stack
                else:
                    ids.append(int(x))
            qid = f"{c['conversation_id']}_event_ordering_{i}"
            gold[qid] = {"query": q["question"], "source_turns": sorted(set(ids)),
                         "total_mentions": q.get("total_mentions"),
                         "user_id": str(c["conversation_id"])}
    return gold


def requested_count(query: str) -> int | None:
    m = _ONLY_RE.search(query)
    if not m:
        return None
    w = m.group(1).lower()
    return int(w) if w.isdigit() else COUNT_WORDS.get(w)


def focus_of(query: str) -> str:
    q = query.strip()
    q = re.sub(r"(?i)^can you (?:list|walk me through)(?: the order in which| in order how)? i brought up (?:the )?different ", "", q)
    q = re.sub(r"(?i)\b(?:throughout|across|during|over) our (?:conversations?|chats?|discussions?|sessions?).*$", "", q)
    q = re.sub(r"(?i),? in order\??.*$", "", q)
    return q.strip(" ,.?")


# ---------------------------------------------------------------- embeddings
class NomicEmbedder:
    def __init__(self, cache_path: Path, model: str = "nomic-embed-text",
                 host: str = "http://127.0.0.1:11434"):
        import requests
        self._rq = requests
        self.model, self.host, self.cache_path = model, host, cache_path
        self.cache: dict[str, list[float]] = {}
        if cache_path.exists():
            self.cache = json.load(open(cache_path, encoding="utf-8"))
        self.dirty = 0

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, text: str) -> list[float]:
        k = self.key(text)
        v = self.cache.get(k)
        if v is None:
            r = self._rq.post(f"{self.host}/api/embeddings",
                              json={"model": self.model, "prompt": text[:8000]}, timeout=120)
            r.raise_for_status()
            v = r.json()["embedding"]
            self.cache[k] = v
            self.dirty += 1
            if self.dirty % 200 == 0:
                self.flush()
        return v

    def flush(self):
        json.dump(self.cache, open(self.cache_path, "w", encoding="utf-8"))


def cosine(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(x * x for x in b)) or 1.0
    return num / (da * db)


# ---------------------------------------------------------------- selectors
def sel_relevance_topk(cands, n, params):
    """Role-aware arm: relevance order, token budget of k*512."""
    budget = params["budget_tokens"]
    out, used = [], 0
    for c in sorted(cands, key=lambda c: -c["score"]):
        if out and used >= budget:
            break
        out.append(c)
        used += c["tokens"]
    return out


def sel_chrono_stratified(cands, n, params):
    """Null control: N turns evenly spaced over the relevant set, no novelty."""
    floor = params["floor"]
    best = max((c["score"] for c in cands), default=0.0)
    rel = sorted((c for c in cands if c["score"] >= floor * best), key=lambda c: c["turn"])
    cap = params["cap_mult"] * n
    if len(rel) <= cap:
        return rel
    idx = [round(i * (len(rel) - 1) / (cap - 1)) for i in range(cap)] if cap > 1 else [0]
    return [rel[i] for i in sorted(set(idx))]


def sel_first_mention(cands, n, params):
    """Chronological greedy novelty: the first relevant turn that is not a
    near-duplicate of an earlier kept turn opens a new 'aspect'."""
    floor, theta, cap = params["floor"], params["theta"], params["cap_mult"] * n
    best = max((c["score"] for c in cands), default=0.0)
    kept = []
    for c in sorted(cands, key=lambda c: c["turn"]):
        if c["score"] < floor * best:
            continue
        if kept and max(cosine(c["emb"], k["emb"]) for k in kept) >= theta:
            continue
        kept.append(c)
        if len(kept) >= cap:
            break
    return kept


def sel_cluster_first(cands, n, params):
    """Cluster the top-M relevant turns (greedy leader clustering in
    relevance order), take the EARLIEST turn of each cluster, keep the N
    clusters with the highest relevance mass, present chronologically."""
    theta, m, cap = params["theta"], params["top_m"], params["cap_mult"] * n
    top = sorted(cands, key=lambda c: -c["score"])[:m]
    clusters: list[dict] = []
    for c in top:
        home = None
        best_sim = theta
        for cl in clusters:
            s = cosine(c["emb"], cl["leader"]["emb"])
            if s >= best_sim:
                home, best_sim = cl, s
        if home is None:
            clusters.append({"leader": c, "members": [c], "mass": c["score"]})
        else:
            home["members"].append(c)
            home["mass"] += c["score"]
    clusters.sort(key=lambda cl: -cl["mass"])
    picked = [min(cl["members"], key=lambda x: x["turn"]) for cl in clusters[:cap]]
    return sorted(picked, key=lambda c: c["turn"])


SELECTORS = {
    "relevance_topk": sel_relevance_topk,
    "chrono_stratified": sel_chrono_stratified,
    "first_mention": sel_first_mention,
    "cluster_first": sel_cluster_first,
}


def grid():
    yield "relevance_topk", {"budget_tokens": 40 * 512}
    for floor in (0.0, 0.5):
        for cap in (1, 2, 3):
            yield "chrono_stratified", {"floor": floor, "cap_mult": cap}
    for floor in (0.0, 0.5):
        for theta in (0.60, 0.70, 0.80, 0.90):
            for cap in (1, 2, 3):
                yield "first_mention", {"floor": floor, "theta": theta, "cap_mult": cap}
    for top_m in (20, 40):
        for theta in (0.60, 0.70, 0.80):
            for cap in (1, 2):
                yield "cluster_first", {"top_m": top_m, "theta": theta, "cap_mult": cap}


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--documents", type=Path, required=True)
    ap.add_argument("--beam-source", type=Path, required=True)
    ap.add_argument("--membership", type=Path,
                    help="event40-organizer-membership-v2.json, used only to cross-check gold")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--embed-cache", type=Path, required=True)
    ap.add_argument("--store-dir", type=Path)
    ap.add_argument("--default-n", type=int, default=5)
    ap.add_argument("--recall-top-k", type=int, default=400)
    ap.add_argument("--novelty-embedder", choices=["nomic", "engine"], default="nomic")
    args = ap.parse_args()

    import tiktoken
    import yantrikdb
    enc = tiktoken.get_encoding("cl100k_base")

    by_user = load_docs(args.documents)
    gold = load_gold(args.beam_source)
    if args.membership:
        mem = json.load(open(args.membership, encoding="utf-8"))
        mismatch = [r["query_id"] for r in mem["results"]
                    if sorted(r["source_turns"]) != gold[r["query_id"]]["source_turns"]]
        if mismatch:
            print("GOLD MISMATCH vs membership artifact:", mismatch, file=sys.stderr)
            return 2
        print(f"gold cross-check vs membership artifact: 40/40 identical")

    store_dir = args.store_dir or Path(tempfile.mkdtemp(prefix="fm-preflight-"))
    store_dir.mkdir(parents=True, exist_ok=True)
    embedder = NomicEmbedder(args.embed_cache) if args.novelty_embedder == "nomic" else None

    # ---- ingest: every turn, per conversation, exactly as the role-aware arm
    t0 = time.perf_counter()
    stores: dict[str, object] = {}
    user_turns: dict[str, dict[int, dict]] = {}
    for uid, docs in by_user.items():
        path = store_dir / f"{uid}.db"
        for f in store_dir.glob(f"{uid}.db*"):
            f.unlink()
        db = yantrikdb.YantrikDB.with_default(str(path))
        turns_here: dict[int, dict] = {}
        for d in docs:
            for t in split_turns(d["content"]):
                if t["turn"] is None:
                    continue
                db.record(t["text"], memory_type="episodic",
                          metadata={"doc_id": d["id"], "turn_id": t["turn"], "speaker_role": t["role"]},
                          source=t["role"] if t["role"] in ("user", "assistant") else "user",
                          created_at=t["created_at"])
                if t["role"] == "user" and t["turn"] not in turns_here:
                    turns_here[t["turn"]] = {"turn": t["turn"], "text": t["text"],
                                             "tokens": len(enc.encode(t["text"], disallowed_special=())),
                                             "created_at": t["created_at"]}
        stores[uid] = db
        user_turns[uid] = turns_here
    print(f"ingested {len(stores)} conversations in {time.perf_counter()-t0:.1f}s")

    # ---- novelty embeddings for every user turn
    t0 = time.perf_counter()
    for uid, turns in user_turns.items():
        db = stores[uid]
        for t in turns.values():
            t["emb"] = embedder.embed(t["text"]) if embedder else list(db.embed(t["text"]))
    if embedder:
        embedder.flush()
    print(f"embedded {sum(len(v) for v in user_turns.values())} user turns in {time.perf_counter()-t0:.1f}s")

    # ---- per-query candidate pools (query-only)
    pools: dict[str, list[dict]] = {}
    qinfo: dict[str, dict] = {}
    for qid, g in gold.items():
        db = stores[g["user_id"]]
        hits = db.recall(query=g["query"], top_k=args.recall_top_k, source="user", skip_reinforce=True)
        scores = {}
        for h in hits:
            tid = (h.get("metadata") or {}).get("turn_id")
            if tid is not None:
                scores[int(tid)] = max(scores.get(int(tid), 0.0), float(h.get("score") or 0.0))
        cands = []
        for tid, t in user_turns[g["user_id"]].items():
            c = dict(t)
            c["score"] = scores.get(tid, 0.0)
            cands.append(c)
        pools[qid] = cands
        n = requested_count(g["query"])
        qinfo[qid] = {"requested_n": n, "n_used": n or args.default_n, "focus": focus_of(g["query"]),
                      "user_turns": len(cands), "recalled": len(scores),
                      "gold": g["source_turns"], "quarantined": qid in QUARANTINE}

    # ---- novelty diagnostic: are gold turns novel w.r.t. EARLIER user turns?
    diag = {"gold_max_prior_sim": [], "nongold_max_prior_sim": [],
            "gold_rel_rank_pct": [], "gold_score_ratio": []}
    for qid, cands in pools.items():
        gset = set(qinfo[qid]["gold"])
        ordered = sorted(cands, key=lambda c: c["turn"])
        by_rel = sorted(cands, key=lambda c: -c["score"])
        best = by_rel[0]["score"] if by_rel and by_rel[0]["score"] > 0 else 1.0
        rank = {c["turn"]: i for i, c in enumerate(by_rel)}
        for i, c in enumerate(ordered):
            prior = ordered[:i]
            s = max((cosine(c["emb"], p["emb"]) for p in prior), default=0.0)
            (diag["gold_max_prior_sim"] if c["turn"] in gset else diag["nongold_max_prior_sim"]).append(s)
            if c["turn"] in gset:
                diag["gold_rel_rank_pct"].append(rank[c["turn"]] / max(1, len(by_rel) - 1))
                diag["gold_score_ratio"].append(c["score"] / best)
    diag_summary = {k: (fmean(v) if v else None) for k, v in diag.items()}
    diag_summary["n_gold"] = len(diag["gold_max_prior_sim"])
    diag_summary["n_nongold"] = len(diag["nongold_max_prior_sim"])
    print("novelty diagnostic:", json.dumps(diag_summary))

    # ---- run the grid
    arms = []
    for name, params in grid():
        fn = SELECTORS[name]
        rows = []
        for qid, cands in pools.items():
            n = qinfo[qid]["n_used"]
            sel = fn(cands, n, params)
            gset = set(qinfo[qid]["gold"])
            picked = {c["turn"] for c in sel}
            hit = len(picked & gset)
            rows.append({"query_id": qid, "quarantined": qid in QUARANTINE,
                         "rows": len(sel), "tokens": sum(c["tokens"] for c in sel),
                         "hit": hit, "recall": hit / len(gset) if gset else 0.0,
                         "precision": hit / len(picked) if picked else 0.0,
                         "selected_turns": sorted(picked)})
        clean = [r for r in rows if not r["quarantined"]]

        def agg(rs):
            return {"n": len(rs), "recall": fmean(r["recall"] for r in rs),
                    "precision": fmean(r["precision"] for r in rs),
                    "rows": fmean(r["rows"] for r in rs), "tokens": fmean(r["tokens"] for r in rs),
                    "queries_recall_ge_0_8": sum(r["recall"] >= 0.8 for r in rs),
                    "queries_recall_0": sum(r["recall"] == 0 for r in rs)}
        arms.append({"selector": name, "params": params, "clean": agg(clean), "all": agg(rows), "rows": rows})

    print(f"\n{'selector':18s} {'params':42s} {'recall':>7s} {'prec':>6s} {'rows':>6s} {'tokens':>7s} {'>=.8':>5s} {'=0':>3s}")
    for a in arms:
        c = a["clean"]
        print(f"{a['selector']:18s} {json.dumps(a['params']):42s} {c['recall']:7.3f} {c['precision']:6.3f} "
              f"{c['rows']:6.1f} {c['tokens']:7.0f} {c['queries_recall_ge_0_8']:5d} {c['queries_recall_0']:3d}")

    out = {
        "protocol": "first-mention-selector-preflight-v1",
        "engine_version": yantrikdb.__version__,
        "novelty_embedder": args.novelty_embedder,
        "documents_sha256": sha256_file(args.documents),
        "beam_source_sha256": sha256_file(args.beam_source),
        "gold_is_selector_input": False,
        "quarantine": sorted(QUARANTINE),
        "recall_top_k": args.recall_top_k,
        "default_n": args.default_n,
        "queries": qinfo,
        "novelty_diagnostic": diag_summary,
        "arms": arms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {args.out}")
    for db in stores.values():
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
