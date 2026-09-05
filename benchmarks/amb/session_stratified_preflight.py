#!/usr/bin/env python3
"""Judge-free preflight 2: are BEAM event-ordering gold turns SESSION-
structured, and can a session-stratified, query-only selector find them?

Follow-up to first_mention_preflight.py, whose novelty/cluster selectors
all measured at ~chance (gold turns are not lexically or semantically
novel relative to earlier turns). Inspection of gold ids (e.g. 4/60/116
against session starts 0/60/116) suggested the generator picks roughly
one source turn per session, close to the session opening.

Part A (structural, no query): for every gold turn, its session index and
its user-turn offset from the session start; how many sessions each
conversation has versus the requested item count.

Part B (selectors, query-only): per session, rank user turns by engine
relevance to the query and keep the top ``r`` among the first ``m`` user
turns of that session. ``m=None`` means the whole session. A pure
structural control (``r=1, m=1``: the first user turn of every session,
no query at all) is included so a benchmark-structure artifact is
visible rather than mistaken for retrieval quality.

Gold is joined only at scoring time. All combinations are reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median

import yantrikdb

try:
    from .first_mention_preflight import (
        QUARANTINE, load_docs, load_gold, requested_count, split_turns, sha256_file,
    )
except ImportError:  # pragma: no cover - direct script execution
    from first_mention_preflight import (
        QUARANTINE, load_docs, load_gold, requested_count, split_turns, sha256_file,
    )


def sessions_of(docs: list[dict]) -> list[list[dict]]:
    """Session = maximal run of turns sharing one header date, in turn order."""
    turns = []
    seen = set()
    for d in docs:
        for t in split_turns(d["content"]):
            if t["turn"] is None or t["turn"] in seen:
                continue
            seen.add(t["turn"])
            t["doc_id"] = d["id"]
            turns.append(t)
    turns.sort(key=lambda t: t["turn"])
    sessions, cur, cur_date = [], [], None
    for t in turns:
        if t["created_at"] != cur_date and cur:
            sessions.append(cur)
            cur = []
        cur_date = t["created_at"]
        cur.append(t)
    if cur:
        sessions.append(cur)
    return sessions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--documents", type=Path, required=True)
    ap.add_argument("--beam-source", type=Path, required=True)
    ap.add_argument("--store-dir", type=Path, required=True,
                    help="stores written by first_mention_preflight.py (reopened, not rebuilt)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--recall-top-k", type=int, default=400)
    args = ap.parse_args()

    by_user = load_docs(args.documents)
    gold = load_gold(args.beam_source)

    # ---- Part A: structure
    struct_rows = []
    offsets, sess_idx_hits = [], []
    for qid, g in gold.items():
        sessions = sessions_of(by_user[g["user_id"]])
        user_sessions = [[t for t in s if t["role"] == "user"] for s in sessions]
        turn_pos = {}
        for si, s in enumerate(user_sessions):
            for oi, t in enumerate(s):
                turn_pos[t["turn"]] = (si, oi)
        gold_pos = [turn_pos.get(t) for t in g["source_turns"]]
        per_session = defaultdict(int)
        for p in gold_pos:
            if p:
                per_session[p[0]] += 1
                offsets.append(p[1])
        struct_rows.append({
            "query_id": qid, "sessions": len(sessions),
            "requested_n": requested_count(g["query"]), "gold_n": len(g["source_turns"]),
            "gold_positions": gold_pos,
            "sessions_with_gold": len(per_session),
            "max_gold_per_session": max(per_session.values(), default=0),
            "user_turns_per_session": [len(s) for s in user_sessions],
        })
    offs = sorted(offsets)
    struct_summary = {
        "gold_refs": len(offs),
        "offset_median": median(offs), "offset_mean": fmean(offs),
        "offset_le_0": sum(o <= 0 for o in offs) / len(offs),
        "offset_le_1": sum(o <= 1 for o in offs) / len(offs),
        "offset_le_2": sum(o <= 2 for o in offs) / len(offs),
        "offset_le_4": sum(o <= 4 for o in offs) / len(offs),
        "queries_one_gold_per_session": sum(r["max_gold_per_session"] == 1 for r in struct_rows),
        "queries_sessions_eq_requested": sum(r["sessions"] == r["requested_n"] for r in struct_rows),
        "mean_sessions": fmean(r["sessions"] for r in struct_rows),
        "mean_user_turns_per_session": fmean(
            fmean(r["user_turns_per_session"]) for r in struct_rows),
    }
    print("STRUCTURE:", json.dumps(struct_summary, indent=1))

    # ---- Part B: session-stratified selectors over reopened stores
    stores = {uid: yantrikdb.YantrikDB.with_default(str(args.store_dir / f"{uid}.db"))
              for uid in by_user}
    arms = []
    grid = [(1, 1), (1, 2), (1, 3), (1, 5), (1, None), (2, 3), (2, 5), (2, None)]
    for r, m in grid:
        rows = []
        for qid, g in gold.items():
            db = stores[g["user_id"]]
            hits = db.recall(query=g["query"], top_k=args.recall_top_k, source="user",
                             skip_reinforce=True)
            score = {}
            for h in hits:
                tid = (h.get("metadata") or {}).get("turn_id")
                if tid is not None:
                    score[int(tid)] = max(score.get(int(tid), 0.0), float(h.get("score") or 0.0))
            sessions = sessions_of(by_user[g["user_id"]])
            picked = []
            for s in sessions:
                users = [t for t in s if t["role"] == "user"]
                window = users if m is None else users[:m]
                ranked = sorted(window, key=lambda t: (-score.get(t["turn"], 0.0), t["turn"]))
                picked.extend(t["turn"] for t in ranked[:r])
            gset = set(g["source_turns"])
            hit = len(set(picked) & gset)
            rows.append({"query_id": qid, "quarantined": qid in QUARANTINE, "rows": len(picked),
                         "hit": hit, "recall": hit / len(gset), "precision": hit / len(picked) if picked else 0.0,
                         "selected_turns": sorted(picked)})
        clean = [x for x in rows if not x["quarantined"]]
        agg = {"n": len(clean), "recall": fmean(x["recall"] for x in clean),
               "precision": fmean(x["precision"] for x in clean),
               "rows": fmean(x["rows"] for x in clean),
               "queries_recall_ge_0_8": sum(x["recall"] >= 0.8 for x in clean),
               "queries_recall_0": sum(x["recall"] == 0 for x in clean)}
        arms.append({"selector": "session_stratified", "params": {"r": r, "m": m},
                     "query_used": not (r == 1 and m == 1), "clean": agg, "rows": rows})
    print(f"\n{'r':>2s} {'m':>5s} {'query?':>6s} {'recall':>7s} {'prec':>6s} {'rows':>5s} {'>=.8':>5s} {'=0':>3s}")
    for a in arms:
        c = a["clean"]
        print(f"{a['params']['r']:2d} {str(a['params']['m']):>5s} {str(a['query_used']):>6s} "
              f"{c['recall']:7.3f} {c['precision']:6.3f} {c['rows']:5.1f} {c['queries_recall_ge_0_8']:5d} {c['queries_recall_0']:3d}")

    json.dump({"protocol": "session-stratified-preflight-v1",
               "engine_version": yantrikdb.__version__,
               "documents_sha256": sha256_file(args.documents),
               "beam_source_sha256": sha256_file(args.beam_source),
               "gold_is_selector_input": False, "quarantine": sorted(QUARANTINE),
               "structure_summary": struct_summary, "structure_rows": struct_rows,
               "arms": arms},
              open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {args.out}")
    for db in stores.values():
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
