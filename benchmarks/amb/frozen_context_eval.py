"""Frozen-context evaluation: score a memory system's CONTEXT, not its answerer.

Both adversarial reviews (codex gpt-5.6-sol, qwen3.8-max, 2026-08-11)
rejected the 0.611-vs-0.862 comparison as confounded: Hindsight differs from
YantrikDB in memory architecture, in an ingest-time LLM extraction pass, AND
in answerer (gemini-3.1-pro-preview vs deepseek-v4-flash). Three variables
moved at once, so the gap cannot be attributed.

This removes two of them. Hindsight's published result files ship the exact
`context` string they injected per query, and all 400 BEAM-100K query_ids
match ours. So their retrieved memory can be replayed through OUR answerer
and OUR judge. Whatever differs then is the memory system alone.

Conditions (answerer and judge held fixed at deepseek-v4-flash):
    A  yantrikdb k=40 context   -> reproduces the 0.611 baseline
    E  hindsight context        -> their memory, our answerer

Reading it:
    E ~= A   memory systems are equivalent; the published gap is the answerer
    E >> A   their memory is genuinely better, independent of model
    E << A   their number depends on gemini-3.1-pro to exploit their context

Scoring reuses the dataset's own rubric judge (`score_result`) and its
per-category prompts, so numbers stay comparable with every run today.
"""
import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from memory_bench.dataset import get_dataset
from memory_bench.llm.ollama import OllamaLLM
from memory_bench.models import QueryResult
from memory_bench.utils import count_tokens
from memory_bench.modes.rag import RAGMode


def load_contexts(path: str) -> dict[str, str]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    return {r["query_id"]: r.get("context") or "" for r in d["results"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True, help="result JSON whose contexts to replay")
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="deepseek-v4-flash:0731-cloud")
    ap.add_argument("--split", default="100k")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ds = get_dataset("beam")
    queries = {q.id: q for q in ds.load_queries(a.split)}
    ctxs = load_contexts(a.contexts)
    ids = [qid for qid in ctxs if qid in queries and ctxs[qid].strip()]
    if a.limit:
        ids = ids[: a.limit]
    print(f"[{a.label}] {len(ids)} queries, answerer+judge = {a.model}", file=sys.stderr, flush=True)

    llm = OllamaLLM(a.model)
    mode = RAGMode(llm=llm)
    judge_llm = OllamaLLM(a.model)

    def one(qid: str):
        q = queries[qid]
        meta = dict(q.meta)
        meta["_prompt_fn"] = lambda query, context, meta=meta: ds.build_rag_prompt(
            query, context, ds.task_type, a.split, meta=meta)
        try:
            ans = mode.answer_from_context(q.query, ctxs[qid], ds.task_type, meta=meta)
        except Exception as e:
            print(f"  {qid}: answer failed {str(e)[:80]}", file=sys.stderr, flush=True)
            return None
        r = QueryResult(
            query_id=qid, query=q.query, answer=ans.answer, reasoning=ans.reasoning,
            # tiktoken, NOT chars//4. The proxy is not merely imprecise, it is
            # BIASED BY TEXT STYLE: conversation prose runs ~4.9 chars/token
            # while Hindsight's extracted statements with markup run ~4.1, so
            # chars//4 understated our context by 18% and made a 28% size
            # difference read as 7%. The harness uses count_tokens; anything
            # comparing against harness numbers must too.
            context=ctxs[qid], context_tokens=count_tokens(ctxs[qid]), retrieve_time_ms=0.0,
            gold_answers=q.gold_answers, correct=False, judge_reason="", meta=meta,
        )
        try:
            r.score = ds.score_result(r, judge_llm)
        except Exception as e:
            print(f"  {qid}: judge failed {str(e)[:80]}", file=sys.stderr, flush=True)
            return None
        r.correct = (r.score or 0) >= 0.5
        return r

    # Checkpoint as results arrive. Writing only at the end made a 40-60
    # minute run unobservable: no progress line and no partial file, so
    # "working" and "wedged" looked identical from outside — distinguishing
    # them meant inspecting the process's TCP connections and CPU delta. A
    # crash at query 390 would also have cost the entire run.
    t0 = time.perf_counter()
    out = []
    ckpt = (a.out or f"outputs/frozen-{a.label}.json") + ".partial"
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, ids), 1):
            if r is not None:
                out.append(r)
            if i % 20 == 0 or i == len(ids):
                done = len(out)
                rub = sum(x.score or 0 for x in out) / max(done, 1)
                rate = i / max(time.perf_counter() - t0, 1e-9) * 60
                eta = (len(ids) - i) / max(rate, 1e-9)
                print(f"  [{a.label}] {i}/{len(ids)}  rubric so far {rub:.3f}  "
                      f"{rate:.1f}/min  eta {eta:.0f}min", file=sys.stderr, flush=True)
                with open(ckpt, "w", encoding="utf-8") as f:
                    json.dump({"label": a.label, "done": i, "scored": done,
                               "rubric_so_far": rub}, f)
    dt = time.perf_counter() - t0

    n = len(out)
    binary = sum(1 for r in out if r.correct)
    rubric = sum(r.score or 0 for r in out) / max(n, 1)
    ctx_tok = sum(r.context_tokens for r in out) / max(n, 1)
    print(f"\n[{a.label}] {binary}/{n} = {binary/max(n,1):.1%} binary | {rubric:.3f} rubric "
          f"| {ctx_tok:,.0f} ctx tokens | {dt/60:.1f} min")

    from collections import defaultdict
    per = defaultdict(lambda: [0, 0, 0.0])
    for r in out:
        c = r.meta.get("question_category", "?")
        per[c][0] += 1
        per[c][1] += r.correct
        per[c][2] += r.score or 0
    for c in sorted(per):
        k, ok, s = per[c]
        print(f"  {c:26} {ok:3d}/{k:<3d} {s/k:.3f}")

    path = a.out or f"outputs/frozen-{a.label}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"label": a.label, "model": a.model, "n": n, "binary": binary,
                   "rubric": rubric, "results": [r.__dict__ for r in out]},
                  f, default=str)
    print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
