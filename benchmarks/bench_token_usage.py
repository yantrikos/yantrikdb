"""How many context tokens does one recall cost an agent?

Measures the three payload shapes a recall result travels in:

  http     GET /v1 recall — every pyo3 field intact
  mcp      the MCP tool projection (rid/text/type/score/scores/why + envelope)
  compact  the lean MCP projection (rid/text/score + warning tags only)

Token counts use the repo's chars/4 convention (packs/evaluate_tiers.py) —
crude, but consistent, and the RATIO between shapes is what matters.

Run:  python benchmarks/bench_token_usage.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yantrikdb import YantrikDB  # noqa: E402

TOKENS = lambda s: len(s) // 4  # noqa: E731  — repo convention

SEED_FACTS = [
    ("User prefers dark mode in every editor and terminal", "preference"),
    ("The staging database migrated to Postgres 17 on May 3rd", "work"),
    ("Alice leads the DevOps rotation until the end of Q3", "people"),
    ("The API rate limit is 600 requests per minute per token", "work"),
    ("Deploys happen through the tar-pipe runbook, never rsync", "work"),
    ("The design system accent color is amber #f6a623", "work"),
    ("Weekly review call moved permanently to Thursday 4pm", "work"),
    ("The old ingest pipeline was retired in favor of YRP streaming", "work"),
    ("Grafana admin credentials live in the team vault, folder infra", "work"),
    ("Customer-facing errors must never include stack traces", "work"),
] * 5  # 50 records — enough for a full top-10 with graph expansion


def mcp_projection(response: dict) -> str:
    items = [{
        "rid": r["rid"], "text": r["text"], "type": r["type"],
        "score": round(r["score"], 4), "importance": r["importance"],
        "created_at": r["created_at"],
        "scores": {k: round(r["scores"][k], 4) for k in
                   ("similarity", "decay", "recency", "importance",
                    "graph_proximity")},
        "why_retrieved": r["why_retrieved"],
    } for r in response["results"]]
    return json.dumps({
        "count": len(items), "results": items,
        "confidence": round(response["confidence"], 4),
        "retrieval_summary": {
            "top_similarity": round(response["retrieval_summary"]["top_similarity"], 4),
            "score_spread": round(response["retrieval_summary"]["score_spread"], 4),
            "sources_used": response["retrieval_summary"]["sources_used"],
            "candidate_count": response["retrieval_summary"]["candidate_count"],
        },
        "hints": [{"hint_type": h["hint_type"], "suggestion": h["suggestion"],
                   "related_entities": h["related_entities"]}
                  for h in response["hints"]],
    })


def compact_projection(response: dict) -> str:
    items = [{
        "rid": r["rid"], "text": r["text"], "score": round(r["score"], 3),
        **({"why": [w for w in r["why_retrieved"] if w.startswith("⚠")]}
           if any(w.startswith("⚠") for w in r["why_retrieved"]) else {}),
    } for r in response["results"]]
    return json.dumps({"count": len(items), "results": items,
                       "confidence": round(response["confidence"], 3)},
                      ensure_ascii=False)


def main() -> None:
    db = YantrikDB(":memory:", 256)
    db.set_embedder_named("potion-base-8M")
    for i, (text, domain) in enumerate(SEED_FACTS):
        db.record_text(f"{text} (note {i})", memory_type="semantic",
                       importance=0.5 + (i % 5) * 0.1, domain=domain)

    query = "what changed about our database and deploy setup?"
    response = db.recall_with_response(query=query, top_k=10)
    full = db.recall_text(query, top_k=10)

    http_payload = json.dumps({"count": len(full), "results": [
        {k: v for k, v in r.items() if k != "embedding"} for r in full]})
    mcp_payload = mcp_projection(response)
    compact_payload = compact_projection(response)

    rows = [("http (every field)", http_payload),
            ("mcp (current default)", mcp_payload),
            ("mcp compact=True", compact_payload)]
    text_tokens = TOKENS(json.dumps(
        [r["text"] for r in response["results"]]))

    print(f"top_k=10 over {len(SEED_FACTS)} records — chars/4 estimate\n")
    print(f"{'shape':<24} {'tokens':>8} {'vs http':>9} {'overhead vs bare text':>22}")
    for name, payload in rows:
        t = TOKENS(payload)
        print(f"{name:<24} {t:>8} {t / TOKENS(http_payload):>8.0%} "
              f"{t / max(text_tokens, 1):>21.1f}x")
    print(f"{'bare answer text':<24} {text_tokens:>8}")


if __name__ == "__main__":
    main()
