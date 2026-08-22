"""Pointwise local-LLM scoring for frozen AMB synthesis candidates.

Unlike the production second pass, this probe does not ask the model to choose
exactly N items. It scores every candidate independently and persists the raw
features so a separate grouped calibration can decide whether any conservative
swap policy generalizes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import urllib.request
from pathlib import Path

try:
    from .calibrate_rollup_membership import (
        SCORER_PROTOCOL_VERSION,
        candidate_payload_sha256,
        load_jsonl,
        ollama_model_metadata,
    )
except ImportError:  # Direct script execution.
    from calibrate_rollup_membership import (
        SCORER_PROTOCOL_VERSION,
        candidate_payload_sha256,
        load_jsonl,
        ollama_model_metadata,
    )


def _schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "resolved_subject": {"type": "string"},
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "relevance": {"type": "number"},
                        "atomicity": {"type": "number"},
                    },
                    "required": ["id", "relevance", "atomicity"],
                },
            }
        },
        "required": ["resolved_subject", "scores"],
    }


def _prompt(row: dict) -> str:
    candidates = "\n".join(
        f"{candidate['id']} | {candidate['item']}"
        for candidate in row.get("candidate_items") or []
    )
    return (
        "First resolve the concrete subject of the user query from repeated "
        "specific terms, entities, tools, or goals in the candidate set. State "
        "that inferred referent in resolved_subject; for example, 'the framework' "
        "may resolve to a named framework repeatedly discussed by candidates. "
        "Do not score any candidate until this resolution is complete. After "
        "resolution, mentally substitute the concrete subject into the query. "
        "A candidate about that concrete subject can be directly relevant even "
        "when it does not repeat the query's generic phrase.\n\n"
        "Then score every candidate independently for membership in the answer to "
        "the user query. Do not select a final subset and do not use candidate "
        "order, date, or turn as evidence of relevance.\n\n"
        "RELEVANCE, 0-100: how directly this candidate is one of the distinct "
        "answer aspects requested by the query. Specific named concepts, "
        "actions, decisions, examples, feedback, and resulting changes beat "
        "generic planning, status, biography, or adjacent activity. Resolve a "
        "generic referent such as 'the framework' or 'my project' from the "
        "repeated specific subject in the candidate set, then judge each item "
        "against that resolved subject. A broad timeline query may legitimately "
        "span several phases, but an item must still advance its named thread.\n\n"
        "ATOMICITY, 0-100: how well the candidate expresses one independently "
        "scorable answer item at the abstraction level implied by the requested "
        f"count ({row.get('requested_item_count')}). Penalize umbrella summaries, "
        "compound lists of unrelated events, and near-duplicate restatements.\n\n"
        "Use the full 0-100 range. Give every supplied ID exactly one score.\n\n"
        f"USER QUERY:\n{row['query']}\n\nCANDIDATES:\n{candidates}\n\n"
        "Return JSON only with this shape: "
        '{"resolved_subject":"specific subject",'
        '"scores":[{"id":"I001","relevance":0,"atomicity":0}]}'
    )


def extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        try:
            value = json.loads(fenced.group(1))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize_scores(raw: dict, expected_ids: list[str]) -> list[dict]:
    by_id = {}
    for score in raw.get("scores") or []:
        candidate_id = str(score.get("id") or "").strip()
        if candidate_id not in expected_ids or candidate_id in by_id:
            continue
        relevance = float(score.get("relevance"))
        atomicity = float(score.get("atomicity"))
        if not math.isfinite(relevance) or not math.isfinite(atomicity):
            continue
        by_id[candidate_id] = {
            "id": candidate_id,
            "relevance": min(max(relevance, 0.0), 100.0),
            "atomicity": min(max(atomicity, 0.0), 100.0),
        }
    missing = [candidate_id for candidate_id in expected_ids if candidate_id not in by_id]
    if missing:
        raise ValueError(f"scorer omitted or malformed candidate IDs: {missing}")
    return [by_id[candidate_id] for candidate_id in expected_ids]


def score_row(
    row: dict,
    *,
    host: str,
    model: str,
    timeout: int,
    num_ctx: int,
    model_metadata: dict,
) -> dict:
    prompt = _prompt(row)
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "think": False,
            "format": _schema(),
            "options": {
                "temperature": 0.0,
                "seed": 0,
                "num_ctx": num_ctx,
                "num_predict": 2048,
            },
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    raw_content = payload["message"]["content"]
    raw = extract_json(raw_content)
    if raw is None:
        raise ValueError(
            f"scorer returned no JSON object: {raw_content[:200]!r}"
        )
    expected_ids = [str(candidate["id"]) for candidate in row.get("candidate_items") or []]
    scores = normalize_scores(raw, expected_ids)
    return {
        "query": row["query"],
        "requested_item_count": row.get("requested_item_count"),
        "candidate_count": len(expected_ids),
        "model": model,
        "model_metadata": model_metadata,
        "scorer_protocol_version": SCORER_PROTOCOL_VERSION,
        "candidate_payload_sha256": candidate_payload_sha256(row),
        "num_ctx": num_ctx,
        "temperature": 0.0,
        "seed": 0,
        "think": False,
        "resolved_subject": str(raw.get("resolved_subject") or "").strip(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "scores": scores,
    }


def validate_cached_rows(
    cached_rows: list[dict],
    rows_by_query: dict[str, dict],
    *,
    model: str,
    num_ctx: int,
    candidate_sha256: str,
    model_metadata: dict,
) -> dict[str, dict]:
    completed = {}
    for cached in cached_rows:
        query = str(cached.get("query") or "")
        if not query or query in completed:
            raise ValueError(f"duplicate or missing cached query {query!r}")
        row = rows_by_query.get(query)
        if row is None:
            raise ValueError(f"cached query is absent from candidate artifact: {query!r}")
        expected = {
            "model": model,
            "model_metadata": model_metadata,
            "scorer_protocol_version": SCORER_PROTOCOL_VERSION,
            "candidate_payload_sha256": candidate_payload_sha256(row),
            "num_ctx": num_ctx,
            "temperature": 0.0,
            "seed": 0,
            "think": False,
            "candidate_artifact_sha256": candidate_sha256,
            "prompt_sha256": hashlib.sha256(_prompt(row).encode()).hexdigest(),
        }
        mismatched = {
            key: (cached.get(key), value)
            for key, value in expected.items()
            if cached.get(key) != value
        }
        if mismatched:
            raise ValueError(
                f"cached scorer metadata does not match {query!r}: {mismatched}"
            )
        expected_ids = [
            str(candidate["id"])
            for candidate in row.get("candidate_items") or []
        ]
        normalize_scores(cached, expected_ids)
        completed[query] = cached
    return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument(
        "--indices",
        help="Optional comma-separated zero-based row indices for a bounded probe.",
    )
    args = parser.parse_args()

    all_rows = load_jsonl(args.candidates)
    candidate_sha256 = hashlib.sha256(args.candidates.read_bytes()).hexdigest()
    model_metadata = ollama_model_metadata(args.host, args.model)
    rows_by_query = {row["query"]: row for row in all_rows}
    if len(rows_by_query) != len(all_rows):
        raise ValueError("candidate artifact contains duplicate queries")
    rows = all_rows
    if args.indices:
        indices = {int(value) for value in args.indices.split(",") if value.strip()}
        rows = [row for index, row in enumerate(rows) if index in indices]
    completed = {}
    if args.out.exists():
        completed = validate_cached_rows(
            load_jsonl(args.out),
            rows_by_query,
            model=args.model,
            num_ctx=args.num_ctx,
            candidate_sha256=candidate_sha256,
            model_metadata=model_metadata,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, 1):
        if row["query"] in completed:
            print(f"{index}/{len(rows)} cached")
            continue
        result = score_row(
            row,
            host=args.host,
            model=args.model,
            timeout=args.timeout,
            num_ctx=args.num_ctx,
            model_metadata=model_metadata,
        )
        result["candidate_artifact_sha256"] = candidate_sha256
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
        print(f"{index}/{len(rows)} scored candidates={result['candidate_count']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
