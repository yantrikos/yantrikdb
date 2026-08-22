"""Audit AMB event-ordering gold items against user-authored source turns.

This diagnostic uses a local Ollama embedding model to align each numbered
gold item with the closest retrieved source turn. It reports weak alignments
and sequences whose matched first-mention turns are not chronological.
"""

import argparse
import json
import math
import re
import urllib.request
from pathlib import Path


_TURN_RE = re.compile(r"\bTurn\s+(\d+)\b", re.IGNORECASE)
_ITEM_RE = re.compile(r"(?<!\d)(\d+)\)\s+")


def split_numbered_items(answer: str) -> list[str]:
    matches = list(_ITEM_RE.finditer(answer))
    return [
        answer[
            match.end():matches[index + 1].start()
            if index + 1 < len(matches) else len(answer)
        ]
        .strip(" ,.;")
        for index, match in enumerate(matches)
    ]


def embed_batch(host: str, model: str, texts: list[str]) -> list[list[float]]:
    body = json.dumps({"model": model, "input": texts}).encode()
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/embed",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)["embeddings"]


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm or 1.0)


def first_turn(text: str) -> int | None:
    match = _TURN_RE.search(text)
    return int(match.group(1)) if match else None


def source_documents(row: dict) -> list[str]:
    """Read legacy document strings or format product recall hits for audit."""
    documents = list(dict.fromkeys(row.get("documents") or []))
    if documents:
        return documents
    formatted = []
    for hit in row.get("hits") or []:
        metadata = hit.get("metadata") or {}
        turn = metadata.get("first_mention_turn")
        prefix = f"[Turn {turn}] User: " if turn is not None else ""
        formatted.append(prefix + str(hit.get("text") or ""))
    return list(dict.fromkeys(formatted))


def result_rows(payload: dict | list) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("results", payload)
        if isinstance(rows, list):
            return rows
    raise ValueError("audit artifact must be a result list or contain results")


def select_first_mention_match(
    matches: list[dict], score_delta: float
) -> dict | None:
    """Choose the earliest turn whose score is close to the best paraphrase."""
    if not matches:
        return None
    score_floor = matches[0]["score"] - score_delta
    plausible = [
        match for match in matches
        if match["score"] >= score_floor and match["turn"] is not None
    ]
    return min(plausible, key=lambda match: match["turn"], default=matches[0])


def select_monotonic_matches(alignments: list[dict]) -> list[dict] | None:
    """Find the highest-scoring nondecreasing mention path across gold items.

    Each item may have several semantically plausible occurrences. Dynamic
    programming keeps the best path ending at each turn, which distinguishes
    missing evidence from an earliest-occurrence choice that breaks sequence.
    """
    if not alignments:
        return []

    states: dict[int, tuple[float, list[dict]]] = {-1: (0.0, [])}
    for alignment in alignments:
        next_states: dict[int, tuple[float, list[dict]]] = {}
        for previous_turn, (total_score, path) in states.items():
            for match in alignment.get("matches") or []:
                turn = match.get("turn")
                if turn is None or turn < previous_turn:
                    continue
                candidate = (total_score + match["score"], [*path, match])
                prior = next_states.get(turn)
                if prior is None or candidate[0] > prior[0]:
                    next_states[turn] = candidate
        if not next_states:
            return None
        states = next_states

    _, best_path = min(
        states.values(),
        key=lambda state: (
            -state[0],
            tuple(match["turn"] for match in state[1]),
        ),
    )
    return best_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--first-mention-delta", type=float, default=0.05)
    args = parser.parse_args()

    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    rows = result_rows(payload)
    embedding_cache: dict[str, list[float]] = {}
    output = []

    for row in rows:
        gold = " ".join(row.get("gold_answers") or [])
        items = split_numbered_items(gold)
        documents = source_documents(row)
        texts = [*items, *documents]
        missing = [text for text in texts if text not in embedding_cache]
        if missing:
            vectors = embed_batch(args.host, args.model, missing)
            embedding_cache.update(zip(missing, vectors))

        alignments = []
        for item in items:
            ranked = sorted(
                (
                    {
                        "score": cosine(
                            embedding_cache[item], embedding_cache[document]
                        ),
                        "turn": first_turn(document),
                        "document": document,
                    }
                    for document in documents
                ),
                key=lambda candidate: candidate["score"],
                reverse=True,
            )[:args.top_k]
            alignments.append({
                "item": item,
                "best_match": ranked[0] if ranked else None,
                "first_mention_match": select_first_mention_match(
                    ranked, args.first_mention_delta
                ),
                "matches": ranked,
            })

        turns = [
            alignment["first_mention_match"]["turn"]
            for alignment in alignments
            if alignment["first_mention_match"] is not None
            and alignment["first_mention_match"]["turn"] is not None
        ]
        chronological = all(
            left <= right for left, right in zip(turns, turns[1:])
        )
        monotonic_matches = select_monotonic_matches(alignments)
        monotonic_turns = (
            [match["turn"] for match in monotonic_matches]
            if monotonic_matches is not None
            else []
        )
        independent_score = sum(
            alignment["best_match"]["score"]
            for alignment in alignments
            if alignment["best_match"] is not None
        )
        monotonic_score = (
            sum(match["score"] for match in monotonic_matches)
            if monotonic_matches is not None
            else None
        )
        result = {
            "query_id": row.get("query_id"),
            "query": row.get("query"),
            "gold_answer": gold,
            "matched_turns": turns,
            "chronological": chronological,
            "monotonic_recoverable": monotonic_matches is not None,
            "monotonic_turns": monotonic_turns,
            "monotonic_score": monotonic_score,
            "independent_best_score": independent_score,
            "monotonic_score_penalty": (
                independent_score - monotonic_score
                if monotonic_score is not None
                else None
            ),
            "alignments": alignments,
        }
        output.append(result)
        scores = [
            alignment["best_match"]["score"]
            for alignment in alignments if alignment["best_match"]
        ]
        print(
            f"{result['query_id']}: turns={turns} "
            f"chronological={chronological} "
            f"monotonic={monotonic_turns or 'unrecoverable'} "
            f"min_score={min(scores, default=0.0):.3f}"
        )

    summary = {
        "queries": len(output),
        "non_chronological": sum(
            not result["chronological"] for result in output
        ),
        "monotonic_recoverable_non_chronological": sum(
            not result["chronological"] and result["monotonic_recoverable"]
            for result in output
        ),
        "model": args.model,
        "first_mention_delta": args.first_mention_delta,
        "results": output,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"non_chronological={summary['non_chronological']}/"
        f"{summary['queries']} wrote={args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
