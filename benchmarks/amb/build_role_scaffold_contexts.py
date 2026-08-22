"""Build user-authoritative contexts with bounded adjacent assistant scaffold."""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


_HERE = Path(__file__).resolve().parent


_HEADER = r"\[(?:(?:[A-Z][a-z]+-\d+-\d+)(?: \| Turn \d+)?|Turn \d+)\]"
_TURN_SPLIT_RE = re.compile(rf"(?=\n*{_HEADER})")
_TURN_RE = re.compile(r"\bTurn\s+(\d+)\b", re.IGNORECASE)
_ROLE_RE = re.compile(
    rf"^(?P<header>{_HEADER})\s+(?P<role>User|Assistant):\s*",
    re.IGNORECASE,
)
_DISPLAY_RE = re.compile(
    r"^\[Speaker:\s+User\s+\|\s+"
    r"(?P<date>[A-Z][a-z]+\s+\d{2},\s+\d{4})"
    r"(?:\s+\|\s+Turn\s+(?P<turn>\d+))?\]"
)


def _selected_turn(document: str) -> int | None:
    match = _DISPLAY_RE.match(document)
    if match is not None and match.group("turn") is not None:
        return int(match.group("turn"))
    match = _TURN_RE.search(document)
    return int(match.group(1)) if match is not None else None


def _chronological_key(document: str, index: int) -> tuple:
    match = _DISPLAY_RE.match(document)
    if match is None:
        return (float("inf"), float("inf"), index)
    occurred_at = datetime.strptime(
        match.group("date"), "%B %d, %Y"
    ).replace(tzinfo=timezone.utc).timestamp()
    turn = match.group("turn")
    return (
        occurred_at,
        float(turn) if turn is not None else float("inf"),
        index,
    )


def _assistant_turns(documents) -> dict[tuple[str, int], str]:
    assistants = {}
    for document in documents:
        user_id = str(document.user_id)
        for part in _TURN_SPLIT_RE.split(document.content):
            turn = part.strip()
            if not turn:
                continue
            role = _ROLE_RE.match(turn)
            turn_match = _TURN_RE.search(turn)
            if (
                role is None
                or role.group("role").casefold() != "assistant"
                or turn_match is None
            ):
                continue
            assistants[(user_id, int(turn_match.group(1)))] = turn
    return assistants


def build_scaffold_documents(
    user_documents: list[str],
    assistant_by_turn: dict[int, str],
    *,
    token_budget: int,
    assistant_tokens: int,
    max_user_documents: int,
    max_assistant_scaffolds: int,
    token_counter: Callable[[str], int] | None = None,
    truncate_assistant: Callable[[str, int], str] | None = None,
) -> tuple[list[str], dict]:
    """Select relevance-ranked bundles, then order the fixed set by occurrence."""
    if (
        token_budget < 1
        or assistant_tokens < 1
        or max_user_documents < 1
        or max_assistant_scaffolds < 0
    ):
        raise ValueError("token and document limits are invalid")
    if token_counter is None or truncate_assistant is None:
        from memory_bench.utils import chunk_text, count_tokens

        token_counter = token_counter or count_tokens
        truncate_assistant = truncate_assistant or (
            lambda text, limit: chunk_text(text, limit)[0]
        )

    bundles = []
    used_tokens = 0
    paired = 0
    for relevance_index, user_document in enumerate(
        user_documents[:max_user_documents]
    ):
        turn = _selected_turn(user_document)
        assistant = (
            assistant_by_turn.get(turn + 1)
            if turn is not None and relevance_index < max_assistant_scaffolds
            else None
        )
        bundle = user_document
        if assistant:
            assistant = truncate_assistant(assistant, assistant_tokens)
            bundle += (
                "\n[Adjacent assistant scaffold; not a user milestone] "
                + assistant
            )
        bundle_tokens = token_counter(bundle)
        if bundles and used_tokens + bundle_tokens > token_budget:
            continue
        if not bundles and bundle_tokens > token_budget:
            bundle = truncate_assistant(user_document, token_budget)
            bundle_tokens = token_counter(bundle)
            assistant = None
        bundles.append((bundle, relevance_index))
        used_tokens += bundle_tokens
        paired += int(bool(assistant))

    ordered = [
        document
        for document, _ in sorted(
            bundles,
            key=lambda item: _chronological_key(item[0], item[1]),
        )
    ]
    return ordered, {
        "selection_mode": "user_evidence_with_adjacent_assistant_scaffold",
        "selected_user_documents": len(ordered),
        "paired_assistant_scaffolds": paired,
        "context_tokens": used_tokens,
        "token_budget": token_budget,
        "assistant_token_cap": assistant_tokens,
        "assistant_scaffold_cap": max_assistant_scaffolds,
        "presentation_order": "chronological_after_selection",
    }


def main() -> int:
    sys.path = [
        entry for entry in sys.path if Path(entry or ".").resolve() != _HERE
    ]
    from memory_bench.dataset import get_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--user-only", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="100k")
    parser.add_argument("--token-budget", type=int, default=20_480)
    parser.add_argument("--assistant-tokens", type=int, default=256)
    parser.add_argument("--max-user-documents", type=int, default=40)
    parser.add_argument("--max-assistant-scaffolds", type=int, default=20)
    args = parser.parse_args()

    payload = json.loads(args.user_only.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload["results"]
    units = {str(row["query_id"]).split("_", 1)[0] for row in rows}
    source_documents = get_dataset("beam").load_documents(
        args.split, user_ids=units
    )
    assistants = _assistant_turns(source_documents)

    output = []
    for row in rows:
        user_id = str(row["query_id"]).split("_", 1)[0]
        assistant_by_turn = {
            turn: text
            for (candidate_user, turn), text in assistants.items()
            if candidate_user == user_id
        }
        documents, trace = build_scaffold_documents(
            list(row.get("documents") or []),
            assistant_by_turn,
            token_budget=args.token_budget,
            assistant_tokens=args.assistant_tokens,
            max_user_documents=args.max_user_documents,
            max_assistant_scaffolds=args.max_assistant_scaffolds,
        )
        transformed = dict(row)
        transformed["documents"] = documents
        transformed["context"] = "\n\n".join(
            f"## Memory {index}\n{document}"
            for index, document in enumerate(documents, 1)
        )
        transformed["selection"] = trace
        output.append(transformed)
        print(
            f"{row['query_id']}: users={trace['selected_user_documents']} "
            f"scaffolds={trace['paired_assistant_scaffolds']} "
            f"tokens={trace['context_tokens']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"results": output}, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
