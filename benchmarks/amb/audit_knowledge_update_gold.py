#!/usr/bin/env python3
"""Audit zero-score BEAM knowledge-update labels against raw user history.

This is a benchmark-integrity diagnostic, not a retrieval experiment. It shows
where the gold and predicted values occur in source turns so stale or ambiguous
labels are not converted into production supersession policy.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path


VALUE_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s*-\s*\d[\d,]*(?:\.\d+)?\s*"
    r"(?:days?|weeks?|months?|years?|hours?)\b"
    r"|\$\s?\d[\d,]*(?:\.\d+)?"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:%|percent|women|books?|cupcakes?|days?|hours?|words?)?\b"
    r"|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}\b"
    r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
    re.IGNORECASE,
)
TURN_HEADER_RE = re.compile(
    r"(?m)^\[(?:(?:[A-Z][a-z]+-\d{1,2}-\d{4})\s*\|\s*)?"
    r"Turn (?P<turn>\d+)\](?: \(cont\.\))?\s+"
    r"(?P<role>User|Assistant):\s*"
)


def value_tokens(text: str) -> set[str]:
    values = {
        re.sub(r"\s+", " ", match.group(0).replace(",", "").strip()).casefold()
        for match in VALUE_RE.finditer(text or "")
    }
    return {
        value for value in values if not re.fullmatch(r"(?:19|20)\d{2}", value)
    }


def load_json_rows(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON list")
    return payload


def parse_document_turns(content: str) -> list[dict]:
    """Parse flattened BEAM document text into source-role turn fragments."""
    matches = list(TURN_HEADER_RE.finditer(content or ""))
    turns = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        turns.append(
            {
                "id": int(match.group("turn")),
                "role": match.group("role").casefold(),
                "content": content[match.end() : end].strip(),
            }
        )
    return turns


def load_conversations(path: Path) -> dict[str, dict]:
    """Load raw-generator conversations or the published flattened cache."""
    rows = load_json_rows(path)
    if all("conversation_id" in row for row in rows):
        return {str(row["conversation_id"]): row for row in rows}
    if not all("user_id" in row and "content" in row for row in rows):
        raise ValueError(f"{path}: unsupported BEAM source schema")

    fragments: dict[str, dict[tuple[int, str], list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for document in rows:
        user_id = str(document["user_id"])
        for turn in parse_document_turns(str(document.get("content") or "")):
            key = (turn["id"], turn["role"])
            fragments[user_id][key].append(turn["content"])

    conversations = {}
    for user_id, by_turn in fragments.items():
        chat = [
            {
                "id": turn_id,
                "role": role,
                "content": " ".join(part for part in parts if part).strip(),
            }
            for (turn_id, role), parts in sorted(by_turn.items())
        ]
        conversations[user_id] = {"conversation_id": user_id, "chat": chat}
    return conversations


def iter_turns(chat: list) -> list[dict]:
    turns = []
    for session in chat:
        if isinstance(session, list):
            turns.extend(turn for turn in session if isinstance(turn, dict))
        elif isinstance(session, dict) and "role" in session:
            turns.append(session)
    return turns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    result_payload = json.loads(args.results.read_text(encoding="utf-8"))
    results = result_payload["results"]
    conversations = load_conversations(args.dataset)
    failures = [
        row
        for row in results
        if (row.get("meta") or {}).get("question_category") == "knowledge_update"
        and float(row.get("score") or 0.0) == 0.0
    ]

    gold_precedes_prediction = 0
    gold_absent_from_user = 0
    for row in failures:
        conv_id = str((row.get("meta") or {})["conversation_id"])
        gold = " ".join(row.get("gold_answers") or [])
        prediction = row.get("answer") or ""
        gold_values = value_tokens(gold)
        predicted_values = value_tokens(prediction) - gold_values
        matches = []
        for turn in iter_turns(conversations[conv_id].get("chat") or []):
            if str(turn.get("role", "")).casefold() != "user":
                continue
            text = re.sub(r"\s*->->.*$", "", str(turn.get("content") or "")).strip()
            values = value_tokens(text)
            labels = []
            if values & gold_values:
                labels.append("GOLD")
            if values & predicted_values:
                labels.append("PRED")
            if labels:
                matches.append((int(turn.get("id") or -1), "+".join(labels), text))

        gold_turns = [turn for turn, label, _ in matches if "GOLD" in label]
        predicted_turns = [turn for turn, label, _ in matches if "PRED" in label]
        stale_label = bool(
            gold_turns and predicted_turns and max(predicted_turns) > max(gold_turns)
        )
        gold_precedes_prediction += int(stale_label)
        gold_absent = not gold_turns
        gold_absent_from_user += int(gold_absent)
        verdict = (
            "GOLD_VALUE_NOT_EXACT_IN_USER"
            if gold_absent
            else "GOLD_PRECEDES_PREDICTION"
            if stale_label
            else "REVIEW"
        )
        print(f"\n{row['query_id']} [{verdict}]")
        print(f"Q: {row['query']}")
        print(f"G: {gold}")
        print(f"A: {prediction}")
        for turn, label, text in matches:
            print(f"  Turn {turn:>3} {label:<9} {text}")

    print(
        f"\nGold value precedes a distinct predicted value in "
        f"{gold_precedes_prediction}/{len(failures)} zero-score cases."
    )
    print(
        f"Gold value has no exact user-turn match in "
        f"{gold_absent_from_user}/{len(failures)} zero-score cases."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
