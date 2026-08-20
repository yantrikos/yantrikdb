"""Select an answer-sized concern context, then restore event chronology."""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path


_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def requested_count(query: str) -> int | None:
    match = re.search(
        r"\b(?:only(?:\s+and\s+only)?|exactly)\s+"
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        query,
        re.IGNORECASE,
    )
    if not match:
        return None
    value = match.group(1).lower()
    return int(value) if value.isdigit() else _COUNT_WORDS[value]


def chronological_key(hit: dict) -> tuple:
    metadata = hit.get("metadata") or {}
    turn = metadata.get("first_mention_turn")
    created_at = hit.get("created_at")
    return (
        float("inf") if created_at is None else float(created_at),
        float("inf") if turn is None else float(turn),
        str(hit.get("rid") or ""),
    )


def normalize_selected_ids(raw: dict, known_ids: set[str], count: int) -> list[str]:
    values = raw.get("selected_ids") or []
    if isinstance(values, str):
        values = [values]
    selected = []
    for value in values:
        candidate_id = str(value).strip().upper()
        if candidate_id in known_ids and candidate_id not in selected:
            selected.append(candidate_id)
        if len(selected) >= count:
            break
    return selected


def _load_ollama(path: Path):
    name = "memory_bench.llm._workspace_ollama_concern_selector"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Ollama provider from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.OllamaLLM, module.Schema


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--query-id", action="append")
    args = parser.parse_args()

    rows = json.loads(args.replay.read_text(encoding="utf-8"))
    selected_queries = set(args.query_id or ())
    if selected_queries:
        rows = [row for row in rows if row["query_id"] in selected_queries]

    OllamaLLM, Schema = _load_ollama(Path(__file__).with_name("ollama.py"))
    llm = OllamaLLM(args.model, think=False, num_predict=1200, num_ctx=65536)
    output = []
    for row in rows:
        count = requested_count(row["query"])
        if count is None:
            raise ValueError(f"cannot determine answer count for {row['query_id']}")
        candidates = {f"C{index:03d}": hit for index, hit in enumerate(row["hits"], 1)}
        lines = []
        for candidate_id, hit in candidates.items():
            metadata = hit.get("metadata") or {}
            lines.append(
                f"{candidate_id} | turn={metadata.get('first_mention_turn')} | "
                f"{hit.get('text') or ''}"
            )
        prompt = (
            "Select source-grounded memory concerns for an answer. Return IDs only; "
            "do not answer or rewrite them. Infer the narrow semantic relation in the "
            "user's wording and reject events that merely share its broad topic. For "
            "example, a hardship involving family is not a way family provided support, "
            "and a schedule or writing tool is not an aspect of refining prose. Prefer "
            "concrete named advice, feedback, decisions, requests, or resulting changes "
            "that form a coherent thread across conversations. Repeated discussion of "
            "one concern counts once. Select by meaning first; dates and turns must not "
            "influence membership because code will order the fixed selection later. "
            f"Select exactly {count} distinct IDs.\n\n"
            f"USER QUERY:\n{row['query']}\n\nCANDIDATES:\n"
            + "\n".join(lines)
            + '\n\nReturn JSON only: {"selected_ids":["C001"]}'
        )
        raw = llm.generate(
            prompt,
            Schema(
                required=["selected_ids"],
                properties={"selected_ids": {"type": "array"}},
            ),
        )
        selected_ids = normalize_selected_ids(raw, set(candidates), count)
        if len(selected_ids) != count:
            raise ValueError(
                f"{row['query_id']}: selector returned {len(selected_ids)} valid IDs, "
                f"expected {count}: {raw!r}"
            )
        hits = sorted((candidates[value] for value in selected_ids), key=chronological_key)
        transformed = dict(row)
        transformed["hits"] = hits
        transformed["documents"] = [
            f"[Turn {(hit.get('metadata') or {}).get('first_mention_turn')}] "
            f"User: {hit.get('text') or ''}"
            for hit in hits
        ]
        transformed["concern_selection"] = {
            "model": args.model,
            "selected_ids": selected_ids,
            "presentation_order": "first_mention",
        }
        output.append(transformed)
        print(
            f"{row['query_id']}: selected={selected_ids} "
            f"turns={[((hit.get('metadata') or {}).get('first_mention_turn')) for hit in hits]}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
