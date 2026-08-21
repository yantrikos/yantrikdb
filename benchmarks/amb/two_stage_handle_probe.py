"""Probe query-time handle selection followed by grouped child selection."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

def index_handles(artifact: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    """Assign deterministic per-artifact IDs and discard invalid empty handles."""
    atomics = {
        str(item["id"]): item
        for item in artifact.get("input_items") or []
        if isinstance(item, dict) and item.get("id")
    }
    handles = {}
    for index, value in enumerate(artifact.get("handles") or [], 1):
        evidence_ids = [
            evidence_id
            for evidence_id in dict.fromkeys(value.get("evidence_ids") or [])
            if evidence_id in atomics
        ]
        if not evidence_ids:
            continue
        handle_id = f"H{index:03d}"
        handle = dict(value)
        handle["id"] = handle_id
        handle["evidence_ids"] = evidence_ids
        handles[handle_id] = handle
    return handles, atomics


def hydrate_handles(
    selected_handle_ids: list[str],
    handles: dict[str, dict],
    atomics: dict[str, dict],
) -> list[dict]:
    """Hydrate children while retaining every selected group membership."""
    hydrated = {}
    for handle_id in selected_handle_ids:
        handle = handles[handle_id]
        for evidence_id in handle["evidence_ids"]:
            item = hydrated.setdefault(
                evidence_id,
                {
                    **atomics[evidence_id],
                    "handle_ids": [],
                    "handle_labels": [],
                },
            )
            item["handle_ids"].append(handle_id)
            item["handle_labels"].append(handle.get("label") or handle_id)
    return list(hydrated.values())


def _load_workspace_ollama(path: Path):
    name = "memory_bench.llm._workspace_ollama_handle_probe"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Ollama provider from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.OllamaLLM


def _handle_selection_prompt(
    query: str, handles: dict[str, dict], count: int, max_handles: int
) -> str:
    rows = [
        {
            "id": handle_id,
            "label": handle.get("label"),
            "anchor_entities": handle.get("anchor_entities") or [],
            "summary": handle.get("summary"),
            "child_count": len(handle["evidence_ids"]),
        }
        for handle_id, handle in handles.items()
    ]
    return (
        "Select the memory topic handles whose child timelines are most likely to "
        "contain the distinct events needed to answer the user query. This is a "
        "retrieval step: do not answer the query and do not choose individual events. "
        "Prefer narrow role/person/concern handles over a broad project handle when "
        "they expose distinct relevant evidence. Include multiple handles when the "
        "requested answer spans people or roles. Reject handles that merely share the "
        "project but describe a different relationship or concern. Select by semantic "
        "fit, not chronology. No expected answer is available. "
        f"The final answer needs {count} items; select 1-{max_handles} handles. "
        "Return JSON only as "
        '{"selected_handle_ids":["H001"],"rationale":"..."}.'
        f"\n\nUSER QUERY:\n{query}\n\nHANDLES:\n"
        + json.dumps(rows, ensure_ascii=True, separators=(",", ":"))
    )


def _item_selection_prompt(query: str, grouped_items: list[dict], count: int) -> str:
    groups: dict[str, dict] = {}
    for item in grouped_items:
        for handle_id, label in zip(item["handle_ids"], item["handle_labels"]):
            group = groups.setdefault(
                handle_id, {"handle_id": handle_id, "label": label, "items": []}
            )
            group["items"].append(
                {
                    "id": item["id"],
                    "turn": item.get("turn"),
                    "text": item.get("text"),
                }
            )
    return (
        "Select the exact source-grounded memory events needed to answer the user "
        "query. Return IDs only; do not rewrite or answer. Compare all handle groups "
        "before selecting. Choose distinct substantive events that form the narrow "
        "semantic thread in the query, not events that only share its broad project. "
        "A correction, denial, retraction, or superseded claim is not an active event "
        "unless the query explicitly asks about corrections. Repeated follow-ups about "
        "one concern count once. Select by meaning first; turn numbers are only for "
        "ordering after membership is fixed. No expected answer is available. "
        f"Select exactly {count} distinct IDs. Return JSON only as "
        '{"selected_ids":["A0001"],"rationale":"..."}.'
        f"\n\nUSER QUERY:\n{query}\n\nGROUPED CHILD TIMELINES:\n"
        + json.dumps(list(groups.values()), ensure_ascii=True, separators=(",", ":"))
    )


_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_ids": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["selected_ids", "rationale"],
}


def main() -> int:
    from memory_bench.dataset import get_dataset
    from memory_bench.models import QueryResult

    try:
        from .global_organizer_probe import _chat_json
        from .select_concern_context import (
            normalize_selected_ids,
            requested_count,
        )
    except ImportError:  # Direct script execution.
        from global_organizer_probe import _chat_json
        from select_concern_context import normalize_selected_ids, requested_count

    parser = argparse.ArgumentParser()
    parser.add_argument("--organizer", type=Path, required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--judge-model")
    parser.add_argument(
        "--include-gold",
        action="store_true",
        help="Include gold text in the saved audit artifact after selection.",
    )
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--split", default="100k")
    parser.add_argument("--max-handles", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.max_handles < 1:
        parser.error("--max-handles must be positive")

    dataset = get_dataset("beam")
    query = next(
        candidate
        for candidate in dataset.load_queries(args.split)
        if candidate.id == args.query_id
    )
    count = requested_count(query.query)
    if count is None:
        parser.error("could not determine requested answer count")

    artifact = json.loads(args.organizer.read_text(encoding="utf-8"))
    handles, atomics = index_handles(artifact)
    if not handles:
        parser.error("organizer artifact has no valid handles")

    handle_schema = {
        "type": "object",
        "properties": {
            "selected_handle_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": args.max_handles,
            },
            "rationale": {"type": "string"},
        },
        "required": ["selected_handle_ids", "rationale"],
    }
    handle_raw, handle_response = _chat_json(
        host=args.host,
        model=args.model,
        prompt=_handle_selection_prompt(
            query.query, handles, count, args.max_handles
        ),
        schema=handle_schema,
        num_predict=1600,
        timeout=args.timeout,
    )
    selected_handle_ids = normalize_selected_ids(
        {"selected_ids": handle_raw.get("selected_handle_ids")},
        set(handles),
        args.max_handles,
    )
    if not selected_handle_ids:
        raise ValueError(f"handle selector returned no valid IDs: {handle_raw!r}")

    grouped_items = hydrate_handles(selected_handle_ids, handles, atomics)
    item_raw, item_response = _chat_json(
        host=args.host,
        model=args.model,
        prompt=_item_selection_prompt(query.query, grouped_items, count),
        schema={
            **_SELECTION_SCHEMA,
            "properties": {
                **_SELECTION_SCHEMA["properties"],
                "selected_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": count,
                    "maxItems": count,
                },
            },
        },
        num_predict=1600,
        timeout=args.timeout,
    )
    hydrated_ids = {item["id"] for item in grouped_items}
    selected_ids = normalize_selected_ids(item_raw, hydrated_ids, count)
    if len(selected_ids) != count:
        raise ValueError(
            f"item selector returned {len(selected_ids)} valid IDs, expected "
            f"{count}: {item_raw!r}"
        )
    selected_items = sorted(
        (atomics[item_id] for item_id in selected_ids),
        key=lambda item: (
            float("inf") if item.get("turn") is None else item["turn"],
            item["id"],
        ),
    )
    answer = "\n".join(
        f"{index}. {item['text']}" for index, item in enumerate(selected_items, 1)
    )
    context = "\n\n".join(
        f"[Turn {item.get('turn')}] User: {item['text']}" for item in selected_items
    )
    payload = {
        "query_id": query.id,
        "query": query.query,
        "organizer_model": artifact.get("model"),
        "selector_model": args.model,
        "selected_handle_ids": selected_handle_ids,
        "selected_handle_labels": [
            handles[handle_id].get("label") for handle_id in selected_handle_ids
        ],
        "handle_selection_rationale": handle_raw.get("rationale"),
        "hydrated_item_count": len(grouped_items),
        "model_selected_item_ids": selected_ids,
        "selected_item_ids": [item["id"] for item in selected_items],
        "selected_turns": [item.get("turn") for item in selected_items],
        "item_selection_rationale": item_raw.get("rationale"),
        "answer": answer,
        "selection_usage": {
            "handle_prompt_tokens": handle_response.get("prompt_eval_count"),
            "handle_completion_tokens": handle_response.get("eval_count"),
            "item_prompt_tokens": item_response.get("prompt_eval_count"),
            "item_completion_tokens": item_response.get("eval_count"),
        },
    }
    if args.include_gold:
        payload["gold_answers"] = query.gold_answers
    if args.judge_model:
        meta = dict(query.meta)
        judged = QueryResult(
            query_id=query.id,
            query=query.query,
            answer=answer,
            reasoning="",
            context=context,
            context_tokens=0,
            retrieve_time_ms=0.0,
            gold_answers=query.gold_answers,
            correct=False,
            judge_reason="",
            meta=meta,
        )
        OllamaLLM = _load_workspace_ollama(Path(__file__).with_name("ollama.py"))
        judge = OllamaLLM(
            args.judge_model, think=False, num_predict=1200, num_ctx=65536
        )
        payload["judge_model"] = args.judge_model
        payload["score"] = dataset.score_result(judged, judge)
        payload["all_rubric_nuggets_matched"] = payload["score"] == 1.0
        payload["score_scope"] = (
            "mean rubric-nugget coverage; ordering and exact-N are not judged"
        )

    rendered = json.dumps(payload, indent=2, ensure_ascii=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
