"""Replay one AMB synthesis query with contextual reranking over a frozen bank."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import urllib.request
from pathlib import Path


_ENTITY_STOP = {
    "assistant", "user", "turn", "personal", "statement", "january",
    "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "montserrat",
    "media", "hub", "film", "festival", "canadian", "canada",
    "jamaican", "jamaica", "toronto", "kingston", "coursera",
    "university", "cafe", "always", "how", "the", "thanks", "sure",
    "yes", "yeah", "here", "okay", "caribbean", "west", "zoom", "blue",
    "lagoon",
}
_CONTINUATION_RE = re.compile(
    r"^\[(?:[A-Z][a-z]+-\d+-\d+ \| )?Turn \d+\]\s+User:\s*"
    r"(?:sure|yes|yeah|here|okay|ok)\b",
    re.IGNORECASE,
)
_CONTEXT_BRIDGE_RE = re.compile(
    r"\b(?:our\s+(?:[A-Z][A-Za-z-]*\s+){0,2}(?:meeting|call|plan|draft|project|paper|application|"
    r"conversation|next\s+steps)|"
    r"that\s+(?:meeting|call|feedback|advice|recommendation|draft|plan)|"
    r"follow(?:ing)?\s+up\b)\b",
    re.IGNORECASE,
)
_PERSON_TOKEN = r"[A-Z][a-z]+(?:['’-][A-Z]?[a-z]+)*"
_PERSON_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        rf"\b(?:met|meet|meeting|contacted|called|emailed|invited|dating)\s+"
        rf"(?P<person>{_PERSON_TOKEN})\b",
        rf"\b(?:advice|feedback|tips|input|help|support|recommendation)"
        rf"(?:\s+I)?(?:\s+(?:got|received))?\s+(?:from|through|by)\s+"
        rf"(?P<person>{_PERSON_TOKEN})\b",
        rf"\b(?P<person>{_PERSON_TOKEN})(?:'s|’s)\s+"
        rf"(?:advice|feedback|tips|input|opinion|review|recommendation|concern)\b",
        rf"\b(?P<person>{_PERSON_TOKEN})"
        rf"(?:,\s+(?:a|an|who)\b[^,]{{0,40}},)?\s+(?:agreed|shared|met|"
        rf"offered|suggested|recommended|told|invited|reviewed|introduced|"
        rf"helped|gave|provided|expressed)\b",
        rf"\b(?:partner|friend|advisor|mentor|producer),?\s+"
        rf"(?P<person>{_PERSON_TOKEN})\b",
        rf"\b(?P<person>{_PERSON_TOKEN})\s+and\s+I\s+"
        rf"(?:(?:are|were|have been|will be)\s+)?(?:discussing|working|"
        rf"meeting|planning|preparing|reviewing|collaborating)\b",
    )
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_provider(repo: Path):
    script_directory = Path(__file__).resolve().parent
    sys.path = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != script_directory
    ]
    _load_module(
        "memory_bench.memory.chronological_presentation",
        repo / "benchmarks" / "amb" / "chronological_presentation.py",
    )
    _load_module(
        "memory_bench.memory.write_synthesis_selection",
        repo / "benchmarks" / "amb" / "write_synthesis_selection.py",
    )
    return _load_module(
        "memory_bench.memory.yantrikdb_contextual_probe",
        repo / "benchmarks" / "amb" / "yantrikdb.py",
    ).YantrikDBGlobalSynthesisMemoryProvider


def _ollama_embeddings(
    texts: list[str], model: str, host: str
) -> list[list[float]]:
    if not host.startswith("http"):
        host = f"http://{host}"
    for wildcard in ("//0.0.0.0", "//[::]", "//::"):
        host = host.replace(wildcard, "//127.0.0.1")
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/embed",
        data=json.dumps({"model": model, "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    embeddings = payload.get("embeddings") or []
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Ollama returned {len(embeddings)} embeddings for {len(texts)} texts"
        )
    return embeddings


def _ollama_model_identity(model: str, host: str) -> dict:
    """Resolve an Ollama tag to the immutable digest used by this run."""
    if not host.startswith("http"):
        host = f"http://{host}"
    for wildcard in ("//0.0.0.0", "//[::]", "//::"):
        host = host.replace(wildcard, "//127.0.0.1")
    try:
        with urllib.request.urlopen(
            f"{host.rstrip('/')}/api/tags", timeout=30
        ) as response:
            models = json.load(response).get("models") or []
    except Exception as exc:  # pragma: no cover - depends on local Ollama
        return {
            "requested": model,
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }

    accepted_names = {model}
    if ":" not in model:
        accepted_names.add(f"{model}:latest")
    match = next(
        (
            row
            for row in models
            if row.get("name") in accepted_names
            or row.get("model") in accepted_names
        ),
        None,
    )
    if match is None:
        return {"requested": model, "status": "not_found"}
    return {
        "requested": model,
        "status": "resolved",
        "name": match.get("name") or match.get("model"),
        "digest": match.get("digest"),
        "modified_at": match.get("modified_at"),
        "details": match.get("details") or {},
    }


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _turn(text: str) -> int | None:
    match = re.search(r"\[(?:[A-Z][a-z]+-\d+-\d+ \| )?Turn (\d+)\]", text)
    return int(match.group(1)) if match else None


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hit_identity(hit: dict) -> str:
    text_hash = hashlib.sha256((hit.get("text") or "").encode()).hexdigest()
    for key in ("rid", "id", "memory_id"):
        if hit.get(key):
            return f"{hit[key]}:text:{text_hash}"
    return f"text:{text_hash}"


def _named_entities(text: str) -> set[str]:
    return {
        person
        for pattern in _PERSON_PATTERNS
        for match in pattern.finditer(text)
        if (person := match.group("person").casefold()) not in _ENTITY_STOP
    }


def _candidate_bank(
    reranked: list[dict],
    *,
    top_k: int,
    entity_seed_k: int,
    entity_closure_slots: int,
    continuation_slots: int,
    context_bridge_slots: int = 0,
) -> tuple[list[dict], dict[str, dict]]:
    """Build a bounded bank from relevance, entity closure, and turn context."""
    selected_ids = {_hit_identity(hit) for hit in reranked[:top_k]}
    reasons = {
        identity: {"status": "kept", "reason": "base_top_k"}
        for identity in selected_ids
    }
    entities_by_id = {
        _hit_identity(hit): _named_entities(hit.get("text") or "")
        for hit in reranked
    }
    entity_frequency: dict[str, int] = {}
    for entities in entities_by_id.values():
        for entity in entities:
            entity_frequency[entity] = entity_frequency.get(entity, 0) + 1
    seed_ranks: dict[str, int] = {}
    for rank, hit in enumerate(reranked[:entity_seed_k], 1):
        for entity in entities_by_id[_hit_identity(hit)]:
            if entity_frequency[entity] >= 2:
                seed_ranks.setdefault(entity, rank)

    entity_candidates: dict[str, list[tuple[float, int, str]]] = {
        entity: [] for entity in seed_ranks
    }
    if entity_closure_slots:
        for rank, hit in enumerate(reranked, 1):
            identity = _hit_identity(hit)
            if identity in selected_ids:
                continue
            shared = sorted(entities_by_id[identity].intersection(seed_ranks))
            for entity in shared:
                entity_candidates[entity].append((
                    -hit["_contextual_score"], rank, identity
                ))
        for candidates in entity_candidates.values():
            candidates.sort()

        positions = {entity: 0 for entity in seed_ranks}
        added = 0
        seed_order = sorted(seed_ranks, key=lambda entity: seed_ranks[entity])
        while added < entity_closure_slots:
            progressed = False
            for entity in seed_order:
                candidates = entity_candidates[entity]
                position = positions[entity]
                while (
                    position < len(candidates)
                    and candidates[position][2] in selected_ids
                ):
                    position += 1
                positions[entity] = position
                if position >= len(candidates):
                    continue
                _, _, identity = candidates[position]
                positions[entity] += 1
                selected_ids.add(identity)
                shared = sorted(entities_by_id[identity].intersection(seed_ranks))
                reasons[identity] = {
                    "status": "kept",
                    "reason": "entity_closure",
                    "entities": shared,
                }
                added += 1
                progressed = True
                if added >= entity_closure_slots:
                    break
            if not progressed:
                break

    selected_turns = {
        turn: rank
        for rank, hit in enumerate(reranked, 1)
        if _hit_identity(hit) in selected_ids
        and (turn := _turn(hit.get("text") or "")) is not None
    }
    continuation_candidates = []
    if continuation_slots:
        for rank, hit in enumerate(reranked, 1):
            identity = _hit_identity(hit)
            turn = _turn(hit.get("text") or "")
            if (
                identity in selected_ids
                or turn is None
                or turn - 2 not in selected_turns
                or not _CONTINUATION_RE.search(hit.get("text") or "")
            ):
                continue
            continuation_candidates.append((
                -hit["_contextual_score"],
                selected_turns[turn - 2],
                rank,
                identity,
                turn - 2,
            ))
        for _, _, _, identity, parent_turn in sorted(continuation_candidates)[
            :continuation_slots
        ]:
            selected_ids.add(identity)
            reasons[identity] = {
                "status": "kept",
                "reason": "direct_user_continuation",
                "parent_turn": parent_turn,
            }

    if context_bridge_slots:
        selected_turns = {
            turn: rank
            for rank, hit in enumerate(reranked, 1)
            if _hit_identity(hit) in selected_ids
            and (turn := _turn(hit.get("text") or "")) is not None
        }
        context_candidates = []
        for rank, hit in enumerate(reranked, 1):
            identity = _hit_identity(hit)
            turn = _turn(hit.get("text") or "")
            if (
                identity in selected_ids
                or turn is None
                or turn - 2 not in selected_turns
                or not _CONTEXT_BRIDGE_RE.search(hit.get("text") or "")
            ):
                continue
            context_candidates.append((
                selected_turns[turn - 2],
                -hit["_contextual_score"],
                rank,
                identity,
                turn - 2,
            ))
        for _, _, _, identity, parent_turn in sorted(context_candidates)[
            :context_bridge_slots
        ]:
            selected_ids.add(identity)
            reasons[identity] = {
                "status": "kept",
                "reason": "context_bridge",
                "parent_turn": parent_turn,
            }

    return (
        [hit for hit in reranked if _hit_identity(hit) in selected_ids],
        reasons,
    )


def _rank_by_turn(reranked: list[dict]) -> dict[int, int]:
    ranks: dict[int, int] = {}
    for rank, hit in enumerate(reranked, 1):
        turn = _turn(hit.get("text") or "")
        if turn is not None:
            ranks.setdefault(turn, rank)
    return ranks


def _evidence_ledger(
    reranked: list[dict], top_k: int, selection_reasons: dict[str, dict] | None = None
) -> list[dict]:
    """Account for every reranked row, including every cutoff discard."""
    if selection_reasons is None:
        selection_reasons = {
            _hit_identity(hit): {"status": "kept", "reason": "base_top_k"}
            for hit in reranked[:top_k]
        }
    return [
        {
            "identity": _hit_identity(hit),
            "turn": _turn(hit.get("text") or ""),
            "created_at": hit.get("created_at"),
            "source_doc_id": (hit.get("metadata") or {}).get("doc_id"),
            "source_turn_id": (hit.get("metadata") or {}).get("turn_id"),
            "source_chunk_idx": (hit.get("metadata") or {}).get("chunk_idx"),
            "contextual_rank": rank,
            "base_rank": hit["_retrieval_rank"],
            "contextual_score": hit["_contextual_score"],
            "text": hit.get("text") or "",
            "text_sha256": hashlib.sha256(
                (hit.get("text") or "").encode("utf-8")
            ).hexdigest(),
            **selection_reasons.get(
                _hit_identity(hit),
                {"status": "dropped", "reason": "outside_candidate_bank"},
            ),
        }
        for rank, hit in enumerate(reranked, 1)
    ]


def _source_evidence_ceiling(
    gold_turns: set[int], rank_by_turn: dict[int, int], selected_turns: set[int]
) -> dict:
    available = sorted(gold_turns.intersection(selected_turns))
    missing = sorted(gold_turns.difference(available))
    total = len(gold_turns)
    return {
        "gold_source_turn_count": total,
        "available_source_turn_count": len(available),
        "source_turn_recall": len(available) / total if total else None,
        "all_gold_source_turns_available": len(available) == total if total else None,
        "available_source_turns": available,
        "missing_source_turns": missing,
        "source_turn_ranks": {
            str(turn): rank_by_turn.get(turn) for turn in sorted(gold_turns)
        },
    }


def _preflight_record(
    *,
    query: str,
    model_identity: dict,
    embed_model: str,
    rerank_pool: int,
    top_k: int,
    user_hits: list[dict],
    reranked: list[dict],
    selected: list[dict],
    selection_reasons: dict[str, dict],
    gold_turns: set[int],
    entity_seed_k: int,
    entity_closure_slots: int,
    continuation_slots: int,
    context_bridge_slots: int = 0,
) -> dict:
    ledger = _evidence_ledger(reranked, top_k, selection_reasons)
    input_rows = [
        {
            "identity": _hit_identity(hit),
            "base_rank": hit["_retrieval_rank"],
            "text_sha256": hashlib.sha256(
                (hit.get("text") or "").encode("utf-8")
            ).hexdigest(),
        }
        for hit in user_hits
    ]
    rerank_rows = [
        {
            "identity": row["identity"],
            "rank": row["contextual_rank"],
            "score_hex": reranked[row["contextual_rank"] - 1][
                "_contextual_score"
            ].hex(),
        }
        for row in ledger
    ]
    rank_by_turn = _rank_by_turn(reranked)
    selected_turns = {
        turn
        for hit in selected
        if (turn := _turn(hit.get("text") or "")) is not None
    }
    candidate_bank_rows = [
        {
            "identity": _hit_identity(hit),
            "contextual_rank": rank_by_turn.get(
                _turn(hit.get("text") or ""), 999999
            ),
            "selection": selection_reasons[_hit_identity(hit)],
        }
        for hit in selected
    ]
    return {
        "status": "ready",
        "config": {
            "embed_model": embed_model,
            "rerank_pool": rerank_pool,
            "top_k": top_k,
            "entity_seed_k": entity_seed_k,
            "entity_closure_slots": entity_closure_slots,
            "continuation_slots": continuation_slots,
            "context_bridge_slots": context_bridge_slots,
        },
        "model_identity": model_identity,
        "input_sha256": _sha256_json({"query": query, "rows": input_rows}),
        "rerank_sha256": _sha256_json(rerank_rows),
        "candidate_bank_sha256": _sha256_json(candidate_bank_rows),
        "source_evidence_ceiling": _source_evidence_ceiling(
            gold_turns, rank_by_turn, selected_turns
        ),
        "evidence_ledger": ledger,
    }


def _compare_preflight(current: dict, path: Path) -> dict:
    prior_artifact = json.loads(path.read_text(encoding="utf-8"))
    prior = prior_artifact.get("preflight", prior_artifact)
    compared = {
        key: {"expected": prior.get(key), "actual": current.get(key)}
        for key in ("input_sha256", "rerank_sha256", "candidate_bank_sha256")
    }
    compared["model_digest"] = {
        "expected": (prior.get("model_identity") or {}).get("digest"),
        "actual": (current.get("model_identity") or {}).get("digest"),
    }
    mismatches = [
        key
        for key, values in compared.items()
        if values["expected"] != values["actual"]
    ]
    return {
        "artifact": str(path),
        "matched": not mismatches,
        "mismatches": mismatches,
        "values": compared,
    }


def _cited_turns(items: list[dict], telemetry: dict) -> set[int]:
    block_turns = telemetry.get("evidence_block_turns") or {}
    return {
        int(turn)
        for item in items
        for evidence_id in item.get("evidence_ids", [])
        for turn in block_turns.get(evidence_id, [])
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--user-id", default="9")
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--rerank-pool", type=int, default=1000)
    parser.add_argument("--entity-seed-k", type=int, default=10)
    parser.add_argument("--entity-closure-slots", type=int, default=0)
    parser.add_argument("--continuation-slots", type=int, default=0)
    parser.add_argument("--context-bridge-slots", type=int, default=0)
    parser.add_argument(
        "--support-quotes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--gold-turns", default="")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="write retrieval/rerank evidence and stop before synthesis",
    )
    parser.add_argument(
        "--compare-preflight",
        type=Path,
        help="refuse synthesis unless hashes match this frozen preflight",
    )
    parser.add_argument(
        "--expect-model-digest",
        help="refuse the run unless Ollama resolves this immutable digest",
    )
    args = parser.parse_args()

    if (
        args.top_k < 1
        or args.rerank_pool < args.top_k
        or args.entity_seed_k < 1
        or args.entity_closure_slots < 0
        or args.continuation_slots < 0
        or args.context_bridge_slots < 0
    ):
        parser.error("require rerank-pool >= top-k >= 1")

    os.environ["YDB_BENCH_SYNTH_BLOCKS"] = str(args.rerank_pool)
    os.environ["YDB_BENCH_SYNTH_SUPPORT_QUOTES"] = (
        "1" if args.support_quotes else "0"
    )
    if os.environ.get("YDB_BENCH_SYNTH_EVIDENCE_CANDIDATES") == "1":
        os.environ["YDB_BENCH_SYNTH_MAX_ITEMS"] = str(max(
            args.top_k + args.entity_closure_slots + args.continuation_slots
            + args.context_bridge_slots,
            int(os.environ.get("YDB_BENCH_SYNTH_MAX_ITEMS", "24")),
        ))
    provider_cls = _load_provider(args.repo.resolve())
    provider = provider_cls()
    try:
        provider.prepare(args.bank.resolve(), {args.user_id}, False)
        recalled = provider._recall(args.query, args.rerank_pool, args.user_id)
        user_hits = provider._select_evidence_hits(recalled)
        model_identity = _ollama_model_identity(args.embed_model, args.ollama_host)
        if args.expect_model_digest and (
            model_identity.get("digest") != args.expect_model_digest
        ):
            raise RuntimeError(
                "Ollama model digest mismatch: expected "
                f"{args.expect_model_digest}, got {model_identity.get('digest')}"
            )

        embeddings = _ollama_embeddings(
            [args.query, *(hit.get("text") or "" for hit in user_hits)],
            args.embed_model,
            args.ollama_host,
        )
        query_embedding, document_embeddings = embeddings[0], embeddings[1:]
        reranked = sorted(
            (
                dict(hit, _contextual_score=_cosine(query_embedding, embedding))
                for hit, embedding in zip(
                    user_hits, document_embeddings, strict=True
                )
            ),
            key=lambda hit: (-hit["_contextual_score"], hit["_retrieval_rank"]),
        )
        selected_bank, selection_reasons = _candidate_bank(
            reranked,
            top_k=args.top_k,
            entity_seed_k=args.entity_seed_k,
            entity_closure_slots=args.entity_closure_slots,
            continuation_slots=args.continuation_slots,
            context_bridge_slots=args.context_bridge_slots,
        )
        selected = [
            dict(
                hit,
                _base_retrieval_rank=hit["_retrieval_rank"],
                _retrieval_rank=rank,
                _candidate_bank_reason=selection_reasons[_hit_identity(hit)],
            )
            for rank, hit in enumerate(selected_bank, 1)
        ]
        gold_turns = {
            int(value) for value in args.gold_turns.split(",") if value.strip()
        }
        rerank_by_turn = _rank_by_turn(reranked)
        preflight = _preflight_record(
            query=args.query,
            model_identity=model_identity,
            embed_model=args.embed_model,
            rerank_pool=args.rerank_pool,
            top_k=args.top_k,
            user_hits=user_hits,
            reranked=reranked,
            selected=selected_bank,
            selection_reasons=selection_reasons,
            gold_turns=gold_turns,
            entity_seed_k=args.entity_seed_k,
            entity_closure_slots=args.entity_closure_slots,
            continuation_slots=args.continuation_slots,
            context_bridge_slots=args.context_bridge_slots,
        )
        comparison = (
            _compare_preflight(preflight, args.compare_preflight)
            if args.compare_preflight else
            None
        )
        artifact = {
            "query": args.query,
            "embed_model": args.embed_model,
            "generator_model": os.environ.get("YDB_BENCH_SYNTH_MODEL", ""),
            "raw_recall_count": len(recalled),
            "user_turn_count": len(user_hits),
            "selected_count": len(selected),
            "preflight": preflight,
            "preflight_comparison": comparison,
            "gold_turn_rerank": {
                str(turn): rerank_by_turn.get(turn)
                for turn in sorted(gold_turns)
            },
            "gold_turns_available": preflight["source_evidence_ceiling"][
                "available_source_turns"
            ],
            "selected_turns": [_turn(hit.get("text") or "") for hit in selected],
            "selected_evidence": [
                row
                for row in preflight["evidence_ledger"]
                if row["status"] == "kept"
            ],
            "candidate_gold_turns": [],
            "result_gold_turns": [],
            "items": [],
            "telemetry": {"synthesis_status": "skipped_preflight"},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

        if comparison and not comparison["matched"]:
            raise RuntimeError(
                "preflight mismatch; refusing synthesis: "
                + ", ".join(comparison["mismatches"])
            )
        if not args.preflight_only:
            items, telemetry = provider._synthesize(args.query, selected)
            candidate_turns = _cited_turns(
                telemetry.get("candidate_items", []), telemetry
            )
            result_turns = _cited_turns(telemetry.get("results", []), telemetry)
            artifact.update({
                "candidate_gold_turns": sorted(
                    gold_turns.intersection(candidate_turns)
                ),
                "result_gold_turns": sorted(gold_turns.intersection(result_turns)),
                "items": items,
                "telemetry": telemetry,
            })
            args.output.write_text(
                json.dumps(artifact, indent=2), encoding="utf-8"
            )

        telemetry = artifact["telemetry"]
        print(json.dumps({
            "output": str(args.output),
            "preflight_only": args.preflight_only,
            "input_sha256": preflight["input_sha256"],
            "rerank_sha256": preflight["rerank_sha256"],
            "candidate_bank_sha256": preflight["candidate_bank_sha256"],
            "model_digest": model_identity.get("digest"),
            "source_evidence_ceiling": preflight["source_evidence_ceiling"],
            "gold_turn_rerank": artifact["gold_turn_rerank"],
            "candidate_gold_turns": artifact["candidate_gold_turns"],
            "result_gold_turns": artifact["result_gold_turns"],
            "candidate_count": telemetry.get("synthesis_items"),
            "result_count": len(artifact["items"]),
            "extraction_support_quote_events": telemetry.get(
                "extraction_support_quote_events", []
            ),
        }, indent=2))
    finally:
        provider.cleanup()


if __name__ == "__main__":
    main()
