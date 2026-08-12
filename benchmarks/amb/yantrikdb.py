"""YantrikDB memory provider — an embedded cognitive memory engine.

Positioning (what makes this run interesting on the leaderboard):
- NO generative model in the memory layer: ingest is chunk -> embed -> index,
  with no LLM extraction/consolidation pass. Retrieval is HNSW cosine fused
  with BM25 lexical rank (engine-side, one SQLite-backed file per bank).
- Fully local: the default build ships a bundled ~7 MB static embedder
  (potion-base-2M, 64d); `pip install yantrikdb` is the entire setup.
- Event-time aware: each BEAM session chunk is recorded with its real
  `time_anchor` as `created_at` (engine 0.14 historical-import surface), so
  the engine's decay/recency lane sees the conversation's actual timeline
  instead of the ingest wall-clock, and `recall_as_of` stays meaningful.

Variants:
- `yantrikdb`         — engine recall (cosine + BM25 fusion), k=20 chunks.
- `yantrikdb-rerank`  — same index, cross-encoder reranking over a wider
                        pool (ms-marco-MiniLM-L6-v2; needs the optional
                        sentence-transformers dependency).

Documents are split with the harness's own `chunk_text` (512-token windows,
same as the bm25/qdrant baselines) for cross-provider comparability; each
chunk records `metadata={"doc_id": ...}` and retrieval returns chunk-level
Documents carrying the parent doc id, exactly like the qdrant provider.
"""
import logging
import time
import os
import re
from datetime import datetime
from pathlib import Path

from ..models import Document
from ..utils import chunk_text, count_tokens
from .base import MemoryProvider

# BEAM's turn header, in every form its formatter emits:
# "[March-15-2024 | Turn 7]", "[March-15-2024]", "[Turn 7]".
#
# Deliberately EXACT rather than a loose "[...]" match. A permissive
# bracket pattern reports 87% of chunks as "carrying a header" because it
# also matches markdown links, code indices and log levels inside the
# conversation body — which is how a first attempt at this measurement
# looked like it was working when it was not.
_HEADER = r"\[(?:[A-Z][a-z]+-\d+-\d+(?: \| Turn \d+)?|Turn \d+)\]"
_HEADER_RE = re.compile(_HEADER)
_TURN_SPLIT_RE = re.compile(rf"(?=\n*{_HEADER})")

logger = logging.getLogger(__name__)

# Retrieval depth (chunks returned to the answer prompt). ~512 tokens/chunk
# so k=20 is a ~10K-token context — half of the qdrant baseline's budget.
_TOP_K = int(os.environ.get("YDB_BENCH_TOPK", "20"))
# Tokens per chunk, and whether to chunk on turn boundaries (see
# _turn_aware_chunks) instead of the harness's fixed 512-token windows.
_CHUNK_TOKENS = int(os.environ.get("YDB_BENCH_CHUNK_TOKENS", "512"))
_TURN_AWARE = os.environ.get("YDB_BENCH_TURN_AWARE", "0") == "1"
# Candidate pool for the rerank variant (cross-encoder reads pool, returns k).
_RERANK_POOL = int(os.environ.get("YDB_BENCH_RERANK_POOL", "50"))
# Confidence gate for the `-floor` variant: keep only results scoring at least
# this fraction of the top hit (engine-side `min_score_ratio`), and drop the
# whole result set when even the top hit is below `_FLOOR_ABS`.
_FLOOR_RATIO = float(os.environ.get("YDB_BENCH_FLOOR_RATIO", "0.7"))
_FLOOR_ABS = float(os.environ.get("YDB_BENCH_FLOOR_ABS", "0.30"))

# Embedder selection. Empty = the bundled default (potion-base-2M, 64d, a
# STATIC model2vec lookup table with no transformer inference). Prior
# measurement on a different corpus: raising dimension WITHIN the potion
# family does not fix paraphrase matching (512d scored worse than 64d), but
# a contextual embedder moved a probe from rank 32 to rank 2. BEAM questions
# are paraphrases of conversation content, so this is the suspected lever.
#   YDB_BENCH_EMBEDDER=potion-base-8M      -> engine registry, downloadable
#   YDB_BENCH_EMBEDDER=ollama:nomic-embed-text -> local contextual, 768d
_EMBEDDER = os.environ.get("YDB_BENCH_EMBEDDER", "").strip()
# Run the engine's cognition loop once after ingest: consolidation, conflict
# scan, pattern mining, trigger expiry, personality. Engine-native (no LLM),
# so it keeps the zero-LLM-ingest property. Measured on one BEAM bank (230
# chunks): 61s, merged 4 records into 1, entities 1508 -> 2402, conflicts
# 0 -> 31 open, and changed the top-20 for 19 of 20 queries while costing
# ZERO queries more than 10% of their context. Whether that churn helps or
# hurts is what the judged arm decides.
_THINK = os.environ.get("YDB_BENCH_THINK", "0") == "1"
# LLM-assisted maintenance after ingest (see memory/cognition.py). Distinct
# from _THINK: that runs the engine's own cognition loop, whose consolidation
# writes an extractive join carrying the cluster MEAN embedding and measured
# -7/80. This writes a model synthesis embedded from its own text, which is
# the repair for that exact mechanism.
_COGNITION = os.environ.get("YDB_BENCH_COGNITION", "0") == "1"


class _OllamaEmbedder:
    """Adapter exposing `encode(text) -> list[float]` over ollama /api/embeddings.

    The engine probes any Python embedder at set time by calling
    `encode("__yantrikdb_probe__")` and rejects anything that does not return
    a numeric vector, so the interface must be exactly this.
    """

    def __init__(self, model: str, host: str | None = None):
        self.model = model
        self._host = (host or os.environ.get("OMB_OLLAMA_URL")
                      or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
        if not self._host.startswith("http"):
            self._host = f"http://{self._host}"
        # OLLAMA_HOST is often a BIND address; 0.0.0.0 is not dialable.
        for wildcard in ("//0.0.0.0", "//[::]", "//::"):
            if wildcard in self._host:
                self._host = self._host.replace(wildcard, "//127.0.0.1")
        self._host = self._host.rstrip("/")
        self.dim = len(self.encode("dimension probe"))

    def encode(self, text: str) -> list[float]:
        import json as _json
        import urllib.request
        body = _json.dumps({"model": self.model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{self._host}/api/embeddings", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return _json.load(r)["embedding"]


def _turn_aware_chunks(text: str, budget: int = _CHUNK_TOKENS) -> list[str]:
    """Chunk on TURN boundaries, keeping every chunk's date/turn header.

    The harness's `chunk_text` splits every 512 tokens regardless of
    structure. Measured consequence on beam/100k: 19 of 20 retrieved chunks
    landed mid-turn and carried NO date and NO turn id, because BEAM writes
    its `[March-15-2024 | Turn 7]` prefixes only at turn boundaries. The
    answerer then cannot order, date, or attribute anything it reads — and
    `event_ordering` scored 7%.

    Turns are the natural unit here: they are what the conversation is made
    of, and they are what carries the timestamp. This packs whole turns up
    to `budget` tokens. A single turn larger than the budget is split, but
    each piece is re-prefixed with that turn's header so no fragment is ever
    undated (`... (cont.)` marks the continuations).
    """
    # Split BEFORE each turn header. Splitting on blank lines does not work:
    # turn bodies contain markdown paragraphs and fenced code, so "\n\n"
    # fragments a turn rather than delimiting it (measured: only 16% of the
    # resulting chunks began with a real header).
    turns = [t.strip() for t in _TURN_SPLIT_RE.split(text) if t.strip()]
    out: list[str] = []
    buf: list[str] = []
    used = 0
    for turn in turns:
        n = count_tokens(turn)
        if n > budget:
            if buf:
                out.append("\n\n".join(buf))
                buf, used = [], 0
            header = _HEADER_RE.match(turn)
            prefix = header.group(0) if header else ""
            for i, piece in enumerate(chunk_text(turn, budget)):
                out.append(piece if i == 0 else f"{prefix} (cont.) {piece}")
            continue
        if used + n > budget and buf:
            out.append("\n\n".join(buf))
            buf, used = [], 0
        buf.append(turn)
        used += n
    if buf:
        out.append("\n\n".join(buf))
    return out or [text]


def _iso_to_epoch(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return None


class YantrikDBMemoryProvider(MemoryProvider):
    name = "yantrikdb"
    description = (
        "Embedded cognitive memory engine (Rust core, one SQLite-backed file "
        "per bank). HNSW cosine + BM25 lexical fusion, event-time-aware "
        "decay/recency, bundled 7 MB local embedder (potion-base-2M, 64d). "
        "Zero LLM calls and zero network in the memory layer. Documents "
        "chunked into 512-token windows; retrieves top-k=20 chunks."
    )
    kind = "local"
    provider = "yantrikdb"
    link = "https://github.com/yantrikos/yantrikdb"
    logo = "https://www.google.com/s2/favicons?sz=32&domain=yantrikdb.com"
    concurrency = 4

    def __init__(self):
        self._dbs: dict[str, object] = {}  # unit_id (or "") -> YantrikDB
        self._store_dir: Path | None = None
        self._per_unit = False

    # ------------------------------------------------------------------
    # storage
    # ------------------------------------------------------------------

    def prepare(self, store_dir: Path, unit_ids: set[str] | None = None, reset: bool = True) -> None:
        root = store_dir / "yantrikdb"
        root.mkdir(parents=True, exist_ok=True)
        self._store_dir = root
        self._per_unit = unit_ids is not None
        if reset:
            self._close_all()
            for f in root.glob("*.db*"):
                f.unlink()
        # Banks open lazily in _db_for — opening 100 engines upfront buys
        # nothing and slows a --query-limit smoke run.

    def cleanup(self) -> None:
        self._close_all()

    def _close_all(self) -> None:
        for db in self._dbs.values():
            try:
                db.close()
            except Exception:  # pragma: no cover - best-effort teardown
                pass
        self._dbs.clear()

    def _db_for(self, user_id: str | None):
        """One engine per isolation unit; a single shared engine (with
        namespace scoping at record/recall time) when the dataset has none."""
        key = user_id if (self._per_unit and user_id) else ""
        db = self._dbs.get(key)
        if db is None:
            import yantrikdb

            if self._store_dir is None:
                # No prepare() (some harness paths) — in-memory fallback.
                path = ":memory:"
            else:
                fname = f"{key}.db" if key else "shared.db"
                path = str(self._store_dir / fname)
            db = self._open(yantrikdb, path)
            self._dbs[key] = db
        return db

    @staticmethod
    def _open(yantrikdb, path: str):
        """Open an engine on the configured embedder.

        `with_default` hard-wires the bundled 64d embedder, so a contextual
        model needs the explicit constructor with a matching `embedding_dim`
        — a dim mismatch is refused at write time, not silently coerced.
        """
        if not _EMBEDDER:
            return yantrikdb.YantrikDB.with_default(path)
        if _EMBEDDER.startswith("ollama:"):
            emb = _OllamaEmbedder(_EMBEDDER.split(":", 1)[1])
            logger.info("embedder %s -> %d dims", _EMBEDDER, emb.dim)
            return yantrikdb.YantrikDB(
                db_path=path, embedding_dim=emb.dim, embedder=emb
            )
        # Engine-registry name (potion-base-8M / -32M): open on the bundled
        # embedder, then swap. The registry knows each model's dimension.
        db = yantrikdb.YantrikDB.with_default(path)
        db.set_embedder_named(_EMBEDDER)
        return db

    def _namespace(self, user_id: str | None) -> str:
        # Per-unit banks already isolate; the shared bank scopes by namespace.
        return "default" if self._per_unit else (user_id or "default")

    # ------------------------------------------------------------------
    # ingest / retrieve
    # ------------------------------------------------------------------

    def ingest(self, documents: list[Document]) -> None:
        for doc in documents:
            db = self._db_for(doc.user_id)
            created_at = _iso_to_epoch(doc.timestamp)
            namespace = self._namespace(doc.user_id)
            pieces = (
                _turn_aware_chunks(doc.content, _CHUNK_TOKENS)
                if _TURN_AWARE
                else chunk_text(doc.content, _CHUNK_TOKENS)
            )
            for idx, chunk in enumerate(pieces):
                db.record(
                    chunk,
                    memory_type="episodic",
                    metadata={"doc_id": doc.id, "chunk_idx": idx},
                    namespace=namespace,
                    created_at=created_at,
                )
        if _COGNITION:
            # LLM-assisted maintenance: engine finds clusters, a 0.8B writes
            # the synthesis, a grounding gate rejects anything not traceable
            # to the sources, engine records it. Ingest itself stays LLM-free
            # — this runs once, after, on detected clusters only.
            from .cognition import consolidate_with_model, resolve_conflicts_with_model
            for key, db in self._dbs.items():
                stats = consolidate_with_model(db, lambda t: db.embed_text(t))
                logger.info("cognition consolidate on %s: %s", key or "(shared)", stats)
                db.scan_conflicts()
                logger.info("cognition conflicts on %s: %s",
                            key or "(shared)", resolve_conflicts_with_model(db))
        if _THINK:
            for key, db in self._dbs.items():
                t0 = time.perf_counter()
                r = db.think()
                logger.info(
                    "think() on %s: %.1fs consolidations=%s conflicts=%s patterns=%s",
                    key or "(shared)", time.perf_counter() - t0,
                    r.get("consolidation_count"), r.get("conflicts_found"),
                    r.get("patterns_new"),
                )

    def _recall(self, query: str, k: int, user_id: str | None) -> list[dict]:
        db = self._db_for(user_id)
        return db.recall(
            query=query,
            top_k=k,
            namespace=None if self._per_unit else self._namespace(user_id),
            # Determinism across runs: recall must not reinforce (mutate
            # adaptive state) during a benchmark sweep.
            skip_reinforce=True,
        )

    @staticmethod
    def _to_documents(hits: list[dict], user_id: str | None) -> list[Document]:
        docs = []
        for h in hits:
            meta = h.get("metadata") or {}
            docs.append(
                Document(
                    id=str(meta.get("doc_id", h.get("rid", ""))),
                    content=h.get("text", ""),
                    user_id=user_id,
                )
            )
        return docs

    async def async_retrieve(self, query: str, k: int = _TOP_K, user_id: str | None = None, query_timestamp: str | None = None):
        """Override the BASE default k, not just retrieve()'s.

        The runner reaches providers through `async_retrieve`, whose base
        signature defaults k=10 and passes it EXPLICITLY to retrieve() — so
        a subclass default on retrieve() never applies on the eval path.
        hybrid_search overrides this for the same reason. Measured cost of
        missing it: every run up to 2026-08-11 evaluated at k=10 while
        reporting configs of k=20 and k=50, and the k=50 ladder arm was
        silently a replicate of the k=20 arm.
        """
        import asyncio
        return await asyncio.to_thread(self.retrieve, query, k, user_id, query_timestamp)

    def retrieve(
        self,
        query: str,
        k: int = _TOP_K,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        hits = self._recall(query, k, user_id)
        raw = [
            {"rid": h.get("rid"), "score": h.get("score"), "why": h.get("why_retrieved")}
            for h in hits
        ]
        return self._to_documents(hits, user_id), {"results": raw, "k": k}


class YantrikDBTemporalMemoryProvider(YantrikDBMemoryProvider):
    """Surfaces the event time the engine already stores, in the context.

    Measured cause, not a guess. On beam/100k, `event_ordering` is the worst
    category by a wide margin (7% binary, 0.122 rubric at n=28). Probing one
    failing query showed why: BEAM's formatter writes `[March-15-2024 |
    Turn 0]` prefixes at TURN boundaries, but the harness chunker splits
    every 512 tokens, so 19 of 20 returned chunks land mid-turn and carry NO
    date and NO turn id. The answerer is handed 20 undated chunks in
    RELEVANCE order and asked to reconstruct first-mention order.

    The engine is not missing the information — the same 20 chunks carry
    three distinct `created_at` values (the session time anchors, recorded
    via the 0.14 historical-import path). Two changes, both of which only
    re-expose data already held:

      * stamp each chunk's event date into its content, so the answerer can
        see when it happened;
      * order the returned chunks CHRONOLOGICALLY rather than by relevance,
        so reading order matches event order.

    Ranking is untouched — the same top-k set is selected by the same
    scoring, then presented in time order. This should help the ordering and
    update categories and is neutral-to-harmful nowhere obvious, but that is
    a prediction: read `event_ordering`, `knowledge_update` and
    `temporal_reasoning` per-category, and check the others for regression.
    """

    name = "yantrikdb-temporal"
    description = (
        "YantrikDB with event-time-aware presentation: each retrieved chunk "
        "is stamped with the date it occurred and the set is ordered "
        "chronologically instead of by relevance. Same selection, same "
        "ranking, same zero LLM calls — only the presentation of the "
        "engine's stored event time changes."
    )
    provider = "yantrikdb"
    variant = "temporal"

    @staticmethod
    def _to_documents(hits: list[dict], user_id: str | None) -> list[Document]:
        ordered = sorted(hits, key=lambda h: h.get("created_at") or 0.0)
        docs = []
        for h in ordered:
            meta = h.get("metadata") or {}
            ts = h.get("created_at")
            stamp = ""
            if ts:
                stamp = f"[{datetime.fromtimestamp(ts).strftime('%B %d, %Y')}] "
            docs.append(
                Document(
                    id=str(meta.get("doc_id", h.get("rid", ""))),
                    content=stamp + h.get("text", ""),
                    user_id=user_id,
                )
            )
        return docs


class YantrikDBFloorMemoryProvider(YantrikDBMemoryProvider):
    """Same index; retrieval refuses to hand over low-confidence context.

    BEAM's `abstention` questions are built so the SPECIFIC fact is absent
    while TANGENTIAL material is present. Measured on beam/100k: handing the
    answerer 5,177 tokens of plausibly-adjacent chunks made it confabulate
    specifics for a question whose gold answer is "no such information", and
    made a second answer open with a correct abstention and then answer
    anyway. Both scored 0. On that category, returning good-looking context
    is worse than returning none.

    A scored retriever can say so. Two engine-side gates:
      * `min_score_ratio` — drop results below a fraction of the top hit
        (the engine's own knob; measured at 0.7 to cost zero answer loss
        while shedding ~27% of results).
      * an absolute floor — if even the TOP hit is weak, return NOTHING, so
        the prompt's context block is empty and the answerer has explicit
        grounds to abstain rather than having to infer absence from noise.

    THE ABSOLUTE FLOOR IS REFUTED — measured, kept for reproducibility.

    Swept over 200 queries (10 conversations x 20 questions, beam/100k),
    scoring every query's top hit. The signal is REAL but far too weak to
    act on: abstention is reliably the lowest-scoring of all ten categories
    (mean top score 0.491 vs 0.549 for the other nine), yet the
    distributions overlap heavily. Net queries gained by gating, counted
    against BEAM's actual 1:9 abstention:answerable balance:

        floor 0.450   gates  4 abstention   loses 11 answerable    -7
        floor 0.490   gates 10 abstention   loses 17 answerable    -7
        floor 0.510   gates 14 abstention   loses 27 answerable   -13
        floor 0.550   gates 17 abstention   loses 81 answerable   -64

    Best threshold is 0.000 — i.e. no gate. And that is an UPPER bound:
    it credits every gated abstention with a pass and charges every gated
    answerable with a failure, both optimistic for the gate.

    The trap worth remembering: optimising the CLASS-BALANCED separation
    rate makes 0.510 look like a +55% win ("gates 70% of abstention, costs
    15% of answerable"). At 1 abstention per 9 answerable that same
    threshold is -13 queries. Rate-based objectives overstate a minority-
    class gate by the class ratio; always optimise the count you actually
    get scored on.

    Not refuted, and untested here: passing confidence to the answerer as a
    SIGNAL rather than a gate (costs nothing on answerable because context
    is preserved), and whether cross-encoder scores separate the classes
    more sharply than bi-encoder cosine.
    """

    name = "yantrikdb-floor"
    description = (
        "YantrikDB engine recall with a confidence gate: min_score_ratio "
        "trims the tail, and an absolute score floor returns NO context at "
        "all when nothing clears it — an explicit 'I have nothing' signal "
        "for abstention questions. Still zero LLM calls in the memory layer."
    )
    provider = "yantrikdb"
    variant = "floor"

    def retrieve(
        self,
        query: str,
        k: int = _TOP_K,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        db = self._db_for(user_id)
        hits = db.recall(
            query=query,
            top_k=k,
            namespace=None if self._per_unit else self._namespace(user_id),
            skip_reinforce=True,
            min_score_ratio=_FLOOR_RATIO,
        )
        top = hits[0]["score"] if hits else 0.0
        gated = bool(hits) and top < _FLOOR_ABS
        if gated:
            hits = []
        raw = {
            "results": [{"rid": h.get("rid"), "score": h.get("score")} for h in hits],
            "top_score": top,
            "floor_abs": _FLOOR_ABS,
            "gated_to_empty": gated,
        }
        return self._to_documents(hits, user_id), raw


class YantrikDBCognitiveMemoryProvider(YantrikDBTemporalMemoryProvider):
    """Uses the engine as a MEMORY SYSTEM, not a chunk index.

    Everything before this arm exercised exactly one engine surface:
    `recall`. That reproduces a documented mistake — the 2026-06 benchmark
    harness "flat-dumped sessions, under-using YDB again" — and it is the
    real asymmetry hiding inside AMB's "rag mode": Hindsight's published
    runs spend minutes-to-hours of LLM extraction per split BUILDING their
    memory before retrieval ever runs (816s for 500K, 92min for 10M, from
    their own result files), while our ingest was chunk+embed+index. The
    comparison was constructed memory vs raw chunks.

    YantrikDB's construction machinery is engine-native (no LLM), so using
    it keeps the zero-LLM-ingest claim intact:

    * INGEST: turn-aware dated chunks (arm B), then `scan_conflicts()` —
      the engine's contradiction detector runs once per bank, and open
      conflicts are cached with each rid's text and date.
      `think()`-consolidation is deliberately NOT enabled in this arm:
      merging near-duplicate chunks has a plausible harm path for
      information_extraction, and it deserves its own measured arm rather
      than riding along unattributed.

    * RETRIEVE: chronological dated presentation (arm D), plus CONFLICT
      SURFACING — when a returned memory participates in an open conflict,
      the counterpart memory is injected (dated) and a factual note names
      the pair. BEAM's contradiction_resolution rubric wants exactly this
      ("state that there is conflicting information, quote BOTH
      statements"), and the dated pair also serves knowledge_update, where
      the answerer's rule is "prefer the more recent". The note reports
      what the engine detected; no text is generated.

    Read against arm D: the delta is conflict machinery alone.
    """

    name = "yantrikdb-cognitive"
    description = (
        "YantrikDB used as a cognitive memory system: turn-aware dated "
        "ingest, engine-native conflict detection at ingest "
        "(scan_conflicts), chronological dated presentation, and conflict "
        "counterpart injection at retrieval. Still zero LLM calls and zero "
        "network in the memory layer."
    )
    provider = "yantrikdb"
    variant = "cognitive"

    _MAX_CONFLICT_NOTES = 3

    def __init__(self):
        super().__init__()
        # unit key -> {rid: (text, created_at)} — the binding has no point
        # read, and retrieval must be able to quote a conflict counterpart
        # that recall did not return.
        self._texts: dict[str, dict[str, tuple[str, float | None]]] = {}
        # unit key -> list of open-conflict dicts, cached after the scan.
        self._conflicts: dict[str, list[dict]] = {}

    def _key(self, user_id: str | None) -> str:
        return user_id if (self._per_unit and user_id) else ""

    def ingest(self, documents: list[Document]) -> None:
        touched: set[str] = set()
        for doc in documents:
            db = self._db_for(doc.user_id)
            key = self._key(doc.user_id)
            cache = self._texts.setdefault(key, {})
            created_at = _iso_to_epoch(doc.timestamp)
            namespace = self._namespace(doc.user_id)
            for idx, chunk in enumerate(_turn_aware_chunks(doc.content, _CHUNK_TOKENS)):
                rid = db.record(
                    chunk,
                    memory_type="episodic",
                    metadata={"doc_id": doc.id, "chunk_idx": idx},
                    namespace=namespace,
                    created_at=created_at,
                )
                cache[rid] = (chunk, created_at)
            touched.add(key)

        for key in touched:
            db = self._dbs[key]
            found = db.scan_conflicts()
            open_conflicts = db.get_conflicts(status="open", limit=500)
            self._conflicts[key] = open_conflicts
            logger.info(
                "unit %s: %d records, conflict scan found %d, %d open",
                key or "(shared)", len(self._texts.get(key, {})),
                len(found), len(open_conflicts),
            )

    @staticmethod
    def _fmt_date(ts: float | None) -> str:
        return datetime.fromtimestamp(ts).strftime("%B %d, %Y") if ts else "undated"

    def retrieve(
        self,
        query: str,
        k: int = _TOP_K,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        hits = self._recall(query, k, user_id)
        docs = self._to_documents(hits, user_id)  # chronological + dated

        key = self._key(user_id)
        texts = self._texts.get(key, {})
        returned = {h.get("rid") for h in hits}
        notes: list[str] = []
        injected: list[Document] = []
        seen_pairs: set[frozenset] = set()
        for c in self._conflicts.get(key, []):
            a, b = c.get("memory_a"), c.get("memory_b")
            pair = frozenset((a, b))
            if pair in seen_pairs or not (a in returned or b in returned):
                continue
            if a not in texts or b not in texts:
                continue  # e.g. --skip-ingestion re-run without the cache
            seen_pairs.add(pair)
            (ta, da), (tb, db_) = texts[a], texts[b]
            # Chronological within the pair, matching the presentation order.
            if (da or 0) > (db_ or 0):
                (a, ta, da), (b, tb, db_) = (b, tb, db_), (a, ta, da)
            for rid, txt, ts in ((a, ta, da), (b, tb, db_)):
                if rid not in returned:
                    injected.append(Document(
                        id=f"conflict-counterpart-{rid[:8]}",
                        content=f"[{self._fmt_date(ts)}] {txt}",
                        user_id=user_id,
                    ))
            notes.append(
                f"The memory dated {self._fmt_date(da)} and the memory dated "
                f"{self._fmt_date(db_)} make conflicting statements"
                + (f" about {c['entity']}" if c.get("entity") else "")
                + f" ({c.get('conflict_type', 'conflict')}: "
                + f"{c.get('detection_reason', 'detected by conflict scan')})."
            )
            if len(notes) >= self._MAX_CONFLICT_NOTES:
                break

        if notes:
            docs = docs + injected + [Document(
                id="memory-system-conflict-note",
                content=(
                    "NOTE FROM THE MEMORY SYSTEM (engine conflict detection): "
                    + " ".join(notes)
                    + " If the question touches these, acknowledge the "
                    "conflict and quote both statements; for current-state "
                    "questions prefer the more recent one."
                ),
                user_id=user_id,
            )]

        raw = {
            "results": [{"rid": h.get("rid"), "score": h.get("score")} for h in hits],
            "conflict_notes": len(notes),
            "injected_counterparts": len(injected),
        }
        return docs, raw


class YantrikDBRerankMemoryProvider(YantrikDBMemoryProvider):
    name = "yantrikdb-rerank"
    description = (
        "YantrikDB engine recall (HNSW cosine + BM25 fusion) with "
        "cross-encoder reranking: pool of 50 candidates re-scored by "
        "ms-marco-MiniLM-L6-v2 reading query and chunk together. Still zero "
        "LLM calls and fully local in the memory layer."
    )
    provider = "yantrikdb"
    variant = "rerank"

    def retrieve(
        self,
        query: str,
        k: int = _TOP_K,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict | None]:
        from yantrikdb import rerank_hits

        pool = self._recall(query, max(_RERANK_POOL, k), user_id)
        hits = rerank_hits(query, pool, top_k=k)
        raw = [
            {"rid": h.get("rid"), "score": h.get("score"), "rerank_score": h.get("rerank_score")}
            for h in hits
        ]
        return self._to_documents(hits, user_id), {"results": raw}
