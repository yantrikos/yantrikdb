"""Adaptive topic resolution for thread recall.

``resolve_thread_topics`` turns a natural-language focus into the
``topic_rids`` list that :meth:`recall_thread_v2` unions into a thread. It
ranks the namespace's query-independent organizer topics by bounded
semantic recall and picks how many to keep from the shape of the query's
own score curve — never from a per-query constant tuned to any benchmark.

Parameter provenance (stated plainly so nobody has to reverse-engineer
the epistemics later):

- ``floor=12`` is DEV-CALIBRATED: it was chosen from a coverage sweep on
  the 40 burned event-ordering dev queries (BEAM 100k) after the frozen
  Stage A run failed with the historical fixed budget of 3. Its
  generalization beyond that dev set is UNCONFIRMED: a BEAM 500k holdout
  seal was invalidated before evaluation (helper contamination,
  disclosed and voided same-day), so no valid holdout evaluation exists
  yet — a future untouched, mutually sealed cohort has to supply it.
- ``cap=16`` is a hard design ceiling on rid fan-out: it bounds the SQL
  union width and the resulting thread size. Dev evidence (non-binding):
  max observed thread length was ~80 rows at 12 topics.

The knee rule is query-only: with ranked scores ``s1 >= s2 >= ... >= sn``
(n <= cap), a cut is searched only at or after ``floor`` — the largest
gap ``s_i - s_(i+1)`` for ``i`` in ``[floor, n-1]`` wins, equal largest
gaps cut LATER (coverage-conservative), and a flat curve above the floor
(largest gap < 1e-9) keeps all ``n``. When ``n <= floor`` everything is
kept and the knee diagnostics are ``None``.

Retrieval oversampling (BOUNDED): organizer topics share the store with
ordinary inference records, so ``recall(top_k=cap)`` can return fewer
than ``cap`` organizer hits. The resolver requests ``max(64, cap * 4)``
rows and, while fewer than ``cap`` organizer topics have surfaced AND
the store may hold more (returned == requested), doubles ``top_k`` and
re-queries — but never beyond ``candidate_cap`` (default 1024; request
sequence 64, 128, 256, 512, 1024). The hard stop keeps an organizer-poor
large store from widening toward the engine's recall maximum, and the
``candidate_cap_reached`` diagnostic makes the resulting beyond-cap
false-negative possibility auditable rather than silent.
``store_exhausted`` records that the store ran out before ``cap`` was
reached — a valid outcome, not an error.
"""

from __future__ import annotations

import math
from typing import Any

# Hard ceiling on both ``cap`` and ``floor`` — bounds rid fan-out.
MAX_TOPIC_CAP = 16
# Two adjacent scores closer than this carry no cut evidence.
FLAT_CURVE_EPSILON = 1e-9
# First oversampled request; doubled until enough organizer hits surface.
INITIAL_OVERSAMPLE_TOP_K = 64
# Hard stop for oversampling growth (candidate_cap upper bound stays
# under the engine's recall maximum).
DEFAULT_CANDIDATE_CAP = 1024
MAX_CANDIDATE_CAP = 10_000


def _validated_budget(name: str, value: Any, upper: int) -> int:
    # bool is an int subclass; a caller passing True/False is a bug, not 1/0.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if not 1 <= value <= upper:
        raise ValueError(f"{name} must be in [1, {upper}], got {value}")
    return value


def resolve_thread_topics(
    db: Any,
    focus: str,
    namespace: str = "default",
    *,
    cap: int = 16,
    floor: int = 12,
    candidate_cap: int = DEFAULT_CANDIDATE_CAP,
) -> dict[str, Any]:
    """Rank organizer topics for ``focus`` and cut the list at its knee.

    Returns a dict with ``selected_rids`` (what to pass as
    ``topic_rids``) plus the full audit surface: ``ranked`` is the entire
    pre-cut list so the knee is recomputable from the return value alone.

    Raises ``TypeError``/``ValueError`` for invalid budgets, a
    ``ValueError`` naming the namespace when no organizer topic exists in
    it, and ``ValueError`` for hits whose scores are missing or
    non-finite (a malformed score cannot be ranked, and silently dropping
    it would move the knee).
    """
    cap = _validated_budget("cap", cap, MAX_TOPIC_CAP)
    floor = _validated_budget("floor", floor, cap)
    if isinstance(candidate_cap, bool) or not isinstance(candidate_cap, int):
        raise TypeError(
            f"candidate_cap must be an int, got {type(candidate_cap).__name__}"
        )
    if not cap <= candidate_cap <= MAX_CANDIDATE_CAP:
        raise ValueError(
            f"candidate_cap must be in [{cap}, {MAX_CANDIDATE_CAP}], "
            f"got {candidate_cap}"
        )

    ranked: list[dict[str, Any]] = []
    seen_rids: set[str] = set()
    top_k = min(max(INITIAL_OVERSAMPLE_TOP_K, cap * 4), candidate_cap)
    recall_rounds = 0
    store_exhausted = False
    candidate_cap_reached = False
    while True:
        hits = db.recall(
            query=focus,
            top_k=top_k,
            namespace=namespace,
            source="inference",
            include_consolidated=True,
            skip_reinforce=True,
        )
        recall_rounds += 1
        ranked.clear()
        seen_rids.clear()
        for hit in hits:
            metadata = hit.get("metadata") or {}
            if metadata.get("organizer_kind") != "query_independent_topic":
                continue
            rid = str(hit.get("rid") or "")
            if not rid or rid in seen_rids:
                continue
            score = hit.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise ValueError(
                    f"organizer hit {rid} has a non-numeric score: {score!r}"
                )
            score = float(score)
            if not math.isfinite(score):
                raise ValueError(
                    f"organizer hit {rid} has a non-finite score: {score!r}"
                )
            seen_rids.add(rid)
            ranked.append(
                {
                    "rid": rid,
                    "label": metadata.get("organizer_label"),
                    "score": score,
                }
            )
        if len(ranked) >= cap:
            break
        if len(hits) < top_k:
            # The store returned everything it has; fewer organizer
            # topics than ``cap`` exist. Valid outcome.
            store_exhausted = True
            break
        if top_k >= candidate_cap:
            # Hard stop: more rows may exist beyond candidate_cap, so an
            # organizer topic could in principle be hiding out there — a
            # possible false negative, flagged for audit, never widened
            # toward the engine maximum.
            candidate_cap_reached = True
            break
        top_k = min(top_k * 2, candidate_cap)

    if not ranked:
        raise ValueError(
            f"no query-independent organizer topics found in namespace "
            f"{namespace!r}; run the organizer before thread resolution"
        )

    # recall returns score-descending order; make the contract explicit and
    # deterministic under ties (score desc, then rid asc).
    ranked.sort(key=lambda r: (-r["score"], r["rid"]))
    ranked = ranked[:cap]
    n = len(ranked)

    cut_index = n
    cut_score: float | None = None
    largest_gap: float | None = None
    flat_curve = False
    if n > floor:
        best_gap = -1.0
        best_i = None
        for i in range(floor, n):  # cut candidates: keep ranked[:i]
            gap = ranked[i - 1]["score"] - ranked[i]["score"]
            # ``>=`` so equal largest gaps cut LATER (coverage-conservative).
            if gap >= best_gap:
                best_gap = gap
                best_i = i
        if best_gap < FLAT_CURVE_EPSILON:
            # No score evidence for a cut anywhere at/after the floor:
            # keep everything up to the cap.
            flat_curve = True
        else:
            assert best_i is not None
            cut_index = best_i
            cut_score = ranked[best_i - 1]["score"]
            largest_gap = best_gap

    selected = ranked[:cut_index]
    return {
        "selected_rids": [r["rid"] for r in selected],
        "selected_count": cut_index,
        "cut_index": cut_index,
        "cut_score": cut_score,
        "largest_gap": largest_gap,
        "ranked": ranked,
        "cap": cap,
        "floor": floor,
        "flat_curve": flat_curve,
        "candidate_cap": candidate_cap,
        "recall_rounds": recall_rounds,
        "store_exhausted": store_exhausted,
        "candidate_cap_reached": candidate_cap_reached,
    }
