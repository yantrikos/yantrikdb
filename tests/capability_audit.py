"""CAPABILITY AUDIT — is the temporal/ordering/abstention weakness REAL?

BEAM says these categories score 0.30-0.50. Five treatment arms failed and
the forensics blamed question ambiguity, wrong golds, and a strict rubric.
That excuse is only credible if the capability itself is sound. This test
removes every excuse: a synthetic corpus where I control every fact, with
questions that have exactly ONE defensible answer, scored MECHANICALLY
(no LLM judge anywhere).

It measures RETRIEVAL — the memory system's actual job — not answering.
If the right records come back in the right order, the memory works and
any downstream failure is the reader's. If they do not, we have a real
defect and this is the instrument to fix it against.

Run against the PUBLISHED wheel, not the source tree.
"""
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

from yantrikdb import YantrikDB

NS = "audit"


def ts(datestr: str) -> float:
    return datetime.strptime(datestr, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()


# ── The corpus: a project log with total ground truth ──
# Six phases, strictly ordered, unambiguous dates, each phase mentioned once
# as a phase-level statement plus two detail records.
PHASES = [
    ("2024-01-08", "kickoff", "Project Meridian kickoff: agreed the scope is a billing reconciliation service."),
    ("2024-02-12", "schema", "Phase two began: designed the ledger schema with double-entry invariants."),
    ("2024-03-18", "ingest", "Phase three began: built the bank-statement ingest pipeline."),
    ("2024-05-02", "matching", "Phase four began: implemented the transaction matching engine."),
    ("2024-06-14", "reporting", "Phase five began: added the monthly reconciliation reports."),
    ("2024-08-01", "launch", "Phase six began: launched Meridian to the finance team."),
]
DETAILS = [
    ("2024-01-15", "Meridian kickoff detail: the team is three engineers and one analyst."),
    ("2024-01-22", "Meridian kickoff detail: chose PostgreSQL for the ledger store."),
    ("2024-02-19", "Ledger schema detail: entries table has account_id, amount_cents, direction."),
    ("2024-02-26", "Ledger schema detail: added a constraint that debits must equal credits."),
    ("2024-03-25", "Ingest detail: parses OFX and CSV statement formats."),
    ("2024-04-02", "Ingest detail: deduplicates by statement fingerprint hash."),
    ("2024-05-09", "Matching detail: fuzzy matches on amount within two cents and date within three days."),
    ("2024-05-16", "Matching detail: unmatched transactions go to a review queue."),
    ("2024-06-21", "Reporting detail: reports export to XLSX and PDF."),
    ("2024-06-28", "Reporting detail: month-end close report runs on the first business day."),
    ("2024-08-08", "Launch detail: finance team trained in a two-hour session."),
    ("2024-08-15", "Launch detail: first production close completed without manual adjustment."),
]
# A value that CHANGES over time — the succession case.
SUCCESSION = [
    ("2024-02-05", "The matching tolerance is set to 5 cents."),
    ("2024-04-20", "The matching tolerance is now 3 cents."),
    ("2024-07-11", "The matching tolerance is now 2 cents."),
]
# Facts DEFINITELY ABSENT from the corpus (abstention ground truth):
ABSENT_TOPICS = [
    "What is the Meridian mobile app release date?",
    "Which cloud provider hosts the Meridian payroll module?",
    "How many customers use the Meridian public API?",
]


def build(db):
    for date, tag, text in PHASES:
        db.record_text(text, memory_type="episodic", importance=0.7,
                       metadata={"phase": tag}, namespace=NS, created_at=ts(date))
    for date, text in DETAILS:
        db.record_text(text, memory_type="episodic", importance=0.5,
                       namespace=NS, created_at=ts(date))
    for date, text in SUCCESSION:
        db.record_text(text, memory_type="semantic", importance=0.8,
                       namespace=NS, created_at=ts(date))


def probe_ordering(db, results):
    """Can retrieval reconstruct the phase sequence?"""
    hits = db.recall(query="the phases of project Meridian in order", top_k=20, namespace=NS, expand_entities=False)
    phase_texts = {p[2]: i for i, p in enumerate(PHASES)}
    got = [(h.get("created_at"), h["text"]) for h in hits if h["text"] in phase_texts]
    found_idx = [phase_texts[t] for _, t in got]
    recalled = len(set(found_idx))
    chrono = sorted(got, key=lambda x: x[0])
    order_ok = [phase_texts[t] for _, t in chrono] == sorted([phase_texts[t] for _, t in chrono])
    results["ordering_phase_recall"] = f"{recalled}/6 phase records retrieved"
    results["ordering_chronological_when_sorted"] = order_ok
    results["ordering_PASS"] = recalled == 6 and order_ok


def probe_window(db, results):
    """Time-window retrieval: exactly the records inside a known window."""
    lo, hi = ts("2024-03-01"), ts("2024-05-31")
    expected = {t for d, _, t in PHASES if lo <= ts(d) <= hi}
    expected |= {t for d, t in DETAILS if lo <= ts(d) <= hi}
    expected |= {t for d, t in SUCCESSION if lo <= ts(d) <= hi}
    try:
        hits = db.recall(query="project work", top_k=50, namespace=NS, time_window=(lo, hi), expand_entities=False)
    except TypeError:
        results["window_PASS"] = "time_window param unsupported in this build"
        return
    got = {h["text"] for h in hits}
    inside = {t for t in got if t in expected}
    leaked = [t for t in got if t not in expected]
    results["window_recall"] = f"{len(inside)}/{len(expected)} in-window records"
    results["window_leaks_outside_window"] = len(leaked)
    results["window_PASS"] = len(inside) == len(expected) and not leaked


def probe_succession(db, results, phase="warm"):
    """Current-value: the LATEST tolerance must outrank the superseded ones.

    Run TWICE — once on a COLD store and once after unrelated queries have
    already run. Cold alone is not a test: the defect this pins was invisible
    cold and only appeared once an unrelated recall had reinforced some other
    record. A currency guarantee that holds only until someone else asks a
    question is not a guarantee.
    """
    hits = db.recall(query="what is the matching tolerance", top_k=5, namespace=NS, expand_entities=False)
    texts = [h["text"] for h in hits]
    current = SUCCESSION[-1][1]
    rank = next((i for i, t in enumerate(texts) if t == current), None)
    results[f"succession_current_rank_{phase}"] = rank
    results[f"succession_all_three_retrieved_{phase}"] = sum(
        1 for _, t in SUCCESSION if t in texts)
    results[f"_succession_ok_{phase}"] = rank == 0


def probe_abstention(db, results):
    """Absent facts must score MEASURABLY lower than present ones."""
    present = db.recall(query="what statement formats does ingest parse", top_k=5, namespace=NS, expand_entities=False)
    p_top = present[0].get("scores", {}).get("similarity") if present else 0
    gaps = []
    for q in ABSENT_TOPICS:
        hits = db.recall(query=q, top_k=5, namespace=NS, expand_entities=False)
        a_top = hits[0].get("scores", {}).get("similarity") if hits else 0
        gaps.append((q[:42], round(a_top, 3)))
    worst_absent = max(g[1] for g in gaps)
    results["abstention_present_similarity"] = round(p_top, 3)
    results["abstention_absent_similarities"] = gaps
    results["abstention_separation"] = round(p_top - worst_absent, 3)
    # A usable abstention signal needs the absent queries to sit clearly below.
    results["abstention_PASS"] = (p_top - worst_absent) > 0.10


def probe_interval(db, results):
    """Temporal arithmetic with UNAMBIGUOUS referents — retrieval half only:
    are both endpoint records retrievable and correctly dated?"""
    a = db.recall(query="Project Meridian kickoff scope agreed", top_k=5, namespace=NS, expand_entities=False)
    b = db.recall(query="launched Meridian to the finance team", top_k=5, namespace=NS, expand_entities=False)
    got_a = next((h for h in a if h["text"] == PHASES[0][2]), None)
    got_b = next((h for h in b if h["text"] == PHASES[-1][2]), None)
    if not (got_a and got_b):
        results["interval_PASS"] = False
        results["interval_note"] = "endpoint record(s) not retrieved"
        return
    days = round((got_b["created_at"] - got_a["created_at"]) / 86400)
    truth = round((ts(PHASES[-1][0]) - ts(PHASES[0][0])) / 86400)
    results["interval_days_from_retrieved_metadata"] = days
    results["interval_ground_truth_days"] = truth
    results["interval_PASS"] = days == truth


def fresh_store():
    db = YantrikDB.with_default(os.path.join(tempfile.mkdtemp(), "audit.db"))
    build(db)
    return db


def run(fn, results, *args):
    """Call a probe; a probe that raises must not kill the audit."""
    try:
        fn(*args)
    except Exception as e:
        results[fn.__name__ + "_ERROR"] = f"{type(e).__name__}: {e}"


def main():
    results = {}

    # COLD: a store nobody has queried yet.
    cold = fresh_store()
    run(probe_succession, results, cold, results, "cold")
    cold.close()

    # WARM: the same corpus, but after unrelated questions have been asked.
    db = fresh_store()
    for fn in (probe_ordering, probe_window):
        run(fn, results, db, results)
    run(probe_succession, results, db, results, "warm")
    for fn in (probe_abstention, probe_interval):
        run(fn, results, db, results)
    db.close()

    # One probe, two conditions: currency must survive other people's queries.
    results["succession_PASS"] = bool(
        results.get("_succession_ok_cold") and results.get("_succession_ok_warm"))
    passes = [k for k, v in results.items() if k.endswith("_PASS") and v is True]
    fails = [k for k, v in results.items() if k.endswith("_PASS") and v is not True]
    print(json.dumps(results, indent=1, default=str))
    print(f"\nCAPABILITY AUDIT: {len(passes)}/{len(passes)+len(fails)} probes PASS")
    if fails:
        print("FAILING:", ", ".join(fails))


if __name__ == "__main__":
    main()
