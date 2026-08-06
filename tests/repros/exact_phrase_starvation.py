"""Repro of the hermes exact-phrase defect on the current engine
(15b141e wheel, bundled 64-dim embedder — their configuration).

Plant the coined phrase verbatim inside a LONG record among frame-shaped
distractors, query the phrase, and instrument every lane: where does the
exact-match record die? Suspects: (a) semantic sim at dim 64 too low to
rank, (b) KEYWORD_RESERVE_MIN_SIM=0.25 floor starving the rescue lane,
(c) FTS candidates outscored by fresh frame-noise."""

import tempfile
import os
import sys

from yantrikdb import YantrikDB

path = os.path.join(tempfile.mkdtemp(), "repro.db")
db = YantrikDB.with_default(path)  # bundled 64-dim embedder like hermes

# Frame-shaped distractors — the "~3,000 records share the shape" case.
for i in range(150):
    db.record_text(
        f"Agent run {i}: the pipeline reported a failure in subsystem {i % 9} "
        f"during the nightly build; retries resolved it and the class of error "
        f"was logged for later triage.",
        "episodic", 0.5, 0.0, 604800.0, {}, "default", 0.8, "general", "user", None,
    )

# The two target records: LONG, phrase verbatim, buried mid/late text.
long_a = (
    "Session retrospective, planning notes and follow-ups. " * 20
    + "NAMED CLASS ADOPTED by core under my name: MISATTRIBUTING FAILURE SURFACES "
    + "— 4 instances found this cycle, each one a case where the reported subsystem "
    + "was not the causal subsystem. "
    + "Additional trailing context about unrelated matters follows. " * 10
)
rid_a = db.record_text(long_a, "semantic", 0.7, 0.0, 604800.0, {}, "default", 0.8, "general", "user", None)
long_b = (
    "Weekly digest of engineering lessons. " * 15
    + "The misattributing failure surfaces named class keeps proving itself: "
    + "blame lands on the surface that reported, not the surface that caused. "
    + "More digest content continues here about other topics entirely. " * 12
)
rid_b = db.record_text(long_b, "semantic", 0.7, 0.0, 604800.0, {}, "default", 0.8, "general", "user", None)

QUERY = "misattributing failure surfaces named class"

print(f"targets: a={rid_a[:13]}... b={rid_b[:13]}...")
for expand in (False, True):
    hits = db.recall(query=QUERY, top_k=10, expand_entities=expand, skip_reinforce=True)
    ranks = {h["rid"]: i + 1 for i, h in enumerate(hits)}
    print(f"\nexpand_entities={expand}: rank_a={ranks.get(rid_a)} rank_b={ranks.get(rid_b)}")
    for h in hits[:5]:
        tag = "TARGET" if h["rid"] in (rid_a, rid_b) else "      "
        print(f"  {tag} sim={h['scores']['similarity']:.3f} score={h['score']:.3f} why={h['why_retrieved']} :: {h['text'][:60]}")

# Instrument the raw similarity between query and targets at dim 64 —
# is it under the 0.25 keyword-reserve floor? Under FTS 0.05?
import sqlite3
import struct

con = sqlite3.connect(path)
q = None
# embed the query via a scratch record (no public embed(); with_default has native embedder -> recall used it)
probe_rid = db.record_text(QUERY, "episodic", 0.5, 0.0, 604800.0, {}, "default", 0.8, "general", "user", None)
blobs = {}
for r, tag in ((probe_rid, "query"), (rid_a, "a"), (rid_b, "b")):
    (b,) = con.execute("SELECT embedding FROM memories WHERE rid=?", (r,)).fetchone()
    blobs[tag] = struct.unpack(f"<{len(b)//4}f", b)
con.close()

def cos(u, v):
    du = sum(x * x for x in u) ** 0.5
    dv = sum(x * x for x in v) ** 0.5
    return sum(a * b for a, b in zip(u, v)) / (du * dv)

print(f"\ndim={len(blobs['query'])}")
print(f"cos(query, target_a) = {cos(blobs['query'], blobs['a']):.4f}   (reserve floor 0.25, FTS floor 0.05)")
print(f"cos(query, target_b) = {cos(blobs['query'], blobs['b']):.4f}")
db.close()
print("REPRO_DONE")
