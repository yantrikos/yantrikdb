"""Bindings that run engine work release the GIL while they do it (#271).

The concurrency tests in `test_gil_release_concurrency.py` prove that a detached
call holds no borrow of the wrapper. They cannot prove the GIL is released: with
the GIL held, another thread never runs during the call, so it never sees a
borrow either, and those tests pass on a build with no `py.detach` at all.

This file proves the release itself. With the switch interval at 100 s the
interpreter never forces a GIL handoff during a test, so an observer thread can
run while a call is in progress only if the call gives the GIL up itself. That
half is deterministic: a call that holds the GIL is seen exactly 0 times, on any
runner. The other half is a matter of attempts: the observer wakes every half
millisecond, so a call is seen once one of its detached stretches overlaps a
wake-up. Calls that spend tens of microseconds or more detached are seen within
a few hundred attempts; a detached stretch of a microsecond or two (a call that
fails at once, inside the detach) can be missed, and such a call is not
witnessed here. Controls keep the witness honest: a no-op must be seen 0 times,
a `sleep` at least once.

The "releases" half needs a second CPU. With one usable CPU, a detached stretch
of a few microseconds ends before the scheduler gets around to the observer,
and with the switch interval at 100 s nothing forces a handoff, so short calls
are missed: under `taskset -c 8` on Linux, 7 of 53 witnesses failed. Those
tests skip there, while the held half and the controls still run. The same
misses happen when all but one CPU is busy. CI runs pytest serially on runners
with 3 or more vCPUs, so it is not affected.

Call the binding bare. Wrapping it in anything that sleeps or blocks (a retry
with backoff, a lock) releases the GIL on the wrapper's behalf and fakes a pass.
"""

import os
import random
import sys
import threading
import time

import pytest

from yantrikdb import YantrikDB
from yantrikdb.consolidate import consolidate

DIM = 32  # not 64/256: those dims auto-attach a bundled engine-native embedder


def _usable_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not on Windows or macOS
        return os.cpu_count() or 1


# For tests that need to SEE a release (the module docstring explains why).
needs_a_second_cpu = pytest.mark.skipif(
    _usable_cpus() < 2,
    reason="seeing a short GIL release needs a second CPU for the observer",
)


class Emb:
    def encode(self, text):
        r = random.Random(hash(text) & 0xFFFFFFFF)
        v = [r.random() for _ in range(DIM)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]


class BlockingEmb(Emb):
    """Blocks inside `encode()` once armed, until released.

    Not armed during `set_embedder()`'s own probe, so attaching it does not block.
    """

    def __init__(self):
        self.armed = False
        self.entered = threading.Event()
        self.release = threading.Event()

    def encode(self, text):
        if self.armed:
            self.entered.set()
            if not self.release.wait(30):
                raise TimeoutError("BlockingEmb was never released")
        return super().encode(text)


@pytest.fixture
def db(tmp_path):
    store = YantrikDB(os.path.join(str(tmp_path), "witness.db"), DIM)
    store.set_embedder(Emb(), allow_unverified_embedder=True)
    for i in range(300):
        store.record(f"record {i} about topic {i % 31} and thing {i % 7}")
    yield store
    store.close()


def _seen_during(call, attempts=2000, interval=100):
    """How many times an observer thread ran while `call()` was in progress.

    Stops at the first sighting, so a call that releases the GIL costs a few
    attempts; only one that holds it runs all of them. 0 after `attempts` calls
    means the call never gave the GIL up for long enough to be seen.

    The pause between calls is jittered: a fixed pause can fall into step with
    the observer's own sleep, so every call starts while the observer is still
    asleep and a short release is missed every time (seen on Linux runners).

    Between calls the main thread sleeps, outside the window, and that is not
    optional. CPython forces a handoff once a waiting thread has waited a full
    switch interval with no switch, so a loop of calls that each hold the GIL
    would be seen as releasing it once their total time passed `interval` (300
    `think()` calls on 0.23.4 did exactly that). The sleep hands the GIL over
    between calls and restarts that clock, so a false sighting needs ONE call
    that holds the GIL for the whole interval.
    """
    state = {"in_call": False, "seen": 0, "stop": False}
    jitter = random.Random(271)

    def observer():
        while not state["stop"]:
            if state["in_call"]:
                state["seen"] += 1
            time.sleep(0.0005)

    old = sys.getswitchinterval()
    sys.setswitchinterval(interval)
    t = threading.Thread(target=observer, daemon=True)
    try:
        t.start()
        for _ in range(attempts):
            state["in_call"] = True
            call()
            state["in_call"] = False
            if state["seen"]:
                break
            time.sleep(0.001 + 0.001 * jitter.random())
    finally:
        state["in_call"] = False
        state["stop"] = True
        t.join()
        sys.setswitchinterval(old)
    return state["seen"]


# ── the witness's own controls ───────────────────────────────────────────


def test_witness_sees_nothing_through_a_call_that_holds_the_gil():
    def busy():
        sum(range(2000))

    assert _seen_during(busy) == 0


def test_witness_sees_nothing_through_held_calls_longer_in_total_than_the_interval():
    # The flaw the between-call sleep fixes, reproduced fast: calls that each
    # hold the GIL for 10 ms, a 50 ms switch interval, 100 calls (about 1 s,
    # twenty intervals). Without the sleep CPython forces a handoff mid-loop
    # and the witness reports a release that never happened.
    def held_10ms():
        end = time.perf_counter() + 0.010
        while time.perf_counter() < end:
            pass

    assert _seen_during(held_10ms, attempts=100, interval=0.05) == 0


def test_witness_sees_a_call_that_releases_the_gil():
    assert _seen_during(lambda: time.sleep(0.002)) >= 1


def test_witness_sees_nothing_through_a_binding_that_holds_the_gil(db):
    # A binding that runs under the GIL, so the witness is checked on the real
    # extension too, not only on pure Python. has_embedder() reads two fields
    # and touches no store, so it has no reason ever to be detached.
    assert _seen_during(lambda: db.has_embedder()) == 0


# ── the bindings #271 has detached so far ───────────────────────────────


@needs_a_second_cpu
def test_recall_with_an_embedding_releases_the_gil(db):
    q = Emb().encode("topic 3")
    assert _seen_during(lambda: db.recall(query_embedding=q, top_k=5)) >= 1


@needs_a_second_cpu
def test_recall_with_a_python_embedder_releases_the_gil(db):
    # encode() runs with the GIL held (it is Python); the search after it must not.
    assert _seen_during(lambda: db.recall(query="topic 3", top_k=5)) >= 1


@needs_a_second_cpu
def test_recall_with_the_engine_embedder_releases_the_gil(tmp_path):
    # 64 dims: the bundled engine-native embedder is attached, no Python one.
    store = YantrikDB(os.path.join(str(tmp_path), "native.db"), 64)
    try:
        if not store.has_embedder():
            pytest.skip("this build has no bundled engine embedder")
        for i in range(50):
            store.record(f"native record {i} about topic {i % 7}")
        assert _seen_during(lambda: store.recall(query="topic 3", top_k=5)) >= 1
    finally:
        store.close()


@needs_a_second_cpu
def test_stats_releases_the_gil(db):
    assert _seen_during(lambda: db.stats()) >= 1


@needs_a_second_cpu
def test_think_releases_the_gil(db):
    assert _seen_during(lambda: db.think()) >= 1


@needs_a_second_cpu
def test_scan_conflicts_releases_the_gil(db):
    assert _seen_during(lambda: db.scan_conflicts()) >= 1


@needs_a_second_cpu
def test_consolidate_releases_the_gil(db):
    assert _seen_during(lambda: consolidate(db)) >= 1


# ── no borrow is held across a Python embedder's encode() ────────────────


def _recall_in_a_thread_blocked_in_encode(db):
    emb = BlockingEmb()
    db.set_embedder(emb, allow_unverified_embedder=True)
    emb.armed = True
    out = {}

    def worker():
        try:
            out["results"] = db.recall(query="topic 3", top_k=5)
        except Exception as e:  # noqa: BLE001 - every failure is the finding
            out["error"] = repr(e)

    t = threading.Thread(target=worker)
    t.start()
    assert emb.entered.wait(10), "recall() never reached the embedder"
    return emb, t, out


def test_set_embedder_while_recall_is_inside_encode(db):
    # Before the shared snapshot, recall() held the wrapper's shared borrow
    # while the Python embedder ran, so this set_embedder() was refused with
    # "Already borrowed" for as long as encode() took.
    emb, t, out = _recall_in_a_thread_blocked_in_encode(db)
    try:
        db.set_embedder(Emb(), allow_unverified_embedder=True)
    finally:
        emb.release.set()
        t.join(30)
    assert not t.is_alive()
    assert "error" not in out, out.get("error")
    assert out["results"]


def test_close_with_nothing_in_flight_closes(db):
    # The documented contract: True when this call performed the exclusive
    # close. It returned False almost every time, because the idle compactor
    # held a strong engine ref across its wait (#271 review).
    assert db.close() is True
    assert db.close() is False  # already closed: idempotent, no second close


def test_close_while_recall_is_inside_encode(db):
    # recall() runs the Python embedder's encode() before it takes the
    # engine (#302 review 4), so a close() landing then is the exclusive
    # close (True), and the recall reports the store closed, as if close()
    # had come first. Before, the recall held the engine through encode():
    # close() returned False and the recall finished on that reference.
    emb, t, out = _recall_in_a_thread_blocked_in_encode(db)
    try:
        assert db.close() is True
    finally:
        emb.release.set()
        t.join(30)
    assert not t.is_alive()
    assert "closed" in out.get("error", ""), out


def test_close_under_write_load_keeps_other_threads_running(tmp_path):
    # close() stops the worker pool (joining a materializer mid-drain) and
    # closes SQLite (a WAL checkpoint). With the GIL held through both, a
    # Python thread beside it froze for the whole close: 15-20 ms idle, up
    # to 1.8 s under write load in the review. Detached, the thread runs.
    for attempt in range(5):
        store = YantrikDB(os.path.join(str(tmp_path), f"busy{attempt}.db"), DIM)
        store.set_embedder(Emb(), allow_unverified_embedder=True)
        for i in range(800):
            store.record(f"pending write {i} about topic {i % 13} and thing {i % 7}")
        gaps = []
        stop = threading.Event()

        def ticker():
            last = time.perf_counter()
            while not stop.is_set():
                time.sleep(0.002)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        t = threading.Thread(target=ticker)
        t.start()
        time.sleep(0.05)
        gaps.clear()
        started = time.perf_counter()
        store.close()
        took = time.perf_counter() - started
        time.sleep(0.05)
        stop.set()
        t.join()
        worst = max(gaps)
        if took < 0.06:
            continue  # too quick to tell a held GIL from a released one
        # Held, the ticker stalls for the whole close (99-100% of it on the
        # previous build); released, its gaps are scheduling noise, which on
        # a loaded macOS runner reached half of a 50 ms close.
        assert worst < took * 0.75, (
            f"the ticker stalled {worst * 1000:.0f} ms during a {took * 1000:.0f} ms close()"
        )
        return
    pytest.skip("close() never took long enough to measure on this machine")


# ── coverage: every public method is witnessed or listed as holding ─────
#
# A method that reaches SQLite and holds the GIL can stop every Python thread
# in the host for up to busy_timeout (5 s) while another thread, or another
# process, holds the write lock: duration is not the criterion, touching the
# store is. So every public method is either in RELEASES, with a witness test,
# or in HOLDS, with the reason it still holds. A new method fails
# test_every_public_method_is_classified until someone decides which.


def _fresh_pack_path(c):
    # seal_pack refuses an existing destination, so every call gets its own.
    c["packs"] = c.get("packs", 0) + 1
    return os.path.join(c["dir"], f"witness-{c['packs']}.ydbpack")


def _swallow(fn, *args, **kwargs):
    # For calls that fail on purpose (an unknown id): the failure comes back
    # from inside the detached call, so the GIL was still given up.
    try:
        fn(*args, **kwargs)
    except Exception:  # noqa: BLE001
        pass


RELEASES = {
    # detached before this sweep (their own tests are above)
    "recall": None,
    "think": None,
    "stats": None,
    "scan_conflicts": None,
    # stops the workers and closes SQLite detached; its witness is
    # test_close_under_write_load_keeps_other_threads_running above (one
    # call per store, so it cannot run in the loop below)
    "close": None,
    # memory
    "recall_as_of": lambda db, c: db.recall_as_of(time.time() + 60, query="topic 3"),
    "record_text": lambda db, c: db.record_text("witness record_text"),
    "recall_text": lambda db, c: db.recall_text("topic 3", 5),
    "recall_with_response": lambda db, c: db.recall_with_response(query="topic 3"),
    "recall_refine": lambda db, c: db.recall_refine(c["q"], refinement_text="thing 2"),
    "query": lambda db, c: db.query(query="topic 3"),
    "get": lambda db, c: db.get(c["rid"]),
    "get_memory": lambda db, c: db.get_memory(c["rid"]),
    "forget": lambda db, c: db.forget("no-such-rid"),
    "list_memories": lambda db, c: db.list_memories(limit=10),
    "list_records": lambda db, c: db.list_records(limit=10),
    "chain_head": lambda db, c: db.chain_head("default"),
    "record_batch": lambda db, c: db.record_batch([{"text": "witness batch"}]),
    "embed": None,  # engine-embedder path; its own test below
    # sync
    "maintenance_debt": lambda db, c: db.maintenance_debt(),
    "session_digest": lambda db, c: db.session_digest(),
    "what_changed_since": lambda db, c: db.what_changed_since(0.0),
    "draft_memories_from_summary": lambda db, c: db.draft_memories_from_summary(
        "We decided to ship on Friday."
    ),
    "record_turn": lambda db, c: db.record_turn("default", "user", "hello"),
    "recent_turns": lambda db, c: db.recent_turns("default"),
    "clear_turns": lambda db, c: db.clear_turns("default"),
    "knowledge_gaps": lambda db, c: db.knowledge_gaps(),
    "task_add": lambda db, c: db.task_add("default", "witness task"),
    "task_update": lambda db, c: db.task_update(c["task"], status="done"),
    "task_list": lambda db, c: db.task_list("default"),
    "task_get": lambda db, c: db.task_get(c["task"]),
    "task_delete": lambda db, c: db.task_delete("no-such-task"),
    # graph
    "relate": lambda db, c: db.relate("alice", "acme", "works_at"),
    "attach_claims": lambda db, c: db.attach_claims(
        c["rid"], [{"src": "alice", "rel_type": "works_at", "dst": "acme"}]
    ),
    "search_entities": lambda db, c: db.search_entities("ali"),
    # conflicts and triggers
    "get_conflicts": lambda db, c: db.get_conflicts(),
    "resolve_conflict": lambda db, c: _swallow(db.resolve_conflict, "no-such-conflict", "keep_a"),
    "get_pending_triggers": lambda db, c: db.get_pending_triggers(),
    "deliver_trigger": lambda db, c: db.deliver_trigger("no-such-trigger"),
    "acknowledge_trigger": lambda db, c: db.acknowledge_trigger("no-such-trigger"),
    "act_on_trigger": lambda db, c: db.act_on_trigger("no-such-trigger"),
    "dismiss_trigger": lambda db, c: db.dismiss_trigger("no-such-trigger"),
    # packs and identity
    "trusted_publishers": lambda db, c: db.trusted_publishers(),
    "embedder_identity": lambda db, c: db.embedder_identity(),
    # ── the long tail (#271 follow-up) ──
    # memory: reads, links, rollups, learning
    "decay": lambda db, c: db.decay(),
    "history": lambda db, c: db.history(c["rid"]),
    "detect_embedder_window": lambda db, c: db.detect_embedder_window(),
    "rechunk_long_records": None,  # needs a window; its own test below
    "record_with_links": lambda db, c: db.record_with_links("witness links", []),
    "record_with_links_partial": lambda db, c: db.record_with_links_partial(
        "witness partial links", []
    ),
    "link": lambda db, c: db.link(c["rid"], c["rid2"], "supports"),
    "unlink": lambda db, c: db.unlink(c["rid"], c["rid2"], "supports"),
    "linked_records": lambda db, c: db.linked_records(c["rid"]),
    "recall_with_links": lambda db, c: db.recall_with_links(query="topic 3"),
    "reify_supersedes_links": lambda db, c: db.reify_supersedes_links(),
    "audit_leak_candidates": lambda db, c: db.audit_leak_candidates(),
    "audit_synthesis_evidence": lambda db, c: db.audit_synthesis_evidence(),
    "recall_feedback": lambda db, c: db.recall_feedback(c["rid"], "relevant", query_text="topic 3"),
    "reject_recalled": lambda db, c: db.reject_recalled([c["rid"]]),
    "note_rollup_impression": lambda db, c: db.note_rollup_impression(c["rid"], "topic 3"),
    "note_rollup_impression_features": lambda db, c: db.note_rollup_impression_features(
        c["rid"], "topic 3"
    ),
    "note_rollup_expansion": lambda db, c: _swallow(
        db.note_rollup_expansion, "no-such-impression", [c["rid2"]]
    ),
    "note_rollup_expansion_features": lambda db, c: _swallow(
        db.note_rollup_expansion_features, "no-such-impression", [c["rid2"]], [0.5]
    ),
    "note_rollup_selection": lambda db, c: db.note_rollup_selection(
        "no-such-impression", c["rid2"]
    ),
    "finalize_rollup_outcome": lambda db, c: _swallow(
        db.finalize_rollup_outcome, "no-such-impression", [c["rid2"]]
    ),
    "rollup_outcome_report": lambda db, c: db.rollup_outcome_report(),
    "rollup_outcome_examples": lambda db, c: db.rollup_outcome_examples(),
    "rollup_membership_report": lambda db, c: db.rollup_membership_report(),
    "rollup_membership_examples": lambda db, c: db.rollup_membership_examples(),
    "run_learning": lambda db, c: db.run_learning(),
    "last_learning_report": lambda db, c: db.last_learning_report(),
    "learned_weights": lambda db, c: db.learned_weights(),
    # storage tiers and replication
    "archive": lambda db, c: db.archive(c["rid2"]),
    "hydrate": lambda db, c: db.hydrate(c["rid2"]),
    "evict": lambda db, c: db.evict(10_000),
    "extract_ops_since": lambda db, c: db.extract_ops_since(limit=20),
    "apply_ops": lambda db, c: db.apply_ops(c["ops"]),
    "get_peer_watermark": lambda db, c: db.get_peer_watermark("peer_1"),
    "set_peer_watermark": lambda db, c: db.set_peer_watermark("peer_1", b"\x00" * 16, "op-1"),
    # maintenance
    "rebuild_vec_index": lambda db, c: db.rebuild_vec_index(),
    "rebuild_graph_index": lambda db, c: db.rebuild_graph_index(),
    "repair_tool_call_artifacts": lambda db, c: db.repair_tool_call_artifacts(),
    "refresh_embeddings": lambda db, c: db.refresh_embeddings(),
    "recalibrate_unused_importance": lambda db, c: db.recalibrate_unused_importance(),
    "split_oversized_episodes": lambda db, c: db.split_oversized_episodes(),
    "auto_resolve_conflicts": lambda db, c: db.auto_resolve_conflicts(),
    "prune_triggers": lambda db, c: db.prune_triggers(),
    "skill_outcome_count": lambda db, c: db.skill_outcome_count(),
    "run_maintenance_cycle": lambda db, c: db.run_maintenance_cycle(dry_run=True),
    "last_maintenance_cycle": lambda db, c: db.last_maintenance_cycle(),
    "auto_relate": lambda db, c: db.auto_relate(),
    "maintain_source_turn_backfill": lambda db, c: db.maintain_source_turn_backfill(),
    "integrity_check": lambda db, c: db.integrity_check(),
    "oplog_plaintext_rows": lambda db, c: db.oplog_plaintext_rows(),
    # settings stored in the database
    "set_status_read_policy": lambda db, c: db.set_status_read_policy(True),
    "synthesis_fanout_cap": lambda db, c: db.synthesis_fanout_cap(),
    "set_synthesis_fanout_cap": lambda db, c: db.set_synthesis_fanout_cap(8),
    "set_provenance_gate_mode": lambda db, c: db.set_provenance_gate_mode("enforce"),
    "set_claim_chain_gate_mode": lambda db, c: db.set_claim_chain_gate_mode("shadow"),
    "set_foreign_sqlite_mode": lambda db, c: db.set_foreign_sqlite_mode("refuse"),
    # sessions, temporal, procedural, categories
    # A second start for the same client raises "active session already
    # exists" from inside the detached call; the first one starts it.
    "session_start": lambda db, c: _swallow(db.session_start, "default", "witness"),
    "session_end": lambda db, c: db.session_end(c["session"]),
    "active_session": lambda db, c: db.active_session(),
    "session_history": lambda db, c: db.session_history(),
    "session_abandon_stale": lambda db, c: db.session_abandon_stale(),
    "stale": lambda db, c: db.stale(),
    "upcoming": lambda db, c: db.upcoming(),
    # An unknown entity raises from inside the detached call.
    "entity_profile": lambda db, c: _swallow(db.entity_profile, "alice"),
    "relationship_depth": lambda db, c: _swallow(db.relationship_depth, "alice"),
    "surface_procedural": lambda db, c: db.surface_procedural(c["q"]),
    "record_procedural": lambda db, c: db.record_procedural("witness procedure"),
    "reinforce_procedural": lambda db, c: db.reinforce_procedural(c["rid"], 0.5),
    "procedural_stats": lambda db, c: db.procedural_stats(),
    "reclassify_conflict": lambda db, c: _swallow(
        db.reclassify_conflict, "no-such-conflict", "preference"
    ),
    "substitution_categories": lambda db, c: db.substitution_categories(),
    "substitution_members": lambda db, c: db.substitution_members("colors"),
    "learn_category_members": lambda db, c: db.learn_category_members(
        "colors", [("teal", 0.9)]
    ),
    "reset_category_to_seed": lambda db, c: _swallow(db.reset_category_to_seed, "colors"),
    # cognition: triggers, patterns, personality, conflicts, facets, threads
    "get_trigger_history": lambda db, c: db.get_trigger_history(),
    "get_patterns": lambda db, c: db.get_patterns(),
    "get_personality": lambda db, c: db.get_personality(),
    "derive_personality": lambda db, c: db.derive_personality(),
    "set_personality_trait": lambda db, c: db.set_personality_trait("openness", 0.5),
    "get_conflict": lambda db, c: db.get_conflict("no-such-conflict"),
    "dismiss_conflict": lambda db, c: db.dismiss_conflict("no-such-conflict"),
    "extract_standing_instructions": lambda db, c: db.extract_standing_instructions(dry_run=True),
    "recall_facets": lambda db, c: db.recall_facets(),
    "recall_thread": lambda db, c: db.recall_thread("default", ["alice"]),
    "recall_thread_v2": lambda db, c: db.recall_thread_v2("default", ["alice"]),
    "reextract_claims": lambda db, c: db.reextract_claims(dry_run=True),
    "reextract_entities": lambda db, c: db.reextract_entities(dry_run=True),
    "extraction_refusal_counts": lambda db, c: db.extraction_refusal_counts(),
    "extraction_refusals": lambda db, c: db.extraction_refusals(),
    "extraction_silver_recall": lambda db, c: db.extraction_silver_recall(),
    # graph and inspection
    "learned_relation_patterns": lambda db, c: db.learned_relation_patterns(),
    "forget_learned_relation_patterns": lambda db, c: db.forget_learned_relation_patterns(),
    "get_edges": lambda db, c: db.get_edges("alice"),
    "get_claims": lambda db, c: db.get_claims("alice"),
    "link_memory_entity": lambda db, c: db.link_memory_entity(c["rid"], "alice"),
    "backfill_memory_entities": lambda db, c: db.backfill_memory_entities(),
    "claims_for_memory": lambda db, c: db.claims_for_memory(c["rid"]),
    "revision_history": lambda db, c: db.revision_history(c["rid"]),
    "memory_entities": lambda db, c: db.memory_entities(c["rid"]),
    # packs that touch the store or the filesystem
    "seal_pack": lambda db, c: db.seal_pack(_fresh_pack_path(c), "witness", "1.0", "witness"),
    # The packs are sealed (and one mounted) by the fixture, so only the
    # binding under test runs inside the observed window: a helper that
    # sealed or mounted on first use put a detached call in the window and
    # faked a sighting. test_pack_witness_lambdas_release_nothing_themselves
    # checks the shape.
    "mount_pack": lambda db, c: _swallow(db.mount_pack, c["pack"], allow_unverified_embedder=True),
    "install_pack": lambda db, c: _swallow(db.install_pack, c["pack"]),
    "uninstall_pack": lambda db, c: db.uninstall_pack("no-such-pack@0"),
    "installed_packs": lambda db, c: db.installed_packs(),
    "remount_installed": lambda db, c: db.remount_installed(),
    # Reads /proc/self/maps on Linux and walks the loaded images on macOS;
    # on Windows it returns at once (see WITNESS_SKIPPED_ON).
    "foreign_sqlite_detected": lambda db, c: db.foreign_sqlite_detected(),
    # Static, but they open (and sign_pack writes) a pack's SQLite file.
    "read_pack_manifest": lambda db, c: YantrikDB.read_pack_manifest(c["pack"]),
    "sign_pack": lambda db, c: YantrikDB.sign_pack(c["pack"], c["secret"]),
    "trust_publisher": lambda db, c: db.trust_publisher(c["pubkey"], "witness"),
    "untrust_publisher": lambda db, c: db.untrust_publisher(c["pubkey"]),
    # Needs an engine-side embedder to reach the store; witnessed on the
    # engine-embedder store below. With only a Python embedder it fails
    # before touching SQLite.
    "adopt_embedder_identity": None,
    "recall_from_packs_for": lambda db, c: db.recall_from_packs_for(
        [c["mounted_pack_id"]], query="topic 3"
    ),
}

# Detached, but not witnessable offline: see the module docstring's limit.
UNWITNESSED = {
    "unmount_pack": (
        "detached because dropping a pack closes its SQLite connection, but "
        "only the first call has a pack to drop, and closing a read-only "
        "connection takes microseconds; mounting one again inside the window "
        "would be a detached call that fakes the sighting"
    ),
    "unmount_all_packs": "the same as unmount_pack",
    "set_embedder_named": (
        "the fetch (download on first use, load from the cache after) runs "
        "detached, but offline an unknown name fails within a microsecond or "
        "two inside the detach, and a real fetch needs the network or a "
        "cached model; the wait for exclusive access and stopping the workers "
        "run detached too, and only the swap and restarting the workers hold it"
    ),
    "reembed": (
        "the same: it resolves the model through the download registry, and "
        "offline an unknown name fails at once inside the detach"
    ),
    "convert_pack": (
        "static, detached for the re-embed; a real conversion needs the target "
        "model, which offline means a download that can only fail at once"
    ),
}

HOLDS = {
    # no store involved
    "with_default": "constructor",
    "generate_pack_keypair": "static helper: no store",
    "pubkey_of": "static helper: no store",
    "sign_bytes": "static helper: no store",
    "verify_bytes": "static helper: no store",
    # calls Python before it reaches the store, so the GIL stays held
    "set_embedder": (
        "its probe and identity check call the Python embedder; #295 "
        "released the wrapper's borrow there, not the GIL"
    ),
    # in flight in their own PRs
    "correct": "#296",
    "record": "#297",
    # pure in-memory reads: an atomic, a field or the mounted-pack list,
    # whose lock is never held across I/O. Detaching would add a GIL handoff
    # (milliseconds when another thread is CPU-bound) to a call that costs
    # microseconds and touches no store.
    "has_embedder": "reads two fields; the witness's held control above",
    "status_read_policy": "atomic load, no store",
    "provenance_gate_mode": "atomic load, no store",
    "claim_chain_gate_mode": "atomic load, no store",
    "foreign_sqlite_mode": "atomic load, no store",
    "pack_dir": "computed from the open path, no store",
    "mounted_packs": "in-memory mounted-pack list, brief lock, no store",
    "pack_context": "in-memory mounted-pack list, brief lock, no store",
    "pack_context_for": "in-memory mounted-pack list, brief lock, no store",
}


def test_every_public_method_is_classified():
    public = {
        n for n in dir(YantrikDB) if not n.startswith("_") and callable(getattr(YantrikDB, n))
    }
    lists = {"RELEASES": set(RELEASES), "UNWITNESSED": set(UNWITNESSED), "HOLDS": set(HOLDS)}
    names = list(lists)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            both = lists[a] & lists[b]
            assert not both, f"listed in {a} AND {b}: {sorted(both)}"
    classified = set().union(*lists.values())
    unclassified = public - classified
    assert not unclassified, (
        f"new public methods must go in RELEASES (with a witness), UNWITNESSED or HOLDS: "
        f"{sorted(unclassified)}"
    )
    stale = classified - public
    assert not stale, f"listed but no longer public: {sorted(stale)}"


@pytest.fixture
def small_db(tmp_path):
    store = YantrikDB(os.path.join(str(tmp_path), "coverage.db"), DIM)
    store.set_embedder(Emb(), allow_unverified_embedder=True)
    rids = [store.record(f"record {i} about topic {i % 11}") for i in range(60)]
    ctx = {"rid": rids[0], "rid2": rids[1], "q": Emb().encode("topic 3"), "dir": str(tmp_path)}
    ctx["task"] = store.task_add("default", "a task to update")
    ctx["session"] = store.session_start("default", "a session to end")
    ctx["ops"] = store.extract_ops_since(limit=5)
    ctx["secret"], ctx["pubkey"] = YantrikDB.generate_pack_keypair()
    # Packs, ready before any witness runs (see RELEASES): one to mount or
    # install, and a differently named one already mounted for recall.
    ctx["pack"] = _fresh_pack_path(ctx)
    store.seal_pack(ctx["pack"], "witness", "1.0", "witness")
    mounted = _fresh_pack_path(ctx)
    store.seal_pack(mounted, "witness-mounted", "1.0", "witness")
    ctx["mounted_pack_id"] = store.mount_pack(mounted, allow_unverified_embedder=True)
    yield store, ctx
    store.close()


# Detached everywhere, but with nothing to do on some platforms, where the
# call is too short to see.
WITNESS_SKIPPED_ON = {
    "foreign_sqlite_detected": ("win32", "on Windows the scan has no maps file to read"),
}


@needs_a_second_cpu
@pytest.mark.parametrize("name", sorted(n for n, call in RELEASES.items() if call is not None))
def test_binding_releases_the_gil(small_db, name):
    if name in WITNESS_SKIPPED_ON and sys.platform == WITNESS_SKIPPED_ON[name][0]:
        pytest.skip(WITNESS_SKIPPED_ON[name][1])
    db, ctx = small_db
    call = RELEASES[name]
    assert _seen_during(lambda: call(db, ctx)) >= 1, f"{name}() held the GIL"


class TruncEmb(Emb):
    """Sees only the first 200 characters, like a model with an input window."""

    def encode(self, text):
        return super().encode(text[:200])


@needs_a_second_cpu
def test_rechunk_long_records_releases_the_gil(tmp_path):
    # rechunk returns at once until a window is known, so give it one: the
    # Python embedder's encode() runs with the GIL held, the store work
    # between the embeds must not.
    store = YantrikDB(os.path.join(str(tmp_path), "rechunk.db"), DIM)
    try:
        store.set_embedder(TruncEmb(), allow_unverified_embedder=True)
        assert store.detect_embedder_window() is not None
        for i in range(40):
            store.record(f"long record {i} " + "and more words about topic %d " % i * 30)
        assert _seen_during(lambda: store.rechunk_long_records()) >= 1
    finally:
        store.close()


@needs_a_second_cpu
def test_embed_with_the_engine_embedder_releases_the_gil(tmp_path):
    store = YantrikDB(os.path.join(str(tmp_path), "native-embed.db"), 64)
    try:
        if not store.has_embedder():
            pytest.skip("this build has no bundled engine embedder")
        assert _seen_during(lambda: store.embed("some text to embed")) >= 1
    finally:
        store.close()


# ── a closed store says so before it judges the arguments ───────────────
#
# Before the sweep a method checked for a closed store as it took its borrow,
# before reading its arguments. A detached method reads them first, so each one
# here must take its engine handle before it parses anything.

class _Unconvertible:
    # py_to_json falls back to str(); this is the one value it cannot convert.
    def __str__(self):
        raise ValueError("not convertible")


CLOSED_FIRST = {
    "attach_claims": lambda db: db.attach_claims("some-rid", [{"rel_type": "works_at"}]),
    "record_with_links": lambda db: db.record_with_links("text", [{"link_type": "supports"}]),
    "record_with_links_partial": lambda db: db.record_with_links_partial(
        "text", [{"link_type": "supports"}]
    ),
    "linked_records": lambda db: db.linked_records("some-rid", "sideways"),
    "note_rollup_expansion_features": lambda db: db.note_rollup_expansion_features(
        "some-impression", ["a", "b"], [0.5]
    ),
    "_log_op": lambda db: db._log_op("witness", None, {"x": _Unconvertible()}),
    "session_start": lambda db: db.session_start("default", "c", {"x": _Unconvertible()}),
    "apply_ops": lambda db: db.apply_ops([{}]),
}


@pytest.mark.parametrize("name", sorted(CLOSED_FIRST))
def test_a_closed_store_is_reported_before_a_bad_argument(tmp_path, name):
    store = YantrikDB(os.path.join(str(tmp_path), "closed.db"), DIM)
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        CLOSED_FIRST[name](store)


# ── HOLDS entries that open PRs will change ──────────────────────────────
#
# correct and record are being detached in their own PRs. Until
# those merge they must still hold the GIL; once one does, this fails, and the
# method moves from HOLDS to RELEASES with a witness entry.

OWNED_BY_OPEN_PRS = {
    "correct": ("#296", lambda db, c: db.correct(c["rid"], "witness", new_importance=0.7)),
    "record": ("#297", lambda db, c: db.record("witness record")),
}


@pytest.mark.parametrize("name", sorted(OWNED_BY_OPEN_PRS))
def test_bindings_owned_by_open_prs_still_hold_the_gil(small_db, name):
    pr, call = OWNED_BY_OPEN_PRS[name]
    db, ctx = small_db
    assert name in HOLDS
    seen = _seen_during(lambda: call(db, ctx), attempts=200)
    assert seen == 0, (
        f"{name}() now releases the GIL: whoever merged {pr} should move it from "
        f"HOLDS to RELEASES with a witness entry, and drop it here"
    )


# ── the other shapes of bindings witnessed above ─────────────────────────


@needs_a_second_cpu
def test_recall_text_with_explain_releases_the_gil(small_db):
    db, _ = small_db
    assert _seen_during(lambda: db.recall_text("topic 3", 5, explain=True)) >= 1


ENGINE_EMBEDDER_RELEASES = {
    "adopt_embedder_identity": lambda db: db.adopt_embedder_identity(),
    "record_text": lambda db: db.record_text("witness record_text, engine embedder"),
    "record_batch": lambda db: db.record_batch([{"text": "witness batch, engine embedder"}]),
    "query": lambda db: db.query(query="engine embedder query"),
}


@pytest.fixture
def engine_embedder_db(tmp_path):
    # 64 dims: the engine attaches its bundled embedder, and no Python
    # embedder is set, so these calls take the engine-embedder route.
    store = YantrikDB(os.path.join(str(tmp_path), "native.db"), 64)
    if not store.has_embedder():
        store.close()
        pytest.skip("this build has no bundled engine embedder")
    for i in range(40):
        store.record_text(f"native record {i} about topic {i % 7}")
    yield store
    store.close()


@needs_a_second_cpu
@pytest.mark.parametrize("name", sorted(ENGINE_EMBEDDER_RELEASES))
def test_engine_embedder_paths_release_the_gil(engine_embedder_db, name):
    call = ENGINE_EMBEDDER_RELEASES[name]
    assert _seen_during(lambda: call(engine_embedder_db)) >= 1, f"{name}() held the GIL"


# The pack witnesses' lambdas, with the binding swapped for has_embedder(),
# which holds the GIL: anything else in the lambda that released it would be
# seen here. (The old first-use helpers sealed a pack inside the window.)
PACK_WITNESS_SHAPES = {
    "mount_pack": lambda db, c: (c["pack"], db.has_embedder()),
    "install_pack": lambda db, c: (c["pack"], db.has_embedder()),
    "recall_from_packs_for": lambda db, c: ([c["mounted_pack_id"]], db.has_embedder()),
}


@pytest.mark.parametrize("name", sorted(PACK_WITNESS_SHAPES))
def test_pack_witness_lambdas_release_nothing_themselves(small_db, name):
    db, ctx = small_db
    shape = PACK_WITNESS_SHAPES[name]
    assert _seen_during(lambda: shape(db, ctx), attempts=200) == 0


# ── coverage beyond the YantrikDB class ──────────────────────────────────
#
# The extension's module-level functions and TenantManager reach the store
# too. Same rule as the methods: witnessed in MODULE_RELEASES, or listed in
# MODULE_HOLDS with the reason.

import yantrikdb._yantrikdb_rust as _ext  # noqa: E402


MODULE_RELEASES = {
    "check_decay_triggers": lambda db, c: _ext.check_decay_triggers(db),
    "check_consolidation_triggers": lambda db, c: _ext.check_consolidation_triggers(db),
    "check_all_triggers": lambda db, c: _ext.check_all_triggers(db),
    "consolidate": None,  # witnessed above (test_consolidate_releases_the_gil)
    "find_consolidation_candidates": lambda db, c: _ext.find_consolidation_candidates(db),
    # Retires its sources on the first call; later calls are refused inside
    # the detached call, after reading them.
    "consolidate_cluster": lambda db, c: _swallow(
        _ext.consolidate_cluster, db, [c["rid"], c["rid2"]], "witness summary"
    ),
    "summarize_cluster": lambda db, c: _ext.summarize_cluster(
        db, [c["rid"], c["rid2"]], "witness synthesis", idempotency_key="witness-summary"
    ),
    "restore_from_consolidation": lambda db, c: _swallow(
        _ext.restore_from_consolidation, db, "no-such-summary"
    ),
    "record_synthesis": lambda db, c: _ext.record_synthesis(
        db, [c["rid"], c["rid2"]], "witness synthesis", "topic", "witness-synthesis"
    ),
    "TenantManager.get": lambda db, c: c["tenants"].get("acme"),
}

MODULE_HOLDS = {
    "_cosine_similarity": "pure computation over the vectors passed in, no store",
    "_extractive_summary": "pure computation over the dicts passed in, no store",
    "_find_clusters": "pure computation over the dicts passed in, no store",
    "tuning_fingerprint": "a compile-time constant",
    "TenantManager.register_tenant": "records a tenant's config in memory, no store",
    "TenantManager.discovered_tenants": "lists a directory, no SQLite",
}


def test_every_module_function_and_tenant_method_is_classified():
    public = {
        n
        for n in dir(_ext)
        if not n.startswith("__")
        and callable(getattr(_ext, n))
        and not isinstance(getattr(_ext, n), type)
    }
    public |= {
        f"TenantManager.{n}"
        for n in dir(_ext.TenantManager)
        if not n.startswith("_") and callable(getattr(_ext.TenantManager, n))
    }
    both = set(MODULE_RELEASES) & set(MODULE_HOLDS)
    assert not both, f"listed in MODULE_RELEASES AND MODULE_HOLDS: {sorted(both)}"
    classified = set(MODULE_RELEASES) | set(MODULE_HOLDS)
    assert not public - classified, (
        f"classify these in MODULE_RELEASES (with a witness) or MODULE_HOLDS: "
        f"{sorted(public - classified)}"
    )
    assert not classified - public, f"listed but gone: {sorted(classified - public)}"


@needs_a_second_cpu
@pytest.mark.parametrize(
    "name", sorted(n for n, call in MODULE_RELEASES.items() if call is not None)
)
def test_module_function_releases_the_gil(small_db, tmp_path, name):
    db, ctx = small_db
    ctx["tenants"] = _ext.TenantManager(os.path.join(str(tmp_path), "tenants"), DIM)
    call = MODULE_RELEASES[name]
    assert _seen_during(lambda: call(db, ctx)) >= 1, f"{name}() held the GIL"
