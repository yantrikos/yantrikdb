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
    # close() mid-call drops the wrapper's handle and reports False, because
    # the in-flight recall() still holds the engine; it finishes on that
    # reference, and the engine closes when it goes.
    emb, t, out = _recall_in_a_thread_blocked_in_encode(db)
    try:
        assert db.close() is False
    finally:
        emb.release.set()
        t.join(30)
    assert not t.is_alive()
    assert "error" not in out, out.get("error")
    assert out["results"]


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
    # Raises here: the engine embeds the drafts itself and cannot reach a
    # Python embedder (pre-existing; #271 follow-up). The error comes back
    # from inside the detached call.
    "draft_memories_from_summary": lambda db, c: _swallow(
        db.draft_memories_from_summary, "We decided to ship on Friday."
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
    "attach_claims": lambda db, c: _swallow(
        db.attach_claims, c["rid"], [{"src": "alice", "rel_type": "works_at", "dst": "acme"}]
    ),
    "search_entities": lambda db, c: db.search_entities("ali"),
    # conflicts and triggers
    "get_conflicts": lambda db, c: db.get_conflicts(),
    "resolve_conflict": lambda db, c: _swallow(db.resolve_conflict, "no-such-conflict", "keep_a"),
    "get_pending_triggers": lambda db, c: db.get_pending_triggers(),
    "deliver_trigger": lambda db, c: _swallow(db.deliver_trigger, "no-such-trigger"),
    "acknowledge_trigger": lambda db, c: _swallow(db.acknowledge_trigger, "no-such-trigger"),
    "act_on_trigger": lambda db, c: _swallow(db.act_on_trigger, "no-such-trigger"),
    "dismiss_trigger": lambda db, c: _swallow(db.dismiss_trigger, "no-such-trigger"),
    # packs and identity
    "trusted_publishers": lambda db, c: db.trusted_publishers(),
    "embedder_identity": lambda db, c: db.embedder_identity(),
}

# Detached, but not witnessable offline: see the module docstring's limit.
UNWITNESSED = {
    "set_embedder_named": (
        "the fetch (download on first use, load from the cache after) runs "
        "detached, but offline an unknown name fails within a microsecond or "
        "two inside the detach, and a real fetch needs the network or a "
        "cached model; stopping the workers (under a millisecond), the swap "
        "and the respawn hold the GIL"
    ),
}

_FOLLOW_UP = "not yet detached: #271 follow-up"

HOLDS = {
    # no store involved
    "with_default": "constructor",
    "convert_pack": "static helper: no store",
    "generate_pack_keypair": "static helper: no store",
    "pubkey_of": "static helper: no store",
    "read_pack_manifest": "static helper: no store",
    "sign_bytes": "static helper: no store",
    "sign_pack": "static helper: no store",
    "verify_bytes": "static helper: no store",
    # exclusive access by design
    "close": "needs the wrapper exclusively; drops its handle, never waits on in-flight calls",
    # in flight in their own PRs
    "set_embedder": "#295 (and its identity check reads the store)",
    "correct": "#296",
    "record": "#297",
}
HOLDS.update(
    {
        name: _FOLLOW_UP
        for name in """
        active_session adopt_embedder_identity apply_ops archive audit_leak_candidates
        audit_synthesis_evidence auto_relate auto_resolve_conflicts backfill_memory_entities
        claim_chain_gate_mode claims_for_memory decay derive_personality detect_embedder_window
        dismiss_conflict entity_profile evict extract_ops_since extract_standing_instructions
        extraction_refusal_counts extraction_refusals extraction_silver_recall
        finalize_rollup_outcome foreign_sqlite_detected foreign_sqlite_mode
        forget_learned_relation_patterns get_claims get_conflict get_edges get_patterns
        get_peer_watermark get_personality get_trigger_history has_embedder history hydrate
        install_pack installed_packs integrity_check last_learning_report
        last_maintenance_cycle learn_category_members learned_relation_patterns
        learned_weights link link_memory_entity linked_records maintain_source_turn_backfill
        memory_entities mount_pack mounted_packs note_rollup_expansion
        note_rollup_expansion_features note_rollup_impression note_rollup_impression_features
        note_rollup_selection oplog_plaintext_rows pack_context pack_context_for pack_dir
        procedural_stats provenance_gate_mode prune_triggers rebuild_graph_index
        rebuild_vec_index recalibrate_unused_importance recall_facets recall_feedback
        recall_from_packs_for recall_thread recall_thread_v2 recall_with_links
        rechunk_long_records reclassify_conflict record_procedural record_with_links
        record_with_links_partial reembed reextract_claims reextract_entities
        refresh_embeddings reify_supersedes_links reinforce_procedural reject_recalled
        relationship_depth remount_installed repair_tool_call_artifacts
        reset_category_to_seed revision_history rollup_membership_examples
        rollup_membership_report rollup_outcome_examples rollup_outcome_report run_learning
        run_maintenance_cycle seal_pack session_abandon_stale session_end session_history
        session_start set_claim_chain_gate_mode set_foreign_sqlite_mode set_peer_watermark
        set_personality_trait set_provenance_gate_mode set_status_read_policy
        set_synthesis_fanout_cap skill_outcome_count split_oversized_episodes stale
        status_read_policy substitution_categories substitution_members surface_procedural
        synthesis_fanout_cap trust_publisher uninstall_pack unlink unmount_all_packs
        unmount_pack untrust_publisher upcoming
        """.split()
    }
)


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
    ctx = {"rid": rids[0], "q": Emb().encode("topic 3")}
    ctx["task"] = store.task_add("default", "a task to update")
    yield store, ctx
    store.close()


@needs_a_second_cpu
@pytest.mark.parametrize("name", sorted(n for n, call in RELEASES.items() if call is not None))
def test_binding_releases_the_gil(small_db, name):
    db, ctx = small_db
    call = RELEASES[name]
    assert _seen_during(lambda: call(db, ctx)) >= 1, f"{name}() held the GIL"


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

CLOSED_FIRST = {
    "attach_claims": lambda db: db.attach_claims("some-rid", [{"rel_type": "works_at"}]),
}


@pytest.mark.parametrize("name", sorted(CLOSED_FIRST))
def test_a_closed_store_is_reported_before_a_bad_argument(tmp_path, name):
    store = YantrikDB(os.path.join(str(tmp_path), "closed.db"), DIM)
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        CLOSED_FIRST[name](store)


# ── HOLDS entries that open PRs will change ──────────────────────────────
#
# set_embedder, correct and record are being detached in their own PRs. Until
# those merge they must still hold the GIL; once one does, this fails, and the
# method moves from HOLDS to RELEASES with a witness entry.

OWNED_BY_OPEN_PRS = {
    "set_embedder": ("#295", lambda db, c: db.set_embedder(Emb(), allow_unverified_embedder=True)),
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
