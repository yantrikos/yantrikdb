"""#117 — a reopened DB must not accept an embedder that did not build its vectors.

The engine already refused a same-dim-different-digest swap in its own
`set_embedder`. Python embedders never reached that path: both the constructor
and the pyo3 `set_embedder` stored the object on the *wrapper*, so the engine's
SearchState held whatever was auto-attached while every query and write was
encoded by an object it had never seen.

Measured on 0.15.3 before the fix: open accepted, recall returned similarity
**0.595** against vectors built by a different model, writes accepted. The
number is the point — cosine distance between two unrelated spaces still looks
like a weak-but-real match, so nothing in the response tells the caller the
answer is meaningless.

These tests run in BOTH directions deliberately. A gate that only ever refuses
is indistinguishable from a gate that refuses everything, and this suite has
been burned by one-directional checks before.
"""

import hashlib
import sqlite3

import pytest

from yantrikdb import YantrikDB

DIM = 64  # BUNDLED_EMBEDDER_DIM — same dim on both sides, so no dim check can catch it


class _Hostile:
    """A valid embedder (encode -> list[float]) from an unrelated space."""

    def encode(self, text):
        if not isinstance(text, str):
            return [self.encode(t) for t in text]
        h = hashlib.blake2b(text.encode("utf-8"), digest_size=DIM).digest()
        v = [(b / 255.0) - 0.5 for b in h]
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]


class _Declaring(_Hostile):
    """Same, but claims an identity the engine can check."""

    def __init__(self, fingerprint):
        self.fingerprint = fingerprint


@pytest.fixture
def built(tmp_path):
    """A store populated by the bundled embedder, plus its recorded digest."""
    db_path = str(tmp_path / "i117.db")
    db = YantrikDB(db_path=db_path, embedding_dim=DIM)
    db.record("the deploy key for node4 is id_deploy", namespace="n")
    db.record("the cluster runs on port 7438", namespace="n")
    db.close()

    conn = sqlite3.connect(db_path)
    try:
        digest = conn.execute(
            "SELECT value FROM meta WHERE key='embedder_digest'"
        ).fetchone()[0]
    finally:
        conn.close()
    return db_path, digest


def test_constructor_refuses_an_embedder_that_did_not_build_the_vectors(built):
    """The path the original report used."""
    db_path, _ = built
    with pytest.raises(RuntimeError) as e:
        YantrikDB(db_path=db_path, embedding_dim=DIM, embedder=_Hostile())
    assert "vectors were built by" in str(e.value)


def test_set_embedder_refuses_an_undeclared_embedder(built):
    db_path, _ = built
    db = YantrikDB(db_path=db_path, embedding_dim=DIM)
    try:
        with pytest.raises(RuntimeError):
            db.set_embedder(_Hostile())
    finally:
        db.close()


def test_set_embedder_refuses_a_wrong_declared_fingerprint(built):
    """Declaring an identity is not the same as declaring the RIGHT one."""
    db_path, _ = built
    db = YantrikDB(db_path=db_path, embedding_dim=DIM)
    try:
        with pytest.raises(RuntimeError):
            db.set_embedder(_Declaring("sha256:deadbeef"))
    finally:
        db.close()


def test_set_embedder_admits_the_matching_fingerprint(built):
    """The other direction: a gate that refuses everything is not a gate."""
    db_path, digest = built
    db = YantrikDB(db_path=db_path, embedding_dim=DIM)
    try:
        db.set_embedder(_Declaring(digest))  # must not raise
    finally:
        db.close()


def test_allow_unverified_embedder_is_the_documented_escape_hatch(built):
    db_path, _ = built
    db = YantrikDB(db_path=db_path, embedding_dim=DIM)
    try:
        db.set_embedder(_Hostile(), allow_unverified_embedder=True)  # must not raise
    finally:
        db.close()

    # and on the constructor, for parity
    db2 = YantrikDB(
        db_path=db_path,
        embedding_dim=DIM,
        embedder=_Hostile(),
        allow_unverified_embedder=True,
    )
    db2.close()


def test_a_store_with_no_recorded_identity_is_not_gated(tmp_path):
    """Nothing recorded means nothing to contradict — attaching must still work,
    or every BYO-embedder user is locked out of their own database."""
    db_path = str(tmp_path / "fresh.db")
    db = YantrikDB(db_path=db_path, embedding_dim=DIM, embedder=_Hostile())
    db.record("a record written by the caller's own embedder", namespace="n")
    hits = db.recall(query="the caller's own embedder", top_k=1, namespace="n")
    assert hits, "a BYO-embedder store must still be usable end to end"
    db.close()


# --- the recorded identity can itself be false -------------------------------
#
# Found on CT128, a 5,699-record production store, when 0.16.0 first opened it:
# `meta` recorded `potion-base-2M / dim 64` while every stored vector was 1536
# bytes — 384 f32, MiniLM. potion-base-2M does not emit 384-dim vectors, so it
# cannot have built them. The row was already wrong before the gate existed;
# the identity is stamped by whatever was ATTACHED at the first `embed()`, not
# by whatever produced vectors handed to `record()`.
#
# The gate refused the open on the strength of that row. Every MCP call then
# failed — lazily, after the service came up clean — and the error named a
# model provably not responsible. `detect_existing_dim` in the core has refused
# to trust this same row since v0.10 ("A STORED VECTOR IS THE AUTHORITY, NOT
# THE RECORDED IDENTITY", citing this very store), so the engine held two
# opposite rulings on one value.
#
# "Cannot verify" must stay a distinct state from "verified mismatch".


@pytest.fixture
def false_identity(tmp_path):
    """A store whose recorded embedder dim contradicts its stored vectors."""
    db_path = str(tmp_path / "false_identity.db")
    db = YantrikDB(db_path=db_path, embedding_dim=DIM)
    db.record("the deploy key for node4 is id_deploy", namespace="n")
    db.close()

    conn = sqlite3.connect(db_path)
    try:
        # The vectors stay DIM-wide; only the claim about them changes.
        conn.execute("UPDATE meta SET value='384' WHERE key='embedder_dim'")
        conn.commit()
        width = conn.execute(
            "SELECT length(embedding) FROM memories WHERE embedding IS NOT NULL LIMIT 1"
        ).fetchone()[0]
    finally:
        conn.close()
    assert width == DIM * 4, "fixture must leave the vectors untouched"
    return db_path


def test_a_recorded_identity_contradicting_the_vectors_does_not_brick_the_db(
    false_identity,
):
    """The CT128 regression: a false row must not be grounds for refusal."""
    with pytest.warns(UserWarning, match="recorded identity is wrong"):
        db = YantrikDB(db_path=false_identity, embedding_dim=DIM, embedder=_Hostile())
    try:
        db.record("written after attaching", namespace="n")
        assert db.recall("deploy key", namespace="n") is not None
    finally:
        db.close()


def test_the_warning_names_both_dimensions_so_it_can_be_repaired(false_identity):
    """A warning that does not say what is wrong is not actionable."""
    with pytest.warns(UserWarning) as rec:
        db = YantrikDB(db_path=false_identity, embedding_dim=DIM, embedder=_Hostile())
    db.close()
    msg = str(rec[0].message)
    assert "384" in msg and str(DIM) in msg
    assert "UNVERIFIED" in msg


def test_set_embedder_also_survives_a_false_recorded_identity(false_identity):
    db = YantrikDB(db_path=false_identity, embedding_dim=DIM)
    try:
        with pytest.warns(UserWarning):
            db.set_embedder(_Hostile())
    finally:
        db.close()


def test_a_consistent_identity_still_refuses(built):
    """The gate must not have been weakened for the case it was built for.

    `built` records an identity whose dim agrees with its vectors, so nothing
    about it is provably false and the refusal must stand.
    """
    db_path, _ = built
    db = YantrikDB(db_path=db_path, embedding_dim=DIM)
    try:
        with pytest.raises(RuntimeError) as e:
            db.set_embedder(_Hostile())
        assert "vectors were built by" in str(e.value)
    finally:
        db.close()
