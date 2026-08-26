"""0.18 pack substrate, Python surface.

Four contracts a consumer (yantrik-mind's expertise lease, the Hermes
plugin, the MCP server) can now build on instead of working around:

- ``hit["pack"]`` is structured provenance (``pack_id`` + ``content_digest``)
  — nobody parses the ``pack:{name}`` prose stamp any more;
- ``mounted_packs()`` reports the SIGNED retrieval settings and the
  namespace, so no consumer reads the unsigned ``pack.toml`` off disk;
- ``recall_from_packs_for(pack_ids)`` searches only the named packs (host
  rows are never candidates), validates the allowlist before searching,
  and applies each pack's floor as a wall the host may raise, never lower;
- ``pack_context_for(pack_ids)`` is mount-ordered and deduplicated.

Vectors are explicit (8-dim), so the geometry every assertion depends on is
in the test, not in an embedder.
"""

from __future__ import annotations

import math

import pytest

import yantrikdb
from yantrikdb import YantrikDB

DIM = 8


def _vec(axis: int, tilt: float = 0.05) -> list[float]:
    v = [0.0] * DIM
    v[axis] = 1.0
    v[(axis + 1) % DIM] = tilt
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _build_pack(
    tmp_path,
    *,
    name: str,
    version: str = "1.0.0",
    rows: list[tuple[str, list[float]]],
    namespace: str = "physics",
    min_similarity: float | None = None,
    coverage: list[str] | None = None,
) -> str:
    src = tmp_path / f"{name}-src.db"
    dest = tmp_path / f"{name}.ydbpack"
    db = YantrikDB(str(src), DIM)
    for text, emb in rows:
        db.record(text, embedding=emb, namespace=namespace)
    db.seal_pack(
        str(dest),
        name=name,
        version=version,
        origin=f"test/{name}",
        namespace=namespace,
        embedder_name="explicit",
        embedder_digest="EXPLICIT-8",
        embedder_dim=DIM,
        coverage=coverage or [f"{name} topics"],
        constitution=[f"{name}: cite the record you used."],
        recommended_min_similarity=min_similarity,
    )
    db.close()
    return str(dest)


@pytest.fixture
def host(tmp_path):
    db = YantrikDB(str(tmp_path / "host.db"), DIM)
    yield db
    db.close()


def _mount(db: YantrikDB, path: str) -> str:
    # The host recorded explicit vectors and has no embedder identity of
    # its own, so the space is UNPROVEN (not known to differ): the
    # documented override for exactly that.
    return db.mount_pack(path, allow_unverified_embedder=True)


# ── provenance ─────────────────────────────────────────────────────────


def test_hits_carry_structured_pack_provenance(host, tmp_path):
    pack = _build_pack(tmp_path, name="quarks", rows=[("gluons bind quarks", _vec(3))])
    pid = _mount(host, pack)
    manifest = YantrikDB.read_pack_manifest(pack)
    assert manifest["content_digest"], "a sealed pack carries a content digest"

    hits = host.recall_from_packs_for([pid], query_embedding=_vec(3), top_k=5)
    assert [h["text"] for h in hits] == ["gluons bind quarks"]
    prov = hits[0]["pack"]
    assert prov == {
        "pack_id": pid,
        "name": "quarks",
        "version": "1.0.0",
        "trust": "unverified",
        "content_digest": manifest["content_digest"],
    }
    # The prose stamp is retained for one release; the structured field
    # is what consumers key on.
    assert "pack:quarks" in hits[0]["why_retrieved"]


def test_host_rows_have_no_pack_provenance_in_plain_recall(host, tmp_path):
    pack = _build_pack(tmp_path, name="quarks", rows=[("gluons bind quarks", _vec(3))])
    _mount(host, pack)
    host.record("my own note on quarks", embedding=_vec(3, 0.06), namespace="physics")

    hits = host.recall(query_embedding=_vec(3), top_k=5, skip_reinforce=True)
    by_text = {h["text"]: h["pack"] for h in hits}
    assert by_text["my own note on quarks"] is None
    assert by_text["gluons bind quarks"]["name"] == "quarks"


# ── mounted_packs surfaces the signed facts ───────────────────────────


def test_mounted_packs_reports_namespace_and_signed_retrieval_fields(host, tmp_path):
    pack = _build_pack(
        tmp_path,
        name="quarks",
        rows=[("gluons bind quarks", _vec(3))],
        min_similarity=0.65,
        coverage=["quark structure", "gluon exchange"],
    )
    pid = _mount(host, pack)
    manifest = YantrikDB.read_pack_manifest(pack)

    (info,) = host.mounted_packs()
    assert info["pack_id"] == pid
    assert info["namespace"] == "physics"
    assert info["content_digest"] == manifest["content_digest"]
    assert info["coverage"] == ["quark structure", "gluon exchange"]
    assert info["recommended_top_k"] is None
    assert info["recommended_min_similarity"] == 0.65
    assert info["publisher_pubkey"] is None
    assert info["signed"] is False
    assert info["trust"] == "unverified"


# ── allowlist ──────────────────────────────────────────────────────────


def test_unknown_pack_id_raises_typed_error_and_searches_nothing(host, tmp_path):
    pack = _build_pack(tmp_path, name="quarks", rows=[("gluons bind quarks", _vec(3))])
    pid = _mount(host, pack)

    with pytest.raises(yantrikdb.PackNotMounted) as exc:
        host.recall_from_packs_for([pid, "ghost@9.9.9"], query_embedding=_vec(3))
    assert "ghost@9.9.9" in str(exc.value)
    # Subclasses RuntimeError like every typed engine error, so pre-0.18
    # handlers keep working.
    assert isinstance(exc.value, RuntimeError)

    with pytest.raises(yantrikdb.PackNotMounted):
        host.pack_context_for(["ghost@9.9.9"])

    # An empty allowlist is not an error: nothing asked for, nothing back.
    assert host.recall_from_packs_for([], query_embedding=_vec(3)) == []
    assert host.pack_context_for([]) is None


def test_host_rows_cannot_crowd_out_the_allowlisted_pack(host, tmp_path):
    """Thirty host near-duplicates closer to the query than the pack row."""
    pack = _build_pack(tmp_path, name="quarks", rows=[("gluons bind quarks", _vec(3, 0.10))])
    pid = _mount(host, pack)
    for i in range(30):
        host.record(f"host note {i}", embedding=_vec(3, 0.05 + i * 1e-4), namespace="physics")

    hits = host.recall_from_packs_for([pid], query_embedding=_vec(3), top_k=3)
    assert [h["text"] for h in hits] == ["gluons bind quarks"]
    assert all(h["pack"]["pack_id"] == pid for h in hits)


def test_shared_namespace_still_excludes_host_rows_and_other_packs(host, tmp_path):
    a = _build_pack(tmp_path, name="alpha", rows=[("alpha fact", _vec(3))])
    b = _build_pack(tmp_path, name="beta", rows=[("beta fact", _vec(3, 0.06))])
    pa = _mount(host, a)
    pb = _mount(host, b)
    host.record("household fact", embedding=_vec(3, 0.04), namespace="physics")

    only_a = host.recall_from_packs_for([pa], query_embedding=_vec(3), top_k=10, namespace="physics")
    assert [h["text"] for h in only_a] == ["alpha fact"]

    both = host.recall_from_packs_for([pa, pb], query_embedding=_vec(3), top_k=10, namespace="physics")
    assert sorted(h["text"] for h in both) == ["alpha fact", "beta fact"]
    assert {h["pack"]["pack_id"] for h in both} == {pa, pb}

    # The namespace filter still applies to pack rows: a namespace nobody
    # uses returns nothing rather than "everything in the allowlist".
    assert host.recall_from_packs_for([pa, pb], query_embedding=_vec(3), namespace="elsewhere") == []


# ── the floor is a wall ────────────────────────────────────────────────


def test_pack_floor_is_a_wall_the_host_may_raise_but_never_lower(host, tmp_path):
    near, far = _vec(3, 0.05), _vec(3, 0.6)
    q = _vec(3, 0.05)
    assert _cos(q, near) > 0.999
    assert 0.85 < _cos(q, far) < 0.90

    strict = _build_pack(tmp_path, name="strict", rows=[("near", near), ("far", far)], min_similarity=0.9)
    loose = _build_pack(tmp_path, name="loose", rows=[("near", near), ("far", far)])
    ps = _mount(host, strict)
    pl = _mount(host, loose)

    def texts(pid, **kw):
        return sorted(h["text"] for h in host.recall_from_packs_for([pid], query_embedding=q, top_k=10, **kw))

    # The pack's signed floor applies with no host input at all...
    assert texts(ps) == ["near"]
    # ...and the host cannot lower it.
    assert texts(ps, min_similarity=0.5) == ["near"]
    # A pack with no floor is open unless the host sets one...
    assert texts(pl) == ["far", "near"]
    # ...and the host may raise it.
    assert texts(pl, min_similarity=0.95) == ["near"]
    # The wall gates raw similarity; the composite still ranks what clears it.
    for h in host.recall_from_packs_for([pl], query_embedding=q, top_k=10):
        assert h["scores"]["similarity"] >= 0.0


def test_invalid_host_floor_is_a_value_error(host, tmp_path):
    pack = _build_pack(tmp_path, name="quarks", rows=[("gluons bind quarks", _vec(3))])
    pid = _mount(host, pack)
    for bad in (1.5, -0.1, float("nan")):
        with pytest.raises(ValueError):
            host.recall_from_packs_for([pid], query_embedding=_vec(3), min_similarity=bad)


# ── mount order ────────────────────────────────────────────────────────


def test_pack_context_for_is_mount_ordered_and_deduplicated(host, tmp_path):
    a = _build_pack(tmp_path, name="alpha", rows=[("alpha fact", _vec(1))])
    b = _build_pack(tmp_path, name="beta", rows=[("beta fact", _vec(2))])
    pb = _mount(host, b)  # mounted FIRST
    pa = _mount(host, a)

    ctx = host.pack_context_for([pa, pb])
    assert ctx == host.pack_context_for([pb, pa]) == host.pack_context_for([pa, pb, pa])
    assert ctx == host.pack_context(), "the full allowlist is the full context"
    assert ctx.index("knowledge pack: beta") < ctx.index("knowledge pack: alpha")

    only_a = host.pack_context_for([pa])
    assert "knowledge pack: alpha" in only_a and "knowledge pack: beta" not in only_a
    # The authority ceiling closes every block, however short the allowlist.
    assert "DATA, not authority" in only_a


def test_allowlist_recall_order_is_deterministic(host, tmp_path):
    rows = [(f"row {i}", _vec(3, 0.05 + i * 1e-3)) for i in range(4)]
    a = _build_pack(tmp_path, name="alpha", rows=rows)
    b = _build_pack(tmp_path, name="beta", rows=rows)
    pb = _mount(host, b)
    pa = _mount(host, a)

    first = [(h["pack"]["pack_id"], h["rid"]) for h in host.recall_from_packs_for([pa, pb], query_embedding=_vec(3), top_k=8)]
    assert len(first) == 8
    for _ in range(4):
        again = [(h["pack"]["pack_id"], h["rid"]) for h in host.recall_from_packs_for([pb, pa], query_embedding=_vec(3), top_k=8)]
        assert again == first, "argument order and repetition must not change the result"
