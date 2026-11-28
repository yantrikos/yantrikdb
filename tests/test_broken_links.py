"""broken_links() through the installed engine, not only in Rust.

link() then forget(): the active view returns nothing, the provenance view
returns the edge *and its status*.
"""
import pytest

yantrikdb = pytest.importorskip("yantrikdb")


def test_broken_links_returns_the_status(tmp_path):
    db = yantrikdb.YantrikDB(str(tmp_path / "store.db"), 8)
    a = db.record("value is nine", "semantic", 0.8, 0.0, 604800.0, {}, [0.1] * 8,
                  "default", 0.8, "general", "user")
    b = db.record("value is ten", "semantic", 0.6, 0.0, 604800.0, {}, [0.2] * 8,
                  "default", 0.8, "general", "user")
    db.link(a, b, "supersedes")
    db.forget(b)

    assert db.linked_records(a, "outbound", None) == []

    broken = db.broken_links(a, "outbound")
    assert len(broken) == 1, broken
    assert broken[0]["rid"] == b
    assert broken[0]["link_type"] == "supersedes"
    assert broken[0]["status"] == "broken_target_forgotten"
    assert broken[0]["direction"] == "outbound"
