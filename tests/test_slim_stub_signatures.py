"""The slim build's stubs take the same keyword arguments as the real methods.

A wheel built with --no-default-features compiles the embedder-download paths
out and keeps stubs that raise RuntimeError with an actionable message. The
stubs once named their parameters `_src`, `_name` and so on, so a caller who
passed arguments by keyword got TypeError instead. On the default build the
same calls reach the real methods and fail for their own reasons (no such
file, no such model); never with TypeError.
"""

import os

import pytest

from yantrikdb import YantrikDB


def _keyword_calls(db, tmp_path):
    missing = os.path.join(str(tmp_path), "no-such.ydbpack")
    return {
        "convert_pack": lambda: YantrikDB.convert_pack(
            src=missing, dest=missing + ".out", embedder_name="no-such-model"
        ),
        "reembed": lambda: db.reembed(new_embedder_name="no-such-model"),
        "set_embedder_named": lambda: db.set_embedder_named(name="no-such-model"),
    }


@pytest.mark.parametrize("name", ["convert_pack", "reembed", "set_embedder_named"])
def test_keyword_arguments_reach_the_method(tmp_path, name):
    db = YantrikDB(os.path.join(str(tmp_path), "slim.db"), 32)
    try:
        call = _keyword_calls(db, tmp_path)[name]
        with pytest.raises(RuntimeError) as err:
            call()
        if "embedder-download" in str(err.value):
            assert "--no-default-features" in str(err.value)
    finally:
        db.close()
