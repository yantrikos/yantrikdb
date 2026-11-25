"""A record written through the Python wrapper must be chunked AT WRITE TIME.

The regression this guards (#276): with an embedder held on the Python side, the
wrapper routed to `record_with_idempotency(..., chunks: &[])`. The engine then
never minted windows — the tail of a long record sat in no vector until somebody
ran `rechunk_long_records()`. The suite stayed green the whole time, because every
existing chunk test wrote through `record_text()`, the path without a Python
embedder.

What is asserted here, in this order:
  1. a long write leaves chunk vectors in the store, with no backfill called;
  2. a query built from the record's TAIL finds it (vector lane only, so the
     lexical index cannot answer for it);
  3. writing the same text again adds windows for the new record and does not
     duplicate or overwrite the first record's windows.
"""

import hashlib

import pytest

from yantrikdb import YantrikDB

DIM = 32
WINDOW_LIMIT = 200


class TruncatingHashEmbedder:
    """Deterministic bag-of-words vectors, truncated like a real model.

    Truncation matters: `detect_embedder_window()` finds the window by watching
    where the vector stops changing as the text grows. An embedder that reads the
    whole string would report "no truncation" and the engine would never chunk.
    """

    def encode(self, text):
        text = text[:WINDOW_LIMIT]
        vec = [0.0] * DIM
        for word in text.split():
            h = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16)
            vec[h % DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


@pytest.fixture
def db(tmp_path):
    store = YantrikDB(str(tmp_path / "chunks.db"), DIM)
    store.set_embedder(TruncatingHashEmbedder(), allow_unverified_embedder=True)
    return store


HEAD = "landschaftsprotokoll wege bewaldung kamm wanderung abschnitt " * 60
TAIL = "quastenflosser zirbelkiefer klangschale nebelhorn amethystvogel"


def long_text():
    return HEAD + TAIL


def chunk_vectors(store):
    stats = store.stats()
    return stats["chunk_vectors"] if isinstance(stats, dict) else stats.chunk_vectors


def test_long_write_mints_chunks_without_a_backfill(db):
    window = db.detect_embedder_window()
    assert window, "the probe must find the truncating window"
    assert window <= WINDOW_LIMIT, f"unexpected window {window}"

    text = long_text()
    assert len(text) > 2 * window, "the text must overflow the window by more than one"

    assert chunk_vectors(db) == 0
    rid = db.record(text)

    # 1. windows exist straight after the write — no rechunk_long_records() call
    assert chunk_vectors(db) > 0, "the write path minted no window vectors (#276)"

    # 2. the TAIL is reachable through the vector lane alone
    emb = TruncatingHashEmbedder().encode(TAIL)
    hits = db.recall(query_embedding=emb, top_k=5)
    results = hits["results"] if isinstance(hits, dict) else hits
    assert rid in [str(h.get("rid")) for h in results], "tail not reachable after write"


def test_second_write_adds_its_own_windows_without_duplicates(db):
    db.detect_embedder_window()
    text = long_text()

    first = db.record(text)
    after_first = chunk_vectors(db)
    assert after_first > 0

    second = db.record(text)
    assert second != first
    # a second record carries its own window set: the count grows, nothing is lost
    assert chunk_vectors(db) == after_first * 2
