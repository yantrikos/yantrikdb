"""refresh_embeddings() on a store whose only embedder is a Python one.

Replicated rows arrive without vectors, and refresh_embeddings() is the repair
that fills them in. The engine's version can only use its own embedder, so on
a follower whose embedder was attached with set_embedder(obj) it raised "No
embedder configured" and the rows could never be repaired. The binding now
re-encodes them with the attached embedder.
"""

import os
import random

from yantrikdb import YantrikDB

DIM = 32  # not 64/256: those dims auto-attach a bundled engine-native embedder


class Emb:
    def __init__(self):
        self.calls = 0

    def encode(self, text):
        self.calls += 1
        r = random.Random(hash(text) & 0xFFFFFFFF)
        v = [r.random() for _ in range(DIM)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]


def test_a_python_embedder_follower_repairs_replicated_rows(tmp_path):
    leader = YantrikDB(os.path.join(str(tmp_path), "leader.db"), DIM)
    follower = YantrikDB(os.path.join(str(tmp_path), "follower.db"), DIM)
    try:
        leader.set_embedder(Emb(), allow_unverified_embedder=True)
        texts = [f"the harbour plan, note {i}, about berth {i % 4}" for i in range(12)]
        for t in texts:
            leader.record(t)
        emb = Emb()
        follower.set_embedder(emb, allow_unverified_embedder=True)
        follower.apply_ops(leader.extract_ops_since(limit=1000))

        before = follower.refresh_embeddings(dry_run=True)
        assert before["unusable_found"] >= len(texts), before

        done = follower.refresh_embeddings(dry_run=False)
        assert done["errors"] == [], done["errors"]
        assert done["refreshed"] == before["unusable_found"], done
        assert emb.calls >= len(texts), "the attached embedder made the vectors"

        after = follower.refresh_embeddings(dry_run=True)
        assert after["unusable_found"] == 0, after
        # The repaired vectors are the attached embedder's: a text's own
        # vector finds it first, at similarity 1.
        hit = follower.recall(query_embedding=Emb().encode(texts[5]), top_k=5, skip_reinforce=True)
        assert hit[0]["text"] == texts[5]
        assert abs(hit[0]["scores"]["similarity"] - 1.0) < 1e-5
    finally:
        leader.close()
        follower.close()
