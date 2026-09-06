"""Write-path cost probe: record 300 synthetic memories, then drain the materializer.

Run it against two installed engine versions and compare the two lines. It caught
the 0.21.0 regression where the lexicon wrote one autocommit per token
(drain 10.5 s -> 17.8 s on this probe; the BEAM capture took twice as long):

    python benchmarks/ingest_write_cost.py
"""
import sys, time, math, tempfile, os, random
import yantrikdb
from yantrikdb import YantrikDB
random.seed(7)
words = ["alpha","beta","gamma","delta","Alice","Moreau","Fennwick","Labs","Berlin","runs","works","the","of","in","Critically","Failed","project","release","version","0.19.0","2026","CT128","API","PR","team","plan","fix","next","done","start"]
texts = [" ".join(random.choice(words) for _ in range(120)) + "." for _ in range(300)]
d = tempfile.mkdtemp(); db = YantrikDB(os.path.join(d, "b.db"), 64)
def unit(s):
    v=[math.sin(s*(i+1)) for i in range(64)]; n=math.sqrt(sum(x*x for x in v)); return [x/n for x in v]
t0 = time.perf_counter()
for i, t in enumerate(texts): db.record(text=t, embedding=unit(i+1))
t1 = time.perf_counter()
for _ in range(40): db.think({"run_consolidation": False, "run_pattern_mining": False, "run_conflict_scan": False})
t2 = time.perf_counter()
print(f"engine {yantrikdb.__version__}: record 300 = {t1-t0:.1f}s, drain = {t2-t1:.1f}s, entities={db.stats()['entities']}")
db.close()
