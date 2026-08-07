# Try it in ten minutes

Two ways in. The first needs no GPU and proves the *measurement* is real,
which is the part worth being sceptical about. The second runs the
capability yourself.

---

## 1. Check the claim without running anything (5 min, CPU only)

```bash
git clone https://github.com/yantrikos/yantrikdb
cd yantrikdb
python -m venv .venv && .venv/bin/pip install yantrikdb
```

**Confirm the test set is sealed.** The briefs are produced by code, not
chosen by hand — so you can verify no test subject appears in training:

```python
import sys; sys.path.insert(0, "packs")
import compile as C
W = C.craft_module("motion-craft")

train, sealed = W.train_briefs(), W.holdout_briefs()
subj = lambda b: b.split("motion for ")[1].split(":")[0]
print(sorted({subj(b) for b in sealed}))          # 6 subjects
print({subj(b) for b in sealed} & {subj(b) for b in train})   # set() — disjoint
```

**Re-grade our published artifacts with our published grader.**

```python
from pathlib import Path
for f in sorted(Path("packs/samples/motion-craft").glob("*.html")):
    p, n, _ = W.grade_text(f.read_text(encoding="utf-8"))
    print(f"{p:>2}/{n}  {f.name}")
```

**Grade something we did not write.** Paste in output from any model —
GPT, Claude, your own — and the same function scores it:

```python
p, n, detail = W.grade_text(open("their-output.html").read())
for check, (ok, why) in detail.items():
    print("PASS" if ok else "FAIL", check, why)
```

The grader is [`motion-craft/craft.py`](motion-craft/craft.py): parsing
and arithmetic, no model in the loop, frozen before any arm ran. If a
check looks wrong, it is readable and you can name which one.

---

## 2. Run the capability (needs a 24 GB GPU)

```bash
python -m venv .venv-compile
.venv-compile/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv-compile/bin/pip install transformers peft accelerate datasets bitsandbytes
```

Install a capability beside a database and serve it:

```bash
python packs/bundle.py verify  dist/motion-craft-0.1.0.ycap
python packs/bundle.py install dist/motion-craft-0.1.0.ycap --db mem.db
python packs/bundle.py list    --db mem.db

.venv-compile/bin/python packs/serve_compiled.py \
    --base Qwen/Qwen3.5-4B --db mem.db --port 11555 &
```

Then mount, use, unmount:

```bash
python packs/cap.py status
python packs/cap.py mount motion-craft
python packs/cap.py ask "Write CSS for a modal that fades and scales in. CSS only."
python packs/cap.py unmount
```

Ask the same question either side of `mount` and diff the answers. On our
run the unmounted model wrote `transition: all 0.3s ease` and the mounted
one wrote `transition: opacity 300ms ease-out, transform 300ms ease-out`
— then unmounting returned the first answer exactly, because the adapter
sits beside the base weights and never modifies them.

---

## 3. Compile your own (2–3 GPU-hours)

The full authoring guide is [CAPABILITIES.md](CAPABILITIES.md). The shape:

```bash
# 1. constitution.md — rules that must hold on every token
# 2. craft.py        — a deterministic checker + brief generators
# 3. synthesise, verified against the checker
python packs/compile.py --pack mypack --synthesize-craft \
    --teacher-api qwen --env-file keys.env --workers 24

# 4. train
.venv-compile/bin/python packs/compile.py --pack mypack-craft --train \
    --base Qwen/Qwen3.5-4B --rank 16 --epochs 20 --quant4 --tag v1

# 5. measure against sealed briefs and a pinned frontier examinee
python packs/evaluate_craft.py --pack mypack --adapter mypack-craft --corpus

# 6. sign and publish
python packs/bundle.py build mypack-craft --tag v1 --key <secret>
```

Cost on our runs: about **$0.15** of teacher calls and **2–3 GPU-hours**
per capability.

---

## What to be sceptical about, from us

- **n = 12** sealed briefs per domain. Small.
- **One base model.** Every number is Qwen3.5-4B. The claim is "on this
  base", not "in general".
- **The grader measures mechanical discipline, not beauty.** The frontier
  model attempts more elaborate artifacts and loses more of them to parse
  failures. That is a real effect and it flatters us.
- **The student is instrument-aligned by construction** — trained on
  artifacts that pass the checker, then graded by the checker. The
  defences are that the briefs are sealed and that the checks encode
  craft users actually feel (reduced motion, compositor-only animation,
  a readable measure), but you should weigh it.
- Compiled **facts** cost 7 of 31 control questions to domain intrusion.
  Compiled **rules** cost nothing (30/31, unchanged). That asymmetry is
  the whole reason for the two-tier design, and it is in
  [`control-when-mounted.json`](control-when-mounted.json).
